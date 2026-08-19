# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single granule inference script."""

import os
import tempfile
from typing import Callable

from absl import app
from absl import flags
from absl import logging
from mapl import config
from mapl import granule_inference
from mapl import io_lib
from mapl import netcdf_io
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.signal

from tensorflow.io import gfile


_MODEL_PATH = flags.DEFINE_string('model_path', None, 'Path to the model.')
_INPUT_FILEPATH = flags.DEFINE_string(
    'input_filepath', None, 'Path to the input file.'
)
_OUTPUT_FILEPATH = flags.DEFINE_string(
    'output_filepath',
    None,
    'Base path/prefix to output data (parquet and npy files).',
)
_CROP_SIZE = flags.DEFINE_integer(
    'crop_size',
    None,
    'Optional size of the central crop (e.g., 512) to test a subset.',
)
_STORE_CANDIDATE_RASTER_DATA = flags.DEFINE_bool(
    'store_candidate_raster_data',
    False,
    'Whether to store the raster data in the candidate plumes parquet file.',
)
_WINDOW_TYPE = flags.DEFINE_enum(
    'window_type',
    'hanning',
    ['hanning', 'tukey'],
    'Type of window to use for merging overlapping patches.',
)
_MAX_CANDIDATES_PER_CLUSTER = flags.DEFINE_integer(
    'max_candidates_per_cluster',
    None,
    'Optional override for the maximum number of candidates per cluster. If not'
    ' provided, it defaults to (model_input_size / stride) ** 2.',
)
_RETAIN_ORIGIN_COMPONENT = flags.DEFINE_bool(
    'retain_origin_component',
    None,
    'Optional override for whether to protect the origin connected component '
    'from being filtered out. If not provided, defaults to True.',
)
_STRIDE = flags.DEFINE_integer(
    'stride', 64, 'Grid spacing in pixels for patch extraction.'
)
_SCALE = flags.DEFINE_float(
    'scale', 60.0, 'Pixel resolution / scale in meters.'
)
_MODEL_INPUT_SIZE = flags.DEFINE_integer(
    'model_input_size', 256, 'Model input image dimension.'
)
_SENSOR_NAME = flags.DEFINE_string(
    'sensor_name', 'emit_l1b', 'Name of the sensor used for inference.'
)
_EPS_DIST = flags.DEFINE_float(
    'eps_dist', 25.0, 'DBSCAN eps distance for clustering plumes.'
)
_EPS_CORR = flags.DEFINE_float(
    'eps_corr', 0.955, 'DBSCAN correlation threshold for clustering plumes.'
)
_MIN_SAMPLES_CLUSTERING = flags.DEFINE_integer(
    'min_samples_clustering', 6, 'DBSCAN min samples for clustering plumes.'
)
_NPZ_FILENAME = flags.DEFINE_string(
    'npz_filename',
    'satellite_gas_statistics_v14_emit.npz',
    'Filename or path of the NPZ file containing spectral statistics.',
)

flags.mark_flag_as_required('model_path')
flags.mark_flag_as_required('input_filepath')
flags.mark_flag_as_required('output_filepath')


def write_to_storage(
    write_fn: Callable[[str], None], dest_path: str, suffix: str
):
  """Writes data to storage using a temporary local file."""
  with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
    tmp_name = tmp_file.name
  try:
    write_fn(tmp_name)
    gfile.copy(tmp_name, dest_path, overwrite=True)
  finally:
    if os.path.exists(tmp_name):
      os.remove(tmp_name)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info(
      'Initializing MaplEmitInference with model: %s', _MODEL_PATH.value
  )
  # Set up inference configuration from flags.
  cfg = config.ExportConfig(
      sensor_name=_SENSOR_NAME.value,
      model_path=_MODEL_PATH.value,  # pyrefly: ignore[bad-argument-type]
      stride=_STRIDE.value,
      scale=_SCALE.value,
      model_input_size=_MODEL_INPUT_SIZE.value,
      cluster_kw=dict(
          eps_dist=_EPS_DIST.value,
          eps_corr=_EPS_CORR.value,
          min_samples=_MIN_SAMPLES_CLUSTERING.value,
          patch_size=65,
      ),
      bands=[],
      ee_project='',
      ee_asset_chunk_size_m=0,
      ee_multithreading=1,
      model_outputs_keys=[],
      batch_size=1,
      gcs_folder='',
      output_ee_asset_id='',
      npz_filename=_NPZ_FILENAME.value,
  )
  inferencer = granule_inference.MaplEmitInference(
      cfg=cfg,
      ee_asset_id='dummy',
      timestamp_ms=0,
  )

  logging.info('Running setup() on MaplEmitInference.')
  inferencer.setup()

  input_path = _INPUT_FILEPATH.value
  if not input_path:
    raise app.UsageError('input_filepath flag must be populated')

  logging.info('Reading input data from NC file.')
  (
      l1b_radiance,
      sun_zenith,
      sensor_zenith,
      crosstrack_ids,
      epsg,
      geotransform,
      mask,
  ) = netcdf_io.read_emit_netcdf(input_path)

  crop_size = _CROP_SIZE.value
  if crop_size is not None:
    logging.info('Applying central crop of size %d', crop_size)
    h, w = l1b_radiance.shape[:2]

    if crop_size > h or crop_size > w:
      logging.warning(
          'Crop size (%d) is larger than image dims (%d, %d). Skipping crop.',
          crop_size,
          h,
          w,
      )
    else:
      y_start = (h - crop_size) // 2
      x_start = (w - crop_size) // 2
      y_end = y_start + crop_size
      x_end = x_start + crop_size

      l1b_radiance = l1b_radiance[y_start:y_end, x_start:x_end]
      sun_zenith = sun_zenith[y_start:y_end, x_start:x_end]
      sensor_zenith = sensor_zenith[y_start:y_end, x_start:x_end]
      crosstrack_ids = crosstrack_ids[y_start:y_end, x_start:x_end]
      mask = mask[y_start:y_end, x_start:x_end]

      # update geotransform
      # geotransform = (left, xres, rx, top, ry, yres)
      gt = list(geotransform)
      gt[0] += x_start * gt[1]  # shift left
      gt[3] += y_start * gt[5]  # shift top
      geotransform = tuple(gt)

  window_functions = {
      'tukey': lambda size: scipy.signal.windows.tukey(size, alpha=0.2),
      'hanning': np.hanning,
  }
  window_fn = window_functions[_WINDOW_TYPE.value]

  logging.info('Running run_inference().')
  enhancement, plumes, _, candidates, deduped_plumes = inferencer.run_inference(
      l1b_radiance=l1b_radiance,
      sun_zenith=sun_zenith,
      sensor_zenith=sensor_zenith,
      crosstrack_ids=crosstrack_ids,
      epsg=epsg,
      geotransform=geotransform,  # pyrefly: ignore[bad-argument-type]
      mask=mask,
      window_fn=window_fn,
      max_candidates_per_cluster_override=_MAX_CANDIDATES_PER_CLUSTER.value,
      retain_origin_component_override=_RETAIN_ORIGIN_COMPONENT.value,
  )

  logging.info('Inference completed.')
  logging.info('Enhancement shape: %s', enhancement.shape)
  logging.info('Found %d plumes.', len(plumes))

  out_path = _OUTPUT_FILEPATH.value
  logging.info('Writing debug and final output data to %s', out_path)

  # Set up a set of formatters for writing the output data to disk.
  c_schema = io_lib.get_candidate_plumes_base_schema(
      store_rasters_data=_STORE_CANDIDATE_RASTER_DATA.value
  )
  c_records = [
      io_lib.format_candidate_plume_base(
          candidate=candidate,
          ee_asset_id='local_test',
          store_rasters_data=_STORE_CANDIDATE_RASTER_DATA.value,
          plume_prob=inferencer.cfg.plume_probability_threshold,
          origin_prob=inferencer.cfg.origin_probability_threshold,
      )
      for candidate in candidates or []
  ]

  p_schema = io_lib.get_predicted_plumes_base_schema()
  p_records = [
      io_lib.format_predicted_plume_base(
          plume=plume,
          ee_asset_id='local_test',
      )
      for plume in deduped_plumes or []
  ]

  out_dir = os.path.dirname(out_path)  # pyrefly: ignore[no-matching-overload]
  if out_dir:
    gfile.makedirs(out_dir)

  write_to_storage(
      lambda p: pq.write_table(
          pa.Table.from_pylist(c_records, schema=c_schema), p
      ),
      f'{out_path}_candidates.parquet',
      '.parquet',
  )
  write_to_storage(
      lambda p: pq.write_table(
          pa.Table.from_pylist(p_records, schema=p_schema), p
      ),
      f'{out_path}_predicted.parquet',
      '.parquet',
  )
  write_to_storage(
      lambda p: np.save(p, enhancement), f'{out_path}_enhancement.npy', '.npy'
  )

  logging.info('Successfully wrote output to %s.', out_path)


if __name__ == '__main__':
  logging.set_stderrthreshold(logging.INFO)
  app.run(main)
