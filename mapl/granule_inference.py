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

"""MAPL-EMIT granule inference."""

import collections
from collections.abc import Iterator, Sequence
import dataclasses
import math
from typing import Any, Callable

from absl import logging
from mapl import deduplication
from mapl import plume_candidate_extraction as extraction
from mapl import spectral_matching
from mapl import wavelengths as wavelengths_lib
import mapl.config
import mapl.data_types
import numpy as np
import tensorflow as tf
import toolz


def _get_border_averaging_weights(mode: str, size: int, dtype):
  """Returns weights for border averaging."""
  if mode == 'l2_dist_avg':
    w = np.array([
        [[math.hypot(i - size / 2, j - size / 2)] for j in range(size)]
        for i in range(size)
    ])
    # We normalize by the minimal distance from the center to the border.
    w_max = math.hypot(0, size / 2)
    w = np.maximum(1 - w / w_max, 0.001)
    return w
  if mode == 'l1_border_dist_avg':
    assert size % 2 == 0, 'size expected to be even'
    border = size // 4
    assert border <= size // 2, 'not tested for border > size//2'
    m = np.tile(
        np.array(list(range(size // 2 - 1, -1, -1)) + list(range(size // 2)))[
            None
        ],
        (size, 1),
    )
    m = np.maximum(m, m.T)  # L1 (Mahalanobis) distance from center.
    w = m.max() - m + 1  # Weights starting at 1.
    if border:
      w = np.minimum(w, border + 1)
    w = w / w.max()  # Normalize to max weight 1.
    w = np.expand_dims(w, -1)
    return w
  return np.ones((size, size, 1), dtype=dtype)


def _epsg_to_utm(epsg: str) -> str:
  """Converts an EPSG string (e.g., 'EPSG:32631') to a UTM zone string."""
  code = int(epsg.split(':')[1])
  if 32601 <= code <= 32660:
    return f'{code - 32600}N'
  elif 32701 <= code <= 32760:
    return f'{code - 32700}S'
  else:
    raise ValueError(f'Unsupported EPSG code for UTM conversion: {epsg}')


def _crop_and_pad(
    arr: np.ndarray, y_s: int, y_e: int, x_s: int, x_e: int
) -> np.ndarray:
  """Crops a 2D or 3D array with zero-padding if the crop goes out of bounds."""
  h, w = arr.shape[0], arr.shape[1]
  pad_y_before = max(0, -y_s)
  pad_y_after = max(0, y_e - h)
  pad_x_before = max(0, -x_s)
  pad_x_after = max(0, x_e - w)

  crop_y_s = max(0, y_s)
  crop_y_e = min(h, y_e)
  crop_x_s = max(0, x_s)
  crop_x_e = min(w, x_e)

  cropped = arr[crop_y_s:crop_y_e, crop_x_s:crop_x_e]

  pad_width = [
      (pad_y_before, pad_y_after),
      (pad_x_before, pad_x_after),
  ]
  if arr.ndim > 2:
    pad_width.append((0, 0))

  return np.pad(cropped, pad_width, mode='constant', constant_values=0)


def run_tiled_inference(
    granule: mapl.data_types.Granule,
    infer_fn: Callable[..., Any],
    input_size: int,
    stride: int,
    batch_size: int,
    batch_preprocessing_fn: (
        Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None
    ),
    outputs_names: Sequence[str] | None,
) -> Iterator[mapl.data_types.GranuleChunk]:
  """Runs tiled inference over a granule using batched processing."""
  h, w = granule.mask.shape[:2]

  w_indices = np.arange(0, w - input_size + 1, stride)
  h_indices = np.arange(0, h - input_size + 1, stride)
  ww, hh = np.meshgrid(w_indices, h_indices)

  batch = []

  def _run_inference_on_batch(
      chunks: list[mapl.data_types.GranuleChunk],
  ) -> Iterator[mapl.data_types.GranuleChunk]:
    if not chunks:
      return

    batch_data = collections.defaultdict(list)
    for cell in chunks:
      for k in list(cell.data.keys()):
        batch_data[k].append(cell.data.pop(k))

    batch_data = toolz.valmap(np.stack, dict(batch_data))

    if batch_preprocessing_fn:
      batch_data = batch_preprocessing_fn(batch_data)

    outputs = infer_fn(batch_data)

    if outputs_names:
      outputs = toolz.keyfilter(outputs_names.count, outputs)

    outputs_np = toolz.valmap(
        lambda x: x.numpy() if hasattr(x, 'numpy') else np.asarray(x), outputs
    )

    for i, cell in enumerate(chunks):
      cell_data = toolz.valmap(lambda x, i=i: np.atleast_3d(x[i]), outputs_np)
      yield mapl.data_types.GranuleChunk(
          data=cell_data,
          mask=cell.mask,
          h_off=cell.h_off,
          w_off=cell.w_off,
          pad_y=cell.pad_y,
          pad_x=cell.pad_x,
      )

  flat_indices = list(zip(ww.flat, hh.flat))

  for w_off, h_off in flat_indices:
    row_slice = slice(h_off, h_off + input_size)
    col_slice = slice(w_off, w_off + input_size)

    patch_mask = granule.mask[row_slice, col_slice]
    pad_y = max(0, input_size - patch_mask.shape[0])
    pad_x = max(0, input_size - patch_mask.shape[1])
    pad = ((0, pad_y), (0, pad_x)) + ((0, 0),) * (patch_mask.ndim - 2)

    patch_mask = np.pad(patch_mask, pad)
    if not patch_mask.any():
      continue

    patch_data = {}
    for k, v in granule.data.items():
      curr_pad = ((0, pad_y), (0, pad_x)) + ((0, 0),) * (v.ndim - 2)
      patch_data[k] = np.pad(v[row_slice, col_slice], curr_pad)

    batch.append(
        mapl.data_types.GranuleChunk(
            h_off=int(h_off),
            w_off=int(w_off),
            pad_y=int(pad_y),
            pad_x=int(pad_x),
            data=patch_data,
            mask=patch_mask,
        )
    )

    if len(batch) >= batch_size:
      yield from _run_inference_on_batch(batch)
      batch.clear()

  if batch:
    yield from _run_inference_on_batch(batch)
  batch.clear()


class MaplEmitInference:
  """MAPL-EMIT inference."""

  def __init__(
      self,
      cfg: mapl.config.ExportConfig,
      ee_asset_id: str = 'dummy',
      timestamp_ms: int = 0,
  ):

    # Initialize SpectralVetting
    wavelengths = np.array(
        [v[0] for v in wavelengths_lib.EMIT_L1B_WAVELENGTHS_BANDWIDTHS.values()]
    )
    band_names = list(wavelengths_lib.EMIT_L1B_WAVELENGTHS_BANDWIDTHS)
    self.spectral_vetter = spectral_matching.SpectralVetting(
        band_names, wavelengths, npz_filename=cfg.npz_filename
    )

    self.cfg = cfg
    self.ee_asset_id = ee_asset_id
    self.timestamp_ms = timestamp_ms
    self.model: Any = None

  def setup(self):
    logging.info('Loading MAPL-EMIT model from %s', self.cfg.model_path)
    try:
      self.model = tf.saved_model.load(self.cfg.model_path)
    except Exception as e:
      raise ValueError(
          f'Could not load model from {self.cfg.model_path}: {e}'
      ) from e

  def run_inference(
      self,
      l1b_radiance: np.ndarray,
      sun_zenith: np.ndarray,
      sensor_zenith: np.ndarray,
      crosstrack_ids: np.ndarray,
      epsg: str,
      geotransform: tuple[float, float, float, float, float, float],
      mask: np.ndarray,
      window_fn: Callable[[int], np.ndarray] = np.hanning,
      max_candidates_per_cluster_override: int | None = None,
      retain_origin_component_override: bool | None = None,
  ) -> tuple[
      np.ndarray,
      list[np.ndarray],
      list[dict[str, Any]],
      Sequence[mapl.data_types.PlumeCandidate],
      Sequence[mapl.data_types.Plume] | None,
  ]:
    """Runs the full MAPL-EMIT plume detection pipeline on a single granule.

    Args:
        l1b_radiance: EMIT L1B radiance data as a numpy array.
        sun_zenith: Sun zenith angles from EMIT L1B OBS file.
        sensor_zenith: Sensor zenith angles from EMIT L1B OBS file.
        crosstrack_ids: Crosstrack IDs from EMIT L1B RAD file.
        epsg: EPSG projection string (e.g., 'EPSG:32631').
        geotransform: Geotransform tuple of 6 floats.
        mask: Validity mask as a numpy array (H, W, 1).
        window_fn: Window function to use for weighted averaging.
        max_candidates_per_cluster_override: Override for max candidates per
          cluster.
        retain_origin_component_override: Override for protecting origin
          component.

    Returns:
        A tuple containing:
        - Enhancement image as a numpy array. Same height/width as l1b_radiance.
        - A list of plumes as numpy arrays. Same height/width as l1b_radiance.
        - A list of metadata dictionaries for each plume.
        - A list of candidate plumes.
        - A list of deduplicated plumes.
    """
    if self.model is None:
      self.setup()

    h, w, _ = l1b_radiance.shape
    input_size = self.cfg.model_input_size
    stride = self.cfg.stride
    utm_zone = _epsg_to_utm(epsg)

    granule = mapl.data_types.Granule(
        ee_asset_id=self.ee_asset_id,
        data={
            'emit_l1b_radiance': l1b_radiance,
            'emit_l1b_to_sun_zenith': sun_zenith,
            'emit_l1b_to_sensor_zenith': sensor_zenith,
            'emit_l1b_crosstrack_id': crosstrack_ids,
        },
        mask=mask,
        geotransform=geotransform,
        utm_zone=utm_zone,
        epsg=epsg,
        timestamp_ms=self.timestamp_ms,
    )

    def wrapped_infer_fn(batch):
      batch = {
          'emit_l1b_radiance': tf.convert_to_tensor(
              batch['emit_l1b_radiance'], dtype=tf.float32
          ),
          'emit_l1b_to_sun_zenith': tf.convert_to_tensor(
              batch['emit_l1b_to_sun_zenith'], dtype=tf.float32
          ),
          'emit_l1b_to_sensor_zenith': tf.convert_to_tensor(
              batch['emit_l1b_to_sensor_zenith'], dtype=tf.float32
          ),
          'emit_l1b_crosstrack_id': tf.convert_to_tensor(
              batch['emit_l1b_crosstrack_id'], dtype=tf.float32
          ),
      }
      return self.model.infer_fn(batch)

    chunks = list(
        run_tiled_inference(
            granule=granule,
            infer_fn=wrapped_infer_fn,
            input_size=input_size,
            stride=stride,
            batch_size=1,
            batch_preprocessing_fn=None,
            outputs_names=None,
        )
    )

    # 2. ChunkedGranule Assembly
    chunked_granule = mapl.data_types.ChunkedGranule(
        ee_asset_id=self.ee_asset_id,
        chunks=chunks,
        mask=mask,
        geotransform=geotransform,
        utm_zone=utm_zone,
        epsg=epsg,
        timestamp_ms=self.timestamp_ms,
    )

    # 3. Candidate Extraction
    candidates = extraction.extract_candidates(
        chunked_granule,
        plume_probability_threshold=self.cfg.plume_probability_threshold,
        origin_probability_threshold=self.cfg.origin_probability_threshold,
        log=False,
        cfg=self.cfg,
        metrics=None,
    )

    # 4. Deduplicate plumes
    w_t = window_fn(input_size)
    tile_weight = np.outer(w_t, w_t)

    deduped_plumes = (
        deduplication.dedupe_plumes_and_calculate_spectral_vetting_inputs(
            chunked_granule=chunked_granule,
            plumes=candidates,
            tile_weight=tile_weight,
            cluster_alg=self.cfg.cluster_alg,
            cluster_kw=self.cfg.cluster_kw,
            plume_probability_threshold=self.cfg.plume_probability_threshold,
            origin_probability_threshold=self.cfg.origin_probability_threshold,
            regularizer=self.cfg.regularizer,
            scale=self.cfg.scale,
            simplify=self.cfg.simplify,
            keep_holes=self.cfg.keep_holes,
            cc_min_component_size=self.cfg.cc_min_component_size,
            border_on_plume_images=self.cfg.border_on_plume_images,
            vetting_patch_size=self.cfg.vetting_patch_size,
            max_candidates_per_cluster=(
                max_candidates_per_cluster_override
                if max_candidates_per_cluster_override is not None
                else int((self.cfg.model_input_size / self.cfg.stride) ** 2)
            ),
            metrics=None,
            retain_origin_component=(
                retain_origin_component_override
                if retain_origin_component_override is not None
                else True
            ),
            log=False,
        )
    )
    # 5. Reconstruct full-size enhancement image (Weighted Averaging)
    acc_data = np.zeros((h, w, 1), dtype=np.float32)
    acc_weights = np.zeros((h, w, 1), dtype=np.float32)
    w_mask = _get_border_averaging_weights(
        'l2_dist_avg', input_size, 'float32'
    )

    for chunk in chunks:
      valid_h = input_size - chunk.pad_y
      valid_w = input_size - chunk.pad_x
      local_slice = (slice(0, valid_h), slice(0, valid_w))

      chunk_conc = np.max(chunk.data['concentration'], axis=-1, keepdims=True)

      chunk_preds = chunk_conc[local_slice]
      chunk_w_mask = w_mask[local_slice] * chunk.mask[local_slice]

      preds_y_slice = slice(chunk.h_off, chunk.h_off + valid_h)
      preds_x_slice = slice(chunk.w_off, chunk.w_off + valid_w)

      acc_data[preds_y_slice, preds_x_slice] += chunk_preds * chunk_w_mask
      acc_weights[preds_y_slice, preds_x_slice] += chunk_w_mask

    enhancement_image = acc_data / np.maximum(acc_weights, 1e-10)
    enhancement_image = enhancement_image.squeeze(-1)

    # 6. Spectral Vetting
    plumes_list = []
    metadata_list = []
    patch_size = self.cfg.vetting_patch_size

    if deduped_plumes:
      for plume in deduped_plumes:
        # Get head location in pixel coordinates
        head_col_px, head_row_px = list(plume.head_point_px.coords)[0]
        y_c, x_c = int(head_row_px), int(head_col_px)
        h_vet, w_vet = patch_size // 2, patch_size // 2
        y_start, y_end = y_c - h_vet, y_c + h_vet
        x_start, x_end = x_c - w_vet, x_c + w_vet

        # Crop inputs to vetting patch size
        v_radiance = _crop_and_pad(l1b_radiance, y_start, y_end, x_start, x_end)
        v_sun_zenith = _crop_and_pad(sun_zenith, y_start, y_end, x_start, x_end)
        v_sensor_zenith = _crop_and_pad(
            sensor_zenith, y_start, y_end, x_start, x_end
        )
        v_concentration = _crop_and_pad(
            enhancement_image, y_start, y_end, x_start, x_end
        )

        # Place plume's binary mask into patch
        v_binary_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
        x_off_plume, y_off_plume = plume.plume_bbox_px[:2]
        ph, pw = plume.binary_mask.shape

        slices = deduplication.get_intersection_slices(
            (x_off_plume, y_off_plume, x_off_plume + pw, y_off_plume + ph),
            (x_start, y_start, x_start + patch_size, y_start + patch_size),
        )
        if slices:
          (sy, sx), (dy, dx) = slices
          v_binary_mask[dy, dx] = plume.binary_mask[sy, sx]

        # Compute background mask
        assert plume.exclusion_mask is not None
        assert plume.validity_mask is not None
        background_mask = ~plume.exclusion_mask & plume.validity_mask

        # Run spectral vetting
        vet_result = self.spectral_vetter.match_spectral_signature(
            v_radiance,
            v_binary_mask,
            background_mask,
            v_concentration,
            v_sun_zenith,
            v_sensor_zenith,
        )

        # Merge results into metadata
        if vet_result is not None:
          vet_dict = dataclasses.asdict(vet_result)
          vet_dict = {
              k: v if np.isfinite(v).all() else None
              for k, v in vet_dict.items()
          }
          metadata = plume.metadata | vet_dict
        else:
          metadata = plume.metadata

        # Format outputs
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[
            y_off_plume : y_off_plume + ph, x_off_plume : x_off_plume + pw
        ] = plume.binary_mask
        plumes_list.append(full_mask)
        metadata_list.append(metadata)

    return (
        enhancement_image,
        plumes_list,
        metadata_list,
        candidates,
        deduped_plumes,
    )
