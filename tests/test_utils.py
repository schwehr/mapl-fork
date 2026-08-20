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

"""Test utilities for MAPL."""

from typing import Any, Optional
from mapl import config
from mapl import data_types
import numpy as np
import shapely.geometry


def _get_test_config(**kwargs: Any) -> config.ExportConfig:
  """Returns an ExportConfig for testing with some defaults."""
  params = dict(
      sensor_name='emit_l1b',
      bands=['b1'],
      ee_project='proj',
      ee_asset_chunk_size_m=100,
      ee_multithreading=1,
      scale=1.0,
      model_input_size=4,
      model_path='path',
      model_outputs_keys=['concentration', 'binary_masks', 'origin_masks'],
      batch_size=1,
      stride=2,
      plume_probability_threshold=0.5,
      origin_probability_threshold=0.4,
      gcs_folder='gcs',
      output_ee_asset_id='id',
      cluster_alg='DBSCAN',
      cluster_kw={'eps': 1.5, 'min_samples': 1},
      cc_min_component_size=1,
      regularizer=0.01,
      simplify=0.0,
      keep_holes=False,
      border_on_plume_images=0,
      min_num_pixels_per_slot=1,
      num_eroded_pixels=0,
      npz_filename='satellite_gas_statistics_v14_emit.npz',
  )
  return config.ExportConfig(**(params | kwargs))  # pyrefly: ignore[bad-argument-type]


def _create_chunk_data(
    chunk_size: int,
    radiance_val: float,
    concentration_val: float,
    origin_pixel: Optional[tuple[int, int]] = (1, 1),
) -> dict[str, np.ndarray]:
  """Creates a dictionary of chunk data for testing."""
  data = {
      'radiance': np.full(
          (chunk_size, chunk_size, 1), radiance_val, dtype=np.float32
      ),
      'to_sun_zenith': np.ones((chunk_size, chunk_size, 1), dtype=np.float32),
      'to_sensor_zenith': np.ones(
          (chunk_size, chunk_size, 1), dtype=np.float32
      ),
      'concentration': np.full(
          (chunk_size, chunk_size, 1), concentration_val, dtype=np.float32
      ),
      'binary_masks': np.full(
          (chunk_size, chunk_size, 1), 1.0, dtype=np.float32
      ),
      'origin_masks': np.zeros((chunk_size, chunk_size, 1), dtype=np.float32),
  }
  if origin_pixel:
    data['origin_masks'][origin_pixel[0], origin_pixel[1], 0] = 1.0
  return data


def _create_chunk(
    chunk_data: dict[str, np.ndarray],
    h_off: int,
    w_off: int,
    mask: Optional[np.ndarray] = None,
) -> data_types.GranuleChunk:
  """Creates a GranuleChunk for testing."""
  if 'radiance' in chunk_data:
    chunk_size = chunk_data['radiance'].shape[0]
  else:
    chunk_size = next(iter(chunk_data.values())).shape[0]

  if mask is None:
    mask = np.ones((chunk_size, chunk_size, 1), dtype='uint8')
  return data_types.GranuleChunk(
      data=chunk_data,
      mask=mask,
      h_off=h_off,
      w_off=w_off,
      pad_y=0,
      pad_x=0,
  )


def _create_plume_candidate(
    chunk_id: int,
    chunk: data_types.GranuleChunk,
    head_point_point: Optional[shapely.geometry.Point],
    head_point_px: Optional[tuple[float, float]],
    plume_bbox_px: tuple[int, int, int, int],
) -> data_types.PlumeCandidate:
  """Creates a PlumeCandidate for testing."""
  return data_types.PlumeCandidate(
      chunk_id=chunk_id,
      geometry=shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
      geometry_px=shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
      slot_id=0,
      metadata={},
      mask=chunk.mask[..., 0],
      head_point=head_point_point,
      head_point_px=(
          shapely.geometry.Point(head_point_px)
          if head_point_px is not None
          else None
      ),
      plume_bbox_px=plume_bbox_px,
      concentration=chunk.data.get('concentration', np.zeros_like(chunk.mask))[
          ..., 0
      ],
      binary_masks=chunk.data.get('binary_masks', np.zeros_like(chunk.mask))[
          ..., 0
      ],
      origin_masks=chunk.data.get('origin_masks', np.zeros_like(chunk.mask))[
          ..., 0
      ],
  )
