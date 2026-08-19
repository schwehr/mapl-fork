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

"""Data types for plume inference."""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any

import numpy as np
import shapely.geometry
import utm


@dataclasses.dataclass(frozen=True)
class Granule:
  """A split of a granule."""

  ee_asset_id: str
  data: dict[str, Any]
  mask: np.ndarray
  geotransform: tuple[float, float, float, float, float, float]
  utm_zone: str
  epsg: str
  timestamp_ms: int


@dataclasses.dataclass(frozen=True)
class GranuleChunk:
  """A chunk of predictions with metadata."""

  data: dict[str, np.ndarray]
  mask: np.ndarray
  h_off: int
  w_off: int
  pad_y: int
  pad_x: int
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ChunkedGranule:
  """A granule with separate chunks of predictions."""

  ee_asset_id: str
  chunks: Sequence[GranuleChunk]
  mask: np.ndarray
  geotransform: tuple[float, float, float, float, float, float]
  utm_zone: str
  epsg: str
  timestamp_ms: int


@dataclasses.dataclass(kw_only=True)
class PlumeCandidate:
  """A single plume instance extracted from a chunk before deduplication."""

  chunk_id: int
  slot_id: int
  geometry: shapely.geometry.Polygon | shapely.geometry.MultiPolygon
  geometry_px: shapely.geometry.Polygon | shapely.geometry.MultiPolygon
  metadata: dict[str, Any]
  concentration: np.ndarray
  binary_masks: np.ndarray
  origin_masks: np.ndarray
  mask: np.ndarray
  head_point: shapely.geometry.Point | None
  head_point_px: shapely.geometry.Point | None
  plume_bbox_px: tuple[int, int, int, int]


@dataclasses.dataclass(kw_only=True)
class GranuleCandidates:
  """A set of candidates extracted at a specific threshold sweep."""

  chunked_granule: ChunkedGranule
  plume_prob: float
  origin_prob: float
  candidates: Sequence[PlumeCandidate]
  model_name: str


@dataclasses.dataclass(kw_only=True)
class PlumeGroupData:
  """Aggregated data for a group of plumes."""

  group_id: int
  candidates: Sequence[PlumeCandidate]
  # Bounding box in pixel coordinates (x_min, y_min, x_max, y_max)
  bbox_px: tuple[int, int, int, int]
  # Aggregated arrays, normalized
  concentration: np.ndarray
  binary_masks: np.ndarray
  origin_masks: np.ndarray
  weights: np.ndarray
  counts: np.ndarray
  mask: np.ndarray


@dataclasses.dataclass
class UtmGridMapping:
  """Universal Transverse Mercator (UTM) Grid Mapping."""

  utm_zone: str
  cell_size: float
  width: int | np.ndarray
  height: int | np.ndarray
  utm_x_min: float | np.ndarray = 0.0
  utm_y_min: float | np.ndarray = 0.0
  use_floor: bool = False

  def __post_init__(self):
    fn = np.floor if self.use_floor else np.round
    self.utm_x_min = fn(self.utm_x_min / self.cell_size) * self.cell_size
    self.utm_y_min = fn(self.utm_y_min / self.cell_size) * self.cell_size

  @property
  def epsg(self) -> str:
    northern_hemisphere = self.utm_zone[-1].upper() >= "N"
    longitude_band = int(self.utm_zone[:-1])
    return f"EPSG:32{6 if northern_hemisphere else 7}{longitude_band:02}"

  @property
  def crs(self) -> tuple[float, float, float, float, float, float]:
    return (  # pyrefly: ignore[bad-return]
        self.cell_size,
        0.0,
        self.utm_x_min,
        0.0,
        -self.cell_size,
        self.utm_y_min + self.cell_size * self.height,
    )

  @property
  def centroid(self) -> tuple[float, float]:
    return (  # pyrefly: ignore[bad-return]
        self.utm_x_min + (self.width * self.cell_size) / 2.0,
        self.utm_y_min + (self.height * self.cell_size) / 2.0,
    )

  @classmethod
  def from_latlon_center(
      cls,
      lat: float,
      lon: float,
      cell_size: float,
      width: int,
      height: int | None = None,
      use_floor: bool = False,
  ):
    """Creates UtmGridMapping from lat/lon center."""
    height = width if height is None else height
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    utm_zone = f"{zone_number}{zone_letter}"
    x0 = easting - cell_size * width / 2.0
    y0 = northing - cell_size * height / 2.0
    return cls(utm_zone, cell_size, width, height, x0, y0, use_floor)

  @functools.cached_property
  def corners_latlon(self) -> np.ndarray:
    """Computes the lat/lon for the four corners of the UTM grid."""
    zone_number = int(self.utm_zone[:-1])
    zone_letter = self.utm_zone[-1]
    x_max = self.utm_x_min + self.width * self.cell_size
    y_max = self.utm_y_min + self.height * self.cell_size
    bottom_left = utm.to_latlon(
        self.utm_x_min, self.utm_y_min, zone_number, zone_letter, strict=False
    )
    bottom_right = utm.to_latlon(
        x_max, self.utm_y_min, zone_number, zone_letter, strict=False
    )
    top_right = utm.to_latlon(
        x_max, y_max, zone_number, zone_letter, strict=False
    )
    top_left = utm.to_latlon(
        self.utm_x_min, y_max, zone_number, zone_letter, strict=False
    )

    return np.array([bottom_left, bottom_right, top_right, top_left])


@dataclasses.dataclass(frozen=True)
class SpectralVettingInputs:
  """Inputs for spectral vetting."""

  radiance: np.ndarray
  binary_mask: np.ndarray
  background_mask: np.ndarray
  concentration: np.ndarray
  to_sun_zenith: np.ndarray
  to_sensor_zenith: np.ndarray
  plume_pixels_count: int
  background_pixels_count: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class Plume:
  """A deduplicated, merged plume."""

  group_id: int
  cluster_size: int
  color: str
  geometry: shapely.geometry.Polygon | shapely.geometry.MultiPolygon
  geometry_px: shapely.geometry.Polygon | shapely.geometry.MultiPolygon
  plume_utm_mapping: UtmGridMapping
  metadata: dict[str, Any]
  raster: np.ndarray
  mask: np.ndarray
  binary_mask: np.ndarray
  head_point: shapely.geometry.Point
  head_point_px: shapely.geometry.Point
  timestamp: int
  plume_bbox_px: tuple[int, int, int, int]

  spectral_vetting_inputs: SpectralVettingInputs | None = None
  exclusion_mask: np.ndarray | None = None
  validity_mask: np.ndarray | None = None


@dataclasses.dataclass(kw_only=True)
class SweepPlumes:
  """A set of deduped plumes extracted at a specific threshold sweep."""

  ee_asset_id: str
  cluster_alg: str
  clustering_params: str
  plume_prob: float
  origin_prob: float
  plumes: Sequence[Plume]
  max_candidates_per_cluster: int
  model_name: str
  window_type: str
  retain_origin_component: bool
