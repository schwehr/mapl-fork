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

"""Pure functions for mask to geometry conversion and candidate extraction."""

from collections.abc import Sequence
from typing import Any


import mapl.config
import mapl.data_types
import numpy as np
import rasterio.control
import rasterio.features
import rasterio.transform
from scipy import ndimage
import shapely.affinity
import shapely.geometry
import shapely.ops
import tqdm.auto as tqdm


def _get_transform(mask: np.ndarray, lat_lon_corners: np.ndarray) -> Any:
  """Returns a list of GroundControlPoints for a given mask."""
  h, w = mask.shape
  pixel_corners = [(h, 0), (h, w), (0, w), (0, 0)]
  gcp = [
      rasterio.control.GroundControlPoint(
          row=r, col=c, x=corner[1], y=corner[0]
      )
      for (r, c), corner in zip(pixel_corners, lat_lon_corners)
  ]
  return rasterio.transform.from_gcps(gcp)


def get_center_of_mass_location(
    mask: np.ndarray,
    lat_lon_corners: np.ndarray,
) -> shapely.geometry.Point | None:
  """Computes weighted center of mass of mask and returns as Shapely Point."""
  if mask.size == 0 or np.all(mask == 0):
    return None

  # Compute center of mass, which is probability-weighted mean of coordinates.
  row, col = ndimage.center_of_mass(mask)

  # Get the geographic coordinates of the center of the pixel
  lon, lat = _get_transform(mask, lat_lon_corners) * (col + 0.5, row + 0.5)

  return shapely.geometry.Point(lon, lat)


def get_head_point(
    origin_probs: np.ndarray,
    utm_mapping: mapl.data_types.UtmGridMapping,
    origin_probability_threshold: float,
    w_off: int = 0,
    h_off: int = 0,
) -> tuple[shapely.geometry.Point | None, shapely.geometry.Point | None]:
  """Computes plume head location in geographic and pixel coordinates."""
  if not (origin_probs > 0).any():
    return None, None

  # Only consider origin pixels above threshold for connected components.
  binary_origin_mask = origin_probs > origin_probability_threshold

  # Find connected components and select the largest one.
  labeled_mask, num_features = ndimage.label(binary_origin_mask)
  if num_features == 0:
    return None, None
  component_sizes = ndimage.sum_labels(
      np.ones_like(binary_origin_mask, dtype=np.int32),
      labeled_mask,
      index=np.arange(1, num_features + 1),
  )
  largest_component_label = np.argmax(component_sizes) + 1
  largest_component_mask = labeled_mask == largest_component_label

  # Mask probabilities with largest component
  origin_probs_largest_component = origin_probs * largest_component_mask

  # Compute center of mass, which is probability-weighted mean of coordinates.
  row, col = ndimage.center_of_mass(origin_probs_largest_component)

  # Get the geographic coordinates of pixel.
  transform = _get_transform(
      origin_probs_largest_component, utm_mapping.corners_latlon
  )
  lon, lat = transform * (col + 0.5, row + 0.5)
  head_point = shapely.geometry.Point(lon, lat)
  head_point_px = shapely.geometry.Point(w_off + col, h_off + row)

  return head_point, head_point_px


def extract_candidates(
    chunked_granule: mapl.data_types.ChunkedGranule,
    plume_probability_threshold: float,
    origin_probability_threshold: float,
    log: bool,
    cfg: mapl.config.ExportConfig,
    metrics: Any,
) -> Sequence[mapl.data_types.PlumeCandidate]:
  """Extracts individual plumes from the model outputs."""
  plumes = []
  loop = enumerate(chunked_granule.chunks)
  if log:
    loop = tqdm.tqdm(loop, total=len(chunked_granule.chunks))
  for chunk_id, chunk in loop:

    # [H, W, N_SLOTS]
    unmasked_binary_masks = (
        chunk.data['binary_masks'] > plume_probability_threshold
    ).astype(np.uint8)

    # Apply the cell's validity mask to the masks
    binary_masks = unmasked_binary_masks * chunk.mask

    # Discard empty slots.
    min_pixels_per_slot = cfg.min_num_pixels_per_slot
    valid_slot = binary_masks.sum(axis=(0, 1)) >= min_pixels_per_slot
    if metrics:
      metrics.plumes_extracted.inc(valid_slot.sum())

    for slot_id in np.where(valid_slot)[0]:
      slot_masked = binary_masks[..., slot_id].astype(np.uint8)

      global_x_min, _, _, global_y_max, _, _ = chunked_granule.geotransform

      w_off = chunk.w_off
      row_off = chunk.h_off
      chunk_height, chunk_width = chunk.mask.shape[:2]

      patch_x_min = global_x_min + (w_off * cfg.scale)
      patch_y_min = global_y_max - ((row_off + chunk_height) * cfg.scale)

      utm_mapping = mapl.data_types.UtmGridMapping(
          utm_zone=chunked_granule.utm_zone,
          cell_size=cfg.scale,
          width=chunk_width,
          height=chunk_height,
          utm_x_min=patch_x_min,
          utm_y_min=patch_y_min,
      )

      geometry = convert_binary_mask_to_shapely_polygon(
          slot_masked,
          utm_mapping.corners_latlon,
          simplify=cfg.simplify,
          keep_holes=cfg.keep_holes,
      )
      if geometry is None:
        continue

      geometry_px = get_geometry_px(
          slot_masked,
          xoff=w_off,
          yoff=row_off,
          simplify=cfg.simplify,
          keep_holes=cfg.keep_holes,
      )
      if geometry_px is None:
        continue

      if metrics:
        metrics.plumes_after_border_check.inc()

      head_point, head_point_px = get_head_point(
          chunk.data['origin_masks'][..., slot_id] * chunk.mask[..., 0],
          utm_mapping,
          origin_probability_threshold,
          chunk.w_off,
          chunk.h_off,
      )

      item = mapl.data_types.PlumeCandidate(
          chunk_id=chunk_id,
          geometry=geometry,
          geometry_px=geometry_px,
          slot_id=slot_id,
          metadata=chunk.metadata.get(slot_id, {}),
          concentration=chunk.data['concentration'][..., slot_id],
          binary_masks=chunk.data['binary_masks'][..., slot_id]
          * chunk.mask[..., 0],
          origin_masks=chunk.data['origin_masks'][..., slot_id]
          * chunk.mask[..., 0],
          mask=chunk.mask[..., 0],
          head_point=head_point,
          head_point_px=head_point_px,
          plume_bbox_px=(
              chunk.w_off,
              chunk.h_off,
              chunk.w_off + chunk_width,
              chunk.h_off + chunk_height,
          ),
      )
      plumes.append(item)

    if metrics:
      metrics.chunks_processed.inc()

  return plumes


def get_head_squares(
    plumes_with_heads: Sequence[mapl.data_types.PlumeCandidate], patch_size: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
  """Returns the head concentration and mask squares."""
  if patch_size <= 0:
    return None, None
  if patch_size % 2 == 0:
    raise ValueError('patch_size must be odd.')

  head_concentration_squares = []
  head_mask_squares = []

  half_size = patch_size // 2
  for p in plumes_with_heads:
    assert p.head_point_px is not None
    # Calculate local head location.
    local_col = int(np.round(p.head_point_px.x - p.plume_bbox_px[0]))
    local_row = int(np.round(p.head_point_px.y - p.plume_bbox_px[1]))

    # Pad concentration to handle boundaries.
    padded_concentration = np.pad(
        p.concentration, half_size, mode='constant', constant_values=0
    )
    # Pad mask to handle boundaries.
    padded_mask = np.pad(p.mask, half_size, mode='constant', constant_values=0)

    # Adjust indices for padded array.
    center_row = local_row + half_size
    center_col = local_col + half_size

    patch = padded_concentration[
        center_row - half_size : center_row + half_size + 1,
        center_col - half_size : center_col + half_size + 1,
    ]
    assert patch.shape == (patch_size, patch_size)
    head_concentration_squares.append(patch)

    mask_patch = padded_mask[
        center_row - half_size : center_row + half_size + 1,
        center_col - half_size : center_col + half_size + 1,
    ]
    assert mask_patch.shape == (patch_size, patch_size)
    head_mask_squares.append(mask_patch)

  return tuple(map(np.array, [head_concentration_squares, head_mask_squares]))  # pyrefly: ignore[bad-return]


def to_mmss(duration_seconds: float) -> str:
  """Formats a duration in seconds as MM:SS."""
  minutes, seconds = divmod(duration_seconds, 60)
  return f'{int(minutes):02d}:{int(seconds):02d}'


def _get_polygon(shapes_gen, keep_holes: bool):
  """Returns a single polygon or MultiPolygon from a shapes generator."""
  polygons = [shapely.geometry.shape(geom) for geom, val in shapes_gen if val]

  if not polygons:
    return None

  # Merge into a single geometry (Polygon or MultiPolygon).
  result = shapely.ops.unary_union(polygons)

  # Handle Holes (fill if requested)
  if not keep_holes:
    if isinstance(result, shapely.geometry.MultiPolygon):
      return shapely.geometry.MultiPolygon(
          [shapely.geometry.Polygon(p.exterior) for p in result.geoms]
      )
    if isinstance(result, shapely.geometry.Polygon):
      return shapely.geometry.Polygon(result.exterior)
    raise ValueError(f'Unsupported polygon type: {type(result)}')

  return result


def convert_binary_mask_to_shapely_polygon(
    mask: np.ndarray,
    lat_lon_corners: np.ndarray,
    simplify: float = 0.0,
    keep_holes: bool = False,
) -> shapely.geometry.Polygon | shapely.geometry.MultiPolygon | None:
  """Converts a binary mask to a Shapely Polygon using Rasterio."""
  shapes_gen = rasterio.features.shapes(
      mask, mask=mask > 0, transform=_get_transform(mask, lat_lon_corners)
  )
  result = _get_polygon(shapes_gen, keep_holes)

  if result is None:
    return None

  if simplify > 0:
    result = result.simplify(simplify, preserve_topology=True)

  # Apply a zero buffer to fix potential issues with invalid geometries.
  return result.buffer(0)


def get_geometry_px(
    mask: np.ndarray,
    xoff: int,
    yoff: int,
    simplify: float = 0.0,
    keep_holes: bool = False,
) -> shapely.geometry.Polygon | shapely.geometry.MultiPolygon | None:
  """Computes geometry in pixel coordinates and shifts to global coordinates."""
  height, width = mask.shape
  pixel_corners = np.array(
      [
          (height, 0),
          (height, width),
          (0, width),
          (0, 0),
      ],
      dtype=np.float64,
  )
  geometry_px = convert_binary_mask_to_shapely_polygon(
      mask,
      pixel_corners,
      simplify=simplify,
      keep_holes=keep_holes,
  )
  if geometry_px is None:
    return None

  # We shift the geometry to the global pixel coordinates.
  return shapely.affinity.translate(geometry_px, xoff=xoff, yoff=yoff)


def filter_small_components(
    binary_masks: np.ndarray,
    probability_threshold: float,
    min_pixels_per_slot: int,
):
  """Filters out small connected components from binary masks in place."""
  for i in range(binary_masks.shape[-1]):
    mask = binary_masks[..., i] > probability_threshold
    # Find connected components in the binary mask. labeled_mask is an
    # array of the same shape as mask, where each connected component
    # is assigned a unique label (integer). num_features is the total
    # number of connected components found.
    labeled_mask, num_features = ndimage.label(mask)
    # Calculate the size of each component by summing up the pixels for
    # each label.
    component_sizes = ndimage.sum_labels(
        np.ones_like(mask, dtype=np.int32),
        labeled_mask,
        index=np.arange(1, num_features + 1),
    )
    # Identify components that are smaller than the minimum size.
    small_components = np.where(component_sizes < min_pixels_per_slot)[0] + 1
    # Create a mask to remove small components.
    mask_to_remove = np.isin(labeled_mask, small_components)
    # Set the pixels of small components to 0.
    binary_masks[..., i][mask_to_remove] = 0.0
