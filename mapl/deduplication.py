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

"""Pure logic for clustering, aggregation, and deduplication."""

from collections.abc import Sequence
import dataclasses
import random
from typing import Any


from mapl import clustering
from mapl import data_types
from mapl import plume_candidate_extraction
import numpy as np
from scipy import ndimage
import shapely.geometry
import toolz
import tqdm.auto as tqdm


def cluster_plumes(
    plumes: Sequence[data_types.PlumeCandidate],
    cluster_alg: str,
    cluster_kw: dict[str, Any],
) -> Sequence[int | None]:
  """Clusters plumes based on their head locations and head concentration corr.

  Args:
    plumes: Sequence of plume candidates.
    cluster_alg: Clustering algorithm to use.
    cluster_kw: Keyword arguments for the clustering algorithm.

  Returns:
    Sequence of group IDs corresponding to the input plumes. Some plume
    candidates may be missing a plume origin location, in which case they will
    have a None group ID (e.g. if the plume mask is over the threshold but the
    origin mask is not).
  """
  plume_group_ids = [None] * len(plumes)
  plumes_with_heads_indices = [
      i for i, p in enumerate(plumes) if p.head_point_px is not None
  ]
  if not plumes_with_heads_indices:
    return plume_group_ids

  plumes_with_heads = [plumes[i] for i in plumes_with_heads_indices]

  head_coords = np.array(
      [(p.head_point_px.x, p.head_point_px.y) for p in plumes_with_heads]  # pyrefly: ignore[missing-attribute]
  )
  if cluster_alg not in clustering.CLUSTERING_ALGORITHMS:
    raise ValueError(f'Unknown clustering algorithm: {cluster_alg}')
  cluster_cls = clustering.CLUSTERING_ALGORITHMS[cluster_alg]

  if 'patch_size' in cluster_kw:
    head_concentration_squares, head_mask_squares = (
        plume_candidate_extraction.get_head_squares(
            plumes_with_heads, patch_size=cluster_kw['patch_size']
        )
    )
  else:
    head_concentration_squares, head_mask_squares = None, None

  group_ids = cluster_cls(**cluster_kw).cluster(  # pytype: disable=not-instantiable
      head_coords,
      head_concentration_squares=head_concentration_squares,  # pyrefly: ignore[bad-argument-type]
      head_mask_squares=head_mask_squares,  # pyrefly: ignore[bad-argument-type]
  )
  for original_idx, gid in zip(plumes_with_heads_indices, group_ids):
    plume_group_ids[original_idx] = gid

  return plume_group_ids


def get_intersection_slices(
    src_box: tuple[int, int, int, int],
    dst_box: tuple[int, int, int, int],
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
  """Computes intersection slices for two boxes.

  Args:
    src_box: (x_min, y_min, x_max, y_max) of the source.
    dst_box: (x_min, y_min, x_max, y_max) of the destination.

  Returns:
    A tuple containing ((src_y_slice, src_x_slice), (dst_y_slice, dst_x_slice))
    if there is an intersection, otherwise None.
  """
  sx_min, sy_min, sx_max, sy_max = src_box
  dx_min, dy_min, dx_max, dy_max = dst_box

  inter_x_min = max(sx_min, dx_min)
  inter_y_min = max(sy_min, dy_min)
  inter_x_max = min(sx_max, dx_max)
  inter_y_max = min(sy_max, dy_max)

  if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
    src_slices = (
        slice(inter_y_min - sy_min, inter_y_max - sy_min),
        slice(inter_x_min - sx_min, inter_x_max - sx_min),
    )
    dst_slices = (
        slice(inter_y_min - dy_min, inter_y_max - dy_min),
        slice(inter_x_min - dx_min, inter_x_max - dx_min),
    )
    return src_slices, dst_slices
  return None


def _aggregate_plume_group_data(
    plumes: Sequence[data_types.PlumeCandidate],
    group_ids: Sequence[int | None],
    regularizer: float,
    tile_weight: np.ndarray,
    granule_mask: np.ndarray,
    plume_probability_threshold: float,
    max_candidates_per_cluster: int,
    log: bool = False,
) -> dict[int, data_types.PlumeGroupData]:
  """Aggregates plumes by group."""
  aggregated_groups = {}

  plumes_with_ids = zip(plumes, group_ids)
  grouped_plumes = toolz.groupby(lambda x: x[1], plumes_with_ids)

  loop = grouped_plumes.items()
  if log:
    loop = tqdm.tqdm(loop, total=len(grouped_plumes))
  for group_id, group_list_tuples in loop:
    group_list = [p for p, _ in group_list_tuples]
    # group_id == None signifies unclustered (equivalent to -1 in dbscan)
    if group_id is None or group_id == -1:
      continue

    # Cap the number of candidates per cluster if a limit is provided.
    # A single sliding window stride pass can only physically cover a coordinate
    # a fixed mathematical number of times N (e.g., (256/64)^2 = 16 times).
    # If a cluster exceeds this physical maximum limit, it indicates that the
    # model predicted the same plume in multiple slots. It seems empirically
    # that the additional slots can be much lower certainty or of lower extend,
    # so if we include them automatically, then when we ensemble we end up with
    # a combined plume probability mask that is less certain - and we end up
    # with significantly smaller-extent plumes. To work around this, we sort
    # all candidates in the cluster by plume size and keep only the top N.
    if len(group_list) > max_candidates_per_cluster:
      # Sort descending by footprint (number of pixels over threshold). Note in
      # this instance binary_masks is actually a probability mask.
      group_list.sort(
          key=lambda p: np.sum(p.binary_masks > plume_probability_threshold),
          reverse=True,
      )
      group_list = group_list[:max_candidates_per_cluster]

    # For each cluster of plumes, calculate the bounding box that contains all
    # plumes in the group.
    min_x = min(p.plume_bbox_px[0] for p in group_list)
    min_y = min(p.plume_bbox_px[1] for p in group_list)
    max_x = max(p.plume_bbox_px[2] for p in group_list)
    max_y = max(p.plume_bbox_px[3] for p in group_list)

    width = max_x - min_x
    height = max_y - min_y

    # Initialize arrays to accumulate weighted predictions and metadata for
    # aggregation. These arrays cover the entire bounding box of the plume
    # group.
    agg_concentration = np.zeros((height, width), dtype=np.float32)
    agg_binary = np.zeros((height, width), dtype=np.float32)
    agg_origin = np.zeros((height, width), dtype=np.float32)
    agg_mask = np.ones((height, width), dtype=np.uint8)
    # We add a regularizer to avoid that high confidence predictions from an
    # outlier plume slot prediction cause high values in the aggregation.
    agg_weight = np.zeros((height, width), dtype=np.float32) + regularizer
    agg_count = np.zeros((height, width), dtype=np.float32)

    # Build agg_mask with valid pixel information from the global granule mask.
    g_h, g_w = granule_mask.shape
    slices = get_intersection_slices(
        (0, 0, g_w, g_h),
        (min_x, min_y, max_x, max_y),
    )
    if slices:
      (sy, sx), (dy, dx) = slices
      mask_slice = granule_mask[sy, sx]
      agg_mask[dy, dx] = mask_slice

    # Accumulate plume predictions.
    # Iterate through each plume in the current group and add its weighted
    # predictions to the aggregation arrays.
    for plume in group_list:
      x0, y0, x1, y1 = plume.plume_bbox_px
      dx, dy = x0 - min_x, y0 - min_y
      w, h = x1 - x0, y1 - y0

      # Slicing the target
      target_slice = (slice(dy, dy + h), slice(dx, dx + w))

      tw = tile_weight[:h, :w]
      agg_concentration[target_slice] += plume.concentration * tw * plume.mask
      agg_binary[target_slice] += plume.binary_masks * tw * plume.mask
      agg_origin[target_slice] += plume.origin_masks * tw * plume.mask
      agg_weight[target_slice] += tw * plume.mask
      agg_count[target_slice] += plume.mask

    # Normalize aggregated predictions by dividing by accumulated weights.
    agg_concentration /= agg_weight
    agg_binary /= agg_weight
    agg_origin /= agg_weight

    aggregated_groups[group_id] = data_types.PlumeGroupData(
        group_id=group_id,
        candidates=group_list,
        bbox_px=(min_x, min_y, max_x, max_y),
        concentration=agg_concentration,
        binary_masks=agg_binary,
        origin_masks=agg_origin,
        weights=agg_weight,
        counts=agg_count,
        mask=agg_mask,
    )
  return aggregated_groups


def _create_plumes(
    plume_groups: dict[int, data_types.PlumeGroupData],
    geotransform: tuple[float, float, float, float, float, float],
    utm_zone: str,
    timestamp_ms: int,
    scale: float,
    plume_probability_threshold: float,
    origin_probability_threshold: float,
    cc_min_component_size: int,
    border_on_plume_images: int,
    simplify: float,
    keep_holes: bool,
    metrics: Any = None,
    retain_origin_component: bool = True,
    log: bool = False,
) -> Sequence[data_types.Plume]:
  """Aggregates and vectorizes plumes."""
  deduped_plumes = []

  loop = plume_groups.values()
  if log:
    loop = tqdm.tqdm(loop, total=len(plume_groups))
  for group_data in loop:
    group_id = group_data.group_id
    group_list = group_data.candidates
    min_x, min_y, max_x, max_y = group_data.bbox_px
    width = max_x - min_x
    height = max_y - min_y
    agg_concentration = group_data.concentration
    agg_binary = group_data.binary_masks
    agg_origin = group_data.origin_masks
    agg_mask = group_data.mask

    # Create UTM mapping for the aggregated plume bounding box.
    global_x_min, _, _, global_y_max, _, _ = geotransform
    patch_x_min = global_x_min + (min_x * scale)
    patch_y_min = global_y_max - ((min_y + height) * scale)

    plume_utm_mapping = data_types.UtmGridMapping(
        utm_zone=utm_zone,
        cell_size=scale,
        width=width,
        height=height,
        utm_x_min=patch_x_min,
        utm_y_min=patch_y_min,
    )

    # Create a binary mask from aggregated probabilities and filter by
    # accumulated mask.
    binary_mask_uint8 = (agg_binary > plume_probability_threshold).astype(
        np.uint8
    )
    binary_mask_uint8 = binary_mask_uint8 * (agg_mask > 0)

    # Calculate origin preliminarily to protect its component
    _, head_point_px_pre = (
        plume_candidate_extraction.get_head_point(
            (agg_origin * binary_mask_uint8),
            plume_utm_mapping,
            origin_probability_threshold,
        )
    )

    # Apply connected components analysis and filter small components
    labeled_mask, num_features = ndimage.label(binary_mask_uint8)
    component_sizes = ndimage.sum_labels(
        np.ones_like(binary_mask_uint8),
        labeled_mask,
        index=np.arange(1, num_features + 1),
    )
    small_components = set(
        np.where(component_sizes < cc_min_component_size)[0] + 1
    )

    # Check if the head point (origin) is in any of the small components, and if
    # so, remove it from the set of components to discard so it is preserved.
    if retain_origin_component and head_point_px_pre is not None:
      head_col = int(np.round(head_point_px_pre.x))
      head_row = int(np.round(head_point_px_pre.y))
      if (
          0 <= head_row < labeled_mask.shape[0]
          and 0 <= head_col < labeled_mask.shape[1]
      ):
        origin_comp_id = labeled_mask[head_row, head_col]
        if origin_comp_id in small_components:
          small_components.remove(origin_comp_id)

    mask_to_remove = np.isin(labeled_mask, list(small_components))
    binary_mask_uint8[mask_to_remove] = 0

    # If all pixels were filtered out, skip this group.
    if binary_mask_uint8.sum() == 0:
      continue

    # Compute the bounds of the polygon in pixels for raster export.
    border = border_on_plume_images
    rows, cols = np.where(binary_mask_uint8)
    y_min, y_max = rows.min(), rows.max() + 1
    x_min, x_max = cols.min(), cols.max() + 1

    h, w = binary_mask_uint8.shape
    y_start, y_end = max(0, y_min - border), min(h, y_max + border)
    x_start, x_end = max(0, x_min - border), min(w, x_max + border)
    s_ = (slice(y_start, y_end), slice(x_start, x_end))

    # Update UTM mapping to match the sliced region with border.
    slice_width = x_end - x_start
    slice_height = y_end - y_start

    new_utm_x_min = plume_utm_mapping.utm_x_min + x_start * scale
    # UTM y increases upwards, pixel y increases downwards
    new_utm_y_min = (
        plume_utm_mapping.utm_y_min + (plume_utm_mapping.height - y_end) * scale
    )

    sliced_utm_mapping = data_types.UtmGridMapping(
        utm_zone=utm_zone,
        cell_size=scale,
        width=slice_width,
        height=slice_height,
        utm_x_min=new_utm_x_min,
        utm_y_min=new_utm_y_min,
    )

    # Calculate head location using center of mass of aggregated origin mask.
    head_point, head_point_px = plume_candidate_extraction.get_head_point(
        (agg_origin * binary_mask_uint8)[s_],
        sliced_utm_mapping,
        origin_probability_threshold,
    )
    if head_point is None or head_point_px is None:
      continue

    head_col_px, head_row_px = head_point_px.x, head_point_px.y

    # Slice all aggregated arrays and mosaic data to the region with border.
    agg_concentration = agg_concentration[s_]
    agg_binary = agg_binary[s_]
    agg_origin = agg_origin[s_]
    agg_mask = agg_mask[s_]
    binary_mask_uint8 = binary_mask_uint8[s_]

    # Convert the final binary mask to a Shapely geometry.
    geometry = (
        plume_candidate_extraction.convert_binary_mask_to_shapely_polygon(
            binary_mask_uint8,
            sliced_utm_mapping.corners_latlon,
            simplify=simplify,
            keep_holes=keep_holes,
        )
    )
    if geometry is None:
      continue

    gx_start = min_x + x_start
    gy_start = min_y + y_start

    geometry_px = plume_candidate_extraction.get_geometry_px(
        binary_mask_uint8,
        xoff=gx_start,
        yoff=gy_start,
        simplify=simplify,
        keep_holes=keep_holes,
    )
    if geometry_px is None:
      continue

    # Stack aggregated channels to form the output raster.
    raster = np.stack([agg_concentration, agg_binary, agg_origin], axis=-1)

    # Add random color for each plot.
    r_val = lambda: random.randint(0, 255)
    color_hex = '%02X%02X%02X' % (r_val(), r_val(), r_val())

    # Use the first item for metadata
    first = group_list[0]
    plume_record = data_types.Plume(
        group_id=group_id,
        cluster_size=len(group_list),
        color=color_hex,
        geometry=geometry,
        geometry_px=geometry_px,
        plume_utm_mapping=sliced_utm_mapping,
        metadata=first.metadata,
        raster=raster,
        mask=agg_mask,
        binary_mask=binary_mask_uint8,
        head_point=head_point,
        # The plume head location is given in pixel coordinates relative to
        # the cropped image `s_`, so we need to convert it to global pixel
        # coordinates relative to the entire granule.
        # `_start` are the x and y offsets of this cropped image relative to
        # the bounding box that encompasses a cluster of related plumes.
        # `min_` are the x and y offsets of the cluster's bounding box
        # relative to the full granule image.
        head_point_px=shapely.geometry.Point(
            head_col_px + gx_start,
            head_row_px + gy_start,
        ),
        timestamp=timestamp_ms,
        plume_bbox_px=(
            gx_start,
            gy_start,
            gx_start + slice_width,
            gy_start + slice_height,
        ),
    )
    deduped_plumes.append(plume_record)

  if metrics:
    metrics.plumes_after_dedup.inc(len(deduped_plumes))

  return deduped_plumes


def _calculate_exclusion_mask(
    plume: data_types.Plume,
    candidates: Sequence[data_types.PlumeCandidate],
    plume_probability_threshold: float,
    vetting_patch_size: int,
    granule_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Calculates a local exclusion mask and validity mask for a plume."""
  head_col_px, head_row_px = list(plume.head_point_px.coords)[0]
  y_c, x_c = int(head_row_px), int(head_col_px)
  h_vet, w_vet = vetting_patch_size // 2, vetting_patch_size // 2
  y_start, y_end = y_c - h_vet, y_c + h_vet
  x_start, x_end = x_c - w_vet, x_c + w_vet

  exclusion_mask = np.zeros(
      (vetting_patch_size, vetting_patch_size), dtype=bool
  )
  for cand in candidates:
    x0, y0, x1, y1 = cand.plume_bbox_px
    if x_start < x1 and x_end > x0 and y_start < y1 and y_end > y0:
      ix_start = max(x_start, x0)
      iy_start = max(y_start, y0)
      ix_end = min(x_end, x1)
      iy_end = min(y_end, y1)

      candidate_dy = slice(iy_start - y0, iy_end - y0)
      candidate_dx = slice(ix_start - x0, ix_end - x0)
      local_dy = slice(iy_start - y_start, iy_end - y_start)
      local_dx = slice(ix_start - x_start, ix_end - x_start)

      cand_binary = cand.binary_masks > plume_probability_threshold
      cand_valid = cand.mask > 0
      if cand_binary.ndim == 3:
        cand_binary = cand_binary.squeeze(-1)
      if cand_valid.ndim == 3:
        cand_valid = cand_valid.squeeze(-1)

      cand_combined = cand_binary * cand_valid
      exclusion_mask[local_dy, local_dx] |= cand_combined[
          candidate_dy, candidate_dx
      ].astype(bool)

  validity_mask = np.zeros((vetting_patch_size, vetting_patch_size), dtype=bool)
  granule_h, granule_w = granule_mask.shape

  overlap_ystart = max(0, y_start)
  overlap_yend = min(granule_h, y_end)
  overlap_xstart = max(0, x_start)
  overlap_xend = min(granule_w, x_end)

  if overlap_ystart < overlap_yend and overlap_xstart < overlap_xend:
    local_dy_valid = slice(overlap_ystart - y_start, overlap_yend - y_start)
    local_dx_valid = slice(overlap_xstart - x_start, overlap_xend - x_start)
    granule_dy_valid = slice(overlap_ystart, overlap_yend)
    granule_dx_valid = slice(overlap_xstart, overlap_xend)
    validity_mask[local_dy_valid, local_dx_valid] = granule_mask[
        granule_dy_valid, granule_dx_valid
    ]

  return exclusion_mask, validity_mask


def _attach_masks(
    chunked_granule: data_types.ChunkedGranule,
    plumes: Sequence[data_types.Plume],
    candidates: Sequence[data_types.PlumeCandidate],
    plume_probability_threshold: float,
    vetting_patch_size: int,
) -> Sequence[data_types.Plume]:
  """Attaches exclusion and validity masks to plumes."""
  results = []
  granule_mask = chunked_granule.mask.squeeze(-1)
  for plume in plumes:
    ex_mask, val_mask = _calculate_exclusion_mask(
        plume,
        candidates,
        plume_probability_threshold,
        vetting_patch_size,
        granule_mask,
    )
    results.append(
        dataclasses.replace(
            plume,
            exclusion_mask=ex_mask,
            validity_mask=val_mask,
        )
    )
  return results


def deduplicate_candidates(
    plumes: Sequence[data_types.PlumeCandidate],
    geotransform: tuple[float, float, float, float, float, float],
    utm_zone: str,
    timestamp_ms: int,
    tile_weight: np.ndarray,
    cluster_alg: str,
    cluster_kw: dict[str, Any],
    plume_probability_threshold: float,
    origin_probability_threshold: float,
    regularizer: float,
    scale: float,
    simplify: float,
    keep_holes: bool,
    cc_min_component_size: int,
    border_on_plume_images: int,
    max_candidates_per_cluster: int,
    granule_mask: np.ndarray | None = None,
    metrics: Any = None,
    retain_origin_component: bool = True,
    log: bool = False,
) -> Sequence[data_types.Plume] | None:
  """Deduplicates plumes using clustering and aggregation."""
  if not plumes:
    return None

  group_ids = cluster_plumes(plumes, cluster_alg, cluster_kw)

  plume_groups = _aggregate_plume_group_data(
      plumes,
      group_ids,
      regularizer,
      tile_weight,
      granule_mask,  # pyrefly: ignore[bad-argument-type]
      plume_probability_threshold,
      max_candidates_per_cluster,
      log,
  )

  deduped_plumes = _create_plumes(
      plume_groups,
      geotransform=geotransform,
      utm_zone=utm_zone,
      timestamp_ms=timestamp_ms,
      scale=scale,
      plume_probability_threshold=plume_probability_threshold,
      origin_probability_threshold=origin_probability_threshold,
      cc_min_component_size=cc_min_component_size,
      border_on_plume_images=border_on_plume_images,
      simplify=simplify,
      keep_holes=keep_holes,
      metrics=metrics,
      retain_origin_component=retain_origin_component,
      log=log,
  )

  if not deduped_plumes:
    return None

  return deduped_plumes


def dedupe_plumes_and_calculate_spectral_vetting_inputs(
    chunked_granule: data_types.ChunkedGranule,
    plumes: Sequence[data_types.PlumeCandidate],
    tile_weight: np.ndarray,
    cluster_alg: str,
    cluster_kw: dict[str, Any],
    plume_probability_threshold: float,
    origin_probability_threshold: float,
    regularizer: float,
    scale: float,
    simplify: float,
    keep_holes: bool,
    cc_min_component_size: int,
    border_on_plume_images: int,
    vetting_patch_size: int,
    max_candidates_per_cluster: int,
    metrics: Any = None,
    retain_origin_component: bool = True,
    log: bool = False,
) -> Sequence[data_types.Plume] | None:
  """Deduplicates plumes using clustering and aggregation."""
  deduped_plumes = deduplicate_candidates(
      plumes=plumes,
      geotransform=chunked_granule.geotransform,
      utm_zone=chunked_granule.utm_zone,
      timestamp_ms=chunked_granule.timestamp_ms,
      tile_weight=tile_weight,
      cluster_alg=cluster_alg,
      cluster_kw=cluster_kw,
      plume_probability_threshold=plume_probability_threshold,
      origin_probability_threshold=origin_probability_threshold,
      regularizer=regularizer,
      scale=scale,
      simplify=simplify,
      keep_holes=keep_holes,
      cc_min_component_size=cc_min_component_size,
      border_on_plume_images=border_on_plume_images,
      granule_mask=chunked_granule.mask.squeeze(-1),
      max_candidates_per_cluster=max_candidates_per_cluster,
      metrics=metrics,
      retain_origin_component=retain_origin_component,
      log=log,
  )
  if not deduped_plumes:
    return None

  return _attach_masks(
      chunked_granule,
      deduped_plumes,
      plumes,
      plume_probability_threshold,
      vetting_patch_size,
  )


def parse_clustering_params(params_str: str) -> dict[str, Any]:
  """Parses clustering parameters from a string."""
  kw = {}
  for item in params_str.split(':'):
    key, val = item.split('=')
    if val.lower() == 'true':
      val = True
    elif val.lower() == 'false':
      val = False
    elif val.lower() == 'none':
      val = None
    else:
      try:
        val = int(val)
      except ValueError:
        try:
          val = float(val)
        except ValueError:
          pass
    kw[key] = val
  return kw
