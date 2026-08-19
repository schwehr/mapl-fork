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

"""Library for writing plumes to parquet."""

from typing import Any

from mapl import data_types
import pyarrow as pa
import shapely.wkt


def get_candidate_plumes_base_schema(store_rasters_data: bool) -> pa.Schema:
  """Returns the base schema for the candidate plumes without sweep params."""
  fields = [
      pa.field("chunk_id", pa.int64()),
      pa.field("slot_id", pa.int64()),
      pa.field("plume_bbox_px", pa.list_(pa.int64())),
      pa.field("ee_asset_id", pa.string()),
      pa.field("geometry_px", pa.string()),
      pa.field("geometry", pa.string()),
      pa.field("head_point_px", pa.string()),
      pa.field("head_point", pa.string()),
  ]
  if store_rasters_data:
    fields.extend([
        pa.field("concentration", pa.list_(pa.float32())),
        pa.field("binary_mask", pa.list_(pa.bool_())),
        pa.field("origin_mask", pa.list_(pa.bool_())),
        pa.field("mask", pa.list_(pa.bool_())),
    ])
  return pa.schema(fields)


def format_candidate_plume_base(
    candidate: data_types.PlumeCandidate,
    ee_asset_id: str,
    store_rasters_data: bool,
    plume_prob: float,
    origin_prob: float,
) -> dict[str, Any]:
  """Formats the base properties of a single candidate plume."""
  # Head may not be present in the prediction so we need to check for None.
  head_point_px = ""
  if candidate.head_point_px is not None:
    head_point_px = shapely.wkt.dumps(candidate.head_point_px)
  head_point = ""
  if candidate.head_point is not None:
    head_point = shapely.wkt.dumps(candidate.head_point)

  metadata = {
      "chunk_id": candidate.chunk_id,
      "slot_id": candidate.slot_id,
      "plume_bbox_px": candidate.plume_bbox_px,
      "ee_asset_id": ee_asset_id,
  }

  geometries = {
      "geometry_px": shapely.wkt.dumps(candidate.geometry_px),
      "geometry": shapely.wkt.dumps(candidate.geometry),
      "head_point_px": head_point_px,
      "head_point": head_point,
  }

  if store_rasters_data:
    binary_mask = candidate.binary_masks > plume_prob
    origin_mask = candidate.origin_masks > origin_prob
    rasters_data = {
        "concentration": candidate.concentration.flatten().tolist(),
        "binary_mask": binary_mask.flatten().tolist(),
        "origin_mask": origin_mask.flatten().tolist(),
        "mask": candidate.mask.astype(bool).flatten().tolist(),
    }
    return {**metadata, **geometries, **rasters_data}
  else:
    return {**metadata, **geometries}


def get_predicted_plumes_base_schema() -> pa.Schema:
  """Returns the base schema for predicted plumes without sweep params."""
  return pa.schema([
      pa.field("ee_asset_id", pa.string()),
      pa.field("geometry_px", pa.string()),
      pa.field("geometry", pa.string()),
      pa.field("head_point_px", pa.string()),
      pa.field("head_point", pa.string()),
      pa.field("ime_emission_rate", pa.float32()),
      pa.field("cluster_size", pa.int64()),
      pa.field("d_norm", pa.float32()),
      pa.field("d_cor", pa.float32()),
      pa.field("fitted_conc", pa.float32()),
      pa.field("baseline_transmittance", pa.list_(pa.float32())),
      pa.field("observed_transmittance_sel", pa.list_(pa.float32())),
      pa.field("modeled_transmittance", pa.list_(pa.float32())),
      pa.field("wavelengths_sel", pa.list_(pa.float32())),
      pa.field("observed_conc", pa.float32()),
      pa.field("spectral_vetting_plume_pixels", pa.int64()),
      pa.field("spectral_vetting_background_pixels", pa.int64()),
      pa.field("ime_pixel_area", pa.float32()),
      pa.field("ime_l_min", pa.float32()),
      pa.field("ime_l_max", pa.float32()),
      pa.field("ime_integrated_mass", pa.float32()),
      pa.field("ime_xp", pa.list_(pa.float32())),
      pa.field("ime_yp", pa.list_(pa.float32())),
      pa.field("hrrr_u_10m", pa.float32()),
      pa.field("hrrr_v_10m", pa.float32()),
      pa.field("era5_u_10m", pa.float32()),
      pa.field("era5_v_10m", pa.float32()),
  ])


def format_predicted_plume_base(
    plume: data_types.Plume,
    ee_asset_id: str,
) -> dict[str, Any]:
  """Formats the base properties of a single predicted plume."""
  record = {
      "ee_asset_id": ee_asset_id,
      "geometry_px": shapely.wkt.dumps(plume.geometry_px),
      "geometry": shapely.wkt.dumps(plume.geometry),
      "head_point_px": shapely.wkt.dumps(plume.head_point_px),
      "head_point": shapely.wkt.dumps(plume.head_point),
      "ime_emission_rate": plume.metadata.get("ime_emission_rate"),
      "cluster_size": plume.cluster_size,
  }
  for key in [
      "d_norm",
      "d_cor",
      "fitted_conc",
      "baseline_transmittance",
      "observed_transmittance_sel",
      "modeled_transmittance",
      "wavelengths_sel",
      "observed_conc",
      "spectral_vetting_plume_pixels",
      "spectral_vetting_background_pixels",
      "ime_pixel_area",
      "ime_l_min",
      "ime_l_max",
      "ime_integrated_mass",
      "ime_xp",
      "ime_yp",
      "hrrr_u",
      "hrrr_v",
      "era5_u",
      "era5_v",
  ]:
    value = plume.metadata.get(key)
    if value is not None:
      output_key = key
      if key in ["hrrr_u", "hrrr_v", "era5_u", "era5_v"]:
        output_key = key + "_10m"
      record[output_key] = value.tolist() if hasattr(value, "tolist") else value
  return record
