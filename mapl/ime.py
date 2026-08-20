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

"""Integrated Methane Enhancement (IME) emission rate computation."""

from ddeq import dplume
from ddeq import emissions
from ddeq import ime
from ddeq import misc
from ddeq import plume_coords
import numpy as np
import pandas as pd
import pyproj
import xarray as xr


def get_source_xarray_dataset(lat, lng):
  point_sources_data = [
      ["Site1", "Site1", lng, lat, "city"],
  ]

  return misc.read_point_sources(point_sources_data)


def compute_ime(
    lat,
    lng,
    wind,
    bounding_box,
    plume_enh_ppm_m,
    plume_mask=None,
    gas="CH4",
    radius=600,
    background_estimate=0.0,
    epsg_code="EPSG:4326",
):
  """Computes the Integrated Methane Enhancement (IME) emission rate.

  This function takes satellite data, source information, and wind data to
  estimate the emission rate using the Integrated Methane Enhancement method.

  Args:
    lat: Latitude of the source.
    lng: Longitude of the source.
    wind: A pair of U-component of the wind vector (m/s) and V-component of the
      wind vector (m/s) in the format (wind_u, wind_v).
    bounding_box: A tuple of bounding box coordinates (min_lat, min_lon,
      max_lat, max_lon) in the format (min_lat, min_lon, max_lat, max_lon).
    plume_enh_ppm_m: A 2D array of the plume enhancement in ppm-m.
    plume_mask: A 2D array of the plume mask.
    gas: Name of the gas being analyzed (e.g., 'CH4').
    radius: Radius for plume coordinate calculation.
    background_estimate: background estimate if plume_mask is not provided.
    epsg_code: EPSG code for the projection.

  Returns:
    The estimated integrated gas mass emission rate.
  """
  # Convert gas concentrations from ppm-m to mol-m-2.
  enh_mol_per_m2 = plume_enh_ppm_m * 0.0000423144
  enh_std = np.ones_like(enh_mol_per_m2) * (np.mean(enh_mol_per_m2) / 10.0)

  rows = enh_mol_per_m2.shape[0]
  cols = enh_mol_per_m2.shape[1]
  sources_xarray = get_source_xarray_dataset(lat, lng)
  ncorners = 4

  # Setup an estimated lat/lng grid.
  res_lon = (bounding_box[3] - bounding_box[1]) / cols
  res_lat = (bounding_box[2] - bounding_box[0]) / rows
  lon_centers = np.linspace(
      bounding_box[1] + 0.5 * res_lon, bounding_box[3] - 0.5 * res_lon, cols
  )
  lat_centers = np.linspace(
      bounding_box[0] + 0.5 * res_lat, bounding_box[2] - 0.5 * res_lat, rows
  )
  lon_array_2d, lat_array_2d = np.meshgrid(lon_centers, lat_centers)
  lonc_array = np.zeros((rows, cols, ncorners))
  latc_array = np.zeros((rows, cols, ncorners))
  half_res_lon = res_lon / 2
  half_res_lat = res_lat / 2
  for i in range(rows):
    for j in range(cols):
      c_lon = lon_array_2d[i, j]  # Center lon of current pixel
      c_lat = lat_array_2d[i, j]  # Center lat of current pixel

      # Define corners relative to the center
      # Order: 0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left
      lonc_array[i, j, 0] = c_lon - half_res_lon
      latc_array[i, j, 0] = c_lat + half_res_lat  # North is positive lat

      lonc_array[i, j, 1] = c_lon + half_res_lon
      latc_array[i, j, 1] = c_lat + half_res_lat

      lonc_array[i, j, 2] = c_lon + half_res_lon
      latc_array[i, j, 2] = c_lat - half_res_lat

      lonc_array[i, j, 3] = c_lon - half_res_lon
      latc_array[i, j, 3] = c_lat - half_res_lat

  time_val = pd.to_datetime(pd.Timestamp.now())

  # Defaults.
  clouds_array = np.full((rows, cols), np.nan)
  psurf_array = np.full((rows, cols), np.nan)

  # Create the xarray Dataset used throughout to estimate the IME emission rate.
  ds = xr.Dataset(
      {
          "time": ([], time_val),
          "lon": (("rows", "cols"), lon_array_2d),
          "lat": (("rows", "cols"), lat_array_2d),
          "lonc": (("rows", "cols", "ncorners"), lonc_array),
          "latc": (("rows", "cols", "ncorners"), latc_array),
          "clouds": (("rows", "cols"), clouds_array),
          "psurf": (("rows", "cols"), psurf_array),
          gas: (("rows", "cols"), enh_mol_per_m2),
          gas + "_std": (("rows", "cols"), enh_std),
      },
      coords={
          "rows": np.arange(rows),
          "cols": np.arange(cols),
          "ncorners": np.arange(ncorners),
      },
  )
  # Add attributes
  ds.attrs["satellite"] = "EMIT"
  ds.attrs["orbit"] = -1
  ds.attrs["lon_eq"] = -1
  ds.attrs["DESCRIPTION"] = "IME Computation Data"
  ds.attrs["units"] = "mol m-2"
  ds[gas].attrs["units"] = "mol m-2"

  plumes = dplume.detect_plumes(
      ds,
      sources_xarray,
      variable=gas,
      variable_std=gas + "_std",
      filter_type="gaussian",
      filter_size=3,
      background=np.ones([rows, cols]) * background_estimate,
      plume_mask=plume_mask,
  )

  data, curves = plume_coords.compute_plume_line_and_coords(
      plumes,
      crs=pyproj.CRS(epsg_code),
      plume_area="hull",
      radius=radius,
  )

  # Setup winds to use for IME.
  speed = np.sqrt(wind[0] ** 2 + wind[1] ** 2)
  angle_from_east = np.degrees(np.arctan2(wind[1], wind[0]))
  direction = (270 - angle_from_east) % 360

  wind_data = xr.Dataset(
      {
          "u": (("source",), [wind[0]]),
          "v": (("source",), [wind[1]]),
          "speed": (("source",), [speed]),
          "direction": (("source",), [direction]),
          "speed_precision": (("source",), [0.9]),
      },
      coords={"source": ["Site1"]},
  )

  data = emissions.prepare_data(data, gas)

  pixel_area = data["pixel_area"].values[0]
  xp = data["xp"].values
  yp = data["yp"].values

  ime_result = ime.estimate_emissions(
      data, wind_data, sources_xarray, curves, gas, variable=f"{gas}_mass"
  )

  l_min = ime_result["L_min"].values[0]
  l_max = ime_result["L_max"].values[0]
  integrated_mass = ime_result[f"integrated_{gas}_mass"].values[0]
  emission_rate = ime_result[f"{gas}_estimated_emissions"].values[0] * 3600.0

  return {
      "pixel_area": pixel_area,
      "xp": xp,
      "yp": yp,
      "l_min": l_min,
      "l_max": l_max,
      "integrated_mass": integrated_mass,
      "emission_rate": emission_rate,
  }
