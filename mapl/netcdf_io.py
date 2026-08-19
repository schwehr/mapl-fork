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

"""NetCDF IO utilities for reading unprojected EMIT L1B NetCDF datasets."""

import math
import os

from absl import logging
import netCDF4 as nc
import numpy as np
import xarray as xr

from tensorflow.io import gfile


def ortho_xr(
    ds: xr.Dataset,
    glt_nodata_value: int = 0,
    fill_value: float | int = -9999,
) -> xr.Dataset:
  """Uses the GLT array to create an orthorectified xarray dataset."""
  glt_ds = np.nan_to_num(
      np.stack([ds["glt_x"].data, ds["glt_y"].data], axis=-1),
      nan=glt_nodata_value,
  ).astype(int)

  var_list = list(ds.data_vars)
  if "flat_field_update" in var_list:
    var_list.remove("flat_field_update")

  data_vars = {}
  for var in var_list:
    raw_ds = ds[var].data
    var_dims = ds[var].dims
    out_ds = orthorectify_with_glt(
        raw_ds,
        glt_ds,
        fill_value=fill_value,
        glt_nodata_value=glt_nodata_value,
    )
    out_ds[out_ds == fill_value] = np.nan

    if raw_ds.ndim == 2:
      out_ds = out_ds.squeeze()
      data_vars[var] = (["latitude", "longitude"], out_ds)
    else:
      data_vars[var] = (["latitude", "longitude", var_dims[-1]], out_ds)

    del raw_ds

  lon, lat = calc_dataset_latlons(ds)
  elev_ds = orthorectify_with_glt(
      ds["elev"].data,
      glt_ds,
      fill_value=fill_value,
      glt_nodata_value=glt_nodata_value,
  )
  elev_ds[elev_ds == fill_value] = np.nan
  del glt_ds

  coords = {
      "latitude": (["latitude"], lat),
      "longitude": (["longitude"], lon),
      **ds.coords,
  }

  for key in ["downtrack", "lat", "lon", "glt_x", "glt_y", "elev"]:
    del coords[key]

  coords["elev"] = (["latitude", "longitude"], np.squeeze(elev_ds))
  out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
  del out_ds  # pyrefly: ignore[unbound-name]

  for var in var_list:
    out_xr[var].attrs = ds[var].attrs
  out_xr.coords["latitude"].attrs = ds["lat"].attrs
  out_xr.coords["longitude"].attrs = ds["lon"].attrs
  out_xr.coords["elev"].attrs = ds["elev"].attrs

  return out_xr


def read_emit_netcdf(
    input_path: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
    tuple[float, ...],
    np.ndarray,
]:
  """Reads and orthorectifies EMIT L1B RAD and OBS data."""
  logging.info("Reading input EMIT data from: %s", input_path)

  input_data, obs_input_data = load_emit_dataset(input_path)

  geotransform = input_data.attrs.get(
      "geotransform", (0.0, 60.0, 0.0, 0.0, 0.0, -60.0)
  )
  projection = input_data.attrs.get("spatial_ref")
  if isinstance(projection, str) and projection.startswith("EPSG:32"):
    epsg = projection
  else:
    lon = geotransform[0]
    lat = geotransform[3]
    utm_zone = math.floor((lon + 180) / 6) + 1
    epsg_code = (32600 if lat >= 0 else 32700) + utm_zone
    epsg = f"EPSG:{int(epsg_code)}"

  crosstrack_ids = np.array(input_data.crosstrack)
  crosstrack_tile = np.tile(
      np.expand_dims(crosstrack_ids, 0),
      (input_data.radiance.shape[0], 1),
  ).astype(np.int64)

  input_glt = np.nan_to_num(
      np.stack([input_data["glt_x"].data, input_data["glt_y"].data], axis=-1),
      nan=0,
  ).astype(int)

  crosstrack_tile = orthorectify_with_glt(
      crosstrack_tile, input_glt, fill_value=-9999, glt_nodata_value=0
  )[:, :, 0]

  logging.info("Applying ortho_xr")
  input_data = ortho_xr(input_data)
  logging.info("Applying ortho_xr to OBS data")
  obs_input_data = ortho_xr(obs_input_data)

  logging.info("Extracting radiance and zenith arrays")

  # Derive mask from radiance data which still have NaNs
  valid_mask = ~np.isnan(input_data.radiance.data).any(axis=-1)
  mask = valid_mask.astype(np.uint8)[..., None]

  l1b_radiance = np.nan_to_num(input_data.radiance.data, nan=0.0).astype(
      np.float32
  )
  sensor_zenith = np.nan_to_num(
      obs_input_data.obs[:, :, 2].data, nan=0.0
  ).astype(np.float32)
  sun_zenith = np.nan_to_num(obs_input_data.obs[:, :, 4].data, nan=0.0).astype(
      np.float32
  )
  crosstrack_ids = crosstrack_tile

  return (
      l1b_radiance,
      sun_zenith,
      sensor_zenith,
      crosstrack_ids,
      epsg,
      geotransform,
      mask,
  )


def _load_single_dataset(
    filepath: str, band_dim: str, granule_id: str
) -> xr.Dataset:
  """Reads a single EMIT NetCDF dataset and returns an xarray.Dataset."""
  with gfile.GFile(filepath, "rb") as f:
    nc_buffer = f.read()
  nc_data = nc.Dataset(filepath, memory=nc_buffer)

  ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc_data))
  loc = xr.open_dataset(xr.backends.NetCDF4DataStore(nc_data, group="location"))

  data_vars = dict(ds.data_vars)

  coords = {
      "downtrack": (["downtrack"], ds.downtrack.data),
      "crosstrack": (["crosstrack"], ds.crosstrack.data),
  } | dict(loc.variables)

  out_xr_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
  out_xr_ds.attrs["granule_id"] = granule_id
  out_xr_ds = out_xr_ds.swap_dims({"bands": band_dim})

  for var in list(ds.data_vars):
    out_xr_ds[var].data[out_xr_ds[var].data == -9999] = np.nan

  return out_xr_ds


def load_emit_dataset(input_path: str) -> tuple[xr.Dataset, xr.Dataset]:
  """Reads EMIT L1B RAD and OBS datasets and returns (rad_ds, obs_ds)."""
  if "_L1B_RAD" not in input_path:
    raise ValueError("Input file must be an EMIT_L1B_RAD file.")

  rad_granule_id = os.path.splitext(os.path.basename(input_path))[0]
  rad_ds = _load_single_dataset(input_path, "wavelengths", rad_granule_id)

  obs_path = input_path.replace("_L1B_RAD", "_L1B_OBS")
  obs_granule_id = rad_granule_id.replace("_L1B_RAD", "_L1B_OBS")
  logging.info("Loading L1B_OBS data (%s)...", obs_path)
  obs_ds = _load_single_dataset(obs_path, "observation_bands", obs_granule_id)

  return rad_ds, obs_ds


def orthorectify_with_glt(
    ds_array: np.ndarray,
    glt_array: np.ndarray,
    fill_value: float | int,
    glt_nodata_value: int,
) -> np.ndarray:
  """Applies the GLT array to a numpy array, using fill_value for nodata."""
  if ds_array.ndim == 2:
    ds_array = ds_array[:, :, np.newaxis]
  out_ds = np.full(
      (glt_array.shape[0], glt_array.shape[1], ds_array.shape[-1]),
      fill_value,
      dtype=np.float32,
  )
  valid_glt = np.all(glt_array != glt_nodata_value, axis=-1)

  glt_array_copy = glt_array.copy()
  glt_array_copy[valid_glt] -= 1
  out_ds[valid_glt, :] = ds_array[
      glt_array_copy[valid_glt, 1], glt_array_copy[valid_glt, 0], :
  ]
  return out_ds


def calc_dataset_latlons(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
  """Calculates the Lat and Lon Coordinate Vectors."""
  gt = ds.geotransform
  dim_x = ds.glt_x.shape[1]
  dim_y = ds.glt_x.shape[0]
  lon = np.zeros(dim_x)
  lat = np.zeros(dim_y)
  for x in np.arange(dim_x):
    x_geo = (gt[0] + 0.5 * gt[1]) + x * gt[1]
    lon[x] = x_geo
  for y in np.arange(dim_y):
    y_geo = (gt[3] + 0.5 * gt[5]) + y * gt[5]
    lat[y] = y_geo
  return lon, lat
