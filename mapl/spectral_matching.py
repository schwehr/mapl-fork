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

"""Spectral matching functionality.

This module provides spectral vetting for GHG plumes using pre-computed
spectral statistics bundled as an NPZ file.
"""

from collections.abc import Sequence
import dataclasses
import importlib.resources
from typing import Tuple

from absl import logging
import numpy as np
from scipy import optimize
from scipy.spatial import distance


@dataclasses.dataclass
class SpectralMatchResult:
  """Dataclass for holding spectral match results."""

  d_norm: float
  d_cor: float
  fitted_conc: float
  baseline_transmittance: np.ndarray
  observed_transmittance_sel: np.ndarray
  modeled_transmittance: np.ndarray
  wavelengths_sel: np.ndarray
  observed_conc: float


@dataclasses.dataclass
class _SpectralData:
  """Container for per-band spectral data loaded from the NPZ file."""

  spectral_responses: list[np.ndarray]
  transmittances: list[np.ndarray]
  spectral_data_added_ppm_m: np.ndarray


@dataclasses.dataclass
class _SpectralDataStandardAndGas:
  """Container for standard and gas spectral data."""

  standard_data: _SpectralData
  gas_data: _SpectralData


def _load_spectral_data_from_npz(
    band_names: Sequence[str],
    npz_filename: str,
    input_type: str = 'emit_l1b',
    gas: str = 'methane',
) -> _SpectralDataStandardAndGas:
  """Loads spectral data from the bundled NPZ file.

  Args:
    band_names: Band names to load (e.g., ['radiance_0', 'radiance_1', ...]).
    npz_filename: Bundled NPZ filename to load.
    input_type: Sensor input type (e.g., 'emit_l1b').
    gas: Gas type (e.g., 'methane').

  Returns:
    _SpectralDataStandardAndGas with standard and gas spectral data.
  """
  ref = importlib.resources.files(__package__) / npz_filename

  if input_type == 'enmap_l1c':
    # ENMAP L1C is the same as EMIT L1B when reading the spectra.
    input_type = 'enmap_l1b'

  std_spectral_responses = []
  std_transmittances = []
  gas_spectral_responses = []
  gas_transmittances = []
  gas_added_ppm_m_values = []

  with importlib.resources.as_file(ref) as npz_path:
    data = np.load(npz_path, allow_pickle=True)

    for band in band_names:
      prefix_std = f'{input_type}_{band}_none'
      prefix_gas = f'{input_type}_{band}_{gas}'

      std_spectral_responses.append(data[f'{prefix_std}_spectral_responses'])
      std_transmittances.append(data[f'{prefix_std}_transmittances'])
      gas_spectral_responses.append(data[f'{prefix_gas}_spectral_responses'])
      gas_transmittances.append(data[f'{prefix_gas}_transmittances'])
      gas_added_ppm_m_values.append(
          data[f'{prefix_gas}_spectral_data_added_ppm-m']
      )

  return _SpectralDataStandardAndGas(
      standard_data=_SpectralData(
          spectral_responses=std_spectral_responses,
          transmittances=std_transmittances,
          spectral_data_added_ppm_m=np.zeros(len(band_names)),
      ),
      gas_data=_SpectralData(
          spectral_responses=gas_spectral_responses,
          transmittances=gas_transmittances,
          spectral_data_added_ppm_m=np.array(
              [v[0] if v.size > 0 else 0.0 for v in gas_added_ppm_m_values]
          ),
      ),
  )


class SpectralVetting:
  """Class for spectral vetting of GHG plumes."""

  def __init__(
      self,
      band_names: Sequence[str],
      wavelengths: np.ndarray,
      npz_filename: str,
      input_type: str = 'emit_l1b',
      gas: str = 'methane',
  ):
    self.band_names = band_names
    self.wavelengths = wavelengths
    self.npz_filename = npz_filename
    self.input_type = input_type
    self.gas = gas

    self.spectral_data = _load_spectral_data_from_npz(
        band_names, self.npz_filename, self.input_type, self.gas
    )
    # These are of shape (num_bands, N) where N is pretty large (matches the
    # shape of the spectral responses of each band)
    self.standard_transmittances = (
        self.spectral_data.standard_data.transmittances
    )
    self.gas_by_std_transmittances = [
        self.spectral_data.gas_data.transmittances[i]
        / (self.spectral_data.standard_data.transmittances[i] + 1e-12)
        for i in range(len(band_names))
    ]

  def _get_strong_gas_absorption_band_mask(self, gas):
    """Returns a mask for bands strongly absorbing the gas."""
    if gas == 'methane':
      return (self.wavelengths >= 2100) & (self.wavelengths <= 2440)
    elif gas == 'co2':
      return (self.wavelengths >= 1930) & (self.wavelengths <= 2130)
    else:
      raise ValueError(f'Gas {gas} not supported.')

  def _get_non_gas_band_mask(self, gas):
    """Returns a mask for bands not affected by the gas for matching."""
    if gas == 'methane':
      return ((self.wavelengths < 1640) | (self.wavelengths > 1690)) & (
          (self.wavelengths < 2100) | (self.wavelengths > 2440)
      )
    elif gas == 'co2':
      return (
          ((self.wavelengths < 1400) | (self.wavelengths > 1470))
          & ((self.wavelengths < 1550) | (self.wavelengths > 1650))
          & ((self.wavelengths < 1900) | (self.wavelengths > 2150))
      )
    else:
      raise ValueError(f'Gas {gas} not supported.')

  def get_observed_spectral_info(
      self,
      radiance: np.ndarray,
      plume_mask: np.ndarray,
      background_mask: np.ndarray,
      enhancement_image: np.ndarray,
      to_sun_zenith_angle: np.ndarray,
      to_sensor_zenith_angle: np.ndarray,
      background_ppm_m_threshold: float,
  ) -> Tuple[np.ndarray, float, float] | None:
    """Computes transmittance ratio between sampled label and background pixels.

    Reference:
    https://www.sciencedirect.com/science/article/pii/S0034425725002640

    Args:
        radiance: Radiance cube (height, width, bands).
        plume_mask: Plume mask (height, width).
        background_mask: Background mask (height, width).
        enhancement_image: Enhancement image (height, width) containing ppm-m
          values.
        to_sun_zenith_angle: Solar zenith angle for each pixel.
        to_sensor_zenith_angle: Sensor zenith angle for each pixel.
        background_ppm_m_threshold: Background ppm-m threshold. Usually 30 for
          methane and 30 * 100 for CO2.

    Returns:
        Tuple containing:
          - transmittance_ratio: Observed transmittance ratio.
          - path_length_factor: Path length factor to use for the plume.
          - plume_enhancement_ppm_m: Average enhancement ppm-m of the selected
            plume pixels.
    """
    h, w, b = radiance.shape

    # Drop masked out radiance pixels from the computation.
    input_not_nan_mask = np.all(np.isfinite(radiance), axis=-1)
    input_non_zero_mask = np.logical_or.reduce(
        np.array(radiance) != 0, axis=-1
    ).astype(bool)
    final_mask = input_not_nan_mask & input_non_zero_mask

    # Select plume pixels from 3x3 regions around top 30 pixels by enhancement.
    plume_mask = plume_mask & final_mask
    # Drop the plume pixels greater than the 99th percentile.
    # Following the configs in
    # https://github.com/emit-sds/plume-vetting/blob/main/pv/config/config.yaml
    if not np.any(plume_mask):
      return None
    plume_mask &= enhancement_image <= np.percentile(
        enhancement_image[plume_mask], 99
    )
    if not np.any(plume_mask):
      return None
    plume_coords = np.argwhere(plume_mask)
    plume_enhancements = enhancement_image[
        plume_coords[:, 0], plume_coords[:, 1]
    ]
    top_indices = np.argsort(plume_enhancements)[::-1]
    # Following the configs in
    # https://github.com/emit-sds/plume-vetting/blob/main/pv/config/config.yaml
    num_top_pixels = 30
    top_plume_coords = plume_coords[
        top_indices[: min(num_top_pixels, len(top_indices))]
    ]

    plume_pixels_mask = np.zeros_like(plume_mask, dtype=bool)
    for r, c in top_plume_coords:
      r_min, r_max = max(0, r - 1), min(h, r + 2)
      c_min, c_max = max(0, c - 1), min(w, c + 2)
      plume_pixels_mask[r_min:r_max, c_min:c_max] = True
    # Resulting plume pixels must be in plume_mask and final_mask.
    plume_pixels_mask &= plume_mask

    target_idx = np.where(plume_pixels_mask.flatten())[0]
    actual_n = target_idx.size
    if actual_n == 0:
      return None

    background_mask = background_mask & final_mask & ~plume_pixels_mask
    background_mask &= enhancement_image <= background_ppm_m_threshold
    bg_indices = np.where(background_mask.flatten())[0]
    if bg_indices.size == 0:
      return None

    x_flat = radiance.reshape(-1, b)
    to_sun_zenith_angle_flat = to_sun_zenith_angle.reshape(-1)
    to_sensor_zenith_angle_flat = to_sensor_zenith_angle.reshape(-1)
    path_length_factor = np.nanmean(
        1.0 / np.cos(np.radians(to_sun_zenith_angle_flat))
        + 1.0 / np.cos(np.radians(to_sensor_zenith_angle_flat))
    )

    target_enhancement = np.mean(enhancement_image.reshape(-1)[target_idx])

    targets = x_flat[target_idx].astype(np.float64)  # (actual_n, Bands)
    bg_pool = x_flat[bg_indices].astype(np.float64)  # (bg_indices, Bands)

    # Mask out gas absorbing bands for matching plume pixels with backgrounds.
    wl_mask = self._get_non_gas_band_mask(self.gas)

    # Uniquely match plume pixels with background pixels based on the distance
    # between their spectral signatures on the non-gas bands. Note that
    # the background pool is much larger than the number of plume pixels.
    cost_matrix = distance.cdist(
        targets[:, wl_mask], bg_pool[:, wl_mask], metric='euclidean'
    )
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)

    costs = cost_matrix[row_ind, col_ind]
    # Following the configs in
    # https://github.com/emit-sds/plume-vetting/blob/main/pv/config/config.yaml
    # They only retain the best 50% matching pairs.
    num_to_keep = int(len(costs) * 0.5)
    best_match_indices = np.argsort(costs)[:num_to_keep]
    row_ind, col_ind = row_ind[best_match_indices], col_ind[best_match_indices]
    matched_target_spectra = targets[row_ind]
    matched_bg_spectra = bg_pool[col_ind]

    mean_target = np.mean(matched_target_spectra, axis=0)
    mean_bg = np.mean(matched_bg_spectra, axis=0)

    transmittance = mean_target / (mean_bg + 1e-20)

    return transmittance, path_length_factor, target_enhancement  # pyrefly: ignore[bad-return]

  def _calculate_ref_transmittance_ratios(
      self,
      conc,
      path_length_factor,
  ):
    return np.array([
        np.sum(
            self.spectral_data.standard_data.spectral_responses[i]
            * np.pow(self.standard_transmittances[i], path_length_factor)
            * np.pow(
                self.gas_by_std_transmittances[i],
                path_length_factor * conc,
            )
        )
        / (
            np.sum(
                self.spectral_data.standard_data.spectral_responses[i]
                * np.pow(self.standard_transmittances[i], path_length_factor)
            )
            + 1e-20
        )
        for i in range(len(self.band_names))
    ])

  def match_spectral_signature(
      self,
      radiance: np.ndarray,
      plume_mask: np.ndarray,
      background_mask: np.ndarray,
      enhancement_ppm_m: np.ndarray,
      to_sun_zenith_angle: np.ndarray,
      to_sensor_zenith_angle: np.ndarray,
      poly_order: int = 6,
  ) -> SpectralMatchResult | None:
    """Matches spectral signature of plume pixels to the gas signature."""
    plume_mask = plume_mask.astype(bool)
    background_mask = background_mask.astype(bool)

    logging.debug('Getting observed spectral info: %s', enhancement_ppm_m)
    if self.gas == 'methane':
      background_ppm_m_threshold = 30.0
    elif self.gas == 'co2':
      background_ppm_m_threshold = 30 * 100
    else:
      raise ValueError(f'Gas {self.gas} not supported.')
    observed_spectral_info = self.get_observed_spectral_info(
        radiance,
        plume_mask,
        background_mask,
        enhancement_ppm_m,
        to_sun_zenith_angle,
        to_sensor_zenith_angle,
        background_ppm_m_threshold,
    )
    if observed_spectral_info is None:
      return None

    observed_transmittance, path_length_factor, observed_conc = (
        observed_spectral_info
    )
    logging.debug(
        'Observed transmittance and mean plume concentration: %s, %s',
        observed_transmittance,
        observed_conc,
    )

    # Gas strongly absorbing wavelengths.
    wavelength_mask = self._get_strong_gas_absorption_band_mask(self.gas)
    wavelengths_sel = self.wavelengths[wavelength_mask]
    observed_transmittance_sel = observed_transmittance[wavelength_mask]

    wav_center = np.mean(wavelengths_sel)
    wav_scale = np.std(wavelengths_sel)
    wavelengths_norm = (wavelengths_sel - wav_center) / wav_scale

    # Defining a transmittance model as follows:
    # T_model = (Polynomial Baseline) * (Pure Gas Transmission)
    def model_func(params):
      conc = params[0]
      coeffs = params[1:]
      poly_baseline = np.polynomial.polynomial.polyval(wavelengths_norm, coeffs)
      ref_transmittance_ratios_sel = self._calculate_ref_transmittance_ratios(
          conc,
          path_length_factor,
      )[wavelength_mask]
      return poly_baseline * ref_transmittance_ratios_sel

    def residuals_func(params, y_observed):
      return model_func(params) - y_observed

    # Guess a flat baseline (coeff=1.0) and small concentration
    # We guess poly coeffs by fitting the T_obs roughly (assuming conc=0 for a
    # moment)
    poly_guess = np.polynomial.polynomial.polyfit(
        wavelengths_norm, observed_transmittance_sel, poly_order
    )
    x0 = np.concatenate(([1.0], poly_guess))

    lower_bounds = np.concatenate(([0.0], np.full(poly_order + 1, -np.inf)))
    upper_bounds = np.full(poly_order + 2, np.inf)

    try:
      result = optimize.least_squares(
          residuals_func,
          x0,
          args=(observed_transmittance_sel,),
          method='trf',
          bounds=(lower_bounds, upper_bounds),
          loss='soft_l1',
      )
    except ValueError as e:
      logging.warning('Value error when optimizing least squares: %s', e)
      # If the initial guess is out of bounds for least squares then return
      # None. This only happens sometimes during initial stages of training where
      # the model predicts random noise.
      return None

    # Note that the fitted_conc corresponds to roughly the plume average
    # concentration here and not the peak.
    fitted_conc = result.x[0]
    fitted_coeffs = result.x[1:]

    # Calculate the "Baseline" (just the polynomial part) for visualization
    # Baseline = Intercept + c1*w + c2*w^2 ...
    baseline_transmittance = np.polynomial.polynomial.polyval(
        wavelengths_norm, fitted_coeffs
    )
    modeled_transmittance = model_func(result.x)

    # Compute metrics
    e_val = observed_transmittance_sel / (baseline_transmittance + 1e-20)
    m_val = self._calculate_ref_transmittance_ratios(
        fitted_conc,
        path_length_factor,
    )[wavelength_mask]
    # Normalized mean abs difference
    d_val = np.mean(np.abs(e_val - m_val))
    d_norm = d_val / np.mean(np.abs(e_val - np.mean(e_val)))
    pearson_r = np.corrcoef(e_val, m_val)[0, 1]
    # Anti correlation
    d_cor = 1 - (pearson_r**2)

    return SpectralMatchResult(
        d_norm=d_norm,
        d_cor=d_cor,
        fitted_conc=fitted_conc
        * self.spectral_data.gas_data.spectral_data_added_ppm_m[0],
        baseline_transmittance=baseline_transmittance,  # pyrefly: ignore[bad-argument-type]
        observed_transmittance_sel=observed_transmittance_sel,
        modeled_transmittance=modeled_transmittance,
        wavelengths_sel=wavelengths_sel,
        observed_conc=observed_conc,
    )
