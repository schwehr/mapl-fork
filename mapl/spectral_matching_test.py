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

from unittest import mock

from absl.testing import absltest
from mapl import spectral_matching
import numpy as np


class SpectralMatchingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.band_names = [
        'b1',
        'b2',
        'b3',
        'b4',
        'b5',
        'b6',
        'b7',
        'b8',
        'b9',
        'b10',
        'b11',
        'b12',
    ]
    self.wavelengths_list = [
        1600,
        1660,
        2000,
        2100,
        2150,
        2200,
        2250,
        2300,
        2350,
        2400,
        2420,
        2500,
    ]
    n_points = 10
    num_bands = len(self.band_names)
    mock_spectral_data = mock.MagicMock()
    mock_spectral_data.standard_data.transmittances = [
        np.ones(n_points) for _ in range(num_bands)
    ]
    mock_spectral_data.standard_data.spectral_responses = [
        np.ones(n_points) for _ in range(num_bands)
    ]
    # If std transmittance is 1.0 and gas transmittance is 0.8,
    # gas_by_std_transmittances will be 0.8.
    mock_spectral_data.gas_data.transmittances = [
        np.full(n_points, 0.8) for _ in range(num_bands)
    ]
    mock_spectral_data.gas_data.spectral_data_added_ppm_m = np.full(
        (num_bands,), 100.0
    )

    self.mock_load = self.enter_context(
        mock.patch(
            'mapl.spectral_matching._load_spectral_data_from_npz',
            return_value=mock_spectral_data,
        )
    )

    self.spectral_vetter = spectral_matching.SpectralVetting(
        self.band_names,
        np.array(self.wavelengths_list),
        npz_filename='test_statistics.npz',
    )

  def test_init_passes_npz_filename(self):
    self.assertEqual(self.spectral_vetter.npz_filename, 'test_statistics.npz')
    self.mock_load.assert_called_once_with(
        self.band_names, 'test_statistics.npz', 'emit_l1b', 'methane'
    )

  def test_get_observed_spectral_info(self):
    height, width, bands = 10, 10, len(self.band_names)
    radiance = np.full((height, width, bands), 2.0)
    plume_mask = np.zeros((height, width), dtype=bool)
    plume_mask[4:6, 4:6] = True
    background_mask = ~plume_mask
    radiance[plume_mask] = 1.0
    enhancement_image = np.zeros((height, width))
    enhancement_image[4:6, 4:6] = 10
    enhancement_image[4, 4] = 20  # make one pixel higher to pass 75 percentile
    to_sun_zenith_angle = np.zeros((height, width))
    to_sensor_zenith_angle = np.zeros((height, width))

    result = self.spectral_vetter.get_observed_spectral_info(
        radiance,
        plume_mask,
        background_mask,
        enhancement_image,
        to_sun_zenith_angle,
        to_sensor_zenith_angle,
        30.0,
    )
    self.assertIsNotNone(result)
    transmittance_ratio, path_length_factor, target_enhancement = result
    self.assertEqual(transmittance_ratio.shape, (bands,))
    self.assertAlmostEqual(path_length_factor, 2.0)
    self.assertAlmostEqual(target_enhancement, 10.0)

    np.testing.assert_allclose(transmittance_ratio, 0.5, atol=1e-6)

  def test_match_spectral_signature(self):
    height, width, bands = 10, 10, len(self.band_names)
    # Background pixels have radiance 2.0, plume pixels 1.0.
    radiance = np.full((height, width, bands), 2.0)
    plume_mask = np.zeros((height, width), dtype=bool)
    plume_mask[4:6, 4:6] = True
    background_mask = ~plume_mask
    radiance[plume_mask] = 1.0

    enhancement_ppm_m = np.zeros((height, width))
    enhancement_ppm_m[4:6, 4:6] = 10
    enhancement_ppm_m[4, 4] = 20  # make one pixel higher to pass 75 percentile
    to_sun_zenith_angle = np.zeros((height, width))
    to_sensor_zenith_angle = np.zeros((height, width))

    result = self.spectral_vetter.match_spectral_signature(
        radiance,
        plume_mask,
        background_mask,
        enhancement_ppm_m,
        to_sun_zenith_angle,
        to_sensor_zenith_angle,
    )
    self.assertIsNotNone(result)
    self.assertTrue(np.isscalar(result.d_norm))
    self.assertTrue(np.isscalar(result.d_cor))
    self.assertTrue(np.isscalar(result.fitted_conc))

    num_wavelength_sel = 8  # bands in 2100-2440 range
    self.assertEqual(result.baseline_transmittance.shape, (num_wavelength_sel,))
    self.assertEqual(
        result.observed_transmittance_sel.shape, (num_wavelength_sel,)
    )
    self.assertEqual(result.modeled_transmittance.shape, (num_wavelength_sel,))

    # Observed transmittance and modeled transmittance should be 1.0 / 2.0.
    np.testing.assert_allclose(
        result.observed_transmittance_sel, 0.5, atol=1e-6
    )
    np.testing.assert_allclose(result.modeled_transmittance, 0.5, atol=1e-6)


class LoadSpectralDataFromNpzTest(absltest.TestCase):

  def test_load_spectral_data_from_npz_resource(self):
    data = spectral_matching._load_spectral_data_from_npz(
        band_names=['radiance_0'],
        npz_filename='satellite_gas_statistics_v14_emit.npz',
        input_type='emit_l1b',
        gas='methane',
    )
    self.assertLen(data.standard_data.spectral_responses, 1)
    self.assertLen(data.gas_data.spectral_responses, 1)


if __name__ == '__main__':
  absltest.main()
