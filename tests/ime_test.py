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

"""Tests for Integrated Methane Enhancement (IME)."""

from absl.testing import absltest
from mapl import ime
import numpy as np
import pyproj


def create_oval_plume(shape, radius_x, radius_y, enhancement):
  rows, cols = shape
  center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
  y, x = np.ogrid[:rows, :cols]
  plume_mask = (
      ((x - center_c) / radius_x) ** 2 + ((y - center_r) / radius_y) ** 2
  ) <= 1
  plume = np.zeros(shape)
  plume[plume_mask] = enhancement
  return plume, plume_mask


class ImeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.shape = (256, 256)
    self.plume_radius_x = 100
    self.plume_radius_y = 10
    self.plume_enh_ppm_m_val = 10.0
    self.background = 0.0
    gas_ppm_m_enh, self.plume_mask_arr = create_oval_plume(
        self.shape,
        self.plume_radius_x,
        self.plume_radius_y,
        self.plume_enh_ppm_m_val,
    )
    self.gas_ppm_m = gas_ppm_m_enh + self.background
    self.pixel_size_m = 10.0
    self.lat_center, self.lon_center = 40.0, 93.0

    transformer = pyproj.Transformer.from_crs(
        "EPSG:32646", "EPSG:4326", always_xy=True
    )
    # UTM zone 46N center coordinates for 93E, 40N
    easting_center, northing_center = 500000, 4428136

    width_m = self.shape[1] * self.pixel_size_m
    height_m = self.shape[0] * self.pixel_size_m

    x_min_utm = easting_center - width_m / 2
    x_max_utm = easting_center + width_m / 2
    y_min_utm = northing_center - height_m / 2
    y_max_utm = northing_center + height_m / 2

    lon_min, lat_min = transformer.transform(x_min_utm, y_min_utm)
    lon_max, lat_max = transformer.transform(x_max_utm, y_max_utm)

    self.bounding_box = (lat_min, lon_min, lat_max, lon_max)

    n_pixels = np.sum(self.plume_mask_arr)
    # ppm-m to mol/m2: val * 0.0000423144
    # mol/m2 to kg/m2 for CH4: val * 0.01604 kg/mol
    # pixel area: 10*10=100m2
    # integrated mass: n_pixels * 100 * 10.0 * 0.0000423144 * 0.01604 kg
    self.expected_ime = (
        n_pixels
        * self.pixel_size_m**2
        * self.plume_enh_ppm_m_val
        * 0.0000423144
        * 0.01604
    )

  def test_compute_ime_with_plume_mask(self):
    ime_result = ime.compute_ime(
        lat=self.lat_center,
        lng=self.lon_center,
        wind=(-1.8, -1.8),
        bounding_box=self.bounding_box,
        plume_mask=self.plume_mask_arr,
        plume_enh_ppm_m=self.gas_ppm_m,
        gas="CH4",
        background_estimate=self.background,
        epsg_code="EPSG:32646",
    )

    self.assertAlmostEqual(
        self.expected_ime, ime_result["integrated_mass"], places=2
    )

  def test_compute_ime_without_plume_mask(self):
    ime_result = ime.compute_ime(
        lat=self.lat_center,
        lng=self.lon_center,
        wind=(1.8, 1.8),
        bounding_box=self.bounding_box,
        plume_mask=None,
        plume_enh_ppm_m=self.gas_ppm_m,
        gas="CH4",
        background_estimate=self.background,
        epsg_code="EPSG:32646",
    )
    self.assertAlmostEqual(
        self.expected_ime, ime_result["integrated_mass"], places=2
    )


if __name__ == "__main__":
  absltest.main()
