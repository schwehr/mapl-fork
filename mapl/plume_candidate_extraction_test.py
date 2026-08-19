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


from absl.testing import absltest
from absl.testing import parameterized
from mapl import data_types
from mapl import plume_candidate_extraction as extraction
from mapl import test_utils
import numpy as np
import shapely.geometry


class PipesTest(parameterized.TestCase):

  @parameterized.parameters(
      (0, '00:00'),
      (59, '00:59'),
      (60, '01:00'),
      (61, '01:01'),
      (120, '02:00'),
      (121, '02:01'),
      (3599, '59:59'),
      (3600, '60:00'),
  )
  def test_to_mmss(self, duration_seconds, expected):
    self.assertEqual(extraction.to_mmss(duration_seconds), expected)


class ExtractPlumesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = test_utils._get_test_config(
        scale=10.0, model_input_size=4, stride=1
    )

  def test_extract_plumes(self):
    chunk_size = 32
    # Create chunk data with a plume
    c1_data = test_utils._create_chunk_data(
        chunk_size, 10.0, 10, origin_pixel=(15, 15)
    )
    # Create a 10x10 square mask in the middle (rows 10-20, cols 10-20)
    c1_data['binary_masks'][:] = 0.0
    c1_data['binary_masks'][10:20, 10:20, :] = 0.9
    c1 = test_utils._create_chunk(c1_data, h_off=0, w_off=0)

    chunked_granule = data_types.ChunkedGranule(
        ee_asset_id='asset1',
        chunks=[c1],
        mask=np.ones((chunk_size, chunk_size, 1), dtype='uint8'),
        geotransform=(0.0, 10.0, 0.0, 40.0, 0.0, -10.0),
        utm_zone='10S',
        epsg='EPSG:32610',
        timestamp_ms=0,
    )

    # ExtractCandidates is the pure logic function called by ExtractPlumes DoFn.
    # We test the pure logic function here.

    results = extraction.extract_candidates(
        chunked_granule,
        plume_probability_threshold=0.5,
        origin_probability_threshold=0.4,
        log=False,
        cfg=self.cfg,
        metrics=None,
    )

    self.assertLen(results, 1)
    plume = results[0]
    self.assertIsInstance(plume, data_types.PlumeCandidate)
    self.assertIsInstance(plume.geometry_px, shapely.geometry.Polygon)


class ConvertBinaryMaskToShapelyPolygonTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.lat_lon_corners = np.array([
        [0, 0],  # bl
        [0, 10],  # br
        [10, 10],  # tr
        [10, 0],  # tl
    ])

  def test_full_mask(self):
    mask = np.ones((10, 10), dtype=np.uint8)
    polygon = extraction.convert_binary_mask_to_shapely_polygon(
        mask, self.lat_lon_corners, keep_holes=True
    )
    self.assertIsInstance(polygon, shapely.geometry.Polygon)
    self.assertAlmostEqual(polygon.area, 100, places=5)
    bounds = polygon.bounds
    self.assertAlmostEqual(bounds[0], 0, places=5)  # minx
    self.assertAlmostEqual(bounds[1], 0, places=5)  # miny
    self.assertAlmostEqual(bounds[2], 10, places=5)  # maxx
    self.assertAlmostEqual(bounds[3], 10, places=5)  # maxy

  def test_empty_mask(self):
    mask = np.zeros((10, 10), dtype=np.uint8)
    polygon = extraction.convert_binary_mask_to_shapely_polygon(
        mask, self.lat_lon_corners, keep_holes=True
    )
    self.assertIsNone(polygon)

  def test_half_mask(self):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, :5] = 1  # left half
    polygon = extraction.convert_binary_mask_to_shapely_polygon(
        mask, self.lat_lon_corners, keep_holes=True
    )
    self.assertIsInstance(polygon, shapely.geometry.Polygon)
    self.assertAlmostEqual(polygon.area, 50, places=5)
    bounds = polygon.bounds
    self.assertAlmostEqual(bounds[0], 0, places=5)
    self.assertAlmostEqual(bounds[1], 0, places=5)
    self.assertAlmostEqual(bounds[2], 5, places=5)
    self.assertAlmostEqual(bounds[3], 10, places=5)

  def test_multi_polygon_mask(self):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:4, 0:4] = 1
    mask[6:10, 6:10] = 1
    polygon = extraction.convert_binary_mask_to_shapely_polygon(
        mask, self.lat_lon_corners, keep_holes=True
    )
    self.assertIsInstance(polygon, shapely.geometry.MultiPolygon)
    self.assertAlmostEqual(polygon.area, 32, places=5)
    bounds = polygon.bounds
    self.assertAlmostEqual(bounds[0], 0, places=5)
    self.assertAlmostEqual(bounds[1], 0, places=5)
    self.assertAlmostEqual(bounds[2], 10, places=5)
    self.assertAlmostEqual(bounds[3], 10, places=5)

  def test_star_mask(self):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:9, 4:6] = 1
    mask[4:6, 1:9] = 1
    polygon = extraction.convert_binary_mask_to_shapely_polygon(
        mask, self.lat_lon_corners, keep_holes=True
    )
    self.assertIsInstance(polygon, shapely.geometry.Polygon)
    self.assertTrue(polygon.is_valid)
    self.assertFalse(polygon.is_empty)
    self.assertAlmostEqual(polygon.area, 28, places=5)

  def test_scaled_multi_polygon_mask(self):
    n = 10
    lat_lon_corners = np.array(
        [[0, 0], [0, n * 10], [n * 10, n * 10], [n * 10, 0]], dtype=np.float32
    )
    mask = np.zeros((n, n), dtype=np.uint8)
    mask[0 : int(n * 0.4), 0 : int(n * 0.4)] = 1
    mask[int(0.6 * n) : n, int(0.6 * n) : n] = 1

    polygon = extraction.convert_binary_mask_to_shapely_polygon(
        mask, lat_lon_corners, keep_holes=True
    )

    self.assertIsInstance(polygon, shapely.geometry.MultiPolygon)
    self.assertAlmostEqual(polygon.area, 3200, places=5)

  def test_multi_polygon_with_hole_mask(self):
    n = 10
    mask = np.zeros((n, n), dtype=np.uint8)
    mask[0 : int(n * 0.4), 0 : int(n * 0.4)] = 1
    mask[int(0.6 * n) : n, int(0.6 * n) : n] = 1
    mask[1:3, 1:3] = 0  # Create a hole in the top-left square.

    polygon = extraction.convert_binary_mask_to_shapely_polygon(
        mask, self.lat_lon_corners, keep_holes=True
    )

    self.assertIsInstance(polygon, shapely.geometry.MultiPolygon)
    self.assertAlmostEqual(polygon.area, mask.sum(), places=5)
    self.assertLen(list(polygon.geoms), 2)
    has_hole = any(p.interiors for p in polygon.geoms)
    self.assertTrue(has_hole)

  def test_square_mask(self):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1  # 6x6 square
    polygon = extraction.convert_binary_mask_to_shapely_polygon(
        mask, self.lat_lon_corners, keep_holes=True
    )
    self.assertIsInstance(polygon, shapely.geometry.Polygon)
    self.assertAlmostEqual(polygon.area, 36, places=5)

  def test_square_with_hole_mask(self):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    mask[3:5, 3:5] = 0  # 2x2 hole
    polygon = extraction.convert_binary_mask_to_shapely_polygon(
        mask, self.lat_lon_corners, keep_holes=True
    )
    self.assertIsInstance(polygon, shapely.geometry.Polygon)
    self.assertAlmostEqual(polygon.area, 32, places=5)
    self.assertLen(list(polygon.interiors), 1)


class PlumeCandidateExtractionLogicTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.lat_lon_corners = np.array([
        [0, 0],  # bl
        [0, 10],  # br
        [10, 10],  # tr
        [10, 0],  # tl
    ])
    self.cfg = test_utils._get_test_config(
        scale=10.0, model_input_size=4, stride=1
    )

  def test_get_center_of_mass_location(self):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 1
    point = extraction.get_center_of_mass_location(mask, self.lat_lon_corners)
    self.assertIsInstance(point, shapely.geometry.Point)
    self.assertAlmostEqual(point.x, 5.0, places=5)
    self.assertAlmostEqual(point.y, 5.0, places=5)

  def test_get_center_of_mass_location_empty(self):
    mask = np.zeros((10, 10), dtype=np.uint8)
    self.assertIsNone(
        extraction.get_center_of_mass_location(mask, self.lat_lon_corners)
    )

  def test_get_head_point(self):
    origin_probs = np.zeros((10, 10), dtype=np.float32)
    origin_probs[4:6, 4:6] = 0.8
    origin_probs[0, 0] = 0.5  # Smaller component

    utm_mapping = data_types.UtmGridMapping(
        utm_zone='10S',
        cell_size=1.0,
        width=10,
        height=10,
        utm_x_min=0,
        utm_y_min=0,
    )

    head_point, head_point_px = extraction.get_head_point(
        origin_probs,
        utm_mapping,
        origin_probability_threshold=0.6,
        w_off=5,
        h_off=5,
    )

    self.assertIsNotNone(head_point)
    self.assertIsNotNone(head_point_px)

  def test_get_head_point_empty(self):
    origin_probs = np.zeros((10, 10), dtype=np.float32)

    utm_mapping = data_types.UtmGridMapping(
        utm_zone='10S',
        cell_size=1.0,
        width=10,
        height=10,
        utm_x_min=0,
        utm_y_min=0,
    )

    head_point, head_point_px = extraction.get_head_point(
        origin_probs, utm_mapping, origin_probability_threshold=0.6
    )
    self.assertIsNone(head_point)
    self.assertIsNone(head_point_px)

  def test_get_head_squares(self):

    plume = data_types.PlumeCandidate(
        chunk_id=0,
        geometry=shapely.geometry.Polygon(),
        geometry_px=shapely.geometry.Polygon(),
        slot_id=0,
        metadata={},
        concentration=np.ones((10, 10), dtype=np.float32),
        binary_masks=np.ones((10, 10), dtype=np.uint8),
        origin_masks=np.ones((10, 10), dtype=np.uint8),
        mask=np.ones((10, 10), dtype=np.uint8),
        head_point=shapely.geometry.Point(0, 0),
        head_point_px=shapely.geometry.Point(5, 5),
        plume_bbox_px=(0, 0, 10, 10),
    )

    conc_sq, mask_sq = extraction.get_head_squares([plume], patch_size=3)
    self.assertIsNotNone(conc_sq)
    self.assertIsNotNone(mask_sq)
    self.assertEqual(conc_sq.shape, (1, 3, 3))
    self.assertEqual(mask_sq.shape, (1, 3, 3))

  def test_get_head_squares_even_patch_size(self):
    with self.assertRaises(ValueError):
      extraction.get_head_squares([], patch_size=2)

  def test_get_head_squares_zero_patch_size(self):
    conc_sq, mask_sq = extraction.get_head_squares([], patch_size=0)
    self.assertIsNone(conc_sq)
    self.assertIsNone(mask_sq)

  def test_get_polygon_keep_holes_false(self):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    mask[4:6, 4:6] = 0  # hole

    polygon = extraction.convert_binary_mask_to_shapely_polygon(
        mask, self.lat_lon_corners, keep_holes=False
    )
    self.assertIsInstance(polygon, shapely.geometry.Polygon)
    # The hole should be filled, so area is 36 instead of 32
    self.assertAlmostEqual(polygon.area, 36, places=5)
    self.assertEmpty(list(polygon.interiors))

  def test_get_geometry_px(self):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1

    polygon = extraction.get_geometry_px(mask, xoff=5, yoff=5)
    self.assertIsInstance(polygon, shapely.geometry.Polygon)
    bounds = polygon.bounds
    self.assertAlmostEqual(bounds[0], 7, places=5)  # minx = 2 + 5
    self.assertAlmostEqual(bounds[1], 7, places=5)  # miny = 2 + 5

  def test_get_geometry_px_empty(self):
    mask = np.zeros((10, 10), dtype=np.uint8)
    self.assertIsNone(extraction.get_geometry_px(mask, xoff=5, yoff=5))

  def test_extract_candidates_empty_slot(self):
    chunk_size = 32
    c1_data = test_utils._create_chunk_data(chunk_size, 10.0, 10)
    c1_data['binary_masks'][:] = 0.0  # Empty
    c1 = test_utils._create_chunk(c1_data, h_off=0, w_off=0)

    chunked_granule = data_types.ChunkedGranule(
        ee_asset_id='asset1',
        chunks=[c1],
        mask=np.ones((chunk_size, chunk_size, 1), dtype='uint8'),
        geotransform=(0.0, 10.0, 0.0, 40.0, 0.0, -10.0),
        utm_zone='10S',
        epsg='EPSG:32610',
        timestamp_ms=0,
    )

    class DummyMetrics:

      def __init__(self):
        self.plumes_extracted = self
        self.plumes_after_mask_check = self
        self.plumes_after_border_check = self
        self.chunks_processed = self

      def inc(self, *args, **kwargs):
        pass

    results = extraction.extract_candidates(
        chunked_granule,
        plume_probability_threshold=0.5,
        origin_probability_threshold=0.4,
        log=False,
        cfg=self.cfg,
        metrics=DummyMetrics(),
    )
    self.assertEmpty(results)


class FilterSmallComponentsTest(absltest.TestCase):

  def test_filter_small_components(self):
    mask = np.zeros((10, 10, 1), dtype=np.float32)
    # Large component (4 pixels)
    mask[1:3, 1:3, 0] = 1.0
    # Small component (1 pixel)
    mask[5, 5, 0] = 1.0

    expected_mask = np.copy(mask)
    expected_mask[5, 5, 0] = 0.0

    extraction.filter_small_components(
        mask, probability_threshold=0.5, min_pixels_per_slot=2
    )

    np.testing.assert_array_equal(mask, expected_mask)

  def test_filter_small_components_multiple_channels(self):
    mask = np.zeros((10, 10, 2), dtype=np.float32)
    # Channel 0
    # Large component (4 pixels)
    mask[1:3, 1:3, 0] = 0.8
    # Small component (1 pixel)
    mask[5, 5, 0] = 0.6

    # Channel 1
    # Large component (4 pixels)
    mask[6:8, 6:8, 1] = 0.9
    # Small component (1 pixel)
    mask[1, 8, 1] = 0.7

    expected_mask = np.copy(mask)
    expected_mask[5, 5, 0] = 0.0
    expected_mask[1, 8, 1] = 0.0

    extraction.filter_small_components(
        mask, probability_threshold=0.5, min_pixels_per_slot=2
    )

    np.testing.assert_array_equal(mask, expected_mask)

  def test_filter_small_components_no_components(self):
    mask = np.zeros((10, 10, 1), dtype=np.float32)

    expected_mask = np.copy(mask)

    extraction.filter_small_components(
        mask, probability_threshold=0.5, min_pixels_per_slot=2
    )

    np.testing.assert_array_equal(mask, expected_mask)


if __name__ == '__main__':
  absltest.main()
