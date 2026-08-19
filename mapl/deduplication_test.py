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

import dataclasses
import sys
from unittest import mock

from absl.testing import absltest
from mapl import data_types
from mapl import deduplication
from mapl import test_utils
import numpy as np
import shapely.geometry


def _run_deduplication(
    cfg, chunked_granule, plumes, max_candidates_per_cluster=None
):
  if max_candidates_per_cluster is None:
    max_candidates_per_cluster = sys.maxsize
  w = np.hanning(cfg.model_input_size)
  tile_weight = np.outer(w, w)
  result = deduplication.dedupe_plumes_and_calculate_spectral_vetting_inputs(
      chunked_granule=chunked_granule,
      plumes=plumes,
      tile_weight=tile_weight,
      cluster_alg=cfg.cluster_alg,
      cluster_kw=cfg.cluster_kw,
      plume_probability_threshold=0.5,
      origin_probability_threshold=0.4,
      regularizer=cfg.regularizer,
      scale=cfg.scale,
      simplify=cfg.simplify,
      keep_holes=cfg.keep_holes,
      cc_min_component_size=cfg.cc_min_component_size,
      border_on_plume_images=cfg.border_on_plume_images,
      vetting_patch_size=cfg.vetting_patch_size,
      max_candidates_per_cluster=max_candidates_per_cluster,
  )
  return result if result is not None else []


class DedupePlumesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = test_utils._get_test_config(
        scale=10.0, model_input_size=3, stride=1
    )

  def test_dedupe_plumes_multiple_physical(self):
    chunk_size = 20
    # Chunk 1: (0,0) to (20,20). Has Plume A at (15,5).
    c1_data = test_utils._create_chunk_data(
        chunk_size, 10.0, 10.0, origin_pixel=(15, 5)
    )
    c1 = test_utils._create_chunk(c1_data, h_off=0, w_off=0)

    # Chunk 2: (0,10) to (20,30). Overlaps.
    # Has Plume A at local (5,5) which is global (15,5).
    # Has Plume B at local (15,15) which is global (25,15).
    c2_data = test_utils._create_chunk_data(
        chunk_size, 20.0, 20.0, origin_pixel=(5, 5)
    )
    c2 = test_utils._create_chunk(c2_data, h_off=0, w_off=10)

    # Candidate A1 in Chunk 1.
    plumes = [
        test_utils._create_plume_candidate(
            0, c1, shapely.geometry.Point(15, 5), (15, 5), (0, 0, 20, 20)
        ),
        # Candidate A2 in Chunk 2.
        test_utils._create_plume_candidate(
            1, c2, shapely.geometry.Point(15, 5), (15, 5), (10, 0, 30, 20)
        ),
        # Candidate B1 in Chunk 2.
        test_utils._create_plume_candidate(
            2, c2, shapely.geometry.Point(25, 15), (25, 15), (10, 0, 30, 20)
        ),
    ]

    chunks = [c1, c2]
    chunked_granule = data_types.ChunkedGranule(
        ee_asset_id='asset1',
        chunks=chunks,
        mask=np.ones((30, 30, 1), dtype='uint8'),
        geotransform=(0.0, 10.0, 0.0, 300.0, 0.0, -10.0),
        utm_zone='10S',
        epsg='EPSG:32610',
        timestamp_ms=0,
    )

    cfg = dataclasses.replace(
        self.cfg, cluster_kw={'eps': 5, 'min_samples': 1}, model_input_size=20
    )
    result_plumes = _run_deduplication(cfg, chunked_granule, plumes)

    self.assertIsInstance(result_plumes, list)
    self.assertLen(result_plumes, 2)  # 2 groups after deduping
    self.assertCountEqual([p.cluster_size for p in result_plumes], [1, 2])

  def test_dedupe_plumes_with_noise(self):
    chunk_size = 3
    c1_data = test_utils._create_chunk_data(chunk_size, 1.0, 10)
    c1 = test_utils._create_chunk(c1_data, h_off=0, w_off=0)
    c2_data = test_utils._create_chunk_data(chunk_size, 1.0, 20)
    c2 = test_utils._create_chunk(c2_data, h_off=10, w_off=10)

    chunks = [c1, c2]
    chunked_granule = data_types.ChunkedGranule(
        ee_asset_id='asset1',
        chunks=chunks,
        mask=np.ones((13, 13, 1), dtype='uint8'),
        geotransform=(
            0.0,
            10.0,
            0.0,
            130.0,
            0.0,
            -10.0,
        ),  # 10m scale, top-left (0,130)
        utm_zone='10S',
        epsg='EPSG:32610',
        timestamp_ms=0,
    )

    plumes = [
        test_utils._create_plume_candidate(
            0,
            c1,
            shapely.geometry.Point(5, 125),
            (0, 0),
            (0, 0, 3, 3),
        ),
        test_utils._create_plume_candidate(
            1,
            c2,
            shapely.geometry.Point(105, 25),
            (10, 10),
            (10, 10, 13, 13),
        ),
    ]

    cfg = dataclasses.replace(
        self.cfg, cluster_kw={'eps': 1.5, 'min_samples': 2}
    )
    result_plumes = _run_deduplication(cfg, chunked_granule, plumes)

    self.assertEmpty(result_plumes)

  def test_dedupe_plumes_mosaicking_with_invalid_pixels(self):
    chunk_size = 3
    self.cfg = dataclasses.replace(self.cfg, model_input_size=chunk_size)
    # Chunk 1 at (0,0)
    c1_data = test_utils._create_chunk_data(chunk_size, 1.0, 10)
    c1 = test_utils._create_chunk(c1_data, h_off=0, w_off=0)
    # Chunk 2 at (0,1)
    c2_data = test_utils._create_chunk_data(chunk_size, 2.0, 20)
    c2_mask = np.ones((chunk_size, chunk_size, 1), dtype='uint8')
    # This pixel (1,0) in chunk 2 overlaps with pixel (1,1) in chunk 1.
    # We set it to invalid in chunk 2.
    c2_mask[1, 0, 0] = 0
    c2 = test_utils._create_chunk(c2_data, h_off=0, w_off=1, mask=c2_mask)

    chunks = [c1, c2]
    chunked_granule = data_types.ChunkedGranule(
        ee_asset_id='asset1',
        chunks=chunks,
        mask=np.ones((3, 4, 1), dtype='uint8'),
        geotransform=(0.0, 10.0, 0.0, 30.0, 0.0, -10.0),
        utm_zone='10S',
        epsg='EPSG:32610',
        timestamp_ms=0,
    )

    plumes = [
        test_utils._create_plume_candidate(
            0,
            c1,
            shapely.geometry.Point(15, 15),
            (1, 1),
            (0, 0, 3, 3),
        ),
        test_utils._create_plume_candidate(
            1,
            c2,
            shapely.geometry.Point(25, 15),
            (2, 1),
            (1, 0, 4, 3),
        ),
    ]

    result_plumes = _run_deduplication(self.cfg, chunked_granule, plumes)

    self.assertLen(result_plumes, 1)
    deduped_plume = result_plumes[0]

    self.assertEqual(deduped_plume.raster.shape, (1, 2, 3))
    self.assertAlmostEqual(
        deduped_plume.raster[0, 0, 0],
        10 / (1 + self.cfg.regularizer),
        places=4,
    )

    self.assertEqual(deduped_plume.binary_mask.shape, (1, 2))
    self.assertEqual(deduped_plume.binary_mask[0, 0], 1)
    self.assertEqual(deduped_plume.binary_mask[0, 1], 1)

  def test_dedupe_plumes_geometry_px(self):
    chunk_size = 4
    c1_data = test_utils._create_chunk_data(
        chunk_size, 10.0, 10, origin_pixel=(1, 1)
    )
    c1_data['binary_masks'][:] = 0.0
    c1_data['binary_masks'][1, 1, 0] = 1.0
    c1 = test_utils._create_chunk(c1_data, h_off=0, w_off=0)

    chunked_granule = data_types.ChunkedGranule(
        ee_asset_id='asset1',
        chunks=[c1],
        mask=np.ones((4, 4, 1), dtype='uint8'),
        geotransform=(
            0.0,
            10.0,
            0.0,
            40.0,
            0.0,
            -10.0,
        ),  # 10m scale, top-left (0,40)
        utm_zone='10S',
        epsg='EPSG:32610',
        timestamp_ms=0,
    )

    plumes = [
        test_utils._create_plume_candidate(
            0,
            c1,
            shapely.geometry.Point(15, 25),
            (1, 1),
            (0, 0, 4, 4),
        )
    ]

    cfg = dataclasses.replace(self.cfg, model_input_size=4)
    result_plumes = _run_deduplication(cfg, chunked_granule, plumes)

    self.assertLen(result_plumes, 1)
    deduped_plume = result_plumes[0]

    self.assertIsInstance(deduped_plume.geometry_px, shapely.geometry.Polygon)
    minx, miny, maxx, maxy = deduped_plume.geometry_px.bounds
    self.assertAlmostEqual(minx, 1.0)
    self.assertAlmostEqual(miny, 1.0)
    self.assertAlmostEqual(maxx, 2.0)
    self.assertAlmostEqual(maxy, 2.0)

    self.assertEqual(
        deduped_plume.head_point_px, shapely.geometry.Point(1.0, 1.0)
    )

  def test_cluster_plumes_passes_correct_patches(self):
    mock_clustering_instance = mock.Mock()
    mock_clustering_instance.cluster.return_value = [0]

    mock_clustering_cls = mock.Mock(return_value=mock_clustering_instance)

    with mock.patch.dict(
        deduplication.clustering.CLUSTERING_ALGORITHMS,
        {'MockClustering': mock_clustering_cls},
    ):
      concentration = np.arange(100 * 100).reshape(100, 100).astype(np.float32)
      chunk_data = test_utils._create_chunk_data(100, 0, 0)
      chunk = test_utils._create_chunk(chunk_data, h_off=0, w_off=0)

      plume = test_utils._create_plume_candidate(
          chunk_id=0,
          chunk=chunk,
          head_point_point=shapely.geometry.Point(0, 0),
          head_point_px=(50.0, 50.0),
          plume_bbox_px=(0, 0, 100, 100),
      )

      plume.concentration = concentration

      # We test cluster_plumes directly here.
      deduplication.cluster_plumes(
          [plume], cluster_alg='MockClustering', cluster_kw={'patch_size': 31}
      )

      mock_clustering_instance.cluster.assert_called_once()
      call_args = mock_clustering_instance.cluster.call_args
      _, kwargs = call_args
      head_squares = kwargs['head_concentration_squares']

      self.assertEqual(head_squares.shape, (1, 31, 31))

      expected_patch = concentration[35:66, 35:66]
      np.testing.assert_array_equal(head_squares[0], expected_patch)

  def test_cluster_plumes_passes_correct_patches_boundary(self):
    mock_clustering_instance = mock.Mock()
    mock_clustering_instance.cluster.return_value = [0]

    mock_clustering_cls = mock.Mock(return_value=mock_clustering_instance)

    with mock.patch.dict(
        deduplication.clustering.CLUSTERING_ALGORITHMS,
        {'MockClustering': mock_clustering_cls},
    ):
      concentration = np.arange(100 * 100).reshape(100, 100).astype(np.float32)

      chunk_data = test_utils._create_chunk_data(100, 0, 0)
      chunk = test_utils._create_chunk(chunk_data, h_off=0, w_off=0)

      plume = test_utils._create_plume_candidate(
          chunk_id=0,
          chunk=chunk,
          head_point_point=shapely.geometry.Point(0, 0),
          head_point_px=(0.0, 0.0),
          plume_bbox_px=(0, 0, 100, 100),
      )
      plume.concentration = concentration

      deduplication.cluster_plumes(
          [plume], cluster_alg='MockClustering', cluster_kw={'patch_size': 31}
      )

      mock_clustering_instance.cluster.assert_called_once()
      call_args = mock_clustering_instance.cluster.call_args
      _, kwargs = call_args
      head_squares = kwargs['head_concentration_squares']

      self.assertEqual(head_squares.shape, (1, 31, 31))

      patch = head_squares[0]
      self.assertEqual(patch[15, 15], concentration[0, 0])
      self.assertTrue(np.all(patch[0:15, 0:15] == 0))
      np.testing.assert_array_equal(patch[15:, 15:], concentration[0:16, 0:16])

  def test_dedupe_plumes_complex_vetting_overlapping_clusters(self):
    chunk_size = 20
    c1_data = test_utils._create_chunk_data(chunk_size, 10.0, 10)

    x = np.arange(chunk_size)
    y = np.arange(chunk_size)
    xv, yv = np.meshgrid(x, y)
    c1_data['radiance'] = (10.0 + xv + yv).astype(np.float32)[..., np.newaxis]
    c1_data['to_sun_zenith'] = (1.0 + 0.1 * xv).astype(np.float32)[
        ..., np.newaxis
    ]
    c1_data['to_sensor_zenith'] = (1.0 + 0.1 * yv).astype(np.float32)[
        ..., np.newaxis
    ]

    c1 = test_utils._create_chunk(c1_data, h_off=0, w_off=0)

    chunked_granule = data_types.ChunkedGranule(
        ee_asset_id='asset1',
        chunks=[c1],
        mask=np.ones((20, 20, 1), dtype='uint8'),
        geotransform=(0.0, 1.0, 0.0, 20.0, 0.0, -1.0),
        utm_zone='10S',
        epsg='EPSG:32610',
        timestamp_ms=0,
    )

    plume_a = test_utils._create_plume_candidate(
        0, c1, shapely.geometry.Point(4, 16), (4, 4), (0, 0, 20, 20)
    )
    plume_a.binary_masks = np.zeros((20, 20), dtype=np.float32)
    plume_a.binary_masks[2:12, 2:12] = 1.0
    plume_a.origin_masks = np.zeros((20, 20), dtype=np.float32)
    plume_a.origin_masks[4, 4] = 1.0

    plume_b = test_utils._create_plume_candidate(
        0, c1, shapely.geometry.Point(16, 4), (16, 16), (0, 0, 20, 20)
    )
    plume_b.binary_masks = np.zeros((20, 20), dtype=np.float32)
    plume_b.binary_masks[8:18, 8:18] = 1.0
    plume_b.origin_masks = np.zeros((20, 20), dtype=np.float32)
    plume_b.origin_masks[16, 16] = 1.0

    cfg = dataclasses.replace(
        self.cfg,
        cluster_kw={'eps': 5, 'min_samples': 1},
        model_input_size=20,
    )

    result_plumes = _run_deduplication(cfg, chunked_granule, [plume_a, plume_b])

    self.assertLen(result_plumes, 2)

  def test_dedupe_plumes_preserves_origin_component(self):
    chunk_size = 10
    cfg = dataclasses.replace(
        self.cfg, model_input_size=chunk_size, cc_min_component_size=3
    )

    c1_data = test_utils._create_chunk_data(
        chunk_size, 1.0, 10, origin_pixel=None
    )
    c1 = test_utils._create_chunk(c1_data, h_off=0, w_off=0)

    chunked_granule = data_types.ChunkedGranule(
        ee_asset_id='asset1',
        chunks=[c1],
        mask=np.ones((10, 10, 1), dtype='uint8'),
        geotransform=(0.0, 1.0, 0.0, 10.0, 0.0, -1.0),
        utm_zone='10S',
        epsg='EPSG:32610',
        timestamp_ms=0,
    )

    plume = test_utils._create_plume_candidate(
        0, c1, shapely.geometry.Point(2, 8), (2, 2), (0, 0, 10, 10)
    )

    # Component 1 (with origin at 2,2): size 2
    # Component 2 (no origin, at 7,7): size 2
    plume.binary_masks = np.zeros((10, 10), dtype=np.float32)
    plume.binary_masks[2, 2] = 1.0
    plume.binary_masks[2, 3] = 1.0
    plume.binary_masks[7, 7] = 1.0
    plume.binary_masks[7, 8] = 1.0

    plume.origin_masks = np.zeros((10, 10), dtype=np.float32)
    plume.origin_masks[2, 2] = 1.0

    result_plumes = _run_deduplication(cfg, chunked_granule, [plume])

    self.assertLen(result_plumes, 1)
    deduped_plume = result_plumes[0]

    self.assertIsInstance(deduped_plume.geometry_px, shapely.geometry.Polygon)
    minx, miny, maxx, maxy = deduped_plume.geometry_px.bounds
    self.assertAlmostEqual(minx, 2.0)
    self.assertAlmostEqual(miny, 2.0)
    self.assertAlmostEqual(maxx, 4.0)
    self.assertAlmostEqual(maxy, 3.0)

  def test_dedupe_plumes_caps_candidates(self):
    # Intent: Verify that candidate capping limits the number of candidates
    # used in ensembling to the top N sorted by footprint size, and that
    # without capping all candidates are used.
    # We use 4 candidates and cap to 2 to show a difference from the uncapped
    # case which uses all 4.
    chunk_size = 10
    cfg = dataclasses.replace(self.cfg, model_input_size=chunk_size)

    # Setup a dummy chunk.
    c1_data = test_utils._create_chunk_data(
        chunk_size, 1.0, 10, origin_pixel=None
    )
    c1 = test_utils._create_chunk(c1_data, h_off=0, w_off=0)

    chunked_granule = data_types.ChunkedGranule(
        ee_asset_id='asset1',
        chunks=[c1],
        mask=np.ones((10, 10, 1), dtype='uint8'),
        geotransform=(0.0, 1.0, 0.0, 10.0, 0.0, -1.0),
        utm_zone='10S',
        epsg='EPSG:32610',
        timestamp_ms=0,
    )

    # Create 4 candidates with overlapping but different length footprints.
    # All share the same origin at (2,2) and will be clustered together.
    # p1 (largest): size 8 (cols 2-9)
    p1 = test_utils._create_plume_candidate(
        0, c1, shapely.geometry.Point(2, 8), (2, 2), (0, 0, 10, 10)
    )
    p1.binary_masks = np.zeros((10, 10), dtype=np.float32)
    p1.binary_masks[2, 2:10] = 1.0
    p1.origin_masks = np.zeros((10, 10), dtype=np.float32)
    p1.origin_masks[2, 2] = 1.0

    # p2 (medium-large): size 6 (cols 2-7)
    p2 = test_utils._create_plume_candidate(
        0, c1, shapely.geometry.Point(2, 8), (2, 2), (0, 0, 10, 10)
    )
    p2.binary_masks = np.zeros((10, 10), dtype=np.float32)
    p2.binary_masks[2, 2:8] = 1.0
    p2.origin_masks = np.zeros((10, 10), dtype=np.float32)
    p2.origin_masks[2, 2] = 1.0

    # p3 (medium-small): size 4 (cols 2-5)
    p3 = test_utils._create_plume_candidate(
        0, c1, shapely.geometry.Point(2, 8), (2, 2), (0, 0, 10, 10)
    )
    p3.binary_masks = np.zeros((10, 10), dtype=np.float32)
    p3.binary_masks[2, 2:6] = 1.0
    p3.origin_masks = np.zeros((10, 10), dtype=np.float32)
    p3.origin_masks[2, 2] = 1.0

    # p4 (smallest): size 2 (cols 2-3)
    p4 = test_utils._create_plume_candidate(
        0, c1, shapely.geometry.Point(2, 8), (2, 2), (0, 0, 10, 10)
    )
    p4.binary_masks = np.zeros((10, 10), dtype=np.float32)
    p4.binary_masks[2, 2:4] = 1.0
    p4.origin_masks = np.zeros((10, 10), dtype=np.float32)
    p4.origin_masks[2, 2] = 1.0

    # Scenario 1: Cap candidates to 2.
    # We mock clustering to group all 4 candidates.
    # Capping should keep only the two largest candidates (p1 and p2)
    # and discard p3 and p4.
    with mock.patch.object(
        deduplication, 'cluster_plumes', return_value=[0, 0, 0, 0]
    ):
      result_plumes = _run_deduplication(
          cfg, chunked_granule, [p1, p2, p3, p4], max_candidates_per_cluster=2
      )

    self.assertLen(result_plumes, 1)
    deduped_plume = result_plumes[0]

    self.assertIsInstance(
        deduped_plume.geometry_px,
        (shapely.geometry.Polygon, shapely.geometry.MultiPolygon),
    )
    # Expected outcome for Cap=2:
    # Only p1 (cols 2-9) and p2 (cols 2-7) are used.
    # Ensembling requires >50% vote.
    # - Cols 2-7: 2/2 candidates agree (100% vote) -> Preserved.
    # - Cols 8-9: 1/2 candidates agree (p1 only)
    #   (50% vote -> <50% with reg) -> Discarded.
    # So the resulting plume should cover cols 2-7, meaning max_x = 8.0.
    minx, miny, maxx, maxy = deduped_plume.geometry_px.bounds
    self.assertAlmostEqual(minx, 2.0)
    self.assertAlmostEqual(miny, 2.0)
    self.assertAlmostEqual(maxx, 8.0)
    self.assertAlmostEqual(maxy, 3.0)

    # Scenario 2: Run without capping (max_candidates_per_cluster=None).
    # All 4 candidates should be used in ensembling.
    # The final plume is determined by majority vote (>50% weight).
    # - Cols 2-3: 4/4 candidates agree (100% vote) -> Preserved.
    # - Cols 4-5: 3/4 candidates agree (p1, p2, p3) (75% vote) -> Preserved.
    # - Cols 6-7: 2/4 candidates agree (p1, p2)
    #   (50% vote -> <50% with reg) -> Discarded.
    # - Cols 8-9: 1/4 candidates agree (p1 only) (25% vote) -> Discarded.
    # So the resulting plume should cover cols 2-5, meaning max_x = 6.0.
    with mock.patch.object(
        deduplication, 'cluster_plumes', return_value=[0, 0, 0, 0]
    ):
      result_plumes_no_cap = _run_deduplication(
          cfg,
          chunked_granule,
          [p1, p2, p3, p4],
          max_candidates_per_cluster=None,
      )

    self.assertLen(result_plumes_no_cap, 1)
    deduped_plume_no_cap = result_plumes_no_cap[0]

    self.assertIsInstance(
        deduped_plume_no_cap.geometry_px,
        (shapely.geometry.Polygon, shapely.geometry.MultiPolygon),
    )
    minx, miny, maxx, maxy = deduped_plume_no_cap.geometry_px.bounds
    self.assertAlmostEqual(minx, 2.0)
    self.assertAlmostEqual(miny, 2.0)
    self.assertAlmostEqual(maxx, 6.0)
    self.assertAlmostEqual(maxy, 3.0)


class CalculateExclusionMaskTest(absltest.TestCase):

  def test_calculate_exclusion_mask_no_overlap(self):
    plume = mock.MagicMock(spec=data_types.Plume)
    plume.head_point_px = shapely.geometry.Point(50, 50)

    candidate = mock.MagicMock(spec=data_types.PlumeCandidate)
    candidate.plume_bbox_px = (0, 0, 10, 10)
    candidate.binary_masks = np.ones((10, 10))
    candidate.mask = np.ones((10, 10))

    granule_mask = np.ones((100, 100), dtype='uint8')

    # Test pure logic function directly
    exclusion_mask, validity_mask = deduplication._calculate_exclusion_mask(
        plume=plume,
        candidates=[candidate],
        plume_probability_threshold=0.5,
        vetting_patch_size=20,
        granule_mask=granule_mask,
    )

    np.testing.assert_array_equal(
        exclusion_mask, np.zeros((20, 20), dtype=bool)
    )
    np.testing.assert_array_equal(validity_mask, np.ones((20, 20), dtype=bool))

  def test_calculate_exclusion_mask_with_overlap(self):
    plume = mock.MagicMock(spec=data_types.Plume)
    plume.head_point_px = shapely.geometry.Point(50, 50)

    candidate = mock.MagicMock(spec=data_types.PlumeCandidate)
    candidate.plume_bbox_px = (30, 30, 45, 45)
    candidate.binary_masks = np.ones((15, 15))
    candidate.mask = np.ones((15, 15))

    granule_mask = np.ones((100, 100), dtype='uint8')

    exclusion_mask, _ = deduplication._calculate_exclusion_mask(
        plume=plume,
        candidates=[candidate],
        plume_probability_threshold=0.5,
        vetting_patch_size=20,
        granule_mask=granule_mask,
    )

    expected_exclusion = np.zeros((20, 20), dtype=bool)
    expected_exclusion[0:5, 0:5] = True

    np.testing.assert_array_equal(exclusion_mask, expected_exclusion)


if __name__ == '__main__':
  absltest.main()
