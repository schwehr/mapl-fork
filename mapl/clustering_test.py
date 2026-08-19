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
from mapl import clustering
import numpy as np


class DistanceCorrelationDbscanClusteringTest(absltest.TestCase):

  def test_dist_corr_dbscan_clustering(self):
    points = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
    # eps_dist=2 should group (0,0)-(0,1) and (10,10)-(10,11).
    # eps_corr=-2.0 ensures all correlations pass.
    # Dummy head concentration squares.
    squares = np.zeros((4, 5, 5))
    masks = np.ones_like(squares)
    manual_dbscan = clustering.DistanceCorrelationDbscanClustering(
        eps_dist=2, eps_corr=-2.0, min_samples=2
    )
    labels = manual_dbscan.cluster(
        points, head_concentration_squares=squares, head_mask_squares=masks
    )
    self.assertEqual(labels[0], labels[1])
    self.assertEqual(labels[2], labels[3])
    self.assertNotEqual(labels[0], labels[2])

  def test_dist_corr_dbscan_clustering_correlation(self):
    points = np.array([[0, 0], [0, 1]])
    # Points are close.
    # Create squares that are perfectly anti-correlated.
    s1 = np.array([[1, 2], [3, 4]])
    s2 = -s1
    # Pad to 5x5
    squares = np.zeros((2, 5, 5))
    squares[0, 1:3, 1:3] = s1
    squares[1, 1:3, 1:3] = s2
    masks = np.ones_like(squares)

    # eps_dist=2 covers the distance.
    # eps_corr=0.5 should filter out the anti-correlated pair (-1.0).
    manual_dbscan = clustering.DistanceCorrelationDbscanClustering(
        eps_dist=2, eps_corr=0.5, min_samples=2
    )
    labels = manual_dbscan.cluster(
        points, head_concentration_squares=squares, head_mask_squares=masks
    )
    # Should be noise (-1) because min_samples=2 and they don't form a cluster.
    self.assertEqual(labels[0], -1)
    self.assertEqual(labels[1], -1)

    # With eps_corr=-2.0, they should cluster.
    manual_dbscan = clustering.DistanceCorrelationDbscanClustering(
        eps_dist=2, eps_corr=-2.0, min_samples=2
    )
    labels = manual_dbscan.cluster(
        points, head_concentration_squares=squares, head_mask_squares=masks
    )
    self.assertEqual(labels[0], 0)
    self.assertEqual(labels[1], 0)

  def test_dist_corr_dbscan_clustering_complex_no_corr(self):
    # Cluster 1: around (0,0) with some spread
    c1 = np.array([
        [0.1, 0.1],
        [-0.1, 0.0],
        [0.0, 0.2],
        [0.2, -0.1],
        [-0.2, 0.1],
        [0.0, 0.0],
    ])
    # Cluster 2: around (10, 10) with some spread
    c2 = np.array([
        [10.1, 10.1],
        [9.9, 10.0],
        [10.0, 10.2],
        [10.2, 9.9],
        [9.8, 10.1],
        [10.0, 10.0],
        [9.9, 9.9],
    ])
    # Cluster 3: around (20, 0) with some spread
    c3 = np.array([
        [20.1, 0.1],
        [19.9, 0.0],
        [20.0, 0.2],
        [20.2, -0.1],
        [19.8, 0.1],
        [20.0, 0.0],
    ])
    # Outliers
    outliers = np.array([[5.0, 5.0], [15.0, 5.0]])

    points = np.vstack([c1, c2, c3, outliers])
    # Dummy squares
    squares = np.zeros((len(points), 5, 5))
    masks = np.ones_like(squares)

    manual_dbscan = clustering.DistanceCorrelationDbscanClustering(
        eps_dist=2.0, eps_corr=-2.0, min_samples=3
    )
    labels = manual_dbscan.cluster(
        points, head_concentration_squares=squares, head_mask_squares=masks
    )

    # Check cluster counts
    unique_labels = set(labels)
    # Should have 3 clusters (0, 1, 2) and noise (-1)
    self.assertIn(-1, unique_labels)
    self.assertLen(unique_labels, 4)  # 3 clusters + noise

    # Check sizes
    # Outliers should be noise
    self.assertEqual(labels[-2], -1)
    self.assertEqual(labels[-1], -1)

    # First 6 should be same cluster
    self.assertLen(set(labels[:6]), 1)
    # Next 7 should be same cluster
    self.assertLen(set(labels[6:13]), 1)
    # Next 6 should be same cluster
    self.assertLen(set(labels[13:19]), 1)

  def test_dist_corr_dbscan_clustering_overlapping_with_corr(self):
    # Cluster 1: Spatially at (0,0), Signal A
    c1_locs = np.array([
        [0.01, 0.01],
        [-0.01, 0.0],
        [0.0, 0.02],
        [0.02, -0.01],
        [0.0, 0.0],
    ])
    # Cluster 2: Spatially at (0,0) (overlapping), Signal B (anti-correlated
    # to A).
    c2_locs = np.array([
        [-0.01, -0.01],
        [0.01, 0.0],
        [0.0, -0.02],
        [-0.02, 0.01],
        [0.01, 0.01],
    ])
    # Cluster 3: Spatially at (10,10), Signal A
    c3_locs = np.array([
        [10.01, 10.01],
        [9.99, 10.0],
        [10.0, 10.02],
        [10.02, 9.99],
        [10.0, 10.0],
    ])

    points = np.vstack([c1_locs, c2_locs, c3_locs])

    # Signal A
    # Use a deterministic pattern
    sig_a = np.array([
        [1.0, 2.0, 1.0, 0.0, 0.0],
        [2.0, 3.0, 2.0, 0.0, 0.0],
        [1.0, 2.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    # Signal B (anti-correlated)
    sig_b = -sig_a

    squares = np.zeros((15, 5, 5))
    squares[:5] = sig_a
    squares[5:10] = sig_b
    squares[10:] = sig_a
    masks = np.ones_like(squares)

    # clustering with high correlation threshold
    # eps_dist=1.0 covers the overlap at (0,0)
    manual_dbscan = clustering.DistanceCorrelationDbscanClustering(
        eps_dist=1.0, eps_corr=0.9, min_samples=3
    )
    labels = manual_dbscan.cluster(
        points, head_concentration_squares=squares, head_mask_squares=masks
    )

    # Should separate c1 and c2 despite spatial overlap
    self.assertLen(set(labels[:5]), 1)  # C1
    self.assertLen(set(labels[5:10]), 1)  # C2
    self.assertLen(set(labels[10:]), 1)  # C3

    # C1 and C2 should be different
    self.assertNotEqual(labels[0], labels[5])


if __name__ == '__main__':
  absltest.main()
