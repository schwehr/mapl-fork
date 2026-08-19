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

"""Clustering algorithms wrapper."""

import abc
from typing import Type

import apache_beam as beam
import numpy as np
from sklearn import cluster as sk_cluster

Metrics = beam.metrics.Metrics


class ClusteringAlgorithm(abc.ABC):
  """Abstract base class for clustering algorithms."""

  @abc.abstractmethod
  def cluster(
      self,
      points: np.ndarray,
      head_concentration_squares: np.ndarray,
      head_mask_squares: np.ndarray,
  ) -> np.ndarray:
    """Clusters the points and returns the labels."""


class DbscanClustering(ClusteringAlgorithm):
  """Wrapper for DBSCAN clustering."""

  def __init__(self, **kwargs):
    self.model = sk_cluster.DBSCAN(**kwargs)

  def cluster(
      self,
      points: np.ndarray,
      head_concentration_squares: np.ndarray,
      head_mask_squares: np.ndarray,
  ) -> np.ndarray:
    del head_concentration_squares  # Unused.
    del head_mask_squares  # Unused.
    self.model.fit(points)
    return self.model.labels_


class MeanShiftClustering(ClusteringAlgorithm):
  """Wrapper for MeanShift clustering."""

  def __init__(self, **kwargs):
    self.model = sk_cluster.MeanShift(**kwargs)

  def cluster(
      self,
      points: np.ndarray,
      head_concentration_squares: np.ndarray,
      head_mask_squares: np.ndarray,
  ) -> np.ndarray:
    del head_concentration_squares  # Unused.
    del head_mask_squares  # Unused.
    self.model.fit(points)
    return self.model.labels_


class DistanceCorrelationDbscanClustering(ClusteringAlgorithm):
  """Manual implementation of DBSCAN clustering."""

  def __init__(
      self,
      eps_dist: float,
      eps_corr: float,
      min_samples: int,
      **unused_kwargs,
  ):
    self.eps_dist = eps_dist
    self.eps_corr = eps_corr
    self.min_samples = min_samples

  def _find_neighbors(
      self,
      idx: int,
      dists: np.ndarray,
      head_concentration_squares: np.ndarray,
      head_mask_squares: np.ndarray,
      origins: np.ndarray,
      eps_dist: float,
      eps_corr: float,
  ) -> list[int]:
    """Finds neighbors based on distance and correlation."""
    spatial_neighbors = np.where(dists[idx] <= eps_dist)[0]
    neighbors = []
    mask1 = head_mask_squares[idx]
    for neighbor_idx in spatial_neighbors:
      mask2 = head_mask_squares[neighbor_idx]
      corr = _calculate_correlation(
          head_concentration_squares[idx],
          origins[idx],
          head_concentration_squares[neighbor_idx],
          origins[neighbor_idx],
          mask1=mask1,
          mask2=mask2,
      )
      if corr > eps_corr:
        neighbors.append(neighbor_idx)
    return neighbors

  def _dbscan_predict(
      self,
      points: np.ndarray,
      head_concentration_squares: np.ndarray,
      head_mask_squares: np.ndarray,
      eps_dist: float,
      eps_corr: float,
      min_samples: int,
  ) -> np.ndarray:
    """Manual implementation of DBSCAN clustering."""
    n = points.shape[0]
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    if n == 0:
      return labels

    # Compute pairwise distances
    dists = np.linalg.norm(points[:, None] - points, axis=2)
    visited = np.zeros(n, dtype=bool)

    # Pre-compute integer origins for correlation
    # points are (x, y)
    origins = np.round(points).astype(int)

    for i in range(n):
      if visited[i]:
        continue
      visited[i] = True

      neighbors = self._find_neighbors(
          i,
          dists,
          head_concentration_squares,
          head_mask_squares,
          origins,
          eps_dist,
          eps_corr,
      )

      if len(neighbors) < min_samples:
        continue  # Noise

      labels[i] = cluster_id
      seeds = list(neighbors)
      k = 0
      while k < len(seeds):
        q = seeds[k]
        k += 1
        if q == i:
          continue

        if not visited[q]:
          visited[q] = True

          q_neighbors = self._find_neighbors(
              q,
              dists,
              head_concentration_squares,
              head_mask_squares,
              origins,
              eps_dist,
              eps_corr,
          )

          if len(q_neighbors) >= min_samples:
            seeds.extend(q_neighbors)

        if labels[q] == -1:
          labels[q] = cluster_id

      cluster_id += 1
    return labels

  def cluster(
      self,
      points: np.ndarray,
      head_concentration_squares: np.ndarray,
      head_mask_squares: np.ndarray,
  ) -> np.ndarray:
    return self._dbscan_predict(
        points,
        head_concentration_squares,
        head_mask_squares,
        self.eps_dist,
        self.eps_corr,
        self.min_samples,
    )


def _calculate_correlation(
    square1: np.ndarray,
    origin1: np.ndarray,
    square2: np.ndarray,
    origin2: np.ndarray,
    mask1: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
) -> float:
  """Calculates the correlation between two squares with offsets."""
  # Origin is (x, y), so index 0 is x (col), index 1 is y (row).
  # Offsets are calculated as (row_offset, col_offset).
  offset_x = origin2[0] - origin1[0]
  offset_y = origin2[1] - origin1[1]

  # Check if the squares overlap.
  size_y, size_x = square1.shape
  if abs(offset_x) >= size_x or abs(offset_y) >= size_y:
    return 0.0

  # Compute the overlapping slices.
  # Square 1
  s1_y_start = max(offset_y, 0)
  s1_y_end = min(size_y + offset_y, size_y)
  s1_x_start = max(offset_x, 0)
  s1_x_end = min(size_x + offset_x, size_x)
  square1_crop = square1[s1_y_start:s1_y_end, s1_x_start:s1_x_end]

  # Square 2
  s2_y_start = max(-offset_y, 0)
  s2_y_end = min(size_y - offset_y, size_y)
  s2_x_start = max(-offset_x, 0)
  s2_x_end = min(size_x - offset_x, size_x)
  square2_crop = square2[s2_y_start:s2_y_end, s2_x_start:s2_x_end]

  if square1_crop.size == 0 or square2_crop.size == 0:
    return 0.0

  # Flatten and compute correlation.
  flat1 = square1_crop.flatten()
  flat2 = square2_crop.flatten()

  # Apply masks if present
  valid_mask = np.ones(flat1.shape, dtype=bool)
  mask1_crop = mask1[s1_y_start:s1_y_end, s1_x_start:s1_x_end]  # pyrefly: ignore[unsupported-operation]
  valid_mask &= mask1_crop.flatten() > 0
  mask2_crop = mask2[s2_y_start:s2_y_end, s2_x_start:s2_x_end]  # pyrefly: ignore[unsupported-operation]
  valid_mask &= mask2_crop.flatten() > 0

  if not valid_mask.any():
    Metrics.counter('DistanceCorrelationDbscan', 'no_valid_mask').inc()
    return 0.0

  flat1 = flat1[valid_mask]
  flat2 = flat2[valid_mask]

  # We return 0 if the variance is 0 (constant square).
  if np.std(flat1) == 0 or np.std(flat2) == 0:
    return 0.0

  return np.corrcoef(flat1, flat2)[0, 1]


CLUSTERING_ALGORITHMS: dict[str, Type[ClusteringAlgorithm]] = {
    'DBSCAN': DbscanClustering,
    'DBSCANCorr': DistanceCorrelationDbscanClustering,
    'MeanShift': MeanShiftClustering,
}
