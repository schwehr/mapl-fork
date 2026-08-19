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

"""Configuration for exporting inference results to EE."""

from collections.abc import Sequence
import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExportConfig:
  """Configuration for exporting inference results to EE.

  Attributes:
    bands: The bands to export.
    ee_project: The EE project to export to.
    ee_asset_chunk_size_m: The size of the chunks to fetch from EE.
    ee_multithreading: The number of threads to use for EE fetching.
    scale: The scale of the model inputs in meters.
    num_eroded_pixels: The number of pixels to erode the mask by.
    model_input_size: The size of the model inputs in pixels.
    sensor_name: The name of the sensor (e.g., emit_l1b) used for inference.
    model_path: The path to the model to use for inference.
    model_outputs_keys: The keys of the model outputs to use for inference.
    batch_size: The batch size to use for inference.
    stride: The stride of the model in pixels.
    plume_probability_threshold: The probability threshold to use for plumes.
    origin_probability_threshold: The probability threshold to use for origins.
    regularizer: A smoothing factor added to tile weights during plume
      aggregation. It prevents division-by-zero issues and reduces noise in
      low-confidence areas by ensuring a minimum weight in the denominator when
      normalizing aggregated predictions.
    gcs_folder: The folder to upload the data to.
    era5_zarr_path: Path to the ERA5 zarr store.
    output_ee_asset_id: The EE asset ID to upload the data to.
    simplify: The simplification tolerance for the plume polygons.
    min_num_pixels_per_slot: The minimum number of pixels per plume slot.
    keep_holes: Whether to keep holes in the plume polygons.
    border_on_plume_images: The number of pixels to add to the border of plume
      images.
    cluster_alg: The clustering algorithm to use for deduping plumes.
    cluster_kw: The keyword arguments to pass to the clustering algorithm.
    cc_min_component_size: The minimum component size for the connected
      components filter.
    npz_filename: The filename or path of the NPZ file containing spectral
      statistics.
    vetting_patch_size: The size of the patch to fetch for spectral vetting in
      pixels (square patch of size `vetting_patch_size` x `vetting_patch_size`).
    use_mock_inference: Whether to use mock inference.
    resume: Whether to enable marker-based resume. Specifically, this uses empty
      marker files stored in GCS to track completion of individual assets. This
      is highly useful for recovering from task preemptions or Earth Engine
      API rate-limit timeouts without having to recompute the entire batch.
  """

  # EE fetching.
  bands: Sequence[str]
  ee_project: str
  ee_asset_chunk_size_m: float
  ee_multithreading: int
  scale: float
  num_eroded_pixels: int = 5

  # Model inference.
  sensor_name: str
  model_input_size: int
  model_path: str
  model_outputs_keys: Sequence[str]
  batch_size: int
  stride: int
  plume_probability_threshold: float = 0.4
  origin_probability_threshold: float = 0.4

  # Tiling.
  regularizer: float = 0.33

  # Export.
  gcs_folder: str
  era5_zarr_path: str = ''
  output_ee_asset_id: str

  # Plume vectorization.
  simplify: float = 1e-3
  min_num_pixels_per_slot: int = 100
  keep_holes: bool = False
  border_on_plume_images: int = 10
  cluster_alg: str = 'DBSCANCorr'
  cluster_kw: dict[str, Any] = dataclasses.field(
      default_factory=lambda: dict(
          eps_dist=25, eps_corr=0.955, min_samples=6, patch_size=65
      )
  )
  cc_min_component_size: int = 25

  # Spectral Vetting.
  npz_filename: str
  vetting_patch_size: int = 200

  # Mock mode.
  use_mock_inference: bool = False

  # Resume.
  resume: bool = True
