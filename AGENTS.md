# Agent Guidelines for MAPL Repository

## Project Overview & Architecture

MAPL (Methane Analysis and Plume Localization) is a scientific library and inference pipeline developed for detecting, localizing, analyzing, and vetting greenhouse gas emissions—specifically methane ($CH_4$) plumes—from airborne and spaceborne hyperspectral imaging spectrometers (primarily NASA's Earth Surface Mineral Dust Source Investigation, EMIT, on the International Space Station).

### End-to-End Processing Pipeline

1. **Granule Ingestion & Orthorectification (`mapl.netcdf_io`)**:
   - Reads unprojected EMIT L1B Radiance (RAD) and Observation (OBS) NetCDF datasets.
   - Utilizes Geographic Lookup Tables (GLT) to perform orthorectification into UTM grid coordinates via `xarray` and `netCDF4`.
   - Extracts radiance cubes, solar and sensor zenith angles, crosstrack detector indices, geotransforms, and validity masks.

2. **Tiled Model Inference (`mapl.granule_inference`, `mapl.single_granule_inference`)**:
   - Cuts large hyperspectral scenes into overlapping patches (e.g., $256 \times 256$ pixels with configurable stride such as 64 or 32).
   - Executes batched neural network inference via TensorFlow SavedModel (`infer_fn`), producing multi-slot predictions: `concentration` (enhancement path length in $ppm\cdot m$), `binary_masks` (plume detection probability), and `origin_masks` (plume origin/head probability).
   - Reconstructs a seamless granule-wide enhancement image via weighted border averaging (e.g., L2 distance weighting or 2D Hanning/Tukey windowing).

3. **Plume Candidate Extraction (`mapl.plume_candidate_extraction`)**:
   - Evaluates prediction slots across all granule chunks against probability thresholds (`plume_probability_threshold`, `origin_probability_threshold`).
   - Converts raster binary masks into vector geometries (Shapely `Polygon` and `MultiPolygon` in both pixel and geographic CRS) using `rasterio.features.shapes`.
   - Localizes plume source/head locations by finding the center of mass of the largest connected component in the origin probability mask.

4. **Deduplication, Clustering & Spatial Ensembling (`mapl.deduplication`, `mapl.clustering`)**:
   - Resolves overlapping candidate detections from adjacent sliding windows and multi-slot predictions.
   - Groups candidates using specialized clustering algorithms:
     - `DBSCAN`: Euclidean distance clustering on head coordinates.
     - `DBSCANCorr` (`DistanceCorrelationDbscanClustering`): Hybrid clustering combining spatial distance with Pearson correlation of cropped head concentration patches.
     - `MeanShift`: Kernel density estimation for mode seeking.
   - Caps candidates per cluster to the theoretical maximum footprint limit (keeping top $N$ by area) to prevent confidence dilution during ensembling.
   - Computes regularized weighted averages over cluster bounding boxes to produce unified plume probability, concentration, and origin rasters.
   - Filters small disconnected components while protecting the connected component containing the plume origin point.

5. **Physical Spectral Vetting (`mapl.spectral_matching`)**:
   - Validates candidate plumes against physical radiative transfer principles to eliminate false positives.
   - Employs precomputed spectral response functions and standard atmospheric transmittances from bundled `.npz` statistics (`satellite_gas_statistics_v14_emit.npz`).
   - Matches plume spectra (top 30 pixels by enhancement) with background pixels across non-absorbing bands using linear sum assignment (Hungarian algorithm).
   - Fits a combined physical transmittance model ($T_{model} = \text{Polynomial Baseline} \times \text{Gas Transmission}(conc)$) across methane absorption bands ($2100\text{--}2440\text{ nm}$) using non-linear least squares (`scipy.optimize.least_squares` with soft L1 loss).
   - Calculates vetting metrics: normalized mean absolute difference (`d_norm`), spectral correlation distance (`d_cor`), and retrieved concentration (`fitted_conc`).

6. **Emission Quantification (`mapl.ime`)**:
   - Computes Integrated Methane Enhancement (IME) and derives emission rates ($kg/h$) using the `ddeq` library and wind vectors (e.g., ERA5 or HRRR 10m wind fields).

7. **Serialization & Export (`mapl.io_lib`)**:
   - Exports structured candidate and predicted plume datasets to Apache Parquet format using PyArrow schemas with WKT geometries, spectral metrics, and emission statistics.

---

## Repository & File Layout

```
mapl-fork/
├── AGENTS.md                                # Repository instructions and architectural guide for agents
├── CHANGELOG.md                             # Release notes and version history
├── CONTRIBUTING.md                          # Contribution guidelines
├── LICENSE                                  # Apache 2.0 license
├── README.md                                # Project documentation, installation, and quickstart
├── pyproject.toml                           # Build configuration, dependencies, pytest, and linting settings
├── uv.lock                                  # Pinned dependency lockfile
├── .python-version                          # Target Python version (3.13)
├── .pre-commit-config.yaml                  # Pre-commit hook configurations (pyink, pylint, ruff, etc.)
│
├── mapl/                                    # Core library source code
│   ├── __init__.py                          # Package root exposing __version__
│   ├── config.py                            # ExportConfig dataclass containing pipeline hyperparameters
│   ├── data_types.py                        # Core dataclasses (Granule, GranuleChunk, ChunkedGranule,
│   │                                        #   PlumeCandidate, Plume, PlumeGroupData, UtmGridMapping, etc.)
│   ├── wavelengths.py                       # EMIT L1B spectral band center wavelengths and bandwidths
│   ├── netcdf_io.py                         # EMIT L1B RAD/OBS NetCDF ingestion and GLT orthorectification
│   ├── clustering.py                        # Clustering algorithms (DBSCAN, DBSCANCorr, MeanShift)
│   ├── deduplication.py                     # Candidate clustering, aggregation, ensembling, and deduplication
│   ├── plume_candidate_extraction.py        # Mask vectorization, center-of-mass origin localization, filtering
│   ├── spectral_matching.py                 # SpectralVetting class, transmittance modeling, and least squares fitting
│   ├── satellite_gas_statistics_v14_emit.npz# Bundled spectral response and transmittance lookup tables
│   ├── ime.py                               # Integrated Methane Enhancement (IME) emission rate calculations
│   ├── granule_inference.py                 # MaplEmitInference pipeline coordinator for full granule processing
│   ├── single_granule_inference.py          # Standalone CLI entrypoint for single granule batch inference
│   ├── io_lib.py                            # PyArrow schema definitions and Parquet serialization helpers
│   ├── configs/                             # Package configuration modules
│   │   └── public/
│   ├── data/                                # Data utilities package
│   └── stats/                               # Statistical utilities package
│
└── tests/                                   # Test suite (absltest & pytest)
    ├── __init__.py                          # Tests package initializer
    ├── test_utils.py                        # Shared test fixtures, mock data generators, and test config factories
    ├── clustering_test.py                   # Tests for spatial & correlation-based DBSCAN clustering
    ├── deduplication_test.py                # Tests for deduplication, candidate capping, ensembling, and masks
    ├── granule_inference_test.py            # Tests for tiled inference, patch cropping/padding, and UTM conversion
    ├── ime_test.py                          # Tests for IME emission rate calculations
    ├── plume_candidate_extraction_test.py   # Tests for candidate extraction, polygonization, and origin finding
    └── spectral_matching_test.py            # Tests for NPZ data loading, observed spectra, and spectral fitting
```

---

## Code & Docstring Style

- **Docstrings**:
  - **CRITICAL RULE**: All module, class, method, and function docstrings must
    strictly follow **Standard Google Python Docstring Style**.
  - Include clearly formatted `Args:`, `Returns:`, `Raises:`, `Yields:`, and
    `Attributes:` sections as applicable.
  - Avoid unstructured, verbose, or legacy docstring formatting.
- **String Formatting**:
  - Always use modern Python **f-strings** (`f"Value: {val}"`) for string
    concatenation and formatting. Never use legacy `%` formatting or
    `.format()`.
- **Type Annotations**:
  - Provide precise, tight type annotations for all function signatures and
    return types.
  - Avoid generic `Any` types; prefer specific types such as `Sequence[int]`,
    `Buffer`, `Self`, or `Literal`.
  - Avoid explicit `Union`/`Optional` types. Use '|'.

## Version Control & Commit Messages

- **Feature Branches**:
  - **CRITICAL RULE**: All code changes and refactoring work MUST be performed
    on dedicated git feature branches (e.g., `git checkout -b <branch-name>`).
  - Never make direct commits on the `main` branch.
- **Code Review**:
  - Always do a code review before committing.
  - Use a different LLM model for the subagent doing the review.
  - Create 1-3 suggestions for improvement to the code based on the current changes.
  - See if there needs to be any changes to `AGENTS.md` based on the current
    changes and propose improvements.
- **Conventional Commits**:
  - All git commit messages MUST adhere to the **Conventional Commits**
    specification (`<type>(<optional scope>): <subject>`).
  - Examples:
    - `feat(dunder): enable the __foo__ feature`
    - `refactor(tests): switch test_init.py from unittest to pytest`
    - `chore(license): Add the SPDX header`
    - `docs: import legacy manuals into docs/ directory`
- **NO Tag or Conversation ID Entries**:
  - **CRITICAL RULE**: Commit messages must **NEVER** contain `TAG=` or `CONV=`
    lines or entries. These are reserved for internal Piper/CL tools and must be
    omitted from all git commits in this repository.

## Package Ecosystem and Environment Management

This repository uses **uv** as its primary Python package manager, virtual environment coordinator, and dependency resolver.

### Tooling & Commands

- **Environment & Lockfile**:
  - Python version: Defined in `.python-version` (`3.13`).
  - Pinned lockfile: `uv.lock`.
  - Dependabot ecosystem: Configured with `package-ecosystem: "uv"` in `.github/dependabot.yaml` for automated dependency bumps.
- **Development & Testing Workflows**:
  - Install git pre-commit hooks: `pre-commit install`
  - Run the test suite: `uv run pytest`
  - Run all pre-commit and linter checks: `uv run pre-commit run --all-files`

### Package Management and Lockfile Updates

When resolving dependencies or updating `uv.lock`, always use the public PyPI index (`https://pypi.org/simple`).

- Always supply `--default-index https://pypi.org/simple` when running
  `uv lock`, `uv add`, or related commands that touch `uv.lock`.
- Note that `pyproject.toml` is configured with:
  ```toml
  [[tool.uv.index]]
  name = "pypi"
  url = "https://pypi.org/simple"
  default = true
  ```
- If running in an internal development environment where pip/system config
  specifies an internal proxy mirror (such as `airlock-proxy`), prepend
  `PIP_CONFIG_FILE=/dev/null` or pass `--default-index https://pypi.org/simple`
  so internal mirror URLs are never written into `uv.lock`.
