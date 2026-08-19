# Methane Analysis and Plume Localization (MAPL)

***This is not an officially supported Google product.***

MAPL is a library for detecting, analyzing, and localizing methane ($CH_4$)
emission plumes from hyperspectral imagery, currently supporting data from
the **Earth Surface Mineral Dust Source Investigation (EMIT)** instrument.

This repository provides code to perform **tiled model inference** using
pretrained models, followed by automated **deduplication** and **plume
vetting**. It produces structured outputs including plume masks, enhancements,
and localized plume source locations in both raster and vector formats.

## Key Features

-   **Tiled Inference:** Efficiently run large hyperspectral scenes through the
    detection model.
-   **Disaggregated Plume Extraction:** Detects overlapping or adjacent plumes
    and disaggregates them: producing individual plume masks plus plume head
    location estimation.
-   **Automated Vetting:** Utilizes spectral and spatial context to limit false
    positives.
-   **Structured Outputs:** Generates Parquet files with plume geometries,
    heads, and detection metrics.

---

## Paper

For more details on the methodology and results, please refer to our paper:
> **[Pointer to Paper/Preprint once available]**

---

## Installation

First, install the library and its dependencies:

```bash
pip install .
```

## Data Access

To run inference, you need to download both the **Radiance (RAD)** and
**Observation (OBS)** granules for your scene of interest from the EMIT
collection on **NASA Earthdata Search** or the **LP DAAC**.

Ensure that both files (e.g., `EMIT_L1B_RAD_*.nc` and `EMIT_L1B_OBS_*.nc`) are
stored in the same directory, as the inference script automatically looks for
the corresponding OBS file based on the RAD file path.

## Running Inference

You can run single granule inference using the provided script. You will need
to supply a path to a pretrained model.

```bash
python3 -m mapl.single_granule_inference \
  --model_path /path/to/pretrained/model \
  --input_filepath /path/to/EMIT_L1B_RAD_001_20220826T174642_2223812_024.nc \
  --output_filepath /path/to/output/prefix
```

### Output

The script will produce:

-   `{prefix}_candidates.parquet`: Candidate plume detections before final
    vetting.
-   `{prefix}_predicted.parquet`: Final filtered/vetted plumes.
-   `{prefix}_enhancement.npy`: Raw enhancement raster (enhancement path
    length in $ppm\cdot m$).

## Output Schema

The `.parquet` files contain the following key columns:

### Candidate Plumes (`_candidates.parquet`)

-   `ee_asset_id`: Identifier for the source granule.
-   `geometry`: Plume polygon in geographic coordinates (WKT string).
-   `geometry_px`: Plume polygon in pixel coordinates (WKT string).
-   `head_point`: Estimated plume source/head location in geographic
    coordinates (WKT string).
-   `head_point_px`: Estimated plume source/head location in pixel
    coordinates (WKT string).
-   `plume_bbox_px`: Bounding box of the plume in pixels.
-   *Optional (if enabled):* `concentration` ($ppm\cdot m$), `binary_mask`,
    `origin_mask`. (Note: column name remains `concentration` for backwards
    compatibility).

### Predicted Plumes (`_predicted.parquet`)

In addition to the geographic and pixel geometry columns listed above, the
predicted plumes file includes vetting and quantification metrics:

-   `fitted_conc`: Retrieved enhancement for the plume ($ppm\cdot m$).
-   `ime_integrated_mass`: Integrated Methane Enhancement mass ($kg$ or $t$
    depending on configuration).
-   `ime_emission_rate`: Estimated emission rate ($kg/h$).
-   `d_norm` / `d_cor`: Spectral matching distance/correlation metrics used
    for vetting.
-   `cluster_size`: Size of the deduplicated cluster.
-   `era5_u_10m` / `era5_v_10m`: Wind components used for quantification
    ($m/s$).

## Ancillary Data (`.npz`)

The repository includes `satellite_gas_statistics_v14.npz`, which contains
pre-computed spectral statistics required for the spectral vetting module
(`SpectralVetting`).

It holds:

-   **Spectral Responses:** Per-band spectral response functions for supported
    sensors (e.g., `emit_l1b`).
-   **Transmittances:** Modeled atmospheric transmittances under standard
    conditions and with added gas enhancements.
-   **Added Enhancements:** Values of added gas (e.g., methane) used to
    simulate enhancements, typically in **$ppm\cdot m$**.

This data allows the code to perform rapid spectral fitting and validation
without needing to run expensive radiative transfer models on the fly.
