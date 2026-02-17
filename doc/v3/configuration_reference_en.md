# Tile-Compile C++ Configuration Reference

This documentation describes all configuration options for `tile_compile.yaml` based on the C++ implementation in `configuration.hpp` and the schema files `tile_compile.schema.json` / `tile_compile.schema.yaml`.

**Source of truth for defaults:** `include/tile_compile/config/configuration.hpp`  
**Schema version:** v3  
**Reference:** Methodology v3.2

**💡 For practical examples and use cases, see:** [Configuration Examples & Best Practices](configuration_examples_practical_en.md)

---

## Table of Contents

1. [Pipeline](#1-pipeline)
2. [Output](#2-output)
3. [Data](#3-data)
4. [Linearity](#4-linearity)
5. [Calibration](#5-calibration)
6. [Assumptions](#6-assumptions)
7. [Normalization](#7-normalization)
8. [Registration](#8-registration)
9. [Tile Denoise](#9-tile-denoise)
10. [Global Metrics](#10-global-metrics)
11. [Tile](#11-tile)
12. [Local Metrics](#12-local-metrics)
13. [Synthetic](#13-synthetic)
14. [Reconstruction](#14-reconstruction)
15. [Debayer](#15-debayer)
16. [Astrometry](#16-astrometry)
17. [PCC](#17-pcc)
18. [Stacking](#18-stacking)
19. [Validation](#19-validation)
20. [Runtime Limits](#20-runtime-limits)

---

## 1. Pipeline

Basic pipeline control.

### `pipeline.mode`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `production`, `test` |
| **Default** | `"production"` |

**Purpose:** Determines the execution mode of the pipeline.

- **`production`**: Complete processing with all quality checks and phases
- **`test`**: Reduced processing for testing purposes (may skip some validations)

### `pipeline.max_frames`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `-1` (unlimited) |
| **Range** | `-1` to `1000000` |

**Purpose:** Maximum number of frames to process. `-1` means process all available frames.

---

## 2. Output

Output file and directory configuration.

### `output.base_dir`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `"./output"` |

**Purpose:** Base directory for all output files. Will be created if it doesn't exist.

### `output.run_name`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `"tile_compile_run"` |

**Purpose:** Name prefix for the output subdirectory. Final output will be in `base_dir/run_name_timestamp/`.

### `output.save_intermediate`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Whether to save intermediate processing results (useful for debugging).

---

## 3. Data

Input data configuration.

### `data.input_dir`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Required** | Yes |

**Purpose:** Directory containing input FITS files.

### `data.file_pattern`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `"*.fits"` |

**Purpose:** Glob pattern for matching input files.

### `data.bayer_pattern`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `RGGB`, `BGGR`, `GRBG`, `GBRG`, `NONE` |
| **Default** | `"RGGB"` |

**Purpose:** Bayer matrix pattern for color filter arrays. `NONE` for monochrome data.

### `data.is_osc`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Whether the data is from One-Shot Color (OSC) cameras.

---

## 4. Linearity

Linearity correction settings.

### `linearity.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable linearity correction.

### `linearity.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `none`, `linearize`, `inverse_gamma` |
| **Default** | `"linearize"` |

**Purpose:** Linearity correction method.

- **`none`**: No correction
- **`linearize`**: Apply linearization curve
- **`inverse_gamma`**: Apply inverse gamma correction

### `linearity.gamma`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `2.2` |
| **Range** | `1.0` to `5.0` |

**Purpose:** Gamma value for inverse gamma correction (used when `method` is `inverse_gamma`).

---

## 5. Calibration

Calibration frame processing.

### `calibration.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable calibration frame processing.

### `calibration.dark_dir`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `""` (disabled) |

**Purpose:** Directory containing dark frames.

### `calibration.flat_dir`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `""` (disabled) |

**Purpose:** Directory containing flat frames.

### `calibration.bias_dir`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `""` (disabled) |

**Purpose:** Directory containing bias frames.

---

## 6. Assumptions

Physical assumptions about the data.

### `assumptions.read_noise`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `5.0` |
| **Units** | electrons |

**Purpose:** Read noise level for noise modeling.

### `assumptions.gain`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `1.0` |
| **Units** | electrons/ADU |

**Purpose:** Camera gain for ADU to electron conversion.

### `assumptions.sky_background`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `100.0` |
| **Units** | ADU |

**Purpose:** Expected sky background level.

---

## 7. Normalization

Frame normalization settings.

### `normalization.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable frame normalization.

### `normalization.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `median`, `mean`, `none` |
| **Default** | `"median"` |

**Purpose:** Normalization method.

- **`median`**: Use median for robust normalization
- **`mean`**: Use mean (less robust to outliers)
- **`none`**: No normalization

---

## 8. Registration

Image registration settings.

### `registration.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable image registration.

### `registration.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `triangle_star_matching`, `star_similarity`, `hybrid_phase_ecc` |
| **Default** | `"triangle_star_matching"` |

**Purpose:** Registration method.

### `registration.max_shift`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `50.0` |
| **Units** | pixels |

**Purpose:** Maximum allowed shift between frames.

### `registration.max_rotation`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `5.0` |
| **Units** | degrees |

**Purpose:** Maximum allowed rotation between frames.

---

## 9. Tile Denoise

Tile-based denoising settings.

### `tile_denoise.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable tile-based denoising.

### `tile_denoise.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `wavelet`, `bilateral`, `none` |
| **Default** | `"wavelet"` |

**Purpose:** Denoising method.

### `tile_denoise.strength`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `0.5` |
| **Range** | `0.0` to `1.0` |

**Purpose:** Denoising strength.

---

## 10. Global Metrics

Global quality metrics.

### `global_metrics.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable global metrics calculation.

### `global_metrics.fwhm_threshold`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `3.0` |
| **Units** | pixels |

**Purpose:** FWHM threshold for star quality assessment.

---

## 11. Tile

Tile processing configuration.

### `tile.size`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `256` |
| **Units** | pixels |

**Purpose:** Size of processing tiles (square).

### `tile.overlap`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `32` |
| **Units** | pixels |

**Purpose:** Overlap between adjacent tiles.

### `tile.min_quality`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `0.5` |
| **Range** | `0.0` to `1.0` |

**Purpose:** Minimum quality threshold for tile acceptance.

---

## 12. Local Metrics

Local quality metrics.

### `local_metrics.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable local metrics calculation.

### `local_metrics.contrast_threshold`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `0.1` |
| **Range** | `0.0` to `1.0` |

**Purpose:** Contrast threshold for local quality assessment.

---

## 13. Synthetic

Synthetic frame generation.

### `synthetic.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enable/disable synthetic frame generation.

### `synthetic.count`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `0` |

**Purpose:** Number of synthetic frames to generate.

---

## 14. Reconstruction

Image reconstruction settings.

### `reconstruction.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `weighted_average`, `median`, `sigma_clipped_mean` |
| **Default** | `"weighted_average"` |

**Purpose:** Reconstruction method.

### `reconstruction.sigma_clip`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `3.0` |
| **Units** | sigma |

**Purpose:** Sigma clipping threshold (used when `method` is `sigma_clipped_mean`).

---

## 15. Debayer

Debayering settings for OSC data.

### `debayer.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable debayering (only for OSC data).

### `debayer.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `bilinear`, `VNG`, `AHD` |
| **Default** | `"bilinear"` |

**Purpose:** Debayering algorithm.

- **`bilinear`**: Fast bilinear interpolation
- **`VNG`**: Variable Number of Gradients
- **`AHD`**: Adaptive Homogeneity-Directed

---

## 16. Astrometry

Astrometry solving settings.

### `astrometry.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enable/disable astrometry solving.

### `astrometry.solver`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `astap`, `siril`, `internal` |
| **Default** | `"astap"` |

**Purpose:** Astrometry solver to use.

### `astrometry.timeout`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `300` |
| **Units** | seconds |

**Purpose:** Timeout for astrometry solving.

---

## 17. PCC

Photometric Color Calibration settings.

### `pcc.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enable/disable photometric color calibration.

### `pcc.catalog`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `siril`, `vizier_gaia`, `vizier_apass`, `auto` |
| **Default** | `"auto"` |

**Purpose:** Star catalog source for color calibration.

### `pcc.source`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `siril`, `vizier_gaia`, `vizier_apass`, `auto` |
| **Default** | `"auto"` |

**Purpose:** Catalog source (same as `catalog`).

### `pcc.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `proportion`, `regression` |
| **Default** | `"proportion"` |

**Purpose:** Color correction method.

- **`proportion`**: Use color proportions (background-preserving)
- **`regression`**: Use linear regression (Siril SPCC style)

---

## 18. Stacking

Final stacking settings.

### `stacking.method`

| Property | Value |
|----------|-------|
| **Type** | string (enum) |
| **Values** | `average`, `median`, `sigma_clipped` |
| **Default** | `"average"` |

**Purpose:** Final stacking method.

### `stacking.sigma_clip`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `3.0` |
| **Units** | sigma |

**Purpose:** Sigma clipping threshold (used when `method` is `sigma_clipped`).

---

## 19. Validation

Validation and quality control.

### `validation.enabled`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `true` |

**Purpose:** Enable/disable validation checks.

### `validation.strict`

| Property | Value |
|----------|-------|
| **Type** | boolean |
| **Default** | `false` |

**Purpose:** Enable strict validation (fails on warnings).

---

## 20. Runtime Limits

Runtime and resource limits.

### `runtime_limits.max_memory_gb`

| Property | Value |
|----------|-------|
| **Type** | float |
| **Default** | `16.0` |
| **Units** | GB |

**Purpose:** Maximum memory usage limit.

### `runtime_limits.max_threads`

| Property | Value |
|----------|-------|
| **Type** | integer |
| **Default** | `0` (auto-detect) |

**Purpose:** Maximum number of threads to use. `0` means use all available cores.

---

## Notes

1. **Schema Validation**: All configuration files should validate against `tile_compile.schema.json`.
2. **Default Values**: Default values are defined in `include/tile_compile/config/configuration.hpp`.
3. **Methodology**: This configuration follows the Tile-Based Quality Reconstruction Methodology v3.2.
4. **Units**: Where specified, units are provided for numeric parameters.
5. **Ranges**: Where specified, valid ranges are provided for numeric parameters.

For more detailed information about the methodology and implementation, see the methodology documentation.
