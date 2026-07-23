# AQMH — Adaptive Quality Map Hyperstacking

AQMH is the default reconstruction path as of v0.3.0. For each input frame a **dense quality map** `Q_map_{f,c}(x,y)` is computed using a **multi-scale Laplacian pyramid**, combining sharpness and SNR metrics into a per-pixel quality value. The final image is reconstructed as a **per-pixel weighted mean** — effective weight `W = G_{f,c} * Q_map_{f,c}(x,y)`, where `G_{f,c}` is the global frame weight from shared preprocessing. No tile grid, no OLA seams.

> **Normative specification:** [AQMH Methodology v0.2.1](../AQMH/aqmh_methodik_en_v0.2.1.md)

## How it works

```
For each frame f, channel c:
  For each pyramid scale s (D_s = 4^s, window R_s = 4 px in downscaled pixels):
    1. Downsample I_{f,c} by D_s (mask-aware area average)
    2. Compute per-window:
         Phi_sharp = local variance of masked Laplacian (sharpness)
         Phi_snr   = local SNR = mu / max(1.4826*MAD, eps)
         Phi_artifact = 1 - clip(outlier_frac / frac_artifact_max, 0, 1)
    3. Psi_s = sigmoid(w_sharp*z(Phi_sharp) + w_snr*z(Phi_snr)) * Phi_artifact
       (z = robust z-score; artifact gate is multiplicative — one bad scale vetos pixel)
    4. Upsample Psi_s to canvas resolution (mask-aware bilinear)
  Q_map_{f,c} = geometric_mean over scales(Psi_s)  # all scales must agree
  Store Q_map to disk cache (default: 1/2-area uint16)

Reconstruction (per canvas-valid pixel p):
  W_{f,c}(p) = G_{f,c} * Q_map_{f,c}(p)
  R_c(p) = sum_f( W_{f,c}(p) * I_{f,c}(p) ) / sum_f( W_{f,c}(p) )
```

## Key parameters (`aqmh.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aqmh.enabled` | `true` | Enable AQMH (false = use classic TILE_RECONSTRUCTION) |
| `aqmh.pyramid.scales` | `4` | Pyramid levels for multi-scale analysis |
| `aqmh.pyramid.base_window_px` | `4` | Window size at lowest pyramid level |
| `aqmh.pyramid.w_sharp` | `0.6` | Sharpness weight in quality index |
| `aqmh.pyramid.w_snr` | `0.4` | SNR weight in quality index |
| `aqmh.pyramid.k_artifact` | `3.0` | MAD multiplier for artifact detection (higher = more tolerant) |
| `aqmh.pyramid.frac_artifact_max` | `0.25` | Max artifact fraction per window before discard |
| `aqmh.storage.resolution_divisor` | `2` | Quality map cache resolution (1/2/4) |
| `aqmh.storage.dtype` | `uint16` | Cache data type (`float32`, `uint16`, or `uint8`) |
| `aqmh.storage.max_resident_maps` | `2` | Max quality maps in RAM simultaneously |
| `aqmh.cherry_pick.enabled` | `false` | Stack only top-quality frames |
| `aqmh.cherry_pick.k_frac` | `0.30` | Fraction of best frames to use (0.30 = best 30%) |
| `aqmh.cherry_pick.k_min_required` | `20` | Run gate and minimum retained samples per pixel |
| `aqmh.diagnostics.enabled` | `true` | Enable AQMH diagnostics phase |
| `aqmh.diagnostics.level` | `full` | Detail level: `none`, `summary`, or `full` |
| `aqmh.diagnostics.format` | `json` | Diagnostic output format: `json` or `binary` |
| `aqmh.reconstruction.chunk_rows` | `0` | Row chunk size (0 = auto from memory budget) |
| `aqmh.global_quality.g_k_scale` | `1.5` | Sigmoid temperature; global weight remains bounded to `[g_floor, 1]` |
| `aqmh.reconstruction.clip_sigma_low/high` | `2.0 / 1.5` | Asymmetric lower/upper clipping thresholds |
| `aqmh.reconstruction.clip_iterations` | `4` | AQMH clipping iterations |

Full parameter documentation: [Configuration Reference — §12b AQMH](../configuration_reference_en.md)
Practical examples: [Configuration Examples — AQMH section](../configuration_examples_practical_en.md)

## When to use AQMH vs. Classic

| Situation | Recommendation |
|-----------|----------------|
| Default / most sessions | **AQMH** (enabled by default) |
| Tile seams or OLA artifacts visible | **AQMH** eliminates seams entirely |
| Strongly varying frame quality (seeing, clouds) | **AQMH** with `cherry_pick.enabled: true`, `resolution_divisor: 1`, `dtype: float32` |
| Very large sessions, RAM-limited | **AQMH** with `storage.resolution_divisor: 4`, `dtype: uint8` |
| Sessions with satellite trails / cosmetic issues | **AQMH** with `k_artifact: 5.0`, `frac_artifact_max: 0.35` |
| Research requiring TBQR tile-weighted OLA | Classic (`aqmh.enabled: false`) |

## Minimal AQMH config

```yaml
aqmh:
  enabled: true          # default — can be omitted
  pyramid:
    k_artifact: 3.0      # default
    frac_artifact_max: 0.25  # default
```

## Disable AQMH (revert to classic)

```yaml
aqmh:
  enabled: false
```

## AQMH Papers

- [AQMH v0.2.0 Paper](../AQMH/zenodo-0.2.0/paper-adaptive_quality_map_hyperstacking_m31_run_20260722_en.pdf) — M31 validation run with v0.2.0 extensions
- [AQMH v0.1.0 Paper](../AQMH/zenodo-0.1.0/paper-adaptive_quality_mask_harvesting_m31_run_20260611_en.pdf) — original method definition and M31-A validation run
