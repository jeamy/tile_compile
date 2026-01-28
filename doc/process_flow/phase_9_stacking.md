# Phase 9: Stacking + Validation (C++)

## Ziel

Erzeuge das finale Output‑Frame aus synthetischen Frames (falls vorhanden) und validiere die Qualität.

## C++‑Implementierung

**Referenz:** `tile_compile_cpp/apps/runner_main.cpp` (Phase 9 + Validation)

### Ablauf

1. **Stacking**
   - Wenn synthetische Frames existieren: Mittelung über `synthetic_frames`.
   - Sonst: Rekonstruktion aus Phase 6 wird weiterverwendet.

2. **Output**
   - `outputs/stacked.fits`
   - `outputs/reconstructed_L.fit`

3. **Validation (nach Stacking)**
   - FWHM‑Verbesserung (medianbasiert)
   - Tile‑Weight‑Varianz
   - Tile‑Pattern‑Check (Sobel‑Linien entlang Tile‑Grenzen)
   - **Kein Hard‑Abort**, aber `run_validation_failed` wird gesetzt.

## C++‑Skizze

```cpp
if (use_synthetic) recon = mean(synthetic_frames);
write_fits(recon, "stacked.fits");
validate(recon, tile_weights, fwhm_stats);
```

## Parameter (Auszug)

- `validation.min_fwhm_improvement_percent`
- `validation.min_tile_weight_variance`
- `validation.require_no_tile_pattern`

