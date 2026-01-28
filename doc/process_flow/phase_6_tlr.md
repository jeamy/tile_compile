# Phase 6: Tile‑Local Registration & Reconstruction (TLR, C++)

## Ziel

Registrierung und Rekonstruktion erfolgen **pro Tile**, optional mit **Rotation pro Tile** (`allow_rotation`). Das verhindert Feldrotations‑Artefakte und kompensiert lokale Drift.

## C++‑Implementierung

**Referenzen:**
- `tile_compile_cpp/apps/runner_main.cpp` (Phase 6)
- `tile_compile_cpp/src/registration/*`
- `tile_compile_cpp/src/image/*`

## Ablauf (Schritt‑für‑Schritt)

1. **Tile‑Load + Normalisierung (deferred)**
2. **Initiale Referenz**: Median‑Stack aller Tile‑Samples
3. **Iterative Rekonstruktion (n = v4.iterations)**
   - ECC‑Warp pro Frame (mit optionaler Rotation)
   - Fallback: Phase‑Correlation bei ECC‑Failure
   - Gewichte: `W_f,t = G_f · L_f,t · R_f,t`
   - `R_f,t = exp(beta · (cc - 1))`
4. **Temporal Smoothing** (median filter) auf Warp‑Trajektorien
5. **Warp‑Delta Filter** (verhindert Doppelsterne)
6. **ECC‑CC Filter** (bewahrt PC‑Fallbacks)
7. **Rekonstruiere Tile** (gewichtetes Mittel)
8. **Variance‑Window Gewicht** `psi(var)`
9. **Overlap‑Add** mit Hanning‑Fenster
10. **Post‑Warp Metriken** (Kontrast/Background/SNR) für Diagnose

## Rotation pro Tile

Wenn `registration.local_tiles.allow_rotation=true`, wird die **volle Warp‑Matrix** aus ECC übernommen (Rotation + Translation). Bei `false` wird nur die Translation angewandt.

## C++‑Skizze

```cpp
for (tile in tiles) {
  ref = median(tile_stack);
  for (iter in 1..iterations) {
    for (frame in frames) {
      init = phasecorr(ref, frame);
      rr = ecc_warp(frame, ref, allow_rotation, init);
      warp = rr.success ? rr.warp : translation(init);
      cc = rr.success ? rr.correlation : 0.0f;
      W = G_f * L_f,t * exp(beta*(cc-1));
    }
    smooth_warps(); filter_warp_delta(); filter_cc();
    ref = weighted_reconstruct();
  }
  tile_rec = weighted_reconstruct();
  overlap_add(tile_rec, psi(var));
}
```

## Adaptive Refinement (in Phase 6)

- Tiles mit **hoher Warp‑Varianz** oder **niedrigem mean‑cc** werden gesplittet.
- Splits verwenden Overlap‑Faktor (`tile.overlap_fraction`).
- Maximal `adaptive_tiles.max_refine_passes`.

## Artefakte

- `artifacts/warp_dx.fits`, `warp_dy.fits`
- `artifacts/fwhm_heatmap.fits`
- `artifacts/invalid_tile_map.fits`
- `artifacts/tile_validation_maps.json`

## Parameter (Auszug)

- `registration.local_tiles.{ecc_cc_min,max_warp_delta_px,temporal_smoothing_window,variance_window_sigma,allow_rotation}`
- `v4.iterations`, `v4.beta`, `v4.adaptive_tiles.*`

