**Plan: Rückbau auf Methodik v3 (C++ Pipeline)**

A. Phasen/Struktur
- Phase-Enum & GUI auf v3 setzen: SCAN_INPUT, REGISTRATION, CHANNEL_SPLIT, NORMALIZATION, GLOBAL_METRICS, TILE_GRID, LOCAL_METRICS, TILE_RECONSTRUCTION, STATE_CLUSTERING, SYNTHETIC_FRAMES, STACKING, DEBAYER, DONE.
- Event-Namen/Progress-Widget entsprechend anpassen; nur diese Phasen emittieren.

B. Konfiguration vereinfachen
- Dateien: tile_compile_cpp/tile_compile.yaml, tile_compile.schema.{json,yaml}, src/io/config.cpp.
- Entfernen: v4.*, adaptive_tiles, warp_probe, warp_field diagnostics, tile_rescale, convergence, auto_skip, phase6_io, beta, memory_limits v4, synthetic auto_skip, etc.
- Belassen/Hinzufügen: v3-Felder (registration.engine [opencv_cfa|siril], sigma_clip, FWHM tile sizing, global/local metric weights); Defaults auf v3 angleichen.

C. Registrierung (v3)
- Eine globale Transformation pro Frame (CFA-aware ECC) oder Siril-Pfad; keine per-Tile ECC in Phase 6/8.
- rotation optional global; lokale Warps entfernen/deaktivieren.

D. Tile-Grid & Metriken (v3)
- Adaptive/HPC Tile-Grid streichen; FWHM-basierte Tilegröße: T = clip(s·FWHM, T_min, T_max), Overlap laut v3.
- Lokale Metriken: v3-Satz (FWHM, roundness, contrast/ENR, background, noise), ohne warp-variance/debug Felder.

E. Rekonstruktion
- Gewichtete Overlap-Add mit 2D-Hanning; keine tile_rescale, kein TLR/warp_consistency/max_warp_delta Filter.
- Fallback: ungewichtetes Mittel bei SumWeights≈0 (ε).

F. Clustering & Synthetic (v3)
- K = clip(floor(N/10), 5..30); kein auto_skip; keine tile_weighted Sondergewichte.
- Synthetic Frames = gewichtete Stacks je Cluster (ohne lokale ECC).

G. Stacking & Debayer
- Sigma-Clipping wie v3; Debayer nach Stacking (Pfad B). Siril-Pfad A: Debayer vorher.

H. Artefakte/Outputs
- Nur v3-Artefakte: global_metrics, tile_grid, local_metrics, tile_reconstruction, clustering, synthetic, stacked. Warp-DX/DY/invalid_tile optional entfernen.
- README/Docs anpassen: „v3 only“.

I. Aufräumen & Tests
- Nicht genutzte Module entfernen/abschalten: adaptive_tile_grid, v4 diagnostics; Tests anpassen.
- Smoke-Test mit 5–10 Frames (OSC) über CLI.

Umsetzungsreihenfolge
1) A+B: Phase-Enum/GUI + Config/Schema bereinigen.
2) C+D: Registrierung vereinfachen, Tile-Grid/Metriken zurückbauen.
3) E: Tile-Reconstruction vereinfachen.
4) F: Clustering/Synthetic v3-Defaults.
5) G+H: Stacking/Debayer/Artifacts konsolidieren.
6) I: Aufräumen, Tests, Smoke-Run.

**Progress 2026-01-30**
- Config/Schema v3-only, sample YAMLs vereinheitlicht.
- GUI/Phasen auf v3 gesetzt.
- Runner: adaptive/v4-Pfade entfernt; Tile-Recon auf globale Warps + Hanning reduziert; lokale Metrics nutzen global warp.
