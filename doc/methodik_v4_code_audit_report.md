# Methodik v4 Code-Audit Report

Datum: 2026-01-27

Dokumentbasis: `doc/tile_basierte_qualitaetsrekonstruktion_methodik_v4.md`

Scope (gepruefte Implementationen):
- C++ Pipeline: `tile_compile_cpp/apps/runner_main.cpp`
- C++ Metriken: `tile_compile_cpp/src/metrics/metrics.cpp`, `tile_compile_cpp/src/metrics/tile_metrics.cpp`
- C++ Tile-Grid/Adaptive: `tile_compile_cpp/src/pipeline/adaptive_tile_grid.cpp`
- Python TLR: `tile_compile_python/runner/tile_local_registration_v4.py`, `tile_compile_python/runner/tile_processor_v4.py`
- Python Runner: `tile_compile_python/tile_compile_runner.py`

## Kurzfazit
Die C++- und Python-Pipelines sind nach den letzten Aenderungen weitgehend v4-konform. Die zuvor kritischen Punkte (R_{f,t}, Median-Referenz, Post-Warp-Metriken, State-Vector, Validierungsartefakte, Linearity-Check sowie STATE_CLUSTERING/SYNTHETIC_FRAMES in Python) sind umgesetzt.

## Findings (Abweichungen / fehlende Implementationen)

### 1) R_{f,t} (Registrierungsqualitaet) wird in C++ nicht genutzt
- Spec: W_{f,t} = G_f * L_{f,t} * R_{f,t}, mit R_{f,t} = exp(beta * (cc - 1)).
- C++ Phase 6 verwendet nur G_f * L_{f,t}; cfg.v4.beta ist definiert, aber nicht eingesetzt.
- Python `TileProcessor` implementiert R_{f,t} (beta-gewichtetes cc).
- Impact: Fehlende Qualitätsdämpfung fuer schwache ECC-Korrelationen -> abweichende Rekonstruktion/Stacking.

Betroffene Dateien:
- `tile_compile_cpp/apps/runner_main.cpp` (Phase 6 Gewichtung)
- `tile_compile_cpp/include/tile_compile/config/configuration.hpp` (beta konfig)
- `tile_compile_python/runner/tile_processor_v4.py` (korrekte R_{f,t}-Nutzung)

**Status (2026-01-27):** C++-TLR nutzt jetzt R_{f,t} via `cfg.v4.beta`.

### 2) Initiale Referenzbildung weicht ab (C++)
- Spec: initiale Referenz pro Tile = Median-Frame (zeitlicher Median aller Frames).
- C++: initiale Referenz ist das erste verfuegbare Tile-Sample aus den ersten 5 Frames (kein Median).
- Python: Median-Referenz implementiert.
- Impact: Bias auf fruehe Frames, geringere Robustheit.

Betroffene Dateien:
- `tile_compile_cpp/apps/runner_main.cpp` (Initial reference in process_tile)
- `tile_compile_python/runner/tile_processor_v4.py` (Median-Referenz)

**Status (2026-01-27):** C++ initiale Referenz ist Median ueber alle Frames.

### 3) Post-Warp-Metriken fehlen in C++
- Spec: Post-Warp-Metriken in Phase 6 (Kontrast/Laplacian-Varianz, Hintergrund, SNR-Proxy).
- C++: keine Implementierung von Post-Warp-Metriken; nur Pre-Warp (Phase 5).
- Python: `compute_post_warp_metrics()` vorhanden (nicht in C++ portiert).
- Impact: fehlende Grundlage fuer State-Vector/Qualitaetsanalyse nach Registrierung.

Betroffene Dateien:
- `tile_compile_cpp/apps/runner_main.cpp` (Phase 6 ohne Post-Warp metrics)
- `tile_compile_python/runner/tile_local_registration_v4.py` (compute_post_warp_metrics)

**Status (2026-01-27):** Post-Warp-Metriken in C++ erfasst (Kontrast, Hintergrund, SNR-Proxy).

### 4) Rekursive Tile-Verfeinerung basiert in C++ nicht auf Warp-Varianz/Korrelation
- Spec: Rekursive Verfeinerung bei hoher Warp-Varianz oder niedriger Korrelation.
- C++: adaptive Tile-Groesse basiert auf Warp-Gradient-Probe vor TLR (keine Rekursion auf Basis der tatsaechlichen Warp-Varianz nach TLR).
- Python: `refine_tiles` im Runner nutzt Warp-Varianz aus Phase 6 und splittet Tiles in Passes.
- Impact: v4-Refinement-Regel nur teilweise umgesetzt; potenziell suboptimale Tile-Groessen.

Betroffene Dateien:
- `tile_compile_cpp/src/pipeline/adaptive_tile_grid.cpp`
- `tile_compile_cpp/apps/runner_main.cpp` (keine rekursive Splits in Phase 6)
- `tile_compile_python/tile_compile_runner.py` (refine_tiles verwendet Warp-Varianz)

**Status (2026-01-27):** C++-Refinement implementiert: Tiles werden nach Phase 6 anhand Warp-Varianz/Mean-CC gesplittet und Phase 5/6 erneut ausgefuehrt (bis `max_refine_passes`). Split-Tiles behalten Overlap gem. `tile.overlap_fraction`.

### 5) State-Clustering verwendet falschen Zustandsvektor (C++)
- Spec: v_f = (G_f, <Q_tile>, Var(Q_tile), <cc>, Var(A_hat), invalid_tile_fraction).
- C++: verwendet (G_f, mean_local, var_local, background, noise); cc/warp_var/invalid_tile_fraction fehlen.
- Python Runner: State-Clustering ist als "simplified" markiert und macht keine echte v4-Clusterung.
- Impact: Clusterung reflektiert nicht die geforderte registrierungsbasierte Qualitaet.

Betroffene Dateien:
- `tile_compile_cpp/apps/runner_main.cpp` (Phase 7 State Vectors)
- `tile_compile_python/tile_compile_runner.py` (Phase 7 simplified)

**Status (2026-01-27):** C++ State-Vector auf v4 angepasst; invalid_tile_fraction pro Frame.

### 6) SYNTHETIC_FRAMES in Python komplett vereinfacht/uebersprungen
- Spec: Pro Cluster ein synthetisches Frame, danach lineares Stacking.
- Python Runner: Phase 8 markiert als "skipped_in_v4_simple".
- C++: implementiert synthetische Frames, aber basierend auf globalen Gewichten und Rohframes (keine lokalen TLR-Resultate, kein R_{f,t}).
- Impact: Python-Implementierung nicht v4-konform; C++ nur teilweise.

Betroffene Dateien:
- `tile_compile_python/tile_compile_runner.py` (Phase 8 skip)
- `tile_compile_cpp/apps/runner_main.cpp` (synthetic frames ohne lokalen Anteil)

**Status (2026-01-27):** Python Runner implementiert jetzt STATE_CLUSTERING + SYNTHETIC_FRAMES (k-means, pro Cluster synthetisches Frame, danach Stacking).

### 7) v4-Validierungsartefakte fehlen (C++)
- Spec: FWHM-Heatmaps, Warp-Vektorfelder, Tile-Invalid-Karten.
- C++: Validierung berechnet FWHM-Verbesserung, Tile-Weight-Varianz, Tile-Pattern-Ratio. Keine Heatmaps/Vektorfelder/Invalid-Karten.
- Python: Validiert nur Mindestanteil gueltiger Tiles.
- Impact: Fehlende Pflichtartefakte und v4-spezifische Abbruchkriterien.

Betroffene Dateien:
- `tile_compile_cpp/apps/runner_main.cpp` (Validation-Block)
- `tile_compile_python/tile_compile_runner.py` (Phase 6 validity check)

### Update (2026-01-27): Artefakte in C++ implementiert
- `artifacts/fwhm_heatmap.fits`
- `artifacts/warp_dx.fits`, `artifacts/warp_dy.fits`
- `artifacts/invalid_tile_map.fits`
- `artifacts/tile_validation_maps.json` (per-Tile Werte: FWHM-Median, dx/dy, invalid)

### 8) Linearity-Check nicht strikt durchgesetzt (Python, optional in C++)
- Spec: Rohdaten sind linear; Verstoss -> Abbruch.
- C++: Linearity-Check nur wenn `cfg.linearity.enabled` gesetzt.
- Python: kein Linearity-Check.
- Impact: In nicht-konfigurierten Runs wird die Grundannahme nicht erzwungen.

Betroffene Dateien:
- `tile_compile_cpp/apps/runner_main.cpp` (cfg.linearity.enabled)
- `tile_compile_python/tile_compile_runner.py` (kein Check)

**Status (2026-01-27):** Linearity-Check ist jetzt Pflicht in C++ und Python (Abbruch bei deaktiviert/unter Threshold).

## Teilkonforme/konforme Bereiche (Auswahl)
- Keine globale Registrierung, tile-lokale Registrierung (C++/Python) implementiert.
- Globale Normalisierung per Frame (C++ robust; Python vereinfacht per Median).
- Overlap-Add mit Hann * psi(Var) in Phase 6/9 (C++/Python).
- Zeitliche Glattung der Warps (Median-Filter in C++/Python; Savitzky-Golay in Python optional).

## Konkrete ToDos fuer volle v4-Konformitaet
Aktuell keine offenen Punkte aus diesem Report.
