# Process Flow Documentation – Tile-basierte Qualitätsrekonstruktion v4 (C++)

## Überblick

Diese Dokumentation beschreibt den **aktuellen C++‑Ablauf** der v4‑Methodik. Python‑Referenzen wurden entfernt; alle Schritte beziehen sich auf die C++‑Implementierung (v. a. `tile_compile_cpp/apps/runner_main.cpp` und `tile_compile_cpp/src/*`).

## Pipeline‑Phasen (aktuell)

| Phase | Name | Kurzbeschreibung |
|------:|------|------------------|
| 0 | SCAN_INPUT | Input‑Frames prüfen, Metadaten ermitteln, Linearity‑Sampling |
| 1 | CHANNEL_SPLIT | Deferred: Bayer‑Split erst bei Tile‑Processing |
| 2 | NORMALIZATION | Deferred: Hintergrund‑Normalisierung beim Tile‑Load |
| 3 | GLOBAL_METRICS | Hintergrund/Noise/Gradient → globale Gewichte G_f |
| 4 | TILE_GRID | Hierarchisch + Warp‑Probe + adaptive Tile‑Größe |
| 5 | LOCAL_METRICS | Lokale Metriken pro Tile/Frame, STAR/STRUCTURE |
| 6 | TILE_RECONSTRUCTION_TLR | Tile‑Local Registration (TLR) + Rekonstruktion |
| 7 | STATE_CLUSTERING | K‑Means auf Zustandsvektoren |
| 8 | SYNTHETIC_FRAMES | TLR‑Rekonstruktion je Cluster (synthetische Frames) |
| 9 | STACKING | Mittelung synthetischer Frames + Validierung |
| 10 | DEBAYER | OSC‑Ausgabe für GUI/Reporting |

## Wichtige v4‑Prinzipien (C++‑Stand)

- **Keine globale Registrierung**: Registration ist **pro Tile** (TLR), inklusive optionaler Rotation pro Tile (`registration.local_tiles.allow_rotation`).
- **Keine globale Referenzgeometrie**: Referenz wird je Tile iterativ aus warpen Frames aufgebaut.
- **Deferred Processing**: Channel‑Split & Normalisierung erfolgen beim Tile‑Load.
- **Adaptive Refinement**: Tiles mit hoher Warp‑Varianz oder niedriger Korrelation werden mit Overlap gesplittet.
- **Linearity‑Check ohne Hard‑Abort**: Nichtlineare Frames werden geloggt und ausgeschlossen; Abbruch nur wenn keine Frames übrig bleiben.
- **Synthetic Frames aus TLR**: Cluster‑Frames werden per TLR rekonstruiert, nicht aus Roh‑Frames gemittelt.

## Presets (C++‑GUI)

**Ablage:** `tile_compile_cpp/*.yaml` und nach Build im `tile_compile_cpp/build/`.

- `tile_compile.yaml` – Default (Baseline v4)
- `tile_compile_eq.yaml` – EQ‑Montierung, ruhiges Seeing
- `tile_compile_altaz.yaml` – Alt/Az, starke Feldrotation
- `tile_compile_polar.yaml` – Polnähe, sehr instabil
- `tile_compile_smart.yaml` – Smart‑Teleskope (allgemein)
- `tile_compile_dwarf.yaml` – DWARF II
- `tile_compile_seestar.yaml` – ZWO SeeStar

**Preset‑Idee (kurz):**
- **EQ/ruhig**: weniger Iterationen, geringere Beta, weniger Refinement.
- **Alt/Az**: mehr Iterationen, höhere Beta, adaptive Tiles aktiv.
- **Polar/instabil**: maximale Refinement‑Tiefe, höhere Robustheit.
- **Smart/DWARF/SeeStar**: moderate ECC‑Schwelle, stärkere Glättung, konservative Warp‑Delta‑Filter.

Details und Begründungen je Preset stehen in `phase_0_overview.md`.

---

## Dokument‑Index

- [Phase 0: Overview & Input Scanning](phase_0_overview.md)
- [Phase 1–2: Deferred Processing](phase_1_deferred.md)
- [Phase 3: Global Metrics](phase_3_global_metrics.md)
- [Phase 4: Tile Grid](phase_4_tile_grid.md)
- [Phase 5: Local Metrics](phase_5_local_metrics.md)
- [Phase 6: TLR](phase_6_tlr.md)
- [Phase 7: Clustering](phase_7_clustering.md)
- [Phase 8: Synthetic Frames](phase_8_synthetic.md)
- [Phase 9: Stacking + Validation](phase_9_stacking.md)
