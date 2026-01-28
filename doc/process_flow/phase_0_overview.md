# Phase 0: Überblick & Input‑Scanning (C++)

## Ziel

Phase 0 bereitet die Pipeline vor und validiert grundlegende Annahmen:
- Rohdaten sind **linear**
- kein globales Alignment
- keine globale Referenzgeometrie
- Bewegung ist **orts‑ und zeitabhängig**
- alle Korrekturen sind **Tile‑lokal**

## Ablauf (C++‑Implementierung)

**Referenz:** `tile_compile_cpp/apps/runner_main.cpp`

1. **Input prüfen**
   - Dateiliste, Bildgröße, Bayer‑Pattern, Farbraum.
   - Mindestanzahl Frames (`data.frames_min`, `assumptions.frames_min`).

2. **Linearity‑Sampling**
   - Stichprobe aus `linearity.max_frames` Frames.
   - Bewertung über `linearity.min_overall_linearity`.
   - **Nichtlineare Frames**: werden **geloggt und ausgeschlossen** (kein Hard‑Abort).
   - Abbruch nur, wenn **keine Frames** übrig bleiben.

3. **Run‑Metadaten**
   - Event‑Log schreibt Start/Ende der Phase.
   - Ergebnis: validierte Frame‑Liste für Phase 3+.

## C++‑Code‑Skizze (vereinfacht)

```cpp
// runner_main.cpp (vereinfacht)
scan_input();
frames = filter_linearity(frames, cfg.linearity);
if (frames.empty()) abort("no frames after linearity");
```

## Output

- `run_events.jsonl`: Phase 0 Start/Ende, ggf. Warnungen.
- Gefilterte Frame‑Liste für Folgephasen.

## Presets – Motivation

**Warum die Presets so parametriert sind:**

- **EQ / ruhiges Seeing** (`tile_compile_eq.yaml`)
  - Weniger Iterationen (`v4.iterations ≈ 2–3`)
  - Niedrigere Beta (`v4.beta ≈ 3–4`)
  - Adaptive Tiles teils deaktiviert
  - Ziel: schnelle, stabile Rekonstruktion

- **Alt/Az / Feldrotation** (`tile_compile_altaz.yaml`)
  - Mehr Iterationen (≈ 4)
  - Höhere Beta (≈ 6)
  - Adaptive Tiles aktiv
  - Ziel: lokale Rotation/Drift kompensieren

- **Polnähe / instabil** (`tile_compile_polar.yaml`)
  - Max. Refinement‑Tiefe
  - Robustere Filter (warp‑delta, smoothing)
  - Ziel: hohe Variabilität lokal stabilisieren

- **Smart‑Teleskope** (`tile_compile_smart.yaml`, `*_dwarf.yaml`, `*_seestar.yaml`)
  - Moderate ECC‑Schwelle
  - Stärkeres temporales Smoothing
  - Konservatives Warp‑Delta
  - Ziel: kurzbelichtete, kleine Optiken mit Drift stabilisieren

## Wichtige Parameter (Auszug)

- `linearity.enabled`, `linearity.max_frames`, `linearity.min_overall_linearity`, `linearity.strictness`
- `assumptions.frames_min`, `assumptions.frames_reduced_threshold`
- `data.linear_required`

