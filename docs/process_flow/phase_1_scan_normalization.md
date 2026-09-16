# Phase 1 — SCAN_INPUT, CHANNEL_SPLIT, NORMALIZATION

> **C++-Implementierung:** `runner_pipeline.cpp` (SCAN_INPUT),
> `runner_phase_metrics.cpp` (`run_phase_channel_split_normalization_global_metrics`)

## SCAN_INPUT

Erste emittierte Phase eines `reconstruct`-Laufs.

**Aufgaben:**

- FITS-Enumeration im `--input-dir` (rekursive Erkennung über
  `io::is_fits_image_path`)
- Header-Auswertung: Dimensionen, Belichtungszeiten (`EXPTIME`), Filter,
  Bayer-Schlüssel
- Farbmodus-Erkennung: `MONO` vs. `OSC`; Bayer-Pattern (RGGB/BGGR/GRBG/GBRG)
- Linearitätsvalidierung der Eingabedaten
- Disk-Space-Precheck für `runs_dir` (scan_dir × 4 als Konservativschätzung)
- Farbmodus-Konsistenz über alle Frames; Fehler sind hier terminal
  (`phase_end ... "error"`)

**Ausgaben:** Scan-Metadaten im Phase-Event; keine Artefaktdateien außer dem
Event-Stream.

## CHANNEL_SPLIT (Metadaten-Phase)

Emittiert sofort nach SCAN_INPUT — **ohne irgendwelche Pixel zu verarbeiten**.

**Event-Payload:**

```json
// OSC:
{"mode": "OSC", "channels": ["R", "G", "B"],
 "bayer_pattern": "RGGB",
 "note": "deferred_to_forward_drizzle_cfa"}
// MONO:
{"mode": "MONO", "channels": ["L"]}
```

**Warum keine echte Trennung:** CFA Forward Drizzle debayert nie. Die
Kanalzuordnung erfolgt erst im Gather pro Quellpixel über
`cfa_channel_for_source_pixel(sx, sy, bayer_pattern, cfa_origin_x,
cfa_origin_y)` — jeder Pixel landet direkt in seiner CFA-Ebene des
Akkumulators. Es entstehen also zu keinem Zeitpunkt materialisierte
R/G/B-Einzeldateien; die Phase dokumentiert nur Modus + Pattern als
Provenienz für Report, UI und Resume.

## NORMALIZATION

**Zweck:** Alle Frames auf ein gemeinsames photometrisches Niveau bringen,
bevor Geometrie und Qualität gemessen werden.

**Pro Frame:**

- Additiver Hintergrund `B_f` — bei OSC **kanalweise** (`B_r`, `B_g`, `B_b`,
  `B_mono` für Diagnose)
- Photometrischer Faktor `P_f` (kanalweise `P_r`, `P_g`, `P_b`)
- Photometrische Skalierung `exposure_ratio`, wenn alle `EXPTIME`-Header
  vorhanden und plausibel sind — sonst `identity_fallback`
- `normalization.enabled: false` ist terminal (`"disabled but required"`)

**Caches und Artefakte:**

- `cache/normalized_frames/` — normalisierte Vollframes (`RunnerFrameCache`),
  wird in NORMALIZED_CACHE versiegelt
- Registrierungs-Proxys (downsampled) pro Frame für die REGISTRATION
- `artifacts/normalization.json` — Modus, Bayer, `B_*`/`P_*` pro Frame
- `artifacts/global_metrics.json` — globale Frame-Metriken und -gewichte
  (`G_f`), die REGISTRATION und GLOBAL_QUALITY als Eingang nutzen

**Formel:**

```
J_f = I_raw − B_f        (hintergrundbereinigt)
I_f = J_f / P_f          (photometrisch normiert)
```

**Parallelisierung:** `compute_adaptive_worker_count` mit
`WorkerParallelProfile::MixedIo` — I/O-gebundene Workerzahl abhängig von
Framezahl und Hauptspeicher.
