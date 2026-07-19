# AQMH_GLOBAL_QUALITY + AQMH_DIAGNOSTICS — AQMH-Erweiterungen

> **Aktueller Code:** `runner_phase_aqmh_global_quality.cpp`, `runner_phase_aqmh_diagnostics.cpp`
> **Phase-Enums:** `AQMH_GLOBAL_QUALITY` (20), `AQMH_DIAGNOSTICS` (22)

Diese beiden Phasen gehören ausschließlich zum AQMH-Zweig. Sie liegen zwischen
`AQMH_MAPS`/`AQMH_RECONSTRUCTION` und dem gemeinsamen `STACKING`.

## AQMH_GLOBAL_QUALITY (Phase 20)

Aus den gespeicherten Quality-Maps werden globale Frame-Faktoren `G_f`
berechnet. Die Faktoren fassen Schärfe, SNR und Hintergrundstrafe pro Frame
deterministisch zusammen. Die Phase verwendet den Cache aus
`<run>/cache/aqmh`; sie berechnet keine Classic-Tile-Gewichte.

```text
AQMH_MAPS -> AQMH_GLOBAL_QUALITY -> AQMH_RECONSTRUCTION
```

Der Gewichtungsvektor wird für die pixelweise Rekonstruktion persistiert bzw.
im laufenden Prozess weitergereicht. Fehlen kompatible Maps oder sind die
Cache-Metadaten ungültig, darf keine neue globale Gewichtung aus einem
inkompatiblen Cache erzeugt werden.

## AQMH_DIAGNOSTICS (Phase 22)

Nach `AQMH_RECONSTRUCTION` werden blockweise Map- und Rekonstruktionsmetriken
berechnet. Die Phase verändert das rekonstruierten Bild nicht. Sie schreibt:

```text
artifacts/aqmh_metrics.json
artifacts/aqmh_heatmaps/*
```

Die Diagnostik ist vom Ergebnisweg getrennt: Ein erfolgreiches AQMH-Bild wird
nicht durch das Vorhandensein einer Heatmap ersetzt. Bei nicht verfügbarer
Diagnostik wird die Phase als `skipped` oder `error` protokolliert, entsprechend
dem aktuellen Runner-Vertrag.

## Übergang

```text
AQMH_MAPS (19)
  -> AQMH_GLOBAL_QUALITY (20)
  -> AQMH_RECONSTRUCTION (21)
  -> AQMH_DIAGNOSTICS (22)
  -> STACKING (12)
```
