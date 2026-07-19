# AQMH_MAPS / LOCAL_METRICS — Qualitätsanalyse

> **Aktueller Code:** `runner_phase_local_metrics.cpp`, `runner_phase_aqmh_maps.cpp`
> **AQMH-Phase:** `Phase::AQMH_MAPS` (Enum 19)
> **Classic-Phase:** `Phase::LOCAL_METRICS` (Enum 8)

## Einordnung

Nach `COMMON_OVERLAP` wird genau ein methodenabhängiger Qualitätsblock ausgeführt:

```text
AQMH:    PREWARP -> COMMON_OVERLAP -> AQMH_MAPS -> AQMH_GLOBAL_QUALITY
Classic: PREWARP -> COMMON_OVERLAP -> LOCAL_METRICS
```

`AQMH_MAPS` ist keine lokale Tile-Metrik-Phase und wird nicht als
`LOCAL_METRICS` emittiert.

## AQMH_MAPS

Für jedes registrierte und vorgewarpte Frame wird eine mehrskalige Quality-Map
berechnet. Die Berechnung verwendet die gemeinsame Canvas-Maske sowie die
Frame-Gültigkeitsmaske. Die Map wird im persistenten Cache
`<run>/cache/aqmh` gespeichert; die Frame-Masken liegen unter
`<run>/cache/aqmh_masks`.

Die Phase umfasst:

- Pyramidale Sharpness-/SNR-Auswertung über die konfigurierten Skalen
- Artefakt- und ungültige-Pixel-Maskierung
- deterministische Speicherung mit Cache-Metadaten, Masken-Hash und Backend
- optionales Prefetching der Maps für die nachfolgende AQMH-Rekonstruktion

Die eigentliche globale Frame-Gewichtung ist eine separate Phase:
`AQMH_GLOBAL_QUALITY` (Enum 20).

### Ergebnisse

```text
artifacts/aqmh_metrics.json       # Basis-/Map-Diagnostik
cache/aqmh/                       # Quality-Map-Cache
cache/aqmh_masks/                 # Frame-Gültigkeitsmasken
```

Ein gültiger Cache muss zu Canvas-Abmessungen, Pyramid-Konfiguration,
Storage-Typ, Masken-Hash und Backend-Metadaten passen. Ein veralteter oder
inkompatibler Cache darf nicht für Resume verwendet werden.

## LOCAL_METRICS (nur Classic)

Bei `method: classic_tile_compile` werden stattdessen lokale Metriken pro
`(frame, tile)` berechnet. Dazu gehören je nach Konfiguration FWHM,
Roundness/Elongation, Kontrast, Star Count und lokale Varianz. Diese Werte
bilden die lokalen Tile-Gewichte `L_f,t` für `TILE_RECONSTRUCTION`.

```text
artifacts/local_metrics.json
```

## Übergang

- AQMH: `AQMH_MAPS` -> `AQMH_GLOBAL_QUALITY` -> `AQMH_RECONSTRUCTION`
- Classic: `LOCAL_METRICS` -> `TILE_RECONSTRUCTION`

`STATE_CLUSTERING` und `SYNTHETIC_FRAMES` gehören ausschließlich zum
Classic-Zweig.
