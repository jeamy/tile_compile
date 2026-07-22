# AQMH — Adaptive Quality Map Hyperstacking

AQMH ist der Standard-Rekonstruktionspfad seit v0.3.0. Für jedes Input-Frame wird eine **dichte Qualitätskarte** `Q_map_{f,c}(x,y)` mit einer **multiskaligen Laplacian-Pyramide** berechnet, die Schärfe- und SNR-Metriken zu einem pixelgenauen Qualitätswert kombiniert. Das finale Bild wird als **pixelgenaues gewichteter Mittelwert** rekonstruiert — effektives Gewicht `W = G_{f,c} * Q_map_{f,c}(x,y)`, wobei `G_{f,c}` das globale Frame-Gewicht aus der gemeinsamen Vorverarbeitung ist. Kein Tile-Raster, keine OLA-Nähte.

> **Normative Spezifikation:** [AQMH-Methodik v0.2.1](../AQMH/aqmh_methodik_de_v0.2.1.md)

## Funktionsweise

```
Für jedes Frame f, Kanal c:
  Für jede Pyramiden-Skala s (D_s = 4^s, Fenster R_s = 4 px in herunterskalierten Pixeln):
    1. I_{f,c} um D_s herunterskalieren (maskenbewusster Flächenmittelwert)
    2. Pro Fenster berechnen:
         Phi_sharp = lokale Varianz des maskierten Laplacian (Schärfe)
         Phi_snr   = lokales SNR = mu / max(1.4826*MAD, eps)
         Phi_artifact = 1 - clip(outlier_frac / frac_artifact_max, 0, 1)
    3. Psi_s = sigmoid(w_sharp*z(Phi_sharp) + w_snr*z(Phi_snr)) * Phi_artifact
       (z = robuster z-Score; Artefakt-Gate ist multiplikativ — eine schlechte Skala vetoed den Pixel)
    4. Psi_s auf Canvas-Auflösung hochskalieren (maskenbewusstes Bilinear)
  Q_map_{f,c} = geometrisches Mittel über Skalen(Psi_s)  # alle Skalen müssen übereinstimmen
  Q_map auf Disk-Cache speichern (Standard: 1/2-Fläche uint16)

Rekonstruktion (pro Canvas-gültigem Pixel p):
  W_{f,c}(p) = G_{f,c} * Q_map_{f,c}(p)
  R_c(p) = sum_f( W_{f,c}(p) * I_{f,c}(p) ) / sum_f( W_{f,c}(p) )
```

## Wichtige Parameter (`aqmh.*`)

| Parameter | Standard | Beschreibung |
|-----------|----------|-------------|
| `aqmh.enabled` | `true` | AQMH aktivieren (false = klassische TILE_RECONSTRUCTION verwenden) |
| `aqmh.pyramid.scales` | `4` | Pyramiden-Level für multiskalige Analyse |
| `aqmh.pyramid.base_window_px` | `4` | Fenstergröße auf der untersten Pyramiden-Stufe |
| `aqmh.pyramid.w_sharp` | `0.6` | Schärfe-Gewicht im Qualitätsindex |
| `aqmh.pyramid.w_snr` | `0.4` | SNR-Gewicht im Qualitätsindex |
| `aqmh.pyramid.k_artifact` | `3.0` | MAD-Multiplikator für Artefakt-Erkennung (höher = toleranter) |
| `aqmh.pyramid.frac_artifact_max` | `0.25` | Max. Artefakt-Anteil pro Fenster vor Verwurf |
| `aqmh.storage.resolution_divisor` | `2` | Qualitätskarten-Cache-Auflösung (1/2/4) |
| `aqmh.storage.dtype` | `uint16` | Cache-Datentyp (`float32`, `uint16` oder `uint8`) |
| `aqmh.storage.max_resident_maps` | `2` | Max. Qualitätskarten gleichzeitig im RAM |
| `aqmh.cherry_pick.enabled` | `false` | Nur die besten Frames stapeln |
| `aqmh.cherry_pick.k_frac` | `0.30` | Anteil der besten Frames (0.30 = beste 30%) |
| `aqmh.cherry_pick.k_min_required` | `20` | Run-Gate und min. behaltene Samples pro Pixel |
| `aqmh.diagnostics.enabled` | `true` | AQMH-Diagnose-Phase aktivieren |
| `aqmh.diagnostics.level` | `full` | Detailgrad: `none`, `summary` oder `full` |
| `aqmh.diagnostics.format` | `json` | Diagnose-Ausgabeformat: `json` oder `binary` |
| `aqmh.reconstruction.chunk_rows` | `0` | Zeilen-Chunk-Größe (0 = auto aus Memory-Budget) |
| `aqmh.global_quality.g_k_scale` | `1.5` | Sigmoid-Temperatur; globales Gewicht bleibt begrenzt auf `[g_floor, 1]` |
| `aqmh.reconstruction.clip_sigma_low/high` | `2.0 / 1.5` | Asymmetrische untere/obere Clipping-Schwellen |
| `aqmh.reconstruction.clip_iterations` | `4` | AQMH-Clipping-Iterationen |

Vollständige Parameter-Dokumentation: [Konfigurationsreferenz — §12b AQMH](../configuration_reference.md)
Praktische Beispiele: [Konfigurationsbeispiele — AQMH-Abschnitt](../configuration_examples_practical_de.md)

## Wann AQMH vs. Klassisch verwenden

| Situation | Empfehlung |
|-----------|------------|
| Standard / die meisten Sessions | **AQMH** (standardmäßig aktiviert) |
| Tile-Nähte oder OLA-Artefakte sichtbar | **AQMH** eliminiert Nähte vollständig |
| Stark schwankende Frame-Qualität (Seeing, Wolken) | **AQMH** mit `cherry_pick.enabled: true`, `resolution_divisor: 1`, `dtype: float32` |
| Sehr große Sessions, RAM-begrenzt | **AQMH** mit `storage.resolution_divisor: 4`, `dtype: uint8` |
| Sessions mit Satellitenspuren / kosmetischen Problemen | **AQMH** mit `k_artifact: 5.0`, `frac_artifact_max: 0.35` |
| Forschung mit TBQR-Tile-gewichteter OLA | Klassisch (`aqmh.enabled: false`) |

## Minimale AQMH-Konfiguration

```yaml
aqmh:
  enabled: true          # Standard — kann weggelassen werden
  pyramid:
    k_artifact: 3.0      # Standard
    frac_artifact_max: 0.25  # Standard
```

## AQMH deaktivieren (auf klassisch zurücksetzen)

```yaml
aqmh:
  enabled: false
```

## AQMH-Papiere

- [AQMH v0.2.0-Papier](../AQMH/zenodo-0.2.0/paper-adaptive_quality_map_hyperstacking_m31_run_20260722_en.pdf) — M31-Validierungsrun mit v0.2.0-Erweiterungen
- [AQMH v0.1.0-Papier](../AQMH/zenodo-0.1.0/) — Original-Methodendefinition und M31-A-Validierungsrun
