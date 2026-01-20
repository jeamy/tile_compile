# Tile-Compile Konfigurationsreferenz (Methodik v4)

Diese Dokumentation beschreibt alle Konfigurationsoptionen für `tile_compile.yaml` basierend auf Methodik v4.

**Schema-Version:** 4  
**Referenz:** Methodik v4 (doc/tile_basierte_qualitaetsrekonstruktion_methodik_v_4.md)

---

## Inhaltsverzeichnis

1. [Pipeline](#pipeline)
2. [Input (automatisch ermittelt)](#input-automatisch-ermittelt)
3. [Assumptions (Annahmen)](#assumptions-annahmen)
4. [v4 (Methodik v4 Parameter)](#v4-methodik-v4-parameter)
5. [Normalization (Normalisierung)](#normalization-normalisierung)
6. [Registration (Tile-Local Registration)](#registration-tile-local-registration)
7. [Global Metrics (Globale Metriken)](#global-metrics-globale-metriken)
8. [Tile (Kachel-Geometrie)](#tile-kachel-geometrie)
9. [Synthetic (Synthetische Frames)](#synthetic-synthetische-frames)
10. [Debayer](#debayer)
11. [Stacking](#stacking)
12. [Validation (Validierung)](#validation-validierung)
13. [Runtime Limits](#runtime-limits)

---

## Pipeline

Grundlegende Pipeline-Einstellungen.

### `pipeline.mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `production`, `test` |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Bestimmt den Ausführungsmodus der Pipeline.

- **`production`**: Vollständige Verarbeitung mit allen Qualitätsprüfungen
- **`test`**: Reduzierte Verarbeitung für schnelle Tests

---

### `pipeline.abort_on_fail`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Bestimmt, ob die Pipeline bei Fehlern abgebrochen wird.

- **`true`**: Pipeline stoppt sofort bei kritischen Fehlern
- **`false`**: Pipeline versucht fortzufahren (nicht empfohlen)

---

## Input (automatisch ermittelt)

Diese Felder werden automatisch aus den FITS-Headern und dem Dateisystem ermittelt. Sie sind **nicht editierbar**, können aber vor dem Run bestätigt werden.

### `input.color_mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `OSC`, `MONO` |
| **Quelle** | FITS-Header |
| **Editierbar** | Bestätigung möglich |

**Zweck:** Farbmodus der Kamera.

- **`OSC`** (One-Shot-Color): Farbkamera mit Bayer-Matrix (CFA)
- **`MONO`**: Monochrome Kamera ohne Farbfilter

---

### `input.bayer_pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `RGGB`, `BGGR`, `GBRG`, `GRBG` |
| **Quelle** | FITS-Header (BAYERPAT) |
| **Standard** | `GBRG` |
| **Editierbar** | Bestätigung möglich |

**Zweck:** Bayer-Muster für OSC/CFA-Kameras.

---

## Assumptions (Annahmen)

Konfigurierbare Schwellenwerte und Annahmen gemäß Methodik v4.

### `assumptions.frames_min`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard** | 50 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Minimale Anzahl an Frames für die Verarbeitung (Hard Assumption).

**Verhalten:** Unterschreitung führt zum **Abbruch** der Pipeline.

---

### `assumptions.frames_optimal`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard** | 800 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Optimale Frame-Anzahl für die vollständige Methodik.

---

### `assumptions.frames_reduced_threshold`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard** | 200 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Schwellenwert für den Reduced Mode.

**Verhalten:**
- `frames < frames_min`: Abbruch
- `frames_min ≤ frames < frames_reduced_threshold`: Reduced Mode (STATE_CLUSTERING und SYNTHETIC_FRAMES werden übersprungen)
- `frames ≥ frames_reduced_threshold`: Normal Mode

---

### `assumptions.warp_variance_warn`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard** | 2.0 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Schwellenwert für Warp-Varianz-Warnung (Methodik v4 §4).

**Verhalten:** Überschreitung erzeugt eine Warnung, aber keinen Abbruch.

---

### `assumptions.warp_variance_max`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard** | 8.0 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Maximale erlaubte Warp-Varianz pro Tile (Methodik v4 §4).

**Verhalten:** Überschreitung markiert Tile als ungültig.

---

### `assumptions.elongation_warn` / `assumptions.elongation_max`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 1 |
| **Standard** | 0.3 / 0.4 |

**Zweck:** Schwellenwerte für Stern-Elongation.

**Definition:** Elongation = 1 - (minor_axis / major_axis). 0 = perfekt rund, 1 = Linie.

---

## v4 (Methodik v4 Parameter)

Zentrale Konfiguration für Tile-wise Local Registration (TLR).

### `v4.iterations`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 1 - 10 |
| **Standard** | 3 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Anzahl der iterativen Referenz-Verfeinerungen pro Tile (Methodik v4 §5.2).

**Empfehlung:**
- **2**: EQ-Montierung, ruhiges Seeing
- **4**: Alt/Az, starke Feldrotation (DEFAULT)
- **5**: Polnähe, sehr instabil

---

### `v4.beta`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.0 - 20.0 |
| **Standard** | 5.0 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Registrierungsgüte-Gewichtung (Methodik v4 §7).

**Formel:** R_{f,t} = exp(β·(cc-1))

**Empfehlung:**
- **3.0**: Ruhige Bedingungen
- **6.0**: Starke Feldrotation (DEFAULT)
- **8.0**: Sehr instabile Bedingungen

---

### `v4.min_valid_tile_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.0 - 1.0 |
| **Standard** | 0.3 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Minimale Fraktion gültiger Tiles für erfolgreiche Rekonstruktion (Methodik v4 §12).

**Verhalten:** Wenn < 30% der Tiles gültig sind, wird die Pipeline abgebrochen.

---

### `v4.adaptive_tiles`

Adaptive Tile-Verfeinerung basierend auf Warp-Varianz (Methodik v4 §4).

#### `v4.adaptive_tiles.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | false |
| **Editierbar** | Ja |

**Zweck:** Aktiviert adaptive Tile-Verfeinerung.

**Empfehlung:**
- **false**: EQ-Montierung, ruhiges Seeing
- **true**: Alt/Az, Feldrotation, instabile Bedingungen

---

#### `v4.adaptive_tiles.max_refine_passes`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 0 - 5 |
| **Standard** | 2 |
| **Editierbar** | Ja |

**Zweck:** Maximale Anzahl von Refinement-Pässen.

**Empfehlung:**
- **2-3**: Normale Feldrotation
- **4**: Sehr instabile Bedingungen

---

#### `v4.adaptive_tiles.refine_variance_threshold`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.0 - 10.0 |
| **Standard** | 0.25 |
| **Editierbar** | Ja |

**Zweck:** Warp-Varianz-Schwelle für Tile-Splitting.

**Formel:** Var(dx) + Var(dy)

**Empfehlung:**
- **0.15**: Starke Feldrotation
- **0.1**: Sehr instabile Bedingungen

---

#### `v4.adaptive_tiles.min_tile_size_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 32 - 512 |
| **Standard** | 64 |
| **Editierbar** | Ja |

**Zweck:** Minimale Tile-Größe in Pixeln (kein Split unterhalb dieser Größe).

---

### `v4.convergence`

Konvergenz-Check für Early Stopping.

#### `v4.convergence.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | false |
| **Editierbar** | Ja |

**Zweck:** Aktiviert Konvergenz-Check für vorzeitigen Abbruch der Iteration.

---

#### `v4.convergence.epsilon_rel`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 1.0e-6 - 1.0e-1 |
| **Standard** | 1.0e-3 |
| **Editierbar** | Ja |

**Zweck:** Relative L2-Norm-Schwelle für Konvergenz.

---

### `v4.memory_limits`

Speicher-Überwachung und -Limits.

#### `v4.memory_limits.rss_warn_mb`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 512 |
| **Standard** | 4096 |
| **Editierbar** | Ja |

**Zweck:** Weiches Speicherlimit (MB) - nur Warnung.

---

#### `v4.memory_limits.rss_abort_mb`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1024 |
| **Standard** | 8192 |
| **Editierbar** | Ja |

**Zweck:** Hartes Speicherlimit (MB) - Abbruch bei Überschreitung.

---

### `v4.diagnostics`

Diagnose-Artefakt-Generierung.

#### `v4.diagnostics.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | true |
| **Editierbar** | Ja |

**Zweck:** Aktiviert Diagnose-Artefakt-Generierung.

---

#### `v4.diagnostics.tile_invalid_map`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | true |
| **Editierbar** | Ja |

**Zweck:** Speichert FITS-Map ungültiger Tiles (`tile_invalid_map.fits`).

---

#### `v4.diagnostics.warp_variance_hist`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | true |
| **Editierbar** | Ja |

**Zweck:** Speichert Warp-Varianz-Statistiken (`warp_variance_stats.json`).

---

## Normalization (Normalisierung)

Pflicht-Einstellungen gemäß Methodik v4.

### `normalization.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Konstant** | true |
| **Editierbar** | Nein |

**Zweck:** Normalisierung ist **immer aktiviert** (Methodik-Zwang).

---

### `normalization.mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `background`, `median` |
| **Standard** | `background` |
| **Editierbar** | Ja |

**Zweck:** Normalisierungsmethode.

- **`background`**: Division durch geschätzten Hintergrundwert (empfohlen)
- **`median`**: Division durch Median des gesamten Frames

---

### `normalization.per_channel`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | true |
| **Editierbar** | Ja |

**Zweck:** Kanalweise Normalisierung (empfohlen für OSC).

---

## Registration (Tile-Local Registration)

Einstellungen für Tile-wise Local Registration (TLR) gemäß Methodik v4 §5.

### `registration.mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Konstant** | `local_tiles` |
| **Editierbar** | Nein |

**Zweck:** Methodik v4 erzwingt tile-lokale Registrierung (keine globale Registrierung).

---

### `registration.local_tiles.ecc_cc_min`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.0 - 1.0 |
| **Standard** | 0.2 |
| **Editierbar** | Ja |

**Zweck:** Minimale ECC-Korrelation für gültige lokale Registrierung.

---

### `registration.local_tiles.min_valid_frames`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard** | 10 |
| **Editierbar** | Ja |

**Zweck:** Mindestanzahl gültiger Frames pro Tile.

---

### `registration.local_tiles.temporal_smoothing_window`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 3 |
| **Standard** | 11 |
| **Editierbar** | Ja |

**Zweck:** Savitzky-Golay-Fenster für temporale Warp-Glättung (muss ungerade sein).

---

### `registration.local_tiles.variance_window_sigma`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0.0 |
| **Standard** | 2.0 |
| **Editierbar** | Ja |

**Zweck:** ψ(var) Skalierungsparameter für Varianz-Fenster (Methodik v4 §9).

---

## Global Metrics (Globale Metriken)

Gewichtung der globalen Frame-Qualitätsmetriken gemäß Methodik v4 §6.

### `global_metrics.weights`

| Komponente | Standard | Beschreibung |
|------------|----------|--------------|
| **background** | 0.4 | Gewicht für Hintergrund-Metrik |
| **noise** | 0.3 | Gewicht für Rausch-Metrik |
| **gradient** | 0.3 | Gewicht für Gradienten-Metrik |

**Formel:** Q_f = α·(-B̃) + β·(-σ̃) + γ·Ẽ

**Constraint:** α + β + γ = 1.0

---

### `global_metrics.clamp`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 numbers] |
| **Standard** | [-3, 3] |

**Zweck:** Clamp-Bereich für Q_f vor der Exponentialfunktion.

**Formel:** G_f = exp(clamp(Q_f, -3, 3))

---

## Tile (Kachel-Geometrie)

Seeing-adaptive Kachel-Erzeugung gemäß Methodik v4 §3.

### `tile.size_factor`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Standard** | 32 |
| **Editierbar** | Ja |

**Zweck:** Faktor s für die Kachelgröße.

**Formel:** T_0 = s × FWHM

---

### `tile.min_size`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Standard** | 64 |
| **Editierbar** | Ja |

**Zweck:** Minimale Kachelgröße T_min in Pixeln.

---

### `tile.max_divisor`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Standard** | 6 |
| **Editierbar** | Ja |

**Zweck:** Maximaler Divisor D für die Kachelgröße.

**Formel:** T_max = min(Breite, Höhe) / D

---

### `tile.overlap_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 0.5 |
| **Standard** | 0.25 |
| **Editierbar** | Ja |

**Zweck:** Überlappungsfraktion o für Kacheln.

---

## Synthetic (Synthetische Frames)

Clustering und synthetische Frame-Erzeugung gemäß Methodik v4 §10.

### `synthetic.weighting`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `global`, `tile_weighted` |
| **Standard** | `tile_weighted` |
| **Editierbar** | Ja |

**Zweck:** Bestimmt, wie synthetische Frames pro Cluster gebildet werden.

- **`global`**: Nur globale Gewichte G_f,c
- **`tile_weighted`**: Tile-basiert mit W_f,t,c = G_f,c × L_f,t,c (empfohlen)

---

### `synthetic.frames_min` / `synthetic.frames_max`

| Eigenschaft | Wert |
|-------------|------|
| **Standard** | 15 / 30 |

**Zweck:** Minimale/Maximale Anzahl synthetischer Frames.

---

## Debayer

### `debayer`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | true |
| **Editierbar** | Ja |

**Zweck:** Debayer/Demosaic des finalen gestackten CFA-Mosaiks.

- **`true`**: Erzeugt `stacked_rgb.fits`
- **`false`**: Nur CFA-Mosaik `stacked.fits`

---

## Stacking

Finales Stacking gemäß Methodik v4 §11.

### `stacking.method`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `average`, `rej` |
| **Standard** | `rej` |

**Zweck:** Stacking-Methode.

- **`rej`**: Sigma-Clipping Rejection (empfohlen)

---

### `stacking.sigma_clip`

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| **sigma_low** | 2.0 | Unterer Sigma-Schwellenwert |
| **sigma_high** | 2.0 | Oberer Sigma-Schwellenwert |
| **max_iters** | 3 | Maximale Iterationen |
| **min_fraction** | 0.5 | Minimale überlebende Frame-Fraktion |

---

## Validation (Validierung)

Qualitätsprüfungen gemäß Methodik v4 §12.

### `validation.min_fwhm_improvement_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Standard** | 5 |

**Zweck:** Minimale FWHM-Verbesserung in Prozent.

---

### `validation.require_no_tile_pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Standard** | true |

**Zweck:** Prüft auf sichtbare Kachel-Muster im Endergebnis.

---

## Runtime Limits

### `runtime_limits.hard_abort_hours`

| Eigenschaft | Wert |
|-------------|------|
| **Standard** | 6 |

**Zweck:** Maximale Laufzeit in Stunden.

---

## v4 Presets

Siehe `doc/v_4_example_configs.md` für vorkonfigurierte Presets:

1. **EQ-Montierung, ruhiges Seeing**: iterations=2, beta=3.0, adaptive=off
2. **Alt/Az, starke Feldrotation**: iterations=4, beta=6.0, adaptive=on (DEFAULT)
3. **Polnähe, sehr instabil**: iterations=5, beta=8.0, adaptive=on, aggressive

---

## Beispiel-Konfiguration (Preset 2)

```yaml
pipeline:
  mode: production
  abort_on_fail: true

v4:
  iterations: 4
  beta: 6.0
  adaptive_tiles:
    enabled: true
    max_refine_passes: 3
    refine_variance_threshold: 0.15

registration:
  mode: local_tiles
  local_tiles:
    ecc_cc_min: 0.2
    min_valid_frames: 10

stacking:
  method: rej
  sigma_clip:
    sigma_low: 2
    sigma_high: 2
```
