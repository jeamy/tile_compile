# Tile-Compile Konfigurationsreferenz (Methodik v4)

Diese Dokumentation beschreibt alle Konfigurationsoptionen für `tile_compile.yaml` basierend auf Methodik v4.

**Schema-Version:** 4  
**Referenz:** Methodik v4 (doc/tile_basierte_qualitaetsrekonstruktion_methodik_v_4.md)

---

## Inhaltsverzeichnis

1. [Pipeline](#pipeline)
2. [Data](#data)
3. [Calibration (Kalibrierung)](#calibration-kalibrierung)
4. [Assumptions (Annahmen)](#assumptions-annahmen)
5. [v4 (Methodik v4 Parameter)](#v4-methodik-v4-parameter)
6. [Normalization (Normalisierung)](#normalization-normalisierung)
7. [Registration (Tile-Local Registration)](#registration-tile-local-registration)
8. [Wiener Denoise](#wiener-denoise)
9. [Global Metrics (Globale Metriken)](#global-metrics-globale-metriken)
10. [Tile (Kachel-Geometrie)](#tile-kachel-geometrie)
11. [Local Metrics (Lokale Metriken)](#local-metrics-lokale-metriken)
12. [Synthetic (Synthetische Frames)](#synthetic-synthetische-frames)
13. [Reconstruction (Rekonstruktion)](#reconstruction-rekonstruktion)
14. [Debayer](#debayer)
15. [Stacking](#stacking)
16. [Validation (Validierung)](#validation-validierung)
17. [Runtime Limits](#runtime-limits)

---

## Pipeline

Grundlegende Pipeline-Einstellungen.

Key: `pipeline`

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

## Data

Diese Felder sind im Schema als Teil von `data` definiert. In der Praxis werden sie oft durch Scan/Metadaten gesetzt (oder vom Benutzer vorgegeben), müssen aber für die Validierung vorhanden sein.

### `data.color_mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte (Schema)** | `OSC` |
| **Quelle** | Scan/Metadaten |
| **Editierbar** | Nein |

**Zweck:** Farbmodus der Kamera.

- **`OSC`** (One-Shot-Color): Farbkamera mit Bayer-Matrix (CFA)

---

### `data.bayer_pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `RGGB`, `BGGR`, `GBRG`, `GRBG` |
| **Quelle** | FITS-Header (BAYERPAT) |
| **Standard** | `GBRG` |
| **Editierbar** | Bestätigung möglich |

**Zweck:** Bayer-Muster für OSC/CFA-Kameras.

---

### `data.image_width` / `data.image_height`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Erforderlich** | Ja |

**Zweck:** Bildabmessungen in Pixeln.

---

### `data.frames_min`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Erforderlich** | Ja |

**Zweck:** Mindestanzahl Frames (für `data`/Vorlauf). Zusätzlich existiert `assumptions.frames_min` als methodischer Abbruch-Schwellenwert.

---

### `data.frames_target`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 0 |
| **Standard (Schema)** | 0 |

**Zweck:** Optionales Limit, wie viele Frames für Analyse-Schritte verwendet werden (0 = alle).

---

### `data.linear_required`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Konstant** | true |

**Zweck:** Methodik v4 erfordert lineare Daten.

---

## Calibration (Kalibrierung)

Kalibrierungs-Parameter (Bias/Dark/Flat). Diese werden in der GUI zusätzlich aus UI-Feldern in das YAML injiziert.

Key: `calibration`

### `calibration.use_bias` / `calibration.use_dark` / `calibration.use_flat`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Erforderlich** | Ja |

**Zweck:** Aktiviert die jeweiligen Kalibrierungsstufen.

---

### `calibration.bias_use_master` / `calibration.dark_use_master` / `calibration.flat_use_master`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard (Schema)** | false |

**Zweck:** Wenn `true`, wird ein bestehendes Master-File verwendet statt es aus einem Verzeichnis zu bauen.

---

### `calibration.dark_auto_select`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard (Schema)** | true |

**Zweck:** Auto-Auswahl eines passenden Dark-Subsets bei gemischten Darks (Exposure/Temp) bevor ein Master Dark gebaut wird.

---

### `calibration.dark_match_exposure_tolerance_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard (Schema)** | 5.0 |

**Zweck:** Erlaubte Abweichung der Belichtungszeit (in %) zwischen Lights und ausgewählten Darks.

---

### `calibration.dark_match_use_temp` / `calibration.dark_match_temp_tolerance_c`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean / number |
| **Standard (Schema)** | false / 2.0 |

**Zweck:** Optionales Temperature-Matching für Darks.

---

### `calibration.bias_dir` / `calibration.darks_dir` / `calibration.flats_dir`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Standard (Schema)** | "" |

**Zweck:** Verzeichnisse für Kalibrierungsframes (wenn nicht per Master-File).

---

### `calibration.bias_master` / `calibration.dark_master` / `calibration.flat_master`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Standard (Schema)** | "" |

**Zweck:** Pfade zu Master-Kalibrierungsfiles.

---

### `calibration.pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Standard (tile_compile.yaml)** | `*.fit*` |

**Zweck:** Glob-Pattern, mit dem Bias/Dark/Flat Frames in den Verzeichnissen gesammelt werden.

---

## Assumptions (Annahmen)

Konfigurierbare Schwellenwerte und Annahmen gemäß Methodik v4.

Key: `assumptions`

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

### `assumptions.exposure_time_tolerance_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard (Schema)** | 5.0 |

**Zweck:** Toleranz (in %) für Belichtungszeit-Abweichungen bei Annahmen/Matching.

---

### `assumptions.tracking_error_max_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard (Schema)** | 5.0 |

**Zweck:** Maximal tolerierter Tracking-Fehler (in Pixeln) als Plausibilitäts-/Abbruchkriterium.

---

### `assumptions.reduced_mode_skip_clustering`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard (Schema)** | false |

**Zweck:** Wenn `true`, wird Clustering im Reduced-Mode übersprungen.

---

### `assumptions.reduced_mode_cluster_range`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 integers] |
| **Standard (Schema)** | [2, 6] |

**Zweck:** Cluster-Anzahlbereich im Reduced-Mode.

---

## Linearity

Optionaler Linearitäts-Check.

Key: `linearity`

### `linearity.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard (Schema)** | false |

---

### `linearity.strictness`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte (Schema)** | `soft`, `hard` |
| **Standard (Schema)** | `soft` |

---

### `linearity.min_overall_linearity`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Standard (Schema)** | 0.95 |

---

### `linearity.max_frames`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard (Schema)** | 2000 |

---

## v4 (Methodik v4 Parameter)

Zentrale Konfiguration für Tile-wise Local Registration (TLR).

Key: `v4`

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

### `v4.parallel_tiles`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 1 - 32 |
| **Standard (tile_compile.yaml)** | 8 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Anzahl paralleler Tile-Worker.

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

#### `v4.adaptive_tiles.use_warp_probe`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard (tile_compile.yaml)** | true |
| **Editierbar** | Ja |

**Zweck:** Aktiviert die optionale **Warp Probe** in der Phase `TILE_GRID`.

**Bedeutung:** Vor dem finalen Tile-Grid wird mit wenigen Frames ein grobes Warp-/Gradient-Feld geschätzt, um die Tile-Dichte positionsabhängig anzupassen.

---

#### `v4.adaptive_tiles.use_hierarchical`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard (tile_compile.yaml)** | true |
| **Editierbar** | Ja |

**Zweck:** Hierarchische Tile-Initialisierung (coarse → split wo nötig).

---

#### `v4.adaptive_tiles.initial_tile_size`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 64 - 512 |
| **Standard (tile_compile.yaml)** | 256 |

**Zweck:** Start-Tile-Größe für hierarchische Initialisierung.

---

#### `v4.adaptive_tiles.probe_window`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 64 - 512 |
| **Standard (tile_compile.yaml)** | 256 |

**Zweck:** Fenstergröße für Warp-Probe.

---

#### `v4.adaptive_tiles.num_probe_frames`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 3 - 10 |
| **Standard (tile_compile.yaml)** | 5 |

**Zweck:** Anzahl zeitlich verteilter Frames für die Warp-Probe.

---

#### `v4.adaptive_tiles.gradient_sensitivity`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.1 - 10.0 |
| **Standard (tile_compile.yaml)** | 2.0 |

**Zweck:** Sensitivität der Tile-Größenanpassung anhand des Warp-Gradienten: `s(x,y) = s0 / (1 + c·grad)`.

---

#### `v4.adaptive_tiles.split_gradient_threshold`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.0 - 1.0 |
| **Standard (tile_compile.yaml)** | 0.3 |

**Zweck:** Schwellwert für hierarchisches Splitting anhand des Gradienten.

---

#### `v4.adaptive_tiles.hierarchical_max_depth`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 1 - 5 |
| **Standard (tile_compile.yaml)** | 3 |

**Zweck:** Maximale Rekursionstiefe beim hierarchischen Splitting.

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

#### `v4.diagnostics.warp_field`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard (tile_compile.yaml)** | true |
| **Editierbar** | Ja |

**Zweck:** Speichert Warp-Vektorfelder (Diagnose).

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

### `v4.debug_tile_registration`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard (tile_compile.yaml)** | true |

**Zweck:** Aktiviert ausführliche Debug-Ausgaben zur Tile-Registration (Konsistenzprüfungen).

---

## Normalization (Normalisierung)

Pflicht-Einstellungen gemäß Methodik v4.

Key: `normalization`

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

Key: `registration`

Sub-Key: `registration.local_tiles`

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

### `registration.local_tiles.max_warp_delta_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.0 - 10.0 |
| **Standard (tile_compile.yaml)** | 0.3 |
| **Editierbar** | Ja |

**Zweck:** Maximale erlaubte Abweichung der Frame-Translation vom Median (in Pixeln). Frames darüber werden für das Tile verworfen.

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

## Wiener Denoise

Optionaler Wiener-Filter auf rekonstruierten Tiles (nach Rekonstruktion, vor Overlap-Add).

Key: `wiener_denoise`

### `wiener_denoise.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | false |
| **Editierbar** | Ja |

**Zweck:** Aktiviert Wiener-Denoising.

---

### `wiener_denoise.snr_threshold`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich (Schema)** | 1.0 - 20.0 |
| **Standard (Schema)** | 5.0 |

**Zweck:** Tiles mit SNR >= Schwelle werden nicht gefiltert.

---

### `wiener_denoise.q_min`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.0 - 1.0 |
| **Standard** | 0.5 |
| **Editierbar** | Ja |

**Zweck:** Minimaler Wiener-Parameter.

---

### `wiener_denoise.q_max`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.0 - 1.0 |
| **Standard** | 1.0 |
| **Editierbar** | Ja |

**Zweck:** Maximaler Wiener-Parameter.

---

### `wiener_denoise.q_step`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.0 - 1.0 |
| **Standard** | 0.1 |
| **Editierbar** | Ja |

**Zweck:** Schrittweite für Wiener-Parameter.

---

### `wiener_denoise.min_snr`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.0 - 10.0 |
| **Standard** | 2.0 |
| **Editierbar** | Ja |

**Zweck:** Minimaler SNR für Wiener-Denoising.

---

### `wiener_denoise.max_iterations`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard** | 10 |
| **Editierbar** | Ja |

**Zweck:** Maximale Anzahl Iterationen für Wiener-Denoising.

---

## Global Metrics (Globale Metriken)

Gewichtung der globalen Frame-Qualitätsmetriken gemäß Methodik v4 §6.

Key: `global_metrics`

### `global_metrics.adaptive_weights`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard (Schema)** | false |

**Zweck:** Aktiviert adaptive Gewichtung der globalen Metriken (statt fixer `global_metrics.weights.*`).

### `global_metrics.weights`

| Komponente | Standard | Beschreibung |
|------------|----------|--------------|
| **`global_metrics.weights.background`** | 0.4 | Gewicht für Hintergrund-Metrik |
| **`global_metrics.weights.noise`** | 0.3 | Gewicht für Rausch-Metrik |
| **`global_metrics.weights.gradient`** | 0.3 | Gewicht für Gradienten-Metrik |

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

Key: `tile`

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

### `tile.star_min_count`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard** | 10 |
| **Editierbar** | Ja |

**Zweck:** Mindestanzahl erkannter Sterne pro Tile, um Star-Stats zu berechnen.

---

## Local Metrics (Lokale Metriken)

Lokale Qualitätsmetriken und Gewichtung (wird in v4 im Rahmen der Tile-Verarbeitung genutzt).

Key: `local_metrics`

Required Keys:

- `local_metrics.star_mode`
- `local_metrics.structure_mode`

### `local_metrics.clamp`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 numbers] |
| **Standard (tile_compile.yaml)** | [-3, 3] |

**Zweck:** Clamp-Bereich für Q_local vor exp(·).

---

### `local_metrics.star_mode.weights`

| Komponente | Standard (tile_compile.yaml) |
|------------|------------------------------|
| **`local_metrics.star_mode.weights.fwhm`** | 0.6 |
| **`local_metrics.star_mode.weights.roundness`** | 0.2 |
| **`local_metrics.star_mode.weights.contrast`** | 0.2 |

**Zweck:** Gewichtung der Stern-basierten lokalen Metrik. Constraint: Summe = 1.0 (Backend-Check).

---

### `local_metrics.structure_mode.background_weight` / `local_metrics.structure_mode.metric_weight`

| Komponente | Standard (tile_compile.yaml) |
|------------|------------------------------|
| **background_weight** | 0.3 |
| **metric_weight** | 0.7 |

**Zweck:** Gewichtung des Struktur-Modus. Constraint: Summe = 1.0 (Backend-Check).

---

## Synthetic (Synthetische Frames)

Clustering und synthetische Frame-Erzeugung gemäß Methodik v4 §10.

Key: `synthetic`

Required Keys:

- `synthetic.clustering`
- `synthetic.clustering.mode`
- `synthetic.clustering.vector`

### `synthetic.weighting`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `global`, `tile_weighted` |
| **Standard (tile_compile.yaml)** | `tile_weighted` |
| **Editierbar** | Ja |

**Zweck:** Bestimmt, wie synthetische Frames pro Cluster gebildet werden.

- **`global`**: Nur globale Gewichte G_f,c
- **`tile_weighted`**: Tile-basiert mit W_f,t,c = G_f,c × L_f,t,c (empfohlen)

---

### `synthetic.clustering.cluster_count_range`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 integer] |
| **Standard (Schema)** | [15, 30] |

**Zweck:** Erlaubter k-Bereich für Clustering (wird i.d.R. automatisch über Silhouette ausgewählt).

---

### `synthetic.clustering.k_selection`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte (Schema)** | `auto`, `fixed` |
| **Standard (Schema)** | `auto` |

**Zweck:** Strategie zur Auswahl von K innerhalb von `synthetic.clustering.cluster_count_range`.

---

## Reconstruction (Rekonstruktion)

Diese Sektion ist im Schema als `reconstruction` definiert und aktuell methodisch fest verdrahtet.

### `reconstruction.weighting_function`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Konstant** | `exponential` |

---

### `reconstruction.window_function`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Konstant** | `hanning` |

---

### `reconstruction.tile_rescale`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Konstant** | `median_after_background_subtraction` |

---

### `synthetic.frames_min` / `synthetic.frames_max`

| Eigenschaft | Wert |
|-------------|------|
| **Standard (tile_compile.yaml)** | 15 / 30 |

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

Key: `stacking`

### `stacking.method`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `average`, `rej` |
| **Standard (tile_compile.yaml)** | `rej` |

**Zweck:** Stacking-Methode.

- **`rej`**: Sigma-Clipping Rejection (empfohlen)

---

### `stacking.input_dir`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Erforderlich** | Ja |
| **Standard (tile_compile.yaml)** | `synthetic` |

**Zweck:** Eingabe-Verzeichnis relativ zum Run-Output (z.B. `synthetic`).

---

### `stacking.input_pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Erforderlich** | Ja |
| **Standard (tile_compile.yaml)** | `syn_*.fits` |

**Zweck:** Glob-Pattern der zu stackenden Frames.

---

### `stacking.output_file`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Erforderlich** | Ja |
| **Standard (tile_compile.yaml)** | `stacked.fit` |

**Zweck:** Name des finalen Stack-Outputs.

---

### `stacking.sigma_clip`

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| **`stacking.sigma_clip.sigma_low`** | 2.0 | Unterer Sigma-Schwellenwert |
| **`stacking.sigma_clip.sigma_high`** | 2.0 | Oberer Sigma-Schwellenwert |
| **`stacking.sigma_clip.max_iters`** | 3 | Maximale Iterationen |
| **`stacking.sigma_clip.min_fraction`** | 0.5 | Minimale überlebende Frame-Fraktion |

---

## Validation (Validierung)

Qualitätsprüfungen gemäß Methodik v4 §12.

Key: `validation`

### `validation.min_fwhm_improvement_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Standard** | 5 |

**Zweck:** Minimale FWHM-Verbesserung in Prozent.

---

### `validation.max_background_rms_increase_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Standard (tile_compile.yaml)** | 0 |

**Zweck:** Maximal erlaubte Zunahme des Hintergrund-RMS in Prozent.

---

### `validation.min_tile_weight_variance`

| Eigenschaft | Wert |
|-------------|------|
| **Standard (tile_compile.yaml)** | 0.1 |

**Zweck:** Minimale Varianz der Tile-Gewichte (Schutz gegen degenerierte Gewichtung).

---

### `validation.require_no_tile_pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Standard** | true |

**Zweck:** Prüft auf sichtbare Kachel-Muster im Endergebnis.

---

## Runtime Limits

### `runtime_limits.tile_analysis_max_factor_vs_stack`

Key: `runtime_limits`

| Eigenschaft | Wert |
|-------------|------|
| **Standard (tile_compile.yaml)** | 3.0 |

**Zweck:** Limit-Faktor für Analyse-/Tile-Phasen relativ zur erwarteten Stack-Zeit.

---

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
