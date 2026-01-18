# Tile-Compile Konfigurationsreferenz

Diese Dokumentation beschreibt alle Konfigurationsoptionen für `tile_compile.yaml` basierend auf dem Schema `tile_compile.schema.yaml`.

**Schema-Version:** 3  
**Referenz:** Methodik v3 (doc/tile_basierte_qualitatsrekonstruktion_methodik_v_3.md)

---

## Inhaltsverzeichnis

1. [Pipeline](#pipeline)
2. [Input (automatisch ermittelt)](#input-automatisch-ermittelt)
3. [Assumptions (Annahmen)](#assumptions-annahmen)
4. [Normalization (Normalisierung)](#normalization-normalisierung)
5. [Registration (Registrierung)](#registration-registrierung)
6. [Tile Denoising (Tile-Rauschfilter)](#tile-denoising-tile-rauschfilter)
7. [Global Metrics (Globale Metriken)](#global-metrics-globale-metriken)
8. [Tile (Kachel-Geometrie)](#tile-kachel-geometrie)
9. [Local Metrics (Lokale Metriken)](#local-metrics-lokale-metriken)
10. [Synthetic (Synthetische Frames)](#synthetic-synthetische-frames)
11. [Reconstruction (Rekonstruktion)](#reconstruction-rekonstruktion)
12. [Debayer](#debayer)
13. [Stacking](#stacking)
14. [Validation (Validierung)](#validation-validierung)
15. [Runtime Limits](#runtime-limits)

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

### `input.image_width`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Quelle** | FITS-Header (NAXIS1) |
| **Editierbar** | Nein |

**Zweck:** Bildbreite in Pixeln, automatisch aus dem FITS-Header gelesen.

---

### `input.image_height`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Quelle** | FITS-Header (NAXIS2) |
| **Editierbar** | Nein |

**Zweck:** Bildhöhe in Pixeln, automatisch aus dem FITS-Header gelesen.

---

### `input.frames_detected`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Quelle** | Dateisystem |
| **Editierbar** | Nein |

**Zweck:** Anzahl der erkannten FITS-Dateien im Input-Verzeichnis.

---

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

**Hinweis:** Wird aus dem FITS-Header abgeleitet, kann aber vor dem Run bestätigt/korrigiert werden.

---

### `input.bayer_pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `RGGB`, `BGGR`, `GBRG`, `GRBG` |
| **Quelle** | FITS-Header (BAYERPAT) |
| **Standard** | `GBRG` |
| **Editierbar** | Bestätigung möglich |

**Zweck:** Bayer-Muster für OSC/CFA-Kameras. Bestimmt die Anordnung der Farbfilter auf dem Sensor.

```
RGGB:       BGGR:       GBRG:       GRBG:
R G R G     B G B G     G B G B     G R G R
G B G B     G R G R     R G R G     B G B G
```

**Wichtig:** Ein falsches Bayer-Pattern führt zu falschen Farben im Endergebnis.

---

## Assumptions (Annahmen)

Konfigurierbare Schwellenwerte und Annahmen gemäß Methodik v3 §2.

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

**Verhalten:** Bei weniger Frames wird die Qualität reduziert, aber die Verarbeitung fortgesetzt.

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
- `frames_min ≤ frames < frames_reduced_threshold`: Reduced Mode
- `frames ≥ frames_reduced_threshold`: Normal Mode

---

### `assumptions.exposure_time_tolerance_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 100 |
| **Standard** | 5 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Maximale erlaubte Abweichung der Belichtungszeit in Prozent (Hard Assumption).

**Beispiel:** Bei 5% und 10s Belichtung sind 9.5s - 10.5s erlaubt.

---

### `assumptions.registration_residual_warn_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard** | 0.5 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Schwellenwert für Registrierungs-Residuum-Warnung in Pixeln.

**Verhalten:** Überschreitung erzeugt eine Warnung, aber keinen Abbruch.

---

### `assumptions.registration_residual_max_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard** | 1.0 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Maximales erlaubtes Registrierungs-Residuum in Pixeln.

**Verhalten:** Überschreitung führt zum **Abbruch**.

---

### `assumptions.elongation_warn`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 1 |
| **Standard** | 0.3 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Schwellenwert für Stern-Elongation-Warnung.

**Definition:** Elongation = 1 - (minor_axis / major_axis). 0 = perfekt rund, 1 = Linie.

---

### `assumptions.elongation_max`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 1 |
| **Standard** | 0.4 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Maximale erlaubte Stern-Elongation.

**Verhalten:** Überschreitung führt zum **Abbruch** (deutet auf Tracking-Probleme hin).

---

### `assumptions.tracking_error_max_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard** | 1.0 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Maximaler erlaubter Tracking-Fehler pro Belichtung in Pixeln (implizite Annahme aus Methodik v3 §1.3).

---

### `assumptions.reduced_mode_skip_clustering`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | true |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Überspringt STATE_CLUSTERING und SYNTHETIC_FRAMES im Reduced Mode.

**Empfehlung:** `true` für schnellere Verarbeitung bei wenigen Frames.

---

### `assumptions.reduced_mode_cluster_range`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 integers] |
| **Standard** | [5, 10] |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Cluster-Anzahl-Bereich für Reduced Mode (falls Clustering nicht übersprungen wird).

---

## Normalization (Normalisierung)

Pflicht-Einstellungen gemäß Methodik v3 §4.

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
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Normalisierungsmethode.

- **`background`**: Division durch geschätzten Hintergrundwert (empfohlen)
- **`median`**: Division durch Median des gesamten Frames

---

### `normalization.per_channel`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Kanalweise Normalisierung.

- **`true`**: Jeder Farbkanal (R, G, B) wird separat normalisiert (empfohlen)
- **`false`**: Alle Kanäle werden gemeinsam normalisiert

---

## Registration (Registrierung)

Einstellungen für die geometrische Ausrichtung gemäß Methodik v3 §3.

### `registration.engine`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `siril`, `opencv_cfa` |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Registrierungs-Engine.

| Engine | Beschreibung | Empfehlung |
|--------|--------------|------------|
| **`opencv_cfa`** | ECC-basierte Registrierung mit CFA-aware Warping, Cosmetic Correction vor Warp | **Empfohlen** |
| **`siril`** | Externe Siril-Registrierung (stern-basiert), Debayer vor Registrierung | Legacy |

**Wichtig:** `opencv_cfa` verhindert "Walking Noise" durch Hotpixel-Korrektur vor dem Warp.

---

### `registration.reference`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `auto` |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Referenz-Frame-Auswahl.

- **`auto`**: Intelligente Auswahl mit zwei Kriterien:
  1. **Mittleres Drittel bevorzugt**: Minimiert die maximale Drift-Distanz zu allen anderen Frames
  2. **Beste Qualität**: Innerhalb des mittleren Drittels wird der Frame mit den meisten Sternen gewählt

**Warum mittleres Drittel?** Bei Aufnahmen mit kontinuierlicher Drift (z.B. Alt-Az-Montierung) ist die maximale Drift-Distanz am geringsten, wenn die Referenz in der Mitte liegt. Dies verbessert die ECC-Konvergenz und verhindert längliche Sterne.

---

### `registration.allow_rotation`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Erlaubt Rotation bei der Registrierung.

- **`true`**: ECC mit MOTION_EUCLIDEAN (Translation + Rotation) - **empfohlen**
- **`false`**: ECC mit MOTION_TRANSLATION (nur Translation)

**Wichtig:** Bei Feldrotation (Alt-Az-Montierung) oder Drift sollte `true` gesetzt werden, um Unschärfe zu vermeiden.

---

### `registration.min_star_matches`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Minimale Anzahl erkannter Sterne pro Frame für erfolgreiche Registrierung.

**Empfehlung:** 10-20 für zuverlässige Registrierung.

---

### `registration.output_dir`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Ausgabeverzeichnis für registrierte Frames (relativ zum Run-Verzeichnis).

**Standard:** `registered`

---

### `registration.registered_filename_pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Dateinamen-Muster für registrierte Frames.

**Standard:** `reg_{index:05d}.fit`

**Platzhalter:** `{index}` wird durch die Frame-Nummer ersetzt.

---

### `registration.siril_script`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Pfad zu einem benutzerdefinierten Siril-Script (nur bei `engine: siril`).

**Standard:** Leer = verwendet `siril_scripts/siril_register_osc.ssf`

---

## Tile Denoising (Tile-Rauschfilter)

Optionale tile-basierte Rauschunterdrückung (Highpass + Soft-Threshold). Die Tile-Geometrie (tile_size/overlap) wird **nicht** separat konfiguriert, sondern aus der Tile-Grid-Berechnung abgeleitet.

### `tile_denoising.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | false |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Aktiviert tile-basierte Rauschunterdrückung vor der Berechnung der lokalen Metriken.

---

### `tile_denoising.kernel_size`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 3 - 63 |
| **Standard** | 15 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Kernel-Größe (ungerade) für die lokale Hintergrundschätzung pro Tile.

---

### `tile_denoising.alpha`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.5 - 5.0 |
| **Standard** | 2.0 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Threshold-Multiplikator für Soft-Thresholding.

Formel: `threshold = alpha × robust_sigma`.

---

## Global Metrics (Globale Metriken)

Gewichtung der globalen Frame-Qualitätsmetriken gemäß Methodik v3 §5.

### `global_metrics.weights.background`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 1 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Gewicht α für die Hintergrund-Metrik im globalen Qualitätsindex.

**Formel:** Q_f = α·(-B̃) + β·(-σ̃) + γ·Ẽ

**Interpretation:** Höheres Gewicht = stärkere Bestrafung von hellem Hintergrund.

---

### `global_metrics.weights.noise`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 1 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Gewicht β für die Rausch-Metrik im globalen Qualitätsindex.

**Interpretation:** Höheres Gewicht = stärkere Bestrafung von verrauschten Frames.

---

### `global_metrics.weights.gradient`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 1 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Gewicht γ für die Gradienten-Metrik im globalen Qualitätsindex.

**Interpretation:** Höheres Gewicht = stärkere Bevorzugung von Frames mit hoher Gradientenenergie (Schärfe).

**Constraint:** α + β + γ = 1.0 (wird vom Backend erzwungen)

---

### `global_metrics.clamp`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 numbers] |
| **Standard** | [-3, 3] |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Clamp-Bereich für Q_f vor der Exponentialfunktion.

**Formel:** G_f = exp(clamp(Q_f, clamp[0], clamp[1]))

**Empfehlung:** [-3, 3] gemäß Methodik v3.1

---

### `global_metrics.adaptive_weights`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | false |
| **Editierbar** | Ja |

**Zweck:** Adaptive Gewichtung basierend auf Metrik-Varianzen.

- **`true`**: α, β, γ werden automatisch angepasst (Methodik v3.1 §3.2)
- **`false`**: Feste Gewichte aus Konfiguration

---

## Tile (Kachel-Geometrie)

Seeing-adaptive Kachel-Erzeugung gemäß Methodik v3 §6.

### `tile.size_factor`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard** | 32 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Faktor s für die Kachelgröße.

**Formel:** T_0 = s × FWHM

**Beispiel:** Bei FWHM = 2.5 px und s = 32 → T_0 = 80 px

---

### `tile.min_size`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard** | 64 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Minimale Kachelgröße T_min in Pixeln.

**Formel:** T = max(T_min, min(T_0, T_max))

---

### `tile.max_divisor`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard** | 6 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Maximaler Divisor D für die Kachelgröße.

**Formel:** T_max = min(Breite, Höhe) / D

**Beispiel:** Bei 1920×1080 und D = 6 → T_max = 180 px

---

### `tile.overlap_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 0.5 |
| **Standard** | 0.25 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Überlappungsfraktion o für Kacheln.

**Formel:** Überlappung O = o × T

**Beispiel:** Bei T = 100 px und o = 0.25 → O = 25 px

---

### `tile.star_min_count`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 0 |
| **Standard** | 3 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Minimale Sternanzahl für stern-basierte lokale Metriken.

**Verhalten:** Kacheln mit weniger Sternen verwenden struktur-basierte Metriken.

---

## Local Metrics (Lokale Metriken)

Kachel-basierte Qualitätsmetriken gemäß Methodik v3 §7.

### `local_metrics.clamp`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 numbers] |
| **Standard** | [-3, 3] |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Clamp-Bereich für Q_local vor der Exponentialfunktion.

---

### `local_metrics.star_mode.weights.fwhm`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 1 |
| **Standard** | 0.6 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Gewicht für FWHM in der stern-basierten lokalen Qualität.

**Interpretation:** Höheres Gewicht = stärkere Bevorzugung von Kacheln mit kleinem FWHM (schärfere Sterne).

---

### `local_metrics.star_mode.weights.roundness`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 1 |
| **Standard** | 0.2 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Gewicht für Rundheit in der stern-basierten lokalen Qualität.

**Interpretation:** Höheres Gewicht = stärkere Bevorzugung von runden Sternen.

---

### `local_metrics.star_mode.weights.contrast`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 1 |
| **Standard** | 0.2 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Gewicht für Kontrast in der stern-basierten lokalen Qualität.

**Constraint:** fwhm + roundness + contrast = 1.0

---

### `local_metrics.structure_mode.background_weight`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 1 |
| **Standard** | 0.3 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Gewicht für Hintergrund in der struktur-basierten lokalen Qualität.

---

### `local_metrics.structure_mode.metric_weight`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 - 1 |
| **Standard** | 0.7 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Gewicht für ENR (Edge-to-Noise Ratio) in der struktur-basierten lokalen Qualität.

**Constraint:** background_weight + metric_weight = 1.0

---

## Synthetic (Synthetische Frames)

Clustering und synthetische Frame-Erzeugung gemäß Methodik v3 §10.

### `synthetic.weighting`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `global`, `tile_weighted` |
| **Standard** | `global` |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Bestimmt, wie synthetische Frames pro Cluster gebildet werden.

- **`global`**: klassisch, nur globale Gewichte `G_f,c`.
- **`tile_weighted`**: tile-basiert mit effektiven Gewichten `W_f,t,c = G_f,c × L_f,t,c` und Overlap-Add (analog Rekonstruktion), um lokale Qualitätsgewinne in `syn_*.fits` zu propagieren.

### `synthetic.frames_min`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard** | 15 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Minimale Anzahl synthetischer Frames (k_min).

---

### `synthetic.frames_max`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Standard** | 30 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Maximale Anzahl synthetischer Frames (k_max).

---

### `synthetic.clustering.mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Konstant** | `state_vector` |
| **Editierbar** | Nein |

**Zweck:** Clustering-Modus (fest auf state_vector).

---

### `synthetic.clustering.k_selection`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Konstant** | `silhouette_auto` |
| **Editierbar** | Nein |

**Zweck:** Automatische k-Auswahl via Silhouette-Score.

---

### `synthetic.clustering.cluster_count_range`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 integers] |
| **Standard** | [15, 30] |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Erlaubter k-Bereich für Clustering.

---

### `synthetic.clustering.vector`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array of enums |
| **Werte** | `global_weight`, `tile_quality_mean`, `tile_quality_variance`, `background`, `noise` |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Komponenten des State-Vektors für Clustering.

**Empfehlung:** Alle Komponenten für optimale Cluster-Trennung.

---

## Reconstruction (Rekonstruktion)

Tile-basierte Rekonstruktion gemäß Methodik v3 §9.

### `reconstruction.weighting_function`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Konstant** | `exponential` |
| **Editierbar** | Nein |

**Zweck:** Gewichtungsfunktion (fest auf exponential gemäß Methodik v3).

**Formel:** W = exp(Q_local)

---

### `reconstruction.window_function`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Konstant** | `hanning` |
| **Editierbar** | Nein |

**Zweck:** Fensterfunktion für Kachel-Überlappung (fest auf Hanning gemäß Methodik v3).

---

### `reconstruction.tile_rescale`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Konstant** | `median_after_background_subtraction` |
| **Editierbar** | Nein |

**Zweck:** Kachel-Reskalierung (fest auf median_after_background_subtraction).

---

## Debayer

### `debayer`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | true |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Debayer/Demosaic des finalen gestackten CFA-Mosaiks.

- **`true`**: Erzeugt `stacked_rgb.fits` (3, H, W) aus `stacked.fit`
- **`false`**: Nur CFA-Mosaik `stacked.fit` wird erzeugt

---

## Stacking

Finales Stacking gemäß Methodik v3 §11.

### `stacking.method`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | enum |
| **Werte** | `average`, `rej` |
| **Standard** | `rej` |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Stacking-Methode.

- **`average`**: Einfacher linearer Mittelwert
- **`rej`**: Sigma-Clipping Rejection, dann linearer Mittelwert (empfohlen)

---

### `stacking.input_dir`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Eingabeverzeichnis für Stacking (relativ zum outputs-Verzeichnis).

**Standard:** `synthetic`

---

### `stacking.input_pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Dateinamen-Muster für Stacking-Input.

**Standard:** `syn_*.fits`

---

### `stacking.output_file`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Ausgabedatei für gestacktes Bild.

**Standard:** `stacked.fit`

---

### `stacking.sigma_clip.sigma_low`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.1 - 20.0 |
| **Standard** | 4.0 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Unterer Sigma-Schwellenwert für Rejection.

**Formel:** Pixel wird abgelehnt wenn z < -sigma_low

**Empfehlung:** 2.5 - 4.0 für typische Daten

---

### `stacking.sigma_clip.sigma_high`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.1 - 20.0 |
| **Standard** | 4.0 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Oberer Sigma-Schwellenwert für Rejection.

**Formel:** Pixel wird abgelehnt wenn z > sigma_high

**Empfehlung:** 2.5 - 4.0 für typische Daten

---

### `stacking.sigma_clip.max_iters`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 1 - 10 |
| **Standard** | 3 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Maximale Anzahl Sigma-Clipping-Iterationen.

---

### `stacking.sigma_clip.min_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.05 - 1.0 |
| **Standard** | 0.5 |
| **Erforderlich** | Nein |
| **Editierbar** | Ja |

**Zweck:** Minimale überlebende Frame-Fraktion pro Pixel.

**Verhalten:** Wenn weniger als min_fraction × N Frames überleben, wird auf unclipped mean zurückgefallen.

---

## Validation (Validierung)

Qualitätsprüfungen gemäß Methodik v3 §12.

### `validation.min_fwhm_improvement_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard** | 5 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Minimale FWHM-Verbesserung in Prozent gegenüber einfachem Stack.

---

### `validation.max_background_rms_increase_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard** | 0 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Maximale erlaubte Hintergrund-RMS-Erhöhung in Prozent.

---

### `validation.min_tile_weight_variance`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard** | 0.1 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Minimale Varianz der Kachel-Gewichte (zeigt an, dass Qualitätsunterschiede erkannt wurden).

---

### `validation.require_no_tile_pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Standard** | true |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Prüft auf sichtbare Kachel-Muster im Endergebnis.

---

## Runtime Limits

Laufzeit-Beschränkungen.

### `runtime_limits.tile_analysis_max_factor_vs_stack`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard** | 3.0 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Maximaler Zeitfaktor für Tile-Analyse relativ zum einfachen Stack.

---

### `runtime_limits.hard_abort_hours`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Standard** | 6 |
| **Erforderlich** | Ja |
| **Editierbar** | Ja |

**Zweck:** Maximale Laufzeit in Stunden bevor die Pipeline abgebrochen wird.

---

## Beispiel-Konfiguration

```yaml
pipeline:
  mode: production
  abort_on_fail: true

registration:
  engine: opencv_cfa
  allow_rotation: true
  min_star_matches: 10

global_metrics:
  weights:
    background: 0.4
    noise: 0.3
    gradient: 0.3

stacking:
  method: rej
  sigma_clip:
    sigma_low: 3
    sigma_high: 3
```
