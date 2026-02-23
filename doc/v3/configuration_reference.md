# Tile-Compile C++ Konfigurationsreferenz

Diese Dokumentation beschreibt alle Konfigurationsoptionen für `tile_compile.yaml` basierend auf der C++ Implementierung in `configuration.hpp` und den Schema-Dateien `tile_compile.schema.json` / `tile_compile.schema.yaml`.

**Quelle der Wahrheit für Defaults:** `include/tile_compile/config/configuration.hpp`  
**Schema-Version:** v3  
**Referenz:** Methodik v3.2

**💡 Für praktische Beispiele und Anwendungsfälle siehe:** [Konfigurationsbeispiele & Best Practices](configuration_examples_practical_de.md)

## Inhaltsverzeichnis

1. [Pipeline](#1-pipeline)
2. [Output](#2-output)
3. [Data](#3-data)
4. [Linearity](#4-linearity)
5. [Calibration](#5-calibration)
6. [Assumptions](#6-assumptions)
7. [Normalization](#7-normalization)
8. [Registration](#8-registration)
9. [Tile Denoise](#9-tile-denoise)
10. [Global Metrics](#10-global-metrics)
11. [Tile](#11-tile)
12. [Local Metrics](#12-local-metrics)
13. [Synthetic](#13-synthetic)
14. [Reconstruction](#14-reconstruction)
15. [Debayer](#15-debayer)
16. [Astrometry](#16-astrometry)
17. [PCC](#17-pcc)
18. [Stacking](#18-stacking)
19. [Validation](#19-validation)
20. [Runtime Limits](#20-runtime-limits)

---

## 1. Pipeline

Grundlegende Pipeline-Steuerung.

### `pipeline.mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `production`, `test` |
| **Default** | `"production"` |

**Zweck:** Bestimmt den Ausführungsmodus der Pipeline.

- **`production`**: Vollständige Verarbeitung mit allen Qualitätsprüfungen und Phasen
- **`test`**: Reduzierte Verarbeitung für schnelle Tests (weniger Iterationen, reduzierte Validierung)

---

### `pipeline.abort_on_fail`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Bestimmt, ob die Pipeline bei kritischen Fehlern sofort abbricht.

- **`true`**: Pipeline stoppt bei `phase_end(error)` — empfohlen für Produktion
- **`false`**: Pipeline versucht fortzufahren (nützlich für Debugging, um alle Phasen-Outputs zu erhalten)

---

## 2. Output

Steuerung der Ausgabeverzeichnisse und welche Zwischenergebnisse geschrieben werden.

### `output.registered_dir`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `"registered"` |

**Zweck:** Unterverzeichnis im Run-Ordner für registrierte Frames (relativ zu `runs/<run_id>/outputs/`).

---

### `output.artifacts_dir`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `"artifacts"` |

**Zweck:** Unterverzeichnis für JSON-Artefakte und Report-Dateien.

---

### `output.write_registered_frames`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Registrierte Frames als FITS auf Disk schreiben.

- **`true`**: Jeder registrierte Frame wird als `reg_XXXXX.fit` gespeichert — **hoher Speicherbedarf!**
- **`false`**: Registrierte Frames werden nur im Speicher gehalten

**Hinweis:** In `tile_compile.yaml` steht `true`, der C++ Default ist `false`. Nur für Debugging empfohlen.

---

### `output.write_global_metrics`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** `global_metrics.json` Artefakt schreiben.

---

### `output.write_global_registration`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** `global_registration.json` Artefakt schreiben (Warp-Matrizen + CC pro Frame).

---

### `output.crop_to_nonzero_bbox`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Finalen Stack auf die Bounding Box aller nicht-null Pixel zuschneiden.

- **`true`**: Entfernt leere Ränder vom finalen Bild. Nur Pixel mit Werten > 0 werden behalten. Reduziert die Dateigröße und entfernt unnötige schwarze Ränder.
- **`false`**: Behält die volle Canvas-Größe bei, inklusive leerer Ränder.

**Hinweis:** Die Funktion wird nach der Stack-Phase aber vor dem Debayer (bei OSC) angewendet. Die Tile-Offsets werden entsprechend angepasst.

---

## 3. Data

Bilddaten-Eigenschaften. Teilweise automatisch aus dem FITS-Header ermittelt, teilweise konfigurierbar.

### `data.image_width`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `0` (automatisch erkannt) |

**Zweck:** Bildbreite in Pixeln. Wird normalerweise automatisch aus dem FITS-Header (NAXIS1) gelesen. Kann in der Config vorbelegt werden, um vor dem Run eine Erwartung zu setzen.

---

### `data.image_height`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `0` (automatisch erkannt) |

**Zweck:** Bildhöhe in Pixeln. Wird aus FITS-Header (NAXIS2) gelesen.

---

### `data.frames_min`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `0` |

**Zweck:** Erwartete Mindestanzahl Frames im Input-Verzeichnis. `0` = keine Prüfung.

**Hinweis:** Nicht zu verwechseln mit `assumptions.frames_min`, das für Pipeline-Entscheidungen (Reduced Mode) verwendet wird.

---

### `data.frames_target`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 0 |
| **Default** | `0` |

**Zweck:** Erwartete Zielanzahl Frames. `0` = kein Ziel. Informativ, kein Abbruch.

---

### `data.color_mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `OSC`, `MONO`, `RGB` |
| **Default** | `"OSC"` |

**Zweck:** Erwarteter Farbmodus der Kamera.

| Modus | Beschreibung |
|-------|-------------|
| **`OSC`** | One-Shot-Color — Farbkamera mit Bayer-Matrix (CFA). Pipeline arbeitet CFA-aware |
| **`MONO`** | Monochrome Kamera ohne Farbfilter. Einzelkanal-Verarbeitung |
| **`RGB`** | RGB-Daten (3 Kanäle). Derzeit nicht aktiv im C++ Runner |

**Verhalten:** Wird aus FITS-Header (`BAYERPAT` vorhanden → OSC, sonst MONO) automatisch erkannt. Bei Abweichung zur Config wird gewarnt, der **erkannte** Modus hat Vorrang.

---

### `data.bayer_pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `"GBRG"` |

**Zweck:** Bayer-Pattern für OSC-Kameras. Bestimmt die Farbfilter-Anordnung auf dem Sensor.

| Pattern | Zeile 0 | Zeile 1 |
|---------|---------|---------|
| `RGGB` | R G | G B |
| `BGGR` | B G | G R |
| `GBRG` | G B | R G |
| `GRBG` | G R | B G |

**Wichtig:** Ein falsches Bayer-Pattern führt zu **komplett falschen Farben**. Wird aus FITS-Header `BAYERPAT` gelesen, Fallback auf Config-Wert.

---

### `data.linear_required`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Schaltet die strikte Entfernung nicht-linearer Frames ein/aus.

- Non-lineare Frames werden **nicht entfernt**, es wird nur gewarnt (`warn_only`).

**Zusammenspiel mit `linearity.enabled`:** Die Linearitätsprüfung muss `enabled=true` sein, damit Warnungen für non-lineare Frames entstehen.

---

## 4. Linearity

Linearitätsprüfung der Input-Frames. Stellt sicher, dass keine nichtlinearen Operationen (Stretch, Curves) angewendet wurden.

### `linearity.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Aktiviert die Linearitätsprüfung in Phase 0 (SCAN_INPUT).

---

### `linearity.max_frames`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `8` |

**Zweck:** Maximale Anzahl Frames in der Stichprobe für die Linearitätsprüfung.

**Verhalten:** Aus N Frames werden bis zu `max_frames` gleichmäßig verteilt ausgewählt und geprüft. Nicht alle Frames werden getestet — die Stichprobe reicht zur zuverlässigen Erkennung.

---

### `linearity.min_overall_linearity`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.0 – 1.0 |
| **Default** | `0.9` |

**Zweck:** Mindest-Linearitäts-Score (0 = komplett nichtlinear, 1 = perfekt linear). Frames unter diesem Schwellenwert gelten als nicht-linear.

---

### `linearity.strictness`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `strict`, `moderate`, `permissive` |
| **Default** | `"strict"` |

**Zweck:** Strictness-Level für die Linearitäts-Validierung.

| Level | Beschreibung |
|-------|-------------|
| **`strict`** | Strenge Prüfung — empfohlen für kalibrierte Daten |
| **`moderate`** | Moderate Toleranz — für leicht vorverarbeitete Daten |
| **`permissive`** | Hohe Toleranz — nur für bekannt problematische Daten |

---

## 5. Calibration

Kalibrierungs-Einstellungen (Bias, Dark, Flat). Wird **vor** der Pipeline auf die Raw-Frames angewendet.

### `calibration.use_bias` / `calibration.use_dark` / `calibration.use_flat`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` (alle drei) |

**Zweck:** Aktiviert die jeweilige Kalibrierung.

- **Bias**: Subtrahiert das Ausleserauschen (Offset)
- **Dark**: Subtrahiert thermisches Rauschen (Dunkelstrom)
- **Flat**: Korrigiert Vignettierung und Staubkörner

---

### `calibration.bias_use_master` / `calibration.dark_use_master` / `calibration.flat_use_master`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` (alle drei) |

**Zweck:** Wenn `true`, wird ein fertiges Master-Frame aus dem angegebenen Pfad geladen statt aus Einzelframes gemittelt.

---

### `calibration.dark_auto_select`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Automatische Dark-Auswahl basierend auf Belichtungszeit (und optional Temperatur).

---

### `calibration.dark_match_exposure_tolerance_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Default** | `5.0` |

**Zweck:** Maximale Abweichung der Belichtungszeit bei Dark-Matching in Prozent.

---

### `calibration.dark_match_use_temp` / `calibration.dark_match_temp_tolerance_c`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean / number |
| **Default** | `false` / `2.0` |

**Zweck:** Wenn `dark_match_use_temp=true`, wird zusätzlich die Sensor-Temperatur für Dark-Matching berücksichtigt (±`temp_tolerance_c` °C).

---

### `calibration.bias_dir` / `calibration.darks_dir` / `calibration.flats_dir`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `""` (leer) |

**Zweck:** Verzeichnispfade für Bias/Dark/Flat-Einzelframes (zum Erzeugen von Master-Frames).

---

### `calibration.bias_master` / `calibration.dark_master` / `calibration.flat_master`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `""` (leer) |

**Zweck:** Pfade zu fertigen Master-Frames (nur wenn `*_use_master=true`).

---

### `calibration.pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `"*.fit*"` |

**Zweck:** Glob-Pattern für Kalibrierungsdateien.

---

## 6. Assumptions

Schwellenwerte und Annahmen für Pipeline-Entscheidungen (Normal Mode vs. Reduced Mode).

### `assumptions.frames_min`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `50` |

**Zweck:** Minimale Frame-Anzahl für eine sinnvolle Verarbeitung (Hard Assumption).

**Verhalten:** Bei weniger Frames wird eine Warnung erzeugt. Der Abbruch hängt von `pipeline.abort_on_fail` ab.

---

### `assumptions.frames_optimal`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `800` |

**Zweck:** Optimale Frame-Anzahl für vollständige Methodik. Rein informativ — erzeugt eine Warnung bei Unterschreitung, aber keinen Abbruch.

---

### `assumptions.frames_reduced_threshold`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `200` |

**Zweck:** Schwellenwert für den Wechsel zwischen Normal Mode und Reduced Mode.

| Frame-Anzahl | Modus |
|-------------|-------|
| `< frames_min` | Warnung / Abbruch |
| `frames_min ≤ N < frames_reduced_threshold` | **Reduced Mode** (kein Clustering, keine synthetischen Frames) |
| `N ≥ frames_reduced_threshold` | **Normal Mode** (alle Phasen) |

---

### `assumptions.exposure_time_tolerance_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Default** | `5.0` |

**Zweck:** Maximale erlaubte Abweichung der Belichtungszeit in Prozent (Hard Assumption).

**Beispiel:** Bei 5% und 10s Belichtung sind 9.5s – 10.5s erlaubt.

---

### `assumptions.reduced_mode_skip_clustering`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Überspringt STATE_CLUSTERING und SYNTHETIC_FRAMES im Reduced Mode.

- **`true`** (empfohlen): Tile-Rekonstruktionsergebnis wird direkt als finales Bild verwendet
- **`false`**: Führt auch im Reduced Mode Clustering durch (mit eingeschränktem K-Bereich)

---

### `assumptions.reduced_mode_cluster_range`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 integers] |
| **Default** | `[5, 10]` |

**Zweck:** Cluster-Anzahl-Bereich [k_min, k_max] für Reduced Mode (nur relevant wenn `reduced_mode_skip_clustering=false`).

---

## 7. Normalization

Hintergrund-Normalisierung (Pflicht gemäß Methodik v3 §4).

### `normalization.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |
| **Konstant** | Muss `true` sein |

**Zweck:** Normalisierung ist **immer aktiviert** (Methodik-Zwang). Wird auf `false` gesetzt, bricht die Pipeline ab.

---

### `normalization.mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `background`, `median` |
| **Default** | `"background"` |

**Zweck:** Normalisierungsmethode.

| Methode | Beschreibung | Empfehlung |
|---------|-------------|------------|
| **`background`** | Sigma-Clip-Background-Maske → Median der Background-Pixel → Division | **Empfohlen** |
| **`median`** | Einfacher Median aller Pixel → Division | Fallback |

Die `background`-Methode verwendet eine Sigma-Clipping-Maske, um Sterne und Objekte auszuschließen und nur den echten Hintergrund zu schätzen.

---

### `normalization.per_channel`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Kanalweise Normalisierung bei OSC-Daten.

- **`true`** (empfohlen): Jeder Bayer-Kanal (R, G, B) wird separat normalisiert. Kompensiert kanalabhängige Hintergrundunterschiede (z.B. Lichtverschmutzung)
- **`false`**: Einheitliche Normalisierung über alle Pixel

---

## 8. Registration

Geometrische Registrierung (Ausrichtung) aller Frames auf einen Referenz-Frame.

### `registration.engine`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `triangle_star_matching`, `star_similarity`, `hybrid_phase_ecc`, `robust_phase_ecc` |
| **Default** | `"triangle_star_matching"` |

**Zweck:** Primäre Registrierungsmethode. Intern wird **immer eine 6-stufige Kaskade** durchlaufen. Der `engine`-Wert bestimmt die bevorzugte Methode.

| Engine | Beschreibung | Stärke |
|--------|-------------|--------|
| **`triangle_star_matching`** | Dreiecks-Asterismus-Matching | **Rotationsinvariant**, ideal für Alt/Az, klarer Himmel |
| **`star_similarity`** | Stern-Paar-Distanz-Matching | Schnell bei kleinen Versätzen |
| **`hybrid_phase_ecc`** | Phase-Korrelation + ECC | Ohne Sternerkennung, für Nebel |
| **`robust_phase_ecc`** | LoG-Gradient-Preprocessing + Pyramiden-Phase+ECC | **Empfohlen bei Wolken/Nebel**, entfernt Gradienten vor Korrelation |

**Kaskade (immer):** Triangle Stars → Star Pairs → Trail Endpoints → AKAZE Features → Robust Phase+ECC → Hybrid Phase+ECC → Identity-Fallback

**Temporal-Smoothing (v3.2.3+, automatisch aktiv):** Bei fehlgeschlagener direkter Registrierung `i→ref` wird automatisch versucht:
1. `i→(i-1)→ref` — Registrierung zum Vorgänger-Frame, dann Warp-Verkettung
2. `i→(i+1)→ref` — Registrierung zum Nachfolger-Frame, dann Warp-Verkettung

Alle verketteten Warps werden mit NCC gegen den Referenz-Frame validiert. Besonders wirksam bei kontinuierlicher Feldrotation (Alt/Az nahe Pol) und Wolken/Nebel. Logs: `[REG-TEMPORAL]`

---

### `registration.allow_rotation`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Erlaubt Rotation bei der Registrierung.

- **`true`** (empfohlen): Similarity/Euclidean Transform (Translation + Rotation + ggf. Skalierung)
- **`false`**: Nur Translation

**Wichtig:** Bei Feldrotation (Alt/Az-Montierung) **muss** `true` gesetzt sein. Keine Rotationslimits — auch >20° Rotation wird akzeptiert.

---

### `registration.star_topk`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 3 |
| **Default** | `120` |

**Zweck:** Anzahl der hellsten Sterne, die für Star-basiertes Matching verwendet werden.

**Hinweis:** In `tile_compile.yaml` steht `100`, der C++ Default ist `120`. Höhere Werte erhöhen die Robustheit bei schwierigen Feldern, aber auch die Rechenzeit.

---

### `registration.star_min_inliers`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 2 |
| **Default** | `6` |

**Zweck:** Minimale Anzahl übereinstimmender Sterne (Inlier) für eine akzeptierte Registrierung.

**Empfehlung:** 5–10. Zu niedrig → falsche Matches werden akzeptiert. Zu hoch → gute Matches werden abgelehnt.

---

### `registration.star_inlier_tol_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `2.5` |

**Zweck:** Toleranz in Pixeln für die Zuordnung von Sternen als Inlier (nach Transformation).

**Hinweis:** In `tile_compile.yaml` steht `3.0`, der C++ Default ist `2.5`. Bezieht sich auf die **halbe Auflösung** (2× Downsample in der Registrierung).

---

### `registration.star_dist_bin_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `2.5` |

**Zweck:** Bin-Breite in Pixeln für das Paar-Abstands-Histogramm in der `star_similarity`-Methode.

**Hinweis:** In `tile_compile.yaml` steht `5.0`, der C++ Default ist `2.5`. Kleinere Werte sind genauer, größere Werte toleranter.

---

### `registration.reject_outliers`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Aktiviert die automatische Verwerfung implausibler globaler Registrierungs-Warps vor PREWARP/TILE-Phasen.

- **`true`**: Low-CC-, Shift-, Reflection- und Scale-Outlier werden auf Identity gesetzt
- **`false`**: Keine zusätzliche Outlier-Verwerfung nach der Registrierung

**Logging:** Jeder verworfene Frame wird als `warning` in `logs/run_events.jsonl` protokolliert und zusätzlich in `phase_end(REGISTRATION)` unter `reg_rejected_frames` abgelegt.

---

### `registration.reject_cc_min_abs`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 1 |
| **Default** | `0.35` |

**Zweck:** Absolute Untergrenze für den Korrelationswert (CC) in der Registrierungs-Outlier-Erkennung.

**Hinweis:** Effektiver CC-Schwellwert ist `max(reject_cc_min_abs, median(CC) - reject_cc_mad_multiplier * MAD(CC))`.

---

### `registration.reject_cc_mad_multiplier`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `4.0` |

**Zweck:** Robustheitsfaktor für den CC-basierten Outlier-Schwellenwert (MAD-basiert).

Kleinere Werte verwerfen aggressiver, größere Werte konservativer.

---

### `registration.reject_shift_px_min`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >=0 |
| **Default** | `25.0` |

**Zweck:** Feste Mindestgrenze (Pixel) für Shift-Outlier-Verwerfung.

**Hinweis:** Effektiver Shift-Grenzwert ist `max(reject_shift_px_min, reject_shift_median_multiplier * median(shift_px))`.

---

### `registration.reject_shift_median_multiplier`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `3.0` |

**Zweck:** Skalenfaktor für den robusten Shift-Outlier-Grenzwert relativ zur Medianverschiebung.

---

### `registration.reject_scale_min` / `registration.reject_scale_max`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number / number |
| **Default** | `0.92` / `1.08` |

**Zweck:** Zulässiger Bereich für den geschätzten Similarity-Scale bei globaler Registrierung.

- Scale außerhalb `[reject_scale_min, reject_scale_max]` wird als Outlier verworfen.
- Zusätzlich werden Reflection-Warps (`det < 0`) immer verworfen.

---

## 8b. Dithering

### `dithering.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Kennzeichnet Dither-Session. Bei aktivem Flag werden in `global_registration.json` zusätzliche Dither-Diagnosen (detected_count/fraction) aus den gemessenen globalen Shifts geschrieben.

---

### `dithering.min_shift_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Default** | `0.5` |

**Zweck:** Mindest-Verschiebung in Pixeln, ab der ein Frame als "gedithert" gezählt wird.

---

## 9. Tile Denoise

Optionale Tile-Denoise-Stufe mit zwei Komponenten:

- `tile_denoise.soft_threshold.*` (Default aktiv)
- `tile_denoise.wiener.*` (Default inaktiv)

### `tile_denoise.soft_threshold.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Aktiviert Soft-Threshold-Denoising pro Tile.

---

### `tile_denoise.soft_threshold.blur_kernel`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 3 |
| **Default** | `31` |

**Zweck:** Kernelgröße für die lokale Hintergrundschätzung (Box-Blur).

---

### `tile_denoise.soft_threshold.alpha`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `1.5` |

**Zweck:** Schwellenfaktor für das Soft-Thresholding (`tau = alpha * sigma_tile`).

---

### `tile_denoise.soft_threshold.skip_star_tiles`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Stern-dominierte Tiles vom Soft-Thresholding ausnehmen.

---

### `tile_denoise.wiener.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Aktiviert den Wiener-Filter in der Tile-Denoise-Stufe.

---

### `tile_denoise.wiener.snr_threshold`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Default** | `5.0` |

**Zweck:** SNR-Schwelle; oberhalb dieses Werts wird typischerweise nicht gefiltert.

---

### `tile_denoise.wiener.q_min`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | -1 bis q_max |
| **Default** | `-0.5` |

---

### `tile_denoise.wiener.q_max`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 1 |
| **Default** | `1.0` |

---

### `tile_denoise.wiener.q_step`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `0.1` |

---

### `tile_denoise.wiener.min_snr`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Default** | `2.0` |

---

### `tile_denoise.wiener.max_iterations`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `10` |

---

### `chroma_denoise` (struktur-schützende Farbrausch-Reduktion)

Optionale, **chroma-selektive** Denoise-Erweiterung für OSC-Daten. Idee: Luminanz/Struktur möglichst erhalten, Farbrauschen primär in Cb/Cr (oder äquivalenten Opponent-Kanälen) reduzieren.

> Hinweis: Wirkt nur im OSC-Pfad. `apply_stage` steuert, ob vor dem Tile-Overlap-Add (`pre_stack_tiles`) oder auf dem finalen linearen RGB-Stack (`post_stack_linear`) gefiltert wird.

```yaml
chroma_denoise:
  enabled: true
  color_space: ycbcr_linear        # ycbcr_linear | opponent_linear
  apply_stage: post_stack_linear   # pre_stack_tiles | post_stack_linear
  protect_luma: true
  luma_guard_strength: 0.75        # 0..1
  star_protection:
    enabled: true
    threshold_sigma: 2.2
    dilate_px: 2
  structure_protection:
    enabled: true
    gradient_percentile: 85
  chroma_wavelet:
    enabled: true
    levels: 3
    threshold_scale: 1.25
    soft_k: 1.0
  chroma_bilateral:
    enabled: true
    sigma_spatial: 1.2
    sigma_range: 0.035
  blend:
    mode: chroma_only
    amount: 0.85                   # 0..1
```

#### Presets

##### 1) Konservativ

Für bereits saubere Daten, minimales Risiko für Farbsättigungs- und Detailverlust.

```yaml
chroma_denoise:
  enabled: true
  color_space: ycbcr_linear
  apply_stage: post_stack_linear
  protect_luma: true
  luma_guard_strength: 0.85
  star_protection: { enabled: true, threshold_sigma: 2.6, dilate_px: 2 }
  structure_protection: { enabled: true, gradient_percentile: 88 }
  chroma_wavelet: { enabled: true, levels: 2, threshold_scale: 0.95, soft_k: 1.0 }
  chroma_bilateral: { enabled: true, sigma_spatial: 1.0, sigma_range: 0.025 }
  blend: { mode: chroma_only, amount: 0.65 }
```

##### 2) Balanced (empfohlen)

Guter Standard für Smart-Telescope-Stacks mit sichtbarem Chroma-Hintergrundrauschen.

```yaml
chroma_denoise:
  enabled: true
  color_space: ycbcr_linear
  apply_stage: post_stack_linear
  protect_luma: true
  luma_guard_strength: 0.75
  star_protection: { enabled: true, threshold_sigma: 2.2, dilate_px: 2 }
  structure_protection: { enabled: true, gradient_percentile: 85 }
  chroma_wavelet: { enabled: true, levels: 3, threshold_scale: 1.25, soft_k: 1.0 }
  chroma_bilateral: { enabled: true, sigma_spatial: 1.2, sigma_range: 0.035 }
  blend: { mode: chroma_only, amount: 0.85 }
```

##### 3) Aggressiv

Für starkes Farbrauschen; erhöhtes Risiko für Desaturierung/"Plastik-Look" in schwachen Nebelbereichen.

```yaml
chroma_denoise:
  enabled: true
  color_space: ycbcr_linear
  apply_stage: post_stack_linear
  protect_luma: true
  luma_guard_strength: 0.65
  star_protection: { enabled: true, threshold_sigma: 1.9, dilate_px: 3 }
  structure_protection: { enabled: true, gradient_percentile: 80 }
  chroma_wavelet: { enabled: true, levels: 4, threshold_scale: 1.55, soft_k: 1.1 }
  chroma_bilateral: { enabled: true, sigma_spatial: 1.6, sigma_range: 0.05 }
  blend: { mode: chroma_only, amount: 1.0 }
```

---

## 10. Global Metrics

Gewichtung der globalen Frame-Qualitätsmetriken (Phase 4: GLOBAL_METRICS).

### `global_metrics.weights.background`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 1 |
| **Default** | `0.4` |

**Zweck:** Gewicht **α** für die Hintergrund-Metrik im globalen Qualitätsindex.

**Formel:** `Q_f = α·(-B̃_f) + β·(-σ̃_f) + γ·Ẽ_f`

**Interpretation:** Höheres Gewicht → stärkere Bestrafung von hellem Hintergrund (Lichtverschmutzung, Dämmerung).

**Hinweis:** In `tile_compile.yaml` steht `0.45`, der C++ Default ist `0.4`.

---

### `global_metrics.weights.noise`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 1 |
| **Default** | `0.3` |

**Zweck:** Gewicht **β** für die Rausch-Metrik.

**Interpretation:** Höheres Gewicht → stärkere Bestrafung von verrauschten Frames (schlechte Kühlung, hohe ISO).

**Hinweis:** In `tile_compile.yaml` steht `0.35`.

---

### `global_metrics.weights.gradient`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 1 |
| **Default** | `0.3` |

**Zweck:** Gewicht **γ** für die Gradienten-Metrik (Sobel-Energie).

**Interpretation:** Höheres Gewicht → stärkere Bevorzugung von Frames mit hoher Gradientenenergie (Schärfe, Struktur).

**Constraint:** **α + β + γ = 1.0** — wird von `cfg.validate()` geprüft.

---

### `global_metrics.clamp`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 numbers] |
| **Default** | `[-3.0, 3.0]` |

**Zweck:** Clamp-Bereich für Q_f vor der Exponentialfunktion.

**Formel:** `G_f = exp(clip(Q_f, clamp[0], clamp[1]))`

- Clamp [-3, +3] → Gewichtsbereich [exp(-3) ≈ 0.05, exp(+3) ≈ 20.1]
- Verhindert extreme Gewichte durch Ausreißer in den Metriken

---

### `global_metrics.adaptive_weights`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Adaptive Gewichtung basierend auf Metrik-Varianzen.

- **`true`**: α, β, γ werden automatisch angepasst (proportional zur Varianz der jeweiligen Metrik)
- **`false`**: Feste Gewichte aus Konfiguration (empfohlen)

---

### `global_metrics.weight_exponent_scale`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `1.0` |

**Zweck:** Exponent-Skalierung `k` für die globale Gewichtung `G_f = exp(k * Q_f)`.

- `k = 1.0`: Standard-Verhalten
- `k > 1.0`: stärkere Trennung guter/schlechter Frames
- `k < 1.0`: flachere Gewichtsverteilung

---

## 11. Tile

Seeing-adaptive Tile-Erzeugung (Phase 5: TILE_GRID).

### `tile.size_factor`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `32` |

**Zweck:** Multiplikator **s** für die Tile-Größe.

**Formel:** `T₀ = s × FWHM`

**Beispiele:**

| FWHM (px) | s=32 | Tile-Größe |
|-----------|------|-----------|
| 2.0 | 64 | 64 px |
| 3.0 | 96 | 96 px |
| 5.0 | 160 | 160 px |

**Empfehlung:** 20–40. Höhere Werte → größere Tiles → weniger Tiles → schneller, aber gröbere lokale Anpassung.

---

### `tile.min_size`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `64` |

**Zweck:** Minimale Tile-Größe **T_min** in Pixeln.

**Formel:** `T = clip(T₀, T_min, T_max)`

**Empfehlung:** 32–128. Zu klein → zu wenige Pixel pro Tile für zuverlässige Metriken. Zu groß → keine lokale Anpassung.

---

### `tile.max_divisor`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `6` |

**Zweck:** Maximale Tile-Größe als Bruchteil der kürzeren Bildseite.

**Formel:** `T_max = min(Breite, Höhe) / max_divisor`

**Beispiel:** Bei 3840×2160 und D=6 → T_max = 2160/6 = 360 px

---

### `tile.overlap_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 0.5 |
| **Default** | `0.25` |

**Zweck:** Überlappungsfraktion zwischen benachbarten Tiles.

**Formel:** `Overlap = fraction × TileSize`, `Stride = TileSize − Overlap`

**Beispiel:** T=100, fraction=0.25 → Overlap=25px, Stride=75px

Die Überlappung ist **kritisch** für die Hanning-Overlap-Add-Rekonstruktion: Zu wenig Overlap → sichtbare Tile-Grenzen. Zu viel → ineffizient.

**Safety:** Wenn stride ≤ 0, wird auf 0.25 zurückgesetzt.

---

### `tile.star_min_count`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 0 |
| **Default** | `10` |

**Zweck:** Minimale Sternanzahl (Median über alle Frames) für die Klassifikation als **STAR-Tile**.

| Median Star Count | Tile-Typ | Metrik-Modus |
|-------------------|----------|-------------|
| `≥ star_min_count` | STAR | FWHM + Roundness + Contrast |
| `< star_min_count` | STRUCTURE | ENR + Background |

---

## 12. Local Metrics

Lokale Tile-Metriken und Qualitäts-Scoring (Phase 6: LOCAL_METRICS).

### `local_metrics.clamp`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 numbers] |
| **Default** | `[-3.0, 3.0]` |

**Zweck:** Clamp-Bereich für Q_local vor der Exponentialfunktion.

**Formel:** `L_f,t = exp(clip(Q_f,t, clamp[0], clamp[1]))`

---

### `local_metrics.star_mode.weights.fwhm`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 1 |
| **Default** | `0.6` |

**Zweck:** Gewicht für FWHM in der stern-basierten lokalen Qualität.

**Formel:** `Q = w_fwhm·(-FWHM̃) + w_round·R̃ + w_contrast·C̃`

Niedriger FWHM = besser → wird negiert. Höchstes Gewicht = dominiert die lokale Qualitätsbewertung.

---

### `local_metrics.star_mode.weights.roundness`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 1 |
| **Default** | `0.2` |

**Zweck:** Gewicht für Sternrundheit. Hohe Rundheit = gutes Tracking.

---

### `local_metrics.star_mode.weights.contrast`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 1 |
| **Default** | `0.2` |

**Zweck:** Gewicht für lokalen Kontrast. Hoher Kontrast = gutes Signal.

**Constraint:** **fwhm + roundness + contrast = 1.0**

---

### `local_metrics.structure_mode.metric_weight`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 1 |
| **Default** | `0.7` |

**Zweck:** Gewicht für ENR (Edge-to-Noise Ratio) in der struktur-basierten lokalen Qualität.

**Formel:** `Q = w_metric·(Ẽ/σ̃) + w_bg·(-B̃)`

---

### `local_metrics.structure_mode.background_weight`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 1 |
| **Default** | `0.3` |

**Zweck:** Gewicht für Hintergrund in der struktur-basierten lokalen Qualität.

**Constraint:** **metric_weight + background_weight = 1.0**

---

## 13. Synthetic

Synthetische Frame-Erzeugung und Clustering (Phase 8+9).

### `synthetic.weighting`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `global`, `tile_weighted` |
| **Default** | `"global"` |

**Zweck:** Bestimmt, wie synthetische Frames pro Cluster gebildet werden.

| Modus | Formel | Beschreibung |
|-------|--------|-------------|
| **`global`** | `synth_k = Σ G_f · warp(I'_f) / Σ G_f` | Nur globale Gewichte — schneller, Standard |
| **`tile_weighted`** | `synth_k = overlap_add(Σ W_f,t · tile_f / Σ W_f,t)` | Tile-basiert wie Rekonstruktion — langsamer, propagiert lokale Qualitätsgewinne |

---

### `synthetic.frames_min`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `5` |

**Zweck:** Minimale Cluster-Größe für die Erzeugung eines synthetischen Frames. Cluster mit weniger Frames werden übersprungen.

---

### `synthetic.frames_max`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `30` |

**Zweck:** Maximale Anzahl synthetischer Frames. Bestimmt gleichzeitig die maximale Cluster-Anzahl (K ≤ frames_max).

---

### `synthetic.clustering.mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `kmeans`, `quantile` |
| **Default** | `"kmeans"` |

**Zweck:** Clustering-Methode.

| Methode | Beschreibung |
|---------|-------------|
| **`kmeans`** | K-Means auf 6D-Zustandsvektor (Standard) |
| **`quantile`** | Quantile-basierte Aufteilung nach globalem Gewicht (Fallback bei degenerierten Clustern) |

**Hinweis:** `quantile` wird automatisch als Fallback verwendet wenn K-Means leere Cluster erzeugt.

---

### `synthetic.clustering.cluster_count_range`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | array [2 integers] |
| **Default** | `[5, 30]` |

**Zweck:** Erlaubter K-Bereich [k_min, k_max] für Clustering.

**Formel:** `K = clip(floor(N / 10), k_min, k_max)`

| N Frames | K (Default [5,30]) |
|----------|-------------------|
| 50 | 5 |
| 100 | 10 |
| 200 | 20 |
| 500 | 30 |

---

## 14. Reconstruction

Tile-basierte Rekonstruktion (Phase 7: TILE_RECONSTRUCTION). Diese Einstellungen sind **fest** gemäß Methodik v3.

### `reconstruction.weighting_function`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `linear` |
| **Default** | `"linear"` |
| **Konstant** | Ja |

**Zweck:** Gewichtungsfunktion für Tile-Rekonstruktion. Fest auf `linear` (W_f,t = G_f × L_f,t, dann Exponential-Mapping in den Gewichten selbst).

---

### `reconstruction.window_function`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `hanning` |
| **Default** | `"hanning"` |
| **Konstant** | Ja |

**Zweck:** Fensterfunktion für Tile-Overlap-Add. Fest auf **Hanning** (2D separabel).

**Formel:** `w(i) = 0.5 × (1 − cos(2π·i / (n−1)))`

---

## 15. Debayer

### `debayer`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Demosaicing des finalen gestackten CFA-Mosaiks in Phase 11.

- **`true`**: Nearest-Neighbor-Demosaic → R/G/B FITS + `stacked_rgb.fits` (3-Plane FITS-Cube)
- **`false`**: Nur CFA-Mosaik `stacked.fits` wird erzeugt (für externe Debayer-Tools)

**Hinweis:** Bei `color_mode=MONO` wird die Debayer-Phase als "ok/MONO" beendet ohne Aktion.

---

## 16. Astrometry

### `astrometry.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Aktiviert Plate Solving (WCS).

---

### `astrometry.astap_bin`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `""` |

**Zweck:** Pfad zur ASTAP-CLI. Leer bedeutet: Systempfad/Standardauflösung.

---

### `astrometry.astap_data_dir`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `""` |

**Zweck:** ASTAP-Datenverzeichnis. Leer bedeutet: Standardpfad.

---

### `astrometry.search_radius`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 1 – 360 |
| **Default** | `180` |

**Zweck:** Suchradius in Grad für das Solving (180 = blind solve).

---

## 17. PCC

### `pcc.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

---

### `pcc.source`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `auto`, `siril`, `vizier_gaia`, `vizier_apass` |
| **Default** | `"auto"` |

---

### `pcc.mag_limit`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 1 – 22 |
| **Default** | `14.0` |

---

### `pcc.mag_bright_limit`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 15 |
| **Default** | `6.0` |

---

### `pcc.aperture_radius_px`, `pcc.annulus_inner_px`, `pcc.annulus_outer_px`

| Key | Typ | Default | Constraint |
|-----|-----|---------|------------|
| `pcc.aperture_radius_px` | number | `8.0` | >0 |
| `pcc.annulus_inner_px` | number | `12.0` | >0 |
| `pcc.annulus_outer_px` | number | `18.0` | >0 |

---

### `pcc.min_stars`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 3 |
| **Default** | `10` |

---

### `pcc.sigma_clip`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `2.5` |

---

### `pcc.siril_catalog_dir`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `""` |

**Zweck:** Lokaler Siril-Katalogpfad; leer = Standardpfad.

---

## 18. Stacking

Finales Stacking der synthetischen Frames (Phase 10: STACKING).

### `stacking.method`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `rej`, `average` |
| **Default** | `"rej"` |

**Zweck:** Stacking-Methode.

| Methode | Beschreibung | Empfehlung |
|---------|-------------|------------|
| **`rej`** | Sigma-Clipping Rejection → dann Mittelwert | **Empfohlen** — entfernt verbleibende Ausreißer |
| **`average`** | Einfacher linearer Mittelwert | Schneller, aber keine Ausreißer-Entfernung |

---

### `stacking.sigma_clip.sigma_low`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `2.0` |

**Zweck:** Unterer Sigma-Schwellenwert für Rejection.

**Formel:** Pixel wird abgelehnt wenn `z < -sigma_low` (z = normalisierte Abweichung vom Median).

**Empfehlung:** 2.0 – 4.0

---

### `stacking.sigma_clip.sigma_high`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `2.0` |

**Zweck:** Oberer Sigma-Schwellenwert für Rejection.

**Formel:** Pixel wird abgelehnt wenn `z > sigma_high`

**Empfehlung:** 2.0 – 4.0

---

### `stacking.sigma_clip.max_iters`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 1 – 10 |
| **Default** | `3` |

**Zweck:** Maximale Sigma-Clipping-Iterationen. Nach jeder Iteration werden abgelehnte Pixel entfernt und der Median neu berechnet.

---

### `stacking.sigma_clip.min_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 1 |
| **Default** | `0.5` |

**Zweck:** Minimale überlebende Frame-Fraktion pro Pixel.

**Verhalten:** Wenn weniger als `min_fraction × N` Frames an einem Pixel überleben, wird auf den **unclipped mean** zurückgefallen (verhindert Artefakte durch zu aggressives Clipping).

---

### `stacking.cluster_quality_weighting.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Aktiviert die v3.2.2-Qualitätsgewichtung für die finale Aggregation synthetischer Cluster-Frames.

**Formel:** `w_k = exp(kappa_cluster * Q_k)`

- `Q_k` = Cluster-Qualitätswert (typisch in `[-3, +3]`)
- `enabled=false`: kein Qualitäts-Weighting, klassisches finales Stacking nach `stacking.method`
- `enabled=true`: finale Aggregation per gewichteter Mittelung; Sigma-Clipping-Stacking wird in dieser Stufe nicht verwendet

---

### `stacking.cluster_quality_weighting.kappa_cluster`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `1.0` |

**Zweck:** Exponent-Faktor für den Einfluss von `Q_k` auf das Gewicht `w_k`.

- größerer Wert → stärkere Trennung guter/schlechter Cluster
- kleinerer Wert → flachere Gewichtsverteilung

**Praktische Sensitivität** (bei `Q_k`-Spanne ungefähr `[-3, +3]`):

| κ | max weight ratio (≈ `e^{6κ}`) | Charakter |
|---:|---:|---|
| 0.3 | ~ `e^{1.8}` ≈ 6 | sehr mild |
| 0.5 | ~ `e^{3}` ≈ 20 | moderat |
| 1.0 | ~ `e^{6}` ≈ 403 | stark |
| 1.5 | ~ `e^{9}` ≈ 8103 | sehr aggressiv |
| 2.0 | ~ `e^{12}` ≈ 162k | praktisch Winner-takes-most |

**Empfehlung (Astrofotografie):**

- Default: `κ = 0.5 ... 1.0`
- `κ = 1.2` nur bei bewusst gewünschtem lucky-imaging-artigem Verhalten
- `κ >= 1.5` oft instabil (numerisch und statistisch)

---

### `stacking.cluster_quality_weighting.cap_enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Aktiviert optional ein Dominanz-Cap für Clustergewichte, damit einzelne Cluster das Endergebnis nicht übermäßig dominieren.

---

### `stacking.cluster_quality_weighting.cap_ratio`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `20.0` |

**Zweck:** Gewichtslimit bei aktivem Cap.

**Formel (nur wenn `cap_enabled=true`):** `w_k <= cap_ratio * median_j(w_j)`

**Praktische Bereiche für `r_cap`:**

| `r_cap` | Verhalten |
|---:|---|
| 5 | sehr konservativ |
| 10 | mild begrenzt |
| 20 | moderat |
| 50 | kaum Eingriff |
| >100 | faktisch deaktiviert |

**Empfehlung:**

- Konservativ stabil: `r_cap = 10`
- Balanciert: `r_cap = 20-30`
- Fast unbegrenzt: `r_cap >= 50`

---

### `stacking.common_overlap_required_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | `(0, 1]` |
| **Default** | `1.0` |

**Zweck:** Definiert die erforderliche Pixel-Abdeckung über nutzbare Frames für alle Berechnungen nach PREWARP.

- `1.0` (empfohlen): strikte Schnittmenge aller nutzbaren Frames
- `< 1.0`: lässt teilweise überdeckte Rand-/Canvas-Bereiche in Statistiken und Tile-Verarbeitung zu

**Empfehlung:** Für Feldrotation (Alt/Az) auf `1.0` lassen, damit keine geometriebedingten Bias- oder Stripe/Grid-Artefakte entstehen.

---

### `stacking.tile_common_valid_min_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | `(0, 1]` |
| **Default** | `0.9` |

**Zweck:** Tile-Akzeptanzschwelle nach Anwendung der Common-Overlap-Maske.

- Ein Tile wird nur verwendet, wenn mindestens dieser Anteil seiner Pixel im gemeinsamen Overlap liegt
- Höhere Werte sind strikter und reduzieren Randkontamination

**Empfehlung:** `0.9` für Produktion; `0.75-0.85` nur bewusst, wenn mehr Randabdeckung zugelassen werden soll.

---

### `stacking.output_stretch`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Optionales lineares Display-Stretching auf den Ausgabedaten (Post-Processing, nicht Teil des linearen Kerns).

---

### `stacking.cosmetic_correction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Optionale kosmetische Korrektur (z. B. Hotpixel) nach dem Stacking.

---

### `stacking.cosmetic_correction_sigma`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `5.0` |

**Zweck:** Schwellwert (MAD-Sigma) für `stacking.cosmetic_correction`.

- Niedrigerer Wert = aggressiver.
- **Hinweis:** Im gestackten Bild können helle Objektkerne hohe lokale Kontraste haben. Zu aggressive Werte können reale Signalspitzen fälschlich als Hotpixel behandeln.

**Empfehlung:**

- MONO / kalibrierte Daten: `5.0`
- OSC / Smart-Telescope ohne Darks: `10.0` (konservativer)

---

### `stacking.per_frame_cosmetic_correction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Hotpixel-Korrektur **pro Frame vor PREWARP/Stacking**.

Diese Option zielt auf **fixe Sensordefekte** (RGB-Einzelpixel), die in jedem Frame an der gleichen Position auftreten und deshalb durch Sigma-Clipping im Stack nicht zuverlässig entfernt werden.

---

### `stacking.per_frame_cosmetic_correction_sigma`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `5.0` |

**Zweck:** Schwellwert (MAD-Sigma) für `stacking.per_frame_cosmetic_correction`.

**Empfehlung:** `5.0` (OSC/Seestar/DWARF: typischerweise passend).

---

## 19. Validation

Qualitätsprüfung des Rekonstruktionsergebnisses (nach Phase 10, vor Debayer).

### `validation.min_fwhm_improvement_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Default** | `0.0` |

**Zweck:** Minimale FWHM-Verbesserung in Prozent (Output-FWHM vs. Seeing-FWHM).

**Formel:** `improvement = (seeing_fwhm - output_fwhm) / seeing_fwhm × 100%`

**Verhalten:** Unterschreitung → `fwhm_improvement_ok = false` → `validation_failed`

---

### `validation.max_background_rms_increase_percent`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Default** | `0.0` |

**Zweck:** Maximale erlaubte Hintergrund-RMS-Erhöhung in Prozent. `0.0` = nicht geprüft.

---

### `validation.min_tile_weight_variance`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Default** | `0.1` |

**Zweck:** Minimale normalisierte Varianz der Tile-Gewichte.

**Formel:** `tile_weight_variance = Var(mean_W_t) / mean(mean_W_t)²`

**Interpretation:** Zu niedrige Varianz → Gewichtung hatte keinen Effekt → entweder alle Frames gleich gut oder Metriken nicht diskriminativ genug.

---

### `validation.require_no_tile_pattern`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Prüft auf sichtbare Tile-Muster im Endergebnis mittels Sobel-Gradient an Tile-Grenzen.

**Methode:** Vergleicht den mittleren Sobel-Gradient an Tile-Grenzen mit dem Gradient 2 Pixel daneben. Ratio > 1.5 → Tile-Pattern erkannt.

**Verhalten:** `tile_pattern_ok = false` → `validation_failed` (Pipeline läuft trotzdem weiter für Debayer)

---

## 20. Runtime Limits

Laufzeit-Beschränkungen.

### `runtime_limits.parallel_workers`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `4` |

**Zweck:** Maximale Anzahl paralleler Worker für Tile-lastige Phasen.

**Hinweis:** Wird zusätzlich durch CPU-Kernanzahl und (bei OSC) durch `runtime_limits.memory_budget` begrenzt.

---

### `runtime_limits.memory_budget`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Einheit** | MiB |
| **Default** | `512` |

**Zweck:** Speicherbudget für den OSC-Memory-Cap in der Tile-Rekonstruktion. Reduziert bei Bedarf die effektive Anzahl paralleler Worker, um RAM-Spitzen zu begrenzen.

### `runtime_limits.tile_analysis_max_factor_vs_stack`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `3.0` |

**Zweck:** Maximaler Zeitfaktor für Tile-Analyse relativ zum einfachen Stack. Wenn die Tile-Analyse länger als `factor × stack_time` dauert, wird eine Warnung erzeugt.

---

### `runtime_limits.hard_abort_hours`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `6.0` |

**Zweck:** Maximale Laufzeit in Stunden bevor die Pipeline abgebrochen wird (Hard Limit).

---

### `runtime_limits.allow_emergency_mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Erlaubt den Emergency Mode bei sehr kleinen Datensätzen (<50 nutzbare Frames). Wenn `false`, wird stattdessen kontrolliert abgebrochen.

---

## Beispiel-Konfiguration (Vollständig)

```yaml
# Pipeline
pipeline:
  mode: production
  abort_on_fail: true

# Output
output:
  registered_dir: registered
  artifacts_dir: artifacts
  write_registered_frames: false
  write_global_metrics: true
  write_global_registration: true

# Data
data:
  color_mode: OSC
  bayer_pattern: GBRG
  linear_required: true

# Linearity
linearity:
  enabled: true
  max_frames: 8
  min_overall_linearity: 0.9
  strictness: strict

# Assumptions
assumptions:
  frames_min: 50
  frames_optimal: 800
  frames_reduced_threshold: 200
  reduced_mode_skip_clustering: true

# Normalization
normalization:
  enabled: true
  mode: background
  per_channel: true

# Registration
registration:
  engine: triangle_star_matching
  allow_rotation: true
  star_topk: 120
  star_min_inliers: 6
  star_inlier_tol_px: 2.5
  star_dist_bin_px: 2.5

# Dithering
dithering:
  enabled: true
  min_shift_px: 0.7

# Tile Denoise
tile_denoise:
  soft_threshold:
    enabled: true
    blur_kernel: 31
    alpha: 1.5
    skip_star_tiles: true
  wiener:
    enabled: false
    snr_threshold: 5.0
    q_min: -0.5
    q_max: 1.0
    q_step: 0.1
    min_snr: 2.0
    max_iterations: 10

# Chroma Denoise
chroma_denoise:
  enabled: true
  color_space: ycbcr_linear
  apply_stage: post_stack_linear
  protect_luma: true
  luma_guard_strength: 0.75
  star_protection:
    enabled: true
    threshold_sigma: 2.2
    dilate_px: 2
  structure_protection:
    enabled: true
    gradient_percentile: 85
  chroma_wavelet:
    enabled: true
    levels: 3
    threshold_scale: 1.25
    soft_k: 1.0
  chroma_bilateral:
    enabled: true
    sigma_spatial: 1.2
    sigma_range: 0.035
  blend:
    mode: chroma_only
    amount: 0.85

# Global Metrics
global_metrics:
  adaptive_weights: false
  weight_exponent_scale: 1.0
  weights:
    background: 0.4
    noise: 0.3
    gradient: 0.3
  clamp: [-3.0, 3.0]

# Tile
tile:
  size_factor: 32
  min_size: 64
  max_divisor: 6
  overlap_fraction: 0.25
  star_min_count: 10

# Local Metrics
local_metrics:
  clamp: [-3.0, 3.0]
  star_mode:
    weights:
      fwhm: 0.6
      roundness: 0.2
      contrast: 0.2
  structure_mode:
    metric_weight: 0.7
    background_weight: 0.3

# Synthetic
synthetic:
  weighting: global
  frames_min: 5
  frames_max: 30
  clustering:
    mode: kmeans
    cluster_count_range: [5, 30]

# Reconstruction (fest)
reconstruction:
  weighting_function: linear
  window_function: hanning

# Debayer
debayer: true

# Astrometry
astrometry:
  enabled: false
  astap_bin: ""
  astap_data_dir: ""
  search_radius: 180

# PCC
pcc:
  enabled: false
  source: auto
  mag_limit: 14.0
  mag_bright_limit: 6.0
  aperture_radius_px: 8.0
  annulus_inner_px: 12.0
  annulus_outer_px: 18.0
  min_stars: 10
  sigma_clip: 2.5
  siril_catalog_dir: ""

# Stacking
stacking:
  method: rej
  sigma_clip:
    sigma_low: 2.0
    sigma_high: 2.0
    max_iters: 3
    min_fraction: 0.5
  cluster_quality_weighting:
    enabled: true
    kappa_cluster: 1.0
    cap_enabled: false
    cap_ratio: 20.0
  output_stretch: false
  cosmetic_correction: false

# Validation
validation:
  min_fwhm_improvement_percent: 0.0
  max_background_rms_increase_percent: 0.0
  min_tile_weight_variance: 0.1
  require_no_tile_pattern: true

# Runtime Limits
runtime_limits:
  parallel_workers: 8
  memory_budget: 1024
  tile_analysis_max_factor_vs_stack: 3.0
  hard_abort_hours: 6.0
  allow_emergency_mode: false
```

---

## Hinweise

### Abweichungen `tile_compile.yaml` vs. C++ Defaults

Die Datei `tile_compile.yaml` im Repository enthält eine **Test-/Debug-Konfiguration** die von den C++ Defaults abweicht:

| Key | `tile_compile.yaml` | C++ Default | Bemerkung |
|-----|---------------------|-------------|-----------|
| `pipeline.abort_on_fail` | `false` | `true` | Debug-freundlich |
| `output.write_registered_frames` | `true` | `false` | Speicherintensiv |
| `global_metrics.weights.background` | `0.45` | `0.4` | Abweichende Gewichtung |
| `global_metrics.weights.noise` | `0.35` | `0.3` | Abweichende Gewichtung |
| `global_metrics.weights.gradient` | `0.20` | `0.3` | Abweichende Gewichtung |
| `registration.star_topk` | `100` | `120` | Weniger Sterne |
| `registration.star_inlier_tol_px` | `3.0` | `2.5` | Toleranter |
| `registration.star_dist_bin_px` | `5.0` | `2.5` | Größere Bins |

### Schema-Validierung

Die Schema-Dateien (`tile_compile.schema.json`, `tile_compile.schema.yaml`) definieren die erlaubten Typen und Wertebereiche. Die C++ Implementierung in `Config::validate()` prüft zusätzlich:

- **Gewichts-Normierung:** α + β + γ = 1.0 (global_metrics.weights)
- **Clamp-Ordnung:** clamp[0] < clamp[1]
- **Normalisierung Pflicht:** `normalization.enabled` muss `true` sein

### Quellen

- **C++ Defaults:** `tile_compile_cpp/include/tile_compile/config/configuration.hpp`
- **Config Parsing:** `tile_compile_cpp/src/io/config.cpp`
- **JSON-Schema:** `tile_compile_cpp/tile_compile.schema.json`
- **YAML-Schema:** `tile_compile_cpp/tile_compile.schema.yaml`
- **Beispiel-Config:** `tile_compile_cpp/tile_compile.yaml`

---

## Anhang A — Funktionsdetails für alle Optionen

Dieser Anhang beschreibt pro Schlüssel explizit das **Laufzeitverhalten** (Wirkung in der Pipeline, Interaktionen, Nebenwirkungen).

### A.1 Pipeline / Output / Data

- `pipeline.mode`: wählt Produktions- vs. Testpfad (gleiche Kernphasen, anderes Striktheits-/Debug-Profil).
- `pipeline.abort_on_fail`: steuert, ob bei `phase_end(error)` sofort abgebrochen wird.
- `output.registered_dir`: Ziel-Unterordner für registrierte Frame-Ausgaben.
- `output.artifacts_dir`: Ziel-Unterordner für JSON-Artefakte (`global_metrics.json`, `tile_reconstruction.json`, ...).
- `output.write_registered_frames`: schreibt pro Frame registrierte FITS; erhöht IO- und Speicherbedarf stark.
- `output.write_global_metrics`: aktiviert Schreiben globaler Metrikvektoren (Frame-Qualitätsdiagnose).
- `output.write_global_registration`: aktiviert Schreiben globaler Warp/CC-Diagnose.
- `data.image_width`, `data.image_height`: optionale Erwartungswerte; normalerweise FITS-headerbasiert erkannt.
- `data.frames_min`: Vorab-Sanity-Schwelle für minimale Input-Anzahl.
- `data.frames_target`: rein informatives Ziel; erzwingt allein keinen Ausschluss.
- `data.color_mode`: erwarteter Aufnahmemodus; Laufzeit-Autodetektion kann mit Warnung übersteuern.
- `data.bayer_pattern`: CFA-Layout für OSC-Verarbeitung und korrekte Farbrekonstruktion.
- `data.linear_required`: koppelt Policy für Linearitätsanforderung an Linearity-Diagnostik.

### A.2 Linearity / Calibration / Assumptions

- `linearity.enabled`: aktiviert Linearitätsdiagnostik in Scan/Frühvalidierung.
- `linearity.max_frames`: Stichprobengröße der Linearitätsprüfung (Speed vs. Sicherheit).
- `linearity.min_overall_linearity`: Schwellwert für Linearity-Pass/Fail.
- `linearity.strictness`: Policy-Mapping (Fail/Warn/Ignore-Verhalten).
- `calibration.use_bias`, `use_dark`, `use_flat`: schaltet jeweilige Master-Kalibrierstufe ein.
- `calibration.bias_use_master`, `dark_use_master`, `flat_use_master`: nutzt explizite Masterdateien statt Directory-Stacking.
- `calibration.dark_auto_select`: automatische Dark-Master-Auswahl nach Belichtungszeit (optional Temperatur).
- `calibration.dark_match_exposure_tolerance_percent`: erlaubte Belichtungsabweichung beim Dark-Matching.
- `calibration.dark_match_use_temp`: aktiviert temperaturbasiertes Dark-Matching.
- `calibration.dark_match_temp_tolerance_c`: erlaubte Temperaturabweichung bei aktivem Temp-Matching.
- `calibration.bias_dir`, `darks_dir`, `flats_dir`: Quellordner für Kalibrierframe-Findung.
- `calibration.bias_master`, `dark_master`, `flat_master`: explizite Pfade zu Master-Kalibrierframes.
- `calibration.pattern`: Glob-Muster für Kalibrierdatei-Lookup.
- `assumptions.frames_min`: Mindestrahmenzahl-Erwartung für stabile Methodik.
- `assumptions.frames_optimal`: Zielrahmenzahl für volle Qualitätsstabilität.
- `assumptions.frames_reduced_threshold`: Umschaltpunkt Reduced- vs. Full-Mode.
- `assumptions.exposure_time_tolerance_percent`: tolerierte Sub-Exposure-Streuung.
- `assumptions.reduced_mode_skip_clustering`: deaktiviert teures State-Clustering im Reduced-Mode.
- `assumptions.reduced_mode_cluster_range`: begrenzter K-Suchraum falls Clustering im Reduced-Mode läuft.

### A.3 Normalization / Registration / Dithering

- `normalization.enabled`: methodikgetrieben verpflichtend (normalerweise aktiv lassen).
- `normalization.mode`: Hintergrund- vs. Median-zentrierte Normalisierungsstrategie.
- `normalization.per_channel`: kanalweise (OSC/RGB) Normalisierung zur Balance-Erhaltung.
- `registration.engine`: bevorzugte Startmethode; Laufzeit nutzt trotzdem Fallback-Kaskade.
- `registration.allow_rotation`: erlaubt Rotationsanteile in globalen Warps (Pflicht für Alt/Az).
- `registration.star_topk`: Anzahl starker Sterne für sternbasierte Engines.
- `registration.star_min_inliers`: minimale akzeptierte Inlier-Korrespondenzen.
- `registration.star_inlier_tol_px`: geometrische Inlier-Toleranz.
- `registration.star_dist_bin_px`: Distanzhistogramm-Quantisierung für Star-Similarity.
- `registration.reject_outliers`: robustes Verwerfen unplausibler Warps nach Matching.
- `registration.reject_cc_min_abs`: absolute NCC-Untergrenze in Outlier-Logik.
- `registration.reject_cc_mad_multiplier`: MAD-Skalierung für robusten CC-Schwellwert.
- `registration.reject_shift_px_min`: absolute Shift-Untergrenze für Shift-Outlier.
- `registration.reject_shift_median_multiplier`: relativer Shift-Schwellwert zur Median-Shift.
- `registration.reject_scale_min`, `reject_scale_max`: erlaubtes Similarity-Scale-Band.
- `dithering.enabled`: aktiviert Dither-Diagnostik im Registration-Artefakt.
- `dithering.min_shift_px`: Mindest-Frame-Shift, um als Dither zu zählen.

### A.4 Tile Denoise / Chroma Denoise

- `tile_denoise.soft_threshold.enabled`: aktiviert räumliches Highpass-Soft-Thresholding.
- `tile_denoise.soft_threshold.blur_kernel`: Kernelgröße für Hintergrundschätzung des Residuals.
- `tile_denoise.soft_threshold.alpha`: Denoise-Stärke (`tau = alpha * sigma`).
- `tile_denoise.soft_threshold.skip_star_tiles`: überspringt Denoise auf stern-dominierten Tiles.
- `tile_denoise.wiener.enabled`: aktiviert frequenzdomänigen Wiener-Zweig.
- `tile_denoise.wiener.snr_threshold`: Wiener-Gate; niedrige SNR wird eher gefiltert.
- `tile_denoise.wiener.q_min`, `q_max`, `q_step`: interner Wiener-Qualitätssuchraum.
- `tile_denoise.wiener.min_snr`: minimale SNR für stabile Wiener-Parametrisierung.
- `tile_denoise.wiener.max_iterations`: Iterationsobergrenze des Wiener-Tunings.
- `chroma_denoise.enabled`: aktiviert chroma-fokussierte Denoise (OSC-Pfad).
- `chroma_denoise.color_space`: Chroma/Luma-Transform (`ycbcr_linear` oder `opponent_linear`).
- `chroma_denoise.apply_stage`: Ausführung vor Tile-OLA oder nach finalem linearem Stack.
- `chroma_denoise.protect_luma`: schützt Luminanzstrukturen vor Chroma-Nebenwirkungen.
- `chroma_denoise.luma_guard_strength`: Stärke der Luma-Schutzmaske.
- `chroma_denoise.star_protection.enabled`: Sternmasken-Schutz für Kerne/Halos.
- `chroma_denoise.star_protection.threshold_sigma`: Schwelle für Sternmaskenbildung.
- `chroma_denoise.star_protection.dilate_px`: Ausdehnungsradius der Sternmaske.
- `chroma_denoise.structure_protection.enabled`: kanten-/strukturabhängiger Chroma-Schutz.
- `chroma_denoise.structure_protection.gradient_percentile`: Gradient-Cutoff für Strukturmaske.
- `chroma_denoise.chroma_wavelet.enabled`: Wavelet-basierte Chroma-Dämpfung.
- `chroma_denoise.chroma_wavelet.levels`: Anzahl Wavelet-Zerlegungsebenen.
- `chroma_denoise.chroma_wavelet.threshold_scale`: Wavelet-Schwellen-Multiplikator.
- `chroma_denoise.chroma_wavelet.soft_k`: Weichheit der Wavelet-Schrumpfung.
- `chroma_denoise.chroma_bilateral.enabled`: bilaterale Glättung auf Chroma-Komponenten.
- `chroma_denoise.chroma_bilateral.sigma_spatial`: räumliche bilaterale Stärke.
- `chroma_denoise.chroma_bilateral.sigma_range`: Farbdistanz-Selektivität bilateral.
- `chroma_denoise.blend.mode`: aktuell chroma-only Blending-Modus.
- `chroma_denoise.blend.amount`: Mischanteil Original vs. denoised Chroma.

### A.5 Global/Local Metrics / Tile / Synthetic / Reconstruction

- `global_metrics.weights.background`, `noise`, `gradient`: gewichtete Terme der globalen Frame-Qualität.
- `global_metrics.clamp`: harte Grenzen vor exponentiellem Weight-Mapping.
- `global_metrics.adaptive_weights`: Auto-Anpassung der Metrikgewichte aus Streuung.
- `global_metrics.weight_exponent_scale`: Trennschärfe in `exp(k*Q)`.
- `tile.size_factor`: Grund-Tilegröße aus gemessenem Seeing/FWHM.
- `tile.min_size`: Untergrenze gegen zu kleine, instabile Tiles.
- `tile.max_divisor`: Obergrenze über Bildseiten-Divisor.
- `tile.overlap_fraction`: Überlappungsanteil für glattes Overlap-Add.
- `tile.star_min_count`: Schwelle STAR vs. STRUCTURE Tile-Klasse.
- `local_metrics.clamp`: Clamp lokaler Qualität vor Gewichtsbildung.
- `local_metrics.star_mode.weights.fwhm`, `roundness`, `contrast`: STAR-Tile-Qualitätsmischung.
- `local_metrics.structure_mode.metric_weight`, `background_weight`: STRUCTURE-Tile-Qualitätsmischung.
- `synthetic.weighting`: Methode zur Synthetic-Frame-Erzeugung (`global` vs `tile_weighted`).
- `synthetic.frames_min`: minimale Clustergröße für Synthetic-Output.
- `synthetic.frames_max`: maximale Anzahl Synthetic-Outputs.
- `synthetic.clustering.mode`: Clustering-Backend für Zustandsgruppen.
- `synthetic.clustering.cluster_count_range`: erlaubtes K-Suchfenster.
- `reconstruction.weighting_function`: Rekonstruktionsgewichtsmodell (aktuell linear).
- `reconstruction.window_function`: Fensterkern für Overlap-Add (aktuell Hanning).

### A.6 Debayer / Astrometry / PCC / Stacking / Validation / Runtime

- `debayer`: aktiviert finale OSC-CFA→RGB-Konvertierung.
- `astrometry.enabled`: aktiviert Plate-Solving-Phase.
- `astrometry.astap_bin`: Pfad zur ASTAP-Binary.
- `astrometry.astap_data_dir`: Pfad zu ASTAP-Katalogdaten.
- `astrometry.search_radius`: Blind-Solve vs. eingeschränkter Suchradius.
- `pcc.enabled`: aktiviert photometrische Farbkalibrierung.
- `pcc.source`: Katalog-/Provider-Auswahl.
- `pcc.mag_limit`, `mag_bright_limit`: Magnitudengrenzen für Sternselektion.
- `pcc.aperture_radius_px`, `annulus_inner_px`, `annulus_outer_px`: Photometrie-Aperturgeometrie.
- `pcc.min_stars`: Mindestanzahl gültiger Sterne für stabilen PCC-Fit.
- `pcc.sigma_clip`: Outlier-Rejection im PCC-Fit.
- `pcc.siril_catalog_dir`: optionaler lokaler Siril-Katalogpfad.
- `stacking.method`: finaler Kombinationsmodus (`rej` vs `average`).
- `stacking.sigma_clip.sigma_low`, `sigma_high`: untere/obere Rejection-Schwellen.
- `stacking.sigma_clip.max_iters`: maximale Clip-Iterationen.
- `stacking.sigma_clip.min_fraction`: Mindestanteil verbleibender Samples (Fallback-Schutz).
- `stacking.cluster_quality_weighting.enabled`: aktiviert Synthetic-Cluster-Qualitätsgewichtung.
- `stacking.cluster_quality_weighting.kappa_cluster`: Exponent der Qualitätsgewichtung.
- `stacking.cluster_quality_weighting.cap_enabled`: expliziter Dominanz-Cap-Schalter.
- `stacking.cluster_quality_weighting.cap_ratio`: Dominanz-Cap-Level bei aktivem Cap.
- **Laufzeit-Schutz:** Für Synthetic-Stacking wird standardmäßig ein Dominanz-Cap angewendet, auch wenn `cap_enabled=false`, um Dynamik-Kollaps diffuser Signale zu verhindern.
- `stacking.output_stretch`: optionales display-orientiertes Post-Scaling auf 16-bit-Spanne.
- `stacking.cosmetic_correction`: optionale Hotpixel-artige Korrektur nach dem Stacking.
- `stacking.cosmetic_correction_sigma`: Schwellwert der Cosmetic-Korrektur.
- `validation.min_fwhm_improvement_percent`: notwendige Schärfeverbesserung.
- `validation.max_background_rms_increase_percent`: Schutz gegen Hintergrundverschlechterung.
- `validation.min_tile_weight_variance`: Sanity-Check gegen degenerierte lokale Gewichte.
- `validation.require_no_tile_pattern`: Checker/Grid-Artefakt-Prüfung.
- `runtime_limits.parallel_workers`: Obergrenze für Worker-Threads.
- `runtime_limits.memory_budget`: Speicherbudget, das effektive Parallelität begrenzen kann.
- `runtime_limits.tile_analysis_max_factor_vs_stack`: Warnschwelle für Laufzeit-Anomalien.
- `runtime_limits.hard_abort_hours`: absolute Runtime-Sicherheitsgrenze.
- `runtime_limits.allow_emergency_mode`: erlaubt Verarbeitung unterhalb normaler Annahmen.