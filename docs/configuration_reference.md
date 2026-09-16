# Tile-Compile C++ Konfigurationsreferenz

Diese Dokumentation beschreibt alle Konfigurationsoptionen für `tile_compile.yaml` basierend auf der C++ Implementierung in `configuration.hpp` und den Schema-Dateien `tile_compile.schema.json` / `tile_compile.schema.yaml`.

**Quelle der Wahrheit für Defaults:** `include/tile_compile/config/configuration.hpp`  
**Schema-Version:** v3  
**Referenz:** Methodik v3.3

**Dokumentationsstand:**
- Single-Method-Pipeline: CFA Forward Drizzle + Multiband ist die einzige
  Rekonstruktionsmethode. Die alten `aqmh.*`-, Classic- und `method:`-Bloecke
  existieren nicht mehr.
- `data.bayer_pattern` ist standardmäßig `auto`; FITS-Header (`BAYERPAT`/`COLORTYP`) haben Vorrang vor dem Config-Wert.
- `data.linear_required` entfernt (deprecated; nicht-lineare Frames werden nur noch gewarnt).
- `bge.fit.robust_loss` und `bge.fit.huber_delta` sind als Benutzerparameter dokumentiert und konfigurierbar.
- `bge.min_valid_sample_fraction_for_apply` und `bge.min_valid_samples_for_apply` sind als kanalweise BGE-Apply-Grenzwerte dokumentiert.
- PCC-Dokumentation umfasst die aktiven Stabilitäts- und Apply-Parameter (`max_condition_number`, `max_residual_rms`, `apply_attenuation`, `chroma_strength`, `k_max`).
- `bge.tile_weight_lambda_structure` ist auf den aktuellen Default `1.0` abgeglichen.
- Die Classic-/AQMH-Stacking-Parameter wurden entfernt; `stacking.common_overlap_required_fraction` lebt als `reconstruction.common_overlap_required_fraction` weiter (§1, §14).


**💡 Für praktische Beispiele und Anwendungsfälle siehe:** [Konfigurationsbeispiele & Best Practices](configuration_examples_practical_de.md)

## Inhaltsverzeichnis

1. [Pipeline](#1-pipeline)
2. [Output](#2-output)
3. [Data](#3-data)
4. [Linearity](#4-linearity)
5. [Calibration](#5-calibration)
7. [Normalization](#7-normalization)
8. [Registration](#8-registration)
9b. [Chroma Denoise](#chroma-denoise)
10. [Global Metrics](#10-global-metrics)
14. [Reconstruction](#14-reconstruction)
15. [Debayer](#15-debayer)
16. [Astrometry](#16-astrometry)
17. [BGE (Background Gradient Extraction)](#17-bge-background-gradient-extraction)
18. [PCC](#18-pcc)
19. [HyperMetric Stretch](#19-hypermetric-stretch)
20. [Stacking](#20-stacking)
22. [Runtime Limits](#22-runtime-limits)
23. [Raw Stack / Preprocessing](#raw-stack-preprocessing)

---

## 1. Pipeline

Die Pipeline ist fest auf eine einzige Rekonstruktionsmethode ausgelegt:
**CFA Forward Drizzle + Multiband** (`tile_compile_runner reconstruct`).
Es gibt keinen Methoden-Selektor mehr.

### Entfernte Legacy-Schluessel (Migration)

Die folgenden Schluessel wurden mit dem Single-Method-Cutover entfernt.
Beim Laden werden sie — falls vorhanden — mit einer Warnung verworfen;
`tile_compile_cli migrate-config <in> <out>` schreibt eine bereinigte
Konfiguration. Semantische Methoden-Selektoren werden fail-closed abgelehnt.

| Alter Schluessel | Status |
|------------------|--------|
| `method` | **Fail-closed** — es gibt genau eine Methode; Schluessel entfernen oder migrieren |
| `reconstruction.engine` | **Fail-closed** — Methoden-Selektor |
| `aqmh.*` | entfernt; der ganze Block wird mit Warnung verworfen |
| `pipeline.*` | entfernt (unbenutzt) |
| `assumptions.*` | entfernt (Reduced-/Emergency-Gating entfallen) |
| `tile.*`, `tile_denoise.*`, `local_metrics.*`, `synthetic.*`, `validation.*` | entfernt (Classic-Tile-Pfad) |
| `runtime_limits.allow_emergency_mode` | entfernt |
| `runtime_limits.tile_analysis_max_factor_vs_stack`, `runtime_limits.tile_reconstruction_diagnostics` | entfernt |
| `stacking.method`, `stacking.sigma_clip.*`, `stacking.cluster_quality_weighting.*`, `stacking.output_stretch`, `stacking.tile_common_valid_min_fraction`, `stacking.cosmetic_correction*` | entfernt (STACKING ist Pass-Through) |

**Zugeordnete (migrierte) Schluessel:**

| Alt | Neu |
|-----|-----|
| `stacking.common_overlap_required_fraction` | `reconstruction.common_overlap_required_fraction` |

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

### `output.write_registered_frames`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Registrierte Frames als FITS auf Disk schreiben.

- **`true`**: Jeder registrierte Frame wird als `reg_XXXXX.fit` gespeichert — **hoher Speicherbedarf!**
- **`false`**: Registrierte Frames werden nur im Speicher gehalten

**Hinweis:** In `tile_compile.yaml` steht `true`, der C++ Default ist `false`. Nur für Debugging empfohlen.

---

### `output.crop_to_nonzero_bbox`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Finalen Stack auf die Bounding Box aller nicht-null Pixel zuschneiden.

- **`true`**: Entfernt leere Ränder vom finalen Bild. Nur Pixel mit Werten > 0 werden behalten. Reduziert die Dateigröße und entfernt unnötige schwarze Ränder.
- **`false`**: Behält die volle Canvas-Größe bei, inklusive leerer Ränder.

**Hinweis:** Die Funktion wird nach der Stack-Phase aber vor dem Debayer (bei OSC) angewendet. Die Tile-Offsets werden entsprechend angepasst. Ungueltige Canvas-Bereiche bleiben maskiert und sollen nicht in nachgelagerte Berechnungen wie Debayer/BGE/PCC einfliessen.

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
| **Typ** | string (enum) |
| **Werte** | `auto`, `RGGB`, `BGGR`, `GRBG`, `GBRG`, `NONE` |
| **Default** | `"auto"` |

**Zweck:** Bayer-Pattern für OSC-Kameras. Bestimmt die Farbfilter-Anordnung auf dem Sensor. `NONE` für monochrome Daten, `auto` für automatische Erkennung aus dem FITS-Header.

| Pattern | Zeile 0 | Zeile 1 |
|---------|---------|---------|
| `RGGB` | R G | G B |
| `BGGR` | B G | G R |
| `GBRG` | G B | R G |
| `GRBG` | G R | B G |

**Laufzeitverhalten:** Für OSC-Daten liest der Runner zuerst die FITS-Header-Felder `BAYERPAT` und `COLORTYP`. Sind sie vorhanden, wird dieser Wert verwendet und der Config-Wert ignoriert. Enthält der Header keine Bayer-Metadaten, wird der konfigurierte Wert als Fallback genutzt. Bei `bayer_pattern: auto` ohne Header-Metadaten bricht der Run mit einem Fehler ab, statt ein kamera-spezifisches Muster zu erraten.

---


## 4. Linearity

Linearitätsdiagnostik der Input-Frames. Die Prüfung sucht Hinweise auf nichtlineare Vorverarbeitung (Stretch, Curves, harte Kompression), ist aber kein direkter Kamerasensor-Linearitätstest.

### `linearity.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Aktiviert die Linearitätsdiagnostik in Phase 0 (SCAN_INPUT).

**Verhalten:** Auffällige Frames werden protokolliert und im aktuellen Runner im Warn-only-Modus behalten.

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

**Zweck:** Mindest-Linearitäts-Score (0 = auffällig, 1 = unauffällig). Liegt die Stichprobe darunter, wird eine Warnung ausgegeben.

---

### `linearity.strictness`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `strict`, `moderate`, `permissive` |
| **Default** | `"strict"` |

**Zweck:** Strictness-Level für die Diagnose-Schwellen.

**Verhalten:** Die harte Entscheidung ist konservativ und objektunabhängig: robuste Verteilungsform plus offensichtliches hartes Clipping/Kompression. Spektral-, Gradienten- und Varianzkennzahlen bleiben Diagnosewerte, weil lineare Frames je nach Objekt (leeres Feld, Sternhaufen, Nebel, Galaxienkern, CFA-Struktur) dort legitimerweise stark variieren.

| Level | Beschreibung |
|-------|-------------|
| **`strict`** | Engste Diagnose-Schwellen — empfohlen für lineare Roh-/kalibrierte Daten |
| **`moderate`** | Tolerantere Diagnose — für leicht vorverarbeitete oder schwierige Motive |
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

**Laufzeitverhalten:** Jede aktivierte Stufe benötigt mindestens eine
konfigurierte Quelle:
- Verzeichnis über `*_dir`
- oder explizites Master-Frame über `*_master`

---

### `calibration.bias_use_master` / `calibration.dark_use_master` / `calibration.flat_use_master`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` (alle drei) |

**Zweck:** Wenn `true`, wird ein fertiges Master-Frame aus dem angegebenen Pfad geladen statt aus Einzelframes gemittelt.

---

### `calibration.dark_already_bias_corrected`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Kennzeichnet ein Master-Dark als bereits bias-korrigiert. Wenn `false` und `use_bias: true`, zieht der Runner den Bias intern zuerst vom Dark ab, damit der Offset nicht doppelt vom Light subtrahiert wird.

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
| **Default** | `*.fit;*.fits;*.fts;*.fit.fz;*.fits.fz;*.fts.fz` |

**Zweck:** Glob-Pattern für Kalibrierungsdateien.

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

**Kaskade:**

- mit `registration.enable_star_pair_fallback=true`:
  Triangle Stars → Star Pairs → Trail Endpoints → AKAZE Features → Robust Phase+ECC → Hybrid Phase+ECC → Identity-Fallback
- mit `registration.enable_star_pair_fallback=false`:
  Triangle Stars → Trail Endpoints → AKAZE Features → Robust Phase+ECC → Hybrid Phase+ECC → Identity-Fallback

### `registration.enable_star_pair_fallback`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Aktiviert/deaktiviert den zusätzlichen Star-Pairs-Fallback zwischen Triangle Stars und Trail Endpoints.

Auf `false` setzen, um die Star-Pairs-Stufe für eine strengere Fallback-Policy zu deaktivieren.

**Hinweis (Strict v3.3.9):** Fuer das strikte Profil `registration.enable_star_pair_fallback: false` setzen.

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
| **Default** | `150` |

**Zweck:** Anzahl der hellsten Sterne, die für Star-basiertes Matching verwendet werden.

**Hinweis:** `tile_compile.yaml` und der C++ Default sind jetzt identisch. Höhere Werte erhöhen die Robustheit bei schwierigen Feldern, aber auch die Rechenzeit.

---

### `registration.star_min_inliers`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 2 |
| **Default** | `4` |

**Zweck:** Minimale Anzahl übereinstimmender Sterne (Inlier) für eine akzeptierte Registrierung.

**Empfehlung:** 5–10. Zu niedrig → falsche Matches werden akzeptiert. Zu hoch → gute Matches werden abgelehnt.

---

### `registration.star_inlier_tol_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `4.0` |

**Zweck:** Toleranz in Pixeln für die Zuordnung von Sternen als Inlier (nach Transformation).

**Hinweis:** `tile_compile.yaml` und der C++ Default sind jetzt identisch. Bezieht sich auf die **halbe Auflösung** (2× Downsample in der Registrierung).

---

### `registration.star_dist_bin_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `5.0` |

**Zweck:** Bin-Breite in Pixeln für das Paar-Abstands-Histogramm in der `star_similarity`-Methode.

**Hinweis:** `tile_compile.yaml` und der C++ Default sind jetzt identisch. Kleinere Werte sind genauer, größere Werte toleranter.

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
| **Default** | `0.25` |

**Zweck:** Absolute Untergrenze für den Korrelationswert (CC) in der Registrierungs-Outlier-Erkennung.
---

### `registration.reject_shift_px_min`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >=0 |
| **Default** | `100.0` |

**Zweck:** Feste Mindestgrenze (Pixel) für Shift-Outlier-Verwerfung.

**Hinweis:** Effektiver Shift-Grenzwert ist `max(reject_shift_px_min, reject_shift_median_multiplier * median(shift_px))`.

---

### `registration.reject_shift_median_multiplier`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `5.0` |

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

----

### `registration.max_blind_chain_depth`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 0 |
| **Maximum** | 100 |
| **Default** | `0` |

**Kompatibilitaet:** Historischer Parameter der deaktivierten Blind-Chain-Rescue. Er wird weiterhin geparst und serialisiert, damit bestehende Konfigurationen gueltig bleiben, beeinflusst die aktuelle Strategie `independent_global_consensus_v2` aber nicht. `0` wird weiterhin als `clamp(N/10, 12, 50)` aufgeloest, falls ein altes Artefakt oder Werkzeug den Legacy-Wert auswertet.

**Fallback-Verhalten:** Ungeloeste Frames werden nicht ueber Kettentiefe gerettet. Sie werden unabhaengig per Seeded-ECC direkt gegen den Master-Referenzframe validiert und fallen danach auf Astrometrie bzw. das globale Transformationsmodell zurueck.

----

### `registration.blind_chain_strong_anchor_cc`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0.01 |
| **Maximum** | 0.5 |
| **Default** | `0.08` |

**Kompatibilitaet:** Historische CC-Schwelle fuer starke Blind-Chain-Anker. Der Wert bleibt schema- und parserkompatibel, wird von `independent_global_consensus_v2` jedoch nicht fuer die Warp-Auswahl verwendet.

----

### `registration.blind_chain_drift_threshold_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0.5 |
| **Maximum** | 10.0 |
| **Default** | `2.0` |

**Kompatibilitaet:** Historische Driftgrenze in Pixeln pro Frame. Da die aktuelle Registrierung keine Nachbarframe-Ketten erzeugt, ist die Metrik nicht anwendbar und der Wert hat keinen Einfluss auf neue Runs.

----

### `registration.use_astrometry`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Aktiviert astrometrische Rescue für Frames, die alle anderen Algorithmen nicht registrieren können.

**Voraussetzungen:**
- ASTAP-Binary muss verfügbar sein (siehe `astrometry.astap_bin`)
- Lokaler Sternenkatalog muss vorhanden sein (siehe `astrometry.astap_data_dir`)

**Hinweis:** Bei sehr hellen Sternen (z.B. Capella) auf `false` setzen, da ASTAP Probleme mit überbelichteten Zentren hat.

----

### `registration.enable_local_background_subtraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Aktiviert lokale Hintergrundsubtraktion vor der Sternerkennung.

**Empfohlen bei:**
- Starkem Mondlicht mit Gradienten
- Starmer Hintergrundstruktur (Nebel, Galaxien)
- Unebenem Hintergrund durch Flat-Korrektur-Fehler

---

### `registration.affine_refinement_enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Standardmäßig aktive, konservative affine Feinkorrektur pro Frame nach der normalen globalen Registrierung. Sie erkennt Sterne im Referenz- und bereits gewarpten Proxy, bildet gegenseitige Nearest-Neighbor-Matches und schätzt mit RANSAC eine kleine affine Korrektur. Die Korrektur wird nur angewendet, wenn Inlierzahl und räumliche Abdeckung ausreichen, Skalierung/Shear/Rotation/Zentrumsverschiebung konservativ bleiben, Median und p90 der identischen Sternpaare ohne RMS-Regression sinken und NCC/Überlappung nicht schlechter werden. Andernfalls bleibt der ursprüngliche Warp unverändert.

**Einheiten und Anwendbarkeit:** Alle Residual-, Matchradius- und Zentrumsverschiebungs-Gates verwenden Proxy-Pixel (bei OSC normalerweise halbe Auflösung). Frames unterhalb des internen p90-Triggers, der Referenzframe und Frames ohne ausreichend verteilte Sterne sind nicht anwendbar und bleiben unverändert. Für einen reinen Kontrolllauf oder zur Diagnose kann die Stufe mit `false` deaktiviert werden.

---

### `registration.smooth_local_refinement_enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Aktiviert die standardmäßig gegatete, geglättete lokale Feinkorrektur pro Frame nach der globalen und, falls akzeptiert, nach der affinen Registrierung. Gegenseitige Nearest-Neighbor-Sternresiduen fitten ein regularisiertes inverses 4x4-Gaussian-Displacement-Feld. Ein deterministischer 25-%-Held-out-Sternsatz muss sich in Median, p90 und RMS verbessern; zusätzlich müssen der gesamte Matchsatz, räumliche Abdeckung, Maximalverschiebung, dicht geprüfte Jacobian-/Lokalskalengrenzen, NCC auf gemeinsamem Support und Überlappung bestehen.

**Einheiten, Grenzen und Wechselwirkungen:** Residuen und die interne Maximalverschiebung verwenden Proxy-Pixel; beim Prewarp wird das Feld auf volle Auflösung skaliert. Interne konservative Grenzen umfassen mindestens 32 Matches, mindestens 24 Trainings- und 8 Held-out-Sterne, 15 % Convex-Hull-Abdeckung, maximal 1,5 Proxy-Pixel Verschiebung, Jacobian-Determinante 0,94–1,06 und lokale Singulärwerte 0,96–1,04. Das Modell läuft am Bildrand auf null aus und wird direkt mit der inversen globalen/affinen Map komponiert, sodass keine zweite Full-Resolution-Resampling-Stufe entsteht. Anwendbar ist es derzeit nur für MONO und OSC mit `registration.debayer_first` und bekanntem Bayer-Muster; CFA-Mosaik- und nicht unterstützte Farbpfade behalten den unveränderten globalen/affinen Warp. Jedes gescheiterte Gate, jeder nicht anwendbare Frame und jede Ausnahme verwenden denselben unveränderten-Warp-Fallback. Per-Frame-Nachweise stehen in `global_registration.json`. Für einen reinen global/affinen Kontrolllauf oder zur Diagnose kann die Stufe mit `false` deaktiviert werden.

---

### `registration.star_shift_radius_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Default** | `200.0` |
| **Minimum** | `10` |
| **Maximum** | `2000` |

**Zweck:** Suchradius für den Shift-Konsistenz-Filter in `triangle_star_matching` (Pixel im Proxy-Bild, halbe Auflösung). Nach dem Dreiecks-Voting wird für jedes Stern-Paar geprüft, welche anderen Paare einen ähnlichen Shift implizieren (innerhalb dieses Radius). Das Paare-Cluster mit dem höchsten Support wird als Anker gewählt; alle inkonsistenten Paare werden verworfen. Der Radius muss den **maximalen erwarteten Inter-Frame-Shift** abdecken.

**Wann anpassen:**
- **Äquatoriale Montierung** mit gutem Tracking (kleine Shifts): `60`
- **Alt/Az-Montierung** (DWARF II, Seestar, mehrstündige Session): `200–400`
- **Sehr lange Alt/Az-Session** (>4h, großer Shift-Bereich): `400–600`

> ⚠️ Zu kleiner Radius (z.B. 60px bei Alt/Az) führt dazu, dass falsche Match-Cluster den echten Shift-Cluster als Anker verdrängen → alle Frames scheitern beim Triangle-Matching.

----

### `registration.prewarp_interpolation`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `bilinear`, `cubic`, `lanczos4` |
| **Default** | `"lanczos4"` |

**Zweck:** Interpolationsmethode fuer die Warp/Resample-Schritte in REGISTRATION (Prewarp der Frames in die gemeinsame Output-Geometrie). `lanczos4` ist schaerfste/empfohlene Methode; `bilinear` ist schneller, aber weicher.

---

### `registration.debayer_first`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Nur OSC. Bei `true` werden Frames vor dem Prewarp demosaiciert (R/G/B-Ebenen werden direkt rekonstruiert). Bei `false` erfolgt ein CFA-Prewarp; das Demosaicing des finalen CFA-Stacks geschieht nach der Rekonstruktion. Siehe §15.

---

### `registration.pre_debayer_method`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `linear`, `vng`, `edge_aware`, `mask` |
| **Default** | `"linear"` |

**Zweck:** Demosaicing-Methode fuer den `debayer_first`-Pfad.

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

## 9b. Chroma Denoise (struktur-schützende Farbrausch-Reduktion) {#chroma-denoise}

Optionale, **chroma-selektive** Denoise-Erweiterung für OSC-Daten. Idee: Luminanz/Struktur möglichst erhalten, Farbrauschen primär in Cb/Cr (oder äquivalenten Opponent-Kanälen) reduzieren.

> Hinweis: Wirkt nur im OSC-Pfad. `apply_stage` steuert, ob vor dem Tile-Overlap-Add (`pre_stack_tiles`), auf dem finalen linearen RGB-Stack (`post_stack_linear`), nach der photometrischen Farbkalibrierung (`post_pcc`) oder an beiden Stack-Stellen (`both`) gefiltert wird. `post_pcc` faengt Chroma-Rauschen ab, das durch die PCC-Matrix-Verstaerkung neu entsteht.

```yaml
chroma_denoise:
  enabled: true
  color_space: ycbcr_linear        # ycbcr_linear | opponent_linear
  apply_stage: post_stack_linear   # pre_stack_tiles | post_stack_linear | post_pcc | both
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

## 14. Reconstruction

Der `reconstruction:`-Block bündelt die Parameter der CFA-Forward-Drizzle-Rekonstruktion mit Multiband-Fusion. Neben `quality.pyramid` enthält er u. a. `drizzle.*` (Kernel, Pixfrac, Chunking, Speicherbudget), `clipping.*` (robustes Sample-Clipping), `coverage_gate.*` (Abdeckungs-Gates) und `multiband.*` (Bandfusion); die vollständige Feldliste steht in `tile_compile.schema.yaml`.

### `reconstruction.quality.pyramid.*` — Lokale Qualitätskarten

Steuerung der Laplacian-Pyramide für die lokalen Qualitätskarten (Source Quality Maps), mit denen Forward Drizzle jeden Bildbereich der Eingabeframes nach Schärfe und SNR gewichtet. Erzeugt in der Phase `SOURCE_QUALITY_MAPS`.

#### `reconstruction.quality.pyramid.scales`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | 1 – 8 |
| **Default** | `4` |

**Zweck:** Anzahl der Pyramidenstufen für die Multiskalenanalyse. Mehr Stufen erfassen mehr Raumfrequenzen, erhöhen aber die Rechenzeit.

---

#### `reconstruction.quality.pyramid.base_window_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `4` |

**Zweck:** Fenstergröße in Pixeln auf der untersten Pyramidenstufe für lokale Metrikberechnung.

---

#### `reconstruction.quality.pyramid.sharpness_weight`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Default** | `0.6` |

**Zweck:** Gewicht der Schärfemetrik im kombinierten Qualitätsindex `Q = sharpness_weight * Q_sharp + snr_weight * Q_snr`. Zusammen mit `snr_weight` bestimmt dies die relative Bedeutung von Schärfe vs. Signal-Rausch-Verhältnis; die Summe beider Gewichte muss größer als 0 sein.

---

#### `reconstruction.quality.pyramid.snr_weight`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | 0 |
| **Default** | `0.4` |

**Zweck:** Gewicht der SNR-Metrik im kombinierten Qualitätsindex. Erhöhen bei stark verrauschten Daten mit hohem Frame-Qualitätsgefälle.

---

#### `reconstruction.quality.pyramid.score_scale`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `1.8` |

**Zweck:** Skaliert den kombinierten lokalen Qualitäts-Score vor dem Sigmoid. Höhere Werte erhöhen die Pixel-Selektivität der Qualitätskarten, damit scharfe Frames lokal stärker bevorzugt werden; die Karte bleibt weiterhin auf `[0,1]` begrenzt.

---

#### `reconstruction.quality.pyramid.artifact_sigma`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `3.0` |

**Zweck:** MAD-Multiplikator für die Artefakt-Erkennung. Pixel, deren lokale Abweichung `artifact_sigma * MAD` überschreitet, werden als Artefakt markiert und erhalten reduziertes Qualitätsgewicht.

- **Höher (z.B. 7–10):** Toleranter gegenüber Ausreißern — mehr Pixel erhalten normales Gewicht
- **Niedriger (z.B. 3–4):** Aggressivere Artefakt-Unterdrückung

---

#### `reconstruction.quality.pyramid.max_artifact_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | >0 – 1 |
| **Default** | `0.25` |

**Zweck:** Maximaler tolerierter Artefakt-Anteil pro Auswertungsfenster. Fenster mit mehr Artefakten als `max_artifact_fraction` werden vollständig verworfen (kein Qualitäts-Beitrag).

- **Erhöhen (z.B. 0.30–0.40):** Bei bekannten, tolerierbaren Artefakten (z.B. Satellitenspuren)
- **Verringern:** Strengere Qualitätsgates

---

#### Weitere `reconstruction.*`-Schluessel

| Schluessel | Typ | Default | Zweck |
|------------|-----|---------|-------|
| `reconstruction.common_overlap_required_fraction` | number | `1.0` | Mindestanteil nutzbarer Frames, in denen ein Pixel gueltig sein muss, um zu `COMMON_OVERLAP` zu gehoeren (migriert von `stacking.common_overlap_required_fraction`). |
| `reconstruction.delete_source_cache_after_run` | boolean | `false` | Loescht den Quell-/Normalisierungs-Cache nach erfolgreichem Lauf (Plattenplatz). |
| `reconstruction.keep_profile_cache_after_run` | boolean | `false` | Behaelt den `drizzle_profile`-Cache fuer Reuse-/Final-Render-Pfade. |
| `reconstruction.diagnostics.level` | enum | `summary` | Diagnoseumfang: `summary`, `full`. |
| `reconstruction.diagnostics.preview_forward_drizzle_uniform` | boolean | `false` | Best-effort Uniform-Kontroll-Diagnose nach SAMPLING_GEOMETRY (`forward_drizzle_uniform_diagnostic.json`). |
| `reconstruction.diagnostics.persist_forward_drizzle_uniform_store` | boolean | `false` | Persistiert ungeclippte Uniform-Ebenen transaktional unter `artifacts/forward_drizzle_uniform_store/`. |
| `reconstruction.drizzle.internal_scale` | integer | `2` | Interner Arbeits-Scale des Drizzle-Grids (`1` oder `2`). |
| `reconstruction.drizzle.output_scale` | integer | `1` | Ausgaberaster relativ zum internen Grid (`1` oder `2`, `<= internal_scale`). |
| `reconstruction.drizzle.kernel` | enum | `square` | Drizzle-Kernel; aktuell nur `square`. |
| `reconstruction.drizzle.pixfrac` | number | `0.8` | Tropfen-Schrumpffaktor der Beitraege `(0, 1]`. |
| `reconstruction.drizzle.robust_passes` | integer | `2` | Robuste Reprojektions-Durchlaeufe `[1, 6]` (migriert von `clip_iterations`). |
| `reconstruction.drizzle.min_clip_contributors` | integer | `5` | Mindestbeitraege, ab denen Sigma/MAD-Clipping aktiv wird. |
| `reconstruction.drizzle.chunk_rows` | integer | `0` | `0` = budgetierte Streifen (<=256 Zeilen); `>0` wird gegen das Budget geprueft. |
| `reconstruction.drizzle.chunk_halo_rows` | integer | `-1` | Kompatibilitaetsfeld; exakte Footprint-Enumeration braucht keinen Output-Halo. |
| `reconstruction.drizzle.memory_budget_mb` | integer | `0` | MeB; `0` erbt `runtime_limits.memory_budget`. |
| `reconstruction.clipping.clip_sigma_low` / `clip_sigma_high` | number | `3.0` / `3.0` | MAD-Sigma-Schwellen der robusten Beitragskontrolle. |
| `reconstruction.clipping.min_fraction` | number | `0.4` | Mindestanteil belegbarer Samples pro Pixel `(0, 1]`. |
| `reconstruction.clipping.min_n_eff` | number | `3.0` | Minimale effektive Beitragszahl fuer einen validen Output-Pixel (`>= 1`). |
| `reconstruction.clipping.guard_fallback` | boolean | `false` | `false` = striktes Veto (Pixel bleibt unbelegt); `true` = Survivor-/Unclipped-Fallback statt schwarzer Pixel. |
| `reconstruction.coverage_gate.min_frames` | integer | `2` | Mindestzahl nutzbarer Frames. |
| `reconstruction.coverage_gate.min_supported_fraction` | number | `0.995` | Mindestanteil gestuetzter Output-Pixel `(0, 1]`. |
| `reconstruction.coverage_gate.min_channel_n_eff_floor` | number | `3.0` | N_eff-Minimum fuer einen als belegt geltenden Kanal. |
| `reconstruction.coverage_gate.min_channel_n_eff_fraction` | number | `0.15` | Mindestanteil der Kanaele ueber der N_eff-Schwelle. |
| `reconstruction.coverage_gate.min_analysis_pixels` | integer | `1024` | Mindestzahl analysierbarer Pixel. |
| `reconstruction.coverage_gate.max_internal_hole_area_px` | integer | `0` | Max. Flaeche interner Loecher im Footprint. |
| `reconstruction.multiband.enabled` | boolean | `true` | Multiband-Fusion aktivieren. |
| `reconstruction.multiband.levels` | integer | `3` | Bandebenen `[1, 4]`; `>=2` erfordert `quality.pyramid.scales >= 2`. |
| `reconstruction.multiband.alpha_cap` | number | `1.0` | Obergrenze des Misch-Alphas `[0, 1]`. |
| `reconstruction.multiband.fine_quality_exponent` / `medium_quality_exponent` | number | `4.0` / `2.0` | Qualitaets-Exponenten der Fein-/Mittelband-Gewichte. |
| `reconstruction.multiband.min_quality_separation` / `full_quality_separation` | number | `0.05` / `0.20` | Qualitaetsseparations-Schwellen der Bandfusion. |
| `reconstruction.multiband.min_effective_samples` / `full_effective_samples` | number | `8.0` / `24.0` | N_eff-Schwellen fuer minimale/volle Bandnutzung. |
| `reconstruction.multiband_validation.fwhm_ratio_max` | number | `0.95` | `> 0`; Median-FWHM des Multiband-Stacks muss `<= x *` raw sein. |
| `reconstruction.multiband_validation.p90_fwhm_ratio_max` | number | `1.0` | `> 0`; p90-FWHM des Multiband-Stacks vs. raw. |
| `reconstruction.multiband_validation.tail_ratio_max` | number | `1.1` | `> 0`; Sternauslaeufer-Metrik des Multiband-Stacks vs. raw. |
| `reconstruction.multiband_validation.elongation_ratio_max` | number | `1.08` | `> 0`; Stern-Elongation des Multiband-Stacks vs. raw. |
| `reconstruction.multiband_validation.background_rms_ratio_max` | number | `1.05` | `> 0`; Hintergrund-RMS vs. `drizzle_uniform` (Raw-Veto und Multiband-Gate). Erhoehen, wenn der abgetastete "Hintergrund" echte schwache Struktur enthaelt (z. B. eine feldfuellende Galaxie) und hoeheres RMS Signal statt Rauschen ist. |
| `reconstruction.multiband_validation.seam_ratio_max` | number | `1.05` | `> 0`; Nahtkontrast vs. `drizzle_uniform` an der Support-Grenze. |
| `reconstruction.multiband_validation.min_stars_fwhm` | integer | `20` | `>= 0`; gematchte Sterne fuer die FWHM-Metrik. |
| `reconstruction.multiband_validation.min_stars_p90_tail_elongation` | integer | `30` | `>= 0`; gematchte Sterne fuer p90-/Tail-/Elongations-Metriken. |
| `reconstruction.multiband_validation.max_fwhm_ci_relative_width` | number | `0.1` | `> 0`; Veto, wenn das Bootstrap-95%-KI des FWHM-Verhaeltnisses breiter als diese relative Breite ist. |

Fail-closed Gates: `FORWARD_DRIZZLE` und `MULTIBAND` brechen kontrolliert ab,
wenn Coverage-/Konfidenz-Gates oder Validierungsschwellen nicht erreicht
werden. Verwirft ein Multiband-Validation-Gate einen Kandidaten, behaelt der
Run stattdessen die `drizzle_uniform`-Kontrolle.

---

## 15. Debayer

Es gibt keinen eigenständigen `debayer`-Konfigurationsschlüssel.

- `OSC` mit `registration.debayer_first: true` (Default): Frames werden vor dem Prewarp demosaiciert; die R/G/B-Ebenen werden direkt rekonstruiert.
- `OSC` mit `debayer_first: false`: CFA-Prewarp; der finale CFA-Stack wird nach der Rekonstruktion demosaiciert.
- `MONO`: kein Demosaicing.

---

## 16. Astrometry

### `astrometry.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Aktiviert Plate Solving (WCS).

**Fallback-Verhalten:** Zuerst wird ASTAP verwendet. Liefert dessen Quad-Matching
keine WCS, versucht der Runner lokal und ohne Netzwerkzugriff einen Near-Solve
gegen den bereits fuer PCC installierten Siril/Gaia-DR3-Katalog. Dafuer muessen
im RGB-Stack `RA`, `DEC`, `FOCALLEN` sowie `XPIXSZ` oder `YPIXSZ` vorhanden
sein. Der Fallback schaetzt nur eine lineare TAN/CD-WCS; SIP-Verzerrungen
werden nicht erzeugt. Ist der lokale Gaia-Katalog nicht installiert oder die
Sternzuordnung nicht robust genug, bleibt die bisherige `solve_failed`-Behandlung
erhalten.

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

## 17. BGE (Background Gradient Extraction)

**NEU in v3.3** - Optionale Hintergrund-Gradienten-Extraktion vor PCC (Methodologie v3.3 §6.3)

BGE entfernt großräumige Hintergrundgradienten (Lichtverschmutzung, Mondlicht, Airglow) **vor** der photometrischen Farbkalibrierung, um Farbverzerrungen durch spektral ungleichmäßige Gradienten zu vermeiden.

**Implementationshinweis:** BGE nutzt Tile-Qualitätsdaten, die aus dem rekonstruierten Bild abgeleitet werden, für die Sample-Selektion/-Gewichtung:
- `type` + `star_count`: Sternreiche STAR-Tiles werden konservativ ausgeschlossen bzw. abgewertet.
- `fwhm`: skaliert die effektive Sternmasken-Dilatation pro Tile.
- `quality_score`: geht als zusätzlicher Gewichtungsfaktor in die Tile-Sample-Relevanz ein.

### `bge.method`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Werte** | `none`, `classic`, `autobge`, `auto` |
| **Default** | `none` |

**Zweck:** Wählt die BGE-Engine. `none` deaktiviert BGE vollständig, `classic` nutzt die bestehende grid-/tile-basierte BGE-Implementierung, `autobge` wählt die zweistufige Poly+RBF-AutoBGE-Implementierung, `auto` misst den Hintergrundgradienten programmatisch und startet AutoBGE nur dann, wenn die Gradienten-Amplitude den Schwellwert `bge.auto_detect.gradient_threshold` überschreitet — mit automatisch abgeleitetem Exclusion-Mask für Extended Sources (Galaxien, Nebel). Dies ist der einzige Ein-/Ausschalter für BGE.

> **Migrationshinweis:** `bge.enabled` (ein legacy boolean-Spiegel dieses Felds) wurde entfernt. Es konnte im Widerspruch zu `bge.method` stehen — z. B. lief BGE bei `enabled: false` neben einem veralteten `method: classic` trotzdem, weil `method` immer maßgeblich war, sobald vorhanden. Eine Config, die weiterhin `bge.enabled` setzt, schlägt jetzt beim Laden mit einem Validierungsfehler fehl, der auf `bge.method` verweist. Ersetze `enabled: true` durch `method: classic` (oder `autobge`) und `enabled: false` durch `method: none`.

### `bge.auto_detect.gradient_threshold`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | `0` |
| **Default** | `0.05` |

**Zweck:** Minimale relative Gradienten-Amplitude, die bei `bge.method: auto` AutoBGE auslöst. Die Amplitude ist `(max − min des linearen Ebenen-Fits) / sky_median` über dem G-Kanal. Werte darunter: BGE wird übersprungen. Empfohlen: `0.03–0.10`; kleiner = empfindlicher (mehr Felder bekommen BGE), größer = konservativer.

### `bge.auto_detect.extended_source_sigma`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | `0` |
| **Default** | `3.0` |

**Zweck:** Detektionsschwelle für Extended Sources (Galaxien-Discs, große Nebel) im Block-Median-Gitter. Blöcke, deren Median `sky_median + N × sky_sigma` überschreitet, werden vom AutoBGE-Sample-Set ausgeschlossen, damit der Poly/RBF-Fit nur echte Himmelspixel verwendet. Empfohlen: `2.5–4.0`.

### `bge.auto_detect.extended_source_dilate_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | `0` |
| **Default** | `50` |

**Zweck:** Morphologische Dilation (Pixel) des erkannten Extended-Source-Exclusion-Masks. Größere Werte geben mehr Sicherheitsmarge um Galaxien-Halos und Nebelränder. Empfohlen: `30–80`.

### `bge.autobge.num_sample_points`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | `0` |
| **Default** | `0` |

**Zweck:** Anzahl der AutoBGE-Sample-Punkte. `0` = automatische Berechnung anhand der Bildgröße (ca. 1 Punkt pro 800 downsampled Pixel, begrenzt auf 200–3000). Höhere Werte = dichtere Abtastung, langsamere Verarbeitung. Empfohlen: `0` (auto) für die meisten Fälle, `800–1500` für große Bilder mit komplexen Gradienten.

### `bge.autobge.poly_degree`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Bereich** | `1 – 6` |
| **Default** | `2` |

**Zweck:** Grad des ersten Polynom-Fits. `2` = quadratisch (erfasst breite Gradienten), `3` = kubisch (erfasst komplexe Gradienten). Höhere Grade riskieren Overfitting an Bildstruktur. Empfohlen: `2` für die meisten Datensätze, `3` nur für sehr starke asymmetrische Gradienten.

### `bge.autobge.rbf_smooth`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Minimum** | `0` |
| **Default** | `0.1` |

**Zweck:** RBF-Glättungsfaktor für den zweiten Fit-Schritt (Residual). Höhere Werte = glatteres Hintergrundmodell, können lokale Gradienten unterfitting. Bereich `0.01–1.0`. Empfohlen: `0.1` für typische Bilder, `0.5` für glatte Gradienten.

### `bge.autobge.downsample_scale`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | `1` |
| **Default** | `4` |

**Zweck:** Downscaling-Faktor für das Arbeitsbild. `4` = 4x Downsample (16x weniger Pixel). Höhere Werte = schnellere Verarbeitung, geringere räumliche Auflösung des Hintergrundmodells. Bereich `1–8`. Empfohlen: `4` für Vollauflösung, `2` für kleine Bilder.

### `bge.autobge.patch_size`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | `3` |
| **Default** | `15` |

**Zweck:** Ungerade Patch-Größe für lokale Hintergrund-Schätzung. Jeder Sample-Punkt misst den Hintergrund in einem Patch dieser Größe. Größere Patches sind robuster, mitteln aber mehr Struktur. Bereich `3–31`. Empfohlen: `15`.

### `bge.autobge.patch_estimator`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `median`, `sigma_clipped_median` |
| **Default** | `"median"` |

**Zweck:** Lokaler Hintergrund-Estimator für jeden Sample-Patch. `median` = schnell und robust für die meisten Bilder. `sigma_clipped_median` = iterativ, verwirft Ausreißer, besser bei vielen Sternen/Cosmic Rays.

### `bge.autobge.stretch_mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `none`, `linear`, `mtf` |
| **Default** | `"linear"` |

**Zweck:** Working-Space-Transform für AutoBGE-Sampling/Fitting. `linear` ist der konservative Default, weil die spätere HyperMetric-Stretch-Phase unabhängig am Ende arbeitet und BGE additiv im linearen Bild bleiben soll. `mtf` ist nur für AutoBGE-Parität/Experimente gedacht.

### `bge.autobge.stretch_target_median`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Bereich** | `(0, 1]` |
| **Default** | `0.25` |

**Zweck:** Zielmedian für den Working-Space-Stretch (nur bei `stretch_mode=mtf`). Kontrolliert die Helligkeit des gestreckten Bildes für Sampling. Niedrigere Werte sampling dunklere Regionen. Bereich `0.1–0.5`.

### `bge.autobge.border_margin`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | `0` |
| **Default** | `10` |

**Zweck:** Pixel-Rand, der vom Sampling ausgeschlossen wird. Randpixel enthalten oft Stacking-Artefakte oder Vignettierung. Erhöhen für Weitwinkel-Bilder mit starken Randeffekten. Bereich `0–100`. Empfohlen: `10–30`.

### `bge.autobge.bright_exclusion_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Bereich** | `(0, 1)` |
| **Default** | `0.5` |

**Zweck:** Anteil der hellsten Pixel, die vom Sampling ausgeschlossen werden. `0.5` = top 50% ausgeschlossen (konservativ bei nebelreichen Feldern). Niedrigere Werte (`0.2–0.3`) riskieren Strukturkontamination. Bereich `0.1–0.8`.

### `bge.autobge.gradient_descent_max_iters`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | `1` |
| **Default** | `100` |

**Zweck:** Max. Iterationen für Sample-Point-Platzierung per Gradient Descent. Jede Iteration verschiebt Sample-Punkte zu dunkleren lokalen Regionen. Höhere Werte finden dunklere Hintergrund-Spots, erhöhen Laufzeit. Bereich `20–500`. Empfohlen: `100`.

### `bge.autobge.random_seed`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Default** | `42` |

**Zweck:** Zufallsseed für deterministische Sample-Point-Generierung. Gleicher Seed + gleiches Bild = identische Ergebnisse. Ändern für alternative Sample-Platzierungen zum Vergleich.

### `bge.autobge.normalize_between_stages`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Wenn `true`, wird das Residual-Bild zwischen Polynom- und RBF-Stufe normalisiert. Verhindert, dass RBF bereits erfasste Gradienten neu fittet. Empfohlen: `true`.

### `bge.autobge.apply_guards`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Wenn `true`, nutzt AutoBGE die gemeinsamen BGE-Apply-Guards (Flatness/Slope-Prüfung), bevor RGB verändert wird. Verhindert Verschlechterung durch schlechte Fits. Empfohlen: `true`.

### `bge.autobge.mono_mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `rgb_duplicate`, `disabled` |
| **Default** | `"rgb_duplicate"` |

**Zweck:** Behandlung von Mono-Bildern. `rgb_duplicate` kopiert den Kanal nach R/G/B vor BGE, erlaubt per-Channel-Korrektur. `disabled` verarbeitet nur den einzelnen Kanal. `rgb_duplicate` für OSC-Bilder, die als Mono debayered wurden.

### `bge.tile_weight_lambda_structure`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Bereich** | `> 0` |
| **Default** | `1.0` |

**Zweck:** Lambda in Tile-Reliabilitätsgewicht `w_t = exp(-lambda * structure_score_t) * (1 - masked_fraction_t)`. Höhere Werte gewichten strukturreiche Tiles aggressiver ab. Bereich `0.5–3.0`. Empfohlen: `1.0` für moderate Felder, `2.0+` für dichte Nebulosität.

### `bge.sample_quantile`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Bereich** | `(0.0, 0.5]` |
| **Default** | `0.20` |

**Zweck:** Quantil für Tile-Hintergrund-Schätzung (v3.3 §6.3.2b). Niedrigere Werte (`0.10–0.15`) sind konservativer, resistent gegen Nebelkontamination. `0.50` = Median, geeignet für stark maskierte Felder. Bereich `(0, 0.5]`.

### `bge.structure_thresh_percentile`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Bereich** | `[0.0, 1.0]` |
| **Default** | `0.90` |

**Zweck:** Perzentil-Schwelle für High-Structure-Tiles (v3.3 §6.3.2a). `0.80` = moderat (schließt top 20% aus), `0.90` = streng (schließt top 10% aus). Niedrigere Werte erhalten mehr Samples, riskieren aber Strukturkontamination. Bereich `0.5–0.95`.

### `bge.min_tiles_per_cell`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Default** | `3` |

**Zweck:** Mindestanzahl Tile-Samples pro Grid-Cell für valide Hintergrund-Schätzung (v3.3 §6.3.3d). Zellen mit weniger Tiles triggern `insufficient_cell_strategy`. Bereich `1–10`. Empfohlen: `3`.

### `bge.min_valid_sample_fraction_for_apply`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Bereich** | `(0.0, 1.0]` |
| **Default** | `0.30` |

**Zweck:** Kanalweise Sicherheitsgrenze für BGE-Apply. Wenn `valid_tile_samples / total_tile_samples` unter diesem Wert liegt, wird BGE für den Kanal übersprungen.

### `bge.min_valid_samples_for_apply`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | `1` |
| **Default** | `96` |

**Zweck:** Absolute kanalweise Sicherheitsgrenze für BGE-Apply. Wenn weniger robuste valide Tile-Samples als dieser Wert vorhanden sind, wird BGE für den Kanal übersprungen.

### `bge.mask.star_dilate_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Default** | `4` |

**Zweck:** Dilatation der Stern-Maske in Pixeln (v3.3 §6.3.2a). Erweitert die Ausschlusszone um Sterne, um Halo-Kontamination der Hintergrund-Samples zu verhindern. Bereich `0–20`. Empfohlen: `4–6` (normal), `8–12` (Weitwinkel/helle Sterne).

### `bge.mask.sat_dilate_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Default** | `4` |

**Zweck:** Dilatation der Sättigungs-Maske in Pixeln (v3.3 §6.3.2a). Erweitert die Ausschlusszone um gesättigte Pixel/cores. Bereich `0–20`. Empfohlen: `4–6`, erhöhen bei Sensoren mit starkem Blooming.

### `bge.grid.N_g`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Default** | `32` |

**Zweck:** Ziel-Grid-Auflösung: `G = min(W,H) / N_g` (v3.3 §6.3.8). Höhere Werte erzeugen feinere Grids für bessere Gradientenerfassung, benötigen aber mehr Samples pro Zelle. Bereich `16–64`. Empfohlen: `32–36` für typische DSO-Aufnahmen, `48+` für Weitwinkel.

### `bge.grid.G_min_px`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Default** | `64` |

**Zweck:** Minimaler Grid-Abstand in Pixeln (v3.3 §6.3.8). Verhindert zu kleine Zellen auf großen Bildern. Bereich `32–128`. Empfohlen: `56–64`.

### `bge.grid.G_max_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Default** | `0.25` |

**Zweck:** Maximaler Grid-Abstand als Bruchteil von `min(W,H)` (v3.3 §6.3.8). Verhindert zu große Zellen auf kleinen Bildern. Bereich `0.1–0.5`. Empfohlen: `0.25`.

### `bge.grid.insufficient_cell_strategy`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `discard`, `nearest`, `radius_expand` |
| **Default** | `"discard"` |

**Zweck:** Strategie für Grid-Cells mit zu wenigen Samples (v3.3 §6.3.3d).

- **`discard`**: Cell wird vom Fit ausgeschlossen (konservativ)
- **`nearest`**: Nearest-Neighbor-Fill (experimentell)
- **`radius_expand`**: Radius-Expansion, vergrößert Suchradius (empfohlen für Rand-Cells)

### `bge.fit.method`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `rbf`, `poly`, `spline`, `bicubic`, `modeled_mask_mesh` |
| **Default** | `"rbf"` |

**Zweck:** Surface-Fitting-Methode (v3.3 §6.3.7).

- **`rbf`**: Radial Basis Functions (empfohlen, flexibel)
- **`poly`**: Robustes Polynom (Order 2-3, breite Gradienten, schneller)
- **`spline`**: Thin-plate Spline
- **`bicubic`**: Bicubic Spline
- **`modeled_mask_mesh`**: Segmentierungs- und maskengestützter Mesh-Sky-Fit mit heller Quellenmodellierung (empfohlen bei großflächigem Nebel/Vordergrund wie M31/M42)

### `bge.fit.robust_loss`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `huber`, `tukey` |
| **Default** | `"huber"` |

**Zweck:** Robust-Loss-Funktion für IRLS (v3.3 §6.3.7). `huber` = quadratisch für kleine Residuen, linear für große (moderate Outlier-Rejektion). `tukey` = verwirft große Residuen komplett (aggressiv). Empfohlen: `huber`.

### `bge.fit.huber_delta`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Default** | `1.5` |

**Zweck:** Huber-Loss-Parameter δ. Residuen < δ quadratisch, > δ linear. Kleinere δ = mehr Outlier-Rejektion. Bereich `0.5–3.0`. Empfohlen: `1.5`.

### `bge.fit.irls_max_iterations`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Default** | `10` |

**Zweck:** Maximale IRLS-Iterationen. Höhere Werte = bessere Konvergenz, langsamere Laufzeit. Bereich `5–20`. Empfohlen: `10`.

### `bge.fit.irls_tolerance`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Default** | `1e-4` |

**Zweck:** IRLS-Konvergenz-Toleranz. Stop bei Parameteränderung < Toleranz. Bereich `1e-6–1e-3`. Empfohlen: `1e-4`.

### `bge.fit.polynomial_order`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Werte** | `2`, `3` |
| **Default** | `2` |

**Zweck:** Polynom-Ordnung (nur wenn `method=poly`). `2` = quadratisch (breite Gradienten, sicher). `3` = kubisch (komplexe asymmetrische Gradienten, höheres Overfitting-Risiko). Empfohlen: `2`.

### `bge.fit.rbf_phi`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `multiquadric`, `thinplate`, `gaussian` |
| **Default** | `"multiquadric"` |

**Zweck:** RBF-Kernel-Typ (nur wenn `method=rbf`, v3.3 §6.3.7).

- **`multiquadric`**: `φ(d;μ) = √(d² + μ²)` (empfohlen)
- **`thinplate`**: `φ(d) = d² log(d)` (scale-invariant)
- **`gaussian`**: `φ(d;μ) = exp(-d²/(2μ²))` (glatt)

### `bge.fit.rbf_mu_factor`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Default** | `1.0` |

**Zweck:** RBF-Shape-Parameter: `μ = rbf_mu_factor * G` (v3.3 §6.3.7). Kontrolliert die Breite der Basisfunktionen. Höhere Werte = glattere Oberflächen. Bereich `0.5–3.0`. Empfohlen: `1.0–1.5`.

### `bge.fit.rbf_lambda`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Default** | `1e-6` |

**Zweck:** RBF-Regularisierung λ (verhindert Overfitting, v3.3 §6.3.7). Höhere Werte = glatter, können unterfitting. Bereich `1e-6–0.1`. Empfohlen: `0.01–0.1`.

### `bge.fit.rbf_epsilon`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Default** | `1e-10` |

**Zweck:** Numerische Stabilisierung für Thin-plate RBF bei d=0 (v3.3 §6.3.7). Verhindert Division durch Null. Bereich `1e-10–1.0`. Empfohlen: `1e-10` (thinplate), `1.0` (multiquadric).

### `bge.autotune.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Aktiviert deterministisches konservatives Auto-Tuning von BGE (v3.3.6 §6.3.7). Wenn `true`, führt BGE eine Cross-Validation-Suche über Estimator, Quantile, Struktur-Schwellen und Fit-Methoden durch. Erhöht Laufzeit, verbessert Qualität bei schwierigen Feldern.

### `bge.autotune.strategy`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `conservative`, `extended` |
| **Default** | `"conservative"` |

**Zweck:** Umfang des Kandidatenraums fuer Auto-Tuning. `conservative` = kleine Menge sicherer Parameterkombinationen (schneller). `extended` = weiter Suchraum (langsamer, besser bei Nebulosität/komplexen Gradienten).

### `bge.autotune.max_evals`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 1 |
| **Default** | `24` |

**Zweck:** Harte Obergrenze getesteter Parameter-Kandidaten pro Kanal. Jede Eval fittet ein volles Hintergrundmodell. Bereich `8–64`. Empfohlen: `24` (conservative), `32` (extended).

### `bge.autotune.holdout_fraction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Bereich** | `[0.05, 0.50]` |
| **Default** | `0.25` |

**Zweck:** Deterministischer Validierungsanteil fuer `E_cv` im Ziel `J`. Anteil der Sample-Punkte für Cross-Validation (nicht für Fit verwendet). Bereich `[0.05, 0.50]`. Empfohlen: `0.20–0.25`.

### `bge.autotune.alpha_flatness`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Minimum** | 0 |
| **Default** | `0.25` |

**Zweck:** Gewichtung des Flatness-Terms `E_flat` in `J`. Höhere Werte priorisieren glattere Hintergründe. Bereich `0–1`. Empfohlen: `0.25–0.40` (schwache Gradienten), `0.40–0.80` (starke Gradienten).

### `bge.autotune.beta_roughness`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | float |
| **Minimum** | 0 |
| **Default** | `0.10` |

**Zweck:** Gewichtung des Roughness-Terms `E_rough` in `J`. Höhere Werte bestrafen bucklige Modelle stärker. Bereich `0–0.5`. Empfohlen: `0.08–0.15`.

---

## 18. PCC

**Implementationshinweis (v3.3.6):** Wenn Tile-Metriken und Tile-Grid verfügbar und konsistent sind, nutzt PCC diese automatisch zur robusten Sterngewichtung:
- `quality_score`: exponentielle Gewichtung pro Stern (Tile-basiert).
- `gradient_energy/noise`: Struktur-Penalty und Reject für stark strukturierte Tiles.
- `star_count`: leichte Abwertung sehr sternreicher Tiles.

### `pcc.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Aktiviert photometrische Farbkalibrierung (PCC). Katalogsternfarben werden abgeglichen, um die RGB-Farbbalance des gestackten Bildes zu kalibrieren.

---

### `pcc.source`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `auto`, `siril`, `vizier_gaia`, `vizier_apass` |
| **Default** | `"auto"` |

**Zweck:** Katalog-/Provider-Auswahl für PCC. `auto` wählt automatisch den besten verfügbaren Katalog.

---

### `pcc.mag_limit`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 1 – 22 |
| **Default** | `14.0` |

**Zweck:** Grenzmagnitude für PCC-Katalogstern-Matching. Höhere Werte schließen schwächere Sterne ein. ACHTUNG: Bei kleinen Sensoren oder dichten Sternfeldern kann mag_limit > 15 Sterne unterhalb der Nachweisgrenze einschließen. Bereich 1–22. Empfohlen: 14.

---

### `pcc.mag_bright_limit`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0 – 15 |
| **Default** | `6.0` |

**Zweck:** Helligkeitsgrenze für PCC-Katalogsterne. Sterne heller als dieser Wert werden ausgeschlossen (gesättigte Sterne geben unzuverlässige Photometrie). Bereich 0–15. Empfohlen: 6.

---

### `pcc.aperture_radius_px`, `pcc.annulus_inner_px`, `pcc.annulus_outer_px`

| Key | Typ | Default | Constraint |
|-----|-----|---------|------------|
| `pcc.aperture_radius_px` | number | `8.0` | >0 |
| `pcc.annulus_inner_px` | number | `12.0` | >0 |
| `pcc.annulus_outer_px` | number | `18.0` | >0 |

**Zweck:** Apertur-/Annulus-Geometrie für Sternphotometrie. `aperture_radius_px` ist der photometrische Apertur-Radius, `annulus_inner_px` und `annulus_outer_px` definieren den Sky-Annulus für lokale Hintergrundschätzung. Verwendet bei `radii_mode=fixed`.

---

### `pcc.min_stars`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Minimum** | 3 |
| **Default** | `10` |

**Zweck:** Mindestanzahl gematchter Katalogsterne für PCC. Darunter wird PCC übersprungen. Bereich: >= 3. Empfohlen: 10.

---

### `pcc.sigma_clip`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `2.5` |

**Zweck:** Sigma-Clipping-Schwelle für PCC-Ausreißer-Ablehnung. Sterne mit Residuen > sigma_clip × std werden abgelehnt. Bereich: > 0. Empfohlen: 2.5.

---

### `pcc.background_model`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `median`, `plane` |
| **Default** | `"plane"` |

**Zweck:** Lokales Annulus-Hintergrundmodell fuer Sternphotometrie (`plane` empfohlen bei Gradienten).

### `pcc.max_condition_number`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | `>= 1.0` |
| **Default** | `3.0` |

**Zweck:** Obergrenze der Matrix-Konditionszahl; verhindert instabile PCC-Loesungen.

### `pcc.max_residual_rms`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | `> 0` |
| **Default** | `0.35` |

**Zweck:** Obergrenze fuer robusten Fit-Residuen-RMS; verwirft verrauschte/instabile PCC-Fits.

---

### `pcc.radii_mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Werte** | `fixed`, `auto_fwhm` |
| **Default** | `"auto_fwhm"` |

**Zweck:** Radiusmodus fuer Apertur/Annulus (`auto_fwhm` = adaptive Radien aus Seeing-FWHM).

---

### `pcc.aperture_fwhm_mult`, `pcc.annulus_inner_fwhm_mult`, `pcc.annulus_outer_fwhm_mult`, `pcc.min_aperture_px`

| Key | Typ | Default | Constraint |
|-----|-----|---------|------------|
| `pcc.aperture_fwhm_mult` | number | `1.8` | >0 |
| `pcc.annulus_inner_fwhm_mult` | number | `3.0` | >0 |
| `pcc.annulus_outer_fwhm_mult` | number | `5.0` | >0 |
| `pcc.min_aperture_px` | number | `4.0` | >0 |

**Zweck:** Konservative Parameter fuer FWHM-adaptive PCC-Radien (v3.3.6 §6.4.2).

---

### `pcc.siril_catalog_dir`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `""` |

**Zweck:** Lokaler Siril-Katalogpfad; leer = Standardpfad.

### `pcc.apply_attenuation`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Aktiviert adaptive Daempfung der PCC-Matrixanwendung in Schatten/Highlights.

### `pcc.background_neutralization_mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `auto` |
| **Erlaubte Werte** | `always`, `auto`, `off` |

**Zweck:** Steuert die Hintergrundneutralisierung nach dem PCC-Apply unabhaengig von `chroma_strength` und vom Matrix-Typ. `always` erzwingt neutrale Hintergrund-Offsets, `off` erhaelt die kanalweisen Hintergrundwerte, und `auto` neutralisiert einen raeumlich kohaerenten globalen Farbstich vollstaendig, schwaecht die Korrektur bei lokal variierender Nebel-/Feldstruktur aber ab oder ueberspringt sie. Diagonale PCC-Gains werden um den jeweiligen Kanalhintergrund angewendet und neutralisieren ihn daher nicht implizit.

### `pcc.chroma_strength`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Default** | `1.0` |

**Zweck:** Globaler Staerkefaktor fuer Chroma-Korrektur bei PCC-Apply.

### `pcc.k_max`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Default** | `3.2` |

**Zweck:** Obergrenze fuer Korrekturstaerke im linearen PCC-Apply (verringert Farbstiche in hellen Strukturen).

---

## 19. HyperMetric Stretch

VeraLux HyperMetric Stretch (HMS) ist eine optionale finale RGB-Stretch-Phase nach PCC. Die Phase arbeitet auf `stacked_rgb_pcc.fits` bzw. den PCC-Kanälen und schreibt standardmäßig `outputs/stacked_rgb_hms.fits`.

### `hypermetric_stretch.enabled`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Aktiviert die HMS-Phase nach PCC. Der Default ist aktiv; bei `false` wird die Phase übersprungen.

### `hypermetric_stretch.require_successful_pcc`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Erzwingt, dass HMS nur nach erfolgreicher PCC ausgeführt wird. Beim Resume historischer Runs kann HMS mit vorhandenen PCC-Artefakten direkt über `--from-phase HYPERMETRIC_STRETCH` gestartet werden.

### `hypermetric_stretch.mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Werte** | `ready_to_use`, `scientific` |
| **Default** | `ready_to_use` |

**Zweck:** `ready_to_use` entspricht dem VeraLux-GUI-Standardmodus mit automatischem Output-Scaling, Zielhintergrund und Soft-Clip. `scientific` lässt dieses finale Ready-to-Use-Scaling weg und erlaubt `linear_expansion`.

### `hypermetric_stretch.sensor_profile`, `hypermetric_stretch.fallback_profile`

| Key | Typ | Default |
|-----|-----|---------|
| `hypermetric_stretch.sensor_profile` | string | `rec709` |
| `hypermetric_stretch.fallback_profile` | string | `rec709` |

**Zweck:** Luminanzgewichte für VeraLux. Der Default ist explizit `rec709`; konkrete VeraLux-Profilnamen können direkt gesetzt werden. `auto` bleibt aus Kompatibilitätsgründen akzeptiert und verwendet aktuell `fallback_profile`, wird aber nicht als Default empfohlen. Die Profile sind in `tile_compile_cpp/src/image/hypermetric_stretch.cpp` in `profiles()` definiert. Die Eingabe wird tolerant normalisiert (Groß-/Kleinschreibung, Leerzeichen und Satzzeichen sind nicht relevant), empfohlen sind aber die exakten Namen:

| YAML-Wert | R | G | B |
|-----------|---|---|---|
| `rec709` | 0.2126 | 0.7152 | 0.0722 |
| `Rec.709 (Recommended)` | 0.2126 | 0.7152 | 0.0722 |
| `Sony IMX571 (ASI2600/QHY268)` | 0.2944 | 0.5021 | 0.2035 |
| `Sony IMX455 (ASI6200/QHY600)` | 0.2987 | 0.5001 | 0.2013 |
| `Sony IMX410 (ASI2400)` | 0.3015 | 0.5050 | 0.1935 |
| `Sony IMX269 (Altair/ToupTek)` | 0.3040 | 0.5010 | 0.1950 |
| `Sony IMX294 (ASI294)` | 0.3068 | 0.5008 | 0.1925 |
| `Sony IMX533 (ASI533)` | 0.2910 | 0.5072 | 0.2018 |
| `Sony IMX676 (ASI676)` | 0.2880 | 0.5100 | 0.2020 |
| `Sony IMX585 (ASI585) - STARVIS 2` | 0.3431 | 0.4822 | 0.1747 |
| `Sony IMX662 (ASI662) - STARVIS 2` | 0.3430 | 0.4821 | 0.1749 |
| `Sony IMX678 (ASI678) - STARVIS 2` | 0.3426 | 0.4825 | 0.1750 |
| `Sony IMX415 (DWARF II)` | 0.2703 | 0.5405 | 0.1892 |
| `Sony IMX462 (ASI462)` | 0.3333 | 0.4866 | 0.1801 |
| `Sony IMX715 (ASI715)` | 0.3410 | 0.4840 | 0.1750 |
| `Sony IMX482 (ASI482)` | 0.3150 | 0.4950 | 0.1900 |
| `Sony IMX183 (ASI183)` | 0.2967 | 0.4983 | 0.2050 |
| `Sony IMX178 (ASI178)` | 0.2346 | 0.5206 | 0.2448 |
| `Sony IMX224 (ASI224)` | 0.3402 | 0.4765 | 0.1833 |
| `Canon EOS (Modern - 60D/600D/500D)` | 0.2600 | 0.5200 | 0.2200 |
| `Canon EOS (Legacy - 300D/40D/20D)` | 0.2450 | 0.5350 | 0.2200 |
| `Nikon DSLR (Modern - D5100/D7200)` | 0.2650 | 0.5100 | 0.2250 |
| `Nikon DSLR (Legacy - D3/D300/D90)` | 0.2500 | 0.5300 | 0.2200 |
| `Fujifilm X-Trans 5 HR` | 0.2800 | 0.5100 | 0.2100 |
| `Panasonic MN34230 (ASI1600)` | 0.2650 | 0.5250 | 0.2100 |
| `ZWO Seestar S50` | 0.3333 | 0.4866 | 0.1801 |
| `ZWO Seestar S30` | 0.2928 | 0.5053 | 0.2019 |
| `Narrowband HOO` | 0.5000 | 0.2500 | 0.2500 |
| `Narrowband SHO` | 0.3333 | 0.3400 | 0.3267 |

### `hypermetric_stretch.adaptive_anchor`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Nutzt den adaptiven VeraLux-Anchor aus dem Luminanzhistogramm. Bei `false` wird der statistische 0.5%-Floor-Anker verwendet.

### `hypermetric_stretch.target_bg`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Bereich** | 0.05 – 0.50 |
| **Default** | `0.20` |

**Zweck:** Zielhintergrund für Auto-LogD und Ready-to-Use Output-Scaling.

### `hypermetric_stretch.protect_b`, `hypermetric_stretch.convergence_power`

| Key | Typ | Default | Constraint |
|-----|-----|---------|------------|
| `hypermetric_stretch.protect_b` | number | `6.0` | >= 0.1 |
| `hypermetric_stretch.convergence_power` | number | `3.5` | 1.0 – 10.0 |

**Zweck:** VeraLux-Hyperbelparameter und Farbkonvergenzstärke.

### `hypermetric_stretch.log_d_mode`, `hypermetric_stretch.fixed_log_d`

| Key | Typ | Default | Constraint |
|-----|-----|---------|------------|
| `hypermetric_stretch.log_d_mode` | string | `auto` | `auto`, `fixed` |
| `hypermetric_stretch.fixed_log_d` | number | `2.0` | 0 – 7 |

**Zweck:** `auto` berechnet LogD aus dem VeraLux-AutoSolver. `fixed` verwendet `fixed_log_d`.

### `hypermetric_stretch.color_strategy`, `hypermetric_stretch.fixed_color_strategy`

| Key | Typ | Default | Constraint |
|-----|-----|---------|------------|
| `hypermetric_stretch.color_strategy` | string | `fixed` | `auto`, `fixed` |
| `hypermetric_stretch.fixed_color_strategy` | number | `0.0` | -1 – 1 |

**Zweck:** Steuert die VeraLux-GUI-Farbstrategie. Der Default `fixed: 0` entspricht dem Python-Slider-Default ohne zusätzliche StarPressure-Automatik.

### `hypermetric_stretch.color_grip`, `hypermetric_stretch.shadow_convergence`, `hypermetric_stretch.linear_expansion`

| Key | Typ | Default | Constraint |
|-----|-----|---------|------------|
| `hypermetric_stretch.color_grip` | number | `1.0` | 0 – 1 |
| `hypermetric_stretch.shadow_convergence` | number | `0.0` | >= 0 |
| `hypermetric_stretch.linear_expansion` | number | `0.0` | 0 – 1 |

**Zweck:** Direkte VeraLux-Parameter. `linear_expansion` wirkt nur in `mode: scientific`; in `ready_to_use` wird es wie in der Python-Vorlage ignoriert.

### `hypermetric_stretch.write_channels`, `hypermetric_stretch.output_rgb`

| Key | Typ | Default |
|-----|-----|---------|
| `hypermetric_stretch.write_channels` | boolean | `false` |
| `hypermetric_stretch.output_rgb` | string | `stacked_rgb_hms.fits` |

**Zweck:** Ausgabesteuerung für HMS. `write_channels: true` schreibt zusätzlich `hms_R.fit`, `hms_G.fit`, `hms_B.fit`.

---

## 20. Stacking

Der klassische Stacking-Schritt ist in der Forward-Drizzle-Pipeline ein
Pass-Through: das rekonstruierte Bild stammt aus FORWARD_DRIZZLE/MULTIBAND;
es werden keine Stack-Methoden, Sigma-Clips oder Cluster-Gewichtungen mehr
ausgefuehrt. Uebrig ist nur die optionale Per-Frame-Kosmetik vor dem Warp.

### `stacking.per_frame_cosmetic_correction`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `false` |

**Zweck:** Hotpixel-Korrektur **pro Frame vor dem Warp**.

Diese Option zielt auf **fixe Sensordefekte** (RGB-Einzelpixel), die in jedem Frame an der gleichen Position auftreten und deshalb durch die robuste Beitragskontrolle in FORWARD_DRIZZLE nicht zuverlässig entfernt werden.

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

## 22. Runtime Limits

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

**Zweck:** Speicherbudget für den OSC-Memory-Cap in der Rekonstruktion. Reduziert bei Bedarf die effektive Anzahl paralleler Worker, um RAM-Spitzen zu begrenzen.

**Autogrow:** Der konfigurierte Wert ist ein Startpunkt, keine harte Obergrenze. Wenn das geplante Arbeits-Set (z. B. zurückgehaltene Registrierungs-Proxys, Quellbilder, Streifen-Scratch) nicht hineinpasst, erhöht der Memory-Planner das effektive Budget in Schritten von 1 GiB, solange freier Hauptspeicher verfügbar ist (gedeckelt auf ~80 % des aktuell verfügbaren RAM). Jede Erhöhung wird als Warnung ins Run-Log geschrieben. Passt das Arbeits-Set auch nach Erreichen des Headroom-Caps nicht, bricht der Run weiterhin fail-closed mit `DRIZZLE_MEMORY_BUDGET` ab.

### `runtime_limits.hard_abort_hours`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Minimum** | >0 |
| **Default** | `6.0` |

**Zweck:** Maximale Laufzeit in Stunden bevor die Pipeline abgebrochen wird (Hard Limit).

**Laufzeitverhalten:** Wird im Hauptlauf und im Resume-Pfad nach größeren
Phasengrenzen geprüft. Bei Überschreitung endet der Lauf mit
`runtime_limit_exceeded`.

---

### `runtime_limits.acceleration_backend`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string (enum) |
| **Default** | `auto` |
| **Gültige Werte** | `auto`, `cpu`, `opencv_cuda`, `opencv_opencl`, `opencl`, `cuda` |

**Zweck:** GPU-Beschleunigung für PREWARP-, TILE_RECONSTRUCTION- und STACKING-Phasen.

**Optionen:**
- `auto` (Standard): Erkennt automatisch verfügbare GPU-Backends zur Laufzeit. Priorität: CUDA → OpenCL → CPU. Fällt sauber auf CPU zurück, wenn Hardware nicht verfügbar.
- `opencv_cuda`: Erzwingt NVIDIA CUDA-Backend (benötigt CUDA-fähigen OpenCV-Build und NVIDIA-GPU).
- `opencv_opencl` / `opencl`: Erzwingt OpenCL-Backend (benötigt OpenCL-fähigen OpenCV-Build; funktioniert mit AMD Radeon, Intel iGPU, NVIDIA-GPUs).
- `cpu`: Deaktiviert GPU-Beschleunigung vollständig.
- `cuda`: Experimentelles natives CUDA-Backend (noch nicht implementiert).

**Hardware-Kompatibilität:**
- **NVIDIA-GPUs:** Sowohl `opencv_cuda` (empfohlen für beste Performance) als auch `opencv_opencl` funktionieren.
- **AMD-GPUs (Radeon RX 470/480/570/580/590, Vega, RDNA):** Verwende `opencv_opencl` oder `auto`.
- **Intel integrierte GPUs:** Verwende `opencv_opencl` oder `auto`.

**Build-Anforderungen:**
- CUDA-Backend: OpenCV mit `WITH_CUDA=ON` und Modulen `opencv2/core/cuda.hpp`, `opencv2/cudawarping.hpp`, `opencv2/cudaarithm.hpp`.
- OpenCL-Backend: OpenCV mit `WITH_OPENCL=ON` und Modul `opencv2/core/ocl.hpp`.

**Hinweis:** Falls das angeforderte Backend nicht verfügbar ist (fehlende OpenCV-Module oder Hardware), fällt die Pipeline mit Warnung auf CPU zurück.

---

## Beispiel-Konfiguration (Vollständig)

```yaml
# Output
output:
  registered_dir: registered
  write_registered_frames: false
  crop_to_nonzero_bbox: true

# Data
data:
  color_mode: OSC
  bayer_pattern: auto

# Linearity
linearity:
  enabled: true
  max_frames: 8
  min_overall_linearity: 0.9
  strictness: strict

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
  prewarp_interpolation: lanczos4
  debayer_first: true
  pre_debayer_method: linear

# Dithering
dithering:
  enabled: true
  min_shift_px: 0.7

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

# Reconstruction (CFA Forward Drizzle + Multiband)
reconstruction:
  delete_source_cache_after_run: false
  keep_profile_cache_after_run: false
  common_overlap_required_fraction: 1.0
  diagnostics:
    level: summary
    preview_forward_drizzle_uniform: false
    persist_forward_drizzle_uniform_store: false
  drizzle:
    internal_scale: 2
    output_scale: 1
    kernel: square
    pixfrac: 0.8
    robust_passes: 2
    min_clip_contributors: 5
    chunk_rows: 0
    chunk_halo_rows: -1
    memory_budget_mb: 0
  clipping:
    clip_sigma_low: 3.0
    clip_sigma_high: 3.0
    min_fraction: 0.4
    min_n_eff: 3.0
    guard_fallback: false
  coverage_gate:
    min_frames: 2
    min_supported_fraction: 0.995
    min_channel_n_eff_floor: 3.0
    min_channel_n_eff_fraction: 0.15
    min_analysis_pixels: 1024
    max_internal_hole_area_px: 0
  multiband:
    enabled: true
    levels: 3
    alpha_cap: 1.0
    fine_quality_exponent: 4.0
    medium_quality_exponent: 2.0
    min_quality_separation: 0.05
    full_quality_separation: 0.20
    min_effective_samples: 8.0
    full_effective_samples: 24.0
  quality:
    pyramid:
      scales: 4
      base_window_px: 4
      sharpness_weight: 0.6
      snr_weight: 0.4
      score_scale: 1.8
      artifact_sigma: 3.0
      max_artifact_fraction: 0.25

# Astrometry
astrometry:
  enabled: true
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

# HyperMetric Stretch
hypermetric_stretch:
  enabled: false
  require_successful_pcc: true
  mode: ready_to_use
  sensor_profile: rec709
  fallback_profile: rec709
  adaptive_anchor: true
  target_bg: 0.15
  protect_b: 6.0
  convergence_power: 3.5
  log_d_mode: auto
  fixed_log_d: 2.0
  color_strategy: fixed
  fixed_color_strategy: 0.0
  color_grip: 1.0
  shadow_convergence: 0.0
  linear_expansion: 0.0
  write_channels: false
  output_rgb: stacked_rgb_hms.fits

# Stacking (nur Per-Frame-Kosmetik; STACKING ist Pass-Through)
stacking:
  per_frame_cosmetic_correction: false
  per_frame_cosmetic_correction_sigma: 5.0

# Runtime Limits
runtime_limits:
  parallel_workers: 8
  memory_budget: 4096
  hard_abort_hours: 6.0
  acceleration_backend: auto
```

---

## Hinweise

### Abweichungen `tile_compile.yaml` vs. C++ Defaults

Die Datei `tile_compile.yaml` im Repository enthält eine **Beispiel-/Szenario-Konfiguration** und weicht bewusst von den reinen C++ Defaults ab. Zusätzlich enthält sie Workflow-Metadaten wie `run_dir`, `log_level` und `scenario_profile`, die nicht Teil des reinen C++-Schemas sind.

| Key | `tile_compile.yaml` | C++ Default | Bemerkung |
|-----|---------------------|-------------|-----------|
| `global_metrics.adaptive_weights` | `true` | `false` | Adaptive Gewichtung aktiv |
| `global_metrics.weight_exponent_scale` | `1.8` | `1.0` | Staerkere Trennschaerfe |
| `global_metrics.weights.*` | `0.25/0.25/0.2/0.2/0.1/0` | `0.4/0.3/0.3/0/0/0` | Fuenf-Metrik-Mix inkl. FWHM/Roundness |
| `registration.star_topk` | `80` | `150` | Weniger Sterne im Matching |
| `reconstruction.diagnostics.level` | `full` | `summary` | Volle Diagnostik |
| `reconstruction.drizzle.robust_passes` | `4` | `2` | Zwei zusaetzliche Passes |
| `reconstruction.clipping.clip_sigma_low` / `clip_sigma_high` | `2.0` / `4.0` | `3.0` / `3.0` | Asymmetrisches Produktions-Tuning |
| `reconstruction.clipping.min_n_eff` | `2.0` | `3.0` | Niedrigere N_eff-Schwelle |
| `reconstruction.quality.pyramid.score_scale` | `0.6` | `1.8` | Flachere Score-Sigmoid |
| `reconstruction.delete_source_cache_after_run` | `true` | `false` | Cache-Bereinigung |
| `runtime_limits.parallel_workers` | `8` | `4` | Mehr Worker |
| `runtime_limits.memory_budget` | `4096` | `512` | Groesseres MeB-Budget |

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

## 23. Raw Stack / Preprocessing {#raw-stack-preprocessing}

Raw Stack ist ein separater Preprocessing-Prozess und gehoert nicht zum normalen `tile_compile.yaml`-Hauptstrang. Die Konfiguration wird ueber die Preprocessing-API und den Raw-Stack-Parametereditor verwendet:

- `GET /api/tools/preprocessing/defaults`
- `GET /api/tools/preprocessing/parameters`
- `PATCH /api/tools/preprocessing/parameters`
- `POST /api/tools/preprocessing/run`

Der Prozess teilt Code und Algorithmen mit Tile-Compile, erscheint aber nicht im normalen Run Studio oder normalen Parameter Studio.

### `preprocessing.mode`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | string |
| **Default** | `linear_prestack` |
| **Werte** | `linear_prestack` |

**Zweck:** Aktiviert den klassischen linearen Pre-Stack-Pfad ohne Tile-Grid, Tile-Rekonstruktion, Synthetic Frames oder State Clustering.

### Input und CFA/Mono

| Parameter | Typ | Default | Zweck |
|-----------|-----|---------|-------|
| `lights_dir` | string | `""` | Light-/Raw-Eingabeordner; in der GUI 1:1 ueber `Input & Scan`-Controls. |
| `bias_dir`, `darks_dir`, `flats_dir`, `darkflats_dir` | string | `""` | Kalibrierordner. |
| `input_mode` | string | `auto` | `auto`, `cfa_osc`, `mono`. |
| `raw_formats` | string | `tile_compile` | Nutzt denselben Raw-/FITS-Importumfang wie Tile-Compile. |
| `bayer_pattern` | string | `auto` | Bayer-Pattern aus Header oder explizit `RGGB`, `GBRG`, `GRBG`, `BGGR`. |
| `cfa_mode` | string | `tile_compile` | CFA-/OSC-Behandlung ueber Tile-Compile-Logik. |
| `mono_mode` | string | `auto` | Mono ohne kuenstliche RGB-/Bayer-Annahmen. |
| `registration_reference` | string | `best_quality` | Referenzframe-Auswahl. |

### `calibration.*`

| Parameter | Typ | Default |
|-----------|-----|---------|
| `calibration.use_bias` | boolean | `false` |
| `calibration.use_dark` | boolean | `false` |
| `calibration.use_flat` | boolean | `false` |
| `calibration.bias_use_master`, `dark_use_master`, `flat_use_master`, `darkflat_use_master` | boolean | `false` |
| `calibration.dark_auto_select` | boolean | `true` |
| `calibration.dark_match_use_temp` | boolean | `false` |
| `calibration.dark_match_exposure_tolerance_percent` | number | `8.0` |
| `calibration.dark_match_temp_tolerance_c` | number | `3.0` |
| `calibration.bias_master`, `dark_master`, `flat_master`, `darkflat_master` | string | `""` |
| `calibration.pattern` | string | `*.fit;*.fits;*.fts;*.fit.fz;*.fits.fz;*.fts.fz` |

### Quality und Stacking

| Parameter | Typ | Default | Werte |
|-----------|-----|---------|-------|
| `quality_filter.mode` | string | `auto` | `auto`, `strict`, `relaxed`, `off` |
| `quality_filter.min_stars` | integer | `30` | >= 0 |
| `quality_filter.max_fwhm_sigma` | number | `2.0` | > 0 |
| `quality_filter.max_eccentricity` | number | `0.65` | 0 - 1 |
| `quality_filter.min_correlation` | number | `0.75` | 0 - 1 |
| `quality_filter.manual_overrides` | object | `{}` | Optionale Frame-Overrides nach Index oder Dateiname, z. B. `"0": {"include": false}`. |
| `rejection.method` | string | `sigma` | `sigma`, `median`, `winsor` |
| `rejection.low`, `rejection.high` | number | `3.0` | > 0 |
| `stacking.normalization` | string | `addscale` | `addscale`, `background`, `median`, `none` |
| `stacking.weighting` | string | `quality` | `quality`, `uniform` |

### Postprocess und HMS

| Parameter | Typ | Default |
|-----------|-----|---------|
| `postprocess.astrometry` | boolean | `true` |
| `postprocess.bge` | boolean | `true` |
| `postprocess.pcc` | boolean | `true` |
| `postprocess.hypermetric_stretch` | boolean | `true` |

HMS ist per Default aktiv. Die Detailparameter entsprechen dem normalen Tile-Compile-HMS-Vertrag und sind im Raw-Stack-Screen nur im Parametereditor editierbar:

| Parameter | Typ | Default |
|-----------|-----|---------|
| `hypermetric_stretch.require_successful_pcc` | boolean | `true` |
| `hypermetric_stretch.mode` | string | `ready_to_use` |
| `hypermetric_stretch.sensor_profile` | string | `rec709` |
| `hypermetric_stretch.fallback_profile` | string | `rec709` |
| `hypermetric_stretch.adaptive_anchor` | boolean | `true` |
| `hypermetric_stretch.target_bg` | number | `0.15` |
| `hypermetric_stretch.protect_b` | number | `6.0` |
| `hypermetric_stretch.convergence_power` | number | `3.5` |
| `hypermetric_stretch.log_d_mode` | string | `auto` |
| `hypermetric_stretch.fixed_log_d` | number | `2.0` |
| `hypermetric_stretch.color_strategy` | string | `fixed` |
| `hypermetric_stretch.fixed_color_strategy` | number | `0.0` |
| `hypermetric_stretch.color_grip` | number | `1.0` |
| `hypermetric_stretch.shadow_convergence` | number | `0.0` |
| `hypermetric_stretch.linear_expansion` | number | `0.0` |
| `hypermetric_stretch.write_channels` | boolean | `false` |
| `hypermetric_stretch.output_rgb` | string | `stacked_rgb_hms.fits` |

### Report

| Parameter | Typ | Default |
|-----------|-----|---------|
| `report.detailed` | boolean | `true` |
| `report.formats` | list | `[json, markdown, html]` |

Raw Stack schreibt Reportdaten unter `artifacts/preprocess/`:

- `preprocessing_report.json`
- `preprocessing_report.md`
- `preprocessing_report.html`
- `frame_quality.csv`
- `rejected_frames.txt`
- `events.jsonl`
- `artifacts_manifest.json`

## Anhang A — Funktionsdetails für alle Optionen

Dieser Anhang beschreibt pro Schlüssel explizit das **Laufzeitverhalten** (Wirkung in der Pipeline, Interaktionen, Nebenwirkungen).

### A.1 Pipeline / Output / Data

- `output.registered_dir`: Ziel-Unterordner für registrierte Frame-Ausgaben.
- `output.write_registered_frames`: schreibt pro Frame registrierte FITS; erhöht IO- und Speicherbedarf stark.
- `output.crop_to_nonzero_bbox`: schneidet den finalen Stack auf die nichtleere Bounding Box zu.
- `data.image_width`, `data.image_height`: optionale Erwartungswerte; normalerweise FITS-headerbasiert erkannt.
- `data.color_mode`: erwarteter Aufnahmemodus; Laufzeit-Autodetektion kann mit Warnung übersteuern.
- `data.bayer_pattern`: CFA-Layout für OSC-Verarbeitung und korrekte Farbrekonstruktion.

### A.2 Linearity / Calibration

- `linearity.enabled`: aktiviert Linearitätsdiagnostik in Scan/Frühvalidierung.
- `linearity.max_frames`: Stichprobengröße der Linearitätsprüfung (Speed vs. Sicherheit).
- `linearity.min_overall_linearity`: Schwellwert für die Linearity-Diagnosewarnung.
- `linearity.strictness`: Schwellenpreset für robuste Verteilungs- und Clippingdiagnostik.
- `calibration.use_bias`, `use_dark`, `use_flat`: schaltet jeweilige Master-Kalibrierstufe ein.
- `calibration.bias_use_master`, `dark_use_master`, `flat_use_master`: nutzt explizite Masterdateien statt Directory-Stacking.
- `calibration.dark_auto_select`: automatische Dark-Master-Auswahl nach Belichtungszeit (optional Temperatur).
- `calibration.dark_match_exposure_tolerance_percent`: erlaubte Belichtungsabweichung beim Dark-Matching.
- `calibration.dark_match_use_temp`: aktiviert temperaturbasiertes Dark-Matching.
- `calibration.dark_match_temp_tolerance_c`: erlaubte Temperaturabweichung bei aktivem Temp-Matching.
- `calibration.bias_dir`, `darks_dir`, `flats_dir`: Quellordner für Kalibrierframe-Findung.
- `calibration.bias_master`, `dark_master`, `flat_master`: explizite Pfade zu Master-Kalibrierframes.
- `calibration.pattern`: Glob-Muster für Kalibrierdatei-Lookup.

### A.3 Normalization / Registration / Dithering

- `normalization.enabled`: methodikgetrieben verpflichtend (normalerweise aktiv lassen).
- `normalization.mode`: Hintergrund- vs. Median-zentrierte Normalisierungsstrategie.
- `normalization.per_channel`: kanalweise (OSC/RGB) Normalisierung zur Balance-Erhaltung.
- `registration.engine`: bevorzugte Startmethode; Laufzeit nutzt trotzdem Fallback-Kaskade.
- `registration.enable_star_pair_fallback`: aktiviert/deaktiviert den zusätzlichen (nicht-normativen) Star-Pairs-Fallback.
- `registration.allow_rotation`: erlaubt Rotationsanteile in globalen Warps (Pflicht für Alt/Az).
- `registration.star_topk`: Anzahl starker Sterne für sternbasierte Engines.
- `registration.star_min_inliers`: minimale akzeptierte Inlier-Korrespondenzen.
- `registration.star_inlier_tol_px`: geometrische Inlier-Toleranz.
- `registration.star_dist_bin_px`: Distanzhistogramm-Quantisierung für Star-Similarity.
- `registration.reject_outliers`: robustes Verwerfen unplausibler Warps nach Matching.
- `registration.reject_cc_min_abs`: absolute NCC-Untergrenze in Outlier-Logik.
- `registration.reject_shift_px_min`: absolute Shift-Untergrenze für Shift-Outlier.
- `registration.reject_shift_median_multiplier`: relativer Shift-Schwellwert zur Median-Shift.
- `registration.reject_scale_min`, `reject_scale_max`: erlaubtes Similarity-Scale-Band.
- `registration.prewarp_interpolation`: Interpolationsmethode der Warp/Resample-Schritte.
- `registration.debayer_first`: OSC — Demosaicing vor Prewarp statt CFA-Prewarp.
- `registration.pre_debayer_method`: Demosaicing-Methode fuer den `debayer_first`-Pfad.
- `dithering.enabled`: aktiviert Dither-Diagnostik im Registration-Artefakt.
- `dithering.min_shift_px`: Mindest-Frame-Shift, um als Dither zu zählen.

### A.4 Chroma Denoise


- `chroma_denoise.enabled`: aktiviert chroma-fokussierte Denoise (OSC-Pfad).
- `chroma_denoise.color_space`: Chroma/Luma-Transform (`ycbcr_linear` oder `opponent_linear`).
- `chroma_denoise.apply_stage`: Ausführung vor Tile-OLA, nach finalem linearem Stack, nach PCC oder an beiden Stack-Stellen (`both`).
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
- `chroma_denoise.extended_source_protection.enabled`: Schutz für Extended Sources (Galaxien-Discs, große Nebel) vor Chroma-Denoising. Aktivieren, wenn das Feld eine Galaxie oder einen großen Nebel enthält, damit der Denoiser die echte Farbe der Quelle nicht als Rauschen behandelt.
- `chroma_denoise.extended_source_protection.luma_sigma`: Detektionsschwelle über dem Himmelshintergrund in σ-Einheiten (0–5). Niedrigere Werte schützen mehr; höhere nur den hellsten Kern.
- `chroma_denoise.extended_source_protection.dilate_px`: Dilations-Radius (Pixel) nach der Erkennung, um PSF-Halos abzudecken.

### A.5 Global Metrics / Reconstruction

- `global_metrics.weights.background`, `noise`, `gradient`, `fwhm`, `roundness`, `star_count`: gewichtete Terme der globalen Frame-Qualität.
- `global_metrics.clamp`: harte Grenzen vor exponentiellem Weight-Mapping.
- `global_metrics.adaptive_weights`: Auto-Anpassung der Metrikgewichte aus Streuung.
- `global_metrics.weight_exponent_scale`: Trennschärfe in `exp(k*Q)`.
- `reconstruction.common_overlap_required_fraction`: Abdeckungs-Schwelle für `COMMON_OVERLAP`.
- `reconstruction.delete_source_cache_after_run`, `keep_profile_cache_after_run`: Cache-Lebenszyklus nach dem Lauf.
- `reconstruction.diagnostics.level`: Diagnoseumfang (`summary`/`full`).
- `reconstruction.diagnostics.preview_forward_drizzle_uniform`, `persist_forward_drizzle_uniform_store`: Uniform-Kontroll-Diagnostik/-Persistenz.
- `reconstruction.drizzle.internal_scale`, `output_scale`: internes vs. Ausgabe-Drizzle-Raster.
- `reconstruction.drizzle.kernel`, `pixfrac`: Kernel und Tropfen-Schrumpfung der Forward-Drizzle-Beiträge.
- `reconstruction.drizzle.robust_passes`: robuste Reprojektions-Durchläufe.
- `reconstruction.drizzle.min_clip_contributors`: Mindestbeitraege für Sample-Clipping.
- `reconstruction.drizzle.chunk_rows`, `chunk_halo_rows`, `memory_budget_mb`: Streaming-/Speicherprofil des Reprojekters.
- `reconstruction.clipping.clip_sigma_low`, `clip_sigma_high`: MAD-Rejection-Schwellen.
- `reconstruction.clipping.min_fraction`, `min_n_eff`: Belegbarkeits-Gates pro Pixel.
- `reconstruction.clipping.guard_fallback`: Fallback statt striktem Veto bei Clip-Versagen.
- `reconstruction.coverage_gate.*`: Fail-closed Deckungs-/Qualitaets-Gates vor FORWARD_DRIZZLE.
- `reconstruction.multiband.*`: Bandfusion-Steuerung (Ebenen, Alpha-Cap, Qualitaets-/N_eff-Schwellen).
- `reconstruction.quality.pyramid.*`: Lokale Qualitaetskarten (siehe §14).

### A.6 Debayer / Astrometry / PCC / HMS / Stacking / Runtime

- Debayer wird ueber `registration.debayer_first`/`pre_debayer_method` gesteuert (siehe §15); es gibt keinen eigenen `debayer`-Block.
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
- `pcc.background_model`: lokales Hintergrundmodell fuer Sternphotometrie.
- `pcc.max_condition_number`, `pcc.max_residual_rms`: Stabilitaetsgrenzen fuer Matrix/Fit.
- `pcc.radii_mode`, `pcc.aperture_fwhm_mult`, `pcc.annulus_inner_fwhm_mult`, `pcc.annulus_outer_fwhm_mult`, `pcc.min_aperture_px`: adaptive Radiussteuerung.
- `pcc.siril_catalog_dir`: optionaler lokaler Siril-Katalogpfad.
- `pcc.apply_attenuation`, `pcc.background_neutralization_mode`, `pcc.chroma_strength`, `pcc.k_max`: optionale PCC-Apply-/Hintergrundneutralisierungs-Kontrollen.
- `hypermetric_stretch.enabled`: aktiviert VeraLux HyperMetric Stretch nach PCC.
- `hypermetric_stretch.require_successful_pcc`: verlangt erfolgreiche PCC-Artefakte vor HMS.
- `hypermetric_stretch.mode`: `ready_to_use` mit finalem Scaling/Soft-Clip oder `scientific` ohne Ready-to-Use-Polish.
- `hypermetric_stretch.sensor_profile`, `fallback_profile`: Luminanzgewichte fuer VeraLux.
- `hypermetric_stretch.adaptive_anchor`: adaptiver Histogramm-Anchor statt statistischem Floor.
- `hypermetric_stretch.target_bg`: Zielhintergrund fuer Auto-LogD und Ready-to-Use-Scaling.
- `hypermetric_stretch.protect_b`, `convergence_power`: VeraLux-Stretch- und Farbkonvergenzparameter.
- `hypermetric_stretch.log_d_mode`, `fixed_log_d`: automatische oder fixe Stretch-Stärke.
- `hypermetric_stretch.color_strategy`, `fixed_color_strategy`, `color_grip`, `shadow_convergence`: Farbstrategie und Hybrid-Grip-Parameter.
- `hypermetric_stretch.linear_expansion`: nur in `mode: scientific` wirksame lineare Expansion.
- `hypermetric_stretch.write_channels`, `output_rgb`: HMS-Ausgabeoptionen.
- `runtime_limits.parallel_workers`: Obergrenze für Worker-Threads.
- `runtime_limits.memory_budget`: Speicherbudget, das effektive Parallelität begrenzen kann.
- `runtime_limits.hard_abort_hours`: absolute Runtime-Sicherheitsgrenze.

## Forward-Drizzle v2: Streaming und Speicherbudget

FORWARD_DRIZZLE verwendet ausschließlich den transaktionalen v2-Bandpfad; es
gibt keinen Methoden- oder Umgebungsvariablen-Schalter zum alten Produzenten.
Source-, Qualitäts- und gestartete Samplebereiche werden bandbegrenzt gelesen,
der CPU-/CUDA-Workspace wird über alle Bänder wiederverwendet und MULTIBAND
verlangt den publizierten v2-Store fail-closed. CUDA-Gerätefehler starten die
gesamte unveröffentlichte Phase auf `cpu_v2` neu. Der separate
CPU-Coverage-/Uniform-Previewpfad bleibt eine standardmäßig deaktivierte
Diagnose.

| Parameter | Einheit, Bereich und Default | Verhalten |
|---|---|---|
| `reconstruction.drizzle.memory_budget_mb` | MiB, Ganzzahl >=0, Default 0 | 0 übernimmt `runtime_limits.memory_budget`; direkte Bibliotheksaufrufe verwenden 512 MiB. Berücksichtigt zurückgehaltene Ergebnisse/Masken, Quellbild samt temporärer Ladekopie, Arbeitsstreifen und Reserve. Verfügbarer Host-/cgroup-Speicher kann das Budget zusätzlich begrenzen. Ist das Budget zu klein, wächst es automatisch in 1-GiB-Schritten (mit Warnung im Log) bis zum Headroom-Cap; siehe `runtime_limits.memory_budget`. |
| `reconstruction.drizzle.chunk_rows` | interne Zielzeilen, Ganzzahl >=0, Default 0 | Auto wählt höchstens 256 Zeilen innerhalb des Budgets. Ein expliziter zu großer Wert wird abgelehnt; passt nicht einmal eine Zeile, erfolgt ein kontrollierter Fehler vor der großen Allokation. |
| `reconstruction.drizzle.chunk_halo_rows` | Zeilen, Ganzzahl >=-1, Default -1 | Kompatibilitätsfeld. Die exakte Quellfootprint-Aufzählung erfasst auch über Streifengrenzen ragende Droplets; im CPU-Uniform-/Coverage-Pfad werden keine Ausgabe-Halozeilen benötigt. |
| `reconstruction.common_overlap_required_fraction` | Anteil, (0,1], Default 1 | Anteil akzeptierter dichter Frame-Footprints für die unabhängige Analysefläche. Keine Schnittmenge aller dünnen R/G/B-Droplets. |
| `reconstruction.diagnostics.preview_forward_drizzle_uniform` | Bool, Default false | Streaming-Diagnose mit Summenstatistik; kein fertiger Stack, kein Resume-Commit. |

Coverage und Uniform benutzen denselben Polygonkernel. `n_eff=(sum B)^2/sum(B^2)`
verwendet geometrische Framegewichte; fehlender Support zählt in der Analysefläche
als null. Eine leere Analysefläche ist ein Gatefehler, kein impliziter Pass.
Coverage hält produktiv nur zwei Vollbild-Bytemasken; Framepuffer sind streifenweise.
Exakte Perzentile verwenden temporäre Floatspools mit begrenztem Lesepuffer,
maximal etwa `4 * aktive_Kanäle * interne_Pixelzahl` Byte Diskdaten. Vorab werden
zusätzlich 64 MiB freier Temporärspeicher verlangt. Die Lochsuche benötigt nur
zwei Scanlines. FITS-Maskenexport benötigt eine Floatzeile statt eines Floatbildes.

CPU-Referenz: ein Worker, feste Framefolge. Streaming kann Quellframes pro Streifen
neu laden; affine Quellzeilen werden geometrisch eingeschränkt, lokale Warps
konservativ erneut geprüft. Mehr I/O ist der bewusste Tausch gegen begrenzten RAM.
Die Preview gibt `estimated_peak_bytes`, `resolved_chunk_rows` und `workers_used`
aus. Dies ist eine Allokationsschätzung, kein gemessener Gesamtprozess-RSS; bereits
vorhandene Registrierungsdaten und konkurrierende Prozesse bleiben gesondert zu
berücksichtigen. Es gibt keinen automatischen Methoden- oder Skalenfallback.
Ein Beispiel steht in `tile_compile_cpp/examples/forward_drizzle_streaming.example.yaml`.

Der gemeinsame M3-Uniform/Raw-Bibliothekspfad rechnet zusätzlich im Worst Case einen Clipping-Kandidaten je Frame, Pixel und Kanal sowie zwei Ausgabestreifen an. Die materialisierende Komfortfunktion rechnet beide Vollausgaben zusätzlich an. Eine nachträgliche Kandidatenprüfung ist kein Ersatz für diese Vorabplanung. Der aktuelle Diagnose-Profilstore materialisiert noch Uniform, exportiert seine Ebenen aber ohne zusätzliche Vollbildkopien.

`reconstruction.diagnostics.persist_forward_drizzle_uniform_store` (Bool, Default `false`) ist unabhängig von der Preview. Aktiviert schreibt es ungeclippte Uniform-Ebenen per Streaming nach `artifacts/forward_drizzle_uniform_store/generation-…/`; `current.json` veröffentlicht die vollständige geprüfte Generation atomar. Das bestehende Drizzle-Budget umfasst zusätzlich 8 MiB für FITS/Metadaten und eine float-Zeile. Zu wenig RAM wird vor Quell-I/O abgewiesen; zu wenig freie Disk vor dem Ebenenschreiben. Ein Diagnosefehler lässt den Produktionslauf weiterlaufen. Alte Generationen bleiben erhalten und belegen Disk; es gibt keine automatische Bereinigung. Der Store ist kein Resume-Phaseneinstieg. Leser müssen `current.json` gegen erwartete Quell-, Sampling- und Algorithmusidentität prüfen; alte flache Stores werden weder automatisch akzeptiert noch umgeschrieben. Der geclippte Uniform/Raw-Bibliotheksstore ist nur noch ein Dev-/Regression-Oracle und nicht produktiv verdrahtet.

Der geprüfte Quellqualitäts-Bibliothekspfad besitzt ein explizites MiB-Budget und prüft konservativ 128 Byte je Quellpixel zuzüglich Quell-/Ladepuffern, Metadaten und Metrik-Scratch vorab. Große native Bilder können früh abgelehnt werden. Das gemeinsame Drizzle-Budget berücksichtigt zusätzlich den übergebenen dichten Quellgewichtsvektor. Store-Commit-Schema 2 bindet Cache- und Qualitätsplanhash; ältere Diagnosestores werden nicht stillschweigend als geprüfte Vorgänger übernommen. Diese Bibliotheks-APIs schalten kein Pipeline-Resume frei.
