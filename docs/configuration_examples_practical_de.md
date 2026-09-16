# Praktische Konfigurationsbeispiele für tile_compile

**[🇬🇧 English Version](configuration_examples_practical_en.md)**

Dieser Leitfaden ergänzt die Konfigurationsreferenz mit praktischen Beispielen, Grenzbereichen und Anwendungsfällen basierend auf der Methodik v3.3.

## Änderungsstand (Single-Method-Cutover)

- Die Pipeline ist fest auf **CFA Forward Drizzle + Multiband** ausgelegt (`tile_compile_runner reconstruct`); es gibt keinen Methoden-Selektor mehr.
- `method`, `pipeline.mode`, `aqmh.*`, `tile.*`, `tile_denoise.*`, `local_metrics.*`, `synthetic.*`, `validation.*`, `assumptions.*` und die Classic-`stacking.*`-Felder wurden entfernt; die Zuordnung alter auf neue Schluessel steht in der Konfigurationsreferenz §1.
- Die ehemaligen `aqmh.reconstruction.*`-Tuningwerte leben unter `registration.*` (Prewarp/Debayer) und `reconstruction.drizzle.*` / `reconstruction.clipping.*` weiter.
- `aqmh.pyramid.*` heisst jetzt `reconstruction.quality.pyramid.*`.
- `stacking.common_overlap_required_fraction` heisst jetzt `reconstruction.common_overlap_required_fraction`.

**Basis-Snippet (Single-Method):**

```yaml
registration:
  engine: triangle_star_matching
  enable_star_pair_fallback: true
  prewarp_interpolation: lanczos4
  debayer_first: true
  pre_debayer_method: linear

reconstruction:
  common_overlap_required_fraction: 1.0
  diagnostics:
    level: full
  drizzle:
    robust_passes: 4
  clipping:
    clip_sigma_low: 2.0
    clip_sigma_high: 4.0
    min_fraction: 0.4
    min_n_eff: 2.0
```

---

## Rekonstruktion (CFA Forward Drizzle + Multiband)

Die Rekonstruktion ist die einzige Methode und immer aktiv. Relevant sind
v. a. die robuste Beitragskontrolle (`clipping.*`), die
Streaming-/Speichersteuerung (`drizzle.*`) und die lokalen Qualitaetskarten
(`quality.pyramid.*`).

**Produktions-Profil (empfohlen, angelehnt an `tile_compile.yaml`):**

```yaml
registration:
  affine_refinement_enabled: true       # nur bei vollständig bestandenen Residual-/NCC-/Overlap-Gates anwenden
  smooth_local_refinement_enabled: true # zusätzlicher Held-out-/Jacobian-Guard; sonst atomarer Warp-Fallback
  prewarp_interpolation: lanczos4       # schärfste Interpolation; cubic/linear = schnellere Fallbacks
  debayer_first: true                   # OSC: vor dem Prewarp debayern, RGB direkt rekonstruieren
  pre_debayer_method: linear            # Demosaicing-Methode des debayer_first-Pfads

reconstruction:
  delete_source_cache_after_run: true   # Cache nach erfolgreichem Lauf loeschen (Plattenplatz)
  diagnostics:
    level: full
  drizzle:
    robust_passes: 4                    # robuste Reprojektions-Passes
  clipping:
    clip_sigma_low: 2.0                 # untere MAD-Schwelle (aggressiver)
    clip_sigma_high: 4.0                # obere MAD-Schwelle (toleranter)
    min_fraction: 0.4                   # Mindestanteil belegbarer Samples
    min_n_eff: 2.0                      # min. effektive Beitragszahl pro Pixel
    guard_fallback: false               # false = striktes Veto bei Clip-Versagen
  quality:
    pyramid:
      scales: 4
      base_window_px: 4
      sharpness_weight: 0.6   # Schärfe-Gewicht im Qualitätsindex
      snr_weight: 0.4         # SNR-Gewicht im Qualitätsindex
      score_scale: 1.8        # Selektivität der lokalen Qualitätskarten
      artifact_sigma: 3.0     # MAD-Multiplikator für Artefakt-Erkennung
      max_artifact_fraction: 0.25  # max. Artefaktanteil pro Fenster
```

**Toleranter gegenüber Artefakten (Satelliten, Wolken):**

```yaml
reconstruction:
  quality:
    pyramid:
      artifact_sigma: 5.0
      max_artifact_fraction: 0.35
  clipping:
    clip_sigma_low: 1.5
```

**Speichersparend (grosse Sessions, RAM-knapp):**

```yaml
reconstruction:
  drizzle:
    memory_budget_mb: 1024   # explizites Budget; 0 erbt runtime_limits.memory_budget
    chunk_rows: 0            # 0 = budgetierte Streifen (<=256 Zeilen)

runtime_limits:
  parallel_workers: 2
  memory_budget: 1024
```

**Konservativ bei schwarzen Artefaktpixeln:**

```yaml
reconstruction:
  clipping:
    guard_fallback: true     # statt Veto den Survivor-/Unclipped-Wert verwenden
```

**Feldfuellende Objekte (z. B. grosse Galaxien):**

Der `background_rms`-Gate misst Rauschen in Regionen, die bei feldfuellenden
Objekten echte schwache Struktur enthalten. Der gewichtete Stack kann dann
mehr "RMS" zeigen, obwohl er mehr Signal bewahrt - und das Gate verwirft ihn
zugunsten der ungewichteten Kontrolle (Detailverlust). Diagnose: im Report /
`forward_drizzle.json` pruefen, ob `selected_candidate` auf
`drizzle_uniform` zurueckfiel, obwohl `background_rms` nur knapp ueber dem
Schwellwert lag.

```yaml
reconstruction:
  multiband_validation:
    background_rms_ratio_max: 1.15   # toleranter, wenn "Hintergrund" Struktur enthaelt
```

Die uebrigen `multiband_validation.*`-Gates (FWHM, Elongation, Naht)
sollten nur mit Beleg aus den Validierungsartefakten gelockert werden.

---

## Background Gradient Extraction (BGE) - NEU in v3.3

**Wann aktivieren:**
- Sichtbare Hintergrundgradienten (Lichtverschmutzung, Mondlicht)
- PCC zeigt Farbverschiebungen über das Bildfeld
- Städtische/vorstädtische Standorte

**Empfohlene Konfiguration:**

```yaml
bge:
  enabled: true
  method: classic
  autotune:
    enabled: false
    strategy: conservative
    max_evals: 24
    holdout_fraction: 0.25
    alpha_flatness: 0.25
    beta_roughness: 0.10
  tile_weight_lambda_structure: 1.0  # Aktueller Default: moderates Down-Weighting für strukturreiche Tiles
  sample_quantile: 0.20  # Konservativ, resistent gegen schwache Objekte
  min_valid_sample_fraction_for_apply: 0.30  # Kanal-Apply-Grenze (rel. Anteil)
  min_valid_samples_for_apply: 96  # Kanal-Apply-Grenze (absolute Samples)
  fit:
    method: rbf  # Flexibel, empfohlen
    robust_loss: huber  # huber | tukey
    huber_delta: 1.5
    rbf_phi: multiquadric  # Guter Kompromiss
    rbf_mu_factor: 1.0  # Standard-Glättung
```

**Für starke Gradienten (z.B. Stadtrand):**

```yaml
bge:
  enabled: true
  method: classic
  sample_quantile: 0.15  # Noch konservativer
  structure_thresh_percentile: 0.95  # Mehr Tiles ausschließen
  min_valid_sample_fraction_for_apply: 0.30
  min_valid_samples_for_apply: 96
  fit:
    method: rbf
    robust_loss: tukey  # aggressivere Outlier-Daempfung
    rbf_phi: multiquadric
    rbf_mu_factor: 0.8  # Etwas weniger Glättung für Details
```

**Für großflächige diffuse Vordergrundobjekte (z.B. M31 / M42):**

```yaml
bge:
  enabled: true
  method: classic
  min_valid_sample_fraction_for_apply: 0.28  # Toleranter fuer dichte Nebel-/Sternfelder
  min_valid_samples_for_apply: 96
  fit:
    method: modeled_mask_mesh  # Vordergrundbewusstes Mesh-Himmelsmodell
```

**Für schwache Gradienten (z.B. Mondlicht):**

```yaml
bge:
  enabled: true
  method: classic
  sample_quantile: 0.25  # Weniger konservativ
  min_valid_sample_fraction_for_apply: 0.30
  min_valid_samples_for_apply: 96
  fit:
    method: poly  # Einfacher für schwache Gradienten
    polynomial_order: 2
```

**AutoBGE explizit wählen (geplant, opt-in):**

```yaml
bge:
  enabled: true       # Legacy-Kompatibilität; method ist maßgeblich
  method: autobge    # none | classic | autobge
  autobge:
    num_sample_points: 0
    poly_degree: 2
    rbf_smooth: 0.1
    downsample_scale: 4
    patch_size: 15
    patch_estimator: median
    stretch_mode: linear  # none | linear | mtf
    stretch_target_median: 0.25
    border_margin: 10
    bright_exclusion_fraction: 0.5
    gradient_descent_max_iters: 100
    random_seed: 42
    normalize_between_stages: true
    apply_guards: true
    mono_mode: rgb_duplicate
```

**Wichtig:** BGE läuft **vor** PCC. Wenn BGE aktiviert ist, sollte PCC danach bessere Ergebnisse liefern.

Seit dem Forward-Drizzle-Cutover verankert AutoBGE alle drei Kanaele an
einem gemeinsamen Sockel (dunkelster Modell-Median) statt pro Kanal am
eigenen Modell-Median. Damit werden kanalabhaengige
Hintergrund-Pedestale vor PCC/HMS egalisiert. Feuert der Slope-Guard
dennoch (typischerweise weil das Modell einen Kanal-Offset nicht
absorbiert hat), wird ein re-anchored Apply getestet, das die
Residuen-Mediane aller Kanaele exakt ausgleicht; es wird nur
akzeptiert, wenn die Kanal-Level-Streuung messbar sinkt und der
Rest-Tilt begrenzt bleibt (Verhaeltnis-Bound oder absolute
Amplitude <= 25 % der entfernten Streuung). Das steht dann als
`guard_override: "level_equalization"` in `artifacts/bge.json`
(pro Kanal `guard_reason: "slope_worsened_but_level_equalized"`).
Ohne Level-Verbesserung verwirft der Guard die Korrektur weiterhin
vollstaendig.

**PCC-v3.3.6-Optionen (empfohlen mit BGE):**

```yaml
pcc:
  background_model: plane      # median | plane
  max_condition_number: 3.0
  max_residual_rms: 0.35
  radii_mode: auto_fwhm        # fixed | auto_fwhm
  aperture_fwhm_mult: 1.8
  annulus_inner_fwhm_mult: 3.0
  annulus_outer_fwhm_mult: 5.0
  min_aperture_px: 4.0
  apply_attenuation: false
  chroma_strength: 1.0
  background_neutralization_mode: auto  # always | auto | off
  k_max: 3.2
```

`chroma_strength` begrenzt die PCC-Farb-Gains, nicht die Hintergrundneutralisierung. `auto` neutralisiert einen raeumlich kohaerenten globalen Farbstich vollstaendig, schuetzt aber lokal variierende Nebel- oder Feldfarbe.

---

## HyperMetric Stretch nach PCC

HMS ist optional und läuft nach PCC. Deaktiviert lassen, wenn nur das lineare kalibrierte Ergebnis benötigt wird; aktivieren, wenn der Run zusätzlich ein direkt betrachtbares VeraLux-gestretchtes RGB erzeugen soll.

**Ready-to-use-Ausgabe:**

```yaml
hypermetric_stretch:
  enabled: true
  require_successful_pcc: true
  mode: ready_to_use
  adaptive_anchor: true
  target_bg: 0.15
  log_d_mode: auto
  color_strategy: fixed
  fixed_color_strategy: 0
  output_rgb: stacked_rgb_hms.fits
```

`ready_to_use` folgt dem VeraLux-GUI-Default: Auto LogD, adaptives Output-Scaling auf den Zielhintergrund und finaler Soft Clip. Das ist der empfohlene Modus für normale finale RGB-Ausgaben.

**Scientific-Modus:**

```yaml
hypermetric_stretch:
  enabled: true
  mode: scientific
  log_d_mode: auto
  linear_expansion: 0.25
  color_grip: 1.0
  shadow_convergence: 0.0
```

`scientific` überspringt das finale Ready-to-Use-Scaling und den Soft Clip und erlaubt `linear_expansion`. Sinnvoll, wenn ein weniger poliertes, kontrollierteres Stretch-Ergebnis für weitere Bearbeitung gewünscht ist.

---

## Gemeinsamer Overlap (`reconstruction.common_overlap_required_fraction`)

**Aktueller sinnvoller Standardwert:**

```yaml
reconstruction:
  common_overlap_required_fraction: 1.0
```

- `1.0` erzwingt die strikte Schnittmenge aller nutzbaren Frames.
- Niedrigere Werte lassen wieder teilweise überdeckte Randpixel in Metriken, BGE/PCC und Hintergrundstatistiken einsickern.

**Empfehlungen nach Setup:**

- **Alt/Az mit Feldrotation:** `1.0` (empfohlen)
- **EQ mit sehr stabiler Nachfuehrung:** `1.0` (empfohlen, wenn Randbias vermieden werden soll)
- **Nur wenn bewusst mehr Randflaeche zugelassen werden soll:** z. B. `0.98` oder `0.95`

---

## Sichtbare Grenzen / Artefakte diagnostizieren

Die Forward-Drizzle-Pipeline erzeugt keine Kachel-Seams mehr (kein
Overlap-Add-Stacking). Wenn sichtbare Artefakte auftreten, pruefe:

- `reconstruction.coverage_gate.*` — Deckungs-Gates vor FORWARD_DRIZZLE
- `reconstruction.clipping.*` — bei zu aggressiven Sigma-Werten kann Signal
  verworfen werden; bei zu losen Werten bleiben Ausreisser stehen
- `reconstruction.clipping.guard_fallback` — `false` laesst nicht belegbare
  Pixel schwarz; `true` faellt auf Survivor-/Unclipped-Werte zurueck
- `reconstruction.diagnostics.level: full` — maximale Diagnose-Artefakte
- nachgelagerte Unterschiede aus `BGE` oder `PCC`

---

## Hotpixel / RGB-Einzelpixel-Artefakte (fixe Sensordefekte)

Wenn im finalen Bild **isolierte rote/grüne/blaue Einzelpixel** bleiben, sind das meist **fixe Hot Pixel** (Sensorfehler), die in jedem Frame an der gleichen Position auftreten. Diese überleben Sigma-Clipping im Stack, weil sie nicht als Ausreißer über Frames hinweg erscheinen.

**Empfehlung:** Hotpixel **pro Frame vor dem Stack** korrigieren.

```yaml
stacking:
  per_frame_cosmetic_correction: true
  per_frame_cosmetic_correction_sigma: 5.0
```

---

## Audit-Hinweis zu entfernten Parametern

Mit dem Single-Method-Cutover wurden die Bloecke `aqmh.*`, `tile.*`,
`tile_denoise.*`, `local_metrics.*`, `synthetic.*`, `validation.*`,
`assumptions.*`, `pipeline.*`, `method` sowie die Classic-`stacking.*`-Felder
(`method`, `sigma_clip.*`, `cluster_quality_weighting.*`, `output_stretch`,
`tile_common_valid_min_fraction`, `cosmetic_correction*`) entfernt.
Die Zuordnung migrierter Schluessel steht in der Konfigurationsreferenz §1.

Die folgenden Praxisbeispiele verwenden nur noch aktuell aktive Parameter aus Code und Schema.

---

## Registrierung (`registration.*`)

Der aktive Schlüssel ist `registration.engine`, nicht mehr `registration.method`.

**Strikt / normnah:**
```yaml
registration:
  engine: triangle_star_matching
  enable_star_pair_fallback: false
  allow_rotation: true
```

**Alt/Az / Feldrotation / schwierige Sterne:**
```yaml
registration:
  engine: triangle_star_matching
  allow_rotation: true
  enable_star_pair_fallback: true
  star_topk: 150
  star_min_inliers: 4
  star_inlier_tol_px: 4.0
  star_dist_bin_px: 5.0
  reject_outliers: true
  reject_cc_min_abs: 0.25
  reject_shift_px_min: 100.0
  reject_shift_median_multiplier: 5.0
  reject_scale_min: 0.92
  reject_scale_max: 1.08
  # Legacy-Kompatibilitaet; bei independent_global_consensus_v2 ohne Einfluss
  max_blind_chain_depth: 0
  blind_chain_strong_anchor_cc: 0.08
  blind_chain_drift_threshold_px: 2.0
  use_astrometry: true            # Astrometrische Rescue bei Bedarf
  enable_local_background_subtraction: false
  star_shift_radius_px: 200       # Alt/Az: 200-400, Äquatorial: 60
  affine_refinement_enabled: true  # gegatet; bei Ablehnung bleibt der globale Warp unverändert
  smooth_local_refinement_enabled: true # Held-out/Jacobian/NCC-gegatet; MONO oder debayer-first
```

**Sternenarm / Nebel / wolkige Daten:**
```yaml
registration:
  engine: robust_phase_ecc
  allow_rotation: true
  reject_outliers: true
  # Legacy-Kompatibilitaet; bei independent_global_consensus_v2 ohne Einfluss
  max_blind_chain_depth: 0
  blind_chain_strong_anchor_cc: 0.08
  blind_chain_drift_threshold_px: 2.0
  use_astrometry: true
  enable_local_background_subtraction: true  # Bei Mondlicht/Gradienten
  star_shift_radius_px: 200
```

**Gut nachgefuehrte EQ-Montierung:**
```yaml
registration:
  engine: triangle_star_matching
  allow_rotation: true
  # Neue Parameter (v2.0) — Defaults
  max_blind_chain_depth: 0
  blind_chain_strong_anchor_cc: 0.08
  blind_chain_drift_threshold_px: 2.0
  use_astrometry: true
  enable_local_background_subtraction: false
  star_shift_radius_px: 60        # Äquatorial mit gutem Tracking
```

**Praxisprofil: M104 / Alt-Az / etwas staerkere Rotation / schlechtes Seeing:**
```yaml
registration:
  engine: triangle_star_matching
  auto_engine: true
  transform_model: affine
  enable_star_pair_fallback: true
  allow_rotation: true
  star_topk: 150
  star_min_inliers: 4
  star_inlier_tol_px: 4.0
  star_shift_radius_px: 200
  reject_outliers: true
  reject_cc_min_abs: 0.25
  use_astrometry: true
  enable_local_background_subtraction: true

global_metrics:
  adaptive_weights: true
  weight_exponent_scale: 1.3
  clamp: [-2.5, 2.5]
```

- Konkretes Vollbeispiel: [`m104.example.yaml`](https://github.com/jeamy/tile_compile/blob/master/tile_compile_cpp/examples/m104.example.yaml)
- Sinn dieses Profils: Multi-Anchor-Registration fuer Alt/Az-Rotation aktiv lassen, schwache Frames nicht wegwerfen, aber gute Frames bei der globalen Gewichtung klar bevorzugen.

---

## Globale Gewichtung (`global_metrics.*`)

Die globale Gewichtung nutzt aktuell die drei Metrikgewichte `background`, `noise`, `gradient` sowie `adaptive_weights`, `clamp` und `weight_exponent_scale`.

**Ausgewogen / Default-nah:**
```yaml
global_metrics:
  adaptive_weights: true
  weight_exponent_scale: 1.2
  weights:
    background: 0.40
    noise: 0.35
    gradient: 0.25
  clamp: [-3.0, 3.0]
```

**Staerkere Trennung guter/schlechter Frames:**
```yaml
global_metrics:
  adaptive_weights: true
  weight_exponent_scale: 1.3
  weights:
    background: 0.40
    noise: 0.35
    gradient: 0.25
  clamp: [-2.5, 2.5]
```

- Empfehlenswert fuer Sessions mit sichtbar wechselhaftem Seeing oder transparenzbedingten Qualitaetsschwankungen.
- Genau diese staerkere Trennung nutzt auch [`m104.example.yaml`](https://github.com/jeamy/tile_compile/blob/master/tile_compile_cpp/examples/m104.example.yaml).

**Weichere Gewichtung bei homogener Session:**
```yaml
global_metrics:
  adaptive_weights: false
  weight_exponent_scale: 0.8
```

---

## Lokale Qualitätskarten (`reconstruction.quality.pyramid.*`)

Die lokale Pixelgewichtung erfolgt ueber die Source-Quality-Pyramide
(Phase `SOURCE_QUALITY_MAPS`), nicht mehr ueber Tile-Metriken.

**Default-nah / robust:**
```yaml
reconstruction:
  quality:
    pyramid:
      scales: 4
      base_window_px: 4
      sharpness_weight: 0.6
      snr_weight: 0.4
      score_scale: 1.8
      artifact_sigma: 3.0
      max_artifact_fraction: 0.25
```

**Schaerfe staerker gewichten (Seeing-Sessions):**
```yaml
reconstruction:
  quality:
    pyramid:
      sharpness_weight: 0.7
      snr_weight: 0.3
      score_scale: 2.5
```

**SNR staerker gewichten (verrauschte, heterogene Sessions):**
```yaml
reconstruction:
  quality:
    pyramid:
      sharpness_weight: 0.4
      snr_weight: 0.6
```

**Mehr Raumfrequenzen erfassen:**
```yaml
reconstruction:
  quality:
    pyramid:
      scales: 6
      base_window_px: 4
```

---

## Laufzeitgrenzen (`runtime_limits.*`)

Das fruehere Reduced-/Emergency-Mode-Gating (`assumptions.*`,
`runtime_limits.allow_emergency_mode`) entfaellt; die Pipeline hat nur noch
einen Modus. Relevant bleiben die Laufzeitgrenzen:

```yaml
runtime_limits:
  parallel_workers: 8        # parallele Worker
  memory_budget: 4096        # MeB; deckelt zusaetzlich die Parallelitaet
  hard_abort_hours: 6.0      # hartes Laufzeitlimit
  acceleration_backend: auto # auto | cpu | opencv_cuda | opencv_opencl | opencl
```

---

## Kamera-spezifische Hinweise (`data.*`, `pcc.*`)

Der aktive Farbmodus-Schluessel ist `data.color_mode`, nicht mehr `data.mode`.

**OSC / Bayer-Kamera:**
```yaml
data:
  color_mode: OSC
  bayer_pattern: auto

pcc:
  enabled: true
  source: auto
  background_model: plane
  radii_mode: auto_fwhm
```

**Mono:**
```yaml
data:
  color_mode: MONO
```

---

## Performance-Optimierung (`runtime_limits.*`, `output.*`)

**Schneller Debug-Lauf:**
```yaml
linearity:
  max_frames: 4

runtime_limits:
  parallel_workers: 2
  memory_budget: 256
  acceleration_backend: cpu

output:
  write_registered_frames: false
```

**Produktion / hohe Qualitaet:**
```yaml
runtime_limits:
  parallel_workers: 8
  memory_budget: 4096
  acceleration_backend: auto
  hard_abort_hours: 6.0

output:
  write_registered_frames: true
```

**Speicher-limitiert:**
```yaml
runtime_limits:
  parallel_workers: 2
  memory_budget: 256
  acceleration_backend: cpu

output:
  write_registered_frames: false
```

---

## Zusammenfassung: Typische Setups

### DWARF II / Seestar S50

```yaml
data:
  color_mode: OSC
  bayer_pattern: auto

```

### DSLR auf EQ-Montierung

```yaml
data:
  color_mode: OSC
  bayer_pattern: auto

```

Fertige Profile im Repository:
- `tile_compile_cpp/examples/ic434.example.yaml`
- `tile_compile_cpp/examples/m31_background_gradient_balanced.example.yaml`

### Mono auf grossem Teleskop

```yaml
data:
  color_mode: MONO

```

## Raw Stack / Preprocessing

Raw Stack verwendet eine separate Preprocessing-Konfiguration ueber die GUI/API, nicht den normalen `tile_compile.yaml`-Hauptstrang. Eingabeordner und Kalibrierframes werden in der GUI 1:1 wie in `Input & Scan` gewaehlt.

### CFA/OSC mit Kalibrierung und Default-Postprocess

```json
{
  "mode": "linear_prestack",
  "lights_dir": "/data/session/lights",
  "bias_dir": "/data/session/bias",
  "darks_dir": "/data/session/darks",
  "flats_dir": "/data/session/flats",
  "input_mode": "cfa_osc",
  "raw_formats": "tile_compile",
  "bayer_pattern": "auto",
  "cfa_mode": "tile_compile",
  "calibration": {
    "use_bias": true,
    "use_dark": true,
    "use_flat": true,
    "dark_auto_select": true
  },
  "quality_filter": {
    "mode": "auto",
    "min_stars": 30,
    "max_fwhm_sigma": 2.0,
    "max_eccentricity": 0.65,
    "min_correlation": 0.75
  },
  "rejection": {
    "method": "sigma",
    "low": 3.0,
    "high": 3.0
  },
  "stacking": {
    "normalization": "addscale",
    "weighting": "quality"
  },
  "postprocess": {
    "astrometry": true,
    "bge": true,
    "pcc": true,
    "hypermetric_stretch": true
  },
  "hypermetric_stretch": {
    "require_successful_pcc": true,
    "mode": "ready_to_use",
    "sensor_profile": "rec709",
    "fallback_profile": "rec709",
    "target_bg": 0.15,
    "output_rgb": "stacked_rgb_hms.fits"
  },
  "report": {
    "detailed": true,
    "formats": ["json", "markdown", "html"]
  }
}
```

### Mono ohne Kalibrierframes

```json
{
  "mode": "linear_prestack",
  "lights_dir": "/data/session/mono_lights",
  "input_mode": "mono",
  "raw_formats": "tile_compile",
  "bayer_pattern": "auto",
  "mono_mode": "auto",
  "quality_filter": {
    "mode": "relaxed",
    "min_stars": 15,
    "min_correlation": 0.65
  },
  "stacking": {
    "normalization": "median",
    "weighting": "quality"
  },
  "postprocess": {
    "astrometry": true,
    "bge": true,
    "pcc": true,
    "hypermetric_stretch": true
  }
}
```

---

Diese Beispiele basieren jetzt auf den aktiven Parametern von Code und Schema (`v3.3.9`-Stand) und sind enger an die gepflegten Repository-Profile angelehnt.

Passen Sie die Werte an Ihre spezifische Hardware und Bedingungen an.

## Forward-Drizzle: Streaming und Speicherbudget (Entwicklungsstand 2026-09-05)

Der CPU-Coverage-/Uniform-Pfad verarbeitet Zielstreifen statt Vollbild-
Akkumulatoren pro Frame oder Worker. Die Preview bleibt standardmäßig deaktiviert.

| Parameter | Einheit, Bereich und Default | Verhalten |
|---|---|---|
| `reconstruction.drizzle.memory_budget_mb` | MiB, Ganzzahl >=0, Default 0 | 0 übernimmt `runtime_limits.memory_budget`; direkte Bibliotheksaufrufe verwenden 512 MiB. Berücksichtigt zurückgehaltene Ergebnisse/Masken, Quellbild samt temporärer Ladekopie, Arbeitsstreifen und Reserve. Verfügbarer Host-/cgroup-Speicher kann das Budget zusätzlich begrenzen. |
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

Für gemeinsame Uniform/Raw-Bibliotheksaufrufe empfiehlt sich ebenfalls `chunk_rows: 0`: Der Kandidatenspeicher wächst mit der Framezahl, deshalb können dieselben Bilddimensionen bei mehr Frames kleinere Streifen erfordern. Passt eine Zeile nicht, erfolgt eine frühe Ablehnung. Die Streaming-API vermeidet beide vollständigen Ausgaben im RAM; ihr Sink muss Streifen unmittelbar verbrauchen.

`reconstruction.diagnostics.persist_forward_drizzle_uniform_store` (Bool, Default `false`) ist unabhängig von der Preview. Aktiviert schreibt es ungeclippte Uniform-Ebenen per Streaming nach `artifacts/forward_drizzle_uniform_store/generation-…/`; `current.json` veröffentlicht die vollständige geprüfte Generation atomar. Das bestehende Drizzle-Budget umfasst zusätzlich 8 MiB für FITS/Metadaten und eine float-Zeile. Zu wenig RAM wird vor Quell-I/O abgewiesen; zu wenig freie Disk vor dem Ebenenschreiben. Ein Diagnosefehler lässt den Lauf weiterlaufen. Alte Generationen bleiben erhalten und belegen Disk; es gibt keine automatische Bereinigung. Der Store ist kein Resume-Phaseneinstieg. Leser müssen `current.json` gegen erwartete Quell-, Sampling- und Algorithmusidentität prüfen; alte flache Stores werden weder automatisch akzeptiert noch umgeschrieben.
Die geprüfte Vorgänger-Bibliotheks-API verwendet ein explizites Quellqualitätsbudget in MiB (Default 512 MiB). Große native Frames können wegen der konservativen Scratch-Schätzung früh abgelehnt werden; diese Prüfung nicht umgehen. Cachemanifeste identifizieren vorhandene normalisierte Float-Rohdateien und führen keine Kalibrierung aus. Store-Commit-Schema 2 bindet Cache- und Qualitätsplanhash.