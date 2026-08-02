# Praktische Konfigurationsbeispiele für tile_compile

**[🇬🇧 English Version](configuration_examples_practical_en.md)**

Dieser Leitfaden ergänzt die Konfigurationsreferenz mit praktischen Beispielen, Grenzbereichen und Anwendungsfällen basierend auf der Methodik v3.3.

## Änderungsstand (2026-07-18)

- AQMH (`aqmh.*`) vollständig dokumentiert und mit praktischen Beispielen versehen.
- HyperMetric Stretch (`hypermetric_stretch.*`) ist als optionale Post-PCC-Phase mit `ready_to_use`- und `scientific`-Modus dokumentiert.
- `bge.fit.robust_loss` und `bge.fit.huber_delta` sind wieder als Benutzerparameter verfügbar.
- Neue BGE-Apply-Grenzwerte `bge.min_valid_sample_fraction_for_apply` und `bge.min_valid_samples_for_apply` dokumentiert.
- PCC-Beispiele wurden auf den aktuellen Parametersatz (ohne `pcc.method`) aktualisiert.
- Assumptions-Beispiele auf die aktiven Runtime-Felder (`frames_min`, `frames_reduced_threshold`, Reduced-Mode-Steuerung) abgeglichen.
- `registration.enable_star_pair_fallback` ergänzt, um die optionale nicht-normative Star-Pair-Stufe zu steuern.
- `bge.tile_weight_lambda_structure` auf den aktuellen Default `1.0` abgeglichen.
- `stacking.common_overlap_required_fraction` und `stacking.tile_common_valid_min_fraction` mit der aktuellen strikten Default-Semantik `1.0 / 1.0` dokumentiert.
- Das Basis-Snippet wurde auf das strikte `v3.3.9`-Profil aktualisiert.
- AQMH-Beispiele auf die objektklassenunabhängige v0.2.1-Baseline abgeglichen: begrenzte globale Sigmoid-Gewichte, `resolution_divisor: 2`, `dtype: uint16`, asymmetrisches Sigma-Clipping `2.0 / 1.5` mit vier Iterationen und duales Validierungsgate gegen Uniformkontrolle und rohe AQMH-Baseline.

**Strict-v3.3.9-Basis-Snippet:**

```yaml
assumptions:
  frames_min: 50
  frames_reduced_threshold: 200

registration:
  engine: triangle_star_matching
  enable_star_pair_fallback: false

stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 1.0
```

---

## AQMH (Adaptive Quality Map Harvesting) - Experimentell

**Wann aktivieren:**
- Hochwertige Sessions mit stark variierender Frame-Qualität (Seeing, Wolken)
- Wenn Tile-Seams oder OLA-Artefakte sichtbar sind
- Als Alternative zur klassischen Tile-OLA-Rekonstruktion

**Standard-Konfiguration (empfohlen):**

```yaml
aqmh:
  enabled: true
  pyramid:
    scales: 4
    base_window_px: 4
    w_sharp: 0.6        # Schärfe-Gewicht im Qualitätsindex
    w_snr: 0.4          # SNR-Gewicht im Qualitätsindex
    score_scale: 1.8    # Selektivität der lokalen AQMH-Qualitätskarte
    k_artifact: 3.0     # MAD-Multiplikator für Artefakt-Erkennung
    frac_artifact_max: 0.25  # max. Artefaktanteil pro Fenster
  storage:
    resolution_divisor: 2   # robuster Standard; 1 für Cherry-Pick/Referenzläufe
    dtype: uint16           # float32 für Cherry-Pick/Referenzläufe
    max_resident_maps: 2
  global_quality:
    g_floor: 0.03
    g_w_sharp: 0.55
    g_w_snr: 0.30
    g_w_background_penalty: 0.25
    g_k_scale: 1.5         # begrenzte Sigmoid-Temperatur
  reconstruction:
    delete_prewarped_cache_after_run: true  # false fuer Resume; Cache liegt unter cache/prewarped_frames
    prewarp_interpolation: linear            # konservativer Prewarp; cubic/lanczos4 gezielt auf Schaerfe testen
    clip_sigma: 2.0
    clip_sigma_low: 2.0
    clip_sigma_high: 2.0
    clip_iterations: 4
    min_fraction: 0.4
    min_n_eff: 2.0
    registration_weight_guard: true
    registration_weight_floor: 0.30
    registration_sequential_factor: 0.92
    registration_predicted_factor: 0.50
    structure_mask_low_q: 0.40
    structure_mask_high_q: 0.90
    structure_mask_blur_sigma_px: 4.0
    debayer_first: true                    # DF-AQMH: Pre-Debayer vor Prewarp, RGB-Reconstruction
    pre_debayer_method: edge_aware         # edge_aware | bilinear | nearest
    rgb_q_map_mode: shared_luma            # shared_luma | per_channel
    rgb_memory_strategy: sequential        # sequential | parallel
  cherry_pick:
    enabled: false
  validation:
    max_seam_score_regression: 0.05
    max_fwhm_regression: 0.02
    max_background_rms_regression: 0.05
    max_tail11_abs_regression: 0.10
    max_elongation_regression: 0.08
  diagnostics:
    level: full
    tau_artifact: 0.20
    q_region: 0.75
    r_morph_canvas_px: 6
    binary_block_size_px: 64
```

**Toleranter gegenüber Artefakten (Satelliten, Wolken):**

```yaml
aqmh:
  enabled: true
  pyramid:
    k_artifact: 5.0
    frac_artifact_max: 0.35
```

**Cherry-Pick Auto-Reject (meiste Frames behalten, nur Extremfälle verwerfen):**

```yaml
aqmh:
  enabled: true
  storage:
    resolution_divisor: 1
    dtype: float32
  cherry_pick:
    enabled: true
    mode: auto_reject
    k_min_required: 20  # Lauf-Gate und Untergrenze Samples pro Pixel
    reject_below_best_fraction: 0.25
    min_keep_fraction: 0.90
```

**Speichersparend (große Sessions, RAM-knapp):**

```yaml
aqmh:
  enabled: true
  storage:
    resolution_divisor: 4   # Viertel-Auflösung für Maps
    dtype: uint8            # 8-bit Quantisierung
    max_resident_maps: 2
```

**AQMH deaktivieren (zurück zu klassischer Tile-OLA):**

```yaml
aqmh:
  enabled: false
```

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
  k_max: 3.2
```

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
  shadow_color_floor: 1.0
```

`scientific` überspringt das finale Ready-to-Use-Scaling und den Soft Clip und erlaubt `linear_expansion`. Sinnvoll, wenn ein weniger poliertes, kontrollierteres Stretch-Ergebnis für weitere Bearbeitung gewünscht ist.

---

## Gemeinsamer Overlap nach PREWARP (`stacking.common_overlap_*`)

**Aktuelle sinnvolle Standardwerte:**

```yaml
stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 1.0
```

- `common_overlap_required_fraction: 1.0` erzwingt die strikte Schnittmenge aller nutzbaren Frames.
- `tile_common_valid_min_fraction: 1.0` bedeutet: Ein Tile ist nur dann gueltig, wenn seine komplette Flaeche innerhalb von `COMMON_OVERLAP` liegt.
- Die Tile-Quote wird ueber die volle Tile-Flaeche berechnet, nicht nur ueber den zufaellig im Canvas liegenden Rest.

**Empfehlungen nach Setup:**

- **Alt/Az mit Feldrotation:** `1.0 / 1.0` (empfohlen)
- **EQ mit sehr stabiler Nachfuehrung:** `1.0 / 1.0` (empfohlen, wenn Randbias vermieden werden soll)
- **Nur wenn bewusst mehr Randflaeche zugelassen werden soll:** z. B. `0.98 / 0.95` oder `0.95 / 0.90`

**Wichtig:** Niedrigere Werte lassen wieder teilweise ueberdeckte Randpixel und Rand-Tiles in lokale Metriken, BGE/PCC und Hintergrundstatistiken einsickern.

---

## Sichtbare Kachelgrenzen diagnostizieren (Artefakte)

Es gibt aktuell keinen dedizierten Seam-Korrektur-Parameterblock.

Wenn sichtbare Kachelstruktur auftritt, prüfe nach dem Run `artifacts/tile_reconstruction.json`, insbesondere:

- `tile_boundary_raw_pair_mean_abs_diff_p95`
- `tile_boundary_normalized_pair_mean_abs_diff_p95`
- `tile_boundary_pair_mean_abs_diff_p95`
- `tile_boundary_post_background_delta_p95_abs`
- `tile_boundary_post_snr_delta_p95_abs`
- `tile_boundary_top_pairs`
- `tile_norm_scale`

Interpretation:

- hohe `tile_boundary_raw_pair_mean_abs_diff_*`-Werte bedeuten, dass sich benachbarte Tiles schon vor der optionalen Tile-Normalisierung deutlich unterscheiden
- wenn `tile_boundary_normalized_pair_mean_abs_diff_*` deutlich höher liegt als der Raw-Wert, verschärft die Tile-Normalisierung die Naht
- hohe `tile_boundary_post_background_delta_*`-Werte deuten auf tileweisen Hintergrunddrift
- hohe `tile_boundary_post_snr_delta_*`-Werte sprechen für divergierende Support-/Qualitätslage benachbarter Tiles
- `tile_boundary_top_pairs` listet die problematischsten Nachbarpaare mit Tile-Indizes, Grid-Positionen, Valid-Counts, Fallback-Flags und Post-Metriken
- über `tile_norm_scale` und `tile_norm_bg_*` an genau diesen Tile-Indizes lässt sich prüfen, ob die Normierung die Tile-Population auseinanderzieht

Wenn die Kachelstruktur sichtbar ist und diese Boundary-Diagnostik ebenfalls hoch ausfällt, zuerst prüfen:

- `tile.overlap_fraction`
- `tile_denoise.*`
- `stacking.output_stretch`
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

Optional kann zusätzlich eine sehr konservative Post-Stack-Kosmetik aktiv bleiben:

```yaml
stacking:
  cosmetic_correction: true
  cosmetic_correction_sigma: 10.0
```

---

## Audit-Hinweis zu Legacy-Parametern

Im Rahmen des Code-/Schema-Abgleichs wurden mehrere veraltete Beispielparameter aus diesem Leitfaden entfernt oder ersetzt.

Nicht mehr aktive Legacy-Keys waren unter anderem:
- `tile.size`, `tile.overlap`, `tile.min_valid_fraction`
- `registration.method`, `registration.max_rotation_deg`, `registration.fallback_to_identity`, `registration.identity_correlation_threshold`, `registration.trail_endpoint_enabled`
- `global_metrics.fwhm_percentile`, `global_metrics.fwhm_outlier_sigma`, `global_metrics.use_robust_background`
- `local_metrics.sharpness_method`, `local_metrics.sharpness_kernel_size`, `local_metrics.sharpness_percentile`, `local_metrics.contrast_percentile`
- der komplette alte Block `reconstruction.*`
- `runtime.min_frames`, `runtime.allow_reduced_mode`, `runtime.max_memory_gb`, `runtime.use_disk_cache`
- `data.mode`
- `output.write_tile_weights`, `output.write_quality_maps`

Die folgenden Praxisbeispiele verwenden nur noch aktuell aktive Parameter aus Code und Schema.

---

## Tile-Erzeugung (`tile.*`)

Die Tile-Erzeugung ist im aktuellen Runner **adaptiv**. Statt eines festen `tile.size` werden die Tiles aus `tile.size_factor`, `tile.min_size`, `tile.max_divisor` und `tile.overlap_fraction` abgeleitet.

**Kurze Brennweite / gutes Seeing:**
```yaml
tile:
  size_factor: 24
  min_size: 48
  max_divisor: 6
  overlap_fraction: 0.30
```

**Allround / Default-nah:**
```yaml
tile:
  size_factor: 32
  min_size: 64
  max_divisor: 6
  overlap_fraction: 0.25
```

**Lange Brennweite / grobe Strukturen / schlechtes Seeing:**
```yaml
tile:
  size_factor: 40
  min_size: 96
  max_divisor: 5
  overlap_fraction: 0.30
```

**Alt/Az mit striktem Randverhalten:**
```yaml
tile:
  size_factor: 24
  min_size: 48
  max_divisor: 6
  overlap_fraction: 0.30

stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 1.0
```

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
  max_shift_px: 80
  reject_outliers: true
  reject_cc_min_abs: 0.25
  reject_shift_px_min: 100.0
  reject_shift_median_multiplier: 5.0
  reject_scale_min: 0.92
  reject_scale_max: 1.08
  # Neue Parameter (v2.0)
  max_blind_chain_depth: 0        # auto (N/10) für Wolkenblöcke
  blind_chain_strong_anchor_cc: 0.08
  blind_chain_drift_threshold_px: 2.0
  use_astrometry: true            # Astrometrische Rescue bei Bedarf
  enable_local_background_subtraction: false
  star_shift_radius_px: 200       # Alt/Az: 200-400, Äquatorial: 60
```

**Sternenarm / Nebel / wolkige Daten:**
```yaml
registration:
  engine: robust_phase_ecc
  allow_rotation: true
  max_shift_px: 80
  reject_outliers: true
  # Neue Parameter (v2.0) — bei starken Gradienten
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
  max_shift_px: 30
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

## Lokale Gewichtung (`local_metrics.*`)

Statt alter Schärfe-Kernel-/Percentile-Schalter sind aktuell die Exponent-Skala `k_local`, die Nachbarschafts-Normierung, die räumliche Regularisierung und die STAR-/STRUCTURE-Gewichte relevant.

**Default-nah / robust:**
```yaml
local_metrics:
  clamp: [-3.0, 3.0]
  k_local: 1.0
  neighborhood_normalization:
    enabled: true
    radius: 1
    blend: 0.5
  spatial_regularization:
    enabled: true
    lambda: 0.35
    passes: 1
```

**Staerkere lokale Differenzierung:**
```yaml
local_metrics:
  k_local: 1.5
```

**Weichere lokale Gewichtung:**
```yaml
local_metrics:
  k_local: 0.7
```

**Sternfelder priorisieren:**
```yaml
local_metrics:
  star_mode:
    weights:
      fwhm: 0.7
      roundness: 0.2
      contrast: 0.1
```

**Diffuse Struktur priorisieren:**
```yaml
local_metrics:
  structure_mode:
    metric_weight: 0.7
    background_weight: 0.3
```

---

## Frame-Anzahl und Modi (`assumptions.*`, `synthetic.*`, `runtime_limits.*`)

Die Umschaltung erfolgt aktuell ueber `assumptions.frames_min` und `assumptions.frames_reduced_threshold`, nicht mehr ueber einen alten `runtime.min_frames`-Block.

**Full Mode (N >= 200):**
```yaml
assumptions:
  frames_min: 50
  frames_reduced_threshold: 200
  reduced_mode_skip_clustering: false

synthetic:
  weighting: tile_weighted
  frames_min: 4
  frames_max: 20
  clustering:
    mode: kmeans
    cluster_count_range: [3, 12]
```

**Reduced Mode (50 <= N < 200):**
```yaml
assumptions:
  frames_min: 50
  frames_reduced_threshold: 200
  reduced_mode_skip_clustering: true
  reduced_mode_cluster_range: [5, 10]
```

**Emergency Mode (nur bewusst):**
```yaml
runtime_limits:
  allow_emergency_mode: true

stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 1.0
  sigma_clip:
    sigma_low: 2.5
    sigma_high: 2.5
    max_iters: 2
```

**Warnung:** `allow_emergency_mode` ist fuer Test-/Rettungslaeufe gedacht, nicht fuer normale Produktion.

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

## Performance-Optimierung (`pipeline.*`, `runtime_limits.*`, `output.*`)

**Schneller Debug-Lauf:**
```yaml
pipeline:
  mode: test

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
pipeline:
  mode: production

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

tile:
  size_factor: 24
  min_size: 48
  max_divisor: 6
  overlap_fraction: 0.30

registration:
  engine: triangle_star_matching
  enable_star_pair_fallback: true
  allow_rotation: true
  max_shift_px: 80
  star_shift_radius_px: 200       # Alt/Az: Shift-Suchradius für mehrstündige Sessions

stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 1.0
  per_frame_cosmetic_correction: true
  per_frame_cosmetic_correction_sigma: 2.5

pcc:
  enabled: true
  source: auto
```

### DSLR auf EQ-Montierung

```yaml
data:
  color_mode: OSC
  bayer_pattern: auto

tile:
  size_factor: 36
  min_size: 96
  max_divisor: 6
  overlap_fraction: 0.35

registration:
  engine: triangle_star_matching
  allow_rotation: true
  max_shift_px: 40

global_metrics:
  adaptive_weights: false
  weight_exponent_scale: 1.0

pcc:
  enabled: true
```

Fertige Profile im Repository:
- `tile_compile_cpp/examples/ic434.example.yaml`
- `tile_compile_cpp/examples/m31_background_gradient_balanced.example.yaml`

### Mono auf grossem Teleskop

```yaml
data:
  color_mode: MONO

tile:
  size_factor: 40
  min_size: 96
  max_divisor: 5
  overlap_fraction: 0.30

registration:
  engine: triangle_star_matching
  allow_rotation: true
  max_shift_px: 20

local_metrics:
  k_local: 1.2
  structure_mode:
    metric_weight: 0.7
    background_weight: 0.3
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
