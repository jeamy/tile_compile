# Praktische Konfigurationsbeispiele für tile_compile

**[🇬🇧 English Version](configuration_examples_practical_en.md)**

Dieser Leitfaden ergänzt die Konfigurationsreferenz mit praktischen Beispielen, Grenzbereichen und Anwendungsfällen basierend auf der Methodik v3.3.

## Änderungsstand (2026-03-03)

- `bge.fit.robust_loss` und `bge.fit.huber_delta` sind wieder als Benutzerparameter verfügbar.
- Neue BGE-Apply-Grenzwerte `bge.min_valid_sample_fraction_for_apply` und `bge.min_valid_samples_for_apply` dokumentiert.
- PCC-Beispiele wurden auf den aktuellen Parametersatz (ohne `pcc.method`) aktualisiert.
- `assumptions.pipeline_profile` (`practical|strict`) für explizite Methodik-Profile ergänzt.
- `registration.enable_star_pair_fallback` ergänzt, um die optionale nicht-normative Star-Pair-Stufe zu steuern.

**Strict-v3.3.6-Basis-Snippet:**

```yaml
assumptions:
  pipeline_profile: strict
  frames_reduced_threshold: 200

registration:
  engine: triangle_star_matching
  enable_star_pair_fallback: false
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
  autotune:
    enabled: false
    strategy: conservative
    max_evals: 24
    holdout_fraction: 0.25
    alpha_flatness: 0.25
    beta_roughness: 0.10
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
  min_valid_sample_fraction_for_apply: 0.28  # Toleranter fuer dichte Nebel-/Sternfelder
  min_valid_samples_for_apply: 96
  fit:
    method: modeled_mask_mesh  # Vordergrundbewusstes Mesh-Himmelsmodell
```

**Für schwache Gradienten (z.B. Mondlicht):**

```yaml
bge:
  enabled: true
  sample_quantile: 0.25  # Weniger konservativ
  min_valid_sample_fraction_for_apply: 0.30
  min_valid_samples_for_apply: 96
  fit:
    method: poly  # Einfacher für schwache Gradienten
    polynomial_order: 2
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

## Gemeinsamer Overlap nach PREWARP (`stacking.common_overlap_*`)

**Neue sinnvolle Standardwerte:**

```yaml
stacking:
  common_overlap_required_fraction: 1.0
  tile_common_valid_min_fraction: 0.90
```

- `common_overlap_required_fraction: 1.0` erzwingt die strikte Schnittmenge aller nutzbaren Frames.
- `tile_common_valid_min_fraction: 0.90` verhindert, dass Rand-Tiles mit wenig gemeinsamer Abdeckung lokale Metriken verzerren.

**Empfehlungen nach Setup:**

- **Alt/Az mit Feldrotation:** `1.0 / 0.90` (beibehalten, empfohlen)
- **EQ mit sehr stabiler Nachführung:** `1.0 / 0.85-0.90`
- **Nur wenn bewusst mehr Randfläche genutzt werden soll:** `0.95 / 0.80-0.85`

**Wichtig:** Niedrigere Werte können Dynamik-/Hintergrundstatistiken wieder durch unterschiedlich überdeckte Randbereiche verfälschen.

---

## Sichtbare Kachelgrenzen reduzieren (`stacking.tile_seam_harmonization.*`)

Der Parameterblock arbeitet in `TILE_RECONSTRUCTION` overlap-basiert:

- benachbarte Tiles werden in ihren realen Überlappungen verglichen
- es werden nur glatte, dunkle Hintergrundpixel verwendet
- daraus wird ein global konsistentes Offset-/Scale-Feld über alle Tiles gelöst

**Guter Startpunkt bei sichtbarer Kachelstruktur:**

```yaml
stacking:
  tile_seam_harmonization:
    enabled: true
    strength: 0.75
    sample_quantile: 0.30
    gradient_quantile: 0.70
    min_sample_fraction: 0.05
    min_samples: 64
    scale_floor_factor: 0.50
    scale_ceil_factor: 2.00
```

**Für klar sichtbare Tile-Nähte im schwachen Hintergrund:**

```yaml
stacking:
  tile_seam_harmonization:
    enabled: true
    strength: 0.90
    sample_quantile: 0.20
    gradient_quantile: 0.50
    min_sample_fraction: 0.03
    min_samples: 48
    scale_floor_factor: 0.80
    scale_ceil_factor: 1.25
```

- Mehr `strength` verstärkt die global gelöste Seam-Korrektur.
- Niedrigere `sample_quantile` und `gradient_quantile` machen die Overlap-Maske konservativer.
- Engere `scale_*`-Grenzen verhindern aggressive tileweise Kontrastsprünge.

**Für große diffuse Objekte (M31, M42, IC434) mit Nebel in den Overlaps:**

```yaml
stacking:
  tile_seam_harmonization:
    enabled: true
    strength: 0.85
    sample_quantile: 0.12
    gradient_quantile: 0.35
    min_sample_fraction: 0.02
    min_samples: 32
    scale_floor_factor: 0.90
    scale_ceil_factor: 1.10
```

- Diese Variante ist strenger bei der Auswahl sauberer Hintergrundpixel.
- Sinnvoll, wenn Nebel- oder Sternstruktur die Overlap-Schätzung sonst kontaminiert.

**Wenn die Kachelstruktur danach immer noch kaum reagiert:**

- Ursache ist wahrscheinlich nicht primär ein Level-/Scale-Seam-Problem.
- Dann eher prüfen:
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

## Tile-Größe (`tile.size`)

**Default:** `256`  
**Bereich:** `64` - `512`  
**Methodik-Vorgabe:** Muss groß genug sein für lokale Schärfe-Metriken, klein genug für räumliche Auflösung

### Anwendungsfälle:

**Kurze Brennweite (< 200mm), gutes Seeing:**
```yaml
tile:
  size: 128
  overlap: 32
```
- Kleinere Tiles erfassen lokale Qualitätsunterschiede besser
- Bei gutem Seeing sind Strukturen feiner verteilt
- Beispiel: DWARF II (f=100mm), Seestar S50 (f=250mm)

**Mittlere Brennweite (200-800mm), normales Seeing:**
```yaml
tile:
  size: 256  # Default
  overlap: 64
```
- Standard für die meisten Anwendungen
- Guter Kompromiss zwischen Auflösung und Rechenzeit
- Beispiel: 80mm Refraktor, 8" SCT

**Lange Brennweite (> 800mm), schlechtes Seeing:**
```yaml
tile:
  size: 384
  overlap: 96
```
- Größere Tiles vermeiden Tile-Artefakte bei großen Strukturen
- Bei schlechtem Seeing sind lokale Qualitätsunterschiede gröber
- Beispiel: 12" SCT (f=2000mm), große Refraktoren

**Alt/Az-Montierung mit Feldrotation:**
```yaml
tile:
  size: 320
  overlap: 80
  min_valid_fraction: 0.6  # Toleranter bei Rotation
```
- Größere Tiles kompensieren Rotationseffekte besser
- Höherer Overlap für sanftere Übergänge

---

## Registrierung (`registration.*`)

### `registration.method`

**Default:** `"triangle_star_matching"`  
**Alternativen:** `star_similarity`, `hybrid_phase_ecc`, `robust_phase_ecc`

**Sternenreiche Felder (> 50 Sterne):**
```yaml
registration:
  method: triangle_star_matching
  min_stars: 15
  max_shift_px: 50
  max_rotation_deg: 5.0
```
- Triangle-Matching ist robust und präzise
- Funktioniert auch bei Rotation und Translation

**Sternenarme Felder (< 20 Sterne), Nebel:**
```yaml
registration:
  method: robust_phase_ecc
  fallback_to_identity: true
  identity_correlation_threshold: 0.3
```
- Phase-Korrelation nutzt Gradientenstrukturen
- Funktioniert auch bei diffusen Nebeln
- Fallback verhindert Abbruch bei schwierigen Frames

**Alt/Az mit Feldrotation (aktuell):**
```yaml
registration:
  engine: triangle_star_matching
  allow_rotation: true  # ZWINGEND bei Alt/Az nahe Pol
  star_topk: 150  # Mehr Sterne für robustere Lösung
  star_min_inliers: 4
  star_inlier_tol_px: 4.0  # Toleranter bei Drift/Feldrotation
  star_dist_bin_px: 5.0
  
  reject_outliers: true
  reject_cc_min_abs: 0.30
  reject_cc_mad_multiplier: 4.0
  reject_shift_px_min: 100.0
  reject_shift_median_multiplier: 5.0
  reject_scale_min: 0.92
  reject_scale_max: 1.08

  # Frames mit fehlgeschlagener direkter Registrierung werden
  # über ein Polynomial-Feldrotationsmodell vorhergesagt,
  # damit alle Frames im Stack bleiben.
```
**Hinweis:** Dieses Verhalten entspricht den aktuellen Beispielprofilen für Alt/Az.

### `registration.max_shift_px`

**Default:** `50`  
**Bereich:** `10` - `200`

**Gut nachgeführt (Äquatorial):**
```yaml
registration:
  max_shift_px: 30
```
- Geringe Drift erwartet
- Striktere Limits verhindern Fehlregistrierungen

**Alt/Az ohne Feldrotations-Kompensation:**
```yaml
registration:
  max_shift_px: 100
```
- Höhere Drift durch Feldrotation
- Mehr Toleranz nötig

**Smart Telescope (DWARF, Seestar) - kurze Belichtungen:**
```yaml
registration:
  max_shift_px: 80
  max_rotation_deg: 8.0
```
- Moderate Drift durch Tracking-Ungenauigkeiten
- Rotation durch Alt/Az-Montierung

---

## Globale Metriken (`global_metrics.*`)

### `global_metrics.fwhm_percentile`

**Default:** `0.5` (Median)  
**Bereich:** `0.1` - `0.9`

**Gutes Seeing (FWHM < 2.5"):**
```yaml
global_metrics:
  fwhm_percentile: 0.3  # Nutze beste 30% der Sterne
  fwhm_outlier_sigma: 2.5
```
- Bei gutem Seeing sind die besten Sterne sehr scharf
- Niedrigerer Percentile fokussiert auf Spitzenwerte

**Schlechtes Seeing (FWHM > 4"):**
```yaml
global_metrics:
  fwhm_percentile: 0.7  # Nutze Mehrheit der Sterne
  fwhm_outlier_sigma: 3.5
```
- Bei schlechtem Seeing große Streuung
- Höherer Percentile vermeidet Ausreißer-Dominanz

**Turbulentes Seeing (stark variabel):**
```yaml
global_metrics:
  fwhm_percentile: 0.5
  fwhm_outlier_sigma: 4.0  # Sehr tolerant
  use_robust_background: true
```

---

## Lokale Metriken (`local_metrics.*`)

### `local_metrics.sharpness_method`

**Default:** `"gradient_energy"`  
**Alternativen:** `laplacian_variance`, `tenengrad`

**Hochauflösende Daten (Sampling < 1"/px):**
```yaml
local_metrics:
  sharpness_method: tenengrad
  sharpness_kernel_size: 5
```
- Tenengrad ist sensitiver für feine Details
- Kleinerer Kernel für hohe Auflösung

**Niedrigauflösende Daten (Sampling > 3"/px):**
```yaml
local_metrics:
  sharpness_method: gradient_energy
  sharpness_kernel_size: 7
```
- Gradient Energy robuster bei grobem Sampling
- Größerer Kernel für niedrige Auflösung

**Smart Telescopes (DWARF: 5.57"/px, Seestar: 3.97"/px):**
```yaml
local_metrics:
  sharpness_method: gradient_energy
  sharpness_kernel_size: 5
  contrast_percentile: 0.7
```

---

## Rekonstruktion (`reconstruction.*`)

### `reconstruction.ola_window`

**Default:** `"hann"`  
**Alternativen:** `bartlett`, `blackman`, `hamming`

**Viele Frames (N > 500), gutes SNR:**
```yaml
reconstruction:
  ola_window: hann
  ola_normalize_per_tile: true
```
- Hann-Fenster: guter Kompromiss
- Normalisierung pro Tile bei hohem SNR sicher

**Wenige Frames (50 < N < 200), niedriges SNR:**
```yaml
reconstruction:
  ola_window: blackman  # Weichere Übergänge
  ola_normalize_per_tile: false
  sigma_clip_threshold: 4.0  # Toleranter
```
- Blackman reduziert Tile-Kanten
- Keine Tile-Normalisierung vermeidet Rausch-Verstärkung

**Emergency Mode (N < 50):**
```yaml
reconstruction:
  ola_window: blackman
  ola_normalize_per_tile: false
  sigma_clip_threshold: 5.0
  min_frames_per_pixel: 3  # Sehr niedrig
```

---

## Frame-Anzahl und Modi

**Methodik v3.2.2 Vorgaben:**
- **Full Mode:** N ≥ 200 (Clustering + Synthetic Frames aktiv)
- **Reduced Mode:** 50 ≤ N < 200 (Clustering deaktiviert)
- **Emergency Mode:** N < 50 (nur mit `runtime.allow_emergency_mode: true`)

### Full Mode (N ≥ 200)

```yaml
runtime:
  min_frames: 200
  allow_reduced_mode: false
  
synthetic:
  enabled: true
  min_cluster_size: 20
  max_clusters: 10
```

### Reduced Mode (50 ≤ N < 200)

```yaml
runtime:
  min_frames: 50
  allow_reduced_mode: true
  
synthetic:
  enabled: false  # Automatisch deaktiviert
```

### Emergency Mode (N < 50) - Nur für Tests!

```yaml
runtime:
  min_frames: 10
  allow_emergency_mode: true  # WARNUNG!
  
tile:
  size: 384  # Größere Tiles
  min_valid_fraction: 0.4  # Sehr tolerant
  
reconstruction:
  sigma_clip_threshold: 5.0
  min_frames_per_pixel: 2
```

**⚠️ Warnung:** Emergency Mode ist nicht für Produktion geeignet!

---

## Brennweiten-spezifische Empfehlungen

### Kurze Brennweite (< 200mm)

**Beispiel: DWARF II (100mm f/4.4), Seestar S50 (250mm f/5)**

```yaml
tile:
  size: 128
  overlap: 32
  
registration:
  method: triangle_star_matching
  min_stars: 20  # Viele Sterne im Feld
  max_shift_px: 60
  
local_metrics:
  sharpness_kernel_size: 5
  contrast_percentile: 0.7
```

### Mittlere Brennweite (200-800mm)

**Beispiel: 80mm Refraktor (480mm f/6), 8" SCT (2000mm f/10)**

```yaml
tile:
  size: 256
  overlap: 64
  
registration:
  method: triangle_star_matching
  min_stars: 10
  max_shift_px: 40
  
local_metrics:
  sharpness_kernel_size: 5
  contrast_percentile: 0.5
```

### Lange Brennweite (> 800mm)

**Beispiel: 12" SCT (3000mm f/10), große Refraktoren**

```yaml
tile:
  size: 384
  overlap: 96
  
registration:
  method: triangle_star_matching
  min_stars: 5  # Weniger Sterne im Feld
  max_shift_px: 30  # Präzises Guiding erwartet
  max_rotation_deg: 2.0
  
local_metrics:
  sharpness_kernel_size: 7
  contrast_percentile: 0.3
```

---

## Seeing-Bedingungen

### Exzellentes Seeing (FWHM < 2")

```yaml
global_metrics:
  fwhm_percentile: 0.2
  fwhm_outlier_sigma: 2.0
  
local_metrics:
  sharpness_percentile: 0.3
  
reconstruction:
  quality_weight_exponent: 2.0  # Stärkere Gewichtung
```

### Gutes Seeing (FWHM 2-3")

```yaml
global_metrics:
  fwhm_percentile: 0.4
  fwhm_outlier_sigma: 2.5
  
local_metrics:
  sharpness_percentile: 0.5
  
reconstruction:
  quality_weight_exponent: 1.5
```

### Moderates Seeing (FWHM 3-4")

```yaml
global_metrics:
  fwhm_percentile: 0.5
  fwhm_outlier_sigma: 3.0
  
local_metrics:
  sharpness_percentile: 0.6
  
reconstruction:
  quality_weight_exponent: 1.0  # Default
```

### Schlechtes Seeing (FWHM > 4")

```yaml
global_metrics:
  fwhm_percentile: 0.7
  fwhm_outlier_sigma: 3.5
  use_robust_background: true
  
local_metrics:
  sharpness_percentile: 0.7
  
reconstruction:
  quality_weight_exponent: 0.8  # Schwächere Gewichtung
  sigma_clip_threshold: 4.0
```

---

## Montierungs-spezifische Einstellungen

### Äquatoriale Montierung (gut nachgeführt)

```yaml
registration:
  method: triangle_star_matching
  max_shift_px: 30
  max_rotation_deg: 2.0
  allow_reflection: false
  
tile:
  min_valid_fraction: 0.8  # Strikt
```

### Alt/Az ohne Derotator

```yaml
registration:
  method: triangle_star_matching
  max_shift_px: 100
  max_rotation_deg: 15.0
  trail_endpoint_enabled: true
  
tile:
  size: 320  # Größer wegen Rotation
  overlap: 80
  min_valid_fraction: 0.6  # Toleranter
```

### Alt/Az mit Derotator (DWARF, Seestar)

```yaml
registration:
  method: triangle_star_matching
  max_shift_px: 60
  max_rotation_deg: 8.0
  
tile:
  size: 256
  overlap: 64
  min_valid_fraction: 0.7
```

---

## Kamera-spezifische Einstellungen

### OSC (One-Shot Color)

```yaml
data:
  mode: OSC
  bayer_pattern: RGGB  # Kamera-abhängig!
  
debayer:
  enabled: true
  method: bilinear
  preserve_bayer_normalization: true
  
pcc:
  enabled: true
  source: auto
  background_model: plane
  radii_mode: auto_fwhm
```

### Monochrome

```yaml
data:
  mode: MONO
  
debayer:
  enabled: false
  
# Kein PCC bei Mono (nur bei RGB-Komposit)
```

---

## Performance-Optimierung

### Schneller Test-Durchlauf

```yaml
pipeline:
  mode: test
  max_frames: 50
  
tile:
  size: 256
  
output:
  write_registered_frames: false
  write_tile_weights: false
```

### Produktion (maximale Qualität)

```yaml
pipeline:
  mode: production
  
tile:
  size: 256
  overlap: 64
  
reconstruction:
  ola_normalize_per_tile: true
  
output:
  write_registered_frames: true
  write_tile_weights: true
  write_quality_maps: true
```

### Speicher-limitiert

```yaml
runtime:
  max_memory_gb: 8.0
  use_disk_cache: true
  
tile:
  size: 192  # Kleiner = weniger RAM
  
output:
  write_registered_frames: false
```

---

## Zusammenfassung: Typische Setups

### DWARF II / Seestar S50

```yaml
data:
  mode: OSC
  bayer_pattern: RGGB
  
tile:
  size: 128
  overlap: 32
  
registration:
  method: triangle_star_matching
  max_shift_px: 80
  max_rotation_deg: 8.0
  
global_metrics:
  fwhm_percentile: 0.5
  
local_metrics:
  sharpness_method: gradient_energy
  sharpness_kernel_size: 5
  
reconstruction:
  ola_window: hann
  quality_weight_exponent: 1.0
  
debayer:
  enabled: true
  method: bilinear
  
pcc:
  enabled: true
  source: auto
```

### DSLR auf Äquatorialer Montierung

```yaml
data:
  mode: OSC
  bayer_pattern: RGGB  # Canon meist RGGB, Nikon meist GBRG
  
tile:
  size: 256
  overlap: 64
  
registration:
  method: triangle_star_matching
  max_shift_px: 40
  max_rotation_deg: 3.0
  
global_metrics:
  fwhm_percentile: 0.4
  
reconstruction:
  quality_weight_exponent: 1.5
  
debayer:
  enabled: true
  method: bilinear
  
pcc:
  enabled: true
```

Fertiges Profil dazu:
- `tile_compile_cpp/examples/tile_compile.canon_equatorial_balanced.example.yaml`

### Mono CCD auf großem Teleskop

```yaml
data:
  mode: MONO
  
tile:
  size: 384
  overlap: 96
  
registration:
  method: triangle_star_matching
  min_stars: 5
  max_shift_px: 20
  max_rotation_deg: 1.0
  
global_metrics:
  fwhm_percentile: 0.3
  
local_metrics:
  sharpness_kernel_size: 7
  
reconstruction:
  quality_weight_exponent: 2.0
```

---

Diese Beispiele basieren auf:
- Methodik v3.2.2 Vorgaben (Linearität, keine Frame-Selektion, Tile-basierte Rekonstruktion)
- Praktischen Erfahrungen mit verschiedenen Setups
- Physikalischen Constraints (Seeing, Brennweite, Montierung)

Passen Sie die Werte an Ihre spezifische Hardware und Bedingungen an!
