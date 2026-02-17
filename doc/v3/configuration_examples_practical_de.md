# Praktische Konfigurationsbeispiele für tile_compile

**[🇬🇧 English Version](configuration_examples_practical_en.md)**

Dieser Leitfaden ergänzt die Konfigurationsreferenz mit praktischen Beispielen, Grenzbereichen und Anwendungsfällen basierend auf der Methodik v3.2.2.

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

**Alt/Az mit Feldrotation:**
```yaml
registration:
  method: triangle_star_matching
  max_rotation_deg: 15.0  # Höhere Toleranz
  allow_reflection: false
  trail_endpoint_enabled: true  # Für Startrails
```

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
  method: proportion
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
