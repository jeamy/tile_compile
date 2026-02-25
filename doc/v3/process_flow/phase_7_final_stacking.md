# COMMON_OVERLAP + SYNTHETIC_FRAMES + STACKING + VALIDATION + DEBAYER + ASTROMETRY + PCC — Finales Stacking und Output

> **C++ Implementierung:** `runner_pipeline.cpp`, `runner_phase_registration.cpp` (aktueller v3.2 Stand)
> **Phase-Enums:** `COMMON_OVERLAP` (7), `SYNTHETIC_FRAMES` (11), `STACKING` (12), *Validation* (kein Enum), `DEBAYER` (13), `ASTROMETRY` (14), `PCC` (15), `DONE` (16)

## Übersicht

Die letzten Phasen der Pipeline arbeiten auf den vorregistrierten/prewarped Frames, berücksichtigen den gemeinsamen Bildbereich (Common Overlap), stacken das finale Signal und erzeugen die finalen FITS-Outputs.

```
┌──────────────────────────────────────────────────────┐
│  PREWARP (Phase 2, vorgelagert)                      │
│  • Alle Frames auf globalen Canvas gewarpt           │
│  • CFA-safe Warp für OSC (Subplane-Methode)          │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  COMMON_OVERLAP (Phase 7)                            │
│  • Pixelweise Valid-Map über alle Frames             │
│  • common_overlap.json + Tile-Valid-Fraktionen       │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  SYNTHETIC_FRAMES (Phase 11)                         │
│  • Pro Cluster: gewichtetes Mittel → synth Frame     │
│  • N Original-Frames → K synthetische Frames         │
│  (übersprungen im Reduced Mode)                      │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  STACKING (Phase 12)                                 │
│  • Sigma-Clipping Rejection ODER Mean                │
│  • Output-Skalierung: × median_bg + pedestal         │
│  • stacked.fits + reconstructed_L.fit                │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  VALIDATION                                          │
│  • FWHM-Verbesserung check                           │
│  • Tile-Weight-Varianz check                         │
│  • Tile-Pattern-Detektion (Sobel)                    │
│  → validation.json                                   │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  DEBAYER (Phase 13) — nur OSC                        │
│  • Nearest-Neighbor Demosaic → R, G, B               │
│  • Canvas/Crop-Offsets für korrekte Bayer-Parität    │
│  • stacked_rgb.fits (3-Plane FITS-Cube)              │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  ASTROMETRY (Phase 14)                               │
│  • ASTAP solve / WCS                                 │
│  • WCS-Header in RGB-Outputs                         │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  PCC (Phase 15)                                      │
│  • Photometric Color Calibration                     │
│  • pcc_R/G/B.fit + stacked_rgb_pcc.fits              │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  DONE (Phase 16)                                     │
│  • run_end(ok) oder run_end(validation_failed)       │
└──────────────────────────────────────────────────────┘
```

## Canvas- und Common-Overlap-Kontext

- **PREWARP** schreibt alle Frames auf einen globalen Canvas (`canvas_width`, `canvas_height`, `tile_offset_x/y`).
- **COMMON_OVERLAP** ermittelt den tatsächlich gemeinsamen, datentragenden Pixelbereich über alle nutzbaren Frames.
- In `TILE_RECONSTRUCTION`/`STACKING` werden Pixel/Tiles gemäß
  - `stacking.common_overlap_required_fraction`
  - `stacking.tile_common_valid_min_fraction`
  gefiltert, damit Rand-/Leerbereiche den Stack nicht verfälschen.
- Bei OSC müssen Canvas-/Crop-Offsets Bayer-paritätskonsistent in Normalisierung/Output-Skalierung durchgereicht werden.

---

## Phase 11: SYNTHETIC_FRAMES

### Funktionsweise

Pro Cluster wird ein **global-gewichtetes Mittel** aller zugehörigen Frames berechnet:

```cpp
auto reconstruct_subset = [&](const std::vector<char> &frame_mask) -> Matrix2Df {
    for (size_t fi = 0; fi < frames.size(); ++fi) {
        if (!frame_mask[fi]) continue;
        auto pair = load_frame_normalized(fi);
        Matrix2Df img = apply_global_warp(pair.first, fi);
        warped.push_back(std::move(img));
        weights_subset.push_back(global_weights[fi]);
    }

    Matrix2Df out = Matrix2Df::Zero(rows, cols);
    for (size_t i = 0; i < warped.size(); ++i)
        out += warped[i] * (weights_subset[i] / wsum);
    return out;
};
```

```
synth_k = Σ_{f ∈ cluster_k} G_f · warp(I'_f) / Σ_{f ∈ cluster_k} G_f
```

### Cluster-Filterung

```cpp
for (int c = 0; c < n_clusters; ++c) {
    // Nur Cluster mit genug Frames
    if (count < synth_min) continue;
    Matrix2Df syn = reconstruct_subset(use_frame);
    synthetic_frames.push_back(syn);
    if (synthetic_frames.size() >= synth_max) break;
}
```

- **Mindestgröße**: Cluster mit `count < frames_min` werden übersprungen
- **Maximum**: Höchstens `frames_max` synthetische Frames
- Synthetische Frames werden als `synthetic_*.fit` gespeichert (mit Output-Skalierung)

### Skip-Bedingungen

| Bedingung | Verhalten |
|-----------|-----------|
| Reduced Mode | Phase skipped, `use_synthetic_frames = false` |
| Keine Frames ≥ frames_min | Skipped wenn N < frames_min, Error sonst |
| Alle Cluster zu klein | Error, Pipeline-Abbruch |

### Konfiguration

| Parameter | Beschreibung | Default |
|-----------|-------------|---------|
| `synthetic.frames_min` | Minimale Cluster-Größe | 3 |
| `synthetic.frames_max` | Maximale Anzahl synth. Frames | 30 |

### Artifact: `synthetic_frames.json`

```json
{
  "num_synthetic": 8,
  "frames_min": 3,
  "frames_max": 30
}
```

---

## Phase 12: STACKING

### Sigma-Clipping Rejection Stacking

```cpp
if (use_synthetic_frames) {
    if (cfg.stacking.method == "rej") {
        recon = reconstruction::sigma_clip_stack(
            synthetic_frames,
            cfg.stacking.sigma_clip.sigma_low,
            cfg.stacking.sigma_clip.sigma_high,
            cfg.stacking.sigma_clip.max_iters,
            cfg.stacking.sigma_clip.min_fraction);
    } else {
        // Mean stacking
        for (const auto &sf : synthetic_frames) recon += sf;
        recon /= synthetic_frames.size();
    }
}
```

| Methode | Config | Beschreibung |
|---------|--------|-------------|
| **rej** | `stacking.method: "rej"` | Sigma-Clipping Rejection (Standard) |
| **mean** | `stacking.method: "mean"` | Einfaches Mittel |

#### Sigma-Clipping Parameter

| Parameter | Beschreibung | Default |
|-----------|-------------|---------|
| `stacking.sigma_clip.sigma_low` | Untere σ-Grenze | 2.5 |
| `stacking.sigma_clip.sigma_high` | Obere σ-Grenze | 3.0 |
| `stacking.sigma_clip.max_iters` | Maximale Iterationen | 5 |
| `stacking.sigma_clip.min_fraction` | Mindestanteil beibehaltener Pixel | 0.5 |

### Reduced Mode

Wenn `use_synthetic_frames = false`, wird das Rekonstruktionsergebnis aus Phase 9 direkt als finales Bild übernommen — kein erneutes Stacking.

### Output-Skalierung

```cpp
Matrix2Df recon_out = recon;
if (detected_mode == ColorMode::OSC) {
    const float bg_luma = 0.25f * output_bg_r + 0.5f * output_bg_g + 0.25f * output_bg_b;
    recon_out *= bg_luma;
    recon_out.array() += output_pedestal;
} else {
    // MONO: origin muss Canvas/Crop-Offset enthalten.
    image::apply_output_scaling_inplace(
        recon_out,
        -debayer_tile_offset_x,
        -debayer_tile_offset_y,
        detected_mode,
        detected_bayer_str,
        output_bg_mono, output_bg_r, output_bg_g, output_bg_b,
        output_pedestal);
}
io::write_fits_float(run_dir / "outputs" / "stacked.fits", recon_out, first_hdr);
io::write_fits_float(run_dir / "outputs" / "reconstructed_L.fit", recon_out, first_hdr);
```

Die Skalierung konvertiert normalisierte Werte zurück in physikalische Einheiten:

| Modus | Skalierung |
|-------|-----------|
| **MONO** | `pixel = pixel × output_bg_mono + 32768` |
| **OSC (Stack-L)** | `pixel = pixel × (0.25·bg_r + 0.5·bg_g + 0.25·bg_b) + 32768` |

- **output_bg_***: Median der Background-Werte über alle Frames (aus NORMALIZATION, Phase 4)
- **Pedestal 32768**: Verhindert negative Werte in FITS (16-bit compatible)

---

## Validation

Die Validierung prüft die Qualität des Rekonstruktionsergebnisses:

### 1. FWHM-Verbesserung

```cpp
float output_fwhm_med = metrics::measure_fwhm_from_image(recon);
float improvement = (seeing_fwhm - output_fwhm) / seeing_fwhm * 100.0f;
v["fwhm_improvement_ok"] = (improvement >= cfg.validation.min_fwhm_improvement_percent);
```

- Vergleich: Seeing-FWHM (Phase 4) vs. Output-FWHM
- **Erwartung**: Output-FWHM sollte kleiner sein (besseres Seeing durch Stacking)

### 2. Tile-Weight-Varianz

```cpp
// Mittleres effektives Gewicht pro Tile
for (size_t ti = 0; ti < tiles.size(); ++ti) {
    float mean_W = Σ (G_f × L_f,t) / N;
    tile_means.push_back(mean_W);
}
// Normalisierte Varianz: Var(tile_means) / mean(tile_means)²
```

- **Prüft**: Gibt es genug Variation in den Tile-Gewichten?
- Zu niedrige Varianz → Gewichtung hat keinen Effekt → Warnung

### 3. Tile-Pattern-Detektion

```cpp
// Sobel-Gradient an Tile-Grenzen vs. Nachbar-Pixel
cv::Sobel(img_cv, gx, CV_32F, 1, 0, 3);
cv::Sobel(img_cv, gy, CV_32F, 0, 1, 3);
cv::magnitude(gx, gy, mag);

for (int x : tile_boundaries_x) {
    float boundary_grad = line_mean_x(x);
    float neighbor_grad = 0.5 * (line_mean_x(x-2) + line_mean_x(x+2));
    float ratio = boundary_grad / neighbor_grad;
    worst_ratio = max(worst_ratio, ratio);
}
tile_pattern_ok = (worst_ratio < 1.5f);
```

- **Prüft**: Sind Tile-Grenzen im finalen Bild sichtbar?
- Vergleicht Sobel-Gradient an Tile-Grenzen mit benachbarten Pixeln
- **Ratio > 1.5**: Tile-Pattern erkannt → Warnung
- Nur wenn `validation.require_no_tile_pattern = true`

### Validation-Ergebnis

```cpp
if (!validation_ok) {
    run_validation_failed = true;
    // Pipeline läuft weiter (DEBAYER wird noch ausgeführt!)
}
```

- Validation-Fehler bricht die Pipeline **nicht** ab
- DEBAYER wird immer noch ausgeführt (GUI braucht die Outputs)
- Run wird am Ende als `"validation_failed"` markiert

### Artifact: `validation.json`

```json
{
  "seeing_fwhm_median": 3.5,
  "output_fwhm_median": 2.8,
  "fwhm_improvement_percent": 20.0,
  "fwhm_improvement_ok": true,
  "tile_weight_variance": 0.15,
  "tile_weight_variance_ok": true,
  "tile_pattern_ratio": 1.02,
  "tile_pattern_ok": true
}
```

---

## Phase 13: DEBAYER

### OSC-Modus

```cpp
if (detected_mode == ColorMode::OSC) {
    auto debayer = image::debayer_nearest_neighbor(recon, detected_bayer);

    Matrix2Df R_out = debayer.R * output_bg_r + output_pedestal;
    Matrix2Df G_out = debayer.G * output_bg_g + output_pedestal;
    Matrix2Df B_out = debayer.B * output_bg_b + output_pedestal;

    io::write_fits_float("reconstructed_R.fit", R_out, first_hdr);
    io::write_fits_float("reconstructed_G.fit", G_out, first_hdr);
    io::write_fits_float("reconstructed_B.fit", B_out, first_hdr);
    io::write_fits_rgb("stacked_rgb.fits", R_out, G_out, B_out, first_hdr);
}
```

- **Nearest-Neighbor Demosaic**: Deterministisch, schnell
- **Per-Kanal Output-Skalierung**: Jeder Kanal × eigener Median-Background + Pedestal
- **Offset-Korrektheit:** Debayer-/Crop-Offsets werden in die Skalierung eingebracht, um Bayer-Paritätsfehler (R/G-Vertauschung) zu vermeiden.
- **stacked_rgb.fits**: 3-Plane FITS-Cube (NAXIS3=3) für direkte RGB-Anzeige

### MONO-Modus

- Phase wird als "ok" mit `mode: "MONO"` beendet
- Keine Demosaic-Operation notwendig
- Output ist bereits in `stacked.fits` / `reconstructed_L.fit`

### Output-Dateien

| Datei | Modus | Beschreibung |
|-------|-------|-------------|
| `stacked.fits` | Beide | Rekonstruiertes Bild (mit Skalierung) |
| `reconstructed_L.fit` | Beide | Identisch zu stacked.fits |
| `reconstructed_R.fit` | OSC | Roter Kanal |
| `reconstructed_G.fit` | OSC | Grüner Kanal |
| `reconstructed_B.fit` | OSC | Blauer Kanal |
| `stacked_rgb.fits` | OSC | RGB-Cube (NAXIS3=3) |
| `stacked_rgb_solve.fits` | OSC | Lineares RGB für Astrometrie/WCS |
| `stacked_rgb_pcc.fits` | OSC | PCC-kalibrierter RGB-Cube |
| `pcc_R.fit` | OSC | PCC-kalibrierter R-Kanal |
| `pcc_G.fit` | OSC | PCC-kalibrierter G-Kanal |
| `pcc_B.fit` | OSC | PCC-kalibrierter B-Kanal |
| `synthetic_*.fit` | Normal | Synthetische Frames (mit Skalierung) |

---

## Phase 14: ASTROMETRY

Wenn aktiviert und RGB-Daten vorhanden sind, wird ein ASTAP-Plate-Solve ausgeführt. Bei Erfolg:

- WCS wird aus `.wcs` geparst
- WCS-Header in `first_hdr` injiziert
- `stacked_rgb.fits` und `stacked_rgb_solve.fits` mit WCS neu geschrieben
- `artifacts/stacked_rgb.wcs` abgelegt

Bei Fehlschlag wird die Phase als `skipped` mit Grund (`astap_not_found`, `solve_failed`, etc.) beendet.

## Phase 15: PCC

PCC läuft nur, wenn:

- `cfg.pcc.enabled = true`
- WCS aus ASTROMETRY vorhanden
- RGB-Daten vorhanden

Outputs bei Erfolg:

- `outputs/pcc_R.fit`
- `outputs/pcc_G.fit`
- `outputs/pcc_B.fit`
- `outputs/stacked_rgb_pcc.fits`

Bei fehlenden Katalogsternen oder Fit-Problemen wird die Phase als `skipped` beendet.

## Phase 16: DONE

```cpp
emitter.phase_start(run_id, Phase::DONE, "DONE", log_file);
emitter.phase_end(run_id, Phase::DONE, "ok", {}, log_file);

if (run_validation_failed) {
    emitter.run_end(run_id, false, "validation_failed", log_file);
    return 1;
}

emitter.run_end(run_id, true, "ok", log_file);
return 0;
```

| Ergebnis | Exit-Code | Status |
|----------|-----------|--------|
| **Erfolgreich** | 0 | `"ok"` |
| **Validation fehlgeschlagen** | 1 | `"validation_failed"` |
| **Pipeline-Fehler** | 1 | `"error"` (bei vorherigem Abbruch) |

---

## CLI-Optionen

```
tile_compile_runner run \
    --config <path>        # config.yaml
    --input-dir <path>     # FITS Input-Verzeichnis
    --runs-dir <path>      # Runs-Ausgabe-Verzeichnis
    --project-root <path>  # Projekt-Root (optional)
    --max-frames <n>       # Frame-Limit (0 = kein Limit)
    --max-tiles <n>        # Tile-Limit im Tile-Pfad (ab Phase 6)
    --dry-run              # Kein Processing
    --stdin                # Config von stdin lesen
```