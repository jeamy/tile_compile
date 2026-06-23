# Pipeline Overview — AQMH (Default) vs. Classic Tile Compile

> **C++ Implementierung:** `runner_pipeline.cpp`, `runner_phase_local_metrics.cpp`
> **Default method:** `aqmh` (Adaptive Quality Map Harvesting)
> **Alternative method:** `classic_tile_compile`

## Übersicht

tile_compile unterstützt zwei Rekonstruktionsmethoden, die über `method` in der Konfiguration gewählt werden:

| Method | Beschreibung | Standard |
|--------|-------------|----------|
| **`aqmh`** | Adaptive Quality Map Harvesting — pixelweise Qualitätsbewertung mit Pyramidendarstellung, Cherry-Pick-Frame-Selektion, unabhängige Rekonstruktion | ✅ Default |
| **`classic_tile_compile`** | Klassische tile-basierte Rekonstruktion mit lokalen Tile-Metriken, Clustering und synthetischen Frames | Optional |

```cpp
// config.cpp: normalizeMethod()
if (config.method.empty()) config.method = "aqmh";
if (config.method == "aqmh") config.aqmh.enabled = true;
else if (config.method == "classic_tile_compile") config.aqmh.enabled = false;
```

## Pipeline-Phasen (gesamt, 0–18)

| ID | Enum | AQMH-Verhalten | Classic-Verhalten |
|----|------|---------------|-------------------|
| 0 | `SCAN_INPUT` | Identisch | Identisch |
| 1 | `REGISTRATION` | Identisch | Identisch |
| 2 | `PREWARP` | Identisch | Identisch |
| 3 | `CHANNEL_SPLIT` | Identisch (Metadaten) | Identisch (Metadaten) |
| 4 | `NORMALIZATION` | Identisch | Identisch |
| 5 | `GLOBAL_METRICS` | Identisch | Identisch |
| 6 | `TILE_GRID` | Identisch | Identisch |
| 7 | `COMMON_OVERLAP` | Identisch | Identisch |
| 8 | `LOCAL_METRICS` | **AQMH_QUALITY_MAPS** — Pyramid-Qualitätskarten pro Frame werden berechnet und gecacht | **Classic LOCAL_METRICS** — Tile-Metriken (FWHM, Roundness, Contrast, Star Count) pro (frame, tile) |
| 9 | `TILE_RECONSTRUCTION` | **Pixelweise AQMH-Rekonstruktion** mit `reconstruct_aqmh_weighted()` — Cherry-Pick-Selektion, Sigma-Clip | **Tile-basierte Rekonstruktion** mit `W_f,t = G_f × L_f,t`, OLA |
| 10 | `STATE_CLUSTERING` | **Skipped** (`aqmh_independent_reconstruction`) | Aktiv (wenn N ≥ Schwellwert) — 6D State-Vector Clustering |
| 11 | `SYNTHETIC_FRAMES` | **Skipped** (`aqmh_independent_reconstruction`) | Aktiv — gewichtete Cluster-Mittelwerte |
| 12 | `STACKING` | Durchlauf der AQMH-Rekonstruktion | Sigma-Clip-Stacking der synthetischen Frames |
| 13 | `DEBAYER` | Identisch | Identisch |
| 14 | `ASTROMETRY` | Identisch (optional) | Identisch (optional) |
| 15 | `BGE` | Identisch (optional) | Identisch (optional) |
| 16 | `PCC` | Identisch (optional) | Identisch (optional) |
| 17 | `HYPERMETRIC_STRETCH` | Identisch (optional) | Identisch (optional) |
| 18 | `DONE` | Identisch | Identisch |

> **Validation** ist ein Qualitätsblock zwischen `STACKING` und `DEBAYER`, aber keine eigene Phase.

## AQMH-spezifische Phasen im Detail

### Phase 8: AQMH_QUALITY_MAPS (ersetzt LOCAL_METRICS)

Wenn `aqmh.enabled = true`:

- Classic Tile-Metriken werden **nicht berechnet** (`compute_classic_local_metrics = false`)
- Stattdessen werden **pyramid-basierte Qualitätskarten** pro Frame berechnet:
  - Multi-Scale-Sharpness und SNR über Pyramid-Level
  - Artefakt-Detektion (`k_artifact`, `frac_artifact_max`)
  - Qualitätskarte `Q_map` pro Frame wird in `QualityMapCache` gespeichert (`runs/<id>/cache/aqmh/`)
- Phase-Anzeige: `AQMH_QUALITY_MAPS` statt `LOCAL_METRICS`
- Artifact: `aqmh_metrics.json` (statt `local_metrics.json`)

```cpp
// runner_phase_local_metrics.cpp
const bool compute_classic_local_metrics = !cfg.aqmh.enabled;
const std::string phase_display_name =
    compute_classic_local_metrics ? "LOCAL_METRICS" : "AQMH_QUALITY_MAPS";
```

#### AQMH-Konfiguration

| Parameter | Beschreibung | Default |
|-----------|-------------|---------|
| `aqmh.pyramid.scales` | Anzahl Pyramid-Level (1–8) | 4 |
| `aqmh.pyramid.base_window_px` | Basis-Fenstergröße | 16 |
| `aqmh.pyramid.w_sharp` | Gewicht Sharpness | 0.6 |
| `aqmh.pyramid.w_snr` | Gewicht SNR | 0.4 |
| `aqmh.pyramid.k_artifact` | Artefakt-Schwellwert | 2.0 |
| `aqmh.pyramid.frac_artifact_max` | Max. Artefakt-Anteil | 0.1 |
| `aqmh.storage.resolution_divisor` | Speicher-Auflösungsteiler (1/2/4) | 2 |
| `aqmh.storage.dtype` | Speichertyp (float32/uint16/uint8) | float32 |
| `aqmh.storage.max_resident_maps` | Max. im RAM gehaltene Karten (0–16) | 4 |
| `aqmh.cherry_pick.enabled` | Cherry-Pick-Frame-Selektion | false |
| `aqmh.cherry_pick.k_min` | Min. Frames pro Pixel | 3 |
| `aqmh.cherry_pick.k_frac` | Fraktion der besten Frames | 0.5 |

### Phase 9: AQMH-Rekonstruktion (ersetzt TILE_RECONSTRUCTION)

Wenn `aqmh.enabled = true`:

- **Pixelweise Rekonstruktion** statt tile-basiert:
  - Für jeden Pixel: Quality-Map-Werte aller Frames werden mit globalen Gewichten `G_f` kombiniert
  - Sigma-Clip-Rejection auf Pixel-Ebene
  - Optional Cherry-Pick: nur die `k` besten Frames pro Pixel werden verwendet
- Kein Clustering, keine synthetischen Frames
- Artifact: `tile_reconstruction.json` mit `"method": "aqmh"` und Cherry-Pick-Diagnostik

```cpp
// runner_pipeline.cpp
if (cfg.aqmh.enabled) {
    auto aqmh_recon = reconstruction::reconstruct_aqmh_weighted(
        frames.size(), aqmh_frame_loader, aqmh_cache.get(),
        global_weights, common_valid_mask,
        canvas_width, canvas_height, aqmh_recon_cfg);
    recon = aqmh_recon.output;
}
```

### Phase 10–11: STATE_CLUSTERING und SYNTHETIC_FRAMES (skipped)

Bei AQMH werden beide Phasen übersprungen:

```cpp
const bool skip_clustering_for_aqmh = cfg.aqmh.enabled;
// -> phase_end(STATE_CLUSTERING, "skipped", {"reason": "aqmh_independent_reconstruction"})
// -> phase_end(SYNTHETIC_FRAMES, "skipped", {"reason": "aqmh_independent_reconstruction"})
```

## Pipeline-Flow-Diagramm (AQMH, Default)

```
┌─────────────────────────────────────────────────────────────┐
│               INPUT: MONO / OSC RAW FITS FRAMES             │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 0: SCAN_INPUT         │
              │  • FITS dimensions + header  │
              │  • Color mode (MONO/OSC)     │
              │  • Bayer pattern detection   │
              │  • Linearity validation      │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 1: REGISTRATION       │
              │  • Cascaded fallbacks        │
              │  • CC / warp metrics         │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 2: PREWARP            │
              │  • Full-frame canvas warp    │
              │  • CFA-safe (OSC)            │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 3: CHANNEL_SPLIT      │
              │  (metadata-only)             │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 4: NORMALIZATION      │
              │  • Sigma-clip BG mask        │
              │  • Additive background B_f   │
              │  • Photometric scale P_f     │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 5: GLOBAL_METRICS     │
              │  • B_f, σ_f, E_f per frame   │
              │  • G_f = exp(α·B̃+β·σ̃+γ·Ẽ)    │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 6: TILE_GRID          │
              │  • FWHM probe (central ROI)  │
              │  • Uniform tile grid         │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 7: COMMON_OVERLAP     │
              │  • Pixelwise valid overlap   │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 8: AQMH_QUALITY_MAPS  │
              │  • Pyramid quality maps      │
              │  • Multi-scale sharpness+SNR │
              │  • QualityMapCache (disk)    │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 9: TILE_RECONSTRUCTION│
              │  • Pixel-wise AQMH weighted  │
              │  • Sigma-clip rejection      │
              │  • Cherry-pick (optional)    │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 10: STATE_CLUSTERING  │
              │  ⏭ SKIPPED (aqmh)            │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 11: SYNTHETIC_FRAMES  │
              │  ⏭ SKIPPED (aqmh)            │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 12: STACKING          │
              │  • Pass-through (AQMH)       │
              │  • Validation block          │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 13: DEBAYER           │
              │  • OSC: NN demosaic -> RGB   │
              │  • MONO: pass-through        │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 14: ASTROMETRY       │
              │  • ASTAP solve / WCS        │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 15: BGE              │
              │  • Optional gradient removal│
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 16: PCC              │
              │  • Photometric color cal.   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 17: HMS              │
              │  • HyperMetric Stretch      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 18: DONE             │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  OUTPUTS:                   │
              │  • stacked.fits             │
              │  • reconstructed_L.fit      │
              │  • stacked_rgb.fits (OSC)   │
              │  • stacked_rgb_solve.fits   │
              │  • stacked_rgb_bge.fits     │
              │  • stacked_rgb_pcc.fits     │
              │  • stacked_rgb_hms.fits     │
              │  • aqmh_metrics.json        │
              │  • 12 artifact JSON files   │
              │  • run_events.jsonl         │
              └─────────────────────────────┘
```

## Classic Tile Compile Pipeline (method: classic_tile_compile)

Bei `method: classic_tile_compile` gelten die Phasen 8–11 wie ursprünglich:

- **Phase 8: LOCAL_METRICS** — Tile-Metriken (FWHM, Roundness, Contrast, Star Count) pro (frame, tile)
- **Phase 9: TILE_RECONSTRUCTION** — `W_f,t = G_f × L_f,t`, tile-basierte gewichtete Rekonstruktion mit OLA
- **Phase 10: STATE_CLUSTERING** — 6D State-Vector Clustering (aktiv wenn N ≥ Schwellwert)
- **Phase 11: SYNTHETIC_FRAMES** — Gewichtete Cluster-Mittelwerte, Frame-Reduktion N → K

Siehe die einzelnen Phase-Dokumente für Details zur Classic-Pipeline.

---

# Phase 0: SCAN_INPUT — Input-Scan, Erkennung und Linearitätsprüfung

> **C++ Implementierung:** `runner_pipeline.cpp`
> **Phase-Enum:** `Phase::SCAN_INPUT`

Phase 0 ist die Eingangsphase der Pipeline. Sie liest den ersten Frame, erkennt den Bildmodus und das Bayer-Pattern, führt eine optionale Linearitätsprüfung durch und bereitet die Run-Infrastruktur vor.

```
┌─────────────────────────────────────────────────────────────┐
│          INPUT: Verzeichnis mit FITS-Frames (*.fit*)        │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  1. Frame-Discovery          │
              │     core::discover_frames()  │
              │     Sortierung + Limit       │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  2. Run-Verzeichnis anlegen  │
              │     runs/<run_id>/           │
              │     ├── logs/                │
              │     ├── outputs/             │
              │     └── artifacts/           │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  3. Erster Frame lesen       │
              │     • Dimensionen (W×H)      │
              │     • NAXIS                  │
              │     • FITS-Header            │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  4. Modus-Erkennung          │
              │     • MONO vs. OSC           │
              │     • Bayer-Pattern          │
              │       (RGGB, GRBG, etc.)     │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  5. Linearitätsprüfung       │
              │     • Stichprobe samplen     │
              │     • validate_linearity()   │
              │     • Rejection oder Warnung │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  OUTPUT:                     │
              │  • frames[] (validiert)      │
              │  • ColorMode, BayerPattern   │
              │  • width, height             │
              │  • linearity_info JSON       │
              └──────────────────────────────┘
```

## Detaillierter Ablauf

### 1. Frame-Discovery und Sortierung

```cpp
auto frames = core::discover_frames(in_dir, "*.fit*");
std::sort(frames.begin(), frames.end());
if (max_frames > 0 && frames.size() > max_frames)
    frames.resize(max_frames);
```

- Sucht alle Dateien im Input-Verzeichnis die `*.fit*` matchen (FITS, FIT, FITS)
- Sortiert alphabetisch für deterministische Reihenfolge
- Optional: Beschränkung auf `--max-frames` Frames (Debug/Test)
- **Abbruch** wenn keine Frames gefunden

### 2. Run-Infrastruktur

```cpp
std::string run_id = core::get_run_id();  // Zeitstempel-basierte ID
fs::path run_dir = runs / run_id;
fs::create_directories(run_dir / "logs");
fs::create_directories(run_dir / "outputs");
fs::create_directories(run_dir / "artifacts");
```

- Eindeutige Run-ID (Zeitstempel-basiert)
- Konfiguration wird als `config.yaml` in den Run-Ordner kopiert
- Event-Log-Datei: `run_events.jsonl` (TeeBuf → stdout + Datei gleichzeitig)

### 3. Erster Frame — Dimensionen und Header

```cpp
auto [width, height, naxis] = io::get_fits_dimensions(frames.front());
auto first = io::read_fits_float(frames.front());
first_frame = std::move(first.first);    // Matrix2Df (Eigen)
first_header = std::move(first.second);  // FitsHeader
```

- Liest den **ersten Frame** vollständig ein
- Extrahiert Bildbreite, Bildhöhe und NAXIS
- Speichert den Frame als `first_frame` für spätere Verwendung
- Speichert den FITS-Header für Output-Dateien

### 4. Farbmodus-Erkennung

```cpp
detected_mode = io::detect_color_mode(first_header, naxis);
detected_bayer = io::detect_bayer_pattern(first_header);
```

| Modus | Erkennung | Verhalten |
|-------|-----------|-----------|
| **MONO** | NAXIS=2 oder kein BAYERPAT | Einzelkanal-Verarbeitung |
| **OSC** | NAXIS=2 + BAYERPAT vorhanden | CFA-aware Verarbeitung |

- **Konfig-Override**: Wenn `data.color_mode` in config gesetzt, wird bei Abweichung gewarnt
- **Bayer-Pattern**: RGGB, GRBG, GBRG, BGGR — wird aus FITS-Header `BAYERPAT` gelesen
- Bei unbekanntem Pattern: Warnung, Fallback auf RGGB

### 5. Linearitätsprüfung

Die Linearitätsprüfung validiert, dass die Frames **keine nichtlinearen Operationen** (Stretch, Curves) erfahren haben.

```cpp
if (cfg.linearity.enabled || cfg.data.linear_required) {
    auto indices = core::sample_indices(frames.size(), cfg.linearity.max_frames);
    for (size_t idx : indices) {
        auto res = metrics::validate_linearity_frame(frame_img, cfg.linearity.strictness);
        if (!res.is_linear) {
            rejected_indices.push_back(idx);
        }
    }
}
```

#### Konfigurationsparameter

| Parameter | Beschreibung | Default |
|-----------|-------------|---------|
| `linearity.enabled` | Linearitätsprüfung aktivieren | `true` |
| `linearity.max_frames` | Maximale Stichprobengröße | 10 |
| `linearity.strictness` | Strictness-Level für Validierung | 0.5 |
| `linearity.min_overall_linearity` | Mindest-Linearitäts-Score | 0.8 |
| `data.linear_required` | Nicht-lineare Frames entfernen | `true` |

#### Verhalten bei nicht-linearen Frames

| `linear_required` | Verhalten |
|--------------------|-----------|
| `true` | Nicht-lineare Frames werden aus `frames[]` **entfernt** |
| `false` | Warnung, Frames bleiben in der Pipeline |

- Bei `linear_required=true` und **alle** Frames rejected: Pipeline bricht mit Error ab
- Linearity-Info wird als JSON in das `scan_extra` Event geschrieben

#### Linearity-Info JSON

```json
{
  "enabled": true,
  "sampled_frames": 10,
  "overall_linearity": 0.9,
  "min_overall_linearity": 0.8,
  "failed_frames": 1,
  "failed_frame_names": ["frame_0023.fit"],
  "flagged_indices": [23],
  "action": "removed",
  "frames_remaining": 99
}
```

## CHANNEL_SPLIT (Phase 3 — Metadaten-Phase)

Direkt nach SCAN_INPUT wird `Phase::CHANNEL_SPLIT` emittiert. In der C++ Implementierung ist dies eine **reine Metadaten-Phase** — die eigentliche Kanaltrennung erfolgt **deferred** während der Normalisierung und Tile-Verarbeitung.

```cpp
if (detected_mode == ColorMode::OSC) {
    extra["mode"] = "OSC";
    extra["channels"] = {"R", "G", "B"};
    extra["bayer_pattern"] = detected_bayer_str;
    extra["note"] = "deferred_to_tile_processing";
} else {
    extra["mode"] = "MONO";
    extra["channels"] = {"L"};
}
```

Bei OSC-Daten bleibt das CFA-Mosaik bis zum Debayer in Phase 13 intakt. Die kanalgetrennte Verarbeitung geschieht implizit über Bayer-Offsets in der Normalisierung.

## Fehlerbehandlung

| Fehler | Verhalten |
|--------|-----------|
| Input-Verzeichnis existiert nicht | Sofortiger Abbruch (return 1) |
| Keine FITS-Frames gefunden | Sofortiger Abbruch (return 1) |
| Erster Frame nicht lesbar | phase_end(error) → run_end(error) → return 1 |
| Alle Frames non-linear | phase_end(error) → run_end(error) → return 1 |
| Config/Header Mismatch | Warnung, Pipeline läuft weiter |

## Event-Emitter-Aufrufe

```
run_start(run_id, {config_path, input_dir, run_dir, frames_discovered, dry_run})
phase_start(SCAN_INPUT)
  [warnings: linearity, mode mismatch]
phase_end(SCAN_INPUT, "ok", {input_dir, frames_scanned, image_width, image_height,
                              color_mode, bayer_pattern, linearity})
phase_start(CHANNEL_SPLIT)
phase_end(CHANNEL_SPLIT, "ok", {mode, channels, bayer_pattern})
```

## Nächste Phase

→ **Phase 1/2: REGISTRATION + PREWARP**, danach **Phase 4: NORMALIZATION — Hintergrund-Normalisierung**
