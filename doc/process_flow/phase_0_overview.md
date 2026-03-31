# Phase 0: SCAN_INPUT — Input-Scan, Erkennung und Linearitätsprüfung

> **C++ Implementierung:** `runner_pipeline.cpp`
> **Phase-Enum:** `Phase::SCAN_INPUT`

## Übersicht

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
