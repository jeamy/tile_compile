# Process Flow Documentation — C++ Pipeline (tile_compile_cpp)

## Übersicht

Diese Dokumentation beschreibt den **tatsächlichen Ablauf der C++ Implementierung** (`tile_compile_cpp/apps/runner_main.cpp`) der Tile-basierten Qualitätsrekonstruktion für Deep-Sky-Objekte gemäß Methodik v3.

Die Pipeline verarbeitet **FITS-Frames** (Mono oder OSC/CFA) und erzeugt ein gewichtetes, Tile-basiertes Rekonstruktionsergebnis mit optionaler Clusterung, synthetischen Frames und Sigma-Clipping-Stacking.

**Implementierung:** C++ mit Eigen, OpenCV, cfitsio, nlohmann/json, YAML-cpp.

## Aktuelle Pipeline-Phasen (C++ Implementierung, v3.2)

Quelle der Phasenreihenfolge: `tile_compile::Phase` in `include/tile_compile/core/types.hpp`.

| ID | Enum | Kurzbeschreibung |
|----|------|------------------|
| 0 | `SCAN_INPUT` | Input-Scan, Header/Modus-Erkennung, Linearitätsprüfung, Disk-Space-Precheck (`scandir*4`) |
| 1 | `REGISTRATION` | Globale Registrierung (kaskadiert), Frame-Usability, Vollbild-Prewarp |
| 2 | `CHANNEL_SPLIT` | Metadaten-Phase (OSC/Mono-Kanalmodell; tatsächliche Kanalarbeit erfolgt später) |
| 3 | `NORMALIZATION` | Globale lineare Normalisierung (background-basiert) |
| 4 | `GLOBAL_METRICS` | Globale Frame-Metriken und Gewichte `G_f` |
| 5 | `TILE_GRID` | Adaptive Tile-Geometrie (Seeing/FWHM-basiert) |
| 6 | `LOCAL_METRICS` | Lokale Tile-Metriken/Klassifikation und lokale Gewichte `L_f,t` |
| 7 | `TILE_RECONSTRUCTION` | Gewichtete tile-basierte Rekonstruktion (Overlap-Add) |
| 8 | `STATE_CLUSTERING` | Zustandsvektor-Clustering (optional, mode-abhängig) |
| 9 | `SYNTHETIC_FRAMES` | Erzeugung synthetischer Frames (optional, mode-abhängig) |
| 10 | `STACKING` | Finales lineares Stacking (inkl. robuster Pixel-Ausreißerbehandlung) |
| 11 | `DEBAYER` | OSC-Debayering und RGB-Ausgabe (bei MONO: pass-through) |
| 12 | `ASTROMETRY` | Plate Solving / WCS |
| 13 | `PCC` | Photometric Color Calibration |
| 14 | `DONE` | Abschlussstatus (`ok` oder `validation_failed`) |

Hinweis: **Validation** ist ein Qualitätsblock zwischen `STACKING` und `DONE`, aber keine eigene Enum-Phase.

## Dokumenten-Struktur

Die Detaildokumente in diesem Ordner bleiben als Deep-Dive erhalten. 
Die **verbindliche Phasenreihenfolge** ist jedoch die oben stehende v3.2-Liste (0..14).

Kurzzuordnung:

- Input + Modus + Linearität + Disk-Precheck → `SCAN_INPUT`
- Registrierung/Prewarp → `REGISTRATION`
- Normalisierung + globale/lokale Gewichte + Rekonstruktion → `NORMALIZATION` bis `TILE_RECONSTRUCTION`
- Optionaler Full-Mode-Block → `STATE_CLUSTERING`, `SYNTHETIC_FRAMES`
- Finalisierungspfad → `STACKING`, `DEBAYER`, `ASTROMETRY`, `PCC`, `DONE`

---

## Pipeline-Flussdiagramm (C++ Implementierung, v3.2)

```
┌─────────────────────────────────────────────────────────────┐
│               INPUT: MONO / OSC RAW FITS FRAMES             │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 0: SCAN_INPUT        │
              │  • FITS dimensions + header  │
              │  • Color mode (MONO/OSC)     │
              │  • Bayer pattern detection   │
              │  • Linearity validation      │
              │  • Frame rejection (optional)│
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 1: REGISTRATION      │
              │  • Kaskadierte Fallbacks     │
              │  • CC/Warp-Metriken          │
              │  • Vollbild-Prewarp          │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 2: CHANNEL_SPLIT     │
              │  (metadata-only)             │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 3: NORMALIZATION     │
              │  • Sigma-clip BG mask        │
              │  • OSC: per-channel BG       │
              │  • MONO: single BG           │
              │  • Scale = 1/Background      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 4: GLOBAL_METRICS    │
              │  • B_f, σ_f, E_f per frame   │
              │  • MAD-normalize → z-scores  │
              │  • G_f = exp(α·B̃+β·σ̃+γ·Ẽ)   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 5: TILE_GRID         │
              │  • FWHM probe (central ROI)  │
              │  • T = clip(s·F, min, max)   │
              │  • Overlap + stride calc     │
              │  • Uniform tile grid         │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 6: LOCAL_METRICS     │
              │  • Per (frame,tile) metrics   │
              │  • Tile type: STAR/STRUCTURE │
              │  • Robust z-score quality Q  │
              │  • L_f,t = exp(clip(Q))      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 7: TILE_RECONSTRUCTION│
              │  • W_f,t = G_f × L_f,t       │
              │  • Weighted sum per tile     │
              │  • BG normalization per tile │
              │  • Hanning window + overlap  │
              │  • Parallel (N threads)      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 8: STATE_CLUSTERING  │
              │  • 6D state vector per frame │
              │  • K-Means (K=N/10)          │
              │  • Quantile fallback         │
              │  (skip in Reduced Mode)      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 9: SYNTHETIC_FRAMES  │
              │  • Weighted mean per cluster │
              │  • N → K frame reduction     │
              │  (skip in Reduced Mode)      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 10: STACKING         │
              │  • Sigma-clip rejection      │
              │  • Or mean of synth frames   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  VALIDATION                 │
              │  • FWHM improvement check    │
              │  • Tile weight variance      │
              │  • Tile pattern detection    │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 11: DEBAYER          │
              │  • OSC: NN demosaic → RGB    │
              │  • MONO: pass-through        │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 12: ASTROMETRY       │
              │  • ASTAP solve / WCS         │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 13: PCC              │
              │  • Photometrische Farbkal.   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 14: DONE             │
              │  • Final status emit         │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  OUTPUTS:                   │
              │  • stacked.fits              │
              │  • reconstructed_L.fit       │
              │  • stacked_rgb.fits (OSC)    │
              │  • stacked_rgb_solve.fits    │
              │  • stacked_rgb_pcc.fits      │
              │  • R/G/B .fit (OSC)          │
              │  • 9 artifact JSON files     │
              │  • run_events.jsonl          │
              └─────────────────────────────┘
```

## Kernprinzipien (C++ Implementierung)

1. **Linearität**: Keine nichtlinearen Operationen (kein Stretch). Linearitätsprüfung in Phase 0
2. **Keine Frame-Selektion**: Alle Frames werden verwendet. Fehlgeschlagene Registrierung → Identity-Warp mit CC=0
3. **Mono + OSC**: Beide Modi in einer einzigen Pipeline, CFA-aware bei OSC
4. **Streng sequentiell**: Keine Rückkopplungen, deterministische Abfolge
5. **Tile-basiert**: Lokale Qualitätsbewertung, Seeing-adaptive Tile-Größe
6. **Global × Lokal**: Effektives Gewicht W_f,t = G_f × L_f,t
7. **Pre-Warping**: Alle Frames werden vor Tile-Extraktion vollständig gewarpt (vermeidet CFA-Artefakte)
8. **Robuste Statistik**: Median, MAD, Sigma-Clipping durchgehend

## Modi

### Full/Normal Mode (N ≥ frames_reduced_threshold)
- Alle Enum-Phasen 0..14 werden durchlaufen
- State-Clustering aktiv
- Synthetische Frames werden erzeugt
- Sigma-Clipping Rejection Stacking
- Optimale Rauschreduktion

### Reduced Mode (N < frames_reduced_threshold)
- Phase 8 (`STATE_CLUSTERING`) wird **übersprungen**
- Phase 9 (`SYNTHETIC_FRAMES`) wird **übersprungen**
- Phase 7 (TILE_RECONSTRUCTION) erzeugt das finale Bild direkt
- Phase 10 (STACKING) übernimmt das rekonstruierte Bild unverändert
- Validierung wird trotzdem durchgeführt

## Qualitätsmetriken

### Globale Metriken (Phase 4: GLOBAL_METRICS)
- **B_f**: Hintergrundniveau des normalisierten Frames (niedriger = besser)
- **σ_f**: Rauschen (niedriger = besser)
- **E_f**: Gradientenergie / Sobel-basiert (höher = besser)
- **Q_f**: Gewichteter Score = α·(-B̃) + β·(-σ̃) + γ·Ẽ (MAD-normalisiert)
- **G_f**: Globales Gewicht = exp(Q_f) mit Clamping [-3, +3]

### Lokale Metriken (Phase 6: LOCAL_METRICS)
- **FWHM**: Full Width Half Maximum (Seeing-Qualität)
- **Roundness**: Sternrundheit (Tracking-Qualität)
- **Contrast**: Lokaler Kontrast (Signal-Stärke)
- **Sharpness**: Schärfe-Metrik
- **Star Count**: Anzahl erkannter Sterne pro Tile
- **Tile Type**: STAR (genug Sterne) vs. STRUCTURE (zu wenige Sterne)
- **L_f,t**: Lokales Gewicht = exp(clip(Q_local))

### Effektives Gewicht
- **W_f,t = G_f × L_f,t**
- Kombiniert Frame-Qualität mit lokaler Tile-Qualität
- Verwendet in Phase 7 für Tile-Rekonstruktion

## Mathematische Notation

```
Indizes:
  f - Frame-Index (0..N-1)
  t - Tile-Index (0..T-1)
  k - Cluster-Index (0..K-1)

Dimensionen:
  N  - Anzahl Frames (nach Linearitäts-Rejection)
  K  - Anzahl Cluster / synthetische Frames
  T  - Anzahl Tiles im Grid
  W,H - Bildbreite/-höhe in Pixel
  F  - Seeing-FWHM in Pixel

Normalisierung:
  I_f      - Original-Frame
  B_f      - Background-Level (OSC: B_r, B_g, B_b separat)
  I'_f     - Normalisiertes Frame = I_f / B_f
  s_f      - NormalizationScale = 1 / B_f

Globale Gewichte:
  B̃_f, σ̃_f, Ẽ_f - MAD-normalisierte Metriken
  Q_f = α·(-B̃_f) + β·(-σ̃_f) + γ·Ẽ_f
  G_f = exp(clip(Q_f, -3, +3))

Lokale Gewichte (STAR-Tiles):
  Q_f,t = w_fwhm·(-FWHM̃) + w_round·R̃ + w_contrast·C̃
  L_f,t = exp(clip(Q_f,t))

Lokale Gewichte (STRUCTURE-Tiles):
  Q_f,t = w_metric·(Ẽ/σ̃) + w_bg·(-B̃)
  L_f,t = exp(clip(Q_f,t))

Effektives Gewicht:
  W_f,t = G_f × L_f,t

Tile-Rekonstruktion:
  tile_t = Σ_f W_f,t · tile_f,t / Σ_f W_f,t
  recon  = overlap_add(hanning(tile_t))

Zustandsvektor (Clustering):
  v_f = [G_f, ⟨Q_local⟩_f, Var(Q_local)_f, CC̄_tiles, WarpVar̄, invalid_frac]

Synthetische Frames:
  synth_k = Σ_{f∈cluster_k} G_f · warp(I'_f) / Σ G_f
```

## Artifact-Dateien

Jeder Run erzeugt 9 JSON-Artefakt-Dateien in `<run_dir>/artifacts/`:

| Datei | Phase | Inhalt |
|-------|-------|--------|
| `normalization.json` | 3 | Mode, Bayer, B_mono/B_r/B_g/B_b pro Frame |
| `global_metrics.json` | 4 | Background, Noise, Gradient, Quality, G_f pro Frame |
| `tile_grid.json` | 5 | Image-Dimensionen, Tile-Liste (x,y,w,h), FWHM, Overlap |
| `global_registration.json` | 1 | Warp-Matrizen (a00,a01,tx,a10,a11,ty) + CC pro Frame |
| `local_metrics.json` | 6 | Tile-Metriken pro Frame×Tile, Tile-Typ, Quality, L_f,t |
| `tile_reconstruction.json` | 7 | Valid-Counts, Mean-CC, Post-Contrast/BG/SNR pro Tile |
| `state_clustering.json` | 8 | Cluster-Labels, Cluster-Sizes, Methode |
| `synthetic_frames.json` | 9 | Anzahl synthetische Frames, frames_min/max |
| `validation.json` | 10 | FWHM-Improvement, Tile-Weight-Var, Pattern-Ratio |

### Report-Generierung und auswertbare Daten

Zur konsolidierten Analyse eines Runs steht der Report-Generator
`tile_compile_cpp/generate_report.py` zur Verfügung.

Aufruf:

```text
python tile_compile_cpp/generate_report.py runs/<run_id>
```

Erzeugte Ausgaben:
- `artifacts/report.html`
- `artifacts/report.css`
- `artifacts/*.png` (Diagramme/Heatmaps)

Verwendete Eingabedaten:
- Artifact-JSONs: `normalization.json`, `global_metrics.json`, `tile_grid.json`,
  `global_registration.json`, `local_metrics.json`, `tile_reconstruction.json`,
  `state_clustering.json`, `synthetic_frames.json`, `validation.json`
- Lauf-Events: `logs/run_events.jsonl`
- Run-Konfiguration: `config.yaml` (wird im Report eingebettet)

Typisch auslesbare Inhalte:
- Normalisierungs- und Hintergrundverläufe (Mono/RGB)
- Globale Qualitätsmetriken und Gewichtsverteilungen
- Sternmetriken (u. a. FWHM, wFWHM, Rundheit, Sternanzahl)
- Registrierungsauswertung (Shift/Rotation/Korrelation)
- Tile-/Rekonstruktions-Heatmaps und lokale Qualitätsindikatoren
- Clustering- und Synthetic-Frame-Übersichten
- Validation-Ergebnisse (inkl. Tile-Pattern-Indikatoren)
- Pipeline-Timeline und Frame-Usage-Funnel

## Verzeichnis-Struktur

```
runs/<run_id>/
├── config.yaml           # Kopie der Lauf-Konfiguration
├── logs/
│   └── run_events.jsonl  # Alle Pipeline-Events (JSONL)
├── artifacts/
│   ├── normalization.json
│   ├── global_metrics.json
│   ├── tile_grid.json
│   ├── global_registration.json
│   ├── local_metrics.json
│   ├── tile_reconstruction.json
│   ├── state_clustering.json
│   ├── synthetic_frames.json
│   ├── validation.json
│   ├── report.html       # (via generate_report.py)
│   └── *.png             # Chart-Bilder
└── outputs/
    ├── stacked.fits
    ├── reconstructed_L.fit
    ├── stacked_rgb.fits       # (nur OSC)
    ├── reconstructed_R.fit    # (nur OSC)
    ├── reconstructed_G.fit    # (nur OSC)
    ├── reconstructed_B.fit    # (nur OSC)
    └── synthetic_*.fit        # (Normal Mode)
```

## Performance-Optimierungen (C++)

- **Eigen-Matrizen**: Vektorisierte Pixel-Operationen (SIMD)
- **OpenCV**: Optimierte Bildverarbeitung (Sobel, Laplacian, warpAffine)
- **Thread-Parallelismus**: TILE_RECONSTRUCTION mit std::thread Workerpool
- **Pre-Warping**: Einmaliges Warping aller Frames statt pro-Tile
- **2× Downsample**: Registrierung auf halber Auflösung (Speedup ~4×)
- **Memory-effizient**: Frames werden per-Phase von Disk geladen
- **cv::setNumThreads(1)**: Verhindert OpenCV-Thread-Contention in parallelen Tiles

## Referenzen

- **Normative Spezifikation**: `/doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.2.md`
- **C++ Implementierung**: `/tile_compile_cpp/apps/runner_main.cpp`
- **Konfiguration**: `/tile_compile_cpp/include/tile_compile/config/configuration.hpp`
- **Report-Generator**: `/tile_compile_cpp/generate_report.py`

## Änderungshistorie

- **2026-02-08**: Vollständige Aktualisierung auf C++ Implementierung (runner_main.cpp)
- **2026-02-08**: Pre-Warping Architektur dokumentiert (CFA-Artefakt-Fix)
- **2026-02-08**: 5-stufige Registrierungskaskade dokumentiert
- **2026-02-08**: Artifact-Dateien und Verzeichnisstruktur hinzugefügt
- **2026-01-17**: Cosmetic Correction und Sigma-Clipping Stacking dokumentiert
- **2026-01-09**: Initiale Erstellung der Process Flow Dokumentation

---

**Hinweis**: Diese Dokumentation beschreibt den **tatsächlichen C++ Code**. Bei Widersprüchen zur normativen Spezifikation gilt der Code als Referenz für das Verhalten, die Spezifikation als Referenz für die Intention.
