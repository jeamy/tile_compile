# Process Flow Documentation — AQMH-Pipeline (tile_compile_cpp)

## Übersicht

Diese Dokumentation beschreibt primär den **tatsächlichen AQMH-Ablauf** der C++-Implementierung (`tile_compile_cpp/apps/runner_pipeline.cpp`). Die klassische Tile-Compile-Rekonstruktion ist am Ende als separate Alternative abgegrenzt.

Die aktuelle Standardpipeline verarbeitet **FITS-Frames** (Mono oder OSC/CFA) und erzeugt eine pixelweise AQMH-Rekonstruktion. Sie verwendet keine klassischen lokalen Tile-Qualitätsmetriken, kein Clustering und keine synthetischen Frames.

**Implementierung:** C++ mit Eigen, OpenCV, cfitsio, nlohmann/json, YAML-cpp.

**GUI3-Integration:** Der produktive GUI-Pfad nutzt das Web-Frontend plus Crow/C++ Backend. Crow orchestriert die C++ Pipeline über `tile_compile_cli` und `tile_compile_runner`; die Verarbeitungslogik bleibt vollständig im C++ Kern.

> **Standardmethode AQMH:** AQMH ersetzt die klassischen lokalen Tile-Metriken und die Tile-Rekonstruktion durch Quality-Maps und eine unabhängige pixelweise Rekonstruktion. Clustering und synthetische Frames werden im AQMH-Pfad übersprungen.

Die verbindlichen Resume-Einstiege, Artefakt- und Cache-Abhängigkeiten stehen
in [Resume-Abhängigkeiten](resume_dependencies_de.md). Besonders wichtig:
frühe Resume-Phasen lösen einen In-Place-Vollständigkeitslauf aus; sie sind
keine Fortsetzung ab der genannten Phase.

## Aktuelle AQMH-Phasen (C++ Implementierung)

Quelle der Phasenreihenfolge: `tile_compile::Phase` in `include/tile_compile/core/types.hpp`.

| ID | Enum | Kurzbeschreibung |
|----|------|------------------|
| 0 | `SCAN_INPUT` | Input-Scan, Header/Modus-Erkennung, Linearitätsprüfung, Disk-Space-Precheck (`scandir*4`) |
| 1 | `REGISTRATION` | Globale Registrierung (kaskadiert), Warp-Qualität / CC |
| 2 | `PREWARP` | Vollbild-Prewarp auf gemeinsamen Canvas (CFA-safe bei OSC) |
| 3 | `CHANNEL_SPLIT` | Metadaten-Phase (OSC/Mono-Kanalmodell; tatsächliche Kanalarbeit erfolgt später) |
| 4 | `NORMALIZATION` | Globale lineare Normalisierung (additiver Hintergrund + photometrische Skala) |
| 5 | `GLOBAL_METRICS` | Globale Frame-Metriken und Gewichte `G_f` |
| 6 | `TILE_GRID` | Adaptive Tile-Geometrie (Seeing/FWHM-basiert) |
| 7 | `COMMON_OVERLAP` | Gemeinsamer datentragender Bereich (global/tile-lokal) |
| 19 | `AQMH_MAPS` | Pyramidale, pixelweise Quality-Maps pro Frame; schreibt Maps und Frame-Diagnostik in `cache/aqmh/` und `artifacts/aqmh_metrics.json` |
| 20 | `AQMH_GLOBAL_QUALITY` | Globale Frame-Gewichte `G_f` aus Schärfe, SNR und Hintergrundstrafe |
| 21 | `AQMH_RECONSTRUCTION` | Pixelweise gewichtete Rekonstruktion mit Support-Maske, robustem Clipping und unverändertem Raw-CFA-Artefakt |
| 22 | `AQMH_DIAGNOSTICS` | Blockdiagnostik, Heatmaps und Rekonstruktionsmetriken |
| 12 | `STACKING` | Finales lineares Stacking (inkl. robuster Pixel-Ausreißerbehandlung) |
| 13 | `DEBAYER` | OSC-Debayering und RGB-Ausgabe (bei MONO: pass-through) |
| 14 | `ASTROMETRY` | Plate Solving / WCS |
| 15 | `BGE` | Optionale Background Gradient Extraction auf RGB vor PCC |
| 16 | `PCC` | Photometric Color Calibration |
| 17 | `HYPERMETRIC_STRETCH` | VeraLux HyperMetric Stretch nach PCC |
| 18 | `DONE` | Abschlussstatus (`ok` oder `validation_failed`) |

`STATE_CLUSTERING` und `SYNTHETIC_FRAMES` werden im AQMH-Pfad als
`skipped` markiert bzw. nicht als ausführende AQMH-Stufen verwendet.
`AQMH_BGE_INPUTS` ist aktuell nur als Enum-Wert definiert und wird im normalen
Runner nicht als eigene ausführende Phase emittiert.

Hinweis: **Validation** ist ein Qualitätsblock zwischen `STACKING` und `DEBAYER`, aber keine eigene Enum-Phase.
Hinweis: **BGE** ist eine optionale **eigene Phase** zwischen `ASTROMETRY` und `PCC`.

## Dokumenten-Struktur

Die Details der AQMH-spezifischen Erweiterungsphasen stehen in
[AQMH-Erweiterungen](phase_8_aqmh_extensions.md).

Die **verbindliche AQMH-Ausführungsreihenfolge** ist die oben stehende Liste:
gemeinsame Vorverarbeitung, `AQMH_MAPS`, `AQMH_GLOBAL_QUALITY`,
`AQMH_RECONSTRUCTION`, `AQMH_DIAGNOSTICS` und danach die gemeinsame
Finalisierung. Die numerischen Enum-Werte 19–22 sind AQMH-spezifische
Phasenwerte und keine Fortsetzung der alten Classic-Phase 8/9.

Kurzzuordnung:

- Input + Modus + Linearität + Disk-Precheck → `SCAN_INPUT`
- Registrierung/Prewarp → `REGISTRATION`, `PREWARP`
- Normalisierung + Canvas-Masken → `NORMALIZATION` bis `COMMON_OVERLAP`
- AQMH-Analyse → `AQMH_MAPS` → `AQMH_GLOBAL_QUALITY`
- AQMH-Rekonstruktion → `AQMH_RECONSTRUCTION` → `AQMH_DIAGNOSTICS`
- Nicht verwendete Classic-Stufen → `STATE_CLUSTERING`, `SYNTHETIC_FRAMES` (`skipped`)
- Finalisierungspfad → `STACKING`, `DEBAYER`, `ASTROMETRY`, `BGE` (optional ausführend, aber eigene Phase), `PCC`, `HYPERMETRIC_STRETCH`, `DONE`

---

## Pipeline-Flussdiagramm (C++ Implementierung, v3.3)

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
              │  • Kaskadierte Fallbacks     │
              │  • CC/Warp-Metriken          │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 2: PREWARP            │
              │  • Full-frame Canvas-Warp    │
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
              │  • Additiver Hintergrund B_f │
              │  • Photometrische Skala P_f  │
              │  • I = (I_raw - B_f) / P_f   │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 5: GLOBAL_METRICS     │
              │  • B_f, σ_f, E_f per frame   │
              │  • MAD-normalize → z-scores  │
              │  • G_f = exp(α·B̃+β·σ̃+γ·Ẽ)    │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 6: TILE_GRID          │
              │  • FWHM probe (central ROI)  │
              │  • T = clip(s·F, min, max)   │
              │  • Overlap + stride calc     │
              │  • Uniform tile grid         │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 7: COMMON_OVERLAP     │
              │  • Pixelwise valid overlap   │
              │  • common_overlap.json       │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 19: AQMH_MAPS          │
              │  • Pyramidale Quality-Maps    │
              │  • Pixelweise Frame-Diagnostik│
              │  • cache/aqmh/                │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 20: AQMH_GLOBAL_QUALITY│
              │  • Schärfe-/SNR-Zusammenfassung│
              │  • Hintergrundstrafe          │
              │  • Globales Frame-Gewicht G_f │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 21: AQMH_RECONSTRUCTION│
              │  • Pixelweise Gewichtung      │
              │  • Support-/Valid-Masken      │
              │  • Robust Welford/Sigma-Clip  │
              │  • Raw-CFA-Artefakt speichern  │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 22: AQMH_DIAGNOSTICS   │
              │  • Blockdiagnostik            │
              │  • Heatmaps / Map-Statistik    │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 12: STACKING          │
              │  • Sigma-clip rejection      │
              │  • Or mean of synth frames   │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  VALIDATION                  │
              │  • FWHM improvement check    │
              │  • Tile weight variance      │
              │  • Tile pattern detection    │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  PHASE 13: DEBAYER           │
              │  • OSC: NN demosaic → RGB    │
              │  • MONO: pass-through        │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 14: ASTROMETRY       │
              │  • ASTAP, lokaler Gaia-Fallback│
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 15: BGE              │
              │  • Optional vor PCC         │
              │  • Gradienten-Subtraktion   │
              │  • artifacts/bge.json       │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 16: PCC              │
              │  • Photometrische Farbkal.  │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 17: HMS              │
              │  • VeraLux HyperMetric      │
              │  • Stretch nach PCC         │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 18: DONE             │
              │  • Final status emit        │
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
              │  • R/G/B .fit (OSC)         │
              │  • 12 artifact JSON files   │
              │  • run_events.jsonl         │
              └─────────────────────────────┘
```

## Kernprinzipien (C++ Implementierung)

1. **Linearität im Kern**: Bis einschließlich PCC bleiben die Core-Phasen linear; HMS ist eine explizite finale Stretch-Phase nach PCC.
2. **Pixelweise Qualitätsauswahl**: AQMH berechnet pro Frame und Pixel Quality-Maps und verwendet globale Frame-Gewichte; es gibt keine Classic-Tile-Gewichte.
3. **Mono + OSC**: Beide Modi in einer einzigen Pipeline, CFA-aware bei OSC
4. **Streng sequentiell**: Keine Rückkopplungen, deterministische Abfolge
5. **AQMH statt Classic-Tiles**: Die Rekonstruktion arbeitet pixelweise auf Quality-Maps, nicht mit `L_f,t`-Tile-Gewichten.
6. **Global × Pixel**: Das globale Frame-Gewicht wird mit der pixelweisen AQMH-Qualität kombiniert.
7. **Pre-Warping**: Alle Frames werden vor der Quality-Map-Berechnung vollständig auf den gemeinsamen Canvas gewarpt.
8. **Robuste Statistik**: Median, MAD, Sigma-Clipping durchgehend

## Modi

### AQMH in jedem Modus
- `AQMH_MAPS`, `AQMH_GLOBAL_QUALITY`, `AQMH_RECONSTRUCTION` und `AQMH_DIAGNOSTICS` bilden den AQMH-Analyse- und Rekonstruktionspfad.
- `STATE_CLUSTERING` und `SYNTHETIC_FRAMES` werden im AQMH-Modus nicht als Classic-Verarbeitung verwendet.
- `STACKING` übernimmt die AQMH-Rekonstruktion; danach folgen `DEBAYER`, `ASTROMETRY`, `BGE`, `PCC` und HMS.
- Die Frame-Anzahl steuert weiterhin Reduced-/Emergency-Gates und Ressourcen, ändert aber nicht die AQMH-Qualitätslogik in Classic-Clustering.

### Classic-Alternative
Nur bei `method: classic_tile_compile` werden `LOCAL_METRICS`,
`TILE_RECONSTRUCTION`, `STATE_CLUSTERING` und `SYNTHETIC_FRAMES` ausgeführt.

## Qualitätsmetriken

### AQMH globale Frame-Qualität (`AQMH_GLOBAL_QUALITY`)
- **B_f**: Hintergrundniveau des normalisierten Frames (niedriger = besser)
- **σ_f**: Rauschen (niedriger = besser)
- **E_f**: Gradientenergie / Sobel-basiert (höher = besser)
- **Q_f**: Gewichteter Score = α·(-B̃) + β·(-σ̃) + γ·Ẽ (MAD-normalisiert)
- **G_f**: globales Frame-Gewicht aus den AQMH-Zusammenfassungen, begrenzt auf den konfigurierten Gewichtsbereich

### AQMH Quality-Maps (`AQMH_MAPS`)
- Pyramidale lokale Varianz-/Schärfeinformationen pro Frame und Pixel
- SNR- und Hintergrundstrafe pro Frame
- Artefakt- und Support-Masken
- Persistenz der Maps unter `cache/aqmh/`

### AQMH-Rekonstruktion (`AQMH_RECONSTRUCTION`)
- Pixelweise Kombination der gültigen Frames mit AQMH-Map und `G_f`
- Robuste Welford-/Sigma-Clipping-Statistik
- Support-aware Ausgabe ohne nachträgliche Classic-Tile-Renormierung

## Mathematische Notation

```
Indizes:
  f - Frame-Index (0..N-1)
  p - Pixel-Index auf dem gemeinsamen Canvas

Dimensionen:
  N  - Anzahl Frames
  W,H - Canvas-Breite/-höhe in Pixel

Normalisierung:
  I_f      - Original-Frame
  B_f      - Background-Level (OSC: B_r, B_g, B_b separat)
  I'_f     - Normalisiertes Frame = I_f / B_f
  I_f^raw  - ursprünglicher linearer Frame
  B_f      - additives Hintergrundniveau
  P_f      - photometrische Skala
  J_f      - hintergrundsubtrahierter Frame = I_f^raw - B_f
  I_f      - normierter Frame = J_f / P_f

AQMH globale Eingaben:
  g_sharp_f       - zusammengefasste Schärfe-/PSF-Qualität
  g_snr_f         - zusammengefasste Signal-Rausch-Qualität
  g_background_f  - Hintergrundstrafe

Globales Frame-Gewicht:
  G_f = compute_aqmh_global_quality(g_sharp_f, g_snr_f, g_background_f)

Pixelweise Rekonstruktion:
  q_f,p = QualityMap(f, p)
  w_f,p = G_f × q_f,p × valid_f,p
  recon_p = robust_weighted_reduce({I_f,p}, {w_f,p})

Die genaue robuste Reduktion berücksichtigt Mindest-Support, effektive
Stichprobengröße, Sigma-Clipping und die konfigurierten AQMH-Gates. Die
Classic-Größen `L_f,t`, Tile-Overlap-Add, Clustering und synthetische Frames
gehören nicht zum AQMH-Rechenpfad.
```

## Artifact-Dateien

Ein AQMH-Run erzeugt die folgenden zentralen Artefakte in
`<run_dir>/artifacts/`; optionale Diagnoseartefakte können hinzukommen:

| Datei | Phase | Inhalt |
|-------|-------|--------|
| `normalization.json` | 4 | Mode, Bayer, B_mono/B_r/B_g/B_b pro Frame |
| `global_metrics.json` | 5 | Globale Normalisierungs-/Inputmetriken pro Frame |
| `tile_grid.json` | 6 | Image-Dimensionen, Tile-Liste (x,y,w,h), FWHM, Overlap |
| `global_registration.json` | 1 | Warp-Matrizen (a00,a01,tx,a10,a11,ty) + CC pro Frame |
| `common_overlap.json` | 7 | Globaler/Tile-weiser gemeinsamer Datenbereich |
| `aqmh_metrics.json` | 8/9 | Quality-Map-Metadaten, Frame-Diagnostik und globale AQMH-Gewichte |
| `aqmh_reconstruction.json` | 10 | AQMH-Rekonstruktionsmetriken und Support-/Clipping-Diagnostik |
| `aqmh_regions.json` / `cache/aqmh_block_diagnostics.jsonl` | 11 | AQMH-Regionen-, Block- und Heatmap-Diagnostik |
| `bge.json` | 15 | Kanalweise BGE-Diagnostik (Samples, Grid-Zellen, Residuen) |
| `validation.json` | 12 | Methodenspezifische Qualitäts- und Supportvalidierung |

`local_metrics.json`, `state_clustering.json` und `synthetic_frames.json`
gehören ausschließlich zur Classic-Alternative und werden bei AQMH nicht als
Qualitätsgrundlage verwendet.

### Report-Generierung und auswertbare Daten

Zur konsolidierten Analyse eines Runs wird der Report über den integrierten CLI-/Backendpfad erzeugt.

Aufruf:

```text
./tile_compile_cli generate-report runs/<run_id>
```

Erzeugte Ausgaben:
- `artifacts/report.html`
- `artifacts/report.css`
- `artifacts/*.png` (Diagramme/Heatmaps)

Verwendete Eingabedaten:
- Artifact-JSONs: `normalization.json`, `global_metrics.json`, `tile_grid.json`,
  `global_registration.json`, `common_overlap.json`, `aqmh_metrics.json`,
  `aqmh_reconstruction.json`, `aqmh_regions.json`, AQMH-Blockdiagnostik,
  `bge.json`, `validation.json`
- Lauf-Events: `logs/run_events.jsonl`
- Run-Konfiguration: `config.yaml` (wird im Report eingebettet)

Typisch auslesbare Inhalte:
- Normalisierungs- und Hintergrundverläufe (Mono/RGB)
- AQMH Quality-Map-Statistiken und globale Frame-Gewichte
- Sternmetriken (u. a. FWHM, wFWHM, Rundheit, Sternanzahl)
- Registrierungsauswertung (Shift/Rotation/Korrelation)
- AQMH-Rekonstruktions- und Support-Heatmaps
- Classic-Clustering- und Synthetic-Frame-Übersichten nur bei `classic_tile_compile`
- BGE-Diagnostik (kanalweise Background-Modelle, Grid-Zellen, Residuenhistogramme)
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
│   ├── common_overlap.json
│   ├── aqmh_metrics.json
│   ├── aqmh_reconstruction.json
│   ├── aqmh_regions.json
│   ├── bge.json
│   ├── validation.json
│   ├── report.html       # erzeugt über CLI-/Backend-Reportpfad
│   ├── *.png             # Chart-Bilder
├── cache/
│   ├── normalized_frames/      # Normalisierungs-Cache
│   ├── prewarped_frames/       # Resume-relevanter Prewarp-Cache
│   ├── aqmh/                   # AQMH Quality-Maps und Metadaten
│   ├── aqmh_masks/             # AQMH Frame-Masken
│   └── aqmh_block_diagnostics.jsonl
└── outputs/
    ├── stacked.fits
    ├── reconstructed_L.fit
    ├── stacked_rgb.fits       # (nur OSC)
    ├── reconstructed_R.fit    # (nur OSC)
    ├── reconstructed_G.fit    # (nur OSC)
    ├── reconstructed_B.fit    # (nur OSC)
    └── synthetic_*.fit        # (nur Classic-Modus)
```

## Performance-Optimierungen (C++)

- **Eigen-Matrizen**: Vektorisierte Pixel-Operationen (SIMD)
- **OpenCV**: Optimierte Bildverarbeitung (Sobel, Laplacian, warpAffine)
- **Thread-Parallelismus**: AQMH-Map-/Rekonstruktions-Worker; Classic zusätzlich `TILE_RECONSTRUCTION` mit Workerpool
- **Pre-Warping**: Einmaliges Warping aller Frames statt pro-Tile
- **2× Downsample**: Registrierung auf halber Auflösung (Speedup ~4×)
- **Memory-effizient**: Frames werden per-Phase von Disk geladen
- **cv::setNumThreads(1)**: Verhindert OpenCV-Thread-Contention in parallelen Tiles
- **CUDA-Worker-Streams**: Ein Non-Default-Stream pro parallelem PREWARP-/AQMH-/Tile-Worker
- **Streaming-AQMH-CUDA-Rekonstruktion**: GPU-Akkumulatoren bleiben resident; Frames/Maps werden einzeln übertragen, daher ist der VRAM-Bedarf unabhängig von der Frameanzahl
- **Paralleles RGB-Stacking**: R/G/B-Reduktionen laufen gleichzeitig auf getrennten CUDA-Streams

## GPU-Ausführung nach Phase

| Phase | CUDA | OpenCL | GPU-Arbeit |
|---|---:|---:|---|
| `PREWARP` | Ja | Ja | Vollbild-Affine-/CFA-Warps |
| `AQMH_MAPS` | Ja | Ja | Pyramidale Filter und lokale Varianz |
| `AQMH_RECONSTRUCTION` | Ja | Nein | Gewichtete Welford-Statistik, Masken, Sigma-Clipping, Akkumulation |
| Klassische `TILE_RECONSTRUCTION` | Ja | Ja | Sigma-Clipping und Overlap-Add |
| `SYNTHETIC_FRAMES` | Ja | Ja | Cluster-Tile-Rekonstruktion |
| `STACKING` / Resume | Ja | Ja | Gewichtete/Sigma-Clip-Reduktion, paralleles RGB |

`REGISTRATION` bleibt CPU-only; die GPU-Ausführung beginnt in `PREWARP`.

`runtime_limits.acceleration_backend: auto` wählt pro unterstützter Phase CUDA,
danach OpenCL und zuletzt CPU. AQMH Cherry-Pick nutzt CPU. CUDA-/OpenCL-Fehler
fallen auf CPU zurück. `artifacts/acceleration_context.json` sowie die
Live-Felder `cpu_workers`, `gpu` und `backend` dokumentieren den effektiven Pfad.

## Referenzen

### Normative Spezifikation
- `/docs/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md`

### C++ Implementierung
- `/tile_compile_cpp/apps/runner_pipeline.cpp`
- **Konfiguration**: `/tile_compile_cpp/include/tile_compile/config/configuration.hpp`
- **Report-Generator**: `/web_backend_cpp/src/services/report_generator.cpp`

---

**Hinweis**: Diese Dokumentation beschreibt den **tatsächlichen C++ Code**. Bei Widersprüchen zur normativen Spezifikation gilt der Code als Referenz für das Verhalten, die Spezifikation als Referenz für die Intention.
