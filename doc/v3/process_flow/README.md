# Process Flow Documentation — C++ Pipeline (tile_compile_cpp)

## Übersicht

Diese Dokumentation beschreibt den **tatsächlichen Ablauf der C++ Implementierung** (`tile_compile_cpp/apps/runner_main.cpp`) der Tile-basierten Qualitätsrekonstruktion für Deep-Sky-Objekte gemäß Methodik v3.

Die Pipeline verarbeitet **FITS-Frames** (Mono oder OSC/CFA) und erzeugt ein gewichtetes, Tile-basiertes Rekonstruktionsergebnis mit optionaler Clusterung, synthetischen Frames und Sigma-Clipping-Stacking.

**Implementierung:** C++ mit Eigen, OpenCV, cfitsio, nlohmann/json, YAML-cpp.

## Aktuelle Pipeline-Phasen (C++ Implementierung)

| Phase | Enum | Name | Quelldatei-Zeilen | Beschreibung |
|-------|------|------|-------------------|--------------|
| 0 | `SCAN_INPUT` | Input-Scan | L196–L368 | FITS lesen, Bayer erkennen, Linearitätsprüfung |
| 1 | `CHANNEL_SPLIT` | Kanaltrennung | L370–L383 | Metadaten-Phase (eigentliche Trennung deferred) |
| 2 | `NORMALIZATION` | Normalisierung | L385–L634 | Hintergrund-Division pro Frame/Kanal |
| 3 | `GLOBAL_METRICS` | Globale Metriken | L636–L717 | Background, Noise, Gradient → G_f |
| 4 | `TILE_GRID` | Tile-Erzeugung | L719–L818 | FWHM-adaptive Tile-Größe + Grid |
| — | *Pre-warp* | Vollbild-Warping | L1132–L1158 | Alle Frames werden vor Tile-Extraktion gewarpt |
| 5 | `REGISTRATION` | Registrierung | L902–L1130 | 5-stufige Kaskade + globaler Warp |
| 6 | `LOCAL_METRICS` | Lokale Metriken | L1162–L1374 | Tile-Typ-Klassifikation + robuste Scores |
| 7 | `TILE_RECONSTRUCTION` | Rekonstruktion | L1381–L1647 | Parallele gewichtete Overlap-Add-Rekonstruktion |
| 8 | `STATE_CLUSTERING` | Clusterung | L1649–L1892 | 6D-Zustandsvektor K-Means |
| 9 | `SYNTHETIC_FRAMES` | Synthetische Frames | L1894–L2052 | Gewichtetes Cluster-Stacking → K Frames |
| 10 | `STACKING` | Finales Stacking | L2054–L2085 | Sigma-Clip oder Mean von synthetischen Frames |
| — | *Validation* | Qualitätsprüfung | L2087–L2223 | FWHM-Verbesserung, Gewichtsvarianz, Tile-Pattern |
| 11 | `DEBAYER` | Demosaicing | L2225–L2260 | CFA → R/G/B + RGB-FITS-Cube (nur OSC) |
| 12 | `DONE` | Finalisierung | L2262–L2277 | Status schreiben, Abschluss |

## Dokumenten-Struktur

### [Phase 0: SCAN_INPUT — Input-Scan und Linearitätsprüfung](phase_0_overview.md)
- FITS-Frame-Discovery und Dimensionserkennung
- Farbmodus-Erkennung (MONO vs. OSC) und Bayer-Pattern
- **Linearitätsprüfung** mit konfigurierbarer Strictness
- Frame-Rejection bei `linear_required=true`
- Run-Verzeichnis und Event-Logging Setup

**Output:** Validierte Frame-Liste, erkannter Modus, Linearitäts-Info

---

### [Phase 1: REGISTRATION — Kaskadierte Registrierung](phase_1_registration.md)
- **Referenz-Frame-Auswahl** nach globalem Gewicht (best G_f)
- **2×-Downsample** für Registrierung (CFA: Green-Proxy; Mono: Mean)
- **5-stufige Registrierungskaskade** mit robusten Fallbacks:
  1. Triangle Star Matching (rotationsinvariant)
  2. Star Pair Matching (kleine Shifts)
  3. AKAZE Feature Matching
  4. Hybrid Phase-Correlation + ECC
  5. Identity-Fallback (Frame bleibt drin, CC=0)
- **Pre-Warping** aller Frames auf Vollbild-Auflösung (CFA-aware)
- **Keine Frame-Selektion**: Jeder Frame wird behalten (v3-Regel)

**Output:** Affine Warp-Matrizen + CC pro Frame, pre-warped Frames

---

### [Phase 2: NORMALIZATION + GLOBAL_METRICS — Normalisierung und globale Gewichtung](phase_2_normalization_metrics.md)
- **Sigma-Clipping Background-Maske** auf grob normalisiertem Bild
- OSC: kanalgetrennte Background-Schätzung (R, G, B via Bayer-Offsets)
- Mono: einzelne Background-Schätzung
- **Division durch Background** → normalisierter Frame
- Globale Metriken: Background B_f, Noise σ_f, Gradient Energy E_f
- **Gewicht G_f = exp(α·(-B̃) + β·(-σ̃) + γ·Ẽ)** mit MAD-Normalisierung

**Output:** NormalizationScales pro Frame, globale Gewichte G_f

---

### [Phase 3: TILE_GRID — Seeing-adaptive Tile-Erzeugung](phase_3_tile_generation.md)
- **FWHM-Messung** auf zentralem 1024×1024 ROI (bis zu 5 Probeframes)
- Tile-Größe: `T = clip(size_factor × FWHM, min_size, max(image)/max_divisor)`
- Overlap-Fraction (konfigurierbar, Clamping auf [0, 0.5])
- Stride = Tile-Size − Overlap-Px
- **Einheitliches reguläres Grid** über das gesamte Bild

**Output:** Tile-Liste mit (x, y, width, height), Grid-Konfiguration

---

### [Phase 4: LOCAL_METRICS — Lokale Tile-Metriken und Qualitäts-Scoring](phase_4_local_metrics.md)
- **Tile-Metriken pro Frame × Tile**: FWHM, Roundness, Contrast, Sharpness, Background, Noise, Gradient Energy, Star Count
- **Tile-Typ-Klassifikation** nach Median Star Count: `STAR` vs. `STRUCTURE`
- **Robuste Normalisierung** (Median + MAD → z-Score) pro Tile über alle Frames
- Qualitäts-Score:
  - STAR-Tiles: `Q = w_fwhm·(-FWHM̃) + w_round·R̃ + w_contrast·C̃`
  - STRUCTURE-Tiles: `Q = w_metric·(Ẽ/σ̃) + w_bg·(-B̃)`
- Clamping + Exponential-Mapping: `L_f,t = exp(clip(Q, clamp_lo, clamp_hi))`

**Output:** Lokale Gewichte L_f,t pro Frame × Tile

---

### [Phase 5: TILE_RECONSTRUCTION — Parallele Tile-Rekonstruktion](phase_5_tile_reconstruction.md)
- **Effektives Gewicht**: W_f,t = G_f × L_f,t
- **Gewichtete Summe**: `tile_rec = Σ W_f,t · tile_f / Σ W_f,t`
- **Per-Tile Background-Normalisierung** gegen Referenz-Tile
- **Hanning-Fensterfunktion** (2D separabel) für Overlap-Add
- **Thread-parallele Verarbeitung** (bis zu N CPU-Cores)
- Post-Metriken pro Tile: Contrast (Laplacian), Background (Median), SNR (P99/MAD)

**Output:** Rekonstruiertes Bild (recon), Post-Tile-Metriken

---

### [Phase 6: STATE_CLUSTERING — Zustandsbasierte Clusterung](phase_6_clustering.md)
- **6D-Zustandsvektor**: `v_f = [G_f, mean(Q_local), var(Q_local), mean_cc, mean_warp_var, invalid_fraction]`
- **z-Score Normalisierung** aller Dimensionen
- **K-Means Clusterung** (20 Iterationen, K = clip(N/10, k_min, k_max))
- **Quantile-Fallback** bei degenerierten (leeren) Clustern
- Optional: Skip in **Reduced Mode** wenn N < frames_reduced_threshold

**Output:** Cluster-Labels pro Frame, Cluster-Größen

---

### [Phase 7: SYNTHETIC + STACKING + VALIDATION + DEBAYER](phase_7_final_stacking.md)
- **Synthetische Frames**: Global-gewichtetes Mittel pro Cluster
  - Nur Cluster mit `count ≥ frames_min`
  - Maximal `frames_max` synthetische Frames
- **Stacking**: Sigma-Clipping Rejection oder Mean
- **Validation**:
  - FWHM-Verbesserung (Output vs. Seeing)
  - Tile-Weight-Varianz (ausreichende Diskriminierung)
  - Tile-Pattern-Detektion (Sobel-Gradient an Tile-Grenzen)
- **Debayer** (nur OSC): Nearest-Neighbor Demosaic → R/G/B FITS + RGB-Cube
- **Output-Skalierung**: Multiplikation mit Median-Background + Pedestal (32768)

**Output:** stacked.fits, reconstructed_L.fit, stacked_rgb.fits (OSC), validation.json

---

## Pipeline-Flussdiagramm (C++ Implementierung)

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
              │  PHASE 1: CHANNEL_SPLIT     │
              │  (metadata only — deferred)  │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 2: NORMALIZATION     │
              │  • Sigma-clip BG mask        │
              │  • OSC: per-channel BG       │
              │  • MONO: single BG           │
              │  • Scale = 1/Background      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 3: GLOBAL_METRICS    │
              │  • B_f, σ_f, E_f per frame   │
              │  • MAD-normalize → z-scores  │
              │  • G_f = exp(α·B̃+β·σ̃+γ·Ẽ)   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 4: TILE_GRID         │
              │  • FWHM probe (central ROI)  │
              │  • T = clip(s·F, min, max)   │
              │  • Overlap + stride calc     │
              │  • Uniform tile grid         │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  PHASE 5: REGISTRATION      │
              │  • Ref = best G_f frame      │
              │  • 2× downsample (CFA-aware) │
              │  • 5-stage cascade:          │
              │    1. Triangle star match    │
              │    2. Star pair matching     │
              │    3. AKAZE features         │
              │    4. Phase-corr + ECC       │
              │    5. Identity fallback      │
              │  • Pre-warp all frames       │
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
              │  • Output scaling + pedestal │
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
              │  PHASE 12: DONE             │
              │  • Final status emit         │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  OUTPUTS:                   │
              │  • stacked.fits              │
              │  • reconstructed_L.fit       │
              │  • stacked_rgb.fits (OSC)    │
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

### Normal Mode (N ≥ frames_reduced_threshold)
- Alle 12 Phasen werden durchlaufen
- State-Clustering aktiv
- Synthetische Frames werden erzeugt
- Sigma-Clipping Rejection Stacking
- Optimale Rauschreduktion

### Reduced Mode (N < frames_reduced_threshold)
- Phase 8 (STATE_CLUSTERING) wird **übersprungen**
- Phase 9 (SYNTHETIC_FRAMES) wird **übersprungen**
- Phase 7 (TILE_RECONSTRUCTION) erzeugt das finale Bild direkt
- Phase 10 (STACKING) übernimmt das rekonstruierte Bild unverändert
- Validierung wird trotzdem durchgeführt

## Qualitätsmetriken

### Globale Metriken (Phase 3: GLOBAL_METRICS)
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
| `normalization.json` | 2 | Mode, Bayer, B_mono/B_r/B_g/B_b pro Frame |
| `global_metrics.json` | 3 | Background, Noise, Gradient, Quality, G_f pro Frame |
| `tile_grid.json` | 4 | Image-Dimensionen, Tile-Liste (x,y,w,h), FWHM, Overlap |
| `global_registration.json` | 5 | Warp-Matrizen (a00,a01,tx,a10,a11,ty) + CC pro Frame |
| `local_metrics.json` | 6 | Tile-Metriken pro Frame×Tile, Tile-Typ, Quality, L_f,t |
| `tile_reconstruction.json` | 7 | Valid-Counts, Mean-CC, Post-Contrast/BG/SNR pro Tile |
| `state_clustering.json` | 8 | Cluster-Labels, Cluster-Sizes, Methode |
| `synthetic_frames.json` | 9 | Anzahl synthetische Frames, frames_min/max |
| `validation.json` | 10 | FWHM-Improvement, Tile-Weight-Var, Pattern-Ratio |

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

- **Normative Spezifikation**: `/doc/tile_basierte_qualitatsrekonstruktion_methodik_v_3.md`
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
