# Process Flow Documentation - Tile-basierte Qualitätsrekonstruktion

## Übersicht

Diese Dokumentation beschreibt detailliert den Ablauf der **Tile-basierten Qualitätsrekonstruktion für DSO (Deep Sky Objects)** gemäß Methodik v3.

Jede Phase ist in einem separaten Dokument mit Diagrammen, Formeln und Beispielen dokumentiert.

## Aktuelle Pipeline-Phasen (Implementierung)

| Phase | Name | Beschreibung |
|-------|------|--------------|
| 0 | SCAN_INPUT | Input-Frames scannen, Bayer-Pattern erkennen |
| 1 | REGISTRATION | Cosmetic Correction + ECC-Registration + Warp |
| 2 | CHANNEL_SPLIT | CFA → R/G/B Kanaltrennung |
| 3 | NORMALIZATION | Hintergrund-Normalisierung pro Kanal |
| 4 | GLOBAL_METRICS | Globale Qualitätsmetriken (Background, Noise, Gradient) |
| 5 | TILE_GRID | FWHM-basierte Tile-Grid-Erzeugung |
| 6 | LOCAL_METRICS | Lokale Tile-Metriken |
| 7 | TILE_RECONSTRUCTION | Gewichtete Tile-Rekonstruktion |
| 8 | STATE_CLUSTERING | Zustandsbasierte Frame-Clusterung |
| 9 | SYNTHETIC_FRAMES | Synthetische Frames pro Cluster |
| 10 | STACKING | Sigma-Clipping Rejection Stacking |
| 11 | DEBAYER | CFA → RGB Demosaicing |
| 12 | DONE | Finalisierung und Report |

## Dokumenten-Struktur

### [Phase 0: Pipeline-Übersicht & Vorverarbeitungspfade](phase_0_overview.md)
- Gesamtübersicht der Pipeline
- **Path A: Siril-basiert** (legacy)
- **Path B: CFA-basiert** (opencv_cfa, empfohlen)
- Vergleich der beiden Pfade
- Übergabepunkt an gemeinsamen Kern

**Wichtige Konzepte:**
- Zwei gleichwertige Vorverarbeitungspfade
- Ab Phase 2 identische Verarbeitung
- Debayer-Zeitpunkt unterschiedlich

---

### [Phase 1: Registrierung und Kanaltrennung](phase_1_registration.md)
- **Cosmetic Correction** (NEU): Hotpixel-Entfernung VOR Warp
- **Path A:** Siril Debayer → Registration → Channel Split (legacy)
- **Path B:** CFA Luminance → Registration (opencv_cfa) → CFA-aware Transform → Debayer → Channel Split
- FWHM-Messung und Sternfindung
- Transformationsschätzung (RANSAC/ECC)
- Qualitätsmetriken (Registrierungsresiduum, Elongation)

**Wichtige Konzepte:**
- **Cosmetic Correction verhindert "Walking Noise"** durch Hotpixel-Interpolation vor Warp
- Sub-Pixel-Registrierung
- Kanalgetrennte Verarbeitung ab hier
- Keine Farbmischung bei Resampling (Path B)

**Output:** 3 separate Kanal-Stacks (R, G, B)

---

### [Phase 2: Globale Normalisierung und Frame-Metriken](phase_2_normalization_metrics.md)
- Hintergrundschätzung (Sigma-Clipping)
- **Globale Normalisierung** (Division durch Hintergrund)
- Rauschschätzung
- Gradientenergie (Sobel-Operator)
- **Globaler Qualitätsindex** Q_f,c und Gewicht G_f,c

**Wichtige Konzepte:**
- Normalisierung erfolgt **exakt einmal**
- Robuste Normalisierung der Metriken mit Median + MAD
- Exponential-Mapping mit Clamping
- Gewichtung: α·(-B̃) + β·(-σ̃) + γ·Ẽ

**Output:** Normalisierte Frames + globale Gewichte

---

### [Phase 3: Tile-Erzeugung (FWHM-basiert)](phase_3_tile_generation.md)
- **FWHM-Messung** (Full Width Half Maximum)
- PSF-Fitting (2D Gaussian)
- Robuste FWHM-Schätzung (Median)
- **Seeing-adaptive Tile-Größe:** T = clip(s·F, T_min, T_max)
- Overlap-Berechnung
- Tile-Grid-Erzeugung
- Tile-Klassifikation (Stern/Struktur/Hintergrund)

**Wichtige Konzepte:**
- FWHM = Maß für Seeing-Qualität
- Tile-Größe proportional zu FWHM
- Overlap für glatte Übergänge
- Einheitliches Grid für alle Frames/Kanäle

**Output:** Tile-Grid + FWHM-Statistiken

---

### [Phase 4: Lokale Tile-Metriken und Qualitätsanalyse](phase_4_local_metrics.md)
- **Stern-Tile-Metriken:** FWHM, Rundheit, Kontrast
- **Struktur-Tile-Metriken:** ENR, lokaler Hintergrund, Varianz
- Lokaler Qualitätsindex Q_local
- **Effektives Gewicht:** W_f,t,c = G_f,c × L_f,t,c

**Wichtige Konzepte:**
- Lokale Seeing-Variationen erfassen
- Tile-Typ-spezifische Metriken
- Kombination global × lokal
- Heatmap-Visualisierung möglich

**Output:** Lokale Gewichte pro Tile/Frame

---

### [Phase 5: Tile-basierte Rekonstruktion](phase_5_tile_reconstruction.md)
- **Gewichtete Rekonstruktion:** I_t,c = Σ W_f,t,c · I'_f,c / Σ W_f,t,c
- Fallback für degenerierte Tiles
- **Fensterfunktion** (Hanning, 2D, separabel; verbindlich)
- **Overlap-Add** für glatte Übergänge
- Tile-Normalisierung (verbindlich; nach Hintergrundsubtraktion)

**Wichtige Konzepte:**
- Herzstück der Methodik
- Jedes Tile separat rekonstruiert
- Weiche Übergänge durch Windowing
- Keine Frame-Selektion (alle Frames verwendet)

**Output:** Rekonstruierte Tiles → finales Bild pro Kanal

---

### [Phase 6: Zustandsbasierte Clusterung und synthetische Frames](phase_6_clustering.md)
- **Zustandsvektor:** v_f,c = (G, ⟨Q_local⟩, Var(Q_local), B, σ)
- K-Means Clusterung (dynamisches K: K = clip(floor(N/10), 5, 30))
- **Synthetische Frames** pro Cluster
- Rauschreduktion durch Cluster-Stacking
- Frame-Reduktion (N → K)

**Wichtige Konzepte:**
- Frames nach Qualitätszustand gruppieren
- Synthetische Frames = "ideale" Frames pro Zustand
- Nur bei N ≥ 200 (sonst Reduced Mode)
- Gewichtserhaltung

**Output:** K synthetische Frames (statt N Original-Frames)

---

### [Phase 7: Finales Stacking](phase_7_final_stacking.md)
- **Sigma-Clipping Rejection Stacking** (NEU)
- Konfigurierbare Parameter: sigma_low, sigma_high, max_iters, min_fraction
- FITS-Speicherung mit Metadaten
- Qualitätskontrolle und Validierung
- Statistik-Report

**Wichtige Konzepte:**
- Gewichtung bereits in synthetischen Frames
- **Sigma-Clipping entfernt verbleibende Ausreißer**
- Kein Drizzle
- RGB/LRGB-Kombination **außerhalb** der Methodik

**Output:** stacked_R.fits, stacked_G.fits, stacked_B.fits, stacked.fit

---

## Pipeline-Flussdiagramm (Aktuell)

```
┌─────────────────────────────────────────────────────────┐
│                    OSC RAW FRAMES                       │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 0: SCAN_INPUT   │
         │ Bayer-Pattern erkennen│
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 1: REGISTRATION │
         │ • Cosmetic Correction │
         │ • ECC Alignment       │
         │ • CFA-aware Warp      │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 2: CHANNEL_SPLIT│
         │ CFA → R/G/B           │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 3: NORMALIZATION│
         │ Background-Division   │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 4: GLOBAL_METRICS│
         │ B, σ, E → G_f,c       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 5: TILE_GRID    │
         │ FWHM-basierte Tiles   │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 6: LOCAL_METRICS│
         │ Tile-Qualität L_f,t,c │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 7: TILE_RECON   │
         │ Gewichtete Rekonstrukt│
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 8: CLUSTERING   │
         │ State-Vector K-Means  │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 9: SYNTHETIC    │
         │ Frames pro Cluster    │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 10: STACKING    │
         │ Sigma-Clip Rejection  │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 11: DEBAYER     │
         │ CFA → RGB Demosaic    │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 12: DONE        │
         │ Report & Finalize     │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ stacked_rgb.fits      │
         │ + Artifacts/Report    │
         └───────────────────────┘
```

## Kernprinzipien

Die Methodik basiert auf folgenden **unveränderlichen Prinzipien**:

1. **Linearität**: Keine nichtlinearen Operationen (kein Stretch)
2. **Keine Frame-Selektion**: Alle Frames werden verwendet
3. **Kanalgetrennt**: R, G, B unabhängig verarbeitet
4. **Streng linear**: Keine Rückkopplungen in der Pipeline
5. **Deterministisch**: Gleiche Inputs → gleiche Outputs
6. **Tile-basiert**: Lokale Qualitätsbewertung
7. **Gewichtet**: Global × Lokal
8. **Seeing-adaptiv**: Tile-Größe basiert auf FWHM

## Modi

### Normal Mode (N ≥ 200 Frames)
- Alle 7 Phasen werden durchlaufen
- Clusterung aktiv
- Synthetische Frames werden erzeugt
- Optimale Rauschreduktion

### Reduced Mode (50 ≤ N < 200 Frames)
- Phase 6 wird **übersprungen**
- Keine Clusterung
- Keine synthetischen Frames
- Phase 5 erzeugt das rekonstruierte Bild R_c pro Kanal
- Phase 7 übernimmt R_c direkt ohne weiteres Stacking
- Validierungswarnung im Report ("Reduced Mode")

### Degraded Mode (N < 50 Frames)
- Pipeline läuft im Reduced Mode ohne Clusterung
- Starke Degradation der Statistik (zu wenig Frames)
- Lauf wird mit entsprechendem Warnlevel gekennzeichnet
- Abbruch nur bei **kritischen** Fehlern (z.B. keine Sterne, Daten nicht linear)

## Qualitätsmetriken

### Globale Metriken (Phase 2)
- **B_f,c**: Hintergrundniveau (niedriger = besser)
- **σ_f,c**: Rauschen (niedriger = besser)
- **E_f,c**: Gradientenergie (höher = besser)
- **G_f,c**: Globales Gewicht = exp(Q_f,c)

### Lokale Metriken (Phase 4)
- **FWHM**: Seeing-Qualität (niedriger = besser)
- **Rundheit**: Tracking-Qualität (höher = besser)
- **Kontrast**: Signal-to-Noise (höher = besser)
- **L_f,t,c**: Lokales Gewicht = exp(Q_local)

### Effektives Gewicht
- **W_f,t,c = G_f,c × L_f,t,c**
- Kombiniert globale und lokale Qualität
- Verwendet in Phase 5 für Tile-Rekonstruktion

## Mathematische Notation

```
Indizes:
  f - Frame-Index (0..N-1)
  t - Tile-Index (0..T-1)
  c - Kanal (R, G, B)
  k - Cluster-Index (0..K-1)
  p - Pixel-Position (x, y)

Dimensionen:
  N - Anzahl Original-Frames
  K - Anzahl Cluster/synthetische Frames
  T - Anzahl Tiles
  W, H - Bildbreite/-höhe
  F - FWHM (Pixel)

Gewichte:
  G_f,c - Globales Frame-Gewicht
  L_f,t,c - Lokales Tile-Gewicht
  W_f,t,c - Effektives Gewicht (G × L)
  W_synth,k,c - Gewicht synthetisches Frame

Synthetic Frames:
  synthetic.weighting = global | tile_weighted

Normalisierung:
  I_f,c - Original-Frame
  I'_f,c - Normalisiertes Frame (I / B)
  B_f,c - Hintergrundniveau
```

## Validierung

Jede Phase enthält **normative Testfälle**:

- ✓ Gewichtsnormierung (α + β + γ = 1)
- ✓ Clamping vor Exponentialfunktion
- ✓ Tile-Size-Monotonie
- ✓ Overlap-Konsistenz
- ✓ Low-weight Tile Fallback
- ✓ Kanaltrennung (keine Kanal-Kopplung)
- ✓ Keine Frame-Selektion
- ✓ Determinismus

## Performance-Optimierungen

Jedes Phasen-Dokument enthält **Performance-Hinweise**:

- Vektorisierte Operationen (NumPy)
- Parallele Verarbeitung (ThreadPoolExecutor, ProcessPoolExecutor)
- Memory-effiziente Implementierungen
- GPU-Beschleunigung (optional, CuPy)
- Caching und Memoization

## Verwendung

1. **Lesen Sie Phase 0** für Gesamtübersicht
2. **Wählen Sie Vorverarbeitungspfad** (A oder B)
3. **Folgen Sie den Phasen sequenziell** (1-7)
4. **Beachten Sie Validierungschecks** in jeder Phase
5. **Prüfen Sie Output-Datenstrukturen** am Ende jeder Phase

## Referenzen

- **Normative Spezifikation**: `/doc/tile_basierte_qualitatsrekonstruktion_methodik_v_3.md`
- **Implementierung**: `/tile_compile_backend/` und `/runner/`
- **Tests**: `/tests/`
- **Validierung**: `/validation/`

## Änderungshistorie

- **2026-01-17**: Cosmetic Correction (Hotpixel-Entfernung vor Warp) hinzugefügt
- **2026-01-17**: Sigma-Clipping Rejection Stacking dokumentiert
- **2026-01-17**: Pipeline-Diagramm auf 13 Phasen aktualisiert
- **2026-01-09**: Initiale Erstellung der detaillierten Process Flow Dokumentation
- Basierend auf Methodik v3

---

**Hinweis**: Diese Dokumentation ist **deskriptiv** und erklärt die normative Spezifikation. Bei Widersprüchen gilt die normative Spezifikation in `tile_basierte_qualitatsrekonstruktion_methodik_v_3.md`.
