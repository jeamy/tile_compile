# Process Flow Documentation - Tile-basierte Qualitätsrekonstruktion

## Übersicht

Diese Dokumentation beschreibt detailliert den Ablauf der **Tile-basierten Qualitätsrekonstruktion für DSO (Deep Sky Objects)** gemäß Methodik v3.

Jede Phase ist in einem separaten Dokument mit Diagrammen, Formeln und Beispielen dokumentiert.

## Dokumenten-Struktur

### [Phase 0: Pipeline-Übersicht & Vorverarbeitungspfade](phase_0_overview.md)
- Gesamtübersicht der Pipeline
- **Path A: Siril-basiert** (empfohlen, produktionsreif)
- **Path B: CFA-basiert** (experimentell, methodisch optimal)
- Vergleich der beiden Pfade
- Übergabepunkt an gemeinsamen Kern

**Wichtige Konzepte:**
- Zwei gleichwertige Vorverarbeitungspfade
- Ab Phase 2 identische Verarbeitung
- Debayer-Zeitpunkt unterschiedlich

---

### [Phase 1: Registrierung und Kanaltrennung](phase_1_registration.md)
- **Path A:** Siril Debayer → Registration → Channel Split
- **Path B:** CFA Luminance → Registration → CFA-aware Transform → Debayer → Channel Split
- FWHM-Messung und Sternfindung
- Transformationsschätzung (RANSAC/ECC)
- Qualitätsmetriken (Registrierungsresiduum, Elongation)

**Wichtige Konzepte:**
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
- Z-Score-Normalisierung der Metriken
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

### [Phase 7: Finales lineares Stacking](phase_7_final_stacking.md)
- **Einfaches lineares Stacking:** I_final = (1/K) · Σ F_synth
- Keine zusätzliche Gewichtung
- FITS-Speicherung mit Metadaten
- Qualitätskontrolle und Validierung
- Statistik-Report

**Wichtige Konzepte:**
- Gewichtung bereits in synthetischen Frames
- Einfacher Durchschnitt (linear)
- Kein Drizzle
- RGB/LRGB-Kombination **außerhalb** der Methodik

**Output:** Rekonstruktion_R.fit, Rekonstruktion_G.fit, Rekonstruktion_B.fit

---

## Pipeline-Flussdiagramm

```
┌─────────────────────────────────────────────────────────┐
│                    OSC RAW FRAMES                       │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼─────┐          ┌─────▼────┐
    │ PATH A   │          │ PATH B   │
    │ (Siril)  │          │ (CFA)    │
    └────┬─────┘          └─────┬────┘
         │                      │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │ PHASE 1: Registration│
         │ & Channel Separation │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │ PHASE 2: Global      │
         │ Normalization        │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │ PHASE 3: Tile        │
         │ Generation (FWHM)    │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │ PHASE 4: Local       │
         │ Tile Metrics         │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │ PHASE 5: Tile-based  │
         │ Reconstruction       │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │ PHASE 6: Clustering  │
         │ & Synthetic Frames   │
         │ (if N ≥ 200)         │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │ PHASE 7: Final       │
         │ Linear Stacking      │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │ R.fit / G.fit / B.fit│
         └──────────────────────┘
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
- Direktes Stacking der Original-Frames in Phase 7
- Validierungswarnung im Report

### Minimum (N < 50 Frames)
- **Abbruch** der Pipeline
- Zu wenig Frames für stabile Statistiken

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

- **2026-01-09**: Initiale Erstellung der detaillierten Process Flow Dokumentation
- Basierend auf Methodik v3

---

**Hinweis**: Diese Dokumentation ist **deskriptiv** und erklärt die normative Spezifikation. Bei Widersprüchen gilt die normative Spezifikation in `tile_basierte_qualitatsrekonstruktion_methodik_v_3.md`.
