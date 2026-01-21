# Process Flow Documentation - Tile-basierte Qualitätsrekonstruktion v4

## Übersicht

Diese Dokumentation beschreibt detailliert den Ablauf der **Tile-basierten Qualitätsrekonstruktion für DSO (Deep Sky Objects)** gemäß **Methodik v4**.

Jede Phase ist in einem separaten Dokument mit Diagrammen, Formeln und Beispielen dokumentiert.

## Aktuelle Pipeline-Phasen (v4 Implementierung)

| Phase | Name | Beschreibung |
|-------|------|--------------|
| 0 | SCAN_INPUT | Input-Frames scannen, Bayer-Pattern erkennen |
| 1 | CHANNEL_SPLIT | CFA → R/G/B Kanaltrennung (deferred to tile processing) |
| 2 | NORMALIZATION | Hintergrund-Normalisierung (applied during tile loading) |
| 3 | GLOBAL_METRICS | Globale Qualitätsmetriken (Background, Noise, Gradient) |
| 4 | TILE_GRID | Adaptive/Hierarchische Tile-Grid-Erzeugung |
| 5 | LOCAL_METRICS | Lokale Tile-Metriken (computed during TLR) |
| 6 | TILE_RECONSTRUCTION_TLR | **Tile-Local Registration & Reconstruction** |
| 7 | STATE_CLUSTERING | Zustandsbasierte Frame-Clusterung |
| 8 | SYNTHETIC_FRAMES | Synthetische Frames pro Cluster |
| 9 | STACKING | Sigma-Clipping Rejection Stacking |

## Kernunterschiede zu v3

### 1. **Tile-Local Registration (TLR)** statt Global Registration
- **v3**: Globale ECC-Registrierung aller Frames → ein Warp-Feld pro Frame
- **v4**: **Jedes Tile registriert sich lokal** → Warp-Feld pro Tile/Frame
- **Vorteil**: Feldrotation, differentielle atmosphärische Refraktion, lokale Seeing-Variationen werden korrekt behandelt

### 2. **Iterative Reconstruction** pro Tile
- **v3**: Gewichtete Rekonstruktion in einem Durchgang
- **v4**: Iterative Verfeinerung (4 Iterationen):
  1. Initiale Warp-Schätzung
  2. Cross-Correlation-basierte Frame-Gewichtung
  3. Rekonstruktion mit gewichteten, gewarpten Frames
  4. Wiederhole bis Konvergenz

### 3. **Adaptive Tile Refinement**
- **v3**: Statisches FWHM-basiertes Grid
- **v4**: Multi-Pass Refinement:
  - **Pass 0**: Initiales Grid (hierarchisch oder warp-probe-basiert)
  - **Pass 1-3**: Splitte Tiles mit hoher Warp-Varianz
  - **Ergebnis**: 30-50% weniger Tiles bei gleicher/besserer Qualität

### 4. **Hierarchisches/Warp-Probe Grid**
- **v3**: Uniformes Grid basierend auf globalem FWHM
- **v4**: 
  - **Warp Probe**: Analysiert Warp-Gradienten vor Tile-Erstellung
  - **Hierarchical**: Rekursive Unterteilung in Hochgradienten-Regionen
  - **Adaptive Tile-Größe**: s(x,y) = s₀ / (1 + c·grad)

### 5. **Keine globale Normalisierung**
- **v3**: Globale Background-Division in Phase 3
- **v4**: Normalisierung während Tile-Loading (on-the-fly)

### 6. **Deferred Channel Split**
- **v3**: Channel Split nach Registration (Phase 2)
- **v4**: Channel Split während Tile-Processing (keine separate Phase)

## Pipeline-Flussdiagramm (v4)

```
┌─────────────────────────────────────────────────────────┐
│                    OSC RAW FRAMES                       │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 0: SCAN_INPUT   │
         │ Bayer-Pattern erkennen│
         │ Frame-Metadaten       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 1: CHANNEL_SPLIT│
         │ (deferred)            │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 2: NORMALIZATION│
         │ (applied during load) │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 3: GLOBAL_METRICS│
         │ B, σ, E → G_f,c       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 4: TILE_GRID    │
         │ • Warp Probe          │
         │ • Hierarchical Init   │
         │ • Adaptive Sizing     │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 5: LOCAL_METRICS│
         │ (computed in TLR)     │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 6: TLR          │
         │ ┌─────────────────┐   │
         │ │ FOR EACH TILE:  │   │
         │ │ • Load Frames   │   │
         │ │ • Debayer       │   │
         │ │ • Normalize     │   │
         │ │ • Iterative:    │   │
         │ │   - Warp Estim. │   │
         │ │   - CC Weights  │   │
         │ │   - Reconstruct │   │
         │ │ • 4 Iterations  │   │
         │ └─────────────────┘   │
         │                       │
         │ Multi-Pass Refinement:│
         │ • Pass 0: Initial     │
         │ • Pass 1-3: Split     │
         │   high-variance tiles │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 7: CLUSTERING   │
         │ State-Vector K-Means  │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 8: SYNTHETIC    │
         │ Frames pro Cluster    │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ PHASE 9: STACKING     │
         │ Sigma-Clip Rejection  │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ stacked_R/G/B.fits    │
         │ + Artifacts/Report    │
         └───────────────────────┘
```

## Dokumenten-Struktur

### [Phase 0: Pipeline-Übersicht & Input-Scanning](phase_0_overview.md)
- Gesamtübersicht der v4-Pipeline
- Input-Frame-Scanning
- Bayer-Pattern-Erkennung
- Frame-Metadaten-Extraktion
- Vergleich v3 vs v4

**Wichtige Konzepte:**
- Keine globale Registration mehr
- Tile-Local Registration (TLR) als Kern
- Deferred Processing (Channel Split, Normalization)

---

### [Phase 1-2: Deferred Processing](phase_1_deferred.md)
- Channel Split (während Tile-Loading)
- Normalisierung (während Tile-Loading)
- On-the-fly Debayering

**Wichtige Konzepte:**
- Kein separater Registrierungsschritt
- Frames bleiben in CFA-Format bis Tile-Processing
- Memory-effizient

---

### [Phase 3: Globale Frame-Metriken](phase_3_global_metrics.md)
- Hintergrundschätzung (Sigma-Clipping)
- Rauschschätzung
- Gradientenergie (Sobel-Operator)
- **Globaler Qualitätsindex** Q_f,c und Gewicht G_f,c

**Wichtige Konzepte:**
- Robuste Normalisierung mit Median + MAD
- Exponential-Mapping mit Clamping
- Gewichtung: α·(-B̃) + β·(-σ̃) + γ·Ẽ

**Output:** Globale Gewichte G_f,c pro Frame/Kanal

---

### [Phase 4: Adaptive Tile-Grid-Erzeugung](phase_4_tile_grid.md)
- **Warp Probe**: Gradient-Field-Analyse
- **Hierarchical Initialization**: Rekursive Unterteilung
- **Adaptive Tile-Größe**: s(x,y) = s₀ / (1 + c·grad)
- Tile-Grid-Optimierung (30-50% Reduktion)

**Wichtige Konzepte:**
- Warp-Gradienten bestimmen Tile-Dichte
- Hierarchische Rekursion (max_depth=3)
- Gradient-Sensitivity-Parameter
- Minimale Tile-Größe (64px)

**Output:** Optimiertes Tile-Grid

---

### [Phase 5: Lokale Metriken (in TLR integriert)](phase_5_local_metrics.md)
- **Computed during TLR** (keine separate Phase)
- Warp-Varianz pro Tile
- Cross-Correlation pro Frame/Tile
- Lokale Qualitätsindizes

**Wichtige Konzepte:**
- Metriken während iterativer Rekonstruktion
- Warp-Varianz = Splitting-Kriterium
- CC-basierte Frame-Gewichtung

---

### [Phase 6: Tile-Local Registration & Reconstruction (TLR)](phase_6_tlr.md)
- **Kernphase der v4-Methodik**
- Iterative Rekonstruktion (4 Iterationen)
- Lokale Warp-Schätzung pro Tile
- Cross-Correlation-basierte Gewichtung: R_{f,t} = exp(β·(cc-1))
- Multi-Pass Adaptive Refinement

**Wichtige Konzepte:**
- Jedes Tile unabhängig
- Disk-Streaming (memory-effizient)
- Iterative Verfeinerung
- Adaptive Tile-Splitting

**Output:** Rekonstruierte Tiles → finales Bild pro Kanal

---

### [Phase 7: Zustandsbasierte Clusterung](phase_7_clustering.md)
- **Zustandsvektor:** v_f,c = (G, ⟨Q_local⟩, Var(Q_local), B, σ)
- K-Means Clusterung (dynamisches K)
- Frame-Reduktion (N → K)

**Wichtige Konzepte:**
- Frames nach Qualitätszustand gruppieren
- Nur bei N ≥ 200 (sonst Reduced Mode)
- Gewichtserhaltung

**Output:** K Cluster-Zuordnungen

---

### [Phase 8: Synthetische Frames](phase_8_synthetic.md)
- Synthetische Frames pro Cluster
- Rauschreduktion durch Cluster-Stacking
- Gewichtserhaltung

**Wichtige Konzepte:**
- "Ideale" Frames pro Zustand
- Frame-Reduktion für finales Stacking
- Gewichtete Kombination

**Output:** K synthetische Frames (statt N Original-Frames)

---

### [Phase 9: Finales Stacking](phase_9_stacking.md)
- **Sigma-Clipping Rejection Stacking**
- Konfigurierbare Parameter
- FITS-Speicherung mit Metadaten
- Qualitätskontrolle und Validierung

**Wichtige Konzepte:**
- Gewichtung bereits in synthetischen Frames
- Sigma-Clipping entfernt verbleibende Ausreißer
- Kein Drizzle

**Output:** stacked_R.fits, stacked_G.fits, stacked_B.fits

---

## Kernprinzipien (v4)

Die Methodik basiert auf folgenden **unveränderlichen Prinzipien**:

1. **Linearität**: Keine nichtlinearen Operationen (kein Stretch)
2. **Keine Frame-Selektion**: Alle Frames werden verwendet
3. **Kanalgetrennt**: R, G, B unabhängig verarbeitet
4. **Tile-Local Registration**: Keine globale Registrierung
5. **Iterative Reconstruction**: 4 Iterationen pro Tile
6. **Deterministisch**: Gleiche Inputs → gleiche Outputs
7. **Adaptive Tiles**: Warp-Gradienten bestimmen Tile-Dichte
8. **Memory-effizient**: Disk-Streaming, deferred processing

## Modi

### Normal Mode (N ≥ 200 Frames)
- Alle Phasen werden durchlaufen
- Clusterung aktiv
- Synthetische Frames werden erzeugt
- Optimale Rauschreduktion

### Reduced Mode (50 ≤ N < 200 Frames)
- Phase 7-8 werden **übersprungen**
- Keine Clusterung
- Keine synthetischen Frames
- Phase 6 erzeugt das rekonstruierte Bild R_c pro Kanal
- Phase 9 übernimmt R_c direkt
- Validierungswarnung im Report ("Reduced Mode")

### Degraded Mode (N < 50 Frames)
- Pipeline läuft im Reduced Mode
- Starke Degradation der Statistik
- Lauf wird mit entsprechendem Warnlevel gekennzeichnet
- Abbruch nur bei **kritischen** Fehlern

## Qualitätsmetriken

### Globale Metriken (Phase 3)
- **B_f,c**: Hintergrundniveau (niedriger = besser)
- **σ_f,c**: Rauschen (niedriger = besser)
- **E_f,c**: Gradientenergie (höher = besser)
- **G_f,c**: Globales Gewicht = exp(Q_f,c)

### Lokale Metriken (Phase 6, während TLR)
- **Warp-Varianz**: Variabilität der Warp-Vektoren
- **Cross-Correlation**: Frame-Tile-Übereinstimmung
- **R_{f,t}**: Frame-Gewicht = exp(β·(cc-1))

### Effektives Gewicht
- **W_f,t,c = G_f,c × R_{f,t}**
- Kombiniert globale und lokale Qualität
- Verwendet in Phase 6 für Tile-Rekonstruktion

## Mathematische Notation

```
Indizes:
  f - Frame-Index (0..N-1)
  t - Tile-Index (0..T-1)
  c - Kanal (R, G, B)
  k - Cluster-Index (0..K-1)
  i - Iterations-Index (0..I-1)
  p - Pixel-Position (x, y)

Dimensionen:
  N - Anzahl Original-Frames
  K - Anzahl Cluster/synthetische Frames
  T - Anzahl Tiles
  I - Anzahl Iterationen (4)
  W, H - Bildbreite/-höhe

Gewichte:
  G_f,c - Globales Frame-Gewicht
  R_{f,t} - Lokales Frame-Gewicht (CC-basiert)
  W_f,t,c - Effektives Gewicht (G × R)

TLR-Parameter:
  β - CC-Sensitivität (6.0 für Alt/Az)
  iterations - Anzahl Iterationen (4)
  refine_variance_threshold - Splitting-Schwelle (0.15)
  max_refine_passes - Max Refinement-Passes (3)

Adaptive Tiles:
  s(x,y) - Tile-Größe an Position (x,y)
  s₀ - Basis-Tile-Größe (256)
  c - Gradient-Sensitivität (2.0)
  grad - Lokaler Warp-Gradient
```

## Validierung

Jede Phase enthält **normative Testfälle**:

- ✓ Gewichtsnormierung (α + β + γ = 1)
- ✓ Clamping vor Exponentialfunktion
- ✓ Tile-Size-Monotonie
- ✓ Warp-Varianz-Konvergenz
- ✓ CC-basierte Gewichtung
- ✓ Iterative Konvergenz
- ✓ Kanaltrennung (keine Kanal-Kopplung)
- ✓ Keine Frame-Selektion
- ✓ Deterministismus

## Performance-Optimierungen

- **Disk-Streaming**: Frames on-demand laden
- **Parallele Tile-Verarbeitung**: 8 Workers (konfigurierbar)
- **Memory-Limits**: RSS-Monitoring mit Abort-Schwelle
- **Adaptive Refinement**: Nur problematische Tiles splitten
- **Deferred Processing**: Channel Split/Normalisierung on-the-fly

## Verwendung

1. **Lesen Sie Phase 0** für Gesamtübersicht
2. **Verstehen Sie TLR** (Phase 6) - das Herzstück
3. **Folgen Sie den Phasen sequenziell** (0-9)
4. **Beachten Sie Validierungschecks** in jeder Phase
5. **Prüfen Sie Output-Datenstrukturen** am Ende jeder Phase

## Referenzen

- **Normative Spezifikation**: `/doc/tile_basierte_qualitaetsrekonstruktion_methodik_v4.md`
- **Konfiguration**: `/doc/configuration_reference_v4.md`
- **Implementierung**: `/tile_compile_backend/` und `/runner/`
- **Tests**: `/tests/`

## Änderungshistorie

- **2026-01-21**: Initiale Erstellung der v4 Process Flow Dokumentation
- Basierend auf Methodik v4 und v3 Process Flow Struktur

---

**Hinweis**: Diese Dokumentation ist **deskriptiv** und erklärt die normative Spezifikation. Bei Widersprüchen gilt die normative Spezifikation in `tile_basierte_qualitaetsrekonstruktion_methodik_v4.md`.
