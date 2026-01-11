# Phase 0: Pipeline Overview & Preprocessing Paths

## Gesamtübersicht

Die Tile-basierte Qualitätsrekonstruktion besteht aus zwei gleichwertigen Vorverarbeitungspfaden (A und B), die ab Phase 2 in eine gemeinsame Pipeline münden.

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: OSC RAW FRAMES                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
             ┌────────▼────────┐
             │  Calibration     │
             │  (Bias/Dark/Flat)│
             └────────┬────────┘
                      │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼─────┐          ┌─────▼────┐
    │  PATH A  │          │  PATH B  │
    │  SIRIL   │          │   CFA    │
    └────┬─────┘          └─────┬────┘
         │                      │
         │  Debayer +           │  CFA-Luminanz
         │  Registration        │  Registration
         │                      │
         │  ┌────────────────┐  │
         │  │ Channel Split  │  │
         │  └────────────────┘  │
         │                      │
         └──────────┬───────────┘
                    │
    ┌───────────────▼────────────────┐
    │   GEMEINSAMER KERN (Phase 2+)  │
    │                                │
    │  • Global Normalization        │
    │  • Frame Metrics               │
    │  • Tile Generation             │
    │  • Local Tile Metrics          │
    │  • Tile Reconstruction         │
    │  • Clustering                  │
    │  • Synthetic Frames            │
    │  • Final Stacking              │
    └────────────────┬───────────────┘
                     │
         ┌───────────▼───────────┐
         │  R.fit / G.fit / B.fit │
         └───────────────────────┘
```

## Calibration (Bias/Dark/Flat) – Einordnung

Die Kalibrierung passiert im Runner vor der Registrierung in **SCAN_INPUT**.

Wenn `calibration.use_bias/use_dark/use_flat` aktiv sind:

- Master-Frames werden entweder aus `*_master` geladen oder aus `*_dir` erzeugt.
- Erzeugte Master (falls gebaut) liegen unter:
  - `runs/<run_id>/outputs/calibration/master_bias.fit`
  - `runs/<run_id>/outputs/calibration/master_dark.fit`
  - `runs/<run_id>/outputs/calibration/master_flat.fit`
- Kalibrierte Lights werden geschrieben nach:
  - `runs/<run_id>/outputs/calibrated/cal_XXXXX.fit`

Ab **PATH A / PATH B** arbeiten die Schritte dann auf diesen **kalibrierten** Frames.

## Path A: Siril-basiert (Empfohlen)

### Eigenschaften
- **Status**: Produktionsreif, bewährt
- **Risiko**: Gering
- **Komplexität**: Niedrig
- **Empfehlung**: Standard für alle Produktionsläufe

### Ablauf

```
┌──────────────┐
│  OSC Frames  │
└───

───┬───────┘
       │
       ▼
┌──────────────────────┐
│  Siril Debayer       │
│  (Interpolation)     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Siril Registration  │
│  • Star Detection    │
│  • Transform Estim.  │
│  • Rotation/Trans.   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Registered RGB      │
│  (3 channels/frame)  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Channel Separation  │
│  RGB → R, G, B       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  To Phase 2          │
│  (3 channel stacks)  │
└──────────────────────┘
```

### Kritische Punkte

1. **Debayer vor Registration**: Siril interpoliert zuerst, dann registriert
2. **Eine Transformation pro Frame**: Geometrisch konsistent
3. **Kanaltrennung NACH Registration**: Verhindert farbabhängige Resampling-Residuen

## Path B: CFA-basiert (Experimentell)

### Eigenschaften
- **Status**: Experimentell
- **Risiko**: Höher (neue Implementierung)
- **Komplexität**: Hoch
- **Vorteil**: Methodisch maximal sauber, keine Farbinterpolation vor Tile-Analyse

### Ablauf

```
┌──────────────┐
│  OSC Frames  │
│  (

CFA Mosaic)│
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  CFA Luminance       │
│  Extraction          │
│  (G-dominant/Sum)    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Registration on     │
│  CFA Luminance       │
│  • RANSAC/ECC        │
│  • Single Transform  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  CFA-aware Transform │
│  • Split to 4 planes │
│  • R, G1, G2, B      │
│  • Same transform    │
│  • Re-interleave     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Registered CFA      │
│  (no color mixing)   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Debayer/Extract     │
│  CFA → R, G, B       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  To Phase 2          │
│  (3 channel stacks)  │
└──────────────────────┘
```

### Kritische Punkte

1. **CFA-Luminanz**: Muss repräsentativ sein (G-Kanal dominant)
2. **Subplane-Zerlegung**: Keine Interpolation zwischen Bayer-Phasen
3. **Identische Transformation**: Farbunabhängig, aber CFA-aware Resampling

## Vergleich der Pfade

| Aspekt | Path A (Siril) | Path B (CFA) |
|--------|----------------|--------------|
| **Interpolation** | Vor Registration | Nach Registration |
| **Farbmischung** | Möglich bei Resampling | Vermieden |
| **Komplexität** | Niedrig | Hoch |
| **Implementierung** | Extern (Siril) | Custom |
| **Validierung** | Jahrelang erprobt | Experimentell |
| **Produktionsreife** | ✓ | In Entwicklung |

## Übergabepunkt an gemeinsamen Kern

Beide Pfade liefern **identische Datenstrukturen**:

```python
# Pro Kanal (R, G, B):
frames[f][x, y]  # f = Frame-Index
                 # x, y = Pixel-Koordinaten
                 # Alle Frames geometrisch aligned
```

### Garantien am Übergabepunkt

1. ✓ Alle Frames geometrisch registriert
2. ✓ Kanäle getrennt (R, G, B)
3. ✓ Linear (kein Stretch)
4. ✓ Einheitliche Geometrie pro Kanal
5. ✓ Keine Frame-Selektion durchgeführt

## Konfiguration

```yaml
preprocessing:
  path: "siril"  # oder "cfa"
  
  siril:
    debayer_method: "VNG"  # oder "AHD", "Bilinear"
    registration_method: "stars"
    
  cfa:
    luminance_method: "g_dominant"  # oder "sum", "weighted"
    registration_method: "ecc"  # oder "ransac"
    subplane_interpolation: "lanczos3"
```

## Validierung

Nach Abschluss von Phase 0/1 werden folgende Checks durchgeführt:

```
✓ Frame count >= 50 (minimum)
✓ All frames same dimensions
✓ All frames registered (residuum < 1.0 px)
✓ Channels separated (R, G, B)
✓ Data is linear (no stretch detected)
✓ No NaN/Inf values
```

## Nächste Phase

→ **Phase 2: Globale Normalisierung und Frame-Metriken**
