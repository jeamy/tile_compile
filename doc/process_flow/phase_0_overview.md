# Phase 0: Pipeline Overview & Input Scanning

## Gesamtübersicht v4

Die Tile-basierte Qualitätsrekonstruktion v4 unterscheidet sich fundamental von v3 durch den Verzicht auf globale Registrierung. Stattdessen verwendet v4 **Tile-Local Registration (TLR)** als Kernkonzept.

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: OSC RAW FRAMES                        │
│                    (CFA Bayer Mosaic)                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
             ┌────────▼────────┐
             │  Calibration     │
             │  (Bias/Dark/Flat)│
             │  [optional]      │
             └────────┬────────┘
                      │
         ┌────────────▼────────────┐
         │   PHASE 0: SCAN_INPUT   │
         │   • Bayer Pattern       │
         │   • Frame Metadata      │
         │   • Dimensions          │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │ PHASE 1-2: DEFERRED     │
         │ (no separate processing)│
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │ PHASE 3: GLOBAL_METRICS │
         │ • Background            │
         │ • Noise                 │
         │ • Gradient Energy       │
         │ → Global Weights G_f,c  │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │ PHASE 4: TILE_GRID      │
         │ • Warp Probe (optional) │
         │ • Hierarchical (opt.)   │
         │ • Adaptive Sizing       │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │ PHASE 5: LOCAL_METRICS  │
         │ (computed in TLR)       │
         └────────────┬────────────┘
                      │
    ┌────────────────▼────────────────┐
    │ PHASE 6: TLR (KERNPHASE)        │
    │                                 │
    │ FOR EACH TILE (parallel):       │
    │ ┌─────────────────────────────┐ │
    │ │ 1. Load CFA tile regions    │ │
    │ │ 2. Debayer → R, G, B        │ │
    │ │ 3. Normalize (on-the-fly)   │ │
    │ │                             │ │
    │ │ FOR i = 1..4 (iterations):  │ │
    │ │   a) Estimate warps         │ │
    │ │   b) Compute CC → R_{f,t}   │ │
    │ │   c) Weighted reconstruct   │ │
    │ │   d) Update weights         │ │
    │ │                             │ │
    │ │ 4. Output: Tile + Metadata  │ │
    │ └─────────────────────────────┘ │
    │                                 │
    │ Multi-Pass Refinement:          │
    │ • Pass 0: Initial tiles         │
    │ • Pass 1-3: Split high-variance │
    │   tiles and re-process          │
    └────────────┬────────────────────┘
                 │
    ┌────────────▼────────────────┐
    │ PHASE 7-9: CLUSTERING +     │
    │ SYNTHETIC + STACKING        │
    │ (wie v3)                    │
    └────────────┬────────────────┘
                 │
    ┌────────────▼────────────────┐
    │ stacked_R/G/B.fits          │
    └─────────────────────────────┘
```

## Kernunterschiede v3 → v4

### 1. Keine globale Registrierung

**v3:**
```
Input → Global ECC Registration → Warped Frames → Tile Processing
```

**v4:**
```
Input → Tile Processing (mit lokaler Registration pro Tile)
```

**Warum?**
- **Feldrotation**: Globale Transformation kann Rotation nicht lokal anpassen
- **Differentielle Refraktion**: Atmosphärische Effekte variieren über Bildfeld
- **Lokales Seeing**: Turbulenz ist räumlich inhomogen
- **Memory**: Keine gewarpten Full-Frame-Kopien nötig

### 2. Deferred Processing

**v3:**
```
Phase 1: Registration → warped RGB frames
Phase 2: Channel Split → R, G, B stacks
Phase 3: Normalization → normalized stacks
Phase 4: Tile Grid
Phase 5: Tile Processing
```

**v4:**
```
Phase 0: Scan Input → CFA frames (unprocessed)
Phase 3: Global Metrics → weights only
Phase 4: Tile Grid
Phase 6: TLR → Load CFA tile → Debayer → Normalize → Register → Reconstruct
```

**Vorteil:**
- **Memory**: Nur aktuelle Tile-Regionen im RAM
- **Disk I/O**: Streaming statt Full-Frame-Loads
- **Flexibilität**: Tile-spezifische Parameter

### 3. Iterative Reconstruction

**v3:**
```
Tile Reconstruction (single-pass):
  weighted_sum = Σ W_f,t,c · I_f,t,c
  tile = weighted_sum / Σ W_f,t,c
```

**v4:**
```
Tile Reconstruction (iterative):
  FOR i = 1..4:
    1. Estimate warp vectors w_f,t for all frames
    2. Compute cross-correlation cc_f,t
    3. Update weights: R_{f,t} = exp(β·(cc_f,t - 1))
    4. Reconstruct: tile = Σ (G_f,c · R_{f,t}) · warp(I_f,t,c) / Σ weights
    5. Use reconstructed tile as reference for next iteration
```

**Vorteil:**
- **Bessere Warp-Schätzung**: Iterative Verfeinerung
- **Robustheit**: Schlechte Frames werden automatisch heruntergewichtet
- **Konvergenz**: Typisch 3-4 Iterationen ausreichend

### 4. Adaptive Tile Refinement

**v3:**
```
Tile Grid: Uniform, FWHM-basiert
  tile_size = scale_factor · FWHM
  grid = create_uniform_grid(tile_size)
```

**v4:**
```
Tile Grid: Adaptive, Multi-Pass
  Pass 0: Initial grid (hierarchical/warp-probe)
  FOR pass = 1..3:
    1. Process all tiles
    2. Compute warp_variance per tile
    3. Split tiles where warp_variance > threshold
    4. Re-process split tiles
  UNTIL no tiles split OR max_passes reached
```

**Vorteil:**
- **Effizienz**: 30-50% weniger Tiles
- **Qualität**: Mehr Tiles nur wo nötig (Feldrotation, Seeing-Variationen)
- **Adaptiv**: Automatische Anpassung an Daten

## Phase 0: Input Scanning

### Ziele

1. Bayer-Pattern erkennen
2. Frame-Dimensionen validieren
3. Metadaten extrahieren
4. Kalibrierung anwenden (optional)

### Ablauf

```
┌──────────────────────────────────────────┐
│  Input: Raw FITS/FIT files               │
│  /path/to/lights/*.fit                   │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  FOR EACH file:                          │
│    1. Read FITS header                   │
│    2. Extract metadata:                  │
│       • NAXIS1, NAXIS2 (dimensions)      │
│       • BAYERPAT (pattern)               │
│       • EXPOSURE, ISO, TEMP              │
│       • DATE-OBS                         │
│    3. Validate:                          │
│       • All frames same dimensions       │
│       • All frames same Bayer pattern    │
│       • Data type: uint16 or float32     │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  Bayer Pattern Detection                 │
│                                          │
│  If BAYERPAT header exists:              │
│    pattern = BAYERPAT                    │
│  Else:                                   │
│    pattern = auto_detect_bayer()         │
│                                          │
│  Supported patterns:                     │
│    • RGGB (most common)                  │
│    • BGGR                                │
│    • GRBG                                │
│    • GBRG                                │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  Optional: Calibration                   │
│                                          │
│  IF calibration.use_bias:                │
│    frame -= master_bias                  │
│  IF calibration.use_dark:                │
│    frame -= master_dark · (exp/exp_dark) │
│  IF calibration.use_flat:                │
│    frame /= master_flat_normalized       │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  Output: Frame Catalog                   │
│  {                                       │
│    'frame_paths': List[str],             │
│    'bayer_pattern': str,                 │
│    'dimensions': (W, H),                 │
│    'frame_count': N,                     │
│    'metadata': List[Dict]                │
│  }                                       │
└──────────────────────────────────────────┘
```

### Bayer Pattern Auto-Detection

```python
def auto_detect_bayer(frame: np.ndarray) -> str:
    """
    Detektiert Bayer-Pattern durch Analyse der Farbkanäle.
    
    Strategie:
    1. Extrahiere 4 Subplanes (2x2 Mosaic)
    2. Berechne Mittelwert jeder Subplane
    3. Identifiziere G-Kanäle (höchster Mittelwert, 2x vorhanden)
    4. Identifiziere R/B-Kanäle (unterschiedliche Mittelwerte)
    """
    # Extract 2x2 subplanes
    p00 = frame[0::2, 0::2]  # Top-left
    p01 = frame[0::2, 1::2]  # Top-right
    p10 = frame[1::2, 0::2]  # Bottom-left
    p11 = frame[1::2, 1::2]  # Bottom-right
    
    # Compute means
    means = {
        (0,0): np.median(p00),
        (0,1): np.median(p01),
        (1,0): np.median(p10),
        (1,1): np.median(p11),
    }
    
    # G-channels have highest mean (2x)
    sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)
    
    # First two are G, last two are R/B
    g1_pos, g2_pos = sorted_means[0][0], sorted_means[1][0]
    rb_positions = [sorted_means[2][0], sorted_means[3][0]]
    
    # R typically has higher mean than B
    r_pos = rb_positions[0] if means[rb_positions[0]] > means[rb_positions[1]] else rb_positions[1]
    b_pos = rb_positions[1] if r_pos == rb_positions[0] else rb_positions[0]
    
    # Map to pattern name
    pattern_map = {
        ((0,0), 'R'): 'RGGB',
        ((0,0), 'B'): 'BGGR',
        ((0,1), 'R'): 'GRBG',
        ((0,1), 'B'): 'GBRG',
    }
    
    return pattern_map.get((r_pos, 'R'), 'RGGB')  # Default: RGGB
```

### Kalibrierung (Optional)

**Master Bias:**
```
master_bias = median(bias_frames)
calibrated = raw - master_bias
```

**Master Dark:**
```
master_dark = median(dark_frames)
dark_scaled = master_dark · (exposure_light / exposure_dark)
calibrated = (raw - master_bias) - dark_scaled
```

**Master Flat:**
```
master_flat_raw = median(flat_frames) - master_bias
master_flat_normalized = master_flat_raw / median(master_flat_raw)
calibrated = ((raw - master_bias) - dark_scaled) / master_flat_normalized
```

**Wichtig:**
- Kalibrierung erfolgt **vor** allen anderen Schritten
- Kalibrierte Frames werden gespeichert: `runs/<run_id>/outputs/calibrated/`
- Alle weiteren Phasen arbeiten auf kalibrierten Frames

## Validierung nach Phase 0

```python
def validate_phase0(catalog):
    checks = []
    
    # Check 1: Frame count
    N = catalog['frame_count']
    if N < 50:
        checks.append(f"⚠ Degraded Mode: {N} frames (< 50)")
    elif N < 200:
        checks.append(f"⚠ Reduced Mode: {N} frames (< 200)")
    else:
        checks.append(f"✓ Normal Mode: {N} frames")
    
    # Check 2: Dimensions
    W, H = catalog['dimensions']
    assert W > 0 and H > 0
    assert W % 2 == 0 and H % 2 == 0  # Even dimensions for Bayer
    checks.append(f"✓ Dimensions: {W}x{H}")
    
    # Check 3: Bayer pattern
    pattern = catalog['bayer_pattern']
    assert pattern in ['RGGB', 'BGGR', 'GRBG', 'GBRG']
    checks.append(f"✓ Bayer Pattern: {pattern}")
    
    # Check 4: All frames readable
    for i, path in enumerate(catalog['frame_paths']):
        assert os.path.exists(path), f"Frame {i} not found: {path}"
    checks.append(f"✓ All {N} frames accessible")
    
    return checks
```

## Output-Datenstruktur

```python
# Phase 0 Output
{
    'frame_paths': List[str],           # Paths to (calibrated) frames
    'bayer_pattern': str,               # 'RGGB', 'BGGR', 'GRBG', 'GBRG'
    'dimensions': Tuple[int, int],      # (W, H)
    'frame_count': int,                 # N
    'metadata': List[Dict[str, Any]],   # Per-frame metadata
    'calibration': {                    # Optional
        'bias_applied': bool,
        'dark_applied': bool,
        'flat_applied': bool,
    }
}
```

## Konfiguration

```yaml
input:
  light_dir: "/path/to/lights"
  pattern: "*.fit"  # or "*.fits"

calibration:
  use_bias: true
  use_dark: true
  use_flat: true
  bias_master: "/path/to/master_bias.fit"  # or bias_dir for auto-creation
  dark_master: "/path/to/master_dark.fit"
  flat_master: "/path/to/master_flat.fit"
```

## Nächste Phase

→ **Phase 1-2: Deferred Processing** (keine separate Verarbeitung, erfolgt in Phase 6)
→ **Phase 3: Globale Frame-Metriken**
