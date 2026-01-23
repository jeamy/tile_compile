# Phase 1-2: Deferred Processing

## Übersicht

In v4 gibt es **keine separaten Phasen** für Channel Split und Normalisierung. Diese Operationen werden **on-the-fly während des Tile-Processings** (Phase 6) durchgeführt.

Dies ist ein fundamentaler Unterschied zu v3 und ermöglicht erhebliche Memory- und Performance-Vorteile.

## Warum Deferred Processing?

### Memory-Effizienz

**v3 Approach:**
```
1. Load all frames (N × W × H × 1 channel CFA)     → ~4 GB
2. Register all frames (N × W × H × 1 channel)     → ~4 GB
3. Debayer all frames (N × W × H × 3 channels)     → ~12 GB
4. Split channels (3 × N × W × H)                  → ~12 GB
5. Normalize all frames (3 × N × W × H)            → ~12 GB

Total peak memory: ~44 GB for 200 frames @ 6000×4000
```

**v4 Approach:**
```
1. Scan frames (metadata only)                     → ~1 MB
2. Compute global metrics (streaming)              → ~100 MB
3. Create tile grid                                → ~1 MB
4. FOR EACH TILE (parallel):
   a) Load CFA tile regions (N × tile_size²)       → ~50 MB
   b) Debayer tile regions (N × tile_size² × 3)    → ~150 MB
   c) Normalize on-the-fly                         → ~0 MB (in-place)
   d) Process tile                                 → ~200 MB
   e) Free memory                                  → ~0 MB

Peak memory per tile: ~400 MB
Total peak (8 parallel tiles): ~3.2 GB
```

**Reduktion: 44 GB → 3.2 GB (93% weniger!)**

### Disk I/O Effizienz

**v3:**
```
Read: N full frames (CFA)
Write: N full frames (registered CFA)
Write: N full frames (debayered RGB)
Write: 3 × N full frames (channel-separated)
Write: 3 × N full frames (normalized)

Total: 9 × N full-frame writes
```

**v4:**
```
Read: N full frames (CFA, streaming per tile)
Write: T tiles (final output)

Total: 0 intermediate full-frame writes
```

### Flexibilität

**v3:**
- Globale Normalisierung → ein target_median für alle Frames
- Channel Split vor Tile-Processing → keine Tile-spezifischen Anpassungen

**v4:**
- Normalisierung während Tile-Loading → Tile-spezifische Parameter möglich
- Debayer während Tile-Loading → Tile-spezifische Debayer-Methoden möglich

## Deferred Channel Split

### Konzept

Frames bleiben im **CFA-Format** bis zum Tile-Processing. Erst wenn eine Tile-Region benötigt wird, erfolgt das Debayering.

```
┌─────────────────────────────────────────┐
│  Frame Storage: CFA Format              │
│                                         │
│  frame_0.fit: [W×H] CFA mosaic          │
│  frame_1.fit: [W×H] CFA mosaic          │
│  ...                                    │
│  frame_N.fit: [W×H] CFA mosaic          │
└────────────┬────────────────────────────┘
             │
             │ (no processing until needed)
             │
             ▼
┌─────────────────────────────────────────┐
│  Tile Processing (Phase 6)              │
│                                         │
│  FOR tile_t with bbox (x0, y0, x1, y1): │
│                                         │
│    FOR frame_f:                         │
│      1. Load CFA region:                │
│         cfa_region = frame_f[y0:y1, x0:x1]│
│                                         │
│      2. Debayer region:                 │
│         rgb_region = debayer(cfa_region)│
│         → R[tile_h, tile_w]             │
│         → G[tile_h, tile_w]             │
│         → B[tile_h, tile_w]             │
│                                         │
│      3. Extract channel:                │
│         channel_region = rgb_region[c]  │
│                                         │
│      4. Process...                      │
└─────────────────────────────────────────┘
```

### Implementierung

```python
def load_tile_region_debayered(
    frame_path: str,
    bbox: Tuple[int, int, int, int],
    channel: str,  # 'R', 'G', or 'B'
    bayer_pattern: str
) -> np.ndarray:
    """
    Lädt eine Tile-Region aus einem CFA-Frame und debayert sie.
    
    Args:
        frame_path: Pfad zum CFA-Frame
        bbox: (x0, y0, x1, y1) Tile-Bounding-Box
        channel: Gewünschter Kanal ('R', 'G', 'B')
        bayer_pattern: Bayer-Pattern ('RGGB', etc.)
    
    Returns:
        Debayerte Tile-Region für den gewünschten Kanal
    """
    x0, y0, x1, y1 = bbox
    
    # Ensure even coordinates for Bayer alignment
    x0 = (x0 // 2) * 2
    y0 = (y0 // 2) * 2
    x1 = ((x1 + 1) // 2) * 2
    y1 = ((y1 + 1) // 2) * 2
    
    # Load CFA region (with small border for debayer)
    border = 4  # For VNG/AHD interpolation
    cfa_region = load_fits_region(
        frame_path,
        (x0 - border, y0 - border, x1 + border, y1 + border)
    )
    
    # Debayer using OpenCV or custom implementation
    rgb_region = debayer_vng(cfa_region, bayer_pattern)
    
    # Extract channel
    channel_idx = {'R': 0, 'G': 1, 'B': 2}[channel]
    channel_region = rgb_region[border:-border, border:-border, channel_idx]
    
    return channel_region
```

### Debayer-Methoden

**VNG (Variable Number of Gradients):**
```python
def debayer_vng(cfa: np.ndarray, pattern: str) -> np.ndarray:
    """
    VNG Debayering (OpenCV implementation).
    
    Adaptive interpolation basierend auf lokalen Gradienten.
    Gute Balance zwischen Qualität und Performance.
    """
    # Map pattern to OpenCV constant
    pattern_map = {
        'RGGB': cv2.COLOR_BAYER_BG2RGB,
        'BGGR': cv2.COLOR_BAYER_RG2RGB,
        'GRBG': cv2.COLOR_BAYER_GB2RGB,
        'GBRG': cv2.COLOR_BAYER_GR2RGB,
    }
    
    return cv2.cvtColor(cfa.astype(np.uint16), pattern_map[pattern])
```

**AHD (Adaptive Homogeneity-Directed):**
```python
def debayer_ahd(cfa: np.ndarray, pattern: str) -> np.ndarray:
    """
    AHD Debayering (LibRaw/dcraw implementation).
    
    Höchste Qualität, aber langsamer als VNG.
    Empfohlen für finale Rekonstruktion.
    """
    # Requires libraw or dcraw binding
    # Not implemented in OpenCV
    pass
```

## Deferred Normalization

### Konzept

Normalisierung erfolgt **on-the-fly** beim Laden der Tile-Regionen, basierend auf den in Phase 3 berechneten globalen Metriken.

```
┌─────────────────────────────────────────┐
│  Phase 3: Global Metrics                │
│                                         │
│  FOR frame_f, channel_c:                │
│    B_f,c = estimate_background(frame_f) │
│                                         │
│  Store: background_map[f, c] = B_f,c    │
└────────────┬────────────────────────────┘
             │
             │ (stored in metadata)
             │
             ▼
┌─────────────────────────────────────────┐
│  Phase 6: Tile Processing               │
│                                         │
│  FOR tile_t:                            │
│    FOR frame_f, channel_c:              │
│      1. Load tile region (debayered)    │
│         I_f,t,c = load_tile_region(...)  │
│                                         │
│      2. Normalize on-the-fly:           │
│         B_f,c = background_map[f, c]    │
│         I'_f,t,c = I_f,t,c / B_f,c      │
│                                         │
│      3. Process normalized tile...      │
└─────────────────────────────────────────┘
```

### Implementierung

```python
def load_tile_normalized(
    frame_path: str,
    bbox: Tuple[int, int, int, int],
    channel: str,
    bayer_pattern: str,
    background: float,
    target_median: float = 209.0
) -> np.ndarray:
    """
    Lädt, debayert und normalisiert eine Tile-Region.
    
    Args:
        frame_path: Pfad zum CFA-Frame
        bbox: Tile-Bounding-Box
        channel: Kanal ('R', 'G', 'B')
        bayer_pattern: Bayer-Pattern
        background: Hintergrundniveau B_f,c
        target_median: Ziel-Median (default: 209.0)
    
    Returns:
        Normalisierte Tile-Region
    """
    # Load and debayer
    tile = load_tile_region_debayered(
        frame_path, bbox, channel, bayer_pattern
    )
    
    # Normalize
    if background > 0:
        tile_normalized = (tile / background) * target_median
    else:
        # Fallback: no normalization
        tile_normalized = tile
    
    return tile_normalized.astype(np.float32)
```

### Normalisierungsmodi

**Background Division (Standard):**
```
I'_f,c = (I_f,c / B_f,c) · target_median

wobei:
  B_f,c = sigma_clipped_median(I_f,c)
  target_median = 209.0 (konfigurierbar)
```

**Per-Channel Normalization:**
```yaml
normalization:
  enabled: true
  mode: background
  per_channel: true  # Separate B_f,c für R, G, B
  target_median: 209.0
```

## Vorteile des Deferred Processing

### 1. Memory-Effizienz
- **Keine Full-Frame-Kopien**: Nur aktuelle Tile-Regionen im RAM
- **Streaming**: Frames werden on-demand geladen
- **Parallele Tiles**: Jeder Worker hat eigenen kleinen Memory-Footprint

### 2. Disk I/O Reduktion
- **Keine Intermediate Files**: Kein Schreiben von debayerten/normalisierten Frames
- **Sequential Reads**: Bessere Disk-Cache-Nutzung
- **Weniger Disk Space**: Keine temporären Full-Frame-Stacks

### 3. Flexibilität
- **Tile-spezifische Parameter**: Debayer-Methode, Normalisierung pro Tile möglich
- **Adaptive Processing**: Unterschiedliche Verarbeitung je nach Tile-Typ
- **Einfache Experimente**: Keine Re-Processing der gesamten Pipeline

### 4. Robustheit
- **Fehler-Isolation**: Fehler in einem Tile betreffen nicht andere
- **Partial Results**: Auch bei Abbruch sind bereits verarbeitete Tiles verfügbar
- **Resume-Fähigkeit**: Einfaches Fortsetzen bei Unterbrechung

## Nachteile und Mitigationen

### Nachteil 1: Wiederholtes Debayering

**Problem:**
- Jedes Tile wird separat debayert
- Bei Overlap werden Regionen mehrfach debayert

**Mitigation:**
- Debayering ist schnell (~10ms pro Tile)
- Parallele Verarbeitung kompensiert
- Gesamtzeit trotzdem geringer als v3 (wegen weniger I/O)

### Nachteil 2: Keine globale Sicht

**Problem:**
- Normalisierung basiert auf globalen Metriken (Phase 3)
- Keine Anpassung während Tile-Processing

**Mitigation:**
- Globale Metriken sind robust (Sigma-Clipping)
- Lokale Anpassungen in TLR (Phase 6) möglich
- Iterative Rekonstruktion kompensiert lokale Variationen

## Validierung

```python
def validate_deferred_processing(tile_data, global_metrics):
    """
    Validiert, dass Deferred Processing korrekt funktioniert.
    """
    checks = []
    
    # Check 1: Tile data is normalized
    for tile in tile_data:
        median = np.median(tile)
        assert 100 < median < 300, f"Tile median {median} out of range"
    checks.append("✓ Tiles are normalized")
    
    # Check 2: No NaN/Inf after debayer+normalize
    for tile in tile_data:
        assert not np.any(np.isnan(tile)), "NaN in tile"
        assert not np.any(np.isinf(tile)), "Inf in tile"
    checks.append("✓ No NaN/Inf in tiles")
    
    # Check 3: Background values used correctly
    for f, c in global_metrics['backgrounds']:
        B = global_metrics['backgrounds'][f, c]
        assert B > 0, f"Invalid background for frame {f}, channel {c}"
    checks.append("✓ Background values valid")
    
    return checks
```

## Konfiguration

```yaml
v4:
  # Deferred processing is always enabled in v4
  # No separate configuration needed
  
  # Debayer method (applied during tile loading)
  debayer_method: "vng"  # or "ahd"
  
  # Normalization (applied during tile loading)
  normalization:
    enabled: true
    mode: background
    per_channel: true
    target_median: 209.0
```

## Nächste Phase

→ **Phase 3: Globale Frame-Metriken** (berechnet Background, Noise, Gradient für Normalisierung)
