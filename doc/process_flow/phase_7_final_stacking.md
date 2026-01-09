# Phase 7: Finales lineares Stacking

## Übersicht

Phase 7 ist die **finale Phase**: Die synthetischen Frames (oder Original-Frames im Reduced Mode) werden zu einem finalen Bild pro Kanal gestackt. Dies ist ein **einfaches lineares Stacking** ohne zusätzliche Gewichtung.

## Ziele

1. Lineares Stacking der synthetischen Frames
2. Kanalweise Verarbeitung (R, G, B separat)
3. Keine zusätzliche Gewichtung (bereits in Phase 6 berücksichtigt)
4. Kein Drizzle oder andere komplexe Verfahren
5. Output: 3 finale FITS-Dateien (R.fit, G.fit, B.fit)

## Zwei Modi

### Normal Mode (N ≥ 200 Frames)

```
Input: Synthetische Frames aus Phase 6
  • K synthetische Frames (K = 15-30)
  • Pro Kanal: F_synth[k][x,y], k = 0..K-1
  • Gewichte: W_synth[k]
```

### Reduced Mode (50 ≤ N < 200 Frames)

```
Input: Original-Frames (Phase 6 übersprungen)
  • N Original-Frames
  • Pro Kanal: I'_f[x,y], f = 0..N-1
  • Gewichte: G_f (globale Gewichte aus Phase 2)
```

## Schritt 7.1: Einfaches lineares Stacking

### Normative Formel

```
Normal Mode:
  I_final,c[x,y] = (1/K) · Σ_k F_synth,k,c[x,y]

Reduced Mode:
  I_final,c[x,y] = (1/N) · Σ_f I'_f,c[x,y]

wobei:
  • KEINE Gewichtung mehr (bereits in synthetischen Frames)
  • Einfacher Durchschnitt
  • Kanalweise (c ∈ {R, G, B})
```

### Warum keine Gewichtung?

```
┌─────────────────────────────────────────┐
│ Gewichtung bereits berücksichtigt:      │
│                                          │
│ Phase 2: Globale Gewichte G_f,c         │
│    ↓                                     │
│ Phase 4: Lokale Gewichte L_f,t,c        │
│    ↓                                     │
│ Phase 5: Tile-Rekonstruktion mit W_f,t,c│
│    ↓                                     │
│ Phase 6: Synthetische Frames             │
│          (gewichtet durch W_synth)       │
│    ↓                                     │
│ Phase 7: Einfaches Mittel                │
│          (keine weitere Gewichtung!)     │
└─────────────────────────────────────────┘

Begründung:
  • Synthetische Frames sind bereits optimal gewichtet
  • Weitere Gewichtung würde zu Doppel-Gewichtung führen
  • Einfaches Mittel ist korrekt und linear
```

### Prozess

```
┌─────────────────────────────────────────┐

│  Input: Synthetische Frames (oder       │
│         Original-Frames im Reduced Mode)│
│  K Frames (oder N im Reduced Mode)      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Initialisierung                        │
│                                          │
│  accumulator = zeros(H, W)              │
│  count = K (oder N)                     │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Akkumulation                           │
│                                          │
│  for k in range(count):                 │
│    accumulator += frames[k]             │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Mittelwert                             │
│                                          │
│  I_final = accumulator / count          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Output: Finales Bild                   │
│  I_final,c[x,y]                         │
└─────────────────────────────────────────┘
```

### Visualisierung: Normal Mode

```
Synthetische Frames (K=20):

Frame 0:      F

rame 1:      Frame 2:      ...  Frame 19:
┌──────┐      ┌──────┐      ┌──────┐           ┌──────┐
│  ★   │      │  ★   │      │  ★   │           │  ★   │
│    ★ │      │    ★ │      │    ★ │           │    ★ │
│      │      │      │      │      │           │      │
│  ★   │      │  ★   │      │  ★   │           │  ★   │
└──────┘      └──────┘      └──────┘           └──────┘
Cluster 0     Cluster 1     Cluster 2          Cluster 19
(45 frames)   (52 frames)   (38 frames)        (25 frames)

           │
           ▼ Einfaches Mittel (1/20 · Σ)
           │
    ┌──────────────┐
    │ Final Stack  │
    │      ★       │  ← Maximale Rauschreduktion
    │        ★     │  ← Optimale Schärfe
    │              │  ← Beste Qualität
    │      ★       │
    └──────────────┘
```

## Schritt 7.2: Speicherung als FITS

### FITS-Header

```python
def create_fits_header(channel, metadata):
    """
    Erstellt FITS-Header mit Metadaten.
    """
    header = fits.Header()
    
    # Basis-Informationen
    header['SIMPLE'] = True
    header['BITPIX'] = -32  # 32-bit float
    header['NAXIS'] = 2
    header['NAXIS1'] = metadata['width']
    header['NAXIS2'] = metadata['height']
    
    # Pipeline-Informationen
    header['PIPELINE'] = 'TileCompile v3'
    header['CHANNEL'] = channel  # 'R', 'G', or 'B'
    header['DATE'] = datetime.now().isoformat()
    
    # Frame-Statistiken
    if metadata['mode'] == 'normal':
        header['MODE'] = 'NORMAL'
        header['NFRAMES'] = metadata['original_frame_count']
        header['NCLUSTERS'] = metadata['synthetic_frame_count']
        header['REDUCTION'] = metadata['reduction_ratio']
    else:
        header['MODE'] = 'REDUCED'
        header['NFRAMES'] = metadata['frame_count']
    
    # Qualitätsmetriken
    header['FWHM'] = metadata['fwhm_median']
    header['TILESIZE'] = metadata['tile_size']
    header['OVERLAP'] = metadata['tile_overlap']
    
    # Registrierung
    header['REGPATH'] = metadata['registration_path']  # 'siril' or 'cfa'
    header['REGRES'] = metadata['registration_residual_median']
    
    # Normalisierung
    header['NORMTYPE'] = 'BACKGROUND_DIVISION'
    
    # Gewichtung
    header['WEIGHTS'] = 'GLOBAL_LOCAL_COMBINED'
    
    # Linearität
    header['LINEAR'] = True
    header['STRETCH'] = False
    
    return header
```

### Speichern

```python
def save_final_stack(image, channel, metadata, output_dir):
    """
    Speichert finales Bild als FITS.
    """
    # Header erstellen
    header = create_fits_header(channel, metadata)
    
    # HDU erstellen
    hdu = fits.PrimaryHDU(data=image, header=header)
    
    # Dateiname
    filename = f"Rekonstruktion_{channel}.fit"
    filepath = os.path.join(output_dir, filename)
    
    # Speichern
    hdu.writeto(filepath, overwrite=True)
    
    print(f"✓ Saved {channel}-channel: {filepath}")
    print(f"  Dimensions: {image.shape}")
    print(f"  Mean: {np.mean(image):.6f}")
    print(f"  Std: {np.std(image):.6f}")
    print(f"  Min: {np.min(image):.6f}")
    print(f"  Max: {np.max(image):.6f}")
```

## Schritt 7.3: Qualitätskontrolle

### Finale Checks

```python
def validate_final_stack(image, channel, metadata):
    """
    Validiert finales gestacktes Bild.
    """
    # Check 1: Keine NaN/Inf
    assert not np.any(np.isnan(image)), f"NaN in final {channel} stack"
    assert not np.any(np.isinf(image)), f"Inf in final {channel} stack"
    
    # Check 2: Positive Werte
    assert np.all(image >= 0), f"Negative values in final {channel} stack"
    
    # Check 3: Vernünftiger Wertebereich
    mean_val = np.mean(image)
    assert 0.5 < mean_val < 2.0, \
        f"Unusual mean value in {channel}: {mean_val}"
    
    # Check 4: Nicht konstant
    std_val = np.std(image)
    assert std_val > 0.001, \
        f"Image appears constant in {channel}: std={std_val}"
    
    # Check 5: Dimensionen
    H, W = image.shape
    assert H == metadata['height'], "Height mismatch"
    assert W == metadata['width'], "Width mismatch"
    
    # Check 6: Linearität (kein Stretch)
    max_val = np.max(image)
    if max_val > 10.0:
        print(f"⚠ Warning: Unusually high max value in {channel}: {max_val}")
    
    print(f"✓ Validation passed for {channel}-channel")
```

### Statistik-Report

```python
def generate_statistics_report(images, metadata):
    """
    Generiert Statistik-Report für finale Stacks.
    """
    report = []
    report.append("=" * 60)
    report.append("FINAL STACK STATISTICS")
    report.append("=" * 60)
    
    # Pro Kanal
    for channel in ['R', 'G', 'B']:
        image = images[channel]
        
        report.append(f"\n{channel}-Channel:")
        report.append(f"  Dimensions: {image.shape[1]} × {image.shape[0]}")
        report.append(f"  Mean:       {np.mean(image):.6f}")
        report.append(f"  Median:     {np.median(image):.6f}")
        report.append(f"  Std:        {np.std(image):.6f}")
        report.append(f"  Min:        {np.min(image):.6f}")
        report.append(f"  Max:        {np.max(image):.6f}")
        report.append(f"  SNR (est):  {np.mean(image) / np.std(image):.2f}")
    
    # Pipeline-Informationen
    report.append("\nPipeline Information:")
    report.append(f"  Mode:              {metadata['mode']}")
    report.append(f"  Original Frames:   {metadata.get('original_frame_count', 'N/A')}")
    
    if metadata['mode'] == 'normal':
        report.append(f"  Synthetic Frames:  {metadata['synthetic_frame_count']}")
        report.append(f"  Reduction Ratio:   {metadata['reduction_ratio']:.1f}:1")
    
    report.append(f"  Registration:      {metadata['registration_path']}")
    report.append(f"  FWHM (median):     {metadata['fwhm_median']:.2f} px")
    report.append(f"  Tile Size:         {metadata['tile_size']} px")
    report.append(f"  Tile Overlap:      {metadata['tile_overlap']} px")
    
    report.append("=" * 60)
    
    return "\n".join(report)
```

## Schritt 7.4: Visualisierung (optional)

### Preview-Bild erstellen

```python
def create_preview_image(images_r, images_g, images_b, output_path):
    """
    Erstellt RGB-Preview (nicht Teil der Methodik, nur zur Kontrolle).
    """
    # Normalisierung für Anzeige (nicht-linear, nur für Preview!)
    def stretch_for_display(image, percentile=99.5):
        vmin = np.percentile(image, 0.5)
        vmax = np.percentile(image, percentile)
        stretched = np.clip((image - vmin) / (vmax - vmin), 0, 1)
        return stretched
    
    # Stretch pro Kanal
    r_stretched = stretch_for_display(images_r)
    g_stretched = stretch_for_display(images_g)
    b_stretched = stretch_for_display(images_b)
    
    # RGB kombinieren
    rgb = np.dstack([r_stretched, g_stretched, b_stretched])
    
    # Als PNG speichern
    from PIL import Image
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    img.save(output_path)
    
    print(f"✓ Preview saved: {output_path}")
```

**Wichtig:** Preview ist **nicht Teil der Methodik**! Die finalen FITS-Dateien bleiben linear.

## Schritt 7.5: Kompletter Workflow

### Implementierung

```python
def phase7_final_stacking(input_frames, metadata, config, output_dir):
    """
    Phase 7: Finales lineares Stacking.
    
    Args:
        input_frames: Dict mit 'R', 'G', 'B' Arrays
                      Normal Mode: (K, H, W) synthetische Frames
                      Reduced Mode: (N, H, W) Original-Frames
        metadata: Pipeline-Metadaten
        config: Konfiguration
        output_dir: Output-Verzeichnis
    
    Returns:
        final_stacks: Dict mit finalen R, G, B Bildern
    """
    final_stacks = {}
    
    # Pro Kanal
    for channel in ['R', 'G', 'B']:
        print(f"\nProcessing {channel}-channel...")
        
        frames = input_frames[channel]
        K = frames.shape[0]  # Anzahl Frames (K oder N)
        
        # Einfaches lineares Stacking
        print(f"  Stacking {K} frames...")
        final = np.mean(frames, axis=0)
        
        # Validierung
        validate_final_stack(final, channel, metadata)
        
        # Speichern
        save_final_stack(final, channel, metadata, output_dir)
        
        final_stacks[channel] = final
    
    # Statistik-Report
    report = generate_statistics_report(final_stacks, metadata)
    print(report)
    
    # Report speichern
    report_path = os.path.join(output_dir, "final_stack_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Optional: Preview
    if config.get('create_preview', False):
        preview_path = os.path.join(output_dir, "preview.png")
        create_preview_image(
            final_stacks['R'],
            final_stacks['G'],
            final_stacks['B'],
            preview_path
        )
    
    return final_stacks
```

## Output-Struktur

```
output_dir/
├── Rekonstruktion_R.fit    # Fi

naler R-Kanal (linear)
├── Rekonstruktion_G.fit    # Finaler G-Kanal (linear)
├── Rekonstruktion_B.fit    # Finaler B-Kanal (linear)
├── final_stack_report.txt  # Statistik-Report
└── preview.png             # Optional: RGB-Preview (nicht-linear)
```

## Beispiel-Output

```
Processing R-channel...
  Stacking 20 frames...
  ✓ Validation passed for R-channel
  ✓ Saved R-channel: /output/Rekonstruktion_R.fit
    Dimensions: (2048, 4096)
    Mean: 1.002345
    Std: 0.123456
    Min: 0.987654
    Max: 3.456789

Processing G-channel...
  Stacking 20 frames...
  ✓ Validation passed for G-channel
  ✓ Saved G-channel: /output/Rekonstruktion_G.fit
    Dimensions: (2048, 4096)
    Mean: 1.001234
    Std: 0.098765
    Min: 0.976543
    Max: 2.987654

Processing B-channel...
  Stacking 20 frames...
  ✓ Validation passed for B-channel
  ✓ Saved B-channel: /output/Rekonstruktion_B.fit
    Dimensions: (2048, 4096)
    Mean: 1.003456
    Std: 0.145678
    Min: 0.965432
    Max: 4.123456

============================================================
FINAL STACK STATISTICS
============================================================

R-Channel:
  Dimensions: 4096 × 2048
  Mean:       1.002345
  Median:     0.998765
  Std:        0.123456
  Min:        0.987654
  Max:        3.456789
  SNR (est):  8.12

G-Channel:
  Dimensions: 4096 × 2048
  Mean:       1.001234
  Median:     0.997654
  Std:        0.098765
  Min:        0.976543
  Max:        2.987654
  SNR (est):  10.14

B-Channel:
  Dimensions: 4096 × 2048
  Mean:       1.003456
  Median:     0.999876
  Std:        0.145678
  Min:        0.965432
  Max:        4.123456
  SNR (est):  6.89

Pipeline Information:
  Mode:              normal
  Original Frames:   800
  Synthetic Frames:  20
  Reduction Ratio:   40.0:1
  Registration:      siril
  FWHM (median):     3.24 px
  Tile Size:         64 px
  Tile Overlap:      16 px
============================================================
```

## Was kommt NACH Phase 7?

### RGB/LRGB-Kombination (außerhalb der Methodik)

```
Die 3 finalen FITS-Dateien können nun kombin

iert werden:

Option 1: RGB-Kombination
  ┌─────────────────────────────────┐
  │ R.fit + G.fit + B.fit → RGB.fit │
  │                                  │
  │ Einfache Kanal-Kombination       │
  │ Kein Teil der Methodik           │
  └─────────────────────────────────┘

Option 2: LRGB-Kombination
  ┌─────────────────────────────────┐
  │ L.fit (Luminanz, separat)        │
  │ R.fit, G.fit, B.fit (Farbe)      │
  │ → LRGB.fit                       │
  │                                  │
  │ Komplexere Kombination           │
  │ Kein Teil der Methodik           │
  └─────────────────────────────────┘

Option 3: Weiterverarbeitung in PixInsight/Siril
  ┌─────────────────────────────────┐
  │ Importiere R.fit, G.fit, B.fit   │
  │ → Stretching, Farbkalibrierung   │
  │ → Finales Bild                   │
  └─────────────────────────────────┘
```

**Wichtig:** RGB/LRGB-Kombination ist **explizit außerhalb** der Methodik!

## Zusammenfassung: Gesamte Pipeline

```
Phase 0: Preprocessing Path Selection
  ↓
Ph

ase 1: Registration & Channel Separation
  ↓ (R, G, B getrennt ab hier)
Phase 2: Global Normalization & Frame Metrics
  ↓
Phase 3: Tile Generation (FWHM-based)
  ↓
Phase 4: Local Tile Metrics
  ↓
Phase 5: Tile-based Reconstruction
  ↓
Phase 6: State-based Clustering & Synthetic Frames
  ↓ (Normal Mode: K synthetische Frames)
  ↓ (Reduced Mode: N Original-Frames)
Phase 7: Final Linear Stacking
  ↓
Output: Rekonstruktion_R.fit
        Rekonstruktion_G.fit
        Rekonstruktion_B.fit

─────────────────────────────────────
ENDE DER METHODIK
─────────────────────────────────────

Optional (außerhalb):
  → RGB/LRGB-Kombination
  → Stretching
  → Farbkalibrierung
  → Finales Bild
```

## Kernprinzipien (Wiederholung)

1. ✓ **Linearität**: Keine nichtlinearen Operationen
2. ✓ **Keine Frame-Selektion**: Alle Frames werden verwendet
3. ✓ **Kanalgetrennt**: R, G, B unabhängig verarbeitet
4. ✓ **Streng linear**: Keine Rückkopplungen
5. ✓ **Deterministisch**: Gleiche Inputs → gleiche Outputs
6. ✓ **Tile-basiert**: Lokale Qualitätsbewertung
7. ✓ **Gewichtet**: Global × Lokal
8. ✓ **Einfaches finales Stacking**: Keine zusätzliche Gewichtung

## Performance-Hinweise

```python
# Memory-effizientes Stacking
def stack_large_frames(frames, chunk_size=100):
    """
    Stackt große Frame-Arrays in Chunks (memory-effizient).
    """
    K, H, W = frames.shape
    
    # Initialisierung
    result = np.zeros((H, W), dtype=np.float64)  # float64 für Akkumulation
    
    # Chunk-weise Akkumulation
    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        chunk = frames[start:end]
        result += np.sum(chunk, axis=0)
    
    # Mittelwert
    result /= K
    
    # Zurück zu float32
    return result.astype(np.float32)

# Paralleles Stacking (alle Kanäle gleichzeitig)
from concurrent.futures import ThreadPoolExecutor

def stack_all_channels_parallel(frames_dict):
    """
    Stackt R, G, B parallel.
    """
    def stack_channel(channel):
        return channel, np.mean(frames_dict[channel], axis=0)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(stack_channel, ['R', 'G', 'B'])
    
    return {channel: stack for channel, stack in results}
```

## Ende der Pipeline

**Die Methodik ist mit Phase 7 abgeschlossen.**

Die finalen FITS-Dateien sind:
- ✓ Linear
- ✓ Kanalgetrennt
- ✓ Optimal gewichtet
- ✓ Bereit für weitere Verarbeitung (außerhalb der Methodik)
