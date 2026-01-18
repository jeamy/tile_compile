# Phase 1: Registrierung und Kanaltrennung

## Übersicht

Phase 1 umfasst die geometrische Ausrichtung aller Frames und die Trennung in separate Farbkanäle. Es gibt zwei Registrierungs-Engines:

| Engine | Konfiguration | Status | Empfehlung |
|--------|---------------|--------|------------|
| **opencv_cfa** | `registration.engine: opencv_cfa` | ✅ Aktiv | **Empfohlen** |
| **siril** | `registration.engine: siril` | Legacy | Nur bei Bedarf |

## Ziele

1. Alle Frames geometrisch ausrichten (Sub-Pixel-Genauigkeit)
2. Kanäle trennen (R, G, B)
3. Linearität bewahren
4. Keine Frame-Selektion
5. Für `registration.engine=opencv_cfa`: eine einzige globale (euclidean/affine)
   Transformation pro Frame, bestimmt über dichte ECC-Korrelation auf einer
   CFA-Luminanz, mit CFA-aware Warping (Subplanes) – analog zur in Methodik
   v3 beschriebenen Pfad-B-Anforderung „eine globale Transformation je Frame
   + CFA-aware Interpolation, keine Bayer-Phasen-Mischung“.

## Path A: Siril-basierte Registrierung (Legacy)

>   **Hinweis:** Siril-Registrierung ist weiterhin verfügbar
> - Debayer vor Registrierung → Farbmischung beim Warp
> - Externe Abhängigkeit: Siril muss installiert sein
>
> **Konfiguration:** `registration.engine: siril`
> **Script:** `siril_scripts/siril_register_osc.ssf` (oder eigenes via `registration.siril_script`)

### Schritt A.1: Debayer

```
┌─────────────────────────────────────┐
│  Input: OSC Frame (Bayer Pattern)   │
│                                     │
│  R G R G R G    ┌─────────────┐     │
│  G B G B G B    │   Debayer   │     │
│  R G R G R G ──►│ Interpolate │──►  │
│  G B G B G B    │   (VNG/AHD) │     │
│  R G R G R G    └─────────────┘     │
│                                     │
│  Output: RGB Frame (3 channels)     │
│  R[x,y], G[x,y], B[x,y]             │
└─────────────────────────────────────┘
```

**Interpolationsmethoden:**
- **VNG** (Variable Number of Gradients): Adaptiv, gute Qualität
- **AHD** (Adaptive Homogeneity-Directed): Sehr gute Qualität, langsamer
- **Bilinear**: Schnell, einfach, geringere Qualität

### Schritt A.2: Sternfindung

```
┌──────────────────────────────────────────┐
│  Luminance = 0.299*R + 0.587*G + 0.114*B │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│  Star Detection (Siril)                  │
│  • Threshold-basiert                     │
│  • Lokale Maxima                         │
│  • PSF-Fitting (Gaussian/Moffat)         │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│  Star List pro Frame                     │
│  stars[i] = {x, y, flux, fwhm, ...}      │
└──────────────────────────────────────────┘
```

**Qualitätskriterien:**
- Mindestens 50-100 Sterne pro Frame
- FWHM-Varianz < 30%
- Elongation < 0.4

### Schritt A.3: Transformationsschätzung

```
Reference Frame (f=0)          Target Fram

e (f=i)
     Stars: {s₁, s₂, ...}          Stars: {t₁, t₂, ...}
            │                              │
            └──────────┬───────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Star Matching       │
            │  • Nearest Neighbor  │
            │  • Triangle Matching │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  RANSAC              │
            │  • Outlier Rejection │
            │  • Robust Estimation │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Transformation T    │
            │  [a b tx]            │
            │  [c d ty]            │
            └──────────────────────┘
```

**Transformationsmodell:**
- Rotation + Translation (Rigid)
- Optional: Affine (mit Skalierung/Scherung)

**Mathematik:**
```
x' = a*x + b*y + tx
y' = c*x + d*y + ty

Für Rotation θ und Translation (tx, ty):
a = cos(θ), b = -sin(θ)
c = sin(θ), d = cos(θ)
```

### Schritt A.4: Anwendung der Transformation

```
┌─────────────────────────────────────────┐
│  Input: RGB Frame + Transformation T    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Pro Kanal (R, G, B):                   │
│                                         │
│  For each pixel (x', y') in output:     │
│    1. Inverse transform: (x,y) = T⁻¹(x',y')│
│    2. Interpolate at (x,y)              │
│       • Lanczos3 (best quality)         │
│       • Cubic (good balance)            │
│       • Linear (fast)                   │
│    3. Write to output[x',y']            │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Output: Registered RGB Frame           │
│  R'[x,y], G'[x,y], B'[x,y]              │
└─────────────────────────────────────────┘
```

**Interpolation-Kernel:**
```
Lanczos3: L(x) = sinc(x) * sinc(x/3)  für |x| < 3
          L(x) = 0                     sonst
```

### Schritt A.5: Kanaltrennung

```
┌─────────────────────────────────────────┐
│  Registered RGB Frames                  │
│  frames[f] = {R[x,y], G[x,y], B[x,y]}   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Channel Separation                     │
│                                         │
│  R_frames[f][x,y] = frames[f].R[x,y]    │
│  G_frames[f][x,y] = frames[f].G[x,y]    │
│  B_frames[f][x,y] = frames[f].B[x,y]    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  3 separate channel stacks              │
│  • R_frames[0..N-1][W,H]                │
│  • G_frames[0..N-1][W,H]                │
│  • B_frames[0..N-1][W,H]                │
└─────────────────────────────────────────┘
```

**Wichtig:** Ab hier keine kanalübergreifenden Operationen mehr!

## Path B: CFA-basierte Registrierung (Empfohlen)

> ✅ **Empfohlen:** `registration.engine: opencv_cfa` (Standard)
> - Cosmetic Correction entfernt Hotpixels VOR dem Warp
> - CFA-aware Warping verhindert Farbmischung
> - Debayer erst nach Stacking → maximale Qualität
> - Keine externe Abhängigkeit (nur OpenCV/Python)

### Schritt B.1: CFA-Luminanz-Extraktion

```
┌─────────────────────────────────────────┐
│  Input: CFA Frame (Bayer Pattern)       │
│                                         │
│  R  G  R  G  R  G                       │
│  G  B  G  B  G  B                       │
│  R  G  R  G  R  G                       │
│  G  B  G  B  G  B                       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Luminance Extraction (no interpolation)│
│                                         │
│  Method 1: G-dominant                   │
│    L[x,y] = G[x,y] if G pixel           │
│    L[x,y] = 0.5*(G_neighbors) else      │
│                                         │
│  Method 2: Weighted sum                 │
│    L[x,y] = 0.3*R + 0.6*G + 0.1*B       │
│    (from available samples)             │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  CFA Luminance Image L[x,y]             │
└─────────────────────────────────────────┘
```

### Schritt B.2: Referenz-Frame-Auswahl

Die Wahl des Referenz-Frames ist kritisch für die Registrierungsqualität, besonders bei Aufnahmen mit kontinuierlicher Drift (z.B. Alt-Az-Montierung).

**Strategie: Mittleres Drittel + Beste Qualität**

```
┌────────────────────────────────────────────────────────────┐
│  Frame-Sequenz: [0, 1, 2, ... , N-1]                       │
│                                                            │
│  ├─────────┼─────────────────────┼─────────┤               │
│  │ Erstes  │   MITTLERES DRITTEL │ Letztes │               │
│  │ Drittel │   (bevorzugt)       │ Drittel │               │
│  │ [0..N/3]│   [N/3 .. 2N/3]     │[2N/3..N]│               │
│  └─────────┴─────────────────────┴─────────┘               │
│                      ↓                                     │
│            Wähle Frame mit meisten Sternen                 │
│            innerhalb des mittleren Drittels                │
└────────────────────────────────────────────────────────────┘
```

**Warum mittleres Drittel?**
- Bei kontinuierlicher Drift ist die **maximale Drift-Distanz** am geringsten
- Frame 1 und Frame N haben beide ähnliche Distanz zur Referenz
- ECC konvergiert besser bei geringerer Drift
- Verhindert längliche Sterne durch falsche Rotationsschätzung

**Beispiel:** Bei 264 Frames wird ein Frame zwischen Index 88-176 gewählt.

### Schritt B.2a: Kettenregistrierung (Chain Registration)

Statt alle Frames auf einen einzelnen Referenz-Frame zu registrieren (was bei Feldrotation versagt), verwendet tile-compile **Kettenregistrierung**:

```
┌─────────────────────────────────────────────────────────────┐
│  Kettenregistrierung                                        │
│                                                             │
│  Rückwärts-Kette:                                          │
│    Frame[ref-1] → Frame[ref]                               │
│    Frame[ref-2] → Frame[ref-1] → Frame[ref]                │
│    ...                                                      │
│                                                             │
│  Vorwärts-Kette:                                           │
│    Frame[ref+1] → Frame[ref]                               │
│    Frame[ref+2] → Frame[ref+1] → Frame[ref]                │
│    ...                                                      │
│                                                             │
│  Akkumulierte Transformation:                               │
│    T[i] = T_step[i] ∘ T_step[i±1] ∘ ... ∘ T_step[ref±1]    │
└─────────────────────────────────────────────────────────────┘
```

**Vorteile der Kettenregistrierung:**
- **Minimale Drift pro Schritt**: Benachbarte Frames sind sehr ähnlich
- **Bessere ECC-Konvergenz**: Hohe Korrelation zwischen Nachbarn
- **Feldrotation-tolerant**: Kleine Rotationsschritte akkumulieren korrekt
- **Keine länglichen Sterne an den Rändern**

**Wichtig (Implementierungsdetails):**
- Es werden **keine Frames verworfen**, auch wenn die Sternanzahl unter `min_star_matches` liegt (nur Warnung). Das verhindert Datenverlust.
- **Cosmetic Correction** wird **vor** Luminanzbildung/ECC angewendet, damit Hotpixel nicht durch Interpolation „verschmiert“ werden.
- Die akkumulierte Transformation muss zur OpenCV-Konvention passen:
  - `cv2.warpAffine(..., flags=cv2.WARP_INVERSE_MAP)` erwartet eine Invers-Abbildung.
  - Bei der Ketten-Akkumulation muss daher die Komposition konsistent erfolgen (sonst wirken die Warps „fast richtig“, der Stack wird aber unscharf).

**ECC (Enhanced Correlation Coefficient):**
```
Maximiere: ρ(T) = Σ I₀(x) · Iᵢ(T(x)) / √(Σ I₀² · Σ Iᵢ²)
```

**Transformationstypen:**
- `allow_rotation: false` → MOTION_TRANSLATION (nur dx, dy)
- `allow_rotation: true` → MOTION_EUCLIDEAN (dx, dy, θ)

### Schritt B.2b: Cosmetic Correction (Hotpixel-Entfernung)

 Vor der Registrierung (Luminanzbildung/ECC) und vor dem Warp wird eine Cosmetic Correction angewendet, um Hotpixels zu entfernen. Dies verhindert "Walking Noise" - das Verschmieren von Hotpixels durch die Interpolation beim Warping.

```
┌─────────────────────────────────────────┐
│  Input: CFA Frame (vor Warp)            │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Cosmetic Correction                    │
│                                         │
│  1. Robuste Statistik berechnen:        │
│     med = median(frame)                 │
│     mad = median(|frame - med|)         │
│     σ_robust = 1.4826 × mad             │
│                                         │
│  2. Hotpixel-Maske erstellen:           │
│     threshold = med + 8 × σ_robust      │
│     hot_mask = frame > threshold        │
│                                         │
│  3. Hotpixels durch lokalen Median      │
│     der 8 Nachbarn ersetzen:            │
│     frame[hot_mask] = local_median      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Output: Bereinigtes CFA Frame          │
│  (keine Hotpixels mehr)                 │
└─────────────────────────────────────────┘
```

**Warum vor dem Warp?**
- Hotpixels sind auf dem Sensor an festen Positionen
- Beim Warp (bilineare Interpolation) werden sie auf mehrere Pixel verteilt
- Nach dem Warp erscheinen sie an verschiedenen Positionen in verschiedenen Frames
- Sigma-Clipping kann sie dann nicht mehr effektiv entfernen

### Schritt B.3: CFA-aware Transformation

```
┌─────────────────────────────────────────┐
│  Input: CFA Frame + Transformation T    │
│                                         │
│  R  G  R  G     Bayer Pattern:          │
│  G  B  G  B     Phase (0,0): R          │
│  R  G  R  G     Phase (0,1): G          │
│  G  B  G  B     Phase (1,0): G          │
│                 Phase (1,1): B          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 1: Decompose to 4 Subplanes       │
│                                         │
│  R_plane[x,y]  = CFA[2x,   2y  ]        │
│  G1_plane[x,y] = CFA[2x,   2y+1]        │
│  G2_plane[x,y] = CFA[2x+1, 2y  ]        │
│  B_plane[x,y]  = CFA[2x+1, 2y+1]        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 2: Apply SAME Transform to each   │
│                                         │
│  R'_plane  = Transform(R_plane,  T)     │
│  G1'_plane = Transform(G1_plane, T)     │
│  G2'_plane = Transform(G2_plane, T)     │
│  B'_plane  = Transform(B_plane,  T)     │
│                                         │
│  (Lanczos3 interpolation per plane)     │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 3: Re-interleave to CFA           │
│                                         │
│  CFA'[2x,   2y  ] = R'_plane[x,y]       │
│  CFA'[2x,   2y+1] = G1'_plane[x,y]      │
│  CFA'[2x+1, 2y  ] = G2'_plane[x,y]      │
│  CFA'[2x+1, 2y+1] = B'_plane[x,y]       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Output: Registered CFA Frame           │
│  (no color phase mixing!)               │
└─────────────────────────────────────────┘
```

**Kritisch:** Keine Interpolation zwischen Bayer-Phasen!

### Schritt B.4: Debayer und Kanaltrennung

```
┌─────────────────────────────────────────┐
│  Registered CFA Frames                  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Debayer (VNG/AHD)                      │
│  CFA → RGB                              │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Channel Separation                     │
│  RGB → R, G, B                          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  3 separate channel stacks              │
│  • R_frames[0..N-1][W,H]                │
│  • G_frames[0..N-1][W,H]                │
│  • B_frames[0..N-1][W,H]                │
└─────────────────────────────────────────┘
```

## Qualitätsmetriken

### Registrierungsresiduum

```
Für jeden Frame f:
  1. Detektiere Sterne in Reference und Frame f
  2. Matche Sterne: (x_ref, y_ref) ↔ (x_f, y_f)
  3. Berechne Residuum nach Transformation:
  
     r_i = √[(x'_i - x_ref,i)² + (y'_i - y_ref,i)²]
     
     wobei (x'_i, y'_i) = T(x_f,i, y_f,i)
  
  4. Robuste Statistik:
     residuum_f = median(r_i)
```

**Qualitätsstufen:**
- ✓ Exzellent: < 0.3 px
- ✓ Gut: < 0.5 px
- ⚠ Akzeptabel: < 1.0 px
- ✗ Inakzeptabel: ≥ 1.0 px

### Stern-Elongation

```
Für jeden detektierten Stern:
  1. Berechne 2. Momente:
     μ₂₀ = Σ (x-x̄)² · I(x,y)
     μ₀₂ = Σ (y-ȳ)² · I(x,y)
     μ₁₁ = Σ (x-x̄)(y-ȳ) · I(x,y)
  
  2. Eigenwerte der Kovarianzmatrix:
     λ₁, λ₂ = eigenvalues([μ₂₀ μ₁₁; μ₁₁ μ₀₂])
  
  3. Elongation:
     e = 1 - λ₂/λ₁  (λ₁ ≥ λ₂)
```

**Qualitätsstufen:**
- ✓ Rund: e < 0.2
- ⚠ Leicht elongiert: e < 0.4
- ✗ Stark elongiert: e ≥ 0.4

## Validierung nach Phase 1

```python
def validate_phase1(frames_r, frames_g, frames_b, metadata):
    checks = []
    
    # Check 1: Frame count
    N = len(frames_r)
    assert N == len(frames_g) == len(frames_b)
    checks.append(f"✓ Frame count: {N}")
    
    # Hinweis auf Moduswahl gemäß Methodik v3.1
    if N < 50:
        checks.append("⚠ Degraded Mode: < 50 Frames (stark reduzierte Statistik, kein Clustering)")
    elif N < 200:
        checks.append("⚠ Reduced Mode: 50–199 Frames (kein Clustering, keine synthetischen Frames)")
    else:
        checks.append("✓ Normal Mode: N ≥ 200 Frames")
    
    # Check 2: Dimensions
    H, W = frames_r[0].shape
    for f in range(N):
        assert frames_r[f].shape == (H, W)
        assert frames_g[f].shape == (H, W)
        assert frames_b[f].shape == (H, W)
    checks.append(f"✓ Dimensions: {W}x{H}")
    
    # Check 3: Registration quality
    residuals = [metadata[f]['registration_residual'] for f in range(N)]
    median_res = np.median(residuals)
    max_res = np.max(residuals)
    
    if median_res < 0.5:
        checks.append(f"✓ Registration: {median_res:.3f} px (excellent)")
    elif median_res < 1.0:
        checks.append(f"⚠ Registration: {median_res:.3f} px (acceptable)")
    else:
        raise ValueError(f"✗ Registration: {median_res:.3f} px (failed)")
    
    # Check 4: No NaN/Inf
    for channel_name, frames in [('R', frames_r), ('G', frames_g), ('B', frames_b)]:
        for f in range(N):
            assert not np.any(np.isnan(frames[f])), f"NaN in {channel_name}[{f}]"
            assert not np.any(np.isinf(frames[f])), f"Inf in {channel_name}[{f}]"
    checks.append("✓ No NaN/Inf values")
    
    # Check 5: Linearity (no stretch)
    for channel_name, frames in [('R', frames_r), ('G', frames_g), ('B', frames_b)]:
        max_val = np.max([np.max(f) for f in frames])
        assert max_val < 1.1, f"Possible stretch in {channel_name}: max={max_val}"
    checks.append("✓ Data is linear")
    
    return checks
```

## Output-Datenstruktur

```python
# Phase 1 Output
{
    'frames': {
        'R': np.ndarray,  # shape: (N, H, W), dtype: float32
        'G': np.ndarray,  # shape: (N, H, W), dtype: float32
        'B': np.ndarray,  # shape: (N, H, W), dtype: float32
    },
    'metadata': {
        'frame_count': int,
        'dimensions': (W, H),
        'registration_method': str,  # 'siril' or 'cfa'
        'registration_residuals': List[float],  # per frame
        'star_counts': List[int],  # per frame
        'elongations': List[float],  # per frame
    }
}
```

## Nächste Phase

→ **Phase 2: Globale Normalisierung und Frame-Metriken**
