# Phase 2: Globale Normalisierung und Frame-Metriken

## Übersicht

Phase 2 normalisiert alle Frames kanalweise und berechnet globale Qualitätsmetriken. Diese Phase ist **identisch** für beide Vorverarbeitungspfade (A und B).

## Ziele

1. Lineare Normalisierung aller Frames (Hintergrundsubtraktion)
2. Berechnung globaler Frame-Metriken (Rauschen, Gradientenergie)
3. Berechnung globaler Qualitätsindizes
4. Vorbereitung für Tile-basierte Analyse

## Input

```python
# Pro Kanal (R, G, B):
frames[f][x, y]  # f = 0..N-1 (Frame-Index)
                 # x, y = Pixel-Koordinaten
                 # Wertebereich: [0, 1] (linear)
```

## Schritt 2.1: Hintergrundschätzung

### Methode: Robuste Statistik

```
┌─────────────────────────────────────────┐
│  Frame f, Kanal c                       │
│  I_f,c[x,y]                             │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 1: Sigma-Clipping                 │
│                                         │
│  μ₀ = median(I_f,c)                     │
│  σ₀ = MAD(I_f,c) * 1.4826               │
│                                         │
│  Iteration (3x):                        │
│    mask = |I - μ| < 3σ                  │
│    μ = mean(I[mask])                    │
│    σ = std(I[mask])                     │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 2: Background Level               │
│                                         │
│  B_f,c = μ_final                        │
│                                         │
│  (Hintergrundniveau des Frames)         │
└─────────────────────────────────────────┘
```

**MAD (Median Absolute Deviation):**
```
MAD = median(|X - median(X)|)
σ ≈ MAD * 1.4826  (für Normalverteilung)
```

### Visualisierung

```
Pixel-Histogramm:
    
    │     ╱╲
    │    ╱  ╲
    │   ╱    ╲___________
    │  ╱    Sky Background
    │ ╱      ↑
    │╱       B_f,c
    └─────────────────────► Pixel Value
    
    Sterne und Nebel →  Rechter Tail
    Hintergrund      →  Peak bei B_f,c
```

## Schritt 2.2: Globale Normalisierung

### Formel (normativ)

```
I'_f,c[x,y] = I_f,c[x,y] / B_f,c
```

**Wichtig:** Diese Normalisierung wird **exakt einmal** durchgeführt, **vor jeder Metrik**.

### Prozess

```
┌─────────────────────────────────────────┐
│  Input: I_f,c[x,y], B_f,c               │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Normalization                          │
│                                         │
│  For each pixel (x,y):                  │
│    I'_f,c[x,y] = I_f,c[x,y] / B_f,c     │
│                                         │
│  Wertebereich nach Normalisierung:      │
│  • Hintergrund ≈ 1.0                    │
│  • Sterne/Nebel > 1.0                   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Output: I'_f,c[x,y] (normalized)       │
└─────────────────────────────────────────┘
```

### Warum Division statt Subtraktion?

```
Subtraktion:  I' = I - B
  Problem: Absolute Skala, nicht vergleichbar

Division:     I' = I / B
  Vorteil: Relative Skala, frames vergleichbar
  Hintergrund = 1.0 für alle Frames
```

## Schritt 2.3: Rauschschätzung

### Methode: Robuste Standardabweichung

```
┌─────────────────────────────────────────┐
│  Normalized Frame I'_f,c[x,y]           │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 1: Hintergrund-Maske              │
│                                         │
│  mask = (I'_f,c < threshold)            │
│  threshold = 1.0 + 3*σ_initial          │
│                                         │
│  → Nur Hintergrund-Pixel, keine Sterne  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 2: Noise Estimation               │
│                                         │
│  σ_f,c = std(I'_f,c[mask])              │
│                                         │
│  (Standardabweichung im Hintergrund)    │
└─────────────────────────────────────────┘
```

### Visualisierung

```
Frame mit Rauschen:

    Hintergrund-Region (mask=True):
    ┌─────────────────────┐
    │ ░░▓░░▓▓░░░▓░░▓░░░░  │  ← Rauschen
    │ ░▓░░░░▓░░▓░░░▓░░▓░  │     σ_f,c = std(diese Pixel)
    │ ░░▓░░░░▓░░░▓░░░░▓░  │
    └─────────────────────┘
    
    Stern-Region (mask=False):
    ┌─────────────────────┐
    │         ███         │  ← Ausgeschlossen
    │        █████        │     (zu hell)
    │       ███████       │
    └─────────────────────┘
```

**Interpretation:**
- Niedriges σ → Gutes Seeing, wenig Rauschen
- Hohes σ → Schlechtes Seeing oder kurze Belichtung

## Schritt 2.4: Gradientenergie

### Methode: Sobel-Operator

```
┌─────────────────────────────────────────┐
│  Normalized Frame I'_f,c[x,y]           │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Sobel Gradients                        │
│                                         │
│  Gx = I' ⊗ [-1  0  1]                  │
│            [-2  0  2]                   │
│            [-1  0  1]                   │
│                                         │
│  Gy = I' ⊗ [-1 -2 -1]                  │
│            [ 0  0  0]                   │
│            [ 1  2  1]                   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Gradient Magnitude                     │
│                                         │
│  G[x,y] = √(Gx² + Gy²)                  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Gradient Energy                        │
│                                         │
│  E_f,c = mean(G²)                       │
│        = (1/N) Σ G[x,y]²                │
└─────────────────────────────────────────┘
```

### Visualisierung

```
Original Frame:        Gradient Magnitude:
                      
    ░░░░░░░░░░            ░░░░░░░░░░
    ░░░░██░░░░            ░░░████░░░
    ░░░████░░░    →       ░░██░░██░░
    ░░░░██░░░░            ░░░████░░░
    ░░░░░░░░░░            ░░░░░░░░░░
    
    Scharfer Stern        Hohe Gradienten an Kanten
    → Hohe E_f,c          → Gute Schärfe
```

**Interpretation:**
- Hohe E → Scharfe Strukturen, gutes Seeing
- Niedrige E → Unscharfe Strukturen, schlechtes Seeing

## Schritt 2.5: Globaler Qualitätsindex

### Formel (normativ)

```
Q_f,c = α·(-B̃_f,c) + β·(-σ̃_f,c) + γ·Ẽ_f,c

wobei (robuste Skalierung mit Median + MAD):
  B̃_f,c = (B_f,c - median(B)) / (1.4826 · MAD(B))
  σ̃_f,c = (σ_f,c - median(σ)) / (1.4826 · MAD(σ))
  Ẽ_f,c = (E_f,c - median(E)) / (1.4826 · MAD(E))
  
  α + β + γ = 1  (Normierung)
  Default: α = 0.4, β = 0.3, γ = 0.3
```

### Prozess

```
┌─────────────────────────────────────────┐
│  Metriken für alle Frames:              │
│  B_f,c, σ_f,c, E_f,c  (f = 0..N-1)      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 1: Robuste Normalisierung         │
│                                         │
│  median_B = median(B_f,c)               │
│  MAD_B    = MAD(B_f,c)                  │
│  B̃_f,c    = (B_f,c - median_B)          │
│             / (1.4826 · MAD_B)          │
│                                         │
│  (analog für σ und E mit median + MAD)  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 2: Gewichtete Kombination         │
│                                         │
│  Q_f,c = α·(-B̃) + β·(-σ̃) + γ·Ẽ          │
│                                         │
│  Vorzeichen:                            │
│  • -B̃: Niedriger Hintergrund ist gut    │
│  • -σ̃: Niedriges Rauschen ist gut       │
│  • +Ẽ: Hohe Gradientenergie ist gut     │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 3: Clamping (Stabilität)          │
│                                         │
│  Q_f,c = clip(Q_f,c, -3, +3)            │
│                                         │
│  → Verhindert extreme Ausreißer         │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Step 4: Exponential Mapping            │
│                                         │
│  G_f,c = exp(Q_f,c)                     │
│                                         │
│  Wertebereich: [e⁻³, e³] ≈ [0.05, 20.1] │
└─────────────────────────────────────────┘
```

### Visualisierung

```
Qualitätsindex über Frames:

G_f,c
  │
20│                    ●
  │              ●         ●
10│        ●                   ●
  │    ●     ●     ●     ●         ●
 5│  ●                               ●
  │●                                   ●
 1├─────────────────────────────────────►
  0   10   20   30   40   50   60   70  Frame f
  
  Hohe Peaks → Beste Frames (gutes Seeing)
  Niedrige Werte → Schlechte Frames
```

### Interpretation der Gewichte

```
α = 0.4  (Hintergrund)
  → Frames mit niedrigem Hintergrund bevorzugen
  → Dunkler Himmel = bessere Bedingungen

β = 0.3  (Rauschen)
  → Frames mit niedrigem Rauschen bevorzugen
  → Weniger Rauschen = bessere Daten

γ = 0.3  (Gradientenergie)
  → Frames mit scharfen Strukturen bevorzugen
  → Hohe Schärfe = gutes Seeing
```

## Schritt 2.6: Validierung

### Checks

```python
def validate_phase2(frames, metrics):
    # Check 1: Gewichtsnormierung
    alpha, beta, gamma = config['weights']
    assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
        "Weights must sum to 1.0"
    
    # Check 2: Clamping
    for f in range(len(frames)):
        Q = metrics[f]['Q']
        assert -3.0 <= Q <= 3.0, \
            f"Q[{f}] = {Q} not clamped to [-3, 3]"
    
    # Check 3: Positive weights
    for f in range(len(frames)):
        G = metrics[f]['G']
        assert G > 0, f"G[{f}] = {G} must be positive"
    
    # Check 4: No NaN/Inf
    for f in range(len(frames)):
        for key in ['B', 'sigma', 'E', 'Q', 'G']:
            val = metrics[f][key]
            assert not np.isnan(val), f"{key}[{f}] is NaN"
            assert not np.isinf(val), f"{key}[{f}] is Inf"
    
    # Check 5: Reasonable ranges
    B_vals = [m['B'] for m in metrics]
    assert all(0 < b < 2 for b in B_vals), \
        "Background levels unreasonable"
    
    sigma_vals = [m['sigma'] for m in metrics]
    assert all(s > 0 for s in sigma_vals), \
        "Noise must be positive"
    
    E_vals = [m['E'] for m in metrics]
    assert all(e >= 0 for e in E_vals), \
        "Gradient energy must be non-negative"
```

## Output-Datenstruktur

```python
# Phase 2 Output
{
    'normalized_frames': {
        'R': np.ndarray,  # shape: (N, H, W), dtype: float32
        'G': np.ndarray,  # normalized by B_f,c
        'B': np.ndarray,
    },
    'global_metrics': {
        'R': [  # Liste von N Dictionaries
            {
                'frame_id': int,
                'B': float,      # Hintergrundniveau
                'sigma': float,  # Rauschen
                'E': float,      # Gradientenergie
                'Q': float,      # Qualitätsindex (clamped)
                'G': float,      # Globales Gewicht exp(Q)
            },
            ...
        ],
        'G': [...],  # analog
        'B': [...],  # analog
    },
    'statistics': {
        'R': {
            'B_mean': float, 'B_std': float,
            'sigma_mean': float, 'sigma_std': float,
            'E_mean': float, 'E_std': float,
            'G_mean': float, 'G_std': float,
        },
        'G': {...},
        'B': {...},
    }
}
```

## Beispiel-Metriken

```
Frame 0 (R-Kanal):
  B_0,R = 0.0234  (Hintergrund)
  σ_0,R = 0.0012  (Rauschen)
  E_0,R = 0.0456  (Gradientenergie)
  
  Nach robuster Skalierung (Median + MAD):
  B̃_0,R = -0.5
  σ̃_0,R = -1.2
  Ẽ_0,R = +1.8
  
  Qualitätsindex:
  Q_0,R = 0.4·(0.5) + 0.3·(1.2) + 0.3·(1.8)
        = 0.2 + 0.36 + 0.54
        = 1.1
  
  Globales Gewicht:
  G_0,R = exp(1.1) ≈ 3.0
  
  → Überdurchschnittlich gutes Frame
```

## Performance-Hinweise

```python
# Effiziente Implementierung
def compute_global_metrics_batch(frames, channel):
    N = len(frames)
    
    # Vektorisierte Hintergrundschätzung
    B = np.array([sigma_clipped_mean(f) for f in frames])
    
    # Normalisierung (broadcast)
    frames_norm = frames / B[:, None, None]
    
    # Rauschen (vektorisiert)
    sigma = np.array([estimate_noise(f) for f in frames_norm])
    
    # Gradientenergie (vektorisiert)
    E = np.array([gradient_energy(f) for f in frames_norm])
    
    # Robuste Skalierung (Median + MAD, vgl. Methodik v3.1)
    def robust_scale(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return (x - med) / (1.4826 * mad)

    B_z = robust_scale(B)
    sigma_z = robust_scale(sigma)
    E_z = robust_scale(E)
    
    # Qualitätsindex
    Q = alpha * (-B_z) + beta * (-sigma_z) + gamma * E_z
    Q = np.clip(Q, -3, 3)
    
    # Gewichte
    G = np.exp(Q)
    
    return frames_norm, B, sigma, E, Q, G
```

## Nächste Phase

→ **Phase 3: Tile-Erzeugung (FWHM-basiert)**
