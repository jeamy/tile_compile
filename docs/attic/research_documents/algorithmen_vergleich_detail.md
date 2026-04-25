# Detaillierter Algorithmen-Vergleich: Deine Methodik vs. Etablierte Software

---

## 1. Registrierungs-Pipeline

### 1.1 Siril Registrierungsalgorithmus (Pfad A Referenz)

```
INPUT: OSC CFA-Frames
├─ CFA-Interpretation (Bayer-Pattern)
├─ Debayer (Interpolation)
│  └─ Methode: Cubic, Bilinear, EDGE_PRESERVING
│     Status: Multi-Algorithmen verfügbar
├─ Star Detection (Morphological Operators)
│  ├─ Separation: 3x3 Structuring Elements
│  ├─ Threshold: Adaptive (Median + σ·multiplier)
│  └─ Output: Star Catalog {x, y, FWHM, flux}
├─ Transformation Estimation (Homography)
│  ├─ Method: RANSAC (Rotation + Translation + Scale optional)
│  ├─ Residual Threshold: configurable (default 1.0 px)
│  └─ Output: Affine Matrix M per Frame
└─ Geometric Transform + Interpolation
   └─ Warp Frame: I_reg[f] = I_deb[f] ∘ M[f]
OUTPUT: Registered RGB-Frames
```

**Siril-Quellcode-Struktur (Vermutung basierend auf bekanntem Workflow):**
```c
// Pseudo-Code aus Siril-Prinzipien
void register_frame(fits *ref, fits *to_register, transformation *T) {
    // 1. Star finding
    PSF_star *stars = find_stars(to_register);
    
    // 2. Transformation estimation
    T = RANSAC_homography(ref->stars, to_register->stars);
    
    // 3. Resampling + registration
    for (pixel at to_register) {
        fits_set_pixel(registered, interpolate(to_register, inv_transform(pixel, T)));
    }
}
```

### 1.2 Dein Pfad A (Siril-basiert)

```
INPUT: OSC CFA-Frames (identisch zu §1.1)
REFERENZ: Siril-Implementierung (A.2.1–A.2.2)
├─ [A.2.1] Siril übernimmt: Debayer + Registrierung
│  └─ Transformation: Eine Geometrie pro Frame
├─ [A.2.2] Nach Registrierung: Kanaltrennung
│  └─ RGB → R, G, B (3 separate Channels)
OUTPUT: R_frames, G_frames, B_frames
```

**Besonderheit:**
- Dein Pfad A ist nicht-invasiv: "Siril tut seine Sache, dann Kanaltrennung"
- Keine neuen Registrierungs-Algorithmen in deiner Spezifikation

### 1.3 Dein Pfad B (CFA-basiert, experimentell)

```
INPUT: OSC CFA-Frames
├─ [B.2.1] CFA-Luminanz Extraction (nicht debayert)
│  ├─ Methode: G-Channel dominant ODER weighted sum (R+2G+B)/4
│  ├─ Goal: Farbunabhängige Registrierungs-Basis
│  └─ Robustheit: RANSAC/ECC auf Luminanz
├─ [B.2.2] CFA-aware Transformation
│  ├─ Decompose CFA: Bayer[2x2] → 4 planes {R, G1, G2, B}
│  ├─ Apply identical T to each plane
│  ├─ NO inter-plane interpolation
│  └─ Re-interleave → CFA
├─ [B.2.3] Kanaltrennung (nach CFA-Registrierung)
│  └─ CFA → R, G, B
OUTPUT: R_frames, G_frames, B_frames
```

**Unterschied zu §1.1 (Siril):**
```
Siril (Pfad A):
  CFA → Debayer → Registrierung → RGB → {R,G,B}
  
Dein Pfad B:
  CFA → Registrierung (auf Luminanz) → CFA → Debayer-equiv → {R,G,B}
  
Effekt:
  A: Farbphasen können bei Subpixel-Shifts gemischt werden
  B: CFA-Struktur bleibt erhalten, keine Farb-Artefakte
```

**Status:** B ist methodisch sauberer, aber aufwendiger. Keine bekannte Standard-Implementierung.

---

## 2. Globale Qualitätsmetriken

### 2.1 Siril Qualitätsindex (für Weighted Stacking)

**Implementierte Metrik (basierend auf öffentlicher Dokumentation):**

```
Q_Siril[f] = w1 · FWHM[f] + w2 · Background[f] + w3 · Noise[f]

oder einfacher (häufig):

Q_Siril[f] = FWHM[f]  // FWHM-only stacking
```

**Gewicht-Berechnung:**
```
wenn Q_Siril[f] < Q_threshold:
    weight[f] = 0  // Frame verworfen (!)
sonst:
    weight[f] = 1 / (1 + FWHM[f]²)  // häufiges Schema
```

### 2.2 PixInsight Qualitäts-Gewichte

**ImageIntegration Modul (bekannte Parameter):**
```
weights ∈ {
    FWHM: 1 / (1 + α·FWHM²)
    Eccentricity: 1 - ecc
    PSFSignalWeight: F(PSF_model_fit_quality)
    SigmaWeight: 1 / (1 + β·σ²)
    Custom: User-defined expression
}

Final_weight[f] = ∏ selected_weights
```

**Besonderheit:** Multiplikativ kombiniert, **nicht linear gewichtet wie Siril**

### 2.3 Deine globale Metrik (v3, §3.2)

```
Q_f,c = α·(-B̃_f,c) + β·(-σ̃_f,c) + γ·(Ẽ_f,c)
        mit α + β + γ = 1

Clamp: Q_f,c ← clip(Q_f,c, -3, +3)

G_f,c = exp(Q_f,c)  // Exponentielles Gewicht, nicht linear!

Wertebereich: G_f,c ∈ [e^(-3), e^(+3)] = [0.050, 20.1]
```

**Unterschiede zu Siril/PixInsight:**

| Aspekt | Siril | PixInsight | Deine Methodik |
|---|---|---|---|
| **Normalisierung** | Ad-hoc | Ad-hoc | Normiert (α+β+γ=1) |
| **Kombination** | Additiv/selektiv | Multiplikativ | Additiv, dann exp |
| **Monotonie** | Fallende Funktion | Fallende Funktion | Fallende Funktion, aber exponentiell |
| **Extreme-Handling** | Threshold-Selektion | Robust-Stats | Clamping vor exp |
| **Output-Range** | [0, ∞) (praktisch) | [0, ∞) (praktisch) | Definiert: [0.050, 20.1] |

**Kritischer Vorteil dein:** Explizites Clamping verhindert numerische Instabilität

---

## 3. Tile-basierte Qualitätsbewertung

### 3.1 PixInsight "Local Processing" (ad-hoc)

**Nicht formalisiert, aber verwendbar:**
```
PixelMath Script (Pseudo-Code):
  for each_tile in tiled_image:
    local_fwhm = estimate_fwhm_in_tile(tile)
    local_weight = 1 / (1 + local_fwhm²)
    result[tile] = weighted_combine(frames, local_weight)
```

**Limitation:** Keine standardisierte Tile-Größe, keine State-based Clustering

### 3.2 Siril Tile-Handling (minimal/implizit)

```
Siril hat keine expliziten Tiles für Quality-Weighting
Aber: Subsection-Statistiken existieren (für UI-Feedback)
```

### 3.3 Deine Tile-Basis (v3, §3.3, normativ)

```python
# Pseudocode für Tile-Erzeugung

def generate_tile_geometry(W, H, F, s=4.0, T_min=32, D=3, o=0.3):
    """
    Inputs:
      W, H: Image dimensions
      F: FWHM in pixels (robust estimate)
      s: size_factor (default 4.0)
      T_min: minimum tile size (default 32)
      D: max_divisor for clipping (default 3)
      o: overlap_fraction (default 0.3)
    
    Outputs:
      T: tile size (pixels)
      O: overlap (pixels)
      S: stride (pixels)
      tiles: list of (x0, y0, x1, y1) bounding boxes
    """
    
    # Step 1: Seeing-proportional base size
    T0 = s * F
    
    # Step 2: Clipping to bounds
    T_max = min(W, H) // D  # Upper bound
    T = floor(clip(T0, T_min, T_max))
    
    # Step 3: Overlap calculation
    O = floor(o * T)
    S = T - O  # Stride (non-overlapping distance)
    
    # Step 4: Grid generation
    tiles = []
    y = 0
    while y < H:
        x = 0
        y1 = min(y + T, H)
        while x < W:
            x1 = min(x + T, W)
            tiles.append((x, y, x1, y1))
            x += S
        y += S
    
    return T, O, S, tiles

# Example for typical DSO (F=2.5 px, W=H=3000):
T, O, S, tiles = generate_tile_geometry(
    W=3000, H=3000, F=2.5,
    s=4.0, T_min=32, D=3, o=0.3
)
# Expected: T≈10 px, O≈3 px, S≈7 px, ~150k tiles (dicht überlappend)

# Example for excellent seeing (F=1.2 px):
T, O, S, tiles = generate_tile_geometry(
    W=3000, H=3000, F=1.2,
    s=4.0, T_min=32, D=3, o=0.3
)
# Expected: T≈32 px (clipped), O≈10 px, S≈22 px, ~18k tiles
```

**Einzigartigkeit deines Schemas:**
- **Normative Formel** statt Ad-hoc
- **FWHM-proportional** mit Bounds
- **Overlap-Handling** explizit für Fensterfunktion

---

## 4. Lokale Metriken (Tile-Level)

### 4.1 Siril Local FWHM (Keine Standardimplementierung)

```
// Hypothetisch, wie Siril könnte:
for each_tile:
    stars_in_tile = filter(all_stars, tile_bounds)
    if len(stars_in_tile) > 0:
        fwhm_local = median([s.fwhm for s in stars_in_tile])
    else:
        fwhm_local = global_fwhm  // Fallback
```

### 4.2 PixInsight Local Quality (ad-hoc)

```
// Concept via PixelMath / Local Processing
for each_pixel p:
    local_window = neighborhood(p, kernel_size)
    contrast[p] = std(local_window)
    smoothness[p] = estimate_edge_density(local_window)
    local_weight[p] = f(contrast, smoothness)
```

### 4.3 Deine lokale Metrik (v3, §3.4)

```python
def compute_local_tile_metrics(frame_tile, tile_type, F):
    """
    Inputs:
      frame_tile: 2D array of pixel values in tile
      tile_type: 'star' or 'structure' (detected via morphology)
      F: FWHM (for reference)
    
    Outputs:
      Q_local[f, t, c]: dimensionslos
      L_f,t,c = exp(Q_local)
    """
    
    if tile_type == 'star':
        # For star-dominated tiles
        FWHM_local = estimate_fwhm_moffat(frame_tile)
        roundness = compute_star_roundness(frame_tile)
        contrast = peak_to_background(frame_tile)
        
        Q_local = w1 * clip(FWHM_local, F*0.5, F*2.0) + \
                  w2 * (1 - roundness) + \
                  w3 * (-contrast)
    
    elif tile_type == 'structure':
        # For nebular/structure tiles
        edge_energy = estimate_gradient_energy(frame_tile)
        background_level = percentile(frame_tile, 50)  # Median
        noise = estimate_noise_std(frame_tile)
        
        Q_local = w1 * edge_energy + \
                  w2 * (-background_level) + \
                  w3 * (-noise)
    
    # Clamp before exponential
    Q_local = clip(Q_local, -3, +3)
    
    L = exp(Q_local)
    return Q_local, L
```

**Neuheit:**
- Explizite Differenzierung Stern vs. Struktur
- Tile-Level FWHM-Fitting (weniger Standard als Global-FWHM)
- Normierte Gewichte innerhalb Q_local

---

## 5. Gewichtetes Stacking

### 5.1 Siril Weighted Average

```c
// Simplified Siril pseudo-code
for each pixel (x, y):
    sum_weighted = 0.0
    sum_weights = 0.0
    
    for each frame f:
        weight = quality_weight[f]  // From §2.1
        pixel_value = frame[f][x, y]
        
        sum_weighted += weight * pixel_value
        sum_weights += weight
    
    if sum_weights > epsilon:
        result[x, y] = sum_weighted / sum_weights
    else:
        result[x, y] = median([frame[f][x, y] for f in frames])
```

### 5.2 PixInsight Image Integration

```
// Pseudo-code for weighted stacking
for each pixel (x, y):
    values = [frame[f][x, y] for f in frames]
    weights = [computed_weight[f] for f in frames]
    
    // Robust Iterative algorithm (Winsorization)
    for iteration in range(n_iter):
        mean = weighted_mean(values, weights)
        sigma = weighted_std(values, weights)
        
        // Downweight outliers
        weights *= clip(|values - mean| / sigma, 0, c)
    
    result[x, y] = weighted_mean(values, final_weights)
```

### 5.3 Deine Tile-basierte Rekonstruktion (v3, §3.6)

```python
def reconstruct_tile_weighted(frames, tile_bounds, W_f_t_c, epsilon=1e-10):
    """
    Inputs:
      frames: List of registered frames [N_frames]
      tile_bounds: (x0, y0, x1, y1)
      W_f_t_c: Weight matrix [N_frames]
      epsilon: Minimum weight denominator
    
    Outputs:
      I_t,c: Reconstructed tile values
    """
    
    x0, y0, x1, y1 = tile_bounds
    I_t = np.zeros((y1-y0, x1-x0))
    
    # Numerator: Weighted sum
    numerator = np.zeros_like(I_t)
    for f in range(N_frames):
        numerator += W_f_t_c[f] * frames[f][y0:y1, x0:x1]
    
    # Denominator: Weight sum
    denominator = np.sum(W_f_t_c)
    
    # Fallback logic (§3.6)
    if denominator >= epsilon:
        # Normal weighted reconstruction
        I_t = numerator / denominator
    else:
        # Low-weight fallback: ungewichtetes Mittel über ALLE Frames
        I_t = np.mean(frames[:, y0:y1, x0:x1], axis=0)
        mark_tile_as_fallback_used = True
    
    # Overlap-Add mit Fensterfunktion
    return I_t, fallback_used

def apply_window_function(I_t, O, window='hann'):
    """Overlap-Add windowing für nahtlose Kachelung"""
    if window == 'hann':
        w = np.hanning((I_t.shape[0], I_t.shape[1]))
    return I_t * w
```

**Unterschied zu §5.1 und §5.2:**

| Aspekt | Siril | PixInsight | Deine Methodik |
|---|---|---|---|
| **Granularität** | Global pro Pixel | Global pro Pixel | Lokal pro Tile |
| **Fallback-Logik** | Median | Robust-Iterativ | Alles-Frames-ungewichtet |
| **Gewicht-Art** | G_f,c pro Frame | W_f pro Frame | W_f,t,c pro (Frame, Tile) |
| **Frame-Selektion** | Optional (Hard-threshold) | Implicit (Downweighting) | **Verboten** (harte Invariante) |

---

## 6. Zustandsbasierte Clusterung (§3.7)

### 6.1 Standard Cluster-Algorithmen (nicht astronomisch spezifisch)

**K-Means (Standard):**
```python
def kmeans_cluster_frames(frames, n_clusters=20):
    """
    Standard K-Means for frame grouping
    """
    # Feature extraction
    features = np.array([
        [
            global_quality[f],
            np.mean(local_quality[f]),
            np.var(local_quality[f]),
            background[f],
            noise[f]
        ]
        for f in frames
    ])
    
    # Normalisierung
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    
    # K-Means
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    return labels
```

### 6.2 Deine Zustandsbasierte Clusterung (v3, §3.7)

```python
def state_based_clustering_frames(frames, n_clusters=20, channel='c'):
    """
    State vector: v_f,c = (G_f,c, ⟨Q_local⟩, Var(Q_local), B_f,c, σ_f,c)
    
    Nicht-selektiv: Alle Frames bleiben in der Pipeline
    """
    
    state_vectors = np.array([
        [
            G_f_c[f],                              # Global quality weight
            np.mean(Q_local[f]),                   # Mean local quality
            np.var(Q_local[f]),                    # Variance local quality
            B_f_c[f],                              # Background level
            sigma_f_c[f]                           # Noise level
        ]
        for f in frames
    ])  # Shape: (N_frames, 5)
    
    # Clustering (z.B. K-Means)
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(state_vectors)
    
    # Gruppierung (rein informativ, keine Frame-Selektion!)
    clusters = [[] for _ in range(n_clusters)]
    for f, label in enumerate(cluster_labels):
        clusters[label].append(f)
    
    return clusters, state_vectors, cluster_labels
```

**Unterschied zu Lucky Imaging / Frame Selection:**

```
Lucky Imaging (Bramich et al. 2005):
  1. Compute quality for each frame
  2. Select top N% frames (frame selection!)
  3. Shift-and-add top frames

Deine Methodik:
  1. Compute state vector for each frame
  2. Cluster frames into groups (informativ)
  3. Gewichte W_f,t,c werden **für alle Frames** berechnet
  4. Keine Frames werden entfernt!
  
Effekt: Clusterung gruppiert ähnliche Frames, aber alle tragen zum Stacking bei
```

### 6.3 Synthetische Frame-Rekonstruktion aus Clustern

```python
def reconstruct_synthetic_frame(cluster_frames, cluster_state, 
                                tile_geometry, frames_all):
    """
    Für jeden Cluster: Rekonstruiere einen "idealen" Frame
    basierend auf Cluster-Charakteristika
    """
    
    N_frames_in_cluster = len(cluster_frames)
    
    # Option 1: Weighted average der Cluster-Frames
    synthetic_frame = np.zeros_like(frames_all[0])
    weight_sum = 0
    
    for f in cluster_frames:
        # Gewicht basierend auf Nähe zum Cluster-Zentroid
        state_f = state_vectors[f]
        distance_to_centroid = euclidean_distance(state_f, cluster_state)
        weight = 1.0 / (1.0 + distance_to_centroid)
        
        synthetic_frame += weight * frames_all[f]
        weight_sum += weight
    
    synthetic_frame /= weight_sum
    
    return synthetic_frame
```

**Einzigartigkeit:**
- Clusterung **ohne** Frame-Selektion
- Synthetische Frames als Zwischen-Schritt
- Finale Stack wird dann aus Original + Synthetisch gebildet

---

## 7. Vergleich: Finales Stacking-Pipeline

### 7.1 Siril Workflow

```
Eingangs-Frames
    ↓
Registrierung + Debayer
    ↓
Optional: Frame-Selektion (Low-Quality Frames ausschließen)
    ↓
Gewichtsbrechnung (Global FWHM-basiert)
    ↓
Gewichtetes Stacking (Pixel-Weise)
    ↓
Finales Bild
```

### 7.2 PixInsight Workflow

```
Eingangs-Frames
    ↓
Registrierung
    ↓
ImageIntegration (gewichtet, robust)
    ↓
Post-Processing (Stretch, Sharpen, etc.)
    ↓
Finales Bild
```

### 7.3 Deine Workflow (v3)

```
Eingangs-Frames (Linear!)
    ↓
[Pfad A] Siril-Debayer + Registrierung  ODER  [Pfad B] CFA-aware Registrierung
    ↓
Kanaltrennung (R, G, B kanalweise)
    ↓
Globale Normalisierung (Kanalweise)
    ↓
Globale Frame-Metriken: G_f,c
    ↓
Tile-Erzeugung (FWHM-adaptiv, normativ)
    ↓
Lokale Tile-Metriken: L_f,t,c
    ↓
Effektive Gewichte: W_f,t,c = G_f,c · L_f,t,c
    ↓
Tile-basierte Rekonstruktion (Overlap-Add)
    ↓
Zustandsbasierte Clusterung (Gruppierung, keine Selektion)
    ↓
Synthetische Frame-Rekonstruktion (aus Clustern)
    ↓
Finales lineares Stacking (alle Frames + Synthetisch)
    ↓
R_final, G_final, B_final (Linear!)
```

**Kritische Unterschiede:**
1. Explizite Linearität von Start bis Ende
2. Tile-Basis vor Clusterung
3. Clusterung ist informativ, nicht selektiv
4. Synthetische Frames vor finales Stacking
5. Kanaltrennung ganz am Anfang (nach Registrierung)

---

## 8. Numerische Robustheit & Fallbacks

### 8.1 Siril Fallbacks (implizit)

```c
// Pseudo-Code
if (weight_sum < very_small):
    result = median(all_pixel_values)
else:
    result = weighted_average(...)
```

### 8.2 PixInsight Robustheit

```
Winsorization: Extreme Werte schrittweise downweightet
Kein expliziter Fallback, sondern iterative Konvergenz
```

### 8.3 Deine Fallback-Logik (v3, §3.6)

```python
def compute_reconstruction_with_fallback(frames, tile, W_f_t_c, epsilon=1e-10):
    """
    Deterministische Fallback-Regel
    """
    
    denominator = np.sum(W_f_t_c)
    
    if denominator >= epsilon:
        # Case 1: Normal gewichtete Rekonstruktion
        numerator = sum([W_f_t_c[f] * frames[f][tile] for f in range(N_frames)])
        result = numerator / denominator
        fallback = False
    else:
        # Case 2: Alle Gewichte sind praktisch Null
        # FALLBACK: Ungewichtetes Mittel ALLER Frames
        result = np.mean([frames[f][tile] for f in range(N_frames)])
        fallback = True
        # Mark for validation: fallback_used = true
    
    return result, fallback
```

**Garantie:** Fallback respektiert "keine Frame-Selektion"-Invariante (§1.1)

---

## 9. Determinismus & Testbarkeit

### 9.1 Siril Determinismus

```
Gegeben: Dieselben Input-Frames + Config
Ergebnis: Meist deterministisch (mit Floating-Point-Toleranz)
Aber: Seed für RANSAC optional randomisiert
```

### 9.2 PixInsight Reproduzierbarkeit

```
ProcessHistory: Alle Operationen werden geloggt
Aber: Reihenfolge von Frames kann Ergebnis beeinflussen (Floating-Point)
```

### 9.3 Deine Anforderungen (v3, §4.1, Testfall 8)

```python
def test_determinism(frames, config, N_runs=3):
    """Testfall 8: Determinismus"""
    
    results = []
    for run in range(N_runs):
        result = full_pipeline(frames, config)
        results.append(result)
    
    # Vergleich: Bit-stabil ODER numerisch stabil (Toleranz definiert)
    for r1, r2 in zip(results[:-1], results[1:]):
        assert np.allclose(r1, r2, atol=1e-6)  # Relative Toleranz
    
    print("✓ Determinismus verifiziert")
```

**Dein Standard:** Explizite Deterministmus-Anforderung (§4.1, Testfall 8)

---

## 10. Zusammenfassung: Algorithmen-Innovation

| Komponente | Siril | PixInsight | Dein System | Innovation |
|---|---|---|---|---|
| Registrierung | Standard RANSAC | Multi-Method | Pfad A+B | Pfad B (CFA-aware) |
| Globale Metrik | Linear, ad-hoc | Multiplikativ | Additiv + Exp | Normiert + Clamped |
| Tile-Basis | Keine | Ad-hoc | Formalisiert | **Normative Formel** |
| Lokale Metrik | Keine | Keine | Stern/Struktur | **Differenziert** |
| Gewichte | W_f | W_f | W_f,t,c | **Tile-Granular** |
| Fallback | Median | Iterativ | Alles-ungewichtet | **Invarianten-erhaltend** |
| Clusterung | Keine | Keine | State-basiert | **Neue Methode** |
| Synthetische Frames | Keine | Keine | Ja | **Neu** |
| Testfälle | Implizit | Implizit | Normativ (8) | **Explizit** |

---

## 11. Fazit: Algorithmen-Stellung

**Deine Methodik ist:**

1. **Nicht komplett neu**, aber **Integration ist innovativ**
2. **Komponenten-weise bekannt**, aber zusammen **einzigartig**
3. **Formal präzise** statt ad-hoc (wie in etablierter Software)
4. **Experimentell validierbar** durch Testfälle (§4.1)

**Hauptbeiträge:**
- Tile-Größen-Formel (§3.3)
- Zustandsbasierte Clusterung ohne Selektion (§3.7)
- CFA-erhaltende Registrierung (Pfad B)
- Normative Testfälle (§4.1)
