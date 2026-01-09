# Phase 6: Zustandsbasierte Clusterung und synthetische Frames

## Übersicht

Phase 6 gruppiert Frames basierend auf ihren Qualitätszuständen und rekonstruiert **synthetische Frames** pro Cluster. Dies reduziert Rauschen und erhöht die Effizienz des finalen Stackings.

## Ziele

1. Frames in Cluster gruppieren (basierend auf Qualitätszustand)
2. Synthetische Frames pro Cluster rekonstruieren
3. Frame-Anzahl reduzieren (N → K Cluster)
4. Rauschen reduzieren durch Cluster-Stacking
5. Vorbereitung für finales lineares Stacking

## Wichtig: Reduced Mode

```
Bei Frame-Anzahl 50-199:
  → Clusterung wird ÜBERSPRUNGEN
  → Direkt zu Phase 7 (finales Stacking)
  → Keine synthetischen Frames
```

## Input

```python
# Aus Phase 2:
global_metrics[c][f] = {
    'B': float,      # Hintergrund
    'sigma': float,  # Rauschen
    'E': float,      # Gradientenergie
    'G': float,      # Globales Gewicht
}

# Aus Phase 4:
local_metrics[c][(f, t)] = {
    'Q_local': float,  # Lokaler Qualitätsindex
    'L': float,        # Lokales Gewicht
}

# Aus Phase 5:
reconstructed_tiles[c][t]  # Rekonstruierte Tiles
```

## Schritt 6.1: Zustandsvektor-Konstruktion

### Formel (normativ)

```
Für jeden Frame f, Kanal c:

v_f,c = (G_f,c, ⟨Q_local⟩_f,c, Var(Q_local)_f,c, B_f,c, σ_f,c)

wobei:
  G_f,c              - Globales Gewicht
  ⟨Q_local⟩_f,c      - Mittelwert der lokalen Q über alle Tiles
  Var(Q_local)_f,c   - Varianz der lokalen Q über alle Tiles
  B_f,c              - Hintergrundniveau
  σ_f,c              - Rauschen
```

### Prozess

```
┌─────────────────────────────────────────┐
│  Frame f, Kanal c                       │
│  Global: G, B, σ, E                     │
│  Lokal: Q_local für alle Tiles          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Berechne Tile-Statistiken              │
│                                          │
│  Q_locals = [Q_local_f,t,c for all t]   │
│                                          │
│  mean_Q = mean(Q_locals)                │
│  var_Q = var(Q_locals)                  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Konstruiere Zustandsvektor             │
│                                          │
│  v_f,c = [G_f,c,                        │
│           mean_Q,                       │
│           var_Q,                        │
│           B_f,c,                        │
│           σ_f,c]                        │
│                                          │
│  Shape: (5,)                            │
└─────────────────────────────────────────┘
```

### Visualisierung: Zustandsraum

```
Zustandsvektor-Komponenten:

1. G_f,c (Globales Gewicht):
   │
20│  ●        ●
  │     ●         ●
10│  ●    ●    ●     ●
  │●                   ●
  └─────────────────────► Frame f
  
2. ⟨Q_local⟩ (Mittlere lokale Qualität):
   │
 2│      ●    ●
  │  ●     ●      ●
 0│     ●    ●       ●
  │  ●                 ●
-2│●                     ●
  └─────────────────────► Frame f
  
3. Var(Q_local) (Lokale Variabilität):
   │
 2│●                     ●
  │  ●                 ●
 1│     ●    ●    ●
  │        ●    ●
 0│           ●
  └─────────────────────► Frame f
  
Interpretation:
  • Hohe Var → Inhomogenes Seeing (unterschiedliche Tile-Qualitäten)
  • Niedrige Var → Homogenes Seeing (gleichmäßige Qualität)
```

### Warum diese Komponenten?

```
G_f,c:
  → Gesamtqualität des Frames
  → Hauptkriterium für Clusterung

⟨Q_local⟩:
  → Durchschnittliche lokale Qualität
  → Ergänzt globale Sicht

Var(Q_local):
  → Seeing-Homogenität
  → Frames mit ähnlicher Variabilität zusammen

B_f,c:
  → Himmelshintergrund
  → Gruppiert Frames mit ähnlichen Bedingungen

σ_f,c:
  → Rauschcharakteristik
  → Frames mit ähnlichem Rauschen zusammen
```

## Schritt 6.2: Feature-Normalisierung

### Z-Score-Normalisierung

```
┌─────────────────────────────────────────┐
│  Zustandsvektoren für alle Frames:      │
│  V = [v_0, v_1, ..., v_{N-1}]          │
│  Shape: (N, 5)                          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Pro Feature-Dimension d:               │
│                                          │
│  μ_d = mean(V[:, d])                    │
│  σ_d = std(V[:, d])                     │
│                                          │
│  V'[:, d] = (V[:, d] - μ_d) / σ_d      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Normalisierte Zustandsvektoren         │
│  V' = [v'_0, v'_1, ..., v'_{N-1}]      │
│                                          │
│  Alle Features haben:                   │
│  • Mittelwert = 0                       │
│  • Standardabweichung = 1               │
└─────────────────────────────────────────┘
```

**Wichtig:** Normalisierung verhindert, dass Features mit großen Werten die Clusterung dominieren.

## Schritt 6.3: K-Means Clusterung

### Algorithmus

```
┌─────────────────────────────────────────┐
│  Input: V' (normalisierte Vektoren)     │
│         K (Anzahl Cluster, 15-30)       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Initialisierung (K-Means++)            │
│                                          │
│  1. Wähle ersten Zentroid zufällig      │
│  2. Für k=2..K:                         │
│     Wähle nächsten Zentroid mit         │
│     Wahrscheinlichkeit ∝ D²             │
│     (D = Distanz zu nächstem Zentroid)  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Iteration (bis Konvergenz):            │
│                                          │
│  E-Step (Assignment):                   │
│    Für jeden Frame f:                   │
│      c_f = argmin_k ||v'_f - μ_k||²    │
│                                          │
│  M-Step (Update):                       │
│    Für jeden Cluster k:                 │
│      μ_k = mean(v'_f für alle f mit c_f=k)│
│                                          │
│  Konvergenz wenn:                       │
│    Keine Änderung in Assignments        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Output: Cluster-Zuordnungen            │
│  cluster_id[f] = k  (k = 0..K-1)        │
└─────────────────────────────────────────┘
```

### Visualisierung: 2D-Projektion

```
Zustandsraum (2D-Projektion via PCA):

PC2 │
    │     ●●●         Cluster 0 (beste Frames)
  2 │    ●●●●●        • Hohes G
    │     ●●●         • Niedriges σ
    │                 • Homogenes Seeing
  1 │  ○○○○○
    │ ○○○○○○○         Cluster 1 (gute Frames)
  0 ├─────────────────
    │      ░░░        Cluster 2 (mittlere Frames)
 -1 │    ░░░░░
    │     ░░░
    │                 Cluster 3 (schlechte Frames)
 -2 │   ▓▓▓           • Niedriges G
    │  ▓▓▓▓▓          • Hohes σ
    └─────────────────────────────► PC1
   -2  -1   0   1   2

Legende:
  ● = Cluster 0 (K=15-30 Cluster insgesamt)
  ○ = Cluster 1
  ░ = Cluster 2
  ▓ = Cluster 3
```

### Cluster-Anzahl K

```
Empfohlene Werte:
  N ≥ 800:  K = 25-30
  N ≥ 400:  K = 20-25
  N ≥ 200:  K = 15-20
  N < 200:  Clusterung überspringen (Reduced Mode)

Begründung:
  • Zu wenig Cluster (K < 15):
    → Zu heterogene Cluster
    → Wenig Rauschreduktion
  
  • Zu viele Cluster (K > 30):
    → Zu wenig Frames pro Cluster
    → Instabile synthetische Frames
```

## Schritt 6.4: Cluster-Validierung

### Qualitätskriterien

```python
def validate_clusters(cluster_assignments, state_vectors, K):
    """
    Validiert Cluster-Qualität.
    """
    # Check 1: Alle Cluster besetzt
    unique_clusters = set(cluster_assignments)
    assert len(unique_clusters) == K, \
        f"Only {len(unique_clusters)}/{K} clusters populated"
    
    # Check 2: Minimale Cluster-Größe
    for k in range(K):
        cluster_size = sum(c == k for c in cluster_assignments)
        assert cluster_size >= 3, \
            f"Cluster {k} has only {cluster_size} frames (min: 3)"
    
    # Check 3: Intra-Cluster-Kohäsion
    for k in range(K):
        cluster_frames = [i for i, c in enumerate(cluster_assignments) if c == k]
        cluster_vecs = state_vectors[cluster_frames]
        
        # Durchschnittliche Intra-Cluster-Distanz
        intra_dist = np.mean([
            np.linalg.norm(v1 - v2)
            for v1 in cluster_vecs
            for v2 in cluster_vecs
        ])
        
        # Sollte klein sein (kohäsive Cluster)
        assert intra_dist < 5.0, \
            f"Cluster {k} has high intra-distance: {intra_dist:.2f}"
    
    # Check 4: Inter-Cluster-Separation
    centroids = []
    for k in range(K):
        cluster_frames = [i for i, c in enumerate(cluster_assignments) if c == k]
        centroid = np.mean(state_vectors[cluster_frames], axis=0)
        centroids.append(centroid)
    
    min_separation = min([
        np.linalg.norm(c1 - c2)
        for i, c1 in enumerate(centroids)
        for j, c2 in enumerate(centroids)
        if i < j
    ])
    
    # Sollte groß sein (gut separierte Cluster)
    assert min_separation > 0.5, \
        f"Clusters poorly separated: {min_separation:.2f}"
```

## Schritt 6.5: Synthetische Frame-Rekonstruktion

### Konzept

```
Cluster k mit Frames {f₁, f₂, ..., f_m}:

Statt alle m Frames einzeln zu stacken:
  → Rekonstruiere 1 synthetisches Frame pro Cluster
  → Repräsentiert den "idealen" Frame dieses Zustands
  → Reduziert Rauschen durch Averaging
```

### Prozess

```
┌─────────────────────────────────────────┐
│  Cluster k, Kanal c                     │
│  Frames: {f₁, f₂, ..., f_m}            │
│  Tiles: rekonstruierte Tiles aus Phase 5│
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Für jedes Tile t:                      │
│                                          │
│  Sammle Tile-Daten aus allen Frames     │
│  im Cluster:                            │
│    tiles_cluster = [I_f,t,c for f in k] │
│                                          │
│  Berechne Gewichte:                     │
│    weights = [W_f,t,c for f in k]       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Gewichtetes Stacking pro Tile:         │
│                                          │
│  I_synth,t,c = Σ_f [W_f,t,c · I_f,t,c]  │
│                / Σ_f W_f,t,c            │
│                                          │
│  (nur über Frames in Cluster k)         │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Overlap-Add (wie Phase 5):             │
│                                          │
│  Kombiniere alle Tiles zu synthetischem │
│  Frame mit Fensterfunktion              │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Output: Synthetisches Frame            │
│  F_synth,k,c[x,y]                       │
│                                          │
│  Repräsentiert Cluster k                │
└─────────────────────────────────────────┘
```

### Visualisierung

```
Cluster 0 (5 Frames):

Frame 0:  Frame 1

:  Frame 2:  Frame 3:  Frame 4:
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│  ★   │  │  ★   │  │  ★   │  │  ★   │  │  ★   │
│    ★ │  │    ★ │  │    ★ │  │    ★ │  │    ★ │
│      │  │      │  │      │  │      │  │      │
│  ★   │  │  ★   │  │  ★   │  │  ★   │  │  ★   │
└──────┘  └──────┘  └──────┘  └──────┘  └──────┘
  W=5.2     W=4.8     W=5.5     W=4.9     W=5.1

           │
           ▼ Gewichtetes Stacking
           │
    ┌──────────────┐
    │  Synthetisch │
    │      ★       │  ← Rauschen reduziert
    │        ★     │  ← Sterne schärfer
    │              │
    │      ★       │
    └──────────────┘
    
    Rauschreduktion: σ_synth ≈ σ_original / √m
    (m = Anzahl Frames im Cluster)
```

### Gewicht des synthetischen Frames

```
Für finales Stacking (Phase 7):

W_synth,k,c = Σ_f∈k W_f,c

wobei:
  W_f,c = G_f,c (globales Gewicht aus Phase 2)
  
Interpretation:
  • Synthetisches Frame repräsentiert alle Frames im Cluster
  • Gewicht = Summe der Original-Gewichte
  • Cluster mit mehr/besseren Frames → höheres Gewicht
```

## Schritt 6.6: Cluster-Statistiken

### Pro Cluster

```python
cluster_stats = {
    'cluster_id': int,
    'frame_count': int,
    'frame_ids': List[int],
    
    # Zustandsvektor-Statistiken
    'mean_G': float,
    'mean_B': float,
    'mean_sigma': float,
    'mean_Q_local': float,
    'var_Q_local': float,
    
    # Synthetisches Frame
    'synthetic_weight': float,  # W_synth
    'synthetic_mean': float,
    'synthetic_std': float,
    
    # Qualität
    'intra_cluster_distance': float,
    'centroid': np.ndarray,  # (5,)
}
```

### Visualisierung: Cluster-Übersicht

```
Cluster-Statistiken (K=20):

Cluster │ Frames │ Mean G │ Mean σ │ W_synth │ Quality
────────┼────────┼────────┼────────┼─────────┼─────────
   0    │   45   │  8.2   │ 0.012  │  369.0  │ ★★★★★
   1    │   52   │  7.8   │ 0.013  │  405.6  │ ★★★★★
   2    │   38   │  6.5   │ 0.015  │  247.0  │ ★★★★
   3    │   41   │  6.1   │ 0.016  │  250.1  │ ★★★★
   4    │   35   │  5.2   │ 0.018  │  182.0  │ ★★★
  ...   │  ...   │  ...   │  ...   │   ...   │  ...
  18    │   28   │  1.8   │ 0.035  │   50.4  │ ★
  19    │   25   │  1.2   │ 0.042  │   30.0  │ ★

Gesamt: 800 Frames → 20 synthetische Frames
Reduktion: 40:1
```

## Schritt 6.7: Validierung

```python
def validate_synthetic_frames(synthetic_frames, clusters, original_frames):
    # Check 1: Anzahl synthetischer Frames
    K = len(clusters)
    assert len(synthetic_frames) == K, \
        f"Expected {K} synthetic frames, got {len(synthetic_frames)}"
    
    # Check 2: Keine NaN/Inf
    for k, frame in enumerate(synthetic_frames):
        assert not np.any(np.isnan(frame)), f"NaN in synthetic frame {k}"
        assert not np.any(np.isinf(frame)), f"Inf in synthetic frame {k}"
    
    # Check 3: Rauschreduktion
    for k, cluster in enumerate(clusters):
        original_noise = np.mean([
            estimate_noise(original_frames[f])
            for f in cluster['frame_ids']
        ])
        synthetic_noise = estimate_noise(synthetic_frames[k])
        
        expected_reduction = np.sqrt(cluster['frame_count'])
        actual_reduction = original_noise / synthetic_noise
        
        # Sollte ungefähr √m sein
        assert 0.5 * expected_reduction < actual_reduction < 2.0 * expected_reduction, \
            f"Unexpected noise reduction in cluster {k}"
    
    # Check 4: Gewichtserhaltung
    total_original_weight = sum(G_f for all frames)
    total_synthetic_weight = sum(W_synth for all clusters)
    
    assert np.isclose(total_original_weight, total_synthetic_weight, rtol=0.01), \
        "Weight not conserved in synthetic frames"
    
    # Check 5: Kanalunabhängigkeit
    assert no_channel_mixing_in_clustering()
```

## Output-Datenstruktur

```python
# Phase 6 Output
{
    'clustering': {
        'method': 'kmeans',
        'n_clusters': int,  # K
        'state_vector_dim': 5,
        'cluster_assignments': np.ndarray,  # (N_frames,)
    },
    'clusters': [
        {
            'cluster_id': int,
            'frame_count': int,
            'frame_ids': List[int],
            'centroid': np.ndarray,  # (5,)
            'intra_distance': float,
            'mean_G': float,
            'mean_sigma': float,
            'synthetic_weight': float,
        },
        ...  # K Cluster
    ],
    'synthetic_frames': {
        'R': np.ndarray,  # shape: (K, H, W)
        'G': np.ndarray,
        'B': np.ndarray,
    },
    'synthetic_weights': {
        'R': np.ndarray,  # shape: (K,)
        'G': np.ndarray,
        'B': np.ndarray,
    },
    'statistics': {
        'original_frame_count': int,  # N
        'synthetic_frame_count': int,  # K
        'reduction_ratio': float,  # N/K
        'total_weight_original': float,
        'total_weight_synthetic': float,
        'mean_noise_reduction': float,
    }
}
```

## Performance-Hinweise

```python
# Effiziente K-Means-Implementierung
from sklearn.cluster import MiniBatchKMeans

def cluster_frames_efficient(state_vectors, K):
    """
    Effiziente Clusterung mit Mini-Batch K-Means.
    """
    # Mini-Batch K-Means (schneller für große N)
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        init='k-means++',
        max_iter=100,
        batch_size=100,
        random_state=42,  # Reproduzierbarkeit
        n_init=10,
    )
    
    # Fit
    cluster_assignments = kmeans.fit_predict(state_vectors)
    
    return cluster_assignments, kmeans.cluster_centers_

# Parallele synthetische Frame-Rekonstruktion
from concurrent.futures import ProcessPoolExecutor

def reconstruct_synthetic_frames_parallel(clusters, tiles, frames, weights):
    """
    Rekonstruiert synthetische Frames parallel.
    """
    def reconstruct_cluster(cluster_id):
        cluster = clusters[cluster_id]
        frame_ids = cluster['frame_ids']
        
        # Rekonstruiere synthetisches Frame
        synth_frame = reconstruct_from_tiles(
            tiles, frames[frame_ids], weights[frame_ids]
        )
        
        return cluster_id, synth_frame
    
    # Parallel processing
    with ProcessPoolExecutor() as executor:
        results = executor.map(reconstruct_cluster, range(len(clusters)))
    
    # Sortiere Ergebnisse
    synthetic_frames = [None] * len(clusters)
    for cluster_id, synth_frame in results:
        synthetic_frames[cluster_id] = synth_frame
    
    return synthetic_frames
```

## Nächste Phase

→ **Phase 7: Finales lineares Stacking**
