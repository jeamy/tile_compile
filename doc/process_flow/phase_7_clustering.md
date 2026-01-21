# Phase 7: Zustandsbasierte Frame-Clusterung

## Übersicht

Phase 7 ist **identisch zu v3**. Nach der Tile-Rekonstruktion werden Frames basierend auf ihrem Qualitätszustand geclustert, um synthetische Frames zu erzeugen.

Diese Phase wird nur im **Normal Mode** (N ≥ 200 Frames) ausgeführt.

## Ziele

1. Frames nach Qualitätszustand gruppieren
2. Frame-Anzahl reduzieren (N → K)
3. Rauschreduktion vorbereiten
4. Gewichte erhalten

## Zustandsvektor

Jeder Frame wird durch einen **Zustandsvektor** charakterisiert:

```
v_f,c = (G_f,c, ⟨Q_local⟩, Var(Q_local), B_f,c, σ_f,c)

wobei:
  G_f,c         - Globales Gewicht (aus Phase 3)
  ⟨Q_local⟩     - Mittlere lokale Qualität über alle Tiles
  Var(Q_local)  - Varianz der lokalen Qualität
  B_f,c         - Hintergrundniveau (aus Phase 3)
  σ_f,c         - Rauschen (aus Phase 3)
```

### Berechnung

```python
def compute_state_vector(
    frame_idx: int,
    channel: str,
    global_metrics: Dict,
    tile_metadata: List[Dict]
) -> np.ndarray:
    """
    Berechnet Zustandsvektor für einen Frame.
    
    Args:
        frame_idx: Frame-Index
        channel: Kanal ('R', 'G', 'B')
        global_metrics: Globale Metriken aus Phase 3
        tile_metadata: Tile-Metadaten aus Phase 6
    
    Returns:
        Zustandsvektor [5 Komponenten]
    """
    c_idx = {'R': 0, 'G': 1, 'B': 2}[channel]
    
    # Global weight
    G = global_metrics['global_weights'][frame_idx, c_idx]
    
    # Local quality statistics from TLR
    # Extract cross-correlations for this frame from all tiles
    local_qualities = []
    for tile_meta in tile_metadata:
        if frame_idx < len(tile_meta['cross_correlations']):
            cc = tile_meta['cross_correlations'][frame_idx]
            local_qualities.append(cc)
    
    mean_local_quality = np.mean(local_qualities)
    var_local_quality = np.var(local_qualities)
    
    # Background and noise
    B = global_metrics['backgrounds'][frame_idx, c_idx]
    sigma = global_metrics['noises'][frame_idx, c_idx]
    
    # State vector
    state_vector = np.array([
        G,
        mean_local_quality,
        var_local_quality,
        B,
        sigma
    ])
    
    return state_vector
```

## K-Means Clusterung

### Dynamisches K

```python
def compute_optimal_k(N: int, min_k: int = 5, max_k: int = 30) -> int:
    """
    Berechnet optimale Cluster-Anzahl.
    
    K = clip(floor(N / 10), min_k, max_k)
    
    Args:
        N: Anzahl Frames
        min_k: Minimale Cluster-Anzahl
        max_k: Maximale Cluster-Anzahl
    
    Returns:
        Optimales K
    """
    K = int(np.floor(N / 10))
    K = np.clip(K, min_k, max_k)
    
    return K
```

**Beispiele:**
- N = 200 → K = 20
- N = 500 → K = 30 (capped)
- N = 100 → K = 10 (Reduced Mode, kein Clustering)

### Clustering-Algorithmus

```python
def cluster_frames(
    state_vectors: np.ndarray,
    K: int,
    max_iterations: int = 100,
    random_state: int = 42
) -> np.ndarray:
    """
    Clustert Frames mittels K-Means.
    
    Args:
        state_vectors: [N × 5] Zustandsvektoren
        K: Anzahl Cluster
        max_iterations: Max K-Means-Iterationen
        random_state: Random Seed (für Reproduzierbarkeit)
    
    Returns:
        Cluster-Zuordnungen [N] (Werte 0..K-1)
    """
    from sklearn.cluster import KMeans
    
    # Normalize state vectors (wichtig für K-Means)
    state_vectors_norm = (state_vectors - np.mean(state_vectors, axis=0)) / np.std(state_vectors, axis=0)
    
    # K-Means clustering
    kmeans = KMeans(
        n_clusters=K,
        max_iter=max_iterations,
        random_state=random_state,
        n_init=10
    )
    
    cluster_labels = kmeans.fit_predict(state_vectors_norm)
    
    return cluster_labels
```

## Cluster-Analyse

### Cluster-Statistiken

```python
def analyze_clusters(
    cluster_labels: np.ndarray,
    state_vectors: np.ndarray,
    global_weights: np.ndarray
) -> Dict:
    """
    Analysiert Cluster-Eigenschaften.
    
    Returns:
        Dict mit Cluster-Statistiken
    """
    K = np.max(cluster_labels) + 1
    
    cluster_stats = []
    for k in range(K):
        mask = cluster_labels == k
        
        # Frames in diesem Cluster
        frame_indices = np.where(mask)[0]
        
        # Mittlerer Zustandsvektor
        mean_state = np.mean(state_vectors[mask], axis=0)
        
        # Mittleres Gewicht
        mean_weight = np.mean(global_weights[mask])
        
        # Cluster-Größe
        size = np.sum(mask)
        
        cluster_stats.append({
            'cluster_id': k,
            'size': size,
            'frame_indices': frame_indices.tolist(),
            'mean_state': mean_state.tolist(),
            'mean_weight': mean_weight,
        })
    
    return {
        'num_clusters': K,
        'clusters': cluster_stats,
    }
```

### Visualisierung

```python
def visualize_clusters(
    state_vectors: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: str = 'clusters.png'
):
    """
    Visualisiert Cluster in 2D (PCA-Projektion).
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    # PCA to 2D
    pca = PCA(n_components=2)
    state_2d = pca.fit_transform(state_vectors)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        state_2d[:, 0],
        state_2d[:, 1],
        c=cluster_labels,
        cmap='tab20',
        alpha=0.6
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Frame Clustering (PCA Projection)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
```

## Modi

### Normal Mode (N ≥ 200)

```python
if N >= 200:
    # Compute state vectors
    state_vectors = [
        compute_state_vector(f, channel, global_metrics, tile_metadata)
        for f in range(N)
    ]
    
    # Determine K
    K = compute_optimal_k(N)
    
    # Cluster
    cluster_labels = cluster_frames(state_vectors, K)
    
    # Analyze
    cluster_stats = analyze_clusters(cluster_labels, state_vectors, global_weights)
    
    print(f"Clustered {N} frames into {K} clusters")
```

### Reduced Mode (50 ≤ N < 200)

```python
if 50 <= N < 200:
    # Skip clustering
    print(f"Reduced Mode: Skipping clustering ({N} frames)")
    cluster_labels = None
    cluster_stats = None
```

## Validierung

```python
def validate_clustering(cluster_labels: np.ndarray, N: int, K: int):
    """
    Validiert Clustering-Ergebnisse.
    """
    checks = []
    
    # Check 1: All frames assigned
    assert len(cluster_labels) == N
    checks.append(f"✓ All {N} frames assigned to clusters")
    
    # Check 2: K clusters used
    unique_clusters = np.unique(cluster_labels)
    assert len(unique_clusters) == K
    checks.append(f"✓ {K} clusters created")
    
    # Check 3: No empty clusters
    for k in range(K):
        count = np.sum(cluster_labels == k)
        assert count > 0, f"Cluster {k} is empty"
    checks.append("✓ No empty clusters")
    
    # Check 4: Reasonable cluster sizes
    cluster_sizes = [np.sum(cluster_labels == k) for k in range(K)]
    min_size = np.min(cluster_sizes)
    max_size = np.max(cluster_sizes)
    ratio = max_size / min_size if min_size > 0 else float('inf')
    
    if ratio > 10:
        checks.append(f"⚠ Unbalanced clusters: ratio {ratio:.1f}")
    else:
        checks.append(f"✓ Balanced clusters: ratio {ratio:.1f}")
    
    return checks
```

## Output-Datenstruktur

```python
# Phase 7 Output
{
    'cluster_labels': np.ndarray,  # [N] cluster assignments (0..K-1)
    'num_clusters': int,           # K
    'cluster_stats': Dict,         # Statistics per cluster
    'state_vectors': np.ndarray,   # [N × 5] for reference
}
```

## Konfiguration

```yaml
clustering:
  enabled: true                    # Nur wenn N >= 200
  min_clusters: 5
  max_clusters: 30
  cluster_range_factor: 10         # K = N / factor
  
  # K-Means Parameter
  max_iterations: 100
  random_state: 42                 # Für Reproduzierbarkeit
```

## Nächste Phase

→ **Phase 8: Synthetische Frames** (erzeugt K synthetische Frames aus Clustern)
