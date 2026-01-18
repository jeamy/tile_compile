import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class StateClustering:
    """
    Implements state-based frame clustering for Methodik v3
    """
    @classmethod
    def cluster_frames(
        cls, 
        frames: List[np.ndarray], 
        metrics: Dict[str, Any], 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform state-based clustering of frames.

        Default behaviour follows Methodik v3 §3.7:
        K = clip(floor(N/10), K_min, K_max), with K_min/K_max taken from
        "cluster_count_range" or falling back to sensible defaults.
        
        Args:
            frames: List of input frames
            metrics: Computed frame metrics
            config: Clustering configuration
        
        Returns:
            Clustering results and frame assignments
        """
        config = config or {}
        
        # Extract state vector features
        state_vectors = cls._compute_state_vectors(frames, metrics)
        n_frames = int(state_vectors.shape[0])
        
        # Preprocessing
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(state_vectors)
        
        # Determine cluster count range (K_min, K_max)
        range_cfg = config.get('cluster_count_range')
        if isinstance(range_cfg, (list, tuple)) and len(range_cfg) == 2:
            k_min, k_max = int(range_cfg[0]), int(range_cfg[1])
        else:
            # Backwards-compatible defaults if no explicit range is provided
            k_min = int(config.get('min_clusters', 5))
            k_max = int(config.get('max_clusters', 30))
        
        k_min = max(1, k_min)
        k_max = max(k_min, k_max)
        
        if n_frames <= 1:
            final_n_clusters = 1
            silhouette_score = -1.0
        else:
            # Methodik v3: K = clip(floor(N/10), K_min, K_max)
            k_default = int(np.clip(n_frames // 10, k_min, k_max))
            
            if config.get('use_silhouette', False):
                # Optional: refine K using silhouette score within [k_min, k_max]
                best_clustering = cls._find_optimal_clustering(
                    scaled_vectors, 
                    k_min, 
                    k_max,
                )
                final_n_clusters = int(best_clustering.get('n_clusters', k_default) or k_default)
                silhouette_score = float(best_clustering.get('silhouette_score', -1.0))
            else:
                final_n_clusters = k_default
                silhouette_score = -1.0
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=final_n_clusters, 
            n_init=10, 
            random_state=42
        )
        cluster_labels = kmeans.fit_predict(scaled_vectors)
        
        # Compute cluster statistics
        cluster_stats = cls._compute_cluster_statistics(
            state_vectors, 
            cluster_labels
        )
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_stats': cluster_stats,
            'n_clusters': int(final_n_clusters),
            'silhouette_score': float(silhouette_score)
        }
    
    @staticmethod
    def _compute_state_vectors(
        frames: List[np.ndarray], 
        metrics: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute state vector for each frame
        
        State vector components:
        - Global quality index
        - Local tile quality average
        - Local tile quality variance
        - Background level
        - Noise level
        """
        # Extract global metrics
        global_metrics = metrics.get('global', {})
        tile_metrics = metrics.get('tiles', {})
        
        state_vectors = []
        for i in range(len(frames)):
            state_vector = [
                # Global quality metrics
                global_metrics.get('G_f_c', [0])[i] if i < len(global_metrics.get('G_f_c', [])) else 0,
                
                # Local tile metrics
                np.mean(tile_metrics.get('Q_local', [[0]])[i]) if i < len(tile_metrics.get('Q_local', [[0]])) else 0,
                np.var(tile_metrics.get('Q_local', [[0]])[i]) if i < len(tile_metrics.get('Q_local', [[0]])) else 0,
                
                # Frame-level metrics
                global_metrics.get('background_level', [0])[i] if i < len(global_metrics.get('background_level', [])) else 0,
                global_metrics.get('noise_level', [0])[i] if i < len(global_metrics.get('noise_level', [])) else 0
            ]
            state_vectors.append(state_vector)
        
        return np.array(state_vectors)
    
    @classmethod
    def _find_optimal_clustering(
        cls, 
        data: np.ndarray, 
        min_clusters: int, 
        max_clusters: int
    ) -> Dict[str, Any]:
        """
        Find optimal number of clusters using silhouette score
        """
        from sklearn.metrics import silhouette_score

        n_samples = int(data.shape[0])
        if n_samples < 2:
            return {
                'n_clusters': 1,
                'silhouette_score': -1
            }

        min_clusters = max(2, int(min_clusters))
        max_clusters = int(max_clusters)
        max_clusters = min(max_clusters, n_samples - 1)
        min_clusters = min(min_clusters, max_clusters)
        
        best_score = -1
        best_n_clusters = min_clusters
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(data)
            
            try:
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except Exception:
                # Fallback if silhouette score computation fails
                continue
        
        return {
            'n_clusters': best_n_clusters,
            'silhouette_score': best_score
        }
    
    @staticmethod
    def _compute_cluster_statistics(
        state_vectors: np.ndarray, 
        cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute detailed statistics for each cluster
        """
        cluster_stats = {}
        unique_clusters = np.unique(cluster_labels)
        
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_data = state_vectors[cluster_mask]
            
            cluster_stats[int(cluster)] = {
                'size': len(cluster_data),
                'mean_vector': np.mean(cluster_data, axis=0).tolist(),
                'std_vector': np.std(cluster_data, axis=0).tolist(),
                'min_vector': np.min(cluster_data, axis=0).tolist(),
                'max_vector': np.max(cluster_data, axis=0).tolist()
            }
        
        return cluster_stats

    @classmethod
    def cluster_frames_quantile_fallback(
        cls,
        frames: List[np.ndarray],
        metrics: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Quantile-based clustering fallback (Methodik v3 §10)
        
        Groups frames by quantiles of their global quality index G_f.
        This is a robust fallback when k-means fails or is unsuitable.
        
        Args:
            frames: List of input frames
            metrics: Computed frame metrics
            config: Clustering configuration
        
        Returns:
            Clustering results with quantile-based assignments
        """
        config = config or {}
        n_quantiles = config.get('fallback_quantiles', 15)
        
        # Extract global quality indices
        global_metrics = metrics.get('global', {})
        G_f = np.array(global_metrics.get('G_f_c', []))
        
        if len(G_f) == 0:
            raise ValueError("No global quality indices available for quantile clustering")
        
        # Compute quantile boundaries
        quantiles = np.linspace(0, 100, n_quantiles + 1)
        boundaries = np.percentile(G_f, quantiles)
        
        # Assign frames to quantile bins
        cluster_labels = np.digitize(G_f, boundaries[1:-1])
        
        # Compute cluster statistics
        cluster_stats = cls._compute_cluster_statistics_simple(
            G_f, cluster_labels
        )
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'n_clusters': n_quantiles,
            'method': 'quantile_fallback',
            'quantile_boundaries': boundaries.tolist(),
            'cluster_stats': cluster_stats,
            'silhouette_score': -1  # Not applicable for quantile clustering
        }
    
    @staticmethod
    def _compute_cluster_statistics_simple(
        values: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute simple statistics for quantile-based clusters
        """
        cluster_stats = {}
        unique_clusters = np.unique(cluster_labels)
        
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_values = values[cluster_mask]
            
            cluster_stats[int(cluster)] = {
                'size': len(cluster_values),
                'mean': float(np.mean(cluster_values)),
                'std': float(np.std(cluster_values)),
                'min': float(np.min(cluster_values)),
                'max': float(np.max(cluster_values))
            }
        
        return cluster_stats

def cluster_channels(
    channels: Dict[str, List[np.ndarray]], 
    metrics: Dict[str, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Perform state-based clustering for each channel
    
    Args:
        channels: Dictionary of channel frames
        metrics: Metrics for each channel
        config: Clustering configuration
    
    Returns:
        Clustering results per channel
    """
    channel_clustering = {}
    
    for channel_name, frames in channels.items():
        channel_metrics = metrics.get(channel_name, {})
        
        try:
            # Try k-means clustering first
            channel_clustering[channel_name] = StateClustering.cluster_frames(
                frames, 
                channel_metrics, 
                config
            )
        except Exception:
            # Fallback to quantile-based clustering
            try:
                channel_clustering[channel_name] = StateClustering.cluster_frames_quantile_fallback(
                    frames,
                    channel_metrics,
                    config
                )
            except Exception:
                # If both fail, skip this channel
                continue
    
    return channel_clustering