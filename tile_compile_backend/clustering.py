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
        Perform state-based clustering of frames
        
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
        
        # Preprocessing
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(state_vectors)
        
        # Clustering parameters
        n_clusters = config.get('n_clusters', 20)
        max_clusters = config.get('max_clusters', 30)
        min_clusters = config.get('min_clusters', 15)
        
        # Adaptive cluster determination (optional)
        best_clustering = cls._find_optimal_clustering(
            scaled_vectors, 
            min_clusters, 
            max_clusters
        )
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=best_clustering['n_clusters'], 
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
            'n_clusters': best_clustering['n_clusters'],
            'silhouette_score': best_clustering['silhouette_score']
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
        
        # Perform clustering for this channel
        channel_clustering[channel_name] = StateClustering.cluster_frames(
            frames, 
            channel_metrics, 
            config
        )
    
    return channel_clustering