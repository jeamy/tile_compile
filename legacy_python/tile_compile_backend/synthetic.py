import numpy as np
from typing import List, Dict, Any, Optional

class SyntheticFrameGenerator:
    """
    Generate synthetic frames based on Methodik v3 principles
    """
    @classmethod
    def generate_synthetic_frames(
        cls, 
        input_frames: List[np.ndarray], 
        metrics: Dict[str, Any], 
        config: Optional[Dict[str, Any]] = None,
        clustering_results: Optional[Dict[str, Any]] = None
    ) -> List[np.ndarray]:
        """
        Generate synthetic frames according to Methodik v3 §3.8
        
        Per Methodik v3:
        - One synthetic frame per cluster (15-30 clusters)
        - Each synthetic frame = linear weighted stack of frames in that cluster
        - No additional weighting beyond global weights G_f,c
        
        Args:
            input_frames: Original input frames
            metrics: Computed metrics for frames (including global weights)
            config: Configuration for synthetic frame generation
            clustering_results: Clustering results (cluster assignments per frame)
        
        Returns:
            List of synthetic frames (15-30 frames)
        """
        config = config or {}
        
        # If clustering results are available, use cluster-based generation
        if clustering_results and 'labels' in clustering_results:
            return cls._generate_from_clusters(input_frames, metrics, clustering_results)
        
        # Fallback: quantile-based pseudo-clustering
        frames_min = int(config.get('frames_min', 15))
        frames_max = int(config.get('frames_max', 30))
        
        # Validate input
        if len(input_frames) < frames_min:
            # If we have fewer frames, return what we can
            return [cls._weighted_average([input_frames[0]], metrics)] if input_frames else []
        
        # Determine number of synthetic frames to generate
        n_synthetic = min(frames_max, max(frames_min, len(input_frames) // 10))
        n_synthetic = max(n_synthetic, frames_min)
        
        # Use quantile-based grouping as fallback
        return cls._generate_quantile_based(input_frames, metrics, n_synthetic)
    
    @staticmethod
    def _linear_average(
        frames: List[np.ndarray], 
        metrics: Optional[Dict[str, Any]] = None
    ) -> List[np.ndarray]:
        """
        Simple linear averaging of frames
        """
        # Convert frames to numpy array
        frame_array = np.array(frames)
        
        # Compute linear average
        synthetic_frame = np.mean(frame_array, axis=0)
        
        return [synthetic_frame]
    
    @staticmethod
    def _weighted_average(
        frames: List[np.ndarray], 
        metrics: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Compute weighted average based on frame metrics
        """
        # Extract global frame weights
        global_metrics = metrics.get('global', {})
        
        # Compute weights based on noise level and background
        noise_levels = global_metrics.get('noise_level', np.ones(len(frames)))
        background_levels = global_metrics.get('background_level', np.zeros(len(frames)))

        noise_levels = np.asarray(noise_levels)
        background_levels = np.asarray(background_levels)
        if noise_levels.ndim != 1:
            noise_levels = np.ravel(noise_levels)
        if background_levels.ndim != 1:
            background_levels = np.ravel(background_levels)

        n = len(frames)
        if noise_levels.shape[0] < n:
            noise_levels = np.pad(noise_levels, (0, n - noise_levels.shape[0]), mode='edge')
        if background_levels.shape[0] < n:
            background_levels = np.pad(background_levels, (0, n - background_levels.shape[0]), mode='edge')
        noise_levels = noise_levels[:n]
        background_levels = background_levels[:n]
        
        # Inverse of noise as weight (lower noise = higher weight)
        weights = 1 / (noise_levels + 1e-8)
        weights /= np.sum(weights)
        
        # Apply weighted averaging
        frame_array = np.array(frames)
        synthetic_frame = np.average(frame_array, weights=weights, axis=0)
        
        return [synthetic_frame]
    
    @staticmethod
    def _noise_injection(
        frames: List[np.ndarray], 
        metrics: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Generate synthetic frames by controlled noise injection
        """
        frame_array = np.array(frames)
        
        # Compute average frame
        avg_frame = np.mean(frame_array, axis=0)
        
        # Get noise characteristics
        noise_level = metrics.get('global', {}).get('noise_level', 0.1)
        noise_level = np.asarray(noise_level)
        if noise_level.size > 1:
            noise_level = float(np.mean(noise_level))
        else:
            noise_level = float(noise_level)
        
        # Generate multiple synthetic frames with noise
        synthetic_frames = []
        for _ in range(3):  # Generate 3 synthetic variants
            noise = np.random.normal(
                0, 
                noise_level, 
                avg_frame.shape
            )
            synthetic_frame = avg_frame + noise
            synthetic_frames.append(synthetic_frame)
        
        return synthetic_frames
    
    @classmethod
    def _generate_from_clusters(
        cls,
        frames: List[np.ndarray],
        metrics: Dict[str, Any],
        clustering_results: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Generate one synthetic frame per cluster (Methodik v3 §3.8)
        
        Each synthetic frame is a weighted linear stack of frames in that cluster,
        using global weights G_f,c from metrics.
        """
        # Support both 'labels' and 'cluster_labels' keys
        labels = clustering_results.get('labels', clustering_results.get('cluster_labels', []))
        if not labels or len(labels) != len(frames):
            return []
        
        # Get global weights G_f,c (support both naming conventions)
        global_metrics = metrics.get('global', {})
        global_weights = global_metrics.get('G_f_c', global_metrics.get('global_weight', np.ones(len(frames))))
        global_weights = np.asarray(global_weights)
        if global_weights.ndim != 1:
            global_weights = np.ravel(global_weights)
        if global_weights.shape[0] < len(frames):
            global_weights = np.pad(global_weights, (0, len(frames) - global_weights.shape[0]), mode='edge')
        global_weights = global_weights[:len(frames)]
        
        # Group frames by cluster
        unique_clusters = sorted(set(labels))
        synthetic_frames = []
        
        for cluster_id in unique_clusters:
            # Get frames in this cluster
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            if not cluster_indices:
                continue
            
            cluster_frames = [frames[i] for i in cluster_indices]
            cluster_weights = global_weights[cluster_indices]
            
            # Normalize weights
            weight_sum = np.sum(cluster_weights)
            if weight_sum > 1e-10:
                cluster_weights = cluster_weights / weight_sum
            else:
                cluster_weights = np.ones(len(cluster_weights)) / len(cluster_weights)
            
            # Weighted linear stack (Methodik v3: no additional weighting)
            frame_array = np.array(cluster_frames, dtype=np.float32)
            synthetic_frame = np.average(frame_array, weights=cluster_weights, axis=0)
            synthetic_frames.append(synthetic_frame)
        
        return synthetic_frames
    
    @classmethod
    def _generate_quantile_based(
        cls,
        frames: List[np.ndarray],
        metrics: Dict[str, Any],
        n_synthetic: int
    ) -> List[np.ndarray]:
        """
        Fallback: Generate synthetic frames using quantile-based grouping
        
        When clustering is not available, group frames by global weight quantiles
        and generate one synthetic frame per group.
        """
        # Get global weights
        global_metrics = metrics.get('global', {})
        global_weights = global_metrics.get('global_weight', np.ones(len(frames)))
        global_weights = np.asarray(global_weights)
        if global_weights.ndim != 1:
            global_weights = np.ravel(global_weights)
        if global_weights.shape[0] < len(frames):
            global_weights = np.pad(global_weights, (0, len(frames) - global_weights.shape[0]), mode='edge')
        global_weights = global_weights[:len(frames)]
        
        # Sort frames by weight
        sorted_indices = np.argsort(global_weights)
        
        # Divide into n_synthetic groups
        synthetic_frames = []
        group_size = len(frames) / n_synthetic
        
        for i in range(n_synthetic):
            start_idx = int(i * group_size)
            end_idx = int((i + 1) * group_size) if i < n_synthetic - 1 else len(frames)
            
            group_indices = sorted_indices[start_idx:end_idx]
            if len(group_indices) == 0:
                continue
            
            group_frames = [frames[idx] for idx in group_indices]
            group_weights = global_weights[group_indices]
            
            # Normalize weights
            weight_sum = np.sum(group_weights)
            if weight_sum > 1e-10:
                group_weights = group_weights / weight_sum
            else:
                group_weights = np.ones(len(group_weights)) / len(group_weights)
            
            # Weighted linear stack
            frame_array = np.array(group_frames, dtype=np.float32)
            synthetic_frame = np.average(frame_array, weights=group_weights, axis=0)
            synthetic_frames.append(synthetic_frame)
        
        return synthetic_frames

def generate_channel_synthetic_frames(
    channels: Dict[str, List[np.ndarray]], 
    metrics: Dict[str, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    clustering_results: Optional[Dict[str, Any]] = None
) -> Dict[str, List[np.ndarray]]:
    """
    Generate synthetic frames for each channel according to Methodik v3 §3.8
    
    Per Methodik v3:
    - §3.7: 15-30 clusters from state vectors
    - §3.8: One synthetic frame per cluster -> 15-30 synthetic frames total
    - Linear stacking without additional weighting
    
    Args:
        channels: Dictionary of channel frames
        metrics: Metrics for each channel (including global weights)
        config: Configuration for synthetic frame generation
        clustering_results: Clustering results from phase 8 (per-channel cluster assignments)
    
    Returns:
        Dictionary of synthetic frames per channel (15-30 frames per channel)
    """
    synthetic_channels = {}
    
    for channel_name, frames in channels.items():
        channel_metrics = metrics.get(channel_name, {})
        
        # Get channel-specific clustering results if available
        channel_clustering = None
        if clustering_results:
            # Clustering results can be per-channel (Dict[str, Dict]) or global (Dict with 'labels')
            if channel_name in clustering_results:
                channel_clustering = clustering_results[channel_name]
            elif 'labels' in clustering_results or 'cluster_labels' in clustering_results:
                # Global clustering results apply to all channels
                channel_clustering = clustering_results
        
        # Generate synthetic frames for this channel
        synthetic_channels[channel_name] = SyntheticFrameGenerator.generate_synthetic_frames(
            frames, 
            channel_metrics, 
            config,
            channel_clustering
        )
    
    return synthetic_channels