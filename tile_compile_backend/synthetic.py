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
        config: Optional[Dict[str, Any]] = None
    ) -> List[np.ndarray]:
        """
        Generate synthetic frames using various strategies
        
        Args:
            input_frames: Original input frames
            metrics: Computed metrics for frames
            config: Configuration for synthetic frame generation
        
        Returns:
            List of synthetic frames
        """
        config = config or {}
        
        # Default generation strategies
        strategies = {
            'linear_average': cls._linear_average,
            'weighted_average': cls._weighted_average,
            'noise_injection': cls._noise_injection
        }
        
        # Select generation method
        method = str(config.get('method', 'linear_average')).lower()
        generator = strategies.get(method, strategies['linear_average'])
        
        # Frame selection parameters
        frames_min = int(config.get('frames_min', 15))
        frames_max = int(config.get('frames_max', 30))
        
        # Validate input
        if len(input_frames) < frames_min:
            raise ValueError(f"Insufficient frames. Minimum required: {frames_min}")
        
        # Select subset of frames
        selected_frames = input_frames[:min(len(input_frames), frames_max)]
        
        # Generate synthetic frames
        synthetic_frames = generator(selected_frames, metrics)
        
        return synthetic_frames
    
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

def generate_channel_synthetic_frames(
    channels: Dict[str, List[np.ndarray]], 
    metrics: Dict[str, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, List[np.ndarray]]:
    """
    Generate synthetic frames for each channel
    
    Args:
        channels: Dictionary of channel frames
        metrics: Metrics for each channel
        config: Configuration for synthetic frame generation
    
    Returns:
        Dictionary of synthetic frames per channel
    """
    synthetic_channels = {}
    
    for channel_name, frames in channels.items():
        channel_metrics = metrics.get(channel_name, {})
        
        # Generate synthetic frames for this channel
        synthetic_channels[channel_name] = SyntheticFrameGenerator.generate_synthetic_frames(
            frames, 
            channel_metrics, 
            config
        )
    
    return synthetic_channels