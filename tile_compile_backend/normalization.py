"""
Normalization module for Methodik v3

Implements global linear normalization (§3.1):
    I'_f = I_f / B_f

This is mandatory before any metric computation.
"""
import numpy as np
from typing import List, Dict, Any, Optional


class LinearNormalizer:
    """
    Implements global linear normalization per Methodik v3 §3.1
    """
    
    @staticmethod
    def normalize_frame(
        frame: np.ndarray,
        background: Optional[float] = None,
        mode: str = 'divide'
    ) -> np.ndarray:
        """
        Normalize a single frame by its background level.
        
        Args:
            frame: Input frame (2D array)
            background: Background level. If None, computed as median.
            mode: 'divide' (I/B) or 'subtract' (I-B)
        
        Returns:
            Normalized frame
        """
        if background is None:
            background = float(np.median(frame))
        
        if background <= 0:
            background = 1e-10  # Prevent division by zero
        
        if mode == 'divide':
            return frame / background
        elif mode == 'subtract':
            return frame - background
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
    
    @classmethod
    def normalize_channel_frames(
        cls,
        frames: List[np.ndarray],
        mode: str = 'divide',
        per_frame: bool = True
    ) -> List[np.ndarray]:
        """
        Normalize all frames for a channel.
        
        Args:
            frames: List of frames for one channel
            mode: Normalization mode ('divide' or 'subtract')
            per_frame: If True, compute background per frame.
                       If False, use global median across all frames.
        
        Returns:
            List of normalized frames
        """
        if per_frame:
            return [cls.normalize_frame(f, mode=mode) for f in frames]
        else:
            # Global background across all frames
            global_background = float(np.median([np.median(f) for f in frames]))
            return [cls.normalize_frame(f, background=global_background, mode=mode) for f in frames]


def normalize_channels(
    channels: Dict[str, List[np.ndarray]],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, List[np.ndarray]]:
    """
    Normalize all channels according to Methodik v3 §3.1.
    
    Args:
        channels: Dictionary mapping channel name to list of frames
        config: Optional configuration with keys:
            - mode: 'divide' (default) or 'subtract'
            - per_channel: If True, normalize each channel independently
            - per_frame: If True, compute background per frame
    
    Returns:
        Dictionary of normalized channel frames
    """
    config = config or {}
    mode = config.get('mode', 'divide')
    per_frame = config.get('per_frame', True)
    
    normalized = {}
    for channel_name, frames in channels.items():
        normalized[channel_name] = LinearNormalizer.normalize_channel_frames(
            frames,
            mode=mode,
            per_frame=per_frame
        )
    
    return normalized
