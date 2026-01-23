import numpy as np
from typing import List, Dict, Tuple, Any, Optional


# Fallback threshold epsilon (Methodik v3 §3.6)
DEFAULT_EPSILON = 1e-10


class TileReconstructor:
    """
    Tile-based image reconstruction following Methodik v3
    """
    def __init__(self, tile_size: int = 64, overlap: float = 0.25, epsilon: float = DEFAULT_EPSILON):
        self.tile_size = tile_size
        self.overlap = overlap
        self.epsilon = epsilon
        self.fallback_tiles: List[Tuple[int, int]] = []  # Track tiles using fallback
    
    def reconstruct_channel(
        self, 
        frames: List[np.ndarray], 
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reconstruct a channel using tile-based approach per Methodik v3 §3.6.
        
        Formula:
            I_t,c(p) = Σ_f W_f,t,c · I_f,c(p) / Σ_f W_f,t,c
        
        With fallback for D_t,c < ε:
            I_t,c(p) = (1/N) · Σ_f I_f,c(p)
        
        Args:
            frames: List of input frames for a channel
            metrics: Precomputed global and tile metrics with keys:
                - global: {G_f_c: [...], ...}
                - tiles: {Q_local: [[...], ...], ...}
        
        Returns:
            Dict with:
                - reconstructed: Reconstructed channel frame
                - fallback_tiles: List of (y, x) tiles using fallback
                - n_fallback: Number of fallback tiles
        """
        self.fallback_tiles = []
        
        # Determine output frame size
        h, w = frames[0].shape
        n_frames = len(frames)
        
        # Initialize output frame and weight accumulation
        output = np.zeros((h, w), dtype=np.float64)
        weight_sum = np.zeros((h, w), dtype=np.float64)
        
        # Extract global quality indices G_f_c
        global_metrics = metrics.get('global', {})
        G_f_c = np.array(global_metrics.get('G_f_c', np.ones(n_frames)))
        
        # Compute step size
        step = int(self.tile_size * (1 - self.overlap))
        
        # Process each tile position
        tile_idx = 0
        for y in range(0, h - self.tile_size + 1, step):
            for x in range(0, w - self.tile_size + 1, step):
                # Extract tile data from all frames
                tile_stack = np.array([
                    frame[y:y+self.tile_size, x:x+self.tile_size].astype(np.float64)
                    for frame in frames
                ])
                
                # Compute effective weights W_f,t,c = G_f,c · L_f,t,c
                L_f_t_c = self._get_local_quality(metrics, tile_idx, n_frames)
                W_f_t_c = G_f_c * L_f_t_c
                
                # Sum of weights D_t,c
                D_t_c = np.sum(W_f_t_c)
                
                # Reconstruct tile
                if D_t_c >= self.epsilon:
                    # Normal weighted reconstruction
                    tile_output = np.sum(
                        tile_stack * W_f_t_c[:, np.newaxis, np.newaxis], 
                        axis=0
                    ) / D_t_c
                else:
                    # Fallback: unweighted mean (Methodik v3 §3.6)
                    tile_output = np.mean(tile_stack, axis=0)
                    self.fallback_tiles.append((y, x))
                
                # Blend tile into output with Hann window
                window = self._create_blending_window(self.tile_size, self.tile_size)
                output[y:y+self.tile_size, x:x+self.tile_size] += tile_output * window
                weight_sum[y:y+self.tile_size, x:x+self.tile_size] += window
                
                tile_idx += 1
        
        # Normalize final output
        output = np.divide(
            output, 
            weight_sum, 
            out=np.zeros_like(output), 
            where=weight_sum > 0
        )
        
        return {
            'reconstructed': output.astype(np.float32),
            'fallback_tiles': self.fallback_tiles,
            'n_fallback': len(self.fallback_tiles),
            'fallback_used': len(self.fallback_tiles) > 0
        }
    
    def _generate_tile_grid(self, h: int, w: int) -> List[Tuple[int, int, List[np.ndarray]]]:
        """
        Generate a grid of tiles with their frame subsets
        """
        step = int(self.tile_size * (1 - self.overlap))
        tile_grid = []
        
        for y in range(0, h - self.tile_size + 1, step):
            for x in range(0, w - self.tile_size + 1, step):
                # Select frames that cover this tile
                tile_frames = self._get_tile_frames(y, x, h, w)
                tile_grid.append((y, x, tile_frames))
        
        return tile_grid
    
    def _get_tile_frames(self, y: int, x: int, h: int, w: int) -> List[np.ndarray]:
        """
        Get frames that contribute to a specific tile
        """
        # Placeholder: Replace with actual frame selection logic
        return []
    
    def _get_local_quality(
        self, 
        metrics: Dict[str, Any], 
        tile_idx: int, 
        n_frames: int
    ) -> np.ndarray:
        """
        Get local quality index L_f,t,c for a specific tile.
        
        Per Methodik v3 §3.4:
            L_f,t,c = exp(Q_local[f,t,c])
        
        Args:
            metrics: Metrics dict with tiles.Q_local
            tile_idx: Index of current tile
            n_frames: Number of frames
        
        Returns:
            Array of L_f_t_c values for all frames
        """
        tile_metrics = metrics.get('tiles', {})
        Q_local = tile_metrics.get('Q_local', None)
        
        if Q_local is None:
            # No local metrics available, return uniform
            return np.ones(n_frames)
        
        # Q_local should be [n_frames][n_tiles] or [n_tiles][n_frames]
        Q_local = np.array(Q_local)
        
        if Q_local.ndim == 1:
            # Single value per tile, replicate for all frames
            if tile_idx < len(Q_local):
                q = float(Q_local[tile_idx])
            else:
                q = 0.0
            return np.full(n_frames, np.exp(np.clip(q, -3, 3)))
        
        if Q_local.ndim == 2:
            # Try to extract per-frame values for this tile
            if Q_local.shape[0] >= n_frames and tile_idx < Q_local.shape[1]:
                q_values = Q_local[:n_frames, tile_idx]
            elif Q_local.shape[1] >= n_frames and tile_idx < Q_local.shape[0]:
                q_values = Q_local[tile_idx, :n_frames]
            else:
                return np.ones(n_frames)
            
            # Clamp and exp
            q_clamped = np.clip(q_values, -3, 3)
            return np.exp(q_clamped)
        
        return np.ones(n_frames)
    
    def _reconstruct_tile(self, tile_frames: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """
        Reconstruct a single tile using weighted averaging
        """
        # Weighted average of frames
        weighted_frames = [frame * weight for frame, weight in zip(tile_frames, weights)]
        return np.mean(weighted_frames, axis=0)
    
    def _blend_tile(self, output: np.ndarray, weights: np.ndarray, 
                    tile_output: np.ndarray, y: int, x: int):
        """
        Blend a tile into the output frame with overlap handling
        """
        tile_h, tile_w = tile_output.shape
        window = self._create_blending_window(tile_h, tile_w)
        
        output[y:y+tile_h, x:x+tile_w] += tile_output * window
        weights[y:y+tile_h, x:x+tile_w] += window
    
    def _create_blending_window(self, h: int, w: int) -> np.ndarray:
        """
        Create a Hann window for smooth tile blending
        """
        y_window = np.hanning(h)
        x_window = np.hanning(w)
        return np.sqrt(np.outer(y_window, x_window))

def reconstruct_channels(
    channels: Dict[str, List[np.ndarray]], 
    metrics: Dict[str, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Reconstruct all channels per Methodik v3 §3.6.
    
    Args:
        channels: Dict mapping channel name to list of frames
        metrics: Dict mapping channel name to metrics dict
        config: Optional config with tile_size, overlap, epsilon
    
    Returns:
        Dict mapping channel name to reconstruction result dict
    """
    config = config or {}
    tile_size = config.get('tile_size', 64)
    overlap = config.get('overlap', 0.25)
    epsilon = config.get('epsilon', DEFAULT_EPSILON)
    
    reconstructor = TileReconstructor(
        tile_size=tile_size, 
        overlap=overlap, 
        epsilon=epsilon
    )
    
    reconstructed_channels = {}
    total_fallback = 0
    
    for channel_name, frames in channels.items():
        channel_metrics = metrics.get(channel_name, {})
        result = reconstructor.reconstruct_channel(frames, channel_metrics)
        reconstructed_channels[channel_name] = result
        total_fallback += result['n_fallback']
    
    return {
        'channels': reconstructed_channels,
        'total_fallback_tiles': total_fallback,
        'any_fallback': total_fallback > 0
    }