import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def _normalize_metric(values: np.ndarray) -> np.ndarray:
    """Normalize metric values to z-scores (mean=0, std=1)."""
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - mean) / std


def _clamp(x: np.ndarray, lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    """Clamp values to [lo, hi] as per Methodik v3 §3.2."""
    return np.clip(x, lo, hi)


class MetricsCalculator:
    @classmethod
    def calculate_global_metrics(
        cls,
        frames: List[np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        clamp_range: Tuple[float, float] = (-3.0, 3.0)
    ) -> Dict[str, Any]:
        """
        Calculate global metrics for a channel per Methodik v3 §3.2.
        
        Metrics per frame f:
        - B_f: Background level
        - σ_f: Noise level  
        - E_f: Gradient energy
        
        Quality index:
            Q_f = α(-B̃_f) + β(-σ̃_f) + γ(Ẽ_f)
            G_f = exp(clamp(Q_f, -3, 3))
        
        Args:
            frames: List of frames for one channel
            weights: Dict with keys 'background', 'noise', 'gradient' (must sum to 1)
            clamp_range: Tuple (lo, hi) for clamping Q before exp()
        
        Returns:
            Dict with per-frame metrics and quality indices
        """
        weights = weights or {'background': 0.4, 'noise': 0.3, 'gradient': 0.3}
        alpha = weights.get('background', 0.4)
        beta = weights.get('noise', 0.3)
        gamma = weights.get('gradient', 0.3)
        
        # Validate weight normalization (Methodik v3 §3.2 Nebenbedingung)
        weight_sum = alpha + beta + gamma
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.6f}")
        
        n_frames = len(frames)
        
        # Per-frame raw metrics
        B_f = np.array([float(np.median(frame)) for frame in frames])
        sigma_f = np.array([float(np.std(frame)) for frame in frames])
        E_f = np.array([float(cls._calculate_gradient_energy(frame)) for frame in frames])
        
        # Normalize to z-scores (tilde notation in spec)
        B_tilde = _normalize_metric(B_f)
        sigma_tilde = _normalize_metric(sigma_f)
        E_tilde = _normalize_metric(E_f)
        
        # Quality index: Q_f = α(-B̃) + β(-σ̃) + γẼ
        # Lower background and noise are better (negative sign)
        # Higher gradient energy is better (positive sign)
        Q_f = alpha * (-B_tilde) + beta * (-sigma_tilde) + gamma * E_tilde
        
        # Clamp before exp() (Methodik v3 §3.2 Stabilitätsregel)
        Q_f_clamped = _clamp(Q_f, clamp_range[0], clamp_range[1])
        
        # Global quality index
        G_f = np.exp(Q_f_clamped)
        
        return {
            'background_level': B_f.tolist(),
            'noise_level': sigma_f.tolist(),
            'gradient_energy': E_f.tolist(),
            'Q_f': Q_f.tolist(),
            'Q_f_clamped': Q_f_clamped.tolist(),
            'G_f_c': G_f.tolist(),
            'weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
            'n_frames': n_frames
        }
    
    @staticmethod
    def _calculate_gradient_energy(frame: np.ndarray) -> float:
        """
        Calculate gradient energy for a single frame
        """
        # Sobel filters for x and y gradients
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = sobel_x.T
        
        grad_x = np.abs(np.convolve(frame.flatten(), sobel_x.flatten(), mode='valid'))
        grad_y = np.abs(np.convolve(frame.flatten(), sobel_y.flatten(), mode='valid'))
        
        # Combined gradient magnitude
        return np.mean(np.sqrt(grad_x**2 + grad_y**2))

class TileMetricsCalculator:
    """
    Calculate local metrics per tile
    """
    def __init__(self, tile_size: int = 64, overlap: float = 0.25):
        self.tile_size = tile_size
        self.overlap = overlap

    @staticmethod
    def _robust_sigma(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        if x.size <= 0:
            return 0.0
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        sig = 1.4826 * mad
        if not np.isfinite(sig) or sig < 1e-12:
            sig = float(np.std(x))
        return float(sig)

    @staticmethod
    def _box_blur_same(t: np.ndarray, k: int) -> np.ndarray:
        t = np.asarray(t, dtype=np.float32)
        h, w = t.shape
        k = int(k)
        if k < 3:
            k = 3
        if (k % 2) == 0:
            k += 1
        max_k = min(h, w)
        if (max_k % 2) == 0:
            max_k -= 1
        if max_k < 3:
            return t
        if k > max_k:
            k = max_k
        pad = k // 2
        a = np.pad(t, ((pad, pad), (pad, pad)), mode='reflect')
        ii = np.cumsum(np.cumsum(a, axis=0), axis=1)
        ii = np.pad(ii, ((1, 0), (1, 0)), mode='constant')
        y0 = np.arange(0, h)
        x0 = np.arange(0, w)
        y1 = y0 + k
        x1 = x0 + k
        s = ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)]
        return (s / float(k * k)).astype(np.float32, copy=False)

    def _tile_highpass(self, tile: np.ndarray) -> np.ndarray:
        t = tile.astype(np.float32, copy=False)
        bg = self._box_blur_same(t, 31)
        return (t - bg).astype(np.float32, copy=False)

    def _tile_background_and_noise(self, tile: np.ndarray) -> tuple[float, float, np.ndarray]:
        t = tile.astype(np.float32, copy=False)
        resid = self._tile_highpass(t)
        bg0 = float(np.median(t))
        sigma0 = self._robust_sigma(resid)
        thr = bg0 + 3.0 * sigma0
        m = t <= thr
        if not np.any(m):
            m = np.ones_like(t, dtype=bool)
        bg = float(np.median(t[m]))
        sig = self._robust_sigma(resid[m])
        return bg, float(sig), resid
    
    def calculate_tile_metrics(self, frame: np.ndarray) -> Dict[str, List[float]]:
        """Calculate metrics for each tile in a frame.

        The returned dictionary contains per-tile lists for:
        - fwhm:            FWHM estimate (star-dominated tiles)
        - roundness:       qualitative roundness metric
        - contrast:        simple local contrast proxy
        - background_level:local background B_local (median)
        - noise_level:     local noise σ (std)
        - gradient_energy: local gradient energy E (per Methodik v3 §3.4 / Anhang A.3)
        """
        tiles = self._generate_tiles(frame)
        
        tile_metrics: Dict[str, List[float]] = {
            'fwhm': [],            # Full Width at Half Maximum
            'roundness': [],       # Star roundness
            'contrast': [],        # Tile contrast
            'background_level': [],
            'noise_level': [],
            'gradient_energy': [], # Local gradient energy E
        }
        
        for tile in tiles:
            tile_fwhm = self._calculate_fwhm(tile)
            tile_round = self._calculate_roundness(tile)
            tile_con = self._calculate_contrast(tile)
            tile_bg, tile_noise, resid = self._tile_background_and_noise(tile)
            tile_E = float(self._calculate_gradient_energy(resid))

            tile_metrics['fwhm'].append(tile_fwhm)
            tile_metrics['roundness'].append(tile_round)
            tile_metrics['contrast'].append(tile_con)
            tile_metrics['background_level'].append(tile_bg)
            tile_metrics['noise_level'].append(tile_noise)
            tile_metrics['gradient_energy'].append(tile_E)
        
        return tile_metrics

    def _generate_tiles(self, frame: np.ndarray) -> List[np.ndarray]:
        """Generate overlapping tiles from a frame."""
        h, w = frame.shape
        step = int(self.tile_size * (1 - self.overlap))
        
        tiles: List[np.ndarray] = []
        for y in range(0, h - self.tile_size + 1, step):
            for x in range(0, w - self.tile_size + 1, step):
                tile = frame[y:y + self.tile_size, x:x + self.tile_size]
                tiles.append(tile)
        
        return tiles
    
    def _calculate_fwhm(self, tile: np.ndarray) -> float:
        """
        Estimate Full Width at Half Maximum
        """
        # Simplified FWHM estimation
        peak = np.max(tile)
        half_max = peak / 2
        
        # Count pixels above half max
        above_half_max = np.sum(tile >= half_max)
        return np.sqrt(above_half_max / np.pi)
    
    def _calculate_roundness(self, tile: np.ndarray) -> float:
        """
        Calculate star roundness
        """
        # Simplified roundness metric
        max_val = np.max(tile)
        max_indices = np.argwhere(tile == max_val)
        
        # Compute spread of maximum points
        spread = np.std(max_indices, axis=0)
        return 1 / (1 + spread.mean())
    
    def _calculate_contrast(self, tile: np.ndarray) -> float:
        """Calculate local contrast."""
        t = tile.astype(np.float32, copy=False)
        t_max = float(np.max(t))
        t_min = float(np.min(t))
        return (t_max - t_min) / (t_max + t_min + 1e-8)

    def _calculate_gradient_energy(self, tile: np.ndarray) -> float:
        """Calculate local gradient energy E for a tile.

        Per Methodik v3 Anhang A.3 wird E typischerweise als Mittelwert von
        |∇I|^2 definiert. Hier verwenden wir eine einfache Approximation mit
        NumPy-Gradienten, die für die nachgelagerte MAD-Normalisierung ausreichend
        ist.
        """
        t = tile.astype(np.float32, copy=False)
        gy, gx = np.gradient(t)
        grad_sq = gx * gx + gy * gy
        return float(np.mean(grad_sq))

def compute_channel_metrics(channels: Dict[str, List[np.ndarray]]) -> Dict[str, Dict]:
    """
    Compute metrics for all channels
    """
    channel_metrics = {}
    tile_calculator = TileMetricsCalculator()
    
    for channel_name, frames in channels.items():
        # Global metrics
        global_metrics = MetricsCalculator.calculate_global_metrics(frames)
        
        # Tile metrics (using first frame as representative)
        tile_metrics = tile_calculator.calculate_tile_metrics(frames[0])
        
        channel_metrics[channel_name] = {
            'global': global_metrics,
            'tiles': tile_metrics
        }
    
    return channel_metrics