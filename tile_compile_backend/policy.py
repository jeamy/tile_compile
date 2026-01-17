import numpy as np
from typing import List, Dict, Any, Optional
import logging

class PolicyValidator:
    """
    Enforces the core invariants of Methodik v3
    """
    @staticmethod
    def validate_linearity(data: np.ndarray) -> bool:
        """
        Check if data is strictly linear
        - No stretch
        - No non-linear operators
        """
        # Check for extreme values or non-linear transformations
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logging.warning("Non-linear or invalid values detected")
            return False
        
        # Rough heuristic for linearity
        std = float(np.std(data))
        if std == 0.0:
            return True
        skewness = np.abs(np.mean((data - np.mean(data))**3) / (std**3))
        return bool(skewness < 0.5)  # Very low skewness suggests linearity

    @staticmethod
    def validate_frame_count(frames: List[np.ndarray], min_frames: int = 800) -> bool:
        """
        Ensure sufficient number of frames
        """
        return len(frames) >= min_frames

    @staticmethod
    def validate_channel_separation(channels: Dict[str, List[np.ndarray]]) -> bool:
        """
        Verify channels are processed independently
        """
        if not all(len(ch) > 0 for ch in channels.values()):
            logging.warning("Empty channels detected")
            return False
        
        # Check geometric consistency across channels
        ref_shape = channels['G'][0].shape
        return all(
            np.array_equal(ch[0].shape, ref_shape) 
            for ch in channels.values() 
            if ch[0] is not None
        )

class PhaseManager:
    """
    Manages pipeline phases with strict validation
    """
    PHASES = [
        'SCAN_INPUT',
        'REGISTRATION',
        'CHANNEL_SPLIT',
        'NORMALIZATION',
        'GLOBAL_METRICS',
        'TILE_GRID',
        'LOCAL_METRICS',
        'TILE_RECONSTRUCTION',
        'STATE_CLUSTERING',
        'SYNTHETIC_FRAMES',
        'STACKING',
        'DEBAYER',
        'DONE',
        'FAILED',
    ]

    def __init__(self):
        self.current_phase = None
        self.phase_data = {}
    
    def advance_phase(self, phase: str, data: Optional[Dict[str, Any]] = None):
        """
        Move to next phase with optional data
        """
        if phase not in self.PHASES:
            raise ValueError(f"Invalid phase: {phase}")
        
        # Validate phase progression
        if self.current_phase:
            current_index = self.PHASES.index(self.current_phase)
            next_index = self.PHASES.index(phase)
            
            if next_index <= current_index:
                raise ValueError(f"Cannot go back from {self.current_phase} to {phase}")
        
        self.current_phase = phase
        if data:
            self.phase_data[phase] = data
        
        logging.info(f"Advancing to phase: {phase}")
        return self

    def get_phase_data(self, phase: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data for a specific phase
        """
        return self.phase_data.get(phase)

    def validate_phase_transition(self, data: Dict[str, Any]) -> bool:
        """
        Perform validation specific to current phase transition
        """
        policy_checks = {
            'SCAN_INPUT': self._validate_scan_input,
            'REGISTRATION': self._validate_registration,
            'CHANNEL_SPLIT': self._validate_channel_split,
            'NORMALIZATION': self._validate_normalization,
            'GLOBAL_METRICS': self._validate_global_metrics,
            'TILE_GRID': self._validate_tile_grid,
            'LOCAL_METRICS': self._validate_local_metrics,
            'TILE_RECONSTRUCTION': self._validate_reconstruction,
            'STATE_CLUSTERING': self._validate_clustering,
            'SYNTHETIC_FRAMES': self._validate_synthetic,
            'STACKING': self._validate_stacking,
        }

        phase = self.current_phase
        if isinstance(data, dict):
            if "registered_frames" in data:
                phase = "REGISTRATION"
            elif "channels" in data:
                phase = "CHANNEL_SPLIT"
            elif "frames" in data:
                phase = "SCAN_INPUT"

        validator = policy_checks.get(phase)
        return validator(data) if validator else True

    def _validate_scan_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate input scan phase
        """
        frames = data.get('frames', [])
        return bool(len(frames) > 0)

    def _validate_registration(self, data: Dict[str, Any]) -> bool:
        """
        Validate registration phase
        """
        frames = data.get('registered_frames', [])
        return len(frames) > 0 and all(PolicyValidator.validate_linearity(frame) for frame in frames)

    def _validate_channel_split(self, data: Dict[str, Any]) -> bool:
        """
        Validate channel split phase
        """
        channels = data.get('channels', {})
        return PolicyValidator.validate_channel_separation(channels)

    def _validate_normalization(self, data: Dict[str, Any]) -> bool:
        """
        Validate normalization phase (Methodik v3 §3.1)
        """
        normalized = data.get('normalized_channels', {})
        if not normalized:
            return False
        # Check all channels have frames
        return all(len(frames) > 0 for frames in normalized.values())

    def _validate_global_metrics(self, data: Dict[str, Any]) -> bool:
        """
        Validate global metrics phase (Methodik v3 §3.2)
        """
        metrics = data.get('global_metrics', {})
        if not metrics:
            return False
        # Check each channel has G_f_c
        for channel_metrics in metrics.values():
            if 'G_f_c' not in channel_metrics:
                return False
        return True

    def _validate_tile_grid(self, data: Dict[str, Any]) -> bool:
        """
        Validate tile grid phase (Methodik v3 §3.3)
        """
        grid = data.get('tile_grid', {})
        if not grid:
            return False
        # Must have tile_size and overlap
        return 'tile_size' in grid and 'overlap' in grid

    def _validate_local_metrics(self, data: Dict[str, Any]) -> bool:
        """
        Validate local metrics phase (Methodik v3 §3.4)
        """
        metrics = data.get('local_metrics', {})
        return bool(metrics)

    def _validate_reconstruction(self, data: Dict[str, Any]) -> bool:
        """
        Validate reconstruction phase (Methodik v3 §3.6)
        """
        result = data.get('reconstruction', {})
        if not result:
            return False
        # Check no NaN/Inf in output
        channels = result.get('channels', {})
        for ch_result in channels.values():
            arr = ch_result.get('reconstructed')
            if arr is not None:
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    logging.error("Reconstruction contains NaN/Inf")
                    return False
        return True

    def _validate_clustering(self, data: Dict[str, Any]) -> bool:
        """
        Validate clustering phase (Methodik v3 §3.7)
        """
        clustering = data.get('clustering', {})
        if not clustering:
            return True  # Optional phase
        # Check cluster count is in valid range
        for ch_clustering in clustering.values():
            n = ch_clustering.get('n_clusters', 0)
            if not (15 <= n <= 30):
                logging.warning(f"Cluster count {n} outside recommended range [15, 30]")
        return True

    def _validate_synthetic(self, data: Dict[str, Any]) -> bool:
        """
        Validate synthetic frames phase (Methodik v3 §3.8)
        """
        synthetic = data.get('synthetic_frames', {})
        if not synthetic:
            return True  # Optional phase
        return True

    def _validate_stacking(self, data: Dict[str, Any]) -> bool:
        """
        Validate stacking phase (Methodik v3 §3.8)
        """
        stacked = data.get('stacked_channels', {})
        if not stacked:
            return False
        # Check no NaN/Inf
        for arr in stacked.values():
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                logging.error("Stacked output contains NaN/Inf")
                return False
        return True


def run_pipeline(input_data):
    """
    Central pipeline runner with strict phase management
    """
    phase_manager = PhaseManager()
    
    try:
        # Scan Input
        phase_manager.advance_phase('SCAN_INPUT', {'frames': input_data})
        
        # Registration
        registered_frames = perform_registration(input_data)
        phase_manager.advance_phase('REGISTRATION', {'registered_frames': registered_frames})
        
        # Channel Split
        channels = split_channels(registered_frames)
        phase_manager.advance_phase('CHANNEL_SPLIT', {'channels': channels})
        
        # Continue with other phases...
        phase_manager.advance_phase('DONE')
        
        return phase_manager
    
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        phase_manager.advance_phase('FAILED')
        raise

# Placeholder functions - to be implemented
def perform_registration(frames):
    # Actual registration logic
    return frames

def split_channels(frames):
    # Actual channel splitting logic
    return {
        'R': [frame for frame in frames],
        'G': [frame for frame in frames],
        'B': [frame for frame in frames]
    }