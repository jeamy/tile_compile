import numpy as np
import scipy.stats as stats
import pywt
from typing import Dict, Any, Optional, Tuple

class LinearityValidator:
    @classmethod
    def validate_frame_linearity(
        cls, 
        frame: np.ndarray, 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive linearity validation for astronomical frames
        
        Args:
            frame: Input astronomical frame
            config: Validation configuration
        
        Returns:
            Linearity validation results
        """
        config = config or {}
        
        # Configure validation parameters
        strictness = config.get('strictness', 'strict')
        
        # Statistical tests
        moment_test = cls._moment_linearity_test(frame)
        spectral_test = cls._spectral_linearity_test(frame)
        spatial_test = cls._spatial_linearity_test(frame)
        
        # Combine test results
        overall_linearity = cls._aggregate_linearity_scores(
            moment_test, 
            spectral_test, 
            spatial_test,
            strictness
        )
        
        return {
            'is_linear': overall_linearity['is_linear'],
            'moment_test': moment_test,
            'spectral_test': spectral_test,
            'spatial_test': spatial_test,
            'linearity_score': overall_linearity['score'],
            'diagnostics': overall_linearity['diagnostics']
        }
    
    @staticmethod
    def _moment_linearity_test(frame: np.ndarray) -> Dict[str, float]:
        """
        Analyze frame using higher-order moment statistics
        """
        # Remove background
        frame_data = frame.flatten()
        frame_data = frame_data[frame_data > 0]  # Ignore background
        
        return {
            'skewness': stats.skew(frame_data),
            'kurtosis': stats.kurtosis(frame_data),
            'variance_coefficient': np.std(frame_data) / np.mean(frame_data)
        }
    
    @staticmethod
    def _spectral_linearity_test(frame: np.ndarray) -> Dict[str, float]:
        """
        Perform spectral analysis using wavelet transform
        """
        # Wavelet decomposition
        coeffs = pywt.wavedec2(frame, 'haar', level=3)
        
        # Analyze detail coefficients
        detail_energies = [
            np.sum(np.abs(coeff)**2) for coeff in coeffs[1:]
        ]
        
        return {
            'wavelet_coherence': np.mean(detail_energies),
            'energy_ratio': detail_energies[0] / np.sum(detail_energies)
        }
    
    @staticmethod
    def _spatial_linearity_test(frame: np.ndarray) -> Dict[str, float]:
        """
        Assess spatial domain linearity
        """
        # Compute gradients
        gy, gx = np.gradient(frame)
        
        return {
            'gradient_consistency': np.std(gx) / np.mean(np.abs(gx)),
            'edge_uniformity': np.std(gy) / np.mean(np.abs(gy))
        }
    
    @classmethod
    def _aggregate_linearity_scores(
        cls,
        moment_test: Dict[str, float],
        spectral_test: Dict[str, float],
        spatial_test: Dict[str, float],
        strictness: str = 'strict'
    ) -> Dict[str, Any]:
        """
        Combine linearity test results
        """
        # Strictness-based thresholds
        thresholds = {
            'strict': {
                'skewness_max': 0.1,
                'kurtosis_max': 0.5,
                'variance_max': 0.2,
                'energy_ratio_min': 0.95,
                'gradient_consistency_max': 0.1
            },
            'moderate': {
                'skewness_max': 0.5,
                'kurtosis_max': 1.0,
                'variance_max': 0.5,
                'energy_ratio_min': 0.9,
                'gradient_consistency_max': 0.3
            },
            'permissive': {
                'skewness_max': 1.0,
                'kurtosis_max': 2.0,
                'variance_max': 1.0,
                'energy_ratio_min': 0.8,
                'gradient_consistency_max': 0.5
            }
        }[strictness]
        
        diagnostics = []
        
        # Check moment linearity
        moment_linear = (
            abs(moment_test['skewness']) < thresholds['skewness_max'] and
            abs(moment_test['kurtosis']) < thresholds['kurtosis_max'] and
            moment_test['variance_coefficient'] < thresholds['variance_max']
        )
        if not moment_linear:
            diagnostics.append('Moment statistics indicate non-linearity')
        
        # Check spectral linearity
        spectral_linear = (
            spectral_test['energy_ratio'] > thresholds['energy_ratio_min']
        )
        if not spectral_linear:
            diagnostics.append('Spectral analysis suggests non-linear transformation')
        
        # Check spatial linearity
        spatial_linear = (
            spatial_test['gradient_consistency'] < thresholds['gradient_consistency_max']
        )
        if not spatial_linear:
            diagnostics.append('Spatial gradient inconsistency detected')
        
        # Compute overall linearity score
        linearity_components = [
            moment_linear, 
            spectral_linear, 
            spatial_linear
        ]
        
        linearity_score = np.mean(linearity_components)
        is_linear = all(linearity_components)
        
        return {
            'is_linear': is_linear,
            'score': linearity_score,
            'diagnostics': diagnostics
        }

def validate_frames_linearity(
    frames: np.ndarray, 
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate linearity for multiple frames
    
    Args:
        frames: Array of frames to validate
        config: Validation configuration
    
    Returns:
        Linearity validation results
    """
    results = []
    rejected_frames = []
    
    for frame in frames:
        validation = LinearityValidator.validate_frame_linearity(frame, config)
        results.append(validation)
        
        if not validation['is_linear']:
            rejected_frames.append(frame)
    
    return {
        'results': results,
        'valid_frames': [f for f, r in zip(frames, results) if r['is_linear']],
        'rejected_frames': rejected_frames,
        'overall_linearity': np.mean([r['is_linear'] for r in results])
    }