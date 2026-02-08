"""
Test Suite: Linearity Enforcement

Tests Methodik v3 linearity validation (hard assumption):
- Linear frame detection
- Non-linear stretch detection
- Histogram clipping detection
- Frame rejection based on linearity

Ensures the pipeline rejects non-linear data as per Methodik v3 §1.1.
"""

import numpy as np
import pytest
from tile_compile_backend.linearity import LinearityValidator, validate_frames_linearity

class TestLinearityEnforcement:
    @pytest.fixture
    def linear_frame(self):
        # Create a synthetic linear frame
        np.random.seed(42)
        return np.random.normal(100, 20, (256, 256))
    
    @pytest.fixture
    def non_linear_frame(self):
        # Create a non-linear frame with artificial stretching
        np.random.seed(42)
        frame = np.random.normal(100, 20, (256, 256))
        return np.power(frame, 2)  # Non-linear transformation
    
    def test_moment_linearity_test(self, linear_frame, non_linear_frame):
        """
        Test moment-based linearity detection
        """
        linear_moments = LinearityValidator._moment_linearity_test(linear_frame)
        non_linear_moments = LinearityValidator._moment_linearity_test(non_linear_frame)
        
        # Linear frame should have low skewness and kurtosis
        assert abs(linear_moments['skewness']) < 1.0
        assert abs(linear_moments['kurtosis']) < 1.0
        
        # Non-linear frame should have higher skewness and kurtosis
        assert abs(non_linear_moments['skewness']) > 1.0
        assert abs(non_linear_moments['kurtosis']) > 1.0
    
    def test_spectral_linearity_test(self, linear_frame, non_linear_frame):
        """
        Test spectral domain linearity detection
        """
        linear_spectral = LinearityValidator._spectral_linearity_test(linear_frame)
        non_linear_spectral = LinearityValidator._spectral_linearity_test(non_linear_frame)
        
        # Linear frame should have high energy ratio
        assert linear_spectral['energy_ratio'] > 0.9
        
        # Non-linear frame should have lower energy ratio
        assert non_linear_spectral['energy_ratio'] < 0.9
    
    def test_spatial_linearity_test(self, linear_frame, non_linear_frame):
        """
        Test spatial domain linearity detection
        """
        linear_spatial = LinearityValidator._spatial_linearity_test(linear_frame)
        non_linear_spatial = LinearityValidator._spatial_linearity_test(non_linear_frame)
        
        # Linear frame should have consistent gradient
        assert linear_spatial['gradient_consistency'] < 0.5
        
        # Non-linear frame may have more inconsistent gradients
        assert non_linear_spatial['gradient_consistency'] > 0.5
    
    def test_frame_linearity_validation(self, linear_frame, non_linear_frame):
        """
        Test comprehensive frame linearity validation
        """
        linear_validation = LinearityValidator.validate_frame_linearity(
            linear_frame, 
            {'strictness': 'strict'}
        )
        non_linear_validation = LinearityValidator.validate_frame_linearity(
            non_linear_frame, 
            {'strictness': 'strict'}
        )
        
        # Linear frame should pass validation
        assert linear_validation['is_linear'] is True
        
        # Non-linear frame should fail validation
        assert non_linear_validation['is_linear'] is False
    
    def test_multiple_frames_linearity(self):
        """
        Test linearity validation for multiple frames
        """
        np.random.seed(42)
        
        # Generate mixed frames
        frames = [
            np.random.normal(100, 20, (256, 256)),  # Linear
            np.random.normal(100, 20, (256, 256)),  # Linear
            np.power(np.random.normal(100, 20, (256, 256)), 2)  # Non-linear
        ]
        
        validation_result = validate_frames_linearity(
            np.array(frames), 
            {'strictness': 'strict'}
        )
        
        assert validation_result['overall_linearity'] < 1.0
        assert len(validation_result['valid_frames']) == 2
        assert len(validation_result['rejected_frames']) == 1