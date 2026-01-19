import numpy as np
import pytest
import cv2
from tile_compile_backend.registration import CFARegistration, BayerPattern

class TestCFARegistration:
    @pytest.fixture
    def sample_frames(self):
        # Create synthetic CFA frames with different characteristics
        np.random.seed(42)
        base_frame = np.random.normal(100, 20, (256, 256))
        
        # Simulate slight translations and rotations
        frames = [base_frame]
        for i in range(5):
            # Create slightly different frame
            noise = np.random.normal(0, 5, base_frame.shape)
            translation_x = np.random.randint(-5, 5)
            translation_y = np.random.randint(-5, 5)
            
            # Apply translation
            translated = np.roll(base_frame + noise, (translation_y, translation_x), axis=(0, 1))
            frames.append(translated)
        
        return frames

    def test_reference_frame_selection(self, sample_frames):
        """
        Test reference frame selection based on star count
        """
        ref_frame, ref_index = CFARegistration._select_reference_frame(sample_frames)
        
        assert ref_frame is not None
        assert 0 <= ref_index < len(sample_frames)
    
    def test_star_counting(self, sample_frames):
        """
        Validate star detection method
        """
        star_counts = [CFARegistration._count_stars(frame) for frame in sample_frames]
        
        assert all(count >= 0 for count in star_counts)
        assert len(star_counts) == len(sample_frames)
    
    def test_subplane_extraction(self, sample_frames):
        """
        Test Bayer pattern subplane extraction
        """
        frame = sample_frames[0]
        subplanes = CFARegistration._extract_subplanes(frame, BayerPattern.RGGB)
        
        assert len(subplanes) == 4  # R, G1, G2, B
        for plane in subplanes:
            assert plane.shape[0] == frame.shape[0] // 2
            assert plane.shape[1] == frame.shape[1] // 2
    
    def test_cfa_registration(self, sample_frames):
        """
        Test full CFA registration process
        """
        registration_result = CFARegistration.register_cfa_frames(
            sample_frames, 
            bayer_pattern=BayerPattern.RGGB
        )
        
        assert 'registered_frames' in registration_result
        assert 'transformations' in registration_result
        assert 'quality_metrics' in registration_result
        
        # Check that number of registered frames matches input
        assert len(registration_result['registered_frames']) > 0
        assert len(registration_result['registered_frames']) <= len(sample_frames)
    
    def test_registration_quality_metrics(self, sample_frames):
        """
        Validate registration quality metrics
        """
        registration_result = CFARegistration.register_cfa_frames(
            sample_frames, 
            bayer_pattern=BayerPattern.RGGB
        )
        
        quality_metrics = registration_result['quality_metrics']
        
        # Quality metrics should be between 0 and 1
        assert all(0 <= metric <= 1 for metric in quality_metrics)
        
        # First frame (reference) should have perfect quality
        assert quality_metrics[0] == 1.0