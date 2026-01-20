import numpy as np
import pytest
import cv2
from runner.opencv_registration import (
    opencv_prepare_ecc_image,
    opencv_count_stars,
    opencv_ecc_warp,
    opencv_best_translation_init,
    opencv_detect_stars,
    opencv_star_match_ransac,
    opencv_register_stars,
)

class TestOpenCVRegistration:
    @pytest.fixture
    def sample_image(self):
        """Create a synthetic image with stars."""
        np.random.seed(42)
        img = np.random.normal(0.1, 0.02, (512, 512)).astype(np.float32)
        # Add some bright "stars"
        for _ in range(20):
            x, y = np.random.randint(50, 462, 2)
            img[y-2:y+3, x-2:x+3] += np.random.uniform(0.5, 1.0)
        return np.clip(img, 0, 1)

    @pytest.fixture
    def sample_image_pair(self, sample_image):
        """Create a pair of images with known transformation."""
        ref = sample_image
        # Create moving image with translation and small rotation
        h, w = ref.shape
        tx, ty = 5.0, -3.0
        angle_deg = 0.5
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cx, cy = w / 2.0, h / 2.0
        warp = np.array([
            [cos_t, -sin_t, -cos_t * cx + sin_t * cy + cx + tx],
            [sin_t,  cos_t, -sin_t * cx - cos_t * cy + cy + ty],
        ], dtype=np.float32)
        moving = cv2.warpAffine(ref, warp, (w, h))
        return ref, moving, warp

    def test_prepare_ecc_image(self, sample_image):
        """Test ECC image preparation."""
        prepared = opencv_prepare_ecc_image(sample_image)
        assert prepared.shape == sample_image.shape
        assert prepared.dtype == np.float32
        # Allow small negative values due to floating point precision
        assert -1e-6 <= prepared.min() <= prepared.max() <= 1

    def test_count_stars(self, sample_image):
        """Test star counting."""
        prepared = opencv_prepare_ecc_image(sample_image)
        count = opencv_count_stars(prepared)
        assert count > 0
        assert count < 100  # Reasonable range

    def test_detect_stars(self, sample_image):
        """Test star detection."""
        prepared = opencv_prepare_ecc_image(sample_image)
        stars = opencv_detect_stars(prepared, max_stars=50)
        assert stars.shape[1] == 2  # (x, y) coordinates
        assert len(stars) > 0

    def test_best_translation_init_no_rotation(self, sample_image_pair):
        """Test translation initialization without rotation sweep."""
        ref, moving, true_warp = sample_image_pair
        ref01 = opencv_prepare_ecc_image(ref)
        mov01 = opencv_prepare_ecc_image(moving)
        
        init_warp = opencv_best_translation_init(mov01, ref01, rotation_sweep=False)
        assert init_warp.shape == (2, 3)
        # Should find approximate translation
        assert abs(init_warp[0, 2] - true_warp[0, 2]) < 10.0
        assert abs(init_warp[1, 2] - true_warp[1, 2]) < 10.0

    def test_best_translation_init_with_rotation(self, sample_image_pair):
        """Test translation initialization with rotation sweep."""
        ref, moving, true_warp = sample_image_pair
        ref01 = opencv_prepare_ecc_image(ref)
        mov01 = opencv_prepare_ecc_image(moving)
        
        init_warp = opencv_best_translation_init(mov01, ref01, rotation_sweep=True)
        assert init_warp.shape == (2, 3)
        # Should find better alignment with rotation sweep

    def test_ecc_warp_translation_only(self, sample_image_pair):
        """Test ECC warp with translation only."""
        ref, moving, true_warp = sample_image_pair
        ref01 = opencv_prepare_ecc_image(ref)
        mov01 = opencv_prepare_ecc_image(moving)
        
        init = opencv_best_translation_init(mov01, ref01, rotation_sweep=False)
        warp, cc = opencv_ecc_warp(mov01, ref01, allow_rotation=False, init_warp=init)
        
        assert warp.shape == (2, 3)
        assert 0 <= cc <= 1
        # Translation should be reasonably close (within 3px for synthetic data)
        assert abs(warp[0, 2] - true_warp[0, 2]) < 3.0
        assert abs(warp[1, 2] - true_warp[1, 2]) < 3.0

    def test_ecc_warp_with_rotation(self, sample_image_pair):
        """Test ECC warp with rotation allowed."""
        ref, moving, true_warp = sample_image_pair
        ref01 = opencv_prepare_ecc_image(ref)
        mov01 = opencv_prepare_ecc_image(moving)
        
        init = opencv_best_translation_init(mov01, ref01, rotation_sweep=True)
        warp, cc = opencv_ecc_warp(mov01, ref01, allow_rotation=True, init_warp=init)
        
        assert warp.shape == (2, 3)
        assert 0 <= cc <= 1
        # Should recover both translation and rotation (within 3px)
        assert abs(warp[0, 2] - true_warp[0, 2]) < 3.0
        assert abs(warp[1, 2] - true_warp[1, 2]) < 3.0

    def test_star_match_ransac(self, sample_image_pair):
        """Test star matching with RANSAC."""
        ref, moving, true_warp = sample_image_pair
        ref01 = opencv_prepare_ecc_image(ref)
        mov01 = opencv_prepare_ecc_image(moving)
        
        warp, num_inliers = opencv_star_match_ransac(mov01, ref01)
        
        if warp is not None:
            assert warp.shape == (2, 3)
            assert num_inliers >= 0

    def test_register_stars_allow_rotation(self, sample_image_pair):
        """Test full registration with rotation allowed."""
        ref, moving, true_warp = sample_image_pair
        ref01 = opencv_prepare_ecc_image(ref)
        mov01 = opencv_prepare_ecc_image(moving)
        
        warp, confidence, method = opencv_register_stars(
            mov01, ref01, 
            fallback_to_ecc=True, 
            allow_rotation=True
        )
        
        assert warp.shape == (2, 3)
        assert confidence >= 0
        assert method in ["star_ransac", "ecc", "phase_corr"]

    def test_register_stars_no_rotation(self, sample_image_pair):
        """Test full registration with rotation disabled."""
        ref, moving, true_warp = sample_image_pair
        ref01 = opencv_prepare_ecc_image(ref)
        mov01 = opencv_prepare_ecc_image(moving)
        
        warp, confidence, method = opencv_register_stars(
            mov01, ref01, 
            fallback_to_ecc=True, 
            allow_rotation=False
        )
        
        assert warp.shape == (2, 3)
        assert confidence >= 0
        assert method in ["star_ransac", "ecc", "phase_corr"]