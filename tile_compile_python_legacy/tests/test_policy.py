"""
Test Suite: Policy Validation

Tests Methodik v3 policy enforcement:
- Linearity validation (hard assumption)
- Frame count validation
- Channel separation verification
- Phase progression testing

Ensures the pipeline strictly adheres to Methodik v3 hard assumptions.
"""

import numpy as np
import pytest
from tile_compile_backend.policy import PolicyValidator, PhaseManager

class TestPolicyValidator:
    def test_linearity_valid(self):
        # Simulate a linear dataset
        data = np.random.normal(0, 1, (100, 100))
        assert PolicyValidator.validate_linearity(data) is True

    def test_linearity_invalid(self):
        # Simulate a non-linear dataset with extreme values
        data = np.random.exponential(1, (100, 100))
        assert PolicyValidator.validate_linearity(data) is False

    def test_frame_count_valid(self):
        frames = [np.zeros((10, 10)) for _ in range(1000)]
        assert PolicyValidator.validate_frame_count(frames) is True

    def test_frame_count_invalid(self):
        frames = [np.zeros((10, 10)) for _ in range(10)]
        assert PolicyValidator.validate_frame_count(frames) is False

    def test_channel_separation_valid(self):
        channels = {
            'R': [np.zeros((100, 100)) for _ in range(5)],
            'G': [np.zeros((100, 100)) for _ in range(5)],
            'B': [np.zeros((100, 100)) for _ in range(5)]
        }
        assert PolicyValidator.validate_channel_separation(channels) is True

    def test_channel_separation_invalid(self):
        channels = {
            'R': [np.zeros((100, 100)) for _ in range(5)],
            'G': [np.zeros((50, 50)) for _ in range(5)],
            'B': []
        }
        assert PolicyValidator.validate_channel_separation(channels) is False

class TestPhaseManager:
    def test_phase_progression(self):
        pm = PhaseManager()
        
        # Valid progression
        pm.advance_phase('SCAN_INPUT', {'frames': [1, 2, 3]})
        pm.advance_phase('REGISTRATION', {'registered_frames': [1, 2, 3]})
        pm.advance_phase('CHANNEL_SPLIT', {'channels': {'R': [1], 'G': [2], 'B': [3]}})
        
        assert pm.current_phase == 'CHANNEL_SPLIT'
        assert len(pm.phase_data) == 3

    def test_invalid_phase_regression(self):
        pm = PhaseManager()
        
        pm.advance_phase('SCAN_INPUT')
        
        with pytest.raises(ValueError):
            pm.advance_phase('SCAN_INPUT')  # Cannot go back to same phase

    def test_phase_data_retrieval(self):
        pm = PhaseManager()
        
        pm.advance_phase('SCAN_INPUT', {'frames': [1, 2, 3]})
        data = pm.get_phase_data('SCAN_INPUT')
        
        assert data == {'frames': [1, 2, 3]}

    def test_phase_validation(self):
        pm = PhaseManager()
        
        # Test scan input validation
        assert pm.validate_phase_transition({'frames': [1, 2, 3]}) is True
        assert pm.validate_phase_transition({'frames': []}) is False

        # Advance to registration
        pm.advance_phase('SCAN_INPUT', {'frames': [1, 2, 3]})
        
        # Test registration validation
        assert pm.validate_phase_transition({
            'registered_frames': [np.zeros((10, 10)), np.zeros((10, 10))]
        }) is True

    def test_pipeline_run(self):
        from tile_compile_backend.policy import run_pipeline
        
        # Simulate input data
        input_data = [np.random.rand(100, 100) for _ in range(50)]
        
        try:
            phase_manager = run_pipeline(input_data)
            assert phase_manager.current_phase == 'DONE'
        except Exception as e:
            pytest.fail(f"Pipeline run failed: {e}")