import numpy as np
import pytest
from tile_compile_backend.synthetic import (
    SyntheticFrameGenerator, 
    generate_channel_synthetic_frames
)

class TestSyntheticFrameGenerator:
    def setup_method(self):
        np.random.seed(42)
        self.frames = [np.random.normal(0, 1, (100, 100)) for _ in range(50)]
        self.metrics = {
            'global': {
                'noise_level': np.random.random(50),
                'background_level': np.random.random(50)
            },
            'tiles': {
                'Q_local': [np.random.random((10, 10)) for _ in range(50)]
            }
        }

    def test_linear_average(self):
        synthetic_frames = SyntheticFrameGenerator.generate_synthetic_frames(
            self.frames, 
            self.metrics, 
            {'method': 'linear_average'}
        )
        
        assert len(synthetic_frames) == 1
        assert synthetic_frames[0].shape == self.frames[0].shape

    def test_weighted_average(self):
        synthetic_frames = SyntheticFrameGenerator.generate_synthetic_frames(
            self.frames, 
            self.metrics, 
            {'method': 'weighted_average'}
        )
        
        assert len(synthetic_frames) == 1
        assert synthetic_frames[0].shape == self.frames[0].shape

    def test_noise_injection(self):
        synthetic_frames = SyntheticFrameGenerator.generate_synthetic_frames(
            self.frames, 
            self.metrics, 
            {'method': 'noise_injection'}
        )
        
        assert len(synthetic_frames) == 3
        for frame in synthetic_frames:
            assert frame.shape == self.frames[0].shape

    def test_frame_count_constraints(self):
        # Test minimum frames
        with pytest.raises(ValueError):
            SyntheticFrameGenerator.generate_synthetic_frames(
                self.frames[:10], 
                self.metrics, 
                {'frames_min': 15}
            )

    def test_channel_synthetic_frames(self):
        channels = {
            'R': self.frames[:25],
            'G': self.frames[25:],
            'B': self.frames[25:]
        }
        
        channel_synthetic_frames = generate_channel_synthetic_frames(
            channels, 
            {'R': self.metrics, 'G': self.metrics, 'B': self.metrics}
        )
        
        assert set(channel_synthetic_frames.keys()) == {'R', 'G', 'B'}
        for channel, frames in channel_synthetic_frames.items():
            assert len(frames) > 0