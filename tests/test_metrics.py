"""
Test Suite: Metrics Calculation

Tests Methodik v3 metrics computation:
- Global frame metrics (G_f_c, Q_f, background, noise)
- Tile-level metrics (FWHM, roundness, contrast)
- Per-frame and per-channel metric arrays
- Gradient energy calculation

Validates that metrics are computed correctly according to Methodik v3 specification.
"""

import numpy as np
import pytest
from tile_compile_backend.metrics import MetricsCalculator, TileMetricsCalculator, compute_channel_metrics

class TestMetricsCalculator:
    def setup_method(self):
        # Create a set of simulated frames
        np.random.seed(42)
        self.frames = [np.random.normal(0, 1, (100, 100)) for _ in range(10)]

    def test_global_metrics(self):
        metrics = MetricsCalculator.calculate_global_metrics(self.frames)
        
        # Methodik v3: Per-frame metrics as lists
        assert 'background_level' in metrics
        assert 'noise_level' in metrics
        assert 'gradient_energy' in metrics
        assert 'G_f_c' in metrics  # Global quality index per frame
        assert 'Q_f' in metrics    # Raw quality index
        
        # Check per-frame arrays have correct length
        n_frames = len(self.frames)
        assert len(metrics['background_level']) == n_frames
        assert len(metrics['noise_level']) == n_frames
        assert len(metrics['gradient_energy']) == n_frames
        assert len(metrics['G_f_c']) == n_frames
        
        # Check reasonable ranges for per-frame values
        assert all(-2 < b < 2 for b in metrics['background_level'])
        assert all(0 < n < 3 for n in metrics['noise_level'])
        assert all(e > 0 for e in metrics['gradient_energy'])
        assert all(g > 0 for g in metrics['G_f_c'])  # exp() is always positive

    def test_gradient_energy_calculation(self):
        frame = np.random.normal(0, 1, (100, 100))
        energy = MetricsCalculator._calculate_gradient_energy(frame)
        
        assert energy > 0
        assert np.isfinite(energy)

class TestTileMetricsCalculator:
    def setup_method(self):
        # Create a simulated astronomical frame
        np.random.seed(42)
        self.frame = np.random.normal(0, 1, (256, 256))
        
        # Simulate a star-like feature
        center_y, center_x = 128, 128
        for y in range(center_y-10, center_y+10):
            for x in range(center_x-10, center_x+10):
                distance = np.sqrt((y-center_y)**2 + (x-center_x)**2)
                self.frame[y, x] += max(0, 10 - distance)

        self.tile_calculator = TileMetricsCalculator(tile_size=64, overlap=0.25)

    def test_tile_metrics(self):
        tile_metrics = self.tile_calculator.calculate_tile_metrics(self.frame)
        
        # Check all metrics are present
        expected_metrics = [
            'fwhm', 
            'roundness', 
            'contrast', 
            'background_level', 
            'noise_level'
        ]
        
        for metric in expected_metrics:
            assert metric in tile_metrics
            assert len(tile_metrics[metric]) > 0

    def test_tile_generation(self):
        tiles = self.tile_calculator._generate_tiles(self.frame)
        
        # Check tiles are generated with correct size
        for tile in tiles:
            assert tile.shape == (64, 64)

    def test_fwhm_calculation(self):
        # Create a simple test frame with a star
        frame = np.zeros((64, 64))
        frame[32, 32] = 10  # Peak at center
        
        fwhm = self.tile_calculator._calculate_fwhm(frame)
        
        # FWHM should be a positive value
        assert fwhm > 0
        assert fwhm < 64  # Cannot be larger than frame

class TestChannelMetricsComputation:
    def setup_method(self):
        np.random.seed(42)
        self.channels = {
            'R': [np.random.normal(0, 1, (100, 100)) for _ in range(10)],
            'G': [np.random.normal(0, 1, (100, 100)) for _ in range(10)],
            'B': [np.random.normal(0, 1, (100, 100)) for _ in range(10)]
        }

    def test_channel_metrics_computation(self):
        channel_metrics = compute_channel_metrics(self.channels)
        
        # Check metrics for each channel
        for channel_name, metrics in channel_metrics.items():
            assert channel_name in ['R', 'G', 'B']
            
            # Check global metrics
            assert 'global' in metrics
            global_metrics = metrics['global']
            assert 'background_level' in global_metrics
            assert 'noise_level' in global_metrics
            assert 'gradient_energy' in global_metrics
            
            # Check tile metrics
            assert 'tiles' in metrics
            tile_metrics = metrics['tiles']
            tile_metric_keys = [
                'fwhm', 
                'roundness', 
                'contrast', 
                'background_level', 
                'noise_level'
            ]
            
            for key in tile_metric_keys:
                assert key in tile_metrics
                assert len(tile_metrics[key]) > 0