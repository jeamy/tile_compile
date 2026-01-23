import numpy as np
import pytest
from tile_compile_backend.tile_grid import TileGridGenerator, generate_multi_channel_grid

class TestTileGridGeneration:
    @pytest.fixture
    def sample_astronomical_frame(self):
        """
        Generate a synthetic astronomical frame
        """
        np.random.seed(42)
        frame = np.random.normal(100, 20, (1024, 1024))
        
        # Simulate some stars
        for _ in range(50):
            x = np.random.randint(0, 1024)
            y = np.random.randint(0, 1024)
            frame[max(0, y-5):min(1024, y+5), 
                  max(0, x-5):min(1024, x+5)] += 500
        
        return frame
    
    def test_frame_characteristics_analysis(self, sample_astronomical_frame):
        """
        Test frame characteristics analysis
        """
        analysis = TileGridGenerator._analyze_frame_characteristics(sample_astronomical_frame)
        
        assert 'mean_intensity' in analysis
        assert 'std_intensity' in analysis
        assert 'gradient_complexity' in analysis
        assert 'star_density' in analysis
        assert 'shape' in analysis
        
        assert analysis['star_density'] > 0
    
    def test_adaptive_tile_size_computation(self, sample_astronomical_frame):
        """
        Test adaptive tile size computation
        """
        frame_analysis = TileGridGenerator._analyze_frame_characteristics(sample_astronomical_frame)
        
        tile_size = TileGridGenerator._compute_adaptive_tile_size(
            sample_astronomical_frame.shape, 
            frame_analysis, 
            min_tile_size=32, 
            max_tile_size=256
        )
        
        assert 32 <= tile_size <= 256
    
    def test_adaptive_overlap_computation(self, sample_astronomical_frame):
        """
        Test adaptive overlap computation
        """
        frame_analysis = TileGridGenerator._analyze_frame_characteristics(sample_astronomical_frame)
        
        overlap = TileGridGenerator._compute_adaptive_overlap(
            frame_analysis, 
            base_overlap=0.25
        )
        
        assert 0 < overlap <= 0.5
    
    def test_adaptive_grid_generation(self, sample_astronomical_frame):
        """
        Test complete adaptive grid generation
        """
        grid_result = TileGridGenerator.generate_adaptive_grid(sample_astronomical_frame)
        
        assert 'tiles' in grid_result
        assert 'tile_size' in grid_result
        assert 'overlap' in grid_result
        assert 'frame_metadata' in grid_result
        assert 'grid_metadata' in grid_result
        
        # Check grid metadata
        grid_metadata = grid_result['grid_metadata']
        assert 'total_tiles' in grid_metadata
        assert 'tile_coordinates' in grid_metadata
        assert 'coverage_percentage' in grid_metadata
        
        # Validate coverage
        assert 0 < grid_metadata['coverage_percentage'] <= 100
    
    def test_multi_channel_grid_generation(self):
        """
        Test grid generation for multiple channels
        """
        np.random.seed(42)
        channels = {
            'R': np.random.normal(100, 20, (1024, 1024)),
            'G': np.random.normal(100, 20, (1024, 1024)),
            'B': np.random.normal(100, 20, (1024, 1024))
        }
        
        multi_channel_grids = generate_multi_channel_grid(channels)
        
        assert set(multi_channel_grids.keys()) == {'R', 'G', 'B'}
        
        for channel, grid_config in multi_channel_grids.items():
            assert 'tiles' in grid_config
            assert len(grid_config['tiles']) > 0