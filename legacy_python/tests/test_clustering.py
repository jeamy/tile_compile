import numpy as np
import pytest
from tile_compile_backend.clustering import StateClustering, cluster_channels

class TestStateClustering:
    def setup_method(self):
        np.random.seed(42)
        
        # Simulate frames with some structure
        self.frames = []
        for i in range(100):
            base_frame = np.random.normal(0, 1, (100, 100))
            
            # Add some structured variation
            if i < 30:
                base_frame += 0.5  # Group 1
            elif i < 60:
                base_frame -= 0.5  # Group 2
            else:
                base_frame *= 1.5  # Group 3
            
            self.frames.append(base_frame)
        
        # Mock metrics
        self.metrics = {
            'global': {
                'G_f_c': np.random.random(100),
                'background_level': np.random.random(100),
                'noise_level': np.random.random(100)
            },
            'tiles': {
                'Q_local': [np.random.random((10, 10)) for _ in range(100)]
            }
        }

    def test_state_vector_computation(self):
        state_vectors = StateClustering._compute_state_vectors(
            self.frames, 
            self.metrics
        )
        
        assert state_vectors.shape[0] == len(self.frames)
        assert state_vectors.shape[1] == 5  # 5 state vector components

    def test_clustering(self):
        clustering_result = StateClustering.cluster_frames(
            self.frames, 
            self.metrics, 
            {
                'min_clusters': 10, 
                'max_clusters': 30
            }
        )
        
        assert 'cluster_labels' in clustering_result
        assert 'cluster_centers' in clustering_result
        assert 'cluster_stats' in clustering_result
        assert 'n_clusters' in clustering_result
        assert 'silhouette_score' in clustering_result
        
        assert len(clustering_result['cluster_labels']) == len(self.frames)
        assert len(set(clustering_result['cluster_labels'])) <= 30
        assert len(set(clustering_result['cluster_labels'])) >= 10

    def test_cluster_statistics(self):
        state_vectors = StateClustering._compute_state_vectors(
            self.frames, 
            self.metrics
        )
        cluster_labels = np.random.randint(0, 5, len(self.frames))
        
        cluster_stats = StateClustering._compute_cluster_statistics(
            state_vectors, 
            cluster_labels
        )
        
        for cluster, stats in cluster_stats.items():
            assert 'size' in stats
            assert 'mean_vector' in stats
            assert 'std_vector' in stats
            assert 'min_vector' in stats
            assert 'max_vector' in stats

    def test_channel_clustering(self):
        channels = {
            'R': self.frames[:50],
            'G': self.frames[50:75],
            'B': self.frames[75:]
        }
        
        channel_clustering = cluster_channels(
            channels, 
            {
                'R': self.metrics, 
                'G': self.metrics, 
                'B': self.metrics
            }
        )
        
        assert set(channel_clustering.keys()) == {'R', 'G', 'B'}
        
        for channel, clustering in channel_clustering.items():
            assert 'cluster_labels' in clustering
            assert 'n_clusters' in clustering