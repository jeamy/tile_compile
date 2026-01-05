import numpy as np
import time
import json
import os
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor

from tile_compile_backend.registration import CFARegistration
from tile_compile_backend.linearity import validate_frames_linearity
from tile_compile_backend.metrics import compute_channel_metrics
from tile_compile_backend.tile_grid import generate_multi_channel_grid
from tile_compile_backend.reconstruction import reconstruct_channels

class PerformanceBenchmark:
    def __init__(self, output_dir: str = 'benchmark_results'):
        """
        Initialize performance benchmark
        
        Args:
            output_dir: Directory to save benchmark results
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.results = {}
    
    def benchmark_registration(
        self, 
        frames: List[np.ndarray], 
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark CFA registration performance
        
        Args:
            frames: Input frames
            iterations: Number of benchmark iterations
        
        Returns:
            Registration performance metrics
        """
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            start_time = time.time()
            result = CFARegistration.register_cfa_frames(frames)
            end_time = time.time()
            
            times.append(end_time - start_time)
            # TODO: Add actual memory tracking
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'total_frames': len(frames),
            'memory_usage': memory_usage
        }
    
    def benchmark_linearity_validation(
        self, 
        frames: List[np.ndarray], 
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark linearity validation performance
        
        Args:
            frames: Input frames
            iterations: Number of benchmark iterations
        
        Returns:
            Linearity validation performance metrics
        """
        times = []
        rejected_rates = []
        
        for _ in range(iterations):
            start_time = time.time()
            result = validate_frames_linearity(np.array(frames))
            end_time = time.time()
            
            times.append(end_time - start_time)
            rejected_rates.append(1 - result['overall_linearity'])
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'mean_rejection_rate': np.mean(rejected_rates),
            'total_frames': len(frames)
        }
    
    def benchmark_metrics_computation(
        self, 
        channels: Dict[str, List[np.ndarray]], 
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark channel metrics computation
        
        Args:
            channels: Channel frames
            iterations: Number of benchmark iterations
        
        Returns:
            Metrics computation performance metrics
        """
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            metrics = compute_channel_metrics(channels)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'channels': list(channels.keys())
        }
    
    def benchmark_tile_grid_generation(
        self, 
        channels: Dict[str, np.ndarray], 
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark tile grid generation
        
        Args:
            channels: Channel frames
            iterations: Number of benchmark iterations
        
        Returns:
            Tile grid generation performance metrics
        """
        times = []
        grid_sizes = []
        
        for _ in range(iterations):
            start_time = time.time()
            grid_result = generate_multi_channel_grid(channels)
            end_time = time.time()
            
            times.append(end_time - start_time)
            grid_sizes.append({
                channel: len(grid['tiles']) 
                for channel, grid in grid_result.items()
            })
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'mean_grid_sizes': {
                channel: np.mean([size[channel] for size in grid_sizes])
                for channel in grid_sizes[0].keys()
            }
        }
    
    def run_comprehensive_benchmark(
        self, 
        frames: List[np.ndarray], 
        parallel: bool = True
    ):
        """
        Run comprehensive performance benchmark
        
        Args:
            frames: Input frames
            parallel: Use parallel processing
        """
        if parallel:
            with ProcessPoolExecutor() as executor:
                # Parallel benchmark execution
                futures = {
                    'registration': executor.submit(
                        self.benchmark_registration, frames
                    ),
                    'linearity': executor.submit(
                        self.benchmark_linearity_validation, frames
                    )
                }
                
                # Collect results
                self.results = {
                    key: future.result() 
                    for key, future in futures.items()
                }
        else:
            # Sequential benchmark
            self.results = {
                'registration': self.benchmark_registration(frames),
                'linearity': self.benchmark_linearity_validation(frames)
            }
        
        # Save results
        with open(
            os.path.join(self.output_dir, 'benchmark_results.json'), 
            'w'
        ) as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self):
        """
        Generate detailed performance report
        """
        report_path = os.path.join(self.output_dir, 'performance_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Performance Benchmark Report\n\n")
            
            for component, metrics in self.results.items():
                f.write(f"## {component.capitalize()} Performance\n")
                for key, value in metrics.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")

def main():
    # Example usage
    import glob
    from imageio import imread
    
    # Load frames from a directory
    frame_paths = glob.glob('synthetic_datasets/**/*.tiff')
    frames = [imread(path) for path in frame_paths[:50]]  # Limit to first 50
    
    benchmark = PerformanceBenchmark()
    benchmark.run_comprehensive_benchmark(frames)
    benchmark.generate_report()

if __name__ == "__main__":
    main()