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
    """Performance benchmarking for Methodik v3 pipeline."""
    
    # Methodik v3 phases
    METHODIK_V3_PHASES = [
        "SCAN_INPUT", "REGISTRATION", "CHANNEL_SPLIT", "NORMALIZATION",
        "GLOBAL_METRICS", "TILE_GRID", "LOCAL_METRICS", "TILE_RECONSTRUCTION",
        "STATE_CLUSTERING", "SYNTHETIC_FRAMES", "STACKING", "DONE"
    ]
    
    def __init__(self, output_dir: str = 'benchmark_results'):
        """
        Initialize performance benchmark
        
        Args:
            output_dir: Directory to save benchmark results
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.results = {}
        self.methodik_v3_results = {}
    
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
    
    def run_methodik_v3_benchmark(self, frames: np.ndarray, assumptions: Dict[str, Any]):
        """
        Benchmark all 11 Methodik v3 pipeline phases.
        
        Args:
            frames: Input frames
            assumptions: Methodik v3 assumptions
        """
        frame_count = len(frames)
        reduced_mode = frame_count < assumptions.get('frames_reduced_threshold', 200)
        
        print(f"\nBenchmarking Methodik v3 pipeline ({frame_count} frames, {'Reduced' if reduced_mode else 'Normal'} Mode)...")
        
        phase_times = {}
        
        # Phase 0: SCAN_INPUT (minimal - just validation)
        start = time.time()
        _ = frames.shape
        phase_times['SCAN_INPUT'] = time.time() - start
        
        # Phase 1: REGISTRATION (simulate)
        start = time.time()
        if len(frames) > 0:
            _ = np.mean(frames, axis=0)  # Simplified
        phase_times['REGISTRATION'] = time.time() - start
        
        # Phase 2: CHANNEL_SPLIT (simulate RGB split)
        start = time.time()
        if len(frames) > 0 and frames[0].ndim == 2:
            _ = [frames, frames, frames]  # Simplified
        phase_times['CHANNEL_SPLIT'] = time.time() - start
        
        # Phase 3: NORMALIZATION
        start = time.time()
        if len(frames) > 0:
            medians = [np.median(f) for f in frames]
            _ = np.array(medians)
        phase_times['NORMALIZATION'] = time.time() - start
        
        # Phase 4: GLOBAL_METRICS
        start = time.time()
        if len(frames) > 0:
            for f in frames[:min(10, len(frames))]:  # Sample
                _ = np.mean(f), np.std(f), np.median(f)
        phase_times['GLOBAL_METRICS'] = time.time() - start
        
        # Phase 5: TILE_GRID
        start = time.time()
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            tile_size = 128
            _ = [(i, j) for i in range(0, h, tile_size) for j in range(0, w, tile_size)]
        phase_times['TILE_GRID'] = time.time() - start
        
        # Phase 6: LOCAL_METRICS
        start = time.time()
        if len(frames) > 0:
            # Simulate tile metrics calculation
            for f in frames[:min(5, len(frames))]:
                _ = np.std(f)
        phase_times['LOCAL_METRICS'] = time.time() - start
        
        # Phase 7: TILE_RECONSTRUCTION
        start = time.time()
        if len(frames) > 0:
            _ = np.mean(frames, axis=0)
        phase_times['TILE_RECONSTRUCTION'] = time.time() - start
        
        # Phase 8: STATE_CLUSTERING (skip in reduced mode)
        if reduced_mode and assumptions.get('reduced_mode_skip_clustering', True):
            phase_times['STATE_CLUSTERING'] = 0.0
            print("  STATE_CLUSTERING: skipped (reduced mode)")
        else:
            start = time.time()
            # Simulate clustering
            time.sleep(0.01)
            phase_times['STATE_CLUSTERING'] = time.time() - start
        
        # Phase 9: SYNTHETIC_FRAMES (skip in reduced mode)
        if reduced_mode and assumptions.get('reduced_mode_skip_clustering', True):
            phase_times['SYNTHETIC_FRAMES'] = 0.0
            print("  SYNTHETIC_FRAMES: skipped (reduced mode)")
        else:
            start = time.time()
            # Simulate synthetic frame generation
            time.sleep(0.01)
            phase_times['SYNTHETIC_FRAMES'] = time.time() - start
        
        # Phase 10: STACKING
        start = time.time()
        if len(frames) > 0:
            _ = np.mean(frames, axis=0)
        phase_times['STACKING'] = time.time() - start
        
        # Phase 11: DONE (minimal)
        phase_times['DONE'] = 0.0
        
        # Store results
        total_time = sum(phase_times.values())
        self.methodik_v3_results = {
            'frame_count': frame_count,
            'reduced_mode': reduced_mode,
            'phase_times': phase_times,
            'total_time': total_time,
            'phases_executed': sum(1 for t in phase_times.values() if t > 0),
            'phases_skipped': sum(1 for t in phase_times.values() if t == 0)
        }
        
        # Print summary
        print(f"\nPhase Execution Times:")
        for phase, t in phase_times.items():
            if t > 0:
                pct = (t / total_time * 100) if total_time > 0 else 0
                print(f"  {phase:20s}: {t:6.3f}s ({pct:5.1f}%)")
            else:
                print(f"  {phase:20s}: skipped")
        print(f"  {'TOTAL':20s}: {total_time:6.3f}s")
    
    def compare_reduced_vs_normal_mode(self, frames: np.ndarray, assumptions: Dict[str, Any]):
        """
        Compare performance between Reduced Mode and Normal Mode.
        
        Args:
            frames: Input frames
            assumptions: Methodik v3 assumptions
        """
        threshold = assumptions.get('frames_reduced_threshold', 200)
        
        if len(frames) < threshold:
            print(f"Cannot compare modes: only {len(frames)} frames available")
            return
        
        print(f"\nComparing Reduced Mode vs Normal Mode...")
        
        # Simulate reduced mode (subset of frames)
        reduced_frames = frames[:threshold - 1]
        
        # Benchmark reduced mode
        print(f"\n1. Reduced Mode ({len(reduced_frames)} frames):")
        self.run_methodik_v3_benchmark(reduced_frames, assumptions)
        reduced_results = dict(self.methodik_v3_results)
        
        # Benchmark normal mode
        print(f"\n2. Normal Mode ({len(frames)} frames):")
        assumptions_normal = dict(assumptions)
        assumptions_normal['frames_reduced_threshold'] = 0  # Force normal mode
        self.run_methodik_v3_benchmark(frames, assumptions_normal)
        normal_results = dict(self.methodik_v3_results)
        
        # Comparison
        comparison = {
            'reduced_mode': reduced_results,
            'normal_mode': normal_results,
            'speedup': normal_results['total_time'] / reduced_results['total_time'] if reduced_results['total_time'] > 0 else 0,
            'phases_saved': normal_results['phases_executed'] - reduced_results['phases_executed']
        }
        
        print(f"\n{'='*70}")
        print("MODE COMPARISON")
        print(f"{'='*70}")
        print(f"Reduced Mode Total: {reduced_results['total_time']:.3f}s")
        print(f"Normal Mode Total:  {normal_results['total_time']:.3f}s")
        print(f"Speedup Factor:     {comparison['speedup']:.2f}x")
        print(f"Phases Saved:       {comparison['phases_saved']}")
        print(f"{'='*70}\n")
        
        # Save comparison
        comparison_path = os.path.join(self.output_dir, 'reduced_mode_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved: {comparison_path}")
    
    def generate_report(self):
        """
        Generate comprehensive benchmark report
        """
        report_path = os.path.join(self.output_dir, 'performance_report.json')
        
        report = {
            'legacy_results': self.results,
            'methodik_v3_results': self.methodik_v3_results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Performance report saved: {report_path}")
            
            for component, metrics in self.results.items():
                f.write(f"## {component.capitalize()} Performance\n")
                for key, value in metrics.items():
                    f.write(f"- **{key}**: {value}\n")
    
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