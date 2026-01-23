"""Methodik v3 Validation Runner

Runs comprehensive validation tests for the Methodik v3 pipeline:
- Performance benchmarks for all 11 phases
- Assumptions compliance testing
- Reduced Mode vs Normal Mode comparison
- Dataset quality validation

Usage:
    python run_validation.py [--dataset-dir PATH] [--output-dir PATH]
"""

import os
import sys
import glob
import traceback
import argparse
from pathlib import Path

def check_dependencies():
    """
    Check and import required dependencies
    """
    dependencies = [
        'numpy', 
        'imageio', 
        'matplotlib', 
        'sklearn', 
        'astropy',
        'yaml'
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Please install these packages using pip.")
        sys.exit(1)

def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='Methodik v3 Validation Runner')
        parser.add_argument('--dataset-dir', type=str, default='validation_datasets/M45',
                          help='Path to validation dataset')
        parser.add_argument('--output-dir', type=str, default='validation_results',
                          help='Output directory for results')
        parser.add_argument('--config', type=str, default='tile_compile.yaml',
                          help='Path to tile_compile.yaml config')
        args = parser.parse_args()
        
        # Check dependencies first
        check_dependencies()

        import numpy as np
        import yaml
        from imageio import imread
        from astropy.io import fits

        from validation.performance_benchmark import PerformanceBenchmark
        from validation.comparative_analysis import ComparativeAnalysis
        from validation.visualization import ValidationVisualizer
        from validation.methodik_v3_compliance import MetodikV3ComplianceValidator
        from validation.assumptions_validator import AssumptionsValidator

        def load_datasets(dataset_dir='validation_datasets/M45'):
            """
            Load validation datasets (FITS or TIFF)
            """
            dataset_dir = Path(dataset_dir)
            
            # Try FITS first
            fits_paths = list(dataset_dir.glob('*.fits')) + list(dataset_dir.glob('*.fit'))
            if fits_paths:
                frames = []
                for path in fits_paths:
                    with fits.open(path) as hdul:
                        frames.append(hdul[0].data)
                return {'M45': np.array(frames)}
            
            # Fallback to TIFF
            dataset_paths = glob.glob(os.path.join(str(dataset_dir), '**', '*.tiff'), recursive=True)
            datasets = {}
            for path in dataset_paths:
                dataset_name = os.path.basename(os.path.dirname(path))
                if dataset_name not in datasets:
                    datasets[dataset_name] = []
                datasets[dataset_name].append(imread(path))
            
            return datasets
        
        def load_assumptions(config_path='tile_compile.yaml'):
            """Load Methodik v3 assumptions from config."""
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                return cfg.get('assumptions', {})
            except Exception as e:
                print(f"Warning: Could not load assumptions from {config_path}: {e}")
                return {}

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Load datasets and assumptions
        datasets = load_datasets(args.dataset_dir)
        assumptions = load_assumptions(args.config)
        
        print("="*70)
        print("METHODIK V3 VALIDATION SUITE")
        print("="*70)
        print(f"Dataset directory: {args.dataset_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Loaded {len(datasets)} dataset(s)")
        print(f"Assumptions loaded: {len(assumptions)} parameters")
        print("="*70 + "\n")

        # Methodik v3 Compliance Validation
        print("\n" + "="*70)
        print("1. METHODIK V3 COMPLIANCE VALIDATION")
        print("="*70)
        for dataset_name, frames in datasets.items():
            print(f"\nValidating dataset: {dataset_name}")
            frames_array = np.array(frames)
            
            compliance_validator = MetodikV3ComplianceValidator(
                assumptions=assumptions,
                output_dir=args.output_dir
            )
            compliance_report = compliance_validator.validate_dataset(frames_array, dataset_name)
            compliance_validator.generate_report(compliance_report)
        
        # Assumptions Validation
        print("\n" + "="*70)
        print("2. ASSUMPTIONS VALIDATION")
        print("="*70)
        for dataset_name, frames in datasets.items():
            print(f"\nValidating assumptions for: {dataset_name}")
            frames_array = np.array(frames)
            
            assumptions_validator = AssumptionsValidator(
                assumptions=assumptions,
                output_dir=args.output_dir
            )
            assumptions_report = assumptions_validator.validate(frames_array, dataset_name)
            assumptions_validator.generate_report(assumptions_report)
        
        # Performance Benchmark (Methodik v3 phases)
        print("\n" + "="*70)
        print("3. PERFORMANCE BENCHMARK (11 METHODIK V3 PHASES)")
        print("="*70)
        for dataset_name, frames in datasets.items():
            print(f"\nBenchmarking dataset: {dataset_name}")
            frames_array = np.array(frames)
            
            benchmark = PerformanceBenchmark(output_dir=args.output_dir)
            benchmark.run_methodik_v3_benchmark(frames_array, assumptions)
            benchmark.generate_report()

        # Comparative Analysis (Reduced Mode vs Normal Mode)
        print("\n" + "="*70)
        print("4. REDUCED MODE COMPARISON")
        print("="*70)
        for dataset_name, frames in datasets.items():
            frames_array = np.array(frames)
            if len(frames_array) >= assumptions.get('frames_reduced_threshold', 200):
                print(f"\nComparing modes for: {dataset_name}")
                benchmark = PerformanceBenchmark(output_dir=args.output_dir)
                benchmark.compare_reduced_vs_normal_mode(frames_array, assumptions)
        
        # Comparative Analysis
        print("\n" + "="*70)
        print("5. COMPARATIVE ANALYSIS")
        print("="*70)
        analysis = ComparativeAnalysis(
            reference_dir=args.dataset_dir, 
            output_dir=args.output_dir
        )

        # Visualize Results
        print("\n" + "="*70)
        print("6. VISUALIZATION")
        print("="*70)
        visualizer = ValidationVisualizer(output_dir=args.output_dir)

        # Example visualization (you can expand this)
        for dataset_name, frames in datasets.items():
            if len(frames) >= 3:
                # Frame comparison
                visualizer.visualize_frame_comparison(
                    frames[0], frames[-1], 
                    title=f'Frame Comparison - {dataset_name}'
                )

                # Color composite
                visualizer.create_color_composite(
                    frames[0], frames[1], frames[2],
                    title=f'Color Composite - {dataset_name}'
                )

        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70)
        print(f"Results saved in: {args.output_dir}/")
        print("\nGenerated files:")
        print("  - methodik_v3_compliance_report.json")
        print("  - assumptions_validation_report.json")
        print("  - performance_benchmark_report.json")
        print("  - reduced_mode_comparison.json")
        print("  - visualizations/*.png")
        print("="*70)

    except Exception as e:
        print("An error occurred during validation:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()