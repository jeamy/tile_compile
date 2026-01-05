import os
import sys
import glob
import traceback

def check_dependencies():
    """
    Check and import required dependencies
    """
    dependencies = [
        'numpy', 
        'imageio', 
        'matplotlib', 
        'sklearn', 
        'astropy'
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
        # Check dependencies first
        check_dependencies()

        import numpy as np
        from imageio import imread

        from validation.performance_benchmark import PerformanceBenchmark
        from validation.comparative_analysis import ComparativeAnalysis
        from validation.visualization import ValidationVisualizer

        def load_datasets(dataset_dir='synthetic_datasets'):
            """
            Load generated synthetic datasets
            """
            dataset_paths = glob.glob(os.path.join(dataset_dir, '**', '*.tiff'), recursive=True)
            
            # Group frames by dataset
            datasets = {}
            for path in dataset_paths:
                dataset_name = os.path.basename(os.path.dirname(path))
                if dataset_name not in datasets:
                    datasets[dataset_name] = []
                datasets[dataset_name].append(imread(path))
            
            return datasets

        # Ensure output directory exists
        os.makedirs('validation_results', exist_ok=True)

        # Load datasets
        datasets = load_datasets()

        # Performance Benchmark
        print("Running Performance Benchmark...")
        for dataset_name, frames in datasets.items():
            print(f"Benchmarking dataset: {dataset_name}")
            
            # Convert to numpy array
            frames = np.array(frames)
            
            benchmark = PerformanceBenchmark(output_dir='validation_results')
            benchmark.run_comprehensive_benchmark(frames)
            benchmark.generate_report()

        # Comparative Analysis
        print("Running Comparative Analysis...")
        analysis = ComparativeAnalysis(
            reference_dir='synthetic_datasets', 
            output_dir='validation_results'
        )

        # Visualize Results
        print("Creating Visualizations...")
        visualizer = ValidationVisualizer(output_dir='validation_results')

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

        print("Validation complete. Results saved in validation_results/")

    except Exception as e:
        print("An error occurred during validation:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()