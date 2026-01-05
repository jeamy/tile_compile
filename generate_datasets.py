import os
import sys
import traceback

def check_and_install_dependencies():
    """
    Check and install required dependencies
    """
    try:
        import subprocess
        import sys

        # List of critical dependencies
        dependencies = [
            'numpy', 
            'astropy', 
            'matplotlib', 
            'imageio'
        ]
        
        missing_deps = []
        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"Missing dependencies: {', '.join(missing_deps)}")
            print("Attempting to install missing packages...")
            
            # Use pip to install missing dependencies
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + 
                                  [dep + '[all]' for dep in missing_deps])
            
            # Verify installation
            for dep in missing_deps:
                __import__(dep)
    
    except Exception as e:
        print(f"Error during dependency check: {e}")
        traceback.print_exc()
        sys.exit(1)

def main():
    try:
        # Check and install dependencies
        check_and_install_dependencies()

        # Now import required modules
        from validation.dataset_generator import SyntheticAstronomicalDataset
        import matplotlib.pyplot as plt
        import imageio

        # Create output directory
        output_dir = 'synthetic_datasets'
        os.makedirs(output_dir, exist_ok=True)

        # Initialize dataset generator
        generator = SyntheticAstronomicalDataset(
            image_size=(2048, 2048),  # Large, high-resolution frames
            star_count=100,            # More stars for complexity
            noise_level=0.05,          # Realistic noise level
            background_level=100       # Typical background intensity
        )

        # Generate datasets with various parameters
        exposure_times = [30.0, 100.0, 300.0]
        seeing_conditions = [0.5, 1.0, 2.0]

        all_datasets = {}

        # Generate multiple datasets
        for exposure in exposure_times:
            for seeing in seeing_conditions:
                # Generate dataset name
                dataset_name = f'exposure_{exposure:.0f}_seeing_{seeing:.1f}'
                
                # Generate frames
                frames = [
                    generator.generate_frame(
                        exposure_time=exposure, 
                        seeing_conditions=seeing
                    ) for _ in range(50)  # 50 frames per configuration
                ]
                
                all_datasets[dataset_name] = frames

        # Save datasets
        generator.save_dataset(all_datasets, output_dir)

        print(f"Generated {len(all_datasets)} synthetic datasets")
        print("Datasets saved in:", output_dir)

        # Optional: Create color composites
        for dataset_name, frames in all_datasets.items():
            if len(frames) >= 3:
                # Create color composite using first three frames
                rgb_composite = generator.generate_color_composite(
                    frames[0], frames[1], frames[2]
                )
                
                # Save color composite
                composite_path = os.path.join(
                    output_dir, 
                    f'{dataset_name}_color_composite.png'
                )
                plt.imsave(composite_path, rgb_composite)

    except Exception as e:
        print("An error occurred during dataset generation:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()