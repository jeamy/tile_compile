import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from typing import List, Dict, Any
import seaborn as sns

class M45DatasetAnalyzer:
    def __init__(self, dataset_path: str = '/home/lux/tile-compile-cache/M45_lights'):
        """
        Initialize dataset analyzer
        
        Args:
            dataset_path: Path to FITS files
        """
        self.dataset_path = dataset_path
        self.fits_files = [f for f in os.listdir(dataset_path) if f.endswith('.fits')]
    
    def collect_frame_statistics(self) -> List[Dict[str, Any]]:
        """
        Collect comprehensive statistics for each frame
        
        Returns:
            List of frame statistics dictionaries
        """
        frame_stats = []
        
        for filename in self.fits_files:
            filepath = os.path.join(self.dataset_path, filename)
            
            try:
                with fits.open(filepath) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    
                    # Compute advanced statistics
                    stats = {
                        'filename': filename,
                        'shape': data.shape,
                        'dtype': str(data.dtype),
                        'mean': float(np.mean(data)),
                        'median': float(np.median(data)),
                        'std': float(np.std(data)),
                        'min': float(np.min(data)),
                        'max': float(np.max(data)),
                        'dynamic_range': float(np.max(data) / (np.min(data) + 1e-8)),
                        # Additional header information
                        'exposure_time': header.get('EXPTIME', 0),
                        'date_obs': header.get('DATE-OBS', 'Unknown')
                    }
                    frame_stats.append(stats)
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        return frame_stats
    
    def visualize_dataset_characteristics(self, frame_stats: List[Dict[str, Any]]):
        """
        Create comprehensive visualizations of dataset characteristics
        
        Args:
            frame_stats: List of frame statistics
        """
        plt.figure(figsize=(20, 12))
        
        # 1. Mean Intensity Distribution
        plt.subplot(2, 3, 1)
        means = [stat['mean'] for stat in frame_stats]
        sns.histplot(means, kde=True)
        plt.title('Mean Intensity Distribution')
        plt.xlabel('Mean Intensity')
        plt.ylabel('Frequency')
        
        # 2. Standard Deviation Distribution
        plt.subplot(2, 3, 2)
        stds = [stat['std'] for stat in frame_stats]
        sns.histplot(stds, kde=True)
        plt.title('Standard Deviation Distribution')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')
        
        # 3. Dynamic Range Distribution
        plt.subplot(2, 3, 3)
        dynamic_ranges = [stat['dynamic_range'] for stat in frame_stats]
        sns.histplot(dynamic_ranges, kde=True)
        plt.title('Dynamic Range Distribution')
        plt.xlabel('Dynamic Range')
        plt.ylabel('Frequency')
        
        # 4. Boxplot of Key Metrics
        plt.subplot(2, 3, 4)
        metrics_data = {
            'Mean': means,
            'Std': stds,
            'Dynamic Range': dynamic_ranges
        }
        plt.boxplot([metrics_data[key] for key in metrics_data.keys()])
        plt.title('Boxplot of Frame Metrics')
        plt.xticks(range(1, len(metrics_data) + 1), list(metrics_data.keys()), rotation=45)
        plt.ylabel('Value')
        
        # 5. Scatter Plot: Mean vs Standard Deviation
        plt.subplot(2, 3, 5)
        plt.scatter(means, stds, alpha=0.6)
        plt.title('Mean vs Standard Deviation')
        plt.xlabel('Mean Intensity')
        plt.ylabel('Standard Deviation')
        
        # 6. Frame Shape Consistency
        plt.subplot(2, 3, 6)
        shapes = [str(stat['shape']) for stat in frame_stats]
        shape_counts = {}
        for shape in shapes:
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        plt.bar(range(len(shape_counts)), list(shape_counts.values()))
        plt.title('Frame Shape Distribution')
        plt.xlabel('Frame Shape')
        plt.ylabel('Count')
        plt.xticks(range(len(shape_counts)), list(shape_counts.keys()), rotation=45)
        
        plt.tight_layout()
        plt.savefig('m45_dataset_analysis.png', dpi=300)
        
        # Print summary statistics
        self.print_dataset_summary(frame_stats)
    
    def print_dataset_summary(self, frame_stats: List[Dict[str, Any]]):
        """
        Print comprehensive dataset summary
        
        Args:
            frame_stats: List of frame statistics
        """
        print("\n--- M45 Dataset Analysis Summary ---")
        print(f"Total Frames: {len(frame_stats)}")
        
        # Aggregate statistics
        summary = {
            'mean': np.mean([stat['mean'] for stat in frame_stats]),
            'median_mean': np.median([stat['mean'] for stat in frame_stats]),
            'std_mean': np.std([stat['mean'] for stat in frame_stats]),
            'median_std': np.median([stat['std'] for stat in frame_stats]),
            'median_dynamic_range': np.median([stat['dynamic_range'] for stat in frame_stats])
        }
        
        print("\nAggregate Statistics:")
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        
        # Frame shape analysis
        shapes = [stat['shape'] for stat in frame_stats]
        unique_shapes = set(shapes)
        print(f"\nUnique Frame Shapes: {unique_shapes}")
        
        return summary

def main():
    # Initialize analyzer
    analyzer = M45DatasetAnalyzer()
    
    # Collect frame statistics
    frame_stats = analyzer.collect_frame_statistics()
    
    # Visualize dataset characteristics
    analyzer.visualize_dataset_characteristics(frame_stats)

if __name__ == '__main__':
    main()