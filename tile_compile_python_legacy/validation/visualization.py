import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.visualization import ZScaleInterval, make_lupton_rgb
import os
from typing import List, Dict, Any, Optional

class ValidationVisualizer:
    def __init__(
        self, 
        output_dir: str = 'validation_results',
        dpi: int = 300
    ):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
            dpi: Resolution of output images
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.dpi = dpi
    
    def visualize_frame_comparison(
        self, 
        original_frame: np.ndarray, 
        processed_frame: np.ndarray, 
        title: Optional[str] = None
    ):
        """
        Compare original and processed frames
        
        Args:
            original_frame: Input astronomical frame
            processed_frame: Processed astronomical frame
            title: Optional visualization title
        """
        # Create ZScale interval for consistent visualization
        zscale = ZScaleInterval()
        
        plt.figure(figsize=(15, 5))
        
        # Original Frame
        plt.subplot(131)
        plt.imshow(zscale(original_frame), cmap='viridis')
        plt.title('Original Frame')
        plt.colorbar(label='Intensity')
        
        # Processed Frame
        plt.subplot(132)
        plt.imshow(zscale(processed_frame), cmap='viridis')
        plt.title('Processed Frame')
        plt.colorbar(label='Intensity')
        
        # Difference Map
        plt.subplot(133)
        diff = np.abs(processed_frame - original_frame)
        plt.imshow(zscale(diff), cmap='hot')
        plt.title('Difference Map')
        plt.colorbar(label='Absolute Difference')
        
        if title:
            plt.suptitle(title)
        
        plt.tight_layout()
        output_path = os.path.join(
            self.output_dir, 
            f'{title.replace(" ", "_")}_comparison.png' if title else 'frame_comparison.png'
        )
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
    
    def create_color_composite(
        self, 
        r_frame: np.ndarray, 
        g_frame: np.ndarray, 
        b_frame: np.ndarray, 
        title: Optional[str] = None
    ):
        """
        Create color composite visualization
        
        Args:
            r_frame: Red channel frame
            g_frame: Green channel frame
            b_frame: Blue channel frame
            title: Optional visualization title
        """
        # Use Lupton et al. color mapping
        rgb_image = make_lupton_rgb(
            r_frame, g_frame, b_frame, 
            Q=10, stretch=0.5
        )
        
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        
        if title:
            plt.title(title)
        
        plt.axis('off')
        output_path = os.path.join(
            self.output_dir, 
            f'{title.replace(" ", "_")}_rgb_composite.png' if title else 'rgb_composite.png'
        )
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_distribution(
        self, 
        metrics: Dict[str, List[float]], 
        title: Optional[str] = None
    ):
        """
        Plot distribution of various metrics
        
        Args:
            metrics: Dictionary of metric lists
            title: Optional plot title
        """
        plt.figure(figsize=(15, 5))
        
        for i, (metric_name, metric_values) in enumerate(metrics.items(), 1):
            plt.subplot(1, len(metrics), i)
            sns.histplot(metric_values, kde=True)
            plt.title(f'{metric_name} Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        
        if title:
            plt.suptitle(title)
        
        plt.tight_layout()
        output_path = os.path.join(
            self.output_dir, 
            f'{title.replace(" ", "_")}_metrics_distribution.png' if title else 'metrics_distribution.png'
        )
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
    
    def plot_performance_summary(
        self, 
        performance_data: Dict[str, Dict[str, float]]
    ):
        """
        Create performance summary visualization
        
        Args:
            performance_data: Performance metrics dictionary
        """
        metrics = ['mean_time', 'std_time', 'registered_frames_count']
        
        plt.figure(figsize=(15, 5))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            values = [data.get(metric, 0) for data in performance_data.values()]
            labels = list(performance_data.keys())
            
            sns.barplot(x=labels, y=values)
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'performance_summary.png')
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()

def main():
    # Example usage demonstrating various visualization capabilities
    np.random.seed(42)
    
    # Create sample frames
    original_frame = np.random.normal(100, 20, (1024, 1024))
    processed_frame = original_frame + np.random.normal(0, 5, original_frame.shape)
    
    visualizer = ValidationVisualizer()
    
    # Frame comparison
    visualizer.visualize_frame_comparison(
        original_frame, 
        processed_frame, 
        title='Registration Comparison'
    )
    
    # Color composite (with dummy channels)
    visualizer.create_color_composite(
        original_frame, 
        processed_frame, 
        processed_frame, 
        title='RGB Astronomical Image'
    )
    
    # Metrics distribution
    metrics = {
        'Registration Quality': np.random.normal(0.9, 0.05, 100),
        'Signal-to-Noise Ratio': np.random.normal(40, 5, 100),
        'Frame Alignment': np.random.normal(0.95, 0.02, 100)
    }
    visualizer.plot_metrics_distribution(
        metrics, 
        title='Reconstruction Metrics'
    )
    
    # Performance summary
    performance_data = {
        'CFA Registration': {
            'mean_time': 0.5,
            'std_time': 0.1,
            'registered_frames_count': 50
        },
        'Alternative Method': {
            'mean_time': 0.7,
            'std_time': 0.2,
            'registered_frames_count': 45
        }
    }
    visualizer.plot_performance_summary(performance_data)

if __name__ == "__main__":
    main()