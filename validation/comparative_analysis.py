import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from tile_compile_backend.registration import CFARegistration
import imageio
import os
from typing import List, Dict, Any

class ComparativeAnalysis:
    def __init__(self, reference_dir: str, output_dir: str = 'comparative_results'):
        """
        Initialize comparative analysis
        
        Args:
            reference_dir: Directory with reference images
            output_dir: Directory to save analysis results
        """
        os.makedirs(output_dir, exist_ok=True)
        self.reference_dir = reference_dir
        self.output_dir = output_dir
        
        self.reference_images = self._load_images(reference_dir)
    
    def _load_images(self, directory: str) -> List[np.ndarray]:
        """
        Load images from a directory
        
        Args:
            directory: Image directory
        
        Returns:
            List of loaded images
        """
        image_paths = [
            os.path.join(directory, f) 
            for f in os.listdir(directory) 
            if f.endswith(('.tiff', '.fit', '.fits'))
        ]
        return [imageio.imread(path) for path in image_paths]
    
    def compare_registration_methods(
        self, 
        input_frames: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Compare different registration methods
        
        Args:
            input_frames: Frames to register
        
        Returns:
            Comparative registration metrics
        """
        methods = {
            'CFA_registration': CFARegistration.register_cfa_frames,
            # Add more registration methods here
        }
        
        results = {}
        
        for method_name, registration_func in methods.items():
            start_time = time.time()
            registered_result = registration_func(input_frames)
            registration_time = time.time() - start_time
            
            registered_frames = registered_result['registered_frames']
            
            # Compute metrics
            results[method_name] = {
                'registration_time': registration_time,
                'registered_frames_count': len(registered_frames),
                'quality_metrics': self._compute_registration_quality(
                    input_frames, 
                    registered_frames
                )
            }
        
        return results
    
    def _compute_registration_quality(
        self, 
        original_frames: List[np.ndarray], 
        registered_frames: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute registration quality metrics
        
        Args:
            original_frames: Original input frames
            registered_frames: Registered frames
        
        Returns:
            Registration quality metrics
        """
        # Ensure frames are comparable
        min_len = min(len(original_frames), len(registered_frames))
        original_frames = original_frames[:min_len]
        registered_frames = registered_frames[:min_len]
        
        ssim_scores = [
            ssim(orig, reg, data_range=orig.max() - orig.min()) 
            for orig, reg in zip(original_frames, registered_frames)
        ]
        
        psnr_scores = [
            psnr(orig, reg) 
            for orig, reg in zip(original_frames, registered_frames)
        ]
        
        return {
            'mean_ssim': np.mean(ssim_scores),
            'std_ssim': np.std(ssim_scores),
            'mean_psnr': np.mean(psnr_scores),
            'std_psnr': np.std(psnr_scores)
        }
    
    def compare_reconstruction_methods(
        self, 
        registered_frames: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Compare different reconstruction methods
        
        Args:
            registered_frames: Registered input frames
        
        Returns:
            Reconstruction method comparisons
        """
        # Placeholder for future reconstruction methods
        methods = {
            'tile_based_reconstruction': None,  # Add actual reconstruction methods
        }
        
        results = {}
        
        for method_name, reconstruction_func in methods.items():
            # Implement reconstruction and comparison logic
            pass
        
        return results
    
    def visualize_comparative_results(
        self, 
        comparison_results: Dict[str, Any]
    ):
        """
        Create visualizations of comparative results
        
        Args:
            comparison_results: Comparison metrics
        """
        # Create comparative plots
        plt.figure(figsize=(15, 5))
        
        # SSIM Comparison
        plt.subplot(131)
        ssim_data = [
            results['quality_metrics']['mean_ssim'] 
            for results in comparison_results.values()
        ]
        sns.barplot(
            x=list(comparison_results.keys()), 
            y=ssim_data
        )
        plt.title('SSIM Comparison')
        plt.xticks(rotation=45)
        
        # PSNR Comparison
        plt.subplot(132)
        psnr_data = [
            results['quality_metrics']['mean_psnr'] 
            for results in comparison_results.values()
        ]
        sns.barplot(
            x=list(comparison_results.keys()), 
            y=psnr_data
        )
        plt.title('PSNR Comparison')
        plt.xticks(rotation=45)
        
        # Registration Time Comparison
        plt.subplot(133)
        time_data = [
            results['registration_time'] 
            for results in comparison_results.values()
        ]
        sns.barplot(
            x=list(comparison_results.keys()), 
            y=time_data
        )
        plt.title('Registration Time')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparative_analysis.png'))
    
    def generate_comparative_report(
        self, 
        comparison_results: Dict[str, Any]
    ):
        """
        Generate a markdown report of comparative analysis
        
        Args:
            comparison_results: Comparison metrics
        """
        report_path = os.path.join(self.output_dir, 'comparative_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Comparative Analysis Report\n\n")
            
            for method, results in comparison_results.items():
                f.write(f"## {method}\n")
                f.write(f"- **Registration Time**: {results['registration_time']:.4f} seconds\n")
                f.write(f"- **Registered Frames**: {results['registered_frames_count']}\n")
                
                # Quality Metrics
                f.write("### Quality Metrics\n")
                for metric, value in results['quality_metrics'].items():
                    f.write(f"- **{metric}**: {value}\n")
                f.write("\n")

def main():
    # Example usage
    reference_dir = 'reference_datasets'
    analysis = ComparativeAnalysis(reference_dir)
    
    # Load input frames
    input_frames = analysis._load_images('input_frames')
    
    # Run comparative analysis
    comparison_results = analysis.compare_registration_methods(input_frames)
    
    # Visualize and report results
    analysis.visualize_comparative_results(comparison_results)
    analysis.generate_comparative_report(comparison_results)

if __name__ == "__main__":
    main()