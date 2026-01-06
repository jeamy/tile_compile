"""
M45 Dataset Analyzer

Analyzes M45 (Pleiades) FITS frames for quality assessment:
- Frame quality metrics (mean, std, SNR, dynamic range)
- Statistical analysis across dataset
- Visualization of quality distributions
- Frame selection recommendations

Used for pre-processing analysis before running the Methodik v3 pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from typing import List, Dict, Any
import seaborn as sns

class M45DatasetAnalyzer:
    def __init__(self, dataset_path: str = '/home/lux/tile-compile-cache/M45_lights'):
        """
        Initialize dataset analyzer with focus on frame quality assessment
        
        Args:
            dataset_path: Path to FITS files
        """
        self.dataset_path = dataset_path
        self.fits_files = [f for f in os.listdir(dataset_path) if f.endswith('.fits')]
    
    def assess_frame_quality(self, data: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive quality assessment for a single frame
        Following Methodik v3 principles
        
        Args:
            data: Numpy array of frame data
        
        Returns:
            Dictionary of quality metrics
        """
        # 1. Background level assessment
        background_level = np.median(data)
        
        # 2. Noise level estimation
        noise_level = np.std(data)
        
        # 3. Signal-to-Noise Ratio
        signal_max = np.max(data)
        snr = signal_max / (noise_level + 1e-8)
        
        # 4. Dynamic Range
        dynamic_range = signal_max / (np.min(data) + 1e-8)
        
        # 5. Gradient Energy (simplified)
        gy, gx = np.gradient(data)
        gradient_energy = np.mean(np.sqrt(gx**2 + gy**2))
        
        # 6. Comprehensive Quality Score
        # Lower scores indicate poorer quality
        quality_score = (
            (1 / (noise_level + 1)) *  # Prefer low noise
            (snr / 100) *  # Higher SNR is better
            (np.log(dynamic_range + 1) / 5) *  # Logarithmic scaling of dynamic range
            (1 / (1 + np.abs(background_level - np.mean(data))))  # Prefer balanced background
        )
        
        return {
            'background_level': float(background_level),
            'noise_level': float(noise_level),
            'signal_to_noise_ratio': float(snr),
            'dynamic_range': float(dynamic_range),
            'gradient_energy': float(gradient_energy),
            'quality_score': float(quality_score)
        }
    
    def collect_frame_statistics(self) -> List[Dict[str, Any]]:
        """
        Collect comprehensive statistics and quality metrics for each frame
        
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
                    
                    # Compute quality metrics
                    quality_metrics = self.assess_frame_quality(data)
                    
                    # Combine with basic frame information
                    stats = {
                        'filename': filename,
                        'shape': data.shape,
                        'dtype': str(data.dtype),
                        'mean': float(np.mean(data)),
                        'median': float(np.median(data)),
                        'std': float(np.std(data)),
                        'min': float(np.min(data)),
                        'max': float(np.max(data)),
                        **quality_metrics
                    }
                    frame_stats.append(stats)
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        return frame_stats
    
    def visualize_quality_metrics(self, frame_stats: List[Dict[str, Any]]):
        """
        Create comprehensive visualizations of frame quality metrics
        
        Args:
            frame_stats: List of frame statistics
        """
        plt.figure(figsize=(20, 15))
        
        # Extract metrics
        quality_scores = [stat['quality_score'] for stat in frame_stats]
        noise_levels = [stat['noise_level'] for stat in frame_stats]
        snr_values = [stat['signal_to_noise_ratio'] for stat in frame_stats]
        dynamic_ranges = [stat['dynamic_range'] for stat in frame_stats]
        gradient_energies = [stat['gradient_energy'] for stat in frame_stats]
        
        # 1. Quality Score Distribution
        plt.subplot(2, 3, 1)
        sns.histplot(quality_scores, kde=True)
        plt.title('Frame Quality Score Distribution')
        plt.xlabel('Quality Score')
        plt.ylabel('Frequency')
        plt.axvline(np.median(quality_scores), color='r', linestyle='--', label='Median')
        plt.legend()
        
        # 2. Noise Level Distribution
        plt.subplot(2, 3, 2)
        sns.histplot(noise_levels, kde=True)
        plt.title('Noise Level Distribution')
        plt.xlabel('Noise Level')
        plt.ylabel('Frequency')
        plt.axvline(np.median(noise_levels), color='r', linestyle='--', label='Median')
        plt.legend()
        
        # 3. Signal-to-Noise Ratio Distribution
        plt.subplot(2, 3, 3)
        sns.histplot(snr_values, kde=True)
        plt.title('Signal-to-Noise Ratio Distribution')
        plt.xlabel('SNR')
        plt.ylabel('Frequency')
        plt.axvline(np.median(snr_values), color='r', linestyle='--', label='Median')
        plt.legend()
        
        # 4. Dynamic Range Distribution
        plt.subplot(2, 3, 4)
        sns.histplot(dynamic_ranges, kde=True)
        plt.title('Dynamic Range Distribution')
        plt.xlabel('Dynamic Range')
        plt.ylabel('Frequency')
        plt.axvline(np.median(dynamic_ranges), color='r', linestyle='--', label='Median')
        plt.legend()
        
        # 5. Gradient Energy Distribution
        plt.subplot(2, 3, 5)
        sns.histplot(gradient_energies, kde=True)
        plt.title('Gradient Energy Distribution')
        plt.xlabel('Gradient Energy')
        plt.ylabel('Frequency')
        plt.axvline(np.median(gradient_energies), color='r', linestyle='--', label='Median')
        plt.legend()
        
        # 6. Quality Score vs Noise Level
        plt.subplot(2, 3, 6)
        plt.scatter(noise_levels, quality_scores, alpha=0.6)
        plt.title('Quality Score vs Noise Level')
        plt.xlabel('Noise Level')
        plt.ylabel('Quality Score')
        
        plt.tight_layout()
        plt.savefig('m45_dataset_quality_analysis.png', dpi=300)
        
        # Print quality assessment summary
        self.print_quality_summary(frame_stats)
    
    def print_quality_summary(self, frame_stats: List[Dict[str, Any]]):
        """
        Print comprehensive quality summary
        
        Args:
            frame_stats: List of frame statistics
        """
        print("\n--- M45 Dataset Quality Assessment ---")
        print(f"Total Frames: {len(frame_stats)}")
        
        # Quality metrics
        quality_metrics = {
            'quality_score': [stat['quality_score'] for stat in frame_stats],
            'noise_level': [stat['noise_level'] for stat in frame_stats],
            'signal_to_noise_ratio': [stat['signal_to_noise_ratio'] for stat in frame_stats],
            'dynamic_range': [stat['dynamic_range'] for stat in frame_stats]
        }
        
        print("\nQuality Metrics Summary:")
        for metric, values in quality_metrics.items():
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Median: {np.median(values):.4f}")
            print(f"  Mean:   {np.mean(values):.4f}")
            print(f"  Std:    {np.std(values):.4f}")
        
        # Identify problematic frames
        problematic_frames = [
            stat for stat in frame_stats 
            if stat['quality_score'] < np.median(quality_metrics['quality_score'])
        ]
        
        print(f"\nPotentially Low-Quality Frames: {len(problematic_frames)} out of {len(frame_stats)}")
        print("\nLow-Quality Frame Details:")
        for frame in problematic_frames[:10]:  # Show first 10
            print(f"Filename: {frame['filename']}")
            print(f"  Quality Score:     {frame['quality_score']:.4f}")
            print(f"  Noise Level:       {frame['noise_level']:.4f}")
            print(f"  Signal-to-Noise:   {frame['signal_to_noise_ratio']:.4f}")
            print(f"  Dynamic Range:     {frame['dynamic_range']:.4f}")

def main():
    # Initialize analyzer
    analyzer = M45DatasetAnalyzer()
    
    # Collect frame statistics
    frame_stats = analyzer.collect_frame_statistics()
    
    # Visualize dataset characteristics
    analyzer.visualize_quality_metrics(frame_stats)

if __name__ == '__main__':
    main()