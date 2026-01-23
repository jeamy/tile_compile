import numpy as np
import astropy.units as u
from astropy.modeling.models import Gaussian2D
from astropy.visualization import make_lupton_rgb
import imageio
import os
from typing import List, Tuple, Dict, Optional

class SyntheticAstronomicalDataset:
    def __init__(
        self, 
        image_size: Tuple[int, int] = (1024, 1024),
        star_count: int = 50,
        noise_level: float = 0.1,
        background_level: float = 100
    ):
        """
        Generate synthetic astronomical datasets
        
        Args:
            image_size: Size of generated images
            star_count: Number of simulated stars
            noise_level: Background noise level
            background_level: Mean background intensity
        """
        self.image_size = image_size
        self.star_count = star_count
        self.noise_level = noise_level
        self.background_level = background_level
        
        self.rng = np.random.default_rng()
    
    def generate_frame(
        self, 
        exposure_time: float = 300.0, 
        seeing_conditions: float = 1.0
    ) -> np.ndarray:
        """
        Generate a single astronomical frame
        
        Args:
            exposure_time: Simulated exposure time in seconds
            seeing_conditions: Atmospheric seeing quality
        
        Returns:
            Simulated astronomical frame
        """
        # Create background
        frame = self.rng.normal(
            loc=self.background_level, 
            scale=self.noise_level * exposure_time, 
            size=self.image_size
        )
        
        # Add stars
        for _ in range(self.star_count):
            # Random star parameters
            x = self.rng.integers(0, self.image_size[1])
            y = self.rng.integers(0, self.image_size[0])
            
            # Star intensity based on exposure
            peak = self.rng.uniform(500, 2000) * exposure_time
            
            # Gaussian star profile with seeing-dependent width
            sigma = 2 * seeing_conditions
            star = Gaussian2D(
                amplitude=peak, 
                x_mean=x, 
                y_mean=y, 
                x_stddev=sigma, 
                y_stddev=sigma
            )
            
            # Create star grid
            yy, xx = np.mgrid[0:self.image_size[0], 0:self.image_size[1]]
            star_grid = star(xx, yy)
            
            # Add star to frame
            frame += star_grid
        
        # Clip and normalize
        frame = np.clip(frame, 0, 65535).astype(np.uint16)
        return frame
    
    def generate_dataset(
        self, 
        num_frames: int = 50,
        exposure_variations: List[float] = [30.0, 100.0, 300.0],
        seeing_variations: List[float] = [0.5, 1.0, 2.0]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Generate complete dataset with variations
        
        Args:
            num_frames: Number of frames per variation
            exposure_variations: Different exposure times
            seeing_variations: Different seeing conditions
        
        Returns:
            Dictionary of datasets
        """
        datasets = {}
        
        for exposure in exposure_variations:
            for seeing in seeing_variations:
                key = f"exp_{exposure}_seeing_{seeing}"
                datasets[key] = [
                    self.generate_frame(
                        exposure_time=exposure, 
                        seeing_conditions=seeing
                    ) 
                    for _ in range(num_frames)
                ]
        
        return datasets
    
    def save_dataset(
        self, 
        datasets: Dict[str, List[np.ndarray]], 
        output_dir: str = 'synthetic_datasets'
    ):
        """
        Save generated datasets to disk
        
        Args:
            datasets: Generated datasets
            output_dir: Directory to save datasets
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, frames in datasets.items():
            dataset_path = os.path.join(output_dir, dataset_name)
            os.makedirs(dataset_path, exist_ok=True)
            
            for i, frame in enumerate(frames):
                # Save as FITS or TIFF
                imageio.imwrite(
                    os.path.join(dataset_path, f"frame_{i:04d}.tiff"), 
                    frame
                )
    
    @staticmethod
    def generate_color_composite(
        r_frame: np.ndarray, 
        g_frame: np.ndarray, 
        b_frame: np.ndarray
    ) -> np.ndarray:
        """
        Create color composite using Lupton et al. method
        
        Args:
            r_frame: Red channel frame
            g_frame: Green channel frame
            b_frame: Blue channel frame
        
        Returns:
            RGB color composite
        """
        return make_lupton_rgb(
            r_frame, g_frame, b_frame, 
            Q=10, stretch=0.5
        )

def main():
    # Example usage
    generator = SyntheticAstronomicalDataset(
        image_size=(2048, 2048),
        star_count=100,
        noise_level=0.05
    )
    
    datasets = generator.generate_dataset()
    generator.save_dataset(datasets)
    
    print(f"Generated {len(datasets)} synthetic datasets")

if __name__ == "__main__":
    main()