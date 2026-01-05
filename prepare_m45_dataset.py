import os
import json
import numpy as np
from astropy.io import fits
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

@dataclass
class FrameMetadata:
    filename: str
    mean: float
    median: float
    std: float
    min: float
    max: float
    is_valid: bool = True
    quality_score: float = 0.0

class M45DatasetPreparer:
    def __init__(
        self, 
        source_path: str = '/home/lux/tile-compile-cache/M45_lights', 
        output_path: str = 'validation_datasets/M45',
        random_seed: int = 42
    ):
        """
        Initialize dataset preparation with configurable parameters
        
        Args:
            source_path: Path to source FITS files
            output_path: Path to save prepared dataset
            random_seed: Seed for reproducible randomization
        """
        self.source_path = source_path
        self.output_path = output_path
        np.random.seed(random_seed)
        
        os.makedirs(output_path, exist_ok=True)
    
    def _analyze_frame(self, data: np.ndarray) -> FrameMetadata:
        """
        Compute comprehensive frame metadata
        
        Args:
            data: Numpy array of frame data
        
        Returns:
            Frame metadata with quality assessment
        """
        metadata = FrameMetadata(
            filename='',
            mean=float(np.mean(data)),
            median=float(np.median(data)),
            std=float(np.std(data)),
            min=float(np.min(data)),
            max=float(np.max(data))
        )
        
        # Simple quality score based on statistical properties
        metadata.quality_score = (
            1.0 / (1 + metadata.std) *  # Prefer low noise
            (1 - abs(metadata.mean - np.median(data)) / (metadata.max + 1e-8))  # Prefer balanced intensity
        )
        
        # Basic validity check
        metadata.is_valid = (
            metadata.quality_score > 0.5 and  # Good quality
            metadata.max > metadata.mean * 2  # Sufficient dynamic range
        )
        
        return metadata
    
    def _normalize_frame(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize frame data
        
        Args:
            data: Input frame data
        
        Returns:
            Normalized frame data
        """
        # Robust normalization with outlier handling
        p1, p99 = np.percentile(data, [1, 99])
        data_clipped = np.clip(data, p1, p99)
        
        return (data_clipped - np.min(data_clipped)) / (np.max(data_clipped) - np.min(data_clipped))
    
    def prepare_dataset(
        self, 
        max_frames: int = 50, 
        min_quality_threshold: float = 0.7
    ) -> Dict:
        """
        Prepare validation dataset
        
        Args:
            max_frames: Maximum number of frames to select
            min_quality_threshold: Minimum quality score for frame selection
        
        Returns:
            Dataset manifest
        """
        # List FITS files
        fits_files = [f for f in os.listdir(self.source_path) if f.endswith('.fits')]
        
        # Frame metadata collection
        frame_metadata_list: List[FrameMetadata] = []
        
        for filename in fits_files:
            try:
                filepath = os.path.join(self.source_path, filename)
                
                with fits.open(filepath) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                
                # Compute frame metadata
                metadata = self._analyze_frame(data)
                metadata.filename = filename
                
                frame_metadata_list.append(metadata)
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        # Filter and sort frames by quality
        valid_frames = [
            frame for frame in frame_metadata_list 
            if frame.is_valid and frame.quality_score >= min_quality_threshold
        ]
        valid_frames.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Select top frames
        selected_frames = valid_frames[:max_frames]
        
        # Prepare dataset manifest
        manifest = {
            'dataset_name': 'M45_Pleiades',
            'total_frames': len(selected_frames),
            'quality_threshold': min_quality_threshold,
            'frames': []
        }
        
        # Process and save selected frames
        for metadata in selected_frames:
            try:
                filepath = os.path.join(self.source_path, metadata.filename)
                output_filepath = os.path.join(self.output_path, metadata.filename)
                
                with fits.open(filepath) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                
                # Normalize data
                normalized_data = self._normalize_frame(data)
                
                # Save normalized FITS
                fits.writeto(output_filepath, normalized_data, header=header, overwrite=True)
                
                # Add to manifest
                frame_entry = asdict(metadata)
                frame_entry['filepath'] = output_filepath
                manifest['frames'].append(frame_entry)
            
            except Exception as e:
                print(f"Error saving {metadata.filename}: {e}")
        
        # Save manifest
        manifest_path = os.path.join(self.output_path, 'dataset_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Prepared validation dataset with {len(selected_frames)} frames")
        print(f"Output path: {self.output_path}")
        print(f"Manifest saved: {manifest_path}")
        
        return manifest

def main():
    preparer = M45DatasetPreparer()
    preparer.prepare_dataset(
        max_frames=50,  # Adjust as needed
        min_quality_threshold=0.7
    )

if __name__ == '__main__':
    main()