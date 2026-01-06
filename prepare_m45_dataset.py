"""
M45 Dataset Preparation for Methodik v3 Validation

This script prepares real M45 (Pleiades) FITS data for validation against
the Methodik v3 pipeline. It extracts frame metadata, validates against
Methodik v3 assumptions, and creates a compliant dataset manifest.

Methodik v3 features:
- Frame count validation (min, optimal, reduced mode thresholds)
- Exposure time consistency checking
- Elongation and registration residual validation
- Compliance reporting

Usage:
    python prepare_m45_dataset.py
"""

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
    
    # Methodik v3 specific fields
    exposure_time: Optional[float] = None
    fwhm: Optional[float] = None
    elongation: Optional[float] = None
    registration_residual: Optional[float] = None
    
    is_valid: bool = True
    quality_score: float = 0.0
    methodik_v3_compliant: bool = False

class M45DatasetPreparer:
    def __init__(
        self, 
        source_path: str = '/home/lux/tile-compile-cache/M45_lights', 
        output_path: str = 'validation_datasets/M45',
        random_seed: int = 42,
        assumptions: Optional[Dict] = None
    ):
        """
        Initialize dataset preparation with configurable parameters
        
        Args:
            source_path: Path to source FITS files
            output_path: Path to save prepared dataset
            random_seed: Seed for reproducible randomization
            assumptions: Methodik v3 assumptions dict (optional)
        """
        self.source_path = source_path
        self.output_path = output_path
        np.random.seed(random_seed)
        
        # Methodik v3 assumptions (defaults from tile_compile.yaml)
        self.assumptions = assumptions or {
            'frames_min': 50,
            'frames_optimal': 800,
            'frames_reduced_threshold': 200,
            'exposure_time_tolerance_percent': 5.0,
            'registration_residual_warn_px': 0.5,
            'registration_residual_max_px': 1.0,
            'elongation_warn': 0.3,
            'elongation_max': 0.4,
            'tracking_error_max_px': 1.0,
        }
        
        os.makedirs(output_path, exist_ok=True)
    
    def _analyze_frame(self, data: np.ndarray, header: fits.Header) -> FrameMetadata:
        """
        Compute comprehensive frame metadata with Methodik v3 fields
        
        Args:
            data: Numpy array of frame data
            header: FITS header
        
        Returns:
            Frame metadata with quality assessment and Methodik v3 compliance
        """
        metadata = FrameMetadata(
            filename='',
            mean=float(np.mean(data)),
            median=float(np.median(data)),
            std=float(np.std(data)),
            min=float(np.min(data)),
            max=float(np.max(data))
        )
        
        # Extract Methodik v3 specific metadata from FITS header
        metadata.exposure_time = float(header.get('EXPTIME', 0.0)) if 'EXPTIME' in header else None
        
        # Try to extract FWHM and elongation if available in header
        # (These would typically be computed by registration/star detection)
        metadata.fwhm = float(header.get('FWHM', 0.0)) if 'FWHM' in header else None
        metadata.elongation = float(header.get('ELONGAT', 0.0)) if 'ELONGAT' in header else None
        
        # More lenient quality score calculation
        metadata.quality_score = (
            1.0 / (1 + metadata.std * 0.1) *  # Slightly more tolerant to noise
            (1 - abs(metadata.mean - np.median(data)) / (metadata.max + 1e-8))  # Prefer balanced intensity
        )
        
        # More permissive validity check
        metadata.is_valid = (
            metadata.quality_score > 0.3 and  # Lowered threshold
            metadata.max > metadata.mean * 1.5  # Less strict dynamic range
        )
        
        # Check Methodik v3 compliance
        metadata.methodik_v3_compliant = self._check_methodik_v3_compliance(metadata)
        
        return metadata
    
    def _check_methodik_v3_compliance(self, metadata: FrameMetadata) -> bool:
        """
        Check if frame metadata complies with Methodik v3 assumptions.
        
        Args:
            metadata: Frame metadata to check
        
        Returns:
            True if compliant with Methodik v3 assumptions
        """
        # Check elongation if available
        if metadata.elongation is not None:
            if metadata.elongation > self.assumptions['elongation_max']:
                return False
        
        # Check registration residual if available
        if metadata.registration_residual is not None:
            if metadata.registration_residual > self.assumptions['registration_residual_max_px']:
                return False
        
        # Basic validity check
        if not metadata.is_valid:
            return False
        
        return True
    
    def validate_dataset_assumptions(self, frames: List[FrameMetadata]) -> Dict:
        """
        Validate entire dataset against Methodik v3 assumptions.
        
        Args:
            frames: List of frame metadata
        
        Returns:
            Validation report dict
        """
        frame_count = len(frames)
        
        report = {
            'frame_count': frame_count,
            'frames_min': self.assumptions['frames_min'],
            'frames_optimal': self.assumptions['frames_optimal'],
            'frames_reduced_threshold': self.assumptions['frames_reduced_threshold'],
            'meets_minimum': frame_count >= self.assumptions['frames_min'],
            'reduced_mode': frame_count < self.assumptions['frames_reduced_threshold'],
            'optimal_mode': frame_count >= self.assumptions['frames_optimal'],
            'warnings': [],
            'errors': [],
        }
        
        # Check frame count
        if frame_count < self.assumptions['frames_min']:
            report['errors'].append(f"Frame count ({frame_count}) below minimum ({self.assumptions['frames_min']})")
        elif frame_count < self.assumptions['frames_reduced_threshold']:
            report['warnings'].append(f"Reduced Mode: {frame_count} frames < {self.assumptions['frames_reduced_threshold']}")
        
        # Check exposure time consistency
        exposure_times = [f.exposure_time for f in frames if f.exposure_time is not None]
        if len(exposure_times) > 1:
            mean_exp = np.mean(exposure_times)
            tolerance = self.assumptions['exposure_time_tolerance_percent'] / 100.0
            deviations = [abs(e - mean_exp) / mean_exp for e in exposure_times]
            max_deviation = max(deviations)
            
            report['exposure_time_mean'] = float(mean_exp)
            report['exposure_time_max_deviation_percent'] = float(max_deviation * 100)
            
            if max_deviation > tolerance:
                report['errors'].append(
                    f"Exposure time deviation ({max_deviation*100:.2f}%) exceeds tolerance ({self.assumptions['exposure_time_tolerance_percent']}%)"
                )
        
        # Check elongation
        elongations = [f.elongation for f in frames if f.elongation is not None]
        if elongations:
            max_elongation = max(elongations)
            report['max_elongation'] = float(max_elongation)
            
            if max_elongation > self.assumptions['elongation_max']:
                report['errors'].append(
                    f"Max elongation ({max_elongation:.3f}) exceeds maximum ({self.assumptions['elongation_max']})"
                )
            elif max_elongation > self.assumptions['elongation_warn']:
                report['warnings'].append(
                    f"Max elongation ({max_elongation:.3f}) above warning threshold ({self.assumptions['elongation_warn']})"
                )
        
        # Check registration residual
        residuals = [f.registration_residual for f in frames if f.registration_residual is not None]
        if residuals:
            max_residual = max(residuals)
            report['max_registration_residual_px'] = float(max_residual)
            
            if max_residual > self.assumptions['registration_residual_max_px']:
                report['errors'].append(
                    f"Max registration residual ({max_residual:.3f} px) exceeds maximum ({self.assumptions['registration_residual_max_px']} px)"
                )
            elif max_residual > self.assumptions['registration_residual_warn_px']:
                report['warnings'].append(
                    f"Max registration residual ({max_residual:.3f} px) above warning threshold ({self.assumptions['registration_residual_warn_px']} px)"
                )
        
        # Count compliant frames
        compliant_frames = sum(1 for f in frames if f.methodik_v3_compliant)
        report['methodik_v3_compliant_frames'] = compliant_frames
        report['methodik_v3_compliance_rate'] = float(compliant_frames / frame_count) if frame_count > 0 else 0.0
        
        return report
    
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
        min_quality_threshold: float = 0.3  # Lowered threshold
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
        
        print("Processing frames:")
        for filename in fits_files:
            try:
                filepath = os.path.join(self.source_path, filename)
                
                with fits.open(filepath) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                
                # Compute frame metadata with header
                metadata = self._analyze_frame(data, header)
                metadata.filename = filename
                
                # Debug print with Methodik v3 info
                exp_time_str = f", ExpTime={metadata.exposure_time:.1f}s" if metadata.exposure_time else ""
                elong_str = f", Elong={metadata.elongation:.3f}" if metadata.elongation else ""
                v3_str = "✓" if metadata.methodik_v3_compliant else "✗"
                print(f"{filename}: Quality={metadata.quality_score:.4f}, Valid={metadata.is_valid}, v3={v3_str}{exp_time_str}{elong_str}")
                
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
        
        print(f"\nSelected {len(selected_frames)} frames out of {len(frame_metadata_list)} total frames")
        
        # Validate against Methodik v3 assumptions
        validation_report = self.validate_dataset_assumptions(selected_frames)
        
        print("\n" + "="*60)
        print("METHODIK V3 VALIDATION REPORT")
        print("="*60)
        print(f"Frame Count: {validation_report['frame_count']}")
        print(f"Meets Minimum: {validation_report['meets_minimum']}")
        print(f"Reduced Mode: {validation_report['reduced_mode']}")
        print(f"Optimal Mode: {validation_report['optimal_mode']}")
        print(f"v3 Compliance Rate: {validation_report['methodik_v3_compliance_rate']*100:.1f}%")
        
        if 'exposure_time_mean' in validation_report:
            print(f"Mean Exposure Time: {validation_report['exposure_time_mean']:.1f}s")
            print(f"Max Exposure Deviation: {validation_report['exposure_time_max_deviation_percent']:.2f}%")
        
        if 'max_elongation' in validation_report:
            print(f"Max Elongation: {validation_report['max_elongation']:.3f}")
        
        if 'max_registration_residual_px' in validation_report:
            print(f"Max Registration Residual: {validation_report['max_registration_residual_px']:.3f} px")
        
        if validation_report['warnings']:
            print("\nWARNINGS:")
            for warning in validation_report['warnings']:
                print(f"  ⚠ {warning}")
        
        if validation_report['errors']:
            print("\nERRORS:")
            for error in validation_report['errors']:
                print(f"  ✗ {error}")
        
        print("="*60 + "\n")
        
        # Prepare dataset manifest with Methodik v3 validation
        manifest = {
            'dataset_name': 'M45_Pleiades',
            'total_frames': len(selected_frames),
            'quality_threshold': min_quality_threshold,
            'methodik_v3_validation': validation_report,
            'assumptions': self.assumptions,
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
        
        print(f"\nPrepared validation dataset with {len(selected_frames)} frames")
        print(f"Output path: {self.output_path}")
        print(f"Manifest saved: {manifest_path}")
        
        return manifest

def main():
    preparer = M45DatasetPreparer()
    preparer.prepare_dataset(
        max_frames=50,  # Adjust as needed
        min_quality_threshold=0.3  # Lowered threshold
    )

if __name__ == '__main__':
    main()