import numpy as np
import cv2
from typing import Any, List, Tuple, Dict, Optional
from enum import Enum

class BayerPattern(Enum):
    RGGB = 'RGGB'
    BGGR = 'BGGR'
    GBRG = 'GBRG'
    GRBG = 'GRBG'

class CFARegistration:
    @classmethod
    def register_cfa_frames(
        cls, 
        frames: List[np.ndarray], 
        bayer_pattern: BayerPattern = BayerPattern.GBRG,
        min_stars: int = 10,
        max_translation: float = 10.0,
        max_rotation: float = 5.0
    ) -> Dict[str, Any]:
        """
        Perform CFA-aware registration on frames
        
        Args:
            frames: List of CFA frames
            bayer_pattern: Bayer color filter arrangement
            min_stars: Minimum stars required for registration
            max_translation: Maximum allowed pixel translation
            max_rotation: Maximum allowed rotation in degrees
        
        Returns:
            Registration results dictionary
        """
        # Reference frame selection
        ref_frame, ref_index = cls._select_reference_frame(frames)
        
        # Subplane extraction
        ref_subplanes = cls._extract_subplanes(ref_frame, bayer_pattern)
        
        # Registration results container
        registration_results = {
            'registered_frames': [],
            'transformations': [],
            'quality_metrics': []
        }
        
        for i, frame in enumerate(frames):
            if i == ref_index:
                # Reference frame remains unchanged
                registration_results['registered_frames'].append(frame)
                registration_results['transformations'].append(np.eye(2, 3, dtype=np.float32))
                registration_results['quality_metrics'].append(1.0)
                continue
            
            # Process current frame
            current_subplanes = cls._extract_subplanes(frame, bayer_pattern)
            
            # Estimate transformation for each subplane
            subplane_transformations = []
            subplane_qualities = []
            
            for ref_plane, current_plane in zip(ref_subplanes, current_subplanes):
                transform, quality = cls._estimate_subplane_transform(
                    ref_plane, 
                    current_plane, 
                    max_translation, 
                    max_rotation
                )
                subplane_transformations.append(transform)
                subplane_qualities.append(quality)
            
            # Validate registration
            if not cls._validate_registration(
                subplane_transformations, 
                subplane_qualities, 
                min_stars
            ):
                continue
            
            # Warp frame using consistent transformation
            registered_frame = cls._warp_cfa_frame(
                frame, 
                subplane_transformations, 
                bayer_pattern
            )
            
            registration_results['registered_frames'].append(registered_frame)
            registration_results['transformations'].append(subplane_transformations)
            registration_results['quality_metrics'].append(np.mean(subplane_qualities))
        
        return registration_results
    
    @staticmethod
    def _select_reference_frame(
        frames: List[np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """
        Select reference frame with maximum star count
        """
        star_counts = [CFARegistration._count_stars(frame) for frame in frames]
        ref_index = np.argmax(star_counts)
        return frames[ref_index], ref_index
    
    @staticmethod
    def _count_stars(frame: np.ndarray, min_quality: float = 0.01) -> int:
        """
        Count stars in a frame using OpenCV
        """
        # Downsample for efficiency
        if frame.shape[0] > 1000 or frame.shape[1] > 1000:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # Gaussian blur to reduce noise
        frame_f32 = frame.astype(np.float32, copy=False)
        blurred = cv2.GaussianBlur(frame_f32, (5, 5), 0)
        
        # Detect corners (potential stars)
        corners = cv2.goodFeaturesToTrack(
            blurred, 
            maxCorners=200, 
            qualityLevel=min_quality, 
            minDistance=10
        )
        
        return len(corners) if corners is not None else 0
    
    @staticmethod
    def _extract_subplanes(
        frame: np.ndarray, 
        bayer_pattern: BayerPattern
    ) -> List[np.ndarray]:
        """
        Extract color subplanes based on Bayer pattern
        """
        h, w = frame.shape[:2]
        subplanes = []
        
        if bayer_pattern == BayerPattern.RGGB:
            subplanes = [
                frame[0::2, 0::2],  # R
                frame[0::2, 1::2],  # G1
                frame[1::2, 0::2],  # G2
                frame[1::2, 1::2]   # B
            ]
        elif bayer_pattern == BayerPattern.BGGR:
            subplanes = [
                frame[1::2, 1::2],  # R
                frame[0::2, 1::2],  # G1
                frame[1::2, 0::2],  # G2
                frame[0::2, 0::2]   # B
            ]
        # Add other Bayer pattern implementations
        
        return subplanes
    
    @staticmethod
    def _estimate_subplane_transform(
        ref_plane: np.ndarray, 
        current_plane: np.ndarray, 
        max_translation: float,
        max_rotation: float
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate transformation for a single subplane
        """
        # Prepare images for ECC registration
        ref_gray = cv2.normalize(ref_plane.astype(np.float32, copy=False), None, 0, 1, cv2.NORM_MINMAX)
        current_gray = cv2.normalize(current_plane.astype(np.float32, copy=False), None, 0, 1, cv2.NORM_MINMAX)
        
        # Initial translation guess
        warp_mode = cv2.MOTION_TRANSLATION
        warp = np.eye(2, 3, dtype=np.float32)
        
        # Define convergence criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
            1000, 
            1e-5
        )
        
        try:
            # Estimate transformation
            cc, warp = cv2.findTransformECC(
                ref_gray, 
                current_gray, 
                warp, 
                warp_mode, 
                criteria
            )
        except cv2.error:
            # Fallback: identity transformation
            warp = np.eye(2, 3, dtype=np.float32)
            cc = 0.0
        
        # Validate transformation
        translation = np.linalg.norm(warp[:, 2])
        rotation = abs(np.arctan2(warp[1, 0], warp[0, 0]) * 180 / np.pi)
        
        if translation > max_translation or rotation > max_rotation:
            warp = np.eye(2, 3, dtype=np.float32)
            cc = 0.0
        
        return warp, float(cc)
    
    @staticmethod
    def _validate_registration(
        transformations: List[np.ndarray], 
        qualities: List[float],
        min_stars: int
    ) -> bool:
        """
        Validate overall registration quality
        """
        # Require minimum quality for most subplanes
        valid_transforms = [q > 0.5 for q in qualities]
        return sum(valid_transforms) >= len(transformations) - 1
    
    @staticmethod
    def _warp_cfa_frame(
        frame: np.ndarray, 
        transformations: List[np.ndarray], 
        bayer_pattern: BayerPattern
    ) -> np.ndarray:
        """
        Apply transformations to CFA frame
        """
        # Reconstruct warped subplanes
        h, w = frame.shape[:2]
        warped_subplanes = []
        
        subplanes = CFARegistration._extract_subplanes(frame, bayer_pattern)
        
        for plane, transform in zip(subplanes, transformations):
            warped = cv2.warpAffine(
                plane, 
                transform, 
                (w//2, h//2), 
                flags=cv2.INTER_LINEAR
            )
            warped_subplanes.append(warped)
        
        # Reassemble CFA frame
        if bayer_pattern == BayerPattern.RGGB:
            reconstructed = np.zeros_like(frame)
            reconstructed[0::2, 0::2] = warped_subplanes[0]
            reconstructed[0::2, 1::2] = warped_subplanes[1]
            reconstructed[1::2, 0::2] = warped_subplanes[2]
            reconstructed[1::2, 1::2] = warped_subplanes[3]
        
        return reconstructed