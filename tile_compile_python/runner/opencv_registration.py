"""
OpenCV-based registration utilities for the tile-compile runner.

Functions for ECC alignment, phase correlation, and star detection using OpenCV.
"""

from typing import Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def opencv_prepare_ecc_image(img: np.ndarray) -> np.ndarray:
    """Prepare image for ECC registration."""
    if cv2 is None:
        raise RuntimeError("OpenCV not available")
    
    f = img.astype("float32", copy=False)
    med = float(np.median(f))
    f = f - med
    sd = float(np.std(f))
    if sd > 0:
        f = f / sd
    bg = cv2.GaussianBlur(f, (0, 0), 12.0)
    f = f - bg
    f = cv2.GaussianBlur(f, (0, 0), 1.0)
    f = cv2.normalize(f, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    return f


def opencv_count_stars(img01: np.ndarray) -> int:
    """Count stars using goodFeaturesToTrack."""
    if cv2 is None:
        return 0
    
    corners = cv2.goodFeaturesToTrack(
        img01,
        maxCorners=1200,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
    )
    return int(0 if corners is None else len(corners))


def opencv_ecc_warp(
    moving01: np.ndarray,
    ref01: np.ndarray,
    allow_rotation: bool,
    init_warp: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute ECC warp matrix."""
    if cv2 is None:
        raise RuntimeError("OpenCV not available")
    
    motion_type = cv2.MOTION_EUCLIDEAN if allow_rotation else cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    cc, warp = cv2.findTransformECC(ref01, moving01, init_warp, motion_type, criteria)
    return warp.astype(np.float32, copy=False), float(cc)


def opencv_phasecorr_translation(moving01: np.ndarray, ref01: np.ndarray) -> tuple[float, float]:
    """Compute translation using phase correlation."""
    if cv2 is None:
        return 0.0, 0.0
    
    win = cv2.createHanningWindow((ref01.shape[1], ref01.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ref01.astype("float32", copy=False), moving01.astype("float32", copy=False), win)
    return float(dx), float(dy)


def opencv_alignment_score(moving01: np.ndarray, ref01: np.ndarray) -> float:
    """Compute alignment score between two images."""
    a = moving01.astype("float32", copy=False)
    b = ref01.astype("float32", copy=False)
    am = float(np.mean(a))
    bm = float(np.mean(b))
    da = a - am
    db = b - bm
    denom = float(np.sqrt(np.sum(da * da) * np.sum(db * db)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(da * db) / denom)


def opencv_best_translation_init(
    moving01: np.ndarray,
    ref01: np.ndarray,
    rotation_sweep: bool = True,
    rotation_range_deg: float = 5.0,
    rotation_steps: int = 11,
) -> np.ndarray:
    """Find best initial warp (translation + optional rotation sweep) for ECC.
    
    Args:
        moving01: Moving image (normalized 0-1)
        ref01: Reference image (normalized 0-1)
        rotation_sweep: If True, test multiple rotation angles around 0
        rotation_range_deg: Max rotation angle to test (±degrees)
        rotation_steps: Number of rotation angles to test (odd recommended)
    
    Returns:
        Best initial warp matrix (2x3 affine)
    """
    dx, dy = opencv_phasecorr_translation(moving01, ref01)
    
    if cv2 is None:
        return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    
    h, w = ref01.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    
    # Build candidate list: translations × rotations
    translations = [
        (dx, dy),
        (0.0, 0.0),
        (dx * 0.5, dy * 0.5),
    ]
    
    if rotation_sweep:
        # Test rotation angles from -range to +range
        angles_deg = np.linspace(-rotation_range_deg, rotation_range_deg, rotation_steps)
    else:
        angles_deg = [0.0]
    
    candidates: list[np.ndarray] = []
    for tx, ty in translations:
        for angle_deg in angles_deg:
            # Build rotation matrix around image center, then add translation
            theta = np.deg2rad(angle_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            # Rotation around center: R @ (p - c) + c + t
            # = R @ p - R @ c + c + t
            # Affine: [[cos, -sin, -cos*cx + sin*cy + cx + tx],
            #          [sin,  cos, -sin*cx - cos*cy + cy + ty]]
            warp = np.array([
                [cos_t, -sin_t, -cos_t * cx + sin_t * cy + cx + tx],
                [sin_t,  cos_t, -sin_t * cx - cos_t * cy + cy + ty],
            ], dtype=np.float32)
            candidates.append(warp)
    
    best = candidates[0]
    best_score = -1.0
    for cand in candidates:
        try:
            warped = cv2.warpAffine(moving01, cand, (w, h), flags=cv2.INTER_LINEAR)
            score = opencv_alignment_score(warped, ref01)
            if score > best_score:
                best_score = score
                best = cand
        except Exception:
            pass
    return best


def opencv_detect_stars(img01: np.ndarray, max_stars: int = 200) -> np.ndarray:
    """Detect star positions using goodFeaturesToTrack.
    
    Args:
        img01: Normalized image (0-1 range)
        max_stars: Maximum number of stars to detect
        
    Returns:
        Array of shape (N, 2) with (x, y) coordinates
    """
    if cv2 is None:
        return np.zeros((0, 2), dtype=np.float32)
    
    corners = cv2.goodFeaturesToTrack(
        img01,
        maxCorners=max_stars,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7,
    )
    if corners is None or len(corners) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    # corners is (N, 1, 2), reshape to (N, 2)
    return corners.reshape(-1, 2).astype(np.float32)


def opencv_star_match_ransac(
    moving01: np.ndarray,
    ref01: np.ndarray,
    max_stars: int = 150,
    ransac_threshold: float = 5.0,
) -> tuple[np.ndarray | None, int]:
    """Match stars between two images and compute affine transform with RANSAC.
    
    Uses nearest-neighbor matching with RANSAC to find robust transformation.
    Works well when rotation between frames is moderate (<10°).
    
    Args:
        moving01: Moving image (normalized 0-1)
        ref01: Reference image (normalized 0-1)
        max_stars: Maximum stars to detect per image
        ransac_threshold: RANSAC inlier threshold in pixels
        
    Returns:
        Tuple of (affine_matrix, num_inliers) or (None, 0) if failed
    """
    if cv2 is None:
        return None, 0
    
    # Detect stars in both images
    pts_mov = opencv_detect_stars(moving01, max_stars)
    pts_ref = opencv_detect_stars(ref01, max_stars)
    
    if len(pts_mov) < 6 or len(pts_ref) < 6:
        return None, 0
    
    # Build KD-tree style matching: for each moving star, find nearest in ref
    # Use brute force for simplicity (fast enough for <200 stars)
    from scipy.spatial import cKDTree
    
    tree_ref = cKDTree(pts_ref)
    
    # Find nearest neighbor for each moving point
    distances, indices = tree_ref.query(pts_mov, k=1)
    
    # Filter by distance (reject matches that are too far)
    max_match_dist = max(moving01.shape) * 0.15  # 15% of image size
    valid_mask = distances < max_match_dist
    
    if np.sum(valid_mask) < 6:
        return None, 0
    
    src_pts = pts_mov[valid_mask]
    dst_pts = pts_ref[indices[valid_mask]]
    
    # Estimate affine transform with RANSAC
    # cv2.estimateAffinePartial2D: rotation + translation + uniform scale
    # cv2.estimateAffine2D: full affine (6 DOF)
    # Use partial (4 DOF) for more robustness
    transform, inliers = cv2.estimateAffinePartial2D(
        src_pts.reshape(-1, 1, 2),
        dst_pts.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.99,
    )
    
    if transform is None:
        return None, 0
    
    num_inliers = int(np.sum(inliers)) if inliers is not None else 0
    return transform.astype(np.float32), num_inliers


def opencv_register_stars(
    moving01: np.ndarray,
    ref01: np.ndarray,
    fallback_to_ecc: bool = True,
    allow_rotation: bool = True,
) -> tuple[np.ndarray, float, str]:
    """Register two images using star matching with RANSAC, with ECC fallback.
    
    Args:
        moving01: Moving image (normalized 0-1)
        ref01: Reference image (normalized 0-1)
        fallback_to_ecc: If True, fall back to ECC if star matching fails
        allow_rotation: Allow rotation in transformation
        
    Returns:
        Tuple of (warp_matrix, confidence, method_used)
        confidence is number of inliers for star matching, or ECC correlation
    """
    # Try star matching first
    warp, num_inliers = opencv_star_match_ransac(moving01, ref01)
    
    if warp is not None and num_inliers >= 10:
        # Good star match
        return warp, float(num_inliers), "star_ransac"
    
    # Star matching failed or insufficient inliers, try ECC
    if fallback_to_ecc:
        init = opencv_best_translation_init(moving01, ref01, rotation_sweep=True)
        try:
            warp, cc = opencv_ecc_warp(moving01, ref01, allow_rotation=allow_rotation, init_warp=init)
            return warp, cc, "ecc"
        except Exception:
            pass
    
    # Last resort: phase correlation only
    dx, dy = opencv_phasecorr_translation(moving01, ref01)
    warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return warp, 0.0, "phase_corr"
