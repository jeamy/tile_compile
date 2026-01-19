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
