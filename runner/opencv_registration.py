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
    mad = float(np.median(np.abs(f)))
    sd = float(np.std(f))
    scale = (1.4826 * mad) if mad > 1e-12 else sd
    if scale > 1e-12:
        f = f / scale
    
    f = cv2.GaussianBlur(f, (0, 0), 0.8)
    small = cv2.GaussianBlur(f, (0, 0), 1.2)
    large = cv2.GaussianBlur(f, (0, 0), 6.0)
    f = small - large
    f = np.maximum(f, 0.0)
    f = cv2.GaussianBlur(f, (0, 0), 0.6)
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
    
    init = init_warp.astype("float32", copy=True)

    if allow_rotation:
        try:
            criteria_t = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-5)
            cc_t, warp_t = cv2.findTransformECC(
                ref01,
                moving01,
                init,
                cv2.MOTION_TRANSLATION,
                criteria_t,
                None,
                7,
            )
            init = warp_t.astype("float32", copy=False)
        except Exception:
            cc_t = 0.0
        try:
            criteria_r = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 250, 5e-6)
            cc_r, warp_r = cv2.findTransformECC(
                ref01,
                moving01,
                init,
                cv2.MOTION_EUCLIDEAN,
                criteria_r,
                None,
                5,
            )
            return warp_r.astype(np.float32, copy=False), float(cc_r)
        except Exception:
            return init.astype(np.float32, copy=False), float(cc_t)
    
    motion_type = cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    cc, warp = cv2.findTransformECC(ref01, moving01, init, motion_type, criteria, None, 7)
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


def opencv_best_translation_init(moving01: np.ndarray, ref01: np.ndarray) -> np.ndarray:
    """Find best initial translation for ECC."""
    dx, dy = opencv_phasecorr_translation(moving01, ref01)
    candidates = [
        np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, dx * 0.5], [0.0, 1.0, dy * 0.5]], dtype=np.float32),
        np.array([[1.0, 0.0, dx * 2.0], [0.0, 1.0, dy * 2.0]], dtype=np.float32),
    ]
    
    if cv2 is None:
        return candidates[0]
    
    best = candidates[0]
    best_score = -1.0
    for cand in candidates:
        try:
            warped = cv2.warpAffine(moving01, cand, (ref01.shape[1], ref01.shape[0]), flags=cv2.INTER_LINEAR)
            score = opencv_alignment_score(warped, ref01)
            if score > best_score:
                best_score = score
                best = cand
        except Exception:
            pass
    return best
