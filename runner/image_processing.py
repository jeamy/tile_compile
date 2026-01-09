"""
Image processing utilities for the tile-compile runner.

Functions for CFA/Bayer processing, channel splitting, normalization, and format conversion.
"""

from typing import Any

import numpy as np


def cfa_downsample_sum2x2(mosaic: np.ndarray) -> np.ndarray:
    """Downsample CFA mosaic by summing 2x2 blocks."""
    h, w = mosaic.shape[:2]
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    if h2 != h or w2 != w:
        mosaic = mosaic[:h2, :w2]
    a = mosaic[0::2, 0::2].astype("float32", copy=False)
    b = mosaic[0::2, 1::2].astype("float32", copy=False)
    c = mosaic[1::2, 0::2].astype("float32", copy=False)
    d = mosaic[1::2, 1::2].astype("float32", copy=False)
    return a + b + c + d


def split_rgb_frame(data: np.ndarray) -> dict[str, np.ndarray]:
    """Split RGB FITS frame into R, G, B channels."""
    if data.ndim == 2:
        return {"R": data, "G": data, "B": data}
    if data.ndim != 3:
        raise RuntimeError(f"expected 2D or 3D FITS, got shape={data.shape}")
    if data.shape[0] == 3:
        return {"R": data[0], "G": data[1], "B": data[2]}
    if data.shape[2] == 3:
        return {"R": data[:, :, 0], "G": data[:, :, 1], "B": data[:, :, 2]}
    raise RuntimeError(f"unsupported RGB FITS layout: shape={data.shape}")


def split_cfa_channels(mosaic: np.ndarray, bayer_pattern: str) -> dict[str, np.ndarray]:
    """Split CFA/Bayer mosaic into R, G, B channels."""
    bp = str(bayer_pattern or "GBRG").strip().upper()
    h, w = mosaic.shape[:2]
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    if h2 != h or w2 != w:
        mosaic = mosaic[:h2, :w2]
    
    # Bayer pattern mapping
    patterns = {
        "RGGB": {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
        "BGGR": {"B": (0, 0), "G1": (0, 1), "G2": (1, 0), "R": (1, 1)},
        "GBRG": {"G1": (0, 0), "B": (0, 1), "R": (1, 0), "G2": (1, 1)},
        "GRBG": {"G1": (0, 0), "R": (0, 1), "B": (1, 0), "G2": (1, 1)},
    }
    
    if bp not in patterns:
        bp = "GBRG"
    
    pat = patterns[bp]
    r_pos = pat["R"]
    b_pos = pat["B"]
    g1_pos = pat["G1"]
    g2_pos = pat["G2"]
    
    r_plane = mosaic[r_pos[0]::2, r_pos[1]::2].astype("float32", copy=False)
    b_plane = mosaic[b_pos[0]::2, b_pos[1]::2].astype("float32", copy=False)
    g1_plane = mosaic[g1_pos[0]::2, g1_pos[1]::2].astype("float32", copy=False)
    g2_plane = mosaic[g2_pos[0]::2, g2_pos[1]::2].astype("float32", copy=False)
    g_plane = (g1_plane + g2_plane) * 0.5
    
    return {
        "R": r_plane,
        "G": g_plane,
        "B": b_plane,
    }


def compute_frame_medians(frames: list[np.ndarray]) -> tuple[list[float], float]:
    """Compute median for each frame and global target median.
    
    Returns:
        (medians, target): List of per-frame medians and global target median
    """
    if not frames:
        return [], 0.0
    meds = [float(np.median(f)) for f in frames]
    target = float(np.median(np.asarray(meds, dtype=np.float32)))
    return meds, target


def normalize_frame(frame: np.ndarray, frame_median: float, target_median: float, mode: str) -> np.ndarray:
    """Normalize a single frame to target median per Methodik v3 §3.1.
    
    Methodik v3 specifies: I'_f = I_f / B_f (divisive normalization)
    
    Args:
        frame: Input frame
        frame_median: Background level B_f of this frame
        target_median: Target median (not used in v3 mode)
        mode: Normalization mode ("background" for v3 divisive, "additive" for legacy)
    
    Returns:
        Normalized frame
    """
    if mode == "background":
        # Methodik v3 §3.1: I'_f = I_f / B_f (divisive normalization)
        if frame_median > 1e-10:
            return (frame / frame_median).astype("float32", copy=False)
        else:
            return frame.astype("float32", copy=False)
    elif mode == "additive":
        # Legacy additive normalization
        return (frame - (frame_median - target_median)).astype("float32", copy=False)
    else:
        return frame.astype("float32", copy=False)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float image to uint8 for visualization."""
    f = np.asarray(img).astype("float32", copy=False)
    mn = float(np.min(f))
    mx = float(np.max(f))
    if mx <= mn:
        return np.zeros_like(f, dtype=np.uint8)
    x = (f - mn) / (mx - mn)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def warp_cfa_mosaic_via_subplanes(mosaic: np.ndarray, warp: np.ndarray) -> np.ndarray:
    """Warp CFA mosaic by warping each Bayer plane separately."""
    import cv2
    
    h, w = mosaic.shape[:2]
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    if h2 != h or w2 != w:
        mosaic = mosaic[:h2, :w2]
    
    a = mosaic[0::2, 0::2]
    b = mosaic[0::2, 1::2]
    c = mosaic[1::2, 0::2]
    d = mosaic[1::2, 1::2]
    
    warp_sub = warp.copy()
    warp_sub[0, 2] /= 2.0
    warp_sub[1, 2] /= 2.0
    
    a_w = cv2.warpAffine(a, warp_sub, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    b_w = cv2.warpAffine(b, warp_sub, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    c_w = cv2.warpAffine(c, warp_sub, (c.shape[1], c.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    d_w = cv2.warpAffine(d, warp_sub, (d.shape[1], d.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    out = np.zeros((h2, w2), dtype=mosaic.dtype)
    out[0::2, 0::2] = a_w
    out[0::2, 1::2] = b_w
    out[1::2, 0::2] = c_w
    out[1::2, 1::2] = d_w
    
    return out
