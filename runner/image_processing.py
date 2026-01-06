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


def normalize_frames(frames: list[np.ndarray], mode: str) -> tuple[list[np.ndarray], float]:
    """Normalize frames to common background level."""
    if not frames:
        return [], 0.0
    meds = [float(np.median(f)) for f in frames]
    target = float(np.median(np.asarray(meds, dtype=np.float32)))
    out = []
    for f, m in zip(frames, meds):
        if mode == "background":
            out.append((f - (m - target)).astype("float32", copy=False))
        else:
            out.append(f)
    return out, target


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
