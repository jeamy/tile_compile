"""
Image processing utilities for the tile-compile runner.

Functions for CFA/Bayer processing, channel splitting, normalization, and format conversion.
"""

from typing import Any

import numpy as np


def _local_median_3x3(data: np.ndarray) -> np.ndarray:
    """Compute local median of 8 neighbors (excluding center) using pure NumPy.
    
    This is a simple implementation that pads the array and computes the median
    of the 8 surrounding pixels for each position.
    """
    h, w = data.shape
    # Pad with edge values
    padded = np.pad(data, 1, mode='edge')
    
    # Collect all 8 neighbors
    neighbors = np.stack([
        padded[0:h, 0:w],      # top-left
        padded[0:h, 1:w+1],    # top
        padded[0:h, 2:w+2],    # top-right
        padded[1:h+1, 0:w],    # left
        # skip center
        padded[1:h+1, 2:w+2],  # right
        padded[2:h+2, 0:w],    # bottom-left
        padded[2:h+2, 1:w+1],  # bottom
        padded[2:h+2, 2:w+2],  # bottom-right
    ], axis=0)
    
    return np.median(neighbors, axis=0).astype("float32")


def cosmetic_correction(
    data: np.ndarray,
    sigma_threshold: float = 8.0,
    hot_only: bool = True,
) -> np.ndarray:
    """Replace hot/cold pixels with median of neighbors.
    
    This MUST be applied BEFORE registration/warp to prevent hotpixels
    from being interpolated and spread across the image ("walking noise").
    
    Args:
        data: 2D image array (float32)
        sigma_threshold: Pixels deviating more than this many robust sigmas
                        from the local median are replaced.
        hot_only: If True, only replace hot pixels (above threshold).
                 If False, also replace cold pixels (below threshold).
    
    Returns:
        Corrected image with hotpixels replaced by local median.
    """
    data = np.asarray(data).astype("float32", copy=True)
    
    # Compute robust statistics
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    robust_sigma = 1.4826 * mad
    
    if robust_sigma <= 0:
        return data
    
    # Find hotpixels
    threshold_hi = med + sigma_threshold * robust_sigma
    hot_mask = data > threshold_hi
    
    if not hot_only:
        threshold_lo = med - sigma_threshold * robust_sigma
        cold_mask = data < threshold_lo
        hot_mask = hot_mask | cold_mask
    
    if not np.any(hot_mask):
        return data
    
    # Compute local median of 8 neighbors
    local_median = _local_median_3x3(data)
    
    # Replace hotpixels with local median
    data[hot_mask] = local_median[hot_mask]
    
    return data


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


def demosaic_cfa(mosaic: np.ndarray, bayer_pattern: str) -> np.ndarray:
    """Demosaic CFA/Bayer mosaic to full-resolution RGB using OpenCV.
    
    Args:
        mosaic: 2D Bayer mosaic (H, W)
        bayer_pattern: Bayer pattern (RGGB, BGGR, GBRG, GRBG)
    
    Returns:
        RGB image (3, H, W) in float32
    """
    import cv2
    
    bp = str(bayer_pattern or "GBRG").strip().upper()
    
    # OpenCV Bayer pattern mapping
    bayer_codes = {
        "RGGB": cv2.COLOR_BAYER_RG2RGB,
        "BGGR": cv2.COLOR_BAYER_BG2RGB,
        "GBRG": cv2.COLOR_BAYER_GB2RGB,
        "GRBG": cv2.COLOR_BAYER_GR2RGB,
    }
    
    if bp not in bayer_codes:
        bp = "GBRG"
    
    # Crop to even dimensions
    h, w = mosaic.shape[:2]
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    if h2 != h or w2 != w:
        mosaic = mosaic[:h2, :w2]
    
    # Normalize to 0-255 range for OpenCV uint8 demosaicing (more stable)
    mosaic_min = np.min(mosaic)
    mosaic_max = np.max(mosaic)
    mosaic_range = mosaic_max - mosaic_min
    
    if mosaic_range > 0:
        # Use uint16 to avoid heavy quantization artifacts in the final RGB.
        # OpenCV's Bayer demosaicing supports 8-bit and 16-bit inputs.
        mosaic_u16 = np.clip((mosaic - mosaic_min) / mosaic_range * 65535.0, 0.0, 65535.0).astype(np.uint16)
    else:
        mosaic_u16 = np.zeros_like(mosaic, dtype=np.uint16)
    
    # Demosaic to RGB (H, W, 3) using VNG algorithm for better quality
    rgb_hwc = cv2.cvtColor(mosaic_u16, bayer_codes[bp])
    
    # Convert back to float32 in original range
    rgb_hwc = rgb_hwc.astype(np.float32)
    if mosaic_range > 0:
        rgb_hwc = rgb_hwc / 65535.0 * mosaic_range + mosaic_min
    
    # Convert to (3, H, W) format
    rgb = np.transpose(rgb_hwc, (2, 0, 1)).astype("float32", copy=False)
    
    return rgb


def reassemble_cfa_mosaic(r_plane: np.ndarray, g_plane: np.ndarray, b_plane: np.ndarray, bayer_pattern: str) -> np.ndarray:
    """Reassemble R, G, B subplanes back into a CFA/Bayer mosaic.
    
    This is the inverse operation of split_cfa_channels.
    
    Args:
        r_plane: Red subplane (H/2, W/2)
        g_plane: Green subplane (H/2, W/2) - averaged from G1+G2
        b_plane: Blue subplane (H/2, W/2)
        bayer_pattern: Bayer pattern (RGGB, BGGR, GBRG, GRBG)
    
    Returns:
        CFA mosaic (H, W) in float32
    """
    bp = str(bayer_pattern or "GBRG").strip().upper()
    
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
    
    h2, w2 = r_plane.shape[:2]
    h = h2 * 2
    w = w2 * 2
    
    mosaic = np.zeros((h, w), dtype=np.float32)
    mosaic[r_pos[0]::2, r_pos[1]::2] = r_plane
    mosaic[b_pos[0]::2, b_pos[1]::2] = b_plane
    mosaic[g1_pos[0]::2, g1_pos[1]::2] = g_plane
    mosaic[g2_pos[0]::2, g2_pos[1]::2] = g_plane
    
    return mosaic


def split_cfa_channels(mosaic: np.ndarray, bayer_pattern: str) -> dict[str, np.ndarray]:
    """Split CFA/Bayer mosaic into R, G, B subplanes (half-resolution, no demosaicing).
    
    This extracts the raw Bayer subplanes without interpolation. Each channel
    will be half the resolution of the input mosaic. Demosaicing happens later
    after stacking.
    """
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
            if target_median > 1e-10:
                return (frame * (target_median / frame_median)).astype("float32", copy=False)
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
    
    a = mosaic[0::2, 0::2].astype("float32", copy=False)
    b = mosaic[0::2, 1::2].astype("float32", copy=False)
    c = mosaic[1::2, 0::2].astype("float32", copy=False)
    d = mosaic[1::2, 1::2].astype("float32", copy=False)
    
    # NOTE: warp is already in subplane coordinates because it is estimated on
    # cfa_downsample_sum2x2() output (half-resolution). Do not rescale translation.
    # NOTE: OpenCV expects this matrix to be applied with WARP_INVERSE_MAP.
    warp_sub = warp.astype("float32", copy=False)
    a2 = warp_sub[:, :2]
    t2 = warp_sub[:, 2]
    delta_a = np.array([-0.25, -0.25], dtype=np.float32)
    delta_b = np.array([0.25, -0.25], dtype=np.float32)
    delta_c = np.array([-0.25, 0.25], dtype=np.float32)
    delta_d = np.array([0.25, 0.25], dtype=np.float32)
    warp_a = np.concatenate([a2, (t2 + a2 @ delta_a - delta_a)[:, None]], axis=1)
    warp_b = np.concatenate([a2, (t2 + a2 @ delta_b - delta_b)[:, None]], axis=1)
    warp_c = np.concatenate([a2, (t2 + a2 @ delta_c - delta_c)[:, None]], axis=1)
    warp_d = np.concatenate([a2, (t2 + a2 @ delta_d - delta_d)[:, None]], axis=1)
    
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    a_w = cv2.warpAffine(a, warp_a, (a.shape[1], a.shape[0]), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    b_w = cv2.warpAffine(b, warp_b, (b.shape[1], b.shape[0]), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    c_w = cv2.warpAffine(c, warp_c, (c.shape[1], c.shape[0]), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    d_w = cv2.warpAffine(d, warp_d, (d.shape[1], d.shape[0]), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    
    out = np.zeros((h2, w2), dtype=np.float32)
    out[0::2, 0::2] = a_w
    out[0::2, 1::2] = b_w
    out[1::2, 0::2] = c_w
    out[1::2, 1::2] = d_w
    
    return out
