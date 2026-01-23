"""Sigma-clipping and linear stacking utilities (pure Python / NumPy).

These helpers are intentionally backend-only and do not touch any runner logic
(disk, logging, etc.). They operate purely on NumPy arrays so that the
pipeline can implement artifact removal and stacking without Siril.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class SigmaClipConfig:
    """Configuration for sigma-clipping.

    Parameters are generic and can be mapped from the YAML/JSON config
    by the runner. All thresholds are expressed in standard deviations
    relative to the current mean.
    """

    sigma_low: float = 3.0
    sigma_high: float = 3.0
    max_iters: int = 3
    min_fraction: float = 0.5

    def clamp(self) -> "SigmaClipConfig":
        """Return a clamped copy to avoid pathological settings."""

        s_lo = float(self.sigma_low)
        s_hi = float(self.sigma_high)
        if not np.isfinite(s_lo) or s_lo <= 0.0:
            s_lo = 3.0
        if not np.isfinite(s_hi) or s_hi <= 0.0:
            s_hi = 3.0

        it = int(self.max_iters)
        if it <= 0:
            it = 1
        if it > 10:
            it = 10

        mf = float(self.min_fraction)
        if not np.isfinite(mf) or mf <= 0.0:
            mf = 0.1
        if mf > 1.0:
            mf = 1.0

        return SigmaClipConfig(sigma_low=s_lo, sigma_high=s_hi, max_iters=it, min_fraction=mf)


def sigma_clip_stack_nd(
    frames: np.ndarray,
    cfg: SigmaClipConfig | Dict[str, Any] | None = None,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Sigma-clip a stack of frames along axis 0.

    Parameters
    ----------
    frames:
        Array of shape (N, ...) with N frames to be combined. The function
        works for 2D, 3D (e.g. RGB), or higher dimensional frames as long as
        the first dimension indexes frames.
    cfg:
        Either a :class:`SigmaClipConfig` instance or a mapping with keys
        ``sigma_low``, ``sigma_high``, ``max_iters``, and ``min_fraction``.

    Returns
    -------
    clipped_mean : np.ndarray
        The mean over non-rejected samples along axis 0.
    mask : np.ndarray
        Boolean mask of shape (N, ...) indicating which samples were kept
        (True) or rejected (False) after the final iteration.
    stats : dict
        Diagnostic information (iterations used, rejected fraction, etc.).
    """

    f = np.asarray(frames)
    if f.ndim < 2:
        raise ValueError("sigma_clip_stack_nd expects an array of shape (N, ...) with N>=1")

    n_frames = int(f.shape[0])
    if n_frames <= 0:
        raise ValueError("sigma_clip_stack_nd received an empty stack (N=0)")

    if isinstance(cfg, dict) or cfg is None:
        cfg = SigmaClipConfig(**(cfg or {}))
    if not isinstance(cfg, SigmaClipConfig):
        raise TypeError("cfg must be SigmaClipConfig or a mapping")
    cfg = cfg.clamp()

    # Work in float32 for memory/speed while keeping enough precision.
    f = f.astype("float32", copy=False)

    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool, copy=False)
        if vm.shape != f.shape:
            raise ValueError(f"valid_mask shape {vm.shape} does not match frames shape {f.shape}")
        mask = vm.copy()
    else:
        mask = np.ones_like(f, dtype=bool)

    base_valid = mask.copy()
    total_samples = float(base_valid.sum(dtype=np.int64))
    if total_samples <= 0.0:
        clipped_mean = np.zeros(f.shape[1:], dtype=np.float32)
        stats: Dict[str, Any] = {
            "frames": n_frames,
            "iterations": 0,
            "kept_fraction": 0.0,
            "rejected_fraction": 0.0,
            "min_fraction": float(cfg.min_fraction),
            "sigma_low": float(cfg.sigma_low),
            "sigma_high": float(cfg.sigma_high),
            "error": "no valid samples (valid_mask is empty)",
        }
        return clipped_mean, mask, stats

    last_mask_sum = mask.sum(dtype=np.int64)

    for it in range(int(cfg.max_iters)):
        # Compute statistics over currently valid samples.
        valid = mask
        counts = valid.sum(axis=0)
        # Avoid division by zero: where counts <= 0, keep mean/std as zeros.
        # Those positions will remain rejected or fall back later.
        counts_safe = np.maximum(counts, 1)

        mean = f.sum(axis=0, where=valid) / counts_safe
        diff = f - mean
        sq = diff * diff
        var = sq.sum(axis=0, where=valid) / counts_safe
        std = np.sqrt(var, dtype="float32")

        # Where std ~ 0, we do not reject anything in this iteration.
        std_safe = np.where(std <= 0.0, 1.0, std)

        z = diff / std_safe
        new_mask = (z >= -cfg.sigma_low) & (z <= cfg.sigma_high) & valid

        new_mask_sum = new_mask.sum(dtype=np.int64)
        # Stop if mask did not change or nothing left to reject.
        if new_mask_sum == last_mask_sum or new_mask_sum == 0:
            mask = new_mask
            break

        mask = new_mask
        last_mask_sum = new_mask_sum

    # Final mean over accepted samples.
    counts_final = mask.sum(axis=0)
    counts_safe_final = np.maximum(counts_final, 1)
    clipped_mean = f.sum(axis=0, where=mask) / counts_safe_final

    # Where fewer than min_fraction of frames survived, fall back to the
    # simple mean over all frames (no rejection) to preserve linearity.
    counts_base = base_valid.sum(axis=0)
    frac = counts_final.astype("float32") / np.maximum(counts_base.astype("float32"), 1.0)
    fallback_mask = frac < float(cfg.min_fraction)
    if np.any(fallback_mask):
        counts_safe_base = np.maximum(counts_base, 1)
        full_mean = f.sum(axis=0, where=base_valid) / counts_safe_base
        # Broadcast-safe assignment: only replace positions that failed.
        clipped_mean = np.where(fallback_mask, full_mean, clipped_mean)

    rejected = (~mask & base_valid).sum(dtype=np.int64)
    stats: Dict[str, Any] = {
        "frames": n_frames,
        "iterations": int(cfg.max_iters),
        "kept_fraction": float(mask.sum(dtype=np.int64) / total_samples),
        "rejected_fraction": float(rejected / total_samples),
        "min_fraction": float(cfg.min_fraction),
        "sigma_low": float(cfg.sigma_low),
        "sigma_high": float(cfg.sigma_high),
    }

    return clipped_mean.astype("float32", copy=False), mask, stats


def simple_mean_stack_nd(frames: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute an unweighted mean over frames along axis 0.

    This is used for the default "linear" stacking path where no rejection
    is desired.
    """

    f = np.asarray(frames).astype("float32", copy=False)
    if f.ndim < 2:
        raise ValueError("simple_mean_stack_nd expects an array of shape (N, ...) with N>=1")
    if valid_mask is None:
        return f.mean(axis=0).astype("float32", copy=False)
    vm = np.asarray(valid_mask).astype(bool, copy=False)
    if vm.shape != f.shape:
        raise ValueError(f"valid_mask shape {vm.shape} does not match frames shape {f.shape}")
    counts = vm.sum(axis=0)
    counts_safe = np.maximum(counts, 1)
    out = f.sum(axis=0, where=vm) / counts_safe
    return out.astype("float32", copy=False)
