from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .fits_utils import read_fits_float


def build_master_mean(paths: list[Path]) -> tuple[np.ndarray, Any] | None:
    if not paths:
        return None
    acc: np.ndarray | None = None
    hdr0: Any = None
    n = 0
    for p in paths:
        try:
            a, hdr = read_fits_float(p)
        except Exception:
            continue
        if acc is None:
            acc = np.zeros_like(a, dtype=np.float32)
            hdr0 = hdr
        if acc.shape != a.shape:
            continue
        acc += a.astype("float32", copy=False)
        n += 1
    if acc is None or n <= 0:
        return None
    return (acc / float(n)).astype("float32", copy=False), hdr0


def bias_correct_dark(
    dark_master: tuple[np.ndarray, Any] | None,
    bias_master: tuple[np.ndarray, Any] | None,
) -> tuple[np.ndarray, Any] | None:
    if dark_master is None:
        return None
    if bias_master is None:
        return dark_master
    dm, hdr = dark_master
    bm, _ = bias_master
    if dm.shape != bm.shape:
        return dark_master
    return (dm - bm).astype("float32", copy=False), hdr


def prepare_flat(
    flat_master: tuple[np.ndarray, Any] | None,
    bias_master: tuple[np.ndarray, Any] | None,
    dark_master: tuple[np.ndarray, Any] | None,
) -> tuple[np.ndarray, Any] | None:
    if flat_master is None:
        return None
    arr, hdr = flat_master
    out = arr.astype("float32", copy=False)

    if bias_master is not None and out.shape == bias_master[0].shape:
        out = out - bias_master[0]
    if dark_master is not None and out.shape == dark_master[0].shape:
        out = out - dark_master[0]

    med = float(np.median(out))
    if np.isfinite(med) and med != 0.0:
        out = out / med

    return out.astype("float32", copy=False), hdr


def apply_calibration(
    img: np.ndarray,
    bias_arr: np.ndarray | None,
    dark_arr: np.ndarray | None,
    flat_arr: np.ndarray | None,
    denom_eps: float = 1e-6,
) -> np.ndarray:
    cal = img.astype("float32", copy=False)

    if bias_arr is not None and cal.shape == bias_arr.shape:
        cal = cal - bias_arr
    if dark_arr is not None and cal.shape == dark_arr.shape:
        cal = cal - dark_arr
    if flat_arr is not None and cal.shape == flat_arr.shape:
        denom = flat_arr
        denom = np.where(np.abs(denom) < float(denom_eps), 1.0, denom)
        cal = cal / denom

    return cal.astype("float32", copy=False)
