#!/usr/bin/env python3
"""Create stage-wise PCC diagnostics artifacts for one run.

Compares internal pipeline stages only (no Siril dependency):
  - solve (stacked_rgb_solve.fits)
  - bge   (stacked_rgb_bge.fits)
  - pcc   (stacked_rgb_pcc.fits)

Outputs (default in <run_dir>/artifacts/pcc_diagnostics):
  - background_mask.fits
  - log_rg_solve.fits
  - log_rg_bge.fits
  - log_rg_pcc.fits
  - log_bg_solve.fits
  - log_bg_bge.fits
  - log_bg_pcc.fits
  - delta_log_rg_bge_minus_solve.fits
  - delta_log_rg_pcc_minus_bge.fits
  - delta_log_bg_bge_minus_solve.fits
  - delta_log_bg_pcc_minus_bge.fits
  - summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    q = max(0.0, min(1.0, q))
    idx = int(q * (values.size - 1))
    part = np.partition(values, idx)
    return float(part[idx])


def _stats(a: np.ndarray) -> dict[str, Any]:
    v = a[np.isfinite(a)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "p01": float(np.percentile(v, 1)),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
    }


def _build_background_mask(rgb: np.ndarray) -> np.ndarray:
    lum = np.median(rgb, axis=0)
    valid = np.isfinite(lum)
    if np.count_nonzero(valid) < 4096:
        return valid

    p60 = _quantile(lum[valid], 0.60)
    gx = np.zeros_like(lum, dtype=np.float32)
    gy = np.zeros_like(lum, dtype=np.float32)
    gx[:, 1:-1] = np.abs(lum[:, 2:] - lum[:, :-2])
    gy[1:-1, :] = np.abs(lum[2:, :] - lum[:-2, :])
    grad = gx + gy
    g70 = _quantile(grad[valid], 0.70)

    return valid & (lum <= p60) & (grad <= g70)


def _log_chroma(rgb: np.ndarray, eps: float = 1.0e-9) -> tuple[np.ndarray, np.ndarray]:
    r = rgb[0].astype(np.float64, copy=False)
    g = rgb[1].astype(np.float64, copy=False)
    b = rgb[2].astype(np.float64, copy=False)
    good = np.isfinite(r) & np.isfinite(g) & np.isfinite(b) & (r > 0.0) & (g > 0.0) & (b > 0.0)

    log_rg = np.full(r.shape, np.nan, dtype=np.float32)
    log_bg = np.full(r.shape, np.nan, dtype=np.float32)
    log_rg[good] = np.log((r[good] + eps) / (g[good] + eps)).astype(np.float32)
    log_bg[good] = np.log((b[good] + eps) / (g[good] + eps)).astype(np.float32)
    return log_rg, log_bg


def _write_fits(path: Path, data: np.ndarray, header: fits.Header | None = None) -> None:
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(path, overwrite=True)


def _pct_change(base: float, value: float) -> float:
    if not np.isfinite(base) or not np.isfinite(value) or abs(base) < 1.0e-20:
        return float("nan")
    return float((value / base - 1.0) * 100.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create stage-wise PCC diagnostics artifacts")
    ap.add_argument("run_dir", type=Path, help="Path to one run directory")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for diagnostics (default: <run_dir>/artifacts/pcc_diagnostics)",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    outputs = run_dir / "outputs"
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (run_dir / "artifacts" / "pcc_diagnostics")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    p_solve = outputs / "stacked_rgb_solve.fits"
    p_bge = outputs / "stacked_rgb_bge.fits"
    p_pcc = outputs / "stacked_rgb_pcc.fits"

    for p in (p_solve, p_bge, p_pcc):
        if not p.exists():
            raise FileNotFoundError(f"Missing FITS file: {p}")

    d_solve = fits.getdata(p_solve).astype(np.float32, copy=False)
    d_bge = fits.getdata(p_bge).astype(np.float32, copy=False)
    d_pcc = fits.getdata(p_pcc).astype(np.float32, copy=False)

    if d_solve.shape != d_bge.shape or d_solve.shape != d_pcc.shape:
        raise RuntimeError(
            f"Shape mismatch: solve={d_solve.shape} bge={d_bge.shape} pcc={d_pcc.shape}"
        )
    if d_solve.ndim != 3 or d_solve.shape[0] != 3:
        raise RuntimeError(f"Expected RGB cube with shape (3,H,W), got {d_solve.shape}")

    hdr = fits.getheader(p_solve)

    bg_mask = _build_background_mask(d_solve)
    log_rg_solve, log_bg_solve = _log_chroma(d_solve)
    log_rg_bge, log_bg_bge = _log_chroma(d_bge)
    log_rg_pcc, log_bg_pcc = _log_chroma(d_pcc)

    delta_log_rg_bge_minus_solve = (log_rg_bge - log_rg_solve).astype(np.float32)
    delta_log_rg_pcc_minus_bge = (log_rg_pcc - log_rg_bge).astype(np.float32)
    delta_log_bg_bge_minus_solve = (log_bg_bge - log_bg_solve).astype(np.float32)
    delta_log_bg_pcc_minus_bge = (log_bg_pcc - log_bg_bge).astype(np.float32)

    _write_fits(out_dir / "background_mask.fits", bg_mask.astype(np.uint8), hdr)
    _write_fits(out_dir / "log_rg_solve.fits", log_rg_solve, hdr)
    _write_fits(out_dir / "log_rg_bge.fits", log_rg_bge, hdr)
    _write_fits(out_dir / "log_rg_pcc.fits", log_rg_pcc, hdr)
    _write_fits(out_dir / "log_bg_solve.fits", log_bg_solve, hdr)
    _write_fits(out_dir / "log_bg_bge.fits", log_bg_bge, hdr)
    _write_fits(out_dir / "log_bg_pcc.fits", log_bg_pcc, hdr)
    _write_fits(
        out_dir / "delta_log_rg_bge_minus_solve.fits",
        delta_log_rg_bge_minus_solve,
        hdr,
    )
    _write_fits(
        out_dir / "delta_log_rg_pcc_minus_bge.fits",
        delta_log_rg_pcc_minus_bge,
        hdr,
    )
    _write_fits(
        out_dir / "delta_log_bg_bge_minus_solve.fits",
        delta_log_bg_bge_minus_solve,
        hdr,
    )
    _write_fits(
        out_dir / "delta_log_bg_pcc_minus_bge.fits",
        delta_log_bg_pcc_minus_bge,
        hdr,
    )

    bg_valid_rg = bg_mask & np.isfinite(log_rg_solve) & np.isfinite(log_rg_bge) & np.isfinite(log_rg_pcc)
    bg_valid_bg = bg_mask & np.isfinite(log_bg_solve) & np.isfinite(log_bg_bge) & np.isfinite(log_bg_pcc)

    solve_rg = _stats(log_rg_solve[bg_valid_rg])
    bge_rg = _stats(log_rg_bge[bg_valid_rg])
    pcc_rg = _stats(log_rg_pcc[bg_valid_rg])
    solve_bg = _stats(log_bg_solve[bg_valid_bg])
    bge_bg = _stats(log_bg_bge[bg_valid_bg])
    pcc_bg = _stats(log_bg_pcc[bg_valid_bg])

    summary = {
        "run_dir": str(run_dir),
        "inputs": {
            "solve": str(p_solve),
            "bge": str(p_bge),
            "pcc": str(p_pcc),
        },
        "shape": list(d_solve.shape),
        "background_mask_fraction": float(np.mean(bg_mask)),
        "log_rg_background": {
            "solve": solve_rg,
            "bge": bge_rg,
            "pcc": pcc_rg,
            "changes_pct": {
                "solve_to_bge_std": _pct_change(solve_rg.get("std", float("nan")), bge_rg.get("std", float("nan"))),
                "bge_to_pcc_std": _pct_change(bge_rg.get("std", float("nan")), pcc_rg.get("std", float("nan"))),
                "solve_to_pcc_std": _pct_change(solve_rg.get("std", float("nan")), pcc_rg.get("std", float("nan"))),
            },
        },
        "log_bg_background": {
            "solve": solve_bg,
            "bge": bge_bg,
            "pcc": pcc_bg,
            "changes_pct": {
                "solve_to_bge_std": _pct_change(solve_bg.get("std", float("nan")), bge_bg.get("std", float("nan"))),
                "bge_to_pcc_std": _pct_change(bge_bg.get("std", float("nan")), pcc_bg.get("std", float("nan"))),
                "solve_to_pcc_std": _pct_change(solve_bg.get("std", float("nan")), pcc_bg.get("std", float("nan"))),
            },
        },
        "delta_log_rg_bge_minus_solve_background": _stats(delta_log_rg_bge_minus_solve[bg_valid_rg]),
        "delta_log_rg_pcc_minus_bge_background": _stats(delta_log_rg_pcc_minus_bge[bg_valid_rg]),
        "delta_log_bg_bge_minus_solve_background": _stats(delta_log_bg_bge_minus_solve[bg_valid_bg]),
        "delta_log_bg_pcc_minus_bge_background": _stats(delta_log_bg_pcc_minus_bge[bg_valid_bg]),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
