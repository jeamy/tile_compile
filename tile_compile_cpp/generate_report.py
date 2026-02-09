#!/usr/bin/env python3
"""
Generate an HTML artifacts report with matplotlib charts for a C++ tile_compile run.

Usage:
    python generate_report.py /path/to/runs/<run_id>

Produces:  <run_dir>/artifacts/report.html  +  report.css  +  *.png charts
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
except ImportError:
    plt = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _basic_stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def _pct(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{100.0 * x:.1f}%"


def _escape_html(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _fig_path(artifacts_dir: Path, name: str) -> Path:
    return artifacts_dir / name


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def _plot_timeseries(vals: list[float], title: str, ylabel: str, out: Path,
                     color: str = "#7aa2f7", *, median_line: bool = True) -> bool:
    if plt is None or not vals:
        return False
    a = np.asarray(vals, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 3), dpi=150)
    ax.plot(np.arange(len(a)), a, lw=0.8, color=color, alpha=0.9)
    if median_line and a.size > 1:
        med = float(np.median(a[np.isfinite(a)])) if np.any(np.isfinite(a)) else 0
        ax.axhline(med, color="#ff6e6e", lw=0.7, ls="--", label=f"median={med:.4g}")
        ax.legend(fontsize=8)
    ax.set_xlabel("frame index", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_histogram(vals: list[float], title: str, xlabel: str, out: Path,
                    color: str = "#7aa2f7", bins: int = 60) -> bool:
    if plt is None or not vals:
        return False
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return False
    lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    if hi <= lo:
        lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo:
        return False
    fig, ax = plt.subplots(figsize=(5.5, 3), dpi=150)
    ax.hist(a, bins=bins, range=(lo, hi), color=color, alpha=0.85, edgecolor="none")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_multi_timeseries(series: dict[str, list[float]], title: str, ylabel: str, out: Path) -> bool:
    if plt is None or not series:
        return False
    colors = {"mono": "#7aa2f7", "r": "#ff6b6b", "g": "#50fa7b", "b": "#6c9eff",
              "R": "#ff6b6b", "G": "#50fa7b", "B": "#6c9eff"}
    fig, ax = plt.subplots(figsize=(7, 3), dpi=150)
    for name, vals in series.items():
        if not vals:
            continue
        a = np.asarray(vals, dtype=np.float64)
        c = colors.get(name, "#7aa2f7")
        ax.plot(np.arange(len(a)), a, lw=0.8, color=c, alpha=0.85, label=name)
    ax.set_xlabel("frame index", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_heatmap_2d(values: list[float], nx: int, ny: int, title: str, out: Path,
                     cmap: str = "viridis") -> bool:
    if plt is None or not values or nx <= 0 or ny <= 0:
        return False
    n = nx * ny
    if len(values) < n:
        return False
    arr = np.asarray(values[:n], dtype=np.float64).reshape(ny, nx)
    fig, ax = plt.subplots(figsize=(max(4, nx * 0.4), max(3, ny * 0.35)), dpi=150)
    im = ax.imshow(arr, cmap=cmap, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("tile x", fontsize=9)
    ax.set_ylabel("tile y", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_spatial_tile_heatmap(
    tiles: list[dict], values: list[float],
    img_w: int, img_h: int,
    title: str, out: Path, cmap: str = "viridis",
    label: str = "value", show_grid: bool = True,
) -> bool:
    """Paint each tile at its real (x,y) position on the full image canvas."""
    if plt is None or not tiles or not values or img_w <= 0 or img_h <= 0:
        return False
    n = min(len(tiles), len(values))
    canvas = np.full((img_h, img_w), np.nan, dtype=np.float64)
    for i in range(n):
        t = tiles[i]
        x0, y0 = int(t["x"]), int(t["y"])
        tw, th = int(t["width"]), int(t["height"])
        x1 = min(x0 + tw, img_w)
        y1 = min(y0 + th, img_h)
        canvas[y0:y1, x0:x1] = values[i]

    vmin = float(np.nanmin(canvas)) if np.any(np.isfinite(canvas)) else 0
    vmax = float(np.nanmax(canvas)) if np.any(np.isfinite(canvas)) else 1

    aspect = img_h / max(1, img_w)
    fw = 8
    fh = max(3, fw * aspect)
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=150)
    im = ax.imshow(canvas, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, img_w, img_h, 0], interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.75, label=label)

    if show_grid:
        for i in range(n):
            t = tiles[i]
            rect = plt.Rectangle((t["x"], t["y"]), t["width"], t["height"],
                                 linewidth=0.3, edgecolor="white", facecolor="none", alpha=0.25)
            ax.add_patch(rect)

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_xlabel("x (px)", fontsize=9)
    ax.set_ylabel("y (px)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_spatial_tile_multi(
    tiles: list[dict], value_sets: list[tuple[str, list[float], str]],
    img_w: int, img_h: int,
    title: str, out: Path, show_grid: bool = True,
) -> bool:
    """Multiple spatial heatmaps side-by-side. value_sets = [(subtitle, values, cmap), ...]"""
    if plt is None or not tiles or not value_sets or img_w <= 0 or img_h <= 0:
        return False
    ncols = len(value_sets)
    aspect = img_h / max(1, img_w)
    fw = min(6, 18 / ncols)
    fh = max(2.5, fw * aspect)
    fig, axes = plt.subplots(1, ncols, figsize=(fw * ncols + 1, fh), dpi=150)
    if ncols == 1:
        axes = [axes]

    n = len(tiles)
    for idx, (subtitle, values, cmap) in enumerate(value_sets):
        ax = axes[idx]
        canvas = np.full((img_h, img_w), np.nan, dtype=np.float64)
        m = min(n, len(values))
        for i in range(m):
            t = tiles[i]
            x0, y0 = int(t["x"]), int(t["y"])
            tw, th = int(t["width"]), int(t["height"])
            canvas[y0:min(y0+th, img_h), x0:min(x0+tw, img_w)] = values[i]

        vmin = float(np.nanmin(canvas)) if np.any(np.isfinite(canvas)) else 0
        vmax = float(np.nanmax(canvas)) if np.any(np.isfinite(canvas)) else 1
        im = ax.imshow(canvas, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[0, img_w, img_h, 0], interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.7)
        if show_grid:
            for i in range(m):
                t = tiles[i]
                rect = plt.Rectangle((t["x"], t["y"]), t["width"], t["height"],
                                     linewidth=0.2, edgecolor="white", facecolor="none", alpha=0.2)
                ax.add_patch(rect)
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_title(subtitle, fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_bar(labels: list[str], values: list[float], title: str, ylabel: str, out: Path,
              colors: list[str] | None = None) -> bool:
    if plt is None or not values:
        return False
    fig, ax = plt.subplots(figsize=(max(4, len(values) * 0.8), 3), dpi=150)
    x = np.arange(len(values))
    c = colors if colors else ["#7aa2f7"] * len(values)
    ax.bar(x, values, color=c, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_warp_scatter(warps: list[dict], ccs: list[float], title: str, out: Path) -> bool:
    if plt is None or not warps:
        return False
    tx = [w.get("tx", 0) for w in warps]
    ty = [w.get("ty", 0) for w in warps]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=150)

    # Translation scatter
    ax = axes[0]
    sc = ax.scatter(tx, ty, c=np.arange(len(tx)), cmap="plasma", s=12, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="frame #", shrink=0.8)
    ax.set_xlabel("tx (px)", fontsize=9)
    ax.set_ylabel("ty (px)", fontsize=9)
    ax.set_title("Translation scatter", fontsize=10)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)

    # Translation timeseries
    ax = axes[1]
    ax.plot(tx, lw=0.8, color="#ff6b6b", label="tx", alpha=0.85)
    ax.plot(ty, lw=0.8, color="#6c9eff", label="ty", alpha=0.85)
    ax.set_xlabel("frame index", fontsize=9)
    ax.set_ylabel("shift (px)", fontsize=9)
    ax.set_title("Translation over time", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # CC timeseries
    ax = axes[2]
    if ccs:
        ax.plot(ccs, lw=0.8, color="#50fa7b", alpha=0.85)
        ax.axhline(float(np.median(ccs)), color="#ff6e6e", lw=0.7, ls="--")
    ax.set_xlabel("frame index", fontsize=9)
    ax.set_ylabel("correlation", fontsize=9)
    ax.set_title("Registration CC", fontsize=10)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def _gen_normalization(artifacts_dir: Path, norm: dict) -> tuple[list[str], list[str]]:
    """Generate normalization charts. Returns (png_files, eval_lines)."""
    pngs: list[str] = []
    evals: list[str] = []

    mode = norm.get("mode", "MONO")
    evals.append(f"mode: {mode}")

    b_mono = norm.get("B_mono", [])
    b_r = norm.get("B_r", [])
    b_g = norm.get("B_g", [])
    b_b = norm.get("B_b", [])

    if mode == "OSC" and (b_r or b_g or b_b):
        fn = "norm_background_osc.png"
        if _plot_multi_timeseries({"R": b_r, "G": b_g, "B": b_b},
                                  "Per-channel background level", "background", _fig_path(artifacts_dir, fn)):
            pngs.append(fn)
        for name, vals in [("R", b_r), ("G", b_g), ("B", b_b)]:
            s = _basic_stats(vals)
            if s["n"]:
                evals.append(f"  {name}: median={s['median']:.4g}, std={s['std']:.4g}, range=[{s['min']:.4g}, {s['max']:.4g}]")
    elif b_mono:
        fn = "norm_background_mono.png"
        if _plot_timeseries(b_mono, "Background level (mono)", "background", _fig_path(artifacts_dir, fn)):
            pngs.append(fn)
        s = _basic_stats(b_mono)
        if s["n"]:
            evals.append(f"  mono: median={s['median']:.4g}, std={s['std']:.4g}")

    return pngs, evals


def _gen_global_metrics(artifacts_dir: Path, gm: dict) -> tuple[list[str], list[str]]:
    pngs: list[str] = []
    evals: list[str] = []

    metrics = gm.get("metrics", [])
    if not metrics:
        evals.append("no metrics data")
        return pngs, evals

    bg = [m.get("background", 0) for m in metrics]
    noise = [m.get("noise", 0) for m in metrics]
    grad = [m.get("gradient_energy", 0) for m in metrics]
    gw = [m.get("global_weight", 0) for m in metrics]
    qs = [m.get("quality_score", 0) for m in metrics]

    evals.append(f"frames: {len(metrics)}")
    weights_cfg = gm.get("weights", {})
    evals.append(f"weights: bg={weights_cfg.get('background', '?')}, noise={weights_cfg.get('noise', '?')}, grad={weights_cfg.get('gradient', '?')}")

    # Background timeseries
    fn = "global_background.png"
    if _plot_timeseries(bg, "Frame background level", "background", _fig_path(artifacts_dir, fn)):
        pngs.append(fn)

    # Noise timeseries
    fn = "global_noise.png"
    if _plot_timeseries(noise, "Frame noise level", "noise", _fig_path(artifacts_dir, fn), color="#ff6b6b"):
        pngs.append(fn)

    # Gradient energy timeseries
    fn = "global_gradient.png"
    if _plot_timeseries(grad, "Frame gradient energy", "gradient energy", _fig_path(artifacts_dir, fn), color="#50fa7b"):
        pngs.append(fn)

    # Global weights timeseries + histogram
    fn = "global_weight_timeseries.png"
    if _plot_timeseries(gw, "Global frame weight G(f)", "weight", _fig_path(artifacts_dir, fn), color="#ffb86c"):
        pngs.append(fn)

    fn = "global_weight_hist.png"
    if _plot_histogram(gw, "Global weight distribution", "weight", _fig_path(artifacts_dir, fn), color="#ffb86c"):
        pngs.append(fn)

    # --- Siril-style per-frame star metrics ---
    fwhm = [m.get("fwhm", 0) for m in metrics]
    wfwhm = [m.get("wfwhm", 0) for m in metrics]
    roundness = [m.get("roundness", 0) for m in metrics]
    star_count = [m.get("star_count", 0) for m in metrics]

    # FWHM timeseries
    fn = "global_fwhm.png"
    if _plot_timeseries(fwhm, "FWHM per frame", "FWHM (px)", _fig_path(artifacts_dir, fn), color="#bd93f9"):
        pngs.append(fn)

    # wFWHM timeseries (weighted by star count)
    fn = "global_wfwhm.png"
    if _plot_timeseries(wfwhm, "Weighted FWHM per frame (wFWHM)", "wFWHM (px)", _fig_path(artifacts_dir, fn), color="#ff79c6"):
        pngs.append(fn)

    # Roundness timeseries
    fn = "global_roundness.png"
    if _plot_timeseries(roundness, "Star roundness per frame (FWHMy/FWHMx)", "roundness", _fig_path(artifacts_dir, fn), color="#8be9fd"):
        pngs.append(fn)

    # Star count timeseries
    fn = "global_star_count.png"
    if _plot_timeseries(star_count, "Detected stars per frame", "stars", _fig_path(artifacts_dir, fn), color="#f1fa8c"):
        pngs.append(fn)

    # FWHM vs Roundness scatter (Siril-style)
    if plt is not None and fwhm and roundness:
        fn = "global_fwhm_vs_roundness.png"
        a_fwhm = np.asarray(fwhm, dtype=np.float64)
        a_round = np.asarray(roundness, dtype=np.float64)
        mask = (a_fwhm > 0) & (a_round > 0) & np.isfinite(a_fwhm) & np.isfinite(a_round)
        if np.sum(mask) > 2:
            fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
            sc = ax.scatter(a_fwhm[mask], a_round[mask],
                            c=np.arange(len(a_fwhm))[mask], cmap="plasma",
                            s=14, alpha=0.8, edgecolors="none")
            fig.colorbar(sc, ax=ax, label="frame #", shrink=0.8)
            ax.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5)
            ax.set_xlabel("FWHM (px)", fontsize=9)
            ax.set_ylabel("Roundness (FWHMy/FWHMx)", fontsize=9)
            ax.set_title("FWHM vs Roundness", fontsize=10)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            fig.savefig(_fig_path(artifacts_dir, fn), bbox_inches="tight")
            plt.close(fig)
            pngs.append(fn)

    # Stats
    s = _basic_stats(gw)
    if s["n"]:
        evals.append(f"G(f): median={s['median']:.4g}, min={s['min']:.4g}, max={s['max']:.4g}, std={s['std']:.4g}")
        if s["min"] > 0:
            evals.append(f"max/min ratio: {s['max'] / s['min']:.2f}")
            low = sum(1 for x in gw if x < s["median"] * 0.2)
            evals.append(f"frames with very low weight (<0.2×median): {low}")
            if s["max"] / s["min"] > 50:
                evals.append("WARNING: extremely wide weight distribution")

    s_fwhm = _basic_stats([f for f in fwhm if f > 0])
    if s_fwhm["n"]:
        evals.append(f"FWHM: median={s_fwhm['median']:.2f} px, range=[{s_fwhm['min']:.2f}, {s_fwhm['max']:.2f}]")
    s_round = _basic_stats([r for r in roundness if r > 0])
    if s_round["n"]:
        evals.append(f"roundness: median={s_round['median']:.3f}, range=[{s_round['min']:.3f}, {s_round['max']:.3f}]")
        if s_round["median"] < 0.7:
            evals.append("WARNING: low roundness — possible tracking/guiding issue or elongated stars")
    s_stars = _basic_stats([float(c) for c in star_count if c > 0])
    if s_stars["n"]:
        evals.append(f"star count: median={s_stars['median']:.0f}, range=[{s_stars['min']:.0f}, {s_stars['max']:.0f}]")

    return pngs, evals


def _gen_tile_grid(artifacts_dir: Path, tg: dict) -> tuple[list[str], list[str]]:
    pngs: list[str] = []
    evals: list[str] = []

    evals.append(f"image: {tg.get('image_width', '?')}×{tg.get('image_height', '?')}")
    evals.append(f"num_tiles: {tg.get('num_tiles', '?')}")
    evals.append(f"tile_size: {tg.get('uniform_tile_size', tg.get('seeing_tile_size', '?'))}")
    evals.append(f"seeing_fwhm_median: {tg.get('seeing_fwhm_median', '?')}")
    evals.append(f"overlap: {tg.get('overlap_fraction', '?')}")
    evals.append(f"stride: {tg.get('stride_px', '?')} px")

    # Draw tile grid overlay
    tiles = tg.get("tiles", [])
    w = tg.get("image_width", 0)
    h = tg.get("image_height", 0)
    if plt is not None and tiles and w > 0 and h > 0:
        fn = "tile_grid_overlay.png"
        fig, ax = plt.subplots(figsize=(6, 6 * h / max(1, w)), dpi=150)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        for t in tiles:
            rect = plt.Rectangle((t["x"], t["y"]), t["width"], t["height"],
                                 linewidth=0.4, edgecolor="#7aa2f7", facecolor="none", alpha=0.6)
            ax.add_patch(rect)
        ax.set_xlabel("x (px)", fontsize=9)
        ax.set_ylabel("y (px)", fontsize=9)
        ax.set_title(f"Tile grid ({len(tiles)} tiles)", fontsize=10)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(_fig_path(artifacts_dir, fn), bbox_inches="tight")
        plt.close(fig)
        pngs.append(fn)

    return pngs, evals


def _gen_registration(artifacts_dir: Path, reg: dict) -> tuple[list[str], list[str]]:
    pngs: list[str] = []
    evals: list[str] = []

    n = reg.get("num_frames", 0)
    scale = reg.get("scale", "?")
    ref = reg.get("ref_frame", 0)
    evals.append(f"frames: {n}, scale: {scale}, ref_frame: {ref}")

    warps = reg.get("warps", [])
    ccs = reg.get("cc", [])

    fn = "registration_overview.png"
    if _plot_warp_scatter(warps, ccs, "Registration", _fig_path(artifacts_dir, fn)):
        pngs.append(fn)

    # CC histogram
    fn = "registration_cc_hist.png"
    if _plot_histogram(ccs, "Registration correlation coefficient", "CC", _fig_path(artifacts_dir, fn), color="#50fa7b"):
        pngs.append(fn)

    # Rotation analysis
    if warps:
        rotations = []
        for w in warps:
            a00 = w.get("a00", 1)
            a01 = w.get("a01", 0)
            angle_deg = math.degrees(math.atan2(a01, a00))
            rotations.append(angle_deg)
        fn = "registration_rotation.png"
        if _plot_timeseries(rotations, "Frame rotation angle", "angle (deg)", _fig_path(artifacts_dir, fn), color="#ff6b6b"):
            pngs.append(fn)

        # Scale analysis
        scales = []
        for w in warps:
            a00 = w.get("a00", 1)
            a01 = w.get("a01", 0)
            s = math.sqrt(a00 ** 2 + a01 ** 2)
            scales.append(s)
        fn = "registration_scale.png"
        if _plot_timeseries(scales, "Frame scale factor", "scale", _fig_path(artifacts_dir, fn), color="#ffb86c"):
            pngs.append(fn)

    # Stats
    if ccs:
        s = _basic_stats(ccs)
        evals.append(f"CC: median={s['median']:.4g}, min={s['min']:.4g}, max={s['max']:.4g}")
        bad = sum(1 for c in ccs if c < 0.5)
        if bad:
            evals.append(f"WARNING: {bad} frames with CC < 0.5")

    if warps:
        tx = [w.get("tx", 0) for w in warps]
        ty = [w.get("ty", 0) for w in warps]
        s_tx = _basic_stats(tx)
        s_ty = _basic_stats(ty)
        evals.append(f"tx: median={s_tx['median']:.2f}, range=[{s_tx['min']:.2f}, {s_tx['max']:.2f}]")
        evals.append(f"ty: median={s_ty['median']:.2f}, range=[{s_ty['min']:.2f}, {s_ty['max']:.2f}]")
        max_shift = max(abs(s_tx["max"] - s_tx["min"]), abs(s_ty["max"] - s_ty["min"]))
        evals.append(f"max shift span: {max_shift:.1f} px")

    return pngs, evals


def _gen_local_metrics(artifacts_dir: Path, lm: dict, tg: dict) -> tuple[list[str], list[str]]:
    pngs: list[str] = []
    evals: list[str] = []

    n_frames = lm.get("num_frames", 0)
    n_tiles = lm.get("num_tiles", 0)
    evals.append(f"frames: {n_frames}, tiles: {n_tiles}")

    tile_metrics = lm.get("tile_metrics", [])
    if not tile_metrics:
        evals.append("no tile metrics data")
        return pngs, evals

    tiles = tg.get("tiles", [])
    img_w = tg.get("image_width", 0)
    img_h = tg.get("image_height", 0)

    # Average metrics across frames
    all_fwhm = [[] for _ in range(n_tiles)]
    all_quality = [[] for _ in range(n_tiles)]
    all_weight = [[] for _ in range(n_tiles)]
    all_contrast = [[] for _ in range(n_tiles)]
    all_star_count = [[] for _ in range(n_tiles)]

    for frame_tiles in tile_metrics:
        for ti, tm in enumerate(frame_tiles):
            if ti >= n_tiles:
                break
            all_fwhm[ti].append(tm.get("fwhm", 0))
            all_quality[ti].append(tm.get("quality_score", 0))
            all_weight[ti].append(tm.get("local_weight", 0))
            all_contrast[ti].append(tm.get("contrast", 0))
            all_star_count[ti].append(tm.get("star_count", 0))

    mean_fwhm = [float(np.mean(v)) if v else 0 for v in all_fwhm]
    mean_quality = [float(np.mean(v)) if v else 0 for v in all_quality]
    mean_weight = [float(np.mean(v)) if v else 0 for v in all_weight]
    mean_contrast = [float(np.mean(v)) if v else 0 for v in all_contrast]
    var_quality = [float(np.var(v)) if v else 0 for v in all_quality]
    var_weight = [float(np.var(v)) if v else 0 for v in all_weight]
    mean_stars = [float(np.mean(v)) if v else 0 for v in all_star_count]

    use_spatial = tiles and img_w > 0 and img_h > 0

    # Spatial FWHM + Quality combined
    fn = "local_fwhm_quality_spatial.png"
    if use_spatial and _plot_spatial_tile_multi(
        tiles, [
            ("Mean FWHM", mean_fwhm, "inferno"),
            ("Mean quality score", mean_quality, "viridis"),
            ("Quality variance", var_quality, "magma"),
        ], img_w, img_h, "Local Metrics — Spatial", _fig_path(artifacts_dir, fn)):
        pngs.append(fn)

    # Spatial weight + contrast + stars combined
    fn = "local_weight_contrast_spatial.png"
    if use_spatial and _plot_spatial_tile_multi(
        tiles, [
            ("Mean local weight", mean_weight, "plasma"),
            ("Weight variance", var_weight, "magma"),
            ("Mean contrast", mean_contrast, "cividis"),
        ], img_w, img_h, "Local Weights & Contrast — Spatial", _fig_path(artifacts_dir, fn)):
        pngs.append(fn)

    # Individual large spatial heatmaps
    fn = "local_fwhm_spatial.png"
    if use_spatial and _plot_spatial_tile_heatmap(
        tiles, mean_fwhm, img_w, img_h, "Mean FWHM per tile",
        _fig_path(artifacts_dir, fn), cmap="inferno", label="FWHM (px)"):
        pngs.append(fn)

    fn = "local_weight_spatial.png"
    if use_spatial and _plot_spatial_tile_heatmap(
        tiles, mean_weight, img_w, img_h, "Mean local weight per tile",
        _fig_path(artifacts_dir, fn), cmap="plasma", label="weight"):
        pngs.append(fn)

    fn = "local_stars_spatial.png"
    if use_spatial and _plot_spatial_tile_heatmap(
        tiles, mean_stars, img_w, img_h, "Mean star count per tile",
        _fig_path(artifacts_dir, fn), cmap="YlGnBu", label="stars"):
        pngs.append(fn)

    # Tile type map (STAR=1, STRUCTURE=0)
    if tile_metrics and tiles and use_spatial:
        type_vals = []
        for ti, tm in enumerate(tile_metrics[0]):
            type_vals.append(1.0 if tm.get("tile_type", "") == "STAR" else 0.0)
        fn = "local_tile_type_map.png"
        if _plot_spatial_tile_heatmap(
            tiles, type_vals, img_w, img_h, "Tile type (yellow=STAR, purple=STRUCTURE)",
            _fig_path(artifacts_dir, fn), cmap="viridis", label="type"):
            pngs.append(fn)

    # Per-frame mean quality timeseries
    per_frame_q = []
    per_frame_w = []
    for frame_tiles in tile_metrics:
        qs = [tm.get("quality_score", 0) for tm in frame_tiles]
        ws = [tm.get("local_weight", 0) for tm in frame_tiles]
        per_frame_q.append(float(np.mean(qs)) if qs else 0)
        per_frame_w.append(float(np.mean(ws)) if ws else 0)
    fn = "local_quality_weight_per_frame.png"
    if _plot_multi_timeseries(
        {"mean quality": per_frame_q, "mean weight": per_frame_w},
        "Per-frame tile quality & weight", "value", _fig_path(artifacts_dir, fn)):
        pngs.append(fn)

    # Stats
    s = _basic_stats(mean_fwhm)
    if s["n"]:
        evals.append(f"mean FWHM: median={s['median']:.3g}, range=[{s['min']:.3g}, {s['max']:.3g}]")
    s = _basic_stats(mean_weight)
    if s["n"]:
        evals.append(f"mean weight: median={s['median']:.3g}, range=[{s['min']:.3g}, {s['max']:.3g}]")
    s = _basic_stats(mean_stars)
    if s["n"]:
        evals.append(f"mean star count: median={s['median']:.1f}, range=[{s['min']:.0f}, {s['max']:.0f}]")

    # Tile type distribution
    type_counts: dict[str, int] = {}
    if tile_metrics:
        for tm in tile_metrics[0]:
            tt = tm.get("tile_type", "?")
            type_counts[tt] = type_counts.get(tt, 0) + 1
        for tt, cnt in type_counts.items():
            evals.append(f"  {tt}: {cnt} tiles")

    return pngs, evals


def _gen_reconstruction(artifacts_dir: Path, recon: dict, tg: dict) -> tuple[list[str], list[str]]:
    pngs: list[str] = []
    evals: list[str] = []

    n_frames = recon.get("num_frames", 0)
    n_tiles = recon.get("num_tiles", 0)
    evals.append(f"frames: {n_frames}, tiles: {n_tiles}")

    valid_counts = recon.get("tile_valid_counts", [])
    mean_cc = recon.get("tile_mean_correlations", [])
    post_contrast = recon.get("tile_post_contrast", [])
    post_bg = recon.get("tile_post_background", [])
    post_snr = recon.get("tile_post_snr_proxy", [])

    tiles = tg.get("tiles", [])
    img_w = tg.get("image_width", 0)
    img_h = tg.get("image_height", 0)
    use_spatial = tiles and img_w > 0 and img_h > 0

    # Spatial heatmaps: valid counts + CC + SNR  (combined)
    fn = "recon_spatial_overview.png"
    if use_spatial and _plot_spatial_tile_multi(
        tiles, [
            ("Valid frame count", [float(v) for v in valid_counts], "YlGn"),
            ("Mean correlation", mean_cc, "viridis"),
            ("Post-recon SNR", post_snr, "plasma"),
        ], img_w, img_h, "Tile Reconstruction — Spatial Overview", _fig_path(artifacts_dir, fn)):
        pngs.append(fn)

    # Large spatial: valid counts (frame usage per tile)
    fn = "recon_valid_counts_spatial.png"
    if use_spatial and _plot_spatial_tile_heatmap(
        tiles, [float(v) for v in valid_counts], img_w, img_h,
        "Valid frames per tile (tile usage)", _fig_path(artifacts_dir, fn),
        cmap="YlGn", label="valid frames"):
        pngs.append(fn)

    # Large spatial: mean CC
    fn = "recon_cc_spatial.png"
    if use_spatial and _plot_spatial_tile_heatmap(
        tiles, mean_cc, img_w, img_h,
        "Mean correlation per tile", _fig_path(artifacts_dir, fn),
        cmap="viridis", label="CC"):
        pngs.append(fn)

    # Large spatial: SNR
    fn = "recon_snr_spatial.png"
    if use_spatial and _plot_spatial_tile_heatmap(
        tiles, post_snr, img_w, img_h,
        "Post-reconstruction SNR per tile", _fig_path(artifacts_dir, fn),
        cmap="plasma", label="SNR"):
        pngs.append(fn)

    # Spatial: post-contrast + post-background
    fn = "recon_contrast_bg_spatial.png"
    if use_spatial and post_contrast and post_bg and _plot_spatial_tile_multi(
        tiles, [
            ("Post contrast", post_contrast, "cividis"),
            ("Post background", post_bg, "gray"),
        ], img_w, img_h, "Post-Reconstruction Contrast & Background", _fig_path(artifacts_dir, fn)):
        pngs.append(fn)

    # Histograms (always useful even without spatial)
    fn = "recon_valid_counts_hist.png"
    if _plot_histogram([float(v) for v in valid_counts], "Valid frame count distribution", "valid frames",
                       _fig_path(artifacts_dir, fn), color="#50fa7b"):
        pngs.append(fn)

    fn = "recon_cc_hist.png"
    if _plot_histogram(mean_cc, "Mean correlation distribution", "CC",
                       _fig_path(artifacts_dir, fn), color="#7aa2f7"):
        pngs.append(fn)

    fn = "recon_snr_hist.png"
    if _plot_histogram(post_snr, "Post-reconstruction SNR distribution", "SNR",
                       _fig_path(artifacts_dir, fn), color="#ffb86c"):
        pngs.append(fn)

    # Stats
    if valid_counts:
        s = _basic_stats([float(v) for v in valid_counts])
        evals.append(f"valid counts: median={s['median']:.0f}, min={s['min']:.0f}, max={s['max']:.0f}")
        low = sum(1 for v in valid_counts if v < 3)
        if low:
            evals.append(f"WARNING: {low} tiles with < 3 valid frames")
    if mean_cc:
        s = _basic_stats(mean_cc)
        evals.append(f"tile CC: median={s['median']:.4g}, min={s['min']:.4g}")
    if post_snr:
        s = _basic_stats(post_snr)
        evals.append(f"post-SNR: median={s['median']:.4g}, min={s['min']:.4g}")

    return pngs, evals


def _gen_clustering(artifacts_dir: Path, cl: dict) -> tuple[list[str], list[str]]:
    pngs: list[str] = []
    evals: list[str] = []

    n_clusters = cl.get("n_clusters", 0)
    method = cl.get("method", "?")
    k_min = cl.get("k_min", "?")
    k_max = cl.get("k_max", "?")
    evals.append(f"n_clusters: {n_clusters}, method: {method}, k_range: [{k_min}, {k_max}]")

    sizes = cl.get("cluster_sizes", [])
    labels = cl.get("cluster_labels", [])

    if sizes and n_clusters > 0:
        fn = "clustering_sizes.png"
        cluster_labels = [f"C{i}" for i in range(len(sizes))]
        colors = ["#7aa2f7", "#ff6b6b", "#50fa7b", "#ffb86c", "#bd93f9", "#f1fa8c", "#ff79c6", "#8be9fd"]
        c = [colors[i % len(colors)] for i in range(len(sizes))]
        if _plot_bar(cluster_labels, sizes, "Cluster sizes", "frames", _fig_path(artifacts_dir, fn), colors=c):
            pngs.append(fn)

        for i, sz in enumerate(sizes):
            evals.append(f"  cluster {i}: {sz} frames")

    if labels:
        fn = "clustering_labels.png"
        if _plot_timeseries(labels, "Cluster label per frame", "cluster", _fig_path(artifacts_dir, fn)):
            pngs.append(fn)

    return pngs, evals


def _gen_synthetic(artifacts_dir: Path, syn: dict) -> tuple[list[str], list[str]]:
    pngs: list[str] = []
    evals: list[str] = []

    n = syn.get("num_synthetic", 0)
    fmin = syn.get("frames_min", "?")
    fmax = syn.get("frames_max", "?")
    evals.append(f"num_synthetic: {n}")
    evals.append(f"frames range: [{fmin}, {fmax}]")

    return pngs, evals


def _gen_validation(artifacts_dir: Path, val: dict) -> tuple[list[str], list[str]]:
    pngs: list[str] = []
    evals: list[str] = []

    seeing = val.get("seeing_fwhm_median", 0)
    output = val.get("output_fwhm_median", 0)
    improvement = val.get("fwhm_improvement_percent", 0)
    fwhm_ok = val.get("fwhm_improvement_ok", None)
    tw_var = val.get("tile_weight_variance", 0)
    tw_ok = val.get("tile_weight_variance_ok", None)
    pattern_ratio = val.get("tile_pattern_ratio", None)
    pattern_ok = val.get("tile_pattern_ok", None)

    evals.append(f"seeing FWHM: {seeing:.3g}")
    evals.append(f"output FWHM: {output:.3g}")
    evals.append(f"FWHM improvement: {improvement:.1f}%  {'OK' if fwhm_ok else 'FAIL'}")
    evals.append(f"tile weight variance: {tw_var:.4g}  {'OK' if tw_ok else 'FAIL'}")
    if pattern_ratio is not None:
        evals.append(f"tile pattern ratio: {pattern_ratio:.3g}  {'OK' if pattern_ok else 'FAIL'}")

    # Summary bar chart
    if plt is not None:
        fn = "validation_summary.png"
        checks = []
        vals_bar = []
        colors_bar = []
        if fwhm_ok is not None:
            checks.append("FWHM\nimprovement")
            vals_bar.append(improvement)
            colors_bar.append("#50fa7b" if fwhm_ok else "#ff5555")
        if tw_ok is not None:
            checks.append("Tile weight\nvariance")
            vals_bar.append(tw_var * 100)
            colors_bar.append("#50fa7b" if tw_ok else "#ff5555")
        if pattern_ok is not None:
            checks.append("Tile pattern\nratio")
            vals_bar.append(pattern_ratio if pattern_ratio else 0)
            colors_bar.append("#50fa7b" if pattern_ok else "#ff5555")
        if checks:
            if _plot_bar(checks, vals_bar, "Validation checks", "value", _fig_path(artifacts_dir, fn), colors=colors_bar):
                pngs.append(fn)

    return pngs, evals


def _gen_timeline(artifacts_dir: Path, events: list[dict]) -> tuple[list[str], list[str]]:
    """Generate pipeline timeline chart from events."""
    pngs: list[str] = []
    evals: list[str] = []

    phase_times: list[tuple[str, float]] = []
    phase_starts: dict[str, str] = {}

    for ev in events:
        t = ev.get("type", "")
        ts = ev.get("ts", ev.get("timestamp", ""))
        pn = ev.get("phase_name", "")
        if t == "phase_start" and pn:
            phase_starts[pn] = ts
        elif t == "phase_end" and pn:
            start_ts = phase_starts.get(pn, "")
            if start_ts and ts:
                try:
                    from datetime import datetime
                    t0 = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    dt = (t1 - t0).total_seconds()
                    phase_times.append((pn, dt))
                    evals.append(f"{pn}: {dt:.1f}s")
                except Exception:
                    pass

    if plt is not None and phase_times:
        fn = "pipeline_timeline.png"
        names = [p[0] for p in phase_times]
        times = [p[1] for p in phase_times]
        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.4)), dpi=150)
        y = np.arange(len(names))
        ax.barh(y, times, color="#7aa2f7", alpha=0.85, height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("seconds", fontsize=9)
        ax.set_title("Pipeline phase durations", fontsize=10)
        ax.invert_yaxis()
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(_fig_path(artifacts_dir, fn), bbox_inches="tight")
        plt.close(fig)
        pngs.append(fn)

    total = sum(t for _, t in phase_times) if phase_times else 0
    evals.insert(0, f"total pipeline time: {total:.1f}s")

    return pngs, evals


# ---------------------------------------------------------------------------
# HTML / CSS output
# ---------------------------------------------------------------------------

def _write_css(path: Path) -> None:
    css = """\
:root {
  --bg: #0b1020;
  --panel: #121a33;
  --card: #0f1730;
  --text: #e8eaf2;
  --muted: rgba(232,234,242,0.75);
  --border: rgba(255,255,255,0.08);
  --accent: #7aa2f7;
  --ok: #3fb950;
  --warn: #ffb86c;
  --bad: #ff5555;
}
* { box-sizing: border-box; }
body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: var(--bg); color: var(--text); }
header { padding: 24px 24px 8px; border-bottom: 1px solid var(--border); }
header h1 { margin:0 0 6px; font-size: 20px; }
header .meta { color: var(--muted); font-size: 13px; line-height: 1.6; }
main { padding: 18px 24px 48px; max-width: 1400px; margin: 0 auto; }
section { margin-top: 28px; }
section h2 { font-size: 16px; margin: 0 0 10px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.card.ok { border-color: rgba(63,185,80,0.6); }
.card.warn { border-color: rgba(255,184,108,0.7); }
.card.bad { border-color: rgba(255,85,85,0.8); }
.card h3 { margin: 0 0 8px; font-size: 13px; color: var(--accent); }
.card img { width: 100%; border-radius: 8px; border: 1px solid var(--border); margin-top: 6px; }
.card ul { margin: 8px 0 0 16px; padding: 0; }
.card ul li { margin: 3px 0; color: var(--muted); font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.card ul li.warn { color: var(--warn); }
.card .badge { display: inline-block; font-size: 10px; padding: 1px 7px; border-radius: 999px; margin-left: 8px; }
.badge.ok { border: 1px solid var(--ok); color: var(--ok); }
.badge.warn { border: 1px solid var(--warn); color: var(--warn); }
.badge.bad { border: 1px solid var(--bad); color: var(--bad); }
.eval-only { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.eval-only h3 { margin: 0 0 8px; font-size: 13px; color: var(--accent); }
.eval-only ul { margin: 8px 0 0 16px; padding: 0; }
.eval-only ul li { margin: 3px 0; color: var(--muted); font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.config { margin-top: 18px; padding: 14px; background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 12px; }
.config summary { cursor: pointer; color: var(--text); font-weight: 600; font-size: 13px; }
.config pre { margin: 10px 0 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.footer { margin-top: 28px; color: var(--muted); font-size: 11px; }
"""
    path.write_text(css, encoding="utf-8")


def _make_card_html(title: str, pngs: list[str], evals: list[str], status: str | None = None) -> str:
    cls = ""
    badge = ""
    if status:
        s = status.lower()
        if s in ("ok", "warn", "bad"):
            cls = f" {s}"
            badge = f'<span class="badge {s}">{_escape_html(status.upper())}</span>'

    imgs = "".join(f'<img src="{_escape_html(p)}" loading="lazy"/>' for p in pngs)
    items = "".join(
        f'<li class="{"warn" if "WARNING" in e.upper() else ""}">{_escape_html(e)}</li>'
        for e in evals if e
    )
    return (
        f'<div class="card{cls}">'
        f'<h3>{_escape_html(title)}{badge}</h3>'
        f'{imgs}'
        f'<ul>{items}</ul>'
        f'</div>'
    )


def _infer_status(evals: list[str]) -> str:
    text = "\n".join(evals).lower()
    if "fail" in text or "error" in text:
        return "bad"
    if "warning" in text:
        return "warn"
    return "ok"


def _write_html(path: Path, title: str, meta_lines: list[str],
                sections: list[tuple[str, str]], config_text: str | None = None) -> None:
    head = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{_escape_html(title)}</title>
  <link rel="stylesheet" href="report.css"/>
</head>
<body>
<header>
  <h1>{_escape_html(title)}</h1>
  <div class="meta">{'<br/>'.join(_escape_html(m) for m in meta_lines if m)}</div>
</header>
<main>
"""
    parts = [head]
    for sec_title, cards_html in sections:
        parts.append(f'<section><h2>{_escape_html(sec_title)}</h2><div class="grid">{cards_html}</div></section>')

    if config_text:
        parts.append(
            '<details class="config"><summary>Config (config.yaml)</summary>'
            f'<pre>{_escape_html(config_text)}</pre></details>'
        )

    parts.append('<div class="footer">Generated by generate_report.py (tile_compile C++ report)</div></main></body></html>\n')
    path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    artifacts_dir = run_dir / "artifacts"
    logs_path = run_dir / "logs" / "run_events.jsonl"
    config_path = run_dir / "config.yaml"

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    norm = _read_json(artifacts_dir / "normalization.json")
    gm = _read_json(artifacts_dir / "global_metrics.json")
    tg = _read_json(artifacts_dir / "tile_grid.json")
    reg = _read_json(artifacts_dir / "global_registration.json")
    lm = _read_json(artifacts_dir / "local_metrics.json")
    recon = _read_json(artifacts_dir / "tile_reconstruction.json")
    cl = _read_json(artifacts_dir / "state_clustering.json")
    syn = _read_json(artifacts_dir / "synthetic_frames.json")
    val = _read_json(artifacts_dir / "validation.json")
    events = _read_jsonl(logs_path)

    config_text = None
    if config_path.exists():
        try:
            config_text = config_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Extract run metadata
    run_start = next((e for e in events if e.get("type") == "run_start"), {})
    run_end = next((e for e in events if e.get("type") == "run_end"), {})

    meta_lines = [
        f"run_id: {run_dir.name}",
        f"run_dir: {run_dir}",
    ]
    if run_start:
        detail = run_start.get("detail", run_start)
        meta_lines.append(f"input_dir: {detail.get('input_dir', '?')}")
        meta_lines.append(f"frames: {detail.get('frames_discovered', '?')}")
        meta_lines.append(f"timestamp: {run_start.get('ts', run_start.get('timestamp', '?'))}")
    if run_end:
        meta_lines.append(f"final status: {run_end.get('status', '?')}")

    # Generate all sections
    sections: list[tuple[str, str]] = []

    # 0. Pipeline timeline
    tl_pngs, tl_evals = _gen_timeline(artifacts_dir, events)
    if tl_pngs or tl_evals:
        sections.append(("Pipeline Timeline", _make_card_html("Phase durations", tl_pngs, tl_evals)))

    # 1. Normalization
    if norm:
        n_pngs, n_evals = _gen_normalization(artifacts_dir, norm)
        sections.append(("Normalization", _make_card_html("Background levels", n_pngs, n_evals, _infer_status(n_evals))))

    # 2. Global Metrics
    if gm:
        g_pngs, g_evals = _gen_global_metrics(artifacts_dir, gm)
        sections.append(("Global Metrics", _make_card_html("Frame quality & weights", g_pngs, g_evals, _infer_status(g_evals))))

    # 3. Tile Grid
    if tg:
        t_pngs, t_evals = _gen_tile_grid(artifacts_dir, tg)
        sections.append(("Tile Grid", _make_card_html("Grid layout", t_pngs, t_evals)))

    # 4. Registration
    if reg:
        r_pngs, r_evals = _gen_registration(artifacts_dir, reg)
        sections.append(("Global Registration", _make_card_html("Frame alignment", r_pngs, r_evals, _infer_status(r_evals))))

    # 5. Local Metrics
    if lm:
        l_pngs, l_evals = _gen_local_metrics(artifacts_dir, lm, tg)
        sections.append(("Local Metrics", _make_card_html("Per-tile quality", l_pngs, l_evals, _infer_status(l_evals))))

    # 6. Tile Reconstruction
    if recon:
        rc_pngs, rc_evals = _gen_reconstruction(artifacts_dir, recon, tg)
        sections.append(("Tile Reconstruction", _make_card_html("Reconstruction stats", rc_pngs, rc_evals, _infer_status(rc_evals))))

    # 7. State Clustering
    if cl:
        cl_pngs, cl_evals = _gen_clustering(artifacts_dir, cl)
        sections.append(("State Clustering", _make_card_html("Cluster analysis", cl_pngs, cl_evals)))

    # 8. Synthetic Frames
    if syn:
        sy_pngs, sy_evals = _gen_synthetic(artifacts_dir, syn)
        sections.append(("Synthetic Frames", _make_card_html("Synthetic frame info", sy_pngs, sy_evals)))

    # 9. Validation
    if val:
        v_pngs, v_evals = _gen_validation(artifacts_dir, val)
        sections.append(("Validation", _make_card_html("Quality validation", v_pngs, v_evals, _infer_status(v_evals))))

    # Write output
    title = f"Tile-Compile Report — {run_dir.name}"
    css_path = artifacts_dir / "report.css"
    html_path = artifacts_dir / "report.html"
    _write_css(css_path)
    _write_html(html_path, title, meta_lines, sections, config_text=config_text)

    print(f"Report: {html_path}")
    return html_path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write("usage: generate_report.py /path/to/runs/<run_id>\n")
        return 2
    run_dir = Path(argv[1]).expanduser()
    if not run_dir.exists() or not run_dir.is_dir():
        sys.stderr.write(f"error: run_dir not found: {run_dir}\n")
        return 2
    generate_report(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
