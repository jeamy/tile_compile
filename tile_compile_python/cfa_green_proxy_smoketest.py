#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits


@dataclass
class Metrics:
    num_tiles: int
    rgb_tile_mean_std: Tuple[float, float, float]
    chroma_tile_std_rg_bg: Tuple[float, float]
    chroma_tile_range_rg_bg: Tuple[Tuple[float, float], Tuple[float, float]]
    adjacent_tile_mean_diff_maxabs_rgb: Tuple[float, float, float]
    boundary_step_mean_abs: float


def _find_latest(path_glob: str) -> Optional[Path]:
    import glob

    paths = [Path(p) for p in glob.glob(path_glob)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime)
    return paths[-1]


def _extract_frame_index(p: Path) -> Optional[int]:
    # Expect at least one numeric group; use the last group as frame index.
    ms = re.findall(r"(\d+)", p.name)
    if not ms:
        return None
    try:
        return int(ms[-1])
    except Exception:
        return None


def _parse_select_list(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part, 10))
    return out


def _safe_unlink_tree(p: Path):
    if p.exists():
        shutil.rmtree(p)


def _parse_run_id(text: str) -> Optional[str]:
    m = re.search(r"\[v4\] Starting run: ([0-9_]+_[0-9a-f]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"\"run_id\"\s*:\s*\"([^\"]+)\"", text)
    if m:
        return m.group(1)
    return None


def _load_rgb(rgb_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(str(rgb_path), memmap=False) as hdul:
        rgb = hdul[0].data
    if rgb is None:
        raise RuntimeError("stacked_rgb.fits has no data")
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        r, g, b = rgb[0], rgb[1], rgb[2]
    elif rgb.ndim == 3 and rgb.shape[2] == 3:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    else:
        raise RuntimeError(f"unexpected stacked_rgb shape: {rgb.shape}")
    return (
        np.asarray(r, dtype=np.float32),
        np.asarray(g, dtype=np.float32),
        np.asarray(b, dtype=np.float32),
    )


def _load_tile_metadata(meta_path: Path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(rgb_path: Path, meta_path: Path) -> Metrics:
    r, g, b = _load_rgb(rgb_path)
    meta = _load_tile_metadata(meta_path)

    means = []
    chroma = []
    coords = []

    for t in meta:
        x0, y0, tw, th = t["bbox"]
        rr = r[y0 : y0 + th, x0 : x0 + tw]
        gg = g[y0 : y0 + th, x0 : x0 + tw]
        bb = b[y0 : y0 + th, x0 : x0 + tw]
        mr, mg, mb = float(rr.mean()), float(gg.mean()), float(bb.mean())
        means.append((mr, mg, mb))
        chroma.append((mr - mg, mb - mg))
        coords.append((x0, y0, tw, th))

    means = np.asarray(means, dtype=np.float64)
    chroma = np.asarray(chroma, dtype=np.float64)

    rgb_tile_mean_std = tuple(float(x) for x in np.std(means, axis=0))
    chroma_tile_std = tuple(float(x) for x in np.std(chroma, axis=0))
    chroma_tile_range = (
        (float(np.min(chroma[:, 0])), float(np.max(chroma[:, 0]))),
        (float(np.min(chroma[:, 1])), float(np.max(chroma[:, 1]))),
    )

    from collections import defaultdict

    rows = defaultdict(list)
    for i, (x0, y0, tw, th) in enumerate(coords):
        rows[int(y0)].append((int(x0), i))

    adj_diffs = []
    for y0, xs in rows.items():
        xs.sort()
        for (_, i_a), (_, i_b) in zip(xs, xs[1:]):
            adj_diffs.append(means[i_b] - means[i_a])

    if adj_diffs:
        adj_diffs = np.asarray(adj_diffs, dtype=np.float64)
        adjacent_tile_mean_diff_maxabs_rgb = tuple(float(x) for x in np.max(np.abs(adj_diffs), axis=0))
    else:
        adjacent_tile_mean_diff_maxabs_rgb = (0.0, 0.0, 0.0)

    # Boundary metric: for each vertical boundary, compare mean of 2px strip left vs right.
    # This detects hard tile edges even if per-tile means vary naturally.
    x_starts = sorted({int(x0) for (x0, y0, tw, th) in coords})
    y_starts = sorted({int(y0) for (x0, y0, tw, th) in coords})

    # infer tile size/step from metadata
    tile_w = int(coords[0][2]) if coords else 0
    tile_h = int(coords[0][3]) if coords else 0

    # infer step via smallest positive delta
    def min_positive_delta(vals):
        ds = [b - a for a, b in zip(vals, vals[1:]) if (b - a) > 0]
        return int(min(ds)) if ds else 0

    step_x = min_positive_delta(x_starts)
    step_y = min_positive_delta(y_starts)

    # boundaries occur at x0 + step_x for interior transitions
    boundary_diffs = []
    if step_x > 0:
        for x0 in x_starts:
            xb = x0 + step_x
            if xb - 2 < 0 or xb + 2 >= r.shape[1]:
                continue
            left = np.stack([r[:, xb - 2 : xb], g[:, xb - 2 : xb], b[:, xb - 2 : xb]], axis=0)
            right = np.stack([r[:, xb : xb + 2], g[:, xb : xb + 2], b[:, xb : xb + 2]], axis=0)
            boundary_diffs.append(float(np.mean(np.abs(right.mean(axis=(1, 2)) - left.mean(axis=(1, 2))))))

    if step_y > 0:
        for y0 in y_starts:
            yb = y0 + step_y
            if yb - 2 < 0 or yb + 2 >= r.shape[0]:
                continue
            top = np.stack([r[yb - 2 : yb, :], g[yb - 2 : yb, :], b[yb - 2 : yb, :]], axis=0)
            bot = np.stack([r[yb : yb + 2, :], g[yb : yb + 2, :], b[yb : yb + 2, :]], axis=0)
            boundary_diffs.append(float(np.mean(np.abs(bot.mean(axis=(1, 2)) - top.mean(axis=(1, 2))))))

    boundary_step_mean_abs = float(np.mean(boundary_diffs)) if boundary_diffs else 0.0

    return Metrics(
        num_tiles=len(coords),
        rgb_tile_mean_std=rgb_tile_mean_std,
        chroma_tile_std_rg_bg=chroma_tile_std,
        chroma_tile_range_rg_bg=chroma_tile_range,
        adjacent_tile_mean_diff_maxabs_rgb=adjacent_tile_mean_diff_maxabs_rgb,
        boundary_step_mean_abs=boundary_step_mean_abs,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument(
        "--select",
        type=str,
        default="",
        help="Comma-separated frame indices to select from filenames (uses last numeric group), e.g. '1,50,120,301'",
    )
    ap.add_argument("--max-tile-size", type=int, default=256)
    ap.add_argument("--workdir", type=str, default="/tmp/cfa_green_proxy_smoketest")
    ap.add_argument("--input-root", type=str, default="/media/data/tile_compile_cache/IC434_lights")
    ap.add_argument("--pattern", type=str, default="raw_IC434_15s_80_*.fits")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    input_dir = workdir / "input"
    _safe_unlink_tree(workdir)
    input_dir.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(Path(args.input_root).glob(args.pattern))
    if not all_paths:
        raise SystemExit(f"no frames found in {args.input_root} with pattern {args.pattern}")

    if args.select:
        want = _parse_select_list(args.select)
        idx_to_path: dict[int, Path] = {}
        for p in all_paths:
            i = _extract_frame_index(p)
            if i is None:
                continue
            idx_to_path.setdefault(i, p)

        missing = [i for i in want if i not in idx_to_path]
        if missing:
            have = sorted(idx_to_path.keys())
            have_preview = have[:20]
            raise SystemExit(
                "missing selected frames: "
                + ",".join(str(x) for x in missing)
                + f". available indices (first 20): {have_preview}"
            )

        src_paths = [idx_to_path[i] for i in want]
    else:
        if len(all_paths) < args.n:
            raise SystemExit(f"not enough frames: found={len(all_paths)} need={args.n}")
        src_paths = all_paths[: args.n]

    frames_n = len(src_paths)

    for p in src_paths:
        shutil.copy2(p, input_dir / p.name)

    cfg_path = workdir / "config.yaml"
    cfg_text = f"""pipeline:
  mode: production
  abort_on_fail: true

data:
  image_width: 3840
  image_height: 2160
  frames_min: {frames_n}
  color_mode: OSC
  bayer_pattern: GBRG
  linear_required: true

calibration:
  use_bias: false
  use_dark: false
  use_flat: false

v4:
  iterations: 2
  beta: 6.0
  min_valid_tile_fraction: 0.3
  parallel_tiles: 1
  adaptive_tiles:
    enabled: false
    max_refine_passes: 0
    refine_variance_threshold: 0.15
    min_tile_size_px: 64
    use_warp_probe: false
    use_hierarchical: false
  convergence:
    enabled: false
  diagnostics:
    enabled: true

normalization:
  enabled: true
  mode: background
  per_channel: true

registration:
  mode: local_tiles
  local_tiles:
    ecc_cc_min: 0.2
    min_valid_frames: 1
    temporal_smoothing_window: 3
    variance_window_sigma: 2.0
    max_tile_size: {args.max_tile_size}

tile_grid:
  fwhm: 8.0
  min_size: 64
  overlap_fraction: 0.25

debayer: true
"""
    cfg_path.write_text(cfg_text, encoding="utf-8")

    out_run = Path("/tmp/output_cfa_green_proxy_smoketest_run.txt")
    with out_run.open("w", encoding="utf-8") as f:
        subprocess.run(
            [
                sys.executable,
                "tile_compile_runner.py",
                "run",
                "--config",
                str(cfg_path),
                "--input-dir",
                str(input_dir),
                "--pattern",
                "*.fits",
            ],
            cwd=str(Path(__file__).resolve().parent),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    log_txt = out_run.read_text(encoding="utf-8", errors="replace")
    run_id = _parse_run_id(log_txt)

    # Find outputs
    runs_dir = workdir / "runs"
    if run_id:
        out_dir = runs_dir / run_id / "outputs"
    else:
        # fallback: newest stacked_rgb
        rgb_latest = _find_latest(str(runs_dir / "*" / "outputs" / "stacked_rgb.fits"))
        if rgb_latest is None:
            raise SystemExit("could not find stacked_rgb.fits")
        out_dir = rgb_latest.parent

    rgb_path = out_dir / "stacked_rgb.fits"
    meta_path = out_dir / "tile_metadata.json"

    if not rgb_path.exists():
        raise SystemExit(f"missing {rgb_path}")
    if not meta_path.exists():
        raise SystemExit(f"missing {meta_path}")

    metrics = compute_metrics(rgb_path, meta_path)

    out_metrics_txt = Path("/tmp/output_cfa_green_proxy_smoketest_metrics.txt")
    out_metrics_json = Path("/tmp/output_cfa_green_proxy_smoketest_metrics.json")

    txt = (
        f"run_id: {run_id}\n"
        f"outputs: {out_dir}\n"
        f"num_tiles: {metrics.num_tiles}\n"
        f"rgb_tile_mean_std: {metrics.rgb_tile_mean_std}\n"
        f"chroma_tile_std_(R-G,B-G): {metrics.chroma_tile_std_rg_bg}\n"
        f"chroma_tile_range_(R-G,B-G): {metrics.chroma_tile_range_rg_bg}\n"
        f"adjacent_tile_mean_diff_maxabs_rgb: {metrics.adjacent_tile_mean_diff_maxabs_rgb}\n"
        f"boundary_step_mean_abs: {metrics.boundary_step_mean_abs}\n"
    )

    out_metrics_txt.write_text(txt, encoding="utf-8")
    out_metrics_json.write_text(
        json.dumps(metrics.__dict__, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
