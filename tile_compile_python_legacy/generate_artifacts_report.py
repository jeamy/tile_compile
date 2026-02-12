#!/usr/bin/env python3

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

try:
    from astropy.io import fits
except Exception:  # pragma: no cover
    fits = None
try:  # Optional preview generation
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class ChannelGlobal:
    background_level: list[float]
    noise_level: list[float]
    gradient_energy: list[float]
    G_f_c: list[float]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
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


def _mad_norm(vals: list[float]) -> np.ndarray:
    if not vals:
        return np.zeros((0,), dtype=np.float32)
    a = np.asarray(vals, dtype=np.float32)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if not np.isfinite(mad) or mad < 1e-12:
        return np.zeros_like(a)
    return (a - med) / float(1.4826 * mad)


def _compute_global_weights(
    background_level: list[float],
    noise_level: list[float],
    gradient_energy: list[float],
    w_bg: float,
    w_noise: float,
    w_grad: float,
) -> list[float]:
    b_n = _mad_norm(background_level)
    n_n = _mad_norm(noise_level)
    g_n = _mad_norm(gradient_energy)
    n = min(b_n.size, n_n.size, g_n.size)
    if n <= 0:
        return []
    q = w_bg * (-b_n[:n]) + w_noise * (-n_n[:n]) + w_grad * (g_n[:n])
    q = np.clip(q, -3.0, 3.0)
    return [float(x) for x in np.exp(q).astype(np.float32, copy=False).tolist()]


def _gradient_energy(frame: np.ndarray) -> float:
    try:
        gx, gy = np.gradient(frame.astype(np.float32, copy=False))
        return float(np.mean(np.hypot(gx, gy)))
    except Exception:
        return float("nan")


def _read_fits_float(path: Path) -> np.ndarray | None:
    if fits is None:
        return None
    try:
        data = fits.getdata(str(path), ext=0)
        if data is None:
            return None
        a = np.asarray(data)
        if a.ndim == 2:
            return a.astype(np.float32, copy=False)
        if a.ndim == 3:
            if a.shape[-1] in (3, 4):
                return np.mean(a[..., :3].astype(np.float32, copy=False), axis=-1)
            if a.shape[0] in (3, 4):
                return np.mean(a[:3, ...].astype(np.float32, copy=False), axis=0)
        return a.astype(np.float32, copy=False)
    except Exception:
        return None


def _try_write_fits_preview_png(
    fits_path: Path,
    png_path: Path,
    *,
    log_scale: bool = True,
    title: str | None = None,
) -> bool:
    if plt is None:
        return False
    data = _read_fits_float(fits_path)
    if data is None or data.ndim != 2:
        return False
    a = np.asarray(data, dtype=np.float32, copy=False)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if log_scale:
        a = np.log1p(np.maximum(a, 0.0))
    vmin = float(np.percentile(a, 1.0))
    vmax = float(np.percentile(a, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(a)), float(np.max(a))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
    plt.figure(figsize=(6, 4), dpi=150)
    plt.imshow(a, cmap="magma", vmin=vmin, vmax=vmax)
    plt.axis("off")
    if title:
        plt.title(title, fontsize=9)
    plt.tight_layout(pad=0.1)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()
    return True


def _try_write_hist_png(fits_path: Path, png_path: Path, *, title: str | None = None) -> bool:
    if plt is None:
        return False
    data = _read_fits_float(fits_path)
    if data is None or data.ndim != 2:
        return False
    a = np.asarray(data, dtype=np.float32, copy=False).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return False
    lo = float(np.percentile(a, 1.0))
    hi = float(np.percentile(a, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(a)), float(np.max(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return False
    plt.figure(figsize=(5.5, 4), dpi=150)
    plt.hist(a, bins=200, range=(lo, hi), color="#7aa2f7", alpha=0.85)
    plt.xlabel("value")
    plt.ylabel("count")
    if title:
        plt.title(title, fontsize=9)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path)
    plt.close()
    return True


def _try_load_report_metrics(artifacts_dir: Path) -> dict[str, Any] | None:
    p = artifacts_dir / "report_metrics.json"
    if not p.exists() or not p.is_file():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _discover_channel_files(channels_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {"R": [], "G": [], "B": []}
    if not channels_dir.exists() or not channels_dir.is_dir():
        return out
    for ch in ("R", "G", "B"):
        out[ch] = sorted([p for p in channels_dir.glob(f"{ch}_*.fit*") if p.is_file()])
    return out


def _try_load_tile_grid(artifacts_dir: Path) -> dict[str, Any] | None:
    p = artifacts_dir / "tile_grid.json"
    if not p.exists() or not p.is_file():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _tile_overlap_from_grid_cfg(grid_cfg: dict[str, Any]) -> tuple[int, float] | None:
    try:
        tile_size = int(grid_cfg.get("tile_size") or 0)
    except Exception:
        tile_size = 0
    if tile_size <= 0:
        return None
    step_size = None
    for k in ("step_size", "step", "tile_step"):
        v = grid_cfg.get(k)
        if v is None:
            continue
        try:
            step_size = int(v)
            break
        except Exception:
            continue
    if step_size is None or step_size <= 0:
        overlap_frac = float(grid_cfg.get("overlap_fraction") or grid_cfg.get("overlap") or 0.25)
        overlap_frac = float(np.clip(overlap_frac, 0.0, 0.9))
        return tile_size, overlap_frac
    overlap_frac = 1.0 - (float(step_size) / float(tile_size))
    overlap_frac = float(np.clip(overlap_frac, 0.0, 0.9))
    return tile_size, overlap_frac


def _welford_update(mean: np.ndarray, m2: np.ndarray, n: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n1 = n + 1
    delta = x - mean
    mean = mean + delta / float(n1)
    delta2 = x - mean
    m2 = m2 + delta * delta2
    return mean, m2


def _compute_tile_summaries(
    channel_files: dict[str, list[Path]],
    tile_size: int,
    overlap: float,
    w_fwhm: float,
    w_round: float,
    w_con: float,
) -> dict[str, Any]:
    try:
        from tile_compile_backend.metrics import TileMetricsCalculator
    except Exception:
        return {}

    step = int(tile_size * (1.0 - float(overlap)))
    step = max(1, step)

    out: dict[str, Any] = {"tile_size": int(tile_size), "overlap": float(overlap), "step": int(step), "channels": {}}

    calc = TileMetricsCalculator(tile_size=int(tile_size), overlap=float(overlap))

    for ch in ("R", "G", "B"):
        files = channel_files.get(ch) or []
        if not files:
            continue

        first = _read_fits_float(files[0])
        if first is None or first.ndim != 2:
            continue
        h, w = first.shape
        n_tiles_y = max(1, (h - tile_size) // step + 1)
        n_tiles_x = max(1, (w - tile_size) // step + 1)
        n_tiles = n_tiles_y * n_tiles_x

        mean_q = np.zeros((n_tiles,), dtype=np.float64)
        m2_q = np.zeros((n_tiles,), dtype=np.float64)
        mean_l = np.zeros((n_tiles,), dtype=np.float64)
        m2_l = np.zeros((n_tiles,), dtype=np.float64)
        n_seen = 0

        per_frame_mean_q: list[float] = []
        per_frame_var_q: list[float] = []

        for p in files:
            f = _read_fits_float(p)
            if f is None or f.ndim != 2 or f.shape != (h, w):
                continue
            tm = calc.calculate_tile_metrics(f)
            fwhm = np.asarray(tm.get("fwhm") or [], dtype=np.float64)
            rnd = np.asarray(tm.get("roundness") or [], dtype=np.float64)
            con = np.asarray(tm.get("contrast") or [], dtype=np.float64)
            if fwhm.size != n_tiles or rnd.size != n_tiles or con.size != n_tiles:
                continue

            def _mad_norm_arr(arr: np.ndarray) -> np.ndarray:
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)))
                if not np.isfinite(mad) or mad < 1e-12:
                    return np.zeros_like(arr)
                return (arr - med) / float(1.4826 * mad)

            fwhm_n = _mad_norm_arr(fwhm)
            rnd_n = _mad_norm_arr(rnd)
            con_n = _mad_norm_arr(con)
            q = (float(w_fwhm) * (-fwhm_n) + float(w_round) * rnd_n + float(w_con) * con_n)
            q = np.clip(q, -3.0, 3.0)
            l = np.exp(q)

            per_frame_mean_q.append(float(np.mean(q)))
            per_frame_var_q.append(float(np.var(q)))

            mean_q, m2_q = _welford_update(mean_q, m2_q, n_seen, q)
            mean_l, m2_l = _welford_update(mean_l, m2_l, n_seen, l)
            n_seen += 1

        if n_seen <= 0:
            continue

        var_q = (m2_q / float(max(1, n_seen - 1))).astype(np.float64, copy=False)
        var_l = (m2_l / float(max(1, n_seen - 1))).astype(np.float64, copy=False)

        out["channels"][ch] = {
            "frame_count": int(n_seen),
            "grid": {"tiles_x": int(n_tiles_x), "tiles_y": int(n_tiles_y), "tiles": int(n_tiles)},
            "tile_quality_mean": [float(x) for x in mean_q.tolist()],
            "tile_quality_variance": [float(x) for x in var_q.tolist()],
            "tile_weight_mean": [float(x) for x in mean_l.tolist()],
            "tile_weight_variance": [float(x) for x in var_l.tolist()],
            "per_frame": {"tile_q_mean": per_frame_mean_q, "tile_q_var": per_frame_var_q},
        }

    return out


def _pct(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{100.0 * x:.1f}%"


def _basic_stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0.0}
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0.0}
    return {
        "n": float(a.size),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def _evaluate_timeseries(name: str, vals: list[float]) -> list[str]:
    s = _basic_stats(vals)
    if int(s.get("n", 0)) <= 1:
        return [f"{name}: insufficient data"]
    med = float(s.get("median") or 0.0)
    span = float(s.get("max") or 0.0) - float(s.get("min") or 0.0)
    rel = (span / abs(med)) if med not in (0.0, -0.0) else float("inf")
    out = [f"{name}: median={med:.4g}, min={float(s.get('min') or 0.0):.4g}, max={float(s.get('max') or 0.0):.4g}"]
    if np.isfinite(rel):
        out.append(f"{name}: span/|median|={rel:.3g}")
    if np.isfinite(rel) and rel > 0.25:
        out.append(f"warning: {name} varies strongly over time")
    return out


def _evaluate_weights(gfc: list[float]) -> list[str]:
    s = _basic_stats(gfc)
    if int(s.get("n", 0)) <= 1:
        return ["G_f,c: insufficient data"]
    med = float(s.get("median") or 0.0)
    mn = float(s.get("min") or 0.0)
    mx = float(s.get("max") or 0.0)
    out = [f"G_f,c: median={med:.4g}, min={mn:.4g}, max={mx:.4g}"]
    if med > 0:
        low = sum(1 for x in gfc if np.isfinite(x) and x < med * 0.2)
        high = sum(1 for x in gfc if np.isfinite(x) and x > med * 5.0)
        out.append(f"frames with very low weight (<0.2×median): {low}")
        out.append(f"frames with very high weight (>5×median): {high}")
        out.append(f"fraction very low: {_pct(low / max(1, int(s.get('n', 0))))}")
    if mn > 0 and mx > 0:
        out.append(f"max/min ratio: {mx / mn:.3g}")
    if mn > 0 and mx / mn > 50:
        out.append("warning: extremely wide weight distribution")
    return out


def _evaluate_tile_grid(tile_grid: dict[str, Any] | None) -> list[str]:
    if not tile_grid:
        return ["tile_grid.json missing"]
    grid_cfg = tile_grid.get("grid_cfg") if isinstance(tile_grid.get("grid_cfg"), dict) else {}
    meta = tile_grid.get("grid_metadata") if isinstance(tile_grid.get("grid_metadata"), dict) else {}
    out = []
    if grid_cfg:
        out.append(f"grid_cfg: {json.dumps(grid_cfg, ensure_ascii=False)}")
    if meta:
        out.append(f"grid_metadata: {json.dumps(meta, ensure_ascii=False)}")
    return out or ["tile_grid.json present"]


def _escape_html(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _infer_status_from_eval_lines(eval_lines: list[str]) -> str | None:
    s = "\n".join([str(x) for x in eval_lines if x]).lower()
    if not s:
        return None
    if any(k in s for k in ("error", "failed", "exception", "traceback")):
        return "BAD"
    if any(k in s for k in ("nan", "inf", "not finite")):
        return "BAD"
    if "warning" in s:
        return "WARN"
    if any(k in s for k in ("insufficient", "missing", "skipped", "fallback")):
        return "WARN"
    return "OK"


def _write_css(path: Path) -> None:
    css = r"""
:root {
  --bg: #0b1020;
  --panel: #121a33;
  --card: #0f1730;
  --text: #e8eaf2;
  --muted: rgba(232,234,242,0.75);
  --ok: #3fb950;
  --warn: #ffb86c;
  --bad: #ff5555;
}
* { box-sizing: border-box; }
body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: var(--bg); color: var(--text); }
header { padding: 24px 24px 8px; border-bottom: 1px solid var(--border); }
header h1 { margin:0 0 6px; font-size: 20px; }
header .meta { color: var(--muted); font-size: 13px; }
main { padding: 18px 24px 48px; max-width: 1200px; margin: 0 auto; }
section { margin-top: 22px; }
section h2 { font-size: 16px; margin: 0 0 10px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 12px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
.card.status-ok { border-color: rgba(63,185,80,0.7); }
.card.status-warn { border-color: rgba(255,184,108,0.8); }
.card.status-bad { border-color: rgba(255,85,85,0.9); }
.card .headerline { display:flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
.card h3 { margin: 0 0 8px; font-size: 14px; }
.card .links { font-size: 12px; margin-bottom: 8px; }
.card .links a { color: var(--accent); text-decoration: none; }
.card img { width: 100%; border-radius: 8px; border: 1px solid var(--border); }
.badge { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); color: var(--muted); }
.badge.ok { border-color: rgba(63,185,80,0.7); color: var(--ok); }
.badge.warn { border-color: rgba(255,184,108,0.8); color: var(--warn); }
.badge.bad { border-color: rgba(255,85,85,0.9); color: var(--bad); }
.kv { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; color: var(--muted); white-space: pre-wrap; }
ul.eval { margin: 8px 0 0 18px; padding: 0; }
ul.eval li { margin: 4px 0; color: var(--muted); font-size: 13px; }
.footer { margin-top: 28px; color: var(--muted); font-size: 12px; }
.config { margin-top: 14px; padding: 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }
.config summary { cursor: pointer; color: var(--text); font-weight: 600; }
.config pre { margin: 10px 0 0 0; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; color: var(--muted); }
""".strip() + "\n"
    path.write_text(css, encoding="utf-8")


def _write_html(
    path: Path,
    title: str,
    meta_lines: list[str],
    sections: list[tuple[str, list[str]]],
    config_text: str | None = None,
) -> None:
    head = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>{_escape_html(title)}</title>
  <link rel=\"stylesheet\" href=\"report.css\"/>
</head>
<body>
<header>
  <h1>{_escape_html(title)}</h1>
  <div class=\"meta\">{_escape_html(' | '.join([x for x in meta_lines if x]))}</div>
</header>
<main>
"""

    body_parts: list[str] = [head]
    for sec_title, cards_html in sections:
        body_parts.append(f"<section><h2>{_escape_html(sec_title)}</h2><div class=\"grid\">")
        body_parts.extend(cards_html)
        body_parts.append("</div></section>")

    if config_text:
        body_parts.append(
            "<details class=\"config\"><summary>Config (used)</summary>"
            f"<pre>{_escape_html(config_text)}</pre></details>"
        )

    body_parts.append(
        "<div class=\"footer\">Generated by generate_artifacts_report.py</div></main></body></html>\n"
    )
    path.write_text("\n".join(body_parts), encoding="utf-8")


def _make_card(title: str, rel_path: str, eval_lines: list[str], status: str | None = None) -> str:
    ev = "".join([f"<li>{_escape_html(x)}</li>" for x in eval_lines if x])
    st = str(status or "").strip().upper()
    cls = ""
    badge = ""
    if st in ("OK", "WARN", "BAD"):
        cls = f" status-{st.lower()}"
        badge = f"<span class=\"badge {st.lower()}\">{_escape_html(st)}</span>"
    return (
        f"<div class=\"card{cls}\">"
        f"<div class=\"headerline\"><h3>{_escape_html(title)}</h3>{badge}</div>"
        f"<div class=\"links\"><a href=\"{_escape_html(rel_path)}\">open</a></div>"
        f"<img src=\"{_escape_html(rel_path)}\" loading=\"lazy\"/>"
        f"<ul class=\"eval\">{ev}</ul>"
        "</div>"
    )


def _make_link_card(title: str, rel_path: str, eval_lines: list[str], status: str | None = None) -> str:
    ev = "".join([f"<li>{_escape_html(x)}</li>" for x in eval_lines if x])
    st = str(status or "").strip().upper()
    cls = ""
    badge = ""
    if st in ("OK", "WARN", "BAD"):
        cls = f" status-{st.lower()}"
        badge = f"<span class=\"badge {st.lower()}\">{_escape_html(st)}</span>"
    return (
        f"<div class=\"card{cls}\">"
        f"<div class=\"headerline\"><h3>{_escape_html(title)}</h3>{badge}</div>"
        f"<div class=\"links\"><a href=\"{_escape_html(rel_path)}\">open</a></div>"
        f"<ul class=\"eval\">{ev}</ul>"
        "</div>"
    )


def _read_config(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "config.yaml"
    if not p.exists() or not p.is_file():
        return {}
    try:
        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_config_text(run_dir: Path) -> str | None:
    p = run_dir / "config.yaml"
    if not p.exists() or not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _weights_from_config(cfg: dict[str, Any]) -> tuple[float, float, float]:
    g = cfg.get("global_metrics") if isinstance(cfg.get("global_metrics"), dict) else {}
    w = g.get("weights") if isinstance(g.get("weights"), dict) else {}
    try:
        wb = float(w.get("background", 0.4))
    except Exception:
        wb = 0.4
    try:
        wn = float(w.get("noise", 0.3))
    except Exception:
        wn = 0.3
    try:
        wg = float(w.get("gradient", 0.3))
    except Exception:
        wg = 0.3
    s = wb + wn + wg
    if s > 1e-12:
        wb, wn, wg = wb / s, wn / s, wg / s
    return wb, wn, wg


def _local_weights_from_config(cfg: dict[str, Any]) -> tuple[float, float, float]:
    lm = cfg.get("local_metrics") if isinstance(cfg.get("local_metrics"), dict) else {}
    sm = lm.get("star_mode") if isinstance(lm.get("star_mode"), dict) else {}
    w = sm.get("weights") if isinstance(sm.get("weights"), dict) else {}
    try:
        wf = float(w.get("fwhm", 0.6))
    except Exception:
        wf = 0.6
    try:
        wr = float(w.get("roundness", 0.2))
    except Exception:
        wr = 0.2
    try:
        wc = float(w.get("contrast", 0.2))
    except Exception:
        wc = 0.2
    s = wf + wr + wc
    if s > 1e-12:
        wf, wr, wc = wf / s, wr / s, wc / s
    return wf, wr, wc


def _compute_global_from_channels(channel_files: dict[str, list[Path]], cfg: dict[str, Any]) -> dict[str, ChannelGlobal]:
    w_bg, w_noise, w_grad = _weights_from_config(cfg)
    out: dict[str, ChannelGlobal] = {}
    for ch in ("R", "G", "B"):
        bg: list[float] = []
        noise: list[float] = []
        grad: list[float] = []
        for p in channel_files.get(ch) or []:
            f = _read_fits_float(p)
            if f is None or f.ndim != 2:
                continue
            bg.append(float(np.median(f)))
            noise.append(float(np.std(f)))
            grad.append(_gradient_energy(f))
        gfc = _compute_global_weights(bg, noise, grad, w_bg, w_noise, w_grad)
        if bg or noise or grad or gfc:
            out[ch] = ChannelGlobal(bg, noise, grad, gfc)
    return out


def _known_artifacts() -> list[tuple[str, list[str]]]:
    return [
        ("NORMALIZATION", ["normalization_background_timeseries.png"]),
        (
            "GLOBAL_METRICS",
            [
                "global_weight_timeseries.png",
                "global_weight_hist.png",
            ],
        ),
        ("TILE_GRID", ["tile_grid_overlay_R.png", "tile_grid_overlay_G.png", "tile_grid_overlay_B.png", "tile_grid.json"]),
        (
            "LOCAL_METRICS",
            [
                "tile_quality_heatmap_R.png",
                "tile_quality_heatmap_G.png",
                "tile_quality_heatmap_B.png",
                "tile_quality_var_heatmap_R.png",
                "tile_quality_var_heatmap_G.png",
                "tile_quality_var_heatmap_B.png",
                "tile_weight_heatmap_R.png",
                "tile_weight_heatmap_G.png",
                "tile_weight_heatmap_B.png",
                "tile_weight_var_heatmap_R.png",
                "tile_weight_var_heatmap_G.png",
                "tile_weight_var_heatmap_B.png",
            ],
        ),
        (
            "TILE_RECONSTRUCTION",
            [
                "reconstruction_weight_sum_R.png",
                "reconstruction_weight_sum_G.png",
                "reconstruction_weight_sum_B.png",
                "reconstruction_preview_R.png",
                "reconstruction_preview_G.png",
                "reconstruction_preview_B.png",
                "reconstruction_absdiff_vs_mean_R.png",
                "reconstruction_absdiff_vs_mean_G.png",
                "reconstruction_absdiff_vs_mean_B.png",
            ],
        ),
        (
            "CLUSTERING",
            [
                "clustering_summary_R.png",
                "clustering_summary_G.png",
                "clustering_summary_B.png",
            ],
        ),
        (
            "STACKING",
            [
                "quality_analysis_combined.png",
            ],
        ),
        (
            "REGISTRATION",
            [
                "registration_absdiff_samples.png",
                "registration_ecc_corr_timeseries.png",
                "registration_ecc_corr_hist.png",
            ],
        ),
        (
            "VALIDATION",
            [
                "fwhm_heatmap.fits",
                "warp_dx.fits",
                "warp_dy.fits",
                "invalid_tile_map.fits",
                "tile_validation_maps.json",
            ],
        ),
        (
            "VALIDATION_PREVIEW",
            [
                "fwhm_heatmap_log.png",
                "fwhm_heatmap_hist.png",
                "warp_dx_log.png",
                "warp_dx_hist.png",
                "warp_dy_log.png",
                "warp_dy_hist.png",
                "invalid_tile_map_log.png",
                "invalid_tile_map_hist.png",
            ],
        ),
    ]


def generate_report(run_dir: Path) -> tuple[Path, Path]:
    run_dir = run_dir.resolve()
    artifacts_dir = run_dir / "artifacts"
    logs_path = run_dir / "logs" / "run_events.jsonl"

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cfg = _read_config(run_dir)
    cfg_text = _read_config_text(run_dir)
    events = _read_jsonl(logs_path)
    run_start = next((e for e in events if e.get("type") == "run_start"), {})

    channels_dir = run_dir / "work" / "channels"
    channel_files = _discover_channel_files(channels_dir)

    tile_grid = _try_load_tile_grid(artifacts_dir)
    report_metrics = _try_load_report_metrics(artifacts_dir)

    # Generate validation previews (log image + histogram) if FITS maps exist.
    if plt is not None:
        for base in ("fwhm_heatmap", "warp_dx", "warp_dy", "invalid_tile_map"):
            fpath = artifacts_dir / f"{base}.fits"
            if not fpath.exists():
                continue
            _try_write_fits_preview_png(
                fpath,
                artifacts_dir / f"{base}_log.png",
                log_scale=True,
                title=f"{base} (log1p)",
            )
            _try_write_hist_png(
                fpath,
                artifacts_dir / f"{base}_hist.png",
                title=f"{base} histogram",
            )

    # If runtime metrics exist, prefer them and avoid post-hoc recomputation.
    global_metrics: dict[str, ChannelGlobal] = {}
    tile_summaries: dict[str, Any] = {}
    if report_metrics is None:
        global_metrics = _compute_global_from_channels(channel_files, cfg)

        if tile_grid and channel_files:
            grid_cfg = tile_grid.get("grid_cfg") if isinstance(tile_grid.get("grid_cfg"), dict) else {}
            ts = _tile_overlap_from_grid_cfg(grid_cfg)
            if ts is not None:
                tile_size, overlap = ts
                wf, wr, wc = _local_weights_from_config(cfg)
                tile_summaries = _compute_tile_summaries(channel_files, tile_size, overlap, wf, wr, wc)

    report_data = {
        "run_dir": str(run_dir),
        "run_id": str(run_dir.name),
        "run_start": run_start,
        "config": cfg,
        "global": {
            ch: {
                "background_level": gm.background_level,
                "noise_level": gm.noise_level,
                "gradient_energy": gm.gradient_energy,
                "G_f_c": gm.G_f_c,
            }
            for ch, gm in global_metrics.items()
        },
        "tile": tile_summaries,
    }

    (artifacts_dir / "report_data.json").write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    known = _known_artifacts()
    present = {p.name for p in artifacts_dir.iterdir() if p.is_file()}

    sections: list[tuple[str, list[str]]] = []

    if tile_grid:
        grid_eval = _evaluate_tile_grid(tile_grid)
    else:
        grid_eval = ["tile_grid.json missing"]

    for sec, files in known:
        cards: list[str] = []
        for fn in files:
            if fn not in present:
                continue
            if fn.endswith(".json"):
                eval_lines = grid_eval if fn == "tile_grid.json" else ["json present"]
                st = None
                if report_metrics is not None:
                    art = report_metrics.get("artifacts") if isinstance(report_metrics.get("artifacts"), dict) else {}
                    entry = art.get(fn) if isinstance(art.get(fn), dict) else None
                    if isinstance(entry, dict):
                        st = entry.get("status")
                st2 = str(st or "").strip().upper()
                cls = f" status-{st2.lower()}" if st2 in ("OK", "WARN", "BAD") else ""
                badge = f"<span class=\"badge {st2.lower()}\">{_escape_html(st2)}</span>" if st2 in ("OK", "WARN", "BAD") else ""
                cards.append(
                    f"<div class=\"card{cls}\">"
                    f"<div class=\"headerline\"><h3>{_escape_html(fn)}</h3>{badge}</div>"
                    f"<div class=\"links\"><a href=\"{_escape_html(fn)}\">open</a></div>"
                    f"<div class=\"kv\">{_escape_html('\n'.join(eval_lines))}</div>"
                    "</div>"
                )
                continue
            if fn.endswith(".fits"):
                eval_lines = ["FITS artifact present"]
                st = None
                if report_metrics is not None:
                    art = report_metrics.get("artifacts") if isinstance(report_metrics.get("artifacts"), dict) else {}
                    entry = art.get(fn) if isinstance(art.get(fn), dict) else None
                    if isinstance(entry, dict):
                        st = entry.get("status")
                        ev = entry.get("evaluations")
                        if isinstance(ev, list):
                            eval_lines = [str(x) for x in ev] or eval_lines
                if report_metrics is None:
                    eval_lines = ["missing artifacts/report_metrics.json (legacy run)."]
                if st is None:
                    st = _infer_status_from_eval_lines(eval_lines)
                cards.append(_make_link_card(fn, fn, eval_lines, status=st))
                continue

            eval_lines: list[str] = []
            status = None

            if report_metrics is not None:
                art = report_metrics.get("artifacts") if isinstance(report_metrics.get("artifacts"), dict) else {}
                entry = art.get(fn) if isinstance(art.get(fn), dict) else None
                if isinstance(entry, dict):
                    status = entry.get("status")
                    ev = entry.get("evaluations")
                    if isinstance(ev, dict):
                        merged: list[str] = []
                        for _k, v in ev.items():
                            if isinstance(v, list):
                                merged.extend([str(x) for x in v])
                            elif isinstance(v, dict):
                                for _k2, v2 in v.items():
                                    if isinstance(v2, list):
                                        merged.extend([str(x) for x in v2])
                        if merged:
                            eval_lines = merged
                    elif isinstance(ev, list):
                        eval_lines = [str(x) for x in ev]

            if fn == "normalization_background_timeseries.png":
                if report_metrics is None:
                    for ch in ("R", "G", "B"):
                        gm = global_metrics.get(ch)
                        if gm is None:
                            continue
                        eval_lines.extend(_evaluate_timeseries(f"{ch} background (post-norm)", gm.background_level))
            elif fn in ("global_weight_timeseries.png", "global_weight_hist.png"):
                if report_metrics is None:
                    for ch in ("R", "G", "B"):
                        gm = global_metrics.get(ch)
                        if gm is None:
                            continue
                        eval_lines.extend([f"{ch}:" ] + _evaluate_weights(gm.G_f_c))
            elif fn.startswith("tile_") and tile_summaries:
                ch = fn.split("_")[-1].split(".")[0]
                chd = tile_summaries.get("channels", {}).get(ch)
                if isinstance(chd, dict):
                    tqm = chd.get("tile_quality_mean") if isinstance(chd.get("tile_quality_mean"), list) else []
                    tqv = chd.get("tile_quality_variance") if isinstance(chd.get("tile_quality_variance"), list) else []
                    if tqm:
                        s = _basic_stats([float(x) for x in tqm])
                        eval_lines.append(f"tile Q_local mean: median={float(s.get('median') or 0.0):.3g}, min={float(s.get('min') or 0.0):.3g}, max={float(s.get('max') or 0.0):.3g}")
                    if tqv:
                        s = _basic_stats([float(x) for x in tqv])
                        eval_lines.append(f"tile Q_local var: median={float(s.get('median') or 0.0):.3g}, max={float(s.get('max') or 0.0):.3g}")
            if not eval_lines:
                if report_metrics is None:
                    eval_lines = [
                        "missing artifacts/report_metrics.json (legacy run).",
                        "Recommendation: re-run pipeline (or resume) to generate runtime metrics, then re-generate report.",
                    ]
                else:
                    eval_lines = ["evaluation missing in report_metrics.json for this artifact"]

            if status is None:
                status = _infer_status_from_eval_lines(eval_lines)

            cards.append(_make_card(fn, fn, eval_lines, status=status))

        if cards:
            sections.append((sec, cards))

    other_pngs = sorted([p.name for p in artifacts_dir.glob("*.png") if p.is_file()])
    known_set = {fn for _sec, fns in known for fn in fns if fn.endswith(".png")}
    extra = [fn for fn in other_pngs if fn not in known_set]
    if extra:
        cards: list[str] = []
        for fn in extra:
            st = None
            if report_metrics is not None:
                art = report_metrics.get("artifacts") if isinstance(report_metrics.get("artifacts"), dict) else {}
                entry = art.get(fn) if isinstance(art.get(fn), dict) else None
                if isinstance(entry, dict):
                    st = entry.get("status")
            cards.append(_make_card(fn, fn, ["unclassified artifact"], status=st))
        sections.append(("OTHER", cards))

    title = f"Tile-Compile Artifacts Report — {run_dir.name}"
    meta_lines: list[str] = []
    meta_lines.append(f"run_dir={run_dir}")
    meta_lines.append(f"input_dir={run_start.get('input_dir')}")
    meta_lines.append(f"frames_discovered={run_start.get('frames_discovered')}")
    meta_lines.append(f"config_hash={run_start.get('config_hash')}")

    cal_cfg = cfg.get("calibration") if isinstance(cfg.get("calibration"), dict) else {}
    use_bias = bool(cal_cfg.get("use_bias"))
    use_dark = bool(cal_cfg.get("use_dark"))
    use_flat = bool(cal_cfg.get("use_flat"))
    if use_bias:
        bias_dir = str(cal_cfg.get("bias_dir") or "").strip()
        bias_master = str(cal_cfg.get("bias_master") or "").strip()
        if bias_dir:
            meta_lines.append(f"bias_dir={bias_dir}")
        if bias_master:
            meta_lines.append(f"bias_master={bias_master}")
        built_bias = run_dir / "outputs" / "calibration" / "master_bias.fit"
        if built_bias.exists():
            meta_lines.append(f"bias_master_built={built_bias}")
    if use_dark:
        darks_dir = str(cal_cfg.get("darks_dir") or "").strip()
        dark_master = str(cal_cfg.get("dark_master") or "").strip()
        if darks_dir:
            meta_lines.append(f"darks_dir={darks_dir}")
        if dark_master:
            meta_lines.append(f"dark_master={dark_master}")
        built_dark = run_dir / "outputs" / "calibration" / "master_dark.fit"
        if built_dark.exists():
            meta_lines.append(f"dark_master_built={built_dark}")
    if use_flat:
        flats_dir = str(cal_cfg.get("flats_dir") or "").strip()
        flat_master = str(cal_cfg.get("flat_master") or "").strip()
        if flats_dir:
            meta_lines.append(f"flats_dir={flats_dir}")
        if flat_master:
            meta_lines.append(f"flat_master={flat_master}")
        built_flat = run_dir / "outputs" / "calibration" / "master_flat.fit"
        if built_flat.exists():
            meta_lines.append(f"flat_master_built={built_flat}")

    css_path = artifacts_dir / "report.css"
    html_path = artifacts_dir / "report.html"
    _write_css(css_path)
    _write_html(html_path, title, meta_lines, sections, config_text=cfg_text)

    return html_path, css_path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write("usage: generate_artifacts_report.py /path/to/run_dir\n")
        return 2
    run_dir = Path(argv[1]).expanduser()
    if not run_dir.exists() or not run_dir.is_dir():
        sys.stderr.write(f"error: run_dir not found: {run_dir}\n")
        return 2
    html_path, _css_path = generate_report(run_dir)
    sys.stdout.write(str(html_path) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
