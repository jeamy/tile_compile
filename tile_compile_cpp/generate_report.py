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

def _gen_normalization(artifacts_dir: Path, norm: dict) -> tuple[list[str], list[str], dict[str, str]]:
    """Generate normalization charts. Returns (png_files, eval_lines, explanations)."""
    pngs: list[str] = []
    evals: list[str] = []
    explanations: dict[str, str] = {}

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
            explanations[fn] = (
                '<h4>Hintergrund pro Kanal</h4>'
                '<p>Zeigt den geschätzten Himmelshintergrund für jeden Frame, '
                'aufgeteilt nach Bayer-Kanälen (R, G, B).</p>'
                '<p><b>Interpretation:</b></p>'
                '<ul>'
                '<li><span class="good">Gut:</span> Alle Kanäle verlaufen stabil und parallel — '
                'gleichmäßige Belichtungsbedingungen.</li>'
                '<li><span class="bad">Schlecht:</span> Starke Sprünge oder Drift deuten auf '
                'wechselnde Wolken, Lichtverschmutzung oder Tau hin.</li>'
                '<li>Große Unterschiede zwischen R/G/B können auf Farbgradienten '
                '(z.B. Mondlicht, Laternen) hinweisen.</li>'
                '<li>Die Normalisierung gleicht diese Unterschiede aus — '
                'je stärker die Schwankung, desto wichtiger ist sie.</li>'
                '</ul>'
            )
        for name, vals in [("R", b_r), ("G", b_g), ("B", b_b)]:
            s = _basic_stats(vals)
            if s["n"]:
                evals.append(f"  {name}: median={s['median']:.4g}, std={s['std']:.4g}, range=[{s['min']:.4g}, {s['max']:.4g}]")
    elif b_mono:
        fn = "norm_background_mono.png"
        if _plot_timeseries(b_mono, "Background level (mono)", "background", _fig_path(artifacts_dir, fn)):
            pngs.append(fn)
            explanations[fn] = (
                '<h4>Hintergrund (Mono)</h4>'
                '<p>Geschätzter Himmelshintergrund pro Frame.</p>'
                '<p><b>Interpretation:</b></p>'
                '<ul>'
                '<li><span class="good">Gut:</span> Flache, stabile Linie — gleichmäßige Bedingungen.</li>'
                '<li><span class="bad">Schlecht:</span> Sprünge oder Drift — Wolken, Tau oder Lichtverschmutzung.</li>'
                '<li>Die rote gestrichelte Linie zeigt den Median.</li>'
                '</ul>'
            )
        s = _basic_stats(b_mono)
        if s["n"]:
            evals.append(f"  mono: median={s['median']:.4g}, std={s['std']:.4g}")

    return pngs, evals, explanations


def _gen_global_metrics(artifacts_dir: Path, gm: dict) -> tuple[list[str], list[str], dict[str, str]]:
    pngs: list[str] = []
    evals: list[str] = []
    explanations: dict[str, str] = {}

    metrics = gm.get("metrics", [])
    if not metrics:
        evals.append("no metrics data")
        return pngs, evals, explanations

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
        explanations[fn] = (
            '<h4>Hintergrund</h4>'
            '<p>Mittlerer Himmelshintergrund jedes Frames nach Normalisierung.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Gut:</span> Gleichmäßiger Verlauf nahe dem Median.</li>'
            '<li><span class="bad">Schlecht:</span> Starke Ausreißer = Wolken, Tau oder Lichtverschmutzung. '
            'Diese Frames erhalten automatisch ein niedrigeres Gewicht.</li>'
            '</ul>'
        )

    # Noise timeseries
    fn = "global_noise.png"
    if _plot_timeseries(noise, "Frame noise level", "noise", _fig_path(artifacts_dir, fn), color="#ff6b6b"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Rauschen</h4>'
            '<p>Geschätztes Rauschen (σ) pro Frame — berechnet über robuste MAD-Statistik.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Niedriger &amp; stabil:</span> Gute Bedingungen, gleichmäßige Belichtung.</li>'
            '<li><span class="bad">Hohe Spitzen:</span> Frames mit erhöhtem Rauschen (kurze Wolkenlücken, '
            'Vibrationen). Werden im Stacking heruntergewichtet.</li>'
            '<li>Generell gilt: weniger Rauschen = besseres Signal-Rausch-Verhältnis im Endergebnis.</li>'
            '</ul>'
        )

    # Gradient energy timeseries
    fn = "global_gradient.png"
    if _plot_timeseries(grad, "Frame gradient energy", "gradient energy", _fig_path(artifacts_dir, fn), color="#50fa7b"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Gradientenenergie</h4>'
            '<p>Maß für die Menge an Bilddetails/Strukturen pro Frame (Laplacian-basiert).</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Hoch &amp; stabil:</span> Scharfe Frames mit viel Detailzeichnung.</li>'
            '<li><span class="bad">Niedrig:</span> Unscharfe Frames (Seeing, Defokus, Nachführfehler).</li>'
            '<li>Frames mit hoher Gradientenenergie erhalten ein höheres Gewicht im Stacking.</li>'
            '</ul>'
        )

    # Global weights timeseries + histogram
    fn = "global_weight_timeseries.png"
    if _plot_timeseries(gw, "Global frame weight G(f)", "weight", _fig_path(artifacts_dir, fn), color="#ffb86c"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Globales Gewicht G(f)</h4>'
            '<p>Kombiniertes Qualitätsgewicht pro Frame aus Hintergrund, Rauschen und Gradientenenergie.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Hohe Werte (≈ 1.0):</span> Beste Frames — tragen am meisten zum Ergebnis bei.</li>'
            '<li><span class="bad">Niedrige Werte (&lt; 0.2):</span> Schlechte Frames — werden stark heruntergewichtet.</li>'
            '<li>Eine breite Verteilung ist normal bei wechselhaften Bedingungen.</li>'
            '<li>Wenn fast alle Gewichte gleich sind, waren die Bedingungen sehr gleichmäßig.</li>'
            '</ul>'
        )

    fn = "global_weight_hist.png"
    if _plot_histogram(gw, "Global weight distribution", "weight", _fig_path(artifacts_dir, fn), color="#ffb86c"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Gewichtsverteilung</h4>'
            '<p>Histogramm der globalen Frame-Gewichte.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Schmale Verteilung rechts:</span> Die meisten Frames sind gut.</li>'
            '<li><span class="neutral">Bimodale Verteilung:</span> Zwei Gruppen — gute und schlechte Frames. '
            'Das Clustering wird diese automatisch trennen.</li>'
            '<li><span class="bad">Breite Verteilung links:</span> Viele schlechte Frames — '
            'das Endergebnis profitiert weniger vom Stacking.</li>'
            '</ul>'
        )

    # --- Siril-style per-frame star metrics ---
    fwhm = [m.get("fwhm", 0) for m in metrics]
    wfwhm = [m.get("wfwhm", 0) for m in metrics]
    roundness = [m.get("roundness", 0) for m in metrics]
    star_count = [m.get("star_count", 0) for m in metrics]

    # FWHM timeseries
    fn = "global_fwhm.png"
    if _plot_timeseries(fwhm, "FWHM per frame", "FWHM (px)", _fig_path(artifacts_dir, fn), color="#bd93f9"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>FWHM (Halbwertsbreite)</h4>'
            '<p>Die FWHM (Full Width at Half Maximum) misst die Sternbreite in Pixeln — '
            'ein direktes Maß für das Seeing und die Schärfe.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Niedrig (&lt; 3 px):</span> Exzellentes Seeing, scharfe Sterne.</li>'
            '<li><span class="neutral">Mittel (3–5 px):</span> Durchschnittliches Seeing.</li>'
            '<li><span class="bad">Hoch (&gt; 5 px):</span> Schlechtes Seeing oder Nachführprobleme.</li>'
            '<li>Stabile Werte = gleichmäßiges Seeing über die Session.</li>'
            '</ul>'
        )

    # wFWHM timeseries (weighted by star count)
    fn = "global_wfwhm.png"
    if _plot_timeseries(wfwhm, "Weighted FWHM per frame (wFWHM)", "wFWHM (px)", _fig_path(artifacts_dir, fn), color="#ff79c6"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Gewichtete FWHM (wFWHM)</h4>'
            '<p>FWHM gewichtet mit der Sternanzahl: wFWHM = FWHM × (ref_stars / stars). '
            'Bestraft Frames mit wenigen erkannten Sternen.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Niedrig:</span> Scharf UND viele Sterne erkannt.</li>'
            '<li><span class="bad">Hoch:</span> Unscharf oder wenige Sterne (Wolken, Defokus).</li>'
            '<li>Besser als reine FWHM, da Frames mit wenigen Sternen '
            '(z.B. durch Wolken) höher bestraft werden.</li>'
            '</ul>'
        )

    # Roundness timeseries
    fn = "global_roundness.png"
    if _plot_timeseries(roundness, "Star roundness per frame (FWHMy/FWHMx)", "roundness", _fig_path(artifacts_dir, fn), color="#8be9fd"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Sternrundheit</h4>'
            '<p>Verhältnis FWHMy/FWHMx — misst wie rund die Sterne sind.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">≈ 1.0:</span> Perfekt runde Sterne — gute Nachführung.</li>'
            '<li><span class="neutral">0.7–0.9:</span> Leicht elongierte Sterne — '
            'akzeptabel, aber Nachführung prüfen.</li>'
            '<li><span class="bad">&lt; 0.7:</span> Stark verzerrte Sterne — '
            'Nachführfehler, Wind oder Verkippung.</li>'
            '<li>Systematischer Drift über die Session deutet auf '
            'Polausrichtungsfehler hin.</li>'
            '</ul>'
        )

    # Star count timeseries
    fn = "global_star_count.png"
    if _plot_timeseries(star_count, "Detected stars per frame", "stars", _fig_path(artifacts_dir, fn), color="#f1fa8c"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Erkannte Sterne</h4>'
            '<p>Anzahl der automatisch erkannten Sterne pro Frame.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Stabil &amp; hoch:</span> Klarer Himmel, gute Bedingungen.</li>'
            '<li><span class="bad">Plötzliche Einbrüche:</span> Wolken, Tau oder Defokus — '
            'diese Frames werden automatisch heruntergewichtet.</li>'
            '<li>Ein langsamer Rückgang kann auf zunehmenden Tau hindeuten.</li>'
            '<li>Die absolute Zahl hängt von Brennweite und Himmelsregion ab.</li>'
            '</ul>'
        )

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
            explanations[fn] = (
                '<h4>FWHM vs. Rundheit</h4>'
                '<p>Scatter-Plot im Siril-Stil: jeder Punkt ist ein Frame, '
                'Farbe = zeitliche Reihenfolge.</p>'
                '<p><b>Interpretation:</b></p>'
                '<ul>'
                '<li><span class="good">Kompakter Cluster links oben:</span> '
                'Gleichmäßig scharfe, runde Sterne — ideale Bedingungen.</li>'
                '<li><span class="bad">Ausreißer rechts unten:</span> '
                'Unscharfe UND elongierte Sterne — schlechteste Frames.</li>'
                '<li>Farbverlauf zeigt zeitliche Entwicklung: '
                'Drift nach rechts = Seeing verschlechtert sich.</li>'
                '<li>Gestrichelte Linie bei Rundheit=1.0 = perfekt rund.</li>'
                '</ul>'
            )

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

    return pngs, evals, explanations


def _gen_tile_grid(artifacts_dir: Path, tg: dict) -> tuple[list[str], list[str], dict[str, str]]:
    pngs: list[str] = []
    evals: list[str] = []
    explanations: dict[str, str] = {}

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
        explanations[fn] = (
            '<h4>Tile-Raster</h4>'
            '<p>Zeigt die Aufteilung des Bildes in überlappende Kacheln (Tiles). '
            'Jede Kachel wird unabhängig gewichtet und rekonstruiert.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li>Die <b>Tile-Größe</b> wird automatisch aus der FWHM berechnet — '
            'größere FWHM → größere Tiles.</li>'
            '<li><b>Overlap</b> (Überlappung) sorgt für nahtlose Übergänge '
            'zwischen Kacheln (Hanning-Fenster).</li>'
            '<li>Mehr Tiles = feinere lokale Qualitätssteuerung, '
            'aber auch mehr Rechenzeit.</li>'
            '<li>Typisch: 50–200 Tiles für ein 1920×1080-Bild.</li>'
            '</ul>'
        )

    return pngs, evals, explanations


def _gen_registration(artifacts_dir: Path, reg: dict) -> tuple[list[str], list[str], dict[str, str]]:
    pngs: list[str] = []
    evals: list[str] = []
    explanations: dict[str, str] = {}

    n = reg.get("num_frames", 0)
    scale = reg.get("scale", "?")
    ref = reg.get("ref_frame", 0)
    evals.append(f"frames: {n}, scale: {scale}, ref_frame: {ref}")

    warps = reg.get("warps", [])
    ccs = reg.get("cc", [])

    fn = "registration_overview.png"
    if _plot_warp_scatter(warps, ccs, "Registration", _fig_path(artifacts_dir, fn)):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Registrierungsübersicht</h4>'
            '<p>Drei Diagramme: (1) Translation-Scatter — Verschiebung jedes Frames '
            'relativ zum Referenzframe. (2) Translation über Zeit. (3) Korrelationskoeffizient.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Kompakter Scatter:</span> Geringe Drift — stabile Montierung.</li>'
            '<li><span class="bad">Große Streuung:</span> Starke Drift oder Windstöße.</li>'
            '<li><b>CC (Korrelation):</b> Maß für die Registrierungsqualität. '
            '<span class="good">CC &gt; 0.8</span> = sehr gut, '
            '<span class="bad">CC &lt; 0.5</span> = problematisch.</li>'
            '<li>Farbverlauf im Scatter zeigt zeitliche Reihenfolge — '
            'systematischer Drift = Polausrichtungsfehler.</li>'
            '</ul>'
        )

    # CC histogram
    fn = "registration_cc_hist.png"
    if _plot_histogram(ccs, "Registration correlation coefficient", "CC", _fig_path(artifacts_dir, fn), color="#50fa7b"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>CC-Verteilung</h4>'
            '<p>Histogramm des Korrelationskoeffizienten (CC) aller Frames.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Peak nahe 1.0:</span> Alle Frames gut registriert.</li>'
            '<li><span class="bad">Viele Werte &lt; 0.5:</span> Registrierung fehlgeschlagen — '
            'diese Frames fallen auf Identity-Fallback zurück.</li>'
            '<li>CC = 0 bedeutet: Frame konnte nicht registriert werden '
            '(z.B. komplett bewölkt).</li>'
            '</ul>'
        )

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
            explanations[fn] = (
                '<h4>Rotation pro Frame</h4>'
                '<p>Rotationswinkel jedes Frames relativ zum Referenzframe.</p>'
                '<p><b>Interpretation:</b></p>'
                '<ul>'
                '<li><span class="good">≈ 0°:</span> Keine Feldrotation — '
                'äquatoriale Montierung oder kurze Session.</li>'
                '<li><span class="neutral">Linearer Anstieg:</span> Normale Feldrotation '
                'bei Alt/Az-Montierung — wird durch Registrierung kompensiert.</li>'
                '<li><span class="bad">Sprünge:</span> Mechanische Probleme '
                '(Kabelzug, Lockerung).</li>'
                '</ul>'
            )

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
            explanations[fn] = (
                '<h4>Skalierungsfaktor</h4>'
                '<p>Skalierung jedes Frames relativ zum Referenzframe.</p>'
                '<p><b>Interpretation:</b></p>'
                '<ul>'
                '<li><span class="good">≈ 1.000:</span> Keine Skalierungsänderung — normal.</li>'
                '<li><span class="bad">Abweichung &gt; 0.01:</span> Fokus-Drift, '
                'Temperaturänderung oder mechanische Probleme.</li>'
                '</ul>'
            )

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

    return pngs, evals, explanations


def _gen_local_metrics(artifacts_dir: Path, lm: dict, tg: dict) -> tuple[list[str], list[str], dict[str, str]]:
    pngs: list[str] = []
    evals: list[str] = []
    explanations: dict[str, str] = {}

    n_frames = lm.get("num_frames", 0)
    n_tiles = lm.get("num_tiles", 0)
    evals.append(f"frames: {n_frames}, tiles: {n_tiles}")

    tile_metrics = lm.get("tile_metrics", [])
    if not tile_metrics:
        evals.append("no tile metrics data")
        return pngs, evals, explanations

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
        explanations[fn] = (
            '<h4>Lokale Metriken — Heatmaps</h4>'
            '<p>Drei räumliche Karten: FWHM, Qualitäts-Score und Qualitätsvarianz pro Tile.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><b>FWHM:</b> <span class="good">Niedrig (dunkel)</span> = scharf, '
            '<span class="bad">hoch (hell)</span> = unscharf. '
            'Variationen zeigen lokales Seeing oder optische Fehler (Koma, Astigmatismus).</li>'
            '<li><b>Quality Score:</b> Z-Score-basiert. '
            '<span class="good">Hoch (hell)</span> = beste Tiles, '
            '<span class="bad">niedrig (dunkel)</span> = schlechteste.</li>'
            '<li><b>Quality Variance:</b> Hohe Varianz = instabile Qualität über die Frames. '
            'Tiles am Bildrand haben oft höhere Varianz.</li>'
            '</ul>'
        )

    # Spatial weight + contrast + stars combined
    fn = "local_weight_contrast_spatial.png"
    if use_spatial and _plot_spatial_tile_multi(
        tiles, [
            ("Mean local weight", mean_weight, "plasma"),
            ("Weight variance", var_weight, "magma"),
            ("Mean contrast", mean_contrast, "cividis"),
        ], img_w, img_h, "Local Weights & Contrast — Spatial", _fig_path(artifacts_dir, fn)):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Gewichte &amp; Kontrast</h4>'
            '<p>Räumliche Verteilung der lokalen Gewichte, Gewichtsvarianz und des Kontrasts.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><b>Local Weight:</b> Exponential des Quality-Scores. '
            '<span class="good">Hoch</span> = Tile trägt stark zum Ergebnis bei.</li>'
            '<li><b>Weight Variance:</b> <span class="good">Niedrig</span> = stabile Qualität, '
            '<span class="bad">hoch</span> = stark schwankend.</li>'
            '<li><b>Contrast:</b> Laplacian-Varianz. '
            '<span class="good">Hoch</span> = viel Detailzeichnung (Sterne, Nebel).</li>'
            '</ul>'
        )

    # Individual large spatial heatmaps
    fn = "local_fwhm_spatial.png"
    if use_spatial and _plot_spatial_tile_heatmap(
        tiles, mean_fwhm, img_w, img_h, "Mean FWHM per tile",
        _fig_path(artifacts_dir, fn), cmap="inferno", label="FWHM (px)"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>FWHM-Karte (Detail)</h4>'
            '<p>Mittlere FWHM pro Tile über alle Frames gemittelt.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Gleichmäßig dunkel:</span> Gute Optik, gleichmäßiges Seeing.</li>'
            '<li><span class="bad">Helle Ecken/Ränder:</span> Optische Aberrationen '
            '(Koma, Bildfeldwölbung). Typisch bei schnellen Optiken.</li>'
            '<li>Zentral scharf, Rand unscharf = Bildfeldwölbung → Flattener prüfen.</li>'
            '</ul>'
        )

    fn = "local_weight_spatial.png"
    if use_spatial and _plot_spatial_tile_heatmap(
        tiles, mean_weight, img_w, img_h, "Mean local weight per tile",
        _fig_path(artifacts_dir, fn), cmap="plasma", label="weight"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Gewichtskarte (Detail)</h4>'
            '<p>Mittleres lokales Gewicht pro Tile. Bestimmt den Beitrag jeder Kachel zum Endergebnis.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Gleichmäßig hell:</span> Alle Bildbereiche tragen gleich bei.</li>'
            '<li><span class="bad">Dunkle Bereiche:</span> Niedrige Qualität — '
            'diese Tiles werden heruntergewichtet (z.B. Ecken mit Koma).</li>'
            '</ul>'
        )

    fn = "local_stars_spatial.png"
    if use_spatial and _plot_spatial_tile_heatmap(
        tiles, mean_stars, img_w, img_h, "Mean star count per tile",
        _fig_path(artifacts_dir, fn), cmap="YlGnBu", label="stars"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Sternverteilung</h4>'
            '<p>Mittlere Anzahl erkannter Sterne pro Tile.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li>Tiles mit vielen Sternen werden im <b>STAR-Modus</b> gewichtet '
            '(FWHM + Rundheit + Kontrast).</li>'
            '<li>Tiles mit wenigen Sternen nutzen den <b>STRUCTURE-Modus</b> '
            '(Gradientenenergie / Rauschen).</li>'
            '<li>Die Verteilung hängt von der Himmelsregion ab — '
            'Milchstraße hat mehr Sterne als Polregion.</li>'
            '</ul>'
        )

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
            explanations[fn] = (
                '<h4>Tile-Typ-Karte</h4>'
                '<p>Gelb = STAR-Tile (genug Sterne für FWHM-basierte Gewichtung). '
                'Lila = STRUCTURE-Tile (Gradientenenergie-basiert).</p>'
                '<p><b>Interpretation:</b></p>'
                '<ul>'
                '<li><b>STAR-Tiles</b> nutzen FWHM, Rundheit und Kontrast '
                'für die Qualitätsbewertung — ideal für Sternfelder.</li>'
                '<li><b>STRUCTURE-Tiles</b> nutzen Gradientenenergie/Rauschen — '
                'besser für Nebel und ausgedehnte Objekte.</li>'
                '<li>Die Schwelle ist konfigurierbar (star_min_count).</li>'
                '</ul>'
            )

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
        explanations[fn] = (
            '<h4>Qualität &amp; Gewicht pro Frame</h4>'
            '<p>Mittlerer Quality-Score und mittleres lokales Gewicht über alle Tiles pro Frame.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Stabil &amp; hoch:</span> Gleichmäßig gute Frames.</li>'
            '<li><span class="bad">Einbrüche:</span> Frames mit schlechter lokaler Qualität '
            '(Wolken, Seeing-Spitzen).</li>'
            '<li>Unterschied zwischen Quality und Weight zeigt, '
            'wie stark die exponentielle Gewichtung differenziert.</li>'
            '</ul>'
        )

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

    return pngs, evals, explanations


def _gen_reconstruction(artifacts_dir: Path, recon: dict, tg: dict) -> tuple[list[str], list[str], dict[str, str]]:
    pngs: list[str] = []
    evals: list[str] = []
    explanations: dict[str, str] = {}

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
        explanations[fn] = (
            '<h4>Rekonstruktionsübersicht</h4>'
            '<p>Drei räumliche Karten: gültige Frame-Anzahl, Korrelation und SNR pro Tile nach der Rekonstruktion.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><b>Valid frames:</b> Wie viele Frames pro Tile beigetragen haben. '
            '<span class="good">Gleichmäßig hoch</span> = gute Abdeckung.</li>'
            '<li><b>Correlation:</b> Registrierungsqualität pro Tile. '
            '<span class="good">&gt; 0.8</span> = sehr gut.</li>'
            '<li><b>SNR:</b> Signal-Rausch-Verhältnis nach Rekonstruktion. '
            '<span class="good">Hoch</span> = gutes Ergebnis.</li>'
            '</ul>'
        )

    # Large spatial: valid counts (frame usage per tile)
    fn = "recon_valid_counts_spatial.png"
    if use_spatial and _plot_spatial_tile_heatmap(
        tiles, [float(v) for v in valid_counts], img_w, img_h,
        "Valid frames per tile (tile usage)", _fig_path(artifacts_dir, fn),
        cmap="YlGn", label="valid frames"):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Gültige Frames pro Tile</h4>'
            '<p>Anzahl der Frames, die für jede Kachel im Stacking verwendet wurden.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Gleichmäßig = Gesamtzahl:</span> Alle Frames tragen zu allen Tiles bei.</li>'
            '<li><span class="bad">Niedrige Werte:</span> Einige Tiles haben wenige gültige Frames — '
            'höheres Rauschen in diesen Bereichen.</li>'
            '<li>Tiles &lt; 3 Frames: Warnung — Sigma-Clipping kann nicht zuverlässig arbeiten.</li>'
            '</ul>'
        )

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
        explanations[fn] = (
            '<h4>SNR-Karte (Detail)</h4>'
            '<p>Signal-Rausch-Verhältnis pro Tile nach der Rekonstruktion.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Hohe Werte (hell):</span> Gutes SNR — klare Details.</li>'
            '<li><span class="bad">Niedrige Werte (dunkel):</span> Verrauschte Bereiche — '
            'wenig Signal oder hoher Hintergrund.</li>'
            '<li>Nebelbereiche haben typischerweise höheres SNR als leerer Himmel.</li>'
            '</ul>'
        )

    # Spatial: post-contrast + post-background
    fn = "recon_contrast_bg_spatial.png"
    if use_spatial and post_contrast and post_bg and _plot_spatial_tile_multi(
        tiles, [
            ("Post contrast", post_contrast, "cividis"),
            ("Post background", post_bg, "gray"),
        ], img_w, img_h, "Post-Reconstruction Contrast & Background", _fig_path(artifacts_dir, fn)):
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Kontrast &amp; Hintergrund</h4>'
            '<p>Post-Rekonstruktions-Kontrast (Laplacian-Varianz) und Hintergrundlevel pro Tile.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><b>Kontrast:</b> <span class="good">Hoch</span> = viel Detailzeichnung. '
            'Nebel und Sternhaufen haben hohen Kontrast.</li>'
            '<li><b>Hintergrund:</b> Sollte räumlich gleichmäßig sein. '
            'Gradienten deuten auf Lichtverschmutzung oder Vignettierung hin.</li>'
            '</ul>'
        )

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

    return pngs, evals, explanations


def _gen_clustering(artifacts_dir: Path, cl: dict) -> tuple[list[str], list[str], dict[str, str]]:
    pngs: list[str] = []
    evals: list[str] = []
    explanations: dict[str, str] = {}

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
            explanations[fn] = (
                '<h4>Cluster-Größen</h4>'
                '<p>Anzahl der Frames pro Cluster nach K-Means-Clustering.</p>'
                '<p><b>Interpretation:</b></p>'
                '<ul>'
                '<li>Frames werden nach Qualitätsmerkmalen gruppiert '
                '(Gewicht, lokale Qualität, Varianz, Korrelation).</li>'
                '<li><span class="good">Ähnlich große Cluster:</span> '
                'Gleichmäßige Bedingungen über die Session.</li>'
                '<li><span class="neutral">Ein dominanter Cluster:</span> '
                'Die meisten Frames sind ähnlich — wenig Variation.</li>'
                '<li>Jeder Cluster erzeugt ein synthetisches Frame '
                'für das finale Stacking.</li>'
                '</ul>'
            )

        for i, sz in enumerate(sizes):
            evals.append(f"  cluster {i}: {sz} frames")

    if labels:
        fn = "clustering_labels.png"
        if _plot_timeseries(labels, "Cluster label per frame", "cluster", _fig_path(artifacts_dir, fn)):
            pngs.append(fn)
            explanations[fn] = (
                '<h4>Cluster-Zuordnung</h4>'
                '<p>Zeigt welchem Cluster jeder Frame zugeordnet wurde.</p>'
                '<p><b>Interpretation:</b></p>'
                '<ul>'
                '<li>Zusammenhängende Blöcke = Bedingungen änderten sich '
                'langsam (z.B. Seeing-Phasen).</li>'
                '<li>Häufige Wechsel = schnell wechselnde Bedingungen '
                '(z.B. durchziehende Wolken).</li>'
                '</ul>'
            )

    return pngs, evals, explanations


def _gen_synthetic(artifacts_dir: Path, syn: dict) -> tuple[list[str], list[str], dict[str, str]]:
    pngs: list[str] = []
    evals: list[str] = []
    explanations: dict[str, str] = {}

    n = syn.get("num_synthetic", 0)
    fmin = syn.get("frames_min", "?")
    fmax = syn.get("frames_max", "?")
    evals.append(f"num_synthetic: {n}")
    evals.append(f"frames range: [{fmin}, {fmax}]")

    return pngs, evals, explanations


def _gen_validation(artifacts_dir: Path, val: dict) -> tuple[list[str], list[str], dict[str, str]]:
    pngs: list[str] = []
    evals: list[str] = []
    explanations: dict[str, str] = {}

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
                explanations[fn] = (
                    '<h4>Validierungsprüfungen</h4>'
                    '<p>Automatische Qualitätsprüfungen des Endergebnisses. '
                    'Grün = bestanden, Rot = nicht bestanden.</p>'
                    '<p><b>Prüfungen:</b></p>'
                    '<ul>'
                    '<li><b>FWHM-Verbesserung:</b> Ist das gestackte Bild schärfer als die Einzelframes? '
                    '<span class="good">Positiv</span> = Verbesserung, '
                    '<span class="bad">negativ</span> = Verschlechterung. '
                    'Bei unterabgetasteten Daten (große Pixel) ist 0% normal.</li>'
                    '<li><b>Tile-Gewichtsvarianz:</b> Sind die Tile-Gewichte ausreichend unterschiedlich? '
                    'Zu geringe Varianz bedeutet, dass die Gewichtung keinen Effekt hat.</li>'
                    '<li><b>Tile-Pattern-Ratio:</b> Gibt es sichtbare Kachelgrenzen im Ergebnis? '
                    '<span class="good">&lt; 1.5</span> = keine sichtbaren Artefakte, '
                    '<span class="bad">&gt; 1.5</span> = Kachelgrenzen sichtbar.</li>'
                    '</ul>'
                )

    return pngs, evals, explanations


def _gen_timeline(artifacts_dir: Path, events: list[dict]) -> tuple[list[str], list[str], dict[str, str]]:
    """Generate pipeline timeline chart from events."""
    pngs: list[str] = []
    evals: list[str] = []
    explanations: dict[str, str] = {}

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
        explanations[fn] = (
            '<h4>Pipeline-Zeitplan</h4>'
            '<p>Dauer jeder Pipeline-Phase in Sekunden.</p>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><b>REGISTRATION:</b> Skaliert mit Frame-Anzahl. '
            'Lange Dauer = viele Frames oder schwierige Registrierung.</li>'
            '<li><b>TILE_RECONSTRUCTION:</b> Meist die längste Phase. '
            'Skaliert mit Tiles × Frames.</li>'
            '<li><b>SYNTHETIC_FRAMES:</b> Hängt von Cluster-Anzahl ab.</li>'
            '<li><b>ASTROMETRY:</b> Plate-Solving via ASTAP. '
            'Kann bei schwachem Signal lange dauern.</li>'
            '<li><b>PCC:</b> Photometrische Farbkalibrierung. '
            'Schnell bei lokalen Katalogen, langsamer bei Online-Abfragen.</li>'
            '</ul>'
        )

    total = sum(t for _, t in phase_times) if phase_times else 0
    evals.insert(0, f"total pipeline time: {total:.1f}s")

    return pngs, evals, explanations


def _gen_frame_usage(artifacts_dir: Path, events: list[dict]) -> tuple[list[str], list[str], dict[str, str]]:
    """Generate frame usage funnel: discovered → linearity → registration → synthetic."""
    pngs: list[str] = []
    evals: list[str] = []
    explanations: dict[str, str] = {}

    # Extract data from events
    run_start = next((e for e in events if e.get("type") == "run_start"), {})
    scan_end = next((e for e in events if e.get("type") == "phase_end" and e.get("phase_name") == "SCAN_INPUT"), {})
    reg_end = next((e for e in events if e.get("type") == "phase_end" and e.get("phase_name") == "REGISTRATION"), {})
    synth_end = next((e for e in events if e.get("type") == "phase_end" and e.get("phase_name") == "SYNTHETIC_FRAMES"), {})

    frames_discovered = run_start.get("frames_discovered", 0)
    linearity = scan_end.get("linearity", {})
    linearity_enabled = linearity.get("enabled", False)
    linearity_failed = linearity.get("failed_frames", 0)
    linearity_action = linearity.get("action", "")
    frames_after_scan = scan_end.get("frames_scanned", frames_discovered)

    frames_usable_reg = reg_end.get("frames_usable", 0)
    frames_excluded_identity = reg_end.get("frames_excluded_identity", 0)
    frames_excluded_negative = reg_end.get("frames_excluded_negative", 0)

    num_synthetic = synth_end.get("num_synthetic", 0)
    synth_status = synth_end.get("status", "")

    # Read synthetic artifact for frames_max
    syn_artifact = _read_json(artifacts_dir / "synthetic_frames.json")
    synth_frames_max = syn_artifact.get("frames_max", None) if syn_artifact else None

    # Build funnel stages
    stages: list[tuple[str, int, str]] = []  # (label, count, reason)
    stages.append(("Entdeckt", frames_discovered, "Input-Verzeichnis gescannt"))

    if linearity_enabled and linearity_failed > 0:
        removed = linearity_failed if linearity_action == "removed" else 0
        if removed > 0:
            stages.append(("Nach Linearität", frames_after_scan,
                           f"{removed} nicht-lineare Frames entfernt"))
        else:
            stages.append(("Nach Linearität", frames_discovered,
                           f"{linearity_failed} Frames markiert (behalten)"))
    elif linearity_enabled:
        stages.append(("Nach Linearität", frames_after_scan, "Alle Frames linear"))

    if frames_usable_reg > 0 or frames_excluded_identity > 0:
        excluded_total = frames_excluded_identity + frames_excluded_negative
        reasons = []
        if frames_excluded_identity > 0:
            reasons.append(f"{frames_excluded_identity} Identity-Fallback")
        if frames_excluded_negative > 0:
            reasons.append(f"{frames_excluded_negative} negative Korrelation")
        reason_str = ", ".join(reasons) if reasons else "Alle registriert"
        stages.append(("Registriert (nutzbar)", frames_usable_reg, reason_str))

    # Eval lines
    for label, count, reason in stages:
        evals.append(f"{label}: {count}  ({reason})")

    # Synthetic frames: not a retention stage (count is capped by frames_max), show as text only
    if num_synthetic > 0:
        src_count = stages[-1][1] if stages else '?'
        max_info = f", max={synth_frames_max}" if synth_frames_max is not None else ""
        evals.append(f"Synthetische Frames: {num_synthetic}  (aus {src_count} Frames via Clustering{max_info})")
    elif synth_status == "skipped":
        reason = synth_end.get("reason", "übersprungen")
        evals.append(f"Synthetische Frames: 0  (Übersprungen: {reason})")

    # Funnel bar chart
    if plt is not None and len(stages) >= 2:
        fn = "frame_usage_funnel.png"
        labels = [s[0] for s in stages]
        counts = [s[1] for s in stages]
        reasons = [s[2] for s in stages]

        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.6 * len(stages))), dpi=150)
        y = list(range(len(stages)))
        max_count = max(counts) if counts else 1

        # Color gradient: green for high retention, red for low
        colors = []
        for c in counts:
            ratio = c / max(1, max_count)
            if ratio > 0.7:
                colors.append("#50fa7b")
            elif ratio > 0.3:
                colors.append("#ffb86c")
            else:
                colors.append("#ff6b6b")

        bars = ax.barh(y, counts, color=colors, edgecolor="none", height=0.6, alpha=0.85)

        for i, (bar, count, reason) in enumerate(zip(bars, counts, reasons)):
            # Count label inside bar
            if count > 0:
                ax.text(bar.get_width() - max_count * 0.02, bar.get_y() + bar.get_height() / 2,
                        str(count), ha="right", va="center", fontsize=10, fontweight="bold",
                        color="white" if count > max_count * 0.15 else "#ccc")
            # Reason label to the right
            ax.text(bar.get_width() + max_count * 0.02, bar.get_y() + bar.get_height() / 2,
                    reason, ha="left", va="center", fontsize=8, color="#999")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Frames", fontsize=9)
        ax.set_title("Frame-Nutzung (Funnel)", fontsize=10)
        ax.invert_yaxis()
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, max_count * 1.5)
        fig.tight_layout()
        fig.savefig(_fig_path(artifacts_dir, fn), bbox_inches="tight")
        plt.close(fig)
        pngs.append(fn)
        explanations[fn] = (
            '<h4>Frame-Nutzung</h4>'
            '<p>Zeigt den „Trichter" der Frame-Verarbeitung: wie viele Frames in jeder Phase '
            'übrig bleiben.</p>'
            '<p><b>Stufen:</b></p>'
            '<ul>'
            '<li><b>Entdeckt:</b> Gesamtzahl der FITS-Dateien im Input-Verzeichnis.</li>'
            '<li><b>Nach Linearität:</b> Frames, die den Linearitätstest bestanden haben. '
            'Nicht-lineare Frames (z.B. Flats, Darks, fehlerhafte Aufnahmen) werden entfernt.</li>'
            '<li><b>Registriert (nutzbar):</b> Frames mit erfolgreicher Registrierung '
            '(CC &gt; Schwelle). Identity-Fallback = Frame konnte nicht registriert werden '
            '(z.B. bewölkt, stark verschoben).</li>'
            '</ul>'
            '<p><b>Interpretation:</b></p>'
            '<ul>'
            '<li><span class="good">Hohe Retention (&gt; 80%):</span> '
            'Gute Datenqualität, wenig Ausschuss.</li>'
            '<li><span class="neutral">50–80% Retention:</span> '
            'Wechselhafte Bedingungen, aber genug Daten.</li>'
            '<li><span class="bad">&lt; 50% Retention:</span> '
            'Viele Frames unbrauchbar — Ursache prüfen (Wolken, Nachführung, Fokus).</li>'
            '</ul>'
        )

    # Loss breakdown chart
    if plt is not None and len(stages) >= 2:
        fn = "frame_loss_breakdown.png"
        loss_labels = []
        loss_counts = []
        loss_colors = []

        if linearity_enabled and linearity_action == "removed" and linearity_failed > 0:
            loss_labels.append("Linearität")
            loss_counts.append(linearity_failed)
            loss_colors.append("#ff6b6b")

        if frames_excluded_identity > 0:
            loss_labels.append("Registrierung\n(Identity)")
            loss_counts.append(frames_excluded_identity)
            loss_colors.append("#ffb86c")

        if frames_excluded_negative > 0:
            loss_labels.append("Registrierung\n(Negativ)")
            loss_counts.append(frames_excluded_negative)
            loss_colors.append("#ff79c6")

        used = frames_usable_reg if frames_usable_reg > 0 else frames_after_scan
        if used > 0:
            loss_labels.append("Verwendet")
            loss_counts.append(used)
            loss_colors.append("#50fa7b")

        if loss_labels and sum(loss_counts) > 0:
            fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
            wedges, texts, autotexts = ax.pie(
                loss_counts, labels=loss_labels, colors=loss_colors,
                autopct=lambda pct: f"{int(round(pct * sum(loss_counts) / 100))}\n({pct:.0f}%)",
                startangle=90, textprops={"fontsize": 9})
            for t in autotexts:
                t.set_fontsize(8)
                t.set_color("white")
                t.set_fontweight("bold")
            ax.set_title("Frame-Verlust nach Ursache", fontsize=10)
            fig.tight_layout()
            fig.savefig(_fig_path(artifacts_dir, fn), bbox_inches="tight")
            plt.close(fig)
            pngs.append(fn)
            explanations[fn] = (
                '<h4>Verlust nach Ursache</h4>'
                '<p>Kreisdiagramm: Wohin gehen die Frames verloren?</p>'
                '<p><b>Kategorien:</b></p>'
                '<ul>'
                '<li><span class="bad">Linearität:</span> Frames mit nicht-linearer Kennlinie '
                '(z.B. Flats, Darks, überbelichtete Frames). '
                'Werden vor der Verarbeitung entfernt.</li>'
                '<li><span class="neutral">Registrierung (Identity):</span> '
                'Frames, die nicht registriert werden konnten — '
                'fallen auf Identity-Transformation zurück. '
                'Häufigste Ursache: Wolken oder starke Drift.</li>'
                '<li><span class="good">Verwendet:</span> '
                'Frames, die erfolgreich registriert wurden und '
                'zum Endergebnis beitragen.</li>'
                '</ul>'
            )

    return pngs, evals, explanations


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
.grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
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
.chart-row { display: flex; gap: 16px; margin-bottom: 14px; align-items: flex-start; }
.chart-col { flex: 2; min-width: 0; }
.chart-col img { width: 100%; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 8px; }
.explain-col { flex: 1; min-width: 220px; background: rgba(255,255,255,0.025); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.explain-col h4 { margin: 0 0 8px; font-size: 12px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.5px; }
.explain-col p { margin: 0 0 8px; font-size: 12px; line-height: 1.55; color: var(--muted); }
.explain-col .good { color: var(--ok); font-weight: 600; }
.explain-col .bad { color: var(--bad); font-weight: 600; }
.explain-col .neutral { color: var(--warn); font-weight: 600; }
.explain-col ul { margin: 4px 0 8px 14px; padding: 0; }
.explain-col ul li { margin: 2px 0; font-size: 11.5px; line-height: 1.5; color: var(--muted); }
.explain-col .metric-box { background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; margin: 8px 0; font-size: 11.5px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--text); line-height: 1.5; }
@media (max-width: 900px) { .chart-row { flex-direction: column; } .explain-col { min-width: unset; } }
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


def _make_chart_row(png: str, explain_html: str) -> str:
    """Build a single chart-row: 2/3 chart image + 1/3 explanation panel."""
    return (
        '<div class="chart-row">'
        f'<div class="chart-col"><img src="{_escape_html(png)}" loading="lazy"/></div>'
        f'<div class="explain-col">{explain_html}</div>'
        '</div>'
    )


def _make_card_html(title: str, pngs: list[str], evals: list[str],
                    status: str | None = None,
                    explanations: dict[str, str] | None = None) -> str:
    """Build a card. If explanations dict maps png filename -> HTML explanation,
    each chart gets a 2/3+1/3 row. Otherwise falls back to simple layout."""
    cls = ""
    badge = ""
    if status:
        s = status.lower()
        if s in ("ok", "warn", "bad"):
            cls = f" {s}"
            badge = f'<span class="badge {s}">{_escape_html(status.upper())}</span>'

    body_parts: list[str] = []

    if explanations:
        for p in pngs:
            expl = explanations.get(p, "")
            if expl:
                body_parts.append(_make_chart_row(p, expl))
            else:
                body_parts.append(f'<img src="{_escape_html(p)}" loading="lazy" style="width:100%;border-radius:8px;border:1px solid rgba(255,255,255,0.08);margin:6px 0"/>')
    else:
        for p in pngs:
            body_parts.append(f'<img src="{_escape_html(p)}" loading="lazy" style="width:100%;border-radius:8px;border:1px solid rgba(255,255,255,0.08);margin:6px 0"/>')

    items = "".join(
        f'<li class="{"warn" if "WARNING" in e.upper() else ""}">{_escape_html(e)}</li>'
        for e in evals if e
    )
    if items:
        body_parts.append(f'<div class="metric-box" style="margin-top:10px"><ul style="margin:0 0 0 14px;padding:0;list-style:disc">{items}</ul></div>')

    return (
        f'<div class="card{cls}">'
        f'<h3>{_escape_html(title)}{badge}</h3>'
        f'{"\n".join(body_parts)}'
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
    tl_pngs, tl_evals, tl_expl = _gen_timeline(artifacts_dir, events)
    if tl_pngs or tl_evals:
        sections.append(("Pipeline Timeline", _make_card_html("Phase durations", tl_pngs, tl_evals, explanations=tl_expl)))

    # 0b. Frame Usage Summary
    fu_pngs, fu_evals, fu_expl = _gen_frame_usage(artifacts_dir, events)
    if fu_pngs or fu_evals:
        sections.append(("Frame Usage", _make_card_html("Frame-Nutzung", fu_pngs, fu_evals, _infer_status(fu_evals), explanations=fu_expl)))

    # 1. Normalization
    if norm:
        n_pngs, n_evals, n_expl = _gen_normalization(artifacts_dir, norm)
        sections.append(("Normalization", _make_card_html("Background levels", n_pngs, n_evals, _infer_status(n_evals), explanations=n_expl)))

    # 2. Global Metrics
    if gm:
        g_pngs, g_evals, g_expl = _gen_global_metrics(artifacts_dir, gm)
        sections.append(("Global Metrics", _make_card_html("Frame quality & weights", g_pngs, g_evals, _infer_status(g_evals), explanations=g_expl)))

    # 3. Tile Grid
    if tg:
        t_pngs, t_evals, t_expl = _gen_tile_grid(artifacts_dir, tg)
        sections.append(("Tile Grid", _make_card_html("Grid layout", t_pngs, t_evals, explanations=t_expl)))

    # 4. Registration
    if reg:
        r_pngs, r_evals, r_expl = _gen_registration(artifacts_dir, reg)
        sections.append(("Global Registration", _make_card_html("Frame alignment", r_pngs, r_evals, _infer_status(r_evals), explanations=r_expl)))

    # 5. Local Metrics
    if lm:
        l_pngs, l_evals, l_expl = _gen_local_metrics(artifacts_dir, lm, tg)
        sections.append(("Local Metrics", _make_card_html("Per-tile quality", l_pngs, l_evals, _infer_status(l_evals), explanations=l_expl)))

    # 6. Tile Reconstruction
    if recon:
        rc_pngs, rc_evals, rc_expl = _gen_reconstruction(artifacts_dir, recon, tg)
        sections.append(("Tile Reconstruction", _make_card_html("Reconstruction stats", rc_pngs, rc_evals, _infer_status(rc_evals), explanations=rc_expl)))

    # 7. State Clustering
    if cl:
        cl_pngs, cl_evals, cl_expl = _gen_clustering(artifacts_dir, cl)
        sections.append(("State Clustering", _make_card_html("Cluster analysis", cl_pngs, cl_evals, explanations=cl_expl)))

    # 8. Synthetic Frames
    if syn:
        sy_pngs, sy_evals, sy_expl = _gen_synthetic(artifacts_dir, syn)
        sections.append(("Synthetic Frames", _make_card_html("Synthetic frame info", sy_pngs, sy_evals, explanations=sy_expl)))

    # 9. Validation
    if val:
        v_pngs, v_evals, v_expl = _gen_validation(artifacts_dir, val)
        sections.append(("Validation", _make_card_html("Quality validation", v_pngs, v_evals, _infer_status(v_evals), explanations=v_expl)))

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
