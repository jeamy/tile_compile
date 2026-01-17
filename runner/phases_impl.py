"""Main pipeline implementation (Methodik v3)."""

# Standard library
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np
from astropy.io import fits

# Local - core utilities
from .assumptions import get_assumptions_config, is_reduced_mode
from .events import phase_start, phase_end, phase_progress, stop_requested
from .fits_utils import is_fits_image_path, read_fits_float, fits_is_cfa, fits_get_bayerpat
from .utils import safe_symlink_or_copy, safe_hardlink_or_copy, pick_output_file
from .calibration import build_master_mean, bias_correct_dark, prepare_flat, apply_calibration

# Local - image processing (disk-based)
from .image_processing import (
    split_cfa_channels,
    demosaic_cfa,
    reassemble_cfa_mosaic,
    split_rgb_frame,
    normalize_frame,
    compute_frame_medians,
    warp_cfa_mosaic_via_subplanes,
    cfa_downsample_sum2x2,
    cosmetic_correction,
)

# Local - registration
from .opencv_registration import (
    opencv_prepare_ecc_image,
    opencv_count_stars,
    opencv_ecc_warp,
    opencv_best_translation_init,
)

# Local - Siril integration
from .siril_utils import validate_siril_script, run_siril_script, extract_siril_save_targets

try:
    import cv2
except Exception:
    cv2 = None

try:
    from tile_compile_backend.metrics import TileMetricsCalculator
except Exception:
    TileMetricsCalculator = None

try:
    from tile_compile_backend.synthetic import generate_channel_synthetic_frames
except Exception:
    generate_channel_synthetic_frames = None

try:
    from tile_compile_backend.clustering import cluster_channels
except Exception:
    cluster_channels = None

try:
    from tile_compile_backend.tile_grid import generate_multi_channel_grid
except Exception:
    generate_multi_channel_grid = None

try:
    from tile_compile_backend.linearity import validate_frames_linearity
except Exception:
    validate_frames_linearity = None

try:
    from tile_compile_backend.sigma_clipping import SigmaClipConfig, sigma_clip_stack_nd, simple_mean_stack_nd
except Exception:
    SigmaClipConfig = None
    sigma_clip_stack_nd = None
    simple_mean_stack_nd = None


def _to_uint8(img: np.ndarray) -> np.ndarray:
    f = np.asarray(img).astype("float32", copy=False)
    mn = float(np.min(f))
    mx = float(np.max(f))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros(f.shape, dtype=np.uint8)
    x = (f - mn) / (mx - mn)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _extract_siril_error_reason(log_file: str | None) -> str | None:
    if not log_file:
        return None

    try:
        p = Path(str(log_file)).expanduser()
        if not p.exists() or not p.is_file():
            return None
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    patterns = [
        re.compile(r"Nicht genug Speicher.*", re.IGNORECASE),
        re.compile(r"Nicht genug Speicherplatz.*", re.IGNORECASE),
        re.compile(r"Out of memory.*", re.IGNORECASE),
        re.compile(r"Script-Ausführung fehlgeschlagen.*", re.IGNORECASE),
        re.compile(r"Fehlgeschlagen.*", re.IGNORECASE),
        re.compile(r"abgebrochen.*", re.IGNORECASE),
    ]
    for raw in reversed(lines[-300:]):
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)
        s = re.sub(r"^\d+:\s*", "", s)
        s = re.sub(r"^log:\s*", "", s, flags=re.IGNORECASE)
        for pat in patterns:
            if pat.search(s):
                return s
    return None


def _note_plotting_issue(artifacts_dir: Path, where: str, err: Exception) -> None:
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fp = artifacts_dir / "plotting_issues.log"
        with fp.open("a", encoding="utf-8", errors="replace") as f:
            f.write(f"[{where}] {type(err).__name__}: {err}\n")
    except Exception:
        return


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _flatten_evaluations(ev: Any) -> list[str]:
    if ev is None:
        return []
    if isinstance(ev, list):
        return [str(x) for x in ev if x is not None]
    if isinstance(ev, dict):
        out: list[str] = []
        for _k, v in ev.items():
            out.extend(_flatten_evaluations(v))
        return out
    return [str(ev)]


def _infer_status_from_evaluations(ev: Any) -> str:
    texts = _flatten_evaluations(ev)
    s = "\n".join(texts).lower()
    if any(k in s for k in ("error", "failed", "exception", "traceback")):
        return "BAD"
    if any(k in s for k in ("nan", "inf", "not finite")):
        return "BAD"
    if "warning" in s:
        return "WARN"
    if any(k in s for k in ("insufficient", "missing", "skipped", "fallback")):
        return "WARN"
    try:
        m = re.search(r"low_weight_fraction=([0-9eE+\-\.]+)", s)
        if m:
            v = float(m.group(1))
            if np.isfinite(v) and v >= 0.15:
                return "WARN"
    except Exception:
        pass
    return "OK"


def _ensure_report_metrics_complete(artifacts_dir: Path) -> None:
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        p = artifacts_dir / "report_metrics.json"
        cur: dict[str, Any] = {}
        if p.exists() and p.is_file():
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                if isinstance(obj, dict):
                    cur = obj
            except Exception:
                cur = {}

        art = cur.get("artifacts") if isinstance(cur.get("artifacts"), dict) else {}
        if not isinstance(art, dict):
            art = {}

        patch_art: dict[str, Any] = {}
        for fp in artifacts_dir.iterdir():
            if not fp.is_file():
                continue
            suf = fp.suffix.lower()
            if suf not in (".png", ".json"):
                continue
            fn = fp.name
            entry = art.get(fn)
            needs = False
            if not isinstance(entry, dict):
                needs = True
            else:
                ev = entry.get("evaluations")
                if ev is None or ev == [] or ev == {}:
                    needs = True
            if not needs:
                continue

            eval_lines: list[str] = ["auto: artifact generated; no specific metrics recorded for this file"]
            try:
                eval_lines.append(f"size_bytes={int(fp.stat().st_size)}")
            except Exception:
                pass
            patch_art[fn] = {
                "phase": "DONE",
                "evaluations": eval_lines,
                "status": "WARN",
            }

        if patch_art:
            _update_report_metrics(artifacts_dir, {"artifacts": patch_art})
    except Exception:
        return


def _update_report_metrics(artifacts_dir: Path, patch: dict[str, Any]) -> None:
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        p = artifacts_dir / "report_metrics.json"
        cur: dict[str, Any] = {}
        if p.exists() and p.is_file():
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                if isinstance(obj, dict):
                    cur = obj
            except Exception:
                cur = {}
        if "version" not in cur:
            cur["version"] = 1
        if "generated_ts" not in cur:
            cur["generated_ts"] = datetime.now(timezone.utc).isoformat()
        _deep_merge(cur, patch)

        try:
            art = cur.get("artifacts") if isinstance(cur.get("artifacts"), dict) else None
            if isinstance(art, dict):
                for _fn, entry in art.items():
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("status") in ("OK", "WARN", "BAD"):
                        continue
                    if "evaluations" in entry:
                        entry["status"] = _infer_status_from_evaluations(entry.get("evaluations"))
        except Exception:
            pass

        cur["generated_ts"] = datetime.now(timezone.utc).isoformat()
        p.write_text(json.dumps(cur, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        return


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


def _eval_timeseries(label: str, vals: list[float]) -> list[str]:
    s = _basic_stats(vals)
    if int(s.get("n", 0.0)) <= 1:
        return [f"{label}: insufficient data"]
    med = float(s.get("median") or 0.0)
    span = float(s.get("max") or 0.0) - float(s.get("min") or 0.0)
    rel = (span / abs(med)) if med not in (0.0, -0.0) else float("inf")
    out = [
        f"{label}: median={med:.4g}, min={float(s.get('min') or 0.0):.4g}, max={float(s.get('max') or 0.0):.4g}",
        f"{label}: span/|median|={rel:.3g}",
    ]
    if np.isfinite(rel) and rel > 0.25:
        out.append(f"warning: {label} varies strongly over time")
    return out


def _eval_weights(vals: list[float]) -> list[str]:
    s = _basic_stats(vals)
    if int(s.get("n", 0.0)) <= 1:
        return ["G_f,c: insufficient data"]
    med = float(s.get("median") or 0.0)
    mn = float(s.get("min") or 0.0)
    mx = float(s.get("max") or 0.0)
    out = [f"G_f,c: median={med:.4g}, min={mn:.4g}, max={mx:.4g}"]
    if med > 0:
        low = sum(1 for x in vals if np.isfinite(x) and x < med * 0.2)
        out.append(f"frames with very low weight (<0.2×median): {low}")
    if mn > 0 and mx > 0:
        out.append(f"max/min ratio: {mx / mn:.3g}")
        if mx / mn > 50:
            out.append("warning: extremely wide weight distribution")
    return out


def _write_quality_analysis_pngs(artifacts_dir: Path, channel_metrics: dict[str, dict[str, Any]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        _note_plotting_issue(artifacts_dir, "quality_analysis_import", e)
        return []

    def _vals(ch: str, *path: str) -> list[float]:
        cur: Any = channel_metrics.get(ch, {})
        for k in path:
            if not isinstance(cur, dict):
                return []
            cur = cur.get(k)
        if cur is None:
            return []
        if isinstance(cur, np.ndarray):
            return [float(x) for x in cur.flatten().tolist()]
        if isinstance(cur, list):
            out: list[float] = []
            for x in cur:
                if isinstance(x, (int, float, np.number)):
                    out.append(float(x))
            return out
        return []

    out_paths: list[str] = []

    def _ch_any(ch: str) -> bool:
        return bool(
            _vals(ch, "global", "background_level")
            or _vals(ch, "global", "noise_level")
            or _vals(ch, "global", "gradient_energy")
            or _vals(ch, "global", "G_f_c")
            or _vals(ch, "tiles", "tile_quality_mean")
            or _vals(ch, "tiles", "tile_quality_variance")
        )

    channels = [ch for ch in ("R", "G", "B") if _ch_any(ch)]
    if not channels:
        return []

    try:
        rows = len(channels)
        cols = 6
        fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for r, ch in enumerate(channels):
            bg = _vals(ch, "global", "background_level")
            noise = _vals(ch, "global", "noise_level")
            grad = _vals(ch, "global", "gradient_energy")
            gfc = _vals(ch, "global", "G_f_c")
            tq_mean = _vals(ch, "tiles", "tile_quality_mean")
            tq_var = _vals(ch, "tiles", "tile_quality_variance")

            ax = axes[r][0]
            if gfc:
                sns.histplot(gfc, kde=True, ax=ax)
                ax.axvline(float(np.median(gfc)), color="r", linestyle="--")
            ax.set_title(f"{ch}: G_f,c")

            ax = axes[r][1]
            if noise:
                sns.histplot(noise, kde=True, ax=ax)
                ax.axvline(float(np.median(noise)), color="r", linestyle="--")
            ax.set_title(f"{ch}: Noise σ")

            ax = axes[r][2]
            if bg:
                sns.histplot(bg, kde=True, ax=ax)
                ax.axvline(float(np.median(bg)), color="r", linestyle="--")
            ax.set_title(f"{ch}: Background B")

            ax = axes[r][3]
            if grad:
                sns.histplot(grad, kde=True, ax=ax)
                ax.axvline(float(np.median(grad)), color="r", linestyle="--")
            ax.set_title(f"{ch}: Gradient E")

            ax = axes[r][4]
            if tq_mean:
                sns.histplot(tq_mean, kde=True, ax=ax)
                ax.axvline(float(np.median(tq_mean)), color="r", linestyle="--")
            ax.set_title(f"{ch}: mean(Q_local)")

            ax = axes[r][5]
            if noise and gfc:
                n = min(len(noise), len(gfc))
                ax.scatter(noise[:n], gfc[:n], alpha=0.6)
                ax.set_xlabel("σ")
                ax.set_ylabel("G_f,c")
                ax.set_title(f"{ch}: G_f,c vs σ")
            elif tq_var and tq_mean:
                n = min(len(tq_var), len(tq_mean))
                ax.scatter(tq_var[:n], tq_mean[:n], alpha=0.6)
                ax.set_xlabel("var(Q_local)")
                ax.set_ylabel("mean(Q_local)")
                ax.set_title(f"{ch}: mean vs var")
            else:
                ax.set_title(f"{ch}: scatter")

        fig.tight_layout()
        outp = artifacts_dir / "quality_analysis_combined.png"
        fig.savefig(str(outp), dpi=200)
        plt.close(fig)
        out_paths.append(str(outp))
    except Exception as e:
        _note_plotting_issue(artifacts_dir, "quality_analysis_runtime", e)
        try:
            plt.close("all")
        except Exception:
            pass
        return out_paths

    return out_paths


def _write_registration_artifacts(
    artifacts_dir: Path,
    registered_files: list[Path],
    corrs: list[float] | None = None,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        _note_plotting_issue(artifacts_dir, "registration_import", e)
        return []

    out_paths: list[str] = []

    def _luminance_from_fits(p: Path) -> np.ndarray | None:
        try:
            data = fits.getdata(str(p), ext=0)
            if data is None:
                return None
            a = np.asarray(data)
            if a.ndim == 2:
                return a.astype("float32", copy=False)
            if a.ndim == 3:
                if a.shape[-1] in (3, 4):
                    return np.mean(a[..., :3].astype("float32", copy=False), axis=-1)
                if a.shape[0] in (3, 4):
                    return np.mean(a[:3, ...].astype("float32", copy=False), axis=0)
            return a.astype("float32", copy=False)
        except Exception:
            return None

    try:
        if not registered_files:
            return []
        idxs = sorted(set([0, max(0, len(registered_files) // 2), max(0, len(registered_files) - 1)]))
        ref = _luminance_from_fits(registered_files[idxs[0]])
        if ref is None:
            return []

        fig, axes = plt.subplots(1, 1 + len(idxs), figsize=(4 * (1 + len(idxs)), 4))
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])
        axes = axes.flatten()

        axes[0].imshow(_to_uint8(ref), cmap="gray", interpolation="nearest")
        axes[0].set_title("ref")
        axes[0].set_axis_off()

        for j, i in enumerate(idxs, start=1):
            cur = _luminance_from_fits(registered_files[i])
            if cur is None:
                continue
            if cur.shape != ref.shape:
                cur = cur[: ref.shape[0], : ref.shape[1]]
            diff = np.abs(cur.astype("float32", copy=False) - ref.astype("float32", copy=False))
            axes[j].imshow(_to_uint8(diff), cmap="magma", interpolation="nearest")
            axes[j].set_title(f"absdiff idx={i}")
            axes[j].set_axis_off()

        fig.tight_layout()
        outp = artifacts_dir / "registration_absdiff_samples.png"
        fig.savefig(str(outp), dpi=200)
        plt.close(fig)
        out_paths.append(str(outp))

        if corrs:
            c = np.asarray(corrs, dtype=np.float32)
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(np.arange(c.shape[0]), c, linewidth=1.0)
            ax.set_title("ECC correlation over frames")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("ecc_corr")
            fig.tight_layout()
            outp = artifacts_dir / "registration_ecc_corr_timeseries.png"
            fig.savefig(str(outp), dpi=200)
            plt.close(fig)
            out_paths.append(str(outp))

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.hist(c, bins=40)
            ax.set_title("ECC correlation histogram")
            ax.set_xlabel("ecc_corr")
            ax.set_ylabel("count")
            fig.tight_layout()
            outp = artifacts_dir / "registration_ecc_corr_hist.png"
            fig.savefig(str(outp), dpi=200)
            plt.close(fig)
            out_paths.append(str(outp))

    except Exception as e:
        _note_plotting_issue(artifacts_dir, "registration_runtime", e)
        try:
            plt.close("all")
        except Exception:
            pass
        return out_paths

    return out_paths


def _write_normalization_artifacts(
    artifacts_dir: Path,
    pre_norm_backgrounds: dict[str, list[float]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        _note_plotting_issue(artifacts_dir, "normalization_import", e)
        return []

    out_paths: list[str] = []
    try:
        if not pre_norm_backgrounds:
            return []
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for ch in ("R", "G", "B"):
            vals = pre_norm_backgrounds.get(ch) or []
            if not vals:
                continue
            ax.plot(np.arange(len(vals)), np.asarray(vals, dtype=np.float32), linewidth=1.0, label=ch)
        ax.set_title("Pre-normalization background B_f (median) per channel")
        ax.set_xlabel("frame_index")
        ax.set_ylabel("B_f")
        ax.legend(loc="best")
        fig.tight_layout()
        outp = artifacts_dir / "normalization_background_timeseries.png"
        fig.savefig(str(outp), dpi=200)
        plt.close(fig)
        out_paths.append(str(outp))
    except Exception as e:
        _note_plotting_issue(artifacts_dir, "normalization_runtime", e)
        try:
            plt.close("all")
        except Exception:
            pass
        return out_paths

    return out_paths


def _write_global_metrics_artifacts(
    artifacts_dir: Path,
    channel_metrics: dict[str, dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        _note_plotting_issue(artifacts_dir, "global_metrics_import", e)
        return []

    out_paths: list[str] = []
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for ch in ("R", "G", "B"):
            gfc = channel_metrics.get(ch, {}).get("global", {}).get("G_f_c")
            if not gfc:
                continue
            g = np.asarray(gfc, dtype=np.float32)
            ax.plot(np.arange(g.shape[0]), g, linewidth=1.0, label=ch)
        ax.set_title("Global weight G_f,c over frames")
        ax.set_xlabel("frame_index")
        ax.set_ylabel("G_f,c")
        ax.legend(loc="best")
        fig.tight_layout()
        outp = artifacts_dir / "global_weight_timeseries.png"
        fig.savefig(str(outp), dpi=200)
        plt.close(fig)
        out_paths.append(str(outp))

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for ch in ("R", "G", "B"):
            gfc = channel_metrics.get(ch, {}).get("global", {}).get("G_f_c")
            if not gfc:
                continue
            g = np.asarray(gfc, dtype=np.float32)
            ax.hist(g, bins=40, alpha=0.5, label=ch)
        ax.set_title("Histogram of global weights G_f,c")
        ax.set_xlabel("G_f,c")
        ax.set_ylabel("count")
        ax.legend(loc="best")
        fig.tight_layout()
        outp = artifacts_dir / "global_weight_hist.png"
        fig.savefig(str(outp), dpi=200)
        plt.close(fig)
        out_paths.append(str(outp))
    except Exception as e:
        _note_plotting_issue(artifacts_dir, "global_metrics_runtime", e)
        try:
            plt.close("all")
        except Exception:
            pass
        return out_paths

    return out_paths


def _write_clustering_artifacts(
    artifacts_dir: Path,
    channel_metrics: dict[str, dict[str, Any]],
    clustering_results: Any,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        _note_plotting_issue(artifacts_dir, "clustering_import", e)
        return []

    out_paths: list[str] = []

    def _labels_for_channel(ch: str) -> list[int] | None:
        if not clustering_results:
            return None
        if isinstance(clustering_results, dict) and ch in clustering_results and isinstance(clustering_results.get(ch), dict):
            cr = clustering_results.get(ch) or {}
            labels = cr.get("labels", cr.get("cluster_labels"))
            if isinstance(labels, list):
                return [int(x) for x in labels]
        if isinstance(clustering_results, dict):
            labels = clustering_results.get("labels", clustering_results.get("cluster_labels"))
            if isinstance(labels, list):
                return [int(x) for x in labels]
        return None

    try:
        for ch in ("R", "G", "B"):
            labels = _labels_for_channel(ch)
            gfc = channel_metrics.get(ch, {}).get("global", {}).get("G_f_c")
            if labels is None or not gfc or len(labels) != len(gfc):
                continue
            lab = np.asarray(labels, dtype=np.int32)
            g = np.asarray(gfc, dtype=np.float32)
            k = int(np.max(lab)) + 1 if lab.size else 0
            if k <= 0:
                continue
            counts = np.bincount(lab, minlength=k)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            ax0, ax1 = axes[0], axes[1]
            ax0.bar(np.arange(k), counts)
            ax0.set_title(f"Cluster sizes ({ch})")
            ax0.set_xlabel("cluster_id")
            ax0.set_ylabel("n_frames")

            data = []
            for ci in range(k):
                vals = g[lab == ci]
                if vals.size:
                    data.append(vals)
                else:
                    data.append(np.asarray([0.0], dtype=np.float32))
            ax1.boxplot(data, showfliers=False)
            ax1.set_title(f"G_f,c per cluster ({ch})")
            ax1.set_xlabel("cluster_id")
            ax1.set_ylabel("G_f,c")
            fig.tight_layout()
            outp = artifacts_dir / f"clustering_summary_{ch}.png"
            fig.savefig(str(outp), dpi=200)
            plt.close(fig)
            out_paths.append(str(outp))

    except Exception as e:
        _note_plotting_issue(artifacts_dir, "clustering_runtime", e)
        try:
            plt.close("all")
        except Exception:
            pass
        return out_paths

    return out_paths


def _write_tile_grid_pngs(
    artifacts_dir: Path,
    rep_frames: dict[str, "np.ndarray"],
    tile_grids: dict[str, dict[str, Any]],
    grid_cfg: dict[str, Any],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        _note_plotting_issue(artifacts_dir, "tile_grid_import", e)
        return []

    out_paths: list[str] = []

    try:
        tile_size = int(grid_cfg.get("tile_size") or 0)
        step_size = int(grid_cfg.get("step_size") or 0)
        if tile_size <= 0 or step_size <= 0:
            return []

        for ch, frame in rep_frames.items():
            if frame is None:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            ax.imshow(_to_uint8(frame), cmap="gray", interpolation="nearest")
            h, w = frame.shape[:2]
            for x in range(0, w - tile_size + 1, step_size):
                ax.axvline(x, color="lime", linewidth=0.5, alpha=0.6)
            ax.axvline(max(0, w - tile_size), color="lime", linewidth=0.5, alpha=0.6)
            for y in range(0, h - tile_size + 1, step_size):
                ax.axhline(y, color="lime", linewidth=0.5, alpha=0.6)
            ax.axhline(max(0, h - tile_size), color="lime", linewidth=0.5, alpha=0.6)
            ax.set_title(f"TILE_GRID overlay ({ch})")
            ax.set_axis_off()
            fig.tight_layout()
            outp = artifacts_dir / f"tile_grid_overlay_{ch}.png"
            fig.savefig(str(outp), dpi=200)
            plt.close(fig)
            out_paths.append(str(outp))

        try:
            import json

            meta_ch = "G" if "G" in tile_grids else (next(iter(tile_grids.keys())) if tile_grids else None)
            grid_meta = tile_grids.get(meta_ch, {}).get("grid_metadata", {}) if meta_ch else {}
            payload = {
                "grid_cfg": grid_cfg,
                "grid_metadata": grid_meta,
            }
            outj = artifacts_dir / "tile_grid.json"
            outj.write_text(json.dumps(payload, indent=2))
            out_paths.append(str(outj))
        except Exception:
            pass

    except Exception as e:
        _note_plotting_issue(artifacts_dir, "tile_grid_runtime", e)
        return out_paths

    return out_paths


def _write_tile_quality_heatmaps(
    artifacts_dir: Path,
    channel_metrics: dict[str, dict[str, Any]],
    grid_cfg: dict[str, Any],
    image_shape: tuple[int, int],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        _note_plotting_issue(artifacts_dir, "tile_heatmaps_import", e)
        return []

    out_paths: list[str] = []

    try:
        h0, w0 = image_shape
        tile_size = int(grid_cfg.get("tile_size") or 0)
        step_size = int(grid_cfg.get("step_size") or 0)
        if tile_size <= 0 or step_size <= 0:
            return []

        n_tiles_y = max(1, (h0 - tile_size) // step_size + 1)
        n_tiles_x = max(1, (w0 - tile_size) // step_size + 1)
        n_tiles = n_tiles_y * n_tiles_x

        for ch in ("R", "G", "B"):
            tiles = channel_metrics.get(ch, {}).get("tiles", {})
            q_mean_tile = tiles.get("Q_local_tile_mean")
            q_var_tile = tiles.get("Q_local_tile_var")
            l_mean_tile = tiles.get("L_local_tile_mean")
            l_var_tile = tiles.get("L_local_tile_var")

            if isinstance(q_mean_tile, list) and len(q_mean_tile) == n_tiles:
                mean_per_tile = np.asarray(q_mean_tile, dtype=np.float32)
            else:
                q_local = tiles.get("Q_local", [])
                if not q_local:
                    continue
                a = np.asarray(q_local, dtype=np.float32)
                if a.ndim != 2 or a.shape[1] <= 0:
                    continue
                if a.shape[1] != n_tiles:
                    continue
                mean_per_tile = np.mean(a, axis=0)

            grid = mean_per_tile.reshape((n_tiles_y, n_tiles_x))
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            im = ax.imshow(grid, cmap="magma", interpolation="nearest")
            ax.set_title(f"Tile mean Q_local ({ch})")
            ax.set_xlabel("tile_x")
            ax.set_ylabel("tile_y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            outp = artifacts_dir / f"tile_quality_heatmap_{ch}.png"
            fig.savefig(str(outp), dpi=200)
            plt.close(fig)
            out_paths.append(str(outp))

            if isinstance(q_var_tile, list) and len(q_var_tile) == n_tiles:
                var_per_tile = np.asarray(q_var_tile, dtype=np.float32)
            else:
                q_local = tiles.get("Q_local", [])
                if not q_local:
                    continue
                a = np.asarray(q_local, dtype=np.float32)
                if a.ndim != 2 or a.shape[1] != n_tiles:
                    continue
                var_per_tile = np.var(a, axis=0)
            grid_v = var_per_tile.reshape((n_tiles_y, n_tiles_x))
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            im = ax.imshow(grid_v, cmap="viridis", interpolation="nearest")
            ax.set_title(f"Tile var Q_local ({ch})")
            ax.set_xlabel("tile_x")
            ax.set_ylabel("tile_y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            outp = artifacts_dir / f"tile_quality_var_heatmap_{ch}.png"
            fig.savefig(str(outp), dpi=200)
            plt.close(fig)
            out_paths.append(str(outp))

            if isinstance(l_mean_tile, list) and len(l_mean_tile) == n_tiles:
                mean_l = np.asarray(l_mean_tile, dtype=np.float32)
                grid_l = mean_l.reshape((n_tiles_y, n_tiles_x))
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                im = ax.imshow(grid_l, cmap="plasma", interpolation="nearest")
                ax.set_title(f"Tile mean L_local ({ch})")
                ax.set_xlabel("tile_x")
                ax.set_ylabel("tile_y")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                outp = artifacts_dir / f"tile_weight_heatmap_{ch}.png"
                fig.savefig(str(outp), dpi=200)
                plt.close(fig)
                out_paths.append(str(outp))

                if isinstance(l_var_tile, list) and len(l_var_tile) == n_tiles:
                    var_l = np.asarray(l_var_tile, dtype=np.float32)
                    grid_lv = var_l.reshape((n_tiles_y, n_tiles_x))
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    im = ax.imshow(grid_lv, cmap="cividis", interpolation="nearest")
                    ax.set_title(f"Tile var L_local ({ch})")
                    ax.set_xlabel("tile_x")
                    ax.set_ylabel("tile_y")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    outp = artifacts_dir / f"tile_weight_var_heatmap_{ch}.png"
                    fig.savefig(str(outp), dpi=200)
                    plt.close(fig)
                    out_paths.append(str(outp))

    except Exception as e:
        _note_plotting_issue(artifacts_dir, "tile_heatmaps_runtime", e)
        try:
            plt.close("all")
        except Exception:
            pass
        return out_paths

    return out_paths


def run_phases_impl(
    run_id: str,
    log_fp,
    dry_run: bool,
    run_dir: Path,
    project_root: Path,
    cfg: dict[str, Any],
    frames: list[Path],
    siril_exe: str | None,
    stop_flag: bool,
    resume_from_phase: Optional[int] = None,
) -> bool:
    outputs_dir = run_dir / "outputs"
    artifacts_dir = run_dir / "artifacts"
    work_dir = run_dir / "work"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        phases = [
            (0, "SCAN_INPUT"),
            (1, "REGISTRATION"),
            (2, "CHANNEL_SPLIT"),
            (3, "NORMALIZATION"),
            (4, "GLOBAL_METRICS"),
            (5, "TILE_GRID"),
            (6, "LOCAL_METRICS"),
            (7, "TILE_RECONSTRUCTION"),
            (8, "STATE_CLUSTERING"),
            (9, "SYNTHETIC_FRAMES"),
            (10, "STACKING"),
            (11, "DEBAYER"),
            (12, "DONE"),
        ]
        for phase_id, phase_name in phases:
            phase_start(run_id, log_fp, phase_id, phase_name)
            if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
                return False
            phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "dry_run"})
        return True

    d = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    frames_min = d.get("frames_min")
    color_mode = str(d.get("color_mode") or "").strip().upper()
    bayer_pattern = str(d.get("bayer_pattern") or "GBRG").strip().upper()
    try:
        frames_min_i = int(frames_min) if frames_min is not None else None
    except Exception:
        frames_min_i = None
    if frames_min_i is not None and frames_min_i > 0 and len(frames) < frames_min_i:
        phase_start(run_id, log_fp, 0, "SCAN_INPUT")
        phase_end(
            run_id,
            log_fp,
            0,
            "SCAN_INPUT",
            "error",
            {"error": f"frames.count={len(frames)} < data.frames_min={frames_min_i}"},
        )
        return False

    registration_cfg = cfg.get("registration") if isinstance(cfg.get("registration"), dict) else {}
    stacking_cfg = cfg.get("stacking") if isinstance(cfg.get("stacking"), dict) else {}
    synthetic_cfg = cfg.get("synthetic") if isinstance(cfg.get("synthetic"), dict) else {}
    debayer_enabled = bool(cfg.get("debayer", True))

    reg_engine = str(registration_cfg.get("engine") or "")

    reg_script_cfg = registration_cfg.get("siril_script")
    reg_script_path = (
        Path(str(reg_script_cfg)).expanduser().resolve()
        if isinstance(reg_script_cfg, str) and reg_script_cfg.strip()
        else (project_root / "siril_register_osc.ssf").resolve()
    )

    if not (isinstance(reg_script_cfg, str) and reg_script_cfg.strip()):
        if not reg_script_path.exists():
            alt = (project_root / "siril_scripts" / "siril_register_osc.ssf").resolve()
            if alt.exists():
                reg_script_path = alt

    stack_method_cfg = str(stacking_cfg.get("method") or "")
    stack_method = stack_method_cfg.strip().lower()
    sigma_clip_cfg = stacking_cfg.get("sigma_clip") if isinstance(stacking_cfg.get("sigma_clip"), dict) else {}

    reg_out_name = str(registration_cfg.get("output_dir") or "registered")
    reg_pattern = str(registration_cfg.get("registered_filename_pattern") or "reg_{index:05d}.fit")

    stack_input_dir_name = str(stacking_cfg.get("input_dir") or reg_out_name)
    stack_input_pattern = str(stacking_cfg.get("input_pattern") or "reg_*.fit")

    stack_output_file = str(stacking_cfg.get("output_file") or "stacked.fit")
    # stack_method was already parsed above to pick a default script.

    # Helper function to check if phase should be skipped during resume
    def should_skip_phase(phase_num: int) -> bool:
        if resume_from_phase is None:
            return False
        return phase_num < resume_from_phase

    if resume_from_phase is not None and resume_from_phase >= 10:
        stacked_path = None
        stacked_hdr = None

        if resume_from_phase <= 10:
            phase_id = 10
            phase_name = "STACKING"
            phase_start(run_id, log_fp, phase_id, phase_name)
            if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
                return False

            stack_src_dir = outputs_dir / Path(stack_input_dir_name)
            syn_r = sorted([p for p in stack_src_dir.glob("synR_*.fits") if p.is_file() and is_fits_image_path(p)])
            syn_g = sorted([p for p in stack_src_dir.glob("synG_*.fits") if p.is_file() and is_fits_image_path(p)])
            syn_b = sorted([p for p in stack_src_dir.glob("synB_*.fits") if p.is_file() and is_fits_image_path(p)])
            common_n = min(len(syn_r), len(syn_g), len(syn_b))

            if common_n > 0:
                syn_r = syn_r[:common_n]
                syn_g = syn_g[:common_n]
                syn_b = syn_b[:common_n]

                def _stack_file_list(_files: list[Path]) -> tuple[np.ndarray, Dict[str, Any], bool]:
                    _frames_list: list[np.ndarray] = []
                    for _fp in _files:
                        try:
                            _arr = np.asarray(fits.getdata(str(_fp), ext=0)).astype("float32", copy=False)
                        except Exception:
                            continue
                        _frames_list.append(_arr)
                    if not _frames_list:
                        return np.zeros((1, 1), dtype=np.float32), {"error": "failed to load frames"}, False
                    _stack_arr = np.stack(_frames_list, axis=0)
                    _use_sigma = SigmaClipConfig is not None and sigma_clip_stack_nd is not None and stack_method == "rej"
                    _sigma_stats: Dict[str, Any] = {}
                    if _use_sigma:
                        _sigma_cfg_dict: Dict[str, Any] = {
                            "sigma_low": float(sigma_clip_cfg.get("sigma_low", 4.0)),
                            "sigma_high": float(sigma_clip_cfg.get("sigma_high", 4.0)),
                            "max_iters": int(sigma_clip_cfg.get("max_iters", 3)),
                            "min_fraction": float(sigma_clip_cfg.get("min_fraction", 0.5)),
                        }
                        try:
                            _clipped_mean, _mask, _stats = sigma_clip_stack_nd(_stack_arr, _sigma_cfg_dict)
                            _final = _clipped_mean.astype("float32", copy=False)
                            _sigma_stats = _stats
                        except Exception as e:  # noqa: BLE001
                            _final = _stack_arr.mean(axis=0).astype("float32", copy=False)
                            _sigma_stats = {"error": str(e)}
                            _use_sigma = False
                    else:
                        _final = _stack_arr.mean(axis=0).astype("float32", copy=False)
                    return _final, _sigma_stats, _use_sigma

                try:
                    hdr_template = fits.getheader(str(syn_r[0]), ext=0)
                except Exception:
                    hdr_template = None

                r_final, r_stats, r_sigma = _stack_file_list(syn_r)
                g_final, g_stats, g_sigma = _stack_file_list(syn_g)
                b_final, b_stats, b_sigma = _stack_file_list(syn_b)

                try:
                    fits.writeto(str(outputs_dir / "stacked_R.fits"), r_final, header=hdr_template, overwrite=True)
                    fits.writeto(str(outputs_dir / "stacked_G.fits"), g_final, header=hdr_template, overwrite=True)
                    fits.writeto(str(outputs_dir / "stacked_B.fits"), b_final, header=hdr_template, overwrite=True)
                except Exception:
                    pass

                final_data = reassemble_cfa_mosaic(r_final, g_final, b_final, bayer_pattern)
                final_out = outputs_dir / Path(stack_output_file)
                final_out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    fits.writeto(str(final_out), final_data, header=hdr_template, overwrite=True)
                except Exception as e:  # noqa: BLE001
                    phase_end(
                        run_id,
                        log_fp,
                        phase_id,
                        phase_name,
                        "error",
                        {"error": f"failed to write stacked output: {e}"},
                    )
                    return False

                extra: Dict[str, Any] = {
                    "siril": None,
                    "method": stack_method,
                    "output": str(final_out),
                    "used_reconstructed_fallback": False,
                    "fallback_reason": None,
                    "sigma_clipping_used": bool(r_sigma or g_sigma or b_sigma),
                    "sigma_stats": {"R": r_stats, "G": g_stats, "B": b_stats},
                }
                phase_end(run_id, log_fp, phase_id, phase_name, "ok", extra)

                stacked_path = final_out
                stacked_hdr = hdr_template
            else:
                stack_files = (
                    sorted([p for p in stack_src_dir.glob(stack_input_pattern) if p.is_file() and is_fits_image_path(p)])
                    if stack_src_dir.exists()
                    else []
                )
                if stack_files:
                    rgb_syn = [p for p in stack_files if re.match(r"^syn_\d{5}\.fits$", p.name)]
                    if rgb_syn:
                        stack_files = sorted(rgb_syn)

                if not stack_files:
                    phase_end(
                        run_id,
                        log_fp,
                        phase_id,
                        phase_name,
                        "error",
                        {"error": "no stacking input frames found", "input_dir": str(stack_src_dir), "input_pattern": stack_input_pattern},
                    )
                    return False

                frames_list: list[np.ndarray] = []
                for fp in stack_files:
                    try:
                        arr = np.asarray(fits.getdata(str(fp), ext=0)).astype("float32", copy=False)
                    except Exception:
                        continue
                    frames_list.append(arr)

                if not frames_list:
                    phase_end(
                        run_id,
                        log_fp,
                        phase_id,
                        phase_name,
                        "error",
                        {"error": "failed to load stacking input frames", "input_dir": str(stack_src_dir)},
                    )
                    return False

                stack_arr = np.stack(frames_list, axis=0)
                use_sigma = SigmaClipConfig is not None and sigma_clip_stack_nd is not None and stack_method == "rej"
                if use_sigma:
                    sigma_cfg_dict: Dict[str, Any] = {
                        "sigma_low": float(sigma_clip_cfg.get("sigma_low", 4.0)),
                        "sigma_high": float(sigma_clip_cfg.get("sigma_high", 4.0)),
                        "max_iters": int(sigma_clip_cfg.get("max_iters", 3)),
                        "min_fraction": float(sigma_clip_cfg.get("min_fraction", 0.5)),
                    }
                    try:
                        clipped_mean, mask, stats = sigma_clip_stack_nd(stack_arr, sigma_cfg_dict)
                        final_data = clipped_mean.astype("float32", copy=False)
                        sigma_stats = stats
                    except Exception as e:  # noqa: BLE001
                        final_data = stack_arr.mean(axis=0).astype("float32", copy=False)
                        sigma_stats = {"error": str(e)}
                        use_sigma = False
                else:
                    final_data = stack_arr.mean(axis=0).astype("float32", copy=False)
                    sigma_stats = {}

                final_out = outputs_dir / Path(stack_output_file)
                final_out.parent.mkdir(parents=True, exist_ok=True)
                hdr_template = None
                try:
                    hdr_template = fits.getheader(str(stack_files[0]), ext=0)
                except Exception:
                    hdr_template = None
                try:
                    fits.writeto(str(final_out), final_data, header=hdr_template, overwrite=True)
                except Exception as e:  # noqa: BLE001
                    phase_end(
                        run_id,
                        log_fp,
                        phase_id,
                        phase_name,
                        "error",
                        {"error": f"failed to write stacked output: {e}"},
                    )
                    return False

                extra: Dict[str, Any] = {
                    "siril": None,
                    "method": stack_method,
                    "output": str(final_out),
                    "used_reconstructed_fallback": False,
                    "fallback_reason": None,
                    "sigma_clipping_used": bool(use_sigma),
                    "sigma_stats": sigma_stats,
                }
                phase_end(run_id, log_fp, phase_id, phase_name, "ok", extra)

                stacked_path = final_out
                stacked_hdr = hdr_template

    # Fast-path: resume from DEBAYER (or later) without re-running earlier phases.
    # This is especially useful when only debayering changes and stacked output already exists.
    if resume_from_phase is not None and resume_from_phase >= 10:
        if stacked_path is None:
            try:
                stacked_path = outputs_dir / Path(stack_output_file)
                if stacked_path.is_file():
                    try:
                        stacked_hdr = fits.getheader(str(stacked_path), ext=0)
                    except Exception:
                        stacked_hdr = None
                else:
                    stacked_path = None
            except Exception:
                stacked_path = None
                stacked_hdr = None

        if resume_from_phase <= 11:
            phase_id = 11
            phase_name = "DEBAYER"
            phase_start(run_id, log_fp, phase_id, phase_name)
            if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
                return False

            if not debayer_enabled:
                phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "disabled"})
            else:
                if stacked_path is None or not stacked_path.is_file():
                    phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "missing stacked output for debayer"})
                    return False

                try:
                    stacked_data = np.asarray(fits.getdata(str(stacked_path), ext=0)).astype("float32", copy=False)
                except Exception as e:  # noqa: BLE001
                    phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"failed to read stacked output: {e}"})
                    return False

                rgb_out = outputs_dir / "stacked_rgb.fits"
                rgb = None
                try:
                    if stacked_data.ndim == 2:
                        rgb = demosaic_cfa(stacked_data, bayer_pattern)
                    elif stacked_data.ndim == 3:
                        if stacked_data.shape[0] == 3:
                            rgb = stacked_data
                        elif stacked_data.shape[2] == 3:
                            rgb = np.transpose(stacked_data, (2, 0, 1)).astype("float32", copy=False)
                except Exception:
                    rgb = None

                if rgb is None:
                    phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "not_applicable"})
                else:
                    try:
                        fits.writeto(str(rgb_out), rgb, header=stacked_hdr, overwrite=True)
                    except Exception as e:  # noqa: BLE001
                        phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"failed to write debayer output: {e}"})
                        return False
                    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"output": str(rgb_out)})

        if resume_from_phase <= 12:
            phase_id = 12
            phase_name = "DONE"
            phase_start(run_id, log_fp, phase_id, phase_name)
            if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
                return False
            try:
                _ensure_report_metrics_complete(artifacts_dir)
            except Exception:
                pass
            phase_end(run_id, log_fp, phase_id, phase_name, "ok", {})

        return True

    phase_id = 0
    phase_name = "SCAN_INPUT"
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
    else:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
            return False
        cfa_flag0 = fits_is_cfa(frames[0]) if frames else None
        header_bayerpat0 = fits_get_bayerpat(frames[0]) if frames else None
        phase_progress(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            1,
            1,
            {
                "substep": "scan",
                "frame_count": len(frames),
                "color_mode": color_mode,
                "bayer_pattern": bayer_pattern,
                "bayer_pattern_header": header_bayerpat0,
                "cfa": cfa_flag0,
            },
        )

    calibration_cfg = cfg.get("calibration") if isinstance(cfg.get("calibration"), dict) else {}
    use_bias = bool(calibration_cfg.get("use_bias"))
    use_dark = bool(calibration_cfg.get("use_dark"))
    use_flat = bool(calibration_cfg.get("use_flat"))

    calibrated_applied = False

    def _hdr_get_float(hdr: Any, keys: list[str]) -> float | None:
        if hdr is None:
            return None
        for k in keys:
            try:
                if isinstance(hdr, dict):
                    v = hdr.get(k)
                else:
                    v = hdr[k] if k in hdr else None
            except Exception:
                v = None
            if v is None:
                continue
            try:
                f = float(v)
                if np.isfinite(f):
                    return f
            except Exception:
                continue
        return None

    def _get_exposure_s(path: Path) -> float | None:
        try:
            hdr = fits.getheader(str(path), ext=0)
        except Exception:
            return None
        return _hdr_get_float(hdr, ["EXPTIME", "EXPOSURE", "EXPOSURETIME", "EXP_TIME", "DURATION"])

    def _get_ccd_temp_c(path: Path) -> float | None:
        try:
            hdr = fits.getheader(str(path), ext=0)
        except Exception:
            return None
        return _hdr_get_float(hdr, ["CCD-TEMP", "CCD_TEMP", "CCD_TEMP_C", "SENSOR_T", "SENSORTEMP", "TEMP", "TEMPERAT"])

    def _collect_calib_files(dir_s: str | None, pattern: str) -> list[Path]:
        if not dir_s:
            return []
        p = Path(str(dir_s)).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        if not p.exists() or not p.is_dir():
            return []
        out = [q for q in p.glob(pattern) if q.is_file() and is_fits_image_path(q)]
        out.sort(key=lambda x: x.name)
        return out

    def _load_master(path_s: str | None) -> tuple[np.ndarray, Any] | None:
        if not path_s:
            return None
        p = Path(str(path_s)).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        if not p.exists() or not p.is_file():
            return None
        try:
            return read_fits_float(p)
        except Exception:
            return None

    if resume_from_phase is None and (use_bias or use_dark or use_flat):
        calibrated_applied = True
        cal_pattern = str(calibration_cfg.get("pattern") or "*.fit*")
        bias_master = _load_master(str(calibration_cfg.get("bias_master") or "").strip() or None) if use_bias else None
        dark_master = _load_master(str(calibration_cfg.get("dark_master") or "").strip() or None) if use_dark else None
        flat_master = _load_master(str(calibration_cfg.get("flat_master") or "").strip() or None) if use_flat else None

        out_cal_dir = outputs_dir / "calibration"
        out_cal_dir.mkdir(parents=True, exist_ok=True)

        if use_bias and bias_master is None:
            bias_files = _collect_calib_files(str(calibration_cfg.get("bias_dir") or "").strip() or None, cal_pattern)
            phase_progress(run_id, log_fp, 0, "SCAN_INPUT", 0, max(1, len(bias_files)), {"substep": "bias_master"})
            bias_master = build_master_mean(bias_files)
            if bias_master is not None:
                fits.writeto(str(out_cal_dir / "master_bias.fit"), bias_master[0], header=bias_master[1], overwrite=True)

        if use_dark and dark_master is None:
            dark_files = _collect_calib_files(str(calibration_cfg.get("darks_dir") or "").strip() or None, cal_pattern)

            dark_auto_select = bool(calibration_cfg.get("dark_auto_select", True))
            tol_pct = calibration_cfg.get("dark_match_exposure_tolerance_percent", 5.0)
            try:
                tol_pct_f = float(tol_pct)
            except Exception:
                tol_pct_f = 5.0
            use_temp_match = bool(calibration_cfg.get("dark_match_use_temp", False))
            temp_tol = calibration_cfg.get("dark_match_temp_tolerance_c", 2.0)
            try:
                temp_tol_f = float(temp_tol)
            except Exception:
                temp_tol_f = 2.0

            selected_dark_files = dark_files
            warnings: list[str] = []

            light_exp_s: float | None = None
            light_temp_c: float | None = None
            try:
                exp_vals: list[float] = []
                temp_vals: list[float] = []
                for p0 in frames[: min(10, len(frames))]:
                    e0 = _get_exposure_s(p0)
                    if e0 is not None and e0 > 0:
                        exp_vals.append(float(e0))
                    if use_temp_match:
                        t0 = _get_ccd_temp_c(p0)
                        if t0 is not None and np.isfinite(t0):
                            temp_vals.append(float(t0))
                if exp_vals:
                    light_exp_s = float(np.median(np.asarray(exp_vals, dtype=np.float64)))
                if temp_vals:
                    light_temp_c = float(np.median(np.asarray(temp_vals, dtype=np.float64)))
            except Exception:
                pass

            if dark_auto_select and dark_files:
                if light_exp_s is None or not np.isfinite(light_exp_s) or light_exp_s <= 0:
                    warnings.append("dark_auto_select: light exposure not found; using all darks")
                else:
                    try:
                        cand: list[Path] = []
                        cand_missing_temp: int = 0
                        for dp in dark_files:
                            de = _get_exposure_s(dp)
                            if de is None or not np.isfinite(de) or de <= 0:
                                continue
                            rel_pct = abs(float(de) - float(light_exp_s)) / float(light_exp_s) * 100.0
                            if rel_pct > tol_pct_f:
                                continue
                            if use_temp_match:
                                if light_temp_c is None or not np.isfinite(light_temp_c):
                                    cand.append(dp)
                                    continue
                                dt = _get_ccd_temp_c(dp)
                                if dt is None or not np.isfinite(dt):
                                    cand_missing_temp += 1
                                    continue
                                if abs(float(dt) - float(light_temp_c)) > temp_tol_f:
                                    continue
                            cand.append(dp)

                        if use_temp_match and (light_temp_c is None or not np.isfinite(light_temp_c)):
                            warnings.append("dark_match_use_temp=true but light CCD temp not found; matching by exposure only")
                        if use_temp_match and cand_missing_temp > 0:
                            warnings.append(f"dark_match_use_temp=true: skipped {cand_missing_temp} darks with missing temp header")

                        if cand:
                            selected_dark_files = cand
                        else:
                            warnings.append("dark_auto_select: no matching darks found within tolerance; using all darks")
                            selected_dark_files = dark_files
                    except Exception:
                        warnings.append("dark_auto_select: selection failed; using all darks")
                        selected_dark_files = dark_files

            phase_progress(
                run_id,
                log_fp,
                0,
                "SCAN_INPUT",
                0,
                max(1, len(dark_files)),
                {
                    "substep": "dark_master",
                    "dark_files_total": len(dark_files),
                    "dark_files_selected": len(selected_dark_files) if selected_dark_files is not None else 0,
                    "light_exposure_s": light_exp_s,
                    "light_ccd_temp_c": light_temp_c,
                    "dark_auto_select": dark_auto_select,
                    "dark_match_exposure_tolerance_percent": tol_pct_f,
                    "dark_match_use_temp": use_temp_match,
                    "dark_match_temp_tolerance_c": temp_tol_f,
                    "warnings": warnings,
                },
            )

            dm = build_master_mean(selected_dark_files)
            dark_master = bias_correct_dark(dm, bias_master)
            if dark_master is not None:
                fits.writeto(str(out_cal_dir / "master_dark.fit"), dark_master[0], header=dark_master[1], overwrite=True)

        if use_flat and flat_master is None:
            flat_files = _collect_calib_files(str(calibration_cfg.get("flats_dir") or "").strip() or None, cal_pattern)
            phase_progress(run_id, log_fp, 0, "SCAN_INPUT", 0, max(1, len(flat_files)), {"substep": "flat_master"})
            fm = build_master_mean(flat_files)
            flat_master = prepare_flat(fm, bias_master, dark_master)
            if flat_master is not None:
                fits.writeto(str(out_cal_dir / "master_flat.fit"), flat_master[0], header=flat_master[1], overwrite=True)

        if (use_bias and bias_master is None) or (use_dark and dark_master is None) or (use_flat and flat_master is None):
            phase_end(
                run_id,
                log_fp,
                0,
                "SCAN_INPUT",
                "error",
                {
                    "error": "calibration requested but master frame could not be resolved",
                    "use_bias": use_bias,
                    "use_dark": use_dark,
                    "use_flat": use_flat,
                },
            )
            return False

        out_lights = outputs_dir / "calibrated"
        out_lights.mkdir(parents=True, exist_ok=True)
        new_frames: list[Path] = []
        total = len(frames)
        for i, p in enumerate(frames):
            phase_progress(run_id, log_fp, 0, "SCAN_INPUT", i + 1, total, {"substep": "calibrate_lights"})
            if stop_requested(run_id, log_fp, 0, "SCAN_INPUT", stop_flag):
                return False
            try:
                img, hdr = read_fits_float(p)
                cal = apply_calibration(
                    img,
                    bias_master[0] if (use_bias and bias_master is not None) else None,
                    dark_master[0] if (use_dark and dark_master is not None) else None,
                    flat_master[0] if (use_flat and flat_master is not None) else None,
                    denom_eps=1e-6,
                )
                outp = out_lights / f"cal_{i+1:05d}.fit"
                fits.writeto(str(outp), cal.astype("float32", copy=False), header=hdr, overwrite=True)
                new_frames.append(outp)
            except Exception as e:
                phase_end(
                    run_id,
                    log_fp,
                    0,
                    "SCAN_INPUT",
                    "error",
                    {"error": "failed to calibrate frame", "frame": str(p), "details": str(e)},
                )
                return False

        if new_frames:
            frames = new_frames

    # Optional linearity validation (Methodik v3 §2.1 / §4)
    linearity_cfg = cfg.get("linearity") if isinstance(cfg.get("linearity"), dict) else {}
    linearity_enabled = bool(linearity_cfg.get("enabled", False))
    if linearity_enabled and "validate_frames_linearity" in globals() and validate_frames_linearity is not None and frames:
        try:
            # Sample up to N frames for linearity check to limit cost
            max_lin_frames = int(linearity_cfg.get("max_frames", 8))
            if max_lin_frames <= 0:
                max_lin_frames = 8
            sample_paths = frames[: min(len(frames), max_lin_frames)]
            sample_arrays: list[np.ndarray] = []
            for p in sample_paths:
                try:
                    img, _hdr = read_fits_float(p)
                    sample_arrays.append(img)
                except Exception:
                    continue
            if sample_arrays:
                arr = np.stack(sample_arrays, axis=0)
                lin_result = validate_frames_linearity(arr, linearity_cfg)
                overall = float(lin_result.get("overall_linearity", 1.0))
                min_overall = float(linearity_cfg.get("min_overall_linearity", 0.9))
                if overall < min_overall:
                    phase_end(
                        run_id,
                        log_fp,
                        0,
                        "SCAN_INPUT",
                        "error",
                        {
                            "error": "linearity validation failed",
                            "overall_linearity": overall,
                            "min_overall_linearity": min_overall,
                            "diagnostics": lin_result.get("diagnostics"),
                        },
                    )
                    return False
        except Exception:
            # In case of unexpected validator failure, continue without hard abort
            pass

    if not should_skip_phase(0):
        cfa_flag0 = fits_is_cfa(frames[0]) if frames else None
        header_bayerpat0 = fits_get_bayerpat(frames[0]) if frames else None
        phase_end(
            run_id,
            log_fp,
            0,
            "SCAN_INPUT",
            "ok",
            {
                "frame_count": len(frames),
                "color_mode": color_mode,
                "bayer_pattern": bayer_pattern,
                "bayer_pattern_header": header_bayerpat0,
                "cfa": cfa_flag0,
                "calibrated": calibrated_applied,
            },
        )

    phase_id = 1
    phase_name = "REGISTRATION"
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
    elif reg_engine == "opencv_cfa":
        phase_start(run_id, log_fp, phase_id, phase_name)
        if cv2 is None:
            phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "opencv (cv2) not available; install python opencv to use registration.engine=opencv_cfa"})
            return False

        allow_rotation = bool(registration_cfg.get("allow_rotation"))
        min_star_matches = registration_cfg.get("min_star_matches")
        try:
            min_star_matches_i = int(min_star_matches) if min_star_matches is not None else 1
        except Exception:
            min_star_matches_i = 1

        reg_out = outputs_dir / reg_out_name
        reg_out.mkdir(parents=True, exist_ok=True)

        # Reference frame selection strategy:
        # 1. Prefer frames in the middle third of the sequence (minimizes max drift distance)
        # 2. Within that range, select the frame with the most detected stars (best quality)
        # This prevents selecting a reference at the sequence edges which causes
        # large drift distances and poor ECC convergence for distant frames.
        n_frames = len(frames)
        middle_start = n_frames // 3
        middle_end = 2 * n_frames // 3
        
        # First pass: collect star counts for all frames
        frame_star_counts: List[tuple[int, int, np.ndarray, Any, Path]] = []  # (idx, stars, lum01, hdr, path)
        for i, p in enumerate(frames):
            try:
                data = fits.getdata(str(p), ext=0)
                if data is None:
                    continue
                lum = cfa_downsample_sum2x2(np.asarray(data))
                lum01 = opencv_prepare_ecc_image(lum)
                stars = opencv_count_stars(lum01)
                hdr = fits.getheader(str(p), ext=0)
                frame_star_counts.append((i, stars, lum01, hdr, p))
            except Exception:
                continue
        
        # Select reference: best frame in middle third, fallback to best overall
        ref_idx = 0
        ref_stars = -1
        ref_lum01: Optional[np.ndarray] = None
        ref_hdr = None
        ref_path = None
        
        # Try middle third first
        middle_candidates = [(i, s, l, h, p) for i, s, l, h, p in frame_star_counts if middle_start <= i < middle_end]
        if middle_candidates:
            best = max(middle_candidates, key=lambda x: x[1])
            ref_idx, ref_stars, ref_lum01, ref_hdr, ref_path = best
        
        # Fallback to best overall if middle third has no valid frames
        if ref_lum01 is None and frame_star_counts:
            best = max(frame_star_counts, key=lambda x: x[1])
            ref_idx, ref_stars, ref_lum01, ref_hdr, ref_path = best

        if ref_lum01 is None or ref_stars < max(1, min_star_matches_i):
            phase_end(
                run_id,
                log_fp,
                phase_id,
                phase_name,
                "error",
                {"error": "failed to select registration reference (insufficient stars)", "min_star_matches": min_star_matches_i, "best_star_count": ref_stars},
            )
            return False

        # Log reference selection info
        ref_in_middle = middle_start <= ref_idx < middle_end
        print(f"[INFO] Reference frame selected: index={ref_idx}, stars={ref_stars}, in_middle_third={ref_in_middle}, middle_range=[{middle_start}, {middle_end})")

        # Chain registration strategy:
        # Instead of registering all frames to a single reference (which fails with field rotation),
        # we register each frame to its neighbor and accumulate transformations.
        # This minimizes drift per registration step and handles field rotation better.
        #
        # Process order: Start from reference, go outward in both directions
        # ref-1 -> ref, ref-2 -> ref-1, ... (backward chain)
        # ref+1 -> ref, ref+2 -> ref+1, ... (forward chain)
        
        # First, load all frames and prepare luminance images
        frame_data: List[tuple[Path, np.ndarray, np.ndarray, Any]] = []  # (path, mosaic_clean, lum01, header)
        for i, p in enumerate(frames):
            try:
                data = fits.getdata(str(p), ext=0)
                if data is None:
                    continue
                hdr = fits.getheader(str(p), ext=0)
                mosaic = np.asarray(data)
                mosaic_clean = cosmetic_correction(mosaic, sigma_threshold=8.0, hot_only=True)
                lum = cfa_downsample_sum2x2(mosaic_clean)
                lum01 = opencv_prepare_ecc_image(lum)
                stars = opencv_count_stars(lum01)
                if stars < max(1, min_star_matches_i):
                    print(f"[WARN] Frame {i}: insufficient stars ({stars} < {min_star_matches_i}), continuing")
                frame_data.append((p, mosaic_clean, lum01, hdr))
            except Exception as e:
                print(f"[WARN] Frame {i}: failed to load: {e}")
                continue
        
        if len(frame_data) < 2:
            phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "insufficient valid frames for registration", "valid_frames": len(frame_data)})
            return False
        
        # Find reference index in filtered frame_data
        ref_idx_filtered = -1
        for i, (p, _, _, _) in enumerate(frame_data):
            if p == ref_path:
                ref_idx_filtered = i
                break
        if ref_idx_filtered < 0:
            ref_idx_filtered = len(frame_data) // 2
        
        print(f"[INFO] Chain registration: {len(frame_data)} valid frames, reference at filtered index {ref_idx_filtered}")
        
        # Compute accumulated warps using chain registration
        # warp[i] transforms frame i to reference frame coordinate system
        identity_warp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        accumulated_warps: List[np.ndarray] = [identity_warp.copy() for _ in frame_data]
        corrs: List[float] = [0.0] * len(frame_data)
        corrs[ref_idx_filtered] = 1.0
        
        # Backward chain: ref-1, ref-2, ...
        for i in range(ref_idx_filtered - 1, -1, -1):
            prev_lum01 = frame_data[i + 1][2]
            curr_lum01 = frame_data[i][2]
            init = opencv_best_translation_init(curr_lum01, prev_lum01)
            try:
                step_warp, cc = opencv_ecc_warp(curr_lum01, prev_lum01, allow_rotation=allow_rotation, init_warp=init)
            except Exception as ex:
                step_warp, cc = init, 0.0
                print(f"[DEBUG] Frame {i}: ECC failed: {ex}")
            if not np.isfinite(cc) or cc < 0.15:
                step_warp, cc = init, float(cc if np.isfinite(cc) else 0.0)
            # Accumulate: warp[i] = step_warp composed with warp[i+1]
            # For 2x3 affine: compose by matrix multiplication
            prev_warp = accumulated_warps[i + 1]
            # Convert to 3x3 for composition
            step_3x3 = np.vstack([step_warp, [0, 0, 1]])
            prev_3x3 = np.vstack([prev_warp, [0, 0, 1]])
            composed_3x3 = step_3x3 @ prev_3x3
            accumulated_warps[i] = composed_3x3[:2, :].astype(np.float32)
            corrs[i] = float(cc)
            print(f"[DEBUG] Chain backward {i}: step_cc={cc:.4f}")
        
        # Forward chain: ref+1, ref+2, ...
        for i in range(ref_idx_filtered + 1, len(frame_data)):
            prev_lum01 = frame_data[i - 1][2]
            curr_lum01 = frame_data[i][2]
            init = opencv_best_translation_init(curr_lum01, prev_lum01)
            try:
                step_warp, cc = opencv_ecc_warp(curr_lum01, prev_lum01, allow_rotation=allow_rotation, init_warp=init)
            except Exception as ex:
                step_warp, cc = init, 0.0
                print(f"[DEBUG] Frame {i}: ECC failed: {ex}")
            if not np.isfinite(cc) or cc < 0.15:
                step_warp, cc = init, float(cc if np.isfinite(cc) else 0.0)
            # Accumulate: warp[i] = step_warp composed with warp[i-1]
            prev_warp = accumulated_warps[i - 1]
            step_3x3 = np.vstack([step_warp, [0, 0, 1]])
            prev_3x3 = np.vstack([prev_warp, [0, 0, 1]])
            composed_3x3 = step_3x3 @ prev_3x3
            accumulated_warps[i] = composed_3x3[:2, :].astype(np.float32)
            corrs[i] = float(cc)
            print(f"[DEBUG] Chain forward {i}: step_cc={cc:.4f}")
        
        # Apply accumulated warps and save registered frames
        registered_count = 0
        for i, (p, mosaic, lum01, src_hdr) in enumerate(frame_data):
            warp = accumulated_warps[i]
            if i == ref_idx_filtered:
                warped = mosaic.astype("float32", copy=False)
                print(f"[DEBUG] Frame {i} is REFERENCE")
            else:
                warped = warp_cfa_mosaic_via_subplanes(mosaic, warp)
            
            try:
                dst_name = reg_pattern.format(index=registered_count + 1)
            except Exception:
                dst_name = f"reg_{registered_count + 1:05d}.fit"
            dst_path = reg_out / dst_name
            fits.writeto(str(dst_path), warped.astype("float32", copy=False), header=src_hdr, overwrite=True)
            registered_count += 1

        extra: Dict[str, Any] = {
            "engine": "opencv_cfa",
            "output_dir": str(reg_out),
            "registered_count": registered_count,
            "reference_index": ref_idx,
            "min_star_matches": min_star_matches_i,
            "allow_rotation": allow_rotation,
        }
        if corrs:
            extra["ecc_corr_min"] = float(min(corrs))
            extra["ecc_corr_mean"] = float(sum(corrs) / len(corrs))
        phase_end(run_id, log_fp, phase_id, phase_name, "ok", extra)

    else:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if reg_engine != "siril":
            phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"registration.engine not supported: {reg_engine!r}"})
            return False
        if not siril_exe:
            phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "siril executable not found"})
            return False


        cfa_flag = fits_is_cfa(frames[0]) if frames else None
        header_bayerpat = fits_get_bayerpat(frames[0]) if frames else None
        warn = None
        if header_bayerpat is not None and header_bayerpat != bayer_pattern:
            warn = "bayer_pattern mismatch (config vs header)"

        using_default_reg_script = not (isinstance(reg_script_cfg, str) and reg_script_cfg.strip())
        if using_default_reg_script and (color_mode != "OSC" or cfa_flag is not True):
            phase_end(
                run_id,
                log_fp,
                phase_id,
                phase_name,
                "error",
                {
                    "error": "default registration script requires OSC/CFA; set registration.siril_script for non-OSC inputs",
                    "color_mode": color_mode,
                    "cfa": cfa_flag,
                },
            )
            return False

        if not reg_script_path.exists() or not reg_script_path.is_file():
            phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"registration script not found: {reg_script_path}"})
            return False

        ok_script, violations = validate_siril_script(reg_script_path)
        if not ok_script:
            phase_end(
                run_id,
                log_fp,
                phase_id,
                phase_name,
                "error",
                {"error": "registration script violates policy", "script": str(reg_script_path), "violations": violations},
            )
            return False

        reg_work = work_dir / "registration"
        reg_work.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(frames, start=1):
            dst = reg_work / f"seq{i:05d}.fit"
            safe_symlink_or_copy(src, dst)

        ok, meta = run_siril_script(
            siril_exe=siril_exe,
            work_dir=reg_work,
            script_path=reg_script_path,
            artifacts_dir=artifacts_dir,
            log_name="siril_registration.log",
            quiet=True,
        )
        if not ok:
            reason = _extract_siril_error_reason(meta.get("log_file") if isinstance(meta, dict) else None)
            phase_end(
                run_id,
                log_fp,
                phase_id,
                phase_name,
                "error",
                {
                    "error": reason or "siril registration failed",
                    "siril": meta,
                    "bayer_pattern": bayer_pattern,
                    "bayer_pattern_header": header_bayerpat,
                    "cfa": cfa_flag,
                    "warning": warn,
                },
            )
            return False

        reg_out = outputs_dir / reg_out_name
        reg_out.mkdir(parents=True, exist_ok=True)
        registered = sorted([p for p in reg_work.iterdir() if p.is_file() and p.name.lower().startswith("r_") and is_fits_image_path(p)])
        if not registered:
            phase_end(
                run_id,
                log_fp,
                phase_id,
                phase_name,
                "error",
                {"error": "no registered frames produced by Siril (expected r_*.fit*)", "siril": meta},
            )
            return False

        moved = 0
        for idx, src in enumerate(registered, start=1):
            try:
                dst_name = reg_pattern.format(index=idx)
            except Exception:
                dst_name = src.name
            safe_hardlink_or_copy(src, reg_out / dst_name)
            moved += 1
        phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "ok",
            {
                "siril": meta,
                "output_dir": str(reg_out),
                "registered_count": moved,
                "bayer_pattern": bayer_pattern,
                "bayer_pattern_header": header_bayerpat,
                "cfa": cfa_flag,
                "warning": warn,
            },
        )

    reg_out_dir = outputs_dir / reg_out_name
    registered_files = sorted([p for p in reg_out_dir.iterdir() if p.is_file() and is_fits_image_path(p)]) if reg_out_dir.exists() else []
    if not registered_files:
        phase_end(run_id, log_fp, 1, "REGISTRATION", "error", {"error": "no registered frames found"})
        return False

    try:
        _write_registration_artifacts(artifacts_dir, registered_files, corrs if isinstance(corrs, list) else None)
    except Exception:
        pass

    try:
        reg_eval: dict[str, Any] = {
            "registration_absdiff_samples.png": {
                "phase": "REGISTRATION",
                "evaluations": [f"registered_frames={len(registered_files)}"],
            }
        }
        if isinstance(corrs, list) and corrs:
            c = [float(x) for x in corrs if isinstance(x, (int, float, np.number))]
            reg_eval["registration_ecc_corr_timeseries.png"] = {
                "phase": "REGISTRATION",
                "evaluations": _eval_timeseries("ecc_corr", c),
                "data": {"ecc_corr": c},
            }
            reg_eval["registration_ecc_corr_hist.png"] = {
                "phase": "REGISTRATION",
                "evaluations": _eval_timeseries("ecc_corr", c),
                "data": {"ecc_corr": c},
            }
        _update_report_metrics(artifacts_dir, {"artifacts": reg_eval})
    except Exception:
        pass

    phase_id = 2
    phase_name = "CHANNEL_SPLIT"
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
    else:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
            return False

    frames_target = d.get("frames_target")
    try:
        frames_target_i = int(frames_target) if frames_target is not None else 0
    except Exception:
        frames_target_i = 0
    analysis_count = len(registered_files) if frames_target_i <= 0 else min(len(registered_files), frames_target_i)

    # Write channel splits to disk to avoid OOM with large datasets
    channels_dir = work_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    channel_files: Dict[str, List[Path]] = {"R": [], "G": [], "B": []}
    
    cfa_registered = None
    total_split = max(1, analysis_count)
    for idx, p in enumerate(registered_files[:analysis_count], start=1):
        data, _hdr = read_fits_float(p)
        is_cfa = (fits_is_cfa(p) is True)
        if cfa_registered is None:
            cfa_registered = is_cfa
        if is_cfa:
            # Prefer BAYERPAT from FITS header if available, fallback to config
            bp_to_use = fits_get_bayerpat(p) or bayer_pattern
            split = split_cfa_channels(data, bp_to_use)
        else:
            try:
                split = split_rgb_frame(data)
            except Exception:
                if data.ndim != 2:
                    phase_end(
                        run_id,
                        log_fp,
                        phase_id,
                        phase_name,
                        "error",
                        {"error": "unsupported registered frame layout for channel split", "frame": str(p), "shape": list(data.shape)},
                    )
                    return False
                split = {"R": data, "G": data, "B": data}
        
        # Write each channel to disk
        for ch in ("R", "G", "B"):
            ch_file = channels_dir / f"{ch}_{idx:05d}.fits"
            fits.writeto(str(ch_file), split[ch], overwrite=True)
            channel_files[ch].append(ch_file)
        
        # Free memory immediately
        del data, split

        if idx % 5 == 0 or idx == total_split:
            phase_progress(run_id, log_fp, phase_id, phase_name, idx, total_split, {})

    phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {
            "registered_dir": str(reg_out_dir),
            "registered_count": len(registered_files),
            "analysis_count": analysis_count,
            "cfa": bool(cfa_registered),
            "channels_dir": str(channels_dir),
        },
    )

    phase_id = 3
    phase_name = "NORMALIZATION"
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
    else:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
            return False

    norm_cfg = cfg.get("normalization") if isinstance(cfg.get("normalization"), dict) else {}
    norm_mode = str(norm_cfg.get("mode") or "background")
    per_channel = bool(norm_cfg.get("per_channel", True))
    norm_target: Optional[float] = None
    
    # Methodik v3 §3.1: B_f (background) must be computed BEFORE normalization
    # Store B_f values for use in GLOBAL_METRICS phase
    pre_norm_backgrounds: Dict[str, List[float]] = {"R": [], "G": [], "B": []}
    
    # Memory-efficient normalization: load frames from disk, compute medians, normalize, write back
    if per_channel:
        phase_progress(run_id, log_fp, phase_id, phase_name, 0, 3, {"step": "per_channel"})
        
        for ch_idx, ch in enumerate(("R", "G", "B"), start=1):
            # Pass 1: Compute medians (B_f) BEFORE normalization (load one frame at a time)
            medians = []
            for ch_file in channel_files[ch]:
                frame = fits.getdata(str(ch_file)).astype("float32", copy=False)
                medians.append(float(np.median(frame)))
                del frame
            
            # Store B_f values for GLOBAL_METRICS phase
            pre_norm_backgrounds[ch] = medians.copy()
            
            target = float(np.median(np.asarray(medians, dtype=np.float32))) if medians else 0.0
            
            # Pass 2: Normalize and write back (load one frame at a time)
            for ch_file, med in zip(channel_files[ch], medians):
                frame = fits.getdata(str(ch_file)).astype("float32", copy=False)
                normalized = normalize_frame(frame, med, target, norm_mode)
                fits.writeto(str(ch_file), normalized, overwrite=True)
                del frame, normalized
            
            phase_progress(run_id, log_fp, phase_id, phase_name, ch_idx, 3, {"channel": ch, "step": "per_channel"})
    else:
        # Global normalization across all channels
        # Pass 1: Compute all medians (B_f) BEFORE normalization
        all_medians = []
        for ch in ("R", "G", "B"):
            ch_medians = []
            for ch_file in channel_files[ch]:
                frame = fits.getdata(str(ch_file)).astype("float32", copy=False)
                med = float(np.median(frame))
                all_medians.append(med)
                ch_medians.append(med)
                del frame
            # Store B_f values for GLOBAL_METRICS phase
            pre_norm_backgrounds[ch] = ch_medians
        
        norm_target = float(np.median(np.asarray(all_medians, dtype=np.float32))) if all_medians else None
        
        # Pass 2: Normalize each channel
        phase_progress(run_id, log_fp, phase_id, phase_name, 0, 3, {"step": "global_target"})
        med_idx = 0
        for ch_idx, ch in enumerate(("R", "G", "B"), start=1):
            for ch_file in channel_files[ch]:
                frame = fits.getdata(str(ch_file)).astype("float32", copy=False)
                med = all_medians[med_idx]
                med_idx += 1
                
                if str(norm_mode).strip().lower() == "median":
                    scale = (norm_target / med) if (norm_target is not None and med not in (0.0, -0.0)) else 1.0
                    normalized = (frame * float(scale)).astype("float32", copy=False)
                else:
                    normalized = (frame - (med - float(norm_target or med))).astype("float32", copy=False)
                
                fits.writeto(str(ch_file), normalized, overwrite=True)
                del frame, normalized
            
            phase_progress(run_id, log_fp, phase_id, phase_name, ch_idx, 3, {"channel": ch, "step": "global_target"})

    phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {"mode": norm_mode, "per_channel": per_channel, "target_median": norm_target},
    )

    try:
        _write_normalization_artifacts(artifacts_dir, pre_norm_backgrounds)
    except Exception:
        pass

    try:
        _update_report_metrics(
            artifacts_dir,
            {
                "artifacts": {
                    "normalization_background_timeseries.png": {
                        "phase": "NORMALIZATION",
                        "evaluations": {
                            ch: _eval_timeseries(f"{ch} pre-norm background B_f", pre_norm_backgrounds.get(ch) or [])
                            for ch in ("R", "G", "B")
                        },
                        "data": {"pre_norm_backgrounds": pre_norm_backgrounds},
                    }
                }
            },
        )
    except Exception:
        pass

    phase_id = 4
    phase_name = "GLOBAL_METRICS"
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
    else:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
            return False

    weights_cfg = cfg.get("global_metrics") if isinstance(cfg.get("global_metrics"), dict) else {}
    w = weights_cfg.get("weights") if isinstance(weights_cfg.get("weights"), dict) else {}
    
    # Methodik v3 §3.2: Check if adaptive weights are enabled
    use_adaptive_weights = weights_cfg.get("adaptive_weights", False)
    
    # Methodik v3 §3.2: Default α=0.4, β=0.3, γ=0.3
    try:
        w_bg = float(w.get("background", 0.4))  # α
    except Exception:
        w_bg = 0.4
    try:
        w_noise = float(w.get("noise", 0.3))  # β
    except Exception:
        w_noise = 0.3
    try:
        w_grad = float(w.get("gradient", 0.3))  # γ
    except Exception:
        w_grad = 0.3
    
    # Methodik v3 §3.2: Validate α + β + γ = 1 (Test Case 1)
    weight_sum = w_bg + w_noise + w_grad
    if abs(weight_sum - 1.0) > 1e-6:
        phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {"error": f"global_metrics weights must sum to 1.0 (α + β + γ = 1), got {weight_sum:.6f}", "weights": {"background": w_bg, "noise": w_noise, "gradient": w_grad}},
        )
        return False

    channel_metrics: Dict[str, Dict[str, Any]] = {}
    total_global = sum(len(channel_files[ch]) for ch in ("R", "G", "B"))
    total_global = max(1, total_global)
    processed_global = 0
    for ch in ("R", "G", "B"):
        # Methodik v3 §3.1: Use B_f computed BEFORE normalization (stored in pre_norm_backgrounds)
        # σ_f and E_f are computed on normalized data
        bgs: List[float] = pre_norm_backgrounds.get(ch, [])
        noises: List[float] = []
        grads: List[float] = []
        for i, ch_file in enumerate(channel_files[ch], start=1):
            f = fits.getdata(str(ch_file)).astype("float32", copy=False)
            # B_f already stored from pre-normalization phase
            noises.append(float(np.std(f)))
            grads.append(float(np.mean(np.hypot(*np.gradient(f.astype("float32", copy=False))))))
            del f
            processed_global += 1
            if processed_global % 5 == 0 or processed_global == total_global:
                phase_progress(
                    run_id,
                    log_fp,
                    phase_id,
                    phase_name,
                    processed_global,
                    total_global,
                    {"channel": ch, "frame": i},
                )
        
        # Fallback if pre_norm_backgrounds not available (e.g., skipped normalization)
        if not bgs or len(bgs) != len(channel_files[ch]):
            bgs = []
            for ch_file in channel_files[ch]:
                f = fits.getdata(str(ch_file)).astype("float32", copy=False)
                bgs.append(float(np.median(f)))
                del f

        def _norm_mad(vals: List[float]) -> List[float]:
            """
            Normalize using MAD (Median Absolute Deviation) - Methodik v3 §A.5
            
            Formula: x̃ = (x - median(x)) / (1.4826 · MAD(x))
            
            More robust against outliers than min/max normalization.
            The factor 1.4826 makes MAD consistent with standard deviation for normal distributions.
            """
            if not vals:
                return []
            a = np.asarray(vals, dtype=np.float32)
            
            # Compute median
            med = float(np.median(a))
            
            # Compute MAD (Median Absolute Deviation)
            mad = float(np.median(np.abs(a - med)))
            
            # Avoid division by zero
            if not np.isfinite(mad) or mad < 1e-12:
                return [0.0 for _ in vals]
            
            # Normalize: x̃ = (x - median) / (1.4826 · MAD)
            # Factor 1.4826 ≈ 1/Φ⁻¹(3/4) makes MAD consistent with σ for normal distributions
            normalized = (a - med) / (1.4826 * mad)
            
            return [float(x) for x in normalized.tolist()]

        # Normalize metrics using MAD (Methodik v3 §A.5)
        bg_n = _norm_mad(bgs)
        noise_n = _norm_mad(noises)
        grad_n = _norm_mad(grads)
        
        # Methodik v3 §3.2: Adaptive Gewichtung 
        # Dynamische Anpassung basierend auf Varianz der Metriken
        if use_adaptive_weights and bgs and noises and grads:
            bg_var = float(np.var(bgs)) if bgs else 0.0
            noise_var = float(np.var(noises)) if noises else 0.0
            grad_var = float(np.var(grads)) if grads else 0.0
            total_var = bg_var + noise_var + grad_var
            
            if total_var > 1e-12:
                # Gewichte basierend auf Varianz
                w_bg_adaptive = bg_var / total_var
                w_noise_adaptive = noise_var / total_var
                w_grad_adaptive = grad_var / total_var
                
                # Constraints: min 0.1, max 0.7
                w_bg_adaptive = float(np.clip(w_bg_adaptive, 0.1, 0.7))
                w_noise_adaptive = float(np.clip(w_noise_adaptive, 0.1, 0.7))
                w_grad_adaptive = float(np.clip(w_grad_adaptive, 0.1, 0.7))
                
                # Renormalisiere auf Summe = 1
                adaptive_sum = w_bg_adaptive + w_noise_adaptive + w_grad_adaptive
                w_bg = w_bg_adaptive / adaptive_sum
                w_noise = w_noise_adaptive / adaptive_sum
                w_grad = w_grad_adaptive / adaptive_sum
        
        # Compute quality scores Q_f,c (Methodik v3 §3.2)
        # Formula: Q_f,c = α*(-B̃) + β*(-σ̃) + γ*Ẽ
        # Lower background and noise are better (negative), higher gradient is better (positive)
        q_f = [float(w_bg * (-b) + w_noise * (-n) + w_grad * g) for b, n, g in zip(bg_n, noise_n, grad_n)]
        
        # Clamp Q_f to [-3, +3] before exp() (Methodik v3 §5, §14 Test Case 2)
        q_f_clamped = [float(np.clip(q, -3.0, 3.0)) for q in q_f]
        
        # Global weights G_f = exp(Q_f_clamped)
        gfc = [float(np.exp(q)) for q in q_f_clamped]

        channel_metrics[ch] = {
            "global": {
                "background_level": bgs,
                "noise_level": noises,
                "gradient_energy": grads,
                "G_f_c": gfc,
                "weights_used": {"background": w_bg, "noise": w_noise, "gradient": w_grad},
                "adaptive_weights_enabled": use_adaptive_weights,
            }
        }

    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"analysis_count": analysis_count})

    try:
        _write_global_metrics_artifacts(artifacts_dir, channel_metrics)
    except Exception:
        pass

    try:
        eval_by_ch: dict[str, Any] = {}
        for ch in ("R", "G", "B"):
            g = channel_metrics.get(ch, {}).get("global", {}) if isinstance(channel_metrics.get(ch, {}), dict) else {}
            b = g.get("background_level") if isinstance(g.get("background_level"), list) else []
            n = g.get("noise_level") if isinstance(g.get("noise_level"), list) else []
            e = g.get("gradient_energy") if isinstance(g.get("gradient_energy"), list) else []
            gf = g.get("G_f_c") if isinstance(g.get("G_f_c"), list) else []
            eval_by_ch[ch] = {
                "global_weight": _eval_weights([float(x) for x in gf if isinstance(x, (int, float, np.number))]),
                "noise": _eval_timeseries(f"{ch} noise σ_f", [float(x) for x in n if isinstance(x, (int, float, np.number))]),
                "background": _eval_timeseries(f"{ch} background B_f", [float(x) for x in b if isinstance(x, (int, float, np.number))]),
                "gradient": _eval_timeseries(f"{ch} gradient E_f", [float(x) for x in e if isinstance(x, (int, float, np.number))]),
            }

        _update_report_metrics(
            artifacts_dir,
            {
                "artifacts": {
                    "global_weight_timeseries.png": {
                        "phase": "GLOBAL_METRICS",
                        "evaluations": eval_by_ch,
                    },
                    "global_weight_hist.png": {
                        "phase": "GLOBAL_METRICS",
                        "evaluations": eval_by_ch,
                    },
                }
            },
        )
    except Exception:
        pass

    phase_id = 5
    phase_name = "TILE_GRID"
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
    else:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
            return False

    tile_cfg = cfg.get("tile") if isinstance(cfg.get("tile"), dict) else {}
    try:
        min_tile_size = int(tile_cfg.get("min_size") or 64)  # Methodik v3 default
    except Exception:
        min_tile_size = 64
    try:
        max_divisor = int(tile_cfg.get("max_divisor") or 6)  # Methodik v3 default
    except Exception:
        max_divisor = 6
    try:
        overlap = float(tile_cfg.get("overlap_fraction") or 0.25)
    except Exception:
        overlap = 0.25
    try:
        size_factor = float(tile_cfg.get("size_factor") or 32)  # Methodik v3 default s
    except Exception:
        size_factor = 32
    
    # Methodik v3 §3.3: Validate overlap_fraction in [0, 0.5]
    if not 0 <= overlap <= 0.5:
        phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"overlap_fraction must be in [0, 0.5], got {overlap}"})
        return False

    # Load first frame from each channel to determine dimensions
    rep = {}
    for ch in ("R", "G", "B"):
        if channel_files[ch]:
            rep[ch] = fits.getdata(str(channel_files[ch][0])).astype("float32", copy=False)
    if not rep:
        phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "no frames for tile grid"})
        return False
    h0, w0 = next(iter(rep.values())).shape[:2]
    
    # Methodik v3 §3.3: Estimate FWHM from star detection
    # Use robust FWHM estimation across multiple frames
    fwhm_estimates = []
    try:
        if cv2 is not None:
            for ch in ("R", "G", "B"):
                if ch in rep:
                    img_u8 = _to_uint8(rep[ch])
                    # Detect stars using goodFeaturesToTrack
                    corners = cv2.goodFeaturesToTrack(img_u8, maxCorners=50, qualityLevel=0.01, minDistance=20)
                    if corners is not None and len(corners) > 3:
                        # Estimate FWHM from star sizes (simplified: use gradient magnitude around stars)
                        gy, gx = np.gradient(rep[ch].astype("float32", copy=False))
                        grad_mag = np.sqrt(gx**2 + gy**2)
                        for corner in corners[:20]:
                            x, y = int(corner[0][0]), int(corner[0][1])
                            if 10 < x < w0 - 10 and 10 < y < h0 - 10:
                                patch = grad_mag[y-5:y+5, x-5:x+5]
                                if patch.size > 0:
                                    # FWHM approximation from gradient spread
                                    fwhm_est = float(np.sum(patch > np.max(patch) * 0.5)) ** 0.5 * 2.0
                                    if 1.0 < fwhm_est < 50.0:
                                        fwhm_estimates.append(fwhm_est)
    except Exception:
        pass
    
    # Use median FWHM or fallback to default
    if fwhm_estimates:
        fwhm = float(np.median(fwhm_estimates))
    else:
        fwhm = 3.0  # Methodik v3 §3.3: Default FWHM if not measurable
    
    # Methodik v3 §3.3 Grenzwertprüfungen:
    # 1. F > 0: Falls FWHM nicht messbar, verwende Default F = 3.0
    if fwhm <= 0:
        fwhm = 3.0
    
    # 2. T_min >= 16: Absolute Untergrenze für Tile-Größe
    min_tile_size = max(16, min_tile_size)
    
    # Methodik v3 §3.3: T_0 = s · F, T = floor(clip(T_0, T_min, floor(min(W, H) / D)))
    T_0 = size_factor * fwhm
    T_max = int(min(h0, w0) // max(1, max_divisor))
    
    # 5. min(W, H) >= T: Falls Bild kleiner als Tile
    if min(h0, w0) < min_tile_size:
        tile_size = min(h0, w0)
        overlap_px = 0
    else:
        tile_size = int(np.floor(np.clip(T_0, min_tile_size, T_max)))
        # 3. T >= T_min: Falls T < T_min nach Berechnung
        tile_size = max(tile_size, min_tile_size)
        
        # Methodik v3 §3.3: O = floor(o · T), S = T - O
        overlap_px = int(np.floor(overlap * tile_size))
        
        # 4. S > 0: Falls S <= 0 (bei extremem Overlap)
        step_size_check = tile_size - overlap_px
        if step_size_check <= 0:
            overlap = 0.25  # Reset to safe default
            overlap_px = int(np.floor(overlap * tile_size))
    
    step_size = tile_size - overlap_px
    
    grid_cfg = {
        "min_tile_size": min_tile_size, 
        "max_tile_size": T_max, 
        "overlap": overlap,
        "fwhm": fwhm,
        "size_factor": size_factor,
        "tile_size": tile_size,
        "overlap_px": overlap_px,
        "step_size": step_size,
    }

    if generate_multi_channel_grid is None:
        phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "tile_grid backend not available"})
        return False
    
    # Use V3 mode with computed FWHM
    v3_grid_cfg = {
        "fwhm": fwhm,
        "size_factor": size_factor,
        "min_size": min_tile_size,
        "max_divisor": max_divisor,
        "overlap_fraction": overlap,
    }
    tile_grids = generate_multi_channel_grid({k: _to_uint8(v) for k, v in rep.items()}, v3_grid_cfg)
    try:
        _write_tile_grid_pngs(artifacts_dir, rep, tile_grids, grid_cfg)
    except Exception:
        pass
    del rep

    try:
        _update_report_metrics(
            artifacts_dir,
            {
                "artifacts": {
                    "tile_grid.json": {
                        "phase": "TILE_GRID",
                        "data": {"grid_cfg": grid_cfg, "v3_grid_cfg": v3_grid_cfg},
                        "evaluations": {
                            "grid_cfg": [json.dumps(grid_cfg, ensure_ascii=False)],
                        },
                    }
                }
            },
        )
    except Exception:
        pass

    try:
        overlay_eval = [f"tile_size={grid_cfg.get('tile_size')}", f"overlap_px={grid_cfg.get('overlap_px')}", f"step_size={grid_cfg.get('step_size')}" ]
        _update_report_metrics(
            artifacts_dir,
            {
                "artifacts": {
                    **{
                        f"tile_grid_overlay_{ch}.png": {
                            "phase": "TILE_GRID",
                            "evaluations": overlay_eval,
                        }
                        for ch in ("R", "G", "B")
                    }
                }
            },
        )
    except Exception:
        pass
    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"grid_cfg": grid_cfg, "fwhm_estimated": fwhm})

    phase_id = 6
    phase_name = "LOCAL_METRICS"
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
    else:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
            return False

    if TileMetricsCalculator is None:
        phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "metrics backend not available"})
        return False

    try:
        tile_size_i = int(tile_grids.get("G", {}).get("tile_size") or tile_grids.get("R", {}).get("tile_size") or min_tile_size)
    except Exception:
        tile_size_i = min_tile_size
    try:
        overlap_i = float(tile_grids.get("G", {}).get("overlap") or overlap)
    except Exception:
        overlap_i = overlap
    tile_calc = TileMetricsCalculator(tile_size=tile_size_i, overlap=overlap_i)

    lm_work = work_dir / "local_metrics"
    lm_work.mkdir(parents=True, exist_ok=True)

    lm_cfg = cfg.get("local_metrics") if isinstance(cfg.get("local_metrics"), dict) else {}
    star_mode = lm_cfg.get("star_mode") if isinstance(lm_cfg.get("star_mode"), dict) else {}
    star_w = star_mode.get("weights") if isinstance(star_mode.get("weights"), dict) else {}
    # Methodik v3 §3.4: Default-Gewichte für Stern-Modus
    try:
        w_fwhm = float(star_w.get("fwhm", 0.6))
    except Exception:
        w_fwhm = 0.6
    try:
        w_round = float(star_w.get("roundness", 0.2))
    except Exception:
        w_round = 0.2
    try:
        w_con = float(star_w.get("contrast", 0.2))
    except Exception:
        w_con = 0.2

    # Struktur-Modus (Methodik v3 §3.4): ENR vs. Hintergrundgewicht
    structure_mode = lm_cfg.get("structure_mode") if isinstance(lm_cfg.get("structure_mode"), dict) else {}
    try:
        w_struct = float(structure_mode.get("metric_weight", 0.7))
    except Exception:
        w_struct = 0.7
    try:
        w_bg_local = float(structure_mode.get("background_weight", 0.3))
    except Exception:
        w_bg_local = 0.3

    total_frames = sum(len(channel_files[ch]) for ch in ("R", "G", "B"))
    processed_frames = 0
    
    for ch in ("R", "G", "B"):
        ch_files = channel_files[ch]
        if not ch_files:
            continue

        ch_l_dir = lm_work / f"L_local_{ch}"
        ch_l_dir.mkdir(parents=True, exist_ok=True)

        # Pass 1: Sammle rohe Tile-Metriken über alle Frames
        per_frame_metrics: List[Dict[str, np.ndarray]] = []
        n_tiles: Optional[int] = None
        for f_idx, ch_file in enumerate(ch_files):
            f = fits.getdata(str(ch_file)).astype("float32", copy=False)
            tm = tile_calc.calculate_tile_metrics(f)
            fwhm = np.asarray(tm.get("fwhm") or [], dtype=np.float32)
            rnd = np.asarray(tm.get("roundness") or [], dtype=np.float32)
            con = np.asarray(tm.get("contrast") or [], dtype=np.float32)
            bg_local = np.asarray(tm.get("background_level") or [], dtype=np.float32)
            noise_local = np.asarray(tm.get("noise_level") or [], dtype=np.float32)
            grad_local = np.asarray(tm.get("gradient_energy") or [], dtype=np.float32)

            if n_tiles is None:
                n_tiles = int(fwhm.size)
            else:
                # Falls ein Frame eine andere Tile-Anzahl hätte, brechen wir mit Fehler ab
                if int(fwhm.size) != n_tiles:
                    phase_end(
                        run_id,
                        log_fp,
                        phase_id,
                        phase_name,
                        "error",
                        {"error": "inconsistent tile count across frames in LOCAL_METRICS", "channel": ch},
                    )
                    return False

            per_frame_metrics.append(
                {
                    "fwhm": fwhm,
                    "roundness": rnd,
                    "contrast": con,
                    "bg_local": bg_local,
                    "noise_local": noise_local,
                    "grad_local": grad_local,
                }
            )
            del f

            processed_frames += 1
            if processed_frames % 5 == 0 or processed_frames == total_frames:
                phase_progress(run_id, log_fp, phase_id, phase_name, processed_frames, total_frames, {"channel": ch, "step": "collect"})

        if n_tiles is None or n_tiles <= 0 or not per_frame_metrics:
            continue

        n_frames_ch = len(per_frame_metrics)

        # Baue Arrays der Form (F, T)
        fwhm_all = np.zeros((n_frames_ch, n_tiles), dtype=np.float32)
        rnd_all = np.zeros_like(fwhm_all)
        con_all = np.zeros_like(fwhm_all)
        bg_all = np.zeros_like(fwhm_all)
        noise_all = np.zeros_like(fwhm_all)
        grad_all = np.zeros_like(fwhm_all)

        for i, mets in enumerate(per_frame_metrics):
            fwhm_all[i, :] = mets["fwhm"]
            rnd_all[i, :] = mets["roundness"]
            con_all[i, :] = mets["contrast"]
            bg_all[i, :] = mets["bg_local"]
            noise_all[i, :] = mets["noise_local"]
            grad_all[i, :] = mets["grad_local"]

        # Heuristik: Stern-Tiles vs. Struktur-Tiles basierend auf Kontrast
        # (einfach, aber genügt für Separate Behandlung gemäß Spec)
        # Wir klassifizieren pro (Frame, Tile).
        contrast_med = float(np.median(con_all)) if np.isfinite(con_all).any() else 0.0
        contrast_thr = contrast_med * 1.5 if contrast_med > 0 else 0.0
        star_mask = (con_all > contrast_thr)
        struct_mask = ~star_mask

        def _mad_norm_subset(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
            """MAD-Normalisierung nur über die Elemente mit mask=True.

            Werte außerhalb der Maske werden auf 0 gesetzt.
            """
            if not mask.any():
                return np.zeros_like(values, dtype=np.float32)
            subset = values[mask].astype(np.float32, copy=False)
            med = float(np.median(subset))
            mad = float(np.median(np.abs(subset - med)))
            if mad < 1e-12 or not np.isfinite(mad):
                out = np.zeros_like(values, dtype=np.float32)
                return out
            normed = (values.astype(np.float32, copy=False) - med) / (1.4826 * mad)
            # Setze Bereiche außerhalb der Maske explizit auf 0
            normed[~mask] = 0.0
            return normed.astype(np.float32, copy=False)

        # Stern-Modus: FWHM, Rundheit, Kontrast (nur auf star_mask)
        fwhm_n = _mad_norm_subset(fwhm_all, star_mask)
        rnd_n = _mad_norm_subset(rnd_all, star_mask)
        con_n = _mad_norm_subset(con_all, star_mask)

        # Struktur-Modus: ENR = E / σ, lokaler Hintergrund (nur auf struct_mask)
        noise_safe = np.where(noise_all <= 0, 1e-6, noise_all.astype(np.float32, copy=False))
        enr_all = grad_all.astype(np.float32, copy=False) / noise_safe
        enr_n = _mad_norm_subset(enr_all, struct_mask)
        bg_n = _mad_norm_subset(bg_all, struct_mask)

        # Q_local gemäß Spec (Stern- bzw. Struktur-Formel)
        q_local = np.zeros_like(fwhm_all, dtype=np.float32)
        # Stern-Tiles
        q_local[star_mask] = (
            w_fwhm * (-fwhm_n[star_mask])
            + w_round * rnd_n[star_mask]
            + w_con * con_n[star_mask]
        ).astype(np.float32, copy=False)
        # Struktur-Tiles
        q_local[struct_mask] = (
            w_struct * enr_n[struct_mask]
            - w_bg_local * bg_n[struct_mask]
        ).astype(np.float32, copy=False)

        # Clamping und lokale Gewichte
        q_local = np.clip(q_local, -3.0, 3.0).astype(np.float32, copy=False)
        l_local = np.exp(q_local).astype(np.float32, copy=False)

        # Pro Frame: Mittelwert und Varianz von Q_local (für Clusterung / Analyse)
        q_mean = [float(np.mean(q_local[i, :])) for i in range(n_frames_ch)]
        q_var = [float(np.var(q_local[i, :])) for i in range(n_frames_ch)]

        # Pro Tile: Mittelwert/Varianz über Frames (für Heatmaps)
        q_tile_mean = [float(x) for x in np.mean(q_local, axis=0).astype(np.float32, copy=False).tolist()]
        q_tile_var = [float(x) for x in np.var(q_local, axis=0).astype(np.float32, copy=False).tolist()]
        l_tile_mean = [float(x) for x in np.mean(l_local, axis=0).astype(np.float32, copy=False).tolist()]
        l_tile_var = [float(x) for x in np.var(l_local, axis=0).astype(np.float32, copy=False).tolist()]

        # Schreibe L_local pro Frame auf Disk und (optional) speichere Q_local in channel_metrics
        for f_idx in range(n_frames_ch):
            try:
                np.save(str(ch_l_dir / f"L_{f_idx+1:05d}.npy"), l_local[f_idx, :].astype(np.float32, copy=False))
            except Exception:
                pass

        channel_metrics[ch]["tiles"] = {
            "tile_quality_mean": q_mean,
            "tile_quality_variance": q_var,
            "L_local_files_dir": str(ch_l_dir),
            "Q_local_tile_mean": q_tile_mean,
            "Q_local_tile_var": q_tile_var,
            "L_local_tile_mean": l_tile_mean,
            "L_local_tile_var": l_tile_var,
            # Vollständige Q_local-Matrix (Frames×Tiles) für tile-basierte Rekonstruktion
            "Q_local": q_local.tolist(),
        }

    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"tile_size": tile_size_i, "overlap": overlap_i})

    try:
        _write_tile_quality_heatmaps(artifacts_dir, channel_metrics, grid_cfg, (h0, w0))
    except Exception:
        pass

    try:
        lm_art: dict[str, Any] = {}
        tile_size_hm = int(grid_cfg.get("tile_size") or 0)
        step_size_hm = int(grid_cfg.get("step_size") or 0)
        if tile_size_hm > 0 and step_size_hm > 0:
            n_tiles_y = max(1, (h0 - tile_size_hm) // step_size_hm + 1)
            n_tiles_x = max(1, (w0 - tile_size_hm) // step_size_hm + 1)
            n_tiles = n_tiles_y * n_tiles_x
            for ch in ("R", "G", "B"):
                tiles = channel_metrics.get(ch, {}).get("tiles", {})
                q_mean_tile = tiles.get("Q_local_tile_mean")
                q_var_tile = tiles.get("Q_local_tile_var")
                l_mean_tile = tiles.get("L_local_tile_mean")
                l_var_tile = tiles.get("L_local_tile_var")

                if not (isinstance(q_mean_tile, list) and len(q_mean_tile) == n_tiles):
                    continue
                if not (isinstance(q_var_tile, list) and len(q_var_tile) == n_tiles):
                    continue

                ev_mean_q = _basic_stats([float(x) for x in q_mean_tile])
                ev_var_q = _basic_stats([float(x) for x in q_var_tile])
                lm_art[f"tile_quality_heatmap_{ch}.png"] = {
                    "phase": "LOCAL_METRICS",
                    "evaluations": [
                        f"tiles={n_tiles} (x={n_tiles_x}, y={n_tiles_y})",
                        f"mean(Q_local): median={float(ev_mean_q.get('median') or 0.0):.3g}, min={float(ev_mean_q.get('min') or 0.0):.3g}, max={float(ev_mean_q.get('max') or 0.0):.3g}",
                    ],
                }
                lm_art[f"tile_quality_var_heatmap_{ch}.png"] = {
                    "phase": "LOCAL_METRICS",
                    "evaluations": [
                        f"tiles={n_tiles} (x={n_tiles_x}, y={n_tiles_y})",
                        f"var(Q_local): median={float(ev_var_q.get('median') or 0.0):.3g}, max={float(ev_var_q.get('max') or 0.0):.3g}",
                    ],
                }

                if isinstance(l_mean_tile, list) and len(l_mean_tile) == n_tiles and isinstance(l_var_tile, list) and len(l_var_tile) == n_tiles:
                    ev_mean_l = _basic_stats([float(x) for x in l_mean_tile])
                    ev_var_l = _basic_stats([float(x) for x in l_var_tile])
                    lm_art[f"tile_weight_heatmap_{ch}.png"] = {
                        "phase": "LOCAL_METRICS",
                        "evaluations": [
                            f"mean(L_local): median={float(ev_mean_l.get('median') or 0.0):.3g}, min={float(ev_mean_l.get('min') or 0.0):.3g}, max={float(ev_mean_l.get('max') or 0.0):.3g}",
                        ],
                    }
                    lm_art[f"tile_weight_var_heatmap_{ch}.png"] = {
                        "phase": "LOCAL_METRICS",
                        "evaluations": [
                            f"var(L_local): median={float(ev_var_l.get('median') or 0.0):.3g}, max={float(ev_var_l.get('max') or 0.0):.3g}",
                        ],
                    }
        if lm_art:
            _update_report_metrics(artifacts_dir, {"artifacts": lm_art})
    except Exception:
        pass

    phase_id = 7
    phase_name = "TILE_RECONSTRUCTION"
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
    else:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
            return False

    reconstructed: Dict[str, np.ndarray] = {}
    hdr0 = None
    try:
        hdr0 = fits.getheader(str(registered_files[0]), ext=0)
    except Exception:
        hdr0 = None
    
    # Epsilon for numerical stability (Methodik v3 §3.6)
    epsilon = 1e-6
    
    # Get tile grid parameters for tile-based reconstruction
    tile_size_recon = grid_cfg.get("tile_size", tile_size)
    overlap_px_recon = grid_cfg.get("overlap_px", int(overlap * tile_size_recon))
    step_recon = tile_size_recon - overlap_px_recon
    
    channels_to_process = [ch for ch in ("R", "G", "B") if channel_files[ch]]
    for ch_idx, ch in enumerate(channels_to_process, start=1):
        gfc = np.asarray(channel_metrics[ch]["global"].get("G_f_c") or [], dtype=np.float32)
        tiles = channel_metrics[ch].get("tiles", {})
        l_local = tiles.get("L_local", [])
        l_local_files_dir_s = tiles.get("L_local_files_dir")
        l_local_files_dir = Path(str(l_local_files_dir_s)).resolve() if l_local_files_dir_s else None
        
        num_frames = len(channel_files[ch])
        
        if num_frames == 0:
            reconstructed[ch] = np.zeros((1, 1), dtype=np.float32)
            out_path = outputs_dir / f"reconstructed_{ch}.fits"
            fits.writeto(str(out_path), reconstructed[ch], header=hdr0, overwrite=True)
            phase_progress(run_id, log_fp, phase_id, phase_name, ch_idx, len(channels_to_process), {"channel": ch})
            continue
        
        # Load first frame to get dimensions
        first_frame = fits.getdata(str(channel_files[ch][0])).astype("float32", copy=False)
        h0, w0 = first_frame.shape[:2]
        del first_frame
        
        # Methodik v3 §3.5-3.6: Tile-based reconstruction with W_f,t,c = G_f,c · L_f,t,c
        # I_t,c(p) = Σ_f W_f,t,c · I_f,c(p) / Σ_f W_f,t,c
        
        # Check if we have tile-level weights
        has_tile_weights = (l_local and len(l_local) == num_frames and all(isinstance(ll, list) and len(ll) > 0 for ll in l_local))
        if not has_tile_weights and l_local_files_dir is not None and l_local_files_dir.exists():
            has_tile_weights = True
        
        if has_tile_weights and gfc.size == num_frames:
            # Full tile-based reconstruction (Methodik v3 §3.6)
            # Compute tile grid
            n_tiles_y = max(1, (h0 - tile_size_recon) // step_recon + 1)
            n_tiles_x = max(1, (w0 - tile_size_recon) // step_recon + 1)
            n_tiles = n_tiles_y * n_tiles_x
            
            # Initialize output and weight accumulator
            out = np.zeros((h0, w0), dtype=np.float32)
            weight_sum = np.zeros((h0, w0), dtype=np.float32)
            
            # Per-Tile-Gesamtgewicht D_t,c (Methodik v3 §3.6)
            D_tiles = np.zeros((n_tiles,), dtype=np.float64)
            
            # Hanning window for overlap-add (Methodik v3 §3.6)
            hann_1d = np.hanning(tile_size_recon).astype(np.float32)
            hann_2d = np.outer(hann_1d, hann_1d)
            
            # Process each frame
            for f_idx, ch_file in enumerate(channel_files[ch]):
                f = fits.getdata(str(ch_file)).astype("float32", copy=False)
                g_f = float(gfc[f_idx]) if f_idx < gfc.size else 1.0
                if l_local and f_idx < len(l_local):
                    l_f = l_local[f_idx]
                elif l_local_files_dir is not None:
                    try:
                        l_arr = np.load(str(l_local_files_dir / f"L_{f_idx+1:05d}.npy"))
                        l_f = [float(x) for x in np.asarray(l_arr, dtype=np.float32).tolist()]
                    except Exception:
                        l_f = []
                else:
                    l_f = []
                
                # Process each tile
                t_idx = 0
                for ty in range(n_tiles_y):
                    for tx in range(n_tiles_x):
                        y0 = ty * step_recon
                        x0 = tx * step_recon
                        y1 = min(y0 + tile_size_recon, h0)
                        x1 = min(x0 + tile_size_recon, w0)
                        
                        # Get tile weight L_f,t,c
                        l_f_t = float(l_f[t_idx]) if t_idx < len(l_f) else 1.0
                        
                        # Methodik v3 §3.5: W_f,t,c = G_f,c · L_f,t,c
                        w_f_t = g_f * l_f_t
                        
                        # Akkumuliere per-Tile-Gewicht D_t,c
                        if np.isfinite(w_f_t) and w_f_t > 0.0:
                            D_tiles[t_idx] += float(w_f_t)
                        
                        # Extract tile
                        tile = f[y0:y1, x0:x1].copy()
                        tile_h, tile_w = tile.shape
                        
                        # Methodik v3 §3.6: Tile-Normalisierung NACH Hintergrundsubtraktion
                        tile_bg = float(np.median(tile))
                        tile = tile - tile_bg  # Subtract background
                        tile_median = float(np.median(np.abs(tile)))
                        if tile_median > 1e-10:
                            tile = tile / tile_median  # Normalize
                        
                        # Apply window function
                        window = hann_2d[:tile_h, :tile_w]
                        
                        # Accumulate weighted tile
                        out[y0:y1, x0:x1] += tile * w_f_t * window
                        weight_sum[y0:y1, x0:x1] += w_f_t * window
                        
                        t_idx += 1
                
                del f
            
            # Normalize by weight sum (Methodik v3 §3.6)
            # Fallback für Tiles mit sehr geringem Gesamtgewicht D_t,c
            low_tile_mask = D_tiles < float(epsilon)
            if np.any(low_tile_mask):
                # Baue Pixelmaske der Low-Weight-Tiles
                low_weight_mask = np.zeros((h0, w0), dtype=bool)
                t_idx = 0
                for ty in range(n_tiles_y):
                    for tx in range(n_tiles_x):
                        if low_tile_mask[t_idx]:
                            y0 = ty * step_recon
                            x0 = tx * step_recon
                            y1 = min(y0 + tile_size_recon, h0)
                            x1 = min(x0 + tile_size_recon, w0)
                            low_weight_mask[y0:y1, x0:x1] = True
                        t_idx += 1

                # Compute unweighted mean for fallback (über alle Frames, wie in Spec)
                fallback_sum = np.zeros((h0, w0), dtype=np.float32)
                fallback_count = np.zeros((h0, w0), dtype=np.float32)
                for ch_file in channel_files[ch]:
                    f = fits.getdata(str(ch_file)).astype("float32", copy=False)
                    fallback_sum += f
                    fallback_count += 1.0
                    del f
                fallback_mean = fallback_sum / np.maximum(fallback_count, 1.0)
                
                # Apply fallback in Low-Weight-Tiles
                weight_sum_safe = np.where(low_weight_mask, 1.0, weight_sum)
                out = np.where(low_weight_mask, fallback_mean, out / weight_sum_safe)
            else:
                out = out / weight_sum
            
            reconstructed[ch] = out.astype(np.float32, copy=False)

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                im = ax.imshow(np.log1p(weight_sum.astype("float32", copy=False)), cmap="inferno", interpolation="nearest")
                ax.set_title(f"log(1+weight_sum) ({ch})")
                ax.set_axis_off()
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                outp = artifacts_dir / f"reconstruction_weight_sum_{ch}.png"
                fig.savefig(str(outp), dpi=200)
                plt.close(fig)

                try:
                    ws = weight_sum.astype("float32", copy=False)
                    ws1 = ws[np.isfinite(ws)]
                    ws_stats = _basic_stats([float(x) for x in ws1.flatten().tolist()])
                    low_frac = float(np.mean(low_weight_mask.astype(np.float32))) if low_weight_mask is not None else 0.0
                    _update_report_metrics(
                        artifacts_dir,
                        {
                            "artifacts": {
                                f"reconstruction_weight_sum_{ch}.png": {
                                    "phase": "TILE_RECONSTRUCTION",
                                    "evaluations": [
                                        f"low_weight_fraction={low_frac:.3g}",
                                        f"weight_sum: median={float(ws_stats.get('median') or 0.0):.3g}, max={float(ws_stats.get('max') or 0.0):.3g}",
                                    ],
                                }
                            }
                        },
                    )
                except Exception:
                    pass

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.imshow(_to_uint8(reconstructed[ch]), cmap="gray", interpolation="nearest")
                ax.set_title(f"reconstructed preview ({ch})")
                ax.set_axis_off()
                fig.tight_layout()
                outp = artifacts_dir / f"reconstruction_preview_{ch}.png"
                fig.savefig(str(outp), dpi=200)
                plt.close(fig)

                try:
                    rstats = _basic_stats([float(x) for x in reconstructed[ch].astype("float32", copy=False).flatten().tolist()])
                    _update_report_metrics(
                        artifacts_dir,
                        {
                            "artifacts": {
                                f"reconstruction_preview_{ch}.png": {
                                    "phase": "TILE_RECONSTRUCTION",
                                    "evaluations": [
                                        f"reconstructed: median={float(rstats.get('median') or 0.0):.3g}, std={float(rstats.get('std') or 0.0):.3g}",
                                    ],
                                }
                            }
                        },
                    )
                except Exception:
                    pass

                n_vis = min(25, num_frames)
                if n_vis > 0:
                    mean_img = None
                    for i0 in range(n_vis):
                        ff = fits.getdata(str(channel_files[ch][i0])).astype("float32", copy=False)
                        if mean_img is None:
                            mean_img = np.zeros_like(ff, dtype=np.float32)
                        mean_img += ff
                        del ff
                    mean_img = mean_img / float(n_vis)
                    diff = np.abs(reconstructed[ch].astype("float32", copy=False) - mean_img.astype("float32", copy=False))
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    ax.imshow(_to_uint8(diff), cmap="magma", interpolation="nearest")
                    ax.set_title(f"abs(recon - mean(first {n_vis})) ({ch})")
                    ax.set_axis_off()
                    fig.tight_layout()
                    outp = artifacts_dir / f"reconstruction_absdiff_vs_mean_{ch}.png"
                    fig.savefig(str(outp), dpi=200)
                    plt.close(fig)

                    try:
                        dstats = _basic_stats([float(x) for x in diff.astype("float32", copy=False).flatten().tolist()])
                        _update_report_metrics(
                            artifacts_dir,
                            {
                                "artifacts": {
                                    f"reconstruction_absdiff_vs_mean_{ch}.png": {
                                        "phase": "TILE_RECONSTRUCTION",
                                        "evaluations": [
                                            f"absdiff: median={float(dstats.get('median') or 0.0):.3g}, max={float(dstats.get('max') or 0.0):.3g}",
                                            f"n_vis={n_vis}",
                                        ],
                                    }
                                }
                            },
                        )
                    except Exception:
                        pass
            except Exception:
                try:
                    plt.close("all")
                except Exception:
                    pass
        else:
            # Fallback: Global weight only (no tile weights available)
            if gfc.size == num_frames:
                wsum = float(np.sum(gfc))
                if wsum > epsilon:
                    w_norm = (gfc / wsum).astype(np.float32, copy=False)
                    out = None
                    for f_idx, ch_file in enumerate(channel_files[ch]):
                        f = fits.getdata(str(ch_file)).astype("float32", copy=False)
                        if out is None:
                            out = np.zeros_like(f, dtype=np.float32)
                        out += f * float(w_norm[f_idx])
                        del f
                    reconstructed[ch] = out
                else:
                    # Unweighted mean
                    out = None
                    count = 0
                    for ch_file in channel_files[ch]:
                        f = fits.getdata(str(ch_file)).astype("float32", copy=False)
                        if out is None:
                            out = np.zeros_like(f, dtype=np.float32)
                        out += f
                        count += 1
                        del f
                    reconstructed[ch] = (out / count).astype(np.float32, copy=False) if count > 0 else out
            else:
                # No weights: unweighted mean
                out = None
                count = 0
                for ch_file in channel_files[ch]:
                    f = fits.getdata(str(ch_file)).astype("float32", copy=False)
                    if out is None:
                        out = np.zeros_like(f, dtype=np.float32)
                    out += f
                    count += 1
                    del f
                reconstructed[ch] = (out / count).astype(np.float32, copy=False) if count > 0 else out

        out_path = outputs_dir / f"reconstructed_{ch}.fits"
        fits.writeto(str(out_path), reconstructed[ch].astype("float32", copy=False), header=hdr0, overwrite=True)
        
        phase_progress(run_id, log_fp, phase_id, phase_name, ch_idx, len(channels_to_process), {"channel": ch})

    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"outputs": [f"reconstructed_{c}.fits" for c in ("R", "G", "B")], "tile_based": has_tile_weights if channels_to_process else False})

    phase_id = 8
    phase_name = "STATE_CLUSTERING"
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
    else:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
            return False

    # Reduced mode check (Methodik v3 §1.4)
    assumptions_cfg = get_assumptions_config(cfg)
    frame_count = len(registered_files)
    reduced_mode = is_reduced_mode(frame_count, assumptions_cfg)
    
    clustering_cfg = synthetic_cfg.get("clustering") if isinstance(synthetic_cfg.get("clustering"), dict) else {}
    clustering_results = None
    clustering_skipped = False
    clustering_fallback_used = False
    phase_progress(run_id, log_fp, phase_id, phase_name, 0, 1, {"step": "start"})
    
    if reduced_mode and assumptions_cfg["reduced_mode_skip_clustering"]:
        # Skip clustering in reduced mode
        clustering_skipped = True
    elif cluster_channels is not None:
        try:
            # Adjust cluster range for reduced mode
            if reduced_mode:
                reduced_range = assumptions_cfg["reduced_mode_cluster_range"]
                clustering_cfg = dict(clustering_cfg)
                clustering_cfg["cluster_count_range"] = reduced_range
            
            num_frames = len(channel_files.get("R", []))
            total_ch = len([ch for ch in ("R", "G", "B") if channel_files.get(ch)])
            total_steps = max(2, total_ch + 1)
            processed_steps = 0

            phase_progress(
                run_id,
                log_fp,
                phase_id,
                phase_name,
                processed_steps,
                total_steps,
                {"step": "cluster", "frames_total": num_frames},
            )

            # Provide lightweight placeholders to avoid loading FITS into RAM.
            channels_for_clustering: Dict[str, List[np.ndarray]] = {}
            for ch in ("R", "G", "B"):
                n = len(channel_files.get(ch) or [])
                if n <= 0:
                    continue
                placeholder = np.zeros((1, 1), dtype=np.float32)
                channels_for_clustering[ch] = [placeholder] * n
                processed_steps += 1
                phase_progress(
                    run_id,
                    log_fp,
                    phase_id,
                    phase_name,
                    processed_steps,
                    total_steps,
                    {"step": "cluster", "channel": ch, "frames_total": num_frames},
                )
                if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
                    return False

            clustering_results = cluster_channels(channels_for_clustering, channel_metrics, clustering_cfg)

            processed_steps = min(total_steps, processed_steps + 1)
            phase_progress(run_id, log_fp, phase_id, phase_name, processed_steps, total_steps, {"step": "cluster_done"})
        except Exception as e:
            # Fallback: Quantile-based clustering (Methodik v3 §3.7)
            # Group frames by quantiles of global quality G_f
            try:
                # Methodik v3 §3.7: Dynamische Cluster-Anzahl K = clip(floor(N/10), K_min, K_max)
                num_frames = len(channel_files.get("R", []))
                K_min = 5   # Minimum für stabile Statistik
                K_max = 30  # Maximum für Effizienz
                n_quantiles = int(np.clip(num_frames // 10, K_min, K_max))
                
                if reduced_mode:
                    n_quantiles = min(n_quantiles, assumptions_cfg["reduced_mode_cluster_range"][1])
                
                clustering_results = {}
                total_ch = 3
                done_ch = 0
                phase_progress(run_id, log_fp, phase_id, phase_name, 0, total_ch, {"step": "quantile_fallback"})
                for ch in ("R", "G", "B"):
                    if ch not in channel_files or not channel_files[ch]:
                        done_ch += 1
                        phase_progress(run_id, log_fp, phase_id, phase_name, done_ch, total_ch, {"step": "quantile_fallback", "channel": ch, "skipped": True})
                        continue
                    gfc = channel_metrics.get(ch, {}).get("global", {}).get("G_f_c", [])
                    if not gfc or len(gfc) != len(channel_files[ch]):
                        done_ch += 1
                        phase_progress(run_id, log_fp, phase_id, phase_name, done_ch, total_ch, {"step": "quantile_fallback", "channel": ch, "skipped": True})
                        continue
                    
                    # Compute quantile boundaries
                    gfc_arr = np.asarray(gfc, dtype=np.float32)
                    quantiles = np.linspace(0, 100, n_quantiles + 1)
                    boundaries = np.percentile(gfc_arr, quantiles)
                    
                    # Assign frames to quantile bins
                    cluster_labels = np.digitize(gfc_arr, boundaries[1:-1])
                    
                    clustering_results[ch] = {
                        "cluster_labels": cluster_labels.tolist(),
                        "n_clusters": n_quantiles,
                        "method": "quantile_fallback",
                        "quantile_boundaries": boundaries.tolist(),
                    }

                    done_ch += 1
                    phase_progress(run_id, log_fp, phase_id, phase_name, done_ch, total_ch, {"step": "quantile_fallback", "channel": ch, "n_clusters": n_quantiles})
                    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
                        return False
                
                clustering_fallback_used = True
            except Exception:
                clustering_results = None

    phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {
            "enabled": bool(clustering_results is not None),
            "reduced_mode": reduced_mode,
            "skipped": clustering_skipped,
            "fallback_used": clustering_fallback_used,
        },
    )

    try:
        _write_clustering_artifacts(artifacts_dir, channel_metrics, clustering_results)
    except Exception:
        pass

    try:
        cl_art: dict[str, Any] = {}
        if isinstance(clustering_results, dict):
            for ch in ("R", "G", "B"):
                labels = None
                if isinstance(clustering_results.get(ch), dict):
                    labels = clustering_results.get(ch, {}).get("labels", clustering_results.get(ch, {}).get("cluster_labels"))
                if labels is None:
                    labels = clustering_results.get("labels", clustering_results.get("cluster_labels"))
                if isinstance(labels, list) and labels:
                    lab = np.asarray([int(x) for x in labels], dtype=np.int32)
                    k = int(np.max(lab)) + 1 if lab.size else 0
                    if k > 0:
                        counts = np.bincount(lab, minlength=k)
                        frac_max = float(np.max(counts) / float(np.sum(counts))) if np.sum(counts) > 0 else 0.0
                        cl_art[f"clustering_summary_{ch}.png"] = {
                            "phase": "STATE_CLUSTERING",
                            "evaluations": [
                                f"clusters={k}",
                                f"largest_cluster_fraction={frac_max:.3g}",
                            ],
                        }
        if cl_art:
            _update_report_metrics(artifacts_dir, {"artifacts": cl_art})
    except Exception:
        pass

    phase_id = 9
    phase_name = "SYNTHETIC_FRAMES"
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
    else:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
            return False

    syn_out = outputs_dir / "synthetic"
    syn_out.mkdir(parents=True, exist_ok=True)
    synthetic_channels: Optional[Dict[str, List[np.ndarray]]] = None
    synthetic_count = 0
    synthetic_skipped = False
    
    # Skip synthetic frames in reduced mode if clustering was skipped
    if reduced_mode and clustering_skipped:
        synthetic_skipped = True
    else:
        try:
            hdr_syn = None
            try:
                hdr_syn = fits.getheader(str(registered_files[0]), ext=0)
            except Exception:
                hdr_syn = None

            def _extract_labels_for_channel(ch_name: str) -> list[int] | None:
                if not clustering_results:
                    return None
                if isinstance(clustering_results, dict) and ch_name in clustering_results and isinstance(clustering_results.get(ch_name), dict):
                    cr = clustering_results.get(ch_name) or {}
                    labels = cr.get("labels", cr.get("cluster_labels"))
                    if isinstance(labels, list):
                        return [int(x) for x in labels]
                if isinstance(clustering_results, dict):
                    labels = clustering_results.get("labels", clustering_results.get("cluster_labels"))
                    if isinstance(labels, list):
                        return [int(x) for x in labels]
                return None

            def _synthetic_from_files(ch_name: str) -> list[Path]:
                labels = _extract_labels_for_channel(ch_name)
                files = channel_files.get(ch_name) or []
                if not files:
                    return []

                g = channel_metrics.get(ch_name, {}).get("global", {}) if isinstance(channel_metrics.get(ch_name), dict) else {}
                gfc = g.get("G_f_c")
                weights = np.asarray(gfc, dtype=np.float32) if gfc is not None else np.ones((len(files),), dtype=np.float32)
                if weights.ndim != 1:
                    weights = np.ravel(weights)
                if weights.shape[0] < len(files):
                    weights = np.pad(weights, (0, len(files) - weights.shape[0]), mode="edge")
                weights = weights[: len(files)]

                if labels is None or len(labels) != len(files):
                    frames_min = int((synthetic_cfg or {}).get("frames_min", 15))
                    frames_max = int((synthetic_cfg or {}).get("frames_max", 30))
                    n_synthetic = min(frames_max, max(frames_min, len(files) // 10))
                    n_synthetic = max(n_synthetic, frames_min)
                    sorted_idx = np.argsort(weights)
                    group_size = float(len(files)) / float(max(1, n_synthetic))
                    labels = []
                    for i in range(n_synthetic):
                        start = int(i * group_size)
                        end = int((i + 1) * group_size) if i < n_synthetic - 1 else len(files)
                        grp = sorted_idx[start:end]
                        for idx in grp.tolist():
                            labels.append((i, int(idx)))
                    labels_sorted = [0 for _ in range(len(files))]
                    for lab, idx in labels:
                        labels_sorted[int(idx)] = int(lab)
                    labels = labels_sorted

                cluster_ids = sorted(set(int(x) for x in labels))
                sample = None
                try:
                    sample = fits.getdata(str(files[0])).astype("float32", copy=False)
                    sample_bytes = int(sample.size) * 4
                except Exception:
                    sample_bytes = 0
                finally:
                    if sample is not None:
                        del sample

                keep_all_acc = True
                if sample_bytes > 0 and len(cluster_ids) > 0:
                    est_bytes = int(sample_bytes) * int(len(cluster_ids))
                    keep_all_acc = est_bytes <= 512 * 1024 * 1024

                out_paths: list[Path] = []
                total_n = max(1, len(files))

                if keep_all_acc:
                    sum_map: Dict[int, np.ndarray] = {}
                    wsum_map: Dict[int, float] = {}

                    for idx, fp in enumerate(files):
                        lab = int(labels[idx])
                        w = float(weights[idx])
                        if not np.isfinite(w) or w <= 0.0:
                            continue
                        frame = fits.getdata(str(fp)).astype("float32", copy=False)
                        if lab not in sum_map:
                            sum_map[lab] = np.zeros_like(frame, dtype=np.float32)
                            wsum_map[lab] = 0.0
                        sum_map[lab] += frame * w
                        wsum_map[lab] += w
                        del frame
                        if (idx + 1) % 25 == 0 or (idx + 1) == total_n:
                            phase_progress(run_id, log_fp, phase_id, phase_name, idx + 1, total_n, {"channel": ch_name})

                    for out_idx, lab in enumerate(cluster_ids, start=1):
                        acc = sum_map.get(lab)
                        wsum = float(wsum_map.get(lab) or 0.0)
                        if acc is None or wsum <= 1e-12:
                            continue
                        syn = (acc / wsum).astype("float32", copy=False)
                        outp = syn_out / f"syn{ch_name}_{out_idx:05d}.fits"
                        fits.writeto(str(outp), syn, header=hdr_syn, overwrite=True)
                        out_paths.append(outp)
                        del syn
                    sum_map.clear()
                    wsum_map.clear()
                else:
                    total_clusters = max(1, len(cluster_ids))
                    for out_idx, cluster_id in enumerate(cluster_ids, start=1):
                        acc = None
                        wsum = 0.0
                        for idx, fp in enumerate(files):
                            if int(labels[idx]) != int(cluster_id):
                                continue
                            w = float(weights[idx])
                            if not np.isfinite(w) or w <= 0.0:
                                continue
                            frame = fits.getdata(str(fp)).astype("float32", copy=False)
                            if acc is None:
                                acc = np.zeros_like(frame, dtype=np.float32)
                            acc += frame * w
                            wsum += w
                            del frame
                            if (idx + 1) % 25 == 0 or (idx + 1) == total_n:
                                phase_progress(
                                    run_id,
                                    log_fp,
                                    phase_id,
                                    phase_name,
                                    idx + 1,
                                    total_n,
                                    {"channel": ch_name, "cluster": out_idx, "clusters_total": total_clusters},
                                )

                        if acc is None or wsum <= 1e-12:
                            continue
                        syn = (acc / wsum).astype("float32", copy=False)
                        outp = syn_out / f"syn{ch_name}_{out_idx:05d}.fits"
                        fits.writeto(str(outp), syn, header=hdr_syn, overwrite=True)
                        out_paths.append(outp)
                        del syn, acc

                return out_paths

            syn_r_paths = _synthetic_from_files("R")
            syn_g_paths = _synthetic_from_files("G")
            syn_b_paths = _synthetic_from_files("B")
            synthetic_count = max(len(syn_r_paths), len(syn_g_paths), len(syn_b_paths))

            # Create combined RGB synthetic frames expected by stacking (syn_*.fits).
            # Use the common count across channels.
            common_n = min(len(syn_r_paths), len(syn_g_paths), len(syn_b_paths))
            for i in range(common_n):
                outp_r = syn_r_paths[i]
                outp_g = syn_g_paths[i]
                outp_b = syn_b_paths[i]

                if outp_r.is_file() and outp_g.is_file() and outp_b.is_file():
                    r = fits.getdata(str(outp_r)).astype("float32", copy=False)
                    g = fits.getdata(str(outp_g)).astype("float32", copy=False)
                    b = fits.getdata(str(outp_b)).astype("float32", copy=False)
                    outp_cfa = syn_out / f"syn_{i+1:05d}.fits"
                    # Reassemble subplanes back to CFA mosaic
                    cfa_mosaic = reassemble_cfa_mosaic(r, g, b, bayer_pattern)
                    fits.writeto(str(outp_cfa), cfa_mosaic, header=hdr_syn, overwrite=True)
                    del r, g, b, cfa_mosaic

            synthetic_channels = {"R": [], "G": [], "B": []}
        except Exception:
            synthetic_channels = None

    phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {
            "synthetic_dir": str(syn_out),
            "synthetic_count": synthetic_count,
            "enabled": bool(synthetic_channels),
            "reduced_mode": reduced_mode,
            "skipped": synthetic_skipped,
            "fallback_reason": "reduced_mode" if (reduced_mode and synthetic_skipped) else ("no_synthetic_frames" if synthetic_count == 0 else None),
        },
    )

    phase_id = 10
    phase_name = "STACKING"
    stacked_path: Path | None = None
    stacked_hdr = None
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
        try:
            stacked_path = outputs_dir / Path(stack_output_file)
            if stacked_path.is_file():
                try:
                    stacked_hdr = fits.getheader(str(stacked_path), ext=0)
                except Exception:
                    stacked_hdr = None
            else:
                stacked_path = None
        except Exception:
            stacked_path = None
            stacked_hdr = None
    
    if stacked_path is None:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
            return False

    # Check if we should fallback to reconstructed frames
    # This happens when:
    # 1. Reduced mode with skipped synthetic frames, OR
    # 2. Synthetic frames are disabled/not generated (synthetic_count == 0)
    use_reconstructed_fallback = (reduced_mode and synthetic_skipped) or (synthetic_count == 0)

    if use_reconstructed_fallback:
        stack_src_dir = outputs_dir
        recon_r = outputs_dir / "reconstructed_R.fits"
        recon_g = outputs_dir / "reconstructed_G.fits"
        recon_b = outputs_dir / "reconstructed_B.fits"
        rgb_path = outputs_dir / "reconstructed_rgb.fits"
        if recon_r.is_file() and recon_g.is_file() and recon_b.is_file():
            try:
                hdr_rgb = fits.getheader(str(recon_r), ext=0)
            except Exception:
                hdr_rgb = None
            try:
                r = np.asarray(fits.getdata(str(recon_r), ext=0)).astype("float32", copy=False)
                g = np.asarray(fits.getdata(str(recon_g), ext=0)).astype("float32", copy=False)
                b = np.asarray(fits.getdata(str(recon_b), ext=0)).astype("float32", copy=False)
                rgb = np.stack([r, g, b], axis=0)
                fits.writeto(str(rgb_path), rgb, header=hdr_rgb, overwrite=True)
            except Exception:
                pass
        stack_files = [rgb_path] if rgb_path.is_file() else sorted([p for p in stack_src_dir.glob("reconstructed_*.fits") if p.is_file()])
    else:
        stack_src_dir = outputs_dir / Path(stack_input_dir_name)
        stack_files = (
            sorted([p for p in stack_src_dir.glob(stack_input_pattern) if p.is_file() and is_fits_image_path(p)])
            if stack_src_dir.exists()
            else []
        )
        if stack_files:
            rgb_syn = [p for p in stack_files if re.match(r"^syn_\d{5}\.fits$", p.name)]
            if rgb_syn:
                stack_files = sorted(rgb_syn)

    if (use_reconstructed_fallback is False) and bool(cfa_registered) and stack_src_dir.exists():
        syn_r = sorted([p for p in stack_src_dir.glob("synR_*.fits") if p.is_file() and is_fits_image_path(p)])
        syn_g = sorted([p for p in stack_src_dir.glob("synG_*.fits") if p.is_file() and is_fits_image_path(p)])
        syn_b = sorted([p for p in stack_src_dir.glob("synB_*.fits") if p.is_file() and is_fits_image_path(p)])
        common_n = min(len(syn_r), len(syn_g), len(syn_b))
        if common_n > 0:
            syn_r = syn_r[:common_n]
            syn_g = syn_g[:common_n]
            syn_b = syn_b[:common_n]

            def _stack_file_list(_files: list[Path]) -> tuple[np.ndarray, Dict[str, Any], bool]:
                _frames_list: list[np.ndarray] = []
                for _fp in _files:
                    try:
                        _arr = np.asarray(fits.getdata(str(_fp), ext=0)).astype("float32", copy=False)
                    except Exception:
                        continue
                    _frames_list.append(_arr)
                if not _frames_list:
                    return np.zeros((1, 1), dtype=np.float32), {"error": "failed to load frames"}, False
                _stack_arr = np.stack(_frames_list, axis=0)
                _use_sigma = SigmaClipConfig is not None and sigma_clip_stack_nd is not None and stack_method == "rej"
                _sigma_stats: Dict[str, Any] = {}
                if _use_sigma:
                    _sigma_cfg_dict: Dict[str, Any] = {
                        "sigma_low": float(sigma_clip_cfg.get("sigma_low", 4.0)),
                        "sigma_high": float(sigma_clip_cfg.get("sigma_high", 4.0)),
                        "max_iters": int(sigma_clip_cfg.get("max_iters", 3)),
                        "min_fraction": float(sigma_clip_cfg.get("min_fraction", 0.5)),
                    }
                    try:
                        _clipped_mean, _mask, _stats = sigma_clip_stack_nd(_stack_arr, _sigma_cfg_dict)
                        _final = _clipped_mean.astype("float32", copy=False)
                        _sigma_stats = _stats
                    except Exception as e:  # noqa: BLE001
                        _final = _stack_arr.mean(axis=0).astype("float32", copy=False)
                        _sigma_stats = {"error": str(e)}
                        _use_sigma = False
                else:
                    _final = _stack_arr.mean(axis=0).astype("float32", copy=False)
                return _final, _sigma_stats, _use_sigma

            try:
                hdr_template = fits.getheader(str(syn_r[0]), ext=0)
            except Exception:
                hdr_template = None

            r_final, r_stats, r_sigma = _stack_file_list(syn_r)
            g_final, g_stats, g_sigma = _stack_file_list(syn_g)
            b_final, b_stats, b_sigma = _stack_file_list(syn_b)

            try:
                fits.writeto(str(outputs_dir / "stacked_R.fits"), r_final, header=hdr_template, overwrite=True)
                fits.writeto(str(outputs_dir / "stacked_G.fits"), g_final, header=hdr_template, overwrite=True)
                fits.writeto(str(outputs_dir / "stacked_B.fits"), b_final, header=hdr_template, overwrite=True)
            except Exception:
                pass

            final_data = reassemble_cfa_mosaic(r_final, g_final, b_final, bayer_pattern)
            final_out = outputs_dir / Path(stack_output_file)
            final_out.parent.mkdir(parents=True, exist_ok=True)
            try:
                fits.writeto(str(final_out), final_data, header=hdr_template, overwrite=True)
            except Exception as e:  # noqa: BLE001
                phase_end(
                    run_id,
                    log_fp,
                    phase_id,
                    phase_name,
                    "error",
                    {"error": f"failed to write stacked output: {e}"},
                )
                return False

            extra: Dict[str, Any] = {
                "siril": None,
                "method": stack_method,
                "output": str(final_out),
                "used_reconstructed_fallback": False,
                "fallback_reason": None,
                "sigma_clipping_used": bool(r_sigma or g_sigma or b_sigma),
                "sigma_stats": {"R": r_stats, "G": g_stats, "B": b_stats},
            }
            phase_end(run_id, log_fp, phase_id, phase_name, "ok", extra)

            stacked_path = final_out
            stacked_hdr = hdr_template

    if not stack_files:
        phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {"error": "no stacking input frames found", "input_dir": str(stack_src_dir), "input_pattern": stack_input_pattern},
        )
        return False

    # Pure-Python linear stacking (optional sigma-clipping for artifact removal).
    #
    # Behaviour:
    # - If SigmaClipConfig / sigma_clip_stack_nd are available and
    #   stacking.method == "rej", we apply sigma-clipping along the stack
    #   axis and then take the mean of the surviving samples.
    # - Otherwise we fall back to a simple unweighted mean over all
    #   stacking input frames, which is Methodik-conform linear stacking.
    
    frames_list: list[np.ndarray] = []
    for fp in stack_files:
        try:
            arr = np.asarray(fits.getdata(str(fp), ext=0)).astype("float32", copy=False)
        except Exception:
            continue
        frames_list.append(arr)

    if not frames_list:
        phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {"error": "failed to load stacking input frames", "input_dir": str(stack_src_dir)},
        )
        return False

    stack_arr = np.stack(frames_list, axis=0)

    # Map stacking configuration to sigma-clipping config if available.
    use_sigma = SigmaClipConfig is not None and sigma_clip_stack_nd is not None and stack_method == "rej"
    if use_sigma:
        # Merge user-provided sigma_clip config (if any) with conservative defaults.
        sigma_cfg_dict: Dict[str, Any] = {
            "sigma_low": float(sigma_clip_cfg.get("sigma_low", 4.0)),
            "sigma_high": float(sigma_clip_cfg.get("sigma_high", 4.0)),
            "max_iters": int(sigma_clip_cfg.get("max_iters", 3)),
            "min_fraction": float(sigma_clip_cfg.get("min_fraction", 0.5)),
        }
        try:
            clipped_mean, mask, stats = sigma_clip_stack_nd(stack_arr, sigma_cfg_dict)
            final_data = clipped_mean.astype("float32", copy=False)
            sigma_stats = stats
        except Exception as e:  # noqa: BLE001
            # On any failure, fall back to simple mean stacking.
            final_data = stack_arr.mean(axis=0).astype("float32", copy=False)
            sigma_stats = {"error": str(e)}
            use_sigma = False
    else:
        final_data = stack_arr.mean(axis=0).astype("float32", copy=False)
        sigma_stats = {}

    final_out = outputs_dir / Path(stack_output_file)
    final_out.parent.mkdir(parents=True, exist_ok=True)

    # Use header from the first valid stacking input as template.
    hdr_template = None
    try:
        hdr_template = fits.getheader(str(stack_files[0]), ext=0)
    except Exception:
        hdr_template = None

    try:
        fits.writeto(str(final_out), final_data, header=hdr_template, overwrite=True)
    except Exception as e:  # noqa: BLE001
        phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {"error": f"failed to write stacked output: {e}"},
        )
        return False

    try:
        out_pngs = _write_quality_analysis_pngs(artifacts_dir, channel_metrics)
        if out_pngs:
            phase_progress(run_id, log_fp, phase_id, phase_name, 1, 1, {"quality_pngs": out_pngs})
    except Exception:
        out_pngs = []

    try:
        qa_eval: dict[str, Any] = {}
        eval_by_ch: dict[str, Any] = {}
        for ch in ("R", "G", "B"):
            g = channel_metrics.get(ch, {}).get("global", {}) if isinstance(channel_metrics.get(ch, {}), dict) else {}
            gf = g.get("G_f_c") if isinstance(g.get("G_f_c"), list) else []
            eval_by_ch[ch] = _eval_weights([float(x) for x in gf if isinstance(x, (int, float, np.number))])
        qa_eval["quality_analysis_combined.png"] = {
            "phase": "STACKING",
            "evaluations": eval_by_ch,
        }
        _update_report_metrics(artifacts_dir, {"artifacts": qa_eval})
    except Exception:
        pass

    extra: Dict[str, Any] = {
        "siril": None,
        "method": stack_method,
        "output": str(final_out),
        "used_reconstructed_fallback": use_reconstructed_fallback,
        "fallback_reason": "no_synthetic_frames" if use_reconstructed_fallback else None,
        "sigma_clipping_used": bool(use_sigma),
        "sigma_stats": sigma_stats,
    }
    phase_end(run_id, log_fp, phase_id, phase_name, "ok", extra)

    stacked_path = final_out
    stacked_hdr = hdr_template

    phase_id = 11
    phase_name = "DEBAYER"
    if should_skip_phase(phase_id):
        phase_start(run_id, log_fp, phase_id, phase_name)
        phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
    else:
        phase_start(run_id, log_fp, phase_id, phase_name)
        if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
            return False

        if not debayer_enabled:
            phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "disabled"})
        else:
            if stacked_path is None or not stacked_path.is_file():
                phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "missing stacked output for debayer"})
                return False

            try:
                stacked_data = np.asarray(fits.getdata(str(stacked_path), ext=0)).astype("float32", copy=False)
            except Exception as e:  # noqa: BLE001
                phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"failed to read stacked output: {e}"})
                return False

            rgb_out = outputs_dir / "stacked_rgb.fits"
            rgb = None
            try:
                if stacked_data.ndim == 2:
                    rgb = demosaic_cfa(stacked_data, bayer_pattern)
                elif stacked_data.ndim == 3:
                    if stacked_data.shape[0] == 3:
                        rgb = stacked_data
                    elif stacked_data.shape[2] == 3:
                        rgb = np.transpose(stacked_data, (2, 0, 1)).astype("float32", copy=False)
            except Exception:
                rgb = None

            if rgb is None:
                phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "not_applicable"})
            else:
                try:
                    fits.writeto(str(rgb_out), rgb, header=stacked_hdr, overwrite=True)
                except Exception as e:  # noqa: BLE001
                    phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"failed to write debayer output: {e}"})
                    return False
                phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"output": str(rgb_out)})

    phase_id = 12
    phase_name = "DONE"
    phase_start(run_id, log_fp, phase_id, phase_name)
    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
        return False
    try:
        _ensure_report_metrics_complete(artifacts_dir)
    except Exception:
        pass
    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {})

    return True


