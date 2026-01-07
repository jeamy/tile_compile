"""Main pipeline implementation (Methodik v3)."""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from astropy.io import fits

from .assumptions import get_assumptions_config, is_reduced_mode
from .events import phase_start, phase_end, phase_progress, stop_requested
from .fits_utils import is_fits_image_path, read_fits_float, fits_is_cfa, fits_get_bayerpat
from .image_processing import split_cfa_channels, split_rgb_frame, normalize_frames, warp_cfa_mosaic_via_subplanes, cfa_downsample_sum2x2
from .opencv_registration import opencv_prepare_ecc_image, opencv_count_stars, opencv_ecc_warp, opencv_best_translation_init
from .siril_utils import validate_siril_script, run_siril_script, extract_siril_save_targets
from .utils import safe_symlink_or_copy, pick_output_file

try:
    import cv2
except Exception:
    cv2 = None

try:
    from tile_compile_backend.metrics import TileMetricsCalculator
    from tile_compile_backend.synthetic import generate_channel_synthetic_frames
    from tile_compile_backend.clustering import cluster_channels
    from tile_compile_backend.tile_grid import generate_multi_channel_grid
except Exception:
    TileMetricsCalculator = None
    generate_channel_synthetic_frames = None
    cluster_channels = None
    generate_multi_channel_grid = None


def _to_uint8(img: np.ndarray) -> np.ndarray:
    f = np.asarray(img).astype("float32", copy=False)
    mn = float(np.min(f))
    mx = float(np.max(f))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros(f.shape, dtype=np.uint8)
    x = (f - mn) / (mx - mn)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


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
            (11, "DONE"),
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

    reg_engine = str(registration_cfg.get("engine") or "")
    stack_engine = str(stacking_cfg.get("engine") or "")

    reg_script_cfg = registration_cfg.get("siril_script")
    reg_script_path = (
        Path(str(reg_script_cfg)).expanduser().resolve()
        if isinstance(reg_script_cfg, str) and reg_script_cfg.strip()
        else (project_root / "siril_register_osc.ssf").resolve()
    )
    stack_script_cfg = stacking_cfg.get("siril_script")
    stack_script_path = (
        Path(str(stack_script_cfg)).expanduser().resolve()
        if isinstance(stack_script_cfg, str) and stack_script_cfg.strip()
        else (project_root / "siril_stack_average.ssf").resolve()
    )

    if not (isinstance(reg_script_cfg, str) and reg_script_cfg.strip()):
        if not reg_script_path.exists():
            alt = (project_root / "siril_scripts" / "siril_register_osc.ssf").resolve()
            if alt.exists():
                reg_script_path = alt
    if not (isinstance(stack_script_cfg, str) and stack_script_cfg.strip()):
        if not stack_script_path.exists():
            alt = (project_root / "siril_scripts" / "siril_stack_average.ssf").resolve()
            if alt.exists():
                stack_script_path = alt

    reg_out_name = str(registration_cfg.get("output_dir") or "registered")
    reg_pattern = str(registration_cfg.get("registered_filename_pattern") or "reg_{index:05d}.fit")

    stack_input_dir_name = str(stacking_cfg.get("input_dir") or reg_out_name)
    stack_input_pattern = str(stacking_cfg.get("input_pattern") or "reg_*.fit")

    stack_output_file = str(stacking_cfg.get("output_file") or "stacked.fit")
    stack_method_cfg = str(stacking_cfg.get("method") or "")
    stack_method = stack_method_cfg.strip().lower()

    phase_id = 0
    phase_name = "SCAN_INPUT"
    phase_start(run_id, log_fp, phase_id, phase_name)
    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
        return False
    cfa_flag0 = fits_is_cfa(frames[0]) if frames else None
    header_bayerpat0 = fits_get_bayerpat(frames[0]) if frames else None
    phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {
            "frame_count": len(frames),
            "color_mode": color_mode,
            "bayer_pattern": bayer_pattern,
            "bayer_pattern_header": header_bayerpat0,
            "cfa": cfa_flag0,
        },
    )

    phase_id = 1
    phase_name = "REGISTRATION"
    phase_start(run_id, log_fp, phase_id, phase_name)
    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
        return False

    if reg_engine == "opencv_cfa":
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

        ref_idx = 0
        ref_stars = -1
        ref_lum01: Optional[np.ndarray] = None
        ref_hdr = None
        ref_path = None
        for i, p in enumerate(frames):
            try:
                data = fits.getdata(str(p), ext=0)
                if data is None:
                    continue
                lum = cfa_downsample_sum2x2(np.asarray(data))
                lum01 = opencv_prepare_ecc_image(lum)
                stars = opencv_count_stars(lum01)
                if stars > ref_stars:
                    ref_stars = stars
                    ref_idx = i
                    ref_lum01 = lum01
                    ref_hdr = fits.getheader(str(p), ext=0)
                    ref_path = p
            except Exception:
                continue

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

        corrs: List[float] = []
        registered_count = 0
        for idx, p in enumerate(frames):
            try:
                src_data = fits.getdata(str(p), ext=0)
                if src_data is None:
                    raise RuntimeError("no FITS data")
                src_hdr = fits.getheader(str(p), ext=0)
                mosaic = np.asarray(src_data)
                lum = cfa_downsample_sum2x2(mosaic)
                lum01 = opencv_prepare_ecc_image(lum)
                stars = opencv_count_stars(lum01)
                if stars < max(1, min_star_matches_i):
                    raise RuntimeError(f"insufficient stars: {stars} < {min_star_matches_i}")

                if idx == ref_idx:
                    warped = mosaic.astype("float32", copy=False)
                    cc = 1.0
                else:
                    init = opencv_best_translation_init(lum01, ref_lum01)
                    try:
                        warp, cc = opencv_ecc_warp(lum01, ref_lum01, allow_rotation=allow_rotation, init_warp=init)
                    except Exception:
                        warp, cc = init, 0.0
                    if not np.isfinite(cc) or cc < 0.15:
                        warp, cc = init, float(cc if np.isfinite(cc) else 0.0)
                    warped = warp_cfa_mosaic_via_subplanes(mosaic, warp)

                try:
                    dst_name = reg_pattern.format(index=registered_count + 1)
                except Exception:
                    dst_name = f"reg_{registered_count + 1:05d}.fit"
                dst_path = reg_out / dst_name
                fits.writeto(str(dst_path), warped.astype("float32", copy=False), header=src_hdr, overwrite=True)
                registered_count += 1
                corrs.append(float(cc))
            except Exception as e:
                phase_end(
                    run_id,
                    log_fp,
                    phase_id,
                    phase_name,
                    "error",
                    {
                        "error": "opencv_cfa registration failed",
                        "frame": str(p),
                        "frame_index": idx,
                        "reference_index": ref_idx,
                        "reference_frame": str(ref_path) if ref_path is not None else None,
                        "details": str(e),
                    },
                )
                return False

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
            phase_end(
                run_id,
                log_fp,
                phase_id,
                phase_name,
                "error",
                {
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
            shutil.copy2(src, reg_out / dst_name)
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

    phase_id = 2
    phase_name = "CHANNEL_SPLIT"
    phase_start(run_id, log_fp, phase_id, phase_name)
    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
        return False

    frames_target = d.get("frames_target")
    try:
        frames_target_i = int(frames_target) if frames_target is not None else 0
    except Exception:
        frames_target_i = 0
    analysis_count = len(registered_files) if frames_target_i <= 0 else min(len(registered_files), frames_target_i)

    channels: Dict[str, List[np.ndarray]] = {"R": [], "G": [], "B": []}
    cfa_registered = None
    total_split = max(1, analysis_count)
    for idx, p in enumerate(registered_files[:analysis_count], start=1):
        data, _hdr = read_fits_float(p)
        is_cfa = (fits_is_cfa(p) is True)
        if cfa_registered is None:
            cfa_registered = is_cfa
        if is_cfa:
            split = split_cfa_channels(data, bayer_pattern)
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
        channels["R"].append(split["R"])
        channels["G"].append(split["G"])
        channels["B"].append(split["B"])

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
        },
    )

    phase_id = 3
    phase_name = "NORMALIZATION"
    phase_start(run_id, log_fp, phase_id, phase_name)
    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
        return False

    norm_cfg = cfg.get("normalization") if isinstance(cfg.get("normalization"), dict) else {}
    norm_mode = str(norm_cfg.get("mode") or "background")
    per_channel = bool(norm_cfg.get("per_channel", True))
    norm_target: Optional[float] = None
    if per_channel:
        phase_progress(run_id, log_fp, phase_id, phase_name, 0, 3, {"step": "per_channel"})
        channels["R"], _ = normalize_frames(channels["R"], norm_mode)
        phase_progress(run_id, log_fp, phase_id, phase_name, 1, 3, {"channel": "R", "step": "per_channel"})
        channels["G"], _ = normalize_frames(channels["G"], norm_mode)
        phase_progress(run_id, log_fp, phase_id, phase_name, 2, 3, {"channel": "G", "step": "per_channel"})
        channels["B"], _ = normalize_frames(channels["B"], norm_mode)
        phase_progress(run_id, log_fp, phase_id, phase_name, 3, 3, {"channel": "B", "step": "per_channel"})
    else:
        meds = [float(np.median(f)) for ch in ("R", "G", "B") for f in channels[ch]]
        norm_target = float(np.median(np.asarray(meds, dtype=np.float32))) if meds else None
        out: Dict[str, List[np.ndarray]] = {"R": [], "G": [], "B": []}
        phase_progress(run_id, log_fp, phase_id, phase_name, 0, 3, {"step": "global_target"})
        for ch_idx, ch in enumerate(("R", "G", "B"), start=1):
            for f in channels[ch]:
                med = float(np.median(f))
                if str(norm_mode).strip().lower() == "median":
                    scale = (norm_target / med) if (norm_target is not None and med not in (0.0, -0.0)) else 1.0
                    out[ch].append((f * float(scale)).astype("float32", copy=False))
                else:
                    out[ch].append((f - (med - float(norm_target or med))).astype("float32", copy=False))
            phase_progress(run_id, log_fp, phase_id, phase_name, ch_idx, 3, {"channel": ch, "step": "global_target"})
        channels = out

    phase_end(
        run_id,
        log_fp,
        phase_id,
        phase_name,
        "ok",
        {"mode": norm_mode, "per_channel": per_channel, "target_median": norm_target},
    )

    phase_id = 4
    phase_name = "GLOBAL_METRICS"
    phase_start(run_id, log_fp, phase_id, phase_name)
    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
        return False

    weights_cfg = cfg.get("global_metrics") if isinstance(cfg.get("global_metrics"), dict) else {}
    w = weights_cfg.get("weights") if isinstance(weights_cfg.get("weights"), dict) else {}
    try:
        w_bg = float(w.get("background", 1.0 / 3.0))
    except Exception:
        w_bg = 1.0 / 3.0
    try:
        w_noise = float(w.get("noise", 1.0 / 3.0))
    except Exception:
        w_noise = 1.0 / 3.0
    try:
        w_grad = float(w.get("gradient", 1.0 / 3.0))
    except Exception:
        w_grad = 1.0 / 3.0

    channel_metrics: Dict[str, Dict[str, Any]] = {}
    total_global = sum(len(channels[ch]) for ch in ("R", "G", "B"))
    total_global = max(1, total_global)
    processed_global = 0
    for ch in ("R", "G", "B"):
        frs = channels[ch]
        bgs: List[float] = []
        noises: List[float] = []
        grads: List[float] = []
        for i, f in enumerate(frs, start=1):
            bgs.append(float(np.median(f)))
            noises.append(float(np.std(f)))
            grads.append(float(np.mean(np.hypot(*np.gradient(f.astype("float32", copy=False))))))
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

        def _norm01(vals: List[float]) -> List[float]:
            if not vals:
                return []
            a = np.asarray(vals, dtype=np.float32)
            mn = float(np.min(a))
            mx = float(np.max(a))
            if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                return [0.0 for _ in vals]
            return [float(x) for x in ((a - mn) / (mx - mn)).tolist()]

        bg_n = _norm01(bgs)
        noise_n = _norm01(noises)
        grad_n = _norm01(grads)
        gfc = [float(w_bg * (1.0 - b) + w_noise * (1.0 - n) + w_grad * g) for b, n, g in zip(bg_n, noise_n, grad_n)]

        channel_metrics[ch] = {
            "global": {
                "background_level": bgs,
                "noise_level": noises,
                "gradient_energy": grads,
                "G_f_c": gfc,
            }
        }

    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"analysis_count": analysis_count})

    phase_id = 5
    phase_name = "TILE_GRID"
    phase_start(run_id, log_fp, phase_id, phase_name)
    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
        return False

    tile_cfg = cfg.get("tile") if isinstance(cfg.get("tile"), dict) else {}
    try:
        min_tile_size = int(tile_cfg.get("min_size") or 32)
    except Exception:
        min_tile_size = 32
    try:
        max_divisor = int(tile_cfg.get("max_divisor") or 8)
    except Exception:
        max_divisor = 8
    try:
        overlap = float(tile_cfg.get("overlap_fraction") or 0.25)
    except Exception:
        overlap = 0.25

    rep = {ch: channels[ch][0] for ch in ("R", "G", "B") if channels[ch]}
    if not rep:
        phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "no frames for tile grid"})
        return False
    h0, w0 = next(iter(rep.values())).shape[:2]
    max_tile_size = max(min_tile_size, int(min(h0, w0) // max(1, max_divisor)))
    grid_cfg = {"min_tile_size": min_tile_size, "max_tile_size": max_tile_size, "overlap": overlap}

    if generate_multi_channel_grid is None:
        phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "tile_grid backend not available"})
        return False
    tile_grids = generate_multi_channel_grid({k: _to_uint8(v) for k, v in rep.items()}, grid_cfg)
    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"grid_cfg": grid_cfg})

    phase_id = 6
    phase_name = "LOCAL_METRICS"
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

    lm_cfg = cfg.get("local_metrics") if isinstance(cfg.get("local_metrics"), dict) else {}
    star_mode = lm_cfg.get("star_mode") if isinstance(lm_cfg.get("star_mode"), dict) else {}
    star_w = star_mode.get("weights") if isinstance(star_mode.get("weights"), dict) else {}
    try:
        w_fwhm = float(star_w.get("fwhm", 1.0 / 3.0))
    except Exception:
        w_fwhm = 1.0 / 3.0
    try:
        w_round = float(star_w.get("roundness", 1.0 / 3.0))
    except Exception:
        w_round = 1.0 / 3.0
    try:
        w_con = float(star_w.get("contrast", 1.0 / 3.0))
    except Exception:
        w_con = 1.0 / 3.0

    total_frames = sum(len(channels[ch]) for ch in ("R", "G", "B"))
    processed_frames = 0
    
    for ch in ("R", "G", "B"):
        q_local: List[List[float]] = []
        q_mean: List[float] = []
        q_var: List[float] = []
        for f_idx, f in enumerate(channels[ch]):
            tm = tile_calc.calculate_tile_metrics(f)
            fwhm = np.asarray(tm.get("fwhm") or [], dtype=np.float32)
            rnd = np.asarray(tm.get("roundness") or [], dtype=np.float32)
            con = np.asarray(tm.get("contrast") or [], dtype=np.float32)
            if fwhm.size == 0:
                q = np.zeros((0,), dtype=np.float32)
            else:
                mn = float(np.min(fwhm))
                mx = float(np.max(fwhm))
                if mx > mn:
                    inv_fwhm = 1.0 - (fwhm - mn) / (mx - mn)
                else:
                    inv_fwhm = np.ones_like(fwhm)
                q = (w_fwhm * inv_fwhm + w_round * rnd + w_con * con).astype(np.float32, copy=False)
            q_local.append([float(x) for x in q.tolist()])
            q_mean.append(float(np.mean(q)) if q.size else 0.0)
            q_var.append(float(np.var(q)) if q.size else 0.0)
            
            processed_frames += 1
            if processed_frames % 5 == 0 or processed_frames == total_frames:
                phase_progress(run_id, log_fp, phase_id, phase_name, processed_frames, total_frames, {"channel": ch})

        channel_metrics[ch]["tiles"] = {"Q_local": q_local, "tile_quality_mean": q_mean, "tile_quality_variance": q_var}

    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"tile_size": tile_size_i, "overlap": overlap_i})

    phase_id = 7
    phase_name = "TILE_RECONSTRUCTION"
    phase_start(run_id, log_fp, phase_id, phase_name)
    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
        return False

    reconstructed: Dict[str, np.ndarray] = {}
    hdr0 = None
    try:
        hdr0 = fits.getheader(str(registered_files[0]), ext=0)
    except Exception:
        hdr0 = None
    
    channels_to_process = [ch for ch in ("R", "G", "B") if channels[ch]]
    for ch_idx, ch in enumerate(channels_to_process, start=1):
        frs = channels[ch]
        gfc = np.asarray(channel_metrics[ch]["global"].get("G_f_c") or [], dtype=np.float32)
        if frs and gfc.size == len(frs) and float(np.sum(gfc)) > 0:
            wsum = float(np.sum(gfc))
            w_norm = (gfc / wsum).astype(np.float32, copy=False)
            out = np.zeros_like(frs[0], dtype=np.float32)
            for f, ww in zip(frs, w_norm):
                out += f.astype(np.float32, copy=False) * float(ww)
            reconstructed[ch] = out
        elif frs:
            reconstructed[ch] = np.mean(np.asarray(frs, dtype=np.float32), axis=0).astype(np.float32, copy=False)
        else:
            reconstructed[ch] = np.zeros((1, 1), dtype=np.float32)

        out_path = outputs_dir / f"reconstructed_{ch}.fits"
        fits.writeto(str(out_path), reconstructed[ch].astype("float32", copy=False), header=hdr0, overwrite=True)
        
        phase_progress(run_id, log_fp, phase_id, phase_name, ch_idx, len(channels_to_process), {"channel": ch})

    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {"outputs": [f"reconstructed_{c}.fits" for c in ("R", "G", "B")]})

    phase_id = 8
    phase_name = "STATE_CLUSTERING"
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
            clustering_results = cluster_channels(channels, channel_metrics, clustering_cfg)
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
        },
    )

    phase_id = 9
    phase_name = "SYNTHETIC_FRAMES"
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
            metrics_for_syn: Dict[str, Dict[str, Any]] = {}
            for ch in ("R", "G", "B"):
                g = channel_metrics.get(ch, {}).get("global", {}) if isinstance(channel_metrics.get(ch), dict) else {}
                t = channel_metrics.get(ch, {}).get("tiles", {}) if isinstance(channel_metrics.get(ch), dict) else {}
                g2 = dict(g)
                if "noise_level" in g2:
                    g2["noise_level"] = np.asarray(g2["noise_level"], dtype=np.float32)
                if "background_level" in g2:
                    g2["background_level"] = np.asarray(g2["background_level"], dtype=np.float32)
                metrics_for_syn[ch] = {"global": g2, "tiles": t}

            if generate_channel_synthetic_frames is not None:
                synthetic_channels = generate_channel_synthetic_frames(channels, metrics_for_syn, synthetic_cfg)
        except Exception:
            synthetic_channels = None

        if synthetic_channels:
            hdr_syn = None
            try:
                hdr_syn = fits.getheader(str(registered_files[0]), ext=0)
            except Exception:
                hdr_syn = None
            syn_r = synthetic_channels.get("R") or []
            syn_g = synthetic_channels.get("G") or []
            syn_b = synthetic_channels.get("B") or []
            synthetic_count = max(len(syn_r), len(syn_g), len(syn_b))
            for i in range(synthetic_count):
                r = np.asarray(syn_r[i]).astype("float32", copy=False) if i < len(syn_r) else None
                g = np.asarray(syn_g[i]).astype("float32", copy=False) if i < len(syn_g) else None
                b = np.asarray(syn_b[i]).astype("float32", copy=False) if i < len(syn_b) else None
                if r is None or g is None or b is None:
                    continue
                outp_rgb = syn_out / f"syn_{i+1:05d}.fits"
                rgb = np.stack([r, g, b], axis=0)
                fits.writeto(str(outp_rgb), rgb, header=hdr_syn, overwrite=True)
                outp_r = syn_out / f"synR_{i+1:05d}.fits"
                outp_g = syn_out / f"synG_{i+1:05d}.fits"
                outp_b = syn_out / f"synB_{i+1:05d}.fits"
                fits.writeto(str(outp_r), r, header=hdr_syn, overwrite=True)
                fits.writeto(str(outp_g), g, header=hdr_syn, overwrite=True)
                fits.writeto(str(outp_b), b, header=hdr_syn, overwrite=True)

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
        },
    )

    phase_id = 10
    phase_name = "STACKING"
    phase_start(run_id, log_fp, phase_id, phase_name)
    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
        return False
    if stack_engine != "siril":
        phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"stacking.engine not supported: {stack_engine!r}"})
        return False
    if not siril_exe:
        phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "siril executable not found"})
        return False

    stack_work = work_dir / "stacking"
    stack_work.mkdir(parents=True, exist_ok=True)
    
    # In reduced mode with skipped synthetic frames, stack reconstructed channels directly
    if reduced_mode and synthetic_skipped:
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

    using_default_stack_script = not (isinstance(stack_script_cfg, str) and stack_script_cfg.strip())
    if using_default_stack_script and stack_method != "average":
        phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {
                "error": "default stacking script is only defined for stacking.method=average; set stacking.siril_script for other methods",
                "stacking_method": stack_method,
            },
        )
        return False
    if not stack_script_path.exists() or not stack_script_path.is_file():
        phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": f"stacking script not found: {stack_script_path}"})
        return False

    ok_script, violations = validate_siril_script(stack_script_path)
    if not ok_script:
        phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {"error": "stacking script violates policy", "script": str(stack_script_path), "violations": violations},
        )
        return False

    for i, src in enumerate(stack_files, start=1):
        dst = stack_work / f"seq{i:05d}.fit"
        safe_symlink_or_copy(src, dst)

    before = sorted([p.name for p in stack_work.iterdir() if p.is_file()])
    ok, meta = run_siril_script(
        siril_exe=siril_exe,
        work_dir=stack_work,
        script_path=stack_script_path,
        artifacts_dir=artifacts_dir,
        log_name="siril_stacking.log",
        quiet=True,
    )
    if not ok:
        phase_end(run_id, log_fp, phase_id, phase_name, "error", {"siril": meta, "method": stack_method})
        return False

    after_files = sorted([p for p in stack_work.iterdir() if p.is_file()])
    after_names = {p.name for p in after_files}
    new_names = sorted(list(after_names.difference(set(before))))

    save_targets = extract_siril_save_targets(stack_script_path)
    candidates: List[Path] = []
    for name in save_targets:
        candidates.extend(
            [
                stack_work / name,
                stack_work / (name + ".fit"),
                stack_work / (name + ".fits"),
                stack_work / (name + ".fts"),
                stack_work / (Path(name).stem + ".fit"),
                stack_work / (Path(name).stem + ".fits"),
                stack_work / (Path(name).stem + ".fts"),
            ]
        )
    out_basename = Path(stack_output_file).name
    candidates.extend(
        [
            stack_work / out_basename,
            stack_work / (out_basename + ".fit"),
            stack_work / (out_basename + ".fits"),
            stack_work / (out_basename + ".fts"),
            stack_work / (Path(out_basename).stem + ".fit"),
            stack_work / (Path(out_basename).stem + ".fits"),
            stack_work / (Path(out_basename).stem + ".fts"),
        ]
    )
    produced = pick_output_file(candidates)
    if produced is None:
        new_fits = [stack_work / n for n in new_names if (stack_work / n).is_file() and is_fits_image_path(stack_work / n)]
        if len(new_fits) == 1:
            produced = new_fits[0]
    if produced is None:
        phase_end(
            run_id,
            log_fp,
            phase_id,
            phase_name,
            "error",
            {
                "error": "expected output not found",
                "siril": meta,
                "method": stack_method,
                "save_targets": save_targets,
                "new_files": new_names,
            },
        )
        return False

    if using_default_stack_script and stack_method == "average":
        n_stack = len(stack_files)
        if n_stack > 0:
            try:
                hdr = fits.getheader(str(produced), ext=0)
                data = fits.getdata(str(produced), ext=0)
                if data is not None:
                    data_f = data.astype("float32", copy=False)
                    data_f = data_f / float(n_stack)
                    fits.writeto(str(produced), data_f, header=hdr, overwrite=True)
            except Exception:
                pass

    final_out = outputs_dir / Path(stack_output_file)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(final_out))
    extra: Dict[str, Any] = {"siril": meta, "method": stack_method, "output": str(final_out)}
    phase_end(run_id, log_fp, phase_id, phase_name, "ok", extra)

    phase_id = 11
    phase_name = "DONE"
    phase_start(run_id, log_fp, phase_id, phase_name)
    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
        return False
    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {})

    return True


