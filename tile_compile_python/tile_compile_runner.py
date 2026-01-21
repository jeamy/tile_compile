#!/usr/bin/env python3
"""
Tile-Compile Runner v4 - Entry Point

Methodik v4 implementation: tile-centric reconstruction without global registration.
All registration is tile-local, no global coordinate system.

Pipeline (v4):
    Phase 0:  SCAN_INPUT
    Phase 1:  CHANNEL_SPLIT
    Phase 2:  NORMALIZATION
    Phase 3:  GLOBAL_METRICS
    Phase 4:  TILE_GRID
    Phase 5:  LOCAL_METRICS
    Phase 6:  TILE_RECONSTRUCTION_TLR
    Phase 7:  STATE_CLUSTERING
    Phase 8:  SYNTHETIC_FRAMES
    Phase 9:  STACKING
    Phase 10: DEBAYER
    Phase 11: DONE
"""

import argparse
import json
import sys
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml
from astropy.io import fits

try:
    import psutil
except ImportError:
    psutil = None

from runner.events import emit
from runner.tile_processor_v4 import (
    TileProcessor,
    TileProcessorConfig,
    overlap_add,
    build_initial_tile_grid,
    refine_tiles,
    global_coarse_normalize,
    compute_global_weights,
)
from runner.adaptive_tile_grid import build_optimized_tile_grid
from runner.v4_parallel import process_tile_job
from runner.utils import (
    copy_config,
    discover_frames,
    read_bytes,
    resolve_project_root,
    sha256_bytes,
)
from runner.fits_utils import fits_is_cfa, fits_get_bayerpat
from runner.image_processing import split_cfa_channels, demosaic_cfa
from generate_artifacts_report import generate_report

EPS = 1e-6


def load_frame(path: Path) -> Optional[np.ndarray]:
    """Load single FITS frame from disk (memmap for memory efficiency)."""
    try:
        with fits.open(str(path), memmap=True) as hdul:
            data = hdul[0].data
            if data is not None:
                return data.astype(np.float32, copy=False)
    except (ValueError, OSError) as e:
        # Fallback for FITS with BZERO/BSCALE/BLANK keywords
        if "memmap" in str(e).lower() or "BZERO" in str(e) or "BSCALE" in str(e):
            # print(f"[WARNING] FITS has BZERO/BSCALE keywords - using memmap=False (higher RAM usage): {path.name}")
            try:
                with fits.open(str(path), memmap=False) as hdul:
                    data = hdul[0].data
                    if data is not None:
                        return data.astype(np.float32, copy=True)
            except Exception as e2:
                print(f"[WARN] Failed to load {path}: {e2}")
        else:
            print(f"[WARN] Failed to load {path}: {e}")
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
    return None


def save_fits(path: Path, data: np.ndarray, header: Optional[fits.Header] = None):
    """Save array as FITS file."""
    hdu = fits.PrimaryHDU(data)
    if header:
        for key, val in header.items():
            if key not in ('SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND'):
                try:
                    hdu.header[key] = val
                except Exception:
                    pass
    hdu.writeto(str(path), overwrite=True)


def run_v4_pipeline(
    run_id: str,
    log_fp,
    dry_run: bool,
    run_dir: Path,
    project_root: Path,
    cfg: Dict[str, Any],
    frame_paths: List[Path],
    color_mode_confirmed: Optional[str] = None,
) -> bool:
    """Execute Methodik v4 tile-centric pipeline.
    
    Args:
        run_id: Unique run identifier
        log_fp: Log file handle
        dry_run: If True, only emit events
        run_dir: Run output directory
        project_root: Project root
        cfg: Configuration dict
        frame_paths: Input frame paths
        
    Returns:
        True if successful
    """
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    def phase_event(phase_id: int, phase_name: str, status: str, data: Dict = None):
        emit({
            "type": "phase_end" if status in ("ok", "error", "skipped") else "phase_start",
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase_id": phase_id,
            "phase_name": phase_name,
            "status": status,
            **(data or {}),
        }, log_fp)
    
    # Phase 0: SCAN_INPUT
    phase_event(0, "SCAN_INPUT", "start")
    if dry_run:
        phase_event(0, "SCAN_INPUT", "skipped", {"reason": "dry_run"})
        return True
    
    print(f"[Phase 0] Scanning {len(frame_paths)} frames...")
    
    # Detect color mode from first frame
    is_cfa = fits_is_cfa(frame_paths[0])
    bayer_pattern = fits_get_bayerpat(frame_paths[0]) if is_cfa else None
    detected_mode = "OSC" if is_cfa else "MONO"
    
    # Validate against confirmed mode if provided
    if color_mode_confirmed:
        if color_mode_confirmed != detected_mode:
            print(f"[Phase 0] WARNING: Detected {detected_mode} but confirmed as {color_mode_confirmed}")
            detected_mode = color_mode_confirmed
    
    print(f"[Phase 0] Color mode: {detected_mode}")
    if detected_mode == "OSC":
        print(f"[Phase 0] Bayer pattern: {bayer_pattern or 'GBRG (default)'}")
    
    # Validate first frame loads
    test_frame = load_frame(frame_paths[0])
    if test_frame is None:
        phase_event(0, "SCAN_INPUT", "error", {"error": "cannot_load_first_frame"})
        return False
    
    # Get reference header
    ref_header = None
    try:
        with fits.open(str(frame_paths[0]), memmap=True) as hdul:
            ref_header = hdul[0].header.copy()
    except (ValueError, OSError):
        # Fallback for FITS with BZERO/BSCALE keywords
        # print(f"[WARNING] Reference FITS has BZERO/BSCALE keywords - using memmap=False")
        try:
            with fits.open(str(frame_paths[0]), memmap=False) as hdul:
                ref_header = hdul[0].header.copy()
        except Exception:
            pass
    except Exception:
        pass
    
    phase_event(0, "SCAN_INPUT", "ok", {
        "frames_scanned": len(frame_paths),
        "color_mode": detected_mode,
        "bayer_pattern": bayer_pattern,
    })
    
    # Phase 1: CHANNEL_SPLIT (metadata only - actual split happens during tile processing)
    phase_event(1, "CHANNEL_SPLIT", "start")
    
    bp = bayer_pattern or cfg.get("data", {}).get("bayer_pattern", "GBRG")
    
    if detected_mode == "OSC":
        print("[Phase 1] OSC mode - channels will be split during tile processing")
        channel_names = ["R", "G", "B"]
        phase_event(1, "CHANNEL_SPLIT", "ok", {
            "mode": "OSC",
            "channels": channel_names,
            "bayer_pattern": bp,
            "note": "deferred_to_tile_processing",
        })
    else:
        # MONO: single channel
        channel_names = ["L"]
        print("[Phase 1] MONO mode - single luminance channel")
        phase_event(1, "CHANNEL_SPLIT", "ok", {
            "mode": "MONO",
            "channels": channel_names,
        })
    
    # Phase 2: NORMALIZATION (compute normalization factors, apply during tile loading)
    phase_event(2, "NORMALIZATION", "start")
    print("[Phase 2] Computing normalization factors...")
    
    # Compute median per frame for normalization
    frame_medians = []
    for path in frame_paths:
        frame = load_frame(path)
        if frame is not None:
            frame_medians.append(float(np.median(frame)))
        else:
            frame_medians.append(0.0)
    
    target_median = float(np.median([m for m in frame_medians if m > 0]))
    print(f"[Phase 2] Target median: {target_median:.2f}")
    
    phase_event(2, "NORMALIZATION", "ok", {
        "target_median": target_median,
        "note": "normalization_applied_during_tile_loading",
    })
    
    # Phase 3: GLOBAL_METRICS (stream frames to compute weights)
    phase_event(3, "GLOBAL_METRICS", "start")
    print("[Phase 3] Computing global weights...")
    
    # Load frames one at a time to compute global metrics
    frames_for_metrics = []
    for i, path in enumerate(frame_paths):
        if i % 50 == 0:
            print(f"[Phase 3]   Loading frame {i}/{len(frame_paths)} for metrics...")
        frame = load_frame(path)
        if frame is not None:
            frames_for_metrics.append(frame)
    
    global_weights = compute_global_weights(frames_for_metrics, cfg)
    print(f"[Phase 3] Computed {len(global_weights)} global weights")
    
    # Free memory
    del frames_for_metrics
    
    phase_event(3, "GLOBAL_METRICS", "ok", {
        "num_frames": len(global_weights),
    })
    
    # Phase 4: TILE_GRID (with optional adaptive optimization)
    phase_event(4, "TILE_GRID", "start")
    print("[Phase 4] Building tile grid...")
    
    shape = test_frame.shape[:2]
    v4_cfg = cfg.get("v4", {})
    adaptive_cfg = v4_cfg.get("adaptive_tiles", {})
    
    if adaptive_cfg.get("use_warp_probe", False) or adaptive_cfg.get("use_hierarchical", False):
        print("[Phase 4] Using adaptive tile grid optimization...")
        tiles, gradient_field = build_optimized_tile_grid(shape, cfg, frame_paths)
        if gradient_field is not None:
            print(f"[Phase 4] Warp gradient field computed (max={np.max(gradient_field):.3f})")
    else:
        tiles = build_initial_tile_grid(shape, cfg)
        gradient_field = None
    
    print(f"[Phase 4] Generated {len(tiles)} tiles")
    phase_event(4, "TILE_GRID", "ok", {"num_tiles": len(tiles), "adaptive": gradient_field is not None})
    
    # Free test frame
    del test_frame
    
    # Phase 5: LOCAL_METRICS (computed during TLR)
    phase_event(5, "LOCAL_METRICS", "start")
    phase_event(5, "LOCAL_METRICS", "ok", {"note": "computed_during_tlr"})
    
    # Phase 6: TILE_RECONSTRUCTION_TLR (disk streaming per tile with adaptive refinement)
    phase_event(6, "TILE_RECONSTRUCTION_TLR", "start")
    print("[Phase 6] Tile-local registration and reconstruction (disk streaming)...")
    
    cfg_obj = TileProcessorConfig(cfg)
    
    # Get v4 configuration
    v4_cfg = cfg.get("v4", {})
    
    # Validate parallel_tiles configuration
    parallel_tiles = v4_cfg.get("parallel_tiles")
    if parallel_tiles is None:
        print("[WARNING] v4.parallel_tiles not configured, defaulting to 1 (serial)")
        v4_cfg["parallel_tiles"] = 1
    
    # Adaptive tile refinement configuration
    adaptive_cfg = v4_cfg.get("adaptive_tiles", {})
    adaptive_enabled = adaptive_cfg.get("enabled", False)
    max_refine_passes = adaptive_cfg.get("max_refine_passes", 0) if adaptive_enabled else 0
    refine_variance_threshold = adaptive_cfg.get("refine_variance_threshold", 0.25)
    min_tile_size = adaptive_cfg.get("min_tile_size_px", 64)
    
    print(f"[Phase 6] Adaptive refinement: {adaptive_enabled} (max passes: {max_refine_passes})")
    
    all_results = {}
    tile_metadata = []
    
    # Memory monitoring setup
    memory_limits_cfg = v4_cfg.get("memory_limits", {})
    rss_warn_mb = memory_limits_cfg.get("rss_warn_mb", 4096)
    rss_abort_mb = memory_limits_cfg.get("rss_abort_mb", 8192)
    proc = psutil.Process(os.getpid()) if psutil else None
    
    # Get parallel configuration
    parallel_tiles = v4_cfg.get("parallel_tiles", 1)
    
    # CI guard: check CPU core count
    cpu_cores = os.cpu_count() or 1
    if parallel_tiles > cpu_cores:
        print(f"[WARNING] parallel_tiles ({parallel_tiles}) exceeds CPU cores ({cpu_cores}), capping to {cpu_cores}")
        parallel_tiles = cpu_cores
    
    print(f"[Phase 6] Using {parallel_tiles} parallel workers")
    
    # Track cumulative progress across all refinement passes
    cumulative_tiles_processed = 0
    initial_tile_count = len(tiles)
    
    # Multi-pass tile refinement loop
    for refine_pass in range(max_refine_passes + 1):
        if refine_pass > 0:
            print(f"[Phase 6] Refinement pass {refine_pass}/{max_refine_passes}...")
        
        warp_variances = []
        pass_results = []
        
        # Prepare jobs for parallel execution
        jobs = []
        for tid, bbox in enumerate(tiles):
            jobs.append((tid, bbox, frame_paths, global_weights, cfg))
        
        # Execute tiles in parallel
        if parallel_tiles > 1:
            print(f"  Processing {len(jobs)} tiles with {parallel_tiles} workers...")
            with ProcessPoolExecutor(max_workers=parallel_tiles) as exe:
                futures = [exe.submit(process_tile_job, j) for j in jobs]
                
                completed = 0
                for f in as_completed(futures):
                    tid, bbox, tile, warps = f.result()
                    
                    completed += 1
                    
                    # Memory monitoring
                    if proc and completed % 50 == 0:
                        rss_mb = proc.memory_info().rss / 1e6
                        if rss_mb > rss_abort_mb:
                            phase_event(6, "TILE_RECONSTRUCTION_TLR", "error", {
                                "error": "memory_limit_exceeded",
                                "rss_mb": rss_mb,
                                "limit_mb": rss_abort_mb,
                            })
                            raise RuntimeError(f"RSS limit exceeded: {rss_mb:.0f} MB > {rss_abort_mb} MB (v4)")
                        elif rss_mb > rss_warn_mb:
                            print(f"[WARNING] High RSS usage: {rss_mb:.0f} MB (limit: {rss_abort_mb} MB)")
                    
                    # Emit progress event for GUI
                    from runner.events import phase_progress
                    total_progress = cumulative_tiles_processed + completed
                    total_tiles = initial_tile_count * (max_refine_passes + 1)
                    phase_progress(
                        run_id=run_id,
                        log_fp=log_fp,
                        phase_id=6,
                        phase_name="TILE_RECONSTRUCTION_TLR",
                        current=total_progress,
                        total=total_tiles,
                    )
                    
                    if completed % 10 == 0:
                        print(f"  Completed {completed}/{len(tiles)} tiles...")
                    
                    # Store results
                    if tile is None:
                        all_results[bbox] = None
                        warp_variances.append(float("inf"))
                    else:
                        all_results[bbox] = tile
                        # Compute warp variance from warps
                        if warps is not None and len(warps) > 0:
                            warp_var = np.var([w[0] for w in warps]) + np.var([w[1] for w in warps])
                        else:
                            warp_var = 0.0
                        warp_variances.append(warp_var)
                        pass_results.append((bbox, tile, warp_var))
        else:
            # Serial fallback (parallel_tiles = 1)
            print(f"  Processing {len(jobs)} tiles serially...")
            for tid, bbox in enumerate(tiles):
                # Memory monitoring
                if proc and tid % 50 == 0:
                    rss_mb = proc.memory_info().rss / 1e6
                    if rss_mb > rss_abort_mb:
                        phase_event(6, "TILE_RECONSTRUCTION_TLR", "error", {
                            "error": "memory_limit_exceeded",
                            "rss_mb": rss_mb,
                            "limit_mb": rss_abort_mb,
                        })
                        raise RuntimeError(f"RSS limit exceeded: {rss_mb:.0f} MB > {rss_abort_mb} MB (v4)")
                    elif rss_mb > rss_warn_mb:
                        print(f"[WARNING] High RSS usage: {rss_mb:.0f} MB (limit: {rss_abort_mb} MB)")
                
                # Emit progress event for GUI
                from runner.events import phase_progress
                total_progress = cumulative_tiles_processed + tid + 1
                total_tiles = initial_tile_count * (max_refine_passes + 1)
                phase_progress(
                    run_id=run_id,
                    log_fp=log_fp,
                    phase_id=6,
                    phase_name="TILE_RECONSTRUCTION_TLR",
                    current=total_progress,
                    total=total_tiles,
                )
                
                if tid % 10 == 0:
                    print(f"  Tile {tid + 1}/{len(tiles)}...")
                
                tp = TileProcessor(
                    tile_id=tid,
                    bbox=bbox,
                    frame_paths=frame_paths,
                    global_weights=global_weights,
                    cfg=cfg_obj,
                )
                tile, warps = tp.run()
                
                meta = tp.get_metadata()
                tile_metadata.append(meta)
                
                if tile is None:
                    all_results[bbox] = None
                    warp_variances.append(float("inf"))
                    continue
                
                all_results[bbox] = tile
                warp_variances.append(meta["warp_variance"])
                pass_results.append((bbox, tile, meta["warp_variance"]))
        
        # Update cumulative progress after this pass
        cumulative_tiles_processed += len(tiles)
        
        # Adaptive refinement: split high-variance tiles
        if refine_pass < max_refine_passes and adaptive_enabled:
            from tile_compile_backend.tile_grid import refine_tiles
            tiles_before = len(tiles)
            tiles = refine_tiles(
                tiles,
                warp_variances,
                threshold=refine_variance_threshold,
                min_size=min_tile_size,
            )
            tiles_after = len(tiles)
            if tiles_after > tiles_before:
                print(f"[Phase 6] Refined {tiles_before} → {tiles_after} tiles (split {tiles_after - tiles_before})")
            else:
                print(f"[Phase 6] No tiles refined (converged)")
                break
    
    # Collect final results
    results = [(bbox, tile, var) for (bbox, tile, var) in pass_results]
    valid_tiles = len([r for r in results if r[1] is not None])
    
    # Validity check (Methodik v4 §12)
    validity_threshold = cfg.get("v4", {}).get("min_valid_tile_fraction", 0.3)
    if valid_tiles < validity_threshold * len(tiles):
        phase_event(6, "TILE_RECONSTRUCTION_TLR", "error", {
            "error": "too_few_valid_tiles",
            "valid": valid_tiles,
            "total": len(tiles),
            "threshold": validity_threshold,
        })
        return False
    
    print(f"[Phase 6] Valid tiles: {valid_tiles}/{len(tiles)}")
    
    phase_event(6, "TILE_RECONSTRUCTION_TLR", "ok", {
        "valid_tiles": valid_tiles,
        "total_tiles": len(tiles),
    })
    
    # Phase 7: STATE_CLUSTERING (simplified)
    phase_event(7, "STATE_CLUSTERING", "start")
    # TODO: Full clustering implementation
    phase_event(7, "STATE_CLUSTERING", "ok", {"note": "simplified"})
    
    # Phase 8: SYNTHETIC_FRAMES (simplified)
    phase_event(8, "SYNTHETIC_FRAMES", "start")
    phase_event(8, "SYNTHETIC_FRAMES", "ok", {"note": "skipped_in_v4_simple"})
    
    # Phase 9: STACKING (overlap-add)
    phase_event(9, "STACKING", "start")
    print("[Phase 9] Overlap-add reconstruction...")
    
    variance_sigma = cfg.get("registration", {}).get("local_tiles", {}).get("variance_window_sigma", 2.0)
    final = overlap_add(results, shape, variance_sigma)
    
    # Save result
    output_path = outputs_dir / "stacked.fits"
    save_fits(output_path, final, ref_header)
    print(f"[Phase 9] Saved: {output_path}")
    
    phase_event(9, "STACKING", "ok", {
        "output": str(output_path),
    })
    
    # Phase 10: DEBAYER
    phase_event(10, "DEBAYER", "start")
    
    if detected_mode == "OSC" and cfg.get("debayer", True):
        print("[Phase 10] Debayering CFA mosaic...")
        rgb = demosaic_cfa(final, bp)
        rgb_path = outputs_dir / "stacked_rgb.fits"
        save_fits(rgb_path, rgb, ref_header)
        print(f"[Phase 10] Saved RGB: {rgb_path}")
        phase_event(10, "DEBAYER", "ok", {
            "mode": "OSC",
            "output": str(rgb_path),
        })
    else:
        print("[Phase 10] MONO mode - no debayering needed")
        phase_event(10, "DEBAYER", "ok", {"mode": "MONO"})
    
    # Phase 11: DONE
    phase_event(11, "DONE", "start")
    
    # Save tile metadata for diagnostics
    meta_path = outputs_dir / "tile_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(tile_metadata, f, indent=2, default=str)
    
    # Phase 11: Diagnostics (v4 YAML-controlled)
    diagnostics_cfg = v4_cfg.get("diagnostics", {})
    if diagnostics_cfg.get("enabled", True):
        print("[Phase 11] Generating diagnostics...")
        
        if diagnostics_cfg.get("tile_invalid_map", True):
            # Save map of invalid tiles
            invalid_map = np.zeros(shape, dtype=np.uint8)
            for meta in tile_metadata:
                if not meta.get("valid", True):
                    bbox = meta.get("bbox")
                    if bbox:
                        x0, y0, w, h = bbox
                        invalid_map[y0:y0+h, x0:x0+w] = 255
            invalid_map_path = outputs_dir / "tile_invalid_map.fits"
            save_fits(invalid_map_path, invalid_map)
            print(f"[Phase 11] Saved invalid tile map: {invalid_map_path}")
        
        if diagnostics_cfg.get("warp_variance_hist", True):
            # Save warp variance histogram data
            variances = [m.get("warp_variance", 0.0) for m in tile_metadata if m.get("valid", True)]
            if variances:
                hist_data = {
                    "variances": variances,
                    "mean": float(np.mean(variances)),
                    "median": float(np.median(variances)),
                    "std": float(np.std(variances)),
                    "min": float(np.min(variances)),
                    "max": float(np.max(variances)),
                }
                hist_path = outputs_dir / "warp_variance_stats.json"
                with open(hist_path, "w") as f:
                    json.dump(hist_data, f, indent=2)
                print(f"[Phase 11] Saved warp variance stats: {hist_path}")
    
    phase_event(11, "DONE", "ok", {
        "output": str(output_path),
        "tile_metadata": str(meta_path),
    })
    
    return True


def main():
    """Main entry point for tile-compile runner v4."""
    parser = argparse.ArgumentParser(description="Tile-Compile Runner (Methodik v4)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run the v4 pipeline")
    run_parser.add_argument("--config", type=str, required=True, help="Path to tile_compile.yaml")
    run_parser.add_argument("--input-dir", type=str, required=True, help="Input directory with FITS frames")
    run_parser.add_argument("--pattern", type=str, default="*.fit*", help="Input file pattern")
    run_parser.add_argument("--runs-dir", type=str, default="runs", help="Runs output directory")
    run_parser.add_argument("--dry-run", action="store_true", help="Dry run (emit events only)")
    run_parser.add_argument("--project-root", type=str, default=None, help="Project root directory")
    run_parser.add_argument("--color-mode-confirmed", type=str, default=None, help="Color mode confirmation (OSC/MONO)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command != "run":
        parser.print_help()
        return 1
    
    # Resolve paths
    config_path = Path(args.config).resolve()
    input_dir = Path(args.input_dir).resolve()
    
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = resolve_project_root(config_path)
    
    runs_dir = (project_root / args.runs_dir).resolve()
    
    # Load configuration
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1
    
    # Discover input frames
    frame_paths = discover_frames(input_dir, args.pattern)
    if not frame_paths:
        print(f"No frames found in {input_dir} matching {args.pattern}", file=sys.stderr)
        return 1
    
    # Generate run ID and create run directory
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config
    config_copy_path = run_dir / "config.yaml"
    copy_config(config_path, config_copy_path)
    
    # Compute config hash
    config_bytes = read_bytes(config_path)
    config_hash = sha256_bytes(config_bytes)
    
    # Create log file
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_events.jsonl"
    
    print(f"[v4] Starting run: {run_id}")
    print(f"[v4] Input: {len(frame_paths)} frames from {input_dir}")
    
    with open(log_path, "w", encoding="utf-8") as log_fp:
        # Emit run_start
        emit({
            "type": "run_start",
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "config_hash": config_hash,
            "input_dir": str(input_dir),
            "input_pattern": args.pattern,
            "frames_discovered": len(frame_paths),
            "dry_run": args.dry_run,
            "methodik_version": "v4",
            "paths": {
                "project_root": str(project_root),
                "run_dir": str(run_dir),
                "config": str(config_copy_path),
                "log": str(log_path),
            },
        }, log_fp)
        
        # Run v4 pipeline
        try:
            success = run_v4_pipeline(
                run_id=run_id,
                log_fp=log_fp,
                dry_run=args.dry_run,
                run_dir=run_dir,
                project_root=project_root,
                cfg=cfg,
                frame_paths=frame_paths,
                color_mode_confirmed=args.color_mode_confirmed,
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            emit({
                "type": "run_error",
                "run_id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "traceback": tb,
            }, log_fp)
            print(f"Pipeline error: {e}\n{tb}", file=sys.stderr)
            success = False
        
        # Generate report
        report_html = None
        report_error = None
        try:
            report_html_path, _ = generate_report(run_dir)
            report_html = str(report_html_path)
        except Exception as e:
            report_error = str(e)
        
        # Emit run_end
        emit({
            "type": "run_end",
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "status": "ok" if success else "error",
            "report_html": report_html,
            "report_error": report_error,
        }, log_fp)
    
    print(f"[v4] Run {'completed' if success else 'failed'}: {run_id}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
