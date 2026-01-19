#!/usr/bin/env python3
"""
Tile-Compile Runner - Entry Point

Refactored modular implementation of the Methodik v3 pipeline runner.
The monolithic runner has been split into logical modules in the runner/ package.

Usage:
    python tile_compile_runner.py [args]
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from runner.assumptions import get_assumptions_config
from runner.events import emit
from runner.phases import run_phases
from runner.phases_impl import run_phases_impl
from generate_artifacts_report import generate_report
from runner.utils import (
    copy_config,
    discover_frames,
    read_bytes,
    resolve_project_root,
    resolve_siril_exe,
    sha256_bytes,
)


def resume_run(args):
    """Resume a run from a specific phase."""
    run_dir = Path(args.run_dir).resolve()
    
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        return 1
    
    # Extract run_id from directory name
    run_id = run_dir.name
    
    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        # Try to find project root from run directory
        project_root = run_dir.parent.parent
    
    # Load config from run directory
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config not found in run directory: {config_path}", file=sys.stderr)
        return 1
    
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1
    
    # Discover frames from registered directory (Phase 1 output)
    reg_cfg = cfg.get("registration") if isinstance(cfg.get("registration"), dict) else {}
    reg_out_name = str(reg_cfg.get("output_dir") or "registered")
    reg_out_dir = run_dir / "outputs" / reg_out_name
    
    if not reg_out_dir.exists():
        print(f"Error: Registered frames directory not found: {reg_out_dir}", file=sys.stderr)
        return 1
    
    from runner.fits_utils import is_fits_image_path
    frames = sorted([p for p in reg_out_dir.iterdir() if p.is_file() and is_fits_image_path(p)])
    
    if not frames:
        print(f"Error: No registered frames found in {reg_out_dir}", file=sys.stderr)
        return 1
    
    # Resolve Siril executable
    siril_exe, siril_source = resolve_siril_exe(project_root)
    
    # Open log file in append mode
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_events.jsonl"
    
    # Emit resume event
    with open(log_path, "a", encoding="utf-8") as log_fp:
        emit(
            {
                "type": "run_resume",
                "run_id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "from_phase": args.from_phase,
                "siril_exe": siril_exe,
                "siril_source": siril_source,
            },
            log_fp,
        )
        
        # Run pipeline phases starting from specified phase
        try:
            success = run_phases_impl(
                run_id=run_id,
                log_fp=log_fp,
                dry_run=False,
                run_dir=run_dir,
                project_root=project_root,
                cfg=cfg,
                frames=frames,
                siril_exe=siril_exe,
                stop_flag=False,
                resume_from_phase=args.from_phase,
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            emit(
                {
                    "type": "run_error",
                    "run_id": run_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                    "traceback": tb,
                },
                log_fp,
            )
            print(f"Pipeline error: {e}\n{tb}", file=sys.stderr)
            success = False

        report_html = None
        report_error = None
        try:
            report_html_path, _ = generate_report(run_dir)
            report_html = str(report_html_path)
        except Exception as e:
            report_error = str(e)
        
        # Emit run_end event
        emit(
            {
                "type": "run_end",
                "run_id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "success": success,
                "status": "ok" if success else "error",
                "report_html": report_html,
                "report_error": report_error,
            },
            log_fp,
        )
    
    return 0 if success else 1


def main():
    """Main entry point for the tile-compile runner."""
    parser = argparse.ArgumentParser(description="Tile-Compile Runner (Methodik v3)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument("--config", type=str, required=True, help="Path to tile_compile.yaml")
    run_parser.add_argument("--input-dir", type=str, required=True, help="Input directory with FITS frames")
    run_parser.add_argument("--pattern", type=str, default="*.fit*", help="Input file pattern")
    run_parser.add_argument("--input-pattern", type=str, default=None, help="Input file pattern (alias)")
    run_parser.add_argument("--runs-dir", type=str, default="runs", help="Runs output directory")
    run_parser.add_argument("--dry-run", action="store_true", help="Dry run (emit events only)")
    run_parser.add_argument("--project-root", type=str, default=None, help="Project root directory")
    run_parser.add_argument("--color-mode-confirmed", type=str, default=None, help="Color mode confirmation (OSC/RGB/MONO)")
    
    # 'resume' subcommand
    resume_parser = subparsers.add_parser("resume", help="Resume a run from a specific phase")
    resume_parser.add_argument("--run-dir", type=str, required=True, help="Path to existing run directory")
    resume_parser.add_argument("--from-phase", type=int, required=True, help="Phase number to resume from (0-11)")
    resume_parser.add_argument("--project-root", type=str, default=None, help="Project root directory")
    
    args = parser.parse_args()
    
    # Handle command
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "resume":
        return resume_run(args)
    
    if args.command != "run":
        parser.print_help()
        return 1
    
    # Handle pattern argument aliases
    if args.input_pattern is None:
        args.input_pattern = args.pattern
    elif args.pattern != "*.fit*":
        args.input_pattern = args.pattern
    
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
    frames = discover_frames(input_dir, args.input_pattern)
    if not frames:
        print(f"No frames found in {input_dir} matching {args.input_pattern}", file=sys.stderr)
        return 1
    
    # Generate run ID and create run directory
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config to run directory
    config_copy_path = run_dir / "config.yaml"
    copy_config(config_path, config_copy_path)
    
    # Compute config hash
    config_bytes = read_bytes(config_path)
    config_hash = sha256_bytes(config_bytes)
    
    # Resolve Siril executable
    siril_exe, siril_source = resolve_siril_exe(project_root)
    
    # Create log file
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_events.jsonl"
    
    # Emit run_start event
    with open(log_path, "w", encoding="utf-8") as log_fp:
        emit(
            {
                "type": "run_start",
                "run_id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "config_path": str(config_path),
                "config_hash": config_hash,
                "input_dir": str(input_dir),
                "input_pattern": args.input_pattern,
                "frames_discovered": len(frames),
                "dry_run": args.dry_run,
                "siril_exe": siril_exe,
                "siril_source": siril_source,
                "paths": {
                    "project_root": str(project_root),
                    "run_dir": str(run_dir),
                    "config": str(config_copy_path),
                    "log": str(log_path),
                },
            },
            log_fp,
        )
        
        # Run pipeline phases
        try:
            success = run_phases(
                run_id=run_id,
                log_fp=log_fp,
                dry_run=args.dry_run,
                run_dir=run_dir,
                project_root=project_root,
                cfg=cfg,
                frames=frames,
                siril_exe=siril_exe,
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            emit(
                {
                    "type": "run_error",
                    "run_id": run_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                    "traceback": tb,
                },
                log_fp,
            )
            print(f"Pipeline error: {e}\n{tb}", file=sys.stderr)
            success = False

        report_html = None
        report_error = None
        try:
            report_html_path, _ = generate_report(run_dir)
            report_html = str(report_html_path)
        except Exception as e:
            report_error = str(e)
        
        # Emit run_end event
        emit(
            {
                "type": "run_end",
                "run_id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "success": success,
                "status": "ok" if success else "error",
                "report_html": report_html,
                "report_error": report_error,
            },
            log_fp,
        )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
