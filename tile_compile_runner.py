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
from runner.utils import (
    copy_config,
    discover_frames,
    read_bytes,
    resolve_project_root,
    resolve_siril_exe,
    sha256_bytes,
)


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
    
    args = parser.parse_args()
    
    # Handle command
    if not args.command or args.command != "run":
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
        
        # Emit run_end event
        emit(
            {
                "type": "run_end",
                "run_id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "success": success,
            },
            log_fp,
        )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
