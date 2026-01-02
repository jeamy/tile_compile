import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


_STOP = False


def _handle_signal(_signum, _frame):
    global _STOP
    _STOP = True


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _json_dumps_canonical(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _emit(event: dict, log_fp=None):
    line = _json_dumps_canonical(event).decode("utf-8")
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    if log_fp is not None:
        log_fp.write(line + "\n")
        log_fp.flush()


def _copy_config(config_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_path)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _discover_frames(input_dir: Path, pattern: str):
    paths = sorted([p for p in input_dir.glob(pattern) if p.is_file()])
    return paths


def _run_phases(run_id: str, log_fp, dry_run: bool):
    phases = [
        (1, "registration"),
        (2, "global_normalization"),
        (3, "global_metrics"),
        (4, "tile_geometry"),
        (5, "local_tile_metrics"),
        (6, "tile_reconstruction"),
        (7, "state_clustering"),
        (8, "synthetic_frames"),
        (9, "final_stacking"),
        (10, "validation"),
    ]

    for phase_id, phase_name in phases:
        if _STOP:
            _emit(
                {
                    "type": "run_stop_requested",
                    "run_id": run_id,
                    "phase": phase_id,
                    "phase_name": phase_name,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
                log_fp,
            )
            return False

        _emit(
            {
                "type": "phase_start",
                "run_id": run_id,
                "phase": phase_id,
                "phase_name": phase_name,
                "ts": datetime.now(timezone.utc).isoformat(),
            },
            log_fp,
        )

        if not dry_run:
            time.sleep(0.15)

        _emit(
            {
                "type": "phase_end",
                "run_id": run_id,
                "phase": phase_id,
                "phase_name": phase_name,
                "ts": datetime.now(timezone.utc).isoformat(),
                "status": "ok",
            },
            log_fp,
        )

    return True


def cmd_run(args) -> int:
    config_path = Path(args.config).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    runs_dir = Path(args.runs_dir).expanduser().resolve()

    if not config_path.exists() or not config_path.is_file():
        sys.stderr.write(f"config not found: {config_path}\n")
        return 2

    if not input_dir.exists() or not input_dir.is_dir():
        sys.stderr.write(f"input_dir not found: {input_dir}\n")
        return 2

    run_id = str(uuid.uuid4())
    ts_compact = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{ts_compact}_{run_id}"

    paths = {
        "run_dir": str(run_dir),
        "runs_dir": str(runs_dir),
        "config_path": str(config_path),
        "input_dir": str(input_dir),
        "pattern": args.pattern,
    }

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "color_mode_confirmed": args.color_mode_confirmed,
    }
    (run_dir / "run_metadata.json").write_bytes(_json_dumps_canonical(run_metadata))

    log_path = run_dir / "logs" / "run_events.jsonl"
    with log_path.open("w", encoding="utf-8") as log_fp:
        config_bytes = _read_bytes(config_path)
        config_hash = _sha256_bytes(config_bytes)
        _copy_config(config_path, run_dir / "config.yaml")
        (run_dir / "config_hash.txt").write_text(config_hash + "\n", encoding="utf-8")

        frames = _discover_frames(input_dir, args.pattern)
        frames_manifest = {
            "input_dir": str(input_dir),
            "pattern": args.pattern,
            "frames": [str(p) for p in frames],
        }
        frames_manifest_bytes = _json_dumps_canonical(frames_manifest)
        frames_manifest_id = _sha256_bytes(frames_manifest_bytes)
        (run_dir / "frames_manifest.json").write_bytes(frames_manifest_bytes)
        (run_dir / "frames_manifest_id.txt").write_text(frames_manifest_id + "\n", encoding="utf-8")

        _emit(
            {
                "type": "run_start",
                "run_id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "paths": paths,
                "config_hash": config_hash,
                "frames_manifest_id": frames_manifest_id,
                "frame_count": len(frames),
                "dry_run": bool(args.dry_run),
                "color_mode_confirmed": args.color_mode_confirmed,
            },
            log_fp,
        )

        ok = _run_phases(run_id, log_fp, dry_run=bool(args.dry_run))

        _emit(
            {
                "type": "run_end",
                "run_id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "status": "ok" if ok else "stopped",
            },
            log_fp,
        )

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tile_compile_runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--input-dir", required=True)
    p_run.add_argument("--pattern", default="*.fit*")
    p_run.add_argument("--runs-dir", default="runs")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--color-mode-confirmed", default=None)
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv=None) -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
