from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_text_if_exists(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        events.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    return events


def get_run_status(run_dir: str) -> dict[str, Any]:
    p = Path(run_dir).expanduser().resolve()

    log_path = p / "logs" / "run_events.jsonl"
    events = _read_jsonl_events(log_path)

    status = "PENDING"
    phase_id = None
    phase_name = None

    # Track current phase from last phase_start/phase_end
    for ev in events:
        t = ev.get("type")
        if t == "phase_start":
            phase_id = ev.get("phase")
            phase_name = ev.get("phase_name")
            status = "RUNNING"
        elif t == "phase_end":
            phase_id = ev.get("phase")
            phase_name = ev.get("phase_name")
            status = "RUNNING"
        elif t == "run_stop_requested":
            status = "ABORTING"
        elif t == "run_end":
            st = ev.get("status")
            if st == "ok":
                status = "SUCCESS"
            elif st == "stopped":
                status = "ABORTED"
            else:
                status = "FAILED"

    config_hash = _read_text_if_exists(p / "config_hash.txt")
    frames_manifest_id = _read_text_if_exists(p / "frames_manifest_id.txt")

    return {
        "status": status,
        "phase": phase_id,
        "phase_name": phase_name,
        "paths": {
            "run_dir": str(p),
            "logs": str((p / "logs")),
            "artifacts": str((p / "artifacts")),
            "outputs": str((p / "outputs")),
        },
        "config_hash": config_hash,
        "frames_manifest_id": frames_manifest_id,
    }
