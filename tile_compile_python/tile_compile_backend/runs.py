from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_text_if_exists(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError):
        return None


def _read_json_if_exists(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _parse_run_dir_name(name: str) -> dict[str, Any]:
    # expected: YYYYMMDD_HHMMSS_<uuid>
    parts = name.split("_", 2)
    if len(parts) >= 3:
        ts_part = "_".join(parts[:2])
        run_id = parts[2]
        try:
            created_at = datetime.strptime(ts_part, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            created_at = None
        return {"run_id": run_id, "created_at": created_at}
    return {"run_id": name, "created_at": None}


def _read_last_status(run_dir: Path) -> str | None:
    log_path = run_dir / "logs" / "run_events.jsonl"
    if not log_path.exists():
        return None

    last = None
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    last = s
    except Exception:
        return None

    if not last:
        return None

    try:
        ev = json.loads(last)
    except Exception:
        return None

    if not isinstance(ev, dict):
        return None

    if ev.get("type") == "run_end":
        st = ev.get("status")
        if st == "ok":
            return "SUCCESS"
        if st == "stopped":
            return "ABORTED"
        if isinstance(st, str):
            return st.upper()
        return "FAILED"

    return "RUNNING"


def list_runs(runs_dir: str) -> list[dict[str, Any]]:
    base = Path(runs_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        return []

    items: list[dict[str, Any]] = []

    try:
        dirs = [x for x in base.iterdir() if x.is_dir()]
    except PermissionError:
        return []
    
    for p in sorted(dirs, key=lambda x: x.name, reverse=True):
        # Skip system directories
        if p.name in {"lost+found", ".Trash", ".Trash-1000"}:
            continue
            
        try:
            parsed = _parse_run_dir_name(p.name)
            run_metadata = _read_json_if_exists(p / "run_metadata.json")

            config_hash = _read_text_if_exists(p / "config_hash.txt")
            frames_manifest_id = _read_text_if_exists(p / "frames_manifest_id.txt")

            status = _read_last_status(p) or "PENDING"

            run_id = (run_metadata or {}).get("run_id") or parsed.get("run_id")
            created_at = (run_metadata or {}).get("created_at") or parsed.get("created_at")

            items.append(
                {
                    "run_id": run_id,
                    "status": status,
                    "created_at": created_at,
                    "run_dir": str(p),
                    "config_hash": config_hash,
                    "frames_manifest_id": frames_manifest_id,
                }
            )
        except PermissionError:
            # Skip directories we can't access
            continue

    return items
