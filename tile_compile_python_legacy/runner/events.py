"""
Event emission and phase tracking for the tile-compile runner.

Handles JSON event output for GUI consumption and logging.
"""

import sys
from datetime import datetime, timezone
from typing import Any

from .utils import json_dumps_canonical


def emit(event: dict, log_fp=None) -> None:
    """Emit event as JSON line to stdout and optional log file."""
    line = json_dumps_canonical(event).decode("utf-8")
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    if log_fp is not None:
        log_fp.write(line + "\n")
        log_fp.flush()


def phase_start(run_id: str, log_fp, phase_id: int, phase_name: str, extra: dict[str, Any] | None = None) -> None:
    """Emit phase_start event."""
    ev: dict[str, Any] = {
        "type": "phase_start",
        "run_id": run_id,
        "phase": phase_id,
        "phase_name": phase_name,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        ev.update(extra)
    emit(ev, log_fp)


def phase_end(
    run_id: str,
    log_fp,
    phase_id: int,
    phase_name: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit phase_end event."""
    ev: dict[str, Any] = {
        "type": "phase_end",
        "run_id": run_id,
        "phase": phase_id,
        "phase_name": phase_name,
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    if extra:
        ev.update(extra)
    emit(ev, log_fp)


def phase_progress(
    run_id: str, 
    log_fp, 
    phase_id: int, 
    phase_name: str, 
    current: int, 
    total: int, 
    extra: dict[str, Any] | None = None
) -> None:
    """Emit phase_progress event."""
    ev: dict[str, Any] = {
        "type": "phase_progress",
        "run_id": run_id,
        "phase": phase_id,
        "phase_name": phase_name,
        "ts": datetime.now(timezone.utc).isoformat(),
        "current": current,
        "total": total,
    }
    if extra:
        ev.update(extra)
    emit(ev, log_fp)


def stop_requested(run_id: str, log_fp, phase_id: int, phase_name: str, stop_flag: bool) -> bool:
    """Check if stop was requested and emit event if so."""
    if not stop_flag:
        return False
    emit(
        {
            "type": "run_stop_requested",
            "run_id": run_id,
            "phase": phase_id,
            "phase_name": phase_name,
            "ts": datetime.now(timezone.utc).isoformat(),
        },
        log_fp,
    )
    return True
