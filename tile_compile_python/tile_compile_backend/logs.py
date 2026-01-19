from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Any


def get_run_logs(run_dir: str, tail: int | None = None) -> dict[str, Any]:
    p = Path(run_dir).expanduser().resolve()
    log_path = p / "logs" / "run_events.jsonl"
    if not log_path.exists():
        return {"run_dir": str(p), "events": []}

    # When the GUI requests only the last N events, it may miss early phase_end
    # events for phase 0/1 (e.g. SCAN_INPUT/REGISTRATION) and show them as pending.
    # To avoid loading the full file into memory, we stream the log and keep:
    # - the last N lines (tail)
    # - the latest phase_end event per phase_name
    tail_n = int(tail) if tail is not None and tail > 0 else None
    tail_lines: deque[str] | list[str]
    if tail_n is not None:
        tail_lines = deque(maxlen=tail_n)
    else:
        tail_lines = []

    phase_end_by_name: dict[str, dict[str, Any]] = {}

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if tail_n is not None:
                tail_lines.append(s)
            else:
                tail_lines.append(s)

            # Fast path: only parse phase_end lines to build a per-phase summary.
            if '"type":"phase_end"' in s or '"type": "phase_end"' in s:
                try:
                    ev = json.loads(s)
                    if isinstance(ev, dict):
                        phase_name = ev.get("phase_name")
                        if isinstance(phase_name, str) and phase_name.strip():
                            phase_end_by_name[phase_name] = ev
                except Exception:
                    pass

    events: list[dict[str, Any]] = []
    for s in tail_lines:
        try:
            ev = json.loads(s)
            if isinstance(ev, dict):
                events.append(ev)
        except Exception:
            # ignore
            pass

    if tail_n is not None and phase_end_by_name:
        # Ensure early phases are not missing just because they happened before tail.
        phase_end_in_tail: set[str] = set()
        for ev in events:
            if ev.get("type") == "phase_end":
                phase_name = ev.get("phase_name")
                if isinstance(phase_name, str) and phase_name.strip():
                    phase_end_in_tail.add(phase_name)

        for phase_name, ev in phase_end_by_name.items():
            if phase_name not in phase_end_in_tail:
                events.append(ev)

    return {"run_dir": str(p), "events": events}
