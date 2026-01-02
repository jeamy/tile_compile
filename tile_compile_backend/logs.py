from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_run_logs(run_dir: str, tail: int | None = None) -> dict[str, Any]:
    p = Path(run_dir).expanduser().resolve()
    log_path = p / "logs" / "run_events.jsonl"
    if not log_path.exists():
        return {"run_dir": str(p), "events": []}

    lines: list[str] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)

    if tail is not None and tail > 0:
        lines = lines[-int(tail) :]

    events: list[dict[str, Any]] = []
    for s in lines:
        try:
            ev = json.loads(s)
            if isinstance(ev, dict):
                events.append(ev)
        except Exception:
            # ignore
            pass

    return {"run_dir": str(p), "events": events}
