from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def tail_run_stream_events(run_dir: Path, *, cursor: int = 0, max_events: int = 300) -> tuple[list[dict[str, Any]], int]:
    path = _find_event_file(run_dir)
    if path is None:
        return [], 0

    events: list[dict[str, Any]] = []
    new_cursor = cursor
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                if line_no <= cursor:
                    continue
                raw = line.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                events.extend(_normalize_stream_events(item))
                new_cursor = line_no
                if len(events) >= max_events:
                    break
    except OSError:
        return [], cursor

    return events[:max_events], new_cursor


def build_queue_progress_event(run_id: str, queue: list[dict[str, Any]], current_index: int | None = None) -> dict[str, Any]:
    total = len(queue)
    done = len([q for q in queue if str(q.get("state", "")).lower() == "ok"])
    running = next((q for q in queue if str(q.get("state", "")).lower() == "running"), None)
    pct = 100.0
    if total > 0:
        pct = round((done / float(total)) * 100.0, 2)

    return {
        "type": "queue_progress",
        "run_id": run_id,
        "phase": None,
        "filter": (running or {}).get("filter"),
        "pct": pct,
        "ts": _event_ts(None),
        "payload": {
            "current_index": current_index,
            "total": total,
            "done": done,
            "queue": queue,
        },
    }


def _normalize_stream_events(raw: dict[str, Any]) -> list[dict[str, Any]]:
    typ = str(raw.get("type", "")).strip()
    run_id = _resolve_run_id(raw)
    phase = _resolve_phase(raw)
    filt = _resolve_filter(raw)
    ts = _event_ts(raw.get("ts"))

    if typ in {"phase_start", "phase_progress", "phase_end", "run_end", "queue_progress", "log_line"}:
        payload = _payload_without_common(raw)
        pct = _resolve_pct(raw)
        ev = {
            "type": typ,
            "run_id": run_id,
            "phase": phase,
            "filter": filt,
            "pct": pct,
            "ts": ts,
            "payload": payload,
        }
        if typ == "phase_progress":
            if "current" in raw:
                ev["current"] = raw.get("current")
            if "total" in raw:
                ev["total"] = raw.get("total")
            if "eta_s" in raw:
                ev["eta_s"] = raw.get("eta_s")
        return [ev]

    # Fallback: unknown event becomes log_line to keep stream lossless for FE.
    return [
        {
            "type": "log_line",
            "run_id": run_id,
            "phase": phase,
            "filter": filt,
            "pct": _resolve_pct(raw),
            "ts": ts,
            "payload": {
                "message": str(raw.get("message") or raw.get("msg") or typ or "event"),
                "raw": raw,
            },
        }
    ]


def _payload_without_common(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    for key in ["type", "run_id", "phase", "phase_name", "filter", "filter_name", "ts", "pct"]:
        out.pop(key, None)
    return out


def _resolve_pct(raw: dict[str, Any]) -> float | None:
    for key in ["pct", "progress"]:
        if key not in raw:
            continue
        try:
            value = float(raw.get(key))
        except (TypeError, ValueError):
            continue
        if value <= 1.0:
            value *= 100.0
        if value < 0.0:
            value = 0.0
        if value > 100.0:
            value = 100.0
        return round(value, 3)
    return None


def _resolve_phase(raw: dict[str, Any]) -> str | None:
    for key in ["phase", "phase_name"]:
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _resolve_filter(raw: dict[str, Any]) -> str | None:
    for key in ["filter", "filter_name"]:
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _resolve_run_id(raw: dict[str, Any]) -> str:
    value = raw.get("run_id")
    if isinstance(value, str) and value:
        return value
    return "unknown"


def _event_ts(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    return datetime.utcnow().isoformat() + "Z"


def _find_event_file(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "logs" / "run_events.jsonl",
        run_dir / "events.jsonl",
        run_dir / "logs" / "events.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
