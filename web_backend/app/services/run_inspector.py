from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PHASE_ORDER: list[str] = [
    "SCAN_INPUT",
    "CHANNEL_SPLIT",
    "NORMALIZATION",
    "GLOBAL_METRICS",
    "TILE_GRID",
    "REGISTRATION",
    "PREWARP",
    "COMMON_OVERLAP",
    "LOCAL_METRICS",
    "TILE_RECONSTRUCTION",
    "STATE_CLUSTERING",
    "SYNTHETIC_FRAMES",
    "STACKING",
    "DEBAYER",
    "ASTROMETRY",
    "BGE",
    "PCC",
]


def discover_runs(runs_dir: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []

    run_paths: dict[Path, Path] = {}
    for event_file in runs_dir.rglob("run_events.jsonl"):
        run_dir = _run_dir_from_event_file(event_file)
        run_paths[run_dir] = event_file
    for event_file in runs_dir.rglob("events.jsonl"):
        run_dir = _run_dir_from_event_file(event_file)
        run_paths.setdefault(run_dir, event_file)

    items: list[dict[str, Any]] = []
    for run_dir, event_file in run_paths.items():
        if not run_dir.exists():
            continue
        mtime = datetime.fromtimestamp(event_file.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        run_id = _extract_run_id_from_events(event_file) or str(run_dir.relative_to(runs_dir))
        status = read_run_status(run_dir).get("status", "unknown")
        items.append(
            {
                "name": run_id.split("/")[-1],
                "path": str(run_dir),
                "run_id": run_id,
                "modified": mtime,
                "status": status,
            }
        )
    items.sort(key=lambda x: x["modified"], reverse=True)
    return items[:limit]


def read_run_status(run_dir: Path) -> dict[str, Any]:
    event_file = _find_event_file(run_dir)
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "status": "unknown",
        "color_mode": _read_run_color_mode(run_dir),
        "current_phase": None,
        "progress": 0.0,
        "phases": [{"phase": p, "status": "pending", "pct": 0.0} for p in PHASE_ORDER],
        "events": [],
    }
    if not event_file:
        return result

    phases: dict[str, dict[str, Any]] = {p: {"phase": p, "status": "pending", "pct": 0.0} for p in PHASE_ORDER}
    encountered_extra: dict[str, dict[str, Any]] = {}
    run_status = "unknown"
    current_phase: str | None = None
    last_progress: dict[str, float] = {}
    events_tail: list[dict[str, Any]] = []

    for ev in _iter_jsonl(event_file):
        events_tail.append(ev)
        if len(events_tail) > 200:
            events_tail.pop(0)

        event_type = str(ev.get("type", ""))
        phase_name = _phase_name_from_event(ev)

        if phase_name:
            phase_state = phases.get(phase_name)
            if phase_state is None:
                phase_state = encountered_extra.setdefault(phase_name, {"phase": phase_name, "status": "pending", "pct": 0.0})

            if event_type == "phase_start":
                phase_state["status"] = "running"
                current_phase = phase_name
                if run_status in {"unknown", "pending"}:
                    run_status = "running"

            elif event_type == "phase_progress":
                progress = _clamp_progress(ev.get("progress"))
                if progress is not None:
                    phase_state["pct"] = max(float(phase_state["pct"]), progress)
                    last_progress[phase_name] = phase_state["pct"]
                phase_state["status"] = "running"
                current_phase = phase_name
                if run_status in {"unknown", "pending"}:
                    run_status = "running"

            elif event_type == "phase_end":
                raw = str(ev.get("status", "unknown")).lower()
                phase_state["status"] = "ok" if raw in {"ok", "skipped"} else raw
                phase_state["pct"] = 1.0 if raw in {"ok", "skipped"} else float(phase_state["pct"])
                if current_phase == phase_name and raw in {"ok", "skipped", "error", "aborted"}:
                    current_phase = None
                if raw in {"error", "aborted"}:
                    run_status = "failed"

        if event_type == "run_end":
            run_status = "completed" if bool(ev.get("success", False)) else "failed"

    if run_status == "unknown":
        if current_phase:
            run_status = "running"
        elif any(p["status"] == "ok" for p in phases.values()):
            run_status = "running"

    phase_list = [phases[p] for p in PHASE_ORDER] + list(encountered_extra.values())
    progress = _overall_progress(phase_list, current_phase, last_progress)
    if run_status == "completed":
        progress = 1.0

    result.update(
        {
            "status": run_status,
            "current_phase": current_phase,
            "progress": round(progress, 4),
            "phases": phase_list,
            "events": events_tail,
        }
    )
    return result


def read_run_log_lines(run_dir: Path, *, tail: int = 200) -> list[str]:
    event_file = _find_event_file(run_dir)
    if event_file is None:
        return []

    events = list(_iter_jsonl(event_file))
    if tail > 0:
        events = events[-tail:]
    return [_format_event_line(ev) for ev in events]


def _iter_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    yield parsed
    except OSError:
        return


def _run_dir_from_event_file(event_file: Path) -> Path:
    if event_file.name == "run_events.jsonl" and event_file.parent.name == "logs":
        return event_file.parent.parent
    return event_file.parent


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


def _read_run_color_mode(run_dir: Path) -> str:
    config_path = run_dir / "config.yaml"
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return "UNKNOWN"
    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return "UNKNOWN"
    if not isinstance(parsed, dict):
        return "UNKNOWN"
    data = parsed.get("data")
    if not isinstance(data, dict):
        return "UNKNOWN"
    color_mode = str(data.get("color_mode") or "").strip().upper()
    return color_mode if color_mode else "UNKNOWN"


def _extract_run_id_from_events(event_file: Path) -> str | None:
    for ev in _iter_jsonl(event_file):
        rid = ev.get("run_id")
        if isinstance(rid, str) and rid:
            return rid
    return None


def _phase_name_from_event(ev: dict[str, Any]) -> str | None:
    phase_name = ev.get("phase_name")
    if isinstance(phase_name, str) and phase_name:
        return phase_name
    phase = ev.get("phase")
    if isinstance(phase, str) and phase:
        return phase
    return None


def _clamp_progress(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _overall_progress(phase_list: list[dict[str, Any]], current_phase: str | None, progress_map: dict[str, float]) -> float:
    if not phase_list:
        return 0.0
    total = len(PHASE_ORDER)
    if total == 0:
        return 0.0

    completed = 0
    for phase in PHASE_ORDER:
        entry = next((p for p in phase_list if p.get("phase") == phase), None)
        if entry and entry.get("status") in {"ok"}:
            completed += 1

    current_component = 0.0
    if current_phase and current_phase in progress_map:
        current_component = progress_map[current_phase]
    elif current_phase:
        entry = next((p for p in phase_list if p.get("phase") == current_phase), None)
        if entry:
            try:
                current_component = float(entry.get("pct", 0.0))
            except (TypeError, ValueError):
                current_component = 0.0

    progress = (completed + current_component) / float(total)
    if progress < 0.0:
        return 0.0
    if progress > 1.0:
        return 1.0
    return progress


def _format_event_line(ev: dict[str, Any]) -> str:
    ts = str(ev.get("ts", ""))
    event_type = str(ev.get("type", "event"))
    phase = _phase_name_from_event(ev)
    msg = str(ev.get("message", "")).strip()
    status = str(ev.get("status", "")).strip()
    progress = _clamp_progress(ev.get("progress"))

    parts = [ts, event_type]
    if phase:
        parts.append(phase)
    if status:
        parts.append(f"status={status}")
    if progress is not None:
        parts.append(f"{progress * 100.0:.1f}%")
    if msg:
        parts.append(msg)
    return " | ".join(parts)
