from __future__ import annotations

import json
from pathlib import Path

from app.services.run_inspector import discover_runs, read_run_log_lines, read_run_status


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_read_run_status_computes_progress_and_phases(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "20260307_000001_x" / "M31"
    events_path = run_dir / "logs" / "run_events.jsonl"
    _write_events(
        events_path,
        [
            {"type": "run_start", "run_id": "20260307_000001_x/M31", "ts": "2026-03-07T00:00:00Z"},
            {"type": "phase_start", "phase_name": "SCAN_INPUT", "ts": "2026-03-07T00:00:01Z"},
            {"type": "phase_end", "phase_name": "SCAN_INPUT", "status": "ok", "ts": "2026-03-07T00:00:02Z"},
            {"type": "phase_start", "phase_name": "NORMALIZATION", "ts": "2026-03-07T00:00:03Z"},
            {"type": "phase_progress", "phase_name": "NORMALIZATION", "progress": 0.5, "ts": "2026-03-07T00:00:04Z"},
        ],
    )

    status = read_run_status(run_dir)
    assert status["status"] == "running"
    assert status["current_phase"] == "NORMALIZATION"
    assert status["progress"] > 0.0
    phase_scan = next(p for p in status["phases"] if p["phase"] == "SCAN_INPUT")
    phase_norm = next(p for p in status["phases"] if p["phase"] == "NORMALIZATION")
    assert phase_scan["status"] == "ok"
    assert phase_norm["status"] == "running"
    assert phase_norm["pct"] >= 0.5


def test_discover_runs_and_logs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_a = runs_dir / "20260307_100000_a" / "objA"
    run_b = runs_dir / "20260307_100001_b" / "objB"

    _write_events(
        run_a / "logs" / "run_events.jsonl",
        [
            {"type": "run_start", "run_id": "20260307_100000_a/objA", "ts": "2026-03-07T10:00:00Z"},
            {"type": "run_end", "success": True, "ts": "2026-03-07T10:10:00Z"},
        ],
    )
    _write_events(
        run_b / "logs" / "run_events.jsonl",
        [
            {"type": "run_start", "run_id": "20260307_100001_b/objB", "ts": "2026-03-07T10:01:00Z"},
            {"type": "warning", "message": "something", "ts": "2026-03-07T10:02:00Z"},
            {"type": "run_end", "success": False, "ts": "2026-03-07T10:03:00Z"},
        ],
    )

    items = discover_runs(runs_dir)
    assert len(items) == 2
    run_ids = {item["run_id"] for item in items}
    assert "20260307_100000_a/objA" in run_ids
    assert "20260307_100001_b/objB" in run_ids

    lines = read_run_log_lines(run_b, tail=2)
    assert len(lines) == 2
    assert "warning" in lines[0]
    assert "run_end" in lines[1]
