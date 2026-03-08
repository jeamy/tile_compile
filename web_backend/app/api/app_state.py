from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.services.run_inspector import PHASE_ORDER, discover_runs, read_run_status
from app.services.scan_summary import latest_scan_job, summarize_scan_job

router = APIRouter(prefix="/app", tags=["app"])


@router.get("/state")
def app_state(request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    ui_events = request.app.state.ui_event_store
    fallback_input_path = str(getattr(request.app.state, "last_scan_input_path", "") or "")
    scan_job = latest_scan_job(request.app.state.job_store)
    scan_summary = summarize_scan_job(scan_job, fallback_input_path=fallback_input_path)
    current_run_id = str(getattr(request.app.state, "current_run_id", "") or "").strip()
    current_run_summary: dict[str, Any] = {}
    if current_run_id:
        try:
            current_run_dir = runtime.resolve_run_dir(current_run_id)
            current_run_status = read_run_status(current_run_dir)
            current_run_summary = {
                "run_id": current_run_id,
                "run_dir": str(current_run_dir),
                "status": current_run_status.get("status", "unknown"),
                "current_phase": current_run_status.get("current_phase"),
                "progress": current_run_status.get("progress", 0.0),
            }
        except Exception:
            current_run_summary = {"run_id": current_run_id, "status": "unknown"}
    recent_runs = discover_runs(runtime.runs_dir, limit=5)
    return {
        "project": {
            "project_root": str(runtime.project_root),
            "runs_dir": str(runtime.runs_dir),
            "default_config_path": str(runtime.default_config_path),
            "current_run_id": request.app.state.current_run_id,
        },
        "scan": {
            "last_input_path": scan_summary.get("input_path", fallback_input_path),
            "last_scan": scan_summary,
        },
        "config": {
            "active_revision_id": request.app.state.active_config_revision_id,
            "revision_count": len(request.app.state.config_revisions),
        },
        "queue": {},
        "run": {"current": current_run_summary},
        "history": {"total_runs": len(recent_runs), "recent": recent_runs},
        "tools": {},
        "events": {"latest_seq": ui_events.latest_seq},
        "i18n": {"locale": "de"},
    }


@router.get("/constants")
def app_constants() -> dict[str, Any]:
    return {
        "phases": PHASE_ORDER,
        "color_modes": ["OSC", "MONO", "RGB"],
        "resume_from": ["ASTROMETRY", "BGE", "PCC"],
    }


@router.get("/ui-events")
def app_ui_events(request: Request, after_seq: int = 0, limit: int = 200) -> dict[str, Any]:
    items = request.app.state.ui_event_store.list(after_seq=max(0, int(after_seq)), limit=int(limit))
    latest_seq = request.app.state.ui_event_store.latest_seq
    return {"items": items, "latest_seq": latest_seq}
