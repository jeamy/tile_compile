from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.services.run_inspector import PHASE_ORDER

router = APIRouter(prefix="/app", tags=["app"])


@router.get("/state")
def app_state(request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    return {
        "project": {
            "project_root": str(runtime.project_root),
            "runs_dir": str(runtime.runs_dir),
            "default_config_path": str(runtime.default_config_path),
            "current_run_id": request.app.state.current_run_id,
        },
        "scan": {},
        "config": {
            "active_revision_id": request.app.state.active_config_revision_id,
            "revision_count": len(request.app.state.config_revisions),
        },
        "queue": {},
        "run": {},
        "history": {},
        "tools": {},
        "i18n": {"locale": "de"},
    }


@router.get("/constants")
def app_constants() -> dict[str, Any]:
    return {
        "phases": PHASE_ORDER,
        "color_modes": ["OSC", "MONO", "RGB"],
        "resume_from": ["ASTROMETRY", "BGE", "PCC"],
    }
