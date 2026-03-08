from __future__ import annotations

from fastapi import FastAPI

from app.api import app_state, config, jobs, runs, scan, system, tools, ws
from app.services.command_runner import BackendRuntime
from app.services.process_manager import InMemoryJobStore
from app.services.ui_events import UiEventStore


def create_app() -> FastAPI:
    app = FastAPI(title="tile_compile GUI2 backend", version="0.1.0")
    runtime = BackendRuntime.autodetect()
    app.state.job_store = InMemoryJobStore()
    app.state.runtime = runtime
    app.state.config_revisions = []
    app.state.active_config_revision_id = None
    app.state.current_run_id = None
    app.state.ui_event_store = UiEventStore(runtime.project_root / "web_backend/runtime/ui_events.jsonl")

    app.include_router(system.router, prefix="/api")
    app.include_router(app_state.router, prefix="/api")
    app.include_router(config.router, prefix="/api")
    app.include_router(scan.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(tools.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    app.include_router(ws.router, prefix="/api")
    return app


app = create_app()
