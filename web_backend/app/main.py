from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import app_state, config, jobs, runs, scan, system, tools, ws
from app.error_handlers import register_error_handlers
from app.services.command_runner import BackendRuntime, CommandPolicy
from app.services.process_manager import InMemoryJobStore
from app.services.ui_events import UiEventStore


def create_app() -> FastAPI:
    app = FastAPI(title="tile_compile GUI2 backend", version="0.1.0")
    register_error_handlers(app)
    runtime = BackendRuntime.autodetect()
    app.state.job_store = InMemoryJobStore()
    app.state.runtime = runtime
    app.state.command_policy = CommandPolicy(runtime)
    app.state.config_revisions = []
    app.state.active_config_revision_id = None
    app.state.current_run_id = None
    app.state.last_scan_input_path = ""
    app.state.ui_event_store = UiEventStore(runtime.project_root / "web_backend/runtime/ui_events.jsonl")

    app.include_router(system.router, prefix="/api")
    app.include_router(app_state.router, prefix="/api")
    app.include_router(config.router, prefix="/api")
    app.include_router(scan.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(tools.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    app.include_router(ws.router, prefix="/api")

    frontend_dir = runtime.project_root / "web_frontend"
    if frontend_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def root_redirect() -> RedirectResponse:
            return RedirectResponse(url="/ui/")

    return app


app = create_app()
