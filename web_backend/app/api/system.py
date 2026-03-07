from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import HealthResponse, VersionResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/version", response_model=VersionResponse)
def version(request: Request) -> VersionResponse:
    runtime = request.app.state.runtime
    cli = f"found:{runtime.cli_path}" if runtime.cli_path.exists() else f"missing:{runtime.cli_path}"
    runner = f"found:{runtime.runner_path}" if runtime.runner_path.exists() else f"missing:{runtime.runner_path}"
    return VersionResponse(cli=cli, runner=runner)
