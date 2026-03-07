from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.schemas import JobAccepted
from app.services.command_runner import launch_background_command

router = APIRouter(tags=["scan"])


@router.post("/scan", response_model=JobAccepted)
def scan(request: Request, payload: dict[str, Any]) -> JobAccepted:
    runtime = request.app.state.runtime
    input_path = payload.get("input_path")
    if not input_path:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "input_path is required"}},
        )
    frames_min = int(payload.get("frames_min", 1))
    with_checksums = bool(payload.get("with_checksums", False))

    cmd = [str(runtime.cli_path), "scan", str(input_path), "--frames-min", str(frames_min)]
    if with_checksums:
        cmd.append("--with-checksums")

    job = request.app.state.job_store.create(
        "scan",
        {
            "input_path": str(input_path),
            "frames_min": frames_min,
            "with_checksums": with_checksums,
            "command": cmd,
        },
    )
    request.app.state.job_store.set_state(job.job_id, "running")
    launch_background_command(
        job_store=request.app.state.job_store,
        job_id=job.job_id,
        command=cmd,
        cwd=runtime.project_root,
    )
    return JobAccepted(job_id=job.job_id, state="running")


@router.get("/scan/quality")
def scan_quality(request: Request) -> dict[str, Any]:
    items = request.app.state.job_store.list()
    scan_job = next((j for j in items if j.job_type == "scan"), None)
    if scan_job is None:
        return {"score": 0.0, "factors": [{"id": "no_scan", "value": 1.0, "label": "No scan run yet"}]}

    result = scan_job.data.get("result", {})
    errors = len(result.get("errors", [])) if isinstance(result, dict) else 0
    warnings = len(result.get("warnings", [])) if isinstance(result, dict) else 0
    score = max(0.0, 1.0 - 0.25 * errors - 0.1 * warnings)
    return {
        "score": round(score, 3),
        "factors": [
            {"id": "errors", "value": errors, "label": "scan errors"},
            {"id": "warnings", "value": warnings, "label": "scan warnings"},
        ],
    }


@router.get("/guardrails")
def guardrails(request: Request) -> dict[str, Any]:
    items = request.app.state.job_store.list()
    scan_job = next((j for j in items if j.job_type == "scan"), None)
    if scan_job is None:
        return {
            "status": "check",
            "checks": [
                {"id": "scan", "status": "check", "label": "Scan ausstehend"},
            ],
        }

    result = scan_job.data.get("result", {})
    errors = result.get("errors", []) if isinstance(result, dict) else []
    checks = [
        {
            "id": "scan_ok",
            "status": "ok" if not errors else "error",
            "label": "Scan erfolgreich" if not errors else "Scan mit Fehlern",
        }
    ]
    status = "ok" if not errors else "error"
    return {"status": status, "checks": checks}
