from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.services.ui_events import record_ui_event

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
def jobs(request: Request) -> dict:
    items = [_job_to_dict(j) for j in request.app.state.job_store.list()]
    return {"items": items}


@router.get("/{job_id}")
def job(job_id: str, request: Request) -> dict:
    j = request.app.state.job_store.get(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": f"job '{job_id}' not found"}})
    return _job_to_dict(j)


@router.post("/{job_id}/cancel")
def cancel(job_id: str, request: Request) -> dict:
    j = request.app.state.job_store.cancel(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": f"job '{job_id}' not found"}})
    record_ui_event(
        request,
        event="job.cancel",
        source="jobs.cancel",
        run_id=str(j.data.get("run_id")) if isinstance(j.data, dict) and j.data.get("run_id") else None,
        job_id=job_id,
        payload={"job_type": j.job_type},
    )
    return {"ok": True}


def _job_to_dict(job_obj: object) -> dict:
    return {
        "job_id": job_obj.job_id,
        "type": job_obj.job_type,
        "state": job_obj.state,
        "pid": job_obj.pid,
        "exit_code": job_obj.exit_code,
        "created_at": job_obj.created_at.isoformat() + "Z",
        "updated_at": job_obj.updated_at.isoformat() + "Z",
        "data": job_obj.data,
    }
