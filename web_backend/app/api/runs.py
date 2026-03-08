from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.schemas import JobAccepted
from app.services.command_runner import SecurityPolicyError, launch_background_command, resolve_python, run_command
from app.services.config_revisions import create_revision, get_revision, restore_revision
from app.services.guardrails import compute_guardrails, has_blocking_guardrail
from app.services.http_errors import http_from_security_error
from app.services.queue_utils import extract_queue_specs
from app.services.run_inspector import discover_runs, read_run_log_lines, read_run_status
from app.services.time_utils import utc_now
from app.services.ui_events import record_ui_event

router = APIRouter(prefix="/runs", tags=["runs"])
_RUN_ID_TS_RE = re.compile(r"_(\d{8}_\d{6})$")
_RUN_ID_INVALID_RE = re.compile(r"[^A-Za-z0-9._-]+")


@router.get("")
def list_runs(request: Request, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    try:
        resolved_runs_dir = runtime.resolve_runs_dir(runs_dir)
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    items = discover_runs(resolved_runs_dir)
    return {"items": items, "total": len(items)}


@router.get("/{run_id}/status")
def run_status(run_id: str, request: Request, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    try:
        run_dir = runtime.resolve_run_dir(run_id, runs_dir)
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    status = read_run_status(run_dir)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": status.get("status", "unknown"),
        "color_mode": status.get("color_mode", "UNKNOWN"),
        "current_phase": status.get("current_phase"),
        "progress": status.get("progress", 0.0),
        "phases": status.get("phases", []),
        "events": status.get("events", []),
    }


@router.get("/{run_id}/logs")
def run_logs(run_id: str, request: Request, tail: int = 200, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    try:
        run_dir = runtime.resolve_run_dir(run_id, runs_dir)
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    lines = read_run_log_lines(run_dir, tail=max(0, int(tail)))
    return {"lines": lines, "cursor": None}


@router.get("/{run_id}/artifacts")
def run_artifacts(run_id: str, request: Request, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    try:
        run_dir = runtime.resolve_run_dir(run_id, runs_dir)
        result = run_command(
            [str(runtime.cli_path), "list-artifacts", str(run_dir)],
            cwd=runtime.project_root,
            command_policy=request.app.state.command_policy,
        )
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    if result.exit_code != 0 or not isinstance(result.parsed_json, dict):
        raise _http_502("list-artifacts failed", result)
    return {"items": result.parsed_json.get("artifacts", [])}


@router.get("/{run_id}/artifacts/view")
def run_artifact_view(run_id: str, path: str, request: Request, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    raw_path = str(path or "").strip()
    if not raw_path:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "path is required"}},
        )
    try:
        run_dir = runtime.resolve_run_dir(run_id, runs_dir)
        run_dir_resolved = run_dir.resolve(strict=False)
        candidate = (run_dir / raw_path).resolve(strict=False) if not Path(raw_path).is_absolute() else Path(raw_path).expanduser().resolve(strict=False)
        if run_dir_resolved not in candidate.parents and candidate != run_dir_resolved:
            raise HTTPException(
                status_code=422,
                detail={"error": {"code": "ARTIFACT_PATH_INVALID", "message": "artifact path must stay inside run directory"}},
            )
        checked = runtime.ensure_path_allowed(candidate, must_exist=True, label="artifact_path")
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    if not checked.is_file():
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "ARTIFACT_NOT_FILE", "message": "artifact path is not a file"}},
        )
    try:
        text = checked.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "ARTIFACT_READ_FAILED", "message": str(exc)}}
        ) from exc
    text = text[:400000]
    parsed_json = None
    try:
        parsed_json = json.loads(text)
    except Exception:
        parsed_json = None
    return {
        "path": str(checked),
        "filename": checked.name,
        "is_json": parsed_json is not None,
        "json": parsed_json,
        "text": text,
    }


@router.post("/start", status_code=202)
def run_start(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    runtime = request.app.state.runtime
    guardrails = compute_guardrails(request.app.state.job_store)
    if has_blocking_guardrail(guardrails):
        record_ui_event(
            request,
            event="run.start.blocked",
            source="runs.run_start",
            payload={"reason": "guardrail_error", "guardrails": guardrails},
        )
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "code": "GUARDRAIL_BLOCKED",
                    "message": "run start blocked by guardrails",
                    "details": guardrails,
                }
            },
        )

    run_name_raw = str(payload.get("run_name", "")).strip()
    run_name = _sanitize_run_token(run_name_raw) if run_name_raw else ""
    if run_name_raw and not run_name:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "RUN_NAME_INVALID", "message": "run_name is invalid after sanitization"}},
        )

    run_id_raw = str(payload.get("run_id", "")).strip()
    if run_name and not run_id_raw:
        run_id_raw = run_name
    run_id_override = _normalize_run_id(run_id_raw) if run_id_raw else None
    if not run_id_override:
        input_hint = _sanitize_run_token(Path(str(payload.get("input_dir", ""))).expanduser().name)
        if input_hint:
            run_id_override = _normalize_run_id(run_name or input_hint)

    try:
        config_target = runtime.ensure_path_allowed(
            Path(str(payload.get("config_path", runtime.default_config_path))).expanduser(),
            label="config_path",
        )
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    config_yaml = _resolve_config_yaml(runtime=runtime, payload={**payload, "config_path": str(config_target)})
    revision = create_revision(
        request.app,
        path=config_target,
        yaml_text=config_yaml,
        source="run_start",
        run_id=run_id_override,
    )
    payload_with_revision = dict(payload)
    if run_id_override:
        payload_with_revision["run_id"] = run_id_override
    payload_with_revision["config_revision_id"] = revision["revision_id"]

    queue_specs = extract_queue_specs(payload_with_revision)
    if queue_specs:
        if not payload_with_revision.get("run_id"):
            payload_with_revision["run_id"] = _normalize_run_id(run_name or "queue")
        for spec in queue_specs:
            input_dir_spec = spec.get("input_dir")
            if input_dir_spec:
                try:
                    checked = runtime.resolve_input_path(
                        str(input_dir_spec),
                        must_exist=True,
                        label="queue_input_dir",
                    )
                except SecurityPolicyError as exc:
                    raise http_from_security_error(exc) from exc
                spec["input_dir"] = str(checked)
            run_label_raw = str(spec.get("run_id", "")).strip()
            if run_label_raw:
                normalized = _normalize_run_id(run_label_raw)
                if not normalized:
                    raise HTTPException(
                        status_code=422,
                        detail={"error": {"code": "RUN_NAME_INVALID", "message": "queue run_label is invalid"}},
                    )
                spec["run_id"] = normalized
        result = _start_queue_run(request, payload_with_revision, queue_specs)
        record_ui_event(
            request,
            event="run.start.queue",
            source="runs.run_start",
            run_id=str(result.get("run_id")),
            job_id=str(result.get("job_id")),
            payload={"revision_id": revision["revision_id"], "queue_size": len(queue_specs)},
        )
        request.app.state.current_run_id = str(result.get("run_id") or "")
        return result

    input_dir = payload.get("input_dir")
    if not input_dir:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "input_dir is required"}},
        )
    if not run_id_override:
        input_name = _sanitize_run_token(Path(str(input_dir)).expanduser().name)
        run_id_override = _normalize_run_id(run_name or input_name or "run")
        payload_with_revision["run_id"] = run_id_override

    try:
        input_dir_checked = runtime.resolve_input_path(str(input_dir), must_exist=True, label="input_dir")
        cmd, stdin_text, resolved_runs_dir = _build_run_command(
            runtime=runtime,
            payload=payload_with_revision,
            input_dir=str(input_dir_checked),
            run_id_override=run_id_override,
        )
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    resolved_runs_dir.mkdir(parents=True, exist_ok=True)
    job = request.app.state.job_store.create(
        "run_start",
        {
            "input_dir": str(input_dir_checked),
            "runs_dir": str(resolved_runs_dir),
            "run_id": run_id_override,
            "command": cmd,
            "config_revision_id": revision["revision_id"],
        },
    )
    request.app.state.job_store.set_state(job.job_id, "running")
    launch_background_command(
        job_store=request.app.state.job_store,
        job_id=job.job_id,
        command=cmd,
        cwd=runtime.project_root,
        stdin_text=stdin_text,
        command_policy=request.app.state.command_policy,
    )
    record_ui_event(
        request,
        event="run.start",
        source="runs.run_start",
        run_id=run_id_override,
        job_id=job.job_id,
        payload={"input_dir": str(input_dir_checked), "runs_dir": str(resolved_runs_dir), "revision_id": revision["revision_id"]},
    )
    request.app.state.current_run_id = run_id_override
    return {"run_id": run_id_override, "job_id": job.job_id}


@router.post("/{run_id}/resume", status_code=202)
def run_resume(run_id: str, request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    runtime = request.app.state.runtime
    from_phase_raw = str(payload.get("from_phase", "")).strip()
    if not from_phase_raw:
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "RESUME_PHASE_REQUIRED", "message": "from_phase is required for resume"}},
        )
    from_phase = from_phase_raw.upper()
    revision_id = str(payload.get("config_revision_id", "")).strip()
    if not revision_id:
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "REVISION_REQUIRED", "message": "config_revision_id is required for resume"}},
        )
    revision = get_revision(request.app, revision_id)
    if revision is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": f"revision '{revision_id}' not found"}},
        )
    try:
        restore_revision(request.app, revision_id)
        run_dir = payload.get("run_dir")
        resolved_run_dir = (
            runtime.ensure_path_allowed(Path(run_dir).expanduser(), must_exist=False, label="run_dir")
            if run_dir
            else runtime.ensure_path_allowed(runtime.resolve_run_dir(run_id, payload.get("runs_dir")), must_exist=False, label="run_dir")
        )
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    cmd = [
        str(runtime.runner_path),
        "resume",
        "--run-dir",
        str(resolved_run_dir),
        "--from-phase",
        from_phase,
    ]
    job = request.app.state.job_store.create(
        "run_resume",
        {
            "run_id": run_id,
            "run_dir": str(resolved_run_dir),
            "from_phase": from_phase,
            "config_revision_id": revision_id,
            "filter_context": payload.get("filter_context"),
            "command": cmd,
        },
    )
    request.app.state.job_store.set_state(job.job_id, "running")
    launch_background_command(
        job_store=request.app.state.job_store,
        job_id=job.job_id,
        command=cmd,
        cwd=runtime.project_root,
        command_policy=request.app.state.command_policy,
    )
    record_ui_event(
        request,
        event="run.resume",
        source="runs.run_resume",
        run_id=run_id,
        job_id=job.job_id,
        payload={"from_phase": from_phase, "config_revision_id": revision_id, "filter_context": payload.get("filter_context")},
    )
    return {"run_id": run_id, "job_id": job.job_id}


@router.post("/{run_id}/stop")
def run_stop(run_id: str, request: Request, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    try:
        run_dir = str(runtime.resolve_run_dir(run_id, runs_dir))
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    cancelled = False
    cancelled_jobs: list[str] = []
    for job in request.app.state.job_store.list():
        if job.state != "running":
            continue
        job_run_dir = str(job.data.get("run_dir", ""))
        job_run_id = str(job.data.get("run_id", ""))
        job_runs_dir = str(job.data.get("runs_dir", ""))
        is_pending_target = run_id.strip().lower() == "pending"
        matches_run = (
            job_run_dir == run_dir
            or job_run_id == run_id
            or (is_pending_target and job_runs_dir and run_dir.startswith(job_runs_dir))
        )
        if matches_run:
            request.app.state.job_store.cancel(job.job_id)
            cancelled = True
            cancelled_jobs.append(job.job_id)
            record_ui_event(
                request,
                event="run.stop",
                source="runs.run_stop",
                run_id=run_id,
                job_id=job.job_id,
                payload={"run_dir": run_dir},
            )
    if not cancelled and run_id.strip().lower() == "pending":
        pending_candidates = [
            job
            for job in request.app.state.job_store.list()
            if job.state == "running" and str(job.job_type).startswith("run_")
        ]
        if len(pending_candidates) == 1:
            job = pending_candidates[0]
            request.app.state.job_store.cancel(job.job_id)
            cancelled = True
            cancelled_jobs.append(job.job_id)
            record_ui_event(
                request,
                event="run.stop",
                source="runs.run_stop",
                run_id=run_id,
                job_id=job.job_id,
                payload={"run_dir": run_dir, "fallback": "single_running_run"},
            )
    killed_pids: list[int] = []
    if not cancelled:
        killed_pids = _terminate_orphan_runner_processes(runtime=runtime, run_id=run_id, run_dir=run_dir)
        if killed_pids:
            cancelled = True
            record_ui_event(
                request,
                event="run.stop.orphan",
                source="runs.run_stop",
                run_id=run_id,
                payload={"run_dir": run_dir, "killed_pids": killed_pids},
            )
    return {"ok": cancelled, "cancelled_jobs": cancelled_jobs, "killed_pids": killed_pids}


@router.post("/{run_id}/set-current")
def run_set_current(run_id: str, request: Request) -> dict[str, Any]:
    request.app.state.current_run_id = run_id
    record_ui_event(
        request,
        event="run.set_current",
        source="runs.run_set_current",
        run_id=run_id,
        payload={"run_id": run_id},
    )
    return {"ok": True, "run_id": run_id}


@router.post("/{run_id}/delete")
def run_delete(run_id: str, request: Request, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    try:
        run_dir = runtime.resolve_run_dir(run_id, runs_dir)
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": f"run '{run_id}' not found"}},
        )
    for job in request.app.state.job_store.list():
        if job.state != "running":
            continue
        if str(job.data.get("run_id", "")) == run_id or str(job.data.get("run_dir", "")) == str(run_dir):
            raise HTTPException(
                status_code=409,
                detail={"error": {"code": "RUN_ACTIVE", "message": "cannot delete active run"}},
            )
    shutil.rmtree(run_dir)
    if str(getattr(request.app.state, "current_run_id", "") or "") == run_id:
        request.app.state.current_run_id = ""
    record_ui_event(
        request,
        event="run.delete",
        source="runs.run_delete",
        run_id=run_id,
        payload={"run_dir": str(run_dir)},
    )
    return {"ok": True, "run_id": run_id}


@router.post("/{run_id}/stats", response_model=JobAccepted, status_code=202)
def run_stats(run_id: str, request: Request, payload: dict[str, Any] | None = None) -> JobAccepted:
    runtime = request.app.state.runtime
    body = payload or {}
    run_dir = body.get("run_dir")
    try:
        resolved_run_dir = (
            runtime.ensure_path_allowed(Path(run_dir).expanduser(), label="run_dir")
            if run_dir
            else runtime.resolve_run_dir(run_id, body.get("runs_dir"))
        )
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc
    python_bin = resolve_python()
    cmd = [python_bin, str(runtime.stats_script), str(resolved_run_dir)]
    job = request.app.state.job_store.create(
        "stats",
        {"run_id": run_id, "run_dir": str(resolved_run_dir), "command": cmd},
    )
    request.app.state.job_store.set_state(job.job_id, "running")
    launch_background_command(
        job_store=request.app.state.job_store,
        job_id=job.job_id,
        command=cmd,
        cwd=runtime.project_root,
        command_policy=request.app.state.command_policy,
    )
    record_ui_event(
        request,
        event="run.stats",
        source="runs.run_stats",
        run_id=run_id,
        job_id=job.job_id,
        payload={"run_dir": str(resolved_run_dir)},
    )
    return JobAccepted(job_id=job.job_id, state="running")


@router.get("/{run_id}/stats/status")
def run_stats_status(run_id: str, request: Request) -> dict[str, Any]:
    for job in request.app.state.job_store.list():
        if job.job_type != "stats":
            continue
        if job.data.get("run_id") != run_id:
            continue
        report_path = Path(job.data.get("run_dir", "")) / "artifacts" / "report.html"
        return {
            "state": job.state,
            "output_dir": str(report_path.parent),
            "report_path": str(report_path),
            "job_id": job.job_id,
        }
    return {"state": "unknown", "output_dir": "", "report_path": ""}


@router.post("/{run_id}/config-revisions/{revision_id}/restore")
def run_revision_restore(run_id: str, revision_id: str, request: Request) -> dict[str, Any]:
    revision = get_revision(request.app, revision_id)
    if revision is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": f"revision '{revision_id}' not found"}},
        )
    restore_revision(request.app, revision_id)
    record_ui_event(
        request,
        event="run.revision.restore",
        source="runs.run_revision_restore",
        run_id=run_id,
        payload={"revision_id": revision_id, "path": revision.get("path")},
    )
    return {"ok": True, "run_id": run_id, "active_revision_id": revision_id}


def _start_queue_run(request: Request, payload: dict[str, Any], queue_specs: list[dict[str, Any]]) -> dict[str, Any]:
    runtime = request.app.state.runtime
    queue_run_id = str(payload.get("run_id") or "").strip()
    if not queue_run_id:
        queue_run_id = _normalize_run_id("queue") or "queue"
    queue_summary = []
    for i, spec in enumerate(queue_specs):
        queue_summary.append(
            {
                "index": i,
                "filter": spec.get("filter") or spec.get("name"),
                "input_dir": str(spec.get("input_dir")),
                "run_id": spec.get("run_id") or "",
                "state": "pending",
                "exit_code": None,
            }
        )

    job = request.app.state.job_store.create(
        "run_queue",
        {
            "run_id": queue_run_id,
            "runs_dir": str(request.app.state.runtime.resolve_runs_dir(payload.get("runs_dir"))),
            "queue": queue_summary,
            "config_revision_id": payload.get("config_revision_id"),
        },
    )
    request.app.state.job_store.set_state(job.job_id, "running")
    record_ui_event(
        request,
        event="run.queue.start",
        source="runs._start_queue_run",
        run_id=queue_run_id,
        job_id=job.job_id,
        payload={"queue": queue_summary},
    )

    def _worker() -> None:
        queue_state = list(queue_summary)
        base_run_id = str(payload.get("run_id") or "").strip()
        base_no_ts = _strip_timestamp_suffix(base_run_id) if base_run_id else "queue"
        try:
            for i, spec in enumerate(queue_specs):
                job_snapshot = request.app.state.job_store.get(job.job_id)
                if job_snapshot is not None and job_snapshot.state == "cancelled":
                    return

                suffix = spec.get("filter") or spec.get("name") or f"q{i + 1:02d}"
                raw_item_run_id = str(spec.get("run_id") or "").strip()
                if raw_item_run_id:
                    run_id_override = _normalize_run_id(raw_item_run_id)
                else:
                    run_id_override = _normalize_run_id(f"{base_no_ts}_{suffix}")
                input_dir = spec.get("input_dir")
                if not input_dir:
                    queue_state[i]["state"] = "error"
                    queue_state[i]["exit_code"] = 2
                    request.app.state.job_store.merge_data(job.job_id, {"queue": queue_state})
                    request.app.state.job_store.set_state(job.job_id, "error")
                    return

                cmd, stdin_text, _ = _build_run_command(
                    runtime=runtime,
                    payload={**payload, **spec},
                    input_dir=str(input_dir),
                    run_id_override=run_id_override,
                )
                request.app.state.command_policy.validate(cmd)
                queue_state[i]["state"] = "running"
                queue_state[i]["run_id"] = run_id_override
                request.app.state.job_store.merge_data(job.job_id, {"queue": queue_state, "current_index": i, "command": cmd})
                record_ui_event(
                    request,
                    event="run.queue.item.start",
                    source="runs._start_queue_run",
                    run_id=str(run_id_override or payload.get("run_id") or "queue"),
                    job_id=job.job_id,
                    payload={"index": i, "filter": suffix, "input_dir": str(input_dir)},
                )

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(runtime.project_root),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                request.app.state.job_store.set_process(job.job_id, proc)
                stdout, stderr = proc.communicate(stdin_text)
                request.app.state.job_store.clear_process(job.job_id)
                request.app.state.job_store.set_exit_code(job.job_id, int(proc.returncode))

                queue_state[i]["state"] = "ok" if proc.returncode == 0 else "error"
                queue_state[i]["exit_code"] = int(proc.returncode)
                request.app.state.job_store.merge_data(
                    job.job_id,
                    {
                        "queue": queue_state,
                        "last_stdout": stdout,
                        "last_stderr": stderr,
                    },
                )
                if proc.returncode != 0:
                    request.app.state.job_store.set_state(job.job_id, "error")
                    record_ui_event(
                        request,
                        event="run.queue.item.error",
                        source="runs._start_queue_run",
                        run_id=str(run_id_override or payload.get("run_id") or "queue"),
                        job_id=job.job_id,
                        payload={"index": i, "filter": suffix, "exit_code": int(proc.returncode)},
                    )
                    return

            request.app.state.job_store.set_state(job.job_id, "ok")
            record_ui_event(
                request,
                event="run.queue.end",
                source="runs._start_queue_run",
                run_id=queue_run_id,
                job_id=job.job_id,
                payload={"status": "ok"},
            )
        except Exception as exc:
            request.app.state.job_store.merge_data(job.job_id, {"error": str(exc), "queue": queue_state})
            if request.app.state.job_store.get(job.job_id) and request.app.state.job_store.get(job.job_id).state != "cancelled":
                request.app.state.job_store.set_state(job.job_id, "error")
            record_ui_event(
                request,
                event="run.queue.error",
                source="runs._start_queue_run",
                run_id=queue_run_id,
                job_id=job.job_id,
                payload={"error": str(exc)},
            )
        finally:
            request.app.state.job_store.clear_process(job.job_id)

    thread = threading.Thread(target=_worker, name=f"run-queue-{job.job_id}", daemon=True)
    thread.start()
    return {"run_id": queue_run_id, "job_id": job.job_id}


def _build_run_command(
    *,
    runtime: Any,
    payload: dict[str, Any],
    input_dir: str,
    run_id_override: str | None,
) -> tuple[list[str], str | None, Path]:
    config_path = payload.get("config_path")
    config_yaml = payload.get("config_yaml")
    runs_dir = payload.get("runs_dir")
    project_root = payload.get("project_root")
    dry_run = bool(payload.get("dry_run", False))
    max_frames = int(payload.get("max_frames", 0))
    max_tiles = int(payload.get("max_tiles", 0))

    resolved_runs_dir = runtime.resolve_runs_dir(runs_dir)
    checked_input_dir = runtime.resolve_input_path(Path(input_dir).expanduser(), must_exist=True, label="input_dir")
    checked_config_path = runtime.ensure_path_allowed(
        Path(str(config_path if config_path else runtime.default_config_path)).expanduser(),
        label="config_path",
    )
    cmd = [
        str(runtime.runner_path),
        "run",
        "--config",
        str(checked_config_path),
        "--input-dir",
        str(checked_input_dir),
        "--runs-dir",
        str(resolved_runs_dir),
    ]

    stdin_text: str | None = None
    if config_yaml:
        cmd[cmd.index("--config") + 1] = "-"
        cmd.append("--stdin")
        stdin_text = str(config_yaml)
    if project_root:
        cmd.extend(["--project-root", str(project_root)])
    if run_id_override:
        cmd.extend(["--run-id", str(run_id_override)])
    if max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])
    if max_tiles > 0:
        cmd.extend(["--max-tiles", str(max_tiles)])
    if dry_run:
        cmd.append("--dry-run")
    return cmd, stdin_text, resolved_runs_dir


def _resolve_config_yaml(*, runtime: Any, payload: dict[str, Any]) -> str:
    if payload.get("config_yaml"):
        return str(payload.get("config_yaml"))

    config_path_raw = payload.get("config_path")
    config_path = runtime.ensure_path_allowed(
        Path(str(config_path_raw)).expanduser() if config_path_raw else runtime.default_config_path,
        label="config_path",
    )
    try:
        return config_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _http_502(message: str, result: Any) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail={
            "error": {
                "code": "BACKEND_COMMAND_FAILED",
                "message": message,
                "details": {
                    "exit_code": getattr(result, "exit_code", None),
                    "stdout": getattr(result, "stdout", ""),
                    "stderr": getattr(result, "stderr", ""),
                },
            }
        },
    )


def _timestamp_suffix() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def _sanitize_run_token(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    value = value.replace("\\", "_").replace("/", "_").replace(" ", "_")
    value = _RUN_ID_INVALID_RE.sub("_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._-")


def _normalize_run_id(raw: str) -> str:
    token = _sanitize_run_token(raw)
    if not token:
        return ""
    if _RUN_ID_TS_RE.search(token):
        return token
    return f"{token}_{_timestamp_suffix()}"


def _strip_timestamp_suffix(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not value:
        return ""
    return _RUN_ID_TS_RE.sub("", value)


def _terminate_orphan_runner_processes(*, runtime: Any, run_id: str, run_dir: str) -> list[int]:
    if os.name != "posix":
        return []
    runner_name = Path(str(runtime.runner_path)).name
    killed: list[int] = []
    for pid, argv in _iter_process_cmdlines():
        if pid == os.getpid():
            continue
        if not _is_runner_command(argv, runner_name):
            continue
        joined = " ".join(argv)
        if run_id not in joined and run_dir not in joined:
            continue
        if _terminate_pid_group(pid):
            killed.append(pid)
    return killed


def _iter_process_cmdlines() -> list[tuple[int, list[str]]]:
    proc_root = Path("/proc")
    out: list[tuple[int, list[str]]] = []
    if not proc_root.exists():
        return out
    for proc_dir in proc_root.iterdir():
        if not proc_dir.is_dir() or not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        cmdline_path = proc_dir / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            continue
        parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        if not parts:
            continue
        out.append((pid, parts))
    return out


def _is_runner_command(argv: list[str], runner_name: str) -> bool:
    if not argv:
        return False
    exe_name = Path(str(argv[0])).name
    if exe_name != runner_name and "tile_compile_runner" not in exe_name:
        return False
    return any(arg in {"run", "resume"} for arg in argv[1:])


def _terminate_pid_group(pid: int) -> bool:
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return False
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            return False
    return True


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
