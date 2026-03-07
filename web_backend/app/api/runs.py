from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.schemas import JobAccepted
from app.services.command_runner import launch_background_command, resolve_python, run_command
from app.services.queue_utils import extract_queue_specs
from app.services.run_inspector import discover_runs, read_run_log_lines, read_run_status

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("")
def list_runs(request: Request, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    resolved_runs_dir = runtime.resolve_runs_dir(runs_dir)
    items = discover_runs(resolved_runs_dir)
    return {"items": items, "total": len(items)}


@router.get("/{run_id}/status")
def run_status(run_id: str, request: Request, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    run_dir = runtime.resolve_run_dir(run_id, runs_dir)
    status = read_run_status(run_dir)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": status.get("status", "unknown"),
        "current_phase": status.get("current_phase"),
        "progress": status.get("progress", 0.0),
        "phases": status.get("phases", []),
        "events": status.get("events", []),
    }


@router.get("/{run_id}/logs")
def run_logs(run_id: str, request: Request, tail: int = 200, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    run_dir = runtime.resolve_run_dir(run_id, runs_dir)
    lines = read_run_log_lines(run_dir, tail=max(0, int(tail)))
    return {"lines": lines, "cursor": None}


@router.get("/{run_id}/artifacts")
def run_artifacts(run_id: str, request: Request, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    run_dir = runtime.resolve_run_dir(run_id, runs_dir)
    result = run_command(
        [str(runtime.cli_path), "list-artifacts", str(run_dir)],
        cwd=runtime.project_root,
    )
    if result.exit_code != 0 or not isinstance(result.parsed_json, dict):
        raise _http_502("list-artifacts failed", result)
    return {"items": result.parsed_json.get("artifacts", [])}


@router.post("/start")
def run_start(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    runtime = request.app.state.runtime
    queue_specs = extract_queue_specs(payload)
    if queue_specs:
        return _start_queue_run(request, payload, queue_specs)

    input_dir = payload.get("input_dir")
    if not input_dir:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "input_dir is required"}},
        )

    cmd, stdin_text, resolved_runs_dir = _build_run_command(
        runtime=runtime,
        payload=payload,
        input_dir=str(input_dir),
        run_id_override=payload.get("run_id"),
    )
    job = request.app.state.job_store.create(
        "run_start",
        {
            "input_dir": str(input_dir),
            "runs_dir": str(resolved_runs_dir),
            "run_id": payload.get("run_id"),
            "command": cmd,
        },
    )
    request.app.state.job_store.set_state(job.job_id, "running")
    launch_background_command(
        job_store=request.app.state.job_store,
        job_id=job.job_id,
        command=cmd,
        cwd=runtime.project_root,
        stdin_text=stdin_text,
    )
    return {"run_id": payload.get("run_id") or "pending", "job_id": job.job_id}


@router.post("/{run_id}/resume")
def run_resume(run_id: str, request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    runtime = request.app.state.runtime
    from_phase = str(payload.get("from_phase", "PCC")).upper()
    run_dir = payload.get("run_dir")
    resolved_run_dir = Path(run_dir).expanduser() if run_dir else runtime.resolve_run_dir(run_id, payload.get("runs_dir"))
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
        {"run_id": run_id, "run_dir": str(resolved_run_dir), "from_phase": from_phase, "command": cmd},
    )
    request.app.state.job_store.set_state(job.job_id, "running")
    launch_background_command(
        job_store=request.app.state.job_store,
        job_id=job.job_id,
        command=cmd,
        cwd=runtime.project_root,
    )
    return {"run_id": run_id, "job_id": job.job_id}


@router.post("/{run_id}/stop")
def run_stop(run_id: str, request: Request, runs_dir: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    run_dir = str(runtime.resolve_run_dir(run_id, runs_dir))
    cancelled = False
    for job in request.app.state.job_store.list():
        if job.state != "running":
            continue
        if job.data.get("run_dir") == run_dir or str(job.data.get("run_id", "")) == run_id:
            request.app.state.job_store.cancel(job.job_id)
            cancelled = True
    return {"ok": cancelled}


@router.post("/{run_id}/set-current")
def run_set_current(run_id: str, request: Request) -> dict[str, Any]:
    request.app.state.current_run_id = run_id
    return {"ok": True, "run_id": run_id}


@router.post("/{run_id}/stats", response_model=JobAccepted)
def run_stats(run_id: str, request: Request, payload: dict[str, Any] | None = None) -> JobAccepted:
    runtime = request.app.state.runtime
    body = payload or {}
    run_dir = body.get("run_dir")
    resolved_run_dir = Path(run_dir).expanduser() if run_dir else runtime.resolve_run_dir(run_id, body.get("runs_dir"))
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


def _start_queue_run(request: Request, payload: dict[str, Any], queue_specs: list[dict[str, Any]]) -> dict[str, Any]:
    runtime = request.app.state.runtime
    queue_summary = []
    for i, spec in enumerate(queue_specs):
        queue_summary.append(
            {
                "index": i,
                "filter": spec.get("filter") or spec.get("name"),
                "input_dir": str(spec.get("input_dir")),
                "run_id": spec.get("run_id"),
                "state": "pending",
                "exit_code": None,
            }
        )

    job = request.app.state.job_store.create(
        "run_queue",
        {
            "run_id": payload.get("run_id"),
            "runs_dir": str(request.app.state.runtime.resolve_runs_dir(payload.get("runs_dir"))),
            "queue": queue_summary,
        },
    )
    request.app.state.job_store.set_state(job.job_id, "running")

    def _worker() -> None:
        queue_state = list(queue_summary)
        base_run_id = payload.get("run_id")
        try:
            for i, spec in enumerate(queue_specs):
                job_snapshot = request.app.state.job_store.get(job.job_id)
                if job_snapshot is not None and job_snapshot.state == "cancelled":
                    return

                suffix = spec.get("filter") or spec.get("name") or f"q{i + 1:02d}"
                run_id_override = spec.get("run_id") or (f"{base_run_id}_{suffix}" if base_run_id else None)
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
                queue_state[i]["state"] = "running"
                queue_state[i]["run_id"] = run_id_override
                request.app.state.job_store.merge_data(job.job_id, {"queue": queue_state, "current_index": i, "command": cmd})

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(runtime.project_root),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
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
                    return

            request.app.state.job_store.set_state(job.job_id, "ok")
        except Exception as exc:
            request.app.state.job_store.merge_data(job.job_id, {"error": str(exc), "queue": queue_state})
            if request.app.state.job_store.get(job.job_id) and request.app.state.job_store.get(job.job_id).state != "cancelled":
                request.app.state.job_store.set_state(job.job_id, "error")
        finally:
            request.app.state.job_store.clear_process(job.job_id)

    thread = threading.Thread(target=_worker, name=f"run-queue-{job.job_id}", daemon=True)
    thread.start()
    return {"run_id": payload.get("run_id") or "queue", "job_id": job.job_id}


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
    cmd = [
        str(runtime.runner_path),
        "run",
        "--config",
        str(config_path if config_path else runtime.default_config_path),
        "--input-dir",
        str(input_dir),
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
