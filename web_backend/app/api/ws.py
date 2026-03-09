from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.command_runner import SecurityPolicyError
from app.services.run_inspector import read_run_status
from app.services.run_stream import build_queue_progress_event, tail_run_stream_events
from app.services.time_utils import utc_now_iso

router = APIRouter(tags=["ws"])


@router.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    cursor = 0
    last_status_sent_at = 0.0
    last_terminal_state: str | None = None
    last_queue_fingerprint: str | None = None

    try:
        while True:
            runtime = websocket.app.state.runtime
            try:
                run_dir = runtime.resolve_run_dir(run_id)
            except SecurityPolicyError as exc:
                await websocket.send_json(
                    {
                        "type": "run_stream_error",
                        "run_id": run_id,
                        "ts": utc_now_iso(),
                        "payload": {
                            "code": exc.code,
                            "message": exc.message,
                            "details": exc.details or {},
                        },
                    }
                )
                await asyncio.sleep(2)
                continue
            stream_events, cursor = await asyncio.to_thread(tail_run_stream_events, run_dir, cursor=cursor, max_events=200)

            for event in stream_events:
                if event.get("run_id") in {None, "unknown"}:
                    event["run_id"] = run_id
                await websocket.send_json(event)

            queue_events = _queue_events_for_run(websocket=websocket, run_id=run_id)
            if queue_events is not None:
                fingerprint = str(queue_events.get("payload", {}))
                if fingerprint != last_queue_fingerprint:
                    await websocket.send_json(queue_events)
                    last_queue_fingerprint = fingerprint

            status = await asyncio.to_thread(read_run_status, run_dir)
            state = str(status.get("status", "unknown")).lower()
            now = asyncio.get_event_loop().time()

            if state in {"completed", "failed"} and last_terminal_state != state:
                await websocket.send_json(
                    {
                        "type": "run_end",
                        "run_id": run_id,
                        "status": "ok" if state == "completed" else "error",
                        "ts": utc_now_iso(),
                        "payload": {
                            "state": state,
                            "progress": status.get("progress", 0.0),
                            "current_phase": status.get("current_phase"),
                        },
                    }
                )
                last_terminal_state = state

            # Resync event for frontend robustness if no fresh low-level stream event arrived.
            if not stream_events and now - last_status_sent_at >= 2.0:
                await websocket.send_json(
                    {
                        "type": "run_status",
                        "run_id": run_id,
                        "state": state,
                        "phase": status.get("current_phase"),
                        "pct": _to_pct(status.get("progress", 0.0)),
                        "ts": utc_now_iso(),
                        "payload": status,
                    }
                )
                last_status_sent_at = now

            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/jobs/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            job = websocket.app.state.job_store.get(job_id)
            await websocket.send_json(
                {
                    "type": "job_progress",
                    "job_id": job_id,
                    "state": job.state if job else "unknown",
                    "pid": job.pid if job else None,
                    "exit_code": job.exit_code if job else None,
                    "ts": utc_now_iso(),
                    "data": job.data if job else {},
                }
            )
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/system")
async def ws_system(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            runtime = websocket.app.state.runtime
            await websocket.send_json(
                {
                    "type": "system_heartbeat",
                    "ts": utc_now_iso(),
                    "status": "ok",
                    "payload": {
                        "cli": str(runtime.cli_path),
                        "runner": str(runtime.runner_path),
                    },
                }
            )
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        return


def _to_pct(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v <= 1.0:
        v *= 100.0
    if v < 0.0:
        return 0.0
    if v > 100.0:
        return 100.0
    return round(v, 3)


def _queue_events_for_run(*, websocket: WebSocket, run_id: str) -> dict[str, Any] | None:
    for job in websocket.app.state.job_store.list():
        if job.job_type != "run_queue":
            continue

        queue = job.data.get("queue", [])
        if not isinstance(queue, list):
            continue

        if str(job.data.get("run_id", "")) == run_id:
            return build_queue_progress_event(run_id, queue, current_index=job.data.get("current_index"))

        if any(str(item.get("run_id", "")) == run_id for item in queue if isinstance(item, dict)):
            return build_queue_progress_event(run_id, queue, current_index=job.data.get("current_index"))

    return None
