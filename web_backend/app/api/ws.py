from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.run_inspector import read_run_status

router = APIRouter(tags=["ws"])


@router.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            runtime = websocket.app.state.runtime
            run_dir = runtime.resolve_run_dir(run_id)
            status = await asyncio.to_thread(
                read_run_status,
                run_dir,
            )
            await websocket.send_json(
                {
                    "type": "run_status",
                    "run_id": run_id,
                    "state": status.get("status", "unknown"),
                    "phase": status.get("current_phase"),
                    "pct": status.get("progress", 0),
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "payload": status,
                }
            )
            await asyncio.sleep(2)
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
                    "ts": datetime.utcnow().isoformat() + "Z",
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
            await websocket.send_json(
                {
                    "type": "system_heartbeat",
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "status": "ok",
                }
            )
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        return
