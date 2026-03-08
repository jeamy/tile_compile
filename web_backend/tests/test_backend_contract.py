from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

if os.getenv("WEB_BACKEND_ENABLE_HTTP_TESTS", "0") != "1":
    pytest.skip("HTTP API integration tests disabled in this environment", allow_module_level=True)

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from app.services.config_revisions import create_revision  # noqa: E402


def test_ui_events_replay_after_mutating_call() -> None:
    app = create_app()
    client = TestClient(app)

    before = client.get("/api/app/ui-events")
    assert before.status_code == 200
    assert before.json()["latest_seq"] == 0

    resp = client.post("/api/scan", json={"input_path": "/tmp/input", "frames_min": 3})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    after = client.get("/api/app/ui-events")
    assert after.status_code == 200
    items = after.json()["items"]
    assert items
    assert items[-1]["event"] == "scan.start"
    assert items[-1]["job_id"] == job_id


def test_run_start_blocked_by_guardrail_error() -> None:
    app = create_app()
    client = TestClient(app)

    scan_job = app.state.job_store.create("scan", {"result": {"errors": [{"message": "bad"}], "warnings": []}})
    app.state.job_store.set_state(scan_job.job_id, "ok")

    resp = client.post("/api/runs/start", json={"input_dir": "/tmp/input"})
    assert resp.status_code == 409
    body = resp.json()
    assert body["detail"]["error"]["code"] == "GUARDRAIL_BLOCKED"


def test_resume_requires_phase_and_revision(tmp_path: Path) -> None:
    app = create_app()
    app.state.runtime.runs_dir = tmp_path
    client = TestClient(app)

    missing = client.post("/api/runs/r1/resume", json={})
    assert missing.status_code == 409
    assert missing.json()["detail"]["error"]["code"] == "RESUME_PHASE_REQUIRED"

    missing_rev = client.post("/api/runs/r1/resume", json={"from_phase": "PCC", "config_revision_id": "cfg_missing"})
    assert missing_rev.status_code == 404

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("pcc:\n  sigma_clip: 2.5\n", encoding="utf-8")
    rev = create_revision(app, path=cfg_path, yaml_text=cfg_path.read_text(encoding="utf-8"), source="test")

    ok = client.post(
        "/api/runs/r1/resume",
        json={"from_phase": "PCC", "config_revision_id": rev["revision_id"], "run_dir": str(tmp_path / "r1")},
    )
    assert ok.status_code == 202
    assert "job_id" in ok.json()


def test_ws_run_emits_phase_events(tmp_path: Path) -> None:
    app = create_app()
    app.state.runtime.runs_dir = tmp_path
    client = TestClient(app)

    run_dir = tmp_path / "r42"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    event_file = logs / "run_events.jsonl"
    event_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "phase_start", "run_id": "r42", "phase": "REGISTRATION", "ts": "2026-03-08T00:00:00Z"}),
                json.dumps(
                    {
                        "type": "phase_progress",
                        "run_id": "r42",
                        "phase": "REGISTRATION",
                        "progress": 0.5,
                        "current": 50,
                        "total": 100,
                        "ts": "2026-03-08T00:00:01Z",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with client.websocket_connect("/api/ws/runs/r42") as ws:
        first = ws.receive_json()
        second = ws.receive_json()

    assert first["type"] == "phase_start"
    assert second["type"] in {"phase_progress", "run_status"}
