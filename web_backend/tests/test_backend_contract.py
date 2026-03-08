from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

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


def test_scan_latest_and_app_state_include_summary() -> None:
    app = create_app()
    client = TestClient(app)

    initial = client.get("/api/scan/latest")
    assert initial.status_code == 200
    assert initial.json()["has_scan"] is False

    app.state.last_scan_input_path = "/tmp/legacy_input"
    scan_job = app.state.job_store.create(
        "scan",
        {
            "input_path": "/tmp/legacy_input",
            "result": {
                "ok": True,
                "input_path": "/tmp/legacy_input",
                "frames_detected": 123,
                "image_width": 3008,
                "image_height": 3008,
                "color_mode": "OSC",
                "color_mode_candidates": ["OSC"],
                "bayer_pattern": "RGGB",
                "requires_user_confirmation": False,
                "errors": [],
                "warnings": [{"code": "w1"}],
            },
        },
    )
    app.state.job_store.set_state(scan_job.job_id, "ok")

    latest = client.get("/api/scan/latest")
    assert latest.status_code == 200
    body = latest.json()
    assert body["has_scan"] is True
    assert body["frames_detected"] == 123
    assert body["color_mode"] == "OSC"
    assert body["image_width"] == 3008
    assert body["image_height"] == 3008
    assert body["bayer_pattern"] == "RGGB"
    assert len(body["warnings"]) == 1

    state = client.get("/api/app/state")
    assert state.status_code == 200
    scan_state = state.json()["scan"]
    assert scan_state["last_input_path"] == "/tmp/legacy_input"
    assert scan_state["last_scan"]["frames_detected"] == 123


def test_app_state_includes_current_run_and_history_summary(tmp_path: Path) -> None:
    app = create_app()
    app.state.runtime.runs_dir = tmp_path
    client = TestClient(app)

    run_dir = tmp_path / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(json.dumps({"type": "run_end", "success": True}) + "\n", encoding="utf-8")
    app.state.current_run_id = "r1"

    resp = client.get("/api/app/state")
    assert resp.status_code == 200
    body = resp.json()

    assert body["run"]["current"]["run_id"] == "r1"
    assert body["run"]["current"]["run_dir"] == str(run_dir)
    assert body["run"]["current"]["status"] == "completed"
    assert body["history"]["total_runs"] == 1
    assert len(body["history"]["recent"]) == 1
    assert body["history"]["recent"][0]["run_id"] == "r1"


def test_run_delete_removes_history_entry_and_clears_current(tmp_path: Path) -> None:
    app = create_app()
    app.state.runtime.runs_dir = tmp_path
    client = TestClient(app)

    run_dir = tmp_path / "r_delete"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(json.dumps({"type": "run_end", "success": True}) + "\n", encoding="utf-8")
    app.state.current_run_id = "r_delete"

    resp = client.post("/api/runs/r_delete/delete")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert not run_dir.exists()
    assert app.state.current_run_id == ""


def test_run_start_blocked_by_guardrail_error() -> None:
    app = create_app()
    client = TestClient(app)

    scan_job = app.state.job_store.create("scan", {"result": {"errors": [{"message": "bad"}], "warnings": []}})
    app.state.job_store.set_state(scan_job.job_id, "ok")

    resp = client.post("/api/runs/start", json={"input_dir": "/tmp/input"})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "GUARDRAIL_BLOCKED"


def test_resume_requires_phase_and_revision(tmp_path: Path) -> None:
    app = create_app()
    app.state.runtime.runs_dir = tmp_path
    client = TestClient(app)

    missing = client.post("/api/runs/r1/resume", json={})
    assert missing.status_code == 409
    assert missing.json()["error"]["code"] == "RESUME_PHASE_REQUIRED"

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


def test_jobs_endpoint_contains_started_and_ended_timestamps() -> None:
    app = create_app()
    client = TestClient(app)

    job = app.state.job_store.create("sample", {"run_id": "r_test"})
    app.state.job_store.set_state(job.job_id, "running")
    app.state.job_store.set_state(job.job_id, "ok")

    resp = client.get(f"/api/jobs/{job.job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "r_test"
    assert data["started_at"] is not None
    assert data["ended_at"] is not None


def test_error_envelope_for_http_exceptions() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.post("/api/runs/r1/resume", json={})
    assert resp.status_code == 409
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == "RESUME_PHASE_REQUIRED"
    assert "message" in body["error"]


def test_error_envelope_for_unknown_route() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "NOT_FOUND"


def test_ws_job_stream_contains_job_data() -> None:
    app = create_app()
    client = TestClient(app)
    job = app.state.job_store.create("scan", {"run_id": "r5", "step": "test"})
    app.state.job_store.set_state(job.job_id, "running")

    with client.websocket_connect(f"/api/ws/jobs/{job.job_id}") as ws:
        event = ws.receive_json()

    assert event["type"] == "job_progress"
    assert event["job_id"] == job.job_id
    assert event["state"] == "running"
    assert event["data"]["run_id"] == "r5"


def test_scan_multi_input_aggregates_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = create_app()
    app.state.runtime.allowed_roots.append(tmp_path)
    client = TestClient(app)

    d1 = tmp_path / "in_l"
    d2 = tmp_path / "in_r"
    d1.mkdir(parents=True, exist_ok=True)
    d2.mkdir(parents=True, exist_ok=True)

    def _fake_run_command(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        input_path = str(command[2])
        if input_path.endswith("in_l"):
            parsed = {
                "ok": True,
                "input_path": input_path,
                "frames_detected": 12,
                "image_width": 3008,
                "image_height": 3008,
                "color_mode": "MONO",
                "color_mode_candidates": ["MONO"],
                "bayer_pattern": None,
                "requires_user_confirmation": False,
                "errors": [],
                "warnings": [],
                "frames": [],
            }
        else:
            parsed = {
                "ok": True,
                "input_path": input_path,
                "frames_detected": 18,
                "image_width": 3008,
                "image_height": 3008,
                "color_mode": "MONO",
                "color_mode_candidates": ["MONO"],
                "bayer_pattern": None,
                "requires_user_confirmation": False,
                "errors": [],
                "warnings": [],
                "frames": [],
            }
        return type("ScanResult", (), {"exit_code": 0, "parsed_json": parsed, "stderr": ""})()

    monkeypatch.setattr("app.api.scan.run_command", _fake_run_command)

    resp = client.post("/api/scan", json={"input_dirs": [str(d1), str(d2)], "frames_min": 3})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    deadline = time.time() + 3.0
    job_state = "running"
    while time.time() < deadline:
        job_resp = client.get(f"/api/jobs/{job_id}")
        assert job_resp.status_code == 200
        job_state = job_resp.json()["state"]
        if job_state in {"ok", "error", "cancelled"}:
            break
        time.sleep(0.05)
    assert job_state == "ok"

    latest = client.get("/api/scan/latest")
    assert latest.status_code == 200
    body = latest.json()
    assert body["ok"] is True
    assert body["frames_detected"] == 30
    assert body["color_mode"] == "MONO"
    assert body["input_dirs"] == [str(d1), str(d2)]
    assert len(body.get("per_dir_results", [])) == 2
