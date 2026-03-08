from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

if os.getenv("WEB_BACKEND_ENABLE_BINARY_TESTS", "0") != "1":
    pytest.skip("binary integration tests disabled (set WEB_BACKEND_ENABLE_BINARY_TESTS=1)", allow_module_level=True)

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


@pytest.fixture(name="client")
def fixture_client(tmp_path: Path) -> TestClient:
    app = create_app()
    runtime = app.state.runtime
    if not runtime.cli_path.exists() or not runtime.runner_path.exists():
        pytest.skip("real cli/runner binaries not found")
    app.state.runtime.runs_dir = tmp_path / "runs"
    app.state.runtime.runs_dir.mkdir(parents=True, exist_ok=True)
    return TestClient(app)


def _wait_terminal(client: TestClient, job_id: str, timeout_s: float = 30.0) -> dict:
    t0 = time.time()
    while True:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        job = resp.json()
        if job["state"] in {"ok", "error", "cancelled"}:
            return job
        if time.time() - t0 > timeout_s:
            raise AssertionError(f"job timeout: {job_id}")
        time.sleep(0.1)


def test_validate_with_real_cli(client: TestClient) -> None:
    resp = client.post("/api/config/validate", json={"config": {}})
    assert resp.status_code == 200
    body = resp.json()
    assert "ok" in body and "errors" in body and "warnings" in body


def test_scan_with_real_cli(client: TestClient, tmp_path: Path) -> None:
    input_dir = tmp_path / "scan_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    resp = client.post("/api/scan", json={"input_path": str(input_dir), "frames_min": 1})
    assert resp.status_code == 202
    job = _wait_terminal(client, resp.json()["job_id"], timeout_s=60.0)
    assert job["state"] in {"ok", "error"}
    assert job["started_at"] is not None


def test_run_resume_stats_with_real_runner(client: TestClient, tmp_path: Path) -> None:
    input_dir = tmp_path / "run_input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # mark scan as successful to avoid guardrail blocking in integration flow
    app = client.app
    scan_job = app.state.job_store.create("scan", {"result": {"errors": [], "warnings": []}})
    app.state.job_store.set_state(scan_job.job_id, "ok")

    run_id = "it_real_runner"
    start = client.post(
        "/api/runs/start",
        json={"input_dir": str(input_dir), "run_id": run_id, "dry_run": True, "max_frames": 1, "max_tiles": 1},
    )
    assert start.status_code == 202
    start_job = _wait_terminal(client, start.json()["job_id"], timeout_s=120.0)
    assert start_job["state"] in {"ok", "error"}

    revs = client.get("/api/config/revisions")
    assert revs.status_code == 200
    active_rev = revs.json().get("active_revision_id")
    assert active_rev

    run_dir = tmp_path / "resume_target"
    run_dir.mkdir(parents=True, exist_ok=True)
    resume = client.post(
        f"/api/runs/{run_id}/resume",
        json={"from_phase": "PCC", "config_revision_id": active_rev, "run_dir": str(run_dir)},
    )
    assert resume.status_code == 202
    resume_job = _wait_terminal(client, resume.json()["job_id"], timeout_s=120.0)
    assert resume_job["state"] in {"ok", "error"}

    stats = client.post(f"/api/runs/{run_id}/stats", json={"run_dir": str(run_dir)})
    assert stats.status_code == 202
    stats_job = _wait_terminal(client, stats.json()["job_id"], timeout_s=120.0)
    assert stats_job["state"] in {"ok", "error"}
