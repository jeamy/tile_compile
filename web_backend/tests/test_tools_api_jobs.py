from __future__ import annotations

import bz2
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


class _FakeResponse:
    def __init__(self, *, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self._body = body
        self._pos = 0
        self.headers = headers or {}

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = len(self._body) - self._pos
        if self._pos >= len(self._body):
            return b""
        chunk = self._body[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


def _wait_job_ok(client: TestClient, job_id: str, timeout_s: float = 4.0) -> dict:
    t0 = time.time()
    while True:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        job = resp.json()
        if job["state"] in {"ok", "error", "cancelled"}:
            return job
        if time.time() - t0 > timeout_s:
            raise AssertionError(f"job {job_id} timeout")
        time.sleep(0.05)


def test_pcc_download_missing_job_success(monkeypatch, tmp_path: Path) -> None:
    app = create_app()
    client = TestClient(app)
    cat_dir = tmp_path / "siril_cat"
    chunk_data = b"x" * 1024
    bz2_data = bz2.compress(chunk_data)

    def _fake_urlopen(req, timeout):  # noqa: ANN001
        _ = req
        _ = timeout
        return _FakeResponse(status=200, body=bz2_data, headers={"Content-Length": str(len(bz2_data))})

    monkeypatch.setattr("app.services.downloads.urllib.request.urlopen", _fake_urlopen)

    resp = client.post(
        "/api/tools/pcc/siril/download-missing",
        json={"catalog_dir": str(cat_dir), "chunk_ids": [0], "retry_count": 0, "timeout_s": 5},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    job = _wait_job_ok(client, job_id)
    assert job["state"] == "ok"
    assert (cat_dir / "siril_cat1_healpix8_xpsamp_0.dat").exists()


def test_pcc_download_missing_job_resume(monkeypatch, tmp_path: Path) -> None:
    app = create_app()
    client = TestClient(app)
    cat_dir = tmp_path / "siril_cat"
    cat_dir.mkdir(parents=True, exist_ok=True)
    chunk_data = b"1234567890" * 150
    bz2_data = bz2.compress(chunk_data)
    bz2_path = cat_dir / "siril_cat1_healpix8_xpsamp_0.dat.bz2"
    split = len(bz2_data) // 2
    bz2_path.write_bytes(bz2_data[:split])
    called = {"range": None}

    def _fake_urlopen(req, timeout):  # noqa: ANN001
        _ = timeout
        called["range"] = req.headers.get("Range")
        assert called["range"] == f"bytes={split}-"
        remaining = bz2_data[split:]
        headers = {
            "Content-Length": str(len(remaining)),
            "Content-Range": f"bytes {split}-{len(bz2_data) - 1}/{len(bz2_data)}",
        }
        return _FakeResponse(status=206, body=remaining, headers=headers)

    monkeypatch.setattr("app.services.downloads.urllib.request.urlopen", _fake_urlopen)

    resp = client.post(
        "/api/tools/pcc/siril/download-missing/retry",
        json={"catalog_dir": str(cat_dir), "chunk_ids": [0], "retry_count": 0, "resume": True, "timeout_s": 5},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    job = _wait_job_ok(client, job_id)
    assert job["state"] == "ok"
    assert called["range"] == f"bytes={split}-"
    assert (cat_dir / "siril_cat1_healpix8_xpsamp_0.dat").exists()
