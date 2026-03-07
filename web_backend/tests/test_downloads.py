from __future__ import annotations

import urllib.error
from pathlib import Path

from app.services.downloads import DownloadOptions, download_file_with_retry


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


def test_download_retry_then_success(monkeypatch, tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    payload = b"hello world"
    calls = {"n": 0}
    states: list[dict] = []

    def _fake_urlopen(req, timeout):  # noqa: ANN001
        _ = req
        _ = timeout
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("temporary")
        return _FakeResponse(status=200, body=payload, headers={"Content-Length": str(len(payload))})

    monkeypatch.setattr("app.services.downloads.urllib.request.urlopen", _fake_urlopen)
    result = download_file_with_retry(
        "https://example.test/file.bin",
        dest,
        options=DownloadOptions(retry_count=2, retry_backoff_s=0.0),
        state_cb=lambda patch: states.append(dict(patch)),
    )
    assert calls["n"] == 2
    assert result.attempts == 2
    assert dest.read_bytes() == payload
    assert any(s.get("retrying") is True for s in states)


def test_download_resume_with_http_range(monkeypatch, tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"hello ")
    full = b"hello world"
    partial_len = len(dest.read_bytes())

    def _fake_urlopen(req, timeout):  # noqa: ANN001
        _ = timeout
        assert req.headers.get("Range") == f"bytes={partial_len}-"
        remaining = full[partial_len:]
        headers = {
            "Content-Length": str(len(remaining)),
            "Content-Range": f"bytes {partial_len}-{len(full) - 1}/{len(full)}",
        }
        return _FakeResponse(status=206, body=remaining, headers=headers)

    monkeypatch.setattr("app.services.downloads.urllib.request.urlopen", _fake_urlopen)
    result = download_file_with_retry(
        "https://example.test/file.bin",
        dest,
        options=DownloadOptions(retry_count=0, resume=True),
    )
    assert result.resumed is True
    assert dest.read_bytes() == full


def test_download_resume_fallback_when_server_returns_200(monkeypatch, tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"old-partial")
    full = b"new full data"

    def _fake_urlopen(req, timeout):  # noqa: ANN001
        _ = req
        _ = timeout
        return _FakeResponse(status=200, body=full, headers={"Content-Length": str(len(full))})

    monkeypatch.setattr("app.services.downloads.urllib.request.urlopen", _fake_urlopen)
    result = download_file_with_retry(
        "https://example.test/file.bin",
        dest,
        options=DownloadOptions(retry_count=0, resume=True),
    )
    assert result.resumed is False
    assert dest.read_bytes() == full
