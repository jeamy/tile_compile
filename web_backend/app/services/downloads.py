from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DownloadOptions:
    timeout_s: int = 120
    retry_count: int = 2
    retry_backoff_s: float = 1.5
    resume: bool = True
    chunk_size: int = 256 * 1024
    user_agent: str = "TileCompileGUI2/1.0"


@dataclass(frozen=True)
class DownloadResult:
    attempts: int
    bytes_written: int
    resumed: bool


class DownloadAborted(RuntimeError):
    pass


ProgressCallback = Callable[[int, int], None]
StateCallback = Callable[[dict], None]


def download_file_with_retry(
    url: str,
    dest_path: Path,
    *,
    options: DownloadOptions | None = None,
    progress_cb: ProgressCallback | None = None,
    state_cb: StateCallback | None = None,
) -> DownloadResult:
    opts = options or DownloadOptions()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    attempts_total = max(1, int(opts.retry_count) + 1)
    last_error: Exception | None = None

    for attempt in range(1, attempts_total + 1):
        try:
            return _download_once(
                url=url,
                dest_path=dest_path,
                attempt=attempt,
                options=opts,
                progress_cb=progress_cb,
                state_cb=state_cb,
            )
        except DownloadAborted:
            raise
        except Exception as exc:
            last_error = exc
            retrying = attempt < attempts_total
            if state_cb:
                state_cb(
                    {
                        "attempt": attempt,
                        "retrying": retrying,
                        "error": str(exc),
                    }
                )
            if not retrying:
                break
            sleep_s = max(0.0, float(opts.retry_backoff_s)) * float(attempt)
            if sleep_s > 0:
                time.sleep(sleep_s)

    assert last_error is not None
    raise last_error


def _download_once(
    *,
    url: str,
    dest_path: Path,
    attempt: int,
    options: DownloadOptions,
    progress_cb: ProgressCallback | None,
    state_cb: StateCallback | None,
) -> DownloadResult:
    existing = dest_path.stat().st_size if options.resume and dest_path.exists() else 0
    headers = {"User-Agent": options.user_agent}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=options.timeout_s) as resp:
        status = int(getattr(resp, "status", 200) or 200)
        total = _content_total(resp.headers.get("Content-Length", "0"), existing, status, resp.headers.get("Content-Range"))

        resumed = existing > 0 and status == 206
        if existing > 0 and status != 206:
            existing = 0
            resumed = False

        mode = "ab" if resumed else "wb"
        if state_cb:
            state_cb(
                {
                    "attempt": attempt,
                    "status_code": status,
                    "resumed": resumed,
                    "existing_bytes": existing,
                    "bytes_total": total,
                }
            )

        received = existing
        with dest_path.open(mode) as f:
            while True:
                chunk = resp.read(options.chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)
                if progress_cb:
                    progress_cb(received, total)
        return DownloadResult(attempts=attempt, bytes_written=received, resumed=resumed)


def _content_total(content_length_header: str, existing: int, status: int, content_range: str | None) -> int:
    try:
        cl = int(content_length_header or "0")
    except ValueError:
        cl = 0
    if status == 206:
        total_from_range = _parse_total_from_content_range(content_range)
        if total_from_range > 0:
            return total_from_range
        return existing + max(0, cl)
    return max(0, cl)


def _parse_total_from_content_range(value: str | None) -> int:
    if not value:
        return 0
    # Expected: "bytes start-end/total"
    if "/" not in value:
        return 0
    total_str = value.rsplit("/", 1)[1].strip()
    if total_str == "*":
        return 0
    try:
        return int(total_str)
    except ValueError:
        return 0
