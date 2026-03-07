from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from subprocess import Popen
from typing import Any


@dataclass
class Job:
    job_id: str
    job_type: str
    state: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    data: dict[str, Any] = field(default_factory=dict)
    pid: int | None = None
    exit_code: int | None = None


class InMemoryJobStore:
    """Thread-safe in-memory job tracking with process attachment for cancellation."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._processes: dict[str, Popen[str]] = {}
        self._lock = threading.Lock()

    def create(self, job_type: str, data: dict[str, Any] | None = None) -> Job:
        with self._lock:
            job_id = f"job_{uuid.uuid4().hex[:10]}"
            job = Job(job_id=job_id, job_type=job_type, data=data or {})
            self._jobs[job_id] = job
            return job

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def set_state(self, job_id: str, state: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.state = state
            job.updated_at = datetime.utcnow()
            return job

    def merge_data(self, job_id: str, patch: dict[str, Any]) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.data.update(patch)
            job.updated_at = datetime.utcnow()
            return job

    def set_process(self, job_id: str, process: Popen[str]) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            self._processes[job_id] = process
            job.pid = process.pid
            job.updated_at = datetime.utcnow()
            return job

    def set_exit_code(self, job_id: str, exit_code: int) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.exit_code = exit_code
            job.updated_at = datetime.utcnow()
            return job

    def clear_process(self, job_id: str) -> None:
        with self._lock:
            self._processes.pop(job_id, None)

    def cancel(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            process = self._processes.get(job_id)
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
            job.state = "cancelled"
            job.updated_at = datetime.utcnow()
            return job
