from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.time_utils import isoformat_z, utc_now


@dataclass
class UiEvent:
    seq: int
    ts: datetime
    event: str
    source: str
    payload: dict[str, Any]
    run_id: str | None = None
    job_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": isoformat_z(self.ts),
            "event": self.event,
            "source": self.source,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "payload": self.payload,
        }


class UiEventStore:
    """Thread-safe ui_event audit log with JSONL persistence for replay."""

    def __init__(self, path: Path, *, max_in_memory: int = 5000) -> None:
        self._path = path
        self._max_in_memory = max(100, int(max_in_memory))
        self._items: list[UiEvent] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        event: str,
        source: str,
        payload: dict[str, Any] | None = None,
        run_id: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            item = UiEvent(
                seq=self._seq,
                ts=utc_now(),
                event=event,
                source=source,
                payload=payload or {},
                run_id=run_id,
                job_id=job_id,
            )
            self._items.append(item)
            if len(self._items) > self._max_in_memory:
                self._items = self._items[-self._max_in_memory :]
            self._append_jsonl(item)
            return item.as_dict()

    def list(self, *, after_seq: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        lim = max(1, min(5000, int(limit)))
        with self._lock:
            filtered = [x.as_dict() for x in self._items if x.seq > after_seq]
        return filtered[:lim]

    @property
    def latest_seq(self) -> int:
        with self._lock:
            return self._seq

    def _append_jsonl(self, item: UiEvent) -> None:
        line = json.dumps(item.as_dict(), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")


def record_ui_event(
    request: Any,
    *,
    event: str,
    source: str,
    payload: dict[str, Any] | None = None,
    run_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any] | None:
    store = getattr(request.app.state, "ui_event_store", None)
    if store is None:
        return None
    return store.append(event=event, source=source, payload=payload, run_id=run_id, job_id=job_id)
