from __future__ import annotations

from typing import Any


def normalize_queue_item(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        return {"input_dir": item}
    if isinstance(item, dict):
        if not item.get("input_dir"):
            return None
        return dict(item)
    return None


def extract_queue_specs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_queue = payload.get("queue")
    if isinstance(raw_queue, list) and raw_queue:
        out: list[dict[str, Any]] = []
        for item in raw_queue:
            normalized = normalize_queue_item(item)
            if normalized is not None:
                out.append(normalized)
        return out

    raw_input_dirs = payload.get("input_dirs")
    if isinstance(raw_input_dirs, list) and len(raw_input_dirs) > 1:
        out: list[dict[str, Any]] = []
        for item in raw_input_dirs:
            normalized = normalize_queue_item(item)
            if normalized is not None:
                out.append(normalized)
        return out
    return []
