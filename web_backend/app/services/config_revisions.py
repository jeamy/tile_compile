from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def create_revision(
    app: Any,
    *,
    path: Path,
    yaml_text: str,
    source: str,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    revision_id = f"cfg_{uuid.uuid4().hex[:10]}"
    revision = {
        "revision_id": revision_id,
        "path": str(path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source": source,
        "run_id": run_id,
        "yaml": yaml_text,
    }
    if extra:
        revision.update(extra)
    app.state.config_revisions.insert(0, revision)
    app.state.active_config_revision_id = revision_id
    return revision


def list_revisions(app: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in app.state.config_revisions:
        cloned = dict(item)
        yaml_text = cloned.pop("yaml", None)
        cloned["has_snapshot"] = bool(isinstance(yaml_text, str) and yaml_text.strip())
        out.append(cloned)
    return out


def get_revision(app: Any, revision_id: str) -> dict[str, Any] | None:
    for item in app.state.config_revisions:
        if str(item.get("revision_id")) == revision_id:
            return item
    return None


def restore_revision(app: Any, revision_id: str) -> dict[str, Any]:
    revision = get_revision(app, revision_id)
    if revision is None:
        raise KeyError(revision_id)

    yaml_text = revision.get("yaml")
    path = Path(str(revision.get("path", ""))).expanduser()
    if isinstance(yaml_text, str) and yaml_text.strip():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml_text, encoding="utf-8")
    elif path.exists():
        # keep existing file if no snapshot text exists
        pass
    else:
        # create empty YAML to avoid missing target paths
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({}, sort_keys=False), encoding="utf-8")

    app.state.active_config_revision_id = revision_id
    return revision
