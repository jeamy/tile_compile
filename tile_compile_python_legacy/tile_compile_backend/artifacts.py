from __future__ import annotations

from pathlib import Path
from typing import Any


def list_artifacts(run_dir: str) -> dict[str, Any]:
    p = Path(run_dir).expanduser().resolve()
    artifacts_dir = p / "artifacts"
    outputs_dir = p / "outputs"

    def collect(root: Path) -> list[dict[str, Any]]:
        if not root.exists() or not root.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for fp in sorted([x for x in root.rglob("*") if x.is_file()]):
            rel = str(fp.relative_to(p))
            out.append({"path": rel, "size": fp.stat().st_size})
        return out

    return {
        "run_dir": str(p),
        "artifacts": collect(artifacts_dir),
        "outputs": collect(outputs_dir),
    }
