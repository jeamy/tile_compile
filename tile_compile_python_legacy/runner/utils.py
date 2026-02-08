"""
Utility functions for the tile-compile runner.

General-purpose helpers for hashing, JSON serialization, file operations, etc.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def json_dumps_canonical(obj: Any) -> bytes:
    """Canonical JSON serialization for hashing."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_bytes(path: Path) -> bytes:
    """Read file as bytes."""
    return path.read_bytes()


def copy_config(config_path: Path, out_path: Path) -> None:
    """Copy configuration file to output directory."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_path)


def discover_frames(input_dir: Path, pattern: str) -> list[Path]:
    """Discover FITS frames matching pattern in input directory."""
    paths = sorted([p for p in input_dir.glob(pattern) if p.is_file()])
    return paths


def safe_symlink_or_copy(src: Path, dst: Path) -> None:
    """Create symlink or copy file if symlink fails."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)


def safe_hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(src, dst)


def resolve_project_root(start: Path) -> Path:
    """Resolve project root by searching for marker files."""
    p = start
    if p.is_file():
        p = p.parent
    p = p.resolve()
    for _ in range(10):
        if (p / "tile_compile.yaml").exists() or (p / "tile_compile.schema.json").exists():
            return p
        if p.parent == p:
            return start.resolve() if start.is_dir() else start.parent.resolve()
        p = p.parent
    return start.resolve() if start.is_dir() else start.parent.resolve()


def load_gui_state(project_root: Path) -> dict:
    """Load GUI state from JSON file."""
    path = project_root / "tile_compile_gui_state.json"
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_siril_exe(project_root: Path) -> tuple[str | None, str]:
    """Resolve Siril executable path."""
    exe = shutil.which("siril")
    if exe:
        return exe, "path"
    
    gui_state = load_gui_state(project_root)
    siril_path_cfg = gui_state.get("siril_path")
    if isinstance(siril_path_cfg, str) and siril_path_cfg.strip():
        candidate = Path(siril_path_cfg.strip()).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return str(candidate), "gui_state"
    
    return None, "none"


def pick_output_file(candidates: list[Path]) -> Path | None:
    """Pick first existing file from candidates."""
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None
