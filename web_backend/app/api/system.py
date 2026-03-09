from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.schemas import HealthResponse, VersionResponse
from app.services.command_runner import SecurityPolicyError
from app.services.http_errors import http_from_security_error

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/version", response_model=VersionResponse)
def version(request: Request) -> VersionResponse:
    runtime = request.app.state.runtime
    cli = f"found:{runtime.cli_path}" if runtime.cli_path.exists() else f"missing:{runtime.cli_path}"
    runner = f"found:{runtime.runner_path}" if runtime.runner_path.exists() else f"missing:{runtime.runner_path}"
    return VersionResponse(cli=cli, runner=runner)


@router.get("/fs/roots")
def fs_roots(request: Request) -> dict:
    runtime = request.app.state.runtime
    roots: list[str] = []
    for root in runtime.allowed_roots:
        resolved = root.expanduser().resolve(strict=False)
        if resolved.exists() and resolved.is_dir():
            roots.append(str(resolved))
    # Stable order with de-duplication
    seen: set[str] = set()
    out: list[str] = []
    for item in roots:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    default_path = None
    try:
        preferred = str(runtime.runs_dir.expanduser().resolve(strict=False))
        if preferred in out:
            default_path = preferred
    except Exception:
        default_path = None
    if default_path is None:
        default_path = out[0] if out else None
    return {"items": out, "default_path": default_path}


@router.get("/fs/list")
def fs_list(request: Request, path: str | None = None, include_files: bool = False) -> dict:
    runtime = request.app.state.runtime
    if not path:
        roots = fs_roots(request).get("items", [])
        if not roots:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "code": "NO_ALLOWED_ROOTS",
                        "message": "no readable allowed roots available for file browser",
                    }
                },
            )
        path = str(roots[0])

    try:
        base = runtime.ensure_path_allowed(Path(path).expanduser(), must_exist=True, label="fs_path")
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc

    resolved_base = base.resolve(strict=False)
    if not resolved_base.is_dir():
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "NOT_A_DIRECTORY", "message": "path is not a directory", "details": {"path": str(resolved_base)}}},
        )

    items = []
    try:
        children = sorted(resolved_base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "FS_LIST_FAILED",
                    "message": "directory cannot be listed",
                    "details": {"path": str(resolved_base), "reason": str(exc)},
                }
            },
        ) from exc

    for child in children:
        is_dir = child.is_dir()
        if not is_dir and not include_files:
            continue
        try:
            checked = runtime.ensure_path_allowed(child, must_exist=True, label="fs_entry")
        except SecurityPolicyError:
            continue
        resolved_child = checked.resolve(strict=False)
        items.append(
            {
                "name": resolved_child.name,
                "path": str(resolved_child),
                "type": "dir" if is_dir else "file",
            }
        )

    parent = None
    if resolved_base.parent != resolved_base:
        try:
            checked_parent = runtime.ensure_path_allowed(resolved_base.parent, must_exist=True, label="fs_parent")
            parent = str(checked_parent.resolve(strict=False))
        except SecurityPolicyError:
            parent = None

    return {
        "path": str(resolved_base),
        "parent": parent,
        "items": items,
    }


@router.post("/fs/grant-root")
def fs_grant_root(request: Request, payload: dict | None = None) -> dict:
    body = payload or {}
    raw_path = str(body.get("path") or "").strip()
    if not raw_path:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "path is required"}},
        )
    candidate = Path(raw_path).expanduser().resolve(strict=False)
    if not candidate.is_absolute():
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "PATH_INVALID", "message": "path must be absolute", "details": {"path": raw_path}}},
        )
    if not candidate.exists():
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "PATH_NOT_FOUND", "message": "path does not exist", "details": {"path": str(candidate)}}},
        )
    if not candidate.is_dir():
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "NOT_A_DIRECTORY", "message": "path is not a directory", "details": {"path": str(candidate)}}},
        )

    runtime = request.app.state.runtime
    for root in runtime.allowed_roots:
        resolved_root = root.expanduser().resolve(strict=False)
        if candidate == resolved_root:
            return {"ok": True, "path": str(candidate), "allowed_roots": [str(x) for x in runtime.allowed_roots]}
    runtime.allowed_roots.append(candidate)
    return {"ok": True, "path": str(candidate), "allowed_roots": [str(x) for x in runtime.allowed_roots]}


@router.post("/fs/open")
def fs_open(request: Request, payload: dict | None = None) -> dict:
    body = payload or {}
    raw_path = str(body.get("path") or "").strip()
    if not raw_path:
      raise HTTPException(
          status_code=400,
          detail={"error": {"code": "BAD_REQUEST", "message": "path is required"}},
      )

    runtime = request.app.state.runtime
    try:
        checked = runtime.ensure_path_allowed(Path(raw_path).expanduser(), must_exist=True, label="open_path")
    except SecurityPolicyError as exc:
        raise http_from_security_error(exc) from exc

    target = checked.resolve(strict=False)
    if sys.platform == "darwin":
        command = ["open", str(target)]
    elif os.name == "nt":
        try:
            os.startfile(str(target))
        except OSError as exc:
            raise HTTPException(
                status_code=422,
                detail={"error": {"code": "OPEN_FAILED", "message": str(exc), "details": {"path": str(target)}}},
            ) from exc
        return {"ok": True, "path": str(target), "command": ["startfile", str(target)]}
    else:
        opener = shutil.which("xdg-open")
        if not opener:
            raise HTTPException(
                status_code=422,
                detail={"error": {"code": "OPEN_UNAVAILABLE", "message": "xdg-open is not available", "details": {"path": str(target)}}},
            )
        command = [opener, str(target)]

    try:
        subprocess.Popen(
            command,
            cwd=str(target.parent if target.is_file() else target),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "OPEN_FAILED", "message": str(exc), "details": {"path": str(target)}}},
        ) from exc
    return {"ok": True, "path": str(target), "command": command}
