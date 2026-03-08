from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request

from app.services.command_runner import CommandExecutionError, run_command, run_json_command
from app.services.config_revisions import create_revision, get_revision, list_revisions, restore_revision
from app.services.ui_events import record_ui_event

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/schema")
def config_schema(request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    result = run_command([str(runtime.cli_path), "get-schema"], cwd=runtime.project_root)
    if result.exit_code != 0 or not isinstance(result.parsed_json, dict):
        raise _http_502_command_failed("failed to fetch schema", result)
    return result.parsed_json


@router.get("/current")
def config_current(request: Request, path: str | None = None) -> dict[str, Any]:
    runtime = request.app.state.runtime
    config_path = Path(path).expanduser() if path else runtime.default_config_path
    result = run_command([str(runtime.cli_path), "load-config", str(config_path)], cwd=runtime.project_root)
    if result.exit_code != 0 or not isinstance(result.parsed_json, dict):
        raise _http_502_command_failed("failed to load config", result)
    return {"config": result.parsed_json.get("yaml", ""), "source": str(config_path)}


@router.post("/validate")
def validate_config(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    strict = bool(payload.get("strict_exit_codes", False))
    config_path = payload.get("path")
    yaml_text = payload.get("yaml")
    config_object = payload.get("config")
    stdin_text: str | None = None
    cmd = [str(runtime.cli_path), "validate-config"]

    if config_path:
        cmd.extend(["--path", str(Path(config_path).expanduser())])
    else:
        if yaml_text:
            stdin_text = str(yaml_text)
        elif isinstance(config_object, dict):
            stdin_text = yaml.safe_dump(config_object, sort_keys=False)
        else:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "BAD_REQUEST", "message": "provide one of: path, yaml, or config"}},
            )
        cmd.append("--stdin")

    if strict:
        cmd.append("--strict-exit-codes")

    result = run_command(cmd, cwd=runtime.project_root, stdin_text=stdin_text)
    if not isinstance(result.parsed_json, dict):
        raise _http_502_command_failed("validate-config returned non-json", result)

    parsed = result.parsed_json
    return {
        "ok": bool(parsed.get("valid", False)),
        "errors": parsed.get("errors", []),
        "warnings": parsed.get("warnings", []),
    }


@router.post("/save")
def save_config(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    path = payload.get("path")
    yaml_text = payload.get("yaml")
    config_object = payload.get("config")
    target = Path(path).expanduser() if path else runtime.default_config_path

    if yaml_text:
        text = str(yaml_text)
    elif isinstance(config_object, dict):
        text = yaml.safe_dump(config_object, sort_keys=False)
    else:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "provide yaml or config object"}},
        )

    try:
        result = run_json_command(
            [str(runtime.cli_path), "save-config", str(target), "--stdin"],
            cwd=runtime.project_root,
            stdin_text=text,
        )
    except CommandExecutionError as exc:
        raise _http_502_command_failed("save-config failed", exc) from exc
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=502,
            detail={"error": {"code": "BAD_BACKEND_RESPONSE", "message": "save-config returned invalid response"}},
        )
    saved_path = Path(str(result.get("path", str(target)))).expanduser()
    revision = create_revision(
        request.app,
        path=saved_path,
        yaml_text=text,
        source="save_config",
    )
    record_ui_event(
        request,
        event="config.save",
        source="config.save",
        payload={"path": str(saved_path), "saved": bool(result.get("saved", False)), "revision_id": revision["revision_id"]},
    )
    return {
        "path": str(saved_path),
        "saved": bool(result.get("saved", False)),
        "revision_id": revision["revision_id"],
    }


@router.get("/presets")
def presets(request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    examples_dir = runtime.project_root / "tile_compile_cpp/examples"
    items: list[dict[str, str]] = []
    if examples_dir.exists():
        for path in sorted(examples_dir.glob("*.yaml")):
            if "example" not in path.name:
                continue
            items.append(
                {
                    "id": path.stem,
                    "name": path.name,
                    "path": str(path),
                }
            )
    return {"items": items}


@router.post("/presets/apply")
def presets_apply(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    preset_path = payload.get("path")
    if not preset_path:
        raise HTTPException(
            status_code=400, detail={"error": {"code": "BAD_REQUEST", "message": "path is required"}}
        )
    path = Path(str(preset_path)).expanduser()
    if not path.is_absolute():
        path = runtime.project_root / path
    try:
        result = run_json_command([str(runtime.cli_path), "load-config", str(path)], cwd=runtime.project_root)
    except CommandExecutionError as exc:
        raise _http_502_command_failed("load-config failed", exc) from exc
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=502,
            detail={"error": {"code": "BAD_BACKEND_RESPONSE", "message": "load-config returned invalid response"}},
        )
    record_ui_event(
        request,
        event="config.preset.apply",
        source="config.presets_apply",
        payload={"preset_path": str(path)},
    )
    return {"config": result.get("yaml", ""), "applied_paths": [str(path)]}


@router.get("/revisions")
def revisions(request: Request) -> dict[str, Any]:
    return {"items": list_revisions(request.app), "active_revision_id": request.app.state.active_config_revision_id}


@router.post("/revisions/{revision_id}/restore")
def revision_restore(revision_id: str, request: Request) -> dict[str, Any]:
    item = get_revision(request.app, revision_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": f"revision '{revision_id}' not found"}},
        )
    revision = restore_revision(request.app, revision_id)
    record_ui_event(
        request,
        event="config.revision.restore",
        source="config.revision_restore",
        payload={"revision_id": revision_id, "path": revision.get("path")},
    )
    return {"ok": True, "active_revision_id": revision_id}


def _http_502_command_failed(message: str, result: Any) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail={
            "error": {
                "code": "BACKEND_COMMAND_FAILED",
                "message": message,
                "details": {
                    "exit_code": getattr(result, "exit_code", None),
                    "stdout": getattr(result, "stdout", ""),
                    "stderr": getattr(result, "stderr", ""),
                },
            }
        },
    )
