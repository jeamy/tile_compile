from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.schemas import JobAccepted
from app.services.command_runner import SecurityPolicyError, launch_background_command, run_command
from app.services.guardrails import compute_guardrails
from app.services.http_errors import http_from_security_error
from app.services.scan_summary import latest_scan_job, summarize_scan_job
from app.services.ui_events import record_ui_event

router = APIRouter(tags=["scan"])


@router.post("/scan", response_model=JobAccepted, status_code=202)
def scan(request: Request, payload: dict[str, Any]) -> JobAccepted:
    runtime = request.app.state.runtime
    input_path = payload.get("input_path")
    input_dirs = payload.get("input_dirs")

    requested_inputs: list[str] = []
    if isinstance(input_dirs, list):
        for item in input_dirs:
            token = str(item or "").strip()
            if token:
                requested_inputs.append(token)
    if not requested_inputs and input_path:
        requested_inputs.append(str(input_path).strip())
    if not requested_inputs:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BAD_REQUEST", "message": "input_path or input_dirs is required"}},
        )
    frames_min = int(payload.get("frames_min", 1))
    with_checksums = bool(payload.get("with_checksums", False))
    resolved_inputs: list[str] = []
    for raw_path in requested_inputs:
        input_path_is_absolute = Path(raw_path).expanduser().is_absolute()
        try:
            checked = runtime.resolve_input_path(
                raw_path,
                must_exist=not input_path_is_absolute,
                label="input_path",
            )
        except SecurityPolicyError as exc:
            raise http_from_security_error(exc) from exc
        resolved_inputs.append(str(checked))
    request.app.state.last_scan_input_path = resolved_inputs[0]

    job = request.app.state.job_store.create(
        "scan",
        {
            "input_path": resolved_inputs[0],
            "input_dirs": resolved_inputs,
            "frames_min": frames_min,
            "with_checksums": with_checksums,
        },
    )
    request.app.state.job_store.set_state(job.job_id, "running")

    if len(resolved_inputs) == 1:
        cmd = [str(runtime.cli_path), "scan", resolved_inputs[0], "--frames-min", str(frames_min)]
        if with_checksums:
            cmd.append("--with-checksums")
        request.app.state.job_store.merge_data(job.job_id, {"command": cmd})
        launch_background_command(
            job_store=request.app.state.job_store,
            job_id=job.job_id,
            command=cmd,
            cwd=runtime.project_root,
            command_policy=request.app.state.command_policy,
        )
    else:
        def _scan_worker() -> None:
            try:
                per_dir_results: list[dict[str, Any]] = []
                color_candidates: list[str] = []
                color_modes_detected: list[str] = []
                frames_detected_total = 0
                image_width = 0
                image_height = 0
                bayer_pattern = None
                requires_confirmation = False
                ok = True
                all_errors: list[Any] = []
                all_warnings: list[Any] = []
                all_frames: list[Any] = []

                for index, resolved_input in enumerate(resolved_inputs):
                    snapshot = request.app.state.job_store.get(job.job_id)
                    if snapshot and snapshot.state == "cancelled":
                        return
                    cmd = [str(runtime.cli_path), "scan", resolved_input, "--frames-min", str(frames_min)]
                    if with_checksums:
                        cmd.append("--with-checksums")
                    result = run_command(
                        cmd,
                        cwd=runtime.project_root,
                        command_policy=request.app.state.command_policy,
                    )
                    parsed = result.parsed_json if isinstance(result.parsed_json, dict) else {}
                    item_ok = bool(parsed.get("ok", result.exit_code == 0))
                    item_errors = parsed.get("errors") if isinstance(parsed.get("errors"), list) else []
                    item_warnings = parsed.get("warnings") if isinstance(parsed.get("warnings"), list) else []
                    if result.exit_code != 0 and not item_errors:
                        item_errors = [
                            {
                                "code": "scan_failed",
                                "message": "scan command failed",
                                "details": {"exit_code": result.exit_code, "stderr": result.stderr},
                            }
                        ]
                    item = {
                        "input_path": resolved_input,
                        "ok": item_ok and result.exit_code == 0 and len(item_errors) == 0,
                        "frames_detected": int(parsed.get("frames_detected", 0)),
                        "image_width": int(parsed.get("image_width", 0)),
                        "image_height": int(parsed.get("image_height", 0)),
                        "color_mode": str(parsed.get("color_mode", "UNKNOWN")),
                        "color_mode_candidates": parsed.get("color_mode_candidates", []),
                        "bayer_pattern": parsed.get("bayer_pattern"),
                        "requires_user_confirmation": bool(parsed.get("requires_user_confirmation", False)),
                        "errors": item_errors,
                        "warnings": item_warnings,
                        "frames": parsed.get("frames", []),
                    }
                    per_dir_results.append(item)

                    ok = ok and bool(item["ok"])
                    frames_detected_total += int(item["frames_detected"])
                    if image_width == 0:
                        image_width = int(item["image_width"])
                    if image_height == 0:
                        image_height = int(item["image_height"])
                    if bayer_pattern is None and item.get("bayer_pattern") is not None:
                        bayer_pattern = item.get("bayer_pattern")
                    requires_confirmation = requires_confirmation or bool(item["requires_user_confirmation"])
                    all_errors.extend(item_errors)
                    all_warnings.extend(item_warnings)
                    if isinstance(item.get("frames"), list):
                        all_frames.extend(item["frames"])

                    color_mode = str(item["color_mode"]).strip()
                    if color_mode and color_mode != "UNKNOWN":
                        color_modes_detected.append(color_mode)
                        if color_mode not in color_candidates:
                            color_candidates.append(color_mode)
                    item_candidates = item.get("color_mode_candidates")
                    if isinstance(item_candidates, list):
                        for candidate in item_candidates:
                            candidate_text = str(candidate).strip()
                            if candidate_text and candidate_text not in color_candidates:
                                color_candidates.append(candidate_text)

                    request.app.state.job_store.merge_data(
                        job.job_id,
                        {
                            "current_index": index,
                            "input_path": resolved_input,
                            "progress": (index + 1) / max(1, len(resolved_inputs)),
                            "per_dir_results": per_dir_results,
                        },
                    )

                final_color_mode = "UNKNOWN"
                unique_modes = sorted(set(color_modes_detected))
                if len(unique_modes) == 1:
                    final_color_mode = unique_modes[0]
                elif len(unique_modes) > 1:
                    requires_confirmation = True

                summary = {
                    "ok": ok and len(all_errors) == 0,
                    "input_path": resolved_inputs[0],
                    "input_dirs": resolved_inputs,
                    "frames_detected": frames_detected_total,
                    "image_width": image_width,
                    "image_height": image_height,
                    "color_mode": final_color_mode,
                    "color_mode_candidates": color_candidates,
                    "bayer_pattern": bayer_pattern,
                    "requires_user_confirmation": requires_confirmation,
                    "errors": all_errors,
                    "warnings": all_warnings,
                    "frames": all_frames,
                    "per_dir_results": per_dir_results,
                }
                request.app.state.job_store.merge_data(job.job_id, {"result": summary, "progress": 1.0})
                request.app.state.job_store.set_state(job.job_id, "ok" if summary["ok"] else "error")
            except Exception as exc:
                request.app.state.job_store.merge_data(job.job_id, {"error": str(exc)})
                snapshot = request.app.state.job_store.get(job.job_id)
                if snapshot and snapshot.state != "cancelled":
                    request.app.state.job_store.set_state(job.job_id, "error")

        thread = threading.Thread(target=_scan_worker, name=f"scan-multi-{job.job_id}", daemon=True)
        thread.start()

    record_ui_event(
        request,
        event="scan.start",
        source="scan.scan",
        job_id=job.job_id,
        payload={
            "input_path": resolved_inputs[0],
            "input_dirs": resolved_inputs,
            "frames_min": frames_min,
            "with_checksums": with_checksums,
        },
    )
    return JobAccepted(job_id=job.job_id, state="running")


@router.get("/scan/quality")
def scan_quality(request: Request) -> dict[str, Any]:
    scan_job = latest_scan_job(request.app.state.job_store)
    fallback_input_path = str(getattr(request.app.state, "last_scan_input_path", "") or "")
    summary = summarize_scan_job(scan_job, fallback_input_path=fallback_input_path)
    if not summary["has_scan"]:
        return {
            "score": 0.0,
            "factors": [{"id": "no_scan", "value": 1.0, "label": "No scan run yet"}],
            "scan": summary,
        }

    errors = len(summary["errors"])
    warnings = len(summary["warnings"])
    score = max(0.0, 1.0 - 0.25 * errors - 0.1 * warnings)
    return {
        "score": round(score, 3),
        "factors": [
            {"id": "errors", "value": errors, "label": "scan errors"},
            {"id": "warnings", "value": warnings, "label": "scan warnings"},
        ],
        "scan": summary,
    }


@router.get("/scan/latest")
def scan_latest(request: Request) -> dict[str, Any]:
    scan_job = latest_scan_job(request.app.state.job_store)
    fallback_input_path = str(getattr(request.app.state, "last_scan_input_path", "") or "")
    return summarize_scan_job(scan_job, fallback_input_path=fallback_input_path)


@router.get("/guardrails")
def guardrails(request: Request) -> dict[str, Any]:
    return compute_guardrails(request.app.state.job_store)
