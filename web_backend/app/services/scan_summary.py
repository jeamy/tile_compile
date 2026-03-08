from __future__ import annotations

from typing import Any


def latest_scan_job(job_store: Any) -> Any | None:
    items = job_store.list()
    return next((job for job in items if getattr(job, "job_type", "") == "scan"), None)


def summarize_scan_job(job: Any, *, fallback_input_path: str = "") -> dict[str, Any]:
    if job is None:
        return {
            "has_scan": False,
            "job_id": None,
            "job_state": "pending",
            "input_path": fallback_input_path,
            "input_dirs": [],
            "ok": False,
            "frames_detected": 0,
            "image_width": 0,
            "image_height": 0,
            "color_mode": "UNKNOWN",
            "color_mode_candidates": [],
            "bayer_pattern": None,
            "requires_user_confirmation": False,
            "errors": [],
            "warnings": [],
            "frames": [],
            "per_dir_results": [],
        }

    raw = job.data.get("result", {}) if isinstance(job.data, dict) else {}
    result = raw if isinstance(raw, dict) else {}
    errors = result.get("errors")
    warnings = result.get("warnings")
    errors_list = errors if isinstance(errors, list) else []
    warnings_list = warnings if isinstance(warnings, list) else []

    input_path = str(result.get("input_path") or job.data.get("input_path") or fallback_input_path or "")
    input_dirs_raw = result.get("input_dirs") if isinstance(result.get("input_dirs"), list) else job.data.get("input_dirs")
    input_dirs = input_dirs_raw if isinstance(input_dirs_raw, list) else []
    frames_detected = int(result.get("frames_detected") or 0)
    image_width = int(result.get("image_width") or 0)
    image_height = int(result.get("image_height") or 0)
    color_mode = str(result.get("color_mode") or "UNKNOWN")
    candidates_raw = result.get("color_mode_candidates")
    color_mode_candidates = candidates_raw if isinstance(candidates_raw, list) else []
    frames_raw = result.get("frames")
    frames = frames_raw if isinstance(frames_raw, list) else []
    per_dir_raw = result.get("per_dir_results")
    per_dir_results = per_dir_raw if isinstance(per_dir_raw, list) else []

    ok_raw = result.get("ok")
    ok = bool(ok_raw) if isinstance(ok_raw, bool) else len(errors_list) == 0

    return {
        "has_scan": True,
        "job_id": job.job_id,
        "job_state": job.state,
        "input_path": input_path,
        "input_dirs": [str(x) for x in input_dirs if str(x)],
        "ok": ok,
        "frames_detected": frames_detected,
        "image_width": image_width,
        "image_height": image_height,
        "color_mode": color_mode,
        "color_mode_candidates": [str(x) for x in color_mode_candidates],
        "bayer_pattern": result.get("bayer_pattern"),
        "requires_user_confirmation": bool(result.get("requires_user_confirmation", False)),
        "errors": errors_list,
        "warnings": warnings_list,
        "frames": frames,
        "per_dir_results": per_dir_results,
    }
