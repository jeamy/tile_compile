from __future__ import annotations

from typing import Any

from app.services.process_manager import InMemoryJobStore


def compute_guardrails(job_store: InMemoryJobStore) -> dict[str, Any]:
    items = job_store.list()
    scan_job = next((j for j in items if j.job_type == "scan"), None)
    if scan_job is None:
        return {
            "status": "check",
            "checks": [
                {"id": "scan", "status": "check", "label": "Scan ausstehend"},
            ],
        }

    result = scan_job.data.get("result", {})
    errors = result.get("errors", []) if isinstance(result, dict) else []
    warnings = result.get("warnings", []) if isinstance(result, dict) else []
    requires_confirm = bool(result.get("requires_user_confirmation", False)) if isinstance(result, dict) else False
    color_mode = str(result.get("color_mode", "UNKNOWN")).upper() if isinstance(result, dict) else "UNKNOWN"
    color_mode_check_status = "check" if requires_confirm or color_mode in {"", "UNKNOWN"} else "ok"
    color_mode_label = "Color mode bestaetigen" if color_mode_check_status == "check" else f"Color mode: {color_mode}"

    checks = [
        {
            "id": "scan_ok",
            "status": "ok" if not errors else "error",
            "label": "Scan erfolgreich" if not errors else "Scan mit Fehlern",
            "count": len(errors),
        },
        {
            "id": "color_mode",
            "status": color_mode_check_status,
            "label": color_mode_label,
            "value": color_mode,
        },
        {
            "id": "scan_warnings",
            "status": "check" if warnings else "ok",
            "label": "Warnungen vorhanden" if warnings else "Keine Scan-Warnungen",
            "count": len(warnings),
        },
    ]
    status = "error" if errors else ("check" if warnings else "ok")
    return {"status": status, "checks": checks}


def has_blocking_guardrail(guardrails: dict[str, Any]) -> bool:
    checks = guardrails.get("checks", [])
    return any(str(item.get("status", "")).lower() == "error" for item in checks if isinstance(item, dict))
