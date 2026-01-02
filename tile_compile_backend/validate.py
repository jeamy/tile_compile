from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from tile_compile_backend.schema import load_schema_json


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    code: str
    path: str
    message: str


def _json_path(parts: list[str | int]) -> str:
    if not parts:
        return "$"
    out = "$"
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            # minimal escaping
            if p.isidentifier():
                out += f".{p}"
            else:
                out += f"['{p}']"
    return out


def _sum_close_to_one(values: list[float], eps: float = 1e-6) -> bool:
    return abs(sum(values) - 1.0) <= eps


def _as_float(x: Any) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def validate_config_yaml_text(
    yaml_text: str,
    schema_path: str | None = None,
) -> dict:
    issues: list[ValidationIssue] = []

    try:
        cfg = yaml.safe_load(yaml_text)
    except Exception as e:  # noqa: BLE001
        issues.append(
            ValidationIssue(
                severity="error",
                code="yaml_parse_error",
                path="$",
                message=str(e),
            )
        )
        return {
            "valid": False,
            "errors": [i.__dict__ for i in issues if i.severity == "error"],
            "warnings": [i.__dict__ for i in issues if i.severity == "warning"],
        }

    if not isinstance(cfg, dict):
        issues.append(
            ValidationIssue(
                severity="error",
                code="config_not_object",
                path="$",
                message="configuration root must be a mapping/object",
            )
        )
        return {
            "valid": False,
            "errors": [i.__dict__ for i in issues if i.severity == "error"],
            "warnings": [i.__dict__ for i in issues if i.severity == "warning"],
        }

    schema = load_schema_json(schema_path)
    validator = Draft202012Validator(schema)

    for err in sorted(validator.iter_errors(cfg), key=lambda e: list(e.path)):
        issues.append(
            ValidationIssue(
                severity="error",
                code="schema_validation_error",
                path=_json_path(list(err.path)),
                message=err.message,
            )
        )

    # Cross-field checks (hard errors)
    def get_path(obj: dict, keys: list[str]) -> Any:
        cur: Any = obj
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    # frames max/min
    s_frames_min = _as_float(get_path(cfg, ["synthetic", "frames_min"]))
    s_frames_max = _as_float(get_path(cfg, ["synthetic", "frames_max"]))
    if s_frames_min is not None and s_frames_max is not None and s_frames_max < s_frames_min:
        issues.append(
            ValidationIssue(
                severity="error",
                code="synthetic_frames_range_invalid",
                path="$.synthetic",
                message="synthetic.frames_max must be >= synthetic.frames_min",
            )
        )

    d_frames_min = _as_float(get_path(cfg, ["data", "frames_min"]))
    d_frames_target = _as_float(get_path(cfg, ["data", "frames_target"]))
    if d_frames_min is not None and d_frames_target is not None and d_frames_target < d_frames_min:
        issues.append(
            ValidationIssue(
                severity="error",
                code="data_frames_target_invalid",
                path="$.data",
                message="data.frames_target must be >= data.frames_min",
            )
        )

    # clamp ordering
    def check_clamp(path: str, arr: Any, code: str) -> None:
        if not isinstance(arr, list) or len(arr) != 2:
            return
        a = _as_float(arr[0])
        b = _as_float(arr[1])
        if a is None or b is None:
            return
        if not a < b:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code=code,
                    path=path,
                    message="clamp must satisfy clamp[0] < clamp[1]",
                )
            )

    check_clamp("$.global_metrics.clamp", get_path(cfg, ["global_metrics", "clamp"]), "global_clamp_invalid")
    check_clamp("$.local_metrics.clamp", get_path(cfg, ["local_metrics", "clamp"]), "local_clamp_invalid")

    # weight normalization
    def check_weights_sum(path: str, weights: Any, keys: list[str], code: str) -> None:
        if not isinstance(weights, dict):
            return
        vals: list[float] = []
        for k in keys:
            v = _as_float(weights.get(k))
            if v is None:
                return
            vals.append(v)
        if not _sum_close_to_one(vals):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code=code,
                    path=path,
                    message=f"weights must be normalized (sum = 1.0). got sum={sum(vals):.6f}",
                )
            )

    check_weights_sum(
        "$.global_metrics.weights",
        get_path(cfg, ["global_metrics", "weights"]),
        ["background", "noise", "gradient"],
        "global_weights_not_normalized",
    )

    check_weights_sum(
        "$.local_metrics.star_mode.weights",
        get_path(cfg, ["local_metrics", "star_mode", "weights"]),
        ["fwhm", "roundness", "contrast"],
        "local_star_weights_not_normalized",
    )

    sm = get_path(cfg, ["local_metrics", "structure_mode"])
    if isinstance(sm, dict):
        bw = _as_float(sm.get("background_weight"))
        mw = _as_float(sm.get("metric_weight"))
        if bw is not None and mw is not None and not _sum_close_to_one([bw, mw]):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="local_structure_weights_not_normalized",
                    path="$.local_metrics.structure_mode",
                    message=f"background_weight + metric_weight must sum to 1.0. got sum={(bw + mw):.6f}",
                )
            )

    # synthetic: cluster_count_range sanity if present
    ccr = get_path(cfg, ["synthetic", "clustering", "cluster_count_range"])
    if isinstance(ccr, list) and len(ccr) == 2:
        a = _as_float(ccr[0])
        b = _as_float(ccr[1])
        if a is not None and b is not None and b < a:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="cluster_count_range_invalid",
                    path="$.synthetic.clustering.cluster_count_range",
                    message="cluster_count_range must satisfy [min, max] with max >= min",
                )
            )

    valid = len([i for i in issues if i.severity == "error"]) == 0
    return {
        "valid": valid,
        "errors": [i.__dict__ for i in issues if i.severity == "error"],
        "warnings": [i.__dict__ for i in issues if i.severity == "warning"],
    }
