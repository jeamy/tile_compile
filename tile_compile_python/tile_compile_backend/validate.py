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

    # ========================================================================
    # Methodik v3 §4.1 Testfälle (normativ)
    # ========================================================================

    # Testfall 4: Overlap-Konsistenz
    tile_overlap = _as_float(get_path(cfg, ["tile", "overlap_fraction"]))
    if tile_overlap is not None:
        if not (0 <= tile_overlap <= 0.5):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="overlap_fraction_invalid",
                    path="$.tile.overlap_fraction",
                    message="overlap_fraction must be in [0, 0.5] per Methodik v3 §3.3",
                )
            )

    # Testfall 3: Tile-Size Monotonie (via size_factor and min_size consistency)
    tile_size_factor = _as_float(get_path(cfg, ["tile", "size_factor"]))
    tile_min_size = _as_float(get_path(cfg, ["tile", "min_size"]))
    tile_max_divisor = _as_float(get_path(cfg, ["tile", "max_divisor"]))
    if tile_size_factor is not None and tile_size_factor <= 0:
        issues.append(
            ValidationIssue(
                severity="error",
                code="tile_size_factor_invalid",
                path="$.tile.size_factor",
                message="size_factor must be > 0",
            )
        )
    if tile_min_size is not None and tile_min_size < 1:
        issues.append(
            ValidationIssue(
                severity="error",
                code="tile_min_size_invalid",
                path="$.tile.min_size",
                message="min_size must be >= 1",
            )
        )
    if tile_max_divisor is not None and tile_max_divisor < 1:
        issues.append(
            ValidationIssue(
                severity="error",
                code="tile_max_divisor_invalid",
                path="$.tile.max_divisor",
                message="max_divisor must be >= 1",
            )
        )

    # Testfall 6: Kanaltrennung - warn if processing.channel_coupling is enabled
    channel_coupling = get_path(cfg, ["processing", "channel_coupling"])
    if channel_coupling is True:
        issues.append(
            ValidationIssue(
                severity="error",
                code="channel_coupling_forbidden",
                path="$.processing.channel_coupling",
                message="Channel coupling is forbidden per Methodik v3 §1 (kanalgetrennt)",
            )
        )

    # Testfall 7: Keine Frame-Selektion
    frame_selection = get_path(cfg, ["processing", "frame_selection"])
    if frame_selection is True:
        issues.append(
            ValidationIssue(
                severity="error",
                code="frame_selection_forbidden",
                path="$.processing.frame_selection",
                message="Frame selection is forbidden per Methodik v3 §1 (keine Frame-Selektion)",
            )
        )

    # ========================================================================
    # Methodik v3 §1.2 Assumption Tolerances
    # ========================================================================
    
    assumptions = get_path(cfg, ["assumptions"])
    if isinstance(assumptions, dict):
        # Validate frames_min < frames_reduced_threshold < frames_optimal
        frames_min = _as_float(assumptions.get("frames_min"))
        frames_reduced = _as_float(assumptions.get("frames_reduced_threshold"))
        frames_optimal = _as_float(assumptions.get("frames_optimal"))
        
        if frames_min is not None and frames_reduced is not None:
            if frames_min >= frames_reduced:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="frames_thresholds_invalid",
                        path="$.assumptions",
                        message=f"frames_min ({frames_min}) must be < frames_reduced_threshold ({frames_reduced})",
                    )
                )
        
        if frames_reduced is not None and frames_optimal is not None:
            if frames_reduced >= frames_optimal:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="frames_thresholds_invalid",
                        path="$.assumptions",
                        message=f"frames_reduced_threshold ({frames_reduced}) must be < frames_optimal ({frames_optimal})",
                    )
                )
        
        # Validate registration residual thresholds
        reg_warn = _as_float(assumptions.get("registration_residual_warn_px"))
        reg_max = _as_float(assumptions.get("registration_residual_max_px"))
        if reg_warn is not None and reg_max is not None:
            if reg_warn >= reg_max:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="registration_thresholds_invalid",
                        path="$.assumptions",
                        message=f"registration_residual_warn_px ({reg_warn}) must be < registration_residual_max_px ({reg_max})",
                    )
                )
        
        # Validate elongation thresholds
        elong_warn = _as_float(assumptions.get("elongation_warn"))
        elong_max = _as_float(assumptions.get("elongation_max"))
        if elong_warn is not None and elong_max is not None:
            if elong_warn >= elong_max:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="elongation_thresholds_invalid",
                        path="$.assumptions",
                        message=f"elongation_warn ({elong_warn}) must be < elongation_max ({elong_max})",
                    )
                )
        
        # Validate reduced mode cluster range
        reduced_cluster_range = get_path(cfg, ["assumptions", "reduced_mode_cluster_range"])
        if isinstance(reduced_cluster_range, list) and len(reduced_cluster_range) == 2:
            a = _as_float(reduced_cluster_range[0])
            b = _as_float(reduced_cluster_range[1])
            if a is not None and b is not None and b < a:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="reduced_mode_cluster_range_invalid",
                        path="$.assumptions.reduced_mode_cluster_range",
                        message="reduced_mode_cluster_range must satisfy [min, max] with max >= min",
                    )
                )

    valid = len([i for i in issues if i.severity == "error"]) == 0
    return {
        "valid": valid,
        "errors": [i.__dict__ for i in issues if i.severity == "error"],
        "warnings": [i.__dict__ for i in issues if i.severity == "warning"],
    }
