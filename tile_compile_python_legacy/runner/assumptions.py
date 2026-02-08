"""
Methodik v3 assumptions configuration and validation.

Handles default values and reduced mode detection for the pipeline.
"""

from typing import Any

# Reduced mode thresholds (Methodik v3 §1.4)
FRAMES_MIN_DEFAULT = 50
FRAMES_OPTIMAL_DEFAULT = 800
FRAMES_REDUCED_THRESHOLD_DEFAULT = 200


def get_assumptions_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Get assumptions configuration with defaults."""
    assumptions = cfg.get("assumptions") if isinstance(cfg.get("assumptions"), dict) else {}
    return {
        "frames_min": int(assumptions.get("frames_min", FRAMES_MIN_DEFAULT)),
        "frames_optimal": int(assumptions.get("frames_optimal", FRAMES_OPTIMAL_DEFAULT)),
        "frames_reduced_threshold": int(assumptions.get("frames_reduced_threshold", FRAMES_REDUCED_THRESHOLD_DEFAULT)),
        "exposure_time_tolerance_percent": float(assumptions.get("exposure_time_tolerance_percent", 5.0)),
        "registration_residual_warn_px": float(assumptions.get("registration_residual_warn_px", 0.5)),
        "registration_residual_max_px": float(assumptions.get("registration_residual_max_px", 1.0)),
        "elongation_warn": float(assumptions.get("elongation_warn", 0.3)),
        "elongation_max": float(assumptions.get("elongation_max", 0.4)),
        "tracking_error_max_px": float(assumptions.get("tracking_error_max_px", 1.0)),
        "reduced_mode_skip_clustering": bool(assumptions.get("reduced_mode_skip_clustering", True)),
        "reduced_mode_cluster_range": assumptions.get("reduced_mode_cluster_range", [5, 10]),
    }


def is_reduced_mode(frame_count: int, assumptions: dict[str, Any]) -> bool:
    """Check if reduced mode should be activated (Methodik v3 §1.4)."""
    return frame_count < assumptions["frames_reduced_threshold"]
