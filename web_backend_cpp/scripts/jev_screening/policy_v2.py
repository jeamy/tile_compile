#!/usr/bin/env python3
"""Validate and read the pre-registered effect-based release policy (config/pi_decisions/release_policy_v2.json).

    policy_v2.py POLICY.json     validates the file and prints the frozen timestamp and the candidates

Also provides `session_valid(...)` (the pairing rules) and `candidate_verdict(...)` (the endpoint check for one candidate on
session-level results) for the evaluation script. It never decides a release: a verdict says whether the pre-registered
endpoint and safety bounds hold on the sessions given, nothing more.
"""
import json
import math
import sys
from datetime import datetime, timezone

CANDIDATE_STATUSES = {"pre_registered", "rejected_by_futility", "blocked"}
PARAMETER_CANDIDATE_KEYS = {"parameter", "control_level", "tested_level", "primary_endpoint", "noise_ratio_ci_upper_max", "safety", "sample"}
SAMPLE_KEYS = {"minimum_test_sessions", "minimum_object_groups", "minimum_sessions_per_object_group",
               "minimum_matched_stars_per_session", "minimum_valid_background_pixels_per_session"}
PAIRING_KEYS = ("identical_input_frames", "identical_build", "identical_registration_reference_frame", "identical_output_raster")


def validate(policy):
    if policy.get("schema_version") != "pi.jev-release-policy.v2":
        raise ValueError("unsupported release policy version")
    if policy.get("confidence_level") != 0.95 or policy.get("statistical_unit") != "independent_session":
        raise ValueError("unsupported confidence level or statistical unit")
    if policy.get("paired_session_ci_method") != "percentile_bootstrap" or policy.get("bootstrap_resamples") != 10000:
        raise ValueError("unsupported uncertainty procedure")
    if policy.get("frozen_before_test") is not True:
        raise ValueError("the policy must be frozen before the test")
    try:
        frozen = datetime.fromisoformat(policy["frozen_at_utc"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid policy freeze timestamp") from error
    if frozen.tzinfo is None or frozen.utcoffset() != timezone.utc.utcoffset(frozen):
        raise ValueError("policy freeze timestamp must be UTC")
    pairing = policy.get("pairing_rules", {})
    if not all(pairing.get(k) is True for k in PAIRING_KEYS):
        raise ValueError("pairing rules must require identical frames, build, reference frame and raster")
    futility = policy.get("futility_rule", {})
    if not isinstance(futility.get("minimum_evaluable_sessions"), int) or futility["minimum_evaluable_sessions"] < 2:
        raise ValueError("futility rule needs an integer minimum of evaluable sessions (>= 2)")
    candidates = policy.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("no candidates")
    for cid, c in candidates.items():
        status = c.get("status")
        if status not in CANDIDATE_STATUSES:
            raise ValueError("unknown candidate status: " + cid)
        if status != "pre_registered":
            if not (c.get("evidence") or c.get("reason")):
                raise ValueError("a rejected or blocked candidate needs its evidence or reason: " + cid)
            continue
        if not PARAMETER_CANDIDATE_KEYS.issubset(c):
            raise ValueError("incomplete pre-registered candidate: " + cid)
        sample = c["sample"]
        if not SAMPLE_KEYS.issubset(sample) or any(not isinstance(sample[k], int) or sample[k] <= 0 for k in SAMPLE_KEYS):
            raise ValueError("invalid sample sizes: " + cid)
        if sample["minimum_object_groups"] * sample["minimum_sessions_per_object_group"] > sample["minimum_test_sessions"]:
            raise ValueError("inconsistent session strata: " + cid)
        bound = c["noise_ratio_ci_upper_max"]
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) or not 0 < bound < 1:
            raise ValueError("the noise bound must be a number in (0, 1): " + cid)
        for key, value in c["safety"].items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid safety bound %s: %s" % (key, cid))
    return frozen


def session_valid(control_run, candidate_run):
    """Pairing rules on two run summaries {frames, build_id, ref_frame, raster}; returns (ok, reasons)."""
    reasons = []
    for key, reason in (("frames", "input_frames_differ"), ("build_id", "build_differs"),
                        ("ref_frame", "registration_reference_frame_differs"), ("raster", "output_raster_differs")):
        if control_run.get(key) != candidate_run.get(key):
            reasons.append(reason)
    return not reasons, reasons


def candidate_verdict(policy, candidate_id, sessions):
    """Endpoint and safety check for one pre-registered candidate.

    `sessions` is a list of dicts: {"group", "noise_ratio", "fwhm_ratio", "elongation_ratio", "signal_ratio", "gate_passed",
    "selected_unchanged", "n_eff_ratio", "matched_stars", "background_pixels"} (one per VALID paired session; the caller has
    already excluded invalid pairs). Returns {"holds": bool, "reasons": [...]}. The interval is a percentile bootstrap over
    sessions, seeded from the policy.
    """
    import numpy as np
    c = policy["candidates"][candidate_id]
    if c["status"] != "pre_registered":
        return {"holds": False, "reasons": ["candidate_status_" + c["status"]]}
    s, reasons = c["sample"], []
    if len(sessions) < s["minimum_test_sessions"]:
        reasons.append("too_few_sessions")
    groups = {}
    for x in sessions:
        groups[x["group"]] = groups.get(x["group"], 0) + 1
    if len([g for g, n in groups.items() if n >= s["minimum_sessions_per_object_group"]]) < s["minimum_object_groups"]:
        reasons.append("object_group_strata_not_met")
    if any(x["matched_stars"] < s["minimum_matched_stars_per_session"] or x["background_pixels"] < s["minimum_valid_background_pixels_per_session"] for x in sessions):
        reasons.append("session_sample_below_minimum")
    if any(not x["gate_passed"] for x in sessions):
        reasons.append("coverage_gate_failed")
    if any(not x["selected_unchanged"] for x in sessions):
        reasons.append("selected_candidate_changed")
    rng = np.random.default_rng(policy["bootstrap_seed"])

    def upper(key):
        v = np.array([x[key] for x in sessions], float)
        if v.size < 2:
            return float("inf")
        meds = np.median(v[rng.integers(0, v.size, (policy["bootstrap_resamples"], v.size))], axis=1)
        return float(np.percentile(meds, 97.5))

    def lower(key):
        v = np.array([x[key] for x in sessions], float)
        if v.size < 2:
            return float("-inf")
        meds = np.median(v[rng.integers(0, v.size, (policy["bootstrap_resamples"], v.size))], axis=1)
        return float(np.percentile(meds, 2.5))
    if sessions:
        if upper("noise_ratio") > c["noise_ratio_ci_upper_max"]:
            reasons.append("noise_endpoint_not_met")
        safety = c["safety"]
        if "star_fwhm_ratio_ci_upper_max" in safety and upper("fwhm_ratio") > safety["star_fwhm_ratio_ci_upper_max"]:
            reasons.append("star_shape_regression")
        if "elongation_ratio_ci_upper_max" in safety and upper("elongation_ratio") > safety["elongation_ratio_ci_upper_max"]:
            reasons.append("elongation_regression")
        if "star_signal_ratio_ci_lower_min" in safety and lower("signal_ratio") < safety["star_signal_ratio_ci_lower_min"]:
            reasons.append("signal_loss")
        if "star_signal_ratio_ci_upper_max" in safety and upper("signal_ratio") > safety["star_signal_ratio_ci_upper_max"]:
            reasons.append("signal_excess")
        if "n_eff_p10_ratio_min" in safety and any(x["n_eff_ratio"] < safety["n_eff_p10_ratio_min"] for x in sessions):
            reasons.append("coverage_worse")
    return {"holds": not reasons, "reasons": reasons}


def futility(policy, sessions):
    """True when at least the minimum number of evaluable sessions is in and the noise interval lies entirely above 1.0."""
    import numpy as np
    n = policy["futility_rule"]["minimum_evaluable_sessions"]
    if len(sessions) < n:
        return False
    rng = np.random.default_rng(policy["bootstrap_seed"])
    v = np.array([x["noise_ratio"] for x in sessions], float)
    meds = np.median(v[rng.integers(0, v.size, (policy["bootstrap_resamples"], v.size))], axis=1)
    return bool(np.percentile(meds, 2.5) > 1.0)


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    frozen = validate(p)
    print("valid; frozen at", frozen.isoformat(), "| candidates:", {k: v["status"] for k, v in p["candidates"].items()})
