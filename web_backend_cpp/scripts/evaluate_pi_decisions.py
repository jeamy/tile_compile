#!/usr/bin/env python3
"""Read-only Jev replay and evidence inventory. Never changes source decisions or runs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import yaml


ABSTAIN = {"keep_current", "insufficient_evidence"}
VALIDATION_KEYS = ("config_ok", "evidence_ok", "policy_ok", "schema_ok")
COMMON_RELEASE_FIELDS = {
    "minimum_test_sessions", "minimum_object_groups", "minimum_sessions_per_object_group",
    "minimum_coverage", "maximum_coverage_loss", "maximum_invalid_patch_rate",
    "maximum_star_shape_regression", "maximum_noise_regression", "maximum_signal_loss",
    "minimum_quality_improvement", "minimum_beneficial_cases", "minimum_unsafe_cases",
    "minimum_beneficial_case_selection_rate", "minimum_correct_unsafe_abstention_rate",
}
COUNT_FIELDS = {"minimum_test_sessions", "minimum_object_groups", "minimum_sessions_per_object_group",
                "minimum_matched_stars_per_session", "minimum_valid_background_pixels_per_session",
                "minimum_unsaturated_signal_apertures_per_session", "minimum_reference_patches_per_session",
                "minimum_beneficial_cases", "minimum_unsafe_cases"}


def validate_release_policy(policy):
    if policy.get("schema_version") != "pi.jev-release-policy.v1":
        raise ValueError("unsupported release policy version")
    if policy.get("confidence_level") != 0.95 or policy.get("statistical_unit") != "independent_session":
        raise ValueError("unsupported confidence level or statistical unit")
    if (policy.get("paired_session_ci_method") != "percentile_bootstrap" or
            policy.get("bootstrap_resamples") != 10000 or
            policy.get("bootstrap_seed") != 20260925 or
            policy.get("decision_rate_ci_method") != "wilson"):
        raise ValueError("unsupported uncertainty procedure")
    if not isinstance(policy.get("frozen_before_test"), bool):
        raise ValueError("invalid policy freeze flag")
    try:
        frozen_at = datetime.fromisoformat(policy["frozen_at_utc"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid policy freeze timestamp") from error
    if frozen_at.tzinfo is None or frozen_at.utcoffset() != timezone.utc.utcoffset(frozen_at):
        raise ValueError("policy freeze timestamp must be UTC")
    if set(policy.get("candidates", {})) != {"enable_adaptive_weights", "set_sensor_profile_dwarf_ii"}:
        raise ValueError("release policy candidate set mismatch")
    for candidate_id, thresholds in policy.get("candidates", {}).items():
        required = COMMON_RELEASE_FIELDS | ({"minimum_matched_stars_per_session",
            "minimum_valid_background_pixels_per_session", "minimum_unsaturated_signal_apertures_per_session",
            "maximum_elongation_regression"} if candidate_id == "enable_adaptive_weights" else {
            "minimum_reference_patches_per_session", "maximum_reference_luminance_error",
            "maximum_display_clipping_increase", "maximum_shadow_contrast_loss",
            "maximum_highlight_contrast_loss", "minimum_exact_camera_match_rate"})
        if not required.issubset(thresholds) or not isinstance(thresholds.get("primary_endpoint"), str):
            raise ValueError(f"incomplete release thresholds: {candidate_id}")
        for key in required:
            value = thresholds[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"non-finite or non-numeric release threshold: {candidate_id}/{key}")
            if key in COUNT_FIELDS:
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"invalid count threshold: {candidate_id}/{key}")
            elif not 0 <= value <= 1:
                raise ValueError(f"threshold outside [0,1]: {candidate_id}/{key}")
        if thresholds["maximum_invalid_patch_rate"] != 0:
            raise ValueError(f"invalid patches cannot be allowed: {candidate_id}")
    if (thresholds["minimum_object_groups"] * thresholds["minimum_sessions_per_object_group"] >
                thresholds["minimum_test_sessions"]):
            raise ValueError(f"inconsistent session strata: {candidate_id}")


def enrolled_after_freeze(session, frozen_at_utc):
    try:
        enrolled = datetime.fromisoformat(session["enrolled_at_utc"].replace("Z", "+00:00"))
        frozen = datetime.fromisoformat(frozen_at_utc.replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError):
        return False
    return enrolled.tzinfo is not None and enrolled > frozen


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0, mid - half), min(1, mid + half)]


def dotted_value(document, key):
    value = document
    for component in key.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"config assertion path absent: {key}")
        value = value[component]
    return value


def load_decision_documents(decisions_dir):
    """Read Jev documents from the live SQLite store, or an exported JSON tree."""
    database_path = decisions_dir.parent / "pi_store_v2.sqlite"
    if database_path.is_file():
        uri = database_path.resolve().as_uri() + "?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                raw_rows = connection.execute(
                    "SELECT proposal_id, part, json FROM jev_documents ORDER BY proposal_id, part"
                ).fetchall()
        except sqlite3.Error as error:
            raise ValueError(f"cannot read Jev SQLite store read-only: {error}") from error
        documents = {}
        source_digests = {}
        for proposal_id, part, raw in raw_rows:
            try:
                documents.setdefault(proposal_id, {})[part] = json.loads(raw)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid Jev document JSON: {proposal_id}/{part}") from error
            source_digests.setdefault(proposal_id, {})[part] = (
                "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
            )
        return documents, source_digests

    documents = {}
    source_digests = {}
    for directory in sorted(decisions_dir.iterdir()):
        if not directory.is_dir():
            continue
        parts = {}
        digests = {}
        for name in ("status", "proposal", "state", "candidates", "request", "response", "outcome"):
            path = directory / (name + ".json")
            if path.is_file():
                parts[name] = load(path)
                digests[name] = digest(path)
        if parts:
            documents[directory.name] = parts
            source_digests[directory.name] = digests
    return documents, source_digests


def canonical_config_digest(path):
    return digest(path)


def decision_run_links(proposal_id, proposal, outcome, runs_dir):
    """Join a proposal outcome to immutable run-start provenance.

    A stored outcome is only a claim. It becomes a verified link when the run
    provenance names this proposal and the exact saved revision, the run's
    config bytes match its provenance digest, all candidate values are
    present, and the run ended successfully.
    """
    if not isinstance(outcome, dict):
        return [], "unobserved"
    runs = outcome.get("runs", [])
    if not isinstance(runs, list):
        return [], "invalid_outcome"
    saved_revisions = proposal.get("saved_revision_ids", [])
    if not isinstance(saved_revisions, list):
        saved_revisions = []

    links = []
    for recorded in runs:
        if not isinstance(recorded, dict):
            links.append({"status": "invalid", "reason": "malformed_outcome_run"})
            continue
        run_id = recorded.get("run_id")
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
            links.append({"status": "invalid", "reason": "unsafe_run_id"})
            continue
        run_dir = runs_dir / run_id
        try:
            resolved_run_dir = run_dir.resolve(strict=True)
            resolved_root = runs_dir.resolve(strict=True)
            if not resolved_run_dir.is_relative_to(resolved_root):
                raise ValueError("run_path_outside_runs_root")
            pi_provenance = load(resolved_run_dir / "artifacts" / "pi_run_provenance.json")
            config_path = resolved_run_dir / "config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config_hash = canonical_config_digest(config_path)
            run_events_path = resolved_run_dir / "logs" / "run_events.jsonl"
            events = [json.loads(line) for line in run_events_path.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
            terminal = next((event for event in reversed(events) if event.get("type") == "run_end"), None)
            revision_id = recorded.get("config_revision_id")
            provenance_revision = pi_provenance.get("config_revision_id")
            provenance_proposal = pi_provenance.get("jev_proposal_id")
            provenance_hash = pi_provenance.get("config_sha256")
            hash_matches = provenance_hash in {config_hash, config_hash.removeprefix("sha256:")}
            config_values_match = all(
                dotted_value(config, update["path"]) == update.get("value")
                for update in proposal.get("updates", [])
            )
            checks = {
                "proposal_id_matches": provenance_proposal == proposal_id,
                "revision_id_matches": bool(revision_id) and revision_id == provenance_revision,
                "revision_was_saved_for_proposal": bool(revision_id) and revision_id in saved_revisions,
                "config_hash_matches_provenance": hash_matches,
                "proposal_values_present": config_values_match,
                "outcome_attribution_has_all_paths": recorded.get("attribution") == "paths_present",
                "run_succeeded": bool(terminal and terminal.get("success") is True),
            }
            verified = all(checks.values())
            links.append({
                "run_id": run_id,
                "status": "verified" if verified else "invalid",
                "config_revision_id": revision_id,
                "run_config_hash": config_hash,
                "input_manifest_hash": (load(resolved_run_dir / "artifacts" / "run_provenance.json")
                                        .get("input_manifest", {}).get("sha256")),
                "checks": checks,
                "reason": None if verified else "run_provenance_link_check_failed",
            })
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError) as error:
            links.append({"run_id": run_id, "status": "invalid", "reason": str(error)})
    status = "verified" if links and all(link.get("status") == "verified" for link in links) else (
        "invalid" if links else "unobserved")
    return links, status


def replay(decisions_dir, runs_dir):
    rows = []
    records, document_digests = load_decision_documents(decisions_dir)
    required = ("status", "proposal", "state", "candidates", "request", "response")
    for proposal_id, documents in sorted(records.items()):
        if not any(name in documents for name in required):
            continue
        if not all(name in documents for name in required):
            raise ValueError(f"incomplete decision record: {proposal_id}")
        status, proposal, state, candidates, request, response = (documents[name] for name in required)
        selected = response.get("selection") or {}
        candidate_id = proposal.get("candidate_id")
        allowed = {item["candidate_id"]: item for item in candidates.get("applicable", [])}
        state_hash = state.get("state_hash")
        hash_consistent = bool(state_hash and state_hash == request.get("state_hash")
                               and state_hash == response.get("state_hash")
                               and state_hash == proposal.get("state_hash"))
        selected_consistent = (selected.get("candidate_id") == candidate_id
                               if response.get("status") == "ok" else True)
        applicable = candidate_id in allowed
        expected_updates = allowed.get(candidate_id, {}).get("updates", [])
        updates_match = proposal.get("updates", []) == expected_updates if applicable else False
        validation = proposal.get("validation") or {}
        validation_ok = all(validation.get(key) is True for key in VALIDATION_KEYS)
        run_links, run_link_status = decision_run_links(
            proposal_id, proposal, documents.get("outcome"), runs_dir)
        verified_run_hashes = [link["run_config_hash"] for link in run_links
                               if link.get("status") == "verified" and link.get("run_config_hash")]
        invalid_applicable_patch = (candidate_id not in ABSTAIN and
                                    (not hash_consistent or not selected_consistent or
                                     not applicable or not updates_match or not validation_ok))
        probability = (selected.get("probabilities") or {}).get(candidate_id)
        rows.append({
            "proposal_id": proposal_id,
            "source_digests": document_digests.get(proposal_id, {}),
            "candidate_id": candidate_id,
            "proposal_status": proposal.get("status"),
            "decision_status": status.get("state"),
            "model_requested": (proposal.get("model") or {}).get("requested"),
            "model_reported": (proposal.get("model") or {}).get("provider_reported"),
            "model_response_digest": document_digests.get(proposal_id, {}).get("response"),
            "policy_version": proposal.get("policy_version"),
            "decision_policy_digest": document_digests.get(proposal_id, {}).get("candidates"),
            "state_hash": state_hash,
            "catalog_version": candidates.get("catalog_version"),
            "hash_consistent": hash_consistent,
            "selected_consistent": selected_consistent,
            "applicable": applicable,
            "updates_match": updates_match,
            "validation_ok": validation_ok,
            "invalid_applicable_patch": invalid_applicable_patch,
            "abstained": candidate_id in ABSTAIN,
            "selection_probability": probability,
            "user_acceptance": ("saved_config" if proposal.get("saved_revision_ids") else
                                "draft_applied" if proposal.get("status") == "applied_to_draft" else
                                "not_observed"),
            "draft_config_hash_after_apply": (proposal.get("config_hash_after") if proposal.get("status") in
                                              {"applied_to_draft", "saved", "run_started"} else None),
            "run_links": run_links,
            "run_link_status": run_link_status,
            "run_config_hash": verified_run_hashes[0] if len(verified_run_hashes) == 1 else None,
            "run_config_hashes": verified_run_hashes,
            "quality_label": None,
            "quality_label_reason": "matched-position and comparable-area measurements unavailable",
        })
    return rows


def memory_inventory(path):
    if path is None:
        return None
    counts = {}
    total = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid PiMemoryStore JSONL line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"invalid PiMemoryStore record line {line_number}")
            total += 1
            status = str(record.get("status", "missing"))
            counts[status] = counts.get(status, 0) + 1
    return {"record_count": total, "status_counts": counts,
            "note": "inventory only; no unverified memory outcome is used as a quality label"}


def evidence(registry, runs_dir):
    sessions = []
    split_by_input = {}
    split_by_frame = {}
    session_ids = set()
    for entry in registry["sessions"]:
        sid = entry["session_id"]
        split = entry["split"]
        if sid in session_ids or split not in ("calibration", "test"):
            raise ValueError(f"invalid or duplicate session/split: {sid}")
        session_ids.add(sid)
        arms = {}
        if set(entry.get("arm_assertions", {})) != set(entry["arms"]):
            raise ValueError(f"missing config arm assertions: {sid}")
        for arm_name, run_id in entry["arms"].items():
            if Path(run_id).name != run_id or not run_id:
                raise ValueError(f"unsafe run id: {run_id}")
            root = runs_dir / run_id / "artifacts"
            config_path = runs_dir / run_id / "config.yaml"
            with config_path.open(encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
            for key, expected in entry["arm_assertions"][arm_name].items():
                if dotted_value(config, key) != expected:
                    raise ValueError(f"config arm assertion failed: {sid}/{arm_name}/{key}")
            provenance = load(root / "run_provenance.json")
            fd = load(root / "forward_drizzle.json")
            input_hash = provenance["input_manifest"]["sha256"]
            entries = provenance["input_manifest"].get("entries")
            if not entries or len(entries) != provenance["input_manifest"]["entry_count"]:
                raise ValueError(f"missing or incomplete frame manifest: {run_id}")
            previous = split_by_input.setdefault(input_hash, split)
            if previous != split:
                raise ValueError(f"input frames cross calibration/test: {sid}")
            for frame in entries:
                frame_hash = frame.get("sha256")
                if not frame_hash:
                    raise ValueError(f"frame hash missing: {run_id}")
                previous_frame_split = split_by_frame.setdefault(frame_hash, split)
                if previous_frame_split != split:
                    raise ValueError(f"frame crosses calibration/test: {sid}")
            validation = fd.get("validation", {})
            selected = fd.get("selected_candidate")
            selected_metrics = validation.get(selected, {})
            arms[arm_name] = {
                "run_id": run_id,
                "config_digest": digest(config_path),
                "provenance_digest": digest(root / "run_provenance.json"),
                "quality_digest": digest(root / "forward_drizzle.json"),
                "input_sha256": input_hash,
                "input_count": provenance["input_manifest"]["entry_count"],
                "build_id": provenance["build"]["build_id"],
                "execution_scope": provenance.get("execution_scope"),
                "selected_candidate": selected,
                "selection_reason": fd.get("selection_reason"),
                "pixels_supported": fd.get("pixels_supported"),
                "candidate_gates": {
                    name: {key: validation.get(name, {}).get(key) for key in
                           ("support_ok", "numerics_ok")}
                    for name in ("drizzle_raw", "drizzle_uniform", "drizzle_multiband")
                },
                "selected_gates": {key: selected_metrics.get(key) for key in
                                   ("support_ok", "numerics_ok")},
                "descriptive_unmatched_metrics": {key: selected_metrics.get(key) for key in
                                                  ("median_fwhm", "elongation", "background_rms", "tail")},
            }
        paired = (len(arms) == 2 and len({arm["input_sha256"] for arm in arms.values()}) == 1
                  and len({arm["input_count"] for arm in arms.values()}) == 1
                  and len({arm["build_id"] for arm in arms.values()}) == 1
                  and len({arm["execution_scope"] for arm in arms.values()}) == 1)
        selected_gates_pass = all(arm["selected_gates"] ==
                                  {"support_ok": True, "numerics_ok": True} for arm in arms.values())
        sessions.append({**entry, "arms": arms, "input_and_build_paired": paired,
                         "selected_gates_pass": selected_gates_pass,
                         "matched_star_quality_available": False,
                         "quality_label": None,
                         "quality_label_reason": "matched-position/area measurements unavailable"})
    return sessions


def evaluate(decisions_dir, registry_path, policy_path, runs_dir, memory_path=None):
    registry, policy = load(registry_path), load(policy_path)
    if registry.get("schema_version") != "pi.jev-evaluation-registry.v1":
        raise ValueError("unsupported registry version")
    validate_release_policy(policy)
    rows = replay(decisions_dir, runs_dir)
    sessions = evidence(registry, runs_dir)
    invalid = sum(row["invalid_applicable_patch"] for row in rows)
    abstained = sum(row["abstained"] for row in rows)
    link_status_counts = {status: sum(row["run_link_status"] == status for row in rows)
                          for status in ("verified", "invalid", "unobserved")}
    verified_run_link_count = sum(
        link.get("status") == "verified"
        for row in rows for link in row.get("run_links", []))
    eligible = {}
    for candidate_id, thresholds in policy["candidates"].items():
        reasons = ["quality_policy_not_computed"]
        if not policy["frozen_before_test"]:
            reasons.append("policy_not_frozen_before_test")
        if any(value is None for value in thresholds.values()):
            reasons.append("numeric_thresholds_incomplete")
        if registry["study_kind"] != "prospective_blinded":
            reasons.append("no_prospective_blinded_test")
        relevant = [s for s in sessions if s["candidate_id"] == candidate_id and s["split"] == "test"]
        if not relevant:
            reasons.append("no_test_sessions")
        if any(not enrolled_after_freeze(s, policy["frozen_at_utc"]) for s in relevant):
            reasons.append("test_session_not_enrolled_after_policy_freeze")
        if any(not s["input_and_build_paired"] for s in relevant):
            reasons.append("unpaired_test_inputs_or_build")
        if any(not s["selected_gates_pass"] for s in relevant):
            reasons.append("selected_quality_gate_failed_or_missing")
        if any(not s["matched_star_quality_available"] for s in relevant):
            reasons.append("matched_quality_missing")
        if invalid:
            reasons.append("invalid_applicable_patch_observed")
        eligible[candidate_id] = {"release_eligible": not reasons, "blocking_reasons": reasons}
    return {
        "schema_version": "pi.jev-evaluation-report.v1",
        "source_digests": {"registry": digest(registry_path), "release_policy": digest(policy_path),
                           "pi_memory_store": digest(memory_path) if memory_path else None},
        "pi_memory_inventory": memory_inventory(memory_path),
        "decision_replay": rows,
        "decision_metrics": {
            "count": len(rows), "invalid_applicable_patch_count": invalid,
            "invalid_applicable_patch_rate": invalid / len(rows) if rows else None,
            "invalid_applicable_patch_rate_ci95": wilson(invalid, len(rows)),
            "abstention_count": abstained,
            "abstention_rate": abstained / len(rows) if rows else None,
            "abstention_rate_ci95": wilson(abstained, len(rows)),
            "candidate_coverage": {cid: sum(row["candidate_id"] == cid for row in rows) for cid in
                                   sorted({row["candidate_id"] for row in rows})},
            "calibration": None,
            "calibration_reason": "no ground-truth outcome labels for proposal probabilities",
            "quality_delta": None,
            "quality_delta_reason": "matched-position and comparable-area measurements unavailable",
            "run_linkage": {
                "proposal_count_by_status": link_status_counts,
                "verified_run_link_count": verified_run_link_count,
                "note": "run identity, saved config revision, exact run config digest, applied values, and successful terminal event are checked; no quality label is inferred",
            },
        },
        "baselines": {
            "current_config": "paired run artifacts where registered; not a Jev trial",
            "rules_only": "not observed",
            "existing_pi_advice": "not observed",
            "rules_plus_jev": "decision records only; no linked controlled quality trial",
            "knn_optional": "not observed",
        },
        "registry_study_kind": registry["study_kind"],
        "sessions": sessions,
        "candidate_release": eligible,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--pi-memory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    for source in (args.decisions_dir, args.runs_dir):
        if output.is_relative_to(source.resolve()):
            parser.error("report output must be outside decision and run stores")
    for source in (args.registry, args.policy, args.pi_memory):
        if source and output == source.resolve():
            parser.error("report output must not replace an input")
    report = evaluate(args.decisions_dir, args.registry, args.policy, args.runs_dir, args.pi_memory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=".jev-report-", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temp, output)
    print(f"wrote {output}; decisions={len(report['decision_replay'])}; "
          f"release_eligible={sum(x['release_eligible'] for x in report['candidate_release'].values())}")


if __name__ == "__main__":
    main()
