#!/usr/bin/env python3
"""PI local-learning offline retraining/export script.

docs/PI/pi_local_learning_plan_de.md, Abschnitt 7 Schritt 6 and Abschnitt 9 Punkt 2: the training
tooling may be Python (only the C++ *inference* side, PiMemoryStore/pi_param_model, has to avoid a
Python runtime dependency). This script does not "train" in the deep-learning sense — Abschnitt 4.2
deliberately chose a weighted nearest-neighbor model precisely so no training step is needed, only a
reference-point export. What this script does:

1. Read the PI memory store's JSONL files directly (no C++ dependency) and reconstruct the same
   merged view PiMemoryStore::list() produces in C++ (memories + latest-review-wins status +
   accumulated outcome history) — see _load_memories() for the exact mirrored logic.
2. For each (domain, target_path), select memories that carry a positive, verified outcome signal
   and export their {feature_vector, value} as pi_models/<domain>/<target_path>/v<N>/
   reference_points.jsonl, exactly the layout web_backend_cpp/src/services/pi/pi_param_model.cpp
   reads (see predict_param_nn()).
3. Score the new reference set via leave-one-out cross-validation using the SAME k-NN logic the C++
   side runs (see _predict_leave_one_out()) and only publish a new version if its score is not worse
   than the currently active version's stored score — the "Rollout-Schutz" from Abschnitt 4.3.
4. For domain "scan", pin the export to the current config_schema's SHA-256 (raw file bytes, same
   algorithm pi_param_model.cpp's compute_file_sha256() uses) so a model copied between installs
   never silently applies a value a newer schema no longer recognizes.

Usage:
    python3 scripts/pi_retrain_models.py --memory-dir <runs/.pi_memory> --models-dir <pi_models> \\
        --schema-path <tile_compile_cpp/tile_compile.schema.yaml>
    python3 scripts/pi_retrain_models.py --list-versions scan/bge.method
    python3 scripts/pi_retrain_models.py --rollback scan/bge.method

With no --memory-dir/--models-dir, both default relative to this script's repo root, matching the
C++ side's defaults (project_root/runs/.pi_memory, project_root/pi_models).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "pi.param-model-metadata.v1"

# Same PoC scope as the C++ side (pi_param_model.cpp log_scan_param_shadow_predictions()) —
# deliberately not every config_schema path.
SCAN_TARGET_PATHS = ["bge.method", "normalization.mode"]

# Minimum reference points before a version is published at all. Abschnitt 4.2 names ~20 as the
# point a nearest-neighbor model becomes useful; below MIN_PUBLISH_SAMPLES it is not worth writing
# a version that will barely differ from "no model" in practice — an intentionally low bar (not 20)
# so the export path itself is exercisable long before real usage volume arrives (Abschnitt 4.3
# "Bootstrap-Realismus").
MIN_PUBLISH_SAMPLES = 3


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                items.append(parsed)
    return items


def _load_memories(memory_dir: Path) -> list[dict]:
    """Mirrors PiMemoryStore::list()'s merge (pi_memory_store.cpp) exactly: latest review per
    memory_id overrides status/scope/outcome; outcomes accumulate into item["outcomes"]."""
    items = read_jsonl(memory_dir / "memories_v2.jsonl")

    latest_reviews: dict[str, dict] = {}
    for review in read_jsonl(memory_dir / "memory_reviews_v2.jsonl"):
        memory_id = review.get("memory_id", "")
        if memory_id:
            latest_reviews[memory_id] = review

    outcome_histories: dict[str, list[dict]] = {}
    for event in read_jsonl(memory_dir / "memory_outcomes_v2.jsonl"):
        memory_id = event.get("memory_id", "")
        if not memory_id:
            continue
        outcome_histories.setdefault(memory_id, []).append(event)

    for item in items:
        memory_id = item.get("memory_id", "")
        review = latest_reviews.get(memory_id)
        if review is not None:
            item["status"] = review.get("status", item.get("status", "candidate"))
            item["review"] = review
            if isinstance(review.get("scope"), dict):
                item["scope"] = review["scope"]
            if isinstance(review.get("outcome"), dict):
                item["outcome"] = review["outcome"]
        item["outcomes"] = outcome_histories.get(memory_id, [])

    return items


@dataclass
class ReferencePoint:
    feature_vector: dict
    value: Any


@dataclass
class ExportResult:
    domain: str
    target_path: str
    points: list[ReferencePoint] = field(default_factory=list)


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def collect_scan_reference_points(memories: list[dict]) -> dict[str, ExportResult]:
    """Eligibility: status == 'accepted'. NOT quality_delta > 0 — quality_delta is currently always
    null (docs/PI/pi_local_learning_plan_de.md, Abschnitt 0.3: same-frames delta computation does
    not exist yet). 'accepted' is the only reliable positive signal available end-to-end today
    (manual /review, or Schritt 2's auto-promotion once it leaves shadow mode). Once delta
    computation exists, this is the one line that should change."""
    results = {p: ExportResult(domain="scan", target_path=p) for p in SCAN_TARGET_PATHS}
    for memory in memories:
        if memory.get("type") != "config_optimization":
            continue
        if memory.get("status") != "accepted":
            continue
        context_signature = memory.get("context_signature")
        feature_vector = None
        if isinstance(context_signature, dict):
            feature_vector = context_signature.get("feature_vector")
        if not isinstance(feature_vector, dict):
            continue
        for update in memory.get("config_updates", []) or []:
            path = update.get("path")
            if path in results and "value" in update:
                results[path].points.append(ReferencePoint(feature_vector, update["value"]))
    return results


def collect_live_edit_reference_points(memories: list[dict]) -> dict[str, ExportResult]:
    """Eligibility: outcome.retained == true (Schritt 4 — known immediately at session close, no
    same-frames wait needed, unlike scan). Only numeric fields — Schritt 5's regression scope.
    Target paths are discovered from the data (op_type.field), not hardcoded, since live-edit
    operations are open-ended."""
    results: dict[str, ExportResult] = {}
    for memory in memories:
        if memory.get("type") != "live_edit_operation":
            continue
        context_signature = memory.get("context_signature")
        feature_vector = None
        if isinstance(context_signature, dict):
            feature_vector = context_signature.get("feature_vector")
        if not isinstance(feature_vector, dict):
            continue
        op_type = memory.get("op_type", "")
        if not op_type:
            continue
        updates = memory.get("config_updates", []) or []
        if not updates:
            continue
        params = updates[0].get("value")
        if not isinstance(params, dict):
            continue
        # retained is checked per outcome event, not just the latest-wins "outcome" field, so a
        # memory recorded once as retained is not silently dropped by a later, unrelated event.
        retained = any(
            isinstance(e.get("outcome"), dict) and e["outcome"].get("retained") is True
            for e in memory.get("outcomes", [])
        )
        if not retained:
            continue
        for field_name, value in params.items():
            if not _numeric(value):
                continue
            target_path = f"{op_type}.{field_name}"
            results.setdefault(target_path, ExportResult(domain="live_edit", target_path=target_path))
            results[target_path].points.append(ReferencePoint(feature_vector, value))
    return results


def _feature_vector_distance(a: dict, b: dict) -> float:
    """Mirrors feature_vector_distance() in pi_feature_vector.cpp — same formula, not literally the
    same code, so keep this in sync by hand if that function changes."""
    a_num = a.get("numeric", {}) or {}
    b_num = b.get("numeric", {}) or {}
    sum_sq = 0.0
    compared = 0
    for key, a_val in a_num.items():
        if not _numeric(a_val):
            continue
        b_val = b_num.get(key)
        if not _numeric(b_val):
            continue
        diff = a_val - b_val
        sum_sq += diff * diff
        compared += 1
    if compared == 0:
        return math.inf

    a_cat = a.get("categorical", {}) or {}
    b_cat = b.get("categorical", {}) or {}
    for key, a_val in a_cat.items():
        if key in b_cat and a_val != b_cat[key]:
            sum_sq += 4.0 * 4.0
    return math.sqrt(sum_sq)


def _predict_knn(points: list[ReferencePoint], query: dict, exclude_index: int | None = None, k: int = 5) -> Any:
    scored = []
    for i, point in enumerate(points):
        if i == exclude_index:
            continue
        distance = _feature_vector_distance(query, point.feature_vector)
        if math.isfinite(distance):
            scored.append((distance, point.value))
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0])
    neighbors = scored[:k]

    if all(_numeric(v) for _, v in neighbors):
        weighted_sum = sum((1.0 / (1.0 + d)) * v for d, v in neighbors)
        total_weight = sum(1.0 / (1.0 + d) for d, _ in neighbors)
        return weighted_sum / total_weight if total_weight > 0 else None

    votes: dict[Any, float] = {}
    for d, v in neighbors:
        key = json.dumps(v, sort_keys=True)
        votes[key] = votes.get(key, 0.0) + 1.0 / (1.0 + d)
    best_key = max(votes, key=votes.get) if votes else None
    return json.loads(best_key) if best_key is not None else None


def leave_one_out_score(points: list[ReferencePoint]) -> float | None:
    """Higher is better, comparable across versions of the SAME target_path only (classification
    accuracy in [0,1] and regression's 1/(1+MAE) in (0,1] are not on a shared scale in general, but
    a target_path's reference values don't change type between versions in practice)."""
    if len(points) < 2:
        return None
    numeric_target = all(_numeric(p.value) for p in points)
    if numeric_target:
        errors = []
        for i, point in enumerate(points):
            predicted = _predict_knn(points, point.feature_vector, exclude_index=i)
            if predicted is None:
                continue
            errors.append(abs(predicted - point.value))
        if not errors:
            return None
        mae = sum(errors) / len(errors)
        return 1.0 / (1.0 + mae)
    correct = 0
    total = 0
    for i, point in enumerate(points):
        predicted = _predict_knn(points, point.feature_vector, exclude_index=i)
        if predicted is None:
            continue
        total += 1
        if predicted == point.value:
            correct += 1
    return (correct / total) if total > 0 else None


def _existing_versions(target_dir: Path) -> list[tuple[int, Path]]:
    if not target_dir.is_dir():
        return []
    versions = []
    for entry in target_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("v"):
            continue
        try:
            versions.append((int(entry.name[1:]), entry))
        except ValueError:
            continue
    return sorted(versions)


def publish_version(
    models_dir: Path,
    domain: str,
    target_path: str,
    points: list[ReferencePoint],
    schema_sha256: str | None,
    dry_run: bool,
) -> str:
    target_dir = models_dir / domain / target_path
    existing = _existing_versions(target_dir)

    if len(points) < MIN_PUBLISH_SAMPLES:
        return f"skip: only {len(points)} eligible reference point(s), need >= {MIN_PUBLISH_SAMPLES}"

    new_score = leave_one_out_score(points)
    if new_score is None:
        return "skip: leave-one-out score could not be computed (need >= 2 comparable points)"

    if existing:
        _, latest_dir = existing[-1]

        # Unchanged data (the common case on a periodic re-run, Abschnitt 7: "periodisch oder nach
        # N neuen Outcomes" — most runs will find nothing new) must not publish a byte-identical
        # duplicate version. Order-independent: export order depends on JSONL iteration order, not
        # meaning. Checked before the score comparison below, which would otherwise let an unchanged
        # set through (new_score == old_score is not "worse").
        old_points = read_jsonl(latest_dir / "reference_points.jsonl")
        old_signature = {(json.dumps(p.get("feature_vector"), sort_keys=True), json.dumps(p.get("value"), sort_keys=True))
                         for p in old_points}
        new_signature = {(json.dumps(p.feature_vector, sort_keys=True), json.dumps(p.value, sort_keys=True))
                         for p in points}
        if old_signature == new_signature:
            return f"skip: unchanged since {latest_dir.name} (same {len(points)} reference points)"

        old_metadata_path = latest_dir / "metadata.json"
        old_score = None
        if old_metadata_path.is_file():
            try:
                old_score = json.loads(old_metadata_path.read_text(encoding="utf-8")).get("validation_score")
            except (json.JSONDecodeError, OSError):
                old_score = None
        if isinstance(old_score, (int, float)) and new_score < old_score:
            return (
                f"skip: new validation_score {new_score:.4f} would regress vs. "
                f"active {latest_dir.name}'s {old_score:.4f} (Rollout-Schutz, Abschnitt 4.3)"
            )

    next_version = (existing[-1][0] + 1) if existing else 1
    version_dir = target_dir / f"v{next_version}"

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "domain": domain,
        "target_path": target_path,
        "n_samples": len(points),
        "validation_score": new_score,
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if domain == "scan" and schema_sha256:
        metadata["config_schema_sha256"] = schema_sha256

    if dry_run:
        return f"would publish {domain}/{target_path}/v{next_version}: n={len(points)}, score={new_score:.4f}"

    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (version_dir / "reference_points.jsonl").open("w", encoding="utf-8") as f:
        for point in points:
            f.write(json.dumps({"feature_vector": point.feature_vector, "value": point.value}) + "\n")

    return f"published {domain}/{target_path}/v{next_version}: n={len(points)}, score={new_score:.4f}"


def cmd_retrain(args: argparse.Namespace) -> int:
    memory_dir = args.memory_dir
    models_dir = args.models_dir
    memories = _load_memories(memory_dir)
    if not memories:
        print(f"No memories found in {memory_dir} — nothing to export "
              f"(Abschnitt 4.3 'Bootstrap-Realismus': expected on a fresh or low-usage install).")

    schema_sha256 = None
    if args.schema_path and args.schema_path.is_file():
        schema_sha256 = hashlib.sha256(args.schema_path.read_bytes()).hexdigest()
    elif args.schema_path:
        print(f"Warning: schema path {args.schema_path} not found — scan exports will be unpinned.",
              file=sys.stderr)

    scan_results = collect_scan_reference_points(memories)
    live_edit_results = collect_live_edit_reference_points(memories)

    any_published = False
    for result in list(scan_results.values()) + list(live_edit_results.values()):
        outcome = publish_version(models_dir, result.domain, result.target_path, result.points,
                                  schema_sha256, args.dry_run)
        print(f"[{result.domain}/{result.target_path}] {outcome}")
        if outcome.startswith("published") or outcome.startswith("would publish"):
            any_published = True

    if not any_published:
        print("No versions published this run — either not enough eligible data yet, or nothing "
              "improved on the currently active version. Both are expected outcomes, not failures.")
    return 0


def cmd_list_versions(args: argparse.Namespace) -> int:
    domain, _, target_path = args.list_versions.partition("/")
    target_dir = args.models_dir / domain / target_path
    versions = _existing_versions(target_dir)
    if not versions:
        print(f"No versions found under {target_dir}")
        return 0
    for version, path in versions:
        metadata_path = path / "metadata.json"
        info = ""
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                info = f" n_samples={metadata.get('n_samples')} validation_score={metadata.get('validation_score')}"
            except (json.JSONDecodeError, OSError):
                pass
        print(f"v{version}{info}  ({path})")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """Deletes the highest version directory, so predict_param_nn() (which always picks the highest
    qualifying version) falls back to the previous one — or to 'no model' if none remain."""
    domain, _, target_path = args.rollback.partition("/")
    target_dir = args.models_dir / domain / target_path
    versions = _existing_versions(target_dir)
    if not versions:
        print(f"No versions to roll back under {target_dir}")
        return 1
    version, path = versions[-1]
    if len(versions) == 1:
        print(f"Rolling back v{version} would leave no model for {domain}/{target_path} — "
              f"predict_param_nn() will report 'no_model' afterwards.")
    if args.dry_run:
        print(f"Would remove {path}")
        return 0
    import shutil
    shutil.rmtree(path)
    print(f"Removed {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--memory-dir", type=Path, default=repo_root / "runs" / ".pi_memory",
                        help="PiMemoryStore directory (default: <repo>/runs/.pi_memory)")
    parser.add_argument("--models-dir", type=Path, default=repo_root / "pi_models",
                        help="pi_models directory (default: <repo>/pi_models)")
    parser.add_argument("--schema-path", type=Path,
                        default=repo_root / "tile_compile_cpp" / "tile_compile.schema.yaml",
                        help="config schema file to pin scan-domain exports against")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, do not write anything")
    parser.add_argument("--list-versions", metavar="DOMAIN/TARGET_PATH",
                        help="list existing versions for one target instead of retraining")
    parser.add_argument("--rollback", metavar="DOMAIN/TARGET_PATH",
                        help="remove the highest version for one target instead of retraining")
    args = parser.parse_args()

    if args.list_versions:
        return cmd_list_versions(args)
    if args.rollback:
        return cmd_rollback(args)
    return cmd_retrain(args)


if __name__ == "__main__":
    raise SystemExit(main())
