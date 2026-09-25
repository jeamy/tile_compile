import json
from pathlib import Path
import tempfile
import unittest

from evaluate_pi_decisions import (enrolled_after_freeze, evaluate, memory_inventory,
                                   validate_release_policy)


CANDIDATE = "enable_adaptive_weights"
POLICY_SOURCE = Path(__file__).resolve().parents[1] / "config/pi_decisions/release_policy_v1.json"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.decisions = self.root / "decisions"
        self.record = self.decisions / "dec_test"
        self.record.mkdir(parents=True)
        self.registry = self.root / "registry.json"
        self.policy = self.root / "policy.json"
        self.runs = self.root / "runs"
        write(self.record / "status.json", {"state": "ready"})
        write(self.record / "state.json", {"state_hash": "sha256:test"})
        write(self.record / "request.json", {"state_hash": "sha256:test"})
        write(self.record / "response.json", {"status": "ok", "state_hash": "sha256:test",
                                              "selection": {"candidate_id": CANDIDATE}})
        write(self.record / "candidates.json", {"catalog_version": 1,
            "applicable": [{"candidate_id": CANDIDATE, "updates": [{"path": "safe", "value": True}]}]})
        write(self.record / "proposal.json", {"candidate_id": CANDIDATE, "status": "ready",
            "state_hash": "sha256:test", "updates": [{"path": "safe", "value": True}],
            "validation": {key: True for key in
                           ("config_ok", "evidence_ok", "policy_ok", "schema_ok")}})
        write(self.registry, {"schema_version": "pi.jev-evaluation-registry.v1",
                              "study_kind": "retrospective_exploratory", "sessions": []})
        write(self.policy, json.loads(POLICY_SOURCE.read_text(encoding="utf-8")))

    def run_eval(self):
        return evaluate(self.decisions, self.registry, self.policy, self.runs)

    def test_valid_replay_stays_unreleased(self):
        report = self.run_eval()
        self.assertEqual(report["decision_metrics"]["invalid_applicable_patch_count"], 0)
        self.assertEqual(report["decision_replay"][0]["user_acceptance"], "not_observed")
        self.assertFalse(report["candidate_release"][CANDIDATE]["release_eligible"])

    def test_invalid_update_and_hash_are_counted(self):
        proposal = json.loads((self.record / "proposal.json").read_text())
        proposal["updates"] = [{"path": "forbidden", "value": True}]
        proposal["state_hash"] = "sha256:wrong"
        write(self.record / "proposal.json", proposal)
        report = self.run_eval()
        self.assertEqual(report["decision_metrics"]["invalid_applicable_patch_count"], 1)
        self.assertFalse(report["decision_replay"][0]["updates_match"])
        self.assertFalse(report["decision_replay"][0]["hash_consistent"])

    def test_incomplete_record_fails_closed(self):
        (self.record / "response.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete decision record"):
            self.run_eval()

    def test_cross_split_input_fails_closed(self):
        sessions = []
        for split in ("calibration", "test"):
            run_id = "run_" + split
            artifact = self.runs / run_id / "artifacts"
            artifact.mkdir(parents=True)
            (self.runs / run_id / "config.yaml").write_text(
                "global_metrics:\n  adaptive_weights: false\n", encoding="utf-8")
            write(artifact / "run_provenance.json", {
                "input_manifest": {"sha256": "same-frames", "entry_count": 1,
                                   "entries": [{"sha256": "same-frame"}]},
                "build": {"build_id": "build"}, "execution_scope": "same"})
            write(artifact / "forward_drizzle.json", {
                "selected_candidate": "drizzle_raw", "validation": {"drizzle_raw": {}}})
            sessions.append({"session_id": split, "split": split, "candidate_id": CANDIDATE,
                             "arms": {"current_config": run_id},
                             "arm_assertions": {"current_config": {"global_metrics.adaptive_weights": False}}})
        write(self.registry, {"schema_version": "pi.jev-evaluation-registry.v1",
                              "study_kind": "retrospective_exploratory", "sessions": sessions})
        with self.assertRaisesRegex(ValueError, "cross calibration/test"):
            self.run_eval()

    def test_wrong_arm_config_fails_closed(self):
        run_id = "run_wrong_arm"
        run_root = self.runs / run_id
        run_root.mkdir(parents=True)
        (run_root / "config.yaml").write_text(
            "global_metrics:\n  adaptive_weights: false\n", encoding="utf-8")
        write(self.registry, {"schema_version": "pi.jev-evaluation-registry.v1",
            "study_kind": "retrospective_exploratory", "sessions": [{
                "session_id": "s", "split": "test", "candidate_id": CANDIDATE,
                "arms": {"candidate_applied": run_id},
                "arm_assertions": {"candidate_applied": {"global_metrics.adaptive_weights": True}}
            }]})
        with self.assertRaisesRegex(ValueError, "config arm assertion failed"):
            self.run_eval()

    def test_memory_inventory_does_not_turn_outcomes_into_labels(self):
        path = self.root / "memory.jsonl"
        path.write_text('{"status":"approved","outcome":{"quality":"great"}}\n',
                        encoding="utf-8")
        inventory = memory_inventory(path)
        self.assertEqual(inventory["status_counts"], {"approved": 1})
        self.assertNotIn("quality", inventory)

    def test_release_policy_rejects_missing_or_unsafe_limit(self):
        policy = json.loads(POLICY_SOURCE.read_text(encoding="utf-8"))
        policy["candidates"][CANDIDATE]["minimum_quality_improvement"] = None
        with self.assertRaisesRegex(ValueError, "non-finite or non-numeric"):
            validate_release_policy(policy)
        policy["candidates"][CANDIDATE]["minimum_quality_improvement"] = 0.05
        policy["candidates"][CANDIDATE]["maximum_invalid_patch_rate"] = 0.01
        with self.assertRaisesRegex(ValueError, "invalid patches cannot be allowed"):
            validate_release_policy(policy)

    def test_test_session_must_follow_policy_freeze(self):
        frozen = "2026-09-25T11:22:54Z"
        self.assertFalse(enrolled_after_freeze({}, frozen))
        self.assertFalse(enrolled_after_freeze({"enrolled_at_utc": frozen}, frozen))
        self.assertTrue(enrolled_after_freeze({"enrolled_at_utc": "2026-09-26T00:00:00Z"}, frozen))


if __name__ == "__main__":
    unittest.main()
