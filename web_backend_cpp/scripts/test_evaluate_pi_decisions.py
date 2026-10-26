import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_pi_decisions as evaluator  # noqa: E402


class ProposalRunLinkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.decisions = self.root / "decisions"
        self.runs = self.root / "runs"
        self.proposal_id = "proposal_001"
        self.run_id = "run_001"
        self.proposal_dir = self.decisions / self.proposal_id
        self.run_dir = self.runs / self.run_id
        self.proposal_dir.mkdir(parents=True)
        (self.run_dir / "artifacts").mkdir(parents=True)
        (self.run_dir / "logs").mkdir()

        config = {"reconstruction": {"drizzle": {"pixfrac": 1.0}}}
        config_path = self.run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        config_hash = "sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest()
        self.proposal = {
            "status": "applied_to_draft",
            "saved_revision_ids": ["revision_001"],
            "updates": [{"path": "reconstruction.drizzle.pixfrac", "value": 1.0}],
        }
        self.outcome = {
            "schema_version": "pi.decision-outcome.v1",
            "runs": [{"run_id": self.run_id, "config_revision_id": "run_revision_001",
                      "saved_config_revision_id": "revision_001",
                      "attribution": "paths_present"}],
        }
        (self.run_dir / "artifacts" / "pi_run_provenance.json").write_text(json.dumps({
            "jev_proposal_id": self.proposal_id,
            "jev_proposal_ids": [self.proposal_id],
            "config_revision_id": "run_revision_001",
            "prior_active_config_revision_id": "revision_001",
            "config_sha256": config_hash,
        }), encoding="utf-8")
        (self.run_dir / "artifacts" / "run_provenance.json").write_text(json.dumps({
            "input_manifest": {"sha256": "sha256:input"},
        }), encoding="utf-8")
        (self.run_dir / "logs" / "run_events.jsonl").write_text(
            json.dumps({"type": "run_end", "success": True}) + "\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_verifies_proposal_revision_config_hash_and_successful_run(self):
        links, status = evaluator.decision_run_links(
            self.proposal_id, self.proposal, self.outcome, self.runs)
        self.assertEqual(status, "verified")
        self.assertEqual(links[0]["status"], "verified")
        self.assertTrue(all(links[0]["checks"].values()))
        self.assertEqual(links[0]["input_manifest_hash"], "sha256:input")

    def test_rejects_mismatched_proposal_and_unsuccessful_run(self):
        provenance_path = self.run_dir / "artifacts" / "pi_run_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["jev_proposal_id"] = "different_proposal"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        (self.run_dir / "logs" / "run_events.jsonl").write_text(
            json.dumps({"type": "run_end", "success": False}) + "\n", encoding="utf-8")

        links, status = evaluator.decision_run_links(
            self.proposal_id, self.proposal, self.outcome, self.runs)
        self.assertEqual(status, "invalid")
        self.assertEqual(links[0]["status"], "invalid")
        self.assertFalse(links[0]["checks"]["proposal_id_matches"])
        self.assertFalse(links[0]["checks"]["run_succeeded"])

    def test_replay_reads_live_sqlite_read_only_and_populates_run_config_hash(self):
        database = self.root / "pi_store_v2.sqlite"
        self.decisions.mkdir(parents=True, exist_ok=True)
        documents = {
            "status": {"state": "done"},
            "proposal": {**self.proposal, "candidate_id": "set_pixfrac", "state_hash": "state_1",
                         "policy_version": "policy_1", "validation": {
                             "config_ok": True, "evidence_ok": True, "policy_ok": True, "schema_ok": True}},
            "state": {"state_hash": "state_1"},
            "candidates": {"catalog_version": "catalog_1", "applicable": [{
                "candidate_id": "set_pixfrac", "updates": self.proposal["updates"]}]},
            "request": {"state_hash": "state_1"},
            "response": {"status": "ok", "state_hash": "state_1",
                         "selection": {"candidate_id": "set_pixfrac"}},
            "outcome": self.outcome,
        }
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE jev_documents (proposal_id TEXT, part TEXT, json TEXT)")
            connection.executemany("INSERT INTO jev_documents VALUES (?, ?, ?)", [
                (self.proposal_id, part, json.dumps(document)) for part, document in documents.items()
            ])
        rows = evaluator.replay(self.decisions, self.runs)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_link_status"], "verified")
        self.assertEqual(rows[0]["run_config_hash"], evaluator.digest(self.run_dir / "config.yaml"))
        self.assertEqual(rows[0]["user_acceptance"], "saved_config")
        self.assertIsNone(rows[0]["quality_label"])

    def test_partial_attribution_does_not_count_as_a_verified_run_link(self):
        self.outcome["runs"][0]["attribution"] = "paths_partial"
        links, status = evaluator.decision_run_links(
            self.proposal_id, self.proposal, self.outcome, self.runs)
        self.assertEqual(status, "invalid")
        self.assertFalse(links[0]["checks"]["outcome_attribution_has_all_paths"])

    def test_validates_both_v1_and_v2_release_policies(self):
        cfg_dir = Path(__file__).resolve().parent.parent / "config" / "pi_decisions"
        policy_v1 = evaluator.load(cfg_dir / "release_policy_v1.json")
        policy_v2 = evaluator.load(cfg_dir / "release_policy_v2.json")
        evaluator.validate_release_policy(policy_v1)
        evaluator.validate_release_policy(policy_v2)

    def test_evaluate_computes_four_baselines_and_v2_matched_star_policy(self):
        cfg_dir = Path(__file__).resolve().parent.parent / "config" / "pi_decisions"
        policy_v2 = evaluator.load(cfg_dir / "release_policy_v2.json")
        policy_path = self.root / "policy_v2.json"
        policy_path.write_text(json.dumps(policy_v2), encoding="utf-8")

        database = self.root / "pi_store_v2.sqlite"
        documents = {
            "status": {"state": "done"},
            "proposal": {**self.proposal, "candidate_id": "set_pixfrac", "state_hash": "state_1",
                         "policy_version": "policy_2", "validation": {
                             "config_ok": True, "evidence_ok": True, "policy_ok": True, "schema_ok": True}},
            "state": {"state_hash": "state_1"},
            "candidates": {"catalog_version": "catalog_1", "applicable": [
                {"candidate_id": "keep_current", "updates": []},
                {"candidate_id": "set_pixfrac", "updates": self.proposal["updates"]},
            ]},
            "request": {"state_hash": "state_1"},
            "response": {"status": "ok", "state_hash": "state_1",
                         "selection": {"candidate_id": "set_pixfrac"}},
            "outcome": self.outcome,
        }
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE jev_documents (proposal_id TEXT, part TEXT, json TEXT)")
            connection.executemany("INSERT INTO jev_documents VALUES (?, ?, ?)", [
                (self.proposal_id, part, json.dumps(document)) for part, document in documents.items()
            ])
            connection.execute("CREATE TABLE memory_records (memory_id TEXT, status TEXT)")
            connection.executemany("INSERT INTO memory_records VALUES (?, ?)", [
                ("mem_1", "confirmed"), ("mem_2", "pending_outcome"),
            ])

        sessions = []
        for idx, obj_cls in enumerate(["galaxy", "galaxy", "galaxy", "nebula", "nebula"]):
            ctrl_id = f"ctrl_{idx}"
            cand_id = f"cand_{idx}"
            for rid, pixfrac in ((ctrl_id, 0.8), (cand_id, 1.0)):
                rdir = self.runs / rid
                (rdir / "artifacts").mkdir(parents=True, exist_ok=True)
                (rdir / "config.yaml").write_text(
                    yaml.safe_dump({"reconstruction": {"drizzle": {"pixfrac": pixfrac}}}), encoding="utf-8")
                (rdir / "artifacts" / "run_provenance.json").write_text(json.dumps({
                    "input_manifest": {"sha256": f"sha256:in_{idx}", "entry_count": 1,
                                       "entries": [{"sha256": f"sha256:frame_{idx}"}]},
                    "build": {"build_id": "build_1"},
                    "execution_scope": "reconstruct_only",
                }), encoding="utf-8")
                (rdir / "artifacts" / "forward_drizzle.json").write_text(json.dumps({
                    "selected_candidate": "drizzle_raw",
                    "selection_reason": "gates_ok",
                    "pixels_supported": 200000,
                    "validation": {
                        "drizzle_raw": {"support_ok": True, "numerics_ok": True,
                                        "median_fwhm": 3.0, "elongation": 1.05, "background_rms": 0.01, "tail": 0.1},
                    },
                }), encoding="utf-8")
            sessions.append({
                "session_id": f"sess_{idx}",
                "object_class": obj_cls,
                "split": "test",
                "enrolled_at_utc": "2026-10-01T00:00:00Z",
                "candidate_id": "set_pixfrac",
                "arms": {"current_config": ctrl_id, "candidate_applied": cand_id},
                "arm_assertions": {
                    "current_config": {"reconstruction.drizzle.pixfrac": 0.8},
                    "candidate_applied": {"reconstruction.drizzle.pixfrac": 1.0},
                },
                "matched_metrics": {
                    "matched_stars": 80,
                    "signal_apertures": 80,
                    "fwhm_ratio_candidate_over_control": {"median": 0.995},
                    "elongation_ratio_candidate_over_control": {"median": 1.001},
                    "signal_ratio_candidate_over_control": {"median": 1.002},
                    "noise": {"ratio_candidate_over_control": 0.91, "valid_background_pixels": 180000},
                    "coverage": {"intersection_over_control": 0.999, "coverage_loss": 0.0},
                },
            })

        registry_path = self.root / "registry.json"
        registry_path.write_text(json.dumps({
            "schema_version": "pi.jev-evaluation-registry.v1",
            "study_kind": "prospective_blinded",
            "sessions": sessions,
        }), encoding="utf-8")

        report = evaluator.evaluate(self.decisions, registry_path, policy_path, self.runs)
        baselines = report["baselines"]
        self.assertEqual(baselines["current_config"]["status"], "observed")
        self.assertEqual(baselines["current_config"]["registered_control_sessions"], 5)
        self.assertEqual(baselines["rules_only"]["status"], "observed")
        self.assertEqual(baselines["rules_only"]["actionable_rule_candidate_count"], 1)
        self.assertEqual(baselines["existing_pi_advice"]["status"], "observed")
        self.assertEqual(baselines["existing_pi_advice"]["memory_record_count"], 2)
        self.assertEqual(baselines["rules_plus_jev"]["status"], "observed")
        self.assertEqual(baselines["rules_plus_jev"]["selected_change_count"], 1)
        self.assertEqual(baselines["knn_optional"]["status"], "disabled")

        rel = report["candidate_release"]
        self.assertFalse(rel["enable_adaptive_weights"]["release_eligible"])
        self.assertIn("rejected_by_futility", rel["enable_adaptive_weights"]["blocking_reasons"])
        self.assertFalse(rel["set_sensor_profile_dwarf_ii"]["release_eligible"])
        self.assertIn("candidate_blocked_in_policy", rel["set_sensor_profile_dwarf_ii"]["blocking_reasons"])
        self.assertTrue(rel["set_pixfrac"]["release_eligible"])
        self.assertEqual(rel["set_pixfrac"]["blocking_reasons"], [])
        self.assertFalse(rel["set_clip_sigmas"]["release_eligible"])
        self.assertIn("no_test_sessions", rel["set_clip_sigmas"]["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
