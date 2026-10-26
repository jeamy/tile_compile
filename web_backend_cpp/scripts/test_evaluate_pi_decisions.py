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


if __name__ == "__main__":
    unittest.main()
