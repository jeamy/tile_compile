import json
import os
import tempfile
import unittest

import prune_evaluation_run as prune


def make_run(root, ended=True, success=True):
    for d in ("logs", "artifacts", "outputs/calibrated", "cache/normalized_frames"):
        os.makedirs(os.path.join(root, d))
    open(os.path.join(root, "config.yaml"), "w").write("x: 1\n")
    open(os.path.join(root, "artifacts", "forward_drizzle.json"), "w").write("{}")
    open(os.path.join(root, "outputs", "result.fit"), "w").write("keep")
    open(os.path.join(root, "outputs", "calibrated", "cal_0.fit"), "wb").write(b"x" * 100)
    open(os.path.join(root, "cache", "normalized_frames", "f"), "wb").write(b"y" * 50)
    with open(os.path.join(root, "logs", "run_events.jsonl"), "w") as fh:
        fh.write(json.dumps({"type": "run_start"}) + "\n")
        if ended:
            fh.write(json.dumps({"type": "run_end", "success": success}) + "\n")


class PruneTest(unittest.TestCase):
    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            make_run(d)
            self.assertEqual(prune.main([d]), 0)
            self.assertTrue(os.path.isdir(os.path.join(d, "cache")))
            self.assertTrue(os.path.isdir(os.path.join(d, "outputs", "calibrated")))

    def test_apply_removes_only_intermediates(self):
        with tempfile.TemporaryDirectory() as d:
            make_run(d)
            self.assertEqual(prune.main([d, "--apply"]), 0)
            self.assertFalse(os.path.exists(os.path.join(d, "cache")))
            self.assertFalse(os.path.exists(os.path.join(d, "outputs", "calibrated")))
            for kept in ("config.yaml", "logs/run_events.jsonl", "artifacts/forward_drizzle.json", "outputs/result.fit"):
                self.assertTrue(os.path.exists(os.path.join(d, kept)), kept)
            self.assertEqual(prune.main([d, "--apply"]), 0)  # idempotent

    def test_refuses_unfinished_or_failed_run(self):
        for ended, success in ((False, False), (True, False)):
            with tempfile.TemporaryDirectory() as d:
                make_run(d, ended=ended, success=success)
                self.assertEqual(prune.main([d, "--apply"]), 2)
                self.assertTrue(os.path.isdir(os.path.join(d, "cache")), "cache of a run that can still be resumed stays")

    def test_refuses_foreign_directory(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "cache"))
            self.assertEqual(prune.main([d, "--apply"]), 2)
            self.assertTrue(os.path.isdir(os.path.join(d, "cache")))

    def test_refuses_symlinked_target(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as elsewhere:
            make_run(d)
            import shutil
            shutil.rmtree(os.path.join(d, "cache"))
            os.symlink(elsewhere, os.path.join(d, "cache"))
            open(os.path.join(elsewhere, "precious"), "w").write("z")
            self.assertEqual(prune.main([d, "--apply"]), 2)
            self.assertTrue(os.path.exists(os.path.join(elsewhere, "precious")))


if __name__ == "__main__":
    unittest.main()
