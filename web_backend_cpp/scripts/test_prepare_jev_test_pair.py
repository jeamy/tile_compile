import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


SCRIPT = Path(__file__).with_name("prepare_jev_test_pair.py")


class PreparePairTests(unittest.TestCase):
    def test_arms_differ_only_in_adaptive_weighting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.yaml"
            dark = root / "master.fit"
            out = root / "pair"
            base.write_text(yaml.safe_dump({
                "calibration": {"use_dark": True, "dark_master": "/old/master.fit"},
                "global_metrics": {"adaptive_weights": True, "other": 7},
                "other_section": {"keep": "same"},
            }), encoding="utf-8")
            dark.write_bytes(b"test-dark-master")
            subprocess.run([sys.executable, str(SCRIPT), "--base-config", str(base),
                            "--dark-master", str(dark), "--output-dir", str(out)],
                           check=True, capture_output=True, text=True)
            off = yaml.safe_load((out / "config_adaptive_off.yaml").read_text(encoding="utf-8"))
            on = yaml.safe_load((out / "config_adaptive_on.yaml").read_text(encoding="utf-8"))
            self.assertFalse(off["global_metrics"]["adaptive_weights"])
            self.assertTrue(on["global_metrics"]["adaptive_weights"])
            on["global_metrics"]["adaptive_weights"] = False
            self.assertEqual(on, off)
            self.assertEqual(off["calibration"]["dark_master"], str(dark.resolve()))
            manifest = json.loads((out / "pair_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["only_difference"], "global_metrics.adaptive_weights")
            self.assertEqual(set(manifest["arms"]), {"off", "on"})


if __name__ == "__main__":
    unittest.main()
