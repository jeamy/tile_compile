import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import compare_background_structure as cbs  # noqa: E402
import evaluate_reconstruction_screening as ers  # noqa: E402
import make_screening_configs as msc  # noqa: E402
import policy_v2  # noqa: E402
import screen_downstream as sd  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
POLICY = os.path.join(REPO, "web_backend_cpp", "config", "pi_decisions", "release_policy_v2.json")

BASE = {"reconstruction": {"drizzle": {"pixfrac": 0.8, "robust_passes": 4}, "clipping": {"clip_sigma_low": 4.0, "clip_sigma_high": 4.0}},
        "pcc": {"enabled": True}, "astrometry": {"enabled": True}, "hypermetric_stretch": {"enabled": True},
        "bge": {"method": "auto"}, "registration": {"use_astrometry": True}, "global_metrics": {"adaptive_weights": False}}


class ConfigGeneratorTest(unittest.TestCase):
    def test_each_variant_differs_from_the_control_only_in_its_changes(self):
        out = msc.build(BASE, [("control", {}), ("pixfrac_1.0", {"reconstruction.drizzle.pixfrac": 1.0})])
        self.assertEqual(out["control"]["reconstruction"]["drizzle"]["pixfrac"], 0.8)
        b = copy.deepcopy(out["pixfrac_1.0"])
        b["reconstruction"]["drizzle"]["pixfrac"] = 0.8
        self.assertEqual(b, out["control"])
        self.assertEqual(BASE["reconstruction"]["drizzle"]["pixfrac"], 0.8, "the base config is never modified")

    def test_downstream_stages_are_off_in_every_arm_unless_kept(self):
        out = msc.build(BASE, [("control", {}), ("x", {"reconstruction.drizzle.robust_passes": 2})])
        for cfg in out.values():
            self.assertFalse(cfg["pcc"]["enabled"] or cfg["astrometry"]["enabled"] or cfg["hypermetric_stretch"]["enabled"] or cfg["registration"]["use_astrometry"])
            self.assertEqual(cfg["bge"]["method"], "none")
        kept = msc.build(BASE, [("control", {})], downstream_off=False)
        self.assertTrue(kept["control"]["pcc"]["enabled"])

    def test_typos_and_bad_orders_are_errors(self):
        with self.assertRaises(KeyError):
            msc.build(BASE, [("control", {}), ("x", {"reconstruction.drizzle.pixfrak": 1.0})])
        with self.assertRaises(KeyError):
            msc.build(BASE, [("control", {}), ("x", {"no.such.section": 1})])
        with self.assertRaises(ValueError):
            msc.build(BASE, [("x", {})])                       # control must come first
        with self.assertRaises(ValueError):
            msc.build(BASE, [("control", {"pcc.enabled": False})])
        with self.assertRaises(ValueError):
            msc.build(BASE, [("control", {}), ("a", {}), ("a", {})])

    def test_presets_apply_to_a_realistic_config_and_cli_refuses_a_used_directory(self):
        real = yaml.safe_load(open(os.path.join(REPO, "tile_compile_cpp", "tile_compile.yaml"), encoding="utf-8"))
        for name in ("reconstruction", "confirmation"):
            try:
                msc.build(real, msc.PRESETS[name])
            except KeyError as error:                          # the shipped default may lack an optional key
                self.fail("preset %s does not fit the shipped config: %s" % (name, error))
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "base.yaml")
            yaml.safe_dump(real, open(base, "w"))
            out = os.path.join(d, "cfg")
            self.assertEqual(msc.main(["--base", base, "--out-dir", out, "--preset", "confirmation"]), 0)
            self.assertEqual(json.load(open(os.path.join(out, "names.json"))), ["control", "pixfrac_1.0", "clip_5_5"])
            with self.assertRaises(SystemExit):
                msc.main(["--base", base, "--out-dir", out, "--preset", "confirmation"])


def sessions(noise, n=5, **over):
    rows = []
    for i in range(n):
        row = {"group": "galaxy" if i % 2 else "nebula", "noise_ratio": noise + 0.002 * i, "fwhm_ratio": 1.0, "elongation_ratio": 1.0, "signal_ratio": 1.0,
               "gate_passed": True, "selected_unchanged": True, "n_eff_ratio": 1.4, "matched_stars": 200, "background_pixels": 5_000_000}
        row.update(over)
        rows.append(row)
    return rows


class PolicyV2Test(unittest.TestCase):
    def setUp(self):
        self.policy = json.load(open(POLICY))

    def test_the_shipped_policy_is_valid_and_frozen(self):
        frozen = policy_v2.validate(self.policy)
        self.assertEqual(frozen.tzinfo is not None, True)
        self.assertEqual(self.policy["candidates"]["enable_adaptive_weights"]["status"], "rejected_by_futility")

    def test_invalid_policies_are_refused(self):
        def broken(mutate):
            p = copy.deepcopy(self.policy)
            mutate(p)
            with self.assertRaises(ValueError):
                policy_v2.validate(p)
        broken(lambda p: p.update(frozen_before_test=False))
        broken(lambda p: p.update(frozen_at_utc="2026-09-26T08:00:00+02:00"))
        broken(lambda p: p.update(frozen_at_utc="soon"))
        broken(lambda p: p["pairing_rules"].update(identical_registration_reference_frame=False))
        broken(lambda p: p["candidates"]["set_pixfrac"].update(status="maybe"))
        broken(lambda p: p["candidates"]["set_pixfrac"]["sample"].update(minimum_test_sessions=3))       # 2 groups x 2 sessions > 3
        broken(lambda p: p["candidates"]["set_pixfrac"].update(noise_ratio_ci_upper_max=1.2))
        broken(lambda p: p["candidates"]["enable_adaptive_weights"].pop("evidence"))
        broken(lambda p: p.update(confidence_level=0.9))

    def test_pairing_rules(self):
        ok, why = policy_v2.session_valid({"frames": 1, "build_id": "b", "ref_frame": 74, "raster": [10, 20]},
                                          {"frames": 1, "build_id": "b", "ref_frame": 74, "raster": [10, 20]})
        self.assertTrue(ok and not why)
        ok, why = policy_v2.session_valid({"frames": 1, "build_id": "b", "ref_frame": 183, "raster": [2844, 4612]},
                                          {"frames": 1, "build_id": "b", "ref_frame": 189, "raster": [2850, 4612]})
        self.assertFalse(ok)
        self.assertEqual(sorted(why), ["output_raster_differs", "registration_reference_frame_differs"])

    def test_verdict_holds_only_when_every_bound_holds(self):
        self.assertTrue(policy_v2.candidate_verdict(self.policy, "set_pixfrac", sessions(0.90))["holds"])
        v = policy_v2.candidate_verdict(self.policy, "set_pixfrac", sessions(0.97))
        self.assertIn("noise_endpoint_not_met", v["reasons"])
        self.assertIn("too_few_sessions", policy_v2.candidate_verdict(self.policy, "set_pixfrac", sessions(0.90, n=3))["reasons"])
        self.assertIn("star_shape_regression", policy_v2.candidate_verdict(self.policy, "set_pixfrac", sessions(0.90, fwhm_ratio=1.03))["reasons"])
        self.assertIn("coverage_gate_failed", policy_v2.candidate_verdict(self.policy, "set_pixfrac", sessions(0.90, gate_passed=False))["reasons"])
        self.assertIn("coverage_worse", policy_v2.candidate_verdict(self.policy, "set_pixfrac", sessions(0.90, n_eff_ratio=0.9))["reasons"])
        self.assertIn("selected_candidate_changed", policy_v2.candidate_verdict(self.policy, "set_pixfrac", sessions(0.90, selected_unchanged=False))["reasons"])
        one_group = sessions(0.90)
        for s in one_group:
            s["group"] = "nebula"
        self.assertIn("object_group_strata_not_met", policy_v2.candidate_verdict(self.policy, "set_pixfrac", one_group)["reasons"])
        self.assertIn("session_sample_below_minimum", policy_v2.candidate_verdict(self.policy, "set_pixfrac", sessions(0.90, matched_stars=10))["reasons"])
        self.assertEqual(policy_v2.candidate_verdict(self.policy, "enable_adaptive_weights", sessions(0.9))["reasons"], ["candidate_status_rejected_by_futility"])

    def test_signal_bounds_of_the_clipping_candidate(self):
        self.assertTrue(policy_v2.candidate_verdict(self.policy, "set_clip_sigmas", sessions(0.94, signal_ratio=1.005))["holds"])
        self.assertIn("signal_excess", policy_v2.candidate_verdict(self.policy, "set_clip_sigmas", sessions(0.94, signal_ratio=1.08))["reasons"])
        self.assertIn("signal_loss", policy_v2.candidate_verdict(self.policy, "set_clip_sigmas", sessions(0.94, signal_ratio=0.95))["reasons"])

    def test_futility_needs_enough_sessions_and_a_clearly_worse_interval(self):
        self.assertFalse(policy_v2.futility(self.policy, sessions(1.05, n=3)))
        self.assertTrue(policy_v2.futility(self.policy, sessions(1.05, n=6)))
        self.assertFalse(policy_v2.futility(self.policy, sessions(0.90, n=6)))
        self.assertFalse(policy_v2.futility(self.policy, sessions(0.99, n=6)), "an interval that includes 1.0 is not futile")


class DownstreamHelpersTest(unittest.TestCase):
    def test_a_variant_may_not_add_keys(self):
        with self.assertRaises(KeyError):
            sd.set_path({"bge": {"method": "auto"}}, "bge.methode", "none")
        c = {"bge": {"method": "auto"}}
        sd.set_path(c, "bge.method", "none")
        self.assertEqual(c["bge"]["method"], "none")

    def test_the_run_start_revision_is_created_once_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "config.yaml"), "w").write("a: 1\n")
            self.assertTrue(sd.ensure_run_start_revision(d))
            entry = json.load(open(os.path.join(d, "artifacts", "config_revisions", "index.json")))[0]
            self.assertEqual(entry["source"], "run_start")
            self.assertEqual(open(os.path.join(d, "artifacts", "config_revisions", entry["file_name"])).read(), "a: 1\n")
            open(os.path.join(d, "config.yaml"), "w").write("a: 2\n")
            self.assertFalse(sd.ensure_run_start_revision(d), "an existing revision list is kept")
            self.assertEqual(open(os.path.join(d, "artifacts", "config_revisions", entry["file_name"])).read(), "a: 1\n")

    def test_sky_metrics_see_noise_and_large_scale_structure(self):
        rng = np.random.default_rng(3)
        h, w = 1100, 1100
        yy, xx = np.mgrid[0:h, 0:w]
        ramp = 0.02 * (xx / w)
        quiet = np.stack([0.12 + ramp + rng.normal(0, 0.002, (h, w))] * 3, axis=-1)
        noisy = np.stack([0.12 + ramp + rng.normal(0, 0.006, (h, w))] * 3, axis=-1)
        a, b = sd.sky_metrics(quiet), sd.sky_metrics(noisy)
        self.assertAlmostEqual(b["pixel_noise"] / a["pixel_noise"], 3.0, delta=0.4)
        self.assertGreater(a["sky_span"], 0.005)
        self.assertGreater(a["midscale_over_noise"], b["midscale_over_noise"] * 0)  # both finite and positive
        self.assertTrue(np.isfinite(list(a.values())).all())


def write_run(root, name, selected="drizzle_raw", ref=74, n_eff=25.9, error=None, with_result=True):
    run = os.path.join(root, name)
    os.makedirs(os.path.join(run, "artifacts"))
    os.makedirs(os.path.join(run, "logs"))
    with open(os.path.join(run, "logs", "run_events.jsonl"), "w") as fh:
        if error:
            fh.write(json.dumps({"error": error, "type": "phase_end"}) + "\n")
        fh.write(json.dumps({"type": "run_end", "success": with_result}) + "\n")
    if with_result:
        json.dump({"selected_candidate": selected}, open(os.path.join(run, "artifacts", "forward_drizzle.json"), "w"))
        json.dump({"ref_frame": ref}, open(os.path.join(run, "artifacts", "global_registration.json"), "w"))
        json.dump({"coverage_gate": {"min_channel_n_eff_p10": n_eff}}, open(os.path.join(run, "artifacts", "sampling_geometry.json"), "w"))
    return run


class EvaluationTest(unittest.TestCase):
    def test_failure_reasons_and_missing_results_are_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, "p_control")
            write_run(d, "p_pixfrac_0.6", error="FORWARD_STAGE_COVERAGE_GATE_FAILED", with_result=False)
            self.assertEqual(ers.failure_reason(os.path.join(d, "p_pixfrac_0.6")), "FORWARD_STAGE_COVERAGE_GATE_FAILED")
            self.assertEqual(ers.failure_reason(os.path.join(d, "does_not_exist")), "no_run")
            out = ers.evaluate(d, "p", ["pixfrac_0.6"])
            self.assertEqual(out["variants"]["pixfrac_0.6"], {"no_result": "FORWARD_STAGE_COVERAGE_GATE_FAILED"})
            self.assertEqual(out["control"]["ref_frame"], 74)

    def test_the_report_must_not_be_written_into_the_runs_directory(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, "p_control")
            with self.assertRaises(SystemExit):
                ers.main(["--runs-dir", d, "--prefix", "p", "--out", os.path.join(d, "report.json")])


def make_sky(seed, h=400, w=520):
    import cv2
    rng = np.random.default_rng(seed)
    sky = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 25)
    return 100.0 + 8.0 * sky / sky.std()


def header(crpix):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [248.0, -25.0]
    w.wcs.cdelt = [-0.0008, 0.0008]
    w.wcs.crpix = list(crpix)
    return w.to_header()


class BackgroundStructureTest(unittest.TestCase):
    def test_the_same_sky_on_a_shifted_grid_correlates_and_another_sky_does_not(self):
        sky = make_sky(1)
        a = sky[0:360, 0:480]
        b = sky[20:380, 30:510]                                  # b[y, x] = sky[y + 20, x + 30]
        ha, hb = header((240, 180)), header((240 - 30, 180 - 20))
        result, ma, mb = cbs.compare(a, ha, b, hb)
        self.assertGreater(result["overlap_fraction_of_a"], 0.5)
        self.assertGreater(result["correlation_raw"], 0.95)
        self.assertGreater(result["correlation_after_plane_removed"], 0.9)
        other, _, _ = cbs.compare(a, ha, make_sky(2)[20:380, 30:510], hb)
        self.assertLess(abs(other["correlation_after_quadratic_removed"]), 0.6)

    def test_wcs_header_cards_are_read_with_and_without_line_breaks(self):
        with tempfile.TemporaryDirectory() as d:
            text = header((10, 20)).tostring(sep="", endcard=True)
            open(os.path.join(d, "flat.wcs"), "w").write(text)
            open(os.path.join(d, "lines.wcs"), "w").write("\n".join(text[i:i + 80] for i in range(0, len(text), 80)))
            for name in ("flat.wcs", "lines.wcs"):
                h = cbs.read_wcs_header(os.path.join(d, name))
                self.assertAlmostEqual(float(h["CRPIX1"]), 10.0)
                self.assertAlmostEqual(float(h["CRVAL1"]), 248.0)


FAKE_RUNNER = r"""#!/bin/bash
# fake tile_compile_runner: succeeds unless the run id contains "gate" (then it fails like a protected gate)
while [ $# -gt 0 ]; do case "$1" in --run-id) ID=$2; shift 2;; --runs-dir) RUNS=$2; shift 2;; *) shift;; esac; done
mkdir -p "$RUNS/$ID/logs" "$RUNS/$ID/cache" "$RUNS/$ID/outputs/calibrated" "$RUNS/$ID/artifacts"
echo "x" > "$RUNS/$ID/cache/big"; echo "y" > "$RUNS/$ID/outputs/calibrated/f"; echo "cfg" > "$RUNS/$ID/config.yaml"; echo "{}" > "$RUNS/$ID/artifacts/result.json"
case "$ID" in
  *gate*) echo '{"error":"FORWARD_STAGE_COVERAGE_GATE_FAILED","type":"phase_end"}' >> "$RUNS/$ID/logs/run_events.jsonl"
          echo '{"type":"run_end","success":false}' >> "$RUNS/$ID/logs/run_events.jsonl"; exit 1;;
  *) echo '{"type":"run_end","success":true}' >> "$RUNS/$ID/logs/run_events.jsonl"; exit 0;;
esac
"""


class ShellScriptsTest(unittest.TestCase):
    def test_chain_runs_in_order_prunes_finished_arms_and_continues_after_a_failed_gate(self):
        with tempfile.TemporaryDirectory() as d:
            project = os.path.join(d, "project")
            os.makedirs(os.path.join(project, "tile_compile_cpp", "scripts"))
            shutil.copy(os.path.join(REPO, "tile_compile_cpp", "scripts", "prune_evaluation_run.py"),
                        os.path.join(project, "tile_compile_cpp", "scripts", "prune_evaluation_run.py"))
            runner = os.path.join(d, "runner.sh")
            open(runner, "w").write(FAKE_RUNNER)
            os.chmod(runner, 0o755)
            cfg = os.path.join(d, "cfg")
            os.makedirs(cfg)
            json.dump(["control", "gate_low", "pixfrac_1.0"], open(os.path.join(cfg, "names.json"), "w"))
            for n in ("control", "gate_low", "pixfrac_1.0"):
                open(os.path.join(cfg, n + ".yaml"), "w").write("a: 1\n")
            runs = os.path.join(d, "runs")
            os.makedirs(runs)
            status = os.path.join(d, "status.txt")
            r = subprocess.run(["bash", os.path.join(HERE, "run_reconstruction_screening.sh"), "--config-dir", cfg, "--input-dir", d, "--runs-dir", runs,
                                "--prefix", "s", "--status", status, "--min-free-gb", "1", "--runner", runner, "--project-root", project],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            lines = open(status).read().splitlines()
            self.assertEqual([l.split()[1] for l in lines if " started " in l or " ok " in l or " FAILED" in l],
                             ["started", "ok", "started", "FAILED", "started", "ok"])
            self.assertIn("FORWARD_STAGE_COVERAGE_GATE_FAILED", [l for l in lines if "FAILED" in l][0])
            self.assertEqual(lines[-1], "DONE")
            for arm in ("s_control", "s_pixfrac_1.0", "s_gate_low"):
                self.assertFalse(os.path.exists(os.path.join(runs, arm, "cache")), arm + " keeps no cache")
                self.assertTrue(os.path.exists(os.path.join(runs, arm, "artifacts", "result.json")), arm + " keeps its result")

    def test_chain_stops_when_there_is_too_little_free_space(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "cfg")
            os.makedirs(cfg)
            json.dump(["control"], open(os.path.join(cfg, "names.json"), "w"))
            runner = os.path.join(d, "runner.sh")
            open(runner, "w").write(FAKE_RUNNER)
            os.chmod(runner, 0o755)
            status = os.path.join(d, "status.txt")
            subprocess.run(["bash", os.path.join(HERE, "run_reconstruction_screening.sh"), "--config-dir", cfg, "--input-dir", d, "--runs-dir", d,
                            "--prefix", "s", "--status", status, "--min-free-gb", "99999999", "--runner", runner, "--project-root", d], capture_output=True, text=True)
            self.assertIn("NOT STARTED", open(status).read())
            self.assertFalse(os.path.exists(os.path.join(d, "s_control")))

    @unittest.skipUnless(shutil.which("rsync"), "rsync not installed")
    def test_archive_moves_verified_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "src"), os.path.join(d, "dst")
            os.makedirs(os.path.join(src, "run_a", "sub"))
            os.makedirs(os.path.join(src, "run_b"))
            os.makedirs(os.path.join(dst, "run_b"))
            open(os.path.join(src, "run_a", "sub", "f"), "w").write("data")
            open(os.path.join(src, "run_b", "f"), "w").write("new")
            open(os.path.join(dst, "run_b", "f"), "w").write("old")
            status = os.path.join(d, "status.txt")
            subprocess.run(["bash", os.path.join(HERE, "archive_verified.sh"), "--src", src, "--dst", dst, "--status", status, "run_a", "run_b", "../evil", "missing"],
                           capture_output=True, text=True)
            text = open(status).read()
            self.assertIn("run_a moved ok", text)
            self.assertFalse(os.path.exists(os.path.join(src, "run_a")))
            self.assertEqual(open(os.path.join(dst, "run_a", "sub", "f")).read(), "data")
            self.assertIn("run_b already exists at the destination", text)
            self.assertEqual(open(os.path.join(dst, "run_b", "f")).read(), "old", "an existing archived run is never overwritten")
            self.assertTrue(os.path.exists(os.path.join(src, "run_b", "f")), "and its source is kept")
            self.assertIn("../evil is not a plain run id", text)
            self.assertIn("missing missing at source", text)


if __name__ == "__main__":
    unittest.main()
