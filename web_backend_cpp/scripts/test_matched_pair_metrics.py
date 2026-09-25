import json
import os
import tempfile
import unittest

import numpy as np

import matched_pair_metrics as mp

H = W = 500


def make_image(sigma, noise, seed=1, extra=(), noise_seed=None, amp=1.0):
    """Gaussian stars on a flat background; the same star positions for every sigma/noise variant."""
    rng = np.random.default_rng(seed)
    ys = rng.integers(40, H - 40, 90)
    xs = rng.integers(40, W - 40, 90)
    # keep them apart so they are isolated
    keep, pts = [], []
    for y, x in zip(ys, xs):
        if all((y - py) ** 2 + (x - px) ** 2 > 40 ** 2 for py, px in pts):
            pts.append((y, x))
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 100.0)
    amps = np.random.default_rng(seed + 50).uniform(80.0, 400.0, len(pts))   # a real star field is not uniform in brightness
    for (y, x), a in zip(pts, amps):
        img += amp * a * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
    for y, x in extra:
        img += amp * 300.0 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
    n = np.random.default_rng(noise_seed if noise_seed is not None else seed + 100).normal(0, noise, (H, W))
    return img + n, pts


class MatchedMetricsTest(unittest.TestCase):
    def test_identical_images_give_unit_ratios(self):
        a, _ = make_image(1.8, 1.0)
        r = mp.compare_images(a, a.copy())
        self.assertGreaterEqual(r["matched_stars"], 8)
        for key in ("fwhm_ratio_candidate_over_control", "elongation_ratio_candidate_over_control", "signal_ratio_candidate_over_control"):
            self.assertAlmostEqual(r[key]["median"], 1.0, places=9)
        self.assertAlmostEqual(r["noise"]["ratio_candidate_over_control"], 1.0, places=9)
        self.assertEqual(r["coverage"]["coverage_loss"], 0.0)

    def test_sharper_candidate_is_detected_per_star(self):
        a, _ = make_image(2.0, 1.0)
        b, _ = make_image(1.8, 1.0)  # same stars, 10 % narrower
        r = mp.compare_images(a, b)
        med = r["fwhm_ratio_candidate_over_control"]["median"]
        self.assertAlmostEqual(med, 0.90, delta=0.03)
        lo, hi = r["fwhm_ratio_candidate_over_control"]["ci95_star_bootstrap"]
        self.assertTrue(lo <= med <= hi and hi < 0.95)

    def test_noise_is_measured_on_background_pixels_only(self):
        a, _ = make_image(1.8, 1.0)
        b, _ = make_image(1.8, 1.5, noise_seed=7)
        r = mp.compare_images(a, b)
        self.assertAlmostEqual(r["noise"]["ratio_candidate_over_control"], 1.5, delta=0.08)
        self.assertGreater(r["noise"]["valid_background_pixels"], 150000)

    def test_coverage_loss_and_common_area(self):
        a, _ = make_image(1.8, 1.0)
        b = a.copy()
        b[:, : W // 10] = 0.0           # the candidate lost the left 10 % of the canvas
        r = mp.compare_images(a, b)
        self.assertAlmostEqual(r["coverage"]["coverage_loss"], 0.10, delta=0.002)
        self.assertAlmostEqual(r["coverage"]["intersection_over_control"], 0.90, delta=0.002)
        # stars inside the hole can not be measured in both images
        self.assertLess(r["matched_stars"], r["detected_isolated_stars"] + 1)

    def test_signal_loss_is_measured_at_the_same_apertures(self):
        a, _ = make_image(1.8, 1.0)
        b, _ = make_image(1.8, 1.0, amp=0.99)   # 1 % less star signal
        r = mp.compare_images(a, b)
        self.assertAlmostEqual(r["signal_ratio_candidate_over_control"]["median"], 0.99, delta=0.005)

    def test_saturated_crowded_and_border_stars_are_excluded(self):
        a, pts = make_image(1.8, 1.0)
        base = mp.compare_images(a, a.copy())["detected_isolated_stars"]
        # a saturated star, a close pair and a border star are added to both images
        extra = [(250, 250), (420, 100), (420, 108), (10, 250)]
        b, _ = make_image(1.8, 1.0, extra=extra)
        b[248:253, 248:253] = 1e5                     # saturated core
        stars, peaks, _ = mp.detect_stars(b, np.ones_like(b, bool))
        positions = {(y, x) for y, x in stars}
        self.assertFalse(any(abs(y - 250) < 3 and abs(x - 250) < 3 for y, x in positions), "saturated star excluded")
        self.assertFalse(any(abs(y - 420) < 3 and 95 < x < 115 for y, x in positions), "crowded pair excluded")
        self.assertFalse(any(y < mp.BORDER for y, x in positions), "border star excluded")
        self.assertGreaterEqual(len(stars), base - 2)

    def test_shape_mismatch_and_empty_overlap_raise(self):
        a, _ = make_image(1.8, 1.0)
        with self.assertRaises(ValueError):
            mp.compare_images(a, a[:-1])
        with self.assertRaises(ValueError):
            mp.compare_images(a, np.zeros_like(a))

    def test_result_is_deterministic(self):
        a, _ = make_image(2.0, 1.0)
        b, _ = make_image(1.9, 1.0)
        self.assertEqual(json.dumps(mp.compare_images(a, b), sort_keys=True), json.dumps(mp.compare_images(a, b), sort_keys=True))

    def test_policy_flags_report_sample_size_only(self):
        a, _ = make_image(1.8, 1.0)
        r = mp.compare_images(a, a.copy())
        policy = {"candidates": {"enable_adaptive_weights": {"minimum_matched_stars_per_session": 5,
                                                              "minimum_valid_background_pixels_per_session": 100000,
                                                              "minimum_unsaturated_signal_apertures_per_session": 5,
                                                              "minimum_coverage": 0.95}}}
        flags = mp.policy_flags(r, policy)
        self.assertTrue(all(flags.values()), flags)
        policy["candidates"]["enable_adaptive_weights"]["minimum_matched_stars_per_session"] = 10_000
        self.assertFalse(mp.policy_flags(r, policy)["matched_stars_ok"])

    def test_run_selection_and_output_location(self):
        with tempfile.TemporaryDirectory() as d:
            for name, sel in (("ctl", "drizzle_raw"), ("cand", "drizzle_uniform")):
                os.makedirs(os.path.join(d, name, "artifacts"))
                os.makedirs(os.path.join(d, name, "outputs"))
                json.dump({"selected_candidate": sel}, open(os.path.join(d, name, "artifacts", "forward_drizzle.json"), "w"))
            with self.assertRaises(ValueError):
                mp.load_pair(os.path.join(d, "ctl"), os.path.join(d, "cand"))   # different selected outputs
            with self.assertRaises(SystemExit):
                mp.main(["--control", os.path.join(d, "ctl"), "--candidate", os.path.join(d, "cand"), "--out", os.path.join(d, "ctl", "x.json")])


if __name__ == "__main__":
    unittest.main()
