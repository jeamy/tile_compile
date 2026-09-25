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

    def _big(self):
        global H, W
        h0, w0 = H, W
        H, W = 620, 640
        try:
            img, _ = make_image(1.8, 1.0, seed=3)
        finally:
            H, W = h0, w0
        return img

    def test_shift_estimation_recovers_the_canvas_offset(self):
        big = self._big()
        control, candidate = big[0:500, 0:500], big[6:506, 3:503]     # candidate[y, x] = big[y + 6, x + 3]
        dy, dx, response = mp.estimate_shift(control, candidate)
        self.assertEqual((dy, dx), (-6, -3))
        self.assertGreater(response, 0.05)
        control2, candidate2 = big[10:510, 20:520], big[4:504, 12:512]
        self.assertEqual(mp.estimate_shift(control2, candidate2)[:2], (6, 8))

    def test_different_canvases_are_compared_after_alignment(self):
        big = self._big()
        control, candidate = big[0:500, 0:500], big[6:506, 3:508]      # other size and origin, same sky and noise
        with self.assertRaises(ValueError):
            mp.compare_images(control, candidate)                     # without --align a raster mismatch is refused
        r = mp.compare_images(control, candidate, align=True)
        self.assertEqual((r["alignment"]["dy"], r["alignment"]["dx"]), (-6, -3))
        self.assertFalse(r["raster"]["identical"])
        self.assertGreaterEqual(r["matched_stars"], 8)
        self.assertAlmostEqual(r["fwhm_ratio_candidate_over_control"]["median"], 1.0, places=6)
        self.assertAlmostEqual(r["noise"]["ratio_candidate_over_control"], 1.0, places=6)

    def test_alignment_refuses_unrelated_images(self):
        a, _ = make_image(1.8, 1.0, seed=1)
        noise = np.random.default_rng(5).normal(100, 1, a.shape)
        with self.assertRaises(ValueError):
            mp.compare_images(a, noise, align=True)

    def test_identical_rasters_report_zero_shift(self):
        a, _ = make_image(1.8, 1.0)
        r = mp.compare_images(a, a.copy(), align=True)
        self.assertEqual((r["alignment"]["dy"], r["alignment"]["dx"]), (0, 0))
        self.assertTrue(r["raster"]["identical"])

    def test_stars_are_rematched_when_the_registration_differs(self):
        a, _ = make_image(1.8, 1.0)
        b = np.roll(a, (2, 1), axis=(0, 1))                 # the same sky, 2.2 px off: another registration
        r = mp.compare_images(a, b)
        self.assertGreaterEqual(r["matched_stars"], 8)
        self.assertAlmostEqual(r["fwhm_ratio_candidate_over_control"]["median"], 1.0, delta=0.005)
        self.assertAlmostEqual(r["signal_ratio_candidate_over_control"]["median"], 1.0, delta=0.005)
        self.assertAlmostEqual(r["run_to_run_geometry"]["total_displacement_px"]["median"], (2 ** 2 + 1 ** 2) ** 0.5, delta=0.6)
        self.assertLess(r["star_offset_px"]["median"], 0.6, "a pure translation is absorbed by the fitted geometry")
        self.assertEqual(r["star_offset_px"]["search_radius"], mp.MATCH_R)

    def test_offsets_beyond_the_search_radius_are_not_mismeasured(self):
        a, _ = make_image(1.8, 1.0)
        b = np.roll(a, (12, 0), axis=(0, 1))                # far outside the search radius: stars are lost, not mis-measured
        r = mp.compare_images(a, b)
        self.assertGreater(r["unmatched_stars"] + r["matched_stars"], 0)
        if r["matched_stars"]:
            self.assertLessEqual(r["star_offset_px"]["p95"], mp.MATCH_R + 1e-9)

    def test_identical_images_have_zero_offset(self):
        a, _ = make_image(1.8, 1.0)
        r = mp.compare_images(a, a.copy())
        self.assertEqual(r["star_offset_px"]["median"], 0.0)
        self.assertEqual(r["unmatched_stars"], 0)

    def test_a_field_dependent_geometry_difference_is_fitted_and_reported(self):
        import cv2
        a, _ = make_image(1.8, 1.0, seed=3)
        c = (a.shape[1] / 2, a.shape[0] / 2)
        rot = cv2.getRotationMatrix2D(c, 1.0, 1.0)              # 1 degree: several pixels of displacement at the edges
        b = cv2.warpAffine(a, rot, (a.shape[1], a.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        r = mp.compare_images(a, b)
        geo = r["run_to_run_geometry"]
        self.assertIsNotNone(geo["fit"])
        self.assertAlmostEqual(abs(geo["fit"]["rotation_deg"]), 1.0, delta=0.1)
        self.assertGreater(geo["total_displacement_px"]["max"], mp.MATCH_R, "the difference is larger than the local search radius")
        self.assertLess(r["star_offset_px"]["median"], 1.5, "after the fitted geometry the residual offset is small")
        self.assertGreaterEqual(r["matched_stars"], 8)
        self.assertAlmostEqual(r["fwhm_ratio_candidate_over_control"]["median"], 1.0, delta=0.04)

    def test_identical_images_report_no_geometry_difference(self):
        a, _ = make_image(1.8, 1.0)
        geo = mp.compare_images(a, a.copy())["run_to_run_geometry"]
        self.assertEqual(geo["total_displacement_px"]["max"], 0.0)
        if geo["fit"]:
            self.assertAlmostEqual(geo["fit"]["rotation_deg"], 0.0, places=6)
            self.assertAlmostEqual(geo["fit"]["scale"], 1.0, places=6)

    def test_the_nearest_maximum_wins_over_a_brighter_neighbour(self):
        import cv2
        img = np.full((80, 80), 100.0)
        yy, xx = np.mgrid[0:80, 0:80]
        img += 200 * np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / (2 * 1.8 ** 2))     # the star
        img += 600 * np.exp(-((yy - 40) ** 2 + (xx - 49) ** 2) / (2 * 1.8 ** 2))     # brighter neighbour 9 px away
        self.assertEqual(mp.refine_center(img, 40, 40, mp.COARSE_R), (40, 40), "coarse search: the star itself, not the brighter neighbour")

    def test_identical_images_never_move_a_star(self):
        a, _ = make_image(1.8, 1.0)
        r = mp.compare_images(a, a.copy())
        self.assertEqual(r["run_to_run_geometry"]["total_displacement_px"]["max"], 0.0)

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
