// M6 tests for the plan-15 three-way candidate selection.
//   * bootstrap median CI is deterministic (seeded, versioned)
//   * multiband == raw exactly  -> Raw is selected (FWHM ratio fails at equality)
//   * multiband a genuine FWHM improvement -> Multiband promoted
//   * a background/seam regression in raw -> Uniform
//   * fewer than 20 effective stars -> FWHM N/A -> Raw stays
//   * multiband_effective drops stars where alpha_final is 0 everywhere

#include "tile_compile/reconstruction/multiband_validation.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <limits>
#include <vector>

using namespace tile_compile;
using namespace tile_compile::reconstruction;

namespace {

// A deterministic star field: `n` Gaussian stars on a jittered grid over a
// noisy background. `sigma` sets the PSF width (FWHM = 2.3548*sigma).
struct Field {
  Matrix2Df image;
  std::vector<std::pair<int, int>> centres;
};
Field make_field(int w, int h, int n, double sigma, double amp, double bg,
                 double noise, uint32_t seed) {
  Field f;
  f.image = Matrix2Df(h, w);
  std::mt19937 rng(seed);
  std::normal_distribution<double> gn(0.0, noise);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) f.image(y, x) = static_cast<float>(bg + gn(rng));

  const int cols = static_cast<int>(std::ceil(std::sqrt(double(n))));
  const int step_x = w / (cols + 1), step_y = h / (cols + 1);
  std::uniform_int_distribution<int> jit(-2, 2);
  int placed = 0;
  for (int gy = 1; gy <= cols && placed < n; ++gy)
    for (int gx = 1; gx <= cols && placed < n; ++gx) {
      const int cx = gx * step_x + jit(rng);
      const int cy = gy * step_y + jit(rng);
      if (cx < 12 || cy < 12 || cx >= w - 12 || cy >= h - 12) continue;
      const int r = static_cast<int>(std::ceil(4 * sigma));
      for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
          const double g =
              amp * std::exp(-(double(dx * dx + dy * dy)) / (2 * sigma * sigma));
          f.image(cy + dy, cx + dx) += static_cast<float>(g);
        }
      f.centres.emplace_back(cx, cy);
      ++placed;
    }
  return f;
}

}  // namespace

TEST_CASE("multiband validation: bootstrap median CI is deterministic and "
          "brackets the median") {
  std::vector<double> v;
  for (int i = 1; i <= 41; ++i) v.push_back(2.0 + 0.02 * (i % 7));
  const auto a = bootstrap_median_ci(v);
  const auto b = bootstrap_median_ci(v);
  REQUIRE(a.median == b.median);
  REQUIRE(a.ci_low == b.ci_low);
  REQUIRE(a.ci_high == b.ci_high);
  REQUIRE(a.ci_low <= a.median);
  REQUIRE(a.median <= a.ci_high);
  REQUIRE(a.relative_width >= 0.0);
}

TEST_CASE("multiband validation: multiband identical to raw (alpha 0) selects "
          "Raw, not Multiband -- the 0.95x FWHM gate fails at equality") {
  const int w = 220, h = 200;
  auto U = make_field(w, h, 49, 2.2, 400.0, 100.0, 3.0, 1);
  auto Rf = make_field(w, h, 49, 1.7, 400.0, 100.0, 3.0, 1);  // raw sharper
  const Matrix2Df &uniform = U.image;
  const Matrix2Df &raw = Rf.image;
  const Matrix2Df multiband = raw;  // alpha 0 everywhere => X_out == raw

  auto stars = prepare_validation_samples(uniform, w, h);
  INFO("detected stars = " << stars.size());
  REQUIRE(stars.size() >= 30);

  const auto res = select_reconstruction_candidate(uniform, raw, multiband, w, h,
                                                   stars);
  INFO("reason: " << res.reason);
  REQUIRE(res.selected == SelectedCandidate::kDrizzleRaw);
}

TEST_CASE("multiband validation: a genuine FWHM improvement in multiband is "
          "promoted") {
  const int w = 260, h = 240;
  // Flux-preserving sharpening: amp scales with (sigma_raw/sigma)^2 so the
  // wings genuinely shrink (tail/elongation must not regress), and multiband
  // is marginally cleaner in the background.
  const double s_u = 2.4, s_r = 2.0, s_m = 1.78;
  const double a0 = 2000.0;
  auto U = make_field(w, h, 81, s_u, a0 * (s_r * s_r) / (s_u * s_u), 100.0, 2.0, 7);
  auto Rf = make_field(w, h, 81, s_r, a0, 100.0, 2.0, 7);
  auto Mf = make_field(w, h, 81, s_m, a0 * (s_r * s_r) / (s_m * s_m), 100.0, 1.7, 7);

  auto stars = prepare_validation_samples(U.image, w, h);
  INFO("stars = " << stars.size());
  REQUIRE(stars.size() >= 30);

  const auto res = select_reconstruction_candidate(U.image, Rf.image, Mf.image,
                                                   w, h, stars);
  INFO("reason: " << res.reason << "  fwhm_r=" << res.raw.median_fwhm.value
                  << " fwhm_m=" << res.multiband.median_fwhm.value
                  << " n=" << res.multiband_star_sample_count);
  REQUIRE(res.multiband.median_fwhm.applicable);
  REQUIRE(res.multiband.median_fwhm.value <
          0.95 * res.raw.median_fwhm.value);
  REQUIRE(res.selected == SelectedCandidate::kDrizzleMultiband);
}

TEST_CASE("multiband validation: a raw background regression forces Uniform") {
  const int w = 220, h = 200;
  auto U = make_field(w, h, 49, 2.2, 400.0, 100.0, 2.0, 3);
  auto Rf = make_field(w, h, 49, 2.0, 400.0, 100.0, 8.0, 3);  // 4x noisier
  auto Mf = make_field(w, h, 49, 1.8, 400.0, 100.0, 8.0, 3);

  auto stars = prepare_validation_samples(U.image, w, h);
  const auto res = select_reconstruction_candidate(U.image, Rf.image, Mf.image,
                                                   w, h, stars);
  INFO("reason: " << res.reason);
  REQUIRE(res.selected == SelectedCandidate::kDrizzleUniform);
}

TEST_CASE("multiband validation: fewer than 20 effective stars -> FWHM N/A -> "
          "Raw stays (small samples are never an implicit pass)") {
  const int w = 260, h = 240;
  auto U = make_field(w, h, 81, 2.2, 400.0, 100.0, 2.0, 11);
  auto Rf = make_field(w, h, 81, 2.0, 400.0, 100.0, 2.0, 11);
  auto Mf = make_field(w, h, 81, 1.6, 400.0, 100.0, 2.0, 11);

  auto stars = prepare_validation_samples(U.image, w, h);
  REQUIRE(stars.size() >= 30);
  // Mark all but 8 stars as non-effective (alpha 0 there).
  int kept = 0;
  for (auto &s : stars) s.multiband_effective = (kept++ < 8);

  const auto res = select_reconstruction_candidate(U.image, Rf.image, Mf.image,
                                                   w, h, stars);
  INFO("reason: " << res.reason << "  eff sample = "
                  << res.multiband_star_sample_count);
  REQUIRE_FALSE(res.multiband.median_fwhm.applicable);
  REQUIRE_FALSE(res.multiband.median_fwhm.reason_if_not_applicable.empty());
  REQUIRE(res.selected == SelectedCandidate::kDrizzleRaw);
}

TEST_CASE("multiband validation: prepare_validation_samples sets "
          "multiband_effective from the per-band alpha maps") {
  const int w = 160, h = 140;
  auto U = make_field(w, h, 36, 2.2, 400.0, 100.0, 2.0, 5);
  auto stars = prepare_validation_samples(U.image, w, h);
  REQUIRE(stars.size() >= 10);

  // One band map: alpha non-zero only in the left half.
  std::vector<float> band(static_cast<size_t>(w) * h, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w / 2; ++x) band[static_cast<size_t>(y) * w + x] = 0.4f;

  auto stars2 = prepare_validation_samples(U.image, w, h, {}, {band});
  int eff = 0, ineff = 0;
  for (const auto &s : stars2) (s.multiband_effective ? eff : ineff)++;
  INFO("effective=" << eff << " ineffective=" << ineff);
  REQUIRE(eff > 0);
  REQUIRE(ineff > 0);
  for (const auto &s : stars2)
    REQUIRE(s.multiband_effective == (s.x < w / 2));
}

TEST_CASE("multiband validation: a degenerate (noise-free) background makes "
          "background_rms non-applicable -> mandatory safety metric N/A -> "
          "Uniform (no misleading ratio path)") {
  const int w = 240, h = 220;
  auto U = make_field(w, h, 64, 2.2, 400.0, 100.0, 0.0, 4);  // no noise
  auto Rf = make_field(w, h, 64, 2.0, 400.0, 100.0, 0.0, 4);
  auto Mf = make_field(w, h, 64, 1.7, 400.0, 100.0, 0.0, 4);
  auto stars = prepare_validation_samples(U.image, w, h);
  const auto res = select_reconstruction_candidate(U.image, Rf.image, Mf.image,
                                                   w, h, stars);
  INFO("reason: " << res.reason);
  // A noise-free control has ~0 background RMS -> the metric is non-applicable
  // and never reaches the ratio() path (whose denominator would be ~0). A
  // mandatory safety metric that is N/A drops Raw to the Uniform control.
  REQUIRE(res.selected == SelectedCandidate::kDrizzleUniform);
  REQUIRE_FALSE(res.raw.background_rms.applicable);
  REQUIRE(std::isfinite(res.raw.background_rms.value));
}

TEST_CASE("multiband validation: a few percent of scattered off-support NaN "
          "(the real OSC working-luminance case) still yields usable stars -- "
          "the support gate must not reject the Uniform control") {
  const int w = 260, h = 240;
  const double s_r = 2.0, s_m = 1.78, a0 = 2000.0;
  auto U = make_field(w, h, 81, 2.4, a0 * (s_r * s_r) / (2.4 * 2.4), 100.0, 2.0, 7);
  auto Rf = make_field(w, h, 81, s_r, a0, 100.0, 2.0, 7);
  auto Mf = make_field(w, h, 81, s_m, a0 * (s_r * s_r) / (s_m * s_m), 100.0, 1.7, 7);
  // Punch ~6% scattered NaN into every candidate (never on a star centre).
  std::mt19937 rng(99);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<std::pair<int, int>> centres = U.centres;
  auto is_centre = [&](int x, int y) {
    for (auto [cx, cy] : centres)
      if (std::abs(cx - x) <= 1 && std::abs(cy - y) <= 1) return true;
    return false;
  };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (u(rng) < 0.06 && !is_centre(x, y)) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        U.image(y, x) = nan; Rf.image(y, x) = nan; Mf.image(y, x) = nan;
      }
  auto stars = prepare_validation_samples(U.image, w, h);
  INFO("stars = " << stars.size());
  REQUIRE(stars.size() >= 30);
  const auto res = select_reconstruction_candidate(U.image, Rf.image, Mf.image,
                                                   w, h, stars);
  INFO("reason: " << res.reason);
  // The support gate no longer rejects everything: all three candidates keep
  // usable stars despite the holes, and FWHM stays applicable.
  REQUIRE(res.uniform.support_ok);
  REQUIRE(res.raw.support_ok);
  REQUIRE(res.multiband.support_ok);
  REQUIRE(res.raw.median_fwhm.applicable);
  REQUIRE(res.multiband.median_fwhm.applicable);
  REQUIRE(res.selected == SelectedCandidate::kDrizzleMultiband);
}

TEST_CASE("multiband validation: a genuine seam jump at the support boundary "
          "blocks multiband even with a better FWHM") {
  const int w = 260, h = 240;
  const double s_r = 2.0, s_m = 1.78, a0 = 2000.0;
  auto U = make_field(w, h, 81, 2.4, a0 * (s_r * s_r) / (2.4 * 2.4), 100.0, 2.0, 7);
  auto Rf = make_field(w, h, 81, s_r, a0, 100.0, 2.0, 7);
  auto Mf = make_field(w, h, 81, s_m, a0 * (s_r * s_r) / (s_m * s_m), 100.0, 1.7, 7);

  // Support mask: a rectangular hole in the middle => an interior boundary.
  std::vector<uint8_t> mask(static_cast<size_t>(w) * h, 1u);
  for (int y = 90; y < 150; ++y)
    for (int x = 110; x < 170; ++x) mask[static_cast<size_t>(y) * w + x] = 0u;
  // Multiband adds a bright step right along the hole's inner edge.
  for (int y = 88; y < 152; ++y) {
    Mf.image(y, 108) += 240.0f; Mf.image(y, 109) += 240.0f;
    Mf.image(y, 171) += 240.0f; Mf.image(y, 172) += 240.0f;
  }
  for (int x = 108; x < 173; ++x) {
    Mf.image(88, x) += 240.0f; Mf.image(89, x) += 240.0f;
    Mf.image(150, x) += 240.0f; Mf.image(151, x) += 240.0f;
  }

  auto stars = prepare_validation_samples(U.image, w, h, mask);
  const auto res = select_reconstruction_candidate(U.image, Rf.image, Mf.image,
                                                   w, h, stars, {}, mask);
  INFO("reason: " << res.reason << "  seam_u=" << res.uniform.seam_score.value
                  << " seam_m=" << res.multiband.seam_score.value);
  REQUIRE(res.multiband.median_fwhm.value < res.raw.median_fwhm.value);  // sharper
  REQUIRE(res.selected == SelectedCandidate::kDrizzleRaw);
  REQUIRE(res.reason.find("seam") != std::string::npos);
}

TEST_CASE("multiband validation: a real support boundary (NaN off-support) "
          "makes seam_score measurable, not the 0 sentinel") {
  // Regression pin for the seam defect (30.46): boundary_seam_score used to
  // sample the boundary pixels themselves -- which have a NaN neighbour by
  // definition -- so on every masked real field it collapsed to 0.0 and
  // ratio(0,0)=inf spuriously rejected Raw for a "seam regression". The fix
  // samples the interior edge (one pixel in, full on-support stencil).
  const int w = 260, h = 240;
  const double s_r = 2.0, s_m = 1.78, a0 = 2000.0;
  auto U = make_field(w, h, 81, 2.4, a0 * (s_r * s_r) / (2.4 * 2.4), 100.0, 2.0, 7);
  auto Rf = make_field(w, h, 81, s_r, a0, 100.0, 2.0, 7);
  auto Mf = make_field(w, h, 81, s_m, a0 * (s_r * s_r) / (s_m * s_m), 100.0, 1.7, 7);

  // A rectangular unsupported hole; off-support pixels are NaN, as in real
  // OSC working luminance. No artificial seam step is injected.
  std::vector<uint8_t> mask(static_cast<size_t>(w) * h, 1u);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int y = 90; y < 150; ++y)
    for (int x = 110; x < 170; ++x) {
      mask[static_cast<size_t>(y) * w + x] = 0u;
      U.image(y, x) = nan; Rf.image(y, x) = nan; Mf.image(y, x) = nan;
    }

  auto stars = prepare_validation_samples(U.image, w, h, mask);
  const auto res = select_reconstruction_candidate(U.image, Rf.image, Mf.image,
                                                   w, h, stars, {}, mask);
  INFO("reason: " << res.reason << "  seam_u=" << res.uniform.seam_score.value);
  // The seam metric is now an actual measurement, not the 0 sentinel.
  REQUIRE(res.uniform.seam_score.applicable);
  REQUIRE(res.raw.seam_score.applicable);
  REQUIRE(res.multiband.seam_score.applicable);
  REQUIRE(std::isfinite(res.uniform.seam_score.value));
  REQUIRE(res.uniform.seam_score.value > 0.0);
  REQUIRE(res.raw.seam_score.value > 0.0);
  // Raw is NOT rejected for a seam regression when there is no real seam.
  REQUIRE(res.reason.find("seam") == std::string::npos);
}

TEST_CASE("multiband validation: scattered per-pixel dropouts dilute the seam "
          "gate to inert -- documents the open seam-form plan question (30.46)") {
  // M42 real data (2026-09-06): the OSC working-luminance support mask is shot
  // through with ~5% scattered single-pixel dropouts, so the interior-edge
  // locus (~1M px) is dominated by dropout-hole edges, not the true support
  // boundary. seam_score then lands at ~1.03 for ALL three candidates
  // (ratios within 0.4%), so a genuine step at the real boundary cannot move
  // the gate. This is a seam-*form* issue (which locus to measure), flagged
  // plan-unconfirmed in 30.42 -- NOT re-patched here. This test pins the
  // current behaviour so a later form change is a deliberate, visible edit.
  const int w = 260, h = 240;
  const double s_r = 2.0, s_m = 1.78, a0 = 2000.0;
  auto U = make_field(w, h, 81, 2.4, a0 * (s_r * s_r) / (2.4 * 2.4), 100.0, 2.0, 7);
  auto Rf = make_field(w, h, 81, s_r, a0, 100.0, 2.0, 7);
  auto Mf = make_field(w, h, 81, s_m, a0 * (s_r * s_r) / (s_m * s_m), 100.0, 1.7, 7);

  std::vector<uint8_t> mask(static_cast<size_t>(w) * h, 1u);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  // A genuine contiguous unsupported region (the "real" boundary) ...
  for (int y = 100; y < 140; ++y)
    for (int x = 120; x < 160; ++x) {
      mask[static_cast<size_t>(y) * w + x] = 0u;
      U.image(y, x) = nan; Rf.image(y, x) = nan; Mf.image(y, x) = nan;
    }
  // ... plus ~5% scattered single-pixel dropouts everywhere else.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<std::pair<int, int>> centres = U.centres;
  auto is_centre = [&](int x, int y) {
    for (auto [cx, cy] : centres)
      if (std::abs(cx - x) <= 1 && std::abs(cy - y) <= 1) return true;
    return false;
  };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (mask[static_cast<size_t>(y) * w + x] && u(rng) < 0.05 &&
          !is_centre(x, y)) {
        mask[static_cast<size_t>(y) * w + x] = 0u;
        U.image(y, x) = nan; Rf.image(y, x) = nan; Mf.image(y, x) = nan;
      }
  // Inject a real step along the contiguous hole's edge -- into MULTIBAND only.
  for (int y = 98; y < 142; ++y) { Mf.image(y, 118) += 200.0f; Mf.image(y, 161) += 200.0f; }

  auto stars = prepare_validation_samples(U.image, w, h, mask);
  const auto res = select_reconstruction_candidate(U.image, Rf.image, Mf.image,
                                                   w, h, stars, {}, mask);
  INFO("seam u/r/m = " << res.uniform.seam_score.value << " / "
                       << res.raw.seam_score.value << " / "
                       << res.multiband.seam_score.value);
  REQUIRE(res.uniform.seam_score.applicable);
  const double su = res.uniform.seam_score.value;
  const double sm = res.multiband.seam_score.value;
  // Self-normalised, so a healthy score sits near 1 -- rule out the 0 sentinel
  // and any runaway.
  REQUIRE(su > 0.5);
  REQUIRE(su < 2.0);
  // The DISCRIMINATING pin: multiband carries a real +200 step along its
  // interior-edge columns (x=118, x=161), uniform carries none -- yet the two
  // seam scores agree to within 5%, because the step is ~44 px in a ~1M-px
  // dropout-dominated edge set. A seam-form fix that restricts the locus to
  // the true boundary would push sm/su well past 1.05 and fail here on
  // purpose, forcing a deliberate update rather than a silent behaviour flip.
  REQUIRE(std::abs(sm / su - 1.0) < 0.05);
}

TEST_CASE("multiband_validation_config_hash is stable and config-sensitive") {
  const std::string base = multiband_validation_config_hash();
  // 64 lowercase hex chars, deterministic across calls.
  REQUIRE(base.size() == 64);
  REQUIRE(base == multiband_validation_config_hash(MultibandValidationConfig{}));

  MultibandValidationConfig tweaked;
  tweaked.fwhm_ratio_max = 0.90;  // a real threshold change
  REQUIRE(multiband_validation_config_hash(tweaked) != base);

  MultibandValidationConfig same_defaults;  // re-stating the defaults -> same hash
  REQUIRE(multiband_validation_config_hash(same_defaults) == base);
}
