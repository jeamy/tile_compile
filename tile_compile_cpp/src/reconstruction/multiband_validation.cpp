#include "tile_compile/reconstruction/multiband_validation.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/reconstruction/aqmh_validation.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace tile_compile::reconstruction {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// SplitMix64 --- deterministic, seeded from the versioned constant so the
// bootstrap CI is reproducible (a wall-clock RNG would make it untestable).
struct SplitMix64 {
  uint64_t s;
  explicit SplitMix64(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  // uniform in [0, n)
  uint64_t below(uint64_t n) { return next() % n; }
};

double median_of(std::vector<double> v) {
  if (v.empty()) return kNaN;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return (n & 1u) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double percentile_sorted(const std::vector<double> &sorted, double p) {
  if (sorted.empty()) return kNaN;
  if (sorted.size() == 1) return sorted.front();
  const double idx = p * static_cast<double>(sorted.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
  const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double frac = idx - static_cast<double>(lo);
  return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

// Extract the (2R+1)^2 luminance patch around a star into `patch` (CV_32F).
// Real drizzle luminance has scattered off-support NaN (an OSC working
// luminance needs R AND G AND B co-support at every pixel), so requiring the
// WHOLE patch finite rejects essentially every star on real data --- even the
// safe Uniform control (M31 run 2026-09-06: 0 of 250). Instead: the centre
// pixel must be finite and at least
// `kMultibandValidationStarPatchMinFiniteFraction` of the patch; the sparse
// remaining holes are filled with the finite-pixel median (a background
// estimate that cannot manufacture a peak). Returns false when the patch is
// out of bounds or too sparse to attempt a fit.
bool extract_star_patch(const Matrix2Df &image, int width, int height, int sx,
                        int sy, int R, cv::Mat &patch) {
  const int x0 = sx - R, y0 = sy - R, x1 = sx + R + 1, y1 = sy + R + 1;
  if (x0 < 0 || y0 < 0 || x1 > width || y1 > height) return false;
  if (!std::isfinite(image(sy, sx))) return false;  // centre must be real
  const int n = (x1 - x0) * (y1 - y0);
  patch.create(y1 - y0, x1 - x0, CV_32F);
  std::vector<float> finite_vals;
  finite_vals.reserve(static_cast<std::size_t>(n));
  int n_finite = 0;
  for (int yy = y0; yy < y1; ++yy)
    for (int xx = x0; xx < x1; ++xx) {
      const float v = image(yy, xx);
      patch.at<float>(yy - y0, xx - x0) = v;
      if (std::isfinite(v)) { ++n_finite; finite_vals.push_back(v); }
    }
  if (static_cast<double>(n_finite) <
      kMultibandValidationStarPatchMinFiniteFraction * static_cast<double>(n))
    return false;
  if (n_finite == n) return true;
  std::nth_element(finite_vals.begin(),
                   finite_vals.begin() + finite_vals.size() / 2,
                   finite_vals.end());
  const float fill = finite_vals[finite_vals.size() / 2];
  for (int yy = 0; yy < patch.rows; ++yy)
    for (int xx = 0; xx < patch.cols; ++xx)
      if (!std::isfinite(patch.at<float>(yy, xx)))
        patch.at<float>(yy, xx) = fill;
  return true;
}

// Per-star FWHM aligned to `stars` (NaN where no usable fit / not effective).
std::vector<double> per_star_fwhm_aligned(const Matrix2Df &image, int width,
                                          int height,
                                          const std::vector<ValidationStar> &stars,
                                          bool effective_only) {
  std::vector<double> out(stars.size(), kNaN);
  const int R = kMultibandValidationStarPatchRadius;
  cv::Mat patch;
  for (std::size_t i = 0; i < stars.size(); ++i) {
    const auto &s = stars[i];
    if (effective_only && !s.multiband_effective) continue;
    if (!extract_star_patch(image, width, height, s.x, s.y, R, patch)) continue;
    const float f = metrics::estimate_fwhm_from_patch(patch);
    if (std::isfinite(f) && f > 0.0f) out[i] = static_cast<double>(f);
  }
  return out;
}

}  // namespace

MedianCi bootstrap_median_ci(const std::vector<double> &values) {
  MedianCi out;
  if (values.empty()) {
    out.median = out.ci_low = out.ci_high = kNaN;
    out.relative_width = kNaN;
    return out;
  }
  out.median = median_of(values);
  SplitMix64 rng(kMultibandValidationBootstrapSeed);
  std::vector<double> resample_medians;
  resample_medians.reserve(kMultibandValidationBootstrapResamples);
  std::vector<double> draw(values.size());
  for (int b = 0; b < kMultibandValidationBootstrapResamples; ++b) {
    for (std::size_t k = 0; k < values.size(); ++k)
      draw[k] = values[rng.below(values.size())];
    resample_medians.push_back(median_of(draw));
  }
  std::sort(resample_medians.begin(), resample_medians.end());
  out.ci_low = percentile_sorted(resample_medians, 0.025);
  out.ci_high = percentile_sorted(resample_medians, 0.975);
  out.relative_width =
      out.median > 0.0 ? (out.ci_high - out.ci_low) / out.median : kNaN;
  return out;
}

std::vector<double> per_star_fwhm(const Matrix2Df &image, int width, int height,
                                  const std::vector<ValidationStar> &stars,
                                  bool effective_only) {
  auto aligned = per_star_fwhm_aligned(image, width, height, stars, effective_only);
  std::vector<double> out;
  for (double v : aligned)
    if (std::isfinite(v)) out.push_back(v);
  return out;
}

std::vector<ValidationStar> prepare_validation_samples(
    const Matrix2Df &uniform_control, int width, int height,
    const std::vector<uint8_t> &validation_mask,
    const std::vector<std::vector<float>> &alpha_final_by_band,
    double alpha_effective_eps) {
  // Star detection is a general validation helper (plan 15.4) --- reused from
  // the existing pairwise path; no legacy decision code is touched.
  const auto ref =
      prepare_aqmh_validation_reference(uniform_control, validation_mask);
  std::vector<ValidationStar> stars;
  stars.reserve(ref.stars.size());
  for (const auto &s : ref.stars) {
    ValidationStar v;
    v.x = s.x;
    v.y = s.y;
    v.peak = s.peak;
    v.multiband_effective = true;
    if (!alpha_final_by_band.empty()) {
      bool any_alpha = false;
      for (const auto &band : alpha_final_by_band) {
        if (band.empty()) continue;
        for (int dy = -1; dy <= 1 && !any_alpha; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int xx = s.x + dx, yy = s.y + dy;
            if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
            const float a = band[static_cast<std::size_t>(yy) * width + xx];
            if (std::isfinite(a) && std::abs(a) > alpha_effective_eps) {
              any_alpha = true;
              break;
            }
          }
      }
      v.multiband_effective = any_alpha;
    }
    stars.push_back(v);
  }
  return stars;
}

namespace {

// Plan-15 seam metric: mean |Laplacian| on the *interior edge* of the
// reconstruction support boundary (fully-supported pixels one step inside the
// boundary --- their 5-point stencil is entirely on-support, so the Laplacian
// is actually computable), divided by the mean |Laplacian| of the candidate's
// OWN deep interior. Self-normalising => a uniform PSF sharpening scales
// numerator and denominator together and leaves the ratio ~1; a real step /
// ring artefact at the mask edge (§14.7 "Maskenkante erzeugt keinen
// Seam-Sprung") spikes only the numerator.
//
// The earlier version sampled the boundary pixels THEMSELVES; those have a
// non-support (NaN) neighbour by definition, so `laplacian_abs` returned NaN
// at every one and the score collapsed to the 0.0 sentinel on every real
// masked field (M42 2026-09-06: all three candidates seam_score==0). Same
// defect class as the whole-patch-finite `star_support_ok` bug (30.45).
//
// Returns NaN when the score cannot be measured (no interior edge, or no
// finite deep-interior reference) --- the caller then marks the metric
// non-applicable rather than treating 0 as a measurement. `edge_cache` holds
// the interior-edge index set, filled on the first call so U/R/M share it.
double laplacian_abs(const Matrix2Df &im, int x, int y) {
  const float c = im(y, x), l = im(y, x - 1), rr = im(y, x + 1),
              u = im(y - 1, x), d = im(y + 1, x);
  if (!(std::isfinite(c) && std::isfinite(l) && std::isfinite(rr) &&
        std::isfinite(u) && std::isfinite(d)))
    return std::numeric_limits<double>::quiet_NaN();
  return std::abs(4.0 * c - l - rr - u - d);
}
double boundary_seam_score(const Matrix2Df &image, int width, int height,
                           const std::vector<uint8_t> &mask,
                           std::vector<std::size_t> *edge_cache) {
  const double kNotMeasurable = std::numeric_limits<double>::quiet_NaN();
  std::vector<std::size_t> local;
  std::vector<std::size_t> &edge = edge_cache ? *edge_cache : local;
  auto valid = [&](int x, int y) {
    return x >= 0 && y >= 0 && x < width && y < height &&
           (mask.empty() || mask[static_cast<std::size_t>(y) * width + x] != 0u);
  };
  // A boundary pixel is on-support with >=1 off-support 4-neighbour.
  auto is_boundary = [&](int x, int y) {
    return valid(x, y) && !(valid(x - 1, y) && valid(x + 1, y) &&
                            valid(x, y - 1) && valid(x, y + 1));
  };
  if (edge.empty()) {
    if (mask.empty() ||
        mask.size() != static_cast<std::size_t>(width) * height)
      return kNotMeasurable;  // no support boundary at all
    // Interior edge: fully-supported pixel adjacent to a boundary pixel.
    for (int y = 2; y < height - 2; ++y)
      for (int x = 2; x < width - 2; ++x) {
        if (!(valid(x, y) && valid(x - 1, y) && valid(x + 1, y) &&
              valid(x, y - 1) && valid(x, y + 1)))
          continue;  // stencil must be on-support
        if (is_boundary(x - 1, y) || is_boundary(x + 1, y) ||
            is_boundary(x, y - 1) || is_boundary(x, y + 1))
          edge.push_back(static_cast<std::size_t>(y) * width + x);
      }
  }
  if (edge.size() <
      static_cast<std::size_t>(kMultibandValidationSeamMinBoundaryPixels))
    return kNotMeasurable;

  double b_acc = 0.0;
  std::size_t b_n = 0;
  for (std::size_t idx : edge) {
    const int x = static_cast<int>(idx % width), y = static_cast<int>(idx / width);
    const double v = laplacian_abs(image, x, y);
    if (std::isfinite(v)) { b_acc += v; ++b_n; }
  }
  // Interior reference: fully-surrounded valid pixels, strided for speed.
  double i_acc = 0.0;
  std::size_t i_n = 0;
  const int stride = std::max(
      1, (width * height) / kMultibandValidationSeamInteriorStrideTarget);
  for (int y = 1; y < height - 1; ++y)
    for (int x = 1; x < width - 1; ++x) {
      if (((static_cast<std::size_t>(y) * width + x) % stride) != 0) continue;
      if (!(valid(x, y) && valid(x - 1, y) && valid(x + 1, y) &&
            valid(x, y - 1) && valid(x, y + 1)))
        continue;
      const double v = laplacian_abs(image, x, y);
      if (std::isfinite(v)) { i_acc += v; ++i_n; }
    }
  if (!b_n || !i_n) return kNotMeasurable;
  const double interior = std::max(i_acc / i_n, 1e-12);
  return (b_acc / b_n) / interior;
}

bool field_numerics_ok(const Matrix2Df &image) {
  for (int y = 0; y < image.rows(); ++y)
    for (int x = 0; x < image.cols(); ++x)
      if (std::isinf(image(y, x))) return false;  // NaN = off-support, allowed
  return true;
}

// Coverage over the star patches: every patch must be fully finite.
bool star_support_ok(const Matrix2Df &image, int width, int height,
                     const std::vector<ValidationStar> &stars) {
  const int R = kMultibandValidationStarPatchRadius;
  cv::Mat patch;
  int usable = 0;
  for (const auto &s : stars)
    if (extract_star_patch(image, width, height, s.x, s.y, R, patch)) ++usable;
  return usable > 0;
}

}  // namespace

MultibandValidationResult select_reconstruction_candidate(
    const Matrix2Df &drizzle_uniform, const Matrix2Df &drizzle_raw,
    const Matrix2Df &drizzle_multiband, int width, int height,
    const std::vector<ValidationStar> &stars,
    const MultibandValidationConfig &cfg,
    const std::vector<uint8_t> &validation_mask) {
  MultibandValidationResult r;
  r.stars_total = static_cast<int>(stars.size());
  for (const auto &s : stars)
    if (s.multiband_effective) ++r.stars_multiband_effective;

  // --- field / tail metrics at the fixed star set (measurement reuse) -------
  AqmhValidationReference ref;
  ref.width = width;
  ref.height = height;
  for (const auto &s : stars) ref.stars.push_back({s.x, s.y, s.peak});
  ref.metrics =
      compare_aqmh_to_reference(drizzle_uniform, ref, validation_mask).aqmh;
  const auto raw_cmp =
      compare_aqmh_to_reference(drizzle_raw, ref, validation_mask);
  const auto mb_cmp =
      compare_aqmh_to_reference(drizzle_multiband, ref, validation_mask);

  auto fill_field = [](ValidationMetric &m, double value, bool applicable,
                       int count, const char *why) {
    m.value = value;
    m.applicable = applicable;
    m.sample_count = count;
    if (!applicable) m.reason_if_not_applicable = why;
  };

  // background_rms is compared against UNIFORM (15.3.4). Robust-noise
  // measurement is semantically correct to reuse (a sharper PSF does not
  // change background noise).
  fill_field(r.uniform.background_rms, ref.metrics.background_rms, true,
             ref.metrics.star_count, "");
  fill_field(r.raw.background_rms, raw_cmp.aqmh.background_rms,
             raw_cmp.background_rms_applicable, raw_cmp.aqmh.star_count,
             "control background_rms degenerate");
  fill_field(r.multiband.background_rms, mb_cmp.aqmh.background_rms,
             mb_cmp.background_rms_applicable, mb_cmp.aqmh.star_count,
             "control background_rms degenerate");

  // seam: a fresh interior-edge metric (plan 14.7), NOT the legacy
  // global-gradient proxy which would penalise every genuine sharpening.
  // boundary_seam_score returns NaN when it cannot be measured; the metric is
  // then non-applicable (not 0). A seam metric is only applicable when the
  // interior-edge set is large enough AND every candidate produced a finite
  // score (they share the geometry, so they stand or fall together).
  std::vector<std::size_t> seam_edge;
  const double seam_u =
      boundary_seam_score(drizzle_uniform, width, height, validation_mask,
                          &seam_edge);
  const double seam_r = boundary_seam_score(drizzle_raw, width, height,
                                            validation_mask, &seam_edge);
  const double seam_m = boundary_seam_score(drizzle_multiband, width, height,
                                            validation_mask, &seam_edge);
  const bool seam_applicable =
      seam_edge.size() >=
          static_cast<std::size_t>(kMultibandValidationSeamMinBoundaryPixels) &&
      std::isfinite(seam_u) && std::isfinite(seam_r) && std::isfinite(seam_m);
  // "no support boundary at all" (seam_edge empty) => the seam constraint does
  // not apply and must NOT reject Raw; "edge present but unmeasurable" is
  // suspicious and IS treated as a failed mandatory safety gate below.
  const bool has_seam_edge = !seam_edge.empty();
  fill_field(r.uniform.seam_score, seam_u, seam_applicable,
             static_cast<int>(seam_edge.size()), "seam not measurable");
  fill_field(r.raw.seam_score, seam_r, seam_applicable,
             static_cast<int>(seam_edge.size()), "seam not measurable");
  fill_field(r.multiband.seam_score, seam_m, seam_applicable,
             static_cast<int>(seam_edge.size()), "seam not measurable");

  const int nstar = mb_cmp.aqmh.star_count;
  const bool tail_count_ok =
      nstar >= cfg.min_stars_p90_tail_elongation &&
      raw_cmp.aqmh.star_count >= cfg.min_stars_p90_tail_elongation;
  fill_field(r.raw.tail, raw_cmp.aqmh.tail11_abs_median, tail_count_ok,
             raw_cmp.aqmh.star_count, "fewer than 30 matched stars");
  fill_field(r.multiband.tail, mb_cmp.aqmh.tail11_abs_median, tail_count_ok,
             nstar, "fewer than 30 matched stars");
  fill_field(r.raw.elongation, raw_cmp.aqmh.elongation_median, tail_count_ok,
             raw_cmp.aqmh.star_count, "fewer than 30 matched stars");
  fill_field(r.multiband.elongation, mb_cmp.aqmh.elongation_median, tail_count_ok,
             nstar, "fewer than 30 matched stars");

  // --- per-star FWHM (new in plan 15): compare raw vs multiband on the SAME
  //     effective-star subset so the populations match. ----------------------
  const auto fwhm_r_al =
      per_star_fwhm_aligned(drizzle_raw, width, height, stars, true);
  const auto fwhm_m_al =
      per_star_fwhm_aligned(drizzle_multiband, width, height, stars, true);
  std::vector<double> fwhm_r, fwhm_m;
  for (std::size_t i = 0; i < stars.size(); ++i)
    if (std::isfinite(fwhm_r_al[i]) && std::isfinite(fwhm_m_al[i])) {
      fwhm_r.push_back(fwhm_r_al[i]);
      fwhm_m.push_back(fwhm_m_al[i]);
    }
  r.multiband_star_sample_count = static_cast<int>(fwhm_m.size());

  const auto ci_r = bootstrap_median_ci(fwhm_r);
  const auto ci_m = bootstrap_median_ci(fwhm_m);
  const int fwhm_n = static_cast<int>(fwhm_m.size());

  const bool fwhm_count_ok = fwhm_n >= cfg.min_stars_fwhm;
  const bool fwhm_ci_ok = fwhm_count_ok && std::isfinite(ci_m.relative_width) &&
                          ci_m.relative_width <= cfg.max_fwhm_ci_relative_width;
  std::string fwhm_why;
  if (!fwhm_count_ok)
    fwhm_why = "fewer than " + std::to_string(cfg.min_stars_fwhm) +
               " effective stars (" + std::to_string(fwhm_n) + " of " +
               std::to_string(r.stars_total) + ")";
  else if (!fwhm_ci_ok)
    fwhm_why = "bootstrap FWHM-median CI relative width exceeds "
               "max_fwhm_ci_relative_width";
  r.raw.median_fwhm = {ci_r.median, fwhm_ci_ok, fwhm_n, ci_r.ci_low,
                       ci_r.ci_high, fwhm_ci_ok ? std::string{} : fwhm_why};
  r.multiband.median_fwhm = {ci_m.median, fwhm_ci_ok, fwhm_n, ci_m.ci_low,
                             ci_m.ci_high, fwhm_ci_ok ? std::string{} : fwhm_why};

  const bool p90_count_ok = fwhm_n >= cfg.min_stars_p90_tail_elongation;
  std::vector<double> fr_sorted = fwhm_r, fm_sorted = fwhm_m;
  std::sort(fr_sorted.begin(), fr_sorted.end());
  std::sort(fm_sorted.begin(), fm_sorted.end());
  const char *p90_why = "fewer than 30 effective stars";
  fill_field(r.raw.p90_fwhm, percentile_sorted(fr_sorted, 0.90), p90_count_ok,
             fwhm_n, p90_why);
  fill_field(r.multiband.p90_fwhm, percentile_sorted(fm_sorted, 0.90),
             p90_count_ok, fwhm_n, p90_why);

  // --- support / numerics -------------------------------------------------
  r.uniform.numerics_ok = field_numerics_ok(drizzle_uniform);
  r.raw.numerics_ok = field_numerics_ok(drizzle_raw);
  r.multiband.numerics_ok = field_numerics_ok(drizzle_multiband);
  r.uniform.support_ok = star_support_ok(drizzle_uniform, width, height, stars);
  r.raw.support_ok = star_support_ok(drizzle_raw, width, height, stars);
  r.multiband.support_ok =
      star_support_ok(drizzle_multiband, width, height, stars);

  // --- 15.3.2 Raw vs Uniform (safety only; N/A star metrics never reject) --
  auto ratio = [](double a, double b) {
    return (b > 0.0) ? a / b : std::numeric_limits<double>::infinity();
  };
  std::string raw_fail;
  if (!r.raw.support_ok) raw_fail = "raw star support invalid";
  else if (!r.raw.numerics_ok) raw_fail = "raw field has non-finite values";
  else if (!r.raw.background_rms.applicable)
    raw_fail = "raw background_rms not applicable (mandatory safety metric)";
  else if (ratio(r.raw.background_rms.value, r.uniform.background_rms.value) >
           cfg.background_rms_ratio_max)
    raw_fail = "raw background_rms regression vs uniform";
  else if (has_seam_edge && !seam_applicable)
    raw_fail = "raw seam_score not measurable (mandatory safety metric)";
  else if (seam_applicable &&
           ratio(r.raw.seam_score.value, r.uniform.seam_score.value) >
               cfg.seam_ratio_max)
    raw_fail = "raw seam_score regression vs uniform at the support boundary";

  if (!raw_fail.empty()) {
    r.selected = SelectedCandidate::kDrizzleUniform;
    r.reason = "raw rejected -> uniform: " + raw_fail;
    return r;
  }

  // --- 15.3.4 Multiband promotion (ALL must hold; N/A => no positive
  //     evidence => Raw stays) --------------------------------------------
  std::string mb_fail;
  auto need = [&](bool applicable, bool pass, const char *na_msg,
                  const char *fail_msg) {
    if (!mb_fail.empty()) return;
    if (!applicable) mb_fail = na_msg;
    else if (!pass) mb_fail = fail_msg;
  };
  if (!r.multiband.support_ok) mb_fail = "multiband star support invalid";
  else if (!r.multiband.numerics_ok)
    mb_fail = "multiband field has non-finite values";
  need(r.multiband.median_fwhm.applicable,
       r.multiband.median_fwhm.value <=
           cfg.fwhm_ratio_max * r.raw.median_fwhm.value,
       "median FWHM not applicable (no positive multiband evidence)",
       "median FWHM improvement below 0.95x raw");
  need(r.multiband.p90_fwhm.applicable,
       r.multiband.p90_fwhm.value <=
           cfg.p90_fwhm_ratio_max * r.raw.p90_fwhm.value,
       "p90 FWHM not applicable", "p90 FWHM regression vs raw");
  need(r.multiband.tail.applicable,
       r.multiband.tail.value <= cfg.tail_ratio_max * r.raw.tail.value,
       "tail not applicable", "tail regression vs raw");
  need(r.multiband.elongation.applicable,
       r.multiband.elongation.value <=
           cfg.elongation_ratio_max * r.raw.elongation.value,
       "elongation not applicable", "elongation regression vs raw");
  need(r.multiband.background_rms.applicable,
       ratio(r.multiband.background_rms.value, r.uniform.background_rms.value) <=
           cfg.background_rms_ratio_max,
       "multiband background_rms not applicable",
       "multiband background_rms regression vs uniform");
  // No support boundary => the seam inequality is vacuous (applicable, passes).
  // Edge present but unmeasurable => no positive multiband evidence => Raw.
  need(!has_seam_edge || seam_applicable,
       !has_seam_edge ||
           ratio(r.multiband.seam_score.value, r.uniform.seam_score.value) <=
               cfg.seam_ratio_max,
       "seam not measurable (no positive multiband evidence)",
       "multiband seam_score regression vs uniform at the support boundary");

  if (mb_fail.empty()) {
    r.selected = SelectedCandidate::kDrizzleMultiband;
    r.reason = "multiband promoted: all 15.3.4 gates pass";
  } else {
    r.selected = SelectedCandidate::kDrizzleRaw;
    r.reason = "multiband not promoted -> raw: " + mb_fail;
  }
  return r;
}

std::string multiband_validation_config_hash(const MultibandValidationConfig &cfg) {
  const nlohmann::json j = {
      {"version", kMultibandValidationVersion},
      {"bootstrap_resamples", kMultibandValidationBootstrapResamples},
      {"bootstrap_seed", kMultibandValidationBootstrapSeed},
      {"star_patch_radius", kMultibandValidationStarPatchRadius},
      {"star_patch_min_finite_fraction",
       kMultibandValidationStarPatchMinFiniteFraction},
      {"seam_min_boundary_pixels", kMultibandValidationSeamMinBoundaryPixels},
      {"seam_interior_stride_target",
       kMultibandValidationSeamInteriorStrideTarget},
      {"fwhm_ratio_max", cfg.fwhm_ratio_max},
      {"p90_fwhm_ratio_max", cfg.p90_fwhm_ratio_max},
      {"tail_ratio_max", cfg.tail_ratio_max},
      {"elongation_ratio_max", cfg.elongation_ratio_max},
      {"background_rms_ratio_max", cfg.background_rms_ratio_max},
      {"seam_ratio_max", cfg.seam_ratio_max},
      {"min_stars_fwhm", cfg.min_stars_fwhm},
      {"min_stars_p90_tail_elongation", cfg.min_stars_p90_tail_elongation},
      {"max_fwhm_ci_relative_width", cfg.max_fwhm_ci_relative_width}};
  const std::string text = j.dump();
  return core::sha256_bytes(std::vector<uint8_t>(text.begin(), text.end()));
}

}  // namespace tile_compile::reconstruction
