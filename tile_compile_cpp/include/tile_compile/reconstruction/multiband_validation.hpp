#pragma once

// Three-way candidate selection for the CFA-forward-drizzle single method ---
// milestone M6, plan section 15. The three immutable stages are
//   drizzle_uniform  --- the safe control
//   drizzle_raw      --- B*G_eff*Q_composite, never post-processed (15.1)
//   drizzle_multiband --- the plan-14 fusion
// measured at ONE fixed star population detected on drizzle_uniform (15.2):
// candidate_vs_raw does NOT re-detect stars on raw.
//
// Selection (15.3):
//  * Raw is chosen over Uniform only if a support / numeric / background-RMS /
//    seam gate is not applicable or is violated. A non-applicable STAR metric
//    alone never rejects Raw.
//  * Multiband is chosen only if EVERY 15.3.4 inequality holds AND support +
//    numerics are fully valid. A non-applicable star metric never counts as
//    positive multiband evidence --- then Raw stays.
//  * Multiband fails  -> Raw.  Raw fails an applicable / mandatory-N/A safety
//    gate -> Uniform.
//
// This module owns only the SELECTION contract and the per-star FWHM
// statistic (median / p90 / bootstrap CI) that plan 15 adds. Field- and
// tail-metric measurement at the fixed star set is reused from the existing
// general validation helpers (measurement only --- no legacy decision code).
// This module does not itself write `selected_candidate` (16.3): the runner's
// MULTIBAND phase calls select_reconstruction_candidate() and serialises the
// result into artifacts/forward_drizzle.json.

#include "tile_compile/core/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tile_compile::reconstruction {

// Versioned constants (NOT config): if the selection outcome ever feeds
// `selected_candidate` these belong in the multiband hash domain.
inline constexpr int kMultibandValidationVersion = 1;
inline constexpr int kMultibandValidationBootstrapResamples = 2000;
inline constexpr uint64_t kMultibandValidationBootstrapSeed =
    0x9E3779B97F4A7C15ull;
// Half-width (px) of the square patch measured around each star for FWHM.
inline constexpr int kMultibandValidationStarPatchRadius = 7;
// A star patch is usable when its centre is finite AND at least this fraction
// of its pixels are finite; the sparse remaining holes are median-filled.
// Real OSC working luminance has scattered off-support NaN, so "whole patch
// finite" rejects every star (M31 2026-09-06: 0 of 250, incl. the control).
inline constexpr double kMultibandValidationStarPatchMinFiniteFraction = 0.75;
// boundary_seam_score (plan 15.3, exact form awaits plan confirmation --- 30.42):
// minimum support-boundary edge pixel count below which the seam metric is
// reported non-applicable, and the interior-reference stride target (the
// interior sample is decimated to about this many pixels for speed).
inline constexpr int kMultibandValidationSeamMinBoundaryPixels = 8;
inline constexpr int kMultibandValidationSeamInteriorStrideTarget = 40000;

struct MultibandValidationConfig {
  // 15.3.4 promotion ratios.
  double fwhm_ratio_max = 0.95;         // median_FWHM_mb  <= 0.95 * raw
  double p90_fwhm_ratio_max = 1.00;     // p90_FWHM_mb     <= 1.00 * raw
  double tail_ratio_max = 1.10;         // tail_mb         <= 1.10 * raw
  double elongation_ratio_max = 1.08;   // elong_mb        <= 1.08 * raw
  double background_rms_ratio_max = 1.05;  // vs UNIFORM
  double seam_ratio_max = 1.05;            // vs UNIFORM
  // 15.3.5 star-count applicability.
  int min_stars_fwhm = 20;
  int min_stars_p90_tail_elongation = 30;
  // 15.3.5 bootstrap 95% CI relative width of the FWHM median.
  double max_fwhm_ci_relative_width = 0.10;
};

// One metric on one candidate: value + explicit applicability (plan 15.3.6).
struct ValidationMetric {
  double value = 0.0;
  bool applicable = false;
  int sample_count = 0;
  double ci_low = 0.0;   // populated for bootstrapped metrics only
  double ci_high = 0.0;
  std::string reason_if_not_applicable;
};

struct CandidateMetrics {
  ValidationMetric median_fwhm;   // bootstrapped; >= min_stars_fwhm + CI width
  ValidationMetric p90_fwhm;      // >= min_stars_p90_tail_elongation
  ValidationMetric tail;          // weighted p-metric, >= 30
  ValidationMetric elongation;    // >= 30
  ValidationMetric background_rms;
  ValidationMetric seam_score;
  bool support_ok = false;        // finite coverage over the star patches
  bool numerics_ok = false;       // no NaN/Inf in the measured field
};

struct ValidationStar {
  int x = 0;
  int y = 0;
  float peak = 0.0f;
  // False when the fused alpha_final is identically 0 across every band in a
  // small neighbourhood of this star: multiband there is measured identical to
  // raw, so the star carries no multiband evidence (plan 30.40 follow-up).
  bool multiband_effective = true;
};

enum class SelectedCandidate { kDrizzleUniform, kDrizzleRaw, kDrizzleMultiband };

struct MultibandValidationResult {
  SelectedCandidate selected = SelectedCandidate::kDrizzleUniform;
  std::string reason;
  CandidateMetrics uniform, raw, multiband;
  int stars_total = 0;
  int stars_multiband_effective = 0;
  // The star set actually used for the multiband star metrics (effective subset).
  int multiband_star_sample_count = 0;
};

// SHA-256 over the versioned selection constants above and the effective
// `MultibandValidationConfig` --- emitted into artifacts/forward_drizzle.json
// as `validation_config_hash` so a `selected_candidate` is reproducible /
// auditable WITHOUT perturbing the drizzle-profile-store identity hash (the
// store bytes do not depend on any of these values). This supersedes the
// 30.42/30.43 note about folding them into `multiband_config_hash`.
std::string multiband_validation_config_hash(
    const MultibandValidationConfig &cfg = {});

// Detect the fixed validation star population on the uniform control. When
// `alpha_final_by_band` is supplied (one map per fused band, row-major
// width*height, empty inner vector = band inactive), each star's
// `multiband_effective` is set from whether ANY band's |alpha| exceeds
// `alpha_effective_eps` within +/-1 px of the star centre.
std::vector<ValidationStar> prepare_validation_samples(
    const Matrix2Df &uniform_control, int width, int height,
    const std::vector<uint8_t> &validation_mask = {},
    const std::vector<std::vector<float>> &alpha_final_by_band = {},
    double alpha_effective_eps = 1e-6);

// Full plan-15 three-way selection. `multiband` may equal `raw` numerically
// (alpha 0 everywhere) --- then the FWHM ratio gate fails at equality and Raw
// is selected. All three images share geometry `width x height`.
MultibandValidationResult select_reconstruction_candidate(
    const Matrix2Df &drizzle_uniform, const Matrix2Df &drizzle_raw,
    const Matrix2Df &drizzle_multiband, int width, int height,
    const std::vector<ValidationStar> &stars,
    const MultibandValidationConfig &cfg = {},
    const std::vector<uint8_t> &validation_mask = {});

// Exposed for tests: per-star FWHM values (px) at the given star centres,
// NaN-free, only stars with a successful patch fit retained.
std::vector<double> per_star_fwhm(const Matrix2Df &image, int width, int height,
                                  const std::vector<ValidationStar> &stars,
                                  bool effective_only);

// Exposed for tests: deterministic bootstrap 95% CI of the median of `values`.
struct MedianCi {
  double median = 0.0;
  double ci_low = 0.0;
  double ci_high = 0.0;
  double relative_width = 0.0;  // (ci_high - ci_low) / median
};
MedianCi bootstrap_median_ci(const std::vector<double> &values);

}  // namespace tile_compile::reconstruction
