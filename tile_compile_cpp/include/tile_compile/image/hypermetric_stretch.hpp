#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <array>
#include <string>
#include <vector>

namespace tile_compile::image {

// The stretch parameters are the public config struct itself (single source of
// truth for defaults, parsing and schema); see config::HyperMetricStretchConfig
// for the per-field documentation.
using HyperMetricStretchConfig = config::HyperMetricStretchConfig;

// The parameters HMS derives from the image it stretches. Every run returns
// them in its diagnostics; passed back in (run_hypermetric_stretch_rgb's
// `frozen`), they are used instead of being derived from the image again, so
// a second image gets exactly the transfer curve of the first. The runner uses
// this to stretch a chroma-denoised image with the curve of its pre-denoise
// state (hypermetric_stretch.params_from_pre_denoise).
struct HyperMetricDerivedParams {
  bool valid = false;
  float input_scale = 1.0f;  // input normalization (legacy 16-bit range etc.)
  float anchor = 0.0f;
  float log_d = 2.0f;
  float star_pressure = 0.0f;
  // ready_to_use output scaling; only meaningful when output_scaling is set.
  bool output_scaling = false;
  float floor_r = 0.0f;
  float floor_g = 0.0f;
  float floor_b = 0.0f;
  float scale = 1.0f;
  bool mtf_applied = false;
  float mtf_midtone = 0.5f;
};

struct HyperMetricStretchDiagnostics {
  bool success = false;
  std::string status;
  std::string profile = "rec709";
  std::string profile_source = "fallback";
  float weights_r = 0.2126f;
  float weights_g = 0.7152f;
  float weights_b = 0.0722f;
  float anchor = 0.0f;
  float log_d = 2.0f;
  float target_bg = 0.15f;
  float protect_b = 6.0f;
  float convergence_power = 3.5f;
  float star_pressure = 0.0f;
  float color_strategy = 0.0f;
  float color_grip = 1.0f;
  float shadow_convergence = 0.0f;
  float linear_expansion = 0.0f;
  float highlight_ceiling_percentile = 100.0f;
  float black_clip_percent = 0.0f;
  float white_clip_percent = 0.0f;
  // Final output scaling (mode ready_to_use): the shared sky-to-floor gap,
  // what bounded it, the shared expansion scale, what bounded that, and the
  // midtone that pins the background to target_bg.
  float output_floor_margin = 0.0f;
  std::string output_margin_source = "none";
  float output_scale = 0.0f;
  std::string output_scale_source = "none";
  float output_mtf_midtone = 0.0f;
  // "image" when derived from the stretched image, "frozen" when taken from
  // the `frozen` argument.
  std::string params_source = "image";
  HyperMetricDerivedParams derived;
  std::string error_message;
};

std::array<float, 3> hypermetric_profile_weights(
    const std::string &profile, const std::string &fallback_profile,
    std::string *resolved_profile = nullptr,
    std::string *profile_source = nullptr);

float hypermetric_hyperbolic_stretch_value(float value, float D, float b,
                                           float SP = 0.0f);

float hypermetric_solve_log_d(std::vector<float> luma_sample,
                              float target_median, float b);

// frozen: parameters of an earlier run (its diagnostics' `derived`) to use
// instead of deriving them from R/G/B; ignored unless frozen->valid. They must
// come from an image of the same scene and pipeline state with the same cfg.
HyperMetricStretchDiagnostics run_hypermetric_stretch_rgb(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
    const HyperMetricStretchConfig &cfg,
    const std::vector<uint8_t> *statistics_mask = nullptr, int mask_rows = 0,
    int mask_cols = 0,
    const std::vector<uint8_t> *output_mask = nullptr,
    const HyperMetricDerivedParams *frozen = nullptr);

} // namespace tile_compile::image
