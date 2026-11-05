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

// anchor_reference_{R,G,B}: optional pre-denoise snapshot of the same image, same dimensions, same
// physical units, only used (and only read, never mutated) when cfg.anchor_from_reference is true.
// Pass all three or none; a partial set or a dimension mismatch is treated as "none" (falls back to
// R/G/B), never an error.
HyperMetricStretchDiagnostics run_hypermetric_stretch_rgb(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
    const HyperMetricStretchConfig &cfg,
    const std::vector<uint8_t> *statistics_mask = nullptr, int mask_rows = 0,
    int mask_cols = 0,
    const std::vector<uint8_t> *output_mask = nullptr,
    const Matrix2Df *anchor_reference_R = nullptr,
    const Matrix2Df *anchor_reference_G = nullptr,
    const Matrix2Df *anchor_reference_B = nullptr);

} // namespace tile_compile::image
