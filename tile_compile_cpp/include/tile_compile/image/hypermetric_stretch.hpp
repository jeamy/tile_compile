#pragma once

#include "tile_compile/core/types.hpp"

#include <array>
#include <string>
#include <vector>

namespace tile_compile::image {

struct HyperMetricStretchConfig {
  bool enabled = true;
  bool require_successful_pcc = true;
  std::string mode = "ready_to_use";
  std::string sensor_profile = "rec709";
  std::string fallback_profile = "rec709";
  bool adaptive_anchor = true;
  float target_bg = 0.15f;
  float protect_b = 6.0f;
  float convergence_power = 3.5f;
  std::string log_d_mode = "auto";
  float fixed_log_d = 2.0f;
  std::string color_strategy = "fixed";
  float fixed_color_strategy = 0.0f;
  float color_grip = 1.0f;
  float shadow_convergence = 0.0f;
  float linear_expansion = 0.0f;
  // Percentile (per output channel, sampled the same way as the 99th-
  // percentile contrast ceiling) used as the "must not clip" highlight
  // reference in ready_to_use's adaptive_output_scaling. 100 (default)
  // reproduces the previous behaviour exactly: the true brightest real
  // pixel never clips, which on a target with one very bright, compact
  // highlight (e.g. a nebula's core) caps the contrast scale far below
  // what the rest of the frame could otherwise use. Lowering it (e.g. to
  // 99.9) allows that small, bounded top fraction of pixels to clip in
  // exchange for materially more contrast in stars and highlight regions;
  // it does not change the mid-tone/background level, which stays pinned
  // to target_bg regardless.
  float highlight_ceiling_percentile = 100.0f;
  bool write_channels = false;
  std::string output_rgb = "stacked_rgb_hms.fits";
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

HyperMetricStretchDiagnostics run_hypermetric_stretch_rgb(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
    const HyperMetricStretchConfig &cfg,
    const std::vector<uint8_t> *statistics_mask = nullptr, int mask_rows = 0,
    int mask_cols = 0,
    const std::vector<uint8_t> *output_mask = nullptr);

} // namespace tile_compile::image
