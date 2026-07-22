#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/aqmh_eps.hpp"
#include "tile_compile/metrics/aqmh_quality_map.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/metrics/aqmh_frame_valid_mask.hpp"
#include "tile_compile/metrics/aqmh_global_quality.hpp"
#include "tile_compile/metrics/aqmh_regions.hpp"
#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"
#include "tile_compile/reconstruction/aqmh_sigma_clip.hpp"
#include "tile_compile/reconstruction/aqmh_cherry_pick.hpp"
#include "tile_compile/reconstruction/aqmh_validation.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

namespace tc = tile_compile;
namespace metrics = tile_compile::metrics;
namespace recon = tile_compile::reconstruction;

namespace {

std::filesystem::path unique_validation_dir(const std::string &name) {
  static int counter = 0;
  return std::filesystem::temp_directory_path() /
         ("tile_compile_val_" + name + "_" + std::to_string(++counter));
}

tc::Matrix2Df make_uniform_frame(int w, int h, float value) {
  return tc::Matrix2Df::Constant(h, w, value);
}

tc::Matrix2Df make_gradient_frame(int w, int h) {
  tc::Matrix2Df f(h, w);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      f(y, x) = static_cast<float>(x + y);
  return f;
}

tc::Matrix2Df make_validation_star_field(int w, int h, int extra_stars = 0) {
  tc::Matrix2Df image(h, w);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      image(y, x) = 100.0f + 0.01f * static_cast<float>((x + 3 * y) % 17);

  int star_index = 0;
  for (int y = 24; y < h - 24; y += 20) {
    for (int x = 24; x < w - 24; x += 20) {
      const float peak = 20.0f + static_cast<float>(star_index % 9);
      for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
          const float r2 = static_cast<float>(dx * dx + dy * dy);
          image(y + dy, x + dx) += peak * std::exp(-0.5f * r2 / 1.4f);
        }
      }
      ++star_index;
    }
  }
  for (int i = 0; i < extra_stars; ++i) {
    const int x = 30 + 17 * i;
    const int y = h - 30;
    image(y, x) += 60.0f + static_cast<float>(i);
  }
  return image;
}

std::vector<uint8_t> full_mask(int w, int h) {
  return std::vector<uint8_t>(static_cast<size_t>(w * h), 1u);
}

std::vector<uint8_t> empty_frame_mask(int w, int h) {
  return std::vector<uint8_t>(static_cast<size_t>(w * h), 0u);
}

} // namespace

TEST_CASE("aqmh_weighted_mad_quickselect_is_deterministic") {
  std::vector<recon::AqmhWeightedSample> samples;
  for (size_t i = 0; i < 101; ++i) {
    const float value = i == 100 ? 1000.0f : static_cast<float>(i % 11);
    samples.push_back({value, 1.0f + static_cast<float>(i % 5), 1.0f, i});
  }
  const auto a = recon::aqmh_sigma_clip(samples, 3.0f, 3, 0.5f, 1.0f);
  const auto b = recon::aqmh_sigma_clip(samples, 3.0f, 3, 0.5f, 1.0f);
  REQUIRE(a.denominator_ok);
  REQUIRE(a.weight_sum == b.weight_sum);
  REQUIRE(a.effective_n == b.effective_n);
  REQUIRE(a.retained.size() == b.retained.size());
  for (size_t i = 0; i < a.retained.size(); ++i)
    REQUIRE(a.retained[i].frame_index == b.retained[i].frame_index);
}

TEST_CASE("aqmh_symmetric_clipping_preserves_background_location") {
  std::vector<recon::AqmhWeightedSample> samples;
  samples.reserve(401);
  for (size_t i = 0; i < 200; ++i) {
    const float deviation = 0.01f * static_cast<float>(i + 1);
    samples.push_back({100.0f - deviation, 1.0f, 1.0f, 2 * i});
    samples.push_back({100.0f + deviation, 1.0f, 1.0f, 2 * i + 1});
  }
  samples.push_back({100.0f, 1.0f, 1.0f, 400});

  const auto result =
      recon::aqmh_sigma_clip(samples, 2.0f, 2.0f, 4, 0.4f, 2.0f);

  REQUIRE(result.denominator_ok);
  REQUIRE(result.retained.size() > 300);
  double weighted_sum = 0.0;
  for (const auto &sample : result.retained)
    weighted_sum += sample.weight * sample.value;
  REQUIRE(weighted_sum / result.weight_sum == Catch::Approx(100.0).margin(1e-5));
}

// §9.1 — Map range: Q_map ∈ [0,1] for all finite source-valid pixels
TEST_CASE("aqmh_validation_01_map_range") {
  const int W = 32, H = 32;
  auto frame = make_gradient_frame(W, H);
  auto mask = full_mask(W, H);
  auto fmask = full_mask(W, H);
  tc::config::AqmhPyramidConfig cfg;
  auto result = metrics::compute_aqmh_quality_map(frame, mask, fmask, W, H, cfg);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const float v = result.q_map(y, x);
      if (std::isfinite(v)) {
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
      }
    }
}

// §9.2 — Output guard: Q_map = 0 where C=0 or M_f=0
TEST_CASE("aqmh_validation_02_output_guard") {
  const int W = 16, H = 16;
  auto frame = make_uniform_frame(W, H, 100.0f);
  auto canvas = full_mask(W, H);
  auto fmask = full_mask(W, H);
  // Invalidate some pixels via M_f
  for (int i = 0; i < H * W / 2; ++i) fmask[i] = 0u;
  tc::config::AqmhPyramidConfig cfg;
  auto result = metrics::compute_aqmh_quality_map(frame, canvas, fmask, W, H, cfg);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const size_t idx = static_cast<size_t>(y * W + x);
      if (fmask[idx] == 0u) {
        REQUIRE(result.q_map(y, x) == 0.0f);
      }
    }
}

// §9.3 — Determinism: identical inputs → identical maps
TEST_CASE("aqmh_validation_03_determinism") {
  const int W = 16, H = 16;
  auto frame = make_gradient_frame(W, H);
  auto mask = full_mask(W, H);
  auto fmask = full_mask(W, H);
  tc::config::AqmhPyramidConfig cfg;
  auto r1 = metrics::compute_aqmh_quality_map(frame, mask, fmask, W, H, cfg);
  auto r2 = metrics::compute_aqmh_quality_map(frame, mask, fmask, W, H, cfg);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      REQUIRE(r1.q_map(y, x) == r2.q_map(y, x));
}

// §9.4 — Unsupported coverage: empty V_c → zero output, no NaN/Inf
TEST_CASE("aqmh_validation_04_unsupported_coverage") {
  const int W = 8, H = 8;
  auto frame = make_uniform_frame(W, H, 1.0f);
  auto canvas = full_mask(W, H);
  auto fmask = empty_frame_mask(W, H); // no valid pixels
  tc::config::AqmhPyramidConfig cfg;
  auto result = metrics::compute_aqmh_quality_map(frame, canvas, fmask, W, H, cfg);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const float v = result.q_map(y, x);
      REQUIRE_FALSE(std::isnan(v));
      REQUIRE_FALSE(std::isinf(v));
    }
}

// §9.5 — Explicit zero-veto: finite maps, all weights zero → unsupported, no unweighted mean
TEST_CASE("aqmh_validation_05_zero_veto") {
  const int W = 4, H = 4;
  auto frame = make_uniform_frame(W, H, 10.0f);
  auto canvas = full_mask(W, H);
  auto fmask = full_mask(W, H);
  tc::config::AqmhPyramidConfig pyramid;
  auto dir = unique_validation_dir("zeroveto");
  std::string mask_hash = metrics::compute_aqmh_canvas_mask_hash(canvas, W, H);
  metrics::QualityMapCache cache(dir, "luma", W, H, pyramid,
                                  tc::config::AqmhStorageConfig{}, mask_hash, "cpu");
  auto qm = metrics::compute_aqmh_quality_map(frame, canvas, fmask, W, H, pyramid);
  cache.write(0, qm.q_map, fmask);
  // All-zero global weights
  tc::VectorXf gw = tc::VectorXf::Zero(1);
  recon::AqmhReconstructionConfig rcfg;
  rcfg.min_n_eff = 1.0f;
  auto loader = [&](size_t, tc::Matrix2Df &out) -> bool {
    out = frame; return true;
  };
  auto mask_loader = [&](size_t, std::vector<uint8_t> &out) -> bool {
    out = fmask; return true;
  };
  auto result = recon::reconstruct_aqmh_weighted(1, loader, &cache, gw,
                                                  canvas, W, H, rcfg, mask_loader);
  REQUIRE(result.zero_veto_pixels > 0);
  // Output must not be an unweighted mean (i.e., must be zero)
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      REQUIRE(result.output(y, x) == 0.0f);
}

// §9.7 — No structural injection: regression within configured limits
TEST_CASE("aqmh_validation_07_no_structural_injection") {
  const int W = 32, H = 32;
  auto frame = make_uniform_frame(W, H, 50.0f);
  auto canvas = full_mask(W, H);
  auto fmask = full_mask(W, H);
  tc::config::AqmhPyramidConfig pyramid;
  auto dir = unique_validation_dir("struct");
  std::string mask_hash = metrics::compute_aqmh_canvas_mask_hash(canvas, W, H);
  metrics::QualityMapCache cache(dir, "luma", W, H, pyramid,
                                  tc::config::AqmhStorageConfig{}, mask_hash, "cpu");
  auto qm = metrics::compute_aqmh_quality_map(frame, canvas, fmask, W, H, pyramid);
  cache.write(0, qm.q_map, fmask);
  tc::VectorXf gw(1); gw << 1.0f;
  recon::AqmhReconstructionConfig rcfg;
  rcfg.min_n_eff = 1.0f;
  auto loader = [&](size_t, tc::Matrix2Df &out) -> bool {
    out = frame; return true;
  };
  auto mask_loader = [&](size_t, std::vector<uint8_t> &out) -> bool {
    out = fmask; return true;
  };
  auto aqmh_result = recon::reconstruct_aqmh_weighted(1, loader, &cache, gw,
                                                       canvas, W, H, rcfg, mask_loader);
  auto control_cfg = rcfg;
  control_cfg.uniform_weights = true;
  auto control_result = recon::reconstruct_aqmh_weighted(1, loader, &cache, gw,
                                                          canvas, W, H, control_cfg, mask_loader);
  auto cmp = recon::compare_aqmh_to_uniform_control(aqmh_result.output, control_result.output);
  tc::config::AqmhValidationConfig vcfg;
  REQUIRE(cmp.seam_score_regression <= vcfg.max_seam_score_regression);
  REQUIRE(cmp.fwhm_regression <= vcfg.max_fwhm_regression);
  REQUIRE(cmp.background_rms_regression <= vcfg.max_background_rms_regression);
}

// §9.10 — Cherry-pick flag: cherry_pick_active reflected in diagnostics
TEST_CASE("aqmh_validation_10_cherry_pick_flag") {
  const int W = 8, H = 8;
  auto frame = make_uniform_frame(W, H, 100.0f);
  auto canvas = full_mask(W, H);
  auto fmask = full_mask(W, H);
  tc::config::AqmhPyramidConfig pyramid;
  auto dir = unique_validation_dir("cherry_flag");
  std::string mask_hash = metrics::compute_aqmh_canvas_mask_hash(canvas, W, H);
  metrics::QualityMapCache cache(dir, "luma", W, H, pyramid,
                                  tc::config::AqmhStorageConfig{}, mask_hash, "cpu");
  auto qm = metrics::compute_aqmh_quality_map(frame, canvas, fmask, W, H, pyramid);
  // Need enough frames for cherry-pick to be meaningful
  const size_t N = 30;
  for (size_t i = 0; i < N; ++i) cache.write(i, qm.q_map, fmask);
  tc::VectorXf gw = tc::VectorXf::Ones(static_cast<int>(N));
  recon::AqmhReconstructionConfig rcfg;
  rcfg.cherry_pick = true;
  rcfg.cherry_pick_k_frac = 0.8f;
  rcfg.cherry_pick_k_min_required = 20;
  rcfg.min_n_eff = 1.0f;
  auto loader = [&](size_t, tc::Matrix2Df &out) -> bool {
    out = frame; return true;
  };
  auto mask_loader = [&](size_t, std::vector<uint8_t> &out) -> bool {
    out = fmask; return true;
  };
  auto result = recon::reconstruct_aqmh_weighted(N, loader, &cache, gw,
                                                  canvas, W, H, rcfg, mask_loader);
  // With k_frac=0.8 and N=30, K_nominal=24 >= k_min_required=20,
  // so cherry-pick should not be force-disabled.
  REQUIRE(result.cherry_pick_per_pixel_mode == true);
}

// §9.11 — Cherry-pick selection-size floor: K_nominal_median < k_min_required → forced off
TEST_CASE("aqmh_validation_11_cherry_pick_forced_disabled") {
  const int W = 8, H = 8;
  auto frame = make_uniform_frame(W, H, 100.0f);
  auto canvas = full_mask(W, H);
  auto fmask = full_mask(W, H);
  tc::config::AqmhPyramidConfig pyramid;
  auto dir = unique_validation_dir("cherry_force");
  std::string mask_hash = metrics::compute_aqmh_canvas_mask_hash(canvas, W, H);
  metrics::QualityMapCache cache(dir, "luma", W, H, pyramid,
                                  tc::config::AqmhStorageConfig{}, mask_hash, "cpu");
  auto qm = metrics::compute_aqmh_quality_map(frame, canvas, fmask, W, H, pyramid);
  const size_t N = 10; // Too few for k_min_required=20
  for (size_t i = 0; i < N; ++i) cache.write(i, qm.q_map, fmask);
  tc::VectorXf gw = tc::VectorXf::Ones(static_cast<int>(N));
  recon::AqmhReconstructionConfig rcfg;
  rcfg.cherry_pick = true;
  rcfg.cherry_pick_k_frac = 0.3f;
  rcfg.cherry_pick_k_min_required = 20;
  rcfg.min_n_eff = 1.0f;
  auto loader = [&](size_t, tc::Matrix2Df &out) -> bool {
    out = frame; return true;
  };
  auto mask_loader = [&](size_t, std::vector<uint8_t> &out) -> bool {
    out = fmask; return true;
  };
  auto result = recon::reconstruct_aqmh_weighted(N, loader, &cache, gw,
                                                  canvas, W, H, rcfg, mask_loader);
  REQUIRE(result.cherry_pick_forced_disabled == true);
}

// §9.14 — Global quality: g_floor < G < 1, finite
TEST_CASE("aqmh_validation_14_global_quality_bounded") {
  std::vector<float> sharp = {0.5f, 0.6f, 0.4f, 0.7f, 0.55f};
  std::vector<float> snr = {10.0f, 12.0f, 8.0f, 15.0f, 11.0f};
  std::vector<float> background = {0.10f, 0.20f, 0.05f, 0.40f, 0.15f};
  tc::config::AqmhGlobalQualityConfig cfg;
  auto result = metrics::compute_aqmh_global_quality(sharp, snr, background, cfg);
  for (size_t i = 0; i < result.weights.size(); ++i) {
    REQUIRE(std::isfinite(result.weights[i]));
    REQUIRE(result.weights[i] >= cfg.g_floor);
    REQUIRE(result.weights[i] <= 1.0f);
  }
}

TEST_CASE("aqmh_validation_comparison_uses_common_control_stars") {
  const auto control = make_validation_star_field(160, 160);
  const auto candidate = make_validation_star_field(160, 160, 4);

  const auto cmp = recon::compare_aqmh_to_uniform_control(candidate, control);

  REQUIRE(cmp.control.star_count >= 12);
  REQUIRE(cmp.aqmh.star_count == cmp.control.star_count);
}

TEST_CASE("aqmh_validation_comparison_handles_mismatched_dimensions") {
  const auto control = make_validation_star_field(160, 160);
  const auto smaller_candidate = make_validation_star_field(96, 96);

  REQUIRE_NOTHROW(
      recon::compare_aqmh_to_uniform_control(smaller_candidate, control));
}

TEST_CASE("aqmh_background_rms_ignores_diffuse_astronomical_structure") {
  constexpr int W = 256;
  constexpr int H = 192;
  tc::Matrix2Df control(H, W);
  tc::Matrix2Df candidate(H, W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const float noise =
          0.2f * std::sin(0.71f * static_cast<float>(x) +
                          1.13f * static_cast<float>(y));
      const float dx = (static_cast<float>(x) - 0.55f * W) / (0.28f * W);
      const float dy = (static_cast<float>(y) - 0.45f * H) / (0.32f * H);
      const float diffuse_signal =
          20.0f * std::exp(-0.5f * (dx * dx + dy * dy));
      control(y, x) = 100.0f + noise;
      candidate(y, x) = control(y, x) + diffuse_signal;
    }
  }

  const auto cmp = recon::compare_aqmh_to_uniform_control(candidate, control);

  REQUIRE(cmp.background_rms_applicable);
  REQUIRE(std::abs(cmp.background_rms_regression) < 0.05f);
}

TEST_CASE("aqmh_validation_mask_excludes_partial_coverage_edge_noise") {
  constexpr int W = 192;
  constexpr int H = 160;
  tc::Matrix2Df control(H, W);
  tc::Matrix2Df candidate(H, W);
  std::vector<uint8_t> common_mask(static_cast<size_t>(W) * H, 0u);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const float base_noise =
          0.2f * std::sin(0.83f * static_cast<float>(x) +
                          1.07f * static_cast<float>(y));
      control(y, x) = 100.0f + base_noise;
      candidate(y, x) = control(y, x);
      if (x < W / 2) {
        common_mask[static_cast<size_t>(y) * W + x] = 1u;
      } else {
        candidate(y, x) +=
            0.3f * std::sin(1.91f * static_cast<float>(x) -
                            2.17f * static_cast<float>(y));
      }
    }
  }

  const auto unmasked = recon::compare_aqmh_to_uniform_control(candidate, control);
  const auto masked =
      recon::compare_aqmh_to_uniform_control(candidate, control, common_mask);

  REQUIRE(unmasked.background_rms_applicable);
  REQUIRE(unmasked.background_rms_regression > 0.05f);
  REQUIRE(masked.background_rms_applicable);
  REQUIRE(std::abs(masked.background_rms_regression) < 0.01f);
}

TEST_CASE("aqmh_background_rms_detects_added_pixel_scale_noise") {
  constexpr int W = 192;
  constexpr int H = 160;
  tc::Matrix2Df control(H, W);
  tc::Matrix2Df candidate(H, W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const float base_noise =
          0.2f * std::sin(0.83f * static_cast<float>(x) +
                          1.07f * static_cast<float>(y));
      const float added_noise =
          0.3f * std::sin(1.91f * static_cast<float>(x) -
                          2.17f * static_cast<float>(y));
      control(y, x) = 100.0f + base_noise;
      candidate(y, x) = control(y, x) + added_noise;
    }
  }

  const auto cmp = recon::compare_aqmh_to_uniform_control(candidate, control);

  REQUIRE(cmp.background_rms_applicable);
  REQUIRE(cmp.background_rms_regression > 0.05f);
}

// §9.14 — Global quality: identical inputs → identical G
TEST_CASE("aqmh_validation_14_global_quality_determinism") {
  std::vector<float> sharp = {0.5f, 0.6f, 0.4f};
  std::vector<float> snr = {10.0f, 12.0f, 8.0f};
  std::vector<float> background = {0.10f, 0.20f, 0.05f};
  tc::config::AqmhGlobalQualityConfig cfg;
  auto r1 = metrics::compute_aqmh_global_quality(sharp, snr, background, cfg);
  auto r2 = metrics::compute_aqmh_global_quality(sharp, snr, background, cfg);
  for (size_t i = 0; i < r1.weights.size(); ++i)
    REQUIRE(r1.weights[i] == r2.weights[i]);
}

TEST_CASE("aqmh_validation_degenerate_control_metrics_are_not_applicable") {
  const int W = 32, H = 32;
  const auto aqmh = make_gradient_frame(W, H);
  const auto control = make_uniform_frame(W, H, 10.0f);

  const auto cmp = recon::compare_aqmh_to_uniform_control(aqmh, control);

  REQUIRE_FALSE(cmp.background_rms_applicable);
  REQUIRE_FALSE(cmp.seam_applicable);
  REQUIRE_FALSE(cmp.tail_applicable);
  REQUIRE_FALSE(cmp.elongation_applicable);
  REQUIRE(cmp.background_rms_regression == Catch::Approx(0.0f));
  REQUIRE(cmp.seam_score_regression == Catch::Approx(0.0f));
  REQUIRE(cmp.tail11_abs_regression == Catch::Approx(0.0f));
  REQUIRE(cmp.elongation_regression == Catch::Approx(0.0f));
}

TEST_CASE("aqmh_baseline_defaults_match_object_agnostic_analysis") {
  tc::config::AqmhStorageConfig storage;
  REQUIRE(storage.resolution_divisor == 2);
  REQUIRE(storage.dtype == "uint16");

  tc::config::AqmhGlobalQualityConfig global;
  REQUIRE(global.g_floor == Catch::Approx(0.03f));
  REQUIRE(global.g_w_sharp == Catch::Approx(0.55f));
  REQUIRE(global.g_w_snr == Catch::Approx(0.30f));
  REQUIRE(global.g_w_background_penalty == Catch::Approx(0.25f));
  REQUIRE(global.g_k_scale == Catch::Approx(1.5f));

  tc::config::AqmhReconstructionConfig reconstruction;
  REQUIRE(reconstruction.clip_sigma == Catch::Approx(2.0f));
  REQUIRE(reconstruction.clip_sigma_low == Catch::Approx(2.0f));
  REQUIRE(reconstruction.clip_sigma_high == Catch::Approx(2.0f));
  REQUIRE(reconstruction.clip_iterations == 4);
  REQUIRE(reconstruction.min_fraction == Catch::Approx(0.40f));
  REQUIRE(reconstruction.registration_weight_floor == Catch::Approx(0.30f));
  REQUIRE(reconstruction.registration_sequential_factor == Catch::Approx(0.92f));
  REQUIRE(reconstruction.registration_predicted_factor == Catch::Approx(0.50f));
  REQUIRE(reconstruction.structure_mask_low_q == Catch::Approx(0.40f));
  REQUIRE(reconstruction.structure_mask_high_q == Catch::Approx(0.90f));
  REQUIRE(reconstruction.structure_mask_blur_sigma_px == Catch::Approx(4.0f));

  tc::config::AqmhValidationConfig validation;
  REQUIRE(validation.max_fwhm_regression == Catch::Approx(0.02f));
  REQUIRE(validation.max_background_rms_regression == Catch::Approx(0.05f));
  REQUIRE(validation.max_seam_score_regression == Catch::Approx(0.05f));
  REQUIRE(validation.max_tail11_abs_regression == Catch::Approx(0.10f));
  REQUIRE(validation.max_elongation_regression == Catch::Approx(0.08f));
}

TEST_CASE("aqmh_schema_exposes_current_baseline_parameters") {
  const auto schema = nlohmann::json::parse(tc::config::get_schema_json());
  const auto &aqmh = schema.at("properties").at("aqmh").at("properties");
  REQUIRE(aqmh.at("storage").at("properties").at("resolution_divisor").at("default") == 2);
  REQUIRE(aqmh.at("storage").at("properties").at("dtype").at("default") == "uint16");
  REQUIRE(aqmh.at("global_quality").at("properties").at("g_k_scale").at("default") == 1.5);
  const auto &reconstruction = aqmh.at("reconstruction").at("properties");
  REQUIRE(reconstruction.at("clip_sigma").at("default") == 2.0);
  REQUIRE(reconstruction.at("clip_sigma_low").at("default") == 2.0);
  REQUIRE(reconstruction.at("clip_sigma_high").at("default") == 2.0);
  REQUIRE(reconstruction.at("clip_iterations").at("default") == 4);
  REQUIRE(reconstruction.at("min_fraction").at("default") == 0.4);
  REQUIRE(reconstruction.at("registration_sequential_factor").at("default") == 0.92);
  REQUIRE(reconstruction.at("registration_predicted_factor").at("default") == 0.50);
  REQUIRE(reconstruction.contains("structure_mask_low_q"));
  REQUIRE(reconstruction.contains("structure_mask_high_q"));
  REQUIRE(reconstruction.contains("structure_mask_blur_sigma_px"));
  const auto &validation = aqmh.at("validation").at("properties");
  REQUIRE(validation.at("max_tail11_abs_regression").at("default") == 0.10);
  REQUIRE(validation.at("max_elongation_regression").at("default") == 0.08);
}

// §9.15 — Storage fidelity: full-resolution cache reproduces Q_map exactly
TEST_CASE("aqmh_validation_15_storage_fidelity") {
  const int W = 16, H = 16;
  auto frame = make_gradient_frame(W, H);
  auto canvas = full_mask(W, H);
  auto fmask = full_mask(W, H);
  tc::config::AqmhPyramidConfig pyramid;
  tc::config::AqmhStorageConfig storage;
  storage.resolution_divisor = 1;
  storage.dtype = "float32";
  auto dir = unique_validation_dir("fidelity");
  std::string mask_hash = metrics::compute_aqmh_canvas_mask_hash(canvas, W, H);
  metrics::QualityMapCache cache(dir, "luma", W, H, pyramid, storage, mask_hash, "cpu");
  auto qm = metrics::compute_aqmh_quality_map(frame, canvas, fmask, W, H, pyramid);
  cache.write(0, qm.q_map, fmask);
  auto read_back = cache.read(0);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const float a = qm.q_map(y, x);
      const float b = read_back(y, x);
      if (std::isfinite(a) && std::isfinite(b))
        REQUIRE(a == Catch::Approx(b).margin(1e-6f));
    }
}

// §9.16 — Channel diagnostics: every (f,c) has distinct diagnostics
TEST_CASE("aqmh_validation_16_channel_diagnostics") {
  const int W = 16, H = 16;
  auto frame1 = make_gradient_frame(W, H);
  auto frame2 = make_uniform_frame(W, H, 200.0f);
  auto canvas = full_mask(W, H);
  auto fmask = full_mask(W, H);
  tc::config::AqmhPyramidConfig cfg;
  auto r1 = metrics::compute_aqmh_quality_map(frame1, canvas, fmask, W, H, cfg);
  auto r2 = metrics::compute_aqmh_quality_map(frame2, canvas, fmask, W, H, cfg);
  // Different frames should produce different diagnostics
  REQUIRE(r1.diagnostics.sharpness_p50 != r2.diagnostics.sharpness_p50);
}

#else
// Catch2 not available — skip validation tests
#endif
