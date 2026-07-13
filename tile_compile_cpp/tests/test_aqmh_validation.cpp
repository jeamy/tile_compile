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
    REQUIRE(result.weights[i] > cfg.g_floor);
    REQUIRE(result.weights[i] < 1.0f);
  }
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

// §9.15 — Storage fidelity: full-resolution cache reproduces Q_map exactly
TEST_CASE("aqmh_validation_15_storage_fidelity") {
  const int W = 16, H = 16;
  auto frame = make_gradient_frame(W, H);
  auto canvas = full_mask(W, H);
  auto fmask = full_mask(W, H);
  tc::config::AqmhPyramidConfig pyramid;
  tc::config::AqmhStorageConfig storage;
  storage.resolution_divisor = 1;
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
