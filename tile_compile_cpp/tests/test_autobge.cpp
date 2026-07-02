#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/core/types.hpp"
#include "tile_compile/image/background_extraction.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <vector>

namespace ti = tile_compile::image;

namespace {

ti::BGEConfig make_autobge_config(int w, int h) {
  ti::BGEConfig cfg{};
  cfg.enabled = true;
  cfg.method = "autobge";
  cfg.autobge.num_sample_points = 0; // auto
  cfg.autobge.poly_degree = 2;
  cfg.autobge.rbf_smooth = 0.1f;
  cfg.autobge.downsample_scale = 2;
  cfg.autobge.patch_size = 5;
  cfg.autobge.patch_estimator = "median";
  cfg.autobge.stretch_mode = "none";
  cfg.autobge.stretch_target_median = 0.25f;
  cfg.autobge.border_margin = 2;
  cfg.autobge.bright_exclusion_fraction = 0.2f;
  cfg.autobge.gradient_descent_max_iters = 5;
  cfg.autobge.mono_mode = "rgb_duplicate";
  cfg.common_mask_rows = h;
  cfg.common_mask_cols = w;
  cfg.common_valid_mask.assign(static_cast<size_t>(w * h), 1);
  return cfg;
}

tile_compile::Matrix2Df make_gradient_image(int w, int h,
                                             float offset,
                                             float gx, float gy) {
  tile_compile::Matrix2Df img(h, w);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      img(y, x) = offset + gx * x + gy * y;
  return img;
}

tile_compile::Matrix2Df add_stars(tile_compile::Matrix2Df img,
                                   int n_stars, int w, int h,
                                   unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dx(2, w - 3);
  std::uniform_int_distribution<int> dy(2, h - 3);
  std::normal_distribution<float> brightness(500.0f, 200.0f);
  for (int i = 0; i < n_stars; ++i) {
    int x = dx(rng), y = dy(rng);
    float b = std::max(100.0f, brightness(rng));
    for (int dy2 = -2; dy2 <= 2; ++dy2)
      for (int dx2 = -2; dx2 <= 2; ++dx2) {
        int xx = x + dx2, yy = y + dy2;
        if (xx >= 0 && xx < w && yy >= 0 && yy < h) {
          float falloff = std::exp(-(dx2 * dx2 + dy2 * dy2) / 2.0f);
          img(yy, xx) += b * falloff;
        }
      }
  }
  return img;
}

} // namespace

// Test 1: downsample_area reduces dimensions correctly
TEST_CASE("autobge_downsample_area_reduces_dimensions") {
  tile_compile::Matrix2Df img(20, 30);
  img.setRandom();
  auto ds = ti::downsample_area(img, 4);
  REQUIRE(ds.rows() == 5);
  REQUIRE(ds.cols() == 7);
}

// Test 2: downsample_area preserves mean for uniform image
TEST_CASE("autobge_downsample_area_preserves_uniform") {
  tile_compile::Matrix2Df img = tile_compile::Matrix2Df::Constant(16, 16, 42.0f);
  auto ds = ti::downsample_area(img, 4);
  REQUIRE(ds.rows() == 4);
  REQUIRE(ds.cols() == 4);
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      REQUIRE(ds(r, c) == Catch::Approx(42.0f).epsilon(1e-5));
}

// Test 3: upscale_lanczos4 returns target dimensions
TEST_CASE("autobge_upscale_lanczos4_target_dimensions") {
  tile_compile::Matrix2Df small(5, 7);
  small.setRandom();
  auto big = ti::upscale_lanczos4(small, 20, 28);
  REQUIRE(big.rows() == 20);
  REQUIRE(big.cols() == 28);
}

// Test 4: transform_to_autobge_working_space with none returns input unchanged
TEST_CASE("autobge_transform_none_returns_input") {
  tile_compile::Matrix2Df img(8, 8);
  img.setRandom();
  img = img.array().abs() + 1.0f;
  ti::BGEConfig::AutoBGEConfig ac{};
  ac.stretch_mode = "none";
  ti::StretchParams params;
  auto out = ti::transform_to_autobge_working_space(img, ac, &params, 0);
  REQUIRE(out.isApprox(img, 1e-5f));
  REQUIRE(params.mode == "none");
}

// Test 5: transform linear round-trip
TEST_CASE("autobge_transform_linear_roundtrip") {
  tile_compile::Matrix2Df img(16, 16);
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 16; ++c)
      img(r, c) = 100.0f + 0.5f * c + 0.3f * r;
  ti::BGEConfig::AutoBGEConfig ac{};
  ac.stretch_mode = "linear";
  ac.stretch_target_median = 0.25f;
  ti::StretchParams params;
  auto stretched = ti::transform_to_autobge_working_space(img, ac, &params, 0);
  auto restored = ti::transform_from_autobge_working_space(stretched, params, 0);
  REQUIRE(restored.isApprox(img, 1e-3f));
}

TEST_CASE("autobge_transform_mtf_roundtrip_uses_configured_target") {
  tile_compile::Matrix2Df img(16, 16);
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 16; ++c)
      img(r, c) = 100.0f + 0.5f * c + 0.3f * r;
  ti::BGEConfig::AutoBGEConfig ac{};
  ac.stretch_mode = "mtf";
  ac.stretch_target_median = 0.35f;
  ti::StretchParams params;
  auto stretched = ti::transform_to_autobge_working_space(img, ac, &params, 0);
  auto restored = ti::transform_from_autobge_working_space(stretched, params, 0);
  REQUIRE(restored.isApprox(img, 1e-3f));
}

// Test 6: generate_autobge_sample_points returns non-empty for valid image
TEST_CASE("autobge_sample_points_nonempty") {
  tile_compile::Matrix2Df img = make_gradient_image(40, 40, 100.0f, 0.5f, 0.3f);
  ti::BGEConfig::AutoBGEConfig ac{};
  ac.num_sample_points = 50;
  ac.patch_size = 5;
  ac.border_margin = 2;
  ac.bright_exclusion_fraction = 0.8f;
  ac.gradient_descent_max_iters = 3;
  auto points = ti::generate_autobge_sample_points(img, ac);
  REQUIRE(!points.empty());
  REQUIRE(points.size() >= 4);
}

TEST_CASE("autobge_random_seed_affects_sample_selection") {
  tile_compile::Matrix2Df img = make_gradient_image(80, 80, 100.0f, 0.2f, 0.1f);
  ti::BGEConfig::AutoBGEConfig ac{};
  ac.num_sample_points = 40;
  ac.patch_size = 5;
  ac.border_margin = 2;
  ac.bright_exclusion_fraction = 0.9f;
  ac.gradient_descent_max_iters = 1;

  std::mt19937 rng_a(11);
  std::mt19937 rng_b(12);
  auto a = ti::generate_autobge_sample_points(img, ac, nullptr, &rng_a);
  auto b = ti::generate_autobge_sample_points(img, ac, nullptr, &rng_b);
  REQUIRE(a.size() >= 4);
  REQUIRE(b.size() >= 4);
  bool different = a.size() != b.size();
  for (size_t i = 0; !different && i < a.size(); ++i)
    different = a[i].x != b[i].x || a[i].y != b[i].y;
  REQUIRE(different);
}

TEST_CASE("autobge_bright_exclusion_rejects_the_bright_fraction") {
  constexpr int W = 100, H = 80;
  tile_compile::Matrix2Df img(H, W);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      img(y, x) = static_cast<float>(x + 1);

  ti::BGEConfig::AutoBGEConfig ac{};
  ac.num_sample_points = 64;
  ac.patch_size = 3;
  ac.border_margin = 1;
  ac.bright_exclusion_fraction = 0.25f;
  ac.gradient_descent_max_iters = 0;

  std::mt19937 rng(42);
  const auto points = ti::generate_autobge_sample_points(img, ac, nullptr, &rng);
  REQUIRE(points.size() >= 16);

  // Rejecting the brightest 25% leaves samples across the first 75 columns.
  // The previous, inverted quantile calculation only admitted the first 25.
  REQUIRE(std::any_of(points.begin(), points.end(),
                      [](const ti::SamplePoint& p) { return p.x >= 50; }));
  REQUIRE(std::all_of(points.begin(), points.end(),
                      [](const ti::SamplePoint& p) { return p.x < 75; }));
}

TEST_CASE("autobge_automatic_sample_points_keep_minimum_spacing") {
  constexpr int W = 120, H = 90;
  auto img = make_gradient_image(W, H, 100.0f, 0.03f, 0.02f);

  ti::BGEConfig::AutoBGEConfig ac{};
  ac.num_sample_points = 80;
  ac.patch_size = 5;
  ac.border_margin = 2;
  ac.bright_exclusion_fraction = 0.1f;
  ac.gradient_descent_max_iters = 20;

  std::mt19937 rng(42);
  const auto points = ti::generate_autobge_sample_points(img, ac, nullptr, &rng);
  REQUIRE(points.size() >= 16);

  const float nominal_spacing =
      std::sqrt(static_cast<float>(W * H) / ac.num_sample_points);
  const float minimum_distance = std::max(1.0f, nominal_spacing * 0.35f);
  const float minimum_distance_sq = minimum_distance * minimum_distance;
  for (size_t i = 0; i < points.size(); ++i) {
    for (size_t j = i + 1; j < points.size(); ++j) {
      const float dx = static_cast<float>(points[i].x - points[j].x);
      const float dy = static_cast<float>(points[i].y - points[j].y);
      REQUIRE(dx * dx + dy * dy >= minimum_distance_sq);
    }
  }
}

TEST_CASE("autobge_fit_minimum_point_floors_are_enforced") {
  constexpr int W = 32, H = 32;
  tile_compile::Matrix2Df img = make_gradient_image(W, H, 100.0f, 0.2f, 0.1f);
  ti::BGEConfig::AutoBGEConfig ac{};
  ac.poly_degree = 2; // 6 terms, implementation requires term count + margin.
  ac.patch_size = 5;
  std::vector<ti::SamplePoint> nine_points;
  for (int i = 0; i < 9; ++i)
    nine_points.push_back({2 + i * 3, 2 + i * 2});
  auto poly = ti::fit_polynomial_autobge(img, nine_points, ac, H, W);
  REQUIRE(poly.cwiseAbs().maxCoeff() == Catch::Approx(0.0f));

  std::vector<ti::SamplePoint> fifteen_points;
  for (int i = 0; i < 15; ++i)
    fifteen_points.push_back({1 + (i * 2) % W, 1 + (i * 3) % H});
  auto rbf = ti::fit_rbf_autobge(img, fifteen_points, ac, H, W);
  REQUIRE(rbf.cwiseAbs().maxCoeff() == Catch::Approx(0.0f));
}

// Test 7: fit_polynomial_autobge recovers linear gradient
TEST_CASE("autobge_fit_polynomial_recovers_linear_gradient") {
  constexpr int W = 60, H = 60;
  const float offset = 100.0f, gx = 0.5f, gy = 0.3f;
  tile_compile::Matrix2Df img = make_gradient_image(W, H, offset, gx, gy);
  ti::BGEConfig::AutoBGEConfig ac{};
  ac.poly_degree = 2;
  ac.patch_size = 5;
  ac.num_sample_points = 80;
  ac.border_margin = 3;
  ac.bright_exclusion_fraction = 0.8f;
  ac.gradient_descent_max_iters = 3;
  auto points = ti::generate_autobge_sample_points(img, ac);
  REQUIRE(points.size() >= 6);
  auto bg = ti::fit_polynomial_autobge(img, points, ac, H, W);
  // Check that background matches the gradient within tolerance
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
      float expected = offset + gx * c + gy * r;
      REQUIRE(bg(r, c) == Catch::Approx(expected).epsilon(0.05));
    }
}

// Test 8: build_autobge_models on uniform+gradient image succeeds
TEST_CASE("autobge_build_models_succeeds_on_gradient") {
  constexpr int W = 64, H = 64;
  auto cfg = make_autobge_config(W, H);
  auto base = make_gradient_image(W, H, 100.0f, 0.4f, 0.2f);
  auto R = add_stars(base, 10, W, H, 1);
  auto G = add_stars(base, 10, W, H, 2);
  auto B = add_stars(base, 10, W, H, 3);
  auto result = ti::build_autobge_models(R, G, B, cfg);
  REQUIRE(result.success);
  REQUIRE(result.channel_models[0].success);
  REQUIRE(result.channel_models[1].success);
  REQUIRE(result.channel_models[2].success);
}

TEST_CASE("autobge_sampling_mask_excludes_points_without_cropping_output") {
  constexpr int W = 64, H = 64;
  auto cfg = make_autobge_config(W, H);
  cfg.sampling_mask_rows = H;
  cfg.sampling_mask_cols = W;
  cfg.sampling_valid_mask.assign(static_cast<size_t>(W * H), 1u);
  for (int y = 16; y < 48; ++y)
    for (int x = 16; x < 48; ++x)
      cfg.sampling_valid_mask[static_cast<size_t>(y * W + x)] = 0u;
  auto base = make_gradient_image(W, H, 100.0f, 0.4f, 0.2f);
  auto result = ti::build_autobge_models(base, base * 1.01f, base * 0.99f, cfg);
  REQUIRE(result.success);
  for (const auto &channel : result.channel_diagnostics)
    for (const auto &point : channel.grid_cells)
      REQUIRE_FALSE((point.center_x >= 16.0f && point.center_x < 48.0f &&
                     point.center_y >= 16.0f && point.center_y < 48.0f));
  REQUIRE(cfg.common_valid_mask[static_cast<size_t>(32 * W + 32)] == 1u);
}

// Test 9: apply_background_extraction with autobge method subtracts gradient
TEST_CASE("autobge_apply_background_extraction_subtracts_gradient") {
  constexpr int W = 64, H = 64;
  auto cfg = make_autobge_config(W, H);
  // Create image with known gradient
  auto base = make_gradient_image(W, H, 100.0f, 0.5f, 0.3f);
  auto R = add_stars(base, 5, W, H, 10);
  auto G = add_stars(base, 5, W, H, 20);
  auto B = add_stars(base, 5, W, H, 30);
  auto horizontal_gradient = [](const tile_compile::Matrix2Df& image) {
    const int quarter = static_cast<int>(image.cols()) / 4;
    const float left = image.leftCols(quarter).mean();
    const float right = image.rightCols(quarter).mean();
    return std::abs(right - left);
  };
  const float original_gradient =
      (horizontal_gradient(R) + horizontal_gradient(G) + horizontal_gradient(B)) / 3.0f;
  // Apply BGE
  std::vector<tile_compile::TileMetrics> empty_metrics;
  tile_compile::TileGrid grid;
  ti::BGEDiagnostics diag;
  bool ok = ti::apply_background_extraction(R, G, B, empty_metrics, grid, cfg, &diag);
  REQUIRE(ok);
  REQUIRE(diag.success);
  // Normalization between stages may preserve or slightly raise the global
  // mean. The relevant result is that the spatial gradient becomes flatter.
  const float corrected_gradient =
      (horizontal_gradient(R) + horizontal_gradient(G) + horizontal_gradient(B)) / 3.0f;
  REQUIRE(corrected_gradient < original_gradient);
}

TEST_CASE("autobge_method_is_authoritative_for_programmatic_config") {
  constexpr int W = 64, H = 64;
  auto cfg = make_autobge_config(W, H);
  cfg.enabled = false;
  cfg.method = "autobge";
  auto base = make_gradient_image(W, H, 100.0f, 0.5f, 0.3f);
  auto R = add_stars(base, 5, W, H, 10);
  auto G = add_stars(base, 5, W, H, 20);
  auto B = add_stars(base, 5, W, H, 30);
  std::vector<tile_compile::TileMetrics> empty_metrics;
  tile_compile::TileGrid grid;
  ti::BGEDiagnostics diag;
  bool ok = ti::apply_background_extraction(R, G, B, empty_metrics, grid, cfg, &diag);
  REQUIRE(ok);
  REQUIRE(diag.bge_method == "autobge");
}

TEST_CASE("autobge_finalize_is_atomic_when_channel_model_missing") {
  constexpr int W = 16, H = 16;
  auto cfg = make_autobge_config(W, H);
  auto R = make_gradient_image(W, H, 100.0f, 0.1f, 0.1f);
  auto G = R;
  auto B = R;
  auto R0 = R;
  auto G0 = G;
  auto B0 = B;

  std::array<ti::BackgroundModel, 3> models;
  models[0].model = tile_compile::Matrix2Df::Constant(H, W, 5.0f);
  models[0].success = true;
  models[1].success = false;
  models[2].model = tile_compile::Matrix2Df::Constant(H, W, 5.0f);
  models[2].success = true;

  ti::BGEDiagnostics diag;
  bool ok = ti::finalize_bge_from_channel_models(R, G, B, models, {}, cfg, &diag);
  REQUIRE_FALSE(ok);
  REQUIRE(R.isApprox(R0));
  REQUIRE(G.isApprox(G0));
  REQUIRE(B.isApprox(B0));
  REQUIRE(diag.failure_reason == "partial_channel_model");
}
#else
int tile_compile_tests_autobge_stub() { return 0; }
#endif
