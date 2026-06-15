 #if __has_include(<catch2/catch_test_macros.hpp>)
 #include "tile_compile/metrics/metrics.hpp"
 #include "tile_compile/core/types.hpp"

 #include <algorithm>
 #include <cmath>
 #include <vector>

 #include <catch2/catch_approx.hpp>
 #include <catch2/catch_test_macros.hpp>

 TEST_CASE("calculate_frame_metrics_background_matches_median") {
     tile_compile::Matrix2Df frame(2, 2);
     frame << 1.0f, 2.0f,
              3.0f, 4.0f;

     auto m = tile_compile::metrics::calculate_frame_metrics(frame);

     REQUIRE(m.background == Catch::Approx(2.5f).epsilon(1e-5));
 }

TEST_CASE("calculate_global_weights_are_positive_and_not_normalized") {
    std::vector<tile_compile::FrameMetrics> ms(3);
    ms[0] = {1.0f, 1.0f, 1.0f, 1.0f};
    ms[1] = {2.0f, 2.0f, 2.0f, 1.0f};
    ms[2] = {3.0f, 3.0f, 3.0f, 1.0f};

    auto w = tile_compile::metrics::calculate_global_weights(ms, 0.4f, 0.3f, 0.3f, -3.0f, 3.0f);

    REQUIRE(w.size() == 3);
    REQUIRE(w[0] > 0.0f);
    REQUIRE(w[1] > 0.0f);
    REQUIRE(w[2] > 0.0f);

    // With these synthetic metrics, the first frame should get the highest weight
    // (lower background/noise), and the third the lowest.
    REQUIRE(w[0] > w[1]);
    REQUIRE(w[1] > w[2]);

    // Must not be normalized to sum=1.
    REQUIRE(w.sum() != Catch::Approx(1.0f).epsilon(1e-5));
}

TEST_CASE("global star weights penalize worse fwhm") {
    std::vector<tile_compile::FrameMetrics> ms(3);
    ms[0] = {1.0f, 1.0f, 1.0f, 1.0f};
    ms[1] = {1.0f, 1.0f, 1.0f, 1.0f};
    ms[2] = {1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<tile_compile::metrics::FrameStarMetrics> stars(3);
    stars[0] = {8.0f, 8.0f, 8.0f, 1.0f, 8.0f, 100};
    stars[1] = {10.0f, 10.0f, 10.0f, 1.0f, 10.0f, 100};
    stars[2] = {14.0f, 14.0f, 14.0f, 1.0f, 14.0f, 100};

    auto w = tile_compile::metrics::calculate_global_weights_with_stars(
        ms, stars, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -3.0f, 3.0f,
        false, 1.0f);

    REQUIRE(w.size() == 3);
    REQUIRE(w[0] > w[1]);
    REQUIRE(w[1] > w[2]);
}

TEST_CASE("adaptive global weights fall back to static weights on tied predictive utility") {
    std::vector<tile_compile::FrameMetrics> ms(5);
    ms[0] = {1.0f, 1.0f, 5.0f, 1.0f};
    ms[1] = {2.0f, 2.0f, 4.0f, 1.0f};
    ms[2] = {3.0f, 3.0f, 3.0f, 1.0f};
    ms[3] = {4.0f, 4.0f, 2.0f, 1.0f};
    ms[4] = {5.0f, 5.0f, 1.0f, 1.0f};

    const auto w_static = tile_compile::metrics::calculate_global_weights(
        ms, 0.4f, 0.3f, 0.3f, -3.0f, 3.0f, false, 1.0f);
    const auto w_adaptive = tile_compile::metrics::calculate_global_weights(
        ms, 0.4f, 0.3f, 0.3f, -3.0f, 3.0f, true, 1.0f);

    REQUIRE(w_static.size() == w_adaptive.size());
    for (int i = 0; i < w_static.size(); ++i) {
        REQUIRE(w_adaptive[i] == Catch::Approx(w_static[i]).margin(1.0e-6));
    }
}

TEST_CASE("adaptive global weights respond to asymmetric predictive utility") {
    std::vector<tile_compile::FrameMetrics> ms(5);
    ms[0] = {3.0f, 1.0f, 5.0f, 1.0f};
    ms[1] = {1.0f, 2.0f, 4.0f, 1.0f};
    ms[2] = {5.0f, 3.0f, 3.0f, 1.0f};
    ms[3] = {2.0f, 4.0f, 2.0f, 1.0f};
    ms[4] = {4.0f, 5.0f, 1.0f, 1.0f};

    const auto w_static = tile_compile::metrics::calculate_global_weights(
        ms, 0.4f, 0.3f, 0.3f, -3.0f, 3.0f, false, 1.0f);
    const auto w_adaptive = tile_compile::metrics::calculate_global_weights(
        ms, 0.4f, 0.3f, 0.3f, -3.0f, 3.0f, true, 1.0f);

    REQUIRE(w_static.size() == w_adaptive.size());
    float max_abs_delta = 0.0f;
    for (int i = 0; i < w_static.size(); ++i) {
        max_abs_delta = std::max(max_abs_delta, std::fabs(w_adaptive[i] - w_static[i]));
        REQUIRE(w_adaptive[i] > 0.0f);
    }
    REQUIRE(max_abs_delta > 1.0e-4f);
}
 #else
 int tile_compile_tests_metrics_stub() { return 0; }
 #endif
