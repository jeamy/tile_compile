 #if __has_include(<catch2/catch_test_macros.hpp>)
 #include "tile_compile/metrics/metrics.hpp"
 #include "tile_compile/core/types.hpp"

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
 #else
 int tile_compile_tests_metrics_stub() { return 0; }
 #endif
