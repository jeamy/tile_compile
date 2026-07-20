 #if __has_include(<catch2/catch_test_macros.hpp>)
 #include "tile_compile/core/utils.hpp"
 #include "tile_compile/image/processing.hpp"
 #include "tile_compile/core/types.hpp"

 #include <algorithm>
 #include <vector>

 #include <catch2/catch_approx.hpp>
 #include <catch2/catch_test_macros.hpp>

 TEST_CASE("normalize_frame_background_scales_median") {
     tile_compile::Matrix2Df frame(2, 2);
     frame << 1.0f, 2.0f,
              3.0f, 4.0f;

     auto out = tile_compile::image::normalize_frame(frame, 10.0f, 1.0f, tile_compile::NormalizationMode::BACKGROUND);

     std::vector<float> vals(out.data(), out.data() + out.size());
     std::sort(vals.begin(), vals.end());
     float med = (vals[1] + vals[2]) / 2.0f;

     REQUIRE(med == Catch::Approx(10.0f).epsilon(1e-5));
 }

 TEST_CASE("quantile_rgb_stretch_maps_to_full_uint32_range") {
     tile_compile::Matrix2Df R(100, 100);
     tile_compile::Matrix2Df G(100, 100);
     tile_compile::Matrix2Df B(100, 100);
     for (int y = 0; y < 100; ++y) {
         for (int x = 0; x < 100; ++x) {
             const float base = 200.0f + static_cast<float>(x);
             R(y, x) = base;
             G(y, x) = base;
             B(y, x) = base;
         }
     }
     R(0, 0) = 60000.0f;
     G(0, 0) = 60000.0f;
     B(0, 0) = 60000.0f;

     const auto stretch = tile_compile::core::stretch_rgb_to_u32_linear_from_zero_inplace(
         R, G, B);

     REQUIRE(stretch.applied);
     REQUIRE(stretch.low == Catch::Approx(0.0f).margin(1e-6));
     REQUIRE(stretch.high == Catch::Approx(60000.0f).margin(1e-3));
     REQUIRE(R(50, 50) > 0.0f);
     REQUIRE(R(50, 50) < 4294967295.0f);
     REQUIRE(R(0, 0) == Catch::Approx(4294967295.0f).margin(1024.0f));
 }

 TEST_CASE("linear_grayscale_stretch_scales_zero_to_max_into_full_u16_range") {
     tile_compile::Matrix2Df img(2, 3);
     img << 0.0f, 100.0f, 200.0f,
            300.0f, 400.0f, 800.0f;

     const auto stretch =
         tile_compile::core::stretch_to_u16_linear_from_zero_inplace(img);

     REQUIRE(stretch.applied);
     REQUIRE(stretch.low == Catch::Approx(0.0f).margin(1e-6));
     REQUIRE(stretch.high == Catch::Approx(800.0f).margin(1e-6));
     REQUIRE(img(0, 0) == Catch::Approx(0.0f).margin(1e-6));
     REQUIRE(img(1, 2) == Catch::Approx(65535.0f).margin(1e-3));
     REQUIRE(img(0, 2) == Catch::Approx(65535.0f * 0.25f).margin(1.0f));
 }

 TEST_CASE("suppress_isolated_chroma_speckles_fixes_single_channel_outlier") {
     tile_compile::Matrix2Df R = tile_compile::Matrix2Df::Constant(7, 7, 100.0f);
     tile_compile::Matrix2Df G = tile_compile::Matrix2Df::Constant(7, 7, 100.0f);
     tile_compile::Matrix2Df B = tile_compile::Matrix2Df::Constant(7, 7, 100.0f);
     R(3, 3) = 170.0f;

     std::vector<uint8_t> mask(static_cast<size_t>(7 * 7), 1u);
     const auto stats = tile_compile::image::suppress_isolated_chroma_speckles_rgb_inplace(
         R, G, B, &mask, 7, 7);

     REQUIRE(stats.corrected_pixels == 1);
     REQUIRE(R(3, 3) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(G(3, 3) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(B(3, 3) == Catch::Approx(100.0f).epsilon(1e-5));
 }

 TEST_CASE("suppress_isolated_chroma_speckles_preserves_non_isolated_bright_structure") {
     tile_compile::Matrix2Df R = tile_compile::Matrix2Df::Constant(7, 7, 100.0f);
     tile_compile::Matrix2Df G = tile_compile::Matrix2Df::Constant(7, 7, 100.0f);
     tile_compile::Matrix2Df B = tile_compile::Matrix2Df::Constant(7, 7, 100.0f);
     for (int y = 2; y <= 4; ++y) {
         for (int x = 2; x <= 4; ++x) {
             R(y, x) = 170.0f;
             G(y, x) = 150.0f;
             B(y, x) = 150.0f;
         }
     }

     std::vector<uint8_t> mask(static_cast<size_t>(7 * 7), 1u);
     const auto stats = tile_compile::image::suppress_isolated_chroma_speckles_rgb_inplace(
         R, G, B, &mask, 7, 7);

     REQUIRE(stats.corrected_pixels == 0);
     REQUIRE(R(3, 3) == Catch::Approx(170.0f).epsilon(1e-5));
     REQUIRE(G(3, 3) == Catch::Approx(150.0f).epsilon(1e-5));
     REQUIRE(B(3, 3) == Catch::Approx(150.0f).epsilon(1e-5));
 }

 TEST_CASE("suppress_isolated_chroma_speckles_fixes_small_red_cluster") {
     tile_compile::Matrix2Df R = tile_compile::Matrix2Df::Constant(9, 9, 100.0f);
     tile_compile::Matrix2Df G = tile_compile::Matrix2Df::Constant(9, 9, 100.0f);
     tile_compile::Matrix2Df B = tile_compile::Matrix2Df::Constant(9, 9, 100.0f);

     R(4, 4) = 175.0f;
     R(4, 5) = 172.0f;
     R(5, 4) = 178.0f;

     std::vector<uint8_t> mask(static_cast<size_t>(9 * 9), 1u);
     const auto stats = tile_compile::image::suppress_isolated_chroma_speckles_rgb_inplace(
         R, G, B, &mask, 9, 9);

     REQUIRE(stats.corrected_pixels >= 3);
     REQUIRE(R(4, 4) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(R(4, 5) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(R(5, 4) == Catch::Approx(100.0f).epsilon(1e-5));
 }

 TEST_CASE("suppress_isolated_chroma_speckles_fixes_small_two_channel_cluster") {
     tile_compile::Matrix2Df R = tile_compile::Matrix2Df::Constant(9, 9, 100.0f);
     tile_compile::Matrix2Df G = tile_compile::Matrix2Df::Constant(9, 9, 100.0f);
     tile_compile::Matrix2Df B = tile_compile::Matrix2Df::Constant(9, 9, 100.0f);

     R(4, 4) = 172.0f;
     B(4, 4) = 168.0f;
     R(4, 5) = 169.0f;
     B(4, 5) = 166.0f;

     std::vector<uint8_t> mask(static_cast<size_t>(9 * 9), 1u);
     const auto stats = tile_compile::image::suppress_isolated_chroma_speckles_rgb_inplace(
         R, G, B, &mask, 9, 9);

     REQUIRE(stats.corrected_pixels >= 2);
     REQUIRE(R(4, 4) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(B(4, 4) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(R(4, 5) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(B(4, 5) == Catch::Approx(100.0f).epsilon(1e-5));
 }

 TEST_CASE("cosmetic_correction_cfa_fixes_local_same_parity_outlier") {
     tile_compile::Matrix2Df mosaic = tile_compile::Matrix2Df::Constant(9, 9, 100.0f);

     mosaic(1, 7) = 90.0f;
     mosaic(3, 7) = 110.0f;
     mosaic(5, 7) = 90.0f;
     mosaic(7, 1) = 110.0f;
     mosaic(7, 3) = 90.0f;
     mosaic(7, 5) = 110.0f;
     mosaic(7, 7) = 90.0f;
     mosaic(3, 3) = 116.0f;

     const auto corrected =
         tile_compile::image::cosmetic_correction_cfa(mosaic, 2.5f, true, 0, 0);

     REQUIRE(corrected(3, 3) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(corrected(1, 7) == Catch::Approx(90.0f).epsilon(1e-5));
     REQUIRE(corrected(3, 7) == Catch::Approx(110.0f).epsilon(1e-5));
 }

 TEST_CASE("cosmetic_correction_cfa_corrects_compact_peak_like_outlier") {
     tile_compile::Matrix2Df mosaic = tile_compile::Matrix2Df::Constant(7, 7, 100.0f);

     for (int y = 2; y <= 4; ++y) {
         for (int x = 2; x <= 4; ++x) {
             mosaic(y, x) = 130.0f;
         }
     }
     mosaic(3, 3) = 150.0f;

     const auto corrected =
         tile_compile::image::cosmetic_correction_cfa(mosaic, 2.5f, true, 0, 0);

     REQUIRE(corrected(3, 3) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(corrected(1, 1) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(corrected(5, 5) == Catch::Approx(100.0f).epsilon(1e-5));
 }

 TEST_CASE("cosmetic_correction_cfa_corrects_strong_outlier_near_broad_structure") {
     tile_compile::Matrix2Df mosaic = tile_compile::Matrix2Df::Constant(9, 9, 100.0f);

     for (int y = 2; y <= 6; ++y) {
         for (int x = 2; x <= 6; ++x) {
             mosaic(y, x) = 120.0f;
         }
     }
     mosaic(4, 4) = 185.0f;

     const auto corrected =
         tile_compile::image::cosmetic_correction_cfa(mosaic, 2.5f, true, 0, 0);

     REQUIRE(corrected(4, 4) == Catch::Approx(120.0f).epsilon(1e-5));
     REQUIRE(corrected(4, 3) == Catch::Approx(120.0f).epsilon(1e-5));
     REQUIRE(corrected(3, 4) == Catch::Approx(120.0f).epsilon(1e-5));
 }
 #else
 int tile_compile_tests_utils_stub() { return 0; }
 #endif
