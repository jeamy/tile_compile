 #if __has_include(<catch2/catch_test_macros.hpp>)
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

 TEST_CASE("cosmetic_correction_cfa_preserves_compact_real_structure") {
     tile_compile::Matrix2Df mosaic = tile_compile::Matrix2Df::Constant(7, 7, 100.0f);

     for (int y = 2; y <= 4; ++y) {
         for (int x = 2; x <= 4; ++x) {
             mosaic(y, x) = 130.0f;
         }
     }
     mosaic(3, 3) = 150.0f;

     const auto corrected =
         tile_compile::image::cosmetic_correction_cfa(mosaic, 2.5f, true, 0, 0);

     REQUIRE(corrected(3, 3) == Catch::Approx(150.0f).epsilon(1e-5));
     REQUIRE(corrected(3, 2) == Catch::Approx(130.0f).epsilon(1e-5));
     REQUIRE(corrected(2, 3) == Catch::Approx(130.0f).epsilon(1e-5));
 }
 #else
 int tile_compile_tests_utils_stub() { return 0; }
 #endif
