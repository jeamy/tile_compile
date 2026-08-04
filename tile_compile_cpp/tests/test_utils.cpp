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
     // Single bright star pixel — should NOT define the scale
     R(0, 0) = 60000.0f;
     G(0, 0) = 60000.0f;
     B(0, 0) = 60000.0f;

     const auto stretch = tile_compile::core::stretch_rgb_to_u32_linear_from_zero_inplace(
         R, G, B);

     REQUIRE(stretch.applied);
     // p1 floor: lowest 1% of values (200 in this uniform data)
     REQUIRE(stretch.low >= 199.0f);
     REQUIRE(stretch.low <= 201.0f);
     // robust_max = p99.9 of all pixels, which is ~299 (not 60000)
     REQUIRE(stretch.high < 1000.0f);  // robust max from the bulk pixels
     REQUIRE(stretch.high > 100.0f);   // should be in the [200,300] range
     // Background pixel (x=0, value=200) is at the floor -> maps to 0
     REQUIRE(R(50, 0) < 1000000.0f);
     // Bright signal pixel (x=90, value=290) should fill most of the range
     REQUIRE(R(50, 90) > 3000000000.0f);
     // Star pixel should be clamped to uint32 max
     REQUIRE(R(0, 0) == Catch::Approx(4294967295.0f).margin(1024.0f));
 }

 TEST_CASE("masked_rgb_stretch_excludes_partial_canvas_outliers") {
     tile_compile::Matrix2Df R(10, 10);
     tile_compile::Matrix2Df G(10, 10);
     tile_compile::Matrix2Df B(10, 10);
     std::vector<uint8_t> common_mask(100, 1);
     for (int row = 0; row < 10; ++row) {
         for (int col = 0; col < 10; ++col) {
             const float val = 100.0f + static_cast<float>(col) * 10.0f;
             R(row, col) = val;
             G(row, col) = val;
             B(row, col) = val;
         }
     }
     R(0, 0) = 1000000.0f;
     G(0, 0) = 1000000.0f;
     B(0, 0) = 1000000.0f;
     common_mask[0] = 0;

     const auto stretch = tile_compile::core::stretch_rgb_to_u32_linear_from_zero_inplace(
         R, G, B, common_mask);

     REQUIRE(stretch.applied);
     REQUIRE(stretch.sample_count == 297);
     // p99.9 of the 99 valid pixels per channel (values 100..190)
     REQUIRE(stretch.high == Catch::Approx(190.0f));
     // p1 floor = 100 (lowest value in valid data)
     REQUIRE(stretch.low == Catch::Approx(100.0f));
     // Background-level pixel (col=0, value=100) is at floor -> maps to 0
     REQUIRE(R(5, 0) < 1000000.0f);
     // Bright pixel (col=9, value=190) maps to full range
     REQUIRE(R(5, 9) == Catch::Approx(4294967295.0f).margin(1024.0f));
     // Masked outlier is still clamped to max
     REQUIRE(R(0, 0) == Catch::Approx(4294967295.0f).margin(1024.0f));
 }

 TEST_CASE("linear_grayscale_stretch_scales_zero_to_max_into_full_u16_range") {
     // 100 pixels: 99 normal pixels at 100..199, one bright outlier at 8000.
     // With background-subtracted p99.9 stretch, the outlier should be clamped
     // and not define the scale; the normal pixels should fill the output range.
     tile_compile::Matrix2Df img(10, 10);
     for (int i = 0; i < 100; ++i)
         img.data()[i] = 100.0f + static_cast<float>(i);  // 100..199
     img(0, 0) = 8000.0f;  // bright outlier (like a star)

     const auto stretch =
         tile_compile::core::stretch_to_u16_linear_from_zero_inplace(img);

     REQUIRE(stretch.applied);
     // p1 floor: ~101 (minimum non-outlier value)
     REQUIRE(stretch.low >= 100.0f);
     REQUIRE(stretch.low <= 102.0f);
     // robust_max = p99.9 of the 99 positive values (excluding outlier effect)
     REQUIRE(stretch.high < 1000.0f);   // should be in the ~199 range
     REQUIRE(stretch.high > 50.0f);
     // Outlier should be clamped to 65535
     REQUIRE(img(0, 0) == Catch::Approx(65535.0f).margin(1.0f));
     // A bright normal pixel (value=180) should occupy a significant fraction
     REQUIRE(img(8, 0) > 30000.0f);
     // A near-floor pixel (value=105) should be small but positive (lifted, not clamped)
     REQUIRE(img(0, 5) > 1000.0f);
     REQUIRE(img(0, 5) < 10000.0f);
 }

 TEST_CASE("rgb_stretch_lifts_below_background_pixels_above_zero") {
     // Simulate sky background at ~200 with noise dipping below to ~180.
     // The p1 floor should sit below the background so that below-bg pixels
     // are lifted to a positive value, not clamped to 0.
     tile_compile::Matrix2Df R(100, 100);
     tile_compile::Matrix2Df G(100, 100);
     tile_compile::Matrix2Df B(100, 100);
     for (int y = 0; y < 100; ++y) {
         for (int x = 0; x < 100; ++x) {
             // Background ~200 with per-pixel noise spread
             const float noise = static_cast<float>((x * 7 + y * 13) % 40) - 20.0f;
             const float val = 200.0f + noise;  // range ~180..219
             R(y, x) = val;
             G(y, x) = val;
             B(y, x) = val;
         }
     }
     // A few bright signal pixels
     R(50, 50) = 400.0f;
     G(50, 50) = 400.0f;
     B(50, 50) = 400.0f;

     const auto stretch = tile_compile::core::stretch_rgb_to_u32_linear_from_zero_inplace(
         R, G, B);

     REQUIRE(stretch.applied);
     // Floor (p1) should be well below the background (~180)
     REQUIRE(stretch.low < 190.0f);
     REQUIRE(stretch.low > 170.0f);
     // A below-background pixel (value ~185) should map to a positive value
     // (it is above the p1 floor, so it must not be clamped to 0)
     bool found_positive_below_bg = false;
     for (int y = 0; y < 100; ++y) {
         for (int x = 0; x < 100; ++x) {
             if (R(y, x) > 0.0f && R(y, x) < 500000000.0f) {
                 found_positive_below_bg = true;
                 break;
             }
         }
         if (found_positive_below_bg) break;
     }
     REQUIRE(found_positive_below_bg);
     // Bright signal pixel should fill most of the range
     REQUIRE(R(50, 50) > 3000000000.0f);
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

 TEST_CASE("cosmetic_correction_fixes_cold_dead_pixel") {
     tile_compile::Matrix2Df frame =
         tile_compile::Matrix2Df::Constant(9, 9, 100.0f);
     frame(4, 4) = 0.0f;

     const auto corrected =
         tile_compile::image::cosmetic_correction(frame, 2.5f, true);

     REQUIRE(corrected(4, 4) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(corrected(3, 3) == Catch::Approx(100.0f).epsilon(1e-5));
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

 TEST_CASE("cosmetic_correction_cfa_fixes_cold_dead_pixel") {
     tile_compile::Matrix2Df mosaic =
         tile_compile::Matrix2Df::Constant(9, 9, 100.0f);
     mosaic(4, 4) = 0.0f;

     const auto corrected =
         tile_compile::image::cosmetic_correction_cfa(mosaic, 2.5f, true, 0, 0);

     REQUIRE(corrected(4, 4) == Catch::Approx(100.0f).epsilon(1e-5));
     REQUIRE(corrected(2, 2) == Catch::Approx(100.0f).epsilon(1e-5));
 }

 TEST_CASE("cosmetic_correction_cfa_preserves_supported_noise_dip") {
     tile_compile::Matrix2Df mosaic =
         tile_compile::Matrix2Df::Constant(9, 9, 100.0f);
     mosaic(2, 2) = 98.0f;
     mosaic(2, 4) = 95.0f;
     mosaic(4, 2) = 96.0f;
     mosaic(4, 4) = 97.0f;

     const auto corrected =
         tile_compile::image::cosmetic_correction_cfa(mosaic, 2.5f, true, 0, 0);

     REQUIRE(corrected(4, 4) == Catch::Approx(97.0f).epsilon(1e-5));
     REQUIRE(corrected(2, 2) == Catch::Approx(98.0f).epsilon(1e-5));
     REQUIRE(corrected(2, 4) == Catch::Approx(95.0f).epsilon(1e-5));
     REQUIRE(corrected(4, 2) == Catch::Approx(96.0f).epsilon(1e-5));
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

 TEST_CASE("two_pass_sigma_clipped_mean_rejects_outliers") {
     // 99 samples at 100.0, one bright outlier at 8000.0.
     std::vector<float> samples(99, 100.0f);
     samples.push_back(8000.0f);
     const float mean =
         tile_compile::core::two_pass_sigma_clipped_mean(samples);
     // The outlier must not pull the mean above the inlier level.
     REQUIRE(mean == Catch::Approx(100.0f).margin(0.5f));
 }
 #else
 int tile_compile_tests_utils_stub() { return 0; }
 #endif
