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
 #else
 int tile_compile_tests_utils_stub() { return 0; }
 #endif
