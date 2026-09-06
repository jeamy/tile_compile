// M5 tests for scale-specific source-space quality maps
// (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md
//  sections 13.1-13.5). The acceptance criterion of M5 is that the composite
// stays byte-comparable with the "übernommene, dokumentierte Quality-Semantik"
// --- i.e. the legacy geometric-mean collapse of the per-scale psi maps. For
// MONO the analysis proxy is the L plane unchanged, so with an all-valid mask
// the new path's pyramid input is bit-identical to the legacy path's and the
// composite must be bit-identical too.

#include "tile_compile/reconstruction/source_quality_maps.hpp"
#include "tile_compile/metrics/aqmh_quality_map.hpp"

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <cstdint>

using namespace tile_compile;
using namespace tile_compile::reconstruction;

namespace {

Matrix2Df synthetic_frame(int w, int h) {
  // Smooth low-frequency structure + a bright compact bump + a faint ramp so
  // robust_zscore has non-degenerate spread at every scale. Fully
  // deterministic.
  Matrix2Df m(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const double base = 120.0 + 12.0 * std::sin(x * 0.21) +
                          9.0 * std::cos(y * 0.17) + 0.03 * (x + y);
      const double dx = x - 0.4 * w, dy = y - 0.55 * h;
      const double bump = 40.0 * std::exp(-(dx * dx + dy * dy) / (2.0 * 25.0));
      m(y, x) = static_cast<float>(base + bump);
    }
  }
  return m;
}

bool bit_equal_or_both_non_finite(float a, float b) {
  const bool fa = std::isfinite(a), fb = std::isfinite(b);
  if (!fa && !fb) return true;
  if (fa != fb) return false;
  return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b);
}

}  // namespace

TEST_CASE("M5 composite is byte-identical to the legacy geometric-mean Q-map "
          "(MONO proxy, all-valid mask)") {
  const int w = 96, h = 96;
  const Matrix2Df frame = synthetic_frame(w, h);
  config::AqmhPyramidConfig cfg;  // defaults

  const auto legacy = metrics::compute_aqmh_quality_map(
      frame, /*canvas_mask=*/{}, w, h, cfg);
  const auto m5 = compute_source_quality_maps(frame, /*source_valid_mask=*/{}, w,
                                              h, cfg);

  REQUIRE(m5.q_map.rows() == legacy.q_map.rows());
  REQUIRE(m5.q_map.cols() == legacy.q_map.cols());
  int mismatches = 0;
  for (int i = 0; i < legacy.q_map.size(); ++i)
    if (!bit_equal_or_both_non_finite(legacy.q_map.data()[i],
                                      m5.q_map.data()[i]))
      ++mismatches;
  REQUIRE(mismatches == 0);
  REQUIRE(m5.diagnostics.computed_scales >= 2);
}

TEST_CASE("M5 sink streams every computed scale with legacy downsample factors "
          "and never holds more than one full scale map resident") {
  const int w = 96, h = 96;
  const Matrix2Df frame = synthetic_frame(w, h);
  config::AqmhPyramidConfig cfg;

  std::vector<int> seen_scale_index;
  std::vector<int> seen_factor;
  int max_concurrent = 0;
  int concurrent = 0;
  QualityScaleMapSink sink = [&](int scale_index, int downsample_factor,
                                 const Matrix2Df &psi) {
    ++concurrent;
    max_concurrent = std::max(max_concurrent, concurrent);
    seen_scale_index.push_back(scale_index);
    seen_factor.push_back(downsample_factor);
    REQUIRE(psi.rows() == h);
    REQUIRE(psi.cols() == w);
    --concurrent;
  };

  const auto m5 = compute_source_quality_maps(frame, {}, w, h, cfg, sink);

  REQUIRE(m5.scale_maps.empty());  // streamed, not retained
  REQUIRE(static_cast<int>(seen_scale_index.size()) ==
          m5.diagnostics.computed_scales);
  for (size_t k = 0; k < seen_scale_index.size(); ++k) {
    REQUIRE(seen_scale_index[k] == static_cast<int>(k));
    REQUIRE(seen_factor[k] == (1 << (2 * static_cast<int>(k))));
  }
  REQUIRE(max_concurrent == 1);
  REQUIRE(m5.diagnostics.peak_resident_scale_maps == 1);
}

TEST_CASE("M5 without a sink retains all scale maps in source geometry") {
  const int w = 96, h = 96;
  const Matrix2Df frame = synthetic_frame(w, h);
  config::AqmhPyramidConfig cfg;

  const auto m5 = compute_source_quality_maps(frame, {}, w, h, cfg);
  REQUIRE(static_cast<int>(m5.scale_maps.size()) ==
          m5.diagnostics.computed_scales);
  for (const auto &sm : m5.scale_maps) {
    REQUIRE(sm.psi.rows() == h);
    REQUIRE(sm.psi.cols() == w);
    REQUIRE(sm.downsample_factor == (1 << (2 * sm.scale_index)));
  }
  REQUIRE(m5.diagnostics.peak_resident_scale_maps ==
          m5.diagnostics.computed_scales);
}

TEST_CASE("M5 hard mask is re-applied to the composite: masked region is "
          "exactly 0, and the deep interior is NaN at the finest scale") {
  const int w = 96, h = 96;
  const Matrix2Df frame = synthetic_frame(w, h);
  config::AqmhPyramidConfig cfg;

  // Mask out a solid block. The guarantee of THIS layer:
  //  (a) the geometric-mean composite re-applies the hard mask -> exactly 0
  //      over the whole block;
  //  (b) far enough from any mask edge (> window radius), even the finest
  //      scale has no finite window support and stays NaN.
  // Window-support leakage in a thin band just inside the mask edge is
  // inherent to the multiscale statistics and matches the legacy path; the
  // plan's zero-veto-through-resampling guarantee (13.5) is enforced by an
  // explicit veto stream in the cache/region-read layer, tested there.
  std::vector<uint8_t> mask(static_cast<size_t>(w) * h, 1u);
  const int bx0 = 8, bx1 = 40, by0 = 8, by1 = 44;  // 32x36 block
  for (int y = by0; y < by1; ++y)
    for (int x = bx0; x < bx1; ++x)
      mask[static_cast<size_t>(y) * w + x] = 0u;

  const int margin = cfg.base_window_px + 4;  // clear of all window support
  bool deep_interior_finite_at_fine_scale = false;
  QualityScaleMapSink sink = [&](int scale_index, int, const Matrix2Df &psi) {
    if (scale_index != 0) return;
    for (int y = by0 + margin; y < by1 - margin; ++y)
      for (int x = bx0 + margin; x < bx1 - margin; ++x)
        if (std::isfinite(psi(y, x))) deep_interior_finite_at_fine_scale = true;
  };

  const auto m5 = compute_source_quality_maps(frame, mask, w, h, cfg, sink);

  for (int y = by0; y < by1; ++y)
    for (int x = bx0; x < bx1; ++x)
      REQUIRE(m5.q_map(y, x) == 0.0f);
  REQUIRE_FALSE(deep_interior_finite_at_fine_scale);
}

TEST_CASE("M5 artifact_confidence is a source-geometry map in [0,1] on its "
          "valid support") {
  const int w = 96, h = 96;
  const Matrix2Df frame = synthetic_frame(w, h);
  config::AqmhPyramidConfig cfg;

  const auto m5 = compute_source_quality_maps(frame, {}, w, h, cfg);
  REQUIRE(m5.artifact_confidence.rows() == h);
  REQUIRE(m5.artifact_confidence.cols() == w);
  int finite_count = 0;
  for (int i = 0; i < m5.artifact_confidence.size(); ++i) {
    const float v = m5.artifact_confidence.data()[i];
    if (std::isfinite(v)) {
      ++finite_count;
      REQUIRE(v >= 0.0f);
      REQUIRE(v <= 1.0f);
    }
  }
  REQUIRE(finite_count > 0);
}
