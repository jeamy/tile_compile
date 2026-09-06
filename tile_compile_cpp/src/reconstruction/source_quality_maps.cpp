#include "tile_compile/reconstruction/source_quality_maps.hpp"

#include "tile_compile/metrics/aqmh_quality_map.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

namespace tile_compile::reconstruction {
namespace {

bool finite_f(float v) {
  return (std::bit_cast<uint32_t>(v) & 0x7f800000u) != 0x7f800000u;
}
float nan_value() { return std::numeric_limits<float>::quiet_NaN(); }

// Bilinear upsample of one scale's map to source geometry. Mirrors the
// interpolation in metrics::aqmh_quality_map.cpp's accumulate_upsampled_log_psi
// (same half-pixel-centred sample position, same clamped 2x2 stencil, same
// support-weighted normalisation) so the streamed scale maps stay consistent
// with the geometric-mean composite that path produces.
//
// Zero-veto preservation (plan 13.5): a source pixel whose interpolation has
// no finite support, or whose interpolated value is <= 0, is written as NaN.
// A hard zero is never turned into a positive value.
Matrix2Df upsample_to_source(const Matrix2Df &src, int out_w, int out_h,
                             int factor) {
  const int rows = static_cast<int>(src.rows());
  const int cols = static_cast<int>(src.cols());
  Matrix2Df out(out_h, out_w);
  out.setConstant(nan_value());

  if (factor <= 1 && rows == out_h && cols == out_w) {
    for (int y = 0; y < out_h; ++y)
      for (int x = 0; x < out_w; ++x) {
        const float v = src(y, x);
        out(y, x) = (finite_f(v) && v > 0.0f) ? v : nan_value();
      }
    return out;
  }

  struct Interp1D {
    int idx0, idx1;
    float w0, w1;
  };
  std::vector<Interp1D> x_lut(static_cast<size_t>(out_w));
  for (int x = 0; x < out_w; ++x) {
    const float sx =
        (static_cast<float>(x) + 0.5f) / static_cast<float>(factor) - 0.5f;
    const int x0 = static_cast<int>(std::floor(sx));
    x_lut[static_cast<size_t>(x)] = {
        std::clamp(x0, 0, cols - 1), std::clamp(x0 + 1, 0, cols - 1),
        std::max(1.0f - std::abs(sx - static_cast<float>(x0)), 0.0f),
        std::max(1.0f - std::abs(sx - static_cast<float>(x0 + 1)), 0.0f)};
  }
  std::vector<Interp1D> y_lut(static_cast<size_t>(out_h));
  for (int y = 0; y < out_h; ++y) {
    const float sy =
        (static_cast<float>(y) + 0.5f) / static_cast<float>(factor) - 0.5f;
    const int y0 = static_cast<int>(std::floor(sy));
    y_lut[static_cast<size_t>(y)] = {
        std::clamp(y0, 0, rows - 1), std::clamp(y0 + 1, 0, rows - 1),
        std::max(1.0f - std::abs(sy - static_cast<float>(y0)), 0.0f),
        std::max(1.0f - std::abs(sy - static_cast<float>(y0 + 1)), 0.0f)};
  }

  for (int y = 0; y < out_h; ++y) {
    const auto &yl = y_lut[static_cast<size_t>(y)];
    for (int x = 0; x < out_w; ++x) {
      const auto &xl = x_lut[static_cast<size_t>(x)];
      const float s00 = src(yl.idx0, xl.idx0);
      const float s01 = src(yl.idx0, xl.idx1);
      const float s10 = src(yl.idx1, xl.idx0);
      const float s11 = src(yl.idx1, xl.idx1);
      const float w00 = yl.w0 * xl.w0;
      const float w01 = yl.w0 * xl.w1;
      const float w10 = yl.w1 * xl.w0;
      const float w11 = yl.w1 * xl.w1;
      double num = 0.0, den = 0.0;
      if (w00 > 0.0f && finite_f(s00)) { num += w00 * s00; den += w00; }
      if (w01 > 0.0f && finite_f(s01)) { num += w01 * s01; den += w01; }
      if (w10 > 0.0f && finite_f(s10)) { num += w10 * s10; den += w10; }
      if (w11 > 0.0f && finite_f(s11)) { num += w11 * s11; den += w11; }
      if (den <= 0.0) continue;
      const float val = static_cast<float>(num / den);
      out(y, x) = (finite_f(val) && val > 0.0f) ? val : nan_value();
    }
  }
  return out;
}

float finite_median(const Matrix2Df &m) {
  std::vector<float> v;
  v.reserve(static_cast<size_t>(m.size()));
  for (int i = 0; i < m.size(); ++i)
    if (finite_f(m.data()[i])) v.push_back(m.data()[i]);
  if (v.empty()) return nan_value();
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  float med = v[mid];
  if (v.size() % 2 == 0)
    med = 0.5f * (med + *std::max_element(v.begin(), v.begin() + mid));
  return med;
}

}  // namespace

SourceQualityMapResult compute_source_quality_maps(
    const Matrix2Df &analysis_proxy,
    const std::vector<uint8_t> &source_valid_mask, int source_width,
    int source_height, const config::AqmhPyramidConfig &cfg,
    const QualityScaleMapSink &sink) {
  // A silently all-zero q_map would be a veto for every pixel downstream, so a
  // geometry disagreement is a hard error, not a quiet empty result.
  if (source_width <= 0 || source_height <= 0 ||
      analysis_proxy.rows() != source_height ||
      analysis_proxy.cols() != source_width) {
    throw std::invalid_argument("SOURCE_QUALITY_MAPS_GEOMETRY_MISMATCH");
  }

  SourceQualityMapResult result;
  result.q_map = Matrix2Df::Zero(source_height, source_width);
  result.artifact_confidence = Matrix2Df::Constant(source_height, source_width,
                                                   nan_value());

  int live_maps = 0;
  int peak_live_maps = 0;
  int observed_scales = 0;
  bool captured_artifact = false;

  metrics::PerScaleQualityHook hook =
      [&](int scale_index, int downsample_factor, const Matrix2Df &psi,
          const Matrix2Df &artifact) {
        ++observed_scales;
        // Upsample this scale to source geometry. This map plus any retained
        // in result.scale_maps are the only full source-geometry maps held
        // here; the geometric-mean composite lives inside
        // compute_aqmh_quality_map as a double log-sum accumulator.
        Matrix2Df psi_src = upsample_to_source(psi, source_width, source_height,
                                               downsample_factor);
        ++live_maps;
        peak_live_maps = std::max(peak_live_maps, live_maps);

        if (!captured_artifact) {
          // Finest computed scale supplies artifact_confidence (least
          // smoothed; native-sampling artefact response). 1 = clean.
          //
          // Plan 14.4: unlike the legacy diagnostic path, insufficient local
          // support must be marked NOT APPLICABLE, not full confidence.
          // Legacy phi_artifact returns exactly 1.0f where its window has < 3
          // valid high-pass samples; those pixels coincide with where psi is
          // NaN (psi needs finite z_sharp/z_snr/artifact, i.e. real support).
          // So mask the artifact map to NaN wherever the finest psi is NaN
          // before upsampling --- a conservative "not applicable" that keeps
          // genuinely clean, well-supported pixels at their true value.
          Matrix2Df artifact_masked = artifact;
          for (int i = 0; i < artifact_masked.size(); ++i)
            if (!finite_f(psi.data()[i]))
              artifact_masked.data()[i] = nan_value();
          result.artifact_confidence = upsample_to_source(
              artifact_masked, source_width, source_height, downsample_factor);
          captured_artifact = true;
        }

        if (sink) {
          sink(scale_index, downsample_factor, psi_src);
        } else {
          ScaleQualityMap m;
          m.scale_index = scale_index;
          m.downsample_factor = downsample_factor;
          m.psi = std::move(psi_src);
          result.scale_maps.push_back(std::move(m));
        }
        --live_maps;
      };

  const auto legacy = metrics::compute_aqmh_quality_map(
      analysis_proxy, source_valid_mask, /*frame_valid_mask=*/{}, source_width,
      source_height, cfg, ::tile_compile::core::AccelerationBackend::cpu,
      /*stream=*/nullptr, hook);

  result.q_map = legacy.q_map;
  result.diagnostics.omitted_scales = legacy.diagnostics.omitted_scales;
  result.diagnostics.computed_scales = observed_scales;
  result.diagnostics.peak_resident_scale_maps =
      sink ? peak_live_maps : static_cast<int>(result.scale_maps.size());
  result.diagnostics.composite_p50 = finite_median(result.q_map);
  return result;
}

}  // namespace tile_compile::reconstruction
