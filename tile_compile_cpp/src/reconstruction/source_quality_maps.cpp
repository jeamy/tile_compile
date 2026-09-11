#include "tile_compile/reconstruction/source_quality_maps.hpp"

#include "tile_compile/metrics/aqmh_quality_map.hpp"
#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

namespace tile_compile::reconstruction {
namespace {

using tile_compile::core::nan_value;

bool finite_f(float v) {
  return (std::bit_cast<uint32_t>(v) & 0x7f800000u) != 0x7f800000u;
}

float finite_median(const Matrix2Df &m) {
  std::vector<float> v;
  v.reserve(static_cast<size_t>(m.size()));
  for (int i = 0; i < m.size(); ++i)
    if (finite_f(m.data()[i])) v.push_back(m.data()[i]);
  return ::tile_compile::core::median_of_or_nan(std::move(v));
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
          const Matrix2Df &psi_src_in, const Matrix2Df &artifact) {
        ++observed_scales;
        // psi_src_in is `psi` already bilinearly upsampled to source geometry
        // by compute_aqmh_quality_map (plan: avoid running that identical
        // interpolation a second time here). This map plus any retained in
        // result.scale_maps are the only full source-geometry maps held here;
        // the geometric-mean composite lives inside compute_aqmh_quality_map
        // as a double log-sum accumulator.
        Matrix2Df psi_src = psi_src_in;
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
          result.artifact_confidence = metrics::upsample_scale_to_source(
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
