#pragma once

// Scale-specific source-space quality maps --- milestone M5 of the
// CFA-forward-drizzle plan (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_
// implementierungsplan_de.md, sections 13.1-13.5).
//
// M5 job (plan 13.1): the legacy compute_aqmh_quality_map() collapses every
// pyramid scale into one geometric-mean Q-map on a PREWARPED frame. The
// multiband reconstruction (section 14) needs the per-scale maps kept
// separate AND expressed in SOURCE coordinates. This module reuses the
// legacy per-scale math verbatim (via the PerScaleQualityHook added to
// metrics::compute_aqmh_quality_map) and only changes what is exposed:
//   - q_map            : the geometric-mean composite, byte-identical to the
//                        legacy path for MONO + all-valid mask (equivalence
//                        test), source geometry;
//   - scale_maps / sink: one psi map per computed scale, upsampled to source
//                        geometry, streamed one at a time so no more than a
//                        single full map plus the running composite is
//                        resident (plan 13.3);
//   - artifact_confidence : phi_artifact at the finest computed scale,
//                        upsampled to source geometry (1 = clean).
//
// The a-trous decomposition, A_artifact / alpha machinery and the multiband
// mix are M6 scope and are NOT implemented here.

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <functional>
#include <limits>
#include <vector>

namespace tile_compile::reconstruction {

struct ScaleQualityMap {
  int scale_index = 0;
  // Legacy pyramid convention: 1 << (2*scale_index), relative to the
  // source-resolution analysis proxy.
  int downsample_factor = 1;
  // Quality map in SOURCE geometry, values in [0,1], NaN where unsupported.
  Matrix2Df psi;
};

struct SourceQualityMapDiagnostics {
  int computed_scales = 0;
  std::vector<int> omitted_scales;
  float composite_p50 = std::numeric_limits<float>::quiet_NaN();
  // Largest number of full source-geometry scale maps held simultaneously by
  // compute_source_quality_maps (excluding the composite accumulator and the
  // returned q_map). With a sink this must stay 1; without a sink it equals
  // computed_scales because scale_maps retains them.
  int peak_resident_scale_maps = 0;
};

struct SourceQualityMapResult {
  Matrix2Df q_map;                        // composite, source geometry
  std::vector<ScaleQualityMap> scale_maps; // populated only when sink is null
  Matrix2Df artifact_confidence;          // source geometry, 1 = clean
  SourceQualityMapDiagnostics diagnostics;
};

// Called once per computed scale with the psi map already upsampled to source
// geometry. The callee is expected to persist and release it immediately.
using QualityScaleMapSink = std::function<void(
    int scale_index, int downsample_factor, const Matrix2Df &psi_source_geom)>;

// `analysis_proxy` is the source-resolution analysis proxy: for MONO the
// normalized L plane unchanged, for OSC the edge-aware full-resolution green
// proxy (compute_source_quality_proxy_v1(...).proxy_full). It must NOT be a
// prewarped image and must NEVER be the reconstruction signal.
//
// `source_valid_mask` is an optional row-major source-geometry mask (empty =
// all valid). Non-finite proxy samples are treated as unsupported regardless.
//
// When `sink` is non-null the per-scale maps are streamed to it and
// `result.scale_maps` is left empty; when null they are returned in
// `result.scale_maps`.
SourceQualityMapResult compute_source_quality_maps(
    const Matrix2Df &analysis_proxy,
    const std::vector<uint8_t> &source_valid_mask, int source_width,
    int source_height, const config::AqmhPyramidConfig &cfg,
    const QualityScaleMapSink &sink = {});

}  // namespace tile_compile::reconstruction
