#pragma once

// CFA-aware source-space quality analysis proxy --- milestone M3 of the
// CFA-forward-drizzle plan (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_
// implementierungsplan_de.md, section 13.2 "CFA-aware Analyseproxy").
//
// M3 minimal scope (plan 23 M3): only proxy_version=1's Quad-Green /
// Green-highpass-MAD analysis proxy and its global (frame-level) noise
// statistic. Scale-specific quality maps, the region-read cache API and the
// sink/callback streaming contract (plan 13.3-13.5) are M5 scope, not
// implemented here.
//
// This computes an ANALYSIS proxy only --- it never modifies the
// reconstruction signal itself (plan 13.2: "ohne das spätere Nutzsignal zu
// verändern").

#include "tile_compile/core/types.hpp"

namespace tile_compile::reconstruction {

struct SourceQualityProxyResult {
  // Quad-Green grid (plan 13.2 step 1): one value per 2x2 Bayer quad,
  // G_quad = 0.5*(G1+G2), half resolution in each dimension. Empty for MONO
  // (proxy_full carries the analysis signal directly there instead).
  Matrix2Df quad_green;
  int quad_width = 0;
  int quad_height = 0;

  // Global (frame-level) MAD-based noise scale (plan 13.2 step 2):
  // hp = quad_green - B3_blur(quad_green) for OSC, or
  // hp = proxy_full - B3_blur(proxy_full) for MONO (no CFA interpolation);
  // sigma_green = 1.4826 * median(|hp - median(hp)|).
  double sigma_green = 0.0;

  // Full source-resolution, edge-aware-interpolated green proxy (plan 13.2
  // step 3): native green sample at G positions, and at R/B positions the
  // horizontal or vertical neighbour-pair average with the lower local
  // gradient (avoids interpolating across an edge). For positional analysis
  // functions only --- never the reconstruction signal. For MONO this is
  // simply the input L plane, unchanged.
  Matrix2Df proxy_full;
};

// proxy_version=1 (plan 13.2). `source` is the normalized CFA frame (OSC) or
// normalized L plane (MONO) --- the existing normalized-cache source (plan
// 10.1), never a prewarped image.
SourceQualityProxyResult compute_source_quality_proxy_v1(
    const Matrix2Df &source, ColorMode color_mode, BayerPattern bayer_pattern,
    int cfa_origin_x, int cfa_origin_y);

// Exposed for unit tests: separable B3-spline ([1,4,6,4,1]/16) blur with
// edge-clamped boundary handling, plan 13.2 step 2's B3_blur().
Matrix2Df b3_spline_blur(const Matrix2Df &input);

// Exposed for unit tests: 1.4826 * median(|x - median(x)|) over all finite
// values of `values` (plan 13.2 step 2's exact MAD-based sigma formula,
// deliberately not the local-mean approximation used elsewhere in the
// codebase for the legacy prewarped Q-map path).
double median_absolute_deviation_sigma(const Matrix2Df &values);

}  // namespace tile_compile::reconstruction
