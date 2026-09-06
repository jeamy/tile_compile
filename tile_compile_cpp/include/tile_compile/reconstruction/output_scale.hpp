#pragma once

// Internal 2x raster -> output geometry (milestone M4, plan section 12).
//
// After candidate selection the reconstruction may live on an internal 2x
// raster (internal_scale = 2). This module converts it to the requested
// output_scale:
//   * 2/1 (production default): one deterministic 2x2 area-average onto
//     image, masks and weight planes, with the strict 4/4-support contract
//     from plan 12.1;
//   * 2/2 and 1/1: identity (no resample).
// Auto selection is deliberately absent (plan: "explizite Modi ... ohne
// Auto").
//
// It also carries plan 12.2's WCS scaling so masks, RGB and WCS never pick
// up different rounding rules.

#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"

namespace tile_compile::reconstruction {

// Plan 12.1's deterministic 2x2 -> 1x area-average:
//   valid_out = valid_00 && valid_01 && valid_10 && valid_11
//   value_out = 0.25 * (v_00 + v_01 + v_10 + v_11)
//   n_eff_out = min(n_eff_00, n_eff_01, n_eff_10, n_eff_11)
// An invalid subpixel never enters as 0 or a partially-normalized mean ---
// the 1x pixel goes invalid. weight_sum is area-averaged the same way as
// value (a documented convention; the plan pins only value and n_eff). An
// odd internal dimension drops its last row/column (same rule the Quad-Green
// grid uses), so the operator is always exact 2x2.
ProfilePlane downsample_profile_plane_2x2(const ProfilePlane &in);

// Applies the 2x2 operator to every present plane of a uniform+raw result
// and returns a result whose planes are at output (1x) resolution.
ForwardDrizzleUniformAndRawResult
downsample_uniform_and_raw_2x2(const ForwardDrizzleUniformAndRawResult &in);

// Memory-bounded 2/1: runs stream_forward_drizzle_uniform_and_raw() and
// emits stripes already downsampled 2x2 -> 1x, so a caller (e.g. the
// transactional profile store) never holds a full internal-resolution
// image. A small row buffer (at most one internal row of carry) turns the
// internal-row stripes into output-row stripes; the result is bit-identical
// to compute_forward_drizzle_uniform_and_raw() followed by
// downsample_uniform_and_raw_2x2(), independent of the internal chunk
// height. Requires drizzle_cfg.internal_scale == 2 (internal dimensions are
// canvas_native*2, always even, so no leftover row). When `multiband` requests
// fine/medium/alpha-confidence, those are carried too: fine/medium by the same
// 2x2 area-average, the channel-min confidence maps by 2x2 min + AND support.
ForwardDrizzlePairDiagnostics stream_forward_drizzle_uniform_and_raw_2x2(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const UniformAndRawStripeSink &output_sink,
    const ForwardDrizzleSubdivisionParams &subdivision_params = {},
    const std::vector<float> &g_eff_by_source_index = {}, size_t retained_bytes = 0,
    const FrameQualityProvider &quality_of = {},
    const MultibandProfileParams &multiband = {});

struct OutputScaleMode {
  int internal_scale = 2;
  int output_scale = 1;  // {1, 2}, <= internal_scale
  bool valid() const {
    return (internal_scale == 1 || internal_scale == 2) &&
           (output_scale == 1 || output_scale == 2) && output_scale <= internal_scale;
  }
  bool needs_2x2_downsample() const { return internal_scale == 2 && output_scale == 1; }
};

struct OutputWcsParams {
  // Shift of the original reference geometry into the native canvas
  // (RegistrationSamplingPlan::canvas_offset_{x,y}_native), in native pixels.
  double canvas_offset_x_native = 0.0;
  double canvas_offset_y_native = 0.0;
  // Removed top/left crop edge, expressed in OUTPUT pixels. For a crop
  // determined in native pixels: crop_origin_out = output_scale * crop_origin_native.
  double crop_origin_x_out = 0.0;
  double crop_origin_y_out = 0.0;
  int output_scale = 1;  // S
};

// --- Plan 12.4: kernel-induced noise correlation ---------------------------
//
// Forward drizzle with a square top-hat droplet of side d = pixfrac *
// internal_scale (internal pixels) spreads each input sample over several
// output pixels, so adjacent output pixels are no longer statistically
// independent. Downstream noise estimators that assume independent per-pixel
// noise (BGE RMS, PCC star SNR, HMS noise, any SNR map) must scale a naive
// per-pixel sigma by this factor to recover the true large-aperture noise.
//
// Exact, from first principles (no fitted constant): for white input noise
// the variance of a large-aperture sum is conserved (= sigma_in^2 * N_out),
// while the naive independent estimate is N_out * sigma_in^2 * S0 / W^2,
// where W = d and S0 = sum_j overlap([j+0.5 - d/2, j+0.5 + d/2], [0,1])^2.
// The sigma correction factor is therefore
//     f = W / sqrt(S0)   (>= 1; exactly 1 when d = 1, i.e. no correlation).
//
// This is the DENSE-input reference case (one input sample per internal
// pixel). The per-channel forward drizzle samples a given colour sparsely
// (R/B every 2 native pixels), which correlates output noise MORE; that
// per-channel refinement is separate and not yet computed here.
double kernel_noise_correlation_sigma_factor(float pixfrac, int internal_scale);

// The 1D output-lag autocorrelation coefficients rho_0..rho_{max_lag}
// (rho_0 == 1). rho_delta = S_delta / S0. Diagnostic; the scalar factor
// above is sqrt(sum over all lags of rho_delta).
std::vector<double> kernel_noise_autocorrelation_1d(float pixfrac, int internal_scale,
                                                    int max_lag = 6);

// Plan 12.2, component-wise, exactly as written:
//   CRPIX_canvas_native = CRPIX_in + canvas_offset_native
//   CRPIX_out = S * (CRPIX_canvas_native - 0.5) + 0.5 - crop_origin_out
//   CD_out    = CD_in / S
// (This is the standard FITS rebin form CRPIX_out = S*CRPIX_in - (S-1)/2
// plus the canvas-offset and explicit-sign crop term.) `naxis*` are left
// unchanged --- the caller sets them from the actual output image size.
astrometry::WCS scale_wcs_to_output(const astrometry::WCS &in, const OutputWcsParams &p);

}  // namespace tile_compile::reconstruction
