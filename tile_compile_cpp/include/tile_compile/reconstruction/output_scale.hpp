#pragma once

// Output-geometry contract for the internal 2x raster (plan section 12).
//
// The reconstruction may live on an internal 2x raster (internal_scale = 2);
// `OutputScaleMode` expresses the requested output_scale:
//   * 2/1: deterministic 2x2 area-average onto image, masks and weight
//     planes, applied inside the forward drizzle v2 production path;
//   * 2/2 and 1/1: identity (no resample).
// Auto selection is deliberately absent (plan: "explizite Modi ... ohne
// Auto").

#include <vector>

namespace tile_compile::reconstruction {

struct OutputScaleMode {
  int internal_scale = 2;
  int output_scale = 1;  // {1, 2}, <= internal_scale
  bool valid() const {
    return (internal_scale == 1 || internal_scale == 2) &&
           (output_scale == 1 || output_scale == 2) && output_scale <= internal_scale;
  }
  bool needs_2x2_downsample() const { return internal_scale == 2 && output_scale == 1; }
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

}  // namespace tile_compile::reconstruction
