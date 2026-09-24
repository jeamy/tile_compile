#pragma once

// Local energy guard (plan 14.5) and support-aware alpha smoothing
// (plan 14.7) for the controlled multi-band reconstruction --- milestone M6.
//
// Energy guard: per band, on the fixed working luminance, with NO star
// concentration exception in the first release. window_radius_j =
// max(3, 2^(j+1)) internal px.
//   scale_raw            = max(MAD_window(D_raw,luma,j), background_band_floor_j)
//   energy_ratio(alpha)  = MAD_window(D_mixed,luma,j(alpha)) / scale_raw
//   D_mixed,luma,j(alpha) = D_R,luma,j + alpha * (D_profile,luma,j - D_R,luma,j)
// The binding limit is energy_ratio <= 1.30. If alpha_pre exceeds it, a
// deterministic six-iteration bisection finds the largest alpha_guarded in
// [0, alpha_pre] that satisfies the limit. Pixel values and Raw bands are
// never hard-clipped. Fewer than 25 valid window pixels => alpha = 0 for that
// band pixel.
//
// Alpha smoothing (plan 14.7): after the guard, alpha is blurred with the
// separable B3 kernel [1,4,6,4,1]/16 ONLY within its own 4-connected support
// component, then min-capped:
//   alpha_blur  = conv(alpha_guarded * support, B3) / conv(support, B3)
//   alpha_final = min(alpha_guarded, alpha_blur)
// The min cap is mandatory: smoothing may only reduce local evidence, never
// lift alpha into an uncertain or vetoed pixel. alpha_guarded == 0 stays
// exactly 0.

#include <cstdint>
#include <vector>

namespace tile_compile::reconstruction {

struct EnergyGuardParams {
  double energy_limit = 1.30;   // binding M6 value (plan 14.5)
  int bisection_iters = 6;
  int min_window_pixels = 25;
};

// window_radius_j for band level j (1-based): max(3, 2^(j+1)).
int energy_guard_window_radius(int level);

// `d_r_luma` / `d_profile_luma` are the per-pixel luminance-combined detail
// maps for one band (row-major, size width*height). `support` marks pixels
// where BOTH are finite/valid. `background_floor` is background_band_floor_j
// (>= 0). Returns alpha_guarded (size width*height); unsupported or
// under-supported (< min_window_pixels) pixels get 0.
std::vector<float> apply_energy_guard(const std::vector<float> &alpha_pre,
                                      const std::vector<float> &d_r_luma,
                                      const std::vector<float> &d_profile_luma,
                                      const std::vector<uint8_t> &support,
                                      int width, int height, int window_radius,
                                      double background_floor,
                                      const EnergyGuardParams &params = {});

// Support-aware B3 smoothing with the mandatory min cap (plan 14.7).
std::vector<float> smooth_alpha_b3(const std::vector<float> &alpha_guarded,
                                   const std::vector<uint8_t> &support,
                                   int width, int height);

// Exposed for tests: 1.4826 * median(|x - median(x)|) over the given values.
double mad_sigma(std::vector<float> values);

}  // namespace tile_compile::reconstruction
