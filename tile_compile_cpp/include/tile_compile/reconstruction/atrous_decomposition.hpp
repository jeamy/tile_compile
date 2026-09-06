#pragma once

// Masked, support-propagating a-trous (starlet) decomposition --- milestone M6
// of the CFA-forward-drizzle plan (docs/AQMH/aqmh_cfa_forward_drizzle_
// multiband_implementierungsplan_de.md, section 14.2).
//
// Shift-invariant a-trous transform with the separable B3-spline kernel
// h = [1, 4, 6, 4, 1] / 16. For level j (1-based) 2^(j-1)-1 zeros are inserted
// between kernel taps, so level 1 is undilated, level 2 inserts one zero, etc.
// Masked convolution propagates a level-specific support (plan 14.2):
//
//   den_j = convolve(M_(j-1), h_j)                       (2D, separable)
//   C_j   = convolve(C_(j-1) * M_(j-1), h_j) / den_j
//   M_j   = M_(j-1) && (den_j >= den_min)
//   D_j   = C_(j-1) - C_j, valid only on M_(j-1) && M_j
//
// The fully-supported separable kernel sums to 1.0 in each direction, so the
// fully-supported 2D weight is exactly 1.0 for EVERY level (the dilated kernel
// still sums to 1). den_min is therefore a fixed fraction of 1.0, identical
// across levels. It is a versioned constant in the multiband hash domain ---
// deliberately NOT a config field, so it cannot be tuned without a
// multiband_config_hash bump (plan 14.2).

#include <cstdint>
#include <vector>

namespace tile_compile::reconstruction {

// Minimum fraction of the (unit) fully-supported kernel weight required for a
// pixel to stay valid at the next level. Part of the multiband hash domain.
inline constexpr double kAtrousDenMinFraction = 0.5;

// Version tag for the a-trous math (kernel, dilation rule, den_min, boundary
// = masked renormalisation with no clamp). Bump on any change.
inline constexpr int kAtrousDecompositionVersion = 1;

struct AtrousBand {
  int level = 0;                    // 1-based: D1 .. D(levels)
  std::vector<float> detail;        // D_j; NaN off (M_(j-1) && M_j)
  std::vector<uint8_t> support;     // M_(j-1) && M_j
};

struct AtrousDecomposition {
  int width = 0;
  int height = 0;
  int levels = 0;
  std::vector<AtrousBand> bands;         // size == levels
  std::vector<float> coarse;             // C_levels; NaN off M_levels
  std::vector<uint8_t> coarse_support;   // M_levels
};

// `value` is one profile-plane channel in row-major order, NaN where
// unsupported; `mask` is that channel's support (1 = valid, size w*h; empty =
// all valid). `levels` in [1, 4]. Boundary handling is masked renormalisation
// only --- out-of-image taps contribute nothing (as if mask 0), never a
// clamped edge value (plan 14.7 "keine Faltung ueber ungueltige
// Canvasbereiche").
AtrousDecomposition atrous_decompose(const std::vector<float> &value,
                                     const std::vector<uint8_t> &mask,
                                     int width, int height, int levels);

// Max |original - (C_levels + sum_j D_j)| over the tightest common valid
// support (M_levels). The plan (14.2) checks the reconstruction identity only
// on the common valid support of the profile.
double atrous_reconstruction_max_error(const AtrousDecomposition &d,
                                       const std::vector<float> &original);

}  // namespace tile_compile::reconstruction
