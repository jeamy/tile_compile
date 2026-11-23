#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tile_compile::reconstruction {

struct ChromaDenoiseStats {
  bool applied = false;
  bool guided_applied = false;
  std::uint64_t valid_pixels = 0;
  double input_chroma_sigma = 0.0;
  double measured_chroma_to_luma_ratio = 0.0;
  // Ratio independent channel noise would give, and measured / expected
  // (1 = independent channel noise; mode=auto scales on this excess).
  double expected_chroma_to_luma_ratio = 0.0;
  double chroma_noise_excess = 0.0;
  double auto_mode_scale = 1.0;
  std::string skip_reason;
  double adaptation = 1.0;
  double effective_blend_amount = 0.0;
  double star_protected_fraction = 0.0;
  double structure_protected_fraction = 0.0;
  double extended_source_raw_fraction = 0.0;
  double extended_source_protected_fraction = 0.0;
  double combined_protected_fraction = 0.0;
  double mean_protection = 0.0;
  double mean_denoise_fraction = 0.0;
  double extended_source_sky_median = 0.0;
  double extended_source_sky_sigma = 0.0;
  double extended_source_threshold = 0.0;
  // Large-scale (block-median-surface) chroma bias removal, see
  // config::ChromaDenoiseConfig::LargeScaleBiasConfig.
  double input_luma_sigma = 0.0;
  double large_scale_bias_removed_rms_c1 = 0.0;
  double large_scale_bias_removed_rms_c2 = 0.0;
  double large_scale_bias_grid_holes_filled_fraction = 0.0;
  // Where the excess was measured: "image" or "pre_denoise"
  // (ChromaDenoiseOptions::excess_r/g/b, chroma_denoise.excess_from_pre_denoise).
  std::string excess_source = "image";
  // Luminance kept unchanged: "color_space" or "hms_profile" (the weights
  // below, ChromaDenoiseOptions::preserve_weights).
  std::string luminance_preserved = "color_space";
  std::array<float, 3> luminance_weights{0.0f, 0.0f, 0.0f};
};

// Inputs of a chroma pass that come from outside its config.
struct ChromaDenoiseOptions {
  // Weights of the luminance to keep for preserve_luminance=hms_profile (the
  // hypermetric_stretch sensor profile's); unused unless have_preserve_weights.
  bool have_preserve_weights = false;
  std::array<float, 3> preserve_weights{0.0f, 0.0f, 0.0f};
  // The image as it would be without the earlier denoise passes, same size as
  // the planes; the chroma noise excess is measured on it instead when all
  // three are set (chroma_denoise.excess_from_pre_denoise).
  const Matrix2Df* excess_r = nullptr;
  const Matrix2Df* excess_g = nullptr;
  const Matrix2Df* excess_b = nullptr;
};

struct ChromaDenoiseDiagnostics {
  Matrix2Df star_mask;
  Matrix2Df structure_mask;
  Matrix2Df extended_source_mask;
  Matrix2Df combined_mask;
  Matrix2Df effective_amount;
};

// Applies the configured chroma denoise to the RGB planes in place. Shared by
// the downstream post-stack output path (apply_stage == "post_pcc").
// `protection_mask_out`, when non-null, is resized to r's shape and filled
// with the combined protection mask (star_protection/structure_protection/
// extended_source_protection, feathered and combined, in [0,1]) used for
// this call -- the same mask that scales denoise strength via
// luma_guard_strength. Diagnostic only; callers may write it out (e.g. as a
// FITS file) to inspect where and how strongly denoise was held back.
ChromaDenoiseStats chroma_denoise_rgb_inplace(
    Matrix2Df& r, Matrix2Df& g, Matrix2Df& b,
    const config::ChromaDenoiseConfig& cfg,
    Matrix2Df* protection_mask_out = nullptr,
    const std::vector<std::uint8_t>* valid_mask = nullptr,
    ChromaDenoiseDiagnostics* diagnostics = nullptr,
    const ChromaDenoiseOptions* options = nullptr);

// Takes earlier denoise passes out of an image further down the pipeline:
// subtracts J * delta per pixel, where delta is what the passes changed
// (after - before; one plane when they changed R, G and B by the same amount,
// as luma_denoise does, else three) and J the linear part of every transform
// in between (BGE subtracts per channel: identity; PCC: its applied linear
// part). Offsets in between drop out. Only on pixels inside output_mask (when
// it matches) that are not all zero (the dead canvas PCC leaves alone).
// False, and nothing changed, when a delta plane does not match the image.
bool remove_denoise_delta_inplace(Matrix2Df& r, Matrix2Df& g, Matrix2Df& b,
                                  const std::vector<Matrix2Df>& delta,
                                  const std::array<std::array<double, 3>, 3>& J,
                                  const std::vector<std::uint8_t>* output_mask = nullptr);

} // namespace tile_compile::reconstruction
