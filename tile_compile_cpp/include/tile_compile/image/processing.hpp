#pragma once

#include "tile_compile/core/types.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace tile_compile::image {

// OSC working luma weights (matches
// reconstruction::kWorkingLumaWeightsOsc = {0.25, 0.5, 0.25}). Was
// hardcoded as `0.25f * R + 0.5f * G + 0.25f * B` at 7 call sites across
// runner_pipeline.cpp/runner_resume.cpp/runner_phase_metrics.cpp.
inline float rgb_to_luma(float r, float g, float b) {
  return 0.25f * r + 0.5f * g + 0.25f * b;
}
inline Matrix2Df rgb_to_luma(const Matrix2Df& r, const Matrix2Df& g,
                             const Matrix2Df& b) {
  return 0.25f * r + 0.5f * g + 0.25f * b;
}

std::map<std::string, Matrix2Df> split_cfa_channels(const Matrix2Df& mosaic, BayerPattern pattern);

Matrix2Df reassemble_cfa_mosaic(const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B,
                                BayerPattern pattern);

Matrix2Df normalize_frame(const Matrix2Df& frame, float target_background,
                          float target_scale, NormalizationMode mode);

Matrix2Df cosmetic_correction(const Matrix2Df& frame, float sigma_threshold, bool correct_hot);

Matrix2Df cosmetic_correction_cfa(const Matrix2Df& mosaic, float sigma_threshold,
                                  bool correct_hot, int origin_x, int origin_y);

struct ChromaSpeckleSuppressionStats {
  int candidate_pixels = 0;
  int corrected_pixels = 0;
};

ChromaSpeckleSuppressionStats suppress_isolated_chroma_speckles_rgb_inplace(
    Matrix2Df& R, Matrix2Df& G, Matrix2Df& B,
    const std::vector<uint8_t>* valid_mask = nullptr,
    int mask_rows = 0, int mask_cols = 0);

// Extract a tile sub-region from an image, clamped to image bounds.
Matrix2Df extract_tile(const Matrix2Df& img, const Tile& t);

} // namespace tile_compile::image
