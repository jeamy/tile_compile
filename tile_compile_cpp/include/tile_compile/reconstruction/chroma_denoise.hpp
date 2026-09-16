#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

namespace tile_compile::reconstruction {

// Applies the configured chroma denoise to the RGB planes in place. Shared by
// the downstream post-stack output path (apply_stage == "post_pcc").
void chroma_denoise_rgb_inplace(Matrix2Df& r, Matrix2Df& g, Matrix2Df& b,
                                const config::ChromaDenoiseConfig& cfg);

} // namespace tile_compile::reconstruction
