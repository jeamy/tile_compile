#pragma once

#include "tile_compile/core/types.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <vector>

namespace tile_compile::reconstruction {

// Descriptive paired diagnostic between the selected linear reconstruction and the final
// stretched G plane. Stars are detected only in `before`; both widths use the same coordinates,
// common validity mask, patch size, and PSF estimator. This is not a quality gate.
nlohmann::json measure_post_run_star_shape(const Matrix2Df& before,
                                          const Matrix2Df& after,
                                          const std::vector<uint8_t>& common_valid_mask);

}  // namespace tile_compile::reconstruction
