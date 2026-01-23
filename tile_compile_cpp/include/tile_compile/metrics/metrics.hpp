#pragma once

#include "tile_compile/core/types.hpp"
#include <vector>

namespace tile_compile::metrics {

FrameMetrics calculate_frame_metrics(const Matrix2Df& frame);

VectorXf calculate_global_weights(const std::vector<FrameMetrics>& metrics,
                                 float w_bg, float w_noise, float w_grad,
                                 float clamp_lo = -3.0f, float clamp_hi = 3.0f);

} // namespace tile_compile::metrics
