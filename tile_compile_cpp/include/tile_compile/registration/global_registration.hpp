#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <string>
#include <vector>

namespace tile_compile::registration {

struct GlobalRegistrationOutput {
    int ref_idx = 0;
    std::string ref_selection_method; // "global_weight" | "quality_score" | "middle"
    float ref_selection_value = 0.0f;

    float downsample_scale = 1.0f;
    std::string engine_used;
    std::vector<WarpMatrix> warps_fullres;
    std::vector<float> scores;
    std::vector<bool> success;
    std::vector<std::string> errors;
};

GlobalRegistrationOutput register_frames_to_reference(
    const std::vector<Matrix2Df>& frames_fullres,
    ColorMode mode,
    BayerPattern bayer,
    const config::RegistrationConfig& rcfg,
    const std::vector<FrameMetrics>* frame_metrics_opt = nullptr,
    const VectorXf* global_weights_opt = nullptr
);

} // namespace tile_compile::registration
