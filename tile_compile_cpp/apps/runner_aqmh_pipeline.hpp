#pragma once

#include "runner_shared.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/reconstruction/aqmh_reconstruction.hpp"
#include "tile_compile/reconstruction/aqmh_validation.hpp"

#include <optional>

#include <chrono>
#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::metrics { class QualityMapCache; }

namespace tile_compile::reconstruction {
class AqmhPrefetchCoordinator;
}

namespace tile_compile::runner {

struct AqmhPipelineContext {
  std::unique_ptr<metrics::QualityMapCache> aqmh_cache;
  VectorXf aqmh_global_weights;
  std::optional<reconstruction::AqmhValidationComparison> control_validation;
  reconstruction::AqmhReconstructionResult recon_result;
  Matrix2Df recon_output;
  Matrix2Df weight_sum;
  std::unique_ptr<reconstruction::AqmhPrefetchCoordinator> prefetch_coordinator;
};

bool run_aqmh_phases(
    const std::string &run_id, const config::Config &cfg,
    const std::filesystem::path &run_dir,
    const std::vector<std::filesystem::path> &frames,
    const std::vector<uint8_t> &frame_has_data,
    const std::vector<uint8_t> &common_valid_mask,
    int canvas_width, int canvas_height, bool osc_mode,
    const DiskCacheFrameStore &prewarped_frames,
    const std::vector<image::NormalizationScales> &norm_scales,
    ColorMode detected_mode, const std::string &detected_bayer_str,
    core::AccelerationContext &acceleration,
    core::EventEmitter &emitter, std::ostream &log_file,
    AqmhPipelineContext &out);

} // namespace tile_compile::runner
