#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace tile_compile::metrics { class QualityMapCache; }

namespace tile_compile::reconstruction {

struct AqmhReconstructionConfig {
  float clip_sigma = 3.0f;
  int clip_iterations = 3;
  float min_fraction = 0.5f;
  float min_n_eff = 2.0f;
  bool cherry_pick = false;
  bool uniform_weights = false;
  bool compute_uniform_control = false;
  float cherry_pick_k_frac = 0.30f;
  int cherry_pick_k_min_required = 20;
  float cherry_pick_margin_min = 0.02f;
  std::vector<config::AqmhCherryPickConfig::Tier> tiered_k_frac;
  int parallel_workers = 1;
  size_t memory_budget_mb = 2048;
  int chunk_rows = 0;           // 0 = auto
};

struct AqmhReconstructionResult {
  Matrix2Df output;
  Matrix2Df weight_sum;
  Matrix2Df uniform_control_output;
  bool acceleration_used = false;
  bool acceleration_fallback = false;
  uint64_t unsupported_pixels = 0;
  uint64_t finite_map_samples = 0;
  uint64_t missing_map_samples = 0;
  uint64_t zero_veto_pixels = 0;
  Matrix2Df cherry_pick_k_map;
  float cherry_pick_active_frac = 0.0f;
  float cherry_pick_mean_k = 0.0f;
  float cherry_pick_median_k = 0.0f;
  int cherry_pick_k_min_observed = 0;
  int cherry_pick_k_max_observed = 0;
  bool cherry_pick_per_pixel_mode = false;
  bool cherry_pick_forced_disabled = false;
  bool cherry_pick_active = false;
  float k_nominal_median = 0.0f;
  float k_effective_p10 = 0.0f;
  float k_effective_p50 = 0.0f;
  float k_effective_p90 = 0.0f;
  bool low_rank_separation = false;
  uint64_t numerical_guard_pixels = 0;
  int chunk_rows = 0;
  int chunk_count = 0;
  bool region_streaming_used = false;
};

using AqmhFrameLoader = std::function<bool(size_t, Matrix2Df&)>;
using AqmhMaskLoader = std::function<bool(size_t, std::vector<uint8_t>&)>;
using AqmhFrameRegionLoader =
    std::function<bool(size_t, int, int, Matrix2Df&)>;
using AqmhMaskRegionLoader =
    std::function<bool(size_t, int, int, std::vector<uint8_t>&)>;
using AqmhProgressCallback = std::function<void(int, int)>;

AqmhReconstructionResult reconstruct_aqmh_weighted(
    size_t frame_count, const AqmhFrameLoader &load_frame,
    metrics::QualityMapCache *q_map_cache, const VectorXf &global_weights,
    const std::vector<uint8_t> &canvas_mask, int width, int height,
    const AqmhReconstructionConfig &cfg,
    const AqmhMaskLoader &load_frame_valid_mask = {},
    const AqmhFrameRegionLoader &load_frame_region = {},
    const AqmhMaskRegionLoader &load_frame_valid_mask_region = {},
    const AqmhProgressCallback &progress = {});

} // namespace tile_compile::reconstruction
