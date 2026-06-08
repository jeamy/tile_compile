#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace tile_compile::metrics {
class QualityMapCache;
}

namespace tile_compile::reconstruction {

// Forward declaration
namespace core { class AccelerationOps; }

struct WeightedTileResult {
  Matrix2Df tile;
  bool fallback_used = false;
  float effective_weight_sum = 0.0f;
};

struct RGBSharedSigmaClipResult {
  Matrix2Df R;
  Matrix2Df G;
  Matrix2Df B;
  bool fallback_used = false;
  float effective_weight_sum = 0.0f;
};

// Configuration for the parallel reconstruction phase.
struct ReconstructionConfig {
    int    parallel_workers                    = 1;
    size_t memory_budget_bytes                 = 512ULL * 1024 * 1024;
};

// Result returned by reconstruct_tiles_parallel().
struct ReconstructTilesResult {
    Matrix2Df output;
    int    tiles_processed                        = 0;
    int    tiles_skipped_dead                     = 0;
    double duration_s                             = 0.0;
    double dead_tile_time_saved_estimate_s        = 0.0;
    int    workers_used                           = 0;
    size_t allocated_frame_batch_bytes            = 0;
    size_t allocated_tile_batch_bytes             = 0;
};

struct AqmhReconstructionConfig {
    float sigma_low = 3.0f;
    float sigma_high = 3.0f;
    float min_fraction = 0.5f;
    float eps_weight = 1.0e-6f;
};

struct AqmhReconstructionResult {
    Matrix2Df output;
    Matrix2Df weight_sum;
    uint64_t unsupported_pixels = 0;
    uint64_t finite_map_samples = 0;
    uint64_t missing_map_samples = 0;
    uint64_t zero_veto_pixels = 0;
};

using AqmhFrameLoader = std::function<bool(size_t, Matrix2Df&)>;

// Independent AQMH reconstruction path.
//
// The implementation is streaming: it calls `load_frame` and `q_map_cache`
// per frame/pass and never requires all source frames or quality maps in RAM.
// Missing maps contribute zero AQMH weight. Finite all-zero map support is an
// explicit AQMH veto and remains unsupported/zero; there is no Classic
// tile-weight fallback.
AqmhReconstructionResult reconstruct_aqmh_weighted(
    size_t frame_count,
    const AqmhFrameLoader& load_frame,
    metrics::QualityMapCache* q_map_cache,
    const VectorXf& global_weights,
    const std::vector<uint8_t>& canvas_mask,
    int width,
    int height,
    const AqmhReconstructionConfig& cfg);

// Parallel implementation — replaces reconstruct_tiles() for new callers.
// dead_tile_mask must have the same size as grid.tiles; pass all-false to
// disable dead-tile skipping.
ReconstructTilesResult reconstruct_tiles_parallel(
    const std::vector<Matrix2Df>&          frames,
    const TileGrid&                        grid,
    const std::vector<std::vector<float>>& tile_weights,
    const std::vector<bool>&               dead_tile_mask,
    const ReconstructionConfig&            cfg);

// Legacy wrapper — kept for backward compatibility.
Matrix2Df reconstruct_tiles(const std::vector<Matrix2Df>& frames,
                            const TileGrid& grid,
                            const std::vector<std::vector<float>>& tile_weights);

// Wiener denoising filter for a single tile
Matrix2Df wiener_tile_filter(const Matrix2Df& tile, float sigma, float snr_tile,
                             float q_struct_tile, bool is_star_tile,
                             const config::WienerDenoiseConfig& cfg);

// Sigma-clipped mean stack of multiple frames
Matrix2Df sigma_clip_stack(const std::vector<Matrix2Df>& frames,
                           float sigma_low, float sigma_high,
                           int max_iters, float min_fraction);

// Per-pixel weighted sigma-clipped mean of tile stack.
// Rejects outlier pixels (e.g. star trails) before computing weighted mean.
Matrix2Df sigma_clip_weighted_tile(const std::vector<Matrix2Df>& tiles,
                                   const std::vector<float>& weights,
                                   float sigma_low, float sigma_high,
                                   int max_iters, float min_fraction);

WeightedTileResult sigma_clip_weighted_tile_with_fallback(
    const std::vector<Matrix2Df>& tiles, const std::vector<float>& weights,
    float sigma_low, float sigma_high, int max_iters, float min_fraction,
    float eps_weight);

RGBSharedSigmaClipResult sigma_clip_weighted_rgb_tile_shared_mask(
    const std::vector<Matrix2Df>& tiles_r,
    const std::vector<Matrix2Df>& tiles_g,
    const std::vector<Matrix2Df>& tiles_b,
    const std::vector<float>& weights,
    float sigma_low,
    float sigma_high,
    int max_iters,
    float min_fraction,
    float eps_weight);

// Highpass + Soft-Threshold denoising for a single tile (Methodik 3.1E §3.3.1)
Matrix2Df soft_threshold_tile_filter(const Matrix2Df& tile,
                                      const config::SoftThresholdConfig& cfg);

// Chroma-selective denoise for linear RGB planes (OSC pipeline).
void chroma_denoise_rgb_inplace(Matrix2Df& r, Matrix2Df& g, Matrix2Df& b,
                                const config::ChromaDenoiseConfig& cfg);

// Generate a 1D support-aware partition window with optional left/right
// overlap ramps. For interior overlap zones, adjacent tiles should use
// complementary overlap lengths so their weights sum to 1.
std::vector<float> make_partition_window_1d(int n, int left_overlap,
                                            int right_overlap);

} // namespace tile_compile::reconstruction
