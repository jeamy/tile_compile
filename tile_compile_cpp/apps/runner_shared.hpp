#pragma once

#include "tile_compile/astrometry/photometric_color_cal.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/image/background_extraction.hpp"

#include <cstdint>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <streambuf>
#include <string>
#include <thread>
#include <atomic>
#include <vector>

namespace tile_compile::runner {

/// Aggregate local tile metrics across multiple frames into a single median-based profile.
std::vector<tile_compile::TileMetrics> aggregate_tile_metrics_across_frames(
    const std::vector<std::vector<tile_compile::TileMetrics>> &local_metrics);

/// Format a byte count for human-readable logs and diagnostics.
std::string format_bytes(uint64_t bytes);

/// Sum file sizes for input-frame disk-space planning.
uint64_t estimate_total_file_bytes(const std::vector<std::filesystem::path> &paths);

/// Workload class used to tune runner worker counts for CPU, IO, or mixed phases.
enum class WorkerParallelProfile {
  CpuBound,
  MixedIo,
  IoHeavy,
};

/// Decision record for selecting synthetic-frame weighting mode.
///
/// The requested mode may be downgraded from tile-weighted to global when
/// boundary diagnostics show a high risk of visible tile seams.
struct SyntheticWeightingDecision {
  std::string requested_weighting = "global";
  std::string effective_weighting = "global";
  bool tile_seam_guard_triggered = false;
  int boundary_pair_count = 0;
  float boundary_pair_mean_abs_diff_p95 = 0.0f;
  float boundary_pair_scale_ratio_deviation_p95 = 0.0f;
  float boundary_post_background_delta_p95_abs = 0.0f;
  float local_weight_mean_abs_delta_p95 = 0.0f;
  float local_weight_correlation_p05 = 1.0f;
};

/// Choose a conservative worker count for the current phase and machine limits.
///
/// Combines `runtime_limits.parallel_workers`, memory budget, workload type,
/// and input-frame sizes. CPU-heavy phases can safely use more workers; IO-heavy
/// phases are throttled to reduce disk pressure.
int compute_adaptive_worker_count(
    const config::Config &cfg, size_t task_count,
    const std::vector<std::filesystem::path> &frames,
    WorkerParallelProfile profile);

/// Determine default parallel workers for CPU bound tasks without memory capping.
int default_parallel_workers(size_t items, int requested_workers = 0);

/// Platform-aware shell quoting for external commands.
std::string shell_quote(const std::string &s);

/// Wrap a command for execution via std::system (cmd /c "..." on Windows).
std::string system_cmd(const std::string &cmd);

/// Resolve an ASTAP CLI binary path across platforms.
std::filesystem::path resolve_astap_binary_path(const std::string &astap_bin_cfg,
                                                const std::string &astap_data_dir);

/// Simple parallel-for loop over indices [0, count).
template <typename Fn>
void parallel_for_indices(size_t count, int workers, Fn fn) {
  if (count == 0) return;
  workers = std::max(1, std::min<int>(workers, static_cast<int>(count)));
  if (workers == 1) {
    for (size_t i = 0; i < count; ++i) fn(i);
    return;
  }
  std::atomic<size_t> next{0};
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers));
  for (int w = 0; w < workers; ++w) {
    threads.emplace_back([&]() {
      while (true) {
        const size_t i = next.fetch_add(1);
        if (i >= count) break;
        fn(i);
      }
    });
  }
  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }
}


/// Memory-budget plan for frame sub-batching.
///
/// This mirrors the Tile Compile OSC reconstruction rule:
/// `workers * sub_batch * pixels_per_worker * channels * sizeof(float)` should
/// fit into 80% of `runtime_limits.memory_budget`.
struct FrameSubBatchPlan {
  size_t frame_sub_batch_size = 0;
  int effective_workers = 1;
  uint64_t memory_budget_bytes = 0;
  uint64_t bytes_per_frame_per_worker = 0;
  bool budget_too_small_for_requested_workers = false;
  bool sub_batch_limited = false;
};

FrameSubBatchPlan compute_memory_capped_frame_sub_batch(
    size_t frame_count,
    size_t pixels_per_worker,
    int channels,
    int requested_workers,
    int memory_budget_mb);

/// Resolve the PCC aperture FWHM used by auto-radius photometry.
///
/// Estimates a representative star size from RGB channels and falls back to a
/// caller-provided FWHM when image-based estimation is unavailable. `source_out`
/// reports whether the value came from image measurement, fallback, or default.
double resolve_pcc_auto_fwhm_px(const Matrix2Df &R, const Matrix2Df &G,
                                const Matrix2Df &B,
                                bool have_fallback_fwhm = false,
                                double fallback_fwhm_px = 0.0,
                                std::string *source_out = nullptr);

/// Apply seam-risk guardrails to synthetic-frame weighting selection.
inline SyntheticWeightingDecision decide_synthetic_weighting(
    const std::string &requested_weighting, int boundary_pair_count,
    float boundary_pair_mean_abs_diff_p95,
    float boundary_pair_scale_ratio_deviation_p95,
    float boundary_post_background_delta_p95_abs,
    float local_weight_mean_abs_delta_p95,
    float local_weight_correlation_p05) {
  SyntheticWeightingDecision out;
  out.requested_weighting = requested_weighting;
  out.effective_weighting = requested_weighting;
  out.boundary_pair_count = boundary_pair_count;
  out.boundary_pair_mean_abs_diff_p95 = boundary_pair_mean_abs_diff_p95;
  out.boundary_pair_scale_ratio_deviation_p95 =
      boundary_pair_scale_ratio_deviation_p95;
  out.boundary_post_background_delta_p95_abs =
      boundary_post_background_delta_p95_abs;
  out.local_weight_mean_abs_delta_p95 = local_weight_mean_abs_delta_p95;
  out.local_weight_correlation_p05 = local_weight_correlation_p05;

  if (requested_weighting != "tile_weighted") {
    return out;
  }

  constexpr int kMinObservedBoundaryPairs = 8;
  // Guard only against clearly visible, severe seam regressions.
  // Earlier thresholds were tight enough to reject historically good runs.
  constexpr float kBoundaryMeanAbsDiffP95 = 0.25f;
  constexpr float kBoundaryPostBackgroundDeltaP95 = 0.25f;

  const bool enough_pairs = boundary_pair_count >= kMinObservedBoundaryPairs;
  const bool severe_boundary_regression =
      (std::isfinite(boundary_pair_mean_abs_diff_p95) &&
       boundary_pair_mean_abs_diff_p95 > kBoundaryMeanAbsDiffP95);
  const bool severe_background_step =
      (std::isfinite(boundary_post_background_delta_p95_abs) &&
       boundary_post_background_delta_p95_abs > kBoundaryPostBackgroundDeltaP95);

  if (enough_pairs && severe_boundary_regression && severe_background_step) {
    out.effective_weighting = "global";
    out.tile_seam_guard_triggered = true;
  }

  return out;
}

/// Sentinel used for pixels outside the common-overlap mask.
inline float common_overlap_invalid_value() {
  return std::numeric_limits<float>::quiet_NaN();
}

/// Apply the COMMON_OVERLAP mask to one tile in-place.
///
/// Pixels outside the global common-overlap mask are written as NaN so later
/// metrics/reconstruction paths can ignore them without maintaining a separate
/// mask per tile. The helper is deliberately inline because it is used in hot
/// loops across local metrics, reconstruction, and diagnostics.
inline void apply_common_overlap_to_tile_inplace(
    Matrix2Df &tile, const Tile &t, const std::vector<uint8_t> &common_valid_mask,
    int common_mask_width, int common_mask_height) {
  if (tile.rows() != t.height || tile.cols() != t.width)
    return;
  if (common_mask_width <= 0 || common_mask_height <= 0 ||
      common_valid_mask.empty()) {
    return;
  }

  const int tile_cols = static_cast<int>(tile.cols());
  const size_t mask_size = common_valid_mask.size();
  float *tile_data = tile.data();
  const float invalid = common_overlap_invalid_value();

  for (int yy = 0; yy < t.height; ++yy) {
    const int gy = t.y + yy;
    const size_t tile_row_off =
        static_cast<size_t>(yy) * static_cast<size_t>(tile_cols);
    if (gy < 0 || gy >= common_mask_height) {
      for (int xx = 0; xx < t.width; ++xx) {
        tile_data[tile_row_off + static_cast<size_t>(xx)] = invalid;
      }
      continue;
    }

    const size_t row_off =
        static_cast<size_t>(gy) * static_cast<size_t>(common_mask_width);

    for (int xx = 0; xx < t.width; ++xx) {
      const int gx = t.x + xx;
      if (gx < 0 || gx >= common_mask_width) {
        tile_data[tile_row_off + static_cast<size_t>(xx)] = invalid;
        continue;
      }
      const size_t mask_idx = row_off + static_cast<size_t>(gx);
      if (mask_idx >= mask_size || common_valid_mask[mask_idx] == 0) {
        tile_data[tile_row_off + static_cast<size_t>(xx)] = invalid;
      }
    }
  }
}

/// Apply a common-overlap mask to a tile and report whether finite data remains.
inline bool apply_common_overlap_to_tile_inplace_and_check_nonzero(
    Matrix2Df &tile, const Tile &t, const std::vector<uint8_t> &common_valid_mask,
    int common_mask_width, int common_mask_height) {
  if (tile.rows() != t.height || tile.cols() != t.width)
    return false;
  if (common_mask_width <= 0 || common_mask_height <= 0 ||
      common_valid_mask.empty()) {
    return false;
  }

  const int tile_cols = static_cast<int>(tile.cols());
  const size_t mask_size = common_valid_mask.size();
  float *tile_data = tile.data();
  bool any_valid = false;
  const float invalid = common_overlap_invalid_value();

  for (int yy = 0; yy < t.height; ++yy) {
    const int gy = t.y + yy;
    const size_t tile_row_off =
        static_cast<size_t>(yy) * static_cast<size_t>(tile_cols);

    if (gy < 0 || gy >= common_mask_height) {
      for (int xx = 0; xx < t.width; ++xx) {
        tile_data[tile_row_off + static_cast<size_t>(xx)] = invalid;
      }
      continue;
    }

    const size_t row_off =
        static_cast<size_t>(gy) * static_cast<size_t>(common_mask_width);

    for (int xx = 0; xx < t.width; ++xx) {
      const int gx = t.x + xx;
      float &v = tile_data[tile_row_off + static_cast<size_t>(xx)];
      if (gx < 0 || gx >= common_mask_width) {
        v = invalid;
        continue;
      }
      const size_t mask_idx = row_off + static_cast<size_t>(gx);
      if (mask_idx >= mask_size || common_valid_mask[mask_idx] == 0) {
        v = invalid;
        continue;
      }
      if (std::isfinite(v)) {
        any_valid = true;
      }
    }
  }
  return any_valid;
}

/// Apply the common-overlap mask to a full mono/luma frame in-place.
inline bool apply_common_overlap_to_frame_inplace_and_check_nonzero(
    Matrix2Df &frame, const std::vector<uint8_t> &common_valid_mask,
    int common_mask_width, int common_mask_height) {
  if (frame.rows() != common_mask_height || frame.cols() != common_mask_width) {
    return false;
  }
  if (common_mask_width <= 0 || common_mask_height <= 0 ||
      common_valid_mask.empty()) {
    return false;
  }

  const size_t mask_size = common_valid_mask.size();
  float *frame_data = frame.data();
  bool any_valid = false;
  const float invalid = common_overlap_invalid_value();
  for (int y = 0; y < common_mask_height; ++y) {
    const size_t row_off =
        static_cast<size_t>(y) * static_cast<size_t>(common_mask_width);
    for (int x = 0; x < common_mask_width; ++x) {
      const size_t idx = row_off + static_cast<size_t>(x);
      if (idx >= mask_size || common_valid_mask[idx] == 0) {
        frame_data[idx] = invalid;
        continue;
      }
      if (std::isfinite(frame_data[idx])) {
        any_valid = true;
      }
    }
  }
  return any_valid;
}

/// Apply the common-overlap mask consistently to full RGB frames.
inline bool apply_common_overlap_to_rgb_frames_inplace_and_check_nonzero(
    Matrix2Df &r_frame, Matrix2Df &g_frame, Matrix2Df &b_frame,
    const std::vector<uint8_t> &common_valid_mask, int common_mask_width,
    int common_mask_height) {
  if (r_frame.rows() != common_mask_height ||
      r_frame.cols() != common_mask_width ||
      g_frame.rows() != common_mask_height ||
      g_frame.cols() != common_mask_width ||
      b_frame.rows() != common_mask_height ||
      b_frame.cols() != common_mask_width) {
    return false;
  }
  if (common_mask_width <= 0 || common_mask_height <= 0 ||
      common_valid_mask.empty()) {
    return false;
  }

  const size_t mask_size = common_valid_mask.size();
  float *r_data = r_frame.data();
  float *g_data = g_frame.data();
  float *b_data = b_frame.data();
  bool any_valid = false;
  const float invalid = common_overlap_invalid_value();
  const size_t total =
      static_cast<size_t>(common_mask_width) * static_cast<size_t>(common_mask_height);
  for (size_t idx = 0; idx < total; ++idx) {
    if (idx >= mask_size || common_valid_mask[idx] == 0) {
      r_data[idx] = invalid;
      g_data[idx] = invalid;
      b_data[idx] = invalid;
      continue;
    }
    if (std::isfinite(r_data[idx]) || std::isfinite(g_data[idx]) ||
        std::isfinite(b_data[idx])) {
      any_valid = true;
    }
  }
  return any_valid;
}

/// Apply the common-overlap mask consistently to RGB tile planes.
inline bool apply_common_overlap_to_rgb_tiles_inplace_and_check_nonzero(
    Matrix2Df &r_tile, Matrix2Df &g_tile, Matrix2Df &b_tile, const Tile &t,
    const std::vector<uint8_t> &common_valid_mask, int common_mask_width,
    int common_mask_height) {
  if (r_tile.rows() != t.height || r_tile.cols() != t.width ||
      g_tile.rows() != t.height || g_tile.cols() != t.width ||
      b_tile.rows() != t.height || b_tile.cols() != t.width) {
    return false;
  }
  if (common_mask_width <= 0 || common_mask_height <= 0 ||
      common_valid_mask.empty()) {
    return false;
  }

  const int tile_cols = static_cast<int>(r_tile.cols());
  const size_t mask_size = common_valid_mask.size();
  float *r_data = r_tile.data();
  float *g_data = g_tile.data();
  float *b_data = b_tile.data();
  bool any_valid = false;
  const float invalid = common_overlap_invalid_value();

  for (int yy = 0; yy < t.height; ++yy) {
    const int gy = t.y + yy;
    const size_t tile_row_off =
        static_cast<size_t>(yy) * static_cast<size_t>(tile_cols);

    if (gy < 0 || gy >= common_mask_height) {
      for (int xx = 0; xx < t.width; ++xx) {
        const size_t idx = tile_row_off + static_cast<size_t>(xx);
        r_data[idx] = invalid;
        g_data[idx] = invalid;
        b_data[idx] = invalid;
      }
      continue;
    }

    const size_t row_off =
        static_cast<size_t>(gy) * static_cast<size_t>(common_mask_width);

    for (int xx = 0; xx < t.width; ++xx) {
      const int gx = t.x + xx;
      const size_t idx = tile_row_off + static_cast<size_t>(xx);
      if (gx < 0 || gx >= common_mask_width) {
        r_data[idx] = invalid;
        g_data[idx] = invalid;
        b_data[idx] = invalid;
        continue;
      }
      const size_t mask_idx = row_off + static_cast<size_t>(gx);
      if (mask_idx >= mask_size || common_valid_mask[mask_idx] == 0) {
        r_data[idx] = invalid;
        g_data[idx] = invalid;
        b_data[idx] = invalid;
        continue;
      }
      if (std::isfinite(r_data[idx]) || std::isfinite(g_data[idx]) ||
          std::isfinite(b_data[idx])) {
        any_valid = true;
      }
    }
  }
  return any_valid;
}

/// Fast tile-gating helper used after COMMON_OVERLAP masking.
inline bool tile_has_nonzero_common_data(
    const Matrix2Df &tile, size_t tile_index,
    const std::vector<uint8_t> &tile_common_valid) {
  if (tile_index >= tile_common_valid.size() || tile_common_valid[tile_index] == 0)
    return false;
  const float *ptr = tile.data();
  for (Eigen::Index i = 0; i < tile.size(); ++i) {
    if (std::isfinite(ptr[i])) {
      return true;
    }
  }
  return false;
}

/// Detect common disk-full messages from exception strings and library errors.
bool message_indicates_disk_full(const std::string &message);

/// Load a mono FITS canvas mask and validate it against expected dimensions.
bool load_canvas_mask_fits(const std::filesystem::path &mask_path, int rows,
                           int cols, std::vector<uint8_t> &out_mask,
                           std::string &error_out);

/// Load and validate the canvas mask that belongs to RGB output planes.
bool load_canvas_mask_for_rgb(const std::filesystem::path &mask_path,
                              const Matrix2Df &R, const Matrix2Df &G,
                              const Matrix2Df &B,
                              std::vector<uint8_t> &out_mask,
                              int &rows_out, int &cols_out,
                              std::string &error_out);

/// Integer bounds of a set of warped frame corners on the output canvas.
struct WarpBounds {
  int min_x = 0;
  int min_y = 0;
  int max_x = 0;
  int max_y = 0;

  [[nodiscard]] int width() const { return max_x - min_x; }
  [[nodiscard]] int height() const { return max_y - min_y; }
};

/// Invert a 2x3 affine warp matrix, returning false for singular matrices.
bool invert_affine_warp(const WarpMatrix &w, WarpMatrix &inv);

/// Compute the minimal canvas bounds that contain all warped input frames.
WarpBounds compute_warps_bounds(int width, int height,
                                const std::vector<WarpMatrix> &warps);

/// Axis-aligned crop rectangle in image coordinates.
struct CropBox {
  int x{0};
  int y{0};
  int width{0};
  int height{0};

  [[nodiscard]] bool valid() const { return width > 0 && height > 0; }
};

/// Find the bounding box of finite/nonzero reconstructed data.
CropBox compute_nonzero_data_bbox(const Matrix2Df &luma,
                                  const Matrix2Df *r = nullptr,
                                  const Matrix2Df *g = nullptr,
                                  const Matrix2Df *b = nullptr);

/// Find the largest crop box supported by the common-valid mask and data planes.
CropBox compute_largest_valid_crop_box(const Matrix2Df &luma,
                                       const std::vector<uint8_t> &common_valid_mask,
                                       int mask_rows, int mask_cols,
                                       const Matrix2Df *r = nullptr,
                                       const Matrix2Df *g = nullptr,
                                       const Matrix2Df *b = nullptr);

/// Convert runner configuration into the image-module BGE runtime config.
image::BGEConfig to_image_bge_config(const config::BGEConfig &src);
/// Convert runner configuration into the astrometry-module PCC runtime config.
astrometry::PCCConfig to_astrometry_pcc_config(const config::PCCConfig &src);

/// Build the downsampled registration proxy used by global registration.
///
/// OSC inputs use a CFA-aware green-channel proxy; mono inputs use a simple
/// 2x2 mean downsample. The returned image is intentionally lower resolution
/// than the source frame, so translation components must be scaled before they
/// are applied to full-resolution frames.
Matrix2Df build_registration_proxy(const Matrix2Df &img, ColorMode detected_mode,
                                   const std::string &detected_bayer_str);

/// Serialize BGE diagnostics into the artifact/report JSON shape.
tile_compile::core::json bge_diag_to_json(const image::BGEDiagnostics &diag,
                                          bool requested,
                                          bool have_tile_data,
                                          bool metrics_tiles_match);

/// Result of selecting/querying the PCC star catalog backend.
struct PCCCatalogQueryResult {
  std::vector<astrometry::GaiaStar> stars;
  std::string used_source;
};

/// Query the configured PCC catalog source for stars covering the solved WCS.
PCCCatalogQueryResult query_pcc_catalog_stars(const astrometry::WCS &wcs,
                                              const config::PCCConfig &cfg,
                                              std::ostream &log_stream,
                                              const std::string &log_prefix);

/// Stream buffer that mirrors writes to two destination stream buffers.
class TeeBuf : public std::streambuf {
public:
  TeeBuf(std::streambuf *a, std::streambuf *b);

protected:
  int overflow(int c) override;
  int sync() override;

private:
  std::streambuf *a_;
  std::streambuf *b_;
};

/// Disk-backed fixed-size frame store.
///
/// Frames are written as raw float matrices in a run-local cache directory and
/// loaded/memory-mapped on demand. This keeps long runs under the configured
/// memory budget while still allowing random tile extraction in later phases.
class DiskCacheFrameStore {
public:
  DiskCacheFrameStore();
  DiskCacheFrameStore(const std::filesystem::path &cache_dir, size_t n_frames,
                      int rows, int cols);
  ~DiskCacheFrameStore();

  DiskCacheFrameStore(const DiskCacheFrameStore &) = delete;
  DiskCacheFrameStore &operator=(const DiskCacheFrameStore &) = delete;
  DiskCacheFrameStore(DiskCacheFrameStore &&o) noexcept;
  DiskCacheFrameStore &operator=(DiskCacheFrameStore &&o) noexcept;

  /// Persist one full frame at index `fi`.
  void store(size_t fi, const Matrix2Df &frame);
  /// Load a full frame into memory.
  Matrix2Df load(size_t fi) const;
  /// Return a mapped pointer to full-frame data; valid until cache cleanup.
  const float *frame_data(size_t fi) const;
  /// Extract a tile by value, applying optional canvas coordinate offsets.
  Matrix2Df extract_tile(size_t fi, const Tile &t, int offset_x = 0,
                         int offset_y = 0) const;
  /// Extract a tile into an existing matrix to avoid repeated allocations.
  bool extract_tile_into(size_t fi, const Tile &t, Matrix2Df &out,
                         int offset_x = 0, int offset_y = 0) const;
  /// Release all currently mapped frame views without deleting cached files.
  void clear_mappings() const;

  /// Whether a frame has been stored for `fi`.
  bool has_data(size_t fi) const;
  /// Number of frame slots managed by the store.
  size_t size() const;
  /// Stored frame row count.
  int rows() const;
  /// Stored frame column count.
  int cols() const;

  /// Remove cached files and release all mappings.
  void cleanup();

private:
  const float *mapped_frame_ptr(size_t fi) const;
  void invalidate_mapping(size_t fi);
  std::filesystem::path frame_path(size_t fi) const;

  std::filesystem::path cache_dir_;
  int rows_ = 0;
  int cols_ = 0;
  size_t frame_bytes_ = 0;
  std::vector<uint8_t> has_data_;
  mutable std::mutex mapped_mutex_;
  mutable std::vector<void *> mapped_views_;
};

/// Shared per-run cache for normalized frames and registration proxies.
///
/// Normalized full frames are disk-backed through `DiskCacheFrameStore`; compact
/// registration proxies stay in memory because they are repeatedly reused by
/// direct registration, anchor promotion, and rescue passes.
class RunnerFrameCache {
public:
  RunnerFrameCache();
  RunnerFrameCache(const std::filesystem::path &cache_dir, size_t n_frames,
                   int rows, int cols);

  RunnerFrameCache(const RunnerFrameCache &) = delete;
  RunnerFrameCache &operator=(const RunnerFrameCache &) = delete;
  RunnerFrameCache(RunnerFrameCache &&) = delete;
  RunnerFrameCache &operator=(RunnerFrameCache &&) = delete;

  /// Store a normalized full-resolution frame in the disk cache.
  void store_normalized(size_t fi, const Matrix2Df &frame);
  /// Load a normalized full-resolution frame from the disk cache.
  Matrix2Df load_normalized(size_t fi) const;
  /// Try to load a normalized frame, returning false if not available.
  bool try_load_normalized(size_t fi, Matrix2Df &out) const;
  /// Whether normalized frame data is available for `fi`.
  bool has_normalized(size_t fi) const;

  /// Store the downsampled registration proxy for `fi`.
  void store_registration_proxy(size_t fi, const Matrix2Df &proxy);
  /// Load a cached registration proxy if present.
  bool try_load_registration_proxy(size_t fi, Matrix2Df &out) const;

  /// Number of frame slots in the cache.
  size_t size() const;
  /// Normalized frame row count.
  int rows() const;
  /// Normalized frame column count.
  int cols() const;
  /// Remove disk-backed normalized frames and clear in-memory proxies.
  void cleanup();

private:
  DiskCacheFrameStore normalized_frames_;
  mutable std::mutex proxy_mutex_;
  std::vector<uint8_t> has_registration_proxy_;
  std::vector<Matrix2Df> registration_proxies_;
};

} // namespace tile_compile::runner
