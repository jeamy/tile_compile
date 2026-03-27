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
#include <vector>

namespace tile_compile::runner {

std::string format_bytes(uint64_t bytes);

uint64_t estimate_total_file_bytes(const std::vector<std::filesystem::path> &paths);

enum class WorkerParallelProfile {
  CpuBound,
  MixedIo,
  IoHeavy,
};

struct SyntheticWeightingDecision {
  std::string requested_weighting = "global";
  std::string effective_weighting = "global";
  bool tile_seam_guard_triggered = false;
  int boundary_pair_count = 0;
  float boundary_pair_mean_abs_diff_p95 = 0.0f;
  float boundary_pair_scale_ratio_deviation_p95 = 0.0f;
  float local_weight_mean_abs_delta_p95 = 0.0f;
  float local_weight_correlation_p05 = 1.0f;
};

int compute_adaptive_worker_count(
    const config::Config &cfg, size_t task_count,
    const std::vector<std::filesystem::path> &frames,
    WorkerParallelProfile profile);

inline SyntheticWeightingDecision decide_synthetic_weighting(
    const std::string &requested_weighting, int boundary_pair_count,
    float boundary_pair_mean_abs_diff_p95,
    float boundary_pair_scale_ratio_deviation_p95,
    float local_weight_mean_abs_delta_p95,
    float local_weight_correlation_p05) {
  SyntheticWeightingDecision out;
  out.requested_weighting = requested_weighting;
  out.effective_weighting = requested_weighting;
  out.boundary_pair_count = boundary_pair_count;
  out.boundary_pair_mean_abs_diff_p95 = boundary_pair_mean_abs_diff_p95;
  out.boundary_pair_scale_ratio_deviation_p95 =
      boundary_pair_scale_ratio_deviation_p95;
  out.local_weight_mean_abs_delta_p95 = local_weight_mean_abs_delta_p95;
  out.local_weight_correlation_p05 = local_weight_correlation_p05;

  if (requested_weighting != "tile_weighted") {
    return out;
  }

  constexpr int kMinObservedBoundaryPairs = 8;
  constexpr float kBoundaryMeanAbsDiffP95 = 0.010f;
  constexpr float kBoundaryScaleRatioDeviationP95 = 0.050f;
  constexpr float kLocalWeightMeanAbsDeltaP95 = 3.0f;
  constexpr float kLocalWeightCorrelationP05 = 0.10f;

  const bool enough_pairs = boundary_pair_count >= kMinObservedBoundaryPairs;
  const bool boundary_regression =
      (std::isfinite(boundary_pair_mean_abs_diff_p95) &&
       boundary_pair_mean_abs_diff_p95 > kBoundaryMeanAbsDiffP95) ||
      (std::isfinite(boundary_pair_scale_ratio_deviation_p95) &&
       boundary_pair_scale_ratio_deviation_p95 > kBoundaryScaleRatioDeviationP95);
  const bool weight_disagreement =
      (std::isfinite(local_weight_mean_abs_delta_p95) &&
       local_weight_mean_abs_delta_p95 > kLocalWeightMeanAbsDeltaP95) ||
      (std::isfinite(local_weight_correlation_p05) &&
       local_weight_correlation_p05 < kLocalWeightCorrelationP05);

  if (enough_pairs && boundary_regression && weight_disagreement) {
    out.effective_weighting = "global";
    out.tile_seam_guard_triggered = true;
  }

  return out;
}

inline float common_overlap_invalid_value() {
  return std::numeric_limits<float>::quiet_NaN();
}

// Hot-path helper: applies COMMON_OVERLAP mask to a tile in-place.
// Keeps behavior consistent across pipeline phases and avoids duplicate lambdas.
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

// Fast tile-gating helper used after COMMON_OVERLAP masking.
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

bool message_indicates_disk_full(const std::string &message);

bool load_canvas_mask_fits(const std::filesystem::path &mask_path, int rows,
                           int cols, std::vector<uint8_t> &out_mask,
                           std::string &error_out);

bool load_canvas_mask_for_rgb(const std::filesystem::path &mask_path,
                              const Matrix2Df &R, const Matrix2Df &G,
                              const Matrix2Df &B,
                              std::vector<uint8_t> &out_mask,
                              int &rows_out, int &cols_out,
                              std::string &error_out);

struct CropBox {
  int x{0};
  int y{0};
  int width{0};
  int height{0};

  [[nodiscard]] bool valid() const { return width > 0 && height > 0; }
};

CropBox compute_nonzero_data_bbox(const Matrix2Df &luma,
                                  const Matrix2Df *r = nullptr,
                                  const Matrix2Df *g = nullptr,
                                  const Matrix2Df *b = nullptr);

CropBox compute_largest_valid_crop_box(const Matrix2Df &luma,
                                       const std::vector<uint8_t> &common_valid_mask,
                                       int mask_rows, int mask_cols,
                                       const Matrix2Df *r = nullptr,
                                       const Matrix2Df *g = nullptr,
                                       const Matrix2Df *b = nullptr);

image::BGEConfig to_image_bge_config(const config::BGEConfig &src);
astrometry::PCCConfig to_astrometry_pcc_config(const config::PCCConfig &src);

Matrix2Df build_registration_proxy(const Matrix2Df &img, ColorMode detected_mode,
                                   const std::string &detected_bayer_str);

tile_compile::core::json bge_diag_to_json(const image::BGEDiagnostics &diag,
                                          bool requested,
                                          bool have_tile_data,
                                          bool metrics_tiles_match);

struct PCCCatalogQueryResult {
  std::vector<astrometry::GaiaStar> stars;
  std::string used_source;
};

PCCCatalogQueryResult query_pcc_catalog_stars(const astrometry::WCS &wcs,
                                              const config::PCCConfig &cfg,
                                              std::ostream &log_stream,
                                              const std::string &log_prefix);

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

  void store(size_t fi, const Matrix2Df &frame);
  Matrix2Df load(size_t fi) const;
  const float *frame_data(size_t fi) const;
  Matrix2Df extract_tile(size_t fi, const Tile &t, int offset_x = 0,
                         int offset_y = 0) const;
  bool extract_tile_into(size_t fi, const Tile &t, Matrix2Df &out,
                         int offset_x = 0, int offset_y = 0) const;

  bool has_data(size_t fi) const;
  size_t size() const;
  int rows() const;
  int cols() const;

  void cleanup();

private:
  const float *mapped_frame_ptr(size_t fi) const;
  void invalidate_mapping(size_t fi);
  void clear_mappings();
  std::filesystem::path frame_path(size_t fi) const;

  std::filesystem::path cache_dir_;
  int rows_ = 0;
  int cols_ = 0;
  size_t frame_bytes_ = 0;
  std::vector<uint8_t> has_data_;
  mutable std::mutex mapped_mutex_;
  mutable std::vector<void *> mapped_views_;
};

class RunnerFrameCache {
public:
  RunnerFrameCache();
  RunnerFrameCache(const std::filesystem::path &cache_dir, size_t n_frames,
                   int rows, int cols);

  RunnerFrameCache(const RunnerFrameCache &) = delete;
  RunnerFrameCache &operator=(const RunnerFrameCache &) = delete;
  RunnerFrameCache(RunnerFrameCache &&) = delete;
  RunnerFrameCache &operator=(RunnerFrameCache &&) = delete;

  void store_normalized(size_t fi, const Matrix2Df &frame);
  Matrix2Df load_normalized(size_t fi) const;
  bool has_normalized(size_t fi) const;

  void store_registration_proxy(size_t fi, const Matrix2Df &proxy);
  bool try_load_registration_proxy(size_t fi, Matrix2Df &out) const;

  size_t size() const;
  int rows() const;
  int cols() const;
  void cleanup();

private:
  DiskCacheFrameStore normalized_frames_;
  mutable std::mutex proxy_mutex_;
  std::vector<uint8_t> has_registration_proxy_;
  std::vector<Matrix2Df> registration_proxies_;
};

} // namespace tile_compile::runner
