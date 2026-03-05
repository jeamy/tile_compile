#pragma once

#include "tile_compile/astrometry/photometric_color_cal.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/image/background_extraction.hpp"

#include <cstdint>
#include <filesystem>
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

int compute_adaptive_worker_count(
    const config::Config &cfg, size_t task_count,
    const std::vector<std::filesystem::path> &frames,
    WorkerParallelProfile profile);

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

  for (int yy = 0; yy < t.height; ++yy) {
    const int gy = t.y + yy;
    if (gy < 0 || gy >= common_mask_height)
      continue;

    const size_t row_off =
        static_cast<size_t>(gy) * static_cast<size_t>(common_mask_width);
    const size_t tile_row_off =
        static_cast<size_t>(yy) * static_cast<size_t>(tile_cols);

    for (int xx = 0; xx < t.width; ++xx) {
      const int gx = t.x + xx;
      if (gx < 0 || gx >= common_mask_width) {
        tile_data[tile_row_off + static_cast<size_t>(xx)] = 0.0f;
        continue;
      }
      const size_t mask_idx = row_off + static_cast<size_t>(gx);
      if (mask_idx >= mask_size || common_valid_mask[mask_idx] == 0) {
        tile_data[tile_row_off + static_cast<size_t>(xx)] = 0.0f;
      }
    }
  }
}

// Fast tile-gating helper used after COMMON_OVERLAP masking.
inline bool tile_has_nonzero_common_data(
    const Matrix2Df &tile, size_t tile_index,
    const std::vector<uint8_t> &tile_common_valid) {
  if (tile_index >= tile_common_valid.size() || tile_common_valid[tile_index] == 0)
    return false;
  const float *ptr = tile.data();
  for (Eigen::Index i = 0; i < tile.size(); ++i) {
    if (ptr[i] > 0.0f) {
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

image::BGEConfig to_image_bge_config(const config::BGEConfig &src);
astrometry::PCCConfig to_astrometry_pcc_config(const config::PCCConfig &src);

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
  Matrix2Df extract_tile(size_t fi, const Tile &t, int offset_x = 0,
                         int offset_y = 0) const;

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

} // namespace tile_compile::runner
