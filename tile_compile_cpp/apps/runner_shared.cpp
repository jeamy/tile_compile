#include "runner_shared.hpp"

#include "tile_compile/core/utils.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#include <windows.h>
#include <fileapi.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tile_compile::runner {

namespace fs = std::filesystem;
namespace core = tile_compile::core;
namespace image = tile_compile::image;
namespace config = tile_compile::config;
namespace astrometry = tile_compile::astrometry;

std::string format_bytes(uint64_t bytes) {
  static const char *kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < (sizeof(kUnits) / sizeof(kUnits[0]))) {
    value /= 1024.0;
    ++unit;
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(unit == 0 ? 0 : 2) << value << " "
      << kUnits[unit];
  return oss.str();
}

uint64_t estimate_total_file_bytes(const std::vector<fs::path> &paths) {
  uint64_t total = 0;
  for (const auto &p : paths) {
    std::error_code ec;
    const auto sz = fs::file_size(p, ec);
    if (ec) {
      continue;
    }
    if (sz > 0 &&
        total <= std::numeric_limits<uint64_t>::max() -
                     static_cast<uint64_t>(sz)) {
      total += static_cast<uint64_t>(sz);
    } else {
      total = std::numeric_limits<uint64_t>::max();
      break;
    }
  }
  return total;
}

bool message_indicates_disk_full(const std::string &message) {
  const std::string m = core::to_lower(message);
  return (m.find("no space left on device") != std::string::npos) ||
         (m.find("disk full") != std::string::npos) ||
         (m.find("not enough space") != std::string::npos) ||
         (m.find("enospc") != std::string::npos);
}

image::BGEConfig to_image_bge_config(const config::BGEConfig &src) {
  image::BGEConfig dst;
  dst.enabled = src.enabled;
  dst.sample_quantile = src.sample_quantile;
  dst.structure_thresh_percentile = src.structure_thresh_percentile;
  dst.min_tiles_per_cell = src.min_tiles_per_cell;
  dst.mask.star_dilate_px = src.mask.star_dilate_px;
  dst.mask.sat_dilate_px = src.mask.sat_dilate_px;
  dst.grid.N_g = src.grid.N_g;
  dst.grid.G_min_px = src.grid.G_min_px;
  dst.grid.G_max_fraction = src.grid.G_max_fraction;
  dst.grid.insufficient_cell_strategy = src.grid.insufficient_cell_strategy;
  dst.fit.method = src.fit.method;
  dst.fit.robust_loss = src.fit.robust_loss;
  dst.fit.huber_delta = src.fit.huber_delta;
  dst.fit.irls_max_iterations = src.fit.irls_max_iterations;
  dst.fit.irls_tolerance = src.fit.irls_tolerance;
  dst.fit.polynomial_order = src.fit.polynomial_order;
  dst.fit.rbf_phi = src.fit.rbf_phi;
  dst.fit.rbf_mu_factor = src.fit.rbf_mu_factor;
  dst.fit.rbf_lambda = src.fit.rbf_lambda;
  dst.fit.rbf_epsilon = src.fit.rbf_epsilon;
  dst.autotune.enabled = src.autotune.enabled;
  dst.autotune.max_evals = src.autotune.max_evals;
  dst.autotune.holdout_fraction = src.autotune.holdout_fraction;
  dst.autotune.alpha_flatness = src.autotune.alpha_flatness;
  dst.autotune.beta_roughness = src.autotune.beta_roughness;
  dst.autotune.strategy = src.autotune.strategy;
  return dst;
}

astrometry::PCCConfig to_astrometry_pcc_config(const config::PCCConfig &src) {
  astrometry::PCCConfig dst;
  dst.aperture_radius_px = src.aperture_radius_px;
  dst.annulus_inner_px = src.annulus_inner_px;
  dst.annulus_outer_px = src.annulus_outer_px;
  dst.mag_limit = src.mag_limit;
  dst.mag_bright_limit = src.mag_bright_limit;
  dst.min_stars = src.min_stars;
  dst.sigma_clip = src.sigma_clip;
  dst.background_model = src.background_model;
  dst.max_condition_number = src.max_condition_number;
  dst.max_residual_rms = src.max_residual_rms;
  dst.radii_mode = src.radii_mode;
  dst.aperture_fwhm_mult = src.aperture_fwhm_mult;
  dst.annulus_inner_fwhm_mult = src.annulus_inner_fwhm_mult;
  dst.annulus_outer_fwhm_mult = src.annulus_outer_fwhm_mult;
  dst.min_aperture_px = src.min_aperture_px;
  return dst;
}

TeeBuf::TeeBuf(std::streambuf *a, std::streambuf *b) : a_(a), b_(b) {}

int TeeBuf::overflow(int c) {
  if (c == EOF)
    return EOF;
  const int ra = a_ ? a_->sputc(static_cast<char>(c)) : c;
  const int rb = b_ ? b_->sputc(static_cast<char>(c)) : c;
  return (ra == EOF || rb == EOF) ? EOF : c;
}

int TeeBuf::sync() {
  int ra = a_ ? a_->pubsync() : 0;
  int rb = b_ ? b_->pubsync() : 0;
  return (ra == 0 && rb == 0) ? 0 : -1;
}

DiskCacheFrameStore::DiskCacheFrameStore() = default;

DiskCacheFrameStore::DiskCacheFrameStore(const fs::path &cache_dir,
                                         size_t n_frames, int rows, int cols)
    : cache_dir_(cache_dir), rows_(rows), cols_(cols),
      frame_bytes_(static_cast<size_t>(rows) * static_cast<size_t>(cols) *
                   sizeof(float)),
      has_data_(n_frames, false) {
  fs::create_directories(cache_dir_);
}

DiskCacheFrameStore::~DiskCacheFrameStore() { cleanup(); }

DiskCacheFrameStore::DiskCacheFrameStore(DiskCacheFrameStore &&o) noexcept
    : cache_dir_(std::move(o.cache_dir_)), rows_(o.rows_), cols_(o.cols_),
      frame_bytes_(o.frame_bytes_), has_data_(std::move(o.has_data_)) {}

DiskCacheFrameStore &DiskCacheFrameStore::operator=(DiskCacheFrameStore &&o) noexcept {
  if (this != &o) {
    cleanup();
    cache_dir_ = std::move(o.cache_dir_);
    rows_ = o.rows_;
    cols_ = o.cols_;
    frame_bytes_ = o.frame_bytes_;
    has_data_ = std::move(o.has_data_);
  }
  return *this;
}

void DiskCacheFrameStore::store(size_t fi, const Matrix2Df &frame) {
  if (frame.rows() != rows_ || frame.cols() != cols_) {
    std::cerr << "[DiskCache] Frame " << fi << " size mismatch: got " 
              << frame.rows() << "x" << frame.cols() << ", expected " 
              << rows_ << "x" << cols_ << std::endl;
    return;
  }
  fs::path p = frame_path(fi);
#ifdef _WIN32
  HANDLE hFile = CreateFileW(p.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile == INVALID_HANDLE_VALUE)
    return;
  DWORD written = 0;
  const char *src = reinterpret_cast<const char *>(frame.data());
  WriteFile(hFile, src, static_cast<DWORD>(frame_bytes_), &written, NULL);
  CloseHandle(hFile);
  if (written == frame_bytes_) {
    has_data_[fi] = true;
  }
#else
  int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0)
    return;
  size_t written = 0;
  const char *src = reinterpret_cast<const char *>(frame.data());
  while (written < frame_bytes_) {
    ssize_t n = ::write(fd, src + written, frame_bytes_ - written);
    if (n <= 0)
      break;
    written += static_cast<size_t>(n);
  }
  ::close(fd);
  if (written == frame_bytes_) {
    has_data_[fi] = true;
  }
#endif
}

Matrix2Df DiskCacheFrameStore::load(size_t fi) const {
  if (fi >= has_data_.size() || !has_data_[fi])
    return Matrix2Df();
  fs::path p = frame_path(fi);
#ifdef _WIN32
  HANDLE hFile = CreateFileW(p.c_str(), GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile == INVALID_HANDLE_VALUE)
    return Matrix2Df();
  HANDLE hMapping = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
  CloseHandle(hFile);
  if (!hMapping)
    return Matrix2Df();
  void *ptr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, frame_bytes_);
  CloseHandle(hMapping);
  if (!ptr)
    return Matrix2Df();
  Matrix2Df out(rows_, cols_);
  std::memcpy(out.data(), ptr, frame_bytes_);
  UnmapViewOfFile(ptr);
  return out;
#else
  int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0)
    return Matrix2Df();
  void *ptr = ::mmap(nullptr, frame_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (ptr == MAP_FAILED)
    return Matrix2Df();
  Matrix2Df out(rows_, cols_);
  std::memcpy(out.data(), ptr, frame_bytes_);
  ::munmap(ptr, frame_bytes_);
  return out;
#endif
}

Matrix2Df DiskCacheFrameStore::extract_tile(size_t fi, const Tile &t,
                                            int offset_x,
                                            int offset_y) const {
  if (fi >= has_data_.size() || !has_data_[fi])
    return Matrix2Df();
  fs::path p = frame_path(fi);
#ifdef _WIN32
  HANDLE hFile = CreateFileW(p.c_str(), GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile == INVALID_HANDLE_VALUE)
    return Matrix2Df();
  HANDLE hMapping = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
  CloseHandle(hFile);
  if (!hMapping)
    return Matrix2Df();
  void *ptr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, frame_bytes_);
  CloseHandle(hMapping);
  if (!ptr)
    return Matrix2Df();
#else
  int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0)
    return Matrix2Df();
  void *ptr = ::mmap(nullptr, frame_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (ptr == MAP_FAILED)
    return Matrix2Df();
#endif

  const float *src = static_cast<const float *>(ptr);
  int x0 = std::max(0, t.x + offset_x);
  int y0 = std::max(0, t.y + offset_y);
  int tw = t.width;
  int th = t.height;
  if (x0 + tw > cols_)
    tw = cols_ - x0;
  if (y0 + th > rows_)
    th = rows_ - y0;
  if (tw <= 0 || th <= 0) {
#ifdef _WIN32
    UnmapViewOfFile(ptr);
#else
    ::munmap(ptr, frame_bytes_);
#endif
    return Matrix2Df();
  }

  Matrix2Df tile(th, tw);
  for (int r = 0; r < th; ++r) {
    const float *row_src = src + static_cast<size_t>(y0 + r) *
                                     static_cast<size_t>(cols_) +
                           static_cast<size_t>(x0);
    float *row_dst =
        tile.data() + static_cast<size_t>(r) * static_cast<size_t>(tw);
    std::memcpy(row_dst, row_src, static_cast<size_t>(tw) * sizeof(float));
  }
#ifdef _WIN32
  UnmapViewOfFile(ptr);
#else
  ::munmap(ptr, frame_bytes_);
#endif
  return tile;
}

bool DiskCacheFrameStore::has_data(size_t fi) const {
  return fi < has_data_.size() && has_data_[fi];
}

size_t DiskCacheFrameStore::size() const { return has_data_.size(); }

int DiskCacheFrameStore::rows() const { return rows_; }

int DiskCacheFrameStore::cols() const { return cols_; }

void DiskCacheFrameStore::cleanup() {
  if (!cache_dir_.empty() && fs::exists(cache_dir_)) {
    std::error_code ec;
    fs::remove_all(cache_dir_, ec);
  }
  has_data_.clear();
}

fs::path DiskCacheFrameStore::frame_path(size_t fi) const {
  return cache_dir_ / (std::to_string(fi) + ".raw");
}

} // namespace tile_compile::runner
