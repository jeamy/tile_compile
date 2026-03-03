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

namespace {

core::json bge_value_stats_to_json(const image::BGEValueStats &s) {
  return core::json{{"n", s.n},
                    {"min", s.min},
                    {"max", s.max},
                    {"median", s.median},
                    {"mean", s.mean},
                    {"std", s.std}};
}

} // namespace

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
  dst.min_valid_sample_fraction_for_apply =
      src.min_valid_sample_fraction_for_apply;
  dst.min_valid_samples_for_apply = src.min_valid_samples_for_apply;
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
  dst.apply_attenuation = src.apply_attenuation;
  dst.chroma_strength = src.chroma_strength;
  dst.k_max = src.k_max;
  return dst;
}

core::json bge_diag_to_json(const image::BGEDiagnostics &diag,
                            bool requested,
                            bool have_tile_data,
                            bool metrics_tiles_match) {
  core::json out;
  out["requested"] = requested;
  out["attempted"] = diag.attempted;
  out["success"] = diag.success;
  out["have_tile_data"] = have_tile_data;
  out["metrics_tiles_match"] = metrics_tiles_match;
  out["image_width"] = diag.image_width;
  out["image_height"] = diag.image_height;
  out["grid_spacing"] = diag.grid_spacing;
  out["method"] = diag.method;
  out["robust_loss"] = diag.robust_loss;
  out["insufficient_cell_strategy"] = diag.insufficient_cell_strategy;
  out["autotune"] = {
      {"enabled", diag.autotune_enabled},
      {"strategy", diag.autotune_strategy},
      {"max_evals", diag.autotune_max_evals},
      {"evals_performed", diag.autotune_evals},
      {"fallback_used", diag.autotune_fallback_used},
      {"best",
       {
           {"fit_method", diag.autotune_selected_fit_method},
           {"sample_quantile", diag.autotune_selected_sample_quantile},
           {"structure_thresh_percentile",
            diag.autotune_selected_structure_thresh_percentile},
           {"rbf_mu_factor", diag.autotune_selected_rbf_mu_factor},
           {"objective", diag.autotune_best_objective},
           {"objective_raw", diag.autotune_best_objective_raw},
           {"objective_normalized", diag.autotune_best_objective_normalized},
           {"cv_rms", diag.autotune_best_cv_rms},
           {"flatness", diag.autotune_best_flatness},
           {"roughness", diag.autotune_best_roughness},
       }},
      // Backward-compatible flat aliases
      {"evals", diag.autotune_evals},
      {"best_objective", diag.autotune_best_objective},
      {"best_objective_raw", diag.autotune_best_objective_raw},
      {"best_objective_normalized", diag.autotune_best_objective_normalized},
      {"best_cv_rms", diag.autotune_best_cv_rms},
      {"best_flatness", diag.autotune_best_flatness},
      {"best_roughness", diag.autotune_best_roughness},
      {"selected_fit_method", diag.autotune_selected_fit_method},
      {"selected_sample_quantile", diag.autotune_selected_sample_quantile},
      {"selected_structure_thresh_percentile",
       diag.autotune_selected_structure_thresh_percentile},
      {"selected_rbf_mu_factor", diag.autotune_selected_rbf_mu_factor},
  };
  out["safety_fallback"] = {
      {"triggered", diag.safety_fallback_triggered},
      {"method", diag.safety_fallback_method},
      {"reason", diag.safety_fallback_reason},
  };
  out["channels"] = core::json::array();

  int channels_applied = 0;
  int channels_fit_success = 0;
  int tile_samples_valid_total = 0;
  int tile_samples_total_total = 0;
  int grid_cells_valid_total = 0;

  for (const auto &ch : diag.channels) {
    if (ch.applied)
      ++channels_applied;
    if (ch.fit_success)
      ++channels_fit_success;
    tile_samples_valid_total += ch.tile_samples_valid;
    tile_samples_total_total += ch.tile_samples_total;
    grid_cells_valid_total += ch.grid_cells_valid;

    core::json ch_json;
    ch_json["channel"] = ch.channel_name;
    ch_json["applied"] = ch.applied;
    ch_json["fit_success"] = ch.fit_success;
    ch_json["autotune"] = {
        {"enabled", ch.autotune_enabled},
        {"evals_performed", ch.autotune_evals},
        {"fallback_used", ch.autotune_fallback_used},
        {"selected_fit_method", ch.autotune_selected_fit_method},
        {"selected_grid_spacing", ch.autotune_selected_grid_spacing},
        {"best",
         {
             {"fit_method", ch.autotune_selected_fit_method},
             {"sample_quantile", ch.autotune_selected_sample_quantile},
             {"structure_thresh_percentile",
              ch.autotune_selected_structure_thresh_percentile},
             {"rbf_mu_factor", ch.autotune_selected_rbf_mu_factor},
             {"objective", ch.autotune_best_objective},
             {"objective_raw", ch.autotune_best_objective_raw},
             {"objective_normalized", ch.autotune_best_objective_normalized},
             {"cv_rms", ch.autotune_best_cv_rms},
             {"flatness", ch.autotune_best_flatness},
             {"roughness", ch.autotune_best_roughness},
         }},
    };
    ch_json["tile_samples_total"] = ch.tile_samples_total;
    ch_json["tile_samples_valid"] = ch.tile_samples_valid;
    ch_json["grid_cells_valid"] = ch.grid_cells_valid;
    ch_json["fit_rms_residual"] = ch.fit_rms_residual;
    ch_json["mean_shift"] = ch.mean_shift;
    ch_json["guard_flat_pre"] = ch.guard_flat_pre;
    ch_json["guard_flat_post"] = ch.guard_flat_post;
    ch_json["guard_slope_pre"] = ch.guard_slope_pre;
    ch_json["guard_slope_post"] = ch.guard_slope_post;
    ch_json["guard_rejected"] = ch.guard_rejected;
    ch_json["input_stats"] = bge_value_stats_to_json(ch.input_stats);
    ch_json["output_stats"] = bge_value_stats_to_json(ch.output_stats);
    ch_json["model_stats"] = bge_value_stats_to_json(ch.model_stats);
    ch_json["sample_bg_stats"] = bge_value_stats_to_json(ch.sample_bg_stats);
    ch_json["sample_weight_stats"] = bge_value_stats_to_json(ch.sample_weight_stats);
    ch_json["residual_stats"] = bge_value_stats_to_json(ch.residual_stats);
    ch_json["sample_bg_values"] = ch.sample_bg_values;
    ch_json["sample_weight_values"] = ch.sample_weight_values;
    ch_json["residual_values"] = ch.residual_values;
    ch_json["grid_cells"] = core::json::array();
    for (const auto &gc : ch.grid_cells) {
      ch_json["grid_cells"].push_back({
          {"cell_x", gc.cell_x},
          {"cell_y", gc.cell_y},
          {"center_x", gc.center_x},
          {"center_y", gc.center_y},
          {"bg_value", gc.bg_value},
          {"weight", gc.weight},
          {"n_samples", gc.n_samples},
          {"valid", gc.valid},
      });
    }

    out["channels"].push_back(std::move(ch_json));
  }

  out["summary"] = {
      {"channels_total", static_cast<int>(diag.channels.size())},
      {"channels_applied", channels_applied},
      {"channels_fit_success", channels_fit_success},
      {"tile_samples_total", tile_samples_total_total},
      {"tile_samples_valid", tile_samples_valid_total},
      {"grid_cells_valid", grid_cells_valid_total},
  };
  return out;
}

PCCCatalogQueryResult query_pcc_catalog_stars(const astrometry::WCS &wcs,
                                              const config::PCCConfig &cfg,
                                              std::ostream &log_stream,
                                              const std::string &log_prefix) {
  PCCCatalogQueryResult out;
  const double search_r = wcs.search_radius_deg();

  auto try_siril = [&]() -> bool {
    std::string cat_dir = cfg.siril_catalog_dir;
    if (cat_dir.empty()) {
      cat_dir = astrometry::default_siril_gaia_catalog_dir();
    }
    if (!astrometry::is_siril_gaia_catalog_available(cat_dir)) {
      return false;
    }
    log_stream << log_prefix << " Querying Siril Gaia catalog at RA="
               << wcs.crval1 << " Dec=" << wcs.crval2
               << " r=" << search_r << " deg" << std::endl;
    out.stars = astrometry::siril_gaia_cone_search(
        cat_dir, wcs.crval1, wcs.crval2, search_r, cfg.mag_limit);
    if (!out.stars.empty()) {
      out.used_source = "siril";
      return true;
    }
    return false;
  };

  auto try_vizier_gaia = [&]() -> bool {
    log_stream << log_prefix << " Querying VizieR Gaia DR3 at RA="
               << wcs.crval1 << " Dec=" << wcs.crval2
               << " r=" << search_r << " deg" << std::endl;
    out.stars = astrometry::vizier_gaia_cone_search(
        wcs.crval1, wcs.crval2, search_r, cfg.mag_limit);
    if (!out.stars.empty()) {
      out.used_source = "vizier_gaia";
      return true;
    }
    return false;
  };

  auto try_vizier_apass = [&]() -> bool {
    log_stream << log_prefix << " Querying VizieR APASS DR9 at RA="
               << wcs.crval1 << " Dec=" << wcs.crval2
               << " r=" << search_r << " deg" << std::endl;
    out.stars = astrometry::vizier_apass_cone_search(
        wcs.crval1, wcs.crval2, search_r, cfg.mag_limit);
    if (!out.stars.empty()) {
      out.used_source = "vizier_apass";
      return true;
    }
    return false;
  };

  if (cfg.source == "siril") {
    try_siril();
  } else if (cfg.source == "vizier_gaia") {
    try_vizier_gaia();
  } else if (cfg.source == "vizier_apass") {
    try_vizier_apass();
  } else {
    if (!try_siril()) {
      log_stream << log_prefix
                 << " Siril catalog not available, trying VizieR Gaia..."
                 << std::endl;
      if (!try_vizier_gaia()) {
        log_stream << log_prefix
                   << " VizieR Gaia failed, trying VizieR APASS..."
                   << std::endl;
        try_vizier_apass();
      }
    }
  }

  log_stream << log_prefix << " Found " << out.stars.size() << " catalog stars"
             << " (source: " << (out.used_source.empty() ? "none" : out.used_source)
             << ")" << std::endl;
  return out;
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
    std::cout << "[DiskCache] Frame " << fi << " size mismatch: got " 
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
