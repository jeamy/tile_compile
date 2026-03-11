#include "runner_shared.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>

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

void unmap_view(void *ptr, size_t bytes) {
  if (ptr == nullptr) {
    return;
  }
#ifdef _WIN32
  UnmapViewOfFile(ptr);
#else
  ::munmap(ptr, bytes);
#endif
}

size_t sample_average_file_bytes(const std::vector<fs::path> &paths,
                                 size_t max_samples) {
  if (paths.empty() || max_samples == 0) {
    return 0;
  }
  const size_t sample_count = std::min(max_samples, paths.size());
  const size_t stride = std::max<size_t>(1, paths.size() / sample_count);
  uint64_t total = 0;
  size_t used = 0;
  for (size_t i = 0; i < paths.size() && used < sample_count; i += stride) {
    std::error_code ec;
    const auto sz = fs::file_size(paths[i], ec);
    if (ec || sz <= 0) {
      continue;
    }
    total += static_cast<uint64_t>(sz);
    ++used;
  }
  if (used == 0 && !paths.empty()) {
    std::error_code ec;
    const auto sz = fs::file_size(paths.front(), ec);
    if (!ec && sz > 0) {
      return static_cast<size_t>(sz);
    }
  }
  return (used > 0) ? static_cast<size_t>(total / used) : 0;
}

int cap_workers_for_io_profile(size_t avg_frame_bytes, size_t task_count,
                               WorkerParallelProfile profile) {
  constexpr size_t MiB = 1024u * 1024u;

  int io_cap = std::numeric_limits<int>::max();
  if (profile == WorkerParallelProfile::IoHeavy) {
    if (avg_frame_bytes >= 96u * MiB) {
      io_cap = 2;
    } else if (avg_frame_bytes >= 64u * MiB) {
      io_cap = 3;
    } else if (avg_frame_bytes >= 32u * MiB) {
      io_cap = 4;
    } else if (avg_frame_bytes >= 16u * MiB) {
      io_cap = 6;
    }
  } else if (profile == WorkerParallelProfile::MixedIo) {
    if (avg_frame_bytes >= 96u * MiB) {
      io_cap = 3;
    } else if (avg_frame_bytes >= 64u * MiB) {
      io_cap = 4;
    } else if (avg_frame_bytes >= 32u * MiB) {
      io_cap = 6;
    } else if (avg_frame_bytes >= 16u * MiB) {
      io_cap = 8;
    }
  }

  if (task_count >= 1000) {
    io_cap = std::min(io_cap,
                      profile == WorkerParallelProfile::IoHeavy ? 4 : 6);
  } else if (task_count >= 400) {
    io_cap = std::min(io_cap,
                      profile == WorkerParallelProfile::IoHeavy ? 5 : 8);
  }
  return io_cap;
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

int compute_adaptive_worker_count(
    const config::Config &cfg, size_t task_count,
    const std::vector<std::filesystem::path> &frames,
    WorkerParallelProfile profile) {
  int workers = cfg.runtime_limits.parallel_workers;
  if (workers < 1) {
    workers = 1;
  }
  const int cpu_cores = static_cast<int>(std::thread::hardware_concurrency());
  if (cpu_cores > 0) {
    workers = std::min(workers, cpu_cores);
  }
  if (task_count > 0) {
    workers =
        std::min(workers, static_cast<int>(std::max<size_t>(1, task_count)));
  }
  workers = std::max(1, workers);
  if (workers <= 1 || profile == WorkerParallelProfile::CpuBound ||
      frames.empty()) {
    return workers;
  }

  const size_t avg_frame_bytes = sample_average_file_bytes(frames, 24);
  if (avg_frame_bytes == 0) {
    return workers;
  }

  const int io_cap =
      cap_workers_for_io_profile(avg_frame_bytes, task_count, profile);
  if (io_cap <= 0 || io_cap == std::numeric_limits<int>::max()) {
    return workers;
  }
  return std::max(1, std::min(workers, io_cap));
}

bool message_indicates_disk_full(const std::string &message) {
  const std::string m = core::to_lower(message);
  return (m.find("no space left on device") != std::string::npos) ||
         (m.find("disk full") != std::string::npos) ||
         (m.find("not enough space") != std::string::npos) ||
         (m.find("enospc") != std::string::npos);
}

bool load_canvas_mask_fits(const fs::path &mask_path, int rows, int cols,
                           std::vector<uint8_t> &out_mask,
                           std::string &error_out) {
  if (rows <= 0 || cols <= 0) {
    error_out = "invalid target image size for canvas mask";
    return false;
  }
  if (!fs::exists(mask_path)) {
    error_out = "missing canvas mask: " + mask_path.string();
    return false;
  }
  try {
    auto mask_pair = tile_compile::io::read_fits_float(mask_path);
    const auto &mask_img = mask_pair.first;
    if (mask_img.rows() != rows || mask_img.cols() != cols) {
      error_out = "canvas mask size mismatch: got " +
                  std::to_string(mask_img.cols()) + "x" +
                  std::to_string(mask_img.rows()) + ", expected " +
                  std::to_string(cols) + "x" + std::to_string(rows);
      return false;
    }

    out_mask.assign(static_cast<size_t>(rows * cols), static_cast<uint8_t>(0));
    int valid_count = 0;
    for (int y = 0; y < rows; ++y) {
      for (int x = 0; x < cols; ++x) {
        if (mask_img(y, x) > 0.5f) {
          out_mask[static_cast<size_t>(y * cols + x)] = 1;
          ++valid_count;
        }
      }
    }
    if (valid_count <= 0) {
      error_out = "canvas mask contains zero valid pixels";
      return false;
    }
    return true;
  } catch (const std::exception &e) {
    error_out = std::string("cannot read canvas mask: ") + e.what();
    return false;
  }
}

bool load_canvas_mask_for_rgb(const fs::path &mask_path, const Matrix2Df &R,
                              const Matrix2Df &G, const Matrix2Df &B,
                              std::vector<uint8_t> &out_mask, int &rows_out,
                              int &cols_out, std::string &error_out) {
  rows_out = 0;
  cols_out = 0;
  if (R.rows() <= 0 || R.cols() <= 0 || R.rows() != G.rows() ||
      R.rows() != B.rows() || R.cols() != G.cols() || R.cols() != B.cols()) {
    error_out = "invalid RGB dimensions";
    return false;
  }
  rows_out = R.rows();
  cols_out = R.cols();
  return load_canvas_mask_fits(mask_path, rows_out, cols_out, out_mask,
                               error_out);
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
      has_data_(n_frames, static_cast<uint8_t>(0)),
      mapped_views_(n_frames, nullptr) {
  fs::create_directories(cache_dir_);
}

DiskCacheFrameStore::~DiskCacheFrameStore() { cleanup(); }

DiskCacheFrameStore::DiskCacheFrameStore(DiskCacheFrameStore &&o) noexcept
    : cache_dir_(std::move(o.cache_dir_)), rows_(o.rows_), cols_(o.cols_),
      frame_bytes_(o.frame_bytes_), has_data_(std::move(o.has_data_)),
      mapped_views_(std::move(o.mapped_views_)) {
  o.rows_ = 0;
  o.cols_ = 0;
  o.frame_bytes_ = 0;
  o.cache_dir_.clear();
  o.has_data_.clear();
  o.mapped_views_.clear();
}

DiskCacheFrameStore &DiskCacheFrameStore::operator=(DiskCacheFrameStore &&o) noexcept {
  if (this != &o) {
    cleanup();
    cache_dir_ = std::move(o.cache_dir_);
    rows_ = o.rows_;
    cols_ = o.cols_;
    frame_bytes_ = o.frame_bytes_;
    has_data_ = std::move(o.has_data_);
    mapped_views_ = std::move(o.mapped_views_);
    o.rows_ = 0;
    o.cols_ = 0;
    o.frame_bytes_ = 0;
    o.cache_dir_.clear();
    o.has_data_.clear();
    o.mapped_views_.clear();
  }
  return *this;
}

void DiskCacheFrameStore::store(size_t fi, const Matrix2Df &frame) {
  if (fi >= has_data_.size()) {
    return;
  }
  if (frame.rows() != rows_ || frame.cols() != cols_) {
    std::cout << "[DiskCache] Frame " << fi << " size mismatch: got " 
              << frame.rows() << "x" << frame.cols() << ", expected " 
              << rows_ << "x" << cols_ << std::endl;
    has_data_[fi] = static_cast<uint8_t>(0);
    return;
  }
  // Drop a stale mapping before rewriting the file.
  invalidate_mapping(fi);
  fs::path p = frame_path(fi);
#ifdef _WIN32
  HANDLE hFile = CreateFileW(p.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile == INVALID_HANDLE_VALUE) {
    has_data_[fi] = static_cast<uint8_t>(0);
    return;
  }
  DWORD written = 0;
  const char *src = reinterpret_cast<const char *>(frame.data());
  WriteFile(hFile, src, static_cast<DWORD>(frame_bytes_), &written, NULL);
  CloseHandle(hFile);
  if (written == frame_bytes_) {
    has_data_[fi] = static_cast<uint8_t>(1);
  } else {
    has_data_[fi] = static_cast<uint8_t>(0);
  }
#else
  int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) {
    has_data_[fi] = static_cast<uint8_t>(0);
    return;
  }
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
    has_data_[fi] = static_cast<uint8_t>(1);
  } else {
    has_data_[fi] = static_cast<uint8_t>(0);
  }
#endif
}

Matrix2Df DiskCacheFrameStore::load(size_t fi) const {
  const float *src = mapped_frame_ptr(fi);
  if (src == nullptr)
    return Matrix2Df();
  Matrix2Df out(rows_, cols_);
  std::memcpy(out.data(), src, frame_bytes_);
  return out;
}

const float *DiskCacheFrameStore::frame_data(size_t fi) const {
  return mapped_frame_ptr(fi);
}

Matrix2Df DiskCacheFrameStore::extract_tile(size_t fi, const Tile &t,
                                            int offset_x,
                                            int offset_y) const {
  Matrix2Df tile;
  if (!extract_tile_into(fi, t, tile, offset_x, offset_y)) {
    return Matrix2Df();
  }
  return tile;
}

bool DiskCacheFrameStore::extract_tile_into(size_t fi, const Tile &t,
                                            Matrix2Df &out, int offset_x,
                                            int offset_y) const {
  const float *src = mapped_frame_ptr(fi);
  if (src == nullptr) {
    out.resize(0, 0);
    return false;
  }
  int x0 = std::max(0, t.x + offset_x);
  int y0 = std::max(0, t.y + offset_y);
  int tw = t.width;
  int th = t.height;
  if (x0 + tw > cols_)
    tw = cols_ - x0;
  if (y0 + th > rows_)
    th = rows_ - y0;
  if (tw <= 0 || th <= 0) {
    out.resize(0, 0);
    return false;
  }

  if (out.rows() != th || out.cols() != tw) {
    out.resize(th, tw);
  }
  for (int r = 0; r < th; ++r) {
    const float *row_src = src + static_cast<size_t>(y0 + r) *
                                     static_cast<size_t>(cols_) +
                           static_cast<size_t>(x0);
    float *row_dst =
        out.data() + static_cast<size_t>(r) * static_cast<size_t>(tw);
    std::memcpy(row_dst, row_src, static_cast<size_t>(tw) * sizeof(float));
  }
  return true;
}

const float *DiskCacheFrameStore::mapped_frame_ptr(size_t fi) const {
  if (fi >= has_data_.size() || has_data_[fi] == 0) {
    return nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(mapped_mutex_);
    if (fi < mapped_views_.size() && mapped_views_[fi] != nullptr) {
      return static_cast<const float *>(mapped_views_[fi]);
    }
  }

  fs::path p = frame_path(fi);
  void *new_view = nullptr;
#ifdef _WIN32
  HANDLE hFile = CreateFileW(p.c_str(), GENERIC_READ, 0, NULL, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile == INVALID_HANDLE_VALUE) {
    return nullptr;
  }
  HANDLE hMapping = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
  if (!hMapping) {
    CloseHandle(hFile);
    return nullptr;
  }
  new_view = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, frame_bytes_);
  CloseHandle(hMapping);
  CloseHandle(hFile);
  if (!new_view) {
    return nullptr;
  }
#else
  int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) {
    return nullptr;
  }
  new_view = ::mmap(nullptr, frame_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (new_view == MAP_FAILED) {
    return nullptr;
  }
#endif

  void *existing_view = nullptr;
  {
    std::lock_guard<std::mutex> lock(mapped_mutex_);
    if (fi >= mapped_views_.size()) {
      unmap_view(new_view, frame_bytes_);
      return nullptr;
    }
    if (mapped_views_[fi] == nullptr) {
      mapped_views_[fi] = new_view;
      return static_cast<const float *>(new_view);
    }
    existing_view = mapped_views_[fi];
  }

  // Another worker installed the mapping first.
  unmap_view(new_view, frame_bytes_);
  return static_cast<const float *>(existing_view);
}

void DiskCacheFrameStore::invalidate_mapping(size_t fi) {
  void *view = nullptr;
  {
    std::lock_guard<std::mutex> lock(mapped_mutex_);
    if (fi >= mapped_views_.size()) {
      return;
    }
    view = mapped_views_[fi];
    mapped_views_[fi] = nullptr;
  }
  unmap_view(view, frame_bytes_);
}

void DiskCacheFrameStore::clear_mappings() {
  std::vector<void *> views;
  {
    std::lock_guard<std::mutex> lock(mapped_mutex_);
    views.swap(mapped_views_);
  }
  for (void *view : views) {
    unmap_view(view, frame_bytes_);
  }
}

bool DiskCacheFrameStore::has_data(size_t fi) const {
  return fi < has_data_.size() && has_data_[fi] != 0;
}

size_t DiskCacheFrameStore::size() const { return has_data_.size(); }

int DiskCacheFrameStore::rows() const { return rows_; }

int DiskCacheFrameStore::cols() const { return cols_; }

void DiskCacheFrameStore::cleanup() {
  clear_mappings();
  if (!cache_dir_.empty() && fs::exists(cache_dir_)) {
    std::error_code ec;
    fs::remove_all(cache_dir_, ec);
  }
  has_data_.clear();
  cache_dir_.clear();
  rows_ = 0;
  cols_ = 0;
  frame_bytes_ = 0;
}

fs::path DiskCacheFrameStore::frame_path(size_t fi) const {
  return cache_dir_ / (std::to_string(fi) + ".raw");
}

} // namespace tile_compile::runner
