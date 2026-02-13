#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/mode_gating.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/image/processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/metrics/linearity.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/metrics/tile_metrics.hpp"
#include "tile_compile/pipeline/adaptive_tile_grid.hpp"
#include "tile_compile/reconstruction/reconstruction.hpp"
#include "tile_compile/registration/global_registration.hpp"
#include "tile_compile/registration/registration.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/astrometry/gaia_catalog.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"

#include <QCoreApplication>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <limits>
#include <opencv2/opencv.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef HAVE_CLI11
#include <CLI/CLI.hpp>
#endif

namespace fs = std::filesystem;

namespace {

using tile_compile::ColorMode;
using tile_compile::Matrix2Df;
using tile_compile::Tile;
using tile_compile::WarpMatrix;

namespace core = tile_compile::core;
namespace image = tile_compile::image;
namespace metrics = tile_compile::metrics;
namespace reconstruction = tile_compile::reconstruction;
namespace registration = tile_compile::registration;
namespace astro = tile_compile::astrometry;

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

class TeeBuf : public std::streambuf {
public:
  TeeBuf(std::streambuf *a, std::streambuf *b) : a_(a), b_(b) {}

protected:
  int overflow(int c) override {
    if (c == EOF)
      return EOF;
    const int ra = a_ ? a_->sputc(static_cast<char>(c)) : c;
    const int rb = b_ ? b_->sputc(static_cast<char>(c)) : c;
    return (ra == EOF || rb == EOF) ? EOF : c;
  }

  int sync() override {
    int ra = a_ ? a_->pubsync() : 0;
    int rb = b_ ? b_->pubsync() : 0;
    return (ra == 0 && rb == 0) ? 0 : -1;
  }

private:
  std::streambuf *a_;
  std::streambuf *b_;
};

using NormalizationScales = image::NormalizationScales;

// Disk-backed frame store using mmap for memory-efficient access to prewarped
// frames.  Frames are written as raw float binaries to a temp directory and
// lazily memory-mapped on read.  The OS page cache handles eviction, keeping
// resident memory bounded by available RAM rather than total dataset size.
class DiskCacheFrameStore {
public:
  DiskCacheFrameStore() = default;

  DiskCacheFrameStore(const fs::path &cache_dir, size_t n_frames,
                      int rows, int cols)
      : cache_dir_(cache_dir), rows_(rows), cols_(cols),
        frame_bytes_(static_cast<size_t>(rows) * static_cast<size_t>(cols) *
                     sizeof(float)),
        has_data_(n_frames, false) {
    fs::create_directories(cache_dir_);
  }

  ~DiskCacheFrameStore() { cleanup(); }

  DiskCacheFrameStore(const DiskCacheFrameStore &) = delete;
  DiskCacheFrameStore &operator=(const DiskCacheFrameStore &) = delete;
  DiskCacheFrameStore(DiskCacheFrameStore &&o) noexcept
      : cache_dir_(std::move(o.cache_dir_)), rows_(o.rows_), cols_(o.cols_),
        frame_bytes_(o.frame_bytes_), has_data_(std::move(o.has_data_)) {}
  DiskCacheFrameStore &operator=(DiskCacheFrameStore &&o) noexcept {
    if (this != &o) {
      cleanup();
      cache_dir_ = std::move(o.cache_dir_);
      rows_ = o.rows_; cols_ = o.cols_;
      frame_bytes_ = o.frame_bytes_;
      has_data_ = std::move(o.has_data_);
    }
    return *this;
  }

  // Write a prewarped frame to disk.
  void store(size_t fi, const Matrix2Df &frame) {
    if (frame.rows() != rows_ || frame.cols() != cols_) return;
    fs::path p = frame_path(fi);
    int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return;
    size_t written = 0;
    const char *src = reinterpret_cast<const char *>(frame.data());
    while (written < frame_bytes_) {
      ssize_t n = ::write(fd, src + written, frame_bytes_ - written);
      if (n <= 0) break;
      written += static_cast<size_t>(n);
    }
    ::close(fd);
    if (written == frame_bytes_) {
      has_data_[fi] = true;
    }
  }

  // Read a full frame via mmap, copy into Matrix2Df, munmap immediately.
  Matrix2Df load(size_t fi) const {
    if (fi >= has_data_.size() || !has_data_[fi]) return Matrix2Df();
    fs::path p = frame_path(fi);
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) return Matrix2Df();
    void *ptr = ::mmap(nullptr, frame_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (ptr == MAP_FAILED) return Matrix2Df();
    Matrix2Df out(rows_, cols_);
    std::memcpy(out.data(), ptr, frame_bytes_);
    ::munmap(ptr, frame_bytes_);
    return out;
  }

  // Extract a tile ROI directly from mmap without loading the full frame.
  Matrix2Df extract_tile(size_t fi, const Tile &t) const {
    if (fi >= has_data_.size() || !has_data_[fi]) return Matrix2Df();
    fs::path p = frame_path(fi);
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) return Matrix2Df();
    void *ptr = ::mmap(nullptr, frame_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (ptr == MAP_FAILED) return Matrix2Df();

    const float *src = static_cast<const float *>(ptr);
    int x0 = std::max(0, t.x);
    int y0 = std::max(0, t.y);
    int tw = t.width;
    int th = t.height;
    if (x0 + tw > cols_) tw = cols_ - x0;
    if (y0 + th > rows_) th = rows_ - y0;
    if (tw <= 0 || th <= 0) { ::munmap(ptr, frame_bytes_); return Matrix2Df(); }

    Matrix2Df tile(th, tw);
    for (int r = 0; r < th; ++r) {
      const float *row_src = src + static_cast<size_t>(y0 + r) *
                                       static_cast<size_t>(cols_) +
                                   static_cast<size_t>(x0);
      float *row_dst = tile.data() + static_cast<size_t>(r) *
                                         static_cast<size_t>(tw);
      std::memcpy(row_dst, row_src, static_cast<size_t>(tw) * sizeof(float));
    }
    ::munmap(ptr, frame_bytes_);
    return tile;
  }

  bool has_data(size_t fi) const {
    return fi < has_data_.size() && has_data_[fi];
  }

  size_t size() const { return has_data_.size(); }
  int rows() const { return rows_; }
  int cols() const { return cols_; }

  void cleanup() {
    if (!cache_dir_.empty() && fs::exists(cache_dir_)) {
      std::error_code ec;
      fs::remove_all(cache_dir_, ec);
    }
    has_data_.clear();
  }

private:
  fs::path frame_path(size_t fi) const {
    return cache_dir_ / (std::to_string(fi) + ".raw");
  }

  fs::path cache_dir_;
  int rows_ = 0;
  int cols_ = 0;
  size_t frame_bytes_ = 0;
  std::vector<bool> has_data_;
};

} // namespace

void print_usage() {
  std::cout << "Usage: tile_compile_runner <command> [options]\n\n"
            << "Commands:\n"
            << "  run      Run the pipeline\n"
            << "  resume   Resume a run from a specific phase\n"
            << "\nOptions:\n"
            << "  --config <path>       Path to config.yaml (run)\n"
            << "  --input-dir <path>    Input directory with FITS frames (run)\n"
            << "  --runs-dir <path>     Directory for run outputs (run)\n"
            << "  --project-root <path> Project root directory (run)\n"
            << "  --run-dir <path>      Existing run directory (resume)\n"
            << "  --from-phase <name>   Phase to resume from (resume; default: PCC)\n"
            << "  --max-frames <n>      Limit number of frames (0 = no limit)\n"
            << "  --max-tiles <n>       Limit number of tiles in Phase 5/6 (0 "
               "= no limit)\n"
            << "  --dry-run             Dry run (no actual processing)\n"
            << std::endl;
}

int run_command(const std::string &config_path, const std::string &input_dir,
                const std::string &runs_dir, const std::string &project_root,
                bool dry_run, int max_frames, int max_tiles,
                bool config_from_stdin) {
  using namespace tile_compile;

  fs::path cfg_path(config_path);
  fs::path in_dir(input_dir);
  fs::path runs(runs_dir);

  const bool use_stdin_config = config_from_stdin || (config_path == "-");
  fs::path proj_root;

  if (!fs::exists(in_dir)) {
    std::cerr << "Error: Input directory not found: " << input_dir << std::endl;
    return 1;
  }

  config::Config cfg;
  std::string cfg_text;
  if (use_stdin_config) {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    cfg_text = ss.str();
    if (cfg_text.empty()) {
      std::cerr << "Error: --stdin provided but no config YAML received"
                << std::endl;
      return 1;
    }
    YAML::Node node = YAML::Load(cfg_text);
    cfg = config::Config::from_yaml(node);
    cfg.validate();
    proj_root =
        project_root.empty() ? fs::current_path() : fs::path(project_root);
  } else {
    if (!fs::exists(cfg_path)) {
      std::cerr << "Error: Config file not found: " << config_path << std::endl;
      return 1;
    }
    cfg = config::Config::load(cfg_path);
    cfg.validate();
    proj_root = project_root.empty() ? core::resolve_project_root(cfg_path)
                                     : fs::path(project_root);
  }

  auto frames = core::discover_frames(in_dir, "*.fit*");
  std::sort(frames.begin(), frames.end());
  if (max_frames > 0 && frames.size() > static_cast<size_t>(max_frames)) {
    frames.resize(static_cast<size_t>(max_frames));
  }
  if (frames.empty()) {
    std::cerr << "Error: No FITS frames found in " << input_dir << std::endl;
    return 1;
  }

  std::string run_id = core::get_run_id();
  fs::path run_dir = runs / run_id;
  fs::create_directories(run_dir / "logs");
  fs::create_directories(run_dir / "outputs");
  fs::create_directories(run_dir / "artifacts");

  if (use_stdin_config) {
    std::ofstream out(run_dir / "config.yaml", std::ios::out);
    out << cfg_text;
  } else {
    core::copy_config(cfg_path, run_dir / "config.yaml");
  }

  std::ofstream event_log_file(run_dir / "logs" / "run_events.jsonl");
  TeeBuf tee_buf(std::cout.rdbuf(), event_log_file.rdbuf());
  std::ostream log_file(&tee_buf);

  core::EventEmitter emitter;
  emitter.run_start(run_id,
                    {{"config_path", config_path},
                     {"input_dir", input_dir},
                     {"run_dir", run_dir.string()},
                     {"frames_discovered", frames.size()},
                     {"dry_run", dry_run}},
                    log_file);

  std::cout << "Run ID: " << run_id << std::endl;
  std::cout << "Frames: " << frames.size() << std::endl;
  std::cout << "Output: " << run_dir.string() << std::endl;
  if (max_tiles > 0) {
    std::cout << "Max tiles (Phase 5/6): " << max_tiles << std::endl;
  }

  if (dry_run) {
    emitter.phase_start(run_id, Phase::SCAN_INPUT, "SCAN_INPUT", log_file);
    emitter.phase_end(run_id, Phase::SCAN_INPUT, "skipped",
                      {{"reason", "dry_run"}, {"input_dir", input_dir}},
                      log_file);

    std::cout << "Dry run - no processing" << std::endl;
    emitter.run_end(run_id, true, "ok", log_file);
    return 0;
  }

  // Phase 0: SCAN_INPUT (// Methodik v3)
  emitter.phase_start(run_id, Phase::SCAN_INPUT, "SCAN_INPUT", log_file);

  int width = 0;
  int height = 0;
  int naxis = 0;
  ColorMode detected_mode = ColorMode::MONO;
  BayerPattern detected_bayer = BayerPattern::UNKNOWN;
  Matrix2Df first_frame;
  io::FitsHeader first_header;

  try {
    std::tie(width, height, naxis) = io::get_fits_dimensions(frames.front());
    auto first = io::read_fits_float(frames.front());
    first_frame = std::move(first.first);
    first_header = std::move(first.second);

    detected_mode = io::detect_color_mode(first_header, naxis);
    detected_bayer = io::detect_bayer_pattern(first_header);
  } catch (const std::exception &e) {
    emitter.phase_end(run_id, Phase::SCAN_INPUT, "error",
                      {{"error", e.what()}, {"input_dir", input_dir}},
                      log_file);
    emitter.run_end(run_id, false, "error", log_file);
    std::cerr << "Error during SCAN_INPUT: " << e.what() << std::endl;
    return 1;
  }

  std::string detected_mode_str = color_mode_to_string(detected_mode);
  std::string detected_bayer_str = bayer_pattern_to_string(detected_bayer);

  if (!cfg.data.color_mode.empty() &&
      cfg.data.color_mode != detected_mode_str) {
    emitter.warning(run_id,
                    "Detected color mode '" + detected_mode_str +
                        "' differs from config.data.color_mode '" +
                        cfg.data.color_mode + "'",
                    log_file);
  }
  if (!cfg.data.bayer_pattern.empty() && detected_mode == ColorMode::OSC &&
      cfg.data.bayer_pattern != detected_bayer_str &&
      detected_bayer != BayerPattern::UNKNOWN) {
    emitter.warning(run_id,
                    "Detected bayer pattern '" + detected_bayer_str +
                        "' differs from config.data.bayer_pattern '" +
                        cfg.data.bayer_pattern + "'",
                    log_file);
  }

  core::json linearity_info;
  std::vector<size_t> rejected_indices;
  if (cfg.linearity.enabled || cfg.data.linear_required) {
    auto indices = core::sample_indices(frames.size(), cfg.linearity.max_frames);
    int failed = 0;
    float score_sum = 0.0f;
    std::vector<std::string> failed_names;
    for (size_t idx : indices) {
      Matrix2Df frame_img;
      if (idx == 0) {
        frame_img = first_frame;
      } else {
        frame_img = io::read_fits_float(frames[idx]).first;
      }
      metrics::LinearityFrameResult res =
          metrics::validate_linearity_frame(frame_img, cfg.linearity.strictness);
      score_sum += res.is_linear ? 1.0f : 0.0f;
      if (!res.is_linear) {
        failed++;
        rejected_indices.push_back(idx);
        if (failed_names.size() < 5) {
          failed_names.push_back(frames[idx].filename().string());
        }
      }
    }

    float overall_linearity =
        indices.empty() ? 0.0f
                        : (score_sum / static_cast<float>(indices.size()));
    linearity_info["enabled"] = true;
    linearity_info["sampled_frames"] = static_cast<int>(indices.size());
    linearity_info["overall_linearity"] = overall_linearity;
    linearity_info["min_overall_linearity"] =
        cfg.linearity.min_overall_linearity;
    linearity_info["failed_frames"] = failed;
    if (!failed_names.empty()) {
      linearity_info["failed_frame_names"] = failed_names;
    }

    if (failed > 0) {
      emitter.warning(
          run_id,
          "Linearity check: " + std::to_string(failed) +
              " sampled frames flagged non-linear (overall_linearity=" +
              std::to_string(overall_linearity) + ")",
          log_file);
    }
  } else {
    emitter.warning(
        run_id,
        "Linearity check disabled by config; continuing without enforcement.",
        log_file);
    linearity_info["enabled"] = false;
  }

  if (!rejected_indices.empty()) {
    std::sort(rejected_indices.begin(), rejected_indices.end());
    rejected_indices.erase(
        std::unique(rejected_indices.begin(), rejected_indices.end()),
        rejected_indices.end());
    linearity_info["flagged_indices"] = core::json::array();
    linearity_info["flagged_names"] = core::json::array();
    for (size_t idx : rejected_indices) {
      linearity_info["flagged_indices"].push_back(static_cast<int>(idx));
      if (idx < frames.size()) {
        linearity_info["flagged_names"].push_back(
            frames[idx].filename().string());
      }
    }

    if (cfg.data.linear_required) {
      // Remove non-linear frames from the pipeline
      std::set<size_t> reject_set(rejected_indices.begin(),
                                  rejected_indices.end());
      std::vector<std::filesystem::path> kept;
      kept.reserve(frames.size() - rejected_indices.size());
      for (size_t i = 0; i < frames.size(); ++i) {
        if (reject_set.find(i) == reject_set.end()) {
          kept.push_back(frames[i]);
        }
      }
      frames = std::move(kept);
      emitter.warning(
          run_id,
          "Linearity: " + std::to_string(rejected_indices.size()) +
              " non-linear frames removed, " +
              std::to_string(frames.size()) + " frames remaining",
          log_file);
      linearity_info["action"] = "removed";
      linearity_info["frames_remaining"] = static_cast<int>(frames.size());

      if (frames.empty()) {
        emitter.phase_end(run_id, Phase::SCAN_INPUT, "error",
                          {{"error", "All frames rejected by linearity check"},
                           {"linearity", linearity_info}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file);
        std::cerr << "Error: All frames rejected by linearity check."
                  << std::endl;
        return 1;
      }
    } else {
      emitter.warning(
          run_id,
          "Linearity: " + std::to_string(rejected_indices.size()) +
              " frames flagged non-linear (kept, linear_required=false)",
          log_file);
      linearity_info["action"] = "warn_only";
    }
  }

  core::json scan_extra = {
      {"input_dir", input_dir},
      {"frames_scanned", frames.size()},
      {"image_width", width},
      {"image_height", height},
      {"color_mode", detected_mode_str},
      {"bayer_pattern", detected_bayer_str},
  };

  {
    const uint64_t scan_dir_bytes = estimate_total_file_bytes(frames);
    const uint64_t required_min_bytes =
        (scan_dir_bytes > std::numeric_limits<uint64_t>::max() / 4)
            ? std::numeric_limits<uint64_t>::max()
            : (scan_dir_bytes * 4ULL);

    std::error_code ec_space;
    const auto space_info = fs::space(runs, ec_space);
    if (!ec_space) {
      const uint64_t available_bytes =
          static_cast<uint64_t>(space_info.available);
      scan_extra["runs_device_available_bytes"] = available_bytes;
      scan_extra["scan_input_total_bytes"] = scan_dir_bytes;
      scan_extra["required_min_bytes_scandir_x4"] = required_min_bytes;

      if (available_bytes < required_min_bytes) {
        const std::string msg =
            "Insufficient disk space on runs device: available=" +
            format_bytes(available_bytes) +
            ", required_min(scandir*4)=" + format_bytes(required_min_bytes);
        emitter.phase_end(run_id, Phase::SCAN_INPUT, "error",
                          {{"error", msg},
                           {"runs_device_available_bytes", available_bytes},
                           {"scan_input_total_bytes", scan_dir_bytes},
                           {"required_min_bytes_scandir_x4", required_min_bytes},
                           {"runs_dir", runs.string()}},
                          log_file);
        emitter.run_end(run_id, false, "insufficient_disk_space", log_file);
        std::cerr << "Error during SCAN_INPUT: " << msg << std::endl;
        return 1;
      }
    } else {
      emitter.warning(run_id,
                      "Disk-space precheck skipped: cannot query free space for " +
                          runs.string() + " (" + ec_space.message() + ")",
                      log_file);
    }
  }

  if (!linearity_info.is_null()) {
    scan_extra["linearity"] = linearity_info;
  }

  emitter.phase_end(run_id, Phase::SCAN_INPUT, "ok", scan_extra, log_file);

  // Phase 1: CHANNEL_SPLIT (metadata-only; actual split happens later)
  emitter.phase_start(run_id, Phase::CHANNEL_SPLIT, "CHANNEL_SPLIT", log_file);

  core::json extra;
  if (detected_mode == ColorMode::OSC) {
    extra["mode"] = "OSC";
    extra["channels"] = core::json::array({"R", "G", "B"});
    extra["bayer_pattern"] = detected_bayer_str;
    extra["note"] = "deferred_to_tile_processing";
  } else {
    extra["mode"] = "MONO";
    extra["channels"] = core::json::array({"L"});
  }
  emitter.phase_end(run_id, Phase::CHANNEL_SPLIT, "ok", extra, log_file);

  // Phase 2: NORMALIZATION (// Methodik v3 §3)
  emitter.phase_start(run_id, Phase::NORMALIZATION, "NORMALIZATION", log_file);

  if (!cfg.normalization.enabled) {
    emitter.phase_end(
        run_id, Phase::NORMALIZATION, "error",
        {{"error", "NORMALIZATION: disabled but required"}}, log_file);
    emitter.run_end(run_id, false, "error", log_file);
    return 1;
  }

  const float eps_b = 1.0e-6f;
  std::vector<NormalizationScales> norm_scales(frames.size());
  std::vector<float> B_mono(frames.size(), 0.0f);
  std::vector<float> B_r(frames.size(), 0.0f);
  std::vector<float> B_g(frames.size(), 0.0f);
  std::vector<float> B_b(frames.size(), 0.0f);

  for (size_t i = 0; i < frames.size(); ++i) {
    const auto &path = frames[i];
    try {
      auto frame_pair = io::read_fits_float(path);
      const Matrix2Df &img = frame_pair.first;

      NormalizationScales s;
      {
        std::vector<float> all;
        all.reserve(static_cast<size_t>(img.size()));
        for (Eigen::Index k = 0; k < img.size(); ++k) {
          all.push_back(img.data()[k]);
        }
        const float b0 = core::median_of(all);
        const float coarse_scale = (b0 > eps_b) ? (1.0f / b0) : 1.0f;
        Matrix2Df coarse_norm = img * coarse_scale;
        cv::Mat coarse_cv(coarse_norm.rows(), coarse_norm.cols(), CV_32F,
                          coarse_norm.data());
        const cv::Mat1b bg_mask =
            metrics::build_background_mask_sigma_clip(coarse_cv, 3.0f, 3);

        if (detected_mode == ColorMode::OSC) {
          s.is_osc = true;
          int r_row, r_col, b_row, b_col;
          image::bayer_offsets(detected_bayer_str, r_row, r_col, b_row, b_col);

          std::vector<float> pr_bg;
          std::vector<float> pg_bg;
          std::vector<float> pb_bg;
          pr_bg.reserve(static_cast<size_t>(img.size()) / 4);
          pg_bg.reserve(static_cast<size_t>(img.size()) / 2);
          pb_bg.reserve(static_cast<size_t>(img.size()) / 4);

          for (int y = 0; y < img.rows(); ++y) {
            const uint8_t *mrow = bg_mask.ptr<uint8_t>(y);
            const int py = y & 1;
            for (int x = 0; x < img.cols(); ++x) {
              if (mrow[x] == 0)
                continue;
              const int px = x & 1;
              const float v = img(y, x);
              if (py == r_row && px == r_col) {
                pr_bg.push_back(v);
              } else if (py == b_row && px == b_col) {
                pb_bg.push_back(v);
              } else {
                pg_bg.push_back(v);
              }
            }
          }

          float br = pr_bg.empty() ? 0.0f : core::median_of(pr_bg);
          float bg = pg_bg.empty() ? 0.0f : core::median_of(pg_bg);
          float bb = pb_bg.empty() ? 0.0f : core::median_of(pb_bg);

          if (!(br > eps_b)) {
            std::vector<float> pr;
            pr.reserve(static_cast<size_t>(img.size()) / 4);
            for (int y = 0; y < img.rows(); ++y) {
              const int py = y & 1;
              for (int x = 0; x < img.cols(); ++x) {
                const int px = x & 1;
                if (py == r_row && px == r_col)
                  pr.push_back(img(y, x));
              }
            }
            br = core::estimate_background_sigma_clip(std::move(pr));
          }
          if (!(bg > eps_b)) {
            std::vector<float> pg;
            pg.reserve(static_cast<size_t>(img.size()) / 2);
            for (int y = 0; y < img.rows(); ++y) {
              const int py = y & 1;
              for (int x = 0; x < img.cols(); ++x) {
                const int px = x & 1;
                if (!((py == r_row && px == r_col) ||
                      (py == b_row && px == b_col)))
                  pg.push_back(img(y, x));
              }
            }
            bg = core::estimate_background_sigma_clip(std::move(pg));
          }
          if (!(bb > eps_b)) {
            std::vector<float> pb;
            pb.reserve(static_cast<size_t>(img.size()) / 4);
            for (int y = 0; y < img.rows(); ++y) {
              const int py = y & 1;
              for (int x = 0; x < img.cols(); ++x) {
                const int px = x & 1;
                if (py == b_row && px == b_col)
                  pb.push_back(img(y, x));
              }
            }
            bb = core::estimate_background_sigma_clip(std::move(pb));
          }

          if (!(br > eps_b) || !(bg > eps_b) || !(bb > eps_b)) {
            emitter.phase_end(
                run_id, Phase::NORMALIZATION, "error",
                {{"error", "NORMALIZATION: invalid background estimate"}},
                log_file);
            emitter.run_end(run_id, false, "error", log_file);
            return 1;
          }

          s.scale_r = 1.0f / br;
          s.scale_g = 1.0f / bg;
          s.scale_b = 1.0f / bb;
          B_r[i] = br;
          B_g[i] = bg;
          B_b[i] = bb;
        } else {
          std::vector<float> p_bg;
          p_bg.reserve(static_cast<size_t>(img.size()));
          for (int y = 0; y < img.rows(); ++y) {
            const uint8_t *mrow = bg_mask.ptr<uint8_t>(y);
            for (int x = 0; x < img.cols(); ++x) {
              if (mrow[x] != 0)
                p_bg.push_back(img(y, x));
            }
          }
          float b = p_bg.empty() ? 0.0f : core::median_of(p_bg);
          if (!(b > eps_b)) {
            std::vector<float> p;
            p.reserve(static_cast<size_t>(img.size()));
            for (Eigen::Index k = 0; k < img.size(); ++k) {
              p.push_back(img.data()[k]);
            }
            b = core::estimate_background_sigma_clip(std::move(p));
          }
          if (!(b > eps_b)) {
            emitter.phase_end(
                run_id, Phase::NORMALIZATION, "error",
                {{"error", "NORMALIZATION: invalid background estimate"}},
                log_file);
            emitter.run_end(run_id, false, "error", log_file);
            return 1;
          }
          s.scale_mono = 1.0f / b;
          B_mono[i] = b;
        }
      }
      norm_scales[i] = s;
    } catch (const std::exception &e) {
      emitter.phase_end(run_id, Phase::NORMALIZATION, "error",
                        {{"error", e.what()}}, log_file);
      emitter.run_end(run_id, false, "error", log_file);
      std::cerr << "Error during NORMALIZATION: " << e.what() << std::endl;
      return 1;
    }

    const float progress =
        frames.empty()
            ? 1.0f
            : static_cast<float>(i + 1) / static_cast<float>(frames.size());
    emitter.phase_progress(run_id, Phase::NORMALIZATION, progress,
                           "normalize " + std::to_string(i + 1) + "/" +
                               std::to_string(frames.size()),
                           log_file);
  }

  {
    core::json artifact;
    artifact["mode"] = (detected_mode == ColorMode::OSC) ? "OSC" : "MONO";
    artifact["bayer_pattern"] = detected_bayer_str;
    artifact["B_mono"] = core::json::array();
    artifact["B_r"] = core::json::array();
    artifact["B_g"] = core::json::array();
    artifact["B_b"] = core::json::array();
    for (size_t i = 0; i < frames.size(); ++i) {
      artifact["B_mono"].push_back(B_mono[i]);
      artifact["B_r"].push_back(B_r[i]);
      artifact["B_g"].push_back(B_g[i]);
      artifact["B_b"].push_back(B_b[i]);
    }
    core::write_text(run_dir / "artifacts" / "normalization.json",
                     artifact.dump(2));
  }

  const float output_pedestal = 0.0f;
  const float output_bg_mono = core::median_finite_positive(B_mono, 1.0f);
  const float output_bg_r = core::median_finite_positive(B_r, 1.0f);
  const float output_bg_g = core::median_finite_positive(B_g, 1.0f);
  const float output_bg_b = core::median_finite_positive(B_b, 1.0f);

  emitter.phase_end(run_id, Phase::NORMALIZATION, "ok",
                    {
                        {"num_frames", static_cast<int>(frames.size())},
                    },
                    log_file);

  // Phase 3: GLOBAL_METRICS
  emitter.phase_start(run_id, Phase::GLOBAL_METRICS, "GLOBAL_METRICS",
                      log_file);

  std::vector<FrameMetrics> frame_metrics;
  frame_metrics.resize(frames.size());
  std::vector<metrics::FrameStarMetrics> frame_star_metrics;
  frame_star_metrics.resize(frames.size());
  int ref_star_count = 0;

  // First pass: compute metrics + star metrics for all frames
  for (size_t i = 0; i < frames.size(); ++i) {
    const auto &path = frames[i];
    try {
      auto frame_pair = io::read_fits_float(path);
      Matrix2Df img = frame_pair.first;
      if (img.size() <= 0) {
        emitter.warning(run_id,
                        "GLOBAL_METRICS: empty frame for " +
                            path.filename().string(),
                        log_file);
        FrameMetrics m;
        m.background = 0.0f;
        m.noise = 0.0f;
        m.gradient_energy = 0.0f;
        m.quality_score = 1.0f;
        frame_metrics[i] = m;
        frame_star_metrics[i] = metrics::FrameStarMetrics{};
      } else {
        image::apply_normalization_inplace(img, norm_scales[i], detected_mode,
                                    detected_bayer_str, 0, 0);
        FrameMetrics m = metrics::calculate_frame_metrics(img);
        // Methodik v3: for the global background metric B_f, use the raw
        // (pre-normalization) background estimate from the normalization stage.
        if (detected_mode == ColorMode::OSC) {
          const float b_raw = 0.25f * B_r[i] + 0.5f * B_g[i] + 0.25f * B_b[i];
          if (std::isfinite(b_raw))
            m.background = b_raw;
        } else {
          const float b_raw = B_mono[i];
          if (std::isfinite(b_raw))
            m.background = b_raw;
        }
        frame_metrics[i] = m;
        frame_star_metrics[i] = metrics::measure_frame_stars(img, 0);
      }
    } catch (const std::exception &e) {
      emitter.phase_end(run_id, Phase::GLOBAL_METRICS, "error",
                        {{"error", e.what()}}, log_file);
      emitter.run_end(run_id, false, "error", log_file);
      std::cerr << "Error during GLOBAL_METRICS: " << e.what() << std::endl;
      return 1;
    }

    const float progress =
        frames.empty()
            ? 1.0f
            : static_cast<float>(i + 1) / static_cast<float>(frames.size());
    emitter.phase_progress(run_id, Phase::GLOBAL_METRICS, progress,
                           "metrics " + std::to_string(i + 1) + "/" +
                               std::to_string(frames.size()),
                           log_file);
  }

  // Determine reference star count (max) and recompute wFWHM
  for (const auto &sm : frame_star_metrics) {
    if (sm.star_count > ref_star_count) ref_star_count = sm.star_count;
  }
  if (ref_star_count > 0) {
    for (auto &sm : frame_star_metrics) {
      if (sm.star_count > 0 && sm.fwhm > 0) {
        sm.wfwhm = sm.fwhm * static_cast<float>(ref_star_count) /
                   static_cast<float>(sm.star_count);
      }
    }
  }

  VectorXf global_weights = metrics::calculate_global_weights(
      frame_metrics, cfg.global_metrics.weights.background,
      cfg.global_metrics.weights.noise, cfg.global_metrics.weights.gradient,
      cfg.global_metrics.clamp[0], cfg.global_metrics.clamp[1],
      cfg.global_metrics.adaptive_weights,
      cfg.global_metrics.weight_exponent_scale);

  {
    core::json artifact;
    artifact["metrics"] = core::json::array();
    for (size_t i = 0; i < frame_metrics.size(); ++i) {
      core::json m;
      m["background"] = frame_metrics[i].background;
      m["noise"] = frame_metrics[i].noise;
      m["gradient_energy"] = frame_metrics[i].gradient_energy;
      m["quality_score"] = frame_metrics[i].quality_score;
      m["global_weight"] = (i < static_cast<size_t>(global_weights.size()))
                               ? global_weights[static_cast<int>(i)]
                               : 0.0f;
      m["fwhm"] = frame_star_metrics[i].fwhm;
      m["fwhm_x"] = frame_star_metrics[i].fwhm_x;
      m["fwhm_y"] = frame_star_metrics[i].fwhm_y;
      m["roundness"] = frame_star_metrics[i].roundness;
      m["wfwhm"] = frame_star_metrics[i].wfwhm;
      m["star_count"] = frame_star_metrics[i].star_count;
      artifact["metrics"].push_back(m);
    }

    artifact["weights"] = {
        {"background", cfg.global_metrics.weights.background},
        {"noise", cfg.global_metrics.weights.noise},
        {"gradient", cfg.global_metrics.weights.gradient}};
    artifact["clamp"] = {cfg.global_metrics.clamp[0],
                         cfg.global_metrics.clamp[1]};
    artifact["adaptive_weights"] = cfg.global_metrics.adaptive_weights;
    core::write_text(run_dir / "artifacts" / "global_metrics.json",
                     artifact.dump(2));
  }

  emitter.phase_end(run_id, Phase::GLOBAL_METRICS, "ok",
                    {
                        {"num_frames", static_cast<int>(frame_metrics.size())},
                    },
                    log_file);

  // --- Memory release: frame_star_metrics no longer needed ---
  { std::vector<metrics::FrameStarMetrics>().swap(frame_star_metrics); }

  // Phase 4: TILE_GRID (with adaptive optimization)
  emitter.phase_start(run_id, Phase::TILE_GRID, "TILE_GRID", log_file);

  float seeing_fwhm_med = 3.0f;
  {
    // Robust FWHM probing: measure on up to 5 evenly-spaced frames,
    // take median of all successful measurements (not just the first).
    const size_t n_probe = std::min<size_t>(5, frames.size());
    std::vector<float> fwhm_probes;
    fwhm_probes.reserve(n_probe);
    for (size_t pi = 0; pi < n_probe; ++pi) {
      size_t fi =
          (n_probe <= 1) ? 0 : (pi * (frames.size() - 1)) / (n_probe - 1);
      const int roi_w = std::min(width, 1024);
      const int roi_h = std::min(height, 1024);
      const int roi_x0 = std::max(0, (width - roi_w) / 2);
      const int roi_y0 = std::max(0, (height - roi_h) / 2);

      Matrix2Df img =
          io::read_fits_region_float(frames[fi], roi_x0, roi_y0, roi_w, roi_h);
      image::apply_normalization_inplace(img, norm_scales[fi], detected_mode,
                                  detected_bayer_str, roi_x0, roi_y0);
      float fwhm = metrics::measure_fwhm_from_image(img);
      if (fwhm > 0.0f && std::isfinite(fwhm))
        fwhm_probes.push_back(fwhm);
    }
    if (!fwhm_probes.empty())
      seeing_fwhm_med = core::median_of(fwhm_probes);
  }

  int seeing_tile_size = 0;
  float overlap_fraction = cfg.tile.overlap_fraction;
  int overlap_px = 0;
  int stride_px = 0;
  {
    float F = seeing_fwhm_med;
    if (!(F > 0.0f) || !std::isfinite(F)) {
      F = 3.0f;
    }

    const int tmin = std::max(16, cfg.tile.min_size);
    const int D = std::max(1, cfg.tile.max_divisor);
    int tmax = std::max(1, std::min(width, height) / D);
    if (tmax < tmin) tmax = tmin;

    const float t0 = static_cast<float>(cfg.tile.size_factor) * F;
    const float tc = std::min(std::max(t0, static_cast<float>(tmin)),
                              static_cast<float>(tmax));
    seeing_tile_size = static_cast<int>(std::floor(tc));
    if (seeing_tile_size < tmin) seeing_tile_size = tmin;

    overlap_fraction = std::min(0.5f, std::max(0.0f, overlap_fraction));
    overlap_px = static_cast<int>(std::floor(overlap_fraction * static_cast<float>(seeing_tile_size)));
    stride_px = seeing_tile_size - overlap_px;
    if (stride_px <= 0) {
      overlap_fraction = 0.25f;
      overlap_px = static_cast<int>(std::floor(overlap_fraction * static_cast<float>(seeing_tile_size)));
      stride_px = seeing_tile_size - overlap_px;
    }
  }

  std::vector<Tile> tiles;
  int uniform_tile_size = seeing_tile_size;
  tiles = tile_compile::pipeline::build_initial_tile_grid(
      width, height, uniform_tile_size, overlap_fraction);

  {
    core::json artifact;
    artifact["image_width"] = width;
    artifact["image_height"] = height;
    artifact["num_tiles"] = static_cast<int>(tiles.size());
    artifact["overlap_fraction"] = overlap_fraction;
    artifact["seeing_fwhm_median"] = seeing_fwhm_med;
    artifact["seeing_tile_size"] = seeing_tile_size;
    artifact["seeing_overlap_px"] = overlap_px;
    artifact["stride_px"] = stride_px;
    artifact["tile_config"] = {
        {"size_factor", cfg.tile.size_factor},
        {"min_size", cfg.tile.min_size},
        {"max_divisor", cfg.tile.max_divisor},
        {"overlap_fraction", overlap_fraction},
    };
    artifact["uniform_tile_size"] = uniform_tile_size;

    artifact["tiles"] = core::json::array();
    for (const auto &t : tiles) {
      artifact["tiles"].push_back({
          {"x", t.x},
          {"y", t.y},
          {"width", t.width},
          {"height", t.height},
      });
    }

    core::write_text(run_dir / "artifacts" / "tile_grid.json",
                     artifact.dump(2));
  }

  emitter.phase_end(run_id, Phase::TILE_GRID, "ok",
                    {
                        {"num_tiles", static_cast<int>(tiles.size())},
                        {"gradient_field", false},
                    },
                    log_file);

  // Helpers for Phase 5/6
  auto load_frame_normalized =
      [&](size_t frame_index) -> std::pair<Matrix2Df, io::FitsHeader> {
    auto frame_pair = io::read_fits_float(frames[frame_index]);
    Matrix2Df img = frame_pair.first;
    image::apply_normalization_inplace(img, norm_scales[frame_index], detected_mode,
                                detected_bayer_str, 0, 0);
    return {img, frame_pair.second};
  };

  // extract_tile is now image::extract_tile (canonical module function)

  // make_hann_1d is now reconstruction::make_hann_1d (canonical module function)

  std::vector<Tile> tiles_phase56 = tiles;
  if (max_tiles > 0 && tiles_phase56.size() > static_cast<size_t>(max_tiles)) {
    tiles_phase56.resize(static_cast<size_t>(max_tiles));
  }

  std::vector<std::vector<TileMetrics>> local_metrics;
  std::vector<std::vector<float>> local_weights;
  std::vector<float> tile_fwhm_median;
  std::vector<int> tile_valid_counts;
  std::vector<uint8_t> tile_fallback_used;
  std::vector<float> tile_warp_variances;
  std::vector<float> tile_mean_correlations;
  std::vector<float> tile_post_contrast;
  std::vector<float> tile_post_background;
  std::vector<float> tile_post_snr;
  std::vector<float> tile_mean_dx;
  std::vector<float> tile_mean_dy;
  std::vector<float> tile_quality_median;
  std::vector<uint8_t> tile_is_star;
  std::vector<std::atomic<int>> frame_valid_tile_counts(frames.size());
  Matrix2Df recon;
  Matrix2Df recon_R;
  Matrix2Df recon_G;
  Matrix2Df recon_B;
  Matrix2Df weight_sum;

  Matrix2Df first_img;
  io::FitsHeader first_hdr;
  {
    auto first_pair = load_frame_normalized(0);
    first_img = std::move(first_pair.first);
    first_hdr = std::move(first_pair.second);
  }

  // Config parameters (v3: single global registration, no tile-ECC)
  int min_valid_frames = 1;

  emitter.phase_start(run_id, Phase::REGISTRATION, "REGISTRATION", log_file);

  std::vector<WarpMatrix> global_frame_warps(frames.size(),
                                             registration::identity_warp());
  std::vector<float> global_frame_cc(frames.size(), 0.0f);
  std::string global_reg_status = "skipped";
  core::json global_reg_extra;
  int global_ref_idx = 0;
  // pick best by global_weight, fallback quality_score, fallback mid-frame
  if (!frame_metrics.empty()) {
    float best_w = -1.0f;
    for (int i = 0; i < static_cast<int>(frame_metrics.size()); ++i) {
      float w = (i < global_weights.size()) ? global_weights[i]
                                            : frame_metrics[i].quality_score;
      if (w > best_w) {
        best_w = w;
        global_ref_idx = i;
      }
    }
    if (best_w < 0.05f) {
      // fallback to highest quality_score
      float best_q = frame_metrics[global_ref_idx].quality_score;
      for (int i = 0; i < static_cast<int>(frame_metrics.size()); ++i) {
        if (frame_metrics[i].quality_score > best_q) {
          best_q = frame_metrics[i].quality_score;
          global_ref_idx = i;
        }
      }
    }
  }
  if (global_ref_idx < 0 || global_ref_idx >= static_cast<int>(frames.size())) {
    global_ref_idx = static_cast<int>(frames.size() / 2);
  }
  global_reg_extra["ref_frame"] = global_ref_idx;

  if (!frames.empty()) {
    try {
      Matrix2Df ref_full;
      {
        auto pair = load_frame_normalized(static_cast<size_t>(global_ref_idx));
        ref_full = std::move(pair.first);
      }
      if (ref_full.size() <= 0) {
        global_reg_status = "error";
        global_reg_extra["error"] = "ref_frame_empty";
      } else {
        Matrix2Df ref_reg = (detected_mode == ColorMode::OSC)
                                ? image::cfa_green_proxy_downsample2x2(
                                      ref_full, detected_bayer_str)
                                : registration::downsample2x2_mean(ref_full);
        float global_reg_scale = 1.0f;
        if (ref_reg.rows() > 0) {
          int full_h2 = ref_full.rows() - (ref_full.rows() % 2);
          global_reg_scale =
              static_cast<float>(full_h2) / static_cast<float>(ref_reg.rows());
        }
        Matrix2Df ref_ecc = registration::prepare_ecc_image(ref_reg);

        // Diagnostic: proxy image stats
        {
          float rmin = ref_reg.minCoeff();
          float rmax = ref_reg.maxCoeff();
          float rmean = ref_reg.mean();
          std::cerr << "[REG-DIAG] ref_reg " << ref_reg.rows() << "x" << ref_reg.cols()
                    << " min=" << rmin << " max=" << rmax << " mean=" << rmean
                    << std::endl;
        }

        for (size_t fi = 0; fi < frames.size(); ++fi) {
          if (static_cast<int>(fi) == global_ref_idx) {
            global_frame_warps[fi] = registration::identity_warp();
            global_frame_cc[fi] = 1.0f;
          } else {
            Matrix2Df mov_full;
            {
              auto pair = load_frame_normalized(fi);
              mov_full = std::move(pair.first);
            }
            if (mov_full.size() <= 0) {
              global_frame_warps[fi] = registration::identity_warp();
              global_frame_cc[fi] = 0.0f;
            } else {
              Matrix2Df mov_reg = (detected_mode == ColorMode::OSC)
                                      ? image::cfa_green_proxy_downsample2x2(
                                            mov_full, detected_bayer_str)
                                      : registration::downsample2x2_mean(mov_full);
              // Diagnostic: first few moving frames
              if (fi < 3) {
                float mmin = mov_reg.minCoeff();
                float mmax = mov_reg.maxCoeff();
                float mmean = mov_reg.mean();
                std::cerr << "[REG-DIAG] mov_reg[" << fi << "] "
                          << mov_reg.rows() << "x" << mov_reg.cols()
                          << " min=" << mmin << " max=" << mmax
                          << " mean=" << mmean << std::endl;
              }

              if (mov_reg.rows() != ref_reg.rows() ||
                  mov_reg.cols() != ref_reg.cols()) {
                global_frame_warps[fi] = registration::identity_warp();
                global_frame_cc[fi] = 0.0f;
              } else {
                // Delegate to canonical cascade in module
                auto sfr = registration::register_single_frame(
                    mov_reg, ref_reg, cfg.registration);

                if (sfr.reg.success) {
                  global_frame_cc[fi] = sfr.reg.correlation;
                  WarpMatrix w_full = sfr.reg.warp;
                  w_full(0, 2) *= global_reg_scale;
                  w_full(1, 2) *= global_reg_scale;
                  global_frame_warps[fi] = w_full;
                } else {
                  global_frame_warps[fi] = registration::identity_warp();
                  global_frame_cc[fi] = 0.0f;
                }

                // Per-frame logging
                if (fi < 5 || fi == frames.size() - 1 ||
                    (fi % 50 == 0)) {
                  std::cerr << "[REG] frame " << fi << "/" << frames.size()
                            << " method=" << sfr.method_used
                            << " ncc_id=" << sfr.ncc_identity
                            << " cc=" << global_frame_cc[fi]
                            << std::endl;
                }
              }
            }
          }

          const float p = frames.empty()
                              ? 1.0f
                              : static_cast<float>(fi + 1) /
                                    static_cast<float>(frames.size());
          emitter.phase_progress(run_id, Phase::REGISTRATION, p,
                                 "global_reg " + std::to_string(fi + 1) + "/" +
                                     std::to_string(frames.size()),
                                 log_file);
        }

        global_reg_status = "ok";
        try {
          core::json j;
          j["num_frames"] = static_cast<int>(frames.size());
          j["scale"] = global_reg_scale;
          j["ref_frame"] = global_ref_idx;
          j["cc"] = core::json::array();
          j["warps"] = core::json::array();
          j["dithering"] = {
              {"enabled", cfg.dithering.enabled},
              {"min_shift_px", cfg.dithering.min_shift_px},
              {"detected_fraction", 0.0},
          };
          int shifts_detected = 0;
          for (size_t fi = 0; fi < frames.size(); ++fi) {
            const auto &w = global_frame_warps[fi];
            j["cc"].push_back(global_frame_cc[fi]);
            const float shift_mag =
                std::sqrt(w(0, 2) * w(0, 2) + w(1, 2) * w(1, 2));
            if (cfg.dithering.enabled && shift_mag >= cfg.dithering.min_shift_px) {
              shifts_detected++;
            }
            j["warps"].push_back({
                {"a00", w(0, 0)},
                {"a01", w(0, 1)},
                {"tx", w(0, 2)},
                {"a10", w(1, 0)},
                {"a11", w(1, 1)},
                {"ty", w(1, 2)},
                {"shift_px", shift_mag},
            });
          }
          if (!frames.empty()) {
            j["dithering"]["detected_fraction"] =
                static_cast<double>(shifts_detected) /
                static_cast<double>(frames.size());
            j["dithering"]["detected_count"] = shifts_detected;
            j["dithering"]["total_frames"] = static_cast<int>(frames.size());
          }
          core::write_text(run_dir / "artifacts" / "global_registration.json",
                           j.dump(2));

          if (cfg.output.write_registered_frames) {
            fs::create_directories(run_dir / cfg.output.registered_dir);
            // first_header is available from outer scope

            for (size_t fi = 0; fi < frames.size(); ++fi) {
              if (static_cast<size_t>(fi) >= global_frame_warps.size())
                continue;
              const auto &w = global_frame_warps[fi];

              auto pair = load_frame_normalized(fi);
              Matrix2Df img = std::move(pair.first);
              if (img.size() <= 0)
                continue;

              Matrix2Df out_img = image::apply_global_warp(img, w,
                                                          detected_mode);

              std::ostringstream name;
              name << "frame_" << std::setw(4) << std::setfill('0') << fi
                   << ".fits";
              io::write_fits_float(run_dir / cfg.output.registered_dir /
                                       name.str(),
                                   out_img, first_header);
            }
          }

        } catch (...) {
        }
      }
    } catch (const std::exception &e) {
      global_reg_status = "error";
      global_reg_extra["error"] = e.what();
    } catch (...) {
      global_reg_status = "error";
      global_reg_extra["error"] = "unknown_error";
    }
  } else {
    global_reg_extra["reason"] = "no_frames";
  }

  // Methodik v3: no frame selection. Registration may fall back to identity,
  // but all frames are still kept for subsequent weighting/stacking.
  int n_cc_positive = 0;
  int n_cc_zero = 0;
  int n_cc_negative = 0;
  for (size_t fi = 0; fi < frames.size(); ++fi) {
    if (global_frame_cc[fi] > 0.0f) {
      ++n_cc_positive;
    } else if (global_frame_cc[fi] < 0.0f) {
      ++n_cc_negative;
    } else {
      ++n_cc_zero;
    }
  }
  std::cerr << "[REG] cc>0: " << n_cc_positive << ", cc==0: " << n_cc_zero
            << ", cc<0: " << n_cc_negative << std::endl;
  global_reg_extra["frames_cc_positive"] = n_cc_positive;
  global_reg_extra["frames_cc_zero"] = n_cc_zero;
  global_reg_extra["frames_cc_negative"] = n_cc_negative;

  emitter.phase_end(run_id, Phase::REGISTRATION, global_reg_status,
                    global_reg_extra, log_file);

  // Pre-warp all frames at full resolution before tile extraction.
  // Applying rotation warps to small tile ROIs is fundamentally broken:
  // warpAffine needs source pixels outside the tile boundary that don't
  // exist, causing CFA pattern corruption (colored tile rectangles).
  //
  // Disk-backed: frames are written as raw float binaries and mmap'd on
  // demand, so RAM usage is bounded by OS page cache rather than N*W*H*4.
  DiskCacheFrameStore prewarped_frames(
      run_dir / ".prewarped_cache", frames.size(), height, width);
  std::vector<uint8_t> frame_has_data(frames.size(), 0);
  int n_frames_with_data = 0;
  for (size_t fi = 0; fi < frames.size(); ++fi) {
    auto pair = load_frame_normalized(fi);
    Matrix2Df img = std::move(pair.first);
    if (img.size() <= 0) {
      continue;
    }
    const auto &w = global_frame_warps[fi];
    const float eps = 1.0e-6f;
    const bool is_identity =
        std::fabs(w(0, 0) - 1.0f) < eps && std::fabs(w(0, 1)) < eps &&
        std::fabs(w(1, 0)) < eps && std::fabs(w(1, 1) - 1.0f) < eps &&
        std::fabs(w(0, 2)) < eps && std::fabs(w(1, 2)) < eps;
    Matrix2Df warped;
    if (is_identity) {
      warped = std::move(img);
    } else {
      warped = image::apply_global_warp(img, w, detected_mode);
    }
    if (warped.size() > 0) {
      prewarped_frames.store(fi, warped);
      if (prewarped_frames.has_data(fi)) {
        frame_has_data[fi] = 1;
        ++n_frames_with_data;
      }
    }
  }

  const int n_usable_frames = n_frames_with_data;
  constexpr int kReducedModeMinFrames = 50;
  const core::ModeGateDecision gate = core::evaluate_mode_gate(
      n_usable_frames, cfg.assumptions.frames_reduced_threshold,
      cfg.runtime_limits.allow_emergency_mode, kReducedModeMinFrames);
  const bool emergency_mode = gate.emergency_mode;
  if (gate.should_abort) {
    std::ostringstream oss;
    oss << "Insufficient usable frames after registration/warp: "
        << n_usable_frames << " (<" << kReducedModeMinFrames
        << "). Set runtime_limits.allow_emergency_mode=true to force "
           "emergency reduced mode.";
    emitter.run_end(run_id, false, "insufficient_frames", log_file);
    std::cerr << "Error: " << oss.str() << std::endl;
    return 1;
  }

  const bool reduced_mode = gate.reduced_mode;
  const bool skip_clustering_in_reduced =
      (reduced_mode && cfg.assumptions.reduced_mode_skip_clustering);
  constexpr float kEpsWeight = 1.0e-6f;
  constexpr float kEpsMedian = 1.0e-6f;
  constexpr float kEpsWeightSum = 1.0e-6f;

  bool run_validation_failed = false;

  while (true) {
    // Phase 5: LOCAL_METRICS (compute tile metrics per frame)
    emitter.phase_start(run_id, Phase::LOCAL_METRICS, "LOCAL_METRICS",
                        log_file);

    local_metrics.assign(frames.size(), {});
    local_weights.assign(frames.size(), {});

    for (size_t fi = 0; fi < frames.size(); ++fi) {
      local_metrics[fi].reserve(tiles_phase56.size());
      local_weights[fi].reserve(tiles_phase56.size());

      if (!frame_has_data[fi]) {
        for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
          TileMetrics z;
          z.fwhm = 0.0f;
          z.roundness = 0.0f;
          z.contrast = 0.0f;
          z.sharpness = 0.0f;
          z.background = 0.0f;
          z.noise = 0.0f;
          z.gradient_energy = 0.0f;
          z.star_count = 0;
          z.type = TileType::STRUCTURE;
          z.quality_score = 0.0f;
          local_metrics[fi].push_back(z);
          local_weights[fi].push_back(0.0f);
        }
        continue;
      }

      for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        const Tile &t = tiles_phase56[ti];
        Matrix2Df tile_img = prewarped_frames.extract_tile(fi, t);

        if (tile_img.size() <= 0) {
          TileMetrics z;
          z.fwhm = 0.0f;
          z.roundness = 0.0f;
          z.contrast = 0.0f;
          z.sharpness = 0.0f;
          z.background = 0.0f;
          z.noise = 0.0f;
          z.gradient_energy = 0.0f;
          z.star_count = 0;
          z.type = TileType::STRUCTURE;
          z.quality_score = 0.0f;
          local_metrics[fi].push_back(z);
          local_weights[fi].push_back(0.0f);
          continue;
        }

        TileMetrics tm = metrics::calculate_tile_metrics(tile_img);
        local_metrics[fi].push_back(tm);
        local_weights[fi].push_back(1.0f);
      }

      const float p = frames.empty() ? 1.0f
                                     : static_cast<float>(fi + 1) /
                                           static_cast<float>(frames.size());
      emitter.phase_progress(run_id, Phase::LOCAL_METRICS, p,
                             "local_metrics " + std::to_string(fi + 1) + "/" +
                                 std::to_string(frames.size()),
                             log_file);
    }

    std::vector<uint8_t> tile_star_flags(tiles_phase56.size(), 0);
    {
      // robust_tilde is now core::robust_zscore (canonical module function)

      auto clip3 = [&](float x) -> float {
        return std::min(std::max(x, cfg.local_metrics.clamp[0]),
                        cfg.local_metrics.clamp[1]);
      };

      const int star_thr = cfg.tile.star_min_count;
      const float eps = 1.0e-12f;

      const size_t n_frames = local_metrics.size();
      const size_t n_tiles = tiles_phase56.size();
      for (size_t ti = 0; ti < n_tiles; ++ti) {
        std::vector<float> fwhm;
        std::vector<float> roundness;
        std::vector<float> contrast;
        std::vector<float> bg;
        std::vector<float> energy_over_noise;
        std::vector<float> star_counts;

        fwhm.reserve(n_frames);
        roundness.reserve(n_frames);
        contrast.reserve(n_frames);
        bg.reserve(n_frames);
        energy_over_noise.reserve(n_frames);
        star_counts.reserve(n_frames);

        // Collect metrics from frames that actually have image data
        std::vector<size_t> usable_indices;
        usable_indices.reserve(n_frames);
        for (size_t fi = 0; fi < n_frames; ++fi) {
          if (!frame_has_data[fi])
            continue;
          usable_indices.push_back(fi);
          const TileMetrics &tm = local_metrics[fi][ti];
          // STAR mode uses FWHM directly (no log transform)
          fwhm.push_back(tm.fwhm);
          roundness.push_back(tm.roundness);
          contrast.push_back(tm.contrast);
          bg.push_back(tm.background);
          // STRUCTURE mode uses robust z-score of (E / σ)
          const float denom = tm.noise;
          const float ratio = (denom > eps) ? (tm.gradient_energy / denom) : 0.0f;
          energy_over_noise.push_back(ratio);
          star_counts.push_back(static_cast<float>(tm.star_count));
        }

        std::vector<float> sc_tmp = star_counts;
        float sc_med = sc_tmp.empty() ? 0.0f : core::median_of(sc_tmp);
        const TileType tile_type = (sc_med >= static_cast<float>(star_thr))
                                       ? TileType::STAR
                                       : TileType::STRUCTURE;
        tile_star_flags[ti] = (tile_type == TileType::STAR) ? 1 : 0;

        std::vector<float> fwhm_t, r_t, c_t, b_t, en_t;
        core::robust_zscore(fwhm, fwhm_t);
        core::robust_zscore(roundness, r_t);
        core::robust_zscore(contrast, c_t);
        core::robust_zscore(bg, b_t);
        core::robust_zscore(energy_over_noise, en_t);

        // Assign z-score-based weights to usable frames
        for (size_t ui = 0; ui < usable_indices.size(); ++ui) {
          size_t fi = usable_indices[ui];
          TileMetrics &tm = local_metrics[fi][ti];
          tm.type = tile_type;

          float q = 0.0f;
          if (tile_type == TileType::STAR) {
            q = cfg.local_metrics.star_mode.weights.fwhm * (-fwhm_t[ui]) +
                cfg.local_metrics.star_mode.weights.roundness * (r_t[ui]) +
                cfg.local_metrics.star_mode.weights.contrast * (c_t[ui]);
          } else {
            q = cfg.local_metrics.structure_mode.metric_weight * (en_t[ui]) +
                cfg.local_metrics.structure_mode.background_weight * (-b_t[ui]);
          }

          q = clip3(q);
          tm.quality_score = q;
          local_weights[fi][ti] = std::exp(q);
        }
      }

      core::json artifact;
      artifact["num_frames"] = static_cast<int>(frames.size());
      artifact["num_tiles"] = static_cast<int>(tiles_phase56.size());
      artifact["tile_metrics"] = core::json::array();

      for (size_t fi = 0; fi < local_metrics.size(); ++fi) {
        core::json fm = core::json::array();
        for (size_t ti = 0; ti < local_metrics[fi].size(); ++ti) {
          const auto &m = local_metrics[fi][ti];
          fm.push_back({
              {"fwhm", m.fwhm},
              {"roundness", m.roundness},
              {"contrast", m.contrast},
              {"sharpness", m.sharpness},
              {"background", m.background},
              {"noise", m.noise},
              {"gradient_energy", m.gradient_energy},
              {"star_count", m.star_count},
              {"tile_type", (m.type == TileType::STAR) ? "STAR" : "STRUCTURE"},
              {"quality_score", m.quality_score},
              {"local_weight", local_weights[fi][ti]},
          });
        }
        artifact["tile_metrics"].push_back(fm);
      }

      core::write_text(run_dir / "artifacts" / "local_metrics.json",
                       artifact.dump(2));
    }

    emitter.phase_end(run_id, Phase::LOCAL_METRICS, "ok",
                      {
                          {"num_frames", static_cast<int>(frames.size())},
                          {"num_tiles", static_cast<int>(tiles_phase56.size())},
                      },
                      log_file);

    // Precompute per-tile median quality and type (for Wiener denoise gating)
    tile_quality_median.assign(tiles_phase56.size(), 0.0f);
    tile_is_star = tile_star_flags;
    if (!local_metrics.empty()) {
      for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        std::vector<float> qs;
        qs.reserve(local_metrics.size());
        for (size_t fi = 0; fi < local_metrics.size(); ++fi) {
          if (!frame_has_data[fi])
            continue;
          if (ti < local_metrics[fi].size()) {
            qs.push_back(local_metrics[fi][ti].quality_score);
          }
        }
        tile_quality_median[ti] = qs.empty() ? 0.0f : core::median_of(qs);
      }
    }

    // Precompute per-tile median FWHM (for FWHM heatmap validation artifact)
    tile_fwhm_median.assign(tiles_phase56.size(), 0.0f);
    if (!local_metrics.empty()) {
      for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        std::vector<float> fwhms;
        fwhms.reserve(local_metrics.size());
        for (size_t fi = 0; fi < local_metrics.size(); ++fi) {
          if (!frame_has_data[fi])
            continue;
          if (ti < local_metrics[fi].size()) {
            fwhms.push_back(local_metrics[fi][ti].fwhm);
          }
        }
        tile_fwhm_median[ti] = fwhms.empty() ? 0.0f : core::median_of(fwhms);
      }
    }

    // Phase 6: TILE_RECONSTRUCTION (Methodik v3)
    emitter.phase_start(run_id, Phase::TILE_RECONSTRUCTION,
                        "TILE_RECONSTRUCTION", log_file);

    const int passes_total = 1;
    // Helper: post-warp metrics (// Methodik v3 §6)
    auto compute_post_warp_metrics =
        [&](const Matrix2Df &warped) -> std::tuple<float, float, float> {
      if (warped.size() <= 0)
        return {0.0f, 0.0f, 0.0f};
      cv::Mat wcv(warped.rows(), warped.cols(), CV_32F,
                  const_cast<float *>(warped.data()));
      cv::Mat lap;
      cv::Laplacian(wcv, lap, CV_32F);
      cv::Scalar mean_sd, stddev_sd;
      cv::meanStdDev(lap, mean_sd, stddev_sd);
      float contrast = static_cast<float>(stddev_sd[0] * stddev_sd[0]);

      std::vector<float> px;
      px.reserve(static_cast<size_t>(warped.size()));
      for (Eigen::Index k = 0; k < warped.size(); ++k) {
        px.push_back(warped.data()[k]);
      }
      float background = core::median_of(px);

      float snr = 0.0f;
      if (!px.empty()) {
        float mad = core::robust_sigma_mad(px);
        std::vector<float> sorted_px = px;
        std::sort(sorted_px.begin(), sorted_px.end());
        float p99 = core::percentile_from_sorted(sorted_px, 99.0f);
        snr = (p99 - background) / (mad + 1.0e-6f);
      }

      return {contrast, background, snr};
    };

    const bool osc_mode = (detected_mode == ColorMode::OSC);

    recon = Matrix2Df::Zero(first_img.rows(), first_img.cols());
    weight_sum = Matrix2Df::Zero(first_img.rows(), first_img.cols());
    if (osc_mode) {
      recon_R = Matrix2Df::Zero(first_img.rows(), first_img.cols());
      recon_G = Matrix2Df::Zero(first_img.rows(), first_img.cols());
      recon_B = Matrix2Df::Zero(first_img.rows(), first_img.cols());
    }

    const int prev_cv_threads_recon = cv::getNumThreads();
    cv::setNumThreads(1);

    // Parallel processing configuration
    int parallel_tiles = 4;
    int cpu_cores = std::thread::hardware_concurrency();
    if (cpu_cores == 0)
      cpu_cores = 1;
    if (parallel_tiles > cpu_cores) {
      std::cout << "[WARNING] parallel_tiles (" << parallel_tiles
                << ") exceeds CPU cores (" << cpu_cores << "), capping to "
                << cpu_cores << std::endl;
      parallel_tiles = cpu_cores;
    }
    if (parallel_tiles < 1)
      parallel_tiles = 1;

    // OSC RGB stacking can be memory-heavy. Cap worker count based on an
    // estimate of per-tile storage (one channel worth of frame tiles).
    if (osc_mode && !tiles_phase56.empty()) {
      size_t max_tile_px = 0;
      for (const auto &t : tiles_phase56) {
        size_t px = static_cast<size_t>(std::max(0, t.width)) *
                    static_cast<size_t>(std::max(0, t.height));
        if (px > max_tile_px)
          max_tile_px = px;
      }
      // Peak per worker is ~ N_frames * tile_pixels * sizeof(float)
      // (channel stacking is sequential, so not multiplied by 3).
      const size_t bytes_per_worker =
          max_tile_px * static_cast<size_t>(std::max(1, n_frames_with_data)) *
          sizeof(float);
      const size_t budget = 512ull * 1024ull * 1024ull; // conservative
      if (bytes_per_worker > 0) {
        int max_workers_by_mem = static_cast<int>(budget / bytes_per_worker);
        if (max_workers_by_mem < 1)
          max_workers_by_mem = 1;
        if (parallel_tiles > max_workers_by_mem) {
          std::cout << "[Phase 6] OSC memory cap: reducing parallel workers from "
                    << parallel_tiles << " to " << max_workers_by_mem
                    << " (est. " << (bytes_per_worker / (1024.0 * 1024.0))
                    << " MiB per worker)" << std::endl;
          parallel_tiles = max_workers_by_mem;
        }
      }
    }

    std::cout << "[Phase 6] Using " << parallel_tiles
              << " parallel workers for " << tiles_phase56.size() << " tiles"
              << std::endl;

    tile_valid_counts.assign(tiles_phase56.size(), 0);
    tile_fallback_used.assign(tiles_phase56.size(), 0);
    tile_warp_variances.assign(tiles_phase56.size(), 0.0f);
    tile_mean_correlations.assign(tiles_phase56.size(), 0.0f);
    tile_post_contrast.assign(tiles_phase56.size(), 0.0f);
    tile_post_background.assign(tiles_phase56.size(), 0.0f);
    tile_post_snr.assign(tiles_phase56.size(), 0.0f);
    tile_mean_dx.assign(tiles_phase56.size(), 0.0f);
    tile_mean_dy.assign(tiles_phase56.size(), 0.0f);
    for (auto &c : frame_valid_tile_counts)
      c.store(0);

    // Thread-safe structures for parallel processing
    std::mutex recon_mutex;
    std::mutex progress_mutex;
    std::atomic<size_t> tiles_completed{0};
    std::atomic<size_t> tiles_failed{0};

    // Pre-compute Hanning windows once for the uniform tile size (all tiles
    // share the same dimensions), avoiding redundant recomputation per tile.
    const std::vector<float> shared_hann_x = reconstruction::make_hann_1d(uniform_tile_size);
    const std::vector<float> shared_hann_y = reconstruction::make_hann_1d(uniform_tile_size);

    // Worker function for parallel tile processing (v3: global warp only, no
    // local ECC)
    auto process_tile = [&](size_t ti) {
      const Tile &t = tiles_phase56[ti];

      auto load_tile_normalized = [&](size_t fi) -> Matrix2Df {
        // Extract tile from pre-warped full frame (already normalized + warped)
        return prewarped_frames.extract_tile(fi, t);
      };

      // Use shared pre-computed windows for uniform tiles; compute only for
      // non-uniform edge tiles (rare).  Both ternary arms must be lvalues to
      // avoid copying the shared vector.
      std::vector<float> hann_x_local;
      std::vector<float> hann_y_local;
      if (t.width != uniform_tile_size) hann_x_local = reconstruction::make_hann_1d(t.width);
      if (t.height != uniform_tile_size) hann_y_local = reconstruction::make_hann_1d(t.height);
      const std::vector<float> &hann_x = (t.width == uniform_tile_size) ? shared_hann_x : hann_x_local;
      const std::vector<float> &hann_y = (t.height == uniform_tile_size) ? shared_hann_y : hann_y_local;

      std::vector<float> weights;
      weights.reserve(frames.size());

      Matrix2Df tile_rec;
      Matrix2Df tile_rec_R;
      Matrix2Df tile_rec_G;
      Matrix2Df tile_rec_B;
      size_t n_valid = 0;
      bool used_weight_fallback = false;

      if (osc_mode) {
        // Methodik v3 (OSC): stack in RGB space (debayer-before-stack).
        // Important: keep peak memory bounded. We therefore stack channels
        // sequentially (R then G then B) instead of holding 3× frame tiles.

        const int origin_x = std::max(0, t.x);
        const int origin_y = std::max(0, t.y);

        std::vector<size_t> valid_frames;
        std::vector<float> weights_valid;
        std::vector<Matrix2Df> valid_mosaics;
        valid_frames.reserve(frames.size());
        weights_valid.reserve(frames.size());
        valid_mosaics.reserve(frames.size());

        // Single pass: extract tiles, determine validity, compute weights,
        // and cache mosaics to avoid re-extraction during per-channel stacking.
        for (size_t fi = 0; fi < frames.size(); ++fi) {
          if (!frame_has_data[fi])
            continue;
          Matrix2Df tile_mosaic = load_tile_normalized(fi);
          if (tile_mosaic.rows() != t.height || tile_mosaic.cols() != t.width)
            continue;

          valid_frames.push_back(fi);
          valid_mosaics.push_back(std::move(tile_mosaic));
          frame_valid_tile_counts[fi].fetch_add(1);

          float G_f = (fi < static_cast<size_t>(global_weights.size()))
                          ? global_weights[static_cast<int>(fi)]
                          : 1.0f;
          float L_ft =
              (fi < local_weights.size() && ti < local_weights[fi].size())
                  ? local_weights[fi][ti]
                  : 1.0f;
          weights_valid.push_back(G_f * L_ft);
        }

        if (valid_frames.empty()) {
          tiles_failed++;
          return;
        }

        auto stack_channel = [&](int which) -> Matrix2Df {
          std::vector<Matrix2Df> chan_tiles;
          chan_tiles.reserve(valid_mosaics.size());

          for (size_t k = 0; k < valid_mosaics.size(); ++k) {
            auto deb = image::debayer_nearest_neighbor(valid_mosaics[k], detected_bayer,
                                                       origin_x, origin_y);
            if (which == 0) {
              chan_tiles.push_back(std::move(deb.R));
            } else if (which == 1) {
              chan_tiles.push_back(std::move(deb.G));
            } else {
              chan_tiles.push_back(std::move(deb.B));
            }
          }

          auto wr = reconstruction::sigma_clip_weighted_tile_with_fallback(
              chan_tiles, weights_valid, cfg.stacking.sigma_clip.sigma_low,
              cfg.stacking.sigma_clip.sigma_high,
              cfg.stacking.sigma_clip.max_iters,
              cfg.stacking.sigma_clip.min_fraction, kEpsWeight);
          used_weight_fallback = used_weight_fallback || wr.fallback_used;
          return std::move(wr.tile);
        };

        tile_rec_R = stack_channel(0);
        tile_rec_G = stack_channel(1);
        tile_rec_B = stack_channel(2);
        valid_mosaics.clear();

        if (tile_rec_R.size() <= 0 || tile_rec_G.size() <= 0 ||
            tile_rec_B.size() <= 0) {
          tiles_failed++;
          return;
        }

        // Post-metrics are computed on G as a stable luminance proxy.
        tile_rec = tile_rec_G;
        n_valid = valid_frames.size();
      } else {
        std::vector<Matrix2Df> warped_tiles;
        warped_tiles.reserve(frames.size());

        for (size_t fi = 0; fi < frames.size(); ++fi) {
          if (!frame_has_data[fi])
            continue;
          Matrix2Df tile_img = load_tile_normalized(fi);
          if (tile_img.rows() != t.height || tile_img.cols() != t.width)
            continue;

          warped_tiles.push_back(tile_img);
          frame_valid_tile_counts[fi].fetch_add(1);
          float G_f = (fi < static_cast<size_t>(global_weights.size()))
                          ? global_weights[static_cast<int>(fi)]
                          : 1.0f;
          float L_ft =
              (fi < local_weights.size() && ti < local_weights[fi].size())
                  ? local_weights[fi][ti]
                  : 1.0f;
          weights.push_back(G_f * L_ft);
        }

        if (warped_tiles.empty()) {
          tiles_failed++;
          return;
        }

        auto wr = reconstruction::sigma_clip_weighted_tile_with_fallback(
            warped_tiles, weights, cfg.stacking.sigma_clip.sigma_low,
            cfg.stacking.sigma_clip.sigma_high,
            cfg.stacking.sigma_clip.max_iters,
            cfg.stacking.sigma_clip.min_fraction, kEpsWeight);
        tile_rec = std::move(wr.tile);
        used_weight_fallback = used_weight_fallback || wr.fallback_used;
        n_valid = warped_tiles.size();
      }

      tile_valid_counts[ti] = static_cast<int>(n_valid);
      tile_fallback_used[ti] = used_weight_fallback ? 1u : 0u;
      tile_warp_variances[ti] = 0.0f;
      tile_mean_correlations[ti] = 1.0f;
      tile_mean_dx[ti] = 0.0f;
      tile_mean_dy[ti] = 0.0f;

      // Methodik 3.1E §3.3.1: Tile denoising after stacking, before OLA.
      // 1. Soft-Threshold (Highpass + shrinkage) — always first (spatial domain)
      bool is_star = (ti < tile_is_star.size()) && tile_is_star[ti];
      if (cfg.tile_denoise.soft_threshold.enabled &&
          !(cfg.tile_denoise.soft_threshold.skip_star_tiles && is_star)) {
        tile_rec = reconstruction::soft_threshold_tile_filter(
            tile_rec, cfg.tile_denoise.soft_threshold);
        if (osc_mode) {
          tile_rec_R = reconstruction::soft_threshold_tile_filter(
              tile_rec_R, cfg.tile_denoise.soft_threshold);
          tile_rec_G = reconstruction::soft_threshold_tile_filter(
              tile_rec_G, cfg.tile_denoise.soft_threshold);
          tile_rec_B = reconstruction::soft_threshold_tile_filter(
              tile_rec_B, cfg.tile_denoise.soft_threshold);
        }
      }

      // 2. Wiener filter (frequency domain) — applied after soft-threshold
      float tile_noise = (ti < tile_quality_median.size())
                             ? tile_quality_median[ti]
                             : 0.0f;
      float tile_snr = (tile_post_snr.size() > ti) ? tile_post_snr[ti] : 0.0f;
      float tile_q = (ti < tile_quality_median.size())
                          ? tile_quality_median[ti]
                          : 0.0f;
      if (cfg.tile_denoise.wiener.enabled) {
        // Estimate noise from tile residual for Wiener filter
        auto estimate_tile_noise = [](const Matrix2Df &t_img) -> float {
          if (t_img.size() <= 0) return 0.0f;
          cv::Mat m(t_img.rows(), t_img.cols(), CV_32F,
                    const_cast<float *>(t_img.data()));
          cv::Mat bg_m;
          cv::blur(m, bg_m, cv::Size(31, 31), cv::Point(-1, -1),
                   cv::BORDER_REFLECT_101);
          cv::Mat r = m - bg_m;
          cv::Scalar mu, sd;
          cv::meanStdDev(r, mu, sd);
          return static_cast<float>(sd[0]);
        };
        float sigma_est = estimate_tile_noise(tile_rec);
        tile_rec = reconstruction::wiener_tile_filter(
            tile_rec, sigma_est, tile_snr, tile_q, is_star,
            cfg.tile_denoise.wiener);
        if (osc_mode) {
          float sig_r = estimate_tile_noise(tile_rec_R);
          tile_rec_R = reconstruction::wiener_tile_filter(
              tile_rec_R, sig_r, tile_snr, tile_q, is_star,
              cfg.tile_denoise.wiener);
          float sig_g = estimate_tile_noise(tile_rec_G);
          tile_rec_G = reconstruction::wiener_tile_filter(
              tile_rec_G, sig_g, tile_snr, tile_q, is_star,
              cfg.tile_denoise.wiener);
          float sig_b = estimate_tile_noise(tile_rec_B);
          tile_rec_B = reconstruction::wiener_tile_filter(
              tile_rec_B, sig_b, tile_snr, tile_q, is_star,
              cfg.tile_denoise.wiener);
        }
      }

      if (osc_mode && cfg.chroma_denoise.enabled &&
          cfg.chroma_denoise.apply_stage == "pre_stack_tiles") {
        reconstruction::chroma_denoise_rgb_inplace(
            tile_rec_R, tile_rec_G, tile_rec_B, cfg.chroma_denoise);
        tile_rec = 0.25f * tile_rec_R + 0.5f * tile_rec_G + 0.25f * tile_rec_B;
      }

      auto [c, b, s] = compute_post_warp_metrics(tile_rec);
      tile_post_contrast[ti] = c;
      tile_post_background[ti] = b;
      tile_post_snr[ti] = s;

      // Methodik 3.1E §3.6: Tile normalization before overlap-add.
      // 1. Subtract background: T' = T - median(T)
      // 2. Normalize: T'' = T' / median(|T'|)  (if median > ε)
      // This eliminates patchwork artifacts from tiles with different
      // background levels before the Hanning-windowed overlap-add.
      std::vector<float> norm_tmp;
      auto normalize_tile_for_ola = [&](Matrix2Df &t_img,
                                        std::vector<float> &tmp) -> float {
        if (t_img.size() <= 0) return 0.0f;
        // Compute median via partial sort on a reusable copy buffer
        tmp.resize(static_cast<size_t>(t_img.size()));
        std::copy(t_img.data(), t_img.data() + t_img.size(), tmp.begin());
        size_t mid = tmp.size() / 2;
        std::nth_element(tmp.begin(), tmp.begin() + static_cast<long>(mid), tmp.end());
        float bg = tmp[mid];
        // Subtract background
        for (Eigen::Index k = 0; k < t_img.size(); ++k)
          t_img.data()[k] -= bg;
        // Compute median of absolute values for scale normalization
        for (size_t i = 0; i < tmp.size(); ++i)
          tmp[i] = std::fabs(t_img.data()[static_cast<Eigen::Index>(i)]);
        std::nth_element(tmp.begin(), tmp.begin() + static_cast<long>(mid), tmp.end());
        float med_abs = tmp[mid];
        if (med_abs > kEpsMedian) {
          float inv = 1.0f / med_abs;
          for (Eigen::Index k = 0; k < t_img.size(); ++k)
            t_img.data()[k] *= inv;
        }
        return bg; // return original background for later restoration
      };

      float tile_bg_mono = normalize_tile_for_ola(tile_rec, norm_tmp);
      float tile_bg_R = 0.0f, tile_bg_G = 0.0f, tile_bg_B = 0.0f;
      if (osc_mode) {
        tile_bg_R = normalize_tile_for_ola(tile_rec_R, norm_tmp);
        tile_bg_G = normalize_tile_for_ola(tile_rec_G, norm_tmp);
        tile_bg_B = normalize_tile_for_ola(tile_rec_B, norm_tmp);
      }

      // Store tile backgrounds for global restoration after overlap-add
      tile_post_background[ti] = osc_mode ? tile_bg_G : tile_bg_mono;

      {
        std::lock_guard<std::mutex> lock(recon_mutex);
        int x0 = std::max(0, t.x);
        int y0 = std::max(0, t.y);
        for (int yy = 0; yy < tile_rec.rows(); ++yy) {
          for (int xx = 0; xx < tile_rec.cols(); ++xx) {
            int iy = y0 + yy;
            int ix = x0 + xx;
            if (iy < 0 || iy >= recon.rows() || ix < 0 || ix >= recon.cols())
              continue;
            float win = hann_y[static_cast<size_t>(yy)] *
                        hann_x[static_cast<size_t>(xx)];
            if (osc_mode) {
              recon_R(iy, ix) += tile_rec_R(yy, xx) * win;
              recon_G(iy, ix) += tile_rec_G(yy, xx) * win;
              recon_B(iy, ix) += tile_rec_B(yy, xx) * win;
            } else {
              recon(iy, ix) += tile_rec(yy, xx) * win;
            }
            weight_sum(iy, ix) += win;
          }
        }
      }

      size_t done = ++tiles_completed;
      if (done % 20 == 0 || done == tiles_phase56.size()) {
        std::lock_guard<std::mutex> lock(progress_mutex);
        emitter.phase_progress_counts(
            run_id, Phase::TILE_RECONSTRUCTION, static_cast<int>(done),
            static_cast<int>(tiles_phase56.size()), "tiles", "tiles", log_file);
      }
    };

    // Execute tiles in parallel or serial based on parallel_tiles setting
    if (parallel_tiles > 1) {
      std::cout << "  Processing " << tiles_phase56.size() << " tiles with "
                << parallel_tiles << " workers..." << std::endl;

      std::vector<std::thread> workers;
      std::atomic<size_t> next_tile{0};

      for (int w = 0; w < parallel_tiles; ++w) {
        workers.emplace_back([&]() {
          while (true) {
            size_t ti = next_tile.fetch_add(1);
            if (ti >= tiles_phase56.size())
              break;
            process_tile(ti);
          }
        });
      }

      for (auto &worker : workers) {
        worker.join();
      }

      std::cout << "  Completed " << tiles_completed.load() << " tiles ("
                << tiles_failed.load() << " failed)" << std::endl;
    } else {
      std::cout << "  Processing " << tiles_phase56.size()
                << " tiles serially..." << std::endl;
      for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
        process_tile(ti);
      }
    }

    cv::setNumThreads(prev_cv_threads_recon);

    // Normalize reconstruction
    const float eps_ws = kEpsWeightSum;
    if (osc_mode) {
      // Fallback: first normalized frame (only used for rare holes).
      auto fb = image::debayer_nearest_neighbor(first_img, detected_bayer, 0, 0);

      for (int i = 0; i < recon.size(); ++i) {
        float ws = weight_sum.data()[i];
        if (ws > eps_ws) {
          recon_R.data()[i] /= ws;
          recon_G.data()[i] /= ws;
          recon_B.data()[i] /= ws;
        } else {
          recon_R.data()[i] = fb.R.data()[i];
          recon_G.data()[i] = fb.G.data()[i];
          recon_B.data()[i] = fb.B.data()[i];
        }
      }

      // Keep a luminance proxy for validation + downstream metrics.
      recon = 0.25f * recon_R + 0.5f * recon_G + 0.25f * recon_B;
    } else {
      for (int i = 0; i < recon.size(); ++i) {
        float ws = weight_sum.data()[i];
        if (ws > eps_ws) {
          recon.data()[i] /= ws;
        } else {
          recon.data()[i] = first_img.data()[i];
        }
      }
    }

    // Methodik 3.1E §3.6: Restore global background after overlap-add.
    // The tile normalization subtracted per-tile backgrounds before OLA.
    // Now restore the median background level across all valid tiles.
    {
      std::vector<float> tile_bgs;
      tile_bgs.reserve(tiles_phase56.size());
      for (size_t i = 0; i < tiles_phase56.size(); ++i) {
        if (tile_valid_counts[i] > 0)
          tile_bgs.push_back(tile_post_background[i]);
      }
      if (!tile_bgs.empty()) {
        float global_bg = core::median_of(tile_bgs);
        if (osc_mode) {
          for (int i = 0; i < recon_R.size(); ++i) {
            recon_R.data()[i] += global_bg;
            recon_G.data()[i] += global_bg;
            recon_B.data()[i] += global_bg;
          }
        } else {
          for (int i = 0; i < recon.size(); ++i)
            recon.data()[i] += global_bg;
        }
      }
    }

    // --- Memory release: weight_sum and first_img no longer needed ---
    weight_sum.resize(0, 0);
    first_img.resize(0, 0);

    // Write reconstruction artifacts (v3)
    {
      core::json artifact;
      artifact["num_frames"] = static_cast<int>(frames.size());
      artifact["num_tiles"] = static_cast<int>(tiles_phase56.size());
      artifact["tile_valid_counts"] = core::json::array();
      artifact["tile_fallback_used"] = core::json::array();
      artifact["tile_mean_correlations"] = core::json::array();
      artifact["tile_post_contrast"] = core::json::array();
      artifact["tile_post_background"] = core::json::array();
      artifact["tile_post_snr_proxy"] = core::json::array();
      for (size_t i = 0; i < tiles_phase56.size(); ++i) {
        artifact["tile_valid_counts"].push_back(tile_valid_counts[i]);
        artifact["tile_fallback_used"].push_back(
            tile_fallback_used[i] != 0u);
        artifact["tile_mean_correlations"].push_back(tile_mean_correlations[i]);
        artifact["tile_post_contrast"].push_back(tile_post_contrast[i]);
        artifact["tile_post_background"].push_back(tile_post_background[i]);
        artifact["tile_post_snr_proxy"].push_back(tile_post_snr[i]);
      }
      core::write_text(run_dir / "artifacts" / "tile_reconstruction.json",
                       artifact.dump(2));
    }

    emitter.phase_end(
        run_id, Phase::TILE_RECONSTRUCTION, "ok",
        {
            {"output", (run_dir / "outputs" / "reconstructed_L.fit").string()},
            {"valid_tiles",
             std::count_if(tile_valid_counts.begin(), tile_valid_counts.end(),
                           [&](int c) { return c >= min_valid_frames; })},
            {"fallback_tiles",
             std::count_if(tile_fallback_used.begin(), tile_fallback_used.end(),
                           [&](uint8_t v) { return v != 0u; })},
        },
        log_file);

    // Phase 7: STATE_CLUSTERING (// Methodik v3 §10)
    emitter.phase_start(run_id, Phase::STATE_CLUSTERING, "STATE_CLUSTERING",
                        log_file);

    // Reduced Mode: optionally skip clustering/synthetic frames when N is small
    bool use_synthetic_frames = true;
    std::string synthetic_skip_reason;
    float synthetic_skip_weight_spread = 0.0f;
    float synthetic_skip_quality_spread = 0.0f;
    int synthetic_skip_eligible_clusters = 0;
    std::vector<int> cluster_labels(static_cast<size_t>(frames.size()), 0);
    int n_clusters = 1;
    if (skip_clustering_in_reduced) {
      use_synthetic_frames = false;
      synthetic_skip_reason = emergency_mode ? "emergency_mode"
                                             : "reduced_mode";
      emitter.phase_end(run_id, Phase::STATE_CLUSTERING, "skipped",
                        {{"reason", synthetic_skip_reason},
                         {"usable_frame_count", n_usable_frames},
                         {"frames_reduced_threshold",
                          cfg.assumptions.frames_reduced_threshold},
                         {"emergency_mode", emergency_mode}},
                        log_file);
    }

    if (!skip_clustering_in_reduced) {
      // Build state vectors for clustering (v3.2 core vector):
      // [G_f, mean_local_quality, var_local_quality, B_f, sigma_f]
      const int n_frames_cluster = static_cast<int>(frames.size());
      std::vector<std::vector<float>> state_vectors(
          static_cast<size_t>(n_frames_cluster));

      std::vector<float> G_for_cluster(static_cast<size_t>(n_frames_cluster),
                                       1.0f);

      for (size_t fi = 0; fi < frames.size(); ++fi) {
        float G_f = (fi < static_cast<size_t>(global_weights.size()))
                        ? global_weights[static_cast<int>(fi)]
                        : 1.0f;
        float bg =
            (fi < frame_metrics.size()) ? frame_metrics[fi].background : 0.0f;
        float noise =
            (fi < frame_metrics.size()) ? frame_metrics[fi].noise : 0.0f;

        // Compute mean/var of local tile quality for this frame
        float mean_local = 0.0f, var_local = 0.0f;
        if (fi < local_metrics.size() && !local_metrics[fi].empty()) {
          for (const auto &tm : local_metrics[fi]) {
            mean_local += tm.quality_score;
          }
          mean_local /= static_cast<float>(local_metrics[fi].size());
          for (const auto &tm : local_metrics[fi]) {
            float diff = tm.quality_score - mean_local;
            var_local += diff * diff;
          }
          var_local /= static_cast<float>(local_metrics[fi].size());
        }
        state_vectors[fi] = {G_f, mean_local, var_local, bg, noise};
        G_for_cluster[fi] = G_f;
      }

      std::vector<std::vector<float>> X = state_vectors;
      std::vector<float> state_means;
      std::vector<float> state_stds;
      std::vector<std::string> final_feature_list = {
          "global_weight",
          "mean_local_quality",
          "var_local_quality",
          "background",
          "noise"};
      if (n_frames_cluster > 0) {
        const size_t D = X[0].size();
        state_means.assign(D, 0.0f);
        state_stds.assign(D, 0.0f);

        for (size_t d = 0; d < D; ++d) {
          double sum = 0.0;
          for (size_t i = 0; i < X.size(); ++i)
            sum += static_cast<double>(X[i][d]);
          state_means[d] = static_cast<float>(sum / static_cast<double>(X.size()));
          double var = 0.0;
          for (size_t i = 0; i < X.size(); ++i) {
            double diff =
                static_cast<double>(X[i][d]) - static_cast<double>(state_means[d]);
            var += diff * diff;
          }
          var /= std::max<double>(1.0, static_cast<double>(X.size()));
          state_stds[d] = static_cast<float>(std::sqrt(std::max(0.0, var)));
        }

        const float eps = kEpsWeight;
        for (size_t i = 0; i < X.size(); ++i) {
          for (size_t d = 0; d < D; ++d) {
            float sd = state_stds[d];
            X[i][d] = (sd > eps) ? ((X[i][d] - state_means[d]) / sd) : 0.0f;
          }
        }
      }

      // Determine cluster count: K = clip(floor(N/10), K_min, K_max)
      int k_min = cfg.synthetic.clustering.cluster_count_range[0];
      int k_max = cfg.synthetic.clustering.cluster_count_range[1];
      int k_default = std::max(k_min, std::min(k_max, n_frames_cluster / 10));

      // Simple k-means clustering
      n_clusters = std::min(k_default, n_frames_cluster);

      std::string clustering_method = "kmeans";

      if (n_clusters > 1 && n_frames_cluster > 1) {
        // K-means++ initialization: pick first center uniformly at random,
        // then each subsequent center with probability proportional to D(x)²
        // (squared distance to nearest existing center).
        std::mt19937 rng(42); // fixed seed for reproducibility
        std::vector<std::vector<float>> centers;
        centers.reserve(static_cast<size_t>(n_clusters));

        // First center: pick middle frame (deterministic, reproducible)
        centers.push_back(X[static_cast<size_t>(n_frames_cluster / 2)]);

        std::vector<double> min_dist_sq(X.size(),
                                         std::numeric_limits<double>::max());
        for (int c = 1; c < n_clusters; ++c) {
          // Update min distances to nearest center (only need to check latest)
          const auto &last_center = centers.back();
          for (size_t fi = 0; fi < X.size(); ++fi) {
            double d2 = 0.0;
            for (size_t d = 0; d < X[fi].size(); ++d) {
              double diff = static_cast<double>(X[fi][d]) -
                            static_cast<double>(last_center[d]);
              d2 += diff * diff;
            }
            if (d2 < min_dist_sq[fi])
              min_dist_sq[fi] = d2;
          }
          // Sample next center with probability proportional to D(x)²
          std::discrete_distribution<size_t> dist(min_dist_sq.begin(),
                                                   min_dist_sq.end());
          size_t next = dist(rng);
          centers.push_back(X[next]);
        }

        // K-means iterations
        for (int iter = 0; iter < 20; ++iter) {
          // Assign labels
          for (size_t fi = 0; fi < X.size(); ++fi) {
            float best_dist = std::numeric_limits<float>::max();
            int best_c = 0;
            for (int c = 0; c < n_clusters; ++c) {
              float dist = 0.0f;
              for (size_t d = 0; d < X[fi].size(); ++d) {
                float diff = X[fi][d] - centers[static_cast<size_t>(c)][d];
                dist += diff * diff;
              }
              if (dist < best_dist) {
                best_dist = dist;
                best_c = c;
              }
            }
            cluster_labels[fi] = best_c;
          }

          // Update centers
          std::vector<std::vector<float>> new_centers(
              static_cast<size_t>(n_clusters),
              std::vector<float>(X[0].size(), 0.0f));
          std::vector<int> counts(static_cast<size_t>(n_clusters), 0);
          for (size_t fi = 0; fi < X.size(); ++fi) {
            int c = cluster_labels[fi];
            for (size_t d = 0; d < X[fi].size(); ++d) {
              new_centers[static_cast<size_t>(c)][d] += X[fi][d];
            }
            counts[static_cast<size_t>(c)]++;
          }
          for (int c = 0; c < n_clusters; ++c) {
            if (counts[static_cast<size_t>(c)] > 0) {
              for (size_t d = 0; d < new_centers[static_cast<size_t>(c)].size();
                   ++d) {
                new_centers[static_cast<size_t>(c)][d] /=
                    static_cast<float>(counts[static_cast<size_t>(c)]);
              }
            }
          }
          centers = new_centers;
        }
      }

      {
        std::vector<int> counts(static_cast<size_t>(n_clusters), 0);
        for (int lbl : cluster_labels) {
          if (lbl >= 0 && lbl < n_clusters)
            counts[static_cast<size_t>(lbl)]++;
        }

        bool degenerate = false;
        for (int c = 0; c < n_clusters; ++c) {
          if (counts[static_cast<size_t>(c)] <= 0) {
            degenerate = true;
            break;
          }
        }

        if (degenerate && n_clusters > 1) {
          clustering_method = "quantile";
          std::vector<std::pair<float, int>> order;
          order.reserve(G_for_cluster.size());
          for (size_t i = 0; i < G_for_cluster.size(); ++i) {
            order.push_back({G_for_cluster[i], static_cast<int>(i)});
          }
          std::sort(
              order.begin(), order.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
          for (size_t r = 0; r < order.size(); ++r) {
            int label = static_cast<int>((r * static_cast<size_t>(n_clusters)) /
                                         std::max<size_t>(1, order.size()));
            if (label >= n_clusters)
              label = n_clusters - 1;
            cluster_labels[static_cast<size_t>(order[r].second)] = label;
          }
        }
      }

      {
        core::json artifact;
        artifact["n_clusters"] = n_clusters;
        artifact["k_min"] = k_min;
        artifact["k_max"] = k_max;
        artifact["method"] = clustering_method;
        artifact["feature_names"] = core::json::array();
        for (const auto &name : final_feature_list)
          artifact["feature_names"].push_back(name);
        artifact["standardization"] = {
            {"method", "zscore"},
            {"eps", kEpsWeight},
            {"means", core::json::array()},
            {"stds", core::json::array()},
        };
        for (float v : state_means)
          artifact["standardization"]["means"].push_back(v);
        for (float v : state_stds)
          artifact["standardization"]["stds"].push_back(v);
        artifact["state_vectors_raw"] = core::json::array();
        artifact["state_vectors_standardized"] = core::json::array();
        for (size_t i = 0; i < state_vectors.size(); ++i) {
          core::json raw = core::json::array();
          core::json stdv = core::json::array();
          for (float v : state_vectors[i])
            raw.push_back(v);
          for (float v : X[i])
            stdv.push_back(v);
          artifact["state_vectors_raw"].push_back(std::move(raw));
          artifact["state_vectors_standardized"].push_back(std::move(stdv));
        }
        artifact["cluster_labels"] = core::json::array();
        for (int lbl : cluster_labels)
          artifact["cluster_labels"].push_back(lbl);
        artifact["cluster_sizes"] = core::json::array();
        for (int c = 0; c < n_clusters; ++c) {
          int count = static_cast<int>(
              std::count(cluster_labels.begin(), cluster_labels.end(), c));
          artifact["cluster_sizes"].push_back(count);
        }
        core::write_text(run_dir / "artifacts" / "state_clustering.json",
                         artifact.dump(2));
      }

      emitter.phase_end(run_id, Phase::STATE_CLUSTERING, "ok",
                        {{"n_clusters", n_clusters}}, log_file);
    }

    // --- Memory release: local_metrics no longer needed after clustering ---
    { std::vector<std::vector<TileMetrics>>().swap(local_metrics); }

    // Phase 8: SYNTHETIC_FRAMES (// Methodik v3 §11)
    emitter.phase_start(run_id, Phase::SYNTHETIC_FRAMES, "SYNTHETIC_FRAMES",
                        log_file);

    struct RGBFrame {
      Matrix2Df R;
      Matrix2Df G;
      Matrix2Df B;
    };

    std::vector<Matrix2Df> synthetic_frames;
    std::vector<RGBFrame> synthetic_rgb_frames;

    auto reconstruct_subset =
        [&](const std::vector<char> &frame_mask) -> Matrix2Df {
      if (cfg.synthetic.weighting == "tile_weighted") {
        Matrix2Df out = Matrix2Df::Zero(height, width);
        Matrix2Df weight_ola = Matrix2Df::Zero(height, width);
        std::atomic<bool> any_tile{false};

        auto normalize_tile_for_ola = [&](Matrix2Df &t_img,
                                          std::vector<float> &tmp) {
          (void)t_img;
          (void)tmp;
          // Preserve synthetic-frame photometric scale. Per-tile median/scale
          // normalization here can imprint tile structure and compress signal.
        };

        struct HannCacheEntry {
          std::vector<float> x;
          std::vector<float> y;
        };
        std::unordered_map<uint64_t, HannCacheEntry> hann_cache;
        hann_cache.reserve(tiles_phase56.size());
        for (const auto &tile : tiles_phase56) {
          const uint64_t key =
              (static_cast<uint64_t>(static_cast<uint32_t>(tile.width)) << 32) |
              static_cast<uint32_t>(tile.height);
          if (hann_cache.find(key) != hann_cache.end())
            continue;
          hann_cache.emplace(key, HannCacheEntry{reconstruction::make_hann_1d(tile.width),
                                                 reconstruction::make_hann_1d(tile.height)});
        }

        std::mutex recon_mutex;
        std::atomic<size_t> next_tile{0};

        int subset_workers = 1;
        if (tiles_phase56.size() > 1) {
          int cpu_cores = static_cast<int>(std::thread::hardware_concurrency());
          if (cpu_cores <= 0)
            cpu_cores = 1;
          subset_workers = std::min<int>(cpu_cores,
                                         static_cast<int>(tiles_phase56.size()));
          subset_workers = std::max(1, std::min(subset_workers, 8));
        }

        auto process_tile = [&]() {
          std::vector<Matrix2Df> cluster_tiles;
          std::vector<float> cluster_weights;
          cluster_tiles.reserve(frame_mask.size());
          cluster_weights.reserve(frame_mask.size());
          std::vector<float> norm_tmp;

          while (true) {
            const size_t ti = next_tile.fetch_add(1);
            if (ti >= tiles_phase56.size())
              break;

            const Tile &t = tiles_phase56[ti];
            cluster_tiles.clear();
            cluster_weights.clear();

            for (size_t fi = 0; fi < frame_mask.size() && fi < frames.size(); ++fi) {
              if (!frame_mask[fi] || !frame_has_data[fi])
                continue;
              Matrix2Df tile_img = prewarped_frames.extract_tile(fi, t);
              if (tile_img.rows() != t.height || tile_img.cols() != t.width)
                continue;
              cluster_tiles.push_back(std::move(tile_img));

              float G_f = (fi < static_cast<size_t>(global_weights.size()))
                              ? global_weights[static_cast<int>(fi)]
                              : 1.0f;
              float L_ft =
                  (fi < local_weights.size() && ti < local_weights[fi].size())
                      ? local_weights[fi][ti]
                      : 1.0f;
              float w = G_f * L_ft;
              cluster_weights.push_back((std::isfinite(w) && w > 0.0f) ? w
                                                                        : 0.0f);
            }

            if (cluster_tiles.empty())
              continue;

            Matrix2Df tile_rec;
            if (cluster_tiles.size() == 1) {
              tile_rec = std::move(cluster_tiles.front());
            } else {
              auto wr = reconstruction::sigma_clip_weighted_tile_with_fallback(
                  cluster_tiles, cluster_weights,
                  cfg.stacking.sigma_clip.sigma_low,
                  cfg.stacking.sigma_clip.sigma_high,
                  cfg.stacking.sigma_clip.max_iters,
                  cfg.stacking.sigma_clip.min_fraction, kEpsWeight);
              tile_rec = std::move(wr.tile);
            }
            if (tile_rec.rows() != t.height || tile_rec.cols() != t.width)
              continue;

            // v3.2 §5.7.1: normalize tile before overlap-add.
            normalize_tile_for_ola(tile_rec, norm_tmp);

            const uint64_t key =
                (static_cast<uint64_t>(static_cast<uint32_t>(t.width)) << 32) |
                static_cast<uint32_t>(t.height);
            const auto cache_it = hann_cache.find(key);
            if (cache_it == hann_cache.end())
              continue;
            const std::vector<float> &hann_x = cache_it->second.x;
            const std::vector<float> &hann_y = cache_it->second.y;

            const int x0 = std::max(0, t.x);
            const int y0 = std::max(0, t.y);

            {
              std::lock_guard<std::mutex> lock(recon_mutex);
              for (int yy = 0; yy < tile_rec.rows(); ++yy) {
                for (int xx = 0; xx < tile_rec.cols(); ++xx) {
                  const int iy = y0 + yy;
                  const int ix = x0 + xx;
                  if (iy < 0 || iy >= out.rows() || ix < 0 || ix >= out.cols())
                    continue;
                  const float win = hann_y[static_cast<size_t>(yy)] *
                                    hann_x[static_cast<size_t>(xx)];
                  out(iy, ix) += tile_rec(yy, xx) * win;
                  weight_ola(iy, ix) += win;
                }
              }
            }
            any_tile.store(true, std::memory_order_relaxed);
          }
        };

        if (subset_workers > 1) {
          std::vector<std::thread> workers;
          workers.reserve(static_cast<size_t>(subset_workers));
          for (int w = 0; w < subset_workers; ++w) {
            workers.emplace_back(process_tile);
          }
          for (auto &worker : workers) {
            if (worker.joinable())
              worker.join();
          }
        } else {
          process_tile();
        }

        if (!any_tile.load(std::memory_order_relaxed))
          return Matrix2Df();

        Matrix2Df fallback;
        for (size_t fi = 0; fi < frame_mask.size() && fi < frames.size(); ++fi) {
          if (!frame_mask[fi] || !frame_has_data[fi])
            continue;
          fallback = prewarped_frames.load(fi);
          if (fallback.size() > 0)
            break;
        }
        if (fallback.size() != out.size()) {
          fallback = Matrix2Df::Zero(out.rows(), out.cols());
        }

        for (Eigen::Index i = 0; i < out.size(); ++i) {
          float ws = weight_ola.data()[i];
          if (ws > kEpsWeightSum) {
            out.data()[i] /= ws;
          } else {
            out.data()[i] = fallback.data()[i];
          }
        }
        return out;
      }

      // Accumulate weighted sum directly to avoid copying full-res frames.
      Matrix2Df out;
      float wsum = 0.0f;

      for (size_t fi = 0; fi < frame_mask.size() && fi < frames.size(); ++fi) {
        if (!frame_mask[fi] || !frame_has_data[fi])
          continue;
        Matrix2Df src = prewarped_frames.load(fi);
        if (src.size() <= 0)
          continue;
        float w = (fi < static_cast<size_t>(global_weights.size()))
                      ? global_weights[static_cast<int>(fi)]
                      : 1.0f;
        if (out.size() == 0) {
          out = Matrix2Df::Zero(src.rows(), src.cols());
        }
        out += src * w;
        wsum += w;
      }

      if (out.size() == 0)
        return Matrix2Df();
      if (wsum > kEpsWeight)
        out /= wsum;
      return out;
    };
    int synth_min = cfg.synthetic.frames_min;
    int synth_max = cfg.synthetic.frames_max;

    if (!use_synthetic_frames) {
      core::json extra;
      if (!synthetic_skip_reason.empty()) {
        extra["reason"] = synthetic_skip_reason;
      } else {
        extra["reason"] = emergency_mode ? "emergency_mode"
                                          : "reduced_mode";
      }
      if (synthetic_skip_eligible_clusters > 0) {
        extra["eligible_clusters"] = synthetic_skip_eligible_clusters;
        extra["weight_spread"] = synthetic_skip_weight_spread;
        extra["quality_spread"] = synthetic_skip_quality_spread;
      }
      extra["usable_frame_count"] = n_usable_frames;
      extra["frames_reduced_threshold"] =
          cfg.assumptions.frames_reduced_threshold;
      extra["emergency_mode"] = emergency_mode;
      emitter.phase_end(run_id, Phase::SYNTHETIC_FRAMES, "skipped", extra,
                        log_file);
    } else {
      std::vector<int> cluster_sizes(n_clusters, 0);
      for (size_t fi = 0; fi < frames.size(); ++fi) {
        int c = cluster_labels[fi];
        if (c >= 0 && c < n_clusters)
          cluster_sizes[c]++;
      }
      int eligible_clusters = 0;
      for (int c = 0; c < n_clusters; ++c) {
        if (cluster_sizes[c] >= synth_min)
          eligible_clusters++;
      }
      const int target_synth = std::min(eligible_clusters, synth_max);
      int clusters_done = 0;
      int synth_done = 0;

      for (int c = 0; c < n_clusters; ++c) {
        std::vector<char> use_frame(frames.size(), 0);
        int count = 0;
        for (size_t fi = 0; fi < frames.size(); ++fi) {
          if (cluster_labels[fi] != c)
            continue;
          use_frame[fi] = 1;
          if (frame_has_data[fi])
            count++;
        }
        clusters_done++;
        emitter.phase_progress_counts(
            run_id, Phase::SYNTHETIC_FRAMES, clusters_done, n_clusters,
            "Cluster " + std::to_string(c + 1) + " von " +
                std::to_string(n_clusters),
            "synthetic " + std::to_string(synth_done) + "/" +
                std::to_string(target_synth),
            log_file);
        if (count < synth_min)
          continue;
        Matrix2Df syn = reconstruct_subset(use_frame);
        if (syn.size() == 0)
          continue;

        if (detected_mode == ColorMode::OSC) {
          auto deb = image::debayer_nearest_neighbor(syn, detected_bayer, 0, 0);
          RGBFrame rgb;
          rgb.R = std::move(deb.R);
          rgb.G = std::move(deb.G);
          rgb.B = std::move(deb.B);
          synthetic_rgb_frames.push_back(std::move(rgb));
        }

        synthetic_frames.push_back(std::move(syn));
        synth_done = static_cast<int>(synthetic_frames.size());
        if (static_cast<int>(synthetic_frames.size()) >= synth_max)
          break;
      }

      if (synthetic_frames.empty()) {
        // If there are not enough frames to satisfy frames_min, treat as a
        // valid skip.
        if (static_cast<int>(frames.size()) < synth_min) {
          use_synthetic_frames = false;
          emitter.phase_end(run_id, Phase::SYNTHETIC_FRAMES, "skipped",
                            {{"reason", "insufficient_frames"},
                             {"frame_count", static_cast<int>(frames.size())},
                             {"frames_min", synth_min}},
                            log_file);
        } else {
          emitter.phase_end(
              run_id, Phase::SYNTHETIC_FRAMES, "error",
              {{"error", "SYNTHETIC_FRAMES: no synthetic frames"}}, log_file);
          emitter.run_end(run_id, false, "error", log_file);
          return 1;
        }
      }
    }

    if (use_synthetic_frames) {
      // Save synthetic frames
      for (size_t si = 0; si < synthetic_frames.size(); ++si) {
        std::string fname = "synthetic_" + std::to_string(si) + ".fit";
        Matrix2Df out = synthetic_frames[si];
        image::apply_output_scaling_inplace(out, 0, 0, detected_mode,
            detected_bayer_str, output_bg_mono, output_bg_r, output_bg_g,
            output_bg_b, output_pedestal);
        io::write_fits_float(run_dir / "outputs" / fname, out, first_hdr);
      }

      {
        core::json artifact;
        artifact["num_synthetic"] = static_cast<int>(synthetic_frames.size());
        artifact["frames_min"] = synth_min;
        artifact["frames_max"] = synth_max;
        artifact["weighting"] = cfg.synthetic.weighting;
        core::write_text(run_dir / "artifacts" / "synthetic_frames.json",
                         artifact.dump(2));
      }

      emitter.phase_end(
          run_id, Phase::SYNTHETIC_FRAMES, "ok",
          {{"num_synthetic", static_cast<int>(synthetic_frames.size())},
           {"weighting", cfg.synthetic.weighting}},
          log_file);
    }

    // --- Memory release: prewarped_frames disk cache no longer needed ---
    // Deletes all temp .raw files from the disk cache directory.
    prewarped_frames.cleanup();
    { std::vector<uint8_t>().swap(frame_has_data); }

    // Phase 9: STACKING (final overlap-add already done in Phase 6)
    emitter.phase_start(run_id, Phase::STACKING, "STACKING", log_file);

    if (use_synthetic_frames) {
      // Filter out empty (0×0) synthetic frames (empty cluster outputs)
      std::vector<Matrix2Df> valid_synth;
      valid_synth.reserve(synthetic_frames.size());

      // For OSC: keep a parallel list of per-frame RGB planes so we can
      // stack in RGB space and avoid debayering after sigma-clipped stacking.
      std::vector<Matrix2Df> synth_R;
      std::vector<Matrix2Df> synth_G;
      std::vector<Matrix2Df> synth_B;
      if (detected_mode == ColorMode::OSC) {
        synth_R.reserve(synthetic_frames.size());
        synth_G.reserve(synthetic_frames.size());
        synth_B.reserve(synthetic_frames.size());
      }

      for (size_t i = 0; i < synthetic_frames.size(); ++i) {
        auto &sf = synthetic_frames[i];
        if (sf.size() <= 0)
          continue;

        if (detected_mode == ColorMode::OSC) {
          if (i < synthetic_rgb_frames.size() &&
              synthetic_rgb_frames[i].R.size() > 0) {
            synth_R.push_back(std::move(synthetic_rgb_frames[i].R));
            synth_G.push_back(std::move(synthetic_rgb_frames[i].G));
            synth_B.push_back(std::move(synthetic_rgb_frames[i].B));
          } else {
            auto deb = image::debayer_nearest_neighbor(sf, detected_bayer, 0, 0);
            synth_R.push_back(std::move(deb.R));
            synth_G.push_back(std::move(deb.G));
            synth_B.push_back(std::move(deb.B));
          }
        }

        valid_synth.push_back(std::move(sf));
      }

      std::cerr << "[STACKING] " << valid_synth.size() << " / "
                << synthetic_frames.size() << " non-empty synthetic frames"
                << std::endl;

      if (!valid_synth.empty()) {
        if (detected_mode == ColorMode::OSC &&
            !synth_R.empty() && synth_R.size() == valid_synth.size()) {
          if (cfg.stacking.method == "rej") {
            recon_R = reconstruction::sigma_clip_stack(
                synth_R, cfg.stacking.sigma_clip.sigma_low,
                cfg.stacking.sigma_clip.sigma_high,
                cfg.stacking.sigma_clip.max_iters,
                cfg.stacking.sigma_clip.min_fraction);
            recon_G = reconstruction::sigma_clip_stack(
                synth_G, cfg.stacking.sigma_clip.sigma_low,
                cfg.stacking.sigma_clip.sigma_high,
                cfg.stacking.sigma_clip.max_iters,
                cfg.stacking.sigma_clip.min_fraction);
            recon_B = reconstruction::sigma_clip_stack(
                synth_B, cfg.stacking.sigma_clip.sigma_low,
                cfg.stacking.sigma_clip.sigma_high,
                cfg.stacking.sigma_clip.max_iters,
                cfg.stacking.sigma_clip.min_fraction);
          } else {
            recon_R = Matrix2Df::Zero(synth_R[0].rows(), synth_R[0].cols());
            recon_G = Matrix2Df::Zero(synth_G[0].rows(), synth_G[0].cols());
            recon_B = Matrix2Df::Zero(synth_B[0].rows(), synth_B[0].cols());
            for (size_t k = 0; k < synth_R.size(); ++k) {
              recon_R += synth_R[k];
              recon_G += synth_G[k];
              recon_B += synth_B[k];
            }
            recon_R /= static_cast<float>(synth_R.size());
            recon_G /= static_cast<float>(synth_G.size());
            recon_B /= static_cast<float>(synth_B.size());
          }
          recon = 0.25f * recon_R + 0.5f * recon_G + 0.25f * recon_B;
        } else {
          if (cfg.stacking.method == "rej") {
            recon = reconstruction::sigma_clip_stack(
                valid_synth, cfg.stacking.sigma_clip.sigma_low,
                cfg.stacking.sigma_clip.sigma_high,
                cfg.stacking.sigma_clip.max_iters,
                cfg.stacking.sigma_clip.min_fraction);
          } else {
            recon = Matrix2Df::Zero(valid_synth[0].rows(), valid_synth[0].cols());
            for (const auto &sf2 : valid_synth) {
              recon += sf2;
            }
            recon /= static_cast<float>(valid_synth.size());
          }
        }
      }
    }

    // Optional post-processing (not part of the linear quality core).
    if (cfg.stacking.cosmetic_correction) {
      recon = image::cosmetic_correction(recon, 5.0f, true);
      if (detected_mode == ColorMode::OSC) {
        recon_R = image::cosmetic_correction(recon_R, 5.0f, true);
        recon_G = image::cosmetic_correction(recon_G, 5.0f, true);
        recon_B = image::cosmetic_correction(recon_B, 5.0f, true);
      }
    }

    if (detected_mode == ColorMode::OSC && cfg.chroma_denoise.enabled &&
        cfg.chroma_denoise.apply_stage == "post_stack_linear") {
      reconstruction::chroma_denoise_rgb_inplace(
          recon_R, recon_G, recon_B, cfg.chroma_denoise);
      recon = 0.25f * recon_R + 0.5f * recon_G + 0.25f * recon_B;
    }

    Matrix2Df recon_out = recon;
    if (detected_mode == ColorMode::OSC) {
      const float bg_luma = 0.25f * output_bg_r + 0.5f * output_bg_g +
                            0.25f * output_bg_b;
      recon_out *= bg_luma;
      recon_out.array() += output_pedestal;
    } else {
      image::apply_output_scaling_inplace(recon_out, 0, 0, detected_mode,
          detected_bayer_str, output_bg_mono, output_bg_r, output_bg_g,
          output_bg_b, output_pedestal);
    }

    // Linear stretch to full 16-bit range [0..65535]
    if (cfg.stacking.output_stretch) {
      float vmin = std::numeric_limits<float>::max();
      float vmax = std::numeric_limits<float>::lowest();
      for (Eigen::Index k = 0; k < recon_out.size(); ++k) {
        float v = recon_out.data()[k];
        if (std::isfinite(v)) {
          if (v < vmin) vmin = v;
          if (v > vmax) vmax = v;
        }
      }
      float range = vmax - vmin;
      if (range > 1.0e-6f) {
        float scale = 65535.0f / range;
        for (Eigen::Index k = 0; k < recon_out.size(); ++k) {
          recon_out.data()[k] = (recon_out.data()[k] - vmin) * scale;
        }
        std::cout << "[Stacking] Output stretch: [" << vmin << ".." << vmax
                  << "] -> [0..65535]" << std::endl;
      }
    }

    try {
      std::error_code ec_space;
      const auto space_info = fs::space(run_dir, ec_space);
      if (!ec_space) {
        const uint64_t required_stack_bytes =
            static_cast<uint64_t>(std::max<Eigen::Index>(0, recon_out.size())) *
            sizeof(float) * 2ULL;
        const uint64_t available_bytes =
            static_cast<uint64_t>(space_info.available);
        if (available_bytes < required_stack_bytes) {
          const std::string msg =
              "Disk full risk before STACKING write: available=" +
              format_bytes(available_bytes) +
              ", required_estimate=" + format_bytes(required_stack_bytes);
          emitter.phase_end(run_id, Phase::STACKING, "error",
                            {{"error", msg},
                             {"runs_device_available_bytes", available_bytes},
                             {"required_estimate_bytes", required_stack_bytes},
                             {"outputs_dir", (run_dir / "outputs").string()}},
                            log_file);
          emitter.run_end(run_id, false, "insufficient_disk_space", log_file);
          std::cerr << "Error during STACKING: " << msg << std::endl;
          return 1;
        }
      }

      io::write_fits_float(run_dir / "outputs" / "stacked.fits", recon_out,
                           first_hdr);
      io::write_fits_float(run_dir / "outputs" / "reconstructed_L.fit",
                           recon_out, first_hdr);
    } catch (const std::exception &e) {
      const bool disk_full = message_indicates_disk_full(e.what());
      const std::string msg =
          disk_full
              ? ("Disk full while writing STACKING outputs to " +
                 (run_dir / "outputs").string() + ": " + e.what())
              : (std::string("STACKING output write failed: ") + e.what());
      emitter.phase_end(run_id, Phase::STACKING, "error",
                        {{"error", msg},
                         {"outputs_dir", (run_dir / "outputs").string()}},
                        log_file);
      emitter.run_end(run_id, false,
                      disk_full ? "insufficient_disk_space" : "error",
                      log_file);
      std::cerr << "Error during STACKING: " << msg << std::endl;
      return 1;
    }

    emitter.phase_end(
        run_id, Phase::STACKING, "ok",
        {{"note", use_synthetic_frames ? "overlap_add_done_in_phase6"
                                       : "reduced_mode_reuse_phase6"}},
        log_file);

    {
      bool validation_ok = true;
      core::json v;

      float output_fwhm_med = metrics::measure_fwhm_from_image(recon);

      float fwhm_improvement_percent = 0.0f;
      if (seeing_fwhm_med > 1.0e-6f && output_fwhm_med > 0.0f) {
        fwhm_improvement_percent =
            (seeing_fwhm_med - output_fwhm_med) / seeing_fwhm_med * 100.0f;
      }
      v["seeing_fwhm_median"] = seeing_fwhm_med;
      v["output_fwhm_median"] = output_fwhm_med;
      v["fwhm_improvement_percent"] = fwhm_improvement_percent;
      if (fwhm_improvement_percent <
          cfg.validation.min_fwhm_improvement_percent) {
        validation_ok = false;
        v["fwhm_improvement_ok"] = false;
      } else {
        v["fwhm_improvement_ok"] = true;
      }

      float tile_weight_variance = 0.0f;
      {
        std::vector<float> tile_means;
        tile_means.reserve(tiles_phase56.size());
        for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
          double sum = 0.0;
          int cnt = 0;
          for (size_t fi = 0; fi < frames.size(); ++fi) {
            float G_f = (fi < static_cast<size_t>(global_weights.size()))
                            ? global_weights[static_cast<int>(fi)]
                            : 1.0f;
            float L_ft =
                (fi < local_weights.size() && ti < local_weights[fi].size())
                    ? local_weights[fi][ti]
                    : 1.0f;
            sum += static_cast<double>(G_f * L_ft);
            cnt++;
          }
          tile_means.push_back(
              cnt > 0 ? static_cast<float>(sum / static_cast<double>(cnt))
                      : 0.0f);
        }
        double mean = 0.0;
        for (float x : tile_means)
          mean += static_cast<double>(x);
        mean /= std::max<double>(1.0, static_cast<double>(tile_means.size()));
        double var = 0.0;
        for (float x : tile_means) {
          double d = static_cast<double>(x) - mean;
          var += d * d;
        }
        var /= std::max<double>(1.0, static_cast<double>(tile_means.size()));
        tile_weight_variance =
            static_cast<float>(var / (mean * mean + 1.0e-12));
      }
      v["tile_weight_variance"] = tile_weight_variance;
      if (tile_weight_variance < cfg.validation.min_tile_weight_variance) {
        validation_ok = false;
        v["tile_weight_variance_ok"] = false;
      } else {
        v["tile_weight_variance_ok"] = true;
      }

      bool tile_pattern_ok = true;
      if (cfg.validation.require_no_tile_pattern) {
        cv::Mat img_cv(recon.rows(), recon.cols(), CV_32F,
                       const_cast<float *>(recon.data()));
        cv::Mat gx, gy;
        cv::Sobel(img_cv, gx, CV_32F, 1, 0, 3);
        cv::Sobel(img_cv, gy, CV_32F, 0, 1, 3);
        cv::Mat mag;
        cv::magnitude(gx, gy, mag);

        std::vector<int> xb;
        std::vector<int> yb;
        xb.reserve(tiles.size());
        yb.reserve(tiles.size());
        for (const auto &t : tiles) {
          if (t.x > 0)
            xb.push_back(t.x);
          if (t.y > 0)
            yb.push_back(t.y);
        }
        std::sort(xb.begin(), xb.end());
        xb.erase(std::unique(xb.begin(), xb.end()), xb.end());
        std::sort(yb.begin(), yb.end());
        yb.erase(std::unique(yb.begin(), yb.end()), yb.end());

        auto line_mean_x = [&](int x) -> float {
          if (x < 0 || x >= mag.cols)
            return 0.0f;
          double sum = 0.0;
          for (int y = 0; y < mag.rows; ++y)
            sum += static_cast<double>(mag.at<float>(y, x));
          return static_cast<float>(sum / static_cast<double>(mag.rows));
        };
        auto line_mean_y = [&](int y) -> float {
          if (y < 0 || y >= mag.rows)
            return 0.0f;
          double sum = 0.0;
          for (int x = 0; x < mag.cols; ++x)
            sum += static_cast<double>(mag.at<float>(y, x));
          return static_cast<float>(sum / static_cast<double>(mag.cols));
        };

        float worst_ratio = 1.0f;
        for (int x : xb) {
          float b = line_mean_x(x);
          float n = 0.5f * (line_mean_x(x - 2) + line_mean_x(x + 2));
          float r = b / (n + 1.0e-12f);
          if (r > worst_ratio)
            worst_ratio = r;
        }
        for (int y : yb) {
          float b = line_mean_y(y);
          float n = 0.5f * (line_mean_y(y - 2) + line_mean_y(y + 2));
          float r = b / (n + 1.0e-12f);
          if (r > worst_ratio)
            worst_ratio = r;
        }
        v["tile_pattern_ratio"] = worst_ratio;
        tile_pattern_ok = (worst_ratio < 1.5f);
        v["tile_pattern_ok"] = tile_pattern_ok;
        if (!tile_pattern_ok)
          validation_ok = false;
      }

      core::write_text(run_dir / "artifacts" / "validation.json", v.dump(2));

      // Do not abort here: we still want to run DEBAYER so GUI gets outputs.
      // We will mark the run as validation_failed at the end.
      if (!validation_ok) {
        run_validation_failed = true;
      }
    }

    // Phase 10: DEBAYER (for OSC data)
    emitter.phase_start(run_id, Phase::DEBAYER, "DEBAYER", log_file);

    Matrix2Df R_out, G_out, B_out;
    Matrix2Df R_disk, G_disk, B_disk;
    bool have_rgb = false;
    fs::path stacked_rgb_path = run_dir / "outputs" / "stacked_rgb.fits";
    fs::path stacked_rgb_solve_path = run_dir / "outputs" / "stacked_rgb_solve.fits";

    if (detected_mode == ColorMode::OSC) {
      if (recon_R.size() == recon.size() && recon_R.size() > 0 &&
          recon_G.size() == recon.size() && recon_B.size() == recon.size()) {
        R_out = recon_R;
        G_out = recon_G;
        B_out = recon_B;
      } else {
        // Fallback (should be rare): debayer luminance proxy.
        auto debayer = image::debayer_nearest_neighbor(recon, detected_bayer, 0, 0);
        R_out = debayer.R;
        G_out = debayer.G;
        B_out = debayer.B;
      }
      have_rgb = true;
      // Restore per-channel background levels to undo the per-channel
      // normalization (scale_r=1/bg_r etc.).  This preserves the camera's
      // native color response and produces a neutral sky background.
      R_out *= output_bg_r;
      G_out *= output_bg_g;
      B_out *= output_bg_b;
      R_out.array() += output_pedestal;
      G_out.array() += output_pedestal;
      B_out.array() += output_pedestal;

      // Keep a separate copy for on-disk outputs (may be stretched for viewing)
      R_disk = R_out;
      G_disk = G_out;
      B_disk = B_out;

      // Linear stretch RGB to full 16-bit range (joint min/max preserves color)
      if (cfg.stacking.output_stretch) {
        float vmin = std::numeric_limits<float>::max();
        float vmax = std::numeric_limits<float>::lowest();
        for (auto *ch : {&R_disk, &G_disk, &B_disk}) {
          for (Eigen::Index k = 0; k < ch->size(); ++k) {
            float v = ch->data()[k];
            if (std::isfinite(v)) {
              if (v < vmin) vmin = v;
              if (v > vmax) vmax = v;
            }
          }
        }
        float range = vmax - vmin;
        if (range > 1.0e-6f) {
          float scale = 65535.0f / range;
          for (auto *ch : {&R_disk, &G_disk, &B_disk}) {
            for (Eigen::Index k = 0; k < ch->size(); ++k) {
              ch->data()[k] = (ch->data()[k] - vmin) * scale;
            }
          }
          std::cout << "[Debayer] RGB output stretch: [" << vmin << ".." << vmax
                    << "] -> [0..65535]" << std::endl;
        }
      }

      io::write_fits_float(run_dir / "outputs" / "reconstructed_R.fit", R_disk,
                           first_hdr);
      io::write_fits_float(run_dir / "outputs" / "reconstructed_G.fit", G_disk,
                           first_hdr);
      io::write_fits_float(run_dir / "outputs" / "reconstructed_B.fit", B_disk,
                           first_hdr);

      // Save stacked_rgb.fits as 3-plane RGB cube (NAXIS3=3) for viewing
      io::write_fits_rgb(stacked_rgb_path, R_disk, G_disk, B_disk, first_hdr);
      // Write an additional linear (non-stretched) cube for plate solving.
      io::write_fits_rgb(stacked_rgb_solve_path, R_out, G_out, B_out, first_hdr);

      emitter.phase_end(
          run_id, Phase::DEBAYER, "ok",
          {{"mode", "OSC"},
           {"bayer_pattern", bayer_pattern_to_string(detected_bayer)},
           {"output_rgb", stacked_rgb_path.string()},
           {"output_rgb_solve", stacked_rgb_solve_path.string()}},
          log_file);
    } else {
      emitter.phase_end(run_id, Phase::DEBAYER, "ok", {{"mode", "MONO"}},
                        log_file);
    }

    // Phase 11: ASTROMETRY (plate solve via ASTAP)
    emitter.phase_start(run_id, Phase::ASTROMETRY, "ASTROMETRY", log_file);

    astro::WCS wcs;
    bool have_wcs = false;

    if (!cfg.astrometry.enabled) {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "disabled"}}, log_file);
    } else if (!have_rgb) {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "no_rgb_data"}}, log_file);
    } else {
      // Determine ASTAP paths (config or defaults)
      std::string astap_data = cfg.astrometry.astap_data_dir;
      if (astap_data.empty()) {
        const char *home = std::getenv("HOME");
        if (home) astap_data = std::string(home) + "/.local/share/tile_compile/astap";
      }
      std::string astap_bin = cfg.astrometry.astap_bin;
      if (astap_bin.empty()) astap_bin = astap_data + "/astap_cli";

      if (!fs::exists(astap_bin)) {
        emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                          {{"reason", "astap_not_found"},
                           {"astap_bin", astap_bin}}, log_file);
      } else {
        auto shell_quote = [](const std::string &s) -> std::string {
          std::string out;
          out.reserve(s.size() + 2);
          out.push_back(static_cast<char>(39));
          for (char c : s) {
            if (c == static_cast<char>(39))
              out += "'\\''";
            else
              out.push_back(c);
          }
          out.push_back(static_cast<char>(39));
          return out;
        };

        // Run ASTAP plate solve on the linear (non-stretched) RGB cube
        std::string cmd = shell_quote(astap_bin) + " -f " +
            shell_quote(stacked_rgb_solve_path.string()) +
            " -d " + shell_quote(astap_data) +
            " -r " + std::to_string(cfg.astrometry.search_radius);

        std::cerr << "[ASTROMETRY] Running: " << cmd << std::endl;
        int ret = std::system(cmd.c_str());

        // ASTAP writes a .wcs file next to the input
        fs::path wcs_path = stacked_rgb_solve_path;
        wcs_path.replace_extension(".wcs");

        if (ret == 0 && fs::exists(wcs_path)) {
          try {
            wcs = astro::parse_wcs_file(wcs_path.string());
            have_wcs = wcs.valid();
          } catch (const std::exception &e) {
            std::cerr << "[ASTROMETRY] WCS parse error: " << e.what() << std::endl;
          }
        }

        if (have_wcs) {
          // Inject WCS keywords into first_hdr so all subsequent
          // FITS outputs (PCC etc.) inherit the astrometric solution.
          // ASTAP does not write WCS into FLOAT_IMG FITS (BITPIX=-32).
          first_hdr.numeric_values["CRVAL1"] = wcs.crval1;
          first_hdr.numeric_values["CRVAL2"] = wcs.crval2;
          first_hdr.numeric_values["CRPIX1"] = wcs.crpix1;
          first_hdr.numeric_values["CRPIX2"] = wcs.crpix2;
          first_hdr.numeric_values["CD1_1"]  = wcs.cd1_1;
          first_hdr.numeric_values["CD1_2"]  = wcs.cd1_2;
          first_hdr.numeric_values["CD2_1"]  = wcs.cd2_1;
          first_hdr.numeric_values["CD2_2"]  = wcs.cd2_2;
          first_hdr.numeric_values["EQUINOX"] = 2000.0;
          first_hdr.string_values["CTYPE1"]  = "RA---TAN";
          first_hdr.string_values["CTYPE2"]  = "DEC--TAN";
          first_hdr.string_values["CUNIT1"]  = "deg";
          first_hdr.string_values["CUNIT2"]  = "deg";
          first_hdr.bool_values["PLTSOLVD"] = true;

          // Re-write outputs with WCS keywords
          if (have_rgb) {
            try {
              io::write_fits_rgb(stacked_rgb_path, R_disk, G_disk, B_disk, first_hdr);
              std::cerr << "[ASTROMETRY] WCS keywords written to " << stacked_rgb_path << std::endl;
            } catch (const std::exception &e) {
              std::cerr << "[ASTROMETRY] Could not update stacked_rgb.fits: " << e.what() << std::endl;
            }
            try {
              io::write_fits_rgb(stacked_rgb_solve_path, R_out, G_out, B_out, first_hdr);
            } catch (const std::exception &) {
            }
          }

          // Copy .wcs to run artifacts directory
          fs::path wcs_artifact = run_dir / "artifacts" / "stacked_rgb.wcs";
          try {
            fs::copy_file(wcs_path, wcs_artifact,
                          fs::copy_options::overwrite_existing);
            std::cerr << "[ASTROMETRY] WCS saved to " << wcs_artifact << std::endl;
          } catch (const std::exception &e) {
            std::cerr << "[ASTROMETRY] Could not copy .wcs: " << e.what() << std::endl;
          }

          emitter.phase_end(run_id, Phase::ASTROMETRY, "ok",
                            {{"ra", wcs.crval1},
                             {"dec", wcs.crval2},
                             {"pixel_scale_arcsec", wcs.pixel_scale_arcsec()},
                             {"rotation_deg", wcs.rotation_deg()},
                             {"fov_w_deg", wcs.fov_width_deg()},
                             {"fov_h_deg", wcs.fov_height_deg()},
                             {"wcs_file", wcs_artifact.string()}},
                            log_file);
        } else {
          emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                            {{"reason", "solve_failed"},
                             {"exit_code", ret}}, log_file);
        }
      }
    }

    // --- Memory release: R_disk/G_disk/B_disk no longer needed after astrometry ---
    R_disk.resize(0, 0);
    G_disk.resize(0, 0);
    B_disk.resize(0, 0);

    // Phase 12: PCC (Photometric Color Calibration)
    emitter.phase_start(run_id, Phase::PCC, "PCC", log_file);

    if (!cfg.pcc.enabled) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "disabled"}}, log_file);
    } else if (!have_wcs) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "no_wcs"}}, log_file);
    } else if (!have_rgb) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "no_rgb_data"}}, log_file);
    } else {
      // Catalog source selection with fallback
      // auto: siril → vizier_gaia → vizier_apass
      double search_r = wcs.search_radius_deg();
      std::string source = cfg.pcc.source;
      std::string used_source;
      std::vector<astro::GaiaStar> stars;

      auto try_siril = [&]() -> bool {
        std::string cat_dir = cfg.pcc.siril_catalog_dir;
        if (cat_dir.empty()) cat_dir = astro::default_siril_gaia_catalog_dir();
        if (!astro::is_siril_gaia_catalog_available(cat_dir)) return false;
        std::cerr << "[PCC] Querying Siril Gaia catalog at RA="
                  << wcs.crval1 << " Dec=" << wcs.crval2
                  << " r=" << search_r << " deg" << std::endl;
        stars = astro::siril_gaia_cone_search(
            cat_dir, wcs.crval1, wcs.crval2, search_r, cfg.pcc.mag_limit);
        if (!stars.empty()) { used_source = "siril"; return true; }
        return false;
      };

      auto try_vizier_gaia = [&]() -> bool {
        std::cerr << "[PCC] Querying VizieR Gaia DR3 at RA="
                  << wcs.crval1 << " Dec=" << wcs.crval2
                  << " r=" << search_r << " deg" << std::endl;
        stars = astro::vizier_gaia_cone_search(
            wcs.crval1, wcs.crval2, search_r, cfg.pcc.mag_limit);
        if (!stars.empty()) { used_source = "vizier_gaia"; return true; }
        return false;
      };

      auto try_vizier_apass = [&]() -> bool {
        std::cerr << "[PCC] Querying VizieR APASS DR9 at RA="
                  << wcs.crval1 << " Dec=" << wcs.crval2
                  << " r=" << search_r << " deg" << std::endl;
        stars = astro::vizier_apass_cone_search(
            wcs.crval1, wcs.crval2, search_r, cfg.pcc.mag_limit);
        if (!stars.empty()) { used_source = "vizier_apass"; return true; }
        return false;
      };

      if (source == "siril") {
        try_siril();
      } else if (source == "vizier_gaia") {
        try_vizier_gaia();
      } else if (source == "vizier_apass") {
        try_vizier_apass();
      } else {
        // auto: try all sources in order
        if (!try_siril()) {
          std::cerr << "[PCC] Siril catalog not available, trying VizieR Gaia..." << std::endl;
          if (!try_vizier_gaia()) {
            std::cerr << "[PCC] VizieR Gaia failed, trying VizieR APASS..." << std::endl;
            try_vizier_apass();
          }
        }
      }

      std::cerr << "[PCC] Found " << stars.size() << " catalog stars"
                << " (source: " << (used_source.empty() ? "none" : used_source) << ")"
                << std::endl;

      if (stars.empty()) {
        emitter.phase_end(run_id, Phase::PCC, "skipped",
                          {{"reason", "no_catalog_stars"},
                           {"search_radius_deg", search_r},
                           {"source", source}}, log_file);
      } else {
        // Build PCC config from pipeline config
        astro::PCCConfig pcc_cfg;
        pcc_cfg.aperture_radius_px = cfg.pcc.aperture_radius_px;
        pcc_cfg.annulus_inner_px = cfg.pcc.annulus_inner_px;
        pcc_cfg.annulus_outer_px = cfg.pcc.annulus_outer_px;
        pcc_cfg.mag_limit = cfg.pcc.mag_limit;
        pcc_cfg.mag_bright_limit = cfg.pcc.mag_bright_limit;
        pcc_cfg.min_stars = cfg.pcc.min_stars;
        pcc_cfg.sigma_clip = cfg.pcc.sigma_clip;

        auto result = astro::run_pcc(R_out, G_out, B_out, wcs, stars, pcc_cfg);

        if (result.success) {
          // Save PCC-corrected RGB as separate files (originals stay intact)
          Matrix2Df R_pcc_disk = R_out;
          Matrix2Df G_pcc_disk = G_out;
          Matrix2Df B_pcc_disk = B_out;
          if (cfg.stacking.output_stretch) {
            float vmin = std::numeric_limits<float>::max();
            float vmax = std::numeric_limits<float>::lowest();
            for (auto *ch : {&R_pcc_disk, &G_pcc_disk, &B_pcc_disk}) {
              for (Eigen::Index k = 0; k < ch->size(); ++k) {
                float v = ch->data()[k];
                if (std::isfinite(v)) {
                  if (v < vmin) vmin = v;
                  if (v > vmax) vmax = v;
                }
              }
            }
            float range = vmax - vmin;
            if (range > 1.0e-6f) {
              float scale = 65535.0f / range;
              for (auto *ch : {&R_pcc_disk, &G_pcc_disk, &B_pcc_disk}) {
                for (Eigen::Index k = 0; k < ch->size(); ++k) {
                  ch->data()[k] = (ch->data()[k] - vmin) * scale;
                }
              }
            }
          }
          io::write_fits_float(run_dir / "outputs" / "pcc_R.fit",
                               R_pcc_disk, first_hdr);
          io::write_fits_float(run_dir / "outputs" / "pcc_G.fit",
                               G_pcc_disk, first_hdr);
          io::write_fits_float(run_dir / "outputs" / "pcc_B.fit",
                               B_pcc_disk, first_hdr);
          io::write_fits_rgb(run_dir / "outputs" / "stacked_rgb_pcc.fits",
                             R_pcc_disk, G_pcc_disk, B_pcc_disk, first_hdr);

          core::json matrix_json = core::json::array();
          for (int r = 0; r < 3; ++r) {
            matrix_json.push_back({result.matrix[r][0],
                                   result.matrix[r][1],
                                   result.matrix[r][2]});
          }

          emitter.phase_end(run_id, Phase::PCC, "ok",
                            {{"stars_matched", result.n_stars_matched},
                             {"stars_used", result.n_stars_used},
                             {"residual_rms", result.residual_rms},
                             {"matrix", matrix_json},
                             {"source", used_source}},
                            log_file);
        } else {
          emitter.phase_end(run_id, Phase::PCC, "skipped",
                            {{"reason", "fit_failed"},
                             {"error", result.error_message},
                             {"stars_matched", result.n_stars_matched},
                             {"source", used_source}},
                            log_file);
        }
      }
    }

    // --- Memory release: all large image buffers before final exit ---
    R_out.resize(0, 0);
    G_out.resize(0, 0);
    B_out.resize(0, 0);
    recon.resize(0, 0);
    recon_R.resize(0, 0);
    recon_G.resize(0, 0);
    recon_B.resize(0, 0);
    { std::vector<std::vector<float>>().swap(local_weights); }
    { std::vector<Matrix2Df>().swap(synthetic_frames); }

    // Phase 13: DONE
    emitter.phase_start(run_id, Phase::DONE, "DONE", log_file);
    emitter.phase_end(run_id, Phase::DONE, "ok", {}, log_file);

    if (run_validation_failed) {
      emitter.run_end(run_id, false, "validation_failed", log_file);

      std::cout << "Pipeline completed with validation_failed" << std::endl;
      return 1;
    }

    emitter.run_end(run_id, true, "ok", log_file);

    std::cout << "Pipeline completed successfully" << std::endl;
    return 0;
  }

} // anonymous namespace

int resume_command(const std::string &run_dir_path, const std::string &from_phase) {
  using namespace tile_compile;

  namespace core = tile_compile::core;
  namespace io = tile_compile::io;
  namespace astro = tile_compile::astrometry;

  fs::path run_dir(run_dir_path);
  if (!fs::exists(run_dir) || !fs::is_directory(run_dir)) {
    std::cerr << "Error: run_dir not found: " << run_dir_path << std::endl;
    return 1;
  }

  fs::path cfg_path = run_dir / "config.yaml";
  if (!fs::exists(cfg_path)) {
    std::cerr << "Error: config.yaml not found in run_dir: " << cfg_path << std::endl;
    return 1;
  }

  config::Config cfg;
  try {
    cfg = config::Config::load(cfg_path);
    cfg.validate();
  } catch (const std::exception &e) {
    std::cerr << "Error: failed to load/validate config.yaml: " << e.what() << std::endl;
    return 1;
  }

  std::string run_id = run_dir.filename().string();
  fs::create_directories(run_dir / "logs");

  // Append to existing events log (do not overwrite)
  std::ofstream event_log_file(run_dir / "logs" / "run_events.jsonl",
                              std::ios::out | std::ios::app);
  TeeBuf tee_buf(std::cout.rdbuf(), event_log_file.rdbuf());
  std::ostream log_file(&tee_buf);

  core::emit_event("resume_start", run_id,
                   {{"run_dir", run_dir.string()},
                    {"from_phase", from_phase}},
                   log_file);

  std::string phase_l = core::to_lower(from_phase);
  if (phase_l.empty()) phase_l = "pcc";

  // Load linear RGB cube (prefer solve cube)
  fs::path rgb_path = run_dir / "outputs" / "stacked_rgb_solve.fits";
  if (!fs::exists(rgb_path)) {
    rgb_path = run_dir / "outputs" / "stacked_rgb.fits";
  }
  if (!fs::exists(rgb_path)) {
    std::cerr << "Error: missing stacked RGB cube in run outputs" << std::endl;
    core::emit_event("resume_end", run_id, {{"success", false}, {"status", "missing_rgb"}}, log_file);
    return 1;
  }

  io::RGBImage rgb;
  try {
    rgb = io::read_fits_rgb(rgb_path);
  } catch (const std::exception &e) {
    std::cerr << "Error: failed to read RGB FITS: " << e.what() << std::endl;
    core::emit_event("resume_end", run_id, {{"success", false}, {"status", "read_rgb_failed"}}, log_file);
    return 1;
  }

  auto inject_wcs_keywords = [](io::FitsHeader &hdr, const astro::WCS &wcs) {
    hdr.numeric_values["CRVAL1"] = wcs.crval1;
    hdr.numeric_values["CRVAL2"] = wcs.crval2;
    hdr.numeric_values["CRPIX1"] = wcs.crpix1;
    hdr.numeric_values["CRPIX2"] = wcs.crpix2;
    hdr.numeric_values["CD1_1"] = wcs.cd1_1;
    hdr.numeric_values["CD1_2"] = wcs.cd1_2;
    hdr.numeric_values["CD2_1"] = wcs.cd2_1;
    hdr.numeric_values["CD2_2"] = wcs.cd2_2;
    hdr.numeric_values["EQUINOX"] = 2000.0;
    hdr.string_values["CTYPE1"] = "RA---TAN";
    hdr.string_values["CTYPE2"] = "DEC--TAN";
    hdr.string_values["CUNIT1"] = "deg";
    hdr.string_values["CUNIT2"] = "deg";
    hdr.bool_values["PLTSOLVD"] = true;
  };

  // Load WCS (prefer artifact copy; fallback to ASTAP output next to solve cube)
  astro::WCS wcs;
  bool have_wcs = false;
  fs::path wcs_path = run_dir / "artifacts" / "stacked_rgb.wcs";
  if (!fs::exists(wcs_path)) {
    fs::path wcs_path2 = rgb_path;
    wcs_path2.replace_extension(".wcs");
    if (fs::exists(wcs_path2)) wcs_path = wcs_path2;
  }
  if (fs::exists(wcs_path)) {
    try {
      wcs = astro::parse_wcs_file(wcs_path.string());
      have_wcs = wcs.valid();
    } catch (const std::exception &) {
      have_wcs = false;
    }
  }

  auto run_astrometry_if_needed = [&]() {
    if (have_wcs) return;

    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::ASTROMETRY, "ASTROMETRY", log_file);

    if (!cfg.astrometry.enabled) {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped", {{"reason", "disabled"}}, log_file);
      return;
    }

    // Determine ASTAP paths (config or defaults)
    std::string astap_data = cfg.astrometry.astap_data_dir;
    if (astap_data.empty()) {
      const char *home = std::getenv("HOME");
      if (home) astap_data = std::string(home) + "/.local/share/tile_compile/astap";
    }
    std::string astap_bin = cfg.astrometry.astap_bin;
    if (astap_bin.empty()) astap_bin = astap_data + "/astap_cli";

    if (!fs::exists(astap_bin)) {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "astap_not_found"}, {"astap_bin", astap_bin}},
                        log_file);
      return;
    }

    auto shell_quote = [](const std::string &s) -> std::string {
      std::string out;
      out.reserve(s.size() + 2);
      out.push_back(static_cast<char>(39));
      for (char c : s) {
        if (c == static_cast<char>(39))
          out += "'\\''";
        else
          out.push_back(c);
      }
      out.push_back(static_cast<char>(39));
      return out;
    };

    std::string cmd = shell_quote(astap_bin) + " -f " + shell_quote(rgb_path.string()) +
                      " -d " + shell_quote(astap_data) +
                      " -r " + std::to_string(cfg.astrometry.search_radius);

    std::cerr << "[ASTROMETRY][resume] Running: " << cmd << std::endl;
    int ret = std::system(cmd.c_str());

    fs::path wcs_out = rgb_path;
    wcs_out.replace_extension(".wcs");

    if (ret == 0 && fs::exists(wcs_out)) {
      try {
        wcs = astro::parse_wcs_file(wcs_out.string());
        have_wcs = wcs.valid();
      } catch (const std::exception &) {
        have_wcs = false;
      }
    }

    if (have_wcs) {
      // Copy .wcs to run artifacts directory for future resumes
      fs::path wcs_artifact = run_dir / "artifacts" / "stacked_rgb.wcs";
      try {
        fs::create_directories(wcs_artifact.parent_path());
        fs::copy_file(wcs_out, wcs_artifact, fs::copy_options::overwrite_existing);
      } catch (const std::exception &) {
      }

      emitter.phase_end(run_id, Phase::ASTROMETRY, "ok",
                        {{"ra", wcs.crval1},
                         {"dec", wcs.crval2},
                         {"pixel_scale_arcsec", wcs.pixel_scale_arcsec()},
                         {"rotation_deg", wcs.rotation_deg()},
                         {"wcs_file", (run_dir / "artifacts" / "stacked_rgb.wcs").string()}},
                        log_file);
    } else {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "solve_failed"}, {"exit_code", ret}},
                        log_file);
    }
  };

  if (phase_l == "astrometry") {
    run_astrometry_if_needed();
    phase_l = "pcc";
  } else if (phase_l != "pcc") {
    std::cerr << "Error: resume --from-phase supports only ASTROMETRY or PCC" << std::endl;
    core::emit_event("resume_end", run_id, {{"success", false}, {"status", "unsupported_phase"}}, log_file);
    return 1;
  }

  if (phase_l == "pcc") {
    run_astrometry_if_needed();

    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::PCC, "PCC", log_file);

    if (!cfg.pcc.enabled) {
      emitter.phase_end(run_id, Phase::PCC, "skipped", {{"reason", "disabled"}}, log_file);
      core::emit_event("resume_end", run_id, {{"success", true}, {"status", "ok"}}, log_file);
      return 0;
    }

    if (!have_wcs) {
      emitter.phase_end(run_id, Phase::PCC, "skipped", {{"reason", "no_wcs"}}, log_file);
      core::emit_event("resume_end", run_id, {{"success", false}, {"status", "no_wcs"}}, log_file);
      return 1;
    }

    // Build WCS keywords into header used for writing PCC outputs
    io::FitsHeader out_hdr = rgb.header;
    inject_wcs_keywords(out_hdr, wcs);

    double search_r = wcs.search_radius_deg();
    std::string source = cfg.pcc.source;
    std::string used_source;
    std::vector<astro::GaiaStar> stars;

    auto try_siril = [&]() -> bool {
      std::string cat_dir = cfg.pcc.siril_catalog_dir;
      if (cat_dir.empty()) cat_dir = astro::default_siril_gaia_catalog_dir();
      if (!astro::is_siril_gaia_catalog_available(cat_dir)) return false;
      std::cerr << "[PCC][resume] Querying Siril Gaia catalog at RA="
                << wcs.crval1 << " Dec=" << wcs.crval2
                << " r=" << search_r << " deg" << std::endl;
      stars = astro::siril_gaia_cone_search(cat_dir, wcs.crval1, wcs.crval2,
                                           search_r, cfg.pcc.mag_limit);
      if (!stars.empty()) { used_source = "siril"; return true; }
      return false;
    };

    auto try_vizier_gaia = [&]() -> bool {
      std::cerr << "[PCC][resume] Querying VizieR Gaia DR3 at RA="
                << wcs.crval1 << " Dec=" << wcs.crval2
                << " r=" << search_r << " deg" << std::endl;
      stars = astro::vizier_gaia_cone_search(wcs.crval1, wcs.crval2,
                                            search_r, cfg.pcc.mag_limit);
      if (!stars.empty()) { used_source = "vizier_gaia"; return true; }
      return false;
    };

    auto try_vizier_apass = [&]() -> bool {
      std::cerr << "[PCC][resume] Querying VizieR APASS DR9 at RA="
                << wcs.crval1 << " Dec=" << wcs.crval2
                << " r=" << search_r << " deg" << std::endl;
      stars = astro::vizier_apass_cone_search(wcs.crval1, wcs.crval2,
                                             search_r, cfg.pcc.mag_limit);
      if (!stars.empty()) { used_source = "vizier_apass"; return true; }
      return false;
    };

    if (source == "siril") {
      try_siril();
    } else if (source == "vizier_gaia") {
      try_vizier_gaia();
    } else if (source == "vizier_apass") {
      try_vizier_apass();
    } else {
      if (!try_siril()) {
        std::cerr << "[PCC][resume] Siril catalog not available, trying VizieR Gaia..." << std::endl;
        if (!try_vizier_gaia()) {
          std::cerr << "[PCC][resume] VizieR Gaia failed, trying VizieR APASS..." << std::endl;
          try_vizier_apass();
        }
      }
    }

    std::cerr << "[PCC][resume] Found " << stars.size() << " catalog stars"
              << " (source: " << (used_source.empty() ? "none" : used_source) << ")"
              << std::endl;

    if (stars.empty()) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "no_catalog_stars"},
                         {"search_radius_deg", search_r},
                         {"source", source}},
                        log_file);
      core::emit_event("resume_end", run_id, {{"success", false}, {"status", "no_catalog_stars"}}, log_file);
      return 1;
    }

    astro::PCCConfig pcc_cfg;
    pcc_cfg.aperture_radius_px = cfg.pcc.aperture_radius_px;
    pcc_cfg.annulus_inner_px = cfg.pcc.annulus_inner_px;
    pcc_cfg.annulus_outer_px = cfg.pcc.annulus_outer_px;
    pcc_cfg.mag_limit = cfg.pcc.mag_limit;
    pcc_cfg.mag_bright_limit = cfg.pcc.mag_bright_limit;
    pcc_cfg.min_stars = cfg.pcc.min_stars;
    pcc_cfg.sigma_clip = cfg.pcc.sigma_clip;

    auto result = astro::run_pcc(rgb.R, rgb.G, rgb.B, wcs, stars, pcc_cfg);

    if (!result.success) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "fit_failed"},
                         {"error", result.error_message},
                         {"stars_matched", result.n_stars_matched},
                         {"source", used_source}},
                        log_file);
      core::emit_event("resume_end", run_id, {{"success", false}, {"status", "fit_failed"}}, log_file);
      return 1;
    }

    Matrix2Df R_pcc_disk = rgb.R;
    Matrix2Df G_pcc_disk = rgb.G;
    Matrix2Df B_pcc_disk = rgb.B;

    if (cfg.stacking.output_stretch) {
      float vmin = std::numeric_limits<float>::max();
      float vmax = std::numeric_limits<float>::lowest();
      for (auto *ch : {&R_pcc_disk, &G_pcc_disk, &B_pcc_disk}) {
        for (Eigen::Index k = 0; k < ch->size(); ++k) {
          float v = ch->data()[k];
          if (std::isfinite(v)) {
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
          }
        }
      }
      float range = vmax - vmin;
      if (range > 1.0e-6f) {
        float scale = 65535.0f / range;
        for (auto *ch : {&R_pcc_disk, &G_pcc_disk, &B_pcc_disk}) {
          for (Eigen::Index k = 0; k < ch->size(); ++k) {
            ch->data()[k] = (ch->data()[k] - vmin) * scale;
          }
        }
      }
    }

    io::write_fits_float(run_dir / "outputs" / "pcc_R.fit", R_pcc_disk, out_hdr);
    io::write_fits_float(run_dir / "outputs" / "pcc_G.fit", G_pcc_disk, out_hdr);
    io::write_fits_float(run_dir / "outputs" / "pcc_B.fit", B_pcc_disk, out_hdr);
    io::write_fits_rgb(run_dir / "outputs" / "stacked_rgb_pcc.fits",
                       R_pcc_disk, G_pcc_disk, B_pcc_disk, out_hdr);

    core::json matrix_json = core::json::array();
    for (int r = 0; r < 3; ++r) {
      matrix_json.push_back({result.matrix[r][0], result.matrix[r][1], result.matrix[r][2]});
    }

    emitter.phase_end(run_id, Phase::PCC, "ok",
                      {{"stars_matched", result.n_stars_matched},
                       {"stars_used", result.n_stars_used},
                       {"residual_rms", result.residual_rms},
                       {"matrix", matrix_json},
                       {"source", used_source}},
                      log_file);
  }

  core::emit_event("resume_end", run_id, {{"success", true}, {"status", "ok"}}, log_file);
  return 0;
}

int main(int argc, char *argv[]) {
  QCoreApplication qapp(argc, argv);  // needed for Qt6::Network event loop

#ifdef HAVE_CLI11
  CLI::App app{"Tile-Compile Runner (C++)"};

  std::string config_path, input_dir, runs_dir, project_root;
  std::string resume_run_dir;
  std::string resume_from_phase = "PCC";
  bool dry_run = false;
  int max_frames = 0;
  int max_tiles = 0;
  bool config_from_stdin = false;

  auto run_cmd = app.add_subcommand("run", "Run the pipeline");
  run_cmd->add_option("--config", config_path, "Path to config.yaml")
      ->required();
  run_cmd->add_option("--input-dir", input_dir, "Input directory")->required();
  run_cmd->add_option("--runs-dir", runs_dir, "Runs directory")->required();
  run_cmd->add_option("--project-root", project_root, "Project root");
  run_cmd->add_option("--max-frames", max_frames,
                      "Limit number of frames (0 = no limit)");
  run_cmd->add_option("--max-tiles", max_tiles,
                      "Limit number of tiles in Phase 5/6 (0 = no limit)");
  run_cmd->add_flag("--dry-run", dry_run, "Dry run");
  run_cmd->add_flag("--stdin", config_from_stdin,
                    "Read config YAML from stdin (use with --config -)");

  auto resume_cmd = app.add_subcommand("resume", "Resume an existing run (currently PCC-only)");
  resume_cmd->add_option("--run-dir", resume_run_dir, "Existing run directory")
      ->required();
  resume_cmd->add_option("--from-phase", resume_from_phase,
                         "Phase to resume from: ASTROMETRY|PCC")
      ->default_val("PCC");

  CLI11_PARSE(app, argc, argv);

  if (run_cmd->parsed()) {
    return run_command(config_path, input_dir, runs_dir, project_root, dry_run,
                       max_frames, max_tiles, config_from_stdin);
  }

  if (resume_cmd->parsed()) {
    return resume_command(resume_run_dir, resume_from_phase);
  }

  print_usage();
  return 1;
#else
  if (argc < 2) {
    print_usage();
    return 1;
  }

  std::string command = argv[1];
  std::string config_path, input_dir, runs_dir, project_root;
  std::string resume_run_dir;
  std::string resume_from_phase = "PCC";
  bool dry_run = false;
  int max_frames = 0;
  int max_tiles = 0;
  bool config_from_stdin = false;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc)
      config_path = argv[++i];
    else if (arg == "--input-dir" && i + 1 < argc)
      input_dir = argv[++i];
    else if (arg == "--runs-dir" && i + 1 < argc)
      runs_dir = argv[++i];
    else if (arg == "--project-root" && i + 1 < argc)
      project_root = argv[++i];
    else if (arg == "--run-dir" && i + 1 < argc)
      resume_run_dir = argv[++i];
    else if (arg == "--from-phase" && i + 1 < argc)
      resume_from_phase = argv[++i];
    else if (arg == "--max-frames" && i + 1 < argc)
      max_frames = std::stoi(argv[++i]);
    else if (arg == "--max-tiles" && i + 1 < argc)
      max_tiles = std::stoi(argv[++i]);
    else if (arg == "--dry-run")
      dry_run = true;
    else if (arg == "--stdin")
      config_from_stdin = true;
  }

  if (command == "run") {
    if (config_path.empty() || input_dir.empty() || runs_dir.empty()) {
      std::cerr << "Error: --config, --input-dir, and --runs-dir are required"
                << std::endl;
      return 1;
    }
    return run_command(config_path, input_dir, runs_dir, project_root, dry_run,
                       max_frames, max_tiles, config_from_stdin);
  }

  if (command == "resume") {
    if (resume_run_dir.empty()) {
      std::cerr << "Error: resume requires --run-dir <path>" << std::endl;
      return 1;
    }
    return resume_command(resume_run_dir, resume_from_phase);
  }

  print_usage();
  return 1;
#endif
}
