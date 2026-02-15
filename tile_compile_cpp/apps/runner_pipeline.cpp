#include "runner_pipeline.hpp"

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
#include "tile_compile/pipeline/adaptive_tile_grid.hpp"
#include "tile_compile/reconstruction/reconstruction.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/astrometry/gaia_catalog.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"

#include "runner_phase_metrics.hpp"
#include "runner_phase_local_metrics.hpp"
#include "runner_phase_registration.hpp"
#include "runner_shared.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {
using tile_compile::ColorMode;
using tile_compile::Matrix2Df;
using tile_compile::Tile;
using tile_compile::WarpMatrix;

namespace image = tile_compile::image;
namespace astro = tile_compile::astrometry;
using tile_compile::runner::TeeBuf;
using tile_compile::runner::estimate_total_file_bytes;
using tile_compile::runner::format_bytes;
using tile_compile::runner::message_indicates_disk_full;

using NormalizationScales = image::NormalizationScales;
} // namespace

int run_pipeline_command(const std::string &config_path, const std::string &input_dir,
                const std::string &runs_dir, const std::string &project_root,
                const std::string &run_id_override,
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

  auto frames = core::discover_frames(in_dir, "*");
  frames.erase(
      std::remove_if(frames.begin(), frames.end(),
                     [](const fs::path &p) { return !io::is_fits_image_path(p); }),
      frames.end());
  std::sort(frames.begin(), frames.end());
  if (max_frames > 0 && frames.size() > static_cast<size_t>(max_frames)) {
    frames.resize(static_cast<size_t>(max_frames));
  }
  if (frames.empty()) {
    std::cerr << "Error: No FITS frames found in " << input_dir << std::endl;
    return 1;
  }

  std::string run_id = run_id_override.empty() ? core::get_run_id() : run_id_override;
  fs::path run_dir = fs::absolute(runs / run_id);
  fs::create_directories(run_dir / "logs");
  fs::create_directories(run_dir / "outputs");
  fs::create_directories(run_dir / "artifacts");

  if (use_stdin_config) {
    std::ofstream out(run_dir / "config.yaml", std::ios::out);
    out << cfg_text;
  } else {
    core::copy_config(cfg_path, run_dir / "config.yaml");
  }

  std::ofstream event_log_file(run_dir / "logs" / "run_events.jsonl",
                               std::ios::out | std::ios::trunc);
  if (!event_log_file.is_open()) {
    std::cerr << "Error: cannot open events log file: "
              << (run_dir / "logs" / "run_events.jsonl") << std::endl;
    return 1;
  }
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

  const bool header_has_color_hint =
      (naxis >= 3) || (detected_bayer != BayerPattern::UNKNOWN) ||
      first_header.get_string("COLORTYP").has_value();
  const ColorMode cfg_color_mode =
      cfg.data.color_mode.empty() ? ColorMode::MONO
                                  : (cfg.data.color_mode == "RGB"
                                         ? ColorMode::RGB
                                         : (cfg.data.color_mode == "OSC"
                                                ? ColorMode::OSC
                                                : ColorMode::MONO));
  const bool cfg_color_mode_valid =
      cfg.data.color_mode == "MONO" || cfg.data.color_mode == "OSC" ||
      cfg.data.color_mode == "RGB";
  const BayerPattern cfg_bayer = cfg.data.bayer_pattern.empty()
                                     ? BayerPattern::UNKNOWN
                                     : string_to_bayer_pattern(
                                           cfg.data.bayer_pattern);
  const bool cfg_looks_like_default_osc =
      (cfg.data.color_mode == "OSC" &&
       (cfg.data.bayer_pattern.empty() || cfg.data.bayer_pattern == "GBRG"));

  if (!header_has_color_hint && cfg_color_mode_valid) {
    // For hint-less FITS, avoid forcing OSC from implicit defaults.
    // This keeps MONO/SW datasets processable without BAYERPAT.
    if (!cfg_looks_like_default_osc) {
      detected_mode = cfg_color_mode;
      detected_mode_str = color_mode_to_string(detected_mode);
      emitter.warning(run_id,
                      "FITS header has no clear color hint; using "
                      "config.data.color_mode='" +
                          cfg.data.color_mode + "' as fallback",
                      log_file);
    } else {
      detected_mode = ColorMode::MONO;
      detected_mode_str = color_mode_to_string(detected_mode);
      emitter.warning(run_id,
                      "FITS header has no clear color hint; default OSC/BAYER "
                      "config would be ambiguous, using MONO fallback",
                      log_file);
    }
  }
  if (detected_mode == ColorMode::OSC &&
      detected_bayer == BayerPattern::UNKNOWN &&
      cfg_bayer != BayerPattern::UNKNOWN) {
    detected_bayer = cfg_bayer;
    detected_bayer_str = bayer_pattern_to_string(detected_bayer);
    emitter.warning(run_id,
                    "FITS header has no valid BAYER pattern; using "
                    "config.data.bayer_pattern='" +
                        cfg.data.bayer_pattern + "' as fallback",
                    log_file);
  } else if (detected_mode != ColorMode::OSC) {
    detected_bayer = BayerPattern::UNKNOWN;
    detected_bayer_str = bayer_pattern_to_string(detected_bayer);
  }
  if (width <= 0 && cfg.data.image_width > 0) {
    width = cfg.data.image_width;
    emitter.warning(run_id,
                    "FITS header missing image_width; using "
                    "config.data.image_width fallback",
                    log_file);
  }
  if (height <= 0 && cfg.data.image_height > 0) {
    height = cfg.data.image_height;
    emitter.warning(run_id,
                    "FITS header missing image_height; using "
                    "config.data.image_height fallback",
                    log_file);
  }

  if (header_has_color_hint && !cfg.data.color_mode.empty() &&
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

    emitter.warning(
        run_id,
        "Linearity: " + std::to_string(rejected_indices.size()) +
            " frames flagged non-linear (kept, warn-only mode)",
        log_file);
    linearity_info["action"] = "warn_only";
    linearity_info["frames_remaining"] = static_cast<int>(frames.size());
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

  runner::PhaseMetricsContext phase_metrics_ctx;
  if (!runner::run_phase_channel_split_normalization_global_metrics(
          run_id, cfg, frames, run_dir, detected_mode, detected_bayer_str,
          emitter, log_file, phase_metrics_ctx)) {
    return 1;
  }

  auto &norm_scales = phase_metrics_ctx.norm_scales;
  auto &frame_metrics = phase_metrics_ctx.frame_metrics;
  VectorXf global_weights = phase_metrics_ctx.global_weights;
  const float output_pedestal = phase_metrics_ctx.output_pedestal;
  const float output_bg_mono = phase_metrics_ctx.output_bg_mono;
  const float output_bg_r = phase_metrics_ctx.output_bg_r;
  const float output_bg_g = phase_metrics_ctx.output_bg_g;
  const float output_bg_b = phase_metrics_ctx.output_bg_b;

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
  float overlap_clipped = cfg.tile.overlap_fraction;
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

    overlap_clipped = std::min(0.5f, std::max(0.0f, overlap_fraction));
    overlap_fraction = overlap_clipped;
    overlap_px = static_cast<int>(
        std::floor(overlap_clipped * static_cast<float>(seeing_tile_size)));
    stride_px = seeing_tile_size - overlap_px;
    if (stride_px <= 0) {
      overlap_clipped = std::min(0.5f, std::max(0.0f, 0.25f));
      overlap_fraction = overlap_clipped;
      overlap_px = static_cast<int>(
          std::floor(overlap_clipped * static_cast<float>(seeing_tile_size)));
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
        {"overlap_clipped", overlap_clipped},
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
  std::vector<float> tile_norm_bg_r;
  std::vector<float> tile_norm_bg_g;
  std::vector<float> tile_norm_bg_b;
  std::vector<float> tile_norm_scale;
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

  runner::PhaseRegistrationContext phase_registration_ctx;
  if (!runner::run_phase_registration_prewarp(
          run_id, cfg, frames, run_dir, height, width, detected_mode,
          detected_bayer_str, norm_scales, frame_metrics, global_weights,
          first_header, emitter, log_file, phase_registration_ctx)) {
    return 1;
  }

  auto &prewarped_frames = phase_registration_ctx.prewarped_frames;
  auto &frame_has_data = phase_registration_ctx.frame_has_data;
  const int n_usable_frames = phase_registration_ctx.n_usable_frames;
  int min_valid_frames = phase_registration_ctx.min_valid_frames;

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
    if (!runner::run_phase_local_metrics(
            run_id, cfg, frames, run_dir, frame_has_data, tiles_phase56,
            prewarped_frames, emitter, log_file, local_metrics, local_weights,
            tile_quality_median, tile_is_star, tile_fwhm_median)) {
      return 1;
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
    int parallel_tiles = cfg.runtime_limits.parallel_workers;
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
      const int n_frames_with_data_i = n_usable_frames;
      const size_t bytes_per_worker =
          max_tile_px * static_cast<size_t>(std::max(1, n_frames_with_data_i)) *
          sizeof(float);
      const size_t budget =
          static_cast<size_t>(std::max(1, cfg.runtime_limits.memory_budget)) *
          1024ull * 1024ull;
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
    tile_norm_bg_r.assign(tiles_phase56.size(), 0.0f);
    tile_norm_bg_g.assign(tiles_phase56.size(), 0.0f);
    tile_norm_bg_b.assign(tiles_phase56.size(), 0.0f);
    tile_norm_scale.assign(tiles_phase56.size(), 1.0f);
    tile_post_snr.assign(tiles_phase56.size(), 0.0f);
    tile_mean_dx.assign(tiles_phase56.size(), 0.0f);
    tile_mean_dy.assign(tiles_phase56.size(), 0.0f);
    for (auto &c : frame_valid_tile_counts)
      c.store(0);

    // NOTE (2026-02): In reduced/emergency mode, Phase 7 output is the final
    // stack (Phase 8/9 skipped). For small DSLR-derived sets (Canon/Nikon RAW
    // -> FITS) the v3.2 per-tile median(abs)-scale normalization before OLA can
    // imprint the tile lattice (stride-sized checker/grid) into the final image.
    // Therefore:
    //  - Full mode: keep v3.2 §5.7.1 normalization (used as intermediate stage)
    //  - Reduced/emergency: disable per-tile normalization to preserve visual
    //    continuity and avoid tile-contrast pumping in the final output
    const bool apply_phase7_tile_norm = !skip_clustering_in_reduced;

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

      if (apply_phase7_tile_norm) {
        // Methodik v3.2 §5.7.1 (verbindlich): Tile-Normalisierung vor OLA.
        // OSC-Schutz: gemeinsamer Scale-Faktor fuer R/G/B, um Farbrelationen
        // nicht tileweise zu verzerren.
        std::vector<float> norm_tmp;
        auto median_from_matrix = [&](const Matrix2Df &src,
                                      bool abs_values) -> float {
          if (src.size() <= 0) {
            return 0.0f;
          }
          norm_tmp.resize(static_cast<size_t>(src.size()));
          if (abs_values) {
            for (Eigen::Index i = 0; i < src.size(); ++i) {
              norm_tmp[static_cast<size_t>(i)] = std::fabs(src.data()[i]);
            }
          } else {
            std::copy(src.data(), src.data() + src.size(), norm_tmp.begin());
          }
          const size_t mid = norm_tmp.size() / 2;
          std::nth_element(norm_tmp.begin(),
                           norm_tmp.begin() + static_cast<long>(mid),
                           norm_tmp.end());
          return norm_tmp[mid];
        };
        auto center_by_median = [&](Matrix2Df &src) -> float {
          const float bg = median_from_matrix(src, false);
          for (Eigen::Index i = 0; i < src.size(); ++i) {
            src.data()[i] -= bg;
          }
          return bg;
        };

        if (osc_mode) {
          const float bg_r = center_by_median(tile_rec_R);
          const float bg_g = center_by_median(tile_rec_G);
          const float bg_b = center_by_median(tile_rec_B);
          tile_norm_bg_r[ti] = bg_r;
          tile_norm_bg_g[ti] = bg_g;
          tile_norm_bg_b[ti] = bg_b;

          tile_rec =
              0.25f * tile_rec_R + 0.5f * tile_rec_G + 0.25f * tile_rec_B;
          const float m_shared = median_from_matrix(tile_rec, true);
          if (m_shared >= kEpsMedian) {
            const float inv = 1.0f / m_shared;
            for (Eigen::Index i = 0; i < tile_rec_R.size(); ++i) {
              tile_rec_R.data()[i] *= inv;
              tile_rec_G.data()[i] *= inv;
              tile_rec_B.data()[i] *= inv;
            }
            for (Eigen::Index i = 0; i < tile_rec.size(); ++i) {
              tile_rec.data()[i] *= inv;
            }
            tile_norm_scale[ti] = m_shared;
          } else {
            tile_norm_scale[ti] = 1.0f;
          }
        } else {
          const float bg = center_by_median(tile_rec);
          tile_norm_bg_r[ti] = bg;
          const float m = median_from_matrix(tile_rec, true);
          if (m >= kEpsMedian) {
            const float inv = 1.0f / m;
            for (Eigen::Index i = 0; i < tile_rec.size(); ++i) {
              tile_rec.data()[i] *= inv;
            }
            tile_norm_scale[ti] = m;
          } else {
            tile_norm_scale[ti] = 1.0f;
          }
        }
      }

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
            static_cast<int>(tiles_phase56.size()),
            "workers=" + std::to_string(parallel_tiles), "tiles", log_file);
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

      if (apply_phase7_tile_norm) {
        // Methodik v3.2 §5.7.2 (optional): global robust tile background
        // restore.
        std::vector<float> bg_r_vals;
        std::vector<float> bg_g_vals;
        std::vector<float> bg_b_vals;
        std::vector<float> m_vals;
        bg_r_vals.reserve(tiles_phase56.size());
        bg_g_vals.reserve(tiles_phase56.size());
        bg_b_vals.reserve(tiles_phase56.size());
        m_vals.reserve(tiles_phase56.size());
        for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
          if (tile_valid_counts[ti] <= 0) {
            continue;
          }
          bg_r_vals.push_back(tile_norm_bg_r[ti]);
          bg_g_vals.push_back(tile_norm_bg_g[ti]);
          bg_b_vals.push_back(tile_norm_bg_b[ti]);
          m_vals.push_back(tile_norm_scale[ti]);
        }
        if (!bg_r_vals.empty() && !bg_g_vals.empty() && !bg_b_vals.empty() &&
            !m_vals.empty()) {
          const float bg_r = core::median_of(bg_r_vals);
          const float bg_g = core::median_of(bg_g_vals);
          const float bg_b = core::median_of(bg_b_vals);
          const float m_global = std::max(kEpsMedian, core::median_of(m_vals));
          for (int i = 0; i < recon_R.size(); ++i) {
            if (weight_sum.data()[i] > eps_ws) {
              recon_R.data()[i] = recon_R.data()[i] * m_global + bg_r;
              recon_G.data()[i] = recon_G.data()[i] * m_global + bg_g;
              recon_B.data()[i] = recon_B.data()[i] * m_global + bg_b;
            }
          }
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

      if (apply_phase7_tile_norm) {
        std::vector<float> bg_vals;
        std::vector<float> m_vals;
        bg_vals.reserve(tiles_phase56.size());
        m_vals.reserve(tiles_phase56.size());
        for (size_t ti = 0; ti < tiles_phase56.size(); ++ti) {
          if (tile_valid_counts[ti] > 0) {
            bg_vals.push_back(tile_norm_bg_r[ti]);
            m_vals.push_back(tile_norm_scale[ti]);
          }
        }
        if (!bg_vals.empty() && !m_vals.empty()) {
          const float bg = core::median_of(bg_vals);
          const float m_global = std::max(kEpsMedian, core::median_of(m_vals));
          for (int i = 0; i < recon.size(); ++i) {
            if (weight_sum.data()[i] > eps_ws) {
              recon.data()[i] = recon.data()[i] * m_global + bg;
            }
          }
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
    std::vector<float> synthetic_cluster_quality;

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
          subset_workers = std::min<int>(cfg.runtime_limits.parallel_workers,
                                         cpu_cores);
          subset_workers =
              std::min<int>(subset_workers, static_cast<int>(tiles_phase56.size()));
          subset_workers = std::max(1, subset_workers);
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
        std::vector<float> cluster_q_values;
        cluster_q_values.reserve(frames.size());
        const float k_global =
            std::max(cfg.global_metrics.weight_exponent_scale, kEpsWeight);
        const float q_min = cfg.global_metrics.clamp[0];
        const float q_max = cfg.global_metrics.clamp[1];
        for (size_t fi = 0; fi < frames.size(); ++fi) {
          if (cluster_labels[fi] != c)
            continue;
          use_frame[fi] = 1;
          if (frame_has_data[fi]) {
            count++;
            const float G_f = (fi < static_cast<size_t>(global_weights.size()))
                                  ? global_weights[static_cast<int>(fi)]
                                  : 1.0f;
            const float q_f = std::clamp(
                std::log(std::max(G_f, kEpsWeight)) / k_global, q_min, q_max);
            if (std::isfinite(q_f)) {
              cluster_q_values.push_back(q_f);
            }
          }
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
        const float q_k =
            cluster_q_values.empty() ? 0.0f : core::median_of(cluster_q_values);

        if (detected_mode == ColorMode::OSC) {
          auto deb = image::debayer_nearest_neighbor(syn, detected_bayer, 0, 0);
          RGBFrame rgb;
          rgb.R = std::move(deb.R);
          rgb.G = std::move(deb.G);
          rgb.B = std::move(deb.B);
          synthetic_rgb_frames.push_back(std::move(rgb));
        }

        synthetic_frames.push_back(std::move(syn));
        synthetic_cluster_quality.push_back(q_k);
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
        artifact["cluster_quality"] = core::json::array();
        for (float qk : synthetic_cluster_quality) {
          artifact["cluster_quality"].push_back(qk);
        }
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
      std::vector<float> valid_synth_q;
      valid_synth_q.reserve(synthetic_frames.size());

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
        if (i < synthetic_cluster_quality.size()) {
          valid_synth_q.push_back(synthetic_cluster_quality[i]);
        } else {
          valid_synth_q.push_back(0.0f);
        }
      }

      std::cerr << "[STACKING] " << valid_synth.size() << " / "
                << synthetic_frames.size() << " non-empty synthetic frames"
                << std::endl;

      if (!valid_synth.empty()) {
        const bool use_quality_weighting =
            cfg.stacking.cluster_quality_weighting.enabled;
        std::vector<float> cluster_stack_weights;
        if (use_quality_weighting) {
          cluster_stack_weights.resize(valid_synth_q.size(), 1.0f);
          const float kappa = cfg.stacking.cluster_quality_weighting.kappa_cluster;
          for (size_t i = 0; i < valid_synth_q.size(); ++i) {
            cluster_stack_weights[i] = std::exp(kappa * valid_synth_q[i]);
            if (!std::isfinite(cluster_stack_weights[i]) ||
                cluster_stack_weights[i] <= 0.0f) {
              cluster_stack_weights[i] = 1.0f;
            }
          }
          if (cfg.stacking.cluster_quality_weighting.cap_enabled &&
              !cluster_stack_weights.empty()) {
            std::vector<float> tmp_w = cluster_stack_weights;
            const float med_w = core::median_of(tmp_w);
            const float cap =
                std::max(kEpsWeight,
                         cfg.stacking.cluster_quality_weighting.cap_ratio * med_w);
            for (float &w : cluster_stack_weights) {
              if (w > cap)
                w = cap;
            }
          }
        }

        if (detected_mode == ColorMode::OSC &&
            !synth_R.empty() && synth_R.size() == valid_synth.size()) {
          if (!use_quality_weighting && cfg.stacking.method == "rej") {
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
            float wsum = 0.0f;
            for (size_t k = 0; k < synth_R.size(); ++k) {
              const float wk = use_quality_weighting
                                   ? cluster_stack_weights[k]
                                   : 1.0f;
              recon_R += synth_R[k] * wk;
              recon_G += synth_G[k] * wk;
              recon_B += synth_B[k] * wk;
              wsum += wk;
            }
            const float denom = std::max(kEpsWeight, wsum);
            recon_R /= denom;
            recon_G /= denom;
            recon_B /= denom;
          }
          recon = 0.25f * recon_R + 0.5f * recon_G + 0.25f * recon_B;
        } else {
          if (!use_quality_weighting && cfg.stacking.method == "rej") {
            recon = reconstruction::sigma_clip_stack(
                valid_synth, cfg.stacking.sigma_clip.sigma_low,
                cfg.stacking.sigma_clip.sigma_high,
                cfg.stacking.sigma_clip.max_iters,
                cfg.stacking.sigma_clip.min_fraction);
          } else {
            recon = Matrix2Df::Zero(valid_synth[0].rows(), valid_synth[0].cols());
            float wsum = 0.0f;
            for (size_t idx = 0; idx < valid_synth.size(); ++idx) {
              const float wk =
                  use_quality_weighting ? cluster_stack_weights[idx] : 1.0f;
              recon += valid_synth[idx] * wk;
              wsum += wk;
            }
            recon /= std::max(kEpsWeight, wsum);
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

        std::vector<float> boundary_ratios;
        boundary_ratios.reserve(xb.size() + yb.size());

        float worst_ratio = 1.0f;
        for (int x : xb) {
          float b = line_mean_x(x);
          float n = 0.5f * (line_mean_x(x - 2) + line_mean_x(x + 2));
          float r = b / (n + 1.0e-12f);
          boundary_ratios.push_back(r);
          if (r > worst_ratio)
            worst_ratio = r;
        }
        for (int y : yb) {
          float b = line_mean_y(y);
          float n = 0.5f * (line_mean_y(y - 2) + line_mean_y(y + 2));
          float r = b / (n + 1.0e-12f);
          boundary_ratios.push_back(r);
          if (r > worst_ratio)
            worst_ratio = r;
        }

        float p95_ratio = worst_ratio;
        if (!boundary_ratios.empty()) {
          const size_t p95_idx = static_cast<size_t>(
              std::floor(0.95 * static_cast<double>(boundary_ratios.size() - 1)));
          std::nth_element(boundary_ratios.begin(),
                           boundary_ratios.begin() + static_cast<long>(p95_idx),
                           boundary_ratios.end());
          p95_ratio = boundary_ratios[p95_idx];
        }

        v["tile_pattern_ratio"] = worst_ratio;
        v["tile_pattern_ratio_p95"] = p95_ratio;
        v["tile_pattern_boundary_count"] = static_cast<int>(boundary_ratios.size());
        tile_pattern_ok = (worst_ratio < 1.5f) && (p95_ratio < 1.25f);
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

}
