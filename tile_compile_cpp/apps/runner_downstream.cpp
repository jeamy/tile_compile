#include "runner_downstream.hpp"
#include "tile_compile/core/atomic_output.hpp"

#include "tile_compile/astrometry/gaia_catalog.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/pipeline_contract.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/background_extraction.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/image/hypermetric_stretch.hpp"
#include "tile_compile/image/normalization.hpp"
#include "tile_compile/image/processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/metrics/aqmh_frame_valid_mask.hpp"
#include "tile_compile/reconstruction/reconstruction.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"
#include "tile_compile/pipeline/adaptive_tile_grid.hpp"

#include "runner_shared.hpp"
#include "runner_phase_post_stack_output.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {
using namespace tile_compile;
namespace runner = tile_compile::runner;
/// @brief Parses tile metrics json.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
TileMetrics parse_tile_metrics_json(const tile_compile::core::json &j) {
  TileMetrics tm{};
  tm.fwhm = j.value("fwhm", 0.0f);
  tm.roundness = j.value("roundness", 0.0f);
  tm.contrast = j.value("contrast", 0.0f);
  tm.sharpness = j.value("sharpness", 0.0f);
  tm.background = j.value("background", 0.0f);
  tm.noise = j.value("noise", 0.0f);
  tm.gradient_energy = j.value("gradient_energy", 0.0f);
  tm.star_count = j.value("star_count", 0);
  tm.quality_score = j.value("quality_score", 0.0f);
  const std::string type = j.value("tile_type", "STRUCTURE");
  tm.type = (type == "STAR") ? TileType::STAR : TileType::STRUCTURE;
  return tm;
}

/// @brief Loads tile grid from artifact.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool load_tile_grid_from_artifact(const fs::path &tile_grid_path,
                                  TileGrid &out,
                                  std::string &error_out) {
  if (!fs::exists(tile_grid_path)) {
    error_out = "missing tile_grid.json";
    return false;
  }
  try {
    const auto j = tile_compile::core::json::parse(
        tile_compile::core::read_text(tile_grid_path));
    if (!j.contains("tiles") || !j["tiles"].is_array()) {
      error_out = "tile_grid.json missing tiles[]";
      return false;
    }

    out.tile_size = j.value("uniform_tile_size", 0);
    out.overlap_fraction = j.value("overlap_fraction", 0.0f);
    out.rows = 0;
    out.cols = 0;
    out.tiles.clear();
    out.tiles.reserve(j["tiles"].size());

    std::map<int, int> y_to_row;
    std::map<int, int> x_to_col;
    for (const auto &tj : j["tiles"]) {
      Tile t{};
      t.x = tj.value("x", 0);
      t.y = tj.value("y", 0);
      t.width = tj.value("width", 0);
      t.height = tj.value("height", 0);
      t.row = 0;
      t.col = 0;
      out.tiles.push_back(t);
      y_to_row.emplace(t.y, 0);
      x_to_col.emplace(t.x, 0);
    }

    if (out.tiles.empty()) {
      error_out = "tile_grid.json has no tiles";
      return false;
    }
    if (out.tile_size <= 0) {
      out.tile_size = std::max(1, out.tiles.front().width);
    }

    int row_idx = 0;
    for (auto &kv : y_to_row)
      kv.second = row_idx++;
    int col_idx = 0;
    for (auto &kv : x_to_col)
      kv.second = col_idx++;

    for (auto &t : out.tiles) {
      t.row = y_to_row[t.y];
      t.col = x_to_col[t.x];
    }
    out.rows = static_cast<int>(y_to_row.size());
    out.cols = static_cast<int>(x_to_col.size());
    return true;
  } catch (const std::exception &e) {
    error_out = std::string("tile_grid parse failed: ") + e.what();
    return false;
  }
}

/// @brief Loads aggregated tile metrics.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool load_aggregated_tile_metrics(const fs::path &local_metrics_path,
                                  std::vector<TileMetrics> &out,
                                  std::string &error_out) {
  if (!fs::exists(local_metrics_path)) {
    error_out = "missing local_metrics.json";
    return false;
  }
  try {
    const auto j = tile_compile::core::json::parse(
        tile_compile::core::read_text(local_metrics_path));
    if (!j.contains("tile_metrics") || !j["tile_metrics"].is_array() ||
        j["tile_metrics"].empty()) {
      error_out = "local_metrics.json missing tile_metrics[][]";
      return false;
    }

    const auto &all_frames = j["tile_metrics"];
    size_t n_tiles = 0;
    if (all_frames.front().is_array()) {
      n_tiles = all_frames.front().size();
    }
    if (n_tiles == 0) {
      error_out = "local_metrics.json has zero tiles";
      return false;
    }

    const bool consistent = std::all_of(
        all_frames.begin(), all_frames.end(),
        [n_tiles](const auto &fm) { return fm.is_array() && fm.size() == n_tiles; });

    std::vector<std::vector<TileMetrics>> parsed_metrics(all_frames.size());
    for (size_t f = 0; f < all_frames.size(); ++f) {
      const auto &fm = all_frames[f];
      size_t f_tiles = fm.is_array() ? fm.size() : 0;
      parsed_metrics[f].reserve(f_tiles);
      for (size_t t = 0; t < f_tiles; ++t) {
        parsed_metrics[f].push_back(parse_tile_metrics_json(fm[t]));
      }
    }

    if (!consistent) {
      out = parsed_metrics.empty() ? std::vector<TileMetrics>() : parsed_metrics.front();
      return !out.empty();
    }

    out = tile_compile::runner::aggregate_tile_metrics_across_frames(parsed_metrics);
    return true;
  } catch (const std::exception &e) {
    error_out = std::string("local_metrics parse failed: ") + e.what();
    return false;
  }
}

std::vector<TileMetrics> build_aqmh_bge_tile_metrics_from_rgb(
    const TileGrid &grid, const tile_compile::Matrix2Df &R,
    const tile_compile::Matrix2Df &G, const tile_compile::Matrix2Df &B,
    const std::vector<uint8_t> &valid_mask, int mask_rows, int mask_cols) {
  std::vector<TileMetrics> out;
  out.reserve(grid.tiles.size());
  const bool mask_ok =
      mask_rows == R.rows() && mask_cols == R.cols() &&
      G.rows() == R.rows() && B.rows() == R.rows() &&
      G.cols() == R.cols() && B.cols() == R.cols() &&
      valid_mask.size() == static_cast<size_t>(R.rows() * R.cols());

  for (const auto &tile : grid.tiles) {
    TileMetrics tm{};
    tm.fwhm = 0.0f;
    tm.roundness = 0.0f;
    tm.contrast = 0.0f;
    tm.sharpness = 0.0f;
    tm.background = 0.0f;
    tm.noise = 0.0f;
    tm.gradient_energy = 0.0f;
    tm.star_count = 0;
    tm.type = TileType::STRUCTURE;
    tm.quality_score = 0.0f;

    const int x0 = std::max(0, tile.x);
    const int y0 = std::max(0, tile.y);
    const int x1 = std::min(tile.x + tile.width, static_cast<int>(R.cols()));
    const int y1 = std::min(tile.y + tile.height, static_cast<int>(R.rows()));
    if (x1 <= x0 || y1 <= y0 || !mask_ok) {
      out.push_back(tm);
      continue;
    }

    std::vector<float> values;
    values.reserve(static_cast<size_t>((x1 - x0) * (y1 - y0)));
    double gradient_sum = 0.0;
    size_t gradient_count = 0;
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) {
        const size_t idx =
            static_cast<size_t>(y) * static_cast<size_t>(mask_cols) +
            static_cast<size_t>(x);
        if (valid_mask[idx] == 0) {
          continue;
        }
        const float rv = R(y, x);
        const float gv = G(y, x);
        const float bv = B(y, x);
        if (!(std::isfinite(rv) && std::isfinite(gv) && std::isfinite(bv))) {
          continue;
        }
        const float luma = 0.2126f * rv + 0.7152f * gv + 0.0722f * bv;
        if (!std::isfinite(luma)) {
          continue;
        }
        values.push_back(luma);

        const int xm = std::max(x0, x - 1);
        const int xp = std::min(x1 - 1, x + 1);
        const int ym = std::max(y0, y - 1);
        const int yp = std::min(y1 - 1, y + 1);
        const float l_xm =
            0.2126f * R(y, xm) + 0.7152f * G(y, xm) + 0.0722f * B(y, xm);
        const float l_xp =
            0.2126f * R(y, xp) + 0.7152f * G(y, xp) + 0.0722f * B(y, xp);
        const float l_ym =
            0.2126f * R(ym, x) + 0.7152f * G(ym, x) + 0.0722f * B(ym, x);
        const float l_yp =
            0.2126f * R(yp, x) + 0.7152f * G(yp, x) + 0.0722f * B(yp, x);
        if (std::isfinite(l_xm) && std::isfinite(l_xp) &&
            std::isfinite(l_ym) && std::isfinite(l_yp)) {
          gradient_sum += std::fabs(l_xp - l_xm) + std::fabs(l_yp - l_ym);
          ++gradient_count;
        }
      }
    }
    if (!values.empty()) {
      std::vector<float> median_values = values;
      tm.background = tile_compile::core::median_of(median_values);
      std::vector<float> noise_values = values;
      tm.noise = tile_compile::core::robust_sigma_mad(noise_values);
      tm.gradient_energy =
          gradient_count > 0
              ? static_cast<float>(gradient_sum /
                                   static_cast<double>(gradient_count))
              : 0.0f;
      tm.contrast = tm.gradient_energy;
      tm.sharpness = tm.gradient_energy;
    }
    out.push_back(tm);
  }

  return out;
}

tile_compile::image::HyperMetricStretchConfig to_image_hms_config(
    const tile_compile::config::HyperMetricStretchConfig &src) {
  tile_compile::image::HyperMetricStretchConfig dst;
  dst.enabled = src.enabled;
  dst.require_successful_pcc = src.require_successful_pcc;
  dst.mode = src.mode;
  dst.sensor_profile = src.sensor_profile;
  dst.fallback_profile = src.fallback_profile;
  dst.adaptive_anchor = src.adaptive_anchor;
  dst.target_bg = src.target_bg;
  dst.protect_b = src.protect_b;
  dst.convergence_power = src.convergence_power;
  dst.log_d_mode = src.log_d_mode;
  dst.fixed_log_d = src.fixed_log_d;
  dst.color_strategy = src.color_strategy;
  dst.fixed_color_strategy = src.fixed_color_strategy;
  dst.color_grip = src.color_grip;
  dst.shadow_convergence = src.shadow_convergence;
  dst.linear_expansion = src.linear_expansion;
  dst.write_channels = src.write_channels;
  dst.output_rgb = src.output_rgb;
  return dst;
}

}

namespace tile_compile::runner {
int run_rgb_downstream(const fs::path &run_dir, const std::string &run_id,
    const config::Config &cfg, std::string phase_upper, std::ostream &log_file,
    const std::function<bool(const std::string &)> &abort_if_runtime_limit_exceeded,
    bool forward) {
  namespace core = tile_compile::core;
  namespace io = tile_compile::io;
  namespace astro = tile_compile::astrometry;
  namespace image = tile_compile::image;
  std::string phase_l = core::to_lower(phase_upper);
  auto write_atomic_rgb = [](const fs::path &path, const Matrix2Df &r,
      const Matrix2Df &g, const Matrix2Df &b, const io::FitsHeader &header) {
    core::AtomicOutput out(path);
    io::write_fits_rgb(out.path(),r,g,b,header);
    out.commit();
  };

  fs::path stacked_rgb_path = run_dir / "outputs" / "stacked_rgb.fits";
  fs::path stacked_rgb_solve_path = run_dir / "outputs" / "stacked_rgb_solve.fits";
  if (fs::exists(stacked_rgb_solve_path) && fs::exists(stacked_rgb_path)) {
    try {
      const auto [solve_w, solve_h, solve_planes] =
          io::get_fits_dimensions(stacked_rgb_solve_path);
      const auto [rgb_w, rgb_h, rgb_planes] =
          io::get_fits_dimensions(stacked_rgb_path);
      if ((solve_planes != 3 || solve_w != rgb_w || solve_h != rgb_h) &&
          rgb_planes == 3) {
        auto stacked_rgb = io::read_fits_rgb(stacked_rgb_path);
        write_atomic_rgb(stacked_rgb_solve_path, stacked_rgb.R,
                           stacked_rgb.G, stacked_rgb.B, stacked_rgb.header);
        std::cout << "[ASTROMETRY][resume] Rewrote stale "
                  << "stacked_rgb_solve.fits as RGB cube from stacked_rgb.fits"
                  << std::endl;
      }
    } catch (const std::exception &e) {
      std::cout << "[ASTROMETRY][resume] Could not validate "
                << "stacked_rgb_solve.fits shape: " << e.what() << std::endl;
    }
  }
  fs::path rgb_path = stacked_rgb_solve_path;
  if (!fs::exists(rgb_path)) {
    rgb_path = stacked_rgb_path;
  }
  if (!fs::exists(rgb_path)) {
    std::cerr << "Error: missing stacked RGB cube in run outputs" << std::endl;
    core::emit_event((forward ? "downstream_end" : "resume_end"), run_id,
                     {{"success", false},
                      {"status", "missing_rgb"},
                      {"error", "missing stacked RGB cube in run outputs"}},
                     log_file);
    return 1;
  }

  io::RGBImage rgb;
  try {
    rgb = io::read_fits_rgb(rgb_path);
  } catch (const std::exception &e) {
    std::cerr << "Error: failed to read RGB FITS: " << e.what() << std::endl;
    core::emit_event((forward ? "downstream_end" : "resume_end"), run_id,
                     {{"success", false},
                      {"status", "read_rgb_failed"},
                      {"error", e.what()}},
                     log_file);
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

  astro::WCS wcs;
  bool have_wcs = false;
  fs::path wcs_path = run_dir / "artifacts" / "stacked_rgb.wcs";
  if (!fs::exists(wcs_path)) {
    fs::path wcs_path2 = rgb_path;
    wcs_path2.replace_extension(".wcs");
    if (fs::exists(wcs_path2))
      wcs_path = wcs_path2;
  }
  if (!forward && fs::exists(wcs_path)) {
    try {
      wcs = astro::parse_wcs_file(wcs_path.string());
      have_wcs = wcs.valid();
    } catch (const std::exception &) {
      have_wcs = false;
    }
  }

  std::string astrometry_resume_error;

  auto resolve_astrometry_image_height = [&]() -> int {
    if (cfg.data.image_height > 0) {
      return cfg.data.image_height;
    }
    const fs::path events_path = run_dir / "logs" / "run_events.jsonl";
    std::ifstream in(events_path);
    std::string line;
    while (std::getline(in, line)) {
      try {
        const auto event = core::json::parse(line);
        if (event.value("phase_name", std::string()) != "SCAN_INPUT" ||
            event.value("status", std::string()) != "ok" ||
            !event.contains("image_height")) {
          continue;
        }
        const int image_height = event["image_height"].get<int>();
        if (image_height > 0) {
          return image_height;
        }
      } catch (const std::exception &) {
      }
    }
    return 0;
  };

  bool astrometry_ran = false;
  auto run_astrometry_if_needed = [&](bool force_rerun = false) -> bool {
    if (forward && astrometry_ran) return have_wcs || !cfg.astrometry.enabled;
    astrometry_ran = true;
    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::ASTROMETRY, "ASTROMETRY", log_file);

    if (have_wcs && !force_rerun) {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "existing_wcs"},
                         {"wcs_file",
                          (run_dir / "artifacts" / "stacked_rgb.wcs")
                              .string()}},
                        log_file);
      return true;
    }

    if (force_rerun) {
      have_wcs = false;
    }

    if (!cfg.astrometry.enabled) {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "disabled"}}, log_file);
      return true;
    }

    std::string astap_data = cfg.astrometry.astap_data_dir;
    if (astap_data.empty()) {
#ifdef _WIN32
      if (const char *la = std::getenv("LOCALAPPDATA"); la && la[0] != '\0') {
        astap_data = std::string(la) + "\\tile_compile\\astap";
      }
#else
      const char *home = std::getenv("HOME");
      if (home)
        astap_data = std::string(home) + "/.local/share/tile_compile/astap";
#endif
    }
    fs::path astap_bin_path = runner::resolve_astap_binary_path(cfg.astrometry.astap_bin, astap_data);
    // If the resolved binary lives outside the configured data dir, use its parent as data dir
    if (!astap_bin_path.empty()) {
      std::error_code ec;
      fs::path data_dir_path(astap_data);
      auto relative = fs::relative(astap_bin_path, data_dir_path, ec);
      if (ec || relative.empty() || relative.begin() == relative.end() || *relative.begin() == "..") {
        astap_data = astap_bin_path.parent_path().string();
      }
    }

    if (astap_bin_path.empty()) {
      const std::string reported_bin = cfg.astrometry.astap_bin.empty() ? astap_data + "/astap_cli" : cfg.astrometry.astap_bin;
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "astap_not_found"},
                         {"astap_bin", reported_bin}},
                        log_file);
      return true;
    }

    const auto astap_stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path astap_output_prefix =
        fs::temp_directory_path() /
        ("tile_compile_astap_" + run_id + "_" +
         std::to_string(astap_stamp));
    fs::path wcs_out = astap_output_prefix;
    wcs_out.replace_extension(".wcs");

    std::string cmd = runner::shell_quote(astap_bin_path.string()) + " -f " +
                      runner::shell_quote(rgb_path.string()) + " -d " +
                      runner::shell_quote(astap_data) + " -r " +
                      std::to_string(cfg.astrometry.search_radius) +
                      " -wcs -o " +
                      runner::shell_quote(astap_output_prefix.string());

    std::cout << "[ASTROMETRY][resume] Running: " << cmd << std::endl;
    int ret = std::system(runner::system_cmd(cmd).c_str());
    std::string astrometry_solver = "astap";
    int local_gaia_image_stars = 0;
    int local_gaia_catalog_stars = 0;
    int local_gaia_inlier_stars = 0;
    bool local_gaia_reflected = false;
    std::string local_gaia_error = "not_attempted";

    if (ret == 0 && fs::exists(wcs_out)) {
      try {
        wcs = astro::parse_wcs_file(wcs_out.string());
        have_wcs = wcs.valid();
      } catch (const std::exception &) {
        have_wcs = false;
      }
    }

    const int solve_height = rgb.R.rows() > 0
                                ? static_cast<int>(rgb.R.rows())
                                : resolve_astrometry_image_height();
    if (!have_wcs && solve_height > 0) {
      const auto canvas_fov_deg =
          runner::estimate_astap_sensor_fov_deg(rgb.header, solve_height);
      if (canvas_fov_deg) {
        const std::vector<double> fov_candidates = {
            *canvas_fov_deg * 0.75,
            *canvas_fov_deg * 0.60,
            *canvas_fov_deg,
            *canvas_fov_deg * 0.50};
        for (double cand_fov : fov_candidates) {
          std::ostringstream fov_ss;
          fov_ss << std::fixed << std::setprecision(3) << cand_fov;
          const std::string retry_cmd = cmd + " -fov " + fov_ss.str();
          std::cout << "[ASTROMETRY][resume] Retry with FOV " << fov_ss.str()
                    << " deg: " << retry_cmd << std::endl;
          ret = std::system(retry_cmd.c_str());
          if (ret == 0 && fs::exists(wcs_out)) {
            try {
              wcs = astro::parse_wcs_file(wcs_out.string());
              have_wcs = wcs.valid();
              if (have_wcs) break;
            } catch (const std::exception &) {
              have_wcs = false;
            }
          }
        }
      }
    }

    if (!have_wcs) {
      try {
        const auto local_gaia = runner::solve_with_local_gaia_catalog(rgb);
        local_gaia_image_stars = local_gaia.image_stars;
        local_gaia_catalog_stars = local_gaia.catalog_stars;
        local_gaia_inlier_stars = local_gaia.inlier_stars;
        local_gaia_reflected = local_gaia.reflected_solution;
        local_gaia_error = local_gaia.error_message;
        if (local_gaia.success && runner::write_wcs_sidecar(local_gaia.wcs, wcs_out)) {
          wcs = local_gaia.wcs;
          have_wcs = true;
          astrometry_solver = "local_gaia_dr3";
          std::cout << "[ASTROMETRY][resume] Local Gaia DR3 fallback solved: image_stars="
                    << local_gaia_image_stars << " catalog_stars="
                    << local_gaia_catalog_stars << " inliers="
                    << local_gaia_inlier_stars << " reflected="
                    << (local_gaia_reflected ? "true" : "false") << std::endl;
        } else {
          std::cout << "[ASTROMETRY][resume] Local Gaia DR3 fallback failed: "
                    << local_gaia.error_message << std::endl;
        }
      } catch (const std::exception &e) {
        local_gaia_error = e.what();
        std::cerr << "[ASTROMETRY][resume] Local Gaia DR3 fallback error: "
                  << e.what() << std::endl;
      }
    }

    if (have_wcs) {
      fs::path wcs_artifact = run_dir / "artifacts" / "stacked_rgb.wcs";
      try {
        fs::create_directories(wcs_artifact.parent_path());
        fs::copy_file(wcs_out, wcs_artifact,
                      fs::copy_options::overwrite_existing);
      } catch (const std::exception &) {
      }

      emitter.phase_end(run_id, Phase::ASTROMETRY, "ok",
                        {{"solver", astrometry_solver},
                         {"ra", wcs.crval1},
                         {"dec", wcs.crval2},
                         {"pixel_scale_arcsec", wcs.pixel_scale_arcsec()},
                         {"rotation_deg", wcs.rotation_deg()},
                         {"local_gaia_image_stars", local_gaia_image_stars},
                         {"local_gaia_catalog_stars", local_gaia_catalog_stars},
                         {"local_gaia_inlier_stars", local_gaia_inlier_stars},
                         {"local_gaia_reflected", local_gaia_reflected},
                         {"wcs_file",
                          (run_dir / "artifacts" / "stacked_rgb.wcs")
                              .string()}},
                        log_file);
      return true;
    } else {
      // Re-solve failed — try to fall back to existing WCS file
      if (!forward && fs::exists(wcs_path)) {
        try {
          wcs = astro::parse_wcs_file(wcs_path.string());
          have_wcs = wcs.valid();
        } catch (const std::exception &) {
          have_wcs = false;
        }
      }
      if (have_wcs) {
        emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                          {{"reason", "solve_failed_existing_wcs"}, {"exit_code", ret}},
                          log_file);
        return true;
      } else {
        astrometry_resume_error =
            "astap plate solve failed (exit code " + std::to_string(ret) +
            "); local Gaia DR3 fallback: " + local_gaia_error;
        emitter.phase_end(run_id, Phase::ASTROMETRY, "error",
                          {{"reason", "solve_failed"},
                           {"exit_code", ret},
                           {"local_gaia_error", local_gaia_error},
                           {"local_gaia_image_stars", local_gaia_image_stars},
                           {"local_gaia_catalog_stars", local_gaia_catalog_stars},
                           {"local_gaia_inlier_stars", local_gaia_inlier_stars},
                           {"local_gaia_reflected", local_gaia_reflected}},
                          log_file);
        return false;
      }
    }
  };

  fs::path stacked_rgb_bge_path = run_dir / "outputs" / "stacked_rgb_bge.fits";
  fs::path stacked_rgb_bge_linear_path =
      run_dir / "outputs" / "stacked_rgb_bge_linear.fits";
  std::vector<TileMetrics> bge_tile_metrics;
  TileGrid bge_tile_grid;
  bool bge_have_local_metrics = false;
  bool bge_have_bge_grid = false;
  bool bge_metrics_tiles_match = false;
  bool bge_tile_context_loaded = false;
  std::string bge_tile_metrics_source = "none";
  bool seeing_fwhm_loaded = false;
  bool have_seeing_fwhm = false;
  double seeing_fwhm_median = 0.0;

  auto load_seeing_fwhm_if_needed = [&]() {
    if (seeing_fwhm_loaded) return;
    seeing_fwhm_loaded = true;

    const std::array<fs::path, 2> candidates = {
        run_dir / "artifacts" / "tile_grid.json",
        run_dir / "artifacts" / "validation.json"};
    for (const auto &path : candidates) {
      if (!fs::exists(path)) continue;
      try {
        const auto j = core::json::parse(core::read_text(path));
        if (j.contains("seeing_fwhm_median") &&
            j["seeing_fwhm_median"].is_number()) {
          const double f = j["seeing_fwhm_median"].get<double>();
          if (std::isfinite(f) && f > 0.0) {
            seeing_fwhm_median = f;
            have_seeing_fwhm = true;
            return;
          }
        }
      } catch (const std::exception &) {
      }
    }
  };

  auto load_bge_tile_context_if_needed = [&]() {
    if (bge_tile_context_loaded) return;
    bge_tile_context_loaded = true;

    std::string local_err;
    std::string grid_err;
    const bool ok_local = !forward && load_aggregated_tile_metrics(
        run_dir / "artifacts" / "local_metrics.json", bge_tile_metrics, local_err);
    const bool ok_grid = !forward && load_tile_grid_from_artifact(
        run_dir / "artifacts" / "tile_grid.json", bge_tile_grid, grid_err);

    bge_have_local_metrics = ok_local && !bge_tile_metrics.empty();
    // Set BGE tile metrics source based on reconstruction method
    if (forward || cfg.method == "aqmh") {
        bge_tile_metrics_source = "aqmh_output";
    } else {
        bge_tile_metrics_source = bge_have_local_metrics ? "classic_local_metrics" : "none";
    }
    bge_have_bge_grid = ok_grid && !bge_tile_grid.tiles.empty();
    if ((forward || cfg.aqmh.enabled) && !bge_have_bge_grid && rgb.R.rows() > 0 &&
        rgb.R.cols() > 0) {
      load_seeing_fwhm_if_needed();
      const float fwhm = have_seeing_fwhm
                             ? static_cast<float>(seeing_fwhm_median)
                             : 3.0f;
      const int tmin = std::max(16, cfg.tile.min_size);
      const int divisor = std::max(1, cfg.tile.max_divisor);
      const int tmax = std::max(
          tmin, std::min(static_cast<int>(rgb.R.cols()),
                         static_cast<int>(rgb.R.rows())) /
                    divisor);
      const int tile_size = static_cast<int>(std::floor(std::clamp(
          static_cast<float>(cfg.tile.size_factor) * fwhm,
          static_cast<float>(tmin), static_cast<float>(tmax))));
      const float overlap =
          std::clamp(cfg.tile.overlap_fraction, 0.0f, 0.5f);
      bge_tile_grid.tile_size = tile_size;
      bge_tile_grid.overlap_fraction = overlap;
      bge_tile_grid.tiles = tile_compile::pipeline::build_initial_tile_grid(
          rgb.R.cols(), rgb.R.rows(), tile_size, overlap);
      for (const auto &tile : bge_tile_grid.tiles) {
        bge_tile_grid.rows = std::max(bge_tile_grid.rows, tile.row + 1);
        bge_tile_grid.cols = std::max(bge_tile_grid.cols, tile.col + 1);
      }
      bge_have_bge_grid = !bge_tile_grid.tiles.empty();
      if (bge_have_bge_grid) {
        std::cout << "[BGE][resume] Reconstructed AQMH BGE grid from RGB output: "
                  << bge_tile_grid.tiles.size() << " tiles, tile_size="
                  << tile_size << ", overlap=" << overlap << std::endl;
      }
    }
    bge_metrics_tiles_match =
        bge_have_local_metrics && bge_have_bge_grid &&
        (bge_tile_metrics.size() == bge_tile_grid.tiles.size());

    if (!ok_local) {
      std::cout << "[BGE][resume] Warning: " << local_err << std::endl;
    }
    if (!ok_grid) {
      std::cout << "[BGE][resume] Warning: " << grid_err << std::endl;
    }
  };

	  auto write_stretched_rgb_snapshot = [&](const fs::path &path,
	                                          const Matrix2Df &R_src,
	                                          const Matrix2Df &G_src,
	                                          const Matrix2Df &B_src,
	                                          const io::FitsHeader &hdr,
	                                          bool apply_stretch,
	                                          const char* stage_tag) {
	    std::vector<uint8_t> canvas_mask;
	    std::vector<uint8_t> statistics_mask;
	    std::string canvas_mask_error;
	    std::string statistics_mask_error;
	    int canvas_rows = 0;
	    int canvas_cols = 0;
	    int statistics_rows = 0;
	    int statistics_cols = 0;
	    tile_compile::runner::load_canvas_mask_for_rgb(
	            run_dir / "outputs" / "canvas_mask.fits", R_src, G_src, B_src,
	            canvas_mask, canvas_rows, canvas_cols, canvas_mask_error);
	    if (!tile_compile::runner::load_canvas_mask_for_rgb(
	            run_dir / "outputs" / "common_overlap_mask.fits", R_src, G_src,
	            B_src, statistics_mask, statistics_rows, statistics_cols,
	            statistics_mask_error)) {
	      statistics_mask = canvas_mask;
	    }
	    runner::write_stretched_rgb_snapshot(
	        path, R_src, G_src, B_src, canvas_mask, statistics_mask, canvas_rows,
	        canvas_cols, hdr, apply_stretch, stage_tag);
  };

  auto write_linear_rgb_snapshot = [&](const fs::path &path,
                                       const Matrix2Df &R_src,
                                       const Matrix2Df &G_src,
                                       const Matrix2Df &B_src,
                                       const io::FitsHeader &hdr) {
    write_stretched_rgb_snapshot(path, R_src, G_src, B_src, hdr, false,
                                 "BGE");
  };

  std::string bge_resume_error;

  auto run_bge_phase = [&]() -> bool {
    namespace image = tile_compile::image;
    core::EventEmitter emitter;
    const std::string bge_phase_label =
        (cfg.bge.method == "none")    ? "BGE (Skipped)" :
        (cfg.bge.method == "classic") ? "BGE (Classic)" :
                                        "BGE (AutoBGE)";
    emitter.phase_start(run_id, Phase::BGE, "BGE", log_file,
                        {{"label", bge_phase_label},
                         {"bge_method", cfg.bge.method}});

    io::FitsHeader bge_hdr = rgb.header;
    if (have_wcs) {
      inject_wcs_keywords(bge_hdr, wcs);
    }

    if (cfg.bge.method == "none") {
      std::error_code ec_linear;
      std::error_code ec_display;
      fs::remove(stacked_rgb_bge_linear_path, ec_linear);
      fs::remove(stacked_rgb_bge_path, ec_display);
      emitter.phase_end(run_id, Phase::BGE, "skipped",
                        {{"reason", "disabled"},
                         {"bge_method", cfg.bge.method},
                         {"artifact", (run_dir / "artifacts" / "bge.json").string()}},
                        log_file);
      return true;
    }

    load_bge_tile_context_if_needed();

    image::BGEDiagnostics bge_diag;
    image::BGEConfig bge_cfg =
        tile_compile::runner::to_image_bge_config(cfg.bge);
    bge_cfg.max_workers = cfg.runtime_limits.parallel_workers;
    std::string mask_error;
    const int rows = static_cast<int>(rgb.R.rows());
    const int cols = static_cast<int>(rgb.R.cols());
    if (rows <= 0 || cols <= 0 || rgb.G.rows() != rows ||
        rgb.B.rows() != rows || rgb.G.cols() != cols ||
        rgb.B.cols() != cols) {
      mask_error = "invalid RGB dimensions";
      bge_resume_error = mask_error;
      emitter.phase_end(run_id, Phase::BGE, "error",
                        {{"reason", "output_canvas_mask_invalid"},
                         {"error", mask_error}},
                        log_file);
      return false;
    }
    if (!tile_compile::runner::load_canvas_mask_for_rgb(
            run_dir / "outputs" / "canvas_mask.fits", rgb.R, rgb.G, rgb.B,
            bge_cfg.common_valid_mask, bge_cfg.common_mask_rows,
            bge_cfg.common_mask_cols, mask_error)) {
      bge_resume_error = mask_error;
      emitter.phase_end(run_id, Phase::BGE, "error",
                        {{"reason", "output_canvas_mask_invalid"},
                         {"error", mask_error}},
                        log_file);
      return false;
    }
    std::cout << "[BGE][resume] Using canvas mask from outputs/canvas_mask.fits ("
              << bge_cfg.common_mask_cols << "x" << bge_cfg.common_mask_rows
              << ")" << std::endl;
    tile_compile::runner::apply_autobge_exclusion_polygons(
        cfg.bge, rows, cols, bge_cfg);

    if (cfg.aqmh.enabled && !bge_have_local_metrics && bge_have_bge_grid) {
      bge_tile_metrics = build_aqmh_bge_tile_metrics_from_rgb(
          bge_tile_grid, rgb.R, rgb.G, rgb.B, bge_cfg.common_valid_mask,
          bge_cfg.common_mask_rows, bge_cfg.common_mask_cols);
      bge_tile_metrics_source = "aqmh_output";
    }

    const bool bge_have_tile_metrics = !bge_tile_metrics.empty();
    const bool bge_have_tile_data = bge_have_tile_metrics && bge_have_bge_grid;
    bge_metrics_tiles_match =
        bge_have_tile_data &&
        (bge_tile_metrics.size() == bge_tile_grid.tiles.size());

    if (cfg.bge.method == "autobge" ||
        (bge_have_tile_data && bge_metrics_tiles_match)) {
      Matrix2Df R_bge = rgb.R;
      Matrix2Df G_bge = rgb.G;
      Matrix2Df B_bge = rgb.B;
      const bool bge_success = image::apply_background_extraction(
          R_bge, G_bge, B_bge, bge_tile_metrics, bge_tile_grid,
          bge_cfg, &bge_diag);
      if (bge_success) {
        rgb.R = std::move(R_bge);
        rgb.G = std::move(G_bge);
        rgb.B = std::move(B_bge);
      }
    } else {
      std::cout << "[BGE][resume] Skipping BGE fit (missing/mismatched tile artifacts)"
                << std::endl;
    }

    core::json bge_artifact = tile_compile::runner::bge_diag_to_json(
        bge_diag, (cfg.bge.method != "none"), bge_have_tile_data, bge_metrics_tiles_match);
    bge_artifact["have_local_metrics"] = bge_have_local_metrics;
    bge_artifact["have_tile_metrics"] = bge_have_tile_metrics;
    bge_artifact["tile_metrics_source"] = bge_tile_metrics_source;
    bge_artifact["have_bge_grid"] = bge_have_bge_grid;
    bge_artifact["local_metrics_tiles"] = static_cast<int>(bge_tile_metrics.size());
    bge_artifact["bge_grid_tiles"] = static_cast<int>(bge_tile_grid.tiles.size());
    bge_artifact["config"] = {
        {"enabled", (cfg.bge.method != "none")},
        {"method", cfg.bge.method},
        {"autobge",
         {
             {"num_sample_points", cfg.bge.autobge.num_sample_points},
             {"poly_degree", cfg.bge.autobge.poly_degree},
             {"rbf_smooth", cfg.bge.autobge.rbf_smooth},
             {"downsample_scale", cfg.bge.autobge.downsample_scale},
             {"patch_size", cfg.bge.autobge.patch_size},
             {"patch_estimator", cfg.bge.autobge.patch_estimator},
             {"stretch_mode", cfg.bge.autobge.stretch_mode},
             {"stretch_target_median", cfg.bge.autobge.stretch_target_median},
             {"border_margin", cfg.bge.autobge.border_margin},
             {"bright_exclusion_fraction",
              cfg.bge.autobge.bright_exclusion_fraction},
             {"gradient_descent_max_iters",
              cfg.bge.autobge.gradient_descent_max_iters},
             {"random_seed", cfg.bge.autobge.random_seed},
             {"normalize_between_stages",
              cfg.bge.autobge.normalize_between_stages},
             {"apply_guards", cfg.bge.autobge.apply_guards},
             {"mono_mode", cfg.bge.autobge.mono_mode},
         }},
        {"classic",
         {
        {"sample_quantile", cfg.bge.sample_quantile},
        {"sample_estimator", cfg.bge.sample_estimator},
        {"min_sample_bg_value", cfg.bge.min_sample_bg_value},
        {"structure_thresh_percentile", cfg.bge.structure_thresh_percentile},
        {"min_tiles_per_cell", cfg.bge.min_tiles_per_cell},
        {"min_valid_sample_fraction_for_apply",
         cfg.bge.min_valid_sample_fraction_for_apply},
        {"min_valid_samples_for_apply", cfg.bge.min_valid_samples_for_apply},
        {"tile_weight_lambda_structure",
         cfg.bge.tile_weight_lambda_structure},
        {"mask",
         {
             {"star_dilate_px", cfg.bge.mask.star_dilate_px},
             {"sat_dilate_px", cfg.bge.mask.sat_dilate_px},
         }},
        {"grid",
         {
             {"N_g", cfg.bge.grid.N_g},
             {"G_min_px", cfg.bge.grid.G_min_px},
             {"G_max_fraction", cfg.bge.grid.G_max_fraction},
             {"insufficient_cell_strategy", cfg.bge.grid.insufficient_cell_strategy},
         }},
        {"fit",
         {
             {"method", cfg.bge.fit.method},
             {"robust_loss", cfg.bge.fit.robust_loss},
             {"huber_delta", cfg.bge.fit.huber_delta},
             {"irls_max_iterations", cfg.bge.fit.irls_max_iterations},
             {"irls_tolerance", cfg.bge.fit.irls_tolerance},
             {"polynomial_order", cfg.bge.fit.polynomial_order},
             {"rbf_phi", cfg.bge.fit.rbf_phi},
             {"rbf_mu_factor", cfg.bge.fit.rbf_mu_factor},
             {"rbf_lambda", cfg.bge.fit.rbf_lambda},
             {"rbf_epsilon", cfg.bge.fit.rbf_epsilon},
         }},
        {"autotune",
         {
             {"enabled", cfg.bge.autotune.enabled},
             {"max_evals", cfg.bge.autotune.max_evals},
             {"holdout_fraction", cfg.bge.autotune.holdout_fraction},
             {"alpha_flatness", cfg.bge.autotune.alpha_flatness},
             {"beta_roughness", cfg.bge.autotune.beta_roughness},
             {"strategy", cfg.bge.autotune.strategy},
         }},
        }},
    };
    const fs::path bge_artifact_path = run_dir / "artifacts" / "bge.json";
    core::write_text(bge_artifact_path, bge_artifact.dump(2));
    if (bge_diag.success) {
      write_linear_rgb_snapshot(stacked_rgb_bge_linear_path, rgb.R, rgb.G, rgb.B,
                                bge_hdr);
      write_stretched_rgb_snapshot(stacked_rgb_bge_path, rgb.R, rgb.G, rgb.B,
                                  bge_hdr, cfg.stacking.output_stretch, "BGE");
    } else {
      std::error_code ec_linear;
      std::error_code ec_display;
      fs::remove(stacked_rgb_bge_linear_path, ec_linear);
      fs::remove(stacked_rgb_bge_path, ec_display);
    }

    core::json phase_extra = {
        {"requested", (cfg.bge.method != "none")},
        {"attempted", bge_diag.attempted},
        {"success", bge_diag.success},
        {"have_tile_data", bge_have_tile_data},
        {"metrics_tiles_match", bge_metrics_tiles_match},
        {"artifact", bge_artifact_path.string()},
    };
    if (cfg.bge.method != "autobge" && !bge_have_tile_data) {
      phase_extra["reason"] = "no_tile_data";
    } else if (cfg.bge.method != "autobge" && !bge_metrics_tiles_match) {
      phase_extra["reason"] = "tile_metric_grid_mismatch";
    } else if (bge_diag.attempted && !bge_diag.success) {
      phase_extra["reason"] =
          bge_diag.failure_reason.empty() ? "fit_failed"
                                          : bge_diag.failure_reason;
    }

    emitter.phase_end(run_id, Phase::BGE, bge_diag.success ? "ok" : "skipped",
                      phase_extra, log_file);
    // A rejected BGE candidate is a guarded no-op, not a resume failure. The
    // normal pipeline continues from the unchanged linear RGB in this case.
    // Hard failures (invalid dimensions or masks) return false above.
    return true;
  };

  if (phase_l == "astrometry") {
    if (!run_astrometry_if_needed(true) || (forward && cfg.astrometry.enabled && !have_wcs)) {
      core::emit_event((forward ? "downstream_end" : "resume_end"), run_id,
                       {{"success", false},
                        {"status", "astrometry_failed"},
                        {"error", astrometry_resume_error.empty()
                            ? "astrometry phase failed during resume"
                            : astrometry_resume_error}},
                       log_file);
      return 1;
    }
    if (abort_if_runtime_limit_exceeded("ASTROMETRY")) {
      return 1;
    }
    phase_l = "bge";
  }
  if (phase_l == "bge") {
    (void)run_astrometry_if_needed();
    if (abort_if_runtime_limit_exceeded("ASTROMETRY")) {
      return 1;
    }
    if (!run_bge_phase()) {
      core::emit_event((forward ? "downstream_end" : "resume_end"), run_id,
                       {{"success", false},
                        {"status", "bge_failed"},
                        {"error", bge_resume_error.empty()
                            ? "BGE phase failed during resume"
                            : bge_resume_error}},
                       log_file);
      return 1;
    }
    if (abort_if_runtime_limit_exceeded("BGE")) {
      return 1;
    }
    phase_l = "pcc";
  } else if (phase_l != "pcc") {
    std::cerr << "Error: unsupported resume phase: " << phase_upper
              << std::endl;
    core::emit_event((forward ? "downstream_end" : "resume_end"), run_id,
                     {{"success", false},
                      {"status", "unsupported_phase"},
                      {"from_phase", phase_upper},
                      {"error", "unsupported resume phase: " + phase_upper}},
                     log_file);
    return 1;
  }

  if (phase_l == "pcc") {
    if (!run_astrometry_if_needed()) {
      core::emit_event((forward ? "downstream_end" : "resume_end"), run_id,
                       {{"success", false},
                        {"status", "astrometry_failed"},
                        {"error", astrometry_resume_error.empty()
                            ? "astrometry phase failed during resume"
                            : astrometry_resume_error}},
                       log_file);
      return 1;
    }
    if (abort_if_runtime_limit_exceeded("ASTROMETRY")) {
      return 1;
    }

    if (fs::exists(stacked_rgb_bge_linear_path)) {
      try {
        rgb = io::read_fits_rgb(stacked_rgb_bge_linear_path);
        std::cout << "[PCC][resume] Using precomputed linear BGE snapshot: "
                  << stacked_rgb_bge_linear_path << std::endl;
      } catch (const std::exception &e) {
        std::cout << "[PCC][resume] Warning: failed to load stacked_rgb_bge_linear.fits: "
                  << e.what() << std::endl;
      }
    }

    const fs::path pcc_input_rgb_path =
        fs::exists(stacked_rgb_bge_linear_path) ? stacked_rgb_bge_linear_path
                                                : rgb_path;

    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::PCC, "PCC", log_file);

    io::FitsHeader out_hdr = rgb.header;
    if (have_wcs) {
      inject_wcs_keywords(out_hdr, wcs);
    }

    if (!cfg.pcc.enabled) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "disabled"},
                         {"input_rgb", pcc_input_rgb_path.string()}},
                        log_file);
      core::emit_event((forward ? "downstream_end" : "resume_end"), run_id,
                       {{"success", true}, {"status", "ok"}}, log_file);
      return 0;
    }

    if (!have_wcs) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "no_wcs"},
                         {"input_rgb", pcc_input_rgb_path.string()}},
                        log_file);
      core::emit_event((forward ? "downstream_end" : "resume_end"), run_id,
                       {{"success", false},
                        {"status", "no_wcs"},
                        {"error", "no valid WCS solution available for PCC"}},
                       log_file);
      return 1;
    }

    double search_r = wcs.search_radius_deg();
    std::string source = cfg.pcc.source;
    tile_compile::runner::PCCCatalogQueryResult catalog =
        tile_compile::runner::query_pcc_catalog_stars(
            wcs, cfg.pcc, std::cout, "[PCC][resume]");
    std::string used_source = catalog.used_source;
    std::vector<astro::GaiaStar> stars = std::move(catalog.stars);

    if (stars.empty()) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "no_catalog_stars"},
                         {"search_radius_deg", search_r},
                         {"source", source},
                         {"input_rgb", pcc_input_rgb_path.string()}},
                        log_file);
      core::emit_event(
          (forward ? "downstream_end" : "resume_end"), run_id,
          {{"success", false},
           {"status", "no_catalog_stars"},
           {"error", "PCC catalog query returned no stars"},
           {"search_radius_deg", search_r},
           {"source", used_source}},
          log_file);
      return 1;
    }

    astro::PCCConfig pcc_cfg =
        tile_compile::runner::to_astrometry_pcc_config(cfg.pcc);
    {
      std::string mask_error;
      int rows = static_cast<int>(rgb.R.rows());
      int cols = static_cast<int>(rgb.R.cols());
      if (rows <= 0 || cols <= 0 || rgb.G.rows() != rows ||
          rgb.B.rows() != rows || rgb.G.cols() != cols ||
          rgb.B.cols() != cols) {
        mask_error = "invalid RGB dimensions";
        emitter.phase_end(run_id, Phase::PCC, "error",
                          {{"reason", "output_canvas_mask_invalid"},
                           {"error", mask_error}},
                          log_file);
        core::emit_event(
            (forward ? "downstream_end" : "resume_end"), run_id,
            {{"success", false},
             {"status", "output_canvas_mask_invalid"},
             {"error", mask_error}},
            log_file);
        return 1;
      }
      if (!tile_compile::runner::load_canvas_mask_for_rgb(
              run_dir / "outputs" / "canvas_mask.fits", rgb.R, rgb.G, rgb.B,
              pcc_cfg.output_valid_mask, rows, cols, mask_error)) {
        emitter.phase_end(run_id, Phase::PCC, "error",
                          {{"reason", "output_canvas_mask_invalid"},
                           {"error", mask_error}},
                          log_file);
        core::emit_event(
            (forward ? "downstream_end" : "resume_end"), run_id,
            {{"success", false},
             {"status", "output_canvas_mask_invalid"},
             {"error", mask_error}},
            log_file);
        return 1;
      }
      std::vector<uint8_t> analysis_mask;
      std::string analysis_mask_error;
      int analysis_rows = 0;
      int analysis_cols = 0;
      fs::path analysis_mask_path =
          run_dir / "outputs" / "common_overlap_mask.fits";
      if (!fs::exists(analysis_mask_path)) {
        analysis_mask_path = run_dir / "outputs" / "canvas_mask.fits";
      }
      if (!tile_compile::runner::load_canvas_mask_for_rgb(
              analysis_mask_path, rgb.R, rgb.G, rgb.B, analysis_mask,
              analysis_rows, analysis_cols, analysis_mask_error)) {
        emitter.phase_end(run_id, Phase::PCC, "error",
                          {{"reason", "analysis_mask_invalid"},
                           {"error", analysis_mask_error}},
                          log_file);
        core::emit_event(
            (forward ? "downstream_end" : "resume_end"), run_id,
            {{"success", false},
             {"status", "analysis_mask_invalid"},
             {"error", analysis_mask_error}},
            log_file);
        return 1;
      }
      pcc_cfg.common_valid_mask = std::move(analysis_mask);
      pcc_cfg.common_mask_rows = analysis_rows;
      pcc_cfg.common_mask_cols = analysis_cols;
      pcc_cfg.output_mask_rows = rows;
      pcc_cfg.output_mask_cols = cols;
      std::cout << "[PCC][resume] Using COMMON_OVERLAP analysis mask and full output canvas mask ("
                << cols << "x" << rows << ")" << std::endl;
    }

    if (pcc_cfg.radii_mode == "auto_fwhm") {
      load_seeing_fwhm_if_needed();
      std::string pcc_auto_fwhm_source;
      const double F = tile_compile::runner::resolve_pcc_auto_fwhm_px(
          rgb.R, rgb.G, rgb.B, have_seeing_fwhm, seeing_fwhm_median,
          &pcc_auto_fwhm_source);
      const double r_ap = std::max(static_cast<double>(pcc_cfg.min_aperture_px),
                                   pcc_cfg.aperture_fwhm_mult * F);
      const double r_in = std::max(r_ap + 1.0,
                                   pcc_cfg.annulus_inner_fwhm_mult * F);
      const double r_out = std::max(r_in + 2.0,
                                    pcc_cfg.annulus_outer_fwhm_mult * F);
      pcc_cfg.aperture_radius_px = r_ap;
      pcc_cfg.annulus_inner_px = r_in;
      pcc_cfg.annulus_outer_px = r_out;
      std::cout << "[PCC][resume] auto_fwhm radii source: "
                << pcc_auto_fwhm_source
                << " (F=" << F << ")" << std::endl;
    }

    auto result = astro::run_pcc(rgb.R, rgb.G, rgb.B, wcs, stars, pcc_cfg);

    if (!result.success) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "fit_failed"},
                         {"error", result.error_message},
                         {"stars_matched", result.n_stars_matched},
                         {"stars_used", result.n_stars_used},
                         {"residual_rms", result.residual_rms},
                         {"determinant", result.determinant},
                         {"condition_number", result.condition_number},
                         {"apply_mode", result.apply_mode},
                         {"apply_attenuation", pcc_cfg.apply_attenuation},
                         {"chroma_strength", pcc_cfg.chroma_strength},
                         {"k_max", pcc_cfg.k_max},
                         {"radii_mode", pcc_cfg.radii_mode},
                         {"aperture_radius_px", pcc_cfg.aperture_radius_px},
                         {"annulus_inner_px", pcc_cfg.annulus_inner_px},
                         {"annulus_outer_px", pcc_cfg.annulus_outer_px},
                         {"source", used_source},
                         {"input_rgb", pcc_input_rgb_path.string()}},
                        log_file);
      core::emit_event((forward ? "downstream_end" : "resume_end"), run_id,
                       {{"success", false},
                        {"status", "fit_failed"},
                        {"error", result.error_message},
                        {"residual_rms", result.residual_rms},
                        {"stars_matched", result.n_stars_matched},
                        {"stars_used", result.n_stars_used}},
                       log_file);
      return 1;
    }

    const auto chroma_speckle_stats =
        image::suppress_isolated_chroma_speckles_rgb_inplace(
            rgb.R, rgb.G, rgb.B, &pcc_cfg.common_valid_mask,
            pcc_cfg.common_mask_rows, pcc_cfg.common_mask_cols);
    if (chroma_speckle_stats.corrected_pixels > 0) {
      std::cout << "[PCC][resume] Post-PCC chroma speckle suppressor corrected "
                << chroma_speckle_stats.corrected_pixels
                << " isolated pixels (candidates="
                << chroma_speckle_stats.candidate_pixels << ")" << std::endl;
    }
    if (io::detect_color_mode(rgb.header, 2) == ColorMode::OSC &&
        cfg.chroma_denoise.enabled &&
        cfg.chroma_denoise.apply_stage == "post_pcc") {
      reconstruction::chroma_denoise_rgb_inplace(
          rgb.R, rgb.G, rgb.B, cfg.chroma_denoise);
    }

    const fs::path pcc_r_path = run_dir / "outputs" / "pcc_R.fit";
    const fs::path pcc_g_path = run_dir / "outputs" / "pcc_G.fit";
    const fs::path pcc_b_path = run_dir / "outputs" / "pcc_B.fit";
    const fs::path pcc_rgb_path = run_dir / "outputs" / "stacked_rgb_pcc.fits";
    std::error_code ec_r;
    std::error_code ec_g;
    std::error_code ec_b;
    std::error_code ec_rgb;
    fs::remove(pcc_r_path, ec_r);
    fs::remove(pcc_g_path, ec_g);
    fs::remove(pcc_b_path, ec_b);
    fs::remove(pcc_rgb_path, ec_rgb);
    io::write_fits_float(pcc_r_path, rgb.R, out_hdr);
    io::write_fits_float(pcc_g_path, rgb.G, out_hdr);
    io::write_fits_float(pcc_b_path, rgb.B, out_hdr);
    // stacked_rgb_pcc.fits must remain LINEAR float32 — it is the HMS input.
    // Never apply output_stretch here; HMS needs the original linear data.
    write_atomic_rgb(pcc_rgb_path, rgb.R, rgb.G, rgb.B, out_hdr);

    core::json matrix_json = core::json::array();
    for (int r = 0; r < 3; ++r) {
      matrix_json.push_back(
          {result.matrix[r][0], result.matrix[r][1], result.matrix[r][2]});
    }

    emitter.phase_end(run_id, Phase::PCC, "ok",
                      {{"stars_matched", result.n_stars_matched},
                       {"stars_used", result.n_stars_used},
                       {"residual_rms", result.residual_rms},
                       {"determinant", result.determinant},
                       {"condition_number", result.condition_number},
                       {"apply_mode", result.apply_mode},
                       {"apply_attenuation", pcc_cfg.apply_attenuation},
                       {"chroma_strength", pcc_cfg.chroma_strength},
                       {"k_max", pcc_cfg.k_max},
                       {"radii_mode", pcc_cfg.radii_mode},
                       {"aperture_radius_px", pcc_cfg.aperture_radius_px},
                       {"annulus_inner_px", pcc_cfg.annulus_inner_px},
                       {"annulus_outer_px", pcc_cfg.annulus_outer_px},
                       {"isolated_chroma_speckles_corrected",
                        chroma_speckle_stats.corrected_pixels},
                       {"isolated_chroma_speckle_candidates",
                        chroma_speckle_stats.candidate_pixels},
                       {"matrix", matrix_json},
                       {"source", used_source},
                       {"input_rgb", pcc_input_rgb_path.string()}},
                      log_file);

    if (abort_if_runtime_limit_exceeded("PCC")) {
      return 1;
    }

    if (cfg.hypermetric_stretch.enabled) {
      emitter.phase_start(run_id, Phase::HYPERMETRIC_STRETCH,
                          "HYPERMETRIC_STRETCH", log_file);
      image::HyperMetricStretchConfig hms_cfg =
          to_image_hms_config(cfg.hypermetric_stretch);
      auto hms_diag = image::run_hypermetric_stretch_rgb(
          rgb.R, rgb.G, rgb.B, hms_cfg, &pcc_cfg.common_valid_mask,
          pcc_cfg.common_mask_rows, pcc_cfg.common_mask_cols,
          &pcc_cfg.output_valid_mask);
      if (!hms_diag.success) {
        emitter.phase_end(run_id, Phase::HYPERMETRIC_STRETCH, "error",
                          {{"reason", "stretch_failed"},
                           {"error", hms_diag.error_message}},
                          log_file);
        core::emit_event((forward ? "downstream_end" : "resume_end"), run_id,
                         {{"success", false},
                          {"status", "stretch_failed"},
                          {"error", hms_diag.error_message}},
                         log_file);
        return 1;
      }

      io::FitsHeader hms_hdr = out_hdr;
      hms_hdr.set("HMS", true);
      hms_hdr.set("HMSVER", std::string("1"));
      hms_hdr.set("HMSMODE", hms_cfg.mode);
      hms_hdr.set("HMSPROF", hms_diag.profile);
      hms_hdr.set("HMSWR", static_cast<double>(hms_diag.weights_r));
      hms_hdr.set("HMSWG", static_cast<double>(hms_diag.weights_g));
      hms_hdr.set("HMSWB", static_cast<double>(hms_diag.weights_b));
      hms_hdr.set("HMSANCH", static_cast<double>(hms_diag.anchor));
      hms_hdr.set("HMSLOGD", static_cast<double>(hms_diag.log_d));
      hms_hdr.set("HMSB", static_cast<double>(hms_diag.protect_b));
      hms_hdr.set("HMSTGBG", static_cast<double>(hms_diag.target_bg));
      hms_hdr.set("HMSCONV", static_cast<double>(hms_diag.convergence_power));
      hms_hdr.set("HMSSTAR", static_cast<double>(hms_diag.star_pressure));

      fs::path hms_rgb_path(hms_cfg.output_rgb);
      if (hms_rgb_path.is_relative()) {
        hms_rgb_path = run_dir / "outputs" / hms_rgb_path;
      }
      std::error_code hms_ec;
      if (!forward) fs::remove(hms_rgb_path, hms_ec);
      write_atomic_rgb(hms_rgb_path, rgb.R, rgb.G, rgb.B, hms_hdr);
      if (hms_cfg.write_channels) {
        io::write_fits_float(run_dir / "outputs" / "hms_R.fit", rgb.R,
                             hms_hdr);
        io::write_fits_float(run_dir / "outputs" / "hms_G.fit", rgb.G,
                             hms_hdr);
        io::write_fits_float(run_dir / "outputs" / "hms_B.fit", rgb.B,
                             hms_hdr);
      }

      emitter.phase_end(
          run_id, Phase::HYPERMETRIC_STRETCH, "ok",
          {{"input_stage", "pcc"},
           {"output_rgb", hms_rgb_path.string()},
           {"profile", hms_diag.profile},
           {"profile_source", hms_diag.profile_source},
           {"anchor", hms_diag.anchor},
           {"log_d", hms_diag.log_d},
           {"target_bg", hms_diag.target_bg},
           {"star_pressure", hms_diag.star_pressure},
           {"color_strategy", hms_diag.color_strategy},
           {"color_grip", hms_diag.color_grip},
           {"shadow_convergence", hms_diag.shadow_convergence},
           {"black_clip_percent", hms_diag.black_clip_percent},
           {"white_clip_percent", hms_diag.white_clip_percent}},
          log_file);
      if (abort_if_runtime_limit_exceeded("HYPERMETRIC_STRETCH")) {
        return 1;
      }
    }
  }

  core::emit_event((forward ? "downstream_end" : "resume_end"), run_id, {{"success", true}, {"status", "ok"}},
                   log_file);
  return 0;
}
} // namespace tile_compile::runner
