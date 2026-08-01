#include "runner_resume.hpp"

#include "tile_compile/astrometry/gaia_catalog.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/events.hpp"
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

#include "runner_shared.hpp"
#include "runner_phase_post_stack_output.hpp"
#include "runner_phase_aqmh_reconstruction.hpp"
#include "runner_phase_aqmh_diagnostics.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using tile_compile::Tile;
using tile_compile::TileGrid;
using tile_compile::TileMetrics;
using tile_compile::TileType;
using tile_compile::runner::WarpBounds;
using tile_compile::runner::compute_warps_bounds;
namespace runner = tile_compile::runner;

/// @brief Implements shell quote.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string shell_quote(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back(static_cast<char>(39));
  for (char c : s) {
    if (c == static_cast<char>(39)) {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out.push_back(static_cast<char>(39));
  return out;
}

/// @brief Normalizes phase name.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string normalize_phase_name(std::string phase) {
  std::transform(phase.begin(), phase.end(), phase.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return phase;
}

/// @brief Checks inplace rerun phase.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool is_inplace_rerun_phase(const std::string &phase_upper) {
  static const std::vector<std::string> kPhases = {
      "SCAN_INPUT",        "CHANNEL_SPLIT",   "NORMALIZATION",
      "GLOBAL_METRICS",    "TILE_GRID",       "REGISTRATION",
      "PREWARP",           "COMMON_OVERLAP",  "LOCAL_METRICS",
      "TILE_RECONSTRUCTION", "STATE_CLUSTERING", "SYNTHETIC_FRAMES",
      "DEBAYER"};
  return std::find(kPhases.begin(), kPhases.end(), phase_upper) !=
         kPhases.end();
}

// AQMH phases that can reuse cached quality maps and skip directly to
// reconstruction without a full pipeline rerun (§ Resume spec).
// STACKING is included so legacy runs without the immutable raw
// reconstruction can regenerate it before entering the shared stacking path.
bool is_aqmh_cache_resume_phase(const std::string &phase_upper) {
  static const std::vector<std::string> kPhases = {
      "AQMH_MAPS", "AQMH_GLOBAL_QUALITY", "AQMH_RECONSTRUCTION",
      "AQMH_DIAGNOSTICS", "STACKING"};
  return std::find(kPhases.begin(), kPhases.end(), phase_upper) !=
         kPhases.end();
}

bool validate_df_cache_metadata(const fs::path &run_dir, size_t frame_count,
                                std::string &error_out) {
  const fs::path metadata_path =
      run_dir / "artifacts" / "pre_debayer_metadata.json";
  try {
    const auto metadata = tile_compile::core::json::parse(
        tile_compile::core::read_text(metadata_path));
    if (metadata.value("format_version", 0) != 1 ||
        !metadata.value("complete", false) ||
        metadata.value("frame_count", 0u) != frame_count ||
        metadata.value("rows", 0) <= 0 || metadata.value("cols", 0) <= 0 ||
        metadata.value("color_mode", std::string()) != "OSC" ||
        metadata.value("channel_order", tile_compile::core::json::array()) !=
            tile_compile::core::json::array({"R", "G", "B"}) ||
        metadata.value("luma_weights", tile_compile::core::json::array()) !=
            tile_compile::core::json::array({0.25, 0.5, 0.25})) {
      error_out = "DF metadata is incomplete or incompatible";
      return false;
    }
  } catch (const std::exception &e) {
    error_out = "cannot read DF metadata " + metadata_path.string() + ": " +
                e.what();
    return false;
  }
  return true;
}

struct ResumeOutputScaling {
  float scale_mono = 1.0f;
  float scale_r = 1.0f;
  float scale_g = 1.0f;
  float scale_b = 1.0f;
  float bg_mono = 0.0f;
  float bg_r = 0.0f;
  float bg_g = 0.0f;
  float bg_b = 0.0f;
};

bool load_resume_output_scaling(const fs::path &run_dir,
                                ResumeOutputScaling &out,
                                std::string &error_out) {
  const fs::path path = run_dir / "artifacts" / "normalization.json";
  tile_compile::core::json artifact;
  try {
    artifact = tile_compile::core::json::parse(
        tile_compile::core::read_text(path));
  } catch (const std::exception &e) {
    error_out = "cannot read normalization artifact " + path.string() + ": " +
                e.what();
    return false;
  }
  const auto values = [&](const char *key, bool positive,
                          std::vector<float> &result) {
    const auto it = artifact.find(key);
    if (it == artifact.end() || !it->is_array()) return false;
    for (const auto &value : *it) {
      if (!value.is_number()) continue;
      const float v = value.get<float>();
      if (std::isfinite(v) && (!positive || v > 0.0f)) result.push_back(v);
    }
    return !result.empty();
  };
  std::vector<float> p_mono, p_r, p_g, p_b, b_mono, b_r, b_g, b_b;
  if (!values("P_mono", true, p_mono) || !values("P_r", true, p_r) ||
      !values("P_g", true, p_g) || !values("P_b", true, p_b) ||
      !values("B_mono", false, b_mono) || !values("B_r", false, b_r) ||
      !values("B_g", false, b_g) || !values("B_b", false, b_b)) {
    error_out = "normalization artifact is missing usable scale or background values";
    return false;
  }
  out.scale_mono = tile_compile::core::median_finite_positive(p_mono, 1.0f);
  out.scale_r = tile_compile::core::median_finite_positive(p_r, 1.0f);
  out.scale_g = tile_compile::core::median_finite_positive(p_g, 1.0f);
  out.scale_b = tile_compile::core::median_finite_positive(p_b, 1.0f);
  out.bg_mono = tile_compile::core::median_finite(b_mono, 0.0f);
  out.bg_r = tile_compile::core::median_finite(b_r, 0.0f);
  out.bg_g = tile_compile::core::median_finite(b_g, 0.0f);
  out.bg_b = tile_compile::core::median_finite(b_b, 0.0f);
  return true;
}

/// @brief Writes canvas mask fits.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool write_canvas_mask_fits(const fs::path &mask_path,
                            const std::vector<uint8_t> &mask, int rows,
                            int cols,
                            const tile_compile::io::FitsHeader &header,
                            std::string &error_out) {
  if (rows <= 0 || cols <= 0) {
    error_out = "invalid canvas mask dimensions";
    return false;
  }
  if (mask.size() != static_cast<size_t>(rows * cols)) {
    error_out = "canvas mask size mismatch while writing";
    return false;
  }
  tile_compile::Matrix2Df mask_img(rows, cols);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      mask_img(y, x) =
          (mask[static_cast<size_t>(y * cols + x)] != 0) ? 1.0f : 0.0f;
    }
  }
  try {
    fs::create_directories(mask_path.parent_path());
    tile_compile::io::write_fits_float(mask_path, mask_img, header);
    return true;
  } catch (const std::exception &e) {
    error_out = std::string("cannot write canvas mask: ") + e.what();
    return false;
  }
}

/// @brief Loads registration canvas offsets.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool load_registration_canvas_offsets(const fs::path &run_dir, int frame_width,
                                      int frame_height,
                                      tile_compile::ColorMode detected_mode,
                                      int &offset_x_out, int &offset_y_out,
                                      std::string &error_out) {
  const fs::path artifact_path = run_dir / "artifacts" / "global_registration.json";
  if (!fs::exists(artifact_path)) {
    error_out = "missing global_registration.json";
    return false;
  }

  std::vector<tile_compile::WarpMatrix> warps;
  try {
    const auto j =
        tile_compile::core::json::parse(tile_compile::core::read_text(artifact_path));
    if (!j.contains("warps") || !j["warps"].is_array()) {
      error_out = "global_registration.json has no warps array";
      return false;
    }
    for (const auto &jw : j["warps"]) {
      if (!jw.is_object()) {
        continue;
      }
      if (jw.value("source", std::string{}) == "unresolved") {
        continue;
      }
      tile_compile::WarpMatrix w = tile_compile::WarpMatrix::Identity();
      w(0, 0) = jw.value("a00", 1.0f);
      w(0, 1) = jw.value("a01", 0.0f);
      w(1, 0) = jw.value("a10", 0.0f);
      w(1, 1) = jw.value("a11", 1.0f);
      w(0, 2) = jw.value("tx", 0.0f);
      w(1, 2) = jw.value("ty", 0.0f);
      warps.push_back(w);
    }
  } catch (const std::exception &e) {
    error_out = std::string("failed to parse global_registration.json: ") + e.what();
    return false;
  }

  const WarpBounds bbox = compute_warps_bounds(frame_width, frame_height, warps);
  int offset_x = -bbox.min_x;
  int offset_y = -bbox.min_y;
  if (detected_mode == tile_compile::ColorMode::OSC) {
    if ((offset_x & 1) != 0) {
      offset_x = (offset_x + 1) & ~1;
    }
    if ((offset_y & 1) != 0) {
      offset_y = (offset_y + 1) & ~1;
    }
  }
  offset_x_out = offset_x;
  offset_y_out = offset_y;
  return true;
}

/// @brief Reads latest run start input dir.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<std::string> read_latest_run_start_input_dir(
    const fs::path &run_events_path) {
  if (!fs::exists(run_events_path)) {
    return std::nullopt;
  }
  std::ifstream in(run_events_path);
  if (!in) {
    return std::nullopt;
  }

  std::optional<std::string> input_dir;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    try {
      const auto ev = tile_compile::core::json::parse(line);
      if (ev.value("type", std::string()) != "run_start") {
        continue;
      }
      const std::string value = ev.value("input_dir", std::string());
      if (!value.empty()) {
        input_dir = value;
      }
    } catch (const std::exception &) {
    }
  }
  return input_dir;
}

std::optional<runner::CropBox> read_latest_stacking_crop(
    const fs::path &run_events_path) {
  if (!fs::exists(run_events_path)) return std::nullopt;
  std::ifstream in(run_events_path);
  if (!in) return std::nullopt;

  std::optional<runner::CropBox> crop;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    try {
      const auto ev = tile_compile::core::json::parse(line);
      if (ev.value("type", std::string()) != "phase_end" ||
          ev.value("phase_name", std::string()) != "STACKING" ||
          ev.value("status", std::string()) != "ok" ||
          !ev.value("crop_applied", false)) {
        continue;
      }
      runner::CropBox candidate{
          ev.value("crop_x", 0), ev.value("crop_y", 0),
          ev.value("crop_width", 0), ev.value("crop_height", 0)};
      if (candidate.valid()) crop = candidate;
    } catch (const std::exception &) {
    }
  }
  return crop;
}

/// @brief Implements current executable path.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
fs::path current_executable_path() {
  std::error_code ec;
  const fs::path proc_self("/proc/self/exe");
  if (fs::exists(proc_self, ec)) {
    const fs::path resolved = fs::read_symlink(proc_self, ec);
    if (!ec && !resolved.empty()) {
      return resolved;
    }
  }
  return fs::current_path() / "tile_compile_runner";
}

/// @brief Creates run revision id.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string make_run_revision_id() {
  std::string ts = tile_compile::core::get_iso_timestamp();
  std::string compact;
  compact.reserve(ts.size());
  for (char c : ts) {
    if ((c >= '0' && c <= '9') || c == 'T' || c == 'Z') {
      compact.push_back(c);
    }
  }
  if (compact.empty()) {
    compact = "unknown";
  }
  return "run_cfg_" + compact;
}

/// @brief Implements add run config revision.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string add_run_config_revision(const fs::path &run_dir,
                                    const std::string &yaml_text,
                                    const std::string &source,
                                    const std::string &run_id) {
  if (yaml_text.empty()) {
    return "";
  }

  namespace core = tile_compile::core;

  std::error_code ec;
  const fs::path revisions_dir = run_dir / "artifacts" / "config_revisions";
  fs::create_directories(revisions_dir, ec);

  const std::string revision_id = make_run_revision_id();
  const fs::path yaml_path = revisions_dir / (revision_id + ".yaml");
  {
    std::ofstream yaml_out(yaml_path, std::ios::out | std::ios::trunc);
    if (!yaml_out) {
      return "";
    }
    yaml_out << yaml_text;
    yaml_out.close();
    if (!yaml_out) {
      return "";
    }
  }

  const fs::path index_path = revisions_dir / "index.json";
  core::json index = core::json::array();
  if (fs::exists(index_path)) {
    try {
      index = core::json::parse(core::read_text(index_path));
      if (!index.is_array()) {
        index = core::json::array();
      }
    } catch (const std::exception &) {
      index = core::json::array();
    }
  }

  index.push_back({
      {"revision_id", revision_id},
      {"file_name", yaml_path.filename().string()},
      {"source", source},
      {"created_at", core::get_iso_timestamp()},
      {"run_id", run_id},
  });
  core::write_text(index_path, index.dump(2));
  return revision_id;
}

/// @brief Implements rerun existing run in place.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int rerun_existing_run_in_place(const fs::path &run_dir,
                                const std::string &run_id,
                                const std::string &from_phase) {
  namespace core = tile_compile::core;

  const auto input_dir = read_latest_run_start_input_dir(
      run_dir / "logs" / "run_events.jsonl");
  if (!input_dir.has_value() || input_dir->empty()) {
    std::cerr << "Error: cannot determine input_dir from existing run log"
              << std::endl;
    return 1;
  }

  const fs::path cfg_path = run_dir / "config.yaml";
  std::ifstream cfg_in(cfg_path);
  if (!cfg_in) {
    std::cerr << "Error: config.yaml not found in run_dir: " << cfg_path
              << std::endl;
    return 1;
  }
  const std::string cfg_text((std::istreambuf_iterator<char>(cfg_in)),
                             std::istreambuf_iterator<char>());
  if (cfg_text.empty()) {
    std::cerr << "Error: config.yaml is empty: " << cfg_path << std::endl;
    return 1;
  }

  const std::string revision_id =
      add_run_config_revision(run_dir, cfg_text, "resume_selected_config", run_id);
  fs::path rerun_config_path = cfg_path;
  if (!revision_id.empty()) {
    rerun_config_path =
        run_dir / "artifacts" / "config_revisions" / (revision_id + ".yaml");
  }

  const fs::path exe_path = current_executable_path();
  std::ostringstream cmd;
  cmd << shell_quote(exe_path.string()) << " run"
      << " --config " << shell_quote(rerun_config_path.string())
      << " --input-dir " << shell_quote(*input_dir)
      << " --runs-dir " << shell_quote(run_dir.parent_path().string())
      << " --project-root " << shell_quote(fs::current_path().string())
      << " --run-id " << shell_quote(run_id);

  std::cout << "[RESUME][rerun] Replaying full pipeline in place for requested "
            << "phase " << from_phase << ": " << cmd.str() << std::endl;

  const int ret = std::system(cmd.str().c_str());

  std::ofstream event_log_file(run_dir / "logs" / "run_events.jsonl",
                               std::ios::out | std::ios::app);
  if (event_log_file) {
    tile_compile::runner::TeeBuf tee_buf(std::cout.rdbuf(),
                                         event_log_file.rdbuf());
    std::ostream log_file(&tee_buf);
    core::emit_event(
        "resume_end", run_id,
        {{"success", ret == 0},
         {"status", ret == 0 ? "ok" : "rerun_failed"},
         {"from_phase", from_phase},
         {"mode", "inplace_full_rerun"}},
        log_file);
  }

  return (ret == 0) ? 0 : 1;
}

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

}  // namespace

/// @brief Implements resume command.
/// @details Part of the resume command path that reconstructs downstream artifacts from an existing run directory; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int resume_command(const std::string &run_dir_path, const std::string &from_phase) {
  using namespace tile_compile;

  namespace core = tile_compile::core;
  namespace io = tile_compile::io;
  namespace astro = tile_compile::astrometry;
  namespace image = tile_compile::image;

  fs::path run_dir(run_dir_path);
  if (!fs::exists(run_dir) || !fs::is_directory(run_dir)) {
    std::cerr << "Error: run_dir not found: " << run_dir_path << std::endl;
    return 1;
  }

  fs::path cfg_path = run_dir / "config.yaml";
  if (!fs::exists(cfg_path)) {
    std::cerr << "Error: config.yaml not found in run_dir: " << cfg_path
              << std::endl;
    return 1;
  }

  config::Config cfg;
  try {
    cfg = config::Config::load(cfg_path);
    cfg.validate();
  } catch (const std::exception &e) {
    std::cerr << "Error: failed to load/validate config.yaml: " << e.what()
              << std::endl;
    return 1;
  }

  std::string run_id = run_dir.filename().string();
  std::string phase_upper = normalize_phase_name(from_phase);
  if (phase_upper.empty()) {
    phase_upper = "PCC";
  }
  std::string phase_l = core::to_lower(phase_upper);

  if (is_inplace_rerun_phase(phase_upper)) {
    return rerun_existing_run_in_place(run_dir, run_id, phase_upper);
  }

  const fs::path aqmh_raw_reconstruction =
      run_dir / "outputs" / "aqmh_reconstructed_raw.fit";
  if (cfg.aqmh.enabled && phase_upper == "STACKING" &&
      fs::is_regular_file(aqmh_raw_reconstruction)) {
    phase_l = "stacking";
  } else
  // All AQMH phases that have cached quality maps can resume directly at
  // AQMH_RECONSTRUCTION, reusing the existing prewarp cache + map cache.
  // This avoids a full pipeline rerun for AQMH_MAPS/GLOBAL_QUALITY/DIAGNOSTICS.
  // For STACKING it also regenerates the immutable raw reconstruction when a
  // legacy run does not contain that resume artifact.
  if (cfg.aqmh.enabled && is_aqmh_cache_resume_phase(phase_upper)) {
    phase_l = "aqmh_reconstruction";
    phase_upper = "AQMH_RECONSTRUCTION";
  }

  fs::create_directories(run_dir / "logs");

  std::ofstream event_log_file(run_dir / "logs" / "run_events.jsonl",
                               std::ios::out | std::ios::app);
  tile_compile::runner::TeeBuf tee_buf(std::cout.rdbuf(), event_log_file.rdbuf());
  std::ostream log_file(&tee_buf);

  core::emit_event("resume_start", run_id,
                   {{"run_dir", run_dir.string()}, {"from_phase", phase_upper}},
                   log_file);
  const auto resume_started_at = std::chrono::steady_clock::now();
  auto abort_if_runtime_limit_exceeded =
      [&](const std::string &checkpoint) -> bool {
    const double elapsed_hours =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      resume_started_at)
            .count() /
        3600.0;
    if (elapsed_hours <= cfg.runtime_limits.hard_abort_hours) {
      return false;
    }
    core::emit_event("runtime_limit_exceeded", run_id,
                     {{"checkpoint", checkpoint},
                      {"elapsed_hours", elapsed_hours},
                      {"hard_abort_hours",
                       cfg.runtime_limits.hard_abort_hours}},
                     log_file);
    core::emit_event("resume_end", run_id,
                     {{"success", false},
                      {"status", "runtime_limit_exceeded"},
                      {"checkpoint", checkpoint}},
                     log_file);
    std::cerr << "Error: runtime limit exceeded during resume at "
              << checkpoint << " (" << elapsed_hours << " h > "
              << cfg.runtime_limits.hard_abort_hours << " h)" << std::endl;
    return true;
  };

  if (phase_l == "hypermetric_stretch" || phase_l == "hms") {
    namespace image = tile_compile::image;
    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::HYPERMETRIC_STRETCH,
                        "HYPERMETRIC_STRETCH", log_file);

    const fs::path outputs_dir = run_dir / "outputs";
    Matrix2Df R;
    Matrix2Df G;
    Matrix2Df B;
    io::FitsHeader hdr;
    std::string input_stage;
    std::string input_rgb;
    try {
      const fs::path pcc_r = outputs_dir / "pcc_R.fit";
      const fs::path pcc_g = outputs_dir / "pcc_G.fit";
      const fs::path pcc_b = outputs_dir / "pcc_B.fit";
      if (fs::exists(pcc_r) && fs::exists(pcc_g) && fs::exists(pcc_b)) {
        std::tie(R, hdr) = io::read_fits_float(pcc_r);
        G = io::read_fits_pixels_float(pcc_g);
        B = io::read_fits_pixels_float(pcc_b);
        input_stage = "pcc_channels";
        input_rgb = pcc_r.string() + ";" + pcc_g.string() + ";" + pcc_b.string();
      } else if (fs::exists(outputs_dir / "stacked_rgb_pcc.fits")) {
        auto rgb = io::read_fits_rgb(outputs_dir / "stacked_rgb_pcc.fits");
        R = std::move(rgb.R);
        G = std::move(rgb.G);
        B = std::move(rgb.B);
        hdr = rgb.header;
        input_stage = "pcc_rgb";
        input_rgb = (outputs_dir / "stacked_rgb_pcc.fits").string();
      } else if (!cfg.hypermetric_stretch.require_successful_pcc &&
                 fs::exists(outputs_dir / "stacked_rgb_bge_linear.fits")) {
        auto rgb = io::read_fits_rgb(outputs_dir / "stacked_rgb_bge_linear.fits");
        R = std::move(rgb.R);
        G = std::move(rgb.G);
        B = std::move(rgb.B);
        hdr = rgb.header;
        input_stage = "bge_linear_no_pcc";
        input_rgb = (outputs_dir / "stacked_rgb_bge_linear.fits").string();
      } else if (!cfg.hypermetric_stretch.require_successful_pcc &&
                 fs::exists(outputs_dir / "stacked_rgb_solve.fits")) {
        auto rgb = io::read_fits_rgb(outputs_dir / "stacked_rgb_solve.fits");
        R = std::move(rgb.R);
        G = std::move(rgb.G);
        B = std::move(rgb.B);
        hdr = rgb.header;
        input_stage = "linear_no_pcc";
        input_rgb = (outputs_dir / "stacked_rgb_solve.fits").string();
      }
    } catch (const std::exception &e) {
      emitter.phase_end(run_id, Phase::HYPERMETRIC_STRETCH, "error",
                        {{"reason", "read_input_failed"},
                         {"error", e.what()}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "read_input_failed"}},
                       log_file);
      return 1;
    }

    if (R.size() == 0 || G.size() == 0 || B.size() == 0) {
      emitter.phase_end(run_id, Phase::HYPERMETRIC_STRETCH, "skipped",
                        {{"reason", "missing_pcc_artifacts"},
                         {"require_successful_pcc",
                          cfg.hypermetric_stretch.require_successful_pcc}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "missing_pcc_artifacts"}},
                       log_file);
      return 1;
    }

    std::vector<uint8_t> statistics_mask;
    std::vector<uint8_t> output_mask;
    std::vector<uint8_t> *statistics_mask_ptr = nullptr;
    std::vector<uint8_t> *output_mask_ptr = nullptr;
    int mask_rows = static_cast<int>(R.rows());
    int mask_cols = static_cast<int>(R.cols());
    std::string statistics_mask_error;
    if (tile_compile::runner::load_canvas_mask_for_rgb(
            outputs_dir / "common_overlap_mask.fits", R, G, B,
            statistics_mask, mask_rows, mask_cols, statistics_mask_error)) {
      statistics_mask_ptr = &statistics_mask;
    } else {
      std::cout << "[HMS][resume] Warning: common-overlap mask unavailable: "
                << statistics_mask_error << "; using full image statistics"
                << std::endl;
      mask_rows = static_cast<int>(R.rows());
      mask_cols = static_cast<int>(R.cols());
    }

    std::string output_mask_error;
    int output_mask_rows = mask_rows;
    int output_mask_cols = mask_cols;
    if (tile_compile::runner::load_canvas_mask_for_rgb(
            outputs_dir / "canvas_mask.fits", R, G, B, output_mask,
            output_mask_rows, output_mask_cols, output_mask_error)) {
      output_mask_ptr = &output_mask;
    } else {
      std::cout << "[HMS][resume] Warning: output canvas mask unavailable: "
                << output_mask_error << "; using full image output" << std::endl;
    }

    image::HyperMetricStretchConfig hms_cfg =
        to_image_hms_config(cfg.hypermetric_stretch);
    hms_cfg.enabled = true;
    auto hms_diag = image::run_hypermetric_stretch_rgb(
        R, G, B, hms_cfg, statistics_mask_ptr, mask_rows, mask_cols,
        output_mask_ptr);
    if (!hms_diag.success) {
      emitter.phase_end(run_id, Phase::HYPERMETRIC_STRETCH, "error",
                        {{"reason", "stretch_failed"},
                         {"error", hms_diag.error_message}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "stretch_failed"}},
                       log_file);
      return 1;
    }

    hdr.set("HMS", true);
    hdr.set("HMSVER", std::string("1"));
    hdr.set("HMSMODE", hms_cfg.mode);
    hdr.set("HMSPROF", hms_diag.profile);
    hdr.set("HMSWR", static_cast<double>(hms_diag.weights_r));
    hdr.set("HMSWG", static_cast<double>(hms_diag.weights_g));
    hdr.set("HMSWB", static_cast<double>(hms_diag.weights_b));
    hdr.set("HMSANCH", static_cast<double>(hms_diag.anchor));
    hdr.set("HMSLOGD", static_cast<double>(hms_diag.log_d));
    hdr.set("HMSB", static_cast<double>(hms_diag.protect_b));
    hdr.set("HMSTGBG", static_cast<double>(hms_diag.target_bg));
    hdr.set("HMSCONV", static_cast<double>(hms_diag.convergence_power));
    hdr.set("HMSSTAR", static_cast<double>(hms_diag.star_pressure));

    fs::path hms_rgb_path(hms_cfg.output_rgb);
    if (hms_rgb_path.is_relative()) {
      hms_rgb_path = outputs_dir / hms_rgb_path;
    }
    {
      std::error_code ec;
      fs::remove(hms_rgb_path, ec);
    }
    try {
      io::write_fits_rgb(hms_rgb_path, R, G, B, hdr);
      if (hms_cfg.write_channels) {
        io::write_fits_float(outputs_dir / "hms_R.fit", R, hdr);
        io::write_fits_float(outputs_dir / "hms_G.fit", G, hdr);
        io::write_fits_float(outputs_dir / "hms_B.fit", B, hdr);
      }
    } catch (const std::exception &e) {
      emitter.phase_end(run_id, Phase::HYPERMETRIC_STRETCH, "error",
                        {{"reason", "write_output_failed"},
                         {"error", e.what()}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "write_output_failed"}},
                       log_file);
      return 1;
    }

    emitter.phase_end(
        run_id, Phase::HYPERMETRIC_STRETCH, "ok",
        {{"input_rgb", input_rgb},
         {"input_stage", input_stage},
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
    core::emit_event("resume_end", run_id,
                     {{"success", true}, {"status", "ok"}}, log_file);
    return 0;
  }

  if (phase_l == "aqmh_reconstruction") {
    core::EventEmitter emitter;
    const fs::path metrics_path = run_dir / "artifacts" / "aqmh_metrics.json";
    const fs::path cache_meta_path = run_dir / "cache" / "aqmh" /
                                     "aqmh_cache.json";
    if (!fs::exists(metrics_path) || !fs::exists(cache_meta_path)) {
      std::cerr << "Error: AQMH resume requires aqmh_metrics.json and cache metadata"
                << std::endl;
      return 1;
    }
    core::json metrics_json;
    core::json cache_meta;
    try {
      metrics_json = core::json::parse(core::read_text(metrics_path));
      cache_meta = core::json::parse(core::read_text(cache_meta_path));
    } catch (const std::exception &e) {
      std::cerr << "Error: invalid AQMH resume metadata: " << e.what()
                << std::endl;
      return 1;
    }
    const int canvas_width = cache_meta.value("full_width", 0);
    const int canvas_height = cache_meta.value("full_height", 0);
    const auto diagnostics = metrics_json.value("diagnostics", core::json::array());
    const size_t frame_count = diagnostics.is_array() ? diagnostics.size() : 0u;
    if (canvas_width <= 0 || canvas_height <= 0 || frame_count == 0) {
      std::cerr << "Error: incomplete AQMH cache dimensions/frame count"
                << std::endl;
      return 1;
    }

    Matrix2Df canvas_mask_image;
    io::FitsHeader resume_header;
    try {
      std::tie(canvas_mask_image, resume_header) =
          io::read_fits_float(run_dir / "outputs" / "canvas_mask.fits");
    } catch (const std::exception &e) {
      std::cerr << "Error: cannot read AQMH canvas mask: " << e.what()
                << std::endl;
      return 1;
    }
    std::vector<uint8_t> reconstruction_valid_mask(
        static_cast<size_t>(canvas_width) * canvas_height, 0u);
    if (canvas_mask_image.rows() == canvas_height &&
        canvas_mask_image.cols() == canvas_width) {
      for (size_t i = 0; i < reconstruction_valid_mask.size(); ++i)
        reconstruction_valid_mask[i] =
            canvas_mask_image.data()[i] > 0.0f ? 1u : 0u;
    } else {
      tile_compile::metrics::FrameValidMaskStore mask_store(
          run_dir / "cache" / "aqmh_masks", canvas_width, canvas_height);
      bool rebuilt_mask = false;
      for (size_t fi = 0; fi < frame_count; ++fi) {
        std::vector<uint8_t> frame_mask = mask_store.read(fi);
        if (frame_mask.size() != reconstruction_valid_mask.size()) continue;
        rebuilt_mask = true;
        for (size_t i = 0; i < reconstruction_valid_mask.size(); ++i) {
          reconstruction_valid_mask[i] =
              (reconstruction_valid_mask[i] != 0u || frame_mask[i] != 0u)
                  ? 1u
                  : 0u;
        }
      }
      if (!rebuilt_mask) {
        std::cerr << "Error: AQMH canvas-mask dimensions differ from cache and "
                     "per-frame AQMH masks are unavailable"
                  << std::endl;
        return 1;
      }
    }

    Matrix2Df common_overlap_mask_image;
    try {
      std::tie(common_overlap_mask_image, std::ignore) = io::read_fits_float(
          run_dir / "outputs" / "common_overlap_mask.fits");
    } catch (const std::exception &e) {
      std::cerr << "Error: AQMH resume requires common_overlap_mask.fits: "
                << e.what() << std::endl;
      return 1;
    }
    std::vector<uint8_t> common_valid_mask(
        static_cast<size_t>(canvas_width) * canvas_height, 0u);
    if (common_overlap_mask_image.rows() == canvas_height &&
        common_overlap_mask_image.cols() == canvas_width) {
      for (size_t i = 0; i < common_valid_mask.size(); ++i)
        common_valid_mask[i] =
            common_overlap_mask_image.data()[i] > 0.0f ? 1u : 0u;
    } else {
      // STACKING may have persisted a cropped overlap mask while AQMH caches
      // retain the full reconstruction canvas. Reuse the validated full
      // reconstruction mask rather than rejecting an otherwise valid resume.
      common_valid_mask = reconstruction_valid_mask;
      std::cerr << "Warning: cropped common-overlap mask differs from AQMH "
                   "cache; using the full reconstruction mask for resume"
                << std::endl;
    }

    runner::DiskCacheFrameStore prewarped_frames(
        run_dir / "cache" / "prewarped_frames", frame_count, canvas_height,
        canvas_width, true);
    const bool df_requested =
        cfg.aqmh.enabled && cfg.aqmh.reconstruction.debayer_first &&
        io::detect_color_mode(resume_header, 2) == ColorMode::OSC;
    // Debayer-First-AQMH: attach and validate the complete RGB prewarp cache.
    std::unique_ptr<runner::DiskCacheFrameStoreRGB> prewarped_frames_rgb;
    const fs::path rgb_cache_dir =
        run_dir / "cache" / "prewarped_frames_rgb";
    if (df_requested && fs::is_directory(rgb_cache_dir)) {
      prewarped_frames_rgb = std::make_unique<runner::DiskCacheFrameStoreRGB>(
          rgb_cache_dir, frame_count, canvas_height, canvas_width, true);
    }
    if (df_requested) {
      std::string df_error;
      if (!prewarped_frames_rgb ||
          prewarped_frames_rgb->size() != frame_count ||
          prewarped_frames_rgb->rows() != canvas_height ||
          prewarped_frames_rgb->cols() != canvas_width ||
          !validate_df_cache_metadata(run_dir, frame_count, df_error)) {
        core::emit_event("resume_end", run_id,
                         {{"success", false},
                          {"status", "df_cache_invalid"},
                          {"reason", df_error.empty()
                                         ? "RGB prewarp cache missing"
                                         : df_error}},
                         log_file);
        std::cerr << "Error: invalid Debayer-First RGB cache: " << df_error
                  << std::endl;
        return 1;
      }
    }
    std::vector<uint8_t> frame_has_data(frame_count, 0u);
    size_t available_frames = 0;
    for (size_t fi = 0; fi < frame_count; ++fi) {
      const bool has_data = df_requested
                                ? prewarped_frames_rgb->has_data(fi)
                                : prewarped_frames.has_data(fi);
      frame_has_data[fi] = has_data ? 1u : 0u;
      available_frames += has_data;
    }
    if (available_frames == 0) {
      core::emit_event("resume_end", run_id,
                       {{"success", false},
                        {"status", "prewarped_cache_missing"},
                        {"reason", "no_reusable_prewarped_cache_frames"},
                        {"cache_dir", (df_requested ? rgb_cache_dir
                                                      : run_dir / "cache" / "prewarped_frames").string()}},
                       log_file);
      std::cerr << "Error: no reusable prewarped frames found" << std::endl;
      return 1;
    }

    VectorXf global_weights(static_cast<Eigen::Index>(frame_count));
    global_weights.setZero();
    for (size_t fi = 0; fi < frame_count; ++fi) {
      if (fi < diagnostics.size())
        global_weights[static_cast<Eigen::Index>(fi)] =
            diagnostics[fi].value("global_quality", 0.0f);
    }
    const std::string mask_hash =
        tile_compile::metrics::compute_aqmh_canvas_mask_hash(
            reconstruction_valid_mask, canvas_width, canvas_height);
    auto aqmh_cache = std::make_unique<tile_compile::metrics::QualityMapCache>(
        run_dir / "cache" / "aqmh",
        cache_meta.value("map_stream_id", std::string("luma")),
        canvas_width, canvas_height, cfg.aqmh.pyramid, cfg.aqmh.storage,
        mask_hash, cache_meta.value("execution_backend", std::string("cpu")));
    std::vector<fs::path> frame_slots(frame_count);
    core::AccelerationContext acceleration(
        cfg.runtime_limits.acceleration_backend);
    runner::AqmhReconstructionPhaseResult phase_result;
    emitter.phase_start(run_id, Phase::AQMH_RECONSTRUCTION,
                        "AQMH_RECONSTRUCTION", log_file);
    const auto phase_started = std::chrono::steady_clock::now();
    if (!runner::run_phase_aqmh_reconstruction(
            run_id, cfg, run_dir, frame_slots, frame_has_data,
            reconstruction_valid_mask, common_valid_mask, canvas_width,
            canvas_height,
            io::detect_color_mode(resume_header, 2) == ColorMode::OSC,
            prewarped_frames, aqmh_cache,
            global_weights, acceleration, emitter, log_file, phase_started,
            cv::getNumThreads(), phase_result, nullptr,
            prewarped_frames_rgb.get())) {
      return 1;
    }
    try {
      io::write_fits_float(aqmh_raw_reconstruction, phase_result.raw_output,
                           resume_header);
      io::write_fits_float(run_dir / "outputs" / "reconstructed_L.fit",
                           phase_result.output, resume_header);
      // Debayer-First-AQMH: persist per-channel RGB reconstructions.
      if (phase_result.debayer_first_used) {
        io::FitsHeader resume_rgb_header = resume_header;
        resume_rgb_header.set("DEBAYER", "PRE_STACK");
        if (phase_result.df_output_R.size() > 0)
          io::write_fits_float(run_dir / "outputs" / "reconstructed_R.fit",
                               phase_result.df_output_R, resume_rgb_header);
        if (phase_result.df_output_G.size() > 0)
          io::write_fits_float(run_dir / "outputs" / "reconstructed_G.fit",
                               phase_result.df_output_G, resume_rgb_header);
        if (phase_result.df_output_B.size() > 0)
          io::write_fits_float(run_dir / "outputs" / "reconstructed_B.fit",
                               phase_result.df_output_B, resume_rgb_header);
      }
    } catch (const std::exception &e) {
      std::cerr << "Error: cannot persist AQMH resume output: " << e.what()
                << std::endl;
      return 1;
    }
    if (!runner::run_phase_aqmh_diagnostics(
            run_id, cfg, run_dir, phase_result.recon, aqmh_cache.get(),
            common_valid_mask, frame_has_data, canvas_width, canvas_height,
            emitter, log_file))
      return 1;
    phase_l = "stacking";
  }

  if (phase_l == "stacking") {
    namespace image = tile_compile::image;
    namespace reconstruction = tile_compile::reconstruction;

    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::STACKING, "STACKING", log_file);
    emitter.phase_progress(run_id, Phase::STACKING, 0.0f,
                           "preparing cached reconstruction", log_file);

    std::vector<std::pair<int, fs::path>> synthetic_entries;
    const fs::path outputs_dir = run_dir / "outputs";
    if (cfg.aqmh.enabled && fs::is_regular_file(aqmh_raw_reconstruction)) {
      synthetic_entries.emplace_back(0, aqmh_raw_reconstruction);
    } else if (!cfg.aqmh.enabled && fs::exists(outputs_dir) &&
               fs::is_directory(outputs_dir)) {
      for (const auto &entry : fs::directory_iterator(outputs_dir)) {
        if (!entry.is_regular_file()) {
          continue;
        }
        const fs::path path = entry.path();
        if (path.extension() != ".fit" && path.extension() != ".fits") {
          continue;
        }
        const std::string stem = path.stem().string();
        const std::string prefix = "synthetic_";
        if (stem.rfind(prefix, 0) != 0) {
          continue;
        }
        try {
          const int index = std::stoi(stem.substr(prefix.size()));
          synthetic_entries.emplace_back(index, path);
        } catch (const std::exception &) {
        }
      }
    }
    std::sort(synthetic_entries.begin(), synthetic_entries.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    if (synthetic_entries.empty()) {
      const std::string reason = cfg.aqmh.enabled
          ? "missing_aqmh_raw_reconstruction"
          : "missing_synthetic_outputs";
      const std::string msg = cfg.aqmh.enabled
          ? "AQMH STACKING resume requires outputs/aqmh_reconstructed_raw.fit; "
            "resume from AQMH_RECONSTRUCTION to regenerate the immutable CFA "
            "source"
          : "missing synthetic_*.fit outputs for STACKING resume";
      emitter.phase_end(run_id, Phase::STACKING, "error",
                        {{"reason", reason},
                         {"outputs_dir", outputs_dir.string()},
                         {"error", msg}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", reason}},
                       log_file);
      std::cerr << "Error: " << msg << std::endl;
      return 1;
    }

    io::FitsHeader first_hdr;
    Matrix2Df first_synth;
    try {
      std::tie(first_synth, first_hdr) = io::read_fits_float(synthetic_entries.front().second);
    } catch (const std::exception &e) {
      emitter.phase_end(run_id, Phase::STACKING, "error",
                        {{"reason", "read_synthetic_failed"},
                         {"file", synthetic_entries.front().second.string()},
                         {"error", e.what()}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "read_synthetic_failed"}},
                       log_file);
      return 1;
    }
    if (first_synth.size() <= 0) {
      emitter.phase_end(run_id, Phase::STACKING, "error",
                        {{"reason", "empty_synthetic"},
                         {"file", synthetic_entries.front().second.string()}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "empty_synthetic"}},
                       log_file);
      return 1;
    }

    const ColorMode detected_mode = io::detect_color_mode(first_hdr, 2);
    const BayerPattern detected_bayer = io::detect_bayer_pattern(first_hdr);
    const std::string detected_bayer_str = bayer_pattern_to_string(detected_bayer);
    ResumeOutputScaling aqmh_output_scaling;
    const bool restore_aqmh_output_scaling =
        cfg.aqmh.enabled && detected_mode == ColorMode::OSC;
    if (restore_aqmh_output_scaling) {
      std::string scaling_error;
      if (!load_resume_output_scaling(run_dir, aqmh_output_scaling,
                                      scaling_error)) {
        emitter.phase_end(run_id, Phase::STACKING, "error",
                          {{"reason", "normalization_artifact_invalid"},
                           {"error", scaling_error}},
                          log_file);
        core::emit_event("resume_end", run_id,
                         {{"success", false},
                          {"status", "normalization_artifact_invalid"}},
                         log_file);
        std::cerr << "Error: " << scaling_error << std::endl;
        return 1;
      }
    }
    std::vector<float> synthetic_cluster_quality;
    std::vector<float> synthetic_cluster_mass;
    const fs::path synthetic_artifact_path =
        run_dir / "artifacts" / "synthetic_frames.json";
    if (fs::exists(synthetic_artifact_path)) {
      try {
        const auto j = core::json::parse(core::read_text(synthetic_artifact_path));
        if (j.contains("cluster_quality") && j["cluster_quality"].is_array()) {
          for (const auto &jq : j["cluster_quality"]) {
            synthetic_cluster_quality.push_back(jq.get<float>());
          }
        }
        if (j.contains("cluster_mass") && j["cluster_mass"].is_array()) {
          for (const auto &jm : j["cluster_mass"]) {
            synthetic_cluster_mass.push_back(jm.get<float>());
          }
        }
      } catch (const std::exception &e) {
        std::cout << "[STACKING][resume] Warning: failed to parse "
                  << synthetic_artifact_path << ": " << e.what() << std::endl;
      }
    }

    const auto resume_crop = cfg.aqmh.enabled
        ? read_latest_stacking_crop(run_dir / "logs" / "run_events.jsonl")
        : std::optional<runner::CropBox>{};
    if (resume_crop.has_value()) {
      const auto &crop = *resume_crop;
      if (crop.x < 0 || crop.y < 0 || crop.x + crop.width > first_synth.cols() ||
          crop.y + crop.height > first_synth.rows()) {
        emitter.phase_end(run_id, Phase::STACKING, "error",
                          {{"reason", "invalid_stored_crop"},
                           {"crop_x", crop.x}, {"crop_y", crop.y},
                           {"crop_width", crop.width},
                           {"crop_height", crop.height}},
                          log_file);
        core::emit_event("resume_end", run_id,
                         {{"success", false}, {"status", "invalid_stored_crop"}},
                         log_file);
        return 1;
      }
      first_synth = first_synth.block(crop.y, crop.x, crop.height, crop.width).eval();
    }

    const fs::path common_overlap_mask_path =
        run_dir / "outputs" / "common_overlap_mask.fits";
    const fs::path output_mask_path = run_dir / "outputs" / "canvas_mask.fits";
    std::vector<uint8_t> common_valid_mask;
    std::string canvas_mask_error;
    if (!tile_compile::runner::load_canvas_mask_fits(
            output_mask_path, first_synth.rows(), first_synth.cols(),
            common_valid_mask, canvas_mask_error)) {
      emitter.phase_end(run_id, Phase::STACKING, "error",
                        {{"reason", "canvas_mask_invalid"},
                         {"canvas_mask", output_mask_path.string()},
                         {"error", canvas_mask_error}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "canvas_mask_invalid"}},
                       log_file);
      return 1;
    }
    std::vector<uint8_t> analysis_valid_mask = common_valid_mask;
    const fs::path analysis_mask_path = common_overlap_mask_path;
    if (fs::exists(analysis_mask_path)) {
      std::string analysis_mask_error;
      std::vector<uint8_t> loaded_analysis_mask;
      if (!tile_compile::runner::load_canvas_mask_fits(
              analysis_mask_path, first_synth.rows(), first_synth.cols(),
              loaded_analysis_mask, analysis_mask_error)) {
        emitter.phase_end(run_id, Phase::STACKING, "error",
                          {{"reason", "analysis_mask_invalid"},
                           {"analysis_mask", analysis_mask_path.string()},
                           {"error", analysis_mask_error}},
                          log_file);
        core::emit_event(
            "resume_end", run_id,
            {{"success", false}, {"status", "analysis_mask_invalid"}},
            log_file);
        return 1;
      }
      analysis_valid_mask = std::move(loaded_analysis_mask);
    }

    int debayer_tile_offset_x = 0;
    int debayer_tile_offset_y = 0;
    if (detected_mode == ColorMode::OSC) {
      const auto input_dir = read_latest_run_start_input_dir(
          run_dir / "logs" / "run_events.jsonl");
      if (input_dir.has_value() && !input_dir->empty()) {
        auto input_frames = core::discover_frames(fs::path(*input_dir), "*");
        input_frames.erase(
            std::remove_if(input_frames.begin(), input_frames.end(),
                           [](const fs::path &p) {
                             return !io::is_fits_image_path(p);
                           }),
            input_frames.end());
        if (!input_frames.empty()) {
          int frame_width = 0;
          int frame_height = 0;
          int naxis = 0;
          try {
            std::tie(frame_width, frame_height, naxis) =
                io::get_fits_dimensions(input_frames.front());
            std::string offset_error;
            if (!load_registration_canvas_offsets(
                    run_dir, frame_width, frame_height, detected_mode,
                    debayer_tile_offset_x, debayer_tile_offset_y, offset_error)) {
              std::cout << "[STACKING][resume] Warning: " << offset_error
                        << std::endl;
            }
          } catch (const std::exception &e) {
            std::cout << "[STACKING][resume] Warning: failed to determine input "
                         "frame dimensions: "
                      << e.what() << std::endl;
          }
        }
      }
    }
    if (resume_crop.has_value()) {
      debayer_tile_offset_x -= resume_crop->x;
      debayer_tile_offset_y -= resume_crop->y;
    }

    core::AccelerationContext acceleration(
        cfg.runtime_limits.acceleration_backend);
    const auto stacking_acceleration =
        acceleration.selection_for(core::AccelerationPhase::stacking);
    const core::AccelerationOps stacking_ops(
        acceleration, core::AccelerationPhase::stacking);
    core::WorkerCudaStreams stacking_streams(
        stacking_acceleration.selected ==
            core::AccelerationBackend::opencv_cuda,
        detected_mode == ColorMode::OSC ? 3u : 1u);
    {
      std::ostringstream msg;
      msg << "STACKING acceleration "
          << core::acceleration_selection_summary(stacking_acceleration)
          << " cpu_workers=" << (detected_mode == ColorMode::OSC ? 3 : 1)
          << " gpu=" << (stacking_acceleration.using_gpu ? "yes" : "no")
          << " backend="
          << core::acceleration_backend_name(stacking_acceleration.selected);
      if (!stacking_acceleration.request_honored &&
          !stacking_acceleration.fallback_reason.empty()) {
        emitter.warning(run_id, msg.str(), log_file);
      }
      std::cout << "[STACKING][resume] " << msg.str() << std::endl;
    }

    std::vector<Matrix2Df> valid_synth;
    valid_synth.reserve(synthetic_entries.size());
    std::vector<float> valid_synth_q;
    valid_synth_q.reserve(synthetic_entries.size());
    std::vector<float> valid_synth_mass;
    valid_synth_mass.reserve(synthetic_entries.size());
    std::vector<Matrix2Df> synth_R;
    std::vector<Matrix2Df> synth_G;
    std::vector<Matrix2Df> synth_B;
    if (detected_mode == ColorMode::OSC) {
      synth_R.reserve(synthetic_entries.size());
      synth_G.reserve(synthetic_entries.size());
      synth_B.reserve(synthetic_entries.size());
    }

    for (const auto &[index, path] : synthetic_entries) {
      Matrix2Df syn;
      try {
        syn = io::read_fits_pixels_float(path);
      } catch (const std::exception &e) {
        emitter.phase_end(run_id, Phase::STACKING, "error",
                          {{"reason", "read_synthetic_failed"},
                           {"file", path.string()},
                           {"error", e.what()}},
                          log_file);
        core::emit_event("resume_end", run_id,
                         {{"success", false}, {"status", "read_synthetic_failed"}},
                         log_file);
        return 1;
      }
      if (syn.size() <= 0) {
        continue;
      }
      if (resume_crop.has_value()) {
        const auto &crop = *resume_crop;
        if (crop.x + crop.width > syn.cols() ||
            crop.y + crop.height > syn.rows()) {
          emitter.phase_end(run_id, Phase::STACKING, "error",
                            {{"reason", "synthetic_crop_out_of_bounds"},
                             {"file", path.string()}},
                            log_file);
          core::emit_event("resume_end", run_id,
                           {{"success", false},
                            {"status", "synthetic_crop_out_of_bounds"}},
                           log_file);
          return 1;
        }
        syn = syn.block(crop.y, crop.x, crop.height, crop.width).eval();
      }
      if (detected_mode == ColorMode::OSC) {
        auto deb = image::debayer_nearest_neighbor(
            syn, detected_bayer, -debayer_tile_offset_x, -debayer_tile_offset_y);
        synth_R.push_back(std::move(deb.R));
        synth_G.push_back(std::move(deb.G));
        synth_B.push_back(std::move(deb.B));
      }
      valid_synth.push_back(std::move(syn));
      if (index >= 0 &&
          static_cast<size_t>(index) < synthetic_cluster_quality.size()) {
        valid_synth_q.push_back(synthetic_cluster_quality[static_cast<size_t>(index)]);
      } else {
        valid_synth_q.push_back(0.0f);
      }
      if (index >= 0 &&
          static_cast<size_t>(index) < synthetic_cluster_mass.size()) {
        valid_synth_mass.push_back(
            synthetic_cluster_mass[static_cast<size_t>(index)]);
      } else {
        valid_synth_mass.push_back(1.0f);
      }
    }

    emitter.phase_progress(
        run_id, Phase::STACKING, 0.2f,
        "loaded reconstruction inputs " + std::to_string(valid_synth.size()) +
            "/" + std::to_string(synthetic_entries.size()),
        log_file);
    if (valid_synth.empty()) {
      emitter.phase_end(run_id, Phase::STACKING, "error",
                        {{"reason", "no_valid_synthetic_frames"}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "no_valid_synthetic"}},
                       log_file);
      return 1;
    }

    constexpr float kEpsWeight = 1.0e-6f;
    std::vector<float> cluster_stack_weights;
    if (cfg.stacking.cluster_quality_weighting.enabled) {
      cluster_stack_weights.resize(valid_synth_q.size(), 1.0f);
      const float kappa = cfg.stacking.cluster_quality_weighting.kappa_cluster;
      std::vector<float> q_values = valid_synth_q;
      const float q_ref = q_values.empty() ? 0.0f : core::median_of(q_values);
      for (size_t i = 0; i < valid_synth_q.size(); ++i) {
        const float mass =
            (i < valid_synth_mass.size() && std::isfinite(valid_synth_mass[i]) &&
             valid_synth_mass[i] > kEpsWeight)
                ? valid_synth_mass[i]
                : 1.0f;
        const float q_rel = std::clamp(valid_synth_q[i] - q_ref, -3.0f, 3.0f);
        cluster_stack_weights[i] = mass * std::exp(kappa * q_rel);
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
          if (w > cap) {
            w = cap;
          }
        }
      }
    }

    Matrix2Df recon;
    Matrix2Df recon_R;
    Matrix2Df recon_G;
    Matrix2Df recon_B;
    const bool use_quality_weighting =
        cfg.stacking.cluster_quality_weighting.enabled;
    if (detected_mode == ColorMode::OSC && synth_R.size() == valid_synth.size()) {
      if (!use_quality_weighting && cfg.stacking.method == "rej") {
        auto stack_channel = [&](const std::vector<Matrix2Df> &channel,
                                 size_t stream_index) {
          return stacking_ops.sigma_clip_stack(
              channel, cfg.stacking.sigma_clip.sigma_low,
              cfg.stacking.sigma_clip.sigma_high,
              cfg.stacking.sigma_clip.max_iters,
              cfg.stacking.sigma_clip.min_fraction,
              stacking_streams.get(stream_index));
        };
        auto future_r = std::async(std::launch::async, stack_channel,
                                   std::cref(synth_R), 0u);
        auto future_g = std::async(std::launch::async, stack_channel,
                                   std::cref(synth_G), 1u);
        recon_B = stack_channel(synth_B, 2u);
        recon_R = future_r.get();
        recon_G = future_g.get();
      } else {
        std::vector<float> stack_weights(synth_R.size(), 1.0f);
        if (use_quality_weighting &&
            cluster_stack_weights.size() == synth_R.size()) {
          stack_weights = cluster_stack_weights;
        }
        auto reduce_channel = [&](const std::vector<Matrix2Df> &channel,
                                  size_t stream_index) {
          return stacking_ops.sigma_clip_reduce(
              channel, stack_weights, cfg.stacking.sigma_clip.sigma_low,
              cfg.stacking.sigma_clip.sigma_high,
              cfg.stacking.sigma_clip.max_iters,
              cfg.stacking.sigma_clip.min_fraction, kEpsWeight,
              stacking_streams.get(stream_index));
        };
        auto future_r = std::async(std::launch::async, reduce_channel,
                                   std::cref(synth_R), 0u);
        auto future_g = std::async(std::launch::async, reduce_channel,
                                   std::cref(synth_G), 1u);
        auto wr_b = reduce_channel(synth_B, 2u);
        auto wr_r = future_r.get();
        auto wr_g = future_g.get();
        recon_R = std::move(wr_r.tile);
        recon_G = std::move(wr_g.tile);
        recon_B = std::move(wr_b.tile);
      }
      recon = 0.25f * recon_R + 0.5f * recon_G + 0.25f * recon_B;
    } else {
      if (!use_quality_weighting && cfg.stacking.method == "rej") {
        recon = stacking_ops.sigma_clip_stack(
            valid_synth, cfg.stacking.sigma_clip.sigma_low,
            cfg.stacking.sigma_clip.sigma_high, cfg.stacking.sigma_clip.max_iters,
            cfg.stacking.sigma_clip.min_fraction, stacking_streams.get(0));
      } else {
        std::vector<float> stack_weights(valid_synth.size(), 1.0f);
        if (use_quality_weighting &&
            cluster_stack_weights.size() == valid_synth.size()) {
          stack_weights = cluster_stack_weights;
        }
        auto wr = stacking_ops.sigma_clip_reduce(
            valid_synth, stack_weights, cfg.stacking.sigma_clip.sigma_low,
            cfg.stacking.sigma_clip.sigma_high,
            cfg.stacking.sigma_clip.max_iters,
            cfg.stacking.sigma_clip.min_fraction, kEpsWeight,
            stacking_streams.get(0));
        recon = std::move(wr.tile);
      }
    }

    emitter.phase_progress(run_id, Phase::STACKING, 0.55f,
                           "stacked cached reconstruction", log_file);
    if (cfg.stacking.cosmetic_correction) {
      const float cosmetic_sigma = cfg.stacking.cosmetic_correction_sigma;
      recon = image::cosmetic_correction(recon, cosmetic_sigma, true);
      if (detected_mode == ColorMode::OSC && recon_R.size() == recon.size()) {
        recon_R = image::cosmetic_correction(recon_R, cosmetic_sigma, true);
        recon_G = image::cosmetic_correction(recon_G, cosmetic_sigma, true);
        recon_B = image::cosmetic_correction(recon_B, cosmetic_sigma, true);
      }
    }

    if (detected_mode == ColorMode::OSC && cfg.chroma_denoise.enabled &&
        cfg.chroma_denoise.apply_stage == "post_stack_linear" &&
        recon_R.size() == recon.size()) {
      reconstruction::chroma_denoise_rgb_inplace(
          recon_R, recon_G, recon_B, cfg.chroma_denoise);
      recon = 0.25f * recon_R + 0.5f * recon_G + 0.25f * recon_B;
    }

    emitter.phase_progress(run_id, Phase::STACKING, 0.75f,
                           "applied post-stack processing", log_file);
    auto stretch_luma_for_output = [&](Matrix2Df &luma) {
      if (!cfg.stacking.output_stretch) {
        return;
      }
      const auto stretch =
          tile_compile::core::stretch_to_u16_linear_from_zero_inplace(luma);
      if (!stretch.applied) {
        return;
      }
      std::cout << "[STACKING][resume] Output linear stretch ["
                << stretch.low << ".." << stretch.high << "] -> [0..65535]"
                << " samples=" << stretch.sample_count << std::endl;
    };

    runner::CropBox stacking_crop_box{
        0, 0, static_cast<int>(recon.cols()), static_cast<int>(recon.rows())};
    bool stacking_crop_applied = false;
    if (cfg.output.crop_to_nonzero_bbox && recon.size() > 0) {
      const int full_rows = recon.rows();
      const int full_cols = recon.cols();
      const bool have_rgb_full =
          (recon_R.rows() == full_rows && recon_R.cols() == full_cols &&
           recon_G.rows() == full_rows && recon_G.cols() == full_cols &&
           recon_B.rows() == full_rows && recon_B.cols() == full_cols);
      const size_t full_mask_px =
          static_cast<size_t>(full_rows) * static_cast<size_t>(full_cols);
      if (common_valid_mask.size() != full_mask_px) {
        const std::string msg =
            "internal canvas mask size mismatch during crop";
        emitter.phase_end(run_id, Phase::STACKING, "error",
                          {{"reason", "canvas_mask_size_mismatch"},
                           {"error", msg},
                           {"mask_pixels",
                            static_cast<uint64_t>(common_valid_mask.size())},
                           {"expected_mask_pixels",
                            static_cast<uint64_t>(full_mask_px)}},
                          log_file);
        core::emit_event("resume_end", run_id,
                         {{"success", false},
                          {"status", "canvas_mask_size_mismatch"}},
                         log_file);
        return 1;
      }
      if (analysis_valid_mask.size() != full_mask_px) {
        emitter.phase_end(run_id, Phase::STACKING, "error",
                          {{"reason", "analysis_mask_size_mismatch"},
                           {"mask_pixels", static_cast<uint64_t>(
                                               analysis_valid_mask.size())},
                           {"expected_mask_pixels",
                            static_cast<uint64_t>(full_mask_px)}},
                          log_file);
        core::emit_event(
            "resume_end", run_id,
            {{"success", false}, {"status", "analysis_mask_size_mismatch"}},
            log_file);
        return 1;
      }

      stacking_crop_box = cfg.aqmh.enabled
          ? tile_compile::runner::compute_support_mask_bbox(
                common_valid_mask, full_rows, full_cols)
          : tile_compile::runner::compute_nonzero_data_bbox(
                recon, have_rgb_full ? &recon_R : nullptr,
                have_rgb_full ? &recon_G : nullptr,
                have_rgb_full ? &recon_B : nullptr);
      if (!stacking_crop_box.valid()) {
        emitter.phase_end(run_id, Phase::STACKING, "error",
                          {{"reason", "empty_valid_crop"},
                           {"error",
                            "crop_to_nonzero_bbox produced empty valid canvas"}},
                          log_file);
        core::emit_event("resume_end", run_id,
                         {{"success", false}, {"status", "empty_valid_crop"}},
                         log_file);
        return 1;
      }

      const int crop_x = stacking_crop_box.x;
      const int crop_y = stacking_crop_box.y;
      const int crop_w = stacking_crop_box.width;
      const int crop_h = stacking_crop_box.height;
      stacking_crop_applied =
          (crop_x != 0 || crop_y != 0 || crop_w != full_cols ||
           crop_h != full_rows);
      if (stacking_crop_applied) {
        recon = recon.block(crop_y, crop_x, crop_h, crop_w).eval();
        if (have_rgb_full) {
          recon_R = recon_R.block(crop_y, crop_x, crop_h, crop_w).eval();
          recon_G = recon_G.block(crop_y, crop_x, crop_h, crop_w).eval();
          recon_B = recon_B.block(crop_y, crop_x, crop_h, crop_w).eval();
        }
        debayer_tile_offset_x -= crop_x;
        debayer_tile_offset_y -= crop_y;

        std::vector<uint8_t> cropped_mask(
            static_cast<size_t>(crop_h * crop_w), static_cast<uint8_t>(0));
        std::vector<uint8_t> cropped_analysis_mask(
            static_cast<size_t>(crop_h * crop_w), static_cast<uint8_t>(0));
        for (int y = 0; y < crop_h; ++y) {
          const int sy = crop_y + y;
          const size_t src_row_off =
              static_cast<size_t>(sy) * static_cast<size_t>(full_cols);
          const size_t dst_row_off =
              static_cast<size_t>(y) * static_cast<size_t>(crop_w);
          for (int x = 0; x < crop_w; ++x) {
            const int sx = crop_x + x;
            cropped_mask[dst_row_off + static_cast<size_t>(x)] =
                common_valid_mask[src_row_off + static_cast<size_t>(sx)];
            cropped_analysis_mask[dst_row_off + static_cast<size_t>(x)] =
                analysis_valid_mask[src_row_off + static_cast<size_t>(sx)];
          }
        }
        common_valid_mask.swap(cropped_mask);
        analysis_valid_mask.swap(cropped_analysis_mask);

        std::string mask_write_error;
        if (!write_canvas_mask_fits(run_dir / "outputs" / "canvas_mask.fits",
                                    common_valid_mask, crop_h, crop_w, first_hdr,
                                    mask_write_error) ||
            !write_canvas_mask_fits(
                run_dir / "outputs" / "common_overlap_mask.fits",
                analysis_valid_mask, crop_h, crop_w, first_hdr,
                mask_write_error)) {
          emitter.phase_end(run_id, Phase::STACKING, "error",
                            {{"reason", "canvas_mask_write_failed"},
                             {"error", mask_write_error}},
                            log_file);
          core::emit_event("resume_end", run_id,
                           {{"success", false},
                            {"status", "canvas_mask_write_failed"}},
                           log_file);
          return 1;
        }
      }
    }

    const bool have_resume_rgb_recon =
        detected_mode == ColorMode::OSC && recon_R.size() == recon.size() &&
        recon_G.size() == recon.size() && recon_B.size() == recon.size() &&
        recon_R.size() > 0;
    const bool defer_aqmh_osc_scaling_to_rgb_writer =
        restore_aqmh_output_scaling && cfg.aqmh.enabled &&
        detected_mode == ColorMode::OSC && !have_resume_rgb_recon;
    if (restore_aqmh_output_scaling && !defer_aqmh_osc_scaling_to_rgb_writer) {
      image::apply_output_scaling_inplace(
          recon, -debayer_tile_offset_x, -debayer_tile_offset_y,
          detected_mode, detected_bayer_str, aqmh_output_scaling.scale_mono,
          aqmh_output_scaling.scale_r, aqmh_output_scaling.scale_g,
          aqmh_output_scaling.scale_b, aqmh_output_scaling.bg_mono,
          aqmh_output_scaling.bg_r, aqmh_output_scaling.bg_g,
          aqmh_output_scaling.bg_b, 0.0f);
      if (recon_R.size() == recon.size() && recon_G.size() == recon.size() &&
          recon_B.size() == recon.size()) {
        recon_R.array() = recon_R.array() * aqmh_output_scaling.scale_r +
                          aqmh_output_scaling.bg_r;
        recon_G.array() = recon_G.array() * aqmh_output_scaling.scale_g +
                          aqmh_output_scaling.bg_g;
        recon_B.array() = recon_B.array() * aqmh_output_scaling.scale_b +
                          aqmh_output_scaling.bg_b;
      }
    }

    emitter.phase_progress(run_id, Phase::STACKING, 0.9f,
                           "writing stacked output", log_file);
    // Use the shared post-stack output writer for consistent behavior between
    // pipeline and resume paths (stretch, crop, scaling, format).
    runner::OutputScaling resume_scaling;
    if (restore_aqmh_output_scaling) {
      resume_scaling.scale_r = aqmh_output_scaling.scale_r;
      resume_scaling.scale_g = aqmh_output_scaling.scale_g;
      resume_scaling.scale_b = aqmh_output_scaling.scale_b;
      resume_scaling.scale_mono = aqmh_output_scaling.scale_mono;
      resume_scaling.bg_r = aqmh_output_scaling.bg_r;
      resume_scaling.bg_g = aqmh_output_scaling.bg_g;
      resume_scaling.bg_b = aqmh_output_scaling.bg_b;
      resume_scaling.bg_mono = aqmh_output_scaling.bg_mono;
      resume_scaling.pedestal = 0.0f;
    }
    runner::PostStackOutputConfig post_cfg;
    post_cfg.output_stretch = cfg.stacking.output_stretch;
    post_cfg.crop_to_nonzero_bbox = false;  // crop already handled above
    post_cfg.aqmh_enabled = cfg.aqmh.enabled;
    post_cfg.cosmetic_correction = false;
    runner::PostStackOutputResult post_result;
    if (!runner::write_post_stack_outputs(
            recon, recon_R, recon_G, recon_B,
            common_valid_mask, analysis_valid_mask,
            resume_scaling, detected_mode, detected_bayer_str,
            debayer_tile_offset_x, debayer_tile_offset_y,
            first_hdr, post_cfg, run_dir, run_id,
            emitter, log_file, post_result)) {
      emitter.phase_end(run_id, Phase::STACKING, "error",
                        {{"reason", "output_write_failed"},
                         {"error", post_result.error}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "output_write_failed"}},
                       log_file);
      return 1;
    }

    const auto stacking_input_batch = core::make_device_frame_batch(
        valid_synth.size(), recon.rows(), recon.cols(),
        detected_mode == ColorMode::OSC ? 3 : 1);
    emitter.phase_end(
        run_id, Phase::STACKING, "ok",
        {{"acceleration",
          core::acceleration_selection_to_json(stacking_acceleration)},
         {"device_frame_batch_input",
          core::device_frame_batch_to_json(stacking_input_batch)},
         {"input_frames", static_cast<int>(valid_synth.size())},
         {"crop_applied", stacking_crop_applied},
         {"crop_source", cfg.aqmh.enabled ? "reconstruction_support_mask"
                                           : "nonzero_data_bbox"},
         {"crop_x", stacking_crop_box.x},
         {"crop_y", stacking_crop_box.y},
         {"crop_width", stacking_crop_box.width},
         {"crop_height", stacking_crop_box.height},
         {"output_luma", (run_dir / "outputs" / "stacked.fits").string()},
         {"debayer_method", post_result.debayer_method}},
        log_file);
    if (abort_if_runtime_limit_exceeded("STACKING")) {
      return 1;
    }

    emitter.phase_start(run_id, Phase::DEBAYER, "DEBAYER", log_file);
    if (detected_mode == ColorMode::OSC) {
      emitter.phase_end(
          run_id, Phase::DEBAYER, "ok",
          {{"mode", "OSC"},
           {"bayer_pattern", detected_bayer_str},
           {"output_rgb", (run_dir / "outputs" / "stacked_rgb.fits").string()},
           {"output_rgb_solve",
            (run_dir / "outputs" / "stacked_rgb_solve.fits").string()}},
          log_file);
      phase_l = "astrometry";
    } else {
      emitter.phase_end(run_id, Phase::DEBAYER, "ok", {{"mode", "MONO"}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", true}, {"status", "ok"}}, log_file);
      return 0;
    }
    if (abort_if_runtime_limit_exceeded("DEBAYER")) {
      return 1;
    }
  }

  fs::path rgb_path = run_dir / "outputs" / "stacked_rgb_solve.fits";
  fs::path stacked_rgb_path = run_dir / "outputs" / "stacked_rgb.fits";
  fs::path stacked_rgb_solve_path = run_dir / "outputs" / "stacked_rgb_solve.fits";
  if (!fs::exists(rgb_path)) {
    rgb_path = stacked_rgb_path;
  }
  if (!fs::exists(rgb_path)) {
    std::cerr << "Error: missing stacked RGB cube in run outputs" << std::endl;
    core::emit_event("resume_end", run_id,
                     {{"success", false}, {"status", "missing_rgb"}},
                     log_file);
    return 1;
  }

  io::RGBImage rgb;
  try {
    rgb = io::read_fits_rgb(rgb_path);
  } catch (const std::exception &e) {
    std::cerr << "Error: failed to read RGB FITS: " << e.what() << std::endl;
    core::emit_event("resume_end", run_id,
                     {{"success", false}, {"status", "read_rgb_failed"}},
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
  if (fs::exists(wcs_path)) {
    try {
      wcs = astro::parse_wcs_file(wcs_path.string());
      have_wcs = wcs.valid();
    } catch (const std::exception &) {
      have_wcs = false;
    }
  }

  auto run_astrometry_if_needed = [&](bool force_rerun = false) -> bool {
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

    std::string cmd = runner::shell_quote(astap_bin_path.string()) + " -f " +
                      runner::shell_quote(rgb_path.string()) + " -d " +
                      runner::shell_quote(astap_data) + " -r " +
                      std::to_string(cfg.astrometry.search_radius);

    std::cout << "[ASTROMETRY][resume] Running: " << cmd << std::endl;
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
      fs::path wcs_artifact = run_dir / "artifacts" / "stacked_rgb.wcs";
      try {
        fs::create_directories(wcs_artifact.parent_path());
        fs::copy_file(wcs_out, wcs_artifact,
                      fs::copy_options::overwrite_existing);
      } catch (const std::exception &) {
      }

      emitter.phase_end(run_id, Phase::ASTROMETRY, "ok",
                        {{"ra", wcs.crval1},
                         {"dec", wcs.crval2},
                         {"pixel_scale_arcsec", wcs.pixel_scale_arcsec()},
                         {"rotation_deg", wcs.rotation_deg()},
                         {"wcs_file",
                          (run_dir / "artifacts" / "stacked_rgb.wcs")
                              .string()}},
                        log_file);
      return true;
    } else {
      // Re-solve failed — try to fall back to existing WCS file
      if (fs::exists(wcs_path)) {
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
        emitter.phase_end(run_id, Phase::ASTROMETRY, "error",
                          {{"reason", "solve_failed"}, {"exit_code", ret}},
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

    const fs::path tile_grid_path = run_dir / "artifacts" / "tile_grid.json";
    if (!fs::exists(tile_grid_path)) return;

    try {
      const auto j = core::json::parse(core::read_text(tile_grid_path));
      if (j.contains("seeing_fwhm_median") &&
          j["seeing_fwhm_median"].is_number()) {
        const double f = j["seeing_fwhm_median"].get<double>();
        if (std::isfinite(f) && f > 0.0) {
          seeing_fwhm_median = f;
          have_seeing_fwhm = true;
        }
      }
    } catch (const std::exception &) {
      have_seeing_fwhm = false;
    }
  };

  auto load_bge_tile_context_if_needed = [&]() {
    if (bge_tile_context_loaded) return;
    bge_tile_context_loaded = true;

    std::string local_err;
    std::string grid_err;
    const bool ok_local = load_aggregated_tile_metrics(
        run_dir / "artifacts" / "local_metrics.json", bge_tile_metrics, local_err);
    const bool ok_grid = load_tile_grid_from_artifact(
        run_dir / "artifacts" / "tile_grid.json", bge_tile_grid, grid_err);

    bge_have_local_metrics = ok_local && !bge_tile_metrics.empty();
    // Set BGE tile metrics source based on reconstruction method
    if (cfg.method == "aqmh") {
        bge_tile_metrics_source = "aqmh_output";
    } else {
        bge_tile_metrics_source = bge_have_local_metrics ? "classic_local_metrics" : "none";
    }
    bge_have_bge_grid = ok_grid && !bge_tile_grid.tiles.empty();
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
    io::write_fits_rgb(path, R_src, G_src, B_src, hdr);
  };

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
        bge_diag, cfg.bge.enabled, bge_have_tile_data, bge_metrics_tiles_match);
    bge_artifact["have_local_metrics"] = bge_have_local_metrics;
    bge_artifact["have_tile_metrics"] = bge_have_tile_metrics;
    bge_artifact["tile_metrics_source"] = bge_tile_metrics_source;
    bge_artifact["have_bge_grid"] = bge_have_bge_grid;
    bge_artifact["local_metrics_tiles"] = static_cast<int>(bge_tile_metrics.size());
    bge_artifact["bge_grid_tiles"] = static_cast<int>(bge_tile_grid.tiles.size());
    bge_artifact["config"] = {
        {"enabled", cfg.bge.enabled},
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
        {"requested", cfg.bge.enabled},
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
    if (!run_astrometry_if_needed(true)) {
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "astrometry_failed"}},
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
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "bge_failed"}},
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
    core::emit_event("resume_end", run_id,
                     {{"success", false},
                      {"status", "unsupported_phase"},
                      {"from_phase", phase_upper}},
                     log_file);
    return 1;
  }

  if (phase_l == "pcc") {
    if (!run_astrometry_if_needed()) {
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "astrometry_failed"}},
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
      core::emit_event("resume_end", run_id,
                       {{"success", true}, {"status", "ok"}}, log_file);
      return 0;
    }

    if (!have_wcs) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "no_wcs"},
                         {"input_rgb", pcc_input_rgb_path.string()}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "no_wcs"}}, log_file);
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
          "resume_end", run_id,
          {{"success", false}, {"status", "no_catalog_stars"}}, log_file);
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
            "resume_end", run_id,
            {{"success", false}, {"status", "output_canvas_mask_invalid"}},
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
            "resume_end", run_id,
            {{"success", false}, {"status", "output_canvas_mask_invalid"}},
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
            "resume_end", run_id,
            {{"success", false}, {"status", "analysis_mask_invalid"}},
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
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "fit_failed"}},
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
    io::write_fits_rgb(pcc_rgb_path, rgb.R, rgb.G, rgb.B, out_hdr);

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
        core::emit_event("resume_end", run_id,
                         {{"success", false}, {"status", "stretch_failed"}},
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
      fs::remove(hms_rgb_path, hms_ec);
      io::write_fits_rgb(hms_rgb_path, rgb.R, rgb.G, rgb.B, hms_hdr);
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

  core::emit_event("resume_end", run_id, {{"success", true}, {"status", "ok"}},
                   log_file);
  return 0;
}
