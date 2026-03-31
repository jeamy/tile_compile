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
#include "tile_compile/image/processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/reconstruction/reconstruction.hpp"

#include "runner_shared.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

std::string normalize_phase_name(std::string phase) {
  std::transform(phase.begin(), phase.end(), phase.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return phase;
}

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

struct WarpBounds {
  int min_x = 0;
  int min_y = 0;
  int max_x = 0;
  int max_y = 0;

  [[nodiscard]] int width() const { return max_x - min_x; }
  [[nodiscard]] int height() const { return max_y - min_y; }
};

bool invert_affine_warp(const tile_compile::WarpMatrix &w,
                        tile_compile::WarpMatrix &inv) {
  const float a = w(0, 0);
  const float b = w(0, 1);
  const float c = w(1, 0);
  const float d = w(1, 1);
  const float tx = w(0, 2);
  const float ty = w(1, 2);
  const float det = a * d - b * c;
  if (std::fabs(det) < 1.0e-12f) {
    return false;
  }
  const float inv_det = 1.0f / det;
  inv(0, 0) = d * inv_det;
  inv(0, 1) = -b * inv_det;
  inv(1, 0) = -c * inv_det;
  inv(1, 1) = a * inv_det;
  inv(0, 2) = -(inv(0, 0) * tx + inv(0, 1) * ty);
  inv(1, 2) = -(inv(1, 0) * tx + inv(1, 1) * ty);
  return true;
}

WarpBounds compute_warps_bounds(int width, int height,
                                const std::vector<tile_compile::WarpMatrix> &warps) {
  WarpBounds b;
  if (width <= 0 || height <= 0 || warps.empty()) {
    b.max_x = std::max(0, width);
    b.max_y = std::max(0, height);
    return b;
  }

  const float corners_x[4] = {0.0f, static_cast<float>(width), 0.0f,
                              static_cast<float>(width)};
  const float corners_y[4] = {0.0f, 0.0f, static_cast<float>(height),
                              static_cast<float>(height)};

  bool init = false;
  float min_xf = 0.0f;
  float min_yf = 0.0f;
  float max_xf = 0.0f;
  float max_yf = 0.0f;
  for (const auto &w : warps) {
    tile_compile::WarpMatrix fwd;
    if (!invert_affine_warp(w, fwd)) {
      continue;
    }
    for (int i = 0; i < 4; ++i) {
      const float x = corners_x[i];
      const float y = corners_y[i];
      const float tx = fwd(0, 0) * x + fwd(0, 1) * y + fwd(0, 2);
      const float ty = fwd(1, 0) * x + fwd(1, 1) * y + fwd(1, 2);
      if (!init) {
        min_xf = max_xf = tx;
        min_yf = max_yf = ty;
        init = true;
      } else {
        min_xf = std::min(min_xf, tx);
        min_yf = std::min(min_yf, ty);
        max_xf = std::max(max_xf, tx);
        max_yf = std::max(max_yf, ty);
      }
    }
  }

  if (!init) {
    b.max_x = std::max(0, width);
    b.max_y = std::max(0, height);
    return b;
  }

  b.min_x = static_cast<int>(std::floor(min_xf));
  b.min_y = static_cast<int>(std::floor(min_yf));
  b.max_x = static_cast<int>(std::ceil(max_xf));
  b.max_y = static_cast<int>(std::ceil(max_yf));
  return b;
}

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

    if (!consistent) {
      out.clear();
      out.reserve(n_tiles);
      for (size_t ti = 0; ti < n_tiles; ++ti) {
        out.push_back(parse_tile_metrics_json(all_frames.front()[ti]));
      }
      return !out.empty();
    }

    auto median_or_zero = [](std::vector<float> vals) -> float {
      if (vals.empty()) return 0.0f;
      return tile_compile::core::median_of(vals);
    };

    out.assign(n_tiles, TileMetrics{});
    for (size_t ti = 0; ti < n_tiles; ++ti) {
      std::vector<float> fwhm_vals;
      std::vector<float> round_vals;
      std::vector<float> contrast_vals;
      std::vector<float> sharp_vals;
      std::vector<float> bg_vals;
      std::vector<float> noise_vals;
      std::vector<float> grad_vals;
      std::vector<float> q_vals;
      std::vector<float> star_count_vals;
      int star_votes = 0;
      int structure_votes = 0;

      fwhm_vals.reserve(all_frames.size());
      round_vals.reserve(all_frames.size());
      contrast_vals.reserve(all_frames.size());
      sharp_vals.reserve(all_frames.size());
      bg_vals.reserve(all_frames.size());
      noise_vals.reserve(all_frames.size());
      grad_vals.reserve(all_frames.size());
      q_vals.reserve(all_frames.size());
      star_count_vals.reserve(all_frames.size());

      for (const auto &fm : all_frames) {
        const TileMetrics tm = parse_tile_metrics_json(fm[ti]);
        if (std::isfinite(tm.fwhm)) fwhm_vals.push_back(tm.fwhm);
        if (std::isfinite(tm.roundness)) round_vals.push_back(tm.roundness);
        if (std::isfinite(tm.contrast)) contrast_vals.push_back(tm.contrast);
        if (std::isfinite(tm.sharpness)) sharp_vals.push_back(tm.sharpness);
        if (std::isfinite(tm.background)) bg_vals.push_back(tm.background);
        if (std::isfinite(tm.noise)) noise_vals.push_back(tm.noise);
        if (std::isfinite(tm.gradient_energy)) grad_vals.push_back(tm.gradient_energy);
        if (std::isfinite(tm.quality_score)) q_vals.push_back(tm.quality_score);
        star_count_vals.push_back(static_cast<float>(tm.star_count));
        if (tm.type == TileType::STAR) {
          ++star_votes;
        } else {
          ++structure_votes;
        }
      }

      TileMetrics agg{};
      agg.fwhm = median_or_zero(std::move(fwhm_vals));
      agg.roundness = median_or_zero(std::move(round_vals));
      agg.contrast = median_or_zero(std::move(contrast_vals));
      agg.sharpness = median_or_zero(std::move(sharp_vals));
      agg.background = median_or_zero(std::move(bg_vals));
      agg.noise = median_or_zero(std::move(noise_vals));
      agg.gradient_energy = median_or_zero(std::move(grad_vals));
      agg.quality_score = median_or_zero(std::move(q_vals));
      agg.star_count = static_cast<int>(
          std::lround(median_or_zero(std::move(star_count_vals))));
      agg.type = (star_votes >= structure_votes) ? TileType::STAR
                                                 : TileType::STRUCTURE;
      out[ti] = agg;
    }
    return true;
  } catch (const std::exception &e) {
    error_out = std::string("local_metrics parse failed: ") + e.what();
    return false;
  }
}

}  // namespace

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

  if (phase_l == "stacking") {
    namespace image = tile_compile::image;
    namespace reconstruction = tile_compile::reconstruction;

    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::STACKING, "STACKING", log_file);

    std::vector<std::pair<int, fs::path>> synthetic_entries;
    const fs::path outputs_dir = run_dir / "outputs";
    if (fs::exists(outputs_dir) && fs::is_directory(outputs_dir)) {
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
      const std::string msg = "missing synthetic_*.fit outputs for STACKING resume";
      emitter.phase_end(run_id, Phase::STACKING, "error",
                        {{"reason", "missing_synthetic_outputs"},
                         {"outputs_dir", outputs_dir.string()},
                         {"error", msg}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "missing_synthetic"}},
                       log_file);
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

    std::vector<uint8_t> common_valid_mask;
    std::string canvas_mask_error;
    if (!tile_compile::runner::load_canvas_mask_fits(
            run_dir / "outputs" / "canvas_mask.fits", first_synth.rows(),
            first_synth.cols(), common_valid_mask, canvas_mask_error)) {
      emitter.phase_end(run_id, Phase::STACKING, "error",
                        {{"reason", "canvas_mask_invalid"},
                         {"canvas_mask",
                          (run_dir / "outputs" / "canvas_mask.fits").string()},
                         {"error", canvas_mask_error}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "canvas_mask_invalid"}},
                       log_file);
      return 1;
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

    const auto stacking_acceleration = core::select_acceleration_backend(
        cfg.runtime_limits.acceleration_backend,
        core::AccelerationPhase::stacking);
    const core::AccelerationOps stacking_ops(stacking_acceleration);
    {
      std::ostringstream msg;
      msg << "STACKING acceleration "
          << core::acceleration_selection_summary(stacking_acceleration);
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
        recon_R = stacking_ops.sigma_clip_stack(
            synth_R, cfg.stacking.sigma_clip.sigma_low,
            cfg.stacking.sigma_clip.sigma_high, cfg.stacking.sigma_clip.max_iters,
            cfg.stacking.sigma_clip.min_fraction);
        recon_G = stacking_ops.sigma_clip_stack(
            synth_G, cfg.stacking.sigma_clip.sigma_low,
            cfg.stacking.sigma_clip.sigma_high, cfg.stacking.sigma_clip.max_iters,
            cfg.stacking.sigma_clip.min_fraction);
        recon_B = stacking_ops.sigma_clip_stack(
            synth_B, cfg.stacking.sigma_clip.sigma_low,
            cfg.stacking.sigma_clip.sigma_high, cfg.stacking.sigma_clip.max_iters,
            cfg.stacking.sigma_clip.min_fraction);
      } else {
        std::vector<float> stack_weights(synth_R.size(), 1.0f);
        if (use_quality_weighting &&
            cluster_stack_weights.size() == synth_R.size()) {
          stack_weights = cluster_stack_weights;
        }
        auto wr_r = stacking_ops.sigma_clip_reduce(
            synth_R, stack_weights, cfg.stacking.sigma_clip.sigma_low,
            cfg.stacking.sigma_clip.sigma_high,
            cfg.stacking.sigma_clip.max_iters,
            cfg.stacking.sigma_clip.min_fraction, kEpsWeight);
        auto wr_g = stacking_ops.sigma_clip_reduce(
            synth_G, stack_weights, cfg.stacking.sigma_clip.sigma_low,
            cfg.stacking.sigma_clip.sigma_high,
            cfg.stacking.sigma_clip.max_iters,
            cfg.stacking.sigma_clip.min_fraction, kEpsWeight);
        auto wr_b = stacking_ops.sigma_clip_reduce(
            synth_B, stack_weights, cfg.stacking.sigma_clip.sigma_low,
            cfg.stacking.sigma_clip.sigma_high,
            cfg.stacking.sigma_clip.max_iters,
            cfg.stacking.sigma_clip.min_fraction, kEpsWeight);
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
            cfg.stacking.sigma_clip.min_fraction);
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
            cfg.stacking.sigma_clip.min_fraction, kEpsWeight);
        recon = std::move(wr.tile);
      }
    }

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

      stacking_crop_box = tile_compile::runner::compute_nonzero_data_bbox(
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
          }
        }
        common_valid_mask.swap(cropped_mask);

        std::string mask_write_error;
        if (!write_canvas_mask_fits(run_dir / "outputs" / "canvas_mask.fits",
                                    common_valid_mask, crop_h, crop_w, first_hdr,
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

    Matrix2Df recon_out = recon;
    stretch_luma_for_output(recon_out);
    try {
      io::write_fits_float(run_dir / "outputs" / "stacked.fits", recon_out, first_hdr);
      io::write_fits_float(run_dir / "outputs" / "reconstructed_L.fit", recon_out,
                           first_hdr);
    } catch (const std::exception &e) {
      emitter.phase_end(run_id, Phase::STACKING, "error",
                        {{"reason", "write_failed"}, {"error", e.what()}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "stack_write_failed"}},
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
         {"crop_x", stacking_crop_box.x},
         {"crop_y", stacking_crop_box.y},
         {"crop_width", stacking_crop_box.width},
         {"crop_height", stacking_crop_box.height},
         {"output_luma", (run_dir / "outputs" / "stacked.fits").string()}},
        log_file);
    if (abort_if_runtime_limit_exceeded("STACKING")) {
      return 1;
    }

    emitter.phase_start(run_id, Phase::DEBAYER, "DEBAYER", log_file);
    if (detected_mode == ColorMode::OSC) {
      Matrix2Df R_out;
      Matrix2Df G_out;
      Matrix2Df B_out;
      if (recon_R.size() == recon.size() && recon_R.size() > 0) {
        R_out = recon_R;
        G_out = recon_G;
        B_out = recon_B;
      } else {
        auto debayer = image::debayer_nearest_neighbor(
            recon, detected_bayer, -debayer_tile_offset_x, -debayer_tile_offset_y);
        R_out = std::move(debayer.R);
        G_out = std::move(debayer.G);
        B_out = std::move(debayer.B);
      }
      try {
        io::write_fits_float(run_dir / "outputs" / "reconstructed_R.fit", R_out,
                             first_hdr);
        io::write_fits_float(run_dir / "outputs" / "reconstructed_G.fit", G_out,
                             first_hdr);
        io::write_fits_float(run_dir / "outputs" / "reconstructed_B.fit", B_out,
                             first_hdr);
        Matrix2Df R_stack_disk = R_out;
        Matrix2Df G_stack_disk = G_out;
        Matrix2Df B_stack_disk = B_out;
        if (cfg.stacking.output_stretch) {
          float vmin = std::numeric_limits<float>::max();
          float vmax = std::numeric_limits<float>::lowest();
          for (auto *ch : {&R_stack_disk, &G_stack_disk, &B_stack_disk}) {
            for (Eigen::Index k = 0; k < ch->size(); ++k) {
              const float v = ch->data()[k];
              if (std::isfinite(v) && v > 0.0f) {
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
              }
            }
          }
          const float range = vmax - vmin;
          if (range > 1.0e-6f) {
            const float scale = 65535.0f / range;
            for (auto *ch : {&R_stack_disk, &G_stack_disk, &B_stack_disk}) {
              for (Eigen::Index k = 0; k < ch->size(); ++k) {
                const float v = ch->data()[k];
                if (std::isfinite(v) && v > 0.0f) {
                  ch->data()[k] = (v - vmin) * scale;
                } else {
                  ch->data()[k] = 0.0f;
                }
              }
            }
            std::cout << "[STACKING][resume] RGB output stretch: [" << vmin
                      << ".." << vmax << "] -> [0..65535]" << std::endl;
          }
        }
        io::write_fits_rgb(run_dir / "outputs" / "stacked_rgb.fits",
                           R_stack_disk, G_stack_disk, B_stack_disk, first_hdr);
        io::write_fits_rgb(run_dir / "outputs" / "stacked_rgb_solve.fits", R_out,
                           G_out, B_out, first_hdr);
      } catch (const std::exception &e) {
        emitter.phase_end(run_id, Phase::DEBAYER, "error",
                          {{"reason", "write_failed"}, {"error", e.what()}},
                          log_file);
        core::emit_event("resume_end", run_id,
                         {{"success", false}, {"status", "debayer_write_failed"}},
                         log_file);
        return 1;
      }
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

  auto run_astrometry_if_needed = [&](bool force_rerun = false) {
    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::ASTROMETRY, "ASTROMETRY", log_file);

    if (have_wcs && !force_rerun) {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "existing_wcs"},
                         {"wcs_file",
                          (run_dir / "artifacts" / "stacked_rgb.wcs")
                              .string()}},
                        log_file);
      return;
    }

    if (force_rerun) {
      have_wcs = false;
    }

    if (!cfg.astrometry.enabled) {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "disabled"}}, log_file);
      return;
    }

    std::string astap_data = cfg.astrometry.astap_data_dir;
    if (astap_data.empty()) {
      const char *home = std::getenv("HOME");
      if (home)
        astap_data = std::string(home) + "/.local/share/tile_compile/astap";
    }
    std::string astap_bin = cfg.astrometry.astap_bin;
    if (astap_bin.empty())
      astap_bin = astap_data + "/astap_cli";

    if (!fs::exists(astap_bin)) {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "astap_not_found"},
                         {"astap_bin", astap_bin}},
                        log_file);
      return;
    }

    std::string cmd = shell_quote(astap_bin) + " -f " +
                      shell_quote(rgb_path.string()) + " -d " +
                      shell_quote(astap_data) + " -r " +
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
    } else {
      emitter.phase_end(run_id, Phase::ASTROMETRY, "skipped",
                        {{"reason", "solve_failed"}, {"exit_code", ret}},
                        log_file);
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
	    Matrix2Df R_disk = R_src;
	    Matrix2Df G_disk = G_src;
	    Matrix2Df B_disk = B_src;
	    std::vector<uint8_t> canvas_mask;
	    std::string canvas_mask_error;
	    int canvas_rows = 0;
	    int canvas_cols = 0;
	    if (tile_compile::runner::load_canvas_mask_for_rgb(
	            run_dir / "outputs" / "canvas_mask.fits", R_disk, G_disk, B_disk,
	            canvas_mask, canvas_rows, canvas_cols, canvas_mask_error)) {
	      image::enforce_canvas_mask_on_rgb(R_disk, G_disk, B_disk, canvas_mask);
	    }
	    if (apply_stretch) {
	      const auto stretch =
	          tile_compile::core::stretch_rgb_to_u16_linear_from_zero_inplace(
	              R_disk, G_disk, B_disk);
      if (stretch.applied) {
        std::cout << "[" << stage_tag
                  << "][resume] RGB output "
                  << "linear"
                  << " stretch ["
                  << stretch.low << ".." << stretch.high << "] -> [0..65535]"
                  << " samples=" << stretch.sample_count << std::endl;
      }
    }
    std::error_code ec;
    fs::remove(path, ec);
    io::write_fits_rgb(path, R_disk, G_disk, B_disk, hdr);
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
    emitter.phase_start(run_id, Phase::BGE, "BGE", log_file);

    io::FitsHeader bge_hdr = rgb.header;
    if (have_wcs) {
      inject_wcs_keywords(bge_hdr, wcs);
    }

    if (!cfg.bge.enabled) {
      std::error_code ec_linear;
      std::error_code ec_display;
      fs::remove(stacked_rgb_bge_linear_path, ec_linear);
      fs::remove(stacked_rgb_bge_path, ec_display);
      emitter.phase_end(run_id, Phase::BGE, "skipped",
                        {{"reason", "disabled"},
                         {"artifact", (run_dir / "artifacts" / "bge.json").string()}},
                        log_file);
      return true;
    }

    load_bge_tile_context_if_needed();
    const bool bge_have_tile_data = bge_have_local_metrics && bge_have_bge_grid;

    image::BGEDiagnostics bge_diag;
    if (bge_have_tile_data && bge_metrics_tiles_match) {
      image::BGEConfig bge_cfg =
          tile_compile::runner::to_image_bge_config(cfg.bge);
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
    bge_artifact["have_bge_grid"] = bge_have_bge_grid;
    bge_artifact["local_metrics_tiles"] = static_cast<int>(bge_tile_metrics.size());
    bge_artifact["bge_grid_tiles"] = static_cast<int>(bge_tile_grid.tiles.size());
    bge_artifact["config"] = {
        {"sample_quantile", cfg.bge.sample_quantile},
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
    if (!bge_have_tile_data) {
      phase_extra["reason"] = "no_tile_data";
    } else if (!bge_metrics_tiles_match) {
      phase_extra["reason"] = "tile_metric_grid_mismatch";
    } else if (bge_diag.attempted && !bge_diag.success) {
      phase_extra["reason"] =
          bge_diag.failure_reason.empty() ? "fit_failed"
                                          : bge_diag.failure_reason;
    }

    emitter.phase_end(run_id, Phase::BGE, bge_diag.success ? "ok" : "skipped",
                      phase_extra, log_file);
    return true;
  };

  if (phase_l == "astrometry") {
    run_astrometry_if_needed(true);
    if (abort_if_runtime_limit_exceeded("ASTROMETRY")) {
      return 1;
    }
    phase_l = "bge";
  }
  if (phase_l == "bge") {
    run_astrometry_if_needed();
    if (abort_if_runtime_limit_exceeded("ASTROMETRY")) {
      return 1;
    }
    if (!run_bge_phase()) {
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "canvas_mask_invalid"}},
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
    run_astrometry_if_needed();
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
      if (!tile_compile::runner::load_canvas_mask_for_rgb(
              run_dir / "outputs" / "canvas_mask.fits", rgb.R, rgb.G, rgb.B,
              analysis_mask, analysis_rows, analysis_cols, analysis_mask_error)) {
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
    write_stretched_rgb_snapshot(
        pcc_rgb_path, rgb.R, rgb.G, rgb.B, out_hdr,
        cfg.stacking.output_stretch, "PCC");

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
    if (io::detect_color_mode(rgb.header, 2) == ColorMode::OSC &&
        cfg.chroma_denoise.enabled &&
        cfg.chroma_denoise.apply_stage == "post_pcc") {
      reconstruction::chroma_denoise_rgb_inplace(
          rgb.R, rgb.G, rgb.B, cfg.chroma_denoise);
    }

    if (abort_if_runtime_limit_exceeded("PCC")) {
      return 1;
    }
  }

  core::emit_event("resume_end", run_id, {{"success", true}, {"status", "ok"}},
                   log_file);
  return 0;
}
