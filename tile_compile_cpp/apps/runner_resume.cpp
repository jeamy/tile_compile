#include "runner_resume.hpp"
#include "runner_downstream.hpp"

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
#include "runner_phase_aqmh_reconstruction.hpp"
#include "runner_phase_aqmh_diagnostics.hpp"

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

using tile_compile::Tile;
using tile_compile::TileGrid;
using tile_compile::TileMetrics;
using tile_compile::TileType;
using tile_compile::runner::WarpBounds;
using tile_compile::runner::compute_warps_bounds;
namespace runner = tile_compile::runner;

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
  cmd << runner::shell_quote(exe_path.string()) << " run"
      << " --config " << runner::shell_quote(rerun_config_path.string())
      << " --input-dir " << runner::shell_quote(*input_dir)
      << " --runs-dir " << runner::shell_quote(run_dir.parent_path().string())
      << " --project-root " << runner::shell_quote(fs::current_path().string())
      << " --run-id " << runner::shell_quote(run_id);

  {
    std::ofstream event_log_file(run_dir / "logs" / "run_events.jsonl",
                                 std::ios::out | std::ios::app);
    if (event_log_file) {
      tile_compile::runner::TeeBuf tee_buf(std::cout.rdbuf(),
                                           event_log_file.rdbuf());
      std::ostream log_file(&tee_buf);
      core::emit_event(
          "resume_start", run_id,
          {{"run_dir", run_dir.string()},
           {"from_phase", from_phase},
           {"mode", "inplace_full_rerun"}},
          log_file);
    }
  }

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

  if (const std::string reason = core::pipeline_unavailable_reason();
      !reason.empty()) {
    std::cerr << "Error: " << reason << std::endl;
    return 1;
  }

#ifdef TILE_COMPILE_LEGACY_REFERENCE
  std::cerr << "Error: LEGACY_REFERENCE_RESUME_DISABLED --- reference fixtures cannot resume user runs" << std::endl;
  return 1;
#endif

  fs::path run_dir(run_dir_path);
  if (!fs::exists(run_dir) || !fs::is_directory(run_dir)) {
    std::cerr << "Error: run_dir not found: " << run_dir_path << std::endl;
    return 1;
  }

  // M0 / plan section 18.1: before opening any artifact for writing, read the
  // run's pipeline_contract_version. If this binary speaks the single-method
  // contract but the run was written under an older (Classic / PREWARP-AQMH)
  // contract, the resume is refused fail-closed. While the active pipeline is
  // still legacy (kPipelineContractVersionActive == 0) this check is dormant:
  // a legacy binary resuming a legacy run is allowed.
  {
    int run_contract_version = -1;  // missing field == legacy
    const fs::path prov_path = run_dir / "artifacts" / "run_provenance.json";
    if (fs::is_regular_file(prov_path)) {
      try {
        const auto prov =
            core::json::parse(core::read_text(prov_path));
        if (prov.contains("pipeline_contract_version") &&
            prov["pipeline_contract_version"].is_number_integer()) {
          run_contract_version =
              prov["pipeline_contract_version"].get<int>();
        }
      } catch (...) {
        run_contract_version = -1;
      }
    }
    if (core::pipeline_contract_is_single_method(
            core::kPipelineContractVersionActive) &&
        !core::pipeline_contract_is_single_method(run_contract_version)) {
      std::cerr
          << "Error: LEGACY_RUN_NOT_RESUMABLE --- this run was written under "
          << "pipeline contract '"
          << core::pipeline_contract_label(run_contract_version)
          << "' (version " << run_contract_version
          << ") and cannot be resumed with the single-method runner (contract '"
          << core::pipeline_contract_label(
                 core::kPipelineContractVersionActive)
          << "'). Start a full new run from the unchanged source frames."
          << std::endl;
      return 1;
    }
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
      if (cfg.pcc.enabled && fs::exists(pcc_r) && fs::exists(pcc_g) &&
          fs::exists(pcc_b)) {
        std::tie(R, hdr) = io::read_fits_float(pcc_r);
        G = io::read_fits_pixels_float(pcc_g);
        B = io::read_fits_pixels_float(pcc_b);
        input_stage = "pcc_channels";
        input_rgb = pcc_r.string() + ";" + pcc_g.string() + ";" + pcc_b.string();
      } else if (cfg.pcc.enabled &&
                 fs::exists(outputs_dir / "stacked_rgb_pcc.fits")) {
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
                       {{"success", false},
                        {"status", "read_input_failed"},
                        {"error", e.what()}},
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
                       {{"success", false},
                        {"status", "missing_pcc_artifacts"},
                        {"error", "no PCC output artifacts found and require_successful_pcc is true"}},
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
        runner::to_image_hms_config(cfg.hypermetric_stretch);
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
                       {{"success", false},
                        {"status", "stretch_failed"},
                        {"error", hms_diag.error_message}},
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
                       {{"success", false},
                        {"status", "write_output_failed"},
                        {"error", e.what()}},
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
    if (common_overlap_mask_image.rows() != canvas_height ||
        common_overlap_mask_image.cols() != canvas_width) {
      std::cerr << "Error: AQMH common-overlap mask dimensions differ from "
                   "the reconstruction cache"
                << std::endl;
      return 1;
    }
    std::vector<uint8_t> common_valid_mask(
        static_cast<size_t>(canvas_width) * canvas_height, 0u);
    for (size_t i = 0; i < common_valid_mask.size(); ++i)
      common_valid_mask[i] =
          common_overlap_mask_image.data()[i] > 0.0f ? 1u : 0u;

    runner::DiskCacheFrameStore prewarped_frames(
        run_dir / "cache" / "prewarped_frames", frame_count, canvas_height,
        canvas_width, true);
    std::vector<uint8_t> frame_has_data(frame_count, 0u);
    size_t available_frames = 0;
    for (size_t fi = 0; fi < frame_count; ++fi) {
      frame_has_data[fi] = prewarped_frames.has_data(fi) ? 1u : 0u;
      available_frames += frame_has_data[fi] != 0u;
    }
    if (available_frames == 0) {
      core::emit_event("resume_end", run_id,
                       {{"success", false},
                        {"status", "prewarped_cache_missing"},
                        {"reason", "no_reusable_prewarped_cache_frames"},
                        {"cache_dir", (run_dir / "cache" / "prewarped_frames").string()}},
                       log_file);
      std::cerr << "Error: no reusable cache/prewarped_frames frames found"
                << std::endl;
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
            cv::getNumThreads(), phase_result)) {
      return 1;
    }
    try {
      io::write_fits_float(aqmh_raw_reconstruction, phase_result.raw_output,
                           resume_header);
      io::write_fits_float(run_dir / "outputs" / "reconstructed_L.fit",
                           phase_result.output, resume_header);
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
                       {{"success", false},
                        {"status", "read_synthetic_failed"},
                        {"error", e.what()},
                        {"file", synthetic_entries.front().second.string()}},
                       log_file);
      return 1;
    }
    if (first_synth.size() <= 0) {
      emitter.phase_end(run_id, Phase::STACKING, "error",
                        {{"reason", "empty_synthetic"},
                         {"file", synthetic_entries.front().second.string()}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false},
                        {"status", "empty_synthetic"},
                        {"error", "first synthetic frame has zero pixel data"},
                        {"file", synthetic_entries.front().second.string()}},
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
                          {"status", "normalization_artifact_invalid"},
                          {"error", scaling_error}},
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
                         {{"success", false},
                          {"status", "invalid_stored_crop"},
                          {"error", "stored crop bbox is out of bounds for synthetic frame"},
                          {"crop_x", crop.x}, {"crop_y", crop.y},
                          {"crop_width", crop.width},
                          {"crop_height", crop.height}},
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
                       {{"success", false},
                        {"status", "canvas_mask_invalid"},
                        {"error", canvas_mask_error}},
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
            {{"success", false},
             {"status", "analysis_mask_invalid"},
             {"error", analysis_mask_error}},
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
                         {{"success", false},
                          {"status", "read_synthetic_failed"},
                          {"error", e.what()},
                          {"file", path.string()}},
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
                            {"status", "synthetic_crop_out_of_bounds"},
                            {"error", "stored crop bbox exceeds synthetic frame dimensions"},
                            {"file", path.string()}},
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
                       {{"success", false},
                        {"status", "no_valid_synthetic"},
                        {"error", "all synthetic frame reads produced empty data"}},
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
      recon = image::rgb_to_luma(recon_R, recon_G, recon_B);
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
      recon = image::rgb_to_luma(recon_R, recon_G, recon_B);
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
                          {"status", "canvas_mask_size_mismatch"},
                          {"error", msg}},
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
            {{"success", false},
             {"status", "analysis_mask_size_mismatch"},
             {"error", "analysis mask pixel count does not match reconstruction canvas"}},
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
                         {{"success", false},
                          {"status", "empty_valid_crop"},
                          {"error", "crop_to_nonzero_bbox produced empty valid canvas"}},
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
                            {"status", "canvas_mask_write_failed"},
                            {"error", mask_write_error}},
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
                       {{"success", false},
                        {"status", "output_write_failed"},
                        {"error", post_result.error}},
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

  return runner::run_rgb_downstream(run_dir, run_id, cfg, normalize_phase_name(phase_l),
      log_file, abort_if_runtime_limit_exceeded, false);
}
