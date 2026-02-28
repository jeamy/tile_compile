#include "runner_resume.hpp"

#include "tile_compile/astrometry/gaia_catalog.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/background_extraction.hpp"
#include "tile_compile/io/fits_io.hpp"

#include "runner_shared.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using tile_compile::Tile;
using tile_compile::TileGrid;
using tile_compile::TileMetrics;
using tile_compile::TileType;

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

tile_compile::core::json bge_value_stats_to_json(
    const tile_compile::image::BGEValueStats &s) {
  return tile_compile::core::json{{"n", s.n},
                                  {"min", s.min},
                                  {"max", s.max},
                                  {"median", s.median},
                                  {"mean", s.mean},
                                  {"std", s.std}};
}

tile_compile::core::json bge_diag_to_json(
    const tile_compile::image::BGEDiagnostics &diag,
    bool requested,
    bool have_tile_data,
    bool metrics_tiles_match) {
  namespace core = tile_compile::core;
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
      {"best_objective", diag.autotune_best_objective},
      {"best_objective_raw", diag.autotune_best_objective_raw},
      {"best_objective_normalized", diag.autotune_best_objective_normalized},
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
        {"selected_grid_spacing", ch.autotune_selected_grid_spacing},
        {"best",
         {
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
    ch_json["input_stats"] = bge_value_stats_to_json(ch.input_stats);
    ch_json["output_stats"] = bge_value_stats_to_json(ch.output_stats);
    ch_json["model_stats"] = bge_value_stats_to_json(ch.model_stats);
    ch_json["sample_bg_stats"] = bge_value_stats_to_json(ch.sample_bg_stats);
    ch_json["sample_weight_stats"] = bge_value_stats_to_json(ch.sample_weight_stats);
    ch_json["residual_stats"] = bge_value_stats_to_json(ch.residual_stats);
    ch_json["sample_bg_values"] = ch.sample_bg_values;
    ch_json["sample_weight_values"] = ch.sample_weight_values;
    ch_json["residual_values"] = ch.residual_values;
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
  fs::create_directories(run_dir / "logs");

  std::ofstream event_log_file(run_dir / "logs" / "run_events.jsonl",
                               std::ios::out | std::ios::app);
  tile_compile::runner::TeeBuf tee_buf(std::cout.rdbuf(), event_log_file.rdbuf());
  std::ostream log_file(&tee_buf);

  core::emit_event("resume_start", run_id,
                   {{"run_dir", run_dir.string()}, {"from_phase", from_phase}},
                   log_file);

  std::string phase_l = core::to_lower(from_phase);
  if (phase_l.empty())
    phase_l = "pcc";

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

  auto run_astrometry_if_needed = [&]() {
    if (have_wcs)
      return;

    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::ASTROMETRY, "ASTROMETRY", log_file);

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

    std::string cmd = shell_quote(astap_bin) + " -f " +
                      shell_quote(rgb_path.string()) + " -d " +
                      shell_quote(astap_data) + " -r " +
                      std::to_string(cfg.astrometry.search_radius);

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
  std::vector<TileMetrics> bge_tile_metrics;
  TileGrid bge_tile_grid;
  bool bge_have_local_metrics = false;
  bool bge_have_bge_grid = false;
  bool bge_metrics_tiles_match = false;
  bool bge_tile_context_loaded = false;

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
      std::cerr << "[BGE][resume] Warning: " << local_err << std::endl;
    }
    if (!ok_grid) {
      std::cerr << "[BGE][resume] Warning: " << grid_err << std::endl;
    }
  };

  auto write_stretched_rgb_snapshot = [&](const fs::path &path,
                                          const io::FitsHeader &hdr,
                                          bool apply_stretch,
                                          const char* stage_tag) {
    Matrix2Df R_disk = rgb.R;
    Matrix2Df G_disk = rgb.G;
    Matrix2Df B_disk = rgb.B;
    if (apply_stretch) {
      float vmin = std::numeric_limits<float>::max();
      float vmax = std::numeric_limits<float>::lowest();
      for (auto *ch : {&R_disk, &G_disk, &B_disk}) {
        for (Eigen::Index k = 0; k < ch->size(); ++k) {
          const float v = ch->data()[k];
          if (std::isfinite(v) && v > 0.0f) {
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
          }
        }
      }
      const float range = vmax - vmin;
      if (range > 1.0e-6f) {
        const float scale = 65535.0f / range;
        for (auto *ch : {&R_disk, &G_disk, &B_disk}) {
          for (Eigen::Index k = 0; k < ch->size(); ++k) {
            const float v = ch->data()[k];
            if (std::isfinite(v) && v > 0.0f) {
              ch->data()[k] = (v - vmin) * scale;
            } else {
              ch->data()[k] = 0.0f;
            }
          }
        }
        std::cout << "[" << stage_tag << "][resume] RGB output stretch: ["
                  << vmin << ".." << vmax << "] -> [0..65535]" << std::endl;
      }
    }
    std::error_code ec;
    fs::remove(path, ec);
    io::write_fits_rgb(path, R_disk, G_disk, B_disk, hdr);
  };

  auto run_bge_phase = [&]() {
    namespace image = tile_compile::image;
    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::BGE, "BGE", log_file);

    io::FitsHeader bge_hdr = rgb.header;
    if (have_wcs) {
      inject_wcs_keywords(bge_hdr, wcs);
    }

    if (!cfg.bge.enabled) {
      const bool bge_is_final_output = cfg.bge.enabled && !cfg.pcc.enabled;
      write_stretched_rgb_snapshot(
          stacked_rgb_bge_path, bge_hdr,
          cfg.stacking.output_stretch && bge_is_final_output, "BGE");
      emitter.phase_end(run_id, Phase::BGE, "skipped",
                        {{"reason", "disabled"},
                         {"artifact", (run_dir / "artifacts" / "bge.json").string()}},
                        log_file);
      return;
    }

    load_bge_tile_context_if_needed();
    const bool bge_have_tile_data = bge_have_local_metrics && bge_have_bge_grid;

    image::BGEDiagnostics bge_diag;
    if (bge_have_tile_data && bge_metrics_tiles_match) {
      image::BGEConfig bge_cfg =
          tile_compile::runner::to_image_bge_config(cfg.bge);

      (void)image::apply_background_extraction(rgb.R, rgb.G, rgb.B,
                                               bge_tile_metrics, bge_tile_grid,
                                               bge_cfg, &bge_diag);
    } else {
      std::cerr << "[BGE][resume] Skipping BGE fit (missing/mismatched tile artifacts)"
                << std::endl;
    }

    core::json bge_artifact =
        bge_diag_to_json(bge_diag, cfg.bge.enabled, bge_have_tile_data,
                         bge_metrics_tiles_match);
    bge_artifact["have_local_metrics"] = bge_have_local_metrics;
    bge_artifact["have_bge_grid"] = bge_have_bge_grid;
    bge_artifact["local_metrics_tiles"] = static_cast<int>(bge_tile_metrics.size());
    bge_artifact["bge_grid_tiles"] = static_cast<int>(bge_tile_grid.tiles.size());
    bge_artifact["config"] = {
        {"sample_quantile", cfg.bge.sample_quantile},
        {"structure_thresh_percentile", cfg.bge.structure_thresh_percentile},
        {"min_tiles_per_cell", cfg.bge.min_tiles_per_cell},
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
    const bool bge_is_final_output = cfg.bge.enabled && !cfg.pcc.enabled;
    write_stretched_rgb_snapshot(
        stacked_rgb_bge_path, bge_hdr,
        cfg.stacking.output_stretch && bge_is_final_output, "BGE");

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
      phase_extra["reason"] = "fit_failed";
    }

    emitter.phase_end(run_id, Phase::BGE, bge_diag.success ? "ok" : "skipped",
                      phase_extra, log_file);
  };

  if (phase_l == "astrometry") {
    run_astrometry_if_needed();
    phase_l = "bge";
  }
  if (phase_l == "bge") {
    run_astrometry_if_needed();
    run_bge_phase();
    phase_l = "pcc";
  } else if (phase_l != "pcc") {
    std::cerr << "Error: resume --from-phase supports ASTROMETRY, BGE, or PCC"
              << std::endl;
    core::emit_event("resume_end", run_id,
                     {{"success", false}, {"status", "unsupported_phase"}},
                     log_file);
    return 1;
  }

  if (phase_l == "pcc") {
    run_astrometry_if_needed();

    if (cfg.bge.enabled && fs::exists(stacked_rgb_bge_path)) {
      try {
        rgb = io::read_fits_rgb(stacked_rgb_bge_path);
        std::cerr << "[PCC][resume] Using precomputed BGE snapshot: "
                  << stacked_rgb_bge_path << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "[PCC][resume] Warning: failed to load stacked_rgb_bge.fits: "
                  << e.what() << std::endl;
      }
    }

    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::PCC, "PCC", log_file);

    io::FitsHeader out_hdr = rgb.header;
    if (have_wcs) {
      inject_wcs_keywords(out_hdr, wcs);
    }

    if (!cfg.pcc.enabled) {
      if (cfg.bge.enabled) {
        write_stretched_rgb_snapshot(
            stacked_rgb_bge_path, out_hdr, cfg.stacking.output_stretch, "BGE");
      } else if (cfg.stacking.output_stretch) {
        write_stretched_rgb_snapshot(
            stacked_rgb_path, out_hdr, true, "FINAL");
        write_stretched_rgb_snapshot(
            stacked_rgb_solve_path, out_hdr, true, "FINAL");
      }
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "disabled"},
                         {"input_rgb_bge", stacked_rgb_bge_path.string()}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", true}, {"status", "ok"}}, log_file);
      return 0;
    }

    if (!have_wcs) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "no_wcs"},
                         {"input_rgb_bge", stacked_rgb_bge_path.string()}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "no_wcs"}}, log_file);
      return 1;
    }

    double search_r = wcs.search_radius_deg();
    std::string source = cfg.pcc.source;
    std::string used_source;
    std::vector<astro::GaiaStar> stars;

    auto try_siril = [&]() -> bool {
      std::string cat_dir = cfg.pcc.siril_catalog_dir;
      if (cat_dir.empty())
        cat_dir = astro::default_siril_gaia_catalog_dir();
      if (!astro::is_siril_gaia_catalog_available(cat_dir))
        return false;
      std::cerr << "[PCC][resume] Querying Siril Gaia catalog at RA="
                << wcs.crval1 << " Dec=" << wcs.crval2 << " r=" << search_r
                << " deg" << std::endl;
      stars = astro::siril_gaia_cone_search(cat_dir, wcs.crval1, wcs.crval2,
                                            search_r, cfg.pcc.mag_limit);
      if (!stars.empty()) {
        used_source = "siril";
        return true;
      }
      return false;
    };

    auto try_vizier_gaia = [&]() -> bool {
      std::cerr << "[PCC][resume] Querying VizieR Gaia DR3 at RA=" << wcs.crval1
                << " Dec=" << wcs.crval2 << " r=" << search_r << " deg"
                << std::endl;
      stars = astro::vizier_gaia_cone_search(wcs.crval1, wcs.crval2, search_r,
                                             cfg.pcc.mag_limit);
      if (!stars.empty()) {
        used_source = "vizier_gaia";
        return true;
      }
      return false;
    };

    auto try_vizier_apass = [&]() -> bool {
      std::cerr << "[PCC][resume] Querying VizieR APASS DR9 at RA=" << wcs.crval1
                << " Dec=" << wcs.crval2 << " r=" << search_r << " deg"
                << std::endl;
      stars = astro::vizier_apass_cone_search(wcs.crval1, wcs.crval2, search_r,
                                              cfg.pcc.mag_limit);
      if (!stars.empty()) {
        used_source = "vizier_apass";
        return true;
      }
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
        std::cerr << "[PCC][resume] Siril catalog not available, trying VizieR Gaia..."
                  << std::endl;
        if (!try_vizier_gaia()) {
          std::cerr << "[PCC][resume] VizieR Gaia failed, trying VizieR APASS..."
                    << std::endl;
          try_vizier_apass();
        }
      }
    }

    std::cerr << "[PCC][resume] Found " << stars.size() << " catalog stars"
              << " (source: " << (used_source.empty() ? "none" : used_source)
              << ")" << std::endl;

    if (stars.empty()) {
      emitter.phase_end(run_id, Phase::PCC, "skipped",
                        {{"reason", "no_catalog_stars"},
                         {"search_radius_deg", search_r},
                         {"source", source},
                         {"input_rgb_bge", stacked_rgb_bge_path.string()}},
                        log_file);
      core::emit_event(
          "resume_end", run_id,
          {{"success", false}, {"status", "no_catalog_stars"}}, log_file);
      return 1;
    }

    astro::PCCConfig pcc_cfg =
        tile_compile::runner::to_astrometry_pcc_config(cfg.pcc);

    load_bge_tile_context_if_needed();
    if (bge_metrics_tiles_match) {
      pcc_cfg.use_tile_quality_weighting = true;
      pcc_cfg.tile_grid = bge_tile_grid;
      pcc_cfg.tile_metrics = bge_tile_metrics;
    }

    if (pcc_cfg.radii_mode == "auto_fwhm") {
      // Resume path has no guaranteed seeing estimate artifact available:
      // use deterministic spec fallback FWHM=0.
      const double F = 0.0;
      const double r_ap = std::max(static_cast<double>(pcc_cfg.min_aperture_px),
                                   pcc_cfg.aperture_fwhm_mult * F);
      const double r_in = std::max(r_ap + 1.0,
                                   pcc_cfg.annulus_inner_fwhm_mult * F);
      const double r_out = std::max(r_in + 2.0,
                                    pcc_cfg.annulus_outer_fwhm_mult * F);
      pcc_cfg.aperture_radius_px = r_ap;
      pcc_cfg.annulus_inner_px = r_in;
      pcc_cfg.annulus_outer_px = r_out;
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
                         {"source", used_source},
                         {"input_rgb_bge", stacked_rgb_bge_path.string()}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "fit_failed"}},
                       log_file);
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
          const float v = ch->data()[k];
          if (std::isfinite(v) && v > 0.0f) {
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
          }
        }
      }
      const float range = vmax - vmin;
      if (range > 1.0e-6f) {
        const float scale = 65535.0f / range;
        for (auto *ch : {&R_pcc_disk, &G_pcc_disk, &B_pcc_disk}) {
          for (Eigen::Index k = 0; k < ch->size(); ++k) {
            const float v = ch->data()[k];
            if (std::isfinite(v) && v > 0.0f) {
              ch->data()[k] = (v - vmin) * scale;
            } else {
              ch->data()[k] = 0.0f;
            }
          }
        }
        std::cout << "[PCC][resume] RGB output stretch: [" << vmin << ".."
                  << vmax << "] -> [0..65535]" << std::endl;
      }
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
    io::write_fits_float(pcc_r_path, R_pcc_disk, out_hdr);
    io::write_fits_float(pcc_g_path, G_pcc_disk, out_hdr);
    io::write_fits_float(pcc_b_path, B_pcc_disk, out_hdr);
    io::write_fits_rgb(pcc_rgb_path, R_pcc_disk, G_pcc_disk, B_pcc_disk, out_hdr);

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
                       {"matrix", matrix_json},
                       {"source", used_source},
                       {"input_rgb_bge", stacked_rgb_bge_path.string()}},
                      log_file);
  }

  core::emit_event("resume_end", run_id, {{"success", true}, {"status", "ok"}},
                   log_file);
  return 0;
}
