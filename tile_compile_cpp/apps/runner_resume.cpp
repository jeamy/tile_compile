#include "runner_resume.hpp"

#include "tile_compile/astrometry/gaia_catalog.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"

#include "runner_shared.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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
  if (!fs::exists(rgb_path)) {
    rgb_path = run_dir / "outputs" / "stacked_rgb.fits";
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

  if (phase_l == "astrometry") {
    run_astrometry_if_needed();
    phase_l = "pcc";
  } else if (phase_l != "pcc") {
    std::cerr << "Error: resume --from-phase supports only ASTROMETRY or PCC"
              << std::endl;
    core::emit_event("resume_end", run_id,
                     {{"success", false}, {"status", "unsupported_phase"}},
                     log_file);
    return 1;
  }

  if (phase_l == "pcc") {
    run_astrometry_if_needed();

    core::EventEmitter emitter;
    emitter.phase_start(run_id, Phase::PCC, "PCC", log_file);

    if (!cfg.pcc.enabled) {
      emitter.phase_end(run_id, Phase::PCC, "skipped", {{"reason", "disabled"}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", true}, {"status", "ok"}}, log_file);
      return 0;
    }

    if (!have_wcs) {
      emitter.phase_end(run_id, Phase::PCC, "skipped", {{"reason", "no_wcs"}},
                        log_file);
      core::emit_event("resume_end", run_id,
                       {{"success", false}, {"status", "no_wcs"}}, log_file);
      return 1;
    }

    io::FitsHeader out_hdr = rgb.header;
    inject_wcs_keywords(out_hdr, wcs);

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
                         {"source", source}},
                        log_file);
      core::emit_event(
          "resume_end", run_id,
          {{"success", false}, {"status", "no_catalog_stars"}}, log_file);
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
          float v = ch->data()[k];
          if (std::isfinite(v)) {
            if (v < vmin)
              vmin = v;
            if (v > vmax)
              vmax = v;
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
    io::write_fits_rgb(run_dir / "outputs" / "stacked_rgb_pcc.fits", R_pcc_disk,
                       G_pcc_disk, B_pcc_disk, out_hdr);

    core::json matrix_json = core::json::array();
    for (int r = 0; r < 3; ++r) {
      matrix_json.push_back(
          {result.matrix[r][0], result.matrix[r][1], result.matrix[r][2]});
    }

    emitter.phase_end(run_id, Phase::PCC, "ok",
                      {{"stars_matched", result.n_stars_matched},
                       {"stars_used", result.n_stars_used},
                       {"residual_rms", result.residual_rms},
                       {"matrix", matrix_json},
                       {"source", used_source}},
                      log_file);
  }

  core::emit_event("resume_end", run_id, {{"success", true}, {"status", "ok"}},
                   log_file);
  return 0;
}
