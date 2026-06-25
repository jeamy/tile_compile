#include "tile_compile/registration/astrometric_rescue.hpp"
#include "tile_compile/registration/global_registration.hpp"
#include "tile_compile/registration/registration.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/core/utils.hpp"

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace tile_compile::registration {

namespace fs = std::filesystem;
namespace core = tile_compile::core;
namespace astro = tile_compile::astrometry;

/// @brief Checks astap available.
/// @details Part of ASTAP-backed astrometric registration rescue helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool is_astap_available(const std::string& astap_bin_hint,
                        const std::string& astap_data_dir_hint) {
  // ASTAP-Pfad bestimmen
  std::string astap_data = astap_data_dir_hint;
  if (astap_data.empty()) {
    const char* home = std::getenv("HOME");
    if (home) {
      astap_data = std::string(home) + "/.local/share/tile_compile/astap";
    }
  }

  std::string astap_bin = astap_bin_hint;
  if (astap_bin.empty()) {
    astap_bin = astap_data + "/astap_cli";
  }

  // Prüfen ob Binary existiert
  if (!fs::exists(astap_bin)) {
    return false;
  }

  // Prüfen ob Katalog-Daten existieren (mindestens eine .290 Datei)
  bool has_catalog = false;
  if (fs::exists(astap_data) && fs::is_directory(astap_data)) {
    for (const auto& entry : fs::directory_iterator(astap_data)) {
      if (entry.path().extension() == ".290" ||
          entry.path().extension() == ".291") {
        has_catalog = true;
        break;
      }
    }
  }

  return has_catalog;
}

// Wrap command for Windows cmd.exe (cmd /c "...")
static std::string system_cmd(const std::string& cmd) {
#ifdef _WIN32
  return "cmd /c \"" + cmd + "\"";
#else
  return cmd;
#endif
}

// Shell-Quote Helper (kopiert aus runner_pipeline.cpp)
/// @brief Implements shell quote.
/// @details Part of ASTAP-backed astrometric registration rescue helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static std::string shell_quote(const std::string& s) {
#ifdef _WIN32
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('"');
  for (char c : s) {
    if (c == '"') out += "\\"";
    else out.push_back(c);
  }
  out.push_back('"');
  return out;
#else
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('\'');
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out.push_back('\'');
  return out;
#endif
}

/// @brief Implements try astrometric rescue.
/// @details Part of ASTAP-backed astrometric registration rescue helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
AstrometricRescueResult try_astrometric_rescue(
    const Matrix2Df& mov,
    const Matrix2Df& ref,
    const std::string& astap_bin_in,
    const std::string& astap_data_dir_in,
    float search_radius_deg,
    float ncc_threshold) {

  AstrometricRescueResult result;
  result.success = false;
  result.correlation = 0.0f;

  // ASTAP-Pfade auflösen
  std::string astap_data = astap_data_dir_in;
  if (astap_data.empty()) {
    const char* home = std::getenv("HOME");
    if (home) {
      astap_data = std::string(home) + "/.local/share/tile_compile/astap";
    }
  }

  std::string astap_bin = astap_bin_in;
  if (astap_bin.empty()) {
    astap_bin = astap_data + "/astap_cli";
  }

  // Verfügbarkeit prüfen
  if (!fs::exists(astap_bin)) {
    result.error_message = "astap_not_found";
    return result;
  }

  // Temporäre FITS-Dateien erstellen
  fs::path temp_dir = fs::temp_directory_path();
  std::string timestamp = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());

  fs::path mov_path = temp_dir / ("astrometric_rescue_mov_" + timestamp + ".fits");
  fs::path ref_path = temp_dir / ("astrometric_rescue_ref_" + timestamp + ".fits");
  fs::path mov_wcs_path = mov_path;
  mov_wcs_path.replace_extension(".wcs");
  fs::path ref_wcs_path = ref_path;
  ref_wcs_path.replace_extension(".wcs");

  // Aufräumen bei Exit
  struct TempCleaner {
    std::vector<fs::path> paths;
/// @brief Implements ~TempCleaner.
/// @details Part of ASTAP-backed astrometric registration rescue helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
    ~TempCleaner() {
      for (const auto& p : paths) {
        if (fs::exists(p)) {
          fs::remove(p);
        }
      }
    }
  } cleaner;

  // FITS-Dateien schreiben
  try {
    // Einfache FITS-Header für die Frames
    io::FitsHeader mov_hdr;
    mov_hdr.int_values["NAXIS1"] = mov.cols();
    mov_hdr.int_values["NAXIS2"] = mov.rows();
    mov_hdr.int_values["BITPIX"] = -32;  // FLOAT
    io::write_fits_float(mov_path, mov, mov_hdr);
    cleaner.paths.push_back(mov_path);

    io::FitsHeader ref_hdr;
    ref_hdr.int_values["NAXIS1"] = ref.cols();
    ref_hdr.int_values["NAXIS2"] = ref.rows();
    ref_hdr.int_values["BITPIX"] = -32;
    io::write_fits_float(ref_path, ref, ref_hdr);
    cleaner.paths.push_back(ref_path);
    cleaner.paths.push_back(mov_wcs_path);
    cleaner.paths.push_back(ref_wcs_path);
  } catch (const std::exception& e) {
    result.error_message = "fits_write_failed: " + std::string(e.what());
    return result;
  }

  // ASTAP für Moving-Frame aufrufen
  std::string cmd_mov = shell_quote(astap_bin) + " -f " +
      shell_quote(mov_path.string()) +
      " -d " + shell_quote(astap_data) +
      " -r " + std::to_string(search_radius_deg);

  std::cout << "[ASTROMETRIC-RESCUE] Running ASTAP for moving frame: " << cmd_mov << std::endl;
  int ret_mov = std::system(system_cmd(cmd_mov).c_str());

  // WCS parsen
  astro::WCS wcs_mov;
  bool have_wcs_mov = false;
  if (ret_mov == 0 && fs::exists(mov_wcs_path)) {
    try {
      wcs_mov = astro::parse_wcs_file(mov_wcs_path.string());
      have_wcs_mov = wcs_mov.valid();
    } catch (...) {
      have_wcs_mov = false;
    }
  }

  if (!have_wcs_mov) {
    result.error_message = "plate_solve_failed";
    return result;
  }

  // ASTAP für Referenz-Frame aufrufen
  std::string cmd_ref = shell_quote(astap_bin) + " -f " +
      shell_quote(ref_path.string()) +
      " -d " + shell_quote(astap_data) +
      " -r " + std::to_string(search_radius_deg);

  std::cout << "[ASTROMETRIC-RESCUE] Running ASTAP for reference frame: " << cmd_ref << std::endl;
  int ret_ref = std::system(system_cmd(cmd_ref).c_str());

  astro::WCS wcs_ref;
  bool have_wcs_ref = false;
  if (ret_ref == 0 && fs::exists(ref_wcs_path)) {
    try {
      wcs_ref = astro::parse_wcs_file(ref_wcs_path.string());
      have_wcs_ref = wcs_ref.valid();
    } catch (...) {
      have_wcs_ref = false;
    }
  }

  if (!have_wcs_ref) {
    result.error_message = "ref_plate_solve_failed";
    return result;
  }

  // WCS zu Warp konvertieren
  result.wcs = wcs_mov;
  result.warp = wcs_to_similarity_warp(wcs_mov, wcs_ref, 1.0f);

  // Mit NCC validieren
  Matrix2Df warped = apply_warp(mov, result.warp);
  cv::Mat valid_mask = warp_valid_mask(mov, result.warp);
  result.correlation = compute_ncc_masked(warped, ref, valid_mask);

  std::cout << "[ASTROMETRIC-RESCUE] NCC validation: " << result.correlation << std::endl;

  if (result.correlation >= ncc_threshold) {
    result.success = true;
    std::cout << "[ASTROMETRIC-RESCUE] SUCCESS - frame registered via astrometry" << std::endl;
  } else {
    result.error_message = "ncc_too_low";
    result.success = false;
  }

  return result;
}

/// @brief Implements try astrometric rescue from paths.
/// @details Part of ASTAP-backed astrometric registration rescue helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
AstrometricRescueResult try_astrometric_rescue_from_paths(
    const std::string& mov_fits_path,
    const std::string& ref_fits_path,
    const Matrix2Df& mov_proxy,
    const Matrix2Df& ref_proxy,
    const std::string& astap_bin_in,
    const std::string& astap_data_dir_in,
    float scale_factor,
    float search_radius_deg,
    float ncc_threshold) {

  AstrometricRescueResult result;
  result.success = false;
  result.correlation = 0.0f;

  std::string astap_data = astap_data_dir_in;
  if (astap_data.empty()) {
    const char* home = std::getenv("HOME");
    if (home) astap_data = std::string(home) + "/.local/share/tile_compile/astap";
  }
  std::string astap_bin = astap_bin_in;
  if (astap_bin.empty()) astap_bin = astap_data + "/astap_cli";

  if (!fs::exists(astap_bin)) {
    result.error_message = "astap_not_found";
    return result;
  }
  if (!fs::exists(mov_fits_path) || !fs::exists(ref_fits_path)) {
    result.error_message = "fits_path_not_found";
    return result;
  }

  // ASTAP derives the .wcs output next to the input file
  fs::path mov_wcs_path = fs::path(mov_fits_path);
  mov_wcs_path.replace_extension(".wcs");
  fs::path ref_wcs_path = fs::path(ref_fits_path);
  ref_wcs_path.replace_extension(".wcs");

  struct TempCleaner {
    std::vector<fs::path> paths;
/// @brief Implements ~TempCleaner.
/// @details Part of ASTAP-backed astrometric registration rescue helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
    ~TempCleaner() {
      for (const auto& p : paths) {
        if (fs::exists(p)) fs::remove(p);
      }
    }
  } cleaner;
  cleaner.paths.push_back(mov_wcs_path);
  cleaner.paths.push_back(ref_wcs_path);

  // Solve moving frame
  std::string cmd_mov = shell_quote(astap_bin) + " -f " +
      shell_quote(mov_fits_path) +
      " -d " + shell_quote(astap_data) +
      " -r " + std::to_string(search_radius_deg);
  int ret_mov = std::system(system_cmd(cmd_mov).c_str());

  astro::WCS wcs_mov;
  bool have_wcs_mov = false;
  if (ret_mov == 0 && fs::exists(mov_wcs_path)) {
    try {
      wcs_mov = astro::parse_wcs_file(mov_wcs_path.string());
      have_wcs_mov = wcs_mov.valid();
    } catch (...) {
      std::cerr << "[ASTROMETRIC_RESCUE] Warning: Failed to parse moving WCS file: " << mov_wcs_path << std::endl;
    }
  }
  if (!have_wcs_mov) {
    result.error_message = "plate_solve_failed";
    return result;
  }

  // Solve reference frame
  std::string cmd_ref = shell_quote(astap_bin) + " -f " +
      shell_quote(ref_fits_path) +
      " -d " + shell_quote(astap_data) +
      " -r " + std::to_string(search_radius_deg);
  int ret_ref = std::system(system_cmd(cmd_ref).c_str());

  astro::WCS wcs_ref;
  bool have_wcs_ref = false;
  if (ret_ref == 0 && fs::exists(ref_wcs_path)) {
    try {
      wcs_ref = astro::parse_wcs_file(ref_wcs_path.string());
      have_wcs_ref = wcs_ref.valid();
    } catch (...) {
      std::cerr << "[ASTROMETRIC_RESCUE] Warning: Failed to parse reference WCS file: " << ref_wcs_path << std::endl;
    }
  }
  if (!have_wcs_ref) {
    result.error_message = "ref_plate_solve_failed";
    return result;
  }

  // Build warp at proxy scale (full-res WCS -> proxy-res warp)
  result.wcs = wcs_mov;
  result.warp = wcs_to_similarity_warp(wcs_mov, wcs_ref, 1.0f / scale_factor);

  // Validate with NCC on proxy images
  Matrix2Df warped = apply_warp(mov_proxy, result.warp);
  cv::Mat valid_mask = warp_valid_mask(mov_proxy, result.warp);
  result.correlation = compute_ncc_masked(warped, ref_proxy, valid_mask);

  if (result.correlation >= ncc_threshold) {
    result.success = true;
  } else {
    result.error_message = "ncc_too_low";
  }

  return result;
}

/// @brief Implements wcs to similarity warp.
/// @details Part of ASTAP-backed astrometric registration rescue helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
WarpMatrix wcs_to_similarity_warp(
    const astro::WCS& wcs_mov,
    const astro::WCS& wcs_ref,
    float scale_factor) {

  // Build a similarity warp (inverse map: ref pixel -> mov pixel) from two
  // WCS solutions.  The approach:
  //   1. Derive relative rotation and scale from the CD matrices.
  //   2. Compute the translation by picking N reference-frame pixel positions,
  //      projecting them through wcs_ref to sky coords, then back through
  //      wcs_mov to moving-frame pixel coords.  The median delta is the
  //      translation component.
  // This is correct even when CRPIX values differ arbitrarily between frames.

  // ---- Rotation + scale from CD matrices ----
  double theta_mov = std::atan2(wcs_mov.cd2_1, wcs_mov.cd1_1);
  double scale_mov = std::sqrt(wcs_mov.cd1_1 * wcs_mov.cd1_1 +
                               wcs_mov.cd2_1 * wcs_mov.cd2_1);

  double theta_ref = std::atan2(wcs_ref.cd2_1, wcs_ref.cd1_1);
  double scale_ref = std::sqrt(wcs_ref.cd1_1 * wcs_ref.cd1_1 +
                               wcs_ref.cd2_1 * wcs_ref.cd2_1);

  if (scale_ref < 1e-20) {
    return WarpMatrix::Identity();  // degenerate WCS
  }

  double dtheta = theta_mov - theta_ref;
  double dscale = (scale_mov / scale_ref) * static_cast<double>(scale_factor);

  float ct = static_cast<float>(std::cos(dtheta) * dscale);
  float st = static_cast<float>(std::sin(dtheta) * dscale);

  // ---- Translation via sky-projection of reference image corners ----
  // Sample several points in reference pixel space, project to sky via
  // wcs_ref, then back to pixel space via wcs_mov.  Average the offset.
  const int W = (wcs_ref.naxis1 > 0) ? wcs_ref.naxis1 : 512;
  const int H = (wcs_ref.naxis2 > 0) ? wcs_ref.naxis2 : 512;

  // 5-point sample: centre + 4 midpoints of image quadrants
  const double sample_px[5] = { W * 0.5, W * 0.25, W * 0.75, W * 0.25, W * 0.75 };
  const double sample_py[5] = { H * 0.5, H * 0.25, H * 0.25, H * 0.75, H * 0.75 };

  double sum_dx = 0.0, sum_dy = 0.0;
  int valid = 0;
  for (int s = 0; s < 5; ++s) {
    double ra, dec;
    wcs_ref.pixel_to_sky(sample_px[s], sample_py[s], ra, dec);

    double mov_px, mov_py;
    if (!wcs_mov.sky_to_pixel(ra, dec, mov_px, mov_py)) {
      continue;
    }

    // The warp is inverse (ref -> mov), so the translation is mov - ref
    sum_dx += mov_px - sample_px[s];
    sum_dy += mov_py - sample_py[s];
    ++valid;
  }

  if (valid == 0) {
    return WarpMatrix::Identity();  // projections failed
  }

  float tx = static_cast<float>(sum_dx / valid);
  float ty = static_cast<float>(sum_dy / valid);

  WarpMatrix w = WarpMatrix::Identity();
  w(0, 0) =  ct;
  w(0, 1) = -st;
  w(1, 0) =  st;
  w(1, 1) =  ct;
  w(0, 2) =  tx;
  w(1, 2) =  ty;

  return w;
}

} // namespace tile_compile::registration
