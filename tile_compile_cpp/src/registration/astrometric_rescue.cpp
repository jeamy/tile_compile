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

// Shell-Quote Helper (kopiert aus runner_pipeline.cpp)
static std::string shell_quote(const std::string& s) {
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
}

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
  int ret_mov = std::system(cmd_mov.c_str());

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
  int ret_ref = std::system(cmd_ref.c_str());

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

WarpMatrix wcs_to_similarity_warp(
    const astro::WCS& wcs_mov,
    const astro::WCS& wcs_ref,
    float scale_factor) {

  // CD-Matrix zu Similarity-Parameter
  // CD = [cd1_1 cd1_2; cd2_1 cd2_2] enthält Scale + Rotation

  // Extrahiere Rotation und Scale aus CD-Matrix (moving frame)
  double theta_mov = std::atan2(wcs_mov.cd2_1, wcs_mov.cd1_1);
  double scale_mov = std::sqrt(wcs_mov.cd1_1 * wcs_mov.cd1_1 +
                                wcs_mov.cd2_1 * wcs_mov.cd2_1);

  // Extrahiere Rotation und Scale aus CD-Matrix (reference frame)
  double theta_ref = std::atan2(wcs_ref.cd2_1, wcs_ref.cd1_1);
  double scale_ref = std::sqrt(wcs_ref.cd1_1 * wcs_ref.cd1_1 +
                                wcs_ref.cd2_1 * wcs_ref.cd2_1);

  // Relative Transformation
  double dtheta = theta_mov - theta_ref;
  double dscale = scale_mov / scale_ref;

  // Konvertiere Referenz-CRPIX zu Pixel-Koordinaten
  // Wir müssen das Moving-Frame so transformieren, dass sein CRPIX
  // auf den Referenz-CRPIX abgebildet wird

  WarpMatrix w = WarpMatrix::Identity();

  // Rotation
  float ct = std::cos(dtheta);
  float st = std::sin(dtheta);
  w(0, 0) = ct * dscale * scale_factor;
  w(0, 1) = -st * dscale * scale_factor;
  w(1, 0) = st * dscale * scale_factor;
  w(1, 1) = ct * dscale * scale_factor;

  // Translation: CRPIX对齐
  // Ziel: mov.crpix sollte auf ref.crpix abgebildet werden
  double dx = wcs_ref.crpix1 - wcs_mov.crpix1;
  double dy = wcs_ref.crpix2 - wcs_mov.crpix2;

  w(0, 2) = dx;
  w(1, 2) = dy;

  return w;
}

} // namespace tile_compile::registration
