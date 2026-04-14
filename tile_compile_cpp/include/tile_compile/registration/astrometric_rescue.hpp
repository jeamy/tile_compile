#pragma once

#include "tile_compile/registration/registration.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include <string>

namespace tile_compile::registration {

// §4.13 — Astrometrische Rescue für nicht-registrierte Frames
// Nutzt ASTAP Plate-Solving als letzter Rettungsversuch

struct AstrometricRescueResult {
  bool success = false;
  WarpMatrix warp = identity_warp();
  float correlation = 0.0f;  // NCC gegen Referenz
  astrometry::WCS wcs;      // Gefundene WCS-Lösung
  std::string error_message;
};

// Prüft ob ASTAP verfügbar ist (Binary + Katalog)
bool is_astap_available(const std::string& astap_bin_hint = "",
                        const std::string& astap_data_dir_hint = "");

// Führt astrometrisches Plate-Solving für ein einzelnes Frame durch
// und validiert das Ergebnis per NCC gegen den Referenzframe.
// mov / ref sind die (ggf. downsampled) Proxy-Bilder für NCC-Validierung.
AstrometricRescueResult try_astrometric_rescue(
    const Matrix2Df& mov,
    const Matrix2Df& ref,
    const std::string& astap_bin,
    const std::string& astap_data_dir,
    float search_radius_deg = 180.0f,
    float ncc_threshold = 0.20f
);

// Überladung mit Original-Frame-Pfaden: übergibt die echten FITS-Dateien
// direkt an ASTAP (bessere Sternzahl, korrekte Header). Der Warp wird
// anschließend auf Proxy-Auflösung (scale_factor) skaliert.
AstrometricRescueResult try_astrometric_rescue_from_paths(
    const std::string& mov_fits_path,
    const std::string& ref_fits_path,
    const Matrix2Df& mov_proxy,
    const Matrix2Df& ref_proxy,
    const std::string& astap_bin,
    const std::string& astap_data_dir,
    float scale_factor,          // full_res / proxy_res
    float search_radius_deg = 180.0f,
    float ncc_threshold = 0.20f
);

// Konvertiert WCS (CD-Matrix) zu Similarity-Warp
// Skaliert vom gelösten Frame auf Referenzframe
WarpMatrix wcs_to_similarity_warp(
    const astrometry::WCS& wcs_mov,
    const astrometry::WCS& wcs_ref,
    float scale_factor = 1.0f
);

} // namespace tile_compile::registration
