#pragma once

#include "tile_compile/astrometry/gaia_catalog.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/core/types.hpp"

#include <array>
#include <string>
#include <vector>

namespace tile_compile::astrometry {

// 3x3 color correction matrix
using ColorMatrix = std::array<std::array<double, 3>, 3>;

// Result of aperture photometry for one star
struct StarPhotometry {
    double ra, dec;           // sky position
    double px, py;            // pixel position
    double flux_r, flux_g, flux_b;  // instrumental flux
    double cat_r, cat_g, cat_b;     // catalog synthetic flux
    float  mag;               // catalog magnitude
    bool   valid;             // true if measurement is usable
};

// PCC configuration
struct PCCConfig {
    double aperture_radius_px = 8.0;   // aperture radius in pixels
    double annulus_inner_px   = 12.0;  // sky annulus inner radius
    double annulus_outer_px   = 18.0;  // sky annulus outer radius
    double mag_limit          = 14.0;  // faintest catalog star to use
    double mag_bright_limit   = 6.0;   // brightest (avoid saturation)
    int    min_stars          = 10;    // minimum stars for reliable fit
    double sigma_clip         = 2.5;   // sigma clipping for outlier rejection
};

// PCC result
struct PCCResult {
    ColorMatrix matrix;       // 3x3 color correction matrix
    int    n_stars_matched;   // stars matched in image
    int    n_stars_used;      // stars used after outlier rejection
    double residual_rms;      // RMS of fit residuals
    bool   success;
    std::string error_message;
};

// Perform aperture photometry on catalog stars in the stacked RGB image
std::vector<StarPhotometry> measure_stars(
    const Matrix2Df &R, const Matrix2Df &G, const Matrix2Df &B,
    const WCS &wcs,
    const std::vector<GaiaStar> &catalog_stars,
    const PCCConfig &config = PCCConfig());

// Fit a 3x3 color correction matrix from star measurements
PCCResult fit_color_matrix(const std::vector<StarPhotometry> &stars,
                           const PCCConfig &config = PCCConfig());

// Apply the color correction matrix to RGB channels (in-place)
void apply_color_matrix(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
                        const ColorMatrix &matrix);

// Full PCC pipeline: catalog query + photometry + matrix fit + apply
// Returns the result; R/G/B are modified in-place if successful
PCCResult run_pcc(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
                  const WCS &wcs,
                  const std::vector<GaiaStar> &catalog_stars,
                  const PCCConfig &config = PCCConfig());

// Default Bayer filter transmission curves for a typical CMOS sensor
// Returns (wavelength_nm, transmission) pairs
struct FilterCurves {
    std::vector<double> wl;   // wavelength in nm
    std::vector<double> tx_r; // red filter transmission
    std::vector<double> tx_g; // green filter transmission
    std::vector<double> tx_b; // blue filter transmission
};

// Get default filter curves for a generic OSC sensor
FilterCurves default_osc_filter_curves();

} // namespace tile_compile::astrometry
