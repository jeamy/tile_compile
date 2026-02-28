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
    double quality_weight;    // optional tile-quality-based weight
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

    // Local annulus background model
    std::string background_model = "plane"; // median | plane
    double max_condition_number = 2.0; // stability guard (>= 1)
    double max_residual_rms = 0.35;    // robust residual guard

    // Adaptive radii controls (resolved in runner for auto_fwhm)
    std::string radii_mode = "auto_fwhm"; // fixed | auto_fwhm
    double aperture_fwhm_mult = 1.8;
    double annulus_inner_fwhm_mult = 3.0;
    double annulus_outer_fwhm_mult = 5.0;
    double min_aperture_px = 4.0;

    // Optional tile-quality hints for robust star weighting.
    bool use_tile_quality_weighting = false;
    TileGrid tile_grid;
    std::vector<TileMetrics> tile_metrics;
    float tile_quality_kappa = 0.25f;
    float tile_structure_ref = 2.0f;
    float tile_structure_reject = 8.0f;
    float tile_weight_min = 0.25f;
    float tile_weight_max = 2.0f;
};

// PCC result
struct PCCResult {
    ColorMatrix matrix;       // 3x3 color correction matrix
    int    n_stars_matched;   // stars matched in image
    int    n_stars_used;      // stars used after outlier rejection
    double residual_rms;      // RMS of fit residuals
    double determinant;       // determinant of fitted matrix
    double condition_number;  // condition number of fitted matrix
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
