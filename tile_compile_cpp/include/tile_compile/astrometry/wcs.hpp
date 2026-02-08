#pragma once

#include <cmath>
#include <string>

namespace tile_compile::astrometry {

// Simple WCS (World Coordinate System) for TAN projection
// Supports pixel <-> sky (RA/Dec) coordinate conversion
// Parses ASTAP .wcs output files
struct WCS {
    // Reference pixel (1-indexed, FITS convention)
    double crpix1 = 0.0;
    double crpix2 = 0.0;

    // Reference sky coordinates (degrees)
    double crval1 = 0.0;  // RA
    double crval2 = 0.0;  // Dec

    // CD matrix (degrees/pixel) — encodes scale + rotation
    double cd1_1 = 0.0;
    double cd1_2 = 0.0;
    double cd2_1 = 0.0;
    double cd2_2 = 0.0;

    // Image dimensions
    int naxis1 = 0;
    int naxis2 = 0;

    // Convenience: pixel scale (arcsec/pixel) and rotation (degrees)
    double pixel_scale_arcsec() const {
        double s1 = std::sqrt(cd1_1 * cd1_1 + cd2_1 * cd2_1);
        double s2 = std::sqrt(cd1_2 * cd1_2 + cd2_2 * cd2_2);
        return 0.5 * (s1 + s2) * 3600.0;
    }

    double rotation_deg() const {
        return std::atan2(cd2_1, cd1_1) * 180.0 / M_PI;
    }

    double fov_width_deg() const {
        return naxis1 * std::sqrt(cd1_1 * cd1_1 + cd2_1 * cd2_1);
    }

    double fov_height_deg() const {
        return naxis2 * std::sqrt(cd1_2 * cd1_2 + cd2_2 * cd2_2);
    }

    // Pixel (0-indexed) -> sky (RA, Dec in degrees)
    // Uses TAN (gnomonic) projection
    void pixel_to_sky(double px, double py, double &ra_deg, double &dec_deg) const {
        // Convert to 1-indexed for FITS convention
        double dx = (px + 1.0) - crpix1;
        double dy = (py + 1.0) - crpix2;

        // Intermediate world coordinates (degrees)
        double xi  = cd1_1 * dx + cd1_2 * dy;
        double eta = cd2_1 * dx + cd2_2 * dy;

        // Convert to radians
        constexpr double D2R = M_PI / 180.0;
        double xi_r  = xi * D2R;
        double eta_r = eta * D2R;
        double ra0_r  = crval1 * D2R;
        double dec0_r = crval2 * D2R;

        // TAN (gnomonic) deprojection
        double sin_dec0 = std::sin(dec0_r);
        double cos_dec0 = std::cos(dec0_r);
        double denom = cos_dec0 - eta_r * sin_dec0;

        ra_deg  = std::atan2(xi_r, denom) / D2R + crval1;
        dec_deg = std::atan2((sin_dec0 + eta_r * cos_dec0) * std::cos(ra_deg * D2R - ra0_r), denom) / D2R;

        // Normalize RA to [0, 360)
        while (ra_deg < 0.0) ra_deg += 360.0;
        while (ra_deg >= 360.0) ra_deg -= 360.0;
    }

    // Sky (RA, Dec in degrees) -> pixel (0-indexed)
    // Returns false if point is behind the projection
    bool sky_to_pixel(double ra_deg, double dec_deg, double &px, double &py) const {
        constexpr double D2R = M_PI / 180.0;
        double ra_r   = ra_deg * D2R;
        double dec_r  = dec_deg * D2R;
        double ra0_r  = crval1 * D2R;
        double dec0_r = crval2 * D2R;

        double sin_dec  = std::sin(dec_r);
        double cos_dec  = std::cos(dec_r);
        double sin_dec0 = std::sin(dec0_r);
        double cos_dec0 = std::cos(dec0_r);
        double delta_ra = ra_r - ra0_r;
        double cos_dra  = std::cos(delta_ra);

        // TAN projection
        double denom = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_dra;
        if (denom <= 0.0) return false;  // behind projection

        double xi_r  = (cos_dec * std::sin(delta_ra)) / denom;
        double eta_r = (sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_dra) / denom;

        // Convert from radians to degrees
        double xi  = xi_r / D2R;
        double eta = eta_r / D2R;

        // Invert CD matrix: [dx, dy] = CD^-1 * [xi, eta]
        double det = cd1_1 * cd2_2 - cd1_2 * cd2_1;
        if (std::abs(det) < 1e-30) return false;

        double dx = ( cd2_2 * xi - cd1_2 * eta) / det;
        double dy = (-cd2_1 * xi + cd1_1 * eta) / det;

        // Convert back to 0-indexed pixel
        px = dx + crpix1 - 1.0;
        py = dy + crpix2 - 1.0;
        return true;
    }

    // Check if a sky coordinate falls within the image
    bool contains(double ra_deg, double dec_deg) const {
        double px, py;
        if (!sky_to_pixel(ra_deg, dec_deg, px, py)) return false;
        return px >= 0 && px < naxis1 && py >= 0 && py < naxis2;
    }

    // Search radius (degrees) that covers the entire image diagonal
    double search_radius_deg() const {
        double diag_px = std::sqrt(double(naxis1) * naxis1 + double(naxis2) * naxis2);
        return 0.6 * diag_px * pixel_scale_arcsec() / 3600.0;  // slight margin
    }

    bool valid() const {
        return naxis1 > 0 && naxis2 > 0 &&
               (std::abs(cd1_1) > 0 || std::abs(cd1_2) > 0);
    }
};

// Parse an ASTAP .wcs file (FITS-like keyword=value format)
WCS parse_wcs_file(const std::string &path);

// Build WCS from CDELT+CROTA (older ASTAP format) if CD matrix not present
WCS wcs_from_cdelt_crota(double crval1, double crval2,
                         double crpix1, double crpix2,
                         double cdelt1, double cdelt2,
                         double crota2, int naxis1, int naxis2);

} // namespace tile_compile::astrometry
