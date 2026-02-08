#include "tile_compile/astrometry/photometric_color_cal.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

namespace tile_compile::astrometry {

// ─── Default OSC filter curves (generic CMOS Bayer) ─────────────────────

FilterCurves default_osc_filter_curves() {
    FilterCurves fc;
    // Wavelength range: 380-720 nm in 5nm steps
    for (double wl = 380.0; wl <= 720.0; wl += 5.0) {
        fc.wl.push_back(wl);

        // Approximate Bayer filter transmissions (Gaussian-like)
        // Blue: peak ~470nm, FWHM ~80nm
        double b = std::exp(-0.5 * std::pow((wl - 470.0) / 40.0, 2));
        // Green: peak ~530nm, FWHM ~90nm
        double g = std::exp(-0.5 * std::pow((wl - 530.0) / 45.0, 2));
        // Red: peak ~620nm, FWHM ~80nm, with IR tail
        double r = std::exp(-0.5 * std::pow((wl - 620.0) / 40.0, 2));
        if (wl > 650.0) r = std::max(r, 0.3 * std::exp(-0.5 * std::pow((wl - 680.0) / 30.0, 2)));

        fc.tx_r.push_back(r);
        fc.tx_g.push_back(g);
        fc.tx_b.push_back(b);
    }
    return fc;
}

// ─── Aperture photometry ────────────────────────────────────────────────

static double aperture_flux(const Matrix2Df &img, double cx, double cy,
                            double r_ap, double r_ann_in, double r_ann_out) {
    int rows = img.rows();
    int cols = img.cols();

    int x0 = std::max(0, static_cast<int>(cx - r_ann_out - 1));
    int x1 = std::min(cols - 1, static_cast<int>(cx + r_ann_out + 1));
    int y0 = std::max(0, static_cast<int>(cy - r_ann_out - 1));
    int y1 = std::min(rows - 1, static_cast<int>(cy + r_ann_out + 1));

    double r_ap2 = r_ap * r_ap;
    double r_in2 = r_ann_in * r_ann_in;
    double r_out2 = r_ann_out * r_ann_out;

    // Collect sky annulus pixels for background estimation
    std::vector<float> sky_pixels;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            double dx = x - cx;
            double dy = y - cy;
            double d2 = dx * dx + dy * dy;
            if (d2 >= r_in2 && d2 <= r_out2) {
                sky_pixels.push_back(img(y, x));
            }
        }
    }

    if (sky_pixels.size() < 10) return -1.0;  // not enough sky pixels

    // Median sky background
    std::sort(sky_pixels.begin(), sky_pixels.end());
    double sky_bg = sky_pixels[sky_pixels.size() / 2];

    // Sum aperture flux minus background
    double total = 0.0;
    int n_ap = 0;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            double dx = x - cx;
            double dy = y - cy;
            if (dx * dx + dy * dy <= r_ap2) {
                total += img(y, x) - sky_bg;
                ++n_ap;
            }
        }
    }

    return (n_ap > 0) ? total : -1.0;
}

std::vector<StarPhotometry> measure_stars(
    const Matrix2Df &R, const Matrix2Df &G, const Matrix2Df &B,
    const WCS &wcs,
    const std::vector<GaiaStar> &catalog_stars,
    const PCCConfig &config) {

    std::vector<StarPhotometry> result;
    result.reserve(catalog_stars.size());

    // Get default filter curves for synthetic catalog fluxes
    FilterCurves fc = default_osc_filter_curves();

    int rows = R.rows();
    int cols = R.cols();
    double margin = config.annulus_outer_px + 2.0;

    for (const auto &star : catalog_stars) {
        // Magnitude filter
        if (star.mag > config.mag_limit || star.mag < config.mag_bright_limit) continue;

        // Project to pixel
        double px, py;
        if (!wcs.sky_to_pixel(star.ra, star.dec, px, py)) continue;

        // Check within image bounds with margin
        if (px < margin || px >= cols - margin ||
            py < margin || py >= rows - margin) continue;

        StarPhotometry sp;
        sp.ra = star.ra;
        sp.dec = star.dec;
        sp.px = px;
        sp.py = py;
        sp.mag = star.mag;

        // Measure instrumental flux in each channel
        sp.flux_r = aperture_flux(R, px, py, config.aperture_radius_px,
                                  config.annulus_inner_px, config.annulus_outer_px);
        sp.flux_g = aperture_flux(G, px, py, config.aperture_radius_px,
                                  config.annulus_inner_px, config.annulus_outer_px);
        sp.flux_b = aperture_flux(B, px, py, config.aperture_radius_px,
                                  config.annulus_inner_px, config.annulus_outer_px);

        // Compute synthetic catalog flux from XP spectrum
        if (!star.xp_flux.empty()) {
            sp.cat_r = synthetic_flux(star.xp_flux, fc.wl, fc.tx_r);
            sp.cat_g = synthetic_flux(star.xp_flux, fc.wl, fc.tx_g);
            sp.cat_b = synthetic_flux(star.xp_flux, fc.wl, fc.tx_b);
        } else {
            sp.cat_r = sp.cat_g = sp.cat_b = 0.0;
        }

        // Validate: all fluxes must be positive
        sp.valid = (sp.flux_r > 0 && sp.flux_g > 0 && sp.flux_b > 0 &&
                    sp.cat_r > 0 && sp.cat_g > 0 && sp.cat_b > 0);

        result.push_back(sp);
    }

    return result;
}

// ─── Color matrix fitting ───────────────────────────────────────────────
//
// PCC computes per-channel scale factors so that the instrumental color
// ratios (R/G, B/G) match the catalog color ratios.  This is a diagonal
// correction — no channel mixing — which preserves the image's color
// structure while adjusting the white balance to a photometric reference.
//
// For each star:  ratio_inst_r = flux_r / flux_g
//                 ratio_cat_r  = cat_r  / cat_g
//                 correction_r = ratio_cat_r / ratio_inst_r
// The per-channel scale is the sigma-clipped median of these corrections.
// Green is the reference channel (scale_g = 1.0).

PCCResult fit_color_matrix(const std::vector<StarPhotometry> &stars,
                           const PCCConfig &config) {
    PCCResult res;
    res.success = false;
    res.matrix = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};  // identity default

    // Collect valid stars
    std::vector<const StarPhotometry*> valid;
    for (const auto &s : stars) {
        if (s.valid) valid.push_back(&s);
    }

    res.n_stars_matched = static_cast<int>(valid.size());

    if (static_cast<int>(valid.size()) < config.min_stars) {
        res.error_message = "Not enough valid stars (" +
                            std::to_string(valid.size()) + " < " +
                            std::to_string(config.min_stars) + ")";
        return res;
    }

    // For each star, compute the per-channel correction factor
    // correction_c = (cat_c / cat_g) / (inst_c / inst_g)
    //              = (cat_c * inst_g) / (cat_g * inst_c)
    std::vector<double> corr_r, corr_b;
    corr_r.reserve(valid.size());
    corr_b.reserve(valid.size());

    for (const auto *s : valid) {
        double cr = (s->cat_r * s->flux_g) / (s->cat_g * s->flux_r);
        double cb = (s->cat_b * s->flux_g) / (s->cat_g * s->flux_b);
        // Sanity: reject extreme corrections
        if (cr > 0.1 && cr < 10.0) corr_r.push_back(cr);
        if (cb > 0.1 && cb < 10.0) corr_b.push_back(cb);
    }

    if (static_cast<int>(corr_r.size()) < config.min_stars ||
        static_cast<int>(corr_b.size()) < config.min_stars) {
        res.error_message = "Not enough stars with valid color ratios";
        return res;
    }

    // Iterative sigma-clipped median
    auto sigma_clipped_median = [&](std::vector<double> &vals, double sigma) -> double {
        for (int iter = 0; iter < 5; ++iter) {
            std::sort(vals.begin(), vals.end());
            double med = vals[vals.size() / 2];
            double mad = 0;
            for (double v : vals) mad += std::abs(v - med);
            mad /= vals.size();
            double threshold = sigma * mad * 1.4826;  // MAD to sigma
            if (threshold < 1e-10) break;
            std::vector<double> kept;
            for (double v : vals) {
                if (std::abs(v - med) <= threshold) kept.push_back(v);
            }
            if (static_cast<int>(kept.size()) < config.min_stars) break;
            vals = kept;
        }
        std::sort(vals.begin(), vals.end());
        return vals[vals.size() / 2];
    };

    double scale_r = sigma_clipped_median(corr_r, config.sigma_clip);
    double scale_b = sigma_clipped_median(corr_b, config.sigma_clip);
    double scale_g = 1.0;  // Green is reference

    // Build diagonal color matrix
    res.matrix = {{{scale_r, 0, 0}, {0, scale_g, 0}, {0, 0, scale_b}}};
    res.n_stars_used = static_cast<int>(std::min(corr_r.size(), corr_b.size()));
    res.residual_rms = 0.0;

    // Compute RMS of color ratio residuals
    double sum_sq = 0;
    int count = 0;
    for (const auto *s : valid) {
        double cr = (s->cat_r * s->flux_g) / (s->cat_g * s->flux_r);
        double cb = (s->cat_b * s->flux_g) / (s->cat_g * s->flux_b);
        if (cr > 0.1 && cr < 10.0 && cb > 0.1 && cb < 10.0) {
            sum_sq += (cr - scale_r) * (cr - scale_r) + (cb - scale_b) * (cb - scale_b);
            ++count;
        }
    }
    res.residual_rms = (count > 0) ? std::sqrt(sum_sq / count) : 0.0;

    res.success = true;
    std::cerr << "[PCC] Scale factors: R=" << scale_r
              << " G=" << scale_g << " B=" << scale_b
              << " (" << res.n_stars_used << " stars)" << std::endl;
    return res;
}

// ─── Apply color matrix ─────────────────────────────────────────────────

void apply_color_matrix(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
                        const ColorMatrix &m) {
    int rows = R.rows();
    int cols = R.cols();

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float r = R(y, x);
            float g = G(y, x);
            float b = B(y, x);

            R(y, x) = static_cast<float>(m[0][0] * r + m[0][1] * g + m[0][2] * b);
            G(y, x) = static_cast<float>(m[1][0] * r + m[1][1] * g + m[1][2] * b);
            B(y, x) = static_cast<float>(m[2][0] * r + m[2][1] * g + m[2][2] * b);
        }
    }
}

// ─── Full PCC pipeline ──────────────────────────────────────────────────

PCCResult run_pcc(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
                  const WCS &wcs,
                  const std::vector<GaiaStar> &catalog_stars,
                  const PCCConfig &config) {

    std::cerr << "[PCC] Measuring " << catalog_stars.size()
              << " catalog stars in image..." << std::endl;

    auto photometry = measure_stars(R, G, B, wcs, catalog_stars, config);

    int n_valid = 0;
    for (const auto &s : photometry) if (s.valid) ++n_valid;
    std::cerr << "[PCC] " << n_valid << "/" << photometry.size()
              << " stars with valid photometry" << std::endl;

    auto result = fit_color_matrix(photometry, config);

    if (result.success) {
        std::cerr << "[PCC] Color matrix fit: " << result.n_stars_used
                  << " stars, RMS=" << result.residual_rms << std::endl;
        std::cerr << "[PCC] Matrix:" << std::endl;
        for (int i = 0; i < 3; ++i) {
            std::cerr << "  [" << result.matrix[i][0] << ", "
                      << result.matrix[i][1] << ", "
                      << result.matrix[i][2] << "]" << std::endl;
        }

        apply_color_matrix(R, G, B, result.matrix);
        std::cerr << "[PCC] Color correction applied." << std::endl;
    } else {
        std::cerr << "[PCC] Failed: " << result.error_message << std::endl;
    }

    return result;
}

} // namespace tile_compile::astrometry
