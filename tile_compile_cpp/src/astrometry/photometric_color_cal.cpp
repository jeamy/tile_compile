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

    // Normalize instrumental and catalog fluxes to green channel
    // This makes the fit scale-invariant
    auto normalize = [&valid](std::vector<double> &r, std::vector<double> &g,
                              std::vector<double> &b, bool catalog) {
        double sum_g = 0;
        for (const auto *s : valid) {
            sum_g += catalog ? s->cat_g : s->flux_g;
        }
        double scale = (sum_g > 0) ? valid.size() / sum_g : 1.0;
        r.resize(valid.size());
        g.resize(valid.size());
        b.resize(valid.size());
        for (size_t i = 0; i < valid.size(); ++i) {
            if (catalog) {
                r[i] = valid[i]->cat_r * scale;
                g[i] = valid[i]->cat_g * scale;
                b[i] = valid[i]->cat_b * scale;
            } else {
                r[i] = valid[i]->flux_r * scale;
                g[i] = valid[i]->flux_g * scale;
                b[i] = valid[i]->flux_b * scale;
            }
        }
    };

    std::vector<double> inst_r, inst_g, inst_b;
    std::vector<double> cat_r, cat_g, cat_b;
    normalize(inst_r, inst_g, inst_b, false);
    normalize(cat_r, cat_g, cat_b, true);

    // Iterative sigma-clipped least squares
    // Fit: [cat_r, cat_g, cat_b]^T = M * [inst_r, inst_g, inst_b]^T
    // We fit each output channel independently: cat_c = m_c0*inst_r + m_c1*inst_g + m_c2*inst_b

    int n = static_cast<int>(valid.size());
    std::vector<bool> inlier(n, true);

    ColorMatrix best_matrix = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    double best_rms = 1e30;
    int n_used = n;

    for (int iter = 0; iter < 5; ++iter) {
        // Count inliers
        int ni = 0;
        for (int i = 0; i < n; ++i) if (inlier[i]) ++ni;
        if (ni < config.min_stars) break;

        // Build system: A * x = b for each output channel
        Eigen::MatrixXd A(ni, 3);
        Eigen::VectorXd b_r(ni), b_g(ni), b_b(ni);

        int row = 0;
        for (int i = 0; i < n; ++i) {
            if (!inlier[i]) continue;
            A(row, 0) = inst_r[i];
            A(row, 1) = inst_g[i];
            A(row, 2) = inst_b[i];
            b_r(row) = cat_r[i];
            b_g(row) = cat_g[i];
            b_b(row) = cat_b[i];
            ++row;
        }

        // Solve via least squares
        Eigen::Vector3d x_r = A.colPivHouseholderQr().solve(b_r);
        Eigen::Vector3d x_g = A.colPivHouseholderQr().solve(b_g);
        Eigen::Vector3d x_b = A.colPivHouseholderQr().solve(b_b);

        best_matrix = {{{x_r(0), x_r(1), x_r(2)},
                        {x_g(0), x_g(1), x_g(2)},
                        {x_b(0), x_b(1), x_b(2)}}};

        // Compute residuals
        std::vector<double> residuals(n, 0.0);
        double sum_sq = 0.0;
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (!inlier[i]) continue;
            double pr = x_r(0) * inst_r[i] + x_r(1) * inst_g[i] + x_r(2) * inst_b[i];
            double pg = x_g(0) * inst_r[i] + x_g(1) * inst_g[i] + x_g(2) * inst_b[i];
            double pb = x_b(0) * inst_r[i] + x_b(1) * inst_g[i] + x_b(2) * inst_b[i];
            double dr = pr - cat_r[i];
            double dg = pg - cat_g[i];
            double db = pb - cat_b[i];
            residuals[i] = std::sqrt(dr * dr + dg * dg + db * db);
            sum_sq += residuals[i] * residuals[i];
            ++count;
        }

        best_rms = (count > 0) ? std::sqrt(sum_sq / count) : 0.0;
        n_used = count;

        // Sigma clip
        if (iter < 4 && best_rms > 0) {
            for (int i = 0; i < n; ++i) {
                if (inlier[i] && residuals[i] > config.sigma_clip * best_rms) {
                    inlier[i] = false;
                }
            }
        }
    }

    res.matrix = best_matrix;
    res.n_stars_used = n_used;
    res.residual_rms = best_rms;
    res.success = (n_used >= config.min_stars);
    if (!res.success) {
        res.error_message = "Too few stars after sigma clipping (" +
                            std::to_string(n_used) + ")";
    }
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
