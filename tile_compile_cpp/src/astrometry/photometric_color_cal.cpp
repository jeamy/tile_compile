#include "tile_compile/astrometry/photometric_color_cal.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace tile_compile::astrometry {

namespace {

struct PlaneFit {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double mx = 0.0;
    double my = 0.0;
    bool ok = false;
};

static PlaneFit fit_sky_plane_huber(const std::vector<double> &xs,
                                    const std::vector<double> &ys,
                                    const std::vector<double> &vs,
                                    double huber_delta,
                                    int max_iters) {
    PlaneFit pf;
    const size_t n = vs.size();
    if (n < 20) return pf;

    for (size_t i = 0; i < n; ++i) {
        pf.mx += xs[i];
        pf.my += ys[i];
    }
    pf.mx /= static_cast<double>(n);
    pf.my /= static_cast<double>(n);

    std::vector<double> w(n, 1.0);

    auto solve_wls = [&](double &a, double &b, double &c) -> bool {
        double S00 = 0.0, S01 = 0.0, S02 = 0.0, S11 = 0.0, S12 = 0.0, S22 = 0.0;
        double T0 = 0.0, T1 = 0.0, T2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double wi = w[i];
            const double X = xs[i] - pf.mx;
            const double Y = ys[i] - pf.my;
            S00 += wi;
            S01 += wi * X;
            S02 += wi * Y;
            S11 += wi * X * X;
            S12 += wi * X * Y;
            S22 += wi * Y * Y;
            const double vi = vs[i];
            T0 += wi * vi;
            T1 += wi * vi * X;
            T2 += wi * vi * Y;
        }

        double A[3][4] = {
            {S00, S01, S02, T0},
            {S01, S11, S12, T1},
            {S02, S12, S22, T2},
        };

        for (int r = 0; r < 3; ++r) {
            int piv = r;
            for (int rr = r + 1; rr < 3; ++rr) {
                if (std::fabs(A[rr][r]) > std::fabs(A[piv][r])) piv = rr;
            }
            if (std::fabs(A[piv][r]) < 1e-12) return false;
            if (piv != r) {
                for (int c0 = r; c0 < 4; ++c0) std::swap(A[r][c0], A[piv][c0]);
            }

            const double div = A[r][r];
            for (int c0 = r; c0 < 4; ++c0) A[r][c0] /= div;
            for (int rr = 0; rr < 3; ++rr) {
                if (rr == r) continue;
                const double f = A[rr][r];
                for (int c0 = r; c0 < 4; ++c0) A[rr][c0] -= f * A[r][c0];
            }
        }

        a = A[0][3];
        b = A[1][3];
        c = A[2][3];
        return true;
    };

    double a = 0.0, b = 0.0, c = 0.0;
    if (!solve_wls(a, b, c)) return pf;

    for (int it = 0; it < std::max(1, max_iters); ++it) {
        std::vector<double> r(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            const double X = xs[i] - pf.mx;
            const double Y = ys[i] - pf.my;
            r[i] = vs[i] - (a + b * X + c * Y);
        }

        std::vector<double> absr = r;
        for (double &v : absr) v = std::fabs(v);
        std::sort(absr.begin(), absr.end());
        double mad = absr[absr.size() / 2] * 1.4826;
        mad = std::max(1e-9, mad);
        const double delta = std::max(1e-6, huber_delta * mad);

        for (size_t i = 0; i < n; ++i) {
            const double ar = std::fabs(r[i]);
            w[i] = (ar <= delta) ? 1.0 : (delta / ar);
        }

        const double a_prev = a;
        const double b_prev = b;
        const double c_prev = c;
        if (!solve_wls(a, b, c)) return pf;

        const double step = std::fabs(a - a_prev) + std::fabs(b - b_prev) + std::fabs(c - c_prev);
        if (step < 1e-8) break;
    }

    pf.a = a;
    pf.b = b;
    pf.c = c;
    pf.ok = true;
    return pf;
}

static int find_tile_index_for_pixel(const TileGrid &grid, double px, double py) {
    if (grid.tiles.empty()) return -1;
    const int x = static_cast<int>(std::lround(px));
    const int y = static_cast<int>(std::lround(py));
    for (size_t i = 0; i < grid.tiles.size(); ++i) {
        const auto &t = grid.tiles[i];
        if (x >= t.x && x < (t.x + t.width) && y >= t.y && y < (t.y + t.height)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static double tile_quality_weight_for_star(double px, double py,
                                           const PCCConfig &config,
                                           bool *reject_out) {
    if (reject_out) *reject_out = false;
    if (!config.use_tile_quality_weighting) return 1.0;
    if (config.tile_metrics.empty() || config.tile_grid.tiles.empty()) return 1.0;
    if (config.tile_metrics.size() != config.tile_grid.tiles.size()) return 1.0;

    const int idx = find_tile_index_for_pixel(config.tile_grid, px, py);
    if (idx < 0) return 1.0;

    const TileMetrics &tm = config.tile_metrics[static_cast<size_t>(idx)];
    const float noise = std::max(1.0e-6f, tm.noise);
    const float structure = std::isfinite(tm.gradient_energy) ? (tm.gradient_energy / noise) : 0.0f;

    if (std::isfinite(structure) && structure > config.tile_structure_reject) {
        if (reject_out) *reject_out = true;
        return 0.0;
    }

    const float q = std::isfinite(tm.quality_score) ? tm.quality_score : 0.0f;
    const double quality_term =
        std::exp(static_cast<double>(config.tile_quality_kappa) * static_cast<double>(q));
    const double structure_term = std::exp(
        -0.35 * static_cast<double>(std::max(0.0f, structure - config.tile_structure_ref)));
    const double star_penalty =
        1.0 / (1.0 + 0.03 * static_cast<double>(std::max(0, tm.star_count - 4)));

    double w = quality_term * structure_term * star_penalty;
    const double wmin = std::max(1.0e-3, static_cast<double>(config.tile_weight_min));
    const double wmax = std::max(wmin, static_cast<double>(config.tile_weight_max));
    w = std::clamp(w, wmin, wmax);
    return w;
}

} // namespace

// ─── Default OSC filter curves (kept for API compatibility) ─────────────

FilterCurves default_osc_filter_curves() {
    FilterCurves fc;
    for (double wl = 336.0; wl <= 1020.0; wl += 2.0) {
        fc.wl.push_back(wl);
        double b = 0.0;
        if (wl >= 380 && wl <= 530) {
            if (wl < 400)       b = (wl - 380) / 20.0;
            else if (wl <= 500) b = 1.0;
            else                b = (530 - wl) / 30.0;
        }
        double g = 0.0;
        if (wl >= 460 && wl <= 620) {
            if (wl < 500)       g = (wl - 460) / 40.0;
            else if (wl <= 580) g = 1.0;
            else                g = (620 - wl) / 40.0;
        }
        double r = 0.0;
        if (wl >= 570 && wl <= 720) {
            if (wl < 600)       r = (wl - 570) / 30.0;
            else if (wl <= 680) r = 1.0;
            else                r = (720 - wl) / 40.0;
        }
        fc.tx_r.push_back(r);
        fc.tx_g.push_back(g);
        fc.tx_b.push_back(b);
    }
    return fc;
}

// ─── Teff estimation from XP spectrum ───────────────────────────────────
//
// Estimate effective temperature from the Gaia XP sampled spectrum by
// fitting the ratio of blue-band to red-band integrated flux to a
// Planck function.  This avoids the need for filter curves entirely.

static double estimate_teff_from_xp(const std::vector<float> &xp_flux) {
    if (xp_flux.size() != XPSAMPLED_LEN) return 0.0;

    // Integrate flux in blue (400-500nm) and red (600-700nm) bands
    double blue_sum = 0, red_sum = 0;
    int blue_n = 0, red_n = 0;
    for (int i = 0; i < XPSAMPLED_LEN; ++i) {
        double wl = XPSAMPLED_WL_START + i * XPSAMPLED_WL_STEP;
        double f = static_cast<double>(xp_flux[i]);
        if (f <= 0) continue;
        if (wl >= 400 && wl <= 500) { blue_sum += f; ++blue_n; }
        if (wl >= 600 && wl <= 700) { red_sum  += f; ++red_n;  }
    }
    if (blue_n == 0 || red_n == 0 || red_sum <= 0) return 0.0;

    double ratio = (blue_sum / blue_n) / (red_sum / red_n);

    // Map blue/red ratio to Teff using Planck function lookup.
    // Precomputed: for Planck B(λ,T), ratio of mean flux in 400-500nm
    // to mean flux in 600-700nm:
    //   T=3000K → ~0.10,  T=5000K → ~0.55,  T=7000K → ~1.05
    //   T=10000K → ~1.55, T=20000K → ~2.20, T=40000K → ~2.60
    // Use bisection on Planck function for accuracy.
    auto planck_ratio = [](double T) -> double {
        constexpr double h = 6.62607015e-34;
        constexpr double c = 2.99792458e8;
        constexpr double k = 1.380649e-23;
        double b_sum = 0, r_sum = 0;
        for (double wl = 400; wl <= 500; wl += 2.0) {
            double wl_m = wl * 1e-9;
            b_sum += (1.0 / (wl_m * wl_m * wl_m * wl_m * wl_m)) /
                     (std::exp(h * c / (wl_m * k * T)) - 1.0);
        }
        for (double wl = 600; wl <= 700; wl += 2.0) {
            double wl_m = wl * 1e-9;
            r_sum += (1.0 / (wl_m * wl_m * wl_m * wl_m * wl_m)) /
                     (std::exp(h * c / (wl_m * k * T)) - 1.0);
        }
        return (r_sum > 0) ? b_sum / r_sum : 0.0;
    };

    // Bisection: find T such that planck_ratio(T) ≈ ratio
    double T_lo = 2000, T_hi = 50000;
    for (int i = 0; i < 40; ++i) {
        double T_mid = 0.5 * (T_lo + T_hi);
        if (planck_ratio(T_mid) < ratio)
            T_lo = T_mid;
        else
            T_hi = T_mid;
    }
    double teff = 0.5 * (T_lo + T_hi);
    return (teff >= 2000 && teff <= 50000) ? teff : 0.0;
}

// ─── Blackbody temperature to linear sRGB ───────────────────────────────
//
// Converts a color temperature to linear sRGB (r,g,b) normalized so
// max(r,g,b) = 1.  Uses the CIE 1931 2° standard observer + Planck
// function → XYZ → linear sRGB (D65).
//
// This is the same approach as Siril's TempK2rgb() but without lcms2.

static void teff_to_rgb(double T, double &r, double &g, double &b) {
    r = g = b = 0;
    if (T < 1000 || T > 50000) return;

    // CIE 1931 2° color matching functions (5nm steps, 380-780nm)
    // Tabulated x̄, ȳ, z̄ — we use a simplified analytical approximation
    // (multi-lobe Gaussian fit by Wyman et al. 2013)
    auto xFit = [](double wl) -> double {
        double t1 = (wl - 442.0) * ((wl < 442.0) ? 0.0624 : 0.0374);
        double t2 = (wl - 599.8) * ((wl < 599.8) ? 0.0264 : 0.0323);
        double t3 = (wl - 501.1) * ((wl < 501.1) ? 0.0490 : 0.0382);
        return 0.362 * std::exp(-0.5*t1*t1) + 1.056 * std::exp(-0.5*t2*t2)
               - 0.065 * std::exp(-0.5*t3*t3);
    };
    auto yFit = [](double wl) -> double {
        double t1 = (wl - 568.8) * ((wl < 568.8) ? 0.0213 : 0.0247);
        double t2 = (wl - 530.9) * ((wl < 530.9) ? 0.0613 : 0.0322);
        return 0.821 * std::exp(-0.5*t1*t1) + 0.286 * std::exp(-0.5*t2*t2);
    };
    auto zFit = [](double wl) -> double {
        double t1 = (wl - 437.0) * ((wl < 437.0) ? 0.0845 : 0.0278);
        double t2 = (wl - 459.0) * ((wl < 459.0) ? 0.0385 : 0.0725);
        return 1.217 * std::exp(-0.5*t1*t1) + 0.681 * std::exp(-0.5*t2*t2);
    };

    constexpr double h = 6.62607015e-34;
    constexpr double c = 2.99792458e8;
    constexpr double k = 1.380649e-23;

    double X = 0, Y = 0, Z = 0;
    for (double wl = 380; wl <= 780; wl += 5.0) {
        double wl_m = wl * 1e-9;
        double planck = (2.0 * h * c * c / (wl_m * wl_m * wl_m * wl_m * wl_m)) /
                        (std::exp(h * c / (wl_m * k * T)) - 1.0);
        X += planck * xFit(wl);
        Y += planck * yFit(wl);
        Z += planck * zFit(wl);
    }

    // CIE XYZ → linear sRGB (D65 illuminant)
    // sRGB matrix (IEC 61966-2-1)
    r =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
    b =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;

    // Clamp negatives, then normalize so sum = 1.
    // Sum-normalization makes cat values comparable across different Teff
    // (unlike max=1 which changes the reference channel per star).
    r = std::max(0.0, r);
    g = std::max(0.0, g);
    b = std::max(0.0, b);
    double s = r + g + b;
    if (s > 0) { r /= s; g /= s; b /= s; }
}

// ─── Aperture photometry ────────────────────────────────────────────────

static double aperture_flux(const Matrix2Df &img, double cx, double cy,
                            double r_ap, double r_ann_in, double r_ann_out,
                            const std::string &background_model) {
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
    std::vector<double> sky_x;
    std::vector<double> sky_y;
    sky_x.reserve(static_cast<size_t>(std::max(1, (x1 - x0 + 1) * (y1 - y0 + 1) / 2)));
    sky_y.reserve(sky_x.capacity());
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            double dx = x - cx;
            double dy = y - cy;
            double d2 = dx * dx + dy * dy;
            if (d2 >= r_in2 && d2 <= r_out2) {
                float v = img(y, x);
                if (std::isfinite(v) && v > 0.0f) {
                    sky_pixels.push_back(v);
                    sky_x.push_back(static_cast<double>(x));
                    sky_y.push_back(static_cast<double>(y));
                }
            }
        }
    }

    if (sky_pixels.size() < 10) return -1.0;  // not enough sky pixels

    // Median sky background (fallback and noise guard reference)
    std::sort(sky_pixels.begin(), sky_pixels.end());
    double sky_bg_median = sky_pixels[sky_pixels.size() / 2];
    PlaneFit plane_fit{};
    bool use_plane_bg = false;

    if (background_model == "plane") {
        std::vector<double> sky_vals;
        sky_vals.reserve(sky_pixels.size());
        for (float v : sky_pixels) sky_vals.push_back(static_cast<double>(v));
        plane_fit = fit_sky_plane_huber(sky_x, sky_y, sky_vals, 1.5, 8);
        use_plane_bg = plane_fit.ok;
        if (use_plane_bg) {
            sky_bg_median = plane_fit.a + plane_fit.b * (cx - plane_fit.mx) +
                            plane_fit.c * (cy - plane_fit.my);
        }
    }

    if (!(std::isfinite(sky_bg_median) && sky_bg_median > 0.0)) return -1.0;
    const size_t n_sky = sky_pixels.size();
    const float q1 = sky_pixels[(n_sky * 1) / 4];
    const float q3 = sky_pixels[(n_sky * 3) / 4];
    const double iqr = static_cast<double>(q3) - static_cast<double>(q1);
    if (!(std::isfinite(iqr) && iqr >= 0.0)) return -1.0;
    if (iqr > 0.35 * sky_bg_median) return -1.0;

    // Sum aperture flux minus background.
    // For model=plane subtract the local plane value at each aperture pixel.
    double total = 0.0;
    int n_ap = 0;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            double dx = x - cx;
            double dy = y - cy;
            if (dx * dx + dy * dy <= r_ap2) {
                float v = img(y, x);
                if (std::isfinite(v) && v > 0.0f) {
                    double bg_here = sky_bg_median;
                    if (use_plane_bg) {
                        bg_here = plane_fit.a + plane_fit.b * (x - plane_fit.mx) +
                                  plane_fit.c * (y - plane_fit.my);
                    }
                    if (!(std::isfinite(bg_here) && bg_here > 0.0)) continue;
                    total += static_cast<double>(v) - bg_here;
                    ++n_ap;
                }
            }
        }
    }

    return (n_ap > 0) ? total : -1.0;
}

static float sampled_high_percentile(const Matrix2Df &img, int step, float q) {
    const int rows = img.rows();
    const int cols = img.cols();
    if (rows <= 0 || cols <= 0 || step <= 0) {
        return std::numeric_limits<float>::infinity();
    }

    std::vector<float> samples;
    samples.reserve((rows / step + 1) * (cols / step + 1));
    for (int y = 0; y < rows; y += step) {
        for (int x = 0; x < cols; x += step) {
            const float v = img(y, x);
            if (std::isfinite(v) && v > 0.0f) {
                samples.push_back(v);
            }
        }
    }

    if (samples.size() < 64) {
        return std::numeric_limits<float>::infinity();
    }

    std::sort(samples.begin(), samples.end());
    q = std::clamp(q, 0.0f, 1.0f);
    const size_t idx = static_cast<size_t>(q * static_cast<float>(samples.size() - 1));
    return samples[idx];
}

std::vector<StarPhotometry> measure_stars(
    const Matrix2Df &R, const Matrix2Df &G, const Matrix2Df &B,
    const WCS &wcs,
    const std::vector<GaiaStar> &catalog_stars,
    const PCCConfig &config) {

    std::vector<StarPhotometry> result;
    result.reserve(catalog_stars.size());

    int rows = R.rows();
    int cols = R.cols();
    double margin = config.annulus_outer_px + 2.0;

    // Reject saturation-near stars before fitting PCC: stars close to sensor
    // non-linearity bias color ratios and can drive casts in bright structures.
    const float sat_r = sampled_high_percentile(R, 8, 0.999f);
    const float sat_g = sampled_high_percentile(G, 8, 0.999f);
    const float sat_b = sampled_high_percentile(B, 8, 0.999f);
    constexpr float sat_guard_frac = 0.995f;
    int n_sat_rejected = 0;
    int n_tile_rejected = 0;
    std::vector<double> tile_weights_used;
    tile_weights_used.reserve(catalog_stars.size());

    for (const auto &star : catalog_stars) {
        // Magnitude filter
        if (star.mag > config.mag_limit || star.mag < config.mag_bright_limit) continue;

        // Project to pixel
        double px, py;
        if (!wcs.sky_to_pixel(star.ra, star.dec, px, py)) continue;

        // Check within image bounds with margin
        if (px < margin || px >= cols - margin ||
            py < margin || py >= rows - margin) continue;

        const int cx = static_cast<int>(std::lround(px));
        const int cy = static_cast<int>(std::lround(py));
        float peak_r = 0.0f;
        float peak_g = 0.0f;
        float peak_b = 0.0f;
        for (int yy = std::max(0, cy - 1); yy <= std::min(rows - 1, cy + 1); ++yy) {
            for (int xx = std::max(0, cx - 1); xx <= std::min(cols - 1, cx + 1); ++xx) {
                peak_r = std::max(peak_r, R(yy, xx));
                peak_g = std::max(peak_g, G(yy, xx));
                peak_b = std::max(peak_b, B(yy, xx));
            }
        }

        const bool near_sat_r = std::isfinite(sat_r) && sat_r > 0.0f && peak_r >= sat_guard_frac * sat_r;
        const bool near_sat_g = std::isfinite(sat_g) && sat_g > 0.0f && peak_g >= sat_guard_frac * sat_g;
        const bool near_sat_b = std::isfinite(sat_b) && sat_b > 0.0f && peak_b >= sat_guard_frac * sat_b;
        if (near_sat_r || near_sat_g || near_sat_b) {
            ++n_sat_rejected;
            continue;
        }

        StarPhotometry sp;
        sp.ra = star.ra;
        sp.dec = star.dec;
        sp.px = px;
        sp.py = py;
        sp.mag = star.mag;

        bool tile_reject = false;
        sp.quality_weight = tile_quality_weight_for_star(px, py, config, &tile_reject);
        if (tile_reject || !(std::isfinite(sp.quality_weight) && sp.quality_weight > 0.0)) {
            ++n_tile_rejected;
            continue;
        }

        // Measure instrumental flux in each channel
        sp.flux_r = aperture_flux(R, px, py, config.aperture_radius_px,
                                  config.annulus_inner_px, config.annulus_outer_px,
                                  config.background_model);
        sp.flux_g = aperture_flux(G, px, py, config.aperture_radius_px,
                                  config.annulus_inner_px, config.annulus_outer_px,
                                  config.background_model);
        sp.flux_b = aperture_flux(B, px, py, config.aperture_radius_px,
                                  config.annulus_inner_px, config.annulus_outer_px,
                                  config.background_model);

        // PCC method: get Teff, then convert to expected linear sRGB.
        // cat_r/g/b store the expected color (normalized, max=1).
        // Priority: 1) teff from catalog  2) estimate from XP spectrum
        double teff = star.teff;
        if (teff <= 0 && !star.xp_flux.empty())
            teff = estimate_teff_from_xp(star.xp_flux);

        if (teff > 0) {
            teff_to_rgb(teff, sp.cat_r, sp.cat_g, sp.cat_b);
        } else {
            sp.cat_r = sp.cat_g = sp.cat_b = 0.0;
        }

        // Validate: all fluxes must be positive
        sp.valid = (sp.flux_r > 0 && sp.flux_g > 0 && sp.flux_b > 0 &&
                    sp.cat_r > 0 && sp.cat_g > 0 && sp.cat_b > 0);

        if (sp.valid) {
            tile_weights_used.push_back(sp.quality_weight);
        }

        result.push_back(sp);
    }

    std::cerr << "[PCC] Saturation guard: rejected=" << n_sat_rejected
              << " sat_r=" << sat_r << " sat_g=" << sat_g << " sat_b=" << sat_b
              << " frac=" << sat_guard_frac << std::endl;
    if (!tile_weights_used.empty()) {
        double w_sum = 0.0;
        double w_min = std::numeric_limits<double>::infinity();
        double w_max = 0.0;
        for (double w : tile_weights_used) {
            w_sum += w;
            w_min = std::min(w_min, w);
            w_max = std::max(w_max, w);
        }
        std::cerr << "[PCC] Tile-quality weighting: enabled="
                  << (config.use_tile_quality_weighting ? "true" : "false")
                  << " rejected=" << n_tile_rejected
                  << " mean=" << (w_sum / static_cast<double>(tile_weights_used.size()))
                  << " min=" << w_min
                  << " max=" << w_max << std::endl;
    } else if (config.use_tile_quality_weighting) {
        std::cerr << "[PCC] Tile-quality weighting: enabled=true rejected="
                  << n_tile_rejected << " no valid weighted stars" << std::endl;
    }

    return result;
}

// ─── Color matrix fitting (Siril PCC method) ────────────────────────────
//
// Follows the Siril PCC algorithm (get_pcc_white_balance_coeffs):
//   1. For each star: Teff → expected RGB via Planck→CIE XYZ→sRGB
//   2. Per-star correction factor: k_c = expected_c / measured_flux_c
//   3. Robust mean of all per-star factors → final kw[R], kw[G], kw[B]
//   4. Normalize so max(kw) = 1
//
// This method does NOT require sensor-specific filter curves.
// It only needs the effective temperature of each star.

static double weighted_median(const std::vector<double> &vals,
                              const std::vector<double> &weights) {
    if (vals.empty() || vals.size() != weights.size()) return 0.0;
    std::vector<std::pair<double, double>> vw;
    vw.reserve(vals.size());
    double w_sum = 0.0;
    for (size_t i = 0; i < vals.size(); ++i) {
        const double w = std::max(1.0e-9, weights[i]);
        vw.emplace_back(vals[i], w);
        w_sum += w;
    }
    if (w_sum <= 0.0) return 0.0;

    std::sort(vw.begin(), vw.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    const double half = 0.5 * w_sum;
    double acc = 0.0;
    for (const auto &p : vw) {
        acc += p.second;
        if (acc >= half) return p.first;
    }
    return vw.back().first;
}

// Weighted robust location estimator with iterative sigma clipping around
// weighted median / weighted MAD.
static double robust_mean_weighted(std::vector<double> &data,
                                   std::vector<double> &weights,
                                   double sigma,
                                   double &deviation_out) {
    deviation_out = 0.0;
    if (data.empty() || data.size() != weights.size()) return 0.0;

    for (int iter = 0; iter < 5; ++iter) {
        const int n = static_cast<int>(data.size());
        if (n < 3) break;
        const double center = weighted_median(data, weights);

        std::vector<double> devs;
        devs.reserve(data.size());
        for (double v : data) devs.push_back(std::abs(v - center));
        const double mad = 1.4826 * weighted_median(devs, weights);
        if (mad < 1.0e-15) break;

        const double threshold = sigma * mad;
        std::vector<double> kept_data;
        std::vector<double> kept_weights;
        kept_data.reserve(data.size());
        kept_weights.reserve(weights.size());
        for (size_t i = 0; i < data.size(); ++i) {
            if (std::abs(data[i] - center) <= threshold) {
                kept_data.push_back(data[i]);
                kept_weights.push_back(weights[i]);
            }
        }
        if (static_cast<int>(kept_data.size()) < 3) break;
        if (kept_data.size() == data.size()) break;
        data.swap(kept_data);
        weights.swap(kept_weights);
    }

    const double center = weighted_median(data, weights);
    double dev_sum = 0.0;
    double w_sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        const double w = std::max(1.0e-9, weights[i]);
        dev_sum += w * std::abs(data[i] - center);
        w_sum += w;
    }
    deviation_out = (w_sum > 0.0) ? (dev_sum / w_sum) : 0.0;
    return center;
}

PCCResult fit_color_matrix(const std::vector<StarPhotometry> &stars,
                           const PCCConfig &config) {
    PCCResult res;
    res.success = false;
    res.matrix = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};  // identity default
    res.n_stars_matched = 0;
    res.n_stars_used = 0;
    res.residual_rms = 0.0;
    res.determinant = 1.0;
    res.condition_number = 1.0;

    // Collect valid stars
    std::vector<const StarPhotometry*> valid;
    for (const auto &s : stars) {
        if (s.valid && s.flux_r > 0 && s.flux_g > 0 && s.flux_b > 0 &&
            s.cat_r > 0 && s.cat_g > 0 && s.cat_b > 0)
            valid.push_back(&s);
    }

    res.n_stars_matched = static_cast<int>(valid.size());

    if (static_cast<int>(valid.size()) < config.min_stars) {
        res.error_message = "Not enough valid stars (" +
                            std::to_string(valid.size()) + " < " +
                            std::to_string(config.min_stars) + ")";
        return res;
    }

    // PCC via log-chromaticity deltas (Siril-like robust color ratio solve).
    // We fit signed deltas in log space:
    //   d_rg = log(cat_r/cat_g) - log(flux_r/flux_g)
    //   d_bg = log(cat_b/cat_g) - log(flux_b/flux_g)
    // Then derive scales with G as anchor:
    //   s_r/s_g = exp(d_rg),  s_b/s_g = exp(d_bg),  s_g = 1.
    std::vector<double> d_rg_vec, d_bg_vec;
    std::vector<double> w_rg_vec, w_bg_vec;
    d_rg_vec.reserve(valid.size());
    d_bg_vec.reserve(valid.size());
    w_rg_vec.reserve(valid.size());
    w_bg_vec.reserve(valid.size());

    for (const auto *s : valid) {
        if (!(s->flux_r > 0.0 && s->flux_g > 0.0 && s->flux_b > 0.0 &&
              s->cat_r > 0.0 && s->cat_g > 0.0 && s->cat_b > 0.0)) {
            continue;
        }

        const double meas_rg = s->flux_r / s->flux_g;
        const double meas_bg = s->flux_b / s->flux_g;
        const double cat_rg = s->cat_r / s->cat_g;
        const double cat_bg = s->cat_b / s->cat_g;
        if (!(meas_rg > 0.0 && meas_bg > 0.0 && cat_rg > 0.0 && cat_bg > 0.0)) {
            continue;
        }

        const double w = std::clamp(s->quality_weight, 1.0e-3, 10.0);
        d_rg_vec.push_back(std::log(cat_rg) - std::log(meas_rg));
        d_bg_vec.push_back(std::log(cat_bg) - std::log(meas_bg));
        w_rg_vec.push_back(w);
        w_bg_vec.push_back(w);
    }

    if (d_rg_vec.size() < static_cast<size_t>(config.min_stars) ||
        d_bg_vec.size() < static_cast<size_t>(config.min_stars)) {
        res.error_message = "Not enough robust stars for log-chromaticity PCC";
        return res;
    }

    double dev_rg = 0.0;
    double dev_bg = 0.0;
    const double d_rg = robust_mean_weighted(d_rg_vec, w_rg_vec, config.sigma_clip, dev_rg);
    const double d_bg = robust_mean_weighted(d_bg_vec, w_bg_vec, config.sigma_clip, dev_bg);

    if (d_rg_vec.size() < static_cast<size_t>(config.min_stars) ||
        d_bg_vec.size() < static_cast<size_t>(config.min_stars)) {
        res.error_message = "Not enough stars after weighted sigma clipping";
        return res;
    }

    // G is the luminance anchor (G=1.0), matching Siril's get_pcc_white_balance_coeffs.
    // R and B are scaled relative to G from the log-chromaticity deltas.
    // Chroma compression toward identity is applied only to R and B deviations from 1.
    constexpr double chroma_strength = 0.70;  // 1.0 = full correction, 0.0 = no correction
    auto compress_deviation = [&](double raw_scale) {
        // compress the log-delta toward zero, keeping G-anchor at 1
        const double log_k = std::log(std::max(raw_scale, 1e-9));
        return std::exp(log_k * chroma_strength);
    };

    double kw_r = compress_deviation(std::exp(d_rg));
    double kw_g = 1.0;   // G is always the anchor
    double kw_b = compress_deviation(std::exp(d_bg));

    // Guardrails relative to G=1: cap R and B deviations.
    // Tighten bounds when robust scatter is high to avoid chroma runaway.
    const double dev_max = std::max(dev_rg, dev_bg);
    double k_max = 1.20;
    if (dev_max > 0.30) {
        k_max = 1.08;
    } else if (dev_max > 0.20) {
        k_max = 1.12;
    }
    const double k_min = 1.0 / k_max;
    kw_r = std::clamp(kw_r, k_min, k_max);
    kw_b = std::clamp(kw_b, k_min, k_max);

    std::cerr << "[PCC] Log-chroma deltas: d_rg=" << d_rg << " (dev=" << dev_rg << ")"
              << " d_bg=" << d_bg << " (dev=" << dev_bg << ")"
              << " stars=" << d_rg_vec.size() << std::endl;
    std::cerr << "[PCC] Guardrail bounds: min=" << k_min
              << " max=" << k_max << std::endl;
    if (!w_rg_vec.empty()) {
        double w_sum = 0.0;
        double w_min = std::numeric_limits<double>::infinity();
        double w_max = 0.0;
        for (double w : w_rg_vec) {
            w_sum += w;
            w_min = std::min(w_min, w);
            w_max = std::max(w_max, w);
        }
        std::cerr << "[PCC] Weighted fit stats: n=" << w_rg_vec.size()
                  << " mean_w=" << (w_sum / static_cast<double>(w_rg_vec.size()))
                  << " min_w=" << w_min
                  << " max_w=" << w_max << std::endl;
    }

    // Build diagonal color matrix
    res.matrix = {{{kw_r, 0, 0}, {0, kw_g, 0}, {0, 0, kw_b}}};
    res.n_stars_used = static_cast<int>(std::min(d_rg_vec.size(), d_bg_vec.size()));
    res.residual_rms = std::max(dev_rg, dev_bg);
    res.determinant = kw_r * kw_g * kw_b;
    const double s0 = std::abs(kw_r);
    const double s1 = std::abs(kw_g);
    const double s2 = std::abs(kw_b);
    const double s_max = std::max({s0, s1, s2});
    const double s_min = std::min({s0, s1, s2});
    res.condition_number =
        (s_min > 1.0e-12) ? (s_max / s_min) : std::numeric_limits<double>::infinity();

    std::cerr << "[PCC] Matrix stability: det=" << res.determinant
              << " cond=" << res.condition_number
              << " residual=" << res.residual_rms << std::endl;

    if (!std::isfinite(res.determinant) || res.determinant <= 0.0) {
        res.error_message = "Unstable PCC matrix: non-positive determinant";
        return res;
    }
    if (!std::isfinite(res.condition_number) ||
        res.condition_number > config.max_condition_number) {
        res.error_message = "Unstable PCC matrix: condition number " +
                            std::to_string(res.condition_number) + " exceeds " +
                            std::to_string(config.max_condition_number);
        return res;
    }
    if (!std::isfinite(res.residual_rms) ||
        res.residual_rms > config.max_residual_rms) {
        res.error_message = "PCC residual RMS " + std::to_string(res.residual_rms) +
                            " exceeds " + std::to_string(config.max_residual_rms);
        return res;
    }

    res.success = true;
    std::cerr << "[PCC] Scale factors: R=" << kw_r
              << " G=" << kw_g << " B=" << kw_b << std::endl;
    return res;
}

// ─── Apply color matrix ─────────────────────────────────────────────────

// Estimate per-channel background as the median of a subsample
static float estimate_background(const Matrix2Df &img) {
    int rows = img.rows();
    int cols = img.cols();
    // Sample every 8th pixel for speed
    std::vector<float> samples;
    samples.reserve((rows / 8 + 1) * (cols / 8 + 1));
    for (int y = 0; y < rows; y += 8)
        for (int x = 0; x < cols; x += 8)
        {
            float v = img(y, x);
            if (std::isfinite(v) && v > 0.0f) {
                samples.push_back(v);
            }
        }
    if (samples.empty())
        return 0.0f;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static float percentile_sorted(const std::vector<float> &sorted, float q) {
    if (sorted.empty()) return 0.0f;
    if (q <= 0.0f) return sorted.front();
    if (q >= 1.0f) return sorted.back();
    const size_t idx = static_cast<size_t>(q * static_cast<float>(sorted.size() - 1));
    return sorted[idx];
}

void apply_color_matrix(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
                        const ColorMatrix &m) {
    int rows = R.rows();
    int cols = R.cols();

    // Background-aware application:
    // We subtract each channel's estimated sky background so the matrix acts on
    // signal above background (consistent with the aperture photometry step).
    // Then we add back a *single* neutral background level so the output sky is
    // approximately neutral (instead of preserving any channel bias from light
    // pollution / sensor response).
    float bg_r = estimate_background(R);
    float bg_g = estimate_background(G);
    float bg_b = estimate_background(B);
    float bg_out = (bg_r + bg_g + bg_b) / 3.0f;

    std::cerr << "[PCC] Background levels: R=" << bg_r
              << " G=" << bg_g << " B=" << bg_b
              << " -> bg_out=" << bg_out << std::endl;

    // Highlight-safe blending: in very bright signal regions we attenuate
    // the color correction strength towards identity to avoid nebula-core
    // over-tinting from a globally fitted matrix.
    std::vector<float> signal_luma;
    signal_luma.reserve((rows / 4 + 1) * (cols / 4 + 1));
    for (int y = 0; y < rows; y += 4) {
        for (int x = 0; x < cols; x += 4) {
            const float r0 = R(y, x);
            const float g0 = G(y, x);
            const float b0 = B(y, x);
            if (!(std::isfinite(r0) && r0 > 0.0f && std::isfinite(g0) && g0 > 0.0f &&
                  std::isfinite(b0) && b0 > 0.0f)) {
                continue;
            }
            const float dr = r0 - bg_r;
            const float dg = g0 - bg_g;
            const float db = b0 - bg_b;
            const float luma = std::max(0.0f, 0.2126f * dr + 0.7152f * dg + 0.0722f * db);
            if (luma > 0.0f) {
                signal_luma.push_back(luma);
            }
        }
    }

    float shadow_lo = 0.0f;
    float shadow_hi = 0.0f;
    float blend_lo = 0.0f;
    float blend_hi = 0.0f;
    if (signal_luma.size() >= 32) {
        std::sort(signal_luma.begin(), signal_luma.end());
        shadow_lo = percentile_sorted(signal_luma, 0.01f);
        shadow_hi = percentile_sorted(signal_luma, 0.20f);
        blend_lo = percentile_sorted(signal_luma, 0.90f);
        blend_hi = percentile_sorted(signal_luma, 0.995f);
        if (!(std::isfinite(shadow_lo) && std::isfinite(shadow_hi) && shadow_hi > shadow_lo)) {
            shadow_lo = 0.0f;
            shadow_hi = 0.0f;
        }
        if (!(std::isfinite(blend_lo) && std::isfinite(blend_hi) && blend_hi > blend_lo)) {
            blend_lo = 0.0f;
            blend_hi = 0.0f;
        }
    }

    std::cerr << "[PCC] Shadow blend thresholds: lo=" << shadow_lo
              << " hi=" << shadow_hi
              << " samples=" << signal_luma.size() << std::endl;
    std::cerr << "[PCC] Highlight blend thresholds: lo=" << blend_lo
              << " hi=" << blend_hi
              << " samples=" << signal_luma.size() << std::endl;

    // Also attenuate in very low-signal regions to avoid tinting background
    // chroma noise/gradients into red-green clouding.
    constexpr float shadow_atten_floor = 0.10f;
    std::cerr << "[PCC] Shadow attenuation floor=" << shadow_atten_floor << std::endl;

    // Keep a minimum correction in highlights to avoid swinging too far back to
    // raw sensor response (which can reintroduce greenish bias in bright cores).
    constexpr float atten_floor = 0.25f;
    std::cerr << "[PCC] Highlight attenuation floor=" << atten_floor << std::endl;

    size_t valid_px = 0;
    const size_t total_px = static_cast<size_t>(rows) * static_cast<size_t>(cols);

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const float r0 = R(y, x);
            const float g0 = G(y, x);
            const float b0 = B(y, x);
            if (!(std::isfinite(r0) && r0 > 0.0f && std::isfinite(g0) && g0 > 0.0f &&
                  std::isfinite(b0) && b0 > 0.0f)) {
                R(y, x) = 0.0f;
                G(y, x) = 0.0f;
                B(y, x) = 0.0f;
                continue;
            }

            ++valid_px;

            float dr = r0 - bg_r;
            float dg = g0 - bg_g;
            float db = b0 - bg_b;

            const float luma = std::max(0.0f, 0.2126f * dr + 0.7152f * dg + 0.0722f * db);
            float atten_shadows = 1.0f;
            if (shadow_hi > shadow_lo && luma < shadow_hi) {
                float t = (luma - shadow_lo) / (shadow_hi - shadow_lo);
                t = std::clamp(t, 0.0f, 1.0f);
                const float s = t * t * (3.0f - 2.0f * t);
                atten_shadows = shadow_atten_floor + (1.0f - shadow_atten_floor) * s;
            }

            float atten_highlights = 1.0f;
            if (blend_hi > blend_lo && luma > blend_lo) {
                float t = (luma - blend_lo) / (blend_hi - blend_lo);
                t = std::clamp(t, 0.0f, 1.0f);
                // smoothstep
                const float s = t * t * (3.0f - 2.0f * t);
                atten_highlights = 1.0f - (1.0f - atten_floor) * s;
            }
            const float atten = std::min(atten_shadows, atten_highlights);

            const float m00 = static_cast<float>(1.0 + atten * (m[0][0] - 1.0));
            const float m01 = static_cast<float>(atten * m[0][1]);
            const float m02 = static_cast<float>(atten * m[0][2]);
            const float m10 = static_cast<float>(atten * m[1][0]);
            const float m11 = static_cast<float>(1.0 + atten * (m[1][1] - 1.0));
            const float m12 = static_cast<float>(atten * m[1][2]);
            const float m20 = static_cast<float>(atten * m[2][0]);
            const float m21 = static_cast<float>(atten * m[2][1]);
            const float m22 = static_cast<float>(1.0 + atten * (m[2][2] - 1.0));

            float nr = m00 * dr + m01 * dg + m02 * db;
            float ng = m10 * dr + m11 * dg + m12 * db;
            float nb = m20 * dr + m21 * dg + m22 * db;

            R(y, x) = bg_out + nr;
            G(y, x) = bg_out + ng;
            B(y, x) = bg_out + nb;
        }
    }

    const size_t skipped_px = (total_px >= valid_px) ? (total_px - valid_px) : 0;
    const double frac = (total_px > 0) ? (static_cast<double>(valid_px) / static_cast<double>(total_px)) : 0.0;
    std::cerr << "[PCC] Apply valid pixels: " << valid_px << "/" << total_px
              << " (" << (100.0 * frac) << "%)  skipped=" << skipped_px
              << std::endl;
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
