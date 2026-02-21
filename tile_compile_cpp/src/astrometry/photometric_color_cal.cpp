#include "tile_compile/astrometry/photometric_color_cal.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace tile_compile::astrometry {

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
                float v = img(y, x);
                if (std::isfinite(v) && v > 0.0f) {
                    sky_pixels.push_back(v);
                }
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
                float v = img(y, x);
                if (std::isfinite(v) && v > 0.0f) {
                    total += static_cast<double>(v) - sky_bg;
                    ++n_ap;
                }
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

        result.push_back(sp);
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

// Robust mean with iterative sigma-clipping (like Siril's
// siril_stats_robust_mean).  Sorts data, clips outliers, returns mean.
static double robust_mean(std::vector<double> &data, double sigma,
                          double &deviation_out) {
    deviation_out = 0;
    if (data.empty()) return 0;

    std::sort(data.begin(), data.end());

    // Remove obvious outliers (values that are FLT_MAX or negative)
    while (!data.empty() && data.back() > 1e10) data.pop_back();
    while (!data.empty() && data.front() < 0) data.erase(data.begin());
    if (data.empty()) return 0;

    // Iterative sigma-clipping
    for (int iter = 0; iter < 5; ++iter) {
        int n = static_cast<int>(data.size());
        if (n < 3) break;

        double sum = 0;
        for (double v : data) sum += v;
        double mean = sum / n;

        // MAD (median absolute deviation)
        std::vector<double> devs;
        devs.reserve(n);
        for (double v : data) devs.push_back(std::abs(v - mean));
        std::sort(devs.begin(), devs.end());
        double mad = devs[devs.size() / 2] * 1.4826;
        if (mad < 1e-15) break;

        double threshold = sigma * mad;
        std::vector<double> kept;
        for (double v : data) {
            if (std::abs(v - mean) <= threshold)
                kept.push_back(v);
        }
        if (static_cast<int>(kept.size()) < 3) break;
        data = kept;
    }

    // Final mean and deviation
    double sum = 0;
    for (double v : data) sum += v;
    double mean = sum / data.size();

    double dev_sum = 0;
    for (double v : data) dev_sum += std::abs(v - mean);
    deviation_out = dev_sum / data.size();

    return mean;
}

PCCResult fit_color_matrix(const std::vector<StarPhotometry> &stars,
                           const PCCConfig &config) {
    PCCResult res;
    res.success = false;
    res.matrix = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};  // identity default

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

    // PCC via normalized color proportions.
    //
    // For OSC debayered images, point-source fluxes have a systematic
    // channel scaling artifact (R,B ~1.5-1.7× G) from Bayer interpolation.
    // The background is already neutral.  To avoid this artifact biasing
    // the calibration, we normalize BOTH measured and expected fluxes to
    // sum=1 per star, then compute per-star correction factors on the
    // color *proportions* only.
    //
    // Per star:  meas_frac_c = flux_c / (flux_r + flux_g + flux_b)
    //            cat_frac_c  = cat_c  / (cat_r  + cat_g  + cat_b)
    //            k_c = cat_frac_c / meas_frac_c
    //
    // The Bayer artifact scales all channels by a constant factor per star
    // (R and B by ~1.7, G by 1.0), which cancels in the fraction.

    std::vector<double> kr_vec, kg_vec, kb_vec;
    kr_vec.reserve(valid.size());
    kg_vec.reserve(valid.size());
    kb_vec.reserve(valid.size());

    for (const auto *s : valid) {
        double ftot = s->flux_r + s->flux_g + s->flux_b;
        double ctot = s->cat_r  + s->cat_g  + s->cat_b;
        if (ftot <= 0 || ctot <= 0) continue;

        double mf_r = s->flux_r / ftot;
        double mf_g = s->flux_g / ftot;
        double mf_b = s->flux_b / ftot;
        double cf_r = s->cat_r / ctot;
        double cf_g = s->cat_g / ctot;
        double cf_b = s->cat_b / ctot;

        if (mf_r > 0.01 && mf_g > 0.01 && mf_b > 0.01) {
            kr_vec.push_back(cf_r / mf_r);
            kg_vec.push_back(cf_g / mf_g);
            kb_vec.push_back(cf_b / mf_b);
        }
    }

    double dev_r = 0, dev_g = 0, dev_b = 0;
    double kw_r = robust_mean(kr_vec, config.sigma_clip, dev_r);
    double kw_g = robust_mean(kg_vec, config.sigma_clip, dev_g);
    double kw_b = robust_mean(kb_vec, config.sigma_clip, dev_b);

    std::cerr << "[PCC] Proportion corrections: R=" << kw_r << " (dev=" << dev_r << ")"
              << " G=" << kw_g << " (dev=" << dev_g << ")"
              << " B=" << kw_b << " (dev=" << dev_b << ")"
              << " (" << kr_vec.size() << " stars)" << std::endl;

    if (kw_r <= 0 || kw_g <= 0 || kw_b <= 0) {
        res.error_message = "PCC failed: non-positive proportion correction";
        return res;
    }

    // Normalize so max = 1 (only scale down, never up)
    double maxk = std::max({kw_r, kw_g, kw_b});
    kw_r /= maxk;
    kw_g /= maxk;
    kw_b /= maxk;

    // Build diagonal color matrix
    res.matrix = {{{kw_r, 0, 0}, {0, kw_g, 0}, {0, 0, kw_b}}};
    res.n_stars_used = static_cast<int>(std::min({kr_vec.size(),
                                                   kg_vec.size(),
                                                   kb_vec.size()}));
    res.residual_rms = std::max({dev_r, dev_g, dev_b});

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

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float dr = R(y, x) - bg_r;
            float dg = G(y, x) - bg_g;
            float db = B(y, x) - bg_b;

            float nr = static_cast<float>(m[0][0] * dr + m[0][1] * dg + m[0][2] * db);
            float ng = static_cast<float>(m[1][0] * dr + m[1][1] * dg + m[1][2] * db);
            float nb = static_cast<float>(m[2][0] * dr + m[2][1] * dg + m[2][2] * db);

            R(y, x) = bg_out + nr;
            G(y, x) = bg_out + ng;
            B(y, x) = bg_out + nb;
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
