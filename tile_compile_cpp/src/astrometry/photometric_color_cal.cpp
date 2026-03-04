#include "tile_compile/astrometry/photometric_color_cal.hpp"
#include "tile_compile/image/background_extraction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace tile_compile::astrometry {

namespace {

namespace image = tile_compile::image;

struct PlaneFit {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double mx = 0.0;
    double my = 0.0;
    bool ok = false;
};

struct TileLookupCache {
    int bin_size = 64;
    int bins_x = 0;
    int bins_y = 0;
    bool valid = false;
    std::vector<std::vector<int>> bins;
};

static TileLookupCache build_tile_lookup_cache(const TileGrid &grid) {
    TileLookupCache cache;
    if (grid.tiles.empty()) return cache;

    int max_x = 0;
    int max_y = 0;
    for (const auto &t : grid.tiles) {
        max_x = std::max(max_x, t.x + t.width);
        max_y = std::max(max_y, t.y + t.height);
    }
    if (max_x <= 0 || max_y <= 0) return cache;

    cache.bins_x = std::max(1, (max_x + cache.bin_size - 1) / cache.bin_size);
    cache.bins_y = std::max(1, (max_y + cache.bin_size - 1) / cache.bin_size);
    cache.bins.assign(static_cast<size_t>(cache.bins_x * cache.bins_y), {});

    for (size_t i = 0; i < grid.tiles.size(); ++i) {
        const auto &t = grid.tiles[i];
        const int tx0 = std::max(0, t.x);
        const int ty0 = std::max(0, t.y);
        const int tx1 = std::max(tx0, t.x + t.width - 1);
        const int ty1 = std::max(ty0, t.y + t.height - 1);

        const int bx0 = std::clamp(tx0 / cache.bin_size, 0, cache.bins_x - 1);
        const int by0 = std::clamp(ty0 / cache.bin_size, 0, cache.bins_y - 1);
        const int bx1 = std::clamp(tx1 / cache.bin_size, 0, cache.bins_x - 1);
        const int by1 = std::clamp(ty1 / cache.bin_size, 0, cache.bins_y - 1);

        for (int by = by0; by <= by1; ++by) {
            for (int bx = bx0; bx <= bx1; ++bx) {
                cache.bins[static_cast<size_t>(by * cache.bins_x + bx)]
                    .push_back(static_cast<int>(i));
            }
        }
    }

    cache.valid = true;
    return cache;
}

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

static int find_tile_index_for_pixel(const TileGrid &grid, double px, double py,
                                     const TileLookupCache *cache = nullptr) {
    if (grid.tiles.empty()) return -1;
    const int x = static_cast<int>(std::lround(px));
    const int y = static_cast<int>(std::lround(py));

    auto tile_contains = [x, y](const Tile &t) {
        return x >= t.x && x < (t.x + t.width) && y >= t.y && y < (t.y + t.height);
    };

    if (cache != nullptr && cache->valid && x >= 0 && y >= 0) {
        const int bx = x / cache->bin_size;
        const int by = y / cache->bin_size;
        if (bx >= 0 && bx < cache->bins_x && by >= 0 && by < cache->bins_y) {
            const auto &bucket =
                cache->bins[static_cast<size_t>(by * cache->bins_x + bx)];
            for (int idx : bucket) {
                if (idx < 0 || idx >= static_cast<int>(grid.tiles.size())) continue;
                if (tile_contains(grid.tiles[static_cast<size_t>(idx)])) {
                    return idx;
                }
            }
        }
    }

    for (size_t i = 0; i < grid.tiles.size(); ++i) {
        const auto &t = grid.tiles[i];
        if (tile_contains(t)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static double tile_quality_weight_for_star(double px, double py,
                                           const PCCConfig &config,
                                           bool *reject_out,
                                           const TileLookupCache *tile_lookup = nullptr) {
    if (reject_out) *reject_out = false;
    if (!config.use_tile_quality_weighting) return 1.0;
    if (config.tile_metrics.empty() || config.tile_grid.tiles.empty()) return 1.0;
    if (config.tile_metrics.size() != config.tile_grid.tiles.size()) return 1.0;

    const int idx = find_tile_index_for_pixel(config.tile_grid, px, py, tile_lookup);
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

static double annulus_safe_fraction(const std::vector<uint8_t> &safe_mask,
                                    int rows, int cols,
                                    double cx, double cy,
                                    double r_in, double r_out) {
    if (rows <= 0 || cols <= 0 || safe_mask.empty()) return 0.0;
    const int x0 = std::max(0, static_cast<int>(cx - r_out - 1));
    const int x1 = std::min(cols - 1, static_cast<int>(cx + r_out + 1));
    const int y0 = std::max(0, static_cast<int>(cy - r_out - 1));
    const int y1 = std::min(rows - 1, static_cast<int>(cy + r_out + 1));
    const double r_in2 = r_in * r_in;
    const double r_out2 = r_out * r_out;

    int n_total = 0;
    int n_safe = 0;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const double dx = x - cx;
            const double dy = y - cy;
            const double d2 = dx * dx + dy * dy;
            if (d2 < r_in2 || d2 > r_out2) continue;
            ++n_total;
            const size_t idx = static_cast<size_t>(y * cols + x);
            if (idx < safe_mask.size() && safe_mask[idx] != 0) ++n_safe;
        }
    }
    return (n_total > 0) ? (static_cast<double>(n_safe) / static_cast<double>(n_total)) : 0.0;
}

static std::vector<uint8_t> build_common_support_mask_from_rgb(const Matrix2Df &R,
                                                                const Matrix2Df &G,
                                                                const Matrix2Df &B) {
    const int rows = R.rows();
    const int cols = R.cols();
    std::vector<uint8_t> mask(static_cast<size_t>(std::max(0, rows * cols)), 0);
    if (rows <= 0 || cols <= 0) return mask;

    // Canvas dead area is encoded as non-finite sentinel (and legacy all-zero).
    // Keep any finite non-all-zero pixel as potentially valid support.
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const float r = R(y, x);
            const float g = G(y, x);
            const float b = B(y, x);
            if (!(std::isfinite(r) && std::isfinite(g) && std::isfinite(b))) continue;
            if (r == 0.0f && g == 0.0f && b == 0.0f) continue;
            mask[static_cast<size_t>(y * cols + x)] = 1;
        }
    }
    return mask;
}

static double radial_support_fraction(const std::vector<uint8_t> &mask,
                                      int rows, int cols,
                                      double cx, double cy,
                                      double r_inner, double r_outer) {
    if (rows <= 0 || cols <= 0 || mask.empty()) return 0.0;
    const int x0 = std::max(0, static_cast<int>(cx - r_outer - 1));
    const int x1 = std::min(cols - 1, static_cast<int>(cx + r_outer + 1));
    const int y0 = std::max(0, static_cast<int>(cy - r_outer - 1));
    const int y1 = std::min(rows - 1, static_cast<int>(cy + r_outer + 1));
    const double r_in2 = std::max(0.0, r_inner * r_inner);
    const double r_out2 = std::max(r_in2, r_outer * r_outer);

    int n_total = 0;
    int n_ok = 0;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const double dx = x - cx;
            const double dy = y - cy;
            const double d2 = dx * dx + dy * dy;
            if (d2 < r_in2 || d2 > r_out2) continue;
            ++n_total;
            const size_t idx = static_cast<size_t>(y * cols + x);
            if (idx < mask.size() && mask[idx] != 0) ++n_ok;
        }
    }
    return (n_total > 0) ? (static_cast<double>(n_ok) / static_cast<double>(n_total)) : 0.0;
}

static ColorMatrix blend_matrix_with_identity_per_channel(
    const ColorMatrix &m, double alpha_r, double alpha_b) {
    alpha_r = std::clamp(alpha_r, 0.0, 1.0);
    alpha_b = std::clamp(alpha_b, 0.0, 1.0);
    // Use a neutral diagonal anchor at the current green scale (m[1][1]).
    // This preserves color-neutral fallback behavior in linear mode:
    // alpha=0 => diag(kg, kg, kg) instead of [1, kg, 1], avoiding red/magenta bias.
    const double kg = (std::isfinite(m[1][1]) && m[1][1] > 0.0) ? m[1][1] : 1.0;
    ColorMatrix out = {{{kg, 0, 0}, {0, kg, 0}, {0, 0, kg}}};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double ident = 0.0;
            if (r == 0 && c == 0) {
                ident = kg;
            } else if (r == 1 && c == 1) {
                ident = kg;
            } else if (r == 2 && c == 2) {
                ident = kg;
            }
            double a = 1.0;
            if (r == 0) {
                a = alpha_r;
            } else if (r == 2) {
                a = alpha_b;
            }
            out[r][c] = ident + a * (m[r][c] - ident);
        }
    }
    return out;
}

static void update_result_matrix_metrics(PCCResult *res) {
    const double m00 = res->matrix[0][0];
    const double m11 = res->matrix[1][1];
    const double m22 = res->matrix[2][2];
    res->determinant = m00 * m11 * m22;
    const double s0 = std::abs(m00);
    const double s1 = std::abs(m11);
    const double s2 = std::abs(m22);
    const double s_max = std::max({s0, s1, s2});
    const double s_min = std::min({s0, s1, s2});
    res->condition_number =
        (s_min > 1.0e-12) ? (s_max / s_min)
                          : std::numeric_limits<double>::infinity();
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
    const std::vector<uint8_t> bg_safe_mask =
        image::build_chroma_background_mask_from_rgb(R, G, B);
    int bg_safe_count = 0;
    for (uint8_t v : bg_safe_mask) {
        if (v != 0) ++bg_safe_count;
    }
    const double bg_safe_fraction =
        bg_safe_mask.empty()
            ? 0.0
            : static_cast<double>(bg_safe_count) / static_cast<double>(bg_safe_mask.size());
    const double min_safe_annulus_fraction =
        std::clamp(0.25 + 0.50 * bg_safe_fraction, 0.25, 0.55);
    std::vector<uint8_t> common_support_mask;
    if (config.common_mask_rows == rows &&
        config.common_mask_cols == cols &&
        config.common_valid_mask.size() == static_cast<size_t>(rows * cols)) {
        common_support_mask = config.common_valid_mask;
    } else {
        common_support_mask = build_common_support_mask_from_rgb(R, G, B);
    }
    int common_support_count = 0;
    for (uint8_t v : common_support_mask) {
        if (v != 0) ++common_support_count;
    }
    const double common_support_fraction =
        common_support_mask.empty()
            ? 0.0
            : static_cast<double>(common_support_count) /
                  static_cast<double>(common_support_mask.size());
    const double min_aperture_common_fraction =
        std::clamp(config.min_aperture_common_fraction, 0.50, 1.0);
    const double min_annulus_common_fraction =
        std::clamp(config.min_annulus_common_fraction, 0.30, 1.0);
    const TileLookupCache tile_lookup =
        (config.use_tile_quality_weighting &&
         !config.tile_metrics.empty() &&
         config.tile_metrics.size() == config.tile_grid.tiles.size())
            ? build_tile_lookup_cache(config.tile_grid)
            : TileLookupCache{};
    const TileLookupCache *tile_lookup_ptr = tile_lookup.valid ? &tile_lookup : nullptr;
    auto run_pass = [&](bool enable_bg_guard,
                        std::vector<StarPhotometry>* out_result,
                        int* n_sat_rejected,
                        int* n_tile_rejected,
                        int* n_common_rejected,
                        int* n_bg_mask_rejected,
                        std::vector<double>* tile_weights_used) {
        out_result->clear();
        out_result->reserve(catalog_stars.size());
        *n_sat_rejected = 0;
        *n_tile_rejected = 0;
        *n_common_rejected = 0;
        *n_bg_mask_rejected = 0;
        tile_weights_used->clear();
        tile_weights_used->reserve(catalog_stars.size());

        for (const auto &star : catalog_stars) {
            if (star.mag > config.mag_limit || star.mag < config.mag_bright_limit) continue;

            double px, py;
            if (!wcs.sky_to_pixel(star.ra, star.dec, px, py)) continue;
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

            const bool near_sat_r = std::isfinite(sat_r) && sat_r > 0.0f &&
                                    peak_r >= sat_guard_frac * sat_r;
            const bool near_sat_g = std::isfinite(sat_g) && sat_g > 0.0f &&
                                    peak_g >= sat_guard_frac * sat_g;
            const bool near_sat_b = std::isfinite(sat_b) && sat_b > 0.0f &&
                                    peak_b >= sat_guard_frac * sat_b;
            if (near_sat_r || near_sat_g || near_sat_b) {
                ++(*n_sat_rejected);
                continue;
            }

            StarPhotometry sp;
            sp.ra = star.ra;
            sp.dec = star.dec;
            sp.px = px;
            sp.py = py;
            sp.mag = star.mag;

            bool tile_reject = false;
            sp.quality_weight =
                tile_quality_weight_for_star(px, py, config, &tile_reject,
                                             tile_lookup_ptr);
            if (tile_reject || !(std::isfinite(sp.quality_weight) && sp.quality_weight > 0.0)) {
                ++(*n_tile_rejected);
                continue;
            }

            const double aperture_common_fraction = radial_support_fraction(
                common_support_mask, rows, cols, px, py, 0.0,
                config.aperture_radius_px);
            const double annulus_common_fraction = radial_support_fraction(
                common_support_mask, rows, cols, px, py,
                config.annulus_inner_px, config.annulus_outer_px);
            if (aperture_common_fraction < min_aperture_common_fraction ||
                annulus_common_fraction < min_annulus_common_fraction) {
                ++(*n_common_rejected);
                continue;
            }

            if (enable_bg_guard) {
                const double safe_fraction = annulus_safe_fraction(
                    bg_safe_mask, rows, cols, px, py,
                    config.annulus_inner_px, config.annulus_outer_px);
                if (safe_fraction < min_safe_annulus_fraction) {
                    ++(*n_bg_mask_rejected);
                    continue;
                }
            }

            sp.flux_r = aperture_flux(R, px, py, config.aperture_radius_px,
                                      config.annulus_inner_px, config.annulus_outer_px,
                                      config.background_model);
            sp.flux_g = aperture_flux(G, px, py, config.aperture_radius_px,
                                      config.annulus_inner_px, config.annulus_outer_px,
                                      config.background_model);
            sp.flux_b = aperture_flux(B, px, py, config.aperture_radius_px,
                                      config.annulus_inner_px, config.annulus_outer_px,
                                      config.background_model);

            double teff = star.teff;
            if (teff <= 0 && !star.xp_flux.empty())
                teff = estimate_teff_from_xp(star.xp_flux);
            if (teff > 0) {
                teff_to_rgb(teff, sp.cat_r, sp.cat_g, sp.cat_b);
            } else {
                sp.cat_r = sp.cat_g = sp.cat_b = 0.0;
            }

            sp.valid = (sp.flux_r > 0 && sp.flux_g > 0 && sp.flux_b > 0 &&
                        sp.cat_r > 0 && sp.cat_g > 0 && sp.cat_b > 0);
            if (sp.valid) {
                tile_weights_used->push_back(sp.quality_weight);
            }
            out_result->push_back(sp);
        }
    };

    std::vector<StarPhotometry> first_pass_result;
    std::vector<double> first_pass_weights;
    int first_sat_rej = 0;
    int first_tile_rej = 0;
    int first_common_rej = 0;
    int first_bg_rej = 0;
    run_pass(true, &first_pass_result,
             &first_sat_rej, &first_tile_rej, &first_common_rej, &first_bg_rej,
             &first_pass_weights);

    auto count_valid = [](const std::vector<StarPhotometry>& stars) {
        int n = 0;
        for (const auto& s : stars) {
            if (s.valid) ++n;
        }
        return n;
    };

    int n_sat_rejected = first_sat_rej;
    int n_tile_rejected = first_tile_rej;
    int n_common_rejected = first_common_rej;
    int n_bg_mask_rejected = first_bg_rej;
    std::vector<double> tile_weights_used = first_pass_weights;
    result = std::move(first_pass_result);
    bool used_bg_guard = true;

    if (count_valid(result) < config.min_stars && first_bg_rej > 0) {
        std::vector<StarPhotometry> relaxed_result;
        std::vector<double> relaxed_weights;
        int relaxed_sat_rej = 0;
        int relaxed_tile_rej = 0;
        int relaxed_common_rej = 0;
        int relaxed_bg_rej = 0;
        run_pass(false, &relaxed_result,
                 &relaxed_sat_rej, &relaxed_tile_rej, &relaxed_common_rej,
                 &relaxed_bg_rej, &relaxed_weights);
        if (count_valid(relaxed_result) > count_valid(result)) {
            result = std::move(relaxed_result);
            tile_weights_used = std::move(relaxed_weights);
            n_sat_rejected = relaxed_sat_rej;
            n_tile_rejected = relaxed_tile_rej;
            n_common_rejected = relaxed_common_rej;
            n_bg_mask_rejected = relaxed_bg_rej;
            used_bg_guard = false;
        }
    }

    std::cout << "[PCC] Saturation guard: rejected=" << n_sat_rejected
              << " sat_r=" << sat_r << " sat_g=" << sat_g << " sat_b=" << sat_b
              << " frac=" << sat_guard_frac << std::endl;
    std::cout << "[PCC] Common-support guard: rejected="
              << n_common_rejected
              << " min_ap=" << min_aperture_common_fraction
              << " min_ann=" << min_annulus_common_fraction
              << " support_fraction=" << common_support_fraction << std::endl;
    std::cout << "[PCC] Background-safe annulus guard: rejected="
              << n_bg_mask_rejected
              << " min_safe_fraction=" << min_safe_annulus_fraction
              << " bg_safe_mask_fraction=" << bg_safe_fraction
              << " enabled=" << (used_bg_guard ? "true" : "false") << std::endl;
    if (!tile_weights_used.empty()) {
        double w_sum = 0.0;
        double w_min = std::numeric_limits<double>::infinity();
        double w_max = 0.0;
        for (double w : tile_weights_used) {
            w_sum += w;
            w_min = std::min(w_min, w);
            w_max = std::max(w_max, w);
        }
        std::cout << "[PCC] Tile-quality weighting: enabled="
                  << (config.use_tile_quality_weighting ? "true" : "false")
                  << " rejected=" << n_tile_rejected
                  << " mean=" << (w_sum / static_cast<double>(tile_weights_used.size()))
                  << " min=" << w_min
                  << " max=" << w_max << std::endl;
    } else if (config.use_tile_quality_weighting) {
        std::cout << "[PCC] Tile-quality weighting: enabled=true rejected="
                  << n_tile_rejected << " no valid weighted stars" << std::endl;
    }

    return result;
}

// ─── Color matrix fitting (Siril SPCC-inspired) ─────────────────────────
//
// Inspired by Siril's get_spcc_white_balance_coeffs:
//   1. Per star: measure image flux ratios (r/g, b/g) and catalog flux ratios
//   2. Robust linear fit: image_rg = a + b * catalog_rg  (repeated median)
//   3. Evaluate fit at a white reference
//   4. Normalize so max(kr, kg=1, kb) = 1
//
// Note: Siril SPCC evaluates at a selected white reference spectrum. Here we
// derive an adaptive white reference from the catalog colors present in-frame.
//
// The repeated median fit (Siegel 1982) is breakdown-point 0.5 and
// handles both slope and intercept robustly unlike simple ratio medians.

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

static double simple_median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2 == 1) return v[n / 2];
    const double hi = v[n / 2];
    std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
    return 0.5 * (v[n / 2 - 1] + hi);
}

// Siegel's repeated median estimator: robust linear fit y = a + b*x.
// Breakdown point 0.5. Returns false if insufficient data.
// deviation_out = MAD of residuals after fit.
static bool repeated_median_fit(const std::vector<double> &x,
                                 const std::vector<double> &y,
                                 double &a_out, double &b_out,
                                 double &deviation_out) {
    const size_t n = x.size();
    deviation_out = 0.0;
    if (n < 3 || x.size() != y.size()) return false;

    // For each i: median over j≠i of (y[j]-y[i])/(x[j]-x[i])
    std::vector<double> slopes_i;
    slopes_i.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> slopes_j;
        slopes_j.reserve(n - 1);
        for (size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const double dx = x[j] - x[i];
            if (std::abs(dx) < 1.0e-15) continue;
            slopes_j.push_back((y[j] - y[i]) / dx);
        }
        if (slopes_j.empty()) continue;
        slopes_i.push_back(simple_median(slopes_j));
    }
    if (slopes_i.size() < 3) return false;

    b_out = simple_median(slopes_i);

    // Intercept: median of (y[i] - b*x[i])
    std::vector<double> intercepts;
    intercepts.reserve(n);
    for (size_t i = 0; i < n; ++i)
        intercepts.push_back(y[i] - b_out * x[i]);
    a_out = simple_median(intercepts);

    // MAD of residuals
    std::vector<double> resid;
    resid.reserve(n);
    for (size_t i = 0; i < n; ++i)
        resid.push_back(std::abs(y[i] - (a_out + b_out * x[i])));
    deviation_out = 1.4826 * simple_median(resid);
    return true;
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

// Recompute robust log-chroma residuals for the actually applied PCC matrix.
static double recompute_residual_rms_for_matrix(const std::vector<StarPhotometry> &stars,
                                                const ColorMatrix &matrix,
                                                double sigma_clip,
                                                int *n_used_out) {
    if (n_used_out != nullptr) *n_used_out = 0;

    std::vector<double> d_rg_vec;
    std::vector<double> d_bg_vec;
    std::vector<double> w_rg_vec;
    std::vector<double> w_bg_vec;
    d_rg_vec.reserve(stars.size());
    d_bg_vec.reserve(stars.size());
    w_rg_vec.reserve(stars.size());
    w_bg_vec.reserve(stars.size());

    for (const auto &s : stars) {
        if (!(s.valid &&
              s.flux_r > 0.0 && s.flux_g > 0.0 && s.flux_b > 0.0 &&
              s.cat_r > 0.0 && s.cat_g > 0.0 && s.cat_b > 0.0)) {
            continue;
        }

        const double fr = s.flux_r;
        const double fg = s.flux_g;
        const double fb = s.flux_b;
        const double mr = matrix[0][0] * fr + matrix[0][1] * fg + matrix[0][2] * fb;
        const double mg = matrix[1][0] * fr + matrix[1][1] * fg + matrix[1][2] * fb;
        const double mb = matrix[2][0] * fr + matrix[2][1] * fg + matrix[2][2] * fb;
        if (!(std::isfinite(mr) && std::isfinite(mg) && std::isfinite(mb) &&
              mr > 0.0 && mg > 0.0 && mb > 0.0)) {
            continue;
        }

        const double meas_rg = mr / mg;
        const double meas_bg = mb / mg;
        const double cat_rg = s.cat_r / s.cat_g;
        const double cat_bg = s.cat_b / s.cat_g;
        if (!(meas_rg > 0.0 && meas_bg > 0.0 && cat_rg > 0.0 && cat_bg > 0.0)) {
            continue;
        }

        const double w = std::clamp(s.quality_weight, 1.0e-3, 10.0);
        d_rg_vec.push_back(std::log(cat_rg) - std::log(meas_rg));
        d_bg_vec.push_back(std::log(cat_bg) - std::log(meas_bg));
        w_rg_vec.push_back(w);
        w_bg_vec.push_back(w);
    }

    if (d_rg_vec.size() < 3 || d_bg_vec.size() < 3) {
        return std::numeric_limits<double>::infinity();
    }

    double dev_rg = 0.0;
    double dev_bg = 0.0;
    (void)robust_mean_weighted(d_rg_vec, w_rg_vec, sigma_clip, dev_rg);
    (void)robust_mean_weighted(d_bg_vec, w_bg_vec, sigma_clip, dev_bg);
    if (n_used_out != nullptr) {
        *n_used_out = static_cast<int>(std::min(d_rg_vec.size(), d_bg_vec.size()));
    }
    return std::max(dev_rg, dev_bg);
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

    // Siril-like robust ratio fit:
    // fit image ratios vs catalog ratios, then evaluate at an adaptive
    // white reference derived from the catalog colors present in this frame.
    std::vector<double> cat_rg_vec, img_rg_vec;
    std::vector<double> cat_bg_vec, img_bg_vec;
    std::vector<double> w_vec;
    cat_rg_vec.reserve(valid.size());
    img_rg_vec.reserve(valid.size());
    cat_bg_vec.reserve(valid.size());
    img_bg_vec.reserve(valid.size());
    w_vec.reserve(valid.size());

    for (const auto *s : valid) {
        if (!(s->flux_r > 0.0 && s->flux_g > 0.0 && s->flux_b > 0.0 &&
              s->cat_r > 0.0 && s->cat_g > 0.0 && s->cat_b > 0.0)) {
            continue;
        }
        const double meas_rg = s->flux_r / s->flux_g;
        const double meas_bg = s->flux_b / s->flux_g;
        const double c_rg = s->cat_r / s->cat_g;
        const double c_bg = s->cat_b / s->cat_g;
        if (!(std::isfinite(meas_rg) && meas_rg > 0.0 &&
              std::isfinite(meas_bg) && meas_bg > 0.0 &&
              std::isfinite(c_rg) && c_rg > 0.0 &&
              std::isfinite(c_bg) && c_bg > 0.0)) {
            continue;
        }
        const double w = std::clamp(s->quality_weight, 1.0e-3, 10.0);
        cat_rg_vec.push_back(c_rg);
        img_rg_vec.push_back(meas_rg);
        cat_bg_vec.push_back(c_bg);
        img_bg_vec.push_back(meas_bg);
        w_vec.push_back(w);
    }

    if (cat_rg_vec.size() < static_cast<size_t>(config.min_stars)) {
        res.error_message = "Not enough valid stars for adaptive-ratio PCC fit";
        return res;
    }

    double a_rg = 0.0, b_rg = 1.0, dev_rg = 0.0;
    double a_bg = 0.0, b_bg = 1.0, dev_bg = 0.0;
    if (!repeated_median_fit(cat_rg_vec, img_rg_vec, a_rg, b_rg, dev_rg)) {
        res.error_message = "repeated_median_fit failed for R/G";
        return res;
    }
    if (!repeated_median_fit(cat_bg_vec, img_bg_vec, a_bg, b_bg, dev_bg)) {
        res.error_message = "repeated_median_fit failed for B/G";
        return res;
    }

    double wrg = weighted_median(cat_rg_vec, w_vec);
    double wbg = weighted_median(cat_bg_vec, w_vec);
    if (!(std::isfinite(wrg) && wrg > 0.0)) wrg = 1.0;
    if (!(std::isfinite(wbg) && wbg > 0.0)) wbg = 1.0;

    double kw_r = 1.0 / (a_rg + b_rg * wrg);
    double kw_g = 1.0;
    double kw_b = 1.0 / (a_bg + b_bg * wbg);
    if (!std::isfinite(kw_r) || kw_r <= 0.0 ||
        !std::isfinite(kw_b) || kw_b <= 0.0) {
        res.error_message = "Degenerate PCC fit: non-positive adaptive-ratio scale factor";
        return res;
    }

    const double raw_k_max = std::max(1.0, static_cast<double>(config.k_max));
    kw_r = std::min(kw_r, raw_k_max);
    kw_g = std::min(kw_g, raw_k_max);
    kw_b = std::min(kw_b, raw_k_max);

    const double kw_max = std::max({kw_r, kw_g, kw_b});
    kw_r = std::max(1.0e-3, kw_r / kw_max);
    kw_g = std::max(1.0e-3, kw_g / kw_max);
    kw_b = std::max(1.0e-3, kw_b / kw_max);

    std::cout << "[PCC] Adaptive white reference: wrg=" << wrg
              << " wbg=" << wbg << std::endl;
    std::cout << "[PCC] Repeated-median fit R/G: a=" << a_rg
              << " b=" << b_rg << " dev=" << dev_rg << std::endl;
    std::cout << "[PCC] Repeated-median fit B/G: a=" << a_bg
              << " b=" << b_bg << " dev=" << dev_bg << std::endl;
    std::cout << "[PCC] Scale factors raw (before norm): R=" << (kw_r * kw_max)
              << " G=" << (kw_g * kw_max) << " B=" << (kw_b * kw_max)
              << std::endl;
    std::cout << "[PCC] Scale factors after norm: R=" << kw_r
              << " G=" << kw_g << " B=" << kw_b
              << " (raw_k_max=" << raw_k_max << ")" << std::endl;

    res.matrix = {{{kw_r, 0, 0}, {0, kw_g, 0}, {0, 0, kw_b}}};
    res.n_stars_used = static_cast<int>(cat_rg_vec.size());
    int residual_used = 0;
    res.residual_rms =
        recompute_residual_rms_for_matrix(stars, res.matrix, config.sigma_clip, &residual_used);
    if (residual_used > 0) {
        res.n_stars_used = residual_used;
    }
    res.determinant = kw_r * kw_g * kw_b;
    const double s0 = std::abs(kw_r);
    const double s1 = std::abs(kw_g);
    const double s2 = std::abs(kw_b);
    const double s_max = std::max({s0, s1, s2});
    const double s_min = std::min({s0, s1, s2});
    res.condition_number =
        (s_min > 1.0e-12) ? (s_max / s_min) : std::numeric_limits<double>::infinity();

    std::cout << "[PCC] Matrix stability: det=" << res.determinant
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
    std::cout << "[PCC] Scale factors: R=" << kw_r
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

constexpr float kShadowAttenFloor = 0.10f;
constexpr float kHighlightAttenFloor = 0.25f;

static std::vector<float> box_blur_with_valid_mask(const std::vector<float> &src,
                                                   const std::vector<uint8_t> &valid,
                                                   int rows, int cols,
                                                   int radius) {
    std::vector<float> out(src.size(), 0.0f);
    if (rows <= 0 || cols <= 0 || src.size() != valid.size() ||
        src.size() != static_cast<size_t>(rows * cols)) {
        return out;
    }
    radius = std::max(0, radius);
    if (radius == 0) return src;

    const int stride = cols + 1;
    std::vector<double> isum(static_cast<size_t>((rows + 1) * (cols + 1)), 0.0);
    std::vector<int> icnt(static_cast<size_t>((rows + 1) * (cols + 1)), 0);
    auto idx = [stride](int y, int x) { return static_cast<size_t>(y * stride + x); };

    for (int y = 0; y < rows; ++y) {
        double row_sum = 0.0;
        int row_cnt = 0;
        for (int x = 0; x < cols; ++x) {
            const size_t p = static_cast<size_t>(y * cols + x);
            if (valid[p] != 0) {
                row_sum += static_cast<double>(src[p]);
                ++row_cnt;
            }
            const size_t q = idx(y + 1, x + 1);
            isum[q] = isum[idx(y, x + 1)] + row_sum;
            icnt[q] = icnt[idx(y, x + 1)] + row_cnt;
        }
    }

    for (int y = 0; y < rows; ++y) {
        const int y0 = std::max(0, y - radius);
        const int y1 = std::min(rows - 1, y + radius);
        for (int x = 0; x < cols; ++x) {
            const int x0 = std::max(0, x - radius);
            const int x1 = std::min(cols - 1, x + radius);
            const size_t p11 = idx(y1 + 1, x1 + 1);
            const size_t p01 = idx(y0, x1 + 1);
            const size_t p10 = idx(y1 + 1, x0);
            const size_t p00 = idx(y0, x0);
            const double sum = isum[p11] - isum[p01] - isum[p10] + isum[p00];
            const int cnt = icnt[p11] - icnt[p01] - icnt[p10] + icnt[p00];
            const size_t p = static_cast<size_t>(y * cols + x);
            if (cnt > 0) {
                out[p] = static_cast<float>(sum / static_cast<double>(cnt));
            } else {
                out[p] = src[p];
            }
        }
    }
    return out;
}

struct PCCAttenuationContext {
    int rows = 0;
    int cols = 0;
    float bg_r = 0.0f;
    float bg_g = 0.0f;
    float bg_b = 0.0f;
    float bg_out = 0.0f;
    float shadow_lo = 0.0f;
    float shadow_hi = 0.0f;
    float blend_lo = 0.0f;
    float blend_hi = 0.0f;
    std::vector<float> luma_smooth;
};

static PCCAttenuationContext build_pcc_attenuation_context(const Matrix2Df &R,
                                                           const Matrix2Df &G,
                                                           const Matrix2Df &B) {
    PCCAttenuationContext ctx;
    ctx.rows = R.rows();
    ctx.cols = R.cols();
    if (ctx.rows <= 0 || ctx.cols <= 0) return ctx;

    ctx.bg_r = estimate_background(R);
    ctx.bg_g = estimate_background(G);
    ctx.bg_b = estimate_background(B);
    ctx.bg_out = (ctx.bg_r + ctx.bg_g + ctx.bg_b) / 3.0f;

    std::vector<float> signal_luma;
    signal_luma.reserve((ctx.rows / 4 + 1) * (ctx.cols / 4 + 1));
    for (int y = 0; y < ctx.rows; y += 4) {
        for (int x = 0; x < ctx.cols; x += 4) {
            const float r0 = R(y, x);
            const float g0 = G(y, x);
            const float b0 = B(y, x);
            if (!(std::isfinite(r0) && r0 > 0.0f &&
                  std::isfinite(g0) && g0 > 0.0f &&
                  std::isfinite(b0) && b0 > 0.0f)) {
                continue;
            }
            const float dr = r0 - ctx.bg_out;
            const float dg = g0 - ctx.bg_out;
            const float db = b0 - ctx.bg_out;
            const float luma =
                std::max(0.0f, 0.2126f * dr + 0.7152f * dg + 0.0722f * db);
            if (luma > 0.0f) signal_luma.push_back(luma);
        }
    }

    if (signal_luma.size() >= 32) {
        std::sort(signal_luma.begin(), signal_luma.end());
        ctx.shadow_lo = percentile_sorted(signal_luma, 0.01f);
        ctx.shadow_hi = percentile_sorted(signal_luma, 0.20f);
        ctx.blend_lo = percentile_sorted(signal_luma, 0.90f);
        ctx.blend_hi = percentile_sorted(signal_luma, 0.995f);
        if (!(std::isfinite(ctx.shadow_lo) && std::isfinite(ctx.shadow_hi) &&
              ctx.shadow_hi > ctx.shadow_lo)) {
            ctx.shadow_lo = 0.0f;
            ctx.shadow_hi = 0.0f;
        }
        if (!(std::isfinite(ctx.blend_lo) && std::isfinite(ctx.blend_hi) &&
              ctx.blend_hi > ctx.blend_lo)) {
            ctx.blend_lo = 0.0f;
            ctx.blend_hi = 0.0f;
        }
    }

    std::vector<float> luma_raw(static_cast<size_t>(ctx.rows * ctx.cols), 0.0f);
    std::vector<uint8_t> luma_valid(static_cast<size_t>(ctx.rows * ctx.cols), 0);
    for (int y = 0; y < ctx.rows; ++y) {
        for (int x = 0; x < ctx.cols; ++x) {
            const size_t p = static_cast<size_t>(y * ctx.cols + x);
            const float r0 = R(y, x);
            const float g0 = G(y, x);
            const float b0 = B(y, x);
            if (!(std::isfinite(r0) && r0 > 0.0f &&
                  std::isfinite(g0) && g0 > 0.0f &&
                  std::isfinite(b0) && b0 > 0.0f)) {
                continue;
            }
            const float dr = r0 - ctx.bg_out;
            const float dg = g0 - ctx.bg_out;
            const float db = b0 - ctx.bg_out;
            luma_raw[p] = std::max(0.0f, 0.2126f * dr + 0.7152f * dg + 0.0722f * db);
            luma_valid[p] = 1;
        }
    }
    ctx.luma_smooth = box_blur_with_valid_mask(luma_raw, luma_valid, ctx.rows, ctx.cols, 12);
    return ctx;
}

static float attenuation_from_luma(float luma, const PCCAttenuationContext &ctx) {
    float atten_shadows = 1.0f;
    if (ctx.shadow_hi > ctx.shadow_lo && luma < ctx.shadow_hi) {
        float t = (luma - ctx.shadow_lo) / (ctx.shadow_hi - ctx.shadow_lo);
        t = std::clamp(t, 0.0f, 1.0f);
        const float s = t * t * (3.0f - 2.0f * t);
        atten_shadows = kShadowAttenFloor + (1.0f - kShadowAttenFloor) * s;
    }

    float atten_highlights = 1.0f;
    if (ctx.blend_hi > ctx.blend_lo && luma > ctx.blend_lo) {
        float t = (luma - ctx.blend_lo) / (ctx.blend_hi - ctx.blend_lo);
        t = std::clamp(t, 0.0f, 1.0f);
        const float s = t * t * (3.0f - 2.0f * t);
        atten_highlights = 1.0f - (1.0f - kHighlightAttenFloor) * s;
    }
    return std::min(atten_shadows, atten_highlights);
}

static inline void apply_color_matrix_to_deltas(const ColorMatrix &m, float atten,
                                                float dr, float dg, float db,
                                                float *nr, float *ng, float *nb) {
    const float m00 = static_cast<float>(1.0 + atten * (m[0][0] - 1.0));
    const float m01 = static_cast<float>(atten * m[0][1]);
    const float m02 = static_cast<float>(atten * m[0][2]);
    const float m10 = static_cast<float>(atten * m[1][0]);
    const float m11 = static_cast<float>(1.0 + atten * (m[1][1] - 1.0));
    const float m12 = static_cast<float>(atten * m[1][2]);
    const float m20 = static_cast<float>(atten * m[2][0]);
    const float m21 = static_cast<float>(atten * m[2][1]);
    const float m22 = static_cast<float>(1.0 + atten * (m[2][2] - 1.0));
    *nr = m00 * dr + m01 * dg + m02 * db;
    *ng = m10 * dr + m11 * dg + m12 * db;
    *nb = m20 * dr + m21 * dg + m22 * db;
}

struct PCCBackgroundSample {
    float dr, dg, db;
    float bg_out;
    float atten;
};

static std::vector<PCCBackgroundSample> build_pcc_background_samples(
    const Matrix2Df &R, const Matrix2Df &G, const Matrix2Df &B,
    const std::vector<uint8_t> &bg_mask,
    const PCCAttenuationContext &ctx) {
    std::vector<PCCBackgroundSample> samples;
    if (ctx.rows <= 0 || ctx.cols <= 0 ||
        bg_mask.size() != static_cast<size_t>(ctx.rows * ctx.cols) ||
        ctx.luma_smooth.size() != static_cast<size_t>(ctx.rows * ctx.cols)) {
        return samples;
    }

    size_t bg_count = 0;
    for (uint8_t v : bg_mask) {
        if (v != 0) ++bg_count;
    }
    if (bg_count == 0) return samples;

    constexpr size_t kTargetSamples = 120000;
    size_t stride = 1;
    if (bg_count > kTargetSamples) {
        stride = static_cast<size_t>(
            std::ceil(std::sqrt(static_cast<double>(bg_count) /
                                static_cast<double>(kTargetSamples))));
        stride = std::max<size_t>(1, stride);
    }

    samples.reserve(std::min(bg_count, kTargetSamples));
    for (int y = 0; y < ctx.rows; y += static_cast<int>(stride)) {
        for (int x = 0; x < ctx.cols; x += static_cast<int>(stride)) {
            const size_t idx = static_cast<size_t>(y * ctx.cols + x);
            if (bg_mask[idx] == 0) continue;
            const float r0 = R(y, x);
            const float g0 = G(y, x);
            const float b0 = B(y, x);
            if (!(std::isfinite(r0) && r0 > 0.0f &&
                  std::isfinite(g0) && g0 > 0.0f &&
                  std::isfinite(b0) && b0 > 0.0f)) {
                continue;
            }
            const float luma = ctx.luma_smooth[idx];
            samples.push_back(PCCBackgroundSample{
                r0 - ctx.bg_out, g0 - ctx.bg_out, b0 - ctx.bg_out,
                ctx.bg_out,
                attenuation_from_luma(luma, ctx)});
        }
    }

    if (samples.size() < 1024 && stride > 1) {
        samples.clear();
        for (int y = 0; y < ctx.rows; ++y) {
            for (int x = 0; x < ctx.cols; ++x) {
                const size_t idx = static_cast<size_t>(y * ctx.cols + x);
                if (bg_mask[idx] == 0) continue;
                const float r0 = R(y, x);
                const float g0 = G(y, x);
                const float b0 = B(y, x);
                if (!(std::isfinite(r0) && r0 > 0.0f &&
                      std::isfinite(g0) && g0 > 0.0f &&
                      std::isfinite(b0) && b0 > 0.0f)) {
                    continue;
                }
                const float luma = ctx.luma_smooth[idx];
                samples.push_back(PCCBackgroundSample{
                    r0 - ctx.bg_out, g0 - ctx.bg_out, b0 - ctx.bg_out,
                    ctx.bg_out,
                    attenuation_from_luma(luma, ctx)});
            }
        }
    }

    return samples;
}

struct PCCBackgroundStdPair {
    double rg_std = std::numeric_limits<double>::infinity();
    double bg_std = std::numeric_limits<double>::infinity();
};

static PCCBackgroundStdPair sampled_background_std_after_matrix(
    const std::vector<PCCBackgroundSample> &samples,
    const ColorMatrix &matrix) {
    PCCBackgroundStdPair out;
    if (samples.empty()) return out;

    size_t n_rg = 0;
    size_t n_bg = 0;
    double mean_rg = 0.0;
    double mean_bg = 0.0;
    double m2_rg = 0.0;
    double m2_bg = 0.0;

    for (const auto &s : samples) {
        float nr = 0.0f;
        float ng = 0.0f;
        float nb = 0.0f;
        apply_color_matrix_to_deltas(matrix, s.atten, s.dr, s.dg, s.db, &nr, &ng, &nb);

        const double rv = static_cast<double>(s.bg_out + nr);
        const double gv = static_cast<double>(s.bg_out + ng);
        const double bv = static_cast<double>(s.bg_out + nb);
        if (!(std::isfinite(gv) && gv > 0.0)) continue;

        if (std::isfinite(rv) && rv > 0.0) {
            const double v = std::log(rv / gv);
            ++n_rg;
            const double d = v - mean_rg;
            mean_rg += d / static_cast<double>(n_rg);
            m2_rg += d * (v - mean_rg);
        }
        if (std::isfinite(bv) && bv > 0.0) {
            const double v = std::log(bv / gv);
            ++n_bg;
            const double d = v - mean_bg;
            mean_bg += d / static_cast<double>(n_bg);
            m2_bg += d * (v - mean_bg);
        }
    }

    if (n_rg >= 512) out.rg_std = std::sqrt(m2_rg / static_cast<double>(n_rg));
    if (n_bg >= 512) out.bg_std = std::sqrt(m2_bg / static_cast<double>(n_bg));
    return out;
}

static bool estimate_channel_background_median(
    const Matrix2Df &ch, const std::vector<uint8_t> &bg_mask, double *median_out) {
    if (median_out == nullptr) return false;
    if (bg_mask.size() != static_cast<size_t>(ch.rows() * ch.cols())) return false;
    std::vector<float> samples;
    samples.reserve(bg_mask.size() / 4);
    for (int y = 0; y < ch.rows(); ++y) {
        for (int x = 0; x < ch.cols(); ++x) {
            const size_t idx = static_cast<size_t>(y * ch.cols() + x);
            if (bg_mask[idx] == 0) continue;
            const float v = ch(y, x);
            if (std::isfinite(v) && v > 0.0f) samples.push_back(v);
        }
    }
    if (samples.size() < 2048) return false;
    std::sort(samples.begin(), samples.end());
    *median_out = static_cast<double>(samples[samples.size() / 2]);
    return std::isfinite(*median_out);
}

static void neutralize_background_offsets(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B) {
    const std::vector<uint8_t> bg_mask =
        image::build_chroma_background_mask_from_rgb(R, G, B);
    if (bg_mask.empty()) return;

    double bg_r = 0.0;
    double bg_g = 0.0;
    double bg_b = 0.0;
    if (!estimate_channel_background_median(R, bg_mask, &bg_r) ||
        !estimate_channel_background_median(G, bg_mask, &bg_g) ||
        !estimate_channel_background_median(B, bg_mask, &bg_b)) {
        return;
    }

    const double bg_ref = (bg_r + bg_g + bg_b) / 3.0;
    const float dr = static_cast<float>(bg_ref - bg_r);
    const float dg = static_cast<float>(bg_ref - bg_g);
    const float db = static_cast<float>(bg_ref - bg_b);

    for (int y = 0; y < R.rows(); ++y) {
        for (int x = 0; x < R.cols(); ++x) {
            const float r = R(y, x);
            const float g = G(y, x);
            const float b = B(y, x);
            if (!(std::isfinite(r) && std::isfinite(g) && std::isfinite(b))) {
                // Preserve non-finite canvas sentinel exactly.
                continue;
            }
            if (r == 0.0f && g == 0.0f && b == 0.0f) {
                // Preserve legacy zero canvas marker.
                continue;
            }
            if (std::isfinite(r)) R(y, x) = std::max(0.0f, r + dr);
            if (std::isfinite(g)) G(y, x) = std::max(0.0f, g + dg);
            if (std::isfinite(b)) B(y, x) = std::max(0.0f, b + db);
        }
    }

    std::cout << "[PCC] Background neutralization applied: "
              << "medians R/G/B=" << bg_r << "/" << bg_g << "/" << bg_b
              << " -> target=" << bg_ref << std::endl;
}


static void apply_color_matrix_impl_simple(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
                                           const ColorMatrix &m, bool verbose) {
    const int rows = R.rows();
    const int cols = R.cols();
    if (rows <= 0 || cols <= 0) return;

    // Linear application with shared background anchoring.
    // Keeping a common background pivot avoids colored sky bias when
    // `apply_attenuation=false` and channel gains differ strongly.
    const PCCAttenuationContext ctx = build_pcc_attenuation_context(R, G, B);

    size_t valid_px = 0;
    const size_t total_px = static_cast<size_t>(rows) * static_cast<size_t>(cols);

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const float r0 = R(y, x);
            const float g0 = G(y, x);
            const float b0 = B(y, x);
            if (!(std::isfinite(r0) && std::isfinite(g0) && std::isfinite(b0))) {
                const float invalid = std::numeric_limits<float>::quiet_NaN();
                R(y, x) = invalid; G(y, x) = invalid; B(y, x) = invalid;
                continue;
            }
            if (r0 == 0.0f && g0 == 0.0f && b0 == 0.0f) {
                // Keep canvas dead area at zero in linear apply mode.
                continue;
            }
            ++valid_px;
            const float dr = r0 - ctx.bg_out;
            const float dg = g0 - ctx.bg_out;
            const float db = b0 - ctx.bg_out;
            float nr = 0.0f;
            float ng = 0.0f;
            float nb = 0.0f;
            apply_color_matrix_to_deltas(m, 1.0f, dr, dg, db, &nr, &ng, &nb);
            R(y, x) = (std::isfinite(nr) ? (ctx.bg_out + nr)
                                         : std::numeric_limits<float>::quiet_NaN());
            G(y, x) = (std::isfinite(ng) ? (ctx.bg_out + ng)
                                         : std::numeric_limits<float>::quiet_NaN());
            B(y, x) = (std::isfinite(nb) ? (ctx.bg_out + nb)
                                         : std::numeric_limits<float>::quiet_NaN());
        }
    }
    if (verbose) {
        const double frac = (total_px > 0) ? (static_cast<double>(valid_px) / static_cast<double>(total_px)) : 0.0;
        std::cerr << "[PCC] Apply(simple, anchored) valid pixels: " << valid_px << "/" << total_px
                  << " (" << (100.0 * frac) << "%)" << std::endl;
    }
}

static void apply_color_matrix_impl(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
                                    const ColorMatrix &m, bool verbose) {
    const PCCAttenuationContext ctx = build_pcc_attenuation_context(R, G, B);
    const int rows = ctx.rows;
    const int cols = ctx.cols;
    if (rows <= 0 || cols <= 0) return;

    if (verbose) {
        std::cout << "[PCC] Background levels: R=" << ctx.bg_r
                  << " G=" << ctx.bg_g << " B=" << ctx.bg_b
                  << " -> bg_out=" << ctx.bg_out << std::endl;
        std::cout << "[PCC] Shadow blend thresholds: lo=" << ctx.shadow_lo
                  << " hi=" << ctx.shadow_hi << std::endl;
        std::cout << "[PCC] Highlight blend thresholds: lo=" << ctx.blend_lo
                  << " hi=" << ctx.blend_hi << std::endl;
        std::cout << "[PCC] Shadow attenuation floor=" << kShadowAttenFloor << std::endl;
        std::cout << "[PCC] Highlight attenuation floor=" << kHighlightAttenFloor << std::endl;
    }

    size_t valid_px = 0;
    const size_t total_px = static_cast<size_t>(rows) * static_cast<size_t>(cols);

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const float r0 = R(y, x);
            const float g0 = G(y, x);
            const float b0 = B(y, x);
            if (!(std::isfinite(r0) && std::isfinite(g0) && std::isfinite(b0))) {
                const float invalid = std::numeric_limits<float>::quiet_NaN();
                R(y, x) = invalid;
                G(y, x) = invalid;
                B(y, x) = invalid;
                continue;
            }
            if (!(r0 > 0.0f && g0 > 0.0f && b0 > 0.0f)) {
                R(y, x) = 0.0f;
                G(y, x) = 0.0f;
                B(y, x) = 0.0f;
                continue;
            }

            ++valid_px;

            const float dr = r0 - ctx.bg_out;
            const float dg = g0 - ctx.bg_out;
            const float db = b0 - ctx.bg_out;
            const size_t idx = static_cast<size_t>(y * cols + x);
            const float atten = attenuation_from_luma(ctx.luma_smooth[idx], ctx);
            float nr = 0.0f;
            float ng = 0.0f;
            float nb = 0.0f;
            apply_color_matrix_to_deltas(m, atten, dr, dg, db, &nr, &ng, &nb);
            R(y, x) = std::isfinite(nr) ? (ctx.bg_out + nr)
                                        : std::numeric_limits<float>::quiet_NaN();
            G(y, x) = std::isfinite(ng) ? (ctx.bg_out + ng)
                                        : std::numeric_limits<float>::quiet_NaN();
            B(y, x) = std::isfinite(nb) ? (ctx.bg_out + nb)
                                        : std::numeric_limits<float>::quiet_NaN();
        }
    }

    const size_t skipped_px = (total_px >= valid_px) ? (total_px - valid_px) : 0;
    const double frac = (total_px > 0) ? (static_cast<double>(valid_px) / static_cast<double>(total_px)) : 0.0;
    if (verbose) {
        std::cout << "[PCC] Apply valid pixels: " << valid_px << "/" << total_px
                  << " (" << (100.0 * frac) << "%)  skipped=" << skipped_px
                  << std::endl;
    }
}

void apply_color_matrix(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
                        const ColorMatrix &m, bool apply_attenuation) {
    if (apply_attenuation) {
        apply_color_matrix_impl(R, G, B, m, true);
    } else {
        apply_color_matrix_impl_simple(R, G, B, m, true);
    }
}

// ─── Full PCC pipeline ──────────────────────────────────────────────────

PCCResult run_pcc(Matrix2Df &R, Matrix2Df &G, Matrix2Df &B,
                  const WCS &wcs,
                  const std::vector<GaiaStar> &catalog_stars,
                  const PCCConfig &config) {

    std::cout << "[PCC] Measuring " << catalog_stars.size()
              << " catalog stars in image..." << std::endl;

    auto photometry = measure_stars(R, G, B, wcs, catalog_stars, config);

    int n_valid = 0;
    for (const auto &s : photometry) if (s.valid) ++n_valid;
    std::cout << "[PCC] " << n_valid << "/" << photometry.size()
              << " stars with valid photometry" << std::endl;

    auto result = fit_color_matrix(photometry, config);

    if (result.success) {
        std::cout << "[PCC] Color matrix fit: " << result.n_stars_used
                  << " stars, RMS=" << result.residual_rms << std::endl;
        std::cout << "[PCC] Matrix:" << std::endl;
        for (int i = 0; i < 3; ++i) {
            std::cout << "  [" << result.matrix[i][0] << ", "
                      << result.matrix[i][1] << ", "
                      << result.matrix[i][2] << "]" << std::endl;
        }

        const Matrix2Df R_in = R;
        const Matrix2Df G_in = G;
        const Matrix2Df B_in = B;
        const std::vector<uint8_t> bg_mask =
            image::build_chroma_background_mask_from_rgb(R_in, G_in, B_in);
        const double pre_rg_std_full =
            static_cast<double>(image::log_chroma_std_background(R_in, G_in, bg_mask));
        const double pre_bg_std_full =
            static_cast<double>(image::log_chroma_std_background(B_in, G_in, bg_mask));

        const PCCAttenuationContext sample_ctx =
            build_pcc_attenuation_context(R_in, G_in, B_in);
        const std::vector<PCCBackgroundSample> bg_samples =
            build_pcc_background_samples(R_in, G_in, B_in, bg_mask, sample_ctx);
        const ColorMatrix identity = {{{1.0, 0.0, 0.0},
                                       {0.0, 1.0, 0.0},
                                       {0.0, 0.0, 1.0}}};
        const PCCBackgroundStdPair pre_sample_std =
            sampled_background_std_after_matrix(bg_samples, identity);

        const double pre_rg_std =
            std::isfinite(pre_sample_std.rg_std) ? pre_sample_std.rg_std : pre_rg_std_full;
        const double pre_bg_std =
            std::isfinite(pre_sample_std.bg_std) ? pre_sample_std.bg_std : pre_bg_std_full;

        const bool use_sampled_eval = bg_samples.size() >= 512;
        if (use_sampled_eval) {
            std::cout << "[PCC] Damping evaluator: sampled background points="
                      << bg_samples.size() << std::endl;
        } else {
            std::cout << "[PCC] Damping evaluator fallback: full-frame candidate evaluation"
                      << std::endl;
        }

        constexpr double kStdHardCap = 1.08;    // never allow >8% background-std worsening
        constexpr double kWorsenBudget = 0.01;  // prefer <=1% worsening when possible
        const double chroma_strength = std::clamp(config.chroma_strength, 0.0, 1.0);
        const std::array<double, 7> base_strengths = {1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.0};
        std::array<double, 7> strengths{};
        for (size_t i = 0; i < base_strengths.size(); ++i) {
            strengths[i] = base_strengths[i] * chroma_strength;
        }
        strengths.back() = 0.0;
        if (chroma_strength < 0.999) {
            std::cout << "[PCC] Chroma strength limit active: " << chroma_strength
                      << std::endl;
        }
        double chosen_alpha_r = 1.0;
        double chosen_alpha_b = 1.0;
        ColorMatrix chosen_matrix = result.matrix;
        double best_score = 0.0;
        double best_post_rg_std = pre_rg_std;
        double best_post_bg_std = pre_bg_std;
        bool found_budget_candidate = false;
        bool found_fallback_candidate = false;
        double best_budget_score = -std::numeric_limits<double>::infinity();
        double best_fallback_score = -std::numeric_limits<double>::infinity();
        ColorMatrix fallback_matrix = chosen_matrix;
        double fallback_alpha_r = chosen_alpha_r;
        double fallback_alpha_b = chosen_alpha_b;
        double fallback_post_rg_std = best_post_rg_std;
        double fallback_post_bg_std = best_post_bg_std;

        auto eval_candidate_std = [&](const ColorMatrix &candidate) {
            if (use_sampled_eval) {
                return sampled_background_std_after_matrix(bg_samples, candidate);
            }
            Matrix2Df Rt = R_in;
            Matrix2Df Gt = G_in;
            Matrix2Df Bt = B_in;
            apply_color_matrix_impl(Rt, Gt, Bt, candidate, false);
            PCCBackgroundStdPair out;
            out.rg_std =
                static_cast<double>(image::log_chroma_std_background(Rt, Gt, bg_mask));
            out.bg_std =
                static_cast<double>(image::log_chroma_std_background(Bt, Gt, bg_mask));
            return out;
        };

        auto hard_ok = [&](double pre, double post) {
            if (!(std::isfinite(pre) && pre > 0.0)) return true;
            return std::isfinite(post) && post <= pre * kStdHardCap;
        };

        for (double alpha_r : strengths) {
            for (double alpha_b : strengths) {
                const ColorMatrix candidate =
                    blend_matrix_with_identity_per_channel(result.matrix, alpha_r, alpha_b);
                const PCCBackgroundStdPair post_std = eval_candidate_std(candidate);
                const double post_rg_std = post_std.rg_std;
                const double post_bg_std = post_std.bg_std;

                // Hard safety cap against strong background chroma worsening.
                const bool rg_hard_ok = hard_ok(pre_rg_std, post_rg_std);
                const bool bg_hard_ok = hard_ok(pre_bg_std, post_bg_std);
                if (!(rg_hard_ok && bg_hard_ok)) {
                    continue;
                }

                const double rel_rg =
                    (std::isfinite(pre_rg_std) && pre_rg_std > 0.0 && std::isfinite(post_rg_std))
                        ? (post_rg_std / pre_rg_std - 1.0)
                        : 0.0;
                const double rel_bg =
                    (std::isfinite(pre_bg_std) && pre_bg_std > 0.0 && std::isfinite(post_bg_std))
                        ? (post_bg_std / pre_bg_std - 1.0)
                        : 0.0;

                const double gain = 0.5 * (alpha_r + alpha_b); // [0..1]
                const double worsen_penalty =
                    std::max(0.0, rel_rg) + std::max(0.0, rel_bg);
                const double improve_bonus =
                    std::max(0.0, -rel_rg) + std::max(0.0, -rel_bg);
                const double imbalance = std::abs(rel_rg - rel_bg);
                const double total_abs = std::abs(rel_rg) + std::abs(rel_bg);

                // Fallback soft objective:
                // stronger correction is rewarded, but background chroma drift is
                // penalized much harder than in earlier revisions.
                const double fallback_score =
                    1.25 * gain - 4.5 * worsen_penalty - 1.4 * imbalance -
                    0.35 * total_abs + 0.70 * improve_bonus;
                if (!found_fallback_candidate || fallback_score > best_fallback_score) {
                    found_fallback_candidate = true;
                    best_fallback_score = fallback_score;
                    fallback_matrix = candidate;
                    fallback_alpha_r = alpha_r;
                    fallback_alpha_b = alpha_b;
                    fallback_post_rg_std = post_rg_std;
                    fallback_post_bg_std = post_bg_std;
                }

                // Primary objective:
                // if any candidate stays within a strict 1% worsening budget, pick
                // the strongest correction among those budget-compliant candidates.
                const bool within_budget =
                    (rel_rg <= kWorsenBudget) && (rel_bg <= kWorsenBudget);
                if (within_budget) {
                    const double budget_score =
                        gain + 0.60 * improve_bonus - 0.80 * imbalance -
                        0.20 * worsen_penalty;
                    if (!found_budget_candidate || budget_score > best_budget_score) {
                        found_budget_candidate = true;
                        best_budget_score = budget_score;
                        chosen_alpha_r = alpha_r;
                        chosen_alpha_b = alpha_b;
                        chosen_matrix = candidate;
                        best_post_rg_std = post_rg_std;
                        best_post_bg_std = post_bg_std;
                    }
                }
            }
        }

        if (found_budget_candidate) {
            best_score = best_budget_score;
        } else if (found_fallback_candidate) {
            best_score = best_fallback_score;
            chosen_alpha_r = fallback_alpha_r;
            chosen_alpha_b = fallback_alpha_b;
            chosen_matrix = fallback_matrix;
            best_post_rg_std = fallback_post_rg_std;
            best_post_bg_std = fallback_post_bg_std;
        } else {
            chosen_alpha_r = 0.0;
            chosen_alpha_b = 0.0;
            chosen_matrix =
                blend_matrix_with_identity_per_channel(result.matrix, 0.0, 0.0);
            best_post_rg_std = pre_rg_std;
            best_post_bg_std = pre_bg_std;
        }

        if (chosen_alpha_r < 0.999 || chosen_alpha_b < 0.999) {
            std::cout << "[PCC] Adaptive damping applied: alpha_r=" << chosen_alpha_r
                      << " alpha_b=" << chosen_alpha_b
                      << " (background chroma guard)" << std::endl;
            std::cout << "[PCC] Background chroma std pre/post: rg="
                      << pre_rg_std << " -> " << best_post_rg_std
                      << ", bg=" << pre_bg_std << " -> " << best_post_bg_std
                      << ", score=" << best_score << std::endl;
            result.matrix = chosen_matrix;
            update_result_matrix_metrics(&result);
        }

        const double residual_before_apply = result.residual_rms;
        int applied_n_stars_used = result.n_stars_used;
        const double applied_residual_rms = recompute_residual_rms_for_matrix(
            photometry, result.matrix, config.sigma_clip, &applied_n_stars_used);
        if (std::isfinite(applied_residual_rms) && applied_n_stars_used > 0) {
            result.residual_rms = applied_residual_rms;
            result.n_stars_used = applied_n_stars_used;
            if (std::abs(result.residual_rms - residual_before_apply) > 1.0e-6) {
                std::cout << "[PCC] Residual RMS updated for applied matrix: "
                          << residual_before_apply << " -> " << result.residual_rms
                          << " (stars=" << result.n_stars_used << ")" << std::endl;
            }
        } else {
            std::cout << "[PCC] Warning: failed to recompute residual RMS for applied matrix; "
                      << "keeping fit residual=" << residual_before_apply << std::endl;
        }

        apply_color_matrix(R, G, B, result.matrix, config.apply_attenuation);
        std::cout << "[PCC] Apply mode: "
                  << (config.apply_attenuation ? "attenuated" : "linear")
                  << std::endl;
        neutralize_background_offsets(R, G, B);
        std::cout << "[PCC] Color correction applied." << std::endl;
    } else {
        std::cerr << "[PCC] Failed: " << result.error_message << std::endl;
    }

    return result;
}

} // namespace tile_compile::astrometry
