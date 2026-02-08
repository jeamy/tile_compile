#include "tile_compile/astrometry/gaia_catalog.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <vector>

namespace tile_compile::astrometry {

namespace fs = std::filesystem;

// ─── Half-float (IEEE 754 binary16) to float conversion ───────────────────

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // Subnormal: normalize
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);  // Inf or NaN
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

// ─── Minimal HEALPix NESTED scheme ───────────────────────────────────────

static constexpr double TWOPI = 2.0 * M_PI;
static constexpr double HALFPI = M_PI / 2.0;

// Convert (theta, phi) to HEALPix NESTED pixel index at given order
// theta = colatitude [0, pi], phi = longitude [0, 2pi)
static int64_t ang2pix_nest(int order, double theta, double phi) {
    int nside = 1 << order;
    double z = std::cos(theta);
    double za = std::abs(z);
    double tt = std::fmod(phi, TWOPI);
    if (tt < 0) tt += TWOPI;
    tt /= HALFPI;  // in [0, 4)

    int64_t npix = 12LL * nside * nside;
    int64_t face, ix, iy;

    if (za <= 2.0 / 3.0) {
        // Equatorial region
        double temp1 = nside * (0.5 + tt);
        double temp2 = nside * z * 0.75;
        int jp = static_cast<int>(temp1 - temp2);
        int jm = static_cast<int>(temp1 + temp2);
        int ifp = jp / nside;
        int ifm = jm / nside;
        face = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8));
        ix = jm & (nside - 1);
        iy = nside - (jp & (nside - 1)) - 1;
    } else {
        int ntt = std::min(static_cast<int>(tt), 3);
        double tp = tt - ntt;
        double tmp = nside * std::sqrt(3.0 * (1.0 - za));
        int jp = static_cast<int>(tp * tmp);
        int jm = static_cast<int>((1.0 - tp) * tmp);
        jp = std::min(jp, nside - 1);
        jm = std::min(jm, nside - 1);
        if (z >= 0) {
            face = ntt;
            ix = nside - jm - 1;
            iy = nside - jp - 1;
        } else {
            face = ntt + 8;
            ix = jp;
            iy = jm;
        }
    }

    // xy2pix NESTED encoding
    int64_t result = 0;
    for (int i = 0; i < order; ++i) {
        result |= ((ix >> i) & 1LL) << (2 * i);
        result |= ((iy >> i) & 1LL) << (2 * i + 1);
    }
    return result + face * nside * nside;
}

// Disc query via grid sampling: sample sky positions within the search disc
// and collect unique HEALPix pixel indices.  Uses the verified ang2pix_nest.
static std::vector<int> query_disc_nested(int order, double theta_center,
                                          double phi_center, double radius_rad) {
    int nside = 1 << order;
    // Pixel angular size ≈ sqrt(4π / npix) ≈ sqrt(π/3) / nside
    double pix_size = std::sqrt(M_PI / 3.0) / nside;
    // Sample step: half a pixel size for good coverage
    double step = pix_size * 0.4;
    // Ensure at least a minimum number of samples across the radius
    int n_steps = std::max(3, static_cast<int>(std::ceil(radius_rad / step)));
    step = radius_rad / n_steps;

    std::set<int> pixel_set;

    // Always include the center pixel
    pixel_set.insert(static_cast<int>(ang2pix_nest(order, theta_center, phi_center)));

    // Sample a grid of points within the disc
    for (int iy = -n_steps; iy <= n_steps; ++iy) {
        double dtheta = iy * step;
        double theta = theta_center + dtheta;
        if (theta < 0.0 || theta > M_PI) continue;

        // At this declination ring, how many phi samples do we need?
        double sin_theta = std::sin(theta);
        double max_dphi = (sin_theta > 1e-10) ? radius_rad / sin_theta : TWOPI;
        int n_phi = std::max(1, static_cast<int>(std::ceil(max_dphi / step)));

        for (int ix = -n_phi; ix <= n_phi; ++ix) {
            double dphi = ix * step;
            double phi = phi_center + dphi;

            // Check angular distance from center
            double cos_dist = std::cos(theta_center) * std::cos(theta) +
                              std::sin(theta_center) * std::sin(theta) *
                              std::cos(dphi);
            if (cos_dist < std::cos(radius_rad * 1.05)) continue;  // small margin

            if (phi < 0) phi += TWOPI;
            if (phi >= TWOPI) phi -= TWOPI;

            pixel_set.insert(static_cast<int>(ang2pix_nest(order, theta, phi)));
        }
    }

    return std::vector<int>(pixel_set.begin(), pixel_set.end());
}

// ─── Siril catalog I/O ──────────────────────────────────────────────────

std::string default_siril_gaia_catalog_dir() {
    const char *home = std::getenv("HOME");
    if (!home) return "";
    return std::string(home) + "/.local/share/siril/siril_cat1_healpix8_xpsamp";
}

bool is_siril_gaia_catalog_available(const std::string &catalog_dir) {
    if (catalog_dir.empty() || !fs::exists(catalog_dir)) return false;

    // Check for at least one matching .dat file
    std::regex pattern("siril_cat\\d+_healpix\\d+_xpsamp_\\d+\\.dat");
    for (const auto &entry : fs::directory_iterator(catalog_dir)) {
        if (entry.is_regular_file() &&
            std::regex_match(entry.path().filename().string(), pattern)) {
            return true;
        }
    }
    return false;
}

static SirilCatHeader read_header(std::ifstream &f) {
    SirilCatHeader h;
    char title_buf[48] = {0};
    f.read(title_buf, 48);
    h.title = std::string(title_buf, strnlen(title_buf, 48));
    f.read(reinterpret_cast<char*>(&h.gaia_version), 1);
    f.read(reinterpret_cast<char*>(&h.healpix_level), 1);
    f.read(reinterpret_cast<char*>(&h.cat_type), 1);
    f.read(reinterpret_cast<char*>(&h.chunked), 1);
    f.read(reinterpret_cast<char*>(&h.chunk_level), 1);
    f.read(reinterpret_cast<char*>(&h.chunk_healpix), 4);
    f.read(reinterpret_cast<char*>(&h.chunk_first_healpixel), 4);
    f.read(reinterpret_cast<char*>(&h.chunk_last_healpixel), 4);
    // Skip spare bytes to reach offset 128
    char spare[63];
    f.read(spare, 63);
    return h;
}

static std::string find_first_chunk(const std::string &dir) {
    std::regex pattern("siril_cat\\d+_healpix\\d+_xpsamp_\\d+\\.dat");
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() &&
            std::regex_match(entry.path().filename().string(), pattern)) {
            return entry.path().filename().string();
        }
    }
    return "";
}

static int convert_healpix_level(int pixel_index, int from_level, int to_level) {
    if (from_level < to_level) {
        return pixel_index << (2 * (to_level - from_level));
    } else {
        return pixel_index >> (2 * (from_level - to_level));
    }
}

static GaiaStar decode_entry(const SourceEntryXPsamp &e) {
    constexpr double SCALE = 360.0 / static_cast<double>(INT32_MAX);
    GaiaStar s;
    s.ra    = e.ra_scaled * SCALE;
    s.dec   = e.dec_scaled * SCALE;
    s.mag   = e.mag_scaled / 1000.0f;
    s.pmra  = static_cast<float>(e.dra_scaled);
    s.pmdec = static_cast<float>(e.ddec_scaled);

    float powexp = std::pow(10.0f, static_cast<float>(e.fexpo));
    s.xp_flux.resize(XPSAMPLED_LEN);
    for (int j = 0; j < XPSAMPLED_LEN; ++j) {
        s.xp_flux[j] = half_to_float(e.flux[j]) / powexp;
    }
    return s;
}

// Angular distance squared (haversine) for filtering
static double angular_dist_sq(double ra1, double dec1, double ra2, double dec2) {
    constexpr double D2R = M_PI / 180.0;
    double dra  = (ra2 - ra1) * D2R;
    double ddec = (dec2 - dec1) * D2R;
    double a = std::sin(ddec / 2.0);
    double b = std::sin(dra / 2.0);
    return a * a + std::cos(dec1 * D2R) * std::cos(dec2 * D2R) * b * b;
}

std::vector<GaiaStar> siril_gaia_cone_search(
    const std::string &catalog_dir,
    double ra_center, double dec_center,
    double radius_deg, double mag_limit) {

    std::vector<GaiaStar> results;

    if (catalog_dir.empty() || !fs::exists(catalog_dir)) return results;
    const std::string &cat_dir = catalog_dir;

    // Find first chunk to read header metadata
    std::string first_chunk = find_first_chunk(cat_dir);
    if (first_chunk.empty()) return results;

    std::ifstream f0(cat_dir + "/" + first_chunk, std::ios::binary);
    if (!f0.is_open()) return results;
    SirilCatHeader hdr = read_header(f0);
    f0.close();

    int hp_level = hdr.healpix_level;    // 8
    int chunk_level = hdr.chunk_level;   // 1

    // Convert center to theta/phi
    constexpr double D2R = M_PI / 180.0;
    double theta = M_PI / 2.0 - dec_center * D2R;
    double phi = ra_center * D2R;
    if (phi < 0) phi += TWOPI;

    double radius_rad = radius_deg * D2R;

    // Find all HEALPix level-8 pixels within the search radius
    std::vector<int> pixel_indices = query_disc_nested(hp_level, theta, phi, radius_rad);

    if (pixel_indices.empty()) return results;

    // Group pixels by chunk
    std::set<int> chunk_ids;
    for (int pix : pixel_indices) {
        chunk_ids.insert(convert_healpix_level(pix, hp_level, chunk_level));
    }

    // Precompute haversine threshold
    double sin_half_r = std::sin(radius_rad / 2.0);
    double dist_threshold = sin_half_r * sin_half_r;
    double mag_threshold = mag_limit * 1000.0;

    // Process each chunk
    for (int chunk_id : chunk_ids) {
        // Build chunk filename
        char fname[128];
        std::snprintf(fname, sizeof(fname), "siril_cat%u_healpix%u_xpsamp_%d.dat",
                      chunk_level, hp_level, chunk_id);
        std::string chunk_path = cat_dir + "/" + fname;

        if (!fs::exists(chunk_path)) {
            std::cerr << "[PCC] Chunk file not found: " << chunk_path << std::endl;
            continue;
        }

        std::ifstream cf(chunk_path, std::ios::binary);
        if (!cf.is_open()) continue;

        SirilCatHeader ch = read_header(cf);

        // Read index
        int nside = 1 << ch.healpix_level;
        uint32_t n_total = 12 * nside * nside;
        uint32_t n_healpixels;
        if (ch.chunked) {
            int nside_c = 1 << ch.chunk_level;
            uint32_t n_chunks = 12 * nside_c * nside_c;
            n_healpixels = n_total / n_chunks;
        } else {
            n_healpixels = n_total;
        }

        std::vector<uint32_t> index(n_healpixels);
        cf.read(reinterpret_cast<char*>(index.data()), n_healpixels * sizeof(uint32_t));

        size_t data_offset = 128 + n_healpixels * sizeof(uint32_t);

        // Find which pixels in this chunk we need
        std::vector<int> local_pixels;
        for (int pix : pixel_indices) {
            if (convert_healpix_level(pix, hp_level, chunk_level) == chunk_id) {
                local_pixels.push_back(pix - static_cast<int>(ch.chunk_first_healpixel));
            }
        }

        // For each pixel range, read the star records
        for (int local_pix : local_pixels) {
            if (local_pix < 0 || local_pix >= static_cast<int>(n_healpixels)) continue;

            uint32_t start_rec = (local_pix == 0) ? 0 : index[local_pix - 1];
            uint32_t end_rec = index[local_pix];
            uint32_t n_recs = end_rec - start_rec;
            if (n_recs == 0) continue;

            cf.seekg(data_offset + start_rec * sizeof(SourceEntryXPsamp));
            std::vector<SourceEntryXPsamp> entries(n_recs);
            cf.read(reinterpret_cast<char*>(entries.data()), n_recs * sizeof(SourceEntryXPsamp));

            for (const auto &e : entries) {
                // Magnitude filter
                if (e.mag_scaled > mag_threshold) continue;

                // Decode position for distance check
                constexpr double SCALE = 360.0 / static_cast<double>(INT32_MAX);
                double ra  = e.ra_scaled * SCALE;
                double dec = e.dec_scaled * SCALE;

                // Distance filter
                if (angular_dist_sq(ra_center, dec_center, ra, dec) > dist_threshold) continue;

                results.push_back(decode_entry(e));
            }
        }
    }

    return results;
}

double synthetic_flux(const std::vector<float> &xp_flux,
                      const std::vector<double> &filter_wl,
                      const std::vector<double> &filter_tx) {
    if (xp_flux.size() != XPSAMPLED_LEN || filter_wl.size() != filter_tx.size() ||
        filter_wl.empty()) {
        return 0.0;
    }

    // Integrate: sum(flux * transmission * dlambda)
    // XP spectrum is on a regular 2nm grid starting at 336nm
    double total = 0.0;
    double norm = 0.0;

    for (size_t i = 0; i < filter_wl.size(); ++i) {
        double wl = filter_wl[i];
        double tx = filter_tx[i];

        // Find the XP bin for this wavelength
        double bin_f = (wl - XPSAMPLED_WL_START) / XPSAMPLED_WL_STEP;
        int bin = static_cast<int>(std::round(bin_f));
        if (bin < 0 || bin >= XPSAMPLED_LEN) continue;

        // Linear interpolation between adjacent bins
        double frac = bin_f - std::floor(bin_f);
        int bin0 = static_cast<int>(std::floor(bin_f));
        int bin1 = bin0 + 1;
        if (bin0 < 0) bin0 = 0;
        if (bin1 >= XPSAMPLED_LEN) bin1 = XPSAMPLED_LEN - 1;

        double flux_val = xp_flux[bin0] * (1.0 - frac) + xp_flux[bin1] * frac;

        // Determine dlambda for this filter sample
        double dlambda = 1.0;
        if (i > 0 && i < filter_wl.size() - 1) {
            dlambda = 0.5 * (filter_wl[i + 1] - filter_wl[i - 1]);
        } else if (filter_wl.size() > 1) {
            dlambda = (i == 0) ? (filter_wl[1] - filter_wl[0])
                               : (filter_wl.back() - filter_wl[filter_wl.size() - 2]);
        }

        total += flux_val * tx * dlambda;
        norm  += tx * dlambda;
    }

    return (norm > 0) ? total / norm : 0.0;
}

} // namespace tile_compile::astrometry
