#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tile_compile::astrometry {

// Number of XP sampled wavelength bins (336-1020 nm, 2nm steps)
constexpr int XPSAMPLED_LEN = 343;
constexpr double XPSAMPLED_WL_START = 336.0;  // nm
constexpr double XPSAMPLED_WL_STEP  = 2.0;    // nm

// A star entry used for PCC.  May come from different catalog sources.
struct GaiaStar {
    double ra;          // degrees
    double dec;         // degrees
    float  mag;         // G magnitude (or V magnitude for APASS)
    float  pmra  = 0;   // proper motion RA (mas/yr)
    float  pmdec = 0;   // proper motion Dec (mas/yr)
    float  teff  = 0;   // effective temperature (K), 0 = unknown
    std::vector<float> xp_flux;  // XP sampled spectrum (343 bins, physical flux)
                                 // empty if source is not Siril XP-sampled
};

// Packed on-disk record (701 bytes)
#pragma pack(push, 1)
struct SourceEntryXPsamp {
    int32_t  ra_scaled;
    int32_t  dec_scaled;
    int16_t  dra_scaled;
    int16_t  ddec_scaled;
    int16_t  mag_scaled;
    int8_t   fexpo;
    uint16_t flux[XPSAMPLED_LEN];
};
#pragma pack(pop)

static_assert(sizeof(SourceEntryXPsamp) == 701, "SourceEntryXPsamp must be 701 bytes");

// Siril catalog file header (128 bytes on disk)
struct SirilCatHeader {
    std::string title;
    uint8_t  gaia_version;
    uint8_t  healpix_level;
    uint8_t  cat_type;
    uint8_t  chunked;
    uint8_t  chunk_level;
    uint32_t chunk_healpix;
    uint32_t chunk_first_healpixel;
    uint32_t chunk_last_healpixel;
};

// Get the conventional default Siril catalog directory path
// ($HOME/.local/share/siril/siril_cat1_healpix8_xpsamp)
std::string default_siril_gaia_catalog_dir();

// Check if the Siril Gaia XPsamp catalog is installed at the given path
bool is_siril_gaia_catalog_available(const std::string &catalog_dir);

// Cone search: find all stars within radius_deg of (ra_center, dec_center)
// with magnitude <= mag_limit.  catalog_dir must point to the directory
// containing the siril_cat*_healpix*_xpsamp_*.dat files.
// Returns empty vector if catalog not available or no stars found
std::vector<GaiaStar> siril_gaia_cone_search(
    const std::string &catalog_dir,
    double ra_center, double dec_center,
    double radius_deg, double mag_limit = 16.0);

// Compute synthetic broadband flux for a star given its XP spectrum
// and a filter transmission curve (wavelength in nm, transmission 0-1)
// filter_wl and filter_tx must have the same size
double synthetic_flux(const std::vector<float> &xp_flux,
                      const std::vector<double> &filter_wl,
                      const std::vector<double> &filter_tx);

// ─── VizieR online catalog queries ──────────────────────────────────────
//
// These perform synchronous HTTP GET requests to the VizieR TAP service.
// They return GaiaStar entries with teff populated (no xp_flux).
// On network error or timeout, they return an empty vector.

// Query Gaia DR3 via VizieR TAP.  Returns stars with valid Teff.
// Uses teff_gspphot from gaiadr3.astrophysical_parameters.
std::vector<GaiaStar> vizier_gaia_cone_search(
    double ra_center, double dec_center,
    double radius_deg, double mag_limit = 16.0);

// Query APASS DR9 via VizieR TAP.  Returns stars with Teff estimated
// from Johnson B-V color index.
std::vector<GaiaStar> vizier_apass_cone_search(
    double ra_center, double dec_center,
    double radius_deg, double mag_limit = 16.0);

// Convert Johnson B-V color index to effective temperature (K).
// Uses the Ballesteros (2012) formula.
double bv_to_teff(double bv);

// ─── Siril catalog download ─────────────────────────────────────────────

// Download URL pattern for Siril Gaia XP-sampled catalog chunks.
// Total ~21 GB across 48 chunk files.
std::string siril_catalog_chunk_url(int chunk_id);

// Download a single chunk file to the given directory.
// Returns true on success.  progress_cb is called with (bytes_done, bytes_total).
bool download_siril_catalog_chunk(
    int chunk_id, const std::string &dest_dir,
    std::function<void(size_t, size_t)> progress_cb = nullptr);

// Check which chunk IDs (0-47) are missing from the catalog directory.
std::vector<int> missing_siril_catalog_chunks(const std::string &catalog_dir);

} // namespace tile_compile::astrometry
