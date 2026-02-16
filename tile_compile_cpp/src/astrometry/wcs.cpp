#include "tile_compile/astrometry/wcs.hpp"

#include <cmath>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>

namespace tile_compile::astrometry {

WCS wcs_from_cdelt_crota(double crval1, double crval2,
                         double crpix1, double crpix2,
                         double cdelt1, double cdelt2,
                         double crota2, int naxis1, int naxis2) {
    WCS w;
    w.crval1 = crval1;
    w.crval2 = crval2;
    w.crpix1 = crpix1;
    w.crpix2 = crpix2;
    w.naxis1 = naxis1;
    w.naxis2 = naxis2;

    constexpr double D2R = M_PI / 180.0;
    double cos_r = std::cos(crota2 * D2R);
    double sin_r = std::sin(crota2 * D2R);

    w.cd1_1 =  cdelt1 * cos_r;
    w.cd1_2 = -cdelt2 * sin_r;
    w.cd2_1 =  cdelt1 * sin_r;
    w.cd2_2 =  cdelt2 * cos_r;

    return w;
}

static double parse_fits_double(const std::string &val_str) {
    std::string s = val_str;
    // Remove trailing comment (after /)
    auto slash = s.find('/');
    if (slash != std::string::npos) s = s.substr(0, slash);
    // Trim whitespace
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.pop_back();
    // Remove surrounding quotes if present
    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
        s = s.substr(1, s.size() - 2);
        while (!s.empty() && s.back() == ' ') s.pop_back();
    }

    std::stringstream ss(s);
    ss.imbue(std::locale::classic());
    double result = 0.0;
    ss >> result;
    if (ss.fail()) return 0.0;
    return result;
}

WCS parse_wcs_file(const std::string &path) {
    WCS w;

    std::ifstream f(path);
    if (!f.is_open()) return w;

    double cdelt1 = 0, cdelt2 = 0, crota1 = 0, crota2 = 0;
    bool have_cd = false;

    std::string line;
    while (std::getline(f, line)) {
        // FITS keyword = value / comment
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        // Trim key
        while (!key.empty() && key.back() == ' ') key.pop_back();
        while (!key.empty() && key.front() == ' ') key.erase(key.begin());

        double dval = parse_fits_double(val);

        if (key == "CRVAL1")  w.crval1 = dval;
        else if (key == "CRVAL2")  w.crval2 = dval;
        else if (key == "CRPIX1")  w.crpix1 = dval;
        else if (key == "CRPIX2")  w.crpix2 = dval;
        else if (key == "NAXIS1")  w.naxis1 = static_cast<int>(dval);
        else if (key == "NAXIS2")  w.naxis2 = static_cast<int>(dval);
        else if (key == "CDELT1")  cdelt1 = dval;
        else if (key == "CDELT2")  cdelt2 = dval;
        else if (key == "CROTA1")  crota1 = dval;
        else if (key == "CROTA2")  crota2 = dval;
        else if (key == "CD1_1")   { w.cd1_1 = dval; have_cd = true; }
        else if (key == "CD1_2")   { w.cd1_2 = dval; have_cd = true; }
        else if (key == "CD2_1")   { w.cd2_1 = dval; have_cd = true; }
        else if (key == "CD2_2")   { w.cd2_2 = dval; have_cd = true; }
    }

    // If NAXIS1/NAXIS2 missing, infer from CRPIX (center pixel convention)
    if (w.naxis1 == 0 && w.crpix1 > 0)
        w.naxis1 = static_cast<int>(std::round(w.crpix1 * 2.0));
    if (w.naxis2 == 0 && w.crpix2 > 0)
        w.naxis2 = static_cast<int>(std::round(w.crpix2 * 2.0));

    // If no CD matrix, build from CDELT + CROTA
    if (!have_cd && (std::abs(cdelt1) > 0 || std::abs(cdelt2) > 0)) {
        double rot = (std::abs(crota2) > 0) ? crota2 : crota1;
        w = wcs_from_cdelt_crota(w.crval1, w.crval2, w.crpix1, w.crpix2,
                                 cdelt1, cdelt2, rot, w.naxis1, w.naxis2);
    }

    return w;
}

} // namespace tile_compile::astrometry
