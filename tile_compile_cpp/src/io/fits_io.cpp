#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/core/errors.hpp"
#include "tile_compile/core/utils.hpp"

#include <fitsio.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>

namespace tile_compile::io {

namespace {

/// @brief Implements cfitsio status text.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string cfitsio_status_text(int status) {
    char msg[FLEN_STATUS] = {0};
    fits_get_errstatus(status, msg);
    return std::string(msg);
}

/// @brief Implements cfitsio disk full.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool cfitsio_disk_full(int status_text_code, const std::string& text) {
    (void)status_text_code;
    const std::string lower = core::to_lower(text);
    return (lower.find("no space left on device") != std::string::npos) ||
           (lower.find("not enough space") != std::string::npos) ||
           (lower.find("disk full") != std::string::npos) ||
           (lower.find("enospc") != std::string::npos);
}

/// @brief Implements fits write error message.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string fits_write_error_message(const std::string& action,
                                     const fs::path& path,
                                     int status) {
    const std::string status_msg = cfitsio_status_text(status);
    std::ostringstream oss;
    if (cfitsio_disk_full(status, status_msg)) {
        oss << "Disk full while writing FITS output: " << path.string()
            << " (" << action << ", cfitsio_status=" << status
            << ", reason=\"" << status_msg << "\")";
    } else {
        oss << "Cannot " << action << ": " << path.string()
            << " (cfitsio_status=" << status
            << ", reason=\"" << status_msg << "\")";
    }
    return oss.str();
}

/// @brief Implements move to first image hdu.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool move_to_first_image_hdu(fitsfile* fptr,
                             int& bitpix,
                             int& naxis,
                             long naxes[3],
                             int& status) {
    auto has_valid_image_shape = [&](int current_naxis, const long current_naxes[3]) -> bool {
        return current_naxis >= 2 && current_naxes[0] > 0 && current_naxes[1] > 0;
    };

    auto try_read_znaxis = [&](int& out_naxis, long out_naxes[3]) -> bool {
        int st = 0;
        long znaxis = 0;
        long znaxis1 = 0;
        long znaxis2 = 0;
        fits_read_key(fptr, TLONG, const_cast<char*>("ZNAXIS"), &znaxis, nullptr, &st);
        if (st || znaxis < 2) {
            return false;
        }
        st = 0;
        fits_read_key(fptr, TLONG, const_cast<char*>("ZNAXIS1"), &znaxis1, nullptr, &st);
        if (st || znaxis1 <= 0) {
            return false;
        }
        st = 0;
        fits_read_key(fptr, TLONG, const_cast<char*>("ZNAXIS2"), &znaxis2, nullptr, &st);
        if (st || znaxis2 <= 0) {
            return false;
        }

        out_naxis = static_cast<int>(znaxis);
        out_naxes[0] = znaxis1;
        out_naxes[1] = znaxis2;
        out_naxes[2] = 0;
        return true;
    };

    status = 0;
    fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
    if (status == 0 && has_valid_image_shape(naxis, naxes)) {
        return true;
    }
    if (try_read_znaxis(naxis, naxes)) {
        status = 0;
        return true;
    }

    status = 0;
    int nhdus = 0;
    fits_get_num_hdus(fptr, &nhdus, &status);
    if (status) {
        return false;
    }

    for (int hdu = 2; hdu <= nhdus; ++hdu) {
        int hdu_type = 0;
        status = 0;
        fits_movabs_hdu(fptr, hdu, &hdu_type, &status);
        if (status) {
            continue;
        }

        status = 0;
        fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
        if (status == 0 && has_valid_image_shape(naxis, naxes)) {
            return true;
        }
        if (try_read_znaxis(naxis, naxes)) {
            status = 0;
            return true;
        }
    }

    return false;
}

/// @brief Reads current header.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
FitsHeader read_current_header(fitsfile* fptr, int& status) {
    FitsHeader header;

    char card[FLEN_CARD];
    int nkeys = 0;
    fits_get_hdrspace(fptr, &nkeys, nullptr, &status);

    for (int i = 1; i <= nkeys; ++i) {
        fits_read_record(fptr, i, card, &status);
        if (status) {
            status = 0;
            continue;
        }

        char keyname[FLEN_KEYWORD];
        char value[FLEN_VALUE];
        char comment[FLEN_COMMENT];
        int keylen = 0;

        fits_get_keyname(card, keyname, &keylen, &status);
        if (status) {
            status = 0;
            continue;
        }

        const std::string key(keyname);
        if (key.empty() || key == "COMMENT" || key == "HISTORY" || key == "END") {
            continue;
        }

        fits_parse_value(card, value, comment, &status);
        if (status) {
            status = 0;
            continue;
        }

        // Determine the value type from the raw value string rather than
        // relying on fits_get_keytype, which returns incorrect types for
        // some non-standard FITS producers (e.g. DWARF II: FOCALLEN=100.
        // classified as 'L', FILTER='Astro' as 'L', DATE-OBS='2026-...'
        // as 'F').  The raw value from fits_parse_value preserves the
        // leading quote for string keywords, which we use as the primary
        // type discriminator.
        std::string raw_value(value);
        // Strip leading spaces only — keep the quote for type detection
        size_t first_nonspace = raw_value.find_first_not_of(' ');
        if (first_nonspace == std::string::npos) first_nonspace = 0;

        const bool is_quoted_string =
            (first_nonspace < raw_value.size() && raw_value[first_nonspace] == '\'');

        // Clean value: strip spaces and single quotes for storage
        std::string val_str(raw_value);
        val_str.erase(0, val_str.find_first_not_of(" '"));
        val_str.erase(val_str.find_last_not_of(" '") + 1);

        if (is_quoted_string) {
            // Definitely a string — fits_parse_value confirmed the quote
            header.set(key, val_str);
        } else if (val_str == "T" || val_str == "F") {
            // FITS logical value (unquoted T or F)
            header.set(key, val_str == "T");
        } else {
            // Try integer first (no decimal point), then float, then string
            bool parsed = false;
            if (val_str.find('.') == std::string::npos &&
                val_str.find('e') == std::string::npos &&
                val_str.find('E') == std::string::npos) {
                try {
                    size_t pos = 0;
                    int int_val = std::stoi(val_str, &pos);
                    if (pos == val_str.size()) {
                        header.set(key, int_val);
                        parsed = true;
                    }
                } catch (...) {}
            }
            if (!parsed) {
                try {
                    size_t pos = 0;
                    double dbl_val = std::stod(val_str, &pos);
                    // Accept partial parse only if it consumed the entire
                    // value (e.g. "100." → 100.0, but not "2026-08-13..."
                    // which would parse as 2026 and leave a suffix).
                    if (pos == val_str.size()) {
                        header.set(key, dbl_val);
                        parsed = true;
                    }
                } catch (...) {}
            }
            if (!parsed) {
                header.set(key, val_str);
            }
        }
    }

    status = 0;
    return header;
}

/// @brief Decides whether to skip header key.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool should_skip_header_key(const std::string& key, bool is_rgb_image) {
    if (key == "SIMPLE" || key == "BITPIX" || key == "NAXIS" ||
        key == "NAXIS1" || key == "NAXIS2" || key == "EXTEND" ||
        key == "BZERO" || key == "BSCALE" || key == "ROWORDER") {
        return true;
    }
    if (is_rgb_image && (key == "NAXIS3" || key == "BAYERPAT")) {
        return true;
    }
    return false;
}

/// @brief Writes header keywords.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void write_header_keywords(fitsfile* fptr, const FitsHeader& header,
                           bool is_rgb_image, int& status) {
    for (const auto& [key, value] : header.string_values) {
        if (key.size() <= 8 && !should_skip_header_key(key, is_rgb_image)) {
            fits_update_key(fptr, TSTRING, key.c_str(),
                           const_cast<char*>(value.c_str()), nullptr, &status);
            if (status) status = 0;
        }
    }

    for (const auto& [key, value] : header.numeric_values) {
        if (key.size() <= 8 && !should_skip_header_key(key, is_rgb_image)) {
            double val = value;
            fits_update_key(fptr, TDOUBLE, key.c_str(), &val, nullptr, &status);
            if (status) status = 0;
        }
    }

    for (const auto& [key, value] : header.int_values) {
        if (key.size() <= 8 && !should_skip_header_key(key, is_rgb_image)) {
            int val = value;
            fits_update_key(fptr, TINT, key.c_str(), &val, nullptr, &status);
            if (status) status = 0;
        }
    }

    for (const auto& [key, value] : header.bool_values) {
        if (key.size() <= 8 && !should_skip_header_key(key, is_rgb_image)) {
            int val = value ? 1 : 0;
            fits_update_key(fptr, TLOGICAL, key.c_str(), &val, nullptr, &status);
            if (status) status = 0;
        }
    }
}

/// @brief Reads current pixels float.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df read_current_pixels_float(fitsfile* fptr, const fs::path& path,
                                    long width, long height, long plane,
                                    int& status) {
    const long npixels = width * height;
    Matrix2Df data(height, width);
    long fpixel[3] = {1, 1, plane};

    fits_read_pix(fptr, TFLOAT, fpixel, npixels, nullptr, data.data(), nullptr,
                  &status);
    if (status) {
        throw FitsError("Cannot read FITS pixel data: " + path.string());
    }

    return data;
}

} // namespace

/// @brief Implements get string.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<std::string> FitsHeader::get_string(const std::string& key) const {
    auto it = string_values.find(key);
    if (it != string_values.end()) {
        return it->second;
    }
    return std::nullopt;
}

/// @brief Implements get double.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<double> FitsHeader::get_double(const std::string& key) const {
    auto it = numeric_values.find(key);
    if (it != numeric_values.end()) {
        return it->second;
    }
    return std::nullopt;
}

/// @brief Implements get int.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<int> FitsHeader::get_int(const std::string& key) const {
    auto it = int_values.find(key);
    if (it != int_values.end()) {
        return it->second;
    }
    return std::nullopt;
}

/// @brief Implements get bool.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::optional<bool> FitsHeader::get_bool(const std::string& key) const {
    auto it = bool_values.find(key);
    if (it != bool_values.end()) {
        return it->second;
    }
    return std::nullopt;
}

/// @brief Implements set.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void FitsHeader::set(const std::string& key, const std::string& value) {
    string_values[key] = value;
}

/// @brief Implements set.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void FitsHeader::set(const std::string& key, double value) {
    numeric_values[key] = value;
}

/// @brief Implements set.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void FitsHeader::set(const std::string& key, int value) {
    int_values[key] = value;
}

/// @brief Implements set.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void FitsHeader::set(const std::string& key, bool value) {
    bool_values[key] = value;
}

/// @brief Checks fits image path.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool is_fits_image_path(const fs::path& path) {
    const std::string name = core::to_lower(path.filename().string());
    return core::ends_with(name, ".fit") ||
           core::ends_with(name, ".fits") ||
           core::ends_with(name, ".fts") ||
           core::ends_with(name, ".fit.fz") ||
           core::ends_with(name, ".fits.fz") ||
           core::ends_with(name, ".fts.fz");
}

/// @brief Reads fits float.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::pair<Matrix2Df, FitsHeader> read_fits_float(const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        throw FitsError("Cannot open FITS file: " + path.string());
    }
    
    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;
    
    if (!move_to_first_image_hdu(fptr, bitpix, naxis, naxes, status)) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot read FITS image parameters: " + path.string());
    }
    
    if (naxis < 2) {
        fits_close_file(fptr, &status);
        throw FitsError("FITS file has less than 2 dimensions: " + path.string());
    }
    
    long width = naxes[0];
    long height = naxes[1];
    try {
        Matrix2Df data = read_current_pixels_float(fptr, path, width, height, 1, status);
        FitsHeader header = read_current_header(fptr, status);
        fits_close_file(fptr, &status);
        return {data, header};
    } catch (...) {
        fits_close_file(fptr, &status);
        throw;
    }
}

/// @brief Reads fits header.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
FitsHeader read_fits_header(const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;

    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        throw FitsError("Cannot open FITS file: " + path.string());
    }

    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;

    if (!move_to_first_image_hdu(fptr, bitpix, naxis, naxes, status)) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot read FITS image parameters: " + path.string());
    }

    try {
        FitsHeader header = read_current_header(fptr, status);
        fits_close_file(fptr, &status);
        return header;
    } catch (...) {
        fits_close_file(fptr, &status);
        throw;
    }
}

/// @brief Reads fits pixels float.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df read_fits_pixels_float(const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;

    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        throw FitsError("Cannot open FITS file: " + path.string());
    }

    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;

    if (!move_to_first_image_hdu(fptr, bitpix, naxis, naxes, status)) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot read FITS image parameters: " + path.string());
    }

    if (naxis < 2) {
        fits_close_file(fptr, &status);
        throw FitsError("FITS file has less than 2 dimensions: " + path.string());
    }

    try {
        Matrix2Df data =
            read_current_pixels_float(fptr, path, naxes[0], naxes[1], 1, status);
        fits_close_file(fptr, &status);
        return data;
    } catch (...) {
        fits_close_file(fptr, &status);
        throw;
    }
}

/// @brief Reads fits rgb.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
RGBImage read_fits_rgb(const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;

    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        throw FitsError("Cannot open FITS file: " + path.string());
    }

    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;

    if (!move_to_first_image_hdu(fptr, bitpix, naxis, naxes, status)) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot read FITS image parameters: " + path.string());
    }

    if (naxis < 2) {
        fits_close_file(fptr, &status);
        throw FitsError("FITS file has less than 2 dimensions: " + path.string());
    }

    long w = naxes[0];
    long h = naxes[1];
    long nplanes = (naxis >= 3) ? naxes[2] : 1;
    RGBImage result;
    result.width = static_cast<int>(w);
    result.height = static_cast<int>(h);

    result.header = read_current_header(fptr, status);

    auto read_plane = [&](long plane) -> Matrix2Df {
        int st = 0;
        return read_current_pixels_float(fptr, path, w, h, plane, st);
    };

    try {
        if (nplanes >= 3) {
            result.R = read_plane(1);
            result.G = read_plane(2);
            result.B = read_plane(3);
        } else {
            // Mono image — duplicate to all channels
            result.R = read_plane(1);
            result.G = result.R;
            result.B = result.R;
        }

        fits_close_file(fptr, &status);
        return result;
    } catch (...) {
        fits_close_file(fptr, &status);
        throw;
    }
}

/// @brief Reads fits region float.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
Matrix2Df read_fits_region_float(const fs::path& path, int x0, int y0, int width, int height) {
    fitsfile* fptr = nullptr;
    int status = 0;

    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        throw FitsError("Cannot open FITS file: " + path.string());
    }

    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;

    if (!move_to_first_image_hdu(fptr, bitpix, naxis, naxes, status)) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot read FITS image parameters: " + path.string());
    }

    int img_w = static_cast<int>(naxes[0]);
    int img_h = static_cast<int>(naxes[1]);

    int rx0 = std::max(0, x0);
    int ry0 = std::max(0, y0);
    int rx1 = std::min(img_w, x0 + width);
    int ry1 = std::min(img_h, y0 + height);

    int rw = std::max(0, rx1 - rx0);
    int rh = std::max(0, ry1 - ry0);

    if (rw <= 0 || rh <= 0) {
        fits_close_file(fptr, &status);
        return Matrix2Df();
    }

    Matrix2Df data(rh, rw);

    if (naxis >= 3) {
        // RGB cube: read ROI from first plane to keep scalar ROI API contract.
        long fpixel[3] = {static_cast<long>(rx0 + 1), static_cast<long>(ry0 + 1), 1};
        long lpixel[3] = {static_cast<long>(rx0 + rw), static_cast<long>(ry0 + rh), 1};
        long inc[3] = {1, 1, 1};
        fits_read_subset(fptr, TFLOAT, fpixel, lpixel, inc, nullptr,
                         data.data(), nullptr, &status);
    } else {
        long fpixel[2] = {static_cast<long>(rx0 + 1), static_cast<long>(ry0 + 1)};
        long lpixel[2] = {static_cast<long>(rx0 + rw), static_cast<long>(ry0 + rh)};
        long inc[2] = {1, 1};
        fits_read_subset(fptr, TFLOAT, fpixel, lpixel, inc, nullptr,
                         data.data(), nullptr, &status);
    }
    fits_close_file(fptr, &status);

    if (status) {
        throw FitsError("Cannot read FITS ROI pixel data: " + path.string() +
                        " (cfitsio_status=" + std::to_string(status) +
                        ", reason=\"" + cfitsio_status_text(status) + "\")");
    }

    return data;
}

/// @brief Writes fits float.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void write_fits_float(const fs::path& path, const Matrix2Df& data, const FitsHeader& header) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    std::string filepath = "!" + path.string();
    
    if (fits_create_file(&fptr, filepath.c_str(), &status)) {
        throw FitsError(fits_write_error_message("create FITS file", path, status));
    }
    
    long naxes[2] = {static_cast<long>(data.cols()), static_cast<long>(data.rows())};
    
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    if (status) {
        const int write_status = status;
        int close_status = 0;
        fits_close_file(fptr, &close_status);
        throw FitsError(fits_write_error_message("create FITS image", path, write_status));
    }

    write_header_keywords(fptr, header, false, status);

    // Declare row order: the internal Eigen RowMajor buffer has row 0 at the top
    // (screen convention), which is the opposite of the default FITS bottom-up
    // convention. ROWORDER=TOP-DOWN tells viewers (Siril, DS9, etc.) to display
    // the image without flipping, matching the actual data layout on disk.
    {
        char roworder[] = "TOP-DOWN";
        fits_update_key(fptr, TSTRING, "ROWORDER", roworder,
                        "Row order: row 0 is top of image", &status);
        if (status) status = 0; // non-fatal: proceed even if key cannot be written
    }
    
    // Write pixel data row by row to avoid cfitsio internal buffer issues
    // with very large single-write calls (e.g. 50M+ elements for expanded canvas).
    const long nrows = static_cast<long>(data.rows());
    const long ncols = static_cast<long>(data.cols());
    for (long row = 0; row < nrows; ++row) {
        long fpixel[2] = {1, row + 1};
        const float* row_ptr = data.data() + row * ncols;
        fits_write_pix(fptr, TFLOAT, fpixel, ncols,
                       const_cast<float*>(row_ptr), &status);
        if (status) {
            const int write_status = status;
            int close_status = 0;
            fits_close_file(fptr, &close_status);
            throw FitsError(fits_write_error_message("write FITS pixel data", path, write_status));
        }
    }
    
    fits_close_file(fptr, &status);
    if (status) {
        throw FitsError(fits_write_error_message("close FITS file", path, status));
    }
}

/// @brief Writes fits rgb.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void write_fits_rgb(const fs::path& path, const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B, const FitsHeader& header) {
    if (R.rows() != G.rows() || R.rows() != B.rows() || R.cols() != G.cols() || R.cols() != B.cols()) {
        throw FitsError("RGB channel dimensions must match");
    }

    fitsfile* fptr = nullptr;
    int status = 0;

    std::string filepath = "!" + path.string();

    if (fits_create_file(&fptr, filepath.c_str(), &status)) {
        throw FitsError(fits_write_error_message("create FITS file", path, status));
    }

    // Create 3D image cube: NAXIS1=width, NAXIS2=height, NAXIS3=3 (RGB planes)
    long naxes[3] = {static_cast<long>(R.cols()), static_cast<long>(R.rows()), 3};

    fits_create_img(fptr, FLOAT_IMG, 3, naxes, &status);
    if (status) {
        const int write_status = status;
        int close_status = 0;
        fits_close_file(fptr, &close_status);
        throw FitsError(fits_write_error_message("create FITS RGB image", path, write_status));
    }

    write_header_keywords(fptr, header, true, status);

    // Declare row order: the internal Eigen RowMajor buffer has row 0 at the top
    // (screen convention), which is the opposite of the default FITS bottom-up
    // convention. ROWORDER=TOP-DOWN tells viewers (Siril, DS9, etc.) to display
    // the image without flipping, matching the actual data layout on disk.
    {
        char roworder[] = "TOP-DOWN";
        fits_update_key(fptr, TSTRING, "ROWORDER", roworder,
                        "Row order: row 0 is top of image", &status);
        if (status) status = 0; // non-fatal: proceed even if key cannot be written
    }

    // Write each RGB plane row by row to avoid cfitsio internal buffer issues
    // with very large single-write calls.
    const long nrows_rgb = static_cast<long>(R.rows());
    const long ncols_rgb = static_cast<long>(R.cols());
    const Matrix2Df* planes[3] = {&R, &G, &B};
    for (int p = 0; p < 3; ++p) {
        for (long row = 0; row < nrows_rgb; ++row) {
            long fpixel[3] = {1, row + 1, p + 1};
            const float* row_ptr = planes[p]->data() + row * ncols_rgb;
            fits_write_pix(fptr, TFLOAT, fpixel, ncols_rgb,
                           const_cast<float*>(row_ptr), &status);
            if (status) break;
        }
        if (status) break;
    }

    if (status) {
        const int write_status = status;
        int close_status = 0;
        fits_close_file(fptr, &close_status);
        throw FitsError(fits_write_error_message("write FITS RGB pixel data", path, write_status));
    }

    fits_close_file(fptr, &status);
    if (status) {
        throw FitsError(fits_write_error_message("close FITS file", path, status));
    }
}

void write_fits_rgb_u32(const fs::path& path, const Matrix2Df& R, const Matrix2Df& G,
                       const Matrix2Df& B, const FitsHeader& header) {
    if (R.rows() != G.rows() || R.rows() != B.rows() || R.cols() != G.cols() ||
        R.cols() != B.cols()) {
        throw FitsError("RGB channel dimensions must match");
    }

    fitsfile* fptr = nullptr;
    int status = 0;
    const std::string filepath = "!" + path.string();
    if (fits_create_file(&fptr, filepath.c_str(), &status)) {
        throw FitsError(fits_write_error_message("create FITS file", path, status));
    }

    long naxes[3] = {static_cast<long>(R.cols()), static_cast<long>(R.rows()), 3};
    fits_create_img(fptr, LONG_IMG, 3, naxes, &status);
    if (status) {
        const int write_status = status;
        int close_status = 0;
        fits_close_file(fptr, &close_status);
        throw FitsError(fits_write_error_message("create FITS RGB uint32 image", path, write_status));
    }

    double bscale = 1.0;
    double bzero = 2147483648.0;
    fits_update_key(fptr, TDOUBLE, "BSCALE", &bscale, nullptr, &status);
    fits_update_key(fptr, TDOUBLE, "BZERO", &bzero, nullptr, &status);
    write_header_keywords(fptr, header, true, status);

    char roworder[] = "TOP-DOWN";
    fits_update_key(fptr, TSTRING, "ROWORDER", roworder,
                    "Row order: row 0 is top of image", &status);
    if (status) status = 0;

    const long nrows = static_cast<long>(R.rows());
    const long ncols = static_cast<long>(R.cols());
    const Matrix2Df* planes[3] = {&R, &G, &B};
    for (int p = 0; p < 3 && !status; ++p) {
        for (long row = 0; row < nrows && !status; ++row) {
            long fpixel[3] = {1, row + 1, p + 1};
            const float* row_ptr = planes[p]->data() + row * ncols;
            std::vector<unsigned int> values(static_cast<size_t>(ncols));
            for (long col = 0; col < ncols; ++col) {
                const float value = row_ptr[col];
                const double rounded = std::isfinite(value)
                    ? std::round(static_cast<double>(value)) : 0.0;
                values[static_cast<size_t>(col)] = static_cast<unsigned int>(
                    std::clamp(rounded, 0.0, 4294967295.0));
            }
            fits_write_pix(fptr, TUINT, fpixel, ncols, values.data(), &status);
        }
    }

    if (status) {
        const int write_status = status;
        int close_status = 0;
        fits_close_file(fptr, &close_status);
        throw FitsError(fits_write_error_message("write FITS RGB uint32 pixel data", path, write_status));
    }

    fits_close_file(fptr, &status);
    if (status) {
        throw FitsError(fits_write_error_message("close FITS file", path, status));
    }
}

/// @brief Updates fits header in place.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void update_fits_header_in_place(const fs::path& path, const FitsHeader& header) {
    fitsfile* fptr = nullptr;
    int status = 0;

    if (fits_open_file(&fptr, path.string().c_str(), READWRITE, &status)) {
        throw FitsError("Cannot open FITS file for header update: " +
                        path.string());
    }

    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;
    if (!move_to_first_image_hdu(fptr, bitpix, naxis, naxes, status)) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot read FITS image parameters: " + path.string());
    }

    const bool is_rgb_image = (naxis >= 3 && naxes[2] >= 3);
    write_header_keywords(fptr, header, is_rgb_image, status);
    if (status) {
        const int update_status = status;
        int close_status = 0;
        fits_close_file(fptr, &close_status);
        throw FitsError(fits_write_error_message("update FITS header", path,
                                                 update_status));
    }

    // Ensure ROWORDER is always present after a header update.
    {
        char roworder[] = "TOP-DOWN";
        int ro_status = 0;
        fits_update_key(fptr, TSTRING, "ROWORDER", roworder,
                        "Row order: row 0 is top of image", &ro_status);
        // non-fatal: ignore if key cannot be written
    }

    fits_close_file(fptr, &status);
    if (status) {
        throw FitsError(fits_write_error_message("close FITS file", path, status));
    }
}

/// @brief Detects bayer pattern.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
BayerPattern detect_bayer_pattern(const FitsHeader& header) {
    auto bayerpat = header.get_string("BAYERPAT");
    if (bayerpat) {
        return string_to_bayer_pattern(*bayerpat);
    }
    
    auto colortyp = header.get_string("COLORTYP");
    if (colortyp) {
        std::string ct = core::to_lower(*colortyp);
        if (ct.find("rggb") != std::string::npos) return BayerPattern::RGGB;
        if (ct.find("bggr") != std::string::npos) return BayerPattern::BGGR;
        if (ct.find("grbg") != std::string::npos) return BayerPattern::GRBG;
        if (ct.find("gbrg") != std::string::npos) return BayerPattern::GBRG;
    }
    
    auto xbayroff = header.get_int("XBAYROFF");
    auto ybayroff = header.get_int("YBAYROFF");
    if (xbayroff && ybayroff) {
        int x = *xbayroff % 2;
        int y = *ybayroff % 2;
        if (x == 0 && y == 0) return BayerPattern::RGGB;
        if (x == 1 && y == 0) return BayerPattern::GRBG;
        if (x == 0 && y == 1) return BayerPattern::GBRG;
        if (x == 1 && y == 1) return BayerPattern::BGGR;
    }
    
    return BayerPattern::UNKNOWN;
}

/// @brief Detects color mode.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
ColorMode detect_color_mode(const FitsHeader& header, int naxis) {
    if (naxis >= 3) {
        return ColorMode::RGB;
    }
    
    auto bayerpat = detect_bayer_pattern(header);
    if (bayerpat != BayerPattern::UNKNOWN) {
        return ColorMode::OSC;
    }
    
    auto colortyp = header.get_string("COLORTYP");
    if (colortyp) {
        std::string ct = core::to_lower(*colortyp);
        if (ct.find("mono") != std::string::npos) return ColorMode::MONO;
        if (ct.find("osc") != std::string::npos || ct.find("color") != std::string::npos) {
            return ColorMode::OSC;
        }
    }
    
    return ColorMode::MONO;
}

/// @brief Implements get fits dimensions.
/// @details Part of CFITSIO-backed FITS header/image read and write helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::tuple<int, int, int> get_fits_dimensions(const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        throw FitsError("Cannot open FITS file: " + path.string());
    }
    
    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;
    
    if (!move_to_first_image_hdu(fptr, bitpix, naxis, naxes, status)) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot read FITS dimensions: " + path.string());
    }

    fits_close_file(fptr, &status);
    
    return {static_cast<int>(naxes[0]), static_cast<int>(naxes[1]), naxis};
}

} // namespace tile_compile::io
