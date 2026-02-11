#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/core/errors.hpp"
#include "tile_compile/core/utils.hpp"

#include <fitsio.h>
#include <algorithm>
#include <cstring>

namespace tile_compile::io {

std::optional<std::string> FitsHeader::get_string(const std::string& key) const {
    auto it = string_values.find(key);
    if (it != string_values.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<double> FitsHeader::get_double(const std::string& key) const {
    auto it = numeric_values.find(key);
    if (it != numeric_values.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<int> FitsHeader::get_int(const std::string& key) const {
    auto it = int_values.find(key);
    if (it != int_values.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<bool> FitsHeader::get_bool(const std::string& key) const {
    auto it = bool_values.find(key);
    if (it != bool_values.end()) {
        return it->second;
    }
    return std::nullopt;
}

void FitsHeader::set(const std::string& key, const std::string& value) {
    string_values[key] = value;
}

void FitsHeader::set(const std::string& key, double value) {
    numeric_values[key] = value;
}

void FitsHeader::set(const std::string& key, int value) {
    int_values[key] = value;
}

void FitsHeader::set(const std::string& key, bool value) {
    bool_values[key] = value;
}

bool is_fits_image_path(const fs::path& path) {
    std::string ext = core::to_lower(path.extension().string());
    return ext == ".fit" || ext == ".fits" || ext == ".fts";
}

std::pair<Matrix2Df, FitsHeader> read_fits_float(const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        throw FitsError("Cannot open FITS file: " + path.string());
    }
    
    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;
    
    fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
    if (status) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot read FITS image parameters: " + path.string());
    }
    
    if (naxis < 2) {
        fits_close_file(fptr, &status);
        throw FitsError("FITS file has less than 2 dimensions: " + path.string());
    }
    
    long width = naxes[0];
    long height = naxes[1];
    long npixels = width * height;
    
    std::vector<float> buffer(npixels);
    long fpixel[3] = {1, 1, 1};
    
    fits_read_pix(fptr, TFLOAT, fpixel, npixels, nullptr, buffer.data(), nullptr, &status);
    if (status) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot read FITS pixel data: " + path.string());
    }
    
    FitsHeader header;
    
    char card[FLEN_CARD];
    int nkeys = 0;
    fits_get_hdrspace(fptr, &nkeys, nullptr, &status);
    
    for (int i = 1; i <= nkeys; ++i) {
        fits_read_record(fptr, i, card, &status);
        if (status) continue;
        
        char keyname[FLEN_KEYWORD];
        char value[FLEN_VALUE];
        char comment[FLEN_COMMENT];
        int keylen = 0;
        
        fits_get_keyname(card, keyname, &keylen, &status);
        if (status) {
            status = 0;
            continue;
        }
        
        std::string key(keyname);
        if (key.empty() || key == "COMMENT" || key == "HISTORY" || key == "END") {
            continue;
        }
        
        char dtype;
        fits_get_keytype(card, &dtype, &status);
        if (status) {
            status = 0;
            continue;
        }
        
        fits_parse_value(card, value, comment, &status);
        if (status) {
            status = 0;
            continue;
        }
        
        std::string val_str(value);
        val_str.erase(0, val_str.find_first_not_of(" '"));
        val_str.erase(val_str.find_last_not_of(" '") + 1);
        
        switch (dtype) {
            case 'C':
                header.set(key, val_str);
                break;
            case 'L':
                header.set(key, val_str == "T" || val_str == "1");
                break;
            case 'I':
                try {
                    header.set(key, std::stoi(val_str));
                } catch (...) {
                    header.set(key, val_str);
                }
                break;
            case 'F':
                try {
                    header.set(key, std::stod(val_str));
                } catch (...) {
                    header.set(key, val_str);
                }
                break;
            default:
                header.set(key, val_str);
                break;
        }
    }
    
    fits_close_file(fptr, &status);
    
    Matrix2Df data(height, width);
    for (long y = 0; y < height; ++y) {
        for (long x = 0; x < width; ++x) {
            data(y, x) = buffer[y * width + x];
        }
    }
    
    return {data, header};
}

RGBImage read_fits_rgb(const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;

    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        throw FitsError("Cannot open FITS file: " + path.string());
    }

    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;

    fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
    if (status) {
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
    long npixels = w * h;

    RGBImage result;
    result.width = static_cast<int>(w);
    result.height = static_cast<int>(h);

    // Read header (same as read_fits_float)
    int nkeys = 0;
    fits_get_hdrspace(fptr, &nkeys, nullptr, &status);
    for (int i = 1; i <= nkeys; ++i) {
        char card[FLEN_CARD];
        fits_read_record(fptr, i, card, &status);
        if (status) { status = 0; continue; }
        char keyname[FLEN_KEYWORD];
        int keylen = 0;
        fits_get_keyname(card, keyname, &keylen, &status);
        if (status) { status = 0; continue; }
        std::string key(keyname);
        if (key.empty() || key == "COMMENT" || key == "HISTORY" || key == "END") continue;
        char dtype;
        fits_get_keytype(card, &dtype, &status);
        if (status) { status = 0; continue; }
        char value[FLEN_VALUE], comment[FLEN_COMMENT];
        fits_parse_value(card, value, comment, &status);
        if (status) { status = 0; continue; }
        std::string val_str(value);
        val_str.erase(0, val_str.find_first_not_of(" '"));
        val_str.erase(val_str.find_last_not_of(" '") + 1);
        switch (dtype) {
            case 'C': result.header.set(key, val_str); break;
            case 'L': result.header.set(key, val_str == "T" || val_str == "1"); break;
            case 'I': try { result.header.set(key, std::stoi(val_str)); } catch (...) { result.header.set(key, val_str); } break;
            case 'F': try { result.header.set(key, std::stod(val_str)); } catch (...) { result.header.set(key, val_str); } break;
            default: result.header.set(key, val_str); break;
        }
    }

    auto read_plane = [&](long plane) -> Matrix2Df {
        std::vector<float> buf(static_cast<size_t>(npixels));
        long fpixel[3] = {1, 1, plane};
        int st = 0;
        fits_read_pix(fptr, TFLOAT, fpixel, npixels, nullptr, buf.data(), nullptr, &st);
        if (st) throw FitsError("Cannot read FITS plane " + std::to_string(plane) + ": " + path.string());
        Matrix2Df mat(h, w);
        for (long y = 0; y < h; ++y)
            for (long x = 0; x < w; ++x)
                mat(y, x) = buf[static_cast<size_t>(y * w + x)];
        return mat;
    };

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
}

Matrix2Df read_fits_region_float(const fs::path& path, int x0, int y0, int width, int height) {
    fitsfile* fptr = nullptr;
    int status = 0;

    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        throw FitsError("Cannot open FITS file: " + path.string());
    }

    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;

    fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
    if (status || naxis < 2) {
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

    std::vector<float> buffer(static_cast<size_t>(rw) * static_cast<size_t>(rh));

    long fpixel[2] = {static_cast<long>(rx0 + 1), static_cast<long>(ry0 + 1)};
    long lpixel[2] = {static_cast<long>(rx0 + rw), static_cast<long>(ry0 + rh)};
    long inc[2] = {1, 1};

    fits_read_subset(fptr, TFLOAT, fpixel, lpixel, inc, nullptr, buffer.data(), nullptr, &status);
    fits_close_file(fptr, &status);

    if (status) {
        throw FitsError("Cannot read FITS ROI pixel data: " + path.string());
    }

    Matrix2Df data(rh, rw);
    for (int yy = 0; yy < rh; ++yy) {
        for (int xx = 0; xx < rw; ++xx) {
            data(yy, xx) = buffer[static_cast<size_t>(yy) * static_cast<size_t>(rw) + static_cast<size_t>(xx)];
        }
    }

    return data;
}

void write_fits_float(const fs::path& path, const Matrix2Df& data, const FitsHeader& header) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    std::string filepath = "!" + path.string();
    
    if (fits_create_file(&fptr, filepath.c_str(), &status)) {
        throw FitsError("Cannot create FITS file: " + path.string());
    }
    
    long naxes[2] = {data.cols(), data.rows()};
    
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    if (status) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot create FITS image: " + path.string());
    }

    auto should_skip_key = [](const std::string& key) -> bool {
        return key == "SIMPLE" || key == "BITPIX" || key == "NAXIS" || key == "NAXIS1" || key == "NAXIS2" ||
               key == "EXTEND" || key == "BZERO" || key == "BSCALE";
    };
    
    for (const auto& [key, value] : header.string_values) {
        if (key.size() <= 8) {
            if (should_skip_key(key)) continue;
            fits_update_key(fptr, TSTRING, key.c_str(), 
                           const_cast<char*>(value.c_str()), nullptr, &status);
            if (status) status = 0;
        }
    }
    
    for (const auto& [key, value] : header.numeric_values) {
        if (key.size() <= 8) {
            if (should_skip_key(key)) continue;
            double val = value;
            fits_update_key(fptr, TDOUBLE, key.c_str(), &val, nullptr, &status);
            if (status) status = 0;
        }
    }
    
    for (const auto& [key, value] : header.int_values) {
        if (key.size() <= 8) {
            if (should_skip_key(key)) continue;
            int val = value;
            fits_update_key(fptr, TINT, key.c_str(), &val, nullptr, &status);
            if (status) status = 0;
        }
    }
    
    for (const auto& [key, value] : header.bool_values) {
        if (key.size() <= 8) {
            if (should_skip_key(key)) continue;
            int val = value ? 1 : 0;
            fits_update_key(fptr, TLOGICAL, key.c_str(), &val, nullptr, &status);
            if (status) status = 0;
        }
    }
    
    std::vector<float> buffer(data.size());
    for (long y = 0; y < data.rows(); ++y) {
        for (long x = 0; x < data.cols(); ++x) {
            buffer[y * data.cols() + x] = data(y, x);
        }
    }
    
    long fpixel[2] = {1, 1};
    long nelem = static_cast<long>(data.size());
    fits_write_pix(fptr, TFLOAT, fpixel, nelem, buffer.data(), &status);
    if (status) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot write FITS pixel data: " + path.string());
    }
    
    fits_close_file(fptr, &status);
}

void write_fits_rgb(const fs::path& path, const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B, const FitsHeader& header) {
    if (R.rows() != G.rows() || R.rows() != B.rows() || R.cols() != G.cols() || R.cols() != B.cols()) {
        throw FitsError("RGB channel dimensions must match");
    }

    fitsfile* fptr = nullptr;
    int status = 0;
    
    std::string filepath = "!" + path.string();
    
    if (fits_create_file(&fptr, filepath.c_str(), &status)) {
        throw FitsError("Cannot create FITS file: " + path.string());
    }
    
    // Create 3D image cube: NAXIS1=width, NAXIS2=height, NAXIS3=3 (RGB planes)
    long naxes[3] = {R.cols(), R.rows(), 3};
    
    fits_create_img(fptr, FLOAT_IMG, 3, naxes, &status);
    if (status) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot create FITS RGB image: " + path.string());
    }

    auto should_skip_key = [](const std::string& key) -> bool {
        return key == "SIMPLE" || key == "BITPIX" || key == "NAXIS" || 
               key == "NAXIS1" || key == "NAXIS2" || key == "NAXIS3" || key == "EXTEND" ||
               key == "BAYERPAT" || key == "BZERO" || key == "BSCALE";
    };
    
    for (const auto& [key, value] : header.string_values) {
        if (key.size() <= 8 && !should_skip_key(key)) {
            fits_update_key(fptr, TSTRING, key.c_str(), const_cast<char*>(value.c_str()), nullptr, &status);
            if (status) status = 0;
        }
    }
    
    for (const auto& [key, value] : header.numeric_values) {
        if (key.size() <= 8 && !should_skip_key(key)) {
            double val = value;
            fits_update_key(fptr, TDOUBLE, key.c_str(), &val, nullptr, &status);
            if (status) status = 0;
        }
    }
    
    // Write R plane (z=1)
    std::vector<float> buffer(static_cast<size_t>(R.size()));
    for (long y = 0; y < R.rows(); ++y) {
        for (long x = 0; x < R.cols(); ++x) {
            buffer[static_cast<size_t>(y * R.cols() + x)] = R(y, x);
        }
    }
    long fpixel_r[3] = {1, 1, 1};
    fits_write_pix(fptr, TFLOAT, fpixel_r, static_cast<long>(R.size()), buffer.data(), &status);
    
    // Write G plane (z=2)
    for (long y = 0; y < G.rows(); ++y) {
        for (long x = 0; x < G.cols(); ++x) {
            buffer[static_cast<size_t>(y * G.cols() + x)] = G(y, x);
        }
    }
    long fpixel_g[3] = {1, 1, 2};
    fits_write_pix(fptr, TFLOAT, fpixel_g, static_cast<long>(G.size()), buffer.data(), &status);
    
    // Write B plane (z=3)
    for (long y = 0; y < B.rows(); ++y) {
        for (long x = 0; x < B.cols(); ++x) {
            buffer[static_cast<size_t>(y * B.cols() + x)] = B(y, x);
        }
    }
    long fpixel_b[3] = {1, 1, 3};
    fits_write_pix(fptr, TFLOAT, fpixel_b, static_cast<long>(B.size()), buffer.data(), &status);
    
    if (status) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot write FITS RGB pixel data: " + path.string());
    }
    
    fits_close_file(fptr, &status);
}

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

std::tuple<int, int, int> get_fits_dimensions(const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        throw FitsError("Cannot open FITS file: " + path.string());
    }
    
    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;
    
    fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
    fits_close_file(fptr, &status);
    
    if (status) {
        throw FitsError("Cannot read FITS dimensions: " + path.string());
    }
    
    return {static_cast<int>(naxes[0]), static_cast<int>(naxes[1]), naxis};
}

} // namespace tile_compile::io
