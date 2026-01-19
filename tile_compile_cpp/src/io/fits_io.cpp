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
    
    for (const auto& [key, value] : header.string_values) {
        if (key.size() <= 8) {
            fits_update_key(fptr, TSTRING, key.c_str(), 
                           const_cast<char*>(value.c_str()), nullptr, &status);
        }
    }
    
    for (const auto& [key, value] : header.numeric_values) {
        if (key.size() <= 8) {
            double val = value;
            fits_update_key(fptr, TDOUBLE, key.c_str(), &val, nullptr, &status);
        }
    }
    
    for (const auto& [key, value] : header.int_values) {
        if (key.size() <= 8) {
            int val = value;
            fits_update_key(fptr, TINT, key.c_str(), &val, nullptr, &status);
        }
    }
    
    for (const auto& [key, value] : header.bool_values) {
        if (key.size() <= 8) {
            int val = value ? 1 : 0;
            fits_update_key(fptr, TLOGICAL, key.c_str(), &val, nullptr, &status);
        }
    }
    
    std::vector<float> buffer(data.size());
    for (long y = 0; y < data.rows(); ++y) {
        for (long x = 0; x < data.cols(); ++x) {
            buffer[y * data.cols() + x] = data(y, x);
        }
    }
    
    long fpixel[2] = {1, 1};
    fits_write_pix(fptr, TFLOAT, fpixel, data.size(), buffer.data(), &status);
    if (status) {
        fits_close_file(fptr, &status);
        throw FitsError("Cannot write FITS pixel data: " + path.string());
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
