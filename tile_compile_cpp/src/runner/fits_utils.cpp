#include "tile_compile/runner/fits_utils.hpp"

#include <fitsio.h>
#include <algorithm>

namespace tile_compile::runner {

bool is_fits_image_path(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".fit" || ext == ".fits" || ext == ".fts";
}

std::optional<bool> fits_is_cfa(const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        return std::nullopt;
    }
    
    char bayerpat[FLEN_VALUE];
    char comment[FLEN_COMMENT];
    int bp_status = 0;
    fits_read_key(fptr, TSTRING, "BAYERPAT", bayerpat, comment, &bp_status);
    
    fits_close_file(fptr, &status);
    
    if (bp_status == 0) {
        return true;
    }
    return false;
}

std::optional<std::string> fits_get_bayerpat(const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        return std::nullopt;
    }
    
    char bayerpat[FLEN_VALUE];
    char comment[FLEN_COMMENT];
    int bp_status = 0;
    fits_read_key(fptr, TSTRING, "BAYERPAT", bayerpat, comment, &bp_status);
    
    fits_close_file(fptr, &status);
    
    if (bp_status == 0) {
        std::string bp_str(bayerpat);
        // Trim whitespace and quotes
        bp_str.erase(0, bp_str.find_first_not_of(" \t\n\r'\""));
        bp_str.erase(bp_str.find_last_not_of(" \t\n\r'\"") + 1);
        
        if (!bp_str.empty()) {
            std::transform(bp_str.begin(), bp_str.end(), bp_str.begin(), ::toupper);
            return bp_str;
        }
    }
    
    return std::nullopt;
}

std::pair<Matrix2Df, io::FitsHeader> read_fits_float(const fs::path& path) {
    return io::read_fits_float(path);
}

Matrix2Df load_frame(const fs::path& path) {
    try {
        auto [data, header] = io::read_fits_float(path);
        return data;
    } catch (...) {
        return Matrix2Df();
    }
}

} // namespace tile_compile::runner
