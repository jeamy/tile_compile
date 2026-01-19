#pragma once

#include "tile_compile/core/types.hpp"
#include <map>
#include <optional>
#include <string>

namespace tile_compile::io {

struct FitsHeader {
    std::map<std::string, std::string> string_values;
    std::map<std::string, double> numeric_values;
    std::map<std::string, int> int_values;
    std::map<std::string, bool> bool_values;
    
    std::optional<std::string> get_string(const std::string& key) const;
    std::optional<double> get_double(const std::string& key) const;
    std::optional<int> get_int(const std::string& key) const;
    std::optional<bool> get_bool(const std::string& key) const;
    
    void set(const std::string& key, const std::string& value);
    void set(const std::string& key, double value);
    void set(const std::string& key, int value);
    void set(const std::string& key, bool value);
};

bool is_fits_image_path(const fs::path& path);

std::pair<Matrix2Df, FitsHeader> read_fits_float(const fs::path& path);

void write_fits_float(const fs::path& path, const Matrix2Df& data, const FitsHeader& header);

BayerPattern detect_bayer_pattern(const FitsHeader& header);

ColorMode detect_color_mode(const FitsHeader& header, int naxis = 2);

std::tuple<int, int, int> get_fits_dimensions(const fs::path& path);

} // namespace tile_compile::io
