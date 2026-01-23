#pragma once

#include "tile_compile/core/types.hpp"
#include "tile_compile/io/fits_io.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <utility>

namespace fs = std::filesystem;

namespace tile_compile::runner {

/**
 * Check if path has FITS extension.
 */
bool is_fits_image_path(const fs::path& path);

/**
 * Check if FITS file is CFA/Bayer mosaic.
 * Returns nullopt if file cannot be read.
 */
std::optional<bool> fits_is_cfa(const fs::path& path);

/**
 * Get Bayer pattern from FITS header.
 * Returns nullopt if not found or file cannot be read.
 */
std::optional<std::string> fits_get_bayerpat(const fs::path& path);

/**
 * Read FITS file as float32 array with header.
 * Throws on error.
 */
std::pair<Matrix2Df, io::FitsHeader> read_fits_float(const fs::path& path);

/**
 * Load frame from FITS file.
 * Returns empty matrix on error.
 */
Matrix2Df load_frame(const fs::path& path);

} // namespace tile_compile::runner
