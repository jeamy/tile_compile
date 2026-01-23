#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace tile_compile::runner {

/**
 * Discover FITS frames matching pattern in input directory.
 */
std::vector<fs::path> discover_frames(const fs::path& input_dir, const std::string& pattern);

/**
 * Copy configuration file to output directory.
 */
void copy_config(const fs::path& config_path, const fs::path& out_path);

/**
 * Create symlink or copy file if symlink fails.
 */
void safe_symlink_or_copy(const fs::path& src, const fs::path& dst);

/**
 * Compute SHA256 hash of file.
 */
std::string sha256_file(const fs::path& path);

/**
 * Compute SHA256 hash of bytes.
 */
std::string sha256_bytes(const std::vector<uint8_t>& data);

/**
 * Read file as bytes.
 */
std::vector<uint8_t> read_bytes(const fs::path& path);

/**
 * Canonical JSON serialization for hashing.
 */
std::string json_dumps_canonical(const nlohmann::json& obj);

} // namespace tile_compile::runner
