#pragma once

#include "types.hpp"
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace tile_compile::core {

namespace fs = std::filesystem;

// Time utilities
std::string get_iso_timestamp();
std::string get_run_id();

// File utilities
std::vector<fs::path> discover_frames(const fs::path& input_dir, const std::string& pattern = "*.fit*");
std::vector<uint8_t> read_bytes(const fs::path& path);
std::string read_text(const fs::path& path);
void write_text(const fs::path& path, const std::string& text);
void safe_hardlink_or_copy(const fs::path& src, const fs::path& dst);
fs::path pick_output_file(const fs::path& dir, const std::string& prefix, const std::string& ext);

// Hash utilities
std::string sha256_bytes(const std::vector<uint8_t>& data);
std::string sha256_file(const fs::path& path);

// Config utilities
void copy_config(const fs::path& src, const fs::path& dst);
fs::path resolve_project_root(const fs::path& config_path);

// Math utilities
float compute_median(const Matrix2Df& data);
float compute_median(const VectorXf& data);
float compute_mad(const Matrix2Df& data);
float compute_robust_sigma(const Matrix2Df& data);
float compute_percentile(const VectorXf& data, float percentile);

// String utilities
std::string to_lower(const std::string& s);
bool ends_with(const std::string& str, const std::string& suffix);
bool starts_with(const std::string& str, const std::string& prefix);
std::vector<std::string> split(const std::string& str, char delimiter);
std::string join(const std::vector<std::string>& parts, const std::string& delimiter);

// Glob pattern matching
bool glob_match(const std::string& pattern, const std::string& str);
std::vector<fs::path> glob(const fs::path& dir, const std::string& pattern);

} // namespace tile_compile::core
