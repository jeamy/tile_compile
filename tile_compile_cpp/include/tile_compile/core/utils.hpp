#pragma once

#include "types.hpp"
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace tile_compile::core {

namespace fs = std::filesystem;

// Time utilities
std::string get_iso_timestamp();
std::string get_run_id();

// File utilities
std::vector<fs::path> discover_frames(const fs::path& input_dir, const std::string& pattern = "*.fit;*.fits;*.fts;*.fit.fz;*.fits.fz;*.fts.fz");
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

// Statistical utilities (canonical implementations — do NOT duplicate)
float median_of(std::vector<float> v);
float mad_of(std::vector<float> v, float median);
float stddev_of(const std::vector<float>& v);
float robust_sigma_mad(std::vector<float>& pixels);
float percentile_from_sorted(const std::vector<float>& sorted, float pct);
float percentile_of(std::vector<float>& values, float pct);
float estimate_background_sigma_clip(std::vector<float> pixels);
std::vector<size_t> sample_indices(size_t count, int max_samples);

// Robust z-score normalization: (x - median) / (1.4826 * MAD)
void robust_zscore(const std::vector<float>& v, std::vector<float>& out);

// Median of finite positive values, with fallback
float median_finite_positive(const std::vector<float>& v, float fallback);
float median_finite(const std::vector<float>& v, float fallback);

struct StretchResult {
    bool applied = false;
    float low = 0.0f;
    float high = 0.0f;
    size_t sample_count = 0;
};

StretchResult stretch_to_u16_linear_from_zero_inplace(
    Matrix2Df& img);

StretchResult stretch_rgb_to_u32_linear_from_zero_inplace(
    Matrix2Df& r,
    Matrix2Df& g,
    Matrix2Df& b);

StretchResult stretch_rgb_to_u32_linear_from_zero_inplace(
    Matrix2Df& r,
    Matrix2Df& g,
    Matrix2Df& b,
    const std::vector<uint8_t>& statistics_mask);

/// Compute the effective OMP thread count for a parallel region, respecting
/// the configured worker limit. Call this instead of using bare `#pragma omp`
/// without `num_threads`. The result accounts for:
///   - max_configured_workers (from runtime_limits.parallel_workers)
///   - hardware_concurrency as upper bound
///   - already being inside a parallel region (returns 1)
///   - work_items: never use more threads than items to process
///
/// Usage:
///   const int nt = core::omp_effective_threads(cfg_workers, work_items);
///   #pragma omp parallel for num_threads(nt) schedule(static)
inline int omp_effective_threads(int max_configured_workers, int work_items = 0) {
#if defined(_OPENMP)
    if (omp_in_parallel()) return 1;
    int threads = std::max(1, max_configured_workers);
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 0) threads = std::min(threads, hw);
    if (work_items > 0) threads = std::min(threads, work_items);
    return std::max(1, threads);
#else
    (void)max_configured_workers;
    (void)work_items;
    return 1;
#endif
}

// String utilities
std::string to_lower(const std::string& s);
bool ends_with(const std::string& str, const std::string& suffix);
bool starts_with(const std::string& str, const std::string& prefix);
std::vector<std::string> split(const std::string& str, char delimiter);
std::string join(const std::vector<std::string>& parts, const std::string& delimiter);

// Glob pattern matching
bool glob_match(const std::string& pattern, const std::string& str);

} // namespace tile_compile::core
