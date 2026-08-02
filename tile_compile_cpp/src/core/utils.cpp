#include "tile_compile/core/utils.hpp"
#include "tile_compile/core/errors.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <random>
#include <regex>
#include <sstream>
#include <thread>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifdef _WIN32
#include <time.h>
#endif

#include <openssl/sha.h>

namespace tile_compile::core {

/// @brief Implements get iso timestamp.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::tm tm_buf;
#ifdef _WIN32
    gmtime_s(&tm_buf, &time_t_now);
#else
    gmtime_r(&time_t_now, &tm_buf);
#endif
    
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return oss.str();
}

/// @brief Implements get run id.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string get_run_id() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << '_';
    
    const char* hex = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
        oss << hex[dis(gen)];
    }
    
    return oss.str();
}

/// @brief Implements discover frames.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<fs::path> discover_frames(const fs::path& input_dir, const std::string& pattern) {
    std::vector<fs::path> frames;
    
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        return frames;
    }
    
    std::vector<std::string> patterns;
    {
        std::string token;
        for (char c : pattern) {
            if (c == ';' || c == ',') {
                // trim token
                size_t b = token.find_first_not_of(" \t");
                if (b != std::string::npos) {
                    size_t e = token.find_last_not_of(" \t");
                    patterns.push_back(token.substr(b, e - b + 1));
                }
                token.clear();
            } else {
                token.push_back(c);
            }
        }
        size_t b = token.find_first_not_of(" \t");
        if (b != std::string::npos) {
            size_t e = token.find_last_not_of(" \t");
            patterns.push_back(token.substr(b, e - b + 1));
        }
    }
    if (patterns.empty()) {
        patterns.push_back(pattern.empty() ? "*" : pattern);
    }

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            for (const auto& pat : patterns) {
                if (glob_match(pat, filename)) {
                    frames.push_back(entry.path());
                    break;
                }
            }
        }
    }
    
    std::sort(frames.begin(), frames.end());
    return frames;
}

/// @brief Reads bytes.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<uint8_t> read_bytes(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw IOError("Cannot open file: " + path.string());
    }
    
    auto size = file.tellg();
    if (size <= 0) {
        throw IOError("Empty or unreadable file: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw IOError("Cannot read file: " + path.string());
    }
    
    return buffer;
}

/// @brief Reads text.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string read_text(const fs::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw IOError("Cannot open file: " + path.string());
    }
    
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

/// @brief Writes text.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void write_text(const fs::path& path, const std::string& text) {
    std::ofstream file(path);
    if (!file) {
        throw IOError("Cannot create file: " + path.string());
    }
    file << text;
}

/// @brief Implements safe hardlink or copy.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void safe_hardlink_or_copy(const fs::path& src, const fs::path& dst) {
    std::error_code ec;
    fs::create_hard_link(src, dst, ec);
    if (ec) {
        fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
    }
}

/// @brief Implements pick output file.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
fs::path pick_output_file(const fs::path& dir, const std::string& prefix, const std::string& ext) {
    fs::create_directories(dir);
    
    int counter = 0;
    fs::path candidate;
    do {
        std::ostringstream oss;
        oss << prefix;
        if (counter > 0) {
            oss << "_" << counter;
        }
        oss << ext;
        candidate = dir / oss.str();
        ++counter;
    } while (fs::exists(candidate) && counter < 10000);
    
    return candidate;
}

/// @brief Implements sha256 bytes.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string sha256_bytes(const std::vector<uint8_t>& data) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(data.data(), data.size(), hash);
    
    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(hash[i]);
    }
    return oss.str();
}

/// @brief Implements sha256 file.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string sha256_file(const fs::path& path) {
    auto data = read_bytes(path);
    return sha256_bytes(data);
}

/// @brief Copies config.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void copy_config(const fs::path& src, const fs::path& dst) {
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
}

/// @brief Resolves project root.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
fs::path resolve_project_root(const fs::path& config_path) {
    fs::path p = config_path;
    if (fs::is_regular_file(p)) {
        p = p.parent_path();
    }
    p = fs::absolute(p);
    
    while (true) {
        if (fs::exists(p / "tile_compile_runner.py") || 
            fs::exists(p / "tile_compile.yaml") ||
            fs::exists(p / "CMakeLists.txt")) {
            return p;
        }
        if (p.parent_path() == p) {
            return fs::absolute(config_path).parent_path();
        }
        p = p.parent_path();
    }
}

/// @brief Converts lower.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

/// @brief Implements ends with.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// @brief Implements starts with.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool starts_with(const std::string& str, const std::string& prefix) {
    if (prefix.size() > str.size()) return false;
    return str.compare(0, prefix.size(), prefix) == 0;
}

/// @brief Implements split.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> parts;
    std::istringstream iss(str);
    std::string part;
    while (std::getline(iss, part, delimiter)) {
        parts.push_back(part);
    }
    return parts;
}

/// @brief Joins.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::string join(const std::vector<std::string>& parts, const std::string& delimiter) {
    std::ostringstream oss;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) oss << delimiter;
        oss << parts[i];
    }
    return oss.str();
}

/// @brief Matches or expands glob patterns for  match.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
bool glob_match(const std::string& pattern, const std::string& str) {
    std::string regex_pattern;
    for (char c : pattern) {
        switch (c) {
            case '*': regex_pattern += ".*"; break;
            case '?': regex_pattern += "."; break;
            case '.': regex_pattern += "\\."; break;
            case '[': regex_pattern += "["; break;
            case ']': regex_pattern += "]"; break;
            default: regex_pattern += c; break;
        }
    }
    
    std::regex re(regex_pattern, std::regex::icase);
    return std::regex_match(str, re);
}

// --- Statistical utilities (canonical, single implementation) ---

/// @brief Implements median of.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float median_of(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    const size_t n = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    const float hi = v[mid];
    if ((n % 2) == 1) return hi;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid - 1), v.end());
    const float lo = v[mid - 1];
    return 0.5f * (lo + hi);
}

float mad_of(std::vector<float> v, float median) {
    if (v.empty()) return 0.0f;
    for (float& x : v) x = std::fabs(x - median);
    return median_of(std::move(v));
}

/// @brief Implements stddev of.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float stddev_of(const std::vector<float>& v) {
    if (v.size() < 2) return 0.0f;
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x);
    const double mean = sum / static_cast<double>(v.size());
    double var = 0.0;
    for (float x : v) {
        const double d = static_cast<double>(x) - mean;
        var += d * d;
    }
    var /= static_cast<double>(v.size());
    return (var > 0.0) ? static_cast<float>(std::sqrt(var)) : 0.0f;
}

/// @brief Implements robust sigma mad.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float robust_sigma_mad(std::vector<float>& pixels) {
    if (pixels.empty()) return 0.0f;
    float med = median_of(pixels);
    for (float& x : pixels) x = std::fabs(x - med);
    float mad = median_of(std::move(pixels));
    return kMadToSigma * mad;
}

/// @brief Implements percentile from sorted.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float percentile_from_sorted(const std::vector<float>& sorted, float pct) {
    if (sorted.empty()) return 0.0f;
    float clamped = std::min(std::max(pct, 0.0f), 100.0f);
    float pos = (clamped / 100.0f) * static_cast<float>(sorted.size() - 1);
    size_t idx = static_cast<size_t>(std::round(pos));
    idx = std::min(idx, sorted.size() - 1);
    return sorted[idx];
}

/// @brief Implements percentile of.
/// @details Sorts the values in place and returns the requested percentile.
/// Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers.
float percentile_of(std::vector<float>& values, float pct) {
    std::sort(values.begin(), values.end());
    return percentile_from_sorted(values, pct);
}

/// @brief Estimates background sigma clip.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float estimate_background_sigma_clip(std::vector<float> pixels) {
    if (pixels.empty()) return 0.0f;
    for (int iter = 0; iter < 5; ++iter) {
        float mu = median_of(pixels);
        float sigma = stddev_of(pixels);
        if (!(sigma > 0.0f)) break;

        std::vector<float> clipped;
        clipped.reserve(pixels.size());
        const float thr = 3.0f * sigma;
        for (float x : pixels) {
            if (std::fabs(x - mu) < thr) {
                clipped.push_back(x);
            }
        }
        if (clipped.size() == pixels.size() || clipped.empty()) break;
        pixels.swap(clipped);
    }
    return median_of(pixels);
}

/// @brief Computes a two-pass sigma-clipped mean.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float two_pass_sigma_clipped_mean(std::vector<float> pixels) {
    if (pixels.empty()) return 0.0f;
    const float med = median_of(pixels);

    std::vector<float> abs_dev;
    abs_dev.reserve(pixels.size());
    for (float x : pixels) {
        abs_dev.push_back(std::fabs(x - med));
    }
    const float mad = median_of(abs_dev);
    const float sigma = kMadToSigma * mad;

    const float thr = 3.0f * sigma;
    double sum = 0.0;
    size_t n = 0;
    for (float x : pixels) {
        if (std::fabs(x - med) <= thr) {
            sum += static_cast<double>(x);
            ++n;
        }
    }
    if (n > 0) {
        return static_cast<float>(sum / static_cast<double>(n));
    }
    return med;
}

/// @brief Implements sample indices.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
std::vector<size_t> sample_indices(size_t count, int max_samples) {
    std::vector<size_t> out;
    if (count == 0 || max_samples <= 0) return out;
    size_t n = std::min(count, static_cast<size_t>(max_samples));
    if (n == 1) {
        out.push_back(0);
        return out;
    }
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(n - 1);
        size_t idx = static_cast<size_t>(std::round(t * static_cast<float>(count - 1)));
        if (out.empty() || out.back() != idx) {
            out.push_back(idx);
        }
    }
    return out;
}

/// @brief Implements robust zscore.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void robust_zscore(const std::vector<float>& v, std::vector<float>& out) {
    out.assign(v.size(), 0.0f);
    if (v.empty())
        return;
    std::vector<float> tmp = v;
    float med = median_of(tmp);
    for (float& x : tmp)
        x = std::fabs(x - med);
    float mad = median_of(tmp);
    float sigma = kMadToSigma * mad;
    if (!(sigma > 0.0f)) {
        std::fill(out.begin(), out.end(), 0.0f);
        return;
    }
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = (v[i] - med) / sigma;
    }
}

/// @brief Implements median finite positive.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float median_finite_positive(const std::vector<float>& v, float fallback) {
    std::vector<float> p;
    p.reserve(v.size());
    for (float x : v) {
        if (std::isfinite(x) && x > 0.0f)
            p.push_back(x);
    }
    if (p.empty())
        return fallback;
    return median_of(p);
}

/// @brief Implements median finite.
/// @details Part of filesystem, hashing, robust statistics, string, sampling, and output scaling helpers; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
float median_finite(const std::vector<float>& v, float fallback) {
    std::vector<float> p;
    p.reserve(v.size());
    for (float x : v) {
        if (std::isfinite(x))
            p.push_back(x);
    }
    if (p.empty())
        return fallback;
    return median_of(p);
}

/// @brief Implements stretch to u16 linear from zero inplace using robust
/// maximum (p99.9). Pixels above the robust max are clamped to 65535.
StretchResult stretch_to_u16_linear_from_zero_inplace(Matrix2Df& img) {
    StretchResult result;
    std::vector<float> finite_positive;
    finite_positive.reserve(static_cast<size_t>(img.size()) / 2);
    for (Eigen::Index i = 0; i < img.size(); ++i) {
        const float v = img.data()[i];
        if (std::isfinite(v) && v > 0.0f)
            finite_positive.push_back(v);
    }

    result.sample_count = finite_positive.size();
    if (finite_positive.empty()) return result;

    // Compute p1 as floor and p99.9 as robust maximum.
    // The stretch maps [floor, robust_max] -> [0, 65535].  Using p1 (not p10
    // or the background itself) as the floor keeps below-background pixels
    // visible: the sky background lands at a small positive value instead of
    // consuming >60% of the range, and only the darkest 1% of noise clips to 0.
    const size_t idx_1 = static_cast<size_t>(
        std::clamp(0.01, 0.0, 1.0) * static_cast<double>(finite_positive.size() - 1));
    std::nth_element(finite_positive.begin(),
                     finite_positive.begin() + idx_1,
                     finite_positive.end());
    const float floor_level = finite_positive[idx_1];

    const size_t idx_999 = static_cast<size_t>(
        0.999 * static_cast<double>(finite_positive.size() - 1));
    std::nth_element(finite_positive.begin(),
                     finite_positive.begin() + idx_999,
                     finite_positive.end());
    const float robust_max = finite_positive[idx_999];

    result.low = floor_level;
    result.high = robust_max;
    const float range = robust_max - floor_level;
    if (!(range > 1.0e-6f)) return result;

    const float scale = 65535.0f / range;
    for (Eigen::Index i = 0; i < img.size(); ++i) {
        const float v = img.data()[i];
        if (std::isfinite(v)) {
            const float stretched = (v - floor_level) * scale;
            img.data()[i] = std::clamp(stretched, 0.0f, 65535.0f);
        } else {
            img.data()[i] = 0.0f;
        }
    }

    result.applied = true;
    return result;
}

StretchResult stretch_rgb_to_u32_linear_from_zero_inplace(
    Matrix2Df& r,
    Matrix2Df& g,
    Matrix2Df& b) {
    static const std::vector<uint8_t> empty_mask;
    return stretch_rgb_to_u32_linear_from_zero_inplace(r, g, b, empty_mask);
}

/// @brief Implements per-channel RGB stretch to u32 linear from zero inplace.
/// Each channel gets its own robust p1 floor and p99.9 ceiling; this prevents
/// channels with different dynamic ranges from clipping each other.
///
/// @note Per-channel stretch eliminates cross-channel clipping (e.g. a dim G
/// channel being crushed to 0 by an R-derived floor), but it also normalises
/// each channel to the full output range, which removes the *relative*
/// brightness relationship between channels. A subsequent photometric color
/// calibration (PCC) step is required to restore a physically correct color
/// balance. Without PCC the stretched image may appear color-neutral or
/// slightly desaturated rather than preserving the original channel ratios.
StretchResult stretch_rgb_to_u32_linear_from_zero_inplace(
    Matrix2Df& r,
    Matrix2Df& g,
    Matrix2Df& b,
    const std::vector<uint8_t>& statistics_mask) {
    StretchResult result;
    if (r.rows() != g.rows() || r.rows() != b.rows() ||
        r.cols() != g.cols() || r.cols() != b.cols() ||
        (!statistics_mask.empty() &&
         statistics_mask.size() != static_cast<size_t>(r.size()))) {
        return result;
    }

    // Compute p1 floor and p99.9 ceiling per channel, then apply per channel.
    std::vector<float> all_values;
    all_values.reserve(static_cast<size_t>(r.size()) * 3 / 4);

    auto compute_channel_stretch = [&](const Matrix2Df& ch,
                                       float& floor_level,
                                       float& robust_max) -> size_t {
        std::vector<float> values;
        values.reserve(static_cast<size_t>(ch.size()));
        for (Eigen::Index i = 0; i < ch.size(); ++i) {
            if (!statistics_mask.empty() &&
                statistics_mask[static_cast<size_t>(i)] == 0) {
                continue;
            }
            const float v = ch.data()[i];
            if (std::isfinite(v) && v > 0.0f)
                values.push_back(v);
        }
        if (values.empty())
            return 0;
        const size_t idx_1 = static_cast<size_t>(
            std::clamp(0.01, 0.0, 1.0) *
            static_cast<double>(values.size() - 1));
        std::nth_element(values.begin(), values.begin() + idx_1,
                         values.end());
        floor_level = values[idx_1];

        const size_t idx_999 = static_cast<size_t>(
            std::clamp(0.999, 0.0, 1.0) *
            static_cast<double>(values.size() - 1));
        std::nth_element(values.begin(), values.begin() + idx_999,
                         values.end());
        robust_max = values[idx_999];

        std::copy(values.begin(), values.end(), std::back_inserter(all_values));
        return values.size();
    };

    size_t r_count = compute_channel_stretch(r, result.low_r, result.high_r);
    size_t g_count = compute_channel_stretch(g, result.low_g, result.high_g);
    size_t b_count = compute_channel_stretch(b, result.low_b, result.high_b);
    result.sample_count = all_values.size();
    if (r_count == 0 && g_count == 0 && b_count == 0)
        return result;

    result.low = std::min({result.low_r, result.low_g, result.low_b});
    result.high = std::max({result.high_r, result.high_g, result.high_b});

    auto apply_channel = [](Matrix2Df& ch, float floor_level,
                            float robust_max) {
        const float range = robust_max - floor_level;
        if (!(range > 1.0e-6f))
            return;
        constexpr float target = 4294967295.0f;
        const float scale = target / range;
        for (Eigen::Index i = 0; i < ch.size(); ++i) {
            const float v = ch.data()[i];
            if (std::isfinite(v)) {
                const float stretched = (v - floor_level) * scale;
                ch.data()[i] = std::clamp(stretched, 0.0f, target);
            } else {
                ch.data()[i] = 0.0f;
            }
        }
    };

    apply_channel(r, result.low_r, result.high_r);
    apply_channel(g, result.low_g, result.high_g);
    apply_channel(b, result.low_b, result.high_b);

    result.applied = true;
    return result;
}

} // namespace tile_compile::core
