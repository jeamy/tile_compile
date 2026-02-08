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

#include <openssl/sha.h>

namespace tile_compile::core {

std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::tm tm_buf;
    gmtime_r(&time_t_now, &tm_buf);
    
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return oss.str();
}

std::string get_run_id() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::tm tm_buf;
    localtime_r(&time_t_now, &tm_buf);
    
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

std::vector<fs::path> discover_frames(const fs::path& input_dir, const std::string& pattern) {
    std::vector<fs::path> frames;
    
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        return frames;
    }
    
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (glob_match(pattern, filename)) {
                frames.push_back(entry.path());
            }
        }
    }
    
    std::sort(frames.begin(), frames.end());
    return frames;
}

std::vector<uint8_t> read_bytes(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw IOError("Cannot open file: " + path.string());
    }
    
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw IOError("Cannot read file: " + path.string());
    }
    
    return buffer;
}

std::string read_text(const fs::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw IOError("Cannot open file: " + path.string());
    }
    
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream file(path);
    if (!file) {
        throw IOError("Cannot create file: " + path.string());
    }
    file << text;
}

void safe_hardlink_or_copy(const fs::path& src, const fs::path& dst) {
    std::error_code ec;
    fs::create_hard_link(src, dst, ec);
    if (ec) {
        fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
    }
}

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

std::string sha256_bytes(const std::vector<uint8_t>& data) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(data.data(), data.size(), hash);
    
    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(hash[i]);
    }
    return oss.str();
}

std::string sha256_file(const fs::path& path) {
    auto data = read_bytes(path);
    return sha256_bytes(data);
}

void copy_config(const fs::path& src, const fs::path& dst) {
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
}

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

float compute_median(const Matrix2Df& data) {
    VectorXf flat = Eigen::Map<const VectorXf>(data.data(), data.size());
    return compute_median(flat);
}

float compute_median(const VectorXf& data) {
    if (data.size() == 0) return 0.0f;
    
    std::vector<float> sorted(data.data(), data.data() + data.size());
    std::sort(sorted.begin(), sorted.end());
    
    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n/2 - 1] + sorted[n/2]) / 2.0f;
    } else {
        return sorted[n/2];
    }
}

float compute_mad(const Matrix2Df& data) {
    float median = compute_median(data);
    
    VectorXf deviations(data.size());
    for (Eigen::Index i = 0; i < data.size(); ++i) {
        deviations[i] = std::abs(data.data()[i] - median);
    }
    
    return compute_median(deviations);
}

float compute_robust_sigma(const Matrix2Df& data) {
    return 1.4826f * compute_mad(data);
}

float compute_percentile(const VectorXf& data, float percentile) {
    if (data.size() == 0) return 0.0f;
    
    std::vector<float> sorted(data.data(), data.data() + data.size());
    std::sort(sorted.begin(), sorted.end());
    
    float idx = percentile / 100.0f * (sorted.size() - 1);
    size_t lower = static_cast<size_t>(idx);
    size_t upper = std::min(lower + 1, sorted.size() - 1);
    float frac = idx - lower;
    
    return sorted[lower] * (1.0f - frac) + sorted[upper] * frac;
}

std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool starts_with(const std::string& str, const std::string& prefix) {
    if (prefix.size() > str.size()) return false;
    return str.compare(0, prefix.size(), prefix) == 0;
}

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> parts;
    std::istringstream iss(str);
    std::string part;
    while (std::getline(iss, part, delimiter)) {
        parts.push_back(part);
    }
    return parts;
}

std::string join(const std::vector<std::string>& parts, const std::string& delimiter) {
    std::ostringstream oss;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) oss << delimiter;
        oss << parts[i];
    }
    return oss.str();
}

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

std::vector<fs::path> glob(const fs::path& dir, const std::string& pattern) {
    return discover_frames(dir, pattern);
}

// --- Statistical utilities (canonical, single implementation) ---

float median_of(std::vector<float>& v) {
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

float robust_sigma_mad(std::vector<float>& pixels) {
    if (pixels.empty()) return 0.0f;
    float med = median_of(pixels);
    for (float& x : pixels) x = std::fabs(x - med);
    float mad = median_of(pixels);
    return 1.4826f * mad;
}

float percentile_from_sorted(const std::vector<float>& sorted, float pct) {
    if (sorted.empty()) return 0.0f;
    float clamped = std::min(std::max(pct, 0.0f), 100.0f);
    float pos = (clamped / 100.0f) * static_cast<float>(sorted.size() - 1);
    size_t idx = static_cast<size_t>(std::round(pos));
    idx = std::min(idx, sorted.size() - 1);
    return sorted[idx];
}

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

} // namespace tile_compile::core
