#include "routes/runs_routes.hpp"
#include "routes/route_utils.hpp"
#include "services/report_generator.hpp"
#include "services/run_inspector.hpp"
#include "subprocess_manager.hpp"
#include "services/config_revisions.hpp"
#include "services/scan_summary.hpp"
#include "services/hme_preview_service.hpp"
#include "services/bge_preview_service.hpp"
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <fitsio.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <thread>
#include <chrono>
#include <ctime>
#include <set>
#include <unordered_map>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#ifndef _WIN32
#include <signal.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static crow::response json_resp(const nlohmann::json& j, int status = 200) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}
static crow::response err_resp(const std::string& msg, int status = 400) {
    std::string code = "BAD_REQUEST";
    if (status == 404) code = "NOT_FOUND";
    else if (status == 403) code = "FORBIDDEN";
    else if (status == 422) code = "UNPROCESSABLE_ENTITY";
    else if (status >= 500) code = "INTERNAL_ERROR";
    return json_resp({{"error", {{"code", code}, {"message", msg}, {"details", nlohmann::json::object()}}}}, status);
}
static crow::response err_resp(const std::string& code,
                               const std::string& msg,
                               int status,
                               const nlohmann::json& details) {
    return json_resp({{"error", {{"code", code}, {"message", msg}, {"details", details}}}}, status);
}

static std::string read_method_from_yaml_text_local(const std::string& yaml_text) {
    if (yaml_text.empty()) return "";
    try {
        YAML::Node root = YAML::Load(yaml_text);
        if (root["method"] && root["method"].IsScalar()) {
            return root["method"].as<std::string>();
        }
    } catch (...) {}
    return "";
}

static std::string read_run_method_local(const fs::path& run_dir, const std::string& yaml_override = "") {
    std::string method = read_method_from_yaml_text_local(yaml_override);
    if (!method.empty()) return method;
    std::ifstream f(run_dir / "config.yaml");
    if (f) {
        try {
            YAML::Node root = YAML::Load(f);
            if (root["method"] && root["method"].IsScalar()) {
                return root["method"].as<std::string>();
            }
        } catch (...) {}
    }
    return "";
}

static bool has_nonempty_prewarped_cache(const fs::path& run_dir) {
    std::error_code ec;
    const fs::path cache_dir = run_dir / ".prewarped_cache";
    if (!fs::is_directory(cache_dir, ec)) return false;
    return !fs::is_empty(cache_dir, ec);
}

static std::string read_resume_input_dir_from_events(const fs::path& run_dir) {
    std::ifstream f(run_dir / "logs" / "run_events.jsonl");
    if (!f) return "";
    std::string line;
    std::string last_input_dir;
    while (std::getline(f, line)) {
        if (line.find("input_dir") == std::string::npos) continue;
        try {
            auto ev = nlohmann::json::parse(line);
            if (ev.contains("input_dir") && ev["input_dir"].is_string()) {
                last_input_dir = ev["input_dir"].get<std::string>();
            } else if (ev.contains("payload") && ev["payload"].is_object() &&
                       ev["payload"].contains("input_dir") && ev["payload"]["input_dir"].is_string()) {
                last_input_dir = ev["payload"]["input_dir"].get<std::string>();
            }
        } catch (...) {}
    }
    return last_input_dir;
}

static bool has_synthetic_outputs(const fs::path& run_dir) {
    std::error_code ec;
    const fs::path outputs_dir = run_dir / "outputs";
    if (!fs::is_directory(outputs_dir, ec)) return false;
    for (const auto& entry : fs::directory_iterator(outputs_dir, ec)) {
        if (ec || !entry.is_regular_file(ec)) continue;
        const fs::path path = entry.path();
        const std::string stem = path.stem().string();
        const std::string ext = path.extension().string();
        if (stem.rfind("synthetic_", 0) == 0 && (ext == ".fit" || ext == ".fits")) {
            return true;
        }
    }
    return false;
}

/// @brief Implements sanitize run id.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::string sanitize_run_id(std::string value) {
    for (char& ch : value) {
        bool ok = (ch >= 'a' && ch <= 'z') ||
                  (ch >= 'A' && ch <= 'Z') ||
                  (ch >= '0' && ch <= '9') ||
                  ch == '.' || ch == '_' || ch == '-';
        if (!ok) ch = '_';
    }
    while (!value.empty() && value.front() == '_') value.erase(value.begin());
    while (!value.empty() && value.back() == '_') value.pop_back();
    if (value.empty()) value = "run";
    return value;
}

/// @brief Implements url decode component.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::string url_decode_component(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const char ch = value[i];
        if (ch == '%' && i + 2 < value.size() &&
            std::isxdigit(static_cast<unsigned char>(value[i + 1])) &&
            std::isxdigit(static_cast<unsigned char>(value[i + 2]))) {
            const std::string hex = value.substr(i + 1, 2);
            out.push_back(static_cast<char>(std::stoi(hex, nullptr, 16)));
            i += 2;
            continue;
        }
        if (ch == '+') {
            out.push_back(' ');
            continue;
        }
        out.push_back(ch);
    }
    return out;
}

/// @brief Implements decode base64url.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::string decode_base64url(std::string value) {
    for (char& ch : value) {
        if (ch == '-') ch = '+';
        else if (ch == '_') ch = '/';
    }
    while (value.size() % 4 != 0) value.push_back('=');
    static const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int buffer = 0;
    int bits_collected = 0;
    for (char ch : value) {
        if (ch == '=') break;
        const auto pos = alphabet.find(ch);
        if (pos == std::string::npos) return "";
        buffer = (buffer << 6) | static_cast<int>(pos);
        bits_collected += 6;
        if (bits_collected >= 8) {
            bits_collected -= 8;
            out.push_back(static_cast<char>((buffer >> bits_collected) & 0xFF));
        }
    }
    return out;
}

/// @brief Implements decode run id param.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::string decode_run_id_param(std::string run_id) {
    if (run_id.rfind("b64_", 0) == 0) {
        const std::string decoded = decode_base64url(run_id.substr(4));
        return decoded.empty() ? run_id : decoded;
    }
    if (run_id.find('%') == std::string::npos && run_id.find('+') == std::string::npos) return run_id;
    return url_decode_component(run_id);
}

/// @brief Implements current run timestamp.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::string current_run_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
    return oss.str();
}

/// @brief Implements current run date minutes.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::string current_run_date_minutes() {
    const auto now = std::chrono::system_clock::now();
    const auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%d_%H%M");
    return oss.str();
}

/// @brief Builds effective run id.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::string make_effective_run_id(const nlohmann::json& body) {
    const std::string raw_run_id = body.value("run_id", "");
    if (!raw_run_id.empty()) return sanitize_run_id(raw_run_id);

    const std::string raw_run_name = body.value("run_name", "");
    const std::string base_name = raw_run_name.empty() ? std::string("run")
                                                       : sanitize_run_id(raw_run_name);
    return base_name + "_" + current_run_timestamp();
}

/// @brief Builds queue root run id.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::string make_queue_root_run_id(const nlohmann::json& body) {
    const std::string raw_run_id = body.value("run_id", "");
    if (!raw_run_id.empty()) return sanitize_run_id(raw_run_id);

    const std::string raw_run_name = body.value("run_name", "");
    if (!raw_run_name.empty()) {
        return sanitize_run_id(raw_run_name) + "_" + current_run_timestamp();
    }

    return current_run_date_minutes();
}

/// @brief Implements wildcard to regex.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::string wildcard_to_regex(const std::string& pattern) {
    std::string out;
    out.reserve(pattern.size() * 2 + 4);
    out += '^';
    for (char ch : pattern) {
        switch (ch) {
            case '*': out += ".*"; break;
            case '?': out += '.'; break;
            case '.': case '+': case '(': case ')': case '[': case ']':
            case '{': case '}': case '^': case '$': case '|': case '\\':
                out += '\\';
                out += ch;
                break;
            default:
                out += ch;
                break;
        }
    }
    out += '$';
    return out;
}

/// @brief Implements wildcard match.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static bool wildcard_match(const std::string& pattern, const std::string& value) {
    if (pattern.empty()) return true;
    try {
        return std::regex_match(value, std::regex(wildcard_to_regex(pattern), std::regex::icase));
    } catch (...) {
        return value == pattern;
    }
}

static std::optional<ConfigRevision> resolve_config_revision(const std::shared_ptr<AppState>& state,
                                                             const fs::path& run_dir,
                                                             const std::string& revision_id) {
    if (revision_id.empty()) return std::nullopt;
    if (auto run_revision = get_run_config_revision(run_dir, revision_id)) return run_revision;
    return state->revision_store.get(revision_id);
}

static std::optional<crow::response> resolve_request_run_dir(const std::shared_ptr<AppState>& state,
                                                             const std::string& run_id,
                                                             const std::string& run_dir_str,
                                                             fs::path& run_dir) {
    if (run_dir_str.empty()) {
        try {
            run_dir = state->runtime.resolve_run_dir(run_id);
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
        return std::nullopt;
    }

    auto resolved = state->runtime.resolve_input_path(fs::path(run_dir_str), true);
    run_dir = resolved.path;
    if (resolved.status == PathStatus::not_allowed) {
        return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + run_dir.string(), 403, {{"path", run_dir.string()}});
    }
    if (resolved.status == PathStatus::not_found || !fs::is_directory(run_dir)) {
        return err_resp("PATH_NOT_FOUND", "run_dir does not exist: " + run_dir.string(), 404, {{"path", run_dir.string()}});
    }
    return std::nullopt;
}

static void persist_run_config_snapshot(const fs::path& run_dir,
                                        const std::string& yaml_text,
                                        const std::string& source,
                                        const std::optional<std::string>& run_id = std::nullopt) {
    if (yaml_text.empty()) return;
    auto revisions = list_run_config_revisions(run_dir);
    if (!revisions.empty()) {
        auto latest = get_run_config_revision(run_dir, revisions.front().revision_id);
        if (latest && latest->yaml_text == yaml_text) return;
    }
    add_run_config_revision(run_dir, yaml_text, source, run_id);
}

/// @brief Resolves artifact path.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::optional<fs::path> resolve_artifact_path(const fs::path& run_dir, const std::string& raw_path) {
    const std::string trimmed = raw_path;
    if (trimmed.empty()) return std::nullopt;

    std::error_code ec;
    const fs::path run_dir_resolved = fs::weakly_canonical(run_dir, ec);
    if (ec) return std::nullopt;

    fs::path candidate = fs::path(trimmed);
    if (!candidate.is_absolute()) candidate = run_dir_resolved / candidate;
    candidate = fs::weakly_canonical(candidate, ec);
    if (ec) return std::nullopt;

    if (candidate != run_dir_resolved) {
        const fs::path rel = candidate.lexically_relative(run_dir_resolved);
        if (rel.empty()) return std::nullopt;
        for (const auto& part : rel) {
            if (part == "..") return std::nullopt;
        }
    }
    return candidate;
}

static size_t count_run_event_lines_for_resume(const fs::path& run_dir) {
    const std::vector<fs::path> candidates = {
        run_dir / "logs" / "run_events.jsonl",
        run_dir / "events.jsonl",
        run_dir / "logs" / "events.jsonl"
    };
    for (const auto& path : candidates) {
        if (!fs::exists(path)) continue;
        std::ifstream in(path);
        size_t count = 0;
        std::string line;
        while (std::getline(in, line)) ++count;
        return count;
    }
    return 0;
}

static std::string fits_status_text(int status) {
    char text[FLEN_STATUS]{};
    fits_get_errstatus(status, text);
    return text;
}

static std::vector<float> read_fits_plane_preview(const fs::path& path, long plane,
                                                  long& width, long& height, long& planes) {
    fitsfile* file = nullptr;
    int status = 0;
    if (fits_open_file(&file, path.string().c_str(), READONLY, &status))
        throw std::runtime_error("Cannot open FITS: " + fits_status_text(status));
    int naxis = 0;
    long axes[3]{1, 1, 1};
    if (fits_get_img_dim(file, &naxis, &status) ||
        fits_get_img_size(file, 3, axes, &status) || naxis < 2) {
        fits_close_file(file, &status);
        throw std::runtime_error("Invalid FITS image");
    }
    width = axes[0];
    height = axes[1];
    planes = naxis >= 3 ? axes[2] : 1;
    if (plane < 1 || plane > planes) {
        fits_close_file(file, &status);
        throw std::runtime_error("Missing FITS plane");
    }
    std::vector<float> pixels(static_cast<size_t>(width * height));
    long first[3]{1, 1, plane};
    int any_null = 0;
    if (fits_read_pix(file, TFLOAT, first, width * height, nullptr, pixels.data(), &any_null, &status)) {
        const auto message = fits_status_text(status);
        fits_close_file(file, &status);
        throw std::runtime_error("Cannot read FITS: " + message);
    }
    fits_close_file(file, &status);
    return pixels;
}

static cv::Mat plane_to_mat(const std::vector<float>& values, long width, long height) {
    cv::Mat mat(static_cast<int>(height), static_cast<int>(width), CV_32F);
    for (long y = 0; y < height; ++y)
        for (long x = 0; x < width; ++x)
            mat.at<float>(static_cast<int>(y), static_cast<int>(x)) = values[static_cast<size_t>(y * width + x)];
    return mat;
}

static std::pair<float, float> robust_range(const std::vector<cv::Mat>& planes) {
    std::vector<float> sample;
    for (const auto& mat : planes) {
        const int stride = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(mat.rows * mat.cols) / 250000.0)));
        for (int y = 0; y < mat.rows; y += stride) {
            for (int x = 0; x < mat.cols; x += stride) {
                const float v = mat.at<float>(y, x);
                if (std::isfinite(v)) sample.push_back(v);
            }
        }
    }
    if (sample.empty()) return {0.0f, 1.0f};
    std::sort(sample.begin(), sample.end());
    auto at = [&](double q) {
        const size_t idx = std::min(sample.size() - 1, static_cast<size_t>(std::llround(q * static_cast<double>(sample.size() - 1))));
        return sample[idx];
    };
    float lo = at(0.01);
    float hi = at(0.995);
    if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo) {
        lo = sample.front();
        hi = sample.back();
    }
    if (hi <= lo) hi = lo + 1.0f;
    return {lo, hi};
}

static std::vector<unsigned char> render_fits_preview_png(const fs::path& path, int max_edge = 1400) {
    long w = 0, h = 0, p = 0;
    auto rv = read_fits_plane_preview(path, 1, w, h, p);
    cv::Mat r = plane_to_mat(rv, w, h);
    cv::Mat g = r;
    cv::Mat b = r;
    if (p >= 3) {
        long wg = 0, hg = 0, pg = 0;
        auto gv = read_fits_plane_preview(path, 2, wg, hg, pg);
        g = plane_to_mat(gv, wg, hg);
        long wb = 0, hb = 0, pb = 0;
        auto bv = read_fits_plane_preview(path, 3, wb, hb, pb);
        b = plane_to_mat(bv, wb, hb);
        if (g.cols != r.cols || g.rows != r.rows || b.cols != r.cols || b.rows != r.rows)
            throw std::runtime_error("RGB FITS planes have different dimensions");
    }
    const int edge = std::max(r.rows, r.cols);
    if (edge > max_edge) {
        const double scale = static_cast<double>(max_edge) / static_cast<double>(edge);
        const cv::Size size(std::max(1, static_cast<int>(std::lround(r.cols * scale))),
                            std::max(1, static_cast<int>(std::lround(r.rows * scale))));
        cv::resize(r, r, size, 0, 0, cv::INTER_AREA);
        cv::resize(g, g, size, 0, 0, cv::INTER_AREA);
        cv::resize(b, b, size, 0, 0, cv::INTER_AREA);
    }
    const auto [lo, hi] = robust_range({r, g, b});
    const float denom = std::max(hi - lo, 1e-9f);
    cv::Mat out(r.rows, r.cols, CV_8UC3);
    for (int y = 0; y < r.rows; ++y) {
        for (int x = 0; x < r.cols; ++x) {
            auto convert = [&](float v) -> unsigned char {
                if (!std::isfinite(v)) v = lo;
                float n = std::clamp((v - lo) / denom, 0.0f, 1.0f);
                n = std::pow(n, 0.6f);
                return cv::saturate_cast<unsigned char>(n * 255.0f);
            };
            auto& px = out.at<cv::Vec3b>(y, x);
            px[2] = convert(r.at<float>(y, x));
            px[1] = convert(g.at<float>(y, x));
            px[0] = convert(b.at<float>(y, x));
        }
    }
    std::vector<unsigned char> png;
    if (!cv::imencode(".png", out, png)) throw std::runtime_error("PNG encoding failed");
    return png;
}

/// @brief Applies color mode to yaml.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::string apply_color_mode_to_yaml(const std::string& base_yaml, const std::string& color_mode) {
    if (base_yaml.empty() || color_mode.empty()) return base_yaml;
    try {
        YAML::Node root = YAML::Load(base_yaml);
        if (!root["data"] || !root["data"].IsMap()) root["data"] = YAML::Node(YAML::NodeType::Map);
        root["data"]["color_mode"] = color_mode;
        std::ostringstream oss;
        oss << root;
        return oss.str();
    } catch (...) {
        return base_yaml;
    }
}

static std::string normalize_path_separators(const std::string& p) {
    std::string out = p;
    std::replace(out.begin(), out.end(), '\\', '/');
    return out;
}

static std::string apply_astrometry_paths_to_yaml(const std::string& base_yaml, 
                                                   const std::string& astap_bin,
                                                   const std::string& astap_data_dir) {
    if (base_yaml.empty() || (astap_bin.empty() && astap_data_dir.empty())) return base_yaml;
    try {
        YAML::Node root = YAML::Load(base_yaml);
        if (!root["astrometry"] || !root["astrometry"].IsMap()) root["astrometry"] = YAML::Node(YAML::NodeType::Map);
        if (!astap_bin.empty()) root["astrometry"]["astap_bin"] = normalize_path_separators(astap_bin);
        if (!astap_data_dir.empty()) root["astrometry"]["astap_data_dir"] = normalize_path_separators(astap_data_dir);
        std::ostringstream oss;
        oss << root;
        return oss.str();
    } catch (...) {
        return base_yaml;
    }
}

static std::string effective_config_yaml(const std::shared_ptr<AppState>& state,
                                         const std::string& config_yaml,
                                         const std::string& color_mode,
                                         const std::string& astap_bin = "",
                                         const std::string& astap_data_dir = "") {
    std::string yaml = config_yaml.empty() ? tile_compile::routes::read_file_str(state->runtime.default_config_path) : config_yaml;
    yaml = apply_color_mode_to_yaml(yaml, color_mode);
    yaml = apply_astrometry_paths_to_yaml(yaml, astap_bin, astap_data_dir);
    return yaml;
}

static fs::path materialize_queue_input(const fs::path& input_dir,
                                        const std::string& pattern,
                                        const fs::path& staging_root,
                                        int item_index) {
    if (pattern.empty()) return input_dir;
    fs::path staging_dir = staging_root / ("item_" + std::to_string(item_index + 1));
    std::error_code ec;
    fs::remove_all(staging_dir, ec);
    fs::create_directories(staging_dir, ec);
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (!wildcard_match(pattern, name)) continue;
        fs::path target = staging_dir / entry.path().filename();
        std::error_code link_ec;
        fs::create_symlink(entry.path(), target, link_ec);
        if (link_ec) {
            std::error_code copy_ec;
            fs::copy_file(entry.path(), target, fs::copy_options::overwrite_existing, copy_ec);
        }
    }
    return staging_dir;
}

static std::string derive_queue_run_id(const std::string& base_run_id,
                                       const nlohmann::json& item,
                                       int index,
                                       std::unordered_map<std::string, int>& filter_counts) {
    std::string filter = sanitize_run_id(item.value("filter", ""));
    if (filter.empty() || filter == "run") {
        std::ostringstream fallback;
        fallback << "item-" << (index + 1);
        filter = fallback.str();
    }

    int& count = filter_counts[filter];
    count += 1;
    std::string leaf = filter;
    if (count > 1) {
        leaf += "-" + std::to_string(count);
    }

    if (base_run_id.empty()) return leaf;
    return (fs::path(base_run_id) / leaf).generic_string();
}

/// @brief Implements collect queue items.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static nlohmann::json collect_queue_items(const nlohmann::json& body, const std::string& base_run_id) {
    nlohmann::json queue = nlohmann::json::array();
    if (!body.contains("queue") || !body["queue"].is_array()) return queue;
    int index = 0;
    std::unordered_map<std::string, int> filter_counts;
    for (const auto& raw : body["queue"]) {
        if (!raw.is_object()) continue;
        std::string input_dir = raw.value("input_dir", raw.value("input_path", ""));
        if (input_dir.empty()) continue;
        nlohmann::json item = {
            {"filter", raw.value("filter", "")},
            {"input_dir", input_dir},
            {"pattern", raw.value("pattern", "")},
            {"run_id", derive_queue_run_id(base_run_id, raw, index, filter_counts)},
            {"state", "pending"}
        };
        queue.push_back(item);
        ++index;
    }
    if (!queue.empty()) return queue;
    if (!body.contains("input_dirs") || !body["input_dirs"].is_array()) return queue;
    filter_counts.clear();
    for (const auto& raw : body["input_dirs"]) {
        std::string input_dir;
        if (raw.is_string()) {
            input_dir = raw.get<std::string>();
        } else if (raw.is_object()) {
            input_dir = raw.value("input_dir", raw.value("input_path", ""));
        }
        if (input_dir.empty()) continue;
        nlohmann::json item = {
            {"filter", raw.is_object() ? raw.value("filter", "") : ""},
            {"input_dir", input_dir},
            {"pattern", raw.is_object() ? raw.value("pattern", "") : ""},
            {"run_id", derive_queue_run_id(base_run_id, raw.is_object() ? raw : nlohmann::json::object(), index, filter_counts)},
            {"state", "pending"}
        };
        queue.push_back(item);
        ++index;
    }
    return queue;
}

/// @brief Builds queue cancel requested.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static bool queue_cancel_requested(InMemoryJobStore& store, const std::string& job_id) {
    auto job = store.get(job_id);
    return job && job->state == JobState::cancelled;
}

/// @brief Checks whether terminal job state.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static bool is_terminal_job_state(const std::string& state) {
    return state == "ok" || state == "error" || state == "cancelled";
}

static nlohmann::json queue_job_payload(const nlohmann::json& queue,
                                        int current_index,
                                        const std::string& current_run_id,
                                        const std::string& runs_dir) {
    int done = 0;
    for (const auto& item : queue) {
        std::string state = item.value("state", "pending");
        if (state == "ok") ++done;
    }
    return {
        {"run_id", current_run_id},
        {"runs_dir", runs_dir},
        {"current_index", current_index},
        {"done", done},
        {"total", static_cast<int>(queue.size())},
        {"queue", queue}
    };
}

/// @brief Normalizes queue filter name.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::string normalize_queue_filter_name(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (char ch : raw) {
        if (std::isspace(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-') continue;
        out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
    }
    if (out == "HALPHA") return "HA";
    return out;
}

/// @brief Builds queue items for run.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static nlohmann::json queue_items_for_run(InMemoryJobStore& store, const std::string& run_id) {
    for (const auto& job : store.list(200)) {
        if (job.type != "run_queue") continue;
        if (!job.data.is_object()) continue;
        const auto queue = job.data.value("queue", nlohmann::json::array());
        if (job.data.value("run_id", std::string()) != run_id && !queue_contains_run_id(queue, run_id)) continue;
        if (!queue.is_array()) return nlohmann::json::array();
        return queue;
    }
    return nlohmann::json::array();
}

/// @brief Builds queue filters for run.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static nlohmann::json queue_filters_for_run(InMemoryJobStore& store, const std::string& run_id) {
    const auto queue = queue_items_for_run(store, run_id);
    if (!queue.is_array()) return nlohmann::json::array();
    if (queue.empty()) return nlohmann::json::array();

    struct FilterAggregate {
        std::string label;
        int total{0};
        int ok{0};
        int running{0};
        int pending{0};
        int error{0};
        int cancelled{0};
    };

    nlohmann::json filters = nlohmann::json::array();
    std::vector<std::string> order;
    std::unordered_map<std::string, FilterAggregate> grouped;
    for (const auto& item : queue) {
        if (!item.is_object()) continue;
        std::string filter = normalize_queue_filter_name(item.value("filter", std::string()));
        if (filter.empty()) continue;
        if (grouped.find(filter) == grouped.end()) {
            order.push_back(filter);
            grouped.emplace(filter, FilterAggregate{filter});
        }
        auto& agg = grouped[filter];
        agg.total += 1;
        const std::string state = item.value("state", std::string("pending"));
        if (state == "ok") agg.ok += 1;
        else if (state == "running") agg.running += 1;
        else if (state == "error") agg.error += 1;
        else if (state == "cancelled") agg.cancelled += 1;
        else agg.pending += 1;
    }

    for (const auto& filter : order) {
        const auto& agg = grouped.at(filter);
        std::string state = "pending";
        if (agg.error > 0) state = "error";
        else if (agg.cancelled > 0) state = "cancelled";
        else if (agg.running > 0 || (agg.ok > 0 && agg.ok < agg.total)) state = "running";
        else if (agg.ok > 0 && agg.ok == agg.total) state = "ok";
        filters.push_back({
            {"filter", agg.label},
            {"state", state},
            {"done", agg.ok},
            {"total", agg.total}
        });
    }
    return filters;
}

static std::optional<nlohmann::json> pending_run_status(const std::shared_ptr<AppState>& state,
                                                        const std::string& run_id) {
    const auto latest_job = latest_run_job(state->job_store, run_id);
    if (!latest_job.has_value()) return std::nullopt;

    const std::string runs_dir = latest_job->data.is_object()
        ? latest_job->data.value("runs_dir", state->runtime.runs_dir.string())
        : state->runtime.runs_dir.string();
    const fs::path predicted_run_dir = fs::path(runs_dir) / run_id;
    nlohmann::json result{
        {"run_id", run_id},
        {"run_dir", predicted_run_dir.string()},
        {"status", "unknown"},
        {"color_mode", "UNKNOWN"},
        {"aqmh_enabled", nullptr},
        {"queue", queue_items_for_run(state->job_store, run_id)},
        {"queue_filters", queue_filters_for_run(state->job_store, run_id)},
        {"current_phase", nullptr},
        {"progress", 0.0},
        {"phases", nlohmann::json::array()},
        {"events", nlohmann::json::array()},
    };
    apply_job_state_to_run_status(result, latest_job);
    if (result.value("status", std::string()) == "unknown") return std::nullopt;
    if (result["progress"].is_number()) {
        try {
            result["progress"] = std::clamp(latest_job->progress / 100.0, 0.0, 1.0);
        } catch (...) {}
    }
    return result;
}

#ifndef _WIN32
/// @brief Checks process exists.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static bool pid_exists(pid_t pid) {
    if (pid <= 0) return false;
    if (kill(pid, 0) == 0) return true;
    return errno == EPERM;
}

/// @brief Implements terminate pid group.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static bool terminate_pid_group(pid_t pid) {
    if (pid <= 0) return false;
    // Send to process group (pgid == pid) to also kill child processes
    if (kill(-pid, SIGTERM) != 0) kill(pid, SIGTERM);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (!pid_exists(pid)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (kill(-pid, SIGKILL) != 0) kill(pid, SIGKILL);
    return true;
}

static std::vector<int> terminate_orphan_runner_processes(const BackendRuntime& runtime,
                                                          const std::string& run_id,
                                                          const std::string& run_dir) {
    std::vector<int> killed;
#ifdef __linux__
    const std::string runner_name = fs::path(runtime.runner_exe).filename().string();
    if (!fs::exists("/proc")) return killed;
    for (const auto& entry : fs::directory_iterator("/proc")) {
        if (!entry.is_directory()) continue;
        const std::string pid_text = entry.path().filename().string();
        if (pid_text.empty() || !std::all_of(pid_text.begin(), pid_text.end(), ::isdigit)) continue;
        int pid = 0;
        try {
            pid = std::stoi(pid_text);
        } catch (...) {
            continue;
        }
        if (pid == static_cast<int>(::getpid())) continue;

        std::ifstream cmdline(entry.path() / "cmdline", std::ios::binary);
        if (!cmdline) continue;
        std::string raw((std::istreambuf_iterator<char>(cmdline)), std::istreambuf_iterator<char>());
        if (raw.empty()) continue;

        std::vector<std::string> argv;
        size_t start = 0;
        while (start < raw.size()) {
            size_t end = raw.find('\0', start);
            if (end == std::string::npos) end = raw.size();
            if (end > start) argv.push_back(raw.substr(start, end - start));
            start = end + 1;
        }
        if (argv.empty()) continue;

        const std::string exe_name = fs::path(argv.front()).filename().string();
        if (exe_name != runner_name && exe_name.find("tile_compile_runner") == std::string::npos) continue;
        const bool is_runner = std::any_of(argv.begin() + 1, argv.end(), [](const std::string& arg) {
            return arg == "run" || arg == "resume";
        });
        if (!is_runner) continue;

        std::string joined;
        for (const auto& part : argv) {
            if (!joined.empty()) joined += ' ';
            joined += part;
        }
        if (joined.find(run_id) == std::string::npos && joined.find(run_dir) == std::string::npos) continue;
        if (terminate_pid_group(static_cast<pid_t>(pid))) killed.push_back(pid);
    }
#endif
    return killed;
}
#endif

static std::vector<std::string> runner_run_args(const std::shared_ptr<AppState>& state,
                                                const std::string& config_path,
                                                const std::string& input_dir,
                                                const std::string& runs_dir,
                                                const std::string& run_id) {
    std::vector<std::string> args = {state->runtime.runner_exe, "run"};
    args.push_back("--config"); args.push_back(config_path);
    args.push_back("--input-dir"); args.push_back(input_dir);
    args.push_back("--runs-dir"); args.push_back(runs_dir);
    args.push_back("--project-root"); args.push_back(state->runtime.project_root.string());
    if (!run_id.empty()) { args.push_back("--run-id"); args.push_back(run_id); }
    return args;
}

/// @brief Implements collect input dirs.
/// @details This implementation serves run listing, status, queue, resume, artifact, and report endpoints; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
static std::vector<std::string> collect_input_dirs(const nlohmann::json& body) {
    std::vector<std::string> input_dirs;

    std::string single = body.value("input_dir", body.value("input_path", ""));
    if (!single.empty()) input_dirs.push_back(single);

    if (body.contains("input_dirs") && body["input_dirs"].is_array()) {
        for (const auto& item : body["input_dirs"]) {
            if (item.is_string()) {
                std::string value = item.get<std::string>();
                if (!value.empty()) input_dirs.push_back(value);
            } else if (item.is_object() && item.contains("input_dir") && item["input_dir"].is_string()) {
                std::string value = item["input_dir"].get<std::string>();
                if (!value.empty()) input_dirs.push_back(value);
            } else if (item.is_object() && item.contains("input_path") && item["input_path"].is_string()) {
                std::string value = item["input_path"].get<std::string>();
                if (!value.empty()) input_dirs.push_back(value);
            }
        }
    }

    return input_dirs;
}

/// @brief Registers run endpoints for discovery, launch, resume, stop, artifacts, logs, reports, and queue state.
/// @details This is the route-group entry point called from main during Crow setup.
void register_runs_routes(CrowApp& app,
                           std::shared_ptr<AppState> state) {
 
    CROW_ROUTE(app, "/api/runs").methods("GET"_method)
    ([state]() {
        auto runs = discover_runs(state->runtime.runs_dir, 100);
        return json_resp({{"items", runs}, {"total", runs.size()}});
    });

    CROW_ROUTE(app, "/api/runs/start").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body_opt = tile_compile::routes::parse_body(req);
        if (!body_opt) return err_resp("Invalid JSON");
        auto& body = *body_opt;

        auto guardrails = scan_guardrails(state->job_store);
        if (guardrails.value("status", std::string()) == "error") {
            state->ui_event_store.push("run.start.blocked", "runs.run_start", {{"reason", "guardrail_error"}, {"guardrails", guardrails}});
            return err_resp("GUARDRAIL_BLOCKED", "run start blocked by guardrails", 409, guardrails);
        }

        auto input_dirs = collect_input_dirs(body);
        std::string runs_dir    = body.value("runs_dir", state->runtime.runs_dir.string());
        std::string run_id      = make_effective_run_id(body);
        std::string color_mode  = body.value("color_mode", "");
        std::string config_yaml = body.value("config_yaml", "");
        std::string astap_bin   = body.value("astap_bin", "");
        std::string astap_data_dir = body.value("astap_data_dir", "");
        std::string base_run_id = sanitize_run_id(run_id.empty() ? "run" : run_id);
        if ((body.contains("queue") && body["queue"].is_array()) ||
            (body.contains("input_dirs") && body["input_dirs"].is_array() && !body["input_dirs"].empty())) {
            base_run_id = make_queue_root_run_id(body);
        }
        auto queue_items = collect_queue_items(body, base_run_id);

        if (input_dirs.empty() && queue_items.empty())
            return err_resp("BAD_REQUEST", "input_dir is required", 400, nlohmann::json::object());

        for (auto& dir : input_dirs) {
            fs::path p(dir);
            if (auto err = tile_compile::routes::validate_path(state, p, !p.is_absolute())) return std::move(*err);
            dir = p.string();
        }
        if (queue_items.is_array()) {
            for (auto& item : queue_items) {
                if (!item.is_object()) continue;
                std::string raw_input_dir = item.value("input_dir", "");
                if (raw_input_dir.empty()) continue;
                fs::path p(raw_input_dir);
                if (auto err = tile_compile::routes::validate_path(state, p, !p.is_absolute())) return std::move(*err);
                item["input_dir"] = p.string();
            }
        }

        if (!state->runtime.is_path_allowed(fs::path(runs_dir))) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + runs_dir, 403, {{"path", runs_dir}});
        }

        std::string prepared_config_yaml = effective_config_yaml(state, config_yaml, color_mode, astap_bin, astap_data_dir);
        std::string effective_config_path = state->runtime.default_config_path.string();
        if (!prepared_config_yaml.empty()) {
            std::ofstream out(state->runtime.default_config_path);
            if (!out) return err_resp("Cannot write config: " + state->runtime.default_config_path.string(), 500);
            out << prepared_config_yaml;
        }
        std::string revision_id = state->revision_store.add(state->runtime.default_config_path, prepared_config_yaml, "run_start", base_run_id);
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->active_config_revision_id = revision_id;
        }

        if (!queue_items.empty()) {
            std::string effective_run_id = queue_items.front().value("run_id", base_run_id);
            auto queue_payload = queue_job_payload(queue_items, 0, effective_run_id, runs_dir);
            queue_payload["config_revision_id"] = revision_id;
            std::string job_id = tile_compile::routes::spawn_job_thread(state, "run_queue", effective_run_id, queue_payload,
                [queue_items, runs_dir, effective_config_path, prepared_config_yaml](std::shared_ptr<AppState> state, const std::string& job_id) mutable {
                fs::path staging_root = fs::path(runs_dir) / ".queue_staging" / job_id;
                std::error_code ec;
                fs::create_directories(staging_root, ec);
                nlohmann::json queue = queue_items;
                for (size_t i = 0; i < queue.size(); ++i) {
                    if (queue_cancel_requested(state->job_store, job_id)) {
                        std::string cancelled_run_id = (i < queue.size() && queue[i].is_object()) ? queue[i].value("run_id", "") : std::string();
                        state->job_store.update_state(job_id, JobState::cancelled,
                            queue_job_payload(queue, static_cast<int>(i), cancelled_run_id, runs_dir));
                        fs::remove_all(staging_root, ec);
                        return;
                    }

                    for (size_t j = 0; j < queue.size(); ++j) {
                        std::string s = queue[j].value("state", "pending");
                        if (j < i && s != "ok") queue[j]["state"] = "ok";
                        else if (j == i) queue[j]["state"] = "running";
                        else if (s != "ok") queue[j]["state"] = "pending";
                    }

                    std::string current_run_id = queue[i].value("run_id", "");
                    {
                        std::lock_guard<std::mutex> lk(state->state_mutex);
                        state->current_run_id = current_run_id;
                    }

                    persist_run_config_snapshot(fs::path(runs_dir) / current_run_id,
                                                prepared_config_yaml,
                                                "run_start",
                                                current_run_id);

                    state->job_store.update_state(job_id, JobState::running,
                        queue_job_payload(queue, static_cast<int>(i), current_run_id, runs_dir));
                    state->job_store.update_progress(job_id, queue.empty() ? 100.0 : (100.0 * i / queue.size()));

                    fs::path input_dir = fs::path(queue[i].value("input_dir", ""));
                    std::string pattern = queue[i].value("pattern", "");
                    fs::path effective_input_dir = materialize_queue_input(input_dir, pattern, staging_root, static_cast<int>(i));

                    auto args = runner_run_args(state, effective_config_path, effective_input_dir.string(), runs_dir, current_run_id);
                    std::string child_job_id = state->subprocess_manager.launch("run", args,
                                                                                  state->runtime.project_root.string(),
                                                                                  current_run_id);
                    while (true) {
                        auto child_job = state->job_store.get(child_job_id);
                        if (!child_job) break;
                        std::string child_state = job_state_str(child_job->state);
                        queue[i]["result"] = child_job->data;
                        queue[i]["job_id"] = child_job_id;
                        queue[i]["state"] = child_state;
                        state->job_store.update_state(job_id, JobState::running,
                            queue_job_payload(queue, static_cast<int>(i), current_run_id, runs_dir));
                        state->job_store.update_progress(job_id, queue.empty() ? 100.0 : (100.0 * i / queue.size()));
                        if (queue_cancel_requested(state->job_store, job_id) && !is_terminal_job_state(child_state)) {
                            state->subprocess_manager.cancel(child_job_id);
                        }
                        if (is_terminal_job_state(child_state)) {
                            if (child_state == "ok") {
                                queue[i]["state"] = "ok";
                                break;
                            }
                            JobState final_state = child_state == "cancelled" ? JobState::cancelled : JobState::error;
                            state->job_store.update_state(job_id, final_state,
                                queue_job_payload(queue, static_cast<int>(i), current_run_id, runs_dir),
                                child_job->error_message);
                            fs::remove_all(staging_root, ec);
                            return;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    }
                }

                std::string final_run_id = queue.empty() ? "" : queue.back().value("run_id", "");
                state->job_store.update_progress(job_id, 100.0);
                state->job_store.update_state(job_id, JobState::ok,
                    queue_job_payload(queue, static_cast<int>(queue.size()) - 1, final_run_id, runs_dir));
                fs::remove_all(staging_root, ec);
            });
            state->ui_event_store.push(
                "run.start.queue",
                "runs.run_start",
                {{"revision_id", revision_id}, {"queue_size", static_cast<int>(queue_items.size())}},
                effective_run_id,
                job_id);
            return json_resp({{"job_id", job_id}, {"run_id", effective_run_id}}, 202);
        }

        std::string effective_run_id = sanitize_run_id(run_id.empty() ? "run" : run_id);
        persist_run_config_snapshot(fs::path(runs_dir) / effective_run_id,
                                    prepared_config_yaml,
                                    "run_start",
                                    effective_run_id);
        auto args = runner_run_args(state, effective_config_path, input_dirs.front(), runs_dir, effective_run_id);
        std::string job_id = state->subprocess_manager.launch("run", args,
                                                               state->runtime.project_root.string(),
                                                               effective_run_id);
        state->job_store.update_state(job_id, JobState::running, {
            {"input_dir", input_dirs.front()},
            {"runs_dir", runs_dir},
            {"run_id", effective_run_id},
            {"command", args},
            {"config_revision_id", revision_id}
        });
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->current_run_id = effective_run_id;
        }
        state->ui_event_store.push(
            "run.start",
            "runs.run_start",
            {{"input_dir", input_dirs.front()}, {"runs_dir", runs_dir}, {"revision_id", revision_id}},
            effective_run_id,
            job_id);
        return json_resp({{"job_id", job_id}, {"run_id", effective_run_id}}, 202);
    });

    CROW_ROUTE(app, "/api/runs/<string>/status").methods("GET"_method)
    ([state](const crow::request&, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        const auto job = latest_run_job(state->job_store, run_id);
        std::string alt_runs_dir;
        if (job && job->data.is_object()) {
            alt_runs_dir = job->data.value("runs_dir", "");
        }
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id, alt_runs_dir);
            auto status  = read_run_status(run_dir);
            apply_job_state_to_run_status(status, job);
            apply_runtime_liveness_to_run_status(status, job, state->runtime.runner_exe, run_id, run_dir.string());
            return json_resp({
                {"run_id", run_id},
                {"run_dir", run_dir.string()},
                {"status", status.value("status", "unknown")},
                {"color_mode", status.value("color_mode", "UNKNOWN")},
                {"method", status.value("method", "aqmh")},
                {"aqmh_enabled", status.contains("aqmh_enabled") ? status["aqmh_enabled"] : nlohmann::json(nullptr)},
                {"queue", queue_items_for_run(state->job_store, run_id)},
                {"queue_filters", queue_filters_for_run(state->job_store, run_id)},
                {"current_phase", status.contains("current_phase") ? status["current_phase"] : nlohmann::json(nullptr)},
                {"progress", status.value("progress", 0.0)},
                {"phases", status.value("phases", nlohmann::json::array())},
                {"events", status.value("events", nlohmann::json::array())},
            });
        } catch (const std::exception& e) {
            if (auto pending = pending_run_status(state, run_id)) {
                return json_resp(*pending);
            }
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/config").methods("GET"_method)
    ([state](const crow::request&, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            const fs::path config_path = run_dir / "config.yaml";
            return json_resp({
                {"run_id", run_id},
                {"run_dir", run_dir.string()},
                {"path", config_path.string()},
                {"config", tile_compile::routes::read_file_str(config_path)},
            });
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/config-revisions").methods("GET"_method)
    ([state](const crow::request&, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            auto revisions = list_run_config_revisions(run_dir);
            nlohmann::json items = nlohmann::json::array();
            for (const auto& revision : revisions) items.push_back(config_revision_to_json(revision));
            return json_resp({
                {"run_id", run_id},
                {"run_dir", run_dir.string()},
                {"dir", run_config_revisions_dir(run_dir).string()},
                {"items", items},
            });
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/config-revisions/<string>").methods("GET"_method)
    ([state](const crow::request&, std::string run_id, std::string rev_id) {
        run_id = decode_run_id_param(run_id);
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            auto revision = get_run_config_revision(run_dir, rev_id);
            if (!revision) return err_resp("NOT_FOUND", "revision '" + rev_id + "' not found", 404, nlohmann::json::object());
            return json_resp({
                {"run_id", run_id},
                {"revision_id", rev_id},
                {"path", revision->path},
                {"source", revision->source},
                {"created_at", revision->created_at},
                {"config", revision->yaml_text},
            });
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/stop").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        std::string runs_dir = req.url_params.get("runs_dir") ? req.url_params.get("runs_dir") : state->runtime.runs_dir.string();
        fs::path resolved_run_dir;
        bool has_resolved_run_dir = false;
        if (run_id == "pending") {
            resolved_run_dir = fs::path(runs_dir);
            has_resolved_run_dir = true;
        } else {
            try {
                resolved_run_dir = state->runtime.resolve_run_dir(run_id);
                has_resolved_run_dir = true;
            } catch (...) {}
        }

        auto jobs = state->job_store.list(500);
        nlohmann::json cancelled_jobs = nlohmann::json::array();
        nlohmann::json killed_pids = nlohmann::json::array();
        bool cancelled = false;
        for (const auto& job : jobs) {
            if (job.state != JobState::running && job.state != JobState::pending) continue;

            std::string job_run_dir = job.data.is_object() ? job.data.value("run_dir", std::string()) : std::string();
            std::string job_run_id = job.data.is_object() ? job.data.value("run_id", std::string()) : job.run_id;
            std::string job_runs_dir = job.data.is_object() ? job.data.value("runs_dir", std::string()) : std::string();
            bool matches = job_run_id == run_id;

            if (!matches && has_resolved_run_dir) {
                matches = job_run_dir == resolved_run_dir.string();
            }
            if (!matches && run_id == "pending" && has_resolved_run_dir && !job_runs_dir.empty()) {
                matches = resolved_run_dir.string().rfind(job_runs_dir, 0) == 0;
            }
            if (!matches) continue;

            if (job.pid.has_value()) killed_pids.push_back(*job.pid);
            if (state->subprocess_manager.cancel(job.job_id)) {
                cancelled_jobs.push_back(job.job_id);
                cancelled = true;
            }
        }

        if (!cancelled && run_id == "pending") {
            std::optional<Job> single_running;
            for (const auto& job : jobs) {
                if (job.state != JobState::running) continue;
                if (job.type.rfind("run", 0) != 0) continue;
                if (single_running.has_value()) {
                    single_running.reset();
                    break;
                }
                single_running = job;
            }
            if (single_running.has_value()) {
                if (single_running->pid.has_value()) killed_pids.push_back(*single_running->pid);
                if (state->subprocess_manager.cancel(single_running->job_id)) {
                    cancelled_jobs.push_back(single_running->job_id);
                    cancelled = true;
                }
            }
        }
#ifndef _WIN32
        if (!cancelled) {
            auto orphan_pids = terminate_orphan_runner_processes(state->runtime, run_id, resolved_run_dir.string());
            for (int pid : orphan_pids) killed_pids.push_back(pid);
            if (!orphan_pids.empty()) {
                cancelled = true;
                state->ui_event_store.push("run.stop.orphan", "runs.run_stop", {{"run_dir", resolved_run_dir.string()}, {"killed_pids", orphan_pids}}, run_id);
            }
        }
#endif
        if (cancelled) {
            state->ui_event_store.push("run.stop", "runs.run_stop", {{"run_dir", has_resolved_run_dir ? resolved_run_dir.string() : std::string()}, {"cancelled_jobs", cancelled_jobs}, {"killed_pids", killed_pids}}, run_id);
        }
        return json_resp({{"ok", cancelled}, {"run_id", run_id}, {"cancelled_jobs", cancelled_jobs}, {"killed_pids", killed_pids}});
    });

    CROW_ROUTE(app, "/api/runs/<string>/hme-preview").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        auto body_opt = tile_compile::routes::parse_body(req);
        if (!body_opt) return err_resp("Invalid JSON");
        const auto& body = *body_opt;
        const std::string run_dir_str = body.value("run_dir", std::string());
        fs::path run_dir;
        if (auto err = resolve_request_run_dir(state, run_id, run_dir_str, run_dir)) return std::move(*err);
        const auto preview = tile_compile::web::create_hme_preview(
            run_dir, body.value("params", nlohmann::json::object()));
        if (!preview.ok) return err_resp("HMS_PREVIEW_FAILED", preview.error,
                                         preview.status, nlohmann::json::object());
        crow::response response(200);
        response.set_header("Content-Type", "image/png");
        response.set_header("Cache-Control", "no-store");
        response.set_header("Access-Control-Expose-Headers", "X-HMS-Diagnostics");
        response.set_header("X-HMS-Diagnostics", preview.diagnostics.dump());
        response.body.assign(reinterpret_cast<const char*>(preview.png.data()), preview.png.size());
        return response;
    });

    CROW_ROUTE(app, "/api/runs/<string>/bge-preview").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        auto body_opt = tile_compile::routes::parse_body(req);
        if (!body_opt) return err_resp("Invalid JSON");
        const auto& body = *body_opt;
        fs::path run_dir;
        if (auto err = resolve_request_run_dir(state, run_id,
                body.value("run_dir", std::string()), run_dir)) return std::move(*err);
        const std::string view = body.value("view", std::string("corrected"));
        const auto preview = tile_compile::web::create_bge_preview(
            run_dir, body.value("params", nlohmann::json::object()),
            body.value("exclusion_polygons", nlohmann::json::array()),
            body.value("manual_sample_points", nlohmann::json::array()), view);
        if (!preview.ok) return err_resp("BGE_PREVIEW_FAILED", preview.error,
                                         preview.status, nlohmann::json::object());
        if (view == "diagnostics") return json_resp(preview.diagnostics);
        crow::response response(200);
        response.set_header("Content-Type", "image/png");
        response.set_header("Cache-Control", "no-store");
        response.body.assign(reinterpret_cast<const char*>(preview.png.data()), preview.png.size());
        return response;
    });

    CROW_ROUTE(app, "/api/runs/<string>/resume").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        auto body_opt = tile_compile::routes::parse_body(req);
        if (!body_opt) return err_resp("Invalid JSON");
        auto& body = *body_opt;

        std::string from_phase   = body.value("from_phase",   "");
        std::string run_dir_str  = body.value("run_dir",      "");
        std::string rev_id       = body.value("config_revision_id", "");
        std::string config_yaml  = body.value("config_yaml",  "");
        std::string filter_ctx   = body.value("filter_context", "");
        const bool dry_run       = body.value("dry_run", false);

        if (from_phase.empty()) return err_resp("RESUME_PHASE_REQUIRED", "from_phase is required for resume", 409, nlohmann::json::object());

        fs::path run_dir;
        if (auto err = resolve_request_run_dir(state, run_id, run_dir_str, run_dir)) return std::move(*err);

        const fs::path run_config_path = run_dir / "config.yaml";
        std::string requested_yaml = config_yaml;
        if (requested_yaml.empty() && !rev_id.empty()) {
            auto rev = resolve_config_revision(state, run_dir, rev_id);
            if (!rev) return err_resp("NOT_FOUND", "revision '" + rev_id + "' not found", 404, nlohmann::json::object());
            requested_yaml = rev->yaml_text;
        }

        // Validate that the requested resume phase is feasible given the
        // artifacts that actually exist in the run directory. This must mirror
        // the runner's resume path. AQMH map/reconstruction phases need the
        // prewarp cache; AQMH STACKING can use persisted reconstructed_L.fit.
        {
            static const std::set<std::string> inplace_rerun_phases = {
                "SCAN_INPUT", "CHANNEL_SPLIT", "NORMALIZATION", "GLOBAL_METRICS",
                "TILE_GRID", "REGISTRATION", "PREWARP", "COMMON_OVERLAP",
                "LOCAL_METRICS", "TILE_RECONSTRUCTION", "STATE_CLUSTERING",
                "SYNTHETIC_FRAMES", "DEBAYER"
            };
            static const std::set<std::string> aqmh_cache_resume_phases = {
                "AQMH_MAPS", "AQMH_GLOBAL_QUALITY", "AQMH_METRICS",
                "AQMH_RECONSTRUCTION", "AQMH_DIAGNOSTICS"
            };
            static const std::map<std::string, std::vector<std::string>> phase_required_files = {
                {"DEBAYER",            {"outputs/stacked.fits"}},
                {"ASTROMETRY",         {"outputs/stacked_rgb.fits"}},
                {"BGE",                {"outputs/stacked_rgb_solve.fits"}},
                {"PCC",                {"outputs/stacked_rgb_solve.fits"}},
                {"HYPERMETRIC_STRETCH",{"outputs/pcc_R.fit", "outputs/pcc_G.fit", "outputs/pcc_B.fit"}},
            };
            static const std::set<std::string> supported_resume_phases = {
                "SCAN_INPUT", "REGISTRATION", "PREWARP", "CHANNEL_SPLIT",
                "NORMALIZATION", "GLOBAL_METRICS", "TILE_GRID", "COMMON_OVERLAP",
                "LOCAL_METRICS", "TILE_RECONSTRUCTION", "STATE_CLUSTERING",
                "SYNTHETIC_FRAMES", "AQMH_MAPS", "AQMH_GLOBAL_QUALITY",
                "AQMH_METRICS", "AQMH_RECONSTRUCTION", "AQMH_DIAGNOSTICS",
                "STACKING", "DEBAYER", "ASTROMETRY", "BGE", "PCC",
                "HYPERMETRIC_STRETCH"
            };
            if (!supported_resume_phases.count(from_phase)) {
                return err_resp("RESUME_PHASE_NOT_FEASIBLE",
                    "Cannot resume from phase '" + from_phase + "': this phase is not a supported resume start point.",
                    409, {{"from_phase", from_phase}, {"reason", "unsupported_resume_phase"}});
            }
            const std::string method = read_run_method_local(run_dir, requested_yaml);

            std::error_code ec;
            if (inplace_rerun_phases.count(from_phase)) {
                const fs::path config_path = run_dir / "config.yaml";
                if (!fs::is_regular_file(config_path, ec) || tile_compile::routes::read_file_str(config_path).empty()) {
                    return err_resp("RESUME_PHASE_NOT_FEASIBLE",
                        "Cannot resume from phase '" + from_phase + "': config.yaml is missing or empty, so the runner cannot replay the run in place.",
                        409, {{"from_phase", from_phase}, {"reason", "config_missing"}});
                }
                const std::string input_dir = read_resume_input_dir_from_events(run_dir);
                if (input_dir.empty()) {
                    return err_resp("RESUME_PHASE_NOT_FEASIBLE",
                        "Cannot resume from phase '" + from_phase + "': the original input_dir is missing from logs/run_events.jsonl, so the runner cannot replay the run in place.",
                        409, {{"from_phase", from_phase}, {"reason", "input_dir_missing"}});
                }
            } else if (method == "aqmh" && aqmh_cache_resume_phases.count(from_phase)) {
                bool cache_exists = has_nonempty_prewarped_cache(run_dir);
                if (!cache_exists) {
                    nlohmann::json feasible_phases = nlohmann::json::array(
                        {"SCAN_INPUT", "REGISTRATION", "PREWARP", "STACKING", "DEBAYER",
                         "ASTROMETRY", "BGE", "PCC", "HYPERMETRIC_STRETCH"});
                    return err_resp("RESUME_PHASE_NOT_FEASIBLE",
                        "Cannot resume from phase '" + from_phase + "': runner would resume at '" +
                        "AQMH_RECONSTRUCTION' and requires .prewarped_cache frames, but no reusable "
                        ".prewarped_cache frames are present in the run directory. "
                        "Use an in-place rerun phase such as SCAN_INPUT/REGISTRATION/PREWARP, or resume from a persisted downstream artifact. "
                        "Feasible resume phases for this run: " +
                        [&feasible_phases]() {
                            std::string out;
                            for (const auto& phase : feasible_phases) {
                                if (!out.empty()) out += ", ";
                                out += phase.get<std::string>();
                            }
                            return out;
                        }() + ".",
                        409, {{"from_phase", from_phase}, {"effective_runner_phase", "AQMH_RECONSTRUCTION"},
                              {"reason", "prewarped_cache_missing"},
                              {"cache_dir", (run_dir / ".prewarped_cache").string()},
                              {"feasible_phases", feasible_phases}});
                }
            } else {
                if (from_phase == "STACKING") {
                    if (method == "aqmh") {
                        if (!fs::is_regular_file(run_dir / "outputs" / "reconstructed_L.fit", ec)) {
                            return err_resp("RESUME_PHASE_NOT_FEASIBLE",
                                "Cannot resume from phase 'STACKING': outputs/reconstructed_L.fit is missing.",
                                409, {{"from_phase", from_phase}, {"reason", "artifacts_missing"},
                                      {"missing_files", nlohmann::json::array({"outputs/reconstructed_L.fit"})}});
                        }
                    } else if (!has_synthetic_outputs(run_dir)) {
                        return err_resp("RESUME_PHASE_NOT_FEASIBLE",
                            "Cannot resume from phase 'STACKING': synthetic_*.fit outputs are missing.",
                            409, {{"from_phase", from_phase}, {"reason", "missing_synthetic_outputs"}});
                    }
                }
                auto it = phase_required_files.find(from_phase);
                if (it != phase_required_files.end()) {
                    std::vector<std::string> missing;
                    for (const auto& rel_path : it->second) {
                        if (!fs::is_regular_file(run_dir / rel_path, ec)) {
                            missing.push_back(rel_path);
                        }
                    }
                    if (!missing.empty()) {
                        // Special case: HYPERMETRIC_STRETCH can also work from stacked_rgb_pcc.fits
                        bool fallback_ok = false;
                        if (from_phase == "HYPERMETRIC_STRETCH") {
                            fallback_ok = fs::is_regular_file(run_dir / "outputs" / "stacked_rgb_pcc.fits", ec);
                        }
                        if (!fallback_ok) {
                            nlohmann::json missing_arr = nlohmann::json::array();
                            for (const auto& m : missing) missing_arr.push_back(m);
                            return err_resp("RESUME_PHASE_NOT_FEASIBLE",
                                "Cannot resume from phase '" + from_phase + "': required artifacts missing: " +
                                missing_arr.dump(),
                                409, {{"from_phase", from_phase}, {"reason", "artifacts_missing"},
                                      {"missing_files", missing_arr}});
                        }
                    }
                }
            }
        }

        const size_t event_cursor_before_resume = count_run_event_lines_for_resume(run_dir);

        if (dry_run) {
            return json_resp({
                {"run_id", run_id},
                {"run_dir", run_dir.string()},
                {"from_phase", from_phase},
                {"feasible", true},
                {"dry_run", true},
                {"message", "Resume from phase '" + from_phase + "' is feasible."}
            });
        }

        if (!requested_yaml.empty()) {
            const std::string current_yaml = tile_compile::routes::read_file_str(run_config_path);
            if (!current_yaml.empty() && current_yaml != requested_yaml) {
                persist_run_config_snapshot(run_dir, current_yaml, "resume_previous_config", run_id);
            }
            if (!tile_compile::routes::write_file_str(run_config_path, requested_yaml)) {
                return err_resp("Cannot write: " + run_config_path.string(), 500);
            }
            persist_run_config_snapshot(run_dir, requested_yaml, "resume_selected_config", run_id);
            const std::string active_revision_id =
                state->revision_store.add(run_config_path, requested_yaml, "resume_config", run_id);
            {
                std::lock_guard<std::mutex> lk(state->state_mutex);
                state->active_config_revision_id = active_revision_id;
            }
        }

        std::vector<std::string> args = {state->runtime.runner_exe, "resume"};
        args.push_back("--run-dir"); args.push_back(run_dir.string());
        args.push_back("--from-phase"); args.push_back(from_phase);

        nlohmann::json resume_job_data = {
            {"run_id", run_id},
            {"run_dir", run_dir.string()},
            {"runs_dir", run_dir.parent_path().string()},
            {"from_phase", from_phase},
            {"event_cursor_before_resume", event_cursor_before_resume},
            {"config_revision_id", rev_id},
            {"filter_context", filter_ctx.empty() ? nlohmann::json(nullptr) : nlohmann::json(filter_ctx)},
            {"command", args}
        };
        std::string job_id = state->subprocess_manager.launch("resume", args,
                                                               state->runtime.project_root.string(),
                                                               run_id,
                                                               resume_job_data);
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->current_run_id = run_id;
        }
        state->ui_event_store.push(
            "run.resume",
            "runs.run_resume",
            {{"from_phase", from_phase}, {"config_revision_id", rev_id}, {"filter_context", filter_ctx.empty() ? nlohmann::json(nullptr) : nlohmann::json(filter_ctx)}},
            run_id,
            job_id);
        return json_resp({{"job_id", job_id}, {"run_id", run_id}}, 202);
    });

    CROW_ROUTE(app, "/api/runs/<string>/logs").methods("GET"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        int tail = 250;
        if (req.url_params.get("tail"))
            try { tail = std::stoi(req.url_params.get("tail")); } catch (...) {}
        try {
            std::string run_dir_str = req.url_params.get("run_dir") ? req.url_params.get("run_dir") : "";
            fs::path run_dir;
            if (auto err = resolve_request_run_dir(state, run_id, run_dir_str, run_dir)) return std::move(*err);
            std::string text = read_run_logs(run_dir, tail);
            nlohmann::json lines = nlohmann::json::array();
            std::istringstream iss(text);
            std::string line;
            while (std::getline(iss, line)) lines.push_back(line);
            return json_resp({{"lines", lines}, {"cursor", nullptr}, {"run_dir", run_dir.string()}});
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/artifacts").methods("GET"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        try {
            std::string run_dir_str = req.url_params.get("run_dir") ? req.url_params.get("run_dir") : "";
            fs::path run_dir;
            if (auto err = resolve_request_run_dir(state, run_id, run_dir_str, run_dir)) return std::move(*err);
            auto items   = list_run_artifacts(run_dir);
            return json_resp({{"items", items}, {"run_id", run_id}, {"run_dir", run_dir.string()}});
        } catch (const std::exception& e) {
            if (pending_run_status(state, run_id)) {
                return json_resp({{"items", nlohmann::json::array()}, {"run_id", run_id}});
            }
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/artifacts/view").methods("GET"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        std::string rel_path = req.url_params.get("path") ? req.url_params.get("path") : "";
        if (rel_path.empty()) {
            return err_resp("BAD_REQUEST", "path is required", 400, nlohmann::json::object());
        }
        try {
            std::string run_dir_str = req.url_params.get("run_dir") ? req.url_params.get("run_dir") : "";
            fs::path run_dir;
            if (auto err = resolve_request_run_dir(state, run_id, run_dir_str, run_dir)) return std::move(*err);
            auto full = resolve_artifact_path(run_dir, rel_path);
            if (!full) {
                return err_resp("ARTIFACT_PATH_INVALID", "artifact path must stay inside run directory", 400, nlohmann::json::object());
            }
            if (!fs::exists(*full) || !fs::is_regular_file(*full)) {
                return err_resp("ARTIFACT_NOT_FILE", "artifact path is not a file", 400, nlohmann::json::object());
            }
            std::ifstream f(*full);
            std::string content((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
            std::string ext = full->extension().string();
            if (ext == ".json" || ext == ".jsonl") {
                try {
                    auto j = nlohmann::json::parse(content);
                    return json_resp({{"json", j}, {"is_json", true}, {"text", content}, {"filename", full->filename().string()}, {"path", full->string()}});
                } catch (...) {}
            }
            return json_resp({{"text", content}, {"is_json", false}, {"filename", full->filename().string()}, {"path", full->string()}});
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/image-preview").methods("GET"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        std::string rel_path = req.url_params.get("path") ? req.url_params.get("path") : "";
        if (rel_path.empty()) {
            return err_resp("BAD_REQUEST", "path is required", 400, nlohmann::json::object());
        }
        try {
            std::string run_dir_str = req.url_params.get("run_dir") ? req.url_params.get("run_dir") : "";
            fs::path run_dir;
            if (auto err = resolve_request_run_dir(state, run_id, run_dir_str, run_dir)) return std::move(*err);
            auto full = resolve_artifact_path(run_dir, rel_path);
            if (!full) return err_resp("ARTIFACT_PATH_INVALID", "artifact path must stay inside run directory", 400, nlohmann::json::object());
            if (!fs::exists(*full) || !fs::is_regular_file(*full)) return err_resp("ARTIFACT_NOT_FILE", "artifact path is not a file", 400, nlohmann::json::object());
            const std::string ext = full->extension().string();
            if (ext != ".fits" && ext != ".fit" && ext != ".fts") {
                return err_resp("UNSUPPORTED_PREVIEW_FORMAT", "image-preview currently supports FITS files", 415, nlohmann::json::object());
            }
            const auto png = render_fits_preview_png(*full);
            crow::response res(200);
            res.set_header("Content-Type", "image/png");
            res.set_header("Cache-Control", "no-store");
            res.body.assign(reinterpret_cast<const char*>(png.data()), png.size());
            return res;
        } catch (const std::exception& e) {
            return err_resp("PREVIEW_RENDER_FAILED", e.what(), 422, nlohmann::json::object());
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/artifacts/raw/<path>").methods("GET"_method)
    ([state](const crow::request& req, std::string run_id, std::string rel_path) {
        run_id = decode_run_id_param(run_id);
        try {
            std::string run_dir_str = req.url_params.get("run_dir") ? req.url_params.get("run_dir") : "";
            fs::path run_dir;
            if (auto err = resolve_request_run_dir(state, run_id, run_dir_str, run_dir)) return std::move(*err);
            auto full = resolve_artifact_path(run_dir, rel_path);
            if (!full) return err_resp("ARTIFACT_PATH_INVALID", "artifact path must stay inside run directory", 400, nlohmann::json::object());
            if (!fs::exists(*full) || !fs::is_regular_file(*full)) return err_resp("ARTIFACT_NOT_FILE", "artifact path is not a file", 400, nlohmann::json::object());
            std::ifstream f(*full, std::ios::binary);
            std::string body((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
            crow::response res(200, body);
            std::string ext = full->extension().string();
            if      (ext == ".html") res.set_header("Content-Type", "text/html");
            else if (ext == ".json") res.set_header("Content-Type", "application/json");
            else if (ext == ".png")  res.set_header("Content-Type", "image/png");
            else                     res.set_header("Content-Type", "application/octet-stream");
            return res;
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/delete").methods("POST"_method)
    ([state](const crow::request&, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        try {
            auto run_dir = state->runtime.resolve_run_dir(run_id);
            auto jobs = state->job_store.list(500);
            for (const auto& job : jobs) {
                if (job.state != JobState::running) continue;
                std::string job_run_id = job.data.is_object() ? job.data.value("run_id", std::string()) : std::string();
                std::string job_run_dir = job.data.is_object() ? job.data.value("run_dir", std::string()) : std::string();
                if (job_run_id == run_id || job_run_dir == run_dir.string()) {
                    return err_resp("RUN_ACTIVE", "cannot delete active run", 409, nlohmann::json::object());
                }
            }
            fs::remove_all(run_dir);
            {
                std::lock_guard<std::mutex> lk(state->state_mutex);
                if (state->current_run_id == run_id) state->current_run_id = "";
            }
            state->ui_event_store.push("run.delete", "runs.run_delete", {{"run_dir", run_dir.string()}}, run_id);
            return json_resp({{"ok", true}, {"run_id", run_id}});
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/set-current").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        std::string run_dir_hint;
        if (auto body_opt = tile_compile::routes::parse_body(req))
            run_dir_hint = body_opt->value("run_dir", "");
        // If run_id itself looks like an absolute path, use it as the dir hint
        if (run_dir_hint.empty()) {
            fs::path as_path(run_id);
            if (as_path.is_absolute()) run_dir_hint = run_id;
        }
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->current_run_id  = run_id;
            state->current_run_dir = run_dir_hint;
        }
        state->ui_event_store.push("run.set_current", "runs.run_set_current", {{"run_id", run_id}}, run_id);
        return json_resp({{"ok", true}, {"run_id", run_id}});
    });

    CROW_ROUTE(app, "/api/runs/<string>/stats").methods("POST"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        auto body_opt = tile_compile::routes::parse_body(req);
        std::string run_dir_str = body_opt ? body_opt->value("run_dir", "") : "";

        fs::path run_dir;
        if (auto err = resolve_request_run_dir(state, run_id, run_dir_str, run_dir)) return std::move(*err);

        std::string job_id = tile_compile::routes::spawn_job(state, "stats", run_id,
            nlohmann::json({{"run_id", run_id}, {"run_dir", run_dir.string()}}),
            [run_id, run_dir]() {
                nlohmann::json result = generate_run_report(run_dir);
                result["run_id"] = run_id;
                result["run_dir"] = run_dir.string();
                return result;
            });
        return json_resp({{"job_id", job_id}, {"state", "running"}}, 202);
    });

    CROW_ROUTE(app, "/api/runs/<string>/stats/status").methods("GET"_method)
    ([state](const crow::request& req, std::string run_id) {
        run_id = decode_run_id_param(run_id);
        try {
            std::string run_dir_str = req.url_params.get("run_dir") ? req.url_params.get("run_dir") : "";
            fs::path run_dir;
            if (auto err = resolve_request_run_dir(state, run_id, run_dir_str, run_dir)) return std::move(*err);
            fs::path stats_dir  = run_dir / "artifacts";
            fs::path report_path = stats_dir / "report.html";
            fs::path summary_path = stats_dir / "stats.json";
            std::string state_str = "unknown";
            if (fs::exists(report_path) || fs::exists(summary_path)) state_str = "ok";

            auto jobs = state->job_store.list(200);
            std::string job_id;
            for (auto& j : jobs) {
                if (j.type == "stats" && j.run_id == run_id) {
                    state_str = job_state_str(j.state);
                    job_id = j.job_id;
                    break;
                }
            }
            return json_resp({
                {"state",       state_str},
                {"output_dir",  stats_dir.string()},
                {"report_path", fs::exists(report_path) ? report_path.string() : ""},
                {"summary_path", fs::exists(summary_path) ? summary_path.string() : ""},
                {"job_id",      job_id},
            });
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
    });

    CROW_ROUTE(app, "/api/runs/<string>/config-revisions/<string>/restore").methods("POST"_method)
    ([state](const crow::request&, std::string run_id, std::string rev_id) {
        run_id = decode_run_id_param(run_id);
        fs::path run_dir;
        try {
            run_dir = state->runtime.resolve_run_dir(run_id);
        } catch (const std::exception& e) {
            return err_resp(e.what(), 404);
        }
        auto rev = resolve_config_revision(state, run_dir, rev_id);
        if (!rev) return err_resp("Revision not found: " + rev_id, 404);
        fs::path target = run_dir / "config.yaml";
        if (!state->runtime.is_path_allowed(target)) {
            return err_resp("PATH_NOT_ALLOWED", "Path not allowed: " + target.string(), 403, {{"path", target.string()}});
        }
        const std::string current_yaml = tile_compile::routes::read_file_str(target);
        if (!current_yaml.empty() && current_yaml != rev->yaml_text) {
            persist_run_config_snapshot(run_dir, current_yaml, "restore_previous_config", run_id);
        }
        if (!tile_compile::routes::write_file_str(target, rev->yaml_text)) {
            return err_resp("BACKEND_COMMAND_FAILED", "failed to restore revision", 502, {{"path", target.string()}});
        }
        persist_run_config_snapshot(run_dir, rev->yaml_text, "restore_selected_config", run_id);
        {
            std::lock_guard<std::mutex> lk(state->state_mutex);
            state->active_config_revision_id = rev_id;
        }
        state->ui_event_store.push("run.revision.restore", "runs.run_revision_restore", {{"revision_id", rev_id}, {"path", target.string()}}, run_id);
        return json_resp({{"ok", true}, {"run_id", run_id}, {"active_revision_id", rev_id}});
    });
}
