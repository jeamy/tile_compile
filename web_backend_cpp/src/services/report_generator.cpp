#include "services/report_generator.hpp"
#include "backend_runtime.hpp"
#include "services/run_inspector.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

struct BasicStats {
    int n = 0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double std_dev = 0.0;
    double p01 = 0.0;
    double p99 = 0.0;
};

struct ChartBlock {
    std::string svg;
    std::string explanation_html;
};

struct ReportSection {
    std::string title;
    std::string cards_html;
};

struct ColorStop {
    double pos;
    const char* hex;
};

struct TileSeries {
    std::string title;
    std::vector<double> values;
    std::string cmap;
    std::string label;
};

/// @brief Implements html escape.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string html_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

/// @brief Trims trailing zeros.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string trim_trailing_zeros(std::string s) {
    auto pos = s.find('.');
    if (pos == std::string::npos) return s;
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    if (s == "-0") return "0";
    return s;
}

/// @brief Formats number.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string format_number(double v, int prec = 3) {
    if (!std::isfinite(v)) return "n/a";
    std::ostringstream ss;
    const double av = std::fabs(v);
    if (av >= 10000.0 || (av > 0.0 && av < 0.001)) {
        ss << std::scientific << std::setprecision(2) << v;
        return ss.str();
    }
    ss << std::fixed << std::setprecision(prec) << v;
    return trim_trailing_zeros(ss.str());
}

/// @brief Implements sanitize label.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string sanitize_label(std::string s) {
    std::replace(s.begin(), s.end(), '\n', ' ');
    return s;
}

/// @brief Normalizes channel label.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string normalize_channel_label(std::string s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c)) || c == '-' || c == '_') continue;
        out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    return out;
}

/// @brief Implements preferred channel rank.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
int preferred_channel_rank(const std::string& label) {
    const std::string normalized = normalize_channel_label(label);
    if (normalized == "R" || normalized == "RED") return 0;
    if (normalized == "G" || normalized == "GREEN") return 1;
    if (normalized == "B" || normalized == "BLUE") return 2;
    return 100;
}

/// @brief Implements preferred channel color.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string preferred_channel_color(const std::string& label) {
    const std::string normalized = normalize_channel_label(label);
    if (normalized == "R" || normalized == "RED") return "#ef4444";
    if (normalized == "G" || normalized == "GREEN") return "#22c55e";
    if (normalized == "B" || normalized == "BLUE") return "#3b82f6";
    return "";
}

/// @brief Reads text.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string read_text(const fs::path& path) {
    const BackendGuardLimits limits = backend_guard_limits_from_env();
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    std::string out;
    out.resize(limits.report_text_bytes);
    in.read(out.data(), static_cast<std::streamsize>(out.size()));
    out.resize(static_cast<size_t>(in.gcount()));
    if (in.peek() != EOF) out += "\n...[truncated]";
    return out;
}

/// @brief Implements env or.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string env_or(const char* key, const std::string& fallback = "") {
    if (!key || !*key) return fallback;
    const char* value = std::getenv(key);
    if (!value || !*value) return fallback;
    return std::string(value);
}

/// @brief Normalizes report locale.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string normalize_report_locale(std::string locale) {
    std::transform(locale.begin(), locale.end(), locale.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (locale == "en" || locale.rfind("en_", 0) == 0 || locale.rfind("en-", 0) == 0) return "en";
    return "de";
}

/// @brief Implements report i18n path.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
fs::path report_i18n_path(const std::string& locale) {
    const std::string ui_dir = env_or("TILE_COMPILE_UI_DIR", "");
    if (!ui_dir.empty()) return fs::path(ui_dir) / "i18n" / ("report_" + locale + ".json");
    const std::string project_root = env_or("TILE_COMPILE_PROJECT_ROOT", "");
    if (!project_root.empty()) return fs::path(project_root) / "web_frontend_v3" / "i18n" / ("report_" + locale + ".json");
    return fs::path("web_frontend_v3") / "i18n" / ("report_" + locale + ".json");
}

json read_json_if_exists(const fs::path& path);

/// @brief Loads report translations.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
json load_report_translations(const std::string& locale) {
    const json parsed = read_json_if_exists(report_i18n_path(locale));
    if (parsed.is_object() && parsed.contains("translations") && parsed["translations"].is_object()) {
        return parsed["translations"];
    }
    if (parsed.is_object()) return parsed;
    return json::object();
}

/// Single-pass multi-replacement: builds the output string in one scan.
/// Pairs must be sorted longest-key-first to avoid partial matches.
std::string apply_replacements(const std::string& text,
                               const std::vector<std::pair<std::string, std::string>>& pairs) {
    std::array<std::vector<size_t>, 256> buckets;
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (pairs[i].first.empty()) continue;
        buckets[static_cast<unsigned char>(pairs[i].first.front())].push_back(i);
    }

    std::string out;
    out.reserve(text.size());
    size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        const auto& candidates = buckets[static_cast<unsigned char>(text[pos])];
        for (const size_t pair_idx : candidates) {
            const auto& [needle, replacement] = pairs[pair_idx];
            if (needle.empty()) continue;
            if (text.compare(pos, needle.size(), needle) == 0) {
                out.append(replacement);
                pos += needle.size();
                matched = true;
                break;
            }
        }
        if (!matched) out.push_back(text[pos++]);
    }
    return out;
}

/// @brief Applies report translations.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string apply_report_translations(std::string html, const std::string& locale) {
    const json translations = load_report_translations(locale);
    if (!translations.is_object() || translations.empty()) return html;

    std::vector<std::pair<std::string, std::string>> pairs;
    pairs.reserve(translations.size() + 1);
    for (auto it = translations.begin(); it != translations.end(); ++it) {
        if (!it.value().is_string()) continue;
        pairs.emplace_back(it.key(), it.value().get<std::string>());
    }
    pairs.emplace_back("<html lang=\"en\"", "<html lang=\"" + locale + "\"");
    // Sort longest key first to avoid partial-match shadowing.
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        return a.first.size() > b.first.size();
    });
    return apply_replacements(html, pairs);
}

std::string escape_script_json(std::string text) {
    size_t pos = 0;
    while ((pos = text.find("</", pos)) != std::string::npos) {
        text.replace(pos, 2, "<\\/");
        pos += 3;
    }
    return text;
}

std::string extract_between(const std::string& text, const std::string& begin, const std::string& end) {
    const auto begin_pos = text.find(begin);
    if (begin_pos == std::string::npos) return {};
    const auto content_pos = begin_pos + begin.size();
    const auto end_pos = text.find(end, content_pos);
    if (end_pos == std::string::npos) return {};
    return text.substr(content_pos, end_pos - content_pos);
}

std::string build_language_switch_script(const std::string& locale) {
    std::ostringstream js;
    js << "<script>(function(){";
    js << "let current='" << html_escape(locale) << "';";
    js << "function setActive(){document.querySelectorAll('[data-report-lang]').forEach(function(btn){btn.classList.toggle('active',btn.getAttribute('data-report-lang')===current);});}";
    js << "document.querySelectorAll('[data-report-lang]').forEach(function(btn){btn.disabled=btn.getAttribute('data-report-lang')!==current;});";
    js << "setActive();";
    js << "})();</script>";
    return js.str();
}

/// @brief Reads json if exists.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
json read_json_if_exists(const fs::path& path) {
    const BackendGuardLimits limits = backend_guard_limits_from_env();
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (!ec && size > limits.report_json_file_bytes) {
        return json{
            {"_truncated", true},
            {"_reason", "file_too_large"},
            {"_size_bytes", static_cast<unsigned long long>(size)}
        };
    }
    std::ifstream in(path);
    if (!in) return json::object();
    auto parsed = json::parse(in, nullptr, false);
    if (parsed.is_discarded()) return json::object();
    return parsed;
}

/// @brief Reads jsonl if exists.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::vector<json> read_jsonl_if_exists(const fs::path& path, int max_lines = 100000) {
    const BackendGuardLimits limits = backend_guard_limits_from_env();
    std::ifstream in(path);
    std::vector<json> items;
    if (!in) return items;
    std::deque<json> log_tail;
    std::map<std::string, int> phase_progress_buckets;
    std::map<std::string, int> queue_progress_buckets;
    const auto raw_string = [](const json& obj, const char* key) {
        return obj.contains(key) && obj[key].is_string() ? obj[key].get<std::string>() : std::string();
    };
    const auto raw_number = [](const json& obj, const char* key) {
        return obj.contains(key) && obj[key].is_number() ? obj[key].get<double>() : 0.0;
    };
    const auto raw_percent = [&](const json& obj) {
        double value = raw_number(obj, "progress");
        if (value == 0.0) value = raw_number(obj, "pct");
        return value >= 0.0 && value <= 1.0 ? value * 100.0 : value;
    };
    std::string line;
    int n = 0;
    while (std::getline(in, line) && n < max_lines) {
        if (line.empty()) continue;
        auto j = json::parse(line, nullptr, false);
        if (j.is_discarded() || !j.is_object()) {
            ++n;
            continue;
        }

        const std::string type = raw_string(j, "type");
        if (type == "log_line") {
            log_tail.push_back(std::move(j));
            if (log_tail.size() > limits.report_log_tail) log_tail.pop_front();
            ++n;
            continue;
        }

        if (type == "phase_progress") {
            std::string phase = raw_string(j, "phase");
            if (phase.empty()) phase = raw_string(j, "phase_name");
            int bucket = static_cast<int>(raw_percent(j) / 5.0);
            auto it = phase_progress_buckets.find(phase);
            if (it != phase_progress_buckets.end() && it->second == bucket) {
                ++n;
                continue;
            }
            phase_progress_buckets[phase] = bucket;
        } else if (type == "queue_progress") {
            std::string filter = raw_string(j, "filter");
            int bucket = static_cast<int>(raw_percent(j) / 5.0);
            auto it = queue_progress_buckets.find(filter);
            if (it != queue_progress_buckets.end() && it->second == bucket) {
                ++n;
                continue;
            }
            queue_progress_buckets[filter] = bucket;
        }

        if (items.size() < limits.report_events_max) items.push_back(std::move(j));
        ++n;
    }
    for (const auto& item : log_tail) {
        if (items.size() >= limits.report_events_max) break;
        items.push_back(item);
    }
    return items;
}

/// @brief Implements json string or.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string json_string_or(const json& obj, const char* key, const std::string& fallback = "") {
    if (!obj.is_object() || !obj.contains(key) || obj.at(key).is_null()) return fallback;
    const auto& value = obj.at(key);
    try {
        if (value.is_string()) return value.get<std::string>();
        if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
        if (value.is_number_integer()) return std::to_string(value.get<long long>());
        if (value.is_number_unsigned()) return std::to_string(value.get<unsigned long long>());
        if (value.is_number_float()) return format_number(value.get<double>());
    } catch (...) {}
    return fallback;
}

/// @brief Implements json number or.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
double json_number_or(const json& obj, const char* key, double fallback = 0.0) {
    if (!obj.is_object() || !obj.contains(key) || obj.at(key).is_null()) return fallback;
    const auto& value = obj.at(key);
    try {
        if (value.is_number()) return value.get<double>();
        if (value.is_string()) return std::stod(value.get<std::string>());
    } catch (...) {}
    return fallback;
}

/// @brief Implements json bool or.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
bool json_bool_or(const json& obj, const char* key, bool fallback = false) {
    if (!obj.is_object() || !obj.contains(key) || obj.at(key).is_null()) return fallback;
    const auto& value = obj.at(key);
    try {
        if (value.is_boolean()) return value.get<bool>();
        if (value.is_number_integer()) return value.get<long long>() != 0;
        if (value.is_string()) {
            const auto s = value.get<std::string>();
            return s == "1" || s == "true" || s == "TRUE" || s == "yes";
        }
    } catch (...) {}
    return fallback;
}

/// Returns nullopt when key is absent or null, otherwise the bool value.
std::optional<bool> json_optional_bool(const json& obj, const char* key) {
    if (!obj.is_object() || !obj.contains(key) || obj.at(key).is_null()) return std::nullopt;
    const auto& value = obj.at(key);
    try {
        if (value.is_boolean()) return value.get<bool>();
        if (value.is_number_integer()) return value.get<long long>() != 0;
    } catch (...) {}
    return std::nullopt;
}

/// @brief Implements json double array.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::vector<double> json_double_array(const json& arr) {
    std::vector<double> out;
    if (!arr.is_array()) return out;
    out.reserve(arr.size());
    for (const auto& item : arr) {
        try {
            if (item.is_number()) out.push_back(item.get<double>());
            else if (item.is_string()) out.push_back(std::stod(item.get<std::string>()));
        } catch (...) {}
    }
    return out;
}

/// @brief Implements percent value.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
double percent_value(double raw) {
    if (raw >= 0.0 && raw <= 1.0) return raw * 100.0;
    return raw;
}

/// @brief Implements clamp01.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

/// @brief Implements percentile sorted.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
double percentile_sorted(const std::vector<double>& vals, double q) {
    if (vals.empty()) return 0.0;
    if (vals.size() == 1) return vals.front();
    q = std::clamp(q, 0.0, 1.0);
    const double pos = q * static_cast<double>(vals.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(pos));
    const auto hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return vals[lo];
    const double t = pos - static_cast<double>(lo);
    return vals[lo] * (1.0 - t) + vals[hi] * t;
}

/// @brief Sorts values and returns the requested percentile.
/// @details Convenience wrapper for callers that do not already hold a sorted vector.
/// Internally forwards to percentile_sorted after sorting.
double percentile_of(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    return percentile_sorted(values, q);
}

/// @brief Sorts values and returns the median.
/// @details Convenience wrapper equivalent to percentile_of(values, 0.5).
double median_of(std::vector<double> values) {
    return percentile_of(std::move(values), 0.5);
}

/// @brief Implements basic stats.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
BasicStats basic_stats(std::vector<double> vals) {
    vals.erase(std::remove_if(vals.begin(), vals.end(),
                              [](double v) { return !std::isfinite(v); }),
               vals.end());
    BasicStats s;
    s.n = static_cast<int>(vals.size());
    if (vals.empty()) return s;
    std::sort(vals.begin(), vals.end());
    s.min = vals.front();
    s.max = vals.back();
    s.median = percentile_sorted(vals, 0.5);
    s.p01 = percentile_sorted(vals, 0.01);
    s.p99 = percentile_sorted(vals, 0.99);
    s.mean = std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<double>(vals.size());
    double var = 0.0;
    for (double v : vals) {
        const double d = v - s.mean;
        var += d * d;
    }
    s.std_dev = vals.size() > 1 ? std::sqrt(var / static_cast<double>(vals.size())) : 0.0;
    return s;
}

/// @brief Implements plot bounds.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::pair<double, double> plot_bounds(const std::vector<double>& vals, bool force_unit_range = false) {
    auto s = basic_stats(vals);
    if (s.n == 0) return {0.0, 1.0};
    double lo = s.min;
    double hi = s.max;
    if (force_unit_range && s.min >= 0.0 && s.max <= 1.0) {
        lo = 0.0;
        hi = 1.0;
    } else if (s.n >= 20 && s.p99 > s.p01) {
        lo = s.p01;
        hi = s.p99;
    }
    if (!(hi > lo)) {
        const double pad = std::fabs(lo) > 1e-9 ? std::fabs(lo) * 0.1 : 1.0;
        lo -= pad;
        hi += pad;
    } else {
        const double pad = (hi - lo) * 0.05;
        lo -= pad;
        hi += pad;
    }
    return {lo, hi};
}

/// @brief Implements scale linear.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
double scale_linear(double v, double in_min, double in_max, double out_min, double out_max) {
    if (!(in_max > in_min)) return (out_min + out_max) * 0.5;
    const double t = (v - in_min) / (in_max - in_min);
    return out_min + t * (out_max - out_min);
}

/// @brief Parses iso utc seconds.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<double> parse_iso_utc_seconds(const std::string& raw) {
    if (raw.size() < 19) return std::nullopt;
    int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;
    if (std::sscanf(raw.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d",
                    &year, &month, &day, &hour, &minute, &second) != 6) {
        return std::nullopt;
    }
    double fractional = 0.0;
    auto dot = raw.find('.');
    if (dot != std::string::npos) {
        size_t end = raw.find_first_of("Z+-", dot);
        const auto frac = raw.substr(dot + 1, end == std::string::npos ? std::string::npos : end - dot - 1);
        if (!frac.empty()) {
            try {
                fractional = std::stod("0." + frac);
            } catch (...) {
                fractional = 0.0;
            }
        }
    }
    std::tm tm{};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_sec = second;
#ifdef _WIN32
    const auto epoch = _mkgmtime(&tm);
#else
    const auto epoch = timegm(&tm);
#endif
    if (epoch < 0) return std::nullopt;
    return static_cast<double>(epoch) + fractional;
}

/// @brief Implements phase name from event.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string phase_name_from_event(const json& ev) {
    if (ev.contains("phase_name") && ev["phase_name"].is_string()) return ev["phase_name"].get<std::string>();
    if (ev.contains("phase")) {
        if (ev["phase"].is_string()) return ev["phase"].get<std::string>();
        if (ev["phase"].is_number_integer()) return std::to_string(ev["phase"].get<int>());
    }
    return "";
}

/// @brief Returns a stable match key for phase start/end pairing.
/// @details Uses the integer phase number when available so that display-name
/// mismatches (e.g. AQMH_QUALITY_MAPS vs LOCAL_METRICS for Phase 8) do not
/// prevent correct duration computation. Falls back to phase_name string.
std::string phase_match_key(const json& ev) {
    if (ev.contains("phase") && ev["phase"].is_number_integer()) {
        return "#" + std::to_string(ev["phase"].get<int>());
    }
    return phase_name_from_event(ev);
}

/// @brief Formats event line.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string format_event_line(const json& ev) {
    std::vector<std::string> parts;
    const std::string ts = json_string_or(ev, "ts", json_string_or(ev, "timestamp", ""));
    const std::string type = json_string_or(ev, "type", "event");
    const std::string phase = phase_name_from_event(ev);
    const std::string status = json_string_or(ev, "status", "");
    const std::string message = json_string_or(ev, "message", "");
    if (!ts.empty()) parts.push_back(ts);
    parts.push_back(type);
    if (!phase.empty()) parts.push_back(phase);
    if (!status.empty()) parts.push_back("status=" + status);
    if (ev.contains("progress")) {
        parts.push_back(format_number(percent_value(json_number_or(ev, "progress", 0.0)), 1) + "%");
    }
    if (!message.empty()) parts.push_back(message);
    std::ostringstream out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out << " | ";
        out << parts[i];
    }
    return out.str();
}

/// @brief Implements rgb from hex.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::array<int, 3> rgb_from_hex(const std::string& hex) {
    if (hex.size() != 7 || hex[0] != '#') return {122, 162, 247};
    auto from_pair = [&](size_t pos) {
        return static_cast<int>(std::strtol(hex.substr(pos, 2).c_str(), nullptr, 16));
    };
    return {from_pair(1), from_pair(3), from_pair(5)};
}

/// @brief Implements rgb hex.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string rgb_hex(const std::array<int, 3>& rgb) {
    std::ostringstream ss;
    ss << '#'
       << std::hex << std::setw(2) << std::setfill('0') << std::clamp(rgb[0], 0, 255)
       << std::setw(2) << std::setfill('0') << std::clamp(rgb[1], 0, 255)
       << std::setw(2) << std::setfill('0') << std::clamp(rgb[2], 0, 255);
    return ss.str();
}

/// @brief Implements interpolate color.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string interpolate_color(const std::vector<ColorStop>& stops, double t) {
    if (stops.empty()) return "#7aa2f7";
    t = clamp01(t);
    if (t <= stops.front().pos) return stops.front().hex;
    if (t >= stops.back().pos) return stops.back().hex;
    for (size_t i = 1; i < stops.size(); ++i) {
        if (t > stops[i].pos) continue;
        const auto& a = stops[i - 1];
        const auto& b = stops[i];
        const double span = b.pos - a.pos;
        const double u = span > 0.0 ? (t - a.pos) / span : 0.0;
        const auto ca = rgb_from_hex(a.hex);
        const auto cb = rgb_from_hex(b.hex);
        std::array<int, 3> mixed{};
        for (int k = 0; k < 3; ++k) {
            mixed[k] = static_cast<int>(std::round(ca[k] * (1.0 - u) + cb[k] * u));
        }
        return rgb_hex(mixed);
    }
    return stops.back().hex;
}

/// @brief Implements colormap hex.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string colormap_hex(const std::string& name, double t) {
    static const std::vector<ColorStop> viridis = {
        {0.00, "#440154"}, {0.25, "#3b528b"}, {0.50, "#21918c"}, {0.75, "#5ec962"}, {1.00, "#fde725"}
    };
    static const std::vector<ColorStop> plasma = {
        {0.00, "#0d0887"}, {0.25, "#7e03a8"}, {0.50, "#cc4778"}, {0.75, "#f89441"}, {1.00, "#f0f921"}
    };
    static const std::vector<ColorStop> inferno = {
        {0.00, "#000004"}, {0.25, "#57106e"}, {0.50, "#bc3754"}, {0.75, "#f98e09"}, {1.00, "#fcffa4"}
    };
    static const std::vector<ColorStop> magma = {
        {0.00, "#000004"}, {0.25, "#51127c"}, {0.50, "#b5367a"}, {0.75, "#fb8861"}, {1.00, "#fcfdbf"}
    };
    static const std::vector<ColorStop> cividis = {
        {0.00, "#00204c"}, {0.25, "#434e6c"}, {0.50, "#7c7b78"}, {0.75, "#b7a86d"}, {1.00, "#fee838"}
    };
    static const std::vector<ColorStop> ylgn = {
        {0.00, "#ffffe5"}, {0.25, "#d9f0a3"}, {0.50, "#addd8e"}, {0.75, "#78c679"}, {1.00, "#238443"}
    };
    static const std::vector<ColorStop> ylgnbu = {
        {0.00, "#ffffd9"}, {0.25, "#c7e9b4"}, {0.50, "#7fcdbb"}, {0.75, "#41b6c4"}, {1.00, "#225ea8"}
    };
    static const std::vector<ColorStop> gray = {
        {0.00, "#111827"}, {1.00, "#f8fafc"}
    };
    const auto key = name;
    if (key == "plasma") return interpolate_color(plasma, t);
    if (key == "inferno") return interpolate_color(inferno, t);
    if (key == "magma") return interpolate_color(magma, t);
    if (key == "cividis") return interpolate_color(cividis, t);
    if (key == "YlGn") return interpolate_color(ylgn, t);
    if (key == "YlGnBu") return interpolate_color(ylgnbu, t);
    if (key == "gray") return interpolate_color(gray, t);
    return interpolate_color(viridis, t);
}

/// @brief Implements svg begin.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string svg_begin(int width, int height, const std::string& title) {
    std::ostringstream out;
    out << "<svg class=\"report-chart\" viewBox=\"0 0 " << width << ' ' << height
        << "\" xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" aria-label=\""
        << html_escape(title) << "\">";
    out << "<title>" << html_escape(title) << "</title>";
    out << "<rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\"" << height
        << "\" rx=\"14\" fill=\"#020617\" stroke=\"#1e293b\"/>";
    return out.str();
}

/// @brief Implements svg message.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string svg_message(const std::string& title, const std::string& message, int width = 720, int height = 220) {
    std::ostringstream out;
    out << svg_begin(width, height, title);
    out << "<text x=\"24\" y=\"42\" class=\"svg-title\">" << html_escape(title) << "</text>";
    out << "<text x=\"24\" y=\"110\" class=\"svg-note\">" << html_escape(message) << "</text>";
    out << "</svg>";
    return out.str();
}

void append_y_grid(std::ostringstream& out,
                   double x0,
                   double y0,
                   double width,
                   double height,
                   double min_v,
                   double max_v,
                   int ticks) {
    for (int i = 0; i <= ticks; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(ticks);
        const double y = y0 + height - t * height;
        const double v = min_v + t * (max_v - min_v);
        out << "<line x1=\"" << x0 << "\" y1=\"" << y << "\" x2=\"" << (x0 + width)
            << "\" y2=\"" << y << "\" class=\"svg-grid\"/>";
        out << "<text x=\"" << (x0 - 10) << "\" y=\"" << (y + 4)
            << "\" class=\"svg-tick\" text-anchor=\"end\">" << html_escape(format_number(v, 2))
            << "</text>";
    }
}

void append_x_ticks(std::ostringstream& out,
                    double x0,
                    double y0,
                    double width,
                    int max_index,
                    int ticks) {
    if (max_index <= 0) return;
    for (int i = 0; i <= ticks; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(ticks);
        const double x = x0 + t * width;
        const int idx = static_cast<int>(std::round(t * static_cast<double>(max_index)));
        out << "<line x1=\"" << x << "\" y1=\"" << y0 << "\" x2=\"" << x
            << "\" y2=\"" << (y0 + 6) << "\" class=\"svg-axis\"/>";
        out << "<text x=\"" << x << "\" y=\"" << (y0 + 22)
            << "\" class=\"svg-tick\" text-anchor=\"middle\">" << idx << "</text>";
    }
}

std::string svg_timeseries(const std::vector<double>& raw_values,
                           const std::string& title,
                           const std::string& ylabel,
                           const std::string& color = "#7aa2f7",
                           bool median_line = true,
                           int width = 720,
                           int height = 300) {
    std::vector<std::pair<int, double>> values;
    values.reserve(raw_values.size());
    for (size_t i = 0; i < raw_values.size(); ++i) {
        if (std::isfinite(raw_values[i])) values.push_back({static_cast<int>(i), raw_values[i]});
    }
    if (values.empty()) return svg_message(title, "No data", width, height);

    std::vector<double> ys;
    ys.reserve(values.size());
    for (const auto& item : values) ys.push_back(item.second);
    const auto bounds = plot_bounds(ys, false);
    const auto stats = basic_stats(ys);

    const double x0 = 58.0;
    const double y0 = 34.0;
    const double pw = width - 84.0;
    const double ph = height - 74.0;
    const int max_index = static_cast<int>(raw_values.size() > 1 ? raw_values.size() - 1 : 1);

    std::ostringstream out;
    out << svg_begin(width, height, title);
    out << "<text x=\"24\" y=\"26\" class=\"svg-title\">" << html_escape(title) << "</text>";
    out << "<text x=\"24\" y=\"" << (y0 + ph * 0.5) << "\" class=\"svg-label\" transform=\"rotate(-90 24 "
        << (y0 + ph * 0.5) << ")\">" << html_escape(ylabel) << "</text>";
    append_y_grid(out, x0, y0, pw, ph, bounds.first, bounds.second, 4);
    append_x_ticks(out, x0, y0 + ph, pw, max_index, 5);
    out << "<line x1=\"" << x0 << "\" y1=\"" << (y0 + ph) << "\" x2=\"" << (x0 + pw)
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    out << "<line x1=\"" << x0 << "\" y1=\"" << y0 << "\" x2=\"" << x0
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    out << "<text x=\"" << (x0 + pw * 0.5) << "\" y=\"" << (height - 8)
        << "\" class=\"svg-label\" text-anchor=\"middle\">frame index</text>";

    if (median_line && stats.n > 0) {
        const double ym = scale_linear(stats.median, bounds.first, bounds.second, y0 + ph, y0);
        out << "<line x1=\"" << x0 << "\" y1=\"" << ym << "\" x2=\"" << (x0 + pw)
            << "\" y2=\"" << ym << "\" stroke=\"#f87171\" stroke-dasharray=\"6 4\" stroke-width=\"0.9\" opacity=\"0.9\" vector-effect=\"non-scaling-stroke\"/>";
    }

    std::ostringstream poly;
    for (const auto& [idx, val] : values) {
        const double x = scale_linear(static_cast<double>(idx), 0.0, static_cast<double>(max_index), x0, x0 + pw);
        const double y = scale_linear(val, bounds.first, bounds.second, y0 + ph, y0);
        poly << x << ',' << y << ' ';
    }
    out << "<polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"0.85\" vector-effect=\"non-scaling-stroke\" points=\""
        << poly.str() << "\"/>";
    if (values.size() <= 80) {
        for (const auto& [idx, val] : values) {
            const double x = scale_linear(static_cast<double>(idx), 0.0, static_cast<double>(max_index), x0, x0 + pw);
            const double y = scale_linear(val, bounds.first, bounds.second, y0 + ph, y0);
            out << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"1.4\" fill=\"" << color << "\"/>";
        }
    }
    out << "</svg>";
    return out.str();
}

std::string svg_multi_timeseries(const std::map<std::string, std::vector<double>>& series,
                                 const std::string& title,
                                 const std::string& ylabel,
                                 int width = 720,
                                 int height = 320) {
    std::vector<double> all_values;
    size_t max_len = 0;
    for (const auto& [_, vals] : series) {
        max_len = std::max(max_len, vals.size());
        for (double v : vals) if (std::isfinite(v)) all_values.push_back(v);
    }
    if (all_values.empty() || max_len == 0) return svg_message(title, "No data", width, height);

    static const std::vector<std::string> palette = {
        "#f87171", "#4ade80", "#60a5fa", "#fbbf24", "#c084fc", "#22d3ee", "#fb7185", "#a3e635"
    };
    const auto bounds = plot_bounds(all_values, false);
    const double x0 = 58.0;
    const double y0 = 42.0;
    const double pw = width - 84.0;
    const double ph = height - 88.0;
    const int max_index = static_cast<int>(max_len > 1 ? max_len - 1 : 1);

    std::ostringstream out;
    out << svg_begin(width, height, title);
    out << "<text x=\"24\" y=\"26\" class=\"svg-title\">" << html_escape(title) << "</text>";
    out << "<text x=\"24\" y=\"" << (y0 + ph * 0.5) << "\" class=\"svg-label\" transform=\"rotate(-90 24 "
        << (y0 + ph * 0.5) << ")\">" << html_escape(ylabel) << "</text>";
    append_y_grid(out, x0, y0, pw, ph, bounds.first, bounds.second, 4);
    append_x_ticks(out, x0, y0 + ph, pw, max_index, 5);
    out << "<line x1=\"" << x0 << "\" y1=\"" << (y0 + ph) << "\" x2=\"" << (x0 + pw)
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    out << "<line x1=\"" << x0 << "\" y1=\"" << y0 << "\" x2=\"" << x0
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    out << "<text x=\"" << (x0 + pw * 0.5) << "\" y=\"" << (height - 8)
        << "\" class=\"svg-label\" text-anchor=\"middle\">frame index</text>";

    std::vector<std::pair<std::string, const std::vector<double>*>> ordered_series;
    ordered_series.reserve(series.size());
    for (const auto& [name, vals] : series) ordered_series.push_back({name, &vals});
    std::stable_sort(ordered_series.begin(), ordered_series.end(),
                     [](const auto& a, const auto& b) {
                         return preferred_channel_rank(a.first) < preferred_channel_rank(b.first);
                     });

    size_t color_index = 0;
    double legend_x = x0 + 6.0;
    const double legend_y = 32.0;
    for (const auto& [name, vals_ptr] : ordered_series) {
        const auto& vals = *vals_ptr;
        std::ostringstream poly;
        bool has_any = false;
        for (size_t i = 0; i < vals.size(); ++i) {
            if (!std::isfinite(vals[i])) continue;
            has_any = true;
            const double x = scale_linear(static_cast<double>(i), 0.0, static_cast<double>(max_index), x0, x0 + pw);
            const double y = scale_linear(vals[i], bounds.first, bounds.second, y0 + ph, y0);
            poly << x << ',' << y << ' ';
        }
        if (!has_any) continue;
        const std::string color = [&]() {
            const std::string preferred = preferred_channel_color(name);
            if (!preferred.empty()) return preferred;
            return palette[color_index % palette.size()];
        }();
        out << "<polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"0.85\" vector-effect=\"non-scaling-stroke\" points=\""
            << poly.str() << "\"/>";
        out << "<line x1=\"" << legend_x << "\" y1=\"" << legend_y << "\" x2=\"" << (legend_x + 16)
            << "\" y2=\"" << legend_y << "\" stroke=\"" << color << "\" stroke-width=\"1.2\" vector-effect=\"non-scaling-stroke\"/>";
        out << "<text x=\"" << (legend_x + 22) << "\" y=\"" << (legend_y + 4)
            << "\" class=\"svg-tick\">" << html_escape(name) << "</text>";
        legend_x += 92.0;
        ++color_index;
    }
    out << "</svg>";
    return out.str();
}

std::string svg_histogram(const std::vector<double>& raw_values,
                          const std::string& title,
                          const std::string& xlabel,
                          const std::string& color = "#7aa2f7",
                          int bins = 50,
                          int width = 640,
                          int height = 300) {
    std::vector<double> values;
    values.reserve(raw_values.size());
    for (double v : raw_values) if (std::isfinite(v)) values.push_back(v);
    if (values.size() < 2) return svg_message(title, "Not enough data", width, height);

    const auto stats = basic_stats(values);
    double lo = stats.n >= 20 ? stats.p01 : stats.min;
    double hi = stats.n >= 20 ? stats.p99 : stats.max;
    if (!(hi > lo)) {
        lo = stats.min - 0.5;
        hi = stats.max + 0.5;
    }
    bins = std::clamp(bins, 5, 80);
    std::vector<int> counts(static_cast<size_t>(bins), 0);
    for (double v : values) {
        if (v < lo || v > hi) continue;
        double t = (v - lo) / (hi - lo);
        if (t >= 1.0) t = 0.999999;
        const int idx = static_cast<int>(t * bins);
        counts[static_cast<size_t>(std::clamp(idx, 0, bins - 1))] += 1;
    }
    const int max_count = *std::max_element(counts.begin(), counts.end());
    if (max_count <= 0) return svg_message(title, "No values within histogram range", width, height);

    const double x0 = 54.0;
    const double y0 = 34.0;
    const double pw = width - 78.0;
    const double ph = height - 74.0;
    const double bin_w = pw / static_cast<double>(bins);

    std::ostringstream out;
    out << svg_begin(width, height, title);
    out << "<text x=\"24\" y=\"26\" class=\"svg-title\">" << html_escape(title) << "</text>";
    append_y_grid(out, x0, y0, pw, ph, 0.0, static_cast<double>(max_count), 4);
    out << "<line x1=\"" << x0 << "\" y1=\"" << (y0 + ph) << "\" x2=\"" << (x0 + pw)
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    out << "<line x1=\"" << x0 << "\" y1=\"" << y0 << "\" x2=\"" << x0
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    for (int i = 0; i < bins; ++i) {
        const double h = scale_linear(static_cast<double>(counts[static_cast<size_t>(i)]), 0.0,
                                      static_cast<double>(max_count), 0.0, ph);
        const double x = x0 + i * bin_w;
        const double y = y0 + ph - h;
        out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << std::max(1.0, bin_w - 1.0)
            << "\" height=\"" << h << "\" fill=\"" << color << "\" opacity=\"0.88\"/>";
    }
    const double median_x = scale_linear(stats.median, lo, hi, x0, x0 + pw);
    out << "<line x1=\"" << median_x << "\" y1=\"" << y0 << "\" x2=\"" << median_x
        << "\" y2=\"" << (y0 + ph) << "\" stroke=\"#f87171\" stroke-dasharray=\"6 4\" stroke-width=\"0.9\" vector-effect=\"non-scaling-stroke\"/>";
    out << "<text x=\"" << x0 << "\" y=\"" << (height - 8)
        << "\" class=\"svg-tick\" text-anchor=\"start\">" << html_escape(format_number(lo, 2)) << "</text>";
    out << "<text x=\"" << (x0 + pw * 0.5) << "\" y=\"" << (height - 8)
        << "\" class=\"svg-label\" text-anchor=\"middle\">" << html_escape(xlabel) << "</text>";
    out << "<text x=\"" << (x0 + pw) << "\" y=\"" << (height - 8)
        << "\" class=\"svg-tick\" text-anchor=\"end\">" << html_escape(format_number(hi, 2)) << "</text>";
    out << "</svg>";
    return out.str();
}

std::string svg_scatter(const std::vector<double>& raw_x,
                        const std::vector<double>& raw_y,
                        const std::optional<std::vector<double>>& color_values,
                        const std::string& title,
                        const std::string& xlabel,
                        const std::string& ylabel,
                        const std::string& cmap = "plasma",
                        int width = 620,
                        int height = 420) {
    struct Point {
        double x;
        double y;
        double c;
    };
    std::vector<Point> pts;
    const size_t n = std::min(raw_x.size(), raw_y.size());
    pts.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(raw_x[i]) || !std::isfinite(raw_y[i])) continue;
        const double c = color_values && i < color_values->size() && std::isfinite((*color_values)[i])
            ? (*color_values)[i]
            : static_cast<double>(i);
        pts.push_back({raw_x[i], raw_y[i], c});
    }
    if (pts.size() < 2) return svg_message(title, "Not enough data", width, height);

    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> cs;
    xs.reserve(pts.size());
    ys.reserve(pts.size());
    cs.reserve(pts.size());
    for (const auto& p : pts) {
        xs.push_back(p.x);
        ys.push_back(p.y);
        cs.push_back(p.c);
    }
    const auto xb = plot_bounds(xs, false);
    const auto yb = plot_bounds(ys, false);
    const auto cb = plot_bounds(cs, false);

    const double x0 = 58.0;
    const double y0 = 34.0;
    const double pw = width - 108.0;
    const double ph = height - 72.0;
    const double cbx = x0 + pw + 24.0;
    const double cbw = 12.0;

    std::ostringstream out;
    out << svg_begin(width, height, title);
    out << "<text x=\"24\" y=\"26\" class=\"svg-title\">" << html_escape(title) << "</text>";
    append_y_grid(out, x0, y0, pw, ph, yb.first, yb.second, 4);
    out << "<line x1=\"" << x0 << "\" y1=\"" << (y0 + ph) << "\" x2=\"" << (x0 + pw)
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    out << "<line x1=\"" << x0 << "\" y1=\"" << y0 << "\" x2=\"" << x0
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    out << "<text x=\"" << (x0 + pw * 0.5) << "\" y=\"" << (height - 8)
        << "\" class=\"svg-label\" text-anchor=\"middle\">" << html_escape(xlabel) << "</text>";
    out << "<text x=\"24\" y=\"" << (y0 + ph * 0.5) << "\" class=\"svg-label\" transform=\"rotate(-90 24 "
        << (y0 + ph * 0.5) << ")\">" << html_escape(ylabel) << "</text>";
    out << "<text x=\"" << x0 << "\" y=\"" << (height - 8)
        << "\" class=\"svg-tick\" text-anchor=\"start\">" << html_escape(format_number(xb.first, 2)) << "</text>";
    out << "<text x=\"" << (x0 + pw) << "\" y=\"" << (height - 8)
        << "\" class=\"svg-tick\" text-anchor=\"end\">" << html_escape(format_number(xb.second, 2)) << "</text>";

    const double radius = pts.size() > 600 ? 1.6 : 2.4;
    for (const auto& p : pts) {
        const double x = scale_linear(p.x, xb.first, xb.second, x0, x0 + pw);
        const double y = scale_linear(p.y, yb.first, yb.second, y0 + ph, y0);
        const double t = cb.second > cb.first ? (p.c - cb.first) / (cb.second - cb.first) : 0.5;
        out << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"" << radius
            << "\" fill=\"" << colormap_hex(cmap, t) << "\" opacity=\"0.82\"/>";
    }

    for (int i = 0; i < 64; ++i) {
        const double t0 = static_cast<double>(i) / 64.0;
        const double y = y0 + ph - t0 * ph;
        out << "<rect x=\"" << cbx << "\" y=\"" << y << "\" width=\"" << cbw << "\" height=\"" << (ph / 64.0 + 1.0)
            << "\" fill=\"" << colormap_hex(cmap, t0) << "\"/>";
    }
    out << "<rect x=\"" << cbx << "\" y=\"" << y0 << "\" width=\"" << cbw << "\" height=\"" << ph
        << "\" fill=\"none\" class=\"svg-axis\"/>";
    out << "<text x=\"" << (cbx + cbw + 6) << "\" y=\"" << (y0 + 4)
        << "\" class=\"svg-tick\">" << html_escape(format_number(cb.second, 2)) << "</text>";
    out << "<text x=\"" << (cbx + cbw + 6) << "\" y=\"" << (y0 + ph)
        << "\" class=\"svg-tick\">" << html_escape(format_number(cb.first, 2)) << "</text>";
    out << "</svg>";
    return out.str();
}

std::string svg_bar(const std::vector<std::string>& labels,
                    const std::vector<double>& values,
                    const std::string& title,
                    const std::string& ylabel,
                    const std::vector<std::string>& colors = {},
                    int width = 660,
                    int height = 320) {
    if (labels.empty() || labels.size() != values.size()) return svg_message(title, "No data", width, height);
    const double max_val = std::max(0.0, *std::max_element(values.begin(), values.end()));
    const double top_val = max_val > 0.0 ? max_val * 1.12 : 1.0;
    const double x0 = 52.0;
    const double y0 = 34.0;
    const double pw = width - 76.0;
    const double ph = height - 92.0;
    const double step = pw / static_cast<double>(labels.size());
    const double bar_w = std::max(6.0, step * 0.72);
    const double min_visible_h = 2.0;

    std::ostringstream out;
    out << svg_begin(width, height, title);
    out << "<text x=\"24\" y=\"26\" class=\"svg-title\">" << html_escape(title) << "</text>";
    append_y_grid(out, x0, y0, pw, ph, 0.0, top_val, 4);
    out << "<line x1=\"" << x0 << "\" y1=\"" << (y0 + ph) << "\" x2=\"" << (x0 + pw)
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    out << "<line x1=\"" << x0 << "\" y1=\"" << y0 << "\" x2=\"" << x0
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    out << "<text x=\"24\" y=\"" << (y0 + ph * 0.5) << "\" class=\"svg-label\" transform=\"rotate(-90 24 "
        << (y0 + ph * 0.5) << ")\">" << html_escape(ylabel) << "</text>";

    for (size_t i = 0; i < labels.size(); ++i) {
        const double x = x0 + i * step + (step - bar_w) * 0.5;
        const std::string color = [&]() {
            if (i < colors.size()) return colors[i];
            const std::string preferred = preferred_channel_color(labels[i]);
            if (!preferred.empty()) return preferred;
            return colormap_hex("plasma", static_cast<double>(i) / std::max<size_t>(1, labels.size() - 1));
        }();
        const double raw_h = std::max(0.0, scale_linear(values[i], 0.0, top_val, 0.0, ph));
        const double draw_h = raw_h > 0.0 ? std::max(raw_h, min_visible_h) : 0.0;
        const double y = y0 + ph - draw_h;
        out << "<rect x=\"" << x << "\" y=\"" << y0 << "\" width=\"" << bar_w << "\" height=\"" << ph
            << "\" rx=\"3\" fill=\"none\" stroke=\"#1e293b\" stroke-width=\"0.8\" opacity=\"0.55\"/>";
        if (draw_h > 0.0) {
            out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << bar_w << "\" height=\"" << draw_h
                << "\" rx=\"3\" fill=\"" << color << "\" opacity=\"0.9\"/>";
        } else {
            out << "<line x1=\"" << (x + 1.0) << "\" y1=\"" << (y0 + ph - 1.0) << "\" x2=\"" << (x + bar_w - 1.0)
                << "\" y2=\"" << (y0 + ph - 1.0) << "\" stroke=\"" << color << "\" stroke-width=\"2\" opacity=\"0.9\"/>";
        }
        if (labels.size() <= 12) {
            out << "<text x=\"" << (x + bar_w * 0.5) << "\" y=\"" << (y - 6)
                << "\" class=\"svg-tick\" text-anchor=\"middle\">" << html_escape(format_number(values[i], 2))
                << "</text>";
        }
        const std::string label = sanitize_label(labels[i]);
        const double lx = x + bar_w * 0.5;
        const double ly = y0 + ph + 18.0;
        out << "<text x=\"" << lx << "\" y=\"" << ly
            << "\" class=\"svg-tick\" text-anchor=\"end\" transform=\"rotate(-25 " << lx << ' ' << ly << ")\">"
            << html_escape(label) << "</text>";
    }
    out << "</svg>";
    return out.str();
}

std::string svg_bar_horizontal(const std::vector<std::string>& labels,
                               const std::vector<double>& values,
                               const std::string& title,
                               const std::string& xlabel,
                               const std::vector<std::string>& colors = {},
                               int width = 760) {
    if (labels.empty() || labels.size() != values.size()) return svg_message(title, "No data", width, 220);
    const int height = std::max(180, 88 + static_cast<int>(labels.size()) * 36);
    const double max_val = std::max(1.0, *std::max_element(values.begin(), values.end()) * 1.18);
    const double x0 = 180.0;
    const double y0 = 36.0;
    const double pw = width - 220.0;
    const double ph = height - 74.0;
    const double step = ph / static_cast<double>(labels.size());
    const double bar_h = std::max(14.0, step * 0.68);
    const double min_visible_w = 2.0;

    std::ostringstream out;
    out << svg_begin(width, height, title);
    out << "<text x=\"24\" y=\"26\" class=\"svg-title\">" << html_escape(title) << "</text>";
    for (int i = 0; i <= 4; ++i) {
        const double t = static_cast<double>(i) / 4.0;
        const double x = x0 + t * pw;
        const double v = t * max_val;
        out << "<line x1=\"" << x << "\" y1=\"" << y0 << "\" x2=\"" << x << "\" y2=\"" << (y0 + ph)
            << "\" class=\"svg-grid\"/>";
        out << "<text x=\"" << x << "\" y=\"" << (height - 10)
            << "\" class=\"svg-tick\" text-anchor=\"middle\">" << html_escape(format_number(v, 2)) << "</text>";
    }
    out << "<line x1=\"" << x0 << "\" y1=\"" << y0 << "\" x2=\"" << x0
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    out << "<line x1=\"" << x0 << "\" y1=\"" << (y0 + ph) << "\" x2=\"" << (x0 + pw)
        << "\" y2=\"" << (y0 + ph) << "\" class=\"svg-axis\"/>";
    out << "<text x=\"" << (x0 + pw * 0.5) << "\" y=\"" << (height - 10)
        << "\" class=\"svg-label\" text-anchor=\"middle\">" << html_escape(xlabel) << "</text>";

    for (size_t i = 0; i < labels.size(); ++i) {
        const double y = y0 + i * step + (step - bar_h) * 0.5;
        const std::string color = i < colors.size() ? colors[i] : colormap_hex("viridis", static_cast<double>(i) / std::max<size_t>(1, labels.size() - 1));
        const double raw_w = std::max(0.0, scale_linear(values[i], 0.0, max_val, 0.0, pw));
        const double draw_w = raw_w > 0.0 ? std::max(raw_w, min_visible_w) : 0.0;
        out << "<rect x=\"" << x0 << "\" y=\"" << y << "\" width=\"" << pw << "\" height=\"" << bar_h
            << "\" rx=\"4\" fill=\"none\" stroke=\"#1e293b\" stroke-width=\"0.8\" opacity=\"0.55\"/>";
        if (draw_w > 0.0) {
            out << "<rect x=\"" << x0 << "\" y=\"" << y << "\" width=\"" << draw_w << "\" height=\"" << bar_h
                << "\" rx=\"4\" fill=\"" << color << "\" opacity=\"0.9\"/>";
        } else {
            out << "<line x1=\"" << (x0 + 1.0) << "\" y1=\"" << (y + 1.0) << "\" x2=\"" << (x0 + 1.0)
                << "\" y2=\"" << (y + bar_h - 1.0) << "\" stroke=\"" << color << "\" stroke-width=\"2\" opacity=\"0.9\"/>";
        }
        out << "<text x=\"" << (x0 - 10) << "\" y=\"" << (y + bar_h * 0.5 + 4)
            << "\" class=\"svg-tick\" text-anchor=\"end\">" << html_escape(sanitize_label(labels[i]))
            << "</text>";
        out << "<text x=\"" << (x0 + draw_w + 8) << "\" y=\"" << (y + bar_h * 0.5 + 4)
            << "\" class=\"svg-tick\">" << html_escape(format_number(values[i], 2)) << "</text>";
    }
    out << "</svg>";
    return out.str();
}

std::string svg_pie(const std::vector<std::string>& labels,
                    const std::vector<double>& values,
                    const std::vector<std::string>& colors,
                    const std::string& title,
                    int width = 620,
                    int height = 320) {
    if (labels.empty() || labels.size() != values.size()) return svg_message(title, "No data", width, height);
    double total = 0.0;
    for (double v : values) if (v > 0.0 && std::isfinite(v)) total += v;
    if (total <= 0.0) return svg_message(title, "No positive values", width, height);
    const double cx = 150.0;
    const double cy = height * 0.56;
    const double r = 84.0;
    const double legend_x = 290.0;
    constexpr double pi = 3.14159265358979323846;

    auto polar_x = [&](double angle) { return cx + std::cos(angle) * r; };
    auto polar_y = [&](double angle) { return cy + std::sin(angle) * r; };

    std::ostringstream out;
    out << svg_begin(width, height, title);
    out << "<text x=\"24\" y=\"26\" class=\"svg-title\">" << html_escape(title) << "</text>";
    double angle = -pi * 0.5;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (!(values[i] > 0.0) || !std::isfinite(values[i])) continue;
        const double span = (values[i] / total) * pi * 2.0;
        const double end = angle + span;
        const int large_arc = span > pi ? 1 : 0;
        const std::string color = i < colors.size() ? colors[i] : colormap_hex("plasma", static_cast<double>(i) / std::max<size_t>(1, labels.size() - 1));
        if (span >= pi * 2.0 - 1e-6) {
            out << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r << "\" fill=\"" << color << "\"/>";
        } else {
            out << "<path d=\"M " << cx << ' ' << cy << " L " << polar_x(angle) << ' ' << polar_y(angle)
                << " A " << r << ' ' << r << " 0 " << large_arc << " 1 " << polar_x(end) << ' ' << polar_y(end)
                << " Z\" fill=\"" << color << "\" opacity=\"0.92\"/>";
        }
        const double ly = 72.0 + i * 22.0;
        out << "<rect x=\"" << legend_x << "\" y=\"" << (ly - 11) << "\" width=\"12\" height=\"12\" rx=\"2\" fill=\"" << color << "\"/>";
        out << "<text x=\"" << (legend_x + 20) << "\" y=\"" << ly
            << "\" class=\"svg-tick\">" << html_escape(sanitize_label(labels[i])) << " ("
            << html_escape(format_number(values[i], 1)) << ", " << html_escape(format_number(values[i] / total * 100.0, 1))
            << "%)</text>";
        angle = end;
    }
    out << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << (r * 0.42) << "\" fill=\"#020617\" stroke=\"#1e293b\"/>";
    out << "<text x=\"" << cx << "\" y=\"" << (cy - 2) << "\" class=\"svg-label\" text-anchor=\"middle\">total</text>";
    out << "<text x=\"" << cx << "\" y=\"" << (cy + 18) << "\" class=\"svg-title-small\" text-anchor=\"middle\">"
        << html_escape(format_number(total, 0)) << "</text>";
    out << "</svg>";
    return out.str();
}

/// @brief Implements svg tile overlay.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string svg_tile_overlay(const json& tiles, int img_w, int img_h, const std::string& title, int width = 760, int height = 520) {
    if (!tiles.is_array() || tiles.empty() || img_w <= 0 || img_h <= 0) return svg_message(title, "No tile geometry", width, height);
    const double scale = std::min(620.0 / static_cast<double>(img_w), 400.0 / static_cast<double>(img_h));
    const double panel_w = img_w * scale;
    const double panel_h = img_h * scale;
    const double x0 = 48.0;
    const double y0 = 56.0;

    std::ostringstream out;
    out << svg_begin(width, height, title);
    out << "<text x=\"24\" y=\"28\" class=\"svg-title\">" << html_escape(title) << "</text>";
    out << "<rect x=\"" << x0 << "\" y=\"" << y0 << "\" width=\"" << panel_w << "\" height=\"" << panel_h
        << "\" fill=\"#0f172a\" stroke=\"#475569\"/>";
    for (const auto& tile : tiles) {
        const double x = x0 + json_number_or(tile, "x", 0.0) * scale;
        const double y = y0 + json_number_or(tile, "y", 0.0) * scale;
        const double w = json_number_or(tile, "width", 0.0) * scale;
        const double h = json_number_or(tile, "height", 0.0) * scale;
        out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << w << "\" height=\"" << h
            << "\" fill=\"none\" stroke=\"#7aa2f7\" stroke-width=\"0.8\" opacity=\"0.75\"/>";
    }
    out << "<text x=\"" << x0 << "\" y=\"" << (height - 14)
        << "\" class=\"svg-tick\">" << img_w << " x " << img_h << " px</text>";
    out << "</svg>";
    return out.str();
}

std::string svg_spatial_tile_heatmap(const json& tiles,
                                     const std::vector<double>& values,
                                     int img_w,
                                     int img_h,
                                     const std::string& title,
                                     const std::string& label,
                                     const std::string& cmap = "viridis",
                                     bool force_unit_interval = false,
                                     bool show_grid = true,
                                     int width = 760,
                                     int height = 520) {
    if (!tiles.is_array() || tiles.empty() || values.empty() || img_w <= 0 || img_h <= 0) {
        return svg_message(title, "No spatial tile data", width, height);
    }
    const size_t n = std::min(tiles.size(), values.size());
    std::vector<double> used_values;
    used_values.reserve(n);
    for (size_t i = 0; i < n; ++i) if (std::isfinite(values[i])) used_values.push_back(values[i]);
    if (used_values.empty()) return svg_message(title, "No finite tile values", width, height);

    auto s = basic_stats(used_values);
    double lo = s.min;
    double hi = s.max;
    const bool flat_map = !(s.max > s.min);
    if (flat_map) {
        lo = s.min;
        hi = s.max;
    } else if (force_unit_interval) {
        lo = 0.0;
        hi = 1.0;
    } else if (s.n >= 20 && s.p99 > s.p01) {
        lo = s.p01;
        hi = s.p99;
    }

    const double scale = std::min(620.0 / static_cast<double>(img_w), 400.0 / static_cast<double>(img_h));
    const double panel_w = img_w * scale;
    const double panel_h = img_h * scale;
    const double x0 = 44.0;
    const double y0 = 56.0;
    const double cbx = x0 + panel_w + 26.0;
    const double cbw = 16.0;

    std::ostringstream out;
    out << svg_begin(width, height, title);
    out << "<text x=\"24\" y=\"28\" class=\"svg-title\">" << html_escape(title) << "</text>";
    out << "<rect x=\"" << x0 << "\" y=\"" << y0 << "\" width=\"" << panel_w << "\" height=\"" << panel_h
        << "\" fill=\"#0f172a\" stroke=\"#475569\"/>";
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) continue;
        const auto& tile = tiles.at(i);
        const double x = x0 + json_number_or(tile, "x", 0.0) * scale;
        const double y = y0 + json_number_or(tile, "y", 0.0) * scale;
        const double w = json_number_or(tile, "width", 0.0) * scale;
        const double h = json_number_or(tile, "height", 0.0) * scale;
        const double t = hi > lo ? (values[i] - lo) / (hi - lo) : 0.5;
        out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << w << "\" height=\"" << h
            << "\" fill=\"" << colormap_hex(cmap, t) << "\"";
        if (show_grid) out << " stroke=\"#0f172a\" stroke-width=\"0.4\"";
        out << "/>";
    }
    if (flat_map) {
        out << "<rect x=\"" << cbx << "\" y=\"" << y0 << "\" width=\"" << cbw << "\" height=\"" << panel_h
            << "\" fill=\"" << colormap_hex(cmap, 0.5) << "\"/>";
    } else {
        for (int i = 0; i < 64; ++i) {
            const double t = static_cast<double>(i) / 63.0;
            const double y = y0 + panel_h - t * panel_h;
            out << "<rect x=\"" << cbx << "\" y=\"" << y << "\" width=\"" << cbw << "\" height=\"" << (panel_h / 63.0 + 1.0)
                << "\" fill=\"" << colormap_hex(cmap, t) << "\"/>";
        }
    }
    out << "<rect x=\"" << cbx << "\" y=\"" << y0 << "\" width=\"" << cbw << "\" height=\"" << panel_h
        << "\" fill=\"none\" class=\"svg-axis\"/>";
    out << "<text x=\"" << cbx << "\" y=\"" << (height - 16)
        << "\" class=\"svg-label\">" << html_escape(label) << "</text>";
    if (flat_map) {
        out << "<text x=\"" << (cbx + cbw + 8) << "\" y=\"" << (y0 + panel_h * 0.5)
            << "\" class=\"svg-tick\">" << html_escape(format_number(s.min, 2)) << "</text>";
        out << "<text x=\"" << (cbx + cbw + 8) << "\" y=\"" << (y0 + panel_h * 0.5 + 16)
            << "\" class=\"svg-tick\">konstant</text>";
    } else {
        out << "<text x=\"" << (cbx + cbw + 8) << "\" y=\"" << (y0 + 4)
            << "\" class=\"svg-tick\">" << html_escape(format_number(hi, 2)) << "</text>";
        out << "<text x=\"" << (cbx + cbw + 8) << "\" y=\"" << (y0 + panel_h)
            << "\" class=\"svg-tick\">" << html_escape(format_number(lo, 2)) << "</text>";
    }
    out << "</svg>";
    return out.str();
}


std::string svg_matrix_heatmap(const std::vector<double>& values,
                               int cols,
                               int rows,
                               const std::string& title,
                               const std::string& label,
                               const std::string& cmap = "viridis",
                               double lo = 0.0,
                               double hi = 1.0,
                               int width = 760,
                               int height = 520) {
    if (cols <= 0 || rows <= 0 || values.size() < static_cast<size_t>(cols * rows)) {
        return svg_message(title, "No matrix data", width, height);
    }
    constexpr int max_svg_heatmap_cols = 80;
    constexpr int max_svg_heatmap_rows = 48;
    std::vector<double> downsampled_values;
    int render_cols = cols;
    int render_rows = rows;
    int sample_step_x = 1;
    int sample_step_y = 1;
    if (cols > max_svg_heatmap_cols || rows > max_svg_heatmap_rows) {
        sample_step_x = std::max(1, static_cast<int>(std::ceil(static_cast<double>(cols) / max_svg_heatmap_cols)));
        sample_step_y = std::max(1, static_cast<int>(std::ceil(static_cast<double>(rows) / max_svg_heatmap_rows)));
        render_cols = (cols + sample_step_x - 1) / sample_step_x;
        render_rows = (rows + sample_step_y - 1) / sample_step_y;
        downsampled_values.assign(static_cast<size_t>(render_cols * render_rows),
                                  std::numeric_limits<double>::quiet_NaN());
        for (int by = 0; by < render_rows; ++by) {
            for (int bx = 0; bx < render_cols; ++bx) {
                double sum = 0.0;
                int count = 0;
                const int y_end = std::min(rows, (by + 1) * sample_step_y);
                const int x_end = std::min(cols, (bx + 1) * sample_step_x);
                for (int y = by * sample_step_y; y < y_end; ++y) {
                    for (int x = bx * sample_step_x; x < x_end; ++x) {
                        const double v = values[static_cast<size_t>(y * cols + x)];
                        if (!std::isfinite(v)) continue;
                        sum += v;
                        ++count;
                    }
                }
                if (count > 0) downsampled_values[static_cast<size_t>(by * render_cols + bx)] = sum / count;
            }
        }
    }
    const std::vector<double>& render_values = downsampled_values.empty() ? values : downsampled_values;
    const double x0 = 44.0;
    const double y0 = 56.0;
    const double max_panel_w = 620.0;
    const double max_panel_h = 400.0;
    const double cell = std::max(1.0, std::min(max_panel_w / render_cols, max_panel_h / render_rows));
    const double panel_w = render_cols * cell;
    const double panel_h = render_rows * cell;
    const double cbx = x0 + panel_w + 26.0;
    const double cbw = 16.0;
    if (!(hi > lo)) {
        std::vector<double> finite;
        finite.reserve(render_values.size());
        for (double v : render_values) if (std::isfinite(v)) finite.push_back(v);
        if (!finite.empty()) {
            const auto stats = basic_stats(finite);
            lo = stats.min;
            hi = stats.max;
        }
    }
    const bool flat_map = !(hi > lo);
    std::ostringstream out;
    out << svg_begin(width, height, title);
    out << "<text x=\"24\" y=\"28\" class=\"svg-title\">" << html_escape(title) << "</text>";
    out << "<rect x=\"" << x0 << "\" y=\"" << y0 << "\" width=\"" << panel_w << "\" height=\"" << panel_h
        << "\" fill=\"#0f172a\" stroke=\"#475569\"/>";
    for (int y = 0; y < render_rows; ++y) {
        for (int x = 0; x < render_cols; ++x) {
            const double v = render_values[static_cast<size_t>(y * render_cols + x)];
            if (!std::isfinite(v)) continue;
            const double t = flat_map ? 0.5 : std::clamp((v - lo) / (hi - lo), 0.0, 1.0);
            out << "<rect x=\"" << (x0 + x * cell) << "\" y=\"" << (y0 + y * cell)
                << "\" width=\"" << std::ceil(cell) << "\" height=\"" << std::ceil(cell)
                << "\" fill=\"" << colormap_hex(cmap, t) << "\"/>";
        }
    }
    if (flat_map) {
        out << "<rect x=\"" << cbx << "\" y=\"" << y0 << "\" width=\"" << cbw << "\" height=\"" << panel_h
            << "\" fill=\"" << colormap_hex(cmap, 0.5) << "\"/>";
    } else {
        for (int i = 0; i < 64; ++i) {
            const double t = static_cast<double>(i) / 63.0;
            const double y = y0 + panel_h - t * panel_h;
            out << "<rect x=\"" << cbx << "\" y=\"" << y << "\" width=\"" << cbw << "\" height=\"" << (panel_h / 63.0 + 1.0)
                << "\" fill=\"" << colormap_hex(cmap, t) << "\"/>";
        }
    }
    out << "<rect x=\"" << cbx << "\" y=\"" << y0 << "\" width=\"" << cbw << "\" height=\"" << panel_h
        << "\" fill=\"none\" class=\"svg-axis\"/>";
    out << "<text x=\"" << cbx << "\" y=\"" << (height - 16)
        << "\" class=\"svg-label\">" << html_escape(label) << "</text>";
    if (sample_step_x > 1 || sample_step_y > 1) {
        out << "<text x=\"24\" y=\"" << (height - 16)
            << "\" class=\"svg-note\">downsampled " << cols << "x" << rows
            << " to " << render_cols << "x" << render_rows << "</text>";
    }
    out << "<text x=\"" << (cbx + cbw + 8) << "\" y=\"" << (y0 + 4)
        << "\" class=\"svg-tick\">" << html_escape(format_number(flat_map ? lo : hi, 2)) << "</text>";
    out << "<text x=\"" << (cbx + cbw + 8) << "\" y=\"" << (y0 + panel_h)
        << "\" class=\"svg-tick\">" << html_escape(format_number(lo, 2)) << "</text>";
    out << "</svg>";
    return out.str();
}

std::optional<double> finite_json_number(const json& j, const std::string& key) {
    if (!j.contains(key) || !j[key].is_number()) return std::nullopt;
    const double value = j[key].get<double>();
    if (!std::isfinite(value)) return std::nullopt;
    return value;
}

std::vector<double> json_diag_values(const json& diagnostics, const std::string& key) {
    std::vector<double> values;
    if (!diagnostics.is_array()) return values;
    for (const auto& item : diagnostics) {
        if (!item.is_object() || !item.value("written", true)) continue;
        if (auto value = finite_json_number(item, key)) values.push_back(*value);
    }
    return values;
}

fs::path aqmh_cache_dir(const fs::path& run_dir, const json& metrics) {
    const std::string raw = json_string_or(metrics, "cache_dir", "");
    if (!raw.empty()) {
        fs::path path(raw);
        if (path.is_absolute()) return path;
        return run_dir / path;
    }
    return run_dir / "cache" / "aqmh";
}

std::vector<fs::path> aqmh_cache_files(const fs::path& cache_dir, const std::string& stream_id) {
    std::vector<fs::path> files;
    std::error_code ec;
    if (!fs::is_directory(cache_dir, ec) || ec) return files;
    const std::string prefix = "aqmh_" + (stream_id.empty() ? std::string() : stream_id + "_");
    for (const auto& entry : fs::directory_iterator(cache_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0 && entry.path().extension() == ".bin") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<fs::path> sample_evenly(const std::vector<fs::path>& files, size_t max_count) {
    if (max_count == 0 || files.size() <= max_count) return files;
    std::vector<fs::path> sampled;
    sampled.reserve(max_count);
    for (size_t i = 0; i < max_count; ++i) {
        const size_t idx = static_cast<size_t>(std::llround(
            static_cast<double>(i) * static_cast<double>(files.size() - 1) /
            static_cast<double>(max_count - 1)));
        if (sampled.empty() || sampled.back() != files[idx]) sampled.push_back(files[idx]);
    }
    return sampled;
}

std::optional<std::vector<double>> read_aqmh_cache_map(const fs::path& path, int width, int height, const std::string& dtype) {
    if (width <= 0 || height <= 0) return std::nullopt;
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::vector<double> out;
    out.reserve(static_cast<size_t>(width * height));
    for (int i = 0; i < width * height; ++i) {
        if (dtype == "float32") {
            float v = 0.0f;
            in.read(reinterpret_cast<char*>(&v), sizeof(float));
            if (!in) return std::nullopt;
            out.push_back(std::clamp(static_cast<double>(v), 0.0, 1.0));
        } else if (dtype == "uint16") {
            uint16_t v = 0;
            in.read(reinterpret_cast<char*>(&v), sizeof(uint16_t));
            if (!in) return std::nullopt;
            out.push_back(static_cast<double>(v) / 65535.0);
        } else if (dtype == "uint8") {
            uint8_t v = 0;
            in.read(reinterpret_cast<char*>(&v), sizeof(uint8_t));
            if (!in) return std::nullopt;
            out.push_back(static_cast<double>(v) / 255.0);
        } else {
            return std::nullopt;
        }
    }
    return out;
}

struct AqmhMapAggregate {
    int count = 0;
    std::vector<double> mean;
    std::vector<double> stddev;
    std::vector<double> artifact_frequency;
    std::vector<double> min_map;
    std::vector<std::pair<std::string, std::vector<double>>> examples;
};

AqmhMapAggregate aggregate_aqmh_maps(const std::vector<fs::path>& files,
                                     int width,
                                     int height,
                                     const std::string& dtype,
                                     double artifact_threshold) {
    AqmhMapAggregate agg;
    const size_t n = static_cast<size_t>(width * height);
    if (n == 0) return agg;
    std::vector<double> sum(n, 0.0), sumsq(n, 0.0), artifacts(n, 0.0), min_map(n, 1.0);
    std::vector<std::tuple<double, std::string, std::vector<double>>> examples;
    for (const auto& path : files) {
        auto maybe_map = read_aqmh_cache_map(path, width, height, dtype);
        if (!maybe_map) continue;
        const auto& values = *maybe_map;
        double mean = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double v = values[i];
            mean += v;
            sum[i] += v;
            sumsq[i] += v * v;
            if (v < artifact_threshold) artifacts[i] += 1.0;
            min_map[i] = std::min(min_map[i], v);
        }
        mean /= static_cast<double>(n);
        examples.emplace_back(mean, path.stem().string(), values);
        ++agg.count;
    }
    if (agg.count == 0) return agg;
    agg.mean.resize(n);
    agg.stddev.resize(n);
    agg.artifact_frequency.resize(n);
    agg.min_map = std::move(min_map);
    for (size_t i = 0; i < n; ++i) {
        agg.mean[i] = sum[i] / agg.count;
        const double variance = std::max(0.0, (sumsq[i] / agg.count) - agg.mean[i] * agg.mean[i]);
        agg.stddev[i] = std::sqrt(variance);
        agg.artifact_frequency[i] = artifacts[i] / agg.count;
    }
    std::sort(examples.begin(), examples.end(), [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
    const std::array<size_t, 3> example_indices = {size_t{0}, examples.size() / 2, examples.size() - 1};
    for (size_t idx : example_indices) {
        if (idx < examples.size()) agg.examples.push_back({std::get<1>(examples[idx]), std::get<2>(examples[idx])});
    }
    return agg;
}

struct AqmhReportMapAggregate {
    int count = 0;
    int cols = 0;
    int rows = 0;
    std::vector<double> mean;
    std::vector<double> artifact_frequency;
    std::pair<std::string, std::vector<double>> example;
};

bool read_aqmh_value(std::ifstream& in, const std::string& dtype, double& out) {
    if (dtype == "float32") {
        float v = 0.0f;
        in.read(reinterpret_cast<char*>(&v), sizeof(float));
        if (!in) return false;
        out = std::clamp(static_cast<double>(v), 0.0, 1.0);
        return true;
    }
    if (dtype == "uint16") {
        uint16_t v = 0;
        in.read(reinterpret_cast<char*>(&v), sizeof(uint16_t));
        if (!in) return false;
        out = static_cast<double>(v) / 65535.0;
        return true;
    }
    if (dtype == "uint8") {
        uint8_t v = 0;
        in.read(reinterpret_cast<char*>(&v), sizeof(uint8_t));
        if (!in) return false;
        out = static_cast<double>(v) / 255.0;
        return true;
    }
    return false;
}

AqmhReportMapAggregate aggregate_aqmh_maps_streamed(const std::vector<fs::path>& files,
                                                    int width,
                                                    int height,
                                                    const std::string& dtype,
                                                    double artifact_threshold,
                                                    int out_cols,
                                                    int out_rows) {
    AqmhReportMapAggregate agg;
    if (width <= 0 || height <= 0 || out_cols <= 0 || out_rows <= 0) return agg;
    agg.cols = out_cols;
    agg.rows = out_rows;
    const size_t out_n = static_cast<size_t>(out_cols * out_rows);
    std::vector<double> sum(out_n, 0.0), artifact_sum(out_n, 0.0);
    std::vector<uint64_t> samples(out_n, 0);
    const size_t example_idx = files.empty() ? 0 : files.size() / 2;

    for (size_t file_idx = 0; file_idx < files.size(); ++file_idx) {
        std::ifstream in(files[file_idx], std::ios::binary);
        if (!in) continue;
        std::vector<double> example_sum;
        std::vector<uint64_t> example_samples;
        const bool capture_example = file_idx == example_idx;
        if (capture_example) {
            example_sum.assign(out_n, 0.0);
            example_samples.assign(out_n, 0);
        }

        bool ok = true;
        for (int y = 0; ok && y < height; ++y) {
            const int by = std::min(out_rows - 1, static_cast<int>((static_cast<int64_t>(y) * out_rows) / height));
            for (int x = 0; x < width; ++x) {
                double v = 0.0;
                if (!read_aqmh_value(in, dtype, v)) {
                    ok = false;
                    break;
                }
                const int bx = std::min(out_cols - 1, static_cast<int>((static_cast<int64_t>(x) * out_cols) / width));
                const size_t bi = static_cast<size_t>(by * out_cols + bx);
                sum[bi] += v;
                artifact_sum[bi] += v < artifact_threshold ? 1.0 : 0.0;
                samples[bi] += 1;
                if (capture_example) {
                    example_sum[bi] += v;
                    example_samples[bi] += 1;
                }
            }
        }
        if (!ok) continue;
        ++agg.count;
        if (capture_example) {
            agg.example.first = files[file_idx].stem().string();
            agg.example.second.assign(out_n, std::numeric_limits<double>::quiet_NaN());
            for (size_t i = 0; i < out_n; ++i) {
                if (example_samples[i] > 0) {
                    agg.example.second[i] = example_sum[i] / static_cast<double>(example_samples[i]);
                }
            }
        }
    }

    if (agg.count == 0) return agg;
    agg.mean.assign(out_n, std::numeric_limits<double>::quiet_NaN());
    agg.artifact_frequency.assign(out_n, std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < out_n; ++i) {
        if (samples[i] == 0) continue;
        const double denom = static_cast<double>(samples[i]);
        agg.mean[i] = sum[i] / denom;
        agg.artifact_frequency[i] = artifact_sum[i] / denom;
    }
    return agg;
}

/// @brief Renders kv table.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string render_kv_table(const std::vector<std::pair<std::string, std::string>>& rows) {
    std::ostringstream html;
    html << "<table class=\"kv\"><tbody>";
    for (const auto& [key, value] : rows) {
        html << "<tr><th>" << html_escape(key) << "</th><td>" << html_escape(value) << "</td></tr>";
    }
    html << "</tbody></table>";
    return html.str();
}

/// @brief Renders artifacts list.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string render_artifacts_list(const json& artifacts, size_t max_items = 40) {
    std::ostringstream html;
    html << "<ul class=\"artifact-list\">";
    size_t count = 0;
    if (artifacts.is_array()) {
        for (const auto& item : artifacts) {
            if (count >= max_items) break;
            const std::string path = json_string_or(item, "path", "");
            const auto size = static_cast<long long>(json_number_or(item, "size", 0.0));
            html << "<li><code>" << html_escape(path) << "</code> <span class=\"muted\">("
                 << size << " bytes)</span></li>";
            ++count;
        }
    }
    if (count == 0) html << "<li class=\"muted\">No artifacts found</li>";
    if (artifacts.is_array() && artifacts.size() > count) {
        html << "<li class=\"muted\">+" << (artifacts.size() - count) << " more files</li>";
    }
    html << "</ul>";
    return html.str();
}

/// @brief Renders phase summary.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string render_phase_summary(const json& status) {
    if (!status.contains("phases") || !status["phases"].is_array()) {
        return "<p class=\"muted\">No phase information available.</p>";
    }
    std::ostringstream html;
    html << "<table class=\"phases\"><thead><tr><th>Phase</th><th>Status</th><th>Progress</th></tr></thead><tbody>";
    for (const auto& phase : status["phases"]) {
        html << "<tr><td>" << html_escape(json_string_or(phase, "phase", "")) << "</td>"
             << "<td>" << html_escape(json_string_or(phase, "status", "")) << "</td>"
             << "<td>" << html_escape(format_number(percent_value(json_number_or(phase, "pct", 0.0)), 1)) << "%</td></tr>";
    }
    html << "</tbody></table>";
    return html.str();
}

/// @brief Renders event tail.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string render_event_tail(const std::vector<json>& events, size_t max_lines = 24) {
    std::ostringstream text;
    const size_t start = events.size() > max_lines ? events.size() - max_lines : 0;
    for (size_t i = start; i < events.size(); ++i) {
        text << format_event_line(events[i]) << '\n';
    }
    return "<pre class=\"log-tail\">" + html_escape(text.str()) + "</pre>";
}

/// @brief Implements infer status.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string infer_status(const std::vector<std::string>& evals) {
    std::string text;
    for (const auto& line : evals) {
        text += line;
        text.push_back('\n');
    }
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (text.find("fail") != std::string::npos || text.find("error") != std::string::npos) return "bad";
    if (text.find("warning") != std::string::npos || text.find("skipped") != std::string::npos) return "warn";
    return "ok";
}

std::string explain_panel(const std::string& title,
                          const std::vector<std::string>& paragraphs,
                          const std::vector<std::string>& bullets = {},
                          const std::string& metric_box_html = "") {
    std::ostringstream html;
    html << "<h4>" << html_escape(title) << "</h4>";
    for (const auto& paragraph : paragraphs) {
        if (paragraph.empty()) continue;
        html << "<p>" << paragraph << "</p>";
    }
    if (!bullets.empty()) {
        html << "<ul>";
        for (const auto& bullet : bullets) {
            if (bullet.empty()) continue;
            html << "<li>" << bullet << "</li>";
        }
        html << "</ul>";
    }
    if (!metric_box_html.empty()) {
        html << "<div class=\"metric-box\">" << metric_box_html << "</div>";
    }
    return html.str();
}

/// @brief Builds chart row.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::string make_chart_row(const ChartBlock& chart) {
    std::ostringstream html;
    html << "<div class=\"chart-row\"><div class=\"chart-col\">" << chart.svg << "</div>";
    if (!chart.explanation_html.empty()) {
        html << "<div class=\"explain-col\">" << chart.explanation_html << "</div>";
    }
    html << "</div>";
    return html.str();
}

std::string make_card_html(const std::string& title,
                           const std::vector<ChartBlock>& charts,
                           const std::vector<std::string>& evals,
                           const std::string& status = "") {
    std::ostringstream html;
    std::string badge;
    if (!status.empty()) {
        badge = "<span class=\"badge " + status + "\">" + html_escape(status) + "</span>";
    }
    html << "<article class=\"card " << html_escape(status) << "\"><h3>" << html_escape(title) << badge << "</h3>";
    for (const auto& chart : charts) {
        if (!chart.svg.empty()) html << make_chart_row(chart);
    }
    if (!evals.empty()) {
        html << "<div class=\"metric-box\"><ul>";
        for (const auto& line : evals) {
            if (line.empty()) continue;
            const bool warn = line.find("WARNING") != std::string::npos ||
                              line.find("FAIL") != std::string::npos ||
                              line.find("ERROR") != std::string::npos;
            html << "<li" << (warn ? " class=\"warn\"" : "") << ">" << html_escape(line) << "</li>";
        }
        html << "</ul></div>";
    }
    html << "</article>";
    return html.str();
}

std::string make_plain_card_html(const std::string& title,
                                 const std::string& body_html,
                                 const std::string& status = "") {
    std::ostringstream html;
    std::string badge;
    if (!status.empty()) {
        badge = "<span class=\"badge " + status + "\">" + html_escape(status) + "</span>";
    }
    html << "<article class=\"card " << html_escape(status) << "\"><h3>" << html_escape(title) << badge << "</h3>";
    html << body_html;
    html << "</article>";
    return html.str();
}

std::string human_phase_reason(const std::string& phase,
                               const std::string& status,
                               const std::string& reason) {
    if (phase == "BGE" && reason == "surface_fit_failed") {
        return "BGE wurde angefordert und gestartet, aber kein Kanal konnte einen belastbaren Hintergrund-Surface-Fit anwenden.";
    }
    if (phase == "ASTROMETRY" && reason == "disabled") {
        return "Astrometrie war in der Konfiguration deaktiviert.";
    }
    if (phase == "ASTROMETRY" && reason == "existing_wcs") {
        return "Astrometrie wurde übersprungen, weil bereits eine WCS-Lösung vorhanden war.";
    }
    if (phase == "SYNTHETIC_FRAMES" && reason == "disabled") {
        return "Synthetische Frames waren in der Konfiguration deaktiviert.";
    }
    if (phase == "STATE_CLUSTERING" && reason == "reduced_mode_skip_clustering") {
        return "State-Clustering wurde im Reduced-Mode bewusst übersprungen.";
    }
    if (!reason.empty()) {
        return "Die Phase meldete als Grund: " + reason + ".";
    }
    if (status == "skipped") {
        return "Die Phase wurde übersprungen; das Event enthält keinen spezifischeren Grund.";
    }
    if (status == "error" || status == "failed" || status == "aborted") {
        return "Die Phase wurde abgebrochen oder fehlerhaft beendet; Details stehen in den Event-Feldern und Logs.";
    }
    return "Die Phase endete nicht mit Status ok.";
}

std::vector<std::pair<std::string, std::string>> scalar_event_details(const json& ev) {
    static const std::vector<std::string> preferred = {
        "reason", "error", "message", "artifact", "requested", "attempted", "success",
        "have_tile_data", "have_local_metrics", "metrics_tiles_match",
        "frames_usable", "reg_rejected_frames", "num_synthetic", "source",
        "stars_used", "stars_matched", "residual_rms"
    };
    std::vector<std::pair<std::string, std::string>> rows;
    for (const auto& key : preferred) {
        if (!ev.contains(key) || ev.at(key).is_null()) continue;
        const std::string value = json_string_or(ev, key.c_str(), "");
        if (!value.empty()) rows.push_back({key, value});
    }
    return rows;
}

std::string render_bge_phase_details(const json& bge) {
    if (!bge.is_object() || bge.empty()) return "";

    std::ostringstream html;
    const json summary = bge.contains("summary") && bge["summary"].is_object()
        ? bge["summary"] : json::object();
    const json cfg = bge.contains("config") && bge["config"].is_object()
        ? bge["config"] : json::object();
    const double min_fraction = json_number_or(cfg, "min_valid_sample_fraction_for_apply", 0.0);
    const int min_samples = static_cast<int>(json_number_or(cfg, "min_valid_samples_for_apply", 0.0));
    
    // Show tile metrics source (AQMH-first: aqmh_output vs classic_local_metrics)
    std::string tile_metrics_source = json_string_or(bge, "tile_metrics_source", "");
    if (!tile_metrics_source.empty()) {
        std::string source_label;
        if (tile_metrics_source == "aqmh_output") {
            source_label = "AQMH output";
        } else if (tile_metrics_source == "classic_local_metrics") {
            source_label = "Classic Local Metrics";
        } else {
            source_label = tile_metrics_source;
        }
        html << "<p class=\"muted\">BGE input source: <strong>" << html_escape(source_label) << "</strong>.</p>";
    }

    html << "<p class=\"muted\">BGE artifact summary: channels applied "
         << html_escape(json_string_or(summary, "channels_applied", "0")) << "/"
         << html_escape(json_string_or(summary, "channels_total", "0"))
         << ", fit success " << html_escape(json_string_or(summary, "channels_fit_success", "0"))
         << ", valid tile samples " << html_escape(json_string_or(summary, "tile_samples_valid", "0"))
         << "/" << html_escape(json_string_or(summary, "tile_samples_total", "0")) << ".</p>";

    if (min_fraction > 0.0 || min_samples > 0) {
        html << "<p class=\"muted\">Apply guard: mindestens "
             << min_samples << " valide Samples und "
             << format_number(min_fraction * 100.0, 1) << "% valide Sample-Quote pro Kanal.</p>";
    }

    if (bge.contains("channels") && bge["channels"].is_array() && !bge["channels"].empty()) {
        html << "<table class=\"phases\"><thead><tr><th>Channel</th><th>Samples</th><th>Fit</th><th>Applied</th></tr></thead><tbody>";
        for (const auto& ch : bge["channels"]) {
            const double total = json_number_or(ch, "tile_samples_total", 0.0);
            const double valid = json_number_or(ch, "tile_samples_valid", 0.0);
            const double ratio = total > 0.0 ? valid / total : 0.0;
            html << "<tr><td>" << html_escape(json_string_or(ch, "channel", "?")) << "</td>"
                 << "<td>" << html_escape(format_number(valid, 0)) << "/"
                 << html_escape(format_number(total, 0)) << " ("
                 << html_escape(format_number(ratio * 100.0, 1)) << "%)</td>"
                 << "<td>" << (json_bool_or(ch, "fit_success", false) ? "true" : "false") << "</td>"
                 << "<td>" << (json_bool_or(ch, "applied", false) ? "true" : "false") << "</td></tr>";
        }
        html << "</tbody></table>";
    }
    return html.str();
}

std::optional<ReportSection> gen_phase_issue_summary(const std::vector<json>& events,
                                                     const json& bge) {
    struct Issue {
        std::string phase;
        std::string status;
        std::string ts;
        std::string reason;
        std::string description;
        std::vector<std::pair<std::string, std::string>> details;
        std::string extra_html;
    };

    std::vector<Issue> issues;
    for (const auto& ev : events) {
        if (json_string_or(ev, "type", "") != "phase_end") continue;
        const std::string status = json_string_or(ev, "status", "");
        if (status.empty() || status == "ok") continue;
        const std::string phase = phase_name_from_event(ev);
        const std::string reason = json_string_or(ev, "reason", json_string_or(ev, "error", ""));
        Issue item;
        item.phase = phase.empty() ? "unknown" : phase;
        item.status = status;
        item.ts = json_string_or(ev, "ts", json_string_or(ev, "timestamp", ""));
        item.reason = reason;
        item.description = human_phase_reason(item.phase, status, reason);
        item.details = scalar_event_details(ev);
        if (item.phase == "BGE") item.extra_html = render_bge_phase_details(bge);
        issues.push_back(std::move(item));
    }

    if (issues.empty()) {
        return ReportSection{
            "Phase Issues Summary",
            make_plain_card_html(
                "Abgebrochene oder übersprungene Phasen",
                "<p class=\"muted\">Keine Phase wurde abgebrochen oder übersprungen.</p>",
                "ok")
        };
    }

    std::ostringstream cards;
    for (const auto& issue : issues) {
        std::ostringstream body;
        body << "<p><strong>Status:</strong> " << html_escape(issue.status);
        if (!issue.reason.empty()) body << " · <strong>Reason:</strong> <code>" << html_escape(issue.reason) << "</code>";
        if (!issue.ts.empty()) body << " · <span class=\"muted\">" << html_escape(issue.ts) << "</span>";
        body << "</p>";
        body << "<p>" << html_escape(issue.description) << "</p>";
        if (!issue.details.empty()) {
            body << "<table class=\"kv\"><tbody>";
            for (const auto& [key, value] : issue.details) {
                body << "<tr><th>" << html_escape(key) << "</th><td>" << html_escape(value) << "</td></tr>";
            }
            body << "</tbody></table>";
        }
        body << issue.extra_html;
        const std::string severity = issue.status == "skipped" ? "warn" : "bad";
        cards << make_plain_card_html(issue.phase, body.str(), severity);
    }

    return ReportSection{"Phase Issues Summary", cards.str()};
}

json build_report_summary_json(const fs::path& run_dir,
                               const json& status,
                               const json& artifacts,
                               const std::vector<json>& events) {
    json phase_items = json::array();
    if (status.contains("phases") && status["phases"].is_array()) {
        for (const auto& p : status["phases"]) {
            phase_items.push_back({
                {"phase", json_string_or(p, "phase", "")},
                {"status", json_string_or(p, "status", "")},
                {"progress_percent", percent_value(json_number_or(p, "pct", 0.0))}
            });
        }
    }

    std::map<std::string, int> event_counts;
    for (const auto& ev : events) event_counts[json_string_or(ev, "type", "unknown")] += 1;
    json event_count_items = json::object();
    for (const auto& [key, value] : event_counts) event_count_items[key] = value;

    json artifact_items = json::array();
    if (artifacts.is_array()) {
        for (const auto& item : artifacts) {
            artifact_items.push_back({
                {"path", json_string_or(item, "path", "")},
                {"size_bytes", static_cast<long long>(json_number_or(item, "size", 0.0))}
            });
        }
    }

    return {
        {"run_id", run_dir.filename().string()},
        {"run_dir", run_dir.string()},
        {"status", json_string_or(status, "status", "unknown")},
        {"current_phase", json_string_or(status, "current_phase", "")},
        {"progress_percent", percent_value(json_number_or(status, "progress", 0.0))},
        {"artifact_count", artifacts.is_array() ? static_cast<int>(artifacts.size()) : 0},
        {"event_count", static_cast<int>(events.size())},
        {"event_counts", event_count_items},
        {"phases", phase_items},
        {"artifacts", artifact_items},
        {"report_format", "inline_svg"},
    };
}

std::optional<ReportSection> gen_overview(const fs::path& run_dir,
                                          const json& status,
                                          const json& artifacts,
                                          const std::vector<json>& events) {
    std::vector<std::pair<std::string, std::string>> rows = {
        {"run_id", run_dir.filename().string()},
        {"run_dir", run_dir.string()},
        {"status", json_string_or(status, "status", "unknown")},
        {"current_phase", json_string_or(status, "current_phase", "")},
        {"progress", format_number(percent_value(json_number_or(status, "progress", 0.0)), 1) + "%"},
    };
    for (const auto& ev : events) {
        const auto type = json_string_or(ev, "type", "");
        if (type == "run_start") {
            const auto input_dir = json_string_or(ev, "input_dir", "");
            const auto discovered = json_string_or(ev, "frames_discovered", "");
            const auto ts = json_string_or(ev, "ts", "");
            if (!input_dir.empty()) rows.push_back({"input_dir", input_dir});
            if (!discovered.empty()) rows.push_back({"frames_discovered", discovered});
            if (!ts.empty()) rows.push_back({"started", ts});
        }
        if (type == "run_end") {
            const auto ts = json_string_or(ev, "ts", "");
            const auto st = json_string_or(ev, "status", "");
            if (!ts.empty()) rows.push_back({"finished", ts});
            if (!st.empty()) rows.push_back({"final_status", st});
        }
    }

    std::ostringstream cards;
    cards << make_plain_card_html("Run Summary", render_kv_table(rows));
    cards << make_plain_card_html("Pipeline Phases", render_phase_summary(status));
    cards << make_plain_card_html("Artifacts", render_artifacts_list(artifacts));
    if (!events.empty()) cards << make_plain_card_html("Recent Events", render_event_tail(events));

    if (cards.str().empty()) return std::nullopt;
    return ReportSection{"Overview", cards.str()};
}

/// @brief Generates timeline.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_timeline(const std::vector<json>& events) {
    struct PhaseStart { std::string display_name; double secs; };
    std::map<std::string, PhaseStart> phase_starts;
    std::vector<std::string> labels;
    std::vector<double> durations;
    std::vector<std::string> evals;
    for (const auto& ev : events) {
        const auto type = json_string_or(ev, "type", "");
        const auto display = phase_name_from_event(ev);
        const auto key = phase_match_key(ev);
        const auto ts = json_string_or(ev, "ts", json_string_or(ev, "timestamp", ""));
        if (key.empty() || ts.empty()) continue;
        const auto secs = parse_iso_utc_seconds(ts);
        if (!secs) continue;
        if (type == "phase_start") {
            phase_starts[key] = {display, *secs};
        } else if (type == "phase_end") {
            auto it = phase_starts.find(key);
            if (it == phase_starts.end()) continue;
            const double dt = std::max(0.0, *secs - it->second.secs);
            labels.push_back(it->second.display_name);
            durations.push_back(dt);
            evals.push_back(it->second.display_name + ": " + format_number(dt, 1) + " s");
        }
    }
    if (labels.empty()) return std::nullopt;
    const double total = std::accumulate(durations.begin(), durations.end(), 0.0);
    evals.insert(evals.begin(), "total pipeline time: " + format_number(total, 1) + " s");
    std::vector<ChartBlock> charts = {{
        svg_bar_horizontal(labels, durations, "Pipeline phase durations", "seconds"),
        explain_panel(
            "Pipeline-Laufzeit pro Phase",
            {
                "Jeder Balken zeigt die gemessene Netto-Dauer einer Pipeline-Phase zwischen <code>phase_start</code> und <code>phase_end</code>.",
                "Der Plot beantwortet primär die Frage, <em>wo</em> die Laufzeit verbrannt wird: I/O, Registrierung, lokale Metriken, Rekonstruktion oder nachgelagerte Korrekturen."
            },
            {
                "<span class=\"good\">Unauffälliger Befund:</span> Die meiste Laufzeit liegt in fachlich erwartbaren Phasen wie Registrierung, lokalen Metriken oder Rekonstruktion; diese Schritte skalieren direkt mit Frameanzahl, Bildgröße und Tilezahl.",
                "<span class=\"neutral\">Normaler Befund:</span> Einzelne lange Balken sind plausibel, wenn viele Frames, große Bilder oder viele Tiles verarbeitet wurden; entscheidend ist, ob die lange Phase zum Datenumfang passt.",
                "<span class=\"bad\">Prüfbedarf:</span> Unverhältnismäßig lange Scan-, Load- oder PREWARP-Phasen sprechen eher für I/O-, Pfad- oder Datenlayout-Probleme als für normalen Bildinhalt.",
                "Starke Unterschiede zwischen nominal ähnlichen Phasen können auf Fallbacks, Wiederholungen oder instabile Eingangsdaten hinweisen."
            }
        )
    }};
    return ReportSection{"Pipeline Timeline", make_card_html("Phase durations", charts, evals, "ok")};
}

/// @brief Generates frame usage.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_frame_usage(const std::vector<json>& events, const json& synthetic) {
    json run_start = json::object();
    json scan_end = json::object();
    json reg_end = json::object();
    json synth_end = json::object();
    for (const auto& ev : events) {
        const auto type = json_string_or(ev, "type", "");
        const auto phase = phase_name_from_event(ev);
        if (type == "run_start") run_start = ev;
        if (type == "phase_end" && phase == "SCAN_INPUT") scan_end = ev;
        if (type == "phase_end" && phase == "REGISTRATION") reg_end = ev;
        if (type == "phase_end" && phase == "SYNTHETIC_FRAMES") synth_end = ev;
    }

    const int frames_discovered = static_cast<int>(json_number_or(run_start, "frames_discovered", 0.0));
    const json linearity = scan_end.is_object() && scan_end.contains("linearity") && scan_end["linearity"].is_object()
        ? scan_end["linearity"] : json::object();
    const bool linearity_enabled = json_bool_or(linearity, "enabled", false);
    const int linearity_failed = static_cast<int>(json_number_or(linearity, "failed_frames", 0.0));
    const std::string linearity_action = json_string_or(linearity, "action", "");
    const int frames_after_scan = static_cast<int>(json_number_or(scan_end, "frames_scanned", frames_discovered));

    int frames_usable_reg = static_cast<int>(json_number_or(reg_end, "frames_usable", 0.0));
    const int reg_rejected = static_cast<int>(json_number_or(reg_end, "reg_rejected_frames", 0.0));
    const int frames_cc_negative = static_cast<int>(json_number_or(reg_end, "frames_cc_negative", 0.0));
    if (frames_usable_reg == 0) {
        const int num_frames = static_cast<int>(json_number_or(reg_end, "num_frames", 0.0));
        if (num_frames > 0) frames_usable_reg = std::max(0, num_frames - reg_rejected);
    }
    const int frames_excluded_negative = static_cast<int>(json_number_or(reg_end, "frames_excluded_negative", frames_cc_negative));
    const int frames_excluded_identity = std::max(0, reg_rejected - frames_excluded_negative);
    const int num_synthetic = static_cast<int>(json_number_or(synth_end, "num_synthetic", 0.0));
    const int synth_frames_max = static_cast<int>(json_number_or(synthetic, "frames_max", 0.0));
    const std::string synth_status = json_string_or(synth_end, "status", "");

    struct Stage {
        std::string label;
        double count;
        std::string reason;
    };
    std::vector<Stage> stages;
    if (frames_discovered > 0) stages.push_back({"Discovered", static_cast<double>(frames_discovered), "Input scan"});
    if (linearity_enabled) {
        if (linearity_action == "removed" && linearity_failed > 0) {
            stages.push_back({"After linearity", static_cast<double>(frames_after_scan),
                              std::to_string(linearity_failed) + " removed"});
        } else {
            stages.push_back({"After linearity", static_cast<double>(frames_after_scan), "Linearity checked"});
        }
    }
    if (frames_usable_reg > 0) {
        std::vector<std::string> reasons;
        if (frames_excluded_identity > 0) reasons.push_back(std::to_string(frames_excluded_identity) + " identity");
        if (frames_excluded_negative > 0) reasons.push_back(std::to_string(frames_excluded_negative) + " negative CC");
        const std::string reason = reasons.empty()
            ? "All usable"
            : reasons.front() + (reasons.size() > 1 ? std::string(", ") + reasons.back() : std::string());
        stages.push_back({"Registered usable", static_cast<double>(frames_usable_reg), reason});
    }
    if (stages.size() < 2) return std::nullopt;

    std::vector<std::string> evals;
    std::vector<std::string> labels;
    std::vector<double> counts;
    std::vector<std::string> colors;
    const double max_count = stages.front().count > 0.0 ? stages.front().count : 1.0;
    for (const auto& stage : stages) {
        labels.push_back(stage.label);
        counts.push_back(stage.count);
        const double retention = stage.count / max_count;
        colors.push_back(retention > 0.8 ? "#4ade80" : retention > 0.5 ? "#fbbf24" : "#f87171");
        evals.push_back(stage.label + ": " + format_number(stage.count, 0) + " (" + stage.reason + ")");
    }
    if (num_synthetic > 0) {
        std::ostringstream line;
        line << "synthetic frames: " << num_synthetic << " from " << format_number(stages.back().count, 0) << " source frames";
        if (synth_frames_max > 0) line << " (frames_max=" << synth_frames_max << ")";
        evals.push_back(line.str());
    } else if (synth_status == "skipped") {
        evals.push_back("synthetic frames: skipped");
    }

    std::vector<std::string> loss_labels;
    std::vector<double> loss_values;
    std::vector<std::string> loss_colors;
    if (linearity_enabled && linearity_action == "removed" && linearity_failed > 0) {
        loss_labels.push_back("Linearity");
        loss_values.push_back(static_cast<double>(linearity_failed));
        loss_colors.push_back("#f87171");
    }
    if (frames_excluded_identity > 0) {
        loss_labels.push_back("Identity fallback");
        loss_values.push_back(static_cast<double>(frames_excluded_identity));
        loss_colors.push_back("#fbbf24");
    }
    if (frames_excluded_negative > 0) {
        loss_labels.push_back("Negative CC");
        loss_values.push_back(static_cast<double>(frames_excluded_negative));
        loss_colors.push_back("#f472b6");
    }
    if (frames_usable_reg > 0) {
        loss_labels.push_back("Used");
        loss_values.push_back(static_cast<double>(frames_usable_reg));
        loss_colors.push_back("#4ade80");
    }

    std::vector<ChartBlock> charts = {{
        svg_bar_horizontal(labels, counts, "Frame usage funnel", "frames", colors),
        explain_panel(
            "Frame-Funnel",
            {
                "Der Funnel zeigt, wie viele Frames nach den wichtigsten Akzeptanzstufen übrig bleiben: entdeckt, nach Scan/Linearity, nach Registrierung und ggf. vor der Synthetik.",
                "Damit wird sofort sichtbar, ob Verluste früh im Intake oder erst später durch Registrierung und Qualitätsfilter entstehen."
            },
            {
                "<span class=\"good\">Unauffälliger Befund:</span> Die Framezahl fällt von links nach rechts nur moderat ab; besonders zwischen Scan und Registrierung bleibt der größte Teil des Materials nutzbar.",
                "<span class=\"neutral\">Normaler Befund:</span> Kleine Verluste durch Linearity-Checks oder CC-basierte Ablehnung sind erwartbar, weil die Pipeline einzelne problematische Frames bewusst aussortiert.",
                "<span class=\"bad\">Prüfbedarf:</span> Ein starker Einbruch vor oder nach der Registrierung bedeutet, dass ein großer Teil des Datensatzes geometrisch oder photometrisch nicht robust verwertbar war.",
                "Die Balkenfarbe codiert die Retention relativ zur Anfangsmenge: grün = hoch, gelb = merklicher Verlust, rot = kritischer Verlust."
            }
        )
    }};
    if (!loss_labels.empty()) {
        charts.push_back({
            svg_pie(loss_labels, loss_values, loss_colors, "Frame loss breakdown"),
            explain_panel(
                "Verlustursachen",
                {
                    "Das Kreisdiagramm zerlegt alle erkannten Frames in effektiv genutztes Material und die wichtigsten Verlustursachen, etwa Linearity-Ausschluss, Identity-Fallbacks oder negative Registrierungs-CC-Werte.",
                    "Damit wird nicht nur sichtbar, <em>wie viel</em> Material verloren ging, sondern auch, <em>warum</em> es aus der weiteren Verarbeitung herausgefallen ist."
                },
                {
                    "<span class=\"good\">Unauffälliger Befund:</span> Ein großer grüner Anteil bedeutet, dass viele Frames effektiv genutzt wurden und die späteren Pipeline-Stufen statistisch gut abgestützt sind.",
                    "<span class=\"neutral\">Normaler Befund:</span> Kleinere gelbe Segmente sind typische Verluste durch vorsichtige Linearity-Prüfung, Registrierung oder konservative Qualitätsfilter.",
                    "<span class=\"bad\">Prüfbedarf:</span> Große nicht-grüne Segmente weisen auf Akquisitionsprobleme, Wolken, Drift, starke Transparenzwechsel oder eine schwache Registrierbarkeit des Materials hin."
                }
            )
        });
    }
    return ReportSection{"Frame Usage", make_card_html("Frame retention", charts, evals, infer_status(evals))};
}

/// @brief Generates normalization.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_normalization(const json& norm) {
    if (!norm.is_object() || norm.empty()) return std::nullopt;
    const std::string mode = json_string_or(norm, "mode", "MONO");
    std::vector<std::string> evals = {"mode: " + mode};
    std::vector<ChartBlock> charts;

    const auto b_mono = json_double_array(norm.value("B_mono", json::array()));
    const auto b_r = json_double_array(norm.value("B_r", json::array()));
    const auto b_g = json_double_array(norm.value("B_g", json::array()));
    const auto b_b = json_double_array(norm.value("B_b", json::array()));

    if (mode == "OSC" && (!b_r.empty() || !b_g.empty() || !b_b.empty())) {
        charts.push_back({
            svg_multi_timeseries({{"R", b_r}, {"G", b_g}, {"B", b_b}}, "Per-channel background level", "background"),
            explain_panel(
                "Hintergrund pro OSC-Kanal",
                {
                    "Der Plot zeigt den geschätzten Himmelshintergrund pro Frame getrennt nach R-, G- und B-Kanal.",
                    "Er ist wichtig, um Transparenzwechsel, Mondlicht, Farbgradienten und kanalabhängige Hintergrundverschiebungen früh zu erkennen."
                },
                {
                    "<span class=\"good\">Unauffälliger Befund:</span> Die R-, G- und B-Kanäle bleiben über die Session relativ stabil und zeigen eine ähnliche zeitliche Form; die spätere Normalisierung muss dann nur moderate Korrekturen leisten.",
                    "<span class=\"neutral\">Normaler Befund:</span> Driften alle Kanäle gemeinsam und langsam in dieselbe Richtung, passt das häufig zu langsam veränderlichen Aufnahmebedingungen wie zunehmender Luftfeuchte, sinkender Objekt-Höhe, Mondlicht oder Himmelsaufhellung.",
                    "<span class=\"bad\">Prüfbedarf:</span> Harte Sprünge, voneinander entkoppelte Kanäle oder große Offset-Unterschiede sprechen für Wolken, Gradienten, Farbverschiebungen oder einzelne problematische Aufnahmeabschnitte.",
                    "Die Normalisierung muss genau diese Unterschiede später kompensieren; je stärker die Schwankung, desto wichtiger ist der Schritt."
                }
            )
        });
        for (const auto& item : std::vector<std::pair<std::string, std::vector<double>>>{{"R", b_r}, {"G", b_g}, {"B", b_b}}) {
            const auto s = basic_stats(item.second);
            if (s.n > 0) {
                evals.push_back(item.first + ": median=" + format_number(s.median, 4) +
                                ", std=" + format_number(s.std_dev, 4) +
                                ", range=[" + format_number(s.min, 4) + ", " + format_number(s.max, 4) + "]");
            }
        }
    } else if (!b_mono.empty()) {
        charts.push_back({
            svg_timeseries(b_mono, "Background level", "background"),
            explain_panel(
                "Mono-Hintergrund",
                {
                    "Dies ist der globale Hintergrundschätzer pro Frame für MONO-Daten.",
                    "Der Verlauf zeigt, ob der Datensatz über die Session hinweg photometrisch stabil geblieben ist."
                },
                {
                    "<span class=\"good\">Unauffälliger Befund:</span> Der Hintergrund bleibt nahe am Median und schwankt nur kleinräumig; der Stack wird dadurch photometrisch gleichmäßig gestützt.",
                    "<span class=\"neutral\">Normaler Befund:</span> Eine langsame Drift ist oft noch gut handhabbar, weil sie durch Hintergrundnormalisierung und Gewichtung abgefedert werden kann.",
                    "<span class=\"bad\">Prüfbedarf:</span> Harte Peaks oder Einbrüche deuten typischerweise auf Wolken, wechselnde Lichtverschmutzung, Tau oder andere Transparenzsprünge hin."
                }
            )
        });
        const auto s = basic_stats(b_mono);
        if (s.n > 0) {
            evals.push_back("mono: median=" + format_number(s.median, 4) +
                            ", std=" + format_number(s.std_dev, 4));
        }
    }

    if (charts.empty()) return std::nullopt;
    return ReportSection{"Normalization", make_card_html("Background levels", charts, evals, infer_status(evals))};
}

/// @brief Generates global metrics.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_global_metrics(const json& gm) {
    if (!gm.is_object() || !gm.contains("metrics") || !gm["metrics"].is_array() || gm["metrics"].empty()) return std::nullopt;

    std::vector<double> bg, noise, grad, gw, fwhm, wfwhm, roundness, star_count;
    for (const auto& m : gm["metrics"]) {
        bg.push_back(json_number_or(m, "background", NAN));
        noise.push_back(json_number_or(m, "noise", NAN));
        grad.push_back(json_number_or(m, "gradient_energy", NAN));
        gw.push_back(json_number_or(m, "global_weight", NAN));
        fwhm.push_back(json_number_or(m, "fwhm", NAN));
        wfwhm.push_back(json_number_or(m, "wfwhm", NAN));
        roundness.push_back(json_number_or(m, "roundness", NAN));
        star_count.push_back(json_number_or(m, "star_count", NAN));
    }

    std::vector<ChartBlock> charts = {
        {svg_timeseries(bg, "Frame background level", "background"), explain_panel(
            "Frame-Hintergrund",
            {
                "Zeigt den mittleren Hintergrund jedes Frames nach dem Scan-/Metrikschritt.",
                "Der Plot beantwortet, wie stabil der Himmelshintergrund über die Session war und welche Frames photometrisch auffällig sind."
            },
            {
                "<span class=\"good\">Gut:</span> Gleichmäßiger Verlauf nahe am Median.",
                "<span class=\"bad\">Auffällig:</span> Peaks oder starke Drift sprechen für Wolken, Tau, Mondlicht oder Lichtverschmutzungswechsel.",
                "Frames mit problematischem Hintergrund werden im globalen Gewicht später deutlich abgestraft."
            }
        )},
        {svg_timeseries(noise, "Frame noise level", "noise", "#f87171"), explain_panel(
            "Rauschpegel pro Frame",
            {
                "Dies ist ein robuster Rauschschätzer für jedes Einzelbild.",
                "Hohe Werte bedeuten nicht automatisch schlechte Sterne, aber ein schlechteres Signal-Rausch-Verhältnis und damit geringeren Nutzwert im Stack."
            },
            {
                "<span class=\"good\">Gut:</span> Niedriger und relativ stabiler Rauschlevel.",
                "<span class=\"neutral\">Neutral:</span> Etwas Streuung ist bei realen Bedingungen normal.",
                "<span class=\"bad\">Auffällig:</span> Einzelne hohe Spitzen deuten oft auf Wolken, instabile Transparenz oder sehr schwache Frames."
            }
        )},
        {svg_timeseries(grad, "Frame gradient energy", "gradient", "#4ade80"), explain_panel(
            "Gradientenenergie",
            {
                "Die Gradientenenergie misst, wie viel Struktur, Kantenenergie und lokaler Detailkontrast im Frame steckt.",
                "Sie reagiert sowohl auf echte Schärfe als auch auf störende Strukturen wie Wolkenkanten und ist deshalb bewusst nur als Einzelmetrik zu lesen, nicht als Gesamtnote."
            },
            {
                "<span class=\"good\">Gut:</span> Hohe Werte sind nur dann positiv, wenn gleichzeitig FWHM, Hintergrund und Rauschen ebenfalls plausibel gut aussehen.",
                "<span class=\"neutral\">Neutral:</span> Ein isoliert hoher Gradient ist noch kein Qualitätsbeweis, sondern nur ein Hinweis auf viel lokale Struktur.",
                "<span class=\"bad\">Auffällig:</span> Hohe Gradientenspitzen bei gleichzeitig schlechtem Hintergrund oder hohem Rauschen kommen oft von Wolken, Gradienten oder unruhigen Strukturen und nicht von echter Bildqualität."
            }
        )},
        {svg_timeseries(gw, "Global frame weight", "weight", "#fbbf24"), explain_panel(
            "Globales Frame-Gewicht",
            {
                "Dies ist das kombinierte globale Qualitätsgewicht, mit dem der Frame später in der Pipeline bewertet wird.",
                "Es verdichtet Hintergrund, Rauschen und Strukturinformation zu einer einzelnen Nutzbarkeitskennzahl und ist damit die bereinigte Gesamtaussage über die Verwendbarkeit eines Frames."
            },
            {
                "<span class=\"good\">Gut:</span> Hohe Werte markieren Frames, bei denen die Gesamtkombination aus niedrigem Hintergrund, niedrigem Rauschen und brauchbarer Struktur stimmt.",
                "<span class=\"neutral\">Neutral:</span> Eine gewisse Streuung ist bei wechselnden Bedingungen normal, weil nicht jeder Frame gleich gut ist.",
                "<span class=\"bad\">Auffällig:</span> Viele sehr niedrige Gewichte bedeuten, dass ein relevanter Teil des Datensatzes in der Gesamtschau nur schwach oder problematisch ist, selbst wenn einzelne Einzelmetriken zeitweise gut aussahen."
            }
        )},
        {svg_histogram(gw, "Global weight distribution", "weight", "#fbbf24"), explain_panel(
            "Verteilung der globalen Gewichte",
            {
                "Das Histogramm zeigt, wie die Einzelgewichte über alle Frames verteilt sind.",
                "Es hilft zu unterscheiden, ob der Datensatz homogen gut, zweigeteilt oder breit streuend problematisch ist."
            },
            {
                "<span class=\"good\">Gut:</span> Ein kompakter Peak im oberen Bereich bedeutet gleichmäßig gutes Material.",
                "<span class=\"neutral\">Neutral:</span> Eine zweigipflige Verteilung weist oft auf gute und schlechte Subsets innerhalb derselben Session hin.",
                "<span class=\"bad\">Auffällig:</span> Eine breite linke Flanke oder viele sehr kleine Gewichte sprechen für viele schwache Frames."
            }
        )},
        {svg_timeseries(fwhm, "FWHM per frame", "FWHM (px)", "#c084fc"), explain_panel(
            "FWHM pro Frame",
            {
                "Die FWHM beschreibt die Sternbreite in Pixeln und ist einer der direktesten Schärfe-Indikatoren im Datensatz.",
                "Niedrige FWHM bedeutet kompakte Sterne, hohe FWHM steht für Seeing-, Fokus- oder Nachführprobleme."
            },
            {
                "<span class=\"good\">Gut:</span> Niedrige und möglichst stabile Werte.",
                "<span class=\"neutral\">Neutral:</span> Sanfte Trends können echtes Seeing-Drift über die Nacht zeigen.",
                "<span class=\"bad\">Auffällig:</span> Hohe Peaks oder breite Streuung bedeuten deutliche Unschärfephasen."
            }
        )},
        {svg_timeseries(roundness, "Roundness per frame", "roundness", "#22d3ee"), explain_panel(
            "Sternrundheit",
            {
                "Die Rundheit beschreibt, wie kreisförmig die Sternprofile sind; Werte nahe 1 sind ideal.",
                "Sie reagiert besonders auf Trackingfehler, Wind, Verkippung oder systematische Sternelongation."
            },
            {
                "<span class=\"good\">Gut:</span> Werte nahe 1 mit geringer Streuung.",
                "<span class=\"neutral\">Neutral:</span> Leichte Abweichungen sind tolerierbar, wenn sie nicht systematisch wegdriften.",
                "<span class=\"bad\">Auffällig:</span> Niedrige oder stark schwankende Werte deuten auf elongierte Sterne und mechanische/geometrische Probleme hin."
            }
        )},
        {svg_timeseries(star_count, "Detected stars per frame", "stars", "#fde047"), explain_panel(
            "Erkannte Sterne",
            {
                "Der Plot zeigt, wie viele Sterne im jeweiligen Frame robust detektiert wurden.",
                "Er ist ein schneller Proxy für Transparenz, Fokus und Nutzbarkeit des Materials."
            },
            {
                "<span class=\"good\">Gut:</span> Hohe und stabile Sternanzahl.",
                "<span class=\"neutral\">Neutral:</span> Moderate Schwankungen je nach Feldinhalt oder Seeing sind normal.",
                "<span class=\"bad\">Auffällig:</span> Einbrüche sprechen oft für Wolken, Defokus, Tau oder starke Hintergrundprobleme."
            }
        )},
        {svg_scatter(fwhm, roundness, star_count, "FWHM vs roundness", "FWHM (px)", "roundness"), explain_panel(
            "FWHM gegen Rundheit",
            {
                "Jeder Punkt repräsentiert einen Frame. Die Position kombiniert zwei zentrale Sternqualitätsmetriken: Schärfe und Form.",
                "Die Farbskala folgt der Frame-Reihenfolge und macht dadurch auch zeitliche Drift sichtbar."
            },
            {
                "<span class=\"good\">Gut:</span> Ein kompakter Cluster bei niedriger FWHM und Rundheit nahe 1.",
                "<span class=\"neutral\">Neutral:</span> Mehrere Cluster können auf unterschiedliche Wetter- oder Fokusphasen hindeuten.",
                "<span class=\"bad\">Auffällig:</span> Ausreißer mit hoher FWHM und schlechter Rundheit sind die schlechtesten Kandidaten im Datensatz.",
                "Ein systematischer Farbverlauf der Punkte zeigt, ob sich die Qualität im Laufe der Session verbessert oder verschlechtert hat."
            }
        )},
    };

    std::vector<std::string> evals;
    evals.push_back("frames: " + std::to_string(gm["metrics"].size()));
    if (gm.contains("weights") && gm["weights"].is_object()) {
        const auto& w = gm["weights"];
        evals.push_back("weights: bg=" + json_string_or(w, "background", "?") +
                        ", noise=" + json_string_or(w, "noise", "?") +
                        ", grad=" + json_string_or(w, "gradient", "?"));
    }
    const auto s_w = basic_stats(gw);
    if (s_w.n > 0) {
        evals.push_back("G(f): median=" + format_number(s_w.median, 4) +
                        ", min=" + format_number(s_w.min, 4) +
                        ", max=" + format_number(s_w.max, 4));
        if (s_w.min > 0.0 && s_w.max / s_w.min > 50.0) evals.push_back("WARNING: extremely wide weight distribution");
    }
    const auto s_f = basic_stats(fwhm);
    if (s_f.n > 0) evals.push_back("FWHM: median=" + format_number(s_f.median, 2) + " px");
    const auto s_r = basic_stats(roundness);
    if (s_r.n > 0) {
        evals.push_back("roundness: median=" + format_number(s_r.median, 3));
        if (s_r.median < 0.7) evals.push_back("WARNING: low median roundness");
    }
    const auto s_s = basic_stats(star_count);
    if (s_s.n > 0) evals.push_back("star count: median=" + format_number(s_s.median, 0));

    return ReportSection{"Pipeline-wide Frame Metrics", make_card_html("Shared input-frame quality (all reconstruction methods)", charts, evals, infer_status(evals))};
}

/// @brief Generates tile grid.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_tile_grid(const json& tg) {
    if (!tg.is_object() || !tg.contains("tiles") || !tg["tiles"].is_array() || tg["tiles"].empty()) return std::nullopt;
    const int img_w = static_cast<int>(json_number_or(tg, "image_width", 0.0));
    const int img_h = static_cast<int>(json_number_or(tg, "image_height", 0.0));
    std::vector<std::string> evals = {
        "image: " + std::to_string(img_w) + "x" + std::to_string(img_h),
        "num_tiles: " + json_string_or(tg, "num_tiles", "?"),
        "tile_size: " + json_string_or(tg, "uniform_tile_size", json_string_or(tg, "seeing_tile_size", "?")),
        "seeing_fwhm_median: " + json_string_or(tg, "seeing_fwhm_median", "?"),
        "overlap_fraction: " + json_string_or(tg, "overlap_fraction", "?"),
        "stride_px: " + json_string_or(tg, "stride_px", "?"),
    };
    std::vector<ChartBlock> charts = {{
        svg_tile_overlay(tg["tiles"], img_w, img_h, "Tile grid overlay"),
        explain_panel(
            "Tile-Raster",
            {
                "Die Grafik zeigt die reale Zerlegung des Bildes in überlappende Tiles.",
                "Dieses Raster ist die Grundlage für lokale Metriken, tile-spezifische Gewichte und die spätere Rekonstruktion."
            },
            {
                "Mehr Tiles bedeuten feinere lokale Steuerung, aber auch mehr Rechen- und Verwaltungsaufwand.",
                "Die Überlappung ist notwendig, damit beim Zusammenbau keine harten Kachelgrenzen sichtbar bleiben.",
                "Ein plausibles Raster deckt das gesamte Bild homogen ab und zeigt keine offensichtlichen Lücken."
            }
        )
    }};
    return ReportSection{"Pipeline-wide Tile Grid", make_card_html("Shared geometry (not Classic reconstruction statistics)", charts, evals, "ok")};
}

/// @brief Generates registration.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_registration(const json& reg) {
    if (!reg.is_object() || !reg.contains("warps") || !reg["warps"].is_array() || reg["warps"].empty()) return std::nullopt;
    std::vector<double> ccs = json_double_array(reg.value("cc", json::array()));
    std::vector<double> tx, ty, rotations, scales;
    for (const auto& w : reg["warps"]) {
        const double tx_v = json_number_or(w, "tx", NAN);
        const double ty_v = json_number_or(w, "ty", NAN);
        const double a00 = json_number_or(w, "a00", 1.0);
        const double a01 = json_number_or(w, "a01", 0.0);
        tx.push_back(tx_v);
        ty.push_back(ty_v);
        rotations.push_back(std::atan2(a01, a00) * 180.0 / 3.14159265358979323846);
        scales.push_back(std::sqrt(a00 * a00 + a01 * a01));
    }

    std::vector<ChartBlock> charts = {
        {svg_scatter(tx, ty, ccs, "Translation scatter", "tx (px)", "ty (px)"), explain_panel(
            "Translations-Scatter",
            {
                "Jeder Punkt zeigt die erkannte Verschiebung eines Frames relativ zum Referenzframe in x- und y-Richtung.",
                "Die Farbskala transportiert die Reihenfolge der Frames und hilft, Driftmuster zu erkennen."
            },
            {
                "<span class=\"good\">Gut:</span> Kompakte Punktwolke ohne starke Ausreißer.",
                "<span class=\"neutral\">Neutral:</span> Eine langsame gerichtete Drift ist bei Session-Drift möglich.",
                "<span class=\"bad\">Auffällig:</span> Große Streuung oder isolierte Cluster sprechen für instabile Nachführung, Wind oder problematische Registrierung."
            }
        )},
        {svg_multi_timeseries({{"tx", tx}, {"ty", ty}}, "Translation over time", "shift (px)"), explain_panel(
            "Translation über die Zeit",
            {
                "Hier werden die Registrierungsverschiebungen als Zeitreihe getrennt für tx und ty gezeigt.",
                "Das macht sichtbar, ob die Montierung gleichmäßig driftet oder ob es abrupte Sprünge gab."
            },
            {
                "<span class=\"good\">Gut:</span> Ruhige, langsam verlaufende Kurven.",
                "<span class=\"bad\">Auffällig:</span> Stufen, Sprünge oder chaotische Zickzack-Muster deuten auf Tracking- oder Matching-Probleme hin."
            }
        )},
        {svg_histogram(ccs, "Registration CC distribution", "CC", "#4ade80"), explain_panel(
            "Registrierungs-CC",
            {
                "Das Histogramm zeigt die Verteilung des Registrierungs-Korrelationskoeffizienten über alle Einzelbilder.",
                "Der CC-Wert beschreibt, wie zuverlässig ein Einzelbild geometrisch und photometrisch zum Referenzbild passt."
            },
            {
                "<span class=\"good\">Unauffälliger Befund:</span> Eine enge Häufung bei hohen CC-Werten bedeutet, dass die meisten Einzelbilder stabil und eindeutig auf das Referenzbild registriert werden konnten.",
                "<span class=\"neutral\">Normaler Befund:</span> Ein kleiner linker Ausläufer ist bei schwierigen Sessions plausibel, solange nur wenige Einzelbilder deutlich schwächer korrelieren.",
                "<span class=\"bad\">Prüfbedarf:</span> Viele niedrige CC-Werte bedeuten, dass zahlreiche Einzelbilder geometrisch oder photometrisch nur unsicher zum Referenzbild passen; typische Ursachen sind Wolken, Drift, schwache Sterne, Fokusänderungen oder wechselnde Transparenz."
            }
        )},
        {svg_timeseries(rotations, "Rotation angle", "deg", "#f87171"), explain_panel(
            "Rotation pro Frame",
            {
                "Zeigt den relativen Rotationswinkel jedes Frames gegenüber dem Referenzframe.",
                "Besonders relevant bei Feldrotation, Alt/Az-Aufnahmen oder langen Sessions mit Rotationsanteil."
            },
            {
                "<span class=\"good\">Gut:</span> Kleiner und glatter Verlauf, wenn kaum Rotation erwartet wird.",
                "<span class=\"neutral\">Neutral:</span> Gleichmaessige monotone Rotation kann physikalisch normal sein.",
                "<span class=\"bad\">Auffällig:</span> Sprunghafte Richtungswechsel oder starke Ausreißer sprechen eher für Registrierungsinstabilität als für echte Geometrie."
            }
        )},
        {svg_timeseries(scales, "Scale factor", "scale", "#fbbf24"), explain_panel(
            "Skalenfaktor",
            {
                "Der Plot zeigt, ob Frames relativ zum Referenzframe vergrößert oder verkleinert werden mussten.",
                "Skalendrift ist oft ein Hinweis auf Fokuswanderung, Atmosphaerik oder inkonsistente Geometrie."
            },
            {
                "<span class=\"good\">Gut:</span> Werte nahe 1 mit geringer Streuung.",
                "<span class=\"bad\">Auffällig:</span> Systematische Drift oder starke Ausreißer deuten auf optische/geometrische Instabilität hin."
            }
        )},
    };

    std::vector<std::string> evals = {
        "frames: " + json_string_or(reg, "num_frames", "?") +
        ", scale: " + json_string_or(reg, "scale", "?") +
        ", ref_frame: " + json_string_or(reg, "ref_frame", "?")
    };
    const auto s_cc = basic_stats(ccs);
    if (s_cc.n > 0) {
        evals.push_back("CC: median=" + format_number(s_cc.median, 4) +
                        ", min=" + format_number(s_cc.min, 4) +
                        ", max=" + format_number(s_cc.max, 4));
        int bad = 0;
        for (double v : ccs) if (std::isfinite(v) && v < 0.5) ++bad;
        if (bad > 0) evals.push_back("WARNING: " + std::to_string(bad) + " frames with CC < 0.5");
    }
    const auto s_tx = basic_stats(tx);
    const auto s_ty = basic_stats(ty);
    if (s_tx.n > 0) evals.push_back("tx range=[" + format_number(s_tx.min, 2) + ", " + format_number(s_tx.max, 2) + "]");
    if (s_ty.n > 0) evals.push_back("ty range=[" + format_number(s_ty.min, 2) + ", " + format_number(s_ty.max, 2) + "]");

    return ReportSection{"Global Registration", make_card_html("Frame alignment", charts, evals, infer_status(evals))};
}

/// @brief Generates local metrics.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_local_metrics(const json& lm, const json& tg) {
    if (!lm.is_object() || !lm.contains("tile_metrics") || !lm["tile_metrics"].is_array() || lm["tile_metrics"].empty()) return std::nullopt;
    const int n_frames = static_cast<int>(json_number_or(lm, "num_frames", 0.0));
    const int n_tiles = static_cast<int>(json_number_or(lm, "num_tiles", 0.0));
    if (n_tiles <= 0) return std::nullopt;

    std::vector<std::vector<double>> all_fwhm(static_cast<size_t>(n_tiles));
    std::vector<std::vector<double>> all_quality(static_cast<size_t>(n_tiles));
    std::vector<std::vector<double>> all_weight(static_cast<size_t>(n_tiles));
    std::vector<std::vector<double>> all_stars(static_cast<size_t>(n_tiles));
    std::vector<double> per_frame_quality;
    std::vector<double> per_frame_weight;
    std::vector<double> tile_type_map(static_cast<size_t>(n_tiles), 0.0);
    bool have_tile_types = false;

    size_t frame_index = 0;
    for (const auto& frame_tiles : lm["tile_metrics"]) {
        if (!frame_tiles.is_array()) continue;
        std::vector<double> frame_q;
        std::vector<double> frame_w;
        size_t ti = 0;
        for (const auto& tm : frame_tiles) {
            if (ti >= static_cast<size_t>(n_tiles) || !tm.is_object()) break;
            const double fwhm = json_number_or(tm, "fwhm", NAN);
            const double quality = json_number_or(tm, "quality_score", NAN);
            const double weight = json_number_or(tm, "local_weight", NAN);
            const double stars = json_number_or(tm, "star_count", NAN);
            if (std::isfinite(fwhm)) all_fwhm[ti].push_back(fwhm);
            if (std::isfinite(quality)) {
                all_quality[ti].push_back(quality);
                frame_q.push_back(quality);
            }
            if (std::isfinite(weight)) {
                all_weight[ti].push_back(weight);
                frame_w.push_back(weight);
            }
            if (std::isfinite(stars)) all_stars[ti].push_back(stars);
            if (frame_index == 0) {
                const auto type = json_string_or(tm, "tile_type", "");
                if (!type.empty()) {
                    have_tile_types = true;
                    tile_type_map[ti] = type == "STAR" ? 1.0 : 0.0;
                }
            }
            ++ti;
        }
        per_frame_quality.push_back(frame_q.empty() ? 0.0 : std::accumulate(frame_q.begin(), frame_q.end(), 0.0) / static_cast<double>(frame_q.size()));
        per_frame_weight.push_back(frame_w.empty() ? 0.0 : std::accumulate(frame_w.begin(), frame_w.end(), 0.0) / static_cast<double>(frame_w.size()));
        ++frame_index;
    }

    auto mean_of = [](const std::vector<std::vector<double>>& values) {
        std::vector<double> out(values.size(), 0.0);
        for (size_t i = 0; i < values.size(); ++i) {
            if (!values[i].empty()) {
                out[i] = std::accumulate(values[i].begin(), values[i].end(), 0.0) / static_cast<double>(values[i].size());
            }
        }
        return out;
    };

    const auto mean_fwhm = mean_of(all_fwhm);
    const auto mean_quality = mean_of(all_quality);
    const auto mean_weight = mean_of(all_weight);
    const auto mean_stars = mean_of(all_stars);

    const int img_w = static_cast<int>(json_number_or(tg, "image_width", 0.0));
    const int img_h = static_cast<int>(json_number_or(tg, "image_height", 0.0));
    const json tiles = tg.contains("tiles") ? tg["tiles"] : json::array();

    std::vector<ChartBlock> charts;
    if (tiles.is_array() && !tiles.empty() && img_w > 0 && img_h > 0) {
        charts.push_back({svg_spatial_tile_heatmap(tiles, mean_fwhm, img_w, img_h, "Mean FWHM per tile", "FWHM (px)", "inferno"),
                          explain_panel("Mittlere FWHM pro Tile",
                                        {"Diese Heatmap zeigt, in welchen Bildregionen die Sterne im Mittel schärfer oder unschärfer sind.",
                                         "Sie macht räumlich sichtbar, ob das Bildfeld homogen fokussiert ist oder ob Rand-/Eckenprobleme vorliegen."},
                                        {"<span class=\"good\">Gut:</span> Homogene Verteilung ohne starke Hotspots.",
                                         "<span class=\"bad\">Auffällig:</span> Lokale Inseln mit hoher FWHM deuten auf Feldkrümmung, Tilt oder ortsabhängige Bildqualitätsprobleme hin."})});
        charts.push_back({svg_spatial_tile_heatmap(tiles, mean_quality, img_w, img_h, "Mean quality score per tile", "quality", "viridis"),
                          explain_panel("Mittlerer Qualitätsscore pro Tile",
                                        {"Der Score aggregiert lokale Bildqualität über alle Frames für jede Bildregion.",
                                         "Die Karte zeigt damit, wo die Pipeline im Feld konsistent gutes oder schwaches Material gesehen hat."},
                                        {"<span class=\"good\">Gut:</span> Hohe und relativ gleichmäßige Qualität über das Feld.",
                                         "<span class=\"bad\">Auffällig:</span> Deutliche Flecken oder Gradienten zeigen räumlich ungleichmäßige Datengüte."})});
        charts.push_back({svg_spatial_tile_heatmap(tiles, mean_weight, img_w, img_h, "Mean local weight per tile", "weight", "plasma"),
                          explain_panel("Mittleres lokales Gewicht",
                                        {"Diese Heatmap zeigt, welche Tiles im Mittel stark bzw. schwach in die lokale Rekonstruktion eingehen.",
                                         "Sie ist besonders wichtig, um zu sehen, ob einzelne Bildregionen systematisch untergewichtet werden."},
                                        {"<span class=\"good\">Gut:</span> Plausible Unterschiede ohne extreme Null-/Hotspot-Zonen.",
                                         "<span class=\"bad\">Auffällig:</span> Sehr schwache Regionen markieren Feldbereiche mit dauerhaft schlechter lokaler Nutzbarkeit."})});
        charts.push_back({svg_spatial_tile_heatmap(tiles, mean_stars, img_w, img_h, "Mean stars per tile", "stars", "YlGnBu"),
                          explain_panel("Mittlere Sternanzahl pro Tile",
                                        {"Hier wird sichtbar, welche Tiles im Mittel viele bzw. wenige detektierbare Sterne enthalten.",
                                         "Das ist sowohl feldabhängig als auch qualitätsabhängig und erklärt Unterschiede in lokaler Stabilität."},
                                        {"<span class=\"good\">Gut:</span> Sternreiche Regionen liefern robuste lokale Metriken.",
                                         "<span class=\"neutral\">Neutral:</span> Sternarme Hintergrund- oder Nebelbereiche sind nicht automatisch schlecht, aber statistisch schwächer abgestützt."})});
        if (have_tile_types) {
            charts.push_back({svg_spatial_tile_heatmap(tiles, tile_type_map, img_w, img_h, "Tile type map", "STAR=1", "viridis", true),
                              explain_panel("Tile-Typ-Karte",
                                            {"Die Karte visualisiert die Tile-Klassifikation der lokalen Metrikstufe, z. B. sterngetrieben versus struktur-/hintergrunddominiert.",
                                             "Sie hilft zu verstehen, warum verschiedene Regionen spaeter unterschiedlich gewichtet oder verarbeitet werden."},
                                            {"Helle Tiles markieren typischerweise Stern-/Strukturmodus.",
                                             "Ein plausibles Muster folgt grob dem Bildinhalt; chaotische Klassifikation kann auf instabile lokale Metriken hindeuten."})});
        }
    }
    charts.push_back({
        svg_multi_timeseries({{"mean quality", per_frame_quality}, {"mean weight", per_frame_weight}},
                             "Per-frame tile quality and weight", "value"),
        explain_panel("Frame-Mittel über alle Tiles",
                      {"Die beiden Kurven mitteln lokale Qualität und lokales Gewicht pro Frame über das gesamte Feld.",
                       "Damit wird sichtbar, wie sich die lokale Feldqualität zeitlich entwickelt, ohne einzelne Tiles isoliert betrachten zu müssen."},
                      {"<span class=\"good\">Gut:</span> Beide Kurven bleiben relativ stabil und folgen plausibel dem Sessionverlauf.",
                       "<span class=\"bad\">Auffällig:</span> Starke Einbrüche markieren Frames, in denen die lokale Bildqualität breitflächig kollabiert ist."})
    });

    std::vector<std::string> evals = {
        "frames: " + std::to_string(n_frames) + ", tiles: " + std::to_string(n_tiles)
    };
    const auto s_f = basic_stats(mean_fwhm);
    if (s_f.n > 0) evals.push_back("mean FWHM: median=" + format_number(s_f.median, 3));
    const auto s_w = basic_stats(mean_weight);
    if (s_w.n > 0) evals.push_back("mean weight: median=" + format_number(s_w.median, 3));
    const auto s_s = basic_stats(mean_stars);
    if (s_s.n > 0) evals.push_back("mean star count: median=" + format_number(s_s.median, 1));
    if (have_tile_types) {
        int star_tiles = 0;
        for (double v : tile_type_map) if (v > 0.5) ++star_tiles;
        evals.push_back("STAR tiles: " + std::to_string(star_tiles) + ", STRUCTURE tiles: " +
                        std::to_string(std::max(0, n_tiles - star_tiles)));
    }

    return ReportSection{"Local Metrics", make_card_html("Per-tile quality", charts, evals, infer_status(evals))};
}

/// @brief Generates reconstruction.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_reconstruction(const json& recon, const json& tg) {
    if (!recon.is_object()) return std::nullopt;
    const auto valid_counts = json_double_array(recon.value("tile_valid_counts", json::array()));
    const auto mean_cc = json_double_array(recon.value("tile_mean_correlations", json::array()));
    const auto post_bg = json_double_array(recon.value("tile_post_background", json::array()));
    const auto post_contrast = json_double_array(recon.value("tile_post_contrast", json::array()));
    const auto post_snr = json_double_array(recon.value("tile_post_snr_proxy", json::array()));
    if (valid_counts.empty() && mean_cc.empty() && post_snr.empty()) return std::nullopt;

    const int img_w = static_cast<int>(json_number_or(tg, "image_width", 0.0));
    const int img_h = static_cast<int>(json_number_or(tg, "image_height", 0.0));
    const json tiles = tg.contains("tiles") ? tg["tiles"] : json::array();

    std::vector<ChartBlock> charts;
    if (tiles.is_array() && !tiles.empty() && img_w > 0 && img_h > 0) {
        if (!valid_counts.empty()) charts.push_back({svg_spatial_tile_heatmap(tiles, valid_counts, img_w, img_h, "Valid frames per tile", "frames", "YlGn"),
                                                     explain_panel("Gueltige Frames pro Tile",
                                                                   {"Diese Karte zeigt, wie viele Einzelbilder je Tile nach allen relevanten Filtern effektiv in die Rekonstruktion eingegangen sind.",
                                                                    "Sie macht sichtbar, wo die Pipeline lokal statistisch stark oder dünn abgestützt arbeitet.",
                                                                    "Wenn die Karte nahezu einfarbig ist, bedeutet das hier meist tatsächlich eine gleichmäßige Abdeckung und nicht automatisch ein Problem im Rendering."},
                                                                   {"<span class=\"good\">Gut:</span> Möglichst homogene und ausreichend hohe Counts.",
                                                                    "<span class=\"bad\">Auffällig:</span> Tiles mit sehr niedrigen Counts sind lokal fragiler und können Bias oder Artefaktrisiko tragen."})});
        if (!mean_cc.empty()) charts.push_back({svg_spatial_tile_heatmap(tiles, mean_cc, img_w, img_h, "Mean correlation per tile", "CC", "viridis"),
                                                explain_panel("Mittlere Korrelation pro Tile",
                                                              {"Zeigt die mittlere Ähnlichkeit der in ein Tile eingehenden Framebeiträge.",
                                                               "Hohe Werte bedeuten lokal konsistente Geometrie und Signalstruktur.",
                                                               "Wenn die Karte überall fast identisch aussieht oder numerisch bei 1.0 sättigt, ist die Metrik in diesem Run kaum noch diskriminierend und trennt die Tiles nicht mehr sichtbar."},
                                                              {"<span class=\"good\">Gut:</span> Hohe, gleichmäßige CC-Werte.",
                                                               "<span class=\"bad\">Auffällig:</span> Lokale CC-Einbrüche deuten auf problematische Ausrichtung oder wechselhafte lokale Datenqualität hin."})});
        if (!post_snr.empty()) charts.push_back({svg_spatial_tile_heatmap(tiles, post_snr, img_w, img_h, "Post-reconstruction SNR", "SNR", "plasma"),
                                                 explain_panel("Post-Rekonstruktions-SNR",
                                                               {"Diese Heatmap zeigt einen tileweisen SNR-Proxy nach der lokalen Rekonstruktion.",
                                                                "Sie beantwortet, in welchen Bildbereichen die Rekonstruktion statistisch stark oder schwach ausfaellt."},
                                                               {"<span class=\"good\">Gut:</span> Hohe Werte in signalreichen Regionen ohne unplausible Flecken.",
                                                                "<span class=\"bad\">Auffällig:</span> Sehr niedrige oder stark inhomogene SNR-Muster können auf instabile Tile-Beiträge hinweisen."})});
        if (!post_contrast.empty()) charts.push_back({svg_spatial_tile_heatmap(tiles, post_contrast, img_w, img_h, "Post contrast per tile", "contrast", "cividis"),
                                                      explain_panel("Post-Kontrast pro Tile",
                                                                    {"Der Plot zeigt, wie stark der lokale Kontrast nach der Rekonstruktion ausfällt.",
                                                                     "Er hilft dabei, flache Regionen von detailreichen und eventuell überbetonten Regionen zu unterscheiden.",
                                                                     "Anders als Counts oder CC ist diese Karte typischerweise nicht homogen: das Motiv selbst erzeugt echte räumliche Kontrastunterschiede über das Feld."},
                                                                    {"<span class=\"good\">Gut:</span> Kontrast folgt dem Motiv und wirkt plausibel räumlich verteilt.",
                                                                     "<span class=\"bad\">Auffällig:</span> Isolierte Kontrastinseln oder harte Unterschiede zwischen Nachbartiles können auf Rekonstruktionsartefakte hinweisen."})});
        if (!post_bg.empty()) charts.push_back({svg_spatial_tile_heatmap(tiles, post_bg, img_w, img_h, "Post background per tile", "background", "gray"),
                                                explain_panel("Post-Hintergrund pro Tile",
                                                              {"Diese Karte zeigt den lokalen Hintergrund nach der Rekonstruktion.",
                                                               "Sie ist wichtig, um Restgradienten oder tileweise Offset-Unterschiede sichtbar zu machen."},
                                                              {"<span class=\"good\">Gut:</span> Ruhiger, homogen wirkender Hintergrund.",
                                                               "<span class=\"bad\">Auffällig:</span> Räumliche Hintergrundsprünge können später sichtbare Tile- oder Gradientenartefakte erzeugen."})});
    }
    if (!valid_counts.empty()) charts.push_back({svg_histogram(valid_counts, "Valid frame count distribution", "valid frames", "#4ade80"),
                                                 explain_panel("Verteilung gültiger Frame-Anzahlen",
                                                               {"Histogramm der effektiven Beitragshäufigkeit pro Tile.",
                                                                "Es zeigt, ob wenige Tiles statistisch aus dem Rahmen fallen oder ob die Rekonstruktion breit abgestützt ist."},
                                                               {"<span class=\"good\">Gut:</span> Konzentration in einem plausiblen, nicht zu niedrigen Bereich.",
                                                                "<span class=\"bad\">Auffällig:</span> Eine starke linke Flanke zeigt viele dünn abgestützte Tiles."})});
    if (!mean_cc.empty()) charts.push_back({svg_histogram(mean_cc, "Mean correlation distribution", "CC", "#60a5fa"),
                                            explain_panel("Verteilung tileweiser Korrelation",
                                                          {"Zeigt, wie sich die mittlere Tile-Korrelation über das gesamte Feld verteilt.",
                                                           "Damit lässt sich erkennen, ob lokale Ausrichtung/Konsistenz großflächig gut oder nur partiell robust ist."},
                                                          {"<span class=\"good\">Gut:</span> Schwerpunkt bei hohen Werten.",
                                                           "<span class=\"bad\">Auffällig:</span> Breite Verteilung oder viele kleine Werte deuten auf schwache tileweise Konsistenz hin."})});
    if (!post_snr.empty()) charts.push_back({svg_histogram(post_snr, "Post-reconstruction SNR distribution", "SNR", "#fbbf24"),
                                             explain_panel("Verteilung des Post-SNR",
                                                           {"Dieses Histogramm verdichtet den tileweisen SNR-Proxy zu einer globalen Übersicht.",
                                                            "Es hilft zu sehen, ob die Rekonstruktion überwiegend robust oder nur in Teilflächen stark ist."},
                                                           {"<span class=\"good\">Gut:</span> Solider Schwerpunkt ohne langen Niedrig-SNR-Auslauf.",
                                                            "<span class=\"bad\">Auffällig:</span> Viele Tiles mit schwachem SNR reduzieren die Stabilität des Endergebnisses."})});

    std::vector<std::string> evals = {
        "frames: " + json_string_or(recon, "num_frames", "?") + ", tiles: " + json_string_or(recon, "num_tiles", "?")
    };
    if (!valid_counts.empty()) {
        const auto s = basic_stats(valid_counts);
        evals.push_back("valid counts: median=" + format_number(s.median, 0) +
                        ", min=" + format_number(s.min, 0) +
                        ", max=" + format_number(s.max, 0));
        if (!(s.max > s.min)) evals.push_back("valid counts: tile map is constant");
        int low = 0;
        for (double v : valid_counts) if (std::isfinite(v) && v < 3.0) ++low;
        if (low > 0) evals.push_back("WARNING: " + std::to_string(low) + " tiles with < 3 valid frames");
    }
    if (!mean_cc.empty()) {
        const auto s = basic_stats(mean_cc);
        evals.push_back("tile CC: median=" + format_number(s.median, 4) +
                        ", min=" + format_number(s.min, 4));
        if (!(s.max > s.min)) evals.push_back("tile CC: tile map is constant");
    }
    if (!post_snr.empty()) {
        const auto s = basic_stats(post_snr);
        evals.push_back("post-SNR: median=" + format_number(s.median, 3) +
                        ", min=" + format_number(s.min, 3));
    }

    const std::string reconstruction_method = json_string_or(recon, "method", "unknown");
    std::string method_label = reconstruction_method;
    std::transform(method_label.begin(), method_label.end(), method_label.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return ReportSection{method_label + " Reconstruction",
                         make_card_html(method_label + "-specific reconstruction statistics",
                                        charts, evals, infer_status(evals))};
}

/// @brief Generates clustering.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_clustering(const json& cl) {
    if (!cl.is_object() || !cl.contains("cluster_sizes") || !cl["cluster_sizes"].is_array()) return std::nullopt;
    const auto sizes = json_double_array(cl.value("cluster_sizes", json::array()));
    if (sizes.empty()) return std::nullopt;
    std::vector<std::string> labels;
    labels.reserve(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) labels.push_back("C" + std::to_string(i));
    const auto cluster_labels = json_double_array(cl.value("cluster_labels", json::array()));

    std::vector<ChartBlock> charts = {
        {svg_bar(labels, sizes, "Cluster sizes", "frames"), explain_panel(
            "Clustergrößen",
            {
                "Jeder Balken zeigt, wie viele Frames einem Clustering-Zustand bzw. Qualitätscluster zugeordnet wurden.",
                "Der Plot beantwortet, ob die Session aus wenigen dominanten Zuständen oder vielen kleinen Subgruppen besteht."
            },
            {
                "<span class=\"good\">Gut:</span> Plausible Clusterverteilung ohne unerklärliche Mini-Clusterflut.",
                "<span class=\"neutral\">Neutral:</span> Ungleiche Cluster sind normal, wenn Wetter- oder Qualitätsphasen unterschiedlich lang waren.",
                "<span class=\"bad\">Auffällig:</span> Viele sehr kleine Cluster können auf instabile Merkmale oder überempfindliches Clustering hinweisen."
            }
        )}
    };
    if (!cluster_labels.empty()) {
        charts.push_back({svg_timeseries(cluster_labels, "Cluster label over time", "cluster", "#60a5fa", false),
                          explain_panel(
                              "Clusterlabel über die Zeit",
                              {
                                  "Die Zeitreihe zeigt für jeden Frame, welchem Cluster er zugeordnet wurde.",
                                  "So wird sichtbar, ob Cluster echte Session-Phasen repräsentieren oder nur bunt durchmischt auftreten."
                              },
                              {
                                  "<span class=\"good\">Gut:</span> Längere zusammenhängende Blöcke können reale Zustandsphasen der Session abbilden.",
                                  "<span class=\"bad\">Auffällig:</span> Starkes Hin-und-Her zwischen Clustern in kurzer Folge spricht eher für verrauschte Merkmale als für stabile Zustände."
                              }
                          )});
    }

    std::vector<std::string> evals = {
        "n_clusters: " + json_string_or(cl, "n_clusters", "?") +
        ", method: " + json_string_or(cl, "method", "?") +
        ", k_range: [" + json_string_or(cl, "k_min", "?") + ", " + json_string_or(cl, "k_max", "?") + "]"
    };
    for (size_t i = 0; i < sizes.size(); ++i) {
        evals.push_back("cluster " + std::to_string(i) + ": " + format_number(sizes[i], 0) + " frames");
    }

    return ReportSection{"State Clustering", make_card_html("Cluster analysis", charts, evals, "ok")};
}

/// @brief Generates synthetic.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_synthetic(const json& syn) {
    if (!syn.is_object() || syn.empty()) return std::nullopt;
    std::vector<std::string> evals = {
        "num_synthetic: " + json_string_or(syn, "num_synthetic", "0"),
        "frames range: [" + json_string_or(syn, "frames_min", "?") + ", " + json_string_or(syn, "frames_max", "?") + "]",
        "weighting: " + json_string_or(syn, "weighting", "global")
    };
    std::vector<ChartBlock> charts;
    const auto quality = json_double_array(syn.value("cluster_quality", json::array()));
    if (!quality.empty()) {
        std::vector<std::string> labels;
        labels.reserve(quality.size());
        for (size_t i = 0; i < quality.size(); ++i) labels.push_back("S" + std::to_string(i));
        charts.push_back({svg_bar(labels, quality, "Synthetic cluster quality", "quality", {}, 640, 300),
                          explain_panel(
                              "Qualität synthetischer Frames",
                              {
                                  "Jeder Balken entspricht einem synthetischen Frame bzw. dem zugrunde liegenden Cluster-Qualitätsscore.",
                                  "Der Plot zeigt, welche Cluster spaeter besonders stark oder schwach in die finale Aggregation eingehen."
                              },
                              {
                                  "<span class=\"good\">Gut:</span> Mehrere solide Cluster mit plausibler Qualität.",
                                  "<span class=\"neutral\">Neutral:</span> Einzelne schwache Cluster sind tolerierbar, wenn starke Cluster dominieren.",
                                  "<span class=\"bad\">Auffällig:</span> Überwiegend schwache oder stark streuende Clusterqualität reduziert den Nutzen der Synthetik."
                              }
                          )});
    }
    return ReportSection{"Synthetic Frames", make_card_html("Synthetic frame summary", charts, evals, "ok")};
}

/// @brief Generates bge.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_bge(const json& bge) {
    if (!bge.is_object() || bge.empty()) return std::nullopt;
    std::vector<std::string> evals = {
        "requested=" + std::string(json_bool_or(bge, "requested", false) ? "true" : "false") +
        ", attempted=" + std::string(json_bool_or(bge, "attempted", false) ? "true" : "false") +
        ", success=" + std::string(json_bool_or(bge, "success", false) ? "true" : "false")
    };

    if (bge.contains("summary") && bge["summary"].is_object()) {
        const auto& s = bge["summary"];
        evals.push_back("channels applied: " + json_string_or(s, "channels_applied", "0") + "/" + json_string_or(s, "channels_total", "0"));
        evals.push_back("fit success: " + json_string_or(s, "channels_fit_success", "0"));
        evals.push_back("valid tile samples: " + json_string_or(s, "tile_samples_valid", "0") + "/" + json_string_or(s, "tile_samples_total", "0"));
    }

    std::vector<ChartBlock> charts;
    if (bge.contains("channels") && bge["channels"].is_array() && !bge["channels"].empty()) {
        std::vector<std::string> labels;
        std::vector<double> mean_shifts;
        std::vector<double> residual_stds;
        std::vector<double> valid_ratios;
        for (const auto& ch : bge["channels"]) {
            labels.push_back(json_string_or(ch, "channel", "?"));
            mean_shifts.push_back(json_number_or(ch, "mean_shift", 0.0));
            residual_stds.push_back(ch.contains("residual_stats") && ch["residual_stats"].is_object()
                ? json_number_or(ch["residual_stats"], "std", 0.0) : 0.0);
            const double total = json_number_or(ch, "tile_samples_total", 0.0);
            const double valid = json_number_or(ch, "tile_samples_valid", 0.0);
            valid_ratios.push_back(total > 0.0 ? valid / total : 0.0);
        }
        charts.push_back({svg_bar(labels, mean_shifts, "BGE mean shift", "shift"), explain_panel(
            "BGE Mean Shift",
            {
                "Dieser Balkenplot zeigt die additive Hintergrundverschiebung, die BGE pro Kanal schätzen musste.",
                "Er beschreibt also, wie stark der lokale Hintergrund vor der Korrektur versetzt war."
            },
            {
                "Größere Werte bedeuten stärkere Korrektureingriffe.",
                "<span class=\"bad\">Prüfbedarf:</span> Sehr ungleiche Kanäle oder extreme Shifts deuten auf deutliche Gradienten, Kanalversätze oder Farbhintergrundprobleme hin; solche Fälle sollten gegen das lineare Zwischenbild geprüft werden."
            }
        )});
        charts.push_back({svg_bar(labels, residual_stds, "BGE residual std", "std"), explain_panel(
            "BGE Residual-Spread",
            {
                "Zeigt die Streuung der Residuen an den BGE-Stützpunkten nach dem Fit.",
                "Damit wird bewertet, wie sauber das Modell den Hintergrund erklären konnte."
            },
            {
                "<span class=\"good\">Unauffälliger Befund:</span> Eine kleine Residual-Standardabweichung bedeutet, dass die modellierte Hintergrundfläche die Stützpunkte konsistent erklärt und nur geringe Restfehler bleiben.",
                "<span class=\"bad\">Prüfbedarf:</span> Hohe Residuen sprechen für zu komplexe Bildstrukturen, zu wenig gültige Hintergrundsamples oder ein Modell, das den realen Gradienten nicht angemessen beschreibt."
            }
        )});
        charts.push_back({svg_bar(labels, valid_ratios, "Valid tile-sample ratio", "ratio"), explain_panel(
            "Anteil gültiger BGE-Samples",
            {
                "Dieser Plot zeigt, welcher Anteil der theoretisch verfügbaren BGE-Samples pro Kanal tatsächlich als gültig in den Fit einging.",
                "Ein niedriger Anteil bedeutet, dass der Fit auf wenig belastbare Stützpunkte zurückgreifen musste."
            },
            {
                "<span class=\"good\">Unauffälliger Befund:</span> Ein hoher gültiger Anteil pro Kanal bedeutet, dass BGE über das Feld ausreichend viele robuste Hintergrundstützpunkte hatte.",
                "<span class=\"bad\">Prüfbedarf:</span> Niedrige Werte reduzieren die Stabilität des Fits; die Pipeline kann die Korrektur dann überspringen oder sichtbare Restgradienten zurücklassen."
            }
        )});
    }

    return ReportSection{"Background Gradient Extraction (BGE)", make_card_html("BGE diagnostics", charts, evals, infer_status(evals))};
}

std::optional<ReportSection> gen_aqmh_metrics(const fs::path& run_dir, const json& metrics, const json& regions) {
    if (!metrics.is_object() || metrics.empty()) return std::nullopt;
    std::vector<ChartBlock> charts;
    std::vector<std::string> evals;
    const json diagnostics = metrics.contains("diagnostics") ? metrics["diagnostics"] : json::array();
    const auto map_means = json_diag_values(diagnostics, "map_mean");
    const auto map_p10 = json_diag_values(diagnostics, "map_p10");
    const auto map_p50 = json_diag_values(diagnostics, "map_p50");
    const auto map_p90 = json_diag_values(diagnostics, "map_p90");
    const auto artifact_fracs = json_diag_values(diagnostics, "artifact_frac");

    evals.push_back("frames total: " + std::to_string(static_cast<int>(json_number_or(metrics, "frames_total", 0.0))) +
                    ", written: " + std::to_string(static_cast<int>(json_number_or(metrics, "frames_written", 0.0))));
    evals.push_back("cache: " + std::to_string(static_cast<int>(json_number_or(metrics, "stored_width", 0.0))) + "x" +
                    std::to_string(static_cast<int>(json_number_or(metrics, "stored_height", 0.0))) + " " +
                    json_string_or(metrics, "dtype", "?") + " (full " +
                    std::to_string(static_cast<int>(json_number_or(metrics, "full_width", 0.0))) + "x" +
                    std::to_string(static_cast<int>(json_number_or(metrics, "full_height", 0.0))) + ")");
    if (!map_means.empty()) {
        const auto stats = basic_stats(map_means);
        evals.push_back("map_mean: min=" + format_number(stats.min, 4) +
                        ", median=" + format_number(stats.median, 4) +
                        ", max=" + format_number(stats.max, 4) +
                        ", mean=" + format_number(stats.mean, 4));
    }
    if (!artifact_fracs.empty()) {
        const auto stats = basic_stats(artifact_fracs);
        evals.push_back("artifact_fraction: min=" + format_number(stats.min * 100.0, 1) + "%" +
                        ", median=" + format_number(stats.median * 100.0, 1) + "%" +
                        ", max=" + format_number(stats.max * 100.0, 1) + "%" +
                        ", mean=" + format_number(stats.mean * 100.0, 1) + "%");
    }
    if (regions.is_object() && regions.contains("summary")) {
        const auto& summary = regions["summary"];
        evals.push_back("regions: total=" + std::to_string(static_cast<int>(json_number_or(summary, "total_regions", 0.0))) +
                        ", avg_size=" + format_number(json_number_or(summary, "avg_region_size_px", 0.0), 1) + "px");
    }

    const int stored_w = static_cast<int>(json_number_or(metrics, "stored_width", 0.0));
    const int stored_h = static_cast<int>(json_number_or(metrics, "stored_height", 0.0));
    const std::string dtype = json_string_or(metrics, "dtype", "");
    const std::string stream_id = json_string_or(metrics, "map_stream_id", "luma");
    const fs::path cache_dir = aqmh_cache_dir(run_dir, metrics);
    const auto files = aqmh_cache_files(cache_dir, stream_id);
    constexpr size_t max_aqmh_report_maps = 8;
    const auto sampled_files = sample_evenly(files, max_aqmh_report_maps);
    const auto agg = aggregate_aqmh_maps_streamed(sampled_files, stored_w, stored_h, dtype, 0.2, 80, 48);
    evals.push_back("cache maps streamed: " + std::to_string(agg.count) + "/" + std::to_string(files.size()) +
                    " sampled from " + cache_dir.string());
    if (agg.count > 0) {
        charts.push_back({svg_matrix_heatmap(agg.mean, agg.cols, agg.rows, "AQMH mean quality map", "Q mean", "viridis", 0.0, 1.0),
                          "<h4>AQMH mean quality map</h4><p>Gestreamte, report-kleine Vorschau der mittleren AQMH-Qualitaet; die Full-Resolution-Cache-Maps werden dabei nicht im Speicher gehalten.</p>"});
        charts.push_back({svg_matrix_heatmap(agg.artifact_frequency, agg.cols, agg.rows, "AQMH artifact frequency map", "artifact frequency", "inferno", 0.0, 1.0),
                          "<h4>AQMH artifact frequency map</h4><p>Gestreamte Vorschau des Anteils niedriger AQMH-Qualitaetswerte.</p>"});
        if (!agg.example.second.empty()) {
            charts.push_back({svg_matrix_heatmap(agg.example.second, agg.cols, agg.rows, "AQMH quality map example: " + agg.example.first, "Q", "viridis", 0.0, 1.0),
                              "<h4>AQMH quality map example</h4><p>Gestreamt heruntergerechnete AQMH-Qualitaetskarte: <code>" + html_escape(agg.example.first) + "</code>.</p>"});
        }
    }

    std::vector<std::vector<double>> metric_rows;
    std::vector<std::string> metric_labels;
    struct MetricSeriesRef {
        const char* label;
        const std::vector<double>* values;
    };
    const std::array<MetricSeriesRef, 5> metric_series{{
        {"map_p10", &map_p10},
        {"map_p50", &map_p50},
        {"map_p90", &map_p90},
        {"map_mean", &map_means},
        {"artifact_frac", &artifact_fracs},
    }};
    for (const auto& entry : metric_series) {
        if (entry.values != nullptr && !entry.values->empty()) {
            metric_labels.emplace_back(entry.label);
            metric_rows.push_back(*entry.values);
        }
    }
    if (!metric_rows.empty()) {
        size_t cols = 0;
        for (const auto& row : metric_rows) cols = std::max(cols, row.size());
        std::vector<double> matrix(metric_rows.size() * cols, std::numeric_limits<double>::quiet_NaN());
        for (size_t y = 0; y < metric_rows.size(); ++y) {
            for (size_t x = 0; x < metric_rows[y].size(); ++x) matrix[y * cols + x] = metric_rows[y][x];
        }
        charts.push_back({svg_matrix_heatmap(matrix, static_cast<int>(cols), static_cast<int>(metric_rows.size()), "AQMH frame metric matrix", "frame metrics", "viridis", 0.0, 1.0, 760, 300),
                          "<h4>AQMH Frame Metric Matrix</h4><p>Kompakte Heatmap der AQMH-Frame-Diagnostik aus <code>aqmh_metrics.json</code>.</p>"});
    }

    return ReportSection{"AQMH Metrics", make_card_html("AQMH quality metrics", charts, evals, infer_status(evals))};
}


/// @brief Generates validation.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_validation(const json& val) {
    if (!val.is_object() || val.empty()) return std::nullopt;
    const bool is_aqmh = json_string_or(val, "method", "aqmh") == "aqmh";
    const double improvement = json_number_or(val, "fwhm_improvement_percent", 0.0);
    const auto fwhm_ok_opt = json_optional_bool(val, "fwhm_improvement_ok");

    std::vector<std::string> labels;
    std::vector<double> values;
    std::vector<std::string> colors;

    // FWHM: always shown, but informational (cyan) when not evaluated (AQMH without star detection)
    labels.push_back("FWHM improvement");
    values.push_back(improvement);
    if (!fwhm_ok_opt.has_value()) {
        colors.push_back("#22d3ee");  // cyan = informational
    } else {
        colors.push_back(*fwhm_ok_opt ? "#4ade80" : "#f87171");
    }

    if (!is_aqmh) {
        const double tw_var = json_number_or(val, "tile_weight_variance", 0.0);
        const double pattern_ratio = json_number_or(val, "tile_pattern_ratio", 0.0);
        const bool tw_ok = json_bool_or(val, "tile_weight_variance_ok", false);
        const bool pattern_ok = json_bool_or(val, "tile_pattern_ok", false);
        labels.push_back("Tile weight variance");
        values.push_back(tw_var * 100.0);
        colors.push_back(tw_ok ? "#4ade80" : "#f87171");
        if (val.contains("tile_pattern_ratio")) {
            labels.push_back("Tile pattern ratio");
            values.push_back(pattern_ratio);
            colors.push_back(pattern_ok ? "#4ade80" : "#f87171");
        }
    } else {
        // AQMH-specific quality metrics
        const double map_var = json_number_or(val, "aqmh_map_mean_variance", -1.0);
        const double artifact_avg = json_number_or(val, "aqmh_artifact_frac_avg", -1.0);
        if (map_var >= 0.0) {
            labels.push_back("AQMH map variance");
            values.push_back(map_var * 1000.0);
            colors.push_back(map_var > 1e-5 ? "#4ade80" : "#fb923c");
        }
        if (artifact_avg >= 0.0) {
            labels.push_back("AQMH artifact frac");
            values.push_back(artifact_avg * 100.0);
            colors.push_back(artifact_avg < 0.3 ? "#4ade80" : "#f87171");
        }
    }

    std::vector<ChartBlock> charts = {{
        svg_bar(labels, values, "Validation checks", "value", colors),
        explain_panel(
            "Validierungschecks",
            {
                is_aqmh
                    ? "AQMH-Modus: FWHM ist informativ (kein Pass/Fail). Stattdessen werden AQMH-spezifische Qualitätsmetriken aus den Qualitätskarten angezeigt."
                    : "Die Balken zeigen die wichtigsten numerischen Endkontrollen des Ergebnisses, z. B. FWHM-Verbesserung, Tile-Weight-Varianz und optional den Tile-Pattern-Check.",
                "Die Farbe macht sofort sichtbar, welche Checks bestanden und welche fehlgeschlagen sind."
            },
            {
                "<span class=\"good\">Bestandener Check:</span> Der jeweilige Messwert liegt im akzeptierten Bereich und stützt die technische Plausibilität des Endprodukts.",
                "<span class=\"bad\">Fehlgeschlagener Check:</span> Der Messwert verletzt die definierte Grenze und markiert ein konkretes Qualitätsrisiko, das im Bild oder in den vorgelagerten Phasen geprüft werden sollte.",
                "Die absolute Balkenhoehe ist nur im Kontext des jeweiligen Checks interpretierbar; entscheidend ist die Kombination aus Wert und PASS/FAIL."
            }
        )
    }};

    std::vector<std::string> evals;
    evals.push_back("seeing FWHM: " + json_string_or(val, "seeing_fwhm_median", "?"));
    evals.push_back("output FWHM: " + json_string_or(val, "output_fwhm_median", "?"));
    if (!fwhm_ok_opt.has_value()) {
        evals.push_back("FWHM improvement: " + format_number(improvement, 1) + "% (informational)");
    } else {
        evals.push_back("FWHM improvement: " + format_number(improvement, 1) + "% " + (*fwhm_ok_opt ? std::string("OK") : std::string("FAIL")));
    }
    if (!is_aqmh) {
        const double tw_var = json_number_or(val, "tile_weight_variance", 0.0);
        const double pattern_ratio = json_number_or(val, "tile_pattern_ratio", 0.0);
        const bool tw_ok = json_bool_or(val, "tile_weight_variance_ok", false);
        const bool pattern_ok = json_bool_or(val, "tile_pattern_ok", false);
        evals.push_back("tile weight variance: " + format_number(tw_var, 4) + " " + (tw_ok ? std::string("OK") : std::string("FAIL")));
        if (val.contains("tile_pattern_ratio")) {
            evals.push_back("tile pattern ratio: " + format_number(pattern_ratio, 3) + " " + (pattern_ok ? std::string("OK") : std::string("FAIL")));
        }
    } else {
        const double map_avg = json_number_or(val, "aqmh_map_mean_avg", -1.0);
        const double map_var = json_number_or(val, "aqmh_map_mean_variance", -1.0);
        const double artifact_avg = json_number_or(val, "aqmh_artifact_frac_avg", -1.0);
        const int n_eval = static_cast<int>(json_number_or(val, "aqmh_frames_evaluated", 0.0));
        if (map_avg >= 0.0) evals.push_back("AQMH map mean avg: " + format_number(map_avg, 4));
        if (map_var >= 0.0) evals.push_back("AQMH map mean variance: " + format_number(map_var, 6));
        if (artifact_avg >= 0.0) evals.push_back("AQMH artifact frac avg: " + format_number(artifact_avg, 3));
        if (n_eval > 0) evals.push_back("AQMH frames evaluated: " + std::to_string(n_eval));
    }
    return ReportSection{"Validation", make_card_html("Quality validation", charts, evals, infer_status(evals))};
}

/// @brief Generates common overlap.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
std::optional<ReportSection> gen_common_overlap(const json& co) {
    if (!co.is_object() || !co.contains("tiles") || !co["tiles"].is_array() || co["tiles"].empty()) return std::nullopt;
    std::vector<double> ratios;
    ratios.reserve(co["tiles"].size());
    int valid_count = 0;
    for (const auto& tile : co["tiles"]) {
        ratios.push_back(json_number_or(tile, "common_ratio", NAN));
        if (json_bool_or(tile, "common_valid", false)) ++valid_count;
    }
    std::vector<ChartBlock> charts = {
        {svg_histogram(ratios, "Tile common-overlap ratio", "common ratio", "#22d3ee"), explain_panel(
            "Common-Overlap-Ratio pro Tile",
            {
                "Das Histogramm beschreibt, welcher Anteil der Pixel pro Tile in der gemeinsamen, über alle nutzbaren Frames stabil überlappenden Region liegt.",
                "Es ist damit ein wichtiger Indikator für geometrische Abdeckung und statistische Fairness lokaler Metriken."
            },
            {
                "<span class=\"good\">Unauffälliger Befund:</span> Liegt der Schwerpunkt nahe hoher Ratios, wurden die meisten Tiles über viele Frames hinweg gemeinsam und geometrisch stabil abgedeckt.",
                "<span class=\"neutral\">Normaler Befund:</span> Ein Abfall an den Bildrändern ist bei Feldrotation, Dithering oder ungleichmäßiger Abdeckung häufig physikalisch plausibel.",
                "<span class=\"bad\">Prüfbedarf:</span> Viele niedrige Ratios bedeuten, dass große Teile des Felds lokal nur schwach gemeinsam beobachtet wurden; lokale Metriken und Rekonstruktion sind dort statistisch weniger belastbar."
            }
        )}
    };
    const int img_w = static_cast<int>(json_number_or(co, "canvas_width", 0.0));
    const int img_h = static_cast<int>(json_number_or(co, "canvas_height", 0.0));
    if (img_w > 0 && img_h > 0) {
        charts.push_back({
            svg_spatial_tile_heatmap(co["tiles"], ratios, img_w, img_h, "Spatial common-overlap ratio", "common ratio", "viridis"),
            explain_panel(
                "Räumliche Common-Overlap-Karte",
                {
                    "Diese Karte zeigt die gemeinsame Abdeckung nicht als Verteilung, sondern direkt an der realen Bildposition jedes Tiles.",
                    "Damit erkennt man sofort, welche Feldbereiche geometrisch gut abgestützt sind und wo die Session lokal ausdünnt."
                },
                {
                    "<span class=\"good\">Unauffälliger Befund:</span> Eine homogene, breitflächig hohe Abdeckung bedeutet, dass lokale Messwerte im Großteil des Felds auf vergleichbarer Statistik beruhen.",
                    "<span class=\"neutral\">Normaler Befund:</span> Ein gleichmäßiger Abfall an den Rändern ist bei realer Feldrotation oder Dithering oft normal, solange keine isolierten Löcher entstehen.",
                    "<span class=\"bad\">Prüfbedarf:</span> Inselartige Lücken oder starke Inhomogenität können lokale Bias-Effekte, instabile Gewichtung und Rekonstruktionsartefakte begünstigen."
                }
            )
        });
    }
    const auto s = basic_stats(ratios);
    std::vector<std::string> evals = {
        "canvas: " + json_string_or(co, "canvas_width", "?") + "x" + json_string_or(co, "canvas_height", "?"),
        "usable/loaded frames: " + json_string_or(co, "usable_frames", "?") + "/" + json_string_or(co, "loaded_frames", "?"),
        "common pixels: " + json_string_or(co, "common_pixels", "?") + " (" + format_number(percent_value(json_number_or(co, "common_fraction", 0.0)), 1) + "%)",
        "tiles common-valid: " + std::to_string(valid_count) + "/" + std::to_string(co["tiles"].size())
    };
    if (s.n > 0) {
        evals.push_back("tile common-ratio median=" + format_number(s.median, 3) +
                        ", min=" + format_number(s.min, 3) +
                        ", max=" + format_number(s.max, 3));
    }
    return ReportSection{"Common Overlap", make_card_html("Post-PREWARP overlap diagnostics", charts, evals, infer_status(evals))};
}

std::string build_report_html(const fs::path& run_dir,
                              const json& status,
                              const json& artifacts,
                              const std::vector<json>& events,
                              const json& norm,
                              const json& gm,
                              const json& tg,
                              const json& reg,
                              const json& lm,
                              const json& recon,
                              const json& cl,
                              const json& syn,
                              const json& bge,
                              const json& val,
                              const json& aqmh_metrics,
                              const json& aqmh_regions,
                              const json& common_overlap,
                              const std::string& config_yaml,
                              const std::string& locale) {
    std::vector<std::string> meta_lines = {
        "run_id: " + run_dir.filename().string(),
        "run_dir: " + run_dir.string(),
    };
    for (const auto& ev : events) {
        const auto type = json_string_or(ev, "type", "");
        if (type == "run_start") {
            const auto input_dir = json_string_or(ev, "input_dir", "");
            const auto frames = json_string_or(ev, "frames_discovered", "");
            const auto ts = json_string_or(ev, "ts", "");
            if (!input_dir.empty()) meta_lines.push_back("input_dir: " + input_dir);
            if (!frames.empty()) meta_lines.push_back("frames: " + frames);
            if (!ts.empty()) meta_lines.push_back("timestamp: " + ts);
        }
        if (type == "run_end") {
            const auto st = json_string_or(ev, "status", "");
            if (!st.empty()) meta_lines.push_back("final status: " + st);
        }
    }

    std::vector<ReportSection> sections;
    auto add = [&](std::optional<ReportSection> sec) {
        if (sec && !sec->cards_html.empty()) sections.push_back(std::move(*sec));
    };
    add(gen_overview(run_dir, status, artifacts, events));
    add(gen_timeline(events));
    add(gen_frame_usage(events, syn));
    add(gen_normalization(norm));
    add(gen_global_metrics(gm));
    add(gen_tile_grid(tg));
    add(gen_registration(reg));
    add(gen_local_metrics(lm, tg));
    add(gen_reconstruction(recon, tg));
    add(gen_aqmh_metrics(run_dir, aqmh_metrics, aqmh_regions));
    add(gen_clustering(cl));
    add(gen_synthetic(syn));
    add(gen_bge(bge));
    add(gen_validation(val));
    add(gen_common_overlap(common_overlap));
    add(gen_phase_issue_summary(events, bge));

    std::ostringstream html;
    html << "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
         << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
         << "<title>" << html_escape("Tile-Compile Report - " + run_dir.filename().string()) << "</title>"
         << "<style>"
         << ":root{color-scheme:dark;--bg:#020617;--panel:#0f172a;--panel2:#111827;--line:#334155;--text:#e2e8f0;--muted:#94a3b8;--good:#4ade80;--warn:#fbbf24;--bad:#f87171;}"
         << "*{box-sizing:border-box;}body{margin:0;background:radial-gradient(circle at top,#0f172a,#020617 60%);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,sans-serif;}"
         << "header{padding:32px 28px 18px;border-bottom:1px solid rgba(148,163,184,.16);background:linear-gradient(180deg,rgba(15,23,42,.88),rgba(2,6,23,.96));position:sticky;top:0;backdrop-filter:blur(10px);z-index:5;}"
         << "header h1{margin:0 0 8px;font-size:28px;}header .meta{color:var(--muted);font-size:13px;display:flex;flex-wrap:wrap;gap:8px 16px;}"
         << ".header-top{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;}.language-switch{display:inline-flex;align-items:center;gap:4px;padding:3px;border:1px solid rgba(148,163,184,.24);border-radius:999px;background:rgba(2,6,23,.55);flex:0 0 auto;}"
         << ".language-switch button{border:0;border-radius:999px;background:transparent;color:var(--muted);font:600 12px/1 ui-sans-serif,system-ui,sans-serif;padding:7px 10px;cursor:pointer;}.language-switch button.active{background:#2563eb;color:#fff;}.language-switch button:focus-visible{outline:2px solid #93c5fd;outline-offset:2px;}"
         << "main{padding:24px 22px 34px;max-width:1600px;margin:0 auto;}section{margin:0 0 28px;}section h2{margin:0 0 14px;font-size:19px;}"
         << ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;align-items:start;}"
         << ".card{background:linear-gradient(180deg,rgba(17,24,39,.96),rgba(15,23,42,.96));border:1px solid rgba(148,163,184,.15);border-radius:16px;padding:16px 16px 12px;box-shadow:0 8px 28px rgba(0,0,0,.18);}"
         << ".card h3{margin:0 0 12px;font-size:16px;display:flex;align-items:center;gap:10px;}"
         << ".badge{display:inline-flex;align-items:center;border-radius:999px;padding:2px 8px;font-size:11px;text-transform:uppercase;border:1px solid currentColor;}"
         << ".badge.ok,.card.ok .badge{color:var(--good);} .badge.warn,.card.warn .badge{color:var(--warn);} .badge.bad,.card.bad .badge{color:var(--bad);}"
         << ".chart-row{display:grid;grid-template-columns:minmax(0,1fr);gap:10px;margin:0 0 14px;align-items:start;}"
         << ".chart-col{min-width:0;}"
         << ".explain-col{color:var(--muted);font-size:13px;background:rgba(255,255,255,.025);border:1px solid rgba(148,163,184,.12);border-radius:12px;padding:12px 13px;}"
         << ".explain-col h4{margin:0 0 8px;font-size:12px;color:#93c5fd;text-transform:uppercase;letter-spacing:.04em;}"
         << ".explain-col p{margin:0 0 8px;line-height:1.55;} .explain-col ul{margin:4px 0 0 16px;padding:0;} .explain-col li{margin:4px 0;line-height:1.5;} .explain-col .good{color:var(--good);font-weight:600;} .explain-col .neutral{color:var(--warn);font-weight:600;} .explain-col .bad{color:var(--bad);font-weight:600;}"
         << ".metric-box{border-top:1px solid rgba(148,163,184,.12);padding-top:10px;margin-top:6px;} .metric-box ul{margin:0;padding-left:18px;} .metric-box li{margin:3px 0;} .metric-box li.warn{color:#fecaca;}"
         << ".kv,.phases{width:100%;border-collapse:collapse;} .kv th,.kv td,.phases th,.phases td{padding:8px 10px;border-bottom:1px solid rgba(148,163,184,.12);text-align:left;vertical-align:top;} .kv th,.phases th{color:#cbd5e1;width:36%;font-weight:600;}"
         << ".artifact-list{margin:0;padding-left:18px;} .artifact-list li{margin:3px 0;} .muted{color:var(--muted);} code,pre{font-family:ui-monospace,SFMono-Regular,monospace;} pre{margin:0;white-space:pre-wrap;overflow:auto;background:#020617;border:1px solid rgba(148,163,184,.14);border-radius:12px;padding:12px;}"
         << "details.config{margin-top:26px;background:rgba(15,23,42,.92);border:1px solid rgba(148,163,184,.15);border-radius:14px;padding:14px 16px;} details summary{cursor:pointer;font-weight:600;}"
         << ".footer{margin-top:18px;color:var(--muted);font-size:12px;}"
         << "svg.report-chart{width:100%;height:auto;display:block;} svg.report-chart line,svg.report-chart polyline,svg.report-chart path{vector-effect:non-scaling-stroke;} .svg-title{fill:#e2e8f0;font-size:14px;font-weight:700;} .svg-title-small{fill:#e2e8f0;font-size:18px;font-weight:700;} .svg-label{fill:#94a3b8;font-size:11px;} .svg-note{fill:#94a3b8;font-size:13px;} .svg-tick{fill:#94a3b8;font-size:10px;} .svg-axis{stroke:#64748b;stroke-width:0.8;} .svg-grid{stroke:#1e293b;stroke-width:0.7;}"
         << "@media (min-width:1100px){.chart-row{grid-template-columns:minmax(0,1fr) minmax(0,1fr);}}"
         << "@media (max-width:720px){.header-top{flex-direction:column;}.language-switch{align-self:flex-start;}}"
         << "</style></head><body>";
    html << "<header><div class=\"header-top\"><div id=\"report-header-content\"><!--REPORT_HEADER_BEGIN--><h1>Tile-Compile Report</h1><div class=\"meta\">";
    for (const auto& line : meta_lines) html << "<span>" << html_escape(line) << "</span>";
    html << "</div><!--REPORT_HEADER_END--></div><nav class=\"language-switch\" aria-label=\"Report language\"><button type=\"button\" data-report-lang=\"de\">Deutsch</button><button type=\"button\" data-report-lang=\"en\">English</button></nav></div></header><main><div id=\"report-content\"><!--REPORT_CONTENT_BEGIN-->";
    for (const auto& section : sections) {
        html << "<section><h2>" << html_escape(section.title) << "</h2><div class=\"grid\">" << section.cards_html << "</div></section>";
    }
    html << "<!--REPORT_CONTENT_END--></div>";
    if (!config_yaml.empty()) {
        html << "<details class=\"config\"><summary>Config (config.yaml)</summary><pre>" << html_escape(config_yaml) << "</pre></details>";
    }
    html << "<div class=\"footer\">Generated by tile_compile_web_backend (C++ inline SVG report)</div>";
    html << "</main>__REPORT_LANGUAGE_SCRIPT__</body></html>";
    const std::string base_html = html.str();
    std::string localized = apply_report_translations(base_html, locale);
    const std::string marker = "__REPORT_LANGUAGE_SCRIPT__";
    const auto marker_pos = localized.find(marker);
    if (marker_pos != std::string::npos) {
        localized.replace(marker_pos, marker.size(), build_language_switch_script(locale));
    }
    return localized;
}

} // namespace

/// @brief Generates run report.
/// @details This implementation turns run artifacts and events into the generated HTML report payload; it keeps JSON shapes, filesystem
/// access, process handling, and error reporting localized to this backend component.
nlohmann::json generate_run_report(const fs::path& run_dir) {
    try {
        const std::string locale = normalize_report_locale(env_or("TILE_COMPILE_REPORT_LOCALE", "de"));
        const fs::path artifacts_dir = run_dir / "artifacts";
        fs::create_directories(artifacts_dir);

        const fs::path report_path = artifacts_dir / "report.html";
        const fs::path summary_path = artifacts_dir / "stats.json";

        const auto status = read_run_status(run_dir);
        auto artifacts_before = list_run_artifacts(run_dir);

        std::vector<json> events;
        for (const auto& candidate : {
                run_dir / "logs" / "run_events.jsonl",
                run_dir / "events.jsonl",
                run_dir / "logs" / "events.jsonl",
            }) {
            events = read_jsonl_if_exists(candidate);
            if (!events.empty()) break;
        }

        const json norm = read_json_if_exists(artifacts_dir / "normalization.json");
        const json gm = read_json_if_exists(artifacts_dir / "global_metrics.json");
        const json tg = read_json_if_exists(artifacts_dir / "tile_grid.json");
        const json reg = read_json_if_exists(artifacts_dir / "global_registration.json");
        const json lm = read_json_if_exists(artifacts_dir / "local_metrics.json");
        json recon = read_json_if_exists(artifacts_dir / "aqmh_reconstruction.json");
        if (recon.empty()) recon = read_json_if_exists(artifacts_dir / "tile_reconstruction.json");
        const json cl = read_json_if_exists(artifacts_dir / "state_clustering.json");
        const json syn = read_json_if_exists(artifacts_dir / "synthetic_frames.json");
        const json bge = read_json_if_exists(artifacts_dir / "bge.json");
        const json val = read_json_if_exists(artifacts_dir / "validation.json");
        const json aqmh_metrics = read_json_if_exists(artifacts_dir / "aqmh_metrics.json");
        const json aqmh_regions = read_json_if_exists(artifacts_dir / "aqmh_regions.json");
        const json common_overlap = read_json_if_exists(artifacts_dir / "common_overlap.json");
        const std::string config_yaml = read_text(run_dir / "config.yaml");

        const std::string report_html = build_report_html(run_dir, status, artifacts_before, events,
                                                          norm, gm, tg, reg, lm, recon, cl,
                                                          syn, bge, val, aqmh_metrics, aqmh_regions,
                                                          common_overlap, config_yaml, locale);

        std::ofstream report_out(report_path, std::ios::binary);
        if (!report_out) {
            return {
                {"ok", false},
                {"error", "cannot write report.html"},
                {"report_path", report_path.string()},
                {"summary_path", summary_path.string()},
            };
        }
        report_out << report_html;
        report_out.close();

        const auto artifacts_after = list_run_artifacts(run_dir);
        const json summary = build_report_summary_json(run_dir, status, artifacts_after, events);

        std::ofstream summary_out(summary_path, std::ios::binary);
        if (!summary_out) {
            return {
                {"ok", false},
                {"error", "cannot write stats.json"},
                {"report_path", report_path.string()},
                {"summary_path", summary_path.string()},
            };
        }
        summary_out << summary.dump(2);

        return {
            {"ok", true},
            {"run_id", run_dir.filename().string()},
            {"output_dir", artifacts_dir.string()},
            {"report_path", report_path.string()},
            {"summary_path", summary_path.string()},
            {"artifact_count", artifacts_after.is_array() ? artifacts_after.size() : 0},
            {"event_count", events.size()},
            {"report_format", "inline_svg"},
            {"report_locale", locale},
        };
    } catch (const std::exception& e) {
        return {
            {"ok", false},
            {"run_id", run_dir.filename().string()},
            {"output_dir", (run_dir / "artifacts").string()},
            {"report_path", (run_dir / "artifacts" / "report.html").string()},
            {"summary_path", (run_dir / "artifacts" / "stats.json").string()},
            {"error", e.what()},
        };
    } catch (...) {
        return {
            {"ok", false},
            {"run_id", run_dir.filename().string()},
            {"output_dir", (run_dir / "artifacts").string()},
            {"report_path", (run_dir / "artifacts" / "report.html").string()},
            {"summary_path", (run_dir / "artifacts" / "stats.json").string()},
            {"error", "unknown report generation error"},
        };
    }
}
