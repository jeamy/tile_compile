#include "services/report_generator.hpp"
#include "services/run_inspector.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;

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

std::string trim_trailing_zeros(std::string s) {
    auto pos = s.find('.');
    if (pos == std::string::npos) return s;
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    if (s == "-0") return "0";
    return s;
}

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

std::string sanitize_label(std::string s) {
    std::replace(s.begin(), s.end(), '\n', ' ');
    return s;
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

json read_json_if_exists(const fs::path& path) {
    std::ifstream in(path);
    if (!in) return json::object();
    auto parsed = json::parse(in, nullptr, false);
    if (parsed.is_discarded()) return json::object();
    return parsed;
}

std::vector<json> read_jsonl_if_exists(const fs::path& path, int max_lines = 100000) {
    std::ifstream in(path);
    std::vector<json> items;
    if (!in) return items;
    std::string line;
    int n = 0;
    while (std::getline(in, line) && n < max_lines) {
        if (line.empty()) continue;
        auto j = json::parse(line, nullptr, false);
        if (!j.is_discarded() && j.is_object()) items.push_back(std::move(j));
        ++n;
    }
    return items;
}

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

double json_number_or(const json& obj, const char* key, double fallback = 0.0) {
    if (!obj.is_object() || !obj.contains(key) || obj.at(key).is_null()) return fallback;
    const auto& value = obj.at(key);
    try {
        if (value.is_number()) return value.get<double>();
        if (value.is_string()) return std::stod(value.get<std::string>());
    } catch (...) {}
    return fallback;
}

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

double percent_value(double raw) {
    if (raw >= 0.0 && raw <= 1.0) return raw * 100.0;
    return raw;
}

double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

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

double scale_linear(double v, double in_min, double in_max, double out_min, double out_max) {
    if (!(in_max > in_min)) return (out_min + out_max) * 0.5;
    const double t = (v - in_min) / (in_max - in_min);
    return out_min + t * (out_max - out_min);
}

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

std::string phase_name_from_event(const json& ev) {
    if (ev.contains("phase_name") && ev["phase_name"].is_string()) return ev["phase_name"].get<std::string>();
    if (ev.contains("phase")) {
        if (ev["phase"].is_string()) return ev["phase"].get<std::string>();
        if (ev["phase"].is_number_integer()) return std::to_string(ev["phase"].get<int>());
    }
    return "";
}

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

std::array<int, 3> rgb_from_hex(const std::string& hex) {
    if (hex.size() != 7 || hex[0] != '#') return {122, 162, 247};
    auto from_pair = [&](size_t pos) {
        return static_cast<int>(std::strtol(hex.substr(pos, 2).c_str(), nullptr, 16));
    };
    return {from_pair(1), from_pair(3), from_pair(5)};
}

std::string rgb_hex(const std::array<int, 3>& rgb) {
    std::ostringstream ss;
    ss << '#'
       << std::hex << std::setw(2) << std::setfill('0') << std::clamp(rgb[0], 0, 255)
       << std::setw(2) << std::setfill('0') << std::clamp(rgb[1], 0, 255)
       << std::setw(2) << std::setfill('0') << std::clamp(rgb[2], 0, 255);
    return ss.str();
}

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
            << "\" y2=\"" << ym << "\" stroke=\"#f87171\" stroke-dasharray=\"6 4\" stroke-width=\"1.5\" opacity=\"0.9\"/>";
    }

    std::ostringstream poly;
    for (const auto& [idx, val] : values) {
        const double x = scale_linear(static_cast<double>(idx), 0.0, static_cast<double>(max_index), x0, x0 + pw);
        const double y = scale_linear(val, bounds.first, bounds.second, y0 + ph, y0);
        poly << x << ',' << y << ' ';
    }
    out << "<polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"2.4\" points=\""
        << poly.str() << "\"/>";
    if (values.size() <= 80) {
        for (const auto& [idx, val] : values) {
            const double x = scale_linear(static_cast<double>(idx), 0.0, static_cast<double>(max_index), x0, x0 + pw);
            const double y = scale_linear(val, bounds.first, bounds.second, y0 + ph, y0);
            out << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"2.6\" fill=\"" << color << "\"/>";
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

    size_t color_index = 0;
    double legend_x = x0 + 6.0;
    const double legend_y = 32.0;
    for (const auto& [name, vals] : series) {
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
        const std::string color = palette[color_index % palette.size()];
        out << "<polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"2.2\" points=\""
            << poly.str() << "\"/>";
        out << "<line x1=\"" << legend_x << "\" y1=\"" << legend_y << "\" x2=\"" << (legend_x + 16)
            << "\" y2=\"" << legend_y << "\" stroke=\"" << color << "\" stroke-width=\"3\"/>";
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
        << "\" y2=\"" << (y0 + ph) << "\" stroke=\"#f87171\" stroke-dasharray=\"6 4\" stroke-width=\"1.5\"/>";
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
        const double h = scale_linear(values[i], 0.0, top_val, 0.0, ph);
        const double y = y0 + ph - h;
        const std::string color = i < colors.size() ? colors[i] : colormap_hex("plasma", static_cast<double>(i) / std::max<size_t>(1, labels.size() - 1));
        out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << bar_w << "\" height=\"" << h
            << "\" rx=\"3\" fill=\"" << color << "\" opacity=\"0.9\"/>";
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
        const double w = scale_linear(values[i], 0.0, max_val, 0.0, pw);
        const std::string color = i < colors.size() ? colors[i] : colormap_hex("viridis", static_cast<double>(i) / std::max<size_t>(1, labels.size() - 1));
        out << "<rect x=\"" << x0 << "\" y=\"" << y << "\" width=\"" << w << "\" height=\"" << bar_h
            << "\" rx=\"4\" fill=\"" << color << "\" opacity=\"0.9\"/>";
        out << "<text x=\"" << (x0 - 10) << "\" y=\"" << (y + bar_h * 0.5 + 4)
            << "\" class=\"svg-tick\" text-anchor=\"end\">" << html_escape(sanitize_label(labels[i]))
            << "</text>";
        out << "<text x=\"" << (x0 + w + 8) << "\" y=\"" << (y + bar_h * 0.5 + 4)
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
    if (s.min >= 0.0 && s.max <= 1.0) {
        lo = 0.0;
        hi = 1.0;
    } else if (s.n >= 20 && s.p99 > s.p01) {
        lo = s.p01;
        hi = s.p99;
    }
    if (!(hi > lo)) {
        lo -= 1.0;
        hi += 1.0;
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
    for (int i = 0; i < 64; ++i) {
        const double t = static_cast<double>(i) / 63.0;
        const double y = y0 + panel_h - t * panel_h;
        out << "<rect x=\"" << cbx << "\" y=\"" << y << "\" width=\"" << cbw << "\" height=\"" << (panel_h / 63.0 + 1.0)
            << "\" fill=\"" << colormap_hex(cmap, t) << "\"/>";
    }
    out << "<rect x=\"" << cbx << "\" y=\"" << y0 << "\" width=\"" << cbw << "\" height=\"" << panel_h
        << "\" fill=\"none\" class=\"svg-axis\"/>";
    out << "<text x=\"" << cbx << "\" y=\"" << (height - 16)
        << "\" class=\"svg-label\">" << html_escape(label) << "</text>";
    out << "<text x=\"" << (cbx + cbw + 8) << "\" y=\"" << (y0 + 4)
        << "\" class=\"svg-tick\">" << html_escape(format_number(hi, 2)) << "</text>";
    out << "<text x=\"" << (cbx + cbw + 8) << "\" y=\"" << (y0 + panel_h)
        << "\" class=\"svg-tick\">" << html_escape(format_number(lo, 2)) << "</text>";
    out << "</svg>";
    return out.str();
}

std::string render_kv_table(const std::vector<std::pair<std::string, std::string>>& rows) {
    std::ostringstream html;
    html << "<table class=\"kv\"><tbody>";
    for (const auto& [key, value] : rows) {
        html << "<tr><th>" << html_escape(key) << "</th><td>" << html_escape(value) << "</td></tr>";
    }
    html << "</tbody></table>";
    return html.str();
}

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

std::string render_event_tail(const std::vector<json>& events, size_t max_lines = 24) {
    std::ostringstream text;
    const size_t start = events.size() > max_lines ? events.size() - max_lines : 0;
    for (size_t i = start; i < events.size(); ++i) {
        text << format_event_line(events[i]) << '\n';
    }
    return "<pre class=\"log-tail\">" + html_escape(text.str()) + "</pre>";
}

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

std::optional<ReportSection> gen_timeline(const std::vector<json>& events) {
    std::map<std::string, double> phase_starts;
    std::vector<std::string> labels;
    std::vector<double> durations;
    std::vector<std::string> evals;
    for (const auto& ev : events) {
        const auto type = json_string_or(ev, "type", "");
        const auto phase = phase_name_from_event(ev);
        const auto ts = json_string_or(ev, "ts", json_string_or(ev, "timestamp", ""));
        if (phase.empty() || ts.empty()) continue;
        const auto secs = parse_iso_utc_seconds(ts);
        if (!secs) continue;
        if (type == "phase_start") {
            phase_starts[phase] = *secs;
        } else if (type == "phase_end") {
            auto it = phase_starts.find(phase);
            if (it == phase_starts.end()) continue;
            const double dt = std::max(0.0, *secs - it->second);
            labels.push_back(phase);
            durations.push_back(dt);
            evals.push_back(phase + ": " + format_number(dt, 1) + " s");
        }
    }
    if (labels.empty()) return std::nullopt;
    const double total = std::accumulate(durations.begin(), durations.end(), 0.0);
    evals.insert(evals.begin(), "total pipeline time: " + format_number(total, 1) + " s");
    std::vector<ChartBlock> charts = {{
        svg_bar_horizontal(labels, durations, "Pipeline phase durations", "seconds"),
        "<p>Duration per recorded pipeline phase. Long bars highlight the phases that dominate runtime.</p>"
    }};
    return ReportSection{"Pipeline Timeline", make_card_html("Phase durations", charts, evals, "ok")};
}

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
        "<p>Retention of frames through the main acceptance stages.</p>"
    }};
    if (!loss_labels.empty()) {
        charts.push_back({
            svg_pie(loss_labels, loss_values, loss_colors, "Frame loss breakdown"),
            "<p>Loss causes split by stage. A large non-green slice usually points to acquisition or registration problems.</p>"
        });
    }
    return ReportSection{"Frame Usage", make_card_html("Frame retention", charts, evals, infer_status(evals))};
}

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
            "<p>Per-channel background estimate across frames. Stable, parallel lines indicate consistent acquisition conditions.</p>"
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
            "<p>Global background estimate per frame. Drift or spikes usually track clouds, gradients, or transparency changes.</p>"
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
        {svg_timeseries(bg, "Frame background level", "background"), "<p>Background level per frame.</p>"},
        {svg_timeseries(noise, "Frame noise level", "noise", "#f87171"), "<p>Estimated frame noise. Peaks usually indicate bad transparency or poor signal.</p>"},
        {svg_timeseries(grad, "Frame gradient energy", "gradient", "#4ade80"), "<p>Detail/edge energy over time.</p>"},
        {svg_timeseries(gw, "Global frame weight", "weight", "#fbbf24"), "<p>Combined frame quality weight used by the pipeline.</p>"},
        {svg_histogram(gw, "Global weight distribution", "weight", "#fbbf24"), "<p>Distribution of global weights. A wide left tail means many weak frames.</p>"},
        {svg_timeseries(fwhm, "FWHM per frame", "FWHM (px)", "#c084fc"), "<p>Frame sharpness proxy from star FWHM.</p>"},
        {svg_timeseries(roundness, "Roundness per frame", "roundness", "#22d3ee"), "<p>Roundness close to 1 indicates near-circular stars.</p>"},
        {svg_timeseries(star_count, "Detected stars per frame", "stars", "#fde047"), "<p>Number of detected stars per frame.</p>"},
        {svg_scatter(fwhm, roundness, star_count, "FWHM vs roundness", "FWHM (px)", "roundness"), "<p>Frames cluster best when FWHM stays low and roundness stays near 1.</p>"},
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

    return ReportSection{"Global Metrics", make_card_html("Frame quality and weights", charts, evals, infer_status(evals))};
}

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
        "<p>Tile placement used for local weighting and reconstruction.</p>"
    }};
    return ReportSection{"Tile Grid", make_card_html("Grid layout", charts, evals, "ok")};
}

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
        {svg_scatter(tx, ty, ccs, "Translation scatter", "tx (px)", "ty (px)"), "<p>Translation spread relative to the reference frame.</p>"},
        {svg_multi_timeseries({{"tx", tx}, {"ty", ty}}, "Translation over time", "shift (px)"), "<p>Frame-to-frame translation drift over time.</p>"},
        {svg_histogram(ccs, "Registration CC distribution", "CC", "#4ade80"), "<p>Correlation coefficient distribution for registration quality.</p>"},
        {svg_timeseries(rotations, "Rotation angle", "deg", "#f87171"), "<p>Rotation angle of each frame relative to the reference frame.</p>"},
        {svg_timeseries(scales, "Scale factor", "scale", "#fbbf24"), "<p>Scale drift can reveal focus or geometry instability.</p>"},
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
                          "<p>Spatial distribution of average tile FWHM.</p>"});
        charts.push_back({svg_spatial_tile_heatmap(tiles, mean_quality, img_w, img_h, "Mean quality score per tile", "quality", "viridis"),
                          "<p>Average tile quality score over all frames.</p>"});
        charts.push_back({svg_spatial_tile_heatmap(tiles, mean_weight, img_w, img_h, "Mean local weight per tile", "weight", "plasma"),
                          "<p>Average local weight per tile used during reconstruction.</p>"});
        charts.push_back({svg_spatial_tile_heatmap(tiles, mean_stars, img_w, img_h, "Mean stars per tile", "stars", "YlGnBu"),
                          "<p>Average star count per tile.</p>"});
        if (have_tile_types) {
            charts.push_back({svg_spatial_tile_heatmap(tiles, tile_type_map, img_w, img_h, "Tile type map", "STAR=1", "viridis"),
                              "<p>Tile classification in the first stored metric frame. Bright tiles correspond to STAR mode.</p>"});
        }
    }
    charts.push_back({
        svg_multi_timeseries({{"mean quality", per_frame_quality}, {"mean weight", per_frame_weight}},
                             "Per-frame tile quality and weight", "value"),
        "<p>Frame-level averages across all local tiles.</p>"
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
                                                     "<p>Frame usage count per tile.</p>"});
        if (!mean_cc.empty()) charts.push_back({svg_spatial_tile_heatmap(tiles, mean_cc, img_w, img_h, "Mean correlation per tile", "CC", "viridis"),
                                                "<p>Average correlation of contributing frames per tile.</p>"});
        if (!post_snr.empty()) charts.push_back({svg_spatial_tile_heatmap(tiles, post_snr, img_w, img_h, "Post-reconstruction SNR", "SNR", "plasma"),
                                                 "<p>Tile-wise post-reconstruction SNR proxy.</p>"});
        if (!post_contrast.empty()) charts.push_back({svg_spatial_tile_heatmap(tiles, post_contrast, img_w, img_h, "Post contrast per tile", "contrast", "cividis"),
                                                      "<p>Tile-wise contrast after reconstruction.</p>"});
        if (!post_bg.empty()) charts.push_back({svg_spatial_tile_heatmap(tiles, post_bg, img_w, img_h, "Post background per tile", "background", "gray"),
                                                "<p>Tile-wise background after reconstruction.</p>"});
    }
    if (!valid_counts.empty()) charts.push_back({svg_histogram(valid_counts, "Valid frame count distribution", "valid frames", "#4ade80"),
                                                 "<p>Distribution of tile contribution counts.</p>"});
    if (!mean_cc.empty()) charts.push_back({svg_histogram(mean_cc, "Mean correlation distribution", "CC", "#60a5fa"),
                                            "<p>Distribution of reconstructed tile correlations.</p>"});
    if (!post_snr.empty()) charts.push_back({svg_histogram(post_snr, "Post-reconstruction SNR distribution", "SNR", "#fbbf24"),
                                             "<p>Distribution of the tile SNR proxy after reconstruction.</p>"});

    std::vector<std::string> evals = {
        "frames: " + json_string_or(recon, "num_frames", "?") + ", tiles: " + json_string_or(recon, "num_tiles", "?")
    };
    if (!valid_counts.empty()) {
        const auto s = basic_stats(valid_counts);
        evals.push_back("valid counts: median=" + format_number(s.median, 0) +
                        ", min=" + format_number(s.min, 0) +
                        ", max=" + format_number(s.max, 0));
        int low = 0;
        for (double v : valid_counts) if (std::isfinite(v) && v < 3.0) ++low;
        if (low > 0) evals.push_back("WARNING: " + std::to_string(low) + " tiles with < 3 valid frames");
    }
    if (!mean_cc.empty()) {
        const auto s = basic_stats(mean_cc);
        evals.push_back("tile CC: median=" + format_number(s.median, 4) +
                        ", min=" + format_number(s.min, 4));
    }
    if (!post_snr.empty()) {
        const auto s = basic_stats(post_snr);
        evals.push_back("post-SNR: median=" + format_number(s.median, 3) +
                        ", min=" + format_number(s.min, 3));
    }

    return ReportSection{"Tile Reconstruction", make_card_html("Reconstruction statistics", charts, evals, infer_status(evals))};
}

std::optional<ReportSection> gen_clustering(const json& cl) {
    if (!cl.is_object() || !cl.contains("cluster_sizes") || !cl["cluster_sizes"].is_array()) return std::nullopt;
    const auto sizes = json_double_array(cl.value("cluster_sizes", json::array()));
    if (sizes.empty()) return std::nullopt;
    std::vector<std::string> labels;
    labels.reserve(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) labels.push_back("C" + std::to_string(i));
    const auto cluster_labels = json_double_array(cl.value("cluster_labels", json::array()));

    std::vector<ChartBlock> charts = {
        {svg_bar(labels, sizes, "Cluster sizes", "frames"), "<p>Number of frames per clustering bucket.</p>"}
    };
    if (!cluster_labels.empty()) {
        charts.push_back({svg_timeseries(cluster_labels, "Cluster label over time", "cluster", "#60a5fa", false),
                          "<p>Cluster assignment of each frame over time.</p>"});
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
                          "<p>Quality scores of the synthetic frames or source clusters.</p>"});
    }
    return ReportSection{"Synthetic Frames", make_card_html("Synthetic frame summary", charts, evals, "ok")};
}

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
        charts.push_back({svg_bar(labels, mean_shifts, "BGE mean shift", "shift"), "<p>Additive mean shift per channel.</p>"});
        charts.push_back({svg_bar(labels, residual_stds, "BGE residual std", "std"), "<p>Residual spread at the BGE support points.</p>"});
        charts.push_back({svg_bar(labels, valid_ratios, "Valid tile-sample ratio", "ratio"), "<p>Fraction of valid BGE samples per channel.</p>"});
    }

    return ReportSection{"Background Gradient Extraction (BGE)", make_card_html("BGE diagnostics", charts, evals, infer_status(evals))};
}

std::optional<ReportSection> gen_validation(const json& val) {
    if (!val.is_object() || val.empty()) return std::nullopt;
    const double improvement = json_number_or(val, "fwhm_improvement_percent", 0.0);
    const double tw_var = json_number_or(val, "tile_weight_variance", 0.0);
    const double pattern_ratio = json_number_or(val, "tile_pattern_ratio", 0.0);
    const bool fwhm_ok = json_bool_or(val, "fwhm_improvement_ok", false);
    const bool tw_ok = json_bool_or(val, "tile_weight_variance_ok", false);
    const bool pattern_ok = json_bool_or(val, "tile_pattern_ok", false);

    std::vector<std::string> labels;
    std::vector<double> values;
    std::vector<std::string> colors;
    labels.push_back("FWHM improvement");
    values.push_back(improvement);
    colors.push_back(fwhm_ok ? "#4ade80" : "#f87171");
    labels.push_back("Tile weight variance");
    values.push_back(tw_var * 100.0);
    colors.push_back(tw_ok ? "#4ade80" : "#f87171");
    if (val.contains("tile_pattern_ratio")) {
        labels.push_back("Tile pattern ratio");
        values.push_back(pattern_ratio);
        colors.push_back(pattern_ok ? "#4ade80" : "#f87171");
    }

    std::vector<ChartBlock> charts = {{
        svg_bar(labels, values, "Validation checks", "value", colors),
        "<p>Green bars passed their check, red bars failed.</p>"
    }};
    std::vector<std::string> evals = {
        "seeing FWHM: " + json_string_or(val, "seeing_fwhm_median", "?"),
        "output FWHM: " + json_string_or(val, "output_fwhm_median", "?"),
        "FWHM improvement: " + format_number(improvement, 1) + "% " + (fwhm_ok ? std::string("OK") : std::string("FAIL")),
        "tile weight variance: " + format_number(tw_var, 4) + " " + (tw_ok ? std::string("OK") : std::string("FAIL"))
    };
    if (val.contains("tile_pattern_ratio")) {
        evals.push_back("tile pattern ratio: " + format_number(pattern_ratio, 3) + " " + (pattern_ok ? std::string("OK") : std::string("FAIL")));
    }
    return ReportSection{"Validation", make_card_html("Quality validation", charts, evals, infer_status(evals))};
}

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
        {svg_histogram(ratios, "Tile common-overlap ratio", "common ratio", "#22d3ee"), "<p>Distribution of common overlap across tiles.</p>"}
    };
    const int img_w = static_cast<int>(json_number_or(co, "canvas_width", 0.0));
    const int img_h = static_cast<int>(json_number_or(co, "canvas_height", 0.0));
    if (img_w > 0 && img_h > 0) {
        charts.push_back({
            svg_spatial_tile_heatmap(co["tiles"], ratios, img_w, img_h, "Spatial common-overlap ratio", "common ratio", "viridis"),
            "<p>Spatial overlap coverage after prewarp and registration.</p>"
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
                              const json& common_overlap,
                              const std::string& config_yaml) {
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
    add(gen_clustering(cl));
    add(gen_synthetic(syn));
    add(gen_bge(bge));
    add(gen_validation(val));
    add(gen_common_overlap(common_overlap));

    std::ostringstream html;
    html << "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
         << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
         << "<title>" << html_escape("Tile-Compile Report - " + run_dir.filename().string()) << "</title>"
         << "<style>"
         << ":root{color-scheme:dark;--bg:#020617;--panel:#0f172a;--panel2:#111827;--line:#334155;--text:#e2e8f0;--muted:#94a3b8;--good:#4ade80;--warn:#fbbf24;--bad:#f87171;}"
         << "*{box-sizing:border-box;}body{margin:0;background:radial-gradient(circle at top,#0f172a,#020617 60%);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,sans-serif;}"
         << "header{padding:32px 28px 18px;border-bottom:1px solid rgba(148,163,184,.16);background:linear-gradient(180deg,rgba(15,23,42,.88),rgba(2,6,23,.96));position:sticky;top:0;backdrop-filter:blur(10px);z-index:5;}"
         << "header h1{margin:0 0 8px;font-size:28px;}header .meta{color:var(--muted);font-size:13px;display:flex;flex-wrap:wrap;gap:8px 16px;}"
         << "main{padding:24px 22px 34px;max-width:1600px;margin:0 auto;}section{margin:0 0 28px;}section h2{margin:0 0 14px;font-size:19px;}"
         << ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;align-items:start;}"
         << ".card{background:linear-gradient(180deg,rgba(17,24,39,.96),rgba(15,23,42,.96));border:1px solid rgba(148,163,184,.15);border-radius:16px;padding:16px 16px 12px;box-shadow:0 8px 28px rgba(0,0,0,.18);}"
         << ".card h3{margin:0 0 12px;font-size:16px;display:flex;align-items:center;gap:10px;}"
         << ".badge{display:inline-flex;align-items:center;border-radius:999px;padding:2px 8px;font-size:11px;text-transform:uppercase;border:1px solid currentColor;}"
         << ".badge.ok,.card.ok .badge{color:var(--good);} .badge.warn,.card.warn .badge{color:var(--warn);} .badge.bad,.card.bad .badge{color:var(--bad);}"
         << ".chart-row{display:grid;grid-template-columns:minmax(0,1fr);gap:10px;margin:0 0 14px;}"
         << ".explain-col{color:var(--muted);font-size:13px;} .explain-col p{margin:0;}"
         << ".metric-box{border-top:1px solid rgba(148,163,184,.12);padding-top:10px;margin-top:6px;} .metric-box ul{margin:0;padding-left:18px;} .metric-box li{margin:3px 0;} .metric-box li.warn{color:#fecaca;}"
         << ".kv,.phases{width:100%;border-collapse:collapse;} .kv th,.kv td,.phases th,.phases td{padding:8px 10px;border-bottom:1px solid rgba(148,163,184,.12);text-align:left;vertical-align:top;} .kv th,.phases th{color:#cbd5e1;width:36%;font-weight:600;}"
         << ".artifact-list{margin:0;padding-left:18px;} .artifact-list li{margin:3px 0;} .muted{color:var(--muted);} code,pre{font-family:ui-monospace,SFMono-Regular,monospace;} pre{margin:0;white-space:pre-wrap;overflow:auto;background:#020617;border:1px solid rgba(148,163,184,.14);border-radius:12px;padding:12px;}"
         << "details.config{margin-top:26px;background:rgba(15,23,42,.92);border:1px solid rgba(148,163,184,.15);border-radius:14px;padding:14px 16px;} details summary{cursor:pointer;font-weight:600;}"
         << ".footer{margin-top:18px;color:var(--muted);font-size:12px;}"
         << "svg.report-chart{width:100%;height:auto;display:block;} .svg-title{fill:#e2e8f0;font-size:14px;font-weight:700;} .svg-title-small{fill:#e2e8f0;font-size:18px;font-weight:700;} .svg-label{fill:#94a3b8;font-size:11px;} .svg-note{fill:#94a3b8;font-size:13px;} .svg-tick{fill:#94a3b8;font-size:10px;} .svg-axis{stroke:#64748b;stroke-width:1;} .svg-grid{stroke:#1e293b;stroke-width:1;}"
         << "@media (min-width:1100px){.chart-row{grid-template-columns:minmax(0,1fr) 270px;align-items:start;}}"
         << "</style></head><body>";
    html << "<header><h1>Tile-Compile Report</h1><div class=\"meta\">";
    for (const auto& line : meta_lines) html << "<span>" << html_escape(line) << "</span>";
    html << "</div></header><main>";
    for (const auto& section : sections) {
        html << "<section><h2>" << html_escape(section.title) << "</h2><div class=\"grid\">" << section.cards_html << "</div></section>";
    }
    if (!config_yaml.empty()) {
        html << "<details class=\"config\"><summary>Config (config.yaml)</summary><pre>" << html_escape(config_yaml) << "</pre></details>";
    }
    html << "<div class=\"footer\">Generated by tile_compile_web_backend (C++ inline SVG report)</div>";
    html << "</main></body></html>";
    return html.str();
}

} // namespace

nlohmann::json generate_run_report(const fs::path& run_dir) {
    try {
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
        const json recon = read_json_if_exists(artifacts_dir / "tile_reconstruction.json");
        const json cl = read_json_if_exists(artifacts_dir / "state_clustering.json");
        const json syn = read_json_if_exists(artifacts_dir / "synthetic_frames.json");
        const json bge = read_json_if_exists(artifacts_dir / "bge.json");
        const json val = read_json_if_exists(artifacts_dir / "validation.json");
        const json common_overlap = read_json_if_exists(artifacts_dir / "common_overlap.json");
        const std::string config_yaml = read_text(run_dir / "config.yaml");

        const std::string report_html = build_report_html(run_dir, status, artifacts_before, events,
                                                          norm, gm, tg, reg, lm, recon, cl,
                                                          syn, bge, val, common_overlap, config_yaml);

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
