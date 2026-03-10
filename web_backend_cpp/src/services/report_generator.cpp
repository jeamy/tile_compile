#include "services/report_generator.hpp"
#include "services/run_inspector.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

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
    try {
        return json::parse(in);
    } catch (...) {
        return json::object();
    }
}

std::vector<json> read_jsonl_if_exists(const fs::path& path, int max_lines = 4000) {
    std::ifstream in(path);
    std::vector<json> items;
    if (!in) return items;
    std::string line;
    int n = 0;
    while (std::getline(in, line) && n < max_lines) {
        if (line.empty()) continue;
        try {
            auto j = json::parse(line);
            if (j.is_object()) items.push_back(std::move(j));
        } catch (...) {}
        ++n;
    }
    return items;
}

std::string format_number(double v, int prec = 3) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

std::string render_kv_table(const std::vector<std::pair<std::string, std::string>>& rows) {
    std::ostringstream html;
    html << "<table class=\"kv\"><tbody>";
    for (const auto& row : rows) {
        html << "<tr><th>" << html_escape(row.first) << "</th><td>" << html_escape(row.second) << "</td></tr>";
    }
    html << "</tbody></table>";
    return html.str();
}

std::string render_artifacts_list(const json& artifacts) {
    std::ostringstream html;
    html << "<ul class=\"artifacts\">";
    for (const auto& item : artifacts) {
        const std::string path = item.value("path", "");
        const auto size = item.value("size", 0LL);
        html << "<li><code>" << html_escape(path) << "</code>"
             << " <span class=\"muted\">(" << size << " bytes)</span></li>";
    }
    html << "</ul>";
    return html.str();
}

std::string render_events_summary(const std::vector<json>& events) {
    std::map<std::string, int> counts;
    for (const auto& ev : events) {
        std::string type = ev.value("type", "unknown");
        counts[type] += 1;
    }
    std::vector<std::pair<std::string, std::string>> rows;
    for (const auto& kv : counts) rows.push_back({kv.first, std::to_string(kv.second)});
    if (rows.empty()) rows.push_back({"events", "0"});
    return render_kv_table(rows);
}

std::string render_phase_summary(const json& status) {
    if (!status.contains("phases") || !status["phases"].is_array()) {
        return "<p class=\"muted\">Keine Phaseninformationen vorhanden.</p>";
    }
    std::ostringstream html;
    html << "<table class=\"phases\"><thead><tr><th>Phase</th><th>Status</th><th>Fortschritt</th></tr></thead><tbody>";
    for (const auto& p : status["phases"]) {
        html << "<tr><td>" << html_escape(p.value("phase", "")) << "</td>"
             << "<td>" << html_escape(p.value("status", "")) << "</td>"
             << "<td>" << html_escape(format_number(p.value("pct", 0.0), 1)) << "%</td></tr>";
    }
    html << "</tbody></table>";
    return html.str();
}

std::string build_report_html(const fs::path& run_dir,
                              const json& status,
                              const json& artifacts,
                              const std::vector<json>& events,
                              const std::string& log_tail,
                              const std::string& config_yaml) {
    std::vector<std::pair<std::string, std::string>> summary_rows = {
        {"Run-ID", run_dir.filename().string()},
        {"Status", status.value("status", "unknown")},
        {"Aktuelle Phase", status.value("current_phase", "")},
        {"Fortschritt", format_number(status.value("progress", 0.0), 1) + "%"},
        {"Run-Verzeichnis", run_dir.string()},
    };

    std::ostringstream html;
    html << "<!doctype html><html lang=\"de\"><head><meta charset=\"utf-8\">"
         << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
         << "<title>tile_compile Report - " << html_escape(run_dir.filename().string()) << "</title>"
         << "<style>"
         << "body{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px;}"
         << "h1,h2{margin:0 0 12px 0;}"
         << ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-top:16px;}"
         << ".card{background:#111827;border:1px solid #334155;border-radius:12px;padding:16px;}"
         << ".kv,.phases{width:100%;border-collapse:collapse;}"
         << ".kv th,.kv td,.phases th,.phases td{padding:8px;border-bottom:1px solid #334155;text-align:left;vertical-align:top;}"
         << ".muted{color:#94a3b8;}"
         << "code,pre{font-family:ui-monospace,SFMono-Regular,monospace;}"
         << "pre{white-space:pre-wrap;background:#020617;border:1px solid #1e293b;border-radius:8px;padding:12px;overflow:auto;}"
         << "ul.artifacts{margin:0;padding-left:20px;}"
         << "a{color:#93c5fd;}"
         << "</style></head><body>";

    html << "<h1>tile_compile Report</h1>";
    html << "<p class=\"muted\">Automatisch vom Crow/C++ Backend erzeugt.</p>";
    html << "<div class=\"grid\">";

    html << "<section class=\"card\"><h2>Zusammenfassung</h2>";
    html << render_kv_table(summary_rows);
    html << "</section>";

    html << "<section class=\"card\"><h2>Phasen</h2>";
    html << render_phase_summary(status);
    html << "</section>";

    html << "<section class=\"card\"><h2>Event-Statistik</h2>";
    html << render_events_summary(events);
    html << "</section>";

    html << "<section class=\"card\"><h2>Artefakte</h2>";
    html << render_artifacts_list(artifacts);
    html << "</section>";

    html << "<section class=\"card\"><h2>Konfiguration</h2><pre>" << html_escape(config_yaml) << "</pre></section>";
    html << "<section class=\"card\"><h2>Letzte Log-Zeilen</h2><pre>" << html_escape(log_tail) << "</pre></section>";

    html << "</div></body></html>";
    return html.str();
}

} // namespace

nlohmann::json generate_run_report(const fs::path& run_dir) {
    const fs::path artifacts_dir = run_dir / "artifacts";
    fs::create_directories(artifacts_dir);

    const auto status = read_run_status(run_dir);
    const auto artifacts = list_run_artifacts(run_dir);
    const auto log_tail = read_run_logs(run_dir, 400);

    std::string config_yaml = read_text(run_dir / "config.yaml");
    if (config_yaml.empty()) config_yaml = read_text(run_dir / "config.yml");
    if (config_yaml.empty()) config_yaml = read_text(run_dir / "tile_compile.yaml");

    std::vector<json> events;
    for (const auto& candidate : {
        run_dir / "logs" / "run_events.jsonl",
        run_dir / "events.jsonl",
        run_dir / "logs" / "events.jsonl"
    }) {
        events = read_jsonl_if_exists(candidate);
        if (!events.empty()) break;
    }

    const fs::path report_path = artifacts_dir / "report.html";
    const std::string html = build_report_html(run_dir, status, artifacts, events, log_tail, config_yaml);

    std::ofstream out(report_path, std::ios::binary);
    if (!out) {
        return {
            {"ok", false},
            {"error", "cannot write report.html"},
            {"report_path", report_path.string()},
        };
    }
    out << html;

    return {
        {"ok", true},
        {"run_id", run_dir.filename().string()},
        {"output_dir", artifacts_dir.string()},
        {"report_path", report_path.string()},
        {"artifact_count", artifacts.size()},
        {"event_count", events.size()},
    };
}
