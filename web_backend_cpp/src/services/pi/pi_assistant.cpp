#include "services/pi/pi_assistant.hpp"

#include "services/pi/pi_tool_registry.hpp"

#include <algorithm>
#include <cctype>

namespace tile_compile::pi {
namespace {

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool contains_any(const std::string& haystack, std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (haystack.find(needle) != std::string::npos) return true;
    }
    return false;
}

} // namespace

PiAssistant::PiAssistant(std::shared_ptr<AppState> state)
    : _state(std::move(state)) {}

nlohmann::json PiAssistant::answer(const std::string& question) const {
    const std::string normalized = lower_copy(question);
    PiToolRegistry tools(_state);

    nlohmann::json evidence = nlohmann::json::array();
    std::string answer;

    if (contains_any(normalized, {"schema", "parameter", "config", "konfig"})) {
        auto schema = tools.call_tool("config.schema.summary", nlohmann::json::object());
        evidence.push_back({{"tool", "config.schema.summary"}, {"result", schema["result"]}});
        const auto& result = schema["result"];
        if (result.value("available", false)) {
            answer = "Das Config-Schema ist verfuegbar. Es enthaelt " +
                std::to_string(result.value("top_level_count", 0)) +
                " Top-Level-Bereiche. Fuer konkrete Parameter sollte der naechste Ausbau ein Detail-Tool pro Pfad bereitstellen.";
        } else {
            answer = "Das Config-Schema ist aktuell nicht verfuegbar: " + result.value("error", std::string("unknown"));
        }
    } else if (contains_any(normalized, {"scan", "frames", "frame", "aufnahme"})) {
        auto context = tools.call_tool("context.overview", nlohmann::json::object());
        evidence.push_back({{"tool", "context.overview"}, {"result", context["result"]}});
        const auto& latest = context["result"]["latest_scan_job"];
        if (!latest.is_null() && latest.contains("result")) {
            const auto& scan = latest["result"];
            answer = "Der letzte Scan ist im Kontext verfuegbar: " +
                std::to_string(scan.value("frames_detected", scan.value("frames_total", 0))) +
                " Frames, Farbmodus " + scan.value("color_mode", std::string("UNKNOWN")) + ".";
        } else {
            answer = "Es ist noch kein abgeschlossener Scan im PI-Kontext verfuegbar.";
        }
    } else if (contains_any(normalized, {"report", "artifact", "artefakt", "phase", "fehler", "warning", "warnung"})) {
        auto report = tools.call_tool("run.report.summary", nlohmann::json::object());
        auto artifacts = tools.call_tool("run.artifacts.summary", nlohmann::json::object());
        evidence.push_back({{"tool", "run.report.summary"}, {"result", report.value("result", nlohmann::json::object())}});
        evidence.push_back({{"tool", "run.artifacts.summary"}, {"result", artifacts.value("result", nlohmann::json::object())}});
        if (report.value("ok", false)) {
            const auto& result = report["result"];
            const std::string status = result["status"].value("status", std::string("unknown"));
            const int artifact_count = result["artifact_summary"].value("count", 0);
            answer = "Der ausgewaehlte Run hat Status " + status + ". Es sind " +
                std::to_string(artifact_count) +
                " Artefakte im read-only Kontext sichtbar. Report-Stats sind " +
                std::string(result.value("report_stats_available", false) ? "verfuegbar." : "noch nicht verfuegbar.");
        } else {
            answer = "Fuer Report-/Artefaktfragen ist aktuell kein Run-Kontext verfuegbar.";
        }
    } else if (contains_any(normalized, {"run", "job", "status"})) {
        auto context = tools.call_tool("context.overview", nlohmann::json::object());
        evidence.push_back({{"tool", "context.overview"}, {"result", context["result"]}});
        const auto& jobs = context["result"]["jobs"];
        answer = "Der PI-Kontext enthaelt " +
            std::to_string(jobs.is_array() ? jobs.size() : 0) +
            " aktuelle Job-Summaries. Detailanalyse von Run-Reports ist als naechstes Tool vorgesehen.";
    } else {
        auto context = tools.call_tool("context.overview", nlohmann::json::object());
        evidence.push_back({{"tool", "context.overview"}, {"result", context["result"]}});
        answer = "Diese lokale PI-Assistant-Stufe kann aktuell Schema-, Config-, Scan- und Job-Kontext zusammenfassen. Fuer freie Diagnosefragen braucht der naechste Ausbau Report-/Artefakt-Tools und optional den PI Sidecar.";
    }

    return {
        {"schema_version", "pi.assistant-answer.v1"},
        {"mode", "local_read_only"},
        {"question", question},
        {"answer", answer},
        {"evidence", evidence}
    };
}

} // namespace tile_compile::pi
