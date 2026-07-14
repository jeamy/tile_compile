#include "routes/pi_routes.hpp"

#include "app_state.hpp"
#include "routes/route_utils.hpp"
#include "services/pi/pi_assistant.hpp"
#include "services/pi/pi_context_builder.hpp"
#include "services/pi/pi_action_validator.hpp"
#include "services/pi/pi_memory_store.hpp"
#include "services/pi/pi_storage_paths.hpp"
#include "services/pi/pi_tool_registry.hpp"
#include "subprocess_manager.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <yaml-cpp/yaml.h>

using namespace tile_compile::routes;

namespace {

std::optional<nlohmann::json> parse_body(const crow::request& req) {
    if (req.body.empty()) return nlohmann::json::object();
    auto parsed = nlohmann::json::parse(req.body, nullptr, false);
    if (parsed.is_discarded()) return std::nullopt;
    if (!parsed.is_object()) return nlohmann::json::object();
    return parsed;
}

int int_query_param(const crow::request& req, const char* name, int fallback) {
    const char* raw = req.url_params.get(name);
    if (!raw) return fallback;
    try {
        return std::stoi(raw);
    } catch (...) {
        return fallback;
    }
}

void trim_json_array_to_latest(nlohmann::json& items, int limit) {
    if (!items.is_array() || limit <= 0) return;
    while (static_cast<int>(items.size()) > limit) items.erase(items.begin());
}

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

nlohmann::json detect_run_chat_problem_hints(const std::string& message) {
    const std::string text = lower_copy(message);
    nlohmann::json hints = nlohmann::json::array();
    auto add = [&](const std::string& id, const std::string& label, const std::string& confidence) {
        hints.push_back({{"id", id}, {"label", label}, {"confidence", confidence}});
    };
    if (contains_any(text, {"schwarzen kern", "schwarzer kern", "black core", "black cores", "donut", "sternkern"})) {
        add("black_star_cores", "Sterne mit dunklem/schwarzem Kern", "high");
    }
    if (contains_any(text, {"beschnitten", "abgeschnitten", "cropped", "crop", "nicht einbezogen", "outside", "rand"})) {
        add("cropped_nebula", "Nebel oder Randstruktur wirkt beschnitten", "high");
    }
    if (contains_any(text, {"kaum sichtbar", "zu dunkel", "dunkel", "faint", "too dark", "nebula not visible", "nebel"})) {
        add("faint_nebula", "Nebelanteile sind zu schwach sichtbar", "medium");
    }
    if (contains_any(text, {"gradient", "hintergrund", "background", "vignette", "vignett"})) {
        add("background_gradient", "Hintergrundgradient oder Vignettierung", "medium");
    }
    if (contains_any(text, {"farbstich", "gruen", "magenta", "color cast", "colour cast", "farbe"})) {
        add("color_cast", "Farbstich oder unausgewogene Farbe", "medium");
    }
    if (contains_any(text, {"tile", "kachel", "muster", "pattern", "seam", "naht"})) {
        add("tile_pattern", "Tile-/Kachelmuster sichtbar", "medium");
    }
    if (contains_any(text, {"unscharf", "blur", "soft", "fwhm", "elongated", "eier", "verzogen"})) {
        add("soft_or_elongated_stars", "Sterne unscharf oder verzogen", "medium");
    }
    if (hints.empty()) {
        add("general_quality_issue", "Allgemeines sichtbares Qualitaetsproblem", "low");
    }
    return hints;
}

nlohmann::json append_text_item(const std::string& text, const std::string& evidence = "") {
    nlohmann::json item = {{"text", text}};
    if (!evidence.empty()) item["evidence_ref"] = evidence;
    return item;
}

nlohmann::json build_run_chat_action_plan(const std::string& run_id, const nlohmann::json& hints) {
    nlohmann::json actions = nlohmann::json::array();
    int index = 1;
    auto add_set = [&](const std::string& path, const nlohmann::json& value, const std::string& rationale) {
        actions.push_back({
            {"id", "run_chat_" + std::to_string(index++)},
            {"type", "config.set"},
            {"path", path},
            {"value", value},
            {"rationale", rationale}
        });
    };

    for (const auto& hint : hints) {
        const std::string id = hint.value("id", std::string());
        if (id == "cropped_nebula") {
            add_set("output.crop_to_nonzero_bbox", false,
                    "Wenn Nebel am Rand abgeschnitten wirkt, zuerst ohne automatisches Crop testen.");
        } else if (id == "faint_nebula") {
            add_set("bge.enabled", false,
                    "Bei ausgedehntem Nebel kann Hintergrundextraktion echte schwache Nebelanteile abschwaechen.");
            add_set("normalization.mode", "median",
                    "Median-Normalisierung ist fuer ausgedehnte Nebel oft konservativer als Hintergrund-Normalisierung.");
        } else if (id == "black_star_cores") {
            add_set("stacking.cosmetic_correction", false,
                    "Dunkle Sternkerne koennen durch zu aggressive kosmetische Korrektur/Rejection entstehen; als A/B-Test deaktivieren.");
        } else if (id == "tile_pattern") {
            add_set("tile.overlap_fraction", 0.35,
                    "Mehr Tile-Overlap kann sichtbare Kacheluebergaenge reduzieren.");
        }
    }

    return {
        {"schema_version", "pi.action-plan.v1"},
        {"source", "pi.run-chat"},
        {"run_id", run_id},
        {"mutation_free", true},
        {"actions", actions}
    };
}

nlohmann::json build_run_chat_answer(const std::shared_ptr<AppState>& state,
                                     const std::string& run_id,
                                     const std::string& message) {
    tile_compile::pi::PiToolRegistry tools(state);
    nlohmann::json report = tools.call_tool("run.report.summary", {{"run_id", run_id}});
    nlohmann::json artifacts = tools.call_tool("run.artifacts.summary", {{"run_id", run_id}});
    const nlohmann::json hints = detect_run_chat_problem_hints(message);

    tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
    nlohmann::json memories = store.retrieve({{"type", "config_optimization"}}, 5);

    nlohmann::json evidence = nlohmann::json::array({
        {{"id", "report"}, {"tool", "run.report.summary"}, {"available", report.value("ok", false)}, {"result", report.value("result", nlohmann::json::object())}},
        {{"id", "artifacts"}, {"tool", "run.artifacts.summary"}, {"available", artifacts.value("ok", false)}, {"result", artifacts.value("result", nlohmann::json::object())}},
        {{"id", "memories"}, {"tool", "pi.memory.retrieve"}, {"available", !memories.empty()}, {"result", memories}}
    });

    nlohmann::json likely_causes = nlohmann::json::array();
    nlohmann::json checks = nlohmann::json::array();
    nlohmann::json recommendations = nlohmann::json::array();

    for (const auto& hint : hints) {
        const std::string id = hint.value("id", std::string());
        if (id == "black_star_cores") {
            likely_causes.push_back(append_text_item(
                "Dunkle Sternkerne passen zu zu aggressiver kosmetischer Korrektur, Sigma-Rejection, lokaler Hintergrundbehandlung oder Stretch/Star-Protect-Artefakten.",
                "report"));
            checks.push_back(append_text_item(
                "Vergleiche lineares Stack, gestretchtes Ergebnis und ggf. Zwischenergebnisse vor/nach kosmetischer Korrektur und Rejection.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "Als A/B-Test kosmetische Korrektur oder Rejection weniger aggressiv setzen und nur ab betroffener Phase neu rechnen.",
                "report"));
        } else if (id == "cropped_nebula") {
            likely_causes.push_back(append_text_item(
                "Beschnittener Nebel deutet auf Crop-to-nonzero-BBox, Common-Overlap nach Registrierung oder ein zu enges gueltiges Rekonstruktionsfenster hin.",
                "artifacts"));
            checks.push_back(append_text_item(
                "Pruefe common_overlap, Registration-Artefakte und ob das finale Output kleiner als die registrierten Frames ist.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "Testweise `output.crop_to_nonzero_bbox=false` und Common-Overlap/Registrierungsdiagnostik pruefen.",
                "report"));
        } else if (id == "faint_nebula") {
            likely_causes.push_back(append_text_item(
                "Schwacher Nebel kann durch Hintergrundextraktion, Hintergrund-Normalisierung oder zu dunklen Stretch-Zielhintergrund entstehen.",
                "report"));
            checks.push_back(append_text_item(
                "Pruefe BGE-Report, Hintergrundkarten, Histogramm/Stretch-Parameter und ob ausgedehnte Emission als Hintergrund behandelt wurde.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "Bei M42/ausgedehnten Nebeln konservativ testen: BGE aus, Median-Normalisierung, danach Stretch neu bewerten.",
                "memories"));
        } else if (id == "background_gradient") {
            likely_causes.push_back(append_text_item(
                "Gradienten koennen aus Vignettierung, Mond/Light-Pollution, fehlenden Flats oder BGE-Unter-/Ueberfit stammen.",
                "report"));
            checks.push_back(append_text_item(
                "BGE-Diagnostik und Flat-/Kalibrierstatus pruefen; nicht automatisch Nebel als Gradient wegfitten.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "BGE nur mit konservativen Masken/Validierung verwenden und Ergebnis gegen BGE-off vergleichen.",
                "report"));
        } else if (id == "color_cast") {
            likely_causes.push_back(append_text_item(
                "Farbstich passt zu Bayer-Pattern, PCC-Sternauswahl, Hintergrundneutralisierung oder starker Gradientenbehandlung.",
                "report"));
            checks.push_back(append_text_item(
                "PCC-Report, Bayer-Pattern und Background-Neutralization-Status pruefen.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "PCC-Parameter und Bayer-Pattern erst anhand Report/Headers bestaetigen, dann gezielt neu rechnen.",
                "report"));
        } else if (id == "tile_pattern") {
            likely_causes.push_back(append_text_item(
                "Tile-Muster spricht fuer zu wenig Overlap, zu starke lokale Gewichtung oder inkonsistente lokale Rekonstruktion.",
                "report"));
            checks.push_back(append_text_item(
                "AQMH-/Tile-Artefakte, lokale Metrikkarten und Rekonstruktionsdiagnostik pruefen.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "Tile-Overlap erhoehen und lokale Regularisierung/Tile-Groesse gegenpruefen.",
                "report"));
        } else if (id == "soft_or_elongated_stars") {
            likely_causes.push_back(append_text_item(
                "Weiche oder verzogene Sterne passen zu Registrierungsfehlern, Seeing-Streuung, Fokusdrift oder falscher Frame-Gewichtung.",
                "report"));
            checks.push_back(append_text_item(
                "Registration-Report, FWHM-Verlauf und verworfene/gewichtete Frames pruefen.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "Registration und Qualitätsgewichtung vor Stretch/Color-Fixes validieren.",
                "report"));
        }
    }

    if (recommendations.empty()) {
        recommendations.push_back(append_text_item(
            "Zuerst Report und Artefakte pruefen, dann nur eine Parametergruppe als A/B-Test aendern.",
            "report"));
    }

    const nlohmann::json action_plan = build_run_chat_action_plan(run_id, hints);
    return {
        {"schema_version", "pi.run-chat-answer.v1"},
        {"mode", "local_read_only"},
        {"question", message},
        {"run_id", run_id},
        {"context", {
            {"schema_version", "pi.run-chat-context.v1"},
            {"run_id", run_id},
            {"problem_hints", hints},
            {"report_available", report.value("ok", false)},
            {"artifacts_available", artifacts.value("ok", false)},
            {"memory_count", memories.size()}
        }},
        {"summary", "Ich behandle die Beschreibung als Hinweis, nicht als bewiesene Ursache. Die naechsten Schritte sollten Report, Artefakte und gezielte A/B-Tests verbinden."},
        {"likely_causes", likely_causes},
        {"checks", checks},
        {"recommendations", recommendations},
        {"evidence", evidence},
        {"action_plan", action_plan},
        {"action_plan_validation", tile_compile::pi::validate_action_plan_shape(action_plan)}
    };
}

nlohmann::json pi_audit_log(const std::shared_ptr<AppState>& state, int limit) {
    nlohmann::json items = nlohmann::json::array();
    const int event_scan_limit = std::max(limit, std::min(10000, limit * 10));
    for (const auto& event : state->ui_event_store.list(0, std::max(1, event_scan_limit))) {
        if (event.event.rfind("pi.", 0) != 0 && event.event.rfind("config.ai.", 0) != 0) continue;
        nlohmann::json item = ui_event_to_json(event);
        item["audit_type"] = event.event.rfind("pi.", 0) == 0 ? "pi_event" : "config_ai_event";
        items.push_back(std::move(item));
    }

    tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
    for (const auto& memory : store.list(100000)) {
        items.push_back({
            {"audit_type", "memory_candidate"},
            {"memory_id", memory.value("memory_id", std::string())},
            {"type", memory.value("type", std::string())},
            {"status", memory.value("status", std::string("candidate"))},
            {"created_at", memory.value("created_at", std::string())},
            {"source", memory.value("source", std::string())},
            {"analysis_id", memory.value("analysis_id", std::string())},
            {"summary", memory.value("summary", std::string())}
        });
        if (!memory.contains("review") || !memory["review"].is_object()) continue;
        nlohmann::json item = {
            {"audit_type", "memory_review"},
            {"memory_id", memory.value("memory_id", std::string())},
            {"type", memory.value("type", std::string())},
            {"status", memory.value("status", std::string())},
            {"created_at", memory.value("created_at", std::string())},
            {"review", memory["review"]},
            {"summary", memory.value("summary", std::string())}
        };
        items.push_back(std::move(item));
    }
    trim_json_array_to_latest(items, limit);

    return {
        {"schema_version", "pi.audit.v1"},
        {"privacy_class", "metadata_only"},
        {"items", items},
        {"count", items.size()},
        {"latest_event_seq", state->ui_event_store.latest_seq()}
    };
}

} // namespace

namespace {

nlohmann::json preview_action_plan(const nlohmann::json& plan) {
    nlohmann::json updates = nlohmann::json::array();
    nlohmann::json actions = plan.contains("actions") && plan["actions"].is_array()
        ? plan["actions"]
        : nlohmann::json::array();
    for (const auto& action : actions) {
        if (!action.is_object()) continue;
        const std::string type = action.value("type", std::string());
        if (type == "config.set" && action.contains("path") && action["path"].is_string() && action.contains("value")) {
            updates.push_back({
                {"path", action["path"]},
                {"value", action["value"]},
                {"action_id", action.value("id", std::string())},
                {"rationale", action.value("rationale", std::string())}
            });
        } else if (type == "config.patch" && action.contains("updates") && action["updates"].is_array()) {
            for (const auto& update : action["updates"]) {
                if (!update.is_object()) continue;
                nlohmann::json item = update;
                item["action_id"] = action.value("id", std::string());
                updates.push_back(std::move(item));
            }
        }
    }
    return {
        {"schema_version", "pi.action-preview.v1"},
        {"mutation_free", true},
        {"action_count", actions.size()},
        {"config_updates", updates},
        {"config_update_count", updates.size()}
    };
}

nlohmann::json yaml_to_json(const YAML::Node& node) {
    if (!node || node.IsNull()) return nullptr;
    if (node.IsMap()) {
        nlohmann::json out = nlohmann::json::object();
        for (auto it = node.begin(); it != node.end(); ++it) out[it->first.as<std::string>()] = yaml_to_json(it->second);
        return out;
    }
    if (node.IsSequence()) {
        nlohmann::json out = nlohmann::json::array();
        for (auto it = node.begin(); it != node.end(); ++it) out.push_back(yaml_to_json(*it));
        return out;
    }
    try { return node.as<bool>(); } catch (...) {}
    try { return node.as<long long>(); } catch (...) {}
    try { return node.as<double>(); } catch (...) {}
    try { return node.as<std::string>(); } catch (...) {}
    return nullptr;
}

YAML::Node json_to_yaml_node(const nlohmann::json& value) {
    if (value.is_object()) {
        YAML::Node node(YAML::NodeType::Map);
        for (auto it = value.begin(); it != value.end(); ++it) node[it.key()] = json_to_yaml_node(it.value());
        return node;
    }
    if (value.is_array()) {
        YAML::Node node(YAML::NodeType::Sequence);
        for (const auto& item : value) node.push_back(json_to_yaml_node(item));
        return node;
    }
    if (value.is_boolean()) return YAML::Node(value.get<bool>());
    if (value.is_number_integer()) return YAML::Node(value.get<long long>());
    if (value.is_number_unsigned()) return YAML::Node(value.get<unsigned long long>());
    if (value.is_number_float()) return YAML::Node(value.get<double>());
    if (value.is_null()) return YAML::Node();
    return YAML::Node(value.get<std::string>());
}

std::string yaml_dump(const nlohmann::json& value) {
    YAML::Node node = json_to_yaml_node(value);
    std::ostringstream out;
    out << node;
    return out.str();
}

void set_dotted(nlohmann::json& root, const std::string& dotted_path, const nlohmann::json& value) {
    std::vector<std::string> parts;
    std::istringstream iss(dotted_path);
    std::string part;
    while (std::getline(iss, part, '.')) {
        if (!part.empty()) parts.push_back(part);
    }
    if (parts.empty()) return;
    nlohmann::json* node = &root;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
        if (!node->contains(parts[i]) || !(*node)[parts[i]].is_object()) (*node)[parts[i]] = nlohmann::json::object();
        node = &(*node)[parts[i]];
    }
    (*node)[parts.back()] = value;
}

nlohmann::json load_preview_base_config(const nlohmann::json& body, const std::shared_ptr<AppState>& state) {
    if (body.contains("base_config") && body["base_config"].is_object()) return body["base_config"];
    if (body.contains("config") && body["config"].is_object()) return body["config"];
    if (body.contains("yaml") && body["yaml"].is_string()) {
        return yaml_to_json(YAML::Load(body["yaml"].get<std::string>()));
    }
    std::ifstream in(state->runtime.default_config_path);
    if (!in) return nlohmann::json::object();
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.empty()) return nlohmann::json::object();
    return yaml_to_json(YAML::Load(text));
}

nlohmann::json build_validated_preview(const nlohmann::json& plan,
                                       const nlohmann::json& body,
                                       const std::shared_ptr<AppState>& state) {
    nlohmann::json preview = preview_action_plan(plan);
    nlohmann::json base = load_preview_base_config(body, state);
    if (!base.is_object()) base = nlohmann::json::object();
    nlohmann::json patched = base;
    for (const auto& update : preview["config_updates"]) {
        if (!update.is_object() || !update.contains("path") || !update["path"].is_string() || !update.contains("value")) continue;
        set_dotted(patched, update["path"].get<std::string>(), update["value"]);
    }
    const std::string base_yaml = yaml_dump(base);
    const std::string patched_yaml = yaml_dump(patched);
    SubprocessResult validate_res = run_subprocess({state->runtime.cli_exe, "validate-config", "--stdin"},
                                                   state->runtime.project_root.string(),
                                                   patched_yaml);
    auto validation = nlohmann::json::parse(validate_res.stdout_str, nullptr, false);
    if (validation.is_discarded() || !validation.is_object()) {
        validation = {
            {"valid", false},
            {"errors", nlohmann::json::array({"validate-config returned non-json output"})},
            {"warnings", nlohmann::json::array()}
        };
    }
    preview["base_config"] = base;
    preview["patched_config"] = patched;
    preview["base_yaml"] = base_yaml;
    preview["patched_yaml"] = patched_yaml;
    preview["yaml_changed"] = base_yaml != patched_yaml;
    preview["validation"] = validation;
    preview["config_valid"] = validate_res.exit_code == 0 && validation.value("valid", false);
    return preview;
}

nlohmann::json apply_validated_preview(const nlohmann::json& preview,
                                       const std::shared_ptr<AppState>& state) {
    const std::string patched_yaml = preview.value("patched_yaml", std::string());
    fs::path target = state->runtime.default_config_path;
    SubprocessResult save_res = run_subprocess({state->runtime.cli_exe, "save-config", target.string(), "--stdin"},
                                               state->runtime.project_root.string(),
                                               patched_yaml);
    auto save_payload = nlohmann::json::parse(save_res.stdout_str, nullptr, false);
    if (save_payload.is_discarded() || !save_payload.is_object()) {
        save_payload = nlohmann::json::object();
    }
    if (save_res.exit_code != 0) {
        return {
            {"ok", false},
            {"error", {
                {"code", "BACKEND_COMMAND_FAILED"},
                {"message", "save-config failed"},
                {"exit_code", save_res.exit_code},
                {"stderr", save_res.stderr_str},
                {"stdout", save_res.stdout_str}
            }}
        };
    }
    fs::path saved_path = save_payload.contains("path") && save_payload["path"].is_string()
        ? fs::path(save_payload["path"].get<std::string>())
        : target;
    std::string rev_id = state->revision_store.add(saved_path, patched_yaml, "pi_action_plan");
    {
        std::lock_guard<std::mutex> lk(state->state_mutex);
        state->active_config_revision_id = rev_id;
    }
    state->ui_event_store.push("pi.action_plan.apply", "pi.action_plan_apply", {
        {"path", saved_path.string()},
        {"revision_id", rev_id},
        {"config_update_count", preview.value("config_update_count", 0)}
    });
    return {
        {"ok", true},
        {"saved", save_payload.value("saved", true)},
        {"path", saved_path.string()},
        {"revision_id", rev_id}
    };
}

} // namespace

void tile_compile::routes::register_pi_routes(CrowApp& app, std::shared_ptr<AppState> state) {
    CROW_ROUTE(app, "/api/pi/context").methods("GET"_method)
    ([state]() {
        tile_compile::pi::PiContextBuilder builder(state);
        return json_resp(builder.build_overview_context());
    });

    CROW_ROUTE(app, "/api/pi/tools").methods("GET"_method)
    ([state]() {
        tile_compile::pi::PiToolRegistry registry(state);
        return json_resp(registry.list_tools());
    });

    CROW_ROUTE(app, "/api/pi/tools/call").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string name = body->value("name", std::string());
        if (name.empty()) return err_resp("BAD_REQUEST", "tool name is required", 400);
        const nlohmann::json input = body->contains("input") && (*body)["input"].is_object()
            ? (*body)["input"]
            : nlohmann::json::object();
        tile_compile::pi::PiToolRegistry registry(state);
        const auto result = registry.call_tool(name, input);
        return json_resp(result, result.value("ok", false) ? 200 : 404);
    });

    CROW_ROUTE(app, "/api/pi/assistant/ask").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string question = body->value("question", std::string());
        if (question.empty()) return err_resp("BAD_REQUEST", "question is required", 400);
        tile_compile::pi::PiAssistant assistant(state);
        return json_resp(assistant.answer(question));
    });

    CROW_ROUTE(app, "/api/pi/storage").methods("GET"_method)
    ([state](const crow::request&) {
        return json_resp(tile_compile::pi::pi_storage_status(state));
    });

    CROW_ROUTE(app, "/api/pi/storage").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string storage_dir = body->value("storage_dir", std::string());
        std::filesystem::path resolved;
        std::string error_code;
        std::string error_message;
        if (!tile_compile::pi::set_pi_storage_dir(state, storage_dir, resolved, error_code, error_message)) {
            return err_resp(error_code.empty() ? "BAD_REQUEST" : error_code,
                            error_message.empty() ? "failed to save PI storage directory" : error_message,
                            error_code == "PATH_NOT_ALLOWED" ? 403 : 400);
        }
        state->ui_event_store.push("pi.storage.save", "pi.storage", {{"storage_dir", resolved.string()}});
        return json_resp(tile_compile::pi::pi_storage_status(state));
    });

    CROW_ROUTE(app, "/api/pi/run-chat").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string run_id = body->value("run_id", std::string());
        const std::string message = body->value("message", std::string());
        if (run_id.empty()) return err_resp("BAD_REQUEST", "run_id is required", 400);
        if (message.empty()) return err_resp("BAD_REQUEST", "message is required", 400);
        try {
            return json_resp(build_run_chat_answer(state, run_id, message));
        } catch (const std::exception& e) {
            return err_resp("RUN_CONTEXT_UNAVAILABLE", e.what(), 400);
        }
    });

    CROW_ROUTE(app, "/api/pi/memories").methods("GET"_method)
    ([state](const crow::request& req) {
        const int limit = std::max(1, std::min(500, int_query_param(req, "limit", 100)));
        const std::string status_filter = req.url_params.get("status") ? std::string(req.url_params.get("status")) : "";
        tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
        const bool filtered = !status_filter.empty() && status_filter != "all";
        nlohmann::json items = store.list(filtered ? 100000 : limit);
        if (filtered) {
            nlohmann::json filtered_items = nlohmann::json::array();
            for (const auto& item : items) {
                if (item.value("status", std::string()) == status_filter) filtered_items.push_back(item);
            }
            items = std::move(filtered_items);
            trim_json_array_to_latest(items, limit);
        }
        return json_resp({
            {"schema_version", "pi.memories-list.v1"},
            {"memory_dir", store.memory_dir().string()},
            {"items", items},
            {"count", items.size()}
        });
    });

    CROW_ROUTE(app, "/api/pi/memories/export").methods("GET"_method)
    ([state](const crow::request& req) {
        const std::string privacy = req.url_params.get("privacy")
            ? std::string(req.url_params.get("privacy"))
            : std::string("metadata_only");
        const bool include_reviews = !req.url_params.get("include_reviews") ||
            std::string(req.url_params.get("include_reviews")) != "0";
        tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
        return json_resp(store.export_bundle(privacy, include_reviews));
    });

    CROW_ROUTE(app, "/api/pi/memories/import").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const bool dry_run = body->value("dry_run", false);
        const nlohmann::json bundle = body->contains("bundle") && (*body)["bundle"].is_object()
            ? (*body)["bundle"]
            : *body;
        try {
            tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
            return json_resp(store.import_bundle(bundle, dry_run));
        } catch (const std::invalid_argument& e) {
            return err_resp("BAD_REQUEST", e.what(), 400);
        } catch (const std::exception& e) {
            return err_resp("BACKEND_COMMAND_FAILED", e.what(), 502);
        }
    });

    CROW_ROUTE(app, "/api/pi/memories/dedupe").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const bool dry_run = body->value("dry_run", false);
        try {
            tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
            return json_resp(store.dedupe(dry_run));
        } catch (const std::exception& e) {
            return err_resp("BACKEND_COMMAND_FAILED", e.what(), 502);
        }
    });

    CROW_ROUTE(app, "/api/pi/memories/<string>/review").methods("POST"_method)
    ([state](const crow::request& req, const std::string& memory_id) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string status = body->value("status", std::string());
        const std::string reviewer = body->value("reviewer", std::string("user"));
        const std::string note = body->value("note", std::string());
        const nlohmann::json outcome = body->contains("outcome") && (*body)["outcome"].is_object()
            ? (*body)["outcome"]
            : nlohmann::json::object();
        if (status.empty()) return err_resp("BAD_REQUEST", "status is required", 400);
        try {
            tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
            const auto review = store.review(memory_id, status, reviewer, note, outcome);
            return json_resp({{"ok", true}, {"review", review}});
        } catch (const std::invalid_argument& e) {
            return err_resp("BAD_REQUEST", e.what(), 400);
        } catch (const std::exception& e) {
            return err_resp("BACKEND_COMMAND_FAILED", e.what(), 502);
        }
    });

    CROW_ROUTE(app, "/api/pi/memories/retrieve").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const int limit = std::max(1, std::min(100, body->value("limit", 10)));
        const nlohmann::json query = body->contains("query") && (*body)["query"].is_object()
            ? (*body)["query"]
            : *body;
        tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
        const auto matches = store.retrieve(query, limit);
        return json_resp({
            {"schema_version", "pi.memory-retrieval.v1"},
            {"matches", matches},
            {"count", matches.size()}
        });
    });

    CROW_ROUTE(app, "/api/pi/audit").methods("GET"_method)
    ([state](const crow::request& req) {
        const int limit = std::max(1, std::min(1000, int_query_param(req, "limit", 200)));
        return json_resp(pi_audit_log(state, limit));
    });

    CROW_ROUTE(app, "/api/pi/action-plans/validate").methods("POST"_method)
    ([](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const nlohmann::json plan = body->contains("plan") && (*body)["plan"].is_object()
            ? (*body)["plan"]
            : *body;
        return json_resp(tile_compile::pi::validate_action_plan_shape(plan));
    });

    CROW_ROUTE(app, "/api/pi/action-plans/preview").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const nlohmann::json plan = body->contains("plan") && (*body)["plan"].is_object()
            ? (*body)["plan"]
            : *body;
        const auto validation = tile_compile::pi::validate_action_plan_shape(plan);
        if (!validation.value("valid", false)) {
            return json_resp({
                {"ok", false},
                {"validation", validation}
            }, 400);
        }
        return json_resp({
            {"ok", true},
            {"validation", validation},
            {"preview", build_validated_preview(plan, *body, state)}
        });
    });

    CROW_ROUTE(app, "/api/pi/action-plans/apply").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const bool confirmed = body->value("confirmed", false) || body->value("reviewed", false);
        if (!confirmed) {
            return err_resp("REVIEW_REQUIRED", "confirmed=true is required before applying a PI action plan", 409);
        }
        const nlohmann::json plan = body->contains("plan") && (*body)["plan"].is_object()
            ? (*body)["plan"]
            : *body;
        const auto validation = tile_compile::pi::validate_action_plan_shape(plan);
        if (!validation.value("valid", false)) {
            return json_resp({
                {"ok", false},
                {"validation", validation}
            }, 400);
        }
        const auto preview = build_validated_preview(plan, *body, state);
        if (!preview.value("config_valid", false)) {
            return json_resp({
                {"ok", false},
                {"validation", validation},
                {"preview", preview},
                {"error", {{"code", "CONFIG_INVALID"}, {"message", "preview config validation failed"}}}
            }, 400);
        }
        if (body->contains("expected_patched_yaml") && (*body)["expected_patched_yaml"].is_string()
            && (*body)["expected_patched_yaml"].get<std::string>() != preview.value("patched_yaml", std::string())) {
            return json_resp({
                {"ok", false},
                {"validation", validation},
                {"preview", preview},
                {"error", {{"code", "STALE_PREVIEW"}, {"message", "expected_patched_yaml does not match current preview"}}}
            }, 409);
        }
        auto applied = apply_validated_preview(preview, state);
        if (!applied.value("ok", false)) {
            return json_resp(applied, 502);
        }
        return json_resp({
            {"ok", true},
            {"validation", validation},
            {"preview", preview},
            {"revision_id", applied["revision_id"]},
            {"path", applied["path"]},
            {"saved", applied["saved"]}
        });
    });
}
