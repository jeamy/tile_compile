#include "routes/pi_routes.hpp"

#include "app_state.hpp"
#include "routes/route_utils.hpp"
#include "services/pi/pi_assistant.hpp"
#include "services/pi/pi_context_builder.hpp"
#include "services/pi/pi_action_validator.hpp"
#include "services/pi/pi_memory_store.hpp"
#include "services/pi/pi_tool_registry.hpp"
#include "subprocess_manager.hpp"

#include <algorithm>
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

std::filesystem::path pi_memory_dir(const std::shared_ptr<AppState>& state) {
    return state->runtime.runs_dir / ".pi_memory";
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

nlohmann::json pi_audit_log(const std::shared_ptr<AppState>& state, int limit) {
    nlohmann::json items = nlohmann::json::array();
    const int event_scan_limit = std::max(limit, std::min(10000, limit * 10));
    for (const auto& event : state->ui_event_store.list(0, std::max(1, event_scan_limit))) {
        if (event.event.rfind("pi.", 0) != 0 && event.event.rfind("config.ai.", 0) != 0) continue;
        nlohmann::json item = ui_event_to_json(event);
        item["audit_type"] = event.event.rfind("pi.", 0) == 0 ? "pi_event" : "config_ai_event";
        items.push_back(std::move(item));
    }

    tile_compile::pi::PiMemoryStore store(pi_memory_dir(state));
    for (const auto& memory : store.list(100000)) {
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

    CROW_ROUTE(app, "/api/pi/memories").methods("GET"_method)
    ([state](const crow::request& req) {
        const int limit = std::max(1, std::min(500, int_query_param(req, "limit", 100)));
        const std::string status_filter = req.url_params.get("status") ? std::string(req.url_params.get("status")) : "";
        tile_compile::pi::PiMemoryStore store(pi_memory_dir(state));
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
        tile_compile::pi::PiMemoryStore store(pi_memory_dir(state));
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
            tile_compile::pi::PiMemoryStore store(pi_memory_dir(state));
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
            tile_compile::pi::PiMemoryStore store(pi_memory_dir(state));
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
            tile_compile::pi::PiMemoryStore store(pi_memory_dir(state));
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
        tile_compile::pi::PiMemoryStore store(pi_memory_dir(state));
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
