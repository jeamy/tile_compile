#include "routes/pi_decision_routes.hpp"

#include "routes/route_utils.hpp"
#include "services/ai_service.hpp"
#include "services/pi/pi_decision_outcome.hpp"
#include "services/pi/pi_decision_service.hpp"
#include "services/pi/pi_pre_rules.hpp"
#include "services/pi/pi_post_run.hpp"
#include "services/pi/pi_assistant_event_store.hpp"
#include "services/pi/pi_analysis_context_store.hpp"
#include "services/pi/pi_run_index.hpp"
#include "services/pi/pi_scan_manifest.hpp"
#include "services/pi/pi_json_io.hpp"
#include "services/pi/pi_storage_paths.hpp"
#include "services/scan_metrics_cache.hpp"
#include "services/scan_summary.hpp"
#include "subprocess_manager.hpp"
#include "time_utils.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <random>
#include <set>
#include <thread>

namespace tile_compile::routes {
namespace {

using nlohmann::json;
namespace fs = std::filesystem;
using namespace tile_compile::pi;

std::string new_proposal_id() {
    std::random_device rd;
    static const char* hex = "0123456789abcdef";
    std::string suffix;
    for (int i = 0; i < 8; ++i) suffix += hex[rd() % 16];
    std::string ts = utc_now_iso();
    ts.erase(std::remove_if(ts.begin(), ts.end(), [](char c) { return c == '-' || c == ':'; }), ts.end());
    return "dec_" + ts + "_" + suffix;
}

fs::path catalog_dir(const std::shared_ptr<AppState>& state) {
    if (const char* e = std::getenv("PI_DECISIONS_CATALOG_DIR"); e && *e) return e;
    return state->runtime.project_root / "web_backend_cpp" / "config" / "pi_decisions";
}

std::optional<json> read_json_path(const fs::path& p) { return read_json_file_opt(p); }

json post_run_provider_projection(const json& state, const json& advice) {
    const json run = state.value("run", json::object());
    const json rec = state.value("reconstruction", json::object());
    json projection = {
        {"domain", "post_run"},
        {"run", {{"success", run.value("success", false)},
                 {"final_image_available", run.value("final_image_available", false)},
                 {"completed_phase_count", run.value("phases_completed", json::array()).size()}}},
        {"reconstruction", json::object()},
        {"findings", json::array()},
        {"not_measured", state.value("not_measured", json::array())}
    };
    const std::string selected = rec.value("selected_candidate", std::string());
    if (!selected.empty() && std::all_of(selected.begin(), selected.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
        })) projection["reconstruction"]["selected_candidate"] = selected;
    const json cands = rec.value("candidates", json::object());
    for (const auto& [candidate, values] : cands.items()) {
        if (!std::all_of(candidate.begin(), candidate.end(), [](unsigned char c) {
                return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
            }) || !values.is_object()) continue;
        json safe = json::object();
        for (const char* key : {"numerics_ok", "support_ok", "median_fwhm", "p90_fwhm", "tail",
                                "elongation", "background_rms", "seam_score"}) {
            if (!values.contains(key)) continue;
            const auto& value = values[key];
            if (value.is_boolean() || (value.is_number() && std::isfinite(value.get<double>()))) safe[key] = value;
            else if (value.is_object() && value.contains("value") && value["value"].is_number() &&
                     std::isfinite(value["value"].get<double>())) safe[key] = value["value"];
        }
        projection["reconstruction"][candidate] = std::move(safe);
    }
    for (const auto& finding : advice.value("findings", json::array())) {
        const std::string code = finding.value("code", std::string());
        const std::string severity = finding.value("severity", std::string());
        if (!code.empty() && (severity == "info" || severity == "warning"))
            projection["findings"].push_back({{"code", code}, {"severity", severity}});
    }
    json diagnostics = json::object();
    const json stage = state.value("stage_diagnostics", json::object());
    const json bge = stage.value("bge_background_slope", json::object());
    json comparisons = json::array();
    for (const auto& item : bge.value("comparisons", json::array())) {
        if (!item.is_object()) continue;
        const json before = item.value("before", json::object());
        const json after = item.value("after", json::object());
        if (!before.contains("value") || !after.contains("value") || !before["value"].is_number() ||
            !after["value"].is_number() || !item.contains("delta") || !item["delta"].is_number()) continue;
        comparisons.push_back({{"channel", item.value("channel", std::string())},
            {"before", before["value"]}, {"after", after["value"]}, {"delta", item["delta"]}});
    }
    if (!comparisons.empty()) diagnostics["bge_slope_comparisons"] = std::move(comparisons);
    const json hms = stage.value("hms_curve_clipping", json::object());
    for (const char* key : {"black_clip_percent", "white_clip_percent"})
        if (hms.contains(key) && hms[key].is_number() && std::isfinite(hms[key].get<double>())) diagnostics[key] = hms[key];
    if (!diagnostics.empty()) projection["diagnostics"] = std::move(diagnostics);
    return projection;
}

json post_run_model_selection(const std::shared_ptr<DecisionService>& svc, const json& state, json& advice,
                              const json& run_config, const DecisionPolicy& policy, const ConfigValidator& validator,
                              bool allow_experimental, const std::string& request_id) {
    const json suggestions = advice.value("suggestions", json::array());
    if (suggestions.empty()) return {{"status", "skipped_no_candidates"}, {"used_for_recommendation", false}};
    try {
        ai::AiSidecarClient client(ai::default_ai_config());
        const json status = client.get("/decisions/status");
        const std::string mode = status.value("mode", std::string("off"));
        if (mode != "suggest" && mode != "shadow") return {{"status", "skipped_mode_off"}, {"used_for_recommendation", false}};
        if (!status.value("has_api_key", false)) return {{"status", "unavailable"}, {"mode", mode}, {"error_code", "no_api_key"}, {"used_for_recommendation", false}};
        if (!allow_experimental) return {{"status", "skipped_experimental_disabled"}, {"used_for_recommendation", false}};

        const json stub = {{"locked_paths", json::array()}, {"blocking_findings", json::array()}, {"groups", json::array()}};
        auto candidates = build_pre_run_candidates(stub, run_config, policy, svc->catalog(), validator);
        std::set<std::string> offered;
        for (const auto& item : suggestions) offered.insert(item.value("candidate_id", std::string()));
        candidates.applicable.erase(std::remove_if(candidates.applicable.begin(), candidates.applicable.end(), [&](const json& item) {
            return !offered.contains(item.value("candidate_id", std::string()));
        }), candidates.applicable.end());
        const json info = provider_candidate_info(candidates, svc->catalog());
        json allowed = json::array({"keep_current", "insufficient_evidence"});
        for (const auto& item : suggestions) {
            const std::string id = item.value("candidate_id", std::string());
            if (!id.empty() && std::find(allowed.begin(), allowed.end(), id) == allowed.end()) allowed.push_back(id);
        }
        json request = {{"request_id", request_id}, {"state_hash", state.value("state_hash", std::string())},
            {"question_set_version", "post-run-decisions.v1"},
            {"state_projection", post_run_provider_projection(state, advice)}, {"allowed_candidates", allowed}};
        if (info.contains("descriptions") && !info["descriptions"].empty()) request["candidate_descriptions"] = info["descriptions"];
        if (info.contains("facts") && !info["facts"].empty()) request["candidate_facts"] = info["facts"];
        const json response = client.post("/decisions", request);
        const bool usable = response.value("status", std::string()) == "ok" &&
            response.contains("selection") && response["selection"].is_object() &&
            std::find(allowed.begin(), allowed.end(), response["selection"].value("candidate_id", std::string())) != allowed.end();
        json result = {{"status", usable ? "ok" : response.value("status", std::string("unavailable"))},
                       {"mode", mode},
                       {"model_requested", status.value("model", std::string())},
                       {"model_reported", response.value("model_reported", json(nullptr))},
                       {"used_for_recommendation", mode == "suggest" && usable}};
        if (response.contains("error_code")) result["error_code"] = response["error_code"];
        if (usable) {
            result["candidate_id"] = response["selection"]["candidate_id"];
            result["probabilities"] = response["selection"].value("probabilities", json::object());
            result["provider_confidence"] = response["selection"].value("provider_confidence", json(nullptr));
            if (mode == "suggest") advice["model_recommendation"] = response["selection"]["candidate_id"];
        }
        return result;
    } catch (const ai::AiSidecarHttpError& e) {
        return {{"status", "unavailable"}, {"error_code", "sidecar_http_" + std::to_string(e.status())}, {"used_for_recommendation", false}};
    } catch (const std::exception&) {
        return {{"status", "unavailable"}, {"error_code", "sidecar_unreachable"}, {"used_for_recommendation", false}};
    }
}

// One service per process, created on first use so a missing catalog only breaks Jev routes.
struct ServiceHolder {
    std::mutex m;
    std::shared_ptr<DecisionService> svc;
    fs::path storage_root;
    std::string error;
};

std::shared_ptr<DecisionService> get_service(const std::shared_ptr<AppState>& state, ServiceHolder& h, std::string& error) {
    std::lock_guard<std::mutex> lock(h.m);
    const auto root = pi_storage_dir(state);
    if (h.svc && h.storage_root == root) return h.svc;
    h.svc.reset();
    try {
        const auto dir = catalog_dir(state);
        const auto cands = read_json_path(dir / "candidates_v1.json");
        const auto prot = read_json_path(dir / "protected_paths_v1.json");
        if (!cands || !prot) throw std::runtime_error("decision catalog not found under " + dir.string());
        DecisionServiceDeps deps;
        deps.sidecar = [](const std::string& method, const std::string& endpoint, const json& payload) -> json {
            ai::AiSidecarClient client(ai::default_ai_config());
            try {
                return method == "GET" ? client.get(endpoint) : client.post(endpoint, payload);
            } catch (const ai::AiSidecarHttpError& e) {
                throw SidecarHttpError(e.status(), e.payload());
            }
        };
        const std::shared_ptr<AppState> st = state;
        deps.validate_config = [st](const json& merged) -> ConfigCheck {
            // The existing validate-config CLI is the single authority for "is this a valid config".
            const SubprocessResult res = run_subprocess({st->runtime.cli_exe, "validate-config", "--stdin"}, st->runtime.project_root.string(), yaml_dump(merged));
            const json parsed = json::parse(res.stdout_str, nullptr, false);
            const bool valid = res.exit_code == 0 && parsed.is_object() && parsed.value("valid", false);
            return {valid, valid ? std::string() : (parsed.is_object() ? parsed.dump().substr(0, 300) : std::string("validate-config failed"))};
        };
        deps.now_iso = [] { return utc_now_iso(); };
        deps.new_id = [] { return new_proposal_id(); };
        h.svc = std::make_shared<DecisionService>(root / "pi_decisions", load_decision_catalog(*cands, *prot), deps);
        h.storage_root = root;
        return h.svc;
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
}

// Server-side reconstruction of the scan inputs: the browser only supplies the config draft and locks.
std::optional<AdviceRequest> build_request(const std::shared_ptr<AppState>& state, const json& body, crow::response& error_out) {
    AdviceRequest r;
    if (!body.contains("yaml") || !body["yaml"].is_string() || body["yaml"].get<std::string>().empty()) {
        error_out = err_resp("DRAFT_MISSING", "yaml (the config draft) is required", 400);
        return std::nullopt;
    }
    try {
        r.base_config = yaml_text_to_json(body["yaml"].get<std::string>());
    } catch (const std::exception& e) {
        error_out = err_resp("DRAFT_INVALID", std::string("config draft is not valid YAML: ") + e.what(), 400);
        return std::nullopt;
    }
    if (!r.base_config.is_object()) {
        error_out = err_resp("DRAFT_INVALID", "config draft must be a YAML mapping", 400);
        return std::nullopt;
    }
    const auto job = latest_scan_job(state->job_store);
    r.scan = summarize_scan_job(job, state->last_scan_input_path);
    if (!r.scan.value("has_scan", false) || !r.scan.value("ok", false)) {
        error_out = err_resp("NO_SCAN", "no successful scan available", 400);
        return std::nullopt;
    }
    r.scan_id = r.scan.value("job_id", std::string());
    const std::string input_path = r.scan.value("input_path", std::string());
    // Same lookup the scan-metrics route performs: a finished job in the job store, else the on-disk cache.
    const std::string object_name = r.scan.value("object_name", r.scan.value("target", std::string()));
    const int frame_count = r.scan.value("frames_detected", r.scan.value("frames_total", 0));
    json metrics = find_cached_scan_metrics(state, input_path, object_name, frame_count);
    if (metrics.empty()) metrics = find_disk_cached_scan_metrics(state, scan_metrics_cache_key(input_path, object_name, frame_count));
    if (metrics.empty() || !metrics.value("ok", false)) {
        error_out = err_resp("NO_SCAN_METRICS", "no scan-metrics result for the current scan; run scan metrics first", 400);
        return std::nullopt;
    }
    r.metrics = metrics;
    try {
        r.dataset_manifest = build_scan_dataset_manifest(input_path, metrics);
    } catch (const std::exception& e) {
        error_out = err_resp("SCAN_INPUT_CHANGED", e.what(), 409);
        return std::nullopt;
    }
    if (body.contains("locked_paths") && body["locked_paths"].is_array())
        for (const auto& p : body["locked_paths"]) if (p.is_string()) r.locked_paths.push_back(p.get<std::string>());
    if (body.contains("session_context") && body["session_context"].is_object()) r.session_context = body["session_context"];
    r.software_version = "tile_compile_web_backend";
    return r;
}

} // namespace

void register_pi_decision_routes(CrowApp& app, std::shared_ptr<AppState> state) {
    auto holder = std::make_shared<ServiceHolder>();

    CROW_ROUTE(app, "/api/pi/decisions/status").methods("GET"_method)
    ([]() {
        try {
            ai::AiSidecarClient client(ai::default_ai_config());
            json s = client.get("/decisions/status");
            s["available"] = true;
            return json_resp(s);
        } catch (const std::exception& e) {
            return json_resp({{"available", false}, {"reason", "sidecar_unreachable"}});
        }
    });

    // Connection check for the Jev settings card: one minimal provider call with a data-free state (see the sidecar).
    CROW_ROUTE(app, "/api/pi/decisions/test").methods("POST"_method)
    ([]() {
        try {
            ai::AiSidecarClient client(ai::default_ai_config());
            return json_resp(client.post("/decisions/test", json::object()));
        } catch (const ai::AiSidecarHttpError& e) {
            return err_resp("PROBE_REJECTED", e.payload().value("message", std::string("probe rejected")), static_cast<int>(e.status()));
        } catch (const std::exception&) {
            return err_resp("SIDECAR_UNREACHABLE", "PI sidecar is not reachable", 502);
        }
    });

    // Read-only view of the Jev request/response log for the UI (the sidecar redacts entries when it writes them).
    CROW_ROUTE(app, "/api/pi/decisions/log").methods("GET"_method)
    ([](const crow::request& req) {
        try {
            std::string path = "/decisions/log";
            if (const char* limit = req.url_params.get("limit")) {
                const std::string l = limit;
                if (l.empty() || l.size() > 5 || l.find_first_not_of("0123456789") != std::string::npos)
                    return err_resp("BAD_REQUEST", "limit must be a number", 400);
                path += "?limit=" + l;
            }
            ai::AiSidecarClient client(ai::default_ai_config());
            return json_resp(client.get(path));
        } catch (const std::exception&) {
            return err_resp("SIDECAR_UNREACHABLE", "PI sidecar is not reachable", 502);
        }
    });

    // Jev card settings (mode / experimental flag / write-only API key) are stored by the sidecar in its own
    // file, independent of PI's provider settings. The body is forwarded untouched and never logged here.
    CROW_ROUTE(app, "/api/pi/decisions/settings").methods("POST"_method)
    ([](const crow::request& req) {
        auto body = parse_body(req);
        if (!body || !body->is_object()) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        json filtered = json::object();
        for (const char* key : {"mode", "allow_experimental_suggestions", "api_key"})
            if (body->contains(key)) filtered[key] = (*body)[key];
        try {
            ai::AiSidecarClient client(ai::default_ai_config());
            json s = client.post("/decisions/settings", filtered);
            s["available"] = true;
            return json_resp(s);
        } catch (const ai::AiSidecarHttpError& e) {
            return err_resp("SETTINGS_REJECTED", e.payload().value("message", std::string("settings rejected")), static_cast<int>(e.status()));
        } catch (const std::exception&) {
            return err_resp("SIDECAR_UNREACHABLE", "PI sidecar is not reachable", 502);
        }
    });

    CROW_ROUTE(app, "/api/pi/assistant/analysis-context").methods("POST"_method)
    ([state](const crow::request& req) {
        const auto body = parse_body(req);
        if (!body || !body->is_object()) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        crow::response failure;
        auto request = build_request(state, *body, failure);
        if (!request) return failure;
        try { return json_resp(PiAnalysisContextStore(pi_storage_dir(state)).resolve(*request)); }
        catch (const std::exception& e) { return err_resp("ANALYSIS_CONTEXT_UNAVAILABLE", e.what(), 503); }
    });

    CROW_ROUTE(app, "/api/pi/assistant/analysis-contexts/<string>").methods("GET"_method)
    ([state](const std::string& uid) {
        try {
            PiAnalysisContextStore store(pi_storage_dir(state));
            const auto context = store.get(uid);
            if (!context) return err_resp("NOT_FOUND", "Unknown analysis UID", 404);
            json result = *context;
            result["revisions"] = store.revisions(uid); result["proposals"] = store.proposals(uid);
            return json_resp(result);
        } catch (const std::exception& e) { return err_resp("ANALYSIS_CONTEXT_UNAVAILABLE", e.what(), 503); }
    });

    CROW_ROUTE(app, "/api/pi/assistant/select-analysis").methods("POST"_method)
    ([state](const crow::request& req) {
        const auto body = parse_body(req);
        if (!body || !body->is_object() || !body->contains("analysis_uid") || !(*body)["analysis_uid"].is_string())
            return err_resp("BAD_REQUEST", "analysis_uid is required", 400);
        try {
            PiAnalysisContextStore store(pi_storage_dir(state));
            const auto context = store.get((*body)["analysis_uid"]);
            if (!context) return err_resp("NOT_FOUND", "Unknown analysis UID", 404);
            if (!set_pi_active_context(state, *context)) return err_resp("CONTEXT_SAVE_FAILED", "Active context could not be persisted", 503);
            return json_resp(*context);
        } catch (const std::exception& e) { return err_resp("ANALYSIS_CONTEXT_UNAVAILABLE", e.what(), 503); }
    });

    CROW_ROUTE(app, "/api/pi/assistant/analysis-thread/<string>").methods("GET"_method)
    ([state, holder](const std::string& uid) {
        try {
            PiAnalysisContextStore store(pi_storage_dir(state));
            if (!store.get(uid)) return err_resp("NOT_FOUND", "Unknown analysis UID", 404);
            const auto links = store.proposals(uid);
            std::string error;
            auto svc = links.empty() ? nullptr : get_service(state, *holder, error);
            if (!links.empty() && !svc) return err_resp("DECISIONS_UNAVAILABLE", error, 503);
            json items = json::array();
            size_t bytes = 0; bool truncated = false;
            for (const auto& link : links) {
                const auto view = svc->view(link["proposal_id"]);
                if (!view) continue;
                json item = {{"event_id", link["proposal_id"]}, {"revision_id", link["revision_id"]},
                    {"context_id", "analysis:" + uid}, {"provider", "jev"}, {"kind", "jev_scan_proposal"}, {"result", *view}};
                bytes += item.dump().size();
                if (bytes > 8 * 1024 * 1024) { truncated = true; break; }
                items.push_back(std::move(item));
            }
            return json_resp({{"schema_version", "pi.assistant-thread.v1"}, {"context_id", "analysis:" + uid}, {"items", items},
                {"proposal_limit", 100}, {"byte_limit", 8388608}, {"truncated", truncated}});
        } catch (const std::exception& e) { return err_resp("ANALYSIS_THREAD_UNAVAILABLE", e.what(), 503); }
    });

    CROW_ROUTE(app, "/api/scan/decisions").methods("POST"_method)
    ([state, holder](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        std::string error;
        auto svc = get_service(state, *holder, error);
        if (!svc) return err_resp("DECISIONS_UNAVAILABLE", error, 503);
        crow::response failure;
        auto advice = build_request(state, *body, failure);
        if (!advice) return failure;
        // One question per candidate group: every group with an applicable change gets its own proposal, so the user
        // can select which recommendations to apply. No group (nothing but baselines) is one proposal, as before.
        AdviceRequest copy = *advice;
        json context;
        try {
            context = PiAnalysisContextStore(pi_storage_dir(state)).resolve(copy);
            if (body->contains("analysis_uid") && (!(*body)["analysis_uid"].is_string() || (*body)["analysis_uid"] != context["analysis_uid"]))
                return err_resp("ANALYSIS_CONTEXT_CHANGED", "Current scan differs from selected analysis context", 409);
            copy.session_context["context_id"] = context["context_id"];
            copy.session_context["analysis_revision_id"] = context["revision_id"];
        } catch (const std::exception& e) { return err_resp("ANALYSIS_CONTEXT_UNAVAILABLE", e.what(), 503); }
        std::vector<std::string> groups;
        try { groups = svc->groups_for(copy); } catch (const std::exception& e) { return err_resp("STORE_FAILED", e.what(), 500); }
        if (groups.empty()) groups.push_back("");
        std::vector<std::string> ids;
        try {
            const auto db = PiDatabase::open(pi_storage_dir(state));
            PiDatabase::Tx tx(*db);
            PiAnalysisContextStore contexts(pi_storage_dir(state));
            for (size_t i = 0; i < groups.size(); ++i) {
                ids.push_back(svc->create());
                contexts.link(context["analysis_uid"], context["revision_id"], ids.back());
            }
            tx.commit();
        } catch (const std::exception& e) { return err_resp("STORE_FAILED", e.what(), 500); }
        std::thread([svc, ids, groups, copy]() {
            for (size_t i = 0; i < ids.size(); ++i) svc->run(ids[i], copy, groups[i]);
        }).detach();
        return json_resp({{"proposal_id", ids.front()}, {"proposal_ids", ids}, {"groups", groups}, {"state", "running"}, {"analysis_context", context}}, 202);
    });

    // Explicit rule-based advice with optional constrained Jev selection: run artifacts stay read-only and
    // the resulting card is stored centrally. A model cannot invent candidates or apply config/start a run.
    // Resume still requires dry-run feasibility and explicit start.
    CROW_ROUTE(app, "/api/pi/post-run/advice").methods("POST"_method)
    ([state, holder](const crow::request& req) {
        auto body = parse_body(req);
        if (!body || !body->is_object()) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        for (const auto* key : {"run_id", "run_uid", "request_id", "run_dir"})
            if (body->contains(key) && !(*body)[key].is_string())
                return err_resp("BAD_REQUEST", std::string(key) + " must be a string", 400);
        const std::string run_id = body->value("run_id", std::string());
        const std::string expected_uid = body->value("run_uid", std::string());
        const std::string request_id = body->value("request_id", std::string());
        const std::string run_dir_hint = body->value("run_dir", std::string());
        if (run_id.empty() || request_id.size() > 128 ||
            request_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos)
            return err_resp("BAD_REQUEST", "run_id is required; request_id must be at most 128 identifier characters", 400);
        fs::path run_dir;
        if (auto err = resolve_hinted_run_dir(state, run_id, run_dir_hint, run_dir)) return std::move(*err);
        std::error_code ec;
        if (!fs::is_directory(run_dir, ec)) return err_resp("NOT_FOUND", "run not found", 404);
        if (auto denied = validate_path(state, run_dir, true)) return std::move(*denied);
        std::string uid;
        try {
            const auto provenance = read_json_file_opt(run_dir / "artifacts/pi_run_provenance.json").value_or(json::object());
            const auto identity = PiRunIndex(pi_storage_dir(state)).resolve(run_dir, provenance.value("run_uid", std::string()));
            uid = identity["run_uid"];
            if (!expected_uid.empty() && uid != expected_uid)
                return err_resp("RUN_IDENTITY_CONFLICT", "Run path does not match requested context UID", 409);
        } catch (const PiRunIdentityConflict& e) { return err_resp("RUN_IDENTITY_CONFLICT", e.what(), 409); }
          catch (const std::exception& e) { return err_resp("RUN_CONTEXT_UNAVAILABLE", e.what(), 503); }
        json run_config = nullptr;
        try { run_config = yaml_text_to_json(read_file_str(run_dir / "config.yaml")); } catch (const std::exception&) { run_config = nullptr; }
        std::string error;
        auto svc = get_service(state, *holder, error);
        if (!svc) return err_resp("DECISIONS_UNAVAILABLE", error, 503);
        PiAssistantEventStore event_store(pi_storage_dir(state));
        const std::string effective_request_id = request_id.empty() ? PiRunIndex::generate_uid() : request_id;
        const std::string id = "jev_" + sha256_prefixed("run:" + uid + ":jev_post_run:" + effective_request_id).substr(7);
        if (const auto old = event_store.get(id)) {
            if (old->value("run_uid", std::string()) != uid)
                return err_resp("ASSISTANT_EVENT_CONFLICT", "Advice event belongs to a different run", 409);
            json response = old->value("result", json::object());
            for (const auto* key : {"event_id", "run_uid", "context_id", "evaluation_method"})
                if (old->contains(key)) response[key] = (*old)[key];
            return json_resp(response);
        }
        std::vector<std::string> dismissed;
        const json dismissal_events = event_store.post_run_dismissals(uid);
        for (const auto& event : dismissal_events) {
            const auto result = event.value("result", json::object());
            if (result.value("dismissed", false)) dismissed.push_back(result.value("candidate_id", std::string()));
        }
        DecisionPolicy policy;
        policy.allow_experimental = body->value("allow_experimental", false);
        const std::shared_ptr<AppState> st = state;
        ConfigValidator validator = [st](const json& merged) -> ConfigCheck {
            const SubprocessResult res = run_subprocess({st->runtime.cli_exe, "validate-config", "--stdin"}, st->runtime.project_root.string(), yaml_dump(merged));
            const json parsed = json::parse(res.stdout_str, nullptr, false);
            const bool valid = res.exit_code == 0 && parsed.is_object() && parsed.value("valid", false);
            return {valid, valid ? std::string() : std::string("validate-config failed")};
        };
        const json ps = build_post_run_state(run_dir, run_config);
        const std::string source_patch_id = read_json_file_opt(run_dir / "artifacts/pi_run_provenance.json")
            .value_or(json::object()).value("jev_proposal_id", std::string());
        json advice = advise_post_run(ps, run_config, policy, svc->catalog(), validator, dismissed);
        const json model_selection = post_run_model_selection(svc, ps, advice, run_config, policy, validator,
            policy.allow_experimental, effective_request_id);
        const json result = {{"state", ps}, {"advice", advice}, {"model_selection", model_selection},
                             {"dismissed_candidates", dismissal_events}, {"source_patch_id", source_patch_id}};
        try {
            const auto event = event_store.append(id, uid, result);
            json response = result;
            for (const auto* key : {"event_id", "run_uid", "context_id", "evaluation_method"}) response[key] = event[key];
            return json_resp(response);
        } catch (const std::invalid_argument& e) { return err_resp("ASSISTANT_EVENT_CONFLICT", e.what(), 409); }
          catch (const std::exception& e) { return err_resp("ASSISTANT_STORE_FAILED", e.what(), 503); }
    });

    CROW_ROUTE(app, "/api/pi/post-run/dismissal").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body || !body->is_object()) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        for (const auto* key : {"run_id", "run_uid", "run_dir", "candidate_id", "advice_event_id", "request_id"})
            if (body->contains(key) && !(*body)[key].is_string())
                return err_resp("BAD_REQUEST", std::string(key) + " must be a string", 400);
        const std::string run_id = body->value("run_id", std::string());
        const std::string run_dir_hint = body->value("run_dir", std::string());
        const std::string expected_uid = body->value("run_uid", std::string());
        const std::string candidate_id = body->value("candidate_id", std::string());
        const std::string advice_event_id = body->value("advice_event_id", std::string());
        const std::string request_id = body->value("request_id", std::string());
        if (run_id.empty() || candidate_id.empty() || candidate_id.size() > 128 ||
            advice_event_id.empty() || request_id.empty() || request_id.size() > 128 ||
            request_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos)
            return err_resp("BAD_REQUEST", "run_id, candidate_id, advice_event_id and an identifier request_id are required", 400);
        if (!body->contains("dismissed") || !(*body)["dismissed"].is_boolean())
            return err_resp("BAD_REQUEST", "dismissed must be a boolean", 400);
        fs::path run_dir;
        if (auto err = resolve_hinted_run_dir(state, run_id, run_dir_hint, run_dir)) return std::move(*err);
        std::error_code ec;
        if (!fs::is_directory(run_dir, ec)) return err_resp("NOT_FOUND", "run not found", 404);
        if (auto denied = validate_path(state, run_dir, true)) return std::move(*denied);
        std::string uid;
        try {
            const auto provenance = read_json_file_opt(run_dir / "artifacts/pi_run_provenance.json").value_or(json::object());
            const auto identity = PiRunIndex(pi_storage_dir(state)).resolve(run_dir, provenance.value("run_uid", std::string()));
            uid = identity["run_uid"];
            if (!expected_uid.empty() && expected_uid != uid)
                return err_resp("RUN_IDENTITY_CONFLICT", "Run path does not match requested context UID", 409);
        } catch (const PiRunIdentityConflict& e) { return err_resp("RUN_IDENTITY_CONFLICT", e.what(), 409); }
          catch (const std::exception& e) { return err_resp("RUN_CONTEXT_UNAVAILABLE", e.what(), 503); }

        try {
            PiAssistantEventStore events(pi_storage_dir(state));
            const auto advice_event = events.get(advice_event_id);
            if (!advice_event || advice_event->value("run_uid", std::string()) != uid ||
                advice_event->value("kind", std::string()) != "jev_post_run")
                return err_resp("ADVICE_EVENT_NOT_FOUND", "Advice event does not belong to this run", 404);
            const json advice = advice_event->value("result", json::object()).value("advice", json::object());
            bool candidate_present = false;
            if ((*body)["dismissed"].get<bool>()) {
                for (const auto& suggestion : advice.value("suggestions", json::array()))
                    if (suggestion.value("candidate_id", std::string()) == candidate_id) candidate_present = true;
            } else {
                const json result = advice_event->value("result", json::object());
                for (const auto& event : result.value("dismissed_candidates", json::array()))
                    if (event.value("result", json::object()).value("candidate_id", std::string()) == candidate_id)
                        candidate_present = true;
            }
            if (!candidate_present)
                return err_resp("CANDIDATE_NOT_IN_ADVICE", "Candidate is not part of this advice event", 409);
            const std::string source_patch_id = read_json_file_opt(run_dir / "artifacts/pi_run_provenance.json")
                .value_or(json::object()).value("jev_proposal_id", std::string());
            const std::string event_id = "jev_dismiss_" + sha256_prefixed(
                uid + ":" + candidate_id + ":" + advice_event_id + ":" + request_id).substr(7);
            const auto event = events.record_post_run_dismissal(event_id, uid, candidate_id,
                advice_event_id, source_patch_id, (*body)["dismissed"].get<bool>());
            return json_resp({{"ok", true}, {"dismissal", event}});
        } catch (const std::invalid_argument& e) { return err_resp("DISMISSAL_CONFLICT", e.what(), 409); }
          catch (const std::exception& e) { return err_resp("DISMISSAL_STORE_FAILED", e.what(), 503); }
    });

    CROW_ROUTE(app, "/api/scan/decisions/<string>").methods("GET"_method)
    ([state, holder](const crow::request&, const std::string& id) {
        std::string error;
        auto svc = get_service(state, *holder, error);
        if (!svc) return err_resp("DECISIONS_UNAVAILABLE", error, 503);
        auto v = svc->view(id);
        if (!v) return err_resp("NOT_FOUND", "proposal not found", 404);
        return json_resp(*v);
    });

    // Applies the selected proposals of one request together (all or nothing) to the caller's draft.
    CROW_ROUTE(app, "/api/scan/decisions/apply-batch").methods("POST"_method)
    ([state, holder](const crow::request& req) {
        auto body = parse_body(req);
        if (!body || !body->is_object()) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        std::vector<std::string> ids;
        if (body->contains("proposal_ids") && (*body)["proposal_ids"].is_array())
            for (const auto& i : (*body)["proposal_ids"]) if (i.is_string()) ids.push_back(i.get<std::string>());
        std::string error;
        auto svc = get_service(state, *holder, error);
        if (!svc) return err_resp("DECISIONS_UNAVAILABLE", error, 503);
        crow::response failure;
        auto advice = build_request(state, *body, failure);
        if (!advice) return failure;
        ServiceResult r = svc->apply_many(ids, *advice);
        if (r.http_status == 200 && r.body.contains("patched_config")) {
            r.body["patched_yaml"] = yaml_dump(r.body["patched_config"]);
            r.body.erase("patched_config");
        }
        return json_resp(r.body, r.http_status);
    });

    CROW_ROUTE(app, "/api/scan/decisions/<string>/apply").methods("POST"_method)
    ([state, holder](const crow::request& req, const std::string& id) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        std::string error;
        auto svc = get_service(state, *holder, error);
        if (!svc) return err_resp("DECISIONS_UNAVAILABLE", error, 503);
        crow::response failure;
        auto advice = build_request(state, *body, failure);
        if (!advice) return failure;
        ServiceResult r = svc->apply(id, *advice);
        if (r.http_status == 200 && r.body.contains("patched_config")) {
            // The draft is handed back as YAML; the caller decides whether/when to save it (no file is written here).
            r.body["patched_yaml"] = yaml_dump(r.body["patched_config"]);
            r.body.erase("patched_config");
        }
        return json_resp(r.body, r.http_status);
    });
}

} // namespace tile_compile::routes
