#include "routes/pi_decision_routes.hpp"

#include "routes/route_utils.hpp"
#include "services/ai_service.hpp"
#include "services/pi/pi_decision_outcome.hpp"
#include "services/pi/pi_decision_service.hpp"
#include "services/pi/pi_post_run.hpp"
#include "services/pi/pi_scan_manifest.hpp"
#include "services/pi/pi_json_io.hpp"
#include "services/pi/pi_storage_paths.hpp"
#include "services/scan_metrics_cache.hpp"
#include "services/scan_summary.hpp"
#include "subprocess_manager.hpp"
#include "time_utils.hpp"

#include <algorithm>
#include <mutex>
#include <random>
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

// One service per process, created on first use so a missing catalog only breaks Jev routes.
struct ServiceHolder {
    std::mutex m;
    std::shared_ptr<DecisionService> svc;
    std::string error;
};

std::shared_ptr<DecisionService> get_service(const std::shared_ptr<AppState>& state, ServiceHolder& h, std::string& error) {
    std::lock_guard<std::mutex> lock(h.m);
    if (h.svc) return h.svc;
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
        h.svc = std::make_shared<DecisionService>(tile_compile::pi::pi_storage_dir(state) / "pi_decisions", load_decision_catalog(*cands, *prot), deps);
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
        std::string id;
        try { id = svc->create(); } catch (const std::exception& e) { return err_resp("STORE_FAILED", e.what(), 500); }
        AdviceRequest copy = *advice;
        std::thread([svc, id, copy]() { svc->run(id, copy); }).detach();
        return json_resp({{"proposal_id", id}, {"state", "running"}}, 202);
    });

    // Post-run advice, only when the user asks for it. Read-only: the run directory is not touched, no model is called and
    // no run is started. A suggestion still needs the resume dry run (feasibility) and a separate start by the user.
    CROW_ROUTE(app, "/api/pi/post-run/advice").methods("POST"_method)
    ([state, holder](const crow::request& req) {
        auto body = parse_body(req);
        if (!body || !body->is_object()) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string run_id = body->value("run_id", std::string());
        if (run_id.empty() || run_id.find('/') != std::string::npos || run_id.find("..") != std::string::npos)
            return err_resp("BAD_REQUEST", "run_id (a run directory name) is required", 400);
        fs::path run_dir;
        try { run_dir = state->runtime.resolve_run_dir(run_id); } catch (const std::exception&) { return err_resp("NOT_FOUND", "run not found", 404); }
        std::error_code ec;
        if (!fs::is_directory(run_dir, ec)) return err_resp("NOT_FOUND", "run not found", 404);
        json run_config = nullptr;
        try { run_config = yaml_text_to_json(read_file_str(run_dir / "config.yaml")); } catch (const std::exception&) { run_config = nullptr; }
        std::string error;
        auto svc = get_service(state, *holder, error);
        if (!svc) return err_resp("DECISIONS_UNAVAILABLE", error, 503);
        std::vector<std::string> dismissed;
        if (body->contains("dismissed_candidates") && (*body)["dismissed_candidates"].is_array())
            for (const auto& d : (*body)["dismissed_candidates"]) if (d.is_string()) dismissed.push_back(d.get<std::string>());
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
        return json_resp({{"state", ps}, {"advice", advise_post_run(ps, run_config, policy, svc->catalog(), validator, dismissed)}});
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
