#pragma once

#include "services/pi/pi_decision_policy.hpp"
#include "services/pi/pi_decision_state.hpp"

#include <filesystem>
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace tile_compile::pi {

// Everything the service needs from the outside, injected so the whole flow is testable without a
// sidecar, a CLI or a clock.
struct DecisionServiceDeps {
    // Calls the agent sidecar: (method "GET"/"POST", endpoint e.g. "/decisions", payload or null).
    // Must throw std::runtime_error on transport failure and DecisionSidecarHttpError-like errors
    // via the `sidecar_http_status` convention below (see call_sidecar()).
    std::function<nlohmann::json(const std::string& method, const std::string& endpoint, const nlohmann::json& payload)> sidecar;
    ConfigValidator validate_config;
    std::function<std::string()> now_iso;
    std::function<std::string()> new_id;   // unique proposal id
};

// HTTP error from the sidecar with the parsed body (thrown by the production `sidecar` adapter).
struct SidecarHttpError : std::runtime_error {
    long status;
    nlohmann::json body;
    SidecarHttpError(long s, nlohmann::json b) : std::runtime_error("sidecar http " + std::to_string(s)), status(s), body(std::move(b)) {}
};

struct AdviceRequest {
    nlohmann::json scan;
    nlohmann::json metrics;
    nlohmann::json base_config;                  // the config DRAFT the proposal would modify
    std::vector<std::string> locked_paths;
    nlohmann::json session_context = nlohmann::json::object();
    nlohmann::json capabilities = nlohmann::json::object();
    nlohmann::json dataset_manifest = nlohmann::json::array();
    std::string scan_id;
    std::string software_version;
};

struct ServiceResult {
    int http_status = 200;
    nlohmann::json body;
};

// Orchestrates one pre-run Jev advice: state (M1) -> candidates (M2) -> optional sidecar call ->
// resolve (M2) -> persistence. All files of a proposal live in <decisions_dir>/<proposal_id>/ and
// never touch the PiMemoryStore or any run directory.
class DecisionService {
public:
    DecisionService(std::filesystem::path decisions_dir, DecisionCatalog catalog, DecisionServiceDeps deps,
                    std::string question_set_version = "decision-questions.v1");

    // Registers a proposal as running (status.json) and returns its id.
    std::string create();
    // Runs the advice synchronously; never throws (failures are recorded as status "failed").
    void run(const std::string& proposal_id, const AdviceRequest& request);
    // API view of a proposal, or nullopt if unknown.
    std::optional<nlohmann::json> view(const std::string& proposal_id) const;
    // Applies a validated proposal to the caller's config DRAFT (never to a file/run). CAS on the
    // draft via re-derived hashes; idempotent for an identical repeat.
    ServiceResult apply(const std::string& proposal_id, const AdviceRequest& current);

    // Optional local policy override (`policy_override.json` in decisions_dir): freezes thresholds for
    // evaluation. Without it no threshold is frozen and only baselines are ever offered.
    DecisionPolicy policy_for_mode(const std::string& mode, bool allow_experimental_flag) const;

private:
    std::filesystem::path dir_;
    DecisionCatalog catalog_;
    DecisionServiceDeps deps_;
    std::string question_set_version_;
    mutable std::mutex mutex_;

    std::filesystem::path pdir(const std::string& id) const { return dir_ / id; }
    void event(const std::string& id, const std::string& name, const nlohmann::json& extra = nlohmann::json::object()) const;
};

} // namespace tile_compile::pi
