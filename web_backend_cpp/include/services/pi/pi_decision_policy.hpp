#pragma once

#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace tile_compile::pi {

inline constexpr const char* kDecisionPolicyVersion = "pi.decision-policy.v1";

// Thresholds are std::optional on purpose: the docs leave their release values open (M5). A
// nullopt threshold means "not frozen" and makes every candidate that needs that evidence
// inapplicable (reason `policy_thresholds_not_frozen`) -- no default is ever invented here.
struct DecisionPolicy {
    std::string version = kDecisionPolicyVersion;
    // Experimental candidates are only applicable when explicitly enabled
    // (allow_experimental_suggestions, default false).
    bool allow_experimental = false;
    std::optional<double> min_measurement_coverage;  // read_ok / measured, in (0, 1]
    std::optional<double> min_metric_agreement;      // median pairwise Spearman rho, in (-1, 1]
};

struct DecisionCatalog {
    int version = 0;
    nlohmann::json candidates = nlohmann::json::array();          // candidates_v1.json["candidates"]
    std::vector<std::string> protected_prefixes;                  // protected_paths_v1.json
    nlohmann::json released_paths = nlohmann::json::array();       // {path, condition}: reachable only via a catalog entry meeting the condition
};

// Parses and structurally validates the two committed catalog files. Throws
// std::invalid_argument on: missing fields, duplicate ids, or ANY catalog candidate update that
// targets a protected path (the catalog itself may never reach a guarded parameter).
DecisionCatalog load_decision_catalog(const nlohmann::json& candidates_v1, const nlohmann::json& protected_paths_v1);

struct ConfigCheck {
    bool valid = false;
    std::string detail;
};
// Validates a COMPLETE merged config (production: the existing `validate-config` CLI, wired in
// M4; tests inject a fake). Must be side-effect free.
using ConfigValidator = std::function<ConfigCheck(const nlohmann::json& merged_config)>;

// A candidate as it flows through the system: only id + version + updates are trusted input; every
// other field is re-derived from the catalog.
struct CandidateValidation {
    std::string candidate_id;
    bool ok = false;
    bool experimental = false;
    bool requires_review = false;
    std::vector<std::string> reasons;         // sorted, unique; empty iff ok
    nlohmann::json updates = nlohmann::json::array();   // [{path, old_value, value, group_id}] iff ok
    nlohmann::json merged_config;             // current config with ALL updates applied iff ok
    nlohmann::json evidence_refs = nlohmann::json::array();
    nlohmann::json rationale = nlohmann::json::object();  // {text_key, params} from real measurements
    // Which check families passed (feeds pi.config-proposal.v1 `validation`).
    bool schema_ok = false, policy_ok = false, evidence_ok = false, config_ok = false;
};

// All-or-nothing: every problem is collected (no short-circuit, so rule order cannot change the
// result) and ONE failure rejects the whole group; nothing is salvaged from a failed candidate.
// `candidate` needs candidate_id, candidate_version, updates[{path,value[,old_value]}]. Path/value
// pairs must equal the catalog entry exactly -- a model or client cannot introduce a new path or
// value.
CandidateValidation validate_decision_candidate(const nlohmann::json& candidate,
                                                const nlohmann::json& state,
                                                const nlohmann::json& current_config,
                                                const DecisionPolicy& policy,
                                                const DecisionCatalog& catalog,
                                                const ConfigValidator& validate_config);

struct ResolveContext {
    std::string proposal_id;
    std::string created_at;          // ISO-8601 UTC, supplied by the caller (no clock in here)
    std::string state_hash;
    std::string question_set_version;
    std::string model_requested;
    nlohmann::json model_reported = nullptr;
};

// Turns a normalized pi.decisions.response.v1 into a pi.config-proposal.v1. The choice is
// re-validated here regardless of what the provider claims; probabilities/confidence are never a
// gate. Statuses: validated | no_change (keep_current) | abstain (insufficient_evidence) |
// unavailable (provider) | rejected (invalid response, unknown/inapplicable candidate, or failed
// validation). `updates` is empty for every status except `validated`.
nlohmann::json resolve_decision(const nlohmann::json& response,
                                const nlohmann::json& state,
                                const nlohmann::json& current_config,
                                const DecisionPolicy& policy,
                                const DecisionCatalog& catalog,
                                const ConfigValidator& validate_config,
                                const ResolveContext& ctx);

// Why a stored proposal may no longer be applied: state/scan/config/locks/dataset identity, policy
// or candidate version changed. Empty result = still fresh.
std::vector<std::string> stale_reasons(const nlohmann::json& proposal,
                                       const nlohmann::json& current_state,
                                       const std::string& current_state_hash,
                                       const DecisionPolicy& policy,
                                       const DecisionCatalog& catalog);

// Helpers shared with pi_pre_rules / tests.
bool path_is_protected(const DecisionCatalog& catalog, const std::string& path);
bool path_is_locked(const std::vector<std::string>& locked_paths, const std::string& path);
nlohmann::json config_get(const nlohmann::json& config, const std::string& dotted_path);
void config_set(nlohmann::json& config, const std::string& dotted_path, const nlohmann::json& value);
bool json_values_equal(const nlohmann::json& a, const nlohmann::json& b);  // 1 == 1.0

} // namespace tile_compile::pi
