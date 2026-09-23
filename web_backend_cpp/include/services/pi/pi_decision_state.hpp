#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace tile_compile::pi {

inline constexpr const char* kDecisionStateSchemaVersion = "pi.decision-state.v1";
inline constexpr const char* kDecisionStateMethodVersion = "scan-metrics:cli_main.v1";

// Policy knobs the state builder needs but must not silently invent (Zielbild §4: nothing is
// treated as measured that was not). Every value used is echoed into state["policy"] so a
// downstream consumer (and the state hash) sees exactly which floor was applied.
struct DecisionStateOptions {
    // A spread (MAD / relative spread) computed from fewer valid frames than this is reported as
    // not_applicable(reason=insufficient_sample), never as a number. The docs leave the release
    // value open (M5); this default only keeps tiny groups from producing a statistically
    // meaningless number that looks valid.
    int min_valid_for_spread = 5;
};

struct PreRunDecisionInputs {
    // scan-result JSON as produced by the scan route (frames_detected, color_mode, bayer_pattern,
    // errors[], warnings[]) and scan-metrics JSON (frames[], aggregate, session_geometry).
    nlohmann::json scan;
    nlohmann::json metrics;
    // Effective config the proposal would modify. Must be a JSON object.
    nlohmann::json base_config;
    // Facts the user stated, each {"value": ..., "source": "user"}. Never merged with measurements.
    nlohmann::json session_context = nlohmann::json::object();
    // Local system facts (booleans, e.g. {"astap_available": true}); no paths.
    nlohmann::json capabilities = nlohmann::json::object();
    // Dotted config paths the user locked.
    std::vector<std::string> locked_paths;
    // Optional dataset manifest: [{"id","size","mtime","sha256"?}] for the SELECTED inputs. Absent
    // -> the fingerprint is derived from scan-metrics frame file names only (identity_strength
    // "metadata").
    nlohmann::json dataset_manifest = nlohmann::json::array();
    std::string scan_id;
    std::string software_version;
    DecisionStateOptions options;
};

struct PreRunDecisionResult {
    // pi.decision-state.v1 (see web_backend_cpp/config/pi_decisions/schemas).
    nlohmann::json state;
    // Non-blocking structured findings: [{code, severity, detail}]. Blocking ones are also in
    // state["blocking_findings"] (codes only).
    nlohmann::json findings = nlohmann::json::array();
    // "sha256:<hex>" over the canonical semantic state (no volatile fields).
    std::string state_hash;
    // Allowlist projection sent to an external provider: no paths, secrets, camera/target text.
    nlohmann::json provider_projection;
};

// Pure: no I/O, no model calls, no clock. Same semantic inputs -> byte-identical state and hash.
PreRunDecisionResult build_pre_run_decision_state(const PreRunDecisionInputs& in);

// Canonical serialization: object keys sorted, arrays keep their order, integral doubles are not
// rewritten, NaN/Infinity anywhere -> std::invalid_argument. Exposed for tests and M2/M3 reuse.
std::string canonical_json_dump(const nlohmann::json& value);
std::string sha256_prefixed(const std::string& bytes);

// Per-frame symmetric axis deviation |ln(roundness)| (roundness = fwhm_y / fwhm_x, may exceed 1).
// Returns false (out untouched) for non-finite or <= 0 input; never a fabricated 0.
bool axis_asymmetry(double roundness, double& out);

} // namespace tile_compile::pi
