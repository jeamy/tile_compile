#pragma once
// Shared synthetic-state helpers for the PI decision tests (no real data, no I/O besides reading the
// committed catalog files).
#include "backend_test_harness.hpp"
#include "services/pi/pi_decision_policy.hpp"
#include "services/pi/pi_decision_state.hpp"

#include <filesystem>
#include <nlohmann/json.hpp>

namespace pi_test {
using nlohmann::json;
using namespace tile_compile::pi;

// The production catalog marks enable_adaptive_weights as rejected. Most tests use it as their stock change candidate, so by
// default it is reopened (experimental_only); pass reopen_rejected=false to get the catalog exactly as shipped.
inline DecisionCatalog load_real_catalog(const std::filesystem::path& repo, bool reopen_rejected = true) {
    const auto dir = repo / "web_backend_cpp/config/pi_decisions";
    json cands = json::parse(slurp_file(dir / "candidates_v1.json"));
    if (reopen_rejected)
        for (auto& c : cands["candidates"])
            if (c.value("applicability", "") == "rejected") c["applicability"] = "experimental_only";
    return load_decision_catalog(cands, json::parse(slurp_file(dir / "protected_paths_v1.json")));
}

inline json make_frame(int idx, const char* filter, double fwhm, double noise, bool ok = true) {
    json f = {{"frame_index", idx}, {"file_name", "f" + std::to_string(idx) + ".fits"}, {"ok", ok},
              {"header", {{"exposure_seconds", 60.0}, {"gain", 100}, {"camera", "Cam"}}}};
    if (filter) f["header"]["filter"] = filter;
    if (ok) {
        f["background"] = 500.0 + 10.0 * noise + 0.01 * idx; f["noise"] = noise; f["gradient_energy"] = 0.01; f["sky_gradient"] = 0.03;
        f["fwhm"] = fwhm; f["roundness"] = 1.0; f["star_count"] = 100;
    }
    return f;
}

// n frames of ONE acquisition group whose fwhm/noise vary, so relative spreads are computable.
inline json spread_frames(int n, const char* filter = "L") {
    json fr = json::array();
    for (int i = 0; i < n; ++i) fr.push_back(make_frame(i, filter, 2.0 + 0.3 * (i % 5), 6.0 + 0.5 * (i % 5) + 0.05 * (i % 4)));
    return fr;
}

inline PreRunDecisionInputs inputs_for(json frames, bool adaptive = false) {
    PreRunDecisionInputs in;
    in.scan = {{"frames_detected", static_cast<int>(frames.size())}, {"color_mode", "OSC"}, {"bayer_pattern", "GRBG"},
               {"errors", json::array()}, {"warnings", json::array()}};
    in.metrics = {{"frames", frames}};
    in.base_config = {{"global_metrics", {{"adaptive_weights", adaptive}}}};
    in.scan_id = "scan-1";
    in.software_version = "test";
    return in;
}

// A policy with explicit, test-only thresholds (the production policy has none frozen yet).
inline DecisionPolicy frozen_test_policy(bool experimental = true) {
    DecisionPolicy p;
    p.allow_experimental = experimental;
    p.min_measurement_coverage = 0.9;
    p.min_metric_agreement = 0.05;
    return p;
}

inline ConfigValidator accepting_validator() {
    return [](const json&) { return ConfigCheck{true, ""}; };
}

} // namespace pi_test
