#include "pi_decision_test_support.hpp"
#include "services/pi/pi_pre_rules.hpp"

#include <algorithm>
#include <cstdio>

using namespace pi_test;
using nlohmann::json;

namespace {
bool has_reason(const json& excluded, const std::string& id, const std::string& reason) {
    for (const auto& e : excluded)
        if (e["candidate_id"] == id)
            for (const auto& r : e["reasons"]) if (r == reason) return true;
    return false;
}
bool is_offered(const PreRunCandidates& c, const std::string& id) {
    for (const auto& a : c.applicable) if (a["candidate_id"] == id) return true;
    return false;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path repo = argc > 4 ? argv[4] : "..";
        const DecisionCatalog catalog = load_real_catalog(repo);
        const json cfg = {{"global_metrics", {{"adaptive_weights", false}}}};
        const auto state_r = build_pre_run_decision_state(inputs_for(spread_frames(10)));
        const json& state = state_r.state;
        const auto ok = accepting_validator();

        // ---- production defaults: nothing frozen, experimental off -> only the baselines ----
        {
            const auto c = build_pre_run_candidates(state, cfg, DecisionPolicy{}, catalog, ok);
            expect_equal(static_cast<long>(c.applicable.size()), 2L, "defaults offer only baselines");
            expect_true(is_offered(c, "keep_current") && is_offered(c, "insufficient_evidence"), "baselines always offered");
            expect_true(has_reason(c.excluded, "enable_adaptive_weights", "experimental_not_enabled"), "experimental reason");
            expect_true(has_reason(c.excluded, "enable_adaptive_weights", "policy_thresholds_not_frozen"), "thresholds-not-frozen reason (no invented default)");
            expect_equal(static_cast<long>(c.allowed_ids().size()), 2L, "allowed ids = baselines");
            expect_equal(c.policy_version, std::string(kDecisionPolicyVersion), "policy version reported");
        }

        // ---- the shipped catalog never offers the rejected candidate, whatever the policy says ----
        {
            const DecisionCatalog shipped = load_real_catalog(repo, false);
            const auto c = build_pre_run_candidates(state, cfg, frozen_test_policy(true), shipped, ok);
            bool offered = false;
            for (const auto& a : c.applicable) offered = offered || a["candidate_id"] == "enable_adaptive_weights";
            expect_true(!offered, "enable_adaptive_weights is not offered (experimental on, thresholds frozen)");
            expect_true(has_reason(c.excluded, "enable_adaptive_weights", "candidate_rejected"), "and the exclusion reason says it is rejected");
        }

        // ---- experimental + frozen test thresholds: offered, flagged ----
        {
            const auto c = build_pre_run_candidates(state, cfg, frozen_test_policy(), catalog, ok);
            expect_equal(static_cast<long>(c.applicable.size()), 3L, "three candidates offered (baselines + adaptive weights)");
            expect_equal(static_cast<long>(c.excluded.size()), 3L, "the camera-table candidate and the two reconstruction candidates are excluded (this draft has neither their control levels nor a DWARF II camera)");
            expect_true(has_reason(c.excluded, "set_pixfrac__1", "precondition_failed:config_equals:reconstruction.drizzle.pixfrac=0.8"), "pixfrac is not offered without its control level");
            expect_true(has_reason(c.excluded, "set_clip_sigmas", "precondition_failed:config_equals:reconstruction.clipping.clip_sigma_low=4.0"), "clipping is not offered without its control level");
            {
                const json cfg_ctl = {{"global_metrics", {{"adaptive_weights", false}}},
                                      {"reconstruction", {{"drizzle", {{"pixfrac", 0.8}}}, {"clipping", {{"clip_sigma_low", 4.0}, {"clip_sigma_high", 4.0}}}}}};
                const auto r = build_pre_run_candidates(state, cfg_ctl, frozen_test_policy(), catalog, ok);
                expect_true(is_offered(r, "set_pixfrac__1") && is_offered(r, "set_clip_sigmas"), "both reconstruction candidates are offered from the control levels");
                const json info = provider_candidate_info(r, catalog);
                expect_true(info["descriptions"].contains("set_pixfrac__1") && info["descriptions"].contains("set_clip_sigmas"), "and described to the provider");
                expect_true(!build_pre_run_candidates(state, cfg_ctl, DecisionPolicy{}, catalog, ok).allowed_ids().empty() &&
                            build_pre_run_candidates(state, cfg_ctl, DecisionPolicy{}, catalog, ok).allowed_ids().size() == 2,
                            "in production (experimental off) they are not offered");
            }
            expect_true(has_reason(c.excluded, "set_sensor_profile_dwarf_ii", "evidence_below_threshold:camera_match"), "excluded because the camera does not match the table");
            {
                json fr = spread_frames(10);
                for (auto& f : fr) f["header"]["camera"] = "DWARF II";
                const auto dwarf_state = build_pre_run_decision_state(inputs_for(fr)).state;
                const auto d = build_pre_run_candidates(dwarf_state, cfg, frozen_test_policy(), catalog, ok);
                bool offered = false;
                for (const auto& a : d.applicable) if (a["candidate_id"] == "set_sensor_profile_dwarf_ii") offered = a["updates"].size() == 2 && a["requires_review"] == true && a["experimental"] == true;
                expect_true(offered, "a DWARF II header with an unset profile offers the sensor-profile candidate (experimental, review)");
                expect_true(build_pre_run_candidates(dwarf_state, cfg, DecisionPolicy{}, catalog, ok).allowed_ids().size() == 2,
                            "in production (experimental off, thresholds not frozen) it is still not offered");
            }
            const json* enable = nullptr;
            for (const auto& a : c.applicable) if (a["candidate_id"] == "enable_adaptive_weights") enable = &a;
            {
                const json info = provider_candidate_info(c, catalog);
                expect_true(info["descriptions"].size() == 1 && info["descriptions"].contains("enable_adaptive_weights"), "descriptions only for offered change candidates");
                expect_true(info["facts"]["enable_adaptive_weights"]["metric_agreement"].is_number(), "resolved measurements are passed as plain facts");
                expect_true(info.dump().find("Cam") == std::string::npos, "no camera name in what the provider is told");
                json fr = spread_frames(10);
                for (auto& f : fr) f["header"]["camera"] = "Secret Camera 9000";
                const auto cs = build_pre_run_decision_state(inputs_for(fr)).state;
                const json info2 = provider_candidate_info(build_pre_run_candidates(cs, cfg, frozen_test_policy(), catalog, ok), catalog);
                expect_true(info2.dump().find("Secret") == std::string::npos, "camera name never reaches provider info");
            }
            expect_true(enable && (*enable)["experimental"] == true && (*enable)["requires_review"] == true, "flagged experimental + review");
            expect_true((*enable)["updates"][0]["old_value"] == false && (*enable)["updates"][0]["value"] == true, "concrete update");
            expect_true((*enable)["rationale"]["params"].contains("metric_agreement"), "rationale from real measurement");
            // offered == accepted later: one code path
            for (const auto& a : c.applicable) {
                if (a["updates"].empty()) continue;
                json cand = {{"candidate_id", a["candidate_id"]}, {"candidate_version", a["candidate_version"]},
                             {"updates", json::array({{{"path", a["updates"][0]["path"]}, {"value", a["updates"][0]["value"]}}})}};
                expect_true(validate_decision_candidate(cand, state, cfg, frozen_test_policy(), catalog, ok).ok, "offered candidate validates identically");
            }
        }

        // ---- exclusions ----
        {
            const auto blocked = build_pre_run_candidates(build_pre_run_decision_state(PreRunDecisionInputs{}).state, cfg, frozen_test_policy(), catalog, ok);
            expect_true(has_reason(blocked.excluded, "enable_adaptive_weights", "blocked:no_readable_frames"), "blocking state excludes real candidates");
            expect_true(is_offered(blocked, "keep_current") && is_offered(blocked, "insufficient_evidence"), "baselines survive a blocked state");

            const json active = {{"global_metrics", {{"adaptive_weights", true}}}};
            expect_true(has_reason(build_pre_run_candidates(state, active, frozen_test_policy(), catalog, ok).excluded, "enable_adaptive_weights", "already_active"), "already active is not a change");

            auto locked_in = inputs_for(spread_frames(10)); locked_in.locked_paths = {"global_metrics.adaptive_weights"};
            expect_true(has_reason(build_pre_run_candidates(build_pre_run_decision_state(locked_in).state, cfg, frozen_test_policy(), catalog, ok).excluded,
                                   "enable_adaptive_weights", "path_locked"), "lock excludes the whole candidate");

            json mix = spread_frames(6, "L");
            for (auto f : spread_frames(6, "R")) { f["file_name"] = "r_" + f["file_name"].get<std::string>(); mix.push_back(f); }
            expect_true(has_reason(build_pre_run_candidates(build_pre_run_decision_state(inputs_for(mix)).state, cfg, frozen_test_policy(), catalog, ok).excluded,
                                   "enable_adaptive_weights", "mixed_groups_no_single_evidence"), "mixed groups are never pooled");

            DecisionPolicy strict = frozen_test_policy(); strict.min_metric_agreement = 0.99;
            expect_true(has_reason(build_pre_run_candidates(state, cfg, strict, catalog, ok).excluded, "enable_adaptive_weights", "evidence_below_threshold:metric_agreement"), "below spread threshold");

            json partial = spread_frames(10);
            for (int i = 6; i < 10; ++i) partial[i] = make_frame(i, "L", 0, 0, false);  // 4 unreadable
            expect_true(has_reason(build_pre_run_candidates(build_pre_run_decision_state(inputs_for(partial)).state, cfg, frozen_test_policy(), catalog, ok).excluded,
                                   "enable_adaptive_weights", "evidence_below_threshold:measurement_coverage"), "low measurement coverage");

            expect_true(has_reason(build_pre_run_candidates(state, cfg, frozen_test_policy(), catalog,
                                                            [](const json&) { return ConfigCheck{false, "no"}; }).excluded,
                                   "enable_adaptive_weights", "config_invalid"), "config validator veto excludes the candidate");
        }

        // ---- deterministic and rule-order independent ----
        {
            DecisionCatalog reversed = catalog;
            std::reverse(reversed.candidates.begin(), reversed.candidates.end());
            const auto a = build_pre_run_candidates(state, cfg, frozen_test_policy(), catalog, ok);
            const auto b = build_pre_run_candidates(state, cfg, frozen_test_policy(), reversed, ok);
            expect_true(a.applicable == b.applicable && a.excluded == b.excluded, "catalog order does not change the result");
            std::vector<std::string> ids = a.allowed_ids();
            expect_true(std::is_sorted(ids.begin(), ids.end()), "applicable sorted by id");
            expect_true(build_pre_run_candidates(state, cfg, frozen_test_policy(), catalog, ok).applicable == a.applicable, "repeatable");
        }

        std::puts("pi_pre_rules: all checks passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
