#include "pi_decision_test_support.hpp"
#include "pi_json_schema_check.hpp"

#include <cstdio>

using namespace pi_test;
using nlohmann::json;

namespace {
bool has(const std::vector<std::string>& v, const std::string& x) { return std::find(v.begin(), v.end(), x) != v.end(); }
json enable_candidate() {
    return {{"candidate_id", "enable_adaptive_weights"}, {"candidate_version", 2},
            {"updates", json::array({{{"path", "global_metrics.adaptive_weights"}, {"value", true}}})}};
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path repo = argc > 4 ? argv[4] : "..";
        const DecisionCatalog catalog = load_real_catalog(repo);
        const json proposal_schema = json::parse(slurp_file(repo / "web_backend_cpp/config/pi_decisions/schemas/pi.config-proposal.v1.schema.json"));
        const auto good_state_r = build_pre_run_decision_state(inputs_for(spread_frames(10)));
        const json& good = good_state_r.state;
        const json cfg = {{"global_metrics", {{"adaptive_weights", false}}}};
        const DecisionPolicy pol = frozen_test_policy();

        // ---- catalog loading is fail-closed ----
        {
            expect_equal(static_cast<long>(catalog.version), 1L, "catalog version");
            expect_true(!catalog.protected_prefixes.empty(), "protected prefixes loaded");
            const auto dir = repo / "web_backend_cpp/config/pi_decisions";
            const json cands = json::parse(slurp_file(dir / "candidates_v1.json"));
            const json prot = json::parse(slurp_file(dir / "protected_paths_v1.json"));
            auto throws = [&](const json& c, const json& p) {
                try { load_decision_catalog(c, p); } catch (const std::invalid_argument&) { return true; }
                return false;
            };
            json bad = cands;
            bad["candidates"][2]["updates"].push_back({{"path", "reconstruction.coverage_gate.min_frames"}, {"value", 1}});
            expect_true(throws(bad, prot), "catalog candidate targeting a protected path is refused at load");
            json no_base = cands;
            no_base["candidates"].erase(0);
            expect_true(throws(no_base, prot), "catalog without keep_current refused");
            json dup = cands;
            dup["candidates"].push_back(cands["candidates"][0]);
            expect_true(throws(dup, prot), "duplicate id refused");
            json wrong = cands; wrong["schema_version"] = "x";
            expect_true(throws(wrong, prot), "wrong catalog schema_version refused");
            json empty_prot = prot; empty_prot["protected_path_prefixes"] = json::array();
            expect_true(throws(cands, empty_prot), "empty protected list refused (fail closed)");
            json no_tier = prot; no_tier["protected_path_prefixes"][0].erase("tier");
            expect_true(throws(cands, no_tier), "protected entry without tier refused");
            json v1 = prot; v1["schema_version"] = "pi.protected-paths.v1";
            expect_true(throws(cands, v1), "old protected-paths schema refused");
            json both = prot; both["released_for_candidates"].push_back({{"path", "pcc.k_max"}, {"condition", "x"}});
            expect_true(throws(cands, both), "a path can not be released and protected");
            json badrel = prot; badrel["released_for_candidates"].push_back({{"path", "x.y"}});
            expect_true(throws(cands, badrel), "malformed released entry refused");
            expect_true(!catalog.released_paths.empty(), "released paths loaded for reference");
        }

        // ---- path helpers: prefix boundaries ----
        // ---- value grids: numeric parameters are offered as fixed levels, never as free values ----
        {
            const auto dir = repo / "web_backend_cpp/config/pi_decisions";
            const json base_cands = json::parse(slurp_file(dir / "candidates_v1.json"));
            const json prot = json::parse(slurp_file(dir / "protected_paths_v1.json"));
            auto with_grid = [&](const json& grid, const char* id = "set_test_pixfrac") {
                json c = base_cands;
                c["candidates"].push_back({{"candidate_id", id}, {"candidate_version", 1}, {"group", "drizzle_sampling"},
                                           {"preconditions", json::array()}, {"required_evidence", json::array({"measurement_coverage"})},
                                           {"updates", json::array()}, {"requires_review", true}, {"applicability", "experimental_only"},
                                           {"provider_description", "Set the drizzle pixel fraction."}, {"value_grid", grid}});
                return c;
            };
            auto throws = [&](const json& c) {
                try { load_decision_catalog(c, prot); } catch (const std::invalid_argument&) { return true; }
                return false;
            };
            const json listed = {{"path", "reconstruction.drizzle.pixfrac"}, {"unit", "ratio"}, {"basis", "test grid"},
                                 {"levels", json::array({0.9, 0.6, 0.7, 0.8})}};
            const DecisionCatalog gc = load_decision_catalog(with_grid(listed), prot);
            auto find = [&](const std::string& id) -> const json* {
                for (const auto& c : gc.candidates) if (c["candidate_id"] == id) return &c;
                return nullptr;
            };
            expect_true(find("set_test_pixfrac") == nullptr, "the grid template itself is not selectable");
            expect_true(find("set_test_pixfrac__0p6") && find("set_test_pixfrac__0p7") && find("set_test_pixfrac__0p8") && find("set_test_pixfrac__0p9"), "one candidate per level");
            expect_equal(static_cast<long>(gc.candidates.size()), static_cast<long>(catalog.candidates.size() + 4), "only the levels were added");
            expect_true((*find("set_test_pixfrac__0p7"))["updates"][0]["value"] == 0.7 && (*find("set_test_pixfrac__0p7"))["grid_of"] == "set_test_pixfrac", "level carries the exact path/value");

            json range = listed; range.erase("levels"); range["min"] = 0.5; range["max"] = 0.9; range["step"] = 0.1;
            const DecisionCatalog rc = load_decision_catalog(with_grid(range), prot);
            long n = 0; bool exact = false;
            for (const auto& c : rc.candidates) if (c.value("grid_of", "") == "set_test_pixfrac") { ++n; exact = exact || c["candidate_id"] == "set_test_pixfrac__0p7"; }
            expect_equal(n, 5L, "min/max/step -> 5 levels");
            expect_true(exact, "0.5+0.1*2 is exactly the level 0.7 (rounded, no float drift)");
            json drift = range; drift["min"] = 0.1; drift["max"] = 0.5;
            const DecisionCatalog dc = load_decision_catalog(with_grid(drift), prot);
            bool clean = false;
            for (const auto& c : dc.candidates) if (c["candidate_id"] == "set_test_pixfrac__0p3") clean = c["updates"][0]["value"].get<double>() == 0.3;
            expect_true(clean, "0.1+3*0.1 is stored as exactly 0.3, not 0.30000000000000004");

            json integer_grid = {{"path", "reconstruction.drizzle.robust_passes"}, {"unit", "count"}, {"basis", "test grid"}, {"integer", true}, {"levels", json::array({2, 4})}};
            const DecisionCatalog ic = load_decision_catalog(with_grid(integer_grid, "set_passes"), prot);
            bool int_ok = false;
            for (const auto& c : ic.candidates) if (c["candidate_id"] == "set_passes__4") int_ok = c["updates"][0]["value"].is_number_integer();
            expect_true(int_ok, "integer grid yields integer values");
            json neg = listed; neg["levels"] = json::array({-1.5, 0.000001, 12});
            const DecisionCatalog nc = load_decision_catalog(with_grid(neg), prot);
            bool labels = false;
            for (const auto& c : nc.candidates) labels = labels || c["candidate_id"] == "set_test_pixfrac__m1p5";
            expect_true(labels, "negative levels get a sign-free label");

            json prot_grid = listed; prot_grid["path"] = "pcc.k_max";
            expect_true(throws(with_grid(prot_grid)), "grid on a protected path refused at load");
            json no_unit = listed; no_unit.erase("unit");
            expect_true(throws(with_grid(no_unit)), "grid without unit refused");
            json no_basis = listed; no_basis.erase("basis");
            expect_true(throws(with_grid(no_basis)), "grid without a stated basis refused (bounds must be justified)");
            json bad_step = range; bad_step["step"] = 0;
            expect_true(throws(with_grid(bad_step)), "step 0 refused");
            json inverted = range; inverted["min"] = 1.0; inverted["max"] = 0.5;
            expect_true(throws(with_grid(inverted)), "min > max refused");
            json huge = range; huge["min"] = 0; huge["max"] = 100; huge["step"] = 1;
            expect_true(throws(with_grid(huge)), "too many levels refused");
            json frac = integer_grid; frac["levels"] = json::array({2, 2.5});
            expect_true(throws(with_grid(frac, "set_passes")), "integer grid with a fractional level refused");
            json nan_level = listed; nan_level["levels"] = json::array({0.5, "x"});
            expect_true(throws(with_grid(nan_level)), "non-numeric level refused");
            json empty_levels = listed; empty_levels["levels"] = json::array();
            expect_true(throws(with_grid(empty_levels)), "empty level list refused");
            expect_true(throws(with_grid(listed, "Set_Pixfrac")), "id outside [a-z0-9_] refused (sidecar alphabet)");
            json own_path = with_grid(listed);
            own_path["candidates"].back()["updates"].push_back({{"path", "reconstruction.drizzle.pixfrac"}, {"value", 0.5}});
            expect_true(throws(own_path), "grid path also listed as a fixed update refused");
            expect_true(throws(with_grid(json(nullptr))), "null grid refused");

            // ---- provider_facts in the catalog: only plain, bounded scalars ----
            {
                auto with_facts = [&](const json& facts) {
                    json c = with_grid(listed);
                    c["candidates"].back()["provider_facts"] = facts;
                    return c;
                };
                expect_true(!throws(with_facts({{"confirmed_sessions", 5}, {"evidence", "matched star pairs"}})), "plain scalar facts load");
                expect_true(throws(with_facts(json::object())), "empty facts refused");
                expect_true(throws(with_facts({{"a", "/home/user/x"}})), "a path-shaped string is refused");
                expect_true(throws(with_facts({{"Bad Key", 1}})), "a key outside [a-z0-9_] is refused");
                expect_true(throws(with_facts({{"a", json::array({1})}})), "a non-scalar is refused");
                expect_true(throws(with_facts({{"a", std::string(40, 'x')}})), "a long string is refused");
                json many = json::object();
                for (int i = 0; i < 9; ++i) many["k" + std::to_string(i)] = i;
                expect_true(throws(with_facts(many)), "more than 8 facts refused");
                expect_true(throws(with_facts(json{{"a", 1.0 / 0.0}})), "a non-finite number is refused");
            }

            // ---- validating levels against a current config ----
            const json cfg_px = {{"reconstruction", {{"drizzle", {{"pixfrac", 0.8}}}}}};
            auto level = [&](const char* id, double v) {
                return json{{"candidate_id", id}, {"candidate_version", 1},
                            {"updates", json::array({{{"path", "reconstruction.drizzle.pixfrac"}, {"value", v}}})}};
            };
            auto ok = validate_decision_candidate(level("set_test_pixfrac__0p7", 0.7), good, cfg_px, pol, gc, accepting_validator());
            expect_true(ok.ok && ok.reasons.empty(), "an on-grid level is accepted");
            expect_true(ok.requires_review && ok.experimental, "grid candidates keep the review flags");
            expect_equal(ok.rationale["text_key"].get<std::string>(), "pi.jev.rationale.set_test_pixfrac.v1", "rationale text is keyed by the template, not the level");
            expect_true(ok.rationale["params"]["grid_value"] == 0.7 && ok.rationale["params"]["grid_unit"] == "ratio", "chosen level and unit reach the rationale");
            expect_true(config_get(ok.merged_config, "reconstruction.drizzle.pixfrac") == 0.7 && config_get(cfg_px, "reconstruction.drizzle.pixfrac") == 0.8, "merge changes the copy only");
            auto same = validate_decision_candidate(level("set_test_pixfrac__0p8", 0.8), good, cfg_px, pol, gc, accepting_validator());
            expect_true(!same.ok && has(same.reasons, "already_active"), "the level equal to the current value is a no-op, not a proposal");
            auto off = validate_decision_candidate(level("set_test_pixfrac__0p7", 0.75), good, cfg_px, pol, gc, accepting_validator());
            expect_true(!off.ok && has(off.reasons, "value_not_allowlisted") && off.updates.empty(), "an off-grid value under a level id is refused, nothing salvaged");
            auto other = validate_decision_candidate(level("set_test_pixfrac__0p75", 0.75), good, cfg_px, pol, gc, accepting_validator());
            expect_true(!other.ok && has(other.reasons, "unknown_candidate"), "a level that is not in the grid is an unknown candidate");
            auto strict = validate_decision_candidate(level("set_test_pixfrac__0p7", 0.7), good, cfg_px, DecisionPolicy{}, gc, accepting_validator());
            expect_true(!strict.ok && has(strict.reasons, "experimental_not_enabled"), "grid candidates obey the experimental switch");

            // ---- the confirmed reconstruction candidates (real catalog): only from the control level the evidence was measured against ----
            auto rc_cfg = [&](double pix, double lo, double hi) {
                return json{{"reconstruction", {{"drizzle", {{"pixfrac", pix}}}, {"clipping", {{"clip_sigma_low", lo}, {"clip_sigma_high", hi}}}}}};
            };
            auto pix_c = [&](double v) {
                return json{{"candidate_id", "set_pixfrac__1"}, {"candidate_version", 1},
                            {"updates", json::array({{{"path", "reconstruction.drizzle.pixfrac"}, {"value", v}}})}};
            };
            auto clip_c = [&](double lo, double hi) {
                return json{{"candidate_id", "set_clip_sigmas"}, {"candidate_version", 1},
                            {"updates", json::array({{{"path", "reconstruction.clipping.clip_sigma_low"}, {"value", lo}},
                                                     {{"path", "reconstruction.clipping.clip_sigma_high"}, {"value", hi}}})}};
            };
            auto p1 = validate_decision_candidate(pix_c(1.0), good, rc_cfg(0.8, 4.0, 4.0), pol, catalog, accepting_validator());
            expect_true(p1.ok && p1.requires_review && p1.experimental, "pixfrac 0.8 -> 1.0 is offered from the control level, reviewed and experimental");
            auto p2 = validate_decision_candidate(pix_c(1.0), good, rc_cfg(0.9, 4.0, 4.0), pol, catalog, accepting_validator());
            expect_true(!p2.ok && has(p2.reasons, "precondition_failed:config_equals:reconstruction.drizzle.pixfrac=0.8"), "another current pixfrac is not the measured control: not offered");
            auto p3 = validate_decision_candidate(pix_c(1.0), good, rc_cfg(1.0, 4.0, 4.0), pol, catalog, accepting_validator());
            expect_true(!p3.ok, "pixfrac already 1.0 is not a proposal");
            auto p4 = validate_decision_candidate(pix_c(0.6), good, rc_cfg(0.8, 4.0, 4.0), pol, catalog, accepting_validator());
            expect_true(!p4.ok && has(p4.reasons, "value_not_allowlisted"), "0.6 (coverage-gate failure) is not on the grid");
            auto c1 = validate_decision_candidate(clip_c(5.0, 5.0), good, rc_cfg(0.8, 4.0, 4.0), pol, catalog, accepting_validator());
            expect_true(c1.ok && c1.updates.size() == 2, "clipping 4/4 -> 5/5 is one atomic two-path proposal");
            auto c2 = validate_decision_candidate(clip_c(5.0, 5.0), good, rc_cfg(0.8, 2.0, 4.0), pol, catalog, accepting_validator());
            expect_true(!c2.ok && has(c2.reasons, "precondition_failed:config_equals:reconstruction.clipping.clip_sigma_low=4.0"),
                        "a 2/4 config is not offered 5/5: that contrast was not pre-registered");
            auto c3 = validate_decision_candidate(clip_c(5.0, 4.0), good, rc_cfg(0.8, 4.0, 4.0), pol, catalog, accepting_validator());
            expect_true(!c3.ok && !c3.updates.size(), "half of the pair is refused as a whole (atomic group)");
            auto c4 = validate_decision_candidate(clip_c(5.0, 5.0), good, json{{"reconstruction", json::object()}}, pol, catalog, accepting_validator());
            expect_true(!c4.ok, "a config that lacks the current value is not offered the change");
            json bad_pre = base_cands; bad_pre["candidates"].back()["preconditions"] = json::array({"config_equals:no_equals_sign"});
            const DecisionCatalog bp = load_decision_catalog(bad_pre, prot);
            auto b1 = validate_decision_candidate(clip_c(5.0, 5.0), good, rc_cfg(0.8, 4.0, 4.0), pol, bp, accepting_validator());
            expect_true(!b1.ok && has(b1.reasons, "unknown_precondition:config_equals:no_equals_sign"), "a malformed config_equals precondition fails closed");
        }

        // ---- object class: a user statement that candidates may require ----
        {
            const auto dir = repo / "web_backend_cpp/config/pi_decisions";
            json c = json::parse(slurp_file(dir / "candidates_v1.json"));
            c["candidates"].push_back({{"candidate_id", "diffuse_only"}, {"candidate_version", 1}, {"group", "denoise"},
                                       {"preconditions", json::array()}, {"required_evidence", json::array({"object_class"})},
                                       {"object_classes", json::array({"diffuse"})}, {"provider_description", "Enable luma denoise for a diffuse target."},
                                       {"updates", json::array({{{"path", "luma_denoise.enabled"}, {"value", true}}})},
                                       {"requires_review", true}, {"applicability", "experimental_only"}});
            const DecisionCatalog oc = load_decision_catalog(c, json::parse(slurp_file(dir / "protected_paths_v1.json")));
            const json cfg_dn = {{"luma_denoise", {{"enabled", false}}}};
            auto state_with = [&](const json& ctx) {
                auto in = inputs_for(spread_frames(10));
                in.session_context = ctx;
                return build_pre_run_decision_state(in);
            };
            const json cand = {{"candidate_id", "diffuse_only"}, {"candidate_version", 1},
                               {"updates", json::array({{{"path", "luma_denoise.enabled"}, {"value", true}}})}};
            auto run = [&](const json& ctx) { return validate_decision_candidate(cand, state_with(ctx).state, cfg_dn, pol, oc, accepting_validator()); };
            const auto none = run(json::object());
            expect_true(!none.ok && has(none.reasons, "evidence_unavailable:object_class") && !none.evidence_ok, "no object class stated -> abstain");
            const auto wrong = run({{"object_class", {{"value", "compact"}, {"source", "user"}}}});
            expect_true(!wrong.ok && has(wrong.reasons, "evidence_below_threshold:object_class"), "a class the candidate does not list -> not applicable");
            const auto right = run({{"object_class", {{"value", "diffuse"}, {"source", "user"}}}});
            expect_true(right.ok && right.rationale["params"]["object_class"] == "diffuse", "listed class -> applicable, class reported");
            const auto bad = run({{"object_class", {{"value", "nebula, very big"}, {"source", "user"}}}});
            expect_true(!bad.ok && has(bad.reasons, "evidence_unavailable:object_class"), "a free-text class is treated as not stated");
            const auto not_user = run({{"object_class", {{"value", "diffuse"}, {"source", "model"}}}});
            expect_true(!not_user.ok && has(not_user.reasons, "evidence_unavailable:object_class"), "only source=user counts");
            expect_true(state_with({{"object_class", {{"value", "diffuse"}, {"source", "user"}}}}).state_hash !=
                            state_with(json::object()).state_hash, "the class is part of the state hash (proposals go stale when it changes)");
        }

        // ---- table-driven camera match (set_sensor_profile_dwarf_ii) ----
        {
            const json cfg_rec = {{"hypermetric_stretch", {{"sensor_profile", "rec709"}, {"fallback_profile", "rec709"}}}};
            const json cand = {{"candidate_id", "set_sensor_profile_dwarf_ii"}, {"candidate_version", 1},
                               {"updates", json::array({{{"path", "hypermetric_stretch.sensor_profile"}, {"value", "Sony IMX415 (DWARF II)"}},
                                                        {{"path", "hypermetric_stretch.fallback_profile"}, {"value", "Sony IMX415 (DWARF II)"}}})}};
            auto with_camera = [&](const json& camera) {
                json fr = spread_frames(10);
                for (auto& f : fr) { if (camera.is_null()) f["header"].erase("camera"); else f["header"]["camera"] = camera; }
                return build_pre_run_decision_state(inputs_for(fr)).state;
            };
            auto check = [&](const json& camera, const json& cfg_in = json()) {
                return validate_decision_candidate(cand, with_camera(camera), cfg_in.is_null() ? cfg_rec : cfg_in, pol, catalog, accepting_validator());
            };
            for (const char* spelling : {"DWARFII", "DWARF II", "dwarf-ii", " Dwarf_II "}) {
                const auto v = check(spelling);
                expect_true(v.ok && v.updates.size() == 2, std::string("both DWARF II spellings and their case/space variants match: ") + spelling);
            }
            expect_true(check("DWARFII").rationale["params"]["camera_match"] == true, "match reported");
            for (const char* other : {"DWARF", "DWARF II PRO", "Camera DWARF II", "DWARFIII", "ZWO ASI2600"}) {
                const auto v = check(other);
                expect_true(!v.ok && has(v.reasons, "evidence_below_threshold:camera_match") && !v.evidence_ok, std::string("no substring or prefix matching: ") + other);
            }
            const auto missing = check(json(nullptr));
            expect_true(!missing.ok && has(missing.reasons, "evidence_unavailable:camera_match"), "no camera in the header -> abstain");
            const auto empty = check("  -- ");
            expect_true(!empty.ok && has(empty.reasons, "evidence_unavailable:camera_match"), "a camera string with no letters/digits -> abstain");
            const json cfg_done = {{"hypermetric_stretch", {{"sensor_profile", "Sony IMX415 (DWARF II)"}, {"fallback_profile", "Sony IMX415 (DWARF II)"}}}};
            const auto noop = check("DWARF II", cfg_done);
            expect_true(!noop.ok && has(noop.reasons, "already_active"), "config already carries the profile -> nothing to propose");
            const json cfg_half = {{"hypermetric_stretch", {{"sensor_profile", "Sony IMX415 (DWARF II)"}, {"fallback_profile", "rec709"}}}};
            expect_true(check("DWARF II", cfg_half).ok, "a half-set group is still proposed as a whole");
            json fr_mixed = spread_frames(6, "L"); json fr_b = spread_frames(6, "R");
            for (auto& f : fr_mixed) f["header"]["camera"] = "DWARF II";
            for (auto& f : fr_b) { f["header"]["camera"] = "DWARF II"; f["file_name"] = "r_" + f["file_name"].get<std::string>(); fr_mixed.push_back(f); }
            const auto mixed_v = validate_decision_candidate(cand, build_pre_run_decision_state(inputs_for(fr_mixed)).state, cfg_rec, pol, catalog, accepting_validator());
            expect_true(!mixed_v.ok && has(mixed_v.reasons, "mixed_groups_no_single_evidence"), "several acquisition groups -> no single evidence");
            json locked_state = with_camera("DWARF II"); locked_state["locked_paths"] = json::array({"hypermetric_stretch.fallback_profile"});
            const auto locked = validate_decision_candidate(cand, locked_state, cfg_rec, pol, catalog, accepting_validator());
            expect_true(!locked.ok && has(locked.reasons, "path_locked"), "a locked member blocks the whole group");

            // the alias table itself is validated at load
            const auto dir = repo / "web_backend_cpp/config/pi_decisions";
            const json base_c = json::parse(slurp_file(dir / "candidates_v1.json"));
            const json prot_c = json::parse(slurp_file(dir / "protected_paths_v1.json"));
            auto throws_c = [&](const json& c) { try { load_decision_catalog(c, prot_c); } catch (const std::invalid_argument&) { return true; } return false; };
            auto find_idx = [&](const json& c) { for (size_t i = 0; i < c["candidates"].size(); ++i) if (c["candidates"][i]["candidate_id"] == "set_sensor_profile_dwarf_ii") return i; return size_t(0); };
            json ambiguous = base_c; json twin = ambiguous["candidates"][find_idx(ambiguous)];
            twin["candidate_id"] = "set_sensor_profile_other"; twin["camera_aliases"] = json::array({"dwarf ii"});
            ambiguous["candidates"].push_back(twin);
            expect_true(throws_c(ambiguous), "one camera selecting two candidates for the same parameter refused at load");
            json empty_alias = base_c; empty_alias["candidates"][find_idx(empty_alias)]["camera_aliases"] = json::array({"--"});
            expect_true(throws_c(empty_alias), "alias without letters/digits refused");
            auto find_ci = [&](const json& c, const std::string& id) { for (size_t i = 0; i < c["candidates"].size(); ++i) if (c["candidates"][i]["candidate_id"] == id) return i; return size_t(0); };
            for (const char* bad_text : {"", "/etc/passwd style", "see https://example.org", "contains api_key here", "non-ascii \xC3\xA4", "sk-or-v1-abc"}) {
                json bt = base_c; bt["candidates"][find_ci(bt, "enable_adaptive_weights")]["provider_description"] = bad_text;
                expect_true(throws_c(bt), std::string("provider_description refused: ") + bad_text);
            }
            json too_long = base_c; too_long["candidates"][find_ci(too_long, "enable_adaptive_weights")]["provider_description"] = std::string(241, 'a');
            expect_true(throws_c(too_long), "provider_description over 240 characters refused");
            json missing_desc = base_c; missing_desc["candidates"][find_ci(missing_desc, "enable_adaptive_weights")].erase("provider_description");
            expect_true(throws_c(missing_desc), "a change candidate without provider_description refused");
            json no_aliases = base_c; no_aliases["candidates"][find_idx(no_aliases)]["camera_aliases"] = json::array();
            expect_true(throws_c(no_aliases), "empty alias list refused");
        }

        // ---- rejected candidates: the catalog as shipped never offers enable_adaptive_weights ----
        {
            const DecisionCatalog shipped = load_real_catalog(repo, false);
            const json cfg_aw = {{"global_metrics", {{"adaptive_weights", false}}}};
            const auto v = validate_decision_candidate(enable_candidate(), good, cfg_aw, frozen_test_policy(true), shipped, accepting_validator());
            expect_true(!v.ok && has(v.reasons, "candidate_rejected") && !v.policy_ok, "a rejected candidate is refused even with experimental on and frozen thresholds");
            expect_true(v.updates.empty() && v.merged_config.is_null(), "nothing is salvaged from a rejected candidate");
            json entry;
            for (const auto& c : shipped.candidates) if (c["candidate_id"] == "enable_adaptive_weights") entry = c;
            expect_true(entry.value("applicability", "") == "rejected" && entry.value("validation_status", "") == "rejected" &&
                            !entry.value("rejected_reason", "").empty() && !entry.value("rejected_evidence", "").empty(),
                        "the shipped entry documents why it is rejected and where the evidence is");
            const auto dir = repo / "web_backend_cpp/config/pi_decisions";
            json c = json::parse(slurp_file(dir / "candidates_v1.json"));
            const json prot_r = json::parse(slurp_file(dir / "protected_paths_v1.json"));
            auto throws_r = [&](const json& cc) { try { load_decision_catalog(cc, prot_r); } catch (const std::invalid_argument&) { return true; } return false; };
            auto idx = [&](const json& cc) { for (size_t i = 0; i < cc["candidates"].size(); ++i) if (cc["candidates"][i]["candidate_id"] == "enable_adaptive_weights") return i; return size_t(0); };
            json no_reason = c; no_reason["candidates"][idx(no_reason)].erase("rejected_reason");
            expect_true(throws_r(no_reason), "a rejected candidate without a reason is refused at load");
            json unknown = c; unknown["candidates"][idx(unknown)]["applicability"] = "sometimes";
            expect_true(throws_r(unknown), "an unknown applicability value is refused at load");
            expect_true(!has(validate_decision_candidate(enable_candidate(), good, cfg_aw, frozen_test_policy(true), catalog, accepting_validator()).reasons, "candidate_rejected"),
                        "the test helper reopens it, so the rest of the suite still exercises the candidate logic");
        }

        expect_true(path_is_protected(catalog, "reconstruction.coverage_gate.min_frames"), "coverage_gate protected");
        expect_true(!path_is_protected(catalog, "reconstruction.drizzle.min_clip_contributors"), "min_clip_contributors released for candidates");
        expect_true(path_is_protected(catalog, "pcc.k_max") && path_is_protected(catalog, "reconstruction.multiband_validation.fwhm_ratio_max"), "acceptance gates stay hard-protected");
        expect_true(path_is_protected(catalog, "calibration.dark_master") && path_is_protected(catalog, "output.crop_to_nonzero_bbox") && path_is_protected(catalog, "reconstruction.common_overlap_required_fraction"), "user-domain paths stay protected");
        expect_true(!path_is_protected(catalog, "reconstruction.clipping.shared_frame_rejection") && !path_is_protected(catalog, "reconstruction.clipping.bimodal_veto") && !path_is_protected(catalog, "calibration.dark_match_temp_tolerance_c") && !path_is_protected(catalog, "runtime_limits.memory_budget"), "released tuning knobs are not protected");
        expect_true(path_is_protected(catalog, "runtime_limits.hard_abort_hours"), "hard_abort_hours stays a user-domain value");
        expect_true(!path_is_protected(catalog, "reconstruction.drizzle.pixfrac"), "pixfrac not protected");
        expect_true(!path_is_protected(catalog, "bge.method"), "bge.method released for candidates");
        expect_true(!path_is_protected(catalog, "bge.methodology") && !path_is_protected(catalog, "calibration.dark_master_extra"), "prefix boundary respected");
        expect_true(path_is_protected(catalog, "method") && !path_is_protected(catalog, "methodology"), "method selector protected, not its lookalike");
        expect_true(path_is_locked({"global_metrics"}, "global_metrics.adaptive_weights"), "parent lock covers child");
        expect_true(path_is_locked({"global_metrics.adaptive_weights"}, "global_metrics"), "writing a parent would overwrite a locked leaf");
        expect_true(!path_is_locked({"global_metrics.adaptive"}, "global_metrics.adaptive_weights"), "lock prefix boundary");
        expect_true(json_values_equal(1, 1.0) && !json_values_equal(1, true), "1 == 1.0 but 1 != true");

        // ---- happy path ----
        {
            const auto v = validate_decision_candidate(enable_candidate(), good, cfg, pol, catalog, accepting_validator());
            expect_true(v.ok && v.reasons.empty(), "valid candidate accepted");
            expect_true(v.experimental && v.requires_review, "experimental candidate requires review");
            expect_equal(static_cast<long>(v.updates.size()), 1L, "one update");
            expect_true(v.updates[0]["old_value"] == false && v.updates[0]["value"] == true, "old/new value");
            expect_equal(v.updates[0]["group_id"].get<std::string>(), "global_weighting", "group id from catalog");
            expect_true(config_get(v.merged_config, "global_metrics.adaptive_weights") == true, "merged config carries the value");
            expect_true(config_get(cfg, "global_metrics.adaptive_weights") == false, "input config untouched");
            expect_true(v.evidence_refs.size() >= 3, "evidence refs recorded");
            expect_true(v.rationale["params"].contains("metric_agreement") && v.rationale["params"].contains("measurement_coverage"), "rationale carries measured numbers");
            expect_true(v.schema_ok && v.policy_ok && v.evidence_ok && v.config_ok, "all check families passed");
        }

        // ---- rejections: every one drops the WHOLE group, nothing salvaged ----
        auto expect_rejected = [&](const json& cand, const json& state, const json& config, const DecisionPolicy& p,
                                   const ConfigValidator& val, const std::string& reason, const std::string& msg) {
            const auto v = validate_decision_candidate(cand, state, config, p, catalog, val);
            expect_true(!v.ok, msg + ": rejected");
            expect_true(has(v.reasons, reason), msg + ": reason " + reason);
            expect_true(v.updates.empty() && v.merged_config.is_null(), msg + ": nothing salvaged");
            return v;
        };
        {
            json locked_state = good; locked_state["locked_paths"] = json::array({"global_metrics.adaptive_weights"});
            auto v = expect_rejected(enable_candidate(), locked_state, cfg, pol, accepting_validator(), "path_locked", "lock");
            expect_true(!v.policy_ok, "lock -> policy_ok false");
            json inj = enable_candidate();
            inj["updates"].push_back({{"path", "reconstruction.coverage_gate.min_frames"}, {"value", 1}});
            expect_rejected(inj, good, cfg, pol, accepting_validator(), "path_not_allowlisted", "injected protected path");
            json newpath = enable_candidate();
            newpath["updates"].push_back({{"path", "bge.method"}, {"value", "none"}});
            expect_rejected(newpath, good, cfg, pol, accepting_validator(), "path_not_allowlisted", "injected new path");
            json badval = enable_candidate(); badval["updates"][0]["value"] = false;
            expect_rejected(badval, good, cfg, pol, accepting_validator(), "value_not_allowlisted", "changed value");
            json partial = enable_candidate(); partial["updates"] = json::array();
            expect_rejected(partial, good, cfg, pol, accepting_validator(), "incomplete_group", "partial group");
            json oldver = enable_candidate(); oldver["candidate_version"] = 7;
            expect_rejected(oldver, good, cfg, pol, accepting_validator(), "candidate_version_mismatch", "version");
            json unk = enable_candidate(); unk["candidate_id"] = "disable_bge";
            expect_rejected(unk, good, cfg, pol, accepting_validator(), "unknown_candidate", "unknown id");
            expect_rejected(json(nullptr), good, cfg, pol, accepting_validator(), "malformed_candidate", "non-object candidate");
            expect_rejected(enable_candidate(), good, cfg, frozen_test_policy(false), accepting_validator(), "experimental_not_enabled", "experimental off");
            expect_rejected(enable_candidate(), good, cfg, DecisionPolicy{}, accepting_validator(), "policy_thresholds_not_frozen", "thresholds not frozen");
            DecisionPolicy strict = pol; strict.min_metric_agreement = 0.99;
            v = expect_rejected(enable_candidate(), good, cfg, strict, accepting_validator(), "evidence_below_threshold:metric_agreement", "spread below threshold");
            expect_true(!v.evidence_ok && v.policy_ok, "evidence family isolated");
            json active = {{"global_metrics", {{"adaptive_weights", true}}}};
            expect_rejected(enable_candidate(), good, active, pol, accepting_validator(), "already_active", "already active");
            expect_rejected(enable_candidate(), good, cfg, pol,
                            [](const json& m) { return ConfigCheck{config_get(m, "global_metrics.adaptive_weights") != true, "no"}; },
                            "config_invalid", "config validator vetoes the merged config");
            expect_rejected(enable_candidate(), good, cfg, pol, ConfigValidator{}, "config_validator_missing", "validator missing fails closed");
            json stale_old = enable_candidate(); stale_old["updates"][0]["old_value"] = true;
            expect_rejected(stale_old, good, cfg, pol, accepting_validator(), "old_value_mismatch", "old_value");
            const auto empty_state = build_pre_run_decision_state(PreRunDecisionInputs{}).state;
            expect_rejected(enable_candidate(), empty_state, cfg, pol, accepting_validator(), "blocked:no_readable_frames", "blocking finding");
            const auto mixed = build_pre_run_decision_state(inputs_for([] { json a = spread_frames(6, "L"), b = spread_frames(6, "R");
                for (auto& f : b) f["file_name"] = "r_" + f["file_name"].get<std::string>(); for (auto& f : b) a.push_back(f); return a; }())).state;
            expect_rejected(enable_candidate(), mixed, cfg, pol, accepting_validator(), "mixed_groups_no_single_evidence", "mixed groups");
            const auto few = build_pre_run_decision_state(inputs_for(spread_frames(3))).state;  // 3 valid < floor -> spread not applicable
            expect_rejected(enable_candidate(), few, cfg, pol, accepting_validator(), "evidence_unavailable:metric_agreement", "spread not computable");
            // sorted, unique reasons regardless of discovery order
            const auto multi = validate_decision_candidate(enable_candidate(), locked_state, active, frozen_test_policy(false), catalog, accepting_validator());
            expect_true(std::is_sorted(multi.reasons.begin(), multi.reasons.end()) &&
                            std::adjacent_find(multi.reasons.begin(), multi.reasons.end()) == multi.reasons.end(), "reasons sorted+unique");
            expect_true(has(multi.reasons, "path_locked") && has(multi.reasons, "already_active") && has(multi.reasons, "experimental_not_enabled"), "all problems reported, no short-circuit");
        }

        // ---- resolve_decision ----
        ResolveContext ctx;
        ctx.proposal_id = "p1"; ctx.created_at = "2026-09-23T10:00:00Z"; ctx.state_hash = good_state_r.state_hash;
        ctx.question_set_version = "q1"; ctx.model_requested = "typesafe/jev-1.13";
        auto response_for = [](const std::string& id, double p = 0.9) {
            return json{{"request_id", "r"}, {"state_hash", "h"}, {"status", "ok"}, {"model_requested", "typesafe/jev-1.13"},
                        {"selection", {{"candidate_id", id}, {"probabilities", {{id, p}}}}}};
        };
        auto resolve = [&](const json& resp, const DecisionPolicy& p = frozen_test_policy(), const json& config = json{{"global_metrics", {{"adaptive_weights", false}}}}) {
            const json r = resolve_decision(resp, good, config, p, catalog, accepting_validator(), ctx);
            const std::string err = schema_check(proposal_schema, r, proposal_schema, "proposal");
            expect_true(err.empty(), "proposal schema-valid: " + err);
            return r;
        };
        {
            const json ok = resolve(response_for("enable_adaptive_weights", 0.01));  // low probability is never a gate
            expect_equal(ok["status"].get<std::string>(), "validated", "validated");
            expect_equal(static_cast<long>(ok["updates"].size()), 1L, "updates carried");
            expect_true(ok["review_required"] == true, "review required");
            expect_true(ok["validation"]["config_ok"] == true, "validation flags");
            expect_equal(resolve(response_for("keep_current"))["status"].get<std::string>(), "no_change", "keep_current");
            expect_equal(resolve(response_for("insufficient_evidence"))["status"].get<std::string>(), "abstain", "abstain");
            expect_true(resolve(response_for("keep_current"))["updates"].empty(), "no_change has no updates");
            const json unk = resolve(response_for("disable_bge"));
            expect_true(unk["status"] == "rejected" && has(unk["reason_codes"].get<std::vector<std::string>>(), "unknown_candidate") && unk["updates"].empty(), "model cannot name a new candidate");
            const json na = resolve(response_for("enable_adaptive_weights"), DecisionPolicy{});
            expect_true(na["status"] == "rejected" && na["updates"].empty(), "inapplicable candidate rejected");
            expect_true(has(na["reason_codes"].get<std::vector<std::string>>(), "candidate_not_applicable"), "reason candidate_not_applicable");
            expect_true(na["validation"]["evidence_ok"] == false, "validation flags show which family failed");
            json unavailable = {{"status", "unavailable"}, {"error_code", "provider_down"}};
            expect_equal(resolve(unavailable)["status"].get<std::string>(), "unavailable", "provider outage");
            json bad = {{"status", "invalid_response"}, {"error_code", "x"}};
            const json br = resolve(bad);
            expect_true(br["status"] == "rejected" && has(br["reason_codes"].get<std::vector<std::string>>(), "provider_invalid_response"), "invalid provider response");
            expect_equal(resolve(json::object())["status"].get<std::string>(), "rejected", "garbage response");
            json no_sel = {{"status", "ok"}};
            expect_equal(resolve(no_sel)["status"].get<std::string>(), "rejected", "ok without selection");
            const json again = resolve(response_for("enable_adaptive_weights", 0.01));
            expect_true(again == ok, "resolve is deterministic");
        }

        // ---- staleness ----
        {
            const json prop = resolve(response_for("enable_adaptive_weights"));
            expect_true(stale_reasons(prop, good, good_state_r.state_hash, pol, catalog).empty(), "fresh proposal");
            auto changed_in = inputs_for(spread_frames(10), true);
            const auto changed = build_pre_run_decision_state(changed_in);
            const auto r = stale_reasons(prop, changed.state, changed.state_hash, pol, catalog);
            expect_true(has(r, "stale:config_hash") && has(r, "stale:state_hash"), "config change makes the proposal stale");
            DecisionPolicy p2 = pol; p2.version = "pi.decision-policy.v2";
            expect_true(has(stale_reasons(prop, good, good_state_r.state_hash, p2, catalog), "stale:policy_version"), "policy version bump");
            DecisionCatalog c2 = catalog; c2.candidates[2]["candidate_version"] = 3;
            expect_true(has(stale_reasons(prop, good, good_state_r.state_hash, pol, c2), "stale:candidate_version"), "candidate version bump");
            auto locked_in = inputs_for(spread_frames(10)); locked_in.locked_paths = {"x.y"};
            const auto locked = build_pre_run_decision_state(locked_in);
            expect_true(has(stale_reasons(prop, locked.state, locked.state_hash, pol, catalog), "stale:locks_hash"), "lock change makes it stale");
        }

        std::puts("pi_decision_policy: all checks passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
