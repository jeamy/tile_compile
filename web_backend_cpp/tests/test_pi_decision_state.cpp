#include "services/pi/pi_decision_state.hpp"

#include "backend_test_harness.hpp"
#include "pi_json_schema_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>

using nlohmann::json;
using namespace tile_compile::pi;

namespace {

json frame(int idx, bool ok, const char* filter, double exposure, double bg, double noise, double fwhm, double round_,
           int stars) {
    json f = {{"frame_index", idx}, {"file_name", "f" + std::to_string(idx) + ".fits"}, {"ok", ok},
              {"header", {{"exposure_seconds", exposure}, {"gain", 100}, {"camera", "SyntheticCam"},
                          {"target", "SyntheticTarget"}}}};
    if (filter) f["header"]["filter"] = filter;
    if (ok) {
        f["background"] = bg; f["noise"] = noise; f["gradient_energy"] = 0.01; f["sky_gradient"] = 0.03;
        f["fwhm"] = fwhm; f["roundness"] = round_; f["star_count"] = stars;
    } else {
        f["error"] = "read_error";
    }
    return f;
}

PreRunDecisionInputs base_inputs(json frames, const std::string& color = "OSC") {
    PreRunDecisionInputs in;
    in.scan = {{"frames_detected", static_cast<int>(frames.size())}, {"color_mode", color},
               {"bayer_pattern", color == "OSC" ? json("GRBG") : json(nullptr)}, {"errors", json::array()},
               {"warnings", json::array()}};
    in.metrics = {{"frames", frames}};
    in.base_config = {{"global_metrics", {{"adaptive_weights", false}}}, {"runs_dir", "/home/x/runs"}};
    in.scan_id = "scan-1";
    in.software_version = "test";
    return in;
}

const json* find_group(const json& state, const std::string& filter) {
    for (const auto& g : state["groups"])
        if ((filter.empty() && g["filter"].is_null()) || (g["filter"].is_string() && g["filter"] == filter)) return &g;
    return nullptr;
}

bool has_code(const json& list, const std::string& code) {
    for (const auto& f : list) {
        if (f.is_string() ? f == code : f.value("code", "") == code) return true;
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path repo = argc > 4 ? argv[4] : "..";
        const json schema = json::parse(slurp_file(repo / "web_backend_cpp/config/pi_decisions/schemas/pi.decision-state.v1.schema.json"));

        // --- axis_asymmetry: symmetric, roundness > 1 allowed, never a fabricated 0 ---
        double a = -1, b = -1, c = -1;
        expect_true(axis_asymmetry(0.5, a) && axis_asymmetry(2.0, b) && axis_asymmetry(1.0, c), "asymmetry for 0.5/1/2");
        expect_equal(a, std::log(2.0), "0.5 -> ln2");
        expect_equal(b, std::log(2.0), "2 -> ln2 (symmetric)");
        expect_equal(c, 0.0, "1 -> 0");
        double untouched = 42.0;
        expect_true(!axis_asymmetry(0.0, untouched) && !axis_asymmetry(-1.0, untouched) &&
                        !axis_asymmetry(std::numeric_limits<double>::quiet_NaN(), untouched),
                    "non-positive / NaN roundness rejected");
        expect_equal(untouched, 42.0, "rejected input leaves out untouched");

        // --- canonical dump: key order, 3 vs 3.0, NaN throws ---
        expect_equal(canonical_json_dump(json::parse(R"({"b":3,"a":[1,2]})")),
                     canonical_json_dump(json::parse(R"({"a":[1,2],"b":3.0})")), "canonical: order and 3 vs 3.0");
        bool threw = false;
        try { canonical_json_dump(json{{"x", std::numeric_limits<double>::infinity()}}); } catch (const std::invalid_argument&) { threw = true; }
        expect_true(threw, "canonical dump rejects Infinity");

        // --- synthetic fixture: full case, schema-valid ---
        {
            const json fx = json::parse(slurp_file(repo / "web_backend_cpp/config/pi_decisions/fixtures/scan_metrics_synthetic.json"));
            PreRunDecisionInputs in = base_inputs(fx["frames"]);
            in.metrics = fx;
            in.scan["frames_detected"] = 5;
            in.locked_paths = {"bge.method"};
            const auto r = build_pre_run_decision_state(in);
            const std::string err = pi_test::schema_check(schema, r.state, schema, "state");
            expect_true(err.empty(), "state satisfies committed schema: " + err);
            expect_equal(static_cast<long>(r.state["coverage"]["read_ok"].get<int>()), 4L, "read_ok");
            // frame 3 (read error) has no header -> its own unknown bucket, never merged with known
            expect_equal(static_cast<long>(r.state["groups"].size()), 2L, "known + unknown group");
            const json* known = find_group(r.state, "L");
            expect_true(known != nullptr, "known L group exists");
            expect_equal(static_cast<long>((*known)["frame_count"].get<int>()), 4L, "known group frames");
            const json& fwhm = (*known)["metrics"]["fwhm"];
            expect_equal(static_cast<long>(fwhm["median"]["valid_count"].get<int>()), 3L, "fwhm=-1 frame excluded");
            expect_equal(fwhm["median"]["value"].get<double>(), 2.95, "fwhm median over valid frames only");
            expect_equal(static_cast<long>((*known)["metric_coverage"]["fwhm"]["invalid"].get<int>()), 1L, "cause kept: invalid");
            // 4 valid frames < default floor 5 -> spread not applicable, not a number
            const json& bg = (*known)["metrics"]["background"];
            expect_equal(bg["mad"]["status"].get<std::string>(), "not_applicable", "spread gated by sample size");
            expect_true(bg["mad"]["value"].is_null(), "no value for gated spread");
            expect_equal(bg["mad"]["reason"].get<std::string>(), "insufficient_sample", "reason kept");
            // gain and unknown-group handling
            const json* unknown = find_group(r.state, "");
            expect_true(unknown != nullptr && (*unknown)["frames_read_failed"].get<int>() == 1, "unknown bucket holds the unreadable frame");
            expect_equal(static_cast<long>((*unknown)["metrics"]["fwhm"]["median"]["valid_count"].get<int>()), 0L, "no values for unread frame");
            expect_equal((*unknown)["metrics"]["fwhm"]["median"]["status"].get<std::string>(), "missing", "unread -> missing, not 0");
            // lower floor -> spread computed
            in.options.min_valid_for_spread = 2;
            const auto r2 = build_pre_run_decision_state(in);
            expect_equal((*find_group(r2.state, "L"))["metrics"]["background"]["mad"]["status"].get<std::string>(), "valid", "spread valid at lower floor");
            expect_true(r2.state_hash != r.state_hash, "policy change changes the state hash");
        }

        // --- metric agreement: Spearman rho between background/noise/fwhm per group ---
        {
            auto agree = [&](json frames, int floor_n = 5) {
                PreRunDecisionInputs in = base_inputs(frames);
                in.options.min_valid_for_spread = floor_n;
                return build_pre_run_decision_state(in);
            };
            json fr = json::array();
            for (int i = 0; i < 6; ++i) fr.push_back(frame(i, true, "L", 60, 10.0 + i, 1.0 + i, 10.0 - i, 1.0, 100));
            const auto r = agree(fr);
            const std::string err = pi_test::schema_check(schema, r.state, schema, "state");
            expect_true(err.empty(), "state with metric_agreement satisfies schema: " + err);
            const json& ma = r.state["groups"][0]["metric_agreement"];
            expect_equal(ma["background~noise"]["value"].get<double>(), 1.0, "perfectly concordant metrics -> rho 1");
            expect_equal(ma["background~fwhm"]["value"].get<double>(), -1.0, "perfectly discordant -> rho -1");
            expect_equal(ma["median"]["value"].get<double>(), -1.0, "median of the three pairs");
            expect_equal(static_cast<long>(ma["median"]["valid_count"].get<int>()), 6L, "valid_count = frames valid in all three");
            expect_true(r.provider_projection["groups"][0].contains("metric_agreement") &&
                            r.provider_projection["groups"][0]["metric_agreement"]["median"]["value"].get<double>() == -1.0,
                        "agreement reaches the provider projection (numbers only)");

            // ties share the mean rank: background {1,1,2,2,3,3} vs noise 1..6 -> 16/sqrt(16*17.5)
            json tied = json::array();
            const double bgs[6] = {1, 1, 2, 2, 3, 3};
            for (int i = 0; i < 6; ++i) tied.push_back(frame(i, true, "L", 60, bgs[i], 1.0 + i, 10.0 - i, 1.0, 100));
            expect_equal(agree(tied).state["groups"][0]["metric_agreement"]["background~noise"]["value"].get<double>(),
                         16.0 / std::sqrt(16.0 * 17.5), "tied ranks use average rank", 1e-12);

            // a constant metric has no rho: invalid, never 0 or NaN; the median then is not applicable
            json flat = json::array();
            for (int i = 0; i < 6; ++i) flat.push_back(frame(i, true, "L", 60, 10.0 + i, 5.0, 10.0 - i, 1.0, 100));
            const auto flat_result = agree(flat);
            const json& fm = flat_result.state["groups"][0]["metric_agreement"];
            expect_equal(fm["background~noise"]["status"].get<std::string>(), "invalid", "constant noise -> invalid");
            expect_equal(fm["background~noise"]["reason"].get<std::string>(), "constant_metric", "reason kept");
            expect_equal(fm["median"]["status"].get<std::string>(), "not_applicable", "median needs all three pairs");
            expect_true(fm["median"]["value"].is_null(), "no fabricated median");

            // below the sample floor: not_applicable, not a number
            json few = json::array();
            for (int i = 0; i < 3; ++i) few.push_back(frame(i, true, "L", 60, 10.0 + i, 1.0 + i, 10.0 - i, 1.0, 100));
            expect_equal(agree(few).state["groups"][0]["metric_agreement"]["median"]["status"].get<std::string>(), "not_applicable",
                         "insufficient sample -> not_applicable");

            // a frame without a reliable star fit (fwhm=-1) is excluded from the pairs, not counted as a value
            json partial = fr;
            partial.push_back(frame(6, true, "L", 60, 99.0, 99.0, -1.0, 1.0, 100));
            expect_equal(static_cast<long>(agree(partial).state["groups"][0]["metric_agreement"]["median"]["valid_count"].get<int>()), 6L,
                         "fwhm<=0 frame excluded from agreement");
            // frame order is not semantic
            json rev = json::array();
            for (size_t i = fr.size(); i-- > 0;) rev.push_back(fr[i]);
            expect_equal(agree(rev).state_hash, r.state_hash, "agreement does not depend on frame order");
        }

        // --- object class: closed enum, only from the user ---
        {
            auto in = base_inputs(json::array({frame(0, true, "L", 60, 10, 1, 3, 1.0, 100), frame(1, true, "L", 60, 11, 1, 3, 1.0, 100)}));
            in.session_context = {{"object_class", {{"value", "star_field"}, {"source", "user"}}}};
            const auto ok = build_pre_run_decision_state(in);
            expect_true(ok.state["session_facts"]["user_stated"].contains("object_class"), "valid class kept");
            expect_true(ok.provider_projection["session_facts"]["user_stated"].contains("object_class"), "class reaches the provider projection");
            in.session_context = {{"object_class", {{"value", "galaxy cluster"}, {"source", "user"}}}};
            const auto bad = build_pre_run_decision_state(in);
            expect_true(!bad.state["session_facts"]["user_stated"].contains("object_class") && has_code(bad.findings, "object_class_invalid"), "invalid class dropped with a finding");
            const std::string err = pi_test::schema_check(schema, ok.state, schema, "state");
            expect_true(err.empty(), "state with a class satisfies the schema: " + err);
        }

        // --- empty inputs: blocking, no fabricated groups ---
        {
            PreRunDecisionInputs in;
            const auto r = build_pre_run_decision_state(in);
            expect_true(has_code(r.state["blocking_findings"], "no_readable_frames"), "empty: no_readable_frames");
            expect_true(has_code(r.state["blocking_findings"], "metrics_missing"), "empty: metrics_missing");
            expect_true(has_code(r.state["blocking_findings"], "color_mode_unresolved"), "empty: color mode unresolved");
            expect_true(has_code(r.state["blocking_findings"], "base_config_invalid"), "empty: base config invalid");
            expect_equal(static_cast<long>(r.state["groups"].size()), 0L, "empty: no groups");
            expect_true(pi_test::schema_check(schema, r.state, schema, "state").empty(), "empty state still schema-valid");
        }

        // --- color modes ---
        {
            json fr = json::array({frame(0, true, "L", 60, 500, 8, 3.0, 1.0, 100)});
            expect_true(!has_code(build_pre_run_decision_state(base_inputs(fr, "MONO")).state["blocking_findings"], "bayer_pattern_missing"), "MONO needs no bayer");
            auto unk = base_inputs(fr, "UNKNOWN");
            expect_true(has_code(build_pre_run_decision_state(unk).state["blocking_findings"], "color_mode_unresolved"), "UNKNOWN blocks");
            auto osc = base_inputs(fr, "OSC");
            osc.scan["bayer_pattern"] = nullptr;
            expect_true(has_code(build_pre_run_decision_state(osc).state["blocking_findings"], "bayer_pattern_missing"), "OSC without bayer blocks");
            auto weird = base_inputs(fr, "RGB");
            expect_true(has_code(build_pre_run_decision_state(weird).state["blocking_findings"], "color_mode_unresolved"), "unknown mode string blocks");
            osc.scan["errors"] = json::array({{{"code", "x"}}});
            expect_true(has_code(build_pre_run_decision_state(osc).state["blocking_findings"], "scan_errors_present"), "scan errors block");
        }

        // --- a single missing metric, zero/negative denominator, NaN/Infinity ---
        {
            json fr = json::array();
            for (int i = 0; i < 6; ++i) fr.push_back(frame(i, true, "L", 60, 0.0, 8, 3.0, 1.0, 100));
            fr[2].erase("noise");
            fr[3]["fwhm"] = std::numeric_limits<double>::quiet_NaN();
            fr[4]["star_count"] = "NaN";  // wrong type: present but not a number
            const auto r = build_pre_run_decision_state(base_inputs(fr));
            const json& g = r.state["groups"][0];
            expect_equal(static_cast<long>(g["metric_coverage"]["noise"]["missing"].get<int>()), 1L, "one missing noise");
            expect_equal(static_cast<long>(g["metric_coverage"]["noise"]["invalid"].get<int>()), 0L, "not invalid");
            expect_equal(static_cast<long>(g["metrics"]["noise"]["median"]["valid_count"].get<int>()), 5L, "5 valid noise");
            expect_equal(static_cast<long>(g["metric_coverage"]["fwhm"]["invalid"].get<int>()), 1L, "NaN fwhm invalid");
            expect_equal(static_cast<long>(g["metric_coverage"]["star_count"]["invalid"].get<int>()), 1L, "string star_count invalid");
            expect_true(has_code(r.findings, "non_finite_input"), "NaN reported as finding");
            // background all 0 -> median 0 -> zero denominator, never inf/NaN
            const json& rs = g["metrics"]["background"]["relative_spread"];
            expect_equal(rs["status"].get<std::string>(), "invalid", "zero denominator invalid");
            expect_equal(rs["reason"].get<std::string>(), "zero_denominator", "reason zero_denominator");
            expect_true(rs["value"].is_null(), "no number for zero denominator");
            expect_true(pi_test::schema_check(schema, r.state, schema, "state").empty(), "NaN/missing case schema-valid");
            expect_true(!r.state_hash.empty(), "hash computable despite NaN input");
        }

        // --- mixed filters / exposures are never pooled; unknown never equals known ---
        {
            json fr = json::array();
            for (int i = 0; i < 3; ++i) fr.push_back(frame(i, true, "L", 60, 500, 8, 3.0, 1.0, 100));
            for (int i = 3; i < 6; ++i) fr.push_back(frame(i, true, "R", 60, 700, 8, 3.0, 1.0, 100));
            for (int i = 6; i < 8; ++i) fr.push_back(frame(i, true, nullptr, 60, 900, 8, 3.0, 1.0, 100));
            for (int i = 8; i < 10; ++i) fr.push_back(frame(i, true, "L", 120, 300, 8, 3.0, 1.0, 100));
            const auto r = build_pre_run_decision_state(base_inputs(fr));
            expect_equal(static_cast<long>(r.state["groups"].size()), 4L, "L60 / R60 / unknown-filter / L120");
            expect_true(has_code(r.findings, "mixed_acquisition_groups"), "mixed groups reported");
            expect_equal((*find_group(r.state, "L"))["metrics"]["background"]["median"]["value"].get<double>() > 0 ? 1L : 0L, 1L, "sanity");
            const json* unknown = find_group(r.state, "");
            expect_equal((*unknown)["metrics"]["background"]["median"]["value"].get<double>(), 900.0, "unknown bucket not merged with known");
        }

        // --- opposing elongations: raw median hides them, axis asymmetry does not ---
        {
            json fr = json::array();
            for (int i = 0; i < 6; ++i) fr.push_back(frame(i, true, "L", 60, 500, 8, 3.0, i % 2 ? 2.0 : 0.5, 100));
            const auto r = build_pre_run_decision_state(base_inputs(fr));
            const json& m = r.state["groups"][0]["metrics"];
            expect_equal(m["roundness"]["median"]["value"].get<double>(), 1.25, "raw roundness kept unchanged (median of 0.5,2 mix)");
            expect_equal(m["axis_asymmetry"]["median"]["value"].get<double>(), std::log(2.0), "asymmetry exposes the elongation");
            expect_equal(m["axis_asymmetry"]["mad"]["value"].get<double>(), 0.0, "all frames equally asymmetric");
        }

        // --- stable hashes, and hashes that must change ---
        {
            json fr = json::array();
            for (int i = 0; i < 5; ++i) fr.push_back(frame(i, true, "L", 60, 500 + i, 8, 3.0, 1.0, 100));
            auto in = base_inputs(fr);
            const auto r1 = build_pre_run_decision_state(in);
            expect_equal(build_pre_run_decision_state(in).state_hash, r1.state_hash, "same input -> same hash");
            auto reversed = in;
            std::reverse(reversed.metrics["frames"].begin(), reversed.metrics["frames"].end());
            expect_equal(build_pre_run_decision_state(reversed).state_hash, r1.state_hash, "frame order is not semantic");
            auto floaty = in;
            floaty.metrics["frames"][0]["star_count"] = 100.0;  // 100 vs 100.0
            expect_equal(build_pre_run_decision_state(floaty).state_hash, r1.state_hash, "100 vs 100.0 hash identically");
            auto locked = in; locked.locked_paths = {"global_metrics.adaptive_weights"};
            expect_true(build_pre_run_decision_state(locked).state_hash != r1.state_hash, "locks change the hash");
            auto locks_order = in; locks_order.locked_paths = {"b.x", "a.y", "a.y"};
            auto locks_order2 = in; locks_order2.locked_paths = {"a.y", "b.x"};
            expect_equal(build_pre_run_decision_state(locks_order).state_hash, build_pre_run_decision_state(locks_order2).state_hash, "lock order/dups not semantic");
            auto cfg = in; cfg.base_config["global_metrics"]["adaptive_weights"] = true;
            expect_true(build_pre_run_decision_state(cfg).state["identity"]["config_hash"] != r1.state["identity"]["config_hash"], "config change -> config_hash");
            auto edited = in; edited.metrics["frames"][1]["background"] = 999.0;
            expect_true(build_pre_run_decision_state(edited).state["identity"]["scan_hash"] != r1.state["identity"]["scan_hash"], "metric edit -> scan_hash");
            auto stamped = in; stamped.scan["generated_at"] = "2026-09-23T00:00:00Z";
            expect_equal(build_pre_run_decision_state(stamped).state_hash, r1.state_hash, "volatile timestamp excluded");
            // manifest: content digest vs metadata strength
            auto with_digest = in;
            const std::string h(64, 'a');
            for (int i = 0; i < 5; ++i)
                with_digest.dataset_manifest.push_back({{"id", "/abs/dir/f" + std::to_string(i) + ".fits"}, {"size", 10}, {"mtime", 1}, {"sha256", h}});
            const auto rd = build_pre_run_decision_state(with_digest);
            expect_equal(rd.state["identity"]["identity_strength"].get<std::string>(), "content_digest", "all digests -> content_digest");
            expect_equal(r1.state["identity"]["identity_strength"].get<std::string>(), "metadata", "no manifest -> metadata");
            expect_true(rd.state["identity"]["dataset_fingerprint"] != r1.state["identity"]["dataset_fingerprint"], "strength/contents in fingerprint");
            expect_true(rd.state.dump().find("/abs/dir") == std::string::npos, "manifest ids reduced to basenames");
        }

        // --- provider projection: allowlist only, no secrets/paths/free text ---
        {
            json fr = json::array();
            for (int i = 0; i < 5; ++i) fr.push_back(frame(i, true, "L", 60, 500, 8, 3.0, 1.0, 100));
            fr[4]["header"]["filter"] = "/etc/passwd";  // a header value that looks like a path
            auto in = base_inputs(fr);
            in.session_context = {{"mount_type", {{"value", "EQ"}, {"source", "user"}}},
                                  {"note", {{"value", "/home/lux/secret notes"}, {"source", "user"}}},
                                  {"forged", {{"value", "AZ"}, {"source", "model"}}}};
            in.capabilities = {{"astap_available", true}, {"astap_path", "/usr/bin/astap"}};
            in.base_config["api_key"] = "sk-or-v1-SECRET";
            in.locked_paths = {"global_metrics.adaptive_weights"};
            const auto r = build_pre_run_decision_state(in);
            const std::string proj = r.provider_projection.dump();
            for (const char* banned : {"SyntheticCam", "SyntheticTarget", "scan-1", "/home", "/usr", "/etc", "sk-or-", "runs_dir", "api_key", "dataset_fingerprint"})
                expect_true(proj.find(banned) == std::string::npos, std::string("projection leaks: ") + banned);
            expect_true(r.provider_projection["config_facts"]["global_metrics.adaptive_weights"] == false, "current config value projected");
            expect_true(r.provider_projection["locked"]["global_metrics.adaptive_weights"] == true, "lock state projected");
            expect_true(r.provider_projection["capabilities"].contains("astap_available") && !r.provider_projection["capabilities"].contains("astap_path"), "only boolean facts");
            expect_true(r.provider_projection["session_facts"]["user_stated"].contains("mount_type"), "plain user fact kept, with source");
            expect_true(!r.provider_projection["session_facts"]["user_stated"].contains("note"), "path-like user fact withheld");
            expect_true(!r.provider_projection["session_facts"]["user_stated"].contains("forged"), "non-user-sourced fact never accepted");
            expect_true(has_code(r.findings, "session_context_ignored") && has_code(r.findings, "capability_ignored") &&
                            has_code(r.findings, "filter_not_projected") && has_code(r.findings, "user_fact_not_projected"),
                        "every withheld/ignored item is reported");
            expect_true(r.state["session_facts"]["measured"].is_object(), "measured and user_stated kept apart");
        }

        std::puts("pi_decision_state: all checks passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
