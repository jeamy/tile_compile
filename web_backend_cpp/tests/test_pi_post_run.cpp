#include "pi_decision_test_support.hpp"
#include "services/pi/pi_post_run.hpp"
#include "services/pi/pi_resume_scope.hpp"

#include <cstdio>
#include <fstream>
#include <map>
#include <unistd.h>

using namespace pi_test;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

void write(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

json metric(double v) { return {{"applicable", true}, {"value", v}, {"sample_count", 50}}; }

json candidate_validation(bool numerics_ok = true, bool support_ok = true) {
    return {{"numerics_ok", numerics_ok}, {"support_ok", support_ok}, {"background_rms", metric(1.1)}, {"median_fwhm", metric(5.0)},
            {"elongation", {{"applicable", false}, {"value", nullptr}}}};
}

// A run directory shaped like a finished forward-drizzle run.
fs::path make_run(const fs::path& root, const std::string& name, bool success = true, bool with_fd = true, bool sel_numerics_ok = true) {
    const fs::path d = root / name;
    write(d / "artifacts/run_provenance.json", json{{"execution_scope", "forward_drizzle_m1_m3"}, {"config", {{"sha256", "abc"}}},
                                                   {"build", {{"build_id", "b1"}}}, {"input_manifest", {{"sha256", "def"}}}}.dump());
    if (with_fd)
        write(d / "artifacts/forward_drizzle.json",
              json{{"selected_candidate", "drizzle_raw"}, {"selection_reason", "multiband not promoted -> raw"}, {"commit_complete", true},
                   {"validation", {{"drizzle_raw", candidate_validation(sel_numerics_ok)}, {"drizzle_multiband", candidate_validation()}, {"drizzle_uniform", candidate_validation()}}}}.dump());
    std::string log = json{{"type", "phase_end"}, {"phase_name", "SCAN_INPUT"}}.dump() + "\n" +
                      json{{"type", "phase_end"}, {"phase_name", "FORWARD_DRIZZLE"}}.dump() + "\n" +
                      json{{"type", "run_end"}, {"status", success ? "final_image_ready" : "failed"}, {"success", success},
                           {"final_image_available", success}, {"execution_scope", "forward_drizzle_m1_m3"}}.dump() + "\n" + "{\"type\":\"phase_end\",\"pha";  // truncated last line
    write(d / "logs/run_events.jsonl", log);
    write(d / "outputs/canvas_mask.fits", "x");
    return d;
}

json control_config() {
    return {{"reconstruction", {{"drizzle", {{"pixfrac", 0.8}}}, {"clipping", {{"clip_sigma_low", 4.0}, {"clip_sigma_high", 4.0}}}}}};
}

std::map<std::string, std::string> snapshot(const fs::path& d) {
    std::map<std::string, std::string> m;
    for (const auto& e : fs::recursive_directory_iterator(d))
        if (e.is_regular_file()) {
            std::ifstream in(e.path());
            m[e.path().string()] = std::string(std::istreambuf_iterator<char>(in), {}) + std::to_string(fs::file_size(e.path()));
        }
    return m;
}

bool has_finding(const json& advice, const std::string& code) {
    for (const auto& f : advice["findings"]) if (f["code"] == code) return true;
    return false;
}
std::vector<std::string> ids(const json& advice) {
    std::vector<std::string> v;
    for (const auto& s : advice["suggestions"]) v.push_back(s["candidate_id"]);
    return v;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const fs::path repo = argc > 4 ? argv[4] : "..";
        const DecisionCatalog catalog = load_real_catalog(repo);
        const auto ok = accepting_validator();
        const fs::path root = fs::temp_directory_path() / ("pi_post_run_test_" + std::to_string(::getpid()));
        fs::remove_all(root);
        fs::create_directories(root);

        // ---- the resume scope table: one copy, latest legal phase per section ----
        {
            using tile_compile::pi::min_resume_phase_for_path;
            expect_true(min_resume_phase_for_path("hypermetric_stretch.target_bg") == "HYPERMETRIC_STRETCH", "hms section -> HYPERMETRIC_STRETCH");
            expect_true(min_resume_phase_for_path("pcc.chroma_strength") == "PCC", "pcc -> PCC");
            expect_true(min_resume_phase_for_path("chroma_denoise.strength") == "PCC", "chroma_denoise is legal from PCC on");
            expect_true(min_resume_phase_for_path("bge.method") == "BGE", "bge -> BGE");
            expect_true(min_resume_phase_for_path("luma_denoise.enabled") == "BGE", "luma_denoise is legal from BGE on");
            expect_true(min_resume_phase_for_path("astrometry.enabled") == "ASTROMETRY", "astrometry -> ASTROMETRY");
            expect_true(min_resume_phase_for_path("global_metrics.adaptive_weights") == "GLOBAL_QUALITY", "global_metrics only from GLOBAL_QUALITY");
            expect_true(!min_resume_phase_for_path("reconstruction.drizzle.pixfrac"), "reconstruction has no resume phase: full run");
            expect_true(!min_resume_phase_for_path("no_such_section.x"), "unknown section has no resume phase");
            expect_true(tile_compile::pi::resume_allowed_sections("NOPE") == nullptr, "unknown phase has no scope");
        }

        // ---- state from run artifacts ----
        const fs::path run = make_run(root, "run_ok");
        const json cfg = control_config();
        const json st = tile_compile::pi::build_post_run_state(run, cfg);
        expect_true(st["schema_version"] == "pi.post-run-decision-state.v1", "state schema");
        expect_true(st["run"]["success"] == true && st["run"]["final_image_available"] == true, "terminal event read");
        expect_true(st["run"]["phases_completed"] == json::array({"SCAN_INPUT", "FORWARD_DRIZZLE"}), "phases in log order; the truncated last line is ignored");
        expect_true(st["reconstruction"]["selected_candidate"] == "drizzle_raw", "selected candidate");
        expect_true(st["reconstruction"]["candidates"]["drizzle_raw"]["median_fwhm"]["value"] == 5.0, "candidate metrics carried");
        expect_true(!st["reconstruction"]["candidates"]["drizzle_raw"]["elongation"].contains("value"), "a not-applicable metric has no value (never zero)");
        expect_true(st["artifacts"]["canvas_mask"] == true && st["artifacts"]["normalized_cache"] == false, "artifact presence");
        expect_true(st["missing"].empty(), "nothing missing");
        expect_true(st["state_hash"].get<std::string>().rfind("sha256:", 0) == 0, "state hashed");
        expect_true(st == tile_compile::pi::build_post_run_state(run, cfg), "state is deterministic");
        expect_true(st["not_measured"].size() == 2, "what is not measured is stated");

        // ---- advice: reconstruction suggestions from the checked catalog only, read-only ----
        const auto before = snapshot(run);
        auto a = tile_compile::pi::advise_post_run(st, cfg, frozen_test_policy(), catalog, ok, {});
        expect_true(a["outcome"] == "suggest_reconstruction", "control-level reconstruction config: suggest reconstruction");
        expect_true(ids(a) == std::vector<std::string>({"set_clip_sigmas", "set_pixfrac__1"}), "both confirmed candidates, sorted");
        for (const auto& s : a["suggestions"]) {
            expect_true(s["resume_mode"] == "full_run" && s["min_resume_phase"].is_null(), "reconstruction.* needs a full run, no resume phase");
            expect_true(s["requires_review"] == true && s["experimental"] == true && s["feasibility"] == "not_checked_use_resume_dry_run", "review flags; feasibility not claimed");
        }
        expect_true(a["model_called"] == false && a["basis"] == "run_artifacts_only", "no model, artifacts only");
        expect_true(snapshot(run) == before, "advice does not touch the run directory");
        expect_true(a == tile_compile::pi::advise_post_run(st, cfg, frozen_test_policy(), catalog, ok, {}), "advice is deterministic");

        // ---- no suggestion when nothing applies / not permitted / dismissed ----
        json done = cfg; done["reconstruction"]["drizzle"]["pixfrac"] = 1.0; done["reconstruction"]["clipping"]["clip_sigma_low"] = 5.0; done["reconstruction"]["clipping"]["clip_sigma_high"] = 5.0;
        expect_true(tile_compile::pi::advise_post_run(st, done, frozen_test_policy(), catalog, ok, {})["outcome"] == "no_change", "already at the tested levels: no change");
        json prod = cfg; prod["reconstruction"]["clipping"]["clip_sigma_low"] = 2.0;
        auto p = tile_compile::pi::advise_post_run(st, prod, frozen_test_policy(), catalog, ok, {});
        expect_true(ids(p) == std::vector<std::string>({"set_pixfrac__1"}), "a 2/4 config is not offered the clipping change");
        auto strict = tile_compile::pi::advise_post_run(st, cfg, DecisionPolicy{}, catalog, ok, {});
        expect_true(strict["outcome"] == "no_change" && strict["suggestions"].empty(), "experimental candidates are not offered unless enabled");
        bool says_experimental = false;
        for (const auto& e : strict["excluded"]) for (const auto& r : e["reasons"]) says_experimental = says_experimental || r == "experimental_not_enabled";
        expect_true(says_experimental, "and the reason is visible");
        auto dis = tile_compile::pi::advise_post_run(st, cfg, frozen_test_policy(), catalog, ok, {"set_pixfrac__1"});
        expect_true(ids(dis) == std::vector<std::string>({"set_clip_sigmas"}), "a dismissed candidate is not suggested again");
        bool logged = false;
        for (const auto& e : dis["excluded"]) logged = logged || (e["candidate_id"] == "set_pixfrac__1" && e["reasons"][0] == "previously_dismissed");
        expect_true(logged, "and the dismissal is reported");
        expect_true(tile_compile::pi::advise_post_run(st, json(nullptr), frozen_test_policy(), catalog, ok, {})["suggestions"].empty(), "no run config: no suggestion");

        // ---- missing evidence: diagnose, never a repair ----
        const fs::path nofd = make_run(root, "run_no_fd", true, false);
        const json st2 = tile_compile::pi::build_post_run_state(nofd, cfg);
        auto d1 = tile_compile::pi::advise_post_run(st2, cfg, frozen_test_policy(), catalog, ok, {});
        expect_true(d1["outcome"] == "diagnose" && has_finding(d1, "reconstruction_artifact_missing") && d1["suggestions"].empty(), "missing forward_drizzle.json: diagnose only");
        expect_true(st2["reconstruction"]["available"] == false, "reconstruction marked unavailable");
        const fs::path failed = make_run(root, "run_failed", false);
        auto d2 = tile_compile::pi::advise_post_run(tile_compile::pi::build_post_run_state(failed, cfg), cfg, frozen_test_policy(), catalog, ok, {});
        expect_true(d2["outcome"] == "diagnose" && has_finding(d2, "run_not_successful") && d2["suggestions"].empty(), "a failed run is diagnosed, not tuned");
        const fs::path badsel = make_run(root, "run_badsel", true, true, false);
        auto d3 = tile_compile::pi::advise_post_run(tile_compile::pi::build_post_run_state(badsel, cfg), done, frozen_test_policy(), catalog, ok, {});
        expect_true(d3["outcome"] == "diagnose" && has_finding(d3, "selected_candidate_numerics_or_support_failed"), "a selected candidate that failed numerics is diagnosed");
        const fs::path empty = root / "run_empty";
        fs::create_directories(empty);
        const json st4 = tile_compile::pi::build_post_run_state(empty, cfg);
        expect_true(st4["missing"].size() >= 3, "an empty run lists what is missing");
        expect_true(tile_compile::pi::advise_post_run(st4, cfg, frozen_test_policy(), catalog, ok, {})["outcome"] == "diagnose", "an empty run is diagnosed");
        expect_true(has_finding(a, "selected_candidate_reason") && a["findings"][0]["severity"] == "info", "the gate-decided raw selection is information, not a defect");

        fs::remove_all(root);
        std::puts("pi_post_run: all checks passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
