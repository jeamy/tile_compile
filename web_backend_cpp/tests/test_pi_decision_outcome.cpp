#include "backend_test_harness.hpp"
#include "services/pi/pi_decision_outcome.hpp"
#include "services/pi/pi_decision_state.hpp"
#include "services/pi/pi_scan_manifest.hpp"

#include <cstdio>
#include <fstream>
#include <map>
#include <unistd.h>

using namespace tile_compile::pi;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
void write_text(const fs::path& p, const std::string& s) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << s;
}
json read_json(const fs::path& p) { return json::parse(slurp_file(p)); }

// path -> bytes for every regular file below root (to prove nothing was written/changed).
std::map<std::string, std::string> snapshot(const fs::path& root) {
    std::map<std::string, std::string> m;
    if (!fs::exists(root)) return m;
    for (const auto& e : fs::recursive_directory_iterator(root))
        if (e.is_regular_file()) m[fs::relative(e.path(), root).string()] = slurp_file(e.path());
    return m;
}

json proposal(const std::string& status, const std::string& applied_at, json updates) {
    return {{"proposal_id", "x"}, {"status", status}, {"applied_at", applied_at}, {"updates", updates},
            {"saved_revision_ids", json::array({"cfg_saved"})}};
}
} // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / ("pi_decision_outcome_test_" + std::to_string(::getpid()));
    try {
        fs::remove_all(root);
        const fs::path dec = root / "pi_decisions";
        const fs::path run = root / "runs" / "run1";
        const fs::path memory = root / "pi_memory";
        write_text(memory / "memories_v2.jsonl", "{\"sentinel\":true}\n");
        write_text(run / "artifacts" / "pi_run_provenance.json",
                   json{{"config_revision_id", "cfg_2"}, {"prior_active_config_revision_id", "cfg_saved"},
                        {"jev_proposal_id", "p_ok"},
                        {"started_at", "2026-09-23T11:00:00Z"}}.dump());
        write_text(run / "config.yaml", "global_metrics:\n  adaptive_weights: true\nname: \"1\"\nn: 3\nx: 0.5\n");
        const json upd_true = json::array({{{"path", "global_metrics.adaptive_weights"}, {"value", true}}});
        write_text(dec / "p_ok" / "proposal.json", proposal("applied_to_draft", "2026-09-23T10:00:00Z", upd_true).dump());
        write_text(dec / "p_late" / "proposal.json", proposal("applied_to_draft", "2026-09-23T12:00:00Z", upd_true).dump());
        write_text(dec / "p_validated" / "proposal.json", proposal("validated", "2026-09-23T10:00:00Z", upd_true).dump());
        write_text(dec / "p_absent" / "proposal.json",
                   proposal("applied_to_draft", "2026-09-23T10:00:00Z", json::array({{{"path", "global_metrics.adaptive_weights"}, {"value", false}}})).dump());
        write_text(dec / "p_partial" / "proposal.json",
                   proposal("applied_to_draft", "2026-09-23T10:00:00Z",
                            json::array({{{"path", "global_metrics.adaptive_weights"}, {"value", true}}, {{"path", "n"}, {"value", 99}}})).dump());
        write_text(dec / "p_unsaved" / "proposal.json",
                   json{{"status", "applied_to_draft"}, {"applied_at", "2026-09-23T10:00:00Z"},
                        {"updates", upd_true}}.dump());

        const json applied_config = yaml_text_to_json(slurp_file(run / "config.yaml"));
        json saved_proposal = proposal("applied_to_draft", "2026-09-23T10:00:00Z", upd_true);
        saved_proposal["config_hash_after"] = sha256_prefixed(canonical_json_dump(applied_config));
        write_text(dec / "p_link" / "proposal.json", saved_proposal.dump());
        expect_true(matches_applied_jev_config(dec, "p_link", applied_config), "exact saved config matches proposal");
        json altered_config = applied_config;
        altered_config["n"] = 4;
        expect_true(!matches_applied_jev_config(dec, "p_link", altered_config), "edited config does not match proposal");
        expect_true(!matches_applied_jev_config(dec, "../escape", applied_config), "invalid proposal id refused");
        record_jev_saved_revision(dec, "p_link", "cfg_saved");
        record_jev_saved_revision(dec, "p_link", "cfg_saved");
        expect_equal(static_cast<long>(read_json(dec / "p_link" / "proposal.json")["saved_revision_ids"].size()), 1L,
                     "repeated save link is idempotent");
        const fs::path input = root / "input";
        write_text(input / "a.fits", "frame");
        const json manifest = build_scan_dataset_manifest(input,
            {{"frames", json::array({{{"file_name", "a.fits"}}})}});
        write_text(dec / "p_link" / "source.json",
                   json{{"input_path", input.string()}, {"dataset_manifest", manifest}}.dump());
        expect_true(matches_saved_jev_run(dec, "p_link", input, applied_config), "saved proposal matches run input and values");
        const fs::path staged = root / "staged";
        fs::create_directories(staged);
        fs::create_symlink(input / "a.fits", staged / "a.fits");
        expect_true(matches_saved_jev_run(dec, "p_link", input, staged, applied_config),
                    "queue staging with identical selected FITS metadata matches");
        write_text(staged / "extra.fits", "extra");
        expect_true(!matches_saved_jev_run(dec, "p_link", input, staged, applied_config),
                    "queue staging with an extra FITS file cannot be attributed");
        json changed_run_config = applied_config;
        changed_run_config["global_metrics"]["adaptive_weights"] = false;
        expect_true(!matches_saved_jev_run(dec, "p_link", input, changed_run_config), "changed proposed run value blocks attribution");
        std::ofstream(input / "a.fits", std::ios::app) << "changed";
        expect_true(!matches_saved_jev_run(dec, "p_link", input, applied_config), "changed input metadata blocks attribution");

        // real yaml loader: bool / quoted string / int / double keep their types
        const json loaded = load_run_config_yaml(run);
        expect_true(loaded["global_metrics"]["adaptive_weights"] == true, "yaml bool");
        expect_true(loaded["name"].is_string() && loaded["name"] == "1", "quoted scalar stays a string");
        expect_true(loaded["n"].is_number_integer() && loaded["n"] == 3, "yaml int");
        expect_true(loaded["x"].is_number_float() && loaded["x"] == 0.5, "yaml double");

        const auto run_before = snapshot(run);
        const auto memory_before = snapshot(memory);

        // ---- attribution + recording ----
        const json m1 = record_jev_outcome_if_needed(dec, "run1", run);
        expect_true(m1["terminal"] == true && m1["reason"] == "recorded", "first call records and is terminal");
        expect_true(fs::exists(dec / "p_ok" / "outcome.json"), "outcome written for paths_present");
        const json o = read_json(dec / "p_ok" / "outcome.json");
        expect_equal(o["runs"][0]["attribution"].get<std::string>(), "paths_present", "attribution present");
        expect_true(o["runs"][0]["comparison_kind"] == "unpaired" && o["runs"][0]["quality_delta"].is_null(), "unpaired, no quality claim");
        expect_equal(o["runs"][0]["config_revision_id"].get<std::string>(), "cfg_2", "provenance revision recorded");
        expect_true(!fs::exists(dec / "p_partial" / "outcome.json"), "unlinked proposal ignored even when values overlap");
        expect_true(!fs::exists(dec / "p_absent" / "outcome.json"), "absent values are not recorded as an outcome");
        expect_true(!fs::exists(dec / "p_late" / "outcome.json"), "proposal applied after run start ignored");
        expect_true(!fs::exists(dec / "p_validated" / "outcome.json"), "non-applied proposal ignored");
        expect_true(!fs::exists(dec / "p_unsaved" / "outcome.json"), "draft-only proposal ignored");
        const fs::path wrong_revision_run = root / "runs" / "wrong_revision";
        write_text(wrong_revision_run / "artifacts" / "pi_run_provenance.json",
                   json{{"prior_active_config_revision_id", "unrelated"},
                        {"started_at", "2026-09-23T11:00:00Z"}}.dump());
        write_text(wrong_revision_run / "config.yaml", slurp_file(run / "config.yaml"));
        expect_true(record_jev_outcome_if_needed(dec, "wrong_revision", wrong_revision_run)["reason"] == "no_linked_proposal",
                    "matching values without an explicit proposal link do not create an outcome");
        expect_true(m1["skipped"].empty(), "unlinked proposals are never evaluated");

        // ---- isolation: nothing written into the run, memory store untouched ----
        expect_true(snapshot(run) == run_before, "run directory byte-identical (read-only)");
        expect_true(snapshot(memory) == memory_before, "PiMemoryStore files untouched");
        expect_true(fs::exists(dec / "_run_markers" / "run1.json"), "marker lives under pi_decisions");

        // ---- idempotence ----
        const auto before_second = snapshot(dec);
        const json m2 = record_jev_outcome_if_needed(dec, "run1", run, [](const fs::path&) -> json { throw std::runtime_error("must not load again"); });
        expect_true(m2 == m1, "terminal marker short-circuits without reading the run config");
        expect_true(snapshot(dec) == before_second, "second call changes nothing");
        fs::remove(dec / "_run_markers" / "run1.json");
        record_jev_outcome_if_needed(dec, "run1", run);
        expect_equal(static_cast<long>(read_json(dec / "p_ok" / "outcome.json")["runs"].size()), 1L, "one entry per run even when the marker is lost");

        // ---- retryable read error ----
        {
            fs::remove(dec / "_run_markers" / "run1.json");
            fs::remove(dec / "p_ok" / "outcome.json");
            const json bad = record_jev_outcome_if_needed(dec, "run1", run, [](const fs::path&) -> json { throw std::runtime_error("disk hiccup"); });
            expect_true(bad["terminal"] == false && bad["reason"] == "config_unreadable", "read error is retryable, not terminal");
            const json good = record_jev_outcome_if_needed(dec, "run1", run);
            expect_true(good["terminal"] == true && fs::exists(dec / "p_ok" / "outcome.json"), "later call succeeds");
        }

        // ---- terminal edge cases ----
        {
            const fs::path run2 = root / "runs" / "run2";
            fs::create_directories(run2);
            const json missing = record_jev_outcome_if_needed(dec, "run2", run2);
            expect_true(missing["reason"] == "no_provenance" && missing["terminal"] == false,
                        "missing provenance can be retried after the run-start race");
            write_text(root / "runs" / "run3" / "artifacts" / "pi_run_provenance.json", json{{"started_at", "2026-09-23T09:00:00Z"}}.dump());
            expect_true(record_jev_outcome_if_needed(dec, "run3", root / "runs" / "run3")["reason"] == "no_linked_proposal", "run without a Jev link is not attributed");
            const auto snap = snapshot(dec);
            expect_true(record_jev_outcome_if_needed(dec, "../evil", run)["reason"] == "invalid_run_id", "path-like run id refused");
            expect_true(snapshot(dec) == snap, "refused id writes nothing");
        }

        const fs::path queued_run = root / "runs" / "queue1" / "L";
        write_text(queued_run / "artifacts" / "pi_run_provenance.json",
                   json{{"jev_proposal_id", "p_ok"}, {"started_at", "2026-09-23T11:00:00Z"}}.dump());
        write_text(queued_run / "config.yaml", slurp_file(run / "config.yaml"));
        const json queued_marker = record_jev_outcome_if_needed(dec, "queue1/L", queued_run);
        expect_true(queued_marker["reason"] == "recorded" && queued_marker["terminal"] == true,
                    "safe nested queue run id records its linked outcome");
        expect_true(fs::exists(dec / "_run_markers" / "queue1" / "L.json"),
                    "nested queue marker remains under pi_decisions");
        expect_true(record_jev_outcome_if_needed(dec, "queue1/../evil", queued_run)["reason"] == "invalid_run_id",
                    "nested traversal is rejected");

        fs::remove_all(root);
        std::puts("pi_decision_outcome: all checks passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        fs::remove_all(root);
        return 1;
    }
    return 0;
}
