#include "services/pi/pi_post_run.hpp"

#include "services/pi/pi_decision_state.hpp"
#include "services/pi/pi_json_io.hpp"
#include "services/pi/pi_pre_rules.hpp"
#include "services/pi/pi_resume_scope.hpp"

#include <algorithm>
#include <fstream>
#include <set>

namespace tile_compile::pi {
namespace fs = std::filesystem;
using nlohmann::json;
namespace {

constexpr const char* kStateSchema = "pi.post-run-decision-state.v1";
constexpr const char* kAdviceSchema = "pi.post-run-advice.v1";
const char* const kCandidateKeys[] = {"drizzle_raw", "drizzle_uniform", "drizzle_multiband"};
const char* const kMetricKeys[] = {"background_rms", "median_fwhm", "p90_fwhm", "elongation", "tail", "seam_score"};

json plain_metric(const json& m) {
    json out = {{"applicable", m.is_object() && m.value("applicable", false)}};
    if (out["applicable"].get<bool>() && m.contains("value") && m["value"].is_number()) {
        out["value"] = m["value"];
        if (m.contains("sample_count") && m["sample_count"].is_number_integer()) out["sample_count"] = m["sample_count"];
    }
    return out;
}

// Phases that finished, in log order, plus the terminal event. Tolerant of a truncated last line.
void read_events(const fs::path& log, json& phases, json& run_end, bool& log_present) {
    std::ifstream in(log);
    log_present = static_cast<bool>(in);
    std::string line;
    std::set<std::string> seen;
    while (std::getline(in, line)) {
        const json e = json::parse(line, nullptr, false);
        if (!e.is_object()) continue;
        const std::string type = e.value("type", std::string());
        if (type == "phase_end" && e.contains("phase_name") && e["phase_name"].is_string()) {
            const std::string name = e["phase_name"].get<std::string>();
            if (seen.insert(name).second) phases.push_back(name);
        } else if (type == "run_end") {
            run_end = e;  // the LAST run_end wins (a resume appends another)
        }
    }
}

} // namespace

json build_post_run_state(const fs::path& run_dir, const json& run_config) {
    json missing = json::array();
    const auto provenance = read_json_file_opt(run_dir / "artifacts" / "run_provenance.json");
    const auto fd = read_json_file_opt(run_dir / "artifacts" / "forward_drizzle.json");
    json phases = json::array(), run_end = nullptr;
    bool log_present = false;
    read_events(run_dir / "logs" / "run_events.jsonl", phases, run_end, log_present);
    if (!provenance) missing.push_back("run_provenance_missing");
    if (!fd) missing.push_back("forward_drizzle_artifact_missing");
    if (!log_present) missing.push_back("run_events_missing");
    else if (run_end.is_null()) missing.push_back("run_end_event_missing");
    if (!run_config.is_object()) missing.push_back("run_config_missing");

    json run = {{"run_id", run_dir.filename().string()}, {"phases_completed", phases}};
    if (run_end.is_object()) {
        run["status"] = run_end.value("status", std::string());
        run["success"] = run_end.value("success", false);
        run["final_image_available"] = run_end.value("final_image_available", false);
        run["execution_scope"] = run_end.value("execution_scope", std::string());
        if (run_end.contains("cache_retention")) run["cache_retention"] = run_end["cache_retention"];
    }
    json identity = json::object();
    if (provenance) {
        identity["config_sha256"] = provenance->value("config", json::object()).value("sha256", std::string());
        identity["build_id"] = provenance->value("build", json::object()).value("build_id", std::string());
        identity["input_manifest_sha256"] = provenance->value("input_manifest", json::object()).value("sha256", std::string());
        identity["execution_scope"] = provenance->value("execution_scope", std::string());
    }

    json reconstruction = {{"available", fd.has_value()}};
    if (fd) {
        reconstruction["selected_candidate"] = fd->value("selected_candidate", std::string());
        reconstruction["selection_reason"] = fd->value("selection_reason", std::string());
        reconstruction["commit_complete"] = fd->value("commit_complete", false);
        json cands = json::object();
        const json val = fd->value("validation", json::object());
        for (const char* c : kCandidateKeys) {
            if (!val.contains(c) || !val[c].is_object()) continue;
            json one = {{"numerics_ok", val[c].value("numerics_ok", false)}, {"support_ok", val[c].value("support_ok", false)}};
            for (const char* m : kMetricKeys)
                if (val[c].contains(m)) one[m] = plain_metric(val[c][m]);
            cands[c] = one;
        }
        reconstruction["candidates"] = cands;
        if (cands.empty()) missing.push_back("candidate_validation_missing");
    }

    std::error_code ec;
    json artifacts = {
        {"canvas_mask", fs::is_regular_file(run_dir / "outputs" / "canvas_mask.fits", ec)},
        {"normalized_cache", fs::is_directory(run_dir / "cache" / "normalized_frames", ec)},
        {"source_quality_cache", fs::is_directory(run_dir / "cache" / "source_quality_maps", ec)},
    };
    json downstream = json::object();
    for (const char* name : {"bge", "pcc", "chroma_denoise", "luma_denoise"})
        downstream[name] = fs::is_regular_file(run_dir / "artifacts" / (std::string(name) + ".json"), ec);

    json state = {{"schema_version", kStateSchema}, {"run", run}, {"identity", identity}, {"reconstruction", reconstruction},
                  {"artifacts", artifacts}, {"downstream_artifacts", downstream}, {"missing", missing},
                  {"not_measured", json::array({"final_stretched_image_quality", "star_shape_after_downstream"})}};
    state["state_hash"] = sha256_prefixed(state.dump());
    return state;
}

json advise_post_run(const json& state, const json& run_config, const DecisionPolicy& policy, const DecisionCatalog& catalog,
                     const ConfigValidator& validate_config, const std::vector<std::string>& dismissed) {
    json findings = json::array(), suggestions = json::array(), excluded = json::array();
    auto finding = [&](const char* code, const char* severity, const std::string& detail) {
        findings.push_back({{"code", code}, {"severity", severity}, {"detail", detail}});
    };
    const json run = state.value("run", json::object());
    const json rec = state.value("reconstruction", json::object());

    bool diagnosable = true;
    if (!run.value("success", false)) { finding("run_not_successful", "warning", run.value("status", std::string("unknown"))); diagnosable = false; }
    if (!rec.value("available", false)) { finding("reconstruction_artifact_missing", "warning", "forward_drizzle.json"); diagnosable = false; }
    else if (!rec.value("commit_complete", false)) { finding("reconstruction_not_committed", "warning", "commit_complete=false"); diagnosable = false; }
    for (const auto& m : state.value("missing", json::array()))
        if (m.is_string()) finding("evidence_missing", "info", m.get<std::string>());

    if (diagnosable) {
        const std::string sel = rec.value("selected_candidate", std::string());
        const json cands = rec.value("candidates", json::object());
        if (!sel.empty() && cands.contains(sel) &&
            (!cands[sel].value("numerics_ok", false) || !cands[sel].value("support_ok", false)))
            finding("selected_candidate_numerics_or_support_failed", "warning", sel);
        // A rejected multiband candidate is a normal, gate-decided outcome, not a defect.
        if (sel == "drizzle_raw" && !rec.value("selection_reason", std::string()).empty())
            finding("selected_candidate_reason", "info", rec["selection_reason"].get<std::string>());
    }

    const std::set<std::string> dismissed_set(dismissed.begin(), dismissed.end());
    if (diagnosable) {
        // Locks and blocking findings of the pre-run state do not exist for a finished run: an empty stub keeps the shared
        // validation path (paths, values, preconditions, config validity) and nothing else.
        const json stub = {{"locked_paths", json::array()}, {"blocking_findings", json::array()}, {"groups", json::array()}};
        const PreRunCandidates c = build_pre_run_candidates(stub, run_config, policy, catalog, validate_config);
        for (const auto& a : c.applicable) {
            const std::string id = a.value("candidate_id", std::string());
            if (a.value("updates", json::array()).empty()) continue;  // baselines
            if (dismissed_set.count(id)) { excluded.push_back({{"candidate_id", id}, {"reasons", json::array({"previously_dismissed"})}}); continue; }
            json ups = a["updates"];
            bool full_run = false;
            std::optional<std::string> earliest;
            for (const auto& u : ups) {
                const auto phase = min_resume_phase_for_path(u.value("path", std::string()));
                if (!phase) { full_run = true; continue; }
                const auto& order = resume_phases_latest_first();
                if (!earliest || std::find(order.begin(), order.end(), *phase) > std::find(order.begin(), order.end(), *earliest)) earliest = phase;
            }
            json s = {{"candidate_id", id}, {"group", a.value("group", std::string())}, {"updates", ups},
                      {"requires_review", a.value("requires_review", true)}, {"experimental", a.value("experimental", true)},
                      {"resume_mode", full_run ? "full_run" : "resume"},
                      {"min_resume_phase", full_run || !earliest ? json(nullptr) : json(*earliest)},
                      {"feasibility", "not_checked_use_resume_dry_run"}};
            suggestions.push_back(s);
        }
        // Why change candidates were NOT offered, for the reconstruction/downstream groups only (pre-run-only evidence
        // candidates would just say "evidence unavailable").
        for (const auto& e : c.excluded) {
            const std::string id = e.value("candidate_id", std::string());
            if (dismissed_set.count(id)) continue;
            json rs = json::array();
            for (const auto& r : e.value("reasons", json::array()))
                if (r.is_string() && r.get<std::string>().rfind("evidence_", 0) != 0) rs.push_back(r);
            if (!rs.empty()) excluded.push_back({{"candidate_id", id}, {"reasons", rs}});
        }
    }

    const bool any_full = std::any_of(suggestions.begin(), suggestions.end(), [](const json& s) { return s["resume_mode"] == "full_run"; });
    std::string outcome = "no_change";
    if (!suggestions.empty()) outcome = any_full ? "suggest_reconstruction" : "suggest_downstream";
    else if (std::any_of(findings.begin(), findings.end(), [](const json& f) { return f["severity"] == "warning"; })) outcome = "diagnose";

    return {{"schema_version", kAdviceSchema}, {"run_id", run.value("run_id", std::string())}, {"outcome", outcome},
            {"findings", findings}, {"suggestions", suggestions}, {"excluded", excluded},
            {"dismissed", json(dismissed)}, {"state_hash", state.value("state_hash", std::string())},
            {"not_measured", state.value("not_measured", json::array())},
            {"basis", "run_artifacts_only"}, {"model_called", false}};
}

} // namespace tile_compile::pi
