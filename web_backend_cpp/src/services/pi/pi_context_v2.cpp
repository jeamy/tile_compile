#include "services/pi/pi_context_v2.hpp"

#include <fstream>
#include <optional>

namespace tile_compile::pi {
namespace {

std::optional<nlohmann::json> parse_json_file(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    if (!ifs) return std::nullopt;
    auto parsed = nlohmann::json::parse(ifs, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return std::nullopt;
    return parsed;
}

void add_fact(nlohmann::json& facts,
              const std::string& id,
              const nlohmann::json& value,
              const std::string& source,
              const std::string& unit = "") {
    nlohmann::json fact = {
        {"id", id},
        {"value", value},
        {"source", source},
        {"applicability", "measured"},
        {"confidence", 1.0}
    };
    if (!unit.empty()) fact["unit"] = unit;
    facts[id] = std::move(fact);
}

} // namespace

nlohmann::json build_scan_pi_context(const SchemaPathMap& schema_paths,
                                     const nlohmann::json& base_config,
                                     const nlohmann::json& scan_result,
                                     const nlohmann::json& scan_metrics,
                                     const std::string& context_kind) {
    nlohmann::json facts = nlohmann::json::object();

    if (scan_result.is_object()) {
        if (scan_result.contains("frames_detected")) add_fact(facts, "dataset.frame_count", scan_result["frames_detected"], "scan_result.frames_detected");
        else if (scan_result.contains("frames_total")) add_fact(facts, "dataset.frame_count", scan_result["frames_total"], "scan_result.frames_total");
        if (scan_result.contains("color_mode")) add_fact(facts, "dataset.color_mode", scan_result["color_mode"], "scan_result.color_mode");
        if (scan_result.contains("bayer_pattern")) add_fact(facts, "dataset.bayer_pattern", scan_result["bayer_pattern"], "scan_result.bayer_pattern");
    }
    if (scan_metrics.is_object() && scan_metrics.contains("aggregate") && scan_metrics["aggregate"].is_object()) {
        const nlohmann::json& aggregate = scan_metrics["aggregate"];
        for (const std::string key : {"fwhm", "background", "noise", "sky_gradient", "gradient_energy", "roundness", "star_count"}) {
            if (aggregate.contains(key)) {
                add_fact(facts, "scan_metrics." + key, aggregate[key], "scan_metrics.aggregate." + key);
            }
        }
    }

    return {
        {"schema_version", "pi.context.v2"},
        {"context_kind", context_kind},
        {"intent", "recommendation"},
        {"parameter_catalog", build_parameter_catalog(schema_paths, base_config)},
        {"facts", facts},
        {"evidence_rules", {
            {"require_fact_ids", true},
            {"schema_claims_must_match_parameter_catalog", true},
            {"diagnostic_only_not_quality_fix", true},
            {"successful_phase_thresholds_must_not_be_tightened_below_observed_value", true}
        }}
    };
}

nlohmann::json build_run_completed_pi_context(const SchemaPathMap& schema_paths,
                                              const nlohmann::json& base_config,
                                              const std::filesystem::path& run_dir,
                                              const nlohmann::json& run_status) {
    nlohmann::json context = build_scan_pi_context(schema_paths, base_config, nlohmann::json::object(), nlohmann::json::object(), "run_completed");
    context["intent"] = "recommendation";
    context["run_identity"] = {
        {"run_id", run_status.value("run_id", run_dir.filename().string())},
        {"run_dir", run_dir.string()},
        {"status", run_status.value("status", std::string())}
    };
    nlohmann::json& facts = context["facts"];

    if (run_status.is_object()) {
        if (run_status.contains("frames_detected")) add_fact(facts, "dataset.frame_count", run_status["frames_detected"], "run_status.frames_detected");
        if (run_status.contains("status")) add_fact(facts, "run.status", run_status["status"], "run_status.status");
    }

    std::ifstream events(run_dir / "logs" / "run_events.jsonl");
    std::string line;
    while (std::getline(events, line)) {
        auto parsed = nlohmann::json::parse(line, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) continue;
        if (parsed.value("type", std::string()) != "phase_end") continue;
        const std::string phase = parsed.value("phase_name", std::string());
        if (phase == "PCC") {
            if (parsed.contains("status")) add_fact(facts, "pcc.status", parsed["status"], "logs/run_events.jsonl:PCC.phase_end");
            if (parsed.contains("source")) add_fact(facts, "pcc.source", parsed["source"], "logs/run_events.jsonl:PCC.phase_end");
            if (parsed.contains("residual_rms")) add_fact(facts, "pcc.residual_rms", parsed["residual_rms"], "logs/run_events.jsonl:PCC.phase_end");
            if (parsed.contains("condition_number")) add_fact(facts, "pcc.condition_number", parsed["condition_number"], "logs/run_events.jsonl:PCC.phase_end");
            if (parsed.contains("stars_matched")) add_fact(facts, "pcc.stars_matched", parsed["stars_matched"], "logs/run_events.jsonl:PCC.phase_end");
            if (parsed.contains("stars_used")) add_fact(facts, "pcc.stars_used", parsed["stars_used"], "logs/run_events.jsonl:PCC.phase_end");
            if (parsed.contains("k_max")) add_fact(facts, "pcc.k_max", parsed["k_max"], "logs/run_events.jsonl:PCC.phase_end");
            if (parsed.contains("matrix") && parsed["matrix"].is_array() && parsed["matrix"].size() >= 3) {
                nlohmann::json diag = nlohmann::json::array();
                for (int i = 0; i < 3; ++i) {
                    if (parsed["matrix"][i].is_array() && parsed["matrix"][i].size() > static_cast<size_t>(i)) {
                        diag.push_back(parsed["matrix"][i][i]);
                    }
                }
                if (diag.size() == 3) add_fact(facts, "pcc.matrix_diag", diag, "logs/run_events.jsonl:PCC.phase_end");
            }
        } else if (phase == "REGISTRATION") {
            if (parsed.contains("reg_rejected_frames")) add_fact(facts, "registration.rejected_count", parsed["reg_rejected_frames"], "logs/run_events.jsonl:REGISTRATION.phase_end");
            if (parsed.contains("frames_cc_positive")) add_fact(facts, "registration.frames_cc_positive", parsed["frames_cc_positive"], "logs/run_events.jsonl:REGISTRATION.phase_end");
            if (parsed.contains("diag") && parsed["diag"].is_object()) {
                const nlohmann::json& diag = parsed["diag"];
                for (const std::string key : {"reg_residual_p90_px_median", "reg_residual_median_px_p90", "reg_model_blended", "reg_model_predicted_rejected"}) {
                    if (diag.contains(key)) add_fact(facts, "registration." + key, diag[key], "logs/run_events.jsonl:REGISTRATION.phase_end.diag");
                }
            }
        } else if (phase == "AQMH_RECONSTRUCTION") {
            for (const std::string key : {"cherry_pick_enabled", "cherry_pick_active_frac", "uniform_control_gate_triggered", "raw_aqmh_preserved_by_guard"}) {
                if (parsed.contains(key)) add_fact(facts, "aqmh." + key, parsed[key], "logs/run_events.jsonl:AQMH_RECONSTRUCTION.phase_end");
            }
        } else if (phase == "ASTROMETRY") {
            if (parsed.contains("status")) add_fact(facts, "astrometry.status", parsed["status"], "logs/run_events.jsonl:ASTROMETRY.phase_end");
            if (parsed.contains("pixel_scale_arcsec")) add_fact(facts, "astrometry.pixel_scale_arcsec", parsed["pixel_scale_arcsec"], "logs/run_events.jsonl:ASTROMETRY.phase_end", "arcsec/px");
        }
    }

    if (auto validation = parse_json_file(run_dir / "artifacts" / "validation.json")) {
        for (const std::string key : {"background_rms_increase_percent", "background_rms_ok", "fwhm_improvement_percent", "fwhm_improvement_ok", "input_background_rms", "output_background_rms"}) {
            if (validation->contains(key)) add_fact(facts, "validation." + key, (*validation)[key], "artifacts/validation.json");
        }
    }
    if (auto aqmh = parse_json_file(run_dir / "artifacts" / "aqmh_reconstruction.json")) {
        for (const std::string key : {"cherry_pick_active", "cherry_pick_enabled", "cherry_pick_forced_disabled", "selected_candidate", "uniform_control_gate_triggered", "raw_aqmh_preserved_by_guard"}) {
            if (aqmh->contains(key)) add_fact(facts, "aqmh." + key, (*aqmh)[key], "artifacts/aqmh_reconstruction.json");
        }
    }
    return context;
}

nlohmann::json build_run_chat_pi_context(const std::string& run_id,
                                         const std::filesystem::path& run_dir,
                                         const nlohmann::json& status,
                                         const nlohmann::json& artifacts,
                                         const nlohmann::json& problem_ids) {
    nlohmann::json facts = nlohmann::json::object();
    add_fact(facts, "run.status", status.value("status", std::string()), "run_status.status");
    add_fact(facts, "run.artifact_count", artifacts.is_array() ? artifacts.size() : 0, "list_run_artifacts");

    nlohmann::json problem_fact = {
        {"id", "run_chat.problem_hints"},
        {"value", problem_ids},
        {"source", "local_run_chat_hint_detector"},
        {"applicability", "inferred"},
        {"confidence", 0.7}
    };
    facts["run_chat.problem_hints"] = std::move(problem_fact);

    return {
        {"schema_version", "pi.context.v2"},
        {"context_kind", "run_live_or_completed_chat"},
        {"intent", "chat"},
        {"run_identity", {
            {"run_id", run_id},
            {"run_dir", run_dir.string()},
            {"status", status.value("status", std::string())}
        }},
        {"parameter_catalog", nlohmann::json::object()},
        {"facts", facts},
        {"artifact_index", artifacts},
        {"evidence_rules", {
            {"require_fact_ids_for_parameter_actions", true},
            {"schema_claims_must_be_explicitly_supported", true},
            {"do_not_invent_defaults_or_schema_ranges", true},
            {"memory_is_lower_priority_than_run_facts", true}
        }}
    };
}

} // namespace tile_compile::pi
