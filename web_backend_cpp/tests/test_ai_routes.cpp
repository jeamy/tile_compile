#include "backend_test_harness.hpp"
#include "fake_sidecar.hpp"
#include "services/ai_service.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

using backend_test::FakeSidecar;

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        const auto initial_config = harness.get_json("/api/ai/config");
        expect_equal(initial_config["_http_status"].get<long>(), 200L, "ai config status");
        expect_true(!initial_config["enabled"].get<bool>(), "ai scan default disabled");
        expect_equal(initial_config["mode"].get<std::string>(), "manual", "ai scan default mode");

        const auto disabled_analysis = harness.post_json("/api/scan/analysis", nlohmann::json::object());
        expect_equal(disabled_analysis["_http_status"].get<long>(), 200L, "disabled scan analysis status");
        expect_equal(disabled_analysis["status"].get<std::string>(), "AI_DISABLED", "disabled scan analysis code");
        expect_true(!disabled_analysis["enabled"].get<bool>(), "disabled scan analysis enabled false");

        const auto patched_config = harness.patch_json("/api/ai/config", {
            {"enabled", true},
            {"mode", "assistive"},
            {"provider", "anthropic"},
            {"model", "claude-test"},
            {"api_key", "must-not-persist"}
        });
        expect_equal(patched_config["_http_status"].get<long>(), 200L, "ai patch status");
        expect_true(patched_config["enabled"].get<bool>(), "ai patch enabled");
        expect_equal(patched_config["mode"].get<std::string>(), "assistive", "ai patch mode");
        expect_equal(patched_config["provider"].get<std::string>(), "anthropic", "ai patch provider");
        expect_equal(patched_config["model"].get<std::string>(), "claude-test", "ai patch model");
        expect_true(!patched_config.contains("api_key"), "ai patch never returns api key");
        const auto stored_config = nlohmann::json::parse(slurp_file(harness.runtime_dir() / "ai_scan_config.json"));
        expect_true(stored_config["enabled"].get<bool>(), "ai config persisted enabled");
        expect_equal(stored_config["provider"].get<std::string>(), "anthropic", "ai config persisted provider");
        expect_equal(stored_config["model"].get<std::string>(), "claude-test", "ai config persisted model");
        expect_true(!stored_config.contains("api_key"), "ai config never persists api key");

        const auto patched_ui_config = harness.patch_json("/api/ai/config", {
            {"ui", {
                {"mount", "Alt/Az"},
                {"object_type", "Nebel"},
                {"camera", "Mono CMOS"},
                {"calibration_darks", true},
                {"calibration_flats", true},
                {"calibration_bias", false},
                {"notes", "wide nebula test"}
            }}
        });
        expect_equal(patched_ui_config["_http_status"].get<long>(), 200L, "ai ui config patch status");
        expect_equal(patched_ui_config["ui"]["mount"].get<std::string>(), "Alt/Az", "ai ui config mount");
        expect_equal(patched_ui_config["provider"].get<std::string>(), "anthropic", "ai ui config preserves provider");
        const auto reloaded_ui_config = harness.get_json("/api/ai/config");
        expect_equal(reloaded_ui_config["ui"]["object_type"].get<std::string>(), "Nebel", "ai ui config persisted object type");
        expect_true(reloaded_ui_config["ui"]["calibration_flats"].get<bool>(), "ai ui config persisted flats");

        const auto malformed_config = harness.patch_json("/api/ai/config", {
            {"enabled", "true"},
            {"mode", false},
            {"provider", true},
            {"model", false},
            {"sidecar_url", true}
        });
        expect_equal(malformed_config["_http_status"].get<long>(), 200L, "ai malformed config patch status");
        expect_true(malformed_config["enabled"].get<bool>(), "ai malformed config bool string enabled");
        expect_equal(malformed_config["mode"].get<std::string>(), "assistive", "ai malformed config keeps mode fallback");
        expect_equal(malformed_config["provider"].get<std::string>(), "anthropic", "ai malformed config keeps provider fallback");
        expect_equal(malformed_config["model"].get<std::string>(), "claude-test", "ai malformed config keeps model fallback");

        const auto no_scan_analysis = harness.post_json("/api/scan/analysis", {
            {"scan_result", {{"has_scan", false}}}
        });
        if (no_scan_analysis["_http_status"].get<long>() != 400L) {
            throw TestFailure("no_scan_analysis unexpected response: " + no_scan_analysis.dump());
        }
        expect_equal(no_scan_analysis["_http_status"].get<long>(), 400L,
                     "enabled scan analysis without scan status: " + no_scan_analysis.dump());
        expect_equal(no_scan_analysis["code"].get<std::string>(), "NO_SCAN", "enabled scan analysis without scan code");

        FakeSidecar sidecar({
            {"schema_version", "pi.scan-analysis.v1"},
            {"summary", "fixture analysis"},
            {"confidence", 0.8},
            {"detected_scenarios", nlohmann::json::array()},
            {"recommendations", {
                {
                    {"path", "data.color_mode"},
                    {"value", "MONO"},
                    {"reason", true},
                    {"confidence", "0.9"},
                    {"risk", false},
                    {"evidence", {"scan_metrics.fwhm.median=2.4", true}}
                },
                {
                    {"path", "data.unknown"},
                    {"value", true},
                    {"reason", "fixture unknown path"},
                    {"confidence", 0.7},
                    {"risk", "medium"}
                },
                {
                    {"path", "data.color_mode"},
                    {"value", 123},
                    {"reason", "fixture wrong type"},
                    {"confidence", 0.6},
                    {"risk", "high"}
                },
                {
                    {"path", "pcc.max_residual_rms"},
                    {"value", 0.05},
                    {"reason", "Current value exceeds the schema-declared maximum and the schema recommends 0.05."},
                    {"confidence", 0.94},
                    {"risk", "high"}
                },
                {
                    {"path", "pcc.k_max"},
                    {"value", 0.5},
                    {"reason", "Current k_max is a physically implausible atmospheric extinction coefficient at the schema maximum."},
                    {"confidence", 0.88},
                    {"risk", "high"}
                },
                {
                    {"path", "reconstruction.quality.pyramid.base_window_px"},
                    {"value", 64},
                    {"reason", "Current value is below the schema range 16-256 and schema recommended value 64."},
                    {"confidence", 0.88},
                    {"risk", "medium"}
                },
                {
                    {"path", "reconstruction.diagnostics.preview_forward_drizzle_uniform"},
                    {"value", true},
                    {"reason", "Improves reconstruction quality by enabling the uniform preview."},
                    {"confidence", 0.8},
                    {"risk", "low"}
                },
                {
                    {"path", "registration.enable_local_background_subtraction"},
                    {"value", true},
                    {"reason", "The schema default is true, so false is a non-default misconfiguration."},
                    {"confidence", 0.72},
                    {"risk", "medium"}
                }
            }},
            {"warnings", nlohmann::json::array()},
            {"review_required", true}
        });
        sidecar.start();

        const auto sidecar_config = harness.patch_json("/api/ai/config", {
            {"enabled", true},
            {"provider", "fixture"},
            {"model", "fixture/model"},
            {"sidecar_url", sidecar.url()}
        });
        expect_equal(sidecar_config["_http_status"].get<long>(), 200L, "ai sidecar config status");

        const auto memory_dir = harness.fixture_root() / "runs" / ".pi_memory";
        std::filesystem::create_directories(memory_dir);
        {
            std::ofstream legacy(memory_dir / "memories.jsonl");
            legacy << nlohmann::json{
                {"schema_version", "pi.memory.v1"},
                {"memory_id", "legacy_memory_must_be_ignored"},
                {"status", "accepted"},
                {"type", "config_optimization"},
                {"summary", "legacy memory must not enter request context"}
            }.dump() << "\n";
            const nlohmann::json ctx = {
                {"schema_version", "pi.context_signature.v1"},
                {"target", {{"object_type", "galaxy"}}},
                {"acquisition", {{"camera_type", "OSC"}}},
                {"pipeline", {{"affected_paths", nlohmann::json::array({"data.color_mode"})}}}
            };
            const nlohmann::json scope = {
                {"applies_when", nlohmann::json::array({"matching fixture context"})},
                {"does_not_apply_when", nlohmann::json::array({"different color mode problem"})},
                {"confidence", 0.5}
            };
            std::ofstream out(memory_dir / "memories_v2.jsonl");
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_accepted"},
                {"id", "mem_scan_context_accepted"},
                {"status", "candidate"},
                {"type", "config_optimization"},
                {"source", "scan_ai_apply"},
                {"privacy_class", "metadata_only"},
                {"summary", "MONO was useful for this fixture"},
                {"context_signature", ctx},
                {"scope", scope},
                {"config_updates", nlohmann::json::array({{{"path", "data.color_mode"}, {"value", "MONO"}}})},
                {"recommendation", {{"explanation", "MONO was useful for this fixture"}}},
                {"evidence", {{"validation", "fixture"}}},
                {"outcome", {{"validation_valid", true}, {"applied_count", 1}}},
                {"validation", {{"valid", true}}},
                {"review", {{"status", "candidate"}, {"reviewed_by", nullptr}, {"reviewed_at", nullptr}, {"notes", ""}}},
                {"retrieval", {{"keywords", nlohmann::json::array({"data.color_mode"})}, {"negative", false}}}
            }.dump() << "\n";
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_rejected"},
                {"id", "mem_scan_context_rejected"},
                {"status", "candidate"},
                {"type", "config_optimization"},
                {"source", "scan_ai_apply"},
                {"privacy_class", "metadata_only"},
                {"summary", "Rejected memory must not become request context"},
                {"context_signature", ctx},
                {"scope", scope},
                {"config_updates", nlohmann::json::array({{{"path", "data.color_mode"}, {"value", "RGB"}}})},
                {"recommendation", {{"explanation", "Rejected memory must not become request context"}}},
                {"evidence", {{"validation", "fixture"}}},
                {"outcome", {{"validation_valid", false}, {"applied_count", 1}}},
                {"validation", {{"valid", true}}},
                {"review", {{"status", "candidate"}, {"reviewed_by", nullptr}, {"reviewed_at", nullptr}, {"notes", ""}}},
                {"retrieval", {{"keywords", nlohmann::json::array({"data.color_mode"})}, {"negative", false}}}
            }.dump() << "\n";
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_wrong_type"},
                {"id", "mem_scan_context_wrong_type"},
                {"status", "candidate"},
                {"type", "config_optimization"},
                {"source", "scan_ai_apply"},
                {"privacy_class", "metadata_only"},
                {"summary", "Accepted memory with invalid historical value must not bypass schema validation"},
                {"context_signature", ctx},
                {"scope", scope},
                {"config_updates", nlohmann::json::array({{{"path", "data.color_mode"}, {"value", 123}}})},
                {"recommendation", {{"explanation", "Accepted memory with invalid historical value must not bypass schema validation"}}},
                {"evidence", {{"validation", "fixture"}}},
                {"outcome", {{"validation_valid", true}, {"applied_count", 1}}},
                {"validation", {{"valid", true}}},
                {"review", {{"status", "candidate"}, {"reviewed_by", nullptr}, {"reviewed_at", nullptr}, {"notes", ""}}},
                {"retrieval", {{"keywords", nlohmann::json::array({"data.color_mode"})}, {"negative", false}}}
            }.dump() << "\n";
        }
        {
            std::ofstream out(memory_dir / "memory_reviews_v2.jsonl");
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_accepted"},
                {"id", "mem_scan_context_accepted"},
                {"status", "accepted"},
                {"reviewed_at", "2026-07-14T00:00:00Z"},
                {"reviewer", "fixture"},
                {"note", "useful"}
            }.dump() << "\n";
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_rejected"},
                {"id", "mem_scan_context_rejected"},
                {"status", "rejected"},
                {"reviewed_at", "2026-07-14T00:00:01Z"},
                {"reviewer", "fixture"},
                {"note", "bad"}
            }.dump() << "\n";
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_wrong_type"},
                {"id", "mem_scan_context_wrong_type"},
                {"status", "accepted"},
                {"reviewed_at", "2026-07-14T00:00:02Z"},
                {"reviewer", "fixture"},
                {"note", "historical context only"}
            }.dump() << "\n";
        }

        const auto analysis = harness.post_json("/api/scan/analysis", {
            {"force", true},
            {"scan_result", {
                {"frames_detected", 12},
                {"color_mode", "OSC"},
                {"frames", nlohmann::json::array({{{"header", {
                    {"OBJECT", "M42"},
                    {"TELESCOP", "RASA 8"},
                    {"INSTRUME", "ASI2600MC"},
                    {"FILTER", "HaOIII"},
                    {"EXPTIME", 180.0},
                    {"DATE-OBS", "2026-01-02T03:04:05"}
                }}}})}
            }},
            {"scan_metrics", {{"frames_total", 12}}},
            {"base_config", {{"data", {{"color_mode", "OSC"}}}}},
            {"model", false}
        });
        expect_equal(analysis["_http_status"].get<long>(), 200L, "scan ai analysis status");
        expect_equal(analysis["schema_version"].get<std::string>(), "pi.scan-analysis.v1", "scan ai schema");
        expect_equal(static_cast<long>(analysis["validated_updates"].size()), 1L, "scan ai validated update count");
        expect_equal(static_cast<long>(analysis["rejected_updates"].size()), 7L, "scan ai rejected update count");
        expect_equal(analysis["validated_updates"][0]["path"].get<std::string>(), "data.color_mode", "scan ai validated path");
        expect_equal(analysis["validated_updates"][0]["reason"].get<std::string>(), "true", "scan ai coerces boolean reason");
        expect_equal(analysis["validated_updates"][0]["risk"].get<std::string>(), "false", "scan ai coerces boolean risk");
        expect_equal(static_cast<long>(analysis["validated_updates"][0]["evidence"].size()), 2L, "scan ai preserves evidence");
        expect_equal(analysis["validation"]["valid"].get<bool>() ? "true" : "false", "true", "scan ai validation ok");
        const auto sidecar_request = sidecar.request_json();
        expect_equal(sidecar_request["ai_request"]["schema_version"].get<std::string>(),
                     "pi.ai-request.v2",
                     "scan ai request includes canonical ai request container");
        expect_equal(sidecar_request["ai_request"]["task"].get<std::string>(),
                     "scan_recommendation",
                     "scan ai canonical request task");
        expect_equal(sidecar_request["ai_request"]["context_signature"]["target"]["object_name"].get<std::string>(),
                     "M42",
                     "scan ai canonical request extracts target from FITS header");
        expect_equal(sidecar_request["ai_request"]["context_signature"]["optics"]["telescope"].get<std::string>(),
                     "RASA 8",
                     "scan ai canonical request extracts telescope from FITS header");
        expect_equal(sidecar_request["ai_request"]["context_signature"]["acquisition"]["filters"][0].get<std::string>(),
                     "HaOIII",
                     "scan ai canonical request extracts filter from FITS header");
        expect_equal(sidecar_request["ai_request"]["context_signature"]["acquisition"]["exposure_seconds"].get<double>(),
                     180.0,
                     "scan ai canonical request extracts exposure from FITS header");
        expect_equal(static_cast<long>(sidecar_request["ai_request"]["positive_memories"].size()), 2L,
                     "scan ai canonical request includes accepted pi memories");
        expect_equal(static_cast<long>(sidecar_request["ai_request"]["negative_memories"].size()), 1L,
                     "scan ai canonical request includes negative pi memories");
        expect_true(sidecar_request["ai_request"].contains("retrieval_coverage_summary"),
                    "scan ai canonical request includes retrieval_coverage_summary prompt section");
        expect_true(sidecar_request["ai_request"]["retrieval_coverage_summary"].is_object(),
                    "scan ai retrieval_coverage_summary is an object");
        expect_true(sidecar_request["ai_request"]["retrieval_coverage_summary"].contains("systemically_missing_context_fields"),
                    "scan ai retrieval_coverage_summary lists systemically_missing_context_fields");
        expect_true(sidecar_request["ai_request"]["retrieval_coverage_summary"].contains("note"),
                    "scan ai retrieval_coverage_summary includes explanatory note for the model");
        expect_true(sidecar_request.contains("pi_context"),
                    "scan ai request includes pi_context");
        expect_equal(sidecar_request["pi_context"]["schema_version"].get<std::string>(),
                     "pi.context.v2",
                     "scan ai request pi context schema");
        expect_true(sidecar_request["pi_context"]["parameter_catalog"].contains("pcc.max_residual_rms"),
                    "scan ai request includes pcc parameter metadata");
        expect_equal(static_cast<long>(sidecar_request["session_context"]["accepted_pi_memories"].size()), 2L,
                     "scan ai request includes accepted pi memories");
        bool found_accepted_memory = false;
        bool found_rejected_memory = false;
        for (const auto& memory : sidecar_request["session_context"]["accepted_pi_memories"]) {
            const std::string memory_id = memory.value("memory_id", std::string());
            if (memory_id == "mem_scan_context_accepted") found_accepted_memory = true;
            if (memory_id == "mem_scan_context_rejected") found_rejected_memory = true;
            expect_true(memory.contains("match_explanation"), "accepted memory context includes retrieval explanation");
            expect_true(memory.contains("match_coverage"), "accepted memory context includes retrieval coverage");
        }
        expect_true(found_accepted_memory, "scan ai request includes reviewed accepted memory");
        expect_true(!found_rejected_memory, "scan ai request excludes rejected memories");
        expect_equal(static_cast<long>(sidecar_request["session_context"]["negative_pi_memories"].size()), 1L,
                     "scan ai request includes negative pi memories");
        expect_equal(sidecar_request["session_context"]["negative_pi_memories"][0]["memory_id"].get<std::string>(),
                     "mem_scan_context_rejected",
                     "scan ai request carries rejected memory as negative signal");
        expect_true(sidecar_request["session_context"]["negative_pi_memories"][0].contains("match_explanation"),
                    "negative memory context includes retrieval explanation");
        bool rejected_wrong_type_from_memory_context = false;
        bool rejected_unsupported_schema_claim = false;
        bool rejected_diagnostic_quality_claim = false;
        bool rejected_default_claim = false;
        for (const auto& rejected : analysis["rejected_updates"]) {
            if (rejected.value("path", std::string()) == "data.color_mode" &&
                rejected.value("reject_reason", std::string()) == "wrong_type") {
                rejected_wrong_type_from_memory_context = true;
            }
            if (rejected.value("path", std::string()) == "pcc.max_residual_rms" &&
                rejected.value("reject_reason", std::string()) == "unsupported_schema_claim") {
                rejected_unsupported_schema_claim = true;
            }
            if (rejected.value("path", std::string()) == "reconstruction.diagnostics.preview_forward_drizzle_uniform" &&
                rejected.value("reject_reason", std::string()) == "diagnostic_only_quality_claim") {
                rejected_diagnostic_quality_claim = true;
            }
            if (rejected.value("path", std::string()) == "registration.enable_local_background_subtraction" &&
                rejected.value("reject_reason", std::string()) == "unsupported_default_claim") {
                rejected_default_claim = true;
            }
        }
        expect_true(rejected_wrong_type_from_memory_context,
                    "accepted memory context cannot bypass config schema validation");
        expect_true(rejected_unsupported_schema_claim,
                    "semantic validator rejects invented schema claims");
        expect_true(rejected_diagnostic_quality_claim,
                    "semantic validator rejects diagnostic-only quality claims");
        expect_true(rejected_default_claim,
                    "semantic validator rejects false schema default claim");
        expect_equal(analysis["action_plan"]["schema_version"].get<std::string>(),
                     "pi.action-plan.v1",
                     "scan ai attaches pi action plan");
        expect_true(analysis["action_plan_validation"]["valid"].get<bool>(),
                    "scan ai action plan validates");

        const auto context_store = harness.post_json("/api/scan/analysis/store", {
            {"analysis", {
                {"schema_version", "pi.scan-analysis.v1"},
                {"summary", "fixture context analysis"},
                {"confidence", 0.8},
                {"detected_scenarios", {"large_frame_count"}},
                {"recommendations", {
                    {
                        {"path", "reconstruction.diagnostics.preview_forward_drizzle_uniform"},
                        {"value", true},
                        {"reason", "fixture diagnostics preview"},
                        {"confidence", 0.9},
                        {"risk", "low"},
                        {"evidence", {"scan_metrics.fwhm.spread"}}
                    },
                    {
                        {"path", "reconstruction.clipping.min_fraction"},
                        {"value", 1.5},
                        {"reason", "fixture invalid high min_fraction"},
                        {"confidence", 0.9},
                        {"risk", "low"},
                        {"evidence", {"scan_metrics.frame_count=610"}}
                    },
                    {
                        {"path", "reconstruction.drizzle.internal_scale"},
                        {"value", 2},
                        {"reason", "fixture invalid internal scale"},
                        {"confidence", 0.9},
                        {"risk", "low"},
                        {"evidence", {"scan_metrics.frame_count=610"}}
                    }
                }},
                {"warnings", nlohmann::json::array()},
                {"review_required", false}
            }},
            {"scan_result", {
                {"frames_detected", 610},
                {"input_path", "/fixture/m42"},
                {"frames", nlohmann::json::array({{{"target", "M42"}}})}
            }},
            {"scan_metrics", {
                {"ok", true},
                {"sample_count", 122},
                {"frames_total", 610},
                {"sampling", {
                    {"strategy", "stratified_header_edges_even_fill"},
                    {"sample_target", 122},
                    {"selected_indices", {0, 1, 2, 607, 608, 609}}
                }},
                {"aggregate", {
                    {"fwhm", {{"median", 9.1}, {"p10", 8.9}, {"p90", 10.0}, {"count", 122}}}
                }},
                {"frames", {
                    {
                        {"index", 0},
                        {"sample_reasons", {"edge_start"}},
                        {"fwhm", 9.1},
                        {"header", {{"target", "M42"}}}
                    }
                }}
            }},
            {"base_config", {
                {"reconstruction", {
                    {"drizzle", {{"internal_scale", 1}}},
                    {"clipping", {{"min_fraction", 0.3}}}
                }}
            }},
            {"config_schema", {
                {"reconstruction.clipping.min_fraction", {{"type", "number"}, {"maximum", 1}}},
                {"reconstruction.drizzle.internal_scale", {{"type", "integer"}, {"enum", {1, 2}}}}
            }}
        });
        expect_equal(context_store["_http_status"].get<long>(), 200L, "context store status");
        expect_equal(context_store["analysis_context"]["frame_count"].get<long>(), 610L,
                     "context store preserves frame count");
        expect_equal(context_store["analysis_context"]["scan_metrics"]["sampling"]["sample_target"].get<long>(), 122L,
                     "context store preserves sampling target");
        expect_equal(static_cast<long>(context_store["analysis_context"]["scan_metrics"]["sampling"]["selected_indices"].size()), 6L,
                     "context store preserves selected indices");
        expect_equal(context_store["analysis_context"]["base_config"]["reconstruction"]["drizzle"]["internal_scale"].get<long>(), 1L,
                     "context store preserves base config");
        expect_true(context_store["analysis_context"]["config_schema"].contains("reconstruction.clipping.min_fraction"),
                    "context store preserves config schema");

        const auto history = harness.get_json("/api/scan/analysis/history?limit=20");
        expect_equal(history["_http_status"].get<long>(), 200L, "analysis history status");
        std::string context_filename;
        const std::string context_id = context_store["analysis_id"].get<std::string>();
        for (const auto& item : history["items"]) {
            if (item.value("analysis_id", std::string()) == context_id) {
                context_filename = item.value("filename", std::string());
                break;
            }
        }
        expect_true(!context_filename.empty(), "context analysis appears in persisted history");
        const auto context_file = harness.get_json("/api/scan/analysis/history/" + context_filename);
        expect_equal(context_file["_http_status"].get<long>(), 200L, "context persisted file status");
        expect_equal(context_file["analysis_context"]["scan_metrics"]["sampling"]["strategy"].get<std::string>(),
                     "stratified_header_edges_even_fill",
                     "context persisted file preserves sampling strategy");
        expect_true(context_store["action_plan_validation"]["valid"].get<bool>(),
                    "stored scan ai action plan validates");

        const std::string analysis_id = analysis["analysis_id"].get<std::string>();
        const auto apply = harness.post_json("/api/scan/analysis/apply", {
            {"analysis_id", analysis_id},
            {"base_config", {
                {"data", {{"color_mode", "OSC"}}},
                {"pcc", {{"max_residual_rms", 0.9}, {"k_max", 2.0}}},
                {"reconstruction", {
                    {"quality", {{"pyramid", {{"base_window_px", 4}}}}}
                }},
                {"reconstruction", {
                    {"diagnostics", {{"preview_forward_drizzle_uniform", false}}}
                }},
                {"registration", {{"enable_local_background_subtraction", false}}}
            }},
            {"selected_paths", {"data.color_mode"}},
            {"persist", true},
            {"learn", true}
        });
        expect_equal(apply["_http_status"].get<long>(), 200L, "scan ai apply status");
        expect_true(apply["ok"].get<bool>(), "scan ai apply ok");
        expect_equal(apply["config"]["data"]["color_mode"].get<std::string>(), "MONO", "scan ai apply config value");
        expect_equal(static_cast<long>(apply["applied_paths"].size()), 1L, "scan ai apply selected count");
        expect_true(apply.contains("revision_id"), "scan ai apply creates revision");
        expect_equal(apply["memory"]["type"].get<std::string>(), "config_optimization", "scan ai apply learns memory");
        expect_true(!apply["memory"].value("duplicate", false), "scan ai apply creates context-specific memory candidate");
        expect_equal(apply["memory"]["status"].get<std::string>(), "candidate", "scan ai learned memory starts as candidate");
        expect_true(apply["memory"].contains("context_signature"), "scan ai learned memory records context signature");
        expect_true(apply["memory"].contains("scope"), "scan ai learned memory records scope");
        expect_equal(apply["memory"]["context_signature"]["target"]["object_name"].get<std::string>(),
                     "M42",
                     "scan ai learned memory preserves FITS-derived target");
        expect_equal(apply["memory"]["context_signature"]["acquisition"]["filters"][0].get<std::string>(),
                     "HaOIII",
                     "scan ai learned memory preserves FITS-derived filter");

        const auto rounded_store = harness.post_json("/api/scan/analysis/store", {
            {"analysis", {
                {"schema_version", "pi.scan-analysis.v1"},
                {"summary", "fixture rounded float"},
                {"confidence", 0.8},
                {"detected_scenarios", nlohmann::json::array()},
                {"recommendations", {
                    {
                        {"path", "reconstruction.clipping.min_fraction"},
                        {"value", 0.29999999999999999},
                        {"reason", "fixture float noise"},
                        {"confidence", 0.9},
                        {"risk", "low"},
                        {"evidence", {"fixture"}}
                    }
                }},
                {"warnings", nlohmann::json::array()},
                {"review_required", false}
            }},
            {"scan_result", {{"frames_detected", 10}}},
            {"base_config", {
                {"reconstruction", {
                    {"clipping", {{"min_fraction", 0.4}}},
                    {"drizzle", {{"internal_scale", 1}}}
                }}
            }},
            {"config_schema", {
                {"reconstruction.clipping.min_fraction", {{"type", "number"}, {"maximum", 1}}}
            }}
        });
        expect_equal(rounded_store["_http_status"].get<long>(), 200L, "rounded float store status");
        expect_equal(static_cast<long>(rounded_store["validated_updates"].size()), 1L,
                     "rounded float store validated updates");
        const std::string rounded_id = rounded_store["analysis_id"].get<std::string>();
        const auto rounded_apply = harness.post_json("/api/scan/analysis/apply", {
            {"analysis_id", rounded_id},
            {"base_config", {
                {"reconstruction", {
                    {"clipping", {{"min_fraction", 0.4}}},
                    {"drizzle", {{"internal_scale", 1}}}
                }}
            }},
            {"selected_paths", {"reconstruction.clipping.min_fraction"}},
            {"persist", false},
            {"learn", true}
        });
        expect_equal(rounded_apply["_http_status"].get<long>(), 200L, "rounded float apply status");
        const std::string rounded_yaml = rounded_apply["config_yaml"].get<std::string>();
        expect_true(rounded_yaml.find("min_fraction: 0.3") != std::string::npos,
                    "rounded float yaml uses compact decimal: " + rounded_yaml);
        expect_true(rounded_yaml.find("0.299999999999999") == std::string::npos,
                    "rounded float yaml omits binary noise: " + rounded_yaml);
        expect_true(rounded_apply["memory"]["outcome"]["validation_valid"].get<bool>(),
                    "learned memory records validation outcome");
        expect_equal(rounded_apply["memory"]["outcome"]["applied_count"].get<long>(), 1L,
                     "learned memory records applied count");
        expect_equal(rounded_apply["memory"]["outcome"]["applied_paths"][0].get<std::string>(),
                     "reconstruction.clipping.min_fraction",
                     "learned memory records applied path");

        const auto missing_apply = harness.post_json("/api/scan/analysis/apply", nlohmann::json::object());
        expect_equal(missing_apply["_http_status"].get<long>(), 400L, "scan ai apply missing id status");

        FakeSidecar account_sidecar({
            {"schema_version", "pi.account-status.v1"},
            {"privacy_class", "metadata_only"},
            {"provider", "openai"},
            {"selected", {
                {"provider", "openai"},
                {"key_configured", true},
                {"auth_source", "env"},
                {"credit_query_supported", false},
                {"subscription_query_supported", false},
                {"billing_url", "https://platform.openai.com/settings/organization/billing/overview"}
            }},
            {"providers", nlohmann::json::array()}
        });
        account_sidecar.start();
        const auto account_config = harness.patch_json("/api/ai/config", {
            {"sidecar_url", account_sidecar.url()}
        });
        expect_equal(account_config["_http_status"].get<long>(), 200L, "ai account sidecar config status");
        const auto account = harness.get_json("/api/ai/account?provider=openai");
        expect_equal(account["_http_status"].get<long>(), 200L, "ai account status route");
        expect_equal(account["schema_version"].get<std::string>(), "pi.account-status.v1", "ai account schema");
        expect_equal(account["selected"]["provider"].get<std::string>(), "openai", "ai account selected provider");
        expect_true(!account["selected"]["credit_query_supported"].get<bool>(),
                    "ai account does not claim automatic credit support");

        const auto models = harness.get_json("/api/ai/models");
        expect_equal(models["_http_status"].get<long>(), 200L, "ai models unavailable status is non-fatal");
        expect_true(!models["available"].get<bool>(), "ai models unavailable flag");
        expect_equal(models["error"]["code"].get<std::string>(), "AI_AGENT_UNAVAILABLE", "ai models unavailable code");

        const auto redacted = tile_compile::ai::redact_ai_payload_for_log({
            {"provider", "anthropic"},
            {"api_key", "secret-key"},
            {"nested", {{"access_token", "secret-token"}}}
        });
        expect_equal(redacted["api_key"].get<std::string>(), "[REDACTED]", "ai log redacts api key");
        expect_equal(redacted["nested"]["access_token"].get<std::string>(), "[REDACTED]",
                     "ai log redacts nested token");

        FakeSidecar auth_error_sidecar({
            {"error", {
                {"code", "INVALID_API_KEY"},
                {"message", "invalid x-api-key"}
            }}
        }, 401);
        auth_error_sidecar.start();
        const auto auth_error_config = harness.patch_json("/api/ai/config", {
            {"sidecar_url", auth_error_sidecar.url()}
        });
        expect_equal(auth_error_config["_http_status"].get<long>(), 200L, "ai auth error sidecar config status");
        const auto auth_error = harness.post_json("/api/ai/auth", {
            {"provider", "anthropic"},
            {"api_key", "secret-key"}
        });
        expect_equal(auth_error["_http_status"].get<long>(), 401L, "ai auth preserves upstream status");
        expect_equal(auth_error["_upstream_status"].get<long>(), 401L, "ai auth exposes upstream status");
        expect_equal(auth_error["error"]["code"].get<std::string>(), "INVALID_API_KEY",
                     "ai auth preserves upstream error payload");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
