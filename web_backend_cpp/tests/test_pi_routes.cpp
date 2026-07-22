#include "backend_test_harness.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fitsio.h>

static void create_minimal_fits(const std::filesystem::path& path, int width = 64, int height = 64) {
    fitsfile* fptr = nullptr;
    int status = 0;
    long naxes[2] = {width, height};
    std::filesystem::create_directories(path.parent_path());
    if (fits_create_file(&fptr, path.string().c_str(), &status)) {
        fits_report_error(stderr, status);
        throw std::runtime_error("fits_create_file failed");
    }
    if (fits_create_img(fptr, 16, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        throw std::runtime_error("fits_create_img failed");
    }
    std::vector<short> pixels(static_cast<size_t>(width) * height, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            pixels[y * width + x] = static_cast<short>(
                ((x + y) * 32000) / (width + height - 2));
        }
    }
    long first[2] = {1, 1};
    if (fits_write_pix(fptr, TSHORT, first, pixels.size(), pixels.data(), &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        throw std::runtime_error("fits_write_pix failed");
    }
    fits_close_file(fptr, &status);
}

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        const auto initial = harness.get_json("/api/pi/context");
        expect_equal(initial["_http_status"].get<long>(), 200L, "pi context status");
        expect_equal(initial["schema_version"].get<std::string>(), "pi.context-overview.v1", "pi context schema");
        expect_equal(initial["privacy_class"].get<std::string>(), "metadata_only", "pi context privacy");
        expect_true(initial["runtime"]["project_root"].contains("name"), "pi context has project root summary");
        expect_true(initial["runtime"]["runs_dir"].contains("name"), "pi context has runs dir summary");
        expect_true(initial["jobs"].is_array(), "pi context jobs array");

        const auto input_dir = harness.make_file("lights/frame_0.fits", "fixture");
        const auto scan = harness.post_json("/api/scan", {
            {"input_dir", input_dir.parent_path().string()},
            {"frames_min", 1}
        });
        expect_equal(scan["_http_status"].get<long>(), 200L, "scan launch status");
        const std::string job_id = scan["job_id"].get<std::string>();
        const auto final_job = harness.wait_for_job(job_id);
        expect_equal(final_job["state"].get<std::string>(), "ok", "scan job ok");

        const auto with_scan = harness.get_json("/api/pi/context");
        expect_equal(with_scan["_http_status"].get<long>(), 200L, "pi context with scan status");
        expect_true(!with_scan["latest_scan_job"].is_null(), "pi context latest scan available");
        expect_equal(with_scan["latest_scan_job"]["type"].get<std::string>(), "scan", "pi context latest scan type");
        expect_true(with_scan["latest_scan_job"]["result"].contains("frames_detected"), "pi context scan result compact");

        const auto tools = harness.get_json("/api/pi/tools");
        expect_equal(tools["_http_status"].get<long>(), 200L, "pi tools list status");
        expect_equal(tools["schema_version"].get<std::string>(), "pi.tools-list.v1", "pi tools list schema");
        expect_equal(tools["tool_registry_version"].get<std::string>(), "1.1.0", "pi tool registry version");
        expect_true(tools["tools"].is_array(), "pi tools array");
        expect_true(tools["tools"].size() >= 2, "pi tools list contains initial tools");
        bool found_bge_preview_tool = false;
        bool found_hms_preview_tool = false;
        bool found_resume_preview_tool = false;
        for (const auto& tool : tools["tools"]) {
            const auto name = tool.value("name", std::string());
            expect_true(tool.value("read_only", false), "pi tool is read-only");
            expect_true(tool.value("mutation_free", false), "pi tool is mutation-free");
            expect_equal(tool["write_policy"].get<std::string>(),
                         "no_direct_writes_use_action_plan_preview_apply",
                         "pi tool write policy");
            if (name == "preview.bge.plan") found_bge_preview_tool = true;
            if (name == "preview.hms.plan") found_hms_preview_tool = true;
            if (name == "preview.resume.plan") found_resume_preview_tool = true;
        }
        expect_true(found_bge_preview_tool, "pi tools list contains bge preview plan");
        expect_true(found_hms_preview_tool, "pi tools list contains hms preview plan");
        expect_true(found_resume_preview_tool, "pi tools list contains generic resume preview plan");

        const auto context_tool = harness.post_json("/api/pi/tools/call", {
            {"name", "context.overview"},
            {"input", nlohmann::json::object()}
        });
        expect_equal(context_tool["_http_status"].get<long>(), 200L, "pi context tool status");
        expect_true(context_tool["ok"].get<bool>(), "pi context tool ok");
        expect_equal(context_tool["result"]["schema_version"].get<std::string>(), "pi.context-overview.v1",
                     "pi context tool result schema");

        const auto schema_tool = harness.post_json("/api/pi/tools/call", {
            {"name", "config.schema.summary"},
            {"input", nlohmann::json::object()}
        });
        expect_equal(schema_tool["_http_status"].get<long>(), 200L, "pi schema tool status");
        expect_true(schema_tool["ok"].get<bool>(), "pi schema tool ok");
        expect_true(schema_tool["result"].contains("available"), "pi schema tool availability field");

        const auto unknown = harness.post_json("/api/pi/tools/call", {
            {"name", "missing.tool"},
            {"input", nlohmann::json::object()}
        });
        expect_equal(unknown["_http_status"].get<long>(), 404L, "pi unknown tool status");
        expect_true(!unknown["ok"].get<bool>(), "pi unknown tool not ok");

        const auto answer = harness.post_json("/api/pi/assistant/ask", {
            {"question", "Was sagt der letzte Scan zu den Frames?"}
        });
        expect_equal(answer["_http_status"].get<long>(), 200L, "pi assistant ask status");
        expect_equal(answer["schema_version"].get<std::string>(), "pi.assistant-answer.v1", "pi assistant answer schema");
        expect_equal(answer["mode"].get<std::string>(), "local_read_only", "pi assistant mode");
        expect_true(answer["answer"].get<std::string>().find("Frames") != std::string::npos,
                    "pi assistant scan answer mentions frames");
        expect_true(answer["evidence"].is_array() && !answer["evidence"].empty(), "pi assistant evidence present");

        const auto run_dir = harness.create_run("pi_fixture_run", {
            {{"type", "phase_start"}, {"phase_name", "BGE"}},
            {{"type", "phase_end"}, {"phase_name", "BGE"}, {"status", "ok"}}
        });
        harness.make_file("runs/pi_fixture_run/artifacts/bge.json", "{\"ok\":true}\n");
        harness.make_file("runs/pi_fixture_run/artifacts/stats.json", "{\"phase_issues\":[],\"summary\":{\"status\":\"ok\"}}\n");

        const auto artifact_tool = harness.post_json("/api/pi/tools/call", {
            {"name", "run.artifacts.summary"},
            {"input", {{"run_id", "pi_fixture_run"}}}
        });
        expect_equal(artifact_tool["_http_status"].get<long>(), 200L, "pi artifact tool status");
        expect_true(artifact_tool["ok"].get<bool>(), "pi artifact tool ok");
        expect_equal(artifact_tool["result"]["run_id"].get<std::string>(), "pi_fixture_run", "pi artifact run id");
        expect_true(artifact_tool["result"]["artifacts"]["count"].get<int>() >= 2, "pi artifact count");

        const auto report_tool = harness.post_json("/api/pi/tools/call", {
            {"name", "run.report.summary"},
            {"input", {{"run_id", "pi_fixture_run"}}}
        });
        expect_equal(report_tool["_http_status"].get<long>(), 200L, "pi report tool status");
        expect_true(report_tool["ok"].get<bool>(), "pi report tool ok");
        expect_true(report_tool["result"]["report_stats_available"].get<bool>(), "pi report stats available");

        const auto bge_preview_tool = harness.post_json("/api/pi/tools/call", {
            {"name", "preview.bge.plan"},
            {"input", {{"run_id", "pi_fixture_run"}}}
        });
        expect_equal(bge_preview_tool["_http_status"].get<long>(), 200L, "pi bge preview tool status");
        expect_true(bge_preview_tool["ok"].get<bool>(), "pi bge preview tool ok");
        expect_equal(bge_preview_tool["result"]["schema_version"].get<std::string>(), "pi.preview-plan.v1",
                     "pi bge preview schema");
        expect_equal(bge_preview_tool["result"]["phase"].get<std::string>(), "BGE", "pi bge preview phase");
        expect_true(bge_preview_tool["result"]["mutation_free"].get<bool>(), "pi bge preview mutation free");
        expect_true(!bge_preview_tool["result"]["safety"]["will_start_job"].get<bool>(), "pi bge preview no job start");
        expect_true(bge_preview_tool["result"]["ready"].get<bool>(), "pi bge preview ready");
        expect_equal(bge_preview_tool["result"]["phase_state"]["phase"].get<std::string>(), "BGE",
                     "pi bge preview includes phase state");
        expect_true(bge_preview_tool["result"]["artifact_summary"]["count"].get<int>() >= 2,
                    "pi bge preview includes artifact summary");
        expect_true(bge_preview_tool["result"]["report_stats_available"].get<bool>(),
                    "pi bge preview includes report stats availability");

        const auto hms_preview_tool = harness.post_json("/api/pi/tools/call", {
            {"name", "preview.hms.plan"},
            {"input", {{"run_id", "pi_fixture_run"}}}
        });
        expect_equal(hms_preview_tool["_http_status"].get<long>(), 200L, "pi hms preview tool status");
        expect_true(hms_preview_tool["ok"].get<bool>(), "pi hms preview tool ok");
        expect_equal(hms_preview_tool["result"]["phase"].get<std::string>(), "HYPERMETRIC_STRETCH",
                     "pi hms preview phase");

        const auto resume_preview_tool = harness.post_json("/api/pi/tools/call", {
            {"name", "preview.resume.plan"},
            {"input", {{"run_id", "pi_fixture_run"}, {"from_phase", "BGE"}}}
        });
        expect_equal(resume_preview_tool["_http_status"].get<long>(), 200L, "pi resume preview tool status");
        expect_true(resume_preview_tool["ok"].get<bool>(), "pi resume preview tool ok");
        expect_equal(resume_preview_tool["result"]["tool_action"].get<std::string>(), "preview.resume",
                     "pi resume preview action type");
        expect_equal(resume_preview_tool["result"]["planned_command"]["args"][4].get<std::string>(), "BGE",
                     "pi resume preview planned phase");

        const auto pcc_preview_tool = harness.post_json("/api/pi/tools/call", {
            {"name", "preview.resume.plan"},
            {"input", {{"run_id", "pi_fixture_run"}, {"from_phase", "PCC"}}}
        });
        expect_equal(pcc_preview_tool["_http_status"].get<long>(), 200L, "pi pcc preview tool status");
        expect_true(pcc_preview_tool["ok"].get<bool>(), "pi pcc preview tool ok");
        expect_equal(pcc_preview_tool["result"]["phase"].get<std::string>(), "PCC",
                     "pi pcc preview phase");

        const auto invalid_bge_preview_tool = harness.post_json("/api/pi/tools/call", {
            {"name", "preview.bge.plan"},
            {"input", {{"run_id", "pi_fixture_run"}, {"yaml", "data:\n  color_mode: OSC\ninvalid: true\n"}}}
        });
        expect_equal(invalid_bge_preview_tool["_http_status"].get<long>(), 200L, "pi invalid bge preview tool status");
        expect_true(invalid_bge_preview_tool["ok"].get<bool>(), "pi invalid bge preview tool ok");
        expect_true(!invalid_bge_preview_tool["result"]["ready"].get<bool>(), "pi invalid bge preview not ready");
        expect_true(!invalid_bge_preview_tool["result"]["config_valid"].get<bool>(), "pi invalid bge config invalid");

        harness.post_json("/api/runs/pi_fixture_run/set-current", nlohmann::json::object());
        const auto report_answer = harness.post_json("/api/pi/assistant/ask", {
            {"question", "Welche Report Artefakte und Phasenprobleme gibt es?"}
        });
        expect_equal(report_answer["_http_status"].get<long>(), 200L, "pi report assistant status");
        expect_true(report_answer["answer"].get<std::string>().find("Artefakte") != std::string::npos,
                    "pi report assistant mentions artifacts");
        expect_true(report_answer["evidence"].size() >= 2, "pi report assistant evidence count");

        const auto run_chat = harness.post_json("/api/pi/run-chat", {
            {"run_id", "pi_fixture_run"},
            {"message", "Unten im Zentrum Sterne mit schwarzen Zentren. Der Nebel, in dem diese Sterne sind, ist abgeschnitten und kaum zu erkennen. Loesungsmoeglichkeiten?"}
        });
        expect_equal(run_chat["_http_status"].get<long>(), 200L, "pi run chat status");
        expect_equal(run_chat["schema_version"].get<std::string>(), "pi.run-chat-answer.v1", "pi run chat schema");
        expect_equal(run_chat["context"]["schema_version"].get<std::string>(), "pi.run-chat-context.v1", "pi run chat context schema");
        expect_true(run_chat["context"]["problem_hints"].is_array() && run_chat["context"]["problem_hints"].size() >= 3,
                    "pi run chat detects problem hints");
        bool saw_black_star_cores = false;
        for (const auto& hint : run_chat["context"]["problem_hints"]) {
            if (hint.value("id", std::string()) == "black_star_cores") saw_black_star_cores = true;
        }
        expect_true(saw_black_star_cores, "pi run chat detects black star centers");
        expect_true(run_chat["likely_causes"].is_array() && !run_chat["likely_causes"].empty(),
                    "pi run chat likely causes");
        expect_true(run_chat["checks"].is_array() && !run_chat["checks"].empty(), "pi run chat checks");
        expect_true(run_chat["recommendations"].is_array() && !run_chat["recommendations"].empty(),
                    "pi run chat recommendations");
        expect_true(run_chat["evidence"].is_array() && run_chat["evidence"].size() >= 2, "pi run chat evidence");
        expect_equal(run_chat["resume_recommendation"]["from_phase"].get<std::string>(),
                     "COMMON_OVERLAP", "pi run chat resume phase recommendation");
        expect_true(run_chat["resume_recommendation"].contains("execution_note"),
                    "pi run chat resume explains full rerun phase");
        expect_true(run_chat["action_plan_validation"]["valid"].get<bool>(), "pi run chat action plan valid");

        const auto run_chat_followup = harness.post_json("/api/pi/run-chat", {
            {"run_id", "pi_fixture_run"},
            {"message", "Welche dieser Empfehlungen war in deiner vorherigen Antwort am wichtigsten?"}
        });
        expect_equal(run_chat_followup["_http_status"].get<long>(), 200L, "pi run chat followup status");
        expect_true(run_chat_followup["context"]["conversation"]["previous_turn_count"].get<long>() >= 1,
                    "pi run chat followup uses persisted previous turns");
        expect_true(run_chat_followup["summary"].get<std::string>().find("Folgefrage") != std::string::npos,
                    "pi run chat followup summary marks continuation");
        expect_true(run_chat_followup["action_plan"]["suppressed_repeated_actions"].is_array() &&
                    !run_chat_followup["action_plan"]["suppressed_repeated_actions"].empty(),
                    "pi run chat followup suppresses repeated parameter suggestions");

        const auto run_chat_no_improvement = harness.post_json("/api/pi/run-chat", {
            {"run_id", "pi_fixture_run"},
            {"message", "Das Ergebnis ist gleich geblieben, keine Verbesserung. Was nun?"}
        });
        expect_equal(run_chat_no_improvement["_http_status"].get<long>(), 200L, "pi run chat negative learning status");
        expect_true(run_chat_no_improvement.contains("negative_learning"),
                    "pi run chat records negative learning when repeated suggestions did not help");
        expect_true(!run_chat_no_improvement["negative_learning"].value("memory_id", std::string()).empty(),
                    "pi run chat negative learning returns memory id");

        const auto saved_chat_history = harness.post_json("/api/pi/run-chat/history", {
            {"run_id", "pi_fixture_run"},
            {"history", {
                {"messages", nlohmann::json::array({{{"role", "user"}, {"content", "Was tun?"}}})},
                {"turns", nlohmann::json::array({{{"message", "Was tun?"}, {"result", run_chat}}})}
            }}
        });
        expect_equal(saved_chat_history["_http_status"].get<long>(), 200L, "pi run chat history save status");
        expect_true(static_cast<long>(saved_chat_history["turns"].size()) >= 2L, "pi run chat history saved turn merge");
        const auto loaded_chat_history = harness.get_json("/api/pi/run-chat/history?run_id=pi_fixture_run");
        expect_equal(loaded_chat_history["_http_status"].get<long>(), 200L, "pi run chat history load status");
        expect_true(static_cast<long>(loaded_chat_history["turns"].size()) >= 2L, "pi run chat history loaded merged turns");

        const auto storage_default = harness.get_json("/api/pi/storage");
        expect_equal(storage_default["_http_status"].get<long>(), 200L, "pi storage default status");
        expect_equal(storage_default["schema_version"].get<std::string>(), "pi.storage.v1", "pi storage schema");
        expect_true(!storage_default["configured"].get<bool>(), "pi storage initially uses default");

        const auto memory_dir = harness.fixture_root() / "pi_custom_storage";
        const auto storage_saved = harness.post_json("/api/pi/storage", {
            {"storage_dir", memory_dir.string()}
        });
        expect_equal(storage_saved["_http_status"].get<long>(), 200L, "pi storage save status");
        expect_true(storage_saved["configured"].get<bool>(), "pi storage marked configured");
        expect_equal(storage_saved["storage_dir"].get<std::string>(), memory_dir.string(), "pi storage saved path");
        std::filesystem::create_directories(memory_dir);
        {
            std::ofstream legacy(memory_dir / "memories.jsonl");
            legacy << nlohmann::json{
                {"schema_version", "pi.memory.v1"},
                {"memory_id", "legacy_must_be_ignored"},
                {"status", "candidate"},
                {"type", "optimization"}
            }.dump() << "\n";
            std::ofstream out(memory_dir / "memories_v2.jsonl");
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_route_fixture"},
                {"id", "mem_route_fixture"},
                {"status", "candidate"},
                {"type", "optimization"},
                {"privacy_class", "metadata_only"},
                {"context_signature", {
                    {"schema_version", "pi.context_signature.v1"},
                    {"target", {{"object_name", "M42"}, {"object_type", "nebula"}}},
                    {"acquisition", {{"camera_name", "ASI2600MC"}, {"camera_type", "OSC"}, {"filters", nlohmann::json::array({"HaOIII"})}}},
                    {"pipeline", {{"affected_paths", nlohmann::json::array({"data.color_mode"})}}},
                    {"problem", {{"classes", nlohmann::json::array({"faint_nebula"})}}}
                }},
                {"scope", {
                    {"applies_when", nlohmann::json::array({"matching context"})},
                    {"does_not_apply_when", nlohmann::json::array({"different target class"})},
                    {"confidence", 0.5}
                }},
                {"recommendation", {{"patch", nlohmann::json::array({{{"path", "data.color_mode"}, {"value", "MONO"}}})}}},
                {"evidence", {{"validation", "fixture"}}},
                {"outcome", {{"validation_valid", true}, {"applied_count", 1}}},
                {"review", {{"status", "candidate"}, {"reviewed_by", nullptr}, {"reviewed_at", nullptr}, {"notes", ""}}},
                {"retrieval", {{"keywords", nlohmann::json::array({"data.color_mode"})}, {"negative", false}}}
            }.dump() << "\n";
        }
        const auto memories = harness.get_json("/api/pi/memories?limit=20");
        expect_equal(memories["_http_status"].get<long>(), 200L, "pi memories list status");
        expect_equal(memories["schema_version"].get<std::string>(), "pi.memories-list.v1", "pi memories list schema");
        expect_equal(static_cast<long>(memories["items"].size()), 1L, "pi memories list count");
        expect_equal(memories["items"][0]["memory_id"].get<std::string>(), "mem_route_fixture",
                     "pi memories ignores legacy v1 store");

        const auto memory_review = harness.post_json("/api/pi/memories/mem_route_fixture/review", {
            {"status", "accepted"},
            {"reviewer", "fixture"},
            {"note", "useful"},
            {"outcome", {{"validation_valid", true}, {"report_status", "ok"}}},
            {"scope", {
                {"applies_when", nlohmann::json::array({"M42 OSC HaOIII"})},
                {"does_not_apply_when", nlohmann::json::array({"galaxy mono"})},
                {"confidence", 0.8}
            }}
        });
        expect_equal(memory_review["_http_status"].get<long>(), 200L, "pi memory review status");
        expect_true(memory_review["ok"].get<bool>(), "pi memory review ok");
        expect_true(memory_review["review"]["outcome"]["validation_valid"].get<bool>(),
                    "pi memory review stores outcome metadata");
        expect_equal(memory_review["review"]["scope"]["applies_when"][0].get<std::string>(),
                     "M42 OSC HaOIII",
                     "pi memory review stores scope metadata");

        const auto accepted_memories = harness.get_json("/api/pi/memories?status=accepted");
        expect_equal(accepted_memories["_http_status"].get<long>(), 200L, "pi accepted memories status");
        expect_equal(static_cast<long>(accepted_memories["items"].size()), 1L, "pi accepted memories count");
        const auto memory_index = harness.get_json("/api/pi/memories/index");
        expect_equal(memory_index["_http_status"].get<long>(), 200L, "pi memory index status");
        expect_equal(memory_index["schema_version"].get<std::string>(), "pi.memory-indices.v2",
                     "pi memory index schema");
        expect_true(memory_index["by_target"]["m42"].is_array(), "pi memory index by target");
        expect_true(memory_index["by_camera"]["asi2600mc"].is_array(), "pi memory index by camera");
        expect_true(memory_index["by_filter"]["haoiii"].is_array(), "pi memory index by filter");
        expect_true(memory_index["by_problem"]["faint_nebula"].is_array(), "pi memory index by problem");

        const auto exported_memories = harness.get_json("/api/pi/memories/export?privacy=metadata_only");
        expect_equal(exported_memories["_http_status"].get<long>(), 200L, "pi memories export status");
        expect_equal(exported_memories["schema_version"].get<std::string>(), "pi.memories-export.v2",
                     "pi memories export schema");
        expect_equal(static_cast<long>(exported_memories["memory_count"].get<size_t>()), 1L,
                     "pi memories export count");

        const auto import_preview = harness.post_json("/api/pi/memories/import", {
            {"bundle", exported_memories},
            {"dry_run", true}
        });
        expect_equal(import_preview["_http_status"].get<long>(), 200L, "pi memories import dry-run status");
        expect_true(import_preview["dry_run"].get<bool>(), "pi memories import dry-run flag");

        const auto dedupe_preview = harness.post_json("/api/pi/memories/dedupe", {
            {"dry_run", true}
        });
        expect_equal(dedupe_preview["_http_status"].get<long>(), 200L, "pi memories dedupe dry-run status");
        expect_true(dedupe_preview["dry_run"].get<bool>(), "pi memories dedupe dry-run flag");

        const auto retrieved_memories = harness.post_json("/api/pi/memories/retrieve", {
            {"query", {
                {"type", "optimization"},
                {"config_updates", nlohmann::json::array({{{"path", "data.color_mode"}, {"value", "MONO"}}})}
            }},
            {"limit", 5}
        });
        expect_equal(retrieved_memories["_http_status"].get<long>(), 200L, "pi memory retrieve status");
        expect_equal(retrieved_memories["schema_version"].get<std::string>(), "pi.memory-retrieval.v2",
                     "pi memory retrieval schema");
        expect_equal(static_cast<long>(retrieved_memories["matches"].size()), 1L, "pi memory retrieve match count");
        expect_true(retrieved_memories.contains("warnings"), "pi memory retrieval includes warnings");

        const auto memory_outcome = harness.post_json("/api/pi/memories/mem_route_fixture/outcome", {
            {"result", "no_improvement"},
            {"feedback", "same output after resume"},
            {"before", {{"quality_score", 0.60}}},
            {"after", {{"quality_score", 0.60}}},
            {"outcome", {
                {"api_key", "secret_should_not_persist"},
                {"preview_path", "/media/private/stacked_rgb_hms.fits"}
            }}
        });
        expect_equal(memory_outcome["_http_status"].get<long>(), 200L, "pi memory outcome evaluator status");
        expect_equal(memory_outcome["outcome"]["verdict"].get<std::string>(), "unchanged",
                     "pi memory outcome evaluator detects unchanged result");
        expect_equal(memory_outcome["review"]["status"].get<std::string>(), "rejected",
                     "pi memory outcome evaluator recommends rejected for unchanged result");
        expect_equal(memory_outcome["review"]["outcome"]["api_key"].get<std::string>(), "<redacted>",
                     "pi memory outcome redacts api keys");
        expect_equal(memory_outcome["review"]["outcome"]["preview_path"]["redacted"].get<std::string>(), "absolute_path",
                     "pi memory outcome redacts absolute paths");

        const auto rejected_memories = harness.get_json("/api/pi/memories?status=rejected");
        expect_equal(rejected_memories["_http_status"].get<long>(), 200L, "pi rejected memories status");
        expect_true(static_cast<long>(rejected_memories["items"].size()) >= 1L, "pi rejected memories include outcome/negative learning");

        const nlohmann::json plan = {
            {"schema_version", "pi.action-plan.v1"},
            {"goal", "fixture preview"},
            {"actions", nlohmann::json::array({
                {
                    {"id", "set_color"},
                    {"type", "config.set"},
                    {"path", "data.color_mode"},
                    {"value", "MONO"},
                    {"rationale", "fixture"}
                }
            })},
            {"post_conditions", nlohmann::json::array()}
        };
        const auto plan_validation = harness.post_json("/api/pi/action-plans/validate", {{"plan", plan}});
        expect_equal(plan_validation["_http_status"].get<long>(), 200L, "pi action validate status");
        expect_true(plan_validation["valid"].get<bool>(), "pi action validate ok");

        const auto preview = harness.post_json("/api/pi/action-plans/preview", {{"plan", plan}});
        expect_equal(preview["_http_status"].get<long>(), 200L, "pi action preview status");
        expect_true(preview["ok"].get<bool>(), "pi action preview ok");
        expect_true(preview["preview"]["mutation_free"].get<bool>(), "pi action preview mutation free");
        expect_equal(static_cast<long>(preview["preview"]["config_updates"].size()), 1L, "pi action preview update count");
        expect_equal(preview["preview"]["config_updates"][0]["path"].get<std::string>(), "data.color_mode",
                     "pi action preview update path");
        expect_equal(preview["preview"]["patched_config"]["data"]["color_mode"].get<std::string>(), "MONO",
                     "pi action preview patched config");
        expect_true(preview["preview"]["yaml_changed"].get<bool>(), "pi action preview yaml changed");
        expect_true(preview["preview"]["config_valid"].get<bool>(), "pi action preview config valid");

        const auto unconfirmed_apply = harness.post_json("/api/pi/action-plans/apply", {{"plan", plan}});
        expect_equal(unconfirmed_apply["_http_status"].get<long>(), 409L, "pi action apply requires review");

        const auto apply = harness.post_json("/api/pi/action-plans/apply", {
            {"plan", plan},
            {"confirmed", true},
            {"expected_patched_yaml", preview["preview"]["patched_yaml"]}
        });
        expect_equal(apply["_http_status"].get<long>(), 200L, "pi action apply status");
        expect_true(apply["ok"].get<bool>(), "pi action apply ok");
        expect_true(apply.contains("revision_id"), "pi action apply revision id");
        expect_true(slurp_file(harness.config_path()).find("MONO") != std::string::npos,
                    "pi action apply wrote config");

        const auto audit = harness.get_json("/api/pi/audit?limit=200");
        expect_equal(audit["_http_status"].get<long>(), 200L, "pi audit status");
        expect_equal(audit["schema_version"].get<std::string>(), "pi.audit.v1", "pi audit schema");
        bool found_action_apply_audit = false;
        bool found_memory_review_audit = false;
        for (const auto& item : audit["items"]) {
            if (item.value("audit_type", std::string()) == "pi_event" &&
                item.value("event", std::string()) == "pi.action_plan.apply") {
                found_action_apply_audit = true;
            }
            if (item.value("audit_type", std::string()) == "memory_review" &&
                item.value("memory_id", std::string()) == "mem_route_fixture") {
                found_memory_review_audit = true;
            }
        }
        expect_true(found_action_apply_audit, "pi audit includes action-plan apply event");
        expect_true(found_memory_review_audit, "pi audit includes memory review");

        const auto revisions = harness.get_json("/api/config/revisions");
        expect_equal(revisions["_http_status"].get<long>(), 200L, "pi action revisions status");
        expect_true(!revisions["items"].empty(), "pi action revision listed");
        expect_equal(revisions["active_revision_id"].get<std::string>(), apply["revision_id"].get<std::string>(),
                     "pi action active revision id");

        auto invalid_plan = plan;
        invalid_plan["actions"][0].erase("value");
        const auto invalid_preview = harness.post_json("/api/pi/action-plans/preview", {{"plan", invalid_plan}});
        expect_equal(invalid_preview["_http_status"].get<long>(), 400L, "pi invalid action preview status");
        expect_true(!invalid_preview["ok"].get<bool>(), "pi invalid action preview not ok");

        auto invalid_config_plan = plan;
        invalid_config_plan["actions"][0]["path"] = "invalid";
        invalid_config_plan["actions"][0]["value"] = true;
        const auto invalid_config_preview = harness.post_json("/api/pi/action-plans/preview", {
            {"plan", invalid_config_plan},
            {"base_config", {{"data", {{"color_mode", "OSC"}}}}}
        });
        expect_equal(invalid_config_preview["_http_status"].get<long>(), 200L, "pi invalid config preview status");
        expect_true(!invalid_config_preview["preview"]["config_valid"].get<bool>(), "pi invalid config preview validation");

        // --- Outcome-Delta-Test ---
        // evaluate_memory_outcome_payload muss FWHM-Delta, Report-Warnings-Delta
        // und user_rating verarbeiten und das Verdict korrekt ableiten.
        {
            // Memory fuer Outcome-Delta-Tests anlegen
            std::ofstream delta_out(memory_dir / "memories_v2.jsonl", std::ios::app);
            const nlohmann::json delta_ctx = {
                {"schema_version", "pi.context_signature.v1"},
                {"target", {{"object_name", "M31"}, {"object_type", "galaxy"}}},
                {"acquisition", {{"camera_type", "MONO"}}},
                {"pipeline", {{"affected_paths", nlohmann::json::array({"bge.enabled"})}}}
            };
            delta_out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_outcome_delta_fixture"},
                {"id", "mem_outcome_delta_fixture"},
                {"status", "candidate"},
                {"type", "config_optimization"},
                {"source", "scan_ai_apply"},
                {"privacy_class", "metadata_only"},
                {"context_signature", delta_ctx},
                {"scope", {
                    {"applies_when", nlohmann::json::array({"galaxy MONO BGE context"})},
                    {"does_not_apply_when", nlohmann::json::array({"OSC context"})},
                    {"confidence", 0.5}
                }},
                {"recommendation", {{"explanation", "delta test fixture"}}},
                {"evidence", {{"validation", "fixture"}, {"run_id", "pi_fixture_run"}}},
                {"outcome", {{"validation_valid", true}, {"applied_count", 1}}},
                {"review", {{"status", "candidate"}, {"reviewed_by", nullptr}, {"reviewed_at", nullptr}, {"notes", ""}}},
                {"retrieval", {{"keywords", nlohmann::json::array({"bge.enabled"})}, {"negative", false}}}
            }.dump() << "\n";
        }

        // Outcome mit FWHM-Verbesserung: soll verdict=improved ergeben und promotable setzen
        const auto outcome_improved = harness.post_json("/api/pi/memories/mem_outcome_delta_fixture/outcome", {
            {"before", {{"fwhm_median", 3.2}, {"report_warnings", nlohmann::json::array({"faint_nebula"})}}},
            {"after",  {{"fwhm_median", 2.5}, {"report_warnings", nlohmann::json::array()},
                         {"artifact_status", {{"bge.json", "ok"}}}}},
            {"feedback", "Sterne sehen schoerfer aus"}
        });
        expect_equal(outcome_improved["_http_status"].get<long>(), 200L, "outcome-delta: improved outcome status");
        expect_equal(outcome_improved["outcome"]["verdict"].get<std::string>(), "improved",
                     "outcome-delta: FWHM improvement detected");
        expect_true(outcome_improved["outcome"].contains("fwhm_before"),
                    "outcome-delta: fwhm_before field present");
        expect_true(outcome_improved["outcome"].contains("fwhm_after"),
                    "outcome-delta: fwhm_after field present");
        expect_true(outcome_improved["outcome"].contains("report_warnings_delta"),
                    "outcome-delta: report_warnings_delta field present");
        expect_equal(outcome_improved["outcome"]["report_warnings_delta"].get<long>(), -1L,
                     "outcome-delta: one fewer warning after");
        expect_true(outcome_improved["outcome"].contains("artifact_status"),
                    "outcome-delta: artifact_status preserved in outcome");
        expect_equal(outcome_improved["review"]["status"].get<std::string>(), "promotable",
                     "outcome-delta: improved verdict sets promotable status");
        // Kein counterexample bei verbessertem Ergebnis
        expect_true(!outcome_improved.contains("counterexample") ||
                     outcome_improved["counterexample"].is_null(),
                     "outcome-delta: no counterexample created for improved verdict");

        // Outcome mit Verschlechterung: soll verdict=worse + counterexample anlegen
        const auto outcome_worse = harness.post_json("/api/pi/memories/mem_outcome_delta_fixture/outcome", {
            {"result", "worse"},
            {"feedback", "Hintergrund jetzt heller"},
            {"before", {{"quality_score", 0.70}}},
            {"after",  {{"quality_score", 0.55}}},
            {"negative_learning", true}
        });
        expect_equal(outcome_worse["_http_status"].get<long>(), 200L, "outcome-delta: worse outcome status");
        expect_equal(outcome_worse["outcome"]["verdict"].get<std::string>(), "worse",
                     "outcome-delta: worse verdict");
        expect_true(outcome_worse["outcome"].contains("quality_score_delta"),
                    "outcome-delta: quality_score_delta present");
        expect_true(outcome_worse["outcome"]["quality_score_delta"].get<double>() < 0.0,
                    "outcome-delta: negative score delta");
        expect_equal(outcome_worse["review"]["status"].get<std::string>(), "rejected",
                     "outcome-delta: worse verdict sets rejected status");
        expect_true(outcome_worse.contains("counterexample") &&
                    !outcome_worse["counterexample"].is_null() &&
                    !outcome_worse["counterexample"].value("memory_id", std::string()).empty(),
                    "outcome-delta: counterexample created for worse verdict");
        expect_equal(outcome_worse["counterexample"]["type"].get<std::string>(), "counterexample",
                     "outcome-delta: counterexample has correct type");

        // User-Rating ueberschreibt quantitatives Verdict
        const auto outcome_user_rated = harness.post_json("/api/pi/memories/mem_outcome_delta_fixture/outcome", {
            {"user_rating", 5},
            {"negative_learning", false}
        });
        expect_equal(outcome_user_rated["_http_status"].get<long>(), 200L, "outcome-delta: user rating status");
        expect_equal(outcome_user_rated["outcome"]["verdict"].get<std::string>(), "improved",
                     "outcome-delta: user rating 5 overrides to improved");

        // --- Promotable → accepted (promote endpoint) ---
        // mem_outcome_delta_fixture ist nach outcome_improved promotable
        const auto promote = harness.post_json("/api/pi/memories/mem_outcome_delta_fixture/promote", {
            {"reviewer", "fixture"},
            {"note", "manually promoted after FWHM improvement confirmed"}
        });
        expect_equal(promote["_http_status"].get<long>(), 200L, "promote: status");
        expect_true(promote["ok"].get<bool>(), "promote: ok flag");
        expect_equal(promote["review"]["status"].get<std::string>(), "accepted",
                     "promote: memory is now accepted");

        // Nicht-promotable Memory kann nicht promoted werden
        const auto promote_non_promotable = harness.post_json("/api/pi/memories/mem_route_fixture/promote", {
            {"reviewer", "fixture"}
        });
        expect_equal(promote_non_promotable["_http_status"].get<long>(), 400L,
                     "promote: non-promotable memory returns 400");

        // --- Post-Run-Trigger (evaluate-run) ---
        // Ein stats.json mit FWHM anlegen damit evaluate_run Metriken lesen kann
        harness.make_file("runs/pi_fixture_run/artifacts/stats.json",
            "{\"phase_issues\":[\"faint_emission\"],\"summary\":{\"status\":\"ok\"},"
            "\"aggregate\":{\"fwhm\":{\"median\":2.8}}}\n");

        // Memory mit run_id-Provenance anlegen
        {
            std::ofstream run_out(memory_dir / "memories_v2.jsonl", std::ios::app);
            run_out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_postrun_fixture"},
                {"id", "mem_postrun_fixture"},
                {"status", "candidate"},
                {"type", "config_optimization"},
                {"source", "scan_ai_apply"},
                {"privacy_class", "metadata_only"},
                {"context_signature", {
                    {"schema_version", "pi.context_signature.v1"},
                    {"target", {{"object_name", "M42"}}},
                    {"acquisition", {{"camera_type", "OSC"}}},
                    {"pipeline", {{"affected_paths", nlohmann::json::array({"bge.enabled"})}}}
                }},
                {"scope", {
                    {"applies_when", nlohmann::json::array({"BGE optimization context"})},
                    {"does_not_apply_when", nlohmann::json::array({"galaxy targets"})},
                    {"confidence", 0.5}
                }},
                {"recommendation", {{"explanation", "post-run trigger fixture"}}},
                {"evidence", {{"validation", "fixture"}, {"run_id", "pi_fixture_run"}}},
                {"outcome", {{"validation_valid", true}, {"applied_count", 1}}},
                {"review", {{"status", "candidate"}, {"reviewed_by", nullptr}, {"reviewed_at", nullptr}, {"notes", ""}}},
                {"retrieval", {{"keywords", nlohmann::json::array({"bge.enabled"})}, {"negative", false}}}
            }.dump() << "\n";
        }

        const auto eval_run = harness.post_json("/api/pi/memories/evaluate-run", {
            {"run_id", "pi_fixture_run"},
            {"result", "improved"},
            {"feedback", "BGE improvement visible"}
        });
        expect_equal(eval_run["_http_status"].get<long>(), 200L, "post-run-trigger: status");
        expect_true(eval_run["ok"].get<bool>(), "post-run-trigger: ok flag");
        expect_equal(eval_run["run_id"].get<std::string>(), "pi_fixture_run",
                     "post-run-trigger: run_id echoed");
        expect_true(eval_run["updated"].get<long>() >= 1L,
                    "post-run-trigger: at least one memory updated");
        expect_true(eval_run["run_metrics"].contains("artifact_status"),
                    "post-run-trigger: run_metrics includes artifact_status");
        expect_true(eval_run["run_metrics"].contains("report_warnings"),
                    "post-run-trigger: run_metrics includes report_warnings from stats.json");

        // Ungueltige run_id
        const auto eval_run_invalid = harness.post_json("/api/pi/memories/evaluate-run", {
            {"run_id", "nonexistent_run_xyz"}
        });
        expect_equal(eval_run_invalid["_http_status"].get<long>(), 404L,
                     "post-run-trigger: nonexistent run returns 404");

        // --- resume_feedback Endpoint ---
        const auto resume_fb = harness.post_json("/api/pi/memories/resume-feedback", {
            {"run_id", "pi_fixture_run"},
            {"from_phase", "BGE"},
            {"result", "improved"},
            {"feedback", "Resuming from BGE reduced gradient artifact"},
            {"context_signature", {
                {"schema_version", "pi.context_signature.v1"},
                {"target", {{"object_name", "M42"}}},
                {"acquisition", {{"camera_type", "OSC"}}},
                {"pipeline", {{"resume_phase", "BGE"}, {"phases", nlohmann::json::array({"BGE"})}}}
            }}
        });
        expect_equal(resume_fb["_http_status"].get<long>(), 200L, "resume-feedback: status");
        expect_true(resume_fb["ok"].get<bool>(), "resume-feedback: ok flag");
        expect_equal(resume_fb["from_phase"].get<std::string>(), "BGE",
                     "resume-feedback: from_phase echoed");
        expect_equal(resume_fb["memory"]["source"].get<std::string>(), "resume_feedback",
                     "resume-feedback: memory source is resume_feedback");
        expect_equal(resume_fb["memory"]["type"].get<std::string>(), "resume_strategy",
                     "resume-feedback: improved feedback creates resume_strategy memory");
        expect_true(resume_fb["outcome"]["verdict"].get<std::string>() == "improved",
                    "resume-feedback: verdict is improved");

        // Negatives Resume-Feedback erzeugt counterexample
        const auto resume_fb_neg = harness.post_json("/api/pi/memories/resume-feedback", {
            {"run_id", "pi_fixture_run"},
            {"from_phase", "PCC"},
            {"result", "worse"},
            {"feedback", "PCC resume made colors worse"}
        });
        expect_equal(resume_fb_neg["_http_status"].get<long>(), 200L, "resume-feedback neg: status");
        expect_equal(resume_fb_neg["memory"]["type"].get<std::string>(), "counterexample",
                     "resume-feedback neg: worse feedback creates counterexample memory");
        expect_equal(resume_fb_neg["memory"]["source"].get<std::string>(), "resume_feedback",
                     "resume-feedback neg: source is resume_feedback");

        // Fehlende Pflichtfelder
        const auto resume_fb_no_phase = harness.post_json("/api/pi/memories/resume-feedback", {
            {"run_id", "pi_fixture_run"},
            {"feedback", "something"}
        });
        expect_equal(resume_fb_no_phase["_http_status"].get<long>(), 400L,
                     "resume-feedback: missing from_phase returns 400");

        // ===== Live Image Chat Tests =====

        // Create a run with a valid FITS output
        const auto live_run_dir = harness.fixture_root() / "runs" / "live_image_test";
        std::filesystem::create_directories(live_run_dir / "outputs");
        std::filesystem::create_directories(live_run_dir / "artifacts");
        create_minimal_fits(live_run_dir / "outputs" / "stacked_rgb_hms.fits", 64, 64);

        // Write run_events.jsonl so the backend recognizes this as a run
        {
            std::ofstream events(live_run_dir / "artifacts" / "run_events.jsonl");
            events << nlohmann::json({
                {"ts", "2026-07-22T12:00:00Z"},
                {"type", "run_start"},
                {"run_id", "live_image_test"},
                {"config", {{"general", {{"color_mode", "OSC"}}}}}
            }).dump() << "\n";
            events << nlohmann::json({
                {"ts", "2026-07-22T12:05:00Z"},
                {"type", "run_end"},
                {"success", true}
            }).dump() << "\n";
        }

        // 1. Create session
        const auto create_resp = harness.post_json("/api/pi/live-image-chat/create", {
            {"run_id", "live_image_test"}
        });
        expect_equal(create_resp["_http_status"].get<long>(), 200L,
                     "live-image-chat create status");
        expect_true(create_resp.contains("session_id"), "live-image-chat create has session_id");
        expect_true(!create_resp["session_id"].get<std::string>().empty(),
                     "live-image-chat create session_id not empty");
        expect_true(create_resp.contains("image_base64"), "live-image-chat create has image_base64");
        expect_true(!create_resp["image_base64"].get<std::string>().empty(),
                     "live-image-chat create image_base64 not empty");
        const std::string session_id = create_resp["session_id"].get<std::string>();

        // 2. Create with invalid run_id
        const auto create_bad = harness.post_json("/api/pi/live-image-chat/create", {
            {"run_id", "nonexistent_run_12345"}
        });
        expect_equal(create_bad["_http_status"].get<long>(), 404L,
                     "live-image-chat create invalid run_id returns 404");

        // 3. Chat with valid session (local fallback - no sidecar)
        const auto chat_resp = harness.post_json("/api/pi/live-image-chat", {
            {"session_id", session_id},
            {"message", "macher heller"}
        });
        expect_equal(chat_resp["_http_status"].get<long>(), 200L,
                     "live-image-chat chat status");
        expect_true(chat_resp.contains("summary"), "live-image-chat chat has summary");
        expect_true(chat_resp.contains("operations"), "live-image-chat chat has operations");
        expect_true(chat_resp.contains("image_base64"), "live-image-chat chat has image_base64");
        expect_true(!chat_resp["image_base64"].get<std::string>().empty(),
                     "live-image-chat chat image_base64 not empty");

        // 4. Chat with invalid session
        const auto chat_bad = harness.post_json("/api/pi/live-image-chat", {
            {"session_id", "invalid_session_id"},
            {"message", "test"}
        });
        expect_equal(chat_bad["_http_status"].get<long>(), 404L,
                     "live-image-chat chat invalid session returns 404");

        // 5. Adjust
        const auto adjust_resp = harness.post_json("/api/pi/live-image-chat/adjust", {
            {"session_id", session_id},
            {"direction", "increase"}
        });
        expect_equal(adjust_resp["_http_status"].get<long>(), 200L,
                     "live-image-chat adjust status");
        expect_true(adjust_resp.contains("image_base64"), "live-image-chat adjust has image_base64");
        expect_true(adjust_resp.contains("adjust_count"), "live-image-chat adjust has adjust_count");

        // 6. Undo
        const auto undo_resp = harness.post_json("/api/pi/live-image-chat/undo", {
            {"session_id", session_id}
        });
        expect_equal(undo_resp["_http_status"].get<long>(), 200L,
                     "live-image-chat undo status");
        expect_true(undo_resp.contains("image_base64"), "live-image-chat undo has image_base64");
        expect_true(undo_resp.contains("can_undo"), "live-image-chat undo has can_undo");
        expect_true(undo_resp.contains("can_redo"), "live-image-chat undo has can_redo");

        // 7. Redo
        const auto redo_resp = harness.post_json("/api/pi/live-image-chat/redo", {
            {"session_id", session_id}
        });
        expect_equal(redo_resp["_http_status"].get<long>(), 200L,
                     "live-image-chat redo status");
        expect_true(redo_resp.contains("image_base64"), "live-image-chat redo has image_base64");

        // 8. Export PNG
        const auto export_resp = harness.post_json("/api/pi/live-image-chat/export", {
            {"session_id", session_id},
            {"format", "png"}
        });
        expect_equal(export_resp["_http_status"].get<long>(), 200L,
                     "live-image-chat export png status");
        expect_true(export_resp["ok"].get<bool>(), "live-image-chat export png ok");
        expect_true(export_resp.contains("path"), "live-image-chat export has path");
        expect_true(std::filesystem::exists(export_resp["path"].get<std::string>()),
                     "live-image-chat export file exists");

        // 9. History endpoint
        const auto history_resp = harness.get_json(
            "/api/pi/live-image-chat/history?run_id=live_image_test");
        expect_equal(history_resp["_http_status"].get<long>(), 200L,
                     "live-image-chat history status");
        expect_true(history_resp.contains("chat_history"), "live-image-chat history has chat_history");
        expect_true(history_resp.contains("operation_history"),
                     "live-image-chat history has operation_history");

        // 10. Close session
        const auto close_resp = harness.post_json("/api/pi/live-image-chat/close", {
            {"session_id", session_id}
        });
        expect_equal(close_resp["_http_status"].get<long>(), 200L,
                     "live-image-chat close status");
        expect_true(close_resp["ok"].get<bool>(), "live-image-chat close ok");

        // 11. Verify session is gone after close
        const auto chat_after_close = harness.post_json("/api/pi/live-image-chat", {
            {"session_id", session_id},
            {"message", "test"}
        });
        expect_equal(chat_after_close["_http_status"].get<long>(), 404L,
                     "live-image-chat chat after close returns 404");

        // 12. Create again - should resume with history
        const auto create_resume = harness.post_json("/api/pi/live-image-chat/create", {
            {"run_id", "live_image_test"}
        });
        expect_equal(create_resume["_http_status"].get<long>(), 200L,
                     "live-image-chat create resume status");
        expect_true(create_resume.value("resumed", false),
                     "live-image-chat create resume flag is true");
        expect_true(create_resume.contains("chat_history"),
                     "live-image-chat create resume has chat_history");
        const std::string resumed_session_id = create_resume["session_id"].get<std::string>();

        // 13. Reset on resumed session
        const auto reset_resp = harness.post_json("/api/pi/live-image-chat/reset", {
            {"session_id", resumed_session_id}
        });
        expect_equal(reset_resp["_http_status"].get<long>(), 200L,
                     "live-image-chat reset status");
        expect_true(reset_resp.contains("image_base64"), "live-image-chat reset has image_base64");
        expect_equal(static_cast<long>(reset_resp["can_undo"].get<bool>()), 0L,
                     "live-image-chat reset can_undo is false");
        expect_equal(static_cast<long>(reset_resp["can_redo"].get<bool>()), 0L,
                     "live-image-chat reset can_redo is false");

    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
