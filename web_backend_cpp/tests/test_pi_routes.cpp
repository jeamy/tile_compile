#include "backend_test_harness.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

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

        const auto memory_dir = harness.fixture_root() / "runs" / ".pi_memory";
        std::filesystem::create_directories(memory_dir);
        {
            std::ofstream out(memory_dir / "memories.jsonl");
            out << nlohmann::json{
                {"schema_version", "pi.memory.v1"},
                {"memory_id", "mem_route_fixture"},
                {"status", "candidate"},
                {"type", "optimization"},
                {"privacy_class", "metadata_only"},
                {"recommendation", {{"patch", nlohmann::json::array({{{"path", "data.color_mode"}, {"value", "MONO"}}})}}},
                {"evidence", {{"validation", "fixture"}}}
            }.dump() << "\n";
        }
        const auto memories = harness.get_json("/api/pi/memories?limit=20");
        expect_equal(memories["_http_status"].get<long>(), 200L, "pi memories list status");
        expect_equal(memories["schema_version"].get<std::string>(), "pi.memories-list.v1", "pi memories list schema");
        expect_equal(static_cast<long>(memories["items"].size()), 1L, "pi memories list count");

        const auto memory_review = harness.post_json("/api/pi/memories/mem_route_fixture/review", {
            {"status", "accepted"},
            {"reviewer", "fixture"},
            {"note", "useful"},
            {"outcome", {{"validation_valid", true}, {"report_status", "ok"}}}
        });
        expect_equal(memory_review["_http_status"].get<long>(), 200L, "pi memory review status");
        expect_true(memory_review["ok"].get<bool>(), "pi memory review ok");
        expect_true(memory_review["review"]["outcome"]["validation_valid"].get<bool>(),
                    "pi memory review stores outcome metadata");

        const auto accepted_memories = harness.get_json("/api/pi/memories?status=accepted");
        expect_equal(accepted_memories["_http_status"].get<long>(), 200L, "pi accepted memories status");
        expect_equal(static_cast<long>(accepted_memories["items"].size()), 1L, "pi accepted memories count");
        expect_equal(accepted_memories["items"][0]["status"].get<std::string>(), "accepted",
                     "pi accepted memory status overlay");

        const auto exported_memories = harness.get_json("/api/pi/memories/export?privacy=metadata_only");
        expect_equal(exported_memories["_http_status"].get<long>(), 200L, "pi memories export status");
        expect_equal(exported_memories["schema_version"].get<std::string>(), "pi.memories-export.v1",
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
        expect_equal(retrieved_memories["schema_version"].get<std::string>(), "pi.memory-retrieval.v1",
                     "pi memory retrieval schema");
        expect_equal(static_cast<long>(retrieved_memories["matches"].size()), 1L, "pi memory retrieve match count");

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
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
