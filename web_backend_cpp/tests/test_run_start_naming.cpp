#include "backend_test_harness.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        harness.make_file("inputs/session_1/frame_0001.fit", "fixture\n");

        const std::string input_dir = (harness.fixture_root() / "inputs" / "session_1").string();

        const auto stale_jev = harness.post_json("/api/runs/start", {
            {"input_dir", input_dir}, {"run_id", "rejected_jev"},
            {"color_mode", "OSC"}, {"config_yaml", "data:\n  color_mode: OSC\n"},
            {"jev_saved_proposal_id", "missing_proposal"}
        });
        expect_equal(stale_jev["_http_status"].get<long>(), 409L, "stale Jev link blocks direct run");
        expect_equal(stale_jev["error"]["code"].get<std::string>(), "JEV_PROPOSAL_STALE", "stale Jev error code");
        expect_true(!std::filesystem::exists(harness.fixture_root() / "runs" / "rejected_jev"),
                    "stale Jev link creates no run directory");

        const auto queued_jev = harness.post_json("/api/runs/start", {
            {"runs_dir", (harness.fixture_root() / "runs").string()},
            {"run_id", "rejected_jev_queue"}, {"color_mode", "OSC"},
            {"config_yaml", "data:\n  color_mode: OSC\n"},
            {"jev_saved_proposal_id", "missing_proposal"},
            {"queue", nlohmann::json::array({{{"input_dir", input_dir}, {"filter", "L"}}})}
        });
        expect_equal(queued_jev["_http_status"].get<long>(), 409L, "Jev link blocks unsupported queue run");
        expect_equal(queued_jev["error"]["code"].get<std::string>(), "JEV_PROPOSAL_STALE", "missing queue Jev proposal error code");
        expect_true(!std::filesystem::exists(harness.fixture_root() / "runs" / "rejected_jev_queue"),
                    "unsupported Jev queue creates no run directory");

        const std::filesystem::path input_file = std::filesystem::path(input_dir) / "frame_0001.fit";
        const std::filesystem::path decisions_dir = harness.fixture_root() / "runs" / ".pi_memory" / "pi_decisions";
        nlohmann::json manifest = nlohmann::json::array({{
            {"id", "frame_0001.fit"},
            {"size", std::filesystem::file_size(input_file)},
            {"mtime", std::filesystem::last_write_time(input_file).time_since_epoch().count()}
        }});
        std::filesystem::create_directories(decisions_dir / "p_queue");
        std::ofstream(decisions_dir / "p_queue" / "source.json") << nlohmann::json{
            {"input_path", input_dir}, {"dataset_manifest", manifest}
        }.dump();
        std::ofstream(decisions_dir / "p_queue" / "proposal.json") << nlohmann::json{
            {"status", "applied_to_draft"}, {"applied_at", "2020-01-01T00:00:00Z"},
            {"saved_revision_ids", nlohmann::json::array({"saved_cfg"})},
            {"updates", nlohmann::json::array({{{"path", "data.color_mode"}, {"value", "OSC"}}})}
        }.dump();
        const auto partial_queue = harness.post_json("/api/runs/start", {
            {"runs_dir", (harness.fixture_root() / "runs").string()},
            {"run_id", "partial_jev_queue"}, {"color_mode", "OSC"},
            {"config_yaml", "data:\n  color_mode: OSC\n"},
            {"jev_saved_proposal_id", "p_queue"},
            {"queue", nlohmann::json::array({{{"input_dir", input_dir}, {"filter", "L"}, {"pattern", "other*"}}})}
        });
        expect_equal(partial_queue["_http_status"].get<long>(), 409L, "queue excluding scanned FITS is rejected");
        expect_equal(partial_queue["error"]["code"].get<std::string>(), "JEV_PROPOSAL_UNSUPPORTED", "partial queue Jev error code");
        expect_true(!std::filesystem::exists(harness.fixture_root() / "runs" / "partial_jev_queue"),
                    "partial Jev queue creates no run directory");

        const auto linked_queue = harness.post_json("/api/runs/start", {
            {"runs_dir", (harness.fixture_root() / "runs").string()},
            {"run_id", "linked_jev_queue"}, {"color_mode", "OSC"},
            {"config_yaml", "data:\n  color_mode: OSC\n"},
            {"jev_saved_proposal_id", "p_queue"},
            {"queue", nlohmann::json::array({{{"input_dir", input_dir}, {"filter", "L"}, {"pattern", "*.fit"}}})}
        });
        expect_equal(linked_queue["_http_status"].get<long>(), 202L, "queue with exact scanned FITS starts");
        const auto linked_queue_job = harness.wait_for_job(linked_queue["job_id"].get<std::string>());
        expect_equal(linked_queue_job["state"].get<std::string>(), "ok", "linked Jev queue completes");
        const std::filesystem::path linked_run = harness.fixture_root() / "runs" / "linked_jev_queue" / "L";
        const auto linked_provenance = nlohmann::json::parse(slurp_file(linked_run / "artifacts" / "pi_run_provenance.json"));
        expect_equal(linked_provenance["jev_proposal_id"].get<std::string>(), "p_queue", "queue run carries checked Jev id");
        const auto linked_outcome = nlohmann::json::parse(slurp_file(decisions_dir / "p_queue" / "outcome.json"));
        expect_equal(linked_outcome["runs"][0]["run_id"].get<std::string>(), "linked_jev_queue/L",
                     "queue outcome records nested run id");

        const auto started = harness.post_json("/api/runs/start", {
            {"input_dir", input_dir},
            {"run_name", "M42 Test"},
            {"color_mode", "OSC"},
            {"config_yaml", "data:\n  color_mode: OSC\n"}
        });
        expect_equal(started["_http_status"].get<long>(), 202L, "run start status");
        expect_json_field(started, "job_id", "run start job id");
        expect_json_field(started, "run_id", "run start run id");

        const std::string generated_run_id = started["run_id"].get<std::string>();
        expect_true(
            std::regex_match(generated_run_id, std::regex(R"(^M42_Test_[0-9]{8}_[0-9]{6}$)")),
            "generated run id should include sanitized run_name and timestamp");

        const auto immediate_status = harness.get_json("/api/runs/" + generated_run_id + "/status");
        expect_equal(immediate_status["_http_status"].get<long>(), 200L, "immediate run status");
        expect_equal(immediate_status["run_id"].get<std::string>(), generated_run_id, "immediate status run id");
        expect_true(
            immediate_status["status"].get<std::string>() == "running" ||
            immediate_status["status"].get<std::string>() == "pending",
            "immediate status should expose pending or running state");

        const auto immediate_artifacts = harness.get_json("/api/runs/" + generated_run_id + "/artifacts");
        expect_equal(immediate_artifacts["_http_status"].get<long>(), 200L, "immediate artifacts status");
        expect_true(immediate_artifacts["items"].is_array(), "immediate artifacts items array");

        const auto job = harness.wait_for_job(started["job_id"].get<std::string>());
        expect_equal(job["run_id"].get<std::string>(), generated_run_id, "job run id matches generated run id");
        expect_true(job["data"]["command"].is_array(), "job command is array");
        expect_equal(job["data"]["command"][1].get<std::string>(), "reconstruct",
                     "job command uses reconstruct");

        std::string config_arg;
        for (size_t i = 0; i + 1 < job["data"]["command"].size(); ++i) {
            if (job["data"]["command"][i].get<std::string>() == "--config") {
                config_arg = job["data"]["command"][i + 1].get<std::string>();
            }
        }
        expect_true(!config_arg.empty(), "runner args include --config path");
        {
            std::ifstream snapshot(config_arg);
            std::ostringstream snapshot_text;
            snapshot_text << snapshot.rdbuf();
            YAML::Node snapshot_root = YAML::Load(snapshot_text.str());
            expect_true(!snapshot_root["method"],
                        "generated config snapshot has no top-level method");
        }

        bool found_run_id_flag = false;
        bool found_generated_run_id = false;
        for (const auto& arg : job["data"]["command"]) {
            const std::string value = arg.get<std::string>();
            if (value == "--run-id") found_run_id_flag = true;
            if (value == generated_run_id) found_generated_run_id = true;
        }
        expect_true(found_run_id_flag, "runner args include --run-id");
        expect_true(found_generated_run_id, "runner args include generated run id");

        const auto explicit_id = harness.post_json("/api/runs/start", {
            {"input_dir", input_dir},
            {"run_name", "ignored-name"},
            {"run_id", "manual_id"},
            {"color_mode", "OSC"}
        });
        expect_equal(explicit_id["_http_status"].get<long>(), 202L, "explicit run id start status");
        expect_equal(explicit_id["run_id"].get<std::string>(), "manual_id", "explicit run id should be preserved");

        const auto queued = harness.post_json("/api/runs/start", {
            {"runs_dir", (harness.fixture_root() / "runs").string()},
            {"color_mode", "OSC"},
            {"queue", nlohmann::json::array({
                {
                    {"input_dir", input_dir},
                    {"filter", "L"}
                },
                {
                    {"input_dir", input_dir},
                    {"filter", "L"}
                },
                {
                    {"input_dir", input_dir},
                    {"filter", "R"}
                }
            })}
        });
        expect_equal(queued["_http_status"].get<long>(), 202L, "queued run start status");
        const std::string queued_run_id = queued["run_id"].get<std::string>();
        expect_true(
            std::regex_match(queued_run_id, std::regex(R"(^[0-9]{8}_[0-9]{4}/L$)")),
            "queued run id should use start date+time root and first filter leaf");

        const auto queue_job = harness.wait_for_job(queued["job_id"].get<std::string>());
        expect_true(queue_job["data"]["queue"].is_array(), "queue payload should expose queue items");
        expect_equal(queue_job["data"]["queue"][0]["run_id"].get<std::string>(), queued_run_id, "first queue item run id");
        expect_true(
            std::regex_match(queue_job["data"]["queue"][1]["run_id"].get<std::string>(), std::regex(R"(^[0-9]{8}_[0-9]{4}/L-2$)")),
            "duplicate filter should receive -2 suffix");
        expect_true(
            std::regex_match(queue_job["data"]["queue"][2]["run_id"].get<std::string>(), std::regex(R"(^[0-9]{8}_[0-9]{4}/R$)")),
            "different filter should use own leaf without numeric suffix");

        const auto queued_named = harness.post_json("/api/runs/start", {
            {"runs_dir", (harness.fixture_root() / "runs").string()},
            {"run_name", "M66 Batch"},
            {"color_mode", "OSC"},
            {"queue", nlohmann::json::array({
                {
                    {"input_dir", input_dir},
                    {"filter", "HA"}
                },
                {
                    {"input_dir", input_dir},
                    {"filter", "OIII"}
                }
            })}
        });
        expect_equal(queued_named["_http_status"].get<long>(), 202L, "named queued run start status");
        const std::string queued_named_run_id = queued_named["run_id"].get<std::string>();
        expect_true(
            std::regex_match(queued_named_run_id, std::regex(R"(^M66_Batch_[0-9]{8}_[0-9]{6}/HA$)")),
            "queued run id with run_name should keep run_name_timestamp root");

        const auto queued_named_job = harness.wait_for_job(queued_named["job_id"].get<std::string>());
        expect_true(
            std::regex_match(queued_named_job["data"]["queue"][1]["run_id"].get<std::string>(), std::regex(R"(^M66_Batch_[0-9]{8}_[0-9]{6}/OIII$)")),
            "named queue should keep run_name_timestamp root for all filters");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
