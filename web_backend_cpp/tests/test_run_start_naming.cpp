#include "backend_test_harness.hpp"
#include "services/pi/pi_jev_store.hpp"

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
        tile_compile::pi::PiJevStore jev_store(decisions_dir);
        jev_store.put("p_queue", "source", {
            {"input_path", input_dir}, {"dataset_manifest", manifest}
        });
        jev_store.put("p_queue", "proposal", {
            {"status", "applied_to_draft"}, {"applied_at", "2020-01-01T00:00:00Z"},
            {"saved_revision_ids", nlohmann::json::array({"saved_cfg"})},
            {"updates", nlohmann::json::array({{{"path", "data.color_mode"}, {"value", "OSC"}}})}
        });
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
        expect_true(std::regex_match(linked_provenance["run_uid"].get<std::string>(), std::regex("run_[0-9a-f]{32}")),
                    "new queue run has stable UID");
        const auto linked_outcome = *jev_store.get("p_queue", "outcome");
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
        const auto generated_run_dir = harness.fixture_root() / "runs" / generated_run_id;
        const auto provenance_text = slurp_file(generated_run_dir / "artifacts" / "pi_run_provenance.json");
        const auto provenance_uid = nlohmann::json::parse(provenance_text)["run_uid"].get<std::string>();
        const auto activated = harness.post_json("/api/runs/" + generated_run_id + "/set-current", {});
        expect_equal(activated["run_uid"].get<std::string>(), provenance_uid, "activation uses provenance UID");
        const auto aliased = harness.post_json("/api/runs/path_alias/set-current", {{"run_dir", generated_run_dir.string()}});
        expect_equal(aliased["run_uid"].get<std::string>(), provenance_uid, "path hint shares exact run identity");
        const auto active_context = harness.get_json("/api/pi/active-context");
        expect_equal(active_context["context"]["run_uid"].get<std::string>(), provenance_uid, "active context endpoint");
        const auto ui_state = nlohmann::json::parse(slurp_file(harness.fixture_root() / "runtime" / "ui_state.json"));
        expect_equal(ui_state["pi_active_context"]["run_uid"].get<std::string>(), provenance_uid, "active context persisted");
        expect_equal(slurp_file(generated_run_dir / "artifacts" / "pi_run_provenance.json"), provenance_text,
                     "activation does not rewrite run provenance");
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
        // Preserve learning data before deleting run outputs; raw lights remain external.
        harness.make_file("runs/" + generated_run_id + "/artifacts/stats.json", "{\"frames\":1,\"noise\":0.12}");
        const auto memory_root = harness.fixture_root() / "runs" / ".pi_memory";
        auto db = tile_compile::pi::PiDatabase::open(memory_root);
        db->execute("CREATE TRIGGER fail_learning_capture BEFORE INSERT ON run_learning_snapshots BEGIN SELECT RAISE(ABORT, 'fixture storage failure'); END");
        const auto blocked_delete = harness.post_json("/api/runs/" + generated_run_id + "/delete", {});
        expect_equal(blocked_delete["_http_status"].get<long>(), 503L, "failed archive blocks file deletion");
        expect_true(std::filesystem::exists(generated_run_dir / "config.yaml"), "run files kept after failed snapshot");
        db->execute("DROP TRIGGER fail_learning_capture");
        const auto deleted = harness.post_json("/api/runs/" + generated_run_id + "/delete", {});
        expect_equal(deleted["_http_status"].get<long>(), 200L, "delete run with learning retained");
        expect_true(deleted["learning_retained"].get<bool>(), "delete confirms learning retained");
        expect_equal(deleted["run_uid"].get<std::string>(), provenance_uid, "delete keeps stable UID");
        expect_true(!std::filesystem::exists(generated_run_dir), "run outputs removed");
        expect_true(std::filesystem::exists(input_file), "raw light remains on disk");
        const auto archived = harness.get_json("/api/pi/run-learning/" + provenance_uid);
        expect_equal(archived["_http_status"].get<long>(), 200L, "deleted run archive readable");
        expect_equal(archived["artifacts_state"].get<std::string>(), "deleted", "deleted artifacts marker");
        expect_true(!archived["excluded_from_learning"].get<bool>(), "file deletion is not learning rejection");
        expect_equal(archived["artifacts"]["artifacts/stats.json"]["data"]["frames"].get<long>(), 1L, "statistics retained");
        expect_true(archived["config"]["yaml"].is_string(), "effective config retained");
        expect_equal(archived["source"]["original_input_dir"].get<std::string>(), input_dir, "raw input origin retained");
        const auto history = harness.get_json("/api/pi/run-learning/" + provenance_uid + "/history");
        expect_true(history["items"].size() >= 2, "start and later snapshots retained");
        expect_equal(harness.get_json("/api/pi/run-learning/" + provenance_uid + "/preview")["_http_status"].get<long>(), 404L,
                     "optional preview absent without rendering enabled");
        const auto excluded = harness.post_json("/api/pi/run-learning/" + provenance_uid + "/exclusion", {
            {"confirmed", true}, {"excluded", true}, {"reason_code", "test_run"}
        });
        expect_equal(excluded["_http_status"].get<long>(), 200L, "learning exclusion is separate explicit action");
        expect_true(harness.get_json("/api/pi/run-learning/" + provenance_uid)["excluded_from_learning"].get<bool>(),
                    "archive exclusion persisted without erasing measurements");

        harness.make_file("runs/raw_inside/config.yaml", "data:\n  color_mode: OSC\n");
        const auto inner_run = harness.fixture_root() / "runs" / "raw_inside";
        harness.make_file("runs/raw_inside/lights/raw.fit", "RAW");
        harness.make_file("runs/raw_inside/artifacts/pi_run_provenance.json", nlohmann::json{
            {"original_input_dir", (inner_run / "lights").string()}
        }.dump());
        const auto refused_raw_delete = harness.post_json("/api/runs/raw_inside/delete", {});
        expect_equal(refused_raw_delete["_http_status"].get<long>(), 409L, "raw files inside run block output deletion");
        expect_true(std::filesystem::exists(inner_run / "lights" / "raw.fit"), "inner raw light protected");
        const auto safe_raw_path = harness.fixture_root() / "raw_moved.fit";
        std::filesystem::rename(inner_run / "lights" / "raw.fit", safe_raw_path);
        const auto after_raw_move = harness.post_json("/api/runs/raw_inside/delete", {});
        expect_equal(after_raw_move["_http_status"].get<long>(), 200L, "deletion allowed after raw source moved out");
        expect_true(std::filesystem::exists(safe_raw_path), "moved raw source retained on disk");
        harness.make_file("runs/incomplete_archive/config.yaml", "data:\n  color_mode: OSC\n");
        harness.make_file("runs/incomplete_archive/artifacts/stats.json", "{broken");
        const auto incomplete_delete = harness.post_json("/api/runs/incomplete_archive/delete", {});
        expect_equal(incomplete_delete["_http_status"].get<long>(), 409L, "invalid statistics block unconfirmed incomplete archive");
        expect_true(std::filesystem::exists(harness.fixture_root() / "runs" / "incomplete_archive" / "config.yaml"),
                    "run kept for incomplete archive decision");
        const auto acknowledged = harness.post_json("/api/runs/incomplete_archive/delete", {{"allow_incomplete_snapshot", true}});
        expect_equal(acknowledged["_http_status"].get<long>(), 200L, "explicit incomplete archive acknowledgement");
        expect_true(!acknowledged["capture_issues"].empty(), "capture gaps remain visible after acknowledged deletion");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
