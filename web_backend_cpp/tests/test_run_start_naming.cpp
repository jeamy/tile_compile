#include "backend_test_harness.hpp"

#include <cstdio>
#include <regex>

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        harness.make_file("inputs/session_1/frame_0001.fit", "fixture\n");

        const std::string input_dir = (harness.fixture_root() / "inputs" / "session_1").string();

        const auto started = harness.post_json("/api/runs/start", {
            {"input_dir", input_dir},
            {"run_name", "M42 Test"},
            {"color_mode", "OSC"}
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
