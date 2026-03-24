#include "backend_test_harness.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

std::string encode_base64url(const std::string& input) {
    static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int val = 0;
    int valb = -6;
    for (unsigned char ch : input) {
        val = (val << 8) + ch;
        valb += 8;
        while (valb >= 0) {
            out.push_back(alphabet[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(alphabet[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4 != 0) out.push_back('=');
    for (char& ch : out) {
        if (ch == '+') ch = '-';
        else if (ch == '/') ch = '_';
    }
    while (!out.empty() && out.back() == '=') out.pop_back();
    return "b64_" + out;
}

}

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    ::setenv("FAKE_TILE_COMPILE_RUNNER_SLEEP_MS", "500", 1);
    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        harness.make_file("inputs/session_1/frame_0001.fit", "fixture\n");
        const std::string input_dir = (harness.fixture_root() / "inputs" / "session_1").string();

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
                    {"filter", "R"}
                }
            })}
        });
        expect_equal(queued["_http_status"].get<long>(), 202L, "queued run start status");
        expect_json_field(queued, "job_id", "queued run job id");
        expect_json_field(queued, "run_id", "queued run first run id");

        const std::string first_run_id = queued["run_id"].get<std::string>();
        const auto slash = first_run_id.find('/');
        expect_true(slash != std::string::npos, "queued child run id should contain root and leaf");
        const std::string root_run_id = first_run_id.substr(0, slash);

        const auto root_status = harness.get_json("/api/runs/" + root_run_id + "/status");
        expect_equal(root_status["_http_status"].get<long>(), 200L, "queue root status should be readable");
        expect_true(root_status["queue"].is_array(), "queue root status should expose queue items");
        expect_equal(static_cast<long>(root_status["queue"].size()), 2L, "queue root queue size");
        expect_equal(root_status["queue"][0]["run_id"].get<std::string>(), first_run_id, "queue root first child run id");

        const std::string job_id = queued["job_id"].get<std::string>();
        bool second_batch_running = false;
        nlohmann::json queue_job = nlohmann::json::object();
        for (int attempt = 0; attempt < 60; ++attempt) {
            queue_job = harness.get_json("/api/jobs/" + job_id);
            if (queue_job["data"].is_object() &&
                queue_job["data"].value("current_index", -1) == 1 &&
                queue_job["data"].contains("queue") &&
                queue_job["data"]["queue"].is_array() &&
                queue_job["data"]["queue"].size() > 1 &&
                queue_job["data"]["queue"][1].value("state", std::string()) == "running") {
                second_batch_running = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        expect_true(second_batch_running, "queue should advance to second batch while job is running");

        const auto aliased_status = harness.get_json("/api/runs/" + encode_base64url(first_run_id) + "/status");
        expect_equal(aliased_status["_http_status"].get<long>(), 200L, "first child status should stay readable while queue advances");
        expect_equal(aliased_status["status"].get<std::string>(), "running", "first child status should follow active queue state");
        expect_true(aliased_status["queue"].is_array(), "aliased status should keep queue payload");
        expect_equal(static_cast<long>(aliased_status["queue"].size()), 2L, "aliased queue size");
        expect_equal(aliased_status["queue"][1]["state"].get<std::string>(), "running", "second batch should be marked running");

        const auto final_job = harness.wait_for_job(job_id);
        expect_equal(final_job["state"].get<std::string>(), "ok", "queue job should finish successfully");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
