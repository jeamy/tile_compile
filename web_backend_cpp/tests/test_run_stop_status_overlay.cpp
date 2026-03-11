#include "backend_test_harness.hpp"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    ::setenv("FAKE_TILE_COMPILE_RUNNER_SLEEP_MS", "1500", 1);

    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        harness.create_run(
            "stop_overlay_run",
            {
                {{"ts", "2026-03-11T12:00:00Z"}, {"type", "phase_start"}, {"phase_name", "REGISTRATION"}},
                {{"ts", "2026-03-11T12:00:02Z"}, {"type", "phase_progress"}, {"phase_name", "REGISTRATION"}, {"progress", 0.42}},
            },
            "OSC");

        harness.make_file("inputs/stop_overlay/frame_0001.fit", "fixture\n");
        const std::string input_dir = (harness.fixture_root() / "inputs" / "stop_overlay").string();

        const auto started = harness.post_json("/api/runs/start", {
            {"input_dir", input_dir},
            {"run_id", "stop_overlay_run"},
            {"color_mode", "OSC"}
        });
        expect_equal(started["_http_status"].get<long>(), 202L, "run start status");
        expect_equal(started["run_id"].get<std::string>(), "stop_overlay_run", "explicit run id");

        const auto stopped = harness.post_json("/api/runs/stop_overlay_run/stop", nlohmann::json::object());
        expect_equal(stopped["_http_status"].get<long>(), 200L, "stop status");
        expect_true(stopped["ok"].get<bool>(), "stop should acknowledge cancellation");

        const auto job = harness.wait_for_job(started["job_id"].get<std::string>(), 5.0);
        expect_equal(job["state"].get<std::string>(), "cancelled", "stopped job state");

        const auto status = harness.get_json("/api/runs/stop_overlay_run/status");
        expect_equal(status["_http_status"].get<long>(), 200L, "run status status");
        expect_equal(status["status"].get<std::string>(), "cancelled", "run status should reflect cancelled job over stale event log");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
