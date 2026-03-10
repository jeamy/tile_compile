#include "backend_test_harness.hpp"

#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        harness.create_run("sample_run", {
            {{"ts", "2026-03-10T10:00:00Z"}, {"type", "phase_start"}, {"phase_name", "SCAN_INPUT"}},
            {{"ts", "2026-03-10T10:00:01Z"}, {"type", "phase_end"}, {"phase_name", "SCAN_INPUT"}, {"status", "ok"}},
            {{"ts", "2026-03-10T10:00:02Z"}, {"type", "phase_start"}, {"phase_name", "ASTROMETRY"}},
            {{"ts", "2026-03-10T10:00:03Z"}, {"type", "phase_progress"}, {"phase_name", "ASTROMETRY"}, {"progress", 0.5}}
        }, "MONO");

        auto status = harness.get_json("/api/runs/sample_run/status");
        expect_equal(status["_http_status"].get<long>(), 200L, "run status code");
        expect_equal(status["run_id"].get<std::string>(), "sample_run", "run id");
        expect_equal(status["status"].get<std::string>(), "running", "run status");
        expect_equal(status["color_mode"].get<std::string>(), "MONO", "run color mode");
        expect_equal(status["current_phase"].get<std::string>(), "ASTROMETRY", "current phase");
        expect_true(status["phases"].is_array(), "phases array");
        expect_true(status["events"].is_array(), "events array");
        expect_true(status["queue_filters"].is_array(), "queue filters array");
        bool found_astrometry = false;
        for (const auto& item : status["phases"]) {
            if (item.value("phase", "") == "ASTROMETRY") {
                found_astrometry = true;
                expect_equal(item["status"].get<std::string>(), "running", "astrometry status");
                expect_equal(item["pct"].get<double>(), 0.5, "astrometry pct", 1e-6);
            }
        }
        expect_true(found_astrometry, "astrometry phase present");

        harness.create_run("set_current_run", {
            {{"ts", "2026-03-10T10:10:00Z"}, {"type", "phase_start"}, {"phase_name", "SCAN_INPUT"}}
        }, "OSC");
        const auto set_current = harness.post_json("/api/runs/set_current_run/set-current", nlohmann::json::object());
        expect_equal(set_current["_http_status"].get<long>(), 200L, "set current status");
        expect_true(set_current["ok"].get<bool>(), "set current ok");

        const auto app_state = harness.get_json("/api/app/state");
        expect_equal(app_state["_http_status"].get<long>(), 200L, "app state status");
        expect_equal(app_state["project"]["current_run_id"].get<std::string>(), "set_current_run", "project current run");
        expect_equal(app_state["run"]["current"]["run_id"].get<std::string>(), "set_current_run", "run current run");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
