#include "backend_test_harness.hpp"

#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        harness.create_run("resume_progress_run", {
            {{"ts", "2026-03-10T10:00:00Z"}, {"type", "phase_start"}, {"phase_name", "ASTROMETRY"}},
            {{"ts", "2026-03-10T10:00:10Z"}, {"type", "phase_end"}, {"phase_name", "ASTROMETRY"}, {"status", "ok"}},
            {{"ts", "2026-03-10T10:00:20Z"}, {"type", "phase_start"}, {"phase_name", "BGE"}},
            {{"ts", "2026-03-10T10:00:30Z"}, {"type", "phase_end"}, {"phase_name", "BGE"}, {"status", "ok"}},
            {{"ts", "2026-03-10T10:01:00Z"}, {"type", "resume_start"}, {"from_phase", "BGE"}},
            {{"ts", "2026-03-10T10:01:01Z"}, {"type", "phase_start"}, {"phase_name", "BGE"}}
        }, "OSC");

        const auto status = harness.get_json("/api/runs/resume_progress_run/status");
        expect_equal(status["_http_status"].get<long>(), 200L, "resume status code");
        expect_equal(status["status"].get<std::string>(), "running", "resume run status");
        expect_equal(status["current_phase"].get<std::string>(), "BGE", "resume current phase");
        bool found_bge = false;
        for (const auto& item : status["phases"]) {
            if (item.value("phase", "") == "BGE") {
                found_bge = true;
                expect_equal(item["status"].get<std::string>(), "running", "bge resumed status");
                expect_equal(item["pct"].get<double>(), 0.0, "bge resumed pct", 1e-9);
            }
        }
        expect_true(found_bge, "bge phase present");

        harness.create_run("skipped_phase_run", {
            {{"ts", "2026-03-10T11:00:00Z"}, {"type", "phase_start"}, {"phase_name", "STATE_CLUSTERING"}},
            {{"ts", "2026-03-10T11:00:01Z"}, {"type", "phase_end"}, {"phase_name", "STATE_CLUSTERING"}, {"status", "skipped"}},
            {{"ts", "2026-03-10T11:00:02Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");

        const auto skipped_status = harness.get_json("/api/runs/skipped_phase_run/status");
        expect_equal(skipped_status["_http_status"].get<long>(), 200L, "skipped status code");
        bool found_skipped = false;
        for (const auto& item : skipped_status["phases"]) {
            if (item.value("phase", "") == "STATE_CLUSTERING") {
                found_skipped = true;
                expect_equal(item["status"].get<std::string>(), "skipped", "state clustering skipped status");
                expect_equal(item["pct"].get<double>(), 1.0, "state clustering skipped pct", 1e-9);
            }
        }
        expect_true(found_skipped, "state clustering phase present");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
