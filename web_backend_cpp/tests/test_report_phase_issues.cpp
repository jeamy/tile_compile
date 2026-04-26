#include "backend_test_harness.hpp"

#include <cstdlib>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    setenv("TILE_COMPILE_BACKEND_REPORT_EVENTS_MAX", "512", 1);
    setenv("TILE_COMPILE_BACKEND_REPORT_JSON_FILE_BYTES", "1048576", 1);
    setenv("TILE_COMPILE_BACKEND_RETAINED_JOBS", "16", 1);

    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        const std::vector<nlohmann::json> events = {
            {{"ts", "2026-04-26T08:20:00Z"}, {"type", "run_start"}, {"run_id", "bge_skipped"}, {"frames_discovered", 10}},
            {{"ts", "2026-04-26T08:20:01Z"}, {"type", "phase_start"}, {"phase_name", "BGE"}},
            {
                {"ts", "2026-04-26T08:20:20Z"},
                {"type", "phase_end"},
                {"phase_name", "BGE"},
                {"status", "skipped"},
                {"reason", "surface_fit_failed"},
                {"requested", true},
                {"attempted", true},
                {"success", false}
            },
            {{"ts", "2026-04-26T08:20:21Z"}, {"type", "run_end"}, {"status", "ok"}}
        };
        const fs::path run_dir = harness.create_run("bge_skipped", events);

        const nlohmann::json bge = {
            {"requested", true},
            {"attempted", true},
            {"success", false},
            {"failure_reason", "surface_fit_failed"},
            {"config", {
                {"min_valid_sample_fraction_for_apply", 0.30},
                {"min_valid_samples_for_apply", 96}
            }},
            {"summary", {
                {"channels_applied", 0},
                {"channels_total", 3},
                {"channels_fit_success", 0},
                {"tile_samples_valid", 444},
                {"tile_samples_total", 1980}
            }},
            {"channels", {
                {
                    {"channel", "R"},
                    {"applied", false},
                    {"fit_success", false},
                    {"tile_samples_valid", 149},
                    {"tile_samples_total", 660}
                },
                {
                    {"channel", "G"},
                    {"applied", false},
                    {"fit_success", false},
                    {"tile_samples_valid", 146},
                    {"tile_samples_total", 660}
                }
            }}
        };
        harness.make_file("runs/bge_skipped/artifacts/bge.json", bge.dump(2));

        const auto stats_job = harness.post_json("/api/runs/bge_skipped/stats", {{"run_dir", run_dir.string()}});
        expect_equal(stats_job["_http_status"].get<long>(), 202L, "stats job accepted");
        const auto done = harness.wait_for_job(stats_job["job_id"].get<std::string>(), 20.0);
        expect_equal(done["state"].get<std::string>(), "ok", "stats job completed");

        const std::string report = slurp_file(run_dir / "artifacts" / "report.html");
        expect_true(report.find("data-report-lang=\"de\"") != std::string::npos, "german language switch rendered");
        expect_true(report.find("data-report-lang=\"en\"") != std::string::npos, "english language switch rendered");
        expect_true(report.find("tileCompileReportSetLanguage") != std::string::npos, "language switch script rendered");
        expect_true(report.find("const templates=") != std::string::npos, "language switch embeds rendered language templates");
        expect_true(report.find("Phase Issues Summary") != std::string::npos, "phase issue section rendered");
        expect_true(report.find("surface_fit_failed") != std::string::npos, "phase reason rendered");
        expect_true(report.find("BGE artifact summary") != std::string::npos, "bge artifact summary rendered");
        expect_true(report.find("149/660") != std::string::npos, "bge channel sample detail rendered");
        expect_true(report.find("30%") != std::string::npos, "bge apply guard rendered");

        harness.stop();
        return 0;
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
