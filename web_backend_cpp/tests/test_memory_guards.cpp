#include "backend_test_harness.hpp"

#include <cstdlib>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

std::string repeated(char ch, size_t count) {
    return std::string(count, ch);
}

}

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    setenv("TILE_COMPILE_BACKEND_SUBPROCESS_CAPTURE_BYTES", "1048576", 1);
    setenv("TILE_COMPILE_BACKEND_JOB_STDIO_STORE_BYTES", "16384", 1);
    setenv("TILE_COMPILE_BACKEND_SCAN_FRAMES_PREVIEW", "20", 1);
    setenv("TILE_COMPILE_BACKEND_SCAN_PER_DIR_FRAMES_PREVIEW", "5", 1);
    setenv("TILE_COMPILE_BACKEND_SCAN_PER_DIR_RESULTS_PREVIEW", "1", 1);
    setenv("TILE_COMPILE_BACKEND_SCAN_MESSAGES_PREVIEW", "16", 1);
    setenv("TILE_COMPILE_BACKEND_SCAN_COLOR_CANDIDATES_PREVIEW", "2", 1);
    setenv("TILE_COMPILE_BACKEND_REPORT_EVENTS_MAX", "256", 1);
    setenv("TILE_COMPILE_BACKEND_REPORT_LOG_TAIL", "32", 1);
    setenv("TILE_COMPILE_BACKEND_REPORT_TEXT_BYTES", "65536", 1);
    setenv("TILE_COMPILE_BACKEND_REPORT_JSON_FILE_BYTES", "1048576", 1);
    setenv("TILE_COMPILE_BACKEND_RETAINED_JOBS", "32", 1);
    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        std::vector<nlohmann::json> events = {
            {{"ts", "2026-03-10T10:00:00Z"}, {"type", "run_start"}, {"run_id", "heavy_run"}, {"input_dir", "/tmp/heavy"}, {"frames_discovered", 2000}},
            {{"ts", "2026-03-10T10:00:01Z"}, {"type", "phase_start"}, {"phase_name", "SCAN_INPUT"}}
        };
        for (int i = 0; i < 1500; ++i) {
            events.push_back({
                {"ts", "2026-03-10T10:00:02Z"},
                {"type", "phase_progress"},
                {"phase_name", "SCAN_INPUT"},
                {"progress", static_cast<double>(i % 100) / 100.0}
            });
        }
        events.push_back({{"ts", "2026-03-10T10:10:00Z"}, {"type", "phase_end"}, {"phase_name", "SCAN_INPUT"}, {"status", "ok"}});
        events.push_back({{"ts", "2026-03-10T10:10:01Z"}, {"type", "run_end"}, {"success", true}, {"status", "ok"}});

        const fs::path run_dir = harness.create_run("heavy_run", events);
        harness.make_file("runs/heavy_run/artifacts/normalization.json", repeated('x', 5 * 1024 * 1024));

        const auto status = harness.get_json("/api/runs/heavy_run/status");
        expect_equal(status["_http_status"].get<long>(), 200L, "heavy run status response");
        expect_true(status["events"].is_array(), "heavy run status events array");
        expect_equal(static_cast<long>(status["events"].size()), 200L, "heavy run status event tail capped");

        const auto app_state = harness.get_json("/api/app/state");
        expect_equal(app_state["_http_status"].get<long>(), 200L, "app state response");
        expect_true(app_state["history"]["recent"].is_array(), "app state recent runs array");

        const auto stats_job = harness.post_json("/api/runs/heavy_run/stats", nlohmann::json::object());
        expect_equal(stats_job["_http_status"].get<long>(), 202L, "stats job accepted");
        const auto stats_done = harness.wait_for_job(stats_job["job_id"].get<std::string>(), 20.0);
        expect_equal(stats_done["state"].get<std::string>(), "ok", "stats job completed");
        expect_true(stats_done["data"]["ok"].get<bool>(), "stats payload ok");

        harness.make_file("input_many_frames_a/.keep", "");
        harness.make_file("input_many_frames_b/.keep", "");

        const auto multi_scan = harness.post_json("/api/scan", {
            {"input_dirs", {
                (harness.fixture_root() / "input_many_frames_a").string(),
                (harness.fixture_root() / "input_many_frames_b").string()
            }},
            {"frames_min", 1}
        });
        expect_equal(multi_scan["_http_status"].get<long>(), 200L, "multi scan accepted");
        const auto multi_scan_done = harness.wait_for_job(multi_scan["job_id"].get<std::string>());
        expect_equal(multi_scan_done["state"].get<std::string>(), "ok", "multi scan completed");
        const auto& multi_result = multi_scan_done["data"]["result"];
        expect_true(multi_result["frames_truncated"].get<bool>(), "multi scan frames truncated");
        expect_equal(static_cast<long>(multi_result["frames"].size()), 20L, "multi scan frames preview limit applied");
        expect_true(multi_result["frames_total"].get<long long>() > static_cast<long long>(multi_result["frames"].size()), "multi scan frames total preserved");
        expect_equal(multi_result["per_dir_results_total"].get<long>(), 2L, "multi scan per-dir count");
        expect_equal(static_cast<long>(multi_result["per_dir_results"].size()), 1L, "multi scan per-dir preview limit applied");
        expect_true(multi_result["per_dir_results"][0]["frames_truncated"].get<bool>(), "multi scan per-dir frames truncated");
        expect_equal(static_cast<long>(multi_result["per_dir_results"][0]["frames"].size()), 5L, "multi scan per-dir frames preview limit applied");
        expect_equal(static_cast<long>(multi_result["per_dir_results"][0]["color_mode_candidates"].size()), 2L, "multi scan color candidates preview limit applied");

        const auto latest_scan = harness.get_json("/api/scan/latest");
        expect_equal(latest_scan["_http_status"].get<long>(), 200L, "latest scan response");
        expect_true(latest_scan["frames_truncated"].get<bool>(), "latest scan exposes truncation");

        const auto single_scan = harness.post_json("/api/scan", {
            {"input_dir", (harness.fixture_root() / "input_many_frames_a").string()},
            {"frames_min", 1}
        });
        expect_equal(single_scan["_http_status"].get<long>(), 200L, "single scan accepted");
        const auto single_scan_done = harness.wait_for_job(single_scan["job_id"].get<std::string>());
        expect_equal(single_scan_done["state"].get<std::string>(), "ok", "single scan completed");
        expect_true(single_scan_done["data"]["result"]["frames_truncated"].get<bool>(), "single scan result compacted");

        const fs::path input_rgb = harness.make_binary_file("images/input_rgb.fit", "rgb");
        const fs::path wcs_file = harness.make_file("images/input_rgb.wcs", "WCSAXES = 2\n");
        const fs::path output_rgb = harness.fixture_root() / "images" / "output_loud_widejson.fit";
        const auto pcc_run = harness.post_json("/api/tools/pcc/run", {
            {"input_rgb", input_rgb.string()},
            {"output_rgb", output_rgb.string()},
            {"wcs_file", wcs_file.string()}
        });
        expect_equal(pcc_run["_http_status"].get<long>(), 202L, "pcc run accepted");
        const auto pcc_done = harness.wait_for_job(pcc_run["job_id"].get<std::string>(), 20.0);
        expect_equal(pcc_done["state"].get<std::string>(), "ok", "pcc run completed");
        expect_true(pcc_done["data"]["stdout_truncated"].get<bool>(), "stdout stored with truncation");
        expect_true(pcc_done["data"]["stderr_truncated"].get<bool>(), "stderr stored with truncation");
        expect_true(pcc_done["data"]["stderr"].get<std::string>().size() <= 16384 + 32, "stderr storage limit applied");
        expect_true(pcc_done["data"]["result"]["debug_blob"].get<std::string>().size() < 200000, "json result compacted");

        for (int i = 0; i < 135; ++i) {
            const fs::path prune_output = harness.fixture_root() / "images" / ("prune_" + std::to_string(i) + ".fit");
            const auto job = harness.post_json("/api/tools/pcc/run", {
                {"input_rgb", input_rgb.string()},
                {"output_rgb", prune_output.string()},
                {"wcs_file", wcs_file.string()}
            });
            expect_equal(job["_http_status"].get<long>(), 202L, "prune pcc run accepted");
            const auto done = harness.wait_for_job(job["job_id"].get<std::string>(), 20.0);
            expect_equal(done["state"].get<std::string>(), "ok", "prune pcc run completed");
        }

        const auto jobs = harness.get_json("/api/jobs?limit=200");
        expect_equal(jobs["_http_status"].get<long>(), 200L, "jobs endpoint response");
        expect_true(jobs["items"].is_array(), "jobs items array");
        expect_true(jobs["items"].size() <= 32, "job retention capped");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
