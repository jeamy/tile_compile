#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "preprocess") {
        std::string runs_dir;
        std::string run_id = "preprocessing_fake";
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--runs-dir" && i + 1 < argc) runs_dir = argv[++i];
            else if (arg == "--run-id" && i + 1 < argc) run_id = argv[++i];
        }
        if (!runs_dir.empty()) {
            const std::filesystem::path run_dir = std::filesystem::path(runs_dir) / run_id;
            const std::filesystem::path artifacts = run_dir / "artifacts" / "preprocess";
            std::filesystem::create_directories(artifacts);
            std::ofstream events(artifacts / "events.jsonl");
            events << "{\"type\":\"run_start\",\"run_id\":\"" << run_id << "\"}\n";
            events << "{\"type\":\"phase_start\",\"phase_name\":\"INPUT_SCAN\"}\n";
            events << "{\"type\":\"phase_end\",\"phase_name\":\"INPUT_SCAN\",\"status\":\"ok\"}\n";
            events << "{\"type\":\"phase_start\",\"phase_name\":\"ASTROMETRY\"}\n";
            events << "{\"type\":\"phase_end\",\"phase_name\":\"ASTROMETRY\",\"status\":\"skipped\",\"reason\":\"astap_not_found\"}\n";
            events << "{\"type\":\"phase_start\",\"phase_name\":\"BGE\"}\n";
            events << "{\"type\":\"phase_end\",\"phase_name\":\"BGE\",\"status\":\"skipped\",\"reason\":\"no_rgb_stack\"}\n";
            events << "{\"type\":\"phase_start\",\"phase_name\":\"PCC\"}\n";
            events << "{\"type\":\"phase_end\",\"phase_name\":\"PCC\",\"status\":\"skipped\",\"reason\":\"missing_wcs\"}\n";
            events << "{\"type\":\"phase_start\",\"phase_name\":\"HYPERMETRIC_STRETCH\"}\n";
            events << "{\"type\":\"phase_end\",\"phase_name\":\"HYPERMETRIC_STRETCH\",\"status\":\"ok\"}\n";
            events << "{\"type\":\"phase_start\",\"phase_name\":\"REPORT\"}\n";
            events << "{\"type\":\"phase_end\",\"phase_name\":\"REPORT\",\"status\":\"ok\"}\n";
            events << "{\"type\":\"run_end\",\"status\":\"ok\"}\n";
            std::ofstream(artifacts / "effective_config.json") << "{}\n";
            std::ofstream(artifacts / "frame_quality.csv") << "index,filename,included,star_count\n0,frame.fit,1,10\n";
            std::ofstream(artifacts / "quality_analysis.json") << "{\"n_total\":1,\"n_accepted\":1,\"n_rejected\":0}\n";
            std::ofstream(artifacts / "rejected_frames.txt") << "";
            std::ofstream(artifacts / "stacking_diagnostics.json") << "{\"frames_used\":1,\"method\":\"sigma\"}\n";
            std::ofstream(artifacts / "bge_diagnostics.json") << "{\"success\":false,\"failure_reason\":\"no_rgb_stack\"}\n";
            std::ofstream(artifacts / "pcc_diagnostics.json") << "{\"success\":false,\"error\":\"missing_wcs\"}\n";
            std::ofstream(artifacts / "hms_diagnostics.json") << "{\"success\":true}\n";
            std::filesystem::create_directories(run_dir / "outputs");
            std::ofstream(run_dir / "outputs" / "stacked_linear.fits") << "fake fits\n";
            std::ofstream(run_dir / "outputs" / "stacked_rgb_hms.fits") << "fake hms fits\n";
            std::ofstream(artifacts / "preprocessing_report.json")
                << "{\"status\":\"ok\",\"phases\":[{\"phase\":\"ASTROMETRY\",\"status\":\"skipped\"}],\"artifacts\":[]}\n";
            std::ofstream(artifacts / "preprocessing_report.md") << "# Preprocessing Report\n";
            std::ofstream(artifacts / "preprocessing_report.html") << "<!doctype html><html><body>Preprocessing Report</body></html>\n";
            std::ofstream(artifacts / "artifacts_manifest.json")
                << "{\"artifacts\":["
                << "{\"type\":\"report\",\"phase\":\"REPORT\",\"path\":\"" << (artifacts / "preprocessing_report.json").string() << "\"},"
                << "{\"type\":\"report_markdown\",\"phase\":\"REPORT\",\"path\":\"" << (artifacts / "preprocessing_report.md").string() << "\"},"
                << "{\"type\":\"report_html\",\"phase\":\"REPORT\",\"path\":\"" << (artifacts / "preprocessing_report.html").string() << "\"}"
                << "]}\n";
            std::cout << nlohmann::json{
                {"ok", true},
                {"run_id", run_id},
                {"run_dir", run_dir.string()},
                {"artifacts_dir", artifacts.string()},
                {"report_json", (artifacts / "preprocessing_report.json").string()},
                {"report_markdown", (artifacts / "preprocessing_report.md").string()},
                {"report_html", (artifacts / "preprocessing_report.html").string()},
                {"frame_quality_csv", (artifacts / "frame_quality.csv").string()},
                {"stacked_linear", (run_dir / "outputs" / "stacked_linear.fits").string()},
                {"stacking_diagnostics", (artifacts / "stacking_diagnostics.json").string()},
            }.dump() << std::endl;
        } else {
            std::cout << nlohmann::json{{"ok", true}, {"run_id", run_id}}.dump() << std::endl;
        }
        return 0;
    }

    int sleep_ms = 100;
    if (const char* raw = std::getenv("FAKE_TILE_COMPILE_RUNNER_SLEEP_MS")) {
        try {
            sleep_ms = std::max(0, std::stoi(raw));
        } catch (...) {}
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    nlohmann::json args = nlohmann::json::array();
    for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
    std::cout << nlohmann::json{{"ok", true}, {"args", args}}.dump() << std::endl;
    return 0;
}
