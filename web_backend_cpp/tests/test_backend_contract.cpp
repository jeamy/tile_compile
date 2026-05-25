#include "backend_test_harness.hpp"

#include <algorithm>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    setenv("TILE_COMPILE_GUI2_INSTALL_ROOT", argv[4], 1);
    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        const auto health = harness.get_json("/api/health");
        expect_equal(health["_http_status"].get<long>(), 200L, "health status");
        expect_equal(health["status"].get<std::string>(), "ok", "health payload");

        const auto constants = harness.get_json("/api/app/constants");
        expect_equal(constants["_http_status"].get<long>(), 200L, "constants status");
        expect_true(constants["phases"].is_array(), "constants phases array");
        expect_true(constants["resume_from"].is_array(), "constants resume array");
        expect_true(std::find(constants["resume_from"].begin(), constants["resume_from"].end(), "TILE_RECONSTRUCTION") != constants["resume_from"].end(),
                    "constants resume includes TILE_RECONSTRUCTION");
        expect_true(std::find(constants["resume_from"].begin(), constants["resume_from"].end(), "PCC") != constants["resume_from"].end(),
                    "constants resume includes PCC");
        expect_equal(constants["color_modes"][0].get<std::string>(), "OSC", "constants color mode 0");
        expect_equal(constants["color_modes"][1].get<std::string>(), "MONO", "constants color mode 1");
        expect_equal(constants["color_modes"][2].get<std::string>(), "RGB", "constants color mode 2");

        const auto current = harness.get_json("/api/config/current");
        expect_equal(current["_http_status"].get<long>(), 200L, "config current status");
        expect_true(current["config"].get<std::string>().find("data:") != std::string::npos, "config current yaml");
        expect_equal(current["source"].get<std::string>(), harness.config_path().string(), "config current source");

        const auto validate = harness.post_json("/api/config/validate", {{"yaml", "data:\n  color_mode: MONO\n"}});
        expect_equal(validate["_http_status"].get<long>(), 200L, "config validate status");
        expect_true(validate["ok"].get<bool>(), "config validate ok");
        expect_true(validate["errors"].is_array(), "config validate errors array");
        expect_true(validate["warnings"].is_array(), "config validate warnings array");

        const auto missing = harness.get_json("/api/runs/does_not_exist/status");
        expect_equal(missing["_http_status"].get<long>(), 404L, "missing run status");
        expect_equal(missing["error"]["code"].get<std::string>(), "NOT_FOUND", "missing run error code");
        expect_true(missing["error"]["details"].is_object(), "missing run error details");

        const auto ui = harness.get("/ui");
        expect_equal(ui.status_code, 200L, "ui status");
        expect_true(ui.body.find("<html") != std::string::npos || ui.body.find("<HTML") != std::string::npos, "ui html body");
        const auto raw_stack_ui = harness.get("/raw-stack.html");
        expect_equal(raw_stack_ui.status_code, 200L, "raw stack ui status");
        expect_true(raw_stack_ui.body.find("raw-stack-start") != std::string::npos, "raw stack ui body");

        const auto astrometry_detect = harness.post_json("/api/tools/astrometry/detect", nlohmann::json::object());
        expect_equal(astrometry_detect["_http_status"].get<long>(), 200L, "astrometry detect status");
        expect_equal(astrometry_detect["data_dir"].get<std::string>(), (std::filesystem::path(argv[4]) / "astap").string(), "astrometry default gui2 install dir");
        expect_equal(astrometry_detect["catalog_dir"].get<std::string>(), (std::filesystem::path(argv[4]) / "astap").string(), "astrometry default catalog dir");

        const auto pcc_status = harness.get_json("/api/tools/pcc/siril/status");
        expect_equal(pcc_status["_http_status"].get<long>(), 200L, "pcc status");
        expect_equal(pcc_status["catalog_dir"].get<std::string>(),
                     (std::filesystem::path(argv[4]) / "pcc" / "siril_cat1_healpix8_xpsamp").string(),
                     "pcc default gui2 install dir");

        const auto preprocessing_defaults = harness.get_json("/api/tools/preprocessing/defaults");
        expect_equal(preprocessing_defaults["_http_status"].get<long>(), 200L, "preprocessing defaults status");
        expect_equal(preprocessing_defaults["config"]["mode"].get<std::string>(), "linear_prestack", "preprocessing mode");
        expect_true(preprocessing_defaults["config"]["postprocess"]["astrometry"].get<bool>(), "preprocessing astrometry default");
        expect_true(preprocessing_defaults["config"]["postprocess"]["bge"].get<bool>(), "preprocessing bge default");
        expect_true(preprocessing_defaults["config"]["postprocess"]["pcc"].get<bool>(), "preprocessing pcc default");
        expect_true(preprocessing_defaults["config"]["postprocess"]["hypermetric_stretch"].get<bool>(), "preprocessing hms default");
        expect_true(preprocessing_defaults["config"]["hypermetric_stretch"].is_object(), "preprocessing hms config block");
        expect_true(preprocessing_defaults["config"]["hypermetric_stretch"]["require_successful_pcc"].get<bool>(),
                    "preprocessing hms require successful pcc default");
        expect_equal(preprocessing_defaults["config"]["hypermetric_stretch"]["mode"].get<std::string>(),
                     "ready_to_use",
                     "preprocessing hms mode default");
        expect_equal(preprocessing_defaults["config"]["hypermetric_stretch"]["sensor_profile"].get<std::string>(),
                     "rec709",
                     "preprocessing hms sensor profile default");
        expect_true(std::find(preprocessing_defaults["config"]["report"]["formats"].begin(),
                              preprocessing_defaults["config"]["report"]["formats"].end(),
                              "html") != preprocessing_defaults["config"]["report"]["formats"].end(),
                    "preprocessing report formats include html");

        const auto preprocessing_parameters = harness.get_json("/api/tools/preprocessing/parameters");
        expect_equal(preprocessing_parameters["_http_status"].get<long>(), 200L, "preprocessing parameters status");
        expect_true(preprocessing_parameters["separate_from_tile_compile"].get<bool>(), "preprocessing separate flag");
        expect_true(preprocessing_parameters["groups"].is_array(), "preprocessing parameter groups");
        {
            bool has_hms_group = false;
            bool has_postprocess_group = false;
            bool has_quality_group = false;
            for (const auto& group : preprocessing_parameters["groups"]) {
                const std::string id = group.value("id", "");
                if (id == "hypermetric_stretch") {
                    has_hms_group = true;
                    expect_true(std::find(group["paths"].begin(), group["paths"].end(), "hypermetric_stretch.output_rgb") != group["paths"].end(),
                                "preprocessing parameters hms group exposes output_rgb");
                }
                if (id == "postprocess") {
                    has_postprocess_group = true;
                    expect_true(std::find(group["paths"].begin(), group["paths"].end(), "postprocess.hypermetric_stretch") != group["paths"].end(),
                                "preprocessing parameters postprocess group exposes hms toggle");
                }
                if (id == "quality_filter") {
                    has_quality_group = true;
                    expect_true(std::find(group["paths"].begin(), group["paths"].end(), "quality_filter.manual_overrides") != group["paths"].end(),
                                "preprocessing parameters quality group exposes manual overrides");
                }
            }
            expect_true(has_hms_group, "preprocessing parameters has hms group");
            expect_true(has_postprocess_group, "preprocessing parameters has postprocess group");
            expect_true(has_quality_group, "preprocessing parameters has quality group");
        }

        std::filesystem::create_directories(harness.fixture_root() / "lights_mono");
        const auto preprocessing_scan = harness.post_json("/api/tools/preprocessing/scan", {
            {"lights_dir", (harness.fixture_root() / "lights_mono").string()}
        });
        expect_equal(preprocessing_scan["_http_status"].get<long>(), 200L, "preprocessing scan start status");
        const auto preprocessing_scan_job = harness.wait_for_job(preprocessing_scan["job_id"].get<std::string>());
        expect_equal(preprocessing_scan_job["state"].get<std::string>(), "ok", "preprocessing scan job ok");

        const auto preprocessing_status = harness.get_json("/api/tools/preprocessing/status?job_id=" + preprocessing_scan["job_id"].get<std::string>());
        expect_equal(preprocessing_status["_http_status"].get<long>(), 200L, "preprocessing scan status");
        expect_true(preprocessing_status["job"]["scan"].is_object(), "preprocessing scan normalized payload");
        expect_equal(preprocessing_status["job"]["scan"]["input_mode"].get<std::string>(), "mono", "preprocessing scan mono mode");

        const auto preprocessing_run = harness.post_json("/api/tools/preprocessing/run", {
            {"lights_dir", (harness.fixture_root() / "lights_mono").string()}
        });
        expect_equal(preprocessing_run["_http_status"].get<long>(), 200L, "preprocessing run start status");
        const auto preprocessing_run_job = harness.wait_for_job(preprocessing_run["job_id"].get<std::string>());
        expect_equal(preprocessing_run_job["state"].get<std::string>(), "ok", "preprocessing run job ok");
        expect_true(preprocessing_run_job["data"].contains("frame_quality_csv"),
                    "preprocessing run reports frame quality artifact");
        expect_true(preprocessing_run_job["data"].contains("stacked_linear"),
                    "preprocessing run reports stacked linear artifact");
        expect_true(preprocessing_run_job["data"].contains("stacking_diagnostics"),
                    "preprocessing run reports stacking diagnostics artifact");
        expect_true(preprocessing_run_job["data"].contains("report_markdown"),
                    "preprocessing run reports markdown report artifact");
        expect_true(preprocessing_run_job["data"].contains("report_html"),
                    "preprocessing run reports html report artifact");
        const std::string preprocessing_run_id = preprocessing_run["run_id"].get<std::string>();

        const auto preprocessing_report = harness.get_json("/api/tools/preprocessing/report?job_id=" + preprocessing_run["job_id"].get<std::string>());
        expect_equal(preprocessing_report["_http_status"].get<long>(), 200L, "preprocessing report status");
        expect_equal(preprocessing_report["run_id"].get<std::string>(), preprocessing_run_id, "preprocessing report returns run id");
        expect_true(preprocessing_report["artifacts_dir"].get<std::string>().find("artifacts/preprocess") != std::string::npos,
                    "preprocessing report uses artifacts preprocess path");
        expect_true(preprocessing_report["report_json"].get<std::string>().find("preprocessing_report.json") != std::string::npos,
                    "preprocessing report_json path contains preprocessing_report.json");
        expect_true(preprocessing_report["report_markdown"].get<std::string>().find("preprocessing_report.md") != std::string::npos,
                    "preprocessing report_markdown path contains preprocessing_report.md");
        expect_true(preprocessing_report["report_html"].get<std::string>().find("preprocessing_report.html") != std::string::npos,
                    "preprocessing report_html path contains preprocessing_report.html");

        const auto preprocessing_artifacts = harness.get_json("/api/runs/" + preprocessing_run_id + "/artifacts");
        expect_equal(preprocessing_artifacts["_http_status"].get<long>(), 200L, "preprocessing run artifacts status");
        {
            bool found_manifest = false;
            bool found_quality = false;
            bool found_events = false;
            bool found_report_html = false;
            for (const auto& item : preprocessing_artifacts["items"]) {
                const std::string rel = item.value("relative_path", item.value("filename", ""));
                if (rel == "artifacts/preprocess/artifacts_manifest.json") found_manifest = true;
                if (rel == "artifacts/preprocess/frame_quality.csv") found_quality = true;
                if (rel == "artifacts/preprocess/events.jsonl") found_events = true;
                if (rel == "artifacts/preprocess/preprocessing_report.html") found_report_html = true;
            }
            expect_true(found_manifest, "preprocessing artifacts include manifest");
            expect_true(found_quality, "preprocessing artifacts include frame quality csv");
            expect_true(found_events, "preprocessing artifacts include events jsonl");
            expect_true(found_report_html, "preprocessing artifacts include html report");
        }

        const auto preprocessing_event_view = harness.get_json("/api/runs/" + preprocessing_run_id + "/artifacts/view?path=artifacts%2Fpreprocess%2Fevents.jsonl");
        expect_equal(preprocessing_event_view["_http_status"].get<long>(), 200L, "preprocessing event artifact view status");
        expect_true(preprocessing_event_view["text"].get<std::string>().find("HYPERMETRIC_STRETCH") != std::string::npos,
                    "preprocessing event artifact contains hms phase");

        const auto preprocessing_report_raw = harness.get("/api/runs/" + preprocessing_run_id + "/artifacts/raw/artifacts/preprocess/preprocessing_report.html");
        expect_equal(preprocessing_report_raw.status_code, 200L, "preprocessing html report raw status");
        expect_true(preprocessing_report_raw.body.find("Preprocessing Report") != std::string::npos,
                    "preprocessing html report raw body");

        const auto preprocessing_run_status = harness.get_json("/api/tools/preprocessing/status?job_id=" + preprocessing_run["job_id"].get<std::string>());
        expect_equal(preprocessing_run_status["_http_status"].get<long>(), 200L, "preprocessing run status ok");
        expect_true(preprocessing_run_status["phases"].is_array(), "preprocessing run status has phases");
        {
            bool found_report_ok = false;
            bool found_stretch_ok = false;
            for (const auto& p : preprocessing_run_status["phases"]) {
                if (p["phase"].get<std::string>() == "REPORT" && p["status"].get<std::string>() == "ok")
                    found_report_ok = true;
                if (p["phase"].get<std::string>() == "HYPERMETRIC_STRETCH" && p["status"].get<std::string>() == "ok")
                    found_stretch_ok = true;
            }
            expect_true(found_report_ok, "preprocessing run status REPORT phase ok");
            expect_true(found_stretch_ok, "preprocessing run status HYPERMETRIC_STRETCH enabled by default");
        }

        const auto preprocessing_patch = harness.patch_json("/api/tools/preprocessing/parameters", {
            {"config", {
                {"postprocess", {{"hypermetric_stretch", false}}},
                {"hypermetric_stretch", {{"target_bg", 0.22}}},
                {"quality_filter", {{"mode", "strict"}, {"manual_overrides", {{"0", {{"include", false}}}}}}}
            }}
        });
        expect_equal(preprocessing_patch["_http_status"].get<long>(), 200L, "preprocessing PATCH parameters status");
        expect_true(preprocessing_patch["ok"].get<bool>(), "preprocessing PATCH parameters ok");
        expect_true(!preprocessing_patch["config"]["postprocess"]["hypermetric_stretch"].get<bool>(),
                    "preprocessing PATCH parameters can disable hypermetric_stretch");
        expect_equal(preprocessing_patch["config"]["quality_filter"]["mode"].get<std::string>(),
                     "strict",
                     "preprocessing PATCH parameters keeps quality override");
        expect_equal(preprocessing_patch["config"]["hypermetric_stretch"]["target_bg"].get<double>(),
                     0.22,
                     "preprocessing PATCH parameters keeps hms target override");
        expect_true(preprocessing_patch["config"]["quality_filter"]["manual_overrides"].is_object(),
                    "preprocessing PATCH parameters keeps manual overrides");
        expect_true(preprocessing_patch["config"]["postprocess"]["astrometry"].get<bool>(),
                    "preprocessing PATCH parameters merges astrometry default");
        expect_true(preprocessing_patch["config"]["postprocess"]["bge"].get<bool>(),
                    "preprocessing PATCH parameters merges bge default");
        expect_true(preprocessing_patch["config"]["postprocess"]["pcc"].get<bool>(),
                    "preprocessing PATCH parameters merges pcc default");

        const auto preprocessing_persisted = harness.get_json("/api/tools/preprocessing/parameters");
        expect_equal(preprocessing_persisted["config"]["quality_filter"]["mode"].get<std::string>(),
                     "strict",
                     "preprocessing parameters persist patched config");
        const auto preprocessing_invalid_patch = harness.patch_json("/api/tools/preprocessing/parameters", {
            {"config", {{"quality_filter", {{"min_correlation", 2.0}}}}}
        });
        expect_equal(preprocessing_invalid_patch["_http_status"].get<long>(), 400L, "preprocessing PATCH rejects invalid config");

        const auto preprocessing_disabled_run = harness.post_json("/api/tools/preprocessing/run", {
            {"lights_dir", (harness.fixture_root() / "lights_mono").string()},
            {"postprocess", {
                {"astrometry", false},
                {"bge", false},
                {"pcc", false},
                {"hypermetric_stretch", false}
            }}
        });
        expect_equal(preprocessing_disabled_run["_http_status"].get<long>(), 200L, "preprocessing disabled postprocess run start status");
        const auto preprocessing_disabled_job = harness.wait_for_job(preprocessing_disabled_run["job_id"].get<std::string>());
        expect_equal(preprocessing_disabled_job["state"].get<std::string>(), "ok", "preprocessing disabled postprocess job ok");
        const auto preprocessing_disabled_status = harness.get_json("/api/tools/preprocessing/status?job_id=" + preprocessing_disabled_run["job_id"].get<std::string>());
        expect_equal(preprocessing_disabled_status["_http_status"].get<long>(), 200L, "preprocessing disabled postprocess status");
        {
            bool astrometry_skipped = false;
            bool bge_skipped = false;
            bool pcc_skipped = false;
            bool hms_skipped = false;
            for (const auto& p : preprocessing_disabled_status["phases"]) {
                const std::string phase = p["phase"].get<std::string>();
                const std::string status = p["status"].get<std::string>();
                if (phase == "ASTROMETRY" && status == "skipped") astrometry_skipped = true;
                if (phase == "BGE" && status == "skipped") bge_skipped = true;
                if (phase == "PCC" && status == "skipped") pcc_skipped = true;
                if (phase == "HYPERMETRIC_STRETCH" && status == "skipped") hms_skipped = true;
            }
            expect_true(astrometry_skipped, "preprocessing disabled astrometry skipped");
            expect_true(bge_skipped, "preprocessing disabled bge skipped");
            expect_true(pcc_skipped, "preprocessing disabled pcc skipped");
            expect_true(hms_skipped, "preprocessing disabled hms skipped");
        }
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
