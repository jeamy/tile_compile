#include "backend_test_harness.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    ::setenv("FAKE_TILE_COMPILE_RUNNER_SLEEP_MS", "1500", 1);
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
        expect_equal(status["status"].get<std::string>(), "aborted", "stale resume run without live job becomes aborted");
        expect_true(status["current_phase"].is_null(), "stale resume current phase cleared");
        bool found_bge = false;
        for (const auto& item : status["phases"]) {
            if (item.value("phase", "") == "BGE") {
                found_bge = true;
                expect_equal(item["status"].get<std::string>(), "aborted", "bge stale resumed status");
                expect_equal(item["pct"].get<double>(), 0.0, "bge resumed pct", 1e-9);
            }
        }
        expect_true(found_bge, "bge phase present");

        const auto skipped_phase_run_dir = harness.create_run("skipped_phase_run", {
            {{"ts", "2026-03-10T11:00:00Z"}, {"type", "phase_start"}, {"phase_name", "STATE_CLUSTERING"}},
            {{"ts", "2026-03-10T11:00:01Z"}, {"type", "phase_end"}, {"phase_name", "STATE_CLUSTERING"}, {"status", "skipped"}},
            {{"ts", "2026-03-10T11:00:02Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");
        {
            std::ofstream config(skipped_phase_run_dir / "config.yaml");
            config << "method: classic_tile_compile\n"
                   << "data:\n  color_mode: OSC\n";
        }

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

        const auto aqmh_run_dir = harness.create_run("aqmh_hides_classic_phases", {
            {{"ts", "2026-03-10T11:10:00Z"}, {"type", "phase_start"}, {"phase_name", "STATE_CLUSTERING"}},
            {{"ts", "2026-03-10T11:10:01Z"}, {"type", "phase_end"}, {"phase_name", "STATE_CLUSTERING"}, {"status", "skipped"}},
            {{"ts", "2026-03-10T11:10:02Z"}, {"type", "phase_start"}, {"phase_name", "SYNTHETIC_FRAMES"}},
            {{"ts", "2026-03-10T11:10:03Z"}, {"type", "phase_end"}, {"phase_name", "SYNTHETIC_FRAMES"}, {"status", "skipped"}},
            {{"ts", "2026-03-10T11:10:04Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");
        {
            std::ofstream config(aqmh_run_dir / "config.yaml");
            config << "method: aqmh\n"
                   << "data:\n  color_mode: OSC\n";
        }

        const auto aqmh_status = harness.get_json("/api/runs/aqmh_hides_classic_phases/status");
        expect_equal(aqmh_status["_http_status"].get<long>(), 200L, "aqmh status code");
        expect_true(aqmh_status["aqmh_enabled"].is_boolean(), "aqmh flag is boolean");
        expect_true(aqmh_status["aqmh_enabled"].get<bool>(), "aqmh flag derived from method");
        expect_equal(aqmh_status["method"].get<std::string>(), "aqmh", "aqmh method parsed from config");
        bool found_aqmh_maps = false;
        bool found_aqmh_local_metrics = false;
        bool found_aqmh_state_clustering = false;
        bool found_aqmh_synthetic_frames = false;
        for (const auto& item : aqmh_status["phases"]) {
            const std::string phase = item.value("phase", "");
            if (phase == "AQMH_MAPS") found_aqmh_maps = true;
            if (phase == "LOCAL_METRICS") found_aqmh_local_metrics = true;
            if (phase == "STATE_CLUSTERING") found_aqmh_state_clustering = true;
            if (phase == "SYNTHETIC_FRAMES") found_aqmh_synthetic_frames = true;
        }
        expect_true(found_aqmh_maps, "aqmh maps phase present");
        expect_true(!found_aqmh_local_metrics, "aqmh hides classic local metrics phase");
        expect_true(!found_aqmh_state_clustering, "aqmh hides state clustering");
        expect_true(!found_aqmh_synthetic_frames, "aqmh hides synthetic frames");

        harness.create_run("aqmh_hides_classic_phases_from_events", {
            {{"ts", "2026-03-10T11:20:00Z"}, {"type", "phase_start"}, {"phase_name", "AQMH_QUALITY_MAPS"}},
            {{"ts", "2026-03-10T11:20:01Z"}, {"type", "phase_end"}, {"phase_name", "LOCAL_METRICS"}, {"status", "ok"}},
            {{"ts", "2026-03-10T11:20:02Z"}, {"type", "phase_start"}, {"phase_name", "STATE_CLUSTERING"}},
            {{"ts", "2026-03-10T11:20:03Z"}, {"type", "phase_end"}, {"phase_name", "STATE_CLUSTERING"}, {"status", "skipped"}},
            {{"ts", "2026-03-10T11:20:04Z"}, {"type", "phase_start"}, {"phase_name", "SYNTHETIC_FRAMES"}},
            {{"ts", "2026-03-10T11:20:05Z"}, {"type", "phase_end"}, {"phase_name", "SYNTHETIC_FRAMES"}, {"status", "skipped"}},
            {{"ts", "2026-03-10T11:20:06Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");

        const auto aqmh_event_status = harness.get_json("/api/runs/aqmh_hides_classic_phases_from_events/status");
        expect_equal(aqmh_event_status["_http_status"].get<long>(), 200L, "default-method aqmh status code");
        expect_true(aqmh_event_status["aqmh_enabled"].is_boolean(), "default-method aqmh flag is boolean");
        expect_true(aqmh_event_status["aqmh_enabled"].get<bool>(), "missing method defaults to aqmh");
        expect_equal(aqmh_event_status["method"].get<std::string>(), "aqmh", "missing method reports aqmh");
        bool found_event_aqmh_maps = false;
        bool found_event_local_metrics = false;
        for (const auto& item : aqmh_event_status["phases"]) {
            const std::string phase = item.value("phase", "");
            if (phase == "AQMH_MAPS") {
                found_event_aqmh_maps = true;
                expect_equal(item["status"].get<std::string>(), "ok", "aqmh maps consumes local_metrics phase_end");
                expect_equal(item["pct"].get<double>(), 1.0, "aqmh maps done pct", 1e-9);
            }
            if (phase == "LOCAL_METRICS") found_event_local_metrics = true;
            expect_true(phase != "STATE_CLUSTERING", "event-derived aqmh hides state clustering");
            expect_true(phase != "SYNTHETIC_FRAMES", "event-derived aqmh hides synthetic frames");
        }
        expect_true(found_event_aqmh_maps, "event-derived aqmh maps phase present");
        expect_true(!found_event_local_metrics, "event-derived aqmh hides classic local metrics");

        const auto aqmh_reconstruction_run_dir = harness.create_run("aqmh_reconstruction_alias_status", {
            {{"ts", "2026-03-10T11:25:00Z"}, {"type", "phase_start"}, {"phase_name", "AQMH_QUALITY_MAPS"}},
            {{"ts", "2026-03-10T11:25:01Z"}, {"type", "phase_end"}, {"phase_name", "LOCAL_METRICS"}, {"status", "ok"}},
            {{"ts", "2026-03-10T11:25:02Z"}, {"type", "phase_start"}, {"phase_name", "TILE_RECONSTRUCTION"}},
            {{"ts", "2026-03-10T11:25:03Z"}, {"type", "phase_end"}, {"phase_name", "TILE_RECONSTRUCTION"}, {"status", "ok"}},
            {{"ts", "2026-03-10T11:25:04Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");
        {
            std::ofstream config(aqmh_reconstruction_run_dir / "config.yaml");
            config << "method: aqmh\n"
                   << "data:\n  color_mode: OSC\n";
        }

        const auto aqmh_reconstruction_status = harness.get_json("/api/runs/aqmh_reconstruction_alias_status/status");
        expect_equal(aqmh_reconstruction_status["_http_status"].get<long>(), 200L, "aqmh reconstruction alias status code");
        bool found_aqmh_reconstruction = false;
        bool found_tile_reconstruction = false;
        for (const auto& item : aqmh_reconstruction_status["phases"]) {
            const std::string phase = item.value("phase", "");
            if (phase == "AQMH_RECONSTRUCTION") {
                found_aqmh_reconstruction = true;
                expect_equal(item["status"].get<std::string>(), "ok", "aqmh reconstruction consumes tile reconstruction phase_end");
                expect_equal(item["pct"].get<double>(), 1.0, "aqmh reconstruction done pct", 1e-9);
            }
            if (phase == "TILE_RECONSTRUCTION") found_tile_reconstruction = true;
        }
        expect_true(found_aqmh_reconstruction, "aqmh reconstruction phase present");
        expect_true(!found_tile_reconstruction, "aqmh hides classic tile reconstruction phase");

        harness.create_run("completed_without_run_end", {
            {{"ts", "2026-03-10T11:30:00Z"}, {"type", "phase_start"}, {"phase_name", "ASTROMETRY"}},
            {{"ts", "2026-03-10T11:30:01Z"}, {"type", "phase_end"}, {"phase_name", "ASTROMETRY"}, {"status", "ok"}},
            {{"ts", "2026-03-10T11:30:02Z"}, {"type", "phase_start"}, {"phase_name", "BGE"}},
            {{"ts", "2026-03-10T11:30:03Z"}, {"type", "phase_end"}, {"phase_name", "BGE"}, {"status", "ok"}},
            {{"ts", "2026-03-10T11:30:04Z"}, {"type", "phase_start"}, {"phase_name", "PCC"}},
            {{"ts", "2026-03-10T11:30:05Z"}, {"type", "phase_end"}, {"phase_name", "PCC"}, {"status", "ok"}}
        }, "OSC");

        const auto completed_without_run_end = harness.get_json("/api/runs/completed_without_run_end/status");
        expect_equal(completed_without_run_end["_http_status"].get<long>(), 200L, "completed without run_end status code");
        expect_equal(completed_without_run_end["status"].get<std::string>(), "completed", "pcc terminal phase implies completed");

        harness.create_run("partial_without_run_end", {
            {{"ts", "2026-03-10T11:45:00Z"}, {"type", "phase_start"}, {"phase_name", "ASTROMETRY"}},
            {{"ts", "2026-03-10T11:45:01Z"}, {"type", "phase_end"}, {"phase_name", "ASTROMETRY"}, {"status", "ok"}}
        }, "OSC");

        const auto partial_without_run_end = harness.get_json("/api/runs/partial_without_run_end/status");
        expect_equal(partial_without_run_end["_http_status"].get<long>(), 200L, "partial without run_end status code");
        expect_equal(partial_without_run_end["status"].get<std::string>(), "unknown", "partial run without active phase must not imply running");

        harness.create_run("skipped_then_resume_ok_run", {
            {{"ts", "2026-03-10T12:00:00Z"}, {"type", "phase_start"}, {"phase_name", "ASTROMETRY"}},
            {{"ts", "2026-03-10T12:00:01Z"}, {"type", "phase_end"}, {"phase_name", "ASTROMETRY"}, {"status", "skipped"}},
            {{"ts", "2026-03-10T12:00:02Z"}, {"type", "run_end"}, {"success", true}},
            {{"ts", "2026-03-10T12:05:00Z"}, {"type", "resume_start"}, {"from_phase", "ASTROMETRY"}},
            {{"ts", "2026-03-10T12:05:05Z"}, {"type", "resume_end"}, {"success", true}, {"status", "ok"}}
        }, "OSC");

        const auto resumed_ok_status = harness.get_json("/api/runs/skipped_then_resume_ok_run/status");
        expect_equal(resumed_ok_status["_http_status"].get<long>(), 200L, "resumed skipped phase status code");
        expect_equal(resumed_ok_status["status"].get<std::string>(), "completed", "resumed skipped run completed status");
        bool found_resumed_phase = false;
        for (const auto& item : resumed_ok_status["phases"]) {
            if (item.value("phase", "") == "ASTROMETRY") {
                found_resumed_phase = true;
                expect_equal(item["status"].get<std::string>(), "ok", "astrometry status upgraded after successful resume");
                expect_equal(item["pct"].get<double>(), 1.0, "astrometry pct upgraded after successful resume", 1e-9);
            }
        }
        expect_true(found_resumed_phase, "astrometry phase present after successful resume");

        harness.create_run("pcc_resume_keeps_target_running", {
            {{"ts", "2026-03-10T13:00:00Z"}, {"type", "phase_start"}, {"phase_name", "ASTROMETRY"}},
            {{"ts", "2026-03-10T13:00:01Z"}, {"type", "phase_end"}, {"phase_name", "ASTROMETRY"}, {"status", "ok"}},
            {{"ts", "2026-03-10T13:00:02Z"}, {"type", "phase_start"}, {"phase_name", "PCC"}},
            {{"ts", "2026-03-10T13:00:03Z"}, {"type", "phase_end"}, {"phase_name", "PCC"}, {"status", "ok"}},
            {{"ts", "2026-03-10T13:00:04Z"}, {"type", "run_end"}, {"success", true}},
            {{"ts", "2026-03-10T13:05:00Z"}, {"type", "resume_start"}, {"from_phase", "PCC"}},
            {{"ts", "2026-03-10T13:05:01Z"}, {"type", "phase_start"}, {"phase_name", "ASTROMETRY"}},
            {{"ts", "2026-03-10T13:05:02Z"}, {"type", "phase_end"}, {"phase_name", "ASTROMETRY"}, {"status", "skipped"}, {"reason", "existing_wcs"}}
        }, "OSC");

        const auto pcc_resume_status = harness.get_json("/api/runs/pcc_resume_keeps_target_running/status");
        expect_equal(pcc_resume_status["_http_status"].get<long>(), 200L, "pcc resume status code");
        expect_equal(pcc_resume_status["status"].get<std::string>(), "aborted", "stale pcc resume run becomes aborted");
        expect_true(pcc_resume_status["current_phase"].is_null(), "stale pcc resume current phase cleared");
        bool found_astrometry_after_pcc_resume = false;
        bool found_pcc_after_pcc_resume = false;
        for (const auto& item : pcc_resume_status["phases"]) {
            if (item.value("phase", "") == "ASTROMETRY") {
                found_astrometry_after_pcc_resume = true;
                expect_equal(item["status"].get<std::string>(), "ok", "astrometry keeps ok when existing wcs is reused for pcc resume");
            }
            if (item.value("phase", "") == "PCC") {
                found_pcc_after_pcc_resume = true;
                expect_equal(item["status"].get<std::string>(), "aborted", "stale pcc phase becomes aborted without live job");
                expect_equal(item["pct"].get<double>(), 0.0, "pcc remains at 0 pct before phase_start", 1e-9);
            }
        }
        expect_true(found_astrometry_after_pcc_resume, "astrometry phase present after pcc resume");
        expect_true(found_pcc_after_pcc_resume, "pcc phase present after pcc resume");

        harness.create_run("resume_overlay_without_events", {
            {{"ts", "2026-03-10T14:00:00Z"}, {"type", "phase_start"}, {"phase_name", "ASTROMETRY"}},
            {{"ts", "2026-03-10T14:00:01Z"}, {"type", "phase_end"}, {"phase_name", "ASTROMETRY"}, {"status", "ok"}},
            {{"ts", "2026-03-10T14:00:02Z"}, {"type", "phase_start"}, {"phase_name", "BGE"}},
            {{"ts", "2026-03-10T14:00:03Z"}, {"type", "phase_end"}, {"phase_name", "BGE"}, {"status", "ok"}},
            {{"ts", "2026-03-10T14:00:04Z"}, {"type", "phase_start"}, {"phase_name", "PCC"}},
            {{"ts", "2026-03-10T14:00:05Z"}, {"type", "phase_end"}, {"phase_name", "PCC"}, {"status", "ok"}},
            {{"ts", "2026-03-10T14:00:06Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");

        const auto resumed = harness.post_json("/api/runs/resume_overlay_without_events/resume", {
            {"from_phase", "BGE"},
            {"run_dir", "runs/resume_overlay_without_events"},
            {"config_yaml", "data:\n  color_mode: OSC\n"}
        });
        expect_equal(resumed["_http_status"].get<long>(), 202L, "resume overlay launch status");

        const auto overlay_status = harness.get_json("/api/runs/resume_overlay_without_events/status");
        expect_equal(overlay_status["_http_status"].get<long>(), 200L, "resume overlay status code");
        expect_equal(overlay_status["status"].get<std::string>(), "running", "resume overlay run status");
        expect_equal(overlay_status["current_phase"].get<std::string>(), "BGE", "resume overlay current phase");
        bool found_overlay_bge = false;
        bool found_overlay_pcc = false;
        for (const auto& item : overlay_status["phases"]) {
            if (item.value("phase", "") == "BGE") {
                found_overlay_bge = true;
                expect_equal(item["status"].get<std::string>(), "running", "resume overlay bge status");
                expect_equal(item["pct"].get<double>(), 0.0, "resume overlay bge pct", 1e-9);
            }
            if (item.value("phase", "") == "PCC") {
                found_overlay_pcc = true;
                expect_equal(item["status"].get<std::string>(), "pending", "resume overlay resets later phases");
                expect_equal(item["pct"].get<double>(), 0.0, "resume overlay resets later phase pct", 1e-9);
            }
        }
        expect_true(found_overlay_bge, "resume overlay target phase present");
        expect_true(found_overlay_pcc, "resume overlay later phase present");

        const auto resumed_job = harness.wait_for_job(resumed["job_id"].get<std::string>(), 5.0);
        expect_equal(resumed_job["state"].get<std::string>(), "ok", "resume overlay job completes");
        expect_equal(resumed_job["data"]["run_dir"].get<std::string>(),
                     (harness.fixture_root() / "runs" / "resume_overlay_without_events").string(),
                     "relative resume run_dir is normalized in job data");
        const auto& resume_command = resumed_job["data"]["command"];
        bool found_normalized_run_dir_arg = false;
        for (size_t i = 0; i + 1 < resume_command.size(); ++i) {
            if (resume_command[i].get<std::string>() == "--run-dir" &&
                resume_command[i + 1].get<std::string>() == (harness.fixture_root() / "runs" / "resume_overlay_without_events").string()) {
                found_normalized_run_dir_arg = true;
            }
        }
        expect_true(found_normalized_run_dir_arg, "relative resume run_dir is normalized before runner launch");

        const auto aqmh_resume_overlay_run_dir = harness.create_run("aqmh_resume_overlay_alias", {
            {{"ts", "2026-03-10T15:00:00Z"}, {"type", "phase_start"}, {"phase_name", "AQMH_QUALITY_MAPS"}},
            {{"ts", "2026-03-10T15:00:01Z"}, {"type", "phase_end"}, {"phase_name", "LOCAL_METRICS"}, {"status", "ok"}},
            {{"ts", "2026-03-10T15:00:02Z"}, {"type", "phase_start"}, {"phase_name", "TILE_RECONSTRUCTION"}},
            {{"ts", "2026-03-10T15:00:03Z"}, {"type", "phase_end"}, {"phase_name", "TILE_RECONSTRUCTION"}, {"status", "ok"}},
            {{"ts", "2026-03-10T15:00:04Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");
        {
            std::ofstream config(aqmh_resume_overlay_run_dir / "config.yaml");
            config << "method: aqmh\n"
                   << "data:\n  color_mode: OSC\n";
        }

        const auto aqmh_resumed = harness.post_json("/api/runs/aqmh_resume_overlay_alias/resume", {
            {"from_phase", "TILE_RECONSTRUCTION"},
            {"run_dir", (harness.fixture_root() / "runs" / "aqmh_resume_overlay_alias").string()},
            {"config_yaml", "method: aqmh\ndata:\n  color_mode: OSC\n"}
        });
        expect_equal(aqmh_resumed["_http_status"].get<long>(), 202L, "aqmh resume overlay launch status");

        const auto aqmh_overlay_status = harness.get_json("/api/runs/aqmh_resume_overlay_alias/status");
        expect_equal(aqmh_overlay_status["_http_status"].get<long>(), 200L, "aqmh resume overlay status code");
        expect_equal(aqmh_overlay_status["status"].get<std::string>(), "running", "aqmh resume overlay run status");
        expect_equal(aqmh_overlay_status["current_phase"].get<std::string>(), "AQMH_RECONSTRUCTION", "aqmh resume overlay current phase");
        bool found_overlay_aqmh_maps = false;
        bool found_overlay_aqmh_reconstruction = false;
        bool found_overlay_stack = false;
        for (const auto& item : aqmh_overlay_status["phases"]) {
            if (item.value("phase", "") == "AQMH_MAPS") {
                found_overlay_aqmh_maps = true;
                expect_equal(item["status"].get<std::string>(), "ok", "aqmh resume overlay keeps earlier aqmh maps ok");
                expect_equal(item["pct"].get<double>(), 1.0, "aqmh resume overlay maps pct", 1e-9);
            }
            if (item.value("phase", "") == "AQMH_RECONSTRUCTION") {
                found_overlay_aqmh_reconstruction = true;
                expect_equal(item["status"].get<std::string>(), "running", "aqmh resume overlay reconstruction status");
                expect_equal(item["pct"].get<double>(), 0.0, "aqmh resume overlay reconstruction pct", 1e-9);
            }
            if (item.value("phase", "") == "STACKING") {
                found_overlay_stack = true;
            }
        }
        expect_true(found_overlay_aqmh_maps, "aqmh maps phase present after resume overlay");
        expect_true(found_overlay_aqmh_reconstruction, "aqmh reconstruction phase present after resume overlay");
        expect_true(!found_overlay_stack, "aqmh resume overlay hides classic stacking phase");

        const auto aqmh_resumed_job = harness.wait_for_job(aqmh_resumed["job_id"].get<std::string>(), 5.0);
        expect_equal(aqmh_resumed_job["state"].get<std::string>(), "ok", "aqmh resume overlay job completes");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
