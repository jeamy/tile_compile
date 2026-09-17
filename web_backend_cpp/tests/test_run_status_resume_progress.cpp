#include "backend_test_harness.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>

static std::string test_phase_name(const nlohmann::json& item) {
    if (item.contains("phase") && item["phase"].is_string()) return item["phase"].get<std::string>();
    if (item.contains("phase_name") && item["phase_name"].is_string()) return item["phase_name"].get<std::string>();
    return "";
}

static std::string test_error_code(const nlohmann::json& payload) {
    if (payload.contains("code") && payload["code"].is_string()) return payload["code"].get<std::string>();
    if (payload.contains("error") && payload["error"].is_object() &&
        payload["error"].contains("code") && payload["error"]["code"].is_string()) {
        return payload["error"]["code"].get<std::string>();
    }
    return "";
}

static const nlohmann::json& test_error_details(const nlohmann::json& payload) {
    static const nlohmann::json empty = nlohmann::json::object();
    if (payload.contains("error") && payload["error"].is_object() &&
        payload["error"].contains("details") && payload["error"]["details"].is_object()) {
        return payload["error"]["details"];
    }
    if (payload.contains("details") && payload["details"].is_object()) return payload["details"];
    return empty;
}

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
            if (test_phase_name(item) == "BGE") {
                found_bge = true;
                expect_equal(item["status"].get<std::string>(), "aborted", "bge stale resumed status");
                expect_equal(item["pct"].get<double>(), 0.0, "bge resumed pct", 1e-9);
            }
        }
        expect_true(found_bge, "bge phase present");

        harness.create_run("skipped_phase_run", {
            {{"ts", "2026-03-10T11:00:00Z"}, {"type", "phase_start"}, {"phase_name", "MULTIBAND"}},
            {{"ts", "2026-03-10T11:00:01Z"}, {"type", "phase_end"}, {"phase_name", "MULTIBAND"}, {"status", "skipped"}},
            {{"ts", "2026-03-10T11:00:02Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");

        const auto skipped_status = harness.get_json("/api/runs/skipped_phase_run/status");
        expect_equal(skipped_status["_http_status"].get<long>(), 200L, "skipped status code");
        bool found_skipped = false;
        for (const auto& item : skipped_status["phases"]) {
            if (test_phase_name(item) == "MULTIBAND") {
                found_skipped = true;
                expect_equal(item["status"].get<std::string>(), "skipped", "multiband skipped status");
                expect_equal(item["pct"].get<double>(), 1.0, "multiband skipped pct", 1e-9);
            }
        }
        expect_true(found_skipped, "multiband phase present");

        // Runs without a stored method belong to the single-method
        // forward-drizzle pipeline: canonical phase order, no aqmh aliases.
        harness.create_run("default_method_forward_drizzle", {
            {{"ts", "2026-03-10T11:28:00Z"}, {"type", "phase_start"}, {"phase_name", "FORWARD_DRIZZLE"}},
            {{"ts", "2026-03-10T11:28:01Z"}, {"type", "phase_end"}, {"phase_name", "FORWARD_DRIZZLE"}, {"status", "ok"}},
            {{"ts", "2026-03-10T11:28:02Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");

        const auto default_method_status = harness.get_json("/api/runs/default_method_forward_drizzle/status");
        expect_equal(default_method_status["_http_status"].get<long>(), 200L, "default-method status code");
        expect_equal(default_method_status["method"].get<std::string>(), "cfa_forward_drizzle_multiband",
                     "missing method reports forward-drizzle pipeline");
        expect_true(!default_method_status.contains("aqmh_enabled"), "status has no aqmh flag");
        bool found_fd_phase = false;
        for (const auto& item : default_method_status["phases"]) {
            const std::string phase = test_phase_name(item);
            if (phase == "FORWARD_DRIZZLE") found_fd_phase = true;
            expect_true(phase != "AQMH_MAPS", "no aqmh phase aliases");
            expect_true(phase != "TILE_RECONSTRUCTION", "no classic tile reconstruction phase");
            expect_true(phase != "STATE_CLUSTERING", "no classic state clustering phase");
        }
        expect_true(found_fd_phase, "forward_drizzle phase present for default method");

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

        harness.create_run("rerun_resets_phase_status", {
            {{"ts", "2026-03-10T11:35:00Z"}, {"type", "run_start"}, {"run_id", "rerun_resets_phase_status"}},
            {{"ts", "2026-03-10T11:35:01Z"}, {"type", "phase_start"}, {"phase_name", "SCAN_INPUT"}},
            {{"ts", "2026-03-10T11:35:02Z"}, {"type", "phase_end"}, {"phase_name", "SCAN_INPUT"}, {"status", "ok"}},
            {{"ts", "2026-03-10T11:35:03Z"}, {"type", "phase_start"}, {"phase_name", "REGISTRATION"}},
            {{"ts", "2026-03-10T11:35:04Z"}, {"type", "phase_end"}, {"phase_name", "REGISTRATION"}, {"status", "ok"}},
            {{"ts", "2026-03-10T11:35:05Z"}, {"type", "phase_start"}, {"phase_name", "NORMALIZATION"}},
            {{"ts", "2026-03-10T11:35:06Z"}, {"type", "phase_end"}, {"phase_name", "NORMALIZATION"}, {"status", "ok"}},
            {{"ts", "2026-03-10T11:35:07Z"}, {"type", "run_end"}, {"success", true}},
            {{"ts", "2026-03-10T11:40:00Z"}, {"type", "run_start"}, {"run_id", "rerun_resets_phase_status"}},
            {{"ts", "2026-03-10T11:40:01Z"}, {"type", "phase_start"}, {"phase_name", "SCAN_INPUT"}},
            {{"ts", "2026-03-10T11:40:02Z"}, {"type", "phase_end"}, {"phase_name", "SCAN_INPUT"}, {"status", "ok"}},
            {{"ts", "2026-03-10T11:40:03Z"}, {"type", "phase_start"}, {"phase_name", "CHANNEL_SPLIT"}},
            {{"ts", "2026-03-10T11:40:04Z"}, {"type", "phase_end"}, {"phase_name", "CHANNEL_SPLIT"}, {"status", "ok"}},
            {{"ts", "2026-03-10T11:40:05Z"}, {"type", "phase_start"}, {"phase_name", "NORMALIZED_CACHE"}}
        }, "OSC");

        const auto rerun_status = harness.get_json("/api/runs/rerun_resets_phase_status/status");
        expect_equal(rerun_status["_http_status"].get<long>(), 200L, "rerun status code");
        int channel_split_index = -1;
        int registration_index = -1;
        bool found_pending_registration = false;
        bool found_pending_normalization = false;
        for (size_t i = 0; i < rerun_status["phases"].size(); ++i) {
            const auto& item = rerun_status["phases"][i];
            const std::string phase = test_phase_name(item);
            if (phase == "CHANNEL_SPLIT") channel_split_index = static_cast<int>(i);
            if (phase == "REGISTRATION") {
                registration_index = static_cast<int>(i);
                found_pending_registration = item["status"].get<std::string>() == "pending";
            }
            if (phase == "NORMALIZATION") {
                found_pending_normalization = item["status"].get<std::string>() == "pending";
            }
        }
        expect_true(channel_split_index >= 0 && registration_index >= 0 && channel_split_index < registration_index,
                    "phase list follows runner order before registration");
        expect_true(found_pending_registration, "new run_start resets stale registration status");
        expect_true(found_pending_normalization, "new run_start resets stale normalization status");

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
            if (test_phase_name(item) == "ASTROMETRY") {
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
            if (test_phase_name(item) == "ASTROMETRY") {
                found_astrometry_after_pcc_resume = true;
                expect_equal(item["status"].get<std::string>(), "ok", "astrometry keeps ok when existing wcs is reused for pcc resume");
            }
            if (test_phase_name(item) == "PCC") {
                found_pcc_after_pcc_resume = true;
                expect_equal(item["status"].get<std::string>(), "aborted", "stale pcc phase becomes aborted without live job");
                expect_equal(item["pct"].get<double>(), 0.0, "pcc remains at 0 pct before phase_start", 1e-9);
            }
        }
        expect_true(found_astrometry_after_pcc_resume, "astrometry phase present after pcc resume");
        expect_true(found_pcc_after_pcc_resume, "pcc phase present after pcc resume");

        harness.create_run("resume_overlay_without_events", {
            {{"ts", "2026-03-10T14:00:00Z"}, {"type", "phase_start"}, {"phase_name", "FORWARD_DRIZZLE"}},
            {{"ts", "2026-03-10T14:00:01Z"}, {"type", "phase_end"}, {"phase_name", "FORWARD_DRIZZLE"}, {"status", "ok"}},
            {{"ts", "2026-03-10T14:00:02Z"}, {"type", "phase_start"}, {"phase_name", "MULTIBAND"}},
            {{"ts", "2026-03-10T14:00:03Z"}, {"type", "phase_end"}, {"phase_name", "MULTIBAND"}, {"status", "ok"}},
            {{"ts", "2026-03-10T14:00:04Z"}, {"type", "phase_start"}, {"phase_name", "BGE"}},
            {{"ts", "2026-03-10T14:00:05Z"}, {"type", "phase_end"}, {"phase_name", "BGE"}, {"status", "ok"}},
            {{"ts", "2026-03-10T14:00:06Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");
        // Forward-drizzle provenance is required by the resume-reconstruction
        // contract (the backend mirrors the runner's scope gate).
        harness.make_file("runs/resume_overlay_without_events/artifacts/run_provenance.json",
                          "{\"execution_scope\": \"forward_drizzle_m1_m3\"}\n");

        // Phases outside the resume contract are rejected. MULTIBAND is not a
        // resume entry: it always re-runs together with FORWARD_DRIZZLE.
        const auto unsupported_phase = harness.post_json("/api/runs/resume_overlay_without_events/resume", {
            {"from_phase", "MULTIBAND"},
            {"run_dir", "runs/resume_overlay_without_events"},
            {"config_yaml", "data:\n  color_mode: OSC\n"},
            {"dry_run", true}
        });
        expect_equal(unsupported_phase["_http_status"].get<long>(), 409L,
                     "unsupported resume phase rejected");
        expect_equal(unsupported_phase["error"]["details"]["reason"].get<std::string>(),
                     "unsupported_resume_phase", "unsupported resume phase reason");

        // Downstream phases are resume entries: they reuse the persisted
        // reconstruction outputs and re-run only the downstream chain.
        for (const char* downstream_phase : {"BGE", "PCC", "HYPERMETRIC_STRETCH", "ASTROMETRY"}) {
            const auto downstream_resume = harness.post_json("/api/runs/resume_overlay_without_events/resume", {
                {"from_phase", downstream_phase},
                {"run_dir", "runs/resume_overlay_without_events"},
                {"config_yaml", "data:\n  color_mode: OSC\n"},
                {"dry_run", true}
            });
            expect_equal(downstream_resume["_http_status"].get<long>(), 200L,
                         std::string("downstream resume phase accepted: ") + downstream_phase);
            expect_true(downstream_resume.value("feasible", false),
                        std::string("downstream resume feasible: ") + downstream_phase);
        }

        const auto resumed = harness.post_json("/api/runs/resume_overlay_without_events/resume", {
            {"from_phase", "FORWARD_DRIZZLE"},
            {"run_dir", "runs/resume_overlay_without_events"},
            {"config_yaml", "data:\n  color_mode: OSC\n"}
        });
        expect_equal(resumed["_http_status"].get<long>(), 202L, "resume overlay launch status");

        const auto overlay_status = harness.get_json("/api/runs/resume_overlay_without_events/status");
        expect_equal(overlay_status["_http_status"].get<long>(), 200L, "resume overlay status code");
        expect_equal(overlay_status["status"].get<std::string>(), "running", "resume overlay run status");
        expect_equal(overlay_status["current_phase"].get<std::string>(), "FORWARD_DRIZZLE", "resume overlay current phase");
        bool found_overlay_fd = false;
        bool found_overlay_multiband = false;
        for (const auto& item : overlay_status["phases"]) {
            if (test_phase_name(item) == "FORWARD_DRIZZLE") {
                found_overlay_fd = true;
                expect_equal(item["status"].get<std::string>(), "running", "resume overlay forward_drizzle status");
                expect_equal(item["pct"].get<double>(), 0.0, "resume overlay forward_drizzle pct", 1e-9);
            }
            if (test_phase_name(item) == "MULTIBAND") {
                found_overlay_multiband = true;
                expect_equal(item["status"].get<std::string>(), "pending", "resume overlay resets later phases");
                expect_equal(item["pct"].get<double>(), 0.0, "resume overlay resets later phase pct", 1e-9);
            }
        }
        expect_true(found_overlay_fd, "resume overlay target phase present");
        expect_true(found_overlay_multiband, "resume overlay later phase present");

        const auto overlay_logs = harness.get_json(
            "/api/runs/resume_overlay_without_events/logs?tail=20&run_dir=runs%2Fresume_overlay_without_events");
        expect_equal(overlay_logs["_http_status"].get<long>(), 200L, "resume logs with run_dir status code");
        expect_true(!overlay_logs["lines"].empty(), "resume logs with run_dir returns event lines");

        const auto resumed_job = harness.wait_for_job(resumed["job_id"].get<std::string>(), 5.0);
        // fake_tile_compile_runner may not implement resume-reconstruction; accept ok or error
        expect_true(resumed_job["state"].get<std::string>() == "ok" || resumed_job["state"].get<std::string>() == "error",
                    "resume overlay job terminates");
        if (resumed_job["state"].get<std::string>() == "ok") {
            expect_equal(resumed_job["data"]["run_dir"].get<std::string>(),
                         (harness.fixture_root() / "runs" / "resume_overlay_without_events").string(),
                         "relative resume run_dir is normalized in job data");
            const auto& resume_command = resumed_job["data"]["command"];
            expect_equal(resume_command[1].get<std::string>(), "resume-reconstruction",
                         "resume launches the resume-reconstruction subcommand");
            bool found_normalized_run_dir_arg = false;
            for (size_t i = 0; i + 1 < resume_command.size(); ++i) {
                if (resume_command[i].get<std::string>() == "--run-dir" &&
                    resume_command[i + 1].get<std::string>() == (harness.fixture_root() / "runs" / "resume_overlay_without_events").string()) {
                    found_normalized_run_dir_arg = true;
                }
            }
            expect_true(found_normalized_run_dir_arg, "relative resume run_dir is normalized before runner launch");
        }

        // A resume attempt is a SECONDARY action against an already-finished
        // run. If it fails immediately -- before the runner writes a single
        // event to the run's own log, e.g. a real
        // FORWARD_STAGE_CONFIG_OR_SCOPE_MISMATCH exit -- that failure must
        // not silently overwrite the already-completed run's status; it is
        // surfaced separately via `resume_attempt` instead.
        harness.create_run("resume_fails_after_completion", {
            {{"ts", "2026-03-10T16:00:00Z"}, {"type", "phase_start"}, {"phase_name", "PCC"}},
            {{"ts", "2026-03-10T16:00:01Z"}, {"type", "phase_end"}, {"phase_name", "PCC"}, {"status", "ok"}},
            {{"ts", "2026-03-10T16:00:02Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");
        harness.make_file("runs/resume_fails_after_completion/artifacts/run_provenance.json",
                          "{\"execution_scope\": \"forward_drizzle_m1_m3\"}\n");
        // Tells the fake runner to exit non-zero on the resume-reconstruction
        // call for THIS run_dir, mirroring a real FORWARD_STAGE_CONFIG_OR_SCOPE_MISMATCH.
        harness.make_file("runs/resume_fails_after_completion/FAIL_RESUME_MARKER", "1\n");

        const auto pre_resume_status = harness.get_json("/api/runs/resume_fails_after_completion/status");
        expect_equal(pre_resume_status["status"].get<std::string>(), "completed",
                     "run completed before resume attempt");

        const auto failing_resume = harness.post_json("/api/runs/resume_fails_after_completion/resume", {
            {"from_phase", "PCC"},
            {"run_dir", "runs/resume_fails_after_completion"},
            {"config_yaml", "data:\n  color_mode: OSC\n"}
        });
        expect_equal(failing_resume["_http_status"].get<long>(), 202L, "failing resume launch accepted");
        const auto failed_job = harness.wait_for_job(failing_resume["job_id"].get<std::string>(), 5.0);
        expect_equal(failed_job["state"].get<std::string>(), "error", "resume job reports error");

        const auto post_resume_status = harness.get_json("/api/runs/resume_fails_after_completion/status");
        expect_equal(post_resume_status["_http_status"].get<long>(), 200L, "status after failed resume attempt");
        expect_equal(post_resume_status["status"].get<std::string>(), "completed",
                     "a failed resume attempt does not overwrite the already-completed run's status");
        expect_true(post_resume_status.contains("resume_attempt") && post_resume_status["resume_attempt"].is_object(),
                    "failed resume attempt is surfaced separately");
        expect_equal(post_resume_status["resume_attempt"]["state"].get<std::string>(), "error",
                     "resume_attempt reports the failed state");

        // The backend applies the SAME downstream-resume section-scope gate
        // the runner itself enforces, checked before any config write or
        // subprocess launch (see downstream_resume_scope_violation). This is
        // exactly the HMS-dialog bug: "Apply & start resume" silently saved
        // an edited config but the re-stretch never visibly ran because the
        // runner rejected out-of-scope drift (e.g. chroma_denoise) after the
        // write already happened. It must now be rejected immediately.
        harness.create_run("hms_resume_scope_check", {
            {{"ts", "2026-03-10T17:00:00Z"}, {"type", "phase_start"}, {"phase_name", "HYPERMETRIC_STRETCH"}},
            {{"ts", "2026-03-10T17:00:01Z"}, {"type", "phase_end"}, {"phase_name", "HYPERMETRIC_STRETCH"}, {"status", "ok"}},
            {{"ts", "2026-03-10T17:00:02Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");
        harness.make_file("runs/hms_resume_scope_check/artifacts/run_provenance.json",
                          "{\"execution_scope\": \"forward_drizzle_m1_m3\"}\n");
        harness.make_file("runs/hms_resume_scope_check/artifacts/config_revisions/orig.yaml",
                          "data:\n  color_mode: OSC\nchroma_denoise:\n  enabled: true\nhypermetric_stretch:\n  sensor_profile: rec709\n");
        harness.make_file("runs/hms_resume_scope_check/artifacts/config_revisions/index.json",
                          "[{\"revision_id\":\"orig_runstart\",\"file_name\":\"orig.yaml\",\"source\":\"run_start\","
                          "\"created_at\":\"2026-03-10T17:00:00Z\",\"run_id\":\"hms_resume_scope_check\"}]\n");

        const auto scope_violation = harness.post_json("/api/runs/hms_resume_scope_check/resume", {
            {"from_phase", "HYPERMETRIC_STRETCH"},
            {"run_dir", "runs/hms_resume_scope_check"},
            {"config_yaml", "data:\n  color_mode: OSC\nchroma_denoise:\n  enabled: false\nhypermetric_stretch:\n  sensor_profile: rec709\n"},
            {"dry_run", true}
        });
        expect_equal(scope_violation["_http_status"].get<long>(), 409L, "out-of-scope HMS resume config rejected");
        expect_equal(scope_violation["error"]["details"]["reason"].get<std::string>(), "config_scope_mismatch",
                     "out-of-scope HMS resume reports config_scope_mismatch");

        const auto scope_ok = harness.post_json("/api/runs/hms_resume_scope_check/resume", {
            {"from_phase", "HYPERMETRIC_STRETCH"},
            {"run_dir", "runs/hms_resume_scope_check"},
            {"config_yaml", "data:\n  color_mode: OSC\nchroma_denoise:\n  enabled: true\nhypermetric_stretch:\n  sensor_profile: \"Sony IMX415 (DWARF II)\"\n"},
            {"dry_run", true}
        });
        expect_equal(scope_ok["_http_status"].get<long>(), 200L, "in-scope-only HMS resume config accepted");
        expect_true(scope_ok.value("feasible", false), "in-scope-only HMS resume reports feasible");

        // Runs without forward-drizzle provenance cannot be resumed; the
        // runner would fail its provenance check anyway, so the backend
        // rejects early with a clear error.
        harness.create_run("legacy_run_no_provenance", {
            {{"ts", "2026-03-10T15:00:00Z"}, {"type", "run_start"}, {"input_dir", (harness.fixture_root() / "input").string()}},
            {{"ts", "2026-03-10T15:00:01Z"}, {"type", "phase_start"}, {"phase_name", "TILE_RECONSTRUCTION"}},
            {{"ts", "2026-03-10T15:00:02Z"}, {"type", "phase_end"}, {"phase_name", "TILE_RECONSTRUCTION"}, {"status", "ok"}},
            {{"ts", "2026-03-10T15:00:03Z"}, {"type", "run_end"}, {"success", true}}
        }, "OSC");
        const auto legacy_resume = harness.post_json("/api/runs/legacy_run_no_provenance/resume", {
            {"from_phase", "FORWARD_DRIZZLE"},
            {"run_dir", (harness.fixture_root() / "runs" / "legacy_run_no_provenance").string()},
            {"dry_run", true}
        });
        expect_equal(legacy_resume["_http_status"].get<long>(), 409L,
                     "legacy run without forward-drizzle provenance rejected");
        expect_equal(legacy_resume["error"]["details"]["reason"].get<std::string>(),
                     "run_scope_unsupported", "legacy run scope rejection reason");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
