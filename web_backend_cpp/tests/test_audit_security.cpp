// Regression tests for the backend fixes of the 2026-11-21 code audit:
// the cross-site origin guard, /api/fs/open without a shell, the run
// directory resolution, "." and ".." as run ids and the runner subcommands
// the live-run detection looks for. The desktop opener is a fixture script;
// nothing is opened on the desktop.
#include "backend_test_harness.hpp"
#include "backend_runtime.hpp"
#include "job_store.hpp"
#include "origin_guard.hpp"
#include "runner_commands.hpp"

#include <chrono>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <thread>

namespace fs = std::filesystem;

namespace {

void check_origin_rules() {
    using origin_guard::origin_allowed;
    expect_true(origin_allowed("", "127.0.0.1:8080"), "no Origin is allowed");
    expect_true(origin_allowed("http://127.0.0.1:8080", "127.0.0.1:8080"), "same origin");
    expect_true(origin_allowed("http://localhost:5173", "127.0.0.1:8080"), "loopback dev server");
    expect_true(origin_allowed("http://127.1.2.3:9000", "127.0.0.1:8080"), "127/8 is loopback");
    expect_true(origin_allowed("http://[::1]:5173", "localhost:8080"), "IPv6 loopback");
    expect_true(origin_allowed("http://NAS.local:8080", "nas.local:8080"), "same host over the LAN, any case");
    expect_true(!origin_allowed("http://nas.local:9999", "nas.local:8080"), "same host, other port");
    expect_true(!origin_allowed("https://evil.example", "127.0.0.1:8080"), "foreign site");
    expect_true(!origin_allowed("http://127.0.0.1.evil.example", "127.0.0.1:8080"), "loopback-looking name");
    expect_true(!origin_allowed("http://localhost@evil.example", "127.0.0.1:8080"), "userinfo trick");
    expect_true(!origin_allowed("null", "127.0.0.1:8080"), "opaque origin (file://, sandbox)");
    expect_true(origin_allowed("http://nas.local:8080", "127.0.0.1:8080", "http://other:1, http://nas.local:8080"),
                "configured extra origin");
    expect_true(!origin_allowed("https://nas.local:8080", "127.0.0.1:8080", "http://nas.local:8080"),
                "extra origin compares the scheme too");
}

void check_runner_subcommands() {
    for (const char* cmd : {"reconstruct", "resume-reconstruction", "preprocess"})
        expect_true(is_runner_run_subcommand(cmd), std::string("run subcommand ") + cmd);
    for (const char* cmd : {"run", "resume", "--run-dir", ""})
        expect_true(!is_runner_run_subcommand(cmd), std::string("not a run subcommand: ") + cmd);
    expect_true(argv_has_runner_run_subcommand({"/x/tile_compile_runner", "reconstruct", "--config", "c"}),
                "argv with reconstruct");
    expect_true(!argv_has_runner_run_subcommand({"reconstruct"}), "the executable itself is not a subcommand");
}

void check_job_store() {
    InMemoryJobStore store;
    const std::string id = store.create("pcc_run");
    store.update_state(id, JobState::running);
    store.cancel(id);
    // A worker that reports progress after the cancel must not revive the job.
    store.update_state(id, JobState::running, {{"step", 2}});
    expect_true(store.get(id)->state == JobState::cancelled, "cancelled job stays cancelled");
    expect_true(store.get(id)->data.value("step", 0) == 2, "its payload is still updated");
    const std::string fast = store.create("scan");
    store.update_state(fast, JobState::ok);
    store.update_state(fast, JobState::running);
    expect_true(store.get(fast)->state == JobState::ok, "a finished job is not set back to running");
    store.update_state(id, JobState::error, {}, "late failure");
    expect_true(store.get(id)->state == JobState::error, "terminal to terminal is still allowed");
    expect_true(store.list(-5).empty(), "negative limit lists nothing");
    expect_true(store.list(0).empty(), "zero limit lists nothing");

    expect_true(job_id_newer("job_10_123", "job_9_99999"), "job 10 is newer than job 9");
    expect_true(!job_id_newer("job_9_99999", "job_10_1"), "job 9 is older than job 10");
    expect_true(job_id_newer("job_1_5", ""), "any job is newer than none");
}

template <typename F>
std::string error_of(F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

void check_resolve_run_dir(const fs::path& scratch) {
    const fs::path project = scratch / "project";
    const fs::path runs = scratch / "runs";
    const fs::path alt = scratch / "alt_runs";
    for (const auto& d : {project, runs / "M31_a", runs / "M31_b", runs / "M42_only", alt / "alt_run",
                          runs / "queue_root" / "L", alt / "alt_queue" / "R"})
        fs::create_directories(d);
    setenv("TILE_COMPILE_PROJECT_ROOT", project.c_str(), 1);
    setenv("TILE_COMPILE_RUNS_DIR", runs.c_str(), 1);
    setenv("TILE_COMPILE_ALLOWED_ROOTS", scratch.c_str(), 1);
    const BackendRuntime rt = BackendRuntime::from_env();

    expect_equal(rt.resolve_run_dir("M31_a").filename().string(), "M31_a", "exact run id");
    expect_equal(rt.resolve_run_dir("M42").filename().string(), "M42_only", "unique prefix");
    expect_true(error_of([&] { rt.resolve_run_dir("M31"); }).find("ambiguous") != std::string::npos,
                "an ambiguous prefix is refused, not resolved to the first listed run");
    expect_true(!error_of([&] { rt.resolve_run_dir(".."); }).empty(), "'..' is refused");
    expect_true(!error_of([&] { rt.resolve_run_dir("."); }).empty(), "'.' is refused");
    expect_equal(rt.resolve_run_dir("queue_root/L").string(), (runs / "queue_root" / "L").string(),
                 "a queue sub-run id");
    expect_equal(rt.resolve_run_dir("alt_queue/R", alt.string()).string(), (alt / "alt_queue" / "R").string(),
                 "a queue sub-run id in alt_runs_dir");
    for (const char* bad : {"queue_root/../M31_a", "queue_root/./L", "queue_root//L", "queue_root/",
                            "../runs/M31_a", "queue_root\\..\\M31_a"})
        expect_true(!error_of([&] { rt.resolve_run_dir(bad); }).empty(), std::string("refused: ") + bad);
    expect_true(!error_of([&] { rt.resolve_run_dir("queue_root/L_"); }).empty(),
                "no prefix matching below a queue root");
    expect_equal(rt.resolve_run_dir((runs / "M31_b").string()).string(), (runs / "M31_b").string(),
                 "an absolute run dir inside an allowed root");
    const fs::path outside = "/usr/share";
    if (fs::is_directory(outside) && !rt.is_path_allowed(outside)) {
        expect_true(!error_of([&] { rt.resolve_run_dir(outside.string()); }).empty(),
                    "an absolute run dir outside the allowed roots is refused");
        expect_true(!error_of([&] { rt.resolve_run_dir("share", "/usr"); }).empty(),
                    "an alt_runs_dir outside the allowed roots is ignored");
    }
    expect_equal(rt.resolve_run_dir("alt_run", alt.string()).filename().string(), "alt_run",
                 "alt_runs_dir inside an allowed root");
}

fs::path write_open_fixture(const fs::path& dir, const fs::path& out) {
    const fs::path script = dir / "fake_open.sh";
    std::ofstream f(script);
    // One line per argument: a shell-parsed path would split or expand.
    f << "#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\"; done > '" << out.string() << ".tmp'\n"
      << "mv '" << out.string() << ".tmp' '" << out.string() << "'\n";
    f.close();
    fs::permissions(script, fs::perms::owner_all);
    return script;
}

std::string wait_for_file(const fs::path& path, double timeout_s = 5.0) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        if (fs::exists(path)) return slurp_file(path);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return "";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    try {
        check_origin_rules();
        check_runner_subcommands();
        check_job_store();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    char tmpl[] = "/tmp/web_backend_cpp_audit_XXXXXX";
    if (!mkdtemp(tmpl)) return 2;
    const fs::path scratch(tmpl);
    struct Cleanup {
        fs::path dir;
        ~Cleanup() { std::error_code ec; fs::remove_all(dir, ec); }
    } cleanup{scratch};

    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        check_resolve_run_dir(scratch);

        const fs::path open_out = scratch / "opened.txt";
        const fs::path opener = write_open_fixture(scratch, open_out);
        harness.set_env("TILE_COMPILE_OPEN_COMMAND", opener.string());
        const fs::path cli_log = scratch / "cli_calls.txt";
        harness.set_env("FAKE_CLI_CALL_LOG", cli_log.string());
        // A UI directory of its own, with a sibling whose name starts with
        // "ui": a string-prefix check would take ui_evil/ for a part of ui/.
        fs::create_directories(scratch / "ui");
        fs::create_directories(scratch / "ui_evil");
        std::ofstream(scratch / "ui" / "index.html") << "UI_INDEX\n";
        std::ofstream(scratch / "ui_evil" / "outside_ui_marker.txt") << "OUTSIDE_UI_MARKER\n";
        harness.set_env("TILE_COMPILE_UI_DIR", (scratch / "ui").string());
        harness.start();

        // The schema depends only on the CLI binary: asked twice, the CLI runs once.
        for (int i = 0; i < 2; ++i) {
            const auto schema = harness.get_json("/api/config/schema");
            expect_equal(schema["_http_status"].get<long>(), 200L, "config schema");
            expect_equal(schema["type"].get<std::string>(), std::string("object"), "schema body");
        }
        {
            std::ifstream in(cli_log);
            std::string line;
            int schema_calls = 0;
            while (std::getline(in, line)) schema_calls += line == "get-schema";
            expect_equal(static_cast<long>(schema_calls), 1L, "get-schema is cached");
        }

        // A path that breaks out of the old `xdg-open '<path>'` shell line.
        const fs::path tricky = harness.fixture_root() / "it's $(echo injected) a dir; ls";
        fs::create_directories(tricky);
        const auto opened = harness.post_json("/api/fs/open", {{"path", tricky.string()}});
        expect_equal(opened["_http_status"].get<long>(), 200L, "fs/open starts the opener");
        expect_equal(wait_for_file(open_out), fs::weakly_canonical(tricky).string() + "\n",
                     "the opener gets the path as one literal argument");

        // Cross-site requests are refused before any route runs, also the
        // ones a browser sends without preflight.
        fs::remove(open_out);
        const std::string body = nlohmann::json({{"path", tricky.string()}}).dump();
        const auto foreign = harness.post_raw("/api/fs/open", body,
            {"Origin: https://evil.example", "Content-Type: text/plain"});
        expect_equal(foreign.status_code, 403L, "foreign origin is rejected");
        expect_true(foreign.body.find("ORIGIN_NOT_ALLOWED") != std::string::npos, "rejection code");
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        expect_true(!fs::exists(open_out), "a rejected request opens nothing");

        const std::string self = "http://127.0.0.1:" + std::to_string(harness.port());
        expect_equal(harness.get_with_headers("/api/health", {"Origin: " + self}).status_code, 200L,
                     "same-origin GET");
        expect_equal(harness.get_with_headers("/api/health", {"Origin: http://localhost:5173"}).status_code, 200L,
                     "loopback dev server");
        expect_equal(harness.get_with_headers("/api/health", {"Origin: http://nas.local:8080", "Host: nas.local:8080"}).status_code,
                     200L, "same host over the LAN");
        expect_equal(harness.get_with_headers("/api/health", {"Origin: https://evil.example"}).status_code, 403L,
                     "foreign GET is rejected (responses carry Access-Control-Allow-Origin: *)");

        // The static UI serves only files below its directory: neither the
        // sibling ui_evil/, nor a ".." path, nor an absolute one.
        expect_true(harness.get_with_headers("/ui/index.html", {}).body.find("UI_INDEX") != std::string::npos,
                    "static UI serves its own files");
        const fs::path marker = scratch / "ui_evil" / "outside_ui_marker.txt";
        for (const std::string& probe : {std::string("/ui/../ui_evil/outside_ui_marker.txt"),
                                         std::string("/ui/%2e%2e/ui_evil/outside_ui_marker.txt"),
                                         "/ui/" + marker.generic_string()}) {
            const auto leaked = harness.get_with_headers(probe, {});
            expect_true(leaked.body.find("OUTSIDE_UI_MARKER") == std::string::npos,
                        "static UI does not serve " + probe);
        }

        // /api/scan/metrics hands input_path to the CLI only inside an allowed root.
        const auto metrics = harness.post_json("/api/scan/metrics", {{"input_path", "/etc"}});
        const long metrics_status = metrics["_http_status"].get<long>();
        expect_true(metrics_status >= 400 && metrics_status < 500,
                    "scan metrics outside the allowed roots is refused: " + metrics.dump());

        // "." and ".." never name the runs directory or its parent.
        harness.make_file("inputs/dots/frame_0001.fit", "fixture\n");
        const std::string input_dir = (harness.fixture_root() / "inputs" / "dots").string();
        const nlohmann::json dots_start = {
            {"input_dir", input_dir}, {"color_mode", "OSC"}, {"config_yaml", "data:\n  color_mode: OSC\n"}};
        auto with_id = [&](const char* id) { auto b = dots_start; b["run_id"] = id; return b; };
        const auto parent = harness.post_json("/api/runs/start", with_id(".."));
        expect_equal(parent["_http_status"].get<long>(), 202L, "run start with run_id ..");
        expect_equal(parent["run_id"].get<std::string>(), "run", "run_id .. becomes run");
        harness.wait_for_job(parent["job_id"].get<std::string>());
        // "." maps to the same name; the existing run is not overwritten.
        const auto self_dir = harness.post_json("/api/runs/start", with_id("."));
        expect_equal(self_dir["_http_status"].get<long>(), 409L, "run_id . collides with run: " + self_dir.dump());
        expect_equal(self_dir["error"]["details"]["run_id"].get<std::string>(), "run", "run_id . is checked as run");
        expect_equal(self_dir["error"]["code"].get<std::string>(), "RUN_CONFIG_SNAPSHOT_CONFLICT", "collision code");
        std::cout << "audit security tests passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
