// Unit checks of the header-only helpers the routes and services share
// (base64url run ids, ISO timestamps, the run event file, /proc argv).
#include "base64url.hpp"
#include "proc_scan.hpp"
#include "run_event_file.hpp"
#include "time_utils.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}
}  // namespace

int main() {
    // base64url: "M31_F1/x?" with and without padding, '-'/'_' alphabet.
    check(decode_base64url("TTMxX0YxL3g_") == "M31_F1/x?", "base64url decodes '_' as '/'");
    check(decode_base64url("TTMxX0YxL3g_") == decode_base64url("TTMxX0YxL3g_===="), "padding is optional");
    check(decode_base64url("Pz4-") == "?>>", "base64url decodes '-' as '+'");
    check(decode_base64url("a!b") == "", "a character outside the alphabet gives empty");
    const std::string long_id(4000, 'A');
    check(decode_base64url(long_id).size() == 3000, "long input decodes without overflow");
    check(decode_b64_run_id("b64_TTMxX0YxL3g_") == "M31_F1/x?", "b64_ run id is decoded");
    check(decode_b64_run_id("b64_!!") == "b64_!!", "undecodable b64_ run id comes back unchanged");
    check(decode_b64_run_id("plain_run") == "plain_run", "plain run id comes back unchanged");

    // ISO timestamps.
    const auto base = parse_iso_utc_seconds("2026-03-10T10:00:00Z");
    check(base && *base == 1773136800.0, "whole seconds, Z");
    const auto frac = parse_iso_utc_seconds("2026-03-10T10:00:00.250Z");
    check(frac && std::fabs(*frac - (*base + 0.25)) < 1e-6, "fraction");
    check(parse_iso_utc_seconds("2026-03-10T10:00:00.xyZ") == base, "non-digit fraction is ignored");
    check(!parse_iso_utc_seconds("garbage"), "garbage does not parse");
    check(!parse_iso_utc_seconds("2026-03-10"), "date only does not parse");

    // Run event file: canonical first, legacy names as fallback.
    const fs::path root = fs::temp_directory_path() / ("tc_shared_helpers_" + std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root / "logs");
    check(!find_run_event_file(root), "no event file");
    std::ofstream(root / "events.jsonl") << "{}\n";
    check(find_run_event_file(root) == root / "events.jsonl", "legacy events.jsonl");
    std::ofstream(root / "logs" / "run_events.jsonl") << "{}\n";
    check(find_run_event_file(root) == root / "logs" / "run_events.jsonl", "canonical file wins");
    fs::remove_all(root);

    // /proc argv.
    const auto self_argv = read_process_argv("/proc/self");
    check(!self_argv.empty(), "own argv readable");
    check(join_argv({"a", "b c", "d"}) == "a b c d", "join_argv");
    check(argv_is_runner_exe({"/opt/x/tile_compile_runner", "reconstruct"}, "other"), "runner by stem");
    check(argv_is_runner_exe({"/opt/x/my_runner"}, "my_runner"), "runner by configured name");
    check(!argv_is_runner_exe({"/usr/bin/bash"}, "my_runner"), "not a runner");
    check(!argv_is_runner_exe({}, "my_runner"), "empty argv");

    if (failures == 0) std::cout << "shared helpers: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
