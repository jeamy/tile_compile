#include "backend_test_harness.hpp"

#include <chrono>
#include <cstdio>
#include <thread>

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        const auto input_rgb = harness.make_binary_file("inputs/pcc_input.fit");
        const auto wcs_file = harness.make_binary_file("inputs/pcc_input.wcs");
        const auto output_rgb = harness.fixture_root() / "outputs" / "pcc_ok.fit";

        auto accepted = harness.post_json("/api/tools/pcc/run", {
            {"input_rgb", input_rgb.string()},
            {"output_rgb", output_rgb.string()},
            {"wcs_file", wcs_file.string()},
            {"source", "auto"},
            {"k_max", 2.5},
            {"min_stars", 12}
        });
        expect_equal(accepted["_http_status"].get<long>(), 202L, "pcc accepted status");
        const auto job = harness.wait_for_job(accepted["job_id"].get<std::string>());
        expect_equal(job["state"].get<std::string>(), "ok", "pcc job ok");
        const auto& data = job["data"];
        expect_json_field(data, "result", "pcc result field");
        expect_equal(data["stars_matched"].get<long>(), 42L, "pcc stars matched");
        expect_equal(data["stars_used"].get<long>(), 37L, "pcc stars used");
        expect_equal(data["output_rgb"].get<std::string>(), output_rgb.string(), "pcc output path");
        expect_true(std::filesystem::exists(output_rgb), "pcc output exists");

        const auto slow_input = harness.make_binary_file("inputs/pcc_slow_input.fit");
        const auto slow_wcs = harness.make_binary_file("inputs/pcc_slow_input.wcs");
        const auto slow_output = harness.fixture_root() / "outputs" / "pcc_slow.fit";
        accepted = harness.post_json("/api/tools/pcc/run", {
            {"input_rgb", slow_input.string()},
            {"output_rgb", slow_output.string()},
            {"wcs_file", slow_wcs.string()},
            {"source", "auto"}
        });
        expect_equal(accepted["_http_status"].get<long>(), 202L, "slow pcc accepted status");
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        const auto cancel = harness.post_json("/api/jobs/" + accepted["job_id"].get<std::string>() + "/cancel", nlohmann::json::object());
        expect_equal(cancel["_http_status"].get<long>(), 200L, "cancel status");
        const auto cancelled_job = harness.wait_for_job(accepted["job_id"].get<std::string>());
        expect_equal(cancelled_job["state"].get<std::string>(), "cancelled", "cancelled job state");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
