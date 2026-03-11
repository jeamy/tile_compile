#include "backend_test_harness.hpp"

#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 5) return 2;
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
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
