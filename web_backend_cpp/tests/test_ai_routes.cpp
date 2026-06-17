#include "backend_test_harness.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {

int reserve_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw TestFailure("failed to create socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(0);
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        throw TestFailure("failed to bind fake sidecar port");
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
        close(fd);
        throw TestFailure("failed to read fake sidecar port");
    }
    const int port = ntohs(addr.sin_port);
    close(fd);
    return port;
}

class FakeSidecar {
public:
    explicit FakeSidecar(nlohmann::json response)
        : _response(std::move(response)), _port(reserve_port()) {}

    ~FakeSidecar() {
        if (_thread.joinable()) _thread.join();
    }

    void start() {
        _thread = std::thread([this]() { serve_once(); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!_ready.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!_ready.load()) throw TestFailure("fake sidecar did not become ready");
    }

    std::string url() const {
        return "http://127.0.0.1:" + std::to_string(_port);
    }

private:
    void serve_once() {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return;
        int reuse = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(static_cast<uint16_t>(_port));
        if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            close(fd);
            return;
        }
        if (listen(fd, 1) != 0) {
            close(fd);
            return;
        }
        _ready.store(true);

        int client = accept(fd, nullptr, nullptr);
        if (client >= 0) {
            char buffer[4096];
            recv(client, buffer, sizeof(buffer), 0);
            const std::string body = _response.dump();
            const std::string http =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: " + std::to_string(body.size()) + "\r\n"
                "Connection: close\r\n\r\n" + body;
            send(client, http.data(), http.size(), 0);
            close(client);
        }
        close(fd);
    }

    nlohmann::json _response;
    int _port{0};
    std::atomic<bool> _ready{false};
    std::thread _thread;
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        harness.start();

        const auto initial_config = harness.get_json("/api/ai/config");
        expect_equal(initial_config["_http_status"].get<long>(), 200L, "ai config status");
        expect_true(!initial_config["enabled"].get<bool>(), "ai scan default disabled");
        expect_equal(initial_config["mode"].get<std::string>(), "manual", "ai scan default mode");

        const auto disabled_analysis = harness.post_json("/api/scan/analysis", nlohmann::json::object());
        expect_equal(disabled_analysis["_http_status"].get<long>(), 200L, "disabled scan analysis status");
        expect_equal(disabled_analysis["status"].get<std::string>(), "AI_DISABLED", "disabled scan analysis code");
        expect_true(!disabled_analysis["enabled"].get<bool>(), "disabled scan analysis enabled false");

        const auto patched_config = harness.patch_json("/api/ai/config", {
            {"enabled", true},
            {"mode", "assistive"},
            {"provider", "anthropic"},
            {"model", "claude-test"},
            {"api_key", "must-not-persist"}
        });
        expect_equal(patched_config["_http_status"].get<long>(), 200L, "ai patch status");
        expect_true(patched_config["enabled"].get<bool>(), "ai patch enabled");
        expect_equal(patched_config["mode"].get<std::string>(), "assistive", "ai patch mode");
        expect_equal(patched_config["provider"].get<std::string>(), "anthropic", "ai patch provider");
        expect_equal(patched_config["model"].get<std::string>(), "claude-test", "ai patch model");
        expect_true(!patched_config.contains("api_key"), "ai patch never returns api key");
        const auto stored_config = nlohmann::json::parse(slurp_file(harness.runtime_dir() / "ai_scan_config.json"));
        expect_true(stored_config["enabled"].get<bool>(), "ai config persisted enabled");
        expect_equal(stored_config["provider"].get<std::string>(), "anthropic", "ai config persisted provider");
        expect_equal(stored_config["model"].get<std::string>(), "claude-test", "ai config persisted model");
        expect_true(!stored_config.contains("api_key"), "ai config never persists api key");

        const auto malformed_config = harness.patch_json("/api/ai/config", {
            {"enabled", "true"},
            {"mode", false},
            {"provider", true},
            {"model", false},
            {"sidecar_url", true}
        });
        expect_equal(malformed_config["_http_status"].get<long>(), 200L, "ai malformed config patch status");
        expect_true(malformed_config["enabled"].get<bool>(), "ai malformed config bool string enabled");
        expect_equal(malformed_config["mode"].get<std::string>(), "assistive", "ai malformed config keeps mode fallback");
        expect_equal(malformed_config["provider"].get<std::string>(), "anthropic", "ai malformed config keeps provider fallback");
        expect_equal(malformed_config["model"].get<std::string>(), "claude-test", "ai malformed config keeps model fallback");

        const auto no_scan_analysis = harness.post_json("/api/scan/analysis", {
            {"scan_result", {{"has_scan", false}}}
        });
        if (no_scan_analysis["_http_status"].get<long>() != 400L) {
            throw TestFailure("no_scan_analysis unexpected response: " + no_scan_analysis.dump());
        }
        expect_equal(no_scan_analysis["_http_status"].get<long>(), 400L,
                     "enabled scan analysis without scan status: " + no_scan_analysis.dump());
        expect_equal(no_scan_analysis["code"].get<std::string>(), "NO_SCAN", "enabled scan analysis without scan code");

        FakeSidecar sidecar({
            {"schema_version", "pi.scan-analysis.v1"},
            {"summary", "fixture analysis"},
            {"confidence", 0.8},
            {"detected_scenarios", nlohmann::json::array()},
            {"recommendations", {
                {
                    {"path", "data.color_mode"},
                    {"value", "MONO"},
                    {"reason", true},
                    {"confidence", "0.9"},
                    {"risk", false},
                    {"evidence", {"scan_metrics.fwhm.median=2.4", true}}
                },
                {
                    {"path", "data.unknown"},
                    {"value", true},
                    {"reason", "fixture unknown path"},
                    {"confidence", 0.7},
                    {"risk", "medium"}
                },
                {
                    {"path", "data.color_mode"},
                    {"value", 123},
                    {"reason", "fixture wrong type"},
                    {"confidence", 0.6},
                    {"risk", "high"}
                }
            }},
            {"warnings", nlohmann::json::array()},
            {"review_required", true}
        });
        sidecar.start();

        const auto sidecar_config = harness.patch_json("/api/ai/config", {
            {"enabled", true},
            {"provider", "fixture"},
            {"model", "fixture/model"},
            {"sidecar_url", sidecar.url()}
        });
        expect_equal(sidecar_config["_http_status"].get<long>(), 200L, "ai sidecar config status");

        const auto analysis = harness.post_json("/api/scan/analysis", {
            {"scan_result", {{"frames_detected", 12}, {"color_mode", "OSC"}}},
            {"base_config", {{"data", {{"color_mode", "OSC"}}}}},
            {"model", false}
        });
        expect_equal(analysis["_http_status"].get<long>(), 200L, "scan ai analysis status");
        expect_equal(analysis["schema_version"].get<std::string>(), "pi.scan-analysis.v1", "scan ai schema");
        expect_equal(static_cast<long>(analysis["validated_updates"].size()), 1L, "scan ai validated update count");
        expect_equal(static_cast<long>(analysis["rejected_updates"].size()), 2L, "scan ai rejected update count");
        expect_equal(analysis["validated_updates"][0]["path"].get<std::string>(), "data.color_mode", "scan ai validated path");
        expect_equal(analysis["validated_updates"][0]["reason"].get<std::string>(), "true", "scan ai coerces boolean reason");
        expect_equal(analysis["validated_updates"][0]["risk"].get<std::string>(), "false", "scan ai coerces boolean risk");
        expect_equal(static_cast<long>(analysis["validated_updates"][0]["evidence"].size()), 2L, "scan ai preserves evidence");
        expect_equal(analysis["validation"]["valid"].get<bool>() ? "true" : "false", "true", "scan ai validation ok");

        const std::string analysis_id = analysis["analysis_id"].get<std::string>();
        const auto apply = harness.post_json("/api/scan/analysis/apply", {
            {"analysis_id", analysis_id},
            {"base_config", {{"data", {{"color_mode", "OSC"}}}}},
            {"selected_paths", {"data.color_mode"}},
            {"persist", true}
        });
        expect_equal(apply["_http_status"].get<long>(), 200L, "scan ai apply status");
        expect_true(apply["ok"].get<bool>(), "scan ai apply ok");
        expect_equal(apply["config"]["data"]["color_mode"].get<std::string>(), "MONO", "scan ai apply config value");
        expect_equal(static_cast<long>(apply["applied_paths"].size()), 1L, "scan ai apply selected count");
        expect_true(apply.contains("revision_id"), "scan ai apply creates revision");

        const auto missing_apply = harness.post_json("/api/scan/analysis/apply", nlohmann::json::object());
        expect_equal(missing_apply["_http_status"].get<long>(), 400L, "scan ai apply missing id status");

        const auto models = harness.get_json("/api/ai/models");
        expect_equal(models["_http_status"].get<long>(), 200L, "ai models unavailable status is non-fatal");
        expect_true(!models["available"].get<bool>(), "ai models unavailable flag");
        expect_equal(models["error"]["code"].get<std::string>(), "AI_AGENT_UNAVAILABLE", "ai models unavailable code");
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
