#include "backend_test_harness.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
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
        if (_thread.joinable()) {
            int fd = socket(AF_INET, SOCK_STREAM, 0);
            if (fd >= 0) {
                sockaddr_in addr{};
                addr.sin_family = AF_INET;
                addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
                addr.sin_port = htons(static_cast<uint16_t>(_port));
                connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
                close(fd);
            }
        }
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

    nlohmann::json request_json() const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!_handled.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        auto parsed = nlohmann::json::parse(_request_body, nullptr, false);
        if (parsed.is_discarded()) return nlohmann::json::object();
        return parsed;
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
            char buffer[32768];
            const ssize_t received = recv(client, buffer, sizeof(buffer) - 1, 0);
            if (received > 0) {
                buffer[received] = '\0';
                const std::string request(buffer, static_cast<size_t>(received));
                const auto body_pos = request.find("\r\n\r\n");
                if (body_pos != std::string::npos) {
                    _request_body = request.substr(body_pos + 4);
                }
            }
            const std::string body = _response.dump();
            const std::string http =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: " + std::to_string(body.size()) + "\r\n"
                "Connection: close\r\n\r\n" + body;
            send(client, http.data(), http.size(), 0);
            close(client);
        }
        _handled.store(true);
        close(fd);
    }

    nlohmann::json _response;
    std::string _request_body;
    int _port{0};
    std::atomic<bool> _ready{false};
    std::atomic<bool> _handled{false};
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

        const auto patched_ui_config = harness.patch_json("/api/ai/config", {
            {"ui", {
                {"mount", "Alt/Az"},
                {"object_type", "Nebel"},
                {"camera", "Mono CMOS"},
                {"calibration_darks", true},
                {"calibration_flats", true},
                {"calibration_bias", false},
                {"notes", "wide nebula test"}
            }}
        });
        expect_equal(patched_ui_config["_http_status"].get<long>(), 200L, "ai ui config patch status");
        expect_equal(patched_ui_config["ui"]["mount"].get<std::string>(), "Alt/Az", "ai ui config mount");
        expect_equal(patched_ui_config["provider"].get<std::string>(), "anthropic", "ai ui config preserves provider");
        const auto reloaded_ui_config = harness.get_json("/api/ai/config");
        expect_equal(reloaded_ui_config["ui"]["object_type"].get<std::string>(), "Nebel", "ai ui config persisted object type");
        expect_true(reloaded_ui_config["ui"]["calibration_flats"].get<bool>(), "ai ui config persisted flats");

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

        const auto memory_dir = harness.fixture_root() / "runs" / ".pi_memory";
        std::filesystem::create_directories(memory_dir);
        {
            std::ofstream legacy(memory_dir / "memories.jsonl");
            legacy << nlohmann::json{
                {"schema_version", "pi.memory.v1"},
                {"memory_id", "legacy_memory_must_be_ignored"},
                {"status", "accepted"},
                {"type", "config_optimization"},
                {"summary", "legacy memory must not enter request context"}
            }.dump() << "\n";
            const nlohmann::json ctx = {
                {"schema_version", "pi.context_signature.v1"},
                {"target", {{"object_type", "galaxy"}}},
                {"acquisition", {{"camera_type", "OSC"}}},
                {"pipeline", {{"affected_paths", nlohmann::json::array({"data.color_mode"})}}}
            };
            const nlohmann::json scope = {
                {"applies_when", nlohmann::json::array({"matching fixture context"})},
                {"does_not_apply_when", nlohmann::json::array({"different color mode problem"})},
                {"confidence", 0.5}
            };
            std::ofstream out(memory_dir / "memories_v2.jsonl");
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_accepted"},
                {"id", "mem_scan_context_accepted"},
                {"status", "candidate"},
                {"type", "config_optimization"},
                {"source", "scan_ai_apply"},
                {"privacy_class", "metadata_only"},
                {"summary", "MONO was useful for this fixture"},
                {"context_signature", ctx},
                {"scope", scope},
                {"config_updates", nlohmann::json::array({{{"path", "data.color_mode"}, {"value", "MONO"}}})},
                {"recommendation", {{"explanation", "MONO was useful for this fixture"}}},
                {"evidence", {{"validation", "fixture"}}},
                {"outcome", {{"validation_valid", true}, {"applied_count", 1}}},
                {"validation", {{"valid", true}}},
                {"review", {{"status", "candidate"}, {"reviewed_by", nullptr}, {"reviewed_at", nullptr}, {"notes", ""}}},
                {"retrieval", {{"keywords", nlohmann::json::array({"data.color_mode"})}, {"negative", false}}}
            }.dump() << "\n";
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_rejected"},
                {"id", "mem_scan_context_rejected"},
                {"status", "candidate"},
                {"type", "config_optimization"},
                {"source", "scan_ai_apply"},
                {"privacy_class", "metadata_only"},
                {"summary", "Rejected memory must not become request context"},
                {"context_signature", ctx},
                {"scope", scope},
                {"config_updates", nlohmann::json::array({{{"path", "data.color_mode"}, {"value", "RGB"}}})},
                {"recommendation", {{"explanation", "Rejected memory must not become request context"}}},
                {"evidence", {{"validation", "fixture"}}},
                {"outcome", {{"validation_valid", false}, {"applied_count", 1}}},
                {"validation", {{"valid", true}}},
                {"review", {{"status", "candidate"}, {"reviewed_by", nullptr}, {"reviewed_at", nullptr}, {"notes", ""}}},
                {"retrieval", {{"keywords", nlohmann::json::array({"data.color_mode"})}, {"negative", false}}}
            }.dump() << "\n";
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_wrong_type"},
                {"id", "mem_scan_context_wrong_type"},
                {"status", "candidate"},
                {"type", "config_optimization"},
                {"source", "scan_ai_apply"},
                {"privacy_class", "metadata_only"},
                {"summary", "Accepted memory with invalid historical value must not bypass schema validation"},
                {"context_signature", ctx},
                {"scope", scope},
                {"config_updates", nlohmann::json::array({{{"path", "data.color_mode"}, {"value", 123}}})},
                {"recommendation", {{"explanation", "Accepted memory with invalid historical value must not bypass schema validation"}}},
                {"evidence", {{"validation", "fixture"}}},
                {"outcome", {{"validation_valid", true}, {"applied_count", 1}}},
                {"validation", {{"valid", true}}},
                {"review", {{"status", "candidate"}, {"reviewed_by", nullptr}, {"reviewed_at", nullptr}, {"notes", ""}}},
                {"retrieval", {{"keywords", nlohmann::json::array({"data.color_mode"})}, {"negative", false}}}
            }.dump() << "\n";
        }
        {
            std::ofstream out(memory_dir / "memory_reviews_v2.jsonl");
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_accepted"},
                {"id", "mem_scan_context_accepted"},
                {"status", "accepted"},
                {"reviewed_at", "2026-07-14T00:00:00Z"},
                {"reviewer", "fixture"},
                {"note", "useful"}
            }.dump() << "\n";
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_rejected"},
                {"id", "mem_scan_context_rejected"},
                {"status", "rejected"},
                {"reviewed_at", "2026-07-14T00:00:01Z"},
                {"reviewer", "fixture"},
                {"note", "bad"}
            }.dump() << "\n";
            out << nlohmann::json{
                {"schema_version", "pi.memory.v2"},
                {"memory_id", "mem_scan_context_wrong_type"},
                {"id", "mem_scan_context_wrong_type"},
                {"status", "accepted"},
                {"reviewed_at", "2026-07-14T00:00:02Z"},
                {"reviewer", "fixture"},
                {"note", "historical context only"}
            }.dump() << "\n";
        }

        const auto analysis = harness.post_json("/api/scan/analysis", {
            {"force", true},
            {"scan_result", {
                {"frames_detected", 12},
                {"color_mode", "OSC"},
                {"frames", nlohmann::json::array({{{"header", {
                    {"OBJECT", "M42"},
                    {"TELESCOP", "RASA 8"},
                    {"INSTRUME", "ASI2600MC"},
                    {"FILTER", "HaOIII"},
                    {"EXPTIME", 180.0},
                    {"DATE-OBS", "2026-01-02T03:04:05"}
                }}}})}
            }},
            {"scan_metrics", {{"frames_total", 12}}},
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
        const auto sidecar_request = sidecar.request_json();
        expect_equal(sidecar_request["ai_request"]["schema_version"].get<std::string>(),
                     "pi.ai-request.v2",
                     "scan ai request includes canonical ai request container");
        expect_equal(sidecar_request["ai_request"]["task"].get<std::string>(),
                     "scan_recommendation",
                     "scan ai canonical request task");
        expect_equal(sidecar_request["ai_request"]["context_signature"]["target"]["object_name"].get<std::string>(),
                     "M42",
                     "scan ai canonical request extracts target from FITS header");
        expect_equal(sidecar_request["ai_request"]["context_signature"]["optics"]["telescope"].get<std::string>(),
                     "RASA 8",
                     "scan ai canonical request extracts telescope from FITS header");
        expect_equal(sidecar_request["ai_request"]["context_signature"]["acquisition"]["filters"][0].get<std::string>(),
                     "HaOIII",
                     "scan ai canonical request extracts filter from FITS header");
        expect_equal(sidecar_request["ai_request"]["context_signature"]["acquisition"]["exposure_seconds"].get<double>(),
                     180.0,
                     "scan ai canonical request extracts exposure from FITS header");
        expect_equal(static_cast<long>(sidecar_request["ai_request"]["positive_memories"].size()), 2L,
                     "scan ai canonical request includes accepted pi memories");
        expect_equal(static_cast<long>(sidecar_request["ai_request"]["negative_memories"].size()), 1L,
                     "scan ai canonical request includes negative pi memories");
        expect_true(sidecar_request["ai_request"].contains("retrieval_coverage_summary"),
                    "scan ai canonical request includes retrieval_coverage_summary prompt section");
        expect_true(sidecar_request["ai_request"]["retrieval_coverage_summary"].is_object(),
                    "scan ai retrieval_coverage_summary is an object");
        expect_true(sidecar_request["ai_request"]["retrieval_coverage_summary"].contains("systemically_missing_context_fields"),
                    "scan ai retrieval_coverage_summary lists systemically_missing_context_fields");
        expect_true(sidecar_request["ai_request"]["retrieval_coverage_summary"].contains("note"),
                    "scan ai retrieval_coverage_summary includes explanatory note for the model");
        expect_equal(static_cast<long>(sidecar_request["session_context"]["accepted_pi_memories"].size()), 2L,
                     "scan ai request includes accepted pi memories");
        bool found_accepted_memory = false;
        bool found_rejected_memory = false;
        for (const auto& memory : sidecar_request["session_context"]["accepted_pi_memories"]) {
            const std::string memory_id = memory.value("memory_id", std::string());
            if (memory_id == "mem_scan_context_accepted") found_accepted_memory = true;
            if (memory_id == "mem_scan_context_rejected") found_rejected_memory = true;
            expect_true(memory.contains("match_explanation"), "accepted memory context includes retrieval explanation");
            expect_true(memory.contains("match_coverage"), "accepted memory context includes retrieval coverage");
        }
        expect_true(found_accepted_memory, "scan ai request includes reviewed accepted memory");
        expect_true(!found_rejected_memory, "scan ai request excludes rejected memories");
        expect_equal(static_cast<long>(sidecar_request["session_context"]["negative_pi_memories"].size()), 1L,
                     "scan ai request includes negative pi memories");
        expect_equal(sidecar_request["session_context"]["negative_pi_memories"][0]["memory_id"].get<std::string>(),
                     "mem_scan_context_rejected",
                     "scan ai request carries rejected memory as negative signal");
        expect_true(sidecar_request["session_context"]["negative_pi_memories"][0].contains("match_explanation"),
                    "negative memory context includes retrieval explanation");
        bool rejected_wrong_type_from_memory_context = false;
        for (const auto& rejected : analysis["rejected_updates"]) {
            if (rejected.value("path", std::string()) == "data.color_mode" &&
                rejected.value("reject_reason", std::string()) == "wrong_type") {
                rejected_wrong_type_from_memory_context = true;
                break;
            }
        }
        expect_true(rejected_wrong_type_from_memory_context,
                    "accepted memory context cannot bypass config schema validation");
        expect_equal(analysis["action_plan"]["schema_version"].get<std::string>(),
                     "pi.action-plan.v1",
                     "scan ai attaches pi action plan");
        expect_true(analysis["action_plan_validation"]["valid"].get<bool>(),
                    "scan ai action plan validates");

        const auto context_store = harness.post_json("/api/scan/analysis/store", {
            {"analysis", {
                {"schema_version", "pi.scan-analysis.v1"},
                {"summary", "fixture context analysis"},
                {"confidence", 0.8},
                {"detected_scenarios", {"large_frame_count"}},
                {"recommendations", {
                    {
                        {"path", "aqmh.cherry_pick.enabled"},
                        {"value", true},
                        {"reason", "fixture cherry pick"},
                        {"confidence", 0.9},
                        {"risk", "low"},
                        {"evidence", {"scan_metrics.fwhm.spread"}}
                    },
                    {
                        {"path", "aqmh.cherry_pick.k_frac"},
                        {"value", 0.88},
                        {"reason", "fixture invalid high k_frac"},
                        {"confidence", 0.9},
                        {"risk", "low"},
                        {"evidence", {"scan_metrics.frame_count=610"}}
                    },
                    {
                        {"path", "aqmh.storage.resolution_divisor"},
                        {"value", 2},
                        {"reason", "fixture invalid downsampled maps with cherry-pick"},
                        {"confidence", 0.9},
                        {"risk", "low"},
                        {"evidence", {"scan_metrics.frame_count=610"}}
                    }
                }},
                {"warnings", nlohmann::json::array()},
                {"review_required", false}
            }},
            {"scan_result", {
                {"frames_detected", 610},
                {"input_path", "/fixture/m42"},
                {"frames", nlohmann::json::array({{{"target", "M42"}}})}
            }},
            {"scan_metrics", {
                {"ok", true},
                {"sample_count", 122},
                {"frames_total", 610},
                {"sampling", {
                    {"strategy", "stratified_header_edges_even_fill"},
                    {"sample_target", 122},
                    {"selected_indices", {0, 1, 2, 607, 608, 609}}
                }},
                {"aggregate", {
                    {"fwhm", {{"median", 9.1}, {"p10", 8.9}, {"p90", 10.0}, {"count", 122}}}
                }},
                {"frames", {
                    {
                        {"index", 0},
                        {"sample_reasons", {"edge_start"}},
                        {"fwhm", 9.1},
                        {"header", {{"target", "M42"}}}
                    }
                }}
            }},
            {"base_config", {
                {"aqmh", {
                    {"storage", {{"resolution_divisor", 2}}},
                    {"cherry_pick", {{"enabled", false}, {"k_frac", 0.3}}}
                }}
            }},
            {"config_schema", {
                {"aqmh.cherry_pick.k_frac", {{"type", "number"}, {"maximum", 1}}},
                {"aqmh.storage.resolution_divisor", {{"type", "integer"}, {"enum", {1, 2, 4}}}}
            }}
        });
        expect_equal(context_store["_http_status"].get<long>(), 200L, "context store status");
        expect_equal(context_store["analysis_context"]["frame_count"].get<long>(), 610L,
                     "context store preserves frame count");
        expect_equal(context_store["analysis_context"]["scan_metrics"]["sampling"]["sample_target"].get<long>(), 122L,
                     "context store preserves sampling target");
        expect_equal(static_cast<long>(context_store["analysis_context"]["scan_metrics"]["sampling"]["selected_indices"].size()), 6L,
                     "context store preserves selected indices");
        expect_equal(context_store["analysis_context"]["base_config"]["aqmh"]["storage"]["resolution_divisor"].get<long>(), 2L,
                     "context store preserves base config");
        expect_true(context_store["analysis_context"]["config_schema"].contains("aqmh.cherry_pick.k_frac"),
                    "context store preserves config schema");

        const auto history = harness.get_json("/api/scan/analysis/history?limit=20");
        expect_equal(history["_http_status"].get<long>(), 200L, "analysis history status");
        std::string context_filename;
        const std::string context_id = context_store["analysis_id"].get<std::string>();
        for (const auto& item : history["items"]) {
            if (item.value("analysis_id", std::string()) == context_id) {
                context_filename = item.value("filename", std::string());
                break;
            }
        }
        expect_true(!context_filename.empty(), "context analysis appears in persisted history");
        const auto context_file = harness.get_json("/api/scan/analysis/history/" + context_filename);
        expect_equal(context_file["_http_status"].get<long>(), 200L, "context persisted file status");
        expect_equal(context_file["analysis_context"]["scan_metrics"]["sampling"]["strategy"].get<std::string>(),
                     "stratified_header_edges_even_fill",
                     "context persisted file preserves sampling strategy");
        expect_true(context_store["action_plan_validation"]["valid"].get<bool>(),
                    "stored scan ai action plan validates");

        const std::string analysis_id = analysis["analysis_id"].get<std::string>();
        const auto apply = harness.post_json("/api/scan/analysis/apply", {
            {"analysis_id", analysis_id},
            {"base_config", {{"data", {{"color_mode", "OSC"}}}}},
            {"selected_paths", {"data.color_mode"}},
            {"persist", true},
            {"learn", true}
        });
        expect_equal(apply["_http_status"].get<long>(), 200L, "scan ai apply status");
        expect_true(apply["ok"].get<bool>(), "scan ai apply ok");
        expect_equal(apply["config"]["data"]["color_mode"].get<std::string>(), "MONO", "scan ai apply config value");
        expect_equal(static_cast<long>(apply["applied_paths"].size()), 1L, "scan ai apply selected count");
        expect_true(apply.contains("revision_id"), "scan ai apply creates revision");
        expect_equal(apply["memory"]["type"].get<std::string>(), "config_optimization", "scan ai apply learns memory");
        expect_true(!apply["memory"].value("duplicate", false), "scan ai apply creates context-specific memory candidate");
        expect_equal(apply["memory"]["status"].get<std::string>(), "candidate", "scan ai learned memory starts as candidate");
        expect_true(apply["memory"].contains("context_signature"), "scan ai learned memory records context signature");
        expect_true(apply["memory"].contains("scope"), "scan ai learned memory records scope");
        expect_equal(apply["memory"]["context_signature"]["target"]["object_name"].get<std::string>(),
                     "M42",
                     "scan ai learned memory preserves FITS-derived target");
        expect_equal(apply["memory"]["context_signature"]["acquisition"]["filters"][0].get<std::string>(),
                     "HaOIII",
                     "scan ai learned memory preserves FITS-derived filter");

        const auto rounded_store = harness.post_json("/api/scan/analysis/store", {
            {"analysis", {
                {"schema_version", "pi.scan-analysis.v1"},
                {"summary", "fixture rounded float"},
                {"confidence", 0.8},
                {"detected_scenarios", nlohmann::json::array()},
                {"recommendations", {
                    {
                        {"path", "aqmh.cherry_pick.k_frac"},
                        {"value", 0.29999999999999999},
                        {"reason", "fixture float noise"},
                        {"confidence", 0.9},
                        {"risk", "low"},
                        {"evidence", {"fixture"}}
                    }
                }},
                {"warnings", nlohmann::json::array()},
                {"review_required", false}
            }},
            {"scan_result", {{"frames_detected", 10}}},
            {"base_config", {
                {"aqmh", {
                    {"cherry_pick", {{"enabled", false}, {"k_frac", 0.4}}},
                    {"storage", {{"resolution_divisor", 1}}}
                }}
            }},
            {"config_schema", {
                {"aqmh.cherry_pick.k_frac", {{"type", "number"}, {"maximum", 1}}}
            }}
        });
        expect_equal(rounded_store["_http_status"].get<long>(), 200L, "rounded float store status");
        expect_equal(static_cast<long>(rounded_store["validated_updates"].size()), 1L,
                     "rounded float store validated updates");
        const std::string rounded_id = rounded_store["analysis_id"].get<std::string>();
        const auto rounded_apply = harness.post_json("/api/scan/analysis/apply", {
            {"analysis_id", rounded_id},
            {"base_config", {
                {"aqmh", {
                    {"cherry_pick", {{"enabled", false}, {"k_frac", 0.4}}},
                    {"storage", {{"resolution_divisor", 1}}}
                }}
            }},
            {"selected_paths", {"aqmh.cherry_pick.k_frac"}},
            {"persist", false},
            {"learn", true}
        });
        expect_equal(rounded_apply["_http_status"].get<long>(), 200L, "rounded float apply status");
        const std::string rounded_yaml = rounded_apply["config_yaml"].get<std::string>();
        expect_true(rounded_yaml.find("k_frac: 0.3") != std::string::npos,
                    "rounded float yaml uses compact decimal: " + rounded_yaml);
        expect_true(rounded_yaml.find("0.299999999999999") == std::string::npos,
                    "rounded float yaml omits binary noise: " + rounded_yaml);
        expect_true(rounded_apply["memory"]["outcome"]["validation_valid"].get<bool>(),
                    "learned memory records validation outcome");
        expect_equal(rounded_apply["memory"]["outcome"]["applied_count"].get<long>(), 1L,
                     "learned memory records applied count");
        expect_equal(rounded_apply["memory"]["outcome"]["applied_paths"][0].get<std::string>(),
                     "aqmh.cherry_pick.k_frac",
                     "learned memory records applied path");

        const auto missing_apply = harness.post_json("/api/scan/analysis/apply", nlohmann::json::object());
        expect_equal(missing_apply["_http_status"].get<long>(), 400L, "scan ai apply missing id status");

        FakeSidecar account_sidecar({
            {"schema_version", "pi.account-status.v1"},
            {"privacy_class", "metadata_only"},
            {"provider", "openai"},
            {"selected", {
                {"provider", "openai"},
                {"key_configured", true},
                {"auth_source", "env"},
                {"credit_query_supported", false},
                {"subscription_query_supported", false},
                {"billing_url", "https://platform.openai.com/settings/organization/billing/overview"}
            }},
            {"providers", nlohmann::json::array()}
        });
        account_sidecar.start();
        const auto account_config = harness.patch_json("/api/ai/config", {
            {"sidecar_url", account_sidecar.url()}
        });
        expect_equal(account_config["_http_status"].get<long>(), 200L, "ai account sidecar config status");
        const auto account = harness.get_json("/api/ai/account?provider=openai");
        expect_equal(account["_http_status"].get<long>(), 200L, "ai account status route");
        expect_equal(account["schema_version"].get<std::string>(), "pi.account-status.v1", "ai account schema");
        expect_equal(account["selected"]["provider"].get<std::string>(), "openai", "ai account selected provider");
        expect_true(!account["selected"]["credit_query_supported"].get<bool>(),
                    "ai account does not claim automatic credit support");

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
