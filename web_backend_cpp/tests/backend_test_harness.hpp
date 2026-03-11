#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <vector>

struct HttpResponse {
    long status_code{0};
    std::string body;
};

class TestFailure : public std::runtime_error {
public:
    explicit TestFailure(const std::string& message) : std::runtime_error(message) {}
};

class BackendHarness {
public:
    BackendHarness(std::string backend_path,
                   std::string cli_path,
                   std::string runner_path,
                   std::string repo_root);
    ~BackendHarness();

    void start();
    void stop();

    HttpResponse get(const std::string& path) const;
    HttpResponse post(const std::string& path, const nlohmann::json& payload) const;
    nlohmann::json get_json(const std::string& path) const;
    nlohmann::json post_json(const std::string& path, const nlohmann::json& payload) const;
    nlohmann::json wait_for_job(const std::string& job_id, double timeout_s = 10.0) const;

    std::filesystem::path create_run(const std::string& run_id,
                                     const std::vector<nlohmann::json>& events,
                                     const std::string& color_mode = "OSC",
                                     const std::string& log_name = "run_events.jsonl");
    std::filesystem::path make_file(const std::string& relative_path, const std::string& content = "fixture\n");
    std::filesystem::path make_binary_file(const std::string& relative_path, const std::string& content = "fixture\n");

    const std::filesystem::path& fixture_root() const { return _fixture_root; }
    const std::filesystem::path& config_path() const { return _config_path; }

private:
    std::string _backend_path;
    std::string _cli_path;
    std::string _runner_path;
    std::filesystem::path _repo_root;
    std::filesystem::path _temp_dir;
    std::filesystem::path _fixture_root;
    std::filesystem::path _runs_dir;
    std::filesystem::path _presets_dir;
    std::filesystem::path _runtime_dir;
    std::filesystem::path _config_path;
    std::filesystem::path _log_path;
    int _port{0};
    pid_t _pid{-1};

    std::string base_url() const;
    void wait_until_ready() const;
};

std::string slurp_file(const std::filesystem::path& path);

void expect_true(bool condition, const std::string& message);
void expect_equal(const std::string& actual, const std::string& expected, const std::string& message);
void expect_equal(long actual, long expected, const std::string& message);
void expect_equal(double actual, double expected, const std::string& message, double epsilon = 1e-9);
void expect_json_field(const nlohmann::json& payload, const std::string& key, const std::string& message);
