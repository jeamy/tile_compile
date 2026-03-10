#include "backend_test_harness.hpp"

#include <curl/curl.h>
#include <arpa/inet.h>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <cstdio>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

size_t curl_write(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

int reserve_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw TestFailure("failed to create socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(0);
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        throw TestFailure("failed to bind ephemeral port");
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
        close(fd);
        throw TestFailure("failed to read ephemeral port");
    }
    const int port = ntohs(addr.sin_port);
    close(fd);
    return port;
}

HttpResponse http_request(const std::string& method,
                          const std::string& url,
                          const std::optional<std::string>& body = std::nullopt) {
    CURL* curl = curl_easy_init();
    if (!curl) throw TestFailure("curl_easy_init failed");

    std::string response_body;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    struct curl_slist* headers = nullptr;
    if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        if (body.has_value()) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body->c_str());
            headers = curl_slist_append(headers, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        }
    }

    CURLcode rc = curl_easy_perform(curl);
    long status_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        throw TestFailure(std::string("HTTP request failed: ") + curl_easy_strerror(rc));
    }
    return {status_code, response_body};
}

fs::path make_temp_dir() {
    char tmpl[] = "/tmp/web_backend_cpp_tests_XXXXXX";
    char* path = mkdtemp(tmpl);
    if (!path) throw TestFailure("mkdtemp failed");
    return fs::path(path);
}

}  // namespace

BackendHarness::BackendHarness(std::string backend_path,
                               std::string cli_path,
                               std::string runner_path,
                               std::string repo_root)
    : _backend_path(std::move(backend_path)),
      _cli_path(std::move(cli_path)),
      _runner_path(std::move(runner_path)),
      _repo_root(fs::path(std::move(repo_root))) {}

BackendHarness::~BackendHarness() {
    stop();
}

std::string BackendHarness::base_url() const {
    return "http://127.0.0.1:" + std::to_string(_port);
}

void BackendHarness::start() {
    stop();

    _temp_dir = make_temp_dir();
    _fixture_root = _temp_dir / "fixture";
    _runs_dir = _fixture_root / "runs";
    _presets_dir = _fixture_root / "presets";
    _runtime_dir = _fixture_root / "runtime";
    _config_path = _fixture_root / "config.yaml";
    _log_path = _temp_dir / "backend.log";
    _port = reserve_port();

    fs::create_directories(_runs_dir);
    fs::create_directories(_presets_dir);
    fs::create_directories(_runtime_dir);
    make_file("config.yaml", "data:\n  color_mode: OSC\n");
    make_file("presets/tile_compile.test.example.yaml", "data:\n  color_mode: MONO\n");

    _pid = fork();
    if (_pid < 0) throw TestFailure("fork failed");

    if (_pid == 0) {
        int log_fd = ::open(_log_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (log_fd >= 0) {
            dup2(log_fd, STDOUT_FILENO);
            dup2(log_fd, STDERR_FILENO);
            close(log_fd);
        }

        setenv("TILE_COMPILE_PROJECT_ROOT", _repo_root.c_str(), 1);
        setenv("TILE_COMPILE_RUNS_DIR", _runs_dir.c_str(), 1);
        setenv("TILE_COMPILE_CONFIG", _config_path.c_str(), 1);
        setenv("TILE_COMPILE_PRESETS_DIR", _presets_dir.c_str(), 1);
        setenv("TILE_COMPILE_RUNTIME_DIR", _runtime_dir.c_str(), 1);
        setenv("TILE_COMPILE_UI_DIR", (_repo_root / "web_frontend").c_str(), 1);
        setenv("TILE_COMPILE_CLI", _cli_path.c_str(), 1);
        setenv("TILE_COMPILE_RUNNER", _runner_path.c_str(), 1);
        setenv("TILE_COMPILE_HOST", "127.0.0.1", 1);
        setenv("TILE_COMPILE_PORT", std::to_string(_port).c_str(), 1);
        setenv("TILE_COMPILE_ALLOWED_ROOTS", (_fixture_root.string() + ":" + _repo_root.string()).c_str(), 1);
        setenv("TILE_COMPILE_INPUT_SEARCH_ROOTS", _fixture_root.c_str(), 1);
        chdir(_repo_root.c_str());
        execl(_backend_path.c_str(), _backend_path.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    wait_until_ready();
}

void BackendHarness::stop() {
    if (_pid > 0) {
        kill(_pid, SIGTERM);
        int status = 0;
        for (int i = 0; i < 50; ++i) {
            pid_t rc = waitpid(_pid, &status, WNOHANG);
            if (rc == _pid) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (waitpid(_pid, &status, WNOHANG) == 0) {
            kill(_pid, SIGKILL);
            waitpid(_pid, &status, 0);
        }
        _pid = -1;
    }
    if (!_temp_dir.empty()) {
        std::error_code ec;
        fs::remove_all(_temp_dir, ec);
        _temp_dir.clear();
    }
}

void BackendHarness::wait_until_ready() const {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (std::chrono::steady_clock::now() < deadline) {
        int status = 0;
        pid_t rc = waitpid(_pid, &status, WNOHANG);
        if (rc == _pid) {
            throw TestFailure("backend exited early:\n" + slurp_file(_log_path));
        }
        try {
            HttpResponse response = get("/api/health");
            if (response.status_code == 200) return;
        } catch (...) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw TestFailure("backend did not become ready:\n" + slurp_file(_log_path));
}

HttpResponse BackendHarness::get(const std::string& path) const {
    return http_request("GET", base_url() + path);
}

HttpResponse BackendHarness::post(const std::string& path, const nlohmann::json& payload) const {
    return http_request("POST", base_url() + path, payload.dump());
}

nlohmann::json BackendHarness::get_json(const std::string& path) const {
    HttpResponse response = get(path);
    auto parsed = nlohmann::json::parse(response.body, nullptr, false);
    if (parsed.is_discarded()) throw TestFailure("invalid JSON response for " + path + ": " + response.body);
    parsed["_http_status"] = response.status_code;
    return parsed;
}

nlohmann::json BackendHarness::post_json(const std::string& path, const nlohmann::json& payload) const {
    HttpResponse response = post(path, payload);
    auto parsed = nlohmann::json::parse(response.body, nullptr, false);
    if (parsed.is_discarded()) throw TestFailure("invalid JSON response for " + path + ": " + response.body);
    parsed["_http_status"] = response.status_code;
    return parsed;
}

nlohmann::json BackendHarness::wait_for_job(const std::string& job_id, double timeout_s) const {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int>(timeout_s * 1000.0));
    nlohmann::json last_payload = nlohmann::json::object();
    while (std::chrono::steady_clock::now() < deadline) {
        last_payload = get_json("/api/jobs/" + job_id);
        const std::string state = last_payload.value("state", "");
        if (state == "ok" || state == "error" || state == "cancelled") return last_payload;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw TestFailure("job did not finish in time: " + last_payload.dump(2));
}

fs::path BackendHarness::create_run(const std::string& run_id,
                                    const std::vector<nlohmann::json>& events,
                                    const std::string& color_mode,
                                    const std::string& log_name) {
    const fs::path run_dir = _runs_dir / run_id;
    const fs::path logs_dir = run_dir / "logs";
    fs::create_directories(logs_dir);
    std::ofstream config(run_dir / "config.yaml");
    config << "data:\n  color_mode: " << color_mode << "\n";
    std::ofstream out(logs_dir / log_name);
    for (const auto& event : events) out << event.dump() << "\n";
    return run_dir;
}

fs::path BackendHarness::make_file(const std::string& relative_path, const std::string& content) {
    const fs::path path = _fixture_root / relative_path;
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << content;
    return path;
}

fs::path BackendHarness::make_binary_file(const std::string& relative_path, const std::string& content) {
    const fs::path path = _fixture_root / relative_path;
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    return path;
}

std::string slurp_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void expect_true(bool condition, const std::string& message) {
    if (!condition) throw TestFailure(message);
}

void expect_equal(const std::string& actual, const std::string& expected, const std::string& message) {
    if (actual != expected) {
        throw TestFailure(message + " (actual='" + actual + "', expected='" + expected + "')");
    }
}

void expect_equal(long actual, long expected, const std::string& message) {
    if (actual != expected) {
        throw TestFailure(message + " (actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected) + ")");
    }
}

void expect_equal(double actual, double expected, const std::string& message, double epsilon) {
    if (std::abs(actual - expected) > epsilon) {
        throw TestFailure(message);
    }
}

void expect_json_field(const nlohmann::json& payload, const std::string& key, const std::string& message) {
    if (!payload.contains(key)) throw TestFailure(message + " (missing key '" + key + "')");
}
