#pragma once

#include "backend_test_harness.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <functional>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace backend_test {
namespace detail {

inline int reserve_sidecar_port() {
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

}  // namespace detail

class FakeSidecar {
public:
    explicit FakeSidecar(nlohmann::json response, int status = 200)
        : _response(std::move(response)), _status(status), _port(detail::reserve_sidecar_port()) {}

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
            std::string request;
            char buffer[8192];
            size_t content_length = 0;
            size_t body_pos = std::string::npos;
            while (true) {
                const ssize_t received = recv(client, buffer, sizeof(buffer), 0);
                if (received <= 0) break;
                request.append(buffer, static_cast<size_t>(received));
                if (body_pos == std::string::npos) {
                    body_pos = request.find("\r\n\r\n");
                    if (body_pos != std::string::npos) {
                        const std::string header = request.substr(0, body_pos);
                        std::string header_lower = header;
                        std::transform(header_lower.begin(), header_lower.end(), header_lower.begin(),
                                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                        const std::string marker = "content-length:";
                        const auto cl_pos = header_lower.find(marker);
                        if (cl_pos != std::string::npos) {
                            const auto value_start = cl_pos + marker.size();
                            const auto value_end = header_lower.find("\r\n", value_start);
                            try {
                                content_length = static_cast<size_t>(std::stoul(
                                    header.substr(value_start, value_end - value_start)));
                            } catch (...) {
                                content_length = 0;
                            }
                        }
                    }
                }
                if (body_pos != std::string::npos && content_length > 0 &&
                    request.size() >= body_pos + 4 + content_length) {
                    break;
                }
                if (body_pos != std::string::npos && content_length == 0) break;
            }
            if (body_pos != std::string::npos) {
                _request_body = request.substr(
                    body_pos + 4, content_length > 0 ? content_length : std::string::npos);
            }
            const std::string body = _response.dump();
            const std::string http =
                "HTTP/1.1 " + std::to_string(_status) + (_status >= 400 ? " Error\r\n" : " OK\r\n") +
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
    int _status{200};
    std::string _request_body;
    int _port{0};
    std::atomic<bool> _ready{false};
    std::atomic<bool> _handled{false};
    std::thread _thread;
};

class FakeMultiSidecar {
public:
    using Handler = std::function<std::pair<int, nlohmann::json>(
        const std::string& method, const std::string& path, const nlohmann::json& body)>;

    explicit FakeMultiSidecar(Handler handler)
        : _handler(std::move(handler)), _port(detail::reserve_sidecar_port()) {}

    ~FakeMultiSidecar() { stop(); }

    void start() {
        _stop.store(false);
        _thread = std::thread([this]() { serve_loop(); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!_ready.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!_ready.load()) throw TestFailure("fake multi sidecar did not become ready");
    }

    void stop() {
        _stop.store(true);
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
            _thread.join();
        }
    }

    std::string url() const { return "http://127.0.0.1:" + std::to_string(_port); }

private:
    void serve_loop() {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return;
        int reuse = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(static_cast<uint16_t>(_port));
        if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(fd, 8) != 0) {
            close(fd);
            return;
        }
        _ready.store(true);
        while (!_stop.load()) {
            int client = accept(fd, nullptr, nullptr);
            if (client < 0) break;
            if (_stop.load()) { close(client); break; }
            std::string request;
            char buffer[8192];
            size_t content_length = 0;
            size_t body_pos = std::string::npos;
            while (true) {
                const ssize_t received = recv(client, buffer, sizeof(buffer), 0);
                if (received <= 0) break;
                request.append(buffer, static_cast<size_t>(received));
                if (body_pos == std::string::npos) {
                    body_pos = request.find("\r\n\r\n");
                    if (body_pos != std::string::npos) {
                        std::string header_lower = request.substr(0, body_pos);
                        std::transform(header_lower.begin(), header_lower.end(), header_lower.begin(),
                                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                        const std::string marker = "content-length:";
                        const auto cl_pos = header_lower.find(marker);
                        if (cl_pos != std::string::npos) {
                            const auto value_start = cl_pos + marker.size();
                            const auto value_end = header_lower.find("\r\n", value_start);
                            try {
                                content_length = static_cast<size_t>(std::stoul(
                                    request.substr(value_start, value_end - value_start)));
                            } catch (...) { content_length = 0; }
                        }
                    }
                }
                if (body_pos != std::string::npos && content_length > 0 &&
                    request.size() >= body_pos + 4 + content_length) break;
                if (body_pos != std::string::npos && content_length == 0) break;
            }
            std::string method = "GET";
            std::string path = "/";
            const auto first_line_end = request.find("\r\n");
            if (first_line_end != std::string::npos) {
                const std::string first_line = request.substr(0, first_line_end);
                const auto sp1 = first_line.find(' ');
                if (sp1 != std::string::npos) {
                    method = first_line.substr(0, sp1);
                    const auto sp2 = first_line.find(' ', sp1 + 1);
                    if (sp2 != std::string::npos) path = first_line.substr(sp1 + 1, sp2 - sp1 - 1);
                }
            }
            nlohmann::json req_body = nlohmann::json::object();
            if (body_pos != std::string::npos && content_length > 0) {
                auto parsed = nlohmann::json::parse(request.substr(body_pos + 4, content_length), nullptr, false);
                if (!parsed.is_discarded()) req_body = std::move(parsed);
            }
            auto [status, resp_json] = _handler(method, path, req_body);
            const std::string body = resp_json.dump();
            const std::string http =
                "HTTP/1.1 " + std::to_string(status) + (status >= 400 ? " Error\r\n" : " OK\r\n") +
                "Content-Type: application/json\r\n"
                "Content-Length: " + std::to_string(body.size()) + "\r\n"
                "Connection: close\r\n\r\n" + body;
            send(client, http.data(), http.size(), 0);
            close(client);
        }
        close(fd);
    }

    Handler _handler;
    int _port{0};
    std::atomic<bool> _ready{false};
    std::atomic<bool> _stop{false};
    std::thread _thread;
};

}  // namespace backend_test
