#pragma once
// Process helpers over /proc (Linux) shared by the startup cleanup, the run
// stop route and the run inspector.
#ifndef _WIN32

#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

/// argv of a process from /proc/<pid>/cmdline (empty if unreadable).
inline std::vector<std::string> read_process_argv(const std::filesystem::path& proc_dir) {
    std::ifstream cmdline(proc_dir / "cmdline", std::ios::binary);
    if (!cmdline) return {};
    const std::string raw((std::istreambuf_iterator<char>(cmdline)), std::istreambuf_iterator<char>());
    std::vector<std::string> argv;
    size_t start = 0;
    while (start < raw.size()) {
        size_t end = raw.find('\0', start);
        if (end == std::string::npos) end = raw.size();
        if (end > start) argv.push_back(raw.substr(start, end - start));
        start = end + 1;
    }
    return argv;
}

/// Calls fn(pid, proc_dir, argv) for every other process with a readable,
/// non-empty argv; fn returns false to stop the scan.
template <typename Fn>
void for_each_process(Fn&& fn) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const int self_pid = static_cast<int>(::getpid());
    for (fs::directory_iterator it("/proc", ec), end; !ec && it != end; it.increment(ec)) {
        const std::string pid_text = it->path().filename().string();
        if (pid_text.empty() || pid_text.size() > 9 ||
            !std::all_of(pid_text.begin(), pid_text.end(),
                         [](unsigned char ch) { return std::isdigit(ch) != 0; }))
            continue;
        const int pid = std::stoi(pid_text);
        if (pid == self_pid) continue;
        const std::vector<std::string> argv = read_process_argv(it->path());
        if (argv.empty()) continue;
        if (!fn(pid, it->path(), argv)) return;
    }
}

/// argv joined by spaces.
inline std::string join_argv(const std::vector<std::string>& argv) {
    std::string joined;
    for (const auto& part : argv) {
        if (!joined.empty()) joined += ' ';
        joined += part;
    }
    return joined;
}

/// True if argv[0] is the runner binary (by the configured name or the
/// tile_compile_runner stem).
inline bool argv_is_runner_exe(const std::vector<std::string>& argv, const std::string& runner_name) {
    if (argv.empty()) return false;
    const std::string exe_name = std::filesystem::path(argv.front()).filename().string();
    return exe_name == runner_name || exe_name.find("tile_compile_runner") != std::string::npos;
}

inline bool pid_exists(pid_t pid) {
    if (pid <= 0) return false;
    if (kill(pid, 0) == 0) return true;
    return errno == EPERM;
}

/// SIGTERM to the process group (pgid == pid, falling back to the process),
/// SIGKILL after 2 s. False if no signal could be sent.
inline bool terminate_pid_group(pid_t pid) {
    if (pid <= 0) return false;
    if (kill(-pid, SIGTERM) != 0 && kill(pid, SIGTERM) != 0) return false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (!pid_exists(pid)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (kill(-pid, SIGKILL) != 0 && kill(pid, SIGKILL) != 0) return false;
    return true;
}

#endif  // _WIN32
