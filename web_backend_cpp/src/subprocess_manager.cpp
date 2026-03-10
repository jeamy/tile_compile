#include "subprocess_manager.hpp"
#include <sstream>
#include <array>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#  include <sys/wait.h>
#  include <fcntl.h>
#  include <signal.h>
#endif

SubprocessResult run_subprocess(const std::vector<std::string>& args,
                                const std::string& cwd,
                                const std::string& stdin_text) {
    SubprocessResult res;
    if (args.empty()) { res.exit_code = -1; return res; }

#ifdef _WIN32
    std::string cmd;
    for (auto& a : args) { cmd += "\"" + a + "\" "; }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE hStdoutR, hStdoutW, hStderrR, hStderrW, hStdinR, hStdinW;
    CreatePipe(&hStdoutR, &hStdoutW, &sa, 0);
    CreatePipe(&hStderrR, &hStderrW, &sa, 0);
    CreatePipe(&hStdinR, &hStdinW, &sa, 0);
    SetHandleInformation(hStdoutR, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(hStderrR, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(hStdinW, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.hStdOutput = hStdoutW;
    si.hStdError  = hStderrW;
    si.hStdInput  = hStdinR;
    si.dwFlags |= STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi{};
    bool ok = CreateProcessA(nullptr, cmd.data(), nullptr, nullptr,
                             TRUE, 0, nullptr,
                             cwd.empty() ? nullptr : cwd.c_str(),
                             &si, &pi);
    CloseHandle(hStdoutW);
    CloseHandle(hStderrW);
    CloseHandle(hStdinR);

    if (!ok) { CloseHandle(hStdinW); res.exit_code = -1; return res; }

    if (!stdin_text.empty()) {
        DWORD written = 0;
        WriteFile(hStdinW, stdin_text.data(), static_cast<DWORD>(stdin_text.size()), &written, nullptr);
    }
    CloseHandle(hStdinW);

    auto read_pipe = [](HANDLE h) {
        std::string out;
        char buf[4096];
        DWORD n;
        while (ReadFile(h, buf, sizeof(buf), &n, nullptr) && n > 0)
            out.append(buf, n);
        return out;
    };
    res.stdout_str = read_pipe(hStdoutR);
    res.stderr_str = read_pipe(hStderrR);
    CloseHandle(hStdoutR);
    CloseHandle(hStderrR);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ec;
    GetExitCodeProcess(pi.hProcess, &ec);
    res.exit_code = (int)ec;
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

#else
    int pfd_out[2], pfd_err[2], pfd_in[2];
    if (pipe(pfd_out) || pipe(pfd_err) || pipe(pfd_in)) { res.exit_code = -1; return res; }

    pid_t pid = fork();
    if (pid < 0) { res.exit_code = -1; return res; }

    if (pid == 0) {
        close(pfd_out[0]); close(pfd_err[0]); close(pfd_in[1]);
        dup2(pfd_out[1], STDOUT_FILENO);
        dup2(pfd_err[1], STDERR_FILENO);
        dup2(pfd_in[0], STDIN_FILENO);
        close(pfd_out[1]); close(pfd_err[1]);
        close(pfd_in[0]);

        if (!cwd.empty()) chdir(cwd.c_str());

        std::vector<const char*> argv;
        for (auto& a : args) argv.push_back(a.c_str());
        argv.push_back(nullptr);
        execvp(argv[0], const_cast<char* const*>(argv.data()));
        _exit(127);
    }

    close(pfd_out[1]); close(pfd_err[1]); close(pfd_in[0]);

    if (!stdin_text.empty()) {
        ssize_t total = 0;
        while (total < static_cast<ssize_t>(stdin_text.size())) {
            ssize_t n = write(pfd_in[1], stdin_text.data() + total, stdin_text.size() - static_cast<size_t>(total));
            if (n <= 0) break;
            total += n;
        }
    }
    close(pfd_in[1]);

    auto drain = [](int fd) {
        std::string out;
        char buf[4096];
        ssize_t n;
        while ((n = read(fd, buf, sizeof(buf))) > 0)
            out.append(buf, (size_t)n);
        close(fd);
        return out;
    };
    res.stdout_str = drain(pfd_out[0]);
    res.stderr_str = drain(pfd_err[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    res.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif

    return res;
}

std::string SubprocessManager::launch(const std::string& type,
                                      const std::vector<std::string>& args,
                                      const std::string& cwd,
                                      const std::string& run_id) {
    std::string job_id = _store.create(type, run_id);
    _store.update_state(job_id, JobState::running);

    auto proc = std::make_shared<BackgroundProcess>();
    proc->job_id = job_id;

    {
        std::lock_guard<std::mutex> lk(_procs_mutex);
        _procs[job_id] = proc;
    }

    proc->thread = std::thread([this, job_id, args, cwd, proc]() {
        auto res = run_subprocess(args, cwd);
        nlohmann::json data = {
            {"stdout",    res.stdout_str},
            {"stderr",    res.stderr_str},
            {"exit_code", res.exit_code},
        };
        if (proc->cancelled.load()) {
            _store.update_state(job_id, JobState::cancelled, data);
        } else if (res.exit_code == 0) {
            try {
                auto j = nlohmann::json::parse(res.stdout_str);
                data["result"] = j;
            } catch (...) {}
            _store.update_state(job_id, JobState::ok, data);
        } else {
            _store.update_state(job_id, JobState::error, data,
                                res.stderr_str.empty() ? "exit " + std::to_string(res.exit_code)
                                                        : res.stderr_str.substr(0, 256));
        }
        std::lock_guard<std::mutex> lk(_procs_mutex);
        _procs.erase(job_id);
    });
    proc->thread.detach();
    return job_id;
}

bool SubprocessManager::cancel(const std::string& job_id) {
    std::lock_guard<std::mutex> lk(_procs_mutex);
    auto it = _procs.find(job_id);
    if (it == _procs.end()) return _store.cancel(job_id);
    it->second->cancelled.store(true);
    _store.cancel(job_id);
    return true;
}

void SubprocessManager::cancel_by_run(const std::string& run_id) {
    auto jobs = _store.list(500);
    for (auto& j : jobs) {
        if (j.run_id == run_id &&
            (j.state == JobState::running || j.state == JobState::pending)) {
            cancel(j.job_id);
        }
    }
}
