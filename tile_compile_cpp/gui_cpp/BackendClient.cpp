#include "BackendClient.hpp"

#include <array>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace tile_compile::gui {

BackendClient::BackendClient(std::string project_root, nlohmann::json constants)
    : project_root_(std::move(project_root)), constants_(std::move(constants)) {}

std::vector<std::string> BackendClient::backend_cmd() const {
    return resolve_backend_cmd();
}

std::vector<std::string> BackendClient::resolve_backend_cmd() const {
    const auto cli = constants_.value("CLI", nlohmann::json::object());
    const std::string backend_bin = cli.value("backend_bin", "");
    
    if (!backend_bin.empty()) {
        return {backend_bin};
    }
    
    throw std::runtime_error("backend_bin not configured in constants.js");
}

nlohmann::json BackendClient::run_json(const std::string &cwd,
                                       const std::vector<std::string> &args,
                                       const std::string &stdin_text,
                                       int timeout_ms) const {
    const auto cmd = resolve_backend_cmd();
    
    // Build command with proper shell escaping
    std::string full_cmd;
    if (!stdin_text.empty()) {
        // Use echo with stdin pipe for commands that need stdin
        full_cmd = "echo '";
        // Escape single quotes in stdin_text
        for (char c : stdin_text) {
            if (c == '\'') {
                full_cmd += "'\\''";
            } else {
                full_cmd += c;
            }
        }
        full_cmd += "' | ";
    }
    
    full_cmd += cmd[0];
    for (const auto &arg : args) {
        // Simple shell escaping for arguments
        full_cmd += " '";
        for (char c : arg) {
            if (c == '\'') {
                full_cmd += "'\\''";
            } else {
                full_cmd += c;
            }
        }
        full_cmd += "'";
    }
    
    full_cmd += " 2>&1";
    
    std::string original_cwd;
    if (!cwd.empty()) {
        char buf[4096];
        if (getcwd(buf, sizeof(buf))) {
            original_cwd = buf;
        }
        if (chdir(cwd.c_str()) != 0) {
            throw std::runtime_error("failed to chdir to " + cwd);
        }
    }

    auto restore_cwd = [&original_cwd]() {
        if (!original_cwd.empty()) {
            const int rc = chdir(original_cwd.c_str());
            (void)rc;
        }
    };
    
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(full_cmd.c_str(), "r"), pclose);
    if (!pipe) {
        restore_cwd();
        throw std::runtime_error("popen() failed");
    }
    
    std::ostringstream result;
    std::array<char, 256> buffer;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result << buffer.data();
    }
    
    restore_cwd();
    
    const std::string output = result.str();
    
    try {
        return nlohmann::json::parse(output);
    } catch (const nlohmann::json::parse_error &e) {
        throw std::runtime_error("Failed to parse backend JSON: " + output.substr(0, 500));
    }
}

}
