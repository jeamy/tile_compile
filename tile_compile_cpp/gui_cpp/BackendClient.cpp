#include "BackendClient.hpp"

#include <QFileInfo>
#include <QProcess>
#include <QStringList>

#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

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
    std::string exe = cmd[0];
#ifdef _WIN32
    if (exe.rfind("./", 0) == 0) {
        exe = ".\\" + exe.substr(2);
    }
    if (!exe.empty()) {
        const std::string exe_ext = ".exe";
        const bool has_exe_suffix = exe.size() >= exe_ext.size() &&
                                    exe.substr(exe.size() - exe_ext.size()) == exe_ext;
        if (!has_exe_suffix) {
            const QString qexe = QString::fromStdString(exe);
            if (!QFileInfo::exists(qexe)) {
                const std::string candidate = exe + exe_ext;
                if (QFileInfo::exists(QString::fromStdString(candidate))) {
                    exe = candidate;
                }
            }
        }
    }
#endif

    if (!exe.empty() && !QFileInfo::exists(QString::fromStdString(exe))) {
        throw std::runtime_error(
            "Backend executable not found: " + exe +
            ". Make sure tile_compile_cli(.exe) is next to the GUI executable (or set CLI.backend_bin accordingly)."
        );
    }

    QProcess proc;
    proc.setProcessChannelMode(QProcess::MergedChannels);
    if (!cwd.empty()) {
        proc.setWorkingDirectory(QString::fromStdString(cwd));
    }

#ifdef _WIN32
    proc.setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments *args) {
        args->flags |= CREATE_NO_WINDOW;
    });
#endif

    QStringList qargs;
    for (const auto &arg : args) {
        qargs << QString::fromStdString(arg);
    }

    proc.start(QString::fromStdString(exe), qargs);
    if (!proc.waitForStarted(5000)) {
        throw std::runtime_error("Failed to start backend process: " + exe);
    }

    if (!stdin_text.empty()) {
        proc.write(stdin_text.data(), static_cast<qint64>(stdin_text.size()));
        proc.closeWriteChannel();
    }

    if (!proc.waitForFinished(timeout_ms)) {
        proc.kill();
        proc.waitForFinished(3000);
        throw std::runtime_error("Backend process timeout after " + std::to_string(timeout_ms) + " ms");
    }

    const std::string output = proc.readAllStandardOutput().toStdString();
    const int exit_code = proc.exitCode();
    const QProcess::ExitStatus exit_status = proc.exitStatus();

    std::ostringstream cmd_preview;
    cmd_preview << exe;
    for (const auto &arg : args) {
        cmd_preview << " " << arg;
    }
    
    try {
        return nlohmann::json::parse(output);
    } catch (const nlohmann::json::parse_error &e) {
        std::ostringstream msg;
        msg << "Failed to parse backend JSON (cmd: " << cmd_preview.str() << ")";
        msg << " (exit_code=" << exit_code
            << ", exit_status=" << (exit_status == QProcess::NormalExit ? "normal" : "crash")
            << "): ";
        msg << output.substr(0, 500);
        throw std::runtime_error(msg.str());
    }
}

}
