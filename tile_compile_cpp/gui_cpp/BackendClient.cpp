#include "BackendClient.hpp"

#include <QCoreApplication>
#include <QDir>
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
        const QString backend_q = QString::fromStdString(backend_bin);
        const QString app_dir = QCoreApplication::applicationDirPath();
        const QString project_root_q = QString::fromStdString(project_root_);

        QStringList candidates;
        auto push_candidate = [&candidates](const QString& p) {
            const QString clean = QDir::cleanPath(p);
            if (!clean.isEmpty() && !candidates.contains(clean)) candidates.push_back(clean);
        };

        const QFileInfo backend_info(backend_q);
        if (backend_info.isAbsolute()) {
            push_candidate(backend_q);
        } else {
            QString stripped = backend_q;
            if (stripped.startsWith("./") || stripped.startsWith(".\\")) {
                stripped = stripped.mid(2);
            }
            push_candidate(QDir(app_dir).absoluteFilePath(backend_q));
            push_candidate(QDir(app_dir).absoluteFilePath(stripped));
            push_candidate(QDir(project_root_q).absoluteFilePath(backend_q));
            push_candidate(QDir(project_root_q).absoluteFilePath(stripped));
            // Windows release layout: constants are often one level above bin/.
            push_candidate(QDir(project_root_q + "/bin").absoluteFilePath(stripped));
        }

#ifdef _WIN32
        QStringList exe_candidates;
        for (const QString& c : candidates) {
            if (!c.endsWith(".exe", Qt::CaseInsensitive)) {
                exe_candidates.push_back(c + ".exe");
            }
        }
        for (const QString& c : exe_candidates) {
            push_candidate(c);
        }
#endif

        for (const QString& c : candidates) {
            if (QFileInfo::exists(c)) {
                return {QDir::toNativeSeparators(c).toStdString()};
            }
        }

        if (!candidates.isEmpty()) {
            return {QDir::toNativeSeparators(candidates.front()).toStdString()};
        }
    }

    throw std::runtime_error("backend_bin not configured in constants.js");
}

nlohmann::json BackendClient::run_json(const std::string &cwd,
                                       const std::vector<std::string> &args,
                                       const std::string &stdin_text,
                                       int timeout_ms) const {
    const auto cmd = resolve_backend_cmd();
    std::string exe = cmd[0];

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
        args->flags &= ~CREATE_NEW_CONSOLE;
        args->startupInfo->dwFlags |= STARTF_USESHOWWINDOW;
        args->startupInfo->wShowWindow = SW_HIDE;
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
