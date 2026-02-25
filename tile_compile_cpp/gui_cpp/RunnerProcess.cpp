#include "RunnerProcess.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace tile_compile::gui {

RunnerProcess::RunnerProcess(QObject *parent) : QObject(parent) {
    proc_ = new QProcess(this);

#ifdef _WIN32
    proc_->setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments *args) {
        args->flags |= CREATE_NO_WINDOW;
    });
#endif
    
    connect(proc_, &QProcess::readyReadStandardOutput, this, &RunnerProcess::handle_stdout);
    connect(proc_, &QProcess::readyReadStandardError, this, &RunnerProcess::handle_stderr);
    connect(proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            [this](int exit_code, QProcess::ExitStatus) {
                emit finished(exit_code);
            });
}

bool RunnerProcess::is_running() const {
    return proc_ && proc_->state() == QProcess::Running;
}

void RunnerProcess::start(const QStringList &cmd, const QString &cwd, const QString &stdin_data) {
    if (is_running()) {
        throw std::runtime_error("runner already running");
    }

    if (cmd.isEmpty()) {
        throw std::runtime_error("empty command");
    }

    proc_->setWorkingDirectory(cwd);
    proc_->setProcessChannelMode(QProcess::SeparateChannels);

    const QString program = cmd[0];
    const QStringList args = cmd.mid(1);

    proc_->start(program, args);

    if (!proc_->waitForStarted(3000)) {
        throw std::runtime_error("failed to start runner process");
    }

    if (!stdin_data.isEmpty()) {
        const QByteArray bytes = stdin_data.toUtf8();
        qint64 written = 0;
        while (written < bytes.size()) {
            const qint64 w = proc_->write(bytes.constData() + written, bytes.size() - written);
            if (w <= 0) break;
            written += w;
            proc_->waitForBytesWritten(3000);
        }
        proc_->closeWriteChannel();
    }
}

void RunnerProcess::stop() {
    if (!proc_ || proc_->state() == QProcess::NotRunning) {
        return;
    }
    
    proc_->terminate();
    
    if (!proc_->waitForFinished(2000)) {
        proc_->kill();
        proc_->waitForFinished(1000);
    }
}

void RunnerProcess::handle_stdout() {
    stdout_buffer_ += QString::fromUtf8(proc_->readAllStandardOutput());
    
    while (true) {
        const int idx = stdout_buffer_.indexOf('\n');
        if (idx < 0) break;
        
        const QString line = stdout_buffer_.left(idx);
        stdout_buffer_ = stdout_buffer_.mid(idx + 1);
        
        if (!line.isEmpty()) {
            emit stdout_line(line);
        }
    }
}

void RunnerProcess::handle_stderr() {
    stderr_buffer_ += QString::fromUtf8(proc_->readAllStandardError());
    
    while (true) {
        const int idx = stderr_buffer_.indexOf('\n');
        if (idx < 0) break;
        
        const QString line = stderr_buffer_.left(idx);
        stderr_buffer_ = stderr_buffer_.mid(idx + 1);
        
        if (!line.isEmpty()) {
            emit stderr_line(line);
        }
    }
}

}
