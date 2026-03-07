#include "CurrentRunTab.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QMessageBox>
#include <QProcess>
#include <QDesktopServices>
#include <QUrl>
#include <QFile>
#include <QCoreApplication>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

namespace tile_compile::gui {

CurrentRunTab::CurrentRunTab(QWidget *parent)
    : QWidget(parent) {
    build_ui();
}

void CurrentRunTab::build_ui() {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);

    auto *current_box = new QGroupBox("Current run");
    auto *cur_layout = new QVBoxLayout(current_box);
    cur_layout->setContentsMargins(12, 18, 12, 12);
    cur_layout->setSpacing(10);

    auto *cur_row = new QHBoxLayout();
    lbl_run_id_ = new QLabel("-");
    lbl_run_id_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    lbl_run_dir_ = new QLabel("-");
    lbl_run_dir_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    cur_row->addWidget(new QLabel("run_id"));
    cur_row->addWidget(lbl_run_id_, 1);
    cur_row->addWidget(new QLabel("run_dir"));
    cur_row->addWidget(lbl_run_dir_, 3);
    cur_layout->addLayout(cur_row);

    auto *cur_btns = new QHBoxLayout();
    btn_refresh_status_ = new QPushButton("Refresh status");
    btn_refresh_logs_ = new QPushButton("Refresh logs");
    btn_refresh_artifacts_ = new QPushButton("Refresh artifacts");
    btn_resume_run_ = new QPushButton("Resume from phase...");
    btn_resume_run_->setEnabled(false);
    logs_tail_ = new QSpinBox();
    logs_tail_->setMinimum(1);
    logs_tail_->setMaximum(1000000);
    logs_tail_->setValue(200);
    logs_filter_ = new QLineEdit("");
    logs_filter_->setPlaceholderText("filter text");
    cur_btns->addWidget(btn_refresh_status_);
    cur_btns->addWidget(btn_refresh_logs_);
    cur_btns->addWidget(btn_resume_run_);
    cur_btns->addWidget(new QLabel("Tail"));
    cur_btns->addWidget(logs_tail_);
    cur_btns->addWidget(new QLabel("Filter"));
    cur_btns->addWidget(logs_filter_, 1);
    btn_generate_report_ = new QPushButton("Generate Report");
    cur_btns->addWidget(btn_refresh_artifacts_);
    cur_btns->addWidget(btn_generate_report_);
    cur_layout->addLayout(cur_btns);

    current_status_ = new QLabel("idle");
    current_status_->setObjectName("StatusLabel");
    cur_layout->addWidget(current_status_);

    current_logs_ = new QPlainTextEdit();
    current_logs_->setReadOnly(true);
    cur_layout->addWidget(current_logs_);

    current_artifacts_ = new QPlainTextEdit();
    current_artifacts_->setReadOnly(true);
    cur_layout->addWidget(current_artifacts_);

    layout->addWidget(current_box, 1);

    connect(btn_refresh_status_, &QPushButton::clicked, this, &CurrentRunTab::on_refresh_status);
    connect(btn_refresh_logs_, &QPushButton::clicked, this, &CurrentRunTab::on_refresh_logs);
    connect(btn_refresh_artifacts_, &QPushButton::clicked, this, &CurrentRunTab::on_refresh_artifacts);
    connect(btn_resume_run_, &QPushButton::clicked, this, &CurrentRunTab::on_resume_run);
    connect(btn_generate_report_, &QPushButton::clicked, this, &CurrentRunTab::on_generate_report);
}

void CurrentRunTab::set_current_run(const QString &run_id, const QString &run_dir) {
    current_run_id_ = run_id;
    current_run_dir_ = run_dir;
    lbl_run_id_->setText(run_id.isEmpty() ? "-" : run_id);
    lbl_run_dir_->setText(run_dir.isEmpty() ? "-" : run_dir);
    btn_resume_run_->setEnabled(!run_id.isEmpty());
}

void CurrentRunTab::on_refresh_status() {
    if (current_run_dir_.isEmpty()) {
        QMessageBox::warning(this, "Refresh status", "No run selected");
        return;
    }

    emit log_message("[ui] refresh status clicked");

    const fs::path events_path =
        fs::path(current_run_dir_.toStdString()) / "logs" / "run_events.jsonl";

    if (!fs::exists(events_path)) {
        current_status_->setText("no events file");
        emit log_message("[ui] status: no events file found");
        return;
    }

    std::string status = "unknown";
    std::string last_phase;
    try {
        std::ifstream in(events_path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            try {
                auto ev = nlohmann::json::parse(line);
                const std::string ev_type = ev.value("type", "");
                if (ev_type == "run_start") {
                    status = "running";
                } else if (ev_type == "phase_start") {
                    last_phase = ev.value("phase_name", "");
                    status = "running: " + last_phase;
                } else if (ev_type == "phase_end") {
                    const std::string phase_status = ev.value("status", "ok");
                    last_phase = ev.value("phase_name", "");
                    if (phase_status == "error") {
                        status = "error in " + last_phase;
                    }
                } else if (ev_type == "run_end") {
                    const std::string run_status = ev.value("status", "");
                    if (run_status == "ok" || run_status == "success") {
                        status = "completed";
                    } else {
                        status = "finished: " + run_status;
                    }
                }
            } catch (...) {
                // skip malformed lines
            }
        }
    } catch (const std::exception &e) {
        status = "error reading events";
        emit log_message(QString("[ui] refresh status error: %1").arg(e.what()));
    }

    current_status_->setText(QString::fromStdString(status));
    emit log_message(QString("[ui] status: %1").arg(QString::fromStdString(status)));
}

void CurrentRunTab::on_refresh_logs() {
    if (current_run_dir_.isEmpty()) {
        QMessageBox::warning(this, "Refresh logs", "No run selected");
        return;
    }

    emit log_message("[ui] refresh logs clicked");

    const fs::path events_path =
        fs::path(current_run_dir_.toStdString()) / "logs" / "run_events.jsonl";

    if (!fs::exists(events_path)) {
        current_logs_->setPlainText("(no events file found)");
        return;
    }

    const int tail = logs_tail_->value();
    const QString filter = logs_filter_->text().trimmed();

    try {
        std::ifstream in(events_path);
        std::vector<std::string> all_lines;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (!filter.isEmpty() &&
                line.find(filter.toStdString()) == std::string::npos) {
                continue;
            }
            all_lines.push_back(line);
        }

        // Apply tail
        size_t start = 0;
        if (static_cast<int>(all_lines.size()) > tail) {
            start = all_lines.size() - static_cast<size_t>(tail);
        }

        QString text;
        for (size_t i = start; i < all_lines.size(); ++i) {
            // Format each JSON line for readability
            try {
                auto ev = nlohmann::json::parse(all_lines[i]);
                const std::string ev_type = ev.value("type", "");
                const std::string ts = ev.value("timestamp", "");
                QString formatted;
                if (!ts.empty()) {
                    formatted = QString("[%1] ").arg(QString::fromStdString(ts));
                }
                formatted += QString("[%1]").arg(QString::fromStdString(ev_type));
                for (auto it = ev.begin(); it != ev.end(); ++it) {
                    if (it.key() != "type" && it.key() != "timestamp") {
                        formatted += QString(" %1=%2")
                            .arg(QString::fromStdString(it.key()))
                            .arg(QString::fromStdString(it.value().dump()));
                    }
                }
                text += formatted + "\n";
            } catch (...) {
                text += QString::fromStdString(all_lines[i]) + "\n";
            }
        }

        current_logs_->setPlainText(text);
        emit log_message(QString("[ui] logs refreshed (%1 lines)").arg(
            static_cast<int>(all_lines.size() - start)));
    } catch (const std::exception &e) {
        current_logs_->setPlainText(
            QString("Error reading logs: %1").arg(e.what()));
        emit log_message(
            QString("[ui] refresh logs error: %1").arg(e.what()));
    }
}

void CurrentRunTab::on_refresh_artifacts() {
    if (current_run_dir_.isEmpty()) {
        QMessageBox::warning(this, "Refresh artifacts", "No run selected");
        return;
    }

    emit log_message("[ui] refresh artifacts clicked");

    const fs::path artifacts_dir =
        fs::path(current_run_dir_.toStdString()) / "artifacts";
    const fs::path outputs_dir =
        fs::path(current_run_dir_.toStdString()) / "outputs";

    QString text;

    // List artifacts
    if (fs::exists(artifacts_dir) && fs::is_directory(artifacts_dir)) {
        text += "=== Artifacts ===\n";
        std::vector<std::string> entries;
        for (const auto &entry : fs::directory_iterator(artifacts_dir)) {
            entries.push_back(entry.path().filename().string());
        }
        std::sort(entries.begin(), entries.end());
        for (const auto &name : entries) {
            // Show file size
            const auto fpath = artifacts_dir / name;
            const auto sz = fs::file_size(fpath);
            text += QString("  %1  (%2 bytes)\n")
                .arg(QString::fromStdString(name))
                .arg(sz);
        }
    } else {
        text += "(no artifacts directory)\n";
    }

    text += "\n";

    // List outputs
    if (fs::exists(outputs_dir) && fs::is_directory(outputs_dir)) {
        text += "=== Outputs ===\n";
        std::vector<std::string> entries;
        for (const auto &entry : fs::directory_iterator(outputs_dir)) {
            entries.push_back(entry.path().filename().string());
        }
        std::sort(entries.begin(), entries.end());
        for (const auto &name : entries) {
            const auto fpath = outputs_dir / name;
            const auto sz = fs::file_size(fpath);
            QString size_str;
            if (sz > 1024 * 1024) {
                size_str = QString("%1 MB").arg(
                    static_cast<double>(sz) / (1024.0 * 1024.0), 0, 'f', 1);
            } else if (sz > 1024) {
                size_str = QString("%1 KB").arg(
                    static_cast<double>(sz) / 1024.0, 0, 'f', 1);
            } else {
                size_str = QString("%1 bytes").arg(sz);
            }
            text += QString("  %1  (%2)\n")
                .arg(QString::fromStdString(name))
                .arg(size_str);
        }
    } else {
        text += "(no outputs directory)\n";
    }

    // Show config.yaml existence
    const fs::path config_path =
        fs::path(current_run_dir_.toStdString()) / "config.yaml";
    if (fs::exists(config_path)) {
        text += "\n=== Config ===\n  config.yaml present\n";
    }

    current_artifacts_->setPlainText(text);
    emit log_message("[ui] artifacts refreshed");
}

void CurrentRunTab::on_resume_run() {
    if (current_run_id_.isEmpty()) {
        QMessageBox::warning(this, "Resume run", "No run selected");
        return;
    }

    emit resume_run_requested(current_run_id_);
}

void CurrentRunTab::on_generate_report() {
    if (current_run_dir_.isEmpty()) {
        QMessageBox::warning(this, "Generate Report", "No run selected");
        return;
    }

    emit log_message("[ui] generate report clicked");
    btn_generate_report_->setEnabled(false);
    btn_generate_report_->setText("Generating...");

    // Find generate_report.py relative to the executable
    const QString exe_dir = QCoreApplication::applicationDirPath();
    // Search order: next to exe, then project root (../generate_report.py)
    QString script;
    for (const auto &candidate : {
             exe_dir + "/generate_report.py",
             exe_dir + "/../generate_report.py",
             exe_dir + "/../../generate_report.py",
         }) {
        if (QFile::exists(candidate)) {
            script = candidate;
            break;
        }
    }
    if (script.isEmpty()) {
        emit log_message("[ui] generate_report.py not found near executable");
        btn_generate_report_->setEnabled(true);
        btn_generate_report_->setText("Generate Report");
        QMessageBox::warning(this, "Generate Report",
                             "generate_report.py not found.\n"
                             "Place it next to the executable or in the project root.");
        return;
    }

    auto *proc = new QProcess(this);
    connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this, proc](int exitCode, QProcess::ExitStatus) {
        const QString out = QString::fromUtf8(proc->readAllStandardOutput()).trimmed();
        const QString err = QString::fromUtf8(proc->readAllStandardError()).trimmed();
        if (!err.isEmpty()) {
            emit log_message(QString("[report] stderr: %1").arg(err));
        }

        btn_generate_report_->setEnabled(true);
        btn_generate_report_->setText("Generate Report");

        if (exitCode == 0 && !out.isEmpty()) {
            emit log_message(QString("[report] generated: %1").arg(out));
            // Open in browser
            QDesktopServices::openUrl(QUrl::fromLocalFile(out));
        } else {
            emit log_message(QString("[report] exit code %1").arg(exitCode));
            if (!out.isEmpty()) emit log_message(QString("[report] %1").arg(out));
        }
        proc->deleteLater();
    });

    emit log_message(QString("[report] running: python3 %1 %2").arg(script, current_run_dir_));
    proc->start("python3", {script, current_run_dir_});
}

}
