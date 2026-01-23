#include "CurrentRunTab.hpp"
#include "../BackendClient.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QMessageBox>
#include <thread>

namespace tile_compile::gui {

CurrentRunTab::CurrentRunTab(BackendClient *backend, QWidget *parent)
    : QWidget(parent), backend_(backend) {
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
    cur_btns->addWidget(btn_refresh_artifacts_);
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
}

void CurrentRunTab::set_current_run(const QString &run_id, const QString &run_dir) {
    current_run_id_ = run_id;
    current_run_dir_ = run_dir;
    lbl_run_id_->setText(run_id.isEmpty() ? "-" : run_id);
    lbl_run_dir_->setText(run_dir.isEmpty() ? "-" : run_dir);
    btn_resume_run_->setEnabled(!run_id.isEmpty());
}

void CurrentRunTab::on_refresh_status() {
    if (current_run_id_.isEmpty()) {
        QMessageBox::warning(this, "Refresh status", "No run selected");
        return;
    }
    
    emit log_message("[ui] refresh status clicked");
    current_status_->setText("refreshing...");
    
    std::thread([this]() {
        try {
            const auto cli_sub = backend_->constants().value("CLI", nlohmann::json::object())
                .value("sub", nlohmann::json::object());
            std::vector<std::string> args = {
                cli_sub.value("GET_RUN_STATUS", "get-run-status"),
                current_run_id_.toStdString()
            };
            
            const auto result = backend_->run_json("", args);
            
            QMetaObject::invokeMethod(this, [this, result]() {
                const std::string status = result.value("status", "unknown");
                current_status_->setText(QString::fromStdString(status));
                emit log_message(QString("[ui] status: %1").arg(QString::fromStdString(status)));
            }, Qt::QueuedConnection);
            
        } catch (const std::exception &e) {
            QMetaObject::invokeMethod(this, [this, e]() {
                current_status_->setText("error");
                emit log_message(QString("[ui] refresh status error: %1").arg(e.what()));
            }, Qt::QueuedConnection);
        }
    }).detach();
}

void CurrentRunTab::on_refresh_logs() {
    if (current_run_id_.isEmpty()) {
        QMessageBox::warning(this, "Refresh logs", "No run selected");
        return;
    }
    
    emit log_message("[ui] refresh logs clicked");
    
    std::thread([this]() {
        try {
            const auto cli_sub = backend_->constants().value("CLI", nlohmann::json::object())
                .value("sub", nlohmann::json::object());
            std::vector<std::string> args = {
                cli_sub.value("GET_RUN_LOGS", "get-run-logs"),
                current_run_id_.toStdString(),
                "--tail", std::to_string(logs_tail_->value())
            };
            
            const QString filter = logs_filter_->text().trimmed();
            if (!filter.isEmpty()) {
                args.push_back("--filter");
                args.push_back(filter.toStdString());
            }
            
            const auto result = backend_->run_json("", args);
            
            QMetaObject::invokeMethod(this, [this, result]() {
                const std::string logs = result.value("logs", "");
                current_logs_->setPlainText(QString::fromStdString(logs));
                emit log_message("[ui] logs refreshed");
            }, Qt::QueuedConnection);
            
        } catch (const std::exception &e) {
            QMetaObject::invokeMethod(this, [this, e]() {
                emit log_message(QString("[ui] refresh logs error: %1").arg(e.what()));
            }, Qt::QueuedConnection);
        }
    }).detach();
}

void CurrentRunTab::on_refresh_artifacts() {
    if (current_run_id_.isEmpty()) {
        QMessageBox::warning(this, "Refresh artifacts", "No run selected");
        return;
    }
    
    emit log_message("[ui] refresh artifacts clicked");
    
    std::thread([this]() {
        try {
            const auto cli_sub = backend_->constants().value("CLI", nlohmann::json::object())
                .value("sub", nlohmann::json::object());
            std::vector<std::string> args = {
                cli_sub.value("LIST_ARTIFACTS", "list-artifacts"),
                current_run_id_.toStdString()
            };
            
            const auto result = backend_->run_json("", args);
            
            QMetaObject::invokeMethod(this, [this, result]() {
                QString artifacts_text;
                if (result.contains("artifacts") && result["artifacts"].is_array()) {
                    for (const auto &artifact : result["artifacts"]) {
                        if (artifact.is_string()) {
                            artifacts_text += QString::fromStdString(artifact.get<std::string>()) + "\n";
                        }
                    }
                }
                current_artifacts_->setPlainText(artifacts_text);
                emit log_message("[ui] artifacts refreshed");
            }, Qt::QueuedConnection);
            
        } catch (const std::exception &e) {
            QMetaObject::invokeMethod(this, [this, e]() {
                emit log_message(QString("[ui] refresh artifacts error: %1").arg(e.what()));
            }, Qt::QueuedConnection);
        }
    }).detach();
}

void CurrentRunTab::on_resume_run() {
    if (current_run_id_.isEmpty()) {
        QMessageBox::warning(this, "Resume run", "No run selected");
        return;
    }
    
    emit resume_run_requested(current_run_id_);
}

}
