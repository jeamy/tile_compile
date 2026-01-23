#include "HistoryTab.hpp"
#include "../BackendClient.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QMessageBox>
#include <thread>

namespace tile_compile::gui {

HistoryTab::HistoryTab(BackendClient *backend, QWidget *parent)
    : QWidget(parent), backend_(backend) {
    build_ui();
}

void HistoryTab::build_ui() {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);
    
    auto *history_box = new QGroupBox("Run history");
    auto *hist_layout = new QVBoxLayout(history_box);
    hist_layout->setContentsMargins(12, 18, 12, 12);
    hist_layout->setSpacing(10);
    
    auto *hist_btns = new QHBoxLayout();
    btn_refresh_runs_ = new QPushButton("Refresh");
    hist_btns->addWidget(btn_refresh_runs_);
    hist_btns->addStretch(1);
    hist_layout->addLayout(hist_btns);
    
    runs_table_ = new QTableWidget();
    runs_table_->setColumnCount(6);
    runs_table_->setHorizontalHeaderLabels({"run_id", "timestamp", "status", "frames", "phases", "run_dir"});
    runs_table_->horizontalHeader()->setStretchLastSection(true);
    runs_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    runs_table_->setSelectionMode(QAbstractItemView::SingleSelection);
    runs_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    hist_layout->addWidget(runs_table_);
    
    layout->addWidget(history_box, 1);
    
    connect(btn_refresh_runs_, &QPushButton::clicked, this, &HistoryTab::on_refresh_runs);
    connect(runs_table_, &QTableWidget::cellClicked, this, &HistoryTab::on_cell_clicked);
}

void HistoryTab::on_refresh_runs() {
    emit log_message("[ui] refresh runs clicked");
    btn_refresh_runs_->setEnabled(false);
    
    std::thread([this]() {
        try {
            const auto cli_sub = backend_->constants().value("CLI", nlohmann::json::object())
                .value("sub", nlohmann::json::object());
            std::vector<std::string> args = {
                cli_sub.value("LIST_RUNS", "list-runs")
            };
            
            const auto result = backend_->run_json("", args);
            
            QMetaObject::invokeMethod(this, [this, result]() {
                populate_runs_table(result);
                btn_refresh_runs_->setEnabled(true);
                emit log_message("[ui] runs refreshed");
            }, Qt::QueuedConnection);
            
        } catch (const std::exception &e) {
            QMetaObject::invokeMethod(this, [this, e]() {
                emit log_message(QString("[ui] refresh runs error: %1").arg(e.what()));
                btn_refresh_runs_->setEnabled(true);
            }, Qt::QueuedConnection);
        }
    }).detach();
}

void HistoryTab::on_cell_clicked(int row, int column) {
    if (row < 0 || row >= runs_table_->rowCount()) {
        return;
    }
    
    const QString run_id = runs_table_->item(row, 0) ? runs_table_->item(row, 0)->text() : "";
    const QString run_dir = runs_table_->item(row, 5) ? runs_table_->item(row, 5)->text() : "";
    
    if (!run_id.isEmpty()) {
        emit run_selected(run_id, run_dir);
        emit log_message(QString("[ui] selected run: %1").arg(run_id));
    }
}

void HistoryTab::populate_runs_table(const nlohmann::json &runs) {
    runs_table_->setRowCount(0);
    
    if (!runs.contains("runs") || !runs["runs"].is_array()) {
        return;
    }
    
    const auto &runs_array = runs["runs"];
    runs_table_->setRowCount(static_cast<int>(runs_array.size()));
    
    for (size_t i = 0; i < runs_array.size(); ++i) {
        const auto &run = runs_array[i];
        
        const std::string run_id = run.value("run_id", "");
        const std::string timestamp = run.value("timestamp", "");
        const std::string status = run.value("status", "");
        const int frames = run.value("frames", 0);
        const std::string phases = run.value("phases_completed", "");
        const std::string run_dir = run.value("run_dir", "");
        
        runs_table_->setItem(i, 0, new QTableWidgetItem(QString::fromStdString(run_id)));
        runs_table_->setItem(i, 1, new QTableWidgetItem(QString::fromStdString(timestamp)));
        runs_table_->setItem(i, 2, new QTableWidgetItem(QString::fromStdString(status)));
        runs_table_->setItem(i, 3, new QTableWidgetItem(QString::number(frames)));
        runs_table_->setItem(i, 4, new QTableWidgetItem(QString::fromStdString(phases)));
        runs_table_->setItem(i, 5, new QTableWidgetItem(QString::fromStdString(run_dir)));
    }
    
    runs_table_->resizeColumnsToContents();
}

}
