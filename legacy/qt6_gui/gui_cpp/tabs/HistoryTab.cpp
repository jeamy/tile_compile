#include "HistoryTab.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QMessageBox>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace tile_compile::gui {

namespace {

struct RunInfo {
    std::string run_id;
    std::string run_dir;
    std::string timestamp;
    std::string status;
    int frames = 0;
    int phases_completed = 0;
};

RunInfo scan_run_dir(const fs::path &dir) {
    RunInfo info;
    info.run_id = dir.filename().string();
    info.run_dir = dir.string();

    const fs::path events_path = dir / "logs" / "run_events.jsonl";
    if (!fs::exists(events_path)) {
        info.status = "no events";
        return info;
    }

    std::ifstream in(events_path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        try {
            auto ev = nlohmann::json::parse(line);
            const std::string ev_type = ev.value("type", "");

            if (ev_type == "run_start") {
                info.timestamp = ev.value("timestamp", "");
                info.frames = ev.value("frames_discovered",
                    ev.value("detail", nlohmann::json::object())
                      .value("frames_discovered", 0));
                info.status = "running";
            } else if (ev_type == "phase_end") {
                const std::string phase_status = ev.value("status", "ok");
                if (phase_status == "ok") {
                    info.phases_completed++;
                } else if (phase_status == "error") {
                    info.status = "error";
                }
            } else if (ev_type == "run_end") {
                const std::string run_status = ev.value("status", "");
                if (run_status == "ok" || run_status == "success") {
                    info.status = "completed";
                } else {
                    info.status = run_status.empty() ? "finished" : run_status;
                }
            }
        } catch (...) {
            // skip malformed lines
        }
    }

    return info;
}

} // namespace

HistoryTab::HistoryTab(QWidget *parent)
    : QWidget(parent) {
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
    runs_table_->setHorizontalHeaderLabels(
        {"run_id", "timestamp", "status", "frames", "phases", "run_dir"});
    runs_table_->horizontalHeader()->setStretchLastSection(true);
    runs_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    runs_table_->setSelectionMode(QAbstractItemView::SingleSelection);
    runs_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    hist_layout->addWidget(runs_table_);

    layout->addWidget(history_box, 1);

    connect(btn_refresh_runs_, &QPushButton::clicked, this,
            &HistoryTab::on_refresh_runs);
    connect(runs_table_, &QTableWidget::cellClicked, this,
            &HistoryTab::on_cell_clicked);
}

void HistoryTab::set_runs_dir(const QString &runs_dir) {
    runs_dir_ = runs_dir;
}

void HistoryTab::on_refresh_runs() {
    emit log_message("[ui] refresh runs clicked");

    if (runs_dir_.isEmpty()) {
        emit log_message("[ui] refresh runs: no runs directory configured");
        return;
    }

    const fs::path runs_path(runs_dir_.toStdString());
    if (!fs::exists(runs_path) || !fs::is_directory(runs_path)) {
        emit log_message(
            QString("[ui] runs directory not found: %1").arg(runs_dir_));
        runs_table_->setRowCount(0);
        return;
    }

    // Collect all run directories
    std::vector<RunInfo> runs;
    try {
        for (const auto &entry : fs::directory_iterator(runs_path)) {
            if (!entry.is_directory()) continue;
            // A run directory should have a logs/ subdirectory
            if (!fs::exists(entry.path() / "logs")) continue;
            try {
                runs.push_back(scan_run_dir(entry.path()));
            } catch (...) {
                // skip broken runs
            }
        }
    } catch (const std::exception &e) {
        emit log_message(
            QString("[ui] refresh runs error: %1").arg(e.what()));
        return;
    }

    // Sort by timestamp descending (newest first)
    std::sort(runs.begin(), runs.end(),
              [](const RunInfo &a, const RunInfo &b) {
                  return a.timestamp > b.timestamp;
              });

    // Populate table
    runs_table_->setRowCount(static_cast<int>(runs.size()));
    for (size_t i = 0; i < runs.size(); ++i) {
        const auto &r = runs[i];
        const int row = static_cast<int>(i);
        runs_table_->setItem(
            row, 0,
            new QTableWidgetItem(QString::fromStdString(r.run_id)));
        runs_table_->setItem(
            row, 1,
            new QTableWidgetItem(QString::fromStdString(r.timestamp)));
        runs_table_->setItem(
            row, 2,
            new QTableWidgetItem(QString::fromStdString(r.status)));
        runs_table_->setItem(
            row, 3, new QTableWidgetItem(QString::number(r.frames)));
        runs_table_->setItem(
            row, 4, new QTableWidgetItem(QString::number(r.phases_completed)));
        runs_table_->setItem(
            row, 5,
            new QTableWidgetItem(QString::fromStdString(r.run_dir)));
    }

    runs_table_->resizeColumnsToContents();
    emit log_message(
        QString("[ui] runs refreshed (%1 runs found)").arg(runs.size()));
}

void HistoryTab::on_cell_clicked(int row, int /*column*/) {
    if (row < 0 || row >= runs_table_->rowCount()) {
        return;
    }

    const QString run_id =
        runs_table_->item(row, 0) ? runs_table_->item(row, 0)->text() : "";
    const QString run_dir =
        runs_table_->item(row, 5) ? runs_table_->item(row, 5)->text() : "";

    if (!run_id.isEmpty()) {
        emit run_selected(run_id, run_dir);
        emit log_message(QString("[ui] selected run: %1").arg(run_id));
    }
}

}
