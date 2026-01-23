#pragma once

#include <QWidget>
#include <QPushButton>
#include <QTableWidget>
#include <string>

#include <nlohmann/json.hpp>

namespace tile_compile::gui {

class BackendClient;

class HistoryTab : public QWidget {
    Q_OBJECT

  public:
    explicit HistoryTab(BackendClient *backend, QWidget *parent = nullptr);

  signals:
    void run_selected(const QString &run_id, const QString &run_dir);
    void log_message(const QString &msg);

  private slots:
    void on_refresh_runs();
    void on_cell_clicked(int row, int column);

  private:
    void build_ui();
    void populate_runs_table(const nlohmann::json &runs);

    BackendClient *backend_;
    
    QPushButton *btn_refresh_runs_ = nullptr;
    QTableWidget *runs_table_ = nullptr;
};

}
