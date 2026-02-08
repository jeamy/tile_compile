#pragma once

#include <QWidget>
#include <QPushButton>
#include <QTableWidget>
#include <string>

namespace tile_compile::gui {

class HistoryTab : public QWidget {
    Q_OBJECT

  public:
    explicit HistoryTab(QWidget *parent = nullptr);

    void set_runs_dir(const QString &runs_dir);

  signals:
    void run_selected(const QString &run_id, const QString &run_dir);
    void log_message(const QString &msg);

  private slots:
    void on_refresh_runs();
    void on_cell_clicked(int row, int column);

  private:
    void build_ui();

    QString runs_dir_;

    QPushButton *btn_refresh_runs_ = nullptr;
    QTableWidget *runs_table_ = nullptr;
};

}
