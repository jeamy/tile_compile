#pragma once

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

namespace tile_compile::gui {

class BackendClient;

class CurrentRunTab : public QWidget {
    Q_OBJECT

  public:
    explicit CurrentRunTab(BackendClient *backend, QWidget *parent = nullptr);

    void set_current_run(const QString &run_id, const QString &run_dir);
    QString get_current_run_id() const { return current_run_id_; }
    QString get_current_run_dir() const { return current_run_dir_; }

  signals:
    void resume_run_requested(const QString &run_id);
    void log_message(const QString &msg);

  private slots:
    void on_refresh_status();
    void on_refresh_logs();
    void on_refresh_artifacts();
    void on_resume_run();

  private:
    void build_ui();

    BackendClient *backend_;
    QString current_run_id_;
    QString current_run_dir_;
    
    QLabel *lbl_run_id_ = nullptr;
    QLabel *lbl_run_dir_ = nullptr;
    QPushButton *btn_refresh_status_ = nullptr;
    QPushButton *btn_refresh_logs_ = nullptr;
    QPushButton *btn_refresh_artifacts_ = nullptr;
    QPushButton *btn_resume_run_ = nullptr;
    QSpinBox *logs_tail_ = nullptr;
    QLineEdit *logs_filter_ = nullptr;
    QLabel *current_status_ = nullptr;
    QPlainTextEdit *current_logs_ = nullptr;
    QPlainTextEdit *current_artifacts_ = nullptr;
};

}
