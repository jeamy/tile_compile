#pragma once

#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QCheckBox>
#include <string>

namespace tile_compile::gui {

class RunTab : public QWidget {
    Q_OBJECT

  public:
    explicit RunTab(const std::string &project_root, QWidget *parent = nullptr);

    QString get_working_dir() const;
    QString get_input_dir() const;
    QString get_runs_dir() const;
    QString get_pattern() const;
    bool is_dry_run() const;
    
    void set_working_dir(const QString &dir);
    void set_input_dir(const QString &dir);
    void set_runs_dir(const QString &dir);
    void set_pattern(const QString &pattern);
    void set_dry_run(bool dry_run);
    
    void set_start_enabled(bool enabled, const QString &tooltip = QString());
    void set_abort_enabled(bool enabled);
    void set_status_text(const QString &text);
    void show_reduced_mode_hint(int frame_count, int threshold);
    void hide_reduced_mode_hint();

  signals:
    void start_run_clicked();
    void abort_run_clicked();
    void working_dir_changed();
    void input_dir_changed();

  private slots:
    void on_browse_working_dir();
    void on_browse_input_dir();

  private:
    void build_ui();

    std::string project_root_;
    
    QLineEdit *working_dir_ = nullptr;
    QLineEdit *input_dir_ = nullptr;
    QLineEdit *runs_dir_ = nullptr;
    QLineEdit *pattern_ = nullptr;
    QCheckBox *dry_run_ = nullptr;
    QPushButton *btn_start_ = nullptr;
    QPushButton *btn_abort_ = nullptr;
    QLabel *lbl_run_ = nullptr;
    QLabel *run_reduced_mode_hint_ = nullptr;
};

}
