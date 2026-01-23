#pragma once

#include <QMainWindow>
#include <QTabWidget>
#include <QLineEdit>
#include <QTextEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QTableWidget>
#include <QTimer>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "PhaseProgressWidget.hpp"
#include "AssumptionsWidget.hpp"
#include "BackendClient.hpp"
#include "RunnerProcess.hpp"
#include "tabs/ScanTab.hpp"
#include "tabs/ConfigTab.hpp"
#include "tabs/RunTab.hpp"
#include "tabs/CurrentRunTab.hpp"
#include "tabs/HistoryTab.hpp"

namespace tile_compile::gui {

class MainWindow : public QMainWindow {
    Q_OBJECT

  public:
    explicit MainWindow(const std::string &project_root, QWidget *parent = nullptr);

  private slots:
    void on_start_run_clicked();
    void on_abort_run_clicked();
    void on_resume_run_clicked();
    void handle_runner_stdout(const QString &line);
    void handle_runner_stderr(const QString &line);
    void handle_runner_finished(int exit_code);
    void schedule_save_gui_state();

  private:
    void build_ui();
    void load_styles();
    void update_controls();
    void append_live(const QString &text);
    void load_gui_state();
    void save_gui_state();
    nlohmann::json collect_gui_state() const;
    void apply_gui_state(const nlohmann::json &state);
    nlohmann::json collect_calibration_from_ui() const;
    
    std::string project_root_;
    nlohmann::json constants_;
    std::unique_ptr<BackendClient> backend_;
    std::unique_ptr<RunnerProcess> runner_;
    
    nlohmann::json last_scan_;
    std::string confirmed_color_mode_;
    bool config_validated_ok_ = false;
    std::string current_run_id_;
    std::string current_run_dir_;
    std::string start_blocked_reason_;
    int frame_count_ = 0;
    
    QTabWidget *tabs_ = nullptr;
    QLabel *lbl_header_ = nullptr;
    
    ScanTab *scan_tab_ = nullptr;
    ConfigTab *config_tab_ = nullptr;
    RunTab *run_tab_ = nullptr;
    CurrentRunTab *current_run_tab_ = nullptr;
    HistoryTab *history_tab_ = nullptr;
    
    AssumptionsWidget *assumptions_widget_ = nullptr;
    PhaseProgressWidget *phase_progress_ = nullptr;
    QPlainTextEdit *live_log_ = nullptr;
    
    QTimer *gui_state_save_timer_ = nullptr;
    
    QString format_event_human(const nlohmann::json &ev) const;
    void ensure_startup_paths();
};

}
