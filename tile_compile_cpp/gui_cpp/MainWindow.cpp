#include "MainWindow.hpp"
#include "GuiConstants.hpp"
#include "tabs/ScanTab.hpp"
#include "tabs/ConfigTab.hpp"
#include "tabs/RunTab.hpp"
#include "tabs/CurrentRunTab.hpp"
#include "tabs/HistoryTab.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QFrame>
#include <QFile>
#include <QTextStream>
#include <QScrollBar>
#include <QMessageBox>
#include <QStringList>
#include <QGroupBox>
#include <filesystem>
#include <thread>

namespace tile_compile::gui {

MainWindow::MainWindow(const std::string &project_root, QWidget *parent)
    : QMainWindow(parent), project_root_(project_root) {
    
    constants_ = read_gui_constants(project_root_);
    backend_ = std::make_unique<BackendClient>(project_root_, constants_);
    runner_ = std::make_unique<RunnerProcess>(this);
    
    setMinimumSize(1000, 700);
    resize(1300, 900);
    setWindowTitle("Tile Compile – Methodik v4");
    
    build_ui();
    load_styles();
    load_gui_state();
    ensure_startup_paths();
    update_controls();
    
    connect(runner_.get(), &RunnerProcess::stdout_line, this, &MainWindow::handle_runner_stdout);
    connect(runner_.get(), &RunnerProcess::stderr_line, this, &MainWindow::handle_runner_stderr);
    connect(runner_.get(), &RunnerProcess::finished, this, &MainWindow::handle_runner_finished);
    
    gui_state_save_timer_ = new QTimer(this);
    gui_state_save_timer_->setSingleShot(true);
    gui_state_save_timer_->setInterval(500);
    connect(gui_state_save_timer_, &QTimer::timeout, this, &MainWindow::save_gui_state);
}

void MainWindow::build_ui() {
    auto *central = new QWidget();
    auto *root = new QVBoxLayout(central);
    root->setContentsMargins(12, 12, 12, 12);
    root->setSpacing(10);
    
    auto *header = new QHBoxLayout();
    auto *title = new QLabel("Tile Compile – Methodik v4");
    title->setStyleSheet("font-size: 18px; font-weight: 600;");
    header->addWidget(title);
    header->addStretch(1);
    lbl_header_ = new QLabel("idle");
    lbl_header_->setObjectName("StatusLabel");
    header->addWidget(lbl_header_);
    root->addLayout(header);
    
    tabs_ = new QTabWidget();
    root->addWidget(tabs_, 1);
    
    auto wrap_scroll = [](QWidget *content) -> QScrollArea* {
        auto *sa = new QScrollArea();
        sa->setWidgetResizable(true);
        sa->setFrameShape(QFrame::NoFrame);
        sa->setWidget(content);
        return sa;
    };
    
    // Scan Tab (with Calibration)
    scan_tab_ = new ScanTab(backend_.get(), project_root_, this);
    tabs_->addTab(wrap_scroll(scan_tab_), "Scan");
    connect(scan_tab_, &ScanTab::scan_completed, this, [this](int frame_count) {
        frame_count_ = frame_count;
        assumptions_widget_->update_reduced_mode_status(frame_count);
        update_controls();
        schedule_save_gui_state();
    });
    connect(scan_tab_, &ScanTab::color_mode_confirmed, this, [this](const QString &mode) {
        confirmed_color_mode_ = mode.toStdString();
        update_controls();
        schedule_save_gui_state();
    });
    connect(scan_tab_, &ScanTab::log_message, this, &MainWindow::append_live);
    connect(scan_tab_, &ScanTab::header_status_changed, this, [this](const QString &status) {
        lbl_header_->setText(status);
    });
    connect(scan_tab_, &ScanTab::input_dir_changed, this, [this](const QString &dir) {
        if (run_tab_ && run_tab_->get_input_dir().isEmpty()) {
            run_tab_->set_input_dir(dir);
        }
        schedule_save_gui_state();
    });
    connect(scan_tab_, &ScanTab::update_controls_requested, this, &MainWindow::update_controls);
    connect(scan_tab_, &ScanTab::calibration_changed, this, &MainWindow::schedule_save_gui_state);
    
    // Configuration Tab
    assumptions_widget_ = new AssumptionsWidget(this);
    config_tab_ = new ConfigTab(backend_.get(), assumptions_widget_, project_root_, this);
    tabs_->addTab(wrap_scroll(config_tab_), "Configuration");
    connect(config_tab_, &ConfigTab::config_edited, this, [this]() {
        config_validated_ok_ = false;
        update_controls();
        schedule_save_gui_state();
    });
    connect(config_tab_, &ConfigTab::config_validated, this, [this](bool ok) {
        config_validated_ok_ = ok;
        update_controls();
    });
    connect(config_tab_, &ConfigTab::log_message, this, &MainWindow::append_live);
    connect(config_tab_, &ConfigTab::header_status_changed, this, [this](const QString &status) {
        lbl_header_->setText(status);
    });
    connect(config_tab_, &ConfigTab::update_controls_requested, this, &MainWindow::update_controls);
    
    // Assumptions Tab
    auto *assumptions_page = new QWidget();
    auto *assumptions_page_layout = new QVBoxLayout(assumptions_page);
    assumptions_page_layout->setContentsMargins(0, 0, 0, 0);
    assumptions_page_layout->setSpacing(10);
    auto *assumptions_box = new QGroupBox("Methodik v4 Assumptions");
    auto *assumptions_box_layout = new QVBoxLayout(assumptions_box);
    assumptions_box_layout->setContentsMargins(12, 18, 12, 12);
    assumptions_box_layout->addWidget(assumptions_widget_);
    assumptions_page_layout->addWidget(assumptions_box, 1);
    tabs_->addTab(wrap_scroll(assumptions_page), "Assumptions");
    connect(assumptions_widget_, &AssumptionsWidget::assumptions_changed, this, [this]() {
        if (frame_count_ > 0) {
            assumptions_widget_->update_reduced_mode_status(frame_count_);
        }
        update_controls();
        schedule_save_gui_state();
    });
    
    // Run Tab
    run_tab_ = new RunTab(project_root_, this);
    tabs_->addTab(wrap_scroll(run_tab_), "Run");
    connect(run_tab_, &RunTab::start_run_clicked, this, &MainWindow::on_start_run_clicked);
    connect(run_tab_, &RunTab::abort_run_clicked, this, &MainWindow::on_abort_run_clicked);
    connect(run_tab_, &RunTab::working_dir_changed, this, &MainWindow::schedule_save_gui_state);
    connect(run_tab_, &RunTab::input_dir_changed, this, &MainWindow::schedule_save_gui_state);
    
    // Pipeline Progress Tab
    auto *progress_page = new QWidget();
    auto *progress_page_layout = new QVBoxLayout(progress_page);
    progress_page_layout->setContentsMargins(0, 0, 0, 0);
    progress_page_layout->setSpacing(10);
    auto *progress_box = new QGroupBox("Pipeline Progress (Methodik v4)");
    auto *progress_box_layout = new QVBoxLayout(progress_box);
    progress_box_layout->setContentsMargins(12, 18, 12, 12);
    phase_progress_ = new PhaseProgressWidget();
    progress_box_layout->addWidget(phase_progress_);
    progress_page_layout->addWidget(progress_box, 1);
    tabs_->addTab(wrap_scroll(progress_page), "Pipeline Progress");
    
    // Current Run Tab
    current_run_tab_ = new CurrentRunTab(backend_.get(), this);
    tabs_->addTab(wrap_scroll(current_run_tab_), "Current run");
    connect(current_run_tab_, &CurrentRunTab::resume_run_requested, this, &MainWindow::on_resume_run_clicked);
    connect(current_run_tab_, &CurrentRunTab::log_message, this, &MainWindow::append_live);
    
    // History Tab
    history_tab_ = new HistoryTab(backend_.get(), this);
    tabs_->addTab(wrap_scroll(history_tab_), "Run history");
    connect(history_tab_, &HistoryTab::run_selected, current_run_tab_, &CurrentRunTab::set_current_run);
    connect(history_tab_, &HistoryTab::log_message, this, &MainWindow::append_live);
    
    // Live log Tab
    auto *live_page = new QWidget();
    auto *live_page_layout = new QVBoxLayout(live_page);
    live_page_layout->setContentsMargins(0, 0, 0, 0);
    live_page_layout->setSpacing(10);
    auto *live_box = new QGroupBox("Live log");
    auto *live_layout = new QVBoxLayout(live_box);
    live_layout->setContentsMargins(12, 18, 12, 12);
    live_layout->setSpacing(10);
    live_log_ = new QPlainTextEdit();
    live_log_->setReadOnly(true);
    live_layout->addWidget(live_log_);
    live_page_layout->addWidget(live_box, 1);
    tabs_->addTab(wrap_scroll(live_page), "Live log");
    
    setCentralWidget(central);
}

void MainWindow::load_styles() {
    const std::string qss_path = project_root_ + "/gui_cpp/styles.qss";
    QFile file(QString::fromStdString(qss_path));
    if (file.open(QFile::ReadOnly | QFile::Text)) {
        QTextStream stream(&file);
        setStyleSheet(stream.readAll());
    }
}

void MainWindow::update_controls() {
    if (!scan_tab_ || !config_tab_ || !run_tab_) {
        return;
    }
    
    const bool is_running = runner_->is_running();
    
    // Get scan state
    const auto last_scan = scan_tab_->get_last_scan();
    const bool needs_confirm = !last_scan.empty() && last_scan.value("requires_user_confirmation", false);
    const bool has_confirm = !confirmed_color_mode_.empty();
    const bool scan_ok = !last_scan.empty() && last_scan.value("ok", false);
    
    // Enable/disable color mode controls
    scan_tab_->update_calibration_controls();
    
    // Calculate blocked reason
    start_blocked_reason_.clear();
    if (!scan_ok) {
        start_blocked_reason_ = last_scan.empty() ? "please run Scan first" : "scan has errors";
    } else if (needs_confirm && !has_confirm) {
        start_blocked_reason_ = "please confirm color mode";
    } else if (!config_validated_ok_) {
        start_blocked_reason_ = "please validate config";
    } else {
        // Check calibration
        const std::string cal_error = scan_tab_->validate_calibration();
        if (!cal_error.empty()) {
            start_blocked_reason_ = cal_error;
        }
    }
    
    // Update run tab controls
    const bool start_enabled = !is_running && start_blocked_reason_.empty();
    run_tab_->set_start_enabled(start_enabled, QString::fromStdString(start_blocked_reason_));
    run_tab_->set_abort_enabled(is_running);
    
    // Update status labels
    if (is_running) {
        run_tab_->set_status_text("running");
    } else if (!start_blocked_reason_.empty()) {
        run_tab_->set_status_text(QString("blocked: %1").arg(QString::fromStdString(start_blocked_reason_)));
    } else {
        run_tab_->set_status_text("ready");
    }
    
    // Reduced mode hints
    const auto assumptions = assumptions_widget_->get_assumptions();
    const int reduced_threshold = assumptions.value("frames_reduced_threshold", 200);
    const int minimum = assumptions.value("frames_min", 50);
    
    if (frame_count_ > 0 && frame_count_ < minimum) {
        run_tab_->show_reduced_mode_hint(frame_count_, minimum);
        phase_progress_->set_reduced_mode(true, frame_count_);
    } else if (frame_count_ > 0 && frame_count_ < reduced_threshold) {
        run_tab_->show_reduced_mode_hint(frame_count_, reduced_threshold);
        phase_progress_->set_reduced_mode(true, frame_count_);
    } else {
        run_tab_->hide_reduced_mode_hint();
        phase_progress_->set_reduced_mode(false, 0);
    }
}

void MainWindow::append_live(const QString &text) {
    live_log_->appendPlainText(text);
    auto *sb = live_log_->verticalScrollBar();
    sb->setValue(sb->maximum());
}

void MainWindow::on_start_run_clicked() {
    if (runner_->is_running()) {
        return;
    }
    
    if (!start_blocked_reason_.empty()) {
        append_live(QString("[ui] blocked: %1").arg(QString::fromStdString(start_blocked_reason_)));
        return;
    }
    
    auto *scan_tab = qobject_cast<ScanTab*>(tabs_->widget(0)->findChild<ScanTab*>());
    auto *config_tab = qobject_cast<ConfigTab*>(tabs_->widget(1)->findChild<ConfigTab*>());
    auto *run_tab = qobject_cast<RunTab*>(tabs_->widget(3)->findChild<RunTab*>());
    
    if (!scan_tab || !config_tab || !run_tab) {
        QMessageBox::critical(this, "Start run", "Internal error: tabs not found");
        return;
    }
    
    const QString input_dir = run_tab->get_input_dir();
    if (input_dir.isEmpty()) {
        QMessageBox::warning(this, "Start run", "Input dir is required");
        return;
    }
    
    append_live("[ui] start run");
    phase_progress_->reset();
    
    // Get runner executable path from constants (relative to exe dir)
    std::string runner_exe = "./tile_compile_runner";
    if (constants_.contains("RUNNER") && constants_["RUNNER"].contains("executable")) {
        runner_exe = constants_["RUNNER"]["executable"].get<std::string>();
    }
    
    // Resolve relative to executable directory
    QString exe_dir = QCoreApplication::applicationDirPath();
    QString runner_path = exe_dir + "/" + QString::fromStdString(runner_exe);
    
    QStringList cmd;
    cmd << runner_path;
    cmd << "run";
    cmd << "--config" << config_tab->get_config_path();
    cmd << "--input-dir" << input_dir;
    cmd << "--runs-dir" << run_tab->get_runs_dir();
    cmd << "--pattern" << run_tab->get_pattern();
    
    if (run_tab->is_dry_run()) {
        cmd << "--dry-run";
    }
    
    if (!confirmed_color_mode_.empty()) {
        cmd << "--color-mode-confirmed" << QString::fromStdString(confirmed_color_mode_);
    }
    
    append_live(QString("[runner] %1").arg(cmd.join(" ")));
    
    try {
        runner_->start(cmd, QString::fromStdString(run_tab->get_working_dir().toStdString()));
        update_controls();
    } catch (const std::exception &e) {
        QMessageBox::critical(this, "Start failed", e.what());
        update_controls();
    }
}

void MainWindow::on_abort_run_clicked() {
    append_live("[ui] abort run");
    runner_->stop();
    update_controls();
}

void MainWindow::on_resume_run_clicked() {
    append_live("[ui] resume run requested (not yet implemented)");
    QMessageBox::information(this, "Resume run", "Resume functionality not yet implemented");
}

void MainWindow::handle_runner_stdout(const QString &line) {
    if (line.trimmed().isEmpty()) {
        return;
    }
    
    try {
        const auto ev = nlohmann::json::parse(line.toStdString());
        if (!ev.is_object()) {
            append_live(line);
            return;
        }
        
        const std::string ev_type = ev.value("type", "");
        
        if (ev_type == "run_start") {
            current_run_id_ = ev.value("run_id", "");
            current_run_dir_ = ev.value("run_dir", "");
            phase_progress_->reset();
            append_live(QString("[run_start] run_id=%1").arg(QString::fromStdString(current_run_id_)));
            
            auto *current_run_tab = qobject_cast<CurrentRunTab*>(tabs_->widget(5)->findChild<CurrentRunTab*>());
            if (current_run_tab) {
                current_run_tab->set_current_run(QString::fromStdString(current_run_id_), 
                                                 QString::fromStdString(current_run_dir_));
            }
        } else if (ev_type == "phase_start") {
            const std::string phase_name = ev.value("phase_name", "");
            phase_progress_->update_phase(phase_name, "running");
            append_live(QString("[phase_start] %1").arg(QString::fromStdString(phase_name)));
        } else if (ev_type == "phase_progress") {
            const std::string phase_name = ev.value("phase_name", "");
            const int current = ev.value("current", 0);
            const int total = ev.value("total", 0);
            const std::string substep = ev.value("substep", "");
            const std::string pass_info = ev.value("pass", "");
            phase_progress_->update_phase(phase_name, "running", current, total, substep, pass_info);
        } else if (ev_type == "phase_end") {
            const std::string phase_name = ev.value("phase_name", "");
            const std::string status = ev.value("status", "ok");
            phase_progress_->update_phase(phase_name, status);
            append_live(QString("[phase_end] %1 status=%2")
                        .arg(QString::fromStdString(phase_name))
                        .arg(QString::fromStdString(status)));
            
            if (status == "error" && ev.contains("error_detail")) {
                const std::string detail = ev["error_detail"].dump();
                phase_progress_->set_error_detail(phase_name, detail);
            }
        } else if (ev_type == "run_end") {
            const std::string status = ev.value("status", "");
            append_live(QString("[run_end] status=%1").arg(QString::fromStdString(status)));
        } else {
            append_live(format_event_human(ev));
        }
    } catch (...) {
        append_live(line);
    }
}

void MainWindow::handle_runner_stderr(const QString &line) {
    append_live(QString("[stderr] %1").arg(line));
}

void MainWindow::handle_runner_finished(int exit_code) {
    append_live(QString("[runner] finished with exit code %1").arg(exit_code));
    update_controls();
}

QString MainWindow::format_event_human(const nlohmann::json &ev) const {
    const std::string type = ev.value("type", "");
    if (type.empty()) {
        return QString::fromStdString(ev.dump());
    }
    
    QString msg = QString("[%1]").arg(QString::fromStdString(type));
    for (auto it = ev.begin(); it != ev.end(); ++it) {
        if (it.key() != "type") {
            msg += QString(" %1=%2").arg(QString::fromStdString(it.key()))
                                    .arg(QString::fromStdString(it.value().dump()));
        }
    }
    return msg;
}

void MainWindow::schedule_save_gui_state() {
    gui_state_save_timer_->start();
}

void MainWindow::load_gui_state() {
    std::thread([this]() {
        try {
            const auto cli_sub = backend_->constants().value("CLI", nlohmann::json::object())
                .value("sub", nlohmann::json::object());
            std::vector<std::string> args = {
                cli_sub.value("LOAD_GUI_STATE", "load-gui-state")
            };
            
            const auto result = backend_->run_json(project_root_, args);
            
            QMetaObject::invokeMethod(this, [this, result]() {
                if (result.contains("state") && result["state"].is_object()) {
                    apply_gui_state(result["state"]);
                }
            }, Qt::QueuedConnection);
            
        } catch (const std::exception &e) {
            QMetaObject::invokeMethod(this, [this, e]() {
                append_live(QString("[ui] load gui state error: %1").arg(e.what()));
            }, Qt::QueuedConnection);
        }
    }).detach();
}

void MainWindow::save_gui_state() {
    const auto state = collect_gui_state();
    const std::string state_json = state.dump();
    
    std::thread([this, state_json]() {
        try {
            const auto cli_sub = backend_->constants().value("CLI", nlohmann::json::object())
                .value("sub", nlohmann::json::object());
            std::vector<std::string> args = {
                cli_sub.value("SAVE_GUI_STATE", "save-gui-state"),
                "--stdin"
            };
            
            backend_->run_json(project_root_, args, state_json, 3000);
            
        } catch (...) {
            // Silent fail on save
        }
    }).detach();
}

nlohmann::json MainWindow::collect_gui_state() const {
    nlohmann::json state = nlohmann::json::object();
    
    if (scan_tab_ && run_tab_) {
        const QString last_input = !scan_tab_->get_scan_input_dir().isEmpty() 
            ? scan_tab_->get_scan_input_dir() 
            : run_tab_->get_input_dir();
        state["lastInputDir"] = last_input.toStdString();
        state["scanInputDir"] = scan_tab_->get_scan_input_dir().toStdString();
        state["calibration"] = scan_tab_->collect_calibration();
        state["lastScan"] = scan_tab_->get_last_scan();
        state["frameCount"] = scan_tab_->get_frame_count();
    }
    
    if (run_tab_) {
        state["inputDir"] = run_tab_->get_input_dir().toStdString();
        state["workingDir"] = run_tab_->get_working_dir().toStdString();
        state["runsDir"] = run_tab_->get_runs_dir().toStdString();
        state["pattern"] = run_tab_->get_pattern().toStdString();
        state["dryRun"] = run_tab_->is_dry_run();
    }
    
    state["confirmedColorMode"] = confirmed_color_mode_;
    
    if (config_tab_) {
        state["configPath"] = config_tab_->get_config_path().toStdString();
        state["configYaml"] = config_tab_->get_config_yaml().toStdString();
        state["configValidatedOk"] = config_tab_->is_config_validated();
    }
    
    if (assumptions_widget_) {
        state["assumptions"] = assumptions_widget_->get_assumptions();
    }
    
    state["activeTab"] = tabs_->currentIndex();
    
    return state;
}

void MainWindow::apply_gui_state(const nlohmann::json &state) {
    tabs_->blockSignals(true);
    
    if (state.contains("scanInputDir") && scan_tab_) {
        scan_tab_->set_input_dir_from_scan(QString::fromStdString(state["scanInputDir"].get<std::string>()));
    }
    
    if (state.contains("calibration") && scan_tab_) {
        scan_tab_->apply_calibration(state["calibration"]);
    }
    
    if (state.contains("lastScan") && scan_tab_) {
        scan_tab_->set_last_scan(state["lastScan"]);
    }
    
    if (state.contains("frameCount")) {
        frame_count_ = state["frameCount"].get<int>();
    }
    
    if (run_tab_) {
        if (state.contains("inputDir")) {
            run_tab_->set_input_dir(QString::fromStdString(state["inputDir"].get<std::string>()));
        }
        if (state.contains("workingDir")) {
            run_tab_->set_working_dir(QString::fromStdString(state["workingDir"].get<std::string>()));
        }
        if (state.contains("runsDir")) {
            run_tab_->set_runs_dir(QString::fromStdString(state["runsDir"].get<std::string>()));
        }
        if (state.contains("pattern")) {
            run_tab_->set_pattern(QString::fromStdString(state["pattern"].get<std::string>()));
        }
        if (state.contains("dryRun")) {
            run_tab_->set_dry_run(state["dryRun"].get<bool>());
        }
    }
    
    if (state.contains("confirmedColorMode")) {
        confirmed_color_mode_ = state["confirmedColorMode"].get<std::string>();
        // Update UI to show confirmed color mode
        if (scan_tab_ && !confirmed_color_mode_.empty()) {
            scan_tab_->set_confirmed_color_mode(QString::fromStdString(confirmed_color_mode_));
        }
    }
    
    if (config_tab_) {
        if (state.contains("configPath")) {
            config_tab_->set_config_path(QString::fromStdString(state["configPath"].get<std::string>()));
        }
        if (state.contains("configYaml")) {
            config_tab_->set_config_yaml(QString::fromStdString(state["configYaml"].get<std::string>()));
        }
        if (state.contains("configValidatedOk")) {
            const bool validated = state["configValidatedOk"].get<bool>();
            config_tab_->set_config_validated(validated);
            config_validated_ok_ = validated;  // Sync MainWindow state
        }
    }
    
    if (state.contains("assumptions") && assumptions_widget_) {
        assumptions_widget_->set_assumptions(state["assumptions"]);
    }
    
    if (state.contains("activeTab")) {
        const int tab_idx = state["activeTab"].get<int>();
        if (tab_idx >= 0 && tab_idx < tabs_->count()) {
            tabs_->setCurrentIndex(tab_idx);
        }
    }
    
    tabs_->blockSignals(false);
    update_controls();
}

void MainWindow::ensure_startup_paths() {
    auto *run_tab = qobject_cast<RunTab*>(tabs_->widget(3)->findChild<RunTab*>());
    if (run_tab && run_tab->get_working_dir().isEmpty()) {
        run_tab->set_working_dir(QString::fromStdString(project_root_));
    }
}

}
