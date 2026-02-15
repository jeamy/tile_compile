#include "ScanTab.hpp"
#include "../BackendClient.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <thread>
#include <filesystem>

namespace tile_compile::gui {

ScanTab::ScanTab(BackendClient *backend, const std::string &project_root, QWidget *parent)
    : QWidget(parent), backend_(backend), project_root_(project_root) {
    build_ui();
}

void ScanTab::build_ui() {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);
    
    auto *scan_box = new QGroupBox("Scan");
    auto *scan_layout = new QVBoxLayout(scan_box);
    scan_layout->setContentsMargins(12, 18, 12, 12);
    scan_layout->setSpacing(10);
    
    auto *scan_form = new QFormLayout();
    scan_form->setSpacing(10);
    
    auto *input_row = new QHBoxLayout();
    scan_input_dir_ = new QLineEdit();
    scan_input_dir_->setMinimumHeight(30);
    scan_input_dir_->setPlaceholderText("/path/to/frames");
    auto *btn_browse_scan_dir = new QPushButton("Browse");
    btn_browse_scan_dir->setMinimumHeight(30);
    btn_browse_scan_dir->setFixedWidth(100);
    btn_scan_ = new QPushButton("Scan");
    btn_scan_->setMinimumHeight(30);
    btn_scan_->setFixedWidth(100);
    lbl_scan_ = new QLabel("idle");
    lbl_scan_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    
    input_row->addWidget(scan_input_dir_);
    input_row->addWidget(btn_browse_scan_dir);
    input_row->addWidget(btn_scan_);
    input_row->addWidget(lbl_scan_);
    scan_form->addRow("Input dir", input_row);
    scan_layout->addLayout(scan_form);
    
    // Calibration section
    auto *calib_box = new QGroupBox("Calibration (Bias/Darks/Flats)");
    auto *calib_layout = new QVBoxLayout(calib_box);
    calib_layout->setContentsMargins(12, 12, 12, 12);
    calib_layout->setSpacing(8);
    auto *calib_form = new QFormLayout();
    calib_form->setSpacing(8);
    
    auto *bias_row = new QHBoxLayout();
    cal_use_bias_ = new QCheckBox("use");
    cal_bias_dir_ = new QLineEdit();
    cal_bias_dir_->setPlaceholderText("/path/to/bias");
    btn_browse_bias_dir_ = new QPushButton("Browse");
    btn_browse_bias_dir_->setFixedWidth(100);
    bias_row->addWidget(cal_use_bias_);
    bias_row->addWidget(cal_bias_dir_, 1);
    bias_row->addWidget(btn_browse_bias_dir_);
    calib_form->addRow("Bias dir", bias_row);
    
    auto *biasm_row = new QHBoxLayout();
    cal_bias_use_master_ = new QCheckBox("use master");
    cal_bias_master_ = new QLineEdit();
    cal_bias_master_->setPlaceholderText("/path/to/master_bias.fit (optional)");
    btn_browse_bias_master_ = new QPushButton("Browse");
    btn_browse_bias_master_->setFixedWidth(100);
    biasm_row->addWidget(cal_bias_use_master_);
    biasm_row->addWidget(cal_bias_master_, 1);
    biasm_row->addWidget(btn_browse_bias_master_);
    calib_form->addRow("Bias master", biasm_row);
    
    auto *dark_row = new QHBoxLayout();
    cal_use_dark_ = new QCheckBox("use");
    cal_darks_dir_ = new QLineEdit();
    cal_darks_dir_->setPlaceholderText("/path/to/darks");
    btn_browse_darks_dir_ = new QPushButton("Browse");
    btn_browse_darks_dir_->setFixedWidth(100);
    dark_row->addWidget(cal_use_dark_);
    dark_row->addWidget(cal_darks_dir_, 1);
    dark_row->addWidget(btn_browse_darks_dir_);
    calib_form->addRow("Darks dir", dark_row);
    
    auto *darkm_row = new QHBoxLayout();
    cal_dark_use_master_ = new QCheckBox("use master");
    cal_dark_master_ = new QLineEdit();
    cal_dark_master_->setPlaceholderText("/path/to/master_dark.fit (optional)");
    btn_browse_dark_master_ = new QPushButton("Browse");
    btn_browse_dark_master_->setFixedWidth(100);
    darkm_row->addWidget(cal_dark_use_master_);
    darkm_row->addWidget(cal_dark_master_, 1);
    darkm_row->addWidget(btn_browse_dark_master_);
    calib_form->addRow("Dark master", darkm_row);
    
    auto *flat_row = new QHBoxLayout();
    cal_use_flat_ = new QCheckBox("use");
    cal_flats_dir_ = new QLineEdit();
    cal_flats_dir_->setPlaceholderText("/path/to/flats");
    btn_browse_flats_dir_ = new QPushButton("Browse");
    btn_browse_flats_dir_->setFixedWidth(100);
    flat_row->addWidget(cal_use_flat_);
    flat_row->addWidget(cal_flats_dir_, 1);
    flat_row->addWidget(btn_browse_flats_dir_);
    calib_form->addRow("Flats dir", flat_row);
    
    auto *flatm_row = new QHBoxLayout();
    cal_flat_use_master_ = new QCheckBox("use master");
    cal_flat_master_ = new QLineEdit();
    cal_flat_master_->setPlaceholderText("/path/to/master_flat.fit (optional)");
    btn_browse_flat_master_ = new QPushButton("Browse");
    btn_browse_flat_master_->setFixedWidth(100);
    flatm_row->addWidget(cal_flat_use_master_);
    flatm_row->addWidget(cal_flat_master_, 1);
    flatm_row->addWidget(btn_browse_flat_master_);
    calib_form->addRow("Flat master", flatm_row);
    
    calib_layout->addLayout(calib_form);
    auto *calib_hint = new QLabel(
        "Hint: if 'use master' is unchecked, a master is built from the selected dir and written to outputs/calibration/master_*.fit"
    );
    calib_hint->setWordWrap(true);
    calib_hint->setObjectName("StatusLabel");
    calib_layout->addWidget(calib_hint);
    scan_layout->addWidget(calib_box);
    
    auto *row2 = new QHBoxLayout();
    scan_frames_min_ = new QSpinBox();
    scan_frames_min_->setMinimum(1);
    scan_frames_min_->setMaximum(1000000);
    scan_frames_min_->setValue(1);
    scan_with_checksums_ = new QCheckBox("With checksums");
    row2->addWidget(new QLabel("Frames min"));
    row2->addWidget(scan_frames_min_);
    row2->addWidget(scan_with_checksums_);
    row2->addStretch(1);
    scan_layout->addLayout(row2);
    
    scan_msg_ = new QLabel("");
    scan_msg_->setWordWrap(true);
    scan_msg_->setObjectName("StatusLabel");
    scan_layout->addWidget(scan_msg_);
    
    auto *row3 = new QHBoxLayout();
    color_mode_select_ = new QComboBox();
    btn_confirm_color_ = new QPushButton("Confirm");
    lbl_confirm_hint_ = new QLabel("");
    lbl_confirm_hint_->setObjectName("StatusLabel");
    row3->addWidget(new QLabel("Color mode"));
    row3->addWidget(color_mode_select_);
    row3->addWidget(btn_confirm_color_);
    row3->addStretch(1);
    scan_layout->addLayout(row3);
    scan_layout->addWidget(lbl_confirm_hint_);
    
    scan_reduced_mode_hint_ = new QLabel("");
    scan_reduced_mode_hint_->setObjectName("ReducedModeWarning");
    scan_reduced_mode_hint_->setVisible(false);
    scan_layout->addWidget(scan_reduced_mode_hint_);
    
    layout->addWidget(scan_box, 1);
    
    connect(btn_scan_, &QPushButton::clicked, this, &ScanTab::on_scan_clicked);
    connect(btn_confirm_color_, &QPushButton::clicked, this, &ScanTab::on_confirm_color_clicked);
    connect(btn_browse_scan_dir, &QPushButton::clicked, this, &ScanTab::on_browse_scan_dir);
    connect(btn_browse_bias_dir_, &QPushButton::clicked, this, &ScanTab::on_browse_bias_dir);
    connect(btn_browse_darks_dir_, &QPushButton::clicked, this, &ScanTab::on_browse_darks_dir);
    connect(btn_browse_flats_dir_, &QPushButton::clicked, this, &ScanTab::on_browse_flats_dir);
    connect(btn_browse_bias_master_, &QPushButton::clicked, this, &ScanTab::on_browse_bias_master);
    connect(btn_browse_dark_master_, &QPushButton::clicked, this, &ScanTab::on_browse_dark_master);
    connect(btn_browse_flat_master_, &QPushButton::clicked, this, &ScanTab::on_browse_flat_master);
    
    connect(cal_use_bias_, &QCheckBox::checkStateChanged, this, &ScanTab::on_calibration_changed);
    connect(cal_use_dark_, &QCheckBox::checkStateChanged, this, &ScanTab::on_calibration_changed);
    connect(cal_use_flat_, &QCheckBox::checkStateChanged, this, &ScanTab::on_calibration_changed);
    connect(cal_bias_use_master_, &QCheckBox::checkStateChanged, this, &ScanTab::on_calibration_changed);
    connect(cal_dark_use_master_, &QCheckBox::checkStateChanged, this, &ScanTab::on_calibration_changed);
    connect(cal_flat_use_master_, &QCheckBox::checkStateChanged, this, &ScanTab::on_calibration_changed);
}

void ScanTab::on_scan_clicked() {
    emit log_message("[ui] scan button clicked");
    
    const QString input_path = scan_input_dir_->text().trimmed();
    if (input_path.isEmpty()) {
        lbl_scan_->setText("error");
        scan_msg_->setText("Input dir is required");
        QMessageBox::warning(this, "Scan Error", "Input directory is required");
        return;
    }
    
    if (!std::filesystem::exists(input_path.toStdString())) {
        lbl_scan_->setText("error");
        scan_msg_->setText(QString("Directory not found: %1").arg(input_path));
        QMessageBox::warning(this, "Scan Error", QString("Directory not found: %1").arg(input_path));
        return;
    }
    
    btn_scan_->setEnabled(false);
    lbl_scan_->setText("scanning...");
    scan_msg_->setText("");
    last_scan_ = nlohmann::json::object();
    confirmed_color_mode_.clear();
    frame_count_ = 0;
    color_mode_select_->clear();
    lbl_confirm_hint_->setText("");
    lbl_confirm_hint_->setStyleSheet("");
    scan_reduced_mode_hint_->setVisible(false);
    emit header_status_changed("scanning...");
    emit update_controls_requested();
    
    const int frames_min = scan_frames_min_->value();
    const bool with_checksums = scan_with_checksums_->isChecked();
    
    std::thread([this, input_path, frames_min, with_checksums]() {
        try {
            std::vector<std::string> args = {
                backend_->backend_cmd()[0].substr(backend_->backend_cmd()[0].find_last_of('/') + 1) == "tile_compile_cli" 
                    ? "scan" : backend_->constants().value("CLI", nlohmann::json::object()).value("sub", nlohmann::json::object()).value("SCAN", "scan"),
                input_path.toStdString(),
                "--frames-min",
                std::to_string(frames_min),
            };
            if (with_checksums) {
                args.push_back("--with-checksums");
            }
            
            const auto result = backend_->run_json(project_root_, args, "", 120000);
            
            QMetaObject::invokeMethod(this, [this, result]() {
                last_scan_ = result;
                const bool ok = result.value("ok", false);
                lbl_scan_->setText(ok ? "ok" : "error");
                
                std::string msg;
                if (result.contains("errors") && result["errors"].is_array() && !result["errors"].empty()) {
                    const auto &first = result["errors"][0];
                    msg = first.value("code", "error") + ": " + first.value("message", "");
                } else if (result.contains("warnings") && result["warnings"].is_array() && !result["warnings"].empty()) {
                    const auto &first = result["warnings"][0];
                    msg = first.value("code", "warning") + ": " + first.value("message", "");
                } else if (ok) {
                    const int frames = result.value("frames_detected", 0);
                    const std::string cm = result.value("color_mode", "");
                    const std::string bp = result.value("bayer_pattern", "");
                    std::vector<std::string> parts;
                    if (frames > 0) parts.push_back("frames_detected=" + std::to_string(frames));
                    if (!cm.empty()) parts.push_back("color_mode=" + cm);
                    if (!bp.empty()) parts.push_back("bayer_pattern=" + bp);
                    for (size_t i = 0; i < parts.size(); ++i) {
                        if (i > 0) msg += ", ";
                        msg += parts[i];
                    }
                }
                scan_msg_->setText(QString::fromStdString(msg));
                
                color_mode_select_->blockSignals(true);
                color_mode_select_->clear();
                std::vector<std::string> candidates;
                if (result.contains("color_mode_candidates") && result["color_mode_candidates"].is_array()) {
                    for (const auto &c : result["color_mode_candidates"]) {
                        if (c.is_string()) {
                            const std::string s = c.get<std::string>();
                            if (!s.empty() && std::find(candidates.begin(), candidates.end(), s) == candidates.end()) {
                                candidates.push_back(s);
                            }
                        }
                    }
                }
                const std::string cm = result.value("color_mode", "");
                if (!cm.empty() && cm != "UNKNOWN" && std::find(candidates.begin(), candidates.end(), cm) == candidates.end()) {
                    candidates.insert(candidates.begin(), cm);
                }
                for (const auto &c : candidates) {
                    color_mode_select_->addItem(QString::fromStdString(c));
                }
                if (!candidates.empty()) {
                    color_mode_select_->setCurrentIndex(0);
                }
                color_mode_select_->blockSignals(false);
                
                if (result.value("requires_user_confirmation", false)) {
                    lbl_confirm_hint_->setText("⚠ Scan could not determine color mode (missing/inconsistent BAYERPAT). Please confirm manually.");
                    lbl_confirm_hint_->setStyleSheet("color: orange; font-weight: bold;");
                } else {
                    lbl_confirm_hint_->setText("");
                    lbl_confirm_hint_->setStyleSheet("");
                }
                
                if (ok && !result.value("requires_user_confirmation", false)) {
                    const std::string cm2 = result.value("color_mode", "");
                    if (!cm2.empty() && cm2 != "UNKNOWN") {
                        confirmed_color_mode_ = cm2;
                        lbl_confirm_hint_->setText(QString("✓ Auto-detected: %1").arg(QString::fromStdString(cm2)));
                        lbl_confirm_hint_->setStyleSheet("color: green; font-weight: bold;");
                    }
                }
                
                frame_count_ = result.value("frames_detected", 0);
                
                // Propagate input dir on success
                if (ok) {
                    emit input_dir_changed(scan_input_dir_->text());
                }
                
                emit header_status_changed("idle");
                emit scan_completed(frame_count_);
                emit update_controls_requested();
                
                btn_scan_->setEnabled(true);
            }, Qt::QueuedConnection);
            
        } catch (const std::exception &e) {
            const std::string err_msg = e.what();
            QMetaObject::invokeMethod(this, [this, err_msg]() {
                lbl_scan_->setText("error");
                scan_msg_->setText(QString::fromStdString(err_msg));
                emit header_status_changed("idle");
                emit update_controls_requested();
                btn_scan_->setEnabled(true);
            }, Qt::QueuedConnection);
        }
    }).detach();
}

void ScanTab::on_confirm_color_clicked() {
    const QString sel = color_mode_select_->currentText().trimmed();
    confirmed_color_mode_ = sel.toStdString();
    emit log_message(QString("[ui] confirmed color_mode=%1").arg(sel));
    
    if (!confirmed_color_mode_.empty()) {
        lbl_confirm_hint_->setText(QString("✓ Confirmed: %1").arg(sel));
        lbl_confirm_hint_->setStyleSheet("color: green; font-weight: bold;");
        emit color_mode_confirmed(sel);
    }
}

void ScanTab::on_browse_scan_dir() {
    const QString start = scan_input_dir_->text().trimmed().isEmpty() 
        ? QString::fromStdString(project_root_) 
        : scan_input_dir_->text();
    const QString p = QFileDialog::getExistingDirectory(this, "Select input directory", start);
    if (!p.isEmpty()) {
        scan_input_dir_->setText(p);
    }
}

void ScanTab::on_browse_bias_dir() {
    const QString start = cal_bias_dir_->text().trimmed().isEmpty() 
        ? QString::fromStdString(project_root_) 
        : cal_bias_dir_->text();
    const QString p = QFileDialog::getExistingDirectory(this, "Select bias directory", start);
    if (!p.isEmpty()) {
        cal_bias_dir_->setText(p);
        on_calibration_changed();
    }
}

void ScanTab::on_browse_darks_dir() {
    const QString start = cal_darks_dir_->text().trimmed().isEmpty() 
        ? QString::fromStdString(project_root_) 
        : cal_darks_dir_->text();
    const QString p = QFileDialog::getExistingDirectory(this, "Select darks directory", start);
    if (!p.isEmpty()) {
        cal_darks_dir_->setText(p);
        on_calibration_changed();
    }
}

void ScanTab::on_browse_flats_dir() {
    const QString start = cal_flats_dir_->text().trimmed().isEmpty() 
        ? QString::fromStdString(project_root_) 
        : cal_flats_dir_->text();
    const QString p = QFileDialog::getExistingDirectory(this, "Select flats directory", start);
    if (!p.isEmpty()) {
        cal_flats_dir_->setText(p);
        on_calibration_changed();
    }
}

void ScanTab::on_browse_bias_master() {
    const QString start = cal_bias_master_->text().trimmed().isEmpty() 
        ? QString::fromStdString(project_root_) 
        : cal_bias_master_->text();
    const QString p = QFileDialog::getOpenFileName(this, "Select master bias", start, 
        "FITS Files (*.fit *.fits *.fts *.fit.fz *.fits.fz *.fts.fz);;All Files (*)");
    if (!p.isEmpty()) {
        cal_bias_master_->setText(p);
        on_calibration_changed();
    }
}

void ScanTab::on_browse_dark_master() {
    const QString start = cal_dark_master_->text().trimmed().isEmpty() 
        ? QString::fromStdString(project_root_) 
        : cal_dark_master_->text();
    const QString p = QFileDialog::getOpenFileName(this, "Select master dark", start, 
        "FITS Files (*.fit *.fits *.fts *.fit.fz *.fits.fz *.fts.fz);;All Files (*)");
    if (!p.isEmpty()) {
        cal_dark_master_->setText(p);
        on_calibration_changed();
    }
}

void ScanTab::on_browse_flat_master() {
    const QString start = cal_flat_master_->text().trimmed().isEmpty() 
        ? QString::fromStdString(project_root_) 
        : cal_flat_master_->text();
    const QString p = QFileDialog::getOpenFileName(this, "Select master flat", start, 
        "FITS Files (*.fit *.fits *.fts *.fit.fz *.fits.fz *.fts.fz);;All Files (*)");
    if (!p.isEmpty()) {
        cal_flat_master_->setText(p);
        on_calibration_changed();
    }
}

void ScanTab::on_calibration_changed() {
    update_calibration_controls();
    emit calibration_changed();
    emit update_controls_requested();
}

void ScanTab::update_calibration_controls() {
    const bool bias_on = cal_use_bias_->isChecked();
    const bool dark_on = cal_use_dark_->isChecked();
    const bool flat_on = cal_use_flat_->isChecked();
    
    const bool bias_master_on = cal_bias_use_master_->isChecked();
    const bool dark_master_on = cal_dark_use_master_->isChecked();
    const bool flat_master_on = cal_flat_use_master_->isChecked();
    
    cal_bias_use_master_->setEnabled(bias_on);
    cal_dark_use_master_->setEnabled(dark_on);
    cal_flat_use_master_->setEnabled(flat_on);
    
    cal_bias_dir_->setEnabled(bias_on && !bias_master_on);
    btn_browse_bias_dir_->setEnabled(bias_on && !bias_master_on);
    cal_bias_master_->setEnabled(bias_on && bias_master_on);
    btn_browse_bias_master_->setEnabled(bias_on && bias_master_on);
    
    cal_darks_dir_->setEnabled(dark_on && !dark_master_on);
    btn_browse_darks_dir_->setEnabled(dark_on && !dark_master_on);
    cal_dark_master_->setEnabled(dark_on && dark_master_on);
    btn_browse_dark_master_->setEnabled(dark_on && dark_master_on);
    
    cal_flats_dir_->setEnabled(flat_on && !flat_master_on);
    btn_browse_flats_dir_->setEnabled(flat_on && !flat_master_on);
    cal_flat_master_->setEnabled(flat_on && flat_master_on);
    btn_browse_flat_master_->setEnabled(flat_on && flat_master_on);
}

std::string ScanTab::validate_calibration() const {
    const auto cal = collect_calibration();
    
    if (cal.value("use_bias", false)) {
        if (cal.value("bias_use_master", false)) {
            const std::string master = cal.value("bias_master", "");
            if (master.empty()) {
                return "calibration: bias enabled (master) but no bias_master set";
            }
        } else {
            const std::string dir = cal.value("bias_dir", "");
            if (dir.empty()) {
                return "calibration: bias enabled (dir) but no bias_dir set";
            }
        }
    }
    
    if (cal.value("use_dark", false)) {
        if (cal.value("dark_use_master", false)) {
            const std::string master = cal.value("dark_master", "");
            if (master.empty()) {
                return "calibration: dark enabled (master) but no dark_master set";
            }
        } else {
            const std::string dir = cal.value("darks_dir", "");
            if (dir.empty()) {
                return "calibration: dark enabled (dir) but no darks_dir set";
            }
        }
    }
    
    if (cal.value("use_flat", false)) {
        if (cal.value("flat_use_master", false)) {
            const std::string master = cal.value("flat_master", "");
            if (master.empty()) {
                return "calibration: flat enabled (master) but no flat_master set";
            }
        } else {
            const std::string dir = cal.value("flats_dir", "");
            if (dir.empty()) {
                return "calibration: flat enabled (dir) but no flats_dir set";
            }
        }
    }
    
    return "";
}

void ScanTab::set_input_dir_from_scan(const QString &dir) {
    scan_input_dir_->setText(dir);
}

QString ScanTab::get_scan_input_dir() const {
    return scan_input_dir_->text().trimmed();
}

nlohmann::json ScanTab::collect_calibration() const {
    const bool use_bias = cal_use_bias_->isChecked();
    const bool use_dark = cal_use_dark_->isChecked();
    const bool use_flat = cal_use_flat_->isChecked();
    
    const bool bias_use_master = cal_bias_use_master_->isChecked();
    const bool dark_use_master = cal_dark_use_master_->isChecked();
    const bool flat_use_master = cal_flat_use_master_->isChecked();
    
    std::string bias_dir = cal_bias_dir_->text().trimmed().toStdString();
    std::string darks_dir = cal_darks_dir_->text().trimmed().toStdString();
    std::string flats_dir = cal_flats_dir_->text().trimmed().toStdString();
    std::string bias_master = cal_bias_master_->text().trimmed().toStdString();
    std::string dark_master = cal_dark_master_->text().trimmed().toStdString();
    std::string flat_master = cal_flat_master_->text().trimmed().toStdString();
    
    if (use_bias) {
        if (bias_use_master) {
            bias_dir = "";
        } else {
            bias_master = "";
        }
    }
    if (use_dark) {
        if (dark_use_master) {
            darks_dir = "";
        } else {
            dark_master = "";
        }
    }
    if (use_flat) {
        if (flat_use_master) {
            flats_dir = "";
        } else {
            flat_master = "";
        }
    }
    
    return {
        {"use_bias", use_bias},
        {"use_dark", use_dark},
        {"use_flat", use_flat},
        {"bias_use_master", bias_use_master},
        {"dark_use_master", dark_use_master},
        {"flat_use_master", flat_use_master},
        {"bias_dir", bias_dir},
        {"darks_dir", darks_dir},
        {"flats_dir", flats_dir},
        {"bias_master", bias_master},
        {"dark_master", dark_master},
        {"flat_master", flat_master},
        {"pattern", "*.fit*"},
    };
}

void ScanTab::apply_calibration(const nlohmann::json &cal) {
    if (cal.contains("use_bias")) {
        cal_use_bias_->setChecked(cal["use_bias"].get<bool>());
    }
    if (cal.contains("use_dark")) {
        cal_use_dark_->setChecked(cal["use_dark"].get<bool>());
    }
    if (cal.contains("use_flat")) {
        cal_use_flat_->setChecked(cal["use_flat"].get<bool>());
    }
    if (cal.contains("bias_use_master")) {
        cal_bias_use_master_->setChecked(cal["bias_use_master"].get<bool>());
    }
    if (cal.contains("dark_use_master")) {
        cal_dark_use_master_->setChecked(cal["dark_use_master"].get<bool>());
    }
    if (cal.contains("flat_use_master")) {
        cal_flat_use_master_->setChecked(cal["flat_use_master"].get<bool>());
    }
    if (cal.contains("bias_dir")) {
        cal_bias_dir_->setText(QString::fromStdString(cal["bias_dir"].get<std::string>()));
    }
    if (cal.contains("darks_dir")) {
        cal_darks_dir_->setText(QString::fromStdString(cal["darks_dir"].get<std::string>()));
    }
    if (cal.contains("flats_dir")) {
        cal_flats_dir_->setText(QString::fromStdString(cal["flats_dir"].get<std::string>()));
    }
    if (cal.contains("bias_master")) {
        cal_bias_master_->setText(QString::fromStdString(cal["bias_master"].get<std::string>()));
    }
    if (cal.contains("dark_master")) {
        cal_dark_master_->setText(QString::fromStdString(cal["dark_master"].get<std::string>()));
    }
    if (cal.contains("flat_master")) {
        cal_flat_master_->setText(QString::fromStdString(cal["flat_master"].get<std::string>()));
    }
}

void ScanTab::set_confirmed_color_mode(const QString &mode) {
    confirmed_color_mode_ = mode.toStdString();
    if (!confirmed_color_mode_.empty()) {
        lbl_confirm_hint_->setText(QString("✓ Confirmed: %1").arg(mode));
        lbl_confirm_hint_->setStyleSheet("color: green; font-weight: bold;");
        
        // Also set in combo box if available
        const int idx = color_mode_select_->findText(mode);
        if (idx >= 0) {
            color_mode_select_->setCurrentIndex(idx);
        }
    }
}

void ScanTab::set_last_scan(const nlohmann::json &scan) {
    last_scan_ = scan;
    if (scan.contains("frames_detected")) {
        frame_count_ = scan.value("frames_detected", 0);
    }
    if (scan.contains("color_mode_candidates") && scan["color_mode_candidates"].is_array()) {
        color_mode_select_->blockSignals(true);
        color_mode_select_->clear();
        for (const auto &c : scan["color_mode_candidates"]) {
            if (c.is_string()) {
                color_mode_select_->addItem(QString::fromStdString(c.get<std::string>()));
            }
        }
        color_mode_select_->blockSignals(false);
    }
}

}
