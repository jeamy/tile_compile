#include "ConfigTab.hpp"
#include "../BackendClient.hpp"
#include "../AssumptionsWidget.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QFile>
#include <QTextStream>
#include <QDir>
#include <QSplitter>
#include <yaml-cpp/yaml.h>
#include <thread>
#include <filesystem>
#include <vector>

namespace tile_compile::gui {

ConfigTab::ConfigTab(BackendClient *backend, AssumptionsWidget *assumptions_widget,
                     const std::string &project_root, QWidget *parent)
    : QWidget(parent), backend_(backend), assumptions_widget_(assumptions_widget), 
      project_root_(project_root) {
    build_ui();
}

void ConfigTab::build_ui() {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(6);

    // --- Top bar: config path + action buttons ---
    auto *top_bar = new QWidget();
    auto *top_layout = new QVBoxLayout(top_bar);
    top_layout->setContentsMargins(8, 6, 8, 6);
    top_layout->setSpacing(6);

    auto *config_row = new QHBoxLayout();
    auto *lbl_path = new QLabel("Config:");
    lbl_path->setFixedWidth(50);
    config_path_ = new QLineEdit("tile_compile.yaml");
    config_path_->setMinimumHeight(25);
    auto *btn_browse_config = new QPushButton("Browse");
    btn_browse_config->setMinimumHeight(25);
    btn_browse_config->setFixedWidth(80);
    config_row->addWidget(lbl_path);
    config_row->addWidget(config_path_, 1);
    config_row->addWidget(btn_browse_config);
    top_layout->addLayout(config_row);

    auto *button_row = new QHBoxLayout();
    btn_cfg_load_ = new QPushButton("Load");
    btn_cfg_load_->setMinimumHeight(25);
    btn_cfg_load_->setFixedWidth(80);
    btn_cfg_save_ = new QPushButton("Save");
    btn_cfg_save_->setMinimumHeight(25);
    btn_cfg_save_->setFixedWidth(80);
    btn_cfg_validate_ = new QPushButton("Validate");
    btn_cfg_validate_->setMinimumHeight(25);
    btn_cfg_validate_->setFixedWidth(80);
    btn_apply_assumptions_ = new QPushButton("Apply Assumptions");
    btn_apply_assumptions_->setMinimumHeight(25);
    btn_apply_assumptions_->setFixedWidth(130);
    lbl_cfg_ = new QLabel("not validated");
    lbl_cfg_->setObjectName("StatusLabel");

    button_row->addWidget(btn_cfg_load_);
    button_row->addWidget(btn_cfg_save_);
    button_row->addWidget(btn_cfg_validate_);
    button_row->addWidget(btn_apply_assumptions_);
    button_row->addWidget(lbl_cfg_);
    button_row->addStretch(1);
    top_layout->addLayout(button_row);

    layout->addWidget(top_bar);

    // --- Horizontal splitter: left = settings, right = YAML editor ---
    auto *splitter = new QSplitter(Qt::Horizontal);
    splitter->setChildrenCollapsible(false);

    // Left panel: scrollable settings
    auto *left_widget = new QWidget();
    auto *left_layout = new QVBoxLayout(left_widget);
    left_layout->setContentsMargins(0, 0, 0, 0);
    left_layout->setSpacing(8);
    build_paths_ui(left_layout);
    left_layout->addStretch(1);

    auto *scroll = new QScrollArea();
    scroll->setWidget(left_widget);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setMinimumWidth(320);

    // Right panel: YAML editor
    auto *right_widget = new QWidget();
    auto *right_layout = new QVBoxLayout(right_widget);
    right_layout->setContentsMargins(0, 0, 0, 0);
    right_layout->setSpacing(4);
    auto *yaml_label = new QLabel("YAML Editor");
    yaml_label->setStyleSheet("font-weight: bold; font-size: 12px; color: #7aa2f7;");
    right_layout->addWidget(yaml_label);

    config_yaml_ = new QTextEdit();
    config_yaml_->setAcceptRichText(false);
    config_yaml_->setPlaceholderText("# YAML config...");
    config_yaml_->setStyleSheet("QTextEdit { font-family: monospace; font-size: 12px; }");
    right_layout->addWidget(config_yaml_, 1);

    splitter->addWidget(scroll);
    splitter->addWidget(right_widget);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 3);

    layout->addWidget(splitter, 1);

    connect(btn_cfg_load_, &QPushButton::clicked, this, &ConfigTab::on_load_config_clicked);
    connect(btn_cfg_save_, &QPushButton::clicked, this, &ConfigTab::on_save_config_clicked);
    connect(btn_cfg_validate_, &QPushButton::clicked, this, &ConfigTab::on_validate_config_clicked);
    connect(btn_apply_assumptions_, &QPushButton::clicked, this, &ConfigTab::on_apply_assumptions_clicked);
    connect(btn_browse_config, &QPushButton::clicked, this, &ConfigTab::on_browse_config);
    connect(config_yaml_, &QTextEdit::textChanged, this, &ConfigTab::on_config_text_changed);
}

void ConfigTab::on_load_config_clicked() {
    emit log_message("[ui] load config button clicked");
    btn_cfg_load_->setEnabled(false);
    lbl_cfg_->setText("loading...");
    emit header_status_changed("loading config...");
    
    const QString path = config_path_->text().trimmed().isEmpty() 
        ? "tile_compile.yaml" 
        : config_path_->text().trimmed();
    
    std::filesystem::path config_path = path.toStdString();
    if (!config_path.is_absolute()) {
        config_path = std::filesystem::path(project_root_) / config_path.string();
    }
    
    if (!std::filesystem::exists(config_path)) {
        lbl_cfg_->setText("error");
        QMessageBox::critical(this, "Load config failed", 
            QString("Config file not found: %1").arg(QString::fromStdString(config_path.string())));
        btn_cfg_load_->setEnabled(true);
        return;
    }
    
    std::thread([this, path]() {
        try {
            const auto cli_sub = backend_->constants().value("CLI", nlohmann::json::object())
                .value("sub", nlohmann::json::object());
            std::vector<std::string> args = {
                cli_sub.value("LOAD_CONFIG", "load-config"),
                path.toStdString()
            };
            
            const auto result = backend_->run_json(project_root_, args);
            
            QMetaObject::invokeMethod(this, [this, result]() {
                const std::string yaml_text = result.value("yaml", "");
                config_yaml_->blockSignals(true);
                config_yaml_->setPlainText(QString::fromStdString(yaml_text));
                config_yaml_->blockSignals(false);
                sync_paths_from_yaml(QString::fromStdString(yaml_text));
                config_validated_ok_ = false;
                lbl_cfg_->setText("not validated");
                extract_assumptions_from_yaml(QString::fromStdString(yaml_text));
                btn_cfg_load_->setEnabled(true);
                emit header_status_changed("idle");
                emit update_controls_requested();
                emit log_message("[ui] config loaded");
            }, Qt::QueuedConnection);
            
        } catch (const std::exception &e) {
            const std::string err_msg = e.what();
            QMetaObject::invokeMethod(this, [this, err_msg]() {
                lbl_cfg_->setText("error");
                emit header_status_changed("idle");
                emit update_controls_requested();
                QMessageBox::critical(this, "Load config failed", QString::fromStdString(err_msg));
                btn_cfg_load_->setEnabled(true);
            }, Qt::QueuedConnection);
        }
    }).detach();
}

void ConfigTab::on_save_config_clicked() {
    emit log_message("[ui] save config button clicked");
    btn_cfg_save_->setEnabled(false);
    lbl_cfg_->setText("saving...");
    emit header_status_changed("saving config...");
    
    const QString path = config_path_->text().trimmed().isEmpty() 
        ? "tile_compile.yaml" 
        : config_path_->text().trimmed();
    const QString yaml_text = config_yaml_->toPlainText();
    
    std::thread([this, path, yaml_text]() {
        try {
            const auto cli_sub = backend_->constants().value("CLI", nlohmann::json::object())
                .value("sub", nlohmann::json::object());
            std::vector<std::string> args = {
                cli_sub.value("SAVE_CONFIG", "save-config"),
                path.toStdString(),
                "--stdin"
            };
            
            backend_->run_json(project_root_, args, yaml_text.toStdString());
            
            QMetaObject::invokeMethod(this, [this]() {
                lbl_cfg_->setText("saved");
                btn_cfg_save_->setEnabled(true);
                emit header_status_changed("idle");
                emit update_controls_requested();
                emit log_message("[ui] config saved");
            }, Qt::QueuedConnection);
            
        } catch (const std::exception &e) {
            const std::string err_msg = e.what();
            QMetaObject::invokeMethod(this, [this, err_msg]() {
                lbl_cfg_->setText("error");
                emit header_status_changed("idle");
                emit update_controls_requested();
                QMessageBox::critical(this, "Save config failed", QString::fromStdString(err_msg));
                btn_cfg_save_->setEnabled(true);
            }, Qt::QueuedConnection);
        }
    }).detach();
}

void ConfigTab::on_validate_config_clicked() {
    emit log_message("[ui] validate config button clicked");
    btn_cfg_validate_->setEnabled(false);
    lbl_cfg_->setText("validating...");
    emit header_status_changed("validating config...");
    
    const QString yaml_text = config_yaml_->toPlainText();
    
    std::thread([this, yaml_text]() {
        try {
            const auto cli_sub = backend_->constants().value("CLI", nlohmann::json::object())
                .value("sub", nlohmann::json::object());
            std::vector<std::string> args = {
                cli_sub.value("VALIDATE_CONFIG", "validate-config"),
                "--yaml", "-", "--stdin"
            };
            
            const auto result = backend_->run_json(project_root_, args, yaml_text.toStdString());
            
            QMetaObject::invokeMethod(this, [this, result]() {
                const bool valid = result.value("valid", false);
                config_validated_ok_ = valid;
                lbl_cfg_->setText(valid ? "ok" : "invalid");
                emit header_status_changed("idle");
                
                if (!valid && result.contains("errors") && result["errors"].is_array()) {
                    std::string msg;
                    for (size_t i = 0; i < std::min(result["errors"].size(), size_t(8)); ++i) {
                        if (i > 0) msg += "\n";
                        msg += result["errors"][i].dump();
                    }
                    if (!msg.empty()) {
                        QMessageBox::warning(this, "Config invalid", QString::fromStdString(msg));
                    }
                }
                
                btn_cfg_validate_->setEnabled(true);
                emit config_validated(valid);
                emit update_controls_requested();
                emit log_message(QString("[ui] config validation: %1").arg(valid ? "ok" : "invalid"));
            }, Qt::QueuedConnection);
            
        } catch (const std::exception &e) {
            const std::string err_msg = e.what();
            QMetaObject::invokeMethod(this, [this, err_msg]() {
                config_validated_ok_ = false;
                lbl_cfg_->setText("error");
                emit header_status_changed("idle");
                emit update_controls_requested();
                QMessageBox::critical(this, "Validate config failed", QString::fromStdString(err_msg));
                btn_cfg_validate_->setEnabled(true);
            }, Qt::QueuedConnection);
        }
    }).detach();
}

void ConfigTab::on_apply_assumptions_clicked() {
    emit log_message("[ui] apply assumptions to config");
    
    try {
        const QString yaml_text = config_yaml_->toPlainText();
        YAML::Node cfg = yaml_text.isEmpty() ? YAML::Node(YAML::NodeType::Map) : YAML::Load(yaml_text.toStdString());
        
        const auto assumptions = assumptions_widget_->get_assumptions();
        cfg["assumptions"] = YAML::Load(assumptions.dump());
        
        YAML::Emitter out;
        out << cfg;
        
        config_yaml_->blockSignals(true);
        config_yaml_->setPlainText(QString::fromStdString(out.c_str()));
        config_yaml_->blockSignals(false);
        
        config_validated_ok_ = false;
        lbl_cfg_->setText("not validated (assumptions applied)");
        emit config_edited();
        
    } catch (const std::exception &e) {
        QMessageBox::warning(this, "Apply Assumptions", QString("Failed to apply assumptions: %1").arg(e.what()));
    }
}

void ConfigTab::on_browse_config() {
    const QString start = config_path_->text().trimmed().isEmpty() 
        ? QString::fromStdString(project_root_) 
        : config_path_->text();
    const QString p = QFileDialog::getOpenFileName(this, "Select config YAML", start, 
        "YAML Files (*.yaml *.yml);;All Files (*)");
    if (!p.isEmpty()) {
        config_path_->setText(p);
    }
}

void ConfigTab::on_config_text_changed() {
    config_validated_ok_ = false;
    lbl_cfg_->setText("not validated");
    if (!syncing_paths_) {
        sync_paths_from_yaml(config_yaml_->toPlainText());
    }
    emit astrometry_paths_changed(edt_astap_bin_->text().trimmed(),
                                  edt_astap_data_dir_->text().trimmed());
    emit config_edited();
    emit update_controls_requested();
}

void ConfigTab::extract_assumptions_from_yaml(const QString &yaml_text) {
    try {
        const YAML::Node cfg = YAML::Load(yaml_text.toStdString());
        if (cfg["assumptions"]) {
            const std::string assumptions_str = YAML::Dump(cfg["assumptions"]);
            const auto assumptions = nlohmann::json::parse(assumptions_str);
            assumptions_widget_->set_assumptions(assumptions);
        }
    } catch (...) {
        // Ignore parse errors
    }
}

QString ConfigTab::get_config_path() const {
    return config_path_->text().trimmed();
}

QString ConfigTab::get_config_yaml() const {
    return config_yaml_->toPlainText();
}

void ConfigTab::set_config_path(const QString &path) {
    config_path_->setText(path);
}

void ConfigTab::set_config_yaml(const QString &yaml) {
    config_yaml_->blockSignals(true);
    config_yaml_->setPlainText(yaml);
    config_yaml_->blockSignals(false);
    sync_paths_from_yaml(yaml);
    emit astrometry_paths_changed(edt_astap_bin_->text().trimmed(),
                                  edt_astap_data_dir_->text().trimmed());
}

void ConfigTab::set_config_validated(bool validated) {
    config_validated_ok_ = validated;
    lbl_cfg_->setText(validated ? "ok" : "not validated");
}

void ConfigTab::build_paths_ui(QVBoxLayout *parent) {
    // Helper: create a path row with browse button
    auto make_path_row = [this](QLineEdit *&edt, const QString &placeholder, bool is_dir) {
        auto *row = new QHBoxLayout();
        edt = new QLineEdit();
        edt->setPlaceholderText(placeholder);
        edt->setMinimumHeight(26);
        row->addWidget(edt, 1);
        auto *btn = new QPushButton("...");
        btn->setFixedSize(30, 26);
        row->addWidget(btn);
        connect(btn, &QPushButton::clicked, this, [this, edt, is_dir]() {
            on_browse_path(edt, is_dir);
        });
        connect(edt, &QLineEdit::editingFinished, this, &ConfigTab::on_path_editor_changed);
        return row;
    };

    // === Astrometry Section ===
    auto *astro_box = new QGroupBox("Astrometry");
    auto *astro_form = new QFormLayout(astro_box);
    astro_form->setSpacing(6);
    astro_form->setContentsMargins(8, 14, 8, 8);

    chk_astro_enabled_ = new QCheckBox("Enabled");
    astro_form->addRow("", chk_astro_enabled_);
    connect(chk_astro_enabled_, &QCheckBox::toggled, this, &ConfigTab::on_path_editor_changed);

    astro_form->addRow("ASTAP binary:", make_path_row(edt_astap_bin_,
        "(default: ~/.local/share/tile_compile/astap/astap_cli)", false));
    astro_form->addRow("ASTAP data dir:", make_path_row(edt_astap_data_dir_,
        "(default: ~/.local/share/tile_compile/astap/)", true));

    spn_astro_search_radius_ = new QSpinBox();
    spn_astro_search_radius_->setRange(1, 360);
    spn_astro_search_radius_->setValue(180);
    spn_astro_search_radius_->setSuffix(" deg");
    astro_form->addRow("Search radius:", spn_astro_search_radius_);
    connect(spn_astro_search_radius_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ConfigTab::on_path_editor_changed);

    parent->addWidget(astro_box);

    // === PCC Section ===
    auto *pcc_box = new QGroupBox("Photometric Color Calibration (PCC)");
    auto *pcc_form = new QFormLayout(pcc_box);
    pcc_form->setSpacing(6);
    pcc_form->setContentsMargins(8, 14, 8, 8);

    chk_pcc_enabled_ = new QCheckBox("Enabled");
    pcc_form->addRow("", chk_pcc_enabled_);
    connect(chk_pcc_enabled_, &QCheckBox::toggled, this, &ConfigTab::on_path_editor_changed);

    cmb_pcc_source_ = new QComboBox();
    cmb_pcc_source_->addItems({"auto", "siril", "vizier_gaia", "vizier_apass"});
    pcc_form->addRow("Source:", cmb_pcc_source_);
    connect(cmb_pcc_source_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ConfigTab::on_path_editor_changed);

    pcc_form->addRow("Siril catalog dir:", make_path_row(edt_siril_catalog_dir_,
        "(default: ~/.local/share/siril/siril_cat1_healpix8_xpsamp/)", true));

    auto make_dspin = [this](QDoubleSpinBox *&spn, double min, double max, double val,
                             int decimals, const QString &suffix) {
        spn = new QDoubleSpinBox();
        spn->setRange(min, max);
        spn->setValue(val);
        spn->setDecimals(decimals);
        if (!suffix.isEmpty()) spn->setSuffix(suffix);
        connect(spn, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, &ConfigTab::on_path_editor_changed);
    };

    make_dspin(spn_pcc_mag_limit_, 1.0, 22.0, 14.0, 1, " mag");
    pcc_form->addRow("Mag limit (faint):", spn_pcc_mag_limit_);

    make_dspin(spn_pcc_mag_bright_, 0.0, 15.0, 6.0, 1, " mag");
    pcc_form->addRow("Mag limit (bright):", spn_pcc_mag_bright_);

    make_dspin(spn_pcc_aperture_, 1.0, 50.0, 8.0, 1, " px");
    pcc_form->addRow("Aperture radius:", spn_pcc_aperture_);

    make_dspin(spn_pcc_ann_inner_, 1.0, 80.0, 12.0, 1, " px");
    pcc_form->addRow("Annulus inner:", spn_pcc_ann_inner_);

    make_dspin(spn_pcc_ann_outer_, 2.0, 100.0, 18.0, 1, " px");
    pcc_form->addRow("Annulus outer:", spn_pcc_ann_outer_);

    spn_pcc_min_stars_ = new QSpinBox();
    spn_pcc_min_stars_->setRange(3, 500);
    spn_pcc_min_stars_->setValue(10);
    pcc_form->addRow("Min stars:", spn_pcc_min_stars_);
    connect(spn_pcc_min_stars_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ConfigTab::on_path_editor_changed);

    make_dspin(spn_pcc_sigma_, 1.0, 10.0, 2.5, 1, " σ");
    pcc_form->addRow("Sigma clip:", spn_pcc_sigma_);

    parent->addWidget(pcc_box);
}

void ConfigTab::on_browse_path(QLineEdit *target, bool directory) {
    QString start = target->text().trimmed();
    if (start.isEmpty()) start = QDir::homePath();

    QString result;
    if (directory) {
        result = QFileDialog::getExistingDirectory(this, "Select directory", start);
    } else {
        result = QFileDialog::getOpenFileName(this, "Select file", start);
    }
    if (!result.isEmpty()) {
        target->setText(result);
        on_path_editor_changed();
    }
}

void ConfigTab::on_path_editor_changed() {
    if (syncing_paths_) return;
    sync_paths_to_yaml();
}

void ConfigTab::sync_paths_to_yaml() {
    syncing_paths_ = true;
    try {
        const QString yaml_text = config_yaml_->toPlainText();
        YAML::Node cfg = yaml_text.trimmed().isEmpty()
            ? YAML::Node(YAML::NodeType::Map)
            : YAML::Load(yaml_text.toStdString());

        // Astrometry
        cfg["astrometry"]["enabled"] = chk_astro_enabled_->isChecked();
        cfg["astrometry"]["astap_bin"] = edt_astap_bin_->text().trimmed().toStdString();
        cfg["astrometry"]["astap_data_dir"] = edt_astap_data_dir_->text().trimmed().toStdString();
        cfg["astrometry"]["search_radius"] = spn_astro_search_radius_->value();

        // PCC
        cfg["pcc"]["enabled"] = chk_pcc_enabled_->isChecked();
        cfg["pcc"]["source"] = cmb_pcc_source_->currentText().toStdString();
        cfg["pcc"]["siril_catalog_dir"] = edt_siril_catalog_dir_->text().trimmed().toStdString();
        cfg["pcc"]["mag_limit"] = spn_pcc_mag_limit_->value();
        cfg["pcc"]["mag_bright_limit"] = spn_pcc_mag_bright_->value();
        cfg["pcc"]["aperture_radius_px"] = spn_pcc_aperture_->value();
        cfg["pcc"]["annulus_inner_px"] = spn_pcc_ann_inner_->value();
        cfg["pcc"]["annulus_outer_px"] = spn_pcc_ann_outer_->value();
        cfg["pcc"]["min_stars"] = spn_pcc_min_stars_->value();
        cfg["pcc"]["sigma_clip"] = spn_pcc_sigma_->value();

        YAML::Emitter out;
        out << cfg;
        config_yaml_->blockSignals(true);
        config_yaml_->setPlainText(QString::fromStdString(out.c_str()));
        config_yaml_->blockSignals(false);

        config_validated_ok_ = false;
        lbl_cfg_->setText("not validated");
        emit astrometry_paths_changed(edt_astap_bin_->text().trimmed(),
                                      edt_astap_data_dir_->text().trimmed());
        emit config_edited();
        emit update_controls_requested();
    } catch (...) {
        // Ignore YAML errors during sync
    }
    emit astrometry_paths_changed(edt_astap_bin_->text().trimmed(),
                                  edt_astap_data_dir_->text().trimmed());
    syncing_paths_ = false;
}

void ConfigTab::sync_paths_from_yaml(const QString &yaml_text) {
    if (syncing_paths_) return;
    syncing_paths_ = true;
    try {
        const YAML::Node cfg = YAML::Load(yaml_text.toStdString());

        auto set_text = [](QLineEdit *edt, const YAML::Node &n) {
            if (n && n.IsScalar()) edt->setText(QString::fromStdString(n.as<std::string>()));
        };
        auto set_bool = [](QCheckBox *chk, const YAML::Node &n) {
            if (n) chk->setChecked(n.as<bool>());
        };
        auto set_int = [](QSpinBox *spn, const YAML::Node &n) {
            if (n) spn->setValue(n.as<int>());
        };
        auto set_dbl = [](QDoubleSpinBox *spn, const YAML::Node &n) {
            if (n) spn->setValue(n.as<double>());
        };

        if (cfg["astrometry"]) {
            auto a = cfg["astrometry"];
            set_bool(chk_astro_enabled_, a["enabled"]);
            set_text(edt_astap_bin_, a["astap_bin"]);
            set_text(edt_astap_data_dir_, a["astap_data_dir"]);
            set_int(spn_astro_search_radius_, a["search_radius"]);
        }

        if (cfg["pcc"]) {
            auto p = cfg["pcc"];
            set_bool(chk_pcc_enabled_, p["enabled"]);
            if (p["source"] && p["source"].IsScalar()) {
                int idx = cmb_pcc_source_->findText(
                    QString::fromStdString(p["source"].as<std::string>()));
                if (idx >= 0) cmb_pcc_source_->setCurrentIndex(idx);
            }
            set_text(edt_siril_catalog_dir_, p["siril_catalog_dir"]);
            set_dbl(spn_pcc_mag_limit_, p["mag_limit"]);
            set_dbl(spn_pcc_mag_bright_, p["mag_bright_limit"]);
            set_dbl(spn_pcc_aperture_, p["aperture_radius_px"]);
            set_dbl(spn_pcc_ann_inner_, p["annulus_inner_px"]);
            set_dbl(spn_pcc_ann_outer_, p["annulus_outer_px"]);
            set_int(spn_pcc_min_stars_, p["min_stars"]);
            set_dbl(spn_pcc_sigma_, p["sigma_clip"]);
        }
    } catch (...) {
        // Ignore parse errors
    }
    syncing_paths_ = false;
}

}
