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
#include <yaml-cpp/yaml.h>
#include <thread>
#include <filesystem>

namespace tile_compile::gui {

ConfigTab::ConfigTab(BackendClient *backend, AssumptionsWidget *assumptions_widget,
                     const std::string &project_root, QWidget *parent)
    : QWidget(parent), backend_(backend), assumptions_widget_(assumptions_widget), 
      project_root_(project_root) {
    build_ui();
}

void ConfigTab::build_ui() {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);
    
    auto *cfg_box = new QGroupBox("Configuration");
    auto *cfg_layout = new QVBoxLayout(cfg_box);
    cfg_layout->setContentsMargins(12, 18, 12, 12);
    cfg_layout->setSpacing(10);
    
    auto *cfg_form = new QFormLayout();
    cfg_form->setSpacing(10);
    
    auto *config_row = new QHBoxLayout();
    config_path_ = new QLineEdit("tile_compile.yaml");
    config_path_->setMinimumHeight(30);
    auto *btn_browse_config = new QPushButton("Browse");
    btn_browse_config->setMinimumHeight(30);
    btn_browse_config->setFixedWidth(100);
    config_row->addWidget(config_path_);
    config_row->addWidget(btn_browse_config);
    cfg_form->addRow("Config path", config_row);
    
    auto *button_row = new QHBoxLayout();
    btn_cfg_load_ = new QPushButton("Load");
    btn_cfg_load_->setMinimumHeight(30);
    btn_cfg_load_->setFixedWidth(100);
    btn_cfg_save_ = new QPushButton("Save");
    btn_cfg_save_->setMinimumHeight(30);
    btn_cfg_save_->setFixedWidth(100);
    btn_cfg_validate_ = new QPushButton("Validate");
    btn_cfg_validate_->setMinimumHeight(30);
    btn_cfg_validate_->setFixedWidth(100);
    lbl_cfg_ = new QLabel("not validated");
    lbl_cfg_->setObjectName("StatusLabel");
    btn_apply_assumptions_ = new QPushButton("Apply Assumptions");
    btn_apply_assumptions_->setMinimumHeight(30);
    btn_apply_assumptions_->setFixedWidth(140);
    
    v4_preset_combo_ = new QComboBox();
    v4_preset_combo_->addItem("Preset 1: EQ-Montierung, ruhiges Seeing", 1);
    v4_preset_combo_->addItem("Preset 2: Alt/Az, starke Feldrotation", 2);
    v4_preset_combo_->addItem("Preset 3: Polnähe, sehr instabil", 3);
    v4_preset_combo_->setCurrentIndex(1);
    v4_preset_combo_->setMinimumHeight(30);
    v4_preset_combo_->setFixedWidth(280);
    
    button_row->addWidget(new QLabel("v4 Preset:"));
    button_row->addWidget(v4_preset_combo_);
    button_row->addWidget(btn_cfg_load_);
    button_row->addWidget(btn_cfg_save_);
    button_row->addWidget(btn_cfg_validate_);
    button_row->addWidget(btn_apply_assumptions_);
    button_row->addWidget(lbl_cfg_);
    button_row->addStretch(1);
    cfg_form->addRow("", button_row);
    cfg_layout->addLayout(cfg_form);
    
    config_yaml_ = new QTextEdit();
    config_yaml_->setAcceptRichText(false);
    config_yaml_->setPlaceholderText("# YAML config...");
    cfg_layout->addWidget(config_yaml_);
    
    layout->addWidget(cfg_box, 1);
    
    connect(btn_cfg_load_, &QPushButton::clicked, this, &ConfigTab::on_load_config_clicked);
    connect(btn_cfg_save_, &QPushButton::clicked, this, &ConfigTab::on_save_config_clicked);
    connect(btn_cfg_validate_, &QPushButton::clicked, this, &ConfigTab::on_validate_config_clicked);
    connect(btn_apply_assumptions_, &QPushButton::clicked, this, &ConfigTab::on_apply_assumptions_clicked);
    connect(v4_preset_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &ConfigTab::on_apply_v4_preset);
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

void ConfigTab::on_apply_v4_preset() {
    const int preset_id = v4_preset_combo_->currentData().toInt();
    emit log_message(QString("[ui] applying v4 preset %1").arg(preset_id));
    
    try {
        const QString yaml_text = config_yaml_->toPlainText();
        YAML::Node cfg = yaml_text.isEmpty() ? YAML::Node(YAML::NodeType::Map) : YAML::Load(yaml_text.toStdString());
        
        if (!cfg["v4"]) {
            cfg["v4"] = YAML::Node(YAML::NodeType::Map);
        }
        
        if (preset_id == 1) {
            cfg["v4"]["iterations"] = 2;
            cfg["v4"]["beta"] = 3.0;
            cfg["v4"]["adaptive_tiles"]["enabled"] = false;
        } else if (preset_id == 2) {
            cfg["v4"]["iterations"] = 4;
            cfg["v4"]["beta"] = 6.0;
            cfg["v4"]["adaptive_tiles"]["enabled"] = true;
            cfg["v4"]["adaptive_tiles"]["max_refine_passes"] = 3;
            cfg["v4"]["adaptive_tiles"]["refine_variance_threshold"] = 0.15;
        } else if (preset_id == 3) {
            cfg["v4"]["iterations"] = 5;
            cfg["v4"]["beta"] = 8.0;
            cfg["v4"]["adaptive_tiles"]["enabled"] = true;
            cfg["v4"]["adaptive_tiles"]["max_refine_passes"] = 4;
            cfg["v4"]["adaptive_tiles"]["refine_variance_threshold"] = 0.1;
        }
        
        YAML::Emitter out;
        out << cfg;
        
        config_yaml_->setPlainText(QString::fromStdString(out.c_str()));
        emit log_message(QString("[ui] v4 preset %1 applied").arg(preset_id));
        
    } catch (const std::exception &e) {
        emit log_message(QString("[ui] error applying preset: %1").arg(e.what()));
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
}

void ConfigTab::set_config_validated(bool validated) {
    config_validated_ok_ = validated;
    lbl_cfg_->setText(validated ? "ok" : "not validated");
}

}
