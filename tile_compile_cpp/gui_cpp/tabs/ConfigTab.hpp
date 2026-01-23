#pragma once

#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QComboBox>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

namespace tile_compile::gui {

class BackendClient;
class AssumptionsWidget;

class ConfigTab : public QWidget {
    Q_OBJECT

  public:
    explicit ConfigTab(BackendClient *backend, AssumptionsWidget *assumptions_widget,
                      const std::string &project_root, QWidget *parent = nullptr);

    QString get_config_path() const;
    QString get_config_yaml() const;
    bool is_config_validated() const { return config_validated_ok_; }
    
    void set_config_path(const QString &path);
    void set_config_yaml(const QString &yaml);
    void set_config_validated(bool validated);

  signals:
    void config_edited();
    void config_validated(bool ok);
    void log_message(const QString &msg);
    void header_status_changed(const QString &status);
    void update_controls_requested();

  private slots:
    void on_load_config_clicked();
    void on_save_config_clicked();
    void on_validate_config_clicked();
    void on_apply_assumptions_clicked();
    void on_apply_v4_preset();
    void on_browse_config();
    void on_config_text_changed();

  private:
    void build_ui();
    void extract_assumptions_from_yaml(const QString &yaml_text);

    BackendClient *backend_;
    AssumptionsWidget *assumptions_widget_;
    std::string project_root_;
    bool config_validated_ok_ = false;
    
    QLineEdit *config_path_ = nullptr;
    QPushButton *btn_cfg_load_ = nullptr;
    QPushButton *btn_cfg_save_ = nullptr;
    QPushButton *btn_cfg_validate_ = nullptr;
    QPushButton *btn_apply_assumptions_ = nullptr;
    QComboBox *v4_preset_combo_ = nullptr;
    QLabel *lbl_cfg_ = nullptr;
    QTextEdit *config_yaml_ = nullptr;
};

}
