#pragma once

#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QSplitter>
#include <QScrollArea>
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
    void on_browse_config();
    void on_config_text_changed();
    void on_browse_path(QLineEdit *target, bool directory);
    void on_path_editor_changed();

  private:
    void build_ui();
    void build_paths_ui(QVBoxLayout *parent);
    void extract_assumptions_from_yaml(const QString &yaml_text);
    void sync_paths_to_yaml();
    void sync_paths_from_yaml(const QString &yaml_text);

    BackendClient *backend_;
    AssumptionsWidget *assumptions_widget_;
    std::string project_root_;
    bool config_validated_ok_ = false;
    bool syncing_paths_ = false;
    
    QLineEdit *config_path_ = nullptr;
    QPushButton *btn_cfg_load_ = nullptr;
    QPushButton *btn_cfg_save_ = nullptr;
    QPushButton *btn_cfg_validate_ = nullptr;
    QPushButton *btn_apply_assumptions_ = nullptr;
    QLabel *lbl_cfg_ = nullptr;
    QTextEdit *config_yaml_ = nullptr;

    // Path editors (synced with YAML)
    QLineEdit *edt_astap_bin_ = nullptr;
    QLineEdit *edt_astap_data_dir_ = nullptr;
    QSpinBox  *spn_astro_search_radius_ = nullptr;
    QCheckBox *chk_astro_enabled_ = nullptr;
    QCheckBox *chk_pcc_enabled_ = nullptr;
    QComboBox *cmb_pcc_source_ = nullptr;
    QLineEdit *edt_siril_catalog_dir_ = nullptr;
    QDoubleSpinBox *spn_pcc_mag_limit_ = nullptr;
    QDoubleSpinBox *spn_pcc_mag_bright_ = nullptr;
    QDoubleSpinBox *spn_pcc_aperture_ = nullptr;
    QDoubleSpinBox *spn_pcc_ann_inner_ = nullptr;
    QDoubleSpinBox *spn_pcc_ann_outer_ = nullptr;
    QSpinBox  *spn_pcc_min_stars_ = nullptr;
    QDoubleSpinBox *spn_pcc_sigma_ = nullptr;
};

}
