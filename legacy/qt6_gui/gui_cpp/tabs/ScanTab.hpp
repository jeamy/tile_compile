#pragma once

#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QListWidget>
#include <QStringList>
#include <string>

#include <nlohmann/json.hpp>

namespace tile_compile::gui {

class BackendClient;

class ScanTab : public QWidget {
    Q_OBJECT

  public:
    explicit ScanTab(BackendClient *backend, const std::string &project_root, QWidget *parent = nullptr);

    nlohmann::json get_scan_result() const { return last_scan_; }
    std::string get_confirmed_color_mode() const { return confirmed_color_mode_; }
    int get_frame_count() const { return frame_count_; }
    
    void set_input_dir_from_scan(const QString &dir);
    QString get_scan_input_dir() const;
    QStringList get_scan_input_dirs() const;
    void set_scan_input_dirs(const QStringList &dirs);
    nlohmann::json get_last_scan() const { return last_scan_; }
    void update_calibration_controls();
    std::string validate_calibration() const;
    nlohmann::json collect_calibration() const;
    void apply_calibration(const nlohmann::json &cal);
    void set_confirmed_color_mode(const QString &mode);
    void set_last_scan(const nlohmann::json &scan);

  signals:
    void scan_completed(int frame_count);
    void color_mode_confirmed(const QString &mode);
    void log_message(const QString &msg);
    void header_status_changed(const QString &status);
    void input_dir_changed(const QString &dir);
    void input_dirs_changed(const QStringList &dirs);
    void calibration_changed();
    void update_controls_requested();

  private slots:
    void on_scan_clicked();
    void on_confirm_color_clicked();
    void on_browse_scan_dir();
    void on_add_scan_dir();
    void on_remove_scan_dir();
    void on_browse_bias_dir();
    void on_browse_darks_dir();
    void on_browse_flats_dir();
    void on_browse_bias_master();
    void on_browse_dark_master();
    void on_browse_flat_master();
    void on_calibration_changed();

  private:
    void build_ui();

    BackendClient *backend_;
    std::string project_root_;
    
    nlohmann::json last_scan_;
    std::string confirmed_color_mode_;
    int frame_count_ = 0;
    
    QLineEdit *scan_input_dir_ = nullptr;
    QListWidget *scan_input_dirs_list_ = nullptr;
    QPushButton *btn_add_scan_dir_ = nullptr;
    QPushButton *btn_remove_scan_dir_ = nullptr;
    QPushButton *btn_scan_ = nullptr;
    QLabel *lbl_scan_ = nullptr;
    QLabel *scan_msg_ = nullptr;
    QSpinBox *scan_frames_min_ = nullptr;
    QCheckBox *scan_with_checksums_ = nullptr;
    QComboBox *color_mode_select_ = nullptr;
    QPushButton *btn_confirm_color_ = nullptr;
    QLabel *lbl_confirm_hint_ = nullptr;
    QLabel *scan_reduced_mode_hint_ = nullptr;
    
    QCheckBox *cal_use_bias_ = nullptr;
    QCheckBox *cal_use_dark_ = nullptr;
    QCheckBox *cal_use_flat_ = nullptr;
    QCheckBox *cal_bias_use_master_ = nullptr;
    QCheckBox *cal_dark_use_master_ = nullptr;
    QCheckBox *cal_flat_use_master_ = nullptr;
    QLineEdit *cal_bias_dir_ = nullptr;
    QLineEdit *cal_darks_dir_ = nullptr;
    QLineEdit *cal_flats_dir_ = nullptr;
    QLineEdit *cal_bias_master_ = nullptr;
    QLineEdit *cal_dark_master_ = nullptr;
    QLineEdit *cal_flat_master_ = nullptr;
    
    QPushButton *btn_browse_bias_dir_ = nullptr;
    QPushButton *btn_browse_darks_dir_ = nullptr;
    QPushButton *btn_browse_flats_dir_ = nullptr;
    QPushButton *btn_browse_bias_master_ = nullptr;
    QPushButton *btn_browse_dark_master_ = nullptr;
    QPushButton *btn_browse_flat_master_ = nullptr;
};

}
