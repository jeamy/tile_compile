#pragma once

#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QGroupBox>
#include <string>

namespace tile_compile::gui {

class PCCTab : public QWidget {
    Q_OBJECT

  public:
    explicit PCCTab(const std::string &project_root, QWidget *parent = nullptr);

  signals:
    void log_message(const QString &msg);

  private slots:
    void on_browse_fits();
    void on_browse_wcs();
    void on_run_pcc();

  private:
    void build_ui();
    void append_log(const QString &msg);

    std::string project_root_;

    // Input
    QLineEdit *edt_fits_path_ = nullptr;
    QLineEdit *edt_wcs_path_ = nullptr;
    QComboBox *cmb_source_ = nullptr;

    // Parameters
    QDoubleSpinBox *spn_mag_limit_ = nullptr;
    QDoubleSpinBox *spn_mag_bright_ = nullptr;
    QDoubleSpinBox *spn_aperture_ = nullptr;
    QDoubleSpinBox *spn_ann_inner_ = nullptr;
    QDoubleSpinBox *spn_ann_outer_ = nullptr;
    QSpinBox *spn_min_stars_ = nullptr;
    QDoubleSpinBox *spn_sigma_ = nullptr;

    // Results
    QLabel *lbl_stars_matched_ = nullptr;
    QLabel *lbl_stars_used_ = nullptr;
    QLabel *lbl_residual_ = nullptr;
    QLabel *lbl_matrix_ = nullptr;

    // Controls
    QPushButton *btn_run_ = nullptr;
    QPushButton *btn_save_ = nullptr;
    QProgressBar *progress_ = nullptr;
    QPlainTextEdit *log_output_ = nullptr;

    // State
    bool pcc_ok_ = false;
};

} // namespace tile_compile::gui
