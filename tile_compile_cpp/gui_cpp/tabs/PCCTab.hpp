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
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QFile>
#include <string>
#include "tile_compile/core/types.hpp"
#include "tile_compile/io/fits_io.hpp"

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
    void on_download_catalog();
    void on_cancel_download();

  private:
    void build_ui();
    void append_log(const QString &msg);
    void update_catalog_status();
    QString siril_catalog_dir() const;
    void download_next_chunk();
    void on_chunk_download_progress(qint64 received, qint64 total);
    void on_chunk_download_finished();

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

    // Catalog download
    QLabel *lbl_catalog_status_ = nullptr;
    QPushButton *btn_download_catalog_ = nullptr;
    QPushButton *btn_cancel_download_ = nullptr;
    QProgressBar *progress_download_ = nullptr;
    QNetworkAccessManager *nam_ = nullptr;
    QNetworkReply *current_reply_ = nullptr;
    QFile *download_file_ = nullptr;
    std::vector<int> chunks_to_download_;
    int current_chunk_idx_ = 0;
    bool download_cancelled_ = false;

    // Controls
    QPushButton *btn_run_ = nullptr;
    QPushButton *btn_save_ = nullptr;
    QProgressBar *progress_ = nullptr;
    QPlainTextEdit *log_output_ = nullptr;

    // State
    bool pcc_ok_ = false;
    tile_compile::Matrix2Df pcc_R_, pcc_G_, pcc_B_;
    tile_compile::io::FitsHeader pcc_hdr_;
};

} // namespace tile_compile::gui
