#pragma once

#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QProgressBar>
#include <QPlainTextEdit>
#include <QGroupBox>
#include <QProcess>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QFile>
#include <string>

namespace tile_compile::gui {

class AstrometryTab : public QWidget {
    Q_OBJECT

  public:
    explicit AstrometryTab(const std::string &project_root, QWidget *parent = nullptr);

    void set_astap_bin(const QString &path);
    void set_astap_data_dir(const QString &path);
    QString get_astap_bin() const;
    QString get_astap_data_dir() const;

  signals:
    void log_message(const QString &msg);

  private slots:
    void on_install_astap();
    void on_download_catalog();
    void on_browse_fits();
    void on_solve();
    void on_solve_finished(int exit_code, QProcess::ExitStatus status);
    void on_save_solved();

  private:
    void build_ui();
    void update_status();
    void append_log(const QString &msg);
    QString astap_data_dir() const;
    QString astap_bin_path() const;
    bool is_astap_installed() const;
    bool is_catalog_installed(const QString &catalog_id) const;
    void start_download(const QUrl &url, const QString &dest_path);
    void on_download_progress(qint64 received, qint64 total);
    void on_download_finished();
    void extract_and_cleanup(const QString &zip_path);
    void parse_wcs(const QString &fits_path);

    std::string project_root_;
    QString astap_bin_override_;
    QString astap_data_dir_override_;

    // Setup section
    QLabel *lbl_astap_status_ = nullptr;
    QLabel *lbl_astap_data_dir_ = nullptr;
    QPushButton *btn_install_astap_ = nullptr;

    // Catalog section
    QComboBox *cmb_catalog_ = nullptr;
    QLabel *lbl_catalog_status_ = nullptr;
    QPushButton *btn_download_catalog_ = nullptr;

    // Solve section
    QLineEdit *edt_fits_path_ = nullptr;
    QPushButton *btn_browse_fits_ = nullptr;
    QPushButton *btn_solve_ = nullptr;
    QPushButton *btn_save_solved_ = nullptr;

    // Results
    QLabel *lbl_ra_ = nullptr;
    QLabel *lbl_dec_ = nullptr;
    QLabel *lbl_fov_ = nullptr;
    QLabel *lbl_rotation_ = nullptr;
    QLabel *lbl_scale_ = nullptr;

    // Progress & Log
    QProgressBar *progress_ = nullptr;
    QPlainTextEdit *log_output_ = nullptr;

    // Network download
    QNetworkAccessManager *nam_ = nullptr;
    QNetworkReply *current_reply_ = nullptr;
    QFile *download_file_ = nullptr;
    enum class DownloadTask { None, InstallAstap, DownloadCatalog };
    DownloadTask download_task_ = DownloadTask::None;
    QString current_download_dest_;

    // Solve state
    QProcess *solve_process_ = nullptr;
    bool solve_ok_ = false;
    QString last_wcs_path_;
};

}
