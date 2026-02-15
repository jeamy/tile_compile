#include "tabs/AstrometryTab.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/io/fits_io.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>
#include <QMessageBox>
#include <QScrollBar>
#include <QStandardPaths>
#include <QUrl>
#include <QDirIterator>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>

namespace tile_compile::gui {

namespace {

// ASTAP CLI download URL (SourceForge, Linux amd64)
const char *ASTAP_CLI_URL =
    "https://sourceforge.net/projects/astap-program/files/"
    "linux_installer/astap_command-line_version_Linux_amd64.zip/download";

struct CatalogInfo {
    const char *id;
    const char *label;
    const char *url;
    const char *description;
};

// SourceForge direct download links for ASTAP star databases
const char *SF_BASE =
    "https://sourceforge.net/projects/astap-program/files/star_databases/";

const CatalogInfo CATALOGS[] = {
    {"d05", "D05 (~100 MB, smallest)",
     "d05_star_database.zip",
     "Smallest database, sufficient for FOV >= 0.6 deg"},
    {"d20", "D20 (~400 MB)",
     "d20_star_database.zip",
     "Medium database, more stars"},
    {"d50", "D50 (~800 MB, recommended)",
     "d50_star_database.zip",
     "Recommended: good balance of size and coverage"},
    {"d80", "D80 (~1.25 GB, largest)",
     "d80_star_database.deb",
     "Largest database, maximum coverage"},
};
const int NUM_CATALOGS = 4;

} // namespace

AstrometryTab::AstrometryTab(const std::string &project_root, QWidget *parent)
    : QWidget(parent), project_root_(project_root) {
    nam_ = new QNetworkAccessManager(this);
    build_ui();
    update_status();
}

void AstrometryTab::build_ui() {
    auto *root = new QVBoxLayout(this);
    root->setContentsMargins(12, 12, 12, 12);
    root->setSpacing(12);

    // === Setup Section ===
    auto *setup_box = new QGroupBox("ASTAP Setup");
    auto *setup_layout = new QVBoxLayout(setup_box);
    setup_layout->setSpacing(8);

    auto *astap_row = new QHBoxLayout();
    lbl_astap_status_ = new QLabel("Checking...");
    lbl_astap_status_->setStyleSheet("font-weight: bold;");
    astap_row->addWidget(new QLabel("ASTAP CLI:"));
    astap_row->addWidget(lbl_astap_status_, 1);
    btn_install_astap_ = new QPushButton("Install ASTAP CLI");
    astap_row->addWidget(btn_install_astap_);
    setup_layout->addLayout(astap_row);

    lbl_astap_data_dir_ = new QLabel(
        "ASTAP data directory: <b>" + astap_data_dir() + "</b>");
    lbl_astap_data_dir_->setWordWrap(true);
    setup_layout->addWidget(lbl_astap_data_dir_);

    root->addWidget(setup_box);

    // === Catalog Section ===
    auto *catalog_box = new QGroupBox("Star Database");
    auto *catalog_layout = new QVBoxLayout(catalog_box);
    catalog_layout->setSpacing(8);

    auto *cat_row = new QHBoxLayout();
    cmb_catalog_ = new QComboBox();
    for (int i = 0; i < NUM_CATALOGS; ++i) {
        cmb_catalog_->addItem(CATALOGS[i].label, CATALOGS[i].id);
    }
    cmb_catalog_->setCurrentIndex(2); // D50 default
    cat_row->addWidget(new QLabel("Database:"));
    cat_row->addWidget(cmb_catalog_, 1);
    btn_download_catalog_ = new QPushButton("Download");
    cat_row->addWidget(btn_download_catalog_);
    catalog_layout->addLayout(cat_row);

    lbl_catalog_status_ = new QLabel("Checking...");
    catalog_layout->addWidget(lbl_catalog_status_);

    root->addWidget(catalog_box);

    // === Solve Section ===
    auto *solve_box = new QGroupBox("Plate Solve");
    auto *solve_layout = new QVBoxLayout(solve_box);
    solve_layout->setSpacing(8);

    auto *fits_row = new QHBoxLayout();
    edt_fits_path_ = new QLineEdit();
    edt_fits_path_->setPlaceholderText("Path to FITS file (e.g. stacked.fits)");
    fits_row->addWidget(edt_fits_path_, 1);
    btn_browse_fits_ = new QPushButton("Browse...");
    fits_row->addWidget(btn_browse_fits_);
    btn_solve_ = new QPushButton("Solve");
    btn_solve_->setStyleSheet("font-weight: bold;");
    fits_row->addWidget(btn_solve_);
    btn_save_solved_ = new QPushButton("Save Solved");
    btn_save_solved_->setEnabled(false);
    btn_save_solved_->setToolTip("Save FITS with WCS header keywords");
    fits_row->addWidget(btn_save_solved_);
    solve_layout->addLayout(fits_row);

    // Results grid
    auto *results = new QFormLayout();
    results->setSpacing(4);
    lbl_ra_ = new QLabel("-");
    lbl_dec_ = new QLabel("-");
    lbl_fov_ = new QLabel("-");
    lbl_rotation_ = new QLabel("-");
    lbl_scale_ = new QLabel("-");
    results->addRow("RA (J2000):", lbl_ra_);
    results->addRow("Dec (J2000):", lbl_dec_);
    results->addRow("Field of View:", lbl_fov_);
    results->addRow("Rotation:", lbl_rotation_);
    results->addRow("Pixel Scale:", lbl_scale_);
    solve_layout->addLayout(results);

    root->addWidget(solve_box);

    // === Progress & Log ===
    progress_ = new QProgressBar();
    progress_->setRange(0, 0); // indeterminate
    progress_->setVisible(false);
    root->addWidget(progress_);

    log_output_ = new QPlainTextEdit();
    log_output_->setReadOnly(true);
    log_output_->setMaximumHeight(200);
    log_output_->setPlaceholderText("Log output...");
    root->addWidget(log_output_, 1);

    // Connections
    connect(btn_install_astap_, &QPushButton::clicked, this, &AstrometryTab::on_install_astap);
    connect(btn_download_catalog_, &QPushButton::clicked, this, &AstrometryTab::on_download_catalog);
    connect(btn_browse_fits_, &QPushButton::clicked, this, &AstrometryTab::on_browse_fits);
    connect(btn_solve_, &QPushButton::clicked, this, &AstrometryTab::on_solve);
    connect(btn_save_solved_, &QPushButton::clicked, this, &AstrometryTab::on_save_solved);
    connect(cmb_catalog_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { update_status(); });
}

QString AstrometryTab::astap_data_dir() const {
    if (!astap_data_dir_override_.isEmpty())
        return astap_data_dir_override_;
    return QDir::homePath() + "/.local/share/tile_compile/astap";
}

QString AstrometryTab::astap_bin_path() const {
    if (!astap_bin_override_.isEmpty())
        return astap_bin_override_;
    return astap_data_dir() + "/astap_cli";
}

void AstrometryTab::set_astap_bin(const QString &path) {
    astap_bin_override_ = path;
    update_status();
}

void AstrometryTab::set_astap_data_dir(const QString &path) {
    astap_data_dir_override_ = path;
    update_status();
}

QString AstrometryTab::get_astap_bin() const {
    return astap_bin_path();
}

QString AstrometryTab::get_astap_data_dir() const {
    return astap_data_dir();
}

bool AstrometryTab::is_astap_installed() const {
    return QFileInfo::exists(astap_bin_path());
}

bool AstrometryTab::is_catalog_installed(const QString &catalog_id) const {
    // ASTAP catalogs are data files (e.g. .290, .1476) in the data directory
    QDir dir(astap_data_dir());
    if (!dir.exists()) return false;
    QStringList filters;
    filters << catalog_id + "_*";
    return !dir.entryList(filters, QDir::Files).isEmpty();
}

void AstrometryTab::update_status() {
    if (lbl_astap_data_dir_) {
        lbl_astap_data_dir_->setText(
            "ASTAP data directory: <b>" + astap_data_dir() + "</b>");
    }

    const bool astap_ok = is_astap_installed();
    if (astap_ok) {
        lbl_astap_status_->setText("Installed \u2713");
        lbl_astap_status_->setStyleSheet("font-weight: bold; color: green;");
        btn_install_astap_->setText("Reinstall");
    } else {
        lbl_astap_status_->setText("Not installed");
        lbl_astap_status_->setStyleSheet("font-weight: bold; color: red;");
        btn_install_astap_->setText("Install ASTAP CLI");
    }

    // Check current catalog
    const int idx = cmb_catalog_->currentIndex();
    if (idx >= 0 && idx < NUM_CATALOGS) {
        const QString cat_id = CATALOGS[idx].id;
        if (is_catalog_installed(cat_id)) {
            lbl_catalog_status_->setText(
                QString("%1: Installed \u2713 \u2014 %2")
                    .arg(cat_id.toUpper())
                    .arg(CATALOGS[idx].description));
            lbl_catalog_status_->setStyleSheet("color: green;");
        } else {
            lbl_catalog_status_->setText(
                QString("%1: Not installed \u2014 %2")
                    .arg(cat_id.toUpper())
                    .arg(CATALOGS[idx].description));
            lbl_catalog_status_->setStyleSheet("color: orange;");
        }
    }

    // Enable/disable buttons based on busy state
    const bool downloading = (current_reply_ != nullptr);
    const bool solving = (solve_process_ != nullptr && solve_process_->state() != QProcess::NotRunning);
    const bool busy = downloading || solving;
    btn_install_astap_->setEnabled(!busy);
    btn_download_catalog_->setEnabled(!busy);
    btn_solve_->setEnabled(astap_ok && !busy);
}

void AstrometryTab::append_log(const QString &msg) {
    log_output_->appendPlainText(msg);
    auto *sb = log_output_->verticalScrollBar();
    sb->setValue(sb->maximum());
    emit log_message(msg);
}

void AstrometryTab::start_download(const QUrl &url, const QString &dest_path) {
    if (current_reply_) {
        append_log("[astap] Download already in progress.");
        return;
    }

    QDir().mkpath(QFileInfo(dest_path).absolutePath());

    download_file_ = new QFile(dest_path, this);
    if (!download_file_->open(QIODevice::WriteOnly)) {
        append_log("[astap] ERROR: Cannot open file for writing: " + dest_path);
        delete download_file_;
        download_file_ = nullptr;
        return;
    }

    current_download_dest_ = dest_path;
    progress_->setRange(0, 100);
    progress_->setValue(0);
    progress_->setVisible(true);

    QNetworkRequest request(url);
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::NoLessSafeRedirectPolicy);
    request.setHeader(QNetworkRequest::UserAgentHeader, "TileCompile/1.0");

    current_reply_ = nam_->get(request);
    connect(current_reply_, &QNetworkReply::downloadProgress,
            this, &AstrometryTab::on_download_progress);
    connect(current_reply_, &QNetworkReply::readyRead, this, [this]() {
        if (download_file_ && current_reply_) {
            download_file_->write(current_reply_->readAll());
        }
    });
    connect(current_reply_, &QNetworkReply::finished,
            this, &AstrometryTab::on_download_finished);

    update_status();
}

void AstrometryTab::on_download_progress(qint64 received, qint64 total) {
    if (total > 0) {
        progress_->setRange(0, 100);
        progress_->setValue(static_cast<int>(received * 100 / total));
    } else {
        progress_->setRange(0, 0); // indeterminate
    }
}

void AstrometryTab::on_download_finished() {
    if (!current_reply_) return;

    const bool ok = (current_reply_->error() == QNetworkReply::NoError);

    // Write remaining data
    if (download_file_) {
        download_file_->write(current_reply_->readAll());
        download_file_->close();
    }

    if (!ok) {
        append_log(QString("[astap] Download failed: %1").arg(current_reply_->errorString()));
        if (download_file_) {
            QFile::remove(current_download_dest_);
        }
    } else {
        append_log("[astap] Download complete.");
        extract_and_cleanup(current_download_dest_);
    }

    current_reply_->deleteLater();
    current_reply_ = nullptr;
    if (download_file_) {
        delete download_file_;
        download_file_ = nullptr;
    }

    progress_->setVisible(false);
    update_status();
}

void AstrometryTab::extract_and_cleanup(const QString &archive_path) {
    append_log("[astap] Extracting: " + archive_path);
    const QString dest_dir = astap_data_dir();

    if (archive_path.endsWith(".deb", Qt::CaseInsensitive)) {
        // .deb: extract with dpkg-deb to temp dir, then move .290 files
        const QString tmp_dir = dest_dir + "/deb_tmp";
        QDir().mkpath(tmp_dir);
        QProcess dpkg;
        dpkg.start("dpkg-deb", {"-x", archive_path, tmp_dir});
        dpkg.waitForFinished(120000);
        const QString err = QString::fromUtf8(dpkg.readAllStandardError()).trimmed();
        if (!err.isEmpty()) append_log(err);

        // Find and move .290 files from extracted tree to data dir
        QDirIterator it(tmp_dir, {"*.290"}, QDir::Files, QDirIterator::Subdirectories);
        int moved = 0;
        while (it.hasNext()) {
            it.next();
            const QString dest = dest_dir + "/" + it.fileName();
            QFile::remove(dest); // overwrite if exists
            QFile::rename(it.filePath(), dest);
            ++moved;
        }
        append_log(QString("[astap] Moved %1 catalog files from .deb").arg(moved));

        // Clean up temp dir
        QDir(tmp_dir).removeRecursively();
    } else {
        // .zip: extract directly
        QProcess unzip;
        unzip.setWorkingDirectory(dest_dir);
        unzip.start("unzip", {"-o", archive_path, "-d", dest_dir});
        unzip.waitForFinished(120000);
        const QString out = QString::fromUtf8(unzip.readAllStandardOutput()).trimmed();
        if (!out.isEmpty()) append_log(out);
        const QString err = QString::fromUtf8(unzip.readAllStandardError()).trimmed();
        if (!err.isEmpty()) append_log(err);
    }

    if (download_task_ == DownloadTask::InstallAstap) {
        // Make executable — find the binary
        QProcess chmod;
        chmod.start("chmod", {"+x", astap_bin_path()});
        chmod.waitForFinished(5000);

        if (is_astap_installed()) {
            append_log("[astap] ASTAP CLI installed successfully.");
        } else {
            // Try to find the actual binary name
            QDir dir(dest_dir);
            QStringList files = dir.entryList(QDir::Files | QDir::Executable);
            if (!files.isEmpty()) {
                const QString found = dir.absoluteFilePath(files.first());
                QFile::rename(found, astap_bin_path());
                QProcess chmod2;
                chmod2.start("chmod", {"+x", astap_bin_path()});
                chmod2.waitForFinished(5000);
                append_log("[astap] Renamed " + files.first() + " -> astap_cli");
            } else {
                append_log("[astap] ERROR: No executable found after extraction.");
            }
        }
    } else if (download_task_ == DownloadTask::DownloadCatalog) {
        append_log("[astap] Catalog extraction complete.");
    }

    QFile::remove(archive_path);
    download_task_ = DownloadTask::None;
}

void AstrometryTab::on_install_astap() {
    QDir().mkpath(astap_data_dir());
    const QString zip_path = astap_data_dir() + "/astap_cli.zip";
    append_log("[astap] Downloading ASTAP CLI...");
    download_task_ = DownloadTask::InstallAstap;
    start_download(QUrl(ASTAP_CLI_URL), zip_path);
}

void AstrometryTab::on_download_catalog() {
    const int idx = cmb_catalog_->currentIndex();
    if (idx < 0 || idx >= NUM_CATALOGS) return;

    const QString cat_id = CATALOGS[idx].id;
    const QString filename = CATALOGS[idx].url; // e.g. "d50_star_database.zip"
    const QString dest_path = astap_data_dir() + "/" + filename;
    const QString full_url = QString(SF_BASE) + filename + "/download";

    append_log(QString("[astap] Downloading %1 star database from SourceForge...").arg(cat_id.toUpper()));
    download_task_ = DownloadTask::DownloadCatalog;
    start_download(QUrl(full_url), dest_path);
}

void AstrometryTab::on_browse_fits() {
    const QString path = QFileDialog::getOpenFileName(
        this, "Select FITS file", QString(),
        "FITS files (*.fits *.fit *.fts *.fits.fz *.fit.fz *.fts.fz);;All files (*)");
    if (!path.isEmpty()) {
        edt_fits_path_->setText(path);
    }
}

void AstrometryTab::on_solve() {
    const QString fits_path = edt_fits_path_->text().trimmed();
    if (fits_path.isEmpty()) {
        QMessageBox::warning(this, "Plate Solve", "Please select a FITS file first.");
        return;
    }
    if (!QFileInfo::exists(fits_path)) {
        QMessageBox::warning(this, "Plate Solve",
                             QString("File not found: %1").arg(fits_path));
        return;
    }
    if (!is_astap_installed()) {
        QMessageBox::warning(this, "Plate Solve", "ASTAP CLI is not installed.");
        return;
    }

    // Clear previous results
    solve_ok_ = false;
    last_wcs_path_.clear();
    btn_save_solved_->setEnabled(false);
    lbl_ra_->setText("-");
    lbl_dec_->setText("-");
    lbl_fov_->setText("-");
    lbl_rotation_->setText("-");
    lbl_scale_->setText("-");

    append_log(QString("[astap] Solving: %1").arg(fits_path));

    progress_->setRange(0, 0);
    progress_->setVisible(true);

    solve_process_ = new QProcess(this);
    connect(solve_process_, &QProcess::readyReadStandardOutput, this, [this]() {
        append_log(QString::fromUtf8(solve_process_->readAllStandardOutput()).trimmed());
    });
    connect(solve_process_, &QProcess::readyReadStandardError, this, [this]() {
        const QString err = QString::fromUtf8(solve_process_->readAllStandardError()).trimmed();
        if (!err.isEmpty()) append_log(err);
    });
    connect(solve_process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &AstrometryTab::on_solve_finished);

    QStringList args;
    args << "-f" << fits_path;
    args << "-d" << astap_data_dir();
    args << "-r" << "180";

    solve_process_->start(astap_bin_path(), args);
    update_status();
}

void AstrometryTab::parse_wcs(const QString &fits_path) {
    QString wcs_path = fits_path;
    if (wcs_path.endsWith(".fits.fz", Qt::CaseInsensitive))
        wcs_path.replace(wcs_path.length() - 8, 8, ".wcs");
    else if (wcs_path.endsWith(".fit.fz", Qt::CaseInsensitive))
        wcs_path.replace(wcs_path.length() - 7, 7, ".wcs");
    else if (wcs_path.endsWith(".fts.fz", Qt::CaseInsensitive))
        wcs_path.replace(wcs_path.length() - 7, 7, ".wcs");
    else if (wcs_path.endsWith(".fits", Qt::CaseInsensitive))
        wcs_path.replace(wcs_path.length() - 5, 5, ".wcs");
    else if (wcs_path.endsWith(".fit", Qt::CaseInsensitive))
        wcs_path.replace(wcs_path.length() - 4, 4, ".wcs");
    else if (wcs_path.endsWith(".fts", Qt::CaseInsensitive))
        wcs_path.replace(wcs_path.length() - 4, 4, ".wcs");
    else
        wcs_path += ".wcs";

    if (!QFileInfo::exists(wcs_path)) {
        append_log("[astap] Warning: WCS file not found: " + wcs_path);
        return;
    }

    // Use locale-independent lib parser (std::from_chars)
    auto wcs = tile_compile::astrometry::parse_wcs_file(wcs_path.toStdString());
    if (!wcs.valid()) {
        append_log("[astap] Warning: Could not parse valid WCS from: " + wcs_path);
        return;
    }

    lbl_ra_->setText(QString::number(wcs.crval1, 'f', 6) + " deg");
    lbl_dec_->setText(QString::number(wcs.crval2, 'f', 6) + " deg");
    lbl_scale_->setText(QString::number(wcs.pixel_scale_arcsec(), 'f', 2) + " arcsec/px");
    lbl_rotation_->setText(QString::number(wcs.rotation_deg(), 'f', 2) + " deg");

    double fov_w = wcs.fov_width_deg();
    double fov_h = wcs.fov_height_deg();
    if (fov_w > 0 && fov_h > 0) {
        lbl_fov_->setText(QString("%1 x %2 deg")
                              .arg(fov_w, 0, 'f', 2)
                              .arg(fov_h, 0, 'f', 2));
    }

    append_log("[astap] WCS solution parsed from: " + wcs_path);
}

void AstrometryTab::on_solve_finished(int exit_code, QProcess::ExitStatus status) {
    progress_->setVisible(false);

    if (exit_code == 0 && status == QProcess::NormalExit) {
        append_log("[astap] Plate solve successful!");
        const QString fits_path = edt_fits_path_->text().trimmed();
        parse_wcs(fits_path);

        // Determine .wcs path
        QString wcs_path = fits_path;
        if (wcs_path.endsWith(".fits.fz", Qt::CaseInsensitive))
            wcs_path.replace(wcs_path.length() - 8, 8, ".wcs");
        else if (wcs_path.endsWith(".fit.fz", Qt::CaseInsensitive))
            wcs_path.replace(wcs_path.length() - 7, 7, ".wcs");
        else if (wcs_path.endsWith(".fts.fz", Qt::CaseInsensitive))
            wcs_path.replace(wcs_path.length() - 7, 7, ".wcs");
        else if (wcs_path.endsWith(".fits", Qt::CaseInsensitive))
            wcs_path.replace(wcs_path.length() - 5, 5, ".wcs");
        else if (wcs_path.endsWith(".fit", Qt::CaseInsensitive))
            wcs_path.replace(wcs_path.length() - 4, 4, ".wcs");
        else if (wcs_path.endsWith(".fts", Qt::CaseInsensitive))
            wcs_path.replace(wcs_path.length() - 4, 4, ".wcs");
        else
            wcs_path += ".wcs";

        if (QFileInfo::exists(wcs_path)) {
            solve_ok_ = true;
            last_wcs_path_ = wcs_path;
            btn_save_solved_->setEnabled(true);
        }
    } else {
        append_log(QString("[astap] Plate solve failed (exit code %1). "
                           "Check that a star database is installed.")
                       .arg(exit_code));
    }

    solve_process_->deleteLater();
    solve_process_ = nullptr;
    update_status();
}

void AstrometryTab::on_save_solved() {
    if (!solve_ok_ || last_wcs_path_.isEmpty()) {
        QMessageBox::warning(this, "Save Solved", "No plate solve result available.");
        return;
    }
    const QString fits_path = edt_fits_path_->text().trimmed();
    if (!QFileInfo::exists(fits_path)) {
        QMessageBox::warning(this, "Save Solved",
                             QString("Source FITS not found: %1").arg(fits_path));
        return;
    }

    // Determine output path
    QFileInfo fi(fits_path);
    QString default_name = fi.completeBaseName() + "_solved." + fi.suffix();
    QString save_path = QFileDialog::getSaveFileName(
        this, "Save Solved FITS", fi.dir().filePath(default_name),
        "FITS files (*.fits *.fit *.fts *.fits.fz *.fit.fz *.fts.fz);;All files (*)");
    if (save_path.isEmpty()) return;

    try {
        // Read original FITS as RGB cube (preserves color)
        auto rgb = tile_compile::io::read_fits_rgb(fits_path.toStdString());
        auto &hdr = rgb.header;

        // Inject WCS keywords from the lib parser (locale-independent)
        auto wcs = tile_compile::astrometry::parse_wcs_file(
            last_wcs_path_.toStdString());
        if (wcs.valid()) {
            hdr.numeric_values["CRVAL1"] = wcs.crval1;
            hdr.numeric_values["CRVAL2"] = wcs.crval2;
            hdr.numeric_values["CRPIX1"] = wcs.crpix1;
            hdr.numeric_values["CRPIX2"] = wcs.crpix2;
            hdr.numeric_values["CD1_1"]  = wcs.cd1_1;
            hdr.numeric_values["CD1_2"]  = wcs.cd1_2;
            hdr.numeric_values["CD2_1"]  = wcs.cd2_1;
            hdr.numeric_values["CD2_2"]  = wcs.cd2_2;
            hdr.string_values["CTYPE1"]  = "RA---TAN";
            hdr.string_values["CTYPE2"]  = "DEC--TAN";
            hdr.string_values["CUNIT1"]  = "deg";
            hdr.string_values["CUNIT2"]  = "deg";
            hdr.numeric_values["EQUINOX"] = 2000.0;
            hdr.bool_values["PLTSOLVD"] = true;
        }

        // Write solved FITS as RGB cube
        tile_compile::io::write_fits_rgb(save_path.toStdString(),
                                         rgb.R, rgb.G, rgb.B, hdr);

        // Also copy .wcs file alongside the saved FITS
        QFileInfo save_fi(save_path);
        QString wcs_dest = save_fi.dir().filePath(save_fi.completeBaseName() + ".wcs");
        QFile::copy(last_wcs_path_, wcs_dest);

        append_log(QString("[astap] Solved FITS saved: %1").arg(save_path));
        append_log(QString("[astap] WCS file copied: %1").arg(wcs_dest));
        QMessageBox::information(this, "Save Solved",
                                 QString("Saved:\n%1\n%2").arg(save_path, wcs_dest));
    } catch (const std::exception &e) {
        QMessageBox::critical(this, "Save Solved",
                              QString("Error saving: %1").arg(e.what()));
        append_log(QString("[astap] Save error: %1").arg(e.what()));
    }
}

} // namespace tile_compile::gui
