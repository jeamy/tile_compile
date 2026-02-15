#include "tabs/PCCTab.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/astrometry/wcs.hpp"
#include "tile_compile/astrometry/gaia_catalog.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"

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
#include <QProcess>

namespace tile_compile::gui {

namespace astro = tile_compile::astrometry;
namespace io = tile_compile::io;

PCCTab::PCCTab(const std::string &project_root, QWidget *parent)
    : QWidget(parent), project_root_(project_root) {
    nam_ = new QNetworkAccessManager(this);
    build_ui();
    update_catalog_status();
}

void PCCTab::build_ui() {
    auto *root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(10);

    // === Input Section ===
    auto *input_box = new QGroupBox("Input");
    auto *input_form = new QFormLayout(input_box);
    input_form->setSpacing(6);
    input_form->setContentsMargins(8, 14, 8, 8);

    auto *fits_row = new QHBoxLayout();
    edt_fits_path_ = new QLineEdit();
    edt_fits_path_->setPlaceholderText("Path to stacked RGB FITS (e.g. stacked_rgb.fits)");
    fits_row->addWidget(edt_fits_path_, 1);
    auto *btn_browse_fits = new QPushButton("...");
    btn_browse_fits->setFixedWidth(30);
    fits_row->addWidget(btn_browse_fits);
    input_form->addRow("RGB FITS:", fits_row);

    auto *wcs_row = new QHBoxLayout();
    edt_wcs_path_ = new QLineEdit();
    edt_wcs_path_->setPlaceholderText("Path to .wcs file (from plate solve)");
    wcs_row->addWidget(edt_wcs_path_, 1);
    auto *btn_browse_wcs = new QPushButton("...");
    btn_browse_wcs->setFixedWidth(30);
    wcs_row->addWidget(btn_browse_wcs);
    input_form->addRow("WCS file:", wcs_row);

    cmb_source_ = new QComboBox();
    cmb_source_->addItems({"siril", "vizier_gaia", "vizier_apass"});
    input_form->addRow("Catalog source:", cmb_source_);

    root->addWidget(input_box);

    // === Siril Gaia Catalog Section ===
    auto *cat_box = new QGroupBox("Siril Gaia DR3 Catalog");
    auto *cat_layout = new QVBoxLayout(cat_box);
    cat_layout->setSpacing(6);

    lbl_catalog_status_ = new QLabel("Checking...");
    lbl_catalog_status_->setWordWrap(true);
    cat_layout->addWidget(lbl_catalog_status_);

    auto *cat_btn_row = new QHBoxLayout();
    btn_download_catalog_ = new QPushButton("Download Missing Chunks");
    cat_btn_row->addWidget(btn_download_catalog_);
    btn_cancel_download_ = new QPushButton("Cancel");
    btn_cancel_download_->setEnabled(false);
    cat_btn_row->addWidget(btn_cancel_download_);
    cat_btn_row->addStretch(1);
    cat_layout->addLayout(cat_btn_row);

    progress_download_ = new QProgressBar();
    progress_download_->setVisible(false);
    cat_layout->addWidget(progress_download_);

    auto *cat_info = new QLabel(
        "<small>The Siril Gaia DR3 XP-Sampled catalog (~21 GB total, 48 chunks) "
        "provides the best PCC results. For quick tests, use VizieR online sources instead.</small>");
    cat_info->setWordWrap(true);
    cat_layout->addWidget(cat_info);

    root->addWidget(cat_box);

    connect(btn_download_catalog_, &QPushButton::clicked, this, &PCCTab::on_download_catalog);
    connect(btn_cancel_download_, &QPushButton::clicked, this, &PCCTab::on_cancel_download);
    connect(cmb_source_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { update_catalog_status(); });

    // === Parameters Section ===
    auto *param_box = new QGroupBox("Parameters");
    auto *param_form = new QFormLayout(param_box);
    param_form->setSpacing(6);
    param_form->setContentsMargins(8, 14, 8, 8);

    auto make_dspin = [](double min, double max, double val, int dec, const QString &suffix) {
        auto *spn = new QDoubleSpinBox();
        spn->setRange(min, max);
        spn->setValue(val);
        spn->setDecimals(dec);
        if (!suffix.isEmpty()) spn->setSuffix(suffix);
        return spn;
    };

    spn_mag_limit_ = make_dspin(1.0, 22.0, 14.0, 1, " mag");
    param_form->addRow("Mag limit (faint):", spn_mag_limit_);

    spn_mag_bright_ = make_dspin(0.0, 15.0, 6.0, 1, " mag");
    param_form->addRow("Mag limit (bright):", spn_mag_bright_);

    spn_aperture_ = make_dspin(1.0, 50.0, 8.0, 1, " px");
    param_form->addRow("Aperture radius:", spn_aperture_);

    spn_ann_inner_ = make_dspin(1.0, 80.0, 12.0, 1, " px");
    param_form->addRow("Annulus inner:", spn_ann_inner_);

    spn_ann_outer_ = make_dspin(2.0, 100.0, 18.0, 1, " px");
    param_form->addRow("Annulus outer:", spn_ann_outer_);

    spn_min_stars_ = new QSpinBox();
    spn_min_stars_->setRange(3, 500);
    spn_min_stars_->setValue(10);
    param_form->addRow("Min stars:", spn_min_stars_);

    spn_sigma_ = make_dspin(1.0, 10.0, 2.5, 1, " \u03c3");
    param_form->addRow("Sigma clip:", spn_sigma_);

    root->addWidget(param_box);

    // === Results Section ===
    auto *results_box = new QGroupBox("Results");
    auto *results_form = new QFormLayout(results_box);
    results_form->setSpacing(4);
    results_form->setContentsMargins(8, 14, 8, 8);

    lbl_stars_matched_ = new QLabel("-");
    lbl_stars_used_ = new QLabel("-");
    lbl_residual_ = new QLabel("-");
    lbl_matrix_ = new QLabel("-");
    lbl_matrix_->setWordWrap(true);
    results_form->addRow("Stars matched:", lbl_stars_matched_);
    results_form->addRow("Stars used:", lbl_stars_used_);
    results_form->addRow("Residual RMS:", lbl_residual_);
    results_form->addRow("Color matrix:", lbl_matrix_);

    root->addWidget(results_box);

    // === Controls ===
    auto *ctrl_row = new QHBoxLayout();
    btn_run_ = new QPushButton("Run PCC");
    btn_run_->setStyleSheet("font-weight: bold;");
    ctrl_row->addWidget(btn_run_);
    btn_save_ = new QPushButton("Save Corrected");
    btn_save_->setEnabled(false);
    ctrl_row->addWidget(btn_save_);
    ctrl_row->addStretch(1);
    root->addLayout(ctrl_row);

    progress_ = new QProgressBar();
    progress_->setRange(0, 0);
    progress_->setVisible(false);
    root->addWidget(progress_);

    log_output_ = new QPlainTextEdit();
    log_output_->setReadOnly(true);
    log_output_->setMaximumHeight(150);
    log_output_->setPlaceholderText("PCC log output...");
    root->addWidget(log_output_, 1);

    // Connections
    connect(btn_browse_fits, &QPushButton::clicked, this, &PCCTab::on_browse_fits);
    connect(btn_browse_wcs, &QPushButton::clicked, this, &PCCTab::on_browse_wcs);
    connect(btn_run_, &QPushButton::clicked, this, &PCCTab::on_run_pcc);
    connect(btn_save_, &QPushButton::clicked, this, [this]() {
        if (!pcc_ok_) return;
        const QString fits_path = edt_fits_path_->text().trimmed();
        QFileInfo fi(fits_path);
        QString default_name = fi.completeBaseName() + "_pcc." + fi.suffix();
        QString save_path = QFileDialog::getSaveFileName(
            this, "Save PCC-Corrected RGB FITS", fi.dir().filePath(default_name),
            "FITS files (*.fits *.fit *.fts *.fits.fz *.fit.fz *.fts.fz);;All files (*)");
        if (save_path.isEmpty()) return;

        try {
            io::write_fits_rgb(save_path.toStdString(), pcc_R_, pcc_G_, pcc_B_, pcc_hdr_);
            append_log("[PCC] Saved corrected RGB: " + save_path);

            // Also save individual channels alongside
            QFileInfo sfi(save_path);
            QString base = sfi.dir().filePath(sfi.completeBaseName());
            io::write_fits_float((base + "_R.fit").toStdString(), pcc_R_, pcc_hdr_);
            io::write_fits_float((base + "_G.fit").toStdString(), pcc_G_, pcc_hdr_);
            io::write_fits_float((base + "_B.fit").toStdString(), pcc_B_, pcc_hdr_);
            append_log("[PCC] Saved channels: " + base + "_R/G/B.fit");
        } catch (const std::exception &e) {
            QMessageBox::critical(this, "Save Error", e.what());
            append_log(QString("[PCC] Save error: %1").arg(e.what()));
        }
    });
}

void PCCTab::append_log(const QString &msg) {
    log_output_->appendPlainText(msg);
    log_output_->verticalScrollBar()->setValue(log_output_->verticalScrollBar()->maximum());
    emit log_message(msg);
}

void PCCTab::on_browse_fits() {
    const QString path = QFileDialog::getOpenFileName(
        this, "Select RGB FITS", QString(),
        "FITS files (*.fits *.fit *.fts *.fits.fz *.fit.fz *.fts.fz);;All files (*)");
    if (!path.isEmpty()) {
        edt_fits_path_->setText(path);
        // Auto-detect .wcs file: try exact name, then _solved variant
        QFileInfo fi(path);
        QString base = fi.completeBaseName();
        QString dir = fi.dir().path();
        QString suffix = fi.suffix();

        QStringList candidates = {
            dir + "/" + base + ".wcs",
            dir + "/" + base + "_solved.wcs",
        };
        // If name ends with _solved, also try without
        if (base.endsWith("_solved")) {
            candidates.append(dir + "/" + base.left(base.length() - 7) + ".wcs");
        }

        bool found = false;
        for (const QString &wcs_path : candidates) {
            if (QFileInfo::exists(wcs_path)) {
                edt_wcs_path_->setText(wcs_path);
                append_log("[PCC] Auto-detected WCS: " + wcs_path);
                found = true;
                break;
            }
        }
        if (!found) {
            append_log("[PCC] No .wcs file found. Please select one manually or run Plate Solve first.");
        }

        // Check if file is mono (warn user)
        try {
            auto [w, h, naxis] = io::get_fits_dimensions(path.toStdString());
            if (naxis < 3) {
                append_log("[PCC] WARNING: Selected FITS appears to be mono (NAXIS=" +
                           QString::number(naxis) + "). PCC needs an RGB color image!");
                append_log("[PCC] If this is a _solved file, select the original RGB FITS instead.");
            }
        } catch (...) {}
    }
}

void PCCTab::on_browse_wcs() {
    const QString path = QFileDialog::getOpenFileName(
        this, "Select WCS file", QString(),
        "WCS files (*.wcs);;All files (*)");
    if (!path.isEmpty()) edt_wcs_path_->setText(path);
}

void PCCTab::on_run_pcc() {
    const QString fits_path = edt_fits_path_->text().trimmed();
    const QString wcs_path = edt_wcs_path_->text().trimmed();

    if (fits_path.isEmpty() || wcs_path.isEmpty()) {
        QMessageBox::warning(this, "PCC", "Please select both a FITS file and a WCS file.");
        return;
    }
    if (!QFileInfo::exists(fits_path)) {
        QMessageBox::warning(this, "PCC", "FITS file not found: " + fits_path);
        return;
    }
    if (!QFileInfo::exists(wcs_path)) {
        QMessageBox::warning(this, "PCC", "WCS file not found: " + wcs_path);
        return;
    }

    pcc_ok_ = false;
    btn_save_->setEnabled(false);
    lbl_stars_matched_->setText("-");
    lbl_stars_used_->setText("-");
    lbl_residual_->setText("-");
    lbl_matrix_->setText("-");

    progress_->setVisible(true);
    append_log("[PCC] Starting photometric color calibration...");

    try {
        // Parse WCS
        astro::WCS wcs = astro::parse_wcs_file(wcs_path.toStdString());
        if (!wcs.valid()) {
            append_log("[PCC] Error: Invalid WCS solution.");
            progress_->setVisible(false);
            return;
        }
        append_log(QString("[PCC] WCS: RA=%1 Dec=%2 scale=%3 arcsec/px naxis=%4x%5")
                       .arg(wcs.crval1, 0, 'f', 6)
                       .arg(wcs.crval2, 0, 'f', 6)
                       .arg(wcs.pixel_scale_arcsec(), 0, 'f', 3)
                       .arg(wcs.naxis1).arg(wcs.naxis2));
        append_log(QString("[PCC] CD: [%1, %2] [%3, %4] CRPIX: %5, %6")
                       .arg(wcs.cd1_1, 0, 'e', 4)
                       .arg(wcs.cd1_2, 0, 'e', 4)
                       .arg(wcs.cd2_1, 0, 'e', 4)
                       .arg(wcs.cd2_2, 0, 'e', 4)
                       .arg(wcs.crpix1, 0, 'f', 1)
                       .arg(wcs.crpix2, 0, 'f', 1));

        // Determine catalog source
        double search_r = wcs.search_radius_deg();
        double mag_lim = spn_mag_limit_->value();
        QString source = cmb_source_->currentText();

        append_log(QString("[PCC] Querying %1: RA=%2 Dec=%3 r=%4 deg mag<%5")
                       .arg(source)
                       .arg(wcs.crval1, 0, 'f', 2)
                       .arg(wcs.crval2, 0, 'f', 2)
                       .arg(search_r, 0, 'f', 2)
                       .arg(mag_lim, 0, 'f', 1));

        std::vector<astro::GaiaStar> stars;

        if (source == "siril") {
            std::string cat_dir = siril_catalog_dir().toStdString();
            if (!astro::is_siril_gaia_catalog_available(cat_dir)) {
                append_log("[PCC] Error: Siril Gaia catalog not found at: " +
                           QString::fromStdString(cat_dir));
                append_log("[PCC] Download it first or switch to a VizieR source.");
                progress_->setVisible(false);
                return;
            }
            stars = astro::siril_gaia_cone_search(cat_dir, wcs.crval1, wcs.crval2, search_r, mag_lim);
        } else if (source == "vizier_gaia") {
            stars = astro::vizier_gaia_cone_search(wcs.crval1, wcs.crval2, search_r, mag_lim);
        } else if (source == "vizier_apass") {
            stars = astro::vizier_apass_cone_search(wcs.crval1, wcs.crval2, search_r, mag_lim);
        }

        append_log(QString("[PCC] Found %1 catalog stars").arg(stars.size()));

        if (stars.empty()) {
            append_log("[PCC] No catalog stars found. Try a different source.");
            progress_->setVisible(false);
            return;
        }

        // Read RGB FITS — supports 3-plane cubes (NAXIS3=3) and separate R/G/B files
        QFileInfo fi(fits_path);
        QString base_dir = fi.dir().path();
        QString r_path = base_dir + "/reconstructed_R.fit";
        QString g_path = base_dir + "/reconstructed_G.fit";
        QString b_path = base_dir + "/reconstructed_B.fit";

        tile_compile::Matrix2Df R, G, B;
        io::FitsHeader hdr;

        if (QFileInfo::exists(r_path) && QFileInfo::exists(g_path) && QFileInfo::exists(b_path)) {
            auto [rd, rh] = io::read_fits_float(r_path.toStdString());
            auto [gd, gh] = io::read_fits_float(g_path.toStdString());
            auto [bd, bh] = io::read_fits_float(b_path.toStdString());
            R = rd; G = gd; B = bd; hdr = rh;
            append_log("[PCC] Using separate R/G/B channel files.");
        } else {
            auto rgb = io::read_fits_rgb(fits_path.toStdString());
            R = rgb.R; G = rgb.G; B = rgb.B; hdr = rgb.header;
            append_log(QString("[PCC] Read RGB FITS: %1x%2").arg(rgb.width).arg(rgb.height));
        }

        // Build PCC config
        astro::PCCConfig pcc_cfg;
        pcc_cfg.aperture_radius_px = spn_aperture_->value();
        pcc_cfg.annulus_inner_px = spn_ann_inner_->value();
        pcc_cfg.annulus_outer_px = spn_ann_outer_->value();
        pcc_cfg.mag_limit = spn_mag_limit_->value();
        pcc_cfg.mag_bright_limit = spn_mag_bright_->value();
        pcc_cfg.min_stars = spn_min_stars_->value();
        pcc_cfg.sigma_clip = spn_sigma_->value();

        auto result = astro::run_pcc(R, G, B, wcs, stars, pcc_cfg);

        if (result.success) {
            pcc_ok_ = true;
            btn_save_->setEnabled(true);

            lbl_stars_matched_->setText(QString::number(result.n_stars_matched));
            lbl_stars_used_->setText(QString::number(result.n_stars_used));
            lbl_residual_->setText(QString::number(result.residual_rms, 'f', 6));

            QString mat_str;
            for (int r = 0; r < 3; ++r) {
                mat_str += QString("[%1, %2, %3]")
                               .arg(result.matrix[r][0], 0, 'f', 4)
                               .arg(result.matrix[r][1], 0, 'f', 4)
                               .arg(result.matrix[r][2], 0, 'f', 4);
                if (r < 2) mat_str += "\n";
            }
            lbl_matrix_->setText(mat_str);

            // Store corrected data for Save button
            pcc_R_ = R;
            pcc_G_ = G;
            pcc_B_ = B;
            pcc_hdr_ = hdr;

            append_log("[PCC] Success! Use 'Save Corrected' to write the result.");
        } else {
            append_log("[PCC] Fit failed: " + QString::fromStdString(result.error_message));
            append_log(QString("[PCC] Stars matched: %1").arg(result.n_stars_matched));
        }
    } catch (const std::exception &e) {
        append_log(QString("[PCC] Error: %1").arg(e.what()));
    }

    progress_->setVisible(false);
}

QString PCCTab::siril_catalog_dir() const {
    return QDir::homePath() + "/.local/share/siril/siril_cat1_healpix8_xpsamp";
}

void PCCTab::update_catalog_status() {
    const QString dir = siril_catalog_dir();
    auto missing = astro::missing_siril_catalog_chunks(dir.toStdString());
    int installed = 48 - static_cast<int>(missing.size());

    bool is_siril = (cmb_source_->currentText() == "siril");
    bool downloading = (current_reply_ != nullptr);

    if (installed == 48) {
        lbl_catalog_status_->setText(
            QString("Installed \u2713 \u2014 48/48 chunks in %1").arg(dir));
        lbl_catalog_status_->setStyleSheet("color: green; font-weight: bold;");
        btn_download_catalog_->setEnabled(false);
    } else if (installed > 0) {
        lbl_catalog_status_->setText(
            QString("Partial \u2014 %1/48 chunks installed in %2 (%3 missing)")
                .arg(installed).arg(dir).arg(missing.size()));
        lbl_catalog_status_->setStyleSheet("color: orange; font-weight: bold;");
        btn_download_catalog_->setEnabled(!downloading);
    } else {
        lbl_catalog_status_->setText(
            QString("Not installed \u2014 %1").arg(dir));
        lbl_catalog_status_->setStyleSheet("color: red; font-weight: bold;");
        btn_download_catalog_->setEnabled(!downloading);
    }

    // Disable Run PCC if siril selected but catalog missing
    if (is_siril && installed == 0) {
        btn_run_->setToolTip("Download the Siril catalog first, or switch to a VizieR source.");
    } else {
        btn_run_->setToolTip("");
    }
}

void PCCTab::on_download_catalog() {
    const QString dir = siril_catalog_dir();
    auto missing = astro::missing_siril_catalog_chunks(dir.toStdString());
    if (missing.empty()) {
        append_log("[PCC] All 48 catalog chunks already installed.");
        return;
    }

    chunks_to_download_ = missing;
    current_chunk_idx_ = 0;
    download_cancelled_ = false;

    btn_download_catalog_->setEnabled(false);
    btn_cancel_download_->setEnabled(true);
    progress_download_->setRange(0, static_cast<int>(missing.size()));
    progress_download_->setValue(0);
    progress_download_->setVisible(true);

    append_log(QString("[PCC] Downloading %1 missing catalog chunks to %2 ...")
                   .arg(missing.size()).arg(dir));

    download_next_chunk();
}

void PCCTab::on_cancel_download() {
    download_cancelled_ = true;
    if (current_reply_) {
        current_reply_->abort();
    }
    btn_cancel_download_->setEnabled(false);
    append_log("[PCC] Download cancelled.");
}

void PCCTab::download_next_chunk() {
    if (download_cancelled_ || current_chunk_idx_ >= static_cast<int>(chunks_to_download_.size())) {
        // Done or cancelled
        progress_download_->setVisible(false);
        btn_download_catalog_->setEnabled(true);
        btn_cancel_download_->setEnabled(false);
        update_catalog_status();
        if (!download_cancelled_) {
            append_log("[PCC] All chunks downloaded successfully.");
        }
        return;
    }

    int chunk_id = chunks_to_download_[current_chunk_idx_];
    std::string url = astro::siril_catalog_chunk_url(chunk_id);
    QString dest_dir = siril_catalog_dir();
    QDir().mkpath(dest_dir);

    char bz2_fname[128];
    std::snprintf(bz2_fname, sizeof(bz2_fname),
                  "siril_cat1_healpix8_xpsamp_%d.dat.bz2", chunk_id);
    QString bz2_path = dest_dir + "/" + bz2_fname;

    append_log(QString("[PCC] Chunk %1/%2: downloading chunk_%3 ...")
                   .arg(current_chunk_idx_ + 1)
                   .arg(chunks_to_download_.size())
                   .arg(chunk_id));

    download_file_ = new QFile(bz2_path, this);
    if (!download_file_->open(QIODevice::WriteOnly)) {
        append_log("[PCC] ERROR: Cannot write to " + bz2_path);
        delete download_file_;
        download_file_ = nullptr;
        return;
    }

    QNetworkRequest req(QUrl(QString::fromStdString(url)));
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                     QNetworkRequest::NoLessSafeRedirectPolicy);
    req.setHeader(QNetworkRequest::UserAgentHeader, "TileCompile/1.0");

    current_reply_ = nam_->get(req);
    connect(current_reply_, &QNetworkReply::downloadProgress,
            this, &PCCTab::on_chunk_download_progress);
    connect(current_reply_, &QNetworkReply::readyRead, this, [this]() {
        if (download_file_ && current_reply_)
            download_file_->write(current_reply_->readAll());
    });
    connect(current_reply_, &QNetworkReply::finished,
            this, &PCCTab::on_chunk_download_finished);
}

void PCCTab::on_chunk_download_progress(qint64 received, qint64 total) {
    // Update progress: base = chunks done, fraction = current chunk progress
    int base = current_chunk_idx_;
    double frac = (total > 0) ? static_cast<double>(received) / total : 0;
    int total_chunks = static_cast<int>(chunks_to_download_.size());
    progress_download_->setRange(0, total_chunks * 100);
    progress_download_->setValue(static_cast<int>((base + frac) * 100));
}

void PCCTab::on_chunk_download_finished() {
    if (!current_reply_) return;

    bool ok = (current_reply_->error() == QNetworkReply::NoError);

    if (download_file_) {
        download_file_->write(current_reply_->readAll());
        download_file_->close();
    }

    QString bz2_path;
    if (download_file_) {
        bz2_path = download_file_->fileName();
        delete download_file_;
        download_file_ = nullptr;
    }

    current_reply_->deleteLater();
    current_reply_ = nullptr;

    if (!ok || download_cancelled_) {
        if (!download_cancelled_)
            append_log("[PCC] Download failed for chunk.");
        if (!bz2_path.isEmpty())
            QFile::remove(bz2_path);
        progress_download_->setVisible(false);
        btn_download_catalog_->setEnabled(true);
        btn_cancel_download_->setEnabled(false);
        update_catalog_status();
        return;
    }

    // Decompress .bz2 → .dat
    append_log("[PCC] Decompressing " + QFileInfo(bz2_path).fileName() + " ...");
    QProcess bzip2;
    bzip2.start("bzip2", {"-dk", bz2_path});
    bzip2.waitForFinished(300000);  // 5 min max per chunk

    if (bzip2.exitCode() != 0) {
        append_log("[PCC] bzip2 decompression failed: " +
                   QString::fromUtf8(bzip2.readAllStandardError()).trimmed());
        QFile::remove(bz2_path);
    } else {
        QFile::remove(bz2_path);  // remove .bz2 after successful decompress
    }

    ++current_chunk_idx_;
    download_next_chunk();
}

} // namespace tile_compile::gui
