#include "PhaseProgressWidget.hpp"

#include <QGroupBox>
#include <QStyle>
#include <Qt>

namespace tile_compile::gui {

namespace {
struct PhaseInfo {
    int id;
    const char *name;
    const char *desc;
};

constexpr PhaseInfo METHODIK_V4_PHASES[] = {
    {0, "SCAN_INPUT", "Eingabe-Validierung"},
    {1, "CHANNEL_SPLIT", "Kanal-Trennung (R/G/B)"},
    {2, "NORMALIZATION", "Globale lineare Normalisierung"},
    {3, "GLOBAL_METRICS", "Globale Frame-Metriken (B, σ, E)"},
    {4, "GLOBAL_REGISTRATION", "Globale Registrierung (dx/dy)"},
    {5, "TILE_GRID", "Seeing-adaptive Tile-Geometrie"},
    {6, "LOCAL_METRICS", "Lokale Tile-Metriken"},
    {7, "TILE_RECONSTRUCTION_TLR", "Tile-lokale Registrierung + Rekonstruktion"},
    {8, "STATE_CLUSTERING", "Zustandsbasiertes Clustering"},
    {9, "SYNTHETIC_FRAMES", "Synthetische Qualitätsframes"},
    {10, "STACKING", "Finales lineares Stacking"},
    {11, "DEBAYER", "Debayer / Demosaicing"},
    {12, "DONE", "Abschluss"},
};

constexpr int NUM_PHASES = sizeof(METHODIK_V4_PHASES) / sizeof(METHODIK_V4_PHASES[0]);
}

PhaseProgressWidget::PhaseProgressWidget(QWidget *parent) : QWidget(parent) {
    build_ui();
}

void PhaseProgressWidget::build_ui() {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);

    reduced_mode_label_ = new QLabel("");
    reduced_mode_label_->setObjectName("ReducedModeWarning");
    reduced_mode_label_->setVisible(false);
    layout->addWidget(reduced_mode_label_);

    progress_bar_ = new QProgressBar();
    progress_bar_->setMinimum(0);
    progress_bar_->setMaximum(NUM_PHASES);
    progress_bar_->setValue(0);
    layout->addWidget(progress_bar_);

    auto *grid = new QGridLayout();
    grid->setSpacing(6);

    for (int i = 0; i < NUM_PHASES; ++i) {
        const auto &phase = METHODIK_V4_PHASES[i];

        auto *name_label = new QLabel(QString("%1. %2").arg(phase.id).arg(phase.name));
        name_label->setToolTip(phase.desc);

        auto *status_label = new QLabel("pending");
        status_label->setObjectName("PhasePending");
        status_label->setMinimumWidth(80);
        status_label->setAlignment(Qt::AlignCenter);

        auto *progress_bar = new QProgressBar();
        progress_bar->setMinimum(0);
        progress_bar->setMaximum(100);
        progress_bar->setValue(0);
        progress_bar->setTextVisible(true);
        progress_bar->setFormat("%p%");
        progress_bar->setMaximumHeight(18);
        progress_bar->setVisible(false);

        grid->addWidget(name_label, i, 0);
        grid->addWidget(status_label, i, 1);
        grid->addWidget(progress_bar, i, 2);

        phase_labels_[phase.id] = name_label;
        phase_status_labels_[phase.id] = status_label;
        phase_progress_bars_[phase.id] = progress_bar;
    }

    layout->addLayout(grid);

    error_label_ = new QLabel("");
    error_label_->setObjectName("PhaseErrorDetail");
    error_label_->setWordWrap(true);
    error_label_->setVisible(false);
    layout->addWidget(error_label_);
    layout->addStretch(1);
}

void PhaseProgressWidget::reset() {
    for (auto &[id, label] : phase_status_labels_) {
        label->setText("pending");
        label->setObjectName("PhasePending");
        label->style()->unpolish(label);
        label->style()->polish(label);
    }
    for (auto &[id, bar] : phase_progress_bars_) {
        bar->setValue(0);
        bar->setVisible(false);
    }
    progress_bar_->setValue(0);
    reduced_mode_label_->setVisible(false);
    last_error_text_.clear();
    if (error_label_) {
        error_label_->setText("");
        error_label_->setVisible(false);
    }
}

void PhaseProgressWidget::set_reduced_mode(bool enabled, int frame_count) {
    reduced_mode_ = enabled;
    if (enabled) {
        reduced_mode_label_->setText(
            QString("⚠ Reduced Mode aktiv (%1 Frames < 200)\n"
                    "STATE_CLUSTERING und SYNTHETIC_FRAMES werden übersprungen.")
                .arg(frame_count));
        reduced_mode_label_->setVisible(true);
    } else {
        reduced_mode_label_->setVisible(false);
    }
}

void PhaseProgressWidget::update_phase(const std::string &phase_name, const std::string &status,
                                       int progress_current, int progress_total,
                                       const std::string &substep, const std::string &pass_info) {
    int phase_id = -1;
    for (int i = 0; i < NUM_PHASES; ++i) {
        if (METHODIK_V4_PHASES[i].name == phase_name) {
            phase_id = METHODIK_V4_PHASES[i].id;
            break;
        }
    }

    if (phase_id < 0 || phase_status_labels_.find(phase_id) == phase_status_labels_.end()) {
        return;
    }

    auto *label = phase_status_labels_[phase_id];
    auto *progress_bar = phase_progress_bars_[phase_id];

    const QString current_text = label->text();
    if ((current_text == "ok" || current_text == "error" || current_text == "skipped") &&
        status == "running") {
        return;
    }

    if (status == "running") {
        label->setText("running");
        label->setObjectName("PhaseRunning");

        if (progress_total > 0 && progress_current >= 0) {
            const int percent = static_cast<int>(100 * progress_current / progress_total);
            progress_bar->setValue(percent);

            QString prefix;
            if (!pass_info.empty() && !substep.empty()) {
                prefix = QString::fromStdString(pass_info) + ": " + QString::fromStdString(substep);
            } else if (!pass_info.empty()) {
                prefix = QString::fromStdString(pass_info);
            } else if (!substep.empty()) {
                prefix = QString::fromStdString(substep);
            }

            if (!prefix.isEmpty()) {
                progress_bar->setFormat(QString("%1: %2/%3 (%4%)").arg(prefix).arg(progress_current).arg(progress_total).arg(percent));
            } else {
                progress_bar->setFormat(QString("%1/%2 (%3%)").arg(progress_current).arg(progress_total).arg(percent));
            }
            progress_bar->setVisible(true);
        } else {
            progress_bar->setValue(0);
            QString fmt = "running...";
            if (!pass_info.empty() && !substep.empty()) {
                fmt = QString::fromStdString(pass_info) + ": " + QString::fromStdString(substep) + "...";
            } else if (!pass_info.empty()) {
                fmt = QString::fromStdString(pass_info) + "...";
            } else if (!substep.empty()) {
                fmt = QString::fromStdString(substep) + "...";
            }
            progress_bar->setFormat(fmt);
            progress_bar->setVisible(true);
        }
    } else if (status == "ok" || status == "success") {
        label->setText("ok");
        label->setObjectName("PhaseOk");
        progress_bar->setVisible(false);
    } else if (status == "error") {
        label->setText("error");
        label->setObjectName("PhaseError");
        progress_bar->setVisible(false);
    } else if (status == "skipped") {
        label->setText("skipped");
        label->setObjectName("PhaseSkipped");
        progress_bar->setVisible(false);
    } else {
        label->setText(QString::fromStdString(status));
        label->setObjectName("PhasePending");
        progress_bar->setVisible(false);
    }

    label->style()->unpolish(label);
    label->style()->polish(label);

    int completed = 0;
    for (const auto &[id, lbl] : phase_status_labels_) {
        const QString txt = lbl->text();
        if (txt == "ok" || txt == "skipped" || txt == "error") {
            ++completed;
        }
    }
    progress_bar_->setValue(completed);
}

void PhaseProgressWidget::set_error_detail(const std::string &phase_name, const std::string &detail) {
    if (detail.empty()) {
        return;
    }
    last_error_text_ = phase_name + ": " + detail;
    error_label_->setText(QString("Abbruchgrund: %1").arg(QString::fromStdString(last_error_text_)));
    error_label_->setVisible(true);
    error_label_->style()->unpolish(error_label_);
    error_label_->style()->polish(error_label_);
}

}
