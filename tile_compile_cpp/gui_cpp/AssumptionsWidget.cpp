#include "AssumptionsWidget.hpp"

#include <QGroupBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>

namespace tile_compile::gui {

namespace {
constexpr double EXPOSURE_TIME_TOLERANCE_PERCENT = 5.0;
constexpr int FRAMES_MIN = 50;
constexpr int FRAMES_OPTIMAL = 800;
constexpr int FRAMES_REDUCED_THRESHOLD = 200;
constexpr double REGISTRATION_RESIDUAL_WARN_PX = 0.5;
constexpr double REGISTRATION_RESIDUAL_MAX_PX = 1.0;
constexpr double ELONGATION_WARN = 0.3;
constexpr double ELONGATION_MAX = 0.4;
constexpr double TRACKING_ERROR_MAX_PX = 1.0;
constexpr bool REDUCED_MODE_SKIP_CLUSTERING = true;
constexpr int REDUCED_MODE_CLUSTER_MIN = 15;
constexpr int REDUCED_MODE_CLUSTER_MAX = 30;
}

AssumptionsWidget::AssumptionsWidget(QWidget *parent) : QWidget(parent) {
    build_ui();
}

void AssumptionsWidget::build_ui() {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(12);

    auto *reset_row = new QHBoxLayout();
    reset_row->addStretch(1);
    auto *btn_reset = new QPushButton("Default");
    connect(btn_reset, &QPushButton::clicked, this, [this]() {
        reset_to_defaults();
        emit assumptions_changed();
    });
    reset_row->addWidget(btn_reset);
    layout->addLayout(reset_row);

    auto *hard_box = new QGroupBox("Hard Assumptions (Verletzung → Abbruch)");
    auto *hard_layout = new QVBoxLayout(hard_box);
    
    const char *hard_items[] = {
        "Lineare Daten (kein Stretch, keine nicht-linearen Operatoren)",
        "Keine Frame-Selektion (Pixel-Level Artefakt-Rejection erlaubt)",
        "Kanal-getrennte Verarbeitung (kein Channel Coupling)",
        "Strikt lineare Pipeline (keine Feedback-Loops)",
    };
    
    for (const char *item : hard_items) {
        auto *lbl = new QLabel(QString("• %1").arg(item));
        lbl->setObjectName("HardAssumption");
        lbl->setWordWrap(true);
        hard_layout->addWidget(lbl);
    }
    
    auto *exp_row = new QHBoxLayout();
    exp_row->addWidget(new QLabel("• Einheitliche Belichtungszeit (Toleranz: ±"));
    exposure_tolerance_ = new QDoubleSpinBox();
    exposure_tolerance_->setRange(0.1, 20.0);
    exposure_tolerance_->setValue(EXPOSURE_TIME_TOLERANCE_PERCENT);
    exposure_tolerance_->setSuffix(" %)");
    exposure_tolerance_->setFixedWidth(100);
    connect(exposure_tolerance_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &AssumptionsWidget::assumptions_changed);
    exp_row->addWidget(exposure_tolerance_);
    exp_row->addStretch(1);
    hard_layout->addLayout(exp_row);
    layout->addWidget(hard_box);

    auto *soft_box = new QGroupBox("Soft Assumptions (mit Toleranzen)");
    auto *soft_layout = new QFormLayout(soft_box);
    soft_layout->setSpacing(8);
    
    auto *frame_row = new QHBoxLayout();
    frames_min_ = new QSpinBox();
    frames_min_->setRange(1, 10000);
    frames_min_->setValue(FRAMES_MIN);
    connect(frames_min_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &AssumptionsWidget::assumptions_changed);
    
    frames_reduced_ = new QSpinBox();
    frames_reduced_->setRange(1, 10000);
    frames_reduced_->setValue(FRAMES_REDUCED_THRESHOLD);
    connect(frames_reduced_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &AssumptionsWidget::assumptions_changed);
    
    frames_optimal_ = new QSpinBox();
    frames_optimal_->setRange(1, 100000);
    frames_optimal_->setValue(FRAMES_OPTIMAL);
    connect(frames_optimal_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &AssumptionsWidget::assumptions_changed);
    
    frame_row->addWidget(new QLabel("min:"));
    frame_row->addWidget(frames_min_);
    frame_row->addWidget(new QLabel("reduced:"));
    frame_row->addWidget(frames_reduced_);
    frame_row->addWidget(new QLabel("optimal:"));
    frame_row->addWidget(frames_optimal_);
    frame_row->addStretch(1);
    soft_layout->addRow("Frame-Anzahl", frame_row);
    
    auto *reg_row = new QHBoxLayout();
    reg_warn_ = new QDoubleSpinBox();
    reg_warn_->setRange(0.01, 10.0);
    reg_warn_->setValue(REGISTRATION_RESIDUAL_WARN_PX);
    reg_warn_->setSuffix(" px");
    reg_warn_->setDecimals(2);
    connect(reg_warn_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &AssumptionsWidget::assumptions_changed);
    
    reg_max_ = new QDoubleSpinBox();
    reg_max_->setRange(0.01, 10.0);
    reg_max_->setValue(REGISTRATION_RESIDUAL_MAX_PX);
    reg_max_->setSuffix(" px");
    reg_max_->setDecimals(2);
    connect(reg_max_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &AssumptionsWidget::assumptions_changed);
    
    reg_row->addWidget(new QLabel("warn:"));
    reg_row->addWidget(reg_warn_);
    reg_row->addWidget(new QLabel("max:"));
    reg_row->addWidget(reg_max_);
    reg_row->addStretch(1);
    soft_layout->addRow("Registrierungs-Residual", reg_row);
    
    auto *elong_row = new QHBoxLayout();
    elong_warn_ = new QDoubleSpinBox();
    elong_warn_->setRange(0.01, 1.0);
    elong_warn_->setValue(ELONGATION_WARN);
    elong_warn_->setDecimals(2);
    connect(elong_warn_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &AssumptionsWidget::assumptions_changed);
    
    elong_max_ = new QDoubleSpinBox();
    elong_max_->setRange(0.01, 1.0);
    elong_max_->setValue(ELONGATION_MAX);
    elong_max_->setDecimals(2);
    connect(elong_max_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &AssumptionsWidget::assumptions_changed);
    
    elong_row->addWidget(new QLabel("warn:"));
    elong_row->addWidget(elong_warn_);
    elong_row->addWidget(new QLabel("max:"));
    elong_row->addWidget(elong_max_);
    elong_row->addStretch(1);
    soft_layout->addRow("Stern-Elongation", elong_row);
    layout->addWidget(soft_box);

    auto *implicit_box = new QGroupBox("Implicit Assumptions (jetzt explizit)");
    auto *implicit_layout = new QFormLayout(implicit_box);
    implicit_layout->setSpacing(8);
    
    auto *tracking_row = new QHBoxLayout();
    tracking_row->addWidget(new QLabel("• Tracking-Fehler max:"));
    tracking_error_max_ = new QDoubleSpinBox();
    tracking_error_max_->setRange(0.1, 10.0);
    tracking_error_max_->setValue(TRACKING_ERROR_MAX_PX);
    tracking_error_max_->setSuffix(" px");
    tracking_error_max_->setDecimals(2);
    tracking_error_max_->setFixedWidth(120);
    connect(tracking_error_max_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &AssumptionsWidget::assumptions_changed);
    tracking_row->addWidget(tracking_error_max_);
    tracking_row->addStretch(1);
    implicit_layout->addRow("", tracking_row);
    
    const char *implicit_items[] = {
        "Stabile optische Konfiguration (Fokus, Feldkrümmung)",
        "Kein systematischer Drift während der Session",
    };
    for (const char *item : implicit_items) {
        auto *lbl = new QLabel(QString("• %1").arg(item));
        lbl->setObjectName("ImplicitAssumption");
        lbl->setWordWrap(true);
        implicit_layout->addRow("", lbl);
    }
    layout->addWidget(implicit_box);

    auto *reduced_box = new QGroupBox("Reduced Mode (50–199 Frames)");
    auto *reduced_layout = new QFormLayout(reduced_box);
    reduced_layout->setSpacing(8);
    
    skip_clustering_ = new QCheckBox("STATE_CLUSTERING und SYNTHETIC_FRAMES überspringen");
    skip_clustering_->setChecked(REDUCED_MODE_SKIP_CLUSTERING);
    connect(skip_clustering_, &QCheckBox::checkStateChanged,
            this, &AssumptionsWidget::assumptions_changed);
    reduced_layout->addRow("", skip_clustering_);
    
    auto *cluster_row = new QHBoxLayout();
    cluster_min_ = new QSpinBox();
    cluster_min_->setRange(1, 100);
    cluster_min_->setValue(REDUCED_MODE_CLUSTER_MIN);
    connect(cluster_min_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &AssumptionsWidget::assumptions_changed);
    
    cluster_max_ = new QSpinBox();
    cluster_max_->setRange(1, 100);
    cluster_max_->setValue(REDUCED_MODE_CLUSTER_MAX);
    connect(cluster_max_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &AssumptionsWidget::assumptions_changed);
    
    cluster_row->addWidget(new QLabel("min:"));
    cluster_row->addWidget(cluster_min_);
    cluster_row->addWidget(new QLabel("max:"));
    cluster_row->addWidget(cluster_max_);
    cluster_row->addStretch(1);
    reduced_layout->addRow("Cluster-Range (falls nicht übersprungen)", cluster_row);
    
    reduced_mode_status_ = new QLabel("");
    reduced_mode_status_->setObjectName("ReducedModeWarning");
    reduced_mode_status_->setVisible(false);
    reduced_layout->addRow("", reduced_mode_status_);
    layout->addWidget(reduced_box);
    layout->addStretch(1);
}

void AssumptionsWidget::reset_to_defaults() {
    exposure_tolerance_->blockSignals(true);
    frames_min_->blockSignals(true);
    frames_reduced_->blockSignals(true);
    frames_optimal_->blockSignals(true);
    reg_warn_->blockSignals(true);
    reg_max_->blockSignals(true);
    elong_warn_->blockSignals(true);
    elong_max_->blockSignals(true);
    tracking_error_max_->blockSignals(true);
    skip_clustering_->blockSignals(true);
    cluster_min_->blockSignals(true);
    cluster_max_->blockSignals(true);

    exposure_tolerance_->setValue(EXPOSURE_TIME_TOLERANCE_PERCENT);
    frames_min_->setValue(FRAMES_MIN);
    frames_reduced_->setValue(FRAMES_REDUCED_THRESHOLD);
    frames_optimal_->setValue(FRAMES_OPTIMAL);
    reg_warn_->setValue(REGISTRATION_RESIDUAL_WARN_PX);
    reg_max_->setValue(REGISTRATION_RESIDUAL_MAX_PX);
    elong_warn_->setValue(ELONGATION_WARN);
    elong_max_->setValue(ELONGATION_MAX);
    tracking_error_max_->setValue(TRACKING_ERROR_MAX_PX);
    skip_clustering_->setChecked(REDUCED_MODE_SKIP_CLUSTERING);
    cluster_min_->setValue(REDUCED_MODE_CLUSTER_MIN);
    cluster_max_->setValue(REDUCED_MODE_CLUSTER_MAX);

    exposure_tolerance_->blockSignals(false);
    frames_min_->blockSignals(false);
    frames_reduced_->blockSignals(false);
    frames_optimal_->blockSignals(false);
    reg_warn_->blockSignals(false);
    reg_max_->blockSignals(false);
    elong_warn_->blockSignals(false);
    elong_max_->blockSignals(false);
    tracking_error_max_->blockSignals(false);
    skip_clustering_->blockSignals(false);
    cluster_min_->blockSignals(false);
    cluster_max_->blockSignals(false);
}

nlohmann::json AssumptionsWidget::get_assumptions() const {
    return {
        {"frames_min", frames_min_->value()},
        {"frames_optimal", frames_optimal_->value()},
        {"frames_reduced_threshold", frames_reduced_->value()},
        {"exposure_time_tolerance_percent", exposure_tolerance_->value()},
        {"registration_residual_warn_px", reg_warn_->value()},
        {"registration_residual_max_px", reg_max_->value()},
        {"elongation_warn", elong_warn_->value()},
        {"elongation_max", elong_max_->value()},
        {"tracking_error_max_px", tracking_error_max_->value()},
        {"reduced_mode_skip_clustering", skip_clustering_->isChecked()},
        {"reduced_mode_cluster_range", nlohmann::json::array({cluster_min_->value(), cluster_max_->value()})},
    };
}

void AssumptionsWidget::set_assumptions(const nlohmann::json &assumptions) {
    if (assumptions.contains("frames_min")) {
        frames_min_->setValue(assumptions["frames_min"].get<int>());
    }
    if (assumptions.contains("frames_optimal")) {
        frames_optimal_->setValue(assumptions["frames_optimal"].get<int>());
    }
    if (assumptions.contains("frames_reduced_threshold")) {
        frames_reduced_->setValue(assumptions["frames_reduced_threshold"].get<int>());
    }
    if (assumptions.contains("exposure_time_tolerance_percent")) {
        exposure_tolerance_->setValue(assumptions["exposure_time_tolerance_percent"].get<double>());
    }
    if (assumptions.contains("registration_residual_warn_px")) {
        reg_warn_->setValue(assumptions["registration_residual_warn_px"].get<double>());
    }
    if (assumptions.contains("registration_residual_max_px")) {
        reg_max_->setValue(assumptions["registration_residual_max_px"].get<double>());
    }
    if (assumptions.contains("elongation_warn")) {
        elong_warn_->setValue(assumptions["elongation_warn"].get<double>());
    }
    if (assumptions.contains("elongation_max")) {
        elong_max_->setValue(assumptions["elongation_max"].get<double>());
    }
    if (assumptions.contains("tracking_error_max_px")) {
        tracking_error_max_->setValue(assumptions["tracking_error_max_px"].get<double>());
    }
    if (assumptions.contains("reduced_mode_skip_clustering")) {
        skip_clustering_->setChecked(assumptions["reduced_mode_skip_clustering"].get<bool>());
    }
    if (assumptions.contains("reduced_mode_cluster_range")) {
        const auto &rng = assumptions["reduced_mode_cluster_range"];
        if (rng.is_array() && rng.size() >= 2) {
            cluster_min_->setValue(rng[0].get<int>());
            cluster_max_->setValue(rng[1].get<int>());
        }
    }
}

void AssumptionsWidget::update_reduced_mode_status(int frame_count) {
    const int threshold = frames_reduced_->value();
    const int minimum = frames_min_->value();
    
    if (frame_count < minimum) {
        reduced_mode_status_->setText(QString("⛔ Frame-Anzahl (%1) unter Minimum (%2)").arg(frame_count).arg(minimum));
        reduced_mode_status_->setVisible(true);
    } else if (frame_count < threshold) {
        reduced_mode_status_->setText(QString("⚠ Reduced Mode aktiv: %1 Frames < %2").arg(frame_count).arg(threshold));
        reduced_mode_status_->setVisible(true);
    } else {
        reduced_mode_status_->setVisible(false);
    }
}

}
