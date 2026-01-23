#include "RunTab.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>

namespace tile_compile::gui {

RunTab::RunTab(const std::string &project_root, QWidget *parent)
    : QWidget(parent), project_root_(project_root) {
    build_ui();
}

void RunTab::build_ui() {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);
    
    auto *run_box = new QGroupBox("Run");
    auto *run_layout = new QVBoxLayout(run_box);
    run_layout->setContentsMargins(12, 18, 12, 12);
    run_layout->setSpacing(10);
    
    auto *rr0 = new QHBoxLayout();
    working_dir_ = new QLineEdit(QString::fromStdString(project_root_));
    auto *btn_browse_working_dir = new QPushButton("Browse");
    rr0->addWidget(new QLabel("Working dir"));
    rr0->addWidget(working_dir_, 1);
    rr0->addWidget(btn_browse_working_dir);
    run_layout->addLayout(rr0);
    
    auto *rr1 = new QHBoxLayout();
    input_dir_ = new QLineEdit("");
    auto *btn_browse_input_dir = new QPushButton("Browse");
    rr1->addWidget(new QLabel("Input dir"));
    rr1->addWidget(input_dir_, 1);
    rr1->addWidget(btn_browse_input_dir);
    run_layout->addLayout(rr1);
    
    auto *rr2 = new QHBoxLayout();
    runs_dir_ = new QLineEdit("runs");
    pattern_ = new QLineEdit("*.fit*");
    dry_run_ = new QCheckBox("Dry run");
    rr2->addWidget(new QLabel("Runs dir"));
    rr2->addWidget(runs_dir_);
    rr2->addWidget(new QLabel("Pattern"));
    rr2->addWidget(pattern_, 1);
    rr2->addWidget(dry_run_);
    run_layout->addLayout(rr2);
    
    auto *rr3 = new QHBoxLayout();
    btn_start_ = new QPushButton("Start");
    btn_abort_ = new QPushButton("Stop");
    lbl_run_ = new QLabel("idle");
    lbl_run_->setObjectName("StatusLabel");
    rr3->addWidget(btn_start_);
    rr3->addWidget(btn_abort_);
    rr3->addStretch(1);
    rr3->addWidget(lbl_run_);
    run_layout->addLayout(rr3);
    
    run_reduced_mode_hint_ = new QLabel("");
    run_reduced_mode_hint_->setObjectName("ReducedModeWarning");
    run_reduced_mode_hint_->setVisible(false);
    run_layout->addWidget(run_reduced_mode_hint_);
    
    layout->addWidget(run_box, 1);
    
    connect(btn_start_, &QPushButton::clicked, this, &RunTab::start_run_clicked);
    connect(btn_abort_, &QPushButton::clicked, this, &RunTab::abort_run_clicked);
    connect(btn_browse_working_dir, &QPushButton::clicked, this, &RunTab::on_browse_working_dir);
    connect(btn_browse_input_dir, &QPushButton::clicked, this, &RunTab::on_browse_input_dir);
    connect(working_dir_, &QLineEdit::editingFinished, this, &RunTab::working_dir_changed);
    connect(input_dir_, &QLineEdit::editingFinished, this, &RunTab::input_dir_changed);
}

void RunTab::on_browse_working_dir() {
    const QString start = working_dir_->text().trimmed().isEmpty() 
        ? QString::fromStdString(project_root_) 
        : working_dir_->text();
    const QString p = QFileDialog::getExistingDirectory(this, "Select working directory", start);
    if (!p.isEmpty()) {
        working_dir_->setText(p);
        emit working_dir_changed();
    }
}

void RunTab::on_browse_input_dir() {
    const QString start = input_dir_->text().trimmed().isEmpty() 
        ? QString::fromStdString(project_root_) 
        : input_dir_->text();
    const QString p = QFileDialog::getExistingDirectory(this, "Select input directory", start);
    if (!p.isEmpty()) {
        input_dir_->setText(p);
        emit input_dir_changed();
    }
}

QString RunTab::get_working_dir() const { return working_dir_->text().trimmed(); }
QString RunTab::get_input_dir() const { return input_dir_->text().trimmed(); }
QString RunTab::get_runs_dir() const { return runs_dir_->text().trimmed(); }
QString RunTab::get_pattern() const { return pattern_->text().trimmed(); }
bool RunTab::is_dry_run() const { return dry_run_->isChecked(); }

void RunTab::set_working_dir(const QString &dir) { working_dir_->setText(dir); }
void RunTab::set_input_dir(const QString &dir) { input_dir_->setText(dir); }
void RunTab::set_runs_dir(const QString &dir) { runs_dir_->setText(dir); }
void RunTab::set_pattern(const QString &pattern) { pattern_->setText(pattern); }
void RunTab::set_dry_run(bool dry_run) { dry_run_->setChecked(dry_run); }

void RunTab::set_start_enabled(bool enabled, const QString &tooltip) {
    btn_start_->setEnabled(enabled);
    btn_start_->setToolTip(tooltip);
}

void RunTab::set_abort_enabled(bool enabled) {
    btn_abort_->setEnabled(enabled);
}

void RunTab::set_status_text(const QString &text) {
    lbl_run_->setText(text);
}

void RunTab::show_reduced_mode_hint(int frame_count, int threshold) {
    run_reduced_mode_hint_->setText(
        QString("⚠ Reduced Mode: %1 Frames < %2\n"
                "STATE_CLUSTERING und SYNTHETIC_FRAMES werden übersprungen.")
            .arg(frame_count).arg(threshold));
    run_reduced_mode_hint_->setVisible(true);
}

void RunTab::hide_reduced_mode_hint() {
    run_reduced_mode_hint_->setVisible(false);
}

}
