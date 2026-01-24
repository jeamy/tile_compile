#pragma once

#include <QWidget>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <map>
#include <string>

#include <nlohmann/json.hpp>

namespace tile_compile::gui {

class AssumptionsWidget : public QWidget {
    Q_OBJECT

  public:
    explicit AssumptionsWidget(QWidget *parent = nullptr);

    nlohmann::json get_assumptions() const;
    void set_assumptions(const nlohmann::json &assumptions);
    void update_reduced_mode_status(int frame_count);
    void reset_to_defaults();

  signals:
    void assumptions_changed();

  private:
    void build_ui();

    QDoubleSpinBox *exposure_tolerance_ = nullptr;
    QSpinBox *frames_min_ = nullptr;
    QSpinBox *frames_reduced_ = nullptr;
    QSpinBox *frames_optimal_ = nullptr;
    QDoubleSpinBox *reg_warn_ = nullptr;
    QDoubleSpinBox *reg_max_ = nullptr;
    QDoubleSpinBox *elong_warn_ = nullptr;
    QDoubleSpinBox *elong_max_ = nullptr;
    QDoubleSpinBox *tracking_error_max_ = nullptr;
    QCheckBox *skip_clustering_ = nullptr;
    QSpinBox *cluster_min_ = nullptr;
    QSpinBox *cluster_max_ = nullptr;
    QLabel *reduced_mode_status_ = nullptr;
};

}
