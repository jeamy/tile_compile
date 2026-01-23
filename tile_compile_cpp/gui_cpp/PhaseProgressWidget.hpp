#pragma once

#include <QWidget>
#include <QLabel>
#include <QProgressBar>
#include <QVBoxLayout>
#include <QGridLayout>
#include <map>
#include <string>

namespace tile_compile::gui {

class PhaseProgressWidget : public QWidget {
    Q_OBJECT

  public:
    explicit PhaseProgressWidget(QWidget *parent = nullptr);

    void reset();
    void set_reduced_mode(bool enabled, int frame_count = 0);
    void update_phase(const std::string &phase_name, const std::string &status,
                      int progress_current = 0, int progress_total = 0,
                      const std::string &substep = std::string(),
                      const std::string &pass_info = std::string());
    void set_error_detail(const std::string &phase_name, const std::string &detail);

  private:
    void build_ui();
    
    QLabel *reduced_mode_label_ = nullptr;
    QProgressBar *progress_bar_ = nullptr;
    QLabel *error_label_ = nullptr;
    
    std::map<int, QLabel*> phase_labels_;
    std::map<int, QLabel*> phase_status_labels_;
    std::map<int, QProgressBar*> phase_progress_bars_;
    
    bool reduced_mode_ = false;
    std::string last_error_text_;
};

}
