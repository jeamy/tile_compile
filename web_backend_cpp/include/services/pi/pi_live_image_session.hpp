#pragma once
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <chrono>
#include <functional>
#include "pi_image_ops.hpp"

namespace tile_compile::pi {

struct LiveImageSession {
    std::string session_id;
    std::string run_id;
    // original_fits and current_fits hold linear float data (CV_32F, BGR,
    // values in [0,1]) derived from the source FITS. All image operations are
    // applied to this data; the 8-bit preview JPEG is rendered on demand.
    cv::Mat original_fits;
    cv::Mat current_fits;
    std::vector<nlohmann::json> undo_stack;
    std::vector<nlohmann::json> redo_stack;
    // Exact pixel states: one pre-operation snapshot per undo entry and one
    // post-operation snapshot per redo entry. This is required for lossy or
    // non-invertible operations such as threshold, crop and denoise.
    std::vector<cv::Mat> undo_snapshots;
    std::vector<cv::Mat> redo_snapshots;
    nlohmann::json chat_history = nlohmann::json::array();
    nlohmann::json operation_history = nlohmann::json::array();
    // Complete editing timeline, including undo/redo actions. This is kept
    // separate from operation_history, which represents only the active stack
    // used to reconstruct the current image.
    nlohmann::json edit_history = nlohmann::json::array();
    nlohmann::json last_adjust_step;
    nlohmann::json last_repeat_operation;
    int adjust_count = 0;
    size_t adjust_base_size = 0;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_accessed;
};

struct UndoRedoResult {
    cv::Mat image;
    std::string summary;
    bool can_undo = false;
    bool can_redo = false;
    int count = 0;
};

class LiveImageSessionStore {
public:
    std::string create(const std::string& run_id, cv::Mat image);
    std::string create(const std::string& run_id, cv::Mat original, cv::Mat current);
    void close(const std::string& session_id);
    void evict_expired(int max_age_seconds = 1800, size_t max_sessions = 5);

    bool with_session(const std::string& session_id,
                      const std::function<void(LiveImageSession&)>& fn);

    ImageOpResult apply_operation(const std::string& session_id,
                                  const nlohmann::json& op);
    ImageOpResult apply_preset(const std::string& session_id,
                               const nlohmann::json& operations);
    ImageOpResult apply_adjust(const std::string& session_id,
                               const std::string& direction);
    ImageOpResult repeat_operation(const std::string& session_id);
    UndoRedoResult undo(const std::string& session_id);
    UndoRedoResult redo(const std::string& session_id);
    cv::Mat reset(const std::string& session_id);

    void set_adjust_step(const std::string& session_id,
                         const nlohmann::json& step);
    void append_chat(const std::string& session_id,
                     const std::string& role, const std::string& content,
                     const nlohmann::json& operations = nullptr);
    nlohmann::json get_chat_history(const std::string& session_id);
    nlohmann::json get_operation_history(const std::string& session_id);

private:
    std::mutex m_mutex;
    std::vector<std::unique_ptr<LiveImageSession>> m_sessions;
    std::string generate_uuid() const;
    void trim_snapshots(LiveImageSession& s, size_t max = 10);
    void rebuild_current_fits(LiveImageSession& s);
};

} // namespace tile_compile::pi
