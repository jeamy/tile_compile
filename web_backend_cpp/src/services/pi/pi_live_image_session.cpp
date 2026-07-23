#include "services/pi/pi_live_image_session.hpp"

#include <opencv2/imgcodecs.hpp>

#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace tile_compile::pi {

std::string LiveImageSessionStore::generate_uuid() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);

    auto hex = [&](uint32_t v) {
        std::ostringstream ss;
        ss << std::hex << std::setfill('0') << std::setw(8) << v;
        return ss.str();
    };

    return hex(dist(gen)) + "-" + hex(dist(gen)) + "-" +
           hex(dist(gen)) + "-" + hex(dist(gen));
}

std::string LiveImageSessionStore::create(const std::string& run_id, cv::Mat fits) {
    return create(run_id, fits, fits);
}

std::string LiveImageSessionStore::create(const std::string& run_id,
                                          cv::Mat original, cv::Mat current) {
    auto session = std::make_unique<LiveImageSession>();
    session->session_id = generate_uuid();
    session->run_id = run_id;
    session->original_fits = original.clone();
    session->current_fits = current.clone();
    auto now = std::chrono::steady_clock::now();
    session->created_at = now;
    session->last_accessed = now;

    std::string id = session->session_id;

    std::lock_guard<std::mutex> lock(m_mutex);
    m_sessions.push_back(std::move(session));
    return id;
}

void LiveImageSessionStore::close(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sessions.erase(
        std::remove_if(m_sessions.begin(), m_sessions.end(),
                       [&](const std::unique_ptr<LiveImageSession>& s) {
                           return s->session_id == session_id;
                       }),
        m_sessions.end());
}

void LiveImageSessionStore::trim_snapshots(LiveImageSession& /*s*/, size_t /*max*/) {
    // Snapshot lifetime is bounded by the undo/redo stacks. The parameter is
    // retained for API compatibility with the session store.
}

void LiveImageSessionStore::rebuild_current_fits(LiveImageSession& s) {
    s.current_fits = s.original_fits.clone();
    std::ofstream dbg("/tmp/crop_debug.log", std::ios::app);
    dbg << "[CROP_DEBUG] rebuild: stack_size=" << s.undo_stack.size()
        << " original=" << s.original_fits.cols << "x" << s.original_fits.rows << "\n";
    for (const auto& entry : s.undo_stack) {
        nlohmann::json op = entry;
        op.erase("snapshot_b64");
        const std::string op_type = op.value("type", std::string());
        dbg << "[CROP_DEBUG]   applying op: " << op_type
            << " input=" << s.current_fits.cols << "x" << s.current_fits.rows
            << " params=" << op.value("params", nlohmann::json::object()).dump() << "\n";
        auto res = apply_image_op_fits(s.current_fits, op);
        dbg << "[CROP_DEBUG]   result: success=" << res.success
            << " error=" << res.error
            << " output=" << res.image.cols << "x" << res.image.rows << "\n";
        if (res.success) {
            s.current_fits = std::move(res.image);
        }
    }
    dbg << "[CROP_DEBUG] rebuild done: " << s.current_fits.cols << "x" << s.current_fits.rows << "\n";
}

void LiveImageSessionStore::evict_expired(int max_age_seconds, size_t max_sessions) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto now = std::chrono::steady_clock::now();

    // Remove expired sessions
    m_sessions.erase(
        std::remove_if(m_sessions.begin(), m_sessions.end(),
                       [&](const std::unique_ptr<LiveImageSession>& s) {
                           auto age = std::chrono::duration_cast<
                               std::chrono::seconds>(now - s->last_accessed).count();
                           return age > max_age_seconds;
                       }),
        m_sessions.end());

    // Enforce max_sessions via LRU
    if (m_sessions.size() > max_sessions) {
        std::sort(m_sessions.begin(), m_sessions.end(),
                  [](const std::unique_ptr<LiveImageSession>& a,
                     const std::unique_ptr<LiveImageSession>& b) {
                      return a->last_accessed > b->last_accessed;
                  });
        m_sessions.resize(max_sessions);
    }
}

bool LiveImageSessionStore::with_session(const std::string& session_id,
                                         const std::function<void(LiveImageSession&)>& fn) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id == session_id) {
            s->last_accessed = std::chrono::steady_clock::now();
            fn(*s);
            return true;
        }
    }
    return false;
}

ImageOpResult LiveImageSessionStore::apply_operation(const std::string& session_id,
                                                      const nlohmann::json& op) {
    ImageOpResult result;
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id == session_id) {
            s->last_accessed = std::chrono::steady_clock::now();
            const cv::Mat before = s->current_fits.clone();
            result = apply_image_op_fits(s->current_fits, op);
            {
                std::ofstream dbg("/tmp/crop_debug.log", std::ios::app);
                dbg << "[CROP_DEBUG] apply_operation: type=" << op.value("type", "")
                    << " success=" << result.success
                    << " input=" << s->current_fits.cols << "x" << s->current_fits.rows
                    << " output=" << result.image.cols << "x" << result.image.rows << "\n";
            }
            if (result.success) {
                if (op.value("type", "") == "reset") {
                    s->current_fits = s->original_fits.clone();
                    s->undo_stack.clear();
                    s->redo_stack.clear();
                    s->undo_snapshots.clear();
                    s->redo_snapshots.clear();
                    s->operation_history.clear();
                    s->edit_history.clear();
                    s->chat_history = nlohmann::json::array();
                    s->adjust_count = 0;
                    s->adjust_base_size = 0;
                    s->last_adjust_step = nullptr;
                    s->last_repeat_operation = nullptr;
                } else {
                    s->current_fits = std::move(result.image);
                    nlohmann::json stack_entry = op;
                    stack_entry["source"] = op.value("source", std::string("chat"));
                    s->undo_stack.push_back(stack_entry);
                    s->undo_snapshots.push_back(before);
                    s->redo_stack.clear();
                    s->redo_snapshots.clear();
                    s->operation_history = s->undo_stack;
                    s->edit_history.push_back({{"action", "apply"}, {"operation", stack_entry}});
                    s->last_repeat_operation = op;
                    std::ofstream dbg2("/tmp/crop_debug.log", std::ios::app);
                    dbg2 << "[CROP_DEBUG]   pushed to stack, size=" << s->undo_stack.size() << "\n";
                }
            }
            return result;
        }
    }
    result.error = "session not found";
    return result;
}

ImageOpResult LiveImageSessionStore::apply_preset(const std::string& session_id,
                                                   const nlohmann::json& operations) {
    ImageOpResult result;
    if (!operations.is_array()) {
        result.error = "preset operations must be an array";
        return result;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id != session_id) continue;
        s->last_accessed = std::chrono::steady_clock::now();
        const cv::Mat original_current = s->current_fits.clone();
        const auto original_undo = s->undo_stack;
        const auto original_redo = s->redo_stack;
        const auto original_undo_snapshots = s->undo_snapshots;
        const auto original_redo_snapshots = s->redo_snapshots;
        const auto original_operation_history = s->operation_history;
        const auto original_edit_history = s->edit_history;
        const auto original_last_repeat = s->last_repeat_operation;
        const auto original_last_adjust = s->last_adjust_step;
        const int original_adjust_count = s->adjust_count;
        const size_t original_adjust_base = s->adjust_base_size;
        for (const auto& raw_op : operations) {
            if (!raw_op.is_object() || !raw_op.contains("type")) continue;
            nlohmann::json op = raw_op;
            op["source"] = "preset";
            const cv::Mat before = s->current_fits.clone();
            auto op_result = apply_image_op_fits(s->current_fits, op);
            if (!op_result.success) {
                s->current_fits = original_current;
                s->undo_stack = original_undo;
                s->redo_stack = original_redo;
                s->undo_snapshots = original_undo_snapshots;
                s->redo_snapshots = original_redo_snapshots;
                s->operation_history = original_operation_history;
                s->edit_history = original_edit_history;
                s->last_repeat_operation = original_last_repeat;
                s->last_adjust_step = original_last_adjust;
                s->adjust_count = original_adjust_count;
                s->adjust_base_size = original_adjust_base;
                result.error = op_result.error;
                return result;
            }
            s->current_fits = std::move(op_result.image);
            s->undo_stack.push_back(op);
            s->undo_snapshots.push_back(before);
            s->edit_history.push_back({{"action", "apply"}, {"operation", op}});
            s->last_repeat_operation = op;
        }
        s->redo_stack.clear();
        s->redo_snapshots.clear();
        s->operation_history = s->undo_stack;
        result.image = s->current_fits.clone();
        result.success = true;
        return result;
    }
    result.error = "session not found";
    return result;
}

ImageOpResult LiveImageSessionStore::repeat_operation(const std::string& session_id) {
    ImageOpResult result;
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id != session_id) continue;
        s->last_accessed = std::chrono::steady_clock::now();
        if (s->last_repeat_operation.is_null() || s->last_repeat_operation.empty()) {
            result.error = "no repeatable operation set";
            return result;
        }
        result = apply_image_op_fits(s->current_fits, s->last_repeat_operation);
        if (!result.success) return result;
        const cv::Mat before = s->current_fits.clone();
        s->current_fits = std::move(result.image);
        result.image = s->current_fits.clone();
        nlohmann::json stack_entry = s->last_repeat_operation;
        stack_entry["source"] = "repeat";
        s->undo_stack.push_back(stack_entry);
        s->undo_snapshots.push_back(before);
        s->redo_stack.clear();
        s->redo_snapshots.clear();
        s->operation_history = s->undo_stack;
        s->edit_history.push_back({{"action", "apply"}, {"operation", stack_entry}});
        return result;
    }
    result.error = "session not found";
    return result;
}

ImageOpResult LiveImageSessionStore::apply_adjust(const std::string& session_id,
                                                   const std::string& direction) {
    ImageOpResult result;
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id == session_id) {
            s->last_accessed = std::chrono::steady_clock::now();
            if (s->last_adjust_step.is_null() || s->last_adjust_step.empty()) {
                result.error = "no adjust step set";
                return result;
            }
            if (direction != "increase" && direction != "decrease") {
                result.error = "invalid adjust direction";
                return result;
            }
            const int new_count = direction == "increase"
                ? std::min(20, s->adjust_count + 1)
                : std::max(0, s->adjust_count - 1);
            cv::Mat rebuilt = s->original_fits.clone();
            std::vector<nlohmann::json> new_stack;
            std::vector<cv::Mat> new_snapshots;
            const size_t base_size = std::min(s->adjust_base_size, s->undo_stack.size());
            for (size_t i = 0; i < base_size; ++i) {
                if (i >= s->undo_snapshots.size()) { result.error = "missing undo snapshot"; return result; }
                auto op_result = apply_image_op_fits(rebuilt, s->undo_stack[i]);
                if (!op_result.success) { result.error = op_result.error; return result; }
                rebuilt = std::move(op_result.image);
                new_stack.push_back(s->undo_stack[i]);
                new_snapshots.push_back(s->undo_snapshots[i]);
            }
            for (int i = 0; i < new_count; ++i) {
                const cv::Mat before = rebuilt.clone();
                auto op_result = apply_image_op_fits(rebuilt, s->last_adjust_step);
                if (!op_result.success) { result.error = op_result.error; return result; }
                rebuilt = std::move(op_result.image);
                nlohmann::json stack_entry = s->last_adjust_step;
                stack_entry["source"] = "adjust";
                new_stack.push_back(std::move(stack_entry));
                new_snapshots.push_back(before);
            }
            s->current_fits = std::move(rebuilt);
            s->undo_stack = std::move(new_stack);
            s->undo_snapshots = std::move(new_snapshots);
            s->redo_stack.clear();
            s->redo_snapshots.clear();
            s->adjust_count = new_count;
            s->operation_history = s->undo_stack;
            s->edit_history.push_back({{"action", "adjust"}, {"direction", direction},
                                       {"operation", s->last_adjust_step}});
            result.image = s->current_fits.clone();
            result.success = true;
            return result;
        }
    }
    result.error = "session not found";
    return result;
}

UndoRedoResult LiveImageSessionStore::undo(const std::string& session_id) {
    UndoRedoResult result;
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id == session_id) {
            s->last_accessed = std::chrono::steady_clock::now();
            result.can_undo = !s->undo_stack.empty();
            result.can_redo = !s->redo_stack.empty();
            if (s->undo_stack.empty()) {
                result.image = s->current_fits.clone();
                return result;
            }
            nlohmann::json entry = s->undo_stack.back();
            if (s->undo_snapshots.empty()) {
                result.summary = "undo snapshot missing";
                return result;
            }
            cv::Mat after = s->current_fits.clone();
            cv::Mat before = s->undo_snapshots.back();
            s->undo_stack.pop_back();
            s->undo_snapshots.pop_back();
            s->current_fits = before.clone();
            s->redo_stack.push_back(entry);
            s->redo_snapshots.push_back(std::move(after));
            s->operation_history = s->undo_stack;
            s->edit_history.push_back({{"action", "undo"}, {"operation", entry}});
            if (entry.value("source", "") == "adjust")
                s->adjust_count = std::max(0, s->adjust_count - 1);

            result.image = s->current_fits.clone();
            result.summary = "Ruckgangig";
            result.can_undo = !s->undo_stack.empty();
            result.can_redo = true;
            result.count = static_cast<int>(s->undo_stack.size());
            return result;
        }
    }
    result.summary = "session not found";
    return result;
}

UndoRedoResult LiveImageSessionStore::redo(const std::string& session_id) {
    UndoRedoResult result;
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id == session_id) {
            s->last_accessed = std::chrono::steady_clock::now();
            result.can_undo = !s->undo_stack.empty();
            result.can_redo = !s->redo_stack.empty();
            if (s->redo_stack.empty()) {
                result.image = s->current_fits.clone();
                return result;
            }
            nlohmann::json entry = s->redo_stack.back();
            if (s->redo_snapshots.empty()) {
                result.summary = "redo snapshot missing";
                return result;
            }
            cv::Mat before = s->current_fits.clone();
            cv::Mat after = s->redo_snapshots.back();
            s->redo_stack.pop_back();
            s->redo_snapshots.pop_back();
            s->undo_stack.push_back(entry);
            s->undo_snapshots.push_back(std::move(before));
            s->current_fits = after.clone();
            s->operation_history = s->undo_stack;
            s->edit_history.push_back({{"action", "redo"}, {"operation", entry}});
            if (entry.value("source", "") == "adjust") ++s->adjust_count;

            result.image = s->current_fits.clone();
            result.summary = "Wiederhergestellt";
            result.can_undo = true;
            result.can_redo = !s->redo_stack.empty();
            result.count = static_cast<int>(s->redo_stack.size());
            return result;
        }
    }
    result.summary = "session not found";
    return result;
}

cv::Mat LiveImageSessionStore::reset(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id == session_id) {
            s->last_accessed = std::chrono::steady_clock::now();
            s->current_fits = s->original_fits.clone();
            s->undo_stack.clear();
            s->redo_stack.clear();
            s->undo_snapshots.clear();
            s->redo_snapshots.clear();
            s->adjust_count = 0;
            s->operation_history.clear();
            s->edit_history.clear();
            s->chat_history = nlohmann::json::array();
            s->last_adjust_step = nullptr;
            s->last_repeat_operation = nullptr;
            s->adjust_base_size = 0;
            return s->current_fits.clone();
        }
    }
    return {};
}

void LiveImageSessionStore::set_adjust_step(const std::string& session_id,
                                             const nlohmann::json& step) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id == session_id) {
            s->last_accessed = std::chrono::steady_clock::now();
            s->last_adjust_step = step;
            s->adjust_count = 0;
            s->adjust_base_size = s->undo_stack.size();
            return;
        }
    }
}

void LiveImageSessionStore::append_chat(const std::string& session_id,
                                         const std::string& role,
                                         const std::string& content,
                                         const nlohmann::json& operations) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id == session_id) {
            s->last_accessed = std::chrono::steady_clock::now();
            nlohmann::json msg = {
                {"role", role},
                {"content", content}
            };
            if (operations.is_array() && !operations.empty()) {
                msg["operations"] = operations;
            }
            s->chat_history.push_back(msg);
            return;
        }
    }
}

nlohmann::json LiveImageSessionStore::get_chat_history(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id == session_id) {
            s->last_accessed = std::chrono::steady_clock::now();
            return s->chat_history;
        }
    }
    return nlohmann::json::array();
}

nlohmann::json LiveImageSessionStore::get_operation_history(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& s : m_sessions) {
        if (s->session_id == session_id) {
            s->last_accessed = std::chrono::steady_clock::now();
            return s->operation_history;
        }
    }
    return nlohmann::json::array();
}

} // namespace tile_compile::pi
