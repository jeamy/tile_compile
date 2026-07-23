#include "services/pi/pi_live_image_session.hpp"

#include <opencv2/imgcodecs.hpp>

#include <random>
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
    auto session = std::make_unique<LiveImageSession>();
    session->session_id = generate_uuid();
    session->run_id = run_id;
    session->original_fits = fits.clone();
    session->current_fits = fits.clone();
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
    // Snapshots are no longer used: undo/redo rebuild from original_fits.
}

void LiveImageSessionStore::rebuild_current_fits(LiveImageSession& s) {
    s.current_fits = s.original_fits.clone();
    for (const auto& entry : s.undo_stack) {
        nlohmann::json op = entry;
        op.erase("snapshot_b64");
        auto res = apply_image_op_fits(s.current_fits, op);
        if (res.success) {
            s.current_fits = std::move(res.image);
        }
    }
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
            result = apply_image_op_fits(s->current_fits, op);
            if (result.success) {
                s->current_fits = std::move(result.image);
                s->undo_stack.push_back(op);
                s->redo_stack.clear();

                // Record in operation_history
                nlohmann::json hist_entry = op;
                hist_entry["timestamp"] = "";
                hist_entry["source"] = "chat";
                s->operation_history.push_back(hist_entry);
            }
            return result;
        }
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
            nlohmann::json op;
            if (direction == "increase") {
                op = s->last_adjust_step;
                s->adjust_count++;
            } else {
                op = invert_op(s->last_adjust_step);
                s->adjust_count = std::max(0, s->adjust_count - 1);
            }
            result = apply_image_op_fits(s->current_fits, op);
            if (result.success) {
                s->current_fits = std::move(result.image);
                s->undo_stack.push_back(op);
                s->redo_stack.clear();

                nlohmann::json hist_entry = op;
                hist_entry["timestamp"] = "";
                hist_entry["source"] = "adjust";
                s->operation_history.push_back(hist_entry);
            }
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
            s->undo_stack.pop_back();
            rebuild_current_fits(*s);
            s->redo_stack.push_back(entry);

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
            s->redo_stack.pop_back();
            s->undo_stack.push_back(entry);
            rebuild_current_fits(*s);

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
            s->adjust_count = 0;
            s->operation_history.clear();
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
