#include "tile_compile/core/events.hpp"
#include "tile_compile/core/utils.hpp"

namespace tile_compile::core {

/// @brief Implements base event.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
json EventEmitter::base_event(const std::string& type, const std::string& run_id) {
    return {
        {"type", type},
        {"run_id", run_id},
        {"ts", get_iso_timestamp()}
    };
}

/// @brief Implements emit.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void EventEmitter::emit(const json& event, std::ostream& out) {
    out << event.dump() << "\n";
    out.flush();
}

/// @brief Runs start.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void EventEmitter::run_start(const std::string& run_id, const json& extra, std::ostream& out) {
    json event = base_event("run_start", run_id);
    for (auto& [key, value] : extra.items()) {
        event[key] = value;
    }
    emit(event, out);
}

/// @brief Runs end.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void EventEmitter::run_end(const std::string& run_id, bool success, 
                           const std::string& status, std::ostream& out) {
    json event = base_event("run_end", run_id);
    event["success"] = success;
    event["status"] = status;
    emit(event, out);
}

/// @brief Implements phase start.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void EventEmitter::phase_start(const std::string& run_id, Phase phase, 
                               const std::string& name, std::ostream& out) {
    json event = base_event("phase_start", run_id);
    event["phase"] = phase_to_int(phase);
    event["phase_name"] = name;
    emit(event, out);
}

/// @brief Implements phase progress.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void EventEmitter::phase_progress(const std::string& run_id, Phase phase, float progress,
                                  const std::string& message, std::ostream& out) {
    json event = base_event("phase_progress", run_id);
    event["phase"] = phase_to_int(phase);
    event["phase_name"] = phase_to_string(phase);
    // Convert float progress to current/total for GUI compatibility
    int current = static_cast<int>(progress * 100);
    int total = 100;
    event["current"] = current;
    event["total"] = total;
    event["progress"] = progress;
    event["substep"] = message;
    emit(event, out);
}

/// @brief Implements phase progress counts.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void EventEmitter::phase_progress_counts(const std::string& run_id, Phase phase, int current, int total,
                                         const std::string& substep, const std::string& pass,
                                         std::ostream& out) {
    json event = base_event("phase_progress", run_id);
    event["phase"] = phase_to_int(phase);
    event["phase_name"] = phase_to_string(phase);
    event["current"] = current;
    event["total"] = total;
    if (total > 0) {
        event["progress"] = static_cast<float>(current) / static_cast<float>(total);
    } else {
        event["progress"] = 0.0f;
    }
    if (!substep.empty()) event["substep"] = substep;
    if (!pass.empty()) event["pass"] = pass;
    emit(event, out);
}

/// @brief Implements phase end.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void EventEmitter::phase_end(const std::string& run_id, Phase phase, 
                             const std::string& status, const json& extra, std::ostream& out) {
    json event = base_event("phase_end", run_id);
    event["phase"] = phase_to_int(phase);
    event["phase_name"] = phase_to_string(phase);
    event["status"] = status;
    for (auto& [key, value] : extra.items()) {
        event[key] = value;
    }
    emit(event, out);
}

/// @brief Implements frame processed.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void EventEmitter::frame_processed(const std::string& run_id, Phase phase, int frame_idx,
                                   int total_frames, const std::string& frame_name, 
                                   std::ostream& out) {
    json event = base_event("frame_processed", run_id);
    event["phase"] = phase_to_int(phase);
    event["frame_idx"] = frame_idx;
    event["total_frames"] = total_frames;
    event["frame_name"] = frame_name;
    emit(event, out);
}

/// @brief Implements warning.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void EventEmitter::warning(const std::string& run_id, const std::string& message, 
                           std::ostream& out) {
    json event = base_event("warning", run_id);
    event["message"] = message;
    emit(event, out);
}

/// @brief Implements error.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void EventEmitter::error(const std::string& run_id, const std::string& message, 
                         std::ostream& out) {
    json event = base_event("error", run_id);
    event["message"] = message;
    emit(event, out);
}

/// @brief Implements emit event.
/// @details Part of structured JSON event emission for runner logs; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void emit_event(const std::string& type, const std::string& run_id,
                const json& data, std::ostream& out) {
    json event = {
        {"type", type},
        {"run_id", run_id},
        {"ts", get_iso_timestamp()}
    };
    for (auto& [key, value] : data.items()) {
        event[key] = value;
    }
    out << event.dump() << "\n";
    out.flush();
}

} // namespace tile_compile::core
