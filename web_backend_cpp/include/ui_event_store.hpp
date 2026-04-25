#pragma once
#include <string>
#include <vector>
#include <deque>
#include <fstream>
#include <mutex>
#include <filesystem>
#include <optional>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

/// @brief Append-only UI event record used by polling and WebSocket endpoints.
/// @details Events carry a monotonic sequence number, timestamp, source, payload, and optional
/// run/job association so clients can resume from a known sequence.
struct UiEvent {
    int seq{0};
    std::string ts;
    std::string event;
    std::string source;
    nlohmann::json payload{};
    std::optional<std::string> run_id;
    std::optional<std::string> job_id;
};

/// @brief Serializes a UI event to the public JSON event stream format.
nlohmann::json ui_event_to_json(const UiEvent& e);

/// @brief Thread-safe bounded UI event buffer with optional JSONL persistence.
/// @details Keeps recent UI/backend events in memory and mirrors them to runtime storage so
/// browser sessions can reconnect without losing recent state transitions.
class UiEventStore {
public:
    explicit UiEventStore(int max_size = 5000) : _max_size(max_size) {}

    /// @brief Opens the JSONL event log path and loads existing events.
    void configure(const fs::path& path);
    /// @brief Appends an event, assigns sequence/timestamp metadata, and persists it.
    void push(const std::string& event,
              const std::string& source,
              const nlohmann::json& payload = {},
              const std::optional<std::string>& run_id = std::nullopt,
              const std::optional<std::string>& job_id = std::nullopt);
    /// @brief Lists events newer than since_seq, capped by limit.
    std::vector<UiEvent> list(int since_seq = 0, int limit = 100) const;
    /// @brief Returns the newest event sequence number currently known to the store.
    int latest_seq() const;

private:
    void append_jsonl(const UiEvent& e);
    void load_jsonl_locked();

    mutable std::mutex _mutex;
    std::deque<UiEvent> _events;
    int _seq{0};
    int _max_size;
    fs::path _path;
    mutable std::ofstream _log_out; // persistent append stream
};
