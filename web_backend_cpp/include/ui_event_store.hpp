#pragma once
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <filesystem>
#include <optional>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

struct UiEvent {
    int seq{0};
    std::string ts;
    std::string event;
    std::string source;
    nlohmann::json payload{};
    std::optional<std::string> run_id;
    std::optional<std::string> job_id;
};

nlohmann::json ui_event_to_json(const UiEvent& e);

class UiEventStore {
public:
    explicit UiEventStore(int max_size = 5000) : _max_size(max_size) {}

    void configure(const fs::path& path);
    void push(const std::string& event,
              const std::string& source,
              const nlohmann::json& payload = {},
              const std::optional<std::string>& run_id = std::nullopt,
              const std::optional<std::string>& job_id = std::nullopt);
    std::vector<UiEvent> list(int since_seq = 0, int limit = 100) const;
    int latest_seq() const;

private:
    void append_jsonl(const UiEvent& e) const;
    void load_jsonl_locked();

    mutable std::mutex _mutex;
    std::deque<UiEvent> _events;
    int _seq{0};
    int _max_size;
    fs::path _path;
};
