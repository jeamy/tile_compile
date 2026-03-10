#pragma once
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <nlohmann/json.hpp>

struct UiEvent {
    int seq{0};
    std::string event;
    nlohmann::json data{};
};

nlohmann::json ui_event_to_json(const UiEvent& e);

class UiEventStore {
public:
    explicit UiEventStore(int max_size = 200) : _max_size(max_size) {}

    void push(const std::string& event, const nlohmann::json& data = {});
    std::vector<UiEvent> list(int since_seq = 0, int limit = 100) const;
    int latest_seq() const;

private:
    mutable std::mutex _mutex;
    std::deque<UiEvent> _events;
    int _seq{0};
    int _max_size;
};
