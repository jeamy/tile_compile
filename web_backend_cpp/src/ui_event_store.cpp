#include "ui_event_store.hpp"

nlohmann::json ui_event_to_json(const UiEvent& e) {
    return {{"seq", e.seq}, {"event", e.event}, {"data", e.data}};
}

void UiEventStore::push(const std::string& event, const nlohmann::json& data) {
    std::lock_guard<std::mutex> lk(_mutex);
    UiEvent e;
    e.seq   = ++_seq;
    e.event = event;
    e.data  = data;
    _events.push_back(std::move(e));
    while ((int)_events.size() > _max_size) _events.pop_front();
}

std::vector<UiEvent> UiEventStore::list(int since_seq, int limit) const {
    std::lock_guard<std::mutex> lk(_mutex);
    std::vector<UiEvent> result;
    for (auto& e : _events) {
        if (e.seq > since_seq) result.push_back(e);
        if ((int)result.size() >= limit) break;
    }
    return result;
}

int UiEventStore::latest_seq() const {
    std::lock_guard<std::mutex> lk(_mutex);
    return _seq;
}
