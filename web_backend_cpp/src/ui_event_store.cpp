#include "ui_event_store.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

std::string utc_now_iso() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto tt = system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

}

nlohmann::json ui_event_to_json(const UiEvent& e) {
    return {
        {"seq", e.seq},
        {"ts", e.ts},
        {"event", e.event},
        {"source", e.source},
        {"run_id", e.run_id.has_value() ? nlohmann::json(*e.run_id) : nlohmann::json(nullptr)},
        {"job_id", e.job_id.has_value() ? nlohmann::json(*e.job_id) : nlohmann::json(nullptr)},
        {"payload", e.payload},
    };
}

void UiEventStore::configure(const fs::path& path) {
    std::lock_guard<std::mutex> lk(_mutex);
    _path = path;
    _events.clear();
    _seq = 0;
    if (!_path.empty()) {
        std::error_code ec;
        fs::create_directories(_path.parent_path(), ec);
        load_jsonl_locked();
    }
}

void UiEventStore::push(const std::string& event,
                        const std::string& source,
                        const nlohmann::json& payload,
                        const std::optional<std::string>& run_id,
                        const std::optional<std::string>& job_id) {
    std::lock_guard<std::mutex> lk(_mutex);
    UiEvent e;
    e.seq = ++_seq;
    e.ts = utc_now_iso();
    e.event = event;
    e.source = source;
    e.payload = payload.is_object() ? payload : nlohmann::json::object();
    e.run_id = run_id;
    e.job_id = job_id;
    _events.push_back(std::move(e));
    while ((int)_events.size() > _max_size) _events.pop_front();
    append_jsonl(_events.back());
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

void UiEventStore::append_jsonl(const UiEvent& e) const {
    if (_path.empty()) return;
    std::ofstream out(_path, std::ios::app);
    if (!out) return;
    out << ui_event_to_json(e).dump() << '\n';
}

void UiEventStore::load_jsonl_locked() {
    if (_path.empty() || !fs::exists(_path)) return;

    std::ifstream in(_path);
    if (!in) return;

    std::string line;
    while (std::getline(in, line)) {
        auto parsed = nlohmann::json::parse(line, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) continue;

        UiEvent e;
        e.seq = parsed.value("seq", 0);
        if (e.seq <= 0) e.seq = _seq + 1;
        e.ts = parsed.value("ts", std::string());
        e.event = parsed.value("event", std::string());
        e.source = parsed.value("source", std::string());
        if (parsed.contains("payload")) e.payload = parsed["payload"];
        else if (parsed.contains("data")) e.payload = parsed["data"];
        else e.payload = nlohmann::json::object();
        if (parsed.contains("run_id") && parsed["run_id"].is_string()) e.run_id = parsed["run_id"].get<std::string>();
        if (parsed.contains("job_id") && parsed["job_id"].is_string()) e.job_id = parsed["job_id"].get<std::string>();

        _seq = std::max(_seq, e.seq);
        _events.push_back(std::move(e));
        while (static_cast<int>(_events.size()) > _max_size) _events.pop_front();
    }
}
