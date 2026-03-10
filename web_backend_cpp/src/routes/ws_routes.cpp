#include "routes/ws_routes.hpp"
#include "services/run_inspector.hpp"
#include <nlohmann/json.hpp>
#include <thread>
#include <chrono>
#include <atomic>
#include <fstream>
#include <sstream>
#include <optional>
#include <iomanip>

namespace fs = std::filesystem;

namespace {

using json = nlohmann::json;

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
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

double to_pct(const json& raw) {
    for (const auto& key : {"pct", "progress"}) {
        if (!raw.contains(key)) continue;
        try {
            double v = raw.at(key).get<double>();
            if (v <= 1.0) v *= 100.0;
            if (v < 0.0) v = 0.0;
            if (v > 100.0) v = 100.0;
            return std::round(v * 1000.0) / 1000.0;
        } catch (...) {}
    }
    return 0.0;
}

std::string resolve_string(const json& raw, const std::string& a, const std::string& b = "") {
    if (raw.contains(a) && raw.at(a).is_string()) return raw.at(a).get<std::string>();
    if (!b.empty() && raw.contains(b) && raw.at(b).is_string()) return raw.at(b).get<std::string>();
    return "";
}

json normalize_event(const json& raw, const std::string& fallback_run_id) {
    if (!raw.is_object()) {
        return {
            {"type", "log_line"},
            {"run_id", fallback_run_id},
            {"phase", nullptr},
            {"filter", nullptr},
            {"pct", nullptr},
            {"ts", utc_now_iso()},
            {"payload", {{"message", raw.dump()}, {"raw", raw}}}
        };
    }

    std::string type = resolve_string(raw, "type");
    if (type.empty()) type = "log_line";
    std::string run_id = resolve_string(raw, "run_id");
    if (run_id.empty()) run_id = fallback_run_id;
    std::string phase = resolve_string(raw, "phase", "phase_name");
    std::string filter = resolve_string(raw, "filter", "filter_name");

    json payload = raw;
    for (const auto& key : {"type", "run_id", "phase", "phase_name", "filter", "filter_name", "ts", "pct"}) {
        payload.erase(key);
    }

    if (type != "phase_start" && type != "phase_progress" && type != "phase_end" &&
        type != "run_end" && type != "queue_progress" && type != "log_line") {
        return {
            {"type", "log_line"},
            {"run_id", run_id},
            {"phase", phase.empty() ? json(nullptr) : json(phase)},
            {"filter", filter.empty() ? json(nullptr) : json(filter)},
            {"pct", to_pct(raw)},
            {"ts", raw.value("ts", utc_now_iso())},
            {"payload", {{"message", raw.value("message", type)}, {"raw", raw}}}
        };
    }

    json ev = {
        {"type", type},
        {"run_id", run_id},
        {"phase", phase.empty() ? json(nullptr) : json(phase)},
        {"filter", filter.empty() ? json(nullptr) : json(filter)},
        {"pct", raw.contains("pct") || raw.contains("progress") ? json(to_pct(raw)) : json(nullptr)},
        {"ts", raw.value("ts", utc_now_iso())},
        {"payload", payload}
    };
    if (type == "phase_progress") {
        if (raw.contains("current")) ev["current"] = raw["current"];
        if (raw.contains("total")) ev["total"] = raw["total"];
        if (raw.contains("eta_s")) ev["eta_s"] = raw["eta_s"];
    }
    return ev;
}

fs::path find_event_file(const fs::path& run_dir) {
    for (const auto& candidate : {
        run_dir / "logs" / "run_events.jsonl",
        run_dir / "events.jsonl",
        run_dir / "logs" / "events.jsonl"
    }) {
        if (fs::exists(candidate)) return candidate;
    }
    return {};
}

struct RunWsContext {
    std::shared_ptr<AppState> state;
    std::string run_id;
    std::atomic<bool> stop{false};
    size_t cursor{0};
    std::string last_terminal_state;
    std::string last_queue_fingerprint;
};

struct JobWsContext {
    std::shared_ptr<AppState> state;
    std::string job_id;
    std::atomic<bool> stop{false};
};

struct SystemWsContext {
    std::shared_ptr<AppState> state;
    std::atomic<bool> stop{false};
};

std::string last_path_component(const std::string& path) {
    auto pos = path.find_last_of('/');
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

std::shared_ptr<RunWsContext> make_run_ctx(const std::shared_ptr<AppState>& state, const std::string& run_id) {
    auto ctx = std::make_shared<RunWsContext>();
    ctx->state = state;
    ctx->run_id = run_id;
    return ctx;
}

std::shared_ptr<JobWsContext> make_job_ctx(const std::shared_ptr<AppState>& state, const std::string& job_id) {
    auto ctx = std::make_shared<JobWsContext>();
    ctx->state = state;
    ctx->job_id = job_id;
    return ctx;
}

std::shared_ptr<SystemWsContext> make_system_ctx(const std::shared_ptr<AppState>& state) {
    auto ctx = std::make_shared<SystemWsContext>();
    ctx->state = state;
    return ctx;
}

json queue_event_for_run(const AppState& state, const std::string& run_id) {
    for (const auto& job : state.job_store.list(200)) {
        if (job.type != "run_queue") continue;
        if (!job.data.is_object()) continue;
        if (job.data.value("run_id", std::string()) != run_id) continue;
        const auto queue = job.data.value("queue", json::array());
        int total = static_cast<int>(queue.is_array() ? queue.size() : 0);
        int done = 0;
        std::string running_filter;
        if (queue.is_array()) {
            for (const auto& item : queue) {
                if (!item.is_object()) continue;
                std::string s = item.value("state", std::string());
                if (s == "ok") done++;
                if (s == "running" && running_filter.empty()) running_filter = item.value("filter", std::string());
            }
        }
        double pct = total > 0 ? std::round((100.0 * done / total) * 100.0) / 100.0 : 100.0;
        return {
            {"type", "queue_progress"},
            {"run_id", run_id},
            {"phase", nullptr},
            {"filter", running_filter.empty() ? json(nullptr) : json(running_filter)},
            {"pct", pct},
            {"ts", utc_now_iso()},
            {"payload", {
                {"current_index", job.data.value("current_index", -1)},
                {"total", total},
                {"done", done},
                {"queue", queue}
            }}
        };
    }
    return json();
}

void send_json(crow::websocket::connection& conn, const json& j) {
    conn.send_text(j.dump());
}

void stream_run(crow::websocket::connection& conn, const std::shared_ptr<RunWsContext>& ctx) {
    while (!ctx->stop.load()) {
        try {
            const auto run_dir = ctx->state->runtime.resolve_run_dir(ctx->run_id);
            const auto event_file = find_event_file(run_dir);
            if (!event_file.empty()) {
                std::ifstream in(event_file);
                std::string line;
                size_t line_no = 0;
                while (std::getline(in, line) && !ctx->stop.load()) {
                    ++line_no;
                    if (line_no <= ctx->cursor) continue;
                    if (line.empty()) continue;
                    try {
                        send_json(conn, normalize_event(json::parse(line), ctx->run_id));
                    } catch (...) {
                        send_json(conn, {
                            {"type", "log_line"},
                            {"run_id", ctx->run_id},
                            {"phase", nullptr},
                            {"filter", nullptr},
                            {"pct", nullptr},
                            {"ts", utc_now_iso()},
                            {"payload", {{"message", line}, {"raw", line}}}
                        });
                    }
                    ctx->cursor = line_no;
                }
            }

            const auto queue_event = queue_event_for_run(*ctx->state, ctx->run_id);
            if (!queue_event.is_null()) {
                const std::string fingerprint = queue_event.dump();
                if (fingerprint != ctx->last_queue_fingerprint) {
                    send_json(conn, queue_event);
                    ctx->last_queue_fingerprint = fingerprint;
                }
            }

            auto status = read_run_status(run_dir);
            const std::string state = status.value("status", std::string("unknown"));
            send_json(conn, {
                {"type", "run_status"},
                {"run_id", ctx->run_id},
                {"state", state},
                {"phase", status.contains("current_phase") ? status["current_phase"] : json(nullptr)},
                {"pct", to_pct(status)},
                {"ts", utc_now_iso()},
                {"payload", status}
            });

            if ((state == "completed" || state == "failed" || state == "cancelled" || state == "aborted") &&
                ctx->last_terminal_state != state) {
                send_json(conn, {
                    {"type", "run_end"},
                    {"run_id", ctx->run_id},
                    {"status", state == "completed" ? "ok" : "error"},
                    {"ts", utc_now_iso()},
                    {"payload", {
                        {"state", state},
                        {"progress", status.value("progress", 0.0)},
                        {"current_phase", status.contains("current_phase") ? status["current_phase"] : json(nullptr)},
                        {"status", state}
                    }}
                });
                ctx->last_terminal_state = state;
            }
        } catch (const std::exception& e) {
            send_json(conn, {
                {"type", "run_stream_error"},
                {"run_id", ctx->run_id},
                {"ts", utc_now_iso()},
                {"payload", {{"message", e.what()}}}
            });
        }
        for (int i = 0; i < 10 && !ctx->stop.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void stream_job(crow::websocket::connection& conn, const std::shared_ptr<JobWsContext>& ctx) {
    while (!ctx->stop.load()) {
        auto job = ctx->state->job_store.get(ctx->job_id);
        json ev = {
            {"type", "job_progress"},
            {"job_id", ctx->job_id},
            {"state", job ? job_state_str(job->state) : std::string("unknown")},
            {"pid", job && job->pid.has_value() ? json(*job->pid) : json(nullptr)},
            {"exit_code", job && job->data.contains("exit_code") ? job->data["exit_code"] : json(nullptr)},
            {"ts", utc_now_iso()},
            {"data", job ? job->data : json::object()}
        };
        send_json(conn, ev);
        for (int i = 0; i < 10 && !ctx->stop.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void stream_system(crow::websocket::connection& conn, const std::shared_ptr<SystemWsContext>& ctx) {
    while (!ctx->stop.load()) {
        send_json(conn, {
            {"type", "system_heartbeat"},
            {"ts", utc_now_iso()},
            {"status", "ok"},
            {"payload", {
                {"cli", ctx->state->runtime.cli_exe},
                {"runner", ctx->state->runtime.runner_exe}
            }}
        });
        for (int i = 0; i < 50 && !ctx->stop.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

template <typename T>
std::shared_ptr<T> take_ctx(crow::websocket::connection& conn) {
    auto holder = static_cast<std::shared_ptr<T>*>(conn.userdata());
    if (!holder) return {};
    return *holder;
}

template <typename T>
void destroy_ctx(crow::websocket::connection& conn) {
    auto holder = static_cast<std::shared_ptr<T>*>(conn.userdata());
    if (!holder) return;
    (*holder)->stop.store(true);
    delete holder;
    conn.userdata(nullptr);
}

} // namespace

void register_ws_routes(CrowApp& app,
                         std::shared_ptr<AppState> state) {

    CROW_WEBSOCKET_ROUTE(app, "/api/ws/runs/<string>")
    .onopen([](crow::websocket::connection& conn) {
        auto ctx = take_ctx<RunWsContext>(conn);
        if (!ctx) return;
        std::thread([&conn, ctx]() { stream_run(conn, ctx); }).detach();
    })
    .onmessage([](crow::websocket::connection&, const std::string&, bool) {})
    .onclose([](crow::websocket::connection& conn, const std::string&) {
        destroy_ctx<RunWsContext>(conn);
    })
    .onaccept([state](const crow::request& req, void** userdata) -> bool {
        std::string run_id = last_path_component(req.url);
        *userdata = new std::shared_ptr<RunWsContext>(make_run_ctx(state, run_id));
        return !run_id.empty();
    });

    CROW_WEBSOCKET_ROUTE(app, "/api/ws/jobs/<string>")
    .onopen([](crow::websocket::connection& conn) {
        auto ctx = take_ctx<JobWsContext>(conn);
        if (!ctx) return;
        std::thread([&conn, ctx]() { stream_job(conn, ctx); }).detach();
    })
    .onmessage([](crow::websocket::connection&, const std::string&, bool) {})
    .onclose([](crow::websocket::connection& conn, const std::string&) {
        destroy_ctx<JobWsContext>(conn);
    })
    .onaccept([state](const crow::request& req, void** userdata) -> bool {
        std::string job_id = last_path_component(req.url);
        *userdata = new std::shared_ptr<JobWsContext>(make_job_ctx(state, job_id));
        return !job_id.empty();
    });

    CROW_WEBSOCKET_ROUTE(app, "/api/ws/system")
    .onopen([](crow::websocket::connection& conn) {
        auto ctx = take_ctx<SystemWsContext>(conn);
        if (!ctx) return;
        std::thread([&conn, ctx]() { stream_system(conn, ctx); }).detach();
    })
    .onmessage([](crow::websocket::connection&, const std::string&, bool) {})
    .onclose([](crow::websocket::connection& conn, const std::string&) {
        destroy_ctx<SystemWsContext>(conn);
    })
    .onaccept([state](const crow::request&, void** userdata) -> bool {
        *userdata = new std::shared_ptr<SystemWsContext>(make_system_ctx(state));
        return true;
    });
}
