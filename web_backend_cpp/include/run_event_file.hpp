#pragma once
#include <filesystem>
#include <optional>
#include <system_error>

/// The run's event log: logs/run_events.jsonl, else the legacy events.jsonl
/// or logs/events.jsonl.
inline std::optional<std::filesystem::path> find_run_event_file(const std::filesystem::path& run_dir) {
    for (const auto& candidate : {run_dir / "logs" / "run_events.jsonl", run_dir / "events.jsonl",
                                  run_dir / "logs" / "events.jsonl"}) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(candidate, ec)) return candidate;
    }
    return std::nullopt;
}
