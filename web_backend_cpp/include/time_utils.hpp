#pragma once
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>

/// Returns the current UTC time as ISO 8601 string, e.g. "2026-04-07T12:00:00Z".
inline std::string utc_now_iso() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto tt  = system_clock::to_time_t(now);
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
