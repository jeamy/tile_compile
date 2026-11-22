#pragma once
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
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

/// Epoch seconds of an ISO-8601 UTC timestamp "YYYY-MM-DDTHH:MM:SS[.fff][Z]"
/// (an offset suffix is not applied). nullopt if it does not parse.
inline std::optional<double> parse_iso_utc_seconds(const std::string& raw) {
    int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;
    if (raw.size() < 19 ||
        std::sscanf(raw.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d", &year, &month, &day, &hour, &minute, &second) != 6) {
        return std::nullopt;
    }
    double fractional = 0.0;
    const auto dot = raw.find('.');
    if (dot != std::string::npos) {
        const auto end = raw.find_first_of("Z+-", dot);
        const std::string frac = raw.substr(dot + 1, end == std::string::npos ? std::string::npos : end - dot - 1);
        if (!frac.empty() && frac.size() < 300 &&
            std::all_of(frac.begin(), frac.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
            fractional = std::stod("0." + frac);
        }
    }
    std::tm tm{};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_sec = second;
#ifdef _WIN32
    const auto epoch = _mkgmtime(&tm);
#else
    const auto epoch = timegm(&tm);
#endif
    if (epoch < 0) return std::nullopt;
    return static_cast<double>(epoch) + fractional;
}
