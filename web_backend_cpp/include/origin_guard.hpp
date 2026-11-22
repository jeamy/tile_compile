#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>

/// @brief Cross-site request guard.
/// @details The backend has no login: whoever can reach the port may start
/// runs, write files and open paths. CORS headers alone do not stop a page on
/// another site from sending such requests (a text/plain POST needs no
/// preflight, and WebSockets ignore CORS), so every request that carries an
/// Origin header is checked here. Allowed are
///   - requests without Origin (curl, the sidecar, same-origin GETs),
///   - an Origin whose host:port is the request's Host (the UI served by
///     this backend, also over the LAN or in Docker),
///   - a loopback Origin on any port (the UI served by a local dev server and
///     pointed at :8080, see web_frontend_v3/js/api/client.js),
///   - the origins listed in TILE_COMPILE_ALLOWED_ORIGINS (comma separated,
///     e.g. "http://nas.local:8080").
namespace origin_guard {

inline std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

inline std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t");
    if (first == std::string::npos) return "";
    const auto last = value.find_last_not_of(" \t/");
    return value.substr(first, last - first + 1);
}

/// @brief "host[:port]" of an http(s) origin, lowercased ("" if not http(s)).
inline std::string origin_authority(const std::string& origin) {
    const std::string o = lower(trim(origin));
    std::string rest;
    if (o.rfind("http://", 0) == 0) rest = o.substr(7);
    else if (o.rfind("https://", 0) == 0) rest = o.substr(8);
    else return "";
    const auto slash = rest.find('/');
    if (slash != std::string::npos) rest = rest.substr(0, slash);
    if (rest.find('@') != std::string::npos) return "";
    return rest;
}

/// @brief Host part of "host[:port]" or "[v6]:port", without brackets.
inline std::string authority_host(const std::string& authority) {
    if (!authority.empty() && authority.front() == '[') {
        const auto close = authority.find(']');
        return close == std::string::npos ? "" : authority.substr(1, close - 1);
    }
    const auto colon = authority.find(':');
    return colon == std::string::npos ? authority : authority.substr(0, colon);
}

inline bool is_loopback_host(const std::string& host) {
    if (host == "localhost" || host == "::1") return true;
    if (host.size() > 10 && host.compare(host.size() - 10, 10, ".localhost") == 0) return true;
    // 127.0.0.0/8, digits and dots only.
    if (host.rfind("127.", 0) != 0) return false;
    int dots = 0;
    for (char ch : host) {
        if (ch == '.') ++dots;
        else if (!std::isdigit(static_cast<unsigned char>(ch))) return false;
    }
    return dots == 3;
}

/// @param origin the request's Origin header ("" if absent)
/// @param host the request's Host header
/// @param extra comma-separated allowed origins
inline bool origin_allowed(const std::string& origin, const std::string& host,
                           const std::string& extra = "") {
    if (trim(origin).empty()) return true;
    const std::string authority = origin_authority(origin);
    if (authority.empty()) return false;  // "null", file://, other schemes
    if (!host.empty() && authority == lower(trim(host))) return true;
    if (is_loopback_host(authority_host(authority))) return true;
    std::stringstream list(extra);
    std::string item;
    while (std::getline(list, item, ',')) {
        if (!trim(item).empty() && lower(trim(item)) == lower(trim(origin))) return true;
    }
    return false;
}

inline std::string configured_extra_origins() {
    const char* value = std::getenv("TILE_COMPILE_ALLOWED_ORIGINS");
    return value ? std::string(value) : std::string();
}

}  // namespace origin_guard
