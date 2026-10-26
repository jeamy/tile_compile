#pragma once
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace tile_compile::pi {

inline nlohmann::json load_user_reason_catalog(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("User reason catalog not found");
    nlohmann::json catalog;
    input >> catalog;
    if (!catalog.is_object() || catalog.value("schema_version", "") != "pi.user-reason-codes.v1" ||
        !catalog.contains("catalog_version") || !catalog["catalog_version"].is_string() ||
        catalog["catalog_version"].get<std::string>().empty() ||
        !catalog.contains("domains") || !catalog["domains"].is_object())
        throw std::runtime_error("Invalid user reason catalog");
    for (const auto& domain : catalog["domains"].items()) {
        if (!domain.value().is_array()) throw std::runtime_error("Invalid reason domain");
        std::set<std::string> codes;
        for (const auto& entry : domain.value()) {
            if (!entry.is_object() || !entry.contains("code") || !entry["code"].is_string() ||
                !entry.contains("label_key") || !entry["label_key"].is_string() ||
                entry["code"].get<std::string>().empty() || entry["label_key"].get<std::string>().empty() ||
                !codes.insert(entry["code"].get<std::string>()).second)
                throw std::runtime_error("Invalid or duplicate user reason code");
        }
    }
    return catalog;
}

inline bool user_reason_codes_valid(const nlohmann::json& catalog, const std::string& version,
                                    const std::string& domain, const nlohmann::json& codes) {
    if (version != catalog.at("catalog_version").get<std::string>() || !codes.is_array() ||
        !catalog.at("domains").contains(domain)) return false;
    std::set<std::string> allowed, seen;
    for (const auto& entry : catalog.at("domains").at(domain)) allowed.insert(entry.at("code"));
    for (const auto& code : codes) {
        if (!code.is_string()) return false;
        const auto text = code.get<std::string>();
        if (!allowed.count(text) || !seen.insert(text).second) return false;
    }
    return true;
}
} // namespace tile_compile::pi
