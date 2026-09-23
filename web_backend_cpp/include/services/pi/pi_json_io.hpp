#pragma once
// Small JSON file helpers for the PI decision store (atomic replace, tolerant read).
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>

namespace tile_compile::pi {

inline std::optional<nlohmann::json> read_json_file_opt(const std::filesystem::path& p) {
    std::ifstream in(p);
    if (!in) return std::nullopt;
    nlohmann::json j = nlohmann::json::parse(in, nullptr, false);
    if (j.is_discarded()) return std::nullopt;
    return j;
}

inline void write_json_file_atomic(const std::filesystem::path& p, const nlohmann::json& j) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    const std::filesystem::path tmp = p.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write " + tmp.string());
        out << j.dump(2);
        out.flush();
        if (!out) throw std::runtime_error("write failed " + tmp.string());
    }
    std::filesystem::rename(tmp, p);
}

inline void append_jsonl(const std::filesystem::path& p, const nlohmann::json& j) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    std::ofstream out(p, std::ios::out | std::ios::app);
    if (!out) throw std::runtime_error("cannot append " + p.string());
    out << j.dump() << "\n";
}

} // namespace tile_compile::pi
