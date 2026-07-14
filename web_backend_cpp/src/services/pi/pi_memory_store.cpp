#include "services/pi/pi_memory_store.hpp"

#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace tile_compile::pi {
namespace {

std::string utc_timestamp_compact() {
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto ticks = seconds.time_since_epoch().count();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_" << ticks;
    return out.str();
}

std::string utc_timestamp_iso() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string string_field(const nlohmann::json& object, const char* key) {
    if (!object.is_object() || !object.contains(key) || !object[key].is_string()) return "";
    return object[key].get<std::string>();
}

nlohmann::json memory_dedupe_signature(const nlohmann::json& memory) {
    nlohmann::json signature = {
        {"type", string_field(memory, "type")},
        {"source", string_field(memory, "source")},
    };
    if (memory.contains("context_signature")) signature["context_signature"] = memory["context_signature"];
    if (memory.contains("scope")) signature["scope"] = memory["scope"];
    if (memory.contains("config_updates")) signature["config_updates"] = memory["config_updates"];
    if (memory.contains("recommendation")) signature["recommendation"] = memory["recommendation"];
    if (memory.contains("avoid")) signature["avoid"] = memory["avoid"];
    return signature;
}

bool allowed_review_status(const std::string& status) {
    return status == "promotable" || status == "accepted" || status == "rejected" || status == "deprecated";
}

void collect_paths(const nlohmann::json& value, std::set<std::string>& paths) {
    if (value.is_string()) {
        const std::string path = value.get<std::string>();
        if (!path.empty()) paths.insert(path);
    } else if (value.is_object()) {
        if (value.contains("path") && value["path"].is_string()) {
            const std::string path = value["path"].get<std::string>();
            if (!path.empty()) paths.insert(path);
        }
        for (auto it = value.begin(); it != value.end(); ++it) collect_paths(it.value(), paths);
    } else if (value.is_array()) {
        for (const auto& item : value) collect_paths(item, paths);
    }
}

nlohmann::json pointer_value(const nlohmann::json& object, const char* pointer) {
    if (!object.is_object()) return nullptr;
    const auto ptr = nlohmann::json::json_pointer(pointer);
    if (!object.contains(ptr)) return nullptr;
    return object.at(ptr);
}

std::string normalized_text(const nlohmann::json& value) {
    if (!value.is_string()) return "";
    std::string out = value.get<std::string>();
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    out.erase(out.begin(), std::find_if(out.begin(), out.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    out.erase(std::find_if(out.rbegin(), out.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), out.end());
    return out;
}

std::set<std::string> normalized_string_set(const nlohmann::json& value) {
    std::set<std::string> out;
    if (value.is_array()) {
        for (const auto& item : value) {
            const std::string text = normalized_text(item);
            if (!text.empty() && text != "unknown") out.insert(text);
        }
    } else {
        const std::string text = normalized_text(value);
        if (!text.empty() && text != "unknown") out.insert(text);
    }
    return out;
}

bool has_known_value(const nlohmann::json& value) {
    if (value.is_null()) return false;
    if (value.is_string()) {
        const std::string text = normalized_text(value);
        return !text.empty() && text != "unknown" && text != "null";
    }
    if (value.is_array()) return !normalized_string_set(value).empty();
    if (value.is_object()) return !value.empty();
    return true;
}

double numeric_value(const nlohmann::json& value, double fallback = 0.0) {
    if (value.is_number()) return value.get<double>();
    if (value.is_string()) {
        try {
            return std::stod(value.get<std::string>());
        } catch (...) {
            return fallback;
        }
    }
    return fallback;
}

bool sensitive_key(const std::string& key) {
    const std::string lowered = normalized_text(key);
    return lowered.find("api_key") != std::string::npos ||
           lowered.find("apikey") != std::string::npos ||
           lowered.find("secret") != std::string::npos ||
           lowered.find("token") != std::string::npos ||
           lowered.find("authorization") != std::string::npos ||
           lowered.find("password") != std::string::npos;
}

bool looks_like_absolute_local_path(const std::string& value) {
    if (value.size() >= 2 && value[0] == '/') return true;
    if (value.size() >= 3 && std::isalpha(static_cast<unsigned char>(value[0])) && value[1] == ':' &&
        (value[2] == '\\' || value[2] == '/')) {
        return true;
    }
    return false;
}

nlohmann::json sanitize_memory_privacy(const nlohmann::json& value, const std::string& key_hint = "") {
    if (sensitive_key(key_hint)) return "<redacted>";
    if (value.is_string()) {
        const std::string text = value.get<std::string>();
        if (looks_like_absolute_local_path(text)) {
            return {
                {"redacted", "absolute_path"},
                {"name", std::filesystem::path(text).filename().string()}
            };
        }
        return value;
    }
    if (value.is_array()) {
        nlohmann::json out = nlohmann::json::array();
        for (const auto& item : value) out.push_back(sanitize_memory_privacy(item, key_hint));
        return out;
    }
    if (value.is_object()) {
        nlohmann::json out = nlohmann::json::object();
        for (auto it = value.begin(); it != value.end(); ++it) {
            out[it.key()] = sanitize_memory_privacy(it.value(), it.key());
        }
        return out;
    }
    return value;
}

void add_match_detail(nlohmann::json& details,
                      const std::string& field,
                      const nlohmann::json& memory_value,
                      const nlohmann::json& query_value,
                      int score) {
    details.push_back({
        {"field", field},
        {"memory", memory_value},
        {"query", query_value},
        {"score", score}
    });
}

int context_match_score(const nlohmann::json& memory,
                        const nlohmann::json& query,
                        nlohmann::json& details,
                        nlohmann::json& coverage) {
    details = nlohmann::json::array();
    std::vector<std::string> missing_query_fields;
    int score = 0;
    int compared = 0;
    int matched = 0;

    const nlohmann::json memory_ctx = memory.contains("context_signature") && memory["context_signature"].is_object()
        ? memory["context_signature"]
        : nlohmann::json::object();
    const nlohmann::json query_ctx = query.contains("context_signature") && query["context_signature"].is_object()
        ? query["context_signature"]
        : nlohmann::json::object();

    auto compare_text = [&](const char* pointer, const char* field, int points) {
        const nlohmann::json memory_value = pointer_value(memory_ctx, pointer);
        const nlohmann::json query_value = pointer_value(query_ctx, pointer);
        if (!has_known_value(query_value)) {
            missing_query_fields.emplace_back(field);
            return;
        }
        ++compared;
        const std::string memory_text = normalized_text(memory_value);
        const std::string query_text = normalized_text(query_value);
        if (!memory_text.empty() && memory_text == query_text) {
            score += points;
            ++matched;
            add_match_detail(details, field, memory_value, query_value, points);
        }
    };

    auto compare_bool = [&](const char* pointer, const char* field, int points) {
        const nlohmann::json memory_value = pointer_value(memory_ctx, pointer);
        const nlohmann::json query_value = pointer_value(query_ctx, pointer);
        if (!query_value.is_boolean()) {
            missing_query_fields.emplace_back(field);
            return;
        }
        ++compared;
        if (memory_value.is_boolean() && memory_value.get<bool>() == query_value.get<bool>()) {
            score += points;
            ++matched;
            add_match_detail(details, field, memory_value, query_value, points);
        }
    };

    auto compare_number_close = [&](const char* pointer, const char* field, int close_points, int near_points, double close_ratio, double near_ratio) {
        const nlohmann::json memory_value = pointer_value(memory_ctx, pointer);
        const nlohmann::json query_value = pointer_value(query_ctx, pointer);
        if (!has_known_value(query_value)) {
            missing_query_fields.emplace_back(field);
            return;
        }
        const double memory_number = numeric_value(memory_value, 0.0);
        const double query_number = numeric_value(query_value, 0.0);
        if (memory_number <= 0.0 || query_number <= 0.0) {
            ++compared;
            return;
        }
        ++compared;
        const double ratio = std::abs(memory_number - query_number) / std::max(memory_number, query_number);
        if (ratio <= close_ratio) {
            score += close_points;
            ++matched;
            add_match_detail(details, field, memory_value, query_value, close_points);
        } else if (ratio <= near_ratio) {
            score += near_points;
            ++matched;
            add_match_detail(details, field, memory_value, query_value, near_points);
        }
    };

    auto compare_set_overlap = [&](const char* pointer, const char* field, int points_per_hit, int max_points) {
        const nlohmann::json memory_value = pointer_value(memory_ctx, pointer);
        const nlohmann::json query_value = pointer_value(query_ctx, pointer);
        const auto memory_set = normalized_string_set(memory_value);
        const auto query_set = normalized_string_set(query_value);
        if (query_set.empty()) {
            missing_query_fields.emplace_back(field);
            return;
        }
        ++compared;
        int hits = 0;
        for (const auto& value : query_set) {
            if (memory_set.count(value)) ++hits;
        }
        if (hits > 0) {
            const int points = std::min(max_points, hits * points_per_hit);
            score += points;
            ++matched;
            add_match_detail(details, field, memory_value, query_value, points);
        }
    };

    compare_text("/target/object_name", "target.object_name", 4);
    compare_text("/target/object_type", "target.object_type", 4);
    compare_text("/target/angular_size_class", "target.angular_size_class", 2);
    compare_bool("/target/has_extended_emission", "target.has_extended_emission", 2);
    compare_text("/acquisition/camera_name", "acquisition.camera_name", 2);
    compare_text("/acquisition/camera_type", "acquisition.camera_type", 3);
    compare_text("/acquisition/color_mode", "acquisition.color_mode", 2);
    compare_set_overlap("/acquisition/filters", "acquisition.filters", 2, 4);
    compare_number_close("/acquisition/frame_count", "acquisition.frame_count", 2, 1, 0.25, 0.60);
    compare_text("/optics/telescope", "optics.telescope", 2);
    compare_number_close("/optics/focal_length_mm", "optics.focal_length_mm", 2, 1, 0.15, 0.35);
    compare_number_close("/optics/f_ratio", "optics.f_ratio", 1, 1, 0.15, 0.30);
    compare_text("/mount/type", "mount.type", 2);
    compare_text("/mount/tracking_quality", "mount.tracking_quality", 1);
    compare_set_overlap("/pipeline/affected_paths", "pipeline.affected_paths", 2, 8);
    compare_set_overlap("/pipeline/phases", "pipeline.phases", 2, 6);
    compare_set_overlap("/problem/classes", "problem.classes", 3, 9);
    compare_set_overlap("/problem/hints", "problem.hints", 2, 6);

    coverage = {
        {"matched_fields", matched},
        {"compared_fields", compared},
        {"missing_query_fields", missing_query_fields}
    };
    return score;
}

nlohmann::json read_jsonl(const std::filesystem::path& path) {
    nlohmann::json items = nlohmann::json::array();
    std::ifstream in(path);
    if (!in) return items;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto parsed = nlohmann::json::parse(line, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) continue;
        items.push_back(std::move(parsed));
    }
    return items;
}

void write_jsonl(const std::filesystem::path& path, const nlohmann::json& items) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("failed to open PI memory store for writing");
    if (items.is_array()) {
        for (const auto& item : items) out << item.dump() << '\n';
    }
    if (!out) throw std::runtime_error("failed to write PI memory store");
}

void add_index_ref(nlohmann::json& index,
                   const std::string& bucket,
                   const std::string& key,
                   const std::string& memory_id) {
    if (bucket.empty() || key.empty() || memory_id.empty()) return;
    if (!index.contains(bucket) || !index[bucket].is_object()) index[bucket] = nlohmann::json::object();
    if (!index[bucket].contains(key) || !index[bucket][key].is_array()) index[bucket][key] = nlohmann::json::array();
    auto& arr = index[bucket][key];
    for (const auto& existing : arr) {
        if (existing.is_string() && existing.get<std::string>() == memory_id) return;
    }
    arr.push_back(memory_id);
}

void add_index_json_value(nlohmann::json& index,
                          const std::string& bucket,
                          const nlohmann::json& value,
                          const std::string& memory_id) {
    if (value.is_string()) {
        const std::string text = normalized_text(value);
        if (!text.empty() && text != "unknown") add_index_ref(index, bucket, text, memory_id);
    } else if (value.is_array()) {
        for (const auto& item : value) add_index_json_value(index, bucket, item, memory_id);
    }
}

void add_index_paths(nlohmann::json& index,
                     const nlohmann::json& memory,
                     const std::string& memory_id) {
    std::set<std::string> paths;
    if (memory.contains("config_updates")) collect_paths(memory["config_updates"], paths);
    if (memory.contains("scope")) collect_paths(memory["scope"], paths);
    if (memory.contains("context_signature") && memory["context_signature"].is_object()) {
        const auto affected = pointer_value(memory["context_signature"], "/pipeline/affected_paths");
        collect_paths(affected, paths);
    }
    for (const auto& path : paths) add_index_ref(index, "by_path", path, memory_id);
}

} // namespace

PiMemoryStore::PiMemoryStore(std::filesystem::path memory_dir)
    : _memory_dir(std::move(memory_dir)) {}

std::filesystem::path PiMemoryStore::memories_path() const {
    return _memory_dir / "memories_v2.jsonl";
}

std::filesystem::path PiMemoryStore::reviews_path() const {
    return _memory_dir / "memory_reviews_v2.jsonl";
}

std::filesystem::path PiMemoryStore::indices_path() const {
    return _memory_dir / "memory_indices_v2.json";
}

std::filesystem::path PiMemoryStore::legacy_memories_path() const {
    return _memory_dir / "memories.jsonl";
}

std::filesystem::path PiMemoryStore::legacy_reviews_path() const {
    return _memory_dir / "memory_reviews.jsonl";
}

nlohmann::json PiMemoryStore::append_candidate(nlohmann::json memory) const {
    if (!memory.is_object()) {
        throw std::invalid_argument("PI memory must be a JSON object");
    }
    memory = sanitize_memory_privacy(memory);
    if (memory.contains("schema_version") && string_field(memory, "schema_version") != kMemorySchemaVersion) {
        throw std::invalid_argument("PI memory schema_version must be pi.memory.v2");
    }

    memory["schema_version"] = kMemorySchemaVersion;
    if (string_field(memory, "memory_id").empty()) {
        memory["memory_id"] = "mem_" + utc_timestamp_compact();
    }
    if (string_field(memory, "id").empty()) {
        memory["id"] = memory["memory_id"];
    }
    if (string_field(memory, "created_at").empty()) {
        memory["created_at"] = utc_timestamp_iso();
    }
    if (string_field(memory, "status").empty()) {
        memory["status"] = "candidate";
    }
    if (string_field(memory, "privacy_class").empty()) {
        memory["privacy_class"] = "metadata_only";
    }

    if (string_field(memory, "status") != "candidate") {
        throw std::invalid_argument("append_candidate only accepts candidate memories");
    }
    if (string_field(memory, "type").empty()) {
        throw std::invalid_argument("PI memory type is required");
    }
    if (!memory.contains("context_signature") || !memory["context_signature"].is_object()) {
        throw std::invalid_argument("PI memory context_signature is required");
    }
    if (!memory.contains("scope") || !memory["scope"].is_object()) {
        throw std::invalid_argument("PI memory scope is required");
    }
    if (!memory.contains("evidence") || !memory["evidence"].is_object()) {
        throw std::invalid_argument("PI memory evidence is required");
    }
    if (!memory.contains("outcome") || !memory["outcome"].is_object()) {
        throw std::invalid_argument("PI memory outcome is required");
    }
    if (!memory.contains("review") || !memory["review"].is_object()) {
        memory["review"] = {
            {"status", "candidate"},
            {"reviewed_by", nullptr},
            {"reviewed_at", nullptr},
            {"notes", ""}
        };
    }
    if (!memory.contains("retrieval") || !memory["retrieval"].is_object()) {
        memory["retrieval"] = nlohmann::json::object();
    }

    const nlohmann::json new_signature = memory_dedupe_signature(memory);
    for (auto existing : list(100000)) {
        if (memory_dedupe_signature(existing) == new_signature) {
            existing["created"] = false;
            existing["duplicate"] = true;
            existing["duplicate_of"] = existing.value("memory_id", std::string());
            return existing;
        }
    }

    std::error_code ec;
    std::filesystem::create_directories(_memory_dir, ec);
    if (ec) {
        throw std::runtime_error("failed to create PI memory directory: " + ec.message());
    }

    std::ofstream out(memories_path(), std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to open PI memory store");
    }
    out << memory.dump() << '\n';
    if (!out) {
        throw std::runtime_error("failed to write PI memory");
    }
    rebuild_indices();
    memory["created"] = true;
    return memory;
}

nlohmann::json PiMemoryStore::list(int limit) const {
    if (limit <= 0) return nlohmann::json::array();
    nlohmann::json items = read_jsonl(memories_path());

    std::map<std::string, nlohmann::json> latest_reviews;
    for (const auto& review : read_jsonl(reviews_path())) {
        const std::string memory_id = string_field(review, "memory_id");
        if (!memory_id.empty()) latest_reviews[memory_id] = review;
    }

    for (auto& item : items) {
        const std::string memory_id = string_field(item, "memory_id");
        auto it = latest_reviews.find(memory_id);
        if (it == latest_reviews.end()) continue;
        item["status"] = it->second.value("status", item.value("status", std::string("candidate")));
        item["review"] = it->second;
        if (it->second.contains("scope") && it->second["scope"].is_object()) {
            item["scope"] = it->second["scope"];
        }
        if (it->second.contains("outcome") && it->second["outcome"].is_object()) {
            item["outcome"] = it->second["outcome"];
        }
    }

    while (static_cast<int>(items.size()) > limit) items.erase(items.begin());
    return items;
}

nlohmann::json PiMemoryStore::review(const std::string& memory_id,
                                     const std::string& status,
                                     const std::string& reviewer,
                                     const std::string& note,
                                     const nlohmann::json& outcome,
                                     const nlohmann::json& scope) const {
    if (memory_id.empty()) throw std::invalid_argument("memory_id is required");
    if (!allowed_review_status(status)) throw std::invalid_argument("unsupported memory review status");

    bool found = false;
    for (const auto& item : list(100000)) {
        if (item.value("memory_id", std::string()) == memory_id) {
            found = true;
            break;
        }
    }
    if (!found) throw std::invalid_argument("memory_id not found");

    nlohmann::json review_event = {
        {"schema_version", kMemorySchemaVersion},
        {"memory_id", memory_id},
        {"id", memory_id},
        {"status", status},
        {"reviewed_at", utc_timestamp_iso()},
        {"reviewer", reviewer.empty() ? "user" : reviewer},
        {"note", note}
    };
    if (outcome.is_object() && !outcome.empty()) {
        review_event["outcome"] = sanitize_memory_privacy(outcome);
    }
    if (scope.is_object() && !scope.empty()) {
        review_event["scope"] = sanitize_memory_privacy(scope);
    }

    std::error_code ec;
    std::filesystem::create_directories(_memory_dir, ec);
    if (ec) throw std::runtime_error("failed to create PI memory directory: " + ec.message());

    std::ofstream out(reviews_path(), std::ios::app);
    if (!out) throw std::runtime_error("failed to open PI memory review store");
    out << review_event.dump() << '\n';
    if (!out) throw std::runtime_error("failed to write PI memory review");
    rebuild_indices();
    return review_event;
}

nlohmann::json PiMemoryStore::indices() const {
    std::ifstream in(indices_path());
    if (in) {
        auto parsed = nlohmann::json::parse(in, nullptr, false);
        if (!parsed.is_discarded() && parsed.is_object() &&
            parsed.value("schema_version", std::string()) == "pi.memory-indices.v2") {
            return parsed;
        }
    }
    return rebuild_indices();
}

nlohmann::json PiMemoryStore::rebuild_indices() const {
    nlohmann::json index = {
        {"schema_version", "pi.memory-indices.v2"},
        {"memory_count", 0},
        {"by_type", nlohmann::json::object()},
        {"by_status", nlohmann::json::object()},
        {"by_path", nlohmann::json::object()},
        {"by_target", nlohmann::json::object()},
        {"by_camera", nlohmann::json::object()},
        {"by_filter", nlohmann::json::object()},
        {"by_problem", nlohmann::json::object()}
    };

    for (const auto& memory : list(100000)) {
        const std::string memory_id = memory.value("memory_id", std::string());
        if (memory_id.empty()) continue;
        index["memory_count"] = index.value("memory_count", 0) + 1;
        add_index_ref(index, "by_type", normalized_text(memory.value("type", std::string())), memory_id);
        add_index_ref(index, "by_status", normalized_text(memory.value("status", std::string())), memory_id);
        add_index_paths(index, memory, memory_id);
        const nlohmann::json ctx = memory.contains("context_signature") && memory["context_signature"].is_object()
            ? memory["context_signature"]
            : nlohmann::json::object();
        add_index_json_value(index, "by_target", pointer_value(ctx, "/target/object_name"), memory_id);
        add_index_json_value(index, "by_target", pointer_value(ctx, "/target/object_type"), memory_id);
        add_index_json_value(index, "by_camera", pointer_value(ctx, "/acquisition/camera_name"), memory_id);
        add_index_json_value(index, "by_camera", pointer_value(ctx, "/acquisition/camera_type"), memory_id);
        add_index_json_value(index, "by_filter", pointer_value(ctx, "/acquisition/filters"), memory_id);
        add_index_json_value(index, "by_problem", pointer_value(ctx, "/problem/classes"), memory_id);
        add_index_json_value(index, "by_problem", pointer_value(ctx, "/problem/hints"), memory_id);
    }

    std::error_code ec;
    std::filesystem::create_directories(_memory_dir, ec);
    if (ec) throw std::runtime_error("failed to create PI memory directory: " + ec.message());
    std::ofstream out(indices_path());
    if (!out) throw std::runtime_error("failed to open PI memory index store");
    out << index.dump(2) << '\n';
    if (!out) throw std::runtime_error("failed to write PI memory indices");
    return index;
}

nlohmann::json PiMemoryStore::retrieve(const nlohmann::json& query, int limit) const {
    nlohmann::json matches = nlohmann::json::array();
    if (limit <= 0) return matches;

    const std::string wanted_type = string_field(query, "type");
    std::set<std::string> wanted_paths;
    if (query.contains("paths")) collect_paths(query["paths"], wanted_paths);
    if (query.contains("config_updates")) collect_paths(query["config_updates"], wanted_paths);

    for (const auto& item : list(100000)) {
        const std::string status = item.value("status", std::string());
        if (status != "accepted") continue;
        int score = 0;
        if (!wanted_type.empty() && item.value("type", std::string()) != wanted_type) continue;
        if (!wanted_type.empty()) score += 3;
        if (status == "accepted") score += 2;
        if (item.contains("outcome") && item["outcome"].is_object()) {
            if (item["outcome"].value("validation_valid", false)) score += 2;
        }

        std::set<std::string> memory_paths;
        collect_paths(item, memory_paths);
        int path_score = 0;
        for (const auto& path : wanted_paths) {
            if (memory_paths.count(path)) path_score += 2;
        }
        score += path_score;

        nlohmann::json match_details = nlohmann::json::array();
        nlohmann::json coverage = nlohmann::json::object();
        const int ctx_score = context_match_score(item, query, match_details, coverage);
        score += ctx_score;

        const bool has_context_query = query.contains("context_signature") && query["context_signature"].is_object();
        if (has_context_query && ctx_score <= 0 && path_score <= 0) continue;
        if (has_context_query && path_score <= 0 && ctx_score < 5) continue;
        if (has_context_query && score < 7) continue;
        if (score <= 0 && !wanted_type.empty()) continue;
        if (score <= 0 && !wanted_paths.empty()) continue;
        if (score <= 0) score = 1;

        nlohmann::json match = item;
        match["retrieval_score"] = score;
        match["context_match_score"] = ctx_score;
        match["path_match_score"] = path_score;
        match["match_explanation"] = match_details;
        match["match_coverage"] = coverage;
        matches.push_back(std::move(match));
    }

    std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
        return a.value("retrieval_score", 0) > b.value("retrieval_score", 0);
    });
    while (static_cast<int>(matches.size()) > limit) matches.erase(matches.end() - 1);
    return matches;
}

nlohmann::json negative_matches_for_query(const nlohmann::json& items,
                                          const nlohmann::json& query,
                                          int limit) {
    nlohmann::json matches = nlohmann::json::array();
    if (limit <= 0) return matches;
    const std::string wanted_type = string_field(query, "type");
    std::set<std::string> wanted_paths;
    if (query.contains("paths")) collect_paths(query["paths"], wanted_paths);
    if (query.contains("config_updates")) collect_paths(query["config_updates"], wanted_paths);
    for (const auto& item : items) {
        const std::string status = item.value("status", std::string());
        if (status != "rejected" && status != "deprecated") continue;
        if (!wanted_type.empty() && item.value("type", std::string()) != wanted_type) continue;
        int score = !wanted_type.empty() ? 3 : 0;
        std::set<std::string> memory_paths;
        collect_paths(item, memory_paths);
        int path_score = 0;
        for (const auto& path : wanted_paths) {
            if (memory_paths.count(path)) path_score += 2;
        }
        score += path_score;
        nlohmann::json match_details = nlohmann::json::array();
        nlohmann::json coverage = nlohmann::json::object();
        const int ctx_score = context_match_score(item, query, match_details, coverage);
        score += ctx_score;
        const bool has_context_query = query.contains("context_signature") && query["context_signature"].is_object();
        if (has_context_query && ctx_score <= 0 && path_score <= 0) continue;
        if (has_context_query && path_score <= 0 && ctx_score < 5) continue;
        if (has_context_query && score < 7) continue;
        if (score <= 0 && (!wanted_type.empty() || !wanted_paths.empty())) continue;
        nlohmann::json match = item;
        match["retrieval_score"] = std::max(score, 1);
        match["context_match_score"] = ctx_score;
        match["path_match_score"] = path_score;
        match["match_explanation"] = match_details;
        match["match_coverage"] = coverage;
        match["retrieval_warning"] = status == "rejected"
            ? "similar_memory_was_rejected"
            : "similar_memory_was_deprecated";
        matches.push_back(std::move(match));
    }
    std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
        return a.value("retrieval_score", 0) > b.value("retrieval_score", 0);
    });
    while (static_cast<int>(matches.size()) > limit) matches.erase(matches.end() - 1);
    return matches;
}

nlohmann::json PiMemoryStore::retrieve_negative(const nlohmann::json& query, int limit) const {
    return negative_matches_for_query(list(100000), query, limit);
}

nlohmann::json PiMemoryStore::export_bundle(const std::string& privacy_class,
                                            bool include_reviews) const {
    nlohmann::json memories = nlohmann::json::array();
    for (const auto& memory : read_jsonl(memories_path())) {
        if (!privacy_class.empty() && privacy_class != "all" &&
            memory.value("privacy_class", std::string("metadata_only")) != privacy_class) {
            continue;
        }
        memories.push_back(memory);
    }

    nlohmann::json reviews = nlohmann::json::array();
    if (include_reviews) {
        std::set<std::string> exported_ids;
        for (const auto& memory : memories) {
            const std::string memory_id = memory.value("memory_id", std::string());
            if (!memory_id.empty()) exported_ids.insert(memory_id);
        }
        for (const auto& review : read_jsonl(reviews_path())) {
            const std::string memory_id = review.value("memory_id", std::string());
            if (exported_ids.count(memory_id)) reviews.push_back(review);
        }
    }

    return {
        {"schema_version", kMemoryExportSchemaVersion},
        {"exported_at", utc_timestamp_iso()},
        {"privacy_class", privacy_class.empty() ? "all" : privacy_class},
        {"memories", memories},
        {"reviews", reviews},
        {"memory_count", memories.size()},
        {"review_count", reviews.size()}
    };
}

nlohmann::json PiMemoryStore::import_bundle(const nlohmann::json& bundle,
                                            bool dry_run) const {
    if (!bundle.is_object() || bundle.value("schema_version", std::string()) != kMemoryExportSchemaVersion) {
        throw std::invalid_argument("PI memory import bundle must have schema_version pi.memories-export.v2");
    }

    std::set<std::string> existing_ids;
    std::set<std::string> existing_signatures;
    for (const auto& memory : read_jsonl(memories_path())) {
        const std::string memory_id = memory.value("memory_id", std::string());
        if (!memory_id.empty()) existing_ids.insert(memory_id);
        existing_signatures.insert(memory_dedupe_signature(memory).dump());
    }

    nlohmann::json memories_to_add = nlohmann::json::array();
    nlohmann::json reviews_to_add = nlohmann::json::array();
    int skipped = 0;

    if (bundle.contains("memories") && bundle["memories"].is_array()) {
        for (auto memory : bundle["memories"]) {
            memory = sanitize_memory_privacy(memory);
            if (!memory.is_object()) {
                ++skipped;
                continue;
            }
            if (memory.contains("schema_version") && memory.value("schema_version", std::string()) != kMemorySchemaVersion) {
                ++skipped;
                continue;
            }
            memory["schema_version"] = kMemorySchemaVersion;
            if (string_field(memory, "type").empty()) {
                ++skipped;
                continue;
            }
            if (string_field(memory, "memory_id").empty()) {
                memory["memory_id"] = "mem_import_" + utc_timestamp_compact() + "_" + std::to_string(memories_to_add.size());
            }
            if (string_field(memory, "created_at").empty()) memory["created_at"] = utc_timestamp_iso();
            if (string_field(memory, "status").empty()) memory["status"] = "candidate";
            if (string_field(memory, "privacy_class").empty()) memory["privacy_class"] = "metadata_only";
            if (!memory.contains("context_signature") || !memory["context_signature"].is_object() ||
                !memory.contains("scope") || !memory["scope"].is_object() ||
                !memory.contains("evidence") || !memory["evidence"].is_object() ||
                !memory.contains("outcome") || !memory["outcome"].is_object()) {
                ++skipped;
                continue;
            }

            const std::string memory_id = memory.value("memory_id", std::string());
            const std::string signature = memory_dedupe_signature(memory).dump();
            if (existing_ids.count(memory_id) || existing_signatures.count(signature)) {
                ++skipped;
                continue;
            }
            existing_ids.insert(memory_id);
            existing_signatures.insert(signature);
            memories_to_add.push_back(std::move(memory));
        }
    }

    if (bundle.contains("reviews") && bundle["reviews"].is_array()) {
        for (const auto& review : bundle["reviews"]) {
            if (!review.is_object()) continue;
            const std::string memory_id = review.value("memory_id", std::string());
            if (memory_id.empty() || !existing_ids.count(memory_id)) continue;
            const std::string status = review.value("status", std::string());
            if (!allowed_review_status(status)) continue;
            reviews_to_add.push_back(review);
        }
    }

    if (!dry_run) {
        std::error_code ec;
        std::filesystem::create_directories(_memory_dir, ec);
        if (ec) throw std::runtime_error("failed to create PI memory directory: " + ec.message());
        {
            std::ofstream out(memories_path(), std::ios::app);
            if (!out) throw std::runtime_error("failed to open PI memory import target");
            for (const auto& memory : memories_to_add) out << memory.dump() << '\n';
        }
        if (!reviews_to_add.empty()) {
            std::ofstream out(reviews_path(), std::ios::app);
            if (!out) throw std::runtime_error("failed to open PI memory review import target");
            for (const auto& review : reviews_to_add) out << review.dump() << '\n';
        }
        rebuild_indices();
    }

    return {
        {"ok", true},
        {"dry_run", dry_run},
        {"imported_memories", memories_to_add.size()},
        {"imported_reviews", reviews_to_add.size()},
        {"skipped", skipped}
    };
}

nlohmann::json PiMemoryStore::dedupe(bool dry_run) const {
    nlohmann::json items = read_jsonl(memories_path());
    nlohmann::json unique = nlohmann::json::array();
    std::set<std::string> signatures;
    int removed = 0;
    for (const auto& item : items) {
        const std::string signature = memory_dedupe_signature(item).dump();
        if (!signatures.insert(signature).second) {
            ++removed;
            continue;
        }
        unique.push_back(item);
    }

    std::string backup_path;
    if (!dry_run && removed > 0) {
        std::error_code ec;
        std::filesystem::create_directories(_memory_dir, ec);
        if (ec) throw std::runtime_error("failed to create PI memory directory: " + ec.message());
        const auto backup = _memory_dir / ("memories.dedupe." + utc_timestamp_compact() + ".bak.jsonl");
        if (std::filesystem::exists(memories_path())) {
            std::filesystem::copy_file(memories_path(), backup, std::filesystem::copy_options::overwrite_existing, ec);
            if (!ec) backup_path = backup.string();
        }
        write_jsonl(memories_path(), unique);
        rebuild_indices();
    }

    return {
        {"ok", true},
        {"dry_run", dry_run},
        {"before_count", items.size()},
        {"after_count", unique.size()},
        {"removed_count", removed},
        {"backup_path", backup_path}
    };
}

} // namespace tile_compile::pi
