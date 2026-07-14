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
    if (memory.contains("config_updates")) signature["config_updates"] = memory["config_updates"];
    if (memory.contains("recommendation")) signature["recommendation"] = memory["recommendation"];
    if (memory.contains("avoid")) signature["avoid"] = memory["avoid"];
    return signature;
}

bool allowed_review_status(const std::string& status) {
    return status == "accepted" || status == "rejected" || status == "deprecated";
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

} // namespace

PiMemoryStore::PiMemoryStore(std::filesystem::path memory_dir)
    : _memory_dir(std::move(memory_dir)) {}

std::filesystem::path PiMemoryStore::memories_path() const {
    return _memory_dir / "memories.jsonl";
}

std::filesystem::path PiMemoryStore::reviews_path() const {
    return _memory_dir / "memory_reviews.jsonl";
}

nlohmann::json PiMemoryStore::append_candidate(nlohmann::json memory) const {
    if (!memory.is_object()) {
        throw std::invalid_argument("PI memory must be a JSON object");
    }
    if (memory.contains("schema_version") && string_field(memory, "schema_version") != kMemorySchemaVersion) {
        throw std::invalid_argument("PI memory schema_version must be pi.memory.v1");
    }

    memory["schema_version"] = kMemorySchemaVersion;
    if (string_field(memory, "memory_id").empty()) {
        memory["memory_id"] = "mem_" + utc_timestamp_compact();
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
    }

    while (static_cast<int>(items.size()) > limit) items.erase(items.begin());
    return items;
}

nlohmann::json PiMemoryStore::review(const std::string& memory_id,
                                     const std::string& status,
                                     const std::string& reviewer,
                                     const std::string& note,
                                     const nlohmann::json& outcome) const {
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
        {"status", status},
        {"reviewed_at", utc_timestamp_iso()},
        {"reviewer", reviewer.empty() ? "user" : reviewer},
        {"note", note}
    };
    if (outcome.is_object() && !outcome.empty()) {
        review_event["outcome"] = outcome;
    }

    std::error_code ec;
    std::filesystem::create_directories(_memory_dir, ec);
    if (ec) throw std::runtime_error("failed to create PI memory directory: " + ec.message());

    std::ofstream out(reviews_path(), std::ios::app);
    if (!out) throw std::runtime_error("failed to open PI memory review store");
    out << review_event.dump() << '\n';
    if (!out) throw std::runtime_error("failed to write PI memory review");
    return review_event;
}

nlohmann::json PiMemoryStore::retrieve(const nlohmann::json& query, int limit) const {
    nlohmann::json matches = nlohmann::json::array();
    if (limit <= 0) return matches;

    const std::string wanted_type = string_field(query, "type");
    std::set<std::string> wanted_paths;
    if (query.contains("paths")) collect_paths(query["paths"], wanted_paths);
    if (query.contains("config_updates")) collect_paths(query["config_updates"], wanted_paths);
    collect_paths(query, wanted_paths);

    for (const auto& item : list(100000)) {
        const std::string status = item.value("status", std::string());
        if (status != "accepted" && status != "candidate") continue;
        int score = 0;
        if (!wanted_type.empty() && item.value("type", std::string()) != wanted_type) continue;
        if (!wanted_type.empty()) score += 3;
        if (status == "accepted") score += 2;
        if (item.contains("outcome") && item["outcome"].is_object()) {
            if (item["outcome"].value("validation_valid", false)) score += 2;
        }

        std::set<std::string> memory_paths;
        collect_paths(item, memory_paths);
        for (const auto& path : wanted_paths) {
            if (memory_paths.count(path)) score += 2;
        }
        if (score <= 0 && !wanted_type.empty()) continue;
        if (score <= 0 && !wanted_paths.empty()) continue;
        if (score <= 0) score = 1;

        nlohmann::json match = item;
        match["retrieval_score"] = score;
        matches.push_back(std::move(match));
    }

    std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
        return a.value("retrieval_score", 0) > b.value("retrieval_score", 0);
    });
    while (static_cast<int>(matches.size()) > limit) matches.erase(matches.end() - 1);
    return matches;
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
        {"schema_version", "pi.memories-export.v1"},
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
    if (!bundle.is_object() || bundle.value("schema_version", std::string()) != "pi.memories-export.v1") {
        throw std::invalid_argument("PI memory import bundle must have schema_version pi.memories-export.v1");
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
