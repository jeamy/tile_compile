#include "services/pi/pi_run_learning_store.hpp"
#include "services/pi/pi_run_index.hpp"
#include "services/pi/pi_decision_state.hpp"
#include "services/pi/pi_decision_outcome.hpp"

#include <openssl/evp.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <set>
#include <stdexcept>

namespace tile_compile::pi {
namespace {
using nlohmann::json;
namespace fs = std::filesystem;
constexpr std::uintmax_t kDocumentLimit = 16 * 1024 * 1024;
constexpr std::uintmax_t kArchiveLimit = 64 * 1024 * 1024;
constexpr std::size_t kPreviewLimit = 2 * 1024 * 1024;
std::int64_t now() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
bool inside(const fs::path& root, const fs::path& file) {
    auto a = fs::weakly_canonical(root), b = fs::weakly_canonical(file);
    auto i = a.begin(), j = b.begin();
    for (; i != a.end(); ++i, ++j) if (j == b.end() || *i != *j) return false;
    return true;
}
struct Collector {
    fs::path root;
    std::uintmax_t bytes = 0;
    json issues = json::array(), missing = json::array();
    std::optional<std::string> text(const std::string& relative, std::uintmax_t limit = kDocumentLimit) {
        const auto path = root / relative;
        if (!fs::exists(path)) { missing.push_back(relative); return std::nullopt; }
        if (!inside(root, path) || !fs::is_regular_file(path)) {
            issues.push_back({{"path", relative}, {"reason", "not_a_local_regular_file"}}); return std::nullopt;
        }
        const auto size = fs::file_size(path);
        if (size > limit || size + bytes > kArchiveLimit) {
            issues.push_back({{"path", relative}, {"reason", "size_limit"}}); return std::nullopt;
        }
        std::ifstream in(path, std::ios::binary);
        std::string data(static_cast<std::size_t>(size), '\0');
        if (!in || !in.read(data.data(), static_cast<std::streamsize>(size))) {
            issues.push_back({{"path", relative}, {"reason", "read_failed"}}); return std::nullopt;
        }
        bytes += size;
        return data;
    }
    std::optional<json> document(const std::string& relative) {
        auto data = text(relative);
        if (!data) return std::nullopt;
        auto parsed = json::parse(*data, nullptr, false);
        if (parsed.is_discarded() || (!parsed.is_object() && !parsed.is_array())) {
            issues.push_back({{"path", relative}, {"reason", "invalid_json"}}); return std::nullopt;
        }
        return json{{"content_sha256", sha256_prefixed(*data)}, {"data", parsed}};
    }
};
std::uint32_t big_endian(const unsigned char* p) {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) | (std::uint32_t(p[2]) << 8) | p[3];
}
}

PiRunLearningStore::PiRunLearningStore(const fs::path& dir) : _db(PiDatabase::open(dir)) {}
json PiRunLearningStore::capture(const fs::path& run_dir, const std::string& run_id,
                                 const std::string& stage, const std::string& status) {
    if (!fs::is_directory(run_dir)) throw std::invalid_argument("Run directory not found");
    Collector reader{run_dir};
    json documents = json::object();
    // Deliberate metadata allowlist: no FITS, caches, transcripts, or arbitrary artifact traversal.
    for (const char* path : {
        "artifacts/pi_run_provenance.json", "artifacts/run_provenance.json", "artifacts/effective_config.json", "artifacts/config_migration.json",
        "artifacts/pi_run_quality.json", "artifacts/stats.json", "artifacts/global_metrics.json",
        "artifacts/global_registration.json", "artifacts/normalization.json", "artifacts/bge.json",
        "artifacts/pcc.json", "artifacts/forward_drizzle.json", "artifacts/sampling_geometry.json",
        "artifacts/registration_sampling.json", "artifacts/forward_common_overlap.json",
        "artifacts/source_quality_plan.json", "artifacts/acceleration_context.json",
        "artifacts/config_validation.json", "artifacts/preprocess/preprocessing_report.json",
        "artifacts/preprocess/preprocessing_registration.json", "artifacts/preprocess/quality_analysis.json",
        "artifacts/preprocess/artifacts_manifest.json"})
        if (auto data = reader.document(path)) documents[path] = *data;
    for (const char* path : {"artifacts/preprocess/frame_quality.csv", "artifacts/preprocess/rejected_frames.txt"}) {
        if (auto text = reader.text(path))
            documents[path] = {{"content_sha256", sha256_prefixed(*text)}, {"data", {{"format", "text"}, {"text", *text}}}};
    }
    json provenance = documents.contains("artifacts/pi_run_provenance.json")
        ? documents["artifacts/pi_run_provenance.json"]["data"] : json::object();
    if (!provenance.is_object()) throw std::runtime_error("Invalid PI provenance");
    PiRunIndex index(_db->dir());
    const auto identity = index.resolve(run_dir, provenance.value("run_uid", std::string()),
        provenance.value("config_sha256", std::string()), provenance.value("started_at", std::string()));
    const std::string uid = identity.at("run_uid");
    json config = {{"available", false}, {"yaml", nullptr}, {"effective", nullptr}, {"sha256", nullptr}};
    if (const auto yaml = reader.text("config.yaml", 2 * 1024 * 1024)) {
        config["available"] = true;
        config["yaml"] = *yaml;
        config["sha256"] = sha256_prefixed(*yaml);
        try { config["effective"] = yaml_text_to_json(*yaml); }
        catch (const std::exception&) { reader.issues.push_back({{"path", "config.yaml"}, {"reason", "invalid_yaml"}}); }
    }
    config["submitted"] = config["effective"];
    config["effective_source"] = "saved_yaml";
    config["parser_defaults_included"] = false;
    if (documents.contains("artifacts/effective_config.json")) {
        const auto& expanded = documents["artifacts/effective_config.json"]["data"];
        if (expanded.is_object() && expanded.contains("source_config_sha256") && expanded["source_config_sha256"].is_string() &&
            expanded.contains("expanded_yaml") && expanded["expanded_yaml"].is_string()) {
            if (config["sha256"] == json("sha256:" + expanded["source_config_sha256"].get<std::string>())) {
                try {
                    auto parsed = yaml_text_to_json(expanded["expanded_yaml"].get<std::string>());
                    if (!parsed.is_object()) throw std::runtime_error("Invalid expanded config");
                    config["effective"] = parsed;
                    config["effective_source"] = "runner_expanded";
                    config["parser_defaults_included"] = true;
                } catch (const std::exception&) {
                    reader.issues.push_back({{"path", "artifacts/effective_config.json"}, {"reason", "invalid_expanded_yaml"}});
                }
            } else config["expanded_config_stale"] = true;
        } else reader.issues.push_back({{"path", "artifacts/effective_config.json"}, {"reason", "invalid_expanded_config"}});
    }
    if (!config["available"].get<bool>()) reader.issues.push_back({{"path", "config.yaml"}, {"reason", "config_unavailable"}});
    else if (!config["effective"].is_object()) reader.issues.push_back({{"path", "config.yaml"}, {"reason", "config_not_object"}});
    json source = {{"original_input_dir", provenance.value("original_input_dir", json(nullptr))},
                   {"effective_input_dir", provenance.value("effective_input_dir", json(nullptr))},
                   {"light_manifest", nullptr}, {"calibration", nullptr}};
    if (documents.contains("artifacts/run_provenance.json")) {
        const auto& p = documents["artifacts/run_provenance.json"]["data"];
        if (p.is_object()) source["light_manifest"] = p.value("input_manifest", json(nullptr));
    }
    json phase_events = json::array();
    for (const char* log : {"logs/run_events.jsonl", "run_events.jsonl", "events.jsonl", "logs/events.jsonl"}) {
        if (!fs::is_regular_file(run_dir / log)) continue;
        const auto text = reader.text(log);
        if (!text) continue;
        std::size_t offset = 0;
        while (offset < text->size()) {
            const auto end = text->find('\n', offset);
            const auto line = text->substr(offset, end == std::string::npos ? std::string::npos : end - offset);
            offset = end == std::string::npos ? text->size() : end + 1;
            const auto event = json::parse(line, nullptr, false);
            if (event.is_discarded() || !event.is_object()) continue;
            const auto payload = event.value("payload", json::object());
            if (event.contains("calibration")) source["calibration"] = event["calibration"];
            else if (payload.is_object() && payload.contains("calibration")) source["calibration"] = payload["calibration"];
            if (event.contains("input_dir") && event["input_dir"].is_string()) source["observed_scan_input_dir"] = event["input_dir"];
            const auto name = event.contains("type") && event["type"].is_string() ? event["type"].get<std::string>()
                : event.contains("event") && event["event"].is_string() ? event["event"].get<std::string>() : std::string();
            if (name == "run_start" || name == "run_end" || name == "phase_start" || name == "phase_end" ||
                name == "error" || name == "warning") {
                if (phase_events.size() < 10000) phase_events.push_back(event);
                else if (phase_events.size() == 10000) {
                    reader.issues.push_back({{"path", log}, {"reason", "event_limit"}});
                    phase_events.push_back({{"event", "archive_limit_reached"}});
                }
            }
        }
        break;
    }
    json snapshot = {
        {"schema_version", "pi.run-learning-snapshot.v1"}, {"run_uid", uid}, {"run_id", run_id},
        {"run_key", identity["run_key"]}, {"stage", stage}, {"status", status},
        {"privacy_class", "local_run_archive_with_paths"}, {"config", config}, {"source", source},
        {"artifacts", documents}, {"phase_events", phase_events},
        {"capture_issues", reader.issues}, {"missing_optional_files", reader.missing},
        {"available_artifacts_captured", reader.issues.empty()}, {"light_manifest_available", !source["light_manifest"].is_null()},
        {"comparison_kind", "unpaired"}, {"quality_delta", nullptr},
        {"validation_state", "unreviewed"}, {"raw_data_stored", false}
    };
    const auto content_hash = sha256_prefixed(snapshot.dump());
    const auto id = "snapshot_" + sha256_prefixed(uid + content_hash).substr(7);
    snapshot["snapshot_id"] = id;
    snapshot["content_sha256"] = content_hash;
    const auto serialized = snapshot.dump();
    if (serialized.size() > kArchiveLimit) throw std::runtime_error("Run learning snapshot exceeds archive size limit");
    PiDatabase::Tx tx(*_db);
    json summary = {{"run_uid", uid}, {"run_id", run_id}, {"snapshot_id", id}, {"stage", stage},
        {"status", status}, {"config_sha256", config["sha256"]}, {"validation_state", "unreviewed"},
        {"issue_count", reader.issues.size()}, {"has_light_manifest", !source["light_manifest"].is_null()}};
    _db->execute("INSERT OR IGNORE INTO run_learning_snapshots(snapshot_id, run_uid, content_sha256, created_at, json, summary_json) VALUES(?, ?, ?, ?, ?, ?)",
                 {id, uid, content_hash, now(), serialized, summary.dump()});
    _db->execute("INSERT INTO run_learning_state(run_uid, artifacts_state, excluded, exclusion_code, latest_snapshot_id) "
                 "VALUES(?, 'available', 0, '', ?) ON CONFLICT(run_uid) DO UPDATE SET "
                 "artifacts_state = 'available', latest_snapshot_id = excluded.latest_snapshot_id", {uid, id});
    tx.commit();
    return get(uid).value();
}

std::optional<json> PiRunLearningStore::get(const std::string& uid) {
    const auto rows = _db->query("SELECT s.json, st.artifacts_state, st.excluded, st.exclusion_code, s.created_at, "
                                "(SELECT snapshot_id FROM run_learning_previews WHERE run_uid = st.run_uid), "
                                "(SELECT source_artifact FROM run_learning_previews WHERE run_uid = st.run_uid), s.created_at "
                                "FROM run_learning_state st JOIN run_learning_snapshots s ON s.snapshot_id = st.latest_snapshot_id "
                                "WHERE st.run_uid = ?", {uid});
    if (rows.empty()) return std::nullopt;
    auto result = pi_sql_json(rows[0][0]);
    result["artifacts_state"] = pi_sql_text(rows[0][1]);
    bool reachable = false;
    for (const auto& alias : _db->query("SELECT run_key FROM run_aliases WHERE run_uid = ?", {uid})) {
        std::error_code error;
        if (fs::is_directory(fs::path(pi_sql_text(alias[0])), error)) reachable = true;
    }
    result["artifacts_reachable"] = reachable;
    result["observed_artifacts_state"] = reachable ? "reachable" : "missing_or_unreachable";
    result["excluded_from_learning"] = pi_sql_int(rows[0][2]) != 0;
    result["exclusion_code"] = pi_sql_text(rows[0][3]);
    result["captured_at_epoch"] = pi_sql_int(rows[0][4]);
    result["preview_snapshot_id"] = std::holds_alternative<std::nullptr_t>(rows[0][5]) ? json(nullptr) : json(pi_sql_text(rows[0][5]));
    result["preview_source_artifact"] = std::holds_alternative<std::nullptr_t>(rows[0][6]) ? json(nullptr) : json(pi_sql_text(rows[0][6]));
    result["preview_visualization"] = "display_only_not_measurement";
    result["captured_at"] = pi_sql_text(rows[0][7]);
    return result;
}
json PiRunLearningStore::history(const std::string& uid, int limit) {
    json result = json::array();
    for (const auto& row : _db->query("SELECT json, created_at FROM run_learning_snapshots WHERE run_uid = ? ORDER BY rowid DESC LIMIT ?",
                                      {uid, std::int64_t(std::clamp(limit, 1, 1000))})) {
        auto value = pi_sql_json(row[0]); value["captured_at_epoch"] = pi_sql_int(row[1]); result.push_back(value);
    }
    return result;
}
json PiRunLearningStore::list(int limit) {
    json result = json::array();
    for (const auto& row : _db->query("SELECT s.summary_json, s.created_at, st.artifacts_state, st.excluded, st.exclusion_code "
                                     "FROM run_learning_state st JOIN run_learning_snapshots s ON s.snapshot_id = st.latest_snapshot_id "
                                     "ORDER BY s.created_at DESC, s.rowid DESC LIMIT ?", {std::int64_t(std::clamp(limit, 1, 1000))})) {
        auto value = pi_sql_json(row[0]);
        value["captured_at_epoch"] = pi_sql_int(row[1]);
        value["artifacts_state"] = pi_sql_text(row[2]);
        value["excluded_from_learning"] = pi_sql_int(row[3]) != 0;
        value["exclusion_code"] = pi_sql_text(row[4]);
        result.push_back(value);
    }
    return result;
}
bool PiRunLearningStore::mark_artifacts_state(const std::string& uid, const std::string& state) {
    static const std::set<std::string> allowed = {"available", "deletion_pending", "deleted", "delete_failed", "missing"};
    if (!allowed.count(state)) throw std::invalid_argument("Invalid run artifacts state");
    return _db->execute("UPDATE run_learning_state SET artifacts_state = ? WHERE run_uid = ?", {state, uid}) > 0;
}
bool PiRunLearningStore::mark_artifacts_deleted(const std::string& uid) {
    return mark_artifacts_state(uid, "deleted");
}
bool PiRunLearningStore::set_excluded(const std::string& uid, bool excluded, const std::string& code) {
    static const std::set<std::string> allowed = {"test_run", "invalid_data", "unreliable_measurements", "user_choice"};
    if (excluded && !allowed.count(code)) throw std::invalid_argument("Invalid learning exclusion code");
    return _db->execute("UPDATE run_learning_state SET excluded = ?, exclusion_code = ? WHERE run_uid = ?",
                        {std::int64_t(excluded), excluded ? code : std::string(), uid}) > 0;
}
void PiRunLearningStore::save_preview(const std::string& uid, const std::string& id, const std::vector<unsigned char>& png,
                                      const std::string& source_artifact) {
    static constexpr std::array<unsigned char, 8> signature = {137,80,78,71,13,10,26,10};
    if (png.size() < 33 || png.size() > kPreviewLimit || !std::equal(signature.begin(), signature.end(), png.begin()) ||
        std::string(reinterpret_cast<const char*>(png.data() + 12), 4) != "IHDR")
        throw std::invalid_argument("Invalid or oversized PNG preview");
    const auto width = big_endian(png.data() + 16), height = big_endian(png.data() + 20);
    if (width == 0 || height == 0 || width > 1024 || height > 1024) throw std::invalid_argument("PNG preview dimensions exceed limit");
    if (_db->query("SELECT 1 FROM run_learning_snapshots WHERE run_uid = ? AND snapshot_id = ?", {uid, id}).empty())
        throw std::invalid_argument("Preview snapshot mismatch");
    std::string encoded(4 * ((png.size() + 2) / 3) + 1, '\0');
    const int count = EVP_EncodeBlock(reinterpret_cast<unsigned char*>(encoded.data()), png.data(), static_cast<int>(png.size()));
    encoded.resize(static_cast<std::size_t>(count));
    _db->execute("INSERT INTO run_learning_previews(run_uid, snapshot_id, png_base64, source_artifact) VALUES(?, ?, ?, ?) "
                 "ON CONFLICT(run_uid) DO UPDATE SET snapshot_id = excluded.snapshot_id, png_base64 = excluded.png_base64, "
                 "source_artifact = excluded.source_artifact", {uid, id, encoded, source_artifact});
}
std::optional<std::vector<unsigned char>> PiRunLearningStore::preview(const std::string& uid) {
    const auto rows = _db->query("SELECT png_base64 FROM run_learning_previews WHERE run_uid = ?", {uid});
    if (rows.empty()) return std::nullopt;
    const auto encoded = pi_sql_text(rows[0][0]);
    if (encoded.empty() || encoded.size() % 4 || encoded.size() > 4 * ((kPreviewLimit + 2) / 3))
        throw std::runtime_error("Invalid stored PNG encoding");
    std::vector<unsigned char> decoded(encoded.size() / 4 * 3);
    int size = EVP_DecodeBlock(decoded.data(), reinterpret_cast<const unsigned char*>(encoded.data()), static_cast<int>(encoded.size()));
    if (size < 0) throw std::runtime_error("Invalid stored PNG encoding");
    if (encoded.back() == '=') --size;
    if (encoded[encoded.size() - 2] == '=') --size;
    decoded.resize(static_cast<std::size_t>(size));
    return decoded;
}
} // namespace tile_compile::pi
