#include "services/pi/pi_decision_record_store.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace tile_compile::pi {
namespace {

using nlohmann::json;

std::mutex& store_mutex() {
    static std::mutex m;
    return m;
}

std::tm gmtime_safe(const std::time_t& t) {
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    return tm;
}

std::string utc_iso_now() {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    const std::tm tm = gmtime_safe(t);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string new_id(const char* prefix) {
    static std::atomic<unsigned long long> counter{0};
    const auto now = std::chrono::system_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const std::tm tm = gmtime_safe(t);
    std::ostringstream out;
    out << prefix << std::put_time(&tm, "%Y%m%d_%H%M%S") << '_' << ns << '_' << counter.fetch_add(1);
    return out.str();
}

std::string str_field(const json& o, const char* key) {
    if (!o.is_object() || !o.contains(key) || !o[key].is_string()) return "";
    return o[key].get<std::string>();
}

const std::set<std::string>& allowed_kinds() {
    static const std::set<std::string> k = {
        "config_apply", "config_reject", "config_undo", "preview_dismissed", "config_manual_edit",
        "llm_proposal", "jev_choice", "jev_override", "live_edit_op", "live_edit_undo",
        "live_edit_keep", "live_edit_expired", "memory_review", "no_change"};
    return k;
}

// kind -> erlaubte Actors (Plan §2.2).
std::set<std::string> allowed_actors_for_kind(const std::string& kind) {
    if (kind == "llm_proposal") return {"llm"};
    if (kind == "jev_choice") return {"jev"};
    if (kind == "live_edit_expired") return {"rule"};
    if (kind == "live_edit_op" || kind == "no_change") return {"user", "llm"};
    if (kind == "live_edit_keep") return {"user", "rule"};
    return {"user"};
}

const std::set<std::string>& allowed_basis_sources() {
    static const std::set<std::string> s = {"measured_fact", "model_probability", "rule", "llm_hypothesis"};
    return s;
}

std::set<std::string> allowed_basis_for_actor(const std::string& actor) {
    if (actor == "user") return {"measured_fact", "model_probability", "rule", "llm_hypothesis"};
    if (actor == "llm") return {"llm_hypothesis"};
    if (actor == "jev") return {"model_probability", "rule"};
    if (actor == "rule") return {"rule"};
    return {};
}

bool looks_absolute_path(const std::string& s) {
    if (s.empty()) return false;
    if (s[0] == '/' || s[0] == '\\') return true;
    return s.size() >= 3 && std::isalpha(static_cast<unsigned char>(s[0])) && s[1] == ':' &&
           (s[2] == '\\' || s[2] == '/');
}

bool has_parent_segment(const std::string& s) {
    return s.find("../") != std::string::npos || s.find("..\\") != std::string::npos || s == "..";
}

std::vector<json> read_jsonl(const std::filesystem::path& p) {
    std::vector<json> out;
    std::ifstream in(p);
    if (!in) return out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        json j = json::parse(line, nullptr, false);
        if (j.is_object()) out.push_back(std::move(j));
    }
    return out;
}

void append_jsonl_line(const std::filesystem::path& p, const json& j) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    if (ec) throw std::runtime_error("failed to create decision record directory: " + ec.message());
    std::ofstream out(p, std::ios::app);
    if (!out) throw std::runtime_error("failed to open " + p.string());
    out << j.dump() << '\n';
    out.flush();
    if (!out) throw std::runtime_error("failed to write " + p.string());
}

json default_user_rationale() {
    return {{"reason_codes", json::array()}, {"catalog_version", ""}, {"text", ""}, {"no_reason_given", false}};
}

// Wendet den Overlay-Stand auf ein Record an. `links` enthaelt nur Links dieses Records.
void apply_links(json& record, const std::vector<json>& links) {
    json outcome_refs = record.contains("outcome_refs") && record["outcome_refs"].is_array()
        ? record["outcome_refs"] : json::array();
    json supersedes = json::array();
    for (const auto& link : links) {
        const std::string type = str_field(link, "type");
        const json data = link.contains("data") && link["data"].is_object() ? link["data"] : json::object();
        if (type == "memory") {
            record["memory_id"] = data.value("memory_id", std::string());
        } else if (type == "outcome") {
            const json ref = data.contains("ref") ? data["ref"] : json(nullptr);
            if (!ref.is_null() && std::find(outcome_refs.begin(), outcome_refs.end(), ref) == outcome_refs.end()) {
                outcome_refs.push_back(ref);
            }
        } else if (type == "supersedes") {
            if (data.contains("decision_id")) supersedes.push_back(data["decision_id"]);
        } else if (type == "run_deleted") {
            record["run_deleted"] = true;
        } else if (type == "redact") {
            record["redacted"] = true;
        }
    }
    record["outcome_refs"] = outcome_refs;
    if (!supersedes.empty()) record["supersedes"] = supersedes;
    if (record.value("redacted", false) && record.contains("rationale") && record["rationale"].is_object() &&
        record["rationale"].contains("user") && record["rationale"]["user"].is_object()) {
        record["rationale"]["user"]["text"] = "";
        record["rationale"]["user"]["text_redacted"] = true;
    }
}

std::map<std::string, std::vector<json>> group_links(const std::vector<json>& links) {
    std::map<std::string, std::vector<json>> by_id;
    for (const auto& l : links) {
        const std::string id = str_field(l, "decision_id");
        if (!id.empty()) by_id[id].push_back(l);
    }
    return by_id;
}

bool record_in_run(const json& record, const std::string& run_uid) {
    if (run_uid.empty() || !record.contains("context_ref") || !record["context_ref"].is_object()) return false;
    const auto& c = record["context_ref"];
    if (str_field(c, "run_uid") == run_uid) return true;
    const std::string image_id = str_field(c, "image_id");
    return image_id.rfind(run_uid + ":", 0) == 0;
}

bool matches_filter(const json& record, const json& filter) {
    if (!filter.is_object()) return true;
    auto eq = [&](const char* key, const std::string& actual) {
        const std::string want = str_field(filter, key);
        return want.empty() || want == actual;
    };
    if (!eq("kind", str_field(record, "kind"))) return false;
    if (!eq("actor", str_field(record, "actor"))) return false;
    if (!eq("parent_decision_id", str_field(record, "parent_decision_id"))) return false;
    const json ctx = record.contains("context_ref") && record["context_ref"].is_object()
        ? record["context_ref"] : json::object();
    if (!eq("context_id", str_field(ctx, "context_id"))) return false;
    if (!eq("run_uid", str_field(ctx, "run_uid"))) return false;
    const std::string since = str_field(filter, "since");
    if (!since.empty() && str_field(record, "created_at") < since) return false;
    const std::string code = str_field(filter, "reason_code");
    if (!code.empty()) {
        bool found = false;
        if (record.contains("rationale") && record["rationale"].is_object() &&
            record["rationale"].contains("user") && record["rationale"]["user"].is_object()) {
            const auto& codes = record["rationale"]["user"].value("reason_codes", json::array());
            for (const auto& c : codes) {
                if (c.is_string() && c.get<std::string>() == code) found = true;
            }
        }
        if (!found) return false;
    }
    return true;
}

} // namespace

std::string scrub_rationale_text(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char ch : text) {
        if (ch == '\n' || ch == '\t') out.push_back(' ');
        else if (ch >= 0x20 || ch >= 0x80) out.push_back(static_cast<char>(ch));
    }
    static const std::regex unix_path(R"((^|[\s"'(])(/(?:[^\s/"')]+/)+[^\s/"')]*))");
    static const std::regex win_path(R"((^|[\s"'(])([A-Za-z]:[\\/][^\s"')]*))");
    out = std::regex_replace(out, unix_path, "$1[Pfad]");
    out = std::regex_replace(out, win_path, "$1[Pfad]");
    if (out.size() > kDecisionRationaleTextMaxChars) {
        std::size_t cut = kDecisionRationaleTextMaxChars;
        while (cut > 0 && (static_cast<unsigned char>(out[cut]) & 0xC0) == 0x80) --cut;
        out.resize(cut);
    }
    return out;
}

void validate_decision_record(const json& r) {
    if (!r.is_object()) throw std::invalid_argument("decision record must be a JSON object");
    if (str_field(r, "schema_version") != kDecisionRecordSchemaVersion) {
        throw std::invalid_argument("decision record schema_version must be pi.decision-record.v1");
    }
    const std::string kind = str_field(r, "kind");
    if (!allowed_kinds().count(kind)) throw std::invalid_argument("unknown decision kind: " + kind);
    const std::string actor = str_field(r, "actor");
    if (!allowed_actors_for_kind(kind).count(actor)) {
        throw std::invalid_argument("actor '" + actor + "' not allowed for kind '" + kind + "'");
    }
    const std::string origin = str_field(r, "origin");
    if (origin != "live" && origin != "legacy_migration") {
        throw std::invalid_argument("origin must be live or legacy_migration");
    }
    const std::string privacy = str_field(r, "privacy_class");
    if (privacy != "metadata_only" && privacy != "metadata_plus_user_text") {
        throw std::invalid_argument("invalid privacy_class");
    }
    if (!r.contains("context_ref") || !r["context_ref"].is_object()) {
        throw std::invalid_argument("context_ref object is required");
    }
    if (r.contains("evidence") && r["evidence"].is_object()) {
        const std::string mref = str_field(r["evidence"], "metrics_ref");
        if (looks_absolute_path(mref) || has_parent_segment(mref)) {
            throw std::invalid_argument("evidence.metrics_ref must be artifact-relative");
        }
    }
    if (r.contains("subject") && r["subject"].is_object() && r["subject"].contains("paths")) {
        if (!r["subject"]["paths"].is_array()) throw std::invalid_argument("subject.paths must be an array");
        for (const auto& p : r["subject"]["paths"]) {
            if (!p.is_object() || str_field(p, "path").empty()) {
                throw std::invalid_argument("subject.paths entries need a path");
            }
        }
    }
    if (r.contains("jev") && r["jev"].is_object() && !r["jev"].empty() && kind.rfind("jev_", 0) != 0) {
        throw std::invalid_argument("jev block is only valid for jev_* kinds");
    }

    if (!r.contains("rationale") || !r["rationale"].is_object()) {
        throw std::invalid_argument("rationale object is required");
    }
    const json& rat = r["rationale"];
    if (!rat.contains("user") || !rat["user"].is_object()) throw std::invalid_argument("rationale.user is required");
    if (!rat.contains("basis") || !rat["basis"].is_array()) throw std::invalid_argument("rationale.basis must be an array");
    const json& user = rat["user"];
    if (!user.contains("reason_codes") || !user["reason_codes"].is_array()) {
        throw std::invalid_argument("rationale.user.reason_codes must be an array");
    }
    for (const auto& c : user["reason_codes"]) {
        if (!c.is_string() || c.get<std::string>().empty()) {
            throw std::invalid_argument("rationale.user.reason_codes must be non-empty strings");
        }
    }
    const bool has_codes = !user["reason_codes"].empty();
    const std::string text = str_field(user, "text");
    const bool no_reason = user.value("no_reason_given", false);
    if (has_codes && str_field(user, "catalog_version").empty()) {
        throw std::invalid_argument("rationale.user.catalog_version is required with reason_codes");
    }
    if (no_reason && (has_codes || !text.empty())) {
        throw std::invalid_argument("no_reason_given excludes reason_codes and text");
    }
    if (text.size() > kDecisionRationaleTextMaxChars) {
        throw std::invalid_argument("rationale.user.text exceeds length limit");
    }
    if ((privacy == "metadata_only") && !text.empty()) {
        throw std::invalid_argument("privacy_class must be metadata_plus_user_text when text is present");
    }
    if (actor != "user" && (has_codes || !text.empty() || no_reason)) {
        throw std::invalid_argument("rationale.user must be empty for non-user actors");
    }
    const auto allowed_basis = allowed_basis_for_actor(actor);
    for (const auto& b : rat["basis"]) {
        if (!b.is_object()) throw std::invalid_argument("rationale.basis entries must be objects");
        const std::string src = str_field(b, "source");
        if (!allowed_basis_sources().count(src)) throw std::invalid_argument("unknown rationale.basis source: " + src);
        if (!allowed_basis.count(src)) {
            throw std::invalid_argument("basis source '" + src + "' not allowed for actor '" + actor + "'");
        }
    }
}

PiDecisionRecordStore::PiDecisionRecordStore(std::filesystem::path dir) : _dir(std::move(dir)) {}

std::filesystem::path PiDecisionRecordStore::records_path() const { return _dir / "decisions_v1.jsonl"; }
std::filesystem::path PiDecisionRecordStore::links_path() const { return _dir / "decision_links_v1.jsonl"; }

json PiDecisionRecordStore::append(json record) const {
    if (!record.is_object()) throw std::invalid_argument("decision record must be a JSON object");
    record["schema_version"] = kDecisionRecordSchemaVersion;
    if (str_field(record, "decision_id").empty()) record["decision_id"] = new_id("dec_");
    if (str_field(record, "created_at").empty()) record["created_at"] = utc_iso_now();
    if (str_field(record, "origin").empty()) record["origin"] = "live";
    if (!record.contains("context_ref") || !record["context_ref"].is_object()) {
        record["context_ref"] = json::object();
    }
    if (!record.contains("rationale") || !record["rationale"].is_object()) record["rationale"] = json::object();
    if (!record["rationale"].contains("user") || !record["rationale"]["user"].is_object()) {
        record["rationale"]["user"] = default_user_rationale();
    }
    if (!record["rationale"].contains("basis") || !record["rationale"]["basis"].is_array()) {
        record["rationale"]["basis"] = json::array();
    }
    json& user = record["rationale"]["user"];
    const json user_defaults = default_user_rationale();
    for (const auto& [k, v] : user_defaults.items()) {
        if (!user.contains(k)) user[k] = v;
    }
    if (user["text"].is_string()) user["text"] = scrub_rationale_text(user["text"].get<std::string>());
    // Fehlender Grund ist ein Wert (Plan §2.1): kein Raten, aber explizit markieren.
    if (str_field(record, "actor") == "user" && user["reason_codes"].is_array() && user["reason_codes"].empty() &&
        str_field(user, "text").empty()) {
        user["no_reason_given"] = true;
    }
    record["privacy_class"] = str_field(user, "text").empty() ? "metadata_only" : "metadata_plus_user_text";
    for (const char* key : {"outcome_refs"}) {
        if (!record.contains(key)) record[key] = json::array();
    }
    if (!record.contains("memory_id")) record["memory_id"] = nullptr;

    validate_decision_record(record);

    std::lock_guard<std::mutex> lock(store_mutex());
    const std::string key = str_field(record, "idempotency_key");
    if (!key.empty()) {
        for (auto& existing : read_jsonl(records_path())) {
            if (str_field(existing, "idempotency_key") == key) {
                existing["duplicate"] = true;
                return existing;
            }
        }
    }
    append_jsonl_line(records_path(), record);
    record["duplicate"] = false;
    return record;
}

json PiDecisionRecordStore::get(const std::string& decision_id) const {
    std::lock_guard<std::mutex> lock(store_mutex());
    auto links = group_links(read_jsonl(links_path()));
    for (auto record : read_jsonl(records_path())) {
        if (str_field(record, "decision_id") != decision_id) continue;
        apply_links(record, links[decision_id]);
        return record;
    }
    return nullptr;
}

json PiDecisionRecordStore::list(const json& filter, int limit) const {
    std::lock_guard<std::mutex> lock(store_mutex());
    auto links = group_links(read_jsonl(links_path()));
    json out = json::array();
    for (auto record : read_jsonl(records_path())) {
        apply_links(record, links[str_field(record, "decision_id")]);
        if (matches_filter(record, filter)) out.push_back(std::move(record));
    }
    if (limit > 0 && static_cast<int>(out.size()) > limit) {
        json trimmed = json::array();
        for (std::size_t i = out.size() - static_cast<std::size_t>(limit); i < out.size(); ++i) trimmed.push_back(out[i]);
        return trimmed;
    }
    return out;
}

json PiDecisionRecordStore::add_link(const std::string& decision_id, const std::string& type, const json& data) const {
    static const std::set<std::string> types = {"memory", "outcome", "supersedes", "run_deleted", "redact"};
    if (!types.count(type)) throw std::invalid_argument("unknown decision link type: " + type);
    if (decision_id.empty()) throw std::invalid_argument("decision_id is required");
    json link = {
        {"schema_version", kDecisionLinkSchemaVersion},
        {"link_id", new_id("dlk_")},
        {"decision_id", decision_id},
        {"type", type},
        {"data", data.is_object() ? data : json::object()},
        {"created_at", utc_iso_now()},
    };
    std::lock_guard<std::mutex> lock(store_mutex());
    bool known = false;
    for (const auto& r : read_jsonl(records_path())) {
        if (str_field(r, "decision_id") == decision_id) { known = true; break; }
    }
    if (!known) throw std::invalid_argument("unknown decision_id: " + decision_id);
    append_jsonl_line(links_path(), link);
    return link;
}

json PiDecisionRecordStore::redact(const std::string& decision_id, const std::string& reason) const {
    return add_link(decision_id, "redact", {{"reason", reason}, {"field", "rationale.user.text"}});
}

int PiDecisionRecordStore::mark_run_deleted(const std::string& run_uid) const {
    if (run_uid.empty()) throw std::invalid_argument("run_uid is required");
    std::vector<std::string> ids;
    {
        std::lock_guard<std::mutex> lock(store_mutex());
        auto links = group_links(read_jsonl(links_path()));
        for (auto record : read_jsonl(records_path())) {
            if (!record_in_run(record, run_uid)) continue;
            const std::string id = str_field(record, "decision_id");
            apply_links(record, links[id]);
            if (!record.value("run_deleted", false) || !record.value("redacted", false)) ids.push_back(id);
        }
    }
    for (const auto& id : ids) {
        add_link(id, "run_deleted", {{"run_uid", run_uid}});
        add_link(id, "redact", {{"reason", "run_deleted"}, {"field", "rationale.user.text"}});
    }
    return static_cast<int>(ids.size());
}

json PiDecisionRecordStore::compact() const {
    std::lock_guard<std::mutex> lock(store_mutex());
    auto links = group_links(read_jsonl(links_path()));
    std::set<std::string> redacted;
    for (const auto& [id, ls] : links) {
        for (const auto& l : ls) {
            if (str_field(l, "type") == "redact") { redacted.insert(id); break; }
        }
    }
    auto records = read_jsonl(records_path());
    int removed = 0;
    for (auto& r : records) {
        if (!redacted.count(str_field(r, "decision_id"))) continue;
        if (r.contains("rationale") && r["rationale"].is_object() && r["rationale"].contains("user") &&
            r["rationale"]["user"].is_object()) {
            auto& u = r["rationale"]["user"];
            if (!str_field(u, "text").empty()) ++removed;
            u["text"] = "";
            u["text_redacted"] = true;
            r["privacy_class"] = "metadata_only";
        }
    }
    const std::filesystem::path tmp = records_path().string() + ".tmp";
    {
        std::error_code ec;
        std::filesystem::create_directories(_dir, ec);
        std::ofstream out(tmp, std::ios::out | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write " + tmp.string());
        for (const auto& r : records) out << r.dump() << '\n';
        out.flush();
        if (!out) throw std::runtime_error("write failed " + tmp.string());
    }
    std::filesystem::rename(tmp, records_path());
    return {{"rewritten", static_cast<int>(records.size())}, {"redacted_texts_removed", removed}};
}

} // namespace tile_compile::pi
