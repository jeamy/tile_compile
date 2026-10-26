#include "services/pi/pi_analysis_context_store.hpp"
#include "services/pi/pi_run_index.hpp"
#include "time_utils.hpp"

namespace tile_compile::pi {
using nlohmann::json;
PiAnalysisContextStore::PiAnalysisContextStore(const std::filesystem::path& dir) : db_(PiDatabase::open(dir)) {}
std::optional<json> PiAnalysisContextStore::get(const std::string& uid) const {
    const auto rows = db_->query("SELECT json FROM analysis_contexts WHERE analysis_uid = ?", {uid});
    return rows.empty() ? std::nullopt : std::optional<json>(pi_sql_json(rows[0][0]));
}
json PiAnalysisContextStore::resolve(const AdviceRequest& r) {
    if (r.scan_id.empty() || r.dataset_manifest.empty() || !r.base_config.is_object())
        throw std::invalid_argument("Analysis requires an authoritative scan, manifest and config draft");
    if (r.locked_paths.size() > 1000) throw std::invalid_argument("Too many analysis locks");
    for (const auto& path : r.locked_paths) if (path.size() > 256) throw std::invalid_argument("Analysis lock path exceeds limit");
    const std::string signature = sha256_prefixed(json{{"scan_id", r.scan_id}, {"dataset_manifest", r.dataset_manifest}}.dump());
    const std::string revision = "arev_" + sha256_prefixed(json{{"signature", signature}, {"config", r.base_config},
        {"metrics", r.metrics}, {"locks", r.locked_paths}}.dump()).substr(7);
    PiDatabase::Tx tx(*db_);
    auto rows = db_->query("SELECT json FROM analysis_contexts WHERE scan_signature = ?", {signature});
    json context;
    if (rows.empty()) {
        const std::string uid = "analysis_" + PiRunIndex::generate_uid();
        context = {{"schema_version", "pi.analysis-context.v1"}, {"kind", "analysis"}, {"analysis_uid", uid},
            {"context_id", "analysis:" + uid}, {"scan_id", r.scan_id}, {"scan_signature", signature},
            {"input_path", r.scan.value("input_path", std::string())}, {"created_at", utc_now_iso()}};
        db_->execute("INSERT INTO analysis_contexts(analysis_uid, scan_signature, json) VALUES(?, ?, ?)", {uid, signature, context.dump()});
    } else context = pi_sql_json(rows[0][0]);
    json detail = {{"revision_id", revision}, {"analysis_uid", context["analysis_uid"]}, {"created_at", utc_now_iso()},
        {"config_sha256", sha256_prefixed(r.base_config.dump())}, {"metrics_sha256", sha256_prefixed(r.metrics.dump())},
        {"manifest_sha256", sha256_prefixed(r.dataset_manifest.dump())}, {"locked_paths", r.locked_paths}};
    db_->execute("INSERT INTO analysis_revisions(revision_id, analysis_uid, json) VALUES(?, ?, ?) ON CONFLICT(revision_id) DO NOTHING",
        {revision, context["analysis_uid"].get<std::string>(), detail.dump()});
    tx.commit();
    context["revision_id"] = revision;
    return context;
}
json PiAnalysisContextStore::revisions(const std::string& uid) const {
    json items = json::array();
    for (const auto& row : db_->query("SELECT json FROM analysis_revisions WHERE analysis_uid = ? ORDER BY rowid DESC LIMIT 100", {uid}))
        items.push_back(pi_sql_json(row[0]));
    return items;
}
void PiAnalysisContextStore::link(const std::string& uid, const std::string& revision, const std::string& proposal) {
    PiDatabase::Tx tx(*db_);
    if (db_->query("SELECT json FROM jev_documents WHERE proposal_id = ? AND part = 'status'", {proposal}).empty())
        throw std::invalid_argument("Unknown analysis proposal");
    const auto previous = db_->query("SELECT analysis_uid, revision_id FROM analysis_proposals WHERE proposal_id = ?", {proposal});
    if (!previous.empty() && (pi_sql_text(previous[0][0]) != uid || pi_sql_text(previous[0][1]) != revision))
        throw std::invalid_argument("Proposal already belongs to a different analysis revision");
    db_->execute("INSERT INTO analysis_proposals(proposal_id, analysis_uid, revision_id) VALUES(?, ?, ?) "
        "ON CONFLICT(proposal_id) DO NOTHING", {proposal, uid, revision});
    tx.commit();
}
json PiAnalysisContextStore::proposals(const std::string& uid) const {
    json items = json::array();
    for (const auto& row : db_->query("SELECT proposal_id, revision_id FROM analysis_proposals WHERE analysis_uid = ? ORDER BY rowid DESC LIMIT 100", {uid}))
        items.push_back({{"proposal_id", pi_sql_text(row[0])}, {"revision_id", pi_sql_text(row[1])}});
    return items;
}
} // namespace tile_compile::pi
