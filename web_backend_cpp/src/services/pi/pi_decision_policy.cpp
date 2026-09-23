#include "services/pi/pi_decision_policy.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>

namespace tile_compile::pi {

using nlohmann::json;

namespace {

std::vector<std::string> split_path(const std::string& dotted) {
    std::vector<std::string> parts;
    std::stringstream ss(dotted);
    std::string part;
    while (std::getline(ss, part, '.')) parts.push_back(part);
    return parts;
}

bool has_prefix_path(const std::string& path, const std::string& prefix) {
    if (prefix.empty()) return false;
    if (prefix.back() == '.') return path.compare(0, prefix.size(), prefix) == 0;
    return path == prefix || (path.size() > prefix.size() && path.compare(0, prefix.size(), prefix) == 0 && path[prefix.size()] == '.');
}

const json* find_catalog_candidate(const DecisionCatalog& catalog, const std::string& id) {
    for (const auto& c : catalog.candidates)
        if (c.value("candidate_id", std::string()) == id) return &c;
    return nullptr;
}

struct EvidenceResult {
    bool ok = false;
    std::vector<std::string> reasons;
    json refs = json::array();
    json params = json::object();
};

std::string fmt_key(const std::string& k) { return k; }

EvidenceResult resolve_measurement_coverage(const json& state, const DecisionPolicy& policy) {
    EvidenceResult r;
    const json cov = state.value("coverage", json::object());
    const long measured = cov.value("measured", 0);
    const long read_ok = cov.value("read_ok", 0);
    r.refs = json::array({"coverage.read_ok", "coverage.measured"});
    if (measured <= 0) { r.reasons.push_back("evidence_unavailable:measurement_coverage"); return r; }
    const double ratio = static_cast<double>(read_ok) / static_cast<double>(measured);
    r.params["measurement_coverage"] = ratio;
    if (!policy.min_measurement_coverage) { r.reasons.push_back("policy_thresholds_not_frozen"); return r; }
    if (ratio < *policy.min_measurement_coverage) { r.reasons.push_back("evidence_below_threshold:measurement_coverage"); return r; }
    r.ok = true;
    return r;
}

// Agreement between the frame-quality metrics (median pairwise Spearman rho of background, noise,
// fwhm) is what makes adaptive weighting act: the weights follow leave-one-out metric correlation,
// not the size of the spread. It says the option will have an effect, never that the effect is good.
EvidenceResult resolve_metric_agreement(const json& state, const DecisionPolicy& policy) {
    EvidenceResult r;
    // Pooling groups with different acquisition conditions would blur the agreement; with several
    // readable groups there is no single honest number, so the evidence is unavailable.
    const json groups = state.value("groups", json::array());  // named: readable[] points into it
    std::vector<const json*> readable;
    for (const auto& g : groups)
        if (g.value("frame_count", 0) - g.value("frames_read_failed", 0) > 0) readable.push_back(&g);
    if (readable.empty()) { r.reasons.push_back("evidence_unavailable:metric_agreement"); return r; }
    if (readable.size() > 1) { r.reasons.push_back("mixed_groups_no_single_evidence"); return r; }
    const json& g = *readable.front();
    const std::string gid = g.value("group_id", std::string("?"));
    r.refs.push_back("groups[" + gid + "].metric_agreement.median");
    const json med = g.value("metric_agreement", json::object()).value("median", json::object());
    if (med.value("status", std::string()) != "valid" || !med.contains("value") || !med["value"].is_number()) {
        r.reasons.push_back("evidence_unavailable:metric_agreement");
        return r;
    }
    const double agreement = med["value"].get<double>();
    r.params["metric_agreement"] = agreement;
    if (!policy.min_metric_agreement) { r.reasons.push_back("policy_thresholds_not_frozen"); return r; }
    if (agreement < *policy.min_metric_agreement) { r.reasons.push_back("evidence_below_threshold:metric_agreement"); return r; }
    r.ok = true;
    return r;
}

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

} // namespace

bool json_values_equal(const json& a, const json& b) {
    if (a.is_number() && b.is_number()) return a.get<double>() == b.get<double>();
    return a == b;
}

json config_get(const json& config, const std::string& dotted_path) {
    const json* cur = &config;
    for (const auto& part : split_path(dotted_path)) {
        if (!cur->is_object() || !cur->contains(part)) return nullptr;
        cur = &(*cur)[part];
    }
    return *cur;
}

void config_set(json& config, const std::string& dotted_path, const json& value) {
    json* cur = &config;
    const auto parts = split_path(dotted_path);
    if (parts.empty()) throw std::invalid_argument("empty config path");
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
        if (!cur->is_object()) throw std::invalid_argument("config path crosses a non-object: " + dotted_path);
        if (!cur->contains(parts[i])) (*cur)[parts[i]] = json::object();
        cur = &(*cur)[parts[i]];
    }
    if (!cur->is_object()) throw std::invalid_argument("config path crosses a non-object: " + dotted_path);
    (*cur)[parts.back()] = value;
}

bool path_is_protected(const DecisionCatalog& catalog, const std::string& path) {
    for (const auto& prefix : catalog.protected_prefixes)
        if (has_prefix_path(path, prefix)) return true;
    return false;
}

bool path_is_locked(const std::vector<std::string>& locked_paths, const std::string& path) {
    for (const auto& lock : locked_paths) {
        // Exact, lock covering the path, or the path covering (i.e. overwriting) a locked leaf.
        if (has_prefix_path(path, lock) || has_prefix_path(lock, path)) return true;
    }
    return false;
}

DecisionCatalog load_decision_catalog(const json& candidates_v1, const json& protected_paths_v1) {
    if (!candidates_v1.is_object() || candidates_v1.value("schema_version", std::string()) != "pi.candidate-catalog.v1")
        throw std::invalid_argument("candidate catalog: wrong or missing schema_version");
    if (!protected_paths_v1.is_object() || protected_paths_v1.value("schema_version", std::string()) != "pi.protected-paths.v1")
        throw std::invalid_argument("protected paths: wrong or missing schema_version");
    DecisionCatalog c;
    c.version = candidates_v1.value("catalog_version", 0);
    if (c.version < 1 || !candidates_v1.contains("candidates") || !candidates_v1["candidates"].is_array())
        throw std::invalid_argument("candidate catalog: missing catalog_version/candidates");
    for (const auto& p : protected_paths_v1.value("protected_path_prefixes", json::array())) {
        if (!p.is_object() || !p.contains("prefix") || !p["prefix"].is_string() || p["prefix"].get<std::string>().empty())
            throw std::invalid_argument("protected paths: malformed prefix entry");
        c.protected_prefixes.push_back(p["prefix"].get<std::string>());
    }
    if (c.protected_prefixes.empty()) throw std::invalid_argument("protected paths: empty list (fail closed)");
    std::set<std::string> ids;
    for (const auto& cand : candidates_v1["candidates"]) {
        const std::string id = cand.value("candidate_id", std::string());
        if (id.empty() || !cand.contains("candidate_version") || !cand.contains("group") || !cand.contains("updates") || !cand["updates"].is_array())
            throw std::invalid_argument("candidate catalog: malformed candidate entry");
        if (!ids.insert(id).second) throw std::invalid_argument("candidate catalog: duplicate id " + id);
        for (const auto& u : cand["updates"]) {
            if (!u.is_object() || !u.contains("path") || !u.contains("value") || !u["path"].is_string())
                throw std::invalid_argument("candidate catalog: malformed update in " + id);
            if (path_is_protected(c, u["path"].get<std::string>()))
                throw std::invalid_argument("candidate catalog: " + id + " targets protected path " + u["path"].get<std::string>());
        }
        c.candidates.push_back(cand);
    }
    for (const char* required : {"keep_current", "insufficient_evidence"})
        if (!ids.count(required)) throw std::invalid_argument(std::string("candidate catalog: missing baseline candidate ") + required);
    return c;
}

CandidateValidation validate_decision_candidate(const json& candidate, const json& state, const json& current_config,
                                                const DecisionPolicy& policy, const DecisionCatalog& catalog,
                                                const ConfigValidator& validate_config) {
    CandidateValidation out;
    std::set<std::string> reasons;
    out.candidate_id = candidate.is_object() ? candidate.value("candidate_id", std::string()) : std::string();

    auto finish = [&](bool config_ran_ok) {
        out.reasons.assign(reasons.begin(), reasons.end());
        auto any = [&](auto pred) { return std::any_of(reasons.begin(), reasons.end(), pred); };
        out.schema_ok = !any([](const std::string& r) {
            return r == "path_not_allowlisted" || r == "value_not_allowlisted" || r == "incomplete_group" ||
                   r == "candidate_version_mismatch" || r == "unknown_candidate" || r == "malformed_candidate" ||
                   r == "config_path_conflict";
        });
        out.policy_ok = !any([](const std::string& r) {
            return r == "path_protected" || r == "path_locked" || starts_with(r, "blocked:") || r == "experimental_not_enabled" ||
                   starts_with(r, "precondition_failed:") || starts_with(r, "unknown_precondition:") || r == "already_active" ||
                   r == "old_value_mismatch";
        });
        out.evidence_ok = !any([](const std::string& r) {
            return starts_with(r, "evidence_") || r == "policy_thresholds_not_frozen" || r == "mixed_groups_no_single_evidence";
        });
        out.config_ok = config_ran_ok;
        out.ok = reasons.empty() && config_ran_ok;
        return out;
    };

    const json* entry = out.candidate_id.empty() ? nullptr : find_catalog_candidate(catalog, out.candidate_id);
    if (!candidate.is_object() || !entry) {
        reasons.insert(candidate.is_object() ? "unknown_candidate" : "malformed_candidate");
        return finish(false);
    }
    out.experimental = entry->value("applicability", std::string("released")) == "experimental_only";
    out.requires_review = entry->value("requires_review", true) || out.experimental;
    if (candidate.value("candidate_version", 0) != entry->value("candidate_version", -1)) reasons.insert("candidate_version_mismatch");

    // ---- allowlist: the group must equal the catalog entry exactly (no new path/value, no partial) ----
    const json& allowed = (*entry)["updates"];
    if (!candidate.contains("updates") || !candidate["updates"].is_array()) {
        reasons.insert("malformed_candidate");
    } else {
        std::set<std::string> seen;
        for (const auto& u : candidate["updates"]) {
            const std::string path = u.is_object() && u.contains("path") && u["path"].is_string() ? u["path"].get<std::string>() : std::string();
            const json* match = nullptr;
            for (const auto& a : allowed) if (a["path"] == path) match = &a;
            if (path.empty() || !u.contains("value") || !match) { reasons.insert("path_not_allowlisted"); continue; }
            seen.insert(path);
            if (!json_values_equal((*match)["value"], u["value"])) reasons.insert("value_not_allowlisted");
        }
        for (const auto& a : allowed) if (!seen.count(a["path"].get<std::string>())) reasons.insert("incomplete_group");
    }

    const bool has_updates = !allowed.empty();
    const std::vector<std::string> locks = [&] {
        std::vector<std::string> v;
        for (const auto& l : state.value("locked_paths", json::array())) if (l.is_string()) v.push_back(l);
        return v;
    }();
    for (const auto& a : allowed) {
        const std::string path = a["path"].get<std::string>();
        if (path_is_protected(catalog, path)) reasons.insert("path_protected");
        if (path_is_locked(locks, path)) reasons.insert("path_locked");
    }
    if (has_updates)
        for (const auto& f : state.value("blocking_findings", json::array()))
            if (f.is_string()) reasons.insert("blocked:" + f.get<std::string>());
    if (out.experimental && !policy.allow_experimental) reasons.insert("experimental_not_enabled");

    // ---- preconditions (unknown ones fail closed) ----
    for (const auto& p : entry->value("preconditions", json::array())) {
        const std::string pre = p.is_string() ? p.get<std::string>() : std::string();
        if (pre == "base_config_valid") {
            if (!current_config.is_object()) reasons.insert("precondition_failed:base_config_valid");
        } else if (pre == "adaptive_weights_is_false") {
            if (config_get(current_config, "global_metrics.adaptive_weights") == json(true)) reasons.insert("already_active");
        } else if (starts_with(pre, "path_unlocked:")) {
            if (path_is_locked(locks, pre.substr(std::string("path_unlocked:").size()))) reasons.insert("path_locked");
        } else {
            reasons.insert("unknown_precondition:" + pre);
        }
    }

    // ---- evidence: resolved against the SAME state, never taken from the model ----
    for (const auto& key : entry->value("required_evidence", json::array())) {
        const std::string k = key.get<std::string>();
        EvidenceResult ev;
        if (k == "measurement_coverage") ev = resolve_measurement_coverage(state, policy);
        else if (k == "metric_agreement") ev = resolve_metric_agreement(state, policy);
        else { ev.reasons.push_back("evidence_unavailable:" + fmt_key(k)); }
        for (const auto& r : ev.reasons) reasons.insert(r);
        for (const auto& ref : ev.refs) out.evidence_refs.push_back(ref);
        for (auto it = ev.params.begin(); it != ev.params.end(); ++it) out.rationale["params"][it.key()] = it.value();
    }
    if (has_updates) out.rationale["text_key"] = "pi.jev.rationale." + out.candidate_id + ".v" + std::to_string(entry->value("candidate_version", 0));

    // ---- old_value (only when a stored proposal is being re-validated) + no-op detection ----
    if (candidate.contains("updates") && candidate["updates"].is_array())
        for (const auto& u : candidate["updates"])
            if (u.is_object() && u.contains("old_value") && u.contains("path") && u["path"].is_string() &&
                !json_values_equal(config_get(current_config, u["path"]), u["old_value"]))
                reasons.insert("old_value_mismatch");
    if (has_updates && current_config.is_object()) {
        bool all_same = true;
        for (const auto& a : allowed) all_same = all_same && json_values_equal(config_get(current_config, a["path"]), a["value"]);
        if (all_same) reasons.insert("already_active");
    }

    // ---- whole-group config validation, only when nothing structural already failed ----
    bool config_ok = false;
    if (reasons.empty()) {
        json merged = current_config;
        try {
            for (const auto& a : allowed) config_set(merged, a["path"].get<std::string>(), a["value"]);
        } catch (const std::invalid_argument&) {
            reasons.insert("config_path_conflict");
            return finish(false);
        }
        if (!validate_config) {
            reasons.insert("config_validator_missing");
        } else {
            const ConfigCheck check = validate_config(merged);
            if (check.valid) config_ok = true;
            else reasons.insert("config_invalid");
        }
        if (config_ok) {
            out.merged_config = std::move(merged);
            for (const auto& a : allowed)
                out.updates.push_back({{"path", a["path"]}, {"old_value", config_get(current_config, a["path"])},
                                       {"value", a["value"]}, {"group_id", entry->value("group", std::string())}});
        }
    }
    return finish(config_ok);
}

json resolve_decision(const json& response, const json& state, const json& current_config, const DecisionPolicy& policy,
                      const DecisionCatalog& catalog, const ConfigValidator& validate_config, const ResolveContext& ctx) {
    json proposal = {
        {"proposal_id", ctx.proposal_id},
        {"domain", "pre_run"},
        {"identity", {{"dataset_fingerprint", state["identity"]["dataset_fingerprint"]}, {"scan_hash", state["identity"]["scan_hash"]},
                      {"config_hash", state["identity"]["config_hash"]}, {"locks_hash", state["identity"]["locks_hash"]}}},
        {"state_hash", ctx.state_hash},
        {"candidate_id", "keep_current"},
        {"candidate_version", 1},
        {"question_set_version", ctx.question_set_version},
        {"policy_version", policy.version},
        {"model", {{"requested", ctx.model_requested}, {"provider_reported", ctx.model_reported}}},
        {"status", "rejected"},
        {"updates", json::array()},
        {"evidence_refs", json::array()},
        {"reason_codes", json::array()},
        {"review_required", false},
        {"validation", {{"schema_ok", false}, {"policy_ok", false}, {"evidence_ok", false}, {"config_ok", false}}},
        {"created_at", ctx.created_at}};

    auto reject = [&](const std::string& status, std::vector<std::string> reasons) {
        proposal["status"] = status;
        std::sort(reasons.begin(), reasons.end());
        proposal["reason_codes"] = reasons;
        return proposal;
    };

    const std::string rstatus = response.is_object() ? response.value("status", std::string()) : std::string();
    // The application error code (never a raw provider string) stays visible next to the generic reason.
    std::vector<std::string> provider_codes;
    if (response.is_object() && response.contains("error_code") && response["error_code"].is_string())
        provider_codes.push_back("provider:" + response["error_code"].get<std::string>());
    if (rstatus == "unavailable") {
        provider_codes.insert(provider_codes.end(), {"provider_unavailable", "validation_not_run"});
        return reject("unavailable", provider_codes);
    }
    if (rstatus != "ok" || !response.contains("selection") || !response["selection"].is_object() ||
        !response["selection"].contains("candidate_id") || !response["selection"]["candidate_id"].is_string()) {
        provider_codes.insert(provider_codes.end(), {"provider_invalid_response", "validation_not_run"});
        return reject("rejected", provider_codes);
    }

    const std::string chosen = response["selection"]["candidate_id"].get<std::string>();
    const json* entry = find_catalog_candidate(catalog, chosen);
    if (!entry) return reject("rejected", {"unknown_candidate", "validation_not_run"});
    proposal["candidate_id"] = chosen;
    proposal["candidate_version"] = entry->value("candidate_version", 1);

    if (chosen == "keep_current" || chosen == "insufficient_evidence") {
        proposal["validation"] = {{"schema_ok", true}, {"policy_ok", true}, {"evidence_ok", true}, {"config_ok", true}};
        proposal["status"] = chosen == "keep_current" ? "no_change" : "abstain";
        return proposal;
    }

    // The provider only names an id; the update group always comes from the catalog.
    json candidate = {{"candidate_id", chosen}, {"candidate_version", entry->value("candidate_version", 1)},
                      {"updates", (*entry)["updates"]}};
    const CandidateValidation v = validate_decision_candidate(candidate, state, current_config, policy, catalog, validate_config);
    proposal["validation"] = {{"schema_ok", v.schema_ok}, {"policy_ok", v.policy_ok}, {"evidence_ok", v.evidence_ok}, {"config_ok", v.config_ok}};
    proposal["evidence_refs"] = v.evidence_refs;
    proposal["review_required"] = v.requires_review;
    if (!v.ok) {
        std::vector<std::string> r = v.reasons;
        r.push_back("candidate_not_applicable");
        return reject("rejected", r);
    }
    proposal["status"] = "validated";
    proposal["updates"] = v.updates;
    proposal["reason_codes"] = json::array({"validated"});
    return proposal;
}

std::vector<std::string> stale_reasons(const json& proposal, const json& current_state, const std::string& current_state_hash,
                                       const DecisionPolicy& policy, const DecisionCatalog& catalog) {
    std::vector<std::string> r;
    const json& id = proposal.value("identity", json::object());
    const json& cur = current_state.value("identity", json::object());
    for (const char* k : {"dataset_fingerprint", "scan_hash", "config_hash", "locks_hash"})
        if (id.value(k, std::string()) != cur.value(k, std::string("<none>"))) r.push_back(std::string("stale:") + k);
    if (proposal.value("state_hash", std::string()) != current_state_hash) r.push_back("stale:state_hash");
    if (proposal.value("policy_version", std::string()) != policy.version) r.push_back("stale:policy_version");
    const json* entry = find_catalog_candidate(catalog, proposal.value("candidate_id", std::string()));
    if (!entry) r.push_back("stale:candidate_removed");
    else if (entry->value("candidate_version", -1) != proposal.value("candidate_version", 0)) r.push_back("stale:candidate_version");
    std::sort(r.begin(), r.end());
    return r;
}

} // namespace tile_compile::pi
