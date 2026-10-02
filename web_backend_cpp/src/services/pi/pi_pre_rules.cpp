#include "services/pi/pi_pre_rules.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace tile_compile::pi {

using nlohmann::json;

std::vector<std::string> PreRunCandidates::allowed_ids() const {
    std::vector<std::string> ids;
    for (const auto& c : applicable) ids.push_back(c["candidate_id"].get<std::string>());
    return ids;
}

PreRunCandidates build_pre_run_candidates(const json& state, const json& current_config, const DecisionPolicy& policy,
                                          const DecisionCatalog& catalog, const ConfigValidator& validate_config) {
    PreRunCandidates out;
    out.catalog_version = catalog.version;
    out.policy_version = policy.version;

    std::vector<const json*> ordered;
    for (const auto& c : catalog.candidates) ordered.push_back(&c);
    std::sort(ordered.begin(), ordered.end(),
              [](const json* a, const json* b) { return (*a)["candidate_id"].get<std::string>() < (*b)["candidate_id"].get<std::string>(); });

    for (const json* c : ordered) {
        const std::string id = (*c)["candidate_id"].get<std::string>();
        // Baselines carry no patch: always selectable, even when the state blocks every real
        // candidate (they are how "nothing to change" / "not enough evidence" is expressed).
        if ((*c)["updates"].empty()) {
            out.applicable.push_back({{"candidate_id", id}, {"candidate_version", (*c)["candidate_version"]},
                                      {"group", (*c)["group"]}, {"experimental", false}, {"requires_review", false},
                                      {"updates", json::array()}, {"evidence_refs", json::array()},
                                      {"rationale", json::object()}});
            continue;
        }
        json candidate = {{"candidate_id", id}, {"candidate_version", (*c)["candidate_version"]}, {"updates", (*c)["updates"]}};
        const CandidateValidation v = validate_decision_candidate(candidate, state, current_config, policy, catalog, validate_config);
        if (v.ok) {
            out.applicable.push_back({{"candidate_id", id}, {"candidate_version", (*c)["candidate_version"]},
                                      {"group", (*c)["group"]}, {"experimental", v.experimental},
                                      {"requires_review", v.requires_review}, {"updates", v.updates},
                                      {"evidence_refs", v.evidence_refs}, {"rationale", v.rationale}});
        } else {
            out.excluded.push_back({{"candidate_id", id}, {"reasons", v.reasons}});
        }
    }
    return out;
}

json provider_candidate_info(const PreRunCandidates& candidates, const DecisionCatalog& catalog) {
    json descriptions = json::object(), facts = json::object();
    for (const auto& a : candidates.applicable) {
        if (a["updates"].empty()) continue;
        const std::string id = a["candidate_id"].get<std::string>();
        for (const auto& entry : catalog.candidates) {
            if (entry.value("candidate_id", std::string()) != id) continue;
            std::string text = entry.value("provider_description", std::string());
            if (entry.contains("grid_value")) {
                std::string v = entry["grid_value"].dump();
                text += " Chosen level: " + v + (entry.contains("grid_unit") ? " " + entry["grid_unit"].get<std::string>() : std::string()) + ".";
            }
            descriptions[id] = text;
        }
        json f = json::object();
        // Static, catalog-held measured evidence (already validated as plain scalars at load time).
        for (const auto& entry : catalog.candidates)
            if (entry.value("candidate_id", std::string()) == id && entry.contains("provider_facts") && entry["provider_facts"].is_object())
                for (auto it = entry["provider_facts"].begin(); it != entry["provider_facts"].end(); ++it) f[it.key()] = it.value();
        const json params = a.value("rationale", json::object()).value("params", json::object());
        for (auto it = params.begin(); it != params.end(); ++it) {
            const json& v = it.value();
            const std::string sv = v.is_string() ? v.get<std::string>() : std::string();
            const bool plain = v.is_boolean() || (v.is_number() && std::isfinite(v.get<double>())) ||
                               (v.is_string() && sv.size() <= 32 && std::all_of(sv.begin(), sv.end(), [](unsigned char ch) { return std::isalnum(ch) || ch == '_' || ch == ' ' || ch == '-'; }));
            if (plain) f[it.key()] = v;
        }
        if (!f.empty()) facts[id] = f;
    }
    return {{"descriptions", descriptions}, {"facts", facts}};
}

} // namespace tile_compile::pi
