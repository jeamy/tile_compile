#include "services/pi/pi_decision_writes.hpp"

#include "services/pi/pi_decision_record_store.hpp"

namespace tile_compile::pi {

using nlohmann::json;

nlohmann::json build_llm_proposal_record(const json& analysis, const std::string& analysis_id) {
    json alternatives = json::array();
    for (const auto& update : analysis.value("validated_updates", json::array())) {
        const auto path = update.value("path", std::string());
        if (path.empty()) continue;
        json paths = json::array();
        paths.push_back({{"path", path}, {"from", nullptr}, {"to", nullptr}});
        alternatives.push_back({
            {"id", path}, {"label", path}, {"probability", nullptr}, {"selected", false},
            {"rejected_by", nullptr}, {"subject", {{"paths", paths}}}
        });
    }
    json basis = json::array();
    basis.push_back({{"source", "llm_hypothesis"}, {"ref", analysis_id}});
    return {
        {"schema_version", kDecisionRecordSchemaVersion},
        {"kind", "llm_proposal"}, {"actor", "llm"}, {"origin", "live"},
        {"idempotency_key", "llm_proposal:" + analysis_id},
        {"context_ref", {{"context_id", "analysis:" + analysis_id}, {"analysis_id", analysis_id}}},
        {"alternatives", alternatives},
        {"llm", {{"model", analysis.value("model", std::string())},
                 {"usage", {{"input_tokens", nullptr}, {"output_tokens", nullptr}, {"cost_usd", nullptr}}}}},
        {"rationale", {
            {"user", {{"reason_codes", json::array()}, {"catalog_version", ""}, {"text", ""}, {"no_reason_given", false}}},
            {"basis", basis}
        }}
    };
}

nlohmann::json build_config_apply_record(const ConfigApplyEvent& event) {
    json paths = json::array();
    for (const auto& path : event.paths) paths.push_back({{"path", path}, {"from", nullptr}, {"to", nullptr}});
    json record = {
        {"schema_version", kDecisionRecordSchemaVersion},
        {"kind", "config_apply"}, {"actor", "user"}, {"origin", "live"},
        {"idempotency_key", event.idempotency_key},
        {"action_plan_id", event.action_plan_id.empty() ? json(nullptr) : json(event.action_plan_id)},
        {"preview_id", event.preview_id.empty() ? json(nullptr) : json(event.preview_id)},
        {"parent_decision_id", event.parent_decision_id.empty() ? json(nullptr) : json(event.parent_decision_id)},
        {"context_ref", {{"context_id", event.context_id}, {"revision_id", event.revision_id},
                         {"config_sha256", event.config_sha256}}},
        {"subject", {{"paths", paths}, {"op", "config.set"}}},
        // The user did not state a reason; the record says so instead of guessing one.
        {"rationale", {
            {"user", {{"reason_codes", json::array()}, {"catalog_version", ""}, {"text", ""}, {"no_reason_given", true}}},
            {"basis", json::array()}
        }}
    };
    return record;
}

std::string find_llm_proposal_id(const PiDecisionRecordStore& store, const std::string& analysis_id) {
    const auto items = store.list({{"kind", "llm_proposal"}, {"context_id", "analysis:" + analysis_id}}, 1);
    if (!items.is_array() || items.empty()) return "";
    return items[0].value("decision_id", std::string());
}

} // namespace tile_compile::pi
