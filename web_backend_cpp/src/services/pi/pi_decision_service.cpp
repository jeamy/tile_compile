#include "services/pi/pi_decision_service.hpp"

#include "services/pi/pi_json_io.hpp"
#include "services/pi/pi_pre_rules.hpp"

#include <algorithm>
#include <stdexcept>

namespace tile_compile::pi {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

PreRunDecisionInputs to_inputs(const AdviceRequest& r) {
    PreRunDecisionInputs in;
    in.scan = r.scan;
    in.metrics = r.metrics;
    in.base_config = r.base_config;
    in.locked_paths = r.locked_paths;
    in.session_context = r.session_context;
    in.capabilities = r.capabilities;
    in.dataset_manifest = r.dataset_manifest;
    in.scan_id = r.scan_id;
    in.software_version = r.software_version;
    return in;
}

json policy_snapshot(const DecisionPolicy& p) {
    return {{"version", p.version},
            {"allow_experimental", p.allow_experimental},
            {"min_measurement_coverage", p.min_measurement_coverage ? json(*p.min_measurement_coverage) : json(nullptr)},
            {"min_metric_agreement", p.min_metric_agreement ? json(*p.min_metric_agreement) : json(nullptr)}};
}

DecisionPolicy policy_from_snapshot(const json& s) {
    DecisionPolicy p;
    p.version = s.value("version", std::string(kDecisionPolicyVersion));
    p.allow_experimental = s.value("allow_experimental", false);
    if (s.contains("min_measurement_coverage") && s["min_measurement_coverage"].is_number()) p.min_measurement_coverage = s["min_measurement_coverage"].get<double>();
    if (s.contains("min_metric_agreement") && s["min_metric_agreement"].is_number()) p.min_metric_agreement = s["min_metric_agreement"].get<double>();
    return p;
}

json comparison_of(const json& updates) {
    json out = json::array();
    for (const auto& u : updates)
        out.push_back({{"path", u["path"]}, {"current", u["old_value"]}, {"proposed", u["value"]},
                       {"changed", !json_values_equal(u["old_value"], u["value"])}});
    return out;
}

json unavailable_response(const std::string& code, const std::string& status = "unavailable") {
    return {{"status", status}, {"error_code", code}, {"model_reported", nullptr}, {"selection", nullptr}};
}

} // namespace

DecisionService::DecisionService(fs::path decisions_dir, DecisionCatalog catalog, DecisionServiceDeps deps, std::string qsv)
    : dir_(std::move(decisions_dir)), catalog_(std::move(catalog)), deps_(std::move(deps)), question_set_version_(std::move(qsv)) {
    if (!deps_.sidecar || !deps_.now_iso || !deps_.new_id) throw std::invalid_argument("DecisionService: incomplete dependencies");
}

void DecisionService::event(const std::string& id, const std::string& name, const json& extra) const {
    try {
        json e = {{"ts", deps_.now_iso()}, {"event", name}};
        for (auto it = extra.begin(); it != extra.end(); ++it) e[it.key()] = it.value();
        append_jsonl(pdir(id) / "events.jsonl", e);
    } catch (const std::exception&) {
        // observability only
    }
}

DecisionPolicy DecisionService::policy_for_mode(const std::string& mode, bool allow_experimental_flag) const {
    DecisionPolicy p;
    // shadow records what the model would say incl. experimental candidates (nothing is presented);
    // suggest shows experimental candidates only when explicitly enabled.
    p.allow_experimental = mode == "shadow" || (mode == "suggest" && allow_experimental_flag);
    json override_json;
    if (auto o = read_json_file_opt(dir_ / "policy_override.json"); o && o->is_object()) {
        override_json = *o;
        if (o->contains("min_measurement_coverage") && (*o)["min_measurement_coverage"].is_number()) p.min_measurement_coverage = (*o)["min_measurement_coverage"].get<double>();
        if (o->contains("min_metric_agreement") && (*o)["min_metric_agreement"].is_number()) p.min_metric_agreement = (*o)["min_metric_agreement"].get<double>();
        p.version = std::string(kDecisionPolicyVersion) + "+override:" + sha256_prefixed(canonical_json_dump(override_json)).substr(7, 8);
    }
    return p;
}

std::string DecisionService::create() {
    const std::string id = deps_.new_id();
    if (id.empty() || id.find('/') != std::string::npos || id.find("..") != std::string::npos) throw std::runtime_error("invalid proposal id");
    write_json_file_atomic(pdir(id) / "status.json", {{"proposal_id", id}, {"state", "running"}, {"created_at", deps_.now_iso()}});
    event(id, "created");
    return id;
}

void DecisionService::run(const std::string& id, const AdviceRequest& request) {
    json status = read_json_file_opt(pdir(id) / "status.json").value_or(json{{"proposal_id", id}, {"created_at", deps_.now_iso()}});
    try {
        json sidecar_status;
        bool sidecar_ok = true;
        try {
            sidecar_status = deps_.sidecar("GET", "/decisions/status", json());
        } catch (const std::exception&) {
            sidecar_ok = false;
        }
        const std::string mode = sidecar_ok ? sidecar_status.value("mode", std::string("off")) : std::string("off");
        const bool flag = sidecar_ok && sidecar_status.value("allow_experimental_suggestions", false);
        const bool has_key = sidecar_ok && sidecar_status.value("has_api_key", false);
        const std::string model = sidecar_ok ? sidecar_status.value("model", std::string()) : std::string();
        const DecisionPolicy policy = policy_for_mode(mode, flag);

        const PreRunDecisionResult sr = build_pre_run_decision_state(to_inputs(request));
        const PreRunCandidates cands = build_pre_run_candidates(sr.state, request.base_config, policy, catalog_, deps_.validate_config);
        write_json_file_atomic(pdir(id) / "state.json", {{"state", sr.state}, {"state_hash", sr.state_hash}, {"findings", sr.findings},
                                                         {"provider_projection", sr.provider_projection}});
        write_json_file_atomic(pdir(id) / "source.json", {{"input_path", request.scan.value("input_path", std::string())},
                                                            {"dataset_manifest", request.dataset_manifest}});
        write_json_file_atomic(pdir(id) / "candidates.json", {{"applicable", cands.applicable}, {"excluded", cands.excluded},
                                                              {"catalog_version", cands.catalog_version}, {"policy", policy_snapshot(policy)}});

        bool has_real = false;
        for (const auto& a : cands.applicable) has_real = has_real || !a["updates"].empty();

        json response;
        bool model_called = false, synthetic_baseline = false;
        if (!sidecar_ok) response = unavailable_response("sidecar_unreachable");
        else if (mode == "off") response = unavailable_response("mode_off");
        else if (!has_key) response = unavailable_response("no_api_key");
        else if (!has_real) {
            // Nothing but the baselines is applicable (e.g. thresholds not frozen): there is no decision
            // to ask a model for, so no call is made and nothing is attributed to the model.
            response = {{"status", "ok"}, {"selection", {{"candidate_id", "keep_current"}, {"probabilities", json::object()}}}, {"model_reported", nullptr}};
            synthetic_baseline = true;
        } else {
            json req = {{"request_id", id}, {"state_hash", sr.state_hash}, {"question_set_version", question_set_version_},
                        {"state_projection", sr.provider_projection}, {"allowed_candidates", cands.allowed_ids()}};
            const json info = provider_candidate_info(cands, catalog_);
            if (!info["descriptions"].empty()) req["candidate_descriptions"] = info["descriptions"];
            if (!info["facts"].empty()) req["candidate_facts"] = info["facts"];
            write_json_file_atomic(pdir(id) / "request.json", req);
            try {
                response = deps_.sidecar("POST", "/decisions", req);
                model_called = true;
            } catch (const SidecarHttpError& e) {
                response = unavailable_response(e.status == 400 ? "request_rejected" : "sidecar_http_" + std::to_string(e.status),
                                                e.status == 400 ? "invalid_response" : "unavailable");
            } catch (const std::exception&) {
                response = unavailable_response("sidecar_unreachable");
            }
        }
        event(id, "provider_response", {{"status", response.value("status", std::string())}, {"error_code", response.value("error_code", std::string())},
                                        {"model_called", model_called}});

        ResolveContext ctx;
        ctx.proposal_id = id;
        ctx.created_at = deps_.now_iso();
        ctx.state_hash = sr.state_hash;
        ctx.question_set_version = question_set_version_;
        ctx.model_requested = synthetic_baseline || model.empty() ? std::string("none") : model;
        ctx.model_reported = response.contains("model_reported") ? response["model_reported"] : json(nullptr);
        json proposal = resolve_decision(response, sr.state, request.base_config, policy, catalog_, deps_.validate_config, ctx);
        if (synthetic_baseline) proposal["reason_codes"].push_back("no_applicable_candidates");

        write_json_file_atomic(pdir(id) / "response.json", response);
        write_json_file_atomic(pdir(id) / "proposal.json", proposal);
        status["state"] = "done";
        status["finished_at"] = deps_.now_iso();
        status["mode"] = mode;
        status["model_called"] = model_called;
        status["synthetic_baseline"] = synthetic_baseline;
        status["policy"] = policy_snapshot(policy);
        write_json_file_atomic(pdir(id) / "status.json", status);
        event(id, "done", {{"proposal_status", proposal["status"]}, {"candidate_id", proposal["candidate_id"]}});
    } catch (const std::exception& e) {
        status["state"] = "failed";
        status["error"] = e.what();
        status["finished_at"] = deps_.now_iso();
        try { write_json_file_atomic(pdir(id) / "status.json", status); } catch (const std::exception&) {}
        event(id, "failed", {{"error", e.what()}});
    }
}

std::optional<json> DecisionService::view(const std::string& id) const {
    if (id.empty() || id.find('/') != std::string::npos || id.find("..") != std::string::npos) return std::nullopt;
    const auto status = read_json_file_opt(pdir(id) / "status.json");
    if (!status) return std::nullopt;
    json out = {{"proposal_id", id}, {"state", status->value("state", std::string("unknown"))}, {"created_at", status->value("created_at", std::string())}};
    if (status->contains("error")) out["error"] = (*status)["error"];
    if (out["state"] != "done") return out;

    const bool shadow = status->value("mode", std::string()) == "shadow";
    json proposal = read_json_file_opt(pdir(id) / "proposal.json").value_or(json::object());
    const json cands = read_json_file_opt(pdir(id) / "candidates.json").value_or(json::object());
    const json st = read_json_file_opt(pdir(id) / "state.json").value_or(json::object());
    out["mode"] = status->value("mode", std::string());
    out["shadow"] = shadow;
    out["model_called"] = status->value("model_called", false);
    out["synthetic_baseline"] = status->value("synthetic_baseline", false);
    // Shadow records what the model said but never exposes an applicable patch.
    json shown = proposal;
    if (shadow) shown["updates"] = json::array();
    out["proposal"] = shown;
    out["comparison"] = shadow ? json::array() : comparison_of(proposal.value("updates", json::array()));
    json evidence = json::object();
    for (const auto& a : cands.value("applicable", json::array()))
        if (a.value("candidate_id", std::string()) == proposal.value("candidate_id", std::string())) {
            evidence = {{"evidence_refs", a["evidence_refs"]}, {"rationale", a["rationale"]}, {"experimental", a["experimental"]}};
        }
    out["evidence"] = evidence;
    json offered = json::array();
    for (const auto& a : cands.value("applicable", json::array()))
        offered.push_back({{"candidate_id", a["candidate_id"]}, {"experimental", a["experimental"]}, {"requires_review", a["requires_review"]}});
    out["offered"] = offered;
    out["excluded"] = cands.value("excluded", json::array());
    out["findings"] = st.value("findings", json::array());
    out["blocking_findings"] = st.contains("state") ? st["state"].value("blocking_findings", json::array()) : json::array();
    out["state_hash"] = st.value("state_hash", std::string());
    return out;
}

ServiceResult DecisionService::apply(const std::string& id, const AdviceRequest& current) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto fail = [](int code, const std::string& err, json extra = json::object()) {
        json b = {{"error", true}, {"code", err}};
        for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
        return ServiceResult{code, b};
    };
    if (id.empty() || id.find('/') != std::string::npos || id.find("..") != std::string::npos) return fail(404, "NOT_FOUND");
    auto status = read_json_file_opt(pdir(id) / "status.json");
    if (!status) return fail(404, "NOT_FOUND");
    if (status->value("state", std::string()) != "done") return fail(409, "NOT_READY");
    if (status->value("mode", std::string()) == "shadow") return fail(403, "SHADOW_MODE");
    auto proposal_opt = read_json_file_opt(pdir(id) / "proposal.json");
    if (!proposal_opt) return fail(404, "NOT_FOUND");
    json proposal = *proposal_opt;
    const std::string pstatus = proposal.value("status", std::string());
    if (pstatus != "validated" && pstatus != "presented" && pstatus != "applied_to_draft")
        return fail(409, "NOT_APPLICABLE", {{"status", pstatus}});

    if (!current.base_config.is_object()) return fail(400, "DRAFT_MISSING");
    const DecisionPolicy policy = policy_from_snapshot(status->value("policy", json::object()));

    if (pstatus == "applied_to_draft") {
        // A repeat is idempotent only for the exact patched draft and the same scan/locks/dataset.
        const auto saved_state = read_json_file_opt(pdir(id) / "state.json");
        if (!saved_state || !saved_state->contains("state")) return fail(409, "PROPOSAL_STALE");
        const PreRunDecisionResult now = build_pre_run_decision_state(to_inputs(current));
        if (now.state["identity"]["config_hash"] != proposal.value("config_hash_after", std::string()))
            return fail(409, "DRAFT_CHANGED");
        json original = (*saved_state)["state"];
        json current_state = now.state;
        original.erase("base_config");
        current_state.erase("base_config");
        original["identity"].erase("config_hash");
        current_state["identity"].erase("config_hash");
        if (original != current_state) return fail(409, "PROPOSAL_STALE");
        return {200, {{"proposal", proposal}, {"updates", proposal["updates"]}, {"patched_config", current.base_config}, {"already_applied", true}}};
    }

    const PreRunDecisionResult sr = build_pre_run_decision_state(to_inputs(current));
    const auto stale = stale_reasons(proposal, sr.state, sr.state_hash, policy, catalog_);
    if (!stale.empty()) {
        event(id, "apply_stale", {{"reasons", stale}});
        return fail(409, "PROPOSAL_STALE", {{"reasons", stale}});
    }
    const json candidate = {{"candidate_id", proposal["candidate_id"]}, {"candidate_version", proposal["candidate_version"]},
                            {"updates", proposal["updates"]}};
    const CandidateValidation v = validate_decision_candidate(candidate, sr.state, current.base_config, policy, catalog_, deps_.validate_config);
    if (!v.ok) {
        event(id, "apply_rejected", {{"reasons", v.reasons}});
        return fail(422, "VALIDATION_FAILED", {{"reasons", v.reasons}});
    }
    proposal["status"] = "applied_to_draft";
    proposal["applied_at"] = deps_.now_iso();
    proposal["config_hash_before"] = sr.state["identity"]["config_hash"];
    proposal["config_hash_after"] = sha256_prefixed(canonical_json_dump(v.merged_config));
    write_json_file_atomic(pdir(id) / "proposal.json", proposal);
    event(id, "applied_to_draft", {{"config_hash_after", proposal["config_hash_after"]}});
    return {200, {{"proposal", proposal}, {"updates", v.updates}, {"patched_config", v.merged_config}, {"already_applied", false}}};
}

} // namespace tile_compile::pi
