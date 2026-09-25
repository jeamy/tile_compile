#include "pi_decision_test_support.hpp"
#include "pi_json_schema_check.hpp"
#include "services/pi/pi_decision_service.hpp"
#include "services/pi/pi_json_io.hpp"

#include <cstdio>
#include <map>
#include <unistd.h>

using namespace pi_test;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
struct Harness {
    fs::path root, dir;
    json sidecar_status;
    std::vector<std::string> calls;       // "GET /x" / "POST /x"
    std::function<json(const json&)> on_decide;
    bool sidecar_down = false;
    int counter = 0;
    std::string clock = "2026-09-23T10:00:00Z";
    std::unique_ptr<DecisionService> svc;

    Harness(const fs::path& repo, const std::string& tag) {
        root = fs::temp_directory_path() / ("pi_decision_service_test_" + tag + "_" + std::to_string(::getpid()));
        fs::remove_all(root);
        dir = root / "pi_decisions";
        fs::create_directories(root);
        sidecar_status = {{"mode", "suggest"}, {"model", "typesafe/jev-1.13"}, {"has_api_key", true}, {"allow_experimental_suggestions", true}};
        rebuild(repo);
    }
    void rebuild(const fs::path& repo) {
        DecisionServiceDeps d;
        d.sidecar = [this](const std::string& method, const std::string& ep, const json& payload) -> json {
            calls.push_back(method + " " + ep);
            if (sidecar_down) throw std::runtime_error("connection refused");
            if (ep == "/decisions/status") return sidecar_status;
            if (!on_decide) throw std::runtime_error("no decide handler");
            return on_decide(payload);
        };
        d.validate_config = accepting_validator();
        d.now_iso = [this] { return clock; };
        d.new_id = [this] { return "dec_" + std::to_string(++counter); };
        svc = std::make_unique<DecisionService>(dir, load_real_catalog(repo), d);
    }
    int posts() const { return static_cast<int>(std::count(calls.begin(), calls.end(), "POST /decisions")); }
    void freeze_thresholds() {
        tile_compile::pi::write_json_file_atomic(dir / "policy_override.json", {{"min_measurement_coverage", 0.9}, {"min_metric_agreement", 0.05}});
    }
    ~Harness() { fs::remove_all(root); }
};

AdviceRequest advice(int frames = 10, bool adaptive = false) {
    auto in = inputs_for(spread_frames(frames), adaptive);
    AdviceRequest r;
    r.scan = in.scan; r.metrics = in.metrics; r.base_config = in.base_config; r.scan_id = "scan-1"; r.software_version = "test";
    r.base_config["runs_dir"] = "/home/x/runs";  // must never reach the provider
    return r;
}

json ok_answer(const json& req, const std::string& id, double p = 0.8) {
    return {{"request_id", req["request_id"]}, {"state_hash", req["state_hash"]}, {"status", "ok"}, {"model_requested", "typesafe/jev-1.13"},
            {"model_reported", "typesafe/jev-1.13-20260917"}, {"selection", {{"candidate_id", id}, {"probabilities", {{id, p}}}, {"provider_confidence", p}}},
            {"usage", {{"input_tokens", 10}, {"output_tokens", 2}, {"cost", 0.00001}}}, {"generation_id", "gen-dec-1"}};
}

std::map<std::string, std::string> tree(const fs::path& root) {
    std::map<std::string, std::string> m;
    if (!fs::exists(root)) return m;
    for (const auto& e : fs::recursive_directory_iterator(root)) if (e.is_regular_file()) m[fs::relative(e.path(), root).string()] = slurp_file(e.path());
    return m;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const fs::path repo = argc > 4 ? argv[4] : "..";
        const json proposal_schema = json::parse(slurp_file(repo / "web_backend_cpp/config/pi_decisions/schemas/pi.config-proposal.v1.schema.json"));
        auto valid = [&](const json& p) { const auto e = schema_check(proposal_schema, p, proposal_schema, "proposal"); expect_true(e.empty(), "proposal schema: " + e); };

        // ---- mode off: no provider call, honest "unavailable" ----
        {
            Harness h(repo, "off");
            h.sidecar_status["mode"] = "off";
            const auto id = h.svc->create();
            h.svc->run(id, advice());
            const auto v = *h.svc->view(id);
            expect_equal(v["state"].get<std::string>(), "done", "run finished");
            expect_equal(v["proposal"]["status"].get<std::string>(), "unavailable", "off -> unavailable");
            expect_true(v["proposal"]["reason_codes"].dump().find("provider:mode_off") != std::string::npos, "reason provider:mode_off");
            expect_equal(static_cast<long>(h.posts()), 0L, "no provider call in mode off");
            valid(v["proposal"]);
        }

        // ---- thresholds not frozen: only baselines, so nothing to ask ----
        {
            Harness h(repo, "baseline");
            const auto id = h.svc->create();
            h.svc->run(id, advice());
            const auto v = *h.svc->view(id);
            expect_equal(v["proposal"]["status"].get<std::string>(), "no_change", "baseline outcome");
            expect_true(v["synthetic_baseline"] == true && v["model_called"] == false, "nothing attributed to the model");
            expect_equal(v["proposal"]["model"]["requested"].get<std::string>(), "none", "model requested none");
            expect_true(v["proposal"]["reason_codes"].dump().find("no_applicable_candidates") != std::string::npos, "reason recorded");
            expect_equal(static_cast<long>(h.posts()), 0L, "no provider call");
            expect_true(v["excluded"].dump().find("policy_thresholds_not_frozen") != std::string::npos, "why the real candidate is excluded is visible");
            valid(v["proposal"]);
        }

        // ---- unreachable sidecar / no key / invalid answers ----
        {
            Harness h(repo, "down");
            h.sidecar_down = true;
            const auto id = h.svc->create();
            h.svc->run(id, advice());
            const auto v = *h.svc->view(id);
            expect_equal(v["proposal"]["status"].get<std::string>(), "unavailable", "sidecar down");
            expect_true(v["proposal"]["reason_codes"].dump().find("provider_unavailable") != std::string::npos, "reason");

            Harness k(repo, "nokey");
            k.sidecar_status["has_api_key"] = false;
            const auto kid = k.svc->create();
            k.svc->run(kid, advice());
            expect_equal((*k.svc->view(kid))["proposal"]["status"].get<std::string>(), "unavailable", "no key");
            expect_equal(static_cast<long>(k.posts()), 0L, "no key -> no call");

            Harness bad(repo, "bad");
            bad.freeze_thresholds();
            bad.on_decide = [](const json& req) { return json{{"request_id", req["request_id"]}, {"status", "invalid_response"}, {"error_code", "unknown_candidate"}}; };
            const auto bid = bad.svc->create();
            bad.svc->run(bid, advice());
            const auto bv = *bad.svc->view(bid);
            expect_equal(bv["proposal"]["status"].get<std::string>(), "rejected", "invalid provider answer");
            expect_true(bv["proposal"]["updates"].empty(), "no updates from an invalid answer");
            expect_equal(static_cast<long>(bad.posts()), 1L, "the provider was asked exactly once");

            Harness inj(repo, "inject");
            inj.freeze_thresholds();
            inj.on_decide = [](const json& req) { return ok_answer(req, "disable_bge"); };  // model names a candidate that does not exist
            const auto iid = inj.svc->create();
            inj.svc->run(iid, advice());
            expect_equal((*inj.svc->view(iid))["proposal"]["status"].get<std::string>(), "rejected", "model cannot inject a new candidate");

            Harness http(repo, "http");
            http.freeze_thresholds();
            http.on_decide = [](const json&) -> json { throw SidecarHttpError(400, json{{"error", true}}); };
            const auto hid = http.svc->create();
            http.svc->run(hid, advice());
            expect_equal((*http.svc->view(hid))["proposal"]["status"].get<std::string>(), "rejected", "sidecar 400 = our request was bad, not an outage");
        }

        // ---- shadow: recorded, never applicable ----
        {
            Harness h(repo, "shadow");
            h.sidecar_status["mode"] = "shadow";
            h.freeze_thresholds();
            h.on_decide = [](const json& req) { return ok_answer(req, "enable_adaptive_weights"); };
            const auto id = h.svc->create();
            h.svc->run(id, advice());
            const auto v = *h.svc->view(id);
            expect_true(v["shadow"] == true && v["proposal"]["updates"].empty() && v["comparison"].empty(), "shadow view exposes no patch");
            const json stored = json::parse(slurp_file(h.dir / id / "proposal.json"));
            expect_true(stored["updates"].size() == 1, "but the full proposal is stored for later evaluation");
            const auto r = h.svc->apply(id, advice());
            expect_equal(static_cast<long>(r.http_status), 403L, "apply refused in shadow");
            expect_equal(r.body["code"].get<std::string>(), "SHADOW_MODE", "code");
        }

        // ---- suggest: full path incl. apply, CAS, idempotence, restart ----
        {
            Harness h(repo, "suggest");
            h.freeze_thresholds();
            json sent;
            h.on_decide = [&](const json& req) { sent = req; return ok_answer(req, "enable_adaptive_weights", 0.7); };
            const auto before = tree(h.root);
            const auto id = h.svc->create();
            h.svc->run(id, advice());
            const auto v = *h.svc->view(id);
            expect_equal(v["proposal"]["status"].get<std::string>(), "validated", "validated");
            valid(v["proposal"]);
            expect_true(v["proposal"]["review_required"] == true && v["evidence"]["experimental"] == true, "experimental flagged");
            expect_equal(static_cast<long>(v["comparison"].size()), 1L, "one comparison row");
            expect_true(v["comparison"][0]["current"] == false && v["comparison"][0]["proposed"] == true && v["comparison"][0]["changed"] == true, "current vs proposed");
            expect_true(v["evidence"]["rationale"]["params"].contains("metric_agreement"), "evidence carries the measured number");
            expect_true(v["offered"].size() == 3, "offered list");
            expect_equal(static_cast<long>(sent["allowed_candidates"].size()), 3L, "provider saw the allowed ids");
            expect_true(sent["candidate_descriptions"].contains("enable_adaptive_weights") &&
                            !sent["candidate_descriptions"].contains("keep_current"), "provider is told what each change candidate does (baselines are described by the question set)");
            expect_true(sent["candidate_facts"]["enable_adaptive_weights"].contains("metric_agreement") &&
                            sent["candidate_facts"]["enable_adaptive_weights"].contains("measurement_coverage"), "and which evidence the backend already resolved for it");
            expect_equal(sent["question_set_version"].get<std::string>(), "decision-questions.v2", "current question set");
            const std::string request_text = slurp_file(h.dir / id / "request.json");
            for (const char* banned : {"/home/x", "runs_dir", "Cam", "scan-1"})
                expect_true(request_text.find(banned) == std::string::npos, std::string("provider request leaks ") + banned);
            expect_true(fs::exists(h.dir / id / "events.jsonl") && fs::exists(h.dir / id / "state.json"), "state/events persisted");

            // stale: draft edited elsewhere after the advice
            auto edited = advice(); edited.base_config["other"] = 1;
            const auto stale = h.svc->apply(id, edited);
            expect_true(stale.http_status == 409 && stale.body["code"] == "PROPOSAL_STALE", "edited draft -> 409 stale");
            expect_true(stale.body["reasons"].dump().find("config_hash") != std::string::npos, "stale reason names the config");
            auto locked = advice(); locked.locked_paths = {"global_metrics.adaptive_weights"};
            expect_true(h.svc->apply(id, locked).body["reasons"].dump().find("locks_hash") != std::string::npos, "lock change -> stale");
            auto other_scan = advice(11);
            expect_true(h.svc->apply(id, other_scan).http_status == 409, "different scan -> stale");
            expect_true(json::parse(slurp_file(h.dir / id / "proposal.json"))["status"] == "validated", "failed applies do not change the proposal");

            // real apply
            h.clock = "2026-09-23T10:05:00Z";
            const auto ok = h.svc->apply(id, advice());
            expect_equal(static_cast<long>(ok.http_status), 200L, "apply ok");
            expect_true(ok.body["patched_config"]["global_metrics"]["adaptive_weights"] == true, "patched draft");
            expect_true(ok.body["patched_config"]["runs_dir"] == "/home/x/runs", "unrelated draft content untouched");
            expect_true(ok.body["already_applied"] == false && ok.body["proposal"]["status"] == "applied_to_draft", "status");
            expect_equal(ok.body["proposal"]["applied_at"].get<std::string>(), "2026-09-23T10:05:00Z", "applied_at");
            valid(ok.body["proposal"]);
            expect_true(ok.body["proposal"]["config_hash_before"] != ok.body["proposal"]["config_hash_after"], "hashes recorded");

            // idempotent repeat (draft now carries the change)
            const auto again = h.svc->apply(id, [] { auto a = advice(10, true); return a; }());
            expect_true(again.http_status == 200 && again.body["already_applied"] == true, "identical repeat is idempotent");
            expect_true(again.body["proposal"]["applied_at"] == ok.body["proposal"]["applied_at"], "repeat does not rewrite applied_at");
            auto changed_after_apply = advice(10, true);
            changed_after_apply.base_config["other"] = 1;
            expect_true(h.svc->apply(id, changed_after_apply).body["code"] == "DRAFT_CHANGED", "repeat rejects unrelated draft edits");
            auto changed_scan_after_apply = advice(11, true);
            expect_true(h.svc->apply(id, changed_scan_after_apply).body["code"] == "PROPOSAL_STALE", "repeat rejects changed scan");
            auto changed_locks_after_apply = advice(10, true);
            changed_locks_after_apply.locked_paths = {"global_metrics.adaptive_weights"};
            expect_true(h.svc->apply(id, changed_locks_after_apply).body["code"] == "PROPOSAL_STALE", "repeat rejects changed locks");
            // draft moved away from the applied values
            expect_equal(static_cast<long>(h.svc->apply(id, advice()).http_status), 409L, "draft reverted -> DRAFT_CHANGED");

            // survives a restart
            h.rebuild(repo);
            expect_true((*h.svc->view(id))["proposal"]["status"] == "applied_to_draft", "state survives a new service instance");
            expect_true(h.svc->apply(id, advice(10, true)).body["already_applied"] == true, "idempotence survives restart");
            expect_equal(static_cast<long>(h.svc->apply("nope", advice()).http_status), 404L, "unknown id");
            expect_equal(static_cast<long>(h.svc->apply("../x", advice()).http_status), 404L, "path-like id refused");
            expect_true(!h.svc->view("../x") && !h.svc->view("missing"), "view refuses unknown/path-like ids");

            // nothing outside pi_decisions/ was created
            for (const auto& [rel, _] : tree(h.root)) expect_true(rel.rfind("pi_decisions/", 0) == 0, "only pi_decisions/ written: " + rel);
            (void)before;
        }

        // ---- config validator veto blocks the offer (whole group) ----
        {
            Harness h(repo, "veto");
            h.freeze_thresholds();
            DecisionServiceDeps d;
            d.sidecar = [&](const std::string&, const std::string& ep, const json&) -> json { if (ep == "/decisions/status") return h.sidecar_status; throw std::runtime_error("must not be called"); };
            d.validate_config = [](const json&) { return ConfigCheck{false, "schema says no"}; };
            d.now_iso = [] { return "2026-09-23T10:00:00Z"; };
            d.new_id = [] { return "dec_v"; };
            DecisionService svc(h.dir, load_real_catalog(repo), d);
            const auto id = svc.create();
            svc.run(id, advice());
            const auto v = *svc.view(id);
            expect_true(v["synthetic_baseline"] == true && v["excluded"].dump().find("config_invalid") != std::string::npos, "vetoed candidate is excluded, nothing asked");
        }

        std::puts("pi_decision_service: all checks passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
