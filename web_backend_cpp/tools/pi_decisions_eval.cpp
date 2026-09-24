// Offline evaluation harness for the Jev pre-run advice (no backend, no sidecar, no run).
//   prepare --scan S.json --metrics M.json --config C.yaml --out DIR [--min-coverage X --min-agreement Y]
//           [--mode shadow|suggest] [--adaptive-override true|false] [--set path=value]...
//   resolve --dir DIR --response R.json
// `prepare` writes DIR/{state,candidates,request}.json; the provider call itself is made by
// agent_service/scripts/jev_eval_call.ts (the real M3 path). `resolve` turns the sidecar response into a
// pi.config-proposal.v1 and prints a current-vs-proposed comparison.
#include "services/pi/pi_decision_outcome.hpp"
#include "services/pi/pi_decision_policy.hpp"
#include "services/pi/pi_decision_state.hpp"
#include "services/pi/pi_json_io.hpp"
#include "services/pi/pi_pre_rules.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
#include <sstream>

using namespace tile_compile::pi;
using nlohmann::json;
namespace fs = std::filesystem;

static std::string slurp(const std::string& p) { std::ifstream in(p); std::stringstream ss; ss << in.rdbuf(); return ss.str(); }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: pi_decisions_eval prepare|resolve ...\n"); return 2; }
    const std::string cmd = argv[1];
    std::map<std::string, std::string> a;
    std::vector<std::string> sets;  // repeatable: --set path=json_or_text
    for (int i = 2; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "--set") sets.push_back(argv[i + 1]);
        else a[argv[i]] = argv[i + 1];
    }
    auto apply_sets = [&](json& cfg) {
        for (const auto& kv : sets) {
            const auto eq = kv.find('=');
            if (eq == std::string::npos || eq == 0) throw std::invalid_argument("--set needs path=value: " + kv);
            const std::string value = kv.substr(eq + 1);
            json parsed = json::parse(value, nullptr, false);
            config_set(cfg, kv.substr(0, eq), parsed.is_discarded() ? json(value) : parsed);
        }
    };
    try {
        const fs::path catalog_dir = a.count("--catalog") ? a["--catalog"] : "web_backend_cpp/config/pi_decisions";
        const DecisionCatalog catalog = load_decision_catalog(json::parse(slurp((catalog_dir / "candidates_v1.json").string())),
                                                              json::parse(slurp((catalog_dir / "protected_paths_v1.json").string())));
        const fs::path dir = a.at(cmd == "prepare" ? "--out" : "--dir");
        json cfg;
        DecisionPolicy policy;
        policy.allow_experimental = (a.count("--mode") ? a["--mode"] : "shadow") != "off";
        if (a.count("--min-coverage")) policy.min_measurement_coverage = std::stod(a["--min-coverage"]);
        if (a.count("--min-agreement")) policy.min_metric_agreement = std::stod(a["--min-agreement"]);
        auto accept = [](const json&) { return ConfigCheck{true, ""}; };  // eval only: real validate-config is a CLI call

        if (cmd == "prepare") {
            cfg = yaml_text_to_json(slurp(a.at("--config")));
            if (a.count("--adaptive-override")) config_set(cfg, "global_metrics.adaptive_weights", a["--adaptive-override"] == "true");
            apply_sets(cfg);
            PreRunDecisionInputs in;
            in.scan = json::parse(slurp(a.at("--scan")));
            in.metrics = json::parse(slurp(a.at("--metrics")));
            in.base_config = cfg;
            in.scan_id = "eval";
            in.software_version = "eval";
            const auto sr = build_pre_run_decision_state(in);
            const auto cands = build_pre_run_candidates(sr.state, cfg, policy, catalog, accept);
            write_json_file_atomic(dir / "state.json", {{"state", sr.state}, {"state_hash", sr.state_hash}, {"findings", sr.findings}});
            write_json_file_atomic(dir / "candidates.json", {{"applicable", cands.applicable}, {"excluded", cands.excluded}});
            bool real = false;
            for (const auto& c : cands.applicable) real = real || !c["updates"].empty();
            if (real)
                {
                json req = {{"request_id", "eval"}, {"state_hash", sr.state_hash}, {"question_set_version", "decision-questions.v2"},
                            {"state_projection", sr.provider_projection}, {"allowed_candidates", cands.allowed_ids()}};
                const json info = provider_candidate_info(cands, catalog);
                if (!info["descriptions"].empty()) req["candidate_descriptions"] = info["descriptions"];
                if (!info["facts"].empty()) req["candidate_facts"] = info["facts"];
                write_json_file_atomic(dir / "request.json", req);
            }
            std::cout << json{{"state_hash", sr.state_hash}, {"blocking", sr.state["blocking_findings"]}, {"groups", sr.state["groups"].size()},
                              {"offered", cands.allowed_ids()}, {"excluded", cands.excluded}, {"request_written", real}}.dump(2) << "\n";
            return 0;
        }
        if (cmd == "resolve") {
            const json st = json::parse(slurp((dir / "state.json").string()));
            cfg = yaml_text_to_json(slurp(a.at("--config")));
            if (a.count("--adaptive-override")) config_set(cfg, "global_metrics.adaptive_weights", a["--adaptive-override"] == "true");
            apply_sets(cfg);
            ResolveContext ctx{"eval", "2026-09-23T00:00:00Z", st["state_hash"], "decision-questions.v2", "typesafe/jev-1.13", nullptr};
            const json resp = json::parse(slurp(a.at("--response")));
            if (resp.contains("model_reported")) ctx.model_reported = resp["model_reported"];
            const json proposal = resolve_decision(resp, st["state"], cfg, policy, catalog, accept, ctx);
            json cmp = json::array();
            for (const auto& u : proposal["updates"]) cmp.push_back({{"path", u["path"]}, {"current", u["old_value"]}, {"proposed", u["value"]}});
            std::cout << json{{"status", proposal["status"]}, {"candidate", proposal["candidate_id"]}, {"reasons", proposal["reason_codes"]},
                              {"review_required", proposal["review_required"]}, {"comparison", cmp}}.dump(2) << "\n";
            write_json_file_atomic(dir / "proposal.json", proposal);
            return 0;
        }
        std::fprintf(stderr, "unknown command\n");
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
