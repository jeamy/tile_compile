#include "services/pi/pi_decision_record_store.hpp"

#include "backend_test_harness.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

using nlohmann::json;

namespace {

template <typename Fn>
bool throws_invalid(Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

json base_record(const std::string& kind, const std::string& actor) {
    return {
        {"kind", kind},
        {"actor", actor},
        {"context_ref", {{"context_id", "ctx_a"}, {"run_uid", "run_1"}}},
        {"subject", {{"paths", json::array({{{"path", "reconstruction.pixfrac"}, {"from", 0.8}, {"to", 0.7}}})}}},
    };
}

std::string slurp(const std::filesystem::path& p) {
    std::ifstream in(p);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

int main() {
    try {
        using tile_compile::pi::PiDecisionRecordStore;
        const auto dir = std::filesystem::temp_directory_path() /
            ("tile_compile_pi_decision_record_test_" + std::to_string(getpid()));
        std::filesystem::remove_all(dir);
        PiDecisionRecordStore store(dir);

        // Defaults: kein Grund angegeben -> expliziter Wert, kein Raten.
        const auto first = store.append(base_record("config_apply", "user"));
        expect_equal(first["schema_version"].get<std::string>(), "pi.decision-record.v1", "schema");
        expect_equal(first["origin"].get<std::string>(), "live", "default origin");
        expect_equal(first["privacy_class"].get<std::string>(), "metadata_only", "privacy without text");
        expect_true(first["rationale"]["user"]["no_reason_given"].get<bool>(), "missing reason is explicit");
        expect_true(first["decision_id"].get<std::string>().rfind("dec_", 0) == 0, "decision id generated");
        expect_true(!first["duplicate"].get<bool>(), "first append is not a duplicate");
        expect_true(std::filesystem::is_regular_file(store.records_path()), "records file exists");

        // Freitext: Pfade werden entfernt, privacy_class folgt.
        auto with_text = base_record("config_reject", "user");
        with_text["rationale"] = {{"user", {
            {"reason_codes", json::array({"artifacts"})},
            {"catalog_version", "pi.user-reason-codes.v1"},
            {"text", "Siehe /home/lux/runs/a/b.fits und C:\\data\\x.fits, zu hart"},
        }}, {"basis", json::array()}};
        const auto txt = store.append(with_text);
        const std::string scrubbed = txt["rationale"]["user"]["text"].get<std::string>();
        expect_true(scrubbed.find("/home/lux") == std::string::npos, "unix path scrubbed");
        expect_true(scrubbed.find("C:\\") == std::string::npos, "windows path scrubbed");
        expect_true(scrubbed.find("zu hart") != std::string::npos, "plain text kept");
        expect_equal(txt["privacy_class"].get<std::string>(), "metadata_plus_user_text", "privacy with text");
        expect_true(!txt["rationale"]["user"]["no_reason_given"].get<bool>(), "reason present");

        // Laengenbegrenzung.
        const std::string longtext(5000, 'a');
        expect_equal(static_cast<long>(tile_compile::pi::scrub_rationale_text(longtext).size()),
                     static_cast<long>(tile_compile::pi::kDecisionRationaleTextMaxChars), "text length capped");

        // Idempotenz.
        auto keyed = base_record("config_apply", "user");
        keyed["idempotency_key"] = "plan_1:apply";
        const auto k1 = store.append(keyed);
        const auto k2 = store.append(keyed);
        expect_equal(k1["decision_id"].get<std::string>(), k2["decision_id"].get<std::string>(), "idempotent id");
        expect_true(k2["duplicate"].get<bool>(), "second append flagged duplicate");
        expect_equal(static_cast<long>(store.list({{"kind", "config_apply"}}, 100).size()), 2L,
                     "duplicate not written twice");

        // Validierung.
        expect_true(throws_invalid([&] { store.append(base_record("config_apply", "llm")); }),
                    "llm actor not allowed for config_apply");
        expect_true(throws_invalid([&] { store.append(base_record("llm_proposal", "user")); }),
                    "user actor not allowed for llm_proposal");
        expect_true(throws_invalid([&] { store.append(base_record("nonsense", "user")); }), "unknown kind");
        {
            auto r = base_record("jev_choice", "jev");
            r["rationale"] = {{"user", json::object()}, {"basis", json::array({{{"source", "llm_hypothesis"}}})}};
            expect_true(throws_invalid([&] { store.append(r); }), "jev actor must not cite llm_hypothesis");
        }
        {
            auto r = base_record("config_apply", "user");
            r["rationale"] = {{"user", {{"no_reason_given", true}, {"reason_codes", json::array({"artifacts"})},
                                        {"catalog_version", "v1"}}}, {"basis", json::array()}};
            expect_true(throws_invalid([&] { store.append(r); }), "no_reason_given excludes codes");
        }
        {
            auto r = base_record("config_reject", "user");
            r["rationale"] = {{"user", {{"reason_codes", json::array({"artifacts"})}}}, {"basis", json::array()}};
            expect_true(throws_invalid([&] { store.append(r); }), "codes need catalog_version");
        }
        {
            auto r = base_record("config_apply", "user");
            r["evidence"] = {{"metrics_ref", "/abs/path/metrics.json"}};
            expect_true(throws_invalid([&] { store.append(r); }), "absolute metrics_ref rejected");
            r["evidence"] = {{"metrics_ref", "artifacts/metrics.json"}};
            store.append(r);
        }
        {
            auto r = base_record("config_apply", "user");
            r["jev"] = {{"confidence", 1.0}};
            expect_true(throws_invalid([&] { store.append(r); }), "jev block only for jev kinds");
        }
        {
            auto r = base_record("llm_proposal", "llm");
            r["rationale"] = {{"user", {{"text", "ich bin ein Nutzer"}}}, {"basis", json::array({{{"source", "llm_hypothesis"}}})}};
            expect_true(throws_invalid([&] { store.append(r); }), "non-user actor must not carry user rationale");
        }

        // Annahmekette Jev: jev_choice -> config_apply.
        auto jev = base_record("jev_choice", "jev");
        jev["rationale"] = {{"user", json::object()},
                            {"basis", json::array({{{"source", "model_probability"}, {"ref", "gen-dec-1"}}})}};
        jev["jev"] = {{"model_reported", "typesafe/jev-1.13-20260917"}, {"confidence", 0.8}};
        const auto jev_rec = store.append(jev);
        auto apply = base_record("config_apply", "user");
        apply["parent_decision_id"] = jev_rec["decision_id"];
        const auto apply_rec = store.append(apply);
        const auto children = store.list({{"parent_decision_id", jev_rec["decision_id"]}}, 10);
        expect_equal(static_cast<long>(children.size()), 1L, "child found via parent_decision_id");
        expect_equal(children[0]["decision_id"].get<std::string>(), apply_rec["decision_id"].get<std::string>(),
                     "child is the apply record");

        // Overlay: memory/outcome Links, ohne das Record zu mutieren.
        const std::string before = slurp(store.records_path());
        store.add_link(apply_rec["decision_id"], "memory", {{"memory_id", "mem_1"}});
        store.add_link(apply_rec["decision_id"], "outcome", {{"ref", "run_1:quality"}});
        store.add_link(apply_rec["decision_id"], "outcome", {{"ref", "run_1:quality"}});
        const auto merged = store.get(apply_rec["decision_id"]);
        expect_equal(merged["memory_id"].get<std::string>(), "mem_1", "memory link merged");
        expect_equal(static_cast<long>(merged["outcome_refs"].size()), 1L, "outcome refs deduplicated");
        expect_equal(slurp(store.records_path()), before, "links do not mutate records");
        expect_true(throws_invalid([&] { store.add_link("dec_unknown", "memory", {}); }), "unknown decision rejected");
        expect_true(throws_invalid([&] { store.add_link(apply_rec["decision_id"], "bogus", {}); }), "unknown link type");
        expect_true(store.get("dec_missing").is_null(), "unknown get is null");

        // Filter reason_code.
        const auto by_code = store.list({{"reason_code", "artifacts"}}, 10);
        expect_equal(static_cast<long>(by_code.size()), 1L, "reason_code filter");

        // Redaktion: Leser unterdrueckt, compact entfernt physisch.
        const std::string reject_id = txt["decision_id"].get<std::string>();
        expect_true(slurp(store.records_path()).find("zu hart") != std::string::npos, "text on disk before redact");
        store.redact(reject_id, "user_request");
        const auto redacted = store.get(reject_id);
        expect_equal(redacted["rationale"]["user"]["text"].get<std::string>(), "", "redacted text suppressed");
        expect_true(redacted["rationale"]["user"]["text_redacted"].get<bool>(), "text_redacted flag");
        expect_true(slurp(store.records_path()).find("zu hart") != std::string::npos, "redact alone keeps bytes");
        const auto stats = store.compact();
        expect_equal(static_cast<long>(stats["redacted_texts_removed"].get<int>()), 1L, "compaction removed text");
        expect_true(slurp(store.records_path()).find("zu hart") == std::string::npos, "text physically gone");
        expect_equal(store.get(reject_id)["privacy_class"].get<std::string>(), "metadata_only",
                     "privacy class downgraded after compaction");
        expect_equal(static_cast<long>(store.list({}, 1000).size()), static_cast<long>(stats["rewritten"].get<int>()),
                     "compaction keeps all records");

        // Run-Loeschung kaskadiert auf Bild-Kontext.
        auto img = base_record("live_edit_op", "user");
        img["context_ref"] = {{"context_id", "ctx_img"}, {"image_id", "run_9:live_edit"}};
        img["rationale"] = {{"user", {{"text", "zu weich"}}}, {"basis", json::array()}};
        const auto img_rec = store.append(img);
        auto other = base_record("config_apply", "user");
        other["context_ref"] = {{"context_id", "ctx_other"}, {"run_uid", "run_10"}};
        other["rationale"] = {{"user", {{"text", "bleibt"}}}, {"basis", json::array()}};
        const auto other_rec = store.append(other);
        expect_equal(static_cast<long>(store.mark_run_deleted("run_9")), 1L, "run deletion hits image context");
        expect_equal(static_cast<long>(store.mark_run_deleted("run_9")), 0L, "run deletion is idempotent");
        const auto deleted = store.get(img_rec["decision_id"]);
        expect_true(deleted["run_deleted"].get<bool>(), "run_deleted flag");
        expect_equal(deleted["rationale"]["user"]["text"].get<std::string>(), "", "deleted run text suppressed");
        expect_equal(store.get(other_rec["decision_id"])["rationale"]["user"]["text"].get<std::string>(), "bleibt",
                     "other run untouched");

        // Limit liefert die letzten Treffer.
        expect_equal(static_cast<long>(store.list({}, 2).size()), 2L, "limit applied");

        std::filesystem::remove_all(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
