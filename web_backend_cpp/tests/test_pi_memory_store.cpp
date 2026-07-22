#include "services/pi/pi_memory_store.hpp"

#include "backend_test_harness.hpp"

#include <filesystem>
#include <unistd.h>

int main() {
    try {
        const auto dir = std::filesystem::temp_directory_path() /
            ("tile_compile_pi_memory_test_" + std::to_string(getpid()));
        std::filesystem::remove_all(dir);

        tile_compile::pi::PiMemoryStore store(dir);
        const nlohmann::json ctx = {
            {"schema_version", "pi.context_signature.v1"},
            {"target", {{"object_name", "M42"}, {"object_type", "emission_nebula"}}},
            {"acquisition", {{"camera_name", "ASI2600MC"}, {"camera_type", "OSC"}, {"filters", nlohmann::json::array({"HaOIII"})}}},
            {"pipeline", {{"affected_paths", nlohmann::json::array({"bge.model"})}}},
            {"problem", {{"classes", nlohmann::json::array({"faint_nebula"})}}}
        };
        const nlohmann::json scope = {
            {"applies_when", nlohmann::json::array({"matching context"})},
            {"does_not_apply_when", nlohmann::json::array({"different target class"})},
            {"confidence", 0.5}
        };
        const auto first = store.append_candidate({
            {"type", "optimization"},
            {"source_session_id", "pi_sess_fixture"},
            {"context_signature", ctx},
            {"scope", scope},
            {"recommendation", {
                {"patch", nlohmann::json::array({{{"path", "bge.model"}, {"value", "rbf"}}})}
            }},
            {"evidence", {{"validation", "fixture"}}},
            {"outcome", {{"validation_valid", true}, {"applied_count", 1}}}
        });

        expect_equal(first["schema_version"].get<std::string>(), "pi.memory.v2", "memory schema");
        expect_equal(first["status"].get<std::string>(), "candidate", "memory default status");
        expect_equal(first["privacy_class"].get<std::string>(), "metadata_only", "memory default privacy");
        expect_true(!first["memory_id"].get<std::string>().empty(), "memory id generated");
        expect_equal(first["id"].get<std::string>(), first["memory_id"].get<std::string>(), "memory id alias generated");
        expect_true(std::filesystem::is_regular_file(store.memories_path()), "memory jsonl exists");
        expect_true(!std::filesystem::exists(store.legacy_memories_path()), "legacy memory jsonl ignored");

        store.append_candidate({
            {"memory_id", "mem_fixture_second"},
            {"type", "failure"},
            {"status", "candidate"},
            {"privacy_class", "metadata_only"},
            {"context_signature", ctx},
            {"scope", scope},
            {"evidence", {{"validation", "fixture"}}},
            {"outcome", {{"validation_valid", false}, {"applied_count", 1}}},
            {"avoid", {{"path", "bge.model"}, {"value", "classic"}}}
        });
        store.append_candidate({
            {"memory_id", "mem_fixture_lower_score"},
            {"type", "optimization"},
            {"status", "candidate"},
            {"privacy_class", "metadata_only"},
            {"context_signature", ctx},
            {"scope", scope},
            {"recommendation", {
                {"patch", nlohmann::json::array({{{"path", "bge.model"}, {"value", "poly"}}})}
            }},
            {"evidence", {{"validation", "fixture"}}},
            {"outcome", {{"validation_valid", false}, {"applied_count", 1}}}
        });

        const auto all = store.list();
        expect_equal(static_cast<long>(all.size()), 3L, "memory list count");
        const auto limited = store.list(1);
        expect_equal(static_cast<long>(limited.size()), 1L, "memory limited list count");
        expect_equal(limited[0]["memory_id"].get<std::string>(), "mem_fixture_lower_score", "memory list keeps newest");

        const auto review = store.review(first["memory_id"].get<std::string>(), "accepted", "fixture", "works");
        store.review("mem_fixture_lower_score", "accepted", "fixture", "less useful");
        expect_equal(review["status"].get<std::string>(), "accepted", "memory review status");
        expect_true(std::filesystem::is_regular_file(store.reviews_path()), "memory review jsonl exists");

        const auto reviewed = store.list();
        expect_equal(reviewed[0]["status"].get<std::string>(), "accepted", "memory list overlays review status");
        expect_true(reviewed[0].contains("review"), "memory list includes latest review");
        const auto indices = store.indices();
        expect_equal(indices["schema_version"].get<std::string>(), "pi.memory-indices.v2",
                     "memory index schema");
        expect_true(std::filesystem::is_regular_file(store.indices_path()), "memory index file exists");
        expect_true(indices["by_type"]["optimization"].is_array(), "memory index by type");
        expect_true(indices["by_status"]["accepted"].is_array(), "memory index by status");
        expect_true(indices["by_path"]["bge.model"].is_array(), "memory index by path");
        expect_true(indices["by_target"]["m42"].is_array(), "memory index by target name");
        expect_true(indices["by_camera"]["asi2600mc"].is_array(), "memory index by camera");
        expect_true(indices["by_filter"]["haoiii"].is_array(), "memory index by filter");
        expect_true(indices["by_problem"]["faint_nebula"].is_array(), "memory index by problem");

        const auto matches = store.retrieve({
            {"type", "optimization"},
            {"config_updates", nlohmann::json::array({{{"path", "bge.model"}, {"value", "rbf"}}})}
        }, 5);
        expect_equal(static_cast<long>(matches.size()), 2L, "memory retrieval match count");
        expect_equal(matches[0]["memory_id"].get<std::string>(), first["memory_id"].get<std::string>(),
                     "memory retrieval ranks positive outcome first");

        const auto deprecated_review = store.review("mem_fixture_second", "deprecated", "fixture", "superseded", {
            {"validation_valid", false},
            {"reason", "regression_after_run"}
        });
        expect_equal(deprecated_review["outcome"]["reason"].get<std::string>(), "regression_after_run",
                     "deprecated memory review stores outcome");
        const auto negative_matches = store.retrieve({{"type", "failure"}, {"paths", nlohmann::json::array({"bge.model"})}}, 5);
        expect_equal(static_cast<long>(negative_matches.size()), 0L, "deprecated memory excluded from retrieval");
        const auto negative_warnings = store.retrieve_negative({{"type", "failure"}, {"paths", nlohmann::json::array({"bge.model"})}}, 5);
        expect_equal(static_cast<long>(negative_warnings.size()), 1L, "deprecated memory returned as warning");
        expect_true(negative_warnings[0].contains("match_coverage"), "negative retrieval explains match coverage");

        const auto string_path_matches = store.retrieve({
            {"type", "optimization"},
            {"paths", nlohmann::json::array({"bge.model"})}
        }, 5);
        expect_equal(static_cast<long>(string_path_matches.size()), 2L,
                     "memory retrieval accepts string path lists");
        expect_true(string_path_matches[0].contains("match_explanation"), "memory retrieval includes match explanation");

        const nlohmann::json irrelevant_ctx = {
            {"schema_version", "pi.context_signature.v1"},
            {"target", {{"object_type", "galaxy"}}},
            {"acquisition", {{"camera_type", "MONO"}}},
            {"pipeline", {{"affected_paths", nlohmann::json::array({"bge.model"})}}}
        };
        store.append_candidate({
            {"memory_id", "mem_fixture_irrelevant_context"},
            {"type", "optimization"},
            {"status", "candidate"},
            {"privacy_class", "metadata_only"},
            {"context_signature", irrelevant_ctx},
            {"scope", scope},
            {"recommendation", {
                {"patch", nlohmann::json::array({{{"path", "bge.model"}, {"value", "galaxy_model"}}})}
            }},
            {"evidence", {{"validation", "fixture"}}},
            {"outcome", {{"validation_valid", true}, {"applied_count", 1}}}
        });
        store.review("mem_fixture_irrelevant_context", "accepted", "fixture", "different setup");

        const auto contextual_matches = store.retrieve({
            {"type", "optimization"},
            {"context_signature", ctx}
        }, 5);
        expect_equal(static_cast<long>(contextual_matches.size()), 2L,
                     "context-only retrieval excludes different target and camera memories");
        expect_true(contextual_matches[0].value("context_match_score", 0) > 0,
                    "contextual retrieval scores signature matches");
        bool found_irrelevant_context = false;
        for (const auto& match : contextual_matches) {
            if (match.value("memory_id", std::string()) == "mem_fixture_irrelevant_context") {
                found_irrelevant_context = true;
            }
        }
        expect_true(!found_irrelevant_context, "contextual retrieval filters unrelated accepted memory");

        const auto duplicate = store.append_candidate({
            {"type", "optimization"},
            {"source_session_id", "pi_sess_fixture"},
            {"context_signature", ctx},
            {"scope", scope},
            {"recommendation", {
                {"patch", nlohmann::json::array({{{"path", "bge.model"}, {"value", "rbf"}}})}
            }},
            {"evidence", {{"validation", "fixture"}}},
            {"outcome", {{"validation_valid", true}, {"applied_count", 1}}}
        });
        expect_true(!duplicate.value("created", true), "duplicate memory is not appended");
        expect_true(duplicate.value("duplicate", false), "duplicate memory is flagged");
        expect_equal(static_cast<long>(store.list().size()), 4L, "duplicate memory keeps list count");

        const auto exported = store.export_bundle("metadata_only", true);
        expect_equal(exported["schema_version"].get<std::string>(), "pi.memories-export.v2",
                     "memory export schema");
        expect_equal(static_cast<long>(exported["memory_count"].get<size_t>()), 4L,
                     "memory export count");

        const auto dry_import = store.import_bundle(exported, true);
        expect_true(dry_import["dry_run"].get<bool>(), "memory import dry run");
        expect_equal(dry_import["imported_memories"].get<long>(), 0L,
                     "memory import skips existing memories");

        const auto dedupe_preview = store.dedupe(true);
        expect_true(dedupe_preview["dry_run"].get<bool>(), "memory dedupe dry run");
        expect_equal(dedupe_preview["removed_count"].get<long>(), 0L,
                     "memory dedupe has no existing duplicates after append dedupe");

        bool rejected_non_candidate = false;
        try {
            store.append_candidate({{"type", "optimization"}, {"status", "accepted"}, {"context_signature", ctx}, {"scope", scope}, {"evidence", nlohmann::json::object()}, {"outcome", nlohmann::json::object()}});
        } catch (const std::invalid_argument&) {
            rejected_non_candidate = true;
        }
        expect_true(rejected_non_candidate, "non-candidate memory rejected");

        bool rejected_missing_context = false;
        try {
            store.append_candidate({{"type", "optimization"}, {"evidence", nlohmann::json::object()}, {"outcome", nlohmann::json::object()}});
        } catch (const std::invalid_argument&) {
            rejected_missing_context = true;
        }
        expect_true(rejected_missing_context, "memory without context signature rejected");

        bool rejected_bad_review = false;
        try {
            store.review(first["memory_id"].get<std::string>(), "candidate", "fixture");
        } catch (const std::invalid_argument&) {
            rejected_bad_review = true;
        }
        expect_true(rejected_bad_review, "unsupported review status rejected");

        // --- Diversity-Cap-Test ---
        // Viele aehnliche Memories einer Klasse duerfen den KI-Prompt nicht fluten.
        // Bei context_signature-Abfrage: max. 2 pro Objekt-/Kamera-Klasse.
        tile_compile::pi::PiMemoryStore dstore(dir / "diversity");
        const nlohmann::json d_ctx = {
            {"schema_version", "pi.context_signature.v1"},
            {"target", {{"object_name", "M31"}, {"object_type", "galaxy"}}},
            {"acquisition", {{"camera_type", "MONO"}}},
            {"pipeline", {{"affected_paths", nlohmann::json::array({"bge.model"})}}}
        };
        const nlohmann::json d_scope = {
            {"applies_when", nlohmann::json::array({"galaxy MONO setup"})},
            {"does_not_apply_when", nlohmann::json::array({"OSC setup"})},
            {"confidence", 0.6}
        };
        for (int i = 0; i < 5; ++i) {
            dstore.append_candidate({
                {"memory_id", "mem_diversity_" + std::to_string(i)},
                {"type", "config_optimization"},
                {"source", "scan_ai_apply"},
                {"privacy_class", "metadata_only"},
                {"context_signature", d_ctx},
                {"scope", d_scope},
                {"recommendation", {{"explanation", "diversity test " + std::to_string(i)},
                                     {"patch", nlohmann::json::array({{{"path", "bge.model"}, {"value", "v" + std::to_string(i)}}})}}},
                {"evidence", {{"validation", "fixture"}}},
                {"outcome", {{"validation_valid", true}, {"applied_count", 1}}}
            });
            dstore.review("mem_diversity_" + std::to_string(i), "accepted", "fixture", "ok");
        }
        const nlohmann::json d_query_ctx = {
            {"schema_version", "pi.context_signature.v1"},
            {"target", {{"object_name", "M31"}, {"object_type", "galaxy"}}},
            {"acquisition", {{"camera_type", "MONO"}}}
        };
        const auto diversity_matches = dstore.retrieve({
            {"context_signature", d_query_ctx}
        }, 10);
        // Cap ist 2 bei Kontext-Abfrage (galaxy|mono Klasse) — nie mehr als 2 dieser Klasse.
        expect_true(static_cast<long>(diversity_matches.size()) <= 2L,
                    "diversity-cap: context query returns at most 2 memories per object/camera class");

        // Ohne Kontext-Abfrage (nur type) ist der Cap 3.
        const auto type_only_matches = dstore.retrieve({{"type", "config_optimization"}}, 10);
        expect_true(static_cast<long>(type_only_matches.size()) <= 3L,
                    "diversity-cap: type-only query returns at most 3 memories per class");

        // --- Cross-Contamination-Test ---
        // Ein M42/Nebel/OSC-Memory darf nicht blind auf einen M104/Galaxie/Mono/LRGB-Kontext
        // angewendet werden, selbst wenn der Config-Pfad uebereinstimmt.
        tile_compile::pi::PiMemoryStore xstore(dir / "xcontam");
        const nlohmann::json nebula_ctx = {
            {"schema_version", "pi.context_signature.v1"},
            {"target", {{"object_name", "M42"}, {"object_type", "emission_nebula"}, {"angular_size_class", "large"}, {"has_extended_emission", true}}},
            {"acquisition", {{"camera_type", "OSC"}, {"color_mode", "OSC"}, {"filters", nlohmann::json::array({"dual_narrowband"})}}},
            {"pipeline", {{"affected_paths", nlohmann::json::array({"bge.enabled"})}, {"phases", nlohmann::json::array({"BGE"})}}}
        };
        const nlohmann::json nebula_scope = {
            {"applies_when", nlohmann::json::array({"target has large diffuse emission"})},
            {"does_not_apply_when", nlohmann::json::array({"compact galaxy target"})},
            {"confidence", 0.7}
        };
        xstore.append_candidate({
            {"memory_id", "mem_nebula_bge_osc"},
            {"type", "config_optimization"},
            {"source", "scan_ai_apply"},
            {"privacy_class", "metadata_only"},
            {"context_signature", nebula_ctx},
            {"scope", nebula_scope},
            {"recommendation", {{"explanation", "Nebula BGE OSC fixture"}}},
            {"evidence", {{"validation", "fixture"}}},
            {"outcome", {{"validation_valid", true}, {"applied_count", 1}}}
        });
        xstore.review("mem_nebula_bge_osc", "accepted", "fixture", "works for nebulae");

        const nlohmann::json galaxy_query_ctx = {
            {"schema_version", "pi.context_signature.v1"},
            {"target", {{"object_name", "M104"}, {"object_type", "galaxy"}, {"angular_size_class", "small"}, {"has_extended_emission", false}}},
            {"acquisition", {{"camera_type", "MONO"}, {"color_mode", "MONO"}, {"filters", nlohmann::json::array({"LRGB"})}}},
            {"pipeline", {{"affected_paths", nlohmann::json::array({"bge.enabled"})}, {"phases", nlohmann::json::array({"BGE"})}}}
        };
        const auto galaxy_matches = xstore.retrieve({
            {"context_signature", galaxy_query_ctx}
        }, 10);
        expect_equal(static_cast<long>(galaxy_matches.size()), 0L,
                     "cross-contamination: M42/Nebel/OSC memory must not match M104/Galaxie/Mono query");

        // Gleicher Config-Pfad darf bei fachlich falsem Kontext nicht als Match gelten.
        const auto galaxy_path_matches = xstore.retrieve({
            {"context_signature", galaxy_query_ctx},
            {"paths", nlohmann::json::array({"bge.enabled"})}
        }, 10);
        expect_equal(static_cast<long>(galaxy_path_matches.size()), 0L,
                     "cross-contamination: path overlap alone must not override mismatched context");

        // Positive Nebula-Anfrage muss weiterhin treffen.
        const nlohmann::json nebula_query_ctx = {
            {"schema_version", "pi.context_signature.v1"},
            {"target", {{"object_name", "M42"}, {"object_type", "emission_nebula"}, {"has_extended_emission", true}}},
            {"acquisition", {{"camera_type", "OSC"}, {"filters", nlohmann::json::array({"dual_narrowband"})}}},
            {"pipeline", {{"affected_paths", nlohmann::json::array({"bge.enabled"})}}}
        };
        const auto nebula_positive_matches = xstore.retrieve({
            {"context_signature", nebula_query_ctx}
        }, 10);
        expect_equal(static_cast<long>(nebula_positive_matches.size()), 1L,
                     "cross-contamination: correct M42/Nebel/OSC context must still produce a match");

        // --- Rejected-Memory-Negativsignal-Test ---
        // Ein rejected Memory muss bei aehnlichem Kontext als explizites Negativsignal auftauchen,
        // d.h. in retrieve_negative mit retrieval_warning und match_explanation zurueckkommen.
        tile_compile::pi::PiMemoryStore rstore(dir / "rejected_signal");
        const nlohmann::json shared_ctx = {
            {"schema_version", "pi.context_signature.v1"},
            {"target", {{"object_name", "NGC7293"}, {"object_type", "planetary_nebula"}}},
            {"acquisition", {{"camera_type", "OSC"}, {"filters", nlohmann::json::array({"Ha"})}}},
            {"pipeline", {{"affected_paths", nlohmann::json::array({"pcc.enabled"})}, {"phases", nlohmann::json::array({"PCC"})}}}
        };
        const nlohmann::json shared_scope = {
            {"applies_when", nlohmann::json::array({"narrowband Ha data with PCC enabled"})},
            {"does_not_apply_when", nlohmann::json::array({"broadband RGB only"})},
            {"confidence", 0.6}
        };
        rstore.append_candidate({
            {"memory_id", "mem_pcc_rejected"},
            {"type", "config_optimization"},
            {"source", "scan_ai_apply"},
            {"privacy_class", "metadata_only"},
            {"context_signature", shared_ctx},
            {"scope", shared_scope},
            {"recommendation", {{"explanation", "PCC was tried but produced color artifacts"}}},
            {"evidence", {{"validation", "fixture"}}},
            {"outcome", {{"validation_valid", false}, {"applied_count", 1}}}
        });
        rstore.review("mem_pcc_rejected", "rejected", "fixture", "produced color artifacts on narrowband");

        const nlohmann::json similar_ctx = {
            {"schema_version", "pi.context_signature.v1"},
            {"target", {{"object_name", "NGC7293"}, {"object_type", "planetary_nebula"}}},
            {"acquisition", {{"camera_type", "OSC"}, {"filters", nlohmann::json::array({"Ha"})}}},
            {"pipeline", {{"affected_paths", nlohmann::json::array({"pcc.enabled"})}}}
        };
        // Das rejected Memory darf nicht als positiver Match erscheinen.
        const auto positive_check = rstore.retrieve({
            {"context_signature", similar_ctx}
        }, 10);
        expect_equal(static_cast<long>(positive_check.size()), 0L,
                     "rejected-signal: rejected memory must not appear in positive retrieve");

        // Das rejected Memory MUSS als Warnung in retrieve_negative erscheinen.
        const auto negative_signal = rstore.retrieve_negative({
            {"context_signature", similar_ctx}
        }, 10);
        expect_equal(static_cast<long>(negative_signal.size()), 1L,
                     "rejected-signal: rejected memory with matching context must appear in retrieve_negative");
        expect_equal(negative_signal[0]["memory_id"].get<std::string>(), "mem_pcc_rejected",
                     "rejected-signal: correct memory id in negative signal");
        expect_equal(negative_signal[0]["retrieval_warning"].get<std::string>(), "similar_memory_was_rejected",
                     "rejected-signal: negative memory carries retrieval_warning");
        expect_true(negative_signal[0].contains("match_explanation"),
                    "rejected-signal: negative memory carries match_explanation for KI-Prompt");
        expect_true(negative_signal[0].contains("match_coverage"),
                    "rejected-signal: negative memory carries match_coverage");
        expect_true(negative_signal[0]["match_explanation"].is_array() &&
                    !negative_signal[0]["match_explanation"].empty(),
                    "rejected-signal: match_explanation is non-empty — model can explain the warning");

        std::filesystem::remove_all(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
