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

        std::filesystem::remove_all(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
