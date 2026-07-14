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
        const auto first = store.append_candidate({
            {"type", "optimization"},
            {"source_session_id", "pi_sess_fixture"},
            {"recommendation", {
                {"patch", nlohmann::json::array({{{"path", "bge.model"}, {"value", "rbf"}}})}
            }},
            {"evidence", {{"validation", "fixture"}}},
            {"outcome", {{"validation_valid", true}, {"applied_count", 1}}}
        });

        expect_equal(first["schema_version"].get<std::string>(), "pi.memory.v1", "memory schema");
        expect_equal(first["status"].get<std::string>(), "candidate", "memory default status");
        expect_equal(first["privacy_class"].get<std::string>(), "metadata_only", "memory default privacy");
        expect_true(!first["memory_id"].get<std::string>().empty(), "memory id generated");
        expect_true(std::filesystem::is_regular_file(store.memories_path()), "memory jsonl exists");

        store.append_candidate({
            {"memory_id", "mem_fixture_second"},
            {"type", "failure"},
            {"status", "candidate"},
            {"privacy_class", "metadata_only"},
            {"avoid", {{"path", "bge.model"}, {"value", "classic"}}}
        });
        store.append_candidate({
            {"memory_id", "mem_fixture_lower_score"},
            {"type", "optimization"},
            {"status", "candidate"},
            {"privacy_class", "metadata_only"},
            {"recommendation", {
                {"patch", nlohmann::json::array({{{"path", "bge.model"}, {"value", "poly"}}})}
            }},
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

        const auto string_path_matches = store.retrieve({
            {"type", "optimization"},
            {"paths", nlohmann::json::array({"bge.model"})}
        }, 5);
        expect_equal(static_cast<long>(string_path_matches.size()), 2L,
                     "memory retrieval accepts string path lists");

        const auto duplicate = store.append_candidate({
            {"type", "optimization"},
            {"source_session_id", "pi_sess_fixture"},
            {"recommendation", {
                {"patch", nlohmann::json::array({{{"path", "bge.model"}, {"value", "rbf"}}})}
            }},
            {"evidence", {{"validation", "fixture"}}}
        });
        expect_true(!duplicate.value("created", true), "duplicate memory is not appended");
        expect_true(duplicate.value("duplicate", false), "duplicate memory is flagged");
        expect_equal(static_cast<long>(store.list().size()), 3L, "duplicate memory keeps list count");

        const auto exported = store.export_bundle("metadata_only", true);
        expect_equal(exported["schema_version"].get<std::string>(), "pi.memories-export.v1",
                     "memory export schema");
        expect_equal(static_cast<long>(exported["memory_count"].get<size_t>()), 3L,
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
            store.append_candidate({{"type", "optimization"}, {"status", "accepted"}});
        } catch (const std::invalid_argument&) {
            rejected_non_candidate = true;
        }
        expect_true(rejected_non_candidate, "non-candidate memory rejected");

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
