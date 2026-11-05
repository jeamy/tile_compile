#include "services/pi/pi_preview_store.hpp"
#include "services/pi/pi_user_reason_catalog.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <stdexcept>

using namespace tile_compile::pi;
using nlohmann::json;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("pi_preview_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        const auto catalog = load_user_reason_catalog(std::filesystem::path(__FILE__).parent_path().parent_path() /
                                                       "config" / "pi_user_reason_codes_v1.json");
        check(user_reason_codes_valid(catalog, "1", "config", json::array({"artifacts"})), "config reason accepted");
        check(user_reason_codes_valid(catalog, "1", "image", json::array({"halo"})), "image reason accepted");
        check(!user_reason_codes_valid(catalog, "1", "config", json::array({"halo"})), "wrong domain rejected");
        check(!user_reason_codes_valid(catalog, "2", "config", json::array({"artifacts"})), "wrong version rejected");
        check(!user_reason_codes_valid(catalog, "1", "config", json::array({"validated"})), "policy codes rejected");
        check(!user_reason_codes_valid(catalog, "1", "image", json::array({"halo", "halo"})), "duplicates rejected");
        {
            // Old database and JSONL files are not opened by the new store.
            std::filesystem::create_directories(dir);
            std::ofstream old_db(dir / "pi_store_v1.sqlite");
            old_db << "not even a valid SQLite database";
            std::ofstream old_jsonl(dir / "decisions_v1.jsonl");
            old_jsonl << "legacy fixture";
        }
        const json plan = {{"actions", json::array()}, {"schema_version", "pi.action-plan.v1"}};
        std::string id;
        {
            PiPreviewStore store(dir);
            auto db = PiDatabase::open(dir);
            check(db->schema_version() == kPiDatabaseSchemaVersion, "fresh current schema");
            check(db->meta_get("test_marker").empty(), "no legacy metadata");
            const auto first = store.create(plan, {{"value", 1}}, 100, 60);
            const auto second = store.create(plan, {{"value", 2}}, 100, 60);
            id = first.at("preview_id");
            check(id != second.at("preview_id").get<std::string>(), "distinct preview IDs");
            check(first.at("action_plan_id") == second.at("action_plan_id"), "stable plan identity");
            check(first.at("config_sha256") != second.at("config_sha256"), "config basis tracked");
            check(store.get(id, 159)->at("state") == "pending", "pending before expiry");
            check(store.get(id, 160)->at("state") == "expired", "expiry boundary");
            check(pi_sql_text(db->query("SELECT state FROM action_previews WHERE preview_id = ?", {id})[0][0]) == "pending",
                  "expiry does not persist synthetic event");
            check(store.transition(id, "applied"), "expired preview can be applied after caller revalidation");
            check(store.transition(id, "applied"), "idempotent transition");
            check(!store.transition(id, "dismissed"), "terminal state cannot be changed");
            check(store.transition(second.at("preview_id"), "dismissed"), "explicit dismissal");
            check(!store.get("unknown", 100), "missing preview");
            check(!store.transition("unknown", "applied"), "missing transition");
            bool invalid = false;
            try { store.create(plan, json::object(), 100, 0); }
            catch (const std::invalid_argument&) { invalid = true; }
            check(invalid, "invalid TTL rejected");
            check(PiPreviewStore::action_plan_id(plan) != PiPreviewStore::action_plan_id({{"actions", 1}}), "plan changes identity");
        }
        {
            PiPreviewStore reopened(dir);
            check(reopened.get(id, 1000)->at("state") == "applied", "state survives reopen");
        }
        std::filesystem::remove_all(dir);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::filesystem::remove_all(dir);
        return 1;
    }
}
