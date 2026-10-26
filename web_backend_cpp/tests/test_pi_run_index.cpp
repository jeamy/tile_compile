#include "services/pi/pi_run_index.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>

using namespace tile_compile::pi;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("pi_run_index_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(root / "runs" / "one");
        std::filesystem::create_directories(root / "runs" / "two");
        std::string uid;
        {
            PiRunIndex index(root / "store");
            const auto first = index.resolve(root / "runs" / "one", "", "hash", "start");
            uid = first.at("run_uid");
            const auto again = index.resolve(root / "runs" / "one" / ".." / "one");
            check(again.at("run_uid") == first.at("run_uid"), "normalized paths share identity");
            const auto other = index.resolve(root / "runs" / "two", "", "hash", "start");
            check(other.at("run_uid") != first.at("run_uid"), "fingerprint never silently merges");
            check(index.candidates("hash", "start").size() == 2, "fingerprint candidates only");
            std::filesystem::rename(root / "runs" / "one", root / "runs" / "renamed");
            const auto moved = index.relink(uid, root / "runs" / "renamed");
            check(moved.at("run_uid") == first.at("run_uid"), "explicit provenance UID links rename");
            check(moved.at("run_keys").size() == 2, "path aliases retained");
            bool conflict = false;
            try { index.resolve(root / "runs" / "two", uid); }
            catch (const std::runtime_error&) { conflict = true; }
            check(conflict, "conflicting UID cannot silently reassign path");
            check(std::filesystem::is_empty(root / "runs" / "renamed"), "historical run untouched");
            check(index.relink(uid, root / "runs" / "renamed").at("run_keys").size() == 2, "relink retry is idempotent");
            bool provenance_conflict = false;
            try { index.relink(uid, root / "runs" / "renamed", other.at("run_uid")); }
            catch (const PiRunIdentityConflict&) { provenance_conflict = true; }
            check(provenance_conflict, "foreign provenance refused");
            bool alias_conflict = false;
            try { index.relink(uid, root / "runs" / "two"); }
            catch (const PiRunIdentityConflict&) { alias_conflict = true; }
            check(alias_conflict, "foreign alias refused");
            bool missing_identity = false;
            try { index.relink(PiRunIndex::generate_uid(), root / "runs" / "renamed"); }
            catch (const std::out_of_range&) { missing_identity = true; }
            check(missing_identity, "relink cannot create unknown identity");
            check(index.candidates("", "start").empty(), "incomplete fingerprint not matched");
        }
        {
            PiRunIndex index(root / "store");
            check(index.resolve(root / "runs" / "one").at("run_uid").get<std::string>() == uid, "identity survives reopen");
        }
        std::filesystem::remove_all(root);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
}
