#include "services/pi/pi_analysis_context_store.hpp"
#include <chrono>
#include <iostream>
using namespace tile_compile::pi;
static void check(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
int main() {
    const auto root = std::filesystem::temp_directory_path() / ("pi_analysis_fixture_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        {
            auto db = PiDatabase::open(root);
            db->meta_set("v7_marker", "keep");
            db->execute("DROP TABLE analysis_proposals"); db->execute("DROP TABLE analysis_revisions"); db->execute("DROP TABLE analysis_contexts");
            db->execute("PRAGMA user_version = 7");
        }
        std::string uid;
        {
            PiAnalysisContextStore store(root);
            check(PiDatabase::open(root)->meta_get("v7_marker") == "keep", "v7 data preserved");
            AdviceRequest r; r.scan_id = "scan_a"; r.scan = {{"input_path", "/fixture/a"}};
            r.dataset_manifest = nlohmann::json::array({{{"path", "light.fits"}, {"size", 100}, {"mtime", 42}}});
            r.base_config = {{"bge", {{"enabled", false}}}}; r.metrics = {{"fwhm", 2.5}};
            const auto first = store.resolve(r); uid = first["analysis_uid"];
            check(store.resolve(r) == first, "stable context and revision");
            r.base_config["bge"]["enabled"] = true;
            const auto edited = store.resolve(r);
            check(edited["analysis_uid"] == uid && edited["revision_id"] != first["revision_id"], "draft revision stays in same thread");
            r.dataset_manifest[0]["mtime"] = 43;
            const auto changed = store.resolve(r);
            check(changed["analysis_uid"] != uid, "changed dataset is not silently merged");
            PiJevStore jev(root / "pi_decisions"); jev.put("proposal_a", "status", {{"state", "running"}});
            store.link(uid, first["revision_id"], "proposal_a"); store.link(uid, first["revision_id"], "proposal_a");
            check(store.proposals(uid).size() == 1 && store.proposals(changed["analysis_uid"]).empty(), "idempotent isolated association");
            bool rejected = false;
            try { store.link(changed["analysis_uid"], changed["revision_id"], "proposal_a"); } catch (const std::exception&) { rejected = true; }
            check(rejected, "proposal cannot move contexts");
            jev.put("proposal_b", "status", {{"state", "running"}}); rejected = false;
            try { store.link(uid, changed["revision_id"], "proposal_b"); } catch (const std::exception&) { rejected = true; }
            check(rejected, "revision must belong to context");
            check(store.revisions(uid).size() == 2, "config revisions retained");
        }
        { PiAnalysisContextStore reopened(root); check(reopened.get(uid).has_value() && reopened.proposals(uid).size() == 1, "reopen retains thread links"); }
        std::filesystem::remove_all(root); // Owned fixture only.
        std::cout << "Analysis context fixtures passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
