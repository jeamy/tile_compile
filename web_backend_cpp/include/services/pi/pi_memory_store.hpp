#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>

namespace tile_compile::pi {

inline constexpr const char* kMemorySchemaVersion = "pi.memory.v2";
inline constexpr const char* kMemoryExportSchemaVersion = "pi.memories-export.v2";
inline constexpr const char* kMemoryRetrievalSchemaVersion = "pi.memory-retrieval.v2";

class PiMemoryStore {
public:
    explicit PiMemoryStore(std::filesystem::path memory_dir);

    const std::filesystem::path& memory_dir() const { return _memory_dir; }
    std::filesystem::path memories_path() const;
    std::filesystem::path reviews_path() const;
    std::filesystem::path outcomes_path() const;
    std::filesystem::path indices_path() const;
    std::filesystem::path legacy_memories_path() const;
    std::filesystem::path legacy_reviews_path() const;

    nlohmann::json append_candidate(nlohmann::json memory) const;
    nlohmann::json list(int limit = 100) const;
    nlohmann::json review(const std::string& memory_id,
                          const std::string& status,
                          const std::string& reviewer,
                          const std::string& note = "",
                          const nlohmann::json& outcome = nlohmann::json::object(),
                          const nlohmann::json& scope = nlohmann::json::object()) const;
    // Attaches measured-outcome data to a memory candidate without forcing a review status
    // transition (unlike review(), whose status must be one of allowed_review_status()). Used by
    // the outcome recorder (docs/PI/pi_local_learning_plan_de.md, Schritt 1c) so a run's measured
    // quality can be recorded automatically; the memory's current status is preserved.
    //
    // Writes to two places: reviews_path() (so list()'s existing "latest outcome wins" merge into
    // item["outcome"] keeps working for retrieval scoring, unchanged) AND outcomes_path(), an
    // accumulating append-only log that list() merges into item["outcomes"] (full history, never
    // overwritten) — Schritt 2's auto-promotion rule ("N >= 3 unabhängige Outcomes") needs to count
    // independent outcomes per memory_id, which a single latest-wins field cannot represent: a
    // memory applied across several runs (e.g. a queued batch sharing one config) must accumulate
    // one outcome per run, not have each new run's result silently replace the last.
    nlohmann::json attach_outcome(const std::string& memory_id,
                                  const nlohmann::json& outcome,
                                  const std::string& recorder = "pi_outcome_recorder",
                                  const std::string& note = "") const;
    nlohmann::json retrieve(const nlohmann::json& query, int limit = 10) const;
    nlohmann::json retrieve_negative(const nlohmann::json& query, int limit = 10) const;
    nlohmann::json indices() const;
    nlohmann::json rebuild_indices() const;
    nlohmann::json export_bundle(const std::string& privacy_class = "metadata_only",
                                 bool include_reviews = true) const;
    nlohmann::json import_bundle(const nlohmann::json& bundle,
                                 bool dry_run = false) const;
    nlohmann::json dedupe(bool dry_run = false) const;

private:
    std::filesystem::path _memory_dir;
};

} // namespace tile_compile::pi
