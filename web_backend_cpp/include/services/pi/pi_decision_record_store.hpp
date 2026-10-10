#pragma once
// Decision-Trace-Store (docs/PI/DECISIONTRACE/pi_decision_trace_plan_de.md, §2.1/2.2/2.8).
//
// SQLite-Tabellen `decision_records` (im Normalbetrieb append-only; Trigger verbieten DELETE ausserhalb
// expliziter Retention-Erasure und verbieten Aendern der Schluesselspalten), `decision_reasons` (Filter/Aggregation)
// und `decision_links` (append-only
// Ereignisse: memory, outcome, supersedes, run_deleted, redact). Leser mergen den Link-Stand ueber
// `decision_id`. Einzige erlaubte Aenderung an einem Record ist die Redaktion von
// `rationale.user.text` (im gespeicherten JSON, in derselben Transaktion wie der redact-Link).
// Keine Garantie fuer physische Loeschung aus WAL, freien Seiten oder Backups.
//
// Hinweis zum Namensraum: `pi_decision_*` (service/policy/state/outcome) gehoert zur Jev-Decision-API.
// Dieser Store ist davon unabhaengig und heisst bewusst "decision record".

#include "services/pi/pi_database.hpp"

#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace tile_compile::pi {

inline constexpr const char* kDecisionRecordSchemaVersion = "pi.decision-record.v1";
inline constexpr const char* kDecisionLinkSchemaVersion = "pi.decision-link.v1";
inline constexpr std::size_t kDecisionRationaleTextMaxChars = 1000;

// Prueft ein Record gegen die serverseitigen Regeln (Plan §2.2): erlaubte kind/actor-Paare,
// Rationale-Struktur, actor x basis-Regeln, no_reason_given XOR reason_codes/text, keine absoluten
// Pfade in evidence.metrics_ref. Wirft std::invalid_argument mit sprechender Meldung.
void validate_decision_record(const nlohmann::json& record);

// Entfernt absolute Pfade aus Freitext und begrenzt die Laenge (Plan §2.1 Metadata-only-Schutz).
std::string scrub_rationale_text(const std::string& text);

class PiDecisionRecordStore {
public:
    explicit PiDecisionRecordStore(std::filesystem::path dir);

    const std::filesystem::path& dir() const { return _dir; }
    std::filesystem::path database_path() const;

    // Normalisiert (Defaults, Text-Scrubbing, privacy_class), validiert und haengt an. Ist
    // `idempotency_key` gesetzt und bereits vorhanden, wird der bestehende Record mit
    // `duplicate=true` zurueckgegeben, ohne zu schreiben.
    nlohmann::json append(nlohmann::json record) const;

    // Einzelner Record mit gemergtem Overlay-Stand; null, wenn unbekannt.
    nlohmann::json get(const std::string& decision_id) const;

    // Filter (alle optional): kind, actor, reason_code, context_id, run_uid, parent_decision_id, since.
    // Liefert die letzten `limit` Treffer in Schreibreihenfolge.
    nlohmann::json list(const nlohmann::json& filter = nlohmann::json::object(), int limit = 200) const;

    // Overlay-Link. Typen: memory, outcome, supersedes, run_deleted, redact.
    nlohmann::json add_link(const std::string& decision_id,
                            const std::string& type,
                            const nlohmann::json& data = nlohmann::json::object()) const;

    // Redaktion: schreibt den redact-Link und entfernt rationale.user.text aus dem JSON (setzt
    // text_redacted und privacy_class=metadata_only) in einer Transaktion.
    nlohmann::json redact(const std::string& decision_id, const std::string& reason) const;

    // Markiert alle Records eines Run-Kontexts (run_uid bzw. image_id `<run_uid>:...`) als geloescht
    // und redigiert deren Freitext (Plan §3.5). Gibt die Anzahl betroffener Records zurueck.
    int mark_run_deleted(const std::string& run_uid) const;

private:
    std::shared_ptr<PiDatabase> db() const;

    std::filesystem::path _dir;
    mutable std::shared_ptr<PiDatabase> _db;
};

} // namespace tile_compile::pi
