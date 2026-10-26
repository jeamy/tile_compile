# Analyse-Kontext — Scan-Jev-Grundbaustein

## Implementiert

Schema v8 erweitert bestehende frische PI-Datenbanken v2–v7 additiv um `analysis_contexts`,
`analysis_revisions` und `analysis_proposals`. Keine Legacy-JSONL-/v1-Importmigration, keine neuen
Decision Records und keine Memory-Promotion.

`PiAnalysisContextStore` leitet die Identitaet ausschliesslich aus dem erfolgreichen Backend-Scan und
dem vorhandenen serverseitigen Dataset-Manifest ab. Der Schluessel ist SHA256 aus Scan-Job-ID und Manifest.
Ein neuer Scan-Job oder geaenderte Manifest-Eintraege bekommen einen separaten Kontext, keine automatische
Zusammenfuehrung nach Objektname oder Input-Basename. Raw-FITS werden weder kopiert noch gehasht oder archiviert;
der Manifestvergleich nutzt die bestehende Pfad-/Groessen-/mtime-Provenienz.

Config-Entwurf, Metriken und Locked-Paths bestimmen die unveraenderliche Revision. Config-/Metrik-Aenderungen
bleiben bei identischem Scan/Manifest im selben Analyse-Thread. Die Registry speichert Signaturen und begrenzte
Revisionsmetadaten, keinen neuen vollstaendigen Transcript-/Raw-Request-Export. Die existierende Jev-Persistenz
behaelt ihre Proposal-Dokumente. Kontext/Revision/Proposal-Links sind gegen Update und Fremdzuordnung geschuetzt.

Neue `POST /api/scan/decisions`-Anfragen registrieren Kontext und Revision, schreiben zentrale Proposal-Links
atomar mit der Proposal-Erzeugung und liefern `analysis_context` mit. Die normale asynchrone Jev-Ausfuehrung
bleibt unveraendert. Optional erwartete `analysis_uid` muss zum aktuellen serverseitigen Scan passen, sonst 409
vor Proposal-Erzeugung oder Provider-Aufruf. Browser-Session-Metadaten sind nicht die Identitaetsautoritaet.
Alte unzugeordnete Proposals werden nicht nachtraeglich erraten oder einem neuen Scan zugeordnet.

## API

- `POST /api/pi/assistant/analysis-context`: vorhandener erfolgreicher Scan und vorhandene Metriken plus
  expliziter YAML-Entwurf/Locks; liefert stabile UID und konkrete Revision. Startet keinen Scan/Provider.
- `GET /api/pi/assistant/analysis-contexts/<uid>`: Kontext, letzte 100 Revisionsmetadaten und Proposal-Links.
- `POST /api/pi/assistant/select-analysis` mit `analysis_uid`: bekannte zentrale Auswahl, ohne Provider-Session.
- `GET /api/pi/assistant/analysis-thread/<uid>`: letzte 100 zugeordnete Proposal-Ansichten, maximal 8 MiB.
  `truncated=true` kennzeichnet die Byte-Grenze; gespeicherte Ereignisse werden nicht geloescht.
- Capabilities: `analysis_contexts=true`, `durable_scan_jev=true`; **`analysis_chat=false` bleibt korrekt**.

Thread-Abruf verwendet `DecisionService::view`, nicht rohe Proposal-JSONs: Shadow-Updates bleiben verborgen,
Modus/Status/Fehler und Kandidaten bleiben sichtbar. Abruf oeffnet keine Provider-Verbindung und benoetigt
keinen aktuell erreichbaren Scan-/Raw-Pfad. Bei fehlender Registry/Proposal-Ansicht wird nichts erfunden.

## Dock

Die bestehende Jev-Seite oeffnet nach neuer Anfrage den gelieferten Analyse-Kontext im Dock. Der angeheftete
Analyse-Kontext wird als Layout-/Identitaetsmetadatum gespeichert und nach Reload wieder ausgewaehlt.
Run- und Analyse-Threads bleiben getrennt. Kontextauswahl nutzt dieselbe serialisierte Queue; ueberholte
Antworten koennen den sichtbaren Kontext nicht ersetzen. Refresh laedt laufende/done/failed-Proposal-Zustaende.
Die begrenzte Analyseansicht und Byte-Kuerzung werden sichtbar benannt.

Dieser Adapter ist bewusst **lesend**. PI-Analyse-Chat, neue Jev-Anfragen aus dem Dock und Parameteruebernahme
in den Analyse-Entwurf sind noch nicht angebunden. Bestehende AI-/Jev-Seiten mit ihren Preview-/Apply-Wegen
bleiben erhalten und werden nicht vorzeitig entfernt. Es gibt keinen automatischen Modell-/Scan-Aufruf beim
Oeffnen oder Wiederherstellen des Kontexts.

## Noch offen

- Dauerhafte freie PI-Analyse-Unterhaltung und Aufnahme bestehender Scan-AI-Ergebnisse mit gleicher,
  serverseitig gesicherter Identitaet; alte globale AI-Caches sind keine autoritative Thread-Zuordnung.
- Gemeinsame Analyse-Entwurfs-/Preview-Karten mit Revision-/Dataset-Drift-Pruefung.
- Explizite Analyse-zu-Run-Abstammungslinks beim Run-Start; keine stille Thread-Zusammenfuehrung.
- Pagination aelterer Proposal-/Revision-Fenster und automatische rein lesende Statusaktualisierung.
- Bild-Adapter, gespeicherter Apply-Recovery sowie freigegebene Retention-/Vergessen-Aktionen.

## Verifikation

Store-Fixtures pruefen v7-Upgrade/Datenbestand, idempotente Identitaet, Config-Revision, Manifestwechsel,
Proposal-Isolation/Fremdrevision und Reopen. HTTP-Fixtures pruefen historischen Abruf/Auswahl ohne aktiven
Scan, unbekannte IDs, fehlenden Entwurf und Shadow-Update-Schutz. Statische Dock-Fixtures pruefen Analyse-
Reload und Trennung vom Run-Thread; keine produktiven Dienste oder Runs werden gestartet.
