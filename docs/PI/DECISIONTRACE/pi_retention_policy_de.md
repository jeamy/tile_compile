# PI-Aufbewahrung und Loeschung — beschlossene Policy

**Status: Policy durch Nutzeranweisung freigegeben.** Sie gilt fuer ab Freigabe neu erzeugte PI-Daten; verworfene Altbestaende werden nicht importiert. Diese Freigabe ist eine Produktentscheidung, keine Behauptung, dass die Bereinigung bereits implementiert sei. Decision-Schreibpunkte und automatische Bereinigungen bleiben gesperrt, bis die unten genannten technischen Gates erfuellt und getestet sind.

**Bereits implementiert:** Reine Run-Dateiloeschung behaelt zentrale Lerndaten. Config, Herkunft, Statistik und optionales PNG liegen im [Run-Lernarchiv](pi_run_learning_archive_de.md). FITS werden nicht in die Datenbank kopiert; Originale bleiben auf der Platte. Dateiloeschung ist weder Lern-Ausschluss noch Vergessen.

## Verbindliche Aufbewahrungsregeln

| Daten | Aufbewahrung | Verbindliches Verhalten |
|---|---|---|
| Action-Plan-Preview | 30 Tage ab Erstellung, unabhaengig von der 1800-Sekunden-Preview-Gueltigkeit | Nach Ablauf des fachlichen TTL bleibt ein Status fuer Diagnose/Verknuepfung erhalten; nach 30 Tagen wird das Preview-Objekt entfernt. Kein synthetischer Decision Record. Ein bereits verknuepfter Decision Record behaelt nur die fuer seine Kette erforderlichen IDs/Hashes, keinen kompletten Plan- oder Config-Text. |
| Jev-Requests und rohe Provider-Antworten | 90 Tage ab Erzeugung | Danach Rohrequest/-antwort entfernen. Ein separater, minimierter Entscheidungs-Snapshot darf Auswahl, Kandidaten, Wahrscheinlichkeiten, Konfidenz, Regel-/Build-Referenz und Kosten enthalten, sofern er keine Freitexte oder absoluten Pfade enthaelt. |
| Nutzer-Freitext in Records | 30 Tage ab Record-Zeitpunkt | Freitext redigieren; Record-ID, Ereignisart, Zeitstempel und nicht-personenbezogene Reason-Codes bleiben als Audit-/Lernmetadaten erhalten. Freitext wird nicht in Exporten ausgegeben. |
| Assistant-/Run-/Bild-Konversationen | 90 Tage nach letzter Nachricht | Konversation und Session-Daten entfernen; ein minimaler Verweis darf den Zustand `expired` anzeigen, aber keine Nachrichtentexte enthalten. Aktivierung eines Runs und Anlegen eines neuen Threads duerfen dadurch nicht blockiert werden. |
| Decision-Record-Metadaten | Bis explizitem PI-Speicher-Reset oder gezieltem Vergessen des zugeordneten Kontexts | Keine automatische Altersloeschung. Keine absoluten Pfade, Rohbilder, Rohantworten oder Freitexte als Metadaten speichern. |
| Akzeptierte allgemeine Memories | Bis Nutzer-Review, gezieltem Entfernen oder PI-Speicher-Reset | Session- oder Run-Kontext-Loeschung entfernt keine unabhaengig akzeptierte allgemeine Memory automatisch. |
| Run-Lernarchiv | Bis gezieltem Vergessen der Run-UID oder PI-Speicher-Reset | Reine Run-Dateiloeschung behaelt das Archiv. Vergessen loescht zentrale Run-Snapshots und run-spezifische Daten; kein automatischer Lern-Ausschluss als Ersatz. |
| Sidecar-Traffic-Logs mit Prompt-/Antworttext | 30 Tage | Aeltere gueltig datierte Zeilen werden bei Sidecar-Logging-Maintenance entfernt; Zeilen mit ungueltigem/fehlendem Zeitstempel werden verworfen, da ihre Aufbewahrungsdauer nicht verifiziert werden kann. |
| Sicherungen der PI-Datenbank | Maximal 14 Tage | Nur konsistente SQLite-Backups per SQLite Backup API oder `VACUUM INTO`. DB-/WAL-Dateien nicht unabhaengig kopieren. Nach Loeschung/Redaktion koennen Backups die alten Inhalte bis zum Ablauf ihrer Frist enthalten; Restore muss die Loesch-/Redaktionsmarker erneut anwenden, bevor Daten lesbar werden. |

Die Fristen sind Obergrenzen. Nutzer koennen Daten vorher gezielt loeschen oder redigieren. Diese Policy gilt fuer die lokale PI-Speicherung in der Anwendung; sie kann keine vom Nutzer ausserhalb der Anwendung erstellten Kopien kontrollieren.

## Explizite Aktionen und Umfang

- **Run-Dateien loeschen:** Vor dem Entfernen Run-Lerndaten sichern. Zentrale Records, Konversationen und Memories bleiben bestehen; kein impliziter Ausschluss und keine Freitext-Redaktion. Implementiert.
- **Vom Lernen ausschliessen:** Lerndaten bleiben erhalten, werden aber fuer kuenftige Lern-/Retrieval-Pfade ausgeschlossen. Keine Loeschung. Implementiert.
- **Run/Bild vergessen:** Separate, bestaetigungspflichtige und gezielte Aktion. Loescht den Run-Archiv-Snapshot, run-/bildgebundene Decision Records und Links, Freitexte sowie Konversationen/Sessions dieses Kontexts aus dem aktiven Store. Generalisierte akzeptierte Memories bleiben erhalten, ausser der Nutzer waehlt sie einzeln aus. Erzeugt keinen negativen Outcome. Implementiert (Backend-Endpunkt, ohne UI-Anbindung); die externe Tombstone-Reconciliation schuetzt verwaltete Restores.
- **Einzelrecord redigieren:** Entfernt den Nutzer-Freitext aus dem aktiven Record; Reason-Codes und nicht-sensitive Audit-Metadaten bleiben erhalten. Ein externer Redaktionsmarker schuetzt verwaltete Backup-Restores. Implementiert.
- **PI-Speicher-Reset:** Bestaetigte, nach Datenkategorien aufgeschluesselte Aktion. Mindestens Memories, Jev-Daten, Decision Records, Preview-Objekte, Konversationen/Sessions und Run-Lernarchive muessen einzeln sichtbar/auswaehlbar sein. Der Nutzer kann eine Kategorie nicht versehentlich durch Loeschung einer anderen verlieren. Regelkataloge und Software/Modelle sind keine PI-Lerndaten.
- **Historische Run-Artefakte:** Keine Aenderung ausser bei einer ausdruecklich angeforderten Run-Dateiloeschung. Das Vergessen eines PI-Kontexts loescht keine Original-Lights/Darks ausserhalb des Run-Archivs.

## Physische Bereinigung und Backups

Eine SQL-Aenderung oder das Setzen eines Redaktionsmarkers ist **keine** physische Loeschgarantie. Nach Bereinigung muessen offene SQLite-Verbindungen beruecksichtigt, WAL konsistent gecheckpointet und Datenbankseiten kontrolliert neu geschrieben werden. Backups muessen spaetestens nach 14 Tagen auslaufen. Bei Restore muessen Tombstones/Redaktionsmarker aus einem separaten, ebenfalls gesicherten Loeschjournal angewandt werden, bevor die restaurierte Datenbank fuer Nutzer/Modelle erreichbar ist. Kann das Loeschjournal nicht gelesen werden, darf die restaurierte Datenbank nicht freigegeben werden.

Die Anwendung verspricht keine sofortige forensisch sichere Vernichtung von Daten in Dateisystem-Snapshots, externen Backups oder vom Nutzer erzeugten Kopien. Fehler bei Checkpoint, Seitenbereinigung, Backup-Ablauf oder Restore-Reconciliation sind sichtbar zu melden; sie duerfen nicht als erfolgreiche Loeschung bestaetigt werden.

## Implementierungsstand und verbleibende Freigabegates

Implementiert: Policy v1 wird in `meta` der PI-SQLite-Datenbank persistiert und ueber `GET /api/pi/retention` mit Wartungsstatus angezeigt. Der Backend-Wartungsjob laeuft beim Start und danach taeglich; `POST /api/pi/retention/maintenance` erlaubt einen bestaetigten manuellen Lauf. Er entfernt abgelaufene Preview-Objekte (30 Tage), alte Jev-Request-/Response-Rohpayloads (90 Tage), redigiert abgelaufenen Nutzer-Freitext (30 Tage) und entfernt inaktive JSON-Konversationen aus den zentralen PI-Verzeichnissen (90 Tage nach letzter Dateiaenderung). Die Bereinigung ist wiederholbar; unbekannte/ungueltige Zeitstempel werden fail-closed behalten. Nach DB-Aenderungen werden WAL gecheckpointet und die DB-Seiten mit `VACUUM` neu geschrieben. Es werden keine Dateien in Run-Verzeichnissen angefasst.

`POST /api/pi/retention/run/<run_uid>/forget` loescht nach bestaetigter UID die zentralen Run-Lerndaten, Assistant-Ereignisse, Decision Records/Links und zentralen Konversationsdateien; ein inhaltsfreier Tombstone bleibt. `POST /api/pi/retention/reset` verlangt eine bestaetigte, explizite Kategorienliste (`memories`, `jev`, `decision_records`, `previews`, `conversations`, `run_learning`). Ein PI-Reset loescht keine Runner-Artefakte, Regelkataloge oder Modelle. Diese Endpunkte loeschen nur aktive Store-Daten; sie garantieren keine sofortige physische Vernichtung externer Kopien.

Managed Backups werden beim Wartungslauf per SQLite Backup API angelegt und nach maximal 14 Tagen entfernt. Ein fsync-gesichertes, ausserhalb der Backup-Datei liegendes Loeschjournal wird vor Vergessen, Reset und Freitext-Redaktion geschrieben. Bestaetigtes Restore validiert Backup, Manifest und Journal, reconciled in einem isolierten Staging-Store und uebernimmt erst danach den Store. Ein Restore-Marker ermoeglicht Start-Recovery; ein fehlgeschlagener Austausch sperrt den DB-Zugriff fail-closed. `GET /api/pi/decision-records/export` liefert nur strukturierte Metadaten und Reason-Codes, ohne Freitext, Jev-Payloads oder Subject-Pfade.

Die Konversationsbereinigung ist auf die flachen JSON-Dateien in `context_chat/`, `run_chat/` und `live_image_chat/` begrenzt und verwendet deren Schreibzeitpunkt als letzte Nachrichtenaktivitaet. Legacy-Konversationen in Run-Artefakten werden absichtlich nicht veraendert. Sidecar-Agent-Sessions verwenden `SessionManager.inMemory()` und werden nach jedem Request disposed; persistente Sidecar-Traffic-Logs werden bei Lese-/Schreibzugriff und danach taeglich nach 30 Tagen bereinigt; Zeilen mit ungueltigem/fehlendem Zeitstempel werden verworfen.

Noch offene Gates:

1. Physische Seitenbereinigung wird versucht und Fehler werden im Wartungsstatus gemeldet. Der Checkpoint-Fehler unter einem unabhaengigen offenen WAL-Leser ist getestet; Der VACUUM-Fehlerpfad ist getestet (Test-Hook vor VACUUM: Fehler wird gemeldet, Status `failed`, vorherige Bereinigung bleibt, Retry erfolgreich). Plattformvarianten (insbesondere die Windows-Verzeichnis-Durability) bleiben nicht verifiziert.
2. Retention-Fristgrenzen, Wiederholung, Kategorienreset, konsistentes SQLite-Backup, Tombstone-Replay, fehlendes Journal, Restore-Recovery und Ablaufgrenze sind getestet. Die bestehenden Exportpfade fuer Memories und Decision-Record-Metadaten sind ohne Record-Freitext verifiziert. Ein vollstaendiger anwendungsweiter Datenexport ist nicht Teil der freigegebenen Policy und wird nicht behauptet.
3. Permanente Decision-Record-Schreibpunkte bleiben deaktiviert, bis die Plattformvarianten (Windows-Durability) verifiziert sind. Der VACUUM-Fehlerpfad ist nicht mehr offen.

## HTTP

- `GET /api/pi/retention`: Policy-Version, konfigurierte Fristen und Status des letzten Wartungslaufs.
- `GET /api/pi/decision-records/export`: strukturierter Decision-Record-Metadatenexport ohne Nutzer-Freitext, Basis-Freitext, Jev-Payloads oder Subject-Pfade.
- `POST /api/pi/retention/maintenance`: `{ "confirmed": true }`; fuehrt Retention aus und erstellt/altert managed Backups.
- `POST /api/pi/retention/backup`: `{ "confirmed": true }`; erstellt einen konsistenten SQLite-Backup-Snapshot.
- `POST /api/pi/retention/restore`: `{ "confirmed": true, "backup_id": "…" }`; spielt externe Loeschmarker vor Freigabe erneut ein.
- `POST /api/pi/retention/run/<run_uid>/forget`: `{ "confirmed": true }`.
- `POST /api/pi/retention/reset`: `{ "confirmed": true, "categories": ["memories", "conversations"] }`.

## Bekannte Grenze des Apply-Pfads

Die Config-Datei und SQLite sind zwei verschiedene Persistenzsysteme. Die aktuelle Preview-Anbindung verhindert normale Doppelklicks und erkennt Wiederholungen nach erfolgreichem Speichern des Apply-Ergebnisses. Ein Absturz zwischen Config-Speicherung und SQLite-Abschluss ist noch nicht transaktional abgesichert. Vor einer Zusage von crash-sicherem Exactly-once sind ein persistentes Apply-Intent und Recovery erforderlich.

Verwandt: [Decision-Trace-Plan](pi_decision_trace_plan_de.md), [Umsetzungsstand](pi_decision_trace_und_unified_assistant_plan_de.md).
