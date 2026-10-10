# PI-Aufbewahrung und Loeschung — Entscheidungsvorlage

**Status: Vorschlag, nicht aktiviert und nicht als Nutzerentscheidung beschlossen.**
Der Altbestand wurde auf Nutzerwunsch verworfen. Diese Vorlage betrifft nur neue Daten.
Decision-Schreibpunkte und automatische Bereinigungen bleiben bis zur Freigabe deaktiviert.
Memory- und Jev-Persistenz sowie Preview-Objekte existieren bereits; deren Bereinigung ist noch nicht implementiert.

**Zusaetzlich beschlossen und implementiert:** reine Run-Dateiloeschung behaelt zentrale Lerndaten.
Config, Herkunft, Statistik und optionales PNG liegen im [Run-Lernarchiv](pi_run_learning_archive_de.md).
FITS werden nicht in die DB kopiert; Originale bleiben auf der Platte. Diese Nutzerentscheidung ist
unabhaengig von den noch offenen Freitext-/Session-Fristen.

## Vorgeschlagene Regeln

| Daten | Vorschlag | Verhalten |
|---|---|---|
| Preview-Geltung | 1800 Sekunden | Bereits implementiert; per `TILE_COMPILE_PI_PREVIEW_TTL_SECONDS` zwischen 1 und 86400 Sekunden. Ablauf ist kein Ablehnungsereignis. |
| Unangewendete Preview-Objekte | 7 Tage | Objektbereinigung ohne synthetischen Decision Record; relevante Plan-Metadaten muessen vorher im Record gesichert sein. |
| Jev-Rohantworten/Requests | 90 Tage | Auswertungssnapshots behalten Auswahl, Alternativen, Wahrscheinlichkeiten und Regelbasis; Rohdaten danach loeschen. |
| Expliziter Nutzer-Freitext | 30 Tage | Aus gespeichertem JSON redigieren; Audit-Link und nicht personenbezogene Grund-Codes behalten. |
| Konversationen | 90 Tage seit letzter Nachricht | Sitzung loeschen, Referenz als fehlend markieren; keine Behinderung von Run-Aktivierung oder neuem Thread. |
| Decision-Metadaten | Bis explizitem PI-Speicher-Reset | Keine automatische Loeschung von Lernketten; keine Freitexte, absoluten Pfade oder Rohantworten als Metadaten tarnen. |
| Akzeptierte Memories | Bis Review/Reset | Bisherige Review-Logik beibehalten; eine Session-Loeschung loescht keine daraus explizit akzeptierte Memory. |
| Backups | Hoechstens 14 Tage | Konsistente SQLite-Backup-API oder `VACUUM INTO`; keine unkoordinierten Kopien von DB/WAL. |

## Explizite Loeschaktionen

- Run-Dateiloeschung: Snapshot sichern, Datei-Lifecycle markieren, zentrale Lerndaten/Memories/Verlauf behalten.
  Kein impliziter Lern-Ausschluss und keine Freitext-Redaktion. Implementiert.
- Vom Lernen ausschliessen: expliziter Code, Daten behalten. Implementierter Exclusion-Endpunkt.
- Run/Bild einschliesslich Lerndaten vergessen: separate bestaetigte Aktion, gezielte Session-/Text-/Archiv-
  Bereinigung; noch zu implementieren. Kein stilles Entfernen akzeptierter allgemeiner Memories.
- Einzelrecord-Redaktion: Nutzertext entfernen, Grund-Codes und Audit behalten.
- PI-Speicher-Reset: nach bestaetigter Auswahl Memories, Jev, Records, Preview-Objekte und Kontext-/Session-Verweise
  entfernen. Konversationen muessen als eigene Auswahl sichtbar sein. Modelle und Regelkataloge sind keine Memories.
- Keine Veraenderung historischer Run-Artefakte ausser der ausdruecklich angeforderten Run-Loeschaktion.

## Physische Bereinigung und Sicherung

Ein SQL-UPDATE ist **keine** Garantie fuer physische Loeschung. Nach Redaktion sind WAL-Checkpoint und eine
explizite Seitenbereinigung erforderlich; Sicherungen und externe Kopien koennen alte Inhalte weiter enthalten.
Die konkrete Wartung muss mit offenen Verbindungen, fehlgeschlagenen Checkpoints, Dateisystemen und Restore
getestet werden. Ohne diese Tests keine Zusage einer forensisch sicheren Loeschung.

## Noch notwendige Implementierung

1. Versionierte Policy-Konfiguration und explizite Freigabe speichern.
2. Record-Schreibpunkte erst nach dieser Freigabe aktivieren.
3. Bereinigung als wiederholbaren Backend-Wartungsschritt mit Tests, Statusanzeige und Fehlerbericht implementieren.
4. Session-/Chat-Loeschung anhand zentraler Referenzen, nie anhand unkontrollierter Nutzerpfade.
5. Backup-Restore, WAL-Bereinigung und Redaktionsgrenzen dokumentieren und testen.

## Bekannte Grenze des Apply-Pfads

Die Config-Datei und SQLite sind zwei verschiedene Persistenzsysteme. Die aktuelle Preview-Anbindung verhindert
normale Doppelklicks und erkennt Wiederholungen nach erfolgreichem Speichern des Apply-Ergebnisses.
Ein Absturz zwischen Config-Speicherung und SQLite-Abschluss ist noch nicht transaktional abgesichert.
Vor einer Zusage von crash-sicherem Exactly-once sind ein persistentes Apply-Intent und Recovery erforderlich.

Verwandt: [Decision-Trace-Plan](pi_decision_trace_plan_de.md),
[Umsetzungsstand](pi_decision_trace_und_unified_assistant_plan_de.md).
