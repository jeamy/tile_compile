# Unified Assistant Dock — erster Run-Adapter

## Implementiert

- `web_frontend_v3/js/assistant/dock.js` wird genau einmal im Hauptfenster gemountet und ueberlebt
  Tab-/Seitenwechsel. Rechts auf Desktop, einklappbares Overlay auf schmalen Ansichten.
- Breite, Offen-/Geschlossen-Zustand und explizit angeheftete Run-UID werden als Layout-/Kontext-Metadaten
  gespeichert. Nachrichten und Entwuerfe werden vom Dock nicht in localStorage gespeichert.
- Ohne angehefteten Kontext folgt der Adapter Aenderungen des aktuellen Runs. Ohne lokalen Run kann der
  zuletzt aktive Backend-Kontext wieder geladen werden. Run History bietet **Im Assistenten oeffnen**.
- Backend-Auswahl ist UID-basiert und persistiert `pi_active_context`, ohne Session zu oeffnen, Run-Dateien
  umzuschreiben oder den globalen Processing-Run automatisch zu wechseln.
- Der Run-Chat im Run Monitor entfällt, wenn das globale Dock gemountet ist. Standalone-/Legacy-Ansichten
  ohne Dock behalten den bisherigen Adapter. Alte Parameter-AI-/Jev- und Bild-Ansichten bleiben bis zu ihrer
  jeweiligen Migration bestehen; kein vorzeitiges Entfernen noch nicht uebernommener Funktionen.
- Ein Run-Thread mit PI-Antwortkarten und explizit ausgeloester Jev-Post-Run-Beratung. Neue Jev-Karten werden
  zentral in SQLite wiederhergestellt und als regelbasierte Backend-Beratung ohne Modellaufruf gekennzeichnet.
  AI-/Jev-Schalter werden respektiert. Vorschlaege im neuen Dock sind vorerst **nur lesbar**, ohne zweiten Apply-Sonderweg.
- Ein ausklappbares **Warum** zeigt die strukturierte Provider-Antwort und laedt vorhandene Kontext-Records
  erst beim Oeffnen. Kontext-Records werden nicht als Beweis fuer eine konkrete Antwort ausgegeben, solange
  die Ereignis-/Proposal-Zuordnung fehlt. LLM-Erklaerungen sind keine gemessene Ursache oder Promotion-Evidenz.
- Fehlende Run-Dateien: erhaltener Verlauf lesbar, neue Beratung deaktiviert. Altes Backend ohne
  UID-Thread-Faehigkeitsnachweis: Senden deaktiviert, keine stille Verwendung des alten Namens-Schluessels.

## Backend- und Session-Identitaet

Neue Run-Chat-Historien liegen unter `context_chat/<sha256(run:<uid>)>.json` in der zentralen PI-Ablage,
**nicht im Run-Verzeichnis**. Der SHA256-Dateiname ist plattformstabil; gleichnamige Runs verschiedener
Verzeichnisse teilen keinen Thread. Die Antwort und Historie enthalten `run_uid` und `context_id=run:<uid>`.

Bestehende zentrale Chat-JSONs und historische Artifact-JSONs koennen lesend als Fallback dienen. Bei Abruf
allein ueber UID werden nur exakt bekannte Pfad-Aliase bzw. deren Artefaktpfade betrachtet. Mehrdeutige,
ausschliesslich nach Basename gespeicherte Konversationen werden nicht automatisch verschiedenen Roots
zugeordnet. Altdateien werden nicht geloescht oder umgeschrieben; dies ist keine Memory-/JSONL-Importmigration.

Run-Chat-Anfragen pruefen einen optionalen `run_uid` gegen den tatsaechlichen Run-Pfad. Widerspruch: HTTP 409
vor Provider-Aufruf. Die bisherigen `run_id`-Aufrufe bleiben kompatibel, werden intern aber der UID zugeordnet.
Der Sidecar bekommt die UID als Session-Schluessel; Prompts/Run-Kontext enthalten weiterhin den wirklichen
Run-Pfad. Eine Provider-Session entsteht erst mit einer expliziten Chat-Anfrage, nicht beim Kontextwechsel.

Ein pro Speicher/UID gemeinsamer Mutex serialisiert komplette Chat-Turns und History-Merges im Backend-Prozess.
Dateischreiben verwendet den vorhandenen Atomic-Replace-Helper. Die bestehenden Grenzen von 24 Nachrichten/
Turns bleiben bestehen; keine neue zeitbasierte Loeschung oder Freitext-Redaktion wird aktiviert.
Mehrprozess-Schreibkoordination und Windows-Atomic-Replace sind noch separat zu verifizieren.

## HTTP

- `GET /api/pi/assistant/capabilities`: derzeit `run_uid_threads=true`, Analyse-/Bild-Chat und Decision-Writes false.
- `GET /api/pi/assistant/run-context?run_id=<Name oder Pfad>`: zentrale Run-Identitaet, keine Chat-Session.
- `POST /api/pi/assistant/select-context` mit `{"run_uid":"..."}`: bekannter Kontext, erreichbarer Alias wenn
  vorhanden; persistierte Auswahl und `artifacts_reachable`, auch ohne Run-Dateien.
- `GET /api/pi/run-chat/history?run_uid=<uid>`: Verlauf ohne benoetigte Run-Dateien.
- `POST /api/pi/run-chat`: bestehender Pfad plus optional erwartete UID; Antwort mit UID-/Kontext-Bezug.
- `POST /api/pi/run-chat/history`: kompatibler History-Merge, auch nach expliziter UID adressierbar.

## Dauerhafte Post-Run-Karten und Thread-Fenster

Schema v7 erweitert bestehende frische PI-Datenbanken additiv um `assistant_thread_events`. Die Tabelle enthaelt
unveraenderliche, UID-gebundene Snapshots explizit angefragter regelbasierter Post-Run-Beratung, keine Chat-
Duplikate, Raw-FITS, neuen Decision Records oder impliziten Memory-Promotions. Reine Run-Dateiloeschung
entfernt diese Karten nicht. Die freigegebene Vergessen-/Retention-Aktion bleibt gesondert zu implementieren.

`POST /api/pi/post-run/advice` akzeptiert neben bisherigen Namen auch erlaubte Run-Pfade sowie erwartete
`run_uid` und optionale `request_id`. Pfad-/UID-Konflikte werden vor Beratung mit 409 abgelehnt. Erfolgreiche
Beratung wird vor HTTP 200 zentral gespeichert; Speicherfehler ergeben 503. Ein gleicher Request-Schluessel
im selben Kontext mit gleichem Ergebnis liefert dieselbe Karte. Anderer Inhalt fuer dieselbe ID: 409, kein
Ueberschreiben. Das ist keine crash-sichere Apply-Transaktion; dieser Endpunkt startet/wendet nichts an.

`GET /api/pi/assistant/thread?run_uid=<uid>` vereinigt vorhandene PI-Turns mit neuen Jev-Karten. Optional
`include_pi=0` bzw. `include_jev=0` fuer die Anzeigeschalter. Lesen benoetigt keine Run-Dateien oder Sidecar-
Verbindung. Neue PI-Turns erhalten Event-ID und Millisekunden-Zeitstempel; alte Turns werden lesend abgebildet,
mit deterministischer Ersatz-ID und ohne Umschreiben. Fehlende alte Zeitstempel werden nicht erfunden.

Das Ansichtsfenster enthaelt maximal die letzten 24 PI-Turns und 100 Jev-Karten; Jev-Anteil max. 8 MiB,
ein gespeichertes Beratungsergebnis max. 4 MiB, gesamte Thread-Antwort max. 16 MiB. Die Jev-Fenstergrenze
loescht keine gespeicherten Ereignisse. PI-Historie behaelt ihre bestehenden 24-Turn-Schreibgrenzen.
Eine Pagination fuer aeltere Jev-Ereignisse bleibt offen. Das Dock weist das Fenster sichtbar aus.

Das Dock nutzt den vereinigten Endpunkt nur bei `thread_history=true`; Jev-Senden erfordert zusaetzlich
`durable_jev_post_run=true`. Gegen vorherige UID-Backends bleibt PI les-/nutzbar, aber es entstehen keine
scheinbar dauerhaften Jev-Karten. History-Requests haben eigene Generationen gegen spaete Antworten nach
Refresh oder Schalterwechsel.

## Schutz gegen Kontextdrift

- Kontextauswahl im Browser ist serialisiert; ueberholte Antworten werden verworfen.
- Senden bindet UID und Pfad vor dem Request. Kontextwechsel waehrend einer Anfrage darf deren Antwort nur
  im urspruenglichen Backend-Thread speichern, nicht in der neuen Ansicht.
- Entwuerfe und laufende Requests sind je Kontext getrennt; keine gemeinsame globale Nachrichtenliste.
- Verzeichnis-/UID-Widersprueche werden serverseitig abgewehrt, nicht durch Vertrauen in Browser-Zustand.

## Noch offen

Dies ist der **erste Adapter**, nicht der abgeschlossene Gesamtumbau:

- Analyse-Kontext samt Revisionen und echter Kontext-Chat-Verallgemeinerung.
- Bild-Facet `<run_uid>:live_edit`, Session-Eviction und gemeinsame Bildoperation-/Preview-Karten.
- Dauerhaftes Mergen der Jev-/PI-/Decision-Ereignisse in `assistant/thread` inklusive Vorfahren.
  Neue Post-Run-Jev-Karten sind bereits dauerhaft und UID-gebunden; Scan-Jev-Proposals und Decision-Links
  sind noch nicht in diesen Thread integriert. Vorher nur im Browser gehaltene Jev-Karten sind nicht rekonstruierbar.
- Gemeinsame Preview-/Apply-/Reason-Picker-Karten und Entfernen der alten Parameter-AI-/Jev-Huellen erst danach.
- Aktivierung permanenter Decision-Schreibpunkte und Vergessen bleiben hinter dem Retention-/Loeschpolitik-Gate.
- Crash-sicheres Apply-Intent/Recovery bleibt ein eigener Backend-Schritt.

## Verifikation

Backend-Fixtures pruefen UID-Historie, Pfad-/UID-Konflikt, Erhalt bei Move/Relink und Dateiloeschung sowie
fehlende Artefakte ohne Session-Start. Statische Browser-Fixtures pruefen einen einzelnen Host, Lazy-Warum,
Kontextwechsel waehrend einer Anfrage, dauerhaft rekonstruierte Jev-Karten mit Regel-Label,
Collapse/Reload, UID-Isolation, Read-only ohne Dateien und 1440/390/320 Pixel. Store-Fixtures pruefen
v6-Upgrade, unveraenderliche/idempotente Events, Kontextkonflikte, Reopen und Byte-Fenster ohne Loeschung.
Es werden keine produktiven Dienste oder Runs gestartet.

```bash
PLAYWRIGHT_MODULE_PATH=/pfad/zu/playwright/index.mjs node web_frontend_v3/tests/assistant-dock.browser.mjs > /tmp/out_dock_browser_test.txt 2>&1
```
