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
- Ein Run-Thread mit PI-Antwortkarten und explizit ausgeloester Jev-Post-Run-Beratung. AI-/Jev-Schalter werden
  respektiert. Vorschlaege im neuen Dock sind vorerst **nur lesbar**, ohne zweiten Apply-Sonderweg.
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
  Jev-Karten des neuen Docks werden derzeit nur in Memory gehalten und sind nach Reload nicht dort rekonstruiert.
- Gemeinsame Preview-/Apply-/Reason-Picker-Karten und Entfernen der alten Parameter-AI-/Jev-Huellen erst danach.
- Aktivierung permanenter Decision-Schreibpunkte und Vergessen bleiben hinter dem Retention-/Loeschpolitik-Gate.
- Crash-sicheres Apply-Intent/Recovery bleibt ein eigener Backend-Schritt.

## Verifikation

Backend-Fixtures pruefen UID-Historie, Pfad-/UID-Konflikt, Erhalt bei Move/Relink und Dateiloeschung sowie
fehlende Artefakte ohne Session-Start. Statische Browser-Fixtures pruefen einen einzelnen Host, Lazy-Warum,
Kontextwechsel waehrend einer Anfrage, Jev-Karte, Collapse/Reload, Read-only ohne Dateien und 1440/390/320 Pixel.
Es werden keine produktiven Dienste oder Runs gestartet.

```bash
PLAYWRIGHT_MODULE_PATH=/pfad/zu/playwright/index.mjs node web_frontend_v3/tests/assistant-dock.browser.mjs > /tmp/out_dock_browser_test.txt 2>&1
```
