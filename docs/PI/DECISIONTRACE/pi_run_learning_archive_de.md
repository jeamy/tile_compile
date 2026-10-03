# Dauerhafte Run-Lerndaten in SQLite

## Verbindliche Trennung

Nutzerentscheidung: Run-Dateien koennen geloescht werden, ohne valide Lerndaten zu verlieren.
Dateiloeschung ist **keine Ablehnung** einer Config, kein schlechtes Outcome und kein Ausschluss vom Lernen.
FITS-Rohdaten bleiben auf der Platte. Sie werden weder kopiert noch als Datenbank-BLOB archiviert.
Optional gibt es ein kleines PNG des Ergebnisses als visuelle Hilfe.

## Implementierter Speicher

`PiRunLearningStore` verwendet die gemeinsame Datei `pi_store_v2.sqlite`, Schema v6:

- `run_learning_snapshots`: unveraenderliche, inhaltsgehashte Snapshot-Versionen je `run_uid`.
- `run_learning_state`: aktuelle Snapshot-Referenz, Datei-Lifecycle und expliziter Lern-Ausschluss.
- `run_learning_previews`: optional ein PNG je Run, mit Bezug auf den konkreten Snapshot und das Ergebnis-Artefakt.

Neue Daten der bisherigen SQLite-Schemata v2 bis v5 bleiben bei Erweiterungen erhalten.
Der verworfene Altbestand (JSONL und alte `pi_store_v1.sqlite`) wird weiterhin nicht importiert.
Identische Erfassung desselben Inhalts und derselben Phase erzeugt keinen weiteren Snapshot.
Andere Configs, Messwerte oder Erfassungsphasen bleiben als eigene Versionen erhalten.

## Gesicherte Inhalte

1. **Identitaet:** UID, Run-ID, normalisierter Pfad, Start-/Endinformationen soweit vorhanden, Revisions- und
   Jev-Bezuege aus PI-Provenance. Software-/Build- und Pipeline-Vertragsversionen aus Runner-Provenance.
2. **Config:** exakter Inhalt von `config.yaml`, SHA256 und geparste Werte. Neue Runner schreiben zusaetzlich
   `artifacts/effective_config.json` mit `Config::to_yaml()` einschliesslich Parser-Defaults und dem Hash der
   zugrunde liegenden Config. Der Snapshot verwendet diese expandierte Config nur bei passendem Hash.
   Nach einem Resume darf ein alter expandierter Snapshot nicht die neue Config ueberdecken.
   Ohne passenden Runner-Snapshot: `effective_source=saved_yaml`, `parser_defaults_included=false`.
3. **Lights:** urspruengliches und effektives Input-Verzeichnis, vorhandenes exaktes `input_manifest`, Dateipfade,
   Groessen, vorhandene SHA256 und Aufnahme-Metadaten. Neue Provenance-Eintraege enthalten verfuegbare
   Headerwerte fuer Belichtung, Temperatur, Kamera, Filter, Gain, Binning, Aufnahmezeit, Bayer-Muster usw.
4. **Kalibration:** tatsaechlich verwendete Bias-/Dark-/Flat-Dateien bzw. Master, Herkunft und Auswahlbefunde aus
   dem `SCAN_INPUT`-Ereignis. Neue Runner protokollieren deren Input-Manifeste und Header-Metadaten.
   Kalibrations-Fingerprints sind explizit `path_size_mtime`, keine neu berechneten Pixel-Hashes.
   Ein extern erstellter Master verrät nicht automatisch seine urspruenglichen Einzelaufnahmen; diese werden nicht erfunden.
5. **Statistik/Validierung:** vorhandene Qualitaets-, Registrierungs-, Normalisierungs-, Rekonstruktions-, BGE-, PCC-,
   Sampling-, Report- und Preprocess-Artefakte; Frame-Quality-CSV und Ablehnungslisten. Ausserdem relevante
   Phasen-, Warnungs-, Fehler- und Abschlussereignisse, nicht beliebige Chat-Dateien oder Pixel-Caches.

Die Erfassung nutzt eine feste Dateiliste, folgt keinen Artefakt-Symlinks aus dem Run heraus und kopiert keine
beliebigen Dateien rekursiv. Fehlende optionale Daten werden gelistet, Lesefehler/ungueltige JSON/zu grosse Dateien
stehen in `capture_issues`. Metadaten sind niemals still als vollstaendig oder wissenschaftlich valide deklariert.

## Zeitpunkt der Erfassung

- Run-Start, vor dem Start des Runner-Prozesses: Config und Aufnahmequelle.
- Abschluss: erfolgreiche, fehlgeschlagene und abgebrochene Runner-Prozesse.
- Resume-Start und Resume-Abschluss: neue Config-/Ergebnisversionen unter derselben UID.
- Status-Poll als Fallback fuer noch nicht archivierte abgeschlossene Runs.
- Nach expliziter Report-/Statistik-Generierung: aktualisierte Statistiken.
- Unmittelbar vor Dateiloeschung: erneute Erfassung aller noch verfuegbaren Daten.

Die neue Archiv-Erfassung schreibt keine Marker in historische Run-Verzeichnisse. Die bereits existierenden
Memory-/Jev-Outcome-Recorder bleiben getrennt; das Archiv erzeugt keine automatische Memory-Promotion.
Es gibt keine pauschale Erfassung aller historischen Verzeichnisse beim Backend-Start.

## Loeschroute

`POST /api/runs/<id>/delete` bleibt kompatibel, optional mit `run_dir` fuer einen exakten Pfad.

1. Laufende Jobs werden auch bei Zugriff ueber einen Pfad-Alias geschuetzt.
2. Snapshot sichern. Scheitert SQLite, HTTP 503 und **keine Dateiloeschung**.
3. Bekannte originale Lights/Kalibrationsquellen innerhalb des Run-Verzeichnisses: HTTP 409,
   keine Loeschung. Originale zuerst separat sichern/verschieben.
4. Lesefehler oder Erfassungsgrenzen: HTTP 409, bis bewusst `allow_incomplete_snapshot=true` uebergeben wird.
   Fehlende optionale Artefakte allein sind kein Lesefehler; entsprechende Luecken bleiben im Snapshot sichtbar.
5. `deletion_pending` setzen, Run-Dateien entfernen, danach `deleted` setzen. Bei Dateifehlern `delete_failed`.
6. Snapshot-Versionen, Config, Statistik, Herkunft, PNG und Lern-Ausschlusszustand bleiben erhalten.
   Die Antwort enthaelt `learning_retained=true`, UID, Snapshot-ID und Erfassungsprobleme.

Ein Absturz zwischen Dateiloeschung und zentraler Abschlussmarkierung kann `deletion_pending` hinterlassen.
Die Snapshot-Daten sind dann bereits gesichert; automatische Recovery dieser Markierung ist noch offen.
Extern geloeschte oder offline verschobene Runs bleiben archiviert. Beim Einzelabruf wird die Erreichbarkeit
bekannter Pfad-Aliase getrennt vom gespeicherten Lifecycle gemeldet (`missing_or_unreachable` ist nicht
automatisch eine bestaetigte Loeschung). Ohne vorherige Erfassung lassen sich externe Loeschungen nicht rueckwirkend retten.

## Lernen und Datenschutz

- `validation_state=unreviewed`: Prozess-Erfolg oder gespeicherte Messwerte beweisen weder wissenschaftliche
  Validitaet noch die Ueberlegenheit einer Config-Aenderung.
- `comparison_kind=unpaired`, `quality_delta=null`: keine erfundene Verbesserung ohne geeigneten Vergleich.
- `excluded_from_learning` wird nur explizit geaendert; neue Snapshots und Dateiloeschung setzen es nicht zurueck.
- Ausschlusscodes: `test_run`, `invalid_data`, `unreliable_measurements`, `user_choice`.
- Archive bleiben bis zu einer ausdruecklichen Vergessen-/Reset-Aktion erhalten. Eine eigene Vergessen-Route
  ist noch nicht implementiert; kein automatisches Purging dieser neuen Run-Lerndaten.
- Das Archiv hat `privacy_class=local_run_archive_with_paths`. Configs und Herkunft duerfen hier lokale Pfade enthalten.
  Es ist **nicht** Bestandteil der bisherigen metadata-only Memory-Exports und wird nicht automatisch an Modelle gesendet.
- Freitext-/Session-Aufbewahrung und Aktivierung permanenter Decision-Schreibpunkte bleiben separate offene Gates.

## Optionales PNG

`TILE_COMPILE_PI_KEEP_RESULT_PREVIEW=1` aktiviert die Erzeugung beim Abschluss/Report-Refresh.
Default: aus. Verwendet wird das am weitesten verarbeitete bekannte Ergebnis (HMS, PCC, BGE, RGB bzw. Mono),
nicht ein Light/Dark. Die Vorschau wird auf 512 Pixel Kantenlaenge reduziert.
Store-Grenzen: PNG, maximal 1024 Pixel je Kante, maximal 2 MiB. Speicherung als Base64 in SQLite.
Die PNG-Referenz nennt Snapshot und Quell-Artefakt; es handelt sich um eine Darstellung, **nicht um Messdaten**.
Es wird nur das zuletzt gespeicherte Preview je UID gehalten, nicht eine vollstaendige Bildhistorie.

## HTTP-Endpunkte

- `GET /api/pi/run-learning?limit=100`: kompakte Archive-Liste, auch geloeschte Runs.
- `GET /api/pi/run-learning/<run_uid>`: letzter Snapshot und Lifecycle.
- `GET /api/pi/run-learning/<run_uid>/history?limit=50`: Snapshot-Versionen.
- `GET /api/pi/run-learning/<run_uid>/preview`: PNG oder 404.
- `POST /api/pi/run-learning/<run_uid>/exclusion`:
  `{"confirmed":true,"excluded":true,"reason_code":"test_run"}`.
  Wieder zulassen: `{"confirmed":true,"excluded":false}`; das ist keine wissenschaftliche Validierung.

Die visuelle Archiv-/Loeschoptionen-Oberflaeche im Run History/Dock ist noch offen.

## Grenzen und Tests

Erfassung: 2 MiB Config, 16 MiB je Dokument/Events-Datei, 64 MiB eingelesener Text insgesamt und 10000
 ausgewaehlte Phasenereignisse. Auch der serialisierte Snapshot ist auf 64 MiB begrenzt.
Dokument-Limits werden als Erfassungsprobleme diagnostiziert; ein zu grosser Gesamtsnapshot bricht die
Archivierung ab (Loeschroute: 503). Keine automatische Loeschung von Run-Dateien bei solchen Problemen.

Tests decken dauerhafte Statistik/Config/Herkunft, native Runner-Events, Parser-Defaults und Hash-/Resume-Schutz,
immutable/idempotente Snapshots, expliziten Ausschluss, SQLite-Upgrade, externe Symlinks, keine FITS-/Chat-Kopien,
PNG-Grenzen und Wiedereroeffnen ab. Run-Start-Fixtures pruefen Archivierung, Loeschung mit Erhalt der Rohquellen,
Speicherfehler vor Loeschung, bestaetigte unvollstaendige Archive und Schutz von Rohquellen innerhalb des
Run-Verzeichnisses (einschliesslich anschliessendem Verschieben vor Dateiloeschung).
