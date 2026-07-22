# HMS Preview GUI – Implementierungsplan

## Ziel und Umfang

Für die Resume-Phase `HYPERMETRIC_STRETCH` (HMS) wird eine interaktive,
webbasierte Konfigurations- und Vorschauansicht ergänzt. Sie verwendet ein im
Run vorhandenes lineares RGB-Artefakt, berechnet den HyperMetric Stretch auf
einem verkleinerten Proxy und zeigt das Ergebnis mit Histogramm und
Clipping-Diagnostik an.

Der erste Ausbau unterstützt RGB-Runs. Mono-Unterstützung und eine Verwendung
außerhalb des Resume-Workflows sind nicht Bestandteil des MVP.

## Verbindlicher Einstieg im Resume-Bereich

Im Run Monitor bleibt die bisherige Auswahl einer Resume-Phase erhalten.

Wenn `HYPERMETRIC_STRETCH` ausgewählt ist, zeigt der rechte Resume-Bereich in
einer gemeinsamen Aktionszeile:

```text
[ HMS konfigurieren ]  [ HyperMetric Stretch ]
                         ^ vorhandenes Phasen-Badge
```

- Der Button steht unmittelbar links neben dem Badge.
- Der Button wird nur bei ausgewählter Phase `HYPERMETRIC_STRETCH` angezeigt.
- Solange kein geeignetes RGB-Eingangsartefakt vorhanden ist, ist der Button
  deaktiviert und ein Tooltip erklärt den Grund.
- Der Klick öffnet die HMS-Konfiguration als Modal innerhalb des Run Monitors.
  Damit bleiben Run-Kontext, Resume-Konfiguration und Monitoring erhalten.
- Der Phasen-Badge selbst startet weiterhin keine Preview.
- Deutsche Beschriftung: `HMS konfigurieren`.
- Englische Beschriftung: `Configure HMS`.

## Benutzerablauf

1. Benutzer wählt im Resume-Bereich `HYPERMETRIC_STRETCH`.
2. Der Button `HMS konfigurieren` erscheint links neben dem Phasen-Badge.
3. Ein Klick öffnet das Modal und lädt die HMS-Werte aus der aktuell im
   Resume-Editor geladenen YAML-Konfiguration.
4. Das Backend löst das Eingangsartefakt auf und erzeugt bzw. lädt einen
   gecachten Proxy.
5. Änderungen an Preview-relevanten Parametern starten nach 500 ms Debounce
   eine neue Vorschau.
6. `Übernehmen & Resume starten` merged die Werte in die aktuell geladene YAML
   und verwendet den bestehenden Resume-Endpoint mit
   `from_phase: "HYPERMETRIC_STRETCH"`.
7. Das Modal schließt; der Run Monitor zeigt den gestarteten Resume-Job.

`Abbrechen` verwirft alle Änderungen. `Zurücksetzen` stellt die Werte wieder
her, mit denen das Modal geöffnet wurde; es verwendet nicht die Core-Defaults.

## Architektur

```text
Run Monitor / Resume
  └─ HMS konfigurieren
       └─ HMS Preview Modal
            ├─ POST /api/runs/<run_id>/hme-preview → PNG
            └─ POST /api/runs/<run_id>/resume
                 from_phase = HYPERMETRIC_STRETCH
                 config_yaml = aktualisierte Resume-YAML

HMS Preview Service
  ├─ sichere Run-Auflösung
  ├─ RGB-Artefaktauflösung
  ├─ threadsicherer Proxy-Cache
  ├─ image::run_hypermetric_stretch_rgb()
  └─ PNG-Encoding + Diagnostics-Header
```

Es wird kein eigener HMS-Apply-Endpoint eingeführt. Der bestehende Endpoint
`POST /api/runs/<run_id>/resume` übernimmt bereits Config-Snapshot,
Config-Revision, Schreiben der `config.yaml`, Event-Erzeugung und Jobstart.
Diese Logik darf nicht dupliziert werden.

## API-Vertrag

### `POST /api/runs/<run_id>/hme-preview`

Der Preview-Endpoint liegt unterhalb eines Runs und verwendet denselben sicheren
Run-Resolver wie die übrigen Run-Endpunkte. Ein beliebiger Serverpfad aus dem
Request wird nicht direkt geöffnet.

Request:

```json
{
  "run_dir": "/optional/alternate/runs/path/run-id",
  "params": {
    "mode": "ready_to_use",
    "sensor_profile": "rec709",
    "fallback_profile": "rec709",
    "adaptive_anchor": true,
    "target_bg": 0.15,
    "protect_b": 6.0,
    "convergence_power": 3.5,
    "log_d_mode": "auto",
    "fixed_log_d": 2.0,
    "color_strategy": "fixed",
    "fixed_color_strategy": 0.0,
    "color_grip": 1.0,
    "shadow_convergence": 0.0,
    "linear_expansion": 0.0
  }
}
```

Response:

- Body: `image/png`
- `X-HMS-Diagnostics`: kompakte JSON-Diagnostik mit `success`, `status`,
  `profile`, `profile_source`, `anchor`, `log_d`, `star_pressure`,
  `black_clip_percent`, `white_clip_percent` und gegebenenfalls
  `error_message`
- `Cache-Control: no-store`

Es gibt im MVP weder Base64-PNG noch Multipart-Response, separaten
Diagnostics-Endpoint oder separaten `solve-log-d`-Endpoint. Der Preview-Aufruf
im Auto-Modus liefert den berechneten Wert bereits als Diagnose.

Backendfehler verwenden das bestehende JSON-Fehlerformat und passende
HTTP-Statuscodes:

- `400`: ungültige Parameter
- `404`: Run oder geeignetes Eingangsartefakt fehlt
- `409`: Run-Zustand erlaubt die Operation nicht
- `422`: FITS-Datei oder Kanalgeometrie ist ungültig
- `500`: Stretch oder PNG-Encoding fehlgeschlagen

### Resume und Apply

Das Frontend aktualisiert ausschließlich die bekannten Felder unter
`hypermetric_stretch` in der aktuell geladenen Resume-YAML. Alle übrigen
Config-Felder und unbekannten YAML-Felder bleiben erhalten.

Anschließend wird aufgerufen:

```json
{
  "from_phase": "HYPERMETRIC_STRETCH",
  "run_dir": "/resolved/run/dir",
  "config_yaml": "<vollständige aktualisierte YAML>",
  "filter_context": "<bestehender Kontext>"
}
```

Ziel ist der bestehende Endpoint `POST /api/runs/<run_id>/resume`.

## Eingangsartefakt-Vertrag

Das Backend verwendet dieselbe zentrale Artefaktauflösung wie der Runner. Die
Logik wird in eine wiederverwendbare Servicefunktion ausgelagert, damit Resume
und Preview keine unterschiedlichen Prioritäten entwickeln.

Priorität:

1. PCC-RGB-Cube, wenn vollständig und lesbar
2. vollständiger Satz aus PCC-R/G/B-Einzelkanälen
3. BGE-linear RGB
4. Solve/linear RGB

Die tatsächlich verwendete Quelle wird in der Diagnose zurückgegeben und im
Modal angezeigt. Unvollständige Kanalsets werden nicht stillschweigend mit
anderen Quellen gemischt. Breite und Höhe aller Kanäle müssen identisch sein.

Die vorhandene Canvas-/Valid-Maske wird geladen, sofern sie zur Bildgeometrie
passt. Eine vorhandene, aber inkompatible Maske ist ein Fehler; ohne Maske darf
nur fortgefahren werden, wenn der reguläre HMS-Pfad dies für dieselbe Quelle
ebenfalls zulässt.

## Proxy-Erzeugung

- Maximale lange Kante: 1600 Pixel; als benannte Konstante konfigurierbar.
- RGB wird mit flächenbasiertem Downsampling (`INTER_AREA`) verkleinert.
- Die Maske wird auf exakt dieselbe Zielgeometrie mit Nearest Neighbor
  verkleinert.
- Die FITS-Orientierung wird entsprechend der tatsächlichen Semantik von
  `read_fits_rgb()` behandelt. Es wird kein ungeprüfter pauschaler Y-Flip
  eingebaut.
- Nicht-finite Werte und die erwartete Eingabeskalierung werden vor dem
  Stretch validiert.
- Die PNG-Erzeugung verwendet eine dokumentierte, reproduzierbare Abbildung
  von Stretch-Ausgabe nach 8 Bit sowie RGB→BGR für OpenCV.

## Proxy-Cache und Nebenläufigkeit

Der Cache wird als eigener Service implementiert, nicht als ungeschützte
`static map` in einer Route.

Cache-Key:

- kanonischer Run-Pfad,
- aufgelöste Quelldateien,
- Dateigröße und Änderungszeit jeder Quelldatei,
- Maskendatei inklusive Größe und Änderungszeit,
- Proxy-Maximalgröße.

Eigenschaften:

- immutable Cache-Einträge mit originalem Proxy-RGB und Proxy-Maske,
- Mutex oder gleichwertige Synchronisierung,
- begrenzte Anzahl bzw. begrenztes Speicherbudget mit LRU-Verdrängung,
- automatische Invalidierung bei geänderten Artefakten,
- kein Festhalten fehlgeschlagener Ladevorgänge als dauerhafter Eintrag.

`run_hypermetric_stretch_rgb()` verändert seine RGB-Argumente. Deshalb erstellt
jeder Preview-Request Arbeitskopien aus dem unveränderten Cache-Eintrag. Ein
Preview darf niemals auf dem Ergebnis des vorherigen Previews aufbauen.

## HMS-Parameter und UI

Preview-relevante Parameter:

| Config-Feld | UI | Wertebereich |
|---|---|---|
| `mode` | Select | `ready_to_use`, `scientific` |
| `sensor_profile` | Select | vom Backend gelieferte Profile |
| `fallback_profile` | Select | vom Backend gelieferte Profile |
| `adaptive_anchor` | Checkbox | bool |
| `target_bg` | Slider + Zahl | 0.05–0.50 |
| `protect_b` | Slider + Zahl | 0.1–15.0 |
| `convergence_power` | Slider + Zahl | 1.0–10.0 |
| `log_d_mode` | Select | `auto`, `fixed` |
| `fixed_log_d` | Slider + Zahl | 0.0–7.0 |
| `color_strategy` | Select | `auto`, `fixed` |
| `fixed_color_strategy` | Slider + Zahl | -1.0–1.0 |
| `color_grip` | Slider + Zahl | 0.0–1.0 |
| `shadow_convergence` | Slider + Zahl | 0.0–3.0 |
| `linear_expansion` | Slider + Zahl | 0.0–1.0 |

`fixed_log_d` ist nur bei `log_d_mode=fixed` editierbar.
`fixed_color_strategy` ist nur bei `color_strategy=fixed` editierbar. Die
Modusabhängigkeit weiterer Controls folgt exakt der Core-Semantik und wird in
einem Contract-Test abgesichert; sie wird nicht nur aus VeraLux kopiert.

Nicht als Preview-Control angeboten werden `enabled`,
`require_successful_pcc`, `write_channels` und `output_rgb`. Diese Werte bleiben
beim YAML-Merge unverändert und werden weiterhin im normalen Config-Editor
verwaltet.

Die Sensorprofile werden nicht als zweite fachliche Liste im Frontend gepflegt.
Das Backend stellt Profil-IDs, Labels und Gewichte über HMS-Metadaten bereit
oder die vorhandene allgemeine Parameter-Metadaten-Route wird erweitert.

## Preview-Modal

Das Modal enthält:

- Bild-Canvas mit Fit-to-screen, Zoom und Pan,
- Pixeltracker in Proxy-Koordinaten und Anzeige des RGB-Werts,
- Controls mit gekoppelten Slider- und Zahlenfeldern,
- RGB-Histogramm mit linearer/logarithmischer Darstellung,
- Black-/White-Clipping-Anzeige,
- verwendetes Eingangsartefakt und berechnete Diagnostik,
- `Übernehmen & Resume starten`, `Zurücksetzen`, `Abbrechen`.

Das Histogramm wird clientseitig aus dem unveränderten Preview-PNG berechnet.
Die vom Core gelieferten Clipping-Werte bleiben die maßgebliche numerische
Diagnostik; Abweichungen durch 8-Bit-Quantisierung werden nicht als Core-Werte
ausgegeben.

Responsive Verhalten:

- Desktop: Canvas links, Controls rechts.
- Schmale Ansicht: Canvas über Controls.
- Modal bleibt per Tastatur bedienbar, hält den Fokus und schließt mit Escape,
  solange kein Apply läuft.

## Request-Steuerung im Frontend

- Parameteränderungen werden um 150 ms entprellt.
- Vor einem neuen Preview wird der vorherige Fetch per `AbortController`
  abgebrochen.
- Zusätzlich trägt jeder Request eine lokale Generation-ID. Nur die Antwort
  der neuesten Generation darf Canvas, Histogramm oder Diagnose aktualisieren.
- Während der ersten Vorschau wird ein Ladezustand angezeigt.
- Ein späterer Fehler lässt die letzte gültige Vorschau sichtbar und zeigt die
  Fehlermeldung daneben.
- Während `Übernehmen & Resume starten` läuft, sind Controls und Schließen
  gesperrt, um Doppelstarts zu verhindern.

## Dateien und Verantwortlichkeiten

Backend:

- `web_backend_cpp/src/routes/runs_routes.cpp`: Run-bezogene Preview-Route
- neuer HMS-Preview-Service: Artefaktauflösung, Proxy-Cache, Stretch, Encoding
- bestehender Resume-Service/Endpoint: unverändert für Apply und Jobstart
- Core: bestehendes `image::run_hypermetric_stretch_rgb()`

Frontend:

- `web_frontend_v3/js/pages/run-monitor.js`: Button links neben Phasen-Badge,
  Modal-Lebenszyklus und Resume-Aufruf
- `web_frontend_v3/js/components/hms-preview.js`: Canvas, Controls und
  Request-Steuerung
- `web_frontend_v3/js/components/histogram.js`: wiederverwendbares Histogramm
- bestehender API-Client: Preview-Methode
- `web_frontend_v3/i18n/de.json` und `en.json`: Labels, Tooltips und Fehlertexte

Da das Modal Teil des Run Monitors ist, wird keine neue globale Page und keine
neue Hash-Route benötigt.

## Fehlerfälle

- Run oder RGB-Artefakt fehlt
- PCC-Kanäle sind unvollständig oder unterschiedlich groß
- FITS ist nicht lesbar oder enthält keine verwertbaren Werte
- Maske ist geometrisch inkompatibel
- Parameter liegen außerhalb des Schemas
- Preview-Request wurde durch eine neuere Änderung abgebrochen
- Cache-Eintrag wurde durch geänderte Artefakte invalidiert
- Stretch oder PNG-Encoding schlägt fehl
- Resume läuft bereits oder Run-Zustand ist nicht zulässig
- YAML-Merge oder Resume-Start schlägt fehl

Alle Fälle erhalten eine konkrete, lokalisierte Meldung. Fehler dürfen weder
stillschweigend auf eine andere Parametrisierung wechseln noch einen Resume
starten.

## Tests und Abnahmekriterien

### Backend

- Parameter-Validierung einschließlich unbekannter und fehlender Felder
- RGB-Cube sowie vollständige R/G/B-Einzelkanäle
- definierte Fallback-Reihenfolge und kein Mischen von Kanalsets
- fehlende, beschädigte und geometrisch inkonsistente FITS-Dateien
- Masken-Downsampling und Maskengeometrie
- Cache-Hit, Invalidierung, LRU und parallele Requests
- zwei identische Requests liefern identische Ergebnisse
- aufeinanderfolgende Requests stretchen stets den originalen Proxy
- Preview-Diagnostics entsprechen einem direkten Core-Aufruf
- sichere Run-Auflösung verhindert Pfadzugriff außerhalb zulässiger Runs

### Frontend

- Button erscheint ausschließlich bei gewähltem `HYPERMETRIC_STRETCH` und
  unmittelbar links neben dem Phasen-Badge
- fehlendes Artefakt deaktiviert den Button mit erklärendem Tooltip
- Controls bilden Auto/Fixed und die Modi korrekt ab
- veraltete oder abgebrochene Antworten überschreiben keine neuere Vorschau
- Reset stellt die beim Öffnen geladenen Werte wieder her
- Histogramm, Zoom, Pan und Tastaturbedienung funktionieren
- Apply verhindert Doppelstart und behandelt Backendfehler

### Integration

- Preview verändert weder FITS-Artefakte noch `config.yaml`
- Apply verändert ausschließlich bekannte `hypermetric_stretch`-Felder
- übrige und unbekannte YAML-Felder bleiben erhalten
- Apply erzeugt über den bestehenden Resume-Endpoint Snapshot und Revision
- Resume startet exakt ab `HYPERMETRIC_STRETCH`
- der gestartete Job erscheint unmittelbar im Run Monitor

## Implementierungsreihenfolge

1. Gemeinsame Artefaktauflösung und Backend-Contract-Tests
2. Proxy-Erzeugung, PNG-Encoding und direkte Core-Tests
3. Thread-sicherer, begrenzter Proxy-Cache
4. Run-bezogener Preview-Endpoint und API-Tests
5. Button links neben dem HMS-Phasen-Badge und Modal-Grundgerüst
6. Controls, Preview-Abbruch und Generation-Steuerung
7. Canvas, Histogramm, Diagnostik und responsive Darstellung
8. YAML-Merge und Aufruf des bestehenden Resume-Endpoints
9. i18n, Accessibility, Integrationstests und Fehlerfälle

## Aufwand

Für eine robuste Implementierung einschließlich Cache, automatisierter Tests,
Fehlerbehandlung und responsivem Modal sind etwa 10–14 Entwicklungstage
realistisch. Ein nicht nebenläufigkeitsfester Prototyp wäre schneller, erfüllt
aber die Abnahmekriterien dieses Plans nicht.
