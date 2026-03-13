# GUI2 Gap-Analyse vs. Legacy-GUI

## Ziel und Grundlage

Dieses Dokument hält die aktuelle Gap-Analyse zwischen der neuen webbasierten GUI2, der vorhandenen Backend-/Frontend-Implementierung und der Legacy-GUI in Qt/C++ fest.

Berücksichtigt wurden dabei:

- GUI2-Sollbild aus `doc/gui2/detailkonzept.md`
- Parameter-Coverage aus `doc/gui2/parameter_coverage_yaml_vs_schema.md`
- Ist-Stand in `web_backend_cpp/src/routes/*`
- Ist-Stand in `web_frontend/*.html` und `web_frontend/src/app.js`
- Legacy-Referenz in `legacy/qt6_gui/gui_cpp/*`

## Kurzfazit

Die Grundarchitektur ist vorhanden und tragfähig:

- Backend-APIs für `scan`, `runs`, `config`, `tools`, `jobs` und `ws` sind implementiert.
- Frontend-Seiten für Dashboard, Input & Scan, Parameter Studio, Run Monitor, History, Astrometry, PCC und Live Log existieren.
- Zentrale Frontend-Bindings sind in `web_frontend/src/app.js` vorhanden.

Die Hauptlücke liegt aktuell nicht mehr in fehlenden Grundbausteinen, sondern in der unvollständigen funktionalen Parität zwischen sichtbarer UI, echter Verdrahtung und Legacy-Verhalten.

## Status nach Bereich

### 1. Bereiche mit guter Abdeckung

- Run-Backend vorhanden
  - Start, Resume, Stop, Status, Logs, Artifacts, Stats, Set Current, Revisions-Restore
- Tool-Backend vorhanden
  - Astrometry: Detect, Install, Catalog Download/Cancel, Solve, Save Solved
  - PCC: Siril Status, Download/Cancel, Online Check, Run, Save Corrected
- Run Monitor grundsätzlich funktional
  - Status laden, WebSocket-Stream, Resume, Stats-Job, Report-/Pfadstatus
- Parameter Studio substanziell umgesetzt
  - Presets, Patch, Validate, YAML-Sync, Dirty Tracking, dynamische Feldzuordnung
- Live Log real verdrahtet
  - Initialer Log-Tail plus WebSocket-Streaming
- Zentrale Browse-/Dateiauswahl ist vorhanden.
  - zentrale Picker-Infrastruktur ist in `web_frontend/tooltips.js` vorhanden
  - nutzt `/api/fs/roots`, `/api/fs/list` und `/api/fs/grant-root`
  - ist auf den relevanten Screens bereits eingebunden

### 2. Teilweise umgesetzt

- History
  - Runs laden, Auswahl, Set Current, Report-Status anzeigen
  - aber kein echter Compare-Workflow und reduzierte Tiefenfunktion im Vergleich zur Legacy-GUI
- Dashboard / Guided Run
  - Run-Start, Scan-Refresh, Guardrails, Presets, Queue-Grundlogik vorhanden
  - einzelne KPI- und Tool-Claims sind noch stärker UI-getrieben als vollständig datengetrieben
- Astrometry-Frontend
  - Backend-Aufrufe vorhanden
  - aber Resultatfelder werden nicht vollständig mit Solve-Ergebnissen befüllt
- PCC-Frontend
  - Backend-Aufrufe vorhanden
  - aber UI-Parameter und Ergebnissynchronisierung sind unvollständig

### 3. Harte Gaps

#### P0 - Kritische Paritätsbrüche

##### 3.1 PCC-Screen nutzt seine sichtbaren Parameter nicht vollständig

Im Frontend wird beim PCC-Run aktuell im Wesentlichen nur Folgendes gesendet:

- `input_rgb`
- `output_rgb`
- `r`
- `g`
- `b`

Nicht oder nicht vollständig an das Backend durchgereicht werden u. a.:

- `wcs_file`
- `source`
- `mag_limit`
- `mag_bright_limit`
- `min_stars`
- `sigma_clip`
- `aperture_radius_px`
- `annulus_inner_px`
- `annulus_outer_px`

Damit ist die sichtbare PCC-UI aktuell nicht funktionsparitätstauglich.

##### 3.2 Astrometry-Ergebnisanzeige unvollständig

Der Solve-Flow ist verdrahtet, aber Ergebnisfelder wie z. B.:

- RA
- Dec
- FOV
- Rotation
- Pixel Scale

werden nicht konsistent aus dem Solve-Ergebnis in die UI gespiegelt.

#### P1 - Hohe Gaps

##### 3.3 PCC-Ergebnisse werden nicht sauber in die UI übertragen

Die PCC-Seite zeigt Ergebnisfelder an, aber im Frontend werden Job-/Logdaten im Wesentlichen nur angehängt, statt die dedizierten Ergebnisfelder strukturiert zu aktualisieren.

##### 3.4 History + Current Run noch nicht auf Legacy-Komfortniveau

Die Legacy-GUI hatte mit `HistoryTab` und `CurrentRunTab` einen detaillierteren Workflow.

In GUI2 ist die Funktionalität verteilt vorhanden, aber nicht vollständig gleichwertig:

- Pfade werden teils nur angezeigt statt echte Aktionen anzubieten
- Artefakt-/Report-Workflows sind reduziert
- Run-Vergleich fehlt

##### 3.5 App-State ist im Backend noch relativ dünn

`/api/app/state` ist vorhanden, liefert aber für mehrere Teilbereiche noch eher minimale Platzhalterstrukturen. Das erhöht die Komplexität im Frontend, weil viel Zustand aus Einzelendpunkten zusammengesetzt werden muss.

#### P2 - Mittlere Gaps

##### 3.6 Schema-/Coverage-Lücken

Laut `parameter_coverage_yaml_vs_schema.md` existieren weiterhin YAML-Parameter, die nicht sauber im Schema abgebildet sind. Das betrifft u. a. relevante Eingabe- und PCC-Felder.

Folge:

- GUI-seitig editierbar oder sichtbar
- aber nicht sauber durch das Schema abgesichert

##### 3.7 Dashboard- und History-Claims nur teilweise eingelöst

Einige im Sollbild vorgesehene Komfortfunktionen sind in der UI sichtbar oder implizit angekündigt, aber noch nicht vollständig durch echte End-to-End-Flows gedeckt.

## Vergleich zur Legacy-GUI

### Gute Parität

- Astrometry-Backend-Operationen sind grundsätzlich vorhanden.
- Zentrale Browse-/Dateiauswahl ist vorhanden.
- Run-Steuerung mit Start/Resume/Stop ist vorhanden.
- Live-Logging ist sogar moderner über WebSockets umgesetzt.
- History-Basisfunktionen sind vorhanden.

### Unvollständige Parität

- PCC-Komplettworkflow
- Astrometry-Result-UI
- Current-Run-Komfortfunktionen
- vollständige Parameternutzung zwischen UI und Backend

## Priorisierte Abarbeitungsreihenfolge

### 1. PCC-Frontend und Backend korrigieren

Alle sichtbaren PCC-Parameter an das Backend durchreichen und die Resultfelder sauber aus dem Job-/Run-Ergebnis befüllen. Da der bisherige Web-Flow nur `pcc-apply` nutzt, betrifft das Frontend, Backend und CLI zugleich.

### 2. Astrometry-Frontend vervollständigen

Status und Solve-Ergebnisse vollständig an UI-Felder anbinden.

### 3. History / Run Monitor auf Legacy-Niveau anheben

Artefakte, Reports, Vergleich und Run-Komfortfunktionen vervollständigen.

### 4. Schema- und Coverage-Lücken schließen

Noch fehlende oder inkonsistent modellierte Parameter in Schema und GUI vereinheitlichen.

## Risiken

- Sichtbare UI ohne echte Verdrahtung erzeugt Fehlvertrauen beim Nutzer.
- Teilweise verdrahtete Tool-Screens sind besonders riskant, weil dort Bedienung korrekt aussieht, aber fachlich nicht vollständig ausgeführt wird.

## Nächster konkreter Arbeitsschritt

Als erstes wird der PCC-Web-Flow auf einen echten End-to-End-Pfad mit WCS, Katalogquelle, Photometrieparametern und Result-Rückgabe umgestellt.
