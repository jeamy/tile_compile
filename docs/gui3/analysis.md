# GUI3 – Frontend-Analyse der bestehenden GUI2

> Stand: Juni 2026  
> Scope: `web_frontend/` – Struktur, Features, Probleme, Migration-Ziele

---

## 1. Aktueller Stand

### 1.1 Architektur-Überblick

| Aspekt | Ist-Zustand |
|---|---|
| Framework | Keines – Vanilla HTML/JS/CSS |
| Build-System | Keines – statische Dateien, direkt vom C++ Backend ausgeliefert |
| Seiten | 11 separate HTML-Dateien |
| JS-Monolith | `src/app.js` – **~10.300 Zeilen**, steuert alle Seiten |
| Styling | 3 CSS-Dateien (`style.css`, `layout-panels.css`, `theme.css`) – ~2.300 Zeilen |
| i18n | JSON-basiert (`de.json`, `en.json`), ~100 KB pro Sprache |
| API-Kommunikation | `ApiClient`-Klasse (fetch + WebSocket) |
| State-Management | `localStorage` + Server-side UI-State (`/api/app/ui-state`) |
| Fonts | Inria Sans / Inria Serif (lokal, OTF) |

### 1.2 Seiten-Inventar

| Seite | Datei | Zeilen HTML | Hauptfunktion |
|---|---|---|---|
| Dashboard | `index.html` | 252 | KPIs, Guided Run, Guardrails, Pipeline-Vorschau |
| Input & Scan | `input-scan.html` | 257 | Eingabeordner, Scan, Kalibrierung, KI-Analyse |
| Parameter Studio | `parameter-studio.html` | 482 | Alle Parameter mit Explain, Szenarien, KI-Empfehlungen, YAML-Diff |
| Assumptions | `assumptions.html` | 87 | Frame-Schwellen, Reduced-Mode-Regeln |
| Run Monitor | `run-monitor.html` | 142 | Phasen, Live Log, Stats, Resume, Artefakte |
| History + Tools | `history-tools.html` | 95 | Run-Historie, Run-Vergleich |
| Raw Stack | `raw-stack.html` | 325 | Preprocessing-Pipeline (Lights → Stack) |
| Astrometry | `astrometry.html` | 105 | ASTAP CLI, Katalog-Download, Plate-Solve |
| PCC | `pcc.html` | 109 | Photometric Color Calibration, Siril/VizieR |
| Live Log | `live-log.html` | 49 | Echtzeit-Log mit Filter |
| Wizard | `wizard.html` | 263 | Geführter Run-Flow (Step 1-3) |

### 1.3 Navigation

- **Pill-Nav** (Top-Bar): 11 Links + Help-Toggle + Locale-Switch + Status-Chips
- **Sidebar**: gleiche 11 Links (Redundanz)
- **Keine Tab-Struktur** – jede Seite ist ein eigener HTML-Seitenwechsel
- **Kein SPA-Router** – vollständige Seiten-Reloads bei Navigation

### 1.4 API-Endpunkte (genutzt durch Frontend)

| Bereich | Endpunkte |
|---|---|
| Scan | `/api/scan`, `/api/scan/latest`, `/api/scan/quality`, `/api/scan/metrics`, `/api/scan/analysis/*` |
| Config | `/api/config/schema`, `/api/config/defaults`, `/api/config/current`, `/api/config/patch`, `/api/config/presets`, `/api/config/validate`, `/api/config/save`, `/api/config/revisions` |
| Runs | `/api/runs` (list, start, status, config, artifacts, delete, stop, resume, stats, logs, set-current) |
| WebSocket | `/api/ws/runs/{runId}` – Live-Phase, Log-Streaming |
| Tools | `/api/tools/astrometry/*`, `/api/tools/pcc/*`, `/api/tools/preprocessing/*` |
| AI | `/api/ai/config`, `/api/ai/models`, `/api/ai/auth/*`, `/api/ai/test` |
| Guardrails | `/api/guardrails` |
| App-State | `/api/app/state`, `/api/app/constants`, `/api/app/ui-state` |
| Jobs | `/api/jobs` (polling für Download-Status etc.) |
| Filesystem | `/api/fs/grant-root`, `/api/fs/open` |

### 1.5 KI-Integration (aktuell)

KI ist an **zwei Stellen** integriert:

1. **Input & Scan** (`scan-ai-panel` in `input-scan.html`):
   - Modell-Auswahl, API-Key-Verwaltung (Provider: anthropic, openai, google, mistral, groq, openrouter)
   - Kontext: Mount-Type, Target-Size, Camera-Type, Kalibrierung, Notizen
   - Analyse erstellen / neu analysieren (Cache-Override)
   - Gespeicherte Analysen laden
   - Datenverkehr-Log (roher API-Verkehr)
   - Ergebnis: Empfehlungen für Parameter

2. **Parameter Studio** (`parameter-ai-panel` in `parameter-studio.html`):
   - Status-Anzeige
   - Gespeicherte Analysen laden
   - Analyse-Ergebnisse anzeigen (selektiv anwendbar)
   - "Ausgewählte anwenden" – übernimmt KI-Empfehlungen in Config

### 1.6 Logging (aktuell)

| Ort | Implementierung |
|---|---|
| Run Monitor | Live Log via WebSocket, monospace div, max 855px, auto-scroll |
| Live Log (Seite) | WebSocket, Filter (All/Info/Warning/Error), Export |
| Astrometry | Statisches monospace div, keine Echtzeit |
| PCC | Statisches monospace div, keine Echtzeit |
| KI-Datenverkehr | Monospace div, max 200px, roher JSON-Verkehr |

---

## 2. Probleme und Schwachstellen

### 2.1 Strukturelle Probleme

- **`app.js` Monolith**: 10.300 Zeilen in einer Datei, keine Module außer `api.js`, `constants.js`, `utils.js`, `i18n.js`
- **Seiten-Reload bei Navigation**: Kein SPA, jeder Klick lädt die Seite neu → State muss über `localStorage`/Server persistiert werden
- **Redundante Navigation**: Pill-Nav und Sidebar zeigen identische Links
- **Doppelte Logik**: Dashboard, Wizard und Input&Scan haben fast identische Input-Formulare (Copy-Paste-Code)
- **Keine Komponenten-Wiederverwendung**: Queue-Editor, Kalibrierungs-Block, Preset-Selector mehrfach dupliziert

### 2.2 UX-Probleme

- **Zu viele Seiten**: 11 Navigationseinträge → kognitive Überlastung
- **Dashboard vs. Wizard vs. Input&Scan**: Drei Eingänge für denselben Workflow, uneinheitlich
- **Assumptions als eigene Seite**: Nur 4 Felder – rechtfertigt keine eigene Seite
- **Live Log als eigene Seite**: Sollte in Run Monitor integriert sein (ist es teilweise schon)
- **Kein Progress-Feedback bei langen Operationen**: Download-Status nur via Polling
- **Keine Toast/Notification-System**: Fehler erscheinen als Text im Footer

### 2.3 Design-Probleme

- **1920px min-width**: Nicht responsive, bricht auf kleineren Screens
- **Inria Serif als Title-Font**: Schön, aber schwer lesbar auf kleinen Größen
- **Inline-Styles massenhaft**: Viele `style="..."` Attribute in HTML, schwer wartbar
- **Kein Dark Mode**: 3 Themes (observatory, slate, sand) aber alle hell
- **Kein Design-System**: Ad-hoc-Komponenten (`ps-btn`, `ps-chip`, etc.) ohne systematische Tokens

### 2.4 KI-Integration-Probleme

- **Zwei getrennte KI-Panels**: Input&Scan und Parameter Studio haben jeweils eigene KI-UI, unverbunden
- **Keine Nachverfolgung**: Angewendete KI-Empfehlungen nicht markiert
- **Datenverkehr-Log**: Rohes JSON, schwer lesbar, keine Filterung
- **Kein Streaming**: KI-Analyse wird als Batch abgewartet, kein Progress

---

## 3. Feature-Inventar für Migration

### 3.1 Input & Scan Features

- Eingabeordner (Browse, +, Queue)
- Dateimuster, Frames Min/Max, Sortierung
- Farbmodus (OSC/MONO), Bayer-Pattern
- Checksummen-Option
- Run-Queue Editor (Filter, Pattern, Run-Label, Aktiv)
- Kalibrierung (Bias/Dark/Flat – Ordner oder Master-Datei)
- Scan ausführen
- Scan-Ergebnis anzeigen (Status, Frames, Color Mode, Bayer, etc.)
- KI-Analyse (Modell, API-Key, Kontext, Analyse erstellen/laden)

### 3.2 Parameter Studio Features

- Kategorie-Liste (25 Kategorien)
- Preset-Selector + Preset-Dir
- Live-Suche über Parameter
- Parameter-Editor (dynamisch aus Schema)
- Explain-Panel (Label, Path, Category, Type, Default, Range, Phase, Description)
- Situation Assistant (Alt/Az, Rotation, Bright Stars, Few Frames, Gradient)
- Szenario-Deltas anwenden
- KI-Empfehlungen anzeigen + anwenden
- YAML-Diff-Preview
- Validierung (Schema + Semantik)
- Speichern / Speichern unter
- Config-Revisionen

### 3.3 Assumptions Features

- Frames Minimum
- Reduced-Mode Schwelle
- Clustering überspringen
- Cluster-Bereich
- Pipeline-Modus-Anzeige (Full/Reduced/Emergency)

### 3.4 Run Monitor Features

- Run starten / stoppen
- Phasen-Liste mit Status und Progress
- Live Log (WebSocket)
- Stats generieren / öffnen
- Report öffnen
- AQMH Cherry-Pick Panel
- Resume + Config-Revision
- Template laden/speichern
- Resume Config Editor (YAML textarea)
- Artefakt-Liste + Viewer
- Run-Ordner öffnen
- Batch-Summary + Verzeichnisstruktur

### 3.5 History Features

- Run-Historie (Liste mit Status, Datum, Name)
- Run auswählen / als Current setzen
- Run-Details (ID, Status, Phase, Progress, Artefakte, Report)
- Run-Vergleich (zwei Runs side-by-side)
- Stats generieren, Report öffnen, Run löschen

### 3.6 Tools Features

**Astrometry**:
- ASTAP CLI Setup (Binary, Data Dir, Detect, Install)
- Star Database (Catalog, Download, Cancel)
- Plate Solve (FITS File, Solve, Save-Solved, WCS-Results)

**PCC**:
- Input (RGB FITS, WCS File)
- Catalog Source (Siril/VizieR, Download Missing, Cancel, Check Online)
- PCC Parameters (mag_limit, sigma, aperture, annulus, k_max, etc.)
- Run PCC, Save Corrected
- Result (Stars matched/used, Residual RMS, Color Matrix)

**Raw Stack**:
- Vollständige Preprocessing-Pipeline
- Input, Calibration, Quality, Stack, Astrometry, BGE, PCC, HyperMetric Stretch
- Eigener Job-Status

---

## 4. Ziel-Struktur für GUI3

### 4.1 Tab-Struktur (vom User vorgegeben)

**Tab 1: Processing**
- Input & Scan
- Parameter Studio + Assumptions (zusammengelegt)
- Run Monitor

**Tab 2: Tools**
- Raw Stack
- Astrometry
- PCC

**Tab 3: History**
- Run-Historie (mit Detail-View, Vergleich, Stats, Report)

**Entfällt**: Dashboard, Wizard, Live Log (als eigene Seite), Assumptions (als eigene Seite)

### 4.2 KI-Integration

- Im Parameter Studio als **zweiter Tab** "AI Empfehlung"
- Verknüpfung mit Scan-KI: Scan-Analyse → Parameter-Studio AI-Tab
- Einheitliches KI-Panel (nicht mehr zwei getrennte)

### 4.3 Logging

- Im Run Monitor **integriert** (nicht eigene Seite)
- Toast-Benachrichtigungen für asynchrone Events
- KI-Datenverkehr als aufklappbares Detail

---

## 5. Technologie-Empfehlungen

| Aspekt | Empfehlung | Begründung |
|---|---|---|
| Framework | **Keines** – HTML5 + CSS3 + Vanilla JS (ES Modules) | Keine Abhängigkeiten, kein Build-Schritt, direkt vom Backend auslieferbar |
| Styling | **CSS Custom Properties** + strukturierte CSS-Dateien | Light/Dark Theme via `data-theme` Attribut, keine Präprozessoren |
| State | **Minimaler Pub/Sub Store** (Vanilla JS, ~30 Zeilen) | `getState()` / `setState()` / `subscribe()` – ausreichend für UI-State |
| Routing | **Hash-basiert** (`#processing`, `#tools`, `#history`) | SPA ohne Seiten-Reloads, kein Router-Package nötig |
| Build | **Keiner** | Dateien werden direkt vom Browser geladen (ES Modules nativ) |
| i18n | **Bestehende JSON-Dateien** + Vanilla JS i18n-Modul | Keine Änderung an Übersetzungs-Dateien nötig |
| Charts/Stats | **Custom SVG/Canvas** oder leichtgewichtiges Vanilla JS | Für Stats-Visualisierung, keine externe Library |
| Komponenten | **Functions die DOM-Elemente zurückgeben** | `createPathInput()`, `createTabBar()`, etc. – einfach, keine Vererbung |

### Architektur-Prinzipien

1. **SPA ohne Framework**: `index.html` als einzige HTML-Datei, JS rendert in `<div id="app-root">`
2. **ES Modules**: `import`/`export` nativ im Browser, keine Bundler nötig
3. **Komponenten-Factory-Pattern**: Jede Komponente = Funktion die `HTMLElement` zurückgibt
4. **CSS Custom Properties**: Alle Design-Tokens als CSS-Variablen, Dark Mode via `html[data-theme="dark"]`
5. **Kein npm/pnpm**: Keine `package.json`, keine `node_modules`, keine Build-Pipeline

---

## 6. Datei-Größen-Überblick

| Datei | Größe | Anmerkung |
|---|---|---|
| `src/app.js` | 424 KB (~10.300 Zeilen) | **Hauptproblem** – muss modularisiert werden |
| `parameter-studio-page.js` | 55 KB (1.307 Zeilen) | Parameter-Studio-Logik |
| `param_editor_index.js` | 37 KB (1.637 Zeilen) | Parameter-Metadaten-Index |
| `i18n/de.json` | 104 KB | Übersetzungen |
| `i18n/en.json` | 94 KB | Übersetzungen |
| `layout-panels.css` | 18 KB (1.095 Zeilen) | Panel-Styles |
| `style.css` | 7.6 KB (444 Zeilen) | Basis-Styles |
| `tooltips.js` | 22 KB | Tooltip-System |
| `shell.js` | 23 KB (392 Zeilen) | Navigation-Shell |
| `src/i18n.js` | 21 KB | i18n-Logik |
| `src/constants.js` | 5 KB | API-Endpunkte |
| `src/api.js` | 2.2 KB | API-Client |
| `src/utils.js` | 4 KB | Utilities |

---

## 7. Backend-Kompatibilität

Das C++ Backend (`web_backend_cpp`) liefert statische Dateien unter `/ui/` aus.  
Für GUI3 muss sichergestellt werden:

- Dateien werden direkt ausgeliefert (kein Build-Output, kein `dist/`)
- API-Endpunkte bleiben unverändert (`/api/*`)
- WebSocket-Pfade bleiben unverändert (`/api/ws/*`)
- Keine Backend-Änderungen erforderlich
- MIME-Type `application/javascript` für `.js` sicherstellen (für ES Module-Imports)

---

## 8. Risiko-Bewertung

| Risiko | Severity | Mitigation |
|---|---|---|
| app.js-Migration sehr aufwändig | Hoch | Phasenweise Migration, Feature-Parity-Checkliste |
| KI-Integration komplex | Mittel | Einheitliches KI-Hook-System, frühes Prototyping |
| Backend-Änderungen nötig | Niedrig | API bleibt stabil, nur Frontend-Neubau |
| i18n-Migration | Niedrig | JSON kann direkt übernommen werden, i18n-Modul aus utils.js migriert |
| Performance bei großen Logs | Mittel | Virtualisierte Log-Liste (Canvas-basiert oder IntersectionObserver) |
