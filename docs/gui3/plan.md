# GUI3 – Modernisierungs-Plan

> Fortsetzung von `analysis.md` – konkreter Umsetzungsplan mit Architektur, Abläufen und Phasen

---

## 1. Architektur-Entscheidung

### Stack
- **HTML5** + **CSS3** (Custom Properties) + **Vanilla JavaScript** (ES Modules)
- **Kein Framework**, kein Build-System, keine Abhängigkeiten
- **CSS Custom Properties** für Design-Tokens (Light/Dark Theme)
- **ES Modules** (`import`/`export`) für Modulare Struktur – nativ im Browser
- **Custom Elements / Web Components** (optional, leichtgewichtig) für wiederverwendbare UI-Komponenten
- **Bestehende i18n** JSON-Dateien werden direkt übernommen
- **Kein npm/pnpm** – Dateien werden direkt vom C++ Backend unter `/ui/` ausgeliefert

### Projekt-Struktur

```
web_frontend_v3/
├── index.html                      # SPA-Shell: Header, Tab-Bar, Content-Root
├── css/
│   ├── tokens.css                  # Design-Tokens (Colors, Spacing, Typography)
│   ├── base.css                    # Reset, Body, Layout-Grundlagen
│   ├── components.css               # Buttons, Cards, Inputs, Badges, Tabs, Toast
│   ├── layout.css                  # Grid, Panels, Sidebar, Header
│   └── pages.css                   # Seitenspezifische Styles (Processing, History)
├── js/
│   ├── main.js                     # Entry point: Initialisierung, Tab-Routing
│   ├── api/
│   │   ├── client.js               # ApiClient (aus api.js migriert)
│   │   └── endpoints.js            # API_ENDPOINTS (aus constants.js)
│   ├── state/
│   │   ├── store.js                # Minimaler globaler State-Manager (Event-basiert)
│   │   ├── config-state.js         # Config-Draft, Validation, Dirty-State
│   │   ├── run-state.js            # Current Run, Phase, Log-Lines
│   │   ├── scan-state.js           # Scan-Results, KI-Analyse
│   │   ├── ui-state.js             # Locale, Theme, Server-UI-State
│   │   └── ai-state.js             # KI-Config, Modelle, Auth
│   ├── components/
│   │   ├── tab-bar.js              # Haupt-Tab-Navigation (Processing/Tools/History)
│   │   ├── sub-tabs.js             # Sub-Tab-Leiste (kontextabhängig)
│   │   ├── header.js               # Top-Bar mit Status-Chips, Locale
│   │   ├── toast.js                # Notification-System
│   │   ├── path-input.js           # Input + Browse-Button
│   │   ├── queue-editor.js         # Run-Queue (wiederverwendet)
│   │   ├── calibration-panel.js    # Bias/Dark/Flat
│   │   ├── preset-selector.js      # Preset + Dir
│   │   ├── guardrail-badges.js     # Guardrail-Status-Chips
│   │   ├── log-viewer.js           # Virtualisierte Log-Liste (Canvas-basiert)
│   │   ├── ai-panel.js             # Einheitliches KI-Panel
│   │   ├── pipeline-indicator.js   # SCAN→REG→...→DONE
│   │   ├── phase-list.js           # Phasen-Liste mit Progress
│   │   ├── explain-panel.js        # Parameter-Erklärungen
│   │   ├── situation-assistant.js  # Szenario-Auswahl + Apply
│   │   ├── yaml-diff.js            # YAML-Diff-Preview
│   │   ├── artifact-list.js        # Artefakt-Liste + Viewer
│   │   └── modal.js                # Modal-Dialog (Artefakt-Viewer etc.)
│   ├── pages/
│   │   ├── processing.js           # Tab 1: Container + Sub-Tab-Orchestrierung
│   │   ├── input-scan.js           # Sub-Tab: Input & Scan
│   │   ├── parameter.js            # Sub-Tab: Parameter + Assumptions + AI-Tab
│   │   ├── run-monitor.js          # Sub-Tab: Run Monitor
│   │   ├── tools.js                # Tab 2: Container + Sub-Tab-Orchestrierung
│   │   ├── raw-stack.js            # Sub-Tab: Raw Stack
│   │   ├── astrometry.js           # Sub-Tab: Astrometry
│   │   ├── pcc.js                  # Sub-Tab: PCC
│   │   ├── history.js              # Tab 3: Container + Sub-Tab-Orchestrierung
│   │   └── run-history.js          # Sub-Tab: Run-Historie
│   ├── i18n/
│   │   ├── i18n.js                 # i18n-Logik (aus utils.js migriert)
│   │   ├── de.json                 # aus web_frontend/i18n/de.json
│   │   └── en.json                 # aus web_frontend/i18n/en.json
│   └── utils/
│       ├── path.js                 # encodeRunIdPathSegment etc.
│       ├── log.js                  # Log-Formatting
│       ├── yaml.js                 # YAML-Diff-Utilities
│       └── dom.js                  # DOM-Helper ($, createElement, etc.)
└── assets/
    └── fonts/                      # Bestehende Inria Fonts (oder Inter)
```

### Architektur-Prinzipien

1. **SPA ohne Framework**: `index.html` ist die einzige HTML-Datei. Tab-Inhalte werden per JS in einen `<div id="app-root">` gerendert. Navigation über Hash-Routing (`#processing`, `#tools`, `#history`).
2. **Modularer State-Manager**: Minimaler Pub/Sub-Store (kein externes Package). Jedes State-Modul exportiert `getState()`, `setState()`, `subscribe()`.
3. **Komponenten als Functions**: Jede Komponente ist eine JS-Function die ein DOM-Element zurückgibt (`createPathInput()`, `createTabBar()`, etc.). Keine Klasse, keine Vererbung – einfach Funktionen.
4. **CSS Custom Properties**: Alle Design-Tokens als CSS-Variablen in `tokens.css`. Dark Mode via `html[data-theme="dark"]`.
5. **Kein Build-Schritt**: Dateien werden direkt vom Browser geladen. ES Modules werden nativ unterstützt (`<script type="module">`).
6. **Bestehende API-Schicht**: `ApiClient` und `API_ENDPOINTS` werden nahezu 1:1 aus GUI2 übernommen.

### State-Persistenz-Konzept

Da GUI3 eine SPA ist (kein Page-Reload), bleibt der In-Memory-State beim Tab-Wechsel erhalten. Für **Page-Reload / Browser-Neustart** gibt es zwei Persistenz-Ebenen:

#### Ebene 1: `localStorage` (client-seitig, sofort)

| State | Key | Inhalt | Warum localStorage |
|---|---|---|---|
| `ui-state` | `gui3.ui` | `locale`, `theme`, `activeTab`, `activeSubTab` | Reine UI-Präferenz, kein Backend-Bezug |
| `input-scan` | `gui3.inputScan` | `inputDir`, `pattern`, `outputDir`, `runName`, `frameMin`, `colorMode`, `sortMode`, `queue[]`, `calibration{bias,dark,flat}` | User soll bei Reload nicht alles neu eingeben; wird vor Scan-Start nicht ans Backend gesendet |
| `ai-config` | `gui3.aiConfig` | `provider`, `model`, `apiKeySaved` (bool, nicht der Key selbst) | Auswahl merken; API-Key wird sicher am Backend gespeichert |

#### Ebene 2: Server-UI-State (`/api/app/ui-state`, backend-seitig)

| State | Pfad | Inhalt | Warum Server |
|---|---|---|---|
| `currentRunId` | `gui3.currentRunId` | Aktuell ausgewählter Run | Muss über Sessions/Clients synchronisiert sein (Backend kennt den Run) |
| `runReadyStatus` | `gui3.runReadyStatus` | Guardrail-Status | Wird vom Backend berechnet, nicht nur UI |
| `currentRunDir` | `gui3.currentRunDir` | Run-Verzeichnis | Backend referenziert diesen Pfad |
| `configRevision` | `gui3.configRevision` | Aktive Config-Revision | Backend muss wissen, welche Revision gilt |
| `pipelineMode` | `gui3.pipelineMode` | Full/Reduced/Emergency | Abhängig von Assumptions + Frame-Count aus Scan |

#### Ebene 3: Nicht persistent (flüchtig, In-Memory nur)

| State | Warum nicht persistent |
|---|---|
| `config-draft` (uncommitted Änderungen im Editor) | Dirty-State soll bei Reload verworfen werden; User muss explizit "Save" klicken |
| `validation-result` | Wird bei Reload neu berechnet aus aktuellem Config-Draft |
| `log-lines[]` (Run Monitor) | Werden via WebSocket neu geladen; Historie via `/api/runs/{id}/logs` |
| `phase-progress` | Live-Daten, bei Reload via WebSocket/API neu abgefragt |
| `ai-analysis` (currentAnalysis) | Wird bei Bedarf neu angefordert oder aus History geladen |
| `ai-trafficLog` | Nur für aktuelle Session relevant |
| `scan-result` (letztes Scan-Ergebnis) | Wird via `/api/scan/latest` neu geladen |
| `run-history[]` | Wird via `/api/runs` neu geladen |
| `artifacts[]` | Wird via `/api/runs/{id}/artifacts` neu geladen |

#### Persistenz-Flow

```
Tab-Wechsel (Processing → Tools → History → Processing)
  → State bleibt im Memory (kein Persistenz nötig)
  → Nur activeTab/activeSubTab wird in localStorage geschrieben

Page-Reload (F5 / Browser-Neustart)
  1. localStorage laden → locale, theme, activeTab, inputForm, aiConfig
  2. Server-UI-State laden (GET /api/app/ui-state) → currentRunId, runReadyStatus, configRevision
  3. Scan-Ergebnis laden (GET /api/scan/latest) → falls vorhanden
  4. Config laden (GET /api/config/current) → für Parameter-Editor
  5. Run-Status laden (GET /api/runs/{currentRunId}/status) → falls Run aktiv
  6. WebSocket verbinden → falls Run läuft
  → UI ist wiederhergestellt

"Save" im Parameter-Editor
  → POST /api/config/save → neue Revision
  → configRevision in Server-UI-State schreiben
  → config-draft wird zurückgesetzt (nicht mehr dirty)
```

#### Implementierung im `store.js`

```javascript
// js/state/store.js
const PERSIST_LOCAL = new Set(["ui-state", "input-scan", "ai-config"]);
const PERSIST_SERVER = new Set(["ui-state"]);  // nur ui-state hat server-seitige Keys

export function persistState(key, state) {
  if (PERSIST_LOCAL.has(key)) {
    localStorage.setItem(`gui3.${key}`, JSON.stringify(state));
  }
}

export function loadPersistedState(key) {
  if (!PERSIST_LOCAL.has(key)) return null;
  const raw = localStorage.getItem(`gui3.${key}`);
  return raw ? JSON.parse(raw) : null;
}

// Server-UI-State sync
export async function syncServerUiState(patch) {
  await api.patch("/api/app/ui-state", patch);
}

export async function loadServerUiState() {
  return await api.get("/api/app/ui-state");
}
```

---

## 2. Tab-Struktur und Navigation

### 2.1 Haupt-Tabs

```
┌─────────────────────────────────────────────────────────────────┐
│  [Logo] tile_compile    [Processing] [Tools] [History]    [DE|EN] [☾/☀]  │
│  Status: ● Run ready  ● Guardrails OK                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  << Sub-Tab-Leiste (kontextabhängig) >>                         │
│                                                                 │
│  [ Tab-Inhalt ]                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Tab 1: Processing

Sub-Tabs (horizontal):

```
[ Input & Scan ]  [ Parameter ]  [ Run Monitor ]
```

**Flow**: Input & Scan → Parameter → Run Monitor (linear, aber frei navigierbar)

**Guardrail-Badges** in der Sub-Tab-Leiste zeigen Status pro Step:
- Input & Scan: 🔵 Scan-Status
- Parameter: ✅/⚠️ Validation-Status
- Run Monitor: ● Run-Status

### 2.3 Tab 2: Tools

Sub-Tabs (horizontal):

```
[ Raw Stack ]  [ Astrometry ]  [ PCC ]
```

Standalone-Tools für Post-Processing: Raw Stack (Preprocessing-Pipeline), Astrometry (Plate Solving), PCC (Photometric Color Calibration). Unabhängig von Pipeline-Runs einsetzbar.

### 2.4 Tab 3: History

Sub-Tabs (horizontal):

```
[ Run History ]
```

Run-Historie mit Detail-View, Run-Vergleich, Stats und Report. Fokus auf vergangene Runs – keine aktiven Tools hier.

---

## 3. Logging-Konzept

### 3.1 Log-Ebenen

| Ebene | Farbe | Verwendung |
|---|---|---|
| `error` | Rot | Fehler, Abbrüche |
| `warning` | Gelb/Orange | Warnungen, Guardrail-Verletzungen |
| `info` | Blau | Phasen-Übergänge, Status |
| `debug` | Grau | Detail-Output, nur auf Wunsch |
| `trace` | Sehr hell | Roh-Daten, KI-Verkehr |

### 3.2 Log-Anzeige im Run Monitor

```
┌──────────────────────────────────────────────────────────────┐
│ Live Log                                    [All▼] [⏸] [⬇]  │
├──────────────────────────────────────────────────────────────┤
│ 21:15:32 INFO   Phase SCAN started                           │
│ 21:15:33 INFO   Found 325 frames in /data/M31                │
│ 21:15:34 INFO   Color mode: OSC, Bayer: RGGB                 │
│ 21:15:35 INFO   Phase SCAN completed (3.2s)                  │
│ 21:15:35 INFO   Phase REGISTRATION started                   │
│ 21:15:42 WARN   Frame 47: low CC=0.31, using sequential      │
│ 21:16:01 INFO   Phase REGISTRATION completed (26.1s)         │
│ ...                                                          │
│ (virtualisierte Liste, max 10.000 Zeilen, auto-scroll)       │
└──────────────────────────────────────────────────────────────┘
```

**Features**:
- **Filter-Dropdown**: All / Info / Warning / Error / Debug (Mehrfachauswahl)
- **Search**: Live-Suche im Log-Text
- **Pause/Resume**: Auto-scroll stoppen/starten
- **Export**: Log als Text-Datei herunterladen
- **Virtualisierung**: Canvas-basiertes Rendering oder `IntersectionObserver`-basierte Zeilen-Wiederverwendung für Performance bei großen Logs
- **Auto-scroll**: Neue Zeilen automatisch nach unten scrollen (wenn nicht pausiert)
- **Phase-Marker**: Phasen-Übergänge als visuelle Trennlinie

### 3.3 Toast-Notifications

Für asynchrone Events (nicht-blockierend):

```
┌──────────────────────────────────┐
│ ✅ Scan completed                │
│ 325 frames, OSC, RGGB            │
│                          [×]     │
└──────────────────────────────────┘
```

- Success (grün), Warning (gelb), Error (rot), Info (blau)
- Auto-dismiss nach 5s (configurierbar)
- Stack-Position: unten rechts
- Klick → Navigation zum relevanten Tab

### 3.4 KI-Datenverkehr-Log

Als **aufklappbares Detail** im AI-Tab (nicht permanent sichtbar):

```
▶ KI-Datenverkehr (ausgeblendet)
▼ KI-Datenverkehr
  Request: POST /api/scan/analysis
  Model: claude-sonnet-4-20250514
  Tokens: 1.247 input / 892 output
  Duration: 4.2s
  [Vollständige Antwort anzeigen...]
```

---

## 4. KI-Empfehlungs-Integration

### 4.1 Konzept: "AI Tab" im Parameter Studio

Der Parameter-Sub-Tab bekommt einen **internen Tab-Umschalter**:

```
[ Parameter ] [ AI Empfehlung ]
```

### 4.2 AI Empfehlung Tab

```
┌──────────────────────────────────────────────────────────────┐
│ AI Empfehlung                                                │
├──────────────────────────────────────────────────────────────┤
│ ┌─ Scan-Kontext ──────────────────────────────────────────┐ │
│ │ Mount: EQ/Tracker    Target: Compact    Camera: OSC     │ │
│ │ Calibration: Darks ✓  Flats ✗  Bias ✗                 │ │
│ │ Notes: Guiding 0.8", M31, alt-az test                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─ Modell ────────────────────────────────────────────────┐ │
│ │ Provider: [anthropic▼]  Model: [claude-sonnet-4▼]      │ │
│ │ API-Key: [••••••••••]  [Key speichern]  ✓ gespeichert   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ [ KI-Analyse erstellen ]  [ Neu analysieren ]  [ History▼ ] │
│                                                              │
│ ┌─ Empfehlungen ──────────────────────────────────────────┐ │
│ │ ☑ registration.engine: hybrid_phase_ecc                 │ │
│ │   Begründung: Alt-Az Mount → starke Feldrotation...     │ │
│ │   Aktuell: triangle_star_matching → Empfohlen: hybrid   │ │
│ │                                                          │ │
│ │ ☑ bge.fit.method: rbf                                   │ │
│ │   Begründung: Starker Gradient bei M31...               │ │
│ │   Aktuell: rbf → Empfohlen: rbf (bereits optimal)       │ │
│ │                                                          │ │
│ │ ☐ aqmh.cherry_pick.enabled: true                        │ │
│ │   Begründung: 325 Frames, einige mit Seeing-Schwankung  │ │
│ │   Aktuell: false → Empfohlen: true (k_frac=0.30)        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ [ Ausgewählte anwenden ]  [ Alle anwenden ]  [ Verwerfen ]  │
│                                                              │
│ ▶ KI-Datenverkehr (ausgeblendet)                             │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 KI-Workflow

```
1. User macht Scan (Input & Scan Tab)
   → Scan-Ergebnisse + Kontext werden gespeichert (scanStore)

2. User wechselt zu Parameter Tab → AI Empfehlung Sub-Tab
   → Scan-Kontext wird automatisch geladen
   → User wählt Modell + gibt API-Key ein (einmalig)
   → User klickt "KI-Analyse erstellen"

3. Backend ruft AI-Provider auf
   → Scan-Daten + Kontext → Prompt → AI-Modell
   → Empfehlungen als strukturierte JSON-Response

4. Empfehlungen werden angezeigt
   → Jede Empfehlung: Parameter, aktueller Wert, empfohlener Wert, Begründung
   → Checkbox pro Empfehlung (selektive Anwendung)
   → "Ausgewählte anwenden" → Config-Patch

5. Angewendete Empfehlungen werden im YAML-Diff sichtbar
   → User kann im Parameter-Tab die Änderungen prüfen
   → Validierung läuft automatisch nach Apply
```

### 4.4 KI-State (Vanilla JS)

```javascript
// js/state/ai-state.js
const state = {
  config: null,              // { enabled, model, provider }
  models: [],                // verfügbare Modelle
  authStatus: new Map(),     // provider -> 'ok' | 'missing' | 'error'
  currentAnalysis: null,     // AiAnalysis | null
  analysisHistory: [],       // AiAnalysis[]
  isAnalyzing: false,
  trafficLog: [],            // TrafficEntry[]
};

const listeners = new Set();

export function getState() { return state; }
export function setState(patch) { Object.assign(state, patch); listeners.forEach(fn => fn(state)); }
export function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }

export async function loadConfig() { /* GET /api/ai/config */ }
export async function loadModels() { /* GET /api/ai/models */ }
export async function saveApiKey(provider, key) { /* POST /api/ai/auth/{provider} */ }
export async function createAnalysis(context) { /* POST /api/scan/analysis */ }
export async function loadAnalysis(id) { /* GET /api/scan/analysis/{id} */ }
export async function applyRecommendations(selected) { /* POST /api/config/patch */ }
```

---

## 5. Design-System

### 5.1 Color Tokens (CSS Custom Properties)

```css
/* css/tokens.css */

/* Light Theme (Default) */
:root {
  --bg:           #ffffff;
  --bg-page:      #eef4fa;
  --bg-page-2:    #dde8f2;
  --surface:      #ffffff;
  --surface-2:    #f8fafb;
  --foreground:   #1b2737;
  --muted:        #5d7087;
  --primary:      #15808d;
  --primary-soft: #d9f0f3;
  --border:       #cfdbe7;
  --success:      #166534;
  --success-bg:   #dff7e8;
  --warning:      #92400e;
  --warning-bg:   #fef3c7;
  --error:        #991b1b;
  --error-bg:     #fee2e2;
  --info:         #1d4ed8;
  --info-bg:      #dbeafe;
  --radius:       12px;
  --radius-sm:    8px;
  --radius-lg:    16px;
}

/* Dark Theme */
html[data-theme="dark"] {
  --bg:           #0f172a;
  --bg-page:      #0d1117;
  --bg-page-2:    #161b22;
  --surface:      #1e293b;
  --surface-2:    #334155;
  --foreground:   #f1f5f9;
  --muted:        #94a3b8;
  --primary:      #2dd4bf;
  --primary-soft: #134e4a;
  --border:       #334155;
  --success:      #4ade75;
  --success-bg:   #052e16;
  --warning:      #fbbf24;
  --warning-bg:   #422006;
  --error:        #f87171;
  --error-bg:     #450a0a;
  --info:         #60a5fa;
  --info-bg:      #172554;
}
```

### 5.2 Typography

```css
:root {
  --font-ui:    "Inter", system-ui, sans-serif;
  --font-title: "Inter", system-ui, sans-serif;
  --font-mono:  "JetBrains Mono", "Fira Code", monospace;
  
  --text-xs:    12px;
  --text-sm:    14px;
  --text-base:  16px;
  --text-lg:    18px;
  --text-xl:    20px;
  --text-2xl:   24px;
}
```

### 5.3 Komponenten (CSS-Klassen)

| Klasse | Verwendung |
|---|---|
| `.tc-tabs` / `.tc-tab` | Haupt-Tabs + Sub-Tabs + Parameter/AI-Umschalter |
| `.tc-card` | Scan-Result, Parameter-Groups, Stats |
| `.tc-input` | Pfad-Eingaben, Zahlen-Felder |
| `.tc-select` | Dropdowns (Presets, Modelle, etc.) |
| `.tc-checkbox` | KI-Empfehlungen, Calibration-Toggles |
| `.tc-btn` / `.tc-btn-primary` / `.tc-btn-secondary` | Aktionen (Scan, Validate, Run, etc.) |
| `.tc-badge` / `.tc-badge-success` / `.tc-badge-warning` / `.tc-badge-error` | Status-Chips, Guardrail-Indikatoren |
| `.tc-modal` | Artefakt-Viewer, Modal-Dialoge |
| `.tc-toast` / `.tc-toast-container` | Notifications |
| `.tc-scroll` | Scrollbare Bereiche (Log, Parameter) |
| `.tc-accordion` | KI-Datenverkehr, erweiterte Optionen |
| `.tc-progress` | Phasen-Fortschritt, Download-Status |
| `.tc-tooltip` | Parameter-Erklärungen |
| `.tc-separator` | Visuelle Trennung |
| `.tc-switch` | Boolean-Parameter (Toggle) |

### 5.4 Komponenten-Factory-Pattern

Jede Komponente ist eine JS-Function, die ein DOM-Element zurückgibt:

```javascript
// js/components/path-input.js
export function createPathInput({ label, value, onBrowse, onChange }) {
  const wrapper = document.createElement('div');
  wrapper.className = 'tc-path-input';
  wrapper.innerHTML = `
    <label class="tc-label">${label}</label>
    <div class="tc-input-row">
      <input type="text" class="tc-input" value="${value || ''}" />
      <button class="tc-btn tc-btn-secondary tc-btn-icon">📂</button>
    </div>
  `;
  const input = wrapper.querySelector('input');
  const btn = wrapper.querySelector('button');
  input.addEventListener('input', () => onChange?.(input.value));
  btn.addEventListener('click', () => onBrowse?.());
  return wrapper;
}
```

### 5.5 Minimaler State-Manager

```javascript
// js/state/store.js
export function createStore(initialState = {}) {
  let state = { ...initialState };
  const listeners = new Set();
  
  return {
    getState: () => state,
    setState: (patch) => {
      state = { ...state, ...patch };
      listeners.forEach(fn => fn(state));
    },
    subscribe: (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
```

---

## 6. Ablauf-Diagramme

### 6.1 Processing-Flow (Tab 1)

```
┌─────────────────────────────────────────────────────────────┐
│                    TAB 1: PROCESSING                         │
│                                                             │
│  ┌─ Input & Scan ─┐    ┌─ Parameter ──┐    ┌─ Run Monitor ┐│
│  │                │    │              │    │              ││
│  │ 1. Input-Dir   │───▶│ 4. Preset    │───▶│ 7. Start Run ││
│  │ 2. Scan        │    │ 5. Parameter │    │ 8. Phasen    ││
│  │ 3. Calibration │    │ 6. Validate  │    │ 9. Logs      ││
│  │                │    │   + AI Tab   │    │ 10. Stats    ││
│  │                │    │   + Assumpt. │    │ 11. Resume   ││
│  └────────────────┘    └──────────────┘    └──────────────┘│
│                                                             │
│  Guardrails: [Scan ✓] [Config ✓] [Cal ✓] [BGE/PCC ✓]       │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Parameter-Tab mit AI-Sub-Tab

```
┌─ Parameter Tab ─────────────────────────────────────────────┐
│                                                             │
│  [ Parameter ] [ AI Empfehlung ]                            │
│                                                             │
│  ┌─ Parameter (aktiv) ──────────────────────────────────┐   │
│  │ ┌─ Kategorien ──┐  ┌─ Editor ──────┐  ┌─ Explain ──┐│   │
│  │ │ • All         │  │ registration │  │ Label: ...  ││   │
│  │ │ • System      │  │ .engine      │  │ Path: ...   ││   │
│  │ │ • Pipeline    │  │ .star_topk   │  │ Default: .. ││   │
│  │ │ • Registration│  │ ...          │  │ Range: ...  ││   │
│  │ │ • BGE         │  │              │  │             ││   │
│  │ │ • AQMH        │  │ [Validate]   │  │ Situation   ││   │
│  │ │ • PCC         │  │ [Save]       │  │ Assistant   ││   │
│  │ │ • Assumptions │  │              │  │             ││   │
│  │ │ ...           │  │              │  │ YAML Diff   ││   │
│  │ └───────────────┘  └──────────────┘  └─────────────┘│   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ AI Empfehlung (bei Tab-Klick) ──────────────────────┐   │
│  │ Scan-Kontext (auto aus Scan)                          │   │
│  │ Modell + API-Key                                      │   │
│  │ [Analyse erstellen]                                   │   │
│  │ Empfehlungen mit Checkboxen                           │   │
│  │ [Ausgewählte anwenden] → Patch Config                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Assumptions-Integration

Assumptions wird als **eigene Kategorie** im Parameter-Editor integriert:

```
Kategorien:
  • All
  • System
  • Pipeline
  ...
  • Assumptions  ← neu integriert (war eigene Seite)
  ...
```

Zusätzlich ein **KPI-Badge** oben im Parameter-Tab:

```
Pipeline-Modus: [Full Mode (≥200 Frames)]  ← live aktualisiert
```

---

## 7. Phasen-Plan

### Phase 1: Setup & Grundgerüst (1-2 Tage)
- `index.html` SPA-Shell (Header, Tab-Bar, Content-Root)
- `tokens.css` + `base.css` + `components.css` Grundgerüst
- Tab-Routing (Hash-basiert: `#processing`, `#tools`, `#history`)
- `store.js` minimaler Pub/Sub State-Manager
- API-Client Migration (`api.js` → `api/client.js`)
- `endpoints.js` (aus `constants.js`)
- i18n-Modul (aus `utils.js` + bestehende JSONs)
- `header.js` + `tab-bar.js` + `sub-tabs.js` + `toast.js`

### Phase 2: Input & Scan (2-3 Tage)
- `path-input.js` Komponente (Input + Browse)
- `queue-editor.js` Komponente
- `calibration-panel.js` Komponente
- Scan-Logik (API-Aufrufe, Result-Anzeige)
- Scan-Result-Card
- `guardrail-badges.js`

### Phase 3: Parameter Studio + Assumptions (3-4 Tage)
- Kategorie-Liste
- Parameter-Editor (dynamisch aus Schema)
- `explain-panel.js`
- `situation-assistant.js`
- `yaml-diff.js`
- Validierung
- Assumptions als Kategorie integriert
- Pipeline-Modus-Badge

### Phase 4: AI Empfehlung Tab (2-3 Tage)
- `ai-panel.js` Komponente (einheitlich)
- Scan-Kontext-Übernahme
- Modell/Key-Verwaltung
- Empfehlungs-Liste mit Checkboxen
- Apply-Logik (Config-Patch)
- KI-Datenverkehr (Accordion)
- AI-History

### Phase 5: Run Monitor (2-3 Tage)
- `phase-list.js` Komponente
- WebSocket-Manager (aus bestehendem Code migriert)
- `log-viewer.js` (virtualisiert, mit Filter/Search/Pause)
- Stats-Panel
- Resume-Panel (Config-Revision, Template, YAML-Editor)
- `artifact-list.js` + `modal.js` Viewer
- Batch-Summary

### Phase 6: Tools + History (3-4 Tage)
- `tools.js` Container (Tab 2: Raw Stack, Astrometry, PCC)
- `raw-stack.js` (Preprocessing-Pipeline)
- `astrometry.js` (ASTAP Setup, Catalog, Solve)
- `pcc.js` (Input, Catalog, Parameters, Run, Result)
- `history.js` Container (Tab 3: Run History)
- `run-history.js` (Liste, Details, Vergleich)

### Phase 7: Polish & Testing (2-3 Tage)
- Dark Mode (CSS Custom Properties umschalten)
- Responsive Layout (Tablet, kleiner Desktop)
- Toast-System für alle asynchronen Events
- Keyboard-Shortcuts
- Error-Handling (try/catch + Toast)
- Performance-Optimierung (Log-Virtualisierung, DOM-Recycling)
- Feature-Parity-Checkliste (gegen GUI2)

### Gesamt: ~15-20 Tage

---

## 8. Feature-Parity-Checkliste

### muss in GUI3 vorhanden sein:

- [ ] Input-Ordner Auswahl + Browse
- [ ] Run-Queue Editor (Filter, Pattern, Label, Aktiv)
- [ ] Kalibrierung (Bias/Dark/Flat, Ordner/Master)
- [ ] Scan ausführen + Ergebnis anzeigen
- [ ] Parameter-Editor mit allen Kategorien
- [ ] Parameter-Suche
- [ ] Explain-Panel
- [ ] Situation Assistant
- [ ] YAML-Diff
- [ ] Validierung (Schema + Semantik)
- [ ] Config speichern / speichern unter
- [ ] Config-Revisionen
- [ ] Preset-Selector + Apply
- [ ] Assumptions (frames_min, reduced_threshold, skip_clustering, cluster_range)
- [ ] Pipeline-Modus-Anzeige
- [ ] KI-Analyse (Modell, Key, Kontext, Analyse erstellen)
- [ ] KI-Empfehlungen anzeigen + anwenden
- [ ] KI-History
- [ ] Run starten / stoppen
- [ ] Phasen-Liste mit Status + Progress
- [ ] Live Log (WebSocket, Filter, Search, Export)
- [ ] Stats generieren + öffnen
- [ ] Report öffnen
- [ ] Resume (Phase, Config-Revision, Template)
- [ ] Artefakt-Liste + Viewer
- [ ] Run-Ordner öffnen
- [ ] Run-Historie (Liste, Details)
- [ ] Run-Vergleich
- [ ] Run löschen
- [ ] Raw Stack (Preprocessing-Pipeline)
- [ ] Astrometry (ASTAP, Catalog, Solve, Save-Solved)
- [ ] PCC (Input, Catalog, Parameters, Run, Save, Result)
- [ ] Guardrails
- [ ] i18n (DE/EN)
- [ ] Locale-Switch
- [ ] Server-UI-State (Persistierung)

### entfällt in GUI3:

- [x] Dashboard (KPIs, Guided Run, Pipeline-Vorschau) → integriert in Processing-Header
- [x] Wizard (eigene Seite) → linearer Flow in Processing-Tabs
- [x] Live Log (eigene Seite) → integriert in Run Monitor
- [x] Assumptions (eigene Seite) → integriert in Parameter-Kategorien

---

## 9. Backend-Anpassungen

**Keine API-Änderungen erforderlich.** Das C++ Backend bleibt unverändert.

Siehe Migration-Strategie (Abschnitt 10) für statische Auslieferung.

---

## 10. Migration-Strategie

### Ablauf

```
Sofort:
  /ui/   → GUI3 (neu, Entwicklung + Produktion)
  /ui2/  → GUI2 (Fallback, vorerst)
```

- GUI3 wird direkt unter `/ui/` entwickelt und getestet
- GUI2 wechselt sofort auf `/ui2/` als Fallback
- Langfristig: `/ui2/` wird entfernt, wenn GUI3 stabil und Feature-Parity bestätigt

### Backend-Anpassung

- `web_backend_cpp` liefert `web_frontend_v3/` unter `/ui/` aus
- `web_backend_cpp` liefert bisheriges `web_frontend/` unter `/ui2/` aus
- **Kein Build-Schritt nötig** – Dateien werden direkt vom Browser geladen (ES Modules nativ)
- Falls nötig: MIME-Type `application/javascript` für `.js` sicherstellen (für ES Module-Imports)
