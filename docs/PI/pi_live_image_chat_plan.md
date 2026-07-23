# PI Live Image Chat — Plan

**Status:** Implemented (Phase 1-2)
**Datum:** 2026-07-22
**Modul:** `web_backend_cpp`, `web_frontend_v3`, `agent_service`

---

## 1. Ziel und Vision

Nach Abschluss eines Runs wird ein Vorschaubild angezeigt. Beim Klick auf dieses
Preview soll ein **realistisches Foto in voller (1:1) Auflösung** generiert werden
(mit Fortschrittsmeldung "Generiere Bild…"). Anhand dieses Fotos soll der Nutzer
über den **Run-Chat mit der PI-AI** live Korrekturen vornehmen können — in
natürlicher Sprache, interaktiv, iterativ.

### Beispiel-Kommandos

| Eingabe (Chat) | Erwartete Aktion |
|----------------|-----------------|
| "helle die mitteltöne auf" | Gamma/Midtone-Anhebung |
| "erhöhe den kontrast ein wenig" | Kontrast-Kurve steiler |
| "erhöhe die farbsättigung" | Sättigung einmalig anheben, danach +/- Buttons zum iterativen Nachjustieren (kein automatischer Loop — siehe 7.3) |
| "unterdrück farbrauschen" | Rauschfilter / Chroma-Denoise |
| "schärfe ein wenig" | Unsharp Mask / Sharpening |

### Abgrenzung zum bestehenden Run-Chat

**Hinweis:** "bis ich stop sage" beschreibt hier nur die *Nutzerintention*
(wiederholtes Verstärken), nicht einen automatisch laufenden Prozess. Die
tatsächliche Umsetzung ist der explizit diskrete Adjust-Mechanismus (+/−
Buttons, Abschnitt 7.3) — es gibt bewusst **keinen** serverseitigen Loop mit
Stop-Erkennung (Begründung: nicht abbrechbare Hintergrundprozesse mit
unklarem Endzustand wären fehleranfällig und schwer zu undo/redo-en).

Der bestehende Run-Chat (`/api/pi/run-chat`) analysiert das Vorschaubild und
empfiehlt **Config-Parameteränderungen** für einen Resume. Der neue Live Image
Chat arbeitet **nicht auf der Pipeline-Config**, sondern auf **Bild-Operationen**
auf das bereits generierte Ausgabebild — wie ein interaktiver Bildeditor, gesteuert
durch natürliche Sprache.

---

## 2. Bestehende Infrastruktur

| Komponente | Datei | Status |
|-----------|-------|--------|
| Run-Preview-Panel | `web_frontend_v3/js/components/run-image-preview.js` | Zeigt kleinstes Preview-Bild (max 1400px PNG) |
| FITS→PNG Renderer (Backend) | `web_backend_cpp/src/routes/runs_routes.cpp:469` | `render_fits_preview_png()`, max 1400px, gamma 0.6 |
| FITS→PNG Renderer (PI) | `web_backend_cpp/src/routes/pi_routes.cpp:346` | `render_fits_preview_png_for_pi()`, max 1024px |
| Run-Chat API | `web_backend_cpp/src/routes/pi_routes.cpp:1649` | `/api/pi/run-chat` — sendet base64 PNG an AI |
| Run-Chat Frontend | `web_frontend_v3/js/pages/run-monitor.js` | `createRunChatPanel()` — Chat-UI im Run-Monitor |
| HMS Preview | `web_frontend_v3/js/components/hms-preview.js` | Live-HMS-Parameter-Vorschau mit Proxy-Bild |
| AI Sidecar | `agent_service/` | Node.js, Multi-Provider, Bild-Analyse |

---

## 3. Architektur

```
┌─────────────────────────────────────────────────────────┐
│ Frontend (GUI3)                                         │
│                                                         │
│  Run Monitor → Preview-Panel                            │
│       │ Klick auf Preview                               │
│       ▼                                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Live Image Viewer (neu)                          │   │
│  │  • Fordert 1:1 JPEG vom Backend an              │   │
│  │  • Zeigt "Generiere Bild…" Overlay              │   │
│  │  • Stellt Bild in zoombarer Ansicht dar         │   │
│  │  • Chat-Panel rechts/neben dem Bild             │   │
│  └──────────────────────────────────────────────────┘   │
│       │ Chat-Nachricht                                  │
│       ▼                                                 │
│  POST /api/pi/live-image-chat                           │
│       { session_id, message }                           │
│       (run_id ist bereits über session_id serverseitig   │
│        an die Session gebunden, siehe 6.2)               │
│                                                         │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│ Backend (web_backend_cpp)                               │
│                                                         │
│  1. Lädt FITS-Output (stacked_rgb_hms/pcc/bge)          │
│  2. Rendert 1:1 JPEG (full resolution, gamma, stretch)  │
│  3. Behält Bild + Session-State im Speicher             │
│  4. Sendet Bild (base64) + Nachricht an AI Sidecar      │
│  5. Empfängt Bild-Operations-JSON                       │
│  6. Wendet Operationen mit OpenCV an                    │
│  7. Re-Encodiert als JPEG                               │
│  8. Gibt neues Bild + Erklärung zurück                  │
│                                                         │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│ AI Sidecar (agent_service)                              │
│                                                         │
│  • Empfängt Bild (base64) + Chat-Nachricht              │
│  • Vision-Modell analysiert Bild                        │
│  • Übersetzt natürliche Sprache → Bild-Operations-JSON  │
│  • Gibt strukturierte Antwort zurück:                   │
│    { operations: [...], summary, stop_flag }            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Bild-Operations-Schema

Die AI gibt strukturierte Operationen zurück, die der Backend deterministisch
anwendet. Dies stellt sicher, dass die AI nie direkt auf Pixel zugreift — sie
beschreibt nur, was gemacht werden soll.

```json
{
  "schema_version": "pi.live-image-chat.v1",
  "session_id": "string",
  "summary": "Ich habe die Mitteltöne aufgehellt und den Kontrast leicht erhöht.",
  "operations": [
    {
      "type": "brightness",
      "params": { "midtones": 0.15, "shadows": 0.0, "highlights": 0.0 }
    },
    {
      "type": "contrast",
      "params": { "amount": 0.1 }
    }
  ],
  "adjustable": true,
  "adjust_step": {
    "type": "brightness",
    "params": { "midtones": 0.05 },
    "label": "Mitteltöne"
  },
  "warnings": []
}
```

Wenn `adjustable: true`, zeigt das Frontend +/- Buttons an. `adjust_step`
definiert die Operation, die bei Klick auf + (vorwärts) oder - (rückwärts)
angewendet wird. So kann der User den Effekt iterativ verstärken oder
verringern, ohne neue Chat-Nachrichten zu tippen.

### Unterstützte Operationen (Phase 1)

| Type | User-Label (Chat/Dropdown) | Parameter | OpenCV-Implementation |
|------|---------------------------|-----------|----------------------|
| `brightness` | Helligkeit / Mitteltöne | `midtones` (-1..1), `shadows` (-1..1), `highlights` (-1..1) | Gamma-Kurve / LUT |
| `contrast` | Kontrast | `amount` (-1..1) | `cv::convertScaleAbs` oder Sigmoid-Kurve |
| `saturation` | Farbsättigung | `amount` (-1..1) | HSV-Sättigungsskalierung |
| `sharpen` | Schärfe | `amount` (0..1), `radius` (0.5..5) | Unsharp Mask: `img + amount * (img - blur(img))` |
| `denoise` | Rauschunterdrückung | `strength` (0..1), `luminance` (bool) | `cv::fastNlMeansDenoisingColored` |
| `rmgreen` | Grüne Farbnebel entfernen | `strength` (0..1) | SCNR: `G -= min(G, R, B) * strength` |
| `clahe` | Lokaler Kontrast (Details) | `cliplimit` (1..10), `tilesize` (8..64) | `cv::createCLAHE()` pro Kanal |
| `bilateral` | Rauschunterdrückung (detailerhaltend) | `d` (3..15), `sigma_color` (10..150), `sigma_space` (10..150) | `cv::bilateralFilter()` |
| `threshold` | Schwarzwert / Weißwert | `black_point` (0..1), `white_point` (0..1) | `cv::threshold()` / LUT-Clipping |
| `invert` | Negativ anzeigen | — | `cv::bitwise_not()` (255 - Pixelwert je Kanal) |
| `crop` | Zuschneiden | `x`, `y`, `w`, `h` | ROI-Ausschnitt |
| `reset` | Zurücksetzen | — | Zurück auf Originalbild |

**Hinweis `crop`:** Crop ist jetzt auch im Live-Image-Chat verfügbar. Die KI
erhält die Bildabmessungen und darf Crop nur bei einer ausdrücklichen
Zuschneideanweisung mit gültigen Pixelkoordinaten vorschlagen. Der lokale
Fallback unterstützt Prozentangaben wie "schneide 10% Rand ab"; ohne Angabe
wird ein konservativer 5%-Rand pro Seite entfernt. Ein manuelles
Maus-Rechteck im Viewer bleibt als mögliche spätere Ergänzung bestehen.

### Unterstützte Operationen (Phase 2 — optional)

| Type | User-Label (Chat/Dropdown) | Parameter | Beschreibung |
|------|---------------------------|-----------|-------------|
| `levels` | Tonwertkorrektur | `in_low`, `in_high`, `gamma`, `out_low`, `out_high` | Levels wie Photoshop |
| `curves` | Tonwertkurve | `points: [[x,y],...]` | Custom Tone Curve |
| `vibrance` | Lebendigkeit (intelligente Sättigung) | `amount` (-1..1) | HSV: S-Skalierung abhängig von aktuellem S-Wert |
| `color_temperature` | Farbtemperatur (warm/kalt) | `amount` (-1..1) | RGB-Addition: R↑B↓ (warm) oder B↑R↓ (kalt) |
| `shadow_recovery` | Schatten aufhellen | `amount` (0..1) | LUT mit Sigmoid im unteren Bereich |
| `highlight_recovery` | Spitzlichter zurückholen | `amount` (0..1) | LUT mit Sigmoid im oberen Bereich |
| `star_desaturation` | Übersteuerte Sterne entschärfen | `amount` (0..1) | HSV: High-V → S reduzieren |
| `fixbanding` | Streifenartefakte entfernen | `amount` (0..1), `sigma` (0.5..5) | Zeilen/Spalten-Median-Subtraktion |
| `unpurple` | Lila Farbsäume entfernen | `amount` (0..1) | HSV: Purple-Range → Sättigung reduzieren |
| `color_balance` | Farbbalance | `r`, `g`, `b` per range (shadows/mids/highs) | Farbbalance |
| `deconvolution` | Entfaltung (Deconvolution) | `psf_radius`, `iterations` | Richardson-Lucy |
| `local_contrast` | Detailkontrast (Klarheit) | `amount` (-1..1), `radius` (10..100) | `img + amount * (img - blur(img, large_radius))` |
| `chroma_denoise` | Farbrauschen unterdrücken | `strength` (0..1) | LAB: nur a/b-Kanal denoise |
| `dehaze` | Kontrast in nebligen Bereichen | `amount` (0..1) | Dark Channel Prior oder lokale Kontrast-Skalierung |

---

## 5. Session-Management

Jede Live-Image-Chat-Session hat einen Zustand:

```json
{
  "session_id": "uuid",
  "run_id": "string",
  "original_image": "cv::Mat (im Speicher)",
  "current_image": "cv::Mat (im Speicher)",
  "operation_history": [
    { "type": "brightness", "params": {...}, "timestamp": "...", "source": "chat" },
    { "type": "contrast", "params": {...}, "timestamp": "...", "source": "chat" },
    { "type": "saturation", "params": {...}, "timestamp": "...", "source": "adjust" }
  ],
  "undo_stack": [
    { "type": "brightness", "params": {...}, "inverse_params": {...} },
    { "type": "contrast", "params": {...}, "inverse_params": {...} },
    { "type": "clahe", "params": {...}, "snapshot": "cv::Mat (vor Anwendung, nur für nicht invertierbare Ops)" }
  ],
  "redo_stack": [],
  "chat_history": [
    { "role": "user", "content": "helle mitteltöne auf" },
    { "role": "assistant", "content": "Ich habe die Mitteltöne angehoben.", "operations": [...] }
  ],
  "last_adjust_step": null,
  "adjust_count": 0,
  "created_at": "ISO timestamp",
  "last_accessed": "ISO timestamp",
  "persisted": true,
  "source_artifact": "outputs/stacked_rgb_hms.png"
}
```

### Persistenz

Chat-Verlauf und Operations-Historie werden — wie jeder andere PI-Chat —
in `.pi_memory` gespeichert, sodass sie nach Browser-Refresh und Session-Neuaufbau
wiederhergestellt werden können.

**Datei:** `<pi_storage_dir>/live_image_chat/<run_id>_<hash>.json`

Analog zu `pi_run_chat_history_path()` (bestehend in `pi_routes.cpp:423`),
aber im Unterverzeichnis `live_image_chat/` statt `run_chat/`.

**Schema (`pi.live-image-chat-history.v1`):**

```json
{
  "schema_version": "pi.live-image-chat-history.v1",
  "run_id": "string",
  "session_id": "uuid (der zuletzt gespeicherten Session)",
  "source_artifact": "outputs/stacked_rgb_hms.png",
  "chat_history": [
    { "role": "user", "content": "helle mitteltöne auf" },
    { "role": "assistant", "content": "Ich habe die Mitteltöne angehoben.", "operations": [...] }
  ],
  "operation_history": [
    { "type": "brightness", "params": {"midtones": 0.15}, "timestamp": "...", "source": "chat" },
    { "type": "saturation", "params": {"amount": 0.05}, "timestamp": "...", "source": "adjust" }
  ],
  "created_at": "ISO timestamp",
  "last_updated": "ISO timestamp"
}
```

**Speicher-Regel:**
- Nach jeder Chat-Nachricht, Adjust-Operation, Undo und Redo wird die
  History-Datei aktualisiert (append + flush).
- `chat_history` bleibt vollständig erhalten. `operation_history` beschreibt
  dagegen bewusst den aktuell wirksamen Operationspfad: Undo entfernt den
  letzten Eintrag, Redo fügt ihn wieder ein, und eine neue Bearbeitung leert
  den Redo-Pfad. Dadurch bleibt die Rekonstruktion exakt.
- `undo_stack` / `redo_stack` und `cv::Mat`-Snapshots werden **nicht**
  persistiert (nur im RAM). Nach Refresh werden die Stacks leer
  initialisiert; Undo/Redo ist dann erst ab der nächsten Operation verfügbar.
- `current_image` wird zusätzlich als `runs/<run_id>/outputs/live_edit.fits`
  persistiert. Diese Datei ist der kanonische letzte Arbeitsstand; die
  ursprüngliche FITS-Datei bleibt unverändert. Fehlt die Arbeitsdatei, wird
  der Stand einmalig aus `operation_history` rekonstruiert.

### Lifecycle

1. **Create:** Klick auf Preview → `POST /api/pi/live-image-chat/create` → generiert 1:1 JPEG, gibt `session_id` zurück
2. **Chat:** `POST /api/pi/live-image-chat` mit `session_id` + Nachricht → AI → Operationen → Bild aktualisiert → neues JPEG. Wenn `adjustable: true`, werden +/- Buttons angezeigt.
3. **Adjust:** User klickt + oder - → `POST /api/pi/live-image-chat/adjust` → wendet `adjust_step` vorwärts oder rückwärts an → neues JPEG
4. **Undo:** User klickt ↶ → `POST /api/pi/live-image-chat/undo` → nimmt letzte Operation vom `undo_stack`, wendet inverse an, pusht auf `redo_stack` → neues JPEG
5. **Redo:** User klickt ↷ → `POST /api/pi/live-image-chat/redo` → nimmt letzte Operation vom `redo_stack`, wendet vorwärts an, pusht auf `undo_stack` → neues JPEG
6. **Reset:** Nach UI-Bestätigung ersetzt `POST /api/pi/live-image-chat/reset`
   `live_edit.fits` durch eine frische Kopie des unveränderten Quellbilds,
   löscht die History-Datei, leert beide Stacks und aktualisiert die Preview.
7. **Export:** `POST /api/pi/live-image-chat/export` → speichert finales Bild als FITS/PNG in Run-Output
8. **Close:** Session wird nach 30 Min Inaktivität aus dem Speicher entfernt. Chat- und Operations-Historie bleiben in `.pi_memory` erhalten.
9. **Resume (nach Browser-Refresh):** `POST /api/pi/live-image-chat/create`
   mit `run_id` erkennt Arbeitsdatei und/oder History. Wenn `live_edit.fits`
   vorhanden ist, wird sie direkt als aktueller Stand geladen. Nur wenn sie
   fehlt, werden alle Operationen aus `operation_history` sequenziell auf das
   Originalbild angewendet und die Arbeitsdatei neu erzeugt.
   - Lade FITS, rendere 1:1 JPEG (wie bei Create)
   - Stelle `chat_history` in der UI her
   - `undo_stack` / `redo_stack` beginnen leer (Undo nur ab neuer Operation)
   - Gib `session_id` + rekonstruiertes JPEG + `chat_history` zurück
   - Frontend stellt Chat-Verlauf und Bild wieder her, User kann
     nahtlos weiterarbeiten

### Speicher-Management

- 1:1 Bild einer typischen Smart-Telescope-Aufnahme (1080×1080 RGB 8-bit) ≈ 3.5 MB
- Pro Session werden 2 `cv::Mat` gehalten (original + current) ≈ 7 MB
- `undo_stack` / `redo_stack` speichern für **invertierbare** Operationen
  (brightness, contrast, saturation, sharpen, rmgreen) nur Operations-JSON,
  keine Bilder — Inverse wird aus den negierten Parametern berechnet.
- Undo/Redo speichern nur Operations-JSON und bauen den aktuellen Float-Stand
  deterministisch aus der unveränderten Quelle und dem Stack neu auf. Auch
  nicht invertierbare Operationen benötigen daher keine Pixel-Snapshots.
- Worst Case bleibt damit ungefähr 5 Sessions × 7 MB Basisbild; zusätzlich
  wächst nur die kleine Operationshistorie.
- Max. 5 gleichzeitige Sessions, LRU-Eviction bei Überschreitung

---

## 6. API-Endpunkte

### 6.1 Create Session

```
POST /api/pi/live-image-chat/create
Body: { "run_id": "string", "run_dir": "string?" }
Response: {
  "session_id": "uuid",
  "image": { "width": 1080, "height": 1080, "format": "jpeg" },
  "jpeg_base64": "string (1:1 JPEG)",
  "source_artifact": "outputs/stacked_rgb_hms.png",
  "resumed": false,
  "chat_history": []
}
```

Wenn bereits eine History-Datei in `.pi_memory/live_image_chat/` für diese
`run_id` existiert, wird die Session aus der History rekonstruiert:
`resumed: true`, `chat_history` enthält die gespeicherten Nachrichten,
und `jpeg_base64` zeigt das rekonstruierte Bild (Original + alle
Operationen aus `operation_history` angewendet).

### 6.2 Chat (Bild-Operation ausführen)

```
POST /api/pi/live-image-chat
Body: {
  "session_id": "string",
  "message": "helle die mitteltöne auf"
}
Response: {
  "schema_version": "pi.live-image-chat.v1",
  "session_id": "string",
  "summary": "Ich habe die Mitteltöne um 15% aufgehellt.",
  "operations": [...],
  "jpeg_base64": "string (aktualisiertes 1:1 JPEG)",
  "adjustable": true,
  "adjust_step": {
    "type": "brightness",
    "params": { "midtones": 0.05 },
    "label": "Mitteltöne"
  },
  "warnings": []
}
```

Wenn `adjustable: true`, zeigt das Frontend +/- Buttons an. `adjust_step`
definiert die Operation, die bei Klick auf + (vorwärts) oder - (rückwärts)
angewendet wird.

### 6.3 Adjust (+/- Button)

```
POST /api/pi/live-image-chat/adjust
Body: {
  "session_id": "string",
  "direction": "increase" | "decrease"
}
Response: {
  "session_id": "string",
  "summary": "Mitteltöne +5% (Schritt 3)",
  "jpeg_base64": "string (aktualisiertes 1:1 JPEG)",
  "adjust_count": 3,
  "warnings": []
}
```

Der Backend wendet `last_adjust_step` vorwärts an (+) oder kehrt ihn um (-).
Bei `decrease` wird die inverse Operation angewendet (z.B. midtones -0.05
statt +0.05). Der Backend verhindert Parameter-Überlauf (Clamping auf
gültige Bereiche).

### 6.4 Undo

```
POST /api/pi/live-image-chat/undo
Body: { "session_id": "string" }
Response: {
  "session_id": "string",
  "summary": "Rückgängig: Mitteltöne +15%",
  "jpeg_base64": "string (aktualisiertes 1:1 JPEG)",
  "can_undo": true,
  "can_redo": true,
  "undo_count": 2,
  "warnings": []
}
```

Nimmt die letzte Operation vom `undo_stack`, wendet die inverse Operation
auf `current_image` an, pusht die Operation auf `redo_stack`. Wenn
`undo_stack` leer ist → `can_undo: false`, keine Änderung.

### 6.5 Redo

```
POST /api/pi/live-image-chat/redo
Body: { "session_id": "string" }
Response: {
  "session_id": "string",
  "summary": "Wiederherstellen: Mitteltöne +15%",
  "jpeg_base64": "string (aktualisiertes 1:1 JPEG)",
  "can_undo": true,
  "can_redo": false,
  "redo_count": 0,
  "warnings": []
}
```

Nimmt die letzte Operation vom `redo_stack`, wendet sie vorwärts an, pusht
auf `undo_stack`. Wenn `redo_stack` leer → `can_redo: false`.

### 6.6 Reset

```
POST /api/pi/live-image-chat/reset
Body: { "session_id": "string" }
Response: { "session_id": "string", "image_base64": "string (Original)", "can_undo": false, "can_redo": false }
```

Setzt `current_image` auf `original_image` zurück, ersetzt nach Bestätigung
`outputs/live_edit.fits` durch den Originalzustand, löscht die persistierte
History und leert beide Stacks.

### 6.7 Export

```
POST /api/pi/live-image-chat/export
Body: { "session_id": "string", "format": "png|fits" }
Response: { "path": "runs/<run_id>/outputs/live_chat_export_<timestamp>.<ext>" }
```

Speichert das finale `current_image` als PNG oder FITS in
`runs/<run_id>/outputs/`. HMS-Parameter werden **nicht** verändert —
der Export ist ein eigenständiges Bild, unabhängig von der Pipeline.

### 6.8 History (Persistenz)

```
GET /api/pi/live-image-chat/history?run_id=string
Response: {
  "schema_version": "pi.live-image-chat-history.v1",
  "run_id": "string",
  "chat_history": [...],
  "operation_history": [...],
  "source_artifact": "string",
  "created_at": "ISO timestamp",
  "last_updated": "ISO timestamp"
}
```

Lädt die gespeicherte Chat- und Operations-Historie aus
`.pi_memory/live_image_chat/<run_id>_<hash>.json`. Wird vom Frontend
nach Browser-Refresh aufgerufen, um den Chat-Verlauf wiederherzustellen.

### 6.9 Close Session

```
POST /api/pi/live-image-chat/close
Body: { "session_id": "string" }
Response: { "closed": true }
```

---

## 7. Frontend-Implementation

### 7.1 Live Image Viewer Komponente

Neue Datei: `web_frontend_v3/js/components/live-image-viewer.js`

```
┌────────────────────────────────────────────────────────┐
│  Live Image Viewer              [↶] [↷] [Reset] [×]│
│ ┌──────────────────────────┬─────────────────────────┐ │
│ │                          │  Chat                   │ │
│ │     1:1 Bild             │  ┌───────────────────┐  │ │
│ │   (zoombar, pan)         │  │ PI: Mitteltöne    │  │ │
│ │                          │  │ aufgehellt um 15% │  │ │
│ │   [Generiere Bild…]      │  └───────────────────┘  │ │
│ │   (Overlay während       │  ┌───────────────────┐  │ │
│ │    Backend rendert)      │  │ User: mehr Kontr. │  │ │
│ │                          │  └───────────────────┘  │ │
│ │                          │  ┌───────────────────┐  │ │
│ │                          │  │ PI: Kontrast +10% │  │ │
│ │                          │  └───────────────────┘  │ │
│ │                          │  ┌───┐ Mitteltöne ┌───┐  │ │
│ │                          │  │ − │            │ + │  │ │
│ │                          │  └───┘            └───┘  │ │
│ │                          │                         │ │
│ │                          │  [Eingabefeld    ] [↵]  │ │
│ │                          │  ┌─────────────────┐    │ │
│ │                          │  │ Befehl wählen…  │ ▾  │ │
│ │                          │  └─────────────────┘    │ │
│ │                          │  ┌─────────────────┐    │ │
│ │                          │  │ Hilfetext zum   │    │ │
│ │                          │  │ gewählten Befehl│    │ │
│ │                          │  │                 │    │ │
│ │                          │  └─────────────────┘    │ │
│ └──────────────────────────┴─────────────────────────┘ │
│  [Export PNG] [Export FITS]                             │
└────────────────────────────────────────────────────────┘

Nach +/- Klick (Adjust aktiv):
┌──────────────────────────┬─────────────────────────┐
│                          │  ┌───────────────────┐  │
│     1:1 Bild             │  │ PI: Sättigung +5% │  │
│   (wird aktualisiert)    │  │ (Schritt 3)       │  │
│                          │  └───────────────────┘  │
│                          │  ┌───┐ Sättigung  ┌───┐  │
│                          │  │ − │   (3)     │ + │  │
│                          │  └───┘           └───┘  │
│                          │  [Eingabefeld    ] [↵]  │
└──────────────────────────┴─────────────────────────┘
```

#### Eigenschaften

- **Zoom/Pan:** Mausrad zoomt, Drag verschiebt (wie HMS-Preview)
- **Overlay:** "Generiere Bild…" während Backend 1:1 rendert
- **Chat-Panel:** Rechts neben dem Bild, wie bestehender Run-Chat aber mit
  Live-Bild-Updates
- **Nach jeder AI-Antwort:** Bild wird durch neues JPEG ersetzt (smooth fade)
- **Vorher/Nachher-Vergleich:** Vor jeder erfolgreichen Chat-, Adjust-, Undo-
  oder Redo-Operation bleibt das unmittelbar vorherige Preview im Viewer
  erhalten. Ein Klick auf das Bild oder den Badge links oben schaltet zwischen
  `VORHER` und `AKTUELL`; der Vorher-Zustand wird über den Badge links oben
  gekennzeichnet.
- **+/- Buttons:** Wenn AI `adjustable: true` zurückgibt, erscheinen +/− Buttons
  mit dem Label der Operation (z.B. "Mitteltöne", "Sättigung"). Klick auf +
  verstärkt den Effekt, Klick auf − verringert ihn. Der Zähler zeigt die Anzahl
  der Adjust-Schritte. Neue Chat-Nachfrage versteckt die +/- Buttons.
- **Undo/Redo:** ↶ (Rückgängig) und ↷ (Wiederherstellen) Buttons in der
  Titelzeile. Jede Operation (Chat, Adjust) wird auf den `undo_stack` gepusht.
  Undo wendet die inverse Operation an, Redo stellt sie wieder her. Die Buttons
  sind deaktiviert, wenn der jeweilige Stack leer ist (`can_undo`/`can_redo`
  aus der API-Antwort). Neue Chat-Nachricht leert den `redo_stack`.
- **Reset:** Setzt auf das Originalbild zurück und leert beide Stacks.
- **Export:** Speichert finales Bild als PNG/FITS im Run-Verzeichnis
  (`runs/<run_id>/outputs/`). HMS-Parameter werden nicht verändert.
- **Befehls-Dropdown:** Unten unter dem Eingabefeld ein Dropdown mit
  vorgegebenen Befehlen in verständlicher Sprache (siehe 7.4). Bei Auswahl
  wird der Befehlstext in das Eingabefeld übertragen und sofort gesendet
  (wie Enter im Chat). Ein Hilfetext erscheint unter dem Dropdown.
- **Freitext & Dropdown:** Beide Wege parallel nutzbar — Freitext für
  erfahrene User, Dropdown für Einsteiger.

### 7.2 Integration in Run-Monitor

In `run-monitor.js`:

1. Klick auf Preview-Bild im `run-image-preview` Panel → öffnet `LiveImageViewer`
2. Viewer als Modal/Overlay oder eigener Tab
3. Übergibt `run_id` und `run_dir`

### 7.3 Chat-Verhalten

- **Eingabe:** Freitext, Enter sendet
- **Antwort:** Text + aktualisiertes Bild
- **Iterativ:** Jede Nachricht baut auf vorherigem Bild auf
- **+/- Adjust-Modus:** Bei Kommandos wie "mehr Sättigung", "weniger Kontrast",
  "helle mitteltöne auf" gibt die AI einen Schritt vor und setzt `adjustable: true`.
  Das Frontend zeigt +/− Buttons an:
  - **+ Klick:** Backend wendet `adjust_step` vorwärts an (z.B. Sättigung +5%)
  - **− Klick:** Backend wendet inverse Operation an (z.B. Sättigung −5%)
  - **Zähler:** Zeigt aktuelle Schrittzahl (z.B. "(3)" = 3× + geklickt)
  - **Neue Chat-Nachricht:** Versteckt +/- Buttons, startet neue Operation
  - **Kein Loop, kein Auto-Stop:** User kontrolliert jeden Schritt diskret
  - Backend clamp die Parameter auf gültige Bereiche (kein Überlauf)
- **Verlauf:** Chat-History wird in Session gespeichert und an AI gesendet.
  Um unbegrenztes Wachsen (bis zu 30 Min Session-Laufzeit) und damit
  steigende Latenz/Kosten pro Anfrage zu vermeiden, werden **maximal die
  letzten 10 Chat-Turns** (User+Assistant je 1 Turn) im Prompt an die AI
  mitgeschickt (`PREVIOUS OPERATIONS` in 8.4). Ältere Turns bleiben zwar
  in `chat_history` für die UI-Anzeige erhalten, fließen aber nicht mehr
  in den AI-Kontext ein. Bei Bedarf kann später eine Zusammenfassung
  älterer Turns ergänzt werden — für Phase 1-6 reicht die feste Grenze.

### 7.4 Befehls-Dropdown mit vorgegebenen Chat-Befehlen

Das Dropdown bietet Einsteigern vorgefertigte Befehle in verständlicher
Sprache. Bei Auswahl erscheint ein kurzer Hilfetext. "An KI senden" schickt
den Befehl als Chat-Nachricht ab.

**Wichtig — Phase-Kennzeichnung:** Jeder Eintrag ist mit `[P1]` oder `[P2]`
markiert. `[P1]`-Einträge sind durch Phase-1-Operationen (Abschnitt 4)
gedeckt und funktionieren, sobald Phase 1-4 umgesetzt sind. `[P2]`-Einträge
benötigen Operationen aus der optionalen Phase 2 (`vibrance`,
`color_temperature`, `unpurple`, `fixbanding`, `star_desaturation`,
`dehaze` — siehe Abschnitt 4, "Phase 2 — optional"), die im Zeitplan
(Abschnitt 9) **keinen eigenen Tag hat**. Damit das Dropdown keine
Buttons zeigt, die technisch nicht wirken können, gilt:

- In Phase 4 (Frontend) werden **nur `[P1]`-Einträge** ins Dropdown
  aufgenommen bzw. aktiv geschaltet.
- `[P2]`-Einträge werden entweder (a) erst in einer neuen **Phase 7:
  Erweiterte Bild-Operationen** ausgeliefert (setzt Umsetzung von
  Abschnitt 4 "Phase 2" voraus, siehe Ergänzung in Abschnitt 9), oder
  (b) im Dropdown als deaktiviert mit Tooltip "Bald verfügbar" angezeigt,
  falls sie schon vor Phase 7 sichtbar sein sollen. Default: (a) — nicht
  anzeigen, bis die Backend-Operation existiert.
- Der AI-System-Prompt (Abschnitt 8.4) darf für `[P2]`-Befehle keine
  Operation vorschlagen, die er nicht kennt — er wird stattdessen im
  `summary` erklären, dass diese Anpassung aktuell nicht unterstützt wird
  (`operations: []`, `warnings: ["operation_not_supported"]`).

Die Befehle sind nach Kategorien gruppiert:

#### Helligkeit & Kontrast

| Dropdown-Eintrag | Phase | Gesendeter Chat-Befehl | Hilfetext |
|------------------|-------|------------------------|-----------|
| Helligkeit erhöhen | P1 | "helle das Bild auf" | Hebt die Mitteltöne an — das Bild wird insgesamt heller. Mit +/- feinjustieren. |
| Helligkeit verringern | P1 | "dunkle das Bild ab" | Senkt die Mitteltöne ab — das Bild wird insgesamt dunkler. Mit +/- feinjustieren. |
| Schatten aufhellen | P1 | "helle die Schatten auf" | Bringt Details in sehr dunklen Bereichen hervor, ohne helle Bereiche zu verändern. (nutzt `brightness.shadows`) |
| Spitzlichter zurückholen | P1 | "reduziere die Spitzlichter" | Dämpft überbelichtete Bereiche (z.B. Sternkerne), bringt Details zurück. (nutzt `brightness.highlights`) |
| Kontrast erhöhen | P1 | "erhöhe den Kontrast" | Verstärkt den Unterschied zwischen hell und dunkel — das Bild wirkt knackiger. |
| Kontrast verringern | P1 | "verringere den Kontrast" | Macht das Bild weicher, reduziert harte Übergänge. |
| Lokaler Kontrast (Details) | P1 | "verstärke lokale Details" | Bringt feine Strukturen in Nebeln und Galaxien hervor. (nutzt `clahe`, kein Undo-Feintuning per +/-, siehe 8.4) |
| Schwarzwert anheben | P1 | "hebe den Schwarzwert an" | Entfernt restliches Hintergrundrauschen — der Himmel wird tiefschwarz. |

#### Farbe

| Dropdown-Eintrag | Phase | Gesendeter Chat-Befehl | Hilfetext |
|------------------|-------|------------------------|-----------|
| Farbsättigung erhöhen | P1 | "erhöhe die Farbsättigung" | Intensiviert die Farben — Sterne und Nebel werden farbenfroher. Mit +/- feinjustieren. |
| Farbsättigung verringern | P1 | "verringere die Farbsättigung" | Reduziert die Farben — das Bild wird dezenter. |
| Lebendigkeit erhöhen | **P2** | "mache die Farben lebendiger" | Intelligente Sättigung — bereits kräftige Farben werden weniger verstärkt als blasse. (benötigt `vibrance`, Phase 2) |
| Farbtemperatur wärmer | **P2** | "mache das Bild wärmer" | Verschiebt die Farben Richtung Rot/Gelb. (benötigt `color_temperature`, Phase 2) |
| Farbtemperatur kälter | **P2** | "mache das Bild kälter" | Verschiebt die Farben Richtung Blau. (benötigt `color_temperature`, Phase 2) |
| Grüne Farbnebel entfernen | P1 | "entferne grüne Farbnebel" | Entfernt störenden Grünstich, der bei One-Shot-Color-Kameras oft auftritt. |
| Lila Farbsäume entfernen | **P2** | "entferne lila Farbsäume" | Reduziert lila/blaue Farbsäume an hellen Sternen. (benötigt `unpurple`, Phase 2) |

#### Rauschen & Schärfe

| Dropdown-Eintrag | Phase | Gesendeter Chat-Befehl | Hilfetext |
|------------------|-------|------------------------|-----------|
| Rauschen unterdrücken | P1 | "unterdrücke das Rauschen" | Reduziert Bildrauschen — das Bild wird glatter, kann aber an Schärfe verlieren. |
| Rauschen detailerhaltend entfernen | P1 | "entferne Rauschen ohne Detailverlust" | Sanfte Rauschunterdrückung, die feine Strukturen erhält. (nutzt `bilateral`) |
| Farbrauschen unterdrücken | P1 | "unterdrücke das Farbrauschen" | Entfernt nur buntes Rauschen, lässt Helligkeitsdetails intakt. (nutzt `denoise` mit `luminance=true`, siehe 8.2-Tabelle) |
| Bild schärfen | P1 | "schärfe das Bild" | Erhöht die Detailwahrnehmung — Sterne und Strukturen wirken klarer. Mit +/- feinjustieren. |
| Streifenartefakte entfernen | **P2** | "entferne Streifen im Bild" | Beseitigt horizontale/vertikale Streifen. (benötigt `fixbanding`, Phase 2) |
| Übersteuerte Sterne entschärfen | **P2** | "entschärfe übersteuerte Sterne" | Reduziert die Farbe in ausgebrannten Sternkernen. (benötigt `star_desaturation`, Phase 2) |

#### Sonstiges

| Dropdown-Eintrag | Phase | Gesendeter Chat-Befehl | Hilfetext |
|------------------|-------|------------------------|-----------|
| Neblige Bereiche klären | **P2** | "klare neblige Bereiche auf" | Erhöht den Kontrast in diffusen, nebligen Regionen. (benötigt `dehaze`, Phase 2) |
| Negativ anzeigen | P1 | "zeige das Bild als Negativ" | Dreht Helligkeit um — nützlich um schwache Strukturen in hellen Bereichen zu erkennen. (nutzt neue `invert`-Operation, siehe Abschnitt 4) |
| Bild zurücksetzen | P1 | "setze das Bild zurück" | Setzt alle Änderungen zurück auf das Originalbild. |

Zählung: **17 × [P1]** (in Phase 4 aktiv) + **7 × [P2]** (erst ab Phase 7) = 24 Einträge gesamt.

#### Implementierung

- Das Dropdown wird aus einer statischen JSON-Datei oder inline im
  `live-image-viewer.js` gespeist — keine Backend-Abfrage nötig
- Struktur pro Eintrag (neues Feld `phase`):
  ```json
  {
    "category": "Helligkeit & Kontrast",
    "label": "Helligkeit erhöhen",
    "command": "helle das Bild auf",
    "help": "Hebt die Mitteltöne an — das Bild wird insgesamt heller. Mit +/- feinjustieren.",
    "phase": 1
  }
  ```
- Das Frontend filtert beim Rendern des Dropdowns standardmäßig auf
  `phase <= FEATURE_PHASE` (Konstante, in Phase 4 auf `1` gesetzt, in
  Phase 7 auf `2` erhöht) — so tauchen `[P2]`-Einträge erst auf, wenn die
  zugehörigen Backend-Operationen existieren.
- Bei Auswahl: `command` wird in das Eingabefeld übertragen und sofort
  gesendet (entspricht Enter im Chat). Hilfetext erscheint in einer Info-Box
  unter dem Dropdown als Erinnerung, was gerade passiert.
- i18n: `label` und `help` werden in `de.json` / `en.json` übersetzt;
  `command` bleibt sprachneutral (AI versteht beide Sprachen)

---

## 8. Backend-Implementation

### 8.1 1:1 JPEG-Generierung

Neue Funktion in `runs_routes.cpp` oder `pi_routes.cpp`:

```cpp
// Rendert FITS als 1:1 JPEG (kein Downscale)
std::vector<unsigned char> render_fits_full_jpeg(
    const fs::path& path, int quality = 92) {
    // Wie render_fits_preview_png, aber:
    // - Kein max_edge Downscale
    // - JPEG statt PNG (kleinere Datei für 1:1)
    // - quality=92 als guter Kompromiss
    // - Gamma 0.6 wie bestehend
}
```

### 8.2 Bild-Operationen anwenden

Neue Datei: `web_backend_cpp/src/services/pi/pi_image_ops.hpp`

```cpp
namespace tile_compile::pi {

struct ImageOpResult {
    cv::Mat image;
    std::string error;
    bool success;
};

ImageOpResult apply_image_op(const cv::Mat& input,
                             const nlohmann::json& op);

cv::Mat apply_brightness(const cv::Mat& img, double midtones,
                         double shadows, double highlights);
cv::Mat apply_contrast(const cv::Mat& img, double amount);
cv::Mat apply_saturation(const cv::Mat& img, double amount);
cv::Mat apply_sharpen(const cv::Mat& img, double amount, double radius);
cv::Mat apply_denoise(const cv::Mat& img, double strength, bool luminance);

// Inverse Operation für Undo berechnen
nlohmann::json invert_op(const nlohmann::json& op);
// brightness: midtones/shadows/highlights negieren (linear, exakt invertierbar)
// contrast: amount negieren (Sigmoid-Näherung, nicht exakt aber ausreichend)
// saturation: amount negieren (HSV-Skalierung, exakt invertierbar)
// sharpen: amount negieren (nur Näherung — Unsharp Mask ist keine exakte
//   Inverse von "Blur", daher Rundungsfehler bei mehrfachem Undo/Redo möglich)
// clahe, bilateral: NICHT invertierbar (nichtlineare Tonwert-Redistribution
//   bzw. Weichzeichnung — "negativer Radius/Cliplimit" ergibt keine Umkehrung)
//   → bei Undo wird Snapshot verwendet, siehe invert_op-Fallback unten
// denoise, threshold, crop, invert: nicht (sinnvoll) invertierbar
//   → bei Undo wird Snapshot verwendet

} // namespace
```

### 8.3 Session-Store

Neue Datei: `web_backend_cpp/src/services/pi/pi_live_image_session.hpp`

```cpp
namespace tile_compile::pi {

struct LiveImageSession {
    std::string session_id;
    std::string run_id;
    cv::Mat original_image;
    cv::Mat current_image;
    nlohmann::json operation_history;
    std::vector<nlohmann::json> undo_stack;
    std::vector<nlohmann::json> redo_stack;
    nlohmann::json chat_history;
    nlohmann::json last_adjust_step;
    int adjust_count = 0;
    std::string created_at;
    std::string last_accessed;
};

class LiveImageSessionStore {
public:
    std::string create(const std::string& run_id, cv::Mat image);
    // Kein rohes get() — Zugriff nur über with_session()-Callback unter
    // Lock (Details und Begründung in Abschnitt 8.3)
    void close(const std::string& session_id);
    void evict_expired(int max_age_seconds = 1800, size_t max_sessions = 5);

    // Undo/Redo — gibt neues JPEG zurück
    struct UndoRedoResult {
        cv::Mat image;
        std::string summary;
        bool can_undo;
        bool can_redo;
        int count;
    };
    UndoRedoResult undo(const std::string& session_id);
    UndoRedoResult redo(const std::string& session_id);
    void reset(const std::string& session_id);
};

} // namespace
```

### 8.4 AI-Prompt für Live Image Chat

```
You are PI Live Image Editor for tile_compile astrophotography software.
The user sees a 1:1 preview of their stacked astronomical image and gives
natural language instructions for image adjustments.

You must return a JSON object with image operations. Do NOT describe what
you would do — return concrete operations with exact parameters.

Available operations:
- brightness: { midtones: -1..1, shadows: -1..1, highlights: -1..1 }
- contrast: { amount: -1..1 }
- saturation: { amount: -1..1 }
- sharpen: { amount: 0..1, radius: 0.5..5 }
- denoise: { strength: 0..1, luminance: bool }
- rmgreen: { strength: 0..1 }
- clahe: { cliplimit: 1..10, tilesize: 8..64 }
- bilateral: { d: 3..15, sigma_color: 10..150, sigma_space: 10..150 }
- threshold: { black_point: 0..1, white_point: 0..1 }
- invert: {}
- reset: {}

Note: clahe, bilateral, denoise and threshold cannot be exactly undone by
negating a parameter (they are non-linear or lossy). Do NOT set
adjustable=true with a negatable adjust_step for these — if finer control
makes sense, ask the user to issue a new command instead of relying on the
+/- buttons for these types.

For commands like "mehr X", "weniger X", "helle X auf", "erhöhe X" — these are
adjustable operations. Set adjustable=true and provide an adjust_step with a
moderate increment (e.g. +5%). The user will click + or - buttons in the UI
to fine-tune the effect. You do not need to loop or detect "stop".

Return exactly:
{
  "schema_version": "pi.live-image-chat.v1",
  "summary": "string — was wurde gemacht (auf Deutsch)",
  "operations": [...],
  "adjustable": false,
  "adjust_step": { "type": "string", "params": {...}, "label": "string" },
  "warnings": []
}

USER MESSAGE: <message>
CURRENT IMAGE ANALYSIS: <AI vision model description of the image>
PREVIOUS OPERATIONS: <history>
```

---

## 9. Detaillierter Implementierungsplan

Jeder Schritt listet die zu erstellenden/modifizierenden Dateien, die
Funktionssignaturen, Abhängigkeiten und den Verifikationsbefehl auf.

---

### Phase 1: Backend — Bild-Operationen (Tag 1-2)

#### Schritt 1.1: `pi_image_ops.hpp` / `pi_image_ops.cpp` ✅

**Neue Dateien:**
- `web_backend_cpp/src/services/pi/pi_image_ops.hpp`
- `web_backend_cpp/src/services/pi/pi_image_ops.cpp`

**CMakeLists.txt:** `pi_image_ops.cpp` zu `tile_compile_web_backend`-Sources hinzufügen.

**Header (`pi_image_ops.hpp`):**

```cpp
#pragma once
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>

namespace tile_compile::pi {

struct ImageOpResult {
    cv::Mat image;
    std::string error;
    bool success = false;
};

// Dispatch: liest "type" und "params" aus op, ruft die passende Funktion
ImageOpResult apply_image_op(const cv::Mat& input, const nlohmann::json& op);

// Inverse Operation für Undo: negiert Parameter, wo möglich.
// Nicht-invertierbare Ops (denoise, crop) → gibt {type:"noop"} zurück,
// der Session-Store verwendet dann einen Snapshot.
nlohmann::json invert_op(const nlohmann::json& op);

// Phase 1 Operationen
cv::Mat apply_brightness(const cv::Mat& img, double midtones,
                         double shadows, double highlights);
cv::Mat apply_contrast(const cv::Mat& img, double amount);
cv::Mat apply_saturation(const cv::Mat& img, double amount);
cv::Mat apply_sharpen(const cv::Mat& img, double amount, double radius);
cv::Mat apply_denoise(const cv::Mat& img, double strength, bool luminance);
cv::Mat apply_rmgreen(const cv::Mat& img, double strength);
cv::Mat apply_clahe(const cv::Mat& img, double cliplimit, int tilesize);
cv::Mat apply_bilateral(const cv::Mat& img, int d,
                        double sigma_color, double sigma_space);
cv::Mat apply_threshold(const cv::Mat& img, double black_point,
                        double white_point);
cv::Mat apply_invert(const cv::Mat& img);
cv::Mat apply_crop(const cv::Mat& img, int x, int y, int w, int h);

// Hilfsfunktionen
double clamp_param(double val, double lo, double hi);
nlohmann::json validate_op(const nlohmann::json& op);

} // namespace tile_compile::pi
```

**Implementation-Details pro Operation:**

| Funktion | Algorithmus | OpenCV-Aufrufe |
|----------|-------------|----------------|
| `apply_brightness` | LUT mit 3-Zonen-Gamma: shadows (0-25%), midtones (25-75%), highlights (75-100%) | `cv::LUT()` pro Kanal mit vorberechneter 256-Eintrag-Tabelle |
| `apply_contrast` | Sigmoid: `v' = 1/(1+exp(-k*(v-0.5)))`, `k = 1 + amount*4` | Pixelweise auf [0,1]-normalisiertem Bild, dann zurück |
| `apply_saturation` | HSV-Konvertierung, S-Kanal skalieren um `(1+amount)` | `cv::cvtColor(BGR2HSV)`, S-Kanal `cv::multiply`, `cv::cvtColor(HSV2BGR)` |
| `apply_sharpen` | Unsharp Mask: `out = img + amount * (img - blur(img))` | `cv::GaussianBlur()`, `cv::addWeighted(img, 1+amount, blur, -amount, 0)` |
| `apply_denoise` | Non-Local Means. Bei `luminance=true`: Konvertierung nach YCrCb, Denoise **nur** auf Y-Kanal (`cv::fastNlMeansDenoising`), Cr/Cb unverändert. Bei `luminance=false`: Denoise auf allen 3 Kanälen. | `luminance=true`: `cv::cvtColor(BGR2YCrCb)` → `cv::fastNlMeansDenoising(Y, strength*10, 7, 21)` → Merge → `cv::cvtColor(YCrCb2BGR)`. `luminance=false`: `cv::fastNlMeansDenoisingColored(img, out, strength*10, strength*10, 7, 21)` |
| `apply_rmgreen` | SCNR: `G_new = G - min(G, R, B) * strength` | Kanal-Split, `cv::min`, `cv::subtract`, Merge |
| `apply_clahe` | CLAHE pro BGR-Kanal | `cv::createCLAHE(cliplimit, Size(tilesize, tilesize))`, pro Kanal `apply()` |
| `apply_bilateral` | Bilateraler Filter | `cv::bilateralFilter(img, out, d, sigma_color, sigma_space)` |
| `apply_threshold` | Black/White-Point Clipping via LUT | 256-Eintrag-LUT: `< bp → 0`, `> wp → 255`, dazwischen linear |
| `apply_invert` | Negativ | `cv::bitwise_not(img, out)` |
| `apply_crop` | ROI-Ausschnitt | `img(cv::Rect(x, y, w, h)).clone()` |

**`invert_op`:**

```cpp
nlohmann::json invert_op(const nlohmann::json& op) {
    auto inv = op;
    auto& p = inv["params"];
    const std::string type = op["type"].get<std::string>();
    if (type == "brightness") {
        p["midtones"] = -p["midtones"].get<double>();
        p["shadows"] = -p["shadows"].get<double>();
        p["highlights"] = -p["highlights"].get<double>();
    } else if (type == "contrast" || type == "saturation"
               || type == "sharpen" || type == "rmgreen") {
        // Diese Ops haben genau einen "amount"-Parameter und sind
        // (näherungsweise) linear invertierbar durch Negation.
        p["amount"] = -p["amount"].get<double>();
    } else {
        // clahe, bilateral, denoise, threshold, crop, invert, reset:
        // kein "amount"-Parameter bzw. mathematisch nicht durch
        // Parameter-Negation invertierbar → Snapshot-basiertes Undo
        // (siehe LiveImageSessionStore::undo(), Abschnitt 8.3)
        return {{"type", "noop"}};
    }
    return inv;
}
```

**Tests:** Neue Datei `web_backend_cpp/tests/test_pi_image_ops.cpp`
- `brightness` mit `midtones=0.5` → Pixel im mittleren Bereich heller
- `contrast` mit `amount=0.5` → Standardabweichung erhöht
- `saturation` mit `amount=1.0` → S-Kanal in HSV verdoppelt
- `sharpen` mit `amount=0.5, radius=2` → Hochfrequenzanteil erhöht
- `denoise` mit `strength=0.5` → Varianz reduziert
- `rmgreen` mit `strength=1.0` → G-Kanal ≤ min(R,B)
- `clahe` mit `cliplimit=3, tilesize=8` → lokaler Kontrast erhöht
- `bilateral` mit `d=9` → Rauschen reduziert, Kanten erhalten
- `threshold` mit `bp=0.1, wp=0.9` → Pixel <0.1 werden 0, >0.9 werden 255
- `invert` → Pixel `255 - v` je Kanal
- `crop` mit `x=10,y=10,w=100,h=100` → Output-Größe 100×100
- `invert_op` für brightness/contrast/saturation/sharpen/rmgreen → Parameter negiert
- `invert_op` für clahe/bilateral/denoise/threshold/crop/invert → `{"type":"noop"}` (Snapshot-Undo)
- `validate_op` mit out-of-range Parametern → Fehler
- `apply_image_op` mit unbekanntem Type → Fehler

**Verifikation:**
```bash
cmake --build build --target tile_compile_web_backend tests -j2 > /tmp/out_build.txt 2>&1
./build/test_pi_image_ops > /tmp/out_test_ops.txt 2>&1
sed -n '1,80p' /tmp/out_test_ops.txt
```

#### Schritt 1.2: `render_fits_full_jpeg()` ✅

**Modifizierte Datei:** `web_backend_cpp/src/routes/runs_routes.cpp` — neue Funktion, basiert auf `render_fits_preview_png()`

```cpp
static std::vector<unsigned char> render_fits_full_jpeg(
    const fs::path& path, int quality = 92) {
    // Gleiche Logik wie render_fits_preview_png(), aber:
    // 1. Kein max_edge Downscale (edge > max_edge Block entfernen)
    // 2. cv::imencode(".jpg", ...) statt ".png"
    // 3. Für Bilder > 4000px: quality auf 85 reduzieren
    // 4. Gamma 0.6 wie bestehend
    int effective_quality = quality;
    if (std::max(r.rows, r.cols) > 4000) effective_quality = 85;
    std::vector<unsigned char> jpeg;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, effective_quality};
    if (!cv::imencode(".jpg", out, jpeg, params))
        throw std::runtime_error("JPEG encoding failed");
    return jpeg;
}
```

**Tests:** In `test_pi_routes.cpp` — FITS laden → JPEG generieren → Header/Größe prüfen.

### Phase 2: Backend — Session-Store & API (Tag 3-4)

#### Schritt 2.1: `pi_live_image_session.hpp` / `pi_live_image_session.cpp` ✅

**Neue Dateien:**
- `web_backend_cpp/src/services/pi/pi_live_image_session.hpp`
- `web_backend_cpp/src/services/pi/pi_live_image_session.cpp`

**CMakeLists.txt:** `pi_live_image_session.cpp` zu Sources hinzufügen.

**Header:**

```cpp
#pragma once
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <chrono>
#include "pi_image_ops.hpp"

namespace tile_compile::pi {

struct LiveImageSession {
    std::string session_id;
    std::string run_id;
    cv::Mat original_image;
    cv::Mat current_image;
    std::vector<nlohmann::json> undo_stack;
    std::vector<nlohmann::json> redo_stack;
    nlohmann::json chat_history;
    nlohmann::json last_adjust_step;
    int adjust_count = 0;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_accessed;
};

struct UndoRedoResult {
    cv::Mat image;
    std::string summary;
    bool can_undo = false;
    bool can_redo = false;
    int count = 0;
};

class LiveImageSessionStore {
public:
    std::string create(const std::string& run_id, cv::Mat image);
    void close(const std::string& session_id);
    void evict_expired(int max_age_seconds = 1800, size_t max_sessions = 5);

    // Alle Mutationen laufen ausschließlich über with_session() — es gibt
    // KEINEN öffentlichen get(), der einen rohen LiveImageSession* nach
    // außen gibt. Grund: ein roher Pointer könnte zwischen Abruf und
    // Verwendung durch evict_expired() (LRU) auf einem anderen Thread
    // invalidiert werden (use-after-free). with_session() hält den
    // Store-Lock für die gesamte Dauer des Callbacks und garantiert damit,
    // dass die Session währenddessen nicht evicted wird.
    // callback gibt true zurück, wenn ein Treffer gefunden wurde.
    bool with_session(const std::string& session_id,
                      const std::function<void(LiveImageSession&)>& fn);

    ImageOpResult apply_operation(const std::string& session_id,
                                  const nlohmann::json& op);
    ImageOpResult apply_adjust(const std::string& session_id,
                               const std::string& direction);
    UndoRedoResult undo(const std::string& session_id);
    UndoRedoResult redo(const std::string& session_id);
    cv::Mat reset(const std::string& session_id);

    void set_adjust_step(const std::string& session_id,
                         const nlohmann::json& step);
    void append_chat(const std::string& session_id,
                     const std::string& role, const std::string& content,
                     const nlohmann::json& operations = nullptr);
    nlohmann::json get_chat_history(const std::string& session_id);
    nlohmann::json get_operation_history(const std::string& session_id);

private:
    std::mutex m_mutex;
    std::vector<std::unique_ptr<LiveImageSession>> m_sessions;
    std::string generate_uuid() const;
};

} // namespace tile_compile::pi
```

**Locking-Regel (wichtig):** `m_mutex` schützt nur den `m_sessions`-Vektor
und die Feldzugriffe innerhalb einer `LiveImageSession` — er wird **nie**
während eines Netzwerk-Calls zum AI-Sidecar gehalten. Der Ablauf in
`pi_routes.cpp` für den Chat-Endpunkt ist daher zweistufig:
1. Lock kurz halten, um Bild + Chat-History für den Sidecar-Request zu
   kopieren (`cv::Mat::clone()` bzw. JSON-Kopie), dann Lock freigeben.
2. HTTP-Call zum Sidecar **außerhalb** des Locks ausführen.
3. Lock erneut kurz halten, um `apply_operation()` mit dem Ergebnis
   aufzurufen (dabei erneut mit `evict_expired`/Undo-Logik konsistent).
Ohne diese Trennung würde ein einzelner langsamer/hängender
Sidecar-Request alle anderen Live-Image-Chat-Sessions blockieren, da
`m_mutex` global für den gesamten Store gilt (kein Per-Session-Lock).

**Implementation-Details:**
- `generate_uuid()`: `std::random_device` + `std::mt19937` + Hex-Formatierung
- `apply_operation()`: `apply_image_op(current_image, op)`, bei Erfolg
  `undo_stack.push_back(op)`, `redo_stack.clear()`, `current_image = result.image`
- `apply_adjust()`: bei "increase" `op = last_adjust_step`; bei "decrease"
  `op = invert_op(last_adjust_step)`; dann `apply_operation()`, `adjust_count++/--`
- `apply_operation()` (Ergänzung): Bevor eine nicht invertierbare Operation
  (clahe, bilateral, denoise, threshold, crop, invert — erkennbar an
  `invert_op(op).type == "noop"`) angewendet wird, klont der Store
  `current_image` **vor** der Anwendung und hängt es als `"snapshot"`-Feld
  an den Eintrag, der auf `undo_stack` gepusht wird. Bei invertierbaren
  Operationen wird kein Snapshot erzeugt.
- `undo()`: `entry = undo_stack.back()`. Falls `entry` ein `"snapshot"`-Feld
  hat: `current_image = entry.snapshot.clone()` (kein `invert_op` nötig).
  Sonst: `inv = invert_op(entry)`, anwenden auf `current_image`.
  Anschließend `redo_stack.push_back(entry)`, `undo_stack.pop_back()`.
- `reset()`: `current_image = original_image.clone()`, beide Stacks leeren
- `evict_expired()`: nach `last_accessed` sortieren, älteste entfernen.
  Zusätzlich: Snapshot-Einträge im `undo_stack` einer Session werden auf
  die letzten 10 begrenzt (älteste zuerst verworfen), um den in Abschnitt 5
  beschriebenen Speicher-Worst-Case einzuhalten.

**Tests:** Neue Datei `web_backend_cpp/tests/test_pi_live_image_session.cpp`
- `create` → Session existiert, `current_image == original_image`
- `apply_operation(brightness)` → `current_image` verändert, `undo_stack.size()==1`
- `undo` → `current_image` zurück, `redo_stack.size()==1`
- `redo` → `current_image` wieder verändert, `undo_stack.size()==1`
- `reset` → `current_image == original_image`, beide Stacks leer
- `apply_adjust("increase")` → `adjust_count == 1`
- `apply_adjust("decrease")` → `adjust_count == 0`
- `evict_expired` mit alter Session → Session entfernt
- `evict_expired` mit >5 Sessions → älteste entfernt
- Thread-Safety: 2 Threads gleichzeitig `apply_operation` → keine Race

**Verifikation:**
```bash
cmake --build build --target tests -j2 > /tmp/out_build.txt 2>&1
./build/test_pi_live_image_session > /tmp/out_test_session.txt 2>&1
sed -n '1,80p' /tmp/out_test_session.txt
```

#### Schritt 2.2: API-Endpunkte in `pi_routes.cpp` ✅

**Modifizierte Datei:** `web_backend_cpp/src/routes/pi_routes.cpp`

**Neue Includes:**
```cpp
#include "services/pi/pi_live_image_session.hpp"
#include "services/pi/pi_image_ops.hpp"
```

**Neuer Member:** `std::unique_ptr<tile_compile::pi::LiveImageSessionStore> live_image_store;`

**9 neue Routen:**

| Route | Methode | Beschreibung |
|-------|---------|-------------|
| `/api/pi/live-image-chat/create` | POST | Findet Output-FITS, rendert 1:1 JPEG, erzeugt Session (oder resume aus History) |
| `/api/pi/live-image-chat` | POST | Sendet Nachricht an AI Sidecar, wendet Operationen an, persistiert History |
| `/api/pi/live-image-chat/adjust` | POST | +/- Button: wendet `last_adjust_step` vorwärts/invers an, persistiert History |
| `/api/pi/live-image-chat/undo` | POST | Pop undo_stack, inverse Operation, push redo_stack, persistiert History |
| `/api/pi/live-image-chat/redo` | POST | Pop redo_stack, vorwärts, push undo_stack, persistiert History |
| `/api/pi/live-image-chat/reset` | POST | current_image = original_image, Stacks leeren, persistiert leere History |
| `/api/pi/live-image-chat/export` | POST | PNG/FITS in `runs/<id>/outputs/` schreiben (HMS unverändert) |
| `/api/pi/live-image-chat/history` | GET | Lädt Chat- + Operations-Historie aus `.pi_memory` |
| `/api/pi/live-image-chat/close` | POST | Session aus RAM entfernen (History bleibt in `.pi_memory`) |

**Chat-Endpunkt AI Sidecar Call:**
```cpp
// POST http://127.0.0.1:3001/live-image-chat
// Body: { prompt: system_prompt + message, image_base64, image_mime: "image/jpeg" }
// Response: { operations, summary, adjustable, adjust_step, warnings }
```

**Lokaler Fallback (kein Sidecar):**
```cpp
nlohmann::json fallback_parse_message(const std::string& msg) {
    // Keyword-Erkennung:
    // "heller"/"aufhellen" → brightness midtones +0.15, adjustable=true
    // "dunkler" → brightness midtones -0.15
    // "kontrast" + "mehr" → contrast +0.1
    // "sättigung"/"farbe" + "mehr" → saturation +0.1
    // "schärfe" → sharpen amount=0.3
    // "rauschen" → denoise strength=0.5
    // "grün" → rmgreen strength=0.5
    // "details"/"lokal" → clahe cliplimit=3
    // "zurück"/"reset" → reset
}
```

**Tests:** In `test_pi_routes.cpp` ergänzen:
- `create` mit gültiger run_id → 200, session_id nicht leer
- `create` mit ungültiger run_id → 404
- `chat` mit gültiger session_id → 200, operations Array
- `chat` mit ungültiger session_id → 404
- `adjust` mit "increase"/"decrease" → 200, adjust_count korrekt
- `undo`/`redo` → 200, can_undo/can_redo korrekt
- `reset` → 200, jpeg_base64 == original
- `export` mit format=png → 200, Datei existiert
- `close` → 200, Folge-`chat`/`adjust`/`undo` auf dieselbe `session_id` → 404

**Verifikation:**
```bash
cmake --build build --target tile_compile_web_backend tests -j2 > /tmp/out_build.txt 2>&1
./build/test_pi_routes > /tmp/out_test_routes.txt 2>&1
sed -n '1,120p' /tmp/out_test_routes.txt
```

### Phase 3: AI Sidecar — Live Image Endpoint (Tag 5)

#### Schritt 3.1: `liveImageChatService.ts` ✅

**Neue Datei:** `agent_service/src/services/liveImageChatService.ts`

**Struktur:** Wie `runChatService.ts`, aber mit eigenem System-Prompt:
- System-Prompt aus Abschnitt 8.4 (erweitert um alle Phase-1-Operationen)
- Bild als base64 JPEG senden
- `parseJsonObject()` auf Response → `{operations, summary, adjustable, adjust_step, warnings}`
- `_meta` mit model, provider, duration_ms hinzufügen

#### Schritt 3.2: Route in `server.ts` ✅

**Modifizierte Datei:** `agent_service/src/server.ts`

Nach dem `/run-chat` Block (Zeile ~115) einfügen:
```typescript
if (req.method === "POST" && url.pathname === "/live-image-chat") {
  const body = await readJson(req);
  appendTrafficLog(`POST /live-image-chat ${JSON.stringify({ ...body, image_base64: body?.image_base64 ? "<image>" : undefined }).substring(0, 10000)}`);
  const service = new LiveImageChatService(config.agent, modelService);
  const result = await service.ask(body);
  appendTrafficLog(`POST /live-image-chat response ${JSON.stringify(result).substring(0, 10000)}`);
  sendJson(res, 200, result);
  return;
}
```

**Verifikation:**
```bash
curl -X POST http://127.0.0.1:3001/live-image-chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"helle das Bild auf","image_base64":""}' > /tmp/out_sidecar.txt 2>&1
sed -n '1,40p' /tmp/out_sidecar.txt
```

### Phase 4: Frontend — Live Image Viewer (Tag 6-7)

#### Schritt 4.1: `live-image-viewer.js` ✅

**Neue Datei:** `web_frontend_v3/js/components/live-image-viewer.js`

**State:**
- `sessionId`, `currentImageSrc`, `adjustActive`, `adjustLabel`, `adjustCount`
- `canUndo`, `canRedo`, `chatHistory[]`, `isLoading`

**DOM-Elemente:**
- `overlay` (fullscreen modal, dark backdrop)
- `imageContainer` (zoom/pan) → `imgEl` (`<img>` für 1:1 JPEG)
- `loadingOverlay` ("Generiere Bild…" Spinner)
- `chatPanel` (Verlauf + Eingabe + Dropdown + Hilfebox)
- `adjustControls` (− Label (+) Zähler)
- `undoBtn`, `redoBtn`, `resetBtn`, `closeBtn` (Titelzeile)
- `exportPngBtn`, `exportFitsBtn` (Footer)

**Funktionen:**

| Funktion | API-Call | Beschreibung |
|----------|----------|-------------|
| `open()` | POST `/create` | Session erzeugen (oder resume aus History), erstes Bild laden, bei `resumed:true` Chat-Verlauf wiederherstellen |
| `sendChat(msg)` | POST `/live-image-chat` | Nachricht senden, Bild aktualisieren |
| `doAdjust(dir)` | POST `/adjust` | +/- Button, Bild aktualisieren |
| `doUndo()` | POST `/undo` | Undo, can_undo/can_redo aktualisieren |
| `doRedo()` | POST `/redo` | Redo, can_undo/can_redo aktualisieren |
| `doReset()` | POST `/reset` | Originalbild wiederherstellen |
| `doExport(fmt)` | POST `/export` | PNG/FITS speichern (HMS unverändert) |
| `doClose()` | POST `/close` | Session schließen, Viewer entfernen |
| `updateImage(b64)` | — | `imgEl.src = "data:image/jpeg;base64,..."` + fade |
| `onSelectPreset(cmd, help)` | — | Hilfetext anzeigen, `sendChat(cmd)` sofort |
| `restoreChatHistory(history)` | — | Chat-Blasen aus `chat_history[]` aufbauen |

**Zoom/Pan:** Mausrad zoomt (`scale *= 1.1/0.9`, clamp 0.1..10), Drag verschiebt.
Wie `hms-preview.js` Zoom-Logik.

**Bild-Update mit Fade:**
```javascript
function updateImage(jpegBase64) {
  imgEl.style.opacity = "0";
  setTimeout(() => {
    imgEl.src = `data:image/jpeg;base64,${jpegBase64}`;
    imgEl.style.opacity = "1";
  }, 150);
}
```

**Dropdown-Logik:**
```javascript
function onSelectPreset(command, help) {
  helpBox.textContent = t(help);
  helpBox.style.display = "block";
  sendChat(command);  // sofort senden, kein extra Button
}
```

**Preset-Befehle:** Statisches Array mit 24 Einträgen (siehe Abschnitt 7.4).
Struktur: `{category, label, command, help}` — i18n-Keys für label/help.

#### Schritt 4.2: Integration in `run-monitor.js` ✅

**Modifizierte Datei:** `web_frontend_v3/js/pages/run-monitor.js`

```javascript
import { createLiveImageViewer } from "../components/live-image-viewer.js";

// Klick auf Preview-Bild:
previewImg.addEventListener("click", () => {
  const viewer = createLiveImageViewer(runId, runDir, () => {
    overlay.remove();  // onClose callback
  });
  viewer.open();
});
```

#### Schritt 4.3: API-Endpunkte in `endpoints.js` ✅

**Modifizierte Datei:** `web_frontend_v3/js/api/endpoints.js`

```javascript
liveImageChat: {
  create: () => "/api/pi/live-image-chat/create",
  chat: () => "/api/pi/live-image-chat",
  adjust: () => "/api/pi/live-image-chat/adjust",
  undo: () => "/api/pi/live-image-chat/undo",
  redo: () => "/api/pi/live-image-chat/redo",
  reset: () => "/api/pi/live-image-chat/reset",
  export: () => "/api/pi/live-image-chat/export",
  history: (runId) => `/api/pi/live-image-chat/history?run_id=${encodeURIComponent(runId)}`,
  close: () => "/api/pi/live-image-chat/close",
},
```

#### Schritt 4.4: CSS ✅

**Modifizierte Datei:** `web_frontend_v3/css/style.css`

Neue Klassen:
- `.live-image-viewer-overlay` — fullscreen modal, dark backdrop
- `.live-image-viewer` — flexbox: image left, chat right
- `.live-image-viewer__image` — zoomable, pannable
- `.live-image-viewer__loading` — overlay with spinner
- `.live-image-viewer__chat` — chat panel right side
- `.live-image-viewer__adjust` — +/− button row
- `.live-image-viewer__dropdown` — preset command dropdown
- `.live-image-viewer__help` — help text box
- `.live-image-viewer__toolbar` — undo/redo/reset/close in title bar
- Responsive: mobile → chat below image (column layout)
- Bestehende CSS-Tokens verwenden, keine inline styles

#### Schritt 4.5: i18n ✅

**Modifizierte Dateien:** `web_frontend_v3/i18n/de.json`, `en.json`

Neue Keys (Auswahl):
- `liveImage.title`, `liveImage.loading`, `liveImage.chatPlaceholder`
- `liveImage.dropdownPlaceholder`, `liveImage.undo`, `liveImage.redo`
- `liveImage.reset`, `liveImage.exportPng`, `liveImage.exportFits`
- `liveImage.noImage`, `liveImage.sessionExpired`, `liveImage.adjustStep`
- 24 × `liveImage.cmd.<id>` (label) + 24 × `liveImage.cmd.<id>.help` (help)

### Phase 5: Export & Persistenz (Tag 8)

#### Schritt 5.1: Export als FITS/PNG ✅

**Im API-Endpunkt `/export` (bereits in Schritt 2.2 definiert):**
- PNG: `cv::imwrite(path, session->current_image)` — verlustfrei
- FITS: `cv::Mat` → FITS über bestehende FITS-Schreibfunktion. Da
  `current_image` bereits das 8-bit gestretchte Anzeigebild ist (siehe
  Klarstellung in Abschnitt 10), speichert dieser Export **kein**
  wissenschaftliches Rohdaten-FITS, sondern das bearbeitete
  Vorschaubild in einem FITS-Container — Frontend-Label entsprechend
  ("FITS (bearbeitetes Bild)")
- Pfad: `runs/<run_id>/outputs/live_chat_export_<timestamp>.<ext>`
- Path-Traversal-Schutz: Pfad normalisieren, muss unter `runs/` liegen
- **HMS-Parameter werden nicht verändert.** Der Export ist ein
  eigenständiges Bild, unabhängig von der Pipeline-Konfiguration.

#### Schritt 5.2: Chat- & Operations-Historie in `.pi_memory` persistieren ✅

**Analog zu bestehendem `pi_run_chat_history_path()` in `pi_routes.cpp:423`.**

**Neue Funktionen in `pi_routes.cpp`:**

```cpp
std::filesystem::path pi_live_image_chat_history_path(
    const std::shared_ptr<AppState>& state,
    const std::string& run_id) {
    // Wie pi_run_chat_history_path(), aber Unterverzeichnis "live_image_chat/"
    // statt "run_chat/"
    // <pi_storage_dir>/live_image_chat/<safe_run_id>_<hash>.json
}

nlohmann::json read_pi_live_image_chat_history(
    const std::shared_ptr<AppState>& state,
    const std::string& run_id) {
    // Lädt JSON-Datei, gibt leeres Schema zurück wenn nicht vorhanden
}

void write_pi_live_image_chat_history(
    const std::shared_ptr<AppState>& state,
    const std::string& run_id,
    nlohmann::json history) {
    // Schreibt JSON-Datei atomar (temp + rename)
    // history enthält: schema_version, run_id, session_id, source_artifact,
    //   chat_history[], operation_history[], created_at, last_updated
}
```

**Persistierungs-Zeitpunkte:**
- Nach jeder Chat-Nachricht (Chat-Endpunkt)
- Nach jeder Adjust-Operation (Adjust-Endpunkt)
- Nach Undo und Redo
- Nach Reset (leert operation_history in der Datei)

**Neue API-Route:**

```cpp
CROW_ROUTE(app, "/api/pi/live-image-chat/history").methods("GET"_method)
([state](const crow::request& req) {
    const std::string run_id = req.url_params.get("run_id")
        ? std::string(req.url_params.get("run_id")) : "";
    if (run_id.empty()) return err_resp("BAD_REQUEST", "run_id required", 400);
    return json_resp(read_pi_live_image_chat_history(state, run_id));
});
```

#### Schritt 5.3: Session-Resume nach Browser-Refresh ✅

**Im Create-Endpunkt (Schritt 2.2) ergänzt:**

```cpp
// Nach dem Laden des FITS und Rendern des 1:1 JPEG:
auto history = read_pi_live_image_chat_history(state, run_id);
bool resumed = !history["operation_history"].empty();

if (resumed) {
    // Rekonstruiere current_image aus Original + operation_history
    cv::Mat reconstructed = original_image.clone();
    for (const auto& op : history["operation_history"]) {
        auto result = tile_compile::pi::apply_image_op(reconstructed, op);
        if (result.success) reconstructed = result.image;
    }
    session->current_image = reconstructed;
    session->chat_history = history["chat_history"];
    // undo_stack / redo_stack bleiben leer
}

// Response enthält resumed-Flag und chat_history:
response["resumed"] = resumed;
response["chat_history"] = session->chat_history;
```

**Frontend (`live-image-viewer.js`):**
- `open()` prüft `resumed` in der Response
- Wenn `resumed: true`: stelle `chat_history` in Chat-Panel dar
- Bild wird aus `jpeg_base64` angezeigt (bereits rekonstruiert)
- User kann nahtlos weiterarbeiten

**Reproduzierbarkeit:**
- `operation_history` in der History-Datei enthält alle Operationen mit
  Parametern und Timestamps
- Eine externe Anwendung oder ein Script kann die History-Datei laden
  und alle Operationen auf das Original-FITS anwenden, um das Ergebnis
  exakt zu reproduzieren
- Format: `pi.live-image-chat-history.v1` (stabil, versioniert)

### Phase 6: Tests & Polish (Tag 9-10)

#### Schritt 6.1: Backend-Tests ✅

**Neue Test-Dateien:**
- `web_backend_cpp/tests/test_pi_image_ops.cpp` — jede Operation mit bekanntem Input/Output
- `web_backend_cpp/tests/test_pi_live_image_session.cpp` — Create/Get/Evict/Close/Undo/Redo

**In `test_pi_routes.cpp` ergänzen:**
- Integration-Test: Create → Chat → Adjust → Undo → Redo → Reset → Export → Close
- Persistenz-Test: Create → Chat → Close → Create (Resume) → chat_history wiederhergestellt, Bild rekonstruiert
- History-Endpoint: GET `/api/pi/live-image-chat/history?run_id=...` → Schema korrekt
- Edge Cases: ungültige session_id, fehlendes Bild, Sidecar nicht erreichbar

**Verifikation:**
```bash
cmake --build build --target tests -j2 > /tmp/out_build.txt 2>&1
./build/test_pi_image_ops > /tmp/out_test_ops.txt 2>&1
./build/test_pi_live_image_session > /tmp/out_test_session.txt 2>&1
./build/test_pi_routes > /tmp/out_test_routes.txt 2>&1
sed -n '1,80p' /tmp/out_test_ops.txt
sed -n '1,80p' /tmp/out_test_session.txt
sed -n '1,120p' /tmp/out_test_routes.txt
```

#### Schritt 6.2: Frontend-Verifikation ✅

- Viewer öffnet/schließt korrekt bei Preview-Klick
- Chat sendet und empfängt, Bild aktualisiert sich mit Fade
- +/- Buttons erscheinen bei `adjustable: true`, verschwinden bei neuer Nachricht
- Undo/Redo-Buttons aktivieren/deaktivieren korrekt
- Reset stellt Originalbild wieder her
- Export speichert Datei im Run-Verzeichnis
- Nach Browser-Refresh: Chat-Verlauf wird wiederhergestellt, Bild rekonstruiert
- Dropdown: Auswahl sendet sofort, Hilfetext erscheint
- Responsive: Desktop (Bild links, Chat rechts) + Mobile (gestapelt)
- Mit statischen Fixtures oder laufendem Backend testen

#### Schritt 6.3: Edge Cases ✅

| Fall | Verhalten |
|------|-----------|
| Run ohne Output-Bild | "Kein Bild verfügbar" Meldung, Viewer nicht öffnen |
| Sehr große Bilder (>4000px) | JPEG-Quality auf 85 reduziert (in `render_fits_full_jpeg`) |
| AI nicht konfiguriert | Lokaler Fallback mit Keyword-Matching (`fallback_parse_message`) |
| Session-Timeout (30 Min) | "Session abgelaufen, bitte neu starten" |
| Sidecar nicht erreichbar | Fallback oder Fehlermeldung "AI nicht verfügbar" |
| Ungültige Operations-Parameter | `validate_op` → Fehler, Operation nicht angewendet |
| Path Traversal bei Export | Pfad-Validierung, muss unter `runs/` liegen |

### Phase 7: Erweiterte Bild-Operationen (Phase 2 aus Abschnitt 4) — umgesetzt

Umgesetzt sind `vibrance`, `color_temperature`, `unpurple`, `fixbanding`,
`star_desaturation` und `dehaze`. `levels`, `curves` sowie
`shadow_recovery`/`highlight_recovery` bleiben optionale spätere Erweiterungen;
letztere sind
durch `brightness.shadows`/`brightness.highlights` in Phase 1 bereits
näherungsweise abgedeckt).

- Die Operationen sind in Backend, linearem FITS-Pfad, Fallback und
  AI-System-Prompt verdrahtet und werden durch Backend-Tests abgedeckt.
- Die Dropdown-`FEATURE_PHASE`-Konstante steht auf `2`; nicht implementierte
  Phase-2-Befehle bleiben ausgeblendet.

---

## 10. Technische Entscheidungen

### JPEG vs PNG vs WebGL

| Format | Vorteile | Nachteile |
|--------|---------|-----------|
| **JPEG** | Klein (~200KB bei 1080px q92), schnell | Verlustbehaftet |
| PNG | Verlustfrei | Groß (~3MB bei 1080px) |
| WebGL | GPU-beschleunigt, interaktiv | Komplex, Overkill |

**Empfehlung:** JPEG mit Quality 92 für Live-Chat (schnell, kompakt).
Export als PNG (verlustfrei) oder FITS.

**Wichtig:** `current_image` ist das bereits gamma-gestretchte, auf 8-bit
reduzierte **Anzeigebild** (Ergebnis von `render_fits_full_jpeg`), nicht
die ursprünglichen FITS-Rohdaten (die z.B. 16/32-bit Float-Werte vor dem
Stretch enthalten). Ein "FITS-Export" von `current_image` ist damit
**kein** wissenschaftlich verwertbares FITS im Sinne von unbearbeiteten
Kalibrierdaten — es ist lediglich ein FITS-Container um ein bereits
verarbeitetes 8-bit-Bild. Der Export-Dialog sollte das entsprechend
kennzeichnen (z.B. Label "FITS (bearbeitetes Vorschaubild)" statt
"FITS (wissenschaftlich)"), damit Nutzer keine falschen Erwartungen an
Weiterverarbeitbarkeit in wissenschaftlicher Software haben.

### Bild im Speicher vs Datei

- `cv::Mat` im Session-Speicher: Schnell, keine I/O, aber RAM-Belastung
- Alternative: Temporäre Datei pro Session: Langsamer, aber RAM-schonend

**Empfehlung:** `cv::Mat` im Speicher (typisches Bild ~3.5MB, max 5 Sessions).

### AI Vision-Modell

- GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro: Alle unterstützen Bild-Eingabe
- Bild als base64 JPEG senden (nicht als URL)
- Bei iterativen Kommandos: Bild nur bei Create senden, danach nur
  Operations-Historie (Kosteneinsparung)
- Alternative: Bei jeder Nachricht aktuelles Bild senden (bessere Qualität,
  höhere Kosten)

**Empfehlung:** Bei jeder 2.-3. Nachricht Bild senden, sonst nur Historie.

**Bildgröße für die Vision-Anfrage begrenzen:** Das 1:1-JPEG, das im
Viewer angezeigt wird, kann bei großen Mosaiken/gestitchten Aufnahmen
deutlich über den üblichen Vision-API-Limits liegen (viele Provider
downscalen/tilen intern ab ~2000-8000px Kantenlänge bzw. haben harte
Byte-Limits pro Bild, z.B. 5-20 MB je nach Provider). Ein 1:1-Bild direkt
unverändert an die Vision-API zu senden hätte zwei Probleme: (1) mögliche
Ablehnung/Fehler bei Überschreitung der Provider-Limits, (2) selbst wenn
akzeptiert, resampled der Provider das Bild intern unkontrolliert — die
KI "sieht" dann ohnehin nicht das exakte 1:1-Pixelraster, wodurch die
Aussage in Abschnitt 4 ("AI greift nie direkt auf Pixel zu") zwar
weiterhin für die *Anwendung* der Operationen gilt, die *Analyse* aber auf
einer downgesampelten Version basiert. Deshalb: Für den Vision-Call wird
zusätzlich zum 1:1-Anzeige-JPEG ein separat auf max. 1568px Kantenlänge
herunterskaliertes JPEG erzeugt (gängiger Vision-API-Sweetspot) und
ausschließlich dieses an den Sidecar/die Vision-API geschickt. Das
1:1-Bild bleibt nur für die Anzeige und für die deterministische
OpenCV-Anwendung der Operationen relevant.

---

## 11. Sicherheits- und Validierungsregeln

- Alle Operations-Parameter werden vor Anwendung validiert (Range-Check)
- AI kann nur die definierten Operationen ausführen — kein beliebiger Code
- Session-ID ist UUID, nicht vorhersagbar
- Export schreibt nur in `runs/<run_id>/outputs/` — kein Path Traversal
- Max. 5 Sessions, 30 Min Timeout — DoS-Schutz
- Bild-Operationen sind nicht-destruktiv: Original bleibt erhalten, Reset möglich
- **Zugriffskontrolle:** Alle 9 Endpunkte (6.1-6.9) durchlaufen dieselbe
  Auth-Middleware wie die bestehenden `/api/pi/*`-Routen. Zusätzlich prüft
  `create` (6.1), dass der authentifizierte User Zugriff auf `run_id` hat
  (gleiche Berechtigung wie beim Öffnen des Run-Monitors), und alle
  Folge-Endpunkte (`chat`, `adjust`, `undo`, `redo`, `reset`, `export`,
  `close`) verifizieren, dass die zur `session_id` gehörende `run_id`
  weiterhin zum selben authentifizierten User gehört. Die UUID allein ist
  kein Autorisierungsnachweis, nur ein Schutz gegen zufälliges Erraten.
- **Fehlerpfad bei ungültigen Operationen:** Schlägt `validate_op` für eine
  von der AI zurückgegebene Operation fehl (z.B. Parameter außerhalb des
  gültigen Bereichs, unbekannter `type`), wird die Operation **nicht** auf
  `current_image` angewendet und **nicht** auf `undo_stack`/
  `operation_history`/`chat_history` gepusht. Die API-Antwort enthält
  `warnings: ["invalid_operation: <grund>"]`, `jpeg_base64` bleibt
  unverändert (aktuelles Bild), und `summary` erklärt dem Nutzer in
  Textform, dass die Änderung nicht angewendet werden konnte. Bricht der
  Request selbst ab (z.B. Sidecar nicht erreichbar), gilt dasselbe:
  Session-Zustand bleibt unverändert, HTTP 200 mit Fehler-`warnings`
  (kein 5xx, damit das Frontend den Chat nicht als abgestürzt behandelt).

---

## 12. Entschiedene Fragen

- [x] **Soll der Live Image Chat auch für Raw Stack GUI verfügbar sein?**
      **Nein** — nur für Run-Monitor nach Run-Abschluss.

- [x] **Soll die Operations-Historie als JSON-Datei im Run gespeichert werden?**
      **Ja** — Chat-Verlauf und Operations-Historie werden in `.pi_memory`
      persistiert (wie jeder PI-Chat), unter
      `<pi_storage_dir>/live_image_chat/<run_id>_<hash>.json`.
      Siehe Abschnitt 5 (Persistenz) und Abschnitt 9, Phase 5.

- [x] **Soll man gespeicherte Sessions fortsetzen können (nach Browser-Refresh)?**
      **Ja** — der Create-Endpunkt erkennt bestehende History-Dateien und
      rekonstruiert `current_image` durch sequentielles Anwenden aller
      Operationen aus `operation_history` auf das Originalbild.
      `undo_stack`/`redo_stack` beginnen leer. Siehe Abschnitt 5, Lifecycle
      Schritt 9, und Abschnitt 9, Phase 5, Schritt 5.3.

- [x] **Mapping von Bild-Operationen auf HMS-Parameter für Resume?**
      **Nein** — HMS-Parameter werden **nicht** verändert. Der Live Image
      Chat speichert das Ergebnis als eigenständiges FITS/PNG in
      `runs/<run_id>/outputs/`. Die vollständige Operations-Historie in der
      History-JSON ermöglicht exakte Reproduktion: ein Script oder externe
      Anwendung kann die History laden und alle Operationen auf das
      Original-FITS anwenden. Siehe Abschnitt 9, Phase 5.
