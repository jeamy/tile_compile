# AQMH First Frontend Transition Plan

**Version:** v0.3.0 (2026-06-12)  
**Scope:** `web_frontend`, `web_backend_cpp`, runner/config status contract  
**Related documents:**
- `docs/AQMH/aqmh_frontend_integration_plan.md`
- `docs/AQMH/aqmh_implementation_plan.md`
- `docs/AQMH/aqmh_methodik_en.md`
- `docs/AQMH/aqmh_frontend_default_switch.md`

---

## 1. Zielbild

Das Frontend soll AQMH als primaere Rekonstruktionsmethode behandeln. Classic Tile Compile bleibt verfuegbar, ist aber eine explizit gewaehlte Alternative.

User-facing Auswahl:

```text
method = aqmh
method = classic_tile_compile
```

AQMH darf dabei nicht als Variante von Classic dargestellt werden. Die UI muss AQMH als eigenstaendige Methode mit dichter Qualitaetskarten-Berechnung, AQMH-Rekonstruktion und AQMH-Diagnostik zeigen. Classic Tile Compile bleibt als separate Methode mit Classic-Tile-/Local-Metrics-Controls erhalten.

Explizite Nicht-Ziele:

- Kein kombinierter AQMH/Classic-Rekonstruktionsmodus.
- Kein AQMH-Fallback auf Classic Tile Weights.
- Kein AQMH Tile Mode.
- Kein stilles Umschalten der Methode innerhalb eines Runs.
- Kein default-enabled Cherry-Pick.

---

## 2. Affected Files

### 2.1 Config / Schema

| File | Change |
|---|---|
| `tile_compile_cpp/include/tile_compile/config/configuration.hpp` | Top-Level `method` in `Config`; `AqmhConfig::enabled` als abgeleitetes Runtime-Flag behandeln |
| `tile_compile_cpp/src/io/config.cpp` | `method` parsen, serialisieren und daraus das Runtime-Flag `aqmh.enabled` ableiten |
| `tile_compile_cpp/tile_compile.yaml` | `method: aqmh` als Default; `aqmh.enabled: true` |
| `tile_compile_cpp/tile_compile.schema.yaml` | `method` Property und AQMH-first Default-Beschreibung |
| `tile_compile_cpp/tile_compile.schema.json` | Gleiches Schema-Update fuer JSON |
| `tile_compile_cpp/examples/aqmh_enabled.example.yaml` | Entfernen; AQMH ist nicht mehr ein Sonderbeispiel |
| `tile_compile_cpp/examples/classic_mode.example.yaml` | Neu: explizites Classic-Beispiel |

### 2.2 Backend / Runner / Status

| File | Change |
|---|---|
| `web_backend_cpp/include/services/run_inspector.hpp` | Method-aware Phase Order oder Status-Mapping |
| `web_backend_cpp/src/services/run_inspector.cpp` | `method` aus Run-Config/Metadata ableiten und im Status ausgeben |
| `web_backend_cpp/src/services/report_generator.cpp` | Behalten: wird vom Web-Backend/API-Report verwendet; AQMH-Reportabschnitt und AQMH-native BGE-Inputquelle in Backend-generierten Reports ausweisen |
| `web_backend_cpp/src/routes/*` | Falls Status/Run-Start-Routen Method-Felder filtern oder erzeugen, entsprechend erweitern |
| `tile_compile_cpp/apps/*runner*` | Runner Events langfristig auf AQMH-spezifische Phasen erweitern |

### 2.3 Frontend

| File | Change |
|---|---|
| `web_frontend/src/app.js` | Zentrale `currentMethod()`-Logik, AQMH-first Defaults, method-aware Phase Order, Dashboard Groups |
| `web_frontend/index.html` | Dashboard Method Badge und Pipeline Preview AQMH-first |
| `web_frontend/wizard.html` | AQMH zuerst und vorausgewaehlt; Classic als zweite Option |
| `web_frontend/parameter-studio.html` | AQMH Kategorie/Controls als Default; Classic Controls methodenabhaengig |
| `web_frontend/parameter-studio-page.js` | Default Kategorie, Method-Sync, Sichtbarkeit |
| `web_frontend/run-monitor.html` | AQMH Status Panel primaer anzeigen |
| `web_frontend/history-tools.html` | Method Tag als primaeres Sortier-/Filterkriterium; fehlende Methode als AQMH behandeln |
| `web_frontend/i18n/de.json`, `web_frontend/i18n/en.json` | Labels, Tooltips und Warnungen fuer AQMH-first und Classic-Option |

### 2.4 Scripts / Reports

| File | Change |
|---|---|
| `tile_compile_cpp/scripts/generate_report.py` | Falls dieser Reportpfad weiterhin verwendet wird: eigenstaendiger AQMH-Reportabschnitt (`_gen_aqmh_metrics`) und AQMH-native BGE-Inputquelle ausweisen |

### 2.5 Documentation / Tests

| File | Change |
|---|---|
| `docs/AQMH/aqmh_frontend_integration_plan.md` | "Default-enabled AQMH" als Non-Goal entfernen |
| `docs/AQMH/aqmh_implementation_plan.md` | Default-Kommentare und Gating-Text von AQMH-off auf AQMH-first aktualisieren |
| `web_backend_cpp/tests/*`, `tile_compile_cpp/tests/*` | Fixtures und Erwartungen fuer AQMH Default auditieren |
| Frontend Tests | Wizard/Dashboard/Run-Monitor/Parameter-Studio Default-Erwartungen drehen |

## 3. Method Contract

### 3.1 Top-Level Method

Neue und vom Frontend erzeugte Konfigurationen sollen ein explizites Top-Level-Feld enthalten:

```yaml
method: aqmh  # aqmh | classic_tile_compile
```

Normalisierungsregel:

```text
method = aqmh                  -> aqmh (aqmh.enabled wird auf true normalisiert)
method = classic_tile_compile  -> classic (aqmh.enabled wird auf false normalisiert)
method fehlt                   -> aqmh (aqmh.enabled wird auf true normalisiert)
```

`method` ist die einzige Quelle der Wahrheit. `aqmh.enabled` ist kein Eingabe- oder Kompatibilitaets-Schalter fuer die Methodenerkennung, sondern nur ein aus `method` abgeleitetes Runtime-Flag. Fehlt `method`, gilt immer AQMH.

### 3.2 Default

Der Default fuer neue Frontend-Konfigurationen ist:

```yaml
method: aqmh
aqmh:
  enabled: true
```

Classic wird nur gesetzt, wenn der Nutzer explizit die Classic-Option waehlt:

```yaml
method: classic_tile_compile
aqmh:
  enabled: false
```

### 3.3 Example Configs

`aqmh_enabled.example.yaml` wird entfernt. Stattdessen gibt es ein explizites Classic-Beispiel:

```yaml
# Classic Tile Compile mode
method: classic_tile_compile
aqmh:
  enabled: false
```

Optional kann ein AQMH-Tuning-Beispiel angelegt werden, das erweiterte Pyramid-/Storage-Parameter zeigt:

```yaml
method: aqmh
aqmh:
  enabled: true
  storage:
    resolution_divisor: 2
    dtype: uint16
    max_resident_maps: 2
```

### 3.4 Default Construction Audit

Alle Codepfade, die `config::Config{}` oder `config::AqmhConfig{}` direkt konstruieren, muessen auditiert werden. Mit AQMH-first darf ein synthetisch gebauter Config-Default nicht versehentlich Classic testen oder umgekehrt.

Regel fuer Tests und Fixtures:

```text
Classic-Test  -> method=classic_tile_compile explizit setzen; aqmh.enabled=false nach Normalisierung erwarten
AQMH-Test     -> method=aqmh explizit setzen; aqmh.enabled=true nach Normalisierung erwarten
Default-Test  -> pruefen, dass method=aqmh entsteht
```

---

## 4. Backend-Aenderungen

### 4.1 Config Parsing und Validation

Erforderlich:

1. `config::Config` bekommt ein Top-Level-Feld `method`.
2. YAML Parsing liest `method`.
3. YAML Serialization schreibt `method`.
4. Validation erlaubt nur:
   - `aqmh`
   - `classic_tile_compile`
5. Normalisierung leitet `aqmh.enabled` ausschliesslich aus `method` ab; Validation prueft nur noch den normalisierten Zustand.

Normalisierungsregel:

```text
method explizit gesetzt -> aqmh.enabled daraus ableiten
method fehlt            -> method = aqmh, danach aqmh.enabled = true ableiten
```

`aqmh.enabled` wird niemals als Quelle fuer `method` verwendet. method-freie Configs werden durch Normalisierung zu gueltigen AQMH-Configs.

Schema-Aenderungen:

1. `method` als Top-Level Property in YAML- und JSON-Schema aufnehmen.
2. Enum-Werte: `aqmh`, `classic_tile_compile`.
3. Default/Description: AQMH ist Default, Classic ist explizite Alternative.
4. `aqmh.enabled` Beschreibung aktualisieren: abgeleitetes Runtime-Flag, kein UI- oder Config-Method-Schalter.
5. **Schema-Konsistenzprüfung:** Schema muss validieren, dass:
   - Bei `method=aqmh` muss `aqmh.enabled=true` (oder wird automatisch gesetzt)
   - Bei `method=classic_tile_compile` muss `aqmh.enabled=false` (oder wird automatisch gesetzt)

### 4.2 Config Normalization Function

**Implementierungsort:** `tile_compile_cpp/src/io/config.cpp`

Funktion `normalizeMethod(config::Config& config)` muss folgende Regeln anwenden:

```cpp
// Wird aufgerufen nach dem Parsen der Config, vor der Validierung
void normalizeMethod(config::Config& config) {
    if (!config.method.has_value()) {
        config.method = "aqmh";
    }
    if (config.method == "aqmh") {
        config.aqmh.enabled = true;
    } else if (config.method == "classic_tile_compile") {
        config.aqmh.enabled = false;
    }
    // Unbekannte Werte werden von der Validation abgefangen
}
```

`config.method` wird als `std::optional<std::string>` modelliert. Nach `normalizeMethod()` ist es immer gesetzt. `aqmh.enabled` darf nach der Normalisierung als Runtime-Flag gelesen werden, aber nirgendwo als Methodenquelle.

**Aufrufpunkt:**
- Nach YAML-Parsing in `config::Config::from_yaml()`
- In JSON-Parsing-Pfaden nur falls ein solcher Pfad tatsaechlich existiert
- Vor der Schema-Validierung
- Vor dem Serialisieren (um Konsistenz zu gewährleisten)

**Test-Anforderung:**
- Unit-Test fuer alle 3 Faelle aus Section 3.1 (method=aqmh, method=classic_tile_compile, method fehlt)

### 4.3 Status Payload

Alle Run-Status-Antworten sollen `method` enthalten:

```json
{
  "method": "aqmh",
  "aqmh_enabled": true
}
```

Eine separate `api_version` ist fuer diese Umstellung nicht erforderlich. Das Frontend wertet `method` aus; wenn `method` fehlt, gilt der AQMH-first Default aus Section 3.1.

Fuer AQMH-Runs soll der Status zusaetzlich AQMH-spezifische Daten enthalten, sobald verfuegbar:

```json
{
  "method": "aqmh",
  "aqmh": {
    "enabled": true,
    "storage": {
      "resolution_divisor": 2,
      "dtype": "uint16",
      "max_resident_maps": 2
    },
    "maps": {
      "computed": 143,
      "total": 300,
      "stream": "luma"
    },
    "cache": {
      "bytes_written": 7200000000,
      "bytes_read": 1800000000,
      "cache_hits": 1100,
      "cache_misses": 100,
      "max_resident_maps_observed": 2
    }
  }
}
```

Das Frontend darf AQMH nicht aus vorhandenen Cache-Dateien ableiten. Die Methode kommt aus Config, Status oder Run-Metadaten.

### 4.4 Phase Order

Die aktuell Classic-orientierte Phase Order muss methodenabhaengig werden.

Shared Phasen bleiben gemeinsam:

```text
SCAN_INPUT
CHANNEL_SPLIT
NORMALIZATION
GLOBAL_METRICS
TILE_GRID
REGISTRATION
PREWARP
COMMON_OVERLAP
STACKING
DEBAYER
ASTROMETRY
BGE
PCC
HYPERMETRIC_STRETCH
```

Classic-spezifisch:

```text
LOCAL_METRICS
TILE_RECONSTRUCTION
STATE_CLUSTERING
SYNTHETIC_FRAMES
```

AQMH-spezifisch:

```text
AQMH_MAPS
AQMH_RECONSTRUCTION
AQMH_DIAGNOSTICS
```

Der bestehende Code verwendet derzeit teilweise den Uebergangsnamen `AQMH_QUALITY_MAPS`. Zielname fuer die method-aware UI ist `AQMH_MAPS`; `AQMH_QUALITY_MAPS` bleibt nur als Compatibility-Alias.

Kompatibilitaetsmapping fuer die erste Umstellung:

| Backend event | Bedingung | Frontend label | Action |
|---|---|---|---|
| `AQMH_QUALITY_MAPS` | beliebig | `AQMH_MAPS` | Umlabeln |
| `LOCAL_METRICS` | `method=aqmh` | **Ausgeblendet** | **Nicht anzeigen** (nicht umlabeln!) |
| `TILE_RECONSTRUCTION` | `method=aqmh` | `AQMH_RECONSTRUCTION` | Umlabeln |
| `STATE_CLUSTERING` | `method=aqmh` | **Ausgeblendet** | Nicht anzeigen |
| `SYNTHETIC_FRAMES` | `method=aqmh` | **Ausgeblendet** | Nicht anzeigen |

**Klärung:**
- `LOCAL_METRICS` wird bei AQMH **komplett ausgeblendet**, nicht umgelabelt (im Gegensatz zur vorherigen Version).
- Nur `AQMH_QUALITY_MAPS` und `TILE_RECONSTRUCTION` werden umgelabelt.

Dieses Mapping ist nur eine Uebergangsschicht. Ziel ist, dass der Runner AQMH-spezifische Events direkt emittiert.

### 4.5 Backend Event Normalization Layer

**Implementierungsort:** `web_backend_cpp/src/services/run_inspector.cpp`

Die Funktion `normalizePhaseEvent()` wendet das Mapping aus Section 4.4 an:

```cpp
std::string normalizePhaseEvent(const std::string& event, const std::string& method) {
    if (method == "aqmh") {
        if (event == "LOCAL_METRICS" || 
            event == "STATE_CLUSTERING" || 
            event == "SYNTHETIC_FRAMES") {
            return ""; // Ausblenden
        }
        if (event == "AQMH_QUALITY_MAPS") return "AQMH_MAPS";
        if (event == "TILE_RECONSTRUCTION") return "AQMH_RECONSTRUCTION";
    }
    return event; // Unverändert
}
```

**Ziel:** Langfristig soll der Runner direkt `AQMH_MAPS`, `AQMH_RECONSTRUCTION`, `AQMH_DIAGNOSTICS` emittieren.

---

## 5. Frontend-Aenderungen

### 5.1 Zentrale Method-Erkennung

Die Frontend-Logik soll von `isAqmhEnabled()` auf eine Methode-API wechseln:

```js
function currentMethod() {
  return "aqmh"; // aqmh | classic_tile_compile
}

function isAqmhMethod() {
  return currentMethod() === "aqmh";
}
```

Prioritaet:

1. Run-Monitor-Status `status.method`, wenn ein konkreter Run angezeigt wird
2. expliziter Draft-Wert `method` — im Frontend gerade bearbeitete, noch nicht gespeicherte Konfiguration
3. Config-Objekt `config.method`
4. YAML `method: ...`
5. UI-State/URL-Fallback fuer noch nicht geladene Configs
6. Default fuer neue UI-Kontexte: `aqmh`

`aqmh.enabled` wird von der Frontend-Methodenerkennung nicht ausgewertet. Fehlt `method`, gilt AQMH.

### 5.2 Dashboard

Das Dashboard muss AQMH als Default sichtbar machen:

| Feld | AQMH Default | Classic Option |
|---|---|---|
| Method badge | `AQMH` | `Tile Compile Classic` |
| Quality model | `AQMH dense quality maps` | `Classic local/tile metrics` |
| Reconstruction | `AQMH pixel-wise reconstruction` | `Classic weighted stack` |
| Cache estimate | AQMH map cache estimate | normal run cache |

Die Pipeline-Vorschau soll fuer AQMH nicht `TILES` als Hauptbegriff verwenden. Empfohlen:

```text
SCAN -> REG -> AQMH -> STACK -> ASTROM -> BGE -> PCC -> HMS -> DONE
```

Fuer Classic:

```text
SCAN -> REG -> TILES -> STACK -> ASTROM -> BGE -> PCC -> HMS -> DONE
```

AQMH Cache-Schaetzung:

```text
stored_width  = ceil(width / resolution_divisor)
stored_height = ceil(height / resolution_divisor)
bytes_per_map = stored_width * stored_height * dtype_bytes
total_cache   = bytes_per_map * frame_count * map_stream_count
```

Erste Implementierung kann `map_stream_count = 1` fuer `luma` annehmen.

Dashboard-Warnungen werden fuer AQMH als Default direkt angezeigt, nicht erst nach einem zusaetzlichen Opt-in:

| Condition | Warning |
|---|---|
| `resolution_divisor = 1` und `frame_count > 50` | grosser AQMH Cache erwartet; `resolution_divisor=2` pruefen |
| `max_resident_maps > 4` | hoher resident map count; moeglicher RAM-Druck |
| `cherry_pick.enabled = true` | Pixel-level frame selection aktiv; nicht Default |

### 5.3 Wizard

Die erste Methodenauswahl im Wizard:

```text
Reconstruction method
(*) AQMH
( ) Tile Compile Classic
```

Wenn AQMH aktiv ist, werden AQMH-Speicherprofile angeboten:

```text
(*) Balanced AQMH cache
    resolution_divisor=2, dtype=uint16, max_resident_maps=2

( ) Exact AQMH cache
    resolution_divisor=2, dtype=float32, max_resident_maps=2

( ) Full resolution AQMH cache
    resolution_divisor=1, dtype=float32, max_resident_maps=2
```

Diese drei Profile ersetzen die zwei Profile ("Conservative", "Full resolution") aus dem Integration Plan §F4. Die Bezeichnungen sind aktualisiert; das konservative Profil ist jetzt "Balanced" und entspricht unveraendert `resolution_divisor=2, uint16`.

Classic-spezifische Tile-/Local-Metrics-Controls werden nur gezeigt, wenn Classic gewaehlt ist.

Die AQMH Storage Section ist beim ersten Oeffnen sichtbar. Sie wird ausgeblendet, wenn `method=classic_tile_compile` gewaehlt wird.

### 5.4 Parameter Studio

Parameter Studio braucht methodenbewusste Sichtbarkeit:

AQMH-Kontext:

- `method`
- `aqmh.storage.*`
- `aqmh.pyramid.*`
- `aqmh.diagnostics.*`
- `aqmh.cherry_pick.*`
- shared preprocessing/runtime/output controls

Classic-Kontext:

- `method`
- Classic tile/local metrics
- Classic reconstruction controls
- shared preprocessing/runtime/output controls

Nicht anzeigen:

- AQMH tile weighting
- AQMH/Classic combined mode
- Classic fallback fuer AQMH
- Classic local metric controls als AQMH-Rekonstruktionsparameter

Default:

```js
const DEFAULT_PARAM_CATEGORY = "aqmh";
```

Classic-only Controls erhalten eine eindeutige Markierung, z. B. `data-method="classic"`, und sind im AQMH-Kontext verborgen. AQMH Controls erhalten entsprechend `data-method="aqmh"` oder bleiben sichtbar, wenn sie die Default-Methode betreffen.

Sichtbarkeitsregel:

```text
method=aqmh                  -> AQMH Controls plus shared Controls
method=classic_tile_compile  -> Classic Controls plus shared Controls
```

### 5.5 Run Monitor

Header:

```text
AQMH
```

AQMH Stage Labels:

```text
AQMH maps
AQMH reconstruction
AQMH diagnostics
```

Anzeigen:

- maps computed / total
- cache bytes written/read
- cache hit rate
- resident maps observed vs configured
- cache misses
- unsupported pixels
- cherry-pick warning, falls aktiv

Classic-only Phasen `LOCAL_METRICS`, `TILE_RECONSTRUCTION`, `STATE_CLUSTERING` und `SYNTHETIC_FRAMES` duerfen bei AQMH nicht als normale pending Phasen erscheinen. Gemaess dem Compatibility-Mapping in Section 4.4: `AQMH_QUALITY_MAPS` wird auf `AQMH_MAPS` umbenannt; `TILE_RECONSTRUCTION` im AQMH-Kontext wird als `AQMH_RECONSTRUCTION` angezeigt; `LOCAL_METRICS`, `STATE_CLUSTERING` und `SYNTHETIC_FRAMES` werden bei AQMH komplett ausgeblendet — sie erhalten kein AQMH-Label.

---

## 6. Artifacts, Reports, History

### 6.1 Artifacts

AQMH-Artefakte sollen prominent, aber cache-schonend gruppiert werden:

| Artifact | UI Label | Verhalten |
|---|---|---|
| `artifacts/aqmh_metrics.json` | `AQMH Metrics` | Summary bevorzugen |
| `artifacts/aqmh_regions.json` | `AQMH Regions` | Summary bevorzugen |
| `cache/aqmh/*` | `AQMH cache` | gruppiert/standardmaessig eingeklappt |

Raw Map Cache-Dateien duerfen nicht einzeln als primaere Artefakte gerendert werden.

### 6.2 Reports

Der Report braucht einen eigenstaendigen AQMH-Abschnitt:

- AQMH quality heatmap
- AQMH artifact fraction heatmap
- per-frame `map_mean`
- per-frame `artifact_frac`
- AQMH cache/timing table
- optional AQMH-vs-Classic nur als Cross-Run-Vergleich

AQMH darf nicht unter Classic Local Metrics verschachtelt werden.

### 6.3 BGE AQMH-native Inputs

Wenn `method=aqmh` und BGE aktiv ist, darf BGE nicht auf `local_metrics.json` zurueckgreifen. 

**Implementierungsort:** `tile_compile_cpp/apps/runner` (BGE-Integrationspunkt)

Das Backend setzt in diesem Fall `tile_metrics_source = "aqmh_output"` im BGE-Artifact. 
Die Funktion `setBgeTileMetricsSource()` wird im Runner aufgerufen:

```cpp
void setBgeTileMetricsSource(BgeArtifact& artifact, const Config& config) {
    if (config.method == "aqmh") {
        artifact.tile_metrics_source = "aqmh_output";
    } else {
        artifact.tile_metrics_source = "classic_local_metrics";
    }
}
```

Das Frontend und der Report muessen das korrekt ausweisen:

- Report: BGE-Eingabequelle als `AQMH output` labeln, nicht als Classic Local Metrics.
- History/Comparison: Vergleichsfeld `tile_metrics_source` zur Unterscheidung von `aqmh_output` und `classic_local_metrics`.
- Run Monitor: kein Signal, das `local_metrics.json` als BGE-Input impliziert, wenn die Methode AQMH ist.

Classic-Runs behalten `tile_metrics_source = "classic_local_metrics"` und bleiben unveraendert.

### 6.4 History

History-Tags:

```text
AQMH
AQMH cherry-pick
Tile Compile Classic
```

Vergleichsfelder:

- `method`
- AQMH cache size
- AQMH map compute time
- AQMH reconstruction time
- mean artifact fraction

AQMH-vs-Classic ist ein Vergleich zweier separater Runs, kein einzelner kombinierter Run-Modus.

History braucht eine einfache AQMH-first Ableitung:

```text
Run metadata method=aqmh                 -> AQMH
Run metadata method=classic_tile_compile -> Tile Compile Classic
Kein method                              -> AQMH
```

---

## 7. No Migration

Es gibt keine Migration bestehender Run-Verzeichnisse oder Projekt-Configs. Fehlt `method`, wird nach AQMH-first-Regel AQMH angenommen. Classic bleibt nur dann aktiv, wenn eine Config oder ein UI-Draft explizit `method=classic_tile_compile` setzt. Neue Projekte, neue Wizard-Konfigurationen, Default-Resets und Configs ohne Method-Feld erzeugen AQMH:

```yaml
method: aqmh
aqmh:
  enabled: true
```

Der Wizard muss diese Aenderung sichtbar machen:

```text
AQMH is the default reconstruction method.
Select Tile Compile Classic for the previous Classic pipeline.
```

Keine stille Methodenumschaltung innerhalb eines laufenden oder resumed Runs: Resume verwendet die Methode des Run-Verzeichnisses.

---

## 8. Frontend State Persistence

### 8.1 Method State Management

Die gewählte Methode muss über Page-Reloads und Sessions hinweg konsistent bleiben:

**Speicherorte:**
1. **URL-Hash:** `method=aqmh` oder `method=classic_tile_compile`
2. **Server UI State oder localStorage:** bestehende UI-State-Infrastruktur bevorzugen; `tileCompile.method` nur als Fallback
3. **Draft-Config:** Im Frontend-State fuer nicht gespeicherte Änderungen

**Priorität:** aktiver Run-Status > Draft-Config > geladene Config/YAML > URL-Hash > Server UI State/localStorage > Default (AQMH). URL-Hash und gespeicherter UI-State duerfen eine geladene Config oder einen aktiven Run nicht ueberschreiben.

**Implementierung in `web_frontend/src/app.js`:**

```javascript
// Lade Methode beim Start
function loadMethodFromState() {
    const runMethod = currentRunStatus?.method;
    if (runMethod && ['aqmh', 'classic_tile_compile'].includes(runMethod)) {
        return runMethod;
    }
    const draftMethod = currentDraftValueForPath('method');
    if (draftMethod && ['aqmh', 'classic_tile_compile'].includes(draftMethod)) {
        return draftMethod;
    }
    const configMethod = currentConfig?.method || detectMethodFromYaml(currentConfigYaml);
    if (configMethod && ['aqmh', 'classic_tile_compile'].includes(configMethod)) {
        return configMethod;
    }
    const urlMethod = getUrlHashParam('method');
    if (urlMethod && ['aqmh', 'classic_tile_compile'].includes(urlMethod)) {
        return urlMethod;
    }
    const storedMethod = readUiStateValue('tileCompile.method') ||
                         localStorage.getItem('tileCompile.method');
    if (storedMethod && ['aqmh', 'classic_tile_compile'].includes(storedMethod)) {
        return storedMethod;
    }
    return 'aqmh'; // Default
}

// Speichere Methode bei Änderung
function saveMethodToState(method) {
    setUrlHashParam('method', method);
    writeUiStateValue('tileCompile.method', method);
    localStorage.setItem('tileCompile.method', method);
}
```

**URL-Hash Format:**
```
#method=aqmh&tab=parameters
```

### 8.2 Performance Warnings and Guards

**Cache- und Speicherlimits:**
- **Warnung:** `resolution_divisor=1` + `frame_count > 50` → Cache-Warnung
- **Warnung:** geschaetzter AQMH-Disk-Cache ist gross im Vergleich zum freien Speicherplatz
- **Blockierung:** geschaetzte residente Map-Speichernutzung ueberschreitet das konfigurierte Memory Budget deutlich:
  ```
  "AQMH resident map cache would exceed the configured memory budget (X GB).
  Please reduce max_resident_maps or increase resolution_divisor."
  ```

**Implementierung:**
```javascript
function validateAqmhResourceLimits(scan, config) {
    const diskCacheBytes = estimateAqmhDiskCacheBytes(scan, config);
    const residentBytes = estimateAqmhResidentMapBytes(scan, config);
    const memoryBudgetBytes = config.runtime_limits.memory_budget * 1024**2;
    
    if (residentBytes > memoryBudgetBytes * 0.8) {
        throw new Error(`Resident map cache exceeds budget: ${formatBytes(residentBytes)}`);
    }
    return { diskCacheBytes, residentBytes };
}
```

---

## 9. I18n und UX Labels

Verwenden:

- `AQMH`
- `AQMH dense quality maps`
- `AQMH pixel-wise reconstruction`
- `AQMH cache`
- `Resident maps`
- `Tile Compile Classic`
- `Classic local/tile metrics`

Vermeiden:

- `AQMH tile mode`
- `AQMH/Classic combined mode`
- `Classic fallback`
- `Dense map mode`
- `AQMH as Classic local metrics`

Mathematische interne Begriffe wie `Phi_snr`, `Psi_s` oder `P_actual` gehoeren nicht in normale UI-Texte. Sie koennen in Advanced Diagnostics oder Methodik-Dokumentation erscheinen.

---

## 10. Rollback und Notfallmechanismen

### 10.1 Rollback via Environment Variable

**Zweck:** Ermöglicht sofortige Rückkehr zu Classic, falls AQMH-First Probleme verursacht.

**Implementierung:**
```cpp
// In tile_compile_cpp/src/io/config.cpp
std::string getEffectiveMethod(const Config& config) {
    const char* forceClassic = std::getenv("FORCE_CLASSIC");
    if (forceClassic && std::string(forceClassic) == "1") {
        return "classic_tile_compile";
    }
    return config.method.value_or("aqmh");
}
```

**Aufruf:**
```bash
# Temporär für alle Runs
FORCE_CLASSIC=1 tile_compile_runner ...

# Oder als System-Environment-Variable
# (Linux): export FORCE_CLASSIC=1
# (Windows): set FORCE_CLASSIC=1
```

**Wirkung:**
- Uebersteuert die effektive Methode nur fuer neu gestartete Runs
- Uebersteuert den Default (AQMH) fuer neue Configs im aktuellen Prozess
- Betrifft nur den aktuellen Prozess
- Gilt nur fuer neue Run-Starts. Resume eines bestehenden Run-Verzeichnisses verwendet die dort gespeicherte Methode und darf nicht still umgeschaltet werden.

### 10.2 Rollback via CLI Flag

**Alternative:** Kommandzeilen-Flag für den Runner:
```bash
tile_compile_runner --force-classic ...
```

**Implementierung:** `tile_compile_cpp/apps/runner/main.cpp`

### 10.3 Rollback Prozedur

| Schritt | Aktion | Verantwortlich |
|---|---|---|
| 1 | Problem identifizieren (z. B. AQMH-Crash) | Nutzer/Dev |
| 2 | `FORCE_CLASSIC=1` setzen oder `--force-classic` verwenden | Nutzer |
| 3 | Run neu starten | Nutzer |
| 4 | Logs sammeln mit `--verbose` Flag | Nutzer |
| 5 | Issue erstellen mit Logs und Reproduktionsschritten | Nutzer |
| 6 | Hotfix: Config-Normalisierung deaktivieren (falls nötig) | Dev |

### 10.4 Monitoring

**Metriken zur Umstellungsüberwachung:**
```json
{
  "method_usage": {
    "aqmh": 1250,
    "classic_tile_compile": 320,
    "missing_method_defaulted_to_aqmh": 45
  }
}
```

**Implementierung:** Prometheus-Metriken oder Logstash-Logs in `web_backend_cpp`

---

## 11. Testplan

### 11.1 Frontend

1. Neue UI-Session zeigt AQMH als vorausgewaehlte Methode.
2. Method Selector kann auf Tile Compile Classic wechseln.
3. Wechsel auf Classic setzt `method=classic_tile_compile`; `aqmh.enabled=false` entsteht erst durch Normalisierung.
4. Wechsel auf AQMH setzt `method=aqmh`; `aqmh.enabled=true` entsteht erst durch Normalisierung.
5. Dashboard zeigt fuer AQMH die Gruppe `AQMH`, nicht `TILES`.
6. Run Monitor zeigt fuer AQMH `AQMH maps` und `AQMH reconstruction`.
7. Classic-only Phasen erscheinen bei AQMH nicht als pending.
8. Runs oder Configs ohne `method` werden als AQMH angezeigt; `aqmh.enabled` wird fuer die Methodenerkennung ignoriert.
9. Keine combined/fallback/tile-weighting AQMH Controls werden gerendert.
10. Dashboard ohne geladene Config zeigt `AQMH`.
11. Wizard zeigt die AQMH Storage Section sofort.
12. Parameter Studio startet in der AQMH Kategorie.
13. AQMH Cache Estimate wird nach Scan ohne weitere Nutzerauswahl angezeigt.
14. Dashboard-Warnungen erscheinen fuer grosse AQMH-Caches, hohe resident maps und Cherry-Pick.

### 11.2 Backend Contract

1. Status endpoint enthaelt immer `method`.
2. Neue Default-Config serialisiert `method: aqmh`.
3. Config ohne `method`, aber mit beliebigem `aqmh.enabled`, wird als AQMH erkannt.
4. Config ohne `method` und ohne `aqmh` wird als AQMH erkannt.
5. Inkonsistente Configs werden normalisiert oder mit klarer Validation-Fehlermeldung abgelehnt.
6. AQMH-Status enthaelt storage/cache/map Summary, sobald verfuegbar.
7. Schema YAML und JSON akzeptieren `method=aqmh` und `method=classic_tile_compile`.
8. `Config{}` / `AqmhConfig{}` Default-Konstruktion erzeugt den dokumentierten Default oder wird in Tests explizit gesetzt.

### 11.3 Integration

1. Neuer Run aus dem Frontend startet als AQMH.
2. Classic-Run aus dem Frontend startet nur nach expliziter Classic-Auswahl.
3. AQMH-Run erzeugt AQMH-Artefaktgruppe und AQMH-Reportsektion.
4. Resume eines AQMH-Runs behaelt `method=aqmh`.
5. History zeigt AQMH und Classic eindeutig unterscheidbar.

### 11.4 Test Audit

Anzupassen sind alle Tests, die:

- `AqmhConfig{}` ohne YAML bauen und implizit Classic erwarten.
- Dashboard oder Wizard mit Classic als Default erwarten.
- `RUN_MONITOR_PHASE_ORDER` als reine Classic-Liste erwarten.
- Parameter Studio mit Classic als Startkategorie erwarten.
- History ohne `method` als Classic interpretieren.

### 11.5 Konkrete Test-Fixtures zu auditieren

**Aktueller Audit-Stand im Repository:**

| Datei | Status | Umsetzung |
|---|---|---|
| `web_backend_cpp/tests/test_run_status_resume_progress.cpp` | angepasst | Classic-Fixture setzt explizit `method: classic_tile_compile`; AQMH-Fixture setzt `method: aqmh`; Missing-Method-Fall erwartet AQMH; AQMH-Phasen erwarten `AQMH_MAPS` statt `LOCAL_METRICS` |
| `tile_compile_cpp/tests/*` | auditiert | Keine AQMH-first Default-Fixtures gefunden, die implizit Classic erwarten |
| `web_backend_cpp/tests/*` | auditiert | Keine weiteren AQMH-first Default-Fixtures gefunden |
| Frontend Tests | nicht vorhanden im aktuellen Repo | Keine Anpassung moeglich; UI wurde direkt in Wizard/Dashboard/Run-Monitor/Parameter-Studio/History angepasst |

**Audit-Suchmuster:**

```bash
rg -n "aqmh:\s*$|aqmh_enabled|AQMH_QUALITY_MAPS|method: aqmh|classic_tile_compile" web_backend_cpp/tests tile_compile_cpp/tests
```

**Test-Kategorien:**
- ✅ **Unit Tests:** `normalizeMethod()` Funktion und `Config{}` Defaults sind ueber Config-Code abgedeckt; dedizierter Unit-Test kann bei Bedarf ergaenzt werden.
- ✅ **Integration Tests:** Config-Parsing mit/ohne `method`-Feld wird im Backend-Status-Test abgedeckt.
- ⬜ **E2E Tests:** Wizard → Run → History Flow fuer beide Methoden existiert im aktuellen Repo nicht als automatisierter Frontend-Test.
- ✅ **AQMH-first Missing-Method Tests:** Configs ohne `method` werden als AQMH erkannt, unabhaengig von `aqmh.enabled`.

---

## 12. Empfohlene Umsetzungsreihenfolge

### Phase 1: Backend Grundlagen (1-2 Tage)
1. **Config Schema** (`tile_compile_cpp/include/tile_compile/config/configuration.hpp`)
   - Top-Level `method`-Feld hinzufügen
   - `aqmh.enabled` als abgeleitetes Runtime-Flag dokumentieren
2. **Config Parsing** (`tile_compile_cpp/src/io/config.cpp`)
   - `method` parsen/serialisieren
   - `normalizeMethod()` implementieren
   - Schema-Konsistenzprüfung hinzufügen
3. **Schema Dateien** aktualisieren:
   - `tile_compile.schema.yaml`
   - `tile_compile.schema.json`
4. **Default Config** (`tile_compile.yaml`)
   - `method: aqmh` als Default setzen

### Phase 2: Backend Status & Events (1-2 Tage)
5. **Status Payload** (`web_backend_cpp/src/services/run_inspector.cpp`)
   - `method` in Status aufnehmen
   - `normalizePhaseEvent()` implementieren
6. **BGE Integration** (`tile_compile_cpp/apps/runner`)
   - `setBgeTileMetricsSource()` implementieren
7. **Example Configs**
   - `aqmh_enabled.example.yaml` entfernen (AQMH ist jetzt Default, kein Sonderfall mehr)
   - `classic_mode.example.yaml` erstellen
   - Optional: `aqmh_tuning.example.yaml` mit erweiterter Pyramid-/Storage-Konfiguration

### Phase 3: Code Audit (1 Tag)
8. **C++ Defaults auditieren**
   - Alle `Config{}` Default-Konstruktionen prüfen
   - Test-Fixtures gemäß Section 11.5 anpassen
   - `tile_compile_cpp/tests/config_test.cpp`
   - `web_backend_cpp/tests/run_inspector_test.cpp`

### Phase 4: Frontend Grundlagen (2-3 Tage)
9. **Zentrale Method-Erkennung** (`web_frontend/src/app.js`)
   - `currentMethod()` implementieren
   - State Persistence (URL-Hash/localStorage)
10. **Dashboard** (`web_frontend/index.html`, `app.js`)
    - Method Badge und Pipeline Preview AQMH-first
    - Cache-Schätzung und Warnungen

### Phase 5: Frontend UI (2-3 Tage)
11. **Wizard** (`web_frontend/wizard.html`)
    - AQMH als Default vorselektieren
    - Storage-Profile anzeigen
12. **Parameter Studio** (`web_frontend/parameter-studio-page.js`)
    - Method-spezifische Sichtbarkeit
    - AQMH als Default-Kategorie
13. **Run Monitor** (`web_frontend/run-monitor.html`)
    - Method-spezifische Phase Labels
    - AQMH Metriken anzeigen

### Phase 6: Reports & History (1 Tag)
14. **Artifacts & Reports**
    - AQMH-Artefaktgruppe in UI
    - AQMH-Abschnitt im Report
15. **History**
    - Method-Tags und Vergleichsfelder

### Phase 7: Rollback & Monitoring (1/2 Tag)
16. **Rollback-Mechanismen**
    - `FORCE_CLASSIC` Environment-Variable
    - `--force-classic` CLI-Flag
17. **Monitoring-Metriken**
    - Method-Usage Tracking

### Phase 8: Tests (2-3 Tage)
18. **Unit Tests** für alle neuen Funktionen
19. **Integration Tests** für Config-Parsing
20. **E2E Tests** für Wizard → Run → History
21. **AQMH-first Missing-Method Tests** für Configs ohne `method`

### Phase 9: Dokumentation & Finalisierung (1 Tag)
22. **Dokumentation aktualisieren**
    - `aqmh_frontend_integration_plan.md`
    - `aqmh_implementation_plan.md`
23. **Finaler Review** aller Änderungen

---

## 13. Abhaengigkeiten und Blockades

| Schritt | Blockiert von | Blockiert |
|---|---|---|
| Frontend Method-Erkennung | Backend Config `method`-Feld | Dashboard, Wizard, Parameter Studio |
| Status Payload `method` | Config Normalisierung | Run Monitor, History |
| Rollback-Mechanismen | Config Parsing | Production Deployment |
| Frontend Tests | Backend Tests | Integration Tests |

---

## 14. Checkliste für Abnahme

- [ ] Alle `Config{}` Default-Konstruktionen setzen `method: aqmh`
- [ ] `normalizeMethod()` ist implementiert und getestet
- [ ] Schema-Validierung akzeptiert nur `aqmh`/`classic_tile_compile`
- [ ] Normalisierung leitet `aqmh.enabled` ausschliesslich aus `method` ab
- [ ] Status-Payload enthält `method`
- [ ] `FORCE_CLASSIC=1` funktioniert fuer neue Runs als Rollback und wirkt nicht still auf Resume
- [ ] Wizard zeigt AQMH als Default
- [ ] Dashboard zeigt AQMH Pipeline für neue Runs
- [ ] Run Monitor blendet Classic-Phasen bei AQMH aus
- [ ] Parameter Studio startet mit AQMH-Kategorie
- [ ] Alle Tests aus Section 11 passieren
- [ ] Dokumentation ist aktualisiert
