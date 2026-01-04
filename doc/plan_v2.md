# Plan v2 – Siril‑Pfad (Methodik v3) Integration

**Ziel:** Rückkehr zu **Siril‑basierter Registrierung** (Pfad A aus Methodik v3) als stabiler Produktionspfad. Ab **Phase 2 / Tile‑Erzeugung** soll die Pipeline strikt der **Methodik v3** folgen und **kanalweise** arbeiten.

**Quellen (bewertet, siehe §1):**
- `doc/tile_basierte_qualitatsrekonstruktion_methodik_v_3.md`
- `doc/methodik_v_3_integration_mapping.md`
- `doc/tile_compile_minimal_run_testmode.md`

**Siril Scripts (Implementierungsbasis):**
- `siril_scripts/siril_register_osc.ssf`
- `siril_scripts/siril_channelsplit.ssf`
- `siril_scripts/siril_stack_average.ssf`
- `siril_scripts/siril_recombine.ssr`
- optional/diagnostisch: `siril_scripts/siril_stack_schannels_linear.ssr`

---

## 1. Dokumentenbewertung (Normativität, Rolle, Konfliktregel)

### 1.1 Bewertungsmatrix

| Dokument | Status | Rolle im Projekt | Normativität | Wenn Konflikt: gewinnt |
|---|---:|---|---:|---|
| `doc/tile_basierte_qualitatsrekonstruktion_methodik_v_3.md` | Normative Spezifikation | Single Source of Truth für Pipeline‑Semantik (A/B‑Pfad, Invarianten, Kernphasen ab Phase 3) | **hoch** | dieses Dokument |
| `doc/methodik_v_3_integration_mapping.md` | Integrations‑Ableitung | Abgleich Methodik ↔ Konfiguration ↔ Backend/GUI (PhaseEnum) | **mittel** | Methodik v3 |
| `doc/tile_compile_minimal_run_testmode.md` | Normative Ableitung | Deterministischer Minimal‑Run für Entwicklung/CI, reduziert aber semantisch identisch | **hoch (für Testmodus)** | Methodik v3 (Produktionssemantik), sonst dieses Dokument |
| `siril_scripts/*.ssf/*.ssr` | Implementierungsartefakte | Konkrete Siril‑Ausführung (Debayer/Registration/ChannelSplit/Stack/Recombine) | **niedrig** | Methodik/Policy |

### 1.2 Konfliktregel (kurz)

1. Methodik v3 ist die oberste normative Quelle.
2. Testmodus‑Dokument ist normativ **nur** für `pipeline.mode: test` (Minimal‑Run).
3. Mapping‑Dokument darf Methodik präzisieren, aber nicht ändern.
4. Siril‑Scripts sind austauschbar; wenn sie der Policy widersprechen, werden sie angepasst oder nicht verwendet.

---

## 2. Nicht‑verhandelbare Invarianten (Abbruch bei Verstoß)

Aus Methodik v3 (verbindlich):
- Daten sind **linear** (kein Stretch, keine nichtlinearen Operatoren)
- **keine Frame‑Selektion** (alle registrierten Frames gehen in die Methodik)
- Verarbeitung ab Kanaltrennung **kanalgetrennt**
- Pipeline ist **streng linear** (keine Rückkopplungen)

Siril‑Policy für diesen Projektpfad:
- Siril darf **nur** übernehmen:
  - Debayer (für OSC)
  - Registrierung (Sternfindung + Transformationsschätzung + Warp)
  - ggf. rein lineares **Summen‑Stacking** als primitive (ohne Rejection/Weighting/Norm)
- Siril darf **nicht**:
  - Selection/Unselection
  - Rejection, Weighting
  - nichtlineare Operationen
  - Drizzle (außer explizit `=0`)

---

## 3. PhaseEnum (Backend/GUI) – Zielzustand

Gemäß `doc/methodik_v_3_integration_mapping.md`:

```text
SCAN_INPUT
REGISTRATION
CHANNEL_SPLIT
NORMALIZATION
GLOBAL_METRICS
TILE_GRID
LOCAL_METRICS
TILE_RECONSTRUCTION
STATE_CLUSTERING
SYNTHETIC_FRAMES
STACKING
DONE
FAILED
```

---

## 4. Integrationsplan (Phasen, Inputs/Outputs, Artefakte)

### Phase SCAN_INPUT

**Input:** rohe Lights (OSC)

**Checks (hard‑fail):**
- linear_required
- ausreichende Framezahl (frames_min)
- FITS Header Plausibilität (Dimensionen, bitpix)
- CFA/OSC Konsistenz + Bayer‑Pattern Konsistenz

**Artefakte:**
- `frames_manifest.json` (vollständige Liste)
- `run_metadata.json` (tools, versionen)

---

### Phase REGISTRATION (Siril)

**Zweck:** Debayer + Registrierung (Siril), Ergebnis sind **registrierte, debayerte RGB‑Frames** (Methodik v3 A.2.1).

**Script:** `siril_scripts/siril_register_osc.ssf`

**Runner‑Verhalten (Soll):**
- Workdir: `work/registration/`
- Materialisierung: `seq00001.fit ...` (symlink/copy)
- Siril call: `siril -d <workdir> -s <script>` (mit `-q` Retry falls verfügbar)
- Script‑Policy‑Validierung + Hash Logging (SHA256)

**Output (Soll):**
- registrierte Frames als RGB‑FITS (Siril erzeugt i.d.R. `r_<seqname>*`)
- Copy nach: `outputs/registered_rgb/` (oder konsistent benannter Ordner)

**Artefakte:**
- `artifacts/siril_registration.log`
- Script Hash/Path im Run‑Log

**Offene Klärung/Standardisierung (wichtig):**
- Exakte Siril Output‑Namen müssen **einheitlich** gesammelt werden (z.B. `r_lights*.fit*`).

---

### Phase CHANNEL_SPLIT (Siril)

**Zweck:** RGB → R/G/B nach Registrierung (Methodik v3 A.2.2). Ab hier keine kanalübergreifenden Operationen.

**Script:** `siril_scripts/siril_channelsplit.ssf`

**Runner‑Verhalten (Soll):**
- Workdir: `work/channel_split/`
- Inputs: registrierte RGB‑Frames (aus Phase REGISTRATION)
- Siril split erzeugt Sequenzen/Frames pro Kanal

**Output (Soll):**
- `outputs/channels/R/` Frames
- `outputs/channels/G/` Frames
- `outputs/channels/B/` Frames

**Artefakte:**
- `artifacts/siril_channelsplit.log`

**Offene Klärung/Standardisierung:**
- Welche Dateinamen/Seq‑Struktur erzeugt `split` genau (z.B. `reg_R*`, `reg_G*`, `reg_B*`). Das muss der Runner zuverlässig finden.

---

### Phase NORMALIZATION (Python)

**Zweck:** Globale lineare Normalisierung (Pflicht, exakt einmal; Methodik v3 §3.1), getrennt pro Kanal.

**Konfig:** `normalization.enabled/mode/per_channel`

**Output (Soll):**
- normalisierte Frames pro Kanal: `outputs/norm/R/*.fit`, `outputs/norm/G/*.fit`, `outputs/norm/B/*.fit`

**Artefakte:**
- Statistik pro Frame/Kanal (Bf, scaling)

---

### Phase GLOBAL_METRICS (Python)

**Zweck:** Pro Frame/Kanal globale Metriken (B, σ, Gradientenergie) + globaler Qualitätsindex (Methodik v3 §3.2).

**Konfig:** `global_metrics.weights`, `global_metrics.clamp`

**Output (Soll):**
- `outputs/metrics/global_metrics.json`

---

### Phase TILE_GRID (Python)

**Zweck:** Fixes Tile‑Raster (Methodik v3 §3.3).

**Konfig:** `tile.*`

**Output (Soll):**
- `outputs/tiles/tile_grid.json`

---

### Phase LOCAL_METRICS (Python)

**Zweck:** Tile‑lokale Metriken pro Frame/Kanal (Methodik v3 §3.4).

**Konfig:** `local_metrics.*`

**Output (Soll):
- `outputs/metrics/local_metrics.(json|npz)`

---

### Phase TILE_RECONSTRUCTION (Python)

**Zweck:** Tile‑basierte Rekonstruktion kanalweise (Methodik v3 §3.6).

**Output (Soll):**
- Rekonstruiertes Bild pro Kanal (oder pro Zustand/Cluster):
  - `outputs/recon/R_recon.fit`
  - `outputs/recon/G_recon.fit`
  - `outputs/recon/B_recon.fit`

---

### Phase STATE_CLUSTERING (optional, Python)

**Zweck:** Zustandsbasierte Clusterung (Methodik v3 §3.7).

**Output (Soll):**
- Clusterzuordnung pro Frame/Kanal

---

### Phase SYNTHETIC_FRAMES (optional, Python)

**Zweck:** Synthetische Frames generieren (Methodik v3 §3.8).

**Output (Soll):**
- `outputs/synthetic/R/syn_*.fit`
- `outputs/synthetic/G/syn_*.fit`
- `outputs/synthetic/B/syn_*.fit`

---

### Phase STACKING (Siril, kanalweise)

**Zweck:** Finales lineares Stacking pro Kanal (Methodik v3 §3.8). Kein Drizzle, keine Gewichtung, keine Selektion.

**Wichtig:** Siril‑`mean` impliziert in 1.4 Rejection‑Parameter → vermeiden.

**Empfohlene Primitive:**
- Siril: `stack <seq> sum -out=<file>`
- Python: Division durch `N` (lineares Mittel) falls „average“ benötigt wird

**Script (Basis):** `siril_scripts/siril_stack_average.ssf`

**Soll‑Erweiterung (für v3):**
- Drei separate Stacking‑Aufrufe (R/G/B), jeweils auf die synthetischen Frames des Kanals.
- Alternativ: ein dediziertes Siril‑Script „stack_channel.ssf“ parametrisiert über Workdir/Inputs (statisch geht nur über Ordnerkonventionen).

**Output (Soll):**
- `outputs/final/stack_R.fit`
- `outputs/final/stack_G.fit`
- `outputs/final/stack_B.fit`

---

## 5. Out‑of‑Scope / Nachgelagert (explizit)

Gemäß Methodik v3:
- RGB/LRGB‑Kombination ist **außerhalb** der Rekonstruktionsmethodik.

Optionaler Siril‑Step (nachgelagert):
- `siril_scripts/siril_recombine.ssr` (`rgbcomp stack_R.fit stack_G.fit stack_B.fit stacked_RGB.fit`)

---

## 6. Kurzfristiger Migrationspfad (1–2 Iterationen)

### Schritt 1: Sofort zurück zu Siril‑Registration
- `tile_compile.yaml`: `registration.engine: siril`
- Sicherstellen, dass Default‑Scripts aus `siril_scripts/` gefunden werden

### Schritt 2: Siril CHANNEL_SPLIT als echte Phase integrieren
- Runner: nach REGISTRATION ein eigenes Workdir + Script‑Run
- Outputs pro Kanal sauber materialisieren

### Schritt 3: Minimal‑Baseline für Validierung
- Ohne Methodik‑Phasen: nur
  - REGISTRATION → CHANNEL_SPLIT → (optional) kanalweises Siril‑Summenstack
- Zweck: Referenzbilder erzeugen, um Rekonstruktionsphasen später gegen Siril‑Baseline zu vergleichen.

---

## 7. Minimal‑Run (Testmodus) – verbindlicher Integritätstest

**Quelle:** `doc/tile_compile_minimal_run_testmode.md` (normativ für Testmodus)

### 7.1 Zweck

Der Minimal‑Run ist **nicht** zur Bildqualität‑Optimierung da, sondern zur Validierung, dass:
- alle Phasen korrekt verkettet sind
- Gewichte/Tiles/Rekonstruktion numerisch stabil sind
- keine methodisch verbotenen Effekte auftreten (Selektion, Nichtlinearität, dynamische Tile‑Änderungen)

### 7.2 Verbindliche Reduktionen gegenüber Produktion

- **Frames:** 32–64
- **Kanäle:** nur **G** (RGB‑Probleme sind kein Gegenstand des Minimal‑Runs)
- **Registrierungspfad:** **muss Siril** sein (keine experimentellen Backends)
- **Tile‑Geometrie:** fix 64px, Overlap 25%
- **Metriken:** aktiv B_f, σ_f, lokale FWHM; deaktiviert Gradientenergie/Strukturmetriken
- **Clusterung/Synthetics:** deaktiviert; direktes finales Stacking

### 7.3 Minimal‑Pipeline (Testmodus, normativ)

Im Testmodus müssen diese Phasen in dieser Reihenfolge laufen (keine Übersprünge):

1. SCAN_INPUT
2. REGISTRATION (Siril)
3. CHANNEL_SPLIT (RGB → G)
4. NORMALIZATION
5. GLOBAL_METRICS (B_f, σ_f)
6. TILE_GRID (fix 64px)
7. LOCAL_METRICS (FWHM)
8. TILE_RECONSTRUCTION
9. STACKING (lineares Mittel)
10. DONE

### 7.4 Erfolgskriterien (Testmodus)

- alle Phasen ohne Abbruch
- keine NaN/Inf
- ΣW_f,t > 0 für ≥ 95% der Tiles
- keine harten Tile‑Übergänge
- Re‑Run mit identischen Daten liefert **bitgleich** identisches Ergebnis

---

## 8. Validierung / Akzeptanzkriterien

**Registrierung:**
- Sterne in `outputs/registered_rgb/` sind punktförmig (keine Drift/Schlieren)

**Kanaltrennung:**
- R/G/B Frames existieren und sind geometrisch deckungsgleich

**Linearität/Policy:**
- Script‑Policy‑Validator findet keine verbotenen Befehle/Optionen
- Kein Frame‑Drop (Anzahl Frames bleibt konstant bis zur optionalen Clusterung)

**Reproduzierbarkeit:**
- Skript‑SHA256, Config‑Hash, Frame‑Manifest werden persistiert

---

## 9. Notizen zu bestehenden Scripts

- `siril_register_osc.ssf` nutzt aktuell:
  - `link . -out=lights`
  - `convert lights -debayer`
  - `register lights`

  **TODO (bei Problemen):** Falls Siril ein anderes Seq‑Naming benötigt, muss das Script/Runner‑Materialisierung konsistent angepasst werden (z.B. `link seq -out=lights` + vorherige `seq*.fit` Materialisierung).

- `siril_channelsplit.ssf` erwartet `r_lights` als Inputseq. Das setzt voraus, dass REGISTRATION genau so benennt. Sonst Runner/Script anpassen.

- `siril_stack_schannels_linear.ssr` ist nützlich als **diagnostischer Siril‑End‑to‑End‑Stack** (ohne Methodik‑Phasen), aber nicht der finale v3‑Stacking‑Step.

---

## 10. GUI v2 (Python/Qt6) – ersetzt Tauri im Root

**Ziel:** Eine lokale Desktop‑GUI in **Python/Qt6**, die funktional/UX‑seitig der bisherigen Tauri‑GUI (`gui-tauri-legacy/`) entspricht.

### 10.1 Funktionsumfang (ident zu Legacy)

Screens/Sections:
- Scan (Input dir, frames_min, Ergebnis‑Summary, ggf. Color‑Mode‑Confirm)
- Configuration (Config‑Path, YAML‑Editor, Load/Save/Validate)
- Run (WorkingDir, Config, InputDir, RunsDir, Pattern, DryRun, Start/Abort)
- Current run (run_id, run_dir, Status, Logs tail/filter, Artifacts)
- Run history (List runs + selection)
- Live log (stdout/stderr der Runner‑Session)

### 10.2 Backend‑Anbindung (transport‑neutral)

Die GUI spricht den Backend‑Funktionsumfang nicht via WebView/IPC, sondern über:

- `tile_compile_backend_cli.py` (bzw. `tile-compile-backend`) für:
  - scan
  - load-config / save-config
  - validate-config
  - list-runs / get-run-status / get-run-logs / list-artifacts
- `tile_compile_runner.py` als Subprocess für:
  - Start Run
  - Live‑Log Anzeige (stdout/stderr)
  - Abort via Signal (SIGINT/SIGTERM)

### 10.3 Commands/Endpoints zentral in `constants.js`

**Regel:** Alle GUI→Backend Kommandos/Endpunkte werden zentral in `gui/constants.js` definiert.

Die Qt‑GUI liest diese Datei und nutzt sie als einzige Quelle für:
- CLI‑Command‑Namen und Subcommands
- Runner‑Invocation

### 10.4 Styling (extern)

**Regel:** GUI‑Styling liegt in einer externen Datei `gui/styles.qss`.

### 10.5 Persistenz

Die GUI nutzt (wie Legacy) `tile_compile_gui_state.json` im Repo‑Root:
- `lastInputDir`
- optional `sirilExe`

### 10.6 Entfernen von Tauri aus dem Root

- Alle Tauri‑Artefakte bleiben ausschließlich unter `gui-tauri-legacy/`.
- Root‑Startskripte werden auf Qt6 umgestellt (keine npm/cargo Abhängigkeit).
