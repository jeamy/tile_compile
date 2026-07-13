# PI Integration Architecture fuer Tile Compile

**Status:** Entwurf  
**Datum:** 2026-07-13  
**Scope:** Ausbau der bestehenden PI Scan-AI zu einer umfassenden Diagnose-,
Optimierungs- und Agenten-Integration fuer Tile Compile.  
**Ausgangspunkt:** PI ist bereits als Parameteroptimierung integriert
(`agent_service`, `web_backend_cpp` AI-Routen, Parameter Studio). 

---

## 1. Zielbild

Tile Compile soll PI nicht nur fuer einzelne Scan-basierte
Parameterempfehlungen nutzen, sondern als sichere, auditierbare
Optimierungsschicht ueber Config, Runs, Reports und Artefakte.

Die Integration wird in drei Capability-Tiers aufgebaut:

1. **PI Assistant (read/explain):** Erklaert Scan-Ergebnisse, Config,
   Run-Status, Reports, Artefakte, Warnungen und Qualitaetsmetriken.
   Read-only.
2. **PI Copilot (propose/preview/apply):** Erstellt validierte
   Config- und Workflow-Vorschlaege, zeigt Diffs und Preview-Ergebnisse,
   Benutzer entscheidet ueber Anwendung.
3. **PI Agent (bounded optimization):** Arbeitet an einem Ziel in einer
   begrenzten Session, erzeugt Config-Revisions, startet erlaubte
   Preview-/Resume-Jobs, bewertet Artefakte, iteriert und schreibt ein
   Journal.

Die wichtigste Regel: **PI mutiert Tile-Compile-Zustand nie direkt.** PI
produziert versionierte Plaene. Das C++ Backend validiert, begrenzt,
previewt und fuehrt aus.

---

## 2. Bestehende Basis

Bereits vorhanden:

- `agent_service/`: Node/TypeScript Sidecar mit
  `@earendil-works/pi-coding-agent`
- `agent_service/src/services/frameAnalysisService.ts`: Prompt-Bau,
  PI-Session, JSON-Antwort, Response-Metadaten
- `web_backend_cpp/src/routes/ai_routes.cpp`: AI-Konfiguration,
  Scan-Analyse, Schema-Export, Recommendation-Validierung, Apply-Flow
- `web_backend_cpp/src/services/ai_service.cpp`: Sidecar-HTTP-Client
- `web_frontend_v3/js/pages/parameter.js`: Parameter Studio mit AI-Tab
- `web_frontend_v3/js/pages/ai-empfehlung.js`: AI-Empfehlungs-UI
- `tile_compile_cpp/tile_compile.schema.json`: autoritaere Config-Schemaquelle
- Run-Artefakte, `run_events.jsonl`, Reports, BGE-/HMS-Preview-Services

Diese Bestandteile bleiben erhalten. Die bestehende
`pi.scan-analysis.v1`-Funktion wird zur ersten konkreten Action-Plan-Quelle
ausgebaut, nicht ersetzt.

---

## 3. Zielarchitektur

```text
web_frontend_v3
  PI Panel / Parameter Studio / Run Monitor / Reports
        |
web_backend_cpp
  PiCenter facade
  PI Tool Registry
  PI Action Plan Validator
  Privacy Gate
  Config Revision Store
  Session Journal
  Run/Artifact/Report Readers
        |
agent_service
  PI SDK sidecar
  outer conversation/provider loop
  no direct mutation authority
        |
tile_compile_cli / tile_compile_runner
  get-schema
  validate-config
  scan-metrics
  preview jobs
  run / resume
```

### 3.1 Verantwortlichkeiten

| Komponente | Verantwortung |
|---|---|
| Frontend | UI, Review, Auswahl, Session-Anzeige, Journal-Ansicht |
| `web_backend_cpp` | Produkt-Autoritaet, Tool Registry, Validation, Privacy, Jobs, Revisions |
| `agent_service` | Modell-/Provider-Lauf, Prompting, Tool-Loop, Streaming |
| `tile_compile_cli` | Schema, Config-Validierung, Hilfsanalysen |
| `tile_compile_runner` | echte Verarbeitung, Runs, Resume, Artefakte |

Der Sidecar darf keine Config-Dateien direkt schreiben, keine Runs direkt
starten und keine Artefakte direkt loeschen oder veraendern.

---

## 4. Nicht verhandelbare Prinzipien

### 4.1 PI-Output ist Daten, keine Mutation

PI erzeugt maschinenlesbare Plaene. Das Backend entscheidet, ob ein Plan
gueltig, erlaubt und anwendbar ist.

Kanonischer Vertrag:

```json
{
  "schema_version": "pi.action-plan.v1",
  "goal": "Improve gradient handling for current IC434 session",
  "confidence": 0.82,
  "actions": [],
  "post_conditions": []
}
```

### 4.2 Das C++ Backend bleibt Produkt-Autoritaet

Alle mutierenden Aktionen laufen ueber bestehende Backend-Pfade:

- Config laden, patchen, validieren und speichern
- Config-Revisions erzeugen
- Jobs starten, stoppen und inspizieren
- Run-IDs vergeben
- Artefakte lesen

### 4.3 Validierung ist mehrstufig

1. Schema-Pruefung gegen `tile_compile.schema.json`
2. Pfad-Allowlist fuer AI-aenderbare Config-Pfade
3. Typ-/Enum-/Min-/Max-Pruefung
4. `tile_compile_cli validate-config --stdin`
5. Optional: Preview-/Resume-Verifikation

### 4.4 Privacy ist ein Runtime-Gate

Provider-Requests gehen nur durch einen zentralen Privacy Gate. Prompts duerfen
nicht beilaufig Pfade, Secrets oder Bilddaten enthalten.

### 4.5 Autonomie ist explizit, budgetiert und reversibel

Agenten-Sessions brauchen:

- Autonomie-Level
- Zeit-, Schritt-, Token-, Kosten- und Run-Budget
- erlaubte Action-Typen
- isolierte Run-IDs
- Config-Revisions
- append-only Journal

---

## 5. Action Plan Contract

### 5.1 `pi.action-plan.v1`

```json
{
  "schema_version": "pi.action-plan.v1",
  "goal": "Optimize current dataset for lower background gradient",
  "summary": "BGE should use a more robust model and PCC should be rerun.",
  "confidence": 0.84,
  "actions": [
    {
      "id": "set_bge_model",
      "type": "config.set",
      "path": "bge.model",
      "value": "rbf",
      "rationale": "Measured sky_gradient is high."
    },
    {
      "id": "preview_bge",
      "type": "preview.bge",
      "inputs": {
        "config_revision": "proposed"
      }
    }
  ],
  "post_conditions": [
    {
      "type": "config.valid"
    },
    {
      "type": "metric.improves",
      "metric": "sky_gradient",
      "direction": "down"
    }
  ],
  "warnings": []
}
```

### 5.2 Erlaubte Action-Typen in der ersten Ausbaustufe

Read-only:

- `config.read`
- `config.schema.read`
- `scan.read`
- `scan.metrics.read`
- `run.status.read`
- `run.events.read`
- `artifact.list`
- `artifact.summary.read`
- `report.summary.read`

Proposal:

- `config.set`
- `config.patch`
- `config.diff.preview`
- `preview.bge.plan`
- `preview.hms.plan`
- `run.resume.plan`

Apply:

- `config.revision.create`
- `config.patch.apply`
- `preview.bge`
- `preview.hms`
- `run.start`
- `run.resume`
- `run.cancel`

Gefaehrliche Aktionen wie Datei-Loeschung, Ueberschreiben bestehender Runs,
Loeschen von Artefakten oder freies Shell-Ausfuehren bleiben nicht Teil der
PI-Toolflaeche.

---

## 6. Tool Registry

Tile Compile braucht eine einzige Tool-Oberflaeche fuer UI,
Sidecar, Agent-Sessions und spaeter MCP.

Backend-Routen:

```text
GET  /api/pi/tools
POST /api/pi/tools/call
POST /api/pi/action-plans/validate
POST /api/pi/action-plans/preview
POST /api/pi/action-plans/apply
POST /api/pi/sessions
GET  /api/pi/sessions/:id
GET  /api/pi/sessions/:id/journal
POST /api/pi/sessions/:id/cancel
```

Tool-Definitionen tragen Versionen:

```json
{
  "name": "config.patch.validate",
  "tool_version": "1.0.0",
  "min_autonomy_level": "L1",
  "input_schema": {},
  "output_schema": {}
}
```

Die Registry erzwingt Autonomie-Level, Privacy-Modus, Budget und Allowlist.
Prompts duerfen diese Regeln wiederholen, aber nicht durchsetzen. Die Registry
ist das Gate.

---

## 7. Autonomie-Level

```text
L0 ReadOnly
  PI darf lesen und erklaeren.
  Keine Config-Patches, keine Jobs.

L1 Propose
  PI darf Action Plans und Config-Diffs vorschlagen.
  Benutzer muss anwenden.

L2 ApplyWithReview
  PI darf validierte Config-Revisions und Preview-Jobs ausfuehren.
  Run-Starts und laengere Resume-Jobs brauchen Review.

L3 BoundedAgent
  PI darf innerhalb eines expliziten Budgets iterieren:
  Config-Revision -> Preview/Resume -> Artefaktbewertung -> naechste Aktion.
  Nur isolierte Run-IDs, kein In-place-Ueberschreiben.
```

Empfohlener Produktpfad:

1. L0 und L1 stabil produktiv machen.
2. L2 als experimentelle Option hinter Feature-Flag.
3. L3 erst nach Journal, Revisions, Verifikation und Evals.

---

## 8. Memories fuer sessionuebergreifendes Lernen

Sessionuebergreifende Memories sind sinnvoll, aber sie duerfen nicht als
unkontrolliertes Prompt-Gedaechtnis implementiert werden. Tile Compile braucht
eine strukturierte, validierte Memory-Schicht, die aus echten Ergebnissen
lernt und jederzeit auditierbar bleibt.

### 8.1 Ziel

PI soll aus erfolgreichen Optimierungen lernen:

- Welche Config-Aenderungen haben fuer eine bestimmte Kamera/Zielklasse
  geholfen?
- Welche BGE-/PCC-/AQMH-Strategien waren bei starkem Hintergrundgradienten
  stabil?
- Welche Parameter waren fuer bestimmte Smart-Telescope-Daten schlecht?
- Welche Resume-Strategien haben Zeit gespart?
- Welche Warnungen waren spaeter tatsaechlich relevant?

### 8.2 Memory-Typen

| Typ | Inhalt | Beispiel |
|---|---|---|
| Dataset Memory | Eigenschaften eines Datenbestands | Kamera, Belichtung, Framezahl, Zielklasse, Metrikprofil |
| Optimization Memory | getestete Config-Aenderung + Ergebnis | `bge.model=rbf` reduzierte `sky_gradient` um 31% |
| Preset Memory | wiederverwendbare Startkonfiguration | "Seestar extended nebula high-gradient" |
| Failure Memory | bekannte schlechte Strategie | `classic` BGE bei IC434-aehnlichen Gradienten verworfen |
| User Preference Memory | explizite Benutzerpraeferenzen | konservative Entrauschung, keine langen Agent-Runs |

### 8.3 Speicherort

Empfohlen:

```text
runs/.pi_memory/
  memories.jsonl
  embeddings.sqlite          optional spaeter
  indexes/
  rejected_memories.jsonl
```

Alternativ fuer installationsweite Memories:

```text
runtime/pi_memory/
```

Projektweite Memories sind fuer Tile Compile sinnvoller als globale Memories,
weil Kamera, Optik, Standort, Targets und Nutzerpraeferenzen stark
projektspezifisch sind.

### 8.4 Memory-Schema

```json
{
  "schema_version": "pi.memory.v1",
  "memory_id": "mem_20260713_143522_a91c",
  "type": "optimization",
  "created_at": "2026-07-13T14:35:22Z",
  "source_session_id": "pi_sess_20260713_140102",
  "source_run_ids": ["ic434_pi_agent_step1", "ic434_pi_agent_step2"],
  "dataset_fingerprint": {
    "camera": "Seestar S50",
    "color_mode": "osc",
    "frame_count_bucket": "100-300",
    "target_class": "emission_nebula",
    "gradient_bucket": "high",
    "fwhm_bucket": "medium",
    "image_size": "1920x1080"
  },
  "condition": {
    "metrics": {
      "sky_gradient": ">=0.05",
      "star_count": ">=50"
    }
  },
  "recommendation": {
    "patch": [
      {
        "path": "bge.model",
        "value": "rbf"
      }
    ],
    "rationale": "RBF BGE performed better on similar high-gradient nebula data."
  },
  "evidence": {
    "before_metrics": {},
    "after_metrics": {},
    "delta": {
      "sky_gradient": -0.31
    },
    "validation": "run_completed_no_critical_errors"
  },
  "confidence": 0.74,
  "status": "candidate",
  "privacy_class": "metadata_only"
}
```

### 8.5 Memory-Lifecycle

Memories sollten nicht automatisch als Wahrheit gelten. Empfohlener Ablauf:

1. **Candidate:** Agent oder Backend erzeugt Memory aus einer Session.
2. **Validated:** Post-Conditions und Run-Metriken bestaetigen Nutzen.
3. **Accepted:** Benutzer akzeptiert Memory oder sie wird mehrfach bestaetigt.
4. **Deprecated:** Memory wird durch spaetere Gegenbeispiele geschwaecht.
5. **Rejected:** Memory war falsch, unsicher oder nicht generalisierbar.

Nur `validated` und `accepted` Memories duerfen in normale PI-Kontexte
einfliessen.

### 8.6 Memory-Gates

Memories duerfen nie direkt Config aendern. Sie duerfen nur:

- Kontext fuer den Assistant liefern
- Prioritaeten fuer Vorschlaege beeinflussen
- Action Plans mit Evidenz ergaenzen
- Presets als Vorschlag anbieten

Jeder daraus abgeleitete Patch laeuft wieder durch Schema-, Config- und
Action-Plan-Validierung.

### 8.7 Retrieval

Fuer den ersten Schritt reicht regelbasiertes Retrieval ohne Embeddings:

- Zielklasse
- Kamera/Instrument
- Farbmodus
- Framezahl-Bucket
- Gradient-Bucket
- FWHM-/Noise-Bucket
- vorhandene Warnungen
- relevante Config-Kategorie

Spaeter kann ein lokaler Embedding-Index ergaenzt werden. Embeddings muessen
lokal berechenbar oder separat privacy-gated sein. Cloud-Embeddings duerfen
nicht implizit aus Run-Daten erzeugt werden.

### 8.8 Negative Memories

Negative Memories sind wichtig, weil sie Fehloptimierungen verhindern:

```json
{
  "type": "failure",
  "condition": {
    "target_class": "emission_nebula",
    "gradient_bucket": "high"
  },
  "avoid": {
    "patch": [
      {
        "path": "bge.model",
        "value": "classic"
      }
    ]
  },
  "evidence": {
    "reason": "Residual gradient increased and PCC warning count rose."
  }
}
```

Negative Memories duerfen Vorschlaege nicht hart blockieren, aber sie sollten
Review erzwingen und im Diff sichtbar sein.

### 8.9 UI fuer Memories

Im PI Panel sollte es eine Memory-Ansicht geben:

- "Aus dieser Session lernen" Checkbox
- Liste neuer Candidate Memories
- Evidence-Diff: vorher/nachher Metriken
- Accept / Reject / Ignore
- Memory-Quelle: Run-ID, Session-ID, Datum
- Privacy-Klasse
- "Bei aehnlichen Daten anwenden" als explizite Option

Default: keine automatische globale Speicherung ohne Benutzerkontrolle.

---

## 9. Privacy-Modi

```text
LocalOnly
  Keine Provider-Aufrufe. Nur lokale Regeln, gespeicherte Memories,
  MockProvider oder spaeter lokale Modelle.

MetadataOnly
  Schema, Config, aggregierte Metriken, Reports ohne Pfade.

PathsAllowed
  Pfade duerfen gesendet werden, aber keine Bilddaten.

ArtifactsAllowed
  ausgewaehlte kleine JSON-/Text-Artefakte erlaubt.

ImageDataAllowed
  nur explizit pro Job. Kein Default.
```

Empfohlener Default: `MetadataOnly`.

Memory-Eintraege tragen eine `privacy_class`. Ein Memory mit hoeherer
Privacy-Klasse darf nicht in einen niedrigeren Privacy-Kontext einfliessen.

---

## 10. Verifikation

PI-Agenten duerfen Optimierungserfolg nicht nur sprachlich behaupten. Das
Backend braucht maschinenlesbare Post-Conditions:

- `config.valid`
- `run.completed`
- `phase.no_critical_errors`
- `artifact.exists`
- `metric.improved`
- `registration.success_rate_above`
- `pcc.solution_valid`
- `bge.gradient_reduced`
- `hms.preview_not_clipped`

Beispiel:

```json
{
  "type": "metric.improved",
  "metric": "sky_gradient",
  "direction": "down",
  "min_relative_delta": 0.10
}
```

Wenn Post-Conditions nicht pruefbar sind, bleibt die Empfehlung ein
unverifizierter Vorschlag und darf keine Memory mit `validated` Status
erzeugen.

---

## 11. Agent Sessions

Session-Vertrag:

```json
{
  "schema_version": "pi.agent-session.v1",
  "goal": "Optimize IC434 background gradient and color calibration",
  "autonomy_level": "L3",
  "privacy_mode": "MetadataOnly",
  "budgets": {
    "max_steps": 8,
    "max_runs": 3,
    "max_minutes": 45,
    "max_tokens": 120000
  },
  "allowed_actions": [
    "config.patch.apply",
    "preview.bge",
    "run.resume"
  ]
}
```

Jede Session schreibt ein append-only Journal:

- Session-Ziel und Budgets
- verwendeter Kontext und Hashes
- Memory-Retrieval-Ergebnisse
- Tool Calls
- Action Plans
- Validation-Resultate
- Config-Diffs
- Run-IDs
- Artefakt-/Metrikvergleiche
- erzeugte Candidate Memories
- Abbruchgruende

Agent-Runs bekommen immer isolierte Run-IDs:

```text
ic434_pi_agent_20260713_143000_step1
ic434_pi_agent_20260713_143000_step2
```

Bestehende Runs werden nicht ueberschrieben.

---

## 12. Code-Struktur

Empfohlene Zielstruktur im Backend:

```text
web_backend_cpp/include/services/pi/
  pi_center.hpp
  pi_action_plan.hpp
  pi_action_validator.hpp
  pi_tool_registry.hpp
  pi_privacy_gate.hpp
  pi_session_store.hpp
  pi_journal.hpp
  pi_memory_store.hpp
  pi_context_builder.hpp

web_backend_cpp/src/services/pi/
  pi_center.cpp
  pi_action_validator.cpp
  pi_tool_registry.cpp
  pi_privacy_gate.cpp
  pi_session_store.cpp
  pi_journal.cpp
  pi_memory_store.cpp
  pi_context_builder.cpp

web_backend_cpp/src/routes/pi_routes.cpp
web_backend_cpp/include/routes/pi_routes.hpp
```

`ai_routes.cpp` kann schrittweise migriert werden. Die bestehenden
`/api/ai/*` und `/api/scan/analysis*` Routen bleiben als Kompatibilitaet
erhalten und nutzen intern spaeter `PiCenter`.

Sidecar-Zielstruktur:

```text
agent_service/src/services/
  frameAnalysisService.ts        bestehend
  piToolClient.ts                neu
  piSessionService.ts            neu
  piMemoryContextService.ts      optional, nur Kontext; Backend bleibt Autoritaet
```

---

## 13. Phasenplan

### Phase 0: Bestehende Scan-AI haerten

- `pi.scan-analysis.v1` intern auf `pi.action-plan.v1` abbilden.
- Recommendation-Validierung aus `ai_routes.cpp` in `PiActionValidator`
  extrahieren.
- Prompt-/Traffic-Logging mit Secret-/Path-Redaction absichern.
- Tests fuer Action-Plan-Parsing, Schema-Fehler und Apply-Flow ergaenzen.

### Phase 1: PI Assistant Read-only

- `PiContextBuilder` fuer Config, Schema, Scan, Metriken, Run-Status,
  Reports und Artefakte.
- Read-only Tools in `PiToolRegistry`.
- PI Panel im Run Monitor und Parameter Studio.
- Fragen zu Config, Fehlern, Warnungen, Phasen und Artefakten beantworten.

### Phase 2: Copilot mit Preview/Apply

- `pi.action-plan.v1` produktiv einfuehren.
- Config-Diff-Preview im Frontend.
- Config-Revisions und Rollback.
- BGE-/HMS-Preview als validierte Tool-Aktionen.
- Apply nur nach Benutzerreview.

### Phase 3: Memory Store

- `pi.memory.v1` JSONL Store.
- Candidate Memories aus abgeschlossenen Sessions erzeugen.
- Memory-Review UI.
- Regelbasiertes Retrieval fuer aehnliche Datensaetze.
- Negative Memories und Deprecation-Status.

### Phase 4: Verifikation und Evals

- Post-Condition-Engine im Backend.
- Vergleich von Run-Artefakten und Metriken.
- Golden Evals fuer:
  - read-only Q&A
  - Config-Patch-Validitaet
  - Preview-Erfolg
  - Memory-Retrieval
  - Rollback/Revisions

### Phase 5: Bounded Agent

- L2/L3 Agent Sessions mit Budgets.
- Isolierte Run-IDs.
- Journal UI.
- Resume-/Preview-Iterationen.
- Candidate Memories aus nachweislich erfolgreichen Agent-Schritten.

### Phase 6: MCP / externe Agenten

- Lokaler MCP-Transport fuer dieselbe Tool Registry.
- Keine neuen Capabilities gegenueber UI.
- Gleiche Privacy-, Budget-, Validation- und Journal-Gates.

---

## 14. Empfohlene erste Umsetzung

Die naechste konkrete Umsetzung sollte klein bleiben:

1. Neues Backend-Modell `PiActionPlan` und Validator anlegen.
2. Bestehende Scan-AI-Empfehlungen in Action Plans konvertieren.
3. `PiMemoryStore` als JSONL Store mit `candidate` Memories vorbereiten.
4. Beim Apply erfolgreicher Empfehlungen optional eine Candidate Memory
   erzeugen.
5. Frontend zeigt "Aus dieser Optimierung lernen" mit Review an.

Damit entsteht sofort der Kern der groesseren Architektur: Plaene,
Validierung, Revisions und kontrolliertes Lernen, ohne schon volle Autonomie
freizuschalten.
