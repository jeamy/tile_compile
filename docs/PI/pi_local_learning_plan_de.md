# PI Local Learning — Design-Dokument

> **Status:** Proposal / Design Document
> **Datum:** 2026-08-25
> **Modul:** `web_backend_cpp` (`services/pi/*`), `agent_service`, `web_frontend_v3`
> **Betrifft:**
> - `web_backend_cpp/src/services/pi/pi_memory_store.cpp` (Memory-Store, Review-Workflow)
> - `web_backend_cpp/src/services/pi/pi_ai_request_builder.cpp` (Prompt-Kontext, Memory-Injektion)
> - `web_backend_cpp/src/services/pi/pi_context_builder.cpp`
> - `web_backend_cpp/src/services/pi/pi_live_image_session.cpp` / `pi_image_ops.cpp` (Live Editor)
> - `agent_service/src/services/frameAnalysisService.ts` (Scan-Analyse → `action_plan`)
> - `agent_service/src/services/liveImageChatService.ts` (Live Image Chat → Bild-Operationen)
> - Verwandte Docs: [`pi_ki_empfehlungen_de.md`](pi_ki_empfehlungen_de.md), [`pi_live_image_chat_plan.md`](pi_live_image_chat_plan.md), [`pi_context_protocol_compression_plan_de.md`](pi_context_protocol_compression_plan_de.md)

---

## 1. Ausgangslage

PI hat heute zwei getrennte, aber strukturell identische KI-gestützte Empfehlungspfade:

| | Scan-Analyse (`frameAnalysisService`) | Live Image Chat (`liveImageChatService`) |
|---|---|---|
| Eingabe | `scan_metrics` (Aggregate, Diagnostics, Registration, Sky-Gradient …), `base_config`, `config_schema`, `positive_/negative_memories` | JPEG-Vorschau (Vision) + Freitext-Kommando + `operation_history` |
| Ausgabe | `action_plan` mit `{path, value, confidence, rationale, evidence, risk}` gegen `config_schema` (`bge`, `pcc`, `registration`, `normalization`, `hypermetric_stretch`, `chroma_denoise`, `tile_denoise`, …) | `{operations: [...], adjustable, adjust_step, repeatable}` gegen die `pi_image_ops`-Operationsliste (`brightness`, `contrast`, `sharpen`, `denoise`, `vibrance`, `dehaze`, …) |
| Entscheidungslogik | Zero-Shot-LLM-Urteil pro Aufruf, neu aus dem JSON-Kontext hergeleitet | Zero-Shot-Vision-LLM-Urteil pro Chat-Turn, aus dem Bild "erraten" |
| Lernen aus Vergangenheit | Nur über `pi_memory_store`: Keyword-/Feld-Overlap-Retrieval (`context_match_score`), Freigabe nur nach manuellem `/review` | **keins** — jeder Turn startet bei null, `operation_history` dient nur als Text-Kontext, nicht als Trainingssignal |
| Kosten pro Anfrage | 18–35 KB JSON + Prompt bis 120 KB extern an den Provider | Volles Bild + Prompt extern an den Provider, bei jedem Chat-Turn erneut |

Beide Pfade haben dasselbe Grundproblem: **es wird bei jeder einzelnen Anfrage neu geraten**, statt aus der Menge bereits gemachter, vom Nutzer akzeptierter oder verworfener Empfehlungen zu lernen. Der einzige vorhandene Lernmechanismus — `pi_memory_store` — ist an einen manuellen Review-Schritt gekoppelt (`status: candidate → accepted/rejected/deprecated` nur per explizitem `/review`-Aufruf), der in der Praxis nicht gepflegt wird. Ergebnis: viel Statistik/Bilddaten raus, kaum brauchbares Lernen zurück.

**These dieses Dokuments:** Scan-Analyse und Live Image Chat sind zwei Frontends für dasselbe Problem — *„gegebene Bild-/Session-Statistik → welcher Parameterwert hat sich bewährt?"* — und sollten sich einen gemeinsamen lokalen Lernkern und einen gemeinsamen, automatisch kuratierten Memory-Store teilen.

---

## 2. Ziel

1. Numerische, hochfrequente Parameterempfehlungen (Scan-Analyse **und** Live-Editor) laufen primär über lokal trainierte, kleine Modelle — kein externer Call, keine Statistik-/Bilddaten-Übertragung, <5 ms Inferenz.
2. Die KI (Cloud-LLM) bleibt im System, aber verschoben: Rationale-Text, Cold-Start (neue Zielobjekt-/Filter-/Op-Kombination ohne Trainingsdaten), Freitext-Interpretation im Live-Chat ("werde die Blautöne los").
3. Memory-Promotion läuft automatisch anhand gemessener Ergebnisqualität statt an ein manuelles Review-Gate.
4. Ein gemeinsames Feature-Schema und ein gemeinsamer Storage-Layer bedienen beide Frontends.

---

## 3. Warum beide Punkte zusammengehören

Beide Pfade haben exakt dieselbe Datenstruktur, nur mit anderem Feature-Raum und anderer Zielmenge:

```
(Feature-Vektor, Parameter-Pfad, gewählter Wert)  →  Ergebnis akzeptiert / korrigiert / verworfen
```

- **Scan-Analyse:** Feature-Vektor = `scan_metrics` (Session-Aggregate über alle Frames), Zielmenge = `config_schema`-Pfade, Ergebnis = wird der Wert im nächsten Run beibehalten/korrigiert (Config-Diff im Folgelauf) und verbessert er die gemessene AQMH-Qualität?
- **Live Editor:** Feature-Vektor = Bildstatistik des *aktuellen* `current_fits`-Zustands (Histogramm, Stern-FWHM/Rundheit, Farbkanal-Balance, Rauschpegel — aus `cv::Mat` günstig berechenbar, ohne Vision-Call), Zielmenge = `pi_image_ops`-Operationstypen, Ergebnis = wird die Operation behalten oder per Undo verworfen/durch eine andere ersetzt?

Beide brauchen: (a) eine automatische Feedback-Quelle statt manueller Kuratierung, (b) ein lokales Modell pro Zielgröße, (c) denselben Memory-Store für "das hat sich bewährt" / "das wurde verworfen". Getrennt umgesetzt entstünde doppelte Infrastruktur (zwei Feature-Pipelines, zwei Trainingsjobs, zwei Storage-Layouts) für dasselbe Muster. Deshalb: ein gemeinsamer Unterbau, zwei dünne Adapter.

---

## 4. Architektur

```
                         ┌───────────────────────────────────────────┐
                         │   PI Local Learning Core (neu, C++)        │
                         │   web_backend_cpp/services/pi/learning/    │
                         │                                             │
                         │   FeatureVector   → { numeric fields,      │
                         │                        kategorial, hash }  │
                         │   ParamModel       → predict(features)     │
                         │                        → {value, conf}     │
                         │   OutcomeRecorder  → append_outcome(...)   │
                         │   ModelRegistry    → load/reload versions  │
                         └───────────┬─────────────────┬───────────────┘
                                     │                  │
                ┌────────────────────┘                  └────────────────────┐
                ▼                                                            ▼
   ┌─────────────────────────────┐                          ┌─────────────────────────────┐
   │ Scan-Analyse Adapter         │                          │ Live-Editor Adapter          │
   │ pi_ai_request_builder.cpp    │                          │ pi_live_image_session.cpp    │
   │                               │                          │                               │
   │ Feature = scan_metrics        │                          │ Feature = current_fits-Stats  │
   │ Ziel = config_schema-Pfade    │                          │ Ziel = pi_image_ops-Typen     │
   │ Fallback = LLM action_plan    │                          │ Fallback = LLM Vision-Chat    │
   └───────────────┬───────────────┘                          └───────────────┬───────────────┘
                   │                                                           │
                   └───────────────────────┬───────────────────────────────────┘
                                            ▼
                         ┌───────────────────────────────────────────┐
                         │   pi_memory_store (erweitert)              │
                         │   status: candidate → accepted/rejected    │
                         │   NEU: auto-promotion via OutcomeRecorder  │
                         └───────────────────────────────────────────┘
```

### 4.1 Feature-Vektor (gemeinsames Schema)

Beide Adapter erzeugen einen flachen, benannten Feature-Vektor (`nlohmann::json`, aber mit fester Feldliste + Versionierung, kein Freitext):

```jsonc
// pi.feature-vector.v1
{
  "schema_version": "pi.feature-vector.v1",
  "domain": "scan" | "live_edit",
  "numeric": {
    "sky_gradient_median": 0.00155,
    "sky_gradient_p90": 0.00173,
    "star_count": 812,
    "fwhm_median": 2.31,
    "roundness_median": 0.91,
    "snr_median": 34.2,
    "noise_sigma": 0.0041,
    "hist_black_clip_frac": 0.0021,
    "hist_white_clip_frac": 0.0004,
    "color_balance_rg": 1.02,
    "color_balance_bg": 0.97,
    "frame_count": 536
    // ... erweiterbar, additiv, nie umbenannt (nur deprecaten)
  },
  "categorical": {
    "target_type": "nebula",
    "bayer_pattern": "GBRG",
    "color_mode": "OSC"
  }
}
```

Für die Scan-Analyse ist das größtenteils **bereits vorhandenes** `scan_metrics`-Material (`aggregate`, `diagnostics`, `registration`) — hier wird nur eine feste, flache Projektion daraus gezogen statt des vollen JSON-Baums, der heute in den Prompt kopiert wird.

Für den Live-Editor ist das **neu**, aber billig: Histogramm, Clip-Fraktionen, Farbkanal-Mittel/-Balance, ggf. grobe Sternstatistik lassen sich aus `LiveImageSession::current_fits` (`cv::Mat`, bereits im Speicher) in wenigen ms in C++ berechnen — ganz ohne den JPEG-Export und ohne Vision-Call, der heute bei *jedem* Chat-Turn passiert.

### 4.2 Parametermodelle

Pro Zielgröße ein kleines, lokales Modell:

- **Diskrete/kategoriale Ziele** (`bge.method`, `normalization.mode`, `chroma_denoise.mode`, welcher `pi_image_ops`-Typ zu einem Freitext-Intent passt) → Gradient-Boosted-Trees-Klassifikator oder, bei sehr wenig Daten, ein einfacher gewichteter Nearest-Neighbor über den Feature-Vektor (funktioniert schon ab ~20 Beispielen, kein Training nötig, nur Distanzsuche).
- **Kontinuierliche Ziele** (`hypermetric_stretch.*`, `chroma_denoise.strength`, `brightness.midtones`, `sharpen.amount`) → kleine Regressions-Heads (Gradient-Boosted-Trees-Regressor oder lineares Modell mit wenigen Features) mit Konfidenzintervall aus der Trainingsdatenstreuung.
- Bewusst **keine** CNN-/Pixel-Modelle (siehe Abschnitt 8, abgegrenzt von der ursprünglichen "GraXpert-Analogie") — der Aufwand für synthetische Trainingsdaten und GPU-Training steht in keinem Verhältnis zum Nutzen, wenn das eigentliche Problem "keine gelernte Zuordnung Statistik→Parameter" ist, nicht "kein Pixel-Modell".

### 4.3 Ablage — lokal, versioniert, performant, portabel im Installationsordner

**Nicht** `~/.local/share/...` (GraXpert-Vorbild war nur der Denkanstoß, nicht die Zielkonvention). Das Projekt hat dafür bereits ein etabliertes, plattformunabhängiges Muster, das sich `pi_models` einfach anschließt statt eine neue Ablage-Philosophie einzuführen:

- `BackendRuntime` legt `runs_dir` standardmäßig als `project_root / "runs"` an (`web_backend_cpp/src/backend_runtime.cpp:250-254`), overridable per `TILE_COMPILE_RUNS_DIR`.
- `pi_storage_dir()` legt den Memory-Store standardmäßig als `runs_dir / ".pi_memory"` an (`web_backend_cpp/src/services/pi/pi_storage_paths.cpp:63-64`), ebenfalls overridable.
- `project_root` selbst ist der Bundle-Payload-Ordner (`${PAYLOAD}` in `packaging/gui3/build_local_{linux,macos}.sh` / `build_local_windows_msys2.cmd`, zur Laufzeit `TILE_COMPILE_PROJECT_ROOT`) — also identisch strukturiert unter Linux, macOS und Windows, weil es kein OS-spezifisches "App-Data"-Verzeichnis ist, sondern Teil des ausgetauschten Installationsordners selbst.

`pi_models` folgt exakt demselben Muster — sichtbar auf oberster Ebene (kein Punkt-Präfix, da Nutzer diese Modelle ggf. zwischen Installationen kopieren/austauschen sollen), Default relativ zu `project_root`, override- und austauschbar per Env-Var:

```cpp
// analog pi_storage_paths.cpp
fs::path default_pi_models_dir(const std::shared_ptr<AppState>& state) {
    return state->runtime.project_root / "pi_models";
}
// override: TILE_COMPILE_PI_MODELS_DIR (gleiches Muster wie TILE_COMPILE_RUNS_DIR)
```

```
<installationsordner>/                 (= ${PAYLOAD}, = TILE_COMPILE_PROJECT_ROOT)
  tile_compile_cpp/
  web_backend_cpp/
  web_frontend_v3/
  agent_service/
  runs/
    .pi_memory/
  pi_models/
    scan/bge.method/v3/model.txt          (LightGBM-Textformat oder eigenes Baum-Format)
    scan/bge.method/v3/metadata.json      (Feature-Liste, Trainingsdatum, n_samples, val_score)
    scan/hypermetric_stretch.black_point/v1/model.txt
    live_edit/op_intent_classifier/v2/model.txt
    live_edit/brightness.midtones/v1/model.txt
    ...
```

- Kein `onnxruntime` nötig für Baum-/Linear-Modelle → header-only Inferenz in C++, keine neue Laufzeit-Dependency, Modelle im KB-Bereich, <1 ms Inferenz.
- `metadata.json` inkl. Hash der Feature-Liste → Runtime erkennt Schema-Drift (z. B. neues Feld) und fällt sauber auf den LLM-Pfad zurück statt mit falschem Feature-Alignment zu inferieren.
- Rollout-Schutz: neue Version wird nur aktiv, wenn ihr Validierungsscore (Holdout aus denselben geloggten Outcomes) die aktuell aktive Version nicht unterschreitet.
- Weil der Ordner Teil des Installationsordners ist (nicht in einem versteckten OS-Nutzerverzeichnis), lässt er sich 1:1 zwischen Rechnern/Betriebssystemen kopieren oder in `packaging/gui3/build_local_*` genau wie `tile_compile_cpp/examples` als vortrainierter Startzustand mit ausgeliefert werden (`cp -a "${PROJECT_ROOT}/pi_models" "${PAYLOAD}/"`).

### 4.4 Outcome-Recorder (das eigentlich neue Stück)

Der fehlende Baustein ist nicht "mehr KI", sondern **ein Logger, der aus bereits vorhandenen Ereignissen automatisch Trainingsbeispiele macht**:

- **Scan-Analyse:** Wenn ein `action_plan`-Vorschlag übernommen wird (Config-Update angewendet) *und* ein Folgelauf mit demselben `run_id`-Stamm eine AQMH-Qualitätsmessung liefert, wird `(feature_vector_at_recommendation_time, path, value, quality_delta)` automatisch als Outcome-Zeile angehängt — kein manuelles Review nötig, das passiert als Nebenprodukt jedes Runs. Datenquelle existiert bereits (`.ai_analyses/*.json`, `global_quality`/`quality_spread` aus AQMH).
- **Live Editor:** Wenn eine Operation im `operation_history` bleibt (nicht per Undo entfernt wird) bis Sessionende/Export, zählt das als positives Outcome; wird sie per Undo entfernt oder durch eine andere Operation am selben Ziel-Parameter ersetzt, als negatives. `LiveImageSession` führt `undo_stack`/`redo_stack`/`operation_history` bereits — der Recorder hängt sich nur als Beobachter an `LiveImageSessionStore::apply_operation` / `undo` / `close` an, ohne die bestehende Undo/Redo-Logik zu verändern.

Damit entsteht der Trainingsdatensatz **beiläufig aus normaler Nutzung**, nicht aus einem Extra-Kurationsschritt.

---

## 5. Erweiterung des Memory-Stores: automatische Promotion statt manuellem Gate

`pi_memory_store` bleibt als Format/API bestehen (`append_candidate`, `list`, `retrieve`, `export_bundle` — Breaking Changes vermeiden), bekommt aber einen zweiten, automatischen Übergangspfad neben dem bestehenden `/review`-Endpunkt:

```cpp
// neu, zusätzlich zu review():
nlohmann::json PiMemoryStore::auto_promote_from_outcome(
    const std::string& memory_id,
    const nlohmann::json& outcome  // { validation_valid, quality_delta, sample_domain }
) const;
```

Regel: ein `candidate`-Memory wird automatisch `accepted`, sobald für seinen `context_signature` **N ≥ 3** unabhängige Outcomes mit `quality_delta > 0` vorliegen (Schwelle konfigurierbar); automatisch `rejected` bei überwiegend negativen Outcomes. Der bestehende manuelle `/review`-Endpunkt bleibt als Override für Edge-Cases und Audit — aber **nicht mehr Voraussetzung** für Sichtbarkeit im Retrieval. Das behebt direkt das benannte Problem: "die derzeit gespeicherten Memories sind vom User kaum zu verwalten."

`retrieve()`/`retrieve_negative()` bekommen zusätzlich zum bestehenden `context_match_score` (Feld-Overlap) einen Score-Term aus der Feature-Vektor-Distanz (Kosinus-/euklidische Distanz auf `numeric`, exakter Match auf `categorical`) — das ist kein Ersatz, sondern eine dritte Signalquelle neben Pfad-Match und Kontext-Overlap, weil reine Keyword-Overlap-Scores bei numerischen Statistiken (z. B. "ähnliches `sky_gradient_median`") strukturell blind sind.

Live-Editor-Operationen werden als eigener `type` (`"live_edit_operation"`) im selben Store abgelegt, mit demselben Outcome-Automatik-Pfad — dadurch teilen sich beide Frontends Speicherort, Export/Import (`export_bundle`), Dedupe (`dedupe()`) und die Review-UI, statt eine zweite Persistenzschicht zu bauen.

---

## 6. Rolle der Cloud-KI danach

Nicht eliminiert, aber verschoben:

| Aufgabe | Vorher | Nachher |
|---|---|---|
| Zahlenempfehlung bei bekanntem Muster (genug Trainingsdaten) | LLM zero-shot pro Aufruf | Lokales Modell, <5 ms, kein externer Call |
| Rationale/Erklärtext zur Empfehlung | LLM generiert Wert *und* Text | LLM bekommt vom Modell bereits Wert+Konfidenz, generiert nur noch den erklärenden Text (kleinerer Prompt, kein Rate-Risiko am Zahlenwert) |
| Cold-Start (neues Ziel/Filter/Op ohne Trainingsdaten) | LLM zero-shot | LLM zero-shot **bleibt** — aber Ergebnis wird automatisch als neuer Memory-Kandidat + Outcome-Tracking erfasst, sodass der nächste ähnliche Fall lokal bedient werden kann |
| Freitext-Intent im Live-Chat ("werd die Blautöne los") | Vision-LLM pro Turn, inkl. Bildübertragung | LLM übersetzt Intent → Op-Typ (Klassifikation, kein Bild nötig sobald genug Trainingsdaten); Parameterwerte kommen vom lokalen Regressionsmodell auf Basis der lokal berechneten Bildstatistik. Bild/Vision-Call bleibt Fallback für neuartige, nicht katalogisierte Anweisungen. |

Damit sinkt sowohl der externe Datenabfluss (Statistiken, teils Bilddaten) als auch die Latenz/Kosten pro Interaktion für den Großteil der Fälle — ohne dem System die Fähigkeit zu nehmen, auf wirklich neue Situationen zu reagieren.

---

## 7. Implementierungsreihenfolge (inkrementell, nicht-brechend)

1. **Outcome-Recorder für Scan-Analyse** (kleinster Schritt, Daten existieren bereits in `.ai_analyses/*.json` + AQMH-Scores): loggt `(feature_vector, path, value, quality_delta)` still mit, kein Verhaltensänderung.
2. **Auto-Promotion in `pi_memory_store`** auf Basis der Recorder-Daten (löst das "kaum zu verwalten"-Problem zuerst, schon vor jedem ML-Training nützlich, weil bereits vorhandene `outcome.validation_valid`-Signale endlich genutzt werden).
3. **Erstes lokales Modell** für 2–3 hochfrequente, klar messbare Scan-Pfade (`bge.method`, `normalization.mode`) als Proof of Concept — Fallback auf LLM bleibt aktiv, Modell läuft zunächst nur "shadow" (Vorhersage wird geloggt, aber nicht ausgeliefert) bis genug Validierungsdaten für einen Vergleich vorliegen.
4. **Live-Editor-Bildstatistik + Outcome-Recorder** (Beobachter an `LiveImageSessionStore`, keine Änderung an Undo/Redo).
5. **Live-Editor-Op-Intent-Klassifikator + Parameter-Regressoren**, zunächst ebenfalls shadow-mode.
6. **Gemeinsames Retraining-Skript** (offline, außerhalb C++, periodisch oder nach N neuen Outcomes), Modell-Registry mit Versionierung/Rollback.
7. Erst nach belastbarem Shadow-Vergleich: lokale Modelle als **primäre** Quelle scharf schalten, LLM auf Rationale/Cold-Start reduzieren (Abschnitt 6).

Jeder Schritt ist einzeln sinnvoll und für sich schon ausrollbar; nichts davon setzt voraus, dass ein späterer Schritt schon existiert.

---

## 8. Explizit nicht Teil dieses Vorschlags

- Kein CNN-/Pixel-Ebenen-Modell (à la GraXpert AI-BGE) — das war der ursprüngliche Gedankenanstoß, aber das eigentliche Problem hier ist "keine gelernte Statistik→Parameter-Zuordnung", nicht "kein neuronales Bildmodell". Bleibt als möglicher Folgeschritt, falls Classic-BGE/AutoBGE nachweislich an Grenzen stoßen — unabhängig von diesem Dokument.
- Kein Ersatz des `pi_context_protocol_compression_plan_de.md`-Ansatzes (kein Byte-Kompressions-Handshake mit dem Modell) — orthogonal, dieses Dokument reduziert eher die Menge dessen, was überhaupt in den Prompt muss.
- Kein Live-Learning/Online-Training im laufenden Betrieb — Training bleibt ein expliziter, versionierter Offline-Schritt mit Validierungs-Gate vor Rollout.

---

## 9. Offene Fragen

1. Schwellenwert `N` für Auto-Promotion (Vorschlag: 3, konfigurierbar über `pi.memory.auto_promotion.min_outcomes`) — abhängig davon, wie schnell in der Praxis genug Runs mit vergleichbarem `context_signature` anfallen.
2. Trainingsframework für die Offline-Modelle (LightGBM vs. eigenes minimalistisches Baum-Format ohne Python-Runtime-Abhängigkeit zur Laufzeit) — Kriterium ist ausschließlich die C++-Inferenzseite, das Trainings-Tooling darf Python sein.
3. Wo genau die Bildstatistik-Berechnung im Live-Editor-Pfad ansetzt (`apply_operation`-Hook vs. eigener periodischer Sampler) — Performance-Grenze: darf die interaktive Latenz der +/- Buttons nicht spürbar erhöhen.
