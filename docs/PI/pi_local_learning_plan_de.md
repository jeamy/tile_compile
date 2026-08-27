# PI Local Learning — Design-Dokument

> **Status:** Schritt 1 (1a/1b/1c) implementiert und nach Review korrigiert (2026-08-27) — kompiliert, End-to-End-Join in einem echten Lauf noch unverifiziert, siehe Abschnitt 0.1. Schritt 2 (Auto-Promotion) folgt.
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

## 0. Vor Implementationsbeginn zu verifizieren (blockierend)

Erneute Analyse (2026-08-27) hat einen unverifizierten Kernannahme-Fehler gefunden, auf dem Schritt 1 der Implementierungsreihenfolge (Abschnitt 7) direkt aufbaut, plus zwei Record-Schema-Entscheidungen, die *vor* dem ersten Logging-Zeilen-Schreiben feststehen müssen, weil ein Log-Format im Nachhinein zu ändern die bis dahin gesammelten Daten entwertet.

**0.1 Fehlt die Verknüpfung Analyse → angewandte Config → Run-Qualität?**

Wichtig: diese Prüfung ist eine reine **Code-/Architekturfrage für zukünftige Runs**, keine Sichtung der vorhandenen `.ai_analyses/*.json`-Dateien. Die liegen dort nur als alte Testläufe aus der Entwicklung dieses Vorschlags, sind für das neue System nicht relevant und können bedenkenlos gelöscht werden (Abschnitt 0.1a) — sie enthalten weder eine definierte Record-Struktur (Abschnitt 0.2) noch eine verknüpfte Ergebnis-Qualität und wären als Trainingsgrundlage ungeeignet, selbst wenn sie blieben.

**Verifiziert (2026-08-27), konkreter Code-Befund statt offener Frage:**

Die Kette zerfällt in zwei Hälften mit sehr unterschiedlichem Zustand:

**a) `action_plan` → übernommene Aktionen: existiert bereits, ist aber toter Code.**
`web_backend_cpp/src/routes/ai_routes.cpp:2229-2244` (Route für `apply`) baut bei `body["learn"] == true` bereits genau die in Abschnitt 0.2 geforderte Struktur: `build_apply_candidate_memory(analysis_id, analysis_data, applied, ...)` legt einen `pi_memory_store`-Kandidaten mit `analysis_id`, `config_updates: applied` (die **vollständige** Menge der angewendeten `path→value`-Paare in einem Record, nicht pro Pfad — deckt sich bereits mit 0.2) und `context_signature` (≈ Feature-Vektor) an. Das Problem: `learn: true` wird **an keiner Stelle im Frontend gesetzt** (`grep -rn "learn.*true" web_frontend_v3/js` liefert keinen Treffer) — der Pfad ist vorhanden, aber inaktiv. Zusätzlich: `state->revision_store.add(saved_path, yaml_text, "pi_scan_ai")` (Zeile ~2216) ruft `ConfigRevisionStore::add()` **ohne** das optionale `run_id`-Argument auf — die entstehende `ConfigRevision` weiß also nicht, in welchem Run sie später verwendet wurde, selbst wenn `learn` aktiv wäre.

**b) `ConfigRevision`/Run → `global_quality`: existiert nicht.**
`ConfigRevision` (`config_revisions.hpp:14-21`) hat kein Feld, das auf eine Analyse verweist — nur `run_id` (optional, hier ungesetzt) und `source` (String-Tag wie `"pi_scan_ai"`). Schwerer wiegt: `tile_compile_cpp/apps/runner_phase_aqmh_global_quality.cpp` berechnet `AqmhGlobalQualityResult` (Gewichte pro Frame), persistiert davon aber nur ein Event-Log-Fragment (`{"weights": count, "masked_invalid_frames": count}` via `emitter.phase_end`) — **kein** `global_quality`- oder `quality_spread`-Skalar wird geschrieben. Bestätigt durch `grep -rn "global_quality|quality_spread" web_backend_cpp/`: **kein Treffer** — der Backend-Layer liest diesen Wert an keiner Stelle. Der zweite Teil der Kette existiert schlicht nicht, weder im Runner-Output noch im Backend.

**Konsequenz für Abschnitt 7:** Schritt 1 war **nicht** "Logger anschließen". Er zerfiel in drei tatsächliche Arbeitspakete — inzwischen alle drei implementiert:

1. **1a (`learn`-Pfad + Provenienz), implementiert:** Checkbox "Lernkandidat speichern" (`ai-empfehlung.js`) ist jetzt Default-an statt Default-aus (die Verdrahtung selbst war entgegen der ursprünglichen Vermutung schon vorhanden — reine Korrektur einer zu ungenauen Analyse). `ai_routes.cpp` berechnet beim Apply zusätzlich `config_sha256` (Hash der angewendeten YAML) und legt ihn zusammen mit `revision_id` im Memory-Kandidaten ab (`build_apply_candidate_memory`). `runs_routes.cpp` schreibt bei **jedem** Run-Start (Einzel- und Queue-Pfad) `runs/<run_id>/artifacts/pi_run_provenance.json` mit demselben `config_sha256` — das durable, neustartsichere Bindeglied, das nicht auf die ohnehin flüchtige In-Memory-`ConfigRevisionStore`-Id-Vergabe angewiesen ist.
2. **1b (Run-Qualität), implementiert mit korrigierter Metrik-Definition:** `global_quality`/`quality_spread` existierten entgegen der ursprünglichen Doku-Annahme **nicht** als Ausgabewerte — `aqmh.global_quality.*` ist nur der Config-Block der Gewichtungsformel. Nutzer-Entscheidung: Run-Qualität = Mittelwert der gültigen AQMH-Frame-Gewichte (`AqmhGlobalQualityResult.weights`, gefiltert über `input_invalid`). `runner_phase_local_metrics.cpp` schreibt jetzt `runs/<run_id>/artifacts/pi_run_quality.json` (`schema_version: pi.run-quality.v1`, `mean_weight`, `valid_frame_fraction`, `frames_total`, `frames_valid`).
3. **1c (Join + Attach), implementiert, nach Review korrigiert:** neues Modul `services/pi/pi_outcome_recorder.{hpp,cpp}` liest beide Artefakte und hängt das Ergebnis über eine neue `PiMemoryStore::attach_outcome()`-Methode an — ohne Statusänderung.

**Review-Korrekturen (drei Blocker, vor jeder Datenerfassung behoben, nicht nachträglich):**
- **Join-Key:** ursprünglich `config_sha256` als *einziger* Match-Key geplant — empirisch praktisch nie treffend, weil Apply-Zeit-YAML (`yaml_dump(patched)`) und Run-Start-YAML (`effective_config_yaml()`, injiziert `color_mode`/Astrometrie-Pfade, anderer Serialisierer im Frontend-Zwischenschritt) nicht byte-identisch sind. Primärer Join-Key ist jetzt die Revisions-Abstammung: `runs_routes.cpp` liest `state->active_config_revision_id` beim Run-Start, *bevor* es überschrieben wird, und schreibt es als `prior_active_config_revision_id` in die Provenienz — abgeglichen gegen `revision_id` im Memory-Kandidaten. `config_sha256` bleibt nur sekundäre Bestätigung, wenn kein Revisions-Match existiert. `memory_dedupe_signature` bezieht jetzt `config_sha256` mit ein, damit ein erneuter Apply nach geänderter Basis-Config nicht den alten Kandidaten mit veraltetem Hash zurückliefert.
- **Latest-wins-Overlay reicht nicht für Schritt 2:** `attach_outcome()` schrieb nur in den bestehenden `reviews_path()`-Overlay, den `list()` als "letzter gewinnt" pro `memory_id` zusammenführt — unzählbar gegen die geplante "N ≥ 3 unabhängige Outcomes"-Regel aus Abschnitt 5, und mehrere Queue-Runs auf demselben Kandidaten hätten sich gegenseitig überschrieben. Neuer, zusätzlicher Append-only-Log `outcomes_path()`, den `list()` vollständig (nicht latest-wins) zu `item["outcomes"]` zusammenführt — akkumuliert, statt zu überschreiben. Der bestehende `reviews_path()`-Pfad bleibt parallel bestehen, damit vorhandener Code, der `item["outcome"].validation_valid` liest, unverändert funktioniert.
- **Marker cachte Fehlschläge dauerhaft:** die ursprüngliche Idempotenz-Prüfung (`fs::exists(marker_path)`) hätte auch `no_memory_candidate`/`error` als endgültig behandelt — der erste Poll nach Run-Ende hätte "keine Daten" für immer festgeschrieben, auch nachdem ein Matching-Bug behoben wird. Jetzt nur `matched: true` und `no_provenance` (Provenienz-Datei entsteht nur einmal beim Run-Start, taucht nie nachträglich auf) als endgültig; alles andere bleibt wiederholbar.

Damit ist bestätigt, dass die im Dokument angenommene Grundstruktur (ein Record pro Diff mit vollem Kontext) bereits im bestehenden `build_apply_candidate_memory`-Code vorgezeichnet war und wiederverwendet werden konnte statt neu entworfen werden zu müssen — ebenso das Outcome-Feld selbst und der Review-Log-Overlay-Mechanismus.

**Verifikationsstand:** Kompiliert, bestehende `pi_memory_store`-Tests grün, Logik zweimal per Code-Trace nachvollzogen (initial + nach Review). **Der End-to-End-Join wurde noch nie live ausgeführt** (kein realer Scan/Run in dieser Session) — der erste tatsächliche `{"matched": true}`-Marker aus einem echten Lauf steht noch aus und ist der eigentliche Beweis, dass 1a–1c zusammen funktionieren, nicht nur einzeln kompilieren.

**Was passiert mit den gelernten Daten, wenn ein Run gelöscht wird?** Der angehängte Outcome lebt im `pi_memory_store` (`runs/.pi_memory/`, außerhalb des Run-Verzeichnisses) und übersteht das Löschen eines Runs, **sobald er einmal geschrieben wurde** — die Herkunfts-Artefakte `pi_run_provenance.json`/`pi_run_quality.json` selbst liegen dagegen in `runs/<run_id>/artifacts/` und verschwinden mit dem Run.

Erste Umsetzung erzwang deshalb den Join direkt vor dem Löschen in der App-Delete-Route — das war falsch und wurde zurückgenommen: Löschen ist eine legitime Nutzerentscheidung ("diesen Run will ich nicht", z. B. ein Fehlversuch), und ein erzwungenes Nacherfassen des Outcomes würde diese Entscheidung stillschweigend übergehen. Stattdessen wird der Outcome jetzt **direkt beim Abschluss des Runs** erfasst, nicht erst beim nächsten Status-Poll: `SubprocessManager::launch()` bekommt einen optionalen `on_complete`-Callback (generischer Mechanismus, kein PI-spezifisches Wissen in `subprocess_manager.cpp` selbst — bewusst dort nicht verankert, um diese breit genutzte, generische Komponente nicht mit Business-Logik zu koppeln), den `runs_routes.cpp` für Run-Jobs auf `record_run_outcome_if_needed()` setzt. Das schließt das Zeitfenster zwischen "Run fertig" und "Nutzer löscht ihn" fast vollständig, ohne Lösch-Semantik zu verändern. Die Status-Poll-Route ruft die Funktion weiterhin zusätzlich auf — dank Marker-Datei ein günstiges, redundantes Sicherheitsnetz, kein Doppelaufwand.

**Zwei Fälle bleiben bewusst ungelöst, nicht verdrängt:**
- Ein Run wird gelöscht, *bevor* der `on_complete`-Callback lief (z. B. Backend-Absturz exakt in diesem Fenster) — sehr kleines, aber reales Restrisiko, kein erzwungener Nacherfassungs-Pfad mehr vorhanden.
- Ein Run wird **auf Dateisystemebene außerhalb der App** gelöscht (`rm -rf`, Dateimanager, o. Ä.). Das ist grundsätzlich nicht hookbar — keine App-Route wird dabei aufgerufen, es gibt keinen Code-Pfad, der das abfangen könnte. Die Konsequenz ist Datenverlust für diesen Run, aber kein stiller Fehlerzustand: `pi_outcome_recorder` schreibt nichts falsch, es wird einfach nichts geschrieben.

**Warum die Lerndaten nicht zusätzlich woanders dupliziert werden:** Der *abgeleitete* Outcome liegt schon bewusst getrennt vom Run-Verzeichnis (im `pi_memory_store`) — das war von Anfang an so entworfen, genau um Run-Löschungen zu überstehen. Was **nicht** verlagert wurde, sind die rohen Quelldateien `pi_run_provenance.json`/`pi_run_quality.json` selbst — die bleiben in `runs/<run_id>/artifacts/`, konsistent mit jedem anderen Run-Diagnostik-Artefakt (z. B. `aqmh_metrics.json`), das dort ebenfalls verschwindet, wenn der Run gelöscht wird. Sie zusätzlich an einem zweiten Ort zu duplizieren, nur für den Fall einer Löschung außerhalb der App, würde erstens das Speicher-Design aus Abschnitt 4.3 (ein Ort pro Artefakttyp, keine Schatten-Kopien) durchbrechen, und zweitens das eigentliche Problem nicht lösen — bei direktem Dateisystem-Löschen außerhalb der App ließe sich so oder so nichts mehr retten, sobald die Löschung tatsächlich stattgefunden hat. Der richtige Hebel ist deshalb "so früh wie möglich zuverlässig erfassen" (die `on_complete`-Lösung), nicht "überall vorsichtshalber kopieren".

**0.1a Altdaten:** `web_backend_cpp/runtime/.ai_analyses/*.json` sind Testartefakte aus der Zeit vor diesem Vorschlag, ohne die in 0.2 festgelegte Record-Struktur und ohne verknüpfte Ergebnis-Qualität. Sie fließen an keiner Stelle dieses Plans als Trainingsdaten ein und sollten gelöscht statt migriert werden, damit niemand sie später versehentlich als Datengrundlage missversteht. Löschen ist unabhängig vom Rest dieses Dokuments und blockiert nichts.

**0.2 Record-Schema für den Outcome-Recorder (§4.4) — Attribution**

Das im Dokument gezeigte Beispiel (`M16_20260809_154801.json`) ändert drei Pfade gleichzeitig (`normalization.mode`, `data.bayer_pattern`, `bge.method`). Ein Log-Format `(feature_vector, path, value, quality_delta)` **pro Pfad** unterstellt eine Kausalzuordnung, die bei gemeinsam angewendeten Änderungen nicht herstellbar ist. Festlegung: es wird **ein Record pro angewendetem Config-Diff** geloggt — die *gesamte* Menge der geänderten `path→value`-Paare plus Delta plus Anzahl geänderter Pfade. Training kann später auf Single-Change-Runs filtern oder die Pfadmenge gemeinsam modellieren; das Log selbst ist die Trainingsmenge und lässt sich nicht rückwirkend korrigieren.

**0.3 Cross-Run-Deltas sind durch unterschiedliche Frames konfundiert, nicht nur durch Mehrfach-Änderungen**

Lauf N und Lauf N+1 haben in der Regel unterschiedliche Eingabe-Frames — ihr Qualitätsunterschied ist kein kontrollierter Vergleich. Bevorzugte Label-Quelle sind daher **Vergleiche auf denselben Frames bei unterschiedlicher Config**: der bestehende `bge.autotune`-CV-Holdout (`configuration.hpp:473`, `max_evals`/`holdout_fraction`) liefert das bereits (Features → beste Config + Score auf fixen Daten), ebenso Resume-/Re-Läufe auf unverändertem Frame-Satz. Cross-Run-Deltas über unterschiedliche Sessions bleiben ein schwächeres Sekundärsignal, nicht die primäre Quelle.

**0.4 Live-Editor-Label — Terminalzustand statt jeder Zwischenstufe**

Die in §4.4 beschriebene Regel ("bleibt im `operation_history` = positiv") kollidiert mit dem `adjust_step`-Mechanismus: fünf +/- Klicks erzeugen fünf Zustände, von denen nur der letzte am Sessionende zählt. Festlegung: Label ist der **Terminalwert pro Op-Typ beim Sessionende/Export**; die Adjust-Trajektorie (Zwischenwerte) wird als Metadatum mitgeloggt, aber nicht als eigene positive Beispiele gezählt.

Diese vier Punkte sind vor dem ersten produktiven Logging-Schritt zu klären; alle übrigen Punkte in Abschnitt 9 sind Verfeinerungen, die die Implementierung nicht blockieren.

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

Zusätzlich gehört bei `domain: "live_edit"` die **zuletzt angewendete Op-Sequenz** (letzte 3–5 Einträge aus `operation_history`, als kategoriale Liste von Op-Typen) in den Feature-Vektor: dieselbe Bildstatistik bedeutet nach `denoise → sharpen` etwas anderes als nach `sharpen → denoise`, weil Bildoperationen nicht kommutativ sind. Ohne diesen Kontext generalisiert ein rein pixelstatistik-basiertes Modell falsch auf Sequenzen, die im Training so nicht vorkamen.

### 4.2 Parametermodelle

Pro Zielgröße ein kleines, lokales Modell:

- **Diskrete/kategoriale Ziele** (`bge.method`, `normalization.mode`, `chroma_denoise.mode`, welcher `pi_image_ops`-Typ zu einem Freitext-Intent passt) → Gradient-Boosted-Trees-Klassifikator oder, bei sehr wenig Daten, ein einfacher gewichteter Nearest-Neighbor über den Feature-Vektor (funktioniert schon ab ~20 Beispielen, kein Training nötig, nur Distanzsuche).
- **Kontinuierliche Ziele** (`hypermetric_stretch.*`, `chroma_denoise.strength`, `brightness.midtones`, `sharpen.amount`) → kleine Regressions-Heads (Gradient-Boosted-Trees-Regressor oder lineares Modell mit wenigen Features) mit Konfidenzintervall aus der Trainingsdatenstreuung.
- Bewusst **keine** CNN-/Pixel-Modelle (siehe Abschnitt 8, abgegrenzt von der ursprünglichen "GraXpert-Analogie") — der Aufwand für synthetische Trainingsdaten und GPU-Training steht in keinem Verhältnis zum Nutzen, wenn das eigentliche Problem "keine gelernte Zuordnung Statistik→Parameter" ist, nicht "kein Pixel-Modell".
- **Konsistenz zwischen Pfaden:** jedes Modell sagt seinen Zielpfad unabhängig vorher (z. B. `bge.method` und `normalization.mode` getrennt) — das im Dokument gezeigte reale Beispiel zeigt aber, dass diese Pfade sich gegenseitig bedingen (BGE auf Nebel-Zielen hängt von `normalization.mode` ab). Bevor eine Menge unabhängiger Modell-Vorhersagen angewendet wird, läuft sie durch dieselbe Schema-Validierung wie ein LLM-`action_plan` (`action_plan_validation`) plus einen Konsistenz-Check auf bekannten Regelpaaren; bei Konflikt gewinnt die konservativere Kombination oder es wird auf den LLM-Pfad zurückgefallen. Das ist Teil von Implementierungsschritt 7 (Scharfschaltung), nicht der ersten Schritte.

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
- **Portabilität verschärft die Kompatibilitätsprüfung:** weil `pi_models/` gezielt zwischen Installationen kopierbar sein soll, reicht der Feature-Hash allein nicht — ein mitkopiertes Modell für `bge.method` kann einen Enum-Wert vorhersagen, den ein neueres `config_schema` nicht mehr kennt. `metadata.json` pinnt deshalb zusätzlich den `config_schema_sha256` (existiert bereits als Feld in `.ai_analyses/*.json` → `_meta.config_schema_sha256`, dieselbe Quelle wiederverwenden). Bei Mismatch: Modell wird ignoriert, Fallback auf LLM-Pfad, kein Silent-Apply eines möglicherweise ungültigen Werts.
- Rollout-Schutz: neue Version wird nur aktiv, wenn ihr Validierungsscore (Holdout aus denselben geloggten Outcomes) die aktuell aktive Version nicht unterschreitet.
- **Kill-Switch:** ein Konfigurationsschalter (`pi.local_models.enabled: false` oder Env-Var) erzwingt pro Installation den reinen LLM-Pfad, unabhängig vom Inhalt von `pi_models/` — Rückfallebene, falls ein mitgeliefertes oder trainiertes Modell in der Praxis schlechter empfiehlt als der bisherige Weg.
- Weil der Ordner Teil des Installationsordners ist (nicht in einem versteckten OS-Nutzerverzeichnis), lässt er sich 1:1 zwischen Rechnern/Betriebssystemen kopieren oder in `packaging/gui3/build_local_*` genau wie `tile_compile_cpp/examples` als vortrainierter Startzustand mit ausgeliefert werden (`cp -a "${PROJECT_ROOT}/pi_models" "${PAYLOAD}/"`).
- **Bootstrap-Realismus:** ein vortrainierter Startzustand setzt voraus, dass irgendwo genug Outcome-Daten über viele Installationen/Sessions hinweg zusammenkommen, um ihn zu erzeugen — für eine einzelne lokale Installation kann das Modell-Layer (Abschnitt 4.2) bei geringer Nutzung dauerhaft unter der für sinnvolles Training nötigen Datenmenge bleiben. Das ist kein Grund, den Ansatz zu verwerfen: Der **Memory-Store mit automatischer Promotion (Abschnitt 5) ist die tragende Schicht** und liefert bereits ab wenigen Dutzend Outcomes Nutzen, unabhängig davon, ob je genug Daten für ein trainiertes Modell zusammenkommen. Das Modell-Layer ist ein optionaler Beschleuniger obendrauf, kein Single Point of Failure des Vorschlags.

### 4.4 Outcome-Recorder (das eigentlich neue Stück)

Der fehlende Baustein ist nicht "mehr KI", sondern **ein Logger, der aus bereits vorhandenen Ereignissen automatisch Trainingsbeispiele macht**. Record-Schema und Label-Definition sind durch Abschnitt 0.2–0.4 festgelegt, hier nur noch die Anbindung:

- **Scan-Analyse:** Wenn ein `action_plan`-Vorschlag übernommen wird (Config-Update angewendet), wird **ein Record pro angewendetem Diff** geloggt: `{feature_vector_at_recommendation_time, applied_paths: [{path, value}, ...], changed_path_count}`. Das Qualitäts-Delta wird nachträglich angehängt, sobald ein Ergebnis auf denselben Frames vorliegt — primär aus `bge.autotune`-CV-Holdout-Vergleichen bzw. Resume-Läufen auf unverändertem Frame-Satz (kontrolliert), Cross-Run-Vergleiche über unterschiedliche Sessions nur als schwächeres Sekundärsignal, klar als solches markiert (`comparison_kind: same_frames | cross_run`). Voraussetzung ist die in Abschnitt 0.1 zu verifizierende Verknüpfung `action_plan → ConfigRevision → Run → global_quality`.
- **Live Editor:** Label ist der **Terminalwert pro Op-Typ beim Sessionende/Export** (Abschnitt 0.4) — nicht jede Zwischenstufe der `adjust_step`-Trajektorie. Eine Operation, deren Terminalwert im `operation_history` verbleibt, zählt als positives Outcome; eine per Undo entfernte oder durch eine andere Operation am selben Ziel-Parameter ersetzte als negatives. Die Adjust-Trajektorie wird als Metadatum am Record mitgeführt (nachvollziehbar, aber nicht als eigene Trainingsbeispiele gezählt). `LiveImageSession` führt `undo_stack`/`redo_stack`/`operation_history` bereits — der Recorder hängt sich nur als Beobachter an `LiveImageSessionStore::apply_operation` / `undo` / `close` an, ohne die bestehende Undo/Redo-Logik zu verändern.

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

**Sicherheitsnetz für die Einführung:** Auto-Promotion wirkt direkt auf `positive_memories_from_session_context` (`pi_ai_request_builder.hpp:10`) und damit unmittelbar auf jeden nächsten LLM-Prompt — anders als die Parametermodelle (Abschnitt 4.2), die vor Scharfschaltung einen Shadow-Mode durchlaufen. Eine falsch kalibrierte Promotion-Regel würde also sofort und unbemerkt die Empfehlungsqualität verschlechtern, die das gesamte Vorhaben verbessern soll. Deshalb bekommt Schritt 2 in Abschnitt 7 denselben Shadow-Mode: Promotion-Entscheidungen werden zunächst nur berechnet und geloggt (`would_promote`/`would_reject` neben dem bestehenden `status`), nicht angewendet, bis die Entscheidungen stichprobenartig geprüft sind.

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

0. ~~Verifikation der Verknüpfungskette~~ — erledigt (Abschnitt 0.1).
1. ~~Outcome-Recorder für Scan-Analyse~~ — **implementiert** (2026-08-27), Details in Abschnitt 0.1:
   - ~~1a.~~ `learn`-Default umgestellt, `config_sha256`/`revision_id` im Memory-Kandidaten, `pi_run_provenance.json` bei jedem Run-Start.
   - ~~1b.~~ `pi_run_quality.json` (Mittelwert der gültigen AQMH-Frame-Gewichte) pro Run persistiert.
   - ~~1c.~~ `pi_outcome_recorder` verknüpft beide über `config_sha256` und hängt das Ergebnis per `PiMemoryStore::attach_outcome()` an — ohne Statusänderung, ohne fabriziertes `quality_delta` (`comparison_kind: "unpaired"`, Delta-Berechnung bleibt dem Offline-Training vorbehalten, Abschnitt 0.3/9). Aufgerufen aus der Status-Poll-Route und — als letzte Gelegenheit vor Datenverlust — aus der Run-Delete-Route.
2. **Auto-Promotion in `pi_memory_store`**, zunächst im Shadow-Mode (Abschnitt 5: `would_promote` geloggt, nicht angewendet) auf Basis der Recorder-Daten — erst nach Prüfung der Shadow-Entscheidungen scharfgeschaltet. Löst das "kaum zu verwalten"-Problem, sobald aktiv.
3. **Erstes lokales Modell** für 2–3 hochfrequente, klar messbare Scan-Pfade (`bge.method`, `normalization.mode`) als Proof of Concept — Fallback auf LLM bleibt aktiv, Modell läuft zunächst nur "shadow" (Vorhersage wird geloggt, aber nicht ausgeliefert) bis genug Validierungsdaten für einen Vergleich vorliegen.
4. **Live-Editor-Bildstatistik + Outcome-Recorder** (Beobachter an `LiveImageSessionStore`, keine Änderung an Undo/Redo; Label = Terminalwert pro Op-Typ bei Sessionende, Abschnitt 0.4).
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

Nicht blockierend — Verfeinerungen, die während oder nach den Implementierungsschritten in Abschnitt 7 entschieden werden können:

1. Schwellenwert `N` für Auto-Promotion (Vorschlag: 3, konfigurierbar über `pi.memory.auto_promotion.min_outcomes`) — abhängig davon, wie schnell in der Praxis genug Runs mit vergleichbarem `context_signature` anfallen.
2. Trainingsframework für die Offline-Modelle (LightGBM vs. eigenes minimalistisches Baum-Format ohne Python-Runtime-Abhängigkeit zur Laufzeit) — Kriterium ist ausschließlich die C++-Inferenzseite, das Trainings-Tooling darf Python sein.
3. Wo genau die Bildstatistik-Berechnung im Live-Editor-Pfad ansetzt (`apply_operation`-Hook vs. eigener periodischer Sampler) — Performance-Grenze: darf die interaktive Latenz der +/- Buttons nicht spürbar erhöhen.
4. Nebenläufigkeit: `ModelRegistry`-Reload nach neuem Training darf laufende Live-Editor-Inferenzen (jeder Op-Call) nicht blockieren oder mit einem halb geladenen Modellstand kollidieren — Locking-Strategie beim Hot-Swap einer neuen Modellversion.
5. Sollen `pi_models/`-Inhalte Teil von `export_bundle` (Abschnitt 5) werden, oder bleiben Modelle explizit von Export/Import der Memories getrennt, weil sie ableitbare Binärartefakte sind statt kuratierten Wissens?
