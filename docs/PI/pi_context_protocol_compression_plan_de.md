# PI Context Protocol und Kompressionsplan

## Ausgangsfrage

PI soll fuer Scan-Empfehlungen, Run-Control-Chat und Korrekturvorschlaege genug Kontext bekommen, um belastbare Aussagen zu machen. Die naheliegende Idee ist ein Protokoll mit der KI auszuhandeln: fragen, welche Kompressionsalgorithmen sie kennt, Daten entsprechend verpacken und die KI entpacken lassen.

Das sollte nicht als primaerer Weg umgesetzt werden.

Ein Sprachmodell kann zwar ueber Formate wie JSON, CSV, Base64, gzip, zstd oder Delta-Encoding reden, aber es ist kein verlaesslicher Decoder fuer komprimierte Bytefolgen. Wenn PI eine gzip- oder zstd-komprimierte Base64-Payload direkt in den Prompt bekommt, kann das Modell sie nicht deterministisch und vollstaendig entpacken. Das fuehrt genau zu der Fehlerklasse, die vermieden werden soll: plausibel klingende Rekonstruktion aus unvollstaendig verstandenem Kontext.

Sinnvoll ist dagegen:

1. Byte-Kompression nur zwischen Backend und Sidecar, mit deterministischer Dekompression vor dem Modell.
2. Semantische Kompression fuer den Prompt: strukturierte Fakten, Aggregate, IDs, Einheiten, Quellen und Regeln.
3. Chunking und Nachlade-Protokolle fuer Details, die nur bei Bedarf in den Modellkontext kommen.

## Ziel

Ein einheitliches Context-Protokoll `pi.context.v2`, das fuer folgende Faelle nutzbar ist:

- Scan-Analyse vor dem Run
- Live-Run-Chat in Run Control
- Analyse eines fertigen Runs
- Korrektur- und Parameterempfehlungen nach fertigem Bild
- Debugging von Fehlern oder schlechten Ergebnissen

PI soll keine Defaults, Schema-Grenzen, Empfehlungen oder Messwerte erfinden muessen. Jede Aussage muss auf explizit uebertragenen Fakten, kuratierter Parametermetadaten oder klar markierten Annahmen basieren.

## Grundsatzentscheidung

### Nicht machen

- Keine gzip/base64- oder zstd/base64-Daten direkt an das Modell schicken.
- Keine Custom-Kompression, die das Modell per Prompt "entpacken" soll.
- Keine Aufforderung wie "Wenn du dieses Format kennst, dekodiere es selbst".
- Keine riesigen Rohartefakte als unstrukturierter JSON-/HTML-Dump.

### Machen

- Backend und Sidecar duerfen Payloads technisch komprimieren, wenn sie vor dem Prompt wieder dekomprimiert werden.
- Der Prompt bekommt nur modell-lesbare, semantisch komprimierte Informationen.
- Alle Detaildaten sind ueber stabile `fact_id`, `artifact_id` und `chunk_id` referenzierbar.
- Empfehlungen werden nach der Modellantwort deterministisch gegen Fakten und Parametermetadaten validiert.

## pi.context.v2

Das Context-Paket soll immer dieselbe Top-Level-Struktur haben:

```json
{
  "schema_version": "pi.context.v2",
  "context_kind": "scan|run_live|run_completed",
  "intent": "recommendation|debug|explain|correction|chat",
  "run_identity": {},
  "dataset_summary": {},
  "config": {},
  "parameter_catalog": {},
  "phase_facts": {},
  "image_quality": {},
  "artifact_index": {},
  "memory_context": {},
  "evidence_rules": {},
  "available_detail_chunks": []
}
```

## Semantische Kompression

Rohdaten werden nicht blind gekuerzt. Sie werden in Fakten uebersetzt:

```json
{
  "id": "pcc.residual_rms",
  "value": 0.3650483,
  "unit": "robust_fit_rms",
  "source": "logs/run_events.jsonl:PCC.phase_end",
  "applicability": "measured",
  "confidence": 1.0
}
```

Beispiele fuer sinnvolle Aggregate:

- `global_weights.p10`, `p50`, `p90`, `max`, `max_to_median`
- `registration.rejected_count`, `rejection_reasons`, `residual_p90`
- `pcc.status`, `stars_used`, `residual_rms`, `condition_number`, `matrix_diag`
- `validation.background_rms_increase_percent`, `background_rms_ok`
- `aqmh.selected_candidate`, `uniform_control_gate_triggered`, `cherry_pick_active`
- `canvas_mask.valid_fraction`, `black_border_fraction`

Das Modell bekommt damit die Bedeutung, nicht nur eine gekuerzte Datei.

## Parameter-Katalog

Jeder empfehlbare Parameter braucht eine autoritative Metadatenzeile:

```json
{
  "path": "pcc.max_residual_rms",
  "type": "number",
  "current_value": 0.9,
  "cpp_default": 0.35,
  "schema_min": ">0",
  "schema_max": null,
  "recommended_range": [0.25, 0.8],
  "unit": "robust PCC residual RMS",
  "phase": "PCC",
  "semantic": "rejects unstable or noisy PCC fits",
  "diagnostic_only": false,
  "disabled_value": null,
  "requires_evidence": [
    "pcc.status",
    "pcc.residual_rms",
    "pcc.stars_used"
  ],
  "hard_rules": [
    "Do not recommend a value below observed successful pcc.residual_rms unless PCC failed",
    "Do not claim schema maximum if schema_max is null",
    "Do not claim schema recommendation if recommended_value is null"
  ]
}
```

Dieser Katalog darf nicht aus Modellwissen entstehen. Er muss aus C++-Defaults, Schema, Konfigurationsdoku und kuratierten Regeln generiert oder validiert werden.

## Detail-Chunks

Nicht alle Daten passen oder muessen in jeden Prompt. Deshalb wird ein zweistufiges Modell verwendet:

1. Initialer Kontext mit kompakten Fakten.
2. Detail-Chunks, die bei Bedarf nachgeladen werden koennen.

Beispiel:

```json
{
  "chunk_id": "registration.rejected_frames.top20",
  "kind": "json_summary",
  "contains": ["frame_index", "cc", "reason", "tx", "ty"],
  "rows": 20,
  "available": true
}
```

Wenn die Chatfrage lautet "Warum wurden die Frames am Anfang abgewertet?", kann der Backend-Chat genau diesen Chunk in den naechsten Prompt aufnehmen.

## Technische Kompression

Technische Kompression ist erlaubt, aber nur ausserhalb des Modellprompts:

- Backend -> Sidecar: optional gzip/zstd.
- Sidecar: dekomprimiert deterministisch.
- Sidecar -> Modell: nur lesbarer JSON-/Text-Kontext.

Falls Provider APIs spaeter native Files, Tools oder Retrieval unterstuetzen, kann der Sidecar die komprimierten Artefakte speichern und dem Modell nur referenzierbare, entpackte Ausschnitte liefern.

## Antwortprotokoll der KI

PI muss Empfehlungen mit Fakten-IDs begruenden:

```json
{
  "path": "global_metrics.weight_exponent_scale",
  "value": 1.2,
  "current_value": 1.8,
  "confidence": 0.62,
  "review_required": true,
  "reason": "Global weights are concentrated; reducing exponent scale should smooth the distribution.",
  "evidence": [
    "global_weights.max_to_median",
    "global_weights.p90_to_p10",
    "pcc.status"
  ]
}
```

Freitext-Evidenz ohne existierende `fact_id` soll nicht als validiert gelten.

## Deterministische Validierung

Nach der Modellantwort prueft Backend/Sidecar:

- Pfad existiert im erlaubten Parameter-Katalog.
- Typ und Enum passen.
- Wert liegt in echten Bounds.
- Empfohlener Wert verletzt keine harte Regel.
- Jede genannte `fact_id` existiert.
- Die Begruendung behauptet keine nicht vorhandenen Schema-Defaults, Maxima oder Empfehlungen.
- Diagnostic-only Parameter werden nicht als Bildqualitaetsfix verkauft.
- Erfolgreiche Phasen werden nicht durch strengere Schwellen kaputt konfiguriert.

Nicht bestandene Vorschlaege gehen nach `rejected_recommendations` mit maschinenlesbarem Grund.

## Prompt-Regeln

Der Prompt soll kuerzer, aber haerter werden:

- Verwende nur Werte aus `facts` und `parameter_catalog`.
- Erfinde keine Messwerte.
- Erfinde keine Schema-Defaults, Maxima oder empfohlenen Werte.
- Wenn ein Feld `null` ist, ist es unbekannt.
- Aktuelle Werte, die dem C++ Default entsprechen, sind keine Fehlkonfiguration.
- Diagnostic-only Parameter duerfen nicht als Rekonstruktionsqualitaetsverbesserung begruendet werden.
- Jede Empfehlung braucht mindestens eine existierende `fact_id`.
- Empfehlungen ohne ausreichende Evidenz muessen `review_required=true` und niedrige Confidence haben.

## Run-Control-Chat

Der Chat in Run Control soll denselben Kontext verwenden wie die Recommendation-Pipeline.

Fuer fertige Runs:

- `context_kind=run_completed`
- vollstaendige Phase-Fakten
- Validierungsresultate
- Bildstatistiken
- Parameter-Katalog mit aktuellem Run-Config
- Artefaktindex

Fuer laufende Runs:

- `context_kind=run_live`
- bisherige Phasen
- aktive Phase
- letzte Events
- bekannte fehlende Daten explizit als `missing_facts`

Der Chat soll keine freien Parameterpatches erzeugen, sondern intern denselben validierten Recommendation-Mechanismus nutzen.

## Memories

Memory ist nur Hilfskontext, keine technische Wahrheit.

Prioritaet:

1. Run-Fakten
2. Parameter-Katalog
3. aktuelle Doku
4. Memory
5. astrophotografische Annahmen

Memory darf keine alten Defaults, angebliche Schema-Grenzen oder pauschale Best Practices ueber aktuelle Fakten stellen.

## Implementierungsphasen

### Phase 1: Fakten-Extraktor

- `pi.context.v2` Type/Schema definieren.
- Run-Artefakte in kanonische Fakten uebersetzen.
- `fact_id`, Quelle, Einheit und Applicability speichern.
- Scan-, Live- und Completed-Kontext unterscheiden.

### Phase 2: Parameter-Katalog

- Empfehlbare Parameter inventarisieren.
- Defaults und Constraints aus C++/Schema/Doku abgleichen.
- Semantische Regeln und Evidence-Anforderungen ergaenzen.
- Alte widerspruechliche UI-/Doku-Defaults sichtbar machen und bereinigen.

### Phase 3: Sidecar-Prompt umbauen

- Keine gekuerzten Roh-Dumps als Hauptkontext.
- Prompt auf Fakten, Parameter-Katalog und Evidence-Regeln ausrichten.
- Antwortformat auf `fact_id`-basierte Evidenz umstellen.

### Phase 4: Validator

- Post-Validation fuer Empfehlungen implementieren.
- Falsche Schema-Behauptungen erkennen.
- Empfehlungen gegen erfolgreiche Run-Metriken pruefen.
- Rejections in API und UI sichtbar machen.

### Phase 5: Run-Control-Chat anbinden

- Chat nutzt denselben Context Builder.
- Detail-Chunks bei Bedarf nachladen.
- Parameterkorrekturen laufen ueber denselben Recommendation-Validator.

### Phase 6: Tests und Fixtures

- Regression-Fixture fuer den M16-Fall:
  - `pcc.max_residual_rms=0.05` muss rejected werden.
  - `pcc.k_max=0.5` muss rejected werden.
  - `aqmh.pyramid.base_window_px=64` darf nicht als Schema-Default akzeptiert werden.
  - `validation.max_background_rms_increase_percent=0` darf nicht als Auto-Disable interpretiert werden.
- Tests fuer fehlende Fakten, diagnostic-only Parameter und Memory-Konflikte.

## Erwartetes Ergebnis

PI kann weiterhin frei formulieren und Zusammenhaenge erkennen, aber technische Wahrheit kommt aus deterministischen Kontextdaten. Kompression passiert dort, wo sie verlaesslich ist: technisch zwischen Prozessen, semantisch im Prompt. Damit werden falsche, selbstbewusste Empfehlungen wie erfundene Schema-Maxima oder unpassende PCC-Grenzwerte systematisch abgefangen.
