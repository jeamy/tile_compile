# PI Run-Chat Empfehlungs-Chat: Daten- und Validierungsplan

Status: Entwurf, 2026-07-17  
Ziel: Der run-spezifische PI-Chat soll belastbare, schema-gueltige und
pipeline-korrekte Empfehlungen liefern. Das betrifft Diagnose, konkrete
Config-Aenderungen, Action-Plans und Resume-Empfehlungen.

## Ausgangsproblem

Der Empfehlungs-Chat kann aktuell plausible, aber technisch falsche
Empfehlungen erzeugen. Ein konkretes Beispiel aus einem M42-Run:

- Es wurden Parameter wie `stretch.star_pressure` und `stretch.protect_b`
  empfohlen.
- Diese Pfade sind fuer `tile_compile` nicht die wirksamen Config-Pfade.
  Der relevante Config-Block heisst `hypermetric_stretch`.
- `star_pressure` ist ein diagnostischer Wert aus HyperMetric Stretch, aber
  kein normaler Config-Parameter.
- Als Resume-Start wurde `HYPERMETRIC_STRETCH` empfohlen, obwohl die
  wahrscheinlichen Ursachen vor HMS lagen: Crop, AQMH-Fallback, Stacking,
  Normalisierung oder BGE.

Das ist kein reines Modellproblem. Ein LLM kann nur dann verlaesslich
empfehlen, wenn es die technische Grenze kennt:

- welche Config-Pfade existieren,
- welche Werte erlaubt sind,
- welche Phase durch welchen Parameter beeinflusst wird,
- welche Artefakte und Diagnosewerte zu welcher Pipeline-Stufe gehoeren,
- welche Werte nur Messwerte sind und nicht gesetzt werden koennen.

Die zentrale Designregel lautet daher:

> Die PIAI darf interpretieren und priorisieren. Schema-Gueltigkeit,
> Action-Plan-Gueltigkeit und Resume-Phase werden deterministisch im Backend
> validiert und berechnet.

## Zielbild

Der Run-Chat erzeugt keine freien Ratschlaege, sondern ein strukturiertes
Ergebnis:

1. Diagnose mit Evidenzreferenzen.
2. Liste fehlender Evidenz, falls eine Ursache nicht belegbar ist.
3. Konkrete Empfehlungen mit Gueltigkeitsbereich.
4. Schema-validen Action-Plan.
5. Pro Action eine minimale Resume-Phase.
6. Eine final berechnete Resume-Empfehlung.
7. Warnungen fuer entfernte oder korrigierte Provider-Vorschlaege.

Der Benutzer soll danach erkennen koennen:

- warum ein Parameter empfohlen wird,
- welche Artefakte oder Vergleichswerte das stuetzen,
- welche Aenderungen wirklich anwendbar sind,
- ab welcher Phase ein Resume sinnvoll ist,
- welche Annahmen noch unsicher sind.

## Daten, die pro Run benoetigt werden

Der Chat braucht einen kompakten, aber vollstaendigen Run-Kontext. Rohdaten
muessen nicht an die PIAI gesendet werden. Wichtig sind strukturierte
Zusammenfassungen, Diagnosewerte und Bildkontext.

Empfohlener Container:

```json
{
  "schema_version": "pi.run-recommendation-context.v1",
  "run_id": "M42-pi_20260717_154431",
  "run_context": {
    "target": {},
    "status": {},
    "phase_status": [],
    "phase_order": []
  },
  "config": {
    "effective": {},
    "raw_path": "redacted-or-relative",
    "schema_summary": {},
    "path_phase_map": {},
    "diagnostic_only_fields": {}
  },
  "artifacts": {},
  "diagnostics": {},
  "image_context": {},
  "comparison_runs": [],
  "previous_turns": [],
  "memories": {}
}
```

### Begruendung

Ein Empfehlungs-Chat ist nur dann robust, wenn er nicht nur das finale Bild
sieht. Viele sichtbare Fehler entstehen frueher in der Pipeline und werden
durch spaetere Phasen nur sichtbarer gemacht. Ohne Phase- und Artefaktkontext
landet die Empfehlung zu schnell beim Stretching, weil Stretching nahe am
finalen Bild liegt.

## Effektive Config

Die PIAI braucht die effektive Config, nicht nur den rohen YAML-Text.

Erforderlich:

- aktuelle Werte nach Defaults und Normalisierung,
- nur nicht-sensitive Werte,
- stabile Pfade im Dot-Path-Format,
- Kennzeichnung von unbekannten oder ignorierten Pfaden.

Beispiel:

```json
{
  "config": {
    "effective": {
      "normalization.mode": "background",
      "output.crop_to_nonzero_bbox": true,
      "tile.min_size": 64,
      "tile.overlap_fraction": 0.25,
      "hypermetric_stretch.protect_b": 6.0
    },
    "unknown_or_ignored_paths": [
      {
        "path": "stretch.star_pressure",
        "reason": "not present in tile_compile schema; ignored by pipeline"
      }
    ]
  }
}
```

### Begruendung

Wenn der Chat nur Roh-YAML sieht, kann er ignorierte oder falsche Pfade als
wirksam behandeln. Genau dadurch entstehen Empfehlungen wie
`stretch.star_pressure`. Die effektive Config muss die Autoritaet sein.

## Schema-Summary fuer die PIAI

Die PIAI braucht keine vollstaendige Schema-Datei mit allen Beschreibungen.
Sie braucht eine kompakte, maschinenlesbare Zusammenfassung erlaubter
Empfehlungspfade.

Beispiel:

```json
{
  "schema_summary": {
    "valid_config_paths": {
      "hypermetric_stretch.protect_b": {
        "type": "number",
        "minimum": 0.1,
        "maximum": 5.0,
        "default": 6.0,
        "recommendation_allowed": true,
        "min_resume_phase": "HYPERMETRIC_STRETCH",
        "description": "B-channel protection during HMS"
      },
      "normalization.mode": {
        "type": "string",
        "enum": ["median", "background"],
        "recommendation_allowed": true,
        "min_resume_phase": "NORMALIZATION"
      },
      "output.crop_to_nonzero_bbox": {
        "type": "boolean",
        "recommendation_allowed": true,
        "min_resume_phase": "COMMON_OVERLAP"
      }
    },
    "invalid_or_diagnostic_paths": {
      "stretch.*": {
        "kind": "invalid",
        "reason": "No effective tile_compile config group named stretch for HMS tuning."
      },
      "star_pressure": {
        "kind": "diagnostic_only",
        "reason": "Estimated by HMS diagnostics; not a configurable input."
      }
    }
  }
}
```

### Begruendung

LLMs sind gut darin, plausible Parameternamen zu erzeugen. Das ist hier ein
Risiko. Der Prompt muss deshalb explizit sagen: Nur Pfade aus
`valid_config_paths` duerfen in den Action-Plan. Alles andere gehoert hoechstens
als Diagnose oder Warnung in Freitext.

## Phase- und Resume-Mapping

Jeder empfehlbare Config-Pfad braucht eine minimale Resume-Phase.

Beispiel:

```json
{
  "path_phase_map": {
    "data.*": "INPUT_SCAN",
    "calibration.*": "CALIBRATION",
    "registration.*": "REGISTRATION",
    "quality_filter.*": "QUALITY_FILTER",
    "normalization.*": "NORMALIZATION",
    "stacking.*": "STACKING",
    "output.crop_to_nonzero_bbox": "COMMON_OVERLAP",
    "tile.*": "TILE_RECONSTRUCTION",
    "aqmh.pyramid.*": "AQMH_MAPS",
    "aqmh.storage.*": "AQMH_MAPS",
    "aqmh.global_quality.*": "AQMH_GLOBAL_QUALITY",
    "aqmh.reconstruction.*": "AQMH_RECONSTRUCTION",
    "bge.*": "BGE",
    "pcc.*": "PCC",
    "hypermetric_stretch.*": "HYPERMETRIC_STRETCH"
  }
}
```

Die finale Resume-Phase wird nicht vom Modell uebernommen. Sie wird aus allen
validierten Actions berechnet:

1. Entferne ungueltige Actions.
2. Ermittle fuer jede Action die minimale Resume-Phase.
3. Waehle die frueheste Phase gemaess Pipeline-Order.
4. Falls keine Action vorhanden ist, liefere nur eine diagnostische Empfehlung
   mit niedriger Confidence.

Beispiel:

```json
{
  "actions": [
    {"path": "hypermetric_stretch.protect_b", "value": 3.5},
    {"path": "output.crop_to_nonzero_bbox", "value": false},
    {"path": "normalization.mode", "value": "median"}
  ],
  "computed_resume_phase": "COMMON_OVERLAP"
}
```

### Begruendung

Wenn mehrere Parameter geaendert werden, ist die spaeteste sichtbare Phase
nicht massgeblich. Massgeblich ist die frueheste betroffene Pipeline-Stufe.
Eine HMS-Resume-Empfehlung waere falsch, sobald eine Aenderung an Crop,
Normalisierung, Stacking, AQMH, BGE oder PCC beteiligt ist.

Wichtig: AQMH ist nicht eine einzige Resume-Stufe. Der Runner kennt
mindestens diese AQMH-nahen Phasen:

- `AQMH_MAPS`: Qualitaetskarten neu berechnen.
- `AQMH_GLOBAL_QUALITY`: globale Gewichte aus vorhandenen Maps neu bewerten.
- `AQMH_RECONSTRUCTION`: Rekonstruktion aus vorhandenen Prewarp-Frames,
  vorhandenen AQMH-Maps und vorhandener Canvas-Maske neu erzeugen.
- `AQMH_DIAGNOSTICS`: Diagnose der rekonstruierten Ausgabe.

`AQMH_RECONSTRUCTION` ist deshalb nur dann die richtige Startphase, wenn die
geanderten Parameter ausschliesslich die Rekonstruktion aus bereits gueltigen
AQMH-Maps betreffen. Wenn Maps, Pyramideneinstellungen, Storage-Aufloesung,
Canvas/Crop, Prewarp, Registration oder Common-Overlap betroffen sind, ist
`AQMH_RECONSTRUCTION` zu spaet.

## Run-Artefakte und Diagnosewerte

Der Chat braucht eine strukturierte Zusammenfassung pro Pipeline-Stufe.

### Crop und Canvas

Erforderlich:

```json
{
  "crop": {
    "crop_to_nonzero_bbox": true,
    "crop_x": 448,
    "crop_y": 588,
    "output_width": 3858,
    "output_height": 2194,
    "canvas_width": 4754,
    "canvas_height": 3370
  }
}
```

Begruendung: Wenn Nebel am Rand oder oben fehlt, ist Crop ein frueher und
haeufiger Hauptverdacht. Ohne diese Werte kann der Chat das Problem faelschlich
dem Stretching zuordnen.

### AQMH

Erforderlich:

```json
{
  "aqmh": {
    "enabled": true,
    "fallback_to_uniform_control": true,
    "uniform_control_blend_accepted": false,
    "uniform_control_blend_alpha": 0.0,
    "background_rms_regression": 637.23,
    "structure_masked_detail_applied": false
  }
}
```

Begruendung: AQMH-Fallbacks veraendern die lokale Rekonstruktion. Wenn ein
Nebelbereich schwach oder flach wirkt, muss der Chat wissen, ob die lokale
Rekonstruktion akzeptiert oder verworfen wurde.

### BGE

Erforderlich:

```json
{
  "bge": {
    "attempted": true,
    "applied": false,
    "success": false,
    "skip_reason": "background_chroma_worsened",
    "method": "autobge"
  }
}
```

Begruendung: BGE kann echten Nebel abschwaechen, aber auch Gradienten
verbessern. Eine pauschale Empfehlung `bge.enabled=false` ist nur sinnvoll,
wenn der Run-Kontext zeigt, dass BGE problematisch war oder der Vergleichsrun
ohne BGE besser war.

### PCC

Erforderlich:

```json
{
  "pcc": {
    "success": true,
    "stars_matched": 343,
    "stars_used": 305,
    "residual_rms": 0.3559,
    "condition_number": 1.701,
    "matrix_diagonal": [1.0689, 1.0, 1.7012]
  }
}
```

Begruendung: PCC-Parameter duerfen nicht aus allgemeinen Regeln gesetzt werden.
Wenn ein guter Vergleichsrun Residuals um 0.32 hatte, ist eine Empfehlung wie
`pcc.max_residual_rms=0.05` offensichtlich zu streng.

### Stacking und Rejection

Erforderlich:

```json
{
  "stacking": {
    "method": "rej",
    "sigma_low": 3.0,
    "sigma_high": 3.0,
    "cosmetic_correction_enabled": true,
    "cosmetic_correction_sigma": 10.0,
    "per_frame_cosmetic_correction_sigma": 5.0,
    "valid_mask_fraction": 0.94
  }
}
```

Begruendung: Schwarze Sternkerne koennen durch Rejection oder kosmetische
Korrektur entstehen. Ohne Stacking-Kontext ist `HYPERMETRIC_STRETCH` als
Resume-Phase nur geraten.

### HyperMetric Stretch

Erforderlich:

```json
{
  "hypermetric_stretch": {
    "enabled": true,
    "input_stage": "pcc",
    "protect_b": 6.0,
    "star_pressure": 0.745,
    "black_clip_percent": 0.0197,
    "white_clip_percent": 0.0160,
    "log_d": 3.951
  }
}
```

Begruendung: HMS-Diagnostik ist wichtig, aber nicht automatisch kausal.
`star_pressure` darf als Evidenz verwendet werden, aber nicht als
Config-Target. HMS-Resume ist nur korrekt, wenn die Aenderungen ausschliesslich
HMS betreffen oder ein Pre-HMS-Artefaktvergleich zeigt, dass das Problem erst
in HMS entsteht.

## Vergleichsrun-Kontext

Wenn der Nutzer einen guten oder schlechten Vergleichsrun nennt, muss der Chat
eine strukturierte Diff bekommen.

Beispiel:

```json
{
  "comparison_runs": [
    {
      "run_id": "m42_20260703_083337",
      "role": "better_reference",
      "config_diff": {
        "normalization.mode": {
          "current": "background",
          "reference": "median"
        },
        "tile.min_size": {
          "current": 64,
          "reference": 48
        },
        "tile.overlap_fraction": {
          "current": 0.25,
          "reference": 0.4
        }
      },
      "diagnostic_diff": {
        "crop_y": {
          "current": 588,
          "reference": 8
        },
        "hypermetric_stretch.star_pressure": {
          "current": 0.745,
          "reference": 0.760
        }
      }
    }
  ]
}
```

### Begruendung

Ein Vergleichsrun ist staerker als allgemeine Astrofotografie-Heuristik. Im
M42-Beispiel widerlegt der Vergleichsrun die einfache These
"hohe star_pressure verursacht schwarze Kerne", weil der bessere Run eine
hoehere `star_pressure` hatte.

## Bilddaten fuer die PIAI

Die PIAI sollte nicht nur ein finales Preview bekommen. Sinnvoll sind kleine,
gezielte Bildkontexte:

- finales Preview,
- Vergleichsrun-Preview,
- Crop des auffaelligen Sternkernbereichs,
- Crop des schwachen Nebelbereichs,
- optional Pre-HMS-Preview,
- optional Post-PCC/Pre-HMS-Preview,
- optional BGE-Preview,
- optional AQMH-Reconstruction-Preview.

Jedes Bild braucht Metadaten:

```json
{
  "image_id": "final_preview",
  "stage": "HYPERMETRIC_STRETCH",
  "width": 3858,
  "height": 2194,
  "crop_x": 448,
  "crop_y": 588,
  "source_artifact": "outputs/stacked_rgb_hms.png"
}
```

### Begruendung

Ein finales Bild kann zeigen, was falsch aussieht, aber nicht, wann es falsch
wurde. Pre-HMS- und Zwischenstufenbilder trennen Darstellungsfehler von
Pipeline-Fehlern.

## Prompt-Vertrag

Der Provider-Prompt muss harte Regeln enthalten.

Pflichtregeln:

- Antworte exakt als JSON-Objekt.
- Nutze nur Pfade aus `schema_summary.valid_config_paths` fuer
  `action_plan.actions`.
- Erzeuge keine Actions fuer Pfade aus `invalid_or_diagnostic_paths`.
- `star_pressure` ist nur Diagnose, kein Config-Pfad.
- `stretch.*` ist ungueltig.
- Jede konkrete Empfehlung mit Wert muss als Action kodiert werden.
- Jede Action muss `evidence_ref` und `min_resume_phase` enthalten.
- Wenn Evidenz fehlt, schreibe das in `missing_evidence` statt eine Ursache zu
  behaupten.
- HMS nur als Resume empfehlen, wenn alle validierten Actions
  `hypermetric_stretch.*` betreffen oder ein Pre-HMS-Vergleich beweist, dass
  das Problem erst in HMS entsteht.
- Vergleichsrun-Diffs haben Vorrang vor allgemeinen Heuristiken.

Beispiel-Promptabschnitt:

```text
CONFIG VALIDITY RULES:
- You may only place paths from valid_config_paths into action_plan.actions.
- Never use stretch.*.
- Never use star_pressure as an action path. It is diagnostic-only.
- If you mention a diagnostic-only field, mark it as evidence, not as a setting.

RESUME RULES:
- For every action, copy min_resume_phase from valid_config_paths.
- Do not choose HYPERMETRIC_STRETCH if any action requires an earlier phase.
- If the observed problem may originate before HMS and no pre-HMS image proves
  otherwise, prefer the earliest plausible diagnostic phase.
```

## Antwortschema der PIAI

Empfohlenes Schema:

```json
{
  "schema_version": "pi.run-chat-answer.v1",
  "summary": "string",
  "diagnosis": [
    {
      "text": "string",
      "confidence": "low|medium|high",
      "evidence_ref": "string"
    }
  ],
  "missing_evidence": [
    {
      "text": "string",
      "would_disambiguate": "string"
    }
  ],
  "recommendations": [
    {
      "text": "string",
      "confidence": "low|medium|high",
      "evidence_ref": "string"
    }
  ],
  "action_plan": {
    "schema_version": "pi.action-plan.v1",
    "source": "pi.run-chat.provider",
    "mutation_free": true,
    "actions": [
      {
        "id": "string",
        "type": "config.set",
        "path": "string",
        "value": "any",
        "min_resume_phase": "string",
        "rationale": "string",
        "evidence_ref": "string",
        "confidence": "low|medium|high"
      }
    ]
  },
  "resume_recommendation": {
    "from_phase": "string",
    "confidence": "low|medium|high",
    "reason": "string"
  },
  "warnings": []
}
```

### Begruendung

`recommendations` duerfen erklaerend sein. `action_plan.actions` muessen
ausfuehrbar sein. Diese Trennung verhindert, dass unscharfe Hinweise direkt zu
Config-Mutationen werden.

## Backend-Validierung nach Provider-Antwort

Nach der Provider-Antwort muss das Backend deterministisch validieren.

Algorithmus:

```text
parse provider JSON
validate response schema shape

for each action in action_plan.actions:
  require type == "config.set"
  require path in valid_config_paths
  require value matches type, enum, min, max
  require path is not diagnostic-only
  attach canonical min_resume_phase from backend map

drop invalid actions
record validation warnings for dropped actions

computed_resume_phase = earliest(min_resume_phase of remaining actions)

if provider resume phase is later than computed_resume_phase:
  override with computed_resume_phase
  add warning

if no valid actions remain:
  mark action_plan as diagnostic_only
  do not offer one-click apply
```

### Begruendung

Der Provider kann trotz Prompt Fehler machen. Die Sicherheitsgrenze darf nicht
im Prompt liegen, sondern im Backend.

## Resume-Korrektur

Die Resume-Empfehlung muss aus validierten Actions berechnet werden.

Beispiel:

```json
{
  "provider_resume": "HYPERMETRIC_STRETCH",
  "valid_actions": [
    {
      "path": "output.crop_to_nonzero_bbox",
      "value": false,
      "min_resume_phase": "COMMON_OVERLAP"
    },
    {
      "path": "normalization.mode",
      "value": "median",
      "min_resume_phase": "NORMALIZATION"
    }
  ],
  "computed_resume": "COMMON_OVERLAP",
  "warning": "Provider resume phase HYPERMETRIC_STRETCH is too late for the validated actions."
}
```

### Begruendung

Eine zu spaete Resume-Phase ist gefaehrlich, weil sie dem Nutzer eine schnelle
Neuberechnung verspricht, die die eigentliche Aenderung gar nicht wirksam
macht. Das Ergebnis sieht dann unveraendert aus, und PI lernt moeglicherweise
falsche Gegenbeispiele.

## Memory-Regeln

PI-Memory soll nicht nur erfolgreiche Empfehlungen speichern, sondern auch
Gegenbeispiele.

Beispiel:

```json
{
  "schema_version": "pi.memory.v2",
  "type": "counterexample",
  "source": "run_chat_feedback",
  "problem": {
    "classes": ["black_star_cores", "faint_nebula", "cropped_nebula"]
  },
  "bad_recommendation": {
    "actions": [
      {"path": "stretch.star_pressure", "value": 0.4}
    ],
    "resume_phase": "HYPERMETRIC_STRETCH"
  },
  "why_wrong": [
    "stretch.star_pressure is not schema-valid",
    "star_pressure is diagnostic-only",
    "comparison run had higher star_pressure but better visual result",
    "crop and AQMH diagnostics indicated earlier pipeline cause"
  ],
  "better_rule": "For this symptom set, inspect crop, AQMH, normalization and stacking before HMS."
}
```

### Begruendung

Ohne negative Memories wiederholt der Chat plausible Fehler. Gerade
Run-Chat-Probleme profitieren von Gegenbeispielen, weil sie oft aus einer
visuellen Fehlinterpretation entstehen.

## M42-Beispiel als Sollverhalten

Wenn der Nutzer meldet:

- Sterne in der Mitte schwarz,
- Nebel oben kaum sichtbar,
- Vergleichsrun sieht besser aus,

dann sollte der Chat folgende Daten priorisieren:

1. Crop-Diff:
   - schlechter Run: `crop_y=588`
   - guter Run: `crop_y=8`
2. AQMH-Diagnostik:
   - schlechter Run: Fallback zu uniform-control
3. Config-Diff:
   - schlechter Run: `normalization.mode=background`
   - guter Run: `normalization.mode=median`
   - schlechter Run: `tile.min_size=64`, `overlap_fraction=0.25`
   - guter Run: `tile.min_size=48`, `overlap_fraction=0.4`
4. HMS-Diagnostik:
   - schlechter Run: `star_pressure≈0.745`
   - guter Run: `star_pressure≈0.760`

Korrekte Schlussfolgerung:

- `star_pressure` ist nicht der Hauptbeweis.
- `stretch.star_pressure` darf nicht empfohlen werden.
- HMS-Resume ist fuer die Hauptkorrektur zu spaet.
- Erste sinnvolle Korrekturen betreffen Crop, Normalisierung, Tile/AQMH oder
  Stacking.

Beispiel fuer validen Action-Plan:

```json
{
  "actions": [
    {
      "type": "config.set",
      "path": "output.crop_to_nonzero_bbox",
      "value": false,
      "min_resume_phase": "COMMON_OVERLAP",
      "rationale": "The bad run cropped away much more top canvas than the reference run."
    },
    {
      "type": "config.set",
      "path": "normalization.mode",
      "value": "median",
      "min_resume_phase": "NORMALIZATION",
      "rationale": "The better reference run used median normalization."
    },
    {
      "type": "config.set",
      "path": "tile.overlap_fraction",
      "value": 0.4,
      "min_resume_phase": "TILE_RECONSTRUCTION",
      "rationale": "The better reference run used more overlap."
    }
  ],
  "computed_resume_phase": "COMMON_OVERLAP"
}
```

Optionaler HMS-A/B-Test:

```json
{
  "type": "config.set",
  "path": "hypermetric_stretch.protect_b",
  "value": 3.5,
  "min_resume_phase": "HYPERMETRIC_STRETCH",
  "rationale": "Fast display-only A/B test, not the main correction."
}
```

Dieser HMS-Test darf nicht die Haupt-Resume-Empfehlung ueberschreiben, wenn
andere Actions fruehere Phasen betreffen.

## Implementierungsplan

### Phase 1: Schema- und Phase-Kontext erzeugen

- Backend-Funktion `build_run_recommendation_schema_summary()` erstellen.
- Aus Config-Schema eine Liste erlaubter Dot-Paths erzeugen.
- Pro Pfad Typ, Enum, Min/Max und Default aufnehmen.
- Pro Pfad minimale Resume-Phase hinterlegen.
- Diagnostische Felder wie `star_pressure` explizit als nicht setzbar markieren.
- Bekannte ungueltige Aliase wie `stretch.*` markieren.

Abnahmekriterien:

- `hypermetric_stretch.protect_b` ist erlaubt.
- `stretch.protect_b` ist verboten.
- `stretch.star_pressure` ist verboten.
- `star_pressure` ist diagnostic-only.

### Phase 2: Run-Kontext erweitern

- `run.report.summary` und `run.artifacts.summary` um kompakte
  Diagnosebereiche erweitern.
- Crop-, AQMH-, BGE-, PCC-, Stacking- und HMS-Diagnostik normalisieren.
- Effektive Config mit unbekannten/ignorierten Pfaden liefern.
- Vergleichsrun-Diffs erzeugen, wenn ein Vergleichsrun genannt oder erkannt
  wird.

Abnahmekriterien:

- M42-Kontext enthaelt `crop_y`, AQMH-Fallback, BGE-Skip, PCC-Stats und
  HMS-Diagnostik.
- Vergleichsrun-Diff zeigt konkrete Parameter- und Diagnoseunterschiede.

### Phase 3: Prompt haerten

- `build_provider_run_chat_prompt()` um Schema- und Resume-Regeln erweitern.
- `schema_summary`, `path_phase_map` und `diagnostic_only_fields` in den
  AI-Request aufnehmen.
- Prompt-Regeln fuer `stretch.*`, `star_pressure` und HMS-Resume aufnehmen.
- Provider auffordern, fehlende Evidenz explizit zu nennen.

Abnahmekriterien:

- Prompt enthaelt eine maschinenlesbare Liste erlaubter Pfade.
- Prompt verbietet `stretch.*`.
- Prompt verlangt `min_resume_phase` pro Action.

### Phase 4: Provider-Action-Plan validieren

- Nach Provider-Antwort jeden Action-Pfad gegen `valid_config_paths` pruefen.
- Werte gegen Typ, Enum, Min/Max pruefen.
- Ungueltige Actions entfernen und als Warning melden.
- `min_resume_phase` aus Backend-Mapping ueberschreiben.
- Keine One-Click-Anwendung erlauben, wenn keine gueltigen Actions bleiben.

Abnahmekriterien:

- Provider-Antwort mit `stretch.star_pressure` wird nicht anwendbar.
- Provider-Antwort mit `hypermetric_stretch.protect_b` bleibt gueltig.
- Ungueltige Actions erscheinen in `warnings` oder `rejected_actions`.

### Phase 5: Resume deterministisch berechnen

- Aus allen validierten Actions die frueheste Phase berechnen.
- Provider-Resume nur als Hinweis verwenden.
- Wenn Provider-Resume zu spaet ist, Backend-Resume ueberschreiben.
- Wenn Action-Plan leer ist, keine scheinbar sichere Resume-Phase ausgeben.

Abnahmekriterien:

- `output.crop_to_nonzero_bbox=false` fuehrt nicht zu `HYPERMETRIC_STRETCH`.
- Kombination aus `normalization.mode` und `hypermetric_stretch.protect_b`
  fuehrt zu `NORMALIZATION`.
- Kombination aus `output.crop_to_nonzero_bbox` und HMS-Aenderung fuehrt zu
  `COMMON_OVERLAP`.
- AQMH-Rekonstruktionsparameter fuehren zu `AQMH_RECONSTRUCTION`, aber
  AQMH-Map-/Pyramidenaenderungen fuehren zu `AQMH_MAPS` oder frueher.

### Phase 5b: Resume-Fehler sauber sichtbar machen

Der Runner darf nach `resume_start` nicht ohne `resume_end` abbrechen. Fuer
jede Resume-Phase gilt:

- bei fruehen Prueffehlern `resume_end success=false` schreiben,
- wenn eine Phase schon begonnen hat, auch `phase_end status=error` schreiben,
- `stderr`-Fehler als strukturiertes Event mit `reason` spiegeln,
- UI-Status darf nicht in einem laufenden Resume haengen bleiben.

Speziell fuer `AQMH_RECONSTRUCTION` muessen diese Fehler strukturiert geloggt
werden:

- fehlendes `artifacts/aqmh_metrics.json`,
- fehlendes `cache/aqmh/aqmh_cache.json`,
- ungueltige AQMH-Metadaten,
- fehlende oder dimensionsfalsche `outputs/canvas_mask.fits`,
- fehlende `cache/aqmh_masks` beim Rebuild der Full-Canvas-Maske,
- fehlende `.prewarped_cache` Frames,
- Fehler beim Persistieren von `reconstructed_L.fit` oder `synthetic_0.fit`,
- Fehler in `AQMH_DIAGNOSTICS`.

Begruendung: Ohne `resume_end` sieht der Run so aus, als wuerde er laufen oder
haengen. Der Empfehlungs-Chat kann dann nicht lernen, dass die Resume-Phase
nicht praktikabel war.

### Phase 6: Memory und Feedback

- Counterexample-Memories fuer abgelehnte oder als falsch markierte
  Empfehlungen speichern.
- Resume-Feedback mit Run-Kontext und Phase speichern.
- Positive Memories nur aus erfolgreichen, validierten und vom Nutzer
  bestaetigten Ergebnissen erzeugen.

Abnahmekriterien:

- Falsche Empfehlung `stretch.star_pressure` wird als Gegenbeispiel lernbar.
- Spaetere Prompts erhalten relevante negative Memories.
- Memories enthalten keine absoluten Rohbildpfade und keine Secrets.

### Phase 7: Tests

Minimal notwendige Tests:

- `run_chat_rejects_invalid_stretch_paths`
- `run_chat_accepts_hypermetric_stretch_paths`
- `run_chat_treats_star_pressure_as_diagnostic_only`
- `run_chat_computes_resume_from_valid_actions`
- `run_chat_overrides_provider_hms_resume_when_actions_need_earlier_phase`
- `run_chat_keeps_hms_resume_for_hms_only_action`
- `run_chat_m42_context_prefers_crop_or_stacking_over_hms`
- `run_chat_records_counterexample_for_rejected_provider_action`

### Begruendung

Diese Tests decken genau die Fehlerklasse ab, die beim M42-Run sichtbar wurde:
gueltig klingende falsche Pfade und zu spaete Resume-Phasen.

## Sicherheits- und Datenschutzregeln

- Keine API-Keys, Tokens oder lokalen Secrets in Prompts.
- Keine absoluten Rohbildpfade in Memories.
- Bilder nur als reduzierte Previews oder Crops senden.
- Config-Pfade und Diagnosewerte duerfen gesendet werden.
- Provider-Antworten nie direkt schreiben.
- Alle schreibenden Aktionen bleiben Preview- und Apply-pflichtig.

## Zusammenfassung

Ein funktionierender PI-Empfehlungs-Chat braucht drei Ebenen:

1. Gute Evidenz: Run-Artefakte, Diagnosewerte, Bilder und Vergleichsrun-Diffs.
2. Harte Grenzen: Schema-gueltige Pfade, diagnostic-only Felder und
   Phase-Mapping.
3. Deterministische Nachbearbeitung: Action-Plan-Validierung und
   Resume-Berechnung im Backend.

Damit wird verhindert, dass die PIAI technische Scheinparameter empfiehlt oder
eine zu spaete Resume-Phase waehlt. Die KI bleibt fuer Interpretation und
Priorisierung zustaendig; die Pipeline-Regeln bleiben im Code.
