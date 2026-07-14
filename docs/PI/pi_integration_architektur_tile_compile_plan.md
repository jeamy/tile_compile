# PI-Integration Architekturplan fuer Tile Compile

Status: rekonstruiert und neu geschrieben am 2026-07-14  
Zieloberflaeche: GUI3 (`web_frontend_v3`). Die alte GUI (`web_frontend`) ist nicht Ziel dieser Integration.

Reset-Entscheidung: Bereits gespeicherte AI-/PI-Memory-Daten muessen nicht
migriert oder kompatibel weitergelesen werden. Die naechste Memory-Iteration
startet bei null mit einem neuen Store- und Schema-Vertrag.

## Ausgangspunkt

Tile Compile hat PI bereits als Parameter-Optimierung im Scan-AI-/Empfehlungsfluss. Die neue Integration erweitert das zu einer kontrollierten PI-Schicht, die Kontext versteht, Empfehlungen in Action-Plans ueberfuehrt, Aenderungen vorab validiert, eine sichere Vorschau liefert, explizit angewendet wird und aus geprueften Optimierungen ueber Sessions hinweg lernen kann.

- Tile Compile arbeitet config- und run-zentriert, daher ist `validate-config` die zentrale Sicherheitsgrenze.
- PI-Antworten werden nicht direkt geschrieben, sondern als `pi.action-plan.v1` modelliert.
- Memories lernen nicht automatisch "Wahrheiten", sondern speichern reviewbare Optimierungs-Erfahrungen.
- Memories werden global gespeichert und ueber Projekte hinweg abgerufen. Projekt-/Run-IDs dienen nur als Evidenzreferenz, nicht als Speichergrenze.
- Memory-Retrieval nutzt eine detaillierte Kontextsignatur aus Objekt, Zieltyp, Aufnahmegeraet, Optik, Kamera, Filter, Belichtungen, Kalibrierung, Montierung, Himmel/Qualitaet und Pipeline-Konfiguration.
- Bisherige AI-/Memory-Dateien sind fuer die neue Architektur keine Quelle der Wahrheit und duerfen ignoriert werden.
- GUI3 bleibt der einzige neue Bedienpfad.

## Zielbild

PI wird in Tile Compile zu einer Assistenz- und Orchestrierungsschicht:

- Kontext lesen: Runtime, aktuelle Config, Scan-Ergebnisse, Jobs, Artefakte, Reports.
- Vorschlagen: Scan-AI-Empfehlungen und spaeter weitere PI-Tools erzeugen strukturierte Action-Plans.
- Pruefen: Action-Plans werden serverseitig formal und config-semantisch validiert.
- Vorschau: GUI3 zeigt mutationfreie YAML-Diffs.
- Anwenden: Schreibende Aktionen brauchen explizite Bestaetigung und erzeugen Config-Revisionen.
- Lernen: Erfolgreich angewendete Optimierungen koennen als Memory-Kandidaten gespeichert und reviewed werden.
- Erinnern: Akzeptierte Memories werden semantisch und strukturell gesucht, mit Evidenz und Gueltigkeitsbereich gewichtet und der KI als begrenzter Kontext gegeben.

## Sicherheitsregeln

- Kein Blind-Write: Jede Config-Aenderung laeuft ueber Action-Plan, Preview und Validierung.
- Kein ungeprueftes Lernen: Memories starten als `candidate`; erst Review macht sie fuer spaetere Sessions belastbar.
- Keine alte GUI: Neue PI-Funktionalitaet wird nur in `web_frontend_v3` eingebaut.
- Privacy by default: Memories speichern Metadaten, Config-Pfade, Gruende und Validierung, aber keine Bilddaten.
- Globale Memory-Policy: Lernen ist nicht projektbezogen. Es werden keine absoluten Bildpfade, keine Rohbilder und keine vertraulichen lokalen Pfade als Knowledge gespeichert. Stattdessen werden stabile, normalisierte Metadaten und optional gehashte Run-/Artefakt-Referenzen genutzt.
- Keine Legacy-Pflicht: Es gibt keinen Migrationspfad fuer bisherige AI-/Memory-Daten. Ein alter Store wird fuer die neue Schicht ignoriert und nicht automatisch ausgewertet.
- Keine unerklaerte KI-Autoritaet: Jede Memory-Nutzung muss im Prompt als historische Erfahrung mit Geltungsbereich, Confidence, Evidenz und moeglichen Gegenbeispielen markiert werden.
- Bestehende Tile-Compile-Kommandos bleiben die Autoritaet fuer Config-Gueltigkeit.

## Rekonstruierter Implementierungsstand

- [x] Plan-Datei in `docs/PI/pi_integration_architektur_tile_compile_plan.md` rekonstruiert.
- [x] Backend-PI-Routen rekonstruiert und in `main.cpp` registriert.
- [x] PI-Service-Dateien fuer Action-Plans, Validator, Context, Tools, Assistant und Memories rekonstruiert.
- [x] CMake-Ziele und Tests fuer PI-Komponenten rekonstruiert.
- [x] Scan-AI-Routen um PI Action-Plan-Anreicherung erweitert.
- [x] Scan-AI Apply um `learn=true` Memory-Kandidaten erweitert.
- [x] GUI3-Endpunkte und AI-Empfehlungsseite um PI Preview, PI Apply und Memories erweitert.
- [x] Alte GUI von versehentlichen PI-Aenderungen bereinigt.
- [x] Agent-Service Traffic-Logging zentralisiert und mit Redaction versehen.

Verifikation:

- [x] `node --check web_frontend_v3/js/pages/ai-empfehlung.js`
- [x] `node --check web_frontend_v3/js/api/endpoints.js`
- [x] `npm run build` in `agent_service`
- [x] Backend Build fuer `tile_compile_web_backend`, `fake_tile_compile_cli`, PI-Tests und AI-Routen-Test
- [x] `ctest --output-on-failure -R 'web_backend_cpp_(pi_routes|pi_memory_store|pi_action_plan|ai_routes)'`

## Phase 0 - Fundament

Ziel: Gemeinsame Sprache und Sicherheitsmodell fuer PI schaffen.

- [x] Action-Plan Schema `pi.action-plan.v1` definieren.
- [x] `config.set` als erste sichere Action implementieren.
- [x] Action-Plan Shape-Validator implementieren.
- [x] Scan-AI `validated_updates` in Action-Plans umwandeln.
- [x] Backend-Tests fuer Action-Plan-Erzeugung und Validierung anlegen.
- [x] PI-Traffic-Log aus Agent-Service duplizierten Stellen herausziehen.
- [x] Secrets, Bearer Tokens, API Keys und lokale Projektpfade im Traffic-Log redacten.

Abnahmekriterien:

- [x] Action-Plan ohne Schema oder ohne gueltige Actions wird abgelehnt.
- [x] Scan-AI-Antworten enthalten `action_plan` und `action_plan_validation`.
- [x] Logging-Fehler brechen Analyse nicht ab.

## Phase 1 - PI Kontext, Tools und Assistant

Ziel: PI kann Tile-Compile-Kontext kontrolliert lesen, ohne Schreibrechte.

- [x] `/api/pi/context` implementieren.
- [x] `/api/pi/tools` implementieren.
- [x] `/api/pi/tools/call` implementieren.
- [x] `/api/pi/assistant/ask` implementieren.
- [x] Tool `context.overview` bereitstellen.
- [x] Tool `config.schema` bereitstellen.
- [x] Tools fuer Artefakt-/Report-Zusammenfassung bereitstellen.
- [x] Vorschau-Tools fuer BGE/HMS als planende Read-/Preview-Tools bereitstellen.

Abnahmekriterien:

- [x] Kontextantwort enthaelt Runtime-, State-, Job- und Scan-Informationen.
- [x] Unbekannte Tools werden abgelehnt.
- [x] Assistant antwortet ohne schreibende Nebenwirkung.
- [x] PI-Routen-Test deckt Kontext, Tools und Assistant ab.

## Phase 2 - Action-Plan Preview und kontrolliertes Anwenden

Ziel: PI-Vorschlaege koennen sicher vorab gesehen und erst dann angewendet werden.

- [x] `/api/pi/action-plans/validate` implementieren.
- [x] `/api/pi/action-plans/preview` implementieren.
- [x] Preview setzt Config-Aenderungen mutationfrei auf eine Basiskonfig.
- [x] Preview prueft Ergebnis mit `validate-config --stdin`.
- [x] `/api/pi/action-plans/apply` implementieren.
- [x] Apply erfordert `confirmed=true`.
- [x] Apply kann `expected_patched_yaml` gegen Preview-Ergebnis pruefen.
- [x] Apply speichert Config-Revision und UI-Event.
- [x] GUI3 AI-Seite erzeugt Action-Plans aus ausgewaehlten Empfehlungen.
- [x] GUI3 zeigt PI Preview als YAML-Diff.
- [x] GUI3 erlaubt explizites PI-Anwenden.

Abnahmekriterien:

- [x] Unbestaetigtes Apply wird abgelehnt.
- [x] Preview erzeugt YAML-Diff ohne Config-Datei zu schreiben.
- [x] Apply erzeugt Revision.
- [x] Tests decken Preview, Apply und invaliden Config-Fall ab.

## Phase 3 - Memories und Lernen ueber Sessions

Ziel: Tile Compile merkt sich nuetzliche Optimierungsentscheidungen reviewbar und sessionuebergreifend.

- [x] PI Memory Store als JSONL-Dateien unter einem konfigurierbaren PI-Storage implementieren.
- [x] Erste Memory-Speicherung implementiert; wird fuer die neue globale Memory-Schicht ignoriert.
- [x] Kandidaten mit `append_candidate` speichern.
- [x] `/api/pi/memories` fuer Liste und Statusfilter implementieren.
- [x] `/api/pi/memories/:id/review` fuer `accepted`, `rejected`, `deprecated` implementieren.
- [x] `/api/pi/memories/retrieve` fuer einfache pfad-/typbasierte Suche implementieren.
- [x] Scan-AI Apply kann bei `learn=true` ein `config_optimization` Memory erzeugen.
- [x] GUI3 AI-Seite zeigt Memory-Liste und Review-Aktionen.
- [ ] Memory-Speicherung von "projektbezogen" auf "global nutzbar" schaerfen: Projekt/Run nur als Provenance, nicht als Retrieval-Grenze.
- [ ] Memory-Kandidaten beim Erzeugen mit einer vollstaendigen Astro-Kontextsignatur anreichern.
- [ ] Memory-Kandidaten nicht nur aus Apply erzeugen, sondern erst als lernwuerdig markieren, wenn eine Outcome-Evidenz oder ein Nutzerfeedback vorhanden ist.
- [ ] Neuen Memory-Store ohne Ruecksicht auf alte gespeicherte AI-/Memory-Daten initialisieren.

Abnahmekriterien:

- [x] Memory Store Test deckt Append, List, Review und Retrieve ab.
- [x] PI-Routen-Test deckt Memory-List, Review und Retrieve ab.
- [x] AI-Routen-Test prueft Memory-Kandidat nach `learn=true`.
- [ ] Memory-JSON enthaelt keine absoluten lokalen Bild-/Projektpfade.
- [ ] Memory-JSON enthaelt mindestens `context_signature`, `evidence`, `scope`, `outcome`, `review` und `provenance`.
- [ ] GUI3 zeigt klar, warum ein Memory global wiederverwendbar ist oder warum es nur lokal/eingeschraenkt gelten darf.

Noch offen in Phase 3:

- [x] Akzeptierte Memories beim naechsten Scan-AI-Request automatisch als Kontext beilegen.
- [x] GUI3 deutlicher anzeigen, welche Memories nur Kandidaten sind und welche accepted/deprecated sind.
- [x] Memory-Deduplikation einbauen, damit identische Config-Optimierungen nicht mehrfach wachsen.

## Phase 4 - Memory Retrieval im Optimierungsfluss

Ziel: PI nutzt akzeptierte Erfahrungen, ohne sie unkritisch zu kopieren.

- [x] Beim Aufbau des Scan-AI Request relevante Memories anhand von Config-Pfaden abrufen.
- [x] Nur `accepted` Memories als starken Kontext verwenden; `candidate` wird nicht in den Request-Kontext uebernommen.
- [x] Memory-Kontext im Prompt klar als "historische Erfahrung" markieren.
- [x] Ablehnungsstatus (`rejected`, `deprecated`) als Negativsignal nutzbar machen.
- [x] Memory-Retrieval mit Tests gegen Fehluebernahme absichern.
- [ ] Retrieval nicht nur anhand von Config-Pfaden, sondern anhand einer gewichteten Kontextsignatur durchfuehren.
- [ ] Positive und negative Memories gemeinsam abrufen: accepted als moegliche Strategie, rejected/deprecated als Warnung vor Wiederholung.
- [ ] Retrieval begrenzen: max. relevante Memories, Diversity nach Objekt-/Optik-/Kamera-Klassen, keine Prompt-Flutung.
- [ ] Retrieval-Erklaerung in AI-Kontext aufnehmen: warum dieses Memory passend ist, welche Felder matchen, welche nicht.

Abnahmekriterien:

- [x] Ein neuer Scan-AI-Request enthaelt passende accepted Memories.
- [x] Rejected Memories werden nicht als Empfehlungskontext genutzt.
- [x] Tests pruefen, dass Memories keine Schema-/Config-Validierung umgehen.
- [ ] Tests pruefen, dass ein Memory fuer z.B. "M42, Nebel, OSC, kurze Brennweite, ausgedehnte Emission" nicht blind auf "M104, Galaxie, Mono/LRGB" angewendet wird.
- [ ] Tests pruefen, dass ein rejected Memory bei aehnlichem Kontext als explizites Negativsignal im Prompt erscheint.

## Phase 5 - Outcome-Metriken und Qualitaetsfeedback

Ziel: Lernen wird besser als blosses "wurde angewendet".

- [ ] Nach Runs relevante Outcome-Metriken erfassen: Validierung, Artefakte, Warnungen, Report-Status, ggf. Qualitaetsmetriken.
- [x] Memory-Kandidaten um Outcome-Felder erweitern.
- [x] GUI3 Review zeigt angewendete Pfade, Gruende, Validierung und Outcomes.
- [x] Accepted-Memories nach positiver Outcome-Evidenz hoeher gewichten.
- [x] Deprecated-Memories fuer verschlechterte oder ueberholte Optimierungen unterstuetzen.
- [ ] Outcome-Delta statt Einzelwert speichern: Vorher/Nachher fuer Report-Warnungen, Sternmetriken, Hintergrundgradient, Farbkalibrierung, Artefaktstatus, Resume-Phase und Nutzerbewertung.
- [ ] Memory erst als `promotable` markieren, wenn mindestens eine positive Evidenz vorliegt: besserer Report, besseres Artefakt, akzeptiertes Nutzerfeedback oder reproduzierter Erfolg.
- [ ] Negative Learning unterstuetzen: Wenn ein Vorschlag keine Verbesserung brachte, als `rejected`/`counterexample` mit gleichem Kontext speichern.

## Phase 6 - Erweiterte PI-Werkzeuge

Ziel: PI kann weitere Tile-Compile-Arbeitsablaeufe planen, aber weiterhin kontrolliert.

- [x] BGE-Plan-Tool mit echten Run-/Config-Daten vertiefen.
- [x] HMS-/Mosaik-Plan-Tool mit echten Projektparametern vertiefen.
- [x] Resume-/Run-Planung als read-only Plan erzeugen.
- [x] Schreibende Tools erst nach Action-Plan/Preview/Apply freischalten.
- [x] Tool-Registry versionieren und dokumentieren.

## Phase 7 - Audit, Export und Betrieb

Ziel: PI-Aktionen bleiben nachvollziehbar und wartbar.

- [x] GUI3 Audit-Ansicht fuer Action-Plans, Applies, Revisionen und Memory-Reviews.
- [x] Export/Import fuer PI Memories mit Privacy-Filter.
- [x] CLI-/Backend-Werkzeug zum Aufraeumen und Deduplizieren von Memories.
- [x] Regressionstests fuer typische Optimierungsfaelle: OSC/MONO, BGE, HMS, AQMH, PCC.
- [x] Nutzer-Dokumentation fuer Workflow: Empfehlung, Preview, Apply, Learn, Review.

## Phase 8 - Run-Chat und natuerliches Qualitaetsfeedback

Ziel: Nach einem abgeschlossenen Run kann der Nutzer in normaler Sprache beschreiben, was am Bild falsch wirkt. PI verbindet diese Beschreibung mit Run-Kontext, Artefakten, Reports, Config und Memories und erzeugt nachvollziehbare Diagnose- und Optimierungsvorschlaege.

Beispiel aus `runs/run_20260714_091851`:

> Sterne oben haben schwarzen Kern. Der Nebel oben wird nicht einbezogen, sondern beschnitten und ist kaum sichtbar. Was kann man tun?

Geplanter Workflow:

1. Nutzer oeffnet einen fertigen Run in GUI3.
2. Chat-Panel bietet ein Eingabefeld fuer natuerliche Problembeschreibung.
3. Backend baut einen `pi.run-chat-context.v1` aus Run-Status, Config-Revision, Report-Stats, Artefakten, Phasenereignissen, Scan-Metriken und relevanten Memories.
4. PI beantwortet in normaler Nutzersprache: wahrscheinliche Ursachen, zu pruefende Artefakte, konkrete naechste Schritte.
5. Wenn sinnvoll, erzeugt PI zusaetzlich einen `pi.action-plan.v1` fuer sichere Config-Aenderungen.
6. GUI3 zeigt Antwort, Evidenz, Artefakt-Links und optional PI Preview/Apply.
7. Nutzer kann den Chat-Ausgang als Memory-Kandidat speichern, wenn ein spaeterer Run die Verbesserung bestaetigt.

Umsetzungsschritte:

- [x] GUI3 Run-Chat-Panel im Run Monitor ergaenzen; History bleibt ohne Chat-Elemente.
- [x] `/api/pi/run-chat` als read-only Diagnose-Endpoint implementieren.
- [x] Run-Kontextbuilder fuer abgeschlossene Runs bauen: Report, Artefakte, Config/Preview-Kontext, relevante Metriken und Memories.
- [x] Natuerliche Nutzerbeschreibung strukturiert in Problem-Hinweise uebersetzen, ohne sie als harte Wahrheit zu behandeln.
- [x] Antwortformat definieren: `summary`, `likely_causes`, `checks`, `recommendations`, `evidence`, optional `action_plan`.
- [x] Typische Bildprobleme als kontrollierte Hinweise modellieren: schwarze Sternkerne, beschnittener Nebel, zu dunkle Nebelanteile, Hintergrundgradient, Farbstich, Tile-Muster, unscharfe Sterne.
- [x] Chat-Antworten mit bestehender PI Preview verbinden; Apply bleibt bewusst separat und reviewpflichtig.
- [x] Letztes Run-Bild in Run History und Run Monitor oberhalb des Chats anzeigen.
- [x] Bildpreview beim ersten Seitenaufbau nur bei fehlender Preview erzeugen; weitere Regeneration nur per explizitem Refresh-Button.
- [x] Folgefragen im Run-Monitor-Chat mit lokalem Chat-Verlauf unterstuetzen.
- [x] PI schlaegt fuer Resume eine passende Startphase vor; Nutzer waehlt sie explizit aus.
- [x] Tests mit Fixture-Run und Beispielproblemen anlegen.

Abnahmekriterien:

- [x] Chat funktioniert ohne Schreibzugriff und ohne vorhandene Bilddaten in Memories zu speichern.
- [x] Antwort nennt konkrete Run-Artefakte oder Report-Fakten als Evidenz.
- [x] Empfehlungen koennen optional als Action-Plan validiert und previewed werden.
- [x] Preview-Refresh wird nicht mehr durch Polling, Resume-Status oder Terminal-Events erzwungen.
- [x] Run History enthaelt keine Chat-Controls oder Run-Chat-Action-Plan-Elemente mehr.
- [x] Nutzertext wie "Sterne haben schwarzen Kern" fuehrt zu nachvollziehbaren Checks statt zu blindem Parameter-Raten.

## Phase 9 - Globaler AI Memory Layer

Ziel: PI wird zu einer professionellen, global lernenden Wissensschicht fuer
Astro-Optimierungen. Gelernt wird nicht "dieses Projekt hatte diese Config",
sondern "unter diesem Aufnahme-/Objekt-/Pipeline-Kontext war diese Strategie
mit dieser Evidenz sinnvoll oder nicht sinnvoll".

Grundsatz:

- Global statt projektbezogen: Memories liegen im zentralen PI-Storage und sind ueber alle Runs/Projekte hinweg nutzbar.
- Kontext statt Anekdote: Jedes Memory muss seine fachliche Geltungsbedingung beschreiben.
- Evidenz statt Bauchgefuehl: Jedes Memory braucht eine Provenance, Outcome-Information und Review-Status.
- Retrieval statt blindem Prompt-Anhaengen: Nur passende, begrenzte, erklaerte Memories werden an die KI gegeben.
- Negative Erfahrungen sind wertvoll: Nicht erfolgreiche Vorschlaege werden als Gegenbeispiele gespeichert.

### 9.1 Memory-Datentypen

Neue oder geschaerfte Memory-Typen:

- `config_optimization`: Eine konkrete Parameterstrategie war unter bestimmtem Kontext sinnvoll.
- `artifact_diagnosis`: Ein sichtbares Problem wurde mit Ursachen/Checks/Phasen verbunden.
- `resume_strategy`: Eine Resume-Phase war fuer eine Problemklasse sinnvoll oder nicht sinnvoll.
- `provider_prompt_pattern`: Ein Prompt-/Kontextmuster fuehrte zu besser strukturierten KI-Antworten.
- `counterexample`: Eine Empfehlung war trotz aehnlichem Kontext nicht hilfreich.
- `user_preference`: Nutzerpraeferenzen fuer Darstellung/Stretch/Detailgrad, sofern explizit bestaetigt.

### 9.2 Globales Memory-Schema `pi.memory.v2`

`pi.memory.v2` ist der neue Startpunkt. Fruehere Drafts oder bisher
gespeicherte AI-Daten werden nicht migriert und nicht kompatibel weitergelesen.
Alte Dateien werden fuer die neue Memory-Schicht ignoriert oder nur als manuell
exportierte Referenz betrachtet, nie als automatisch vertrauenswuerdige Quelle.

Pflichtfelder:

- `schema_version`: `pi.memory.v2`
- `id`: stabile ID
- `type`: Memory-Typ
- `status`: `candidate`, `promotable`, `accepted`, `rejected`, `deprecated`
- `privacy_class`: z.B. `metadata_only`
- `created_at`, `updated_at`
- `source`: `scan_ai_apply`, `run_chat`, `resume_feedback`, `manual_review`, `outcome_evaluator`
- `summary`: kurze fachliche Aussage
- `recommendation`: strukturierte Empfehlung oder Warnung
- `context_signature`: normalisierte Kontextsignatur
- `scope`: Geltungsbereich und Grenzen
- `evidence`: Provenance und Belege
- `outcome`: Ergebnisbeobachtung und Vorher/Nachher-Deltas
- `review`: menschliche Review-Information
- `retrieval`: Such-/Ranking-Hilfen

Beispielstruktur:

```json
{
  "schema_version": "pi.memory.v2",
  "type": "config_optimization",
  "status": "candidate",
  "privacy_class": "metadata_only",
  "summary": "Bei ausgedehnten Nebeln mit OSC-Daten BGE konservativ einsetzen, weil schwache Emission als Hintergrund entfernt werden kann.",
  "context_signature": {
    "target": {
      "object_name": "M42",
      "object_type": "emission_nebula",
      "angular_size_class": "large",
      "has_extended_emission": true
    },
    "acquisition": {
      "camera_type": "OSC",
      "filters": ["dual_narrowband"],
      "exposure_seconds_median": 180,
      "frame_count": 120,
      "total_integration_minutes": 360,
      "calibration": {
        "darks": true,
        "flats": true,
        "bias": false
      }
    },
    "optics": {
      "telescope": "unknown_or_redacted",
      "focal_length_mm": null,
      "f_ratio": null,
      "pixel_scale_arcsec": null
    },
    "mount": {
      "type": "EQ",
      "tracking_quality": "unknown"
    },
    "pipeline": {
      "affected_paths": ["bge.enabled", "stretch.target_background"],
      "phases": ["BGE", "HYPERMETRIC_STRETCH"]
    },
    "quality": {
      "gradient_class": "medium",
      "star_count_class": "high",
      "fwhm_class": "normal"
    }
  },
  "scope": {
    "applies_when": [
      "target has large diffuse emission",
      "background extraction may confuse nebulosity with background"
    ],
    "does_not_apply_when": [
      "compact galaxy target",
      "strong measured gradient dominates the field"
    ],
    "confidence": 0.68
  },
  "recommendation": {
    "action_plan_fragment": {
      "actions": [
        {"type": "config.set", "path": "bge.enabled", "value": false}
      ]
    },
    "explanation": "Testweise BGE deaktivieren oder konservativer konfigurieren und Stretch neu bewerten."
  },
  "evidence": {
    "run_refs": [
      {"run_id_hash": "sha256:...", "artifact_refs": ["report", "stacked_rgb_hms_preview"]}
    ],
    "human_feedback": "Nebel wurde nach BGE-Off sichtbarer.",
    "ai_observation": "Preview zeigte abgeschnittene/schwache Nebelstruktur."
  },
  "outcome": {
    "before": {"nebula_visibility": "weak", "warnings": ["faint_nebula"]},
    "after": {"nebula_visibility": "improved"},
    "delta": {"user_rating": 1},
    "verified": false
  },
  "review": {
    "reviewed_by": null,
    "reviewed_at": null,
    "notes": ""
  },
  "retrieval": {
    "keywords": ["nebula", "BGE", "extended emission", "OSC"],
    "embedding_text": "extended nebula OSC BGE removes faint emission",
    "negative": false
  }
}
```

### 9.3 Kontextsignatur fuer Tile Compile

Die Kontextsignatur wird aus vorhandenen Quellen normalisiert. Fehlende Werte
bleiben `null` oder `unknown`, werden aber nicht erfunden.

Quellen:

- FITS-Header: Objektname, Kamera, Filter, Belichtungszeit, Gain, Temperatur, Datum, Teleskop/Focal Length falls vorhanden.
- Scan-Ergebnis: Frame-Anzahl, Farbmodus, Bayer-Pattern, Dateigruppen, Fehler/Warnungen.
- Scan-Metriken: FWHM, Star Count, Hintergrund, Noise, Gradient, Roundness, Session Geometry.
- Config: relevante Pfade, aktivierte Pipeline-Phasen, BGE/PCC/HMS/AQMH/Stacking/Registration-Parameter.
- Run-Report: Phasenstatus, Artefakte, Warnungen, Qualitaetsmetriken.
- Nutzerangaben: Mount, Objektklasse, Kamera, Kalibrierung, Notizen.
- Run-Chat: natuerliche Problembeschreibung und Bildbeobachtungen.

Normalisierte Felder:

- `target.object_name`, `target.object_type`, `target.angular_size_class`, `target.has_extended_emission`
- `acquisition.camera_name`, `acquisition.camera_type`, `acquisition.color_mode`, `acquisition.filters`
- `acquisition.exposure_seconds_min/median/max`, `frame_count`, `total_integration_minutes`
- `acquisition.gain`, `sensor_temperature_c`, `date_range`
- `calibration.darks/flats/bias/dark_flats`, `calibration.quality_warnings`
- `optics.telescope`, `focal_length_mm`, `aperture_mm`, `f_ratio`, `reducer`, `pixel_scale_arcsec`
- `mount.type`, `tracking_quality`, `field_rotation_risk`
- `sky.moon_phase`, `moon_distance`, `bortle`, `transparency` falls spaeter verfuegbar
- `quality.fwhm_class`, `gradient_class`, `noise_class`, `star_count_class`
- `pipeline.phases`, `pipeline.affected_paths`, `pipeline.resume_phase`

### 9.4 Professioneller KI-Kontext

Scan-AI und Run-Chat duerfen nicht mehr nur "Empfehlungen plus ein paar
Memories" senden. Der Request muss professionell strukturiert sein:

- `task`: klare Aufgabe, z.B. `scan_config_optimization`, `run_quality_diagnosis`, `resume_strategy`.
- `current_context`: normalisierte Kontextsignatur.
- `current_evidence`: Scan-/Run-/Report-/Artefaktdaten, mit Bildpreview wenn das Modell Vision kann.
- `candidate_memories`: passende positive Memories mit Match-Erklaerung.
- `negative_memories`: rejected/deprecated/counterexamples mit Warnung.
- `constraints`: erlaubte Config-Pfade, Safety-Regeln, keine Pfade/Secrets, keine unvalidierten Writes.
- `required_output_schema`: z.B. `pi.scan-analysis.v2` oder `pi.run-chat-answer.v2`.
- `uncertainty_policy`: KI muss fehlende Daten als fehlend markieren und darf sie nicht raten.
- `memory_write_policy`: KI darf Memory-Kandidaten vorschlagen, aber nicht automatisch akzeptieren.

Die KI-Antwort muss professioneller werden:

- Empfehlungen brauchen `rationale`, `evidence_refs`, `expected_effect`, `risk`, `confidence`, `scope`.
- Parameter-Vorschlaege brauchen `path`, `current_value`, `suggested_value`, `why_now`, `why_safe`.
- Bei Bildfragen muss klar sein, ob ein Bild mitgesendet wurde und welche Beobachtung aus dem Bild stammt.
- Bei Folgefragen muss der bisherige Chat-/Run-Kontext enthalten sein.
- Wiederholte wirkungslose Vorschlaege duerfen nicht erneut angeboten werden; stattdessen muss die KI eine andere Hypothese oder ein Gegenbeispiel formulieren.

### 9.5 Memory-Erzeugung

Memory-Kandidaten entstehen aus mehreren Quellen:

- `learn=true` nach Apply: erzeugt nur `candidate`, noch keine akzeptierte Wahrheit.
- Run-Outcome-Evaluator nach Resume/Run: ergaenzt Outcome-Deltas.
- Run-Chat mit Nutzerfeedback: erzeugt `artifact_diagnosis` oder `resume_strategy`.
- Nutzer markiert "hat geholfen" oder "hat nicht geholfen": erzeugt positive oder negative Evidenz.
- Wiederholter Erfolg in aehnlichem Kontext: Kandidat wird `promotable`.

Speicherregeln:

- Kein Memory ohne Kontextsignatur.
- Kein accepted Memory ohne Review.
- Kein globales Memory ohne Scope und `does_not_apply_when`.
- Kein Memory mit Bilddaten; nur Preview-/Artefaktreferenzen und optional Hashes.
- Kein Memory mit absoluten lokalen Pfaden in Knowledge-Feldern.
- Jede Memory-Aenderung erzeugt Audit-Event.

### 9.6 Retrieval und Ranking

Ranking-Signale:

- Config-Pfad-Ueberschneidung.
- Objektklasse und Zielgroesse.
- Kamera-/Filter-/Farbmodus-Aehnlichkeit.
- Pipeline-Phase und Problemklasse.
- Qualitaetsmetriken und Artefaktklasse.
- Outcome-Qualitaet und Review-Status.
- Aktualitaet und Deprecation.
- Gegenbeispiele fuer aehnliche Kontexte.

Retrieval-Ergebnis:

- `matches`: akzeptierte Memories mit Score und Match-Feldern.
- `warnings`: negative/deprecated Memories mit Grund.
- `coverage`: welche Kontextfelder fehlen und daher Confidence senken.
- `prompt_budget`: Begrenzung fuer KI-Kontext.

### 9.7 GUI3-Anforderungen

- Memory-Detailansicht zeigt Kontextsignatur, Scope, Evidenz, Outcome, Review und Retrieval-Treffer.
- Beim Review kann der Nutzer Scope bearbeiten: "gilt fuer Nebel", "nicht fuer Galaxien", "nur OSC", "nur Dualband".
- Run-Chat zeigt, welche Memories die KI benutzt hat und warum.
- AI-Empfehlung zeigt positive und negative Memory-Hinweise getrennt.
- "Learn from this optimization" wird genauer: `Lernkandidat speichern`, danach Review/Outcome erforderlich.
- GUI3 erlaubt globale Memory-Suche nach Objekt, Kamera, Filter, Config-Pfad, Problemklasse und Status.

### 9.8 Umsetzungsschritte

- [ ] `pi.memory.v2` Schema als neuen Baseline-Vertrag definieren; keine Legacy-/Draft-Migration.
- [ ] Store-Reset-Verhalten definieren: neuer globaler Store, alte AI-/Memory-Dateien sichtbar ignorieren, keine automatische Uebernahme.
- [ ] `pi.context_signature.v1` Builder aus Scan, Run, FITS-Headern, Config, GUI-Kontext und Reports implementieren.
- [ ] Memory Store auf globale Indizes erweitern: `by_type`, `by_status`, `by_path`, `by_target`, `by_camera`, `by_filter`, `by_problem`.
- [ ] Retrieval-Service mit Scoring und Match-Erklaerung implementieren.
- [ ] Scan-AI Request auf professionellen Kontextcontainer `pi.ai-request.v2` umstellen.
- [ ] Run-Chat Request auf denselben Kontextcontainer umstellen, inklusive Bildstatus und Chat-Historie.
- [ ] Outcome-Evaluator fuer Run/Resume implementieren und Memory-Kandidaten mit Vorher/Nachher-Deltas aktualisieren.
- [ ] Negative Learning aus Nutzerfeedback und wirkungslosen Resume-Versuchen implementieren.
- [ ] GUI3 Memory-Detail-/Review-Ansicht erweitern.
- [ ] Export/Import fuer `pi.memory.v2` mit Privacy-Filter erweitern.
- [ ] Tests fuer globale Retrieval-Faelle, Scope-Grenzen, negative Memories und Privacy-Redaction anlegen.
- [ ] Tests pruefen, dass alte AI-/Memory-Daten nicht geladen, migriert oder als Retrieval-Kontext verwendet werden.

Abnahmekriterien:

- [ ] Ein neues globales Memory aus einem frueheren Run wird in einem anderen Projekt gefunden, wenn Objekt-/Aufnahme-/Pipeline-Kontext passt.
- [ ] Dasselbe Memory wird nicht oder nur mit niedriger Confidence gefunden, wenn der Kontext fachlich nicht passt.
- [ ] KI-Prompts enthalten explizit positive Memories, negative Memories, Match-Erklaerung und fehlende Kontextfelder.
- [ ] Memory-Kandidaten enthalten Objekt-/Aufnahmedaten, sofern vorhanden: Objekt, Kamera, Teleskop, Filter, Belichtung, Frame-Anzahl, Kalibrierung, Montierung, Qualitaetsmetriken.
- [ ] GUI3 erlaubt Review mit Scope-Anpassung.
- [ ] Kein Memory speichert Rohbilddaten, API-Keys oder absolute lokale Bildpfade.
- [ ] Tests sichern ab, dass accepted Memories weiterhin keine Config-Validierung umgehen.
- [ ] Ein leerer neuer PI-Storage startet deterministisch ohne Altlasten und ohne automatische Migration.

## Memory-Konzept im Detail

Ein Memory ist eine reviewbare, global nutzbare Erfahrung, keine automatische
Regel und keine projektspezifische Notiz.

Typischer Ablauf:

1. Scan-AI erzeugt Empfehlungen.
2. GUI3 zeigt Empfehlungen und PI Preview.
3. Nutzer wendet validierte Aenderungen an.
4. Wenn `learn=true` gesetzt ist, speichert Tile Compile einen Memory-Kandidaten mit Kontextsignatur.
5. Nach Run/Resume werden Outcome-Daten ergaenzt: Was wurde besser, gleich, schlechter oder blieb unklar?
6. Nutzer reviewed den Kandidaten als `accepted`, `rejected` oder `deprecated` und kann den Scope bearbeiten.
7. Spaetere Sessions duerfen passende accepted Memories als Kontext verwenden, muessen aber weiterhin Schema und Config validieren.
8. Rejected/deprecated Memories werden als Gegenbeispiele genutzt, damit die KI nicht dieselbe erfolglose Strategie wiederholt.

Beispiel fuer `config_optimization`:

- `type`: `config_optimization`
- `source`: `scan_ai_apply`
- `status`: `candidate`
- `privacy_class`: `metadata_only`
- `analysis_id`
- `provenance`: Analyse-/Run-/Artefaktreferenzen ohne absolute lokale Bildpfade
- `config_updates`
- `context_signature`: Objekt, Zieltyp, Kamera, Teleskop/Optik, Filter, Belichtungen, Frame-Anzahl, Kalibrierung, Montierung, Qualitaetsklassen, relevante Pipeline-Phasen
- `scope`: Wann diese Erfahrung gilt und wann nicht
- `summary`
- `confidence`
- `detected_scenarios`
- `warnings`
- `validation`
- `outcome`: Vorher/Nachher-Deltas, Nutzerfeedback, Report-/Artefaktstatus
- `review`: Status, Reviewer, Notizen, Scope-Aenderungen

Memory-Qualitaetsregeln:

- Ein Memory ohne `context_signature` darf nicht `accepted` werden.
- Ein Memory ohne Outcome oder Nutzerreview bleibt `candidate`.
- Ein Memory mit nur einem Einzelfall bekommt niedrige Confidence.
- Ein Memory mit widerspruechlichem spaeterem Ergebnis wird `deprecated` oder erhaelt ein `counterexample`.
- Memories duerfen global wirken, aber nur innerhalb ihres fachlich beschriebenen Scopes.
- Die KI bekommt Memories nie als Befehl, sondern als historische Evidenz mit Match-Score und Unsicherheit.

## Professionalisierung der KI-Funktion

Die KI-Funktion soll nicht mehr wie ein einfacher Empfehlungsdialog arbeiten,
sondern wie ein strukturierter Diagnose- und Optimierungsassistent.

Pflichtanforderungen fuer KI-Requests:

- Vollstaendiger Kontextcontainer statt losem Prompt.
- Explizite Aufgabe: Analyse, Diagnose, Resume-Plan, Memory-Bewertung oder Outcome-Bewertung.
- Normalisierte Kontextsignatur.
- Relevante positive und negative Memories mit Match-Begruendung.
- Aktuelle Config und erlaubte Config-Pfade.
- Report-/Artefakt-/Qualitaetsdaten mit Quellenreferenzen.
- Bildpreview nur wenn Provider/Modell Vision unterstuetzt; sonst klarer Hinweis `image_context=false`.
- Striktes Antwortschema mit Empfehlungen, Evidenz, Risiken, Confidence und optionalem Action-Plan.

Pflichtanforderungen fuer KI-Antworten:

- Keine Behauptung ohne Evidenzreferenz oder Unsicherheitsmarkierung.
- Keine Wiederholung identischer Parameterempfehlungen, wenn sie im selben Run bereits ohne Verbesserung getestet wurden.
- Jede Parameterempfehlung erklaert erwartete Wirkung, Risiko, betroffene Phase und kleinste sinnvolle Resume-Phase.
- Bei Memory-Vorschlaegen muss die KI Scope, Evidence und moegliche Gegenbeispiele mitliefern.
- Bei Bildproblemen muss die KI trennen zwischen "im Bild beobachtet", "aus Report abgeleitet" und "nur Nutzerbeschreibung".

## Naechster sinnvoller Schritt

Als naechstes sollte Phase 9 begonnen werden: `pi.memory.v2`,
`pi.context_signature.v1` und der professionelle AI-Request-Container muessen
definiert und implementiert werden. Danach koennen Phase 5 Outcome-Metriken und
Phase 8 Run-Chat auf denselben globalen Memory-Layer schreiben und daraus
retrieven.
