# PI Pre-Rules — Evidenz, Zulässigkeit und Config-Kandidaten

> **Stand:** 2026-09-20, korrigierter Regelkatalog.
> **Status:** Planung; Regeln und neue Verträge sind nicht implementiert.
> **Zweck:** Input-Statistiken für sichere, prüfbare Pre-Run-Vorschläge nutzen.

Verbindliches Zielbild: [Jev Decisions](pi_jev_decisions_plan_de.md). Umsetzung: [Implementierungsplan](pi_jev_implementierungsplan_de.md).

## 1. Grundprinzip

Pre-Rules liefern Befunde, Zulässigkeitsentscheidungen und atomare Config-Kandidaten. Sie schreiben keine Config und starten keinen Run. Eine deterministisch ausgeführte Heuristik bleibt eine Heuristik.

Die frühere Kennzeichnung `[B]` entfällt: Schema-Text, Prompt-Empfehlung, implementierter Mechanismus und nachgewiesene Qualitätsverbesserung sind nicht dasselbe.

Jede Regel enthält getrennt:

| Feld | Bedeutung |
|---|---|
| `source_kind` | `code_contract`, `schema_description`, `prompt_policy`, `hypothesis`, `evaluation` |
| `source_ref` | Konkrete Datei/Symbol oder Evaluations-ID mit Version |
| `mechanism` | Warum die Eingabe für die Entscheidung relevant sein könnte |
| `validation_status` | `unvalidated`, `contract_tested`, `empirically_validated` |
| `effect` | `block`, `warn`, `candidate`, `preserve` |
| `missing_input_action` | `block`, `warn`, `abstain`; nie fehlenden Wert als 0 einsetzen |
| `requires_review` | Für alle neuen Tuning-Kandidaten zunächst wahr |

Numerische Regel-Confidence wie `0.8` entfällt. Modellwahrscheinlichkeit, Evidenzstatus und empirischer Nutzen werden separat geführt.

## 2. Eingangsdaten und ihre Grenzen

`scan` und `scan-metrics` stammen aus `tile_compile_cpp/apps/cli_main.cpp`; der Aufruf wird über die vorhandene Scan-Infrastruktur übernommen. Kein neuer Runner-Befehl wird aus einem Dokumentnamen abgeleitet.

- Frame-Metriken: Hintergrund, Rauschen, lokale Strukturenergie, Quadrantenkontrast (`sky_gradient`), FWHM-Achsen, Roundness, detektierte Sterne und Lesestatus.
- Aggregate enthalten mehr Felder als der heutige `pi.feature-vector.v1`; der neue Entscheidungs-State projiziert die Quellen ausdrücklich.
- `roundness = fwhm_y / fwhm_x`. Ein Wert über 1 ist möglich. Aus gültigen Einzelwerten kann eine symmetrische Achsenabweichung `abs(log(roundness))` gebildet und danach aggregiert werden. Nicht erst den Median transformieren: gegenläufige Abweichungen können sich zuvor verdecken. Dies bleibt eine achsenbezogene Formbeschreibung, keine rotationsinvariante PSF-Messung.
- `sky_gradient` beschreibt Quadrantenunterschied relativ zum Hintergrund; Objektstruktur kann beitragen. Kein alleiniger Nachweis eines entfernbaren Hintergrundgradienten.
- Scan-Sternzahlen hängen von Detektor, Grenzwerten und Stichprobe ab. Keine Gleichsetzung mit registrierbaren, kataloggematchten oder im Stack messbaren Sternen.
- Aufnahmegruppen und Header-Fakten nur verwenden, wenn Quelle, Einheit und Gültigkeit bekannt sind. Kamera-Teilstrings sind keine eindeutige Sensoridentifikation.
- Theoretische Session-Rotation nicht als Messung oder sichere Obergrenze verwenden. Montierung, Meridiandurchgang/Flip und tatsächliche Offsets bleiben eigene Informationen.
- Registrierungs-/Dither-Abdeckung, Stack-Rauschen und PCC-Erfolg fehlen im aktuellen Pre-Run-Scan.
- Kalibrationsdateien und Kataloge sind Systemfakten und notwendige Voraussetzungen, auch wenn ihre Pfade keine Tuning-Ziele sind.

`frames_detected`, untersuchte Frames, gültige Messungen und später tatsächlich beitragende Frames sind unterschiedliche Größen. Für Quotienten müssen Nenner endlich und positiv sein; bei unzureichender Messabdeckung Enthaltung. Roh-/kalibrierte und unterschiedlich skalierte Messungen nicht unmarkiert vergleichen.

## 3. Harte Policy vor Kandidatenbildung

| ID | Bedingung | Wirkung |
|---|---|---|
| `guard_identity` | Farbmodus/Bayer uneindeutig oder Scan blockiert | Kein anwendbarer Patch; nötige Bestätigung im vorhandenen Workflow |
| `guard_versions` | Dataset-/Scan-/Config-/Lock-Revision nicht aktuell | Vorschlag `stale`; neu bewerten |
| `guard_locks` | Kandidat verändert gesperrten Wert | Gesamten Kandidaten ausschließen |
| `guard_evidence` | Pflichtinformation fehlt/ist ungültig | Kandidaten ausschließen; sichtbarer Diagnosegrund |
| `guard_schema` | Pfad, Typ, Enum, Bereich oder Gesamtconfig ungültig | Gesamten Kandidaten ablehnen |
| `guard_atomic` | Abhängige Werte fehlen oder widersprechen sich | Kein Teilpatch |
| `guard_quality` | Patch verändert Schutz-/Akzeptanzgrenzen oder Masken-/Provenienzregeln | Ablehnen |
| `guard_scope` | Patch betrifft nicht freigegebene Parametergruppe | Ablehnen |

Geschützt sind insbesondere `reconstruction.coverage_gate.*`, `reconstruction.multiband_validation.*`, PCC-Grenzen wie `k_max`, `max_residual_rms`, `max_condition_number` sowie Schutzschalter. Exakte Pfade werden in M0 gegen Schema und Implementierung inventarisiert; unbekannte Pfade sind nicht freigegeben.

Schema-Descriptions wie frameabhängige Grenzen für `weight_exponent_scale` sind bis zur expliziten Implementierung keine existierenden Validator-Garantien. Im ersten Lieferumfang bleiben Exponent und Gewichtsvektoren unverändert. Eine spätere Freigabe benötigt vollständige Gewichtsdimensionen, numerisch tolerante Summenprüfung und kontextabhängige Tests.

## 4. Korrigierter Katalog nach Parameterbereich

### 4.1 Identifikation und Kalibration

- `data.color_mode`/`bayer_pattern`: bestätigte Scan-Fakten im bestehenden Identifikationsworkflow übernehmen; Jev entscheidet nicht über widersprüchliche Header.
- Gemischte Belichtungszeiten, Gain, Filter oder Temperaturen: Gruppen und Matching-Verfügbarkeit prüfen. Keine Dark-Toleranz proportional zur Streuung aufweiten.
- Ein passender Master pro Gruppe ist ein Datenbefund, kein Modellurteil. Fehlende Kalibration erzeugt Warnung/Blockierung gemäß bestehendem Vertrag.
- Die frühere pauschale Aussage „Darks dürfen physikalisch nie skaliert werden“ entfällt. Zulässig ist ausschließlich das vom implementierten Kalibrationsverfahren unterstützte Matching/Scaling; keine neue Scaling-Funktion durch Empfehlungen erfinden.
- `use_*`, Master-Pfade und Matching-Toleranzen bleiben in Version 1 außerhalb der Patch-Allowlist.

### 4.2 Normalisierung

Breite Hintergrundverteilung ist ein Diagnosehinweis, kein Beweis für den passenden Modus. Ein füllendes Objekt erfordert belastbaren Kontext zu Füllgrad und Bildmaßstab; „10 Prozent“ wird nicht als validierte Grenze übernommen.

Version 1 belässt die aktuelle Normalisierung. Ein späterer Kandidat benötigt dokumentierte Signal-Erhaltungstests; fehlender Objektkontext führt zur Enthaltung, nicht zu bloßem Review eines unbegründeten Patches.

### 4.3 Registrierung

- Wenige Scan-Sterne: Kandidatenarmut melden. Ein größeres `star_topk` erzeugt nicht automatisch nutzbare Sterne; der vorhandene Detektor/Fallback muss separat berücksichtigt werden.
- Achsenungleichheit: keine automatische Änderung von `star_inlier_tol_px` oder `transform_model`. Intraframe-Unschärfe und Interframe-Transformation getrennt betrachten.
- Keine Rotationssperre allein aus `mount_type=EQ` oder theoretisch kleiner Rotation.
- Kein Wechsel zu `robust_phase_ecc` allein aus hoher Strukturenergie und wenigen Sternen.
- ASTAP-Verfügbarkeit alleine garantiert keinen erfolgreichen Solve.

Version 1 liefert hier Hinweise und fehlende Voraussetzungen; Toleranzen, Engine, Transformmodell und Reject-Parameter bleiben erhalten.

### 4.4 Globale Qualitätsgewichtung

Spread-Verhältnisse wie 1.3 oder 1.5 sind Hypothesen, keine harten Naturgrenzen. Höhere Gewichtung guter Frames und schärferes Pixel-Clipping sind unterschiedliche Mechanismen.

Erster begrenzter Kandidat: `enable_adaptive_weights`, siehe Abschnitt 5. Gewichtsvektoren, Exponent und Clipping werden dabei nicht verändert. Die Aussage „unbalancierte Gewichte verursachen generell Farbstich“ wird nicht als universeller Kausalnachweis übernommen.

### 4.5 Drizzle und Clipping

- Die früheren Schwellen FWHM <= 2.5/ >= 3.8 und 10/16/20/50 Frames werden nicht als automatische Regeln übernommen.
- FWHM und Frame-Anzahl liefern Hinweise auf Sampling und Datenmenge, keine belegte Subpixel-Abdeckung. Insbesondere CFA-Abdeckung kann nicht aus einer einzigen FWHM-Zahl abgeleitet werden.
- `internal_scale`, `output_scale` und `pixfrac` wären eine abhängige Gruppe. Änderungen brauchen eigene Vergleichsevidenz; Version 1 verändert sie nicht.
- `min_clip_contributors <= frames_total` ist keine allgemeine Qualitätsinvariante: Gesamtzahl und lokale Beiträge sind verschieden; ein Mindestwert kann Clipping bewusst deaktivieren. Bestehende Mindestwerte nicht zur Erfüllung dieser früheren Regel absenken.
- `full_frame_estimator` impliziert nicht durch Beschreibungstext einen automatisch erzwungenen Sigmawert. Tatsächlichen Codevertrag prüfen.
- Breiter Frame-Qualitätsspread rechtfertigt keine pauschal engeren Clipping-Sigmas.

### 4.6 Coverage und Multiband

Alle Akzeptanzgrenzen bleiben unverändert. Die frühere Empfehlung, `background_rms_ratio_max` bei füllendem Objekt anzuheben, ist gestrichen.

Geringe Scan-Sternzahl begründet nur Unsicherheit über spätere Validierung. `not_applicable` wird ausschließlich aus der tatsächlichen Post-Run-Messung abgeleitet. Keine Vorhersage einer „evidenzlosen Promotion“ aus Scan-Sternzahlen.

### 4.7 PCC und Astrometrie

Farbmodus, vorhandene Kataloge und tatsächlicher Pipelinevertrag bestimmen Anwendbarkeit. OSC mit mindestens zehn Scan-Sternen ist kein ausreichender Aktivierungsnachweis. Auch RGB muss als eigener Fall behandelt werden.

Apertur-Automatik und vorhandene Schutzgrenzen erhalten. Keine Änderung von Katalogtiefe, PCC-Grenzen oder Hintergrundmodell allein aus Belichtungszeit, Kamerafamilie oder Quadrantenkontrast. Fehlende Voraussetzungen anzeigen; Farbprobleme später anhand von PCC-Artefakten diagnostizieren.

### 4.8 BGE

Die Schwellen 0.02/0.05 bleiben allenfalls gekennzeichnete historische Hypothesen. Weder BGE abschalten noch `classic`, `poly`, `modeled_mask_mesh` oder Autotune-Gewichte allein danach setzen.

Ein BGE-Kandidat benötigt mindestens Objekt-/Hintergrundkontext und ein dokumentiertes Verfahren zur Signal-Erhaltung. Globale Aggregate reichen dafür im ersten Lieferumfang nicht. Maskendilatation hängt nicht allein von Sternzahl ab; räumliche Größen benötigen ein definiertes Pixelraster.

Kein pauschales Ranking „poly immer besser als RBF“. Differenzen zwischen YAML-Profil und Schema-Default werden als Versions-/Profilfrage erfasst, nicht im Rahmen der Jev-Integration still korrigiert.

### 4.9 Stretch, Sensorprofil, Kosmetik und Denoise

- Kein Kamera-Substring-Mapping wie `DWARF -> IMX415`. Ein späteres Mapping braucht eindeutiges Modell, versionierte Tabelle und einen Unknown-Fall.
- Stretch-Ziele und Stärke benötigen den richtigen Stack-Zwischenstand. Nutzerpräferenzen bleiben erhalten.
- Kosmetische Korrektur nicht allein wegen einer Consumer-Kamerafamilie aktivieren; Defekt-/Kalibrationsbefund erforderlich.
- Einzelbild-Rauschen entscheidet nicht über Chroma-/Luma-Dominanz im Stack. Denoise-Änderungen werden auf Post-Run vertagt.
- Masken und Extended-Source-Schutz nicht aufgrund einer ungesicherten Objektklassifikation abschalten.

## 5. Konkreter erster Kandidat und Erweiterungsverfahren

Die erste vollständige Integration bleibt absichtlich begrenzt, erzeugt aber echte Config-Vorschläge:

| ID | Voraussetzungen | Atomarer Patch | Status |
|---|---|---|---|
| `keep_current` | Ausgangsconfig gültig | leer | Immer verfügbar |
| `insufficient_evidence` | Immer auswählbar | leer | Enthaltung, kein Erfolgsscore |
| `enable_adaptive_weights` | Aktuell false; Pfad nicht gesperrt; gültige vergleichbare Qualitätsmetriken innerhalb kompatibler Gruppen; Mindestabdeckung aus versionierter Policy erfüllt | `global_metrics.adaptive_weights: true` | Hypothese; zuerst Shadow, danach nur gemäß M5-Freigabe |

Der State enthält numerisch berechnete Streuungen, Messabdeckung und Einschränkungen. Ob diese eine Umstellung rechtfertigen, wird im Vergleich gegen die Ausgangsconfig evaluiert. Ohne freigegebene Mindestabdeckung/Policy bleibt der Kandidat für anwendbare Vorschläge deaktiviert. Ist adaptive Gewichtung bereits aktiv, wird kein Scheineffekt als Empfehlung ausgegeben.

Weitere Parametergruppen werden einzeln ergänzt: Quelle/Mechanismus -> Preconditions -> exakter Patch -> Abhängigkeiten -> Negativfixtures -> gepaarte Evaluation -> Freigabe. Ein Schema-Enum allein erzeugt keinen Kandidaten. Die übrigen Bereiche dieses Katalogs bleiben Diagnoseumfang oder ausdrücklich spätere Erweiterungen; keine Vollabdeckung aller Config-Parameter behaupten.

## 6. Kandidatenvertrag

Geplanter Anwendungsvertrag, kein Provider-Request:

```json
{
  "candidate_id": "enable_adaptive_weights",
  "candidate_version": 1,
  "group": "global_weighting",
  "validation_status": "unvalidated",
  "requires_review": true,
  "required_evidence": ["quality_spread", "measurement_coverage"],
  "preconditions": ["adaptive_weights_is_false", "path_unlocked"],
  "updates": [{"path": "global_metrics.adaptive_weights", "value": true}],
  "expected_effect_code": "redistribute_frame_weights",
  "risk_code": "quality_gain_unproven"
}
```

Kandidaten führen strukturierte Evidenzreferenzen statt nur freier Erklärungstexte. Vor Anzeige und Übernahme löst das Backend diese Referenzen gegen denselben State auf. Der vorhandene `validated_updates`-Adapter erhält zusätzliche Kandidaten-/Gruppenmetadaten; der Validator kann nicht unverändert bleiben.

## 7. Ausschluss und Konflikte

Version 1 arbeitet mit exakter Allowlist, nicht mit „alles in der Pixeldomäne darf verändert werden“. Diese frühere Faustregel würde auch Schutzgates und unbewiesene Solveränderungen freigeben.

Nicht patchbar: Qualitätsgates, Kalibrationspfade, System-/Runtimewerte, Diagnose-/Cache-Schalter, Nutzerpräferenzen, manuelle Masken, unbekannte Pfade sowie sämtliche nicht explizit freigegebenen Parameter.

Pro Kandidat wird die gesamte gemergte Config geprüft. Konflikte mit Locks oder Voraussetzungen verwerfen den Kandidaten. Es gewinnt weder die zuletzt ausgeführte Regel noch die mit den meisten Inputs. Jev darf sich enthalten; kNN/LLM bleiben unabhängige Vergleichsausgaben.

## 8. Abnahme

Pflichtfälle: fehlende/ungültige Metriken, Nullnenner, gemischte Aufnahmegruppen, Roundness 0.5/1/2, unbekannte Montierung, keine Ditherdaten, geringe Scan-Sternzahl bei unbekannter Stack-Sternzahl, Nutzer-Locks, veraltete Config, geschützte Gates und atomare Ablehnung.

Die vollständige Test- und Lieferreihenfolge steht im [Implementierungsplan](pi_jev_implementierungsplan_de.md). Keine Regel dieses Dokuments gilt allein durch ihre Aufnahme als empirisch validiert.
