# PI Jev Decisions — Zielbild und verbindliche Entscheidungen

> **Stand:** 2026-09-20, nach Code-Review überarbeitet.
> **Status:** Planung. Die hier beschriebene Jev-Integration ist nicht implementiert.
> **Priorität:** zuerst Config-Vorschläge vor dem Run, danach gezielte Empfehlungen nach dem ersten Run.

Verknüpfte Dokumente:

- [Pre-Rules und Kandidatenkatalog](pi_scan_pre_rules_de.md)
- [Detaillierter Implementierungsplan](pi_jev_implementierungsplan_de.md)
- [Bestehender lokaler Lernplan](pi_local_learning_plan_de.md)

## 1. Produktziel und Zuständigkeiten

Jev soll aus Statistiken der Input-Frames, bekannter Aufnahmegeometrie, Kalibrationsverfügbarkeit und der aktuellen Config einen begründeten Config-Vorschlag ableiten. Das Ergebnis ist ein prüfbarer Unterschied zur Ausgangsconfig, kein automatisch gestarteter Run und kein Versprechen einer optimalen Rekonstruktion.

Nach dem ersten Run soll Jev beurteilen, ob eine Änderung überhaupt begründet ist, welche Phase betroffen ist und welche begrenzte Alternative geprüft werden sollte. Ein akzeptables Ergebnis ohne Änderungsbedarf ist ein vollwertiger Erfolg.

| Instanz | Zuständigkeit | Grenze |
|---|---|---|
| Deterministischer Backend-Code | Messwerte aufbereiten, Voraussetzungen prüfen, Kandidaten bilden, Gesamtconfig validieren | Schwellenheuristiken nicht als physikalischen Beweis ausgeben |
| Jev | Zulässige Kandidaten unter dokumentierter Unsicherheit bewerten/auswählen | Keine freien Pfade/Werte, keine Gate-Lockerung, keine Ausführung |
| PI | Rückfragen, Erläuterungen und vorhandene interaktive Bildnachbearbeitung | Eine Erklärung darf den validierten Patch nicht still ändern |
| Lokales kNN | Vergleich im bestehenden Shadow-Pfad | `available=true` reicht nicht für Vorrang oder produktive Freigabe |
| Runner | Bestehende Offline-Pipeline und ihre Schutzmechanismen | Keine Jev-Abhängigkeit oder externen Modellcalls |

`live_edit_op_intent`, einzelne Enum-Cold-Start-Calls und eine modellbasierte Frage „Ist der Patch sicher?“ sind nicht Teil des ersten Produkts. Letztere ersetzt keine Validierung und wäre keine unabhängige Sicherheitsinstanz.

## 2. Verifizierter Ausgangsstand und Korrekturen

Codebasis dieses Reviews: Arbeitsbaum vom 2026-09-20; vorhandene fremde Änderungen sind keine durch diesen Plan abgenommenen Implementierungen.

| Befund | Konsequenz |
|---|---|
| `build_scan_feature_vector()` enthält ausgewählte Aggregate, Frame-Zahlen, Farbmodus und Bayer-Muster; keine vollständige Session-Geometrie, Header, Warnungen oder Registrierung | Eigener versionierter Entscheidungs-State; bestehenden kNN-Vertrag nicht still verändern |
| Scan-`roundness` ist `fwhm_y / fwhm_x` und kann größer als 1 sein | `>=0.9` ist kein Rundheitskriterium; Achsenungleichheit symmetrisch beschreiben, keine Driftursache behaupten |
| Scan-Sternmessung und Registrierungsdetektor sind unterschiedliche Verfahren | Scan-Sternzahl ist weder RANSAC-Inlierzahl noch spätere PCC-/Validierungssternzahl |
| Schema-Beschreibungen enthalten Empfehlungen, die nicht sämtlich ausführbare Validator-Regeln sind | Harte Bedingungen ausdrücklich implementieren und testen; Text nicht als Enforcement deklarieren |
| Recommendation-Validator besitzt nur bestimmte atomare Gewichtsgruppen und kann sonst Teilpatches rekonstruieren | Jev-Kandidaten ganz oder gar nicht validieren |
| kNN-Distanz verwendet unskalierte numerische Differenzen | Kein produktiver Vorrang; separate Feature-Skalierung, Datenabdeckung und Evaluation erforderlich |
| PI Live Image Chat liefert bereits konkrete Operationen mit Bildkontext und Operationshistorie | Nachbearbeitung bleibt bei PI; funktionale Existenz ist keine vollständige Qualitätsabnahme |

Relevante Quellen: `web_backend_cpp/src/services/pi/pi_feature_vector.cpp`, `pi_recommendation_validator.cpp`, `pi_param_model.cpp`, `tile_compile_cpp/apps/cli_main.cpp`, `tile_compile_cpp/src/metrics/metrics.cpp`, `agent_service/src/services/liveImageChatService.ts`.

Die frühere Jev-Operationsliste war unvollständig und enthielt `curves`, obwohl der PI-Prompt Kurven ausschließlich der GUI zuordnet. Diese Liste entfällt.

## 3. Jev-Vertrag und Aussagegrenzen

Jev ist ein Modell für strukturierte Entscheidungen; der Adapter verwendet die Decisions API statt Chat Completions. Als Integrationsreferenzen dienen die [OpenRouter-Beispiele](https://openrouter.ai/labs/jev/compile), die [Modellseite](https://openrouter.ai/typesafe/jev-1.13/) und die [TypeSafe-Einordnung](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

Die exakte Request-/Response-Form, Modellkennung, Limits und Fehlerantworten werden in M0/M3 des Implementierungsplans gegen den dann aktuellen Anbieter-Vertrag geprüft und als Fixtures eingefroren. Keine Annahme über garantierte Antwortlatenz, State-Größe oder eine Patchversion im Response.

- Anwendung kennt `choice`, optionale unabhängige `noul`-Fragen und Enthaltung.
- Auswahloptionen enthalten immer `keep_current` und `insufficient_evidence`.
- Bekannte Messwertvergleiche und Arithmetik berechnet das Backend.
- Szenario-Wahrscheinlichkeit ist nicht die Wahrscheinlichkeit einer Ergebnisverbesserung.
- Anbieter-Confidence gilt ohne lokale Evaluation nicht als astrophotografisch kalibriert.
- Fehlende Informationen werden nicht durch Modell-Confidence ersetzt.
- Modellkennung, Fragen-, Kandidaten-, State- und Policy-Version werden protokolliert; ein Alias ist keine reproduzierbare Versionsbindung.

## 4. Pre-Run: Datenvertrag vor Modellaufruf

Neuer Vertrag `pi.decision-state.v1`, getrennt von `pi.feature-vector.v1`.

| Bereich | Mindestinhalt |
|---|---|
| Identität | Dataset-Fingerprint, Scan-ID/Version, Config-Hash, Software-/Schema-Version |
| Messherkunft | Verfahren/Version, Pixelraster, Einheiten, Roh-/Kalibrationszustand, Samplingstrategie |
| Abdeckung | erkannt, untersucht, erfolgreich gelesen, je Metrik gültig/fehlend; keine Gleichsetzung mit später akzeptierten Frames |
| Statistik | Aggregate und robuste Streuung je Aufnahmegruppe; definierter Status für null/ungültig/nicht anwendbar |
| Gruppen | Kamera, Dimensionen, Farbmodus/Bayer, Filter, Belichtung, Gain/Binning soweit vorhanden; unbekannt explizit |
| Kontext | Montierung, Objektfüllgrad/Bildmaßstab nur mit Quelle und Gültigkeit; Nutzerangabe von Messung unterscheiden |
| Ressourcen | Kalibrationsmatching und Katalog-/Tool-Verfügbarkeit als lokale Fakten; keine externen Dateipfade im Provider-State |
| Ausgangslage | Effektive Config, gesperrte Nutzerwerte, gewünschte Ausgabe, Warnungen und blockierende Scanfehler |

Jeder verwendete Messwert hat `status`, `unit`, `source_ref`, `method_version` und Stichprobenumfang. Fehlend ist nicht null Rauschen, null Sterne oder null Gradient. Quotienten benötigen gültige Nenner; Gruppen mit unterschiedlichen Messbedingungen werden nicht blind zusammengefasst.

Die vorhandene Rotationsschätzung ist als Schätzung zu kennzeichnen, nicht als gemessene Rotation oder garantierte Obergrenze. Dither-Abdeckung und Registrationserfolg sind im aktuellen Einzelbild-Scan unbekannt. Eine zusätzliche Vorregistrierung wäre eine separat zu planende Erweiterung, keine still eingeführte Voraussetzung.

## 5. Kandidaten statt unabhängiger Parameterwetten

Ablauf:

1. Eingaben, Config und Nutzer-Locks prüfen. Blockierende Identifikationsprobleme erzeugen keinen anwendbaren Patch.
2. Pre-Rules erzeugen Befunde, Ausschlüsse und kleine kohärente Kandidaten.
3. Jev erhält nur zulässige Kandidaten mit Voraussetzungen, Messreferenzen und erwarteten Effekten.
4. Backend löst die ausgewählte ID auf, prüft Policy und Gesamtconfig erneut und baut den Vorschlag.
5. UI zeigt Vorher/Nachher, Evidenz, Unsicherheit und Auswirkungen. Übernahme verändert nur den Config-Entwurf.

Ein Kandidat ist eine atomare Gruppe. Für Version 1 verändert er höchstens eine fachliche Parametergruppe; abhängige Werte gehören in dieselbe Gruppe. Keine kombinatorische Mischung von BGE-, Registrierungs-, Drizzle- und Denoise-Presets. Mehrere sinnvolle Alternativen werden getrennt angeboten, nicht automatisch vereinigt.

Priorität der Entscheidung:

1. Harte Gültigkeit, unveränderbare Qualitätsgates und verfügbare Evidenz.
2. Nutzer-Locks und aktuelle Config. Bei Konflikt mit Gültigkeit: blockieren, nicht überschreiben.
3. Explizite Kandidaten-Voraussetzungen und Abhängigkeiten.
4. Evaluierte Auswahlpolicy; unkalibrierte Vorschläge bleiben als experimentell gekennzeichnete Review-Vorschläge.
5. Bei Konflikt, unbekanntem Fall oder fehlender Evidenz: Enthaltung.

Weder „mehr Input-Felder“ noch „deterministisch ausgeführt“ begründet Vorrang. kNN und LLM dürfen Ergebnisse für Evaluation liefern, aber keinen Jev-Patch im Hintergrund verändern.

## 6. Schutzvertrag und Übernahme

- Keine Änderungen an Coverage-/Multiband-Akzeptanzgrenzen, PCC-Schutzgrenzen, Masken-/Provenienzpflichten oder Fail-closed-Verhalten.
- Keine Top-Level-Methodenauswahl; die Rekonstruktion bleibt CFA Forward Drizzle + Multiband. `bge.method` ist ein eigener, davon unabhängiger Parameter.
- Keine Änderungen an Nutzerpräferenzen, Pfaden, CPU/GPU-Wahl oder Laufzeitlimits durch Jev.
- Vorschläge sind an Dataset, Scan, Config, Locks und Policy gebunden. Veraltete Antworten werden verworfen.
- Statusfolge: `draft -> validated -> presented -> applied_to_draft | rejected | stale`; Blockierung/Enthaltung liefern keinen Patch.
- Übernahme erfordert aktuelle Hashes und erneute Gesamtvalidierung; wiederholte Übernahme ist idempotent.
- Beratung oder Übernahme startet weder Run noch Resume. Bestehende Ausführungsaktionen bleiben getrennt.
- Provider-Ausfall ergibt unveränderte Ausgangsconfig plus sichtbaren Beratungsstatus. Bestehende PI-Beratung bleibt unabhängig nutzbar.

## 7. Post-Run: Diagnose vor Änderung

Zweiter Produktabschnitt, nach funktionierender Pre-Run-Beratung. Eingaben sind effektive Config, ausgeführte/übersprungene Phasen, Artefaktversionen, tatsächlich ausgewählter Rekonstruktionskandidat und gültige Zwischenstandsmetriken.

| Ergebnis | Bedeutung |
|---|---|
| `no_change` | Kein hinreichend belegter Änderungsbedarf |
| `diagnose` | Fehlende Messung oder mehrdeutige Ursache; kein Patch |
| `suggest_downstream` | Ursache eingegrenzt, gezielter Patch und geprüfte Resume-Machbarkeit |
| `suggest_reconstruction` | Frühere Verarbeitung betroffen; expliziter Vorschlag für erneute Verarbeitung |

Restgradient führt nicht automatisch zu anderem BGE-Fit; Grünüberschuss nicht zu höherem `k_max`; weiches Endbild nicht automatisch zu weniger Denoise. Benötigt wird eine Zuordnung zu Zwischenständen. Raw, Uniform, Multiband und ausgewähltes Ergebnis bleiben getrennt; Gate-Ablehnungen können durch kein Jev-Qualitätsscore überstimmt werden.

Messungen müssen Bildzustand, lineare/gestreckte Domäne, Skalierung und Canvas-Maske benennen. Sternvergleiche verwenden gematchte Positionen. Nicht anwendbare Metriken sind kein Erfolg und keine Nullabweichung.

Resume-Phase und Vorgängerartefakte werden durch bestehende Backend-/Runner-Verträge geprüft. Fehlende Caches oder Masken verhindern die Resume-Empfehlung. Bestehende Runs bleiben bei der Beratung unverändert.

Stop-Regeln: eine fachliche Änderungsgruppe pro Vergleich; kein identischer bereits verworfener Patch; keine weitere Iteration ohne neues Ergebnis oder neue Evidenz. Keine automatische Optimierungsschleife. Bisheriges Ergebnis bleibt als Vergleichsbasis erhalten.

## 8. PI-Nachbearbeitung

PI behält Live-Chat, Bildkontext, konkrete Bildoperationen und deren bestehende Validierung. Jev darf später einen strukturierten Befund mit Messreferenzen an PI übergeben, beispielsweise zur Prüfung von Farbrauschen. Daraus folgt keine automatische Pixeloperation und keine automatische Änderung künftiger Run-Configs.

Ein zusätzlicher Jev-Intent-Router wird nur separat erwogen, wenn Messungen einen Nutzen gegenüber dem bestehenden PI-Pfad zeigen. Er ist weder Voraussetzung noch Lieferumfang dieses Plans.

## 9. Evaluation, Betrieb und Freigabe

Betriebsmodi: `off`, `shadow`, `suggest`. Standard ist `off`; kein `auto_apply`. `suggest` zeigt regulär nur freigegebene Kandidaten. Experimentelle Kandidaten benötigen zusätzlich die ausdrücklich aktivierte Einstellung `allow_experimental_suggestions` (Default false); harte Evidenz- und Schutzregeln gelten unverändert.

Shadow-Aufzeichnung trennt Modellantwort, angezeigten Vorschlag, Nutzerübernahme und tatsächlich angewendete effektive Config. Nutzerzustimmung und LLM-Agreement sind keine Qualitätslabels. Qualitätsbelege brauchen vergleichbare Ergebnisse auf denselben Frames; Training/Kalibrierung und Test werden nach Aufnahmesession getrennt.

Verglichen werden Ausgangsconfig, Regeln allein, bestehende PI-Beratung und Regeln plus Jev. Ausgewiesen werden ungültige Vorschläge, Enthaltung, Abdeckung, Regressionen pro Qualitätsmetrik, Kalibrierung sowie reale Request-Kosten und Latenz. Eine neue Modell-/Katalogversion braucht erneute Freigabe; fehlender belegter Nutzen verhindert die reguläre Freigabe. Explizite experimentelle Vorschläge bleiben gesondert gekennzeichnet und standardmäßig deaktiviert.

Der externe State ist eine Allowlist-Projektion: keine Schlüssel, absoluten Pfade, vollständigen FITS-Header oder unnötigen Freitexte. Timeout, Antwortgrößenlimit, begrenzte Retries und Deduplizierung sind Pflicht. Provider-JSON ist untrusted Input; nur bekannte Kandidaten-IDs und endliche Wahrscheinlichkeiten werden akzeptiert.

## 10. Umsetzung und Abnahme

Verbindliche Reihenfolge: M0 Verträge und Evidenz -> M1 State -> M2 Kandidaten/Validierung -> M3 Adapter -> M4 Pre-Run-Oberfläche -> M5 Evaluation -> M6 Post-Run -> M7 Integration und Dokumentation.

Alle Meilensteine sind offen. Konkrete Dateien, Schnittstellen, Testfälle und Freigaben stehen im [Implementierungsplan](pi_jev_implementierungsplan_de.md). Dieser Dokumentationsstand ist keine Code-, Provider- oder Bildqualitätsabnahme.
