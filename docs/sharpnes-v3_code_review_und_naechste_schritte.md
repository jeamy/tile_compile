# sharpnes-v3: Code-Review und nächste sinnvolle Schritte

**Datum:** 2026-07-31
**Branch:** `sharpnes-v3`
**Referenzanalyse:** `docs/m31_20260729_stacked_rgb_vergleich.md`
**Scope:** AQMH-Rekonstruktion, CFA/Debayering, Prewarp-Interpolation, Validierungs-Gates, Normal- und Resume-Ausgabepfade

---

## 1. Kurzfassung

Die bisherige Analyse ist in ihrer späteren Fassung grundsätzlich schlüssig: Die ursprüngliche Bewertung über globale Momenten-FWHM war zu schwach, während positionsgematchte Sternmessungen mit radialer FWHM und Peak/Flux deutlich aussagekräftiger sind.

Die Codebasis ist jedoch noch nicht auf einer ausreichend stabilen Grundlage für weitere Schärfeexperimente. Es gibt derzeit:

1. unterschiedliche Debayering-Implementierungen in Normal- und Resume-Pfaden;
2. dokumentierte AHD-/Bilinear-Ergebnisse, die vom aktuellen Normalpfad nicht reproduziert werden;
3. doppelte Gate- und Bildverarbeitungslogik;
4. eine zu weit reichende Relaxierung im Raw-AQMH-Baseline-Guard;
5. fehlende Tests für die entscheidenden Kombinationen der Kandidaten-Gates;
6. unzureichende Trennung zwischen historischer Run-Dokumentation und aktuellem Branch-Zustand.

### Gesamtentscheidung

Ja, die nächsten Schritte sollten mit Codebereinigung beginnen. Dabei sollte es aber nicht um ein rein kosmetisches Entfernen von Duplikaten gehen. Die erste Phase muss eine **verhaltensbewahrende Konsolidierung mit klaren Sicherheitsverträgen** sein. Erst danach sollte die Testbasis erweitert und anschließend ein neues Experiment nach dem anderen durchgeführt werden.

Die empfohlene Reihenfolge ist:

1. aktuellen Zustand einfrieren und reproduzierbar beschreiben;
2. doppelte Pfade und Gate-Logik konsolidieren;
3. Raw-Baseline-Guard und Debayer-Pfad logisch korrigieren;
4. Charakterisierungs- und Regressionstests ergänzen;
5. erst dann neue Runs mit jeweils genau einer Veränderung starten.

---

## 2. Kritische Befunde vor weiteren Runs

### 2.1 Normaler AQMH-Ausgabepfad verwendet weiterhin Nearest Neighbor

Die AQMH-Rekonstruktion leert bei OSC die RGB-Kanäle. Der normale Pipelinepfad erzeugt den finalen RGB-Output anschließend aus dem Mosaik über einen Nearest-Neighbor-Fallback.

Betroffene Stellen:

- `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp`
- `tile_compile_cpp/apps/runner_pipeline.cpp`
- `tile_compile_cpp/apps/runner_phase_post_stack_output.cpp`
- `tile_compile_cpp/apps/runner_resume.cpp`

Der Post-Stack-Helper nutzt dagegen OpenCV Edge-Aware-Debayering. Dieser Helper wird im aktuellen Code über den Resume-Pfad verwendet, nicht als einheitlicher Normalpfad.

**Folge:** Ein dokumentierter AHD- oder bilinearer Normal-Run kann mit dem aktuellen Branch nicht zuverlässig reproduziert werden. Normal-Run und Resume-Run können unterschiedliche Sternprofile und Farbartefakte erzeugen.

Vor weiteren Schärferuns muss daher eine Entscheidung getroffen und umgesetzt werden:

- entweder ein gemeinsamer, explizit benannter Debayerpfad für Normal- und Resume-Läufe;
- oder eine klare Dokumentation, dass beide Pfade absichtlich unterschiedliche Modi verwenden.

Empfehlung: gemeinsamer Output-Helper mit einem expliziten effektiven Modus, zum Beispiel `nearest`, `bilinear` oder `edge_aware`.

### 2.2 AHD-Bezeichnung korrigieren

Die neue OpenCV-Funktion verwendet `cv::COLOR_Bayer*2RGB_EA`. Das ist OpenCV Edge-Aware, nicht zwingend ein AHD-Verfahren. Die Dokumentation sollte deshalb nicht pauschal von AHD sprechen.

Empfohlene Bezeichnung:

```text
OpenCV Edge-Aware demosaicing (EA)
```

Die historische Run-Dokumentation sollte außerdem den Commit oder die genaue Quellversion nennen, mit der der jeweilige Run erzeugt wurde.

### 2.3 Raw-Baseline-Guard korrigiert

Im Ausgangszustand relaxierte `aqmh_raw_baseline_guard_decision()` den Seam-Schwellenwert pauschal auf das Doppelte, sobald irgendeine andere Raw-Gate-Metrik repariert wurde.

Damit konnte ein Kandidat akzeptiert werden, der gegenüber Raw AQMH einen Seam-Regressionswert oberhalb des normalen Grenzwerts besaß, sofern er beispielsweise den Background verbesserte.

Das widerspricht dem eigentlichen Vertrag:

```text
Post-Processing darf die immutable raw AQMH baseline nicht verschlechtern.
```

Der Guard verwendet jetzt eine zentrale Gate-Auswertung und akzeptiert nur noch Kandidaten, deren vollständiger Vergleich gegen Raw AQMH die strikten Gates besteht. Die pauschale Relaxierung und das interne `relaxed`-Ergebnisfeld wurden entfernt. Die Änderung ist durch eine Entscheidungsmatrix in den Tests abgesichert.

### 2.4 Sharpening-Gate ist methodisch nicht vollständig passend

Der Sharpening-Kandidat verwendet globale Momenten-FWHM als zwingendes Verbesserungsmerkmal. Die Vergleichsanalyse zeigt aber selbst, dass diese Metrik weiche und spitze Kerne nicht zuverlässig unterscheidet.

Für einen Star-Core-Kandidaten sollten deshalb mindestens zwei Ebenen getrennt werden:

- globale Sicherheitsmetriken: Background, Seam, Tail, globale FWHM;
- eigentliche Verbesserungsmetriken: positionsgematchte radiale FWHM und Peak/Flux.

Eine globale FWHM-Verbesserung allein darf nicht als Beweis für schärfere Sternkerne gelten.

---

## 3. Doppelte und zu konsolidierende Logik

### 3.1 Validierungs-Gates

Die Gate-Logik ist auf mehrere Stellen verteilt:

- `aqmh_tail_ok()`;
- `tail_gate_ok()`;
- `validation_gate_ok()`;
- lokale Lambdas in `aqmh_raw_baseline_guard_decision()`;
- erneute manuelle Prüfungen im AQMH-Runner.

Das erhöht die Gefahr, dass sich die Definition eines Gates zwischen Kandidatenauswahl, Raw-Guard und finalem Fallback unterscheidet.

**Maßnahme:** Eine zentrale Gate-API einführen, die für jede Metrik zurückgibt:

```text
applicable
value
control
regression
threshold
strict_ok
reason
```

Die Kandidatenauswahl und der Raw-Guard sollten diese API verwenden und nicht eigene Kopien der Bedingungen enthalten.

### 3.2 High-Pass- und Unsharp-Berechnung

`low_frequency_neutralized_aqmh()` und `unsharp_masked_aqmh()` duplizieren die Gaussian-Blur- und pixelweise High-Pass-Logik.

**Maßnahme:** Einen kleinen internen Helper für die Berechnung eines Gaussian-High-Pass-Bildes einführen. Die unterschiedlichen Operationen bleiben darüber klar:

```text
neutralization: base - low_frequency_residual
sharpening:     base + amount * high_pass(base)
```

Die Refaktorierung darf zunächst keine Parameteränderung und keine Änderung des numerischen Verhaltens bewirken.

### 3.3 Normal- und Resume-Output

Der Normalpfad enthält eigene Output-, Crop-, Scaling- und Debayer-Logik. Zusätzlich existiert `runner_phase_post_stack_output.cpp` mit ähnlichen Aufgaben für Resume.

**Maßnahme:** Den gemeinsamen Teil in einem Helper bündeln. Dabei müssen vorab die Verträge festgelegt werden:

- Eingabe ist Mosaic oder RGB;
- Bayer-Origin-Konvention;
- Crop-Anpassung des Origins;
- Scaling vor oder nach Debayering;
- Maskierung vor oder nach Debayering;
- Stretching;
- Ausgabeformat.

Der `have_rgb`-Check muss immer alle drei Kanäle prüfen, nicht nur `R`.

### 3.4 Veraltete `debayer_before_stack`-Reste

`debayer_before_stack` wurde revertiert. Historische Dokumentation darf bleiben, muss aber klar als nicht aktueller Experimentstand markiert sein. Aktive Konfiguration, Schema und Runner dürfen diese Option nicht mehr implizit suggerieren.

Die Diagnosebezeichnung im separaten Preprocessing-Pfad sollte geprüft werden, damit sie nicht mit dem aktuellen AQMH-Pfad verwechselt wird.

---

## 4. Vorgehensplan Phase 0: Zustand einfrieren

Vor dem Refactoring sollte der aktuelle Zustand als technische Baseline festgehalten werden.

### 4.1 Quellzustand

Dokumentieren:

- Branch und Commit-ID;
- Build-Konfiguration;
- Compiler/OpenCV-Version;
- ausgewählter Acceleration-Backend;
- effektiver Debayerpfad;
- effektive Prewarp-Interpolation;
- relevante AQMH-Konfiguration;
- Testfallzahl.

### 4.2 Run-Artefakte

Die bestehenden Runs bleiben unverändert. Für Vergleichszwecke sollen nur lesend erfasst werden:

- effective config;
- phase events;
- AQMH-Validierungsartefakte;
- `selected_candidate`;
- `raw_aqmh_validation`;
- `final_vs_raw_aqmh_validation`;
- finaler Debayer-/Outputpfad;
- Bilddimensionen und Crop-Origins.

Keine vorhandenen Runs oder Outputs überschreiben.

### 4.3 Charakterisierungstests vor Refactoring

Vor einer größeren Konsolidierung sollten Tests für das bisherige Verhalten der folgenden Komponenten vorhanden sein:

- Bayer-Origin-Parität;
- Normal- und Resume-Origin nach Crop;
- Debayering auf konstantem und nicht konstantem CFA-Testbild;
- Raw-Guard;
- Candidate-Selection-Reihenfolge;
- finaler Fallback auf Raw AQMH.

Damit kann die anschließende Codebereinigung gegen Regressionen abgesichert werden.

---

## 5. Vorgehensplan Phase 1: sichere Codebereinigung

Diese Phase darf zunächst keine neuen Schärfeparameter und keine neue Bildverarbeitungs-Idee einführen.

### Schritt 1: Gate-Logik zentralisieren

1. gemeinsame Metrik-Gate-Struktur definieren;
2. `aqmh_tail_ok()` und `tail_gate_ok()` zusammenführen;
3. Raw-Guard auf die zentrale Gate-Auswertung umstellen;
4. keine pauschale Relaxierung der Raw-Baseline zulassen;
5. Gründe und Applicability im Artefakt speichern.

### Schritt 2: Output-Pfade konsolidieren

1. Normal- und Resume-Post-Stack-Output vergleichen;
2. gemeinsamen Helper definieren;
3. Bayer-Origin und Crop-Origin als expliziten Vertrag testen;
4. all-channel RGB-Checks einführen;
5. effective Debayer-Modus im Run-Artefakt speichern.

### Schritt 3: High-Pass-Code konsolidieren

1. gemeinsamen Gaussian-High-Pass-Helper einführen;
2. numerisches Verhalten mit Testbildern vergleichen;
3. keine Änderung an `sigma` oder `amount` während dieses Schrittes;
4. Unsharp-Sharpening weiterhin standardmäßig nur als validierten Kandidaten behandeln.

### Schritt 4: tote oder irreführende Konfigurationspfade entfernen

1. aktive Referenzen auf revertierte `debayer_before_stack`-Option prüfen;
2. historische Dokumentation klar kennzeichnen;
3. keine alte Run-Konfiguration ändern;
4. nicht verwendete Debug-Ausgaben und irreführende Modusnamen entfernen.

---

## 6. Vorgehensplan Phase 2: Testbasis für weitere Experimente

### 6.1 Raw-Baseline-Guard-Matrix

Mindestens folgende Fälle müssen als Unit-Tests abgedeckt werden:

| Fall | Raw vs. Control | Candidate vs. Raw | Erwartung |
|---|---|---|---|
| A | alle Gates gültig | Regression | Ablehnung |
| B | ein Gate ungültig | genau dieses Gate repariert | Annahme möglich |
| C | Background repariert, Seam verschlechtert | Seam über normalem Grenzwert | Ablehnung |
| D | FWHM repariert, Background verschlechtert | Background über Grenze | Ablehnung |
| E | nicht anwendbare Tail-Metrik | keine Sternbasis | nicht blockieren |
| F | alle Metriken degeneriert | keine belastbare Referenz | sichere Standardentscheidung |

### 6.2 CFA- und Debayer-Tests

Die bestehenden konstanten CFA-Tests sind nützlich, prüfen aber hauptsächlich Farbzuordnung und Origin-Parität. Zusätzlich erforderlich sind:

- nicht konstante Farbverläufe;
- einzelner heller Stern auf Hintergrund;
- diagonale Kanten;
- unterschiedliche Origins für alle vier Bayer-Muster;
- NaNs und Canvas-Masken;
- Wertebereiche mit negativem Hintergrund;
- Vergleich Bilinear gegen Edge-Aware;
- Randverhalten.

### 6.3 Normal-/Resume-Parität

Ein kleiner Fixture-Test sollte denselben Mosaic-Input durch beide Outputpfade führen und vergleichen:

- Crop;
- Origin;
- RGB-Kanalgrößen;
- Scaling;
- Maskierung;
- Debayer-Modus;
- erzeugte Pixelwerte innerhalb definierter Toleranzen.

### 6.4 Prewarp-Backend-Tests

Für jede angeforderte Interpolation sollte das effektive Ergebnis protokolliert werden:

```text
requested: linear | cubic | lanczos4
effective: cpu | opencv_cuda | opencv_opencl
fallback_reason: none | unsupported | runtime_error
```

Insbesondere `lanczos4` mit CUDA darf nicht stillschweigend wie ein echter CUDA-Lauf aussehen.

### 6.5 Positionsgematchte Qualitätsmetriken

Für synthetische und reale Fixture-Bilder sollten Tests beziehungsweise Analysewerkzeuge liefern:

- gemeinsame Sternpositionen;
- radiale FWHM;
- Peak/Flux;
- zweite Momente nur als ergänzende Metrik;
- Background-RMS auf separater Background-Maske;
- Tail und Elongation;
- Anteil der Sterne, bei denen Referenz oder Kandidat besser ist.

Die Sternpopulation darf nicht für jede Variante unabhängig neu ausgewählt werden.

---

## 7. Vorgehensplan Phase 3: neue Experimente

Erst wenn Phase 1 und Phase 2 abgeschlossen sind, sollten neue Runs gestartet werden.

### Priorität 1: Prewarp-Interpolation

Ein kontrollierter A/B-Vergleich:

```text
linear vs. cubic vs. lanczos4
```

Alles andere unverändert lassen. HMS und lineare Daten getrennt auswerten. Bei `lanczos4` Backend-Fallbacks aus dem Artefakt berücksichtigen.

### Priorität 2: AQMH-Weight-Tuning

Jeweils nur einen Parameter ändern:

- `w_sharp`;
- `base_window_px`;
- gegebenenfalls `score_scale`.

Nicht gleichzeitig Weight-Tuning und Debayering ändern. Die Auswirkungen müssen an denselben Sternpositionen und mit demselben Raw-Baseline-Guard gemessen werden.

### Priorität 3: Maskierte Post-Stack-Deconvolution

Zunächst als externer, reversibler Prototyp auf existierenden Outputs:

- nur lineare oder exakt kontrolliert normalisierte Daten;
- Sternkernmaske;
- Background außerhalb der Maske;
- PSF aus mehreren Sternen;
- mehrere Iterations-/Regularisierungsstufen;
- harte Ablehnung bei Background- oder Halo-Verschlechterung.

Die erwartete Verbesserung nicht vorab als garantiert annehmen.

### Priorität 4: Registration auf debayerter Luminanz

Als isolierter Proxy-Versuch, ohne sofort die komplette AQMH-Rekonstruktion auf drei Kanäle umzustellen.

### Priorität 5: Proper Debayer-First AQMH

Erst nach den kleineren Versuchen und nur mit einem klaren Architekturdesign:

1. Debayer nach Normalisierung;
2. Registration auf debayerter Luminanz oder RGB;
3. gemeinsame oder kanalabhängige Qualitätskarten bewusst entscheiden;
4. per-Kanal-Reconstruction;
5. Speicher-, Cache- und GPU-Verträge definieren;
6. gegen einen echten AQMH-CFA-Baseline-Run vergleichen.

Das frühere Mean-Stack-Experiment reicht allein nicht aus, um den erwarteten Gewinn von Proper Debayer-First AQMH zu quantifizieren.

---

## 8. Was zunächst nicht gemacht werden sollte

- keine gleichzeitige Änderung von Debayering, Prewarp, AQMH-Gewichten und Deconvolution;
- keine Bewertung nur anhand globaler Momenten-FWHM;
- keine neuen Runs mit unklarem effektivem Outputpfad;
- keine Änderungen an bestehenden `runs/`-Artefakten;
- keine globale Unsharp-Mask als Produktionsstandard;
- keine Ableitung kumulativer Verbesserungen aus getrennten Experimenten;
- keine GPU-Leistungs- oder Qualitätsaussage ohne effektiven Backend-Nachweis.

---

## 9. Abnahmekriterien für eine testfähige Basis

Die Codebasis ist für weitere Schärfetests ausreichend vorbereitet, wenn alle folgenden Punkte erfüllt sind:

- Normal- und Resume-Output verwenden denselben dokumentierten Debayer-Vertrag;
- der Raw-Baseline-Guard relaxiert keine gültige Fremdmetrik mehr pauschal;
- Gate-Logik ist zentralisiert und getestet;
- RGB-Gültigkeit prüft R, G und B gemeinsam;
- effektive Interpolation und Backend-Fallbacks werden protokolliert;
- CFA-Tests decken nicht konstante Muster und Randfälle ab;
- Positionsgematchte Metriken sind reproduzierbar;
- ein vollständiger Build und die gesamte Testsuite laufen erfolgreich;
- jeder neue Run verändert genau eine experimentelle Variable;
- Run-Dokumentation enthält Commit, effective config und selected candidate.

---

## 10. Umgesetzte Schritte 1–7

Die ersten sechs Schritte wurden jetzt im Code umgesetzt beziehungsweise verifiziert:

- Gate-Entscheidungen werden zentral über `evaluate_aqmh_validation_gates()` berechnet;
- die doppelte lokale Gate-Logik im AQMH-Runner wurde entfernt;
- der Raw-Baseline-Guard akzeptiert keine pauschale Seam-Relaxierung mehr;
- das veraltete `relaxed`-Ergebnisfeld wurde entfernt, die externe Artefaktform bleibt mit `relaxed_used: false` kompatibel;
- Normal- und Resume-AQMH-Fallback verwenden denselben OpenCV-Edge-Aware-Debayerpfad;
- der Resume-RGB-Check prüft jetzt alle drei Kanäle;
- Gaussian-Blur/High-Pass-Code für Neutralisierung und Unsharp-Kandidaten verwendet einen gemeinsamen Helper;
- Tests für Gate-Auswertung, Raw-Guard-Seam-Schutz und falsche Debayer-Origin wurden ergänzt.

Schritt 7 wurde als Experimentmatrix vorbereitet, aber nicht als Bildverarbeitungslauf gestartet. Die Repository-Regeln behandeln Runs als explizit freizugebende Operation. Vor einem konkreten Run müssen daher Input, Konfiguration und Ziel-Run ausdrücklich benannt werden.

Noch offen für eine spätere zweite Konsolidierungsrunde ist die vollständige Zusammenführung der umfangreichen Normal- und Resume-Outputlogik für Crop, Scaling und FITS-Ausgabe. Die Debayer-Semantik ist bereits angeglichen; eine vollständige Output-Refaktorierung sollte erst nach zusätzlichen Fixture-Tests erfolgen.

---

## 11. Aktueller Verifikationsstand

Zum Zeitpunkt dieses Reports wurden im vorhandenen Buildtree ausgeführt:

```text
cmake --build build --target tile_compile_runner tests -j2
./build/tests
```

Ergebnis:

```text
Build erfolgreich
tile_compile_runner erfolgreich
All tests passed
29620 assertions in 255 test cases
```

Es wurde kein Backend gestartet. Nach Abschluss der Codebereinigung wurden zwei kontrollierte Läufe auf dem unveränderten Input `/media/tc_ssd/M31_ligths_all` durchgeführt:

- `M31-s1_guardfix_20260731_1`: identische Konfiguration, aktueller Code;
- `M31-s1_lanczos4_20260731_1`: identische Konfiguration, nur `aqmh.reconstruction.prewarp_interpolation: lanczos4`.

Der historische Run `M31-s1_20260731_053413` wurde nur read-only analysiert. Die FITS-Dateien dieses historischen Runs waren am Ende der Analyse nicht mehr im Run-Verzeichnis vorhanden; dessen bereits gelesene JSON-Artefakte bleiben als historische Evidenz verwendbar, ein erneuter Bildvergleich gegen diese FITS-Dateien war dadurch nicht möglich.

---

## 12. Run- und Konfigurationsanalyse 2026-07-31

### 12.1 Kontrolllauf mit aktuellem Guard

`M31-s1_guardfix_20260731_1` lief erfolgreich bis `DONE`.

Der aktuelle Raw-Guard verhielt sich anders als im historischen Lauf:

| Variante | Selected candidate | Struktur-Alpha | Raw-Guard | Uniform-Control-Gate |
|---|---|---:|---|---|
| historischer Run | `structure_masked_detail` | 1.0 | `relaxed_used=true` | nicht ausgelöst |
| aktueller Guard-Fix | `structure_masked_detail` | 0.1796875 | `strict_raw_baseline_pass` | nicht ausgelöst |

Der Vollkandidat mit Alpha 1.0 wurde wegen der Raw-Seam-Regressionsgrenze nicht akzeptiert. Die attenuierte Suche fand stattdessen vier zulässige Kandidaten und wählte Alpha `0.1796875`.

Finale AQMH-Validierung des Guard-Fix-Laufs:

- Background-Regression gegen Uniform Control: `+0.274 %`;
- FWHM-Regression: `-0.91 %`;
- Seam-Regression: `+4.991 %`, knapp unter dem Grenzwert `5 %`;
- Tail-Regression: `+2.90 %`;
- alle Gates bestanden.

Die Raw-Baseline wurde dabei nicht verletzt:

- Background gegenüber Raw: `-18.75 %`;
- FWHM gegenüber Raw: `-1.91 %`;
- Seam gegenüber Raw: `+4.99 %` innerhalb der strikten Grenze.

### 12.2 `lanczos4`-Lauf

`M31-s1_lanczos4_20260731_1` lief technisch bis `DONE`, endete aber mit `validation_failed`. Das war kein OOM-Abbruch.

| Metrik | Guard-Fix | Lanczos4 | Lanczos4-Befund |
|---|---:|---:|---|
| AQMH Background vs. Control | `+0.27 %` | `+26.65 %` | Gate verletzt |
| AQMH FWHM vs. Control | `-0.91 %` | `+0.98 %` | kein Vorteil in diesem Gate |
| AQMH Seam vs. Control | `+0.20 %` | `-6.38 %` | Verbesserung |
| Selected candidate | `structure_masked_detail` | `raw_aqmh` | Raw-Fallback |
| Run-Ende | `ok` | `validation_failed` | erwartete Sicherheitsentscheidung |

Die positionsgematchte Vergleichsmessung der beiden HMS-Ausgaben ergab mit derselben lokalen Näherungsmessung:

- Lanczos4 radialer Kern-FWHM: etwa `17.2 %` kleiner;
- Lanczos4 Peak/Flux: etwa `20.0 %` höher;
- Lanczos4 war bei rund `99.6 %` der gepaarten Sterne radial schärfer.

Dieser Schärfegewinn wird aber durch eine starke Hintergrund-/Rauschverschlechterung erkauft. Lanczos4 ist mit der aktuellen AQMH-Gate-Konfiguration daher **kein akzeptabler Produktionskandidat**.

Der Run belegte zeitweise etwa `13.4 GiB` RSS, bei etwa `43 GiB` virtuellem Adressraum und rund `18 GiB` belegtem Swap. Er wurde nicht durch OOM beendet, erzeugte aber erheblichen Speicherdruck. `runtime_limits.memory_budget: 4096` begrenzt den gesamten Prozess-RSS nicht ausreichend.

### 12.3 Konfigurationsvariablen ohne Wirkung oder mit widersprüchlichem Verhalten

#### Linearity

Die Konfiguration enthält:

```yaml
linearity:
  strictness: strict
  min_overall_linearity: 0.9
```

Im Lauf wurden acht von acht geprüften Frames als nicht linear markiert, `overall_linearity` war `0.0`, trotzdem wurden alle 645 Frames behalten und der Lauf fortgesetzt. `strictness` beeinflusst nur die Prüfgrenzwerte; `min_overall_linearity` wird im Runner nur ins Artefakt geschrieben. Eine tatsächliche Ablehnung oder konfigurierbare Aktion findet nicht statt.

Das ist eine echte Konfigurations-/Implementierungsdiskrepanz. Entweder muss `strictness: strict` tatsächlich Frames oder den Lauf stoppen, oder die Dokumentation muss den Modus als warn-only bezeichnen.

#### AQMH-Diagnostics

Im Run war eingestellt:

```yaml
aqmh:
  diagnostics:
    level: full
    per_frame_blocks: false
    heatmaps: false
    regions: false
    format: json
```

Damit werden nur die Basis-Frame-Diagnosen geschrieben. `binary_block_size_px`, `q_region` und `r_morph_canvas_px` hatten in diesem Lauf keine praktische Wirkung. `level: full` ist hier irreführend, weil die drei umfangreichen Full-Diagnostics-Schalter deaktiviert sind.

#### AQMH-Cache

`max_resident_maps: 2` war konfiguriert. Beide AQMH-Läufe zeigen jedoch:

```text
cache_hits = 0
max_resident_maps_observed = 0
```

Der aktive Region-Streaming-Pfad liest Teilbereiche und befüllt den Full-Map-LRU nicht. Für diesen Lauf war `max_resident_maps` daher wirkungslos beziehungsweise nicht nachweisbar wirksam.

#### Cherry-Pick

`aqmh.cherry_pick.enabled: false`. Damit waren `mode`, `k_frac`, `k_min_required`, `margin_min`, `reject_below_best_fraction`, `min_keep_fraction` und `tiered_k_frac` in beiden Läufen ohne Wirkung.

#### Classic-Konfiguration im AQMH-Lauf

Da `method: aqmh` gesetzt ist, wurden Classic-Pfade übersprungen. In diesem Lauf waren daher nicht aktiv:

- `local_metrics.*`;
- `synthetic.*`;
- `stacking.sigma_clip.*`;
- `stacking.cluster_quality_weighting.*`;
- `tile_denoise.*`;
- top-level `validation.min_fwhm_improvement_percent`;
- top-level `validation.min_tile_weight_variance`;
- top-level `validation.require_no_tile_pattern`.

`global_metrics.*` wird zwar berechnet, dient aber primär der klassischen globalen Qualitäts-/Registrierungslogik und ist nicht identisch mit `aqmh.global_quality.*`.

#### Chroma-Denoising

Die Konfiguration enthielt:

```yaml
chroma_denoise:
  enabled: true
  apply_stage: post_stack_linear
```

Im AQMH-OSC-Pfad werden die RGB-Rekonstruktionskanäle vor diesem Schritt geleert. Die Bedingung für `post_stack_linear` ist damit nicht erfüllt. Das Chroma-Denoising wurde in diesem Lauf nicht auf die AQMH-Ausgabe angewandt.

#### BGE-Autotuning

Der Run verwendete `bge.method: autobge`. Die separate AutoBGE-Implementierung verwendet die generischen `bge.fit.*`, `bge.grid.*`, `bge.mask.*`, `bge.sample_*`- und `bge.autotune.*`-Parameter nicht in derselben Weise wie der klassische BGE-Pfad.

Das Artefakt zeigt entsprechend:

```text
autotune.enabled = true
autotune.evals_performed = 0
```

Die Konfiguration suggeriert aktives Autotuning, der konkrete AutoBGE-Lauf führte aber keine Autotune-Evaluationen aus. Das sollte entweder im Schema getrennt werden oder im Artefakt ausdrücklich als `not_applicable_for_autobge` erscheinen.

#### Top-Level-Validation

`validation.max_background_rms_increase_percent: 0` deaktiviert im aktuellen Code die entsprechende Prüfung, weil nur Werte `> 0` als Limit aktiviert werden. Das ist zusätzlich zur AQMH-eigenen Validierung und kann für Anwender irreführend sein: `0` sieht wie „keine Erhöhung erlaubt“ aus, bedeutet aber „Limit deaktiviert“.

### 12.4 Prewarp-/GPU-Nachvollziehbarkeit

Der Lanczos4-Lauf protokolliert `selected_backend: opencv_cuda` und `interpolation: lanczos4`. Die Artefakte enthalten jedoch keinen effektiven Kernel-/Fallback-Nachweis pro Operation.

Da OpenCV-CUDA-Warping nicht dieselben Interpolationsmodi wie der CPU-Pfad garantiert unterstützt, muss künftig zwischen folgenden Zuständen unterschieden werden:

```text
requested_interpolation: lanczos4
effective_interpolation: lanczos4 | linear | cubic
effective_backend: cpu | opencv_cuda | opencv_opencl
fallback_reason: none | unsupported_interpolation | runtime_error
```

Ohne diese Information ist der Lanczos4-Run zwar reproduzierbar konfiguriert, aber nicht vollständig reproduzierbar hinsichtlich des tatsächlich verwendeten Warpkernels.

---

## 13. Schlussfolgerung aus den Runs

Die aktuelle strikte Raw-Guard-Implementierung arbeitet sicherer als der historische Run und findet trotzdem eine zulässige, attenuierte Strukturblending-Lösung.

Lanczos4 zeigt einen realen Schärfegewinn, ist aber in der aktuellen Form wegen der deutlichen Background-RMS-Verschlechterung nicht produktionsfähig. Ein einfaches Erhöhen der Gate-Grenze wäre nicht sinnvoll; zuerst muss geklärt werden, ob der Rauschzuwachs durch den Kernel, einen GPU-Fallback, Ringing oder die anschließende BGE/PCC-Kette entsteht.

Die nächste sinnvolle technische Arbeit ist daher nicht ein weiterer voller Lanczos-Run, sondern:

1. Linearity-Aktion und `min_overall_linearity` korrigieren;
2. AutoBGE- und Classic-BGE-Konfiguration klar trennen;
3. AQMH-Diagnostics- und Cache-Parameter mit effektiven Zuständen ausstatten;
4. effektiven Prewarp-Backend/Kernels protokollieren;
5. Speicherplanung so ändern, dass `memory_budget` den Prozess-RSS tatsächlich begrenzt;
6. erst danach einen kleineren, speicherbegrenzten Interpolationsvergleich wiederholen.

---

## 14. Test der Vorschläge 4 und 5

### 14.1 Vorschlag 4: Proper Debayer-First AQMH

Der Ansatz ist bereits durch das historische Experiment D+ getestet worden. Die damalige Integration debayerte vor dem Warp, verwendete aber nur Frame-Level-Global-Quality-Gewichte und keinen vollständigen Per-Pixel-AQMH-Reconstruction-Pfad.

Das Ergebnis war kein belastbarer Produktionsnachweis für Proper Debayer-First AQMH:

- D+ verbesserte den isolierten Mean-Stack-Vergleich;
- die integrierte AQMH-Variante erreichte keinen Schärfegewinn;
- die produktive Variante mit per Kanal angewandten AQMH-Maps wurde nicht als sauberer, speicherbegrenzter Architekturtest ausgeführt.

Ein erneuter vollständiger D+-Run wurde deshalb nicht gestartet. Der vorhandene Nachweis reicht aus, um den einfachen RGB-Mean-Stack-Ansatz zu verwerfen; er reicht nicht aus, um Proper Debayer-First AQMH endgültig zu widerlegen.

Für einen späteren echten Test wären erforderlich:

- gemeinsame debayerte Luminanz-Qualitätskarten;
- per Kanal angewandte Per-Pixel-Gewichte;
- explizite Speicherbegrenzung;
- getrennte RGB- und Luminanz-Validierung;
- Vergleich gegen denselben CFA-AQMH-Baseline-Run.

### 14.2 Vorschlag 5: Zwischenstufenanalyse

Die Zwischenstufen wurden für Guard-Fix und Lanczos4 analysiert:

1. `aqmh_reconstructed_raw.fit` — AQMH-Mosaik vor finalem Debayering;
2. `stacked_rgb.fits` — lineares RGB-Ausgabebild;
3. `stacked_rgb_hms.fits` — PCC/HypMetric-Endbild.

Die 2×2-/Bayer-Struktur ist bereits in der Mosaik-Zwischenstufe sichtbar. Das finale Edge-Aware-Debayering erzeugt sie daher nicht primär, sondern übernimmt eine Struktur, die bereits aus CFA-Abtastung und CFA-Prewarp stammt.

Die Hochfrequenzenergie in positionsgematchten Sternkernen war im Lanczos4-Lauf gegenüber Guard-Fix ungefähr erhöht um:

- R: `+28 %`;
- G: `+29 %`;
- B: `+14 %`.

Das erklärt, warum Lanczos4 schärfer aussieht, gleichzeitig aber die Block-/Rasterstruktur stärker hervortritt. Die globale Luminanz-Checker-Metrik allein ist dafür nicht ausreichend, weil das CFA-Mosaik selbst erwartungsgemäß eine 2×2-Struktur besitzt. Entscheidend ist die Kombination aus:

- Mosaik-Zwischenbild;
- RGB-Kanal-Hochfrequenz;
- positionsgematchter Sternmessung;
- Background-/Seam-Gates.

### 14.3 Entscheidung

Die Vorschläge 4 und 5 führen zu keiner unmittelbaren Änderung der Produktionskonfiguration:

- der einfache Debayer-First-Mean-Stack bleibt verworfen;
- Lanczos4 bleibt wegen der Background-Regressionsgrenze verworfen;
- die Blockstruktur entsteht primär vor dem finalen Debayering im CFA-Domain-Processing;
- weitere Varianten 1–3 werden auf Wunsch des Auftraggebers nicht weiter getestet.

---

## 15. Schlussfolgerung

Die vorgeschlagene Reihenfolge des Auftraggebers ist richtig, wenn „doppelten Code entfernen“ als **sichere Konsolidierung von Gate-, Output- und Debayer-Logik** verstanden wird.

Die unmittelbar sinnvollste Arbeitsreihenfolge lautet:

1. Raw-Guard-Vertrag und aktuelle Debayerpfade festlegen;
2. doppelte Gate- und Outputlogik konsolidieren;
3. Normal-/Resume-Parität und CFA-Grenzfälle testen;
4. Backend-/Interpolations-Fallbacks sichtbar machen;
5. danach erst Prewarp- und AQMH-Tuning testen;
6. Deconvolution prototypisch und maskiert prüfen;
7. Proper Debayer-First AQMH erst als abschließenden Architekturversuch angehen.

Damit wird aus dem aktuellen Branch eine nachvollziehbare Testbasis, anstatt weitere Ergebnisse auf teilweise unterschiedlichen und historisch vermischten Pipelinepfaden aufzubauen.
