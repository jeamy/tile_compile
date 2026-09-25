# Jev M5: reproduzierbare Evaluation und Freigabegrenze

Stand: 2026-09-25. Der maschinenlesbare Snapshot liegt in
`pi_jev_m5_report_20260925.json`. M5 stellt einen read-only Replay- und Evidenzbericht bereit; **kein
Kandidat ist regulär freigegeben**. Die acht bestehenden M31/M42/IC5070/M66-Läufe
liegen inhaltlich unverändert unter `/media/data/tile_compile_cache/jev-test/`.
Der Bericht wird bewusst außerhalb von `runs/` erzeugt.

## Reproduzieren

```bash
python3 -m unittest discover -s web_backend_cpp/scripts -p 'test_evaluate_pi_decisions.py' -v > /tmp/out_jev_m5_tests.txt 2>&1
python3 web_backend_cpp/scripts/evaluate_pi_decisions.py \
  --decisions-dir runs/.pi_memory/pi_decisions \
  --registry web_backend_cpp/config/pi_decisions/evaluation_registry_v1.json \
  --policy web_backend_cpp/config/pi_decisions/release_policy_v1.json \
  --runs-dir /media/data/tile_compile_cache/jev-test \
  --pi-memory runs/.pi_memory/memories_v2.jsonl \
  --out /tmp/jev_m5_report.json > /tmp/out_jev_m5_report.txt 2>&1
```

Das Werkzeug benötigt Python 3 und PyYAML. Es liest `pi_decisions/`, den optionalen PiMemoryStore sowie
Run-Provenance/Qualitätsartefakte nur lesend. Der Ausgabepfad darf nicht in
den Decision- oder Run-Stores liegen. Der Bericht enthält Digests der
Entscheidungsdateien, State-Hash, Policy, Modellantwort, Registry und
PiMemoryStore. Er schreibt keine Provider-Anfragen, API-Header oder
Eingabepfade in den Bericht. Bei unvollständigen Entscheidungen, fehlenden
Frame-Hashes oder Frame-Überschneidung zwischen Kalibrierung und Test bricht
er ab. Die behauptete an/aus-Belegung der Vergleichsarme wird gegen die
jeweilige effektive `config.yaml` geprüft. Auch `release_eligible` ist derzeit fail-closed: Es wird ohne
implementierte Mess- und Schwellwertprüfung nie `true`.

## Ist-Befund

- Zwei gespeicherte Jev-Entscheidungen: beide `set_sensor_profile_dwarf_ii`,
  konsistente State-/Antwort-/Kandidaten-/Update-Validierung, null beobachtete
  ungültige anwendbare Patches. Die Wilson-95-%-Obergrenze beträgt bei nur
  zwei Entscheidungen dennoch rund 65,8 %. Diese Daten belegen weder
  Modellkalibrierung noch Bildqualitätsgewinn.
- Vier adaptive-weight-Paare (M31, M42, IC5070, M66) besitzen je Paar exakt
  denselben Frame-Manifest-Hash, dieselbe Frame-Anzahl, dieselbe Build-ID und
  denselben Ausführungsscope. Die Vergleichsarme zeigen `adaptive_weights`
  an/aus, **nicht** vier tatsächlich ausgeführte Jev-Baselines. Alle vier
  Paare wurden vor dem Festlegen der Freigabepolicy untersucht. Der
  Session-Split im Register ist daher nur Daten-Trennung, kein blinder
  prospektiver Test.
- Die vorhandenen `forward_drizzle.json`-Werte enthalten unabhängige
  Sternstichproben. Deren FWHM-/Elongation-Werte bleiben im Report explizit
  `descriptive_unmatched_metrics`; sie sind keine gepaarten Qualitätsdeltas.
  Ein Bildqualitätslabel wird ohne gematchte Sterne, gleiche gültige
  Flächen und Signalmessung nicht vergeben. Abbruch/Fehlschlag wird nicht als
  negative Bildqualität gewertet.
- `filter` und `quality_group` sind im retrospektiven Register `unknown`, wo
  sie aus den verfügbaren Artefakten nicht abgesichert sind. Keine
  nachträgliche Erfindung dieser Strata.

## Eingefrorene Grenzen für neue Test-Sessions

`release_policy_v1.json` wurde am 2026-09-25 um 11:22:54 UTC festgelegt.
Die Werte sind **projektspezifische Akzeptanzgrenzen**, keine aus den vier
bekannten Paaren geschätzten Effekte oder universelle astronomische Normen.
Sie gelten ausschließlich für danach eingeschriebene, unabhängige Sessions.
Änderungen an diesen Werten erfordern eine neue Policy-Version; M31/M42/
IC5070/M66 bleiben retrospektive Exploration und dürfen nicht zur
Bestätigung herangezogen werden. Diese Vorabfestlegung folgt dem allgemeinen
Prinzip, Nichtunterlegenheitsgrenzen vor dem Test zu definieren; die konkrete
Höhe hier ist unsere Produktentscheidung, nicht aus der [FDA-Methodik](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/non-inferiority-clinical-trials)
übernommen.

Für beide Kandidaten gelten: mindestens 12 unabhängige Test-Sessions in
mindestens zwei vorab bestimmten Objektgruppen mit mindestens vier Sessions
je Gruppe; 95-%-Konfidenzintervalle auf **Session-Ebene** (gepaarter
Perzentil-Session-Bootstrap, 10 000 Ziehungen, Seed 20260925), nicht auf der Zahl
einzelner Sterne oder Pixel. Keine ungültigen anwendbaren Patches, kein
versagendes bestehendes Qualitätsgate. Auf einer getrennten, festgelegten
Entscheidungssuite: mindestens 30 unabhängig gelabelte vorteilhafte Fälle
und 75 unsichere/nicht vorteilhafte Fälle; die Wilson-Untergrenzen (95 %)
für korrekte Auswahl und korrektes Enthalten müssen mindestens 0,80 bzw.
0,95 betragen. Enthaltung ist bei fehlendem Qualitätsnachweis korrekt;
eine niedrige rohe Enthaltungsrate ist **kein** Qualitätsziel.
`minimum_coverage` bezeichnet Schnittmenge gültiger Pixel geteilt durch
gültige Kontrollpixel; `maximum_coverage_loss` ist die Differenz der
gültigen Flächenanteile am identischen Ausgaberaster. Alle Verhältnis-Metriken
verwenden denselben vorab fixierten Validbereich beider Arme.

| Kandidat | Primärziel, 95-%-KI | Sicherheitsgrenzen |
|---|---|---|
| `enable_adaptive_weights` | Obergrenze des Session-KI für das Verhältnis gematchter Median-FWHM (an/aus) höchstens **0,95**; gleicher 5-%-Verbesserungsmaßstab wie die vorhandene Multiband-Promotionsgrenze, aber eigener Test | Mindestens 50 gematchte, ungesättigte Sterne, 100 000 gemeinsame gültige Hintergrundpixel und 20 Signalaperturen pro Session. Gemeinsame gültige Fläche mindestens 95 % der Kontrollfläche; Coverage-Verlust höchstens 0,5 Prozentpunkte. In jeder Objektgruppe: KI-Obergrenze FWHM ≤ 1,01, Elongation ≤ 1,01 und Hintergrund-RMS ≤ 1,02 des Kontrollarms; KI-Obergrenze relativer Signalverlust ≤ 1 %. |
| `set_sensor_profile_dwarf_ii` | Untergrenze des Session-KI für die relative Reduktion des unabhängig referenzierten Luminanzfehlers gegenüber `rec709` mindestens **10 %**; obere KI-Grenze des normierten absoluten Fehlers höchstens 5 % | Mindestens 20 ungesättigte, unabhängig kalibrierte Referenzpatches pro Session; exakte DWARF-II-Headerzuordnung in 100 % der anwendbaren Fälle. Alle prä-HMS-linearen Produkte, einschließlich Masken, müssen bitidentisch sein (0 % Sternform-/Rausch-/Signal-/Coverage-Änderung); gemeinsame HMS-gültige Fläche mindestens 99 % der gültigen Kontrollfläche. Zusätzlich höchstens +0,1 Prozentpunkte Clipping und höchstens 2 % Verlust an Schatten- oder Lichterkontrast auf vorab fixierten ROIs. |

Die strengere adaptive Rauschgrenze von +2 % liegt unter dem bestehenden
Multiband-Veto von +5 % und nahe dem beobachteten Nachteil von etwa 1,4–2,2 %;
ohne den geforderten 5-%-Schärfegewinn ist dieser Tausch nicht akzeptiert.
Die bisherigen FWHM-Differenzen von 0,17–0,71 % stammen aus **nicht
gematchten** Sternpopulationen und kalibrieren weder die CI-Breite noch die
Eingriffsgrenze. Das Minimum von 12 Sessions ist nur eine Untergrenze;
bei breitem KI besteht der Kandidat unabhängig von der Sessionzahl nicht.
Die Runtime-Vorregeln `min_measurement_coverage` und besonders
`min_metric_agreement` bleiben davon getrennt und produktiv ungefroren:
hohe beobachtete Agreement-Werte trennten nützliche von unnützen
Adaptive-Weighting-Entscheidungen nicht. Ein numerischer Agreement-Cutoff
würde hier einen nicht belegten Prädiktor vortäuschen. Die neue Policy
qualifiziert einen Kandidaten **erst nach** einem positiven unabhängigen
Qualitätstest und aktiviert ihn nicht selbst.
Die konservative Anforderung gleicher Sterne/Flächen und sauberer Ausrichtung
ist auch mit den [STScI-Hinweisen zu Drizzle-Photometrie](https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-4-characteristics-of-drizzled-data)
vereinbar. Die Zahlen selbst stammen aus der hier beschriebenen
Risikobewertung.

Der Referenz-Luminanzfehler ist der Median des absoluten Fehlers auf
auf `[0,1]` normierten, unabhängig kalibrierten Referenzpatches; die
10-%-Reduktion bezieht sich auf diesen Fehler im `rec709`-Kontrollarm.
Schatten-/Lichter-ROIs müssen vor der Auswertung feststehen und eine
nicht verschwindende Kontroll-Kontrastspanne aufweisen. Für
`set_sensor_profile_dwarf_ii` wäre eine lineare FWHM-Verbesserung das
falsche Ziel: die Profilgewichte wirken erst in HMS. Die Nullgrenzen für
lineare Metriken bedeuten deshalb Identität der linearen Artefakte, nicht
eine unmöglich präzise FWHM-Schätzung. Die 10-%-Fehlerreduktion benötigt
einen **unabhängigen** Kamerareferenzwert; ohne kalibrierte Referenz ist
dieses Ziel nicht auswertbar und der Kandidat bleibt experimentell.

## Messung an gematchten Sternen (2026-09-25)

`web_backend_cpp/scripts/matched_pair_metrics.py` setzt den Messplan der Policy für ein Paar fertiger Läufe um (nur lesend,
Ausgabe nie in einem Run-Ordner): Sterne werden **einmal** im Kontrollbild erkannt (isoliert, ungesättigt, mit Randabstand)
und an denselben Koordinaten in beiden Bildern mit demselben Momentenschätzer gemessen; Rauschen auf denselben gültigen
Hintergrundpixeln (Sterne maskiert), Signalerhalt in denselben Aperturen, Coverage je Arm und auf dem identischen Raster.
Das Sternebenen-Bootstrap-KI ist nur beschreibend; die Policy verlangt Session-KIs. Zehn Tests, Mutationsprüfung
(verschobene Messposition wird erkannt).

Retrospektive Anwendung und erste neue Session (adaptive Gewichtung an gegenüber aus, grüner Kanal, gewählter Ausgang
`drizzle_raw`; Rohdaten in `pi_jev_m5_matched_retrospective_20260925.json`). Das Werkzeug richtet die Bilder aus (Phasenkorrelation,
nur Verschiebung), passt eine robuste affine Abbildung zwischen den Sternpositionen beider Läufe an, sucht jeden Stern im Kandidaten
um die vorhergesagte Position (Radius 4 px) und misst dann mit demselben Schätzer:

| Session | Status | gematchte Sterne | FWHM-Verhältnis (Median, Stern-KI) | Elongation | Signal | Rauschen | Verschiebung Median / Max (px) | Rotation |
|---|---|---|---|---|---|---|---|---|
| M31 | retrospektiv | 2824 | 1,0021 (1,0017-1,0024) | 1,0005 | 0,999 | 1,028 | 0,0 / 4,0 | 0,001° |
| M42 | retrospektiv | 968 | 1,0012 (1,0004-1,0021) | 0,9995 | 1,002 | 1,023 | 0,0 / 4,0 | 0,000° |
| IC5070 | retrospektiv | 2236 | 1,0015 (1,0011-1,0018) | 0,9984 | 1,005 | 1,000 | 0,0 / 3,2 | 0,000° |
| M66 | retrospektiv | 334 | 1,0018 (0,9993-1,0039) | 0,9971 | 1,003 | 1,018 | 0,0 / 2,2 | -0,001° |
| IC4605 | erste Session nach dem Einfrieren; Raster unterschiedlich | 350 | 1,0050 (1,0034-1,0070) | 0,9966 | 1,044 | 0,997 | 5,7 / 13,0 | 0,238° |

Lesart: Die adaptive Gewichtung ist an gematchten Sternen in keiner der fünf Sessions schärfer (Verhältnis 1,001 bis 1,005 statt der
geforderten höchstens 0,95); bei M31 und M42 liegt das Rauschen über der Grenze von +2 %. Die vier ersten Sessions sind explorativ
(vor dem Einfrieren der Policy untersucht). IC4605 ist die erste Session nach dem Einfrieren, aber **nicht policy-konform**: die
Ausgaberaster beider Arme sind verschieden groß (2844 gegen 2850 Zeilen).

**Neuer Befund zur Vergleichbarkeit:** Bei IC4605 haben An und Aus verschiedene Referenzbilder für die Registrierung (183 gegen 189);
bei den vier älteren Paaren war das Referenzbild in beiden Armen dasselbe. Ursache im Code (`runner_phase_registration.cpp`): Unter den
qualitätsgewählten Ankerframes wird das Referenzbild über `global_weights` bewertet, und `global_weights` hängt von `adaptive_weights`
ab. Der Schalter ändert also nicht nur die Gewichtung, sondern kann über das Referenzbild die Geometrie und Abtastung des ganzen Stacks
verändern (hier 0,24° Rotation, bis 13 px Versatz). Ein An/Aus-Vergleich misst damit beide Effekte gemeinsam; die Policy sollte das
Referenzbild in beiden Armen festhalten oder die Referenzabweichung als Ausschlusskriterium führen. Ein Wiederholungslauf des
IC4605-Kontrollarms (`jev_m5_ic4605_off2_20260925`) prüft zusätzlich die Reproduzierbarkeit identisch konfigurierter Läufe.

## Für reguläre Freigabe noch erforderlich

Die Grenzwerte sind nun fixiert, und der matched-position-Metrik-Producer für die adaptive Gewichtung existiert. Für eine Freigabe
fehlen noch unabhängige Referenz-Luminanzdaten (Sensorprofil), mindestens 12 unabhängige Sessions je Kandidat und
kontrollierte Vier-Baseline-Daten (aktuelle Config, Regeln, PI-Beratung,
Regeln+Jev) mit separater Nutzerannahme, tatsächlich angewendeter Config
und Qualitätslabel. Der aktuelle Replay trennt Annahme und Draft-Hash; ein
Run-Config-Hash bleibt ohne gesicherte Run-Verknüpfung `null`. kNN bleibt
optional und getrennt. Erst unabhängige Session-Testdaten erlauben die
numerische Gate-Prüfung. Die bisherige `0/2`-Patchquote und
Agreement-Diagnostik ersetzen keines dieser Kriterien.

Neue Bildverarbeitungsläufe sind vom Nutzer gezielt erlaubt. Wiederholungen
der bereits untersuchten vier Sessions wären trotz der jetzt eingefrorenen
Schwellen kein unabhängiger Bestätigungstest. Deshalb werden bislang
ununtersuchte IC4605- und M104-Eingaben als getrennte An/Aus-Paare vorbereitet.
Ihre Ergebnisse gehören erst nach erfolgreichem Abschluss und einer Prüfung
der Paaridentität in den Evidenzbericht; ein laufender oder fehlgeschlagener
Arm gilt nicht als Qualitätsnachweis.
