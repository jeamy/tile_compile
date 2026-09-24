# PI Jev — Bestandsaufnahme der Kandidaten-Parameter

> **Stand:** 2026-09-24. **Art:** Bestandsaufnahme und Vorschlag, keine Implementierung, kein Lauf.
> **Anlass:** Der Katalog `web_backend_cpp/config/pi_decisions/candidates_v1.json` enthält nur einen echten
> Kandidaten (`enable_adaptive_weights`); Ziel ist, dass Jev alle **sinnvollen** Config-Werte vorschlagen kann
> ([Zielbild](pi_jev_decisions_plan_de.md), [Regelkatalog](pi_scan_pre_rules_de.md)).
> **Quellen:** `tile_compile_cpp/tile_compile.schema.json` (281 Blattparameter in 17 Bereichen),
> `web_backend_cpp/config/pi_decisions/protected_paths_v1.json`, das Praxis-Handbuch
> `docs/configuration_examples_practical_de.md`, der Regelkatalog (Abschnitt 4) und die gepaarten Läufe aus
> [pi_jev_m4_evaluation_m31_m42_de.md](pi_jev_m4_evaluation_m31_m42_de.md) Abschnitte 9-10.
> Nicht geprüft: Aussagen des Praxis-Handbuchs sind nicht empirisch validiert; der Inhalt einzelner Run-Artefakte
> wurde nur dort gelesen, wo unten ausdrücklich genannt.

## 1. Ergebnis in Kürze

1. **Die meisten datenabhängigen Parameter brauchen Post-Run-Evidenz**, nicht den Scan. Der Scan liefert je Frame
   Hintergrund, Rauschen, Strukturenergie, Quadrantenkontrast, FWHM/Roundness und Sternzahl plus Header und eine
   Rotationsschätzung; Registrierung, Stack-Rauschen, Multiband-Vergleich, PCC/BGE-Ergebnisse gibt es erst nach einem
   Run. Die Auswertung an vier Datensätzen zeigte zusätzlich: Scan-FWHM sagt den Lauf-FWHM nicht voraus
   (Spearman etwa 0,05), und die Metrik-Übereinstimmung sagt den Nutzen der adaptiven Gewichtung nicht voraus.
   Ein Vorschlagsmodus "nur vor dem Run" deckt deshalb nur einen kleinen Teil ab.
2. **47 von 281 Parametern sind bereits geschützt** (Kalibration, Laufzeitgrenzen, Coverage-/Validierungs-Gates,
   PCC-Grenzen, `bge.method` u. a.). Zwei davon widersprechen der Praxis-Empfehlung (siehe 4.1).
3. Vorschlag: **zwölf Kandidatengruppen**, gestaffelt nach der Evidenz, die sie brauchen (Abschnitt 5). Jede hat
   feste Wertesets und wird einzeln nach dem Erweiterungsverfahren eingeführt.
4. Fünf Entscheidungen liegen bei dir (Abschnitt 6); ohne sie kann ich den Katalog nicht sinnvoll erweitern.

## 2. Einteilung der 281 Parameter

| Klasse | Bedeutung | Beispiele |
|---|---|---|
| **S** Systemfakt / Betrieb | hängt von Rechner, Pfaden, installierten Katalogen ab; deterministisch zu lösen, nicht Jev-Sache | `calibration.*`, `runtime_limits.*`, `astrometry.astap_*`, `pcc.siril_catalog_dir`, `output.*` |
| **G** geschützt | Akzeptanzgrenzen und Invarianten, nie durch einen Kandidaten erreichbar | `reconstruction.coverage_gate.*`, `reconstruction.multiband_validation.*`, `pcc.k_max` |
| **P** Nutzerpräferenz / Stil | Ergebnis-Geschmack, keine Datenfrage; Nutzerwerte bleiben erhalten | `hypermetric_stretch.target_bg`, `convergence_power`, Denoise-Stärken |
| **D** datenabhängig | Wert sollte von den Daten abhängen; Kandidatenkandidaten | Registrierung, Drizzle, Clipping, Pyramide, Denoise-Aktivierung, BGE-Samples |
| **A** Ableitbar aus Scan-Fakten | reine Fakten ohne Ermessen | `data.color_mode`, `data.bayer_pattern` (Scan-Identifikation) |

Anzahl Blattparameter je Bereich (Schema) und geschützte darunter:

| Bereich | Blätter | geschützt | überwiegende Klasse |
|---|---|---|---|
| `reconstruction` | 56 | 20 | D (Drizzle/Clipping/Pyramide/Multiband), G (Gates) |
| `bge` | 51 | 1 (`bge.method`) | D, mit P |
| `registration` | 27 | 0 | D |
| `chroma_denoise` | 27 | 0 | D/P |
| `hypermetric_stretch` | 26 | 0 | P, D (Sensorprofil) |
| `pcc` | 22 | 3 | S/D |
| `calibration` | 18 | 18 (alles) | S |
| `luma_denoise` | 18 | 0 | D/P |
| `global_metrics` | 9 | 0 | D |
| `data` | 5 | 0 | A |
| `linearity` | 4 | 0 | D (geringes Gewicht) |
| `astrometry` | 4 | 0 | S |
| `runtime_limits` | 4 | 4 (alles) | S |
| `output` | 3 | 1 | S |
| `normalization` | 3 | 0 | D (braucht Objektkontext) |
| `dithering` | 2 | 0 | D |
| `stacking` | 2 | 0 | D |

## 3. Verfügbare Evidenz

**Vor dem Run (Scan/`scan-metrics`, Header, Systemfakten):** Frame-Metriken (siehe oben), Aufnahmegruppen nach
Kamera/Filter/Belichtung/Gain, Farbmodus und Bayer-Muster, Frame-Anzahl, geschätzte Feldrotation und Sitzungsdauer
(Schätzung, keine Messung), Nutzerangaben (`user_stated` im State, z. B. Objektklasse, wenn der Nutzer sie setzt),
Systemfakten (Master-Darks, Kataloge). Bekannte Grenzen: Der Scan misst keine Registrierbarkeit, keine
Dither-Abdeckung, kein Stack-Rauschen, keinen PCC-Erfolg.

**Nach dem Run (Artefakte im Run-Verzeichnis):** unter anderem `global_metrics.json` (Gewichte), `forward_drizzle.json`
(`validation` je Kandidat, `selected_candidate`, `selection_reason`, Clipping-Zähler), `registration_sampling.json`,
`global_registration.json`, `sampling_geometry.json`, `source_quality_plan.json`, `bge.json`, `luma_denoise.json`
(und `chroma_denoise.json` laut Handbuch). Ihre Felder sind für die Kandidaten unten noch je einzeln zu inventarisieren.

## 4. Was die Läufe bisher an Evidenz geliefert haben

- **Adaptive Gewichtung** (M31/M42/IC5070/M66, je gepaart): kein Schärfegewinn, 0-2 % mehr Rauschen. Ein
  Kandidat "adaptive Gewichtung aktivieren" ist damit derzeit unbegründet; der umgekehrte ("deaktivieren") wäre
  erst nach einem Wiederholungslauf zur Grundstreuung belastbar.
- **Multiband wird nie gewählt:** In allen acht Läufen wurde `drizzle_raw` gewählt ("multiband not promoted: median FWHM
  improvement below 0.95x raw"). Das Verhältnis Multiband/Raw des Median-FWHM lag zwischen 0,999 und 1,007
  (Schwelle 0,95). Multiband verändert die Schärfe also praktisch nicht. Das ist eine Post-Run-Aussage über die
  Multiband-Parameter, nicht über die geschützten Gates.
- **Scan-FWHM ist kein Stellvertreter** für den Lauf-FWHM (Spearman 0,05 auf M31/M42), Lauf-FWHM lag bei
  3,75 bis 5,8 px.
- Aus früheren Sessions (Memory, nicht neu geprüft): `luma_denoise.bilateral`/`extended_source_protection` halbieren
  das M42-Hintergrundrauschen, schließen die Lücke zum Vergleichsbild aber nicht; `sigma_range` bei Chroma/Luma
  ist relativ, nicht absolut, alte Configs sind veraltet; Dynamic-Boost-Pilot verfehlte das Laufzeitziel.

## 4.1 Konflikte mit der Schutzliste

- `reconstruction.clipping.shared_frame_rejection` und `bimodal_veto` sind geschützt, das Handbuch empfiehlt
  `shared_frame_rejection` aber gegen Chroma-Speckle (also datenabhängig).
- `bge.method` ist geschützt: BGE ein-/ausschalten oder zwischen `classic`/`autobge` wählen ist damit kein
  Kandidat, nur die BGE-Parameter darunter. Die Praxis-Profile setzen `classic`, das Schema `none`.
- `calibration.*` und `runtime_limits.*` sind komplett geschützt. Das passt zu Klasse S, schließt aber die
  wichtige Auswahl des passenden Master-Darks (Belichtung/Gain) als Jev-Kandidaten aus; sie bleibt deterministisch.

## 5. Vorschlag: Kandidatengruppen

Jeder Kandidat ist eine atomare Gruppe mit festem Wertesatz (Jev wählt, erfindet nichts). Reihenfolge nach nötiger
Evidenz. Die Spalte "Nachweis" nennt, was vor Freigabe gemessen werden muss (gepaarter Lauf gegen die
Ausgangsconfig, Sternform an gematchten Positionen).

### Gruppe A: aus dem Scan begründbar

| # | Kandidat | Patch (Beispiel) | Evidenz | Nachweis / Blocker |
|---|---|---|---|---|
| A1 | Sensorprofil für Stretch | `hypermetric_stretch.sensor_profile`, `fallback_profile` | exakter Kameramodell-String im Header, versionierte Tabelle, Unknown-Fall | Tabelle nötig; Regelkatalog verbietet Substring-Zuordnung; eher deterministisch als Jev |
| A2 | Registrierung bei starker Feldrotation | `registration.allow_rotation`, `star_shift_radius_px` (Handbuch: Alt/Az 200-400, Äquatorial 60) | geschätzte Rotation, Montierungsangabe vom Nutzer | Rotation nur Schätzung; Montierung ist nicht im Scan; Nutzerangabe nötig |
| A3 | Sternarme Daten | `registration.engine` = `robust_phase_ecc` | Scan-Sternzahl, aber Regelkatalog sagt: nicht allein daraus | `auto_engine` deckt Teile schon ab; Nachweis gegen Registrierungserfolg fehlt |

### Gruppe B: braucht Post-Run-Messung (Meilenstein M6 der Umsetzungsplanung)

| # | Kandidat | Patch (Beispiel) | Evidenz (Post-Run) | Nachweis / Blocker |
|---|---|---|---|---|
| B1 | Adaptive Gewichtung ausschalten | `global_metrics.adaptive_weights: false` | `global_metrics.json`, n_eff/N; Vergleich der Läufe | Grundstreuung per Wiederholungslauf messen; heute nur belegt: kein Nutzen |
| B2 | Multiband-Parameter | `reconstruction.multiband.levels`, Exponenten, `alpha_cap` | `forward_drizzle.json` Validierung Multiband gegen Raw | Gates selbst bleiben geschützt; belegt bisher nur "kein Effekt", also unklar, ob Parameter überhaupt helfen |
| B3 | Drizzle-Sampling | `pixfrac`, `internal_scale`, `output_scale` (abhängige Gruppe) | Lauf-FWHM, Dither-Abdeckung (Registrierung) | eigene Vergleichsläufe; `min_clip_contributors` bleibt geschützt |
| B4 | Clipping bei Artefakten | `clip_sigma_low/high`, `robust_passes` | Clipping-Zähler, Artefaktanteil je Run | Frame-Anzahl und Artefaktlage nötig; Schutzschalter bleiben |
| B5 | Lokale Qualitätskarten | `quality.pyramid.sharpness_weight`, `snr_weight`, `score_scale` | Heterogenität von Seeing/Rauschen im Run | eigene Vergleichsläufe |
| B6 | Denoise-Aktivierung und -Stärke | `luma_denoise.*`, `chroma_denoise.*` (Stärken, Schutzmasken, `extended_source_protection`, `large_scale_bias`) | gemessenes Stack-Rauschen, Anteil geschützter Fläche (Chroma-Artefakt) | Objektklasse (kompakt oder diffus) muss vom Nutzer kommen; Bildbeurteilung; frühere Fehlversuche |
| B7 | PCC-Einstellungen | `pcc.source`, `mag_limit`, `min_stars`, `background_model` | PCC-Artefakt (Sterne, Residual) | PCC-Gates bleiben geschützt |
| B8 | BGE-Samples | `sample_quantile`, `structure_thresh_percentile`, `fit.*` | Residuen in `bge.json`, Quadrantenkontrast | Signal-Erhaltung; `bge.method` geschützt (siehe 4.1) |
| B9 | Dither/Kosmetik | `dithering.*`, `stacking.per_frame_cosmetic_correction*` | Dither-Abdeckung, Defektbefund | Defekt-/Dither-Evidenz fehlt bisher |

Nicht als Jev-Kandidaten vorgesehen: Klasse S (Kalibration, Laufzeit, Pfade), Klasse G, `normalization.mode`
(Objektkontext fehlt), Stretch-Numerik (Nutzerpräferenz), `linearity.*` (geringes Gewicht).

## 6. Entscheidungen (getroffen am 2026-09-24)

Die fünf Fragen unten sind beantwortet und in [pi_jev_decisions_plan_de.md](pi_jev_decisions_plan_de.md) Abschnitt 1.1
festgehalten: (1) Post-Run-Beratung nur auf Wunsch des Nutzers, (2) `shared_frame_rejection`, `bimodal_veto` und `bge.method`
werden erreichbar, alle übrigen geschützten Pfade bleiben geschützt, (3) Zahlenwerte nur als Stufen eines geprüften
Gitters, (4) Objektklasse als Nutzerangabe im State, (5) nur Vergleichs- und Referenzdaten aufbewahren.
Umsetzung: M5.1 im [Umsetzungsplan](pi_jev_implementierungsplan_de.md). Ursprüngliche Fragen zur Nachvollziehbarkeit:

1. **Post-Run einbeziehen?** Gruppe B ist der größte Teil und braucht die Post-Run-Beratung samt "ab welcher Phase
   fortsetzen" (Resume-Vertrag). Ohne sie bleibt Jev auf Gruppe A begrenzt.
2. **Schutzliste:** Sollen `shared_frame_rejection`, `bimodal_veto`, `bge.method` erreichbar werden, und mit welchen
   Bedingungen? Oder bleiben sie geschützt und die Praxisempfehlung damit ohne Jev.
3. **Werte:** Feste Wertesets je Kandidat (Jev wählt) oder auch Wertvorschläge innerhalb geprüfter Grenzen
   (anderer Fragetyp, größere Änderung am Aufbau)?
4. **Objektklasse:** Darf der Nutzer die Objektklasse (kompakt, diffus, Sternfeld) als geprüfte Nutzerangabe im State
   setzen? Mehrere Kandidaten (B6, B8) hängen davon ab.
5. **Nachweisaufwand:** Jeder freigegebene Kandidat braucht gepaarte Läufe; pro Datensatz und Arm sind das volle
   Läufe mit viel Speicherbedarf (bei M66 beim Start etwa 250 GB frei). Welche Datensätze gelten als Referenz?

## 7. Risiken und Abhängigkeitsreihenfolge

- Reihenfolge: Schutzliste klären (6.2), dann Objektklasse im State (6.4), dann Gruppe A einzeln, danach die
  Post-Run-Beratung (6.1) und erst dann Gruppe B.
- Jeder Kandidat braucht Negativfixtures (Locks, geschützte Pfade, Enthaltung), Config-Validierung der
  Gesamtconfig und Evidenz, die aus demselben State aufgelöst wird (nie aus der Modellantwort).
- Risiko: mehr Kandidaten bedeuten mehr Modellentscheidungen ohne Nutzennachweis. Ohne gepaarte Läufe sind sie
  Hypothesen (wie `enable_adaptive_weights`), werden nur als experimentell im Review angeboten und nie automatisch
  übernommen.
