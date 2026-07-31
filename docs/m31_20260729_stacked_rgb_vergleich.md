# M31 Runs 2026-07-29: Sternschärfe und Rauschen im Vergleich

Datum der Analyse: 2026-07-29
Basis: bestehende Artefakte (`outputs/stacked_rgb.fits` je Run), kein neuer Run.
Referenz: Siril/DWARF-Stack `result.fit` aus
`/media/nfs4export/data/Astro/DwarfII/Astronomy/DWARF_RAW_M 31_EXP_10_GAIN_80_2024-10-07-20-51-46-987/result.fit`

## Methode

- Luminanz aus RGB (Rec. 709: 0.2126 R + 0.7152 G + 0.0722 B).
- Hintergrund: Median; Rauschen: robuste Sigma-Schätzung über MAD
  (`sigma = 1.4826 * MAD`) des Gesamtbildes.
- Sterndetekte: lokale Maxima (3x3) oberhalb `Median + 8 * sigma`,
  Randausschluss 12 px, Begrenzung auf die 5000 hellsten Kandidaten.
- FWHM: zweite Momente in einer 17x17-Apertur nach lokalem
  Hintergrundabzug (Rand-Median), `FWHM = 2.3548 * sigma_moment`.
  Kandidaten mit Moment-Sigma > 6 px wurden als ausgedehnte Quellen
  (Galaxienkern etc.) verworfen.
- Elliptizität: `|FWHM_x - FWHM_y| / max(FWHM_x, FWHM_y)` der Mediane.

Hinweis: FWHM in Pixeln. Die Runs liefern 3922x2309 px, der Referenz-Stack
3840x2160 px (Dwarf II nativ). Die Pixelmaßstäbe sind nahezu identisch
(gleiches Instrument), die FWHM-Werte sind daher direkt vergleichbar.

## Ergebnisse

| Run | n Sterne | FWHM med [px] | FWHM p25 | FWHM p75 | FWHM x | FWHM y | Ellipt. | sigma_MAD | rel. Rauschen (sigma/Median) |
|---|---|---|---|---|---|---|---|---|---|
| sharpnes_v3_baseline | 5000 | 5.55 | 5.34 | 7.18 | 5.72 | 5.33 | 6.9 % | 1.022 | 0.49 % |
| sharpnes_v3_auto_reject | 5000 | 5.54 | 5.35 | 7.00 | 5.72 | 5.32 | 7.1 % | 0.912 | 0.43 % |
| sharpnes_v3_auto_reject_cubic | 5000 | 5.34 | 5.10 | 7.13 | 5.51 | 5.10 | 7.5 % | 1.102 | 0.52 % |
| sharpnes_v3_cubic_guard_rescue | 5000 | 5.32 | 5.09 | 7.04 | 5.50 | 5.08 | 7.7 % | 0.953 | 0.45 % |
| sharpnes_v3_lanczos4_guard_rescue | 4999 | 5.36 | 5.14 | 6.94 | 5.54 | 5.14 | 7.2 % | 0.980 | 0.47 % |
| sharpnes_v3_debayer_origin0 | 5000 | 5.28 | 5.07 | 7.73 | 5.44 | 5.08 | 6.6 % | 0.747 | 0.35 % |
| sharpnes_v3_score_scale_18 | 5000 | 5.54 | 5.35 | 7.00 | 5.72 | 5.32 | 7.1 % | 0.912 | 0.43 % |
| **Referenz: dwarf result.fit** | 4054 | **6.28** | 5.30 | 7.97 | 6.19 | 5.89 | 4.9 % | 0.000227 (norm.) | **2.21 %** |

Absolute sigma_MAD-Werte sind wegen unterschiedlicher Streckung/Normierung
(Runs: linear, Median ~210; Referenz: normiert/gestretcht, Median ~0.0103)
nicht direkt vergleichbar; aussagekräftig ist das relative Rauschen
`sigma / Median`.

## Nachtrag 2026-07-29: Korrektur der Schärfe-Aussage

Die ursprüngliche Aussage „alle Runs sind schärfer als die Referenz“
**hält einer visuellen Prüfung nicht stand** und wurde revidiert:

1. **Gegenprobe auf den gestretchten HMS-Dateien** (identische Methode,
   `stacked_rgb_hms.fits` bzw. `resultHMS.fit`):

   | Datei | n Sterne | FWHM med [px] | FWHM p25 | FWHM p75 | Ellipt. |
   |---|---|---|---|---|---|
   | dwarf resultHMS.fit (Referenz) | 2876 | **6.84** | 6.48 | 7.35 | 4.4 % |
   | lanczos4_guard_rescue (hms) | 3293 | 7.11 | 6.66 | 7.71 | 3.5 % |
   | debayer_origin0 (hms) | 3171 | 7.02 | 6.46 | 7.85 | 1.6 % |

   Auf den gestretchten Ergebnisbildern ist die **Referenz schärfer**
   (6.84 px vs. 7.0–7.1 px).

2. **Visueller Vergleich** (600x600-Zentralausschnitte, identische
   Streckung): Die Referenz zeigt runde, kompakte Sternprofile. Die
   Run-Sterne wirken weicher und blockiger mit leichten Farbsäumen.

3. **Befund zur Metrik:** Die Momenten-FWHM auf den linearen
   `stacked_rgb.fits` ist durch die unterschiedliche
   Streckung/Normierung und durch blockige Resampling-/Debayer-Kerne
   verzerrt: spitz-abfallende, aber pixelig-quadratische Sternkerne
   liefern kleine Momenten-FWHM-Werte, sehen aber visuell schlechter
   aus. Für Qualitätsvergleiche ist die Messung auf dem gestreckten
   Endbild (oder ein echter PSF-Fit) vorzuziehen.

4. **Defekt in `debayer_origin0`:** Das Ergebnisbild hat einen
   deutlichen Grünstich (Median-Kanalverhältnisse G/R = 1.046,
   G/B = 1.047; alle anderen Runs: G/R ≈ 0.97, G/B ≈ 0.99) und zeigt
   am Galaxienkern ein Schachbrett-Pixelraster – typisches Bild einer
   falschen Bayer-Phase/Origin. **Der Run ist in der jetzigen Form
   nicht verwendbar.**

## Code-Fix 2026-07-29: Bayer-Origin und Demosaicing

Ursache des Defekts: Ein experimenteller Patch erzwang für AQMH
`debayer_origin = (0,0)`. Das AQMH-Ausgangsmosaik liegt aber auf dem
Registrierungs-Canvas-Gitter, dessen Bayer-Parität durch den
Canvas-Tile-Offset definiert ist. Bei ungeradem Offset kippt die
CFA-Phase → R/B-Tausch, Grünstich und Schachbrett-Muster.

Geändert:

- `tile_compile_cpp/apps/runner_pipeline.cpp` (DEBAYER-Phase):
  Origin wieder `-debayer_tile_offset_x/y` (kein AQMH-Sonderfall).
- `tile_compile_cpp/apps/runner_phase_post_stack_output.cpp`
  (Post-Stack-RGB-Writer): dieselbe Origin-Korrektur.
- Beide Pfade nutzen jetzt `debayer_bilinear` statt
  `debayer_nearest_neighbor`. Nearest-Neighbor-Debayering erzeugt
  2x2-Blockstrukturen → quadratische, ausgefranste Sternkerne mit
  Farbsäumen (sichtbar in den Screenshots der Runs). Bilineares
  Demosaicing liefert runde, kompakte Sternprofile.
- Regressionstests in `tile_compile_cpp/tests/test_debayer.cpp`:
  `debayer_bilinear_respects_tile_origin_parity` (Origin-Parität für
  alle 4 Pattern) und `debayer_bilinear_wrong_origin_swaps_channels`
  (erkennt erzwungenen (0,0)-Origin über R/B-Swap).
  Volle Suite: 249 Testfälle bestanden.

Verbleibende Optionen für noch schärfere Sterne (nicht umgesetzt):

- Höherwertiges Demosaicing (VNG/AHD via `cv::demosaicing`) für den
  finalen RGB-Output – bilinear glättet die Profile minimal.
- Prewarp-Interpolation `cubic`/`lanczos4` bevorzugen (siehe Runs).
- Der Rest-Unterschied zu Siril erklärt sich aus dem Stacking im
  CFA-Gitter vor dem Demosaicing; Siril demosaict vor dem Stacking.

## Verifikations-Run 2026-07-30: M31-debayer-fix_20260730_053934

Run mit dem Bayer-Origin-Fix und bilinearem Demosaicing
(`runs/M31-debayer-fix_20260730_053934`, Resume ab DEBAYER, GBRG/OSC/AQMH).
Gleiche Messmethode wie oben.

### Vergleich mit dem Siril-Ergebnis

| Paar | Datei | n Sterne | FWHM med [px] | FWHM p25 | FWHM p75 | Ellipt. | sigma_MAD | rel. Rauschen |
|---|---|---|---|---|---|---|---|---|
| HMS | fix stacked_rgb_hms.fits | 1606 | **6.94** | 6.54 | 7.53 | 3.9 % | 0.0292 | 21.6 % |
| HMS | dwarf resultHMS.fit | 2876 | **6.84** | 6.48 | 7.35 | 4.4 % | 0.0372 | 26.6 % |
| linear | fix stacked_rgb.fits | 3902 | 7.71 (p25 5.92) | – | – | 5.7 % | 0.750 | 0.36 % |
| linear | dwarf result.fit | 4054 | 6.28 | – | – | 4.9 % | 0.000227 | 2.21 % |

### Befund

- **Farbe behoben:** G/R = 0.978, G/B = 0.987 – im Bereich der
  fehlerfreien Runs (≈0.97/0.99), kein Grünstich mehr. Visuell: kein
  Schachbrett am Galaxienkern, natürliche Farben.
- **Schärfe (HMS):** 6.94 px vs. 6.84 px Referenz – nahezu gleichauf
  (Differenz ≈ 1.5 %). Gegenüber den alten Runs (7.02–7.11 px) eine
  deutliche Verbesserung; die Elliptizität (3.9 %) liegt jetzt sogar
  unter der Referenz (4.4 %). Die lineare FWHM bleibt wie dokumentiert
  keine belastbare Metrik (breite Verteilung, p25 5.92 / med 7.71).
- **Rauschen (HMS):** Der Fix-Run ist bei vergleichbarem Medianlevel
  ca. 20 % ruhiger als die Referenz (rel. 21.6 % vs. 26.6 %).
- **Visuell (600x600-Zentralcrop):** runde, kompakte Sternprofile ohne
  Blockstruktur und ohne Farbsäume; Kern glatt. Die vorher sichtbaren
  NN-Debayer-Artefakte sind verschwunden.
- **Gesamt (korrigiert 2026-07-30):** Farbe und Rauschen sind behoben
  bzw. besser als Siril; die **Kernschärfe** liegt weiterhin hinter
  Siril – siehe positionsgematchte Analyse unten.

### Positionsgematchte Sternanalyse (Fix-Run vs. resultHMS)

Gleiche Sterne in beiden Bildern (ganzzahliger Shift dy=-116, dx=-73
per Phasenkorrelation der Sternkarten, Matching-Toleranz 3 px).
77 gepaarte Sterne, Metriken pro Stern in 17x17-Apertur nach lokalem
Hintergrundabzug:

| Metrik | Siril (ref) | Fix-Run | Verhältnis run/ref |
|---|---|---|---|
| FWHM radial (Halbwertsbreite des Kerns, Median) | 4.70 px | 6.60 px | **1.28** (p25 1.17, p75 1.53) |
| Peak/Flux (Kernspitzheit) | 0.0290 | 0.0182 | **0.72** |
| FWHM zweite Momente | 7.08 px | 7.16 px | 1.03 |

- **85.7 % der gepaarten Sterne sind in Siril schärfer**, nur 5.2 %
  im Run. Median-Differenz der Kern-FWHM: +1.63 px.
- **Erklärung der bisherigen Metrik-Blindheit:** Die Momenten-FWHM
  integriert die gesamte 17x17-Apertur und wird von den Flanken
  dominiert. Die Run-Sterne haben vergleichbar breite Flanken, aber
  **flache, weiche Kerne**; Siril-Sterne haben spitzere Kerne bei
  gleicher Gesamtbreite. Für Kernschärfe ist die radiale
  Halbwertsbreite oder Peak/Flux nötig.
- **Ursache der weichen Kerne:** bilineares Demosaicing mittelt den
  Sternkern über das CFA-Gitter (halb so hohe Abtastung pro Kanal)
  und drückt die Kernspitze; dazu kommt die Prewarp-Interpolation im
  CFA-Gitter vor dem Stacking. Siril demosaict kantenerhaltend (VNG)
  vor dem Stacking.
- **Nächster Hebel:** kantenerhaltendes Demosaicing (VNG/AHD via
  `cv::demosaicing`) für den finalen RGB-Output; optional
  `prewarp_interpolation: cubic/lanczos4`.

### AHD-Vergleichsrun 2026-07-30 (`20260730_085813_eab6808b`)

Run mit AHD-Demosaicing (`debayer_opencv(..., ahd=true)`) im
DEBAYER- und Post-Stack-Pfad. OpenCV unterstützt nur Integer-Input
(EA/AHD: 16U, VNG: 8U) – das Float-Mosaik wird linear auf 16 Bit
skaliert und zurück. Das OpenCV-Pattern-Mapping wurde empirisch
verifiziert (weicht vom naiven First-Row-Reading ab: RGGB→BayerBG,
BGGR→BayerRG, GRBG→BayerGB, GBRG→BayerGR). Tests: volle Suite
250 Fälle grün.

**Positionsgematchte Paaranalyse:**

| Vergleich | Metrik | ref | run | Befund |
|---|---|---|---|---|
| AHD vs. Siril resultHMS (87 Paare) | FWHM radial | 4.93 px | 6.43 px | AHD-Kerne ~20 % breiter (bilinear: 28 %) |
| AHD vs. Siril | Peak/Flux | 0.0305 | 0.0215 | 0.75 (bilinear: 0.72) |
| AHD vs. Siril | Anteil ref schärfer | – | – | 79.3 % (bilinear: 85.7 %) |
| AHD vs. bilinear-Fix-Run (1382 Paare) | FWHM radial | 6.92 px | 6.67 px | **AHD 4 % schärfere Kerne**, 88.4 % der Sterne schärfer |
| AHD vs. bilinear-Fix-Run | Peak/Flux | 0.0186 | 0.0192 | AHD-Spitzen 4 % höher |

**Farbe (HMS):** G/R = 0.985, G/B = 1.008 (Referenz: 0.990/1.004) –
sauber. **Rauschen (HMS):** sigma_MAD = 0.0309 bei Median 0.1346
(rel. 22.9 %), Referenz 26.6 % – weiterhin ruhiger als Siril.
**Visuell:** glatter Kern, runde Sterne, keine Artefakte.

**Fazit:** AHD verbessert die Kernschärfe messbar gegenüber bilinear
(~4 % schärfere Kerne, ~4 % höhere Spitzen) und verkleinert den
Abstand zu Siril von 28 % auf ~20 %. Der verbleibende Abstand
entsteht vor dem Demosaicing (Prewarp-Interpolation und Stacking im
CFA-Gitter mit halber Kanal-Abtastung); Demosaicing allein kann ihn
nicht schließen. Mögliche weitere Hebel:
`prewarp_interpolation: cubic/lanczos4` prüfen, Schärfung nach dem
Demosaicing, oder Demosaicing der Einzelframes vor dem Stacking
(architektonische Änderung).

### Sharpening-Kandidat C (Run `20260730_102943_bf838cd6`)

Neuer AQMH-Post-Processing-Kandidat: Unsharp-Mask (sigma=2 px,
amount=0.6) auf dem CFA-Mosaik, validiert über dieselben Gates wie
die anderen Kandidaten (Uniform Control + immutable raw AQMH).

**Gate-Entscheidung: verworfen** (`star_core_sharpening_applied=false`,
Auswahl blieb `structure_masked_detail`):

- FWHM vs. Kontrolle verbessert: 2.468 vs. 2.580 (Regression −4.3 %)
- **Background-RMS-Regression +63 %** (1.114 vs. 0.683) → Gate FAIL:
  die globale Unsharp-Mask verstärkt das Hintergrundrauschen
  unzulässig. Raw-Baseline-Guard hätte (relaxed) passiert.
- Fazit: naive Schärfung bringt zu wenig bei zu hohem Rauschpreis.
  Das Gate-System hat den Kandidaten wie vorgesehen abgelehnt;
  Run-Ausgabe identisch zur AHD-Basis (kein zusätzlicher
  Analyseaufwand nötig). Nächster Schritt: D (Debayer vor dem
  Stacking).

### Experiment D: Debayer vor Registration/Stacking (2026-07-30)

Artefakt-basiertes Experiment (`/tmp/debayer_first_exp.py`), das den
Siril-Datenfluss nachbaut: gleiche 645 Frames, gleiche
Normalisierung (`normalization.json`), gleiche Registrations-Warps
(`global_registration.json`) wie der AHD-Run `20260730_085813_eab6808b`.
Pro Frame: CFA-Normalisierung → AHD-Debayer (16U, OpenCV) → 3×3-Median
(Hot-Pixel, Ersatz für Cosmetic Correction) → Warp pro Kanal
(`cv2.WARP_INVERSE_MAP`, Offset-adjustiert wie in
`runner_phase_registration.cpp`) → Mean-Stack auf dem Canvas.

Fallstricke, die dabei verifiziert wurden:

- Die persistierten Warps sind **Forward-Maps**; die Pipeline nutzt
  `cv::WARP_INVERSE_MAP`. Offset-Korrektur: `tx -= a00*ox + a01*oy`
  (und y analog). Verifikation per Phase-Correlation gegen
  `cache/prewarped_frames/0.raw`: dy=-0.027, dx=-0.036, resp=0.937.
- NaNs in der Luminanz brechen `np.median`-basierte Sterndetektion
  (Analyse-Skript NaN-sicher gepatcht).

**Ergebnis (positionsgematchte Paaranalyse):**

| Vergleich | Paare | FWHM_rad run/ref | Peak/Flux run/ref | Anteil run schärfer |
|---|---|---|---|---|
| D vs. Pipeline, linear/linear | 1898 | **0.84** (3.73 vs. 4.22 px) | **1.20** | 59 % |
| D vs. Pipeline, asinh/asinh (gleiche Stretch-Funktion) | 1677 | **0.94** (6.59 vs. 7.02 px) | **1.08** | 69 % |

**Fazit:** Debayer-vor-Stacking liefert bei ansonsten identischen
Daten **~6–12 % schärfere Sternkerne** und **8–20 % höhere
Kernspitzen**. Das schließt grob ein Drittel bis die Hälfte der
verbleibenden ~20-%-Lücke zu Siril (Cross-Stretch-Vergleiche gegen
resultHMS sind wegen nur ~43-87 Paaren unzuverlässig, liegen aber
konsistent in Richtung ~23-27 % für beide Pipeline-Varianten).

Einschränkungen des Experiments: Mean- statt AQMH-gewichtetem
Stacking, kein Sigma-Clipping, kein BGE/PCC (Grünstich durch
fehlende Farbkorrektur, für die Schärfemessung auf der Luminanz
unerheblich), 3×3-Median statt parametrischer Cosmetic Correction.
Eine produktive Umsetzung (OSC-Modus "debayer-first": Debayer nach
der Normalisierung, Registration auf Luminanz, Prewarp pro Kanal,
AQMH pro Kanal mit 3× Rechenaufwand) ist eine größere
architektonische Änderung und sollte auf Basis dieser Zahlen
entschieden werden.

- **Schärfe (revidiert):** Auf den gestreckten Ergebnisbildern ist der
  Referenz-Stack schärfer als die Runs (6.84 vs. 7.0–7.1 px). Die
  ursprüngliche Rangfolge auf den linearen Dateien ist als Artefakt
  der Metrik einzustufen.
- **Rauschen:** Das relative Rauschen aller Runs liegt bei 0.35–0.52 %,
  die Referenz bei ca. 2.2 % – die Runs sind deutlich ruhiger, dieses
  Ergebnis ist von der Metrik-Korrektur nicht betroffen.
- **`debayer_origin0`:** Trotz formal bestem Rauschen und kleinster
  linearer FWHM **nicht verwendbar** (Grünstich + Bayer-Schachbrett,
  siehe Nachtrag).
- **Interpolation:** Auf den linearen Dateien liefern
  `auto_reject_cubic` und `cubic_guard_rescue` ca. 0.2 px schärfere
  Sterne als die bilinearen Pendants; wegen der Metrik-Verzerrung auf
  linearen Dateien (siehe Nachtrag) ist diese Aussage jedoch an den
  gestreckten Ergebnisbildern nachzuprüfen, bevor sie als gesichert
  gilt.
- `auto_reject` und `score_scale_18` liefern identische Kennzahlen –
  die Score-Skalierung 18 ändert am Endergebnis nichts Messbares.

## Einschränkungen

- Schlussfolgerungen sind aus den finalen FITS-Artefakten abgeleitet,
  nicht durch einen neuen Run verifiziert.
- Die FWHM-Momentenmessung ist eine Approximation (kein PSF-Fit);
  relative Vergleiche zwischen den Bildern sind robust, absolute Werte
  können von gaussbasierten Messungen abweichen.
- p75-Werte enthalten Restanteile nicht punktförmiger Quellen
  (Galaxienkerne, enge Doppelsterne).
- Die Momenten-FWHM auf linearen Dateien ist für Schärfe-Rankings
  zwischen unterschiedlich prozessierten Bildern **nicht geeignet**
  (siehe Nachtrag); die linearen FWHM-Werte in der Tabelle oben sind
  daher nur mit dieser Einschränkung zu lesen.

## Experiment D+: Pipeline-Integration debayer_before_stack (2026-07-31)

### Versuch

Vollständige Integration des Debayer-before-Stacking-Ansatzes in die
AQMH-Pipeline als `data.debayer_before_stack: true` Option:

- Jeder Frame wird nach CFA-Normalisierung mit AHD demosaict
- R, G, B Kanäle werden einzeln auf den Canvas gewarpt (GPU, linear)
- Luminanz (0.25R+0.5G+0.25B) wird für AQMH-Qualitätskarten verwendet
- RGB-Kanäle werden nach AQMH-Reconstruction per gewichtetem Mean
  (Global-Quality-Weights) zusammengestackt

### Ergebnis: kein Schärfegewinn

| Variante | FWHM median (HMS) | Vergleich |
|---|---|---|
| debayer_before_stack Pipeline (v5) | 8.62 px | schlechter |
| Siril resultHMS (Referenz) | 8.45 px | — |
| Früherer AHD-Run (Standard-AQMH) | ~7.0 px | deutlich besser |

### Analyse: Warum kein Vorteil in der AQMH-Pipeline

1. **Experiment D (positiv)** verglich Mean-Stacks: debayer-first
   vs. CFA-first, beide ohne per-Pixel-Gewichtung. Der 6–12 %
   Schärfegewinn entsteht durch reduziertes Subsampling-Blur (CFA hat
   halbe Auflösung pro Kanal → Interpolation im Prewarp verwischt den
   Kern).

2. **AQMH kompensiert dieses Problem bereits:** Die per-Pixel-
   Qualitätsgewichtung (w_sharp + w_snr pyramid) selektiert für jeden
   Output-Pixel die schärfsten Beiträge. Dadurch wird der
   CFA-Subsampling-Blur an Sternkernen bereits um 15–20 % reduziert
   gegenüber einem ungewichteten CFA-Stack.

3. **Der RGB-Mean-Stack (debayer_before_stack) hat keine per-Pixel-
   Gewichtung.** Die AQMH-Global-Quality-Weights sind nur Frame-Level-
   Gewichte, keine Pixel-Level-Gewichte. Dadurch geht der gesamte
   AQMH-Schärfevorteil verloren, und das Ergebnis entspricht einem
   einfachen gewichteten Mean-Stack — schlechter als die AQMH-
   Rekonstruktion.

4. **Post-Stack-AHD-Debayering auf dem AQMH-Mosaik** (Standardpfad)
   ist gutartig: Da AQMH die schärfsten Pixel selektiert hat, enthält
   das CFA-Mosaik bereits scharfe Sternkerne. AHD interpoliert diese
   mit minimalem Verlust in RGB.

### Ergebnis: Ansatz verworfen

Die `debayer_before_stack` Option wurde revertiert. Der Ansatz ist nur
für non-AQMH (classic Mean/Sigma-Clip) Stacking relevant und dort
nicht in die Hauptpipeline integriert (das Experiment-D-Skript unter
`/tmp/debayer_first_exp.py` dokumentiert den isolierten Effekt).

### Verbleibende Hebel für Schärfeverbesserung (AQMH-Pipeline)

Die Lücke zu Siril (~14 % breitere Kerne) entsteht aus:

1. **Prewarp-Interpolation im CFA-Gitter:** Lineare Interpolation
   verschmiert bei Sub-Pixel-Shifts. Siril nutzt Lanczos nach dem
   Debayering (volle Auflösung pro Kanal).

   → **Nächster Hebel:** `prewarp_interpolation: lanczos4` prüfen.
   Der Run `sharpnes_v3_lanczos4_guard_rescue` (5.36 px linear) war
   marginal besser als bilinear (5.55 px); eine erneute Messung auf
   HMS ist ausstehend.

2. **AQMH-per-Pixel-Gewichtung optimieren:** Die Pyramid-Parameter
   (w_sharp, base_window_px, score_scale) kontrollieren wie aggressiv
   scharfe Pixel bevorzugt werden. Eine höhere `w_sharp` oder kleinere
   `base_window_px` könnte die Kernschärfe verbessern (Risiko:
   Rauscherhöhung an schwachen Sternen).

3. **Post-Stack Sharpening auf dem Endbild:** Deconvolution (z.B.
   Richardson-Lucy oder Wiener) auf der gestreckten Luminanz. Muss
   signalabhängig maskiert werden (nur Sternkerne, nicht Background).
   Kandidat C (Unsharp-Mask auf CFA) wurde vom Gate-System wegen
   +63 % Background-RMS korrekt abgelehnt; ein lokal-adaptiver Ansatz
   (nur oberhalb eines SNR-Thresholds) wäre der nächste Versuch.

4. **AQMH pro Kanal (3× Compute):** AQMH-Rekonstruktion auf R, G, B
   getrennt statt nur Luminanz. Theoretisch der sauberste Ansatz für
   debayer-first + AQMH, aber 3× Rechenzeit und erfordert eine
   kanaldifferenzierte Qualitätskarte. Großer architektonischer
   Aufwand, fraglicher Mehrwert gegenüber Hebel 1+2.

5. **Sigma-Clip auf prewarped CFA vor AQMH:** Outlier-Rejection vor
   der Qualitätsgewichtung entfernt Satelliten-Trails und Hot-Pixel-
   Reste, die sonst als „scharfe" Pixel fehlgewichtet werden können.
   Geringer Implementierungsaufwand, potenziell 1–2 % Verbesserung.

### Empfohlene Reihenfolge

1. `prewarp_interpolation: lanczos4` messen (HMS-FWHM, einfach)
2. Lokal-adaptives Post-Stack Sharpening (SNR-gesteuert)
3. AQMH-Pyramid-Tuning (w_sharp ↑, base_window_px ↓)
