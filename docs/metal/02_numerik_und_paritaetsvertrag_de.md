# 02 - Numerik und Paritätsvertrag

## 1. Problem

Der v2-Numerikvertrag (Gate 4, `numerics_decision`) legt fest:

| Größe | Entscheidung |
| --- | --- |
| alle per-Pixel-Akkumulatoren (A/B/B2, B_c/S_c/C_c) | FP64; FP32 verworfen (Rel.-Fehler bis 0.996 bei Auslöschung), Neumaier-FP32 verworfen (1.78e-10 bei 1e5-Strom gegen 1e-12) |
| Affine Geometrie, Polygonüberlapp, Robust-Clip-Reservoir, Skalen-Fold | FP64 |
| CPU<->GPU-Toleranz | Fold 1e-12, Konfidenz 1e-12 |

Metal Shading Language kennt kein `double`. Der Vertrag ist so nicht
erfüllbar; er muss für Metal neu, aber nachweisbar definiert werden.

Weitere Sprachunterschiede mit Numerik-Folgen:

- Kein `atomicAdd(double*)`; 64-Bit-Atomics nur eingeschränkt (Add fehlt).
- Kein Compilerschalter wie `--fmad=false`; Kontraktion wird über
  `fma()`-Nutzung und `mathMode = safe` (kein Fast-Math) kontrolliert.
- `pow`, `sqrt`, `exp` sind in FP32 mit begrenzter Genauigkeit; nur
  `precise::`-Varianten verwenden.

## 2. Präzisionsstufen (Kandidaten)

| Stufe | Geometrie (Eckpunkt-Transformation) | Polygonfläche | Framelokale Ebenen (fa/fbs/fbg/fs2/fq*) | Stromakkumulatoren (accA/B/B2, conf*, V2FullAcc) | Reservoir/Clip |
| --- | --- | --- | --- | --- | --- |
| T0 | FP32 | FP32 | FP32 | FP32 | FP32 |
| T1 | Double-Float (DF) | FP32 relativ zur Zelle | FP32 | DF | DF (Vergleiche) / FP32 (Gewichte) |
| T2 | DF | DF | DF | DF | DF |

Double-Float = Paar (hi, lo) zweier FP32-Werte mit
TwoSum/TwoProd (`fma`-basiert), ca. 44-48 Bit Mantisse. Speicherbedarf ist
gleich zu FP64 (8 Byte pro Wert), der Speicherplan (Gate 5) bleibt gültig.

Begründung der Stufe T1 als Startannahme:

- Absolute interne Koordinaten erreichen einige tausend Pixel; FP32-Abstand
  dort ca. 1e-3 px. Eine FP32-Transformation verfälscht Fläche und
  Zellzuordnung an Kanten. Deshalb wird die Transformation
  `q = (a0*px + a1*py + a2 - band_ox) * sc` in DF gerechnet und danach in
  **zellrelative** FP32-Koordinaten überführt (Werte um [-1, 2], Auflösung
  ca. 1e-7 px).
- Polygonfläche und Zellüberlapp arbeiten zellrelativ in FP32.
- Framelokale Ebenen erhalten pro Zelle nur wenige positive Beiträge; dort
  ist FP32 plausibel. Langstrom-Akkumulation (hunderte bis tausende
  Frames, Auslöschungsfälle) bleibt DF.
- Das ist eine **Hypothese**. Die Auswahl der Stufe erfolgt in MP0 per
  Messung (Abschnitt 4); T2 ist der Rückfall, wenn T1 den Vertrag verfehlt.

## 3. Reihenfolge und Atomics im Framescatter

CUDA `k_scatter_v2` verteilt jede Source-Sample per `atomicAdd(double)` in
framelokale Ebenen; die Additionsreihenfolge ist dort bereits nicht
deterministisch und gilt trotzdem als 1e-12-parität. Für Metal drei
Varianten (Entscheidung in MP0):

| Variante | Prinzip | Vor-/Nachteil |
| --- | --- | --- |
| S1 | `atomic<float>` fetch_add (MSL 3, Familien-Verfügbarkeit **zu verifizieren**) auf FP32-Ebenen | einfach; nichtdeterministische Reihenfolge, FP32-Genauigkeit der Zellsummen |
| S2 | Ziel-Gather: Thread pro Zielzelle, kanonische Reihenfolge (sy, sx) | deterministisch, DF/FP32 frei wählbar; mehr Source-Scans (Gate-1-Analyse: ca. 497 Mio. Samples) |
| S3 | Tile-Binning: Samples nach Zielkachel sortieren, Kachel-Thread akkumuliert lokal | deterministisch, weniger Redundanz; zusätzlicher Sortier-/Bin-Pass |

Zähler (`positive_overlaps`, `discarded`, `degraded`): 32-Bit-Atomics je
Dispatch, Host addiert in 64 Bit (je Frame/Band bleibt jede Teilsumme
unterhalb 2^32; per `static_assert`/Laufzeitprüfung abgesichert).

## 4. Emulationsharness (auf Linux ausführbar)

`metal/emulation/`: C++-Nachbildung von `df_math.h` und der Stufen T0/T1/T2
für die Kernbausteine, ausgeführt gegen die FP64-Referenz
(`forward_drizzle_v2_cpu`):

1. Polygonfläche (`d_polygon_rect_area`), Affine-Leaf-Ecken, Local-Warp-
   Inversion (fester Punkt `q_{n+1} = u - d(q_n)`) und adaptive Subdivision.
2. Fold + Stromakkumulation über die Gate-4-Sequenzen (`wide_weights`,
   `long_stream_1e5`, `cancellation`, ...) und die 21 Adversarial-Fälle.
3. Robust-Clip-Reservoir (Vergleiche, Median/MAD) und Konfidenz-Formel.
4. Ende-zu-Ende: vollständige Band-Rekonstruktion über
   `forward_drizzle_v2_cpu` mit ausgetauschten Bausteinen auf realen
   Fenster-/Warp-Daten aus vorhandenen Run-Artefakten (nur lesen, keine
   Artefakte im Run verändern; Eingaben in ein eigenes Arbeitsverzeichnis
   außerhalb von `runs/` kopieren oder nur lesend öffnen).

Gemessene Größen je Stufe:

| Größe | Bedeutung |
| --- | --- |
| relativer Fehler von A, B, B2, value, n_eff, Konfidenz | Vergleich mit Gate-4-Grenzen |
| Flächenfehler pro Überlapp (absolut, relativ) | Zellzuordnung |
| **Entscheidungs-Kipprate** | Anteil der Pixel/Kanäle, in denen Support-Maske, Clip-Akzeptanz, Reservoir-Auswahl, `robust_state`, `confidence_state`, Bimodal-Veto oder SFR-Votum von der Referenz abweicht |
| Endbild-Abweichung | Maximal-/RMS-Differenz der fusionierten Ausgabe, sowie Differenz der Auswahl-Gates (raw/uniform/multiband) |

## 5. Vertragsentwurf (Grenzen werden aus MP0 gesetzt)

Der Vertrag wird nach MP0 als JSON im Stil der Gate-Dokumente
(`docs/forward_drizzle_v2_gates/`) eingefroren. Struktur:

- **Tier A (strikt, bitgleich erwartet):** Ganzzahlige Ausgaben, die nicht
  von Gleitkommaschwellen abhängen: Frame-Skip-Bookkeeping, Zähler der
  Stream-Position, Reservoir-Hash-Auswahl (`splitmix64`, reine
  Ganzzahlarithmetik, in MSL mit `ulong` exakt).
- **Tier B (Toleranz):** Akkumulatoren, value, b, n_eff, Konfidenz:
  relative Grenze aus MP0 (Ziel: so nahe an 1e-12 wie DF erlaubt;
  Abweichung vom CUDA-Wert wird begründet dokumentiert, nicht stillschweigend
  gelockert).
- **Tier C (statistisch):** Support-Masken, Clip-/Veto-Entscheidungen,
  SFR-Voten: Kipprate-Obergrenze je Ausgabe; Kipper müssen an
  numerisch grenzwertigen Stellen liegen (Nachweis: Abstand zur Schwelle
  in Einheiten der Rechenungenauigkeit), nicht flächig.
- **Tier D (Endergebnis):** Auswahlentscheidungen raw/uniform/multiband
  identisch zur CPU-Referenz auf den Testdatensätzen; Endbild-Differenz
  unter einer aus Rauschen abgeleiteten Grenze (Bruchteil der
  Pixelrauschens, nicht nur FWHM).

Erfüllt keine Stufe T0-T2 Tier B/C/D, ist das ein Stopp-Kriterium: der
FD-v2-Kernel läuft dann nicht auf Metal (Rest bleibt möglich, siehe
`04`, MP3 als eigenständiger Nutzen), statt den Vertrag zu lockern.

## 6. Bildoperationen (Prewarp, Box-Filter)

- `warpAffine`: FP32-nativ, keine FP64-Frage. Referenz ist die vorhandene
  CPU- bzw. `opencv_cuda`-Semantik (`acceleration.cpp`); Vertrag ist
  Toleranz je Pixel (bereits heute CPU vs. CUDA nicht bitgleich), plus
  exakte Gleichheit von `valid_mask` und `has_data` an Randfällen, soweit
  die CPU-Semantik sie deterministisch festlegt.
- Source-Quality-Box-Filter (Summen, Quadratsummen, Zähler): Fensteraddition
  in FP32 kann bei großen Fenstern driften; Summen mit DF oder
  Kahan-Kompensation, Vergleich gegen CPU-Referenz mit dokumentierter
  Toleranz. Beachten: heutige Definition bei `cv::cuda::createBoxFilter`
  (Rand, Normierung) vor dem Port aus `source_quality_map.cpp` übernehmen.

## 7. Plattformabhängigkeit der CPU-Referenz

Auf arm64/AppleClang ist FMA Standard und `-ffp-contract=on` Default. Die
CMake-Liste der "Plan 19.6"-Referenzdateien (`set_source_files_properties`
mit `-ffp-contract=off`) muss auf macOS ebenfalls greifen und wird durch einen
Konfigurationstest (Flag in den Compileraufrufen vorhanden) abgesichert.
`pow/exp`-Ergebnisse der Apple-libm können von glibc abweichen: Golden-/
Hash-Tests, die davon abhängen, werden vorab identifiziert und getrennt
von Metal-Abweichungen bewertet (sonst Fehlzuordnung).
