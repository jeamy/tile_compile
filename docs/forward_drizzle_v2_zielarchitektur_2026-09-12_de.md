# Forward Drizzle v2: Entscheidungs- und Implementierungsplan als AQMH-Ersatz

**Stand:** 2026-09-12
**Status:** Vorimplementierungsplan; Architekturentscheidungen werden erst nach den in diesem Dokument definierten Oracle-, Prototyp- und Mess-Gates freigegeben
**Scope:** Vollständiger Ersatz der Record-/Sort-/Kandidatenarchitektur; keine lokale Optimierung des bestehenden Pfads wird als v2-Zielerfüllung gewertet

## 1. Ziel und Abgrenzung

Forward Drizzle v2 soll AQMH als einzige produktive Rekonstruktionsmethode ersetzen. Die neue Rekonstruktion muss:

- CFA-Samples ohne vorheriges Debayering flusserhaltend auf das Zielraster abbilden;
- helle Sterne, Nebelgradienten, negative normalisierte Hintergründe und sparse R/B-CFA-Abdeckung korrekt behandeln;
- Cosmic Rays, Hotpixel, Satellitenspuren und einzelne fehlerhafte Frames robust unterdrücken;
- geometrische Abdeckung, radiometrische Verwendbarkeit, Schätzer-Support und Profil-Support sauber trennen;
- innerhalb radiometrisch verwendbarer Abdeckung keine statistisch erzeugten schwarzen Löcher produzieren;
- mit der Framezahl asymptotisch linear skalieren;
- RAM, VRAM und temporären Speicher gemeinsam und fail-closed planen;
- CPU und GPU nach demselben mathematischen Vertrag ausführen;
- Coverage, Rekonstruktion, Confidence und Multiband ohne redundante exakte Geometriesweeps erzeugen;
- den bestehenden Trusted-Run-, Resume- und Atomic-Commit-Vertrag erhalten;
- das verbindliche Produktionszeitgate erfüllen.

Die strategischen Ziele sind:

```text
keine Host-Contribution-Records
keine per-Pixel-Sortierung über alle Frames
kein Pixel×Frame-Kandidatenspeicher
keine Device-Allokation im Frame-/Tile-Hotpath
keine statistisch erzeugten Supportlöcher
keine doppelte exakte CFA-Geometrierasterisierung
```

Noch **nicht** vorentschieden sind:

```text
Target-Gather oder dichter Device-Scatter
bounded-memory Robustschätzer und seine Parameter
Fallbackalgorithmus
numerischer FP32/FP64-Vertrag
volle Bandbreite oder X-Tiling
lokale Warp-Inversion und Grid-Lebensdauer
Multiband-Ringbuffer
Resume-Granularität
```

Diese Entscheidungen werden durch die unten definierten Gates getroffen. Das Dokument behauptet nicht mehr, dass Target-Gather, `K=15`, ein 2×2-Support-AND oder vollständiger Verzicht auf X-Tiles bereits bewiesen seien.

### 1.1 Verbindliche Umsetzungsreihenfolge

Die Gates stehen nicht nur im hinteren Arbeitsteil, sondern bilden die
verbindliche Reihenfolge der gesamten Umsetzung:

| Gate | Inhalt | Implementierungsergebnis |
|---:|---|---|
| 0 | Evidenz und Oracles | reproduzierbare Referenz, begrenzte Oracles und vorab festgelegte Abnahmeschwellen |
| 1 | affine Enumeration | gemessene Entscheidung zwischen Target-Gather und dichtem Device-Scatter |
| 2 | Support und Scale-Fold | vier Supportebenen und framekorrekter Zähler-/Nenner-/Quadratgewicht-Fold |
| 3 | Robustschätzer | ausgewähltes bounded-memory Verfahren und vollständige Fallbackzustandsmaschine |
| 4 | Rauschmodell, Confidence und Numerik | verfügbare Eingänge, Rechentypen und CPU/GPU-Toleranzen |
| 5 | gemeinsamer Speicher- und I/O-Plan | RAM, VRAM, Temp, Tileplan, Readerlebensdauer und Read-Amplification |
| 6 | affiner unpublizierter Prototypkern | recordfreier Kern mit persistentem Workspace und vollständiger Telemetrie |
| 7 | Transaktion und Resume | versionierte Artefakte, Checkpoint und Atomic Commit |
| 8 | lokale Warps | geprüfte inverse Repräsentation und recordfreier lokaler Pfad |
| 9 | Streaming-Multiband | Halo, Ringbuffer, Confidence-Kalibrierung und validierte Fusion |
| 10 | Cutover | vollständige Produktionsabnahme; anschließend Entfernung des alten Pfads |

Der Source-Quality-Pack aus §16 ist ein Vorgängerschnitt für Gate 6. Seine
Reader-, Metrik- und Publikationsverträge müssen spätestens mit Gate 5
geschlossen und vor dem affinen Prototypkern implementiert sein.

## 2. Empirischer Ausgangspunkt: `m42-test1`

Der 60-Frame-Run `/media/tc_500/m42-test1` auf 3840×2160-OSC-Daten zeigte:

| Messwert | Wert |
|---|---:|
| Gesamtdauer erfolgreicher Phasen | 1.928,6 s |
| `FORWARD_DRIZZLE` | 1.361,2 s (70,6 %) |
| `SAMPLING_GEOMETRY` | 237,5 s (12,3 %) |
| `SOURCE_QUALITY_MAPS` | 95,2 s (4,9 %) |
| Forward-Drizzle-Stripe-Zeit | 1.224,5 s (90,0 % der FD-Zeit) |
| Source-Samples | 497.664.000 |
| Durchsatz | 365.614 Source-Samples/s |
| interne Pixel/Kanal-Auswertungen | 103.381.848 |
| weggeclippte Framebeiträge | 350.191.643 |
| vollständig verworfene interne Pixel/Kanäle | 2.276.618 |
| CUDA-Y-Bands | 6 |
| maximale X-Tiles je Band | 40 |
| maximale CUDA-Producer-Aufrufe | 14.400 |

### 2.1 Nachgewiesener Bildfehler

Die geometrische Coverage meldete innerhalb der Analysefläche für R, G und B jeweils 100 % Support und keine interne Lücke. Nach dem statistischen Clipping fehlten in derselben Fläche jedoch:

| Kanal | fehlende Pixel | Anteil |
|---|---:|---:|
| R | 406.143 | 5,10 % |
| G | 5.572 | 0,07 % |
| B | 400.013 | 5,02 % |
| mindestens ein Kanal | 793.828 | 9,96 % |

Von 487.513 Lochkomponenten waren 66,8 % Einzelpixel und 96,6 % höchstens vier Pixel groß. Im hellsten Prozent der Bildfläche stieg die Lochrate auf 13,2 %. Das entspricht dem sichtbaren Salz-und-Pfeffer-Muster.

Die Downstream-Maske war exakt gleich `finite(R) && finite(G) && finite(B)`. Ein fehlender Kanal wurde anschließend für alle drei Kanäle auf Schwarz gesetzt. Die Eingangsframes und Qualitätskarten waren endlich; fehlende Darks waren nicht die Ursache der Supportlöcher.

### 2.2 Nachgewiesener Strukturfehler

Der aktuelle CUDA-Pfad führt nur die Polygonrasterisierung auf der GPU aus. Alle folgenden Operationen laufen auf dem Host:

1. Contribution-Records vom Device herunterladen;
2. Records pro Frame/Tile sortieren;
3. Kandidatenmatrix `Kanal × Pixel × Frame` befüllen;
4. pro Pixel/Kanal Median und MAD wiederholt sortieren;
5. Alpha-Quantile mit weiteren Sortierungen berechnen;
6. Uniform, Raw, Fine und Medium reduzieren.

Jeder Frame/Tile-Aufruf allokiert und löscht Devicebuffer, synchronisiert das gesamte Device und lädt Records zurück. Die Reduktion verwendete während des Runs effektiv ungefähr einen CPU-Kern.

### 2.3 Grenzen des Evidenzlaufs

`m42-test1` ist ein belastbarer Fehlernachweis, aber keine freigabefähige wissenschaftliche oder Performance-Referenz:

- dirty Worktree und kein an einen sauberen Commit gebundenes Binary;
- anfänglicher `GLOBAL_QUALITY`-Fehler und anschließender Resume;
- kein vollständiger Kaltlauf in einem Prozess;
- Lauf ohne Darks;
- nur 60 Frames;
- keine lokalen Warp-Modelle (`local_model_samples_total = 0`);
- null effektive Multiband-FWHM-Sterne;
- kein aktiviertes detailliertes CUDA-Unterphasenprofil;
- keine CPU/GPU-Paritätsmessung dieses Runs.

Der Lauf belegt:

- die statistisch erzeugten Supportlöcher;
- deren R/B-Dominanz und Helligkeitsabhängigkeit;
- die Verstärkung durch Output-Support und Downstream-Nullsetzung;
- die Phasenanteile dieses konkreten Builds;
- Band-/X-Tile-Struktur und Producer-Aufrufzahl;
- praktisch deaktiviertes Fine-/Medium-Alpha.

Er belegt nicht:

- Produktionsdurchsatz für 600 Frames;
- lokale-Warp-Korrektheit oder -Performance;
- wissenschaftliche Überlegenheit eines neuen Robustschätzers;
- Gather-vs-Scatter;
- CPU/GPU-Numerikvertrag.

## 3. Nichtziele und bestehende taktische Änderungen

Folgende Maßnahmen sind ausdrücklich keine v2-Zielarchitektur:

- nur `min_fraction` oder `min_n_eff` absenken;
- Supportlöcher nachträglich interpolieren;
- fehlende Kanäle erst im Downstream inpainten;
- den seriellen Record-Reducer lediglich mit OpenMP versehen;
- bestehende per-Pixel-Sortierungen nur durch Scratch-Reuse beschleunigen;
- größere Kandidatentiles durch mehr RAM oder VRAM erzwingen;
- Contribution-Records weiterhin GPU→CPU übertragen;
- einen separaten vollständigen `SAMPLING_GEOMETRY`-Sweep beibehalten;
- Multiband unverändert mit `Q_p90-Q_p50` als zwingender Confidence-Bedingung betreiben.

Die aktuellen Änderungen für `guard_fallback`, row-band-parallele Hostreduktion und wiederverwendeten Alpha-Scratch sind sinnvolle Stabilisierung und Messhilfe für den bestehenden Pfad. Sie lösen jedoch nicht:

- Contribution-Records;
- framezahlabhängige Kandidatenmatrix;
- vollständige Frame-Sortierungen pro Pixel;
- X-Tile-Vervielfachung;
- synchrone CUDA-Kleinstjobs;
- doppelte Coverage-Geometrie;
- 2×2-Fold-Vertrag;
- lokale Warp-Inversion.

Sie werden daher nicht als v2-Meilensteine gezählt.

## 4. Reproduzierbare Referenz und Oracles

Vor jeder v2-Implementierung werden Referenzlauf und Oracles eingefroren.

### 4.1 Referenzbindung

Jeder Evidenz- oder Abnahmelauf bindet:

- Git-Commit und Dirty-Tree-Status;
- Build-ID und Git-Commit;
- Compiler-, CUDA-, Treiber- und Buildprofil;
- CPU, GPU, RAM, SSD und Workerzahl;
- unveränderlichen Config-Snapshot und dessen Run-ID;
- Inputmanifest und stabile Frameidentitäten;
- reproduzierbar definiertes Kalt-/Warmlaufprotokoll für den OS-Dateicache;
- frisches Runverzeichnis;
- vollständige Phase- und Unterphasentelemetrie.

Ein Run mit vorherigem Fehler und Resume ist als Diagnose zulässig, aber keine Kaltlauf-Performancebaseline.

### 4.2 Getrennte Oracles

Der bisherige finale Recordpfad ist wegen seiner Lochsemantik nicht als Ganzes das Oracle. Stattdessen gelten getrennte Referenzen:

1. **Geometrie-Oracle**
   Bestehender exakter Source-Scatter vor Clipping: vollständige positive Overlaps mit stabiler Source-/Leaf-Reihenfolge.

2. **Frame-Kandidaten-Oracle**
   Exakte `A`, `B`, Q- und Artifact-Summen je Frame, Zielpixel und Kanal vor statistischer Reduktion.

3. **Statistik-Oracle**
   Vollständige frameweise robuste Auswertung ohne bounded-memory Approximation. Sie darf bei Source-Support keinen NaN-Supportverlust erzeugen.

4. **Fold-Oracle**
   Exakt spezifizierte Faltung der Profilzähler, Profilnenner und quadratischen Gewichte von Internal- zu Output-Scale.

5. **Support-Oracle**
   Die vier in §5 definierten Supportebenen.

6. **Multiband-Oracle**
   Uniform, Raw und Multiband werden auf identischen Samples, Masken und matched star positions verglichen.

Die Frame-Kandidaten- und Statistik-Oracles werden nicht als vollständiger
`Frame × Pixel × Kanal`-Store für einen 600-Frame-Canvas materialisiert.
Vollständige Oracles gelten für kleine synthetische Bilder; reale Daten werden
über alle Frames, aber nur für deterministisch ausgewählte Tiles, Randfälle,
Helligkeitsklassen und Problemregionen gestreamt ausgewertet. Damit bleibt der
Oracle unabhängig vom zu prüfenden bounded-memory Verfahren, ohne selbst einen
mehrere Terabyte großen Kandidatenstore einzuführen.

### 4.3 Vorab festgelegte Abnahmeschwellen

Vor jedem Prototypvergleich werden die zulässigen Grenzen für fehlende oder
zusätzliche Overlaps, Fluxfehler, Surface-Brightness-Fehler, Bias zum
Frame-Oracle, False Rejection, Ausreißerrest, CPU/GPU-Abweichung, Speicher und
Laufzeit festgeschrieben. Sie werden nicht nach Sichtung der Kandidatenergebnisse
angepasst. Nichtanwendbare Metriken erhalten eine begründete, persistierte
Nichtanwendbarkeit und gelten nicht als bestanden.

## 5. Vierstufiges Supportmodell

### 5.1 Geometry-Support

```text
geometry_support(pixel, channel)
```

Mindestens ein CFA-Droplet besitzt einen positiven geometrischen Overlap mit der Zielzelle. Sourcewert, Sensormaske und Profilgewicht werden hier nicht betrachtet.

### 5.2 Source-Support

```text
source_support(pixel, channel)
```

Mindestens ein Beitrag erfüllt gleichzeitig:

- positiver geometrischer Overlap;
- endlicher Sourcewert;
- gültige Sensormaske;
- kein radiometrischer Hard-Veto des Source-Samples.

Verbindliche Invariante:

```text
source_support(pixel, channel)
    => finite(uniform(pixel, channel))
```

Statistische Robustheit darf Source-Support nicht löschen.

### 5.3 Estimator-Support

```text
estimator_support(pixel, channel)
```

Die radiometrisch gültigen Beiträge reichen für den spezifizierten robusten Primärschätzer. Fehlt Estimator-Support, wird der exakt definierte Fallback verwendet; Source-Support und Uniform bleiben bestehen.

### 5.4 Profile-Support

```text
profile_support(profile, pixel, channel)
```

Der profilspezifische Nenner ist positiv und endlich. Für Raw/Fine/Medium können Q-Veto oder Nullgewicht Profile-Support verhindern, obwohl Uniform gültig bleibt.

Es gilt:

```text
profile_denominator > 0
    => finite(profile_value)
```

### 5.5 Downstream-Vertrag

Der Downstream darf nicht still alle RGB-Kanäle nullen, wenn ein unerwarteter interner Profile-Support-Defekt innerhalb gültigen Source-Supports auftritt. Eine solche Verletzung blockiert den Commit und wird mit Kanal, Pixelzahl und Ursache berichtet.

## 6. Scale-Fold als Zähler-/Nenner-Algebra

Ein pauschales 2×2-Support-AND und der Mittelwert bereits normalisierter Subpixel sind nicht als flusserhaltend vorausgesetzt.

### 6.1 Profilsummen

Für jedes interne Subpixel `j` und Profil `p` existieren:

```text
A_p,j  = gewichtete Wertsumme
B_p,j  = Gewichtsumme
B2_p,j = Summe quadrierter framebezogener Beitragsgewichte
```

Der native Outputpixel wird aus den Summen gebildet:

```text
A_p,out = Σ area_j × A_p,j
B_p,out = Σ area_j × B_p,j
value_p,out = A_p,out / B_p,out
```

Für vier gleich große interne Zellen ist `area_j` identisch. Die Faktoren können algebraisch gekürzt werden, müssen aber im Vertrag explizit bleiben.

Für `n_eff` reicht die Summe der bereits zusammengefassten `B2_p,j` nicht aus,
weil derselbe Frame oder dieselbe robuste Schätzereinheit mehrere interne
Subpixel beitragen kann. Mit dem effektiven framebezogenen Gewicht `b_p,f,j`
gilt:

```text
b_p,f,out = Σ_j area_j × b_p,f,j
B2_p,out  = Σ_f b_p,f,out²
          = Σ_f (Σ_j area_j × b_p,f,j)²
n_eff_p,out = B_p,out² / B2_p,out
```

Die Kreuzterme zwischen internen Subpixeln dürfen nicht verloren gehen. Das
bounded-memory Verfahren muss daher die vier Subpixel je Frame beziehungsweise
je robuster Schätzereinheit vor der abschließenden `B²`-Reduktion falten oder
äquivalente Kreuzsummen führen. Aus `Σ_j area_j² × B2_p,j` darf `B2_p,out`
nicht rekonstruiert werden.

Nicht zulässig als allgemeine Definition ist:

```text
mean(A_p,j / B_p,j)
```

### 6.2 Fehlende interne Teilflächen

Vor Implementierung wird für 1/4, 2/4, 3/4 und 4/4 Source-Support entschieden:

- Mindestflächenabdeckung;
- Renormalisierung oder explizite Nichtabdeckung;
- Profile-Support je Profil;
- Confidence-Absenkung;
- CFA-Kanalkonsistenz;
- Flux- und Surface-Brightness-Semantik.

Diese Entscheidung wird gegen konstante Felder, analytische Gradienten und Punktquellen validiert. Ein einzelnes fehlendes internes Subpixel darf nicht ohne mathematischen Vertrag den vollständigen nativen Pixel verwerfen.

### 6.3 Fold-Gate

Vor Freigabe müssen gelten:

- konstantes Feld bleibt konstant;
- integrierter Punktquellenflux bleibt innerhalb der definierten Toleranz;
- Chunk-/Tile-Grenzen ändern das Ergebnis nicht;
- CPU und GPU besitzen identische Supportentscheidungen;
- 2/1 erzeugt keine statistisch verursachten Einzelpixellöcher.

## 7. Affine Enumeration: Gather-vs-Scatter-Entscheidung

### 7.1 Keine Vorentscheidung für Target-Gather

Die v2-Invariante lautet:

```text
keine Hostrecords
keine N-Frame-Sortierung pro Pixel
keine Pixel×Frame-Kandidatenmatrix
```

Ob sie durch Target-Gather oder dichten Device-Scatter erfüllt wird, entscheidet ein Prototypvergleich.

### 7.2 Reale Gather-Komplexität

Für Target-Gather gilt:

```text
O(N × P_target × C_neighbours)
```

`C_neighbours` hängt von Pixfrac, Transformation, Jacobian, lokaler Verzerrung, interner Skalierung und konservativer Bounding-Box ab.

Für `m42-test1`:

```text
P_internal = 7868 × 4540 ≈ 35,7 Mio.
N × P_internal ≈ 2,14 Mrd. Frame-/Zielpixel-Paare
```

Dem stehen ungefähr 497,7 Mio. Source-Samples im Source-Scatter gegenüber. Gather wird daher nicht allein aufgrund entfallender Records als schneller angenommen.

### 7.3 Konservative affine Kandidatenmenge

Für Zielzelle `T`, affine Abbildung `A` und Source-Droplet `D` muss die Kandidatenmenge eine konservative Minkowski-Grenze erfüllen:

```text
S_candidates ⊇ A⁻¹(T) ⊕ D_reflected
```

Jeder Kandidat wird danach durch exakte Vorwärtsprojektion und Polygon-/Zellintersektion geprüft.

Die Spezifikation muss enthalten:

- offene/geschlossene Zellgrenzen;
- Droplet-Mittelpunkt und Pixfrac-Radius;
- Rundung von ganzzahligen Source-Bounds;
- Rotation, Skalierung und Shear;
- fast singuläre Matrizen;
- internen Scale-Faktor;
- numerische Sicherheitsmarge mit Beweis oder Oracle-Evidenz.

Akzeptanz:

- kein positiver Oracle-Overlap fehlt;
- keine zusätzliche Kandidatenprüfung verändert A/B;
- Flux und Support entsprechen dem Geometrie-Oracle;
- Kandidatenmenge bleibt praktisch begrenzt.

### 7.4 Zu vergleichende affine Prototypen

**Prototyp G – Target-Gather**

- ein Thread pro Frame/Zielpixel;
- inverse konservative Nachbarschaft;
- exakte Vorwärtsprüfung;
- direkte lokale hinreichende Statistik.

**Prototyp S – dichter Device-Scatter**

- ein Thread pro Source-Sample;
- direkte Device-Akkumulation in dichte Frame-/Gruppensummen;
- keine Hostrecords;
- deterministische Reduktionsstrategie oder dokumentierter Numerikvertrag.

Gemessen werden auf realer Canvasgröße:

- vollständige Kernelzeit;
- Nachbar- beziehungsweise Overlapprüfungen;
- Atomic-Contention;
- RAM/VRAM;
- H2D/D2H;
- Kernelstarts;
- Flux-/Support-Parität;
- Skalierung mit Pixfrac, Rotation und Shear.

Erst dieses Gate legt Gather oder Scatter fest.

## 8. Bounded-memory Robustschätzer: Auswahl statt Vorfestlegung

### 8.1 Kein festes `K=15`

Median-of-Means ist ein Kandidat, nicht der bereits gewählte Algorithmus. Ein Cosmic Ray kann ein Gruppenmittel kontaminieren; das Verwerfen der Gruppe entfernt zugleich gute Frames. Ungleichmäßige CFA-Abdeckung verändert Gruppengröße und Gewicht pixelweise. Eine bloße Zuordnung aus der Frame-ID garantiert keine ausgeglichene Belegung.

### 8.2 Kandidatenverfahren

Mindestens folgende Verfahren werden gegen das vollständige Frame-Oracle geprüft:

- deterministisch balanciertes Median-of-Means;
- Huber-/Catoni-M-Schätzer;
- winsorisierte Framewerte;
- gewichtete Quantilhistogramme;
- feste Quantilskizzen;
- blockweise exakte Kandidatenreduktion;
- Hybrid aus exaktem Pilotquantil und bounded Einflussfunktion.

### 8.3 Adversariale Testmatrix

- einzelner Hotpixel;
- persistenter Sensor-Hotpixel mit Dither;
- einzelner Cosmic Ray;
- mehrere Cosmic Rays in verschiedenen Frames;
- Satellitenspur;
- Defokus-/Seeing-Ausreißer;
- Registrierungsfehler;
- helle Sternkante;
- ausgedehnter Nebelgradient;
- negative normalisierte Hintergründe;
- sparse R/B-CFA-Abdeckung;
- ungleiche geometrische Gewichte;
- ungleich verteilte Profile-Q-Gewichte;
- zeitlich korrelierte Ausreißer.

### 8.4 Vergleichsmetriken

Für jeden Pixel/Kanal und jedes Verfahren:

- Bias zum Frame-Oracle;
- absoluter und relativer Fehler;
- Fluxfehler;
- Ausreißerrest;
- Supportentscheidung;
- false rejection guter Daten;
- Verhalten an Gradienten;
- Speicher pro Pixel;
- Operationen und Laufzeit;
- CPU/GPU-Reproduzierbarkeit.

### 8.5 Exakter Fallbackvertrag

Das gewählte Verfahren erhält eine vollständige Zustandsmaschine. Vor Implementierung werden für jeden Fall exakt festgelegt:

```text
Primärschätzer erfolgreich
Degenerierte Skala bei identischen Werten
Guard unterschritten
Zu wenige gültige Kandidaten oder Gruppen
Kein Source-Support
Kein profilspezifischer Nenner
```

Für jeden Zustand sind verbindlich:

- akzeptierte beziehungsweise begrenzte Werte;
- Profilzähler und Profilnenner;
- `weight_sum`, `weight_sum_squared`, `n_eff`;
- Uniform/Raw/Fine/Medium;
- Confidence;
- Supportebene;
- Diagnostic Counter;
- CPU/GPU-Reihenfolge.

Formulierungen wie „winsorisieren oder Median verwenden oder alle Gruppen verwenden“ sind nicht implementierbar und in der freigegebenen Spezifikation unzulässig.

## 9. Rausch-, Gradienten- und Registrierungsmodell

Das bisher skizzierte Modell

```text
sigma_model² = sigma_noise²
             + gradientᵀ × registration_covariance × gradient
             + sigma_sampling²
```

ist nur eine Hypothese. Die benötigten Eingänge existieren noch nicht vollständig.

### 9.1 Registrierung

`FrameSamplingTransform` enthält Residual- und Prediction-Faktoren, aber keine Kovarianz. Zu entscheiden sind:

- affine Parameterkovarianz oder lokale 2×2-Ortskovarianz;
- Koordinatensystem und Einheit;
- Ableitung aus Inlier-Residualen;
- Verhalten bei vorhergesagten Modellen;
- Verhalten bei lokalen Warps;
- Persistenz, Versionierung und Identitätsfelder im Sampling-Plan-Schema.

Ein skalarer `registration_residual_factor` darf nicht still als Kovarianz interpretiert werden.

### 9.2 Rauschen

Zu spezifizieren sind:

- Herkunft von `sigma_noise`;
- Sensor-/Readnoise gegenüber lokalem Hintergrund-MAD;
- Behandlung normalisierter negativer Werte;
- Kanalabhängigkeit;
- Skalierung mit geometrischem Gewicht und Gruppengröße.

### 9.3 Gradient und Sampling

Zu spezifizieren sind:

- Bildquelle des Gradienten;
- Skala und Glättung;
- Randbehandlung;
- Vermeidung eines zirkulären Schätzers, der vom bereits geclippten Ergebnis abhängt;
- Definition von `sigma_sampling` für CFA, Pixfrac und interne Skalierung.

### 9.4 Numerikentscheidung

Für jede Operation wird ein Typ festgelegt und gemessen:

| Operation | zu entscheidende Typen |
|---|---|
| affine/inverse Geometrie | FP32 / FP64 |
| Polygonoverlap | FP32 / FP64 |
| Zähler-/Nennerakkumulation | FP32 / compensated FP32 / FP64 |
| robuste Statistik | FP32 / FP64 |
| Scale-Fold | FP32 / FP64 |
| CPU-Oracle | FP64 |

Auf der GTX 1660 Ti wird FP64 nicht ohne Messung als Standard angenommen. Die Entscheidung erfolgt über wissenschaftliche Fehlerschranke und Produktionsgate.

## 10. Gemeinsame RAM-/VRAM-Planung und Tiling

### 10.1 Keine unbelegte Full-Width-Annahme

Ein Gruppenakkumulator mit ungefähr 56 Byte ergäbe bei 3 Kanälen und 15 Gruppen rund 2520 Byte pro internem Pixel beziehungsweise etwa 19,8 MiB pro interner Zeile bei 7868 Spalten – vor Source-, Q-, Output-, Confidence-, Halo- und Multibandpuffern.

Volle Breite ist daher nur zulässig, wenn die gemeinsame Rechnung sie belegt.

### 10.2 Verbindliche Speicherformel

Für jede Kandidatenarchitektur wird vor Implementierung aufgestellt:

```text
VRAM =
  robust_accumulators(tile)
+ geometry_accumulators(tile)
+ source_device_slots
+ quality_device_slots
+ inverse_geometry
+ output_profile_slots
+ confidence_slots
+ fold_scratch
+ multiband_halo_and_ring
+ CUDA reserve

RAM =
  pinned_source_slots
+ pinned_quality_slots
+ source_reader_cache
+ quality_reader_cache
+ output_staging
+ resume/index metadata
+ filesystem/codec scratch
+ margin
```

Alle Multiplikationen werden overflow-geprüft. Host- und Devicebudget werden getrennt geplant; der kleinere zulässige Tileplan gewinnt.

### 10.3 X-Tile-Fallback

Wenn selbst die minimale Full-Width-Bandhöhe nicht passt, ist X-Tiling erlaubt. Es muss jedoch:

- unabhängig von `frame_count` dimensioniert werden;
- konservative Source-/Quality-Halos besitzen;
- gepinnte Source-/Q-Bänder über benachbarte X-Tiles wiederverwenden;
- keine Inhaltsprüfung oder dafür erforderliche Zusatzlektüre ausführen;
- Read-Amplification messen;
- identische Fold-/Supportergebnisse zu Full-Width liefern.

Nicht zulässig ist ein Tileplan, dessen Breite mit wachsendem `N` gegen wenige Pixel kollabiert.

## 11. Lokale Warp-Architektur

Lokale Warps sind ein eigenes Freigabe-Gate und dürfen nicht aus affinen Ergebnissen extrapoliert werden.

### 11.1 Zu spezifizierende Größen

- Auflösung des inversen Deformationsgitters;
- Interpolationsfehler;
- Jacobian- und Krümmungsgrenzen;
- Newton-Start, -Iteration und -Abbruch;
- konservative Kandidatenbox;
- nicht invertierbare Regionen;
- Grid-Bauzeit;
- Grid-RAM/VRAM;
- Persistenz- und Resumeformat;
- Wiederverwendung über Y- und X-Tiles.

### 11.2 Lebensdauerkandidaten

1. alle Framegrids persistent – einfacher Zugriff, aber Speicher wächst mit `N`;
2. Grid pro Band neu – speicherarm, aber redundante Berechnung;
3. Grid pro Frame disk-/mmap-backed mit begrenzter GPU-Residenz – bounded, aber komplexer;
4. kompakte lokale Modellkoeffizienten plus on-device Inversion – kein volles Grid, aber höhere Kernelkosten.

Die Entscheidung erfolgt durch Speicherformel, Oracle-Parität und lokalen 600-Frame-Prototyp. Der alte Record-Hybridpfad ist kein langfristiger v2-Fallback.

## 12. Coverage-Fusion mit zwei Geometrievarianten

CFA-Droplet-Support und dichter Frame-Footprint sind fachlich getrennt.

### 12.1 Früher günstiger Preflight

Vor teurer Rekonstruktion werden offensichtliche Fehler über eine konservative günstige Prüfung erkannt:

- transformierte Framepolygone;
- Dither-/CFA-Phasenverteilung;
- sichere Obergrenzen der maximal erreichbaren Kanalabdeckung für frühe
  Ablehnung und optional sichere Untergrenzen für reine Diagnose;
- ungültige oder nicht invertierbare Frames;
- erwartete Analysis-Region.

Der Preflight ersetzt nicht das exakte Gate. Wegen zu geringer Coverage darf er
nur ablehnen, wenn eine konservative Obergrenze bereits unter dem erforderlichen
Mindestwert liegt. Eine niedrige Untergrenze beweist keine unzureichende
Coverage.

### 12.2 Exakte integrierte Coverage

Während der Rekonstruktion werden getrennt akkumuliert:

```text
CFA geometry:
  sum_f(B_f,c), sum_f(B_f,c²), contributors_c, channel_support_c

dense footprint:
  frame_footprint_count, dense_common_overlap
```

Die beiden Zustände dürfen nicht aus demselben Droplet-Supportbit abgeleitet werden.
`B_f,c` ist dabei zuerst über alle Droplets desselben Frames, Zielpixels und
Kanals zu aggregieren. Erst dieser Framewert wird quadriert. Das Quadrieren
einzelner Droplet-Overlaps würde eine andere und unzulässige `n_eff`-Semantik
erzeugen.

### 12.3 Commit-Gate

Die Rekonstruktionsgeneration bleibt unveröffentlicht, bis global geprüft sind:

- Source-Support je Kanal;
- p10-`n_eff` je Kanal;
- Analysis-Pixelzahl;
- dichte Common-Overlap-Maske;
- interne geometrische Lochkomponenten;
- unerwartete Nonfinite-Pixel innerhalb Source-Support;
- Fold- und Numerikinvarianten.

Damit wird kein zweiter vollständiger exakter CFA-Sweep benötigt. Der frühe Preflight verhindert, dass offensichtliche Coverage-Fehler erst nach der gesamten Rekonstruktion erkannt werden.

## 13. Persistenter GPU-Workspace und Ausführungsmodell

### 13.1 Workspace

Nach Entscheidung von Gather/Scatter, Robustschätzer und Tileplan wird ein exakter Workspace definiert. Mindestens enthalten sind:

```text
Source-Device-Slots
Quality-Device-Slots
inverse Geometrie oder Modellparameter
robuste/Profil-Akkumulatoren
Coverage-Akkumulatoren
Output-/Fold-Scratch
Multiband-Ringbuffer
pinned Host-Slots
CUDA-Streams und Events
```

Innerhalb des Frame-/Tile-Hotpaths sind verboten:

- `cudaMalloc`;
- `cudaFree`;
- globales `cudaDeviceSynchronize`;
- wiederholte Reader-Konstruktion;
- Inhaltsprüfung von Source-, Q-, Geometrie- oder Profilnutzdaten;
- Host-Download von Contribution-Records.

### 13.2 Pipeline

Source-I/O, Quality-I/O, Upload, Compute, Output-Download und Store-Write werden mit mehreren Slots überlappt. Die genaue Slotzahl folgt aus der Lebensdauertabelle; Double Buffering wird nicht ohne Abhängigkeitsanalyse als ausreichend angenommen.

Jeder Slot besitzt einen expliziten Zustand:

```text
FREE
HOST_FILLING
READY_FOR_UPLOAD
UPLOADING
COMPUTING
READY_FOR_DOWNLOAD
DOWNLOADING
READY_FOR_WRITE
WRITING
```

CUDA-Events und Host-Futures definieren die Übergänge. Globales Synchronisieren ist nur am Phasenabschluss oder bei hartem Fehler zulässig.

## 14. Streaming-Multiband und Halo-Vertrag

### 14.1 À-trous-Radius

Für einen B3-Kernel mit Radius 2 und Dilatationen `1,2,4,...` beträgt der kumulative reine À-trous-Radius bis Level `L`:

```text
R_atrous(L) = 2 × (2^L - 1)
```

Der Gesamthalo ist nicht automatisch `R_atrous`. Hinzu kommen abhängig vom finalen Algorithmus:

- Alpha-/Confidence-Smoothing;
- Energy-Guard-Fenster;
- weitere lokale Validierungsfilter;
- Fold-Randbedarf.

Die Filterverkettung bestimmt, ob Radien addiert oder als Maximum kombiniert werden.

### 14.2 Verbindliche Festlegungen

Vor Implementierung werden spezifiziert:

- Inputhalo je Profil und Level;
- Randregel (Mirror, Clamp, Invalid oder andere definierte Regel);
- gültiger Bandkern;
- erste/letzte Bildzeile;
- Anzahl gleichzeitig residenter Vorgänger-/Folgebänder;
- Ringbuffergröße;
- Zeitpunkt, ab dem ein Outputband unveränderlich ist;
- Wechselwirkung mit X-Tiles;
- Resume-Grenze.

Die Formel muss beweisen, dass der Bandoutput bit- beziehungsweise toleranzidentisch zur Full-Image-Referenz ist.

### 14.3 Producer-/Consumer-Pipeline

Erst nach der Haloanalyse wird die Pipeline festgelegt. Eine bloße Beschreibung `N+1/N/N-1` beweist nicht, dass zwei Outputslots genügen. Die Ringgröße ergibt sich aus Filterradius, Bandkern, asynchronen I/O-Slots und Commitgrenze.

## 15. Confidence-Modell

### 15.1 Fehler des bisherigen Modells

Das bisherige `A_separation` basiert auf `Q_p90-Q_p50`. Sind alle Frames ähnlich gut, fällt die Differenz auf nahezu null. In `m42-test1` lagen die mittleren Detail-Alphas bei `2,98e-10` und `7,32e-8`; Fine/Medium waren damit praktisch deaktiviert.

### 15.2 Anforderungen an den Ersatz

Confidence muss mindestens unterscheiden:

```text
absolute Qualität
Schätzerstabilität
effektive Stichprobengröße
Artifact-Confidence
Registration-Confidence
Profile-Support
```

Verbindlich:

```text
alle Frames ähnlich gut
    => nicht allein deshalb Confidence 0
```

Eine konkrete Formel wird erst nach Wahl des Robustschätzers und des Rausch-/Registrierungsmodells festgelegt. Sie muss aus bounded-memory Statistiken berechenbar sein und darf keine per-Pixel-Heap-Allokationen oder mehrfachen Sortierungen derselben Wertereihe benötigen.

### 15.3 Multiband-Abnahme

- Uniform bleibt immutable control;
- Raw bleibt immutable weighted baseline;
- Multiband wird gegen beide verglichen;
- matched star positions sind verpflichtend;
- null effektive Sterne blockieren eine positive Multiband-Auswahl;
- global praktisch nulles Alpha wird als eigene Diagnose und nicht als unauffälliger Erfolg gemeldet;
- Objektklasse darf Sicherheitsgates nicht dynamisch aufweichen.

## 16. Source-Quality-Pack

Source Quality bleibt ein frameweiser Vorgänger der Rekonstruktion. Seine
Artefakte werden einmal erzeugt und danach ausschließlich gelesen.

Verbindlich sind:

- Source-Proxy pro Frame genau einmal berechnen;
- Pyramidenskalen gemeinsam aufbauen und jede Skala genau einmal auswerten;
- Composite und Global-Quality-Metriken im selben Framejob erzeugen;
- Q-Maps und Global-Quality-Metriken gemeinsam atomar publizieren;
- nur tatsächlich von Raw, Fine, Medium, Artifact oder Confidence konsumierte
  Streams persistieren;
- `(stream, source_index)` über einen einmal aufgebauten Readerindex in O(1)
  auflösen;
- NaN, unendlich und fachlich nicht anwendbar im Artefaktschema eindeutig
  unterscheiden;
- Reader zwischen Global Quality und Forward Drizzle wiederverwenden;
- pro `(Frame, Y-Band, Stream)` eine Q-Bandansicht höchstens einmal dekodieren
  und über alle zugehörigen X-Tiles gepinnt halten;
- Source- und Q-Ansichten unter einem gemeinsamen RAM-Budget führen.

Ein Commit der Q-Map-Generation ohne die dazugehörigen vollständigen Metriken
ist unzulässig. Ein Resume akzeptiert nur eine vollständig publizierte und zur
Sampling-Plan-Generation passende Source-Quality-Generation.

## 17. Trusted-Run- und Read-Amplification-Vertrag

### 17.1 Trusted Run

Im normalen Run gelten Inputs, normalisierter Cache und Artefaktverzeichnis während der Phase als unverändert. Der Produktlauf prüft:

- Schema und Metadaten;
- Run-, Generation- und Algorithmuskennungen;
- Dateiexistenz;
- erwartete Größe und Geometrie;
- Generation-/Checkpoint-Zuordnung.

Er prüft keine Nutzdateninhalte von Source-, Q-, Geometrie-, Profil- oder
Outputdateien. Es gibt dafür weder einen automatischen Prüflauf noch einen
separaten Prüfbefehl. Änderungen an vorhandenen Nutzdaten liegen in der
Verantwortung des Benutzers. Unbekannte Schema-Versionen, Größenänderungen,
fehlende Dateien und unpassende Generationen bleiben fail-closed.

### 17.2 Pflichtzähler

```text
logical_source_bytes
application_source_bytes
storage_source_bytes
logical_quality_bytes
application_quality_bytes
storage_quality_bytes
application_source_read_amplification
storage_source_read_amplification
application_quality_read_amplification
storage_quality_read_amplification
source_halo_bytes
quality_halo_bytes
source_reused_bytes
quality_reused_bytes
reader_constructions
```

Definition:

```text
application_read_amplification = application_bytes / logical_useful_bytes
storage_read_amplification = storage_bytes / logical_useful_bytes
```

Dabei werden drei Ebenen unterschieden:

```text
logical_*_bytes       fachlich tatsächlich benötigte Nutzbytes
application_*_bytes   von Readern angeforderte oder gemappte Bytes
storage_*_bytes       nachweislich vom Speichermedium gelieferte Bytes
```

Page-Cache-Treffer erhöhen die Anwendungsbytes, aber nicht zwingend die
Storagebytes. Die Zähler werden getrennt nach Y-Band, X-Tile, Frame und Stream
aggregiert. Ein Bytezähler ohne logischen Nenner genügt nicht. Für jeden
Tileplan werden maximale Source- und Quality-Read-Amplification vorab als Gate
festgelegt.

Die Reader-Lebensdauer ist Teil des Vertrags:

- pro `(Frame, Y-Band)` wird die benötigte Source-Ansicht höchstens einmal von
  der Anwendung gelesen und über alle zugehörigen X-Tiles gepinnt;
- pro `(Frame, Y-Band, Q-Stream)` wird die Q-Ansicht höchstens einmal dekodiert
  und über alle zugehörigen X-Tiles gepinnt;
- überlappende Zeilen benachbarter Y-Bänder werden innerhalb des gemeinsamen
  Budgets wiederverwendet;
- Worker erhalten keine unabhängigen leeren Readercaches, wenn dadurch
  dieselben Daten erneut gelesen werden;
- eine Eviction darf keine noch verwendete Ansicht invalidieren.

Kann ein Kandidatenplan diese Lebensdauer nicht einhalten, muss seine gemessene
Verstärkung innerhalb der vorab festgelegten Grenze bleiben; andernfalls fällt
er am Speicher-/I/O-Gate durch.

## 18. Resume- und Atomic-Commit-Schema

### 18.1 Versionierte Artefakte

V2 erhält eigene, versionsgebundene Artefakte, beispielsweise:

```text
forward_drizzle_v2_plan.json
forward_drizzle_v2_geometry.json
forward_drizzle_v2_checkpoint.json
forward_drizzle_v2_generation/
```

Der Plan bindet:

- Pipeline- und Algorithmusversion;
- Inputmanifest;
- Config-Snapshot-ID;
- Sampling-Plan-Generation;
- Source-Quality-Generation;
- Numerikmodus;
- Gather-/Scatter-Variante;
- Robustschätzervertrag;
- Support-/Fold-Vertrag;
- Tile-/Bandplan;
- lokale Warp-Repräsentation;
- Trusted-Run-Modus.

### 18.2 Resume-Granularität

Vor Implementierung wird genau eine Semantik gewählt:

1. vollständiger Neustart von Forward Drizzle; oder
2. Resume ausschließlich an atomar abgeschlossenen Band-/Tilegrenzen.

Bei Band-Resume müssen persistiert sein:

- vollständiger Bandoutput oder vollständige weiterverwendbare Akkumulatoren;
- Coverage-Zwischenzustand;
- Common-Overlap-Zustand;
- Multiband-Halo-/Ringzustand oder eine sichere Wiederanlaufüberlappung;
- Größen, Metadaten und Generationen;
- Commitmarke erst nach fsync-/AtomicOutput-Vertrag.

Teilweise beschriebene Bänder werden verworfen. Ein Phase-Event allein begründet keine Resumierbarkeit.

### 18.3 Publikation

- alle Writes erfolgen in einer unpublizierten Generation;
- Coverage-, Support-, Numerik- und Qualitätsgates laufen vor Publikation;
- Fehler verwerfen die Generation;
- `current.json` beziehungsweise der äquivalente Zeiger wird als letzter atomarer Schritt gesetzt;
- historische Runs werden nicht migriert oder überschrieben.

## 19. Pflichttelemetrie

Performanceinstrumentierung ist Bestandteil des Vertrags und nicht von einer Environment-Variable abhängig. Pro Run werden mindestens ausgegeben:

```text
source_io_seconds
quality_io_seconds
host_to_device_seconds
geometry_seconds
gather_or_scatter_kernel_seconds
robust_reduce_seconds
confidence_seconds
fold_seconds
multiband_seconds
device_to_host_seconds
store_write_seconds
coverage_finalize_seconds
frames_processed
bands_processed
x_tiles_processed
source_samples_or_target_pairs
neighbour_tests
overlap_tests
kernel_launches
device_allocations_in_hotpath
fallback_pixels_by_channel
geometry_support_pixels_by_channel
source_support_pixels_by_channel
estimator_support_pixels_by_channel
profile_support_pixels_by_profile_channel
nonfinite_pixels_inside_source_support
```

Zusätzlich:

- GPU-Auslastung;
- CPU-Worker-Auslastung;
- Peak-RAM und Peak-VRAM;
- Temp-Disk;
- Read-Amplification aus §17;
- getrennte Zähler für robuste Downweightings, Fallbacks, Profile-Vetos und echte Nichtabdeckung.

Ein erfolgreicher Commit mit `nonfinite_pixels_inside_source_support > 0` ist unzulässig.

## 20. Absolutes Produktionsgate

### 20.1 End-to-End-Gate

Verbindliche bestehende Produktanforderung:

- Zielbereich 1800–2400 s;
- harte Obergrenze 2400 s;
- 600 Frames à 3840×2160 OSC;
- `internal_scale=2`, `output_scale=1`;
- vom angenommenen Runstart bis zum Commit der finalen HMS-Ausgabe;
- enthalten: Scan, Kalibration mit vorhandenen gültigen Masters, Normalisierung, Registrierung, Geometrie, Q-Maps, Drizzle, Multiband, Ausgabe, erforderliche Astrometrie, BGE, PCC, HMS, Transfers, zulässige Prüfungen und I/O.

Fehlende Mastererzeugung oder externe Downloads werden separat ausgewiesen; keine benötigte Runphase wird aus der End-to-End-Zeit ausgeklammert.

### 20.2 Rekonstruktions-Teilgate

Zusätzlich gilt für einen Kaltstart bis zur committed Rekonstruktionsausgabe:

- unter 1800 s;
- 600 Frames à 3840×2160 OSC;
- affine und lokale Datensätze;
- Astrometrie, BGE, PCC und HMS ausdrücklich deaktiviert;
- keine Wiederverwendung runabhängiger Normalized-/Q-/Geometrie-/Profilcaches;
- kein Resume;
- keine künstliche Frame- oder Auflösungsreduktion.

### 20.3 Referenzklassen

Mindestens:

- ein realer affiner 600-Frame-Datensatz;
- ein realer lokal verzerrter 600-Frame-Datensatz;
- je Datenklasse zwei vollständige frische Runs, die sowohl das jeweils
  anwendbare Rekonstruktions-Teilgate als auch das End-to-End-Gate erfüllen;
- Rotationswinkel, lokale-Modell-Anteil und verworfene Frames mit Gründen ausweisen.

Duplizierte Frames sind nur als gekennzeichneter Lasttest zulässig und ersetzen keinen realen Datensatz.

### 20.4 Eingefrorene Umgebung

Vor Messung werden konkrete CPU/GPU, RAM, SSD, Worker, Treiber, Compiler, Binary, Config und Inputmanifest eingefroren. Ein Hardwarewechsel erzeugt eine neue gekennzeichnete Baseline. Das Produktionsgate ist eine Abnahmeforderung, keine theoretische Hochrechnung.

## 21. Validierungsmatrix

### 21.1 Geometrie und Flux

- Identity-Affine;
- ganzzahlige und gebrochene Translation;
- Rotation, Skalierung und Shear;
- fast singuläre affine Matrizen;
- lokale Warp-Modelle;
- alle Bayer-Patterns und CFA-Origins;
- Pixfrac-Grenzwerte;
- 1/1, 2/2 und 2/1;
- konstantes Feld;
- linearer Gradient;
- Punktquelle mit analytischem Flux;
- Tile-/Band-Grenzen.

### 21.2 Robustheit

- Hotpixel;
- persistenter Sensor-Hotpixel mit Dither;
- Cosmic Ray;
- Satellitenspur;
- defokussierte Frames;
- Registrierungs-Ausreißer;
- negative normalisierte Hintergründe;
- gesättigte Sterne;
- helle Nebelgradienten;
- sparse R/B-CFA-Abdeckung;
- korrelierte Ausreißer und ungleiche Gewichte.

### 21.3 Support und Fold

```text
source_support => finite Uniform
profile_denominator > 0 => finite Profilwert
kein statistisch erzeugtes Loch
kein stilles kanalübergreifendes Nullsetzen
Fold entspricht Zähler-/Nenner-Oracle
R/G/B-Supportentscheidung entspricht dem Vertrag
```

### 21.4 CPU/GPU

- gleiche Geometriekandidaten;
- gleiche Supportebenen;
- gleiche Fallbackzustände;
- Profilwerte innerhalb dokumentierter absoluter/relativer Toleranzen;
- deterministische Wiederholung auf demselben Backend;
- GPU-Fehler verwirft die uncommittete Generation;
- CPU-Neustart erfüllt denselben fachlichen Vertrag.

### 21.5 Skalierung

- verschiedene Framezahlen bis 600/610;
- affine und lokale Modelle;
- linearer Verlauf der frameabhängigen Arbeit;
- bounded Speicher unabhängig von `N`;
- kein `N log N` pro Pixel;
- keine mit `N` kollabierende Tilebreite;
- keine Device-Allokation im Hotpath;
- keine doppelte vollständige CFA-Geometrierasterisierung;
- Read-Amplification innerhalb des festgelegten Gates;
- absolutes Produktionsgate aus §20.

## 22. Entscheidungs- und Implementierungsreihenfolge

### Gate 0: Evidenz einfrieren

- sauberer Commit;
- reproduzierbare affine und lokale Referenzläufe;
- vollständige Artefakt-/Hardwarebindung;
- Oracle-Captures und Pflichttelemetrie.

**Exit:** Die Ausgangslage ist reproduzierbar; `m42-test1` bleibt nur Diagnosebeleg.

### Gate 1: Affine Enumeration

- Minkowski-/Bounding-Vertrag formulieren;
- Target-Gather und dichten Device-Scatter prototypisieren;
- gegen Geometrie- und Frame-Kandidaten-Oracle prüfen;
- auf realer Canvasgröße messen.

**Exit:** Gather oder Scatter ist durch Korrektheit, Speicher und Laufzeit gewählt.

### Gate 2: Support und Scale-Fold

- vier Supportebenen festlegen;
- Zähler-/Nenner-/W²-Fold definieren;
- Teilflächenregel festlegen;
- konstante Felder, Gradienten und Punktflux validieren.

**Exit:** Kein mathematisch offener 2×2- oder Supportfall bleibt.

### Gate 3: Robustschätzer

- vollständiges Frame-Oracle erzeugen;
- Kandidatenverfahren auf adversarialen Fällen vergleichen;
- bounded-memory Verfahren auswählen;
- exakte Fallbackzustandsmaschine spezifizieren.

**Exit:** Schätzer erreicht die festgelegten Qualitäts-/Robustheitsgrenzen ohne Supportverlust.

### Gate 4: Rauschmodell und Numerik

- Noise-, Gradient-, Sampling- und Registrierungseingänge definieren;
- Sampling-Plan-Schema gegebenenfalls erweitern;
- bounded berechenbare Confidence-Formel und ihre hinreichenden Statistiken
  festlegen;
- FP32/compensated-FP32/FP64 messen;
- CPU/GPU-Toleranzen festlegen.

**Exit:** Jede Formel besitzt verfügbare Eingänge und einen numerischen Typ;
alle für Confidence benötigten Buffer gehen in Gate 5 ein.

### Gate 5: Gemeinsamer Speicherplan

- exakte RAM-/VRAM-/Temp-Formel;
- Full-Width- und X-Tile-Pläne;
- Halo-/Ringbuffer-Lebensdauer;
- Trusted-Read-Amplification.

**Exit:** Jede Buffergröße, Lebensdauer und Fallbacktilegröße ist berechnet und getestet.

### Gate 6: Minimaler affiner, unpublizierter Prototypkern

- persistenter GPU-Workspace;
- keine Hostrecords;
- keine Hotpath-Allokationen;
- integrierte CFA-Coverage und dichter Footprint;
- vollständige Telemetrie;
- reale Canvasgröße.

**Exit:** Der noch nicht in den produktiven Runner publizierte affine Kern
besteht Oracle-, Support-, Flux-, Speicher- und das vorab festgelegte
Teilzeitgate.

### Gate 7: Transaktion und Resume

- v2-Artefaktschema;
- Checkpointidentitäten;
- Resume-Granularität;
- Atomic-Commit;
- Fehler- und Restartmatrix.

**Exit:** Kein teilweise publizierter oder semantisch nicht gebundener Zustand ist möglich.

### Gate 8: Lokale Warps

- inverse Repräsentation wählen;
- konservative Bounds beweisen;
- Lebensdauer und Speicher messen;
- lokalen realen Datensatz gegen Oracle prüfen.

**Exit:** Lokale Modelle erfüllen dieselben Verträge und Zeitgates wie affine Modelle.

### Gate 9: Streaming-Multiband

- Kalibrierung des in Gate 4 definierten Confidence-Modells;
- Halo- und Ringbuffervertrag;
- GPU-/CPU-Fusion;
- matched-star-Validierung;
- Raw/Uniform-Fallbackvertrag.

**Exit:** Multiband ist fachlich wirksam, erzeugt keine Supportänderung und besteht Qualitätsgates.

### Gate 10: Cutover

- zwei frische affine und zwei frische lokale Produktionsläufe;
- vollständige 600-Frame-Gates;
- Resume- und Fehlerfälle;
- Dokumentation, Schema und Report konsistent.

**Exit:** Erst danach wird der alte Record-/Sort-/Kandidatenpfad entfernt. Er bleibt nicht als langfristiger Alternativmodus bestehen.

## 23. Architekturvergleich

| Bestehender Pfad | V2-Anforderung |
|---|---|
| Source-Scatter erzeugt Hostrecords | direkte bounded Device-Statistik, Gather oder Scatter nach Gate 1 |
| Records GPU→CPU | nur Profil-/Outputdaten GPU→CPU |
| Sortierung pro Frame/Tile | keine Recordsortierung |
| Sortierung über alle Frames pro Pixel | ausgewählter bounded-memory Robustschätzer |
| Kandidatenmatrix skaliert mit Framezahl | Speicher skaliert mit Tilepixeln und fester Schätzerstruktur |
| Tilebreite kollabiert mit N | Tileplan unabhängig von N |
| `cudaMalloc/free` pro Kleinstjob | persistenter Workspace |
| `cudaDeviceSynchronize` pro Job | Stream-Events und berechnete Slotpipeline |
| Coverage als zweiter CFA-Sweep | günstiger Preflight + exakte integrierte Coverage |
| CFA-Support und dichter Footprint vermischt | zwei getrennte Geometriezustände |
| Clipping kann Support löschen | Statistik reduziert Confidence, nicht Source-Support |
| 2×2-Mittel normalisierter Werte + AND | Zähler-/Nenner-/W²-Fold mit Teilflächenvertrag |
| `Q_p90-Q_p50` kann Alpha nullen | validiertes absolutes Quality-/Stability-Modell |
| redundante Reader- und Inhaltsprüfungen | Trusted-Run-Vertrag und Read-Amplification |
| unklarer Resume nach Phasenumbau | versioniertes v2-Artefakt- und Checkpointschema |
| nur relative Skalierungsforderung | absolutes 600-Frame-Produktionsgate |

## 24. Abschlusskriterium

Forward Drizzle v2 gilt als implementierungsreif, wenn Gates 0–5 abgeschlossen sind. Es gilt als AQMH-Ersatz, wenn Gates 6–10 bestanden sind.

Vor Gate 6 müssen insbesondere entschieden sein:

1. Gather oder dichter Device-Scatter;
2. konservative affine Kandidatengrenze;
3. vierstufiger Supportvertrag;
4. Zähler-/Nenner-/W²-Fold;
5. bounded-memory Robustschätzer;
6. exakter Fallbackalgorithmus;
7. Noise-/Gradient-/Registrierungsmodell;
8. FP32/FP64-Vertrag;
9. gemeinsame RAM-/VRAM-/Temp-Formel;
10. Full-Width-/X-Tile- und Read-Amplification-Vertrag.

Die Kernentscheidung bleibt:

> Die bestehende Record-/Sort-/Kandidatenarchitektur wird ersetzt. Welche recordfreie GPU-Geometrie und welcher bounded-memory Robustschätzer verwendet werden, wird nicht vorab behauptet, sondern durch exakte Oracles, adversariale Qualitätsfälle, vollständige Speicherrechnung und Messung auf realer Canvasgröße entschieden.
