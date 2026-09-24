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

### 1.2 Implementierungsstand und Freigabestatus (2026-09-12)

Gates 0–9 sind abgeschlossen. Nach bestandener Gate-10-Produktionsmatrix ist v2
der einzige produktive FORWARD_DRIZZLE-Pfad. Der frühere
`TC_FORWARD_DRIZZLE_V2`-Schalter, der Legacy-Produzentenbranch und der
Legacy-Fusionsfallback wurden entfernt; ein fehlender oder ungültiger v2-Store
scheitert geschlossen. Niedrigstufige Legacy-Algorithmen bleiben ausschließlich
als Dev-/Regression-Oracles erhalten.

| Gate | Vorhandener Stand | Freigabe |
|---:|---|---|
| 0 | kleine deterministische affine Record-Oracles über Scale 1/2, MONO/OSC, alle Bayer-Patterns, zwei Origins, drei Pixfrac-Werte und fünf affine Transformationen; sauber gebundener 60-Frame-M42-Kaltlauf; persistierte Gate-1-Schwellen | **bestanden**; lokale Referenz wird erst in Gate 8 nach Auswahl der lokalen Repräsentation erzeugt und in Gate 10 produktiv abgenommen |
| 1 | CPU-/CUDA-Target-Gather und CUDA-Dense-Scatter; 21 Fälle auf 3840×2160 → 7868×4540, sieben Transformationen und Pixfrac 0,2/0,8/1,0 | **bestanden: Dense Device Scatter ausgewählt**; alle festen Korrektheits-, Flux-, Repeatability- und 0,75-s-Schwellen erfüllt |
| 2 | vierstufiger Support-/Fold-Vertrag, exakter CPU-Fold, CUDA-Paritätsoracle, Flux-/Teilflächenmatrix | **bestanden**; Chunk-/Tile-Grenzen des späteren Batch-Folds bleiben Gate-6-Abnahmekriterium |
| 3 | Produktions-Clip-Oracle, MoM-Kandidaten (K 17/41) und deterministisches Hash-Reservoir; 21-Fall-Adversarialmatrix inkl. Produktions-N=600 | **bestanden: reservoir_sigma_clip (R=64) ausgewählt**; worst-vs-oracle 0,036 bei eingefrorener Schranke 2,0; bit-identisch zum Oracle für N≤64; kein Replay, kein Supportverlust |
| 4 | Sigma-Modell `noise² + (gx²+gy²)·σ_reg² + half²/3`, Confidence `S_c²/(S_c²+C_c)` mit `fallback_n_eff`, Degraded-Zählung, FP64-Akkumulatoren gemessen | **bestanden**; FP32 und Neumaier-FP32 scheitern an der 1e-12-Schranke, FP64 ausgewählt; Provider-Schema und GPU-Confidence-Parität an Gate 7/6 delegiert |
| 5 | exakte overflow-geprüfte RAM-/VRAM-Formel mit gefrorener Rollentabelle (Reservoir 64×24 B, FP64-Confidence), Slot-gebundene Host-Pinning-Lebensdauer, X-Tile-Fallback, RA-Schranke | **bestanden**; Full-Width auf Produktionsgeometrie bis ~2 GiB Device, N-unabhängig; große Halos (≥14 bei 5 GiB) verletzen die konservative RA-Schranke — Gate-9-Konsequenz dokumentiert |
| 6 | persistenter unpublizierter Prototypkern: eine Allokationsphase, asynchrone Stream-Pipeline, bit-exakter Gate-3-Clip-Port | **bestanden**; 0,016 s/Frame auf Produktionsband, 1 Allokation, 0 globale Syncs, 13.090 B/px ≤ Plan |
| 7 | v2-Artefaktschema mit `band_boundary_resume`, Atomic-Commit, fail-closed Inspection | **bestanden**; 21-Fall-Fehlermatrix, resume = memcmp-identisch zum Dauerlauf |
| 8 | kompakte Koeffizienten + On-Device-Inversion, exakte `invert/subdivide`-Ports | **bestanden**; Depth-2/Inversionsstress/Discard-Parität nachgewiesen, 0,091 s/Frame |
| 9 | Profilproduktion U/R/F/M über geteilter Clip-Maske, Quality-Side-Array, Ring-/Halo-Vertrag, CPU-Fusion auf committeten Bändern | **bestanden**; Device-Parität zum Orakel, RA ≤ 1,1 auf Produktionsplan, 19.954 B/px ≤ amendierter Plan, matched-star-Fallbacks |
| 10 | Runner-/Store-/Fusion-Cutover, Host-Fallback und beschleunigter bandbegrenzter Inputpfad | **Produktions-Performance bestanden**; A1/A2/L1/L2 auf CUDA v2: 251,50/431,02/434,45/557,71 s, alle RA-Grenzen ≤ 1,1; A1 scheiterte erst nach FORWARD_DRIZZLE/MULTIBAND in PCC, finale Fehlervertragsabnahme und Legacy-Entfernung offen |

Die affine Gate-0-Referenz ist versioniert in
`docs/forward_drizzle_v2_gate0_affine_reference_2026-09-12.json`. Sie bindet
den frischen Kaltlauf `/media/tc_500/m42-gate0-ref60` an:

- Git-Commit `5c1d977cce4571d928a13c7442f22366bf0fc679`, `git_dirty=false`;
- Build-ID `9be255713401b6585c2bf2fd411da300aa5908710e49bc54454df1bf1345d120`;
- Binary-SHA-256 `d62644c394c8307fa825c26d48ff5f4d6b6ab58df3673b7d37d00d543e48fc8b`;
- Config-SHA-256 `b19e15be463e11ad5a1a798f53e517b5c36bc3faea9285ed335a4e198a34222e`;
- Inputmanifest-SHA-256 `e81c79695fd18b23008f5ff4f72ecbbb3f0dd9533c6722576cbba3920c70088c`;
- 60 Frames, 3840×2160 OSC, internal/output 2/1, Pixfrac 0,8;
- `guard_fallback=true`, 2.276.618 Fallbacks, 0 verworfene Pixel/Kanäle;
- 0 Nonfinite-Pixel in geometrischer Analyse- und Supportfläche;
- 0 verlorene Canvaspixel und 0 Lochkomponenten in der Analysefläche;
- vollständigen erfolgreichen Kaltlauf ohne Resume in 1285,079 s.

Das Referenzartefakt friert außerdem vor dem nächsten Gate-1-Vergleich die
Korrektheits-, Flux-, Support-, Performance-, Speicher- und
Read-Amplification-Schwellen ein. Gate 0 ist damit für die affine
Ausgangsarchitektur geschlossen. Der Lauf enthält
`local_model_samples_total=0`; dies ist kein lokaler Nachweis. Eine lokale
Referenz vor Gate 8 wäre nicht aussagekräftig, weil inverse Repräsentation,
Bounds und Lebensdauer dort erst ausgewählt werden. Gate 8 erzeugt deshalb
die sauber gebundene lokale Referenz; Gate 10 verlangt weiterhin die
vollständige affine und lokale Produktionsabnahme.

Der erste native affine Vergleich auf einem 1920×1080-Sourcefixture ergab:

| Zielcanvas | Gather | Scatter | Parität |
|---|---:|---:|---|
| 1920×1080 | 0,128 s | 0,051–0,060 s | A/B exakt, Support identisch, zwei Scatter-Läufe bytegleich |
| 3840×2160 | 0,384 s | 0,131–0,132 s | A/B exakt, Support identisch, zwei Scatter-Läufe bytegleich |

Diese Messung ist nur explorativ: Sie wurde nicht auf der realen
7868×4540-M42-Canvas, nicht über die vollständige Transform-/Pixfrac-Matrix
und nicht gegen vorab festgelegte Laufzeitschwellen ausgeführt. Sie ist ein
Signal zugunsten des dichten Scatter, aber keine Gate-1-Entscheidung.

Vor weiterer Gate-6-Arbeit ist die Reihenfolge wiederherzustellen:

1. Gate 0 mit reproduzierbarer Evidenz und Schwellen abschließen;
2. Gate 1 auf realer Canvas und vollständiger affiner Matrix entscheiden;
3. Gate 2 vollständig spezifizieren und validieren;
4. Gate 3 als Kandidatenvergleich durchführen;
5. Gate 4 entscheiden;
6. erst daraus Gate 5 und den tatsächlichen Gate-6-Workspace ableiten.

Bestehende taktische Änderungen des alten Pfads (`guard_fallback`, parallele
Hostreduktion, Alpha-Scratch) bleiben davon getrennt. Insbesondere ist
`guard_fallback` standardmäßig `false`; die historische Lochsemantik ist damit
im Produktionsdefault noch nicht beseitigt und die Änderung zählt nicht als
v2-Supportvertrag.

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

**Entschieden in Gate 2** (Spec und Entscheidung versioniert in
`forward_drizzle_v2_gate2_support_fold_spec_2026-09-12.json` und
`forward_drizzle_v2_gate2_support_fold_decision_2026-09-12.json`):

- es gibt keine Mindestflächenabdeckung im Fold: solange `B_p,out > 0` ist,
  gilt `value = A/B` über der vorhandenen Fläche;
- vier unabhängige Flächenfraktionen (Geometry, Source, Estimator, Profile)
  bewahren die verlorene Teilflächeninformation für Confidence und spätere
  Delivery-Entscheidungen;
- die Supportimplikationen `profile => estimator => source => geometry`
  sind Pflicht; Verletzungen sind `invalid_argument`, keine stille
  Hochstufung;
- ein endlicher Teilflächenwert wird nicht gelöscht; eine endgültige
  Delivery-Mindestabdeckung bleibt ein separater, noch zu fassender Vertrag.

Validiert gegen konstante Felder (volle und partielle Fläche), einen
gewichteten linearen Gradienten mit ungleichen Frame- und Profilgewichten,
eine einzelne Punktquelle, alle vier isolierten Supportebenen, ungültige
Implikationen und nichtendliche positive Zähler. Ein einzelnes fehlendes
internes Subpixel verwirft den nativen Pixel nicht.

### 6.3 Fold-Gate

Vor Freigabe müssen gelten:

- konstantes Feld bleibt konstant;
- integrierter Punktquellenflux bleibt innerhalb der definierten Toleranz;
- Chunk-/Tile-Grenzen ändern das Ergebnis nicht;
- CPU und GPU besitzen identische Supportentscheidungen;
- 2/1 erzeugt keine statistisch verursachten Einzelpixellöcher.

## 7. Affine Enumeration: Gather-vs-Scatter-Entscheidung

### 7.1 Entscheidung: Dense Device Scatter

Die v2-Invariante lautet:

```text
keine Hostrecords
keine N-Frame-Sortierung pro Pixel
keine Pixel×Frame-Kandidatenmatrix
```

Gate 1 hat Dense Device Scatter für die affine Enumeration ausgewählt.
Target-Gather bleibt ein unabhängiges Produktionsgrößen-Oracle, wird aber
nicht zum Ausführungskern. Die Entscheidung ist versioniert in:

- `forward_drizzle_v2_gate1_affine_benchmark_spec_2026-09-12.json`;
- `forward_drizzle_v2_gate1_affine_results_2026-09-12.jsonl`;
- `forward_drizzle_v2_gate1_affine_decision_2026-09-12.json`.

Der Scatter schreibt frame-lokale A/B-Planes auf dem Device. Diese Planes
dürfen im späteren Produktionspfad nicht pro Frame zum Host geladen werden;
sie werden in Gate 3/6 auf dem Device in die ausgewählte bounded-memory
Statistik weitergeführt.

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

**Gate-1-Ergebnis (bestanden 2026-09-12):** Alle 21 Fälle erfüllten
Korrektheit und Repeatability mit 0 A/B-Fehler, 0 Supportabweichungen und
identischer positiver Overlapzahl. Konstantes Feld hatte 0 relativen Fehler;
der Punktquellenfehler lag bei `5,24e-10` und damit unter `1e-9`. Dense
Scatter benötigte einschließlich Source-Upload und Kernel zwischen 0,060 und
0,715 s pro Frame; der vorab fixierte Worst-Case-Grenzwert war 0,75 s. Es
gab 0 globale Device-Synchronisationen und 0 Hotpath-Bufferallokationen.
Target-Gather benötigte für denselben Korrektheitsvergleich 1,088 bis 2,194 s.

Die erste explorative Messung hatte fälschlich den vollständigen D2H-Download
der frame-lokalen A/B-Planes zur steady-state Enumeration gezählt. Spec v2
dokumentiert diese Methodenkorrektur vor dem entscheidenden Clean-Build-Lauf;
der numerische Grenzwert 0,75 s blieb unverändert. Die Downloads bleiben nur
für den vollständigen Oraclevergleich im Harness und sind im gewählten
Produktionsdatenfluss verboten.

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
- reproduzierbarer affiner Referenzlauf auf der aktuellen Architektur;
- vollständige Artefakt-/Hardwarebindung;
- kleine vollständige synthetische Oracles;
- begrenzte reale Oracle-Strategie;
- vorab persistierte Gate-1-Abnahmeschwellen.

**Exit (bestanden 2026-09-12):** Die affine Ausgangslage ist über
`m42-gate0-ref60` und
`forward_drizzle_v2_gate0_affine_reference_2026-09-12.json`
reproduzierbar gebunden; `m42-test1` bleibt nur Diagnosebeleg. Die lokale
Referenz wird nach Auswahl der lokalen Repräsentation in Gate 8 erzeugt und
bleibt Bestandteil der Gate-10-Produktionsabnahme.

### Gate 1: Affine Enumeration

- Minkowski-/Bounding-Vertrag formulieren;
- Target-Gather und dichten Device-Scatter prototypisieren;
- gegen Geometrie- und Frame-Kandidaten-Oracle prüfen;
- auf realer Canvasgröße messen.

**Exit (bestanden 2026-09-12):** Dense Device Scatter ist für affine Frames
ausgewählt. Der Clean-Build `e65dbda1` bestand die 21 Fälle der eingefrorenen
Produktionsmatrix und alle Flux-/Support-/Repeatability-/Performance-Gates.
Target-Gather bleibt Oracle. Vollständige Rohdaten und Entscheidung stehen in
den drei Gate-1-Artefakten aus §7.1. Die Atomic-Repeatability ist an die
getestete GTX 1660 Ti gebunden und muss bei einer neuen GPU-Architektur erneut
abgenommen werden.

### Gate 2: Support und Scale-Fold

- vier Supportebenen festlegen;
- Zähler-/Nenner-/W²-Fold definieren;
- Teilflächenregel festlegen;
- konstante Felder, Gradienten und Punktflux validieren.

**Exit (bestanden 2026-09-12):** Vierstufiger Supportvertrag mit verbindlichen
Implikationen, framebezogener B²-Fold mit Kreuztermen, `value = A/B` über
Teilflächen ohne Subpixel-AND und vier unabhängige Flächenfraktionen sind
festgelegt, exakt auf CPU implementiert und durch ein Single-Thread-
CUDA-Paritätsoracle verifiziert. Die Punkte `estimator_b` pro Eintrag,
Produktions-Batch-Fold und eine mögliche Delivery-Mindestabdeckung sind
explizit an Gate 3 beziehungsweise Gate 6 delegiert; alle Gate-2-Testfälle
bestehen (12 Testfälle, 7.415 Assertions).

### Gate 3: Robustschätzer

- vollständiges Frame-Oracle erzeugen;
- Kandidatenverfahren auf adversarialen Fällen vergleichen;
- bounded-memory Verfahren auswählen;
- exakte Fallbackzustandsmaschine spezifizieren.

**Exit (bestanden 2026-09-12):** `reservoir_sigma_clip` mit R=64 ist als
Primärschätzer ausgewählt. Ein einziger Stream-Pass akkumuliert die uniformen
A/B/B²-Summen und füllt ein deterministisches Hash-Reservoir
(`splitmix64(frame_order ^ seed) < ⌊2⁶⁴·R/N⌋`, N = bekannte Stream-Länge aus
dem Samplingplan); beim Fold läuft exakt der Produktions-Sigma-Clip
(gewichteter Median/MAD, asymmetrische 3σ-Grenzen, 3 Pässe) auf höchstens R
Datensätzen. Für N ≤ R ist das Ergebnis bit-identisch zum Voll-Listen-Oracle;
auf der adversarialen 21-Fall-Matrix beträgt die größte Abweichung vom Oracle
0,036 (Schranke 2,0). Alle Median-of-Means-Kandidaten scheitern an
periodischer 30-%-Kontamination (worst 2,32–2,75), der Zweipass-Winsorizer
zusätzlich am Single-Pass-Gebot — beide bleiben als dokumentierte Referenzen
im Code. `estimator_b = source_b` für alle Frames; B/B² decken den
vollständigen Strom ab, Support kann nie durch Robustheit verloren gehen.
Fallbackzustände: `no_source_support`, `too_few_candidates_fallback`,
`primary_reservoir_sigma_clip`; ein degeneriertes MAD kollabiert die
Clip-Grenzen auf den Median statt zu fehlschlagen. Artefakte:
`forward_drizzle_v2_gate3_robust_estimator_spec_2026-09-12.json`,
`..._results_2026-09-12.jsonl`, `..._decision_2026-09-12.json`. Offene
Delegationen: Reservoir-Ebenen (≤ 64×24 B je Pixel/Kanal) gehen in Gate 5 ein;
device-seitige Reservoir-Akkumulation und Per-Pixel-Clip-Kernel sind Gate 6.

### Gate 4: Rauschmodell und Numerik

- Noise-, Gradient-, Sampling- und Registrierungseingänge definieren;
- Sampling-Plan-Schema gegebenenfalls erweitern;
- bounded berechenbare Confidence-Formel und ihre hinreichenden Statistiken
  festlegen;
- FP32/compensated-FP32/FP64 messen;
- CPU/GPU-Toleranzen festlegen.

**Exit (bestanden 2026-09-12):** Das Rauschmodell ist festgelegt als
`sigma² = sigma_noise_f² + (gx² + gy²)·sigma_reg_f² + half²/3`
(`forward_drizzle_v2_sigma2_model`, FP64); nichtendliche oder negative
Eingänge liefern NaN, werden als `conf_degraded` gezählt und tragen
`sigma2 = 0` bei — niemals ein Wurf im Hotpath. `sigma_reg_f` ist eine
providergelieferte isotrope 1σ-Lageunsicherheit in Source-Pixeln;
`registration_residual_factor`/`model_prediction_factor` sind
Qualitätsfaktoren und werden nicht als Varianzen konsumiert. Die
Plan-Schema-Erweiterung (Providerfelder `sigma_noise_f`/`sigma_reg_f` im
persistierten `FrameSamplingTransform`) ist ausdrücklich an Gate 7 delegiert,
weil sie den Resume-Vertrag berührt.

Confidence folgt `S_c²/(S_c² + C_c)` über den effektiven Satz des Schätzers
(`B_c`, `S_c`, `C_c` = Σb, Σb·σ, Σb²·σ²; drei Doubles plus Zähler je
Pixel/Kanal, O(1)). Effektiver Satz je Schätzer: Reservoir = clip-akzeptierte
Reservoir-Member (innerhalb des Clip-Aufrufs), uniform/median = alle gültigen
Kandidaten, winsorized = alle Gruppen, trimmed = überlebende Gruppen,
Fallbacks = voller gültiger Strom. Fehlt jede σ-Information (`C_c = 0`),
gilt `confidence = n_eff/(n_eff+1)` mit Zustand `fallback_n_eff`; ohne
Support `no_source_support` und 0. Confidence ist ein Diagnosescalar und kann
Support/B/B²/n_eff nie verändern — per Test belegt.

Numerik: gemessen auf fünf Akkumulationssequenzen. Plain-FP32 scheitert
deutlich (rel. Fehler bis 0,996 bei Kancellation, 1,8e-5 auf 1e5-Termen),
Neumaier-FP32 scheitert an der 1e-12-Schranke auf dem Langstrom (1,8e-10).
Ausgewählt ist **FP64 für alle pixelweisen Akkumulatoren** (A/B/B² wie
B_c/S_c/C_c), konsistent zu Fold und Clip; Geometrie, Polygonclip und
Reservoir-Clip bleiben FP64. CPU/GPU-Toleranzen: Fold-Parität 1e-12
(gemessen 0), Confidence-Parität 1e-12 reserviert — ein GPU-Confidence-Kernel
ist Gate-6-Batch-Fold-Gegenstand. Artefakte:
`forward_drizzle_v2_gate4_confidence_numerics_spec_2026-09-12.json`,
`..._results_2026-09-12.jsonl`, `..._decision_2026-09-12.json`.
Gate-5-Eingang: drei Confidence-Doubles plus Degraded-Zähler je
Pixel/Kanal; Tests: 15 V2-Fälle / 7.470 Assertions.

### Gate 5: Gemeinsamer Speicherplan

- exakte RAM-/VRAM-/Temp-Formel;
- Full-Width- und X-Tile-Pläne;
- Halo-/Ringbuffer-Lebensdauer;
- Trusted-Read-Amplification.

**Exit (bestanden 2026-09-12):** Die Speicherformel ist gegen die gefrorene
Rollentabelle geschlossen: `4976 B` je OSC-Zielpixel (Reservoir 64×24 B
dominant, dazu FP64-Confidence `B_c/S_c/C_c` + Zähler, Frame-A/B, robuste
A/B/B², Zentrum/Skala, Fold-Staging, Support) plus kanalunabhängige
8 B/Pixel; MONO `1664 B`. Device-Fixkosten sind Slot-förmig
(Source-/Quality-Device-Slots + Reserve), Host-Pinning ist über die
Pipeline-Slotzahlen gebunden — `frame_count` ist reine Provenienz und ändert
weder `tile_cols` noch `band_rows` (40- vs. 600-Frame-Pläne bitgleich
getestet). Gemessen auf Produktionsgeometrie 7868×4540: Full-Width ohne
X-Tiling bis ~2 GiB Device-Budget (126 Zeilen bei 5 GiB, 99 bei 4 GiB,
44 bei 2 GiB; MONO 379); X-Tile-Fallback `tile_cols = ⌊max_pixels /
min_padded⌋` verifiziert. Halo wird als `2h+1` gepolsterte Zeilen
eingepreist; die konservative RA-Obergrenze
`(H + 2h·bänder)/H` macht Halo 14 bei 5 GiB (RA 1,29) und Halo 62
(RA 63, Kern kollabiert auf 2 Zeilen) infeasible — Gate 9 muss entweder die
Bandkanten-Wiederverwendung nach §17.2 in ein reuse-aware RA-Modell
überführen oder Multiband anders fusionieren. Alle Multiplikationen sind
overflow-geprüft; Verletzungen erzeugen einen nicht-feasiblen Plan.
Artefakte:
`forward_drizzle_v2_gate5_memory_plan_spec_2026-09-12.json`,
`..._results_2026-09-12.jsonl`, `..._decision_2026-09-12.json`.
Tests: 15 V2-Fälle / 7.486 Assertions.
**Nachträgliche Amendment (Gate 6):** Reservoir-Record 32 B (sigma2
persistiert bis zur Faltung), Device-Slots 2·R = 128, Confidence-Stream
S_c/C_c, +64 B Ergebnis-Record/Kanal → 12.826 B je OSC-Nativepixel +
internal Frameplanes; die Gate-5-Ergebnisdatei dokumentiert weiterhin die
Vor-Amendment-Messung (4.976 B), die Entscheidungsdatei führt die
Amendment explizit.

### Gate 6: Minimaler affiner, unpublizierter Prototypkern

- persistenter GPU-Workspace;
- keine Hostrecords;
- keine Hotpath-Allokationen;
- integrierte CFA-Coverage und dichter Footprint;
- vollständige Telemetrie;
- reale Canvasgröße.

**Exit (bestanden 2026-09-12):** `ForwardDrizzleV2CudaPrototypeKernel`
implementiert die gefrorene Gates-1–5-Pipeline bandweise auf einem
persistenten Device-Workspace: `reserve()` ist die einzige
Allokationsphase (eine gezählte Allokation), `accumulate_frame()` reiht
Upload, Plane-Clear, `k_scatter_v2` und `k_fold_accumulate_v2` vollständig
asynchron auf dem Workspace-Stream ein (0 globale Synchronisationen, 0
Hotpath-Allokationen), `finalize()` führt den bit-exakten
Gate-3-Clip-Port `k_finalize_v2` aus, synchronisiert den Stream genau
einmal und lädt nur die kompakten `ForwardDrizzleV2PixelResult`-Records.
Der Scatter schreibt `B_geo` vor der Werteprüfung, sodass nichtendliche
Samples ihren Geometrie-Support behalten; Fold, Full-Stream-A/B/B²,
Coverage, Vier-Ebenen-Supportmaske (u16, Scale ≤ 2), Footprint und
gebundenes Hash-Reservoir (Record 32 B mit sigma2; Slots = 2R = 128,
deterministischer `reservoir_overflow_fallback` jenseits der Kappe)
laufen fusioniert pro nativem Pixel.

Nachweis: Small-Canvas-Paritätsmatrix (Scale 1/2 × MONO/OSC × alle vier
Bayer-Patterns × Origins) ist zustands- und support-bitidentisch zum
CPU-Oracle, Werte ≤ 1e-12 relativ; N=80 > R=64 Sampling korrekt;
leeres Band, Einzelframe, Determinismus und Telemetrie verifiziert.
Produktionsband (3934 native Spalten × 59 Zeilen, 60 M42-Affine,
Pixfrac 0,8, OSC): **0,0164 s maximal pro Frame** (Grenze 0,75 s),
3,05 GiB reserviert ≤ Plan-Peak, 13.090 B/Pixel ≤ amendierte
Gate-5-Formel 13.210 B, 0 Reservoir-Überläufe, `dense_overlap` deckt sich
mit dem unabhängigen CPU-Gather-Orakel (0 im oberen Band).

Gate-5-Rollentabelle durch Gate 6 amendiert: Reservoir-Record 32 B
(+sigma2), 2R-Slots, Confidence-Stream auf S_c/C_c reduziert
(B_c = robustes B), +64 B Ergebnis-Record je Kanal.
Artefakte:
`forward_drizzle_v2_gate6_prototype_kernel_spec_2026-09-12.json`,
`..._results_2026-09-12.jsonl`, `..._decision_2026-09-12.json`.

### Gate 7: Transaktion und Resume

- v2-Artefaktschema;
- Checkpointidentitäten;
- Resume-Granularität;
- Atomic-Commit;
- Fehler- und Restartmatrix.

**Exit (bestanden 2026-09-12):** §18.2 ist entschieden: Resume
ausschließlich an atomar committeten Bandgrenzen
(`band_boundary_resume`). Teilbänder werden verworfen und neu gerechnet;
Reservoir-/Akkumulatorzustand ist transient und wird nie persistiert.
`ForwardDrizzleV2StoreWriter` schreibt in eine unpublizierte
`forward_drizzle_v2_generation-*`: `plan.json` (plan_hash über die
kanonische Serialisierung aller Bindungsfelder), `band-%04d.bin`
(selbstbeschreibender 64-B-Header + 64-B-Records), `checkpoint.json`
kontiguous Präfix-Liste mit Bytes+SHA-256 je Band — atomar als **letzter**
Schritt jeder Bandtransaktion aktualisiert. `finish()` verifiziert alle
Artefakte (Größe + SHA-256), wertet die Commit-Gates aus
(`nonfinite_pixels_inside_source_support == 0`, `bands_processed ==
band_count`), schreibt `commit.json` und publiziert `current.json` als
letzten atomaren Schritt. `inspect_forward_drizzle_v2_store` ist
read-only und fail-closed: `complete` / `resumable(k)` / `fresh` /
`corrupt` mit Fehlergrund. Die Fehler-/Restartmatrix (21 Fälle, Tag
`[gate7]`) beweist: Absturz zwischen Bändern setzt exakt am verifizierten
Präfix fort und erzeugt denselben `checkpoint_hash` und memcmp-identische
Artefakte wie der ununterbrochene Lauf; jede Korruption
(Artefakt-Bitflip, Trunkierung, fehlende Datei, nicht-kontigues oder
manipuliertes Checkpoint, fremder Plankontext, Schema-Mismatch,
mehrdeutige Generationen, Zeiger ins Leere) schlägt geschlossen fehl;
kein teilweise publizierter Zustand ist möglich. Artefakte:
`forward_drizzle_v2_gate7_transaction_resume_spec_2026-09-12.json`,
`..._results_2026-09-12.jsonl`, `..._decision_2026-09-12.json`.

### Gate 8: Lokale Warps — **bestanden**

Gewählte Repräsentation: **kompakte Koeffizienten + On-Device-Inversion**.
Der persistierte `SmoothLocalWarpModel`-Satz (4×4 normalisierte
Gauß-Basis, σ=0,28, 8%-Smoothstep-Taper, ~144 B/Frame) wird unverändert
auf das Device übergeben; es wird kein Deformationsgitter materialisiert
und kein zusätzlicher per-Frame-Speicher resident gehalten.

- Inversion: fp32-Fixpunkt `q_{n+1} = u − d(q_n)`, affiner Seed,
  `max_iter=6`, `tol=0,001 px`, Sicherheitsmarge 64 px — exakter Port von
  `invert_local_source_to_canvas`.
- Subdivision: impliziter 21-Knoten-Baum (1+4+16), 3×3-Inversionen pro
  Knoten, Tiefe ≤2, `position_epsilon=0,05` interne px,
  `area_relative_epsilon=0,005` — exakter Port von `subdivide_local`.
- All-or-nothing: jeder Inversions-/Subdivisionsfehler verwirft das
  gesamte Source-Sample inklusive bereits akzeptierter Blätter; ungültige
  Modelle verwerfen pro Sample (CPU-Semantik), schlagen nicht den Call
  fehl.
- Telemetrie: `scalars[2]`/`stats().local_samples_discarded`.
- Parität: diskrete Entscheidungen (Zustände, Contributoren, Masken,
  Discards) exakt; Magnituden ≤1e-5 rel. wegen fp32-Inversionsinterna.

**Implementierungsbefund (behoben):** `native_rows` des Workspace ist die
Bandhöhe, die Inversions-Bounds des Orakels gelten aber für die **volle
Canvas-Höhe** — die erste Übergabe verwarf ~97% der Samples eines Top-Bands.
`ForwardDrizzleV2KernelConfig.canvas_{width,height}_native` trägt jetzt die
volle Canvas-Geometrie (0 = Band deckt das ganze Canvas).

**Geometrie-Einsicht:** Ein canvas-gefittetes 4×4-Modell hat immer
Knotenabstand ~nc/3 (~1311 px) — Tiefe-2-Subdivision ist auf
Produktionsgeometrie nur über komprimierte Modelldomänen erreichbar.
Die Produktionsmatrix deckt sie deshalb mit einer 13×13-px-Domain-Patch
im Band plus einem Max-Iterations-Inversionsstress-Modell ab; strukturelle
Depth-2-Akzeptanz beweist die Kleincanvas-Matrix (kalibriertes
Schachbrett, `max_leaves>4`).

Messung (7868×4540 intern, Band 59 Zeilen, 60 Frames): max 0,091 s/Frame
(Grenze 0,75 s), 1 Allokation, 0 globale Syncs, 13.090 B/Pixel ≤ Plan.
Artefakte:
`forward_drizzle_v2_gate8_local_warp_spec_2026-09-12.json`,
`..._results_2026-09-12.jsonl`, `..._decision_2026-09-12.json`.

### Gate 9: Streaming-Multiband — **bestanden**

**Exit (bestanden 2026-09-12):** Vier Profile (Uniform `b`, Raw `b·g_eff·q_c`,
Fine `b·g_eff·q0⁴`, Medium `b·g_eff·q1²`) werden device-seitig über **einer**
geteilten Gate-3-Clip-Maske reduziert; Qualität geht nie in die
Clip-Entscheidung ein. Der Quality-Fold mittelt pro Kandidat über exakt
die Finite-Source-Population von `b_src`; fehlende Streams falten als 1,0,
nichtpositive/NaN-Werte als 0 (Veto), Artefakt-Präsenz separat über die
Finite-Fläche (`qaf`). Device-Layout: vier f32-Quality-Upload-Ebenen,
fünf f64-Akkumulatorebenen (Qc/Q0/Q1/Qa/Qaf), `float4`-Reservoir-Side-Array
pro Slot (mit den Records mitsortiert) und eine 16-B-Frame-Meta-Tabelle
(`g_eff`, `is_direct`, `residual_factor`) — alles in der einzigen
`reserve()`-Allokation, nur bei `emit_profiles`.

Alpha-Faktoren: `a_separation = clamp01(Gate-4-Confidence)` — eine
vollständig degradierte Sigma-Population ist explizit „keine
Separations-Evidenz" und kollabiert auf 0; `a_artifact` aus gewichtetem
qa-p10 + Smoothstep, `a_registration` aus der Meta-Tabelle; OSC-Aggregation
per Kanal-Minimum; globaler Near-Zero-Fall wird als eigene Diagnose
gemeldet, nie stillschweigend als Erfolg.

Ring/Halo: `ForwardDrizzleV2MultibandRing` erzwingt In-Order-Band-
Finalisierung (Duplikate/außerhalb = Fehler), meldet den Immutabilitäts-
punkt des Fusionskerns und hält residente Bänder ≤ ⌊2·halo/band_rows⌋+2.
Reuse-aware Read-Amplification: die Anwendung liest jede benötigte
Source-Zeile einmal plus Kanten-Halo; der Produktionsplan erreicht
RA ≤ 1,1 (Halo 64), womit der Gate-5-Befund großer Halos aufgelöst ist.

**Fusionsentscheidung:** CPU-Fusion auf committeten Band-Outputs
(`fuse_multiband_streamed`), bit-identisch zur Whole-Frame-Fusion auf
gestützten Pixeln (NaN-bewusster Vergleich); einzige Caveat bleibt die
dokumentierte B3-Flood-Fill-Streifenkante. Fused-Support ⊆ Uniform-/Raw-
Support. Ein Device-Fusions-Port kann in Gate 10 neu entschieden werden,
falls die gemessene Fusionszeit das Budget verletzt.

**Implementierungsbefunde (behoben):**
- Records werden bei Finalize nach `(x, order)` insertion-sortiert — das
  Quality-Side-Array muss mitwandern; die erste Fassung las `resq` im
  sortierten Index und ordnete Quality-Mittel falsch zu.
- `device_bytes_per_native_pixel` zählte Quality-Upload-Ebenen und die
  Meta-Tabelle fälschlich als Per-Pixel-Zustand; sie sind feste
  Pipeline-Slots wie src/sigma2 und werden jetzt abgezogen.
- Spec-Schätzung 14.600 B/px unterschritt das Side-Array (16 B × 128
  Slots = 2048 B/Kanal); bindend ist der amendierte Plan 20.074 B/px,
  gemessen 19.954 B/px.

Matched-star-Validierung läuft unverändert über
`select_reconstruction_candidate`: null effektive Multiband-Sterne
blockieren die Promotion, `multiband == raw` scheitert am
Verbesserungsgate (Raw gewinnt), Raw-Safety-Regression fällt auf Uniform
zurück; Uniform/Raw bleiben immutable Kontrollen an gematchten Positionen.

Nachweis Produktionsgeometrie (60 Frames, OSC, Band 33 Zeilen): max
0,043 s/Frame (Grenze 0,75 s), 1 Allokation, 0 globale Syncs,
19.954 B/px ≤ Plan. Testmatrix: 12 Fälle, 31.984 Assertions (`[gate9]`),
inkl. Device-Parität Scale 1/2 × MONO/OSC, Q-Veto, g_eff≠1,
Overflow-Fallback, Degraded-Sigma-Device-Fall, Ring/RA, Fusionsparität
Levels 1–4 und Memory-Amendment. Artefakte:
`forward_drizzle_v2_gate9_streaming_multiband_spec_2026-09-12.json`,
`..._results_2026-09-12.jsonl`, `..._decision_2026-09-12.json`.

### Gate 10: Cutover — **Produktionspfad auf v2 umgestellt**

Der erste A2-Versuch wurde nach sieben von 72 Bändern abgebrochen. Der Median
lag bei 298,390 s/Band; der vollständige FORWARD_DRIZZLE wäre aus den gemessenen
Intervallen mit 21.484,08 s zu erwarten gewesen. Ursache war nicht der robuste
Kernel selbst, sondern die Produktionsverdrahtung: für jedes Band wurden alle
610 vollständigen Source-/Sigma²-/Q-Ebenen erneut gelesen, expandiert,
übertragen und über alle Source-Pixel gestartet. Die frühere Schranke
`< 0,75 s/(Frame,Band)` begrenzte diesen Bandfaktor nicht und wurde deshalb als
Cutover-Nachweis verworfen.

Der neu geöffnete Performancepfad beseitigt den Faktor strukturell:

- die globale SplitMix-Reservoirauswahl wird vorab berechnet; für A2 werden 58
  statt 128 Slots reserviert, während der historische `>2R`-Overflow-Fallback
  erhalten bleibt;
- Q-Daten werden nur für ausgewählte Frames und als kompakte uint16/Veto-Daten
  gelesen und übertragen; keine Source-aufgelöste Float-Expansion;
- affine Frames verwenden analytische kanonische Source-Zeilenspannen. Source,
  Sigma²-Nachbarschaft und Q-Zellen werden nur für tatsächlich bandfähige
  Samples gelesen; auch Rotationen um etwa 10° bleiben unter RA 1,1;
- Sigma² wird aus einmal gelesenen Sparse-Source-Nachbarschaften erzeugt; ein
  vollständiger Sigma²-Recompute und -Upload pro Band entfällt;
- lokale Frames konsumieren die committeten Geometry-Cache-Leaves direkt;
  FORWARD_DRIZZLE wiederholt keine lokale Inversion/Subdivision;
- ein CPU-/CUDA-Workspace, Stream und Eventset wird über alle Bänder
  wiederverwendet; Host-Provider- und Driver-Hotpaths wachsen nach Pre-Reserve
  nicht mehr;
- der deterministische Device-Plan begrenzt den dynamischen Pixelzustand auf
  2 GiB. Das lässt auf der 6-GiB-GPU Platz für die zusätzlichen festen
  Source-/Sample-/Q-Puffer und verhindert einen stillen CPU-Fallback;
- die Telemetrie trennt Provider-I/O, Q-Zellen, H2D, Device-Zeit, Commit-Zeit,
  tatsächliche Read-/Sample-Amplifikation und Workspace-Reservationen.

Der identische 24-Frame-Realprovider-Vergleich auf M42 ergab:

| Pfad | FORWARD_DRIZZLE |
|---|---:|
| altes v2 | 421,610 s |
| Legacy CUDA | 565,153 s |
| neues v2, kanonische Spans | **28,930 s** |

Damit ist das neue v2 14,57× schneller als das alte v2 und 19,54× schneller
als Legacy. Source-RA = 1,0068, Q-RA = 1,0156 und gestartete
Sample-Amplifikation = 1,0068; beide Hotpath-Allokationszähler und globale
Device-Synchronisationen sind 0, Workspace-Reservationen = 1. Der
`checkpoint_hash` sowie die rekonstruierten R/G/B-Dateien sind gegenüber den
rechteckigen und X-gekachelten optimierten Kontrollpfaden bitidentisch. Die
konservative A2-Projektion beträgt 1.320,01 s und liegt unter der vorab
eingefrorenen 2.148,408-s-Schranke.

Die frische Produktionsmatrix vom 2026-09-14 bestätigt den beschleunigten Pfad:

| Lauf | Modus | FORWARD_DRIZZLE | Source/Q/Sample-RA | Ergebnis |
|---|---|---:|---:|---|
| A1, M31, 645 Frames | affin | 251,50 s | 1,0141 / 1,0317 / 1,0141 | FORWARD_DRIZZLE und MULTIBAND bestanden; PCC-Revalidierung auf unverändertem RGB nach IQR-Guard-Fix bestanden |
| A2, M42, 610 Frames | affin | 431,02 s | 1,0182 / 1,0407 / 1,0182 | `final_image_ready` |
| L1, M42, 610 Frames | lokal | 434,45 s | 1,0944 / 1,0407 / 1,0182 | `final_image_ready` |
| L2, M42, 610 Frames | lokal, p100 | 557,71 s | 1,0989 / 1,0452 / 1,0228 | `final_image_ready` |

Alle vier Läufe verwendeten `cuda_v2` ohne Fallback, genau eine
Workspace-Reservation, keine Host-Hotpath-Allokation und keine globale
Device-Synchronisation. L1/L2 verarbeiteten jeweils 33.177.600 lokale
Modellsamples und lasen 33.812.405 beziehungsweise 33.971.129 committete
Geometry-Cache-Leaves. A2 ist gegenüber der aus dem abgebrochenen Lauf
abgeleiteten 21.484,08-s-Laufzeit reproduzierbar knapp 50× schneller.

Der nachgelagerte A1-PCC-Ausfall wurde unabhängig vom Drizzlepfad reproduziert.
Alle 645 Frames nahmen teil, ASTAP löste das Bild und der Siril-Katalog lieferte
vollständige XP-Spektren. Ein relativer Annulus-Guard
`IQR > 0,35 × Hintergrund` verwarf jedoch niedrige R/B-Hintergründe noch vor
dem robusten Ebenenfit. Nach Entfernung dieses skalenabhängigen Hard-Rejects
besteht PCC auf dem unveränderten A1-RGB mit 524 gematchten und 478 verwendeten
Sternen, Residual-RMS 0,1854 und Condition Number 1,98. Die bestehenden
Support-, Sättigungs-, Background-Safe-, Huber- und Sigma-Clip-Gates bleiben
erhalten.

Nach bestandener Vollsuite wurde der produktive Cutover vollzogen:

- FORWARD_DRIZZLE verwendet v2 ohne Umgebungs- oder Konfigurationsschalter;
- MULTIBAND verlangt den publizierten v2-Store und scheitert bei fehlendem,
  fremdem oder ungültigem Store geschlossen;
- der Legacy-Produzentenbranch und der Legacy-Fusionsfallback sind aus dem
  Runner entfernt;
- der v2-Bandstore bleibt als einziger transaktionaler Resume-Store erhalten;
- niedrigstufige Legacy-Algorithmen verbleiben nur als Dev-/Regression-Oracles
  und sind aus dem produktiven Runner nicht erreichbar.

**Exit:** Der Produktivpfad ist vollständig v2; es besteht kein langfristiger
Legacy-Alternativmodus.

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
