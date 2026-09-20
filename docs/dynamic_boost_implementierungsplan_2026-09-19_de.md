# Dynamic-Boost: Analyse und Implementierungsplan

Stand: 2026-09-19. Ausgangsstand: `CFA-aware-Forward-Drizzle` / `1cdc1333`.
Die vorhandene DWARF-FITS ist ein **positives Machbarkeitsbeispiel für diese
M42-Aufnahme**: schwache Nebelbereiche lassen sich im Endprodukt deutlich
vom unruhigen Hintergrund abheben. Ziel ist, diesen praktischen Nutzen auch
in `tile_compile` zu erreichen. Der genaue Mechanismus der DWARF-Firmware
ist noch unbekannt. Bestehende Runs und ihre Artefakte bleiben unverändert.

## 0. Neuer Befund aus den vorhandenen FITS-Dateien

Die Referenz liegt unter
`/media/data/Astro/DwarfII/Astronomy/DWARF_RAW_TELE_M 42_EXP_10_GAIN_60_2026-02-13-18-41-06-458/stacked-16_M 42_10s60_Astro_20260213-184149374.fits`.
Sie ist ein 16-Bit-RGB-FITS mit 3840 x 2160 Pixeln und `EXPTIME=5190` s.
`runs/m42-c1/artifacts/forward_drizzle.json` nennt **610 verwendete Frames**
und `drizzle_raw` als gewählten Kandidaten. Dessen lineares
`outputs/stacked_rgb.fits` ist ein Float-RGB-FITS mit 4540 x 3360 Pixeln.

Zur Kontrolle wurden beide FITS-Dateien read-only mit derselben einfachen,
jeweils auf das Bild skalierten Asinh-Vorschau dargestellt und anhand von
Sternen registriert. Für die DWARF-zu-`m42-c1`-Vorschauen ergaben sich
208 RANSAC-Inlier aus 238 SIFT-Zuordnungen. Die schwache Nebelmorphologie
ist an denselben Stellen **auch im linearen 610-Frame-Stack von
`tile_compile` sichtbar**. Die DWARF-Darstellung hat sichtbar einen
ruhigeren Hintergrund. Damit ist die zu prüfende Frage konkret: Warum
bleiben die vorhandenen Nebelkonturen im `tile_compile`-Endprodukt schlechter
vom Hintergrund getrennt? Die Antwort lässt sich nicht aus dem sichtbaren
Vergleich allein einer einzelnen Pipeline-Stufe zuordnen.

Eine erste Zahlenprobe verwendet *nur* diese beiden M42-Produkte und den
610-Frame-Run: auf 1200 Pixel Breite verkleinert, DWARF per Stern-Affine
auf `m42-c1` registriert, dieselben Bildrechtecke in beiden Bildern,
Luminanz `0,2126 R + 0,7152 G + 0,0722 B`. Himmelrechteck im gemeinsamen
Raster: x=900..999, y=430..499. Schwache Nebelrechtecke: West
x=360..434/y=465..514 und Nord x=440..509/y=405..449. Pro Rechteck
werden Werte über dem 90. Perzentil für die robuste Median-/MAD-Zahl
ausgeschlossen. Angegeben ist `(Nebelmedian - Himmelsmedian) /
(1,4826 * MAD des Himmels)`; das ist **Darstellungskontrast in dieser
registrierten Vorschau**, keine physikalische Detektionssignifikanz.

| Produkt | West | Nord |
| --- | ---: | ---: |
| DWARF-FITS | 15,2 | 16,8 |
| `m42-c1` linear `stacked_rgb.fits` | 5,3 | 5,5 |
| `m42-c1` `stacked_rgb_pcc.fits` | 5,5 | 5,7 |
| `m42-c1` `stacked_rgb_hms.fits` | 5,4 | 5,6 |

PCC und HMS erhöhen die Trennung in diesen Rechtecken praktisch nicht.
DWARF erreicht im selben Vorschauverfahren ungefähr das Dreifache.
Die Registrierung interpoliert DWARF zusätzlich; PSF, Hintergrundgradient,
Sternreste und verschiedene Belichtungsselektionen sind noch nicht
herausgerechnet. Die Tabelle ist ein **Richtungsbefund**, kein finales Gate.
Der DWARF-Header nennt 5190 s; die genaue verwendete Frame-Auswahl lässt
sich daraus allein nicht ermitteln. Die Analyse der 610-Frame-Serie bleibt
vollständig getrennt von anderen Datensätzen.

**Phasen-Provenienz in `m42-c1`:** Das BGE-Event meldet bei der
Rekonstruktion und der späteren Fortsetzung `auto_detect_no_gradient` und
`status=skipped`; ein BGE-Effekt erklärt die hier gemessene Differenz also
nicht. `stacked_rgb.fits` und `stacked_rgb_solve.fits` sind bytegleich.
`stacked_rgb_pcc.fits` und `stacked_rgb_hms.fits` wurden am 19. September
bei einer Fortsetzung ab BGE neu geschrieben. Die vorhandene `config.yaml`
setzt Luma-/Chroma-Denoise auf `false`, während die gespeicherte
`config_revisions/run_cfg_20260919T062516Z.yaml` beide auf `true` setzt
und `luma_denoise.json`/`chroma_denoise.json` Pässe von der früheren
Verarbeitung enthalten. Diese Artefakte allein beweisen **nicht**, dass
die zuletzt geschriebenen PCC-/HMS-Dateien diese Pässe enthalten.
Vor einer kausalen Aussage über die Denoise-Stufe sind effektive Config,
Zeitpunkt, Eingangsdatei und Output-Hash jeder Fortsetzung zusammenzuführen.

## 1. Was die M42-Analyse belegt und was nicht

Die [M42-Analyse](m42_dynamik_kontrast_analyse_2026-09-19_de.md) misst im
linearen `stacked_rgb.fits` 26,81 ADU Himmel, 29,21 ADU schwachen Nebel,
32,93 ADU helleren Nebel und 1,64 ADU robuste Streuung im Himmel. Die
Differenzen 2,40/1,64 = 1,46 und 6,12/1,64 = 3,73 beschreiben den Kontrast
*eines Pixels gegenüber der Streuung dieses Patches*. Sie sind keine
Signifikanz eines zusammenhängenden Nebelbereichs und keine Aussage über
alle Ortsfrequenzen. Benachbarte Drizzle-Pixel sind korreliert; für eine
Fläche von N Pixeln gilt deshalb nicht automatisch SNR mal Wurzel(N).

Die fünf getesteten Boost-Varianten liefern starke Evidenz gegen die bisher
getesteten skalaren Gain-Funktionen: Hintergrund wird mitverstärkt, Kern
clippt oder Sterne zeigen Ringe. Die DWARF-Datei belegt zugleich, dass
**eine andere Verarbeitungskette den visuellen Trenneffekt erreicht**.
Die weitere Arbeit soll deshalb zuerst diese Verarbeitungseigenschaft
eingrenzen, statt den Effekt grundsätzlich in Frage zu stellen. Ein
Strukturtensor kann gerichtete Kanten erfassen, fügt aber keine Messdaten
hinzu und schützt Sternflügel nicht von selbst.

Die 610 Frames wurden bereits vom Forward-Drizzle-Estimator genutzt. Eine
erneute Mittelung erhält keinen freien Faktor Wurzel(610). Unter bestimmten
Annahmen kann ein Coadd die relevante Information sogar bewahren; [Zackay
und Ofek, 2017](https://arxiv.org/abs/1512.06879) zeigen das für eine
definierte Gaussian- und Hintergrundgrenze. Der geplante Zusatznutzen ist
deshalb **ein framebasiertes Unsicherheits- und Kohärenzsignal für die
Bildschätzung**: getrennte Framegruppen zeigen, welche schwache Struktur
über unabhängige Teilmengen konsistent ist. Der Zielbefund aus der DWARF-
Datei und das Scheitern der bisherigen lokalen Gain-Varianten reichen aus,
um diesen Architekturpfad umzusetzen. Er garantiert noch kein bestimmtes
Qualitätsergebnis; das entscheidet die Abnahme des neuen Produkts.

Die im älteren M42-Dokument genannten 4,9 % und 16,7 % gehören zu den
dortigen Produktvarianten und dürfen nicht ungeprüft auf `m42-c1`
übertragen werden. Eine nahezu unveränderte Streuung nach Pixelmittelung
kann auch von großflächiger Himmelsstruktur,
korreliertem Rauschen, unterschiedlicher Auflösung oder Tonwertabbildung
kommen. Der Vergleich braucht gleiche Registrierung, Pixelgröße, PSF,
linearen Intensitätsbereich und dieselben festen Messflächen. DWARFs
Firmware und ihre Bearbeitung sind nicht vollständig bekannt.

### Entscheidender Codebefund: 610 verarbeitet, 58 ausgewählt

Im tatsächlichen `m42-c1`-Plan stehen `frame_count: 610`, `estimator:
reservoir_sigma_clip` und `reservoir_size: 64`. Der Run wählte
`drizzle_raw`; die Auswahlbegründung war, dass Multiband die geforderte
FWHM-Verbesserung gegenüber Raw nicht erreicht hat. Der CPU-Finalizer
berechnet den robusten Bildwert `ca/cb` aus der akzeptierten Teilmenge des
Reservoirs. Die implementierte Bedingung
`splitmix64(frame_order ^ seed) < floor(2^64 * 64 / 610)` ergibt mit dem
gespeicherten Seed **exakt 58 global ausgewählte Frameordnungen**; pro Pixel können
davon weitere mangels Support oder durch Clipping entfallen. Der
CUDA-Finalizer tut dasselbe. Die Profile
Uniform, Raw, Fine und Medium werden ebenfalls aus diesen gehaltenen
Kandidaten gebildet. Die vollen 610 Frames bestimmen zwar die laufenden
Uniform-Summen und Support-/`n_eff`-Zähler, aber bei erfolgreichem
Reservoir-Pfad nicht den endgültigen Profil-Bildwert. Die als „Uniform“
exportierte Profil-Kontrolle ist damit im Normalfall ebenfalls eine
Reservoir-Kontrolle, nicht der Voll-Frame-Mittelwert.

Der bestehende [Gate-3-Entscheid](forward_drizzle_v2_gates/forward_drizzle_v2_gate3_robust_estimator_decision_2026-09-12.json)
nennt diese statistische Kostenstelle ausdrücklich: bei N > 64 bestimmen
ungefähr 64 von 600 Frames den Wert. Unter der vereinfachenden Annahme
unabhängiger, gleich gewichteter Messungen hätte ein 58-Frame-Mittelwert
gegenüber 610 Frames eine um `sqrt(610/58) = 3,24` höhere Streuung.
Die beobachtete Vorschau-Kontrastlücke liegt in ähnlicher Größenordnung;
dies ist eine mechanistisch plausible Hypothese, kein kausaler Beweis.
Das beweist **nicht**, dass die
gesamte DWARF-Lücke allein dadurch entsteht. Es ist aber ein konkreter
Verlust an Multi-Frame-Information in genau dem Pfad, der für die
610-Frame-M42-Ausgabe gewählt wurde. Die nächste Implementierung setzt
hier an, ohne weitere Bildvergleichsschleife als Vorbedingung.

Die zu erhaltende Gegenleistung der Reservoir-Wahl ist konkret: Im
Gate-3-Fall mit 600 Frames und 30 % periodisch um +8 verschobenen Werten
liegt der Reservoir-Schätzer bei 100,0207, der Voll-Frame-Clip-Oracle bei
99,9852 und das ungeschützte Mittel bei 102,4025. Der saubere
600-Frame-Fall zeigt zugleich die statistischen Kosten: Reservoir 99,9421,
Oracle 100,0039. Ein Voll-Frame-Mittel ohne robustes Frame-Veto ist deshalb
keine Lösung. Diese synthetischen Zahlen belegen den Schätzermechanismus,
nicht die Ursache der gesamten M42-Bilddifferenz.

### Kausaltest 2026-09-20: Reservoir (~58) gegen ungeclippten Voll-Frame-Mittelwert (610)

Aufbau: neuer Vollrun `20260920_091350_7cf52142` mit der Config von `m42-c1`
und nur `drizzle.min_clip_contributors: 100`. Damit greift im Kernel für jedes
Pixel `too_few_candidates_fallback`, und der Wert ist `accA/accB` über alle
610 Frames (`forward_drizzle_v2_cpu.cpp`, Fallback-Zweig). Beide Läufe wählen
`drizzle_raw` und liegen auf demselben 4540x3360-Raster; es gibt keine
Registrierung und keine Interpolation. Gemessen auf `outputs/stacked_rgb.fits`,
Luminanz und Rechtecke wie in §0. Der Voll-Frame-Wert ist nicht geclippt und
kein Produkt; der Test prüft nur die Rauschskalierung.

| Messung | Reservoir | Voll (610) |
| --- | ---: | ---: |
| Himmel-Sigma, volle Auflösung | 1,85 ADU | 0,54 ADU |
| Nebel/Himmel West / Nord, volle Auflösung | 1,74 / 1,77 | 5,25 / 5,77 |
| Nebel/Himmel West / Nord, 4x4 gemittelt (Maßstab von §0) | 5,1 / 5,3 | 9,8 / 11,0 |
| DWARF (§0, gleicher Maßstab) | 15,2 / 16,8 | |

Befund: Die Himmelstreuung sinkt um den Faktor 3,44 (erwartet
sqrt(610/58) = 3,24); die Nebelhelligkeit über dem Himmel bleibt gleich. Die
Frame-Auswahl ist damit die Hauptursache des feinskaligen Rauschens. Im
Vorschau-Maßstab verdoppelt sich der Kontrast, erreicht DWARF aber nicht
(Restfaktor etwa 1,5). Auf gröberer Skala sinkt die Streuung nur um das
2,1-Fache (4x4) und bei 16x16 kaum noch; dort begrenzt korreliertes Rauschen
oder großräumige Himmelsstruktur. Der Himmelmedian der Vollrun liegt bei
23,4 statt 25,3 ADU (Ursache ungeklärt; die Differenzen bleiben davon
unberührt). Grenzen: zwei Rechtecke, ein Lauf, unclipped. Schlussfolgerung:
P1 ist als Hebel belegt, verspricht aber allein kein DWARF-Niveau; P2 bleibt
für den Restfaktor relevant.

**Nachprüfung (2026-09-20).**

- *Himmelmedian-Offset:* Die Differenz ist ein nahezu konstanter Versatz von
  etwa -2,2 (G-Raw) über die gesamte Stütze, keine Struktur. Er stammt aus dem
  **bestehenden** Reservoir-Clip, nicht aus dem Voll-Mittelwert: die Grenzen
  sind `median - 2*MAD` und `median + 4*MAD` mit der *rohen* MAD (nicht mal
  1,4826), vier Durchgänge, in `forward_drizzle_v2.cpp` (robust-Reduktion).
  Das ist asymmetrisch (etwa -1,35 sigma / +2,7 sigma) und schneidet die untere
  Flanke stärker. Eine Gauss-Simulation mit n=58 ergibt einen Aufwärts-Bias von
  0,26 sigma_frame; gemessen sind etwa 2 ADU bei sigma_frame um 14 ADU (0,14),
  gleiche Richtung und Größenordnung, nicht quantitativ deckungsgleich
  (reale Verteilungen sind nicht gaussisch). Dieselbe Simulation zeigt, dass
  der Clip-Schätzer bei n=58 eine Streuung von 0,203 sigma_frame hat, gegenüber
  0,131 für das einfache 58-Frame-Mittel. Die Clip-Effizienz kostet also
  zusätzlich etwa den Faktor 1,5. Folge für P1: Der Bias bleibt, wenn die
  Pilotgrenzen unverändert auf alle Frames angewandt werden, und ist bei
  Photometrie und Hintergrundmodell zu beachten. Ein symmetrischer Clip wäre
  eine eigene Vertragsänderung.
- *Weitere Himmelboxen:* Streuungsverhältnis Reservoir zu Voll 3,60 (Box am
  Bildzentrum), 3,26 und 3,39 (zwei weitere Boxen); eine vierte Box liegt
  außerhalb der Stütze.
- *Ausreißer im Voll-Mittelwert:* Die 9x9-geglättete Differenz zeigt 8 positive
  und 41 negative zusammenhängende Flecken über 200 Pixel bei 6 sigma. Sie
  liegen nicht als Linienstruktur vor; ob es Sternflügel oder Spuren sind,
  ist nicht einzeln geprüft. Der ungeclippte Mittelwert bleibt deshalb kein
  Produkt.
- *Produkt nach PCC und HMS:* Der Kontrast überlebt die Downstream-Stufen.
  `stacked_rgb_hms.fits`, 4x4-Maßstab, Nebel/Himmel West/Nord: Reservoir
  5,20 / 5,44, Voll 10,84 / 12,22 (PCC: 5,23 / 5,47 gegen 10,53 / 11,87). Die
  Schwellen in §5 gelten damit auch für das angezeigte HMS-Produkt.
- *Simulation gegen Messung:* Aus der Simulation folgt für den Clip-Schätzer
  ein Streuungsverhältnis von etwa 3,24 x 1,55 = 5,0 zwischen Reservoir und
  Vollmittel, gemessen sind 3,3-3,6. Die Lücke ist plausibel der Anteil, der
  nicht mit der Frame-Zahl sinkt (korreliertes/festes Rauschen, vgl.
  16x16-Ergebnis); sie ist nicht nachgemessen.
- *Q-Kosten (Schätzung aus den Phasenzählern des Vollruns):* In
  FORWARD_DRIZZLE werden Q-Daten für 3143 von 33526 (Frame, Band)-Paaren
  verarbeitet (9,4 %): 5,81 GB hochgeladen, `provider_quality_seconds` 19-21 s.
  Für alle Paare ist das etwa Faktor 10,7: rund 62 GB Upload und etwa
  +190-210 s Provider-Q-Zeit, bei linearer Skalierung. Die Q-Karten selbst
  entstehen bereits für alle 610 Frames in SOURCE_QUALITY_MAPS (491 s);
  neu wäre nur Lesen und Hochladen. Kernelzeit dominiert FORWARD_DRIZZLE
  (325 s von 383 s) und ist davon nicht betroffen. Erwartete Zusatzkosten
  damit grob 8-10 % der Kette; das ist eine Extrapolation, keine Messung.
- *Eingefrorene Pilot-Grenzen (Simulation und Kerneltest, 2026-09-20):* Der
  Nutzen von P1 hängt stark von der Clip-Konfiguration ab. Simulation
  (Gauss, N=610, Pilot 58, vier Durchgänge, rohe MAD): Reservoir-Streuung 0,204
  bei `2/4`, Voll-Frame-Wert mit eingefrorenen Grenzen 0,149 (Faktor nur 1,37;
  der Bias von 0,29 bleibt); bei `3/3` 0,167 gegen 0,090 (1,86); bei `4/4`
  0,136 gegen 0,049 (2,79, Bias 0). Zum Vergleich: der ungeclippte Mittelwert
  liegt bei 0,040. Ursache: die Grenzen stammen aus nur 58 Stichproben; ihre
  Schwankung verschiebt den Bias einer asymmetrischen Akzeptanzregion und
  sinkt nicht mit mehr Frames. Folge: mit den Produktionswerten `2/4` erreicht
  P1 nur einen kleinen Teil des in §1 gemessenen Faktors 3,4; der Voll-Frame-
  Modus verlangt weite, symmetrische Grenzen (etwa `4/4`, Schutz nur gegen
  Ausreißer). Das ist eine Konfigurationsbedingung des Modus und in der
  Abnahme zu prüfen; der Run `20260920_105804_99e6a998` (Reservoir mit `3/3`)
  misst den Bias-Anteil am realen M42-Stack.
- *Laufzeit-Basis:* Die Phasensummen sind bereits ohne jeden P1-Zusatz an der
  harten Grenze: `m42-c1` 2986 s, Vollrun 2464 s (SAMPLING_GEOMETRY 1116 s,
  SOURCE_QUALITY_MAPS 491 s, FORWARD_DRIZZLE 383 s). Das Budget von 2400 s hat
  damit **keinen Spielraum** für zusätzliche Q-Lese- und Uploadarbeit.

### Umsetzungsstand P1 (CPU-Oracle) und Realtest, 2026-09-20

Implementiert ist der Pilot-plus-Voll-Frame-Schätzer im CPU-Kernel
(`reconstruction.drizzle.full_frame_estimator`, Default aus; Plan-Estimator
`reservoir_pilot_full_frame`, Plan-Hash gebunden). Der Treiber liefert die
Pilotframes zuerst und ruft `end_pilot()` auf; danach wird jeder Frame gegen
die eingefrorenen Pilotgrenzen geprüft (mit Kanalkonsens wie bisher) und in
die vier Profilsummen einbezogen. Fallbacks: `no_bounds` (zu wenig
Pilotbeiträge, alter Wert bleibt) und `degenerate_pilot` (MAD gleich null,
Pilotwert bleibt). Der Modus läuft nur auf der CPU; das Device meldet
`full_frame_estimator has no device implementation`. Tests: vier neue
Kerneltests (Gleichheit bei N <= 64, Rauschboden, Kontaminationsschutz,
Stream-Vertrag), komplette Suite 471 Fälle grün. Der Clip-Helfer wurde
refaktoriert, die bestehende v2-Suite (82 Fälle, inklusive CUDA-Parität)
bleibt grün.

Realtest `20260920_114702_282591ae` (M42, 610 Frames, Clip `4/4`,
`full_frame_estimator: true`, `drizzle_raw` gewählt), gemessen wie in §1:

| Produkt | Himmel-Sigma (1x1) | Nebel/Himmel W/N, 4x4, linear | dito HMS |
| --- | ---: | ---: | ---: |
| Reservoir `2/4` (`m42-c1`) | 1,85 | 5,09 / 5,33 | 5,20 / 5,44 |
| Reservoir `3/3` | 1,64 | 5,37 / 5,69 | 5,53 / 5,87 |
| ungeclippter Mittelwert 610 | 0,538 | 9,81 / 11,02 | 10,84 / 12,22 |
| **Pilot + Voll-Frame, `4/4`** | **0,568** | **12,31 / 13,64** | **12,64 / 14,04** |
| DWARF (§0) | | 15,2 / 16,8 | |

Der neue Modus übertrifft den ungeclippten Mittelwert (Ausreißerschutz und
q-gewichtete Profile) und erreicht etwa 83 % des DWARF-Kontrasts im
Vorschau-Maßstab (Restfaktor 1,2 statt 1,5). Abgelehnt werden 5,0 % der
Beiträge (610,2 Mio von 12,16 Mrd); 816 896 Pixelkanäle sind
`degenerate_pilot`, 1 569 503 `no_bounds`.

Kosten: SOURCE_QUALITY_MAPS 539 s unverändert; in FORWARD_DRIZZLE stieg die
Q-Verarbeitung erwartungsgemäß auf 35 030 (Frame, Band)-Paare, 63,7 GB
Upload und 226,6 s Provider-Q-Zeit (Schätzung 62 GB und 190-210 s). Die
Kernelzeit auf der CPU ist der Blocker: FORWARD_DRIZZLE dauert 9632 s gegen 383 s
auf CUDA, die Kette 12 056 s. Das 10 %-Gate ist erst nach der CUDA-Portierung
prüfbar.

Prüfung der Schwellen aus §5 (gegen den Reservoir-Lauf `m42-c1`):

| Schwelle | Ergebnis | Bewertung |
| --- | --- | --- |
| Himmel-Sigma-Verhältnis >= 2,5 | 3,26 | erfüllt |
| Nebel/Himmel 4x4 >= 9,0 | 12,3 / 13,6 (HMS 12,6 / 14,0) | erfüllt |
| Nebelhelligkeit +-15 % | -10,7 % West, -5,2 % Nord | erfüllt |
| Stern-FWHM (Halbflussradius) +2 % | Verhältnis 0,9997 (136 Sterne) | erfüllt |
| Stern-Fluss +-1 % | -7,8 % (136 Sterne) | **nicht erfüllt, Referenz fragwürdig** |
| Kern-Clipping (HMS, Anteil >= 0,999) | 1,7e-5 gegen 1,3e-6 | formal höher, beide vernachlässigbar |
| Nullhimmel, Gate-3-Regression | Kerneltests grün; Realdaten siehe unten | erfüllt |

Nullhimmel (Realdaten): In sechs vollständig gestützten Boxen mit dem
niedrigsten Himmelsmedian, Sterne mit einer 25x25-Maske ausgeschlossen, zählt
der 9x9-geglättete pilot44-Stack 74 positive und 0 negative zusammenhängende
Flecken über 5 sigma; Reservoir `2/4` hat +102/-6, der ungeclippte Mittelwert
+40/-46. 99 % der 74 Flecken liegen an derselben Stelle (z > 2,5) auch im
Reservoir- und im ungeclippten Stack. Der neue Modus erzeugt also keine
Struktur, die die anderen Stacks nicht ebenfalls zeigen. Auf grober Skala
(16x16, Box im Himmel) liegt der Boden bei 0,154 gegen 0,198 (ungeclippt) und
0,248 (Reservoir); die Streuung sinkt weiter mit der Binning-Größe, das
korrelierte Rauschen ist damit im neuen Stack kleiner als im ungeclippten.

Laufzeit-Gate, Überlegung vor der Portierung: Die Device-Pipeline arbeitet
mit Doppelpuffer-Slots und Events pro Frame; die Provider-Arbeit für Frame
N+1 überlappt die Device-Arbeit für Frame N (Wall 383 s bei 325 s Kernelzeit
plus 83 s Provider-/Commit-Zeit im Vollrun). Die zusätzlichen 207 s
Q-Lesezeit (Host, 226,6 s statt 19,1 s) können damit teilweise hinter der
Kernelzeit verschwinden; sie kämen voll dazu, wenn die Provider-Zeit die
Kernelzeit übersteigt. Zusätzlich steigt die Kernelzeit durch die
Q-Scatter-Ebenen für alle Frames statt 9,4 %, ihr Umfang ist nicht
bekannt. Die Grenzen (+8,4 % ohne Überlappung, Q-Kernelzeit nicht
eingerechnet) lassen beide Ausgänge zu; entschieden wird erst nach der
Portierung durch Messung.

Zum Sternfluss: Der Reservoir-Clip `2/4` ist nicht der unverzerrte
Bezug. Sein Aufwärts-Bias hängt vom Streuungsniveau des Pixels ab und hebt
Quellen an: Reservoir `3/3` liegt bei 0,936, ungeclippt bei 0,884, der neue
Modus bei 0,922 des `2/4`-Flusses; gegen Reservoir `3/3` beträgt der Unterschied
nur -1,6 %. Der Stern-Fluss-Test muss deshalb gegen eine unverzerrte
Referenz (symmetrischer Clip) formuliert werden. Ungeklärt: Der neue Modus
liegt +3,7 % über dem ungeclippten Mittelwert (verschiedene Gewichtung,
q-gewichtetes Raw-Profil gegen unpondertes Mittel); das ist nicht nachgemessen.

### CUDA-Portierung und Kontrolllauf, 2026-09-20

Der Modus läuft jetzt auch auf CUDA: Fold-Kernel mit Voll-Frame-Zweig,
Pilotbarriere `end_pilot()` über die vorhandenen SFR-Kernel (Bounds-Ausgabe
im Build-Kernel, danach `k_full_seed`) und `k_full_apply` beim Finalize. Der
Treiber erzwingt keine CPU-Ausführung mehr. Tests: fünf Full-Frame-Fälle
(inklusive CUDA-gegen-CPU-Parität mit identischen Zuständen, Zählern und
Werten bis 1e-9), v2-Suite 83 Fälle grün.

Realdaten: CPU-Lauf `20260920_114702_282591ae` und CUDA-Lauf
`20260920_165534_959962ce` liefern **bitidentische** lineare Stacks (maximale
Abweichung 0) und identische Zähler (11 552 106 014 akzeptiert, 610 188 598
verworfen, 816 896 `degenerate_pilot`, 1 569 503 `no_bounds`).

Sauberer Kontrolllauf `20260920_160554_030f8007` (Reservoir, Clip `4/4`,
derselbe Code, nur `full_frame_estimator: false`); damit ändert sich gegen den
neuen Modus nur die Frame-Zahl:

| Produkt | Himmel-Sigma (1x1) | Nebel/Himmel W/N (4x4, linear) | dito HMS |
| --- | ---: | ---: | ---: |
| Reservoir `2/4` (`m42-c1`) | 1,852 | 5,09 / 5,33 | 5,20 / 5,44 |
| Reservoir `4/4` (Kontrolle) | 1,474 | 5,86 / 6,26 | 5,94 / 6,36 |
| Pilot + Voll-Frame `4/4` (CPU = CUDA) | 0,568 | 12,31 / 13,64 | 12,64 / 14,04 |

Durch die Frame-Zahl allein sinkt das Himmel-Sigma um den Faktor 2,60 und
der Kontrast im Vorschau-Maßstab verdoppelt sich (HMS 5,94/6,36 auf
12,64/14,04). Die Schwelle Sigma-Verhältnis >= 2,5 ist gegen die
Kontrolle mit 2,60 knapp erfüllt (gegen `2/4`: 3,26).

Stern-Referenz (139 Sterne, gegen die Kontrolle `4/4`): Fluss Median 0,981
(p16 0,960, p84 1,000), Halbflussradius 0,996. Der `2/4`-Lauf liegt bei 1,066
des Kontrollflusses und bestätigt den Bias des asymmetrischen Clips. Der
Sternfluss des neuen Modus liegt damit 1,9 % unter der Kontrolle, die
Schwelle +-1 % ist knapp verfehlt; der Halbflussradius (FWHM-Ersatz) ist
unauffällig. Ursache nicht isoliert; plausibel ist, dass die frozen 4/4-Grenzen
bei schiefen (Poisson-)Verteilungen an den hellen Pixeln einen etwas anderen
Schnittanteil haben als die Reservoir-Grenzen; ein Gewichtungsunterschied
scheidet aus, beide Läufe nutzen dasselbe q-gewichtete Raw-Profil.

Laufzeit (dieselbe Hardware, Kontrolle und neuer Modus nacheinander, ohne
weitere Last): FORWARD_DRIZZLE 475 s gegen 853 s (+378 s), MULTIBAND 225 s
gegen 307 s (+82 s), SOURCE_QUALITY_MAPS 536 s gegen 606 s (P1-unabhängig,
Streuung), Phasensumme 2886 s gegen 3408 s (+18 %). Dem Modus zuzurechnen sind
FORWARD_DRIZZLE und plausibel MULTIBAND: +378 s (+13,1 %) bis +460 s (+15,9 %).
Aufschlüsselung FORWARD_DRIZZLE: Kernelzeit 334 s auf 581 s (+247 s,
Q-Scatter für alle Frames), Provider-Q-Zeit 18 s auf 232 s (+213 s), Upload
11 s auf 21 s; 35 030 statt 3 143 Q-Frames. Provider (rund 400 s) und Kernel
(581 s) überlappen nur teilweise (Wall 853 s gegen 981 s Summe). Das
10 %-Gate ist damit **nicht erfüllt**. Hebel: die Provider-Q-Arbeit (Host,
seriell) in einem eigenen Thread mit doppelt gepufferten Hostpuffern hinter den
Kernel legen (bis etwa 230 s), oder das Gate anheben. Der Modus bleibt bis
zur Entscheidung Opt-in (Default aus).

Aufbewahrung: Von den Testläufen vom 20.09. bleiben für Nachmessungen nur
`logs/`, die JSON-Artefakte und `outputs/stacked_rgb.fits` sowie
`stacked_rgb_hms.fits` erhalten (`20260920_091350_7cf52142`,
`20260920_105804_99e6a998`, `20260920_114702_282591ae`,
`20260920_160554_030f8007`); Caches, kalibrierte Frames, Geometrie und Stores
wurden entfernt. `20260920_165534_959962ce` (CUDA-Lauf) ist vollständig und
resumierbar erhalten.

Pinned Staging (2026-09-20): `cudaMemcpyAsync` aus den pageable
Provider-Vektoren blockierte den Host hinter den bereits eingereihten Kerneln.
Der Samples-Pfad kopiert jetzt in zwei page-locked Slots (Freigabe über das
Upload-Event des Vor-Vor-Frames; bei fehlgeschlagener Pinned-Allokation gilt
der alte Weg). Ergebnis per Resume ab FORWARD_DRIZZLE auf dem CUDA-Lauf:
FORWARD_DRIZZLE 853 s auf 781 s, `stacked_rgb.fits` und `stacked_rgb_hms.fits`
bitidentisch zur Referenz, v2-Suite (83 Fälle) grün. Die Wall-Zeit liegt jetzt
knapp über der reinen Device-Kernelzeit (575 s gegen 334 s in der Kontrolle);
die Provider-Arbeit (ca. 335 s) ist weitgehend verdeckt. Das 10 %-Gate bleibt
verfehlt: FORWARD_DRIZZLE +306 s (+10,6 % der Kontrollkette), mit MULTIBAND
(292 s gegen 225 s) etwa +13 %. Der verbleibende Hebel ist die Device-
Kernelzeit der Q-Scatter-Ebenen für alle Frames (+241 s) und die MULTIBAND-
Zeit, nicht der Host. Hinweis zur Messung: Ein Resume ab FORWARD_DRIZZLE
verwendet einen vollständigen Store wieder (Bänder `reused`); für eine
Neurechnung muss `artifacts/forward_drizzle_v2` entfernt werden.

M31-Bestätigung (2026-09-20, 645 Frames, Läufe `20260920_192029_aeae0417`
Voll-Frame und `20260920_195715_74d67b31` Kontrolle Reservoir `4/4`, dazu
`m31-c1` mit `2/4`), gemessen auf dem PCC- und HMS-Produkt (`m31-c1` hat kein
lineares `stacked_rgb.fits` mehr), Regionen: drei Himmelsboxen und die
äußeren Scheibenarme (Nordost, Südwest):

| Produkt | Himmel-Sigma (1x1, PCC) | Arm NE / SW (4x4, HMS) |
| --- | ---: | ---: |
| Reservoir `2/4` (`m31-c1`) | 2,819 | 2,86 / 4,00 |
| Reservoir `4/4` (Kontrolle) | 2,256 | 4,17 / 3,98 |
| Pilot + Voll-Frame `4/4` | 0,949 | 8,35 / 7,96 |

Die Frame-Zahl allein senkt das Sigma um 2,38 und verdoppelt den Kontrast
(gegen `2/4`: 2,97 bzw. 2,9-fach). Sternfluss (248 Sterne) gegen die
Kontrolle: Median 0,996 (p16 0,977, p84 1,014), Halbflussradius 0,987; der
Reservoir-Lauf `2/4` liegt bei 1,053 (Clip-Bias). Damit ist die Schwelle
+-1 % hier erfüllt, bei M42 (-1,9 %) knapp verfehlt. Nullhimmel (sechs Boxen
mit dem niedrigsten Himmelsmedian, Sterne maskiert): Voll-Frame 29 positive
und 1 negative kohärente Flecken über 5 sigma gegen 135/0 (Kontrolle) und
154/0 (`2/4`); alle 29 liegen an derselben Stelle auch in beiden anderen
Stacks. Kosten: FORWARD_DRIZZLE 291 s auf 488 s (+68 %), Phasensumme 1800 s
auf 2113 s; dem Modus zuzurechnen sind FORWARD_DRIZZLE +197 s (+10,9 %) und
MULTIBAND +29 s, zusammen +12,6 %. Das 10 %-Gate bleibt knapp verfehlt.

### P2-Bewertung: `luma_denoise` auf dem neuen M42-Stack, 2026-09-20

Test auf einer schlanken Kopie des CUDA-Laufs (`p2_m42_base`, Resume ab
ASTROMETRY, nur `luma_denoise` geändert; die Basisvariante ohne Denoise
reproduziert `stacked_rgb_hms.fits` bitidentisch). `luma_denoise` wirkt auf das
lineare RGB vor BGE/PCC/HMS; das Ausgabe-`stacked_rgb.fits` bleibt unverändert,
gemessen wird deshalb auf `stacked_rgb_pcc.fits` (linear) und
`stacked_rgb_hms.fits`. Varianten: A = Wavelet (Levels 3, Schwelle 1,5) +
`extended_source_protection` (`luma_sigma` 2,5, `dilate_px` 30), Bilateral aus;
B = A + Bilateral (1,5 / 2); C = stärker (Blend 1,0, Levels 4, Schwelle 2,5,
Bilateral 2,5 / 3).

| Variante | Himmel-Sigma PCC (1x1) | Nebel/Himmel W/N, PCC, 4x4 | dito HMS | Kernbereich Laplace-Varianz PCC / HMS |
| --- | ---: | ---: | ---: | ---: |
| Basis (kein Denoise) | 0,585 | 12,5 / 13,9 | 12,6 / 14,0 | 1,00 / 1,00 |
| A Wavelet + ESP | 0,257 (x0,44) | 16,6 / 18,6 | 16,7 / 18,8 | 1,00 / 0,88 |
| B + Bilateral | 0,251 (x0,43) | 16,7 / 18,7 | 17,0 / 19,0 | 1,00 / 0,77 |
| C stärker | 0,255 (x0,44) | 15,7 / 17,7 | 16,1 / 18,1 | 1,00 / 0,71 |
| DWARF (§0) | | | 15,2 / 16,8 | |

Sternfluss (143 Sterne, PCC linear) und Halbflussradius gegen die Basis:
1,000 in allen Varianten (Median, p16/p84 innerhalb 0,4 %); Nebelhelligkeit über
Himmel bleibt erhalten (PCC 2,99/3,36 gegen 3,02/3,35). Nullhimmel (PCC, sechs
Boxen, Sterne maskiert): Basis +73/-0 kohärente Flecken über 5 sigma, A +3/-0,
B +10/-0; alle 3 Flecken von A sind auch in der Basis vorhanden (z > 2,5).

Bewertung: Variante A (Standard-Wavelet plus `extended_source_protection`,
Bilateral aus) ist der Sweet Spot. Das Himmelsrauschen sinkt auf 44 %, der
Kontrast im Vorschau-Maßstab steigt im HMS-Produkt auf 16,7/18,8, also über
den DWARF-Werten von 15,2/16,8, ohne Sternfluss oder -größe zu ändern und ohne
neue Himmelsstruktur. Bilateral bringt nur +1,5 % Kontrast, senkt aber die
HMS-Laplace-Varianz weiter (0,77) und ist nicht zu empfehlen; stärkere
Parameter (C) sind schlechter. Einschränkungen: Der Laplace-Wert im HMS-Kern
enthält auch Rauschen und ist kein reines Detailmaß; der Kontrast im
4x4-Maßstab profitiert davon, dass das Denoise korreliertes Rauschen glättet,
er ist keine Detektionssignifikanz; die DWARF-Zahlen stammen aus einer
interpolierten Registrierung (§0). M31 (kompaktes Objekt) ist damit noch nicht
geprüft. Empfohlener Block für M42:

```yaml
luma_denoise:
  enabled: true
  extended_source_protection: {enabled: true, luma_sigma: 2.5, dilate_px: 30}
  bilateral: {enabled: false}
```

M31-Bestätigung P2 (Kopie `p2_m31_base` des Voll-Frame-Laufs, Resume ab
ASTROMETRY, Variante A mit dem Block aus `m31_dwarf2_full_frame_luma.example.yaml`;
Basis ohne Denoise reproduziert `stacked_rgb_hms.fits` bitidentisch): Himmel-Sigma
PCC 0,949 auf 0,422 (x0,44); Nebel/Himmel der Außenarme NE/SW (4x4) PCC
8,00/7,56 auf 11,55/10,81, HMS 8,35/7,96 auf 11,76/11,19; Armhelligkeit über
dem Himmel erhalten (PCC 3,387/3,171 gegen 3,398/3,211); Kern-Laplace-Varianz
0,93 (PCC) und 0,92 (HMS); Sternfluss (247 Sterne) 1,0008, Halbflussradius
1,000; Nullhimmel 0 gegen 29 kohärente Flecken über 5 sigma (keine neue
Struktur). Damit gilt Variante A auf beiden Zielen (M42, M31); Beispielprofile
`m42_dwarf2_full_frame_luma.example.yaml` und `m31_dwarf2_full_frame_luma.example.yaml`
enthalten sie mit kommentiertem `luma_denoise`-Block. Die Defaults in
`tile_compile.yaml` bleiben unverändert (`luma_denoise.enabled: false`).

Offen: Kanalweise Sternfluss gegen einen unverzerrten Bezug, Nullhimmel an
weiteren Objekten (M31), Runner-/GUI-Sichtbarkeit der Zähler.

## 2. Ziel und Grenzen

Ziel ist **die im DWARF-FITS demonstrierte Trennung schwacher M42-Nebelbereiche
vom Hintergrund mit der `tile_compile`-Kette reproduzierbar zu erreichen**.
Ein ruhiger Hintergrund allein reicht nicht; die registrierten Nebelkonturen
müssen erhalten bleiben. Der wissenschaftlich lineare Stack bleibt als
unveränderte Referenz verfügbar. Ein Anzeigeprodukt darf bewusst geglättet
werden, muss aber eindeutig als solches gekennzeichnet sein.

Die Pipeline behält genau eine Rekonstruktionsmethode (CFA Forward Drizzle +
Multiband) und deren Auswahl zwischen Uniform, Raw und Multiband. Kein neuer
Top-Level-`method`-Schalter, keine Änderung der bestehenden Auswahlgates
durch ein nachgelagertes Anzeigeprodukt. Die Defaults der aktuellen Pipeline
bleiben bis zu bestandener Abnahme unverändert.

## 3. Eingriffspunkte im aktuellen Code

| Bereich | Ist-Zustand | Möglicher Eingriff |
| --- | --- | --- |
| `tile_compile_cpp/src/reconstruction/forward_drizzle_v2_production.cpp` | Der Run-Plan fixiert `reservoir_size=64` bei 610 Frames; die Hash-Auswahl liefert hier 58. Qualitätskarten werden im Provider nur für die gewählten Reservoir-Frames gelesen. | Die ausgewählten Pilot-Frames deterministisch zuerst liefern, danach die übrigen Frames je einmal. Qualitätskarten für alle Frames bereitstellen; Lese-/Upload-Verstärkung und absoluten Laufzeitverbrauch erfassen. |
| `tile_compile_cpp/src/reconstruction/forward_drizzle_v2_cpu.cpp`, `forward_drizzle_v2_driver.cpp`, `forward_drizzle_cuda_device.cu`, v2-Store | Der endgültige Raw-Wert und alle Profile werden aus den hier höchstens 58 gehaltenen Kandidaten berechnet. Volle `B/B2`-Summen und `n_eff` zählen dagegen alle Frames. | Aus dem Pilot robuste Clip-Grenzen bestimmen, danach alle geeigneten Framebeiträge genau einmal in die vier Profile einrechnen. Akzeptierte Gewichte, `n_eff` und Konfidenz passend zum neuen Wert führen; optional zwei Gruppen-Summen für die spätere Rauschführung. Plan-/Store-Version, Speicherrechnung und CPU/CUDA-Parität ändern. |
| `tile_compile_cpp/apps/runner_forward_drizzle.cpp` | Wählt Uniform/Raw/Multiband anhand fester Sterne und schreibt `reconstructed_*`, danach kanonische lineare `stacked_rgb.fits`. | Alte Reservoir-Ausgabe als unveränderte Kontrollspur behalten; neue Voll-Frame-Kandidaten durch dieselben Auswahlgates schicken. Gruppenbilder nur als Evidenzkarten veröffentlichen. |
| `tile_compile_cpp/apps/runner_downstream.cpp` | Optionales `luma_denoise` auf linearem RGB vor BGE/PCC/HMS; HMS verarbeitet `stacked_rgb_pcc.fits`. | Nach PCC einen separaten linearen Dynamic-Boost-Kandidaten aus finalem RGB und Gruppen-Evidenz berechnen, darauf HMS für eine eigene Vorschau anwenden; kanonisches PCC/HMS unberührt lassen. |
| `tile_compile_cpp/src/reconstruction/luma_denoise.cpp`, `tile_compile_cpp/src/image/hypermetric_stretch.cpp` | Räumliche Wavelet-/Bilateral-Stufe und HMS mit optionalen, derzeit nicht empfohlenen Boosts. Die Schutzmaske reduziert den tatsächlichen Denoise-Anteil auch innerhalb ausgedehnter Quellen (`amount_map = blend_amount * (1 - luma_guard_strength * protect)`). | Neue mehrskalige Schätzung als eigenes Modul; Rauschen **innerhalb** schwacher Nebelregionen behandeln. HMS wiederverwenden, `wavelet.boost` und `local_contrast` nicht als Ersatz hochdrehen. |

Der Cache unterstützt bereits begrenzte Rechteck- und Zeilenintervall-Lesezugriffe
(`normalized_source_cache.cpp`), garantiert im Trusted-Run-Modell aber keine
Inhaltsprüfung gleich großer Umschreibungen. Ein neues Resume-Produkt muss an
die bestehenden Manifest-/Plan-Identitäten und seine eigenen Parameter und
Eingangsartefakte gebunden sein. `outputs/stacked_rgb_pcc.fits` ist der
dokumentierte lineare HMS-Resume-Eingang. Seine Semantik darf nicht still
geändert werden.

## 4. Verbindliche Implementierungsfolge

Die Ausgangsentscheidung ist getroffen: Die 610-Frame-M42-Serie und die
DWARF-FITS zeigen die sichtbare Lücke; fünf lokale Gain-Varianten sind
fehlgeschlagen; der produktive Raw-Wert nutzt bei N=610 höchstens die
58 ausgewählten Reservoir-Frames. Weitere Registrierungs-, Crop-, Kurven- oder
PSF-Vergleiche sind **keine Voraussetzung** für den folgenden Eingriff.
Sie bleiben nur Messmittel für die Abnahme eines tatsächlich neuen Produkts.

### P1: Voll-Frame-Schätzer mit Reservoir als Pilot

1. Die bisherige deterministische Reservoir-Auswahl (Zielgröße 64,
   für `m42-c1` tatsächlich 58) als **Pilot** beibehalten,
   samt originalem `frame_order`, Seed und identischer CFA-/Geometrie-
   Behandlung. Im neuen Planmodus diese Pilot-Frames zuerst liefern, in
   ihrer ursprünglichen Reihenfolge; danach alle übrigen Frames ebenfalls
   genau einmal. Die alte Plan-/Store-Version und ihr Output bleiben als
   Kontrollpfad lesbar. Ein neuer `estimator`-Vertrag und Plan-Hash binden
   die Reihenfolge und die geänderten Reduktionsregeln.
2. Im Band-Driver nach dem letzten Pilot-Frame eine definierte Barriere
   einführen. Dort aus dem Pilot pro Pixel/Kanal die vorhandene iterative
   Median-/MAD-Sigma-Clip-Entscheidung samt finalen unteren/oberen Grenzen
   berechnen und für den Rest des Bandes vorhalten.
   Die akzeptierten Pilot-Beiträge in den neuen Summen halten. Für jeden
   folgenden Frame den *bereits gefalteten* Kandidaten `x=a/b` gegen diese
   Grenzen prüfen und nur akzeptierte Beiträge addieren. Das ist ein
   Einmal-Durchlauf über jeden Frame, kein erneutes Einlesen der Serie.
   Bei zu wenig Pilot-Support oder degenerierter Skala gelten explizite
   Fallback-Zustände; kein unmarkiertes Wechseln zu einer anderen Statistik.
   Diese Barriere braucht eine neue Kernel-API: Der heutige Driver ruft
   `finalize()` erst nach allen Frames auf. Der Frame-Hash und die
   `frame_order`-Indizes bleiben auf die ursprüngliche Stream-Reihenfolge
   bezogen, obwohl der neue Driver Pilot-Frames zuerst an den Provider gibt.
3. Für Uniform, Raw, Fine und Medium getrennt `sum(w*x)`, `sum(w)` und
   `sum(w^2)` über **alle akzeptierten** Beiträge führen. Die vorhandenen
   Qualitätsgewichte `g_eff`, `q`, `q0`, `q1` behalten ihre Definition.
   `value`, `weight_sum`, `n_eff` und Konfidenz müssen aus derselben
   akzeptierten Menge stammen. Geometrie-, Quell-, Estimator- und
   Profil-Support bleiben getrennt; keine Unterstützung wird erfunden.
   Zusätzlich `candidate_count`, `pilot_count`, `full_accepted_count`,
   `full_rejected_count` und Fallback-Grund je Band aggregieren.
4. Qualitätskarten sind heute nur für Pilot-Frames im Provider geladen.
   `forward_drizzle_v2_production.cpp` muss sie für alle Frames liefern,
   deren Beitrag ein Profil beeinflusst. Exakte Quell- und Q-Lesebytes,
   dekomprimierte Zellen, Upload-Bytes und physische I/O erfassen; die
   bestehenden gepackten Lesewege und wiederverwendeten Puffer verwenden.
   Kein O(N mal Bildfläche)-Cache und keine ungebundene Hotpath-Allokation.
5. CPU-Oracle zuerst in `forward_drizzle_v2_cpu.cpp`, danach derselbe
   Zustandsautomat in `forward_drizzle_cuda_device.cu`. Die neue
   Speicherrechnung in `forward_drizzle_v2_production.cpp` und Driver
   muss Clip-Grenzen, vier Profil-Summen und optionale Gruppen-Summen
   vor `reserve()` berücksichtigen. Gerätefehler starten die betroffene
   Phase über den getesteten CPU-Fallback neu. Keine globalen
   CPU/GPU-Toleranzen lockern, um abweichende Clip-Entscheidungen zu maskieren.
6. Die Band-Transaktion (`forward_drizzle_v2_store`) um neue Record-/
   Diagnosefelder versionieren. Ein Band wird erst nach vollständiger
   Prüfung atomar committed. Resume verwirft unvollständige Bänder und
   verlangt passende Input-, Config-, Pilot- und Estimator-Hashes.
   Bestehende v1-Stores bleiben lesbar, werden nicht als neuer Vertrag
   ausgegeben.

**Mathematischer Vertrag:** Für jeden akzeptierten Framebeitrag f wird
`w_f >= 0` und `x_f` aus dem bestehenden Fold benutzt. Das neue Profil hat
`X = sum_f(w_f*x_f) / sum_f(w_f)` und
`n_eff = (sum_f w_f)^2 / sum_f(w_f^2)`. Die Pilot-Frames bestimmen die
robusten Grenzen, aber nicht mehr allein den Bildwert. Der alte Raw-Wert
wird im Diagnostikpfad mit ausgegeben, damit die Größe des Eingriffs
sichtbar bleibt. Die Reservoir-Zielgröße 64 ist damit kein versteckter
Belichtungs-Deckel mehr.

**Randfälle und Stop-Regeln für P1:** Eine MAD von null darf weder einen
unmarkierten Voll-Frame-Durchlass noch eine willkürliche Epsilon-Weitung
auslösen. Für diesen Zustand gilt zunächst der bisherige Pilotwert als
expliziter `degenerate_pilot_fallback`; im Fall identischer 100er-Werte
plus eines 150er-Ausreißers bleibt der Wert damit exakt 100. Ein Pixel
mit zu wenig gültigen Pilotbeiträgen verwendet den dokumentierten
`too_few_candidates_fallback` und weist aus, dass der Voll-Frame-Wert dort
nicht robust ist. Ob eine varianzgestützte Untergrenze für die Clipbreite
diese Pixel sicher verbessern kann, ist eine eigene Vertragsänderung mit
separater Ausreißerabnahme. Bei nichtdegeneriertem Pilot bleiben die
Grenzen für den Rest des Bandes eingefroren; `full_accepted_count` zeigt,
wie viel der 610er Serie den Wert tatsächlich bestimmt. Die Gate-3-Matrix
ist hier eine Regression gegen kontaminierte Frames, kein erneuter
Entscheidungstest für das sichtbare DWARF-Ziel.

Eine zweite Lösung bleibt als Fallback-Architektur vorgesehen, falls der
Pilot-vorweg-Ansatz die Robustheitsgates
verfehlt: pro begrenzter Zielkachel die gefalteten Kandidaten aller Frames
in einer transaktionalen Nebenablage materialisieren, dort den vorhandenen
iterativen Voll-Frame-Clip exakt auswerten und danach die vier Profile aus
den akzeptierten Kandidaten bilden. Das braucht O(N mal Kachelfläche)
temporäre Daten statt des heutigen O(Kachelfläche)-Reservoirs; die
Kachelgröße muss deshalb aus Speicher- und I/O-Budget abgeleitet werden.
Es benötigt zusätzliche Schreib-/Lesevorgänge, aber keinen erneuten
Quellframe-Lauf. Die historische Gate-3-Ablehnung des anderen
Zwei-Pass-Winsorized-Schätzers bewertet diesen exakten Clip-Vertrag nicht.
Keine der beiden Varianten wird allein aufgrund synthetischer Tests als
gelöst ausgegeben.

### P2: Zeitliche Evidenz für den Nebel-/Rausch-Schätzer

Im selben akzeptierten Durchlauf zwei deterministisch balancierte
Framegruppen je Profil führen, je Kanal `sum(w*x)` und `sum(w)`.
Die Gruppenzuordnung wird vor dem Lauf aus Frame-Identität,
Zeitblock und `g_eff` festgelegt und gehasht; jede Gruppe soll die gesamte
Aufnahme zeitlich abdecken. Die Gruppensummen sind zusätzliche Statistik,
kein zweiter Bildverarbeitungslauf. Ihre beiden Bilder, Gewichte und
Support-Karten werden bandweise in einer eigenen versionierten
Nebenablage veröffentlicht. Speicher und Store bleiben O(Bandfläche),
unabhängig von 610.

Die Gruppen-Differenz schätzt vor allem nicht gemeinsame Störungen.
Registrierungsfehler, Halos, Hintergrundmodell-Fehler oder ein in beiden
Gruppen gleiches Muster können darin verschwinden. Deshalb darf die
Kohärenzkarte weder allein ein Nebel-Label erzeugen noch eine schwache
Struktur ohne gemeinsame Morphologie in den beiden Gruppen verstärken.
Mit nur zwei Gruppen ist ihre Varianzschätzung zudem unsicher; die
Rauschmodell-/Kovarianzkarte und Nullhimmel-Kontrollen bleiben Teil des
Operators.

Der Runner muss die Gruppen der **tatsächlich ausgewählten** Variante
zuordnen. Bei Raw oder Uniform ist das der jeweilige Profilquotient; bei
Multiband müssen beide Gruppen mit denselben, aus dem vollständigen Stack
bestimmten Alpha-/Fusionsfeldern kombiniert werden. Die Identität
`I_full = (W_A I_A + W_B I_B)/(W_A + W_B)` gilt vor der Fusion für jedes
Profil und wird bandweise geprüft. BGE-/PCC-Operationen, die vor dem
Dynamic-Boost-Anzeigeprodukt liegen, müssen auf Vollbild und Gruppen mit
denselben gespeicherten Parametern angewandt werden; gruppeneigene
Hintergrund- oder Farbfit-Parameter würden die Differenz verfälschen.
Fehlt diese Zuordnung oder eine der Gruppen, ist das Anzeigeprodukt
nicht anwendbar. Die kanonischen Produkte bleiben davon unabhängig.

Ein neues Modul unter `tile_compile_cpp/src/reconstruction/` verarbeitet
anschließend den gewählten **linearen** Rekonstruktionskandidaten und die
zugehörigen beiden Gruppenbilder. Pro Skala liefern Gruppendifferenz und
Rauschmodell die Unsicherheit; gemeinsam gerichtete Koeffizienten
liefern Evidenz für beständige Struktur. Schwache, nicht gestützte
Detailkoeffizienten werden geschrumpft; großflächige Nebelstruktur wird
über ihren räumlichen Zusammenhang geschätzt. Sternkerne und PSF-Flügel
kommen aus dem unveränderten linearen Kandidaten. Die Nebelregion darf
nicht als Ganzes von der Rauschunterdrückung ausgenommen werden.
Der Operator ist eine explizit bearbeitete Anzeigevariante mit
Signifikanz-, Schutz- und Fallback-Karten, kein neuer Rekonstruktionsmodus.

P2 verwendet zunächst eine einfache, dokumentierte mehrskalige
Schrumpfungsregel mit korrelationsbewusster Varianz. Konkret: Auf jeder
Skala werden Koeffizienten des vollständigen Bildes und beider Gruppen
mit derselben Analyse gewonnen. Die Differenz der Gruppen kalibriert die
Streu- und Kovarianzkarte aus Nullhimmelbereichen; der erwartete
Rauschanteil wird im Detailkoeffizienten des Vollbildes verkleinert,
während die niedrigste Frequenzebene, der großräumige Nebelfluss und
validierte Stern-PSF-Bereiche identisch zum Eingang bleiben. Der
Schrumpfungsfaktor liegt in `[0,1]`, ist an Rauschmodell und
Gruppen-Konsistenz gebunden und wird mit jedem Output als Karte
geschrieben. Keine Differenzkarte darf allein positives Nebelsignal
erzeugen. Ein Strukturtensor wird nur ergänzt, wenn der CPU-Referenzoperator
an echten Konturen
nachweislich Quer-Glättung oder Halos zeigt; er ersetzt weder die
Gruppen-Evidenz noch die Stern-PSF-Maske. `luma_denoise.wavelet.boost`
und `hypermetric_stretch.local_contrast` werden nicht als Abkürzung
aktiviert.

### P3: Downstream-Produkt und Konfigurationsvertrag

`runner_forward_drizzle.cpp` schreibt weiterhin Raw-/Uniform-/Multiband-
Kontrollen und wählt den finalen Kandidaten mit den bestehenden
Sicherheitsgates. Diese Gates werden mit dem neuen Voll-Frame-Schätzer
neu geprüft, aber nicht still umgangen. `runner_downstream.cpp` erzeugt
neben dem kanonischen linearen PCC-Input eine separat benannte lineare
Dynamic-Boost-Variante und wendet HMS auf **diese** Variante mit eigenem
Output-Namen an. Der kanonische `stacked_rgb_pcc.fits`-Resume-Eingang
behält seine Bedeutung. Das alte HMS-Produkt bleibt als Kontrolle erhalten.

Neue Felder werden erst mit der funktionierenden Implementierung in
C++-Config, Parser/Writer, Validierung, JSON-/YAML-Schema, Beispielprofilen
und DE/EN-Konfigurationsdokumentation angelegt; dafür gilt
`.devin/skills/update-param-doc/SKILL.md`. Default ist aus, solange die
Abnahme nicht bestanden ist. Lauf- und Resume-Artefakte benennen eindeutig
Estimator-Version, Gruppe/Support, Eingangs- und Config-Hash,
Kandidatenwahl, Denoise-Anteil, Clip-/Fallback-Zähler und Ausgabepfad.

## 5. Abnahme des neuen Verfahrens

Die Tests bestimmen, ob **diese Implementierung** korrekt und nützlich ist.
Sie wiederholen nicht die bereits abgeschlossene Entscheidung, einen
Multi-Frame-Pfad zu bauen.

- **Reduktionsvertrag:** Synthetische 40-/64-/610-Frame-Felder mit bekanntem
  Signal, reinem Gaussian-Rauschen, zeitlich gruppierten Ausreißern,
  kosmischen Treffern, wechselnder Qualität, No-Support und CFA-Mustern.
  Der neue Wert muss alle akzeptierten Frames einbeziehen; Profil-
  Gewichte, `n_eff`, Konfidenz und Support müssen konsistent sein.
  Der historische Gate-3-Kontaminationsfall darf nicht durch einen
  Mittelwert über Ausreißer regressieren. CPU und CUDA müssen dieselben
  diskreten Clip-/Fallback-Entscheidungen treffen.
- **Bildprodukt:** Auf der 610-Frame-M42-Serie den bestehenden
  Reservoir-Output, den Voll-Frame-Kandidaten und das zusätzlich geführte
  Anzeigeprodukt getrennt beurteilen. Die bekannten schwachen Nebelkonturen
  und der ruhige DWARF-Hintergrund sind das konkrete Ziel. Gemessen werden
  Nebel-zu-Himmel-Trennung, korrelierte Rauschleistung, Kern-Clipping,
  Stern-FWHM, Fluss/Farbe und radiale Halo-Profile auf festen Positionen.
  Zusätzlich Nullhimmel und weitere Objekte gegen erfundene Struktur prüfen.
- **Produktionskosten:** Die vollständige 600-Frame-Kette bis HMS darf auf
  festgelegter Hardware gegenüber dem P1-freien Lauf höchstens 10 % länger
  dauern (ersetzt die frühere absolute Grenze von 2400 s, siehe unten).
  `m42-c1` meldet allein für `SAMPLING_GEOMETRY`,
  `SOURCE_QUALITY_MAPS` und `FORWARD_DRIZZLE` 1222, 595 und 533 s;
  zusätzliche Q-Lese-/Upload-Arbeit braucht deshalb eine absolute
  End-to-End-Messung. Eingriffspunkte sind Geometrie-Cache-Zugriffe,
  Q-Map-Lese-/Upload-Pfad und Band-Driver, mit Zählern für logische
  Leseverstärkung, physische I/O, Kernelzeit und Peak-RSS/VRAM.
  Eine nur relative oder 40-/100-Frame-Messung reicht nicht.
- **Resume und Ausgabe:** Abbruch vor Band-Commit, korrupter Store,
  geänderte Config oder fehlende Gruppenkarte müssen fail-closed sein.
  Bei deaktivierter Funktion reproduziert der bisherige Pfad seine
  Bildwerte und Kandidatenwahl. Keine alte Run-Datei wird überschrieben.

**Vorläufige Abnahmeschwellen (vor Implementierung festgelegt, aus dem
Kausaltest in §1; alle auf denselben M42-Rechtecken und demselben Raster wie
dort gemessen):**

| Größe | Schwelle | Referenz |
| --- | --- | --- |
| Himmel-Sigma, Vollauflösung, Verhältnis Reservoir/neu | >= 2,5 | ungeclippter Mittelwert: 3,3-3,6 |
| Nebel/Himmel West und Nord, 4x4-Maßstab | >= 9,0 | Reservoir 5,1/5,3; ungeclippt 9,8/11,0 |
| Nebelhelligkeit über Himmel (Differenz) | +-15 % zum Reservoir-Lauf | ungeclippt: -12 % West |
| Stern-FWHM (median, feste Sterne) | nicht mehr als +2 % gegenüber Reservoir | |
| Stern-Fluss, feste Sterne | +-1 % gegenüber Reservoir | |
| Kern-Clipping (Weißanteil) | nicht höher als Reservoir | 0,00 % |
| Nullhimmel | keine kohärente Struktur über 6 sigma (9x9) in einer Box ohne Nebel | |
| Gate-3-Kontaminationsfall | Reservoir-Ergebnis +-0,05 | 100,0207 |
| Laufzeit gesamt | Zusatz durch P1 <= 10 % gegenüber dem P1-freien Lauf derselben Hardware | Vollrun 2464 s; Extrapolation +8-10 % |

Entscheidung des Nutzers (2026-09-20): Die absolute Grenze von 2400 s wird
durch ein relatives Gate ersetzt, weil schon die P1-freie Basis darüber liegt
(2464 s Vollrun, 2986 s `m42-c1`). Überschreitet P1 die 10 % Zusatzzeit und
lässt sich das durch gepackte Q-Lesewege nicht unterschreiten, wird die
Pilot-vorweg-Variante nicht freigegeben und die Kachel-Nebenablage geprüft.
Der Absatz zur Produktionskosten-Abnahme weiter oben ist in diesem Sinn zu
lesen. Bei den Sternschwellen gelten zusätzlich die vorhandenen
Validierungsgates unverändert.

Ein eindrucksvoller M42-Crop allein reicht für die Aktivierung des Defaults
nicht. Wenn P1 die Lücke bereits schließt, bleibt P2 optionales
Anzeigeprodukt; wenn P1 nicht reicht, ist P2 der nächste festgelegte
Eingriff. Beide Eingriffe sind unabhängig von weiteren Vergleichen der
alten Ausgaben implementierbar.

## Quellen für die methodische Einordnung

- M42-Messungen und dokumentierte Fehlversuche: [lokale M42-Analyse](m42_dynamik_kontrast_analyse_2026-09-19_de.md).
- Der heutige Drizzle-Ansatz und sein Gewichtungsprinzip: [Fruchter und Hook,
  2002](https://arxiv.org/abs/astro-ph/9808087).
- Grenzen der Behauptung, Rohframes enthielten stets zusätzliche nutzbare
  Information gegenüber einem Coadd: [Zackay und Ofek,
  2017](https://arxiv.org/abs/1512.06879).
- Struktur-Tensor als Richtungsinformation für anisotrope Diffusion:
  [Weickert, 1999](https://www.mia.uni-saarland.de/weickert/publications.shtml).
