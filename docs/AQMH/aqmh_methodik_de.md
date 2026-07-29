# Adaptive Quality Mask Harvesting (AQMH) - Methodik v0.1.0

**Status:** Explorative Spezifikation - noch nicht normativ  
**Version:** v0.1.0 (2026-06-05)  
**Zuletzt ueberarbeitet:** 2026-06-05  
**Bezug zum Kernsystem:** Eigenstaendige Rekonstruktionsmethode; kann gemeinsame Preprocessing-Infrastruktur wiederverwenden

---

## 0. Motivation und Ziel

### 0.1 Motivation

AQMH ist eine eigenstaendige, qualitaetskartenbasierte Stacking-Methode. Sie basiert auf der Beobachtung, dass lokale Bildqualitaet haeufig raeumlich heterogen ist: eine Satellitenspur, eine Wolkenkante, ein Registrierungsrest oder ein Hot-Pixel-Cluster kann nur einen kleinen Bereich eines registrierten Frames beeintraechtigen, waehrend der restliche Frame weiterhin nutzbar bleibt.

Jede Methode, die einem grossen raeumlichen Block nur einen lokalen skalaren Qualitaetswert zuweist, hat zwei strukturelle Grenzen:

1. **Intra-regionale Heterogenitaet:** Ein kleiner kontaminierter Bereich kann die Qualitaet beeinflussen, die einem wesentlich groesseren Bereich zugewiesen wird.

2. **Blockgrenzen-Diskontinuitaeten:** Wenn Gewichte innerhalb grober raeumlicher Bloecke konstant sind, kann die Rekonstruktionsqualitaet an Blockgrenzen springen.

### 0.2 Ziel von AQMH

Adaptive Quality Mask Harvesting berechnet fuer jeden Frame ein **kontinuierliches pixelweises Qualitaetsgewichtsfeld** `Q_map_{f,c}(x,y)`. Die Rekonstruktion fuehrt danach einen **pixelweisen qualitaetsgewichteten Mittelwert** aus, der ausschliesslich AQMH-Gewichte verwendet: Jeder Ausgabepixel wird aus den Frame-Samples an derselben Pixelposition rekonstruiert, gewichtet mit dem zugehoerigen AQMH-Qualitaetswert.

Kernziele:

1. Den nutzbaren Anteil jedes Frames auf Pixelebene extrahieren.
2. Raeumliche Blockgrenzen-Diskontinuitaeten im finalen Stack vermeiden.
3. Lokale Qualitaetsheterogenitaet ueber eine Multi-Scale-Analysepyramide modellieren.
4. Deterministische gewichtete Mittelwertrekonstruktion, Canvas-Ausschluss und Nicht-Halluzinations-Invarianten erhalten.
5. Unabhaengig von Classic Tile Compile funktionieren. Classic-Ausgaben duerfen nur als externe Vergleichsbaselines dienen, nicht als AQMH-Eingaben oder Fallbacks.

### 0.3 Unabhaengigkeit und gemeinsame Infrastruktur

AQMH ist eine **eigenstaendige Rekonstruktionsmethode**. Sie darf gemeinsame Pipeline-Infrastruktur wiederverwenden, aber ihr Qualitaetsmodell und ihre Rekonstruktionsgewichte werden nicht aus Classic-Tile-Compile-Local-/Tile-Metriken abgeleitet.

Gemeinsame Infrastruktur kann umfassen:

- Input-Scan und Frame-Auswahl
- Kalibrierung und Registrierung/Prewarping
- globale photometrische Normalisierung
- Common-Overlap-/Canvas-Valid-Maske
- Run-Management, Logging, Artefakte, Reports und UI-Anbindung

Der AQMH-Algorithmus selbst besteht aus:

- Berechnung dichter AQMH-Qualitaetskarten
- pixelweiser AQMH-gewichteter Rekonstruktion
- AQMH-Diagnostik und optionaler Regionsextraktion

Classic Tile Compile und AQMH muessen unabhaengig voneinander ausfuehrbar sein. Das Aktivieren oder Deaktivieren einer Methode darf die mathematische Definition der anderen Methode nicht veraendern.

---

## 1. Prinzipien und Definitionen

### 1.1 Physikalisches Ziel

Die Methode modelliert pixelweise Beobachtungsqualitaet als Produkt zweier trennbarer Komponenten:

- **Frame-Level-Qualitaet:** der globale atmosphaerische Zustand von Frame `f`, erfasst durch `G_{f,c}`. `G_{f,c}` ist eine AQMH-Eingabe, abgeleitet aus gemeinsamen globalen Frame-Diagnostiken und Normalisierung; es ist keine Classic-Tile-/Local-Metrik.
- **Raeumliches Qualitaetsfeld:** die kontinuierliche Qualitaetsverteilung innerhalb des Frames, erfasst durch `Q_map_{f,c}(x,y)`.

Das effektive Pixelgewicht lautet:

`W_{f,c}^{aqmh}(x,y) = G_{f,c} * Q_map_{f,c}(x,y)`

### 1.2 Invarianten (bindend)

Die folgenden Invarianten sind fuer die AQMH-Rekonstruktion bindend. Der optionale Cherry-Pick-Modus ist eine explizite Opt-in-Abweichung und wird separat in Abschnitt 5.3 geregelt.

1. **Keine Frame-Selektion:** Ganze Frames duerfen nicht aufgrund ihrer Qualitaet entfernt werden.
2. **Bedingte photometrische Linearitaet:** Sobald die deterministischen Gewichte berechnet wurden, bleibt die finale Rekonstruktion `R(p) = sum_f w_f(p) * I_f(p) / sum_f w_f(p)` mit `w_f(p) >= 0`. AQMH darf keine nichtlinearen Intensitaetstransformationen auf Samples anwenden, die in den Akkumulator eingehen.
3. **Determinismus:** Alle Qualitaetskartenberechnungen muessen deterministisch und reproduzierbar sein.
4. **Canvas-Ausschluss:** Canvas-ungueltige Pixel werden aus allen AQMH-Akkumulatoren und Statistiken ausgeschlossen. Sie werden nur in finalen Ausgabearrays als null/unsupported geschrieben.
5. **Keine Halluzination:** AQMH gibt nur Gewichte und Masken aus. Es erzeugt oder prognostiziert keine Pixelintensitaeten.

### 1.3 Notation

- `f` Frame-Index
- `c` Kanalindex
- `(x, y)` Pixelkoordinaten im registrierten Canvas
- `Q_map_{f,c}(x,y)` pixelweises Qualitaetsfeld, `in [0, 1]`
- `D_s` Downscale-Faktor auf Pyramidenskala `s`
- `P` konfigurierte maximale Anzahl von Pyramidenskalen
- `S_actual` geordnete Menge tatsaechlich berechneter Pyramidenskalen nach der Small-Image-Omission-Regel
- `P_actual = |S_actual|` tatsaechliche Anzahl berechneter Pyramidenskalen fuer die Fusion
- `R_s` raeumlicher Radius des lokalen Analysefensters auf Skala `s`
- `Psi_s(x,y)` Qualitaetsbeitrag auf Skala `s`
- `W_{f,c}^{aqmh}(x,y)` effektives AQMH-Pixelgewicht
- `B_s(x,y; R)` maskierter lokaler Hintergrundoperator auf Skala `s`

### 1.4 Konvention fuer deterministische Statistik

Alle Mediane, MADs und Quantile in AQMH werden nur ueber endliche Werte und ueber die explizit angegebene gueltige Unterstuetzung berechnet, z. B. `W_s_valid` oder canvas-gueltige Pixel. Wenn die Unterstuetzung leer ist, ist die Statistik ungueltig und die Fallback-Regeln in Abschnitt 2.3.4 gelten.

Fuer deterministische Reproduzierbarkeit werden endliche Werte numerisch aufsteigend sortiert. Der Median ist bei ungerader Samplezahl der mittlere Wert und bei gerader Samplezahl das arithmetische Mittel der beiden mittleren Werte. MAD verwendet dieselbe Median-Konvention auf `|x - median(x)|`. Quantile verwenden lineare Interpolation zwischen sortierten Samples mit Index `q * (n - 1)`, begrenzt auf `[0, n-1]`; fuer `n = 1` wird das einzige Sample zurueckgegeben.

### 1.5 Canvas-Ausschlussvertrag

Canvas-ungueltige Pixel liegen ausserhalb der beobachteten Datendomaene. Sie sind keine nullwertigen Samples, keine Hintergrundsamples, keine Low-Quality-Samples und kein Padding. Sie duerfen AQMH in keiner Phase beeinflussen.

Bindende Regeln:

1. Canvas-ungueltige Quellpixel werden vor der AQMH-Kartenberechnung in ungueltig/NaN umgewandelt.
2. Downsampling verwendet einen Valid-Count-Denominator; ungueltige Pixel tragen weder Wert noch Gewicht bei.
3. Lokale Statistiken, Filter, Mediane, MADs, Laplacian-Responses, Artifact-Fractions, z-score-Populationen und Quantile arbeiten nur auf endlicher canvas-gueltiger Unterstuetzung.
4. Upsampling vom Skalenraum zum Canvas ist maskenbewusst; ungueltige Skalen-Samples interpolieren nicht in gueltige Canvas-Pixel.
5. Die Rekonstruktion iteriert nur ueber canvas-gueltige Ausgabepixel, und die Frame-Sample-Menge `V_c^I(p)` enthaelt nur endliche, canvas-gueltige Quellsamples.
6. Diagnostik und Regionsextraktion verwenden nur canvas-gueltige Unterstuetzung. Rohe ungueltige Canvas-Bereiche koennen aus Formkompatibilitaetsgruenden in Arrays vorhanden sein, muessen aber aus allen Statistiken ausgeschlossen werden.
7. Der finale `Q_map` Canvas-Guard setzt canvas-ungueltige Pixel exakt auf null, nur als Ausgabekonvention. Diese Null darf nie wieder als Datensample in AQMH-Statistiken eingehen.

---

## 2. Berechnung dichter Qualitaetskarten

### 2.1 Ueberblick

Fuer jeden Frame `f` und Kanal `c` wird aus dem prewarped, normalisierten Frame `I_{f,c}` eine dichte Qualitaetskarte `Q_map_{f,c}` berechnet. Die Karte wird in einer **Multi-Scale-Pyramide** mit `P` Skalen berechnet. Jeder Qualitaetsbeitrag pro Skala wird auf Canvas-Aufloesung hochskaliert, danach werden die hochskalierten Beitraege per geometrischem Mittel fusioniert.

### 2.2 Eingabedaten

Eingabe fuer AQMH ist der registrierte, prewarped, photometrisch normalisierte Frame `I_{f,c}(x,y)` aus dem gemeinsamen Preprocessing. Die Common-Overlap-Canvas-Maske gilt: Canvas-ungueltige Pixel werden aus allen Qualitaetskarten-Akkumulatoren ausgeschlossen.

### 2.3 Multi-Scale-Pyramide

#### 2.3.1 Skalendefinition

Definiere `P` Analyseskalen mit Downscale-Faktoren `D_s` und Fensterradien `R_s`:

| Skala `s` | Downscale `D_s` | Fenster `R_s` | Erfasste Struktur |
|---|---|---|---|
| 0 | 1  | 4 px  | Sub-Tile, pixelnahe Defekte, Hot Pixels |
| 1 | 4  | 4 px  | tile-vergleichbar (ca. 16 px Fenster) |
| 2 | 16 | 4 px  | grobe Regionen (ca. 64 px Fenster) |
| 3 | 64 | 4 px  | globaler Frame-Qualitaetskontext (ca. 256 px Fenster) |

Spezifizierte Defaults: `P = 4`, `D_s = 4^s`, `R_s = 4` (in downscaled Pixeln).

Eine Skala `s` wird **ausgelassen**, wenn `D_s > min(W, H) / 16`; das konfigurierte Maximum `P` bleibt unveraendert, waehrend `S_actual` und `P_actual` entsprechend reduziert werden. Aequivalent benoetigt Skala `s` `min(W, H) >= 16 * D_s`. Beispiele fuer die Defaults: Skala 1 (`D=4`) benoetigt `min(W,H) >= 64`; Skala 2 (`D=16`) benoetigt `>= 256`; Skala 3 (`D=64`) benoetigt `>= 1024`. Kleine Bilder lassen daher automatisch die groebsten Skalen weg, z. B. behaelt ein 512-px-Bild nur Skalen 0-2.

#### 2.3.2 Signalberechnung pro Skala

Auf Skala `s` wird eine downscaled Version der Eingabe berechnet:

`I_s(x,y) = downsample(I_{f,c}, D_s)`

mit Flaechenmittelung und canvas-maskenbewusstem Denominator; canvas-ungueltige Pixel werden aus dem Flaechenmittel ausgeschlossen und nicht durch Null ersetzt.

Fuer jedes Pixel `(x,y)` in der downscaled Domaene werden ueber ein lokales Fenster `W_s(x,y)` mit Radius `R_s` die folgenden **drei Qualitaetssignale** berechnet:

##### (a) Lokales Schaerfesignal `Phi_sharp`

`Phi_sharp_s(x,y) = Var_{p in W_s_valid(x,y)}(Lap_valid(I_s)(p))`

Dabei ist `Lap_valid` die maskierte Laplacian-Response mit gueltiger Unterstuetzung, und `Var` ist die lokale Varianz endlicher, gueltiger Laplacian-Werte. `Lap_valid` darf keine gespiegelten, replizierten, nullgefuellten oder canvas-ungueltigen Nachbarn verwenden. Wenn das Zentrumspixel ungueltig ist oder die endliche Stencil-Unterstuetzung fuer eine deterministische Laplacian-Schaetzung nicht ausreicht, ist die Response ungueltig. Das Ergebnis wird auf `[0, +inf)` begrenzt.

An dieser Stelle wird keine explizite globale Skalierung von `Phi_sharp_s` angewendet. Der robuste z-score pro Skala in Abschnitt 2.3.3 ist invariant gegenueber globaler multiplikativer Skalierung (`z(a*Phi) = z(Phi)` fuer `a > 0`), sodass eine `sigma_Lap`-basierte Normalisierung hier keinen Einfluss auf `Psi_s` haette und bewusst weggelassen wird. Implementierungen duerfen `Phi_sharp_s` aus numerischen Gruenden lokal umskalieren, duerfen aber nicht davon ausgehen, dass dies das Ergebnis veraendert.

##### (b) Lokales SNR-Signal `Phi_snr`

`b_s(x,y) = B_s(x,y; R_s) = median_{p in W_s_valid(x,y)} I_s(p)`  
`mu_s(x,y) = mean_{p in W_s_valid(x,y)}(max(I_s(p) - b_s(x,y), 0))`  
`sigma_s(x,y) = MAD_{p in W_s_valid(x,y)}(I_s(p)) * 1.4826`

`Phi_snr_s(x,y) = mu_s(x,y) / max(sigma_s(x,y), eps_aqmh)`

Der Clamp wird nur fuer den Mittelwertterm angewendet. Die SNR-Rauschskala `sigma_s` verwendet rohe endliche Pixelwerte `I_s(p)` in der lokalen Unterstuetzung, nicht den hintergrundsubtrahierten oder geclamp-ten Signalterm; dadurch wird die Rauschschaetzung nicht durch positives Signal-Clipping verzerrt. `B_s` ist ein deterministischer maskierter Median ueber dieselbe lokale gueltige Unterstuetzung `W_s_valid(x,y)`, die auch die anderen lokalen Statistiken verwenden. Wenn `W_s_valid(x,y)` leer ist, ist das Signal nach Abschnitt 2.3.4 ungueltig. Wenn weniger als drei gueltige Pixel verfuegbar sind, duerfen Implementierungen auf `mean(max(I_s(p), 0))` zurueckfallen, muessen aber das Diagnoseflag `scene_dependent_snr = true` setzen.

Scene-Dependence-Guard: `Phi_snr_s` ist ein lokaler Support-Qualitaetsproxy, kein Quellendetektor. Die hintergrundzentrierte Definition soll Source-Content-Bias reduzieren; der Fallback auf den nicht zentrierten Mittelwert ist nur als diagnostisch markierter degradierter Pfad erlaubt.

##### (c) Artefakt-Anomalie-Score `Phi_artifact`

`Phi_artifact_s(x,y)` erkennt lokale Outlier-Gradienten, die auf Satellitenspuren, kosmische Strahlen oder Wolkenkanten hinweisen:

1. Berechne das High-Pass-Residual `hp_s(x,y) = I_s(x,y) - blur(I_s, R_s)(x,y)`, wobei `blur(I_s, R_s)(x,y)` der maskierte lokale Mittelwert von `I_s` ueber `W_s_valid(x,y)` ist.
2. Berechne die lokale robuste Skala `tau_s(x,y) = max(1.4826 * MAD_{p in W_s_valid(x,y)}(hp_s(p)), eps_aqmh)`.
3. Berechne die lokale Outlier-Fraktion: `frac_out_s(x,y) = |{p in W_s_valid(x,y) : |hp_s(p)| > k_artifact * tau_s(x,y)}| / |W_s_valid(x,y)|`  
   mit spezifiziertem Default `k_artifact = 3.0`.
4. `Phi_artifact_s(x,y) = 1 - clip(frac_out_s(x,y) / frac_artifact_max, 0, 1)`  
   mit spezifiziertem Default `frac_artifact_max = 0.25`.

`Phi_artifact_s = 1` bedeutet eine saubere Region; `Phi_artifact_s = 0` bedeutet, dass mindestens `frac_artifact_max` (Default 25%) der Pixel im Fenster Outlier sind.

#### 2.3.3 Qualitaetskarte pro Skala

Die Qualitaetskarte pro Skala `Psi_s(x,y)` ist definiert als:

`Psi_s(x,y) = sigmoid(score_scale * (w_sharp * z(Phi_sharp_s) + w_snr * z(Phi_snr_s))) * Phi_artifact_s(x,y)`

wobei:

- `z(Phi)` die robuste z-score-Normalisierung ist: `z(Phi)(x,y) = (Phi(x,y) - median(Phi_s)) / max(1.4826 * MAD(Phi_s), eps_aqmh)`, angewendet ueber alle endlichen, canvas-gueltigen Pixel auf Skala `s`
- `sigmoid(v) = 1 / (1 + exp(-v))`
- `w_sharp`, `w_snr` konfigurierbare Gewichte sind (spezifizierte Defaults: `w_sharp = 0.6`, `w_snr = 0.4`)
- `score_scale` die lokale Sigmoid-Temperatur ist (aktueller Betriebsdefault: `1.8`, muss `> 0` sein)
- der Artefaktterm `Phi_artifact_s` als multiplikatives Gate wirkt: Er unterdrueckt Regionen mit zu hoher Outlier-Dichte unabhaengig von Schaerfe oder SNR.

Bindende Bedingung: `Psi_s(x,y) in [0, 1]` fuer alle endlichen Eingaben. Der Sigmoid-Term ist strikt positiv, aber das multiplikative Artefakt-Gate kann `Psi_s` exakt auf null setzen.

**Within-frame relativity (bindende Klarstellung):** Weil `z(Phi_sharp_s)` und `z(Phi_snr_s)` pro Frame normalisiert werden (Median/MAD ueber die Pixel dieses Frames auf Skala `s`), ist der **Sigmoid-Faktor von `Q_map` ein innerhalb des Frames relatives Qualitaetsfeld**, keine absolute Cross-Frame-Qualitaetsmessung. Zwei Frames mit unterschiedlichem globalem Seeing erzeugen aehnlich verteilte Sigmoid-Faktoren, jeweils selbstnormalisiert. Folglich gilt im AQMH-Rekonstruktionsgewicht `W_{f,c}^{aqmh} = G_{f,c} * Q_map_{f,c}` (Abschnitt 1.1, 4.3):

- **Zwischen-Frame**-Diskriminierung an einem Pixel wird durch das globale Gewicht `G_{f,c}` und durch **absolute** Gates wie `Phi_artifact_s` getragen, die nicht z-gescored werden und `Q_map` in jedem Frame unabhaengig von anderen Frames gegen null treiben koennen.
- **Innerhalb-Frame**-raeumliche Diskriminierung, also welche Regionen eines einzelnen Frames schaerfer/sauberer sind, wird durch den Sigmoid-Faktor getragen.

Diese Arbeitsteilung ist absichtlich und bindend: Der Sigmoid-Faktor darf nicht als absolute photometrische Qualitaet interpretiert und nicht zum Ranking ganzer Frames verwendet werden.

### 2.3.4 Rand- und Leerfensterregeln

Alle lokalen Statistiken werden nur ueber endliche, canvas-gueltige Pixel berechnet. `W_s_valid(x,y)` sei die gueltige Teilmenge des Analysefensters.

Wenn `|W_s_valid(x,y)| = 0`, werden alle pro-Skala-Signale an `(x,y)` als ungueltig markiert, und der fusionierte `Q_map`-Wert wird spaeter durch den Canvas-Guard auf null gesetzt. Wenn `|W_s_valid(x,y)| > 0`, aber weniger als drei gueltige Pixel vorhanden sind, fallen robuste Skalenschaetzungen auf `eps_aqmh` und lokale Varianzschaetzungen auf null zurueck. Fuer das `Phi_snr_s`-Signal hat die Hintergrundzentrierungs-Fallback-Regel in Abschnitt 2.3.2(b) Vorrang.

Der Default-Randmodus fuer faltungsartige Operationen (`Lap_valid`, `blur`, lokale Fenster und Morphologie-Supportmasken) ist valid-only maskierte Auswertung. Implementierungen duerfen canvas-ungueltige Pixel nicht spiegeln, replizieren, nullfuellen oder sonstwie synthetisch in die Statistik einfuehren. Wenn ein Bibliotheksprimitive diese Support-Regel nicht direkt ausdruecken kann, muessen Implementierungen Zaehler und Valid-Support-Denominator separat berechnen oder einen explizit maskierten Operator verwenden.

### 2.4 Multi-Scale-Fusion

`S_actual` sei die geordnete Menge der Skalen, die nach Anwendung der Omission-Regel aus Abschnitt 2.3.1 tatsaechlich berechnet werden, und `P_actual = |S_actual|`. Jede berechnete `Psi_s` wird per maskenbewusster bilinearer Interpolation auf volle Canvas-Aufloesung hochskaliert:

`Psi_s^{up}(x,y) = upsample_valid(Psi_s, valid_s, D_s)`

wobei `valid_s` die endliche Valid-Support-Maske von `Psi_s` ist. `upsample_valid` interpoliert Zaehler `Psi_s * valid_s` und Supportmaske `valid_s` separat und dividiert anschliessend durch den interpolierten Support. Wenn der interpolierte Support an einem Canvas-Pixel null ist, ist `Psi_s^{up}` an diesem Pixel ungueltig. Ungueltige Skalen-Samples duerfen bei der Interpolation nicht als null behandelt werden, weil dies benachbarte gueltige Canvas-Pixel absenken wuerde.

Fusion per **geometrischem Mittel** ueber die `P_actual` berechneten Skalen:

`Q_map_{f,c}(x,y) = ( prod_{s in S_actual} Psi_s^{up}(x,y) )^{1/P_actual}`

Das konfigurierte `P` ist eine Obergrenze. Es darf nicht als Exponent-Denominator verwendet werden, wenn eine oder mehrere Skalen ausgelassen wurden. Wenn `P_actual = 0`, ist `Q_map` nach dem Canvas-Guard ueberall null.

Das geometrische Mittel wird dem arithmetischen Mittel vorgezogen, weil es verlangt, dass **alle Skalen hohe Qualitaet bestaetigen**. Eine einzelne Skala, die ein Artefakt meldet, unterdrueckt die fusionierte Karte unabhaengig von anderen Skalen. Das implementiert eine konservative "all-clear"-Fusionsphilosophie.

**Zero-scale guard:** Wenn `Psi_s^{up}(x,y) = 0` fuer irgendeine Skala `s`, dann ist `Q_map_{f,c}(x,y) = 0` exakt; eine schlechte Skala vetoiert das Pixel.

**Canvas guard (bindend):** Fuer alle canvas-ungueltigen Pixel `p` wird `Q_map_{f,c}(p) = 0` bedingungslos nach der Fusion gesetzt und ueberschreibt jeden berechneten Wert.

Wenn eine berechnete Skala an einem canvas-gueltigen Pixel ungueltig ist, weil nach maskenbewusstem Upsampling kein gueltiger Skalen-Support vorhanden ist, traegt diese Skala an diesem Pixel ein Zero-Veto bei. Das ist verschieden von canvas-ungueltigen Pixeln, die ausgeschlossen und nur durch den finalen Canvas-Guard genullt werden.

### 2.5 Block-Level-Diagnosezusammenfassungen

Fuer Reports und visuelle Zusammenfassungen darf AQMH Block-Level-Diagnosewerte ableiten, indem `Q_map_{f,c}` ueber einen Anzeigeblock `b` aggregiert wird:

`Q_{f,b,c}^{aqmh} = median_{p in b, canvas-valid} Q_map_{f,c}(p)`

Das Blockraster ist nur eine Reporting-/Visualisierungshilfe. Es ist nicht Teil des AQMH-Rekonstruktionsgewichtsmodells und darf keine blockkonstanten Gewichte in den AQMH-Akkumulator einfuehren.

---

## 3. Qualitaetskarten-Speicher- und Speichermodell

### 3.1 Speicherformat

Konzeptionell ist `Q_map_{f,c}` ein Full-Canvas-Qualitaetsfeld, eines pro Frame und Kanal. Die persistierte Darstellung darf nach Abschnitt 3.2 niedriger aufgeloest oder quantisiert sein. Fuer mehrkanalige CFA-Eingaben, die mit der CFA-proxy-aequivalenten Kernvariante verarbeitet werden, kann statt kanalweiser Karten eine einzelne Luminanzkanal-Karte verwendet werden (konfigurierbar).

Empfohlene Speicherung: auf dem bestehenden `DiskCacheFrameStore` oder in einem parallelen Qualitaetskarten-Disk-Cache mit identischer Indexsemantik.

### 3.2 Speicherbudget

Bei voller Aufloesung benoetigt eine Karte `W * H * 4` Bytes. Fuer einen 24-Mpx-Sensor mit 300 Frames und 3 Kanaelen waere der volle unkomprimierte **On-Disk-Working-Set** etwa `24e6 * 4 * 300 * 3 ~= 86 GB`.

Diese Zahl ist **kein** erlaubtes RAM-Budget. AQMH darf niemals voraussetzen, dass alle Frames, alle prewarped Frames oder alle Qualitaetskarten im Speicher liegen. Wie der Rest von Tile Compile ist AQMH fuer hunderte Frames ausgelegt und muss in jeder Phase als streamingfaehige, disk-cache-gestuetzte Methode implementiert werden.

Bindende Speicherinvariante:

1. Zur AQMH-Kartenberechnungszeit darf jeder Worker nur den aktuellen Quellframe, seine aktuellen Pyramiden-Temporaries und die aktuelle Ausgabekarte halten.
2. Nachdem `Q_map` eines Frames berechnet wurde, muss sie zeitnah in den AQMH-Map-Cache geschrieben werden, und Full-Resolution-Arbeitspuffer muessen freigegeben werden.
3. AQMH-Rekonstruktion muss Frames und Karten ueber begrenzte Provider/Caches lesen. Die Anzahl residenter Quellframes und Full-Resolution-Karten muss durch explizite Speicherlimits begrenzt sein und darf nicht mit der Framezahl skalieren.
4. Eine gueltige Implementierung muss hunderte Frames ohne OOM verarbeiten koennen, indem sie Speicher gegen Disk-IO tauscht.

Daher werden folgende Kompressionsstrategien unterstuetzt:

| Strategie | Beschreibung | Spezifiziert? |
|---|---|---|
| Full resolution float32 | Keine Kompression | Optional |
| 1/4 area float32 | Downscale um 2 pro Achse | **Default** |
| uint16 quantization | Karte auf `[0, 65535]` skaliert | Optional, empfohlene Performance-Variante, wenn bitidentische float32-Cachewerte nicht erforderlich sind |
| uint8 quantization | Karte auf `[0, 255]` skaliert | Optional, kleinerer Cache mit hoeherem Quantisierungsfehler |
| Block-compressed float16 | Float16-Teilbloecke pro Block | Zukuenftig optional; nur gueltig, wenn explizit implementiert |

**Spezifizierter Default:** Speichere `Q_map` bei `1/4` Flaechenaufloesung (Faktor-2-Downscale je Achse, `resolution_divisor = 2`). Die Karte wird bei Bedarf waehrend der Rekonstruktion per bilinearer Interpolation auf volle Aufloesung hochskaliert.

### 3.3 Disk-Cache-Lebenszyklus

Qualitaetskarten werden waehrend der AQMH-Kartenberechnung geschrieben und waehrend der AQMH-Rekonstruktion gelesen. Karten werden invalidiert, wenn der prewarped Quellframe invalidiert wird, wenn sich die Common-Overlap-Maske aendert oder wenn sich eine **kartenbeeinflussende** AQMH-Konfiguration aendert (`pyramid`, `storage` oder Kartenformatversion). Nur-rekonstruktionsbezogene Einstellungen duerfen Map-Cache-Eintraege nicht invalidieren. Implementierungen sollten einen Cache-Metadaten-Hash speichern und validieren, der nur kartenbeeinflussende Eingaben umfasst.

Der Cache ist keine Optimierung, sondern Teil des AQMH-Ausfuehrungsmodells. Implementierungen muessen cache-gestuetzten Zugriff fuer alle grossen datenpro-Frame-Produkte verwenden:

| Stage | Grosse Daten | Erforderliches Zugriffsmuster |
|---|---|---|
| Gemeinsames Preprocessing | kalibrierte/registrierte/prewarped Frames | disk-backed Frame Store; begrenzter residenter Frame-Satz |
| AQMH-Kartenberechnung | Quellframe, Pyramidenpuffer, Ausgabekarte | ein Frame pro Worker; write-through Map Cache |
| AQMH-Rekonstruktion | Quellframes und `Q_map`-Dateien | begrenzter Frame-/Map-Read-Cache; kein Full-Run-Preload |
| AQMH-Diagnostik/Report | Metriken und Zusammenfassungen | aggregierte JSON/Statistiken; Rohkarten bleiben Cache-Artefakte |

---

## 4. Pipeline-Integration

### 4.1 AQMH-Verarbeitungsstufen

AQMH hat eigene algorithmische Stufen. Eine konkrete Anwendung darf diese Stufen aus Engineering-Gruenden innerhalb bestehender Runner-Phasen schedulen, aber dieses Scheduling ist nicht Teil der mathematischen Methode.

```text
AQMH_MAPS
  Berechne dichte Qualitaetskarten Q_map pro Frame

AQMH_RECONSTRUCTION
  Fuehre pixelweises gewichtetes Stacking mit W_aqmh = G * Q_map aus

AQMH_DIAGNOSTICS
  Schreibe Qualitaetskarten-, Rekonstruktions- und optionale Regionsartefakte

AQMH_NATIVE_BGE_INPUTS
  Optionale Postprocessing-Unterstuetzung: leite BGE-Sampling-Hilfen aus dem
  AQMH-Rekonstruktionsoutput und der Canvas-Maske ab, nicht aus Classic
  Tile Compile Local Metrics.
```

Gemeinsame Preprocessing- und Postprocessing-Stufen duerfen wiederverwendet werden, aber Classic-Tile-Compile-Local-Metrics und Tile-Reconstruction sind keine AQMH-Stufen. Wenn BGE fuer einen AQMH-Run aktiviert ist, muessen BGE-Tile-Sampling-Hilfen aus dem AQMH-Rekonstruktionsoutput und der Canvas-Valid-Maske abgeleitet werden. Sie duerfen pro Tile Hintergrund-, robuste Rausch- und Gradienten-/Strukturschaetzungen fuer BGE-Sampling enthalten, sind aber keine AQMH-Rekonstruktionsgewichte und duerfen nicht aus `local_metrics.json` gelesen werden.

### 4.2 AQMH-Kartenberechnung

Fuer jeden Frame `f` und Kanal `c` mit `frame_has_data[f] = true`:

1. Lade den prewarped Frame `I_{f,c}` aus dem `DiskCacheFrameStore`.
2. Wende die Common-Overlap-Canvas-Maske an: setze Pixel ausserhalb der Maske auf NaN (aus allen Fensterstatistiken ausgeschlossen).
3. Fuer `s = 0, ..., P-1`:
   a. Berechne `I_s` per flaechengemitteltem Downsampling mit maskenbewusstem Denominator.
   b. Berechne `Phi_sharp_s`, `Phi_snr_s`, `Phi_artifact_s` (Abschnitt 2.3.2).
   c. Berechne `Psi_s` (Abschnitt 2.3.3).
4. Skaliere alle `Psi_s` auf Canvas-Aufloesung hoch (Abschnitt 2.4).
5. Berechne das fusionierte `Q_map_{f,c}` per geometrischem Mittel (Abschnitt 2.4).
6. Wende den Canvas-Guard an: setze canvas-ungueltige Pixel auf null.
7. Schreibe `Q_map_{f,c}` in den AQMH-Qualitaetskarten-Disk-Cache (in der konfigurierten Speicheraufloesung).

### 4.3 AQMH pixelweise gewichtete Rekonstruktion

Fuer jedes Pixel `p` in canvas-gueltiger Unterstuetzung:

Definiere die endliche Intensitaets-Sample-Menge:

`V_c^{I}(p) = { f | I_{f,c}(p) is finite AND canvas-valid }`

Definiere die Map-verfuegbare Sample-Menge:

`V_c^{map}(p) = { f in V_c^{I}(p) | Q_map_{f,c}(p) is finite }`

Fuer jedes `f in V_c^{I}(p)` ist das effektive AQMH-Pixelgewicht:

`w_{f,c}^{aqmh}(p) = G_{f,c} * Q_map_{f,c}(p)` wenn `f in V_c^{map}(p)`

`w_{f,c}^{aqmh}(p) = 0` wenn das Map-Sample nicht verfuegbar oder nicht endlich ist

Kein Classic-Tile-Compile-Local-/Tile-Gewicht wird als AQMH-Fallback verwendet.

Canvas-ungueltige Ausgabepixel werden nicht rekonstruiert. Sie werden als unsupported/null geschrieben, ohne Frame-Samples, Map-Samples, Sigma-Clipping oder Denominator-Fallback auszuwerten. Canvas-ungueltige Quellpixel sind nie Mitglieder von `V_c^I(p)`.

Der rekonstruierte Pixelwert ist:

`R_c^{aqmh}(p) = sum_{f in V^I} w_{f,c}^{aqmh}(p) * I_{f,c}(p) / sum_{f in V^I} w_{f,c}^{aqmh}(p)`

**Unsupported-Pixel-Handling (bindend):** Vor jedem numerischen Denominator-Guard muessen Implementierungen endliche Map-Samples von nicht verfuegbaren Map-Samples unterscheiden. Eine endliche Null ist ein gueltiges Map-Sample und ein explizites Veto, kein fehlender Wert. Wenn `sum_f max(w_{f,c}^{aqmh}(p), 0) <= eps_weight`, haengt das Fallback-Verhalten davon ab, warum die Summe null ist:

1. Wenn an `p` mindestens ein endliches Map-Sample existiert (`V_c^{map}(p) != empty`) und alle AQMH-Gewichte null sind, weil die verfuegbaren Maps das Pixel explizit vetieren (`Q_map = 0`), wird das Ausgabepixel als unsupported/null markiert. Ersetze das explizite Zero-Veto **nicht** durch einen ungewichteten Mittelwert.
2. Wenn an `p` kein endliches Map-Sample existiert oder alle Map-Samples wegen IO-/Cache-Fehler nicht verfuegbar sind, wird das Ausgabepixel als unsupported/null markiert, und der Run emittiert eine AQMH-Cache-/Map-Verfuegbarkeitswarnung. AQMH darf nicht stillschweigend auf Classic-Tile-Compile-Gewichte wechseln.
3. Wenn Sigma-Clipping nach einer nicht-null Pre-Clip-Gewichtssumme alle Samples entfernt, gelten die AQMH-Sigma-Clipping-Keep-Floor- und Denominator-Guard-Semantiken fuer pixelweise gewichtete Rekonstruktion.

**Sigma-Clipping:** Iteratives gewichtetes Sigma-Clipping wird mit `w_{f,c}^{aqmh}(p)` angewendet. Keep-Floor `min_fraction` und `N_eff`-/`D_eff`-Guards sind AQMH-Rekonstruktionsparameter und muessen deterministisch sein.

---

## 5. Adaptive Region Extraction (optional)

### 5.1 Motivation

Zusaetzlich zur kontinuierlichen Gewichtskarte kann AQMH **binaere Qualitaetsregionen** fuer Diagnoseberichte und fuer den optionalen Cherry-Pick-Stacking-Modus erzeugen.

### 5.2 Qualitaetskontur-Extraktion

Aus der fusionierten `Q_map_{f,c}` werden binaere Regionen per Thresholding extrahiert:

1. Berechne den Frame-Schwellenwert: `tau_f = quantile(Q_map_{f,c}, q_region)` ueber endliche, canvas-gueltige Pixel, mit spezifiziertem Default `q_region = 0.75`.
2. Binaermaske: `M_f(x,y) = 1 iff Q_map_{f,c}(x,y) >= tau_f AND canvas-valid`.
3. Wende morphologisches Opening mit Radius `r_morph_canvas_px` an, um isolierte Rauschregionen zu entfernen. Der spezifizierte Default ist ein **canvas-aequivalenter Radius von 6 px**, entsprechend 3 px beim Default `resolution_divisor = 2`. Wenn Regionsextraktion auf einer gespeicherten/downscaled Map laeuft, verwende `r_morph_map = max(1, round(r_morph_canvas_px / resolution_divisor))`. Morphologie ist auf canvas-gueltige Unterstuetzung beschraenkt: ungueltige Canvas-Pixel liegen ausserhalb der Domaene, sind keine nullwertigen Hintergrundpixel, und die finale Regionsmaske wird immer mit der Canvas-Valid-Maske geschnitten.
4. Extrahiere verbundene Komponenten; beschrifte jede Komponente mit:
   - `Area_r`: Pixelanzahl
   - `MeanQ_r`: mittlerer Qualitaetsscore ueber die Region
   - `Compactness_r = 4*pi*Area_r / Perimeter_r^2` (Polsby-Popper-Score)
5. Ranke Regionen nach `Score_r = MeanQ_r * log(1 + Area_r)`.

Diese Regionen werden im Diagnoseartefakt `aqmh_regions.json` pro Frame berichtet.

### 5.3 Optionaler Cherry-Pick-Stacking-Modus

Wenn `aqmh.cherry_pick.enabled = true`, verwendet pixelweises Stacking einen
explizit berichteten Selektionsmodus. Standard ist `mode = auto_reject`: AQMH
behaelt die meisten lokal nutzbaren Frames und verwirft nur klare lokale
Low-Score-Ausreisser.

Fuer jedes Pixel `p` werden rankbare Samples nach dem cross-frame kalibrierten
Score `S_f(p) = G_{f,c} * Q_map_{f,c}(p)` bewertet. Im Modus `auto_reject`
gelten:

```
S_best(p) = max_f S_f(p)
T(p) = reject_below_best_fraction * S_best(p)
K_floor(p) = max(k_min_required, ceil(min_keep_fraction * N_rankable(p)))
```

Samples unterhalb `T(p)` sind verwerfbar, aber mindestens `K_floor(p)` Samples
bleiben erhalten. Ist der Score-Abstand zwischen letztem behaltenen und erstem
verworfenen Sample kleiner als `margin_min`, wird lokal nichts verworfen.

`mode = top_k` bleibt als Legacy-Regel verfuegbar:

`K(p) = min(N_valid(p), max(k_min_required, floor(k_frac * N_valid(p))))`

**Warnung (bindend):** Cherry-Pick-Modus verletzt die Default-AQMH-No-Frame-Selection-Invariante auf Pixelebene, auch wenn er keine ganzen Frames verwirft. Er darf nur verwendet werden, wenn der Nutzer ihn explizit aktiviert, und muss in Diagnoseausgaben klar markiert werden. Default ist `disabled`.

---

## 6. Qualitaetskarten-Diagnostik

### 6.1 Pro-Frame-Diagnostik

Fuer jeden verarbeiteten Frame `f` werden folgende skalare Diagnosen in `aqmh_metrics.json` geschrieben:

| Feld | Definition |
|---|---|
| `map_mean` | Mittelwert von `Q_map_{f,c}` ueber canvas-gueltige Pixel |
| `map_p10` | 10. Perzentil ueber canvas-gueltige Pixel |
| `map_p90` | 90. Perzentil ueber canvas-gueltige Pixel |
| `artifact_frac` | Anteil canvas-gueltiger Pixel mit `Q_map_{f,c} < tau_artifact` (spezifizierter Default: `tau_artifact = 0.2`) |
| `sharpness_p50` | Median des **pre-z-score** `Phi_sharp_0` auf Skala 0 |
| `snr_p50` | Median des **pre-z-score** `Phi_snr_1` auf Skala 1 |
| `n_regions` | Anzahl Qualitaetsregionen (Abschnitt 5.2) oberhalb des Schwellenwerts |

Wenn die referenzierte Diagnoseskala durch die Small-Image-Skalenregel (Abschnitt 2.3.1) ausgelassen wird, wird das entsprechende Diagnosefeld als `NaN` oder `null` geschrieben, und das Artefakt muss ebenfalls festhalten, dass die Skala nicht verfuegbar war. Mit Defaultwerten ist `sharpness_p50` normalerweise verfuegbar, weil Skala 0 `D=1` hat; `snr_p50` kann nicht verfuegbar sein, wenn Skala 1 ausgelassen wird.

### 6.2 Block-Level-Diagnostik

Fuer jeden Reportblock `b` koennen folgende Werte berichtet werden:

- `aqmh_q_median`: `Q_{f,b,c}^{aqmh}` wie in Abschnitt 2.5 definiert
- `aqmh_q_p10`, `aqmh_q_p90`: 10. und 90. Perzentil innerhalb des Blocks
- `aqmh_artifact_frac`: Anteil der Pixel im Block mit `Q_map < tau_artifact`

### 6.3 Heatmaps

Fuer die Integration in den Reportgenerator emittiert AQMH raeumliche Heatmap-Eintraege fuer das Artefakt `aqmh_metrics.json`:

- mittleres `Q_map` pro Reportblock und Frame
- Artefakt-Fraktion-Heatmap pro Reportblock
- optionale AQMH-vs-Classic-Vergleichsheatmaps nur, wenn beide Methoden separat auf demselben Eingabesatz ausgefuehrt wurden

---

## 7. Konfiguration

### 7.1 Top-Level-Schalter

```yaml
method: aqmh              # optionaler expliziter Methodenschluessel: classic_tile_compile | aqmh
aqmh:
  enabled: false        # default: disabled until validated
```

Wenn `aqmh.enabled: false`, werden alle AQMH-Berechnungen uebersprungen. Wenn die Implementierung noch keinen expliziten Top-Level-`method`-Schluessel unterstuetzt, muss der Runtime-Status trotzdem die abgeleitete Methode anzeigen: `aqmh.enabled = false` bedeutet `classic_tile_compile`, und `aqmh.enabled = true` bedeutet `aqmh`.

### 7.2 Pyramidenkonfiguration

```yaml
aqmh:
  pyramid:
    scales: 4           # Anzahl Skalen P (Default: 4)
    base_window_px: 4   # Fensterradius R_s in downscaled Pixeln (Default: 4)
    w_sharp: 0.6        # Schaerfegewicht im pro-Skala-Sigmoid (Default: 0.6)
    w_snr: 0.4          # SNR-Gewicht im pro-Skala-Sigmoid (Default: 0.4)
    score_scale: 1.8    # lokale Sigmoid-Temperatur (Default: 1.8)
    k_artifact: 3.0     # Outlier-Erkennungsschwelle (Default: 3.0)
    frac_artifact_max: 0.25  # Artefakt-Gate-Schwelle (Default: 0.25)
```

### 7.3 Speicherkonfiguration

```yaml
aqmh:
  storage:
    resolution_divisor: 2   # linearer Divisor pro Achse: 1=voll, 2=halbe Breite/Hoehe (1/4 Flaeche), 4=Viertelbreite/-hoehe (1/16 Flaeche)
    dtype: float32          # float32 | uint16 | uint8 (Default: float32)
    max_resident_maps: 2    # begrenzter Read-Through-Cache waehrend Rekonstruktion; 0 deaktiviert
```

Der Speicherdefault (`resolution_divisor = 2`, `dtype = float32`) entspricht der **1/4-area float32**-Strategie in Abschnitt 3.2. `max_resident_maps` begrenzt, wie viele Full-Resolution-Maps waehrend der AQMH-Rekonstruktion gleichzeitig im RAM gehalten werden duerfen; diese Zahl darf nicht mit der Framezahl skalieren.
`uint16` ist das empfohlene Performance-Cache-Format, wenn bitidentische float32-Cachewerte nicht erforderlich sind. Es quantisiert `Q_map` auf `[0,65535]`, erhaelt exakte Zero-Veto- und Full-Quality-Endpunkte und hat einen maximalen Quantisierungsfehler von etwa `7.7e-6`.

### 7.4 Cherry-Pick-Modus

```yaml
aqmh:
  cherry_pick:
    enabled: false      # muss explizit aktiviert werden; bricht No-Frame-Selection-Invariante auf Pixelebene
    mode: auto_reject
    k_min_required: 20
    reject_below_best_fraction: 0.25
    min_keep_fraction: 0.90
    k_frac: 0.30       # nur fuer mode: top_k
```

### 7.5 Diagnostikkonfiguration

```yaml
aqmh:
  diagnostics:
    tau_artifact: 0.20  # Qualitaetsschwelle fuer artifact_frac-Diagnostik (Default: 0.20)
    q_region: 0.75      # Quantilschwelle fuer Qualitaetsregionsextraktion (Default: 0.75)
    r_morph_canvas_px: 6 # canvas-aequivalenter Radius fuer Qualitaetsregion-Morphologie (Default: 6)
```

`tau_artifact` ist eine **nur diagnostische** Schwelle (siehe Abschnitt 6.1, 6.2). Sie beeinflusst weder Rekonstruktionsgewichte noch das pro-Skala-Artefakt-Gate `Phi_artifact_s`, das durch `k_artifact` und `frac_artifact_max` in Abschnitt 7.2 gesteuert wird.

---

## 8. Numerische Defaults

Alle `eps_aqmh`-Konstanten haben den Default `1e-6`, sofern nicht anders angegeben.

| Parameter | Default | Beschreibung |
|---|---|---|
| `eps_aqmh` | `1e-6` | Denominator-Guard fuer alle AQMH-Divisionen |
| `k_artifact` | `3.0` | Outlier-Sigma-Multiplikator |
| `frac_artifact_max` | `0.25` | Maximal tolerierte Outlier-Fraktion pro Fenster |
| `w_sharp` | `0.6` | Schaerfegewicht im pro-Skala-Qualitaetssigmoid |
| `w_snr` | `0.4` | SNR-Gewicht im pro-Skala-Qualitaetssigmoid |
| `score_scale` | `1.8` | lokale Sigmoid-Temperatur der AQMH-Quality-Map |
| `P` | `4` | Maximale Anzahl Pyramidenskalen; tatsaechliche Anzahl kann wegen Omission-Regel in Abschnitt 2.3.1 niedriger sein |
| `R_s` | `4` | Fensterradius auf jeder Skala (in downscaled Pixeln) |
| `q_region` | `0.75` | Qualitaetsquantilschwelle fuer Regionsextraktion |
| `r_morph_canvas_px` | `6` | Morphologischer Opening-Radius in Canvas-Pixeln; Map-Space-Radius ist `max(1, round(r_morph_canvas_px / resolution_divisor))` |
| `k_frac` | `0.30` | Cherry-Pick-Frame-Fraktion |
| `k_min` | `3` | Mindestframes im Cherry-Pick-Modus |
| `tau_artifact` | `0.20` | Qualitaetsschwelle fuer Artifact-Fraction-Diagnostik |
| `max_resident_maps` | `2` | Maximale Full-Resolution-Maps im RAM waehrend Rekonstruktion |
| `resolution_divisor` | `2` | Speicher-Downscale-Faktor pro Achse (1/4-area Default) |

---

## 9. Validierungsanforderungen

Wenn AQMH aktiviert ist, gelten folgende Validierungsanforderungen:

1. **Map range:** `Q_map_{f,c}(p) in [0, 1]` fuer alle endlichen canvas-gueltigen Pixel.
2. **Canvas guard:** `Q_map_{f,c}(p) = 0` fuer alle canvas-ungueltigen Pixel.
3. **Determinismus:** Identische registrierte Frames und Canvas-Masken erzeugen identische Qualitaetskarten.
4. **Unsupported coverage:** Jedes Pixel mit `V_c^{I}(p) = empty` gibt null/unsupported zurueck; jedes Pixel mit endlichen Intensitaetssamples, aber ohne endliche AQMH-Map-Samples, gibt null/unsupported mit AQMH-Warnung zurueck; kein NaN/Inf im Output.
5. **Explicit zero-veto:** Wenn endliche Maps an einem Pixel existieren und alle verfuegbaren AQMH-Gewichte null sind, bleibt der Output unsupported/null und darf nicht durch einen ungewichteten Mittelwert ersetzt werden.
6. **Block diagnostic consistency:** `Q_{f,b,c}^{aqmh}` entspricht `median(Q_map over b)` innerhalb Floating-Point-Toleranz.
7. **No structural injection:** Seam-Scores, FWHM und Background RMS duerfen gegenueber einem AQMH-deaktivierten Kontrollrun auf demselben Dataset nicht ueber die dokumentierte Validierungstoleranz hinaus regressieren.
8. **Artifact detection:** Bekannte satellitenkontaminierte Frames zeigen erhoehtes `artifact_frac > 0.01` fuer mindestens die kontaminierten Reportbloecke.
9. **Scale omission:** Fuer eine Eingabe mit `P_actual < P` (z. B. `min(W,H) < 64` mit Defaults) verwendet die Fusion `P_actual` als geometrischen-Mittel-Denominator, ausgelassene Skalen werden in Diagnosen erfasst, und nicht verfuegbare Diagnoseskalen werden als `NaN`/`null` geschrieben.
10. **Cherry-pick flag:** Wenn `cherry_pick.enabled = true`, muss das Output-Artefakt `aqmh_metrics.json` `cherry_pick_active: true` enthalten und das Pipeline-Log muss eine `WARNING`-Level-Meldung emittieren.

---

## 10. Scope Boundary

### Gemeinsame Infrastruktur

- Input-Scan, Kalibrierung, Registrierung/Prewarping, globale Normalisierung, Common-Overlap-Maske, Run-Management, Logging, Reports

### AQMH-Methode

- Berechnung dichter Qualitaetskarten
- pixelweise gewichtete AQMH-Rekonstruktion
- Adaptive Region Extraction (Abschnitt 5)
- Cherry-Pick-Stacking (Abschnitt 5.3, nur explizites Opt-in)
- AQMH-Diagnoseartefakte

---

## 11. Kernaussage

AQMH ist eine eigenstaendige deterministische gewichtete-Mittelwert-Rekonstruktionsmethode. Sie verwendet ein kontinuierliches pixelweises Qualitaetsfeld statt blockkonstanter lokaler Gewichte. Jedes Pixel, das in den Rekonstruktionsakkumulator eingeht, tut dies mit einem deterministischen nichtnegativen AQMH-Gewicht, das sowohl globale atmosphaerische Bedingungen als auch lokale raeumliche Qualitaet widerspiegelt, ohne kuenstliche Blockgrenzen und ohne Abhaengigkeit von Classic-Tile-Compile-Gewichten oder Fallback-Verhalten.
