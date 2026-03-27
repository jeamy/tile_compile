# Review der Methodik v3.3.8

Referenzdokument: `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.8_en.md`  
Review-Typ: Spezifikationsanalyse auf mathematische Fehler, Logikfehler und Qualitätsverbesserungen  
Stand: 2026-03-27

## 1. Kurzfazit

Die Spezifikation ist in mehreren Bereichen deutlich präziser als fruehere Fassungen, hat aber noch einige harte mathematische und logische Probleme.

Die groessten Punkte sind:

1. Die tile-weise Normalisierung vor OLA widerspricht dem eigenen Linearitaetsanspruch.
2. Die Definition gueltiger Samples ueber `> 0` erzeugt bei kalibrierten linearen Daten einen systematischen Positiv-Bias.
3. Die finale Cluster-Aggregation erhaelt die Cluster-Masse nicht und kann kleine, gute Cluster stark uebergewichten.
4. Mehrere `clip(...)`-Definitionen sind fuer kleine Bilder oder unguenstige Parameterbereiche mathematisch nicht wohldefiniert.
5. Einige optionale Teile sind spaeter implizit wieder als obligatorisch behandelt.

Wenn diese Punkte korrigiert werden, ist eine reale qualitative Verbesserung der Rekonstruktion wahrscheinlich, insbesondere bei Photometrie, Hintergrundstabilitaet, Nahtfreiheit und numerischer Robustheit.

## 2. Bewertungsrahmen

Diese Analyse ist eine Review des Dokuments selbst, nicht der C++-Implementierung und nicht eines empirischen Benchmarks. Bewertet wurden:

- mathematische Konsistenz,
- logische Konsistenz zwischen den Kapiteln,
- numerische Stabilitaet,
- und zu erwartende Bildqualitaet.

## 3. Kritische Befunde

### 3.1 Nichtlinearitaet der Tile-Normalisierung vor OLA

Referenzen:

- `1.3 Linearity Semantics`, Zeilen 44-50
- `5.7.1 Tile Normalization before OLA`, Zeilen 456-469
- `5.7.1a`, Zeilen 471-487
- `5.7.1b`, Zeilen 489-503

Befund:

Die Operation

`Y_t(p) = (R_{t,c}(p) - bg_t) / m_t`

ist nicht linear und auch nicht global affin im mathematischen Sinn, weil `bg_t = median(R_t)` und `m_t = median(|R_t - bg_t|)` datenabhaengige, nichtlineare Funktionale von `R_t` sind. Dass danach mit globalen Medianwerten `m_global` und `bg_global` rueckskaliert wird, macht die Gesamtabbildung ebenfalls nicht linear, weil auch diese Groessen datenabhaengig sind.

Konsequenz:

- Der Anspruch aus Abschnitt 1.3, dass der Rekonstruktionskern linear bleibt, ist an dieser Stelle verletzt.
- Lokale Kontrastnormierung kann Sternfluesse, diffuse Flaechenhelligkeit und tile-uebergreifende Photometrie veraendern.
- Die Aussage in Zeile 487 und Zeile 503, dies sei noch Teil eines "linear affine normalization path", ist mathematisch nicht korrekt.

Empfehlung:

- Diese Tile-Normalisierung nicht als Teil des obligatorischen linearen Kerns definieren.
- Entweder komplett entfernen, oder klar als optionale, nichtlineare Naht-/Kontrast-Extension klassifizieren.
- Wenn Nahtstabilisierung noetig ist, besser nur additive Randanpassungen oder partion-of-unity-faehige Blend-Gewichte verwenden, nicht datenabhaengige lokale Kontrastnormalisierung.

### 3.2 Positivitaetsbedingung `> 0` erzeugt Bias

Referenzen:

- `5.7 Tile Reconstruction`, Zeilen 421-452
- `5.7.1`, Zeilen 458-467
- `5.7.1a`, Zeilen 477-485

Befund:

Mehrfach wird "gueltig" als `finite and > 0` definiert. Das ist fuer linear kalibrierte Daten problematisch. Nach Bias-/Dark-Subtraktion oder lokaler Hintergrundentfernung koennen valide Pixel negativ oder null sein.

Ein einfaches Gegenbeispiel:

- zwei valide Samples `[-1, +1]`
- korrekter Mittelwert: `0`
- nach der Spezifikation bleibt nur `+1` gueltig
- Ergebnis: `+1`

Das ist ein systematischer Positiv-Bias.

Konsequenz:

- Hintergrund wird nach oben verzerrt.
- Rauschstatistik wird asymmetrisch.
- schwache diffuse Signale werden instabil bewertet.
- der lineare Schaetzer wird implizit beschnitten, obwohl kein physikalischer Grund fuer das Verwerfen negativer Werte vorliegt.

Empfehlung:

- Gueltigkeit ueber `is finite` und Canvas-/Masken-Support definieren, nicht ueber `> 0`.
- Wenn fuer einzelne spaetere Schritte positive Werte gebraucht werden, das dort separat und begruendet definieren.

### 3.3 Finale Cluster-Aggregation erhaelt die Cluster-Masse nicht

Referenzen:

- `5.10.1 Default (global)`, Zeilen 603-607
- `5.11.1`, Zeilen 626-632
- `5.11.2`, Zeilen 634-653

Befund:

Synthetische Frames werden innerhalb eines Clusters mit `G_{f,c}` gebildet:

`S_{k,c} = sum_{f in k} G_{f,c} * I_{f,c} / sum_{f in k} G_{f,c}`

Danach werden Cluster mit

`w_k = exp(kappa_cluster * Q_k)`

aggregiert.

Dabei geht die effektive Cluster-Masse verloren. Zwei Cluster mit gleicher Qualitaet, aber stark unterschiedlicher Anzahl bzw. Summe der Frame-Gewichte, tragen im letzten Schritt gleich stark bei.

Konkretes Beispiel:

- Cluster A: 1000 Frames, `Q_A = 0`
- Cluster B: 10 Frames, `Q_B = 1`
- dann `w_A = 1`, `w_B = e`

Cluster B haette im letzten Schritt etwa 73 % Einfluss, obwohl es nur 1 % der Frames enthaelt. Das ist als Default mathematisch nicht plausibel.

Zusatzproblem:

- Die Qualitaet wird doppelt benutzt: einmal innerhalb von `S_{k,c}` ueber `G_{f,c}`, dann nochmals ueber `w_k`.

Empfehlung:

- Clustergewichte muessen die Cluster-Masse enthalten, z. B.
  `w_{k,c} = M_{k,c} * exp(kappa_cluster * Q_{k,c})`
  mit `M_{k,c} = sum_{f in k} G_{f,c}` oder `M_{k,c} = |k|`.
- Alternativ: ganz auf Phase 9/10 verzichten, wenn Phase 7 bereits die lokal gewichtete Rekonstruktion liefert.

### 3.4 Adaptive Gewichtung aus `Var(z(.))` ist mathematisch schwach

Referenzen:

- `5.3.1`, Zeilen 215-221
- `5.3.3`, Zeilen 243-252

Befund:

Die Metriken werden zuerst robust z-standardisiert. Danach sollen adaptive Gewichte proportional zu `Var(z(B))`, `Var(z(sigma))` und `Var(z(E))` gesetzt werden.

Das ist nur begrenzt sinnvoll, weil die z-Normierung die Skalen bereits weitgehend auf vergleichbare Groessenordnung bringt. Die Varianz der z-Werte misst danach primaer Restform, Tail-Last oder Ausreisserstruktur, aber nicht direkt den Informationsgehalt der Metrik fuer die Qualitaet.

Konsequenz:

- Die adaptiven Gewichte werden leicht instabil.
- In vielen Datensaetzen entstehen nahezu beliebige kleine Differenzen um etwa gleiche Varianzen.
- Die Regel ist mathematisch nicht gut an die beabsichtigte Semantik "wichtigeres Kriterium bekommt mehr Gewicht" gekoppelt.

Empfehlung:

- Entweder feste Gewichte belassen.
- Oder adaptive Gewichte aus echter Praediktivitaet ableiten, z. B. Korrelation mit Referenz-FWHM, Stern-Flux-Stabilitaet oder holdout-basiertem Qualitaetsgewinn.

### 3.5 Mehrere `clip(...)`-Definitionen sind nicht wohldefiniert

Referenzen:

- `5.4 Tile Geometry`, Zeilen 256-288
- `6.3.9 Adaptive Grid Definition`, Zeilen 914-935

Befund A, Tile-Groesse:

`T = floor(clip(T0, T_min, floor(min(W,H)/D)))`

ist nicht definiert, wenn

`floor(min(W,H)/D) < T_min`.

Beispiel:

- `min(W,H) = 48`
- `D = 8`
- obere Grenze `= 6`
- untere Grenze `= 16`

Dann ist `clip(x, 16, 6)` logisch unklar.

Befund B, BGE-Gitter:

`G = clip(max(2*T, min(W,H)/N_g), G_min, G_max)`

mit `G_max = min(W,H)/4` kann die eigene Forderung `G >= 2*T` verletzen. Wenn `2*T > G_max`, liefert `clip(...)` gerade ein `G < 2*T`.

Zusaetzlich gilt fuer kleine Bilder:

- wenn `min(W,H) < 256`, dann `G_max < 64 = G_min`
- auch hier kippen die Clip-Grenzen um

Empfehlung:

- Bounds vor dem Clipping deterministisch ordnen.
- `T_max` und `G_max_eff` explizit definieren.
- Garantien wie `G >= 2*T` duerfen nicht gleichzeitig mit einem kleineren `G_max` gefordert werden.

### 3.6 Sigma-Clipping: fehlender Guard fuer effektive Freiheitsgrade

Referenzen:

- `5.7 Tile Reconstruction`, Zeilen 433-452

Befund:

Die gewichtete Standardabweichung verwendet

`V1 - V2/V1`

im Nenner. Das ist nur stabil, wenn die effektive Stichprobengroesse groesser als 1 ist. Bei stark konzentrierten Gewichten kann dieser Term numerisch gegen 0 gehen, obwohl viele Frames vorhanden sind.

Konsequenz:

- `sigma^{(k)}` kann explodieren oder NaN werden.
- Das Clipping-Verhalten wird in genau den Faellen instabil, in denen bereits eine starke Gewichtsdominanz vorliegt.

Zusatzproblem:

Die Bedingung

`|A^{(k+1)}| >= ceil(min_fraction * |V_{t,c}(p)|)`

ist als Constraint formuliert, aber nicht algorithmisch definiert. Wenn sie verletzt wird, bleibt offen:

- vorherige Menge behalten,
- weniger aggressiv clippen,
- oder auf unclipped mean zurueckfallen.

Empfehlung:

- expliziten Guard auf `N_eff = V1^2 / V2` einfuehren.
- bei `N_eff <= 2 + eps` direkt auf den gewichteten Mittelwert ohne Clipping zurueckfallen.
- Keep-Floor als deterministische Regel formulieren.

### 3.7 Optionalitaet und Normativitaet sind nicht immer konsistent

Referenzen:

- `3. Pipeline Overview`, Zeilen 118-133
- `5.9 State-Based Clustering (Full Mode)`, Zeilen 585-597
- `7.3 Minimum Tests`, Zeilen 1000-1014
- `9. Scope Boundary`, Zeilen 1031-1046

Befund A, Clustering:

- Abschnitt 2.3 erlaubt auch im Reduced Mode clustering-bezogene Konfiguration.
- Abschnitt 5.9 nennt Clustering "Full Mode" und zugleich `Active only for N >= 200`.

Damit bleiben unklare Faelle, z. B.:

- `N >= N_red`, aber `N < 200`
- Reduced Mode mit explizit erlaubter Cluster-Suche

Die Aktivierungslogik ist nicht sauber geschlossen.

Befund B, WCS/PCC:

- WCS/PCC werden in Abschnitt 3 und 9 als optional bezeichnet.
- In `7.3 Minimum Tests` sind WCS- und PCC-Tests trotzdem als normative Minimum Tests aufgefuehrt.

Das ist eine klare Scope-Inkonsistenz.

Empfehlung:

- Aktivierungsbedingungen fuer Clustering exakt auf einen einzigen Mechanismus reduzieren.
- WCS/PCC-Tests nur dann verpflichtend machen, wenn diese Erweiterungen aktiviert sind.

## 4. Weitere relevante Schwachstellen

### 4.1 Hintergrund-Median als Normierungsskala vermischt additive und multiplikative Effekte

Referenzen:

- `5.2 Global Linear Normalization`, Zeilen 194-209
- `5.3.2 Global Quality Index`, Zeilen 223-241

Befund:

Die Normierung

`I_{f,c} = I_{f,c}^{raw} / max(B_{f,c}, eps_bg)`

benutzt den globalen Hintergrundmedian als Multiplikativfaktor. Das ist physikalisch nicht sauber, weil Himmelshintergrund primär ein additiver Stoerterm ist, waehrend Transparenz bzw. Durchsatz ein multiplikativer Effekt ist.

Beispiel:

- identisches Objektsignal,
- gleicher optischer Durchsatz,
- aber doppelter Hintergrund durch Mond oder Lichtverschmutzung

Dann wird das gesamte Bild halbiert, obwohl sich das Objektsignal selbst nicht halbiert hat.

Konsequenz:

- Photometrische Skalierung und Qualitaetsbewertung werden vermischt.
- `B` geht danach nochmals mit negativem Vorzeichen in `Q_{f,c}` ein, also faktisch doppelt.

Empfehlung:

- Additiven Hintergrund und photometrische Skalierung trennen.
- Besser:
  `I_norm = (I_raw - b_f) / s_f`
  mit `b_f` als robustem Hintergrund und `s_f` als photometrischer Skala aus Stern-Flux, Referenzsternen oder Belichtungs-/Gain-Kalibrierung.

### 4.2 Hard Switch STAR vs. STRUCTURE kann lokal instabil sein

Referenzen:

- `5.5.1`, Zeilen 294-297
- `5.5.2`, Zeilen 299-307
- `5.5.3`, Zeilen 309-316

Befund:

Die Klassifikation ist hart. Tiles knapp ueber oder unter `tile.star_min_count` wechseln schlagartig das Modell. Das kann bei grenzwertigen Tiles oder zwischen benachbarten Tiles zu Modellflackern fuehren.

Empfehlung:

- Soft-Mixture statt Hard Switch.
- Zum Beispiel eine kontinuierliche Mischung zwischen STAR- und STRUCTURE-Modell in Abhaengigkeit von Sternanzahl und Strukturkonfidenz.

### 4.3 Raeumliche Regularisierung sollte support- und kantenbewusst sein

Referenzen:

- `5.5.5`, Zeilen 353-388

Befund:

Die Regularisierung ist ein einfacher Nachbarschaftsmittelwert. Das glattet auch echte, physikalisch sinnvolle lokale Unterschiede, etwa PSF-Aenderungen zum Rand oder lokale Registrierungsfehler.

Empfehlung:

- Gewichte mit Tile-Konfidenz, Sternsupport oder metrischer Unsicherheit versehen.
- Fuer bessere Qualitaet eher anisotrope oder confidence-weighted Regularisierung nutzen.

### 4.4 Hann-Fenster am Bildrand nicht vollstaendig spezifiziert

Referenzen:

- `5.7.2`, Zeilen 505-522

Befund:

Die diskrete Hann-Funktion ist an den Tile-Raendern null. Ohne explizite Randbehandlung, Padding oder spezielle Randfenster kann `S` am Bildrand null werden. Dann erzeugt

`I_rec = A / max(S, eps_weight)`

am Rand dunkle Pixel oder unkontrollierte Epsilon-Normalisierung.

Empfehlung:

- Randfenster, Padding oder explizite support-korrigierte partition-of-unity-Fenster definieren.
- Alternativ nur Fensterfamilien verwenden, fuer die Schrittweite und Ueberlappung exakt zur Fenster-Summe passen.

### 4.5 BGE-Autotuning bestraft moeglicherweise reale Gradienten

Referenzen:

- `6.3.7.1`, Zeilen 808-819
- `6.3.7.2`, Zeilen 821-825

Befund:

Das Ziel

`J = E_cv + alpha * E_flat + beta * E_rough`

enthaelt `E_flat`, also eine Strafe auf grosse Flaechengradienten des Modells selbst. Bei real vorhandenen, glatten Himmelsgradienten kann das in Richtung Unterkorrektur treiben.

Die lineare Sortierung und "jedes k-te Feld" als Holdout ist zudem raeumlich schwach und kann aliasing-artige Validationsmuster erzeugen.

Empfehlung:

- eher die Rauheit des Modells und die Restgradienten nach Subtraktion bewerten,
- fuer Validation besser checkerboard- oder blockbasierte raeumliche Splits verwenden.

## 5. Priorisierte Verbesserungen fuer bessere Bildqualitaet

### Prioritaet 1: Photometrisch saubere globale Normierung

Trenne strikt:

- additive Hintergrundschaetzung,
- photometrische Skalierung,
- und Qualitaetsgewichtung.

Das wird wahrscheinlich die groesste reale Verbesserung fuer Stern-Flux-Stabilitaet, Farbkonsistenz und BGE/PCC-Verhalten bringen.

### Prioritaet 2: Tile-Normalisierung aus dem Kern entfernen

Wenn der Kern linear bleiben soll, darf die per-Tile-Median/MAD-Normierung nicht im Pflichtpfad liegen. Fuer Nahtfreiheit sind geometrisch saubere Ueberlappung, support-aware Fenster und bessere lokale Gewichtsschaetzung der robustere Weg.

### Prioritaet 3: Finale Aggregation massenerhaltend machen

Cluster muessen proportional zu ihrer evidenztragenden Masse beitragen. Sonst verliert man Integrationszeit und riskiert instabile, vom Zufall der Clusterbildung getriebene Endbilder.

### Prioritaet 4: Gueltige Samples nicht ueber Vorzeichen definieren

Das ist fuer lineare Astronomiedaten zentral. Sobald negative Werte zugelassen werden, werden Hintergrund, Rauschen und diffuse Signale deutlich sauberer behandelt.

### Prioritaet 5: Softes lokales Qualitaetsmodell statt harter Schalter

Eine weiche Mischung aus Sternmetriken, Strukturmetriken und Konfidenz waere qualitativ meist besser als ein binarer STAR/STRUCTURE-Switch.

### Prioritaet 6: Bessere Regularisierung und Randbehandlung

Seam-Artefakte entstehen oft eher aus OLA-/Rand-/Support-Problemen als aus fehlender lokaler Normierung. Hier lohnt mehr Sorgfalt als bei spaeteren Korrekturtricks.

## 6. Konkrete Spezifikationskorrekturen

### 6.1 Gueltigkeitsdefinition

Statt:

`I_{f,c}(p) is finite and > 0`

besser:

`I_{f,c}(p) is finite and inside valid canvas support`

plus optional separate Masken fuer Sattigung, Defekte oder ausserhalb des Bildfelds.

### 6.2 Globale Normierung

Statt:

`I_{f,c} = I_{f,c}^{raw} / max(B_{f,c}, eps_bg)`

besser:

`I_{f,c}^{norm} = (I_{f,c}^{raw} - b_{f,c}) / max(s_{f,c}, eps_scale)`

mit:

- `b_{f,c}` additive Hintergrundschaetzung
- `s_{f,c}` photometrische Skala

### 6.3 Tile-Groesse

Statt eines nicht abgesicherten `clip(...)`:

- `T_hi = floor(min(W,H)/D)`
- wenn `T_hi < 16`, dann in kleinen Bildern deterministisch auf Sondermodus wechseln
- sonst `T = floor(clip(T0, T_min, T_hi))`

### 6.4 Sigma-Clipping

Ergaenzen:

- Guard auf `N_eff`
- Guard auf `V1 - V2/V1 > eps_var`
- deterministische Regel fuer Keep-Floor-Verletzung

### 6.5 Cluster-Aggregation

Empfohlene robuste Form:

`M_{k,c} = sum_{f in k} G_{f,c}`

`w_{k,c} = M_{k,c} * exp(kappa_cluster * (Q_{k,c} - median_j(Q_{j,c})))`

`R_c = sum_k w_{k,c} S_{k,c} / sum_k w_{k,c}`

Damit bleibt erhalten:

- Qualitaetspriorisierung
- Cluster-Support
- und eine bessere Naeherung an den direkten gewichteten Gesamtschaetzer

### 6.6 Optionale Module auch testseitig optional halten

Normative Minimum Tests sollten nur den obligatorischen Kern pruefen. Tests fuer WCS, PCC, BGE-Autotune oder ML sollten als bedingte Tests formuliert werden.

## 7. Empfohlene Zusatztests

Die aktuellen Minimum Tests sind fuer den Kern nicht ausreichend. Ergaenzen wuerde ich mindestens:

1. Test auf Linearitaetsverletzung des Pflichtpfads.
2. Test, dass negative, aber valide Pixel nicht verworfen werden.
3. Test auf Randabdeckung der OLA-Fenster ohne dunklen Saum.
4. Test auf massenerhaltende Cluster-Aggregation bei ungleichen Cluster-Groessen.
5. Test auf `clip`-Guards bei kleinen Bildern.
6. Test auf stabiles Verhalten bei `N_eff -> 1`.
7. Test, dass optionale Module nur dann Pflicht-Tests ausloesen, wenn sie aktiviert sind.

## 8. Gesamteinschaetzung

Die Methodik hat eine gute Grundidee: keine harte Frame-Selektion, robuste lokale Gewichtung, deterministische Fallbacks und klare Trennung zwischen Kern und Erweiterungen. Der aktuelle Text enthaelt aber noch einige Stellen, an denen der mathematische Anspruch des Dokuments und die tatsaechlich spezifizierten Operationen auseinanderlaufen.

Aus Qualitaetssicht ist die wichtigste Richtung:

- photometrisch sauberer normieren,
- die lokale Rekonstruktion wirklich linear halten,
- die finale Aggregation evidenztreu machen,
- und die Rand-/Support-Logik staerker absichern.

Wenn genau diese Punkte korrigiert werden, sollte die Methodik nicht nur formaler sauber werden, sondern auch sichtbar bessere und stabilere Ergebnisse liefern.
