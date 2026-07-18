# Adaptive Quality Map Hyperstacking (AQMH) – Methodik v0.2.1

**Dokumentstatus:** Normativ für die Konfigurationsfamilie `v0.2.1`.  
**Ersetzt:** `aqmh_methodik_en_v0.2.0.md` bzw. deren deutsche Übersetzung.  
**Sprache:** Deutsch. Für die englische Version siehe `aqmh_methodik_en_v0.2.1.md`.

---

## 0. Dokumenthistorie und Geltungsbereich

### 0.1 Änderungen gegenüber v0.2.0

Version `v0.2.1` ist eine strikte Obermenge von `v0.2.0`. Alle mathematischen
Definitionen, Invarianten, normativen Vorgabewerte und diagnostischen
Anforderungen aus `v0.2.0` bleiben verbindlich. `v0.2.1` formalisiert vier
Erweiterungen, die in der Referenzimplementierung nach der `v0.2.0`-Basis
hinzugekommen sind:

1. **Hintergrundgradienten-Strafe in der globalen Frame-Qualität (§1.5, §4.2).**  
   Ein drittes globales Qualitätssignal bestraft Frames mit starkem
   großskaligem Hintergrundgradienten (z. B. Lichtverschmutzung, Mondglühen).
   Das Signal wird aus einem quadrantenbasierten Sky-Gradient-Schätzer
   gewonnen, nicht aus der AQMH-Quality-Map selbst. Es wird daher als
   zulässige **Cross-Infrastructure-Erweiterung** dokumentiert, die mit dem
   AQMH-Unabhängigkeitsprinzip (§0.3) vereinbar bleibt.

2. **Registrierungs-Gewichtsschutz (§4.3).**  
   Vor der pixelweisen gewichteten Rekonstruktion kann das globale AQMH-Gewicht
   jedes Frames mit einem Registrierungs-Konfidenzfaktor gedämpft werden, der
   aus dem gemeinsamen Global-Registration-Artefakt abgeleitet wird. Dieser
   Schutz ist eine reine Robustheitsmaßnahme; er ändert nicht die
   Rekonstruktionsformel und bleibt ein Frame-Multiplikator.

3. **Adaptive Niederfrequenz-Neutralisierung und strukturmaskierte
   Detail-Blending (§6.3, §6.4).**  
   Nach der gewöhnlichen AQMH-Rekonstruktion kann eine optionale
   Nachverarbeitung niederfrequente „Veil“-Residuen entfernen, indem eine
   stark geglättete Differenz zwischen dem AQMH-Ergebnis und einem
   uniformen Kontrollmittel (ungewichteter Mittelwert) subtrahiert wird.
   Anschließend wird Detail in strukturreichen Regionen selektiv
   zurückgeblendet. Ein Validierungsgate entscheidet, ob das
   neutralisierte/geblendete Ergebnis übernommen wird; andernfalls bleibt das
   rohe AQMH-Ergebnis erhalten. Das uniforme Kontrollmittel darf das rohe
   AQMH-Ergebnis nicht als stillen Fallback ersetzen.

Ziel dieser Ergänzungen ist es, das v0.2.0-Pixelqualitätsmodell zu erhalten,
während drei praktische Fehlermuster reduziert werden:

- Großskalige Hintergrundgradienten, die den finalen Stack dominieren würden,
  weil jeder Frame einen ähnlichen additiven Gradienten trägt.
- Restschwächen der Registrierung (niedrige Korrelation oder tief verkettete
  Vorhersage-Frames), die durch hohe AQMH-Qualitätswerte verstärkt werden.
- Niederfrequenter „Veil“ oder Hintergrund-Nichtlinearität, der durch die
  gewichtete Rekonstruktion selbst entstehen kann.

---

## 0.2 Unabhängigkeit und gemeinsame Infrastruktur

AQMH bleibt eine **unabhängige Rekonstruktionsmethode**. Sie darf gemeinsame
Pipeline-Infrastruktur wiederverwenden, aber ihr Qualitätsmodell und ihre
Rekonstruktionsgewichte dürfen nicht aus den lokalen/Tile-Metriken von
Classic Tile Compile abgeleitet sein.

Gemeinsame Infrastruktur kann umfassen:

- Input-Scan und nicht-qualitätsbasierte Eligibilitätsfilter (Lesbarkeit,
  Verfügbarkeit von Kalibration, explizite Benutzer-Ausschlüsse);
- Kalibration und Registrierung/Prewarping;
- globale photometrische Normalisierung;
- globale Output-Canvas-Maske `C` und frame-spezifische registriert-gültige
  Masken `M_f`;
- Run-Management, Logging, Artefakte, Reports, UI-Plumbing.

Der AQMH-Algorithmus selbst besteht aus:

- Berechnung dichter AQMH-Quality-Maps;
- pixelweiser gewichteter AQMH-Rekonstruktion;
- AQMH-Diagnostik und optionaler Region-Extraktion.

Classic Tile Compile und AQMH müssen unabhängig voneinander ausführbar sein.
Das Aktivieren oder Deaktivieren einer Methode darf die mathematische
Definition der anderen nicht verändern.

### 0.3 Cross-Infrastructure-Erweiterungen

`v0.2.1` erlaubt ausdrücklich die folgenden sorgfältig abgegrenzten Eingaben
aus der gemeinsamen Infrastruktur, sofern sie dokumentiert sind und keine
AQMH-eigenen Signale ersetzen:

- **Sky-Gradient-Zusammenfassung** für die Hintergrundstrafe (§1.5).  
  Dies ist ein Skalar pro Frame, abgeleitet aus dem hintergrundmaskierten
  Eingabebild nach der Registrierung. Er wird nur innerhalb des
  globalen Qualitäts-Sigmoid verwendet und bleibt orthogonal zur Berechnung
  der within-frame `Q_map`.

- **Registrierungs-Konfidenz-Metadaten** für den Gewichtsschutz (§4.3).  
  Dies ist ein Skalarmultiplikator pro Frame, abgeleitet aus dem bestehenden
  Global-Registration-Artefakt. Er modifiziert den globalen Faktor `G_f`,
  beeinflusst aber nicht `Q_map`.

- **Uniformes Kontrollmittel** für die Neutralisierungsvalidierung (§6.3).  
  Der ungewichtete Mittelwert der registrierten Frames wird als Nebenprodukt
  der AQMH-Rekonstruktion berechnet (`compute_uniform_control = true`). Er
  wird nur für die Validierung und optionales Blending nach der Rekonstruktion
  verwendet, nicht als primärer Qualitätseingang.

---

## 1. Prinzipien und Definitionen

### 1.1 Physikalisches Ziel

Die Methode modelliert die pixelweise Beobachtungsqualität als Produkt zweier
separabler Komponenten:

- **Frame-Level-Qualität:** der globale atmosphärische/technische Zustand des
  Frames `f`, erfasst durch den strikt positiven AQMH-Faktor `G_{f,c}`.
- **Räumliches Qualitätsfeld:** die kontinuierliche Qualitätsverteilung im
  Frame, erfasst durch `Q_map_{f,c}(x,y)`.

Für jeden Frame/Kanal leiten wir die globalen Pre-Z-Score-Zusammenfassungen
aus seinen AQMH-Maps ab:

```
g_sharp_{f,c}  = median_{p source-valid}(Phi_sharp_0(p))
g_snr_{f,c}    = median_{p source-valid}(Phi_snr_1(p))
```

wobei der feinste verfügbare Skala verwendet wird, falls Skala 1 ausgelassen
wird.

### 1.2 Effektives Pixelgewicht

Das effektive pixelweise Rekonstruktionsgewicht ist:

```
W_{f,c}^{aqmh}(x,y) = G_{f,c} * Q_map_{f,c}(x,y)
```

`v0.2.1` erlaubt zusätzlich einen Frame-weise Registrierungs-Konfidenz-
Multiplikator `R_f ∈ [r_floor, 1]` (§4.3). Wenn aktiviert, lautet das Gewicht:

```
W_{f,c}^{aqmh}(x,y) = G_{f,c} * R_f * Q_map_{f,c}(x,y)
```

`R_f` ist ein gemeinsamer Infrastruktur-Schutz, kein Bestandteil des
AQMH-Qualitätsmodells. Er muss in `aqmh_reconstruction.json` protokolliert
werden.

### 1.3 Invarianten (verbindlich)

Die folgenden Invarianten aus v0.2.0 bleiben verbindlich:

1. **Keine Frame-Selektion:** Ganze Frames dürfen nicht auf Basis der Qualität
   entfernt werden.
2. **Bedingte photometrische Linearität:** Sobald deterministische Gewichte
   berechnet sind, bleibt die finale Rekonstruktion `R(p) = sum_f w_f(p) *
   I_f(p) / sum_f w_f(p)` mit `w_f(p) >= 0`. AQMH darf keine nichtlinearen
   Intensitätstransformationen auf die in den Akkumulator eingehenden Samples
   anwenden.
3. **Determinismus:** Alle Quality-Map-Berechnungen müssen deterministisch und
   reproduzierbar sein.
4. **Canvas-Ausschluss:** Canvas-ungültige Pixel werden aus allen AQMH-
   Akkumulatoren und Statistiken ausgeschlossen.
5. **Keine Halluzination:** AQMH-Ausgaben sind Gewichte und Masken. Es werden
   keine Pixelintensitäten generiert oder vorhergesagt.
6. **Sample-Count-Suffizienz für optionale Selektionsmodi:** Cherry-Pick und
   jeder andere per-Pixel-Selektionsmodus müssen eine dokumentierte minimale
   Anzahl beibehaltener Samples erzwingen; unterschreitet sie, wird der Modus
   automatisch deaktiviert.

`v0.2.1` ergänzt die folgenden abgeleiteten Invarianten für die neuen
Erweiterungen:

7. **Cross-Infrastructure-Erweiterungen müssen monotone Frame-Level-Schütze
   sein.**  
   Der Registrierungs-Gewichtsschutz und die Hintergrundstrafe dürfen das
   globale Frame-Gewicht nur mit einem nicht-negativen Skalar multiplizieren.
   Sie dürfen `Q_map` nicht verändern, dürfen keine pixelabhängigen Gewichte
   einführen und müssen unabhängig deaktivierbar sein.

8. **Nachverarbeitung muss zwei Referenzen bestehen.**
   Das rohe AQMH-Ergebnis `A` ist die unveränderliche Qualitäts-Baseline. Jeder
   nachverarbeitete Kandidat muss sowohl gegen das uniforme Kontrollmittel `U`
   als auch gegen `A` alle jeweils anwendbaren Regressionsschwellen bestehen.
   Besteht kein Kandidat beide Vergleiche, ist `A` das Endergebnis. Der Run muss
   den ausgewählten Kandidaten, beide Vergleiche und den Auswahlgrund
   aufzeichnen. Eine Verbesserung einer Einzelmetrik darf keine Lockerung einer
   anderen Schwelle auslösen.

### 1.4 Deterministische Statistik-Konvention

(Identisch zu v0.2.0 §1.4.)

Überall wird der robuste Z-Score verwendet: `z(x) = (x - med) / (1.4826 * MAD)`
über finite, source-valid Samples auf derselben Aggregationsebene. Degenerierte
Mengen (MAD = 0 oder weniger als drei finite Samples) erhalten für gültige
Eingaben Z-Score 0 und werden andernfalls als ungültig markiert.

### 1.5 Globale Qualität mit Hintergrundstrafe (v0.2.1-Ergänzung)

`v0.2.1` führt ein optionales drittes globales Qualitätssignal ein:

```
g_background_{f,c} = sky_gradient_f
```

wobei `sky_gradient_f` ein dimensionsloser relativer Großskalengradient ist:

```
sky_gradient_f = (q_max - q_min) / background_f
```

- `background_f` ist der maskierte Medianhintergrund von Frame `f`.
- Das Bild wird in vier Quadranten unterteilt. Für jeden Quadranten `q` wird
  der maskierte Medianhintergrund `b_q` berechnet.
- `q_min = min_q(b_q)`, `q_max = max_q(b_q)`.

Ist `background_f <= 0` oder gibt es weniger als vier gültige Quadrantenwerte,
so ist die Zusammenfassung ungültig.

Die Hintergrundstrafe geht als **subtraktiver** Term in die globale Qualität ein:

```
z_s,f = robust_zscore(g_sharp)
z_n,f = robust_zscore(g_snr)
z_b,f = robust_zscore(g_background)

score_f = w_sharp * z_s,f + w_snr * z_n,f - w_background * z_b,f
G_f     = g_floor + (1 - g_floor) * sigmoid(g_k_scale * score_f)
```

mit `sigmoid(v) = 1 / (1 + exp(-v))`.

Die Gewichte `w_sharp`, `w_snr`, `w_background` werden direkt als
`g_w_sharp`, `g_w_snr`, `g_w_background_penalty` konfiguriert. Zur Laufzeit
werden sie in **effektive Gewichte** `w_*_eff` umgerechnet:

- Ist ein konfiguriertes Gewicht `<= 0`, so ist sein effektives Gewicht `0`.
- Gibt es weniger als drei finite positive Zusammenfassungswerte, so ist das
  effektive Gewicht `0`.
- Ist der Variationskoeffizient `CV = MAD / median` der endlichen positiven
  Zusammenfassungswerte unter `0.01`, so ist das effektive Gewicht `0`. Damit
  wird verhindert, dass nahezu konstante Signale Rauschen verstärken und die
  Qualitätsrangfolge invertieren.

Die verbleibenden effektiven Gewichte werden so renormiert, dass gilt:

```
w_sharp_norm + w_snr_norm + w_background_norm = 1
```

sofern mindestens ein effektives Gewicht positiv ist. Sind alle effektiven
Gewichte null, ist `score_f = 0` und jeder Frame erhält dasselbe Gewicht
`G_f = g_floor + 0.5 * (1 - g_floor)`. Die absolute gemeinsame Skalierung ist
für den gewichteten Mittelwert irrelevant.

**Normative Vorgabewerte (v0.2.1):**

- `g_floor = 0.03`
- `g_w_sharp = 0.55`
- `g_w_snr = 0.30`
- `g_w_background_penalty = 0.25`
- `g_k_scale = 1.5`

Die Sigmoid-Abbildung begrenzt `G_f` auf `[g_floor, 1]`. Dadurch kann kein
einzelner Frame durch ein exponentiell wachsendes globales Gewicht die lokalen
Quality-Maps dominieren. Mit `g_w_background_penalty = 0.0` wird nur die
Hintergrundstrafe deaktiviert; die begrenzte v0.2.1-Abbildung bleibt aktiv.

**Begründung.** Ein starker Großskalengradient bedeutet, dass das vom SNR-
Signal verwendete Hintergrundmodell (`b_s` in §2.3.2(b)) lokal verzerrt ist.
Frames mit sehr flachem Hintergrund tendieren zu saubereren und lineareren
Bedingungen; Frames mit starkem Mondglühen- oder Lichtverschmutzungsgradienten
erhalten ein niedrigeres globales Gewicht, ohne jedoch ganz entfernt zu werden.

**Kompatibilitätshinweis.** Die Hintergrundstrafe ist nicht aus `Q_map`
abgeleitet. Sie ist daher eine **Erweiterung** von v0.2.0, keine
Widersprüchlichkeit. Sie muss in `aqmh_metrics.json` als Frame-Feld
`global_background_penalty_input` mit der Quelle
`global_background_penalty_source: "sky_gradient"` dokumentiert werden. Das
konfigurierte Strafgewicht gehört zur AQMH-Global-Quality-Konfiguration und
nicht zum Artefakt `registration_weight_guard`.

---

## 2. Quality-Map-Berechnung

### 2.1 Übersicht

Identisch zu v0.2.0. AQMH berechnet für jedes Frame/Kanal-Paar eine dichte
Quality-Map durch Fusionierung mehrskaliger lokaler Schärfe-, SNR- und
Artefakt-Anomalie-Signale.

### 2.2 Gemeinsame Vorverarbeitung

Identisch zu v0.2.0. Alle Frames müssen vor der AQMH-Map-Berechnung kalibriert,
registriert und auf einen gemeinsamen Output-Canvas prewarped sein.

### 2.3 Mehrskalige Pyramide

Identisch zu v0.2.0 §2.3. Die vier Skalen, die Auslassungsregel, die Berechnung
von Schärfe, SNR und Artefakt sowie die geometrische Mittelwertfusion bleiben
unverändert.

### 2.4 Mehrskalen-Fusion

Identisch zu v0.2.0 §2.4.

```
Q_map_{f,c}(x,y) = ( prod_{s in S_actual} Psi_s^{up}(x,y) )^{1/P_actual}
```

implementiert als `exp(mean_s(log(Psi_s^{up})))` mit dem Exact-Zero-Veto und
dem Output-Guard.

### 2.5 Block-Level-Diagnose-Zusammenfassungen

Identisch zu v0.2.0 §2.5.

---

## 3. Quality-Map-Speicher und Speichermodell

Der produktive, objektklassenunabhängige Standard ist `uint16` mit
`resolution_divisor = 2`. Der strikte Referenz- und Fidelity-Modus bleibt
Full-Resolution `float32` mit `resolution_divisor = 1`. Cherry-Pick erfordert
den Referenzmodus, weil eine reduzierte oder quantisierte Map die per-Pixel-
Rangfolge verändern kann. Jeder Run muss Auflösung und Datentyp berichten.

Das Hintergrundstrafe-Signal beeinflusst den Quality-Map-Speicher nicht; es ist
ein Frame-skaliger Wert, der während oder nach der Map-Berechnung ermittelt
und in `aqmh_metrics.json` abgelegt wird.

---

## 4. Pipeline-Integration

### 4.1 AQMH-Verarbeitungsstufen

Identisch zu v0.2.0:

```
AQMH_MAPS
AQMH_GLOBAL_QUALITY
AQMH_RECONSTRUCTION
AQMH_DIAGNOSTICS
AQMH_NATIVE_BGE_INPUTS
```

### 4.2 Global-Quality-Stufe (v0.2.1-Aktualisierung)

Die Stufe berechnet nun pro Frame die drei Zusammenfassungsvektoren
`g_sharp`, `g_snr`, `g_background` und übergibt sie an
`compute_aqmh_global_quality`. Das Ergebnis ist ein Vektor `G_f` pro Frame.

Alle drei Eingaben werden pro Frame in `aqmh_metrics.json` erfasst:

- `global_sharpness_input`
- `global_snr_input`
- `global_background_penalty_input`
- `global_background_penalty_source: "sky_gradient"`
- `global_quality_input_invalid`
- `global_quality` (der finale `G_f`)

### 4.3 Registrierungs-Gewichtsschutz (v0.2.1-Ergänzung)

Vor der Rekonstruktion kann der Runner optional einen
**Registrierungs-Gewichtsschutz** auf `G_f` anwenden. Er liest das gemeinsame
`global_registration.json`-Artefakt, das pro Frame die Felder `cc`
(Kreuzkorrelations-Konfidenz), `source` (Registrierungs-Quelltyp) und optional
`chain_depth` enthalten muss.

Für jeden Frame `f`:

1. **Kreuzkorrelations-Abbildung.**

   ```
   t = clamp( (cc_f - cc_floor) / (cc_full - cc_floor), 0, 1 )
   r_cc = r_floor + (1 - r_floor) * t
   ```

   mit den Vorgaben `r_floor = 0.30`, `cc_floor = 0.35`, `cc_full = 0.80`.

2. **Dämpfung nach Registrierungsquelle.**

   - Falls `source == "sequential_refined"`: `r_f *= sequential_factor`
     (Vorgabe `0.92`).
   - Falls `source` den String `"predicted"`, `"interpolated"` enthält oder
     `"unknown"` ist: `r_f *= predicted_factor` (Vorgabe `0.50`).

3. **Kettentiefen-Dämpfung.**

   ```
   depth_penalty = min( depth_max, max(0, depth_f - 1) * depth_penalty_per_step )
   r_f *= (1 - depth_penalty)
   ```

   mit den Vorgaben `depth_penalty_per_step = 0.03`, `depth_max = 0.15`.

4. **Clamping.**

   ```
   R_f = clamp(r_f, r_floor, 1.0)
   ```

Das effektive globale Gewicht in der Rekonstruktion lautet:

```
G_f^{eff} = G_f * R_f
```

Der Schutz ist standardmäßig aktiviert (`registration_weight_guard: true`) und
wird in `aqmh_reconstruction.json` unter `registration_weight_guard` berichtet.

**Begründung.** Tief verkettete oder schwach korrelierte Frames können die
Eligibilitätsfilterung überstehen, aber dennoch eine geringere
Registrierungsgenauigkeit besitzen. Der Schutz verhindert, dass hohe
AQMH-Qualitätswerte Registrierungsfehler verstärken. Da er ein Multiplikator
aus der gemeinsamen Infrastruktur ist, wird er als **Robustheits-
erweiterung** dokumentiert, nicht als Bestandteil des AQMH-Kernqualitätsmodells.

### 4.4 Vorverarbeitung der Registrierungs-NCC (v0.2.1-Ergänzung)

Vor den NCC-Vergleichen der Registrierung werden die normalisierten Proxy-Bilder
für eine robuste Korrelation vorbereitet:

1. Begrenze das Proxy-Bild auf nichtnegative Werte, um negative Werte aus der
   Hintergrundsubtraktion zu entfernen.
2. Wende einen Gaussian-Blur mit `sigma = 1.5 px` an, um den Einfluss von
   Hotpixeln und isolierten Defekten zu reduzieren.
3. Berechne sowohl die Identity-Overlap-NCC als auch die Warped-Overlap-NCC aus
   diesen vorbereiteten Bildern.

Ein Near-Identity-Ergebnis wird nur akzeptiert, wenn alle folgenden Bedingungen
erfüllt sind:

- Die Gesamtverschiebung ist kleiner als `star_inlier_tol_px`.
- Die absolute Rotation liegt unter `0.1°`.
- Die Warped-NCC liegt höchstens `0.02` unter der Identity-Overlap-NCC.
- Die Identity-Overlap-NCC ist größer als `0.7`.

Die letzte Bedingung verhindert, dass eine nahezu nullgroße Verschiebung mit
niedriger Korrelation akzeptiert wird, nur weil der Optimierer keine sinnvolle
Verschiebung gefunden hat. Diese Registrierungsänderung liegt vor AQMH und
verändert `Q_map` nicht; sie verbessert die Zuverlässigkeit der Metadaten
`cc`, `source` und `chain_depth`, die vom Registrierungs-Gewichtsschutz genutzt
werden.

---

## 5. Pixelweise gewichtete Rekonstruktion

### 5.1 Rekonstruktionsformel

Identisch zu v0.2.0. Für jeden Output-Kanal:

```
R(p) = sum_{f in V_c^I(p)} w_f(p) * I_f(p) / sum_{f in V_c^I(p)} w_f(p)
```

wobei `w_f(p) = G_f^{eff} * Q_map_{f,c}(p)` und `V_c^I(p)` die Menge der
Frames mit finite Intensität und finite Quality-Map an Pixel `p` ist.

Der geometrische Support jedes registrierten Frames ist Bestandteil von
`V_c^I(p)`. Prewarp-Pixel außerhalb dieses Supports müssen als ungültig
(`NaN` plus Frame-Support-Maske) erhalten bleiben; nullgefüllte Warpränder
dürfen niemals als reale Intensitätswerte in AQMH eingehen. Dabei gelten drei
getrennte Maskenrollen:

- Die Frame-Support-Maske beschreibt, welche Pixel ein einzelner Frame liefert.
- Die Common-Overlap-Maske dient ausschließlich Analyse, Validierung und
  Kalibrierung mit der konfigurierten Mindestüberdeckung.
- Die Output-Maske ist die Vereinigung der Frame-Support-Masken und erhält alle
  tatsächlich rekonstruierbaren Randstrukturen.

Die Common-Overlap-Maske darf weder die AQMH-Rekonstruktion noch BGE-, PCC- oder
HMS-Ausgaben auf den gemeinsamen Kern beschneiden. Pixel, welche
`min_n_eff` nicht erfüllen, bleiben als nicht ausreichend gestützte Pixel
diagnostizierbar; ihre Existenz darf jedoch nicht dazu führen, dass gültige
Nachbarpixel aus wenigen Frames mit nicht vorhandenen Nullsamples verdünnt
werden.

### 5.2 Sigma-Clipping und effektive Sample-Suffizienz

Iteratives asymmetrisches Sigma-Clipping verwendet standardmäßig
`clip_sigma_low = 2.0`, `clip_sigma_high = 1.5` und `clip_iterations = 4`.
Die strengere Oberseite unterdrückt positive Ausreißer wie Satellitenspuren,
Hotpixel und nicht vollständig maskierte Sterne, ohne die Unterseite gleich
stark abzuschneiden. Es gelten außerdem `min_n_eff = 2.0` und
`min_fraction = 0.40`. Reicht die Sample-Suffizienz nicht aus, darf kein
scheinbar qualitätsverbesserter Wert aus einer zu kleinen Restmenge entstehen.

### 5.3 Cherry-Pick-Stacking-Modus

Identisch zu v0.2.0 §5.3. Das Run-Level-`K_nominal_median`-Floor, die
Per-Pixel-Regel `K(p) = max(k_min_required, K_nominal(p))`, Tiering und die
Rank-Separation-Diagnostik bleiben unverändert.

---

## 6. Robustheits-Erweiterungen nach der Rekonstruktion (v0.2.1)

### 6.1 Uniformes Kontrollmittel

Wenn `compute_uniform_control: true` gesetzt ist, berechnet die
Rekonstruktionsroutine zusätzlich den ungewichteten Mittelwert der
registrierten Frames über dieselbe Menge gültiger Pixel. Dieses Mittel wird
**uniformes Kontrollmittel** `U(p)` genannt:

```
U(p) = sum_{f in V_c^I(p)} I_f(p) / |V_c^I(p)|
```

`U(p)` ist kein qualitätsgewichtetes Ergebnis; es ist ein diagnostischer
Referenzwert.

### 6.2 Validierungsmetriken

Das rohe AQMH-Ergebnis `A(p)` wird mit `U(p)` verglichen. Jeder
nachverarbeitete Kandidat wird zusätzlich mit `A(p)` verglichen. Dazu dienen
dieselben Metriken wie die v0.2.0-Validierung:

- `seam_score_regression`
- `fwhm_regression`
- `background_rms_regression`
- `tail11_abs_regression`
- `elongation_regression`

Für Tail- und Elongationsmetriken werden Sterne genau einmal in der jeweiligen
Referenz erkannt und anschließend in Kandidat und Referenz an denselben
Koordinaten vermessen. Unabhängige Sternerkennungen sind für einen
Regressionsvergleich unzulässig, weil sie unterschiedliche Sternpopulationen
und damit eine künstliche Regression erzeugen können. Tail und Elongation sind
nur bei mindestens zwölf gemeinsam messbaren Sternen anwendbar.

Diese Vergleiche und ihre Anwendbarkeit werden in
`aqmh_reconstruction.json` protokolliert. Nicht anwendbare Metriken gelten
nicht als Fehler, dürfen aber auch nicht als Verbesserung ausgelegt werden.

### 6.3 Adaptive Niederfrequenz-Neutralisierung

Ist das uniforme Kontrollmittel `U(p)` verfügbar, wird ein optionaler
neutralisierter Kandidat `N(p)` berechnet:

```
L(p) = GaussianBlur( A(p) - U(p), sigma = 96 px )
N(p) = A(p) - L(p)
```

Das Blurring verwendet den Reflektionsrandmodus REFLECT101. Die Sigma von
96 px wurde empirisch gewählt, um großskalige „Veil“-Residuen anzugehen, ohne
Struktur auf kleineren Skalen zu beeinträchtigen.

Die Vergleiche `compare(N, U)` und `compare(N, A)` werden berechnet. `N(p)`
wird gegenüber `A(p)` nur dann ausgewählt, wenn der Hintergrundvergleich
anwendbar ist, `A` gegenüber `U` tatsächlich einen schlechteren Hintergrund
besitzt, `N` diesen verbessert und beide Vergleiche alle Schwellen bestehen:

```
background_rms_regression(N, U) < background_rms_regression(A, U)
background_rms_regression(A, U) > 0
gate(N, U) && gate(N, A)
```

Das ausgewählte Basisbild wird `B(p)` genannt (entweder `A(p)` oder `N(p)`).

**Begründung.** Der per-Pixel-gewichtete Mittelwert kann in seltenen Fällen
eine niederfrequente Residuum erzeugen, die glatter ist als der wahre
Hintergrund. Die Subtraktion des geglätteten Unterschieds zum ungewichteten
Mittel entfernt diese additive Niederfrequenzkomponente, während höherfrequentes
Detail erhalten bleibt.

### 6.4 Strukturmaskiertes Detail-Blending

Nach Auswahl der Basis `B(p)` wird ein zweiter optionaler Kandidat berechnet,
der AQMH-Detail in strukturreichen Regionen bewahrt, während der glattere
Hintergrund von `B(p)` erhalten bleibt:

1. Berechne die Gradientenbetrag `grad(U)` des uniformen Kontrollmittels.
2. Erstelle eine weiche Strukturmaske `M_s(p)`, indem der Gradientenbetrag von
   dem Quantil `low_q = 0.40` auf das Quantil `high_q = 0.90` auf `[0, 1]`
   abgebildet und anschließend mit `sigma = 4 px` geglättet wird.
3. Berechne den Detail-Kandidaten:

   ```
   D(p) = U(p) + M_s(p) * (B(p) - U(p))
   ```

   In strukturreichen Regionen (`M_s ≈ 1`) folgt `D(p)` der AQMH-Basis; in
   glatten Regionen (`M_s ≈ 0`) folgt es dem uniformen Kontrollmittel.

`D(p)` wird akzeptiert, wenn er alle Validierungsschwellen sowohl gegen `U`
als auch gegen die unveränderliche Baseline `A` besteht:

```
background_rms_regression(D, U) <= max_background_rms_regression
fwhm_regression(D, U)            <= max_fwhm_regression
seam_score_regression(D, U)      <= max_seam_score_regression
tail11_abs_regression(D, U)      <= max_tail11_abs_regression
elongation_regression(D, U)      <= max_elongation_regression
gate(D, A)
```

und zumindest FWHM oder Seam-Score gegenüber `U` verbessert.

Schlägt `D(p)` fehl, wird eine **Alpha-Blend-Attenuierung** über
`alpha ∈ [0, 1]` gesucht:

```
D_alpha(p) = U(p) + alpha * M_s(p) * (B(p) - U(p))
```

Das größte `alpha`, das `gate(D_alpha, U)` und `gate(D_alpha, A)` besteht und
FWHM oder Seam-Score gegenüber `U` verbessert, wird ausgewählt. Gibt es kein
`alpha > 0`, das beide Vergleiche besteht, bleibt die bereits validierte Basis
`B(p)` erhalten.

### 6.5 Finale Rückfalloption auf Uniformes Kontrollmittel

Nach der gesamten Nachverarbeitung wird das ausgewählte Ausgabebild `O(p)`
nochmals gegen `U(p)` validiert. Falls gilt:

```
background_rms_regression(O, U) > max_background_rms_regression
```

oder eine andere konfigurierte Schwelle überschritten wird, kann der Runner
auf ein Blend mit dem uniformen Kontrollmittel zurückfallen:

```
O_blend(p) = beta * O(p) + (1 - beta) * U(p)
```

wobei `beta` durch binäre Suche über `[0, 1]` als größtes `beta` bestimmt
wird, das `gate(O_blend, U)` und `gate(O_blend, A)` erfüllt. Die Operation wird in
`aqmh_reconstruction.json` als `fallback_to_uniform_control` und
`uniform_control_blend_alpha` protokolliert. Lässt sich kein `beta > 0`
finden, das beide Referenzen erfüllt, wird `A(p)` ausgegeben. Ein vollständiger
Rückfall auf `U(p)` ist nicht zulässig, wenn `U` die AQMH-Baseline regressiert.
Unmittelbar vor dem Schreiben des Ergebnisses erzwingt ein finaler Vergleich
`gate(O_final, A)` diese Invariante unabhängig vom zuvor gewählten Pfad.

### 6.6 Reporting-Anforderungen

`aqmh_reconstruction.json` muss enthalten:

- `low_frequency_neutralization_applied` (`true`/`false`)
- `low_frequency_neutralization_evaluated` (`true`/`false`)
- `low_frequency_neutralization` (Vergleichsmetriken plus `sigma_px`)
- `structure_masked_detail_applied` (`true`/`false`)
- `structure_masked_detail_alpha` (finale Blend-Alpha, falls Attenuierung
  verwendet wurde)
- `structure_masked_detail_validation`
- `fallback_to_uniform_control` (`true`/`false`)
- `uniform_control_blend_accepted` (`true`/`false`)
- `uniform_control_blend_alpha`
- `raw_aqmh_validation`
- `final_vs_raw_aqmh_validation`
- `raw_aqmh_preserved_by_guard` (`true`/`false`)
- `selected_candidate`

---

## 7. Validierung

### 7.1 Regressionsvalidierung gegen das Uniforme Kontrollmittel

Identisch zu v0.2.0 §9. Das rekonstruierte Ergebnis wird mit dem uniformen
Kontrollmittel verglichen. Die Regressionsschwellen bleiben:

- `max_seam_score_regression = 0.05`
- `max_fwhm_regression = 0.02`
- `max_background_rms_regression = 0.05`
- `max_tail11_abs_regression = 0.10`
- `max_elongation_regression = 0.08`

### 7.2 Zero-Veto-Test

Identisch zu v0.2.0. Sind alle Gewichte null, muss die Ausgabe null sein, kein
ungewichteter Mittelwert.

### 7.3 Keine Strukturinjektion

Identisch zu v0.2.0. Die AQMH-Ausgabe darf keine Struktur einführen, die das
uniforme Kontrollmittel um mehr als die konfigurierten Regressionsschwellen
übersteigt.

---

## 8. Diagnostik

Identisch zu v0.2.0 §6 und §7, ergänzt um die in §6.6 genannten Felder.

Per-Frame-Diagnostiken in `aqmh_metrics.json` umfassen:

- `map_mean`, `map_p10`, `map_p90`
- `artifact_frac`
- `sharpness_p50`, `snr_p50`
- `n_regions`
- `global_quality`
- `global_sharpness_input`, `global_snr_input`
- `global_background_penalty_input` und
  `global_background_penalty_source: "sky_gradient"` (v0.2.1)
- `global_quality_input_invalid`

---

## 9. Konfigurationszusammenfassung

### 9.1 Daten und Pipeline

- `data.color_mode`: `OSC` für One-Shot-Color-Daten.
- `data.bayer_pattern`: Standard `auto`. Die FITS-Header-Werte `BAYERPAT` und
  `COLORTYP` haben Vorrang; ein Config-Wert wird nur als Fallback verwendet,
  wenn der Header keine Bayer-Metadaten enthält. Diese Änderung wurde
  eingeführt, damit Beispiel-Config-Defaults Kamera-Metadaten nicht
  überschreiben.

### 9.2 AQMH-Kernparameter (unverändert gegenüber v0.2.0)

- `aqmh.pyramid.scales = 4`
- `aqmh.pyramid.base_window_px = 4`
- `aqmh.pyramid.w_sharp = 0.6`
- `aqmh.pyramid.w_snr = 0.4`
- `aqmh.pyramid.k_artifact = 3.0`
- `aqmh.pyramid.frac_artifact_max = 0.25`
- `aqmh.storage.resolution_divisor = 2`
- `aqmh.storage.dtype = "uint16"`

### 9.3 Globale Qualität (v0.2.1)

- `aqmh.global_quality.g_floor = 0.03`
- `aqmh.global_quality.g_w_sharp = 0.55`
- `aqmh.global_quality.g_w_snr = 0.30`
- `aqmh.global_quality.g_w_background_penalty = 0.25`
- `aqmh.global_quality.g_k_scale = 1.5`

Mit `g_w_background_penalty = 0.0` wird die Hintergrundstrafe deaktiviert; die
begrenzte v0.2.1-Sigmoid-Abbildung bleibt aktiv.

### 9.4 Registrierungs-Gewichtsschutz (v0.2.1)

- `aqmh.reconstruction.registration_weight_guard = true`
- `aqmh.reconstruction.registration_weight_floor = 0.30`
- `aqmh.reconstruction.registration_cc_floor = 0.35`
- `aqmh.reconstruction.registration_cc_full = 0.80`
- `aqmh.reconstruction.registration_sequential_factor = 0.92`
- `aqmh.reconstruction.registration_predicted_factor = 0.50`
- `aqmh.reconstruction.registration_chain_depth_penalty = 0.03`
- `aqmh.reconstruction.registration_chain_depth_max_penalty = 0.15`

### 9.5 Neutralisierung und Blending (v0.2.1)

- Neutralisierungs-Blur-Sigma: `96 px` (fest verdrahtet).
- Strukturmaske-Quantile: `low_q = 0.40`, `high_q = 0.90`.
- Strukturmaske-Blur-Sigma: `4 px`.
- Sigma-Clipping: `low = 2.0`, `high = 1.5`, `4` Iterationen.
- Die Validierungsschwellen stammen aus `aqmh.validation`.

### 9.6 Cherry-Pick (unverändert)

- `aqmh.cherry_pick.enabled = false` standardmäßig.
- `k_frac = 0.30`
- `k_min_required = 20`
- `margin_min = 0.02`

---

## 10. Konformitätsaussage

Ein Run ist **v0.2.1-konform**, genau dann wenn:

1. Er alle verbindlichen Invarianten von v0.2.0 (§1.3) einhält.
2. Er `Q_map` gemäß §2 und §3 berechnet und den verwendeten Speichermodus
   berichtet; Cherry-Pick verwendet zwingend Full-Resolution-Float32.
3. Er die globale Qualität mit der erweiterten Formel aus §1.5 berechnet,
   wenn `g_w_background_penalty > 0`, oder mit der v0.2.0-Formel, wenn der Wert
   `0` ist.
4. Er den Registrierungs-Gewichtsschutz wie in §4.3 beschrieben anwendet,
   wenn er aktiviert ist, und das Ergebnis protokolliert.
5. Er die Nachverarbeitungs-Validierung und die optionale Neutralisierung
   gemäß §6 durchführt und den ausgewählten Kandidaten sowie eventuelle
   Rückfälle aufzeichnet.
6. Er `aqmh_metrics.json`, `aqmh_reconstruction.json` und alle erforderlichen
   Diagnosefelder erzeugt.

Ein Run ist **strikt v0.2.0-konform**, wenn `g_w_background_penalty = 0`,
`registration_weight_guard = false` und alle Nachverarbeitungs-
erweiterungen deaktiviert sind. Ein solcher Run ist eine gültige Teilmenge von
v0.2.1.

---

## 11. Referenzen

- `aqmh_methodik_en_v0.2.0.md` — Basismethodik.
- `tile_compile_cpp/include/tile_compile/config/configuration.hpp` —
  Konfigurationsstrukturen.
- `tile_compile_cpp/src/metrics/aqmh_global_quality.cpp` — Berechnung der
  globalen Qualität.
- `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp` —
  Registrierungs-Gewichtsschutz, Neutralisierung und Blending.
- `tile_compile_cpp/src/metrics/aqmh_quality_map.cpp` — Quality-Map-
  Berechnung.
- `tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp` — Gewichtete
  Rekonstruktion und Cherry-Pick-Logik.
