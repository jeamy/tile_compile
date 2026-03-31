# Tile-basierte Qualitätsrekonstruktion für DSO - Methodik v3.3.9

**Status:** Normative Referenzspezifikation  
**Version:** v3.3.9 (2026-03-27)  
**Zuletzt überarbeitet:** 2026-03-27  
**Gilt für:** `tile_compile.yaml`

---

## 0. Ziel von v3.3.9

Kernziele:

1. mathematische Konsistenz (Notation, Formeln, Randfälle)
2. photometrisch saubere Trennung zwischen additivem Hintergrund und multiplikativer photometrischer Skalierung
3. ein verpflichtender Rekonstruktionskern, der in Pixelwerten linear bleibt und **keine** tileweise nichtlineare Renormalisierung vor OLA verwendet
4. weiches lokales Qualitätsblending statt harter STAR/STRUCTURE-Umschaltung
5. massenerhaltende Cluster-Aggregation im Full Mode
6. support-bewusste Overlap-Add-Semantik einschließlich deterministischer Randabdeckung

---

## 1. Prinzipien und Definitionen

### 1.1 Physikalisches Ziel

Aus vollständig registrierten, linearen Kurzbelichtungs-Frames wird ein räumlich und zeitlich optimal gewichtetes Signal rekonstruiert.

Die Methode modelliert zwei orthogonale Qualitätsachsen:

- **global** (Atmosphäre): Transparenz, Himmelshelligkeit, Rauschen
- **lokal** (Tile): Schärfe, Strukturtragfähigkeit, lokales Hintergrundniveau

### 1.2 Keine Frame-Selektion (Invariante)

**Verboten:** Entfernen ganzer Frames auf Basis von Qualitätskriterien.  
**Erlaubt:** pixelweise Outlier-Verwerfung (Sigma-Clipping), sofern

- sie nur auf Pixelebene wirkt,
- deterministische Parameter verwendet,
- und einen dokumentierten Fallback auf das ungeclippte gültige gewichtete Mittel einschließt.

### 1.3 Linearitätssemantik (präzisiert)

"Strikt linear" (wie in v3.3.6 festgelegt, in v3.3.9 unverändert) bedeutet:

1. **Die photometrische Signalabbildung** bleibt linear (keine globalen nichtlinearen Tonkurven wie Stretch, Asinh, Log).
2. Lineare Rekonstruktionsschritte (Skalierung, gewichtetes Mittel, Overlap-Add) sind verpflichtend.
3. Robuste/statistische Nichtlinearitäten (MAD, Clipping, Sigma-Clipping, adaptive Gate-Entscheidungen) sind als **Hilfsschritte** erlaubt.
4. datenabhängige tileweise Renormalisierung rekonstruierter Pixelwerte (zum Beispiel Median/MAD-Angleichung vor Overlap-Add) ist **nicht Teil des verpflichtenden linearen Kerns**.

---

## 2. Annahmen und Betriebsmodi

### 2.1 Harte Annahmen (Verletzung -> Abbruch)

- Eingangsdaten sind linear (kein Stretch, keine Tonkurven)
- Keine qualitätsbasierte Frame-Selektion
- Registrierte Geometrie wird im selben Pixelreferenzsystem ausgedrückt
- Ab Phase 3 bleibt die Kanal-Semantik im gemeinsamen Core deterministisch und kanal-konsistent

### 2.2 Weiche Annahmen und Runtime-Defaults

| Annahme | Optimal | Minimum | Aktion bei Verletzung |
|---|---:|---:|---|
| Anzahl nutzbarer Frames `N` | >= 800 | `>= assumptions.frames_min` | Reduced Mode für `frames_min .. frames_reduced_threshold-1`; darunter Abort oder Emergency Mode |
| Belichtungszeit-Konsistenz | konstant innerhalb einer Serie | datensatzabhängig | gemischte Serien benötigen explizites Handling/Kalibrierung; keine garantierte Core-Invariante |
| Registrierungsresiduum | < 0.3 px | < 1.0 px | Warnung bei > 0.5 px |
| Sternelongation | < 0.2 | < 0.4 | Warnung bei > 0.3 |

### 2.3 Runtime-konfigurierte Betriebsmodi (verbindlich)

Seien

- `N_min = assumptions.frames_min`
- `N_red = assumptions.frames_reduced_threshold`, mit `N_red >= N_min`

Dann verwendet die aktive Runtime:

- **Full Mode:** `N >= N_red`
- **Reduced Mode:** `N_min <= N < N_red`
- **Unter Minimum:** `N < N_min`

Konsequenzen im Reduced Mode:

- `STATE_CLUSTERING` und `SYNTHETIC_FRAMES` werden übersprungen, wenn `assumptions.reduced_mode_skip_clustering = true`
- wenn Reduced-Mode-Clustering aktiviert bleibt, begrenzt `assumptions.reduced_mode_cluster_range = [K_min, K_max]` die Clustersuche
- wenn Clustering/Synthese übersprungen werden, verwendet das finale Ergebnis die Rekonstruktion direkt weiter

### 2.4 Unterhalb des Minimums

- **N < assumptions.frames_min:** kein regulärer Reduced Mode
- Standardaktion: kontrollierter Abbruch mit Diagnostik
- Optional nur über explizites `runtime_limits.allow_emergency_mode: true`: Emergency Mode mit Warnstatus

### 2.5 Shared-Core-Varianten der Kanal-Semantik (verbindlich)

Frühere Methodikfassungen verwendeten die Bezeichnungen `strict` und `practical`. In v3.3.9 werden sie als **Implementierungsvarianten** behandelt, nicht als aktive Betriebsprofil-Schlüssel in `tile_compile.yaml`.

Zwei konforme Shared-Core-Varianten sind erlaubt:

- **Explizite Per-Channel-Variante**
  - die Kanaltrennung ist bis Phase 2 vollständig abgeschlossen
  - Phasen 3-10 laufen kanalweise
- **CFA-Proxy-äquivalente Variante**
  - der gemeinsame Core darf auf CFA-Proxy-Daten arbeiten, bevor explizites RGB gebildet wird

Für die CFA-Proxy-äquivalente Variante bleiben alle folgenden Punkte zwingend:

1. lineares und deterministisches Rekonstruktionsverhalten im gemeinsamen Core,
2. kanaläquivalente Gewichtungs-/Schätzsemantik (keine versteckte kanalübergreifende Kopplung im Core-Schätzer),
3. Erhalt der CFA-Phase bei geometrischen Operationen,
4. explizite RGB-Domäne vor Farbkalibrierungs-Erweiterungen (BGE/PCC), bei unveränderter Canvas-Masken-Exklusionspolitik.

---

## 3. Pipeline-Überblick (normativ)

1. Registrierung und geometrische Harmonisierung
2. Kanaltrennung (explizit oder verzögert über eine CFA-Proxy-äquivalente Core-Variante)
3. Globale lineare Normalisierung
4. Globale Frame-Metriken und globale Gewichte
5. Tile-Geometrie
6. Lokale Tile-Metriken und lokale Gewichte
7. Tile-Rekonstruktion (Overlap-Add)
8. Zustandsbasiertes Clustering (nur Full Mode)
9. Synthetic Frames (nur Full Mode)
10. Finales lineares Stacking
11. Post-Processing (optional, nicht Teil des Qualitätskerns)

Verbindlicher Kern: 1-10.  
Optional/feature-gated: lokale Denoiser, deterministische Defektpixel-Unterdrückung / Cosmetic Correction, alternative Post-Stack-Clipping-Strategien, WCS/PCC, isolierte Post-PCC-Chroma-Speckle-Unterdrückung.

---

## 4. Registrierung und Kanaltrennung bis Phase 2 (normativ)

Bis einschließlich Phase 2 gilt der CFA-basierte Registrierungs- und Kanaltrennungspfad.
Ab Phase 3 gilt eine der verbindlichen Shared-Core-Varianten aus §2.5:

- expliziter per-channel Core,
- CFA-Proxy-äquivalenter Core.

### 4.1 CFA-basierter Registrierungspfad

- Registrierung auf einer CFA-Luminanz-Proxy-Darstellung
- CFA-bewusstes Warping über Subplanes (`warp_cfa_mosaic_via_subplanes`)
- anschließende Kanaltrennung (explizite per-channel Variante) oder verzögerte Trennung in der Channel-Stack-Stufe (CFA-Proxy-äquivalente Variante)

#### 4.1.1 Optionale Pre-Warp-CFA-Defektpixel-Unterdrückung (feature-gated)

Vor dem CFA-bewussten Warp dürfen Implementierungen einen deterministischen per-frame CFA-Cosmetic-Correction-Schritt anwenden, um isolierte Defektpixel auf dem Rohmosaik zu unterdrücken.

Verbindliche Bedingungen für diesen optionalen Schritt:

1. CFA-Phase und Sample-Geometrie müssen unverändert bleiben; Demosaicing oder kanalübergreifendes Resampling vor dem Warp sind nicht zulässig.
2. Korrekturen müssen lokal und spärlich bleiben und auf isolierte same-parity Ausreißer (zum Beispiel Hot-/Cold-/chromatisch instabile Sensel) oder eine äquivalente deterministische Defektpixel-Signatur beschränkt sein.
3. Ersatzwerte müssen ausschließlich aus endlichen same-parity CFA-Nachbarn oder aus einem äquivalenten CFA-phasenerhaltenden lokalen Schätzer abgeleitet werden.
4. Kompakte reale Bildstruktur muss durch einen deterministischen Strukturschutz erhalten bleiben; Implementierungen dürfen gestützten Mehrpixel-Bildinhalt nicht unter dem Vorwand der Defektunterdrückung löschen.
5. Dieser Schritt ist optional und gehört nicht zum verpflichtenden linearen Qualitätskern der Phasen 3-10.

### 4.2 Registrierungskaskade

Pro Frame:

1. konfigurierbare Primärmethode (`triangle_star_matching` als Default)
2. feste Fallback-Reihenfolge:
   - `trail_endpoint_registration`
   - `feature_registration_similarity` (AKAZE)
   - `robust_phase_ecc`
   - `hybrid_phase_ecc`
   - Identity-Fallback mit Warnung

Akzeptanzkriterium pro Versuch:

- `NCC(warped, ref) > NCC(identity, ref) + delta_ncc`
- Default `delta_ncc = 0.01`

**Randfälle (verbindlich):**

1. **Referenzframe:** `NCC(identity, ref) = 1.0` für den Referenzframe selbst; dadurch ist das Kriterium unerfüllbar. Implementierungen müssen die Identity-Transformation für den Referenzframe bedingungslos akzeptieren, ohne Kaskadenversuche.
2. **Nahezu perfekte Ausrichtung:** Wenn `NCC(identity, ref) >= 1 - delta_ncc`, kann keine Registrierungs-Methode das Kriterium erfüllen. Implementierungen müssen direkt die Identity-Transformation akzeptieren und eine Diagnose loggen.
3. In beiden Fällen ist der Identity-Fallback das korrekte Ergebnis und darf nicht als Kaskadenfehler gezählt werden.

### 4.3 CFA-Proxy-Core-Pfad (verbindlich)

- Globale/lokale Metriken und Tile-Rekonstruktion dürfen auf CFA-Proxy-Eingängen statt auf frühen expliziten RGB-Ebenen arbeiten.
- Das ist nur dann konform, wenn die Kanal-Semantik- und Linearitätsbedingungen aus §2.5 erhalten bleiben.
- Explizite RGB-Daten sind weiterhin vor BGE/PCC und für finale RGB-Ausgaben erforderlich.

---

## 5. Gemeinsamer Core ab Phase 3

### 5.1 Notation (verbindlich)

- `f` Frame-Index, `t` Tile-Index, `c` Kanalindex, `p` Pixel
- `I_{f,c}^{raw}(p)` rohes lineares Eingangsbild pro Frame/Kanal
- `I_{f,c}(p)` normalisiertes Eingangsbild pro Frame/Kanal
- `B_{f,c}` globaler additiver Hintergrund (vor der Normalisierung)
- `P_{f,c}` globale photometrische Skalierung (vor der Normalisierung)
- `sigma_{f,c}` globales Rauschen (nach der Normalisierung)
- `E_{f,c}` globale Gradientenenergie (nach der Normalisierung)
- `Q_{f,c}` globaler Qualitätsindex
- `G_{f,c}` globales Gewicht
- `eta_t` Star-Support-Blendfaktor des Tiles
- `U_{f,t,c}` lokale Metrik-Konfidenz
- `Q_{f,t,c}^{local}` lokaler Qualitätsindex
- `L_{f,t,c}` lokales Gewicht
- `W_{f,t,c}` effektives Gewicht
- `M_{k,c}` Cluster-Masse im Full Mode
- `M_{t,c}(p)` per-Kanal-Supportmaske für OLA (1 = gültiges Canvas-Pixel mit mindestens einem gültigen Frame, 0 = Canvas-ungültig oder keine gültigen Frames)
- `k_local` lokaler Gewichtsskalenfaktor (§5.5.6)
- `k_global` globaler Gewichtsskalenfaktor (§5.3.2)

**Ab hier wird der Kanalindex `c` konsistent verwendet.**

---

## 5.2 Globale lineare Normalisierung (verpflichtend)

Reihenfolge:

1. Additiver Hintergrund aus Rohdaten:
   - `B_{f,c} = median(I_{f,c}^{raw})`
2. Hintergrundsubtraktion:
   - `J_{f,c}(p) = I_{f,c}^{raw}(p) - B_{f,c}`
3. Photometrische Skala aus deterministischer Durchsatz-/Flux-Referenz:
   - `P_{f,c} = photometric_scale(J_{f,c})`

   **Definition von `photometric_scale` (verbindlich):** Implementierungen müssen eine der folgenden deterministischen Methoden in Prioritätsreihenfolge verwenden:

   a. **Ensemble-Sternfluss-Skalierung (empfohlen):** `P_{f,c} = median_s(flux_{f,c,s}) / median_s(flux_{ref,c,s})`, wobei `flux_{f,c,s}` der instrumentelle Aperturfluss des nicht gesättigten Referenzsterns `s` in Frame `f`, Kanal `c` ist. Die Sternmenge muss fest (aus dem Referenzframe bestimmt) und deterministisch sein.

   b. **Belichtungszeitverhältnis:** Wenn alle Frames identischen optischen Durchsatz haben und sich nur die Belichtungszeiten unterscheiden: `P_{f,c} = t_f / t_ref`, wobei `t_ref` die Belichtungszeit des Referenzframes ist.

   c. **Deterministischer Fallback (`P_{f,c} = 1`):** Wenn weder Sternflüsse noch Belichtungs-Metadaten zuverlässig verfügbar sind und der Durchsatz bereits konsistent ist (siehe Constraint 5 unten).

   Constraint: `P_{f,c}` darf nicht ausschließlich aus dem Himmels-Hintergrundniveau `B_{f,c}` abgeleitet werden.
4. Lineare Skalierung:
   - `I_{f,c}(p) = J_{f,c}(p) / max(P_{f,c}, eps_scale)`
5. Metriken auf normalisierten Daten:
   - `sigma_{f,c}`, `E_{f,c}`

Verbindliche Constraints:

1. additive Hintergrundsubtraktion und multiplikative photometrische Skalierung müssen getrennte Operationen bleiben,
2. `P_{f,c}` darf nicht ausschließlich über das Himmels-Hintergrundniveau definiert werden,
3. gültige negative oder Null-Pixelwerte nach Subtraktion/Skalierung bleiben zulässige Samples,
4. globale nichtlineare Tonkurven bleiben verboten,
5. wenn keine verlässliche photometrische Skala geschätzt werden kann und der Belichtungsdurchsatz bereits konsistent ist, dürfen Implementierungen deterministisch `P_{f,c} = 1` verwenden.

Empfohlene Defaults:

- `eps_bg = 1e-6`
- `eps_scale = 1e-6`

---

## 5.3 Globale Metriken und Gewichte

### 5.3.1 Robuste Metrik-Normalisierung

Für eine Metrikfolge `x`:

`z(x_i) = (x_i - median(x)) / max(1.4826 * MAD(x), eps_mad)`

mit `eps_mad = 1e-6`.

### 5.3.2 Globaler Qualitätsindex

`Q_{f,c} = alpha*(-z(B_{f,c})) + beta*(-z(sigma_{f,c})) + gamma*z(E_{f,c})`

Constraint: `alpha + beta + gamma = 1`

Defaults:

- `alpha=0.4, beta=0.3, gamma=0.3`

**Notationshinweis:** Die Symbole `alpha`, `beta`, `gamma` beziehen sich hier ausschließlich auf die Gewichte des globalen Qualitätsindex. Das BGE-Autotuning-Ziel (§6.3.7.1) verwendet die getrennten Symbole `alpha_f` (Flatness-Gewicht) und `beta_r` (Roughness-Gewicht), um Kollisionen zu vermeiden.

**Hinweis zur gemischten Normalisierung:** `B_{f,c}` ist eine Größe vor der Normalisierung, während `sigma_{f,c}` und `E_{f,c}` nach der Normalisierung definiert sind (§5.2 Schritt 5). Die z-Score-Operation macht sie skalenunabhängig, daher ist die Kombination statistisch zulässig. In Datensätzen mit gemischten Belichtungszeiten kann `z(B_{f,c})` eher die Belichtungsdauer als die Himmelshelligkeit abbilden; in solchen Fällen sollte das `alpha`-Gewicht reduziert oder der `B`-z-Score auf belichtungsnormalisierten Daten neu berechnet werden.

Clamping vor der Exponentialfunktion:

`Q_{f,c}^{clamped} = clip(Q_{f,c}, -3, +3)`

Globales Gewicht:

`G_{f,c} = exp(k_global * Q_{f,c}^{clamped})`

mit `k_global > 0`, Default `k_global=1.0`.

### 5.3.3 Optionale adaptive Gewichtung

Der verpflichtende Kern verwendet die statischen Gewichte aus §5.3.2.

Wenn `global_metrics.adaptive_weights=true`, bleiben adaptive Gewichte eine optionale Erweiterung und müssen alle folgenden Bedingungen erfüllen:

1. sie werden aus einem deterministischen predictive-utility-Kriterium abgeleitet, nicht bloß aus `Var(z(.))` bereits normalisierter Metriken,
2. sie werden auf `[0.1, 0.7]` geclippt und auf Summe 1 renormalisiert,
3. sie fallen bei degenerierten oder uninformative utility-Schätzungen auf die statischen Defaults zurück,
4. utility-Ziel und Tie-Break-Regeln müssen dokumentiert werden.

---

## 5.4 Tile-Geometrie

Parameter:

- Bildgröße `W,H`
- robuste Seeing-Schätzung `F` (FWHM in Pixeln)
- `s = tile.size_factor`
- `T_min = tile.min_size`
- `D = tile.max_divisor`
- `o = tile.overlap_fraction`, `0 <= o <= 0.5`

Formeln:

`T0 = s * F`

`o_clipped = clip(o, 0, 0.5)`

`T_hi = floor(min(W,H)/D)`

Größenauswahl (verbindlich):

1. wenn `F <= 0` -> setze `F = 3.0`
2. fordere `T_min >= 16`
3. fordere `D >= 1`
4. wenn `T_hi < T_min`, tritt deterministischer Compact-Tile-Mode ein:
   - `T = min(W,H)`
   - `O = 0`
   - `S = T`
5. andernfalls:
   - `T = floor(clip(T0, T_min, T_hi))`
   - `O = floor(o_clipped * T)`
   - `S = T - O`

Zusätzliche Guards (verbindlich):

1. wenn `S <= 0` -> setze `o_clipped = 0.25`, berechne `O,S` neu
2. wenn `O >= T` -> setze `O = floor(0.25 * T)`, `S = T - O`
3. wenn `T <= 0` -> Abbruch mit Diagnostik

**Hinweis:** Guard 1 und 2 sind unter den genannten Vorbedingungen invariant (`o_clipped ∈ [0, 0.5]` und `T ≥ 1`, garantiert durch Guard 3), daher lösen sie im Normalbetrieb nie aus. Sie bleiben als defensive Schutzmechanismen gegen Implementierungsabweichungen erhalten.

---

## 5.5 Lokale Tile-Metriken

**Canvas-Exklusion (verbindlich für ganz §5.5):** Alle Berechnungen lokaler Tile-Metriken — FWHM, Rundheit, Kontrast, `E/sigma`, Hintergrund `B_{f,t,c}` — müssen ausschließlich auf Pixeln erfolgen, die für diesen Frame sowohl endlich als auch innerhalb des gültigen Canvas-Supports liegen. Canvas-ungültige Pixel müssen aus allen Metrik-Akkumulatoren ausgeschlossen werden. Wenn die Anzahl canvas-gültiger Pixel in einem Tile unter ein Minimum fällt (empfohlen: 16 px), werden die Metriken dieses Tiles für diesen Frame als nicht verfügbar markiert, und der Fallback aus §5.5.1 greift.

### 5.5.1 Weicher lokaler Support-Blend (verbindlich in v3.3.9)

Der verpflichtende Kern verwendet **keine** harte STAR/STRUCTURE-Umschaltung.

Stattdessen erhält jedes Tile einen deterministischen Star-Support-Blendfaktor:

`eta_t = clip(star_count_t / max(tile.star_soft_count, 1), 0, 1)`

mit normativem Default:

- `tile.star_soft_count = tile.star_min_count`

Semantik:

- `eta_t = 1`: vollständig stern-gestütztes Tile
- `eta_t = 0`: rein strukturgetriebenes Tile
- Zwischenwerte blenden beide lokalen Modelle kontinuierlich

Wenn Sternmetriken für einen gegebenen Frame/Tile/Kanal-Term nicht verfügbar oder nicht endlich sind, wird der effektive Sternbeitrag dieses Terms als nicht verfügbar behandelt und das Blending fällt auf die Strukturkomponente zurück.

### 5.5.2 STAR-Tile-Metriken

- `FWHM_{f,t,c}`
- `R_{f,t,c}` (Rundheit)
- `C_{f,t,c}` (Kontrast)

Lokaler Index:

`Q_{f,t,c}^{star} = 0.6*(-z_tilde(FWHM)) + 0.2*z_tilde(R) + 0.2*z_tilde(C)`

### 5.5.3 STRUCTURE-Tile-Metriken

- `(E/sigma)_{f,t,c}`
- `B_{f,t,c}`

Lokaler Index:

`Q_{f,t,c}^{struct} = 0.7*z_tilde(E/sigma) - 0.3*z_tilde(B)`

### 5.5.4 Nachbarschaftsbewusste Metrik-Normalisierung (verbindlich in v3.3.9)

Für jede skalare Tile-Metrikfamilie `m` und jedes Tile `t` wird zunächst über die nutzbaren Frames der tile-lokale robuste z-Score berechnet:

`z_local(m_{f,t,c}) = robust_z(m_{f,t,c}; {m_{f',t,c}}_{f' in F_t})`

Wenn die Nachbarschaftsnormierung aktiviert ist, wird zusätzlich eine gepoolte robuste Lage/Skala über Nachbar-Tiles und alle nutzbaren Frames berechnet:

- `N_r(t)`: Tile-Nachbarschaft mit Manhattan-Radius `r = local_metrics.neighborhood_normalization.radius`
- `P_t(m) = { m_{f',u,c} | u in N_r(t), f' nutzbar }`

`z_pool(m_{f,t,c}) = (m_{f,t,c} - median(P_t(m))) / max(1.4826*MAD(P_t(m)), eps_mad)`

Der tatsächlich vom lokalen Qualitätsmodell verwendete Metrik-z-Score ist dann

`z_tilde(m_{f,t,c}) = (1 - b_local) * z_local(m_{f,t,c}) + b_local * z_pool(m_{f,t,c})`

mit `b_local = local_metrics.neighborhood_normalization.blend`.

Wenn die Nachbarschaftsnormierung deaktiviert ist oder `P_t(m)` leer ist, gilt:

`z_tilde(m_{f,t,c}) = z_local(m_{f,t,c})`

Normative Default-Parameter:

- `local_metrics.neighborhood_normalization.enabled = true`
- `local_metrics.neighborhood_normalization.radius = 1`
- `local_metrics.neighborhood_normalization.blend = 0.5`

Verbindliche Randbedingungen:

1. die Normalisierung ist metriklokal und frame-reihenfolgenunabhängig,
2. nur endliche Metrik-Samples dürfen in die gepoolten Nachbarschaftsstatistiken eingehen,
3. die Nachbarschaftsnormierung verändert Scores nur über die Metriknormalisierung, niemals durch direkte Pixelmanipulation.

### 5.5.5 Räumliche Regularisierung lokaler Scores (verbindlich in v3.3.9)

Zuerst wird der unregularisierte lokale Score berechnet:

`Q_{f,t,c}^{blend} = eta_t * Q_{f,t,c}^{star} + (1 - eta_t) * Q_{f,t,c}^{struct}`

Wenn eine Komponente nicht verfügbar ist, verwende direkt die jeweils andere verfügbare Komponente.

`Q_{f,t,c}^{raw} = clip(Q_{f,t,c}^{blend}, q_min, q_max)`

mit lokalem Default-Clip-Bereich `[q_min, q_max] = [-3, +3]`.

Um zu verhindern, dass benachbarte Tiles in inkompatible lokale Regime kippen, während verlässliche lokale Unterschiede erhalten bleiben, wird das lokale Score-Feld vor der Exponential-Gewichtung auf dem Tile-Nachbarschaftsgraphen regularisiert.

Sei `N(t)` die 4-Nachbarschaft des Tiles `t` im Tile-Raster.

Definiere einen lokalen Konfidenz-Term

`U_{f,t,c} = clip(n_valid_metrics_{f,t,c} / max(n_expected_metrics_{f,t,c}, 1), 0, 1)`

und eine edge-aware Nachbar-Affinität

`A_{t,u}^{(k)} = exp(-|Q_{f,t,c}^{(k)} - Q_{f,u,c}^{(k)}| / tau_local)`

Für jeden Frame `f`, jedes Tile `t` und jeden Pass `k`:

`lambda_{f,t,c}^{eff} = lambda_local * (1 - U_{f,t,c})`

Sei `S_A^{(k)} = sum_{u in N(t)} A_{t,u}^{(k)}`.

**Affinity-Zero-Guard (verbindlich):** Wenn `S_A^{(k)} < eps_affinity`, setze `Q_{f,t,c}^{(k+1)} = Q_{f,t,c}^{(k)}` (keine Regularisierung für dieses Tile / diesen Pass). Andernfalls:

`Q_{f,t,c}^{(k+1)} = (1 - lambda_{f,t,c}^{eff}) * Q_{f,t,c}^{(k)} + lambda_{f,t,c}^{eff} * sum_{u in N(t)} A_{t,u}^{(k)} Q_{f,u,c}^{(k)} / S_A^{(k)}`

Default `eps_affinity = 1e-6`.

mit Initialisierung:

`Q_{f,t,c}^{(0)} = Q_{f,t,c}^{raw}`

und finalem regularisiertem Score nach `P` Pässen:

`Q_{f,t,c}^{reg} = Q_{f,t,c}^{(P)}`

Normative Default-Parameter:

- `local_metrics.spatial_regularization.enabled = true`
- `local_metrics.spatial_regularization.lambda = 0.35`
- `local_metrics.spatial_regularization.passes = 1`
- `local_metrics.spatial_regularization.tau_local = 1.0`
- `local_metrics.spatial_regularization.eps_affinity = 1e-6`

Verbindliche Randbedingungen:

1. nur gültige/common Tiles dürfen teilnehmen,
2. die Regularisierung ist frame-lokal und darf keine verschiedenen Frames koppeln,
3. Tiles ohne gültige Nachbarn oder mit `lambda_{f,t,c}^{eff} <= 0` behalten `Q_{f,t,c}^{reg} = Q_{f,t,c}^{raw}`,
4. die Regularisierung wirkt nur auf lokale Qualitätsscores, niemals direkt auf Pixelwerte.

### 5.5.6 Lokales Gewicht

`Q_{f,t,c}^{local} = clip(Q_{f,t,c}^{reg}, q_min, q_max)`

`L_{f,t,c} = exp(k_local * Q_{f,t,c}^{local})`

mit `k_local > 0`, Default `k_local = 1.0`.

**Symmetrie zum globalen Gewicht:** Die Skalenfaktoren `k_global` (§5.3.2) und `k_local` steuern unabhängig die Empfindlichkeit globaler und lokaler Exponentialgewichtung. Beide sind standardmäßig `1.0`. Ein größeres `k_local` schärft den lokalen Qualitätskontrast; ein Wert gegen `0` egalisiert die lokalen Gewichte über Tiles hinweg.

---

## 5.6 Effektives Gewicht

`W_{f,t,c} = G_{f,c} * L_{f,t,c}`

Semantik:

- `G`: globale atmosphärische Qualität
- `L`: lokale Struktur-/Schärfequalität

---

## 5.7 Tile-Rekonstruktion (konsolidiert)

Die tileweisen Gewichte seien

`w_{f,t,c} = W_{f,t,c}`.

Wenn

`sum_f max(w_{f,t,c}, 0) <= eps_weight`

tritt das Tile in einen deterministischen Gewichtsfallback ein, und alle positiven endlichen Tile-Gewichte werden durch `1` ersetzt.

Für jedes Pixel `p` im Tile `t` sei die gültige Samplemenge

`V_{t,c}(p) = { f | I_{f,c}(p) endlich ist und innerhalb des gültigen Canvas-Supports liegt }`.

Falls `|V_{t,c}(p)| = 0`, setze

`R_{t,c}(p) = 0`.

Für `|V_{t,c}(p)| <= 2` (oder wenn Clipping-Iterationen deaktiviert sind) wird direkt das gültige gewichtete Mittel verwendet:

`R_{t,c}(p) = sum_{f in V_{t,c}(p)} w_{f,t,c} I_{f,c}(p) / sum_{f in V_{t,c}(p)} w_{f,t,c}`.

Andernfalls wird iteratives gewichtetes Sigma-Clipping auf der aktiven Menge `A^{(0)} = V_{t,c}(p)` angewendet:

`mu^{(k)} = sum_{f in A^{(k)}} w_{f,t,c} I_{f,c}(p) / sum_{f in A^{(k)}} w_{f,t,c}`

mit

- `V1 = sum_{f in A^{(k)}} w_{f,t,c}`
- `V2 = sum_{f in A^{(k)}} w_{f,t,c}^2`
- `N_eff = V1^2 / max(V2, eps_weight)`
- `D_eff = V1 - V2 / max(V1, eps_weight)`

Wenn `N_eff <= 2 + eps_neff` oder `D_eff <= eps_var`, beende das Clipping und verwende das gewichtete Mittel über die aktuelle aktive Menge.

Andernfalls:

`sigma^{(k)} = sqrt( sum_{f in A^{(k)}} w_{f,t,c}(I_{f,c}(p)-mu^{(k)})^2 / D_eff )`

und Update

`A_prop^{(k+1)} = { f in A^{(k)} | mu^{(k)} - sigma_low*sigma^{(k)} <= I_{f,c}(p) <= mu^{(k)} + sigma_high*sigma^{(k)} }`

unter Beachtung des Keep-Floors

`|A_prop^{(k+1)}| >= ceil(min_fraction * |V_{t,c}(p)|)`

mit `min_fraction in (0, 1)`, normativem Default `min_fraction = 0.5`.

Deterministische Keep-Floor-Regel:

- wenn die vorgeschlagene Menge den Keep-Floor erfüllt, akzeptiere sie:
  - `A^{(k+1)} = A_prop^{(k+1)}`
- andernfalls beende das Clipping und behalte die vorherige aktive Menge:
  - `A^{(*)} = A^{(k)}`

Der finale Rekonstruktionswert ist das gewichtete Mittel über die final akzeptierte Menge. Wenn Clipping die akzeptierte Menge numerisch leert, fällt die Implementierung auf das ungeclippte gültige gewichtete Mittel über `V_{t,c}(p)` zurück.

Default `eps_weight = 1e-6`.

Empfohlene Guards:

- `eps_neff = 1e-6`
- `eps_var = 1e-12`
- `min_fraction = 0.5` (konfigurierbar via `tile_reconstruction.sigma_clip.min_fraction`)

### 5.7.1 Support-bewusste Fensterung und Overlap-Add (verbindlich in v3.3.9)

Der verpflichtende Kern wendet **keine** tileweise Median/MAD-Renormalisierung vor Overlap-Add an.

Stattdessen ist der OLA-Eingang das rekonstruierte Tile selbst:

`Y_{t,c}(p) = R_{t,c}(p)`

Sei

- `M_{t,c}(p) = 1`, wenn `|V_{t,c}(p)| > 0` UND `R_{t,c}(p)` endlich ist UND das Pixel `p` innerhalb des gültigen Canvas-Supports liegt, sonst `0`
- `omega_t(p) >= 0` das deterministische Blendfenster des Tiles `t`

**Canvas-Semantik (verbindlich):** `M_{t,c}(p) = 0` in allen folgenden Fällen:
1. `|V_{t,c}(p)| = 0` (kein gültiger Frame trägt zu Pixel `p` im Tile `t`, Kanal `c` bei); die Konvention `R_{t,c}(p) = 0` ist nur ein Fill-Wert und darf NICHT in den OLA-Akkumulator eingehen.
2. Pixel `p` liegt außerhalb des gültigen Canvas-Supports (canvas-maskierte Region), unabhängig davon, ob Frames nominell dort abdecken.
3. `R_{t,c}(p)` ist nicht endlich (NaN oder Inf).

Die Supportmaske trägt einen Kanalindex `c`, weil bei CFA-Proxy-äquivalenter Verarbeitung die Anzahl gültiger Frames pro Pixel je Kanal unterschiedlich sein kann.

**Hinweis zur Fill-Zero-Exklusion:** Das Setzen von `M_{t,c}(p) = 0`, wenn `|V| = 0`, verhindert, dass Fill-Zeros den OLA-Nenner `S` an Abdeckungsgrenzen verdünnen. Ohne dies würde `S` Fenstergewichte von Tiles ohne gültige Daten akkumulieren und `I_rec` in teilweise abgedeckten Regionen gegen Null biasen.

Verbindliche OLA-Semantik:

1. `omega_t` darf nur von Tile-Geometrie, Overlap-Geometrie und Randlage abhängen, niemals von Pixelwerten,
2. `omega_t` muss nicht-negativ sein,
3. für jedes gültige Canvas-Pixel `p` müssen Implementierungen garantieren:
   - `sum_t omega_t(p) * M_{t,c}(p) > 0`,
4. für innere Pixel mit vollem Support müssen die Fenster bis auf numerische Toleranz eine Partition der Eins bilden:
   - `sum_t omega_t(p) * M_{t,c}(p) in [1 - tol_ola, 1 + tol_ola]`,
5. Tiles, die eine äußere Bildgrenze berühren, müssen einseitige randbewusste Fenster oder eine äquivalente support-bewusste Renormalisierung verwenden; symmetrische an beiden Enden auf Null gehende Fenster sind an der äußeren Canvas-Grenze nicht zulässig, sofern Regel 3 und 4 nicht konstruktiv dennoch erfüllt bleiben.

Empfohlene Defaults:

- `tol_ola = 1e-6`
- innere Tiles: separable Kosinus-/Tukey-Partitionsfenster
- Rand-Tiles: einseitige Kosinus-Rampen nur über tatsächliche Overlap-Zonen

Akkumulation:

- Zählerakkumulator: `A`
- Support-Akkumulator: `S`

`A += omega_t * M_{t,c} * Y_{t,c}`

`S += omega_t * M_{t,c}`

`I_rec = A / max(S, eps_weight)`

**Canvas-Garantie:** Nur Pixel mit `M_{t,c}(p) = 1` tragen zu `A` und `S` bei. Canvas-ungültige Pixel tragen zu keinem der beiden Akkumulatoren etwas bei und gehen damit in keine Berechnung ein. Das finale `I_rec` an einem canvas-ungültigen Pixel ist `0 / eps_weight` und muss anschließend in der Ausgabe maskiert werden (nicht als gültiger Rekonstruktionswert interpretieren).

### 5.7.2 Boundary-Diagnostik (empfohlen, nicht invasiv)

Um sichtbare Tile-Grenzen zu diagnostizieren, ohne das Rekonstruktionsergebnis zu verändern, dürfen Implementierungen benachbarte Tiles auf dem tatsächlichen OLA-Eingang `R_{t,c}` und auf dem gefensterten Eingang `omega_t * R_{t,c}` auswerten.

Empfohlene Praxis ist, diese Diagnostik zweimal zu emittieren:

- einmal auf den rohen rekonstruierten Tiles `R_{t,c}`
- einmal auf dem gefensterten OLA-Eingang `omega_t * R_{t,c}`

Für jedes benachbarte Tile-Paar `(a,b)` mit Overlap-Domäne `Omega_ab`, definiere:

`Delta_ab(p) = R_{b,c}(p) - R_{a,c}(p)`, für `p in Omega_ab`

Es dürfen nur Samples innerhalb der gemeinsamen canvas-gültigen Domäne beitragen. Maskierte Canvas-Zonen müssen ausgeschlossen und dürfen nicht als gültige Nullwerte behandelt werden.

Empfohlene Paar-Diagnostik:

- `mean_abs_diff_ab = mean_p |Delta_ab(p)|`
- `p95_abs_diff_ab = p95_p |Delta_ab(p)|`
- `mean_signed_diff_ab = mean_p Delta_ab(p)`
- `n_ab = |Omega_ab|` gültige endliche Overlap-Samples

Zusätzlich dürfen Implementierungen per-Pair-Differenzen zusammenfassen in:

- gültigem Frame-Support,
- Post-Rekonstruktions-Hintergrundmetriken,
- Post-Rekonstruktions-SNR-Proxys,
- Post-Rekonstruktions-Mittelkorrelations-Proxys,
- und Fallback-Mismatch-Flags.

Verbindliche Semantik:

1. Diese Diagnostik muss **read-only** sein und darf `R_{t,c}`, `omega_t`, `A`, `S` oder das finale OLA-Ergebnis nicht verändern.
2. Sie darf als Laufzeit-Artefakt für Analyse und Regressionstests emittiert werden.
3. Weil sie nicht in den Schätzer zurückwirkt, verändert sie **nicht** die Linearitätssemantik des Rekonstruktionskerns.

---

## 5.8 Optionale lokale Denoiser (explizit optional)

Diese Schritte sind **nicht Teil des verpflichtenden mathematischen Kerns**, aber zulässige Erweiterungen.

### 5.8.1 Soft-Threshold-High-Pass

- Hintergrund via Box-Blur
- Residuum
- `tau = alpha_d * sigma_tile`
- Soft-Shrinkage
- Rekonstruktion

### 5.8.2 Wiener im Frequenzbereich

- Reflection Padding
- FFT
- Wiener-Transferfunktion
- IFFT und Crop

Nur anwenden, wenn die Gate-Bedingungen erfüllt sind (SNR/Qualität/Tile-Typ).

---

## 5.9 Zustandsbasiertes Clustering (Full Mode)

Aktiv nur im **Full Mode** (`N >= N_red`, siehe §2.3). Zusätzlich erfordert Clustering mindestens 50 nutzbare Frames, um mindestens `K_min = 5` sinnvolle Cluster zu liefern (`K = clip(floor(N/10), K_min, K_max)` ergibt K ≥ 5 für N ≥ 50). Wenn `N_red < 50`, ist der effektive Schwellenwert `max(N_red, 50)`.

**Entfernt:** Der zuvor genannte feste Schwellenwert `N >= 200` ist durch das konfigurierbare Mode-Framework ersetzt. Implementierungen dürfen `200` nicht mehr als Clustering-Gate hart codieren.

Zustandsvektor pro Frame/Kanal (kanalweise oder kanalaggregiert, konfigurierbar):

`v_f = (G_{f,*}, mean_t(Q_{f,t,*}^{local}), var_t(Q_{f,t,*}^{local}), B_{f,*}, sigma_{f,*})`

Anzahl der Cluster:

`K = clip(floor(N/10), K_min, K_max)`

Defaults: `K_min=5`, `K_max=30`.

---

## 5.10 Synthetic Frames

### 5.10.1 Default (global)

Für Cluster `k`:

`S_{k,c} = sum_{f in k} G_{f,c} * I_{f,c} / sum_{f in k} G_{f,c}`

**Zero-Denominator-Fallback:** Wenn `sum_{f in k} G_{f,c} <= eps_weight` (degenerierter Cluster), verwende das ungewichtete Mittel: `S_{k,c} = sum_{f in k} I_{f,c} / |k|`.

### 5.10.2 Optional (`tile_weighted`)

Wenn `synthetic.weighting=tile_weighted`:

- rekonstruiere pro Tile/Kanal mit `W_{f,t,c}`
- füge via derselben support-bewussten OLA-Semantik aus §5.7.1 zu `S_{k,c}` zusammen
- Implementierungen müssen die Tile-Grenz-Diagnostik aus §5.7/§11 überwachen; wenn tile-weighted synthesis gleichzeitig Boundary-Regression und Cross-Tile-Weight-Disagreement zeigt, müssen sie für die Bildung der Synthetic Frames deterministisch auf §5.10.1 (`global`) zurückfallen, statt nichtlineare per-Tile-Renormalisierung wieder einzuführen
- angeforderter und effektiver Synthetic-Weighting-Modus müssen in der Diagnostik festgehalten werden

### 5.10.3 Semantik von Phase 7 vs. 9

- Full Mode mit `global`: Phase 7 liefert primär lokales Qualitätsmodell/Diagnostik; das finale Produkt entsteht aus Phase 9+10.
- Full Mode mit `tile_weighted`: lokale Tile-Qualität wird explizit in Synthetic Frames propagiert, sofern der Seam-Guard oben nicht auf `global` zurückfällt.
- Reduced Mode: die Ausgabe aus Phase 7 ist das direkte Endprodukt.

---

## 5.11 Finales lineares Stacking

### 5.11.1 Definition von Cluster-Qualität und -Masse (verbindlich in v3.3.9)

Für jeden Cluster `k` und Kanal `c` definiere:

`Q_{k,c} = median_{f in k}(Q_{f,c}^{clamped})`

`M_{k,c} = sum_{f in k} G_{f,c}`

wobei `Q_{f,c}^{clamped}` der globale Frame-Qualitätsindex ist, der bereits auf `[-3,+3]` begrenzt wurde.

Wenn `M_{k,c} <= eps_weight`, ersetze deterministisch durch

`M_{k,c} = |k|`.

### 5.11.2 Massenerhaltende qualitätsgewichtete Cluster-Aggregation (verbindlich in v3.3.9)

Cluster werden mittels exponentieller Qualitätsgewichtung **und** Erhalt der Cluster-Masse aggregiert:

`Q_{k,c}^{rel} = clip(Q_{k,c} - median_j(Q_{j,c}), -3, +3)`

`w_{k,c}^{raw} = M_{k,c} * exp(kappa_cluster * Q_{k,c}^{rel})`

mit:

- `kappa_cluster > 0` (empfohlener Default: `kappa_cluster = 1.0`)
- `Q_{k,c}^{rel}` bereits geclippt auf `[-3,+3]`

Optionaler Stabilitäts-Cap (empfohlen):

`w_{k,c} = min(w_{k,c}^{raw}, r_cap * median_j(w_{j,c}^{raw}))`

mit empfohlenem `r_cap` in `[10, 50]`.

Wenn Weight-Capping deaktiviert ist, verwende:

`w_{k,c} = w_{k,c}^{raw}`

Finales Ergebnis pro Kanal:

`R_c = sum_k (w_{k,c} * S_{k,c}) / sum_k w_{k,c}`

**Zero-Denominator-Fallback:** Wenn `sum_k w_{k,c} <= eps_weight` (degenerierte Gewichtsverteilung), fällt die Implementierung auf das Gleichgewichts-Mittel zurück: `R_c = sum_k S_{k,c} / K`. Dieser Fallback muss als Warnung geloggt werden.

### 5.11.3 Semantik

- Bessere atmosphärische Zustände (höheres `Q_{k,c}`) erhalten stärkeren Einfluss.
- Alle Cluster bleiben enthalten (keine harte Zustandsselektion).
- Cluster-Support bleibt über `M_{k,c}` erhalten.
- Der Schätzer bleibt linear in den Synthetic Frames.
- Dominanz wird über optionales Weight-Capping begrenzt.

---

## 6. Post-Processing (nicht Teil des verpflichtenden Kerns)

### 6.1 RGB/LRGB-Kombination

Austauschbar, außerhalb des Rekonstruktionskerns.

### 6.2 Astrometrie (WCS)

Zulässiger Downstream-Schritt, ohne Rückkopplung in die Core-Gewichte.

### 6.3 Hintergrundgradienten-Extraktion vor PCC (BGE) (optional, empfohlen)

Hintergrundgradienten (z.B. künstliche Lichtverschmutzung, Mondlicht, Airglow) können die Photometric Color Calibration (PCC) verzerren, besonders wenn Gradienten spektral nicht gleichmäßig über die Kanäle verteilt sind.  
Zur Abmilderung darf **vor PCC** ein additiver Background Gradient Extraction (BGE)-Schritt angewendet werden.

#### 6.3.1 Prinzip

Für jeden Kanal `c` wird ein glattes großskaliges Hintergrundmodell `B_c(x,y)` geschätzt und subtrahiert:

`I'_c(x,y) = I_c(x,y) - B_c(x,y)`

BGE muss:

- strikt additiv,
- kanalweise,
- unabhängig von der Frame-Gewichtungslogik
- und frei von nichtlinearen Tontransformationen sein.

#### 6.3.2 Tile-getriebenes Sampling-Gitter (verbindlich)

Die Rekonstruktions-Tiles werden als Hintergrund-Sampling-Einheiten wiederverwendet. Ziel ist, **objektfreie** Hintergrundsamples pro Tile zu erhalten.

##### (a) Definition der Hintergrundmaske (verbindlich)

Für jedes Tile `t` und jeden Kanal `c` definiere eine Binärmaske `M_bg`, die Pixel markiert, die als Hintergrundsamples zugelassen sind. `M_bg` muss ausschließen:

1. **Canvas-ungültige Pixel:** Pixel außerhalb des gültigen Canvas-Supports des rekonstruierten Bildes `I_rec`. Canvas-maskierte Regionen dürfen unabhängig von ihren Pixelwerten nie in BGE-Berechnungen eingehen. Dies ist eine verbindliche Einschränkung.
2. **Sterne:** Pixel in `M_star` (aus Sternerkennung oder Segmentierung), optional dilatiert um `mask.star_dilate_px` (empfohlener Default: 2–6 px).
3. **Hochstruktur-Pixel:** Pixel mit `structure_metric(p) > structure_thresh`, wobei `structure_metric` aus lokalen Gradienten abgeleitet wird (z.B. High-Pass-Energie) und `structure_thresh` konfigurierbar ist.
4. **Gesättigte Pixel:** Pixel mit `I >= sat_level` und optionaler Dilatationsmarge `mask.sat_dilate_px`.
5. **Optionale Objektmaske:** falls verfügbar (Nebel-/Galaxienmaske), ausschließen, um Bias in Feldern mit ausgedehnten Objekten zu verhindern.

Falls keine Sternerkennung verfügbar ist, darf `M_star` deterministisch über Thresholding einer Bandpass-/DoG-Antwort plus Dilatation approximiert werden.

##### (b) Robustes Tile-Hintergrundsample (verbindlich, konfigurierbar)

Berechne pro Tile ein robustes Hintergrundsample über ein konfigurierbares Quantil:

`b_{t,c} = quantile_q(R_{t,c}[M_bg])`

mit:

- `q = bge.sample_quantile` in `(0, 0.5]`
- **Default:** `q = 0.20` (20%-Quantil)
- der Median ergibt sich durch `q = 0.50`

Begründung: das niedrigere Quantil ist resistenter gegen schwache Objektkontamination und unvollständige Masken, während der Median in spärlichen Feldern mit starker Maskierung akzeptabel ist.

Das Sample wird mit dem Tile-Zentrum `(x_t, y_t)` verknüpft.

##### (c) Tile-Zuverlässigkeitsgewicht (optional, empfohlen)

Tiles dürfen für das spätere Fitting ein Zuverlässigkeitsgewicht erhalten:

`w_t = exp(-lambda_structure * structure_score_t) * (1 - masked_fraction_t)`

mit:

- `masked_fraction_t = (Anzahl der M_bg=0-Pixel in Tile t) / (Gesamtzahl canvas-gültiger Pixel in Tile t)` — der Anteil canvas-gültiger Pixel, die durch die Hintergrundmaske ausgeschlossen werden.
- `structure_score_t` ist definiert als dimensionslose relative Hochpassenergie:

  `structure_score_t = median_{p in M_bg_t} ( hp(R_{t,c}(p)) / max(sigma_{t,c}^{bg}, eps_sigma) )^2`

  wobei `hp(x)` das Hochpass-Residual nach einem Box-Blur mit Radius `bge.structure_blur_px` (empfohlener Default: 5 px) ist, angewendet nur auf canvas-gültige Pixel innerhalb des Tiles, und `sigma_{t,c}^{bg}` die lokale Hintergrund-Rauschskala auf demselben Tile/Kanal ist. Bevorzugte Quelle: der lokale STRUCTURE-Rauschproxy aus §5.5.3, sofern verfügbar; deterministischer Fallback: eine robuste MAD-basierte Schätzung auf den unmaskierten Hintergrundpixeln `M_bg_t`. Wenn im Tile keine `M_bg`-Pixel verfügbar sind, dann gilt `structure_score_t = +inf` und `w_t = 0`.

- `lambda_structure > 0` ist ein Dämpfungsfaktor (empfohlener Default: `lambda_structure = 2.0`).

Diese Normierung ist verbindlich: Eine globale multiplikative Intensitätsskalierung desselben Bildinhalts darf `structure_score_t` oder `w_t` für sich genommen nicht verändern, abgesehen von Floating-Point-Rauschen.

Diese Formel ist verbindlich und gilt sowohl für §6.3.2(c) als auch für §6.3.4. **Canvas-ungültige Pixel dürfen weder zu `masked_fraction_t` noch zu `structure_score_t` beitragen.**

#### 6.3.3 Aggregation auf ein grobes Gitter (verbindlich)

Um Overfitting kleiner Strukturen zu vermeiden, werden Tile-Samples vor dem Surface-Fit auf ein **gröberes** Gitter aggregiert.

##### (a) Gitterdefinition

Gegeben ein Gitterabstand `G` (siehe 6.3.9), definiere achsenparallele Gitterzellen über der Bildebene. Jede Zelle ist ein Rechteck der Größe `G x G`.

##### (b) Zuordnung von Tiles zu Gitterzellen (verbindlich)

Jedes Tile-Sample `(x_t, y_t, b_{t,c}, w_t)` wird über Integer-Binning seines Zentrums genau einer Gitterzelle zugeordnet:

`cell_x = floor(x_t / G)`  
`cell_y = floor(y_t / G)`

(Alle Tiles, deren Zentren in dieselbe `G x G`-Zelle fallen, gehören zu dieser Zelle.)

##### (c) Aggregation pro Zelle (verbindlich)

Für jede Zelle und jeden Kanal `c` aggregiere alle zugeordneten Tile-Samples:

- Wert: `b_cell = median({b_{t,c}})` (robust)
- Gewicht: `w_cell = median({w_t})` (normativer Default)

##### (d) Unzureichende Samples (verbindlich)

Eine Gitterzelle gilt als **unzureichend**, wenn sie weniger als

`n_cell < bge.min_tiles_per_cell`

enthält.

Empfohlener Default: `bge.min_tiles_per_cell = 3`.

Unzureichende Zellen müssen deterministisch behandelt werden durch:

1. **Discard (Default):** Zelle vom Fit ausschließen, oder
2. **Nearest-neighbor fill:** `(b_cell, w_cell)` durch die nächste ausreichende Zelle ersetzen (gemessen über den euklidischen Abstand der Zellzentren), oder
3. **Radius expansion:** deterministisch Tiles aus Nachbarzellen innerhalb eines Radius `r = k*G` einbeziehen, bis `n_cell >= min_tiles_per_cell`.

Die gewählte Strategie muss konfigurierbar und in der Diagnostik festgehalten sein.

#### 6.3.4 Surface Fitting

Fitte pro Kanal eine glatte Hintergrundfläche über:

- robustes 2D-Polynom (Ordnung 2–3 empfohlen), oder
- Thin-plate Spline, oder
- bikubischen Spline mit robustem Loss, oder
- Radial Basis Function (RBF) mit Glättung (nur mit expliziter Regularisierung empfohlen), oder
- foreground-aware modeled-mask mesh sky surface (`modeled_mask_mesh`) für Szenen mit großen diffusen Vordergrundstrukturen.

Optionale Gewichte: verwende `w_t` wie in §6.3.2(c) definiert (verbindliche Formel: `exp(-lambda_structure * structure_score_t) * (1 - masked_fraction_t)`, mit dimensionslosem `structure_score_t`). Canvas-ungültige Pixel müssen sowohl aus `structure_score_t` als auch aus `masked_fraction_t` ausgeschlossen werden, wie dort spezifiziert.

Robusten Loss (Huber/Tukey) verwenden.

#### 6.3.5 Subtraktion

`I'_c(x,y) = I_c(x,y) - B_c(x,y)`

Keine multiplikative Korrektur erlaubt.

#### 6.3.6 Validierungsanforderungen

Wenn BGE aktiviert ist:

1. Hintergrund-RMS muss sinken oder stabil bleiben.
2. Keine künstliche Krümmung über Tile-Grenzen.
3. Sternflux-Verhältnisse müssen innerhalb der Toleranz stabil bleiben.
4. PCC-Residuen müssen sich gegenüber der No-BGE-Baseline verbessern oder stabil bleiben.

BGE darf die Core-Gewichte (`G`, `L`, `W`) nicht verändern.

#### 6.3.7 Auto-getuntes BGE (optional, konservativ) (verbindlich wenn aktiviert)

Diese optionale Erweiterung erlaubt deterministisches **test–adjust–test**-Tuning von BGE-Parametern, um die Robustheit unter wechselnden Gradientenbedingungen (Lichtverschmutzung, Mondgradienten, Airglow) zu verbessern. Der Rekonstruktionskern bleibt unverändert; BGE bleibt strikt additiv und downstream.

##### 6.3.7.1 Zielgröße (verbindlich)

Für einen gegebenen Kanal definiere eine deterministische helligkeitsnormalisierte Zielgröße:

`B_ref = max(abs(median_i b_i), eps_bg)`

`J = E_cv / B_ref + alpha_f * E_flat / B_ref^2 + beta_r * E_rough / B_ref`

mit:

- `E_cv`: Holdout-RMS der Hintergrundsample-Residuen, ausgewertet auf einem deterministischen Validierungssplit,
- `E_flat`: großskalige Gradientenenergie des gefitteten Hintergrundmodells,
- `E_rough`: Energie der zweiten Ableitung des Modells (bestraft overfitte Welligkeit).
- `B_ref`: robuste Hintergrundamplitude aus den Grid-Zell-Werten `{b_i}` desselben Kandidaten.
- `eps_bg > 0`: Null-sicherer Normierungsboden (empfohlener Default: `1e-12` in normierten Dateneinheiten).

**Notation:** `alpha_f` (Flatness-Gewicht) und `beta_r` (Roughness-Gewicht) sind verschieden von den globalen Qualitätsindex-Gewichten `alpha`, `beta`, `gamma` (§5.3.2).

Alle Terme müssen deterministisch aus demselben Grid-Zell-Set berechnet werden. Diese Normierung ist verbindlich: Eine globale multiplikative Reskalierung der Eingangsdaten darf das Kandidaten-Ranking nicht verändern, abgesehen von vernachlässigbaren Floating-Point-Differenzen.

##### 6.3.7.2 Deterministischer Holdout-Split (verbindlich)

Grid-Zellen müssen nach `(cell_y, cell_x)` sortiert und deterministisch aufgeteilt werden, indem jede k-te Zelle als Validierung gewählt wird, wobei `k = round(1/holdout_fraction)`.

`holdout_fraction` muss vor der Split-Generierung auf `[0.05, 0.50]` geclippt werden.

##### 6.3.7.3 Kandidatensuche (konservative Defaults)

Wenn aktiviert, müssen Implementierungen eine begrenzte Menge an Parameterkandidaten auswerten (harte Obergrenze `max_evals`) und den Kandidaten mit minimalem `J` nach deterministischen Tie-Break-Regeln auswählen (bevorzuge geringere Rauigkeit, dann gröberes effektives Modell).

Konservative Kandidatenfamilien (implementierungsabhängig, aber zu dokumentieren):

- `sample_quantile`: `{q0, 0.35, 0.50}`
- `structure_thresh_percentile`: `{p0, 0.85}`
- `rbf_mu_factor`: `{m0, 1.4}`
- `rbf_lambda`: darf intern dennoch eine Glättungspräferenz anwenden (glättestes akzeptiertes `lambda` wählen)

Der Gitterabstand `G` soll im konservativen Modus unverändert bleiben, sofern keine explizit nicht-konservative Strategie aktiviert ist.

`max_evals` ist eine harte Obergrenze ausgewerteter Kandidaten und muss `>= 1` sein.

##### 6.3.7.4 Konfigurations-Hooks (normative Namen)

- `bge.autotune.enabled: true|false`
- `bge.autotune.max_evals`
- `bge.autotune.holdout_fraction`
- `bge.autotune.alpha_f` (Flatness-Gewicht, entspricht `alpha_f` in der Zielgröße)
- `bge.autotune.beta_r` (Roughness-Gewicht, entspricht `beta_r` in der Zielgröße)
- `bge.autotune.strategy: conservative|extended`

Wenn `enabled=true`, müssen der gewählte Parametersatz und die Zielgrößenkomponenten in der Diagnostik enthalten sein.

Minimale Diagnosefelder (verbindlich):

- `autotune.enabled`
- `autotune.strategy`
- `autotune.max_evals`
- `autotune.evals_performed`
- `autotune.best.sample_quantile`
- `autotune.best.structure_thresh_percentile`
- `autotune.best.rbf_mu_factor`
- `autotune.best.objective` (verbindliche normalisierte Zielgröße `J`)
- `autotune.best.objective_raw` (empfohlene zusätzliche Diagnose)
- `autotune.best.cv_rms`
- `autotune.best.flatness`
- `autotune.best.roughness`
- `autotune.fallback_used`

##### 6.3.7.5 Robustheits- und Fallback-Semantik (verbindlich)

Auto-Tuning muss fail-safe und deterministisch sein:

1. Wenn ein Kandidaten-Fit nicht genügend gültige Zellen/Metriken erzeugen kann, wird dieser Kandidat als fehlgeschlagen markiert.
2. Wenn kein Kandidat erfolgreich ist, muss die Implementierung unverändert auf die Nutzer-/Basis-BGE-Konfiguration zurückfallen.
3. Tie-Break bei gleichen Zielwerten muss deterministisch sein (bevorzuge geringere Rauigkeit, dann gröberes effektives Modell).
4. Auto-Tuning darf die Core-Rekonstruktionsgewichte (`G`, `L`, `W`) nicht verändern und bleibt strikt additiv.

### 6.3.8 Mathematisches Surface-Modell (verbindlich)

Seien die Hintergrundsamples definiert als:

`(x_i, y_i, b_i, w_i)`  für i = 1..M

wobei:

- `(x_i, y_i)` die Mittelpunkte der Gitterzellen sind,
- `b_i` die robuste Hintergrundschätzung ist,
- `w_i` ein optionales Zuverlässigkeitsgewicht ist.

Eine robuste Polynomfläche der Ordnung d (empfohlen d = 2 oder 3) ist definiert als:

`B_c(x,y) = sum_{m+n <= d} a_{mn} x^m y^n`

Die Koeffizienten `a_{mn}` werden bestimmt durch Minimierung von:

`argmin_a sum_i w_i * rho( b_i - B_c(x_i,y_i) )`

wobei `rho` eine robuste Loss-Funktion ist, z.B.:

Huber-Loss:

`rho(r) = 0.5 * r^2                        if |r| <= delta`
`rho(r) = delta * (|r| - 0.5 * delta)      otherwise`

(Äquivalent: `delta * |r| - 0.5 * delta^2` für den linearen Zweig.)

oder Tukey-Biweight-Loss.

Der Fit muss über Iteratively Reweighted Least Squares (IRLS) oder eine äquivalente deterministische robuste Optimierung gelöst werden.

Thin-plate-Spline-Alternative:

`B_c = argmin_B sum_i w_i (b_i - B(x_i,y_i))^2 + lambda * J_TPS(B)`

wobei die Standard-2D-TPS-Biegeenergie ist:

`J_TPS(B) = integral [ (d^2B/dx^2)^2 + 2*(d^2B/dxdy)^2 + (d^2B/dy^2)^2 ] dx dy`

mit Regularisierungsparameter `lambda`, der die Glätte steuert.

Nur großskalige (niederfrequente) Komponenten sind erlaubt; Overfitting ist verboten.

#### 6.3.9 Adaptive Gitterdefinition (verbindlich)

Der Gitterabstand `G` muss mit den Bilddimensionen skalieren, um Under- oder Overfitting zu vermeiden.

Definiere:

`G = clip( max(2*T, min(W,H)/N_g), G_min, G_max )`

Empfohlene Defaults:

- `N_g = 32` (Ziel-Gitterauflösung über die kleinste Bildachse)
- `G_min = 64 px`
- `G_max = min(W,H)/4`

Damit wird sichergestellt:

- das Hintergrundmodell erfasst nur großskalige Gradienten,
- die Gitterdichte passt sich an die Sensorauflösung an,
- kleine Bilder werden nicht überparametrisiert,
- große Mosaike behalten ausreichende räumliche Abtastung.

Implementierungen müssen garantieren, dass die Gitterauflösung gröber als die Tile-Auflösung ist (`G >= 2*T`) **außer** im Compact-Tile-Modus.

**Ausnahme im Compact-Tile-Modus (verbindlich):** Wenn der Compact-Tile-Modus aktiv ist (§5.4 Schritt 4, `T = min(W,H)`), kann die Bedingung `G >= 2*T` nicht erfüllt werden, weil `G_max = min(W,H)/4 < 2*T`. In diesem Fall **muss** BGE deaktiviert werden (`bge.enabled = false` implizit), oder der Grid-Abstand wird auf `G = G_max` festgesetzt und mit einer expliziten Diagnosewarnung protokolliert. Implementierungen müssen dokumentieren, welches Verhalten aktiv ist, und dürfen die `G >= 2*T`-Bedingung nicht stillschweigend verletzen.

---

#### 6.3.10 Mathematische Spezifikation der RBF-Surface (verbindlich, wenn `bge.fit.method = rbf`)

Seien die Grid-Zell-Samples `(r_j, b_j, w_j)` für `j = 1..J`, wobei:

- `r_j = (x_j, y_j)` die Mittelpunkte der Grid-Zellen sind
- `b_j` der aggregierte Hintergrundwert ist
- `w_j >= 0` das Zuverlässigkeitsgewicht der Zelle ist (wie in §6.3.2(c) definiert)

Definiere die RBF-Surface mit affinem Trendterm:

`B_c(r) = sum_{i=1..J} u_i * phi(||r - r_i||; mu) + a0 + a1*x + a2*y`

wobei:

- `u_i` die RBF-Koeffizienten (unbekannt) sind
- `(a0, a1, a2)` ein affiner Term ist (empfohlen; verbessert die Extrapolationsstabilität)
- `mu > 0` der Kernel-Shape-/Scale-Parameter ist
- `phi` der gewählte radiale Kernel ist

##### Unterstützte Kernel (verbindlich)

1. Multiquadric:

   `phi(d; mu) = sqrt(d^2 + mu^2)`

2. Thin-plate-Spline-Kernel:

   `phi(d) = d^2 * log(d + eps_rbf)`

   mit kleinem `eps_rbf > 0` für numerische Stabilität (empfohlen: `eps_rbf = 1e-6 * G`).

3. Gaussian:

   `phi(d; mu) = exp(-d^2 / (2 * mu^2))`

   Für Gaussian wirkt `mu` als Bandbreite.

##### Robuster regularisierter Fit (verbindlich)

Löse die Parameter `theta = (u, a)` durch Minimierung von:

`argmin_theta sum_{j=1..J} w_j * rho(b_j - B_c(r_j)) + lambda * ||u||_2^2`

wobei:

- `rho` der konfigurierte robuste Loss ist (Huber oder Tukey, wie in §6.3.8 definiert)
- `lambda > 0` zwingende Regularisierung bei Verwendung von RBF ist
- die Optimierung deterministisch sein muss (IRLS oder äquivalent)

**RBF ohne Regularisierung (`lambda = 0`) ist verboten.**

Canvas-ungültige Pixel dürfen nicht zum Sample-Set `(r_j, b_j, w_j)` beitragen, das für den Fit verwendet wird.

##### Empfohlene Defaults

- `mu = G` (Grid-Abstand)
- `lambda = 1e-4` (innerhalb `[1e-6, 1e-2]` je nach Gradientenstärke abstimmbar)
- affinen Term standardmäßig einschließen

### 6.4 PCC

Diese Spezifikation empfiehlt, bei vorhandenen räumlichen Hintergrundgradienten BGE vor PCC anzuwenden.

#### 6.4.1 Lokales Hintergrundmodell im Sky-Annulus (verbindlich)

Die PCC-Sternphotometrie muss einen lokalen Hintergrund subtrahieren, der im Sky-Annulus geschätzt wird. Um Gradienten-Bias zu reduzieren, darf das Hintergrundmodell sein:

- `median`: konstanter Median über den Annulus (Legacy), oder
- `plane`: robuster Ebenenfit `bg(dx,dy)=a + b*dx + c*dy` über die Annulus-Pixel (empfohlen bei Gradienten).

Wenn der `plane`-Fit fehlschlägt, muss deterministisch auf `median` zurückgefallen werden.

#### 6.4.2 FWHM-adaptive Radien (optional, empfohlen)

Wenn eine globale Seeing-Schätzung `FWHM` verfügbar ist, dürfen PCC-Radien automatisch gesetzt werden:

- `r_ap = max(min_aperture_px, aperture_fwhm_mult * FWHM)`
- `r_in = max(r_ap + 1, annulus_inner_fwhm_mult * FWHM)`
- `r_out = max(r_in + 2, annulus_outer_fwhm_mult * FWHM)`

Wenn `FWHM <= 0` oder nicht verfügbar ist, muss deterministisch auf `FWHM = 0` zurückgefallen werden, was ergibt:

- `r_ap = min_aperture_px`
- `r_in = max(r_ap + 1, annulus_inner_fwhm_mult * 0) = r_ap + 1`
- `r_out = max(r_in + 2, annulus_outer_fwhm_mult * 0) = r_in + 2`

Empfohlene konservative Defaults:

- `aperture_fwhm_mult = 1.8`
- `annulus_inner_fwhm_mult = 3.0`
- `annulus_outer_fwhm_mult = 5.0`

Diese Änderungen müssen deterministisch bleiben.

#### 6.4.3 Konfigurations-Hooks (normative Namen)

- `pcc.background_model: median|plane`
- `pcc.radii_mode: fixed|auto_fwhm`
- `pcc.aperture_fwhm_mult`
- `pcc.annulus_inner_fwhm_mult`
- `pcc.annulus_outer_fwhm_mult`
- `pcc.min_aperture_px`

Zulässiger Downstream-Schritt, angewendet auf lineare Daten.

#### 6.4.4 Optionale Post-PCC-Unterdrückung isolierter Chroma-Speckles

Nach erfolgreicher PCC dürfen Implementierungen einen deterministischen Cleanup-Schritt für isolierte chromatische Speckles anwenden, bevor die finalen RGB-Ausgaben geschrieben werden.

Verbindliche Bedingungen für diesen optionalen Schritt:

1. Der Schritt arbeitet nur im expliziten RGB-Bereich nach PCC.
2. Er muss auf Pixel innerhalb des gültigen Canvas-Supports beschränkt sein.
3. Eine Korrektur ist nur zulässig, wenn genau ein Kanal ein isolierter lokaler Ausreißer relativ zu benachbarten gültigen RGB-Samples ist oder eine äquivalente deterministische Einzelkanal-Chroma-Ausreißer-Bedingung erfüllt ist.
4. Helle Sternkerne, kompakte reale Strukturen und gestützter mehrpixeliger chromatischer Bildinhalt müssen durch deterministische Luma- und/oder Struktur-Guards geschützt werden.
5. Ersatzwerte müssen aus robusten lokalen Nachbarschaftsstatistiken des betroffenen Kanals oder aus einem äquivalenten deterministischen lokalen Schätzer abgeleitet werden.
6. Dieser Schritt gehört zum optionalen Post-Processing und ist nicht Teil des verpflichtenden linearen Qualitätskerns.

---

## 7. Validierung und Abbruch

### 7.1 Erfolgskriterien

- FWHM-Verbesserung gegenüber dem Referenz-Stack gemäß `validation.min_fwhm_improvement_percent`
- Hintergrund-RMS nicht schlechter als der Referenzwert
- keine systematischen Tile-Seams
- stabile Sternfluss-Verhältnisse und photometrische Skalierung
- stabile Gewichtsverteilungen

### 7.2 Abbruchkriterien

- Datenintegrität verletzt (nichtlinear, unlesbar, inkonsistent)
- Registrierungsfehler über große Teile des Datensatzes
- numerische Instabilität trotz Fallbacks

### 7.3 Minimale Tests (normativ)

1. `alpha+beta+gamma=1` (globale Qualitätsindex-Gewichte, §5.3.2)
2. Clamping vor `exp` (globale und lokale Gewichte)
3. additive Hintergrundsubtraktion und multiplikative photometrische Skalierung bleiben getrennt
4. gültige negative oder null Pixel bleiben zulässige Samples
5. Tile-Monotonie in `F`, einschließlich deterministischem Compact-Tile-Fallback für kleine Bilder
6. Overlap-Konsistenz (`0<=o<=0.5`, explizit `o_clipped=clip(o,0,0.5)`, ganzzahlige `O,S`)
7. Low-Weight-Fallback ohne NaN/Inf
8. Sigma-Clipping-Guards auf `N_eff` und `D_eff`; `min_fraction`-Keep-Floor aktiv
9. weiches lokales Score-Blending bleibt um den Star-Support-Schwellenwert stetig
10. support-aware OLA deckt alle gültigen Randpixel ab und erfüllt die Partition-of-Unity-Toleranz mit `M_{t,c}(p)` wie in §5.7.1 definiert
11. keine Kanal-Kopplung
12. keine qualitätsbasierte Frame-Selektion
13. deterministische Reproduzierbarkeit
14. Registrierungskaskade einschließlich Identity-Fallback; bedingungslose Akzeptanz der Identität für den Referenzframe
15. Erhalt der CFA-Phase
16. Cluster-Aggregation ist massenerhaltend (`M_{k,c}` einbezogen) mit optionalem Dominance-Cap
17. WCS-Round-Trip-Fehler unterhalb von `validation.wcs_roundtrip_max_arcsec` (empfohlener Default: `0.5` Bogensekunden) nur, wenn WCS aktiviert ist
18. PCC-Stabilität: positive Determinante, Konditionszahl unterhalb von `validation.pcc_max_condition_number` (empfohlener Default: `1000`), Residuen unterhalb von `validation.pcc_max_residual_mag` (empfohlener Default: `0.05` mag) nur, wenn PCC aktiviert ist
19. STAR-Metrik-Koeffizientensumme: `0.6 + 0.2 + 0.2 = 1.0` (§5.5.2)
20. STRUCTURE-Metrik-Koeffizientensumme (unsigned): `0.7 + 0.3 = 1.0` (§5.5.3)
21. Canvas-ungültige Pixel tragen Null zu den OLA-Akkumulatoren `A` und `S` (§5.7.1), Null zu lokalen Metriken (§5.5) und Null zum BGE-Sampling (§6.3.2) bei
22. Clustering-Schwellenwert passt zum Modus-Framework: aktiv genau dann, wenn `N >= max(N_red, 50)` (§5.9)
23. BGE-Tile-Zuverlässigkeitsgewichte bleiben unter globaler multiplikativer Intensitätsreskalierung stabil, wenn die lokale Rauschskala konsistent mitreskaliert wird (§6.3.2(c))
24. BGE-Autotune-Kandidatenranking verwendet die normalisierte Zielgröße `J`; `autotune.best.objective` berichtet diesen normalisierten Wert (§6.3.7.1)

Hinweis: Der Legacy-PCC-Test "kein negatives Matrixelement" ist seit v3.3+ **nicht mehr** als hartes Kriterium erforderlich.

---

## 8. Empfohlene numerische Defaults

- `eps_bg = 1e-6`
- `eps_scale = 1e-6`
- `eps_mad = 1e-6`
- `eps_weight = 1e-6`
- `eps_neff = 1e-6`
- `eps_var = 1e-12`
- `tol_ola = 1e-6`
- `eps_affinity = 1e-6`
- `delta_ncc = 0.01`
- `Q`-Clamp global/lokal: `[-3, +3]`
- `k_global = 1.0`
- `k_local = 1.0`
- `min_fraction = 0.5` (Sigma-Clip-Keep-Floor)
- `lambda_bge = 1.0` (Dämpfung des BGE-Structure-Scores)
- `bge.structure_blur_px = 5`
- `validation.wcs_roundtrip_max_arcsec = 0.5`
- `validation.pcc_max_condition_number = 1000`
- `validation.pcc_max_residual_mag = 0.05`

---

## 9. Geltungsbereich: verpflichtender Kern vs. Erweiterung

### Verpflichtender Kern

- CFA-basierter Registrierungspfad bis zur expliziten oder verzögerten (von der Shared-Core-Variante abhängigen) Kanaltrennung
- globale Normalisierung
- globale/lokale Metriken und Gewichte
- Tile-Rekonstruktion einschließlich konsolidierter Fallbacks
- Clustering/Synthese/Final-Stack abhängig vom Betriebsmodus

### Optionale Erweiterung

- deterministische CFA-Defektpixel-Unterdrückung / Cosmetic Correction
- Soft-Threshold / Wiener
- alternative Sigma-Clipping-Strategien
- WCS/PCC
- Post-PCC-Unterdrückung isolierter Chroma-Speckles
- spezialisierte Performance-Backends (GPU, Queue-Worker)

### 9.1 Operative Beispielkonfigurationen (`tile_compile_cpp`)

Für den operativen Einsatz stehen vollständige Referenzkonfigurationen bereit:

- `tile_compile_cpp/examples/full_mode.example.yaml`
- `tile_compile_cpp/examples/reduced_mode.example.yaml`
- `tile_compile_cpp/examples/emergency_mode.example.yaml`
- `tile_compile_cpp/examples/smart_telescope_dwarf_seestar.example.yaml`

Alle Beispielkonfigurationen enthalten die aktive Konfigurationsoberfläche mit Inline-Kommentaren.
Sie legen Runtime-Schwellen wie `assumptions.frames_min` und `assumptions.frames_reduced_threshold` explizit offen.
Die kanal-semantische Shared-Core-Variante ist eine Implementierungseigenschaft, kein benutzerseitiger `assumptions.pipeline_profile`-Schalter.

Vorgehen:

1. passende Beispielkonfiguration kopieren,
2. `run_dir`, `input.pattern` und Sensordaten (`image_width/height`, `bayer_pattern`) anpassen,
3. Runner mit dieser Datei starten.

---

## 9.2 Eingeschränkte ML-Optimierungserweiterung (optional, nicht Core)

Diese Erweiterung führt Machine-Learning-(ML)-Module **nur** ein, um die Schätzung von Gewichten und Zustandsdeskriptoren zu verbessern, während die verpflichtenden Core-Invarianten erhalten bleiben:

### 9.2.1 Verbindliche Invarianten (harte Constraints)

1. **Keine Frame-Selektion:** Ganze Frames dürfen nicht auf Basis von Qualität entfernt werden (unverändert zu v3.2.x).
2. **Strikte photometrische Linearität des Rekonstruktionskerns:** Die finale Rekonstruktion muss ein gewichteter linearer Schätzer über Eingangsframes (und/oder Synthetic Frames) bleiben, also von der Form

   `R(p) = sum_i w_i(p) * X_i(p) / sum_i w_i(p)`

   mit `w_i(p) >= 0` und deterministischen Fallbacks.
3. **Determinismus:** ML-Inferenz muss deterministisch sein (feste Modellgewichte, festes Preprocessing, feste Seeds, wo anwendbar).
4. **Keine halluzinierten Inhalte:** ML-Module dürfen keine neuen räumlichen Strukturen erzeugen. ML-Ausgaben sind auf **Gewichte, Masken, Metriken und Zustandslabels** beschränkt.
5. **Kanaltrennung bleibt erhalten:** ML-Module müssen pro Kanal oder auf explizit definierten kanalaggregierten Features arbeiten; keine implizite kanalübergreifende Kopplung im Core-Schätzer.

### 9.2.2 Erlaubte ML-Ausgaben (eingeschränkt)

ML darf Folgendes ausgeben, sofern die Ausgaben deterministisch und begrenzt sind:

- globaler Qualitätswert pro Frame/Kanal: `Q̂_{f,c}` (dimensionslos, auf `[-3,+3]` gemappt/geclippt)
- globales Gewicht pro Frame/Kanal: `Ĝ_{f,c} = exp(k_global * Q̂_{f,c})`
- lokaler Tile-Qualitätswert: `q̂_{f,t,c}` (dimensionslos, geclippt auf `[-3,+3]`)
- lokales Tile-Gewicht: `L̂_{f,t,c} = exp(q̂_{f,t,c})`
- Pixel-Zuverlässigkeitsmaske (soft, keine harte Verwerfung): `M̂_{f,t,c}(p) in [m_min, 1]` mit empfohlenem `m_min = 0.05`
- Zustandsdeskriptor für Clustering (pro Frame): `v̂_f` (Feature-Vektor)
- Zustandslabels (Cluster) und/oder Übergangswahrscheinlichkeiten (HMM), ausschließlich zur Bildung von Synthetic Frames

Verbotene ML-Ausgaben:

- direkte Vorhersage rekonstruierter Pixelintensitäten (End-to-End-Bilderzeugung)
- Super-Resolution oder Inpainting, die räumliche Details erzeugen, die nicht durch die Eingabe gestützt sind
- jegliches stochastisches Sampling zur Inferenzzeit

### 9.2.3 ML-getriebenes effektives Gewicht (verbindlich)

Wenn ML-Module aktiviert sind, darf das effektive Gewicht auf Pixelebene erweitert werden:

`Ŵ_{f,t,c}(p) = Ĝ_{f,c} * L̂_{f,t,c} * M̂_{f,t,c}(p)`

Die Tile-Rekonstruktion bleibt ein gewichtetes Mittel:

`R_{t,c}(p) = sum_f Ŵ_{f,t,c}(p) * I_{f,c}(p) / sum_f Ŵ_{f,t,c}(p)`

Die Fallback-Regel bleibt unverändert: wenn der Nenner < `eps_weight`, fällt die Implementierung auf das ungewichtete Mittel zurück.

**Integration von ML-Masken und Sigma-Clipping (verbindlich):**

1. **Gültige Sample-Menge:** Die gültige Sample-Menge `V_{t,c}(p)` (§5.7) bleibt über das Canvas-/Finitheits-Kriterium definiert. Die Soft-Maske `M̂_{f,t,c}(p) in [m_min, 1]` ändert die Mitgliedschaft in `V_{t,c}(p)` NICHT. Stattdessen skaliert sie das effektive Gewicht des Frames auf Pixelebene.
2. **Sigma-Clipping mit ML-Gewichten:** Wenn Sigma-Clipping aktiv ist (§5.7), müssen die gewichteten Statistiken pro Iteration das Pixel-Gewicht `Ŵ_{f,t,c}(p)` statt des Tile-Gewichts `w_{f,t,c}` verwenden. Alle übrigen Sigma-Clipping-Semantiken (`min_fraction`-Keep-Floor, Fallback auf das ungeclippte Mittel, `N_eff`-Guard, `D_eff`-Guard) bleiben unverändert.
3. **Keep-Floor mit Soft-Masken:** Der Keep-Floor `ceil(min_fraction * |V_{t,c}(p)|)` zählt alle Frames in `V_{t,c}(p)` unabhängig von ihrem Soft-Maskenwert. Das verhindert, dass nahezu nullige Maskenwerte die aktive Menge trivial verkleinern.
4. **Canvas-Invariante:** Der in §5.7.1 definierte Canvas-Ausschluss gilt bedingungslos. `M̂_{f,t,c}(p) > 0` für ein canvas-ungültiges Pixel macht dieses Pixel nicht zu einem gültigen Sample. Canvas-Ungültigkeit hat Vorrang vor jedem ML-Maskenwert.

### 9.2.4 Empfohlene Lernparadigmen (nicht verbindliche Leitlinien)

Weil Ground Truth typischerweise nicht verfügbar ist, sollte priorisiert werden:

- **Self-supervised Learning:** Konsistenz über zufällige Frame-Subsets, Noise2Self/Noise2Void-artige Ziele für Masken-/Denoising-Proxys (Hinweis: Denoising darf innerhalb des verpflichtenden Rekonstruktionskerns weiterhin nur Masken/Gewichte, nicht Pixel, ausgeben).
- **Weak Supervision über Proxys:** Gewichte so optimieren, dass deterministische Metriken (FWHM, Elliptizität, Hintergrund-RMS, Seam-Score) auf Validierungssätzen verbessert werden.
- **Uncertainty-aware Models:** Unsicherheit ausgeben, um überkonfidentes Downweighting zu vermeiden; Unsicherheit muss in begrenzte Masken/Gewichte gemappt werden.

### 9.2.5 Modelle, die zur eingeschränkten Output-Constraint passen (nicht verbindlich)

- globale Gewichte: Gradient-Boosted Trees (GBM), kleine MLPs auf Frame-Metriken
- Tile-Qualität: kleine CNN-Encoder / leichte ViT-tiny (nur bei ausreichendem Datenvolumen)
- Pixel-Zuverlässigkeitsmasken: U-Net-lite mit Ausgabe `M̂(p)` in `[m_min,1]`

LLMs sind nur für **Konfigurationssynthese, Interpretation von Validierungsreports und Testgenerierung** zulässig, nicht für pixelweise Rekonstruktion.

### 9.2.6 Validierungsanforderungen für die ML-Erweiterung (verbindlich)

Wenn ML aktiviert ist, gelten weiterhin alle verpflichtenden Core-Validierungstests, plus:

1. **Begrenzte Ausgaben:** erzwinge `Q̂ in [-3,+3]`, `M̂ in [m_min,1]`
2. **Deterministische Inferenz:** identische Eingaben liefern identische Gewichte/Masken
3. **Keine Struktursynthese:** Residuen-Korrelation darf keine unphysikalische Hochfrequenz-Injektion zeigen; Seams und Ringing dürfen gegenüber der Non-ML-Baseline nicht zunehmen
4. **Photometrische Konsistenz:** Flux-Verhältnisse von Kalibrationssternen bleiben gegenüber dem Baseline-Core innerhalb der Toleranz (konfigurierbar)
5. **Ablation:** Baseline (ohne ML) vs. ML-aktivierte Verbesserungen auf demselben Datensatz berichten

### 9.2.7 Konfigurations-Hooks (normative Namen)

Vorgeschlagene (nicht erschöpfende) Konfigurationsschlüssel:

- `ml.enable: true|false`
- `ml.global_model.path`
- `ml.tile_model.path`
- `ml.mask_model.path`
- `ml.mask.m_min`
- `ml.inference.device: cpu|gpu`
- `ml.inference.deterministic: true`

Implementierungen müssen fehlende ML-Modelle als kontrollierten Fallback auf den Non-ML-Core behandeln.

*Die mathematische Spezifikation der RBF-Surface wurde in die **§6.3.10** innerhalb des BGE-Abschnitts verschoben, wo sie hingehört.*

---

## 10. Kernaussage

Die Methode ersetzt die starre Suche nach "besten Frames" durch robuste raum-zeitliche Qualitätsmodellierung, verwendet alle Frames ohne qualitätsbasierte Selektion und rekonstruiert Signal dort, wo es physikalisch und statistisch am zuverlässigsten ist.
