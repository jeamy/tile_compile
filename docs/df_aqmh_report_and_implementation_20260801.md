# DF-AQMH vs. stacking program Vergleichsreport (überarbeitet)

**Run:** `20260801_080824_dd07d775` (DF-AQMH, 645 Frames, M31)
**stacking program-Referenz:** `result.fit` (655 Frames, M31, 10s Belichtung, Gain 80)
**Datum:** 2026-08-01

---

## 0. Lesefaden

Dieser Report trennt drei voneinander unabhängige Effekte, die im ersten
Entwurf vermischt waren und nicht gemeinsam analysiert werden dürfen:

| # | Effekt | Ursache | Lösungsbereich |
|---|--------|---------|----------------|
| 1 | Hintergrund-Dynamik geht verloren | Background-Modell wird nicht getrennt vom Residuum bewahrt; globale Scalar-Wiederaddition | Background-Model-Cache pro Frame (Stufe A/B) |
| 2 | Nullbereiche in den Ecken | Canvas-Erweiterung + `common_overlap_required_fraction=1.0` | Canvas-/Overlap-Masken |
| 3 | Validierung vergleicht Äpfel mit Birnen | Uniform-Control aus R-Kanal, Validierung auf Luminanz | Luminanz-Control korrekt bilden |

Zusätzlich gibt es einen echten **Qualitätsbefund** (FWHM/Peak vs. stacking program),
der nicht mit den internen Control-Gates verwechselt werden darf.

---

## 1. Qualitätsbefund: FWHM und Peak/Flux vs. stacking program

### 1.1 Messwerte (normalisiert auf [0,1], gleiche Methode für beide Bilder)

| Kanal | DF FWHM median | stacking program FWHM median | Verhältnis |
|-------|----------------|-------------------|------------|
| R | 11.5 px | 7.5 px | 1.53x schlechter |
| G | 13.4 px | 4.2 px | 3.17x schlechter |
| B | 8.7 px | 8.4 px | 1.04x vergleichbar |

| Kanal | DF Laplacian | stacking program Laplacian | DF DoG(1,3) | stacking program DoG(1,3) |
|-------|-------------|-----------------|-------------|-----------------|
| R | 5.72e-04 | 5.50e-04 | 2.21e-04 | 2.20e-04 |
| G | 4.43e-04 | 5.33e-04 | 2.04e-04 | 2.24e-04 |
| B | 6.92e-04 | 6.00e-04 | 2.40e-04 | 2.31e-04 |

### 1.2 Interpretation

- Median-FWHM ist nur moderat schlechter (R: 1.53x, B: 1.04x), aber der
  G-Kanal ist mit 3.17x deutlich schlechter.
- p90-FWHM und Peak/Flux sind deutlich schlechter; stacking program gewinnt bei fast
  allen Sternen.
- Die Hochfrequenz-Energie (Laplacian, DoG) ist **vergleichbar** - die
  generelle Detailwiedergabe ist nicht das Hauptproblem.
- Das Problem ist die **Sternform**: Sterne sind im DF-AQMH weiter
  aufgeweitet, besonders im G-Kanal.

**Hinweis zur Messunsicherheit:** Die per-Kanal-FWHM-Werte hängen stark von
Sternerkennung, Normierung und Hintergrundsubtraktion ab. Eine unabhängige
positionsgematchte Analyse auf Luminanz (`0.25R + 0.5G + 0.25B` aus
`reconstructed_R/G/B.fit` gegen `result.fit`) ergab Median-FWHM **23.26" (DF)
vs. 22.19" (stacking program)** — nur ~5 % schlechter — aber p90-FWHM
**29.42" vs. 23.26"** (~26 % schlechter). Der G-Kanal als Ausreisser sollte
mit einer separaten Messmethode verifiziert werden, bevor er als primärer
Qualitätsindikator verwendet wird.

### 1.3 Wichtige Abgrenzung

Die im vorherigen Testrun-Log genannte "FWHM-Verbesserung von 61.2%" und die
negative FWHM-Regression (-0.247) beziehen sich auf den Vergleich
**DF-AQMH gegen den internen Uniform-Control**, nicht gegen stacking program.

| Vergleich | FWHM DF | FWHM Referenz | Aussage |
|-----------|---------|---------------|---------|
| DF vs. Uniform-Control (intern) | 3.04 px | 4.03 px | DF ist **besser** als Control |
| DF vs. stacking program (extern) | 11.5 px (R) | 7.5 px (R) | DF ist **schlechter** als stacking program |

Diese beiden Vergleiche dürfen nicht vermischt werden. Der interne
Control-Gate sagt nur, dass AQMH das Signal schärfer rekonstruiert als ein
ungewichteter Stack - nicht, dass das Ergebnis schärfer als das
stacking-program-Ergebnis ist.

---

## 2. Effekt 1: Hintergrund-Dynamik geht verloren

### 2.1 Symptom

| Kanal | DF bg_sig (normalisiert) | stacking program bg_sig | Verhältnis |
|-------|--------------------------|--------------|------------|
| R | 0.0070 | 0.0052 | 1.36x |
| G | 0.0047 | 0.0041 | 1.15x |
| B | 0.0091 | 0.0055 | 1.64x |

DF-AQMH ist in allen Kanälen rauschvoller. Das Histogramm der
Hintergrundwerte ist komprimierter als bei stacking program - die räumliche
Hintergrundstruktur (Gradient, Nebel, Lichtverschmutzung) geht verloren.

**Differenzierung:** Die erhöhte `bg_sig` entsteht teilweise durch
Nullbereiche, Hotpixel und Canvas-Ränder, die das lokal gemessene Rauschen
vergrössern. Das eigentliche Hintergrundproblem ist nicht allein höheres
Rauschen, sondern die **künstliche Kompression** des Hintergrunds: `reconstructed_R.fit` zeigt beispielsweise p1-p90 innerhalb von nur ~4 ADU (Skala 0..4095). Das heisst, der Hintergrund ist fast eine konstante Ebene, während bei stacking program die Hintergrundverteilung die reale räumliche Struktur bewahrt. Eine reine Rauschreduktion ohne räumliche Background-Map würde diese Struktur nicht wiederherstellen.

### 2.2 Ursache: Background-Modell wird nicht getrennt vom Residuum bewahrt

Die Normalisierungs-Phase (`runner_phase_metrics.cpp`) berechnet aktuell pro
Frame und Kanal nur einen **skalaren** Hintergrundwert (Median über die
Background-Maske) und subtrahiert ihn von jedem Pixel:

```cpp
s.background_r = br;   // ein Scalar fuer das gesamte Frame
s.background_g = bg;
s.background_b = bb;
s.scale_r = 1.0f / std::max(pr, eps_b);
// apply_normalization_inplace:
// img(y,x) = (img(y,x) - background_channel) * scale_channel
```

Die AQMH-Reconstruction arbeitet danach auf einem Residuum, in dem die
frameweisen additiven Hintergrundpegel bereits entfernt wurden. Am Ende wird
nur der globale Median aller Frame-Hintergruende wieder addiert:

```cpp
// runner_pipeline.cpp:5565-5573 (Kommentare 5565-5567, Code 5568-5573)
// TODO(bg-model): global scalar background restore is a placeholder.
// Per-frame background is not preserved here; replace with Background-
// Model-Cache accumulation once Stufe A/B is implemented.
R_out *= output_scale_r;
R_out.array() += (output_bg_r + output_pedestal);   // globaler Scalar

// runner_phase_post_stack_output.cpp:181-189 (Kommentare 181-183, Code 184-189)
// TODO(bg-model): global scalar background restore is a placeholder.
// Replace with per-frame background model once Background-Model-Cache
// (Stufe A/B) is implemented.
recon_R.array() *= scaling.scale_r;
recon_R.array() += scaling.bg_r;   // globaler Scalar
```

> **Code-Kommentare aktualisiert:** Die ursprünglich irreführenden Kommentare
> ("already carries the per-frame background field" bzw. "restores each
> frame's normalized background before aggregation") wurden durch
> `TODO(bg-model)`-Marker ersetzt. Die Kommentare machen nun korrekt deutlich,
> dass die globale Scalar-Wiederaddition ein Platzhalter ist und durch den
> Background-Model-Cache aus Stufe A/B ersetzt werden muss.

Der Hauptfehler ist deshalb nicht nur eine falsche letzte Skalierungszeile. Die
Pipeline besitzt keinen separaten Vertrag fuer die additive Background-
Komponente. Sie verwirft frameweise Hintergrundinformation und ersetzt sie
spaeter durch drei globale Scalars. Ein vorhandener Gradient im Residuum kann
teilweise erhalten bleiben; frameweise Offsets und die vollstaendige
Background-Komponente sind jedoch nicht mehr rekonstruierbar.

Ein einfacher Versuch, die Frame-Scalars vor der AQMH-Reconstruction wieder
zu addieren, ist **keine** korrekte Loesung: Die AQMH-Qualitaetsgewichte koennen
dann Backgroundpegel mit Signalqualitaet koppeln und einen stark verschobenen
Control-/Background-Gate erzeugen. Die Background-Komponente muss daher
separat modelliert und akkumuliert werden.

### 2.3 Warum Output-Skalierung allein nicht reicht

Eine bloße Änderung der Output-Skalierung (z.B. anderer Stretch oder
Per-Channel-Stretch) würde das Histogramm zwar optisch strecken, aber die
**verlorene räumliche Hintergrundinformation** nicht wiederherstellen. Die
Information ist bereits vor der AQMH-Reconstruction verloren gegangen.

### 2.4 Lösung: getrenntes Background-Modell mit verbindlichem Datenvertrag

Der Background darf nicht erst am Output rekonstruiert werden. Die
Implementierung muss Background und Signal bereits vor der AQMH-Reconstruction
trennen und beide Datenpfade parallel führen.

#### 2.4.1 Verbindliche Signaldomäne

Die Implementierung verwendet eine feste Referenzdomäne. Es werden keine neuen
Benutzerparameter eingeführt; Gridgröße, Aggregation und Fallback sind bewusste
interne Implementierungsentscheidungen und werden über die Cache-Formatversion
festgeschrieben.

Für jeden Frame und Kanal gilt:

```text
p_frame = exposure / reference_exposure
background_reference = background_model_raw / p_frame
residual_reference = (raw_channel - background_model_raw) / p_frame
raw_channel / p_frame = residual_reference + background_reference
```

Dabei gilt:

- `background_model_raw` wird aus dem kalibrierten Rohframe vor der Scalar-
  Normalisierung geschätzt;
- `residual_reference` wird für Registration, Q-Maps und AQMH verwendet;
- `background_reference` wird niemals mit AQMH-Qualitätsgewichten vermischt;
- beide Komponenten besitzen dieselbe Bayer-/Kanal-/Origin-Semantik;
- die finale lineare Rekonstruktion ist `residual + background` in derselben
  Referenzdomäne;
- danach wird nur noch die bereits vorhandene photometrische Output-Skalierung
  angewendet; ein globaler `output_bg_*`-Offset entfällt im neuen DF-Map-Pfad.

#### 2.4.2 Background-Modell pro Frame

Vor dem Schreiben des normalisierten Frames wird pro Frame eine grobe
Background-Map im CFA-Raum geschätzt:

1. Background-Maske aus dem aktuellen Verfahren verwenden;
2. Sterne, gesättigte Pixel, Hotpixel und ungültige Samples ausschließen;
3. pro Bayer-Farbplane separat arbeiten (R, G1, G2, B); im RGB-Modus R, G, B;
4. ein festes Grid verwenden, zunächst `128 x 72` über den Originalframe;
5. pro Zelle einen robusten Median bestimmen;
6. leere Zellen bilinear aus mindestens drei gültigen Nachbarn innerhalb einer
   3x3- oder 5x5-Umgebung interpolieren; bei weniger gültigen Nachbarn als
   ungültig markieren;
7. keine Extrapolation ohne gültige Supportmarkierung; interpolierte Zellen
   erhalten ein separates Support-Bit (`interpolated`);
8. Map und Support gemeinsam speichern.

Für die Residualbildung werden G1- und G2-Positionen im CFA-Modus mit ihren
jeweiligen Ebenen subtrahiert; für die RGB-Ausgabe werden die gewarpten G1- und
G2-Background-Maps nach der Akkumulierung zu `G = (G1 + G2) / 2` kombiniert.

Das Modell muss für jeden Frame die lokale Information erhalten. Pro Farbebene
wird ein Scalar-Fallback (`B_r`, `B_g1`, `B_g2`, `B_b` bzw. `B_r/B_g/B_b`) nur
noch für eine vollständig leere Map verwendet und als Fallback im Artefakt
markiert.

#### 2.4.3 Eigenständiger Background-Model-Cache

Der Background-Model-Cache wird nicht als Sidecar im bestehenden
full-resolution Prewarp-Cache abgelegt. Der bestehende `DiskCacheFrameStore`
kennt feste Canvas-Dimensionen; ein kleines Grid hätte dort einen inkompatiblen
Datenvertrag. Stattdessen wird ein eigener `BackgroundModelGridStore`
eingeführt. Pro Frame und Farbebene wird eine eigene `.raw`/`.mask`-Datei im
Kanal-Unterverzeichnis abgelegt, damit einzelne Ebenen direkt gelesen und
geschrieben werden können.

```text
cache/background_models_cfa/R/{frame}.raw
cache/background_models_cfa/G1/{frame}.raw
cache/background_models_cfa/G2/{frame}.raw
cache/background_models_cfa/B/{frame}.raw
cache/background_models_cfa/R/{frame}.mask
cache/background_models_cfa/G1/{frame}.mask
cache/background_models_cfa/G2/{frame}.mask
cache/background_models_cfa/B/{frame}.mask
cache/background_models_rgb/R/{frame}.raw
cache/background_models_rgb/G/{frame}.raw
cache/background_models_rgb/B/{frame}.raw
cache/background_models_rgb/R/{frame}.mask
cache/background_models_rgb/G/{frame}.mask
cache/background_models_rgb/B/{frame}.mask
artifacts/background_model.json
```

Verbindliche interne Konstanten für die erste Implementierung:

```text
GRID_WIDTH  = 128
GRID_HEIGHT = 72
MAP_DTYPE   = float32
MASK_DTYPE  = uint8
AGGREGATION = two_pass_sigma_clipped_mean
```

Diese Werte werden nicht als Benutzerparameter angeboten. Sie gehören zur
`format_version` des Background-Cache und können nur mit einer neuen
Cache-Version geändert werden.

`background_model.json` enthält:

- Formatversion;
- Frame-Anzahl und Originaldimensionen;
- Frame-Grid-Dimensionen (`128 x 72`) und Datentyp;
- Canvas-Grid-Dimensionen (`canvas_grid_w x canvas_grid_h`) und Datentyp;
- Bayer-Pattern, Origin und Kanalreihenfolge;
- Referenzdomäne und `p_frame`-Semantik;
- Map-/Support-/Fallback-Zähler;
- Aggregationsverfahren;
- Konfigurations- und Eingabeartefakt-Hash;
- Vollständigkeitsmarker.

Die Residual-Q-Maps bleiben vollständig vom Background-Cache getrennt.


#### 2.4.4 Speicher-, Laufzeit- und OOM-Anforderungen

Das Background-Modell darf nicht die speicher- und laufzeitbasierten
Infrastrukturgrenzen der anderen Phasen verletzen:

- **Grid statt Voller Auflösung:**
  - Die per-Frame Background-Map wird fest als `128 x 72`-Grid gespeichert.
  - CFA speichert vier Float32-Ebenen (R, G1, G2, B), RGB drei. Bei 645 Frames
    sind das im CFA-Modus etwa 113 MiB und im RGB-Modus etwa 85 MiB Rohdaten;
    uint8-Supportmasken (eine pro Ebene) und Metadaten kommen hinzu. Die
    Planung rechnet mit einem Diskbedarf von etwa 120 MiB (CFA) bzw. 100 MiB
    (RGB), nicht mit einstelligen Megabytes.
  - Eine full-resolution Map ist für die Speicherung nicht nötig; beim Prewarp
    wird das Frame-Grid temporär auf volle Frame-Auflösung upgesampelt, mit
    `apply_global_warp` auf den Canvas gewarpt und auf das Canvas-Grid
    downgesampelt. Der temporäre Full-Res-Puffer wird pro Frame wieder
    freigegeben.
  - Der Map-Cache liegt auf Disk; die Gesamtgröße wird vor dem Run berechnet und
    im Artefakt protokolliert.

- **Keine Doppelhaltung im Speicher:**
  - Residuum- und Background-Map dürfen nicht gleichzeitig als volle
    Matrizen im Arbeitsspeicher gehalten werden, wenn beide für denselben
    Frame geladen werden.
  - Die Background-Map wird nur beim initialen Schätzen und beim finalen
    Zusammenführen vollständig benötigt; während AQMH bleibt sie auf dem
    Datenträger.

- **Streaming- und Chunking-Verhalten wie Prewarp:**
  - Map-Schätzung, Prewarp und Akkumulation werden Frame für Frame oder in
    kleinen Batches abgearbeitet.
  - Canvas-übergreifende Akkumulierung verwendet denselben Chunking-Mechanismus
    wie `reconstruct_aqmh` (Zeilen-Slabs), um große Matrizen zu vermeiden.

- **OOM-/Budget-Vertrag:**
  - Die persistente Map-Größe ist durch `(frames * grid_h * grid_w * channels)`
    fest berechenbar; `channels` ist 4 für CFA (R, G1, G2, B) und 3 für RGB.
    Temporäre Upsampling-/Prewarp-Puffer werden separat berechnet.
  - Die Map-Speicherung hängt nicht von der Frame-Größe ab; die Verarbeitung
    selbst darf wegen Upsampling und Warp nicht als framegrößenunabhängig
    bezeichnet werden.
  - Bei unzureichendem Budget wird vor dem Run mit `background_model_oom`
    abgebrochen, nicht stillschweigend auf den Scalar zurückgefallen.

- **GPU-Backend-Beschränkungen:**
  - Die Map-Akkumulation läuft bevorzugt auf der CPU oder in einem separaten,
    kleinen GPU-Kernel. Sie darf nicht denselben Speicherblock wie die
    AQMH-Rekonstruktion beanspruchen.
  - CUDA/OpenCL-Varianten müssen den Cache explizit host-seitig halten oder
    in Chunks auf das Gerät kopieren, nicht im Gesamtbild.

- **Laufzeitbudget:**
  - Map-Schätzung pro Frame darf nicht langsamer sein als die bestehende
    Normalisierung (reines Median/Sigma-Clip über Masken).
  - Background-Prewarp darf nicht langsamer sein als der Residuum-Prewarp,
    da dieselbe Transformationspipeline verwendet wird.

#### 2.4.5 Registration und Akkumulation

- die akkumulierte Canvas-Background-Map wird ebenfalls als Grid gehalten, nicht
  als full-resolution Matrix. Die Canvas-Gridgröße leitet sich aus der
  ursprünglichen Zellgröße ab: `canvas_grid_w = ceil(canvas_w / cell_w)`,
  `canvas_grid_h = ceil(canvas_h / cell_h)` mit
  `cell_w = frame_w / GRID_WIDTH`, `cell_h = frame_h / GRID_HEIGHT`. Am Ende
  der Akkumulierung wird das Canvas-Grid auf die volle Canvas-Auflösung
  upgesampelt;
- pro Frame wird die Frame-Background-Map auf die volle Frame-Auflösung
  upgesampelt, mit derselben Frame-Transform, Canvas-Größe, Interpolation und
  Offsetlogik wie die Residualdaten gewarpt, und anschließend auf das
  Canvas-Grid downgesampelt. Dabei werden Helligkeitswerte bilinear und der
  Support separat und konservativ (Nearest-Neighbor oder Schwellwert
  >0.5 gültiger Pixel pro Zielzelle) behandelt;
- bei RGB-DF werden die drei Background-Kanäle separat gewarpt;
- bei CFA-DF werden die vier Background-Ebenen (R, G1, G2, B) separat gewarpt;
  für die RGB-Ausgabe werden G1 und G2 auf dem Canvas-Grid zu
  `G = (G1 + G2) / 2` kombiniert;
- Background-Support wird mitgeführt und nicht als Helligkeitswert interpoliert;
- die gemeinsame Map wird außerhalb der AQMH-Signalgewichtung akkumuliert;
- pro Canvas-Zelle wird zunächst ein robuster Mittelwert und eine Streuung über
  gültige registrierte Maps bestimmt;
- in einem zweiten Durchlauf werden Werte außerhalb von `median ± 3*MAD`
  verworfen und die verbleibenden Werte gemittelt;
- ein reiner Median-Stack wird nicht verwendet, weil die additive
  Hintergrundkomponente dem gewichteten Mittelwert folgen soll; ein Median würde
  den zeitlichen Mittelwert der Frame-Backgrounds durch einen einzelnen
  robusten Wert ersetzen;
- zusätzlich werden Coverage-, Sample-Count- und Support-Maps geschrieben;
- Bereiche ohne mindestens 50 % der erwarteten gültigen Framebeiträge bleiben
  ungültig und werden nicht als Nullhintergrund ausgegeben.

#### 2.4.6 Reconstruction und Output

1. AQMH rekonstruiert ausschließlich das Residuum.
2. Die Background-Map wird separat auf dem RGB-Canvas akkumuliert.
3. Vor der Output-Skalierung werden Residuum und Background-Map in derselben
   Referenzdomäne addiert. Für den DF-Map-Pfad gilt:
   `output = (residual_reference + background_map_canvas) * output_scale + output_pedestal`.
   Der bisherige globale `output_bg_*`-Offset entfällt.
4. Der Output-Scaler darf danach nur noch die gemeinsame photometrische Skala
   anwenden.
5. Ein globaler `output_bg_*`-Offset darf im DF-Map-Modus nicht zusätzlich
   addiert werden. **Aktueller Code verletzt diese Regel noch**: sowohl
   `runner_pipeline.cpp:5568-5573` als auch `runner_phase_post_stack_output.cpp:184-189`
   addieren `output_bg_r/g/b` bzw. `scaling.bg_r/g/b` als globalen Scalar.
   Die Code-Kommentare sind mittlerweile als `TODO(bg-model)` markiert und
   machen korrekt deutlich, dass dies ein Platzhalter ist. Diese Regel wird
   erst mit Stufe A/B erfüllt.
6. Luminanz wird erst nach der kanalweisen Zusammenführung aus R/G/B gebildet.
7. Stretch, PCC und HMS arbeiten erst auf dem bereits dynamiktreuen linearen
   RGB-Output.

#### 2.4.7 Abbruch- und Fallbackregeln

Ein Run darf nicht als dynamiktreu gelten, wenn:

- Background-Map-Support unter dem Mindestwert liegt;
- eine Frame-Map in mehr als 1 % ihrer Zellen nur durch Scalar-Fallback
  abgedeckt werden kann (keine gültigen Nachbarn für Interpolation);
- mehr als 1 % der Canvas-`analysis_valid_mask`-Fläche ausschließlich durch
  Scalar-Fallback-Frames oder ohne Framebeitrag abgedeckt ist;
- Map- und Residualdimensionen nicht übereinstimmen;
- die Map nicht resumierbar ist;
- die Background-Rekonstruktion außerhalb der Supportmaske Werte erzeugt.

Ein Scalar-Fallback ist pro Frame und Farbebene erlaubt, wenn diese Ebene
keinerlei gültige Hintergrundsamples liefert. Der Frame wird in
`background_model.json` als `scalar_fallback` markiert. Seine Background-Ebene
wird entweder nicht in die Canvas-Map eingeflochten oder nur mit dem Scalar an
den Positionen seines gültigen Footprints eingetragen. Der
Scalar-Fallback-Anteil auf dem Canvas darf die oben genannte 1 %-Grenze nicht
überschreiten.

Bei fehlender Map muss der Run entweder kontrolliert abbrechen oder explizit als
`background_model_fallback_scalar` markiert werden. Ein stiller Rückfall auf
die bisherige globale Median-Wiederaddition ist nicht zulässig.

---

## 3. Effekt 2: Nullbereiche in den Ecken (Canvas-/Overlap-Problem)

### 3.1 Symptom

| Region | DF R Nulls | stacking program R Nulls | DF G Nulls | stacking program G Nulls |
|--------|-----------|---------------|-----------|---------------|
| top-left | 4.7% | 21.2% | 6.4% | 31.2% |
| top-right | 5.3% | 12.2% | 6.9% | 10.4% |
| bot-left | **53.8%** | 1.2% | **56.9%** | 1.6% |
| bot-right | 19.1% | 4.5% | 18.1% | 4.4% |

### 3.2 Ursache

- Durch Feldrotation wurde der Canvas von 3840x2160 auf 3924x2310 erweitert.
- `common_overlap_required_fraction: 1.0` verlangt, dass **alle 645 Frames**
  einen Pixel abdecken.
- In den erweiterten Ecken haben nicht alle Frames Abdeckung → Pixel werden
  auf null gesetzt.
- Overall sind 15.1% des Canvas null (`common_fraction: 0.849`), lokal in
  den Ecken bis zu 57%.

### 3.3 Trennung von Effekt 1

Dies ist **kein** Background-Map-Problem. Die Nullbereiche entstehen durch
die Support-Maske, nicht durch die Hintergrund-Subtraktion. Ein
Background-Model-Cache würde die Nullbereiche nicht füllen - er würde nur
die Hintergrunddynamik in den **gültigen** Pixeln wiederherstellen.

### 3.4 Lösung

Die `common_valid_mask` ist bereits das richtige Werkzeug, um den streng
gegenseitigen Überlappungsbereich zu markieren. Für die Analyse wird sie durch
eine zusätzliche `analysis_valid_mask` ergänzt, die aus der tatsächlichen
Coverage-Map abgeleitet wird.

- `common_overlap_required_fraction` bleibt vorerst bei 1.0. Er senkt sich
  **nicht** pauschal auf 0.85 oder 0.90. Eine Senkung ist erst nach einer
  quantitativen Coverage-/Photometrie-Analyse und einem stabilen
  Hintergrundvergleich in einer späteren Stufe zulässig.
- Die Coverage-Map führt pro Pixel, wie viele Frames dort gültige Daten
  liefern. Aus ihr wird die `analysis_valid_mask` gebildet: Pixel mit
  `coverage >= ceil(0.5 * max_coverage)` gelten als analysierbar.
- Die `common_valid_mask` (coverage == max_coverage) bleibt für den kanonischen
  Output maßgeblich.
- Maskierte Pixel dürfen nicht als physikalischer Wert 0 in den Output
  geschrieben werden. Sie erhalten `NaN` oder werden über eine separate
  Support-/Coverage-Ebene eindeutig als ungültig markiert.
- Für gültige Analysebereiche gilt ein separater Mindest-Support; der
  Analyse-Crop (`analysis_valid_mask`) darf nicht mit der visuellen
  Canvas-Ausdehnung verwechselt werden.
- Ein `union`-Modus ist nicht erforderlich, solange Coverage und Support
  getrennt ausgewertet werden.

---

## 4. Effekt 3: Uniform-Control aus R-Kanal, Validierung auf Luminanz

### 4.1 Bug

**Datei:** `runner_phase_aqmh_reconstruction.cpp`, Zeilen 543-555

Im DF-Pfad wird `aqmh_recon` nach dem R-Kanal (ch==0) zugewiesen und später
nur `aqmh_recon.output` durch die Luminanz ersetzt. Die
`uniform_control_output` stammt weiterhin vom R-Kanal:

```cpp
if (ch == 0) {
    aqmh_recon = ch_recon;  // uniform_control_output = R-Kanal!
}
// Später:
aqmh_recon.output = luma;  // 0.25R + 0.5G + 0.25B
// aber aqmh_recon.uniform_control_output bleibt R-Kanal
```

Die Validierung vergleicht dann **Luminanz gegen R-Kanal-Control**. Das ist
inkonsistent.

### 4.2 Lösung

```cpp
// Nach dem DF-Loop, vor der Validierung:
if (df_R.size() > 0 && df_G.size() > 0 && df_B.size() > 0) {
    // Luminanz-Control aus allen drei Kanal-Controls bilden:
    Matrix2Df control_luma(canvas_height, canvas_width);
    for (int i = 0; i < canvas_height * canvas_width; ++i) {
        control_luma.data()[i] =
            0.25f * df_control_R.data()[i] +
            0.50f * df_control_G.data()[i] +
            0.25f * df_control_B.data()[i];
    }
    aqmh_recon.uniform_control_output = control_luma;
}
```

Dafür müssen die `uniform_control_output`-Werte aller drei Kanäle
gespeichert werden (nur R wird derzeit übernommen).

### 4.3 Trennung von Effekt 1 und 2

Dieser Bug betrifft die **Validierungsentscheidungen** (welche
Post-Processing-Kandidaten angewendet werden), nicht direkt die
Hintergrunddynamik oder die Nullbereiche. Er muss separat korrigiert werden,
damit die Validierung Luminanz gegen Luminanz vergleicht.

---

## 5. Weitere Befunde (sekundär)

### 5.1 Post-Processing nur auf Luminanz, nicht auf RGB

**Datei:** `runner_phase_aqmh_reconstruction.cpp`, Zeilen 543-555 und 565-1045

Die Post-Processing-Kandidaten (Low-Frequency Neutralization,
Structure-Masked-Detail, Star-Core Sharpening) werden auf die Luminanz
angewendet, aber die RGB-Kanäle (`df_output_R/G/B`) werden nicht
aktualisiert. Wenn ein Kandidat angewendet würde, wäre die Luminanz
modifiziert, aber die RGB-Output-Dateien würden die rohe Kanal-Rekonstruktion
enthalten.

### 5.2 Per-Channel Valid-Masks werden gebaut, aber nicht verwendet

**Datei:** `runner_phase_aqmh_reconstruction.cpp`, Zeilen 587-592

Die `df_valid_mask_R/G/B` werden berechnet, aber im Output nie angewendet.
Pixel mit `weight_sum == 0` erscheinen als 0, ohne als ungültig markiert zu
sein.

**Fix (Haupt-Pipeline):** In `runner_pipeline.cpp` werden die Masken jetzt
vor dem Schreiben von `reconstructed_R/G/B.fit` angewendet. Ungültige Pixel
erhalten `NaN` statt `0`. Die `write_output_rgb_snapshot`-Kopien und
`stacked_rgb_solve.fits` übernehmen die NaN-Markierung; der u32-Stretch
konvertiert `NaN` zu `0`, was für Präsentationsoutputs akzeptabel ist.

**Offen:** Resume-Pfad (`runner_resume.cpp` →
`runner_phase_post_stack_output.cpp`) führt die Masken noch nicht, weil
`runner_phase_post_stack_output.cpp` keine per-Kanal-Valid-Masks entgegennimmt
und der Resume-Call sie nicht übergibt. Der `AqmhReconstructionPhaseResult` ist
im Resume-Pfad vollständig vorhanden. Lösung in Stufe C: den Ausgabe-Helper um
Maskenparameter ergänzen und `runner_resume.cpp` so anpassen, dass die Masken
analog zur Haupt-Pipeline vor dem Schreiben von `reconstructed_R/G/B.fit`
angewendet werden. Der Background-Model-Cache-Vertrag muss zudem im Resume-Pfad
validiert werden.

### 5.3 Structure-Masked-Detail-Kandidat abgelehnt

Der Structure-Masked-Detail-Kandidat hat **bessere** Werte als Control und
Raw-AQMH (Background-RMS -0.106, FWHM -0.192, Seam -0.044), wurde aber durch
den Raw-Guard abgelehnt (`tail11_abs_regression: 0.121 > 0.10`).

**Achtung:** Diese Metriken sind **intern** (vs. Uniform-Control) und sagen
nichts über den Vergleich mit stacking program aus. Die Ablehnung ist ein
Validierungs-Entscheidungsproblem, kein direkter Hintergrund-Bug.

### 5.4 Stretch mit gemeinsamem Floor

`stretch_rgb_to_u32_linear_from_zero_inplace` verwendet einen gemeinsamen
Floor/Ceiling für alle drei Kanäle. Der B-Kanal hat nach dem Stretch 71.7%
Nullen (vs. 56.9% linear). Per-Channel-Stretch würde das B-Kanal-Clipping
verhindern, löst aber nicht das Hintergrund-Dynamik-Problem.

---

## 6. Implementierungsplan

Die Umsetzung erfolgt in vier voneinander testbaren Stufen. Keine Stufe darf
sich auf einen visuellen Vergleich allein verlassen.

### 6.1 Stufe A: Background-Model-Cache

**Neue Komponenten:**

- `BackgroundModelGrid` mit festem `128 x 72`-Grid, vier Ebenen im CFA-Modus
  (R, G1, G2, B) und drei Ebenen im RGB-Modus, jeweils mit eigener
  Supportmaske;
- `BackgroundModelGridStore` für Store/Load/Validate im separaten
  `cache/background_models_*`-Cache (pro Kanal eigenes Unterverzeichnis);
- Background-Map-Schätzung aus dem kalibrierten Rohframe vor der
  Residual-Normalisierung;
- JSON-Metadaten in `artifacts/background_model.json` mit Formatversion
  (`format_version: 1`), Gridgröße, Einheiten, Aggregation, Budget, einem
  Cache-Content-Hash (SHA-256 über Konfiguration, Eingabeartefakte und
  Cache-Pfade) und einem Vollständigkeitsmarker;
- RGB-Background-Prewarp mit identischer Transformlogik;
- Resume-Validierung: fehlender, inkompatibler oder hash-mismatchierter Cache
  führt zu kontrolliertem Abbruch oder Recompute.

**Entscheidung:** Es werden keine neuen Benutzerparameter eingeführt. Gridgröße,
Aggregation, Fallbackgrenze und Cacheformat sind feste interne Konstanten der
implementierten Cacheversion. Eine Änderung dieser Werte erzeugt eine neue
Cacheversion, nicht eine neue Konfigurationsoption.

**Erste Implementierungsorte:**

- `runner_phase_metrics.cpp`: Map-Schätzung und Residualbildung;
- `runner_phase_metrics.hpp`: Kontext für Background-Model-Cache
  (Grid-Dimensionen, Kanalanzahl, Formatversion, Content-Hash);
- `runner_phase_registration.cpp`: RGB-Background-Prewarp;
- `runner_shared.hpp/.cpp`: eigenständige `BackgroundModelGridStore`-Klasse;
- `runner_resume.cpp`: Grid-Validierung und Resume-Vertrag;
- `runner_phase_post_stack_output.cpp` und `runner_pipeline.cpp`:
  globale `output_bg_*`-Wiederaddition im DF-Map-Modus durch räumliche
  Background-Map-Addition ersetzen;
- keine Änderungen an Config-Structs, JSON/YAML-Schemata oder Benutzerprofilen;
  die festen Konstanten gehören zur Cache-Formatversion und werden im
  `background_model.json` protokolliert.

Der bestehende `background_model`-Parameter in der PCC-Config ist **nicht**
verwandt: Er steuert die photometrische Hintergrundmodellierung für Color
Correction, nicht die Normalisierungs-Background-Map.

**Akzeptanztests:**

- synthetischer Gradient plus additive Frame-Offsets;
- Bayer-Pattern GBRG und mindestens ein weiteres Pattern;
- Map-Roundtrip mit identischer Dimension und endlichen Werten;
- Gradient nach Normalisierung und Wiederaddition innerhalb definierter
  Toleranz erhalten;
- leerer Gridbereich wird nicht als gültiger Nullwert gespeichert;
- beschädigter oder unvollständiger Cache wird abgelehnt;
- `background_model.json` enthält die feste Cacheversion, Gridgröße, Einheiten,
  Aggregation, Support- und Fallbackzähler;
- Resume erkennt fehlende oder inkompatible Background-Caches eindeutig;
- maximaler persistenter Map-Bedarf ist vor dem Start berechenbar und wird im
  Artefakt protokolliert;
- OOM-Test mit dem bestehenden Memory-Budget und temporären Upsampling-/Prewarp-
  Puffern;
- Background-Map liegt im Grid; kein full-res Cache pro Frame;
- synthetischer Gradient-/Offset-Test besteht mit einer festgelegten relativen
  Fehlergrenze von höchstens 2 % auf gültigen Gridzellen;
- die Map-Schätzung verarbeitet keine Stern-/Sättigungszellen als gültige
  Backgroundwerte;
- es existiert kein stiller Scalar-Fallback oberhalb von 1 % der gültigen
  Zellen.

### 6.2 Stufe B: Getrennte Residual-/Background-Reconstruction

- Q-Maps nur auf dem Residualpfad berechnen;
- Background-Maps unabhängig von AQMH-Qualitätsgewichten akkumulieren;
- gemeinsame R/G/B-Background-Maps und Supportkarten erzeugen;
- Residuum und Map erst auf dem gemeinsamen Canvas addieren;
- `output_bg_*` im DF-Map-Modus nicht zusätzlich anwenden;
- linearen Output ohne Stretch als primäres Analyseartefakt schreiben.

**Akzeptanzkriterien:**

- Background-Perzentile und MAD liegen zwischen Rohreferenz und erwarteter
  Stacking-Varianz;
- räumliche Background-Map bleibt gegenüber dem Inputgradienten erhalten;
- keine deutliche künstliche Kompression der unteren 30 %;
- keine zusätzliche Background-Regression durch die Wiederaddition;
- R/G/B und Luminanz verwenden dieselbe Supportsemantik.

### 6.3 Stufe C: Support, Canvas und Validierung

**Implementierungsorte:**

- `runner_phase_aqmh_reconstruction.cpp`: pro Kanal `uniform_control_R/G/B`
  erzeugen, daraus `uniform_control_luma = 0.25R + 0.5G + 0.25B` bilden und
  in `aqmh_recon.uniform_control_output` schreiben;
- `AqmhReconstructionPhaseResult` (oder `runner_phase_aqmh_reconstruction.hpp`):
  Speicher für `df_control_R/G/B` erweitern;
- `runner_phase_post_stack_output.cpp`: `df_valid_mask_R/G/B` als Parameter
  akzeptieren und anwenden, damit Resume und Haupt-Pipeline konsistent NaN
  schreiben;
- `runner_resume.cpp`: Masken an `runner_phase_post_stack_output.cpp` übergeben
  und Background-Model-Cache-Vertrag validieren;
- `runner_phase_registration.cpp`: Coverage-Map und `analysis_valid_mask` aus
  der Frame-Abdeckung erzeugen.

**Regeln:**

- `common_overlap_required_fraction` bleibt bei 1.0 für den kanonischen Output;
  eine Senkung ist erst nach einer quantitativen Coverage-/Photometrie-Analyse
  in einer späteren Stufe zulässig;
- gemeinsame Abdeckung, Residualsupport und Backgroundsupport getrennt führen;
- aus der Coverage-Map wird `analysis_valid_mask` mit
  `coverage >= ceil(0.5 * max_coverage)` gebildet;
- ungültige Pixel als `NaN`/ungültige Maske behandeln, nicht als physikalische
  Nullwerte;
- `uniform_control_luma = 0.25R + 0.5G + 0.25B` ist die einzige Control-Ebene
  für Luminanz-Validierung;
- Raw-, Control- und Kandidatenmetriken auf identischen Luminanz- und
  Supportmasken berechnen;
- Nullpixelquoten pro Region und Kanal als harte Diagnose ausgeben.

**Akzeptanzkriterien:**

- keine verdeckten Nullbereiche im gültigen Vergleichsbereich;
- Null-/NaN-Anteil getrennt von Background-MAD ausweisen;
- Control-Gates vergleichen Luminanz gegen Luminanz;
- per-channel Valid-Masks werden im finalen RGB-Output (Haupt-Pipeline und
  Resume) tatsächlich angewendet.

### 6.4 Stufe D: Post-Processing und Vergleich

Erst wenn A-C bestanden sind:

- Post-Processing-Kandidaten kanalweise oder konsistent auf Luminanz und RGB
  anwenden;
- Raw-Guard, Control-Gate und externe Vergleichsmetriken getrennt ausweisen;
- Stretch in eine eigene Präsentationsdatei schreiben; die lineare Datei bleibt
  unverändert;
- HMS nur als Präsentationsoutput bewerten;
- FWHM median/p90, Peak/Flux, Background-Perzentile, MAD, Gradientkarten und
  Nullpixelquoten gegen dieselbe Referenzpipeline vergleichen.

Ein Raw-Guard darf nicht einfach gelockert werden, bevor Supportmasken,
Luminanz-Control und Background-Model korrekt sind. `tail11_abs` ist erst dann
als Entscheidungsgrundlage belastbar.

---

## 7. Fazit

DF-AQMH ist **nicht besser** als das stacking program-Ergebnis. Die Schärfe
(Hochfrequenz-Energie) ist vergleichbar, aber:

1. **Sternform:** Per-Kanal-FWHM liegt 1.04-3.17x über stacking program; die
   Luminanz-Median-Schärfe ist nur ~5 % schlechter, der p90-Schwanz aber
   deutlich aufgeweitet (~26 %). Das ist ein echter Qualitätsbefund,
   unabhängig von den internen Control-Gates.
2. **Hintergrund-Dynamik:** Die Pipeline bewahrt die additive Background-
   Komponente nicht getrennt vom AQMH-Residuum. Die globale Median-
   Wiederaddition ist kein ausreichendes physikalisches Rekonstruktionsmodell.
3. **Nullbereiche:** 15% des Canvas (lokal bis 57%) sind null wegen
   `common_overlap_required_fraction=1.0` und Canvas-Erweiterung.
4. **Validierung:** Die interne FWHM-Verbesserung ist nur DF gegen
   Uniform-Control; sie ist kein externer Nachweis gegen stacking program.

Der Report beschreibt damit eine Implementierungsgrundlage, nicht eine bereits
umgesetzte Lösung. Aktuell existieren noch kein Background-Model-Cache, keine
räumliche Background-Akkumulation und kein vollständiger Resume-Vertrag dafür.
Die Umsetzung muss in den Stufen A-D erfolgen. Ein einzelner Output-Scaler,
eine nachträgliche BGE-Phase oder ein direktes Zurückaddieren der Frame-Scalars
sind ausdrücklich keine vollständige Lösung.

**Code-Kommentare aktualisiert:** Die ursprünglich irreführenden Kommentare in
`runner_phase_post_stack_output.cpp:181-183` und `runner_pipeline.cpp:5565-5567`
wurden durch `TODO(bg-model)`-Marker ersetzt. Sie machen nun korrekt deutlich,
dass die globale Scalar-Wiederaddition ein Platzhalter ist, der durch den
Background-Model-Cache aus Stufe A/B ersetzt werden muss.

Die erste produktionsfähige Abnahme ist erst erlaubt, wenn:

- das synthetische Gradient-/Offset-Regressionstestbild die Background-Map
  innerhalb der definierten Toleranz rekonstruiert;
- Residual- und Backgroundpfad getrennt validiert sind;
- Null-/NaN-Support von Background-MAD getrennt ausgewiesen wird;
- Luminanz-Control aus R/G/B gebildet wird;
- lineare Outputs ohne Stretch reproduzierbar gegen die stacking-program-
  Referenz verglichen werden;
- Resume und Cachevalidierung den vollständigen Background-Model-Vertrag
  erfüllen.

---

## 8. Anhang: Offene Fragen und Einschränkungen

1. **Per-Kanal-FWHM und G-Kanal-Ausreisser:**
   Die per-Kanal-FWHM-Werte basieren auf einer einzelnen Messmethode. Der
   G-Kanal-Ausreisser (3.17x) sollte mit einer unabhängigen
   positionsgematchten Luminanz-Methode und ggf. mit reinem
   `reconstructed_G.fit` bestätigt werden, bevor daraus Designentscheidungen
   abgeleitet werden.

2. **Q-Map-Verhalten im `shared_luma`-Modus:**
   Die DF-Rekonstruktion nutzt `rgb_q_map_mode: shared_luma`. Für die
   per-Kanal-Ausgaben (R/G/B) werden luma-basierte Q-Maps wiederverwendet. Ob
   dies den G- oder B-Kanal systematisch beeinträchtigt, ist eine offene
   Frage, die getrennt untersucht werden sollte.

3. **Background-Model als langfristige Architektur:**
   Der Background-Model-Cache (Stufe A/B) ist kein schneller Fix, sondern eine
   neue Pipeline-Komponente. Bis er verfügbar ist, bleiben die linearen Outputs
   `reconstructed_R/G/B.fit` dynamikverarmt.

4. **Unabhängige Luminanz-Metriken (Kreuzverifikation):**
   Eine positionsgematchte Analyse auf Luminanz ergab (Stand dieses Reports):
   - Median-FWHM: 23.26" (DF) vs. 22.19" (stacking program), ~5 % schlechter;
   - p90-FWHM: 29.42" (DF) vs. 23.26" (stacking program), ~26 % schlechter;
   - Peak/Flux: 0.066 (DF) vs. 0.081 (stacking program);
   - Hintergrund p1-p90 in `reconstructed_R.fit`: ~4 ADU (0..4095).
   Diese Werte sind mit der gleichen Referenzpipeline zu reproduzieren, wenn
   sich die Sternerkennung oder Hintergrundmodellierung ändert.
