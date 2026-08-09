# REGISTRATION — Globale Multi-Anchor-Konsensregistrierung + Pre-Warping

> **C++ Implementierung:** `global_registration.cpp`, `registration.cpp`, `runner_phase_registration.cpp`
> **Phase-Enum:** `Phase::REGISTRATION` (gefolgt von eigener `Phase::PREWARP`)

## Übersicht

Die Registrierung richtet alle Frames geometrisch auf ein gemeinsames Referenzsystem aus. Der aktuelle C++-Runner kombiniert eine **6-stufige Einzelbild-Registrierungskaskade** mit einer **festen Multi-Anchor-Konsensstrategie** und optionaler Astrometrie. Das ist speziell fuer lange Alt/Az-Sequenzen mit Feldrotation wichtig, bei denen ein einziger spaeter Referenzframe fuer fruehe Frames oft geometrisch zu weit entfernt ist. Kein Warp wird aus einem vorhergehenden Frame oder aus einem rekursiv promovierten Anchor abgeleitet. Anschliessend werden alle Frames in der **eigenen Pipeline-Phase `PREWARP`** vollaufgeloest vorgewrapt, bevor lokale Tile-Metriken berechnet werden.

**Kernprinzip:** Keine Frame-Selektion. Jeder Frame bleibt im Datenfluss; die Provenienz wird ueber `source`, `chain_depth` und Registration-Telemetrie dokumentiert.

`v3.3.9` ergänzt hier zwei normative Punkte:

1. Vor dem CFA-Warp ist optional eine **deterministische CFA-Defektpixel-Unterdrückung** zulässig, solange CFA-Phase und Sample-Geometrie erhalten bleiben.
2. Die Identity-Akzeptanz für den Referenzframe und für nahezu perfekt ausgerichtete Frames ist ein **korrekter Sonderfall**, kein Registrierungsfehler.

```text
┌──────────────────────────────────────────────────────────┐
│  Fuer jeden Frame f != master_ref:                       │
│                                                          │
│  1. Waehle hochwertige, zeitlich verteilte feste Anker   │
│  2. Loese jeden festen Anker unabhaengig gegen Master    │
│  3. Teste bis zu 5 naechste feste Anker plus Master      │
│  4. Validiere jeden Warp im Master-System (NCC/Overlap)  │
│  5. Akzeptiere Transform-Konsens oder einen eindeutig    │
│     ueberlegenen starken Einzelkandidaten                │
│  6. Sonst: Master-ECC -> Modell -> optionale ASTAP-Pruefung│
│  7. Pre-warp ganzes Bild (CFA-aware bei OSC)             │
└──────────────────────────────────────────────────────────┘
```

## 1. Referenz-Frame-Auswahl

Der Runner verwendet nicht mehr nur einen einzelnen Qualitaets-Ref-Frame, sondern ein **Master-Anchor plus mehrere zeitlich verteilte Referenzanker**:

- Unter `120` Frames: `1` Anchor
- Ab `120` Frames: ungerade Anchor-Zahl mit etwa **1 Anchor pro 80 Frames**
- Formel im aktuellen Runner: `requested_anchor_count = ceil(N / 80)`, dann auf die naechste ungerade Zahl angehoben
- Obergrenze aktuell: `15` angeforderte Anchors
- Beispiele: `120 -> 3`, `240 -> 3`, `325 -> 5`, `1000 -> 13`
- Die Zielpositionen der Anchors werden ueber die Zeitachse verteilt; innerhalb jedes Segments wird der beste Frame nach `global_weights` bzw. `quality_score` gewaehlt.
- Der **Master-Anchor** ist der aktive Referenzframe, der unter den gewaehlten Anchors am naechsten zur Sequenzmitte liegt.
- Der Master-Anchor erhaelt Identity-Warp und `CC=1.0`.

Das verhindert, dass fruehe Frames in langen Alt/Az-Sessions direkt gegen einen viel spaeteren Referenzframe registriert werden muessen.

## 1.1 Multi-Anchor-Direktpass

Nach der Anchor-Auswahl laeuft die direkte Registrierung so:

- Aufnahmezeit-Gaps aus Dateinamen im Format `YYYYMMDD-HHMMSS...` trennen
  unabhaengige Serien. Fuer jedes erkannte Segment werden die Randframes und
  zusaetzliche lokale Quality-Anker erzwungen. Dadurch besitzt auch ein
  kombinierter Run wie `part1 + part2` feste Anker in jeder Aufnahmeserie.
- Angeforderte Anchor-Frames werden jeweils direkt gegen den unveraenderlichen
  Master-Referenzframe geloest und global per Overlap/NCC validiert. Kein Anchor
  erbt die Transformation eines anderen Anchors.
- Fuer jedes normale Frame werden deterministisch die bis zu **fuenf zeitlich
  naechsten festen Anchors plus der Master** getestet. Die
  Frame-zu-Anchor-Transformation wird mit dem unabhaengigen Anchor-zu-Master-Warp
  zusammengesetzt; anschliessend wird jeder Kandidat gegen dasselbe Masterbild
  mit maskiertem Overlap und globalem NCC bewertet.
- Primaer wird nur ein Kandidat mit mindestens einem zweiten uebereinstimmenden
  Kandidaten akzeptiert. Uebereinstimmung bedeutet im Registrierungs-Proxy:
  hoechstens `0.75 deg` Winkeldifferenz, `0.5 %` relative Skalendifferenz und
  hoechstens `3 px` Abstand an Bildmittelpunkt und allen vier Ecken. Dadurch
  bilden auch korrekt gefundene `180 deg`-Meridian-Flip-Warps eine gemeinsame
  Orientierungsfamilie, waehrend ein isolierter Halbturn-Alias keinen Konsens
  erhaelt.
- Ohne Zweierkonsens darf nur der global beste **starke und eindeutig
  ueberlegene** Kandidat passieren: NCC mindestens
  `max(registration.reject_cc_min_abs, 0.35)` und mindestens `0.03` ueber dem Runner-up. Existiert kein global gueltiger
  Runner-up, gilt die strengere absolute NCC-Schwelle `0.50`.
- Normale Frames werden nicht zu neuen Anchors promoted. Ein frueher
  Registrierungsfehler kann daher weder die Anchor-Menge noch spaetere Frames
  beeinflussen. Kandidaten ohne Konsens bzw. eindeutigen Sieger gehen
  `unresolved` in die frame-unabhaengige Rescue aus Abschnitt 1.2.
- `global_registration.json` protokolliert Strategie, exakte Schwellen sowie
  aggregierte Anzahlen fuer Versuche, global gueltige Kandidaten,
  Konsensannahmen, Einzelannahmen und Ablehnungen ohne Konsens.

## 1.2 Globale, frame-unabhaengige Rescue nach dem Direktpass

Die Quell- und Basisauswahl einer gueltigen `direct_global`-Transformation ist
nach dem Direktpass unveraenderlich: Sie wird weder durch einen zeitlichen
Nachbarn noch durch eine Frame-zu-Frame-Kette ersetzt. Die spaeteren,
frame-lokalen affinen und Smooth-Local-Refinements duerfen diesen Basis-Warp
weiterhin nur unter ihren Residual-, Geometrie-, NCC- und Overlap-Gates
verfeinern; sie erzeugen keine Abhaengigkeit zu anderen Frames. Damit kann eine
Fehlregistrierung nicht mehr als Anker fuer nachfolgende Frames wirken oder den
Stern-PSF ueber eine lange Kette verbreitern.

Fuer jedes noch ungeloeste Frame gelten stattdessen unabhaengige globale
Stufen:

1. `seeded_ecc_rescue`: Aus den urspruenglichen `direct_global`- und
   Referenz-Warps wird nur ein geometrischer Interpolations-Seed gebildet. Das
   Frame selbst wird anschliessend direkt gegen den unveraenderlichen
   Master-Referenzframe optimiert und per globalem Overlap/NCC validiert.
2. Die Ergebnisse dieser Stufe werden niemals als Seeds oder Stuetzpunkte fuer
   andere Frames bzw. das Feldrotationsmodell wiederverwendet; die Ausgabe ist
   deshalb unabhaengig von Iterationsrichtung und Frame-Reihenfolge.
3. `model_interpolated` / `model_blended`: globales Feldrotationsmodell als
   geometrischer Rueckfall fuer noch ungeloeste Frames. Nicht global plausible
   Vorhersagen bleiben `unresolved` und werden nicht als scheinbar gueltige
   Ketten weitergereicht.
4. `astrometric_rescue`: Wenn ASTAP konfiguriert und verfuegbar ist, werden
   weiterhin ungeloeste Frames sowie die synthetischen Modellwarps mit niedriger
   CC per Plate Solve geprueft. Eine erfolgreiche astrometrische Loesung ersetzt
   den schwachen Modellwarp; ohne ASTAP bleibt der konservative Modell- bzw.
   `unresolved`-Fallback erhalten.

Die historischen Provenienzen `sequential_refined`, `sequential_rescue`,
`temporal_rescue` und `local_reference_rescue` bleiben fuer die Lesbarkeit alter
Run-Artefakte bekannt, werden von der aktuellen Auswahlstrategie
`independent_global_consensus_v2` aber nicht mehr erzeugt. `global_registration.json`
weist die Strategie und `temporal_chain_transforms_enabled: false` explizit aus.

Vor der Modellvorhersage wird die zeitliche Transformations-Komponente weiterhin
nur als **Outlier-Diagnose** auf Nachbar-Kontinuitaet geprueft. Diese Pruefung
schaetzt oder ersetzt keinen Warp. Ein einzelner grosser Sprung kann damit
Aufnahmeserien trennen, ohne dass ein Nachbarframe zur geometrischen Referenz
eines anderen Frames wird.

## 2. Downsample für Registrierung

```cpp
Matrix2Df ref_reg = (detected_mode == ColorMode::OSC)
    ? image::cfa_green_proxy_downsample2x2(ref_full, detected_bayer_str)
    : registration::downsample2x2_mean(ref_full);
```

| Modus | Downsample-Methode | Faktor | Begründung |
|-------|-------------------|--------|------------|
| **OSC** | CFA Green-Proxy 2×2 | 2× | Nutzt nur G-Pixel, kein Farb-Crosstalk |
| **MONO** | Mean 2×2 | 2× | Einfaches Downsampling |

- **Speedup:** ~4× weniger Pixel für Registrierung
- `global_reg_scale` speichert den Skalierungsfaktor (full_height / reg_height)
- Translationen (tx, ty) werden nach Registrierung mit `global_reg_scale` auf Vollauflösung skaliert

## 3. Registrierungskaskade (6 Stufen)

Für jeden Frame wird die Kaskade sequentiell durchlaufen, bis eine Methode erfolgreich ist:

### Stufe 1: Triangle Star Matching (Primär)

```cpp
RegistrationResult rr = registration::triangle_star_matching(
    mov_p, ref_p, allow_rotation, star_topk, star_min_inliers, star_inlier_tol_px);
```

- **Rotationsinvariant**: Verwendet Dreiecke aus Sternpositionen (Astroalign-Stil)
- Bildet Dreiecke aus Top-30 Sternen, vergleicht Seitenverhältnisse (invariant gegenüber Rotation/Skalierung)
- RANSAC-Konsens über alle Sterne mit `star_min_inliers` und `star_inlier_tol_px`
- **Konfiguration:** `star_topk` (Top-K Sterne), `star_min_inliers`, `star_inlier_tol_px`
- **Ideal für:** Alt/Az-Montierungen mit Feldrotation, ≥6 Sterne
- **Keine Rotationslimits** (entfernt wegen Alt/Az nahe Pol mit >20° Rotation)

### Stufe 2: Trail Endpoint Matching (Star Trails)

```cpp
rr = registration::trail_endpoint_registration(
    mov_p, ref_p, allow_rotation, star_topk, star_min_inliers,
    star_inlier_tol_px, star_dist_bin_px);
```

- **Für Star Trails** bei Feldrotation (Alt/Az) oder langen Belichtungen
- **Morphologischer Top-Hat** (15×15 Ellipse) extrahiert helle dünne Strukturen (Trails)
- **Contour-Analyse** findet die am weitesten entfernten Punkte jedes Trails (= Endpunkte)
- **Brightness-weighted Centroid** verfeinert die Endpunkt-Position sub-pixel-genau
- Kombiniert Trail-Endpunkte + reguläre Sterne für robusteres Matching
- Pair-Distance Similarity Matching mit relaxierten Schwellenwerten:
  - `inlier_tol_px × 2` (doppelte Toleranz, da Endpunkte ungenauer als Sternzentren)
  - `min_inliers / 2` (weniger Inlier nötig, da weniger Punkte verfügbar)
- **Ideal für:** Wolken/Nebel mit wenigen Sternen (<20), Feldrotation

### Stufe 3: AKAZE Feature Matching

```cpp
rr = registration::feature_registration_similarity(mov_p, ref_p, allow_rotation);
```

- **Feature-basiert**: AKAZE Keypoints + Descriptor Matching + RANSAC
- **Rotationsinvariant**: Keine Rotationslimits
- Braucht mind. 8 Feature-Matches, behält Top 30%
- `estimateAffinePartial2D` mit RANSAC → Similarity-Transform
- **Fallback** wenn zu wenige Sterne (z.B. dichte Nebel, Wolken)

### Stufe 4: Robust Multi-Scale Phase+ECC (Gradient-robust)

```cpp
rr = registration::robust_phase_ecc(mov_p, ref_p, allow_rotation);
```

- **Laplacian-of-Gaussian Vorverarbeitung**: Entfernt niederfrequente Gradienten (Nebel, Wolken, Lichtverschmutzung) und bewahrt hochfrequente Strukturen (Sterne, Kanten)
- **3-Level Coarse-to-Fine Pyramide**: 4× → 2× → 1× Auflösung
  - Gröbstes Level: Phase-Correlation (Translation) + Log-Polar DFT (Rotation)
  - Jedes Level: ECC-Refinement (100–200 Iterationen)
- **Ideal für:** Starke Nebel/Wolken-Gradienten + große Rotation

### Stufe 5: Hybrid Phase-Correlation + ECC (Original)

```cpp
rr = registration::hybrid_phase_ecc(mov_p, ref_p, allow_rotation);
```

- **Phase-Correlation** für grobe Translation, dann **ECC-Refinement**
- Arbeitet auf Rohpixeln (ohne Gradient-Preprocessing)
- Akzeptiert nur wenn `correlation >= 0.15`
- Einfacher als Stufe 4, kann bei klaren Bildern ohne Gradienten besser konvergieren

### Stufe 6: Identity-Fallback

```cpp
out.warps_fullres[i] = identity_warp();
out.scores[i] = 0.0f;
out.success[i] = false;
```

- **v3-Regel:** Kein Frame wird ausgeschlossen
- CC=0 → niedrigstes effektives Gewicht, aber Frame bleibt in Pipeline
- In der Praxis: Frame mit Identity-Warp hat minimalen Einfluss auf Rekonstruktion
- Nur aktiv wenn `registration.fallback_to_identity: true` (Default)

## Kaskade — Entscheidungslogik

```
Sterne erkannt (≥6)?
├─ JA → Stufe 1 (Triangle) → Erfolg? → FERTIG
│       └─ NEIN → Stufe 2 (Trail Endpoints) → Erfolg? → FERTIG
│                 └─ NEIN → Stufe 3 (AKAZE) → Erfolg? → FERTIG
│                           └─ NEIN → Stufe 4 (Robust Phase+ECC) → Erfolg? → FERTIG
│                                     └─ NEIN → Stufe 5 (Hybrid Phase+ECC) → Erfolg? → FERTIG
│                                               └─ NEIN → Stufe 6 (Identity)
├─ NEIN (Star Trails) → Stufe 1 scheitert → Stufe 2 (Trail Endpoints) → ...
├─ NEIN (Nebel/Wolken) → Stufe 1–3 scheitern → Stufe 4 (Robust Phase+ECC) → ...
└─ NEIN (komplett leer) → Stufe 1–5 scheitern → Stufe 6 (Identity)
```

## 4. Warp-Skalierung

```cpp
WarpMatrix w_full = rr.warp;
w_full(0, 2) *= global_reg_scale;  // tx skalieren
w_full(1, 2) *= global_reg_scale;  // ty skalieren
```

Die Warp-Matrix wird auf halber Auflösung berechnet. Die Translationskomponenten (tx, ty) werden mit dem Skalierungsfaktor auf Vollauflösung hochskaliert. Die Rotations-/Affin-Komponenten (a00, a01, a10, a11) bleiben unverändert.

## 5. Pre-Warping (CFA-aware)

> Laufzeit-Sichtbarkeit: `phase_start(PREWARP)` / `phase_progress(PREWARP)` / `phase_end(PREWARP)`

`REGISTRATION` selbst ist CPU-only (Sternerkennung, Matching,
Transformationsschätzung und ECC). Die anschließende eigene Phase `PREWARP`
verwendet je nach `runtime_limits.acceleration_backend` CUDA oder OpenCL und
einen CUDA-Stream pro Worker. `gpu=no` während REGISTRATION ist daher erwartet.

**Kritischer Schritt** nach der Registrierung, vor der Tile-Extraktion:

Optional darf davor eine deterministische per-frame CFA cosmetic correction laufen, sofern:

- nur isolierte same-parity CFA-Ausreißer korrigiert werden,
- keine Demosaicing- oder Cross-Channel-Interpolation stattfindet,
- reale kompakte Bildstruktur durch einen Struktur-Guard geschützt bleibt.

```cpp
auto prewarp_worker = [&](int worker_index) {
  while (true) {
    const size_t fi = prewarp_next.fetch_add(1);
    if (fi >= frames.size()) break;
    auto pair = load_frame_normalized(fi);
    Matrix2Df img = std::move(pair.first);
    Matrix2Df warped;
    prewarp_ops.warp_affine_frame(
        std::move(img), global_frame_warps[fi], detected_mode,
        canvas_height, canvas_width, offset_x, offset_y, warped,
        &valid_mask, &has_data, prewarp_streams.get(worker_index));
    prewarped_frames.store(fi, warped);
  }
};
```

### Warum Pre-Warping?

**Problem:** Rotation-Warps auf kleine Tile-ROIs (z.B. 64×64) verursachen **CFA-Pattern-Korruption** — `warpAffine` braucht Quell-Pixel außerhalb des Tile-Rands, die nicht existieren. Das Ergebnis sind sichtbare farbige Rechtecke.

**Lösung:** Alle Frames werden **vor** der Tile-Extraktion auf **voller Bildauflösung** gewarpt.

| Modus | Warp-Methode | CFA-Sicherheit |
|-------|-------------|----------------|
| **MONO** | `AccelerationOps::warp_affine_frame()` | N/A |
| **OSC** | Vier getrennte CFA-Subebenen über `AccelerationOps` | ✓ Keine Bayer-Phasen-Mischung |

CUDA bündelt die vier OSC-Subebenen in einem Worker-Stream und synchronisiert
einmal pro Frame. OpenCL verwendet OpenCV `UMat`; CPU bleibt der Fallback.

`warp_cfa_mosaic_via_subplanes` zerlegt das CFA-Mosaik in 4 Subplanes (R, G1, G2, B), warpt jede separat und interleaved sie zurück.

### Canvas-Expansion und Offsets

Bei Feldrotation/Translation wird ein gemeinsamer Canvas aus der registrierten Bounding-Box erzeugt:

- `canvas_width`, `canvas_height` werden auf **gerade Dimensionen** gerundet (CFA/Subplane-kompatibel).
- `offset_x`, `offset_y` verschieben alle Frames in den positiven Canvas-Bereich.
- Alle globalen Warps werden um diesen Offset in Zielkoordinaten korrigiert.
- Für OSC werden Offsets paritätssicher behandelt (gerade Pixelraster), damit das Bayer-Muster über den Canvas hinweg konsistent bleibt.

Die resultierenden Canvas-Daten werden im PREWARP-Output (`canvas_width/height`, `tile_offset_x/y`) an Folgephasen übergeben und dort für Common-Overlap, Tile-Reconstruction und finale Skalierung verwendet.

## Konfigurationsparameter

| Parameter | Beschreibung | C++ Default |
|-----------|-------------|-------------|
| `registration.engine` | Primäre Engine | `triangle_star_matching` |
| `registration.transform_model` | Globales Warp-Modell | `similarity` |
| `registration.enable_star_pair_fallback` | Optionale Star-Pair-Zwischenstufe | `true` |
| `registration.allow_rotation` | Rotation erlauben (Alt/Az) | `true` |
| `registration.auto_engine` | Alt/Az-Override auf `triangle_star_matching+affine` | `true` |
| `registration.auto_engine_rotation_threshold_deg` | Trigger fuer Auto-Engine | `0.05` |
| `registration.star_topk` | Top-K Sterne fuer Matching | `150` |
| `registration.star_min_inliers` | Mindest-Inlier fuer Akzeptanz | `4` |
| `registration.star_inlier_tol_px` | Inlier-Toleranz in Pixel | `4.0` |
| `registration.star_dist_bin_px` | Distanz-Bin fuer Star Pairs | `5.0` |
| `registration.star_shift_radius_px` | Shift-Konsistenzradius auf Proxy | `200.0` |
| `registration.max_blind_chain_depth` | Maximale Blind-Chain-Tiefe (`0` = auto) | `0` |
| `registration.blind_chain_strong_anchor_cc` | CC-Schwelle fuer starke Chain-Anker | `0.08` |
| `registration.use_astrometry` | ASTAP-Rescue erlauben | `true` |
| `registration.enable_local_background_subtraction` | Lokale Hintergrundsubtraktion fuer Sterndetektion | `false` |
| `output.write_registered_frames` | Registrierte Frames speichern | `false` |

## Artifact: `global_registration.json`

```json
{
  "num_frames": 100,
  "scale": 2.0,
  "ref_frame": 42,
  "source": ["direct_global", "sequential_refined", "..."],
  "chain_depth": [0, 2, "..."],
  "cc": [1.0, 0.95, 0.87, ...],
  "warps": [
    {"a00": 1.0, "a01": 0.0, "tx": 0.0, "a10": 0.0, "a11": 1.0, "ty": 0.0},
    {"a00": 0.999, "a01": -0.012, "tx": 3.5, "a10": 0.012, "a11": 0.999, "ty": -1.2},
    ...
  ],
  "extra": {
    "ref_frame_strategy": "quality_segmented_multi_anchor",
    "requested_ref_frames": [5, 42, 88],
    "active_ref_frames": [18, 42, 63, 88],
    "reg_target_active_anchor_count": 5,
    "reg_promote_limit_per_round": 2,
    "reg_max_direct_anchor_rounds": 3,
    "reg_direct_anchor_rounds": 2,
    "reg_source_counts": {
      "direct_global": 61,
      "sequential_refined": 37,
      "reference": 1,
      "astrometric_rescue": 1
    }
  }
}
```

Wichtige aktuelle Felder:

- `source`: Provenienz pro Frame, z. B. `reference`, `direct_global`, `sequential_refined`, `sequential_rescue`, `astrometric_rescue`, `model_predicted`
- `chain_depth`: effektive Verkettungstiefe pro Frame
- `extra.requested_ref_frames`: angeforderte Quality-Anker
- `extra.active_ref_frames`: tatsaechlich aktive Anchors nach Promotion
- `extra.reg_target_active_anchor_count`: Zielgroesse fuer aktive Anchors nach N-Skalierung
- `extra.reg_promote_limit_per_round`: wie viele neue Anchors eine Promote-Runde maximal aktivieren darf
- `extra.reg_max_direct_anchor_rounds`: hartes Limit fuer wiederholte Direktpaesse
- `extra.reg_source_counts`: Summen je Registrierungsweg
- `extra.reg_direct_anchor_rounds`: wie oft nach Anchor-Promotion nochmals teure Direktpaesse gelaufen sind

## Fehlerbehandlung

| Situation | Verhalten |
|-----------|-----------|
| Ref-Frame leer | `global_reg_status = "error"` |
| Frame leer | Identity-Warp, CC=0 |
| Direkte und indirekte Registrierung scheitern komplett | Identity-Warp oder modellierter Rueckfall; Frame bleibt dokumentiert |
| Exception in Registrierung | Gesamte Phase → "error", Pipeline weiter |
| Star Trails (Feldrotation) | Stufe 2 (Trail Endpoints) übernimmt |
| Nebel/Wolken (keine Sterne) | Stufe 4 (Robust Phase+ECC) übernimmt |

## Nächste Phasen

→ **Phase 3: CHANNEL_SPLIT** (Metadaten)  
→ **Phase 4: NORMALIZATION**  
→ **Phase 5: GLOBAL_METRICS**
