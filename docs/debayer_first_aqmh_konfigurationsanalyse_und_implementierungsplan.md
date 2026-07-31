# Debayer-First-AQMH: Konfigurationsanalyse und Implementierungsplan

**Datum:** 2026-07-31
**Branch:** `debayer-first-aqmh`
**Referenzen:**

- Schärfe-Strategiedokument, Abschnitt 9
- `docs/sharpnes-v3_code_review_und_naechste_schritte.md`
- `docs/m31_20260729_stacked_rgb_vergleich.md` (Experiment D, D+, stacking program-Analyse)

---

## 1. Konfigurationsanalyse

### 1.1 Methodik

Die Analyse basiert auf:

- `tile_compile_cpp/include/tile_compile/config/configuration.hpp` (alle Config-Structs);
- `tile_compile_cpp/tile_compile.schema.json` (JSON-Schema);
- `tile_compile_cpp/tile_compile.yaml` (Defaults);
- `tile_compile_cpp/apps/runner_pipeline.cpp` (Phasen-Reihenfolge);
- `tile_compile_cpp/apps/runner_phase_*.cpp` (Phasen-Implementierungen);
- `tile_compile_cpp/include/tile_compile/core/cfa_warp.hpp` (CFA-Subplane-Warp);
- `tile_compile_cpp/include/tile_compile/image/cfa_processing.hpp` (Debayer-Methoden);
- `tile_compile_cpp/include/tile_compile/reconstruction/aqmh_reconstruction.hpp` (AQMH-Reconstruction);
- `tile_compile_cpp/apps/runner_resume.cpp` (Resume-Vertrag).

Jeder Parameter wird kategorisiert als:

- **beizubehalten:** funktioniert im Debayer-First-Pfad unverändert;
- **anzupassen:** Semantik aendert sich (CFA -> RGB-Domain);
- **obsolet im Debayer-First-Pfad:** wird im neuen Pfad nicht verwendet, aber fuer CFA-Fallback beibehalten;
- **neu zu definieren:** bisher nicht vorhanden, fuer Debayer-First erforderlich.

### 1.2 Phasenuebersicht

```text
Phase 0:  SCAN_INPUT          - Frames scannen, Kalibrierung
Phase 1:  CHANNEL_SPLIT       - bei OSC: Deferred, bei RGB: NOP
Phase 2:  NORMALIZATION       - Per-Frame-Skalierung
Phase 3:  GLOBAL_METRICS      - Frame-Qualitaet (Classic-only)
Phase 4:  REGISTRATION        - Sterne matchen, Warps berechnen
Phase 5:  PREWARP             - Frames auf Canvas warpen
Phase 6:  COMMON_OVERLAP      - Gueltige Pixelmaske
Phase 7:  AQMH_MAPS           - Q-Maps pro Frame
Phase 8:  AQMH_GLOBAL_QUALITY - Frame-Level-Gewichte
Phase 9:  AQMH_RECONSTRUCTION - Per-Pixel-gewichteter Stack
Phase 10: STACKING            - Overlap-Add, finales Stitching
Phase 11: DEBAYER             - Post-Stack-Debayering (entfaellt bei DF)
Phase 12: ASTROMETRY          - Plate Solving
Phase 13: BGE                 - Background Gradient Elimination
Phase 14: PCC                 - Photometric Color Calibration
Phase 15: HYPERMETRIC_STRETCH - Farbstreckung
Phase 16: DONE
```

### 1.3 Parameter-Kategorisierung

#### Phase 0: SCAN_INPUT

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `data.color_mode` | beizubehalten | "OSC" bleibt der Eingabemodus; Debayer-First aendert nicht den Input, sondern den Verarbeitungszeitpunkt |
| `data.bayer_pattern` | beizubehalten | Wird fuer Pre-Debayering benoetigt, nicht fuer Post-Debayering |
| `data.linear_required` | beizubehalten | Unabhaengig von Debayer-Domain |
| `linearity.*` | beizubehalten | Linearitaetspruefung auf Rohdaten, vor Debayering |
| `calibration.*` | beizubehalten | Bias/Dark/Flat auf CFA-Rohdaten, vor Debayering |

#### Phase 1: CHANNEL_SPLIT

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| *(keine Parameter)* | anzupassen | Bei Debayer-First wird hier das Pre-Debayering durchgefuehrt, falls aktiviert. Die Phase wird von einem Metadata-Only-Schritt zu einem aktiven Debayer-Schritt. |

#### Phase 2: NORMALIZATION

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `normalization.enabled` | beizubehalten | |
| `normalization.mode` | beizubehalten | |
| `normalization.per_channel` | anzupassen | Bei CFA: pro CFA-Subplane. Bei Debayer-First: pro R/G/B-Kanal. Die Semantik aendert sich, der Parameter bleibt. |
| `dithering.enabled` | beizubehalten | |
| `dithering.min_shift_px` | beizubehalten | |

#### Phase 3: GLOBAL_METRICS

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `global_metrics.*` | beizubehalten | Classic-only; bei `method: aqmh` nicht aktiv. Keine Aenderung fuer Debayer-First-AQMH. |

#### Phase 4: REGISTRATION

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `registration.*` (alle) | beizubehalten | Registration arbeitet auf Sternen; bei Debayer-First auf debayerter Luminanz statt CFA-Proxy. Die Registrations-Engine selbst aendert sich nicht, nur das Eingangsbild. |

#### Phase 5: PREWARP

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `aqmh.reconstruction.prewarp_interpolation` | anzupassen | Bei CFA: Interpolation auf CFA-Subplanes. Bei Debayer-First: Interpolation auf RGB-Kanaele. Der Parameter bleibt, aber die angewandte Interpolation aendert sich. "linear" bleibt Default; "lanczos4" wird auf RGB sinnvoller als auf CFA. |
| `aqmh.reconstruction.delete_prewarped_cache_after_run` | beizubehalten | |

#### Phase 6: COMMON_OVERLAP

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `stacking.common_overlap_required_fraction` | beizubehalten | |
| `stacking.tile_common_valid_min_fraction` | beizubehalten | |

#### Phase 7: AQMH_MAPS

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `aqmh.enabled` | beizubehalten | |
| `aqmh.pyramid.scales` | beizubehalten | |
| `aqmh.pyramid.base_window_px` | beizubehalten | |
| `aqmh.pyramid.w_sharp` | beizubehalten | |
| `aqmh.pyramid.w_snr` | beizubehalten | |
| `aqmh.pyramid.score_scale` | beizubehalten | |
| `aqmh.pyramid.k_artifact` | beizubehalten | |
| `aqmh.pyramid.frac_artifact_max` | beizubehalten | |
| `aqmh.storage.resolution_divisor` | anzupassen | Bei CFA: Q-Map auf halber Aufloesung. Bei Debayer-First: Q-Map auf debayerter Luminanz (volle Aufloesung). Default kann bei `2` bleiben, aber die effektive Aufloesung der Q-Map steigt. |
| `aqmh.storage.dtype` | beizubehalten | |
| `aqmh.storage.max_resident_maps` | anzupassen | Bei 3-Kanal-Reconstruction kann der Speicherbedarf steigen. Default sollte conservativer werden (z.B. `1` oder `2`). |
| `aqmh.diagnostics.*` | beizubehalten | |

#### Phase 8: AQMH_GLOBAL_QUALITY

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `aqmh.global_quality.*` | beizubehalten | Frame-Level-Gewichte bleiben; werden auf debayerte Luminanz berechnet. |

#### Phase 9: AQMH_RECONSTRUCTION

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `aqmh.reconstruction.clip_sigma` | beizubehalten | Pro Kanal angewendet. |
| `aqmh.reconstruction.clip_sigma_low` | beizubehalten | |
| `aqmh.reconstruction.clip_sigma_high` | beizubehalten | |
| `aqmh.reconstruction.clip_iterations` | beizubehalten | |
| `aqmh.reconstruction.min_fraction` | beizubehalten | |
| `aqmh.reconstruction.min_n_eff` | beizubehalten | |
| `aqmh.reconstruction.chunk_rows` | beizubehalten | |
| `aqmh.reconstruction.memory_budget_mb` | anzupassen | Bei 3-Kanal-Reconstruction ca. 3x Speicher. Default sollte explizit auf einen sichereren Wert gesetzt werden. |
| `aqmh.reconstruction.registration_weight_guard` | beizubehalten | |
| `aqmh.reconstruction.registration_weight_floor` | beizubehalten | |
| `aqmh.reconstruction.registration_cc_floor` | beizubehalten | |
| `aqmh.reconstruction.registration_cc_full` | beizubehalten | |
| `aqmh.reconstruction.registration_sequential_factor` | beizubehalten | |
| `aqmh.reconstruction.registration_predicted_factor` | beizubehalten | |
| `aqmh.reconstruction.registration_chain_depth_penalty` | beizubehalten | |
| `aqmh.reconstruction.registration_chain_depth_max_penalty` | beizubehalten | |
| `aqmh.reconstruction.structure_mask_low_q` | beizubehalten | |
| `aqmh.reconstruction.structure_mask_high_q` | beizubehalten | |
| `aqmh.reconstruction.structure_mask_blur_sigma_px` | beizubehalten | |

#### Phase 10: STACKING

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `stacking.method` | beizubehalten | |
| `stacking.sigma_clip.*` | beizubehalten | |
| `stacking.output_stretch` | beizubehalten | |

#### Phase 11: DEBAYER

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| *(hardcoded)* | obsolet im DF-Pfad | Bei Debayer-First entfaellt das Post-Stack-Debayering. Die Phase wird zu einem NOP oder uebersprungen. Der CFA-Fallback-Pfad behaelt die aktuelle Logik. |

#### Phase 12-15: ASTROMETRY, BGE, PCC, HYPERMETRIC_STRETCH

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `astrometry.*` | beizubehalten | Arbeitet auf RGB; bei DF liegt RGB frueher vor. |
| `bge.*` | beizubehalten | Arbeitet auf RGB; bei DF liegt RGB frueher vor. |
| `pcc.*` | beizubehalten | Arbeitet auf RGB; bei DF liegt RGB frueher vor. |
| `hypermetric_stretch.*` | beizubehalten | Arbeitet auf RGB; bei DF liegt RGB frueher vor. |

#### Phase 16: DONE

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `output.*` | beizubehalten | |
| `runtime_limits.*` | beizubehalten | `memory_budget` und `parallel_workers` werden bei DF wichtiger, aber die Parameter selbst bleiben. |

#### Classic-only-Parameter

| Parameter | Status | Begruendung |
|-----------|--------|-------------|
| `tile.*` | beizubehalten | Classic-only; bei `method: aqmh` nicht aktiv. |
| `local_metrics.*` | beizubehalten | Classic-only. |
| `tile_denoise.*` | beizubehalten | Classic-only. |
| `chroma_denoise.*` | beizubehalten | Classic-only, CFA-spezifisch. Bei DF-AQMH nicht aktiv, aber fuer Classic-Modus beizubehalten. |
| `synthetic.*` | beizubehalten | Classic-only. |
| `stacking.cluster_quality_weighting.*` | beizubehalten | Classic-only. |
| `stacking.cosmetic_correction*` | beizubehalten | Classic-only. |

### 1.4 Neue Parameter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `aqmh.reconstruction.debayer_first` | bool | true | Hauptschalter fuer Debayer-First-AQMH. Wenn `true`: Pre-Debayering vor Prewarp, RGB-Warp, kanalweise AQMH-Reconstruction. DF-AQMH ist der angestrebte Produktionspfad und deshalb der Default; der CFA-Pfad bleibt mit `false` als expliziter Fallback verfuegbar. |
| `aqmh.reconstruction.pre_debayer_method` | string | "edge_aware" | Debayer-Methode vor dem Warp. Optionen: "edge_aware", "bilinear", "nearest". |
| `aqmh.reconstruction.rgb_q_map_mode` | string | "shared_luma" | Q-Map-Modus: "shared_luma" = gemeinsame Q-Maps auf debayerter Luminanz; "per_channel" = separate Q-Maps pro Kanal. |
| `aqmh.reconstruction.rgb_memory_strategy` | string | "sequential" | Speicherstrategie: "sequential" = ein Kanal nach dem anderen; "parallel" = alle Kanaele gleichzeitig (hoher Speicherbedarf). |

### 1.5 Zusammenfassung

```text
Obsolet im DF-Pfad:     0 Parameter (CFA-Pfad als Fallback beibehalten)
Beizubehalten:          ~95 % aller Parameter
Anzupassen:             4 Parameter (per_channel, prewarp_interpolation,
                                   storage.resolution_divisor, memory_budget_mb)
Neu zu definieren:      4 Parameter (debayer_first, pre_debayer_method,
                                     rgb_q_map_mode, rgb_memory_strategy)
```

Die bestehende Konfiguration bleibt fast vollstaendig erhalten. Debayer-First-AQMH ist eine additive Erweiterung, kein Ersatz. `debayer_first` ist `true`, weil DF-AQMH der angestrebte Produktionspfad ist. Der CFA-Pfad bleibt als expliziter Fallback verfuegbar, wenn `debayer_first: false`. Bestehende Runs werden ueber ihre gespeicherte effektive Konfiguration weiterhin eindeutig dem urspruenglichen Pfad zugeordnet.

---

## 2. Implementierungsplan

### 2.1 Uebersicht

Der Plan folgt der im Schärfe-Strategiedokument, Abschnitt 9, definierten Reihenfolge:

```text
9.1 Speicherbegrenzter Prototyp (32-64 Frames)
9.2 Debayer-First-AQMH (volle Implementierung)
9.3 Dither-Verteilung analysieren (Coverage-Analyse)
```

Jeder Schritt hat klare Abnahmekriterien und Abbruchbedingungen.

### 2.2 Architektur-Entscheidungen

#### 2.2.1 Datenfluss: CFA-Pfad vs. Debayer-First-Pfad

```text
CFA-Pfad (aktuell, debayer_first=false):
  CFA-Mosaik
  -> Registration auf CFA-Proxy
  -> Prewarp: CFA-Subplane-Warp (cfa_warp.hpp)
  -> DiskCacheFrameStore (1 Kanal, CFA-Mosaik)
  -> AQMH_MAPS: Q-Maps auf CFA-Luma
  -> AQMH_RECONSTRUCTION: CFA-Mosaik-Reconstruction
  -> STACKING: CFA-Mosaik-Stack
  -> DEBAYER: Post-Stack Edge-Aware
  -> RGB-Output

Debayer-First-Pfad (neu, debayer_first=true):
  CFA-Mosaik
  -> Pre-Debayer: Edge-Aware pro Frame
  -> Registration auf debayerter Luminanz
  -> Prewarp: RGB-Kanal-Warp (cv::warpAffine pro Kanal)
  -> DiskCacheFrameStoreRGB (3 Kanaele)
  -> AQMH_MAPS: Q-Maps auf debayerter Luminanz
  -> AQMH_RECONSTRUCTION: pro Kanal mit shared Q-Maps
  -> STACKING: RGB-Stack (kein Post-Debayer)
  -> RGB-Output
```

#### 2.2.2 Schluessel-Datenstrukturen

**Neu: `DiskCacheFrameStoreRGB`**

```cpp
// Speichert 3 Kanaele (R, G, B) pro Frame auf Festplatte.
// Sequenzieller Zugriff: ein Kanal nach dem anderen.
// Speicherbegrenzung: nur aktiver Kanal im RAM.
class DiskCacheFrameStoreRGB {
  fs::path cache_dir_;
  size_t n_frames_;
  int rows_, cols_;
  // Pro Kanal: {fi}.R.raw, {fi}.G.raw, {fi}.B.raw
  void store(size_t fi, const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B);
  Matrix2Df load_channel(size_t fi, int channel) const;  // 0=R, 1=G, 2=B
  bool has_data(size_t fi) const;
};
```

**Erweitert: `AqmhReconstructionResult`**

```cpp
struct AqmhReconstructionResultRGB {
  Matrix2Df R, G, B;           // Rekonstruierte Kanaele
  Matrix2Df weight_sum_R, weight_sum_G, weight_sum_B;
  Matrix2Df uniform_control_R, uniform_control_G, uniform_control_B;
  Matrix2D8u valid_mask_R, valid_mask_G, valid_mask_B;
  Matrix2Df support_R, support_G, support_B;
  // Metriken pro Kanal und auf Luminanz
};
```

#### 2.2.3 Resume-Vertrag

Neue Artefakte:

- `cache/prewarped_frames_rgb/` (ersetzt `cache/prewarped_frames/` bei DF);
- `pre_debayer_metadata.json` (Bayer-Pattern, Origin, Methode).

Jeder DF-Cache muss zusaetzlich einen Vollstaendigkeitsmarker und eine Manifestdatei mit Frame-Anzahl, Dimensionen, Kanalreihenfolge, Konfigurations-Hash, Cache-Format-Version und Luminanzdefinition besitzen. Einzelne Frame-Dateien gelten nur dann als gueltig, wenn Groesse, Datentyp und endlicher Wertebereich validiert wurden. Ein Cache mit fehlenden oder inkompatiblen Frames darf nicht als resumierbar gelten.

Resume-Validierung muss erkennen, ob ein Run im DF-Modus war, und die entsprechenden Caches validieren. Ein DF-Resume darf niemals stillschweigend auf den CFA-Cache zurueckfallen; umgekehrt darf ein CFA-Resume den RGB-Cache nicht verwenden.

#### 2.2.4 Verbindliche Daten- und Messvertraege

- `max_frames` wird ausschliesslich unter `runtime_limits.max_frames` gefuehrt; `data.max_frames` ist nicht zulaessig.
- `memory_budget_mb` und `runtime_limits.memory_budget_mb` werden in MiB interpretiert. Der kleinere gesetzte Wert ist die harte Prozessgrenze; RSS-Spitzenwert und CUDA-Speicher werden getrennt protokolliert.
- Die Luminanz wird zentral als `0.25 * R + 0.5 * G + 0.25 * B` aus bereits kanalnormierten RGB-Daten berechnet. Dieselbe Funktion wird fuer Registration, Q-Maps, Gates und Vergleichsmetriken verwendet.
- RGB-Kanaele werden in der festen Reihenfolge R, G, B gespeichert. Jeder Kanal besitzt eigene `valid_mask`, `support_map`, `weight_sum` und Outlier-/Sigma-Clip-Statistiken.
- Zulaessige Werte sind `pre_debayer_method = edge_aware|bilinear|nearest`, `rgb_q_map_mode = shared_luma|per_channel` und `rgb_memory_strategy = sequential|parallel`. Der Prototyp erlaubt nur `edge_aware`, `shared_luma` und `sequential`; andere Werte werden mit einer Validierungsfehlermeldung abgelehnt.
- Die Validierung erfolgt immer sowohl auf Luminanz als auch pro Kanal. Ein Luminanz-Gate kann keinen ungueltigen oder deutlich regressiven Einzelkanal ueberstimmen.

---

### 2.3 Schritt 9.1: Speicherbegrenzter Prototyp

**Ziel:** Beweisen, dass Debayer-First-AQMH mit shared Q-Maps auf debayerter Luminanz prinzipiell funktioniert und keine Bayer-Rasterartefakte erzeugt.

**Rahmen:**

- 32-64 Frames über den Runner-Aufruf `--max-frames` (die Konfigurationsdatei steuert nur `linearity.max_frames` fuer die Stichprobe);
- maximal 2 Worker;
- kein BGE/PCC/HMS;
- gemeinsame Luminanz-Q-Maps (`rgb_q_map_mode: "shared_luma"`);
- sequenzielle Kanal-Reconstruction (`rgb_memory_strategy: "sequential"`);
- Pre-Debayer: Edge-Aware (`pre_debayer_method: "edge_aware"`).

#### 9.1.1 Config-Parameter hinzufuegen

**Dateien:**

1. `tile_compile_cpp/include/tile_compile/config/configuration.hpp`:
   - `AqmhReconstructionConfig` erweitern um:
     - `bool debayer_first = true;`
     - `std::string pre_debayer_method = "edge_aware";`
     - `std::string rgb_q_map_mode = "shared_luma";`
     - `std::string rgb_memory_strategy = "sequential";`

2. `tile_compile_cpp/tile_compile.schema.json` und `.schema.yaml`:
   - Entsprechende Properties und Defaults validieren.

3. `tile_compile_cpp/tile_compile.yaml` und die aktiven Beispielprofile:
   - Defaults explizit dokumentieren, insbesondere `debayer_first: true` sowie die jeweiligen String-Defaults.
   - Fuer 9.1 sind nur `shared_luma` und `sequential` freigegeben; nicht implementierte Alternativen muessen mit einer Validierungsfehlermeldung abgelehnt werden.

4. Parser/Serializer in `config_parser.cpp` oder aehnlich:
   - Neue Felder parsen und serialisieren.

5. `docs/configuration_reference.md` und `_en.md`:
   - Neue Parameter dokumentieren.

6. Tests:
   - Config-Parsing-Test fuer neue Parameter;
   - Default-Validierung.

#### 9.1.2 Pre-Debayer-Phase implementieren

**Ort:** `tile_compile_cpp/apps/runner_phase_channel_split_normalization_global_metrics.cpp` (oder neue Funktion in `runner_phase_pre_debayer.cpp`).

**Logik:**

```cpp
if (cfg.aqmh.enabled && cfg.aqmh.reconstruction.debayer_first &&
    detected_mode == ColorMode::OSC) {
    // Pro Frame:
    //   1. CFA-Mosaik laden (nach Kalibrierung)
    //   2. Normalisierung anwenden (CFA-Domain)
    //   3. Pre-Debayer: edge_aware / bilinear / nearest
    //   4. R, G, B speichern (im RAM oder temporaer)
    //   5. Luminanz = 0.25*R + 0.5*G + 0.25*B
    //   6. Luminanz fuer Registration markieren
    //
    // Wichtig: Normalisierung bleibt auf CFA-Domain,
    // damit Photometrie konsistent bleibt.
    // Pre-Debayering erfolgt NACH Normalisierung.
}
```

**Debayer-Aufruf:**

```cpp
auto debayered = image::debayer_opencv(
    cfa_frame, bayer_pattern, origin_x, origin_y, /*ahd=*/true);
// debayered.R, debayered.G, debayered.B
```

**Artefakte:**

- `artifacts/pre_debayer_metadata.json` mit Formatversion, Vollstaendigkeitsmarker, Frame-Anzahl, Dimensionen, Bayer-Pattern, Methode, Kanalreihenfolge, Luminanzgewichten und Cache-Pfaden;
- `cache/debayered_frames/` fuer normierte, debayerte RGB-Frames;
- `cache/prewarped_frames_rgb/` fuer die nachfolgenden RGB-Prewarp-Frames.

**Tests:**

- Unit-Test: Pre-Debayer erzeugt korrekte RGB-Dimensionen (2x CFA);
- Unit-Test: Bayer-Origin wird korrekt weitergegeben;
- Integration-Test: 4-Frame-Mini-Run mit `debayer_first=true`;
- Metadaten-Test: unvollstaendiger RGB-Cache wird als nicht resumierbar markiert.

#### 9.1.3 RGB-Prewarp implementieren

**Ort:** `tile_compile_cpp/apps/runner_phase_registration.cpp` (Prewarp-Abschnitt, ab Zeile ~3155).

**Aenderung:**

```cpp
if (cfg.aqmh.reconstruction.debayer_first) {
    // Statt CFA-Subplane-Warp:
    //   Pro Frame: R, G, B separat mit cv::warpAffine warpen
    //   Interpolation: prewarp_interpolation (linear/cubic/lanczos4)
    //   Border: replicate
    //   Store in DiskCacheFrameStoreRGB
    DiskCacheFrameStoreRGB prewarped_rgb(
        run_dir / "cache" / "prewarped_frames_rgb",
        frames.size(), canvas_height, canvas_width);
    // Pro Worker/Frame:
    //   warpAffine(R, warp, canvas, interp_flag, BORDER_REPLICATE)
    //   warpAffine(G, warp, canvas, interp_flag, BORDER_REPLICATE)
    //   warpAffine(B, warp, canvas, interp_flag, BORDER_REPLICATE)
    //   prewarped_rgb.store(fi, R_w, G_w, B_w)
} else {
    // Bestehender CFA-Subplane-Warp-Pfad (unveraendert)
}
```

**GPU-Pfad:**

- `cv::cuda::warpAffine` pro Kanal (3 Aufrufe statt 4 CFA-Subplane-Aufrufe);
- CPU-Fallback: `cv::warpAffine` pro Kanal.

**Tests:**

- Unit-Test: RGB-Prewarp erzeugt korrekte Dimensionen;
- Unit-Test: Warp-Inverse-Map-Offset-Korrektur identisch zu CFA-Pfad;
- GPU/CPU-Paritaetstest.

#### 9.1.4 Q-Maps auf debayerter Luminanz

**Ort:** `tile_compile_cpp/apps/runner_phase_aqmh_maps.cpp` bzw. `runner_phase_local_metrics.cpp`.

**Aenderung:**

```cpp
if (cfg.aqmh.reconstruction.debayer_first) {
    // Q-Map-Input: debayerte Luminanz aus prewarped_rgb
    // Luminanz = 0.25*R + 0.5*G + 0.25*B
    // Q-Map-Aufloesung: volle Canvas-Aufloesung (resolution_divisor weiterhin anwendbar)
    // QualityMapCache bleibt unverändert, speichert jetzt Luminanz-Q-Maps
} else {
    // Bestehender CFA-Luma-Q-Map-Pfad
}
```

**Tests:**

- Unit-Test: Q-Map-Dimensionen entsprechen Canvas (nicht CFA-Subplane);
- Unit-Test: Q-Map-Werte sind endlich und im erwarteten Bereich.

#### 9.1.5 Kanalweise AQMH-Reconstruction

**Ort:** `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp`.

**Aenderung:**

```cpp
if (cfg.aqmh.reconstruction.debayer_first) {
    // shared_luma-Modus:
    //   1. Luminanz-Q-Maps laden (bereits in AQMH_MAPS berechnet)
    //   2. Pro Kanal (R, G, B):
    //      a. Frames aus DiskCacheFrameStoreRGB.load_channel(fi, ch)
    //      b. reconstruct_aqmh_weighted() mit shared Q-Maps
    //      c. Resultat in AqmhReconstructionResultRGB speichern
    //   3. Validierung auf Luminanz (0.25*R + 0.5*G + 0.25*B)
    //
    // sequential-Modus:
    //   Kanal 0 (R) vollstaendig reconstructen, dann G, dann B.
    //   Nur ein Kanal gleichzeitig im RAM.
    //
    // Andere Modi:
    //   rgb_q_map_mode=per_channel und rgb_memory_strategy=parallel werden
    //   im Prototyp explizit abgelehnt, bis eigene Q-Map-/Speichervertraege
    //   und Tests implementiert sind.
} else {
    // Bestehender CFA-AQMH-Reconstruction-Pfad
}
```

**Wichtig:**

- `AqmhFrameLoader` muss angepasst werden, um aus `DiskCacheFrameStoreRGB` zu laden;
- `AqmhFrameRegionLoader` muss fuer RGB-Region-Streaming angepasst werden;
- Sigma-Clip, Cherry-Pick, Structure-Mask-Blending: pro Kanal anwenden;
- pro Kanal eigene Valid-Mask, Supportkarte und Weight-Sum ableiten;
- Validierung auf Luminanz und zusaetzlich pro Kanal durchfuehren;
- nicht implementierte Q-Map-/Speicherstrategien vor dem Start ablehnen.

**Tests:**

- Unit-Test: 4-Frame-DF-Reconstruction erzeugt 3 Kanaele mit korrekten Dimensionen;
- Unit-Test: Shared-Q-Map-Modus verwendet dieselbe Q-Map fuer alle 3 Kanaele;
- Unit-Test: Sequential-Modus gibt Speicher pro Kanal frei;
- Unit-Test: pro Kanal werden Valid-Mask und Supportdaten erzeugt;
- Integration-Test: Mini-Run mit Validierung;
- Negativtest: `per_channel` und `parallel` werden im 9.1-Prototyp abgelehnt.

#### 9.1.6 Post-Stack-Output anpassen

**Ort:** `tile_compile_cpp/apps/runner_phase_post_stack_output.cpp`.

**Aenderung:**

```cpp
if (debayer_first_was_used) {
    // RGB liegt bereits vor (recon_R, recon_G, recon_B)
    // Kein Post-Stack-Debayering noetig
    // Skalierung, Canvas-Mask, Output wie bisher
} else {
    // Bestehender Post-Stack-Debayer-Pfad
}
```

**Tests:**

- Unit-Test: DF-Output ist RGB mit 3 Kanaelen;
- Unit-Test: Kein Post-Stack-Debayer-Aufruf bei DF;
- Output-FITS-Header aller RGB-Ausgaben markiert `DEBAYER=PRE_STACK`;
- Integrationstest liest den Header von `stacked_rgb.fits` und den drei Kanaldateien.

#### 9.1.7 Resume-Vertrag erweitern

**Ort:** `tile_compile_cpp/apps/runner_resume.cpp`.

**Aenderung:**

- `is_inplace_rerun_phase`: keine Aenderung;
- `is_aqmh_cache_resume_phase`: keine Aenderung;
- Resume-Validierung:
  - Erkennen, ob Run im DF-Modus war (aus `effective_config.json`);
  - Wenn DF: `cache/prewarped_frames_rgb/` und alle drei Kanäle je Frame validieren;
  - `artifacts/pre_debayer_metadata.json` auf Formatversion, Vollstaendigkeit, Frame-Anzahl, Dimensionen, Bayer-Pattern und Kanalreihenfolge validieren;
  - bei DF niemals den CFA-Cache als Ersatz fuer fehlende RGB-Frames verwenden;
  - bei fehlendem oder inkompatiblem RGB-Cache Resume mit einem eindeutigen Fehler abbrechen.

**Tests:**

- Unit-Test: Resume erkennt DF-Modus;
- Unit-Test: Resume validiert RGB-Cache;
- Unit-Test: Resume mit CFA-Cache schlaegt fehl, wenn DF erwartet.

#### 9.1.8 Prototyp-Run

**Config:**

```yaml
data:
  color_mode: OSC
  bayer_pattern: auto
aqmh:
  enabled: true
  reconstruction:
    debayer_first: true
    pre_debayer_method: edge_aware
    rgb_q_map_mode: shared_luma
    rgb_memory_strategy: sequential
    prewarp_interpolation: linear
    memory_budget_mb: 4096
runtime_limits:
  parallel_workers: 2
  memory_budget_mb: 4096

# Runner-Aufruf: tile_compile_runner run --max-frames 64
bge:
  enabled: false
pcc:
  enabled: false
hypermetric_stretch:
  enabled: false
```

**Abnahmekriterien 9.1:**

1. Run erreicht `DONE` ohne OOM;
2. Output ist RGB (3 Kanaele);
3. Keine sichtbare 2x2-Bayer-Blockstruktur;
4. positionsgematchte radiale FWHM mindestens gleich gut wie CFA-AQMH-Baseline;
5. Background-RMS-Regression hoechstens +5 %;
6. positionsgematchter Peak/Flux mindestens gleich gut wie die CFA-AQMH-Baseline;
7. Seam- und Tail-Gates bestehen;
8. Raw-Baseline-Gate besteht auf Luminanz und pro Kanal;
9. RSS < 8 GiB bei 64 Frames und keine Ueberschreitung des kleineren Memory-Budgets.

Die Metriken werden im linearen Datenraum, mit identischem Canvas, Crop, Hintergrundmodell und derselben positionsgematchten Sternliste wie bei der CFA-Baseline berechnet. Mindestens 400 gueltige Sternpaare sind erforderlich; FWHM, Peak/Flux, Background-RMS, Seam und Tail werden als Median sowie als 90. Perzentil protokolliert.

**Abbruchbedingungen:**

- OOM oder RSS > 12 GiB;
- Background-Regression > +10 %;
- sichtbare Rasterstruktur im Output;
- Seam-Regressionsgrenze ueberschritten.

---

### 2.4 Schritt 9.2: Vollstaendige Debayer-First-AQMH-Implementierung

**Ziel:** Den Prototyp auf 645 Frames skalieren und BGE/PCC/HMS aktivieren.

#### 9.2.1 Speicheroptimierung

- `max_resident_maps` automatisch anpassen bei DF;
- Region-Streaming fuer RGB-Kanaele (nur aktive Region im RAM);
- `memory_budget_mb` als harte Begrenzung implementieren (nicht nur Hinweis);
- Worker-Anzahl automatisch reduzieren bei DF.

#### 9.2.2 GPU-Optimierung

- `cv::cuda::warpAffine` mit Stream-Pipelining fuer 3 Kanaele;
- AQMH-CUDA-Kernel fuer RGB-Kanaele wiederverwenden (kanalweise Aufruf);
- Keine neue CUDA-Kernel erforderlich (bestehende sigma-clip/cherry-pick arbeiten kanalunabhaengig).

#### 9.2.3 Validierung auf RGB

- Raw-Baseline-Guard: Luminanz aus RGB rekonstruieren, gegen Raw-AQMH-Luminanz vergleichen;
- Uniform-Control-Gate: pro Kanal oder auf Luminanz;
- Structure-Mask-Blending: pro Kanal mit gemeinsamer Mask aus Luminanz;
- Seam/Tail-Metriken: auf Luminanz.

#### 9.2.4 BGE/PCC/HMS-Anpassung

- BGE: arbeitet bereits auf RGB; keine Aenderung;
- PCC: arbeitet bereits auf RGB; keine Aenderung;
- HMS: arbeitet bereits auf RGB; keine Aenderung;
- Einziger Unterschied: RGB liegt frueher vor (nach AQMH_RECONSTRUCTION statt nach DEBAYER).

#### 9.2.5 Vollstaendiger Run

**Config:**

```yaml
aqmh:
  enabled: true
  reconstruction:
    debayer_first: true
    pre_debayer_method: edge_aware
    rgb_q_map_mode: shared_luma
    rgb_memory_strategy: sequential
    prewarp_interpolation: linear
    memory_budget_mb: 8192
runtime_limits:
  parallel_workers: 2
  max_frames: 645
  memory_budget_mb: 8192
bge:
  enabled: true
pcc:
  enabled: true
hypermetric_stretch:
  enabled: true
```

**Abnahmekriterien 9.2:**

1. Run erreicht `DONE` mit 645 Frames;
2. Alle Abnahmekriterien aus 9.1 erfuellt;
3. BGE/PCC/HMS erfolgreich durchlaufen;
4. positionsgematchte FWHM und Peak/Flux besser als CFA-AQMH-Baseline;
5. kein einzelner RGB-Kanal verletzt die pro-Kanal-Raw-Baseline oder die Background-/Support-Gates;
6. Vergleich mit der Referenz des stacking program (`result.fit`) erfolgt numerisch im linearen Datenraum mit identischem Crop und derselben Sternliste:
   - radiale FWHM und Peak/Flux werden mindestens nicht schlechter als die festgelegte Toleranz;
   - Background-RMS, Seam und Tail bleiben innerhalb der festgelegten Gates;
   - mindestens 400 gueltige positionsgematchte Sternpaare;
7. visueller Vergleich bestaetigt keine Blockstrukturen und wird nur ergaenzend zu den numerischen Kriterien verwendet.

BGE, PCC und HMS werden vor 9.2 anhand ihrer Input-Vertraege geprueft. Fuer jede Phase sind Kanalreihenfolge, Dimensionen, Support-/Valid-Masks, NaN-Semantik und Output-Artefakt zu testen; "keine Aenderung" gilt erst nach bestandenem RGB-Integrationstest.

---

### 2.5 Schritt 9.3: Dither-Verteilung analysieren

**Ziel:** Praeferenz fuer Strategie 3 (CFA-aware Drizzle) evaluieren.

#### 9.3.1 Analyse-Skript

**Datei:** `tools/analyze_dither_coverage.py`.

Das Skript muss ohne Backend und ohne Aenderung an Run-Artefakten reproduzierbar ausfuehrbar sein:

```text
python3 tools/analyze_dither_coverage.py \
  --registration artifacts/global_registration.json \
  --normalization artifacts/normalization.json \
  --stars artifacts/matched_stars.json \
  --output-dir analysis/dither_coverage
```

**Input:**

- `artifacts/global_registration.json` (Warps, CC-Werte, Frame-Indizes);
- `artifacts/normalization.json` (gueltige Frame-Indizes und Ausschlussgruende);
- `artifacts/matched_stars.json` (eine feste, positionsgematchte Sternliste im Canvas-Koordinatensystem).

Fehlende, doppelte oder inkonsistente Frame-Indizes fuehren zu einem Fehler; das Skript darf keine Eingabeartefakte veraendern.

**Ausgabe:**

- `summary.json` mit Bins, Zaehlern, Schwellenwerten und finaler Klassifikation;
- `shift_histogram.csv`;
- Coverage-Karten als CSV und PNG fuer global, lokal und jede Bayer-Phase;
- `residuals.csv` mit CC, Shift und Registrierungsrestfehler;
- `README.txt` mit Kommando, Eingabe-Hashes und Tool-Version.

**Analyse:**

1. X-/Y-Shift-Histogramm aller gueltigen Frames;
2. Verteilung der Subpixel-Positionen modulo 2 Pixel in festen 0,1-Pixel-Bins;
3. Coverage-Karte pro Bayer-Phase (RG, GR, GB, BG);
4. lokale Dither-Abdeckung innerhalb eines festen Radius um jede Position der Sternliste;
5. Registrierungsrestfehler (CC vs. Shift) und Drifttrend ueber Frame-Reihenfolge;
6. Gleichmaessigkeit getrennt fuer X, Y und die gemeinsame modulo-2-Verteilung mittels Chi-Quadrat-Test;
7. Mindest-Coverage und Streuung der Coverage an den fuer die Sternmetriken verwendeten Positionen.

Ein Coverage-Pixel gilt als belegt, wenn mindestens ein gueltiger, nicht verworfener Sample in den definierten Canvas-Bereich faellt. Border-Pixel und ausgeschlossene Frames werden nicht in den Nenner aufgenommen. Die lokale Analyse verwendet denselben gueltigen Sternbereich wie die FWHM-/Peak-Flux-Auswertung.

#### 9.3.2 Entscheidungskriterium

Die Schwellenwerte sind vor der Analyse fest und reproduzierbar anzuwenden. Die globale Coverage muss mindestens 80 % je Bayer-Phase erreichen, die lokale Coverage an mindestens 90 % der gueltigen Sternpositionen mindestens 80 %, und der Registrierungs-CC muss fuer mindestens 90 % der gueltigen Frames groesser als 0.5 sein. Der Chi-Quadrat-Test wird mit festgelegten Bins und Signifikanzniveau `alpha = 0.01` ausgewertet; bei zu kleinen erwarteten Bin-Haeufigkeiten wird der Test als nicht entscheidbar markiert. Ein signifikanter Drifttrend gilt als Ausschlusskriterium.

```text
Wenn alle globalen und lokalen Coverage-/CC-/Drift-Gates bestehen:
  - Klassifikation: PROMISING;
  - Strategie 3 ist vielversprechend;
  - separaten Drizzle-Implementierungsplan erstellen.

Wenn globale Gates bestehen, aber lokale Stern-Coverage oder einzelne Bayer-Phasen nicht:
  - Klassifikation: LOCAL_OR_PARTIAL;
  - keinen Produktionsentscheid treffen;
  - gezielte lokale Drizzle-Studie oder weitere Datenanalyse planen.

Sonst:
  - Klassifikation: INSUFFICIENT;
  - Strategie 3 ist fuer diesen Run nicht gerechtfertigt;
  - Debayer-First-AQMH (9.2) bleibt Produktionsloesung.
```

Vor der Auswertung gegen reale Runs muss das Skript mit synthetischen Shifts getestet werden: gleichmaessige modulo-2-Verteilung, einseitige Drift, fehlende Bayer-Phase und unvollstaendige Frame-Indizes muessen jeweils die erwartete Klassifikation oder einen definierten Fehler erzeugen.

#### 9.3.3 Dokumentation

- Ergebnisse in `docs/debayer_first_aqmh_dither_analyse.md` dokumentieren;
- Entscheidung fuer oder gegen Strategie 3 begruenden;
- Bei positivem Ergebnis: separaten Implementierungsplan fuer Drizzle erstellen.

---

## 3. Datei-Aenderungsuebersicht

### 3.1 Neue Dateien

| Datei | Zweck |
|-------|-------|
| `tile_compile_cpp/apps/runner_phase_pre_debayer.cpp` | Pre-Debayer-Phase (optional, falls nicht in channel_split integriert) |
| `tile_compile_cpp/include/tile_compile/runner/disk_cache_frame_store_rgb.hpp` | RGB-Disk-Cache |
| `tile_compile_cpp/src/runner/disk_cache_frame_store_rgb.cpp` | RGB-Disk-Cache-Implementierung |
| `tile_compile_cpp/tests/test_debayer_first_config.cpp` | Config-Parsing-Tests |
| `tile_compile_cpp/tests/test_debayer_first_reconstruction.cpp` | DF-Reconstruction-Tests |
| `tile_compile_cpp/tests/test_disk_cache_frame_store_rgb.cpp` | RGB-Cache-Tests |
| `tools/analyze_dither_coverage.py` | Dither-Analyse-Skript (9.3) |
| `docs/debayer_first_aqmh_dither_analyse.md` | Dither-Analyse-Ergebnisse (9.3) |

### 3.2 Geaenderte Dateien

| Datei | Aenderung |
|-------|-----------|
| `tile_compile_cpp/include/tile_compile/config/configuration.hpp` | 4 neue Parameter in `AqmhReconstructionConfig` |
| `tile_compile_cpp/tile_compile.schema.json` | 4 neue Properties |
| `tile_compile_cpp/tile_compile.schema.yaml` | 4 neue Properties |
| `tile_compile_cpp/tile_compile.yaml` | Defaults dokumentieren |
| `tile_compile_cpp/src/config/config_parser.cpp` (oder aehnlich) | Neue Parameter parsen/serialisieren |
| `tile_compile_cpp/apps/runner_pipeline.cpp` | DF-Pfad-Verzweigung in Phasen 1, 5, 7, 9, 11 |
| `tile_compile_cpp/apps/runner_phase_registration.cpp` | RGB-Prewarp-Pfad |
| `tile_compile_cpp/apps/runner_phase_metrics.cpp` | Pre-Debayer-RGB-Cache und Metadaten |
| `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp` | Kanalweise Reconstruction mit shared Q-Maps und Valid-Masks |
| `tile_compile_cpp/apps/runner_phase_aqmh_maps.cpp` | Q-Maps auf debayerter Luminanz |
| `tile_compile_cpp/apps/runner_phase_post_stack_output.cpp` | DF-Output ohne Post-Debayer und mit `DEBAYER=PRE_STACK` |
| `tile_compile_cpp/apps/runner_phase_post_stack_output.hpp` | DF-Output-Struct |
| `tile_compile_cpp/apps/runner_resume.cpp` | DF-Cache-Validierung |
| `tile_compile_cpp/apps/runner_shared.hpp` | `DiskCacheFrameStoreRGB`-Deklaration |
| `docs/configuration_reference.md` | Neue Parameter dokumentieren |
| `docs/configuration_reference_en.md` | Neue Parameter dokumentieren |
| `docs/configuration_examples_practical_de.md` | DF-Beispielprofil |
| `docs/configuration_examples_practical_en.md` | DF-Beispielprofil |

### 3.3 Unveraenderte Dateien

Alle CFA-spezifischen Dateien bleiben unveraendert:

- `tile_compile_cpp/include/tile_compile/core/cfa_warp.hpp`;
- `tile_compile_cpp/src/image/cfa_processing.cpp` (Debayer-Funktionen werden vom DF-Pfad aufgerufen, aber nicht geaendert);
- `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_cuda.cu` (wird kanalweise aufgerufen);
- `tile_compile_cpp/src/reconstruction/aqmh_validation.cpp` (Validierung auf Luminanz);
- Alle Classic-only-Dateien.

---

## 4. Test-Strategie

### 4.1 Unit-Tests

| Test | Datei | Abdeckung |
|------|-------|-----------|
| Config-Parsing | `test_debayer_first_config.cpp` | 4 neue Parameter, Defaults, Validierung |
| RGB-Disk-Cache | `test_disk_cache_frame_store_rgb.cpp` | Store/Load/HasData, 3 Kanaele |
| Pre-Debayer | `test_debayer.cpp` (erweitert) | Edge-Aware auf normiertem CFA, Origin-Paritaet |
| RGB-Prewarp | `test_aqmh_reconstruction.cpp` (erweitert) | Warp-Offset-Korrektur, Interpolation |
| DF-Reconstruction | `test_debayer_first_reconstruction.cpp` | 4-Frame-Mini-Run, Shared-Q-Maps, Sequential |
| Resume | bestehende Resume-Tests (erweitert) | DF-Modus-Erkennung, RGB-Cache-Validierung, Manifest-/Hash-Pruefung |
| Per-Kanal-Gates | `test_debayer_first_reconstruction.cpp` | Valid-Masks, Supportkarten, Raw-Baseline und Background je Kanal |
| Dither-Analyse | `tools/test_analyze_dither_coverage.py` | synthetische Shifts, Drift, fehlende Phase, unvollstaendige Indizes |

### 4.2 Integration-Tests

| Test | Rahmen |
|------|--------|
| Mini-Run DF | 4 Frames, `debayer_first=true`, kein BGE/PCC/HMS |
| Mini-Run CFA-Fallback | 4 Frames, `debayer_first=false`, identischer Output wie bisher |
| Resume DF | Mini-Run DF, Resume ab AQMH_RECONSTRUCTION |
| GPU/CPU-Paritaet | 4 Frames DF, GPU vs. CPU, Toleranz wie CFA-Pfad |

### 4.3 Vergleichsmessung

Nach Abschluss von 9.2:

- positionsgematchte Paaranalyse gegen CFA-AQMH-Baseline (`M31-s1_guardfix_20260731_1`);
- positionsgematchte Paaranalyse gegen die Referenz des stacking program;
- Metriken: radiale FWHM, Peak/Flux, Background-RMS, Seam, Tail;
- Visueller Vergleich: Screenshots beider Bilder.

---

## 5. Risiken und Mitigationen

| Risiko | Wahrscheinlichkeit | Mitigation |
|--------|-------------------|------------|
| OOM bei 645 Frames | hoch | Sequential-Modus, `max_resident_maps=1`, `memory_budget_mb` als harte Grenze |
| Q-Maps auf Luminanz suboptimal fuer R/B | mittel | `rgb_q_map_mode: "per_channel"` als Alternative implementieren (nach 9.1) |
| Pre-Debayering erzeugt eigene Artefakte | mittel | Edge-Aware als Default; Bilinear als Fallback; Zwischenstufen analysieren |
| GPU/CPU-Divergenz bei RGB-Warp | niedrig | Bestehende Toleranztests auf RGB erweitern |
| Resume bricht bei DF-Cache | mittel | Resume-Tests früh im Prototyp |
| Validierung auf Luminanz verdeckt Kanalprobleme | mittel | Zusaetzliche pro-Kanal-Background-Pruefung |
| stacking program-Paaranalyse unzuverlaessig | niedrig | Positionsgematchte Metrik mit ausreichend Paaren (>400) |

---

## 6. Zeitschaetzung

Keine konkreten Zeitschaetzungen. Die Schritte sind sequenziell und jeder Schritt hat klare Abnahmekriterien, bevor der naechste beginnt.

---

## 7. Zusammenfassung

```text
9.1 Prototyp (32-64 Frames):
    - Config-Parameter hinzufuegen
    - Pre-Debayer-Phase
    - RGB-Prewarp
    - Q-Maps auf Luminanz
    - Kanalweise Reconstruction (sequential, shared_luma)
    - Post-Stack-Output anpassen
    - Resume-Vertrag erweitern
    - Prototyp-Run mit Abnahmekriterien

9.2 Vollimplementation (645 Frames):
    - Speicheroptimierung
    - GPU-Optimierung
    - Validierung auf RGB
    - BGE/PCC/HMS aktivieren
    - Vollstaendiger Run mit stacking program-Vergleich

9.3 Dither-Analyse:
    - Coverage-Skript
    - Subpixel-Verteilung
    - Entscheidung fuer/gegen Strategie 3
    - Dokumentation
```

Der CFA-Pfad bleibt vollstaendig erhalten. `debayer_first: true` ist der Default, weil DF-AQMH der angestrebte Produktionspfad ist; `debayer_first: false` bleibt als expliziter CFA-Fallback verfuegbar. Bestehende Runs werden ueber ihre effektive Konfiguration und ihre validierten Artefakte unveraendert fortgesetzt. Keine CFA-spezifische Datei wird geloescht.
