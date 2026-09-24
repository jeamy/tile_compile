# Analyse: Redundante Daten-Reloads in Schleifen

**Branch:** `CFA-aware-Forward-Drizzle`
**Datum:** 2026-09-23
**Methode:** Statische Code-Analyse durch grep/glob/read + vier parallele Subagents
**Scope:** `tile_compile_cpp/src/reconstruction/`, `tile_compile_cpp/apps/`, `tile_compile_cpp/src/metrics/`

## Zusammenfassung

Die Analyse identifiziert **28 Funde** in fünf Kategorien:

| Kategorie | Funde | Schwergrad |
|-----------|-------|------------|
| A: Daten-Reload pro Stripe/Chunk | 6 | P0/P1 — dominante Kosten |
| B: Daten-Reload pro Band/Kanal | 5 | P1/P2 |
| C: Redundante Rekursion in Pixel-Loops | 8 | P2/P3 |
| D: Per-Iteration Allokationen | 6 | P3 |
| E: Strukturelle Cross-Pass-Redundanz | 3 | P1/P2 |

Die mit Abstand teuersten Funde sind **A1** (Source-Frame-Reload pro Stripe)
und **A2** (Q-Map-Reload pro Stripe ohne In-Memory-Cache).

---

## Umsetzungsstatus (Nachtrag 2026-09-12, final)

Alle 28 Funde plus die Neufunde N1-N9 wurden gegen den aktuellen Code
verifiziert und -- bis auf die als Fehleinschätzung klassifizierten Punkte --
vollständig umgesetzt. Die Arbeiten erfolgten in zwei Runden: zuerst die
lokalen Invarianten/Allokationen (C1-C8, D2, D6), dann die strukturellen
Loop-/Pass-Umbauten (A1-A6, B1-B3, C7, D3-D5, E1-E3, N1-N9). Verifikation
nach der zweiten Runde: Build gruen (tile_compile_runner, tests,
web_backend_cpp), 549/549 Catch2-Tests, 2.756.148 Assertions -- exakt die
Baseline vor den Umbauten.

| Fund | Status | Umsetzung / Bemerkung |
|------|--------|------------------------|
| A1 | umgesetzt | `drizzle_source_scan_box` (in den Rasterizer extrahiert, identische Bounds) liefert pro (Stripe, Frame) die inverse-gemappte Source-Box; `SourceImageRectProvider` liest nur diese Box banded (`forward_drizzle.cpp`), verdrahtet ueber `VerifiedNormalizedSourceCache::read_rect` (`source_quality_artifact.cpp`) und den Runner-Stash (`runner_phase_registration.cpp`) |
| A2 | umgesetzt | `FrameQualityRectProvider` liest nur die Scan-Box via `read_rect`; die Stream-Funktionen nehmen den Rect-Provider direkt entgegen (Existenz-Probes via Leer-Rect, s. N5). Der Full-Provider bleibt als Adapter-Fallback (`to_rect_provider`) |
| A3 | umgesetzt | `stream_forward_drizzle_uniform` nutzt denselben `source_rect_of`-Pfad; der Diagnose-Pfad im Runner teilt den Whole-Set-/Single-Slot-Stash (`make_normalized_source_providers`) |
| A4 | umgesetzt | `reconstruct_aqmh_weighted` haelt im Non-Region-Fallback einen budget-geprueften Frame/Mask-Stash (`frame_stash`/`mask_stash`); der Cherry-Pick-Gate ist frame-aeusser organisiert, Hauptpass nutzt `eff_frame_region`/`eff_mask_region` |
| A5 | umgesetzt | Der Gate-Pass fuehrt pro Frame genau einen `q_map_cache->read_cached(fi)`-Zugriff aus (LRU-resident fuer den Hauptpass); Masken/Frames kommen bei aktivem Stash daraus |
| A6 | umgesetzt | Die Mask-Validierung befuellt den Stash direkt; validierte Masken werden im Hauptpass wiederverwendet statt erneut geladen |
| B1 | umgesetzt | Sequenzieller RGB-Fallback im Runner teilt einen Mask-Stash (`rgb_mask_stash`) und denselben `aqmh_cache` ueber die drei Plane-Calls; Frame-Pixeldaten bleiben bewusst plane-spezifisch |
| B2 | umgesetzt | `multiband_fusion.cpp`: Profile werden einmal dekomponiert, Baender aus den Profil-Ebenen kombiniert (kein `luma_band`-Redekomponieren pro Band) |
| B3 | umgesetzt | `smooth_alpha_b3`: Zaehler und Nenner in einer fus einzigen separablen Faltung (identische Geometrie, ein Pass statt zwei) |
| B4 | Fehleinschätzung (veraltet) | `compute_global_quality_weights_from_metrics` + `load_source_quality_metrics` existieren und sind verdrahtet (`source_quality_artifact.cpp`, `runner_forward_drizzle.cpp`); `preflight` erzwingt `star_max_corners >= 1`, das Metrics-Artefakt existiert daher immer |
| B5 | Fehleinschätzung | keine Redundanz: R/G/B/L sind unterschiedliche Daten, kein doppeltes Lesen derselben Bytes; One-Plane-Peak-Memory ist bewusstes Spool-Design |
| C1 | umgesetzt | `is_auto_reject` hoisted (`aqmh_reconstruction.cpp`) |
| C2 | umgesetzt | `canvas_valid_flat`-Array; `canvas_valid()` komplett entfernt |
| C3 | umgesetzt | `band_src_by_level`/`band_profile_by_level` (`multiband_fusion.cpp`) |
| C4 | umgesetzt | 2×2-LUT `build_cfa_channel_2x2_lut` (`source_quality_proxy.cpp`) |
| C5 | umgesetzt | `is_osc`/`bayer_pattern`/`cfa_origin_*` hoisted (`forward_drizzle.cpp`) |
| C6 | umgesetzt | `emit_fine`/`emit_medium`/Exponenten hoisted (`forward_drizzle.cpp`) |
| C7 | umgesetzt | `gw_by_fi`: `global_weight` wird einmal pro Frame vorberechnet (`aqmh_reconstruction.cpp`) |
| C8 | umgesetzt | `dx`/`dy` hoisted (`alpha_guard.cpp`) |
| D1 | umgesetzt (mit Korrektur des Ansatzes) | `mad_sigma` braucht die by-value-Kopie fuer den In-Place-Median (urspruenglicher Fix nicht tragfaehig); stattdessen `mad_sigma_mutate` mit gehisstem `dev`-Scratch dort, wo die Eingabe mutiert werden darf (`alpha_guard.cpp:27-48, 83, 124`) |
| D2 | umgesetzt | `wx`/`w`/`w2`/`A`/`B` hoisted (`forward_drizzle.cpp`) |
| D3 | umgesetzt | `result` + Profil-Ebenen + Arbeitsbuffer werden ueber Stripes wiederverwendet; nur das genutzte `[0, n)`-Prefix wird zurueckgesetzt (`forward_drizzle.cpp`) |
| D4 | umgesetzt | `reduce_window` schreibt in caller-seitige, ueber Tiles wiederverwendete Ergebnis-/Scratch-Buffer (`forward_drizzle_contrib_list.cpp`) |
| D5 | umgesetzt | `CudaProducerScratch`: `src_buf_narrowed`/`raw` u. a. werden pro Frame einmal allokiert und ueber Tiles wiederverwendet (`forward_drizzle_contrib_list.cpp`) |
| D6 | umgesetzt | `reserve` fuer `local_effective_k`/`local_margins` (`aqmh_reconstruction.cpp`) |
| E1 | umgesetzt | `build_frame_records` befuellt die Contrib-Listen in einem einzigen Raster-Pass (Count+Fill fusioniert, `forward_drizzle_contrib_list.cpp`) |
| E2 | umgesetzt | `cuda_pair_producer` laedt die Source erst nach dem Band-Cache-Miss (lazy statt vor dem Lookup) |
| E3 | umgesetzt | Affine Inverse werden pro Frame gecacht und ueber Tiles wiederverwendet (`forward_drizzle_contrib_list.cpp`) |

### Anmerkungen zu den strukturellen Umbauten (Runde 2)

- **A1/A2 (Loop-Order):** Statt die Stripe-aeussere Loop umzudrehen (grosse
  Semantik-/Parallelisierungs-Aenderung), wurde die gelesene Datenmenge pro
  (Stripe, Frame) auf die tatsaechlich gescannte Source-Box reduziert:
  `drizzle_source_scan_box` ist die exakt dieselbe Box, die der Rasterizer
  enumeriert (aus dem Enumerator extrahiert, keine zweite Implementierung).
  Banded `read_rect`-Zugriffe auf `VerifiedNormalizedSourceCache` (mmap-nahe
  Zeilen-Reads) und `SourceQualityMapCacheReader` machen den
  Stripe-Factor fuer Source- und Q-Map-I/O faktisch 1x -- ohne die
  Akkumulationsordnung zu aendern. Bit-identisch (549/549 Tests).
- **A4/A5/A6:** Die Pass-Fusion wurde als Budget-gepruefter Stash geloest:
  passt das Frame-/Mask-Set in das konfigurierte Memory-Budget, werden die
  bei der Validierung geladenen Masken und die Frames resident gehalten und
  Gate + Hauptpass lesen ausschliesslich daraus. Ueberschreitet das Set das
  Budget, bleibt das bisherige Verhalten (Region-Loader bevorzugt,
  `read_cached` fuer Q-Maps) unveraendert aktiv.
- **E2-Nuance (bestaetigt und adressiert):** `source_of` ist im Store-Build
  `VerifiedNormalizedSourceCache::load` -- bei LRU-Miss ein voller
  Frame-Read. Die lazy Ordnung (Band-Cache-Check zuerst) eliminiert den
  Aufruf auf dem Hit-Pfad vollstaendig; auf dem Miss-Pfad wird zusaetzlich
  banded via `source_rect_of` gelesen, wenn verdrahtet.
- **B5** ist keine Redundanz im eigentlichen Sinn: R/G/B/L sind
  unterschiedliche Daten, kein zweifaches Lesen derselben Bytes. Das
  Spool-Design (One-Plane-At-A-Time Peak-Memory) ist bewusst sequenziell.
  Kein Fix noetig.
- **Verbliebene Grenzen:** Der CPU-Stream-Pfad iteriert weiterhin
  Stripe-aeusser/Frame-innen -- die Reload-Kosten sind durch banded reads
  beseitigt, die Enumerations-Reihenfolge bleibt wie zuvor. Bei nicht
  verdrahteten Rect-Providern (reine `source_of`-Aufrufer) faellt der Pfad
  auf die Full-Frame-Semantik zurueck (Coverage-Check statt Exakt-Shape, da
  `to_rect_provider` volle Maps mit Origin 0 liefert).

### Zusätzlich behoben (gleiche Musterklasse, nicht in der Ursprungsliste)

- `quantile()` in `aqmh_reconstruction.cpp:26-43`: `std::sort` durch zwei
  `nth_element`-Aufrufe ersetzt (nur Ranks lo/hi werden gebraucht).
- `aqmh_select_top_k`/`aqmh_select_auto_reject`: geteilte Komparatoren
  `aqmh_score_invalid`/`aqmh_score_cmp` auf File-Scope hoisted
  (`aqmh_cherry_pick.cpp`).
- `median_of_or_nan_inplace` statt kopierendem `median_of` in
  `metrics.cpp`, `core/utils.cpp:475` (`robust_sigma_mad`) und
  `autobge.cpp` -- dieselbe Kopiervermeidung wie bei D1 motiviert.
- ByteSink-Deduplizierung nach `core/byte_sink.hpp` und
  `rgb_to_luma`-Deduplizierung nach `image/processing.hpp`
  (Duplikat-Kategorie, nicht Reload-Kategorie).

---

## Kategorie A: Daten-Reload pro Stripe/Chunk

### A1: Source-Frame wird pro Stripe neu geladen (Stream-Forward-Drizzle)

**Status (2026-09-12, Runde 2):** umgesetzt -- statt die Loop-Order
umzudrehen wird die gelesene Datenmenge reduziert: `drizzle_source_scan_box`
(aus dem Enumerator extrahiert, identische Bounds) liefert pro
(Stripe, Frame) die inverse-gemappte Source-Box; `source_rect_of` liest nur
diese Box banded (`VerifiedNormalizedSourceCache::read_rect`). Verdrahtet in
`persist_multiband_store_from_predecessors`,
`persist_forward_drizzle_from_predecessors` und dem Runner-Stash
(`make_normalized_source_providers`). Der 2/1-Pfad delegiert an dieselbe
Funktion und profitiert mit.

**Datei:** `src/reconstruction/forward_drizzle.cpp:1489-1600`
**Funktion:** `stream_forward_drizzle_uniform_and_raw`

**Pattern:**
```
for (int y = 0; y < memory.height; y += memory.rows) {   // Stripe-Loop (außen)
    for (const auto *f : prepared.frames) {               // Frame-Loop (innen)
        const Matrix2Df &source = source_of(f->source_index);  // Zeile 1595
        FrameQualityMaps qm;
        if (quality_of) {
            qm = quality_of(f->source_index);                   // Zeile 1600
        }
        // ... verarbeite Frame f für Stripe y
    }
}
```

**Problem:** Die Schleifenreihenfolge ist Stripe-außen, Frame-innen. Bei 600 Frames
und ~15-Frame LRU-Kapazität (512 MB Budget / ~33 MB pro 4K-Frame) ist die
Cache-Hit-Rate über Stripe-Grenzen hinweg ~0 %. Jeder Frame wird S-mal von Disk
gelesen, wobei S = Anzahl der Stripe-Rows.

**Implementierung im Runner:** `source_of` ist entweder:
- ein Single-Slot-Cache (`runner_phase_registration.cpp:511-520`): genau 1 Frame
  resident → 100 % Miss bei Frame-Wechsel
- `cache.load(index)` (`source_quality_artifact.cpp:385`): LRU mit
  `memory_budget_mb` (Default 512 MB → ~15 Frames bei 4K)

**Impact:** Bei 600 Frames, 4K-Auflösung, ~20 Stripe-Rows:
- 600 × 20 = 12.000 Disk-Reads statt 600
- ~400 GB I/O statt ~20 GB
- **Dominante Kosten der FORWARD_DRIZZLE-Phase**

**Lösungsansatz:**
1. **Frame-außen, Stripe-innen** Schleifenreihenfolge: Akkumuliere alle Stripe-
   Buffer gleichzeitig, verarbeite einen Frame nach dem anderen. Erfordert
   Umstrukturierung der Akkumulatoren (A/B/QA-Buffer müssen für alle Stripe
   parallel leben).
2. **Größeres LRU-Budget**: Wenn `memory_budget_mb` groß genug für alle Frames
   ist (z.B. 20 GB für 600×4K), entfällt das Problem. Konfigurierbar via
   `drizzle.memory_budget_mb`.
3. **Prefetch-Thread** (wie `aqmh_pipeline_overlap.cpp:76-79`): Asynchrones
   Vorladen des nächsten Frames während der aktuelle verarbeitet wird.

**Gegenargument:** Die Single-Slot/LRU-Architektur ist eine bewusste
Speicher-Begrenzung. Bei 600×4K Frames wären ~20 GB nötig, um alle resident zu
halten. Die aktuelle Lösung skaliert mit begrenztem RAM, zahlt aber durch
redundantes I/O.

---

### A2: Q-Map wird pro Stripe neu von Disk gelesen (kein In-Memory-Cache)

**Status (2026-09-12, Runde 2):** umgesetzt -- die Stream-Funktionen nehmen
`FrameQualityRectProvider` direkt entgegen und lesen nur die per-Stripe
Scan-Box via `read_rect` (kein Full-Extent-Adapter mehr auf dem verdrahteten
Pfad; `to_rect_provider` bleibt als Fallback). `SourceQualityMapCacheReader`
hat weiterhin bewusst keinen In-Memory-Cache -- der banded Zugriff
reduziert die dekodierte Datenmenge auf die Box, ein Cache ist nicht mehr
noetig. Der getilte CUDA-Pfad behaelt den T4b-Q-Cache
(1 `read_rect` pro Frame pro Band).

**Datei:** `src/reconstruction/source_quality_artifact.cpp:359-383`
**Funktion:** `quality_of` Lambda in `persist_multiband_store_from_predecessors`

**Pattern:**
```cpp
FrameQualityRectProvider quality_of =
    [reader=qreader.get(), need_medium,
     comp=Matrix2Df(), s0=Matrix2Df(), s1=Matrix2Df(), art=Matrix2Df()](
        std::size_t si, int y0, int y1, int x0, int x1) mutable -> FrameQualityMaps {
    comp = reader->read_rect("composite", si, y0, y1, x0, x1);  // Disk-Read
    if (have_s0) s0 = reader->read_rect("scale_0", si, y0, y1, x0, x1);
    if (have_s1) s1 = reader->read_rect("scale_1", si, y0, y1, x0, x1);
    if (have_art) art = reader->read_rect("artifact", si, y0, y1, x0, x1);
    // ...
};
```

**Problem:** `SourceQualityMapCacheReader` hat **keinen In-Memory-Cache**.
`read_rect` öffnet die `.bin`-Datei und dekodiert Storage-Grid-Zellen bei jedem
Aufruf. Der Kommentar in Zeile 357-358 bestätigt: *"The comp/s0/... members are
reused allocations, NOT a cache"*.

**Impact:** Bei 600 Frames, 4 Q-Map-Streams (composite/scale_0/scale_1/artifact),
~20 Stripe-Rows:
- 600 × 4 × 20 = 48.000 `.bin`-Datei-Öffnungen
- Die I/O-Counter (`bin_loads_`, `bin_cells_decoded_`) werden pro Aufruf
  inkrementiert

**Kontrast zu AQMH:** `metrics::QualityMapCache` (aqmh_quality_map_cache.hpp:44)
hat ein LRU-Cache mit `read_cached(fi)`. Die Forward-Drizzle-Pipeline nutzt
jedoch `SourceQualityMapCacheReader`, der keinen Cache hat.

**Lösungsansatz:** In-Memory-LRU-Cache für `SourceQualityMapCacheReader`
analog zu `QualityMapCache::read_cached()`, oder Prefetch-Thread analog zu
`aqmh_pipeline_overlap.cpp`.

---

### A3: Source-Frame und Q-Map werden im `stream_forward_drizzle_uniform` pro Stripe geladen

**Status (2026-09-12, Runde 2):** umgesetzt --
`stream_forward_drizzle_uniform` nutzt denselben
`source_rect_of`-Banded-Read-Pfad; beide Runner-Diagnose-Call-Sites teilen
`make_normalized_source_providers` (Whole-Set-Stash bei Budget, sonst
Single-Slot). Nur Diagnose-/Coverage-Pfad.

**Datei:** `src/reconstruction/forward_drizzle.cpp:872-914`
**Funktion:** `stream_forward_drizzle_uniform`

**Pattern:**
```cpp
for (int y = 0; y < memory.height; y += memory.rows) {
    // ...
    for (const auto *f : prepared.frames) {
        const Matrix2Df &source = source_of(f->source_index);  // Zeile 884
        // ... verarbeite Frame für Stripe
    }
}
```

**Problem:** Gleiche Schleifenreihenfolge wie A1, aber ohne `quality_of`
(diese Funktion hat keinen Q-Map-Pfad). Wird für die uniform-store Vorschau
und Coverage-Analyse verwendet.

**Impact:** Gleiches Muster wie A1, aber ohne Q-Map-Overhead. Die Funktion
wird für diagnostische Zwecke aufgerufen, nicht für die Hauptrekonstruktion.

---

### A4: AQMH-Reconstruction lädt Frame/Mask/Q-Map pro y0-Chunk neu

**Status (2026-09-12, Runde 2):** umgesetzt -- im Non-Region-Fallback haelt
`reconstruct_aqmh_weighted` einen budget-geprueften Frame/Mask-Stash
(`frame_stash`/`mask_stash`, Gate: Gesamt-Set <= konfiguriertes Budget);
Region-Loader bleiben bevorzugt und laufen ueber `eff_frame_region`/
`eff_mask_region`. Dasselbe Muster im OpenCL-/CUDA-Host-Feed wurde
mitgefuehrt (N3/N4).

**Datei:** `src/reconstruction/aqmh_reconstruction.cpp:355-379`
**Funktion:** `reconstruct_aqmh_weighted`

**Pattern:**
```cpp
for (int y0 = 0; y0 < height; y0 += chunk_rows) {        // Chunk-Loop (außen)
    for (std::ptrdiff_t fi_signed = 0; ...; ++fi_signed) { // Frame-Loop (innen)
        const bool frame_ok = load_frame_region
            ? load_frame_region(fi, y0, rows, frame)
            : load_frame(fi, frame);                       // Zeile 357-358
        Matrix2Df q = q_map_cache->read_cached(fi);        // Zeile 362-363
        load_frame_valid_mask(fi, fm);                     // Zeile 374-375
    }
}
```

**Problem:** Wenn die Region-Loader nicht verfügbar sind, werden volle
Frame/Mask/Q-Map pro Chunk neu geladen. `read_cached` hat zwar ein LRU, aber
`load_frame` und `load_frame_valid_mask` haben keinen Cache.

**Impact:** Per-Frame pro Chunk. Bei 600 Frames und ~10 Chunks:
- 6.000 Frame-Loads + 6.000 Mask-Loads (ohne Region-Loader)
- Mit Region-Loader: nur die relevanten Zeilen, aber immer noch pro Chunk
  neu gelesen

**Lösungsansatz:** Frame-outer/Chunk-inner Loop-Order, oder
Single-Frame-Cache wie in `runner_phase_registration.cpp:509-520`.

---

### A5: Cherry-Pick-Pre-Pass und Haupt-Pass laden dieselben Q-Maps/Masks doppelt

**Status (2026-09-12, Runde 2):** umgesetzt -- der Cherry-Pick-Gate wurde
als frame-aeusserer Pass umgebaut: pro Frame genau ein
`q_map_cache->read_cached(fi)` (LRU-resident fuer den Hauptpass) und --
bei aktivem Stash -- Masken/Frames aus dem Stash statt erneuten Loads.
Keine doppelten (fi, y0)-Reads mehr zwischen Gate und Hauptpass.

**Datei:** `src/reconstruction/aqmh_reconstruction.cpp:188-224` (Cherry-Pick)
und `aqmh_reconstruction.cpp:361-375` (Haupt-Pass)

**Problem:** Wenn Cherry-Pick aktiviert ist, durchläuft der Cherry-Pick-Rankable-Pass
jedes (fi, y0) und lädt `q_map_cache->read_region(fi, y0, rows)` und
`load_frame_valid_mask_region(fi, y0, rows, fm)`. Der Haupt-Pass lädt dann
dieselben Daten für dieselben (fi, y0) erneut.

**Impact:** Verdopplung der Q-Map- und Mask-I/O wenn Cherry-Pick aktiv ist.

**Lösungsansatz:** Fusion der beiden Pässe, oder Zwischenspeicher der
Q-Map/Mask-Daten zwischen den Pässen.

---

### A6: Mask-Validierung lädt Mask, verwirft sie, lädt sie im Haupt-Pass neu

**Status (2026-09-12, Runde 2):** umgesetzt -- die Mask-Validierung befuellt
den Stash direkt; der Hauptpass verwendet die validierten Masken wieder
statt sie neu zu laden.

**Datei:** `src/reconstruction/aqmh_reconstruction.cpp:164-173` (Validierung)
und `aqmh_reconstruction.cpp:371-375` (Haupt-Pass)

**Problem:** Die Validierungsschleife lädt `full_mask` pro Frame, um
`frame_mask_compatible[fi]` zu berechnen. Die Maske wird verworfen. Der
Haupt-Chunk-Loop lädt die Maske dann erneut für jeden Chunk.

**Impact:** Mindestens eine zusätzliche vollständige Mask-Load pro Frame,
plus die Per-Chunk-Reloads aus A4.

---

## Kategorie B: Daten-Reload pro Band/Kanal

### B1: AQMH CPU-Fallback rekonstruiert R, G, B sequenziell mit gleichen Q-Maps/Masks

**Status (2026-09-12, Runde 2):** umgesetzt -- der sequenzielle RGB-Fallback
teilt einen Mask-Stash (`rgb_mask_stash`, Budget: ein Viertel des
konfigurierten Limits) und denselben `aqmh_cache` ueber die drei
Plane-Calls; Q-Maps/Masken werden hoechstens einmal dekodiert.
Frame-Pixeldaten bleiben bewusst plane-spezifisch (R/G/B sind verschiedene
Daten, vgl. B5). CUDA-Pfad unveraendert `run_planes_rgb`.

**Datei:** `apps/runner_phase_aqmh_reconstruction.cpp:720-775`

**Pattern:**
```cpp
auto reconstruct_rgb_plane = [&](const DiskCacheFrameStore &plane_store,
    Matrix2Df &plane_out, const char *channel_name, int channel_index) -> bool {
    auto frame_loader = [&](size_t fi, Matrix2Df &output) -> bool {
        output = plane_store.load(fi);  // lädt Kanal-Frame von Disk
    };
    auto plane_recon = aqmh_reconstruction_ops.reconstruct_aqmh(
        frames.size(), frame_loader, aqmh_cache.get(),  // gleicher aqmh_cache
        ..., aqmh_mask_loader, ...);                     // gleicher mask_loader
};

rgb_ok =
    reconstruct_rgb_plane(*prewarped_frames_r, out.output_R, "R", 0) &&
    reconstruct_rgb_plane(*prewarped_frames_g, out.output_G, "G", 1) &&
    reconstruct_rgb_plane(*prewarped_frames_b, out.output_B, "B", 2);
```

**Problem:** Jeder `reconstruct_aqmh`-Aufruf für R, G, B durchläuft alle Frames
und fragt denselben `aqmh_cache` und `aqmh_mask_store` ab. Die Q-Maps und Masks
sind kanalagnostisch, werden aber dreimal geladen (einmal pro Kanal).

**Impact:** Per-Band (×3 auf CPU). Die Luma-Pass + 3 RGB-Pässe = 4× Sweep über
Frames/Masks/Q-Maps.

**Kontrast:** Die CUDA-Pfad verwendet `session.run_planes_rgb` mit
`{r_loader, g_loader, b_loader}` und verarbeitet alle drei Kanäle in einem
Sweep.

**Lösungsansatz:** CPU-Pfad sollte alle drei Kanäle in einem Sweep
verarbeiten, oder die geladenen Q-Maps/Masks zwischen den Kanal-Pässen
zwischenspeichern.

---

### B2: `luma_band` dekomponiert `raw`/`fine`/`medium` für jedes Band neu

**Status (2026-09-12, Runde 2):** umgesetzt -- die Profile (`raw`/`fine`/
`medium`) werden einmal pro Kanal dekomponiert und die Baender aus den
Profil-Dekompositionen kombiniert; `luma_band` fuehrt keine eigene
Voll-Dekomposition pro Band mehr aus (`multiband_fusion.cpp`).

**Datei:** `src/reconstruction/multiband_fusion.cpp:125-152`
**Aufgerufen von:** `fuse_multiband:186-188`

**Pattern:**
```cpp
void luma_band(const ForwardDrizzleUniformResult &profile, ...,
               int band0, std::vector<float> &out_detail, ...) {
    std::array<AtrousDecomposition, 3> dec;
    for (int c = 0; c < nch; ++c)
        dec[c] = atrous_decompose(ch[c]->value, ch[c]->support, w, h, levels);
    // ... verwendet nur dec[c].bands[band0]
}

// In fuse_multiband:
for (int j = 1; j <= L; ++j) {
    luma_band(raw, mode, width, height, L, j - 1, dr_luma, sr);      // Zeile 187
    luma_band(profile, mode, width, height, L, j - 1, dp_luma, sp); // Zeile 188
}
```

**Problem:** `luma_band` führt eine vollständige `a trous`-Dekomposition über
alle `levels` durch, verwendet aber nur `bands[band0]`. `fuse_multiband` ruft
`luma_band` L-mal auf → `raw` wird L-mal vollständig dekomponiert, `fine`/`medium`
werden L-mal bzw. (L-1)-mal dekomponiert.

Zusätzlich dekomponiert `fuse_multiband_channel` (Zeile 55-61) `raw`/`fine`/`medium`
bereits einmal pro Kanal. Die Dekompositionen in `luma_band` sind redundant.

**Impact:** Per-Band, Full-Image. Die `a trous`-Dekomposition ist O(w×h×levels).
Bei L=4 Levels und 3 Kanälen:
- `raw`: 4× dekomponiert in `luma_band` + 1× in `fuse_multiband_channel` = 5×
- `fine`: 1× in `luma_band` + 1× in `fuse_multiband_channel` = 2×
- `medium`: 3× in `luma_band` + 1× in `fuse_multiband_channel` = 4×

**Lösungsansatz:** Einmal dekomponieren und `bands[band0]` indizieren.
Die Dekompositionen aus `fuse_multiband_channel` an `luma_band` weiterreichen.

---

### B3: `smooth_alpha_b3` führt B3-Faltung zweimal für identische Geometrie aus

**Status (2026-09-12, Runde 2):** umgesetzt -- Zaehler- und Nenner-Faltung
laufen fusioniert in einer separablen `conv_axis`-Runde (identische
Geometrie/Labels, ein Pass statt zwei).

**Datei:** `src/reconstruction/alpha_guard.cpp:174-215`

**Pattern:**
```cpp
auto masked_num = [&](bool weighted) {
    std::vector<double> src(n, 0.0);
    // src[i] = weighted ? alpha_guarded[i] : 1.0
    std::vector<double> hx(n, 0.0);
    // horizontale B3-Faltung
    std::vector<double> out(n, 0.0);
    // vertikale B3-Faltung
    return out;
};

const auto num = masked_num(true);   // Zeile 214
const auto den = masked_num(false);  // Zeile 215
```

**Problem:** `masked_num` wird zweimal aufgerufen. Beide Aufrufe:
1. Allokieren `src`, `hx`, `out` (je n doubles)
2. Führen horizontale + vertikale B3-Faltung über dieselbe `label`/`support`-Geometrie aus
3. Der einzige Unterschied: `src[i]` ist `alpha_guarded[i]` vs `1.0`

**Impact:** Per `smooth_alpha_b3`-Aufruf (1× pro Band in `fuse_multiband`).
Doppelte Full-Image-Faltung + 6 Full-Image-Allokationen.

**Lösungsansatz:** Fused Faltung: berechne `num` und `den` in einem einzigen
horizontalen + vertikalen Pass mit zwei `src`-Werten pro Pixel.

---

### B4: `compute_source_quality_proxy_v1` wird in zwei Phasen unabhängig ausgeführt

**Status (2026-09-12):** Fehleinschätzung (veraltet) -- bereits umgesetzt:
`persist_source_quality_artifact` hat eine `metrics_path`-Überladung, die
`source_quality_metrics-v1.json` lädt und
`compute_global_quality_weights_from_metrics` nutzt
(`source_quality_artifact.cpp:142-195`); der Runner übergibt `metrics_path`
(`runner_forward_drizzle.cpp:431-434`). Da `preflight`
`star_max_corners >= 1` erzwingt, existiert das Metrics-Artefakt in diesem
Fluss immer.

**Datei:** `src/reconstruction/source_quality_map_cache.cpp:646-648`
und `src/reconstruction/global_quality.cpp:47-48`

**Problem:** Sowohl `build_source_quality_map_cache` als auch
`compute_global_quality_weights` rufen `compute_source_quality_proxy_v1` für
dieselben Source-Frames auf. Die SQM-Phase schreibt bereits
`source_quality_metrics-v1.json`, aber die Global-Quality-Phase liest diese
nicht und rechnet neu.

**Teil-Lösung bereits vorhanden:** Der Runner-Kommentar in
`runner_forward_drizzle.cpp:429-430` sagt: *"T3: use the pre-computed metrics
from SOURCE_QUALITY_MAPS instead of reloading and re-running
compute_source_quality_proxy_v1 per frame."* und übergibt `metrics_path`.
Aber `global_quality.cpp` selbst hat keinen Pfad, um die Metrics zu laden.

**Impact:** Per-Frame. Die Proxy-Berechnung ist eine der schwereren
Per-Source-Operationen (B3-Spline-Blur, MAD, Edge-Aware-Green).

---

### B5: Multiband-Delivery liest Spool-Planes pro Kanal einzeln

**Status (2026-09-12):** Fehleinschätzung -- keine Redundanz im eigentlichen
Sinn: R/G/B/L sind unterschiedliche Daten, kein doppeltes Lesen derselben
Bytes. Sequentielles, aber notwendiges I/O; das One-Plane-Peak-Memory-Design
ist bewusst gewählt. Kein Fix nötig.

**Datei:** `apps/runner_forward_drizzle.cpp:699-708`

**Pattern:**
```cpp
auto emit_set = [&](const std::string &prefix, const std::string &candidate){
    for (std::size_t c = 0; c < ch.size(); ++c) {
        const auto plane = reconstruction::read_candidate_spool_plane(
            spool, candidate, static_cast<int>(c));
        io::write_fits_float_rows(p, plane, spool.height, spool.width, h);
    }
};
```

**Problem:** Jeder Kanal (R/G/B oder L) wird separat aus dem Spool gelesen und
geschrieben. `emit_set` wird für `raw`, den selektierten Kandidaten und
optional `uniform`/`multiband` aufgerufen.

**Impact:** Per-Kanal. Niedrig bis mittel — das Spool-Design ist für
One-Plane-At-A-Time Peak-Memory, aber das I/O ist wiederholt.

---

## Kategorie C: Redundante Rekursion in Pixel-Loops

### C1: `cfg.cherry_pick_mode` String-Vergleich pro Pixel

**Status (2026-09-12):** umgesetzt -- `is_auto_reject` einmal am
Funktionsanfang ausgewertet (`aqmh_reconstruction.cpp:191`, Nutzung an
225, 262, 531).

**Datei:** `src/reconstruction/aqmh_reconstruction.cpp:218-223, 249-253, 516-524`

**Problem:** `if (cfg.cherry_pick_mode == "auto_reject")` wird in Per-Pixel-Loops
ausgewertet. Der String ist für den gesamten Funktionsaufruf invariant.

**Lösung:** `const bool is_auto_reject = (cfg.cherry_pick_mode == "auto_reject");`
am Funktionsanfang.

**Impact:** Per-Pixel, hot path. String-Vergleich im innersten Loop.

---

### C2: `canvas_valid()` mit invarianten Checks pro Pixel pro Frame

**Status (2026-09-12):** umgesetzt -- `canvas_valid_flat`-Array in beiden
betroffenen Funktionen; `canvas_valid()` ist komplett entfernt
(`aqmh_reconstruction.cpp:66-71, 114, 182-186, 252, 260`).

**Datei:** `src/reconstruction/aqmh_reconstruction.cpp:110-113, 240-241, 247`

**Problem:** `canvas_valid(canvas_mask, width, height, x, y)` wird pro Pixel
aufgerufen und prüft Bounds + `mask.size()` bei jedem Aufruf.

**Kontrast:** `reconstruct_aqmh_weighted` baut bereits `canvas_valid_flat`
(Zeile 177-182) für den Haupt-Pass, aber der Cherry-Non-Region-Pass nutzt
noch die Funktionsform.

**Lösung:** Flat-Validity-Array einmal pro Chunk aufbauen, direkte Indexierung.

---

### C3: `band_source(j, L)` und `df`/`dm`-Pointer pro Pixel

**Status (2026-09-12):** umgesetzt -- `band_src_by_level`- und
`band_profile_by_level`-Arrays vor dem Pixel-Loop
(`multiband_fusion.cpp:80-86`).

**Datei:** `src/reconstruction/multiband_fusion.cpp:87-95`

**Problem:** `const BandSource src = band_source(j, L);` und
`const AtrousDecomposition &pd = src == BandSource::kFine ? df : dm;`
werden im Per-Pixel-Loop ausgewertet, hängen aber nur von `j` und `L` ab.

**Lösung:** `std::array<BandSource, L>` und `std::array<const AtrousDecomposition*, L>`
vor dem Pixel-Loop aufbauen.

**Impact:** Per-Pixel, per-Band. Klein, aber trivial zu fixen.

---

### C4: `cfa_channel_for_source_pixel` pro Pixel neu berechnet

**Status (2026-09-12):** umgesetzt -- 2×2-`CfaChannel`-LUT
`build_cfa_channel_2x2_lut` vor den Loops
(`source_quality_proxy.cpp:14-29, 140, 151, 172`).

**Datei:** `src/reconstruction/source_quality_proxy.cpp:127-141, 151-158`

**Problem:** Die CFA-Kanal-Farbe jedes Pixels wird aus Bayer-Pattern/Origin
neu berechnet. Das Resultat ist eine periodische 2×2-Maske.

**Lösung:** 2×2 `CfaChannel`-Lookup vor den Loops aufbauen, Indexierung
über `(x&1, y&1)`.

**Impact:** Per-Source-Pixel, O(pixels). Hot für große OSC-Inputs.

---

### C5: `plan.color_mode` und CFA-Parameter pro Source-Pixel gelesen

**Status (2026-09-12):** umgesetzt -- in lokale Variablen kopiert
(`forward_drizzle.cpp:566-581`).

**Datei:** `src/reconstruction/forward_drizzle.cpp:571-574`

**Problem:** `plan.color_mode`, `plan.bayer_pattern`, `plan.cfa_origin_x/y`
werden im Per-Source-Pixel-Hot-Loop von `enumerate_drizzle_stripe_leaf_cells`
gelesen.

**Lösung:** In lokale Variablen kopieren vor den Loops.

**Impact:** Per-Pixel im Rasterizer. Der Compiler mag inline, aber die
Struct-Reads sind im hot path.

---

### C6: `cfg.emit_*` und Exponenten pro akzeptiertem Clip-Kandidat

**Status (2026-09-12):** umgesetzt -- `emit_fine`/`emit_medium`/
`fine_quality_exponent`/`medium_quality_exponent` hoisted
(`forward_drizzle.cpp:1197-1211`).

**Datei:** `src/reconstruction/forward_drizzle.cpp:1189-1192`

**Problem:** `cfg.emit_fine`, `cfg.emit_medium`, `cfg.fine_quality_exponent`,
`cfg.medium_quality_exponent` werden im Per-Pixel-Kandidaten-Loop geprüft.

**Lösung:** In lokale `bool`/`double` Variablen vor dem Loop kopieren.

---

### C7: `global_weight(global_weights, fi)` mehrfach für denselben `fi`

**Status (2026-09-12, Runde 2):** umgesetzt -- `gw_by_fi` berechnet
`global_weight` einmal pro Frame; die bisherigen Aufrufe an
`aqmh_reconstruction.cpp:207, 243, 399`.

**Datei:** `src/reconstruction/aqmh_reconstruction.cpp:200, 236, 386`

**Problem:** `global_weight(global_weights, fi)` wird in drei separaten Loops
für denselben `fi` aufgerufen. Der Wert hängt nur von `global_weights` und `fi`
ab, die während des gesamten Aufrufs konstant sind.

**Lösung:** `std::vector<float> per_frame_gw(frame_count)` einmal am Anfang
berechnen.

---

### C8: `dx[4]`/`dy[4]` Arrays pro Stack-Pop in BFS

**Status (2026-09-12):** umgesetzt -- vor die BFS hoisted
(`alpha_guard.cpp:150-153`).

**Datei:** `src/reconstruction/alpha_guard.cpp:159-160`

**Problem:** `const int dx[4] = {-1, 1, 0, 0};` und `const int dy[4] = {0, 0, -1, 1};`
werden bei jedem Stack-Pop neu deklariert.

**Lösung:** Einmal vor der BFS deklarieren.

**Impact:** Per-Pixel-BFS. Vernachlässigbar, aber trivial zu fixieren.

---

## Kategorie D: Per-Iteration Allokationen

### D1: `mad_sigma` kopiert Vektor by-value und allokiert `dev` pro Aufruf

**Status (2026-09-12, Runde 2):** umgesetzt mit korrigiertem Ansatz -- der
vorgeschlagene by-ref/move-Fix ist nicht tragfaehig, da `mad_sigma` die
eigene destruktible Kopie fuer den In-Place-Median zwingend braucht.
Stattdessen: `mad_sigma_mutate(values, dev)` mit gehisstem `dev`-Scratch
dort, wo die Eingabe mutiert werden darf (`alpha_guard.cpp:27-48`);
`dr_vals`/`dp_vals`/`mix`/`mad_dev` sind Caller-seitig hoisted. Die
verbleibende by-value-Stelle (`dr_vals`, dessen Ordering erhalten bleiben
muss) ist semantisch notwendig, keine Redundanz.

**Datei:** `src/reconstruction/alpha_guard.cpp:29-36`

**Problem:**
```cpp
double mad_sigma(std::vector<float> values) {  // by-value → Kopie
    const double med = median_inplace(values);
    std::vector<float> dev;                     // neue Allokation
    dev.reserve(values.size());
    for (float v : values) dev.push_back(...);
    return 1.4826 * median_inplace(dev);
}
```

`mad_sigma` wird pro Pixel aufgerufen:
- `mad_sigma(dr_vals)` 1× pro Pixel (Zeile 97)
- `mad_sigma(mix)` 1 + `bisection_iters` × pro Pixel (Zeile 111/125)

Der Kommentar in Zeile 100-106 bestätigt: *"mad_sigma takes its argument by
value regardless (it needs its own destructible copy for the in-place median)"*.

**Lösung:** `mad_sigma(std::vector<float>& values)` mit In-Place-Median.
Aufrufer behält eine Kopie wenn der Originalvektor erhalten bleiben muss.
Alternativ: `mad_sigma_inplace(float* begin, size_t n, std::vector<float>& scratch)`.

**Impact:** Per-Pixel, very hot. Kosten wachsen mit `window_radius²` und
`bisection_iters`.

---

### D2: Per-Band `wx`/`w`/`w2`/`A`/`B` Allokationen in `stream_forward_drizzle_uniform`

**Status (2026-09-12):** umgesetzt -- Buffer-Deklarationen aus dem
Stripe-Loop hoisted (`forward_drizzle.cpp:880-886`).

**Datei:** `src/reconstruction/forward_drizzle.cpp:875-882`

**Problem:**
```cpp
for (int y = 0; y < memory.height; y += memory.rows) {
    std::array<std::vector<double>, 3> wx, w, w2, A, B;
    for (int c = 0; c < channels; ++c) {
        wx[c].assign(n, 0);  // neue Allokation pro Stripe
        w[c].assign(n, 0);
        // ...
    }
}
```

**Kontrast:** `stream_forward_drizzle_uniform_and_raw` (Zeile 1459-1479)
hat diese bereits als hoisted Buffer außerhalb des Stripe-Loops mit
`resize`/`fill_n` statt `assign`.

**Lösung:** Dasselbe Pattern wie in `stream_forward_drizzle_uniform_and_raw`
anwenden: Buffer einmal auf Max-Größe allokieren, nur `[0, n)`-Prefix pro
Stripe nullen.

---

### D3: Per-Band `ForwardDrizzleUniformAndRawResult` Profil-Allokationen

**Status (2026-09-12, Runde 2):** umgesetzt -- `result` und die
Profil-Ebenen werden ueber Stripes wiederverwendet; pro Stripe wird nur das
genutzte `[0, n)`-Prefix zurueckgesetzt (Sink-Vertrag bleibt: synchrone
Konsumation pro Stripe).

**Datei:** `src/reconstruction/forward_drizzle.cpp:1493-1509`

**Problem:** `result` und alle Profil-Ebenen (`L`, `R`, `G`, `B`, `fine`, `medium`)
werden in jeder Stripe-Iteration neu allokiert.

**Lösung:** Einmal mit Max-Band-Größe allokieren, pro Stripe zurücksetzen.

---

### D4: Per-Tile `cand`/`counts`/`r` Allokationen in `reduce_window`

**Status (2026-09-12, Runde 2):** umgesetzt -- `reduce_window` schreibt in
caller-seitige Buffer, die ueber Tiles wiederverwendet werden
(`forward_drizzle_contrib_list.cpp`).

**Datei:** `src/reconstruction/forward_drizzle_contrib_list.cpp:605-643`

**Problem:**
```cpp
ForwardDrizzleUniformAndRawResult r;
std::vector<std::vector<ClipCandidate>> cand(channels);
std::vector<std::vector<std::size_t>> counts(channels);
for (int c = 0; c < channels; ++c) {
    cand[c].resize(n * frame_count);  // n = tile_w * rows
    counts[c].assign(n, 0);
}
```

Wird pro Tile aufgerufen. `n * frame_count` `ClipCandidate`s pro Kanal.

**Lösung:** Pre-allokieren auf Max-Tile-Größe, wiederverwenden.

---

### D5: Per-Tile `src_buf_narrowed` und `raw` im CUDA-Producer

**Status (2026-09-12, Runde 2):** umgesetzt -- `CudaProducerScratch` haelt
`src_buf_narrowed`/`raw` u. a. pro Frame und ueber Tiles
(`forward_drizzle_contrib_list.cpp`).

**Datei:** `src/reconstruction/forward_drizzle_contrib_list.cpp:1050, 1072`

**Problem:** `std::vector<float> src_buf_narrowed(...)` und
`std::vector<CudaDrizzleContribRecord> raw(cap)` werden pro Tile konstruiert.

**Lösung:** Einmal pro Frame/Producer allokieren, wiederverwenden.

---

### D6: `local_effective_k`/`local_margins` ohne `reserve`

**Status (2026-09-12):** umgesetzt -- `reserve(pixel_count / num_threads)`
vor dem Pixel-Loop (`aqmh_reconstruction.cpp:485-488`).

**Datei:** `src/reconstruction/aqmh_reconstruction.cpp:464-466, 531-532, 564-565`

**Problem:** `local_effective_k` und `local_margins` werden leer deklariert und
mit `push_back` pro Pixel gefüllt, ohne `reserve`.

**Lösung:** `reserve(pixel_count / num_threads)` vor dem Pixel-Loop.

---

## Kategorie E: Strukturelle Cross-Pass-Redundanz

### E1: `rasterize_drizzle_stripe` wird zweimal aufgerufen (Count + Fill)

**Status (2026-09-12, Runde 2):** umgesetzt -- `build_frame_records` fuellt
die Records in einem einzigen Raster-Pass (Fill-on-demand statt Count+Fill;
`forward_drizzle_contrib_list.cpp`).

**Datei:** `src/reconstruction/forward_drizzle_contrib_list.cpp:75-109`
**Funktion:** `build_frame_records`

**Problem:**
```cpp
// Count-Pass
rasterize_drizzle_stripe(plan, f, g.scale, cfg.pixfrac, y_begin, rows,
    [&](int sx, int sy, int, int, std::size_t, double) {
        if (std::isfinite(static_cast<double>(src(sy, sx)))) ++count;
    }, sub, xb, win_w);

// Fill-Pass (gleiche Geometrie, gleiche src-Checks)
rasterize_drizzle_stripe(plan, f, g.scale, cfg.pixfrac, y_begin, rows,
    [&](int sx, int sy, int c, int leaf, std::size_t i, double k) {
        // ... fülle records
    }, sub, xb, win_w);
```

Die gleiche Leaf-Geometrie, Polygon-Overlap-Flächen, Target-Indizes und
`src(sy,sx)`-Finite-Checks werden zweimal berechnet. Der Count-Pass wird nur
für `reserve` verwendet.

**Lösung:** Safe upper-bound `reserve` + Single-Pass, oder Geometrie einmal
produzieren und zwischenspeichern.

**Impact:** Per-Frame. Verdoppelt die CPU-Rasterisierungsarbeit.

---

### E2: `cuda_pair_producer` ruft `source_of` bedingungslos auf, auch bei Cache-Hit

**Status (2026-09-12, Runde 2):** umgesetzt -- `cuda_pair_producer` laedt
die Source erst nach dem Band-Cache-Miss (lazy). Kostennuance aus dem
Nachtrag bestaetigt: `source_of` ist `VerifiedNormalizedSourceCache::load`
-- LRU-Hit nur zwei Stat-Syscalls, LRU-Miss voller Frame-Read. Die lazy
Ordnung eliminiert den Aufruf auf dem Hit-Pfad; auf dem Miss-Pfad wird
banded via `source_rect_of` gelesen, wenn verdrahtet.

**Datei:** `src/reconstruction/forward_drizzle_contrib_list.cpp:934-1022`

**Problem:**
```cpp
const Matrix2Df &src = source_of(f.source_index);  // Zeile 934 — immer aufgerufen
// ... dann Cache-Lookup
if (src_cache_stats) {
    auto it = src_cache->find(fo);
    if (it != src_cache->end() && ...) {
        src_buf_full = it->second.buf.data();  // Cache-Hit: src nicht verwendet
    }
}
```

`source_of` wird aufgerufen, bevor der Cache-Lookup stattfindet. Bei Cache-Hit
wird `src` nur für einen Shape-Check verwendet, der bereits beim ursprünglichen
Load durchgeführt wurde.

**Lösung:** `source_of` in den Cache-Miss-Zweig verschieben, oder Cache-Lookup
vorher durchführen.

**Impact:** Per-Tile pro Frame auf dem CUDA-Pfad. Wenn `source_of` I/O ist,
redundante Frame-Loads.

---

### E3: Affine Inverse und Source-Y-Band pro Tile neu berechnet

**Status (2026-09-12, Runde 2):** umgesetzt -- die affine Inverse wird pro
(Frame, Band) gecacht und ueber Tiles wiederverwendet; die Source-Band-
Ladung selbst liegt bereits in `CachedSourceBand`
(`forward_drizzle_contrib_list.cpp`).

**Datei:** `src/reconstruction/forward_drizzle_contrib_list.cpp:952-972`

**Problem:** `registration::invert_affine_2x3(s2c, ...)` und die Source-Y-Band-
Berechnung werden pro Tile durchgeführt, hängen aber nur vom Frame und Stripe ab,
nicht vom Tile-X-Window.

**Lösung:** Einmal pro Frame pro Band berechnen, in `CachedSourceBand` speichern.

**Impact:** Per-Tile pro Frame. `invert_affine_2x3` ist die teuerste Operation.

---

## Priorisierung

| Prio | Fund | Beschreibung | Geschätzter Impact | Status (2026-09-12, final) |
|------|------|-------------|-------------------|---------------------|
| **P0** | A1 | Source-Frame pro Stripe neu geladen | 600×S Disk-Reads | umgesetzt (banded reads via Scan-Box) |
| **P0** | A2 | Q-Map pro Stripe ohne In-Memory-Cache | 600×4×S .bin-Opens | umgesetzt (banded `read_rect`; CUDA-Tiled-Pfad hat T4b-Q-Cache) |
| **P1** | B2 | `luma_band` dekomponiert L-mal redundant | L× full-image Dekomposition | umgesetzt (Profile einmal dekomponiert) |
| **P1** | E1 | `rasterize_drizzle_stripe` Count+Fill doppelt | 2× CPU-Rasterisierung | umgesetzt (Single-Pass-Fill) |
| **P1** | B1 | AQMH CPU RGB 3× Sweep mit gleichen Q-Maps | 3× Frame/Mask/Q-Map I/O | umgesetzt (geteilter Mask-Stash + Q-Cache) |
| **P1** | B3 | `smooth_alpha_b3` 2× B3-Faltung für gleiche Geometrie | 2× full-image Faltung | umgesetzt (fusionierter Pass) |
| **P2** | A4 | AQMH Frame/Mask pro Chunk ohne Cache | 600×C Frame-Loads | umgesetzt (Budget-Stash im Non-Region-Fallback) |
| **P2** | A5 | Cherry-Pick + Haupt-Pass laden Q-Maps doppelt | 2× Q-Map I/O | umgesetzt (frame-aeusserer Gate-Pass + Stash) |
| **P2** | E2 | `source_of` bei Cache-Hit trotzdem aufgerufen | Redundante Frame-Loads | umgesetzt (lazy nach Cache-Miss) |
| **P2** | E3 | Affine Inverse pro Tile neu | Per-Tile `invert_affine` | umgesetzt (pro-Frame-Cache) |
| **P2** | D1 | `mad_sigma` by-value + `dev`-Allokation pro Pixel | Per-Pixel Kopie+Allok | umgesetzt (`mad_sigma_mutate` + gehisster `dev`-Scratch) |
| ~~P3~~ | B4 | `compute_source_quality_proxy_v1` 2× in SQM+GQ | Per-Frame Proxy-Neuberechnung | Fehleinschätzung: bereits umgesetzt (Metrics-Artefakt) |
| **P3** | C1-C8 | Invariante Werte in Pixel-Loops | Per-Pixel Overhead | C1-C8 umgesetzt |
| **P3** | D2-D6 | Per-Iteration Allokationen | Per-Stripe/Tile/Pixel malloc | D2-D6 umgesetzt |
| ~~P3~~ | B5 | Multiband-Delivery liest Spool-Planes pro Kanal | -- | Fehleinschätzung: keine Redundanz (versch. Daten) |

## Architekturelle Beobachtungen

1. **AQMH-Pfad hat Prefetch + LRU-Cache** (`aqmh_pipeline_overlap.cpp`,
   `QualityMapCache::read_cached`), Forward-Drizzle-Pfad hat **keinen
   Q-Map-Cache** (`SourceQualityMapCacheReader` liest immer von Disk).

2. **CUDA-Pfad hat `CachedSourceBand`** für Source-Frames, CPU-Pfad hat
   nur Single-Slot-Cache oder LRU mit begrenztem Budget.

3. **`stream_forward_drizzle_uniform_and_raw` hat bereits hoisted Buffer**
   (Zeile 1459-1479), aber `stream_forward_drizzle_uniform` nicht (Zeile 875-882).

4. **Die Schleifenreihenfolge Stripe-außen/Frame-innen** ist die
   Architektur-Entscheidung, die A1/A2/A3/A4 verursacht. Gelöst wurde sie
   nicht durch Umkehrung der Reihenfolge, sondern durch banded reads auf die
   jeweils gescannte Source-Box (`drizzle_source_scan_box`) bzw. den
   AQMH-Frame/Mask-Stash -- die Enumerations- und Akkumulationsordnung ist
   unveraendert, die gelesene Datenmenge pro (Stripe, Frame) sinkt auf die
   Box.

5. **`prepare_drizzle_frames` wird bereits einmal pro Build aufgerufen**
   (§4.4-Kommentar in `drizzle_profile_store.cpp:505-510`), nicht pro Band.
   Dies war ein früheres Problem, das bereits behoben wurde.

## Neufunde (Nachtrag 2026-09-12, gleiches Muster)

Folgende Ausprägungen desselben Musters wurden bei der Verifikation
gefunden; sie sind nicht Teil der ursprünglichen 28 Funde.

### N1: `compute_aqmh_uniform_control` -- gleicher Chunk-außen/Frame-innen-Reload wie A4

**Status (2026-09-12, Runde 2):** umgesetzt -- die Funktion akkumuliert jetzt
frame-aeusser: jeder Frame/jede Maske wird genau einmal geladen (Region-
Loader bevorzugt, Slab-Assemblierung), der Pixel-Loop laeuft zeilenparallel
ueber volle `double`-Akkumulatoren.

**Datei:** `src/reconstruction/aqmh_reconstruction.cpp:79-134`
**Aufgerufen von:** `apps/runner_phase_aqmh_reconstruction.cpp:804-818`

Die Funktion iteriert pro y0-Chunk über alle Frames und lädt Frame + Maske
pro (Chunk × Frame) neu -- dasselbe Muster wie A4. Sie läuft als
Fallback-Sweep zusätzlich zum Hauptpass.

### N2: CUDA-Sigma-Clip-Tile: Stats-Pass + Clip-Pass laden jeweils alle Frames + Q-Maps

**Status (2026-09-12, Runde 2):** obsolet -- die betroffene Funktion
`cuda_reconstruct_aqmh_impl` war unerreichbarer Dead Code (der Dispatch
ruft `reconstruct_aqmh_weighted_cuda`); der gesamte Block wurde entfernt.
Der lebende Sigma-Clip-Tile-Pfad arbeitet auf In-Memory-Tiles und hat
keine Reload-Problematik.

**Datei:** `src/acceleration/acceleration.cpp` (~1587 Stats-Pass, ~1693 Clip-Pass)

Zweifacher Full-Sweep wie A5: Der Stats-Pass lädt alle Frames und Q-Maps,
danach der Clip-Pass dieselben Daten erneut.

### N3: `aqmh_reconstruction_opencl.cpp` -- identisches A4-Host-Feed-Muster

**Status (2026-09-12, Runde 2):** umgesetzt -- Frame/Mask-Stash im
Non-Region-Pfad plus `eff_frame_region`/`eff_mask_region`-Loader analog zum
CPU-Pfad (A4); Q-Maps via `read_region`/`read_cached`. Zusaetzlich wurde
eine latente Inkonsistenz korrigiert: der `!use_region`-Pfad lud einen
Full-Frame, lehnte ihn aber an der `rows`-Dimensionspruefung ab.

**Datei:** `src/reconstruction/aqmh_reconstruction_opencl.cpp:695-769`

Der OpenCL-Host-Feed lädt Frame-Region + Q-Region + Masken-Region pro
(Chunk × Frame) -- identisch zum A4-Pattern.

### N4: `aqmh_reconstruction_cuda.cu` -- identisches Muster, Prefetch nur für Frames

**Status (2026-09-12, Runde 2):** umgesetzt -- Frame/Mask-Stash im
Non-Region-Pfad; der asynchrone Frame-Prefetch und der Host-Feed laufen
ueber die effektiven Region-Loader bzw. den Stash. Q-Map/Maske folgen dem
CPU-Pfad (`read_cached`/`eff_mask_region`).

**Datei:** `src/reconstruction/aqmh_reconstruction_cuda.cu:1484-1672`

Dasselbe A4-Pattern. Der Prefetch-Thread deckt nur Frames ab; Q-Map und
Maske bleiben synchron pro (Chunk × Frame) (Code-Kommentar ~Z. 1461-1462).

### N5: Existenz-Probes rufen den Full-Extent-Q-Provider auf

**Status (2026-09-12, Runde 2):** umgesetzt -- die Probes laufen ueber den
Rect-Provider mit leerem Rect (`y0 == y1`, `x0 == x1` => reiner
Existenz-Check, kein Decode), identisch zum Contrib-List-Pfad
(`forward_drizzle.cpp`).

**Datei:** `src/reconstruction/forward_drizzle.cpp:1284-1296`

`quality_of(si).composite`/`.artifact` als bloßer Null-Check rufen den
Full-Extent-Provider auf -> volle Dekodierung aller Streams pro Frame. Der
Contrib-List-Pfad nutzt bereits den kostenlosen Leer-Rect-Probe
(`forward_drizzle_contrib_list.cpp:497`), der Stream-Pfad nicht.

### N6: `fuse_multiband_store_to_image` liest Halo-Zeilen doppelt

**Status (2026-09-12, Runde 2):** umgesetzt -- ein Rolling-Window-Helfer
haelt die zuletzt gelesene Region pro Stream; der Folge-Chunk liest nur
das Delta statt `[y0-halo, y1+halo)` komplett neu.

**Datei:** `src/reconstruction/source_quality_artifact.cpp:637-649`

Jeder Chunk liest `[y0-halo, y1+halo)`; die Halo-Zeilen werden an jeder
Chunk-Grenze zweimal gelesen, über 4 Profil- + 3 Alpha-Streams.

### N7: Zweite Single-Slot-Source-Cache im Diagnose-Pfad

**Status (2026-09-12, Runde 2):** umgesetzt -- beide Diagnose-Call-Sites
nutzen `make_normalized_source_providers`: Whole-Set-Stash, wenn das Set
ins Budget passt, sonst der bisherige Single-Slot-Fallback; beide Pfade
liefern zusaetzlich `source_rect_of` fuer banded reads.

**Datei:** `apps/runner_phase_registration.cpp:599-610`

Zusätzlich zur bekannten Cache an Z. 511-520 eine zweite
Single-Slot-Source-Cache, die `persist_forward_drizzle_uniform` speist --
derselbe Diagnose-Pfad mit dem A3-Reload-Verhalten.

### N8: `masked_convolve` -- B3-Pendant mit 4 Full-Image-Pässen pro Level

**Status (2026-09-12, Runde 2):** umgesetzt -- `masked_convolve` berechnet
`num`/`den` in einer fusionierten `conv_axis`-Runde (2 statt 4 Full-Image-
Pässe); die Level-Scratch-Buffer `vm`/`md`/`c_cur`/`m_cur` werden ueber
Levels wiederverwendet.

**Datei:** `src/reconstruction/atrous_decomposition.cpp:39-47`

`num`/`den` sind dieselbe separable B3-Faltung über identischer Geometrie
in getrennten `conv_axis`-Pässen (4 Full-Image-Pässe + Allokationen pro
Level) -- fusbar wie B3. Zusätzlich werden `vm`/`md`/`c_cur`/`m_cur` pro
Level neu allokiert.

### N9: Veralteter Kommentar in `drizzle_profile_store.cpp`

**Status (2026-09-12, Runde 2):** umgesetzt -- der Kommentar beschreibt
jetzt die aktuelle Architektur (Tile-Produktion, Sortierung und Reduktion
in `accumulate_pair_by_frame_cuda`, Blit in die Full-Width-Stripe).

**Datei:** `src/reconstruction/drizzle_profile_store.cpp:718-723`

Der Kommentar beschreibt noch die Per-Band-Record-Memo, die §30.81 step 3a
entfernt hat (widerspricht `forward_drizzle_contrib_list.cpp:763-775`).
Nur Konsistenz-/Dokumentationsproblem, kein Code-Defekt.

---

## Begrenzungen dieser Analyse

- Keine Profiling-Messungen durchgeführt; Impact-Schätzungen basieren auf
  Schleifenstruktur und Cache-Kapazitätsrechnung.
- Die tatsächlichen I/O-Kosten hängen von `memory_budget_mb`, Frame-Größe,
  Stripe-Anzahl und LRU-Hit-Rate ab.
- Einige Funde (C1-C8, D6) sind einzeln vernachlässigbar, können sich aber
  in hot Pixel-Loops addieren.
- Die Schleifenreihenfolge Stripe-außen/Frame-innen ist möglicherweise
  bewusst gewählt für Memory-Bounding — eine Umstellung erfordert
  sorgfältige Analyse der Peak-Memory-Auswirkungen.
- Datumsinkonsistenz: Das Dokument ist auf 2026-09-23 datiert, der
  Verifikations-Nachtrag auf 2026-09-12 (Systemdatum). Vermutlich Tippfehler
  im Dateinamen/Header; der Inhalt wurde nicht angepasst.
