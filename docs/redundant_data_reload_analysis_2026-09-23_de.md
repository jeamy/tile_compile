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

## Umsetzungsstatus (Nachtrag 2026-09-12)

Alle 28 Funde wurden gegen den aktuellen Code verifiziert (Zeilennummern sind
seit Erstellung leicht gedriftet, die beschriebenen Muster sind identisch
nachweisbar). Die behobenen Punkte liegen als Worktree-Änderungen vor;
Verifikation: Build grün, 549/549 Catch2-Tests, 2.756.148 Assertions
unverändert.

| Fund | Status | Aktuelle Position / Bemerkung |
|------|--------|-------------------------------|
| A1 | zurückgestellt | `forward_drizzle.cpp` ~1509/1614-1620, Loop-Order unverändert; 2/1-Pfad delegiert an dieselbe Funktion (`output_scale.cpp:389-393`) |
| A2 | zurückgestellt (teilweise gemildert) | Reader weiterhin ohne In-Memory-Cache; getilter CUDA-Pfad hat T4b-Q-Cache (1 Read pro Frame pro Band, `forward_drizzle_contrib_list.cpp:794-813`); CPU-Stream-Pfad dekodiert weiterhin pro (Stripe × Frame) via Full-Extent-Adapter (`drizzle_profile_store.cpp:455-456, 892-894`) |
| A3 | zurückgestellt | `forward_drizzle.cpp:887-898`; nur Diagnose-/Coverage-Pfad |
| A4 | zurückgestellt | real, aber Produktionspfad übergibt immer Region-Loader (`runner_phase_aqmh_reconstruction.cpp:534-545`); Worst-Case (volle Frame-Reloads) trifft nur den Test-Fallback. Gleiches Muster auch im OpenCL-/CUDA-Host-Feed (Neufunde N3/N4) |
| A5 | zurückgestellt | real: Pre-Pass (`aqmh_reconstruction.cpp:197-231`) + Haupt-Pass (374-388) laden dieselben (fi, y0); Fix erfordert Pass-Fusion |
| A6 | zurückgestellt | reduziert auf 1× pro Frame (vorher pro Slab); Laden->Verwerfen->Neuladen bleibt |
| B1 | zurückgestellt | CPU-Fallback sequenziell (`runner_phase_aqmh_reconstruction.cpp:772-775`); CUDA-Pfad nutzt `run_planes_rgb` in einem Sweep |
| B2 | zurückgestellt | unverändert (`multiband_fusion.cpp:136-163, 198-199`) |
| B3 | zurückgestellt | unverändert (`alpha_guard.cpp:176-217`, `masked_num` 2×) |
| B4 | Fehleinschätzung (veraltet) | `compute_global_quality_weights_from_metrics` + `load_source_quality_metrics` existieren und sind verdrahtet (`source_quality_artifact.cpp:142-195`, `runner_forward_drizzle.cpp:431-434`); `preflight` erzwingt `star_max_corners >= 1`, das Metrics-Artefakt existiert daher immer |
| B5 | Fehleinschätzung | keine Redundanz: R/G/B/L sind unterschiedliche Daten, kein doppeltes Lesen derselben Bytes; One-Plane-Peak-Memory ist bewusstes Spool-Design |
| C1 | umgesetzt | `is_auto_reject` hoisted (`aqmh_reconstruction.cpp:191, 225, 262, 531`) |
| C2 | umgesetzt | `canvas_valid_flat`-Array; `canvas_valid()` komplett entfernt (`aqmh_reconstruction.cpp:66-71, 114, 182-186, 252, 260`) |
| C3 | umgesetzt | `band_src_by_level`/`band_profile_by_level` (`multiband_fusion.cpp:80-86`) |
| C4 | umgesetzt | 2×2-LUT `build_cfa_channel_2x2_lut` (`source_quality_proxy.cpp:14-29, 140, 151, 172`) |
| C5 | umgesetzt | `is_osc`/`bayer_pattern`/`cfa_origin_*` hoisted (`forward_drizzle.cpp:566-581`) |
| C6 | umgesetzt | `emit_fine`/`emit_medium`/Exponenten hoisted (`forward_drizzle.cpp:1197-1211`) |
| C7 | zurückgestellt | `global_weight` weiterhin an 3 Stellen (`aqmh_reconstruction.cpp:207, 243, 399`) |
| C8 | umgesetzt | `dx`/`dy` hoisted (`alpha_guard.cpp:150-153`) |
| D1 | teilweise; Lösungsansatz Fehleinschätzung | Caller-Buffer `dr_vals`/`dp_vals`/`mix` hoisted (`alpha_guard.cpp:65-70`); der vorgeschlagene by-ref/move-Fix ist nicht tragfähig, da `mad_sigma` die eigene destruktible Kopie für den In-Place-Median zwingend braucht. Interne `dev`-Allokation bleibt |
| D2 | umgesetzt | `wx`/`w`/`w2`/`A`/`B` hoisted (`forward_drizzle.cpp:880-886`) |
| D3 | zurückgestellt | unverändert: `result` + Profil-Ebenen pro Stripe (`forward_drizzle.cpp:1513-1529`) |
| D4 | zurückgestellt | unverändert: `reduce_window` allokiert `r`/`cand`/`counts` pro Tile (`forward_drizzle_contrib_list.cpp:605-643`) |
| D5 | zurückgestellt | unverändert (`forward_drizzle_contrib_list.cpp:1050, 1072`) |
| D6 | umgesetzt | `reserve` für `local_effective_k`/`local_margins` (`aqmh_reconstruction.cpp:485-488`) |
| E1 | zurückgestellt | unverändert: Count+Fill-Rasterisierung (`forward_drizzle_contrib_list.cpp:75-109`) |
| E2 | zurückgestellt | real; Kosten-Nuance siehe unten |
| E3 | zurückgestellt | unverändert: `invert_affine_2x3` + Band pro Tile (`forward_drizzle_contrib_list.cpp:952-973`) |

### Anmerkungen zur Einschätzung der zurückgestellten Funde

- **A3** ist derselbe Stripe-außen/Frame-innen-Pattern wie A1, aber in
  `stream_forward_drizzle_uniform` -- laut Analyse nur für Diagnose/Coverage
  genutzt, nicht die Hauptrekonstruktion.
- **A4/A5/A6** sind real: Die Chunk-außen/Frame-innen-Schleife von
  `reconstruct_aqmh_weighted` lädt ohne Region-Loader volle Frames pro Chunk
  neu (A4); Cherry-Pick-Vorpass und Haupt-Chunk-Pass rufen beide
  `q_map_cache->read_region`/`load_frame_valid_mask_region` für dieselben
  (fi, y0) auf (A5) -- bei LRU-Kapazität < frame_count echtes doppeltes I/O.
  Verifiziert: Der Produktionspfad übergibt die Region-Loader immer
  (`runner_phase_aqmh_reconstruction.cpp:534-545`, `region_streaming=yes`);
  A4s Worst-Case trifft also nur den Test-Fallback-Pfad. Ein Fix von A5/A6
  erfordert die Fusion der beiden Pässe -- dieselbe Risikoklasse wie
  A1/A2/E1/B1-B3: strukturelle Änderung an einer korrektheitskritischen
  Funktion, ohne Messung, die den Aufwand rechtfertigt.
- **E2** ist real, aber mit einer Kostennuance: `source_of` ist im Store-Build
  `VerifiedNormalizedSourceCache::load` (`source_quality_artifact.cpp:385`).
  Ein LRU-Hit kostet nur zwei Stat-Syscalls (`normalized_source_cache.cpp:
  140-153`), ein LRU-Miss dagegen einen vollen Frame-Read
  (`verify_and_insert`). Da die Tiled-Schleife pro Tile alle Frames in
  aufsteigender Reihenfolge iteriert, dominieren bei
  `frame_count > LRU-Kapazität` (Default 512 MB -> ~15 Frames bei 4K)
  Misses: der `source_of`-Aufruf vor dem Band-Cache-Lookup ist dann ein
  echter Disk-Read pro (Frame, Tile) -- nicht nur ein Funktionsaufruf.
  Die Zurückstellung bleibt vertretbar (dieselbe filigrane
  CUDA-Hybrid-Closure wie D5/E3), aber die Einschätzung "niedrigwertig"
  gilt nur für `frame_count <= LRU-Kapazität`.
- **B5** ist keine Redundanz im eigentlichen Sinn: R/G/B/L sind
  unterschiedliche Daten, kein zweifaches Lesen derselben Bytes. Das Dokument
  benennt selbst "das Spool-Design ist für One-Plane-At-A-Time Peak-Memory" --
  sequentielles, aber notwendiges I/O. Kein Fix nötig.

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

**Status (2026-09-12):** zurückgestellt -- Loop-Order unverändert (jetzt
`forward_drizzle.cpp` ~1509/1614-1620); der 2/1-Pfad delegiert an dieselbe
Funktion (`output_scale.cpp:389-393`).

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

**Status (2026-09-12):** zurückgestellt, teilweise gemildert --
`SourceQualityMapCacheReader` hat weiterhin keinen In-Memory-Cache und der
CPU-Stream-Pfad dekodiert weiterhin pro (Stripe × Frame) via
Full-Extent-Adapter; der getilte CUDA-Pfad hat dagegen den T4b-Q-Cache
(1 `read_rect` pro Frame pro Band, `forward_drizzle_contrib_list.cpp:794-813`).

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

**Status (2026-09-12):** zurückgestellt -- unverändert
(`forward_drizzle.cpp:887-898`), aber nur Diagnose-/Coverage-Pfad.

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

**Status (2026-09-12):** zurückgestellt -- real, aber der Produktionspfad
übergibt immer die Region-Loader (`runner_phase_aqmh_reconstruction.cpp:
534-545`); der Worst-Case (volle Frame-Reloads pro Chunk) trifft nur den
Test-Fallback-Pfad. Dasselbe Muster existiert zusätzlich im OpenCL- und
CUDA-Host-Feed (Neufunde N3/N4).

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

**Status (2026-09-12):** zurückgestellt -- real (Pre-Pass
`aqmh_reconstruction.cpp:197-231`, Haupt-Pass 374-388); ein Fix erfordert die
Fusion beider Pässe (strukturell, korrektheitskritisch).

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

**Status (2026-09-12):** zurückgestellt -- die Validierung wurde bereits auf
1× pro Frame reduziert (vorher pro Slab), aber Laden->Verwerfen->Neuladen im
Hauptpass bleibt bestehen.

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

**Status (2026-09-12):** zurückgestellt -- unverändert
(`runner_phase_aqmh_reconstruction.cpp:772-775`); der CUDA-Pfad nutzt
`run_planes_rgb` bereits in einem Sweep.

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

**Status (2026-09-12):** zurückgestellt -- unverändert
(`multiband_fusion.cpp:136-163, 198-199`).

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

**Status (2026-09-12):** zurückgestellt -- unverändert
(`alpha_guard.cpp:176-217`, `masked_num` 2×).

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

**Status (2026-09-12):** zurückgestellt -- unverändert, Aufrufe an
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

**Status (2026-09-12):** teilweise umgesetzt; Lösungsansatz war eine
Fehleinschätzung -- die Caller-seitigen Buffer `dr_vals`/`dp_vals`/`mix`
wurden hoisted (`alpha_guard.cpp:65-70`), aber der vorgeschlagene
by-ref/move-Fix ist nicht tragfähig: `mad_sigma` braucht die eigene
destruktible Kopie für den In-Place-Median zwingend (siehe auch den
Code-Kommentar an `ratio_at`). Die interne `dev`-Allokation pro Aufruf
bleibt bestehen.

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

**Status (2026-09-12):** zurückgestellt -- unverändert: `result` und alle
Profil-Ebenen werden weiterhin pro Stripe allokiert
(`forward_drizzle.cpp:1513-1529`). Die schwereren `candidates`-/Akkumulator-
Buffer wurden dagegen bereits früher (P6) hoisted.

**Datei:** `src/reconstruction/forward_drizzle.cpp:1493-1509`

**Problem:** `result` und alle Profil-Ebenen (`L`, `R`, `G`, `B`, `fine`, `medium`)
werden in jeder Stripe-Iteration neu allokiert.

**Lösung:** Einmal mit Max-Band-Größe allokieren, pro Stripe zurücksetzen.

---

### D4: Per-Tile `cand`/`counts`/`r` Allokationen in `reduce_window`

**Status (2026-09-12):** zurückgestellt -- unverändert
(`forward_drizzle_contrib_list.cpp:605-643`).

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

**Status (2026-09-12):** zurückgestellt -- unverändert
(`forward_drizzle_contrib_list.cpp:1050, 1072`).

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

**Status (2026-09-12):** zurückgestellt -- unverändert
(`forward_drizzle_contrib_list.cpp:75-109`).

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

**Status (2026-09-12):** zurückgestellt -- real und unverändert
(`source_of` in `forward_drizzle_contrib_list.cpp:934`, Band-Cache-Lookup
erst in 1017-1036). Kostennuance: `source_of` ist
`VerifiedNormalizedSourceCache::load` -- LRU-Hit nur zwei Stat-Syscalls,
LRU-Miss voller Frame-Read. Bei `frame_count > LRU-Kapazität` dominieren
Misses, der Aufruf ist dann ein echter Disk-Read pro (Frame, Tile) -- nicht
nur ein Funktionsaufruf. Zurückstellung dennoch vertretbar (dieselbe
filigrane CUDA-Hybrid-Closure wie D5/E3); Details im Nachtrag.

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

**Status (2026-09-12):** zurückgestellt -- unverändert
(`forward_drizzle_contrib_list.cpp:952-973`); der Code-Kommentar bestätigt
selbst, dass Band/Inverses nur von (Frame, y_begin, rows) abhängen.

**Datei:** `src/reconstruction/forward_drizzle_contrib_list.cpp:952-972`

**Problem:** `registration::invert_affine_2x3(s2c, ...)` und die Source-Y-Band-
Berechnung werden pro Tile durchgeführt, hängen aber nur vom Frame und Stripe ab,
nicht vom Tile-X-Window.

**Lösung:** Einmal pro Frame pro Band berechnen, in `CachedSourceBand` speichern.

**Impact:** Per-Tile pro Frame. `invert_affine_2x3` ist die teuerste Operation.

---

## Priorisierung

| Prio | Fund | Beschreibung | Geschätzter Impact | Status (2026-09-12) |
|------|------|-------------|-------------------|---------------------|
| **P0** | A1 | Source-Frame pro Stripe neu geladen | 600×S Disk-Reads | zurückgestellt |
| **P0** | A2 | Q-Map pro Stripe ohne In-Memory-Cache | 600×4×S .bin-Opens | zurückgestellt (CUDA-Tiled-Pfad gemildert) |
| **P1** | B2 | `luma_band` dekomponiert L-mal redundant | L× full-image Dekomposition | zurückgestellt |
| **P1** | E1 | `rasterize_drizzle_stripe` Count+Fill doppelt | 2× CPU-Rasterisierung | zurückgestellt |
| **P1** | B1 | AQMH CPU RGB 3× Sweep mit gleichen Q-Maps | 3× Frame/Mask/Q-Map I/O | zurückgestellt |
| **P1** | B3 | `smooth_alpha_b3` 2× B3-Faltung für gleiche Geometrie | 2× full-image Faltung | zurückgestellt |
| **P2** | A4 | AQMH Frame/Mask pro Chunk ohne Cache | 600×C Frame-Loads | zurückgestellt (Worst-Case nur Test-Fallback) |
| **P2** | A5 | Cherry-Pick + Haupt-Pass laden Q-Maps doppelt | 2× Q-Map I/O | zurückgestellt |
| **P2** | E2 | `source_of` bei Cache-Hit trotzdem aufgerufen | Redundante Frame-Loads | zurückgestellt (Kostennuance, siehe Nachtrag) |
| **P2** | E3 | Affine Inverse pro Tile neu | Per-Tile `invert_affine` | zurückgestellt |
| **P2** | D1 | `mad_sigma` by-value + `dev`-Allokation pro Pixel | Per-Pixel Kopie+Allok | teilweise; Lösungsansatz nicht tragfähig |
| ~~P3~~ | B4 | `compute_source_quality_proxy_v1` 2× in SQM+GQ | Per-Frame Proxy-Neuberechnung | Fehleinschätzung: bereits umgesetzt (Metrics-Artefakt) |
| **P3** | C1-C8 | Invariante Werte in Pixel-Loops | Per-Pixel Overhead | C1-C6, C8 umgesetzt; C7 offen |
| **P3** | D2-D6 | Per-Iteration Allokationen | Per-Stripe/Tile/Pixel malloc | D2, D6 umgesetzt; D3-D5 offen |
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
   Architektur-Entscheidung, die A1/A2/A3/A4 verursacht. Ein Wechsel zu
   Frame-außen/Stripe-innen würde alle vier Funde gleichzeitig lösen,
   erfordert aber eine Umstrukturierung der Akkumulator-Buffer.

5. **`prepare_drizzle_frames` wird bereits einmal pro Build aufgerufen**
   (§4.4-Kommentar in `drizzle_profile_store.cpp:505-510`), nicht pro Band.
   Dies war ein früheres Problem, das bereits behoben wurde.

## Neufunde (Nachtrag 2026-09-12, gleiches Muster)

Folgende Ausprägungen desselben Musters wurden bei der Verifikation
gefunden; sie sind nicht Teil der ursprünglichen 28 Funde.

### N1: `compute_aqmh_uniform_control` -- gleicher Chunk-außen/Frame-innen-Reload wie A4

**Datei:** `src/reconstruction/aqmh_reconstruction.cpp:79-134`
**Aufgerufen von:** `apps/runner_phase_aqmh_reconstruction.cpp:804-818`

Die Funktion iteriert pro y0-Chunk über alle Frames und lädt Frame + Maske
pro (Chunk × Frame) neu -- dasselbe Muster wie A4. Sie läuft als
Fallback-Sweep zusätzlich zum Hauptpass.

### N2: CUDA-Sigma-Clip-Tile: Stats-Pass + Clip-Pass laden jeweils alle Frames + Q-Maps

**Datei:** `src/acceleration/acceleration.cpp` (~1587 Stats-Pass, ~1693 Clip-Pass)

Zweifacher Full-Sweep wie A5: Der Stats-Pass lädt alle Frames und Q-Maps,
danach der Clip-Pass dieselben Daten erneut.

### N3: `aqmh_reconstruction_opencl.cpp` -- identisches A4-Host-Feed-Muster

**Datei:** `src/reconstruction/aqmh_reconstruction_opencl.cpp:695-769`

Der OpenCL-Host-Feed lädt Frame-Region + Q-Region + Masken-Region pro
(Chunk × Frame) -- identisch zum A4-Pattern.

### N4: `aqmh_reconstruction_cuda.cu` -- identisches Muster, Prefetch nur für Frames

**Datei:** `src/reconstruction/aqmh_reconstruction_cuda.cu:1484-1672`

Dasselbe A4-Pattern. Der Prefetch-Thread deckt nur Frames ab; Q-Map und
Maske bleiben synchron pro (Chunk × Frame) (Code-Kommentar ~Z. 1461-1462).

### N5: Existenz-Probes rufen den Full-Extent-Q-Provider auf

**Datei:** `src/reconstruction/forward_drizzle.cpp:1284-1296`

`quality_of(si).composite`/`.artifact` als bloßer Null-Check rufen den
Full-Extent-Provider auf -> volle Dekodierung aller Streams pro Frame. Der
Contrib-List-Pfad nutzt bereits den kostenlosen Leer-Rect-Probe
(`forward_drizzle_contrib_list.cpp:497`), der Stream-Pfad nicht.

### N6: `fuse_multiband_store_to_image` liest Halo-Zeilen doppelt

**Datei:** `src/reconstruction/source_quality_artifact.cpp:637-649`

Jeder Chunk liest `[y0-halo, y1+halo)`; die Halo-Zeilen werden an jeder
Chunk-Grenze zweimal gelesen, über 4 Profil- + 3 Alpha-Streams.

### N7: Zweite Single-Slot-Source-Cache im Diagnose-Pfad

**Datei:** `apps/runner_phase_registration.cpp:599-610`

Zusätzlich zur bekannten Cache an Z. 511-520 eine zweite
Single-Slot-Source-Cache, die `persist_forward_drizzle_uniform` speist --
derselbe Diagnose-Pfad mit dem A3-Reload-Verhalten.

### N8: `masked_convolve` -- B3-Pendant mit 4 Full-Image-Pässen pro Level

**Datei:** `src/reconstruction/atrous_decomposition.cpp:39-47`

`num`/`den` sind dieselbe separable B3-Faltung über identischer Geometrie
in getrennten `conv_axis`-Pässen (4 Full-Image-Pässe + Allokationen pro
Level) -- fusbar wie B3. Zusätzlich werden `vm`/`md`/`c_cur`/`m_cur` pro
Level neu allokiert.

### N9: Veralteter Kommentar in `drizzle_profile_store.cpp`

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
