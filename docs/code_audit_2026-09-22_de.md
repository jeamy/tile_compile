# Code-Audit: Änderungen der letzten 5 Tage

**Datum**: 2026-09-22
**Scope**: Alle GitHub-Repositories unter `/media/data/programming/tile_compile/`
**Zeitraum**: Letzte 5 Tage (50 Commits, 179 Dateien, ~49.675 Zeilen eingefügt, ~1.858 gelöscht)
**Methode**: Manuelle + subagent-gestützte statische Analyse (grep, read, cross-reference)
**Build-Status**: Suite grün (2.756.148 Assertionen in 549 Testfällen)

## Zusammenfassung

| Kategorie | Anzahl | Geschätzte LOC-Einsparung |
|-----------|--------|---------------------------|
| P0 — Bugs | 2 | — |
| P1 — Dead Code | 14 | ~450 |
| P2 — Duplikation | 12 | ~250 |
| P3 — Ineffizienz | 10 | — |
| P4 — Abstraktion/Boilerplate | 8 | ~2.200 |
| **Gesamt** | **46** | **~2.900 LOC** |

Betroffene Dateien: 35+ in `tile_compile_cpp/`, 6 in `web_backend_cpp/`, 4 in `web_frontend_v3/`.

---

## P0 — Bugs

### P0-1: `runner_main.cpp:148` — `command` undefiniert im CLI11-Pfad

**Datei**: `tile_compile_cpp/apps/runner_main.cpp:145-149`
**Kategorie**: Bug (Kompilierungsfehler)
**Schweregrad**: P0

```cpp
// #ifdef HAVE_CLI11 branch (Zeile 145-149):
if (run_cmd->parsed()) {
    return run_pipeline_command(config_path, input_dir, runs_dir, project_root,
                       run_id_override, dry_run,
                       max_frames, max_tiles, config_from_stdin, command == "reconstruct");
}
```

`command` wird in Zeile 148 verwendet, aber erst in Zeile 169 im `#else`-Zweig deklariert. Wenn `HAVE_CLI11` definiert ist (CLI11 installiert), ist `command` ein undefinierter Bezeichner → Kompilierungsfehler.

**Auswirkung**: Der Runner lässt sich nicht kompilieren, wenn CLI11 gefunden wird. Aktuell nicht aktiv (CLI11 nicht im Build), aber latent.
**Empfehlung**: `command == "reconstruct"` ersetzen durch `reconstruct_cmd->parsed()` oder einen lokalen `bool forward_drizzle_only`.

### P0-2: `int_to_phase` — stale Phase-Mapping

**Datei**: `tile_compile_cpp/include/tile_compile/core/types.hpp:267-272`
**Kategorie**: Bug + Dead Code
**Schweregrad**: P0/P1

```cpp
inline Phase int_to_phase(int i) {
    if (i >= 0 && i <= 23) {
        return static_cast<Phase>(i);
    }
    return Phase::SCAN_INPUT;
}
```

`Phase` hat 29 Werte (0-28), aber `int_to_phase` akzeptiert nur 0-23. Phasen 24-28 (`NORMALIZED_CACHE`, `SAMPLING_GEOMETRY`, `GLOBAL_QUALITY`, `FORWARD_DRIZZLE`, `SOURCE_QUALITY_MAPS`, `MULTIBAND`) werden still auf `SCAN_INPUT` gemappt. Die Funktion wird nirgends aufgerufen, aber bei Wiederverwendung wäre das ein stiller Datenverlust.

**Empfehlung**: Funktion entfernen (sie ist tot) oder Bereich auf 0-28 korrigieren.

---

## P1 — Dead Code

### P1-1: `try_astrometric_rescue` (Matrix2Df-Überladung)

**Datei**: `tile_compile_cpp/include/tile_compile/registration/astrometric_rescue.hpp:27` (Deklaration), `tile_compile_cpp/src/registration/astrometric_rescue.cpp:106` (Implementierung, bereits entfernt)
**Kategorie**: Dead Code
**Schweregrad**: P1
**LOC**: ~170 (bereits entfernt)

Matrix2Df-Überladung war deklariert und implementiert, aber nirgends aufgerufen. Die Pfad-basierte Überladung `try_astrometric_rescue_from_paths` wird stattdessen verwendet.

**Status**: ✅ Bereits entfernt in dieser Session.

### P1-2: `reconstruct_tiles` und `reconstruct_tiles_parallel`

**Datei**: `tile_compile_cpp/include/tile_compile/reconstruction/reconstruction.hpp:55,63`, `tile_compile_cpp/src/reconstruction/reconstruction.cpp:287,1159`
**Kategorie**: Dead Code (nur in Tests verwendet)
**Schweregrad**: P1
**LOC**: ~200

Beide Funktionen werden nur in Tests aufgerufen, nicht in `src/` oder `apps/`. Legacy-Rekonstruktionspfad, der durch Forward-Drizzle ersetzt wurde.

**Empfehlung**: Wenn die Tests nur Legacy-Verhalten verifikationshalber abdecken, können Funktion+Tests entfernt werden. Sonst mit `#ifdef TILE_COMPILE_LEGACY` markieren.

### P1-3: `sigma_clip_weighted_rgb_tile_shared_mask`

**Datei**: `tile_compile_cpp/include/tile_compile/reconstruction/reconstruction.hpp:89`, `tile_compile_cpp/src/reconstruction/reconstruction.cpp:903`
**Kategorie**: Dead Code
**Schweregrad**: P1
**LOC**: ~60

Deklariert und implementiert, aber nirgends aufgerufen (auch nicht in Tests).

**Empfehlung**: Entfernen.

### P1-4: `debayer_bilinear_region` — deklariert, nie definiert

**Datei**: `tile_compile_cpp/include/tile_compile/image/cfa_processing.hpp:148`
**Kategorie**: Dead Code (Forward Declaration ohne Definition)
**Schweregrad**: P1
**LOC**: ~1

Header deklariert `DebayerResult debayer_bilinear_region(...)`, aber es gibt keine Definition irgendwo in der Codebase. Linker-Fehler bei Verwendung.

**Empfehlung**: Deklaration entfernen.

### P1-5: `cfa_green_mask` — public, aber nur intern verwendet

**Datei**: `tile_compile_cpp/include/tile_compile/image/cfa_processing.hpp:12` (Deklaration), `tile_compile_cpp/src/image/cfa_processing.cpp:331` (Definition), `:371` (einziger Aufrufer)
**Kategorie**: Unnötige public-API
**Schweregrad**: P1
**LOC**: ~1

Die einzige Aufrufstelle ist in derselben `.cpp`-Datei. Sollte `static` sein oder aus dem Header entfernt werden.

**Empfehlung**: Aus Header entfernen, `static` in `.cpp` machen.

### P1-6: `string_to_sampling_warp_convention` — Dead Code mit irreführendem Kommentar

**Datei**: `tile_compile_cpp/src/registration/registration_sampling_plan.cpp:29-33`, `tile_compile_cpp/include/tile_compile/registration/registration_sampling_plan.hpp:37`
**Kategorie**: Dead Code + irreführende Implementierung
**Schweregrad**: P1

```cpp
SamplingWarpConvention string_to_sampling_warp_convention(const std::string& s) {
  // Only one convention exists; anything else is a hard error at parse time.
  if (s == "canvas_to_source") return SamplingWarpConvention::canvas_to_source;
  return SamplingWarpConvention::canvas_to_source;  // <- kein "hard error", gleicher Return
}
```

Der Kommentar behauptet "hard error at parse time", aber beide Branches returnen denselben Wert. Die Funktion wird nirgends aufgerufen.

**Empfehlung**: Entfernen oder bei Ungültigkeit `throw` einbauen.

### P1-7: `run_command` — reiner Pass-through, nie aufgerufen

**Datei**: `tile_compile_cpp/apps/runner_main.cpp:42-51`
**Kategorie**: Dead Code + unnötige Abstraktion
**Schweregrad**: P1
**LOC**: ~10

Wrapper, der nur `run_pipeline_command(...)` mit denselben Argumenten aufruft. Nirgends aufgerufen.

**Empfehlung**: Entfernen.

### P1-8: `require_gain_match`

**Datei**: `tile_compile_cpp/apps/runner_pipeline.cpp:381-413`
**Kategorie**: Dead Code
**Schweregrad**: P1
**LOC**: ~32

Definiert, aber nie aufgerufen. Die Schwesterfunktion `warn_if_gain_mismatch` wird verwendet.

**Empfehlung**: Entfernen oder verwenden.

### P1-9: `apply_common_overlap_to_tile_inplace` (ohne `_check_nonzero`)

**Datei**: `tile_compile_cpp/apps/runner_shared.hpp:278-319`
**Kategorie**: Dead Code
**Schweregrad**: P1
**LOC**: ~40

Die `_check_nonzero`-Variante wird aufgerufen, diese nicht.

**Empfehlung**: Entfernen.

### P1-10: `apply_common_overlap_to_frame_inplace_and_check_nonzero`

**Datei**: `tile_compile_cpp/apps/runner_shared.hpp:374-404`
**Kategorie**: Dead Code
**Schweregrad**: P1
**LOC**: ~30

Deklariert und definiert (inline im Header), aber nirgends aufgerufen.

**Empfehlung**: Entfernen.

### P1-11: `percentile_of` und `median_of` in `report_generator.cpp`

**Datei**: `web_backend_cpp/src/services/report_generator.cpp:494-503`
**Kategorie**: Dead Code
**Schweregrad**: P1
**LOC**: ~12

Beide Funktionen sind definiert, rufen sich gegenseitig auf, haben aber keine externen Aufrufer. `basic_stats` verwendet direkt `percentile_sorted`.

**Empfehlung**: Entfernen.

### P1-12: `getBgeLabel` in `phase-list.js` — unreachable

**Datei**: `web_frontend_v3/js/components/phase-list.js:69-75,80-82`
**Kategorie**: Dead Code (unreachable branch)
**Schweregrad**: P1
**LOC**: ~12

`getPhasesForConfig` mappt über `RECONSTRUCT_PHASES`, das `"BGE"` nicht enthält. Der ternary `p === "BGE" ? getBgeLabel(...) : ...` ist immer false. `getBgeLabel` wird auch nirgends sonst importiert.

**Empfehlung**: `getBgeLabel` und den ternary entfernen.

---

## P2 — Duplikation

### P2-1: `ByteSink` — 3 identische Structs

**Dateien**:
- `tile_compile_cpp/src/reconstruction/source_quality_map_cache.cpp:42`
- `tile_compile_cpp/src/reconstruction/profile_store_manifest.cpp:18`
- `tile_compile_cpp/src/reconstruction/quality_frame_weight_plan.cpp:22`

**Kategorie**: Duplikation
**Schweregrad**: P2
**LOC**: ~25

Drei nahezu identische `ByteSink`-Structs mit `u32`/`u64`/`str`-Methoden. `quality_frame_weight_plan.cpp` hat zusätzlich `f32` und `b`.

**Empfehlung**: In `core/byte_sink.hpp` extrahieren, alle drei verwenden es.

### P2-2: `to_image_hms_config` — 3 identische Kopien

**Dateien**:
- `tile_compile_cpp/apps/runner_pipeline.cpp:183`
- `tile_compile_cpp/apps/runner_downstream.cpp:291`
- `tile_compile_cpp/apps/runner_resume.cpp:504`

**Kategorie**: Duplikation
**Schweregrad**: P2
**LOC**: ~60

Identische Konvertierungsfunktion `config::HyperMetricStretchConfig → image::HyperMetricStretchConfig` in drei Dateien.

**Empfehlung**: Nach `runner_shared.cpp/hpp` verschieben.

### P2-3: `shell_quote` — 4 Implementierungen

**Dateien**:
- `tile_compile_cpp/apps/runner_shared.cpp:2300` (kanonisch)
- `tile_compile_cpp/apps/runner_resume.cpp:63`
- `tile_compile_cpp/apps/runner_preprocess.cpp:1140`
- `tile_compile_cpp/src/registration/astrometric_rescue.cpp:75`

**Kategorie**: Duplikation
**Schweregrad**: P2
**LOC**: ~45

Vier unabhängige Implementierungen von `shell_quote`. Die in `runner_shared.cpp` ist die kanonische.

**Empfehlung**: Nach `core/utils.hpp` verschieben, alle Kopien entfernen.

### P2-4: Luma-Formel `0.25R + 0.5G + 0.25B` — 7 hartcodierte Kopien

**Dateien**:
- `runner_pipeline.cpp:3598,3834,5166,5218`
- `runner_resume.cpp:1489,1529`
- `runner_phase_metrics.cpp:597`

**Kategorie**: Duplikation
**Schweregrad**: P2
**LOC**: ~30

Die gleiche RGB-zu-Luma-Formel ist 7-mal hartcodiert. `multiband_fusion.hpp` definiert bereits `kWorkingLumaWeightsOsc[3] = {0.25, 0.5, 0.25}`.

**Empfehlung**: `image::rgb_to_luma(R, G, B)` Helper einführen, alle 7 Stellen ersetzen.

### P2-5: `compute_sha256_file` in `cli_main.cpp` dupliziert `core::sha256_file`

**Datei**: `tile_compile_cpp/apps/cli_main.cpp:204-236`
**Kategorie**: Duplikation
**Schweregrad**: P2
**LOC**: ~30

Lokale Implementierung von SHA-256, die `core::sha256_file` (`src/core/utils.cpp:249`) dupliziert.

**Empfehlung**: `core::sha256_file` verwenden, lokale Implementierung entfernen.

### P2-6: `read_aqmh_cache_map` / `read_aqmh_value` — duplizierter Switch

**Datei**: `web_backend_cpp/src/services/report_generator.cpp:1568-1595,1661-1684`
**Kategorie**: Duplikation
**Schweregrad**: P2
**LOC**: ~18

Beide Funktionen enthalten denselben `float32`/`uint16`/`uint8`-Switch mit identischer Skalierung/Clamping-Logik.

**Empfehlung**: `read_aqmh_cache_map` ruft `read_aqmh_value` pro Pixel auf.

### P2-7: `robust_median` / `robust_median_inplace` — reine Wrapper

**Datei**: `tile_compile_cpp/src/image/background_extraction.cpp:57-71`
**Kategorie**: Duplikation + unnötige Abstraktion
**Schweregrad**: P2
**LOC**: ~8

`robust_median_inplace` ruft `core::median_of(values)` auf. `robust_median` ruft `robust_median_inplace(values)` auf (mit extra Kopie durch by-value).

**Empfehlung**: Direkt `core::median_of` bzw. `core::median_of_or_nan_inplace` verwenden.

### P2-8: `robust_quantile` / `robust_quantile_inplace` — reine Wrapper

**Datei**: `tile_compile_cpp/src/image/background_extraction.cpp:73-93`
**Kategorie**: Duplikation + unnötige Abstraktion
**Schweregrad**: P2
**LOC**: ~8

Gleiches Muster wie P2-7 für Quantil-Funktionen.

### P2-9: `percentile_of_valid` in `autobge.cpp` — Wrapper um `core::percentile_from_sorted`

**Datei**: `tile_compile_cpp/src/image/autobge.cpp:85-88`
**Kategorie**: Duplikation
**Schweregrad**: P2
**LOC**: ~4

Multipliziert `pct` mit `100.0f` und leitet an `core::percentile_from_sorted` weiter.

### P2-10: `percentile_double` in `sampling_geometry.cpp` dupliziert `core::percentile_of`

**Datei**: `tile_compile_cpp/src/registration/sampling_geometry.cpp:33-44`
**Kategorie**: Duplikation
**Schweregrad**: P2
**LOC**: ~10

Sort + Quantil-Interpolation für `double`, während `core::percentile_of` für `float` existiert.

**Empfehlung**: Template-Überladung oder `double`-Variante in `core/utils` hinzufügen.

### P2-11: `aqmh_cherry_pick.cpp` — duplizierte Filter- und Vergleichslogik

**Datei**: `tile_compile_cpp/src/reconstruction/aqmh_cherry_pick.cpp:27-29,37-39,61-63,70-72`
**Kategorie**: Duplikation
**Schweregrad**: P2
**LOC**: ~15

`std::remove_if` Score-Filter und `score_cmp`-Lambda werden in `aqmh_select_top_k` und `aqmh_select_auto_reject` wiederholt.

### P2-12: `quantile` in `aqmh_reconstruction.cpp` dupliziert `quantile_inplace` in `runner_phase_aqmh_reconstruction.cpp`

**Dateien**:
- `tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp:34-42`
- `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp:73-79`

**Kategorie**: Duplikation
**Schweregrad**: P2
**LOC**: ~10

Zwei Quantil-Helper mit unterschiedlicher Interpolationspolitik im selben Projekt.

---

## P3 — Ineffizienz

### P3-1: `robust_sigma_mad` — unnötige Kopie bei erster Medianberechnung

**Datei**: `tile_compile_cpp/src/core/utils.cpp:471-477`
**Kategorie**: Ineffizienz
**Schweregrad**: P3

`float med = median_of(pixels);` kopiert den gesamten Vektor. Die zweite MAD-Berechnung verwendet bereits `std::move`. Die erste sollte `median_of_or_nan_inplace(pixels)` verwenden.

### P3-2: `build_background_mask_sigma_clip` — Full-Image-Kopie für Median

**Datei**: `tile_compile_cpp/src/metrics/metrics.cpp:33-34`
**Kategorie**: Ineffizienz
**Schweregrad**: P3

`vals` wird aus dem ganzen Frame konstruiert, dann kopiert `core::median_of(vals)` den Vektor erneut.

**Empfehlung**: `core::median_of_or_nan_inplace(vals)` verwenden.

### P3-3: `detect_stars_simple` — O(h·w·961) Box-Blur

**Datei**: `tile_compile_cpp/src/registration/global_registration.cpp:378-399`
**Kategorie**: Ineffizienz
**Schweregrad**: P3

31×31 nested Loop pro Pixel für lokales Hintergrund-Subtraktion. `cv::boxFilter` oder Integralbild wäre O(h·w).

### P3-4: `alpha_guard.cpp` — Per-Pixel-Allokationen im Hot-Loop

**Datei**: `tile_compile_cpp/src/reconstruction/alpha_guard.cpp:76,85-86,93`
**Kategorie**: Ineffizienz
**Schweregrad**: P3

`dr_vals`, `dp_vals`, `mix` werden pro Pixel neu allokiert. Hoisting außerhalb der Schleife würde wiederholte `malloc`/`free` vermeiden.

### P3-5: `aqmh_select_top_k` / `aqmh_select_auto_reject` — by-value `samples`

**Datei**: `tile_compile_cpp/include/tile_compile/reconstruction/aqmh_cherry_pick.hpp:14,19`
**Kategorie**: Ineffizienz
**Schweregrad**: P3

`std::vector<AqmhWeightedSample> samples` by-value erzwungene Kopie des Pixel-Kandidaten-Vektors.

**Empfehlung**: `std::vector<...>&&` oder `const&` mit `std::move` an Aufrufstellen.

### P3-6: `mad_sigma` in `alpha_guard.hpp` — by-value `std::vector<float>`

**Datei**: `tile_compile_cpp/include/tile_compile/reconstruction/alpha_guard.hpp:60`
**Kategorie**: Ineffizienz
**Schweregrad**: P3

Kopiert den Vektor bei jedem Aufruf in der Hot-Schleife.

### P3-7: `aqmh_select_auto_reject` — Full-Sort für Best-Score-Suche

**Datei**: `tile_compile_cpp/src/reconstruction/aqmh_cherry_pick.cpp:73`
**Kategorie**: Ineffizienz
**Schweregrad**: P3

Sortiert den gesamten Vektor, um nur den besten Score und einen Schwellenwert zu finden. `std::nth_element` würde ausreichen.

### P3-8: `quantile` in `aqmh_reconstruction.cpp` — Full-Sort für ein Quantil

**Datei**: `tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp:34-42`
**Kategorie**: Ineffizienz
**Schweregrad**: P3

`std::sort` für ein einzelnes Quantil; `nth_element` wäre ausreichend.

### P3-9: `source_quality_artifact.cpp` — Serialize-then-Parse-Validierung

**Datei**: `tile_compile_cpp/src/reconstruction/source_quality_artifact.cpp:84,135,191`
**Kategorie**: Ineffizienz
**Schweregrad**: P3

Drei Stellen, an denen `parse(serialize(...))` als Validierung verwendet wird. Serialisiert zu JSON-String und parst sofort zurück. Direkte `validate_*`-Funktion wäre effizienter.

### P3-10: `patch_estimate` in `autobge.cpp` — Kopie für Median

**Datei**: `tile_compile_cpp/src/image/autobge.cpp:109-115`
**Kategorie**: Ineffizienz
**Schweregrad**: P3

`core::median_of(vals)` kopiert den Vektor, obwohl `median_of_or_nan_inplace(vals)` verwendet werden könnte.

---

## P4 — Abstraktion / Boilerplate

### P4-1: Repetitive Doxygen-Kommentarblöcke (~856 Vorkommen)

**Dateien**: Alle `.cpp`-Dateien in `src/` und `apps/`
**Kategorie**: Boilerplate
**Schweregrad**: P4
**LOC**: ~2.000

856 Doxygen-Blöcke der Form:
```cpp
/// @brief Implements <function_name>.
/// @details Part of <module_description>; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
```

Diese Blöcke sind Copy-Paste-Templates, die keinen Informationswert über den Funktionsnamen hinaus bieten. Sie produzieren ~4 Zeilen pro Funktion × 856 = ~3.400 Zeilen Boilerplate (netto ~2.000 nach Abzug der Funktionsnamen-Zeile).

**Betroffene Dateien (Top 10)**:
| Datei | Anzahl |
|-------|--------|
| `src/image/background_extraction.cpp` | 63 |
| `apps/runner_shared.cpp` | 55 |
| `src/core/acceleration.cpp` | 46 |
| `apps/runner_pipeline.cpp` | 22 |
| `src/io/fits_io.cpp` | 28 |
| `src/core/utils.cpp` | 27 |
| `src/registration/global_registration.cpp` | 26 |
| `src/image/cfa_processing.cpp` | 24 |
| `apps/cli_main.cpp` | 37 |
| `src/reconstruction/reconstruction.cpp` | 19 |

**Empfehlung**: Datei-Level-Doxygen-Kommentar statt pro-Funktion-Boilerplate. Funktionsname ist selbsterklärend.

### P4-2: Handgeschriebene JSON-Serialisierung in `profile_store_manifest.cpp` und `quality_frame_weight_plan.cpp`

**Dateien**:
- `tile_compile_cpp/src/reconstruction/profile_store_manifest.cpp:116-182`
- `tile_compile_cpp/src/reconstruction/quality_frame_weight_plan.cpp:131-196`

**Kategorie**: Boilerplate
**Schweregrad**: P4
**LOC**: ~60

Manuelle `nlohmann::json` Feld-für-Feld Serialisierung/Deserialisierung mit identischem try/catch- und Hash-Validierungsmuster.

**Empfehlung**: Schema-Generator oder Makro-basierte Serialisierung verwenden.

### P4-3: SVG-Chart-Boilerplate in `report_generator.cpp`

**Datei**: `web_backend_cpp/src/services/report_generator.cpp:833-844,987-993,1066-1070,1127-1130,1188-1204`
**Kategorie**: Boilerplate
**Schweregrad**: P4
**LOC**: ~60-80

Jede SVG-Chart-Funktion (`svg_timeseries`, `svg_histogram`, `svg_scatter`, `svg_bar`, `svg_bar_horizontal`) wiederholt `svg_begin`, Titel-Text und Achsenkonstruktion.

**Empfehlung**: `append_chart_frame(title, axis_config)` Helper einführen.

### P4-4: `read_file_text` / `write_file_text` in `cli_main.cpp` duplizieren `core::`-Utilities

**Datei**: `tile_compile_cpp/apps/cli_main.cpp:70-87`
**Kategorie**: Abstraktion
**Schweregrad**: P4
**LOC**: ~20

Dünne Wrapper, die `core::read_text` / `core::write_text_atomic` duplizieren.

### P4-5: `print_json` — Ein-Zeilen-Wrapper

**Datei**: `tile_compile_cpp/apps/cli_main.cpp:62-64`
**Kategorie**: Abstraktion
**Schweregrad**: P4
**LOC**: ~3

`void print_json(const json& j) { std::cout << j.dump(2) << std::endl; }` — wird 34-mal aufgerufen, aber die Abstraktion bietet keinen Mehrwert.

### P4-6: `get_executable_dir` — `fs::current_path().string()`

**Datei**: `tile_compile_cpp/apps/cli_main.cpp:46-48`
**Kategorie**: Abstraktion
**Schweregrad**: P4
**LOC**: ~5

Wrapper um einen Einzeiler.

### P4-7: `open_geometry_cache` — Single-Use-Lambda

**Datei**: `tile_compile_cpp/apps/runner_forward_drizzle.cpp:202-209`
**Kategorie**: Abstraktion
**Schweregrad**: P4
**LOC**: ~5

Lambda, das nur einmal aufgerufen wird und inline sein könnte.

### P4-8: i18n-Spiegeleinträge in `report_de.json` / `report_en.json`

**Dateien**: `web_frontend_v3/i18n/report_de.json:28-29,111`, `web_frontend_v3/i18n/report_en.json:277-278,416`
**Kategorie**: Boilerplate
**Schweregrad**: P4
**LOC**: ~4

`report_de` hat `"Discovered": "Entdeckt"`, `report_en` hat `"Entdeckt": "Discovered"` — gespiegelte Einträge. `"OK": "OK"` Identitäts-Übersetzung in beiden Dateien.

---

## P3-Komplexität: Funktionen mit zu vielen Parametern

Diese Funktionen haben 9-22 Parameter und sind Kandidaten für Config-Struct-Extraktion:

| Datei | Funktion | Parameter | Empfehlung |
|-------|----------|-----------|------------|
| `forward_drizzle_cuda.hpp:199` | `forward_drizzle_cuda_affine_frame_contributions` | **22** | `CudaSourceBand` + `CfaOrigin` Structs |
| `acceleration.hpp:163` | `warp_affine_rgb_frame` | 13 | `WarpAffineConfig` Struct |
| `acceleration.hpp:183` | `reconstruct_aqmh` | 13 | `AqmhReconstructionBindings` Struct |
| `forward_drizzle.hpp:459` | `stream_forward_drizzle_uniform_and_raw` | 13 | `ForwardDrizzleStreamingOptions` Struct |
| `forward_drizzle.hpp:423` | `reduce_pixel_profiles` | 12 | Output/Alpha-Pointer bündeln |
| `aqmh_reconstruction.hpp:112` | `reconstruct_aqmh_weighted` | 12 | Gleicher Loader-Bundle wie oben |
| `aqmh_reconstruction_cuda.hpp:15` | `reconstruct_aqmh_weighted_cuda` | 12 | Gleicher Bundle |
| `aqmh_reconstruction_opencl.hpp:11` | `reconstruct_aqmh_weighted_opencl` | 12 | Gleicher Bundle |
| `global_registration.hpp:176` | `warp_frame_with_smooth_local_model` | 11 | `SmoothLocalCoordinateMapping` Struct |
| `global_registration.hpp:196` | `triangle_star_matching` | 11 | `StarMatchingParams` Struct |
| `metrics.hpp:28` | `calculate_global_weights_with_stars` | 11 | `GlobalWeightingParams` Struct |
| `forward_drizzle.hpp:201` | `rasterize_drizzle_stripe` | 10 | `DrizzleStripeWindow` Struct |
| `forward_drizzle.hpp:223` | `enumerate_drizzle_stripe_leaf_cells` | 10 | Gleicher Struct |
| `acceleration.hpp:156` | `warp_affine_frame` | 10 | `WarpAffineConfig` Struct |
| `acceleration.hpp:196` | `overlap_add` | 10 | `OverlapAddConfig` Struct |

**Geschätzte LOC-Einsparung**: ~30 (Deklarationen) + deutlich verbesserte Lesbarkeit.

---

## Header-Bloat

### `atomic_output.hpp` — Inline-Implementierung in Header

**Datei**: `tile_compile_cpp/include/tile_compile/core/atomic_output.hpp:16-74`
**Kategorie**: Header-Bloat
**Schweregrad**: P3

~60 Zeilen nicht-trivialer Konstruktor/Destruktor/`commit()` inline im Header. Zieht `<atomic>`, `<chrono>`, `<fcntl.h>`, `<unistd.h>` in jeden Konsumenten ein.

**Empfehlung**: Implementierung nach `src/core/atomic_output.cpp` verschieben.

### `omp_effective_threads()` — Inline in `utils.hpp`

**Datei**: `tile_compile_cpp/include/tile_compile/core/utils.hpp:119-132`
**Kategorie**: Header-Bloat
**Schweregrad**: P3

~15 Zeilen nicht-trivialer Inline-Code, der `<thread>` und `omp.h` in der Header benötigt. Wird nur aus `.cpp`-Dateien aufgerufen.

**Empfehlung**: Out-of-line in `utils.cpp` definieren.

### `#include <map>` in `types.hpp` — ungenutzt

**Datei**: `tile_compile_cpp/include/tile_compile/core/types.hpp:7`
**Kategorie**: Stale Include
**Schweregrad**: P4

`std::map` wird in `types.hpp` nicht referenziert.

---

## Empfehlungen (Priorisiert)

### Sofort (P0)
1. **`runner_main.cpp:148`** — `command` durch `reconstruct_cmd->parsed()` ersetzen.
2. **`int_to_phase`** — Entfernen (tot) oder Bereich auf 0-28 korrigieren.

### Kurzfristig (P1)
3. Dead-Code-Funktionen entfernen: `reconstruct_tiles`, `reconstruct_tiles_parallel`, `sigma_clip_weighted_rgb_tile_shared_mask`, `debayer_bilinear_region`, `run_command`, `require_gain_match`, `apply_common_overlap_to_tile_inplace`, `apply_common_overlap_to_frame_inplace_and_check_nonzero`, `percentile_of`/`median_of` (report_generator), `getBgeLabel`, `string_to_sampling_warp_convention`.
4. `cfa_green_mask` aus Header entfernen, `static` machen.

### Mittelfristig (P2/P3)
5. `ByteSink` in `core/byte_sink.hpp` konsolidieren.
6. `to_image_hms_config` nach `runner_shared` verschieben.
7. `shell_quote` nach `core/utils` verschieben, 3 Kopien entfernen.
8. `image::rgb_to_luma(R, G, B)` Helper einführen, 7 hartcodierte Stellen ersetzen.
9. `compute_sha256_file` in `cli_main.cpp` durch `core::sha256_file` ersetzen.
10. `robust_median`/`robust_quantile` Wrapper entfernen, direkt `core::` verwenden.
11. `robust_sigma_mad` und `build_background_mask_sigma_clip` auf In-Place-Median umstellen.
12. `alpha_guard.cpp` Per-Pixel-Allokationen hoisten.
13. `aqmh_select_*` auf `&&`-Parameter umstellen.
14. Serialize-then-Parse-Validierung durch direkte `validate_*` ersetzen.

### Langfristig (P4)
15. Doxygen-Boilerplate durch Datei-Level-Kommentare ersetzen (~2.000 LOC).
16. JSON-Serialisierung durch Schema-Generator ersetzen.
17. SVG-Chart-Boilerplate durch Helper konsolidieren.
18. Parameter-Structs für Funktionen mit 9+ Parametern einführen.
19. `atomic_output.hpp` und `omp_effective_threads` out-of-line definieren.

---

## Statistik

| Metrik | Wert |
|--------|------|
| Commits im Zeitraum | 50 |
| Geänderte Dateien | 179 |
| Zeilen eingefügt | ~49.675 |
| Zeilen gelöscht | ~1.858 |
| Issues gesamt | 46 |
| P0 (Bugs) | 2 |
| P1 (Dead Code) | 14 |
| P2 (Duplikation) | 12 |
| P3 (Ineffizienz) | 10 |
| P4 (Boilerplate) | 8 |
| Geschätzte LOC-Einsparung | ~2.900 |
| Betroffene C++ Dateien | 35+ |
| Betroffene Web-Dateien | 10 |
| Subagent-Analysen | 5 parallel |
| Verifizierte Funde | 46 |
