# 03 - Kernel-Portierungskarte

Quelle: `tile_compile_cpp/src/reconstruction/forward_drizzle_cuda_device.cu`
(Zeilen nach Stand 2026-09-26). "Prod" = im Produktionspfad `cuda_v2`;
"Proto" = Gate-Prototyp/Legacy, nicht portieren, sofern kein Test es
verlangt.

## 1. Kernel

| CUDA-Kernel | Rolle | Status | Metal-Datei / Kernel | Besonderheiten für Metal |
| --- | --- | --- | --- | --- |
| `k_polygon_rect_area_batch`, `d_polygon_rect_area` | exakte Quad/Rechteck-Fläche (Sutherland-Hodgman + Shoelace) | Prod (Baustein) | `polygon.metal` | feste Arrays, keine Rekursion; FP32 zellrelativ (T1), Parität gegen `polygon_rectangle_intersection_area` |
| `k_affine_leaf_corners_batch` | Affine-Eckpunkte | Baustein | `polygon.metal` | Transformation in DF (02, Abschnitt 2) |
| `k_scatter_v2` | affiner Frame-Scatter in framelokale Ebenen (`fa, fbs, fbg, fs2`, Quality-Ebenen) | Prod | `scatter_v2.metal` `scatter_affine` | mehrere `atomicAdd(double)` je Zelle (`fbg`, `fa`, `fbs`, optional `fs2`, Quality) -> Variante S1/S2/S3 (02, Abschnitt 3); Support: Fenster, `tile_x0/x1`, Quality-IO |
| `k_scatter_v2_samples` | kanonische ragged Affine-Sample-Liste (Tranche 8) | Prod | `scatter_v2.metal` `scatter_samples` | Eingabe `ForwardDrizzleV2SourceSample` (16 B); Layout per `static_assert` |
| `k_scatter_v2_cached` | Geometrie-Cache-Leaves (Tranche 6) | Prod | `scatter_v2.metal` `scatter_cached` | `ForwardDrizzleV2CachedLeaf` enthält `double x[4], y[4]`: auf Host in DF (hi/lo) konvertieren; Struktur `metal_types.hpp` |
| `k_scatter_v2_local` | Local-Warp: Fixpunktinversion + Subdivision (3x3, Tiefe <= 2) | Prod | `scatter_v2.metal` `scatter_local` | `ForwardDrizzleV2LocalWarp` ist FP32 (144 B) -> Koeffizienten passen; Inversion `q_{n+1}=u-d(q_n)` toleranzkritisch, DF für Positionen; Diskardierungszähler |
| `k_clear_planes_x` | Ebenen löschen | Prod | `scatter_v2.metal` `clear_planes` | trivial |
| `k_fold_accumulate_v2` | Native-Pixel-Fold, Stromstatistik, Support-Maske, Hash-Reservoir, Full-Frame-Akkumulator | Prod | `fold_v2.metal` `fold_accumulate` | Kern der Präzision (DF); `splitmix64` mit `ulong` exakt; `float4 resq`; `V2ReservoirRecord`/`V2FullAcc`-Layout Host-geteilt |
| `k_fold_native_pixel_v2` | Fold eines Native-Pixels (Baustein/Oracle-nah) | Prod (intern) | `fold_v2.metal` | wie oben |
| `k_finalize_v2` | Finalize: Clip, Konfidenz, Pixelresultat (`ForwardDrizzleV2PixelResult`) | Prod | `finalize_v2.metal` `finalize` | Median/MAD/Clip-Vergleiche -> Kippratenmessung; AoS-Record in 8-Byte-Feldern (DF für `value, b, n_eff, confidence`, der Host konvertiert nach `double`) |
| `k_finalize_v2_sfr_build/_vote/_reduce` | Shared-Frame-Rejection (SFR): Aufbau, Votum, Reduktion | Prod | `finalize_v2.metal` | Reduktion ohne Shared-Memory-Abhängigkeit im CUDA-Code; in MSL `simd`/Threadgroup-Reduktion nutzbar, aber Reihenfolge deterministisch halten |
| `k_full_seed` | Full-Frame-Estimator: Pilotgrenzen einfrieren | Prod (Config `full_frame_estimator`) | `estimator_v2.metal` `full_seed` | Pflichtumfang (README, Entscheidung 5) |
| `k_full_apply` | Full-Frame-Estimator: Nicht-Pilot-Kandidaten gegen Grenzen | Prod | `estimator_v2.metal` `full_apply` | Bimodal-Veto (`bimodal_veto_*` Zähler); `pow(q, exp)` als `precise::pow` |
| `k_affine_target_gather`, `k_affine_coverage_gather/_scatter`, `k_affine_frame_contribs`, `k_affine_dense_scatter` | Gate-1-Prototypen, Legacy-Record-Pfad, SAMPLING_GEOMETRY-Coverage | Proto / Legacy | nicht in MP4; Coverage-Gather (`sampling_geometry.cpp`) optional MP5 | Kein Produktionspfad für Rekonstruktion; Coverage-Gather nur falls SAMPLING_GEOMETRY auf dem Mac beschleunigt werden soll |

## 2. Host-Klassen

| CUDA | Metal (`metal/src`) | Anmerkung |
| --- | --- | --- |
| `ForwardDrizzleV2CudaPrototypeKernel` (`reserve`, `begin_band`, `accumulate_frame*`, `begin/accumulate/finish_affine_*`, `finalize`, `stats`, `last_device_error`) | `ForwardDrizzleV2MetalKernel : ForwardDrizzleV2Kernel` | implementiert direkt die abstrakte Schnittstelle; keine Adapterschicht wie `ForwardDrizzleV2CudaKernel` nötig. Alle Methoden des Interface bilden 1:1 ab, inklusive Piece-Lebenszyklus (Tranche 7) und Kontrakten (Fehlversuch -> `false`, Detail über `last_device_error()`). |
| `ForwardDrizzleV2CudaWorkspace` + `cudaMalloc` (84 Stellen) | `MetalWorkspace` | persistente Puffer je Rolle (`ForwardDrizzleV2BufferRoles`), `MTLResourceStorageModeShared` bzw. `Private` für reine GPU-Ebenen |
| Streams/Events (`ev_up*`, `ev_k*`) | Command-Buffer + `GPUStartTime/EndTime` | Zeiten für `upload/kernel/download_seconds` |
| `forward_drizzle_cuda_device_memory()` | `metal_device_memory()` | Working-Set statt VRAM |
| `forward_drizzle_cuda_runtime_available()` | `metal_runtime_available()` | Gerät vorhanden + Mindestfamilie |

## 3. Bildoperationen

| CUDA/OpenCV | Metal |
| --- | --- |
| `cv::cuda::warpAffine` in `AccelerationOps::warp_affine_frame`, `_rgb_frame`, CFA-4-Ebenen-Pfad (`acceleration.cpp` ca. 324-410) | `warp_affine.metal`: bikubisch (Standard `prewarp_interpolation = cubic`), weitere Modi wie CPU-Semantik, Randbehandlung/`valid_mask`/`has_data` aus `acceleration.cpp` übernehmen |
| `cv::cuda::createBoxFilter` + `multiply` in `source_quality_map.cpp` (ca. 322-380) | `box_filter.metal` (oder MPS `MPSImageBox`): Summen/Quadratsummen/Zähler-Fenster, Randverhalten wie Referenz |
| Aufrufer: `runner_phase_preprocess_pipeline.cpp:893`, `runner_phase_registration.cpp:5022/5245` | nur über `AccelerationOps`, keine Änderung der Phasenlogik |

## 4. Testzuordnung (CUDA-Tests, die je ein Metal-Gegenstück brauchen)

Quelle: `tests/test_forward_drizzle_v2.cpp`, `tests/test_forward_drizzle.cpp`.

| CUDA-Test (Titel) | Metal-Äquivalent (`metal/tests`) | Phase |
| --- | --- | --- |
| CUDA target gather matches CPU gather | Scatter-Parität affin gegen CPU-Referenz (Tier B/C) | MP4a |
| CUDA fold matches all support layers | Fold/Support-Ebenen | MP4b |
| CUDA windowed scatter matches full-source | Fenster-Invarianz | MP4a |
| CUDA persistent workspace matches ... | Workspace-Wiederverwendung, `hotpath_allocations==0` | MP4b |
| CUDA affine target-x pieces match ... | Piece-Lebenszyklus | MP4c |
| CUDA ragged affine sample list matches CPU ... | Samples-Pfad | MP4c |
| CUDA cached-leaf path matches CPU replay | Cache-Leaves | MP4e |
| shared_frame_rejection CUDA matches / consensus vote | SFR | MP4d |
| full-frame estimator CUDA matches CPU | Estimator | MP4f |
| bimodal veto CUDA matches CPU oracle | Bimodal-Veto | MP4f |
| gate10 cuda fault restarts on cpu | GPU-Fehler -> CPU-Neustart (Fault-Injection) | MP5 |
| plan-19.5 polygon_rect_area / affine leaf corners parity | Bausteinparität | MP4a |
| Local-Warp-Tests (Gate 8) | Local-Warp-Parität, Diskardierungszähler | MP4e |

Zusätzlich neu: Layout-Tests (Host/Shader-Strukturen), DF-Emulation vs.
Shader (bitgleich auf gleicher Operationsfolge), Device-Loss/Timeouts,
Budget-/Chunking-Tests, `SKIP` ohne Device.
