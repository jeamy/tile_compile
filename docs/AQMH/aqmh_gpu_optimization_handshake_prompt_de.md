# AQMH GPU-Optimierung — Handshake-Prompt (lokale Ausführung)

> **Zweck.** Dieses Dokument ist ein **kopierfertiger Prompt** für eine
> Claude-Code-Sitzung, die **lokal auf einer Maschine mit CUDA-Toolkit + GPU**
> (und, falls vorhanden, OpenCL-Runtime) läuft. Es überträgt den vollständigen
> Kontext („Handshake"), damit die dortige Sitzung die Optimierung aus
> `aqmh_gpu_optimization_implementation_guide_de.md` **bauen, testen und
> profilen** kann — was in der Cloud-/Sandbox-Umgebung ohne GPU nicht möglich
> ist.

## Wie benutzen?

1. Repo lokal auschecken, Branch `claude/aqmh-gpu-optimization-urpo5e`.
2. Sicherstellen: `nvcc` im PATH, passende OpenCV-CUDA-Build vorhanden
   (`~/.local/opencv-cuda*` laut `CMakeLists.txt:17–19`), GPU sichtbar
   (`nvidia-smi`).
3. Den kompletten Block unter **„=== PROMPT START ==="** in die lokale
   Claude-Code-Sitzung einfügen.
4. Der Sitzung folgen; sie arbeitet WP für WP mit grünem Build+Test vor jedem
   nächsten Schritt.

---

=== PROMPT START ===

Du arbeitest im Repository `tile_compile` auf dem Branch
`claude/aqmh-gpu-optimization-urpo5e`. **Diese Maschine hat eine echte NVIDIA-GPU
und ein CUDA-Toolkit** — anders als die Umgebung, in der die Analyse und die
Implementationsanleitung erstellt wurden. Deine Aufgabe: die GPU-Optimierung der
AQMH-Phasen **bauen, testen, profilen und verifizieren**.

## Pflichtlektüre zuerst (in dieser Reihenfolge)

1. `docs/AQMH/aqmh_gpu_optimization_implementation_guide_de.md` — die
   verbindliche Implementationsanleitung mit Work Packages (WP-A … WP-H),
   Datei-/Zeilen-Ankern, Guardrails, Test- und Rollout-Reihenfolge. **Folge ihr
   als Quelle der Wahrheit.**
2. `AGENTS.md` (Repo-Root) — Build-/Test-Konventionen und GPU-Regeln
   (insb.: GPU-Pfade müssen CPU-Semantik in Toleranz wahren und einen
   getesteten CPU-Fallback behalten).
3. Die betroffenen Quellen, bevor du sie änderst:
   - `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_cuda.cu`
   - `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_opencl.cpp`
   - `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp` (~Z. 610–688,
     die 4 RGB-/Luma-Pässe)
   - `tile_compile_cpp/src/metrics/aqmh_quality_map.cpp`
     (`accelerated_local_variance`, ~Z. 339)
   - `tile_compile_cpp/apps/runner_phase_local_metrics.cpp` (~Z. 470–690)

## Kernkontext (Zusammenfassung der Analyse)

- Bei Debayer-First-RGB läuft **AQMH_RECONSTRUCTION 4× vollständig**
  (Kern/Luma + R + G + B), sequenziell. **Q-Maps, Masken, Global-Weights und
  Cherry-Pick-Scores sind über alle 4 Pässe identisch** — nur die Pixelwerte je
  Kanal unterscheiden sich. Heute werden Q-Maps/Masken 4× gelesen, 4× gepackt,
  4× hochgeladen; Buffer/Streams 4× alloziert/freigegeben. **Das ist der größte
  Hebel.** AQMH_MAPS wird **nicht** ver-4-facht.
- Der CUDA-Kernel hat massiven Thread-privaten Speicher (`values`/`weights`/
  `scores`/`scratch_buf` je `MaxFrames`, plus `tmp_v/tmp_w` in `sigma_clip`) →
  Local-Memory-Spilling, niedrige Occupancy.
- Gather ist nicht-coalesciert (pixel-major Layout).
- Host-Commit ist elementweise/seriell.
- OpenCL sortiert **immer über 1024** unabhängig von `n` (Größenordnung zu
  langsam), ist synchron, hat noch mehr Private-Arrays, `auto_reject` = CPU.
- AQMH_MAPS lagert nur den BoxFilter aus; viele CPU-Worker teilen sich eine GPU
  mit blockierendem `waitForCompletion()` pro Scale.

## Verbindliche Guardrails

- **CPU-Parität wahren.** Leitplanken-Test:
  `aqmh_cuda_reconstruction_matches_cpu_streaming_reference` in
  `tile_compile_cpp/tests/test_aqmh_reconstruction.cpp` — darf nie schwächer
  werden.
- **CPU-Fallback erhalten und getestet.**
- **Keine Änderung persistierter Formate/Cache-Schlüssel** (Q-Map-Cache,
  `execution_backend`, Frame-Mask-Store). fp16/Bit-Packing betrifft nur
  GPU-Transfer-Staging, nie den Cache.
- **Jeder neue Datenpfad hinter einem Flag mit Default = altes Verhalten**, bis
  die Parität auf dieser GPU bestätigt ist.
- **Diagnose-Timings** (`cuda_*_seconds`) weiter befüllen; Tests werten sie aus.
- **Ein WP = ein Commit.** Grüner Build + Tests vor dem nächsten WP.
- **Nicht** den separaten Luma-Pass entfernen (Domänenentscheidung) — nur als
  Frage/Kommentar dokumentieren und beim Menschen rückfragen.

## Build & Test (diese Maschine)

```bash
cd tile_compile_cpp

# CPU-Build (Regressions-/Kompilierbarkeitscheck):
cmake -S . -B build -DBUILD_TESTS=ON -DTILE_COMPILE_ENABLE_CUDA=OFF \
      > /tmp/out_cpu_configure.txt 2>&1
cmake --build build --target tile_compile_runner tests -j"$(nproc)" \
      > /tmp/out_cpu_build.txt 2>&1
./build/tests > /tmp/out_cpu_tests.txt 2>&1

# CUDA-Build (Hauptpfad):
cmake -S . -B build-cuda -DBUILD_TESTS=ON -DTILE_COMPILE_ENABLE_CUDA=ON \
      > /tmp/out_cuda_configure.txt 2>&1
cmake --build build-cuda --target tile_compile_runner tests -j"$(nproc)" \
      > /tmp/out_cuda_build.txt 2>&1
./build-cuda/tests "[aqmh]" > /tmp/out_cuda_aqmh_tests.txt 2>&1
./build-cuda/tests > /tmp/out_cuda_all_tests.txt 2>&1
```
Bei Fehlern: Ausgabedateien lesen, Umgebungs-/Sandboxfehler von Codefehlern
trennen. Vor der ersten Änderung **einmal einen grünen Baseline-Build+Test**
herstellen und die Baseline-Timings festhalten.

## Profiling (bei jedem kernel-relevanten WP)

```bash
# Systemüberblick:
nsys profile -o /tmp/aqmh_nsys ./build-cuda/tile_compile_runner <referenz-args>
# Kernel-Detail (Occupancy, Local-Memory-Traffic, Stalls):
ncu --set full -o /tmp/aqmh_ncu -k regex:aqmh ./build-cuda/tile_compile_runner <args>
```
Dokumentiere pro WP vorher/nachher: `achieved_occupancy`, Local-Memory
load/store, dominanter Stall-Grund, sowie `cuda_h2d_seconds`,
`cuda_kernel_seconds`, `cuda_d2h_seconds`, `cuda_result_commit_seconds` und die
Reconstruction-Gesamtzeit.

## Arbeitsreihenfolge (aus dem Guide, §11)

1. **WP-F** paralleler Host-Commit (einfach, sicher).
2. **WP-A/R2** RGB-Pack-Deduplizierung (Q-Maps/Masken einmal packen, 4× nutzen).
3. **WP-G1** OpenCL Bitonic auf `n` begrenzen (nur wenn OpenCL-Runtime da).
4. **WP-A/R1** Q-Maps GPU-resident (Chunk-außen/Kanal-innen).
5. **WP-B** zweistufiger Multi-Channel-Kernel (Gather+Score+Cherry-Pick einmal,
   Sigma-Clip je Kanal).
6. **WP-C** Local-Mem/Occupancy (`tmp_v/tmp_w` eliminieren, `__launch_bounds__`,
   Blockgröße tunen).
7. **WP-E** fp16-Q-Maps / bit-gepackte Masken (Flag-optional, Toleranztest).
8. **WP-D** Coalescing-Layout (mit Profiling entscheiden).
9. **WP-H** Maps: erst GPU-BoxFilter vs. CPU-Sliding-Window messen, dann
   entscheiden (entfernen oder tiefer offloaden + Worker entkoppeln).
10. **WP-C+** Shared-Memory-kooperativer Sort — nur wenn Profiling ihn als
    Top-Stall ausweist; höchstes Risiko, separater Schritt.

Für jedes WP: neue Paritätstests ergänzen (siehe Guide §9/§10), insb.
`aqmh_cuda_session_multichannel_matches_individual_calls` und
`aqmh_cuda_two_stage_matches_single_stage`.

## Abnahme (Definition of Done, Guide §13)

- CPU- **und** CUDA-Build grün; `./build-cuda/tests "[aqmh]"` grün inkl. neuer
  Paritätstests; volle Suite grün.
- Nsight-Kennzahlen vor/nach je kernel-relevantem WP dokumentiert.
- End-to-End auf Referenzdatensatz: der 4-Pass-Reconstruction-Block ist
  nachweislich amortisiert (Timings belegen es).
- Neue Flags in `configuration_reference*.md` + Schema-JSON/YAML dokumentiert.
- CPU-Fallback unverändert funktionsfähig.

## Vorgehen & Kommunikation

- Beginne mit Baseline-Build+Test+Timings, dann WP-F. Nach jedem WP: Commit mit
  aussagekräftiger Nachricht auf `claude/aqmh-gpu-optimization-urpo5e`, kurze
  Ergebnismeldung (Timings/Kennzahlen), dann nächstes WP.
- Bei Parität-Verletzung: Ursache finden, nicht die Toleranz aufweichen. Wenn
  eine Optimierung Parität prinzipiell verletzt (z. B. fp16-Score-Drift), mache
  sie Flag-optional mit Default aus und dokumentiere die Abweichung.
- Bei Domänenfragen (z. B. Luma-Pass ableitbar?) **rückfragen**, nicht raten.
- Erstelle **keinen** Pull Request, außer der Mensch bittet ausdrücklich darum.

=== PROMPT END ===

---

## Referenzen

- Implementationsanleitung: `docs/AQMH/aqmh_gpu_optimization_implementation_guide_de.md`
- Build/Test-Konventionen: `AGENTS.md` (§„C++ Build And Tests")
- Leitplanken-Test: `tile_compile_cpp/tests/test_aqmh_reconstruction.cpp`
  (`aqmh_cuda_reconstruction_matches_cpu_streaming_reference`)
- CUDA-CMake-Optionen: `tile_compile_cpp/CMakeLists.txt` (`TILE_COMPILE_ENABLE_CUDA`)
