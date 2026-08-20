# AQMH GPU-Optimierung — Detaillierte Implementationsanleitung

> **Hinweis zur Entstehung:** Diese Anleitung wurde in einer Analyse-Session ohne
> lokalen GPU-/`nvcc`-Zugriff erstellt. Sie beschreibt Befunde und Maßnahmen anhand
> von Code-Lesen, nicht von Profiling-Messungen auf echter Hardware. Jede Work
> Package (WP) muss auf einer Maschine mit echtem CUDA-Build vor dem Merge
> **gemessen** werden (siehe §10). Nutze das begleitende Dokument
> `aqmh_gpu_optimization_handshake_prompt_de.md`, um eine lokale Claude-Code-Session
> mit diesem Kontext zu starten.

## 0. Ausgangslage & Zielbild

### 0.1 Betroffene Dateien

| Datei | Rolle |
|---|---|
| `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_cuda.cu` | CUDA-Kernel + Host-Orchestrierung für AQMH_RECONSTRUCTION |
| `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_opencl.cpp` | OpenCL-Spiegel derselben Logik (über `cv::ocl`) |
| `tile_compile_cpp/src/metrics/aqmh_quality_map.cpp` | AQMH_MAPS; nur lokale Varianz/Sharpness ist GPU-offloadet |
| `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp` | Ruft die Rekonstruktion 4× auf (Luma/Core + R + G + B) |
| `tile_compile_cpp/apps/runner_phase_local_metrics.cpp` | Worker-Pool-Orchestrierung für AQMH_MAPS |
| `tile_compile_cpp/apps/runner_shared.cpp` | `compute_aqmh_map_worker_plan` (Worker-Anzahl-Planung) |
| `tile_compile_cpp/src/core/acceleration.cpp` | Backend-Dispatch (`AccelerationOps::reconstruct_aqmh`) |
| `tile_compile_cpp/include/tile_compile/reconstruction/aqmh_reconstruction.hpp` | `AqmhReconstructionConfig`/`AqmhReconstructionResult`, Loader-Typaliase |
| `tile_compile_cpp/tests/test_aqmh_reconstruction.cpp` | Parity-Test (`aqmh_cuda_reconstruction_matches_cpu_streaming_reference`, Zeile ~293) |

### 0.2 Kernbefunde (kondensiert)

1. **4× redundante Rekonstruktionsdurchläufe (Luma + R + G + B).** Jeder Durchlauf
   allokiert eigene GPU-Puffer, lädt Q-Maps/Masken/Gewichte neu und durchläuft den
   kompletten Kernel — obwohl Q-Maps, Canvas-Maske und globale Gewichte für alle
   vier Kanäle identisch sind. `runner_phase_aqmh_reconstruction.cpp:681-684` ruft
   `reconstruct_rgb_plane` sequentiell dreimal nach dem Luma/Core-Pass auf.
2. **Thread-private Arrays im Kernel** (`aqmh_reconstruction_cuda.cu:452-454`,
   `values[MaxFrames]`, `weights[MaxFrames]`, `scores[...]`) sind bei den größeren
   Tiers (512/640/768/1024 Frames) zu groß für Register und spillen in Local Memory
   — das kostet Bandbreite und senkt Occupancy. Zusätzliche temporäre Arrays
   `tmp_v`/`tmp_w` (Zeilen ~313-314, ~357-358) verschärfen das.
3. **Pixel-major Layout** (`aqmh_reconstruction_cuda.cu:463`) führt zu unkoalesziertem
   globalem Speicherzugriff über Frames hinweg.
4. **Serieller Host-Commit** nach jedem Chunk (mehrere Stellen, u. a.
   ~1129-1144, ~1411-1426, ~1451-1465, ~1486-1500) statt paralleler/asynchroner
   Übertragung.
5. **Buffer-Alloc/Free pro Aufruf** (~927-1000, ~1574-1585) statt Session-persistenter
   Puffer — bei 4 Aufrufen pro Bild vervierfacht sich der `cudaMalloc`/`cudaFree`-Overhead.
6. **OpenCL-Pfad ist strukturell schlechter dran:** Bitonic Sort läuft immer über
   `MAX_FRAMES=1024`, unabhängig vom tatsächlichen `n` (Zeilen ~37-142); zusätzliche
   thread-private Arrays (`sort_indices`, `deviations`, `sorted_values`,
   `control_values`/`control_weights`); synchrones, blockierendes `kernel.run(...,true)`
   (Zeile ~798) ohne Double-Buffering; `auto_reject`-Modus fällt komplett auf CPU
   zurück (~575-584).
7. **AQMH_MAPS:** Nur `accelerated_local_variance` (BoxFilter, ~339-462) ist
   GPU-offloadet; der Rest der Pyramide (Laplacian, SNR, Artefakt-Erkennung, PSI)
   läuft CPU-seitig. Worker-Planung (`runner_shared.cpp` `compute_aqmh_map_worker_plan`,
   ~478-520) nutzt `WorkerParallelProfile::CpuBound` auch wenn ein GPU-Backend aktiv
   ist — viele CPU-Threads serialisieren dann auf einer GPU.

### 0.3 Erwarteter Effekt (grobe Einordnung, auf echter HW zu verifizieren)

| Maßnahme | Erwarteter Hebel | Risiko |
|---|---|---|
| WP-A (4×-Dedup) | groß (Q-Map/Masken-Reload entfällt 3×) | mittel (State-Management) |
| WP-B (Two-Stage-Kernel) | mittel–groß bei großen Tiers | mittel |
| WP-C (Occupancy/Local-Mem) | mittel | niedrig |
| WP-D (Coalescing) | mittel | niedrig |
| WP-E (fp16/bit-packed) | klein–mittel (Bandbreite) | mittel (Genauigkeit!) |
| WP-F (paralleler Host-Commit) | klein–mittel, einfach zu holen | niedrig |
| WP-G (OpenCL n-bound + async) | groß für OpenCL-Pfad | niedrig–mittel |
| WP-H (Maps GPU-Ausbau) | unklar, erst messen | mittel |

## 1. Guardrails (verbindlich für alle WPs)

- **CPU-Fallback bleibt immer vorhanden und getestet.** Kein Entfernen von
  CPU-Pfaden. (AGENTS.md: "GPU paths must preserve CPU semantics within
  documented tolerances and retain a tested CPU fallback.")
- **Persistiertes Datenformat ändert sich nicht** (Cache-Dateien, Q-Map-Layout auf
  Disk) ohne expliziten Auftrag.
- **Jede WP ist flag-gated** (neuer Pfad nur aktiv, wenn explizit gewählt/aktiviert),
  bis Parität bewiesen ist.
- **Ein Commit pro WP.** Kein Sammel-Commit über mehrere WPs.
- **Parity-Test ist die Leine:** `aqmh_cuda_reconstruction_matches_cpu_streaming_reference`
  (`tests/test_aqmh_reconstruction.cpp:293`) muss nach jeder WP grün bleiben —
  inklusive der Timing-Felder (`gpu.cuda_host_prepare_seconds`,
  `gpu.cuda_host_chunk_setup_seconds`, `gpu.cuda_host_frame_read_worker_seconds`,
  `gpu.cuda_host_q_map_read_worker_seconds`, `gpu.cuda_host_mask_read_worker_seconds`).
  Toleranzen im Test **nicht** aufweichen, um ihn grün zu bekommen.
- **Domänenentscheidungen nicht selbst treffen.** Insbesondere: Ob der separate
  Luma/Core-Pass entfallen und Luma stattdessen aus R/G/B abgeleitet werden darf
  (`rgb_luma = 0.25*r + 0.50*g + 0.25*b`, `RgbLumaDetailTransfer`,
  `runner_phase_aqmh_reconstruction.cpp:192`), ist eine fachliche Frage
  (Validierungsmetriken hängen am separaten Luma-Pass) — **vorher fragen**, nicht
  eigenmächtig entfernen.

## 2. WP-A — 4×-RGB-Dedup (persistente Session)

### 2.1 Grundidee: `AqmhCudaReconstructionSession`

Statt pro Aufruf (`reconstruct_aqmh_weighted_cuda`) Puffer zu allokieren, Q-Maps/
Maske hochzuladen und am Ende freizugeben, eine Session-Klasse einführen, die
über die 4 Aufrufe (Luma, R, G, B) hinweg lebt:

```cpp
// Skizze — Header ergänzen in aqmh_reconstruction.hpp oder neuen
// aqmh_reconstruction_cuda_session.hpp
class AqmhCudaReconstructionSession {
 public:
  AqmhCudaReconstructionSession(int width, int height,
                                 const std::vector<uint8_t>& canvas_mask,
                                 const AqmhReconstructionConfig& cfg);
  ~AqmhCudaReconstructionSession();

  // Lädt Q-Maps/Maske/Gewichte EINMAL hoch (für den ersten Kanal).
  void upload_shared_inputs(metrics::QualityMapCache* q_map_cache,
                             const VectorXf& global_weights);

  // Rekonstruiert einen Kanal; nutzt die bereits residenten Q-Maps/Maske.
  AqmhReconstructionResult reconstruct_channel(
      size_t frame_count, const AqmhFrameLoader& load_frame,
      const AqmhMaskLoader& load_frame_valid_mask,
      const AqmhFrameRegionLoader& load_frame_region,
      const AqmhMaskRegionLoader& load_frame_valid_mask_region,
      const AqmhProgressCallback& progress);

 private:
  // persistente Device-Puffer über alle Kanäle:
  float* d_q_maps_ = nullptr;        // resident für Luma/R/G/B
  uint8_t* d_canvas_mask_ = nullptr; // resident
  float* d_global_weights_ = nullptr; // resident (identisch für alle Kanäle)
  cudaStream_t stream_ = nullptr, stream2_ = nullptr;
  // Frame-Daten (values) bleiben PRO Kanal neu, da R/G/B/Luma unterschiedliche
  // Pixelwerte haben — nur Q-Maps/Maske/Gewichte sind kanalübergreifend gleich.
};
```

### 2.2 Q-Map/Masken-Residenz — zwei Varianten

- **R1 (konservativ, zuerst umsetzen):** Q-Maps/Maske/Gewichte bleiben über die
  gesamte Chunk-Schleife eines Kanals resident (Status quo pro Kanal), aber die
  **Chunk-Puffer für `q_maps`/`canvas_mask`/`weight_sum`-Infrastruktur werden
  zwischen den 4 Kanal-Aufrufen wiederverwendet** (`cudaMalloc` einmal, nicht 4×).
  Spart Alloc/Free-Overhead (Befund 5), ändert aber noch nicht das Reload-Problem
  von Q-Maps (Befund 1).
- **R2 (weitergehend):** Q-Maps und Canvas-Maske werden echt nur **einmal**
  hochgeladen (beim ersten Kanal) und für R/G/B wiederverwendet, da sie
  kanalunabhängig sind. Erfordert, dass `reconstruct_channel` die
  Upload-Schritte für Q-Maps/Maske überspringen kann, wenn bereits resident.
  Das ist der Schritt mit dem größten Hebel aus Befund 1.

Reihenfolge: **erst R1, dann R2** (siehe Rollout-Tabelle §11) — R1 ist risikoarm
und schon ein Gewinn; R2 verlangt sorgfältiges State-Tracking (welche Chunks/
Regionen sind schon geladen) und mehr Tests.

### 2.3 Score-Wiederverwendung (Cherry-Pick)

Falls `cherry_pick` aktiv ist: Die Scores basieren auf den Q-Maps (kanalunabhängig)
kombiniert mit den Pixelwerten (kanalabhängig) — prüfen, ob der score-relevante
Anteil, der nur von Q-Maps abhängt, zwischengespeichert werden kann. Nur umsetzen,
wenn Profiling zeigt, dass Score-Berechnung ein messbarer Anteil ist (nicht blind
vorab optimieren).

### 2.4 Caller-Anpassung

`runner_phase_aqmh_reconstruction.cpp` (~610-684): `AqmhCudaReconstructionSession`
einmal vor dem Luma/Core-Pass erzeugen, für alle 4 `reconstruct_aqmh`/
`reconstruct_rgb_plane`-Aufrufe wiederverwenden, danach freigeben. Der
Dispatch über `core::AccelerationOps::reconstruct_aqmh` (`acceleration.cpp:2593-2626`)
muss dafür entweder eine Session-Variante bekommen oder die Session unterhalb
dieser Schicht leben (Session lebt in `aqmh_reconstruction_cuda.cu`, wird über
einen opaken Handle durchgereicht).

**Wichtig:** Ob der separate Luma/Core-Pass dabei ganz entfallen kann (Luma aus
R/G/B ableiten statt einen 4. Pass zu rechnen), **nicht eigenmächtig entscheiden**
— das ändert Validierungsmetriken-Semantik (`rgb_luma_validation`,
`raw_rgb_luma_reference`, ~Zeilen 1277-1330). Erst den User fragen.

### 2.5 Neuer Test

Ergänzender Test (analog zum bestehenden Parity-Test), der die 4-Kanal-Session
gegen 4 unabhängige Einzelaufrufe (heutiges Verhalten) vergleicht — Ergebnisse
müssen bitidentisch bzw. innerhalb der bestehenden Toleranz sein, und die
Session-Variante muss messbar schneller sein (Timing-Felder vergleichen).

## 3. WP-B — Two-Stage-Kernel (Select + Reduce)

Statt eines monolithischen Kernels, der pro Pixel Sortierung, Sigma-Clipping und
gewichtete Aggregation in einem Thread erledigt (was die großen thread-privaten
Arrays erzwingt), in zwei Kernel aufteilen:

1. **Select-Kernel:** bestimmt pro Pixel die Menge der behaltenen Samples
   (Cherry-Pick + Sigma-Clip-Maske) und schreibt eine kompakte Bitmaske/Indexliste
   nach Global Memory statt alles im Register/Local Memory zu halten.
2. **Reduce-Kernel:** liest die Maske und aggregiert (gewichteter Mittelwert) —
   kann mit kleineren, tier-angepassten Registerbudgets arbeiten, da hier keine
   vollständige Sortierung mehr nötig ist (nur noch Reduktion über die bereits
   selektierte Teilmenge).

Nutzen: bricht die MaxFrames-großen Stack-Arrays auf; erlaubt bessere Occupancy
für den (teureren) Select-Schritt getrennt vom (leichteren) Reduce-Schritt.
Kosten: ein zusätzlicher Kernel-Launch + Zwischenspeicher in Global Memory —
nur sinnvoll, wenn WP-C/D die Registerdruck-Probleme nicht schon ausreichend lösen.
**Erst nach WP-C/D messen, ob WP-B überhaupt noch nötig ist.**

## 4. WP-C — Occupancy & Local-Memory-Druck

- Register-Nutzung pro Tier mit `nvcc --ptxas-options=-v` bzw. `nsight compute`
  prüfen (`launch_reconstruction_kernel_for_frame_count`, Tier-Dispatch bei
  32/64/128/256/512/640/768/1024 Frames).
- `__launch_bounds__` pro Tier justieren, falls der Compiler ungünstig registriert.
- Prüfen, ob `tmp_v`/`tmp_w` (Zeilen ~313-314, ~357-358) durch In-Place-Umsortierung
  der bestehenden `values`/`weights`-Arrays vermieden werden können (spart
  MaxFrames-große Extra-Arrays).
- Bei sehr großen Tiers (768/1024) evaluieren, ob ein Shared-Memory-Tile pro
  Threadblock (statt rein thread-privater Arrays) die Local-Memory-Spills
  reduziert — Trade-off gegen Bank-Conflicts sorgfältig messen.

## 5. WP-D — Speicherlayout / Coalescing

- Aktuelles Pixel-major-Layout (`aqmh_reconstruction_cuda.cu:463`) auf
  Frame-major bzw. ein gekacheltes Layout (z. B. Pixel-Block × Frame) umstellen,
  sodass benachbarte Threads (= benachbarte Pixel) beim Lesen eines gegebenen
  Frames benachbarte Adressen berühren.
- Das betrifft sowohl den Host-seitigen Chunk-Aufbau (SoA-Umkopieren beim Laden)
  als auch den Kernel-Zugriff selbst — beide Seiten konsistent ändern.
- Mit `ncu --set full` Memory-Throughput vor/nach vergleichen.

## 6. WP-E — Reduzierte Bandbreite (fp16 Q-Maps, bit-gepackte Masken)

- Q-Maps von `float32` auf `float16`/`__half` für die Device-Repräsentation
  reduzieren (Host-seitig bleibt `float32` als Wahrheit; Konvertierung beim
  Upload). **Genauigkeitseinfluss muss explizit gegen den Parity-Test geprüft
  werden** — ggf. nur für Q-Maps (Gewichtungsfaktor), nicht für rekonstruierte
  Pixelwerte selbst.
- Canvas-/Frame-Masken sind bereits `uint8_t` pro Pixel — auf 1-Bit-gepackt
  (`uint32_t`-Wörter à 32 Pixel) umstellen spart Bandbreite beim Laden, kostet
  Bit-Test-Overhead im Kernel. Nur lohnend, wenn Masken-Load laut Profiling
  ins Gewicht fällt.
- **Diese WP ändert reale Zahlenwerte (Rundung) — nur nach expliziter Freigabe
  und mit verschärftem, nicht aufgeweichtem Toleranz-Check umsetzen.**

## 7. WP-F — Paralleler/asynchroner Host-Commit

- Die seriellen Commit-Loops nach jedem Chunk (~1129-1144, ~1411-1426,
  ~1451-1465, ~1486-1500) auf `cudaMemcpyAsync` + Event-basierte Synchronisation
  umstellen, sodass Host-Commit von Chunk N mit GPU-Arbeit an Chunk N+1
  überlappt (Double-Buffering existiert für `stream`/`stream2` bereits teilweise —
  konsequent für alle Commit-Stellen nutzen, nicht nur einen Teil).
- Das ist die risikoärmste WP (reine Host-Orchestrierung, keine Kernel-Änderung,
  keine Zahlenänderung) — **als erste WP umsetzen** (siehe Rollout §11).

## 8. WP-G — OpenCL: n-bound Sort + Async/Double-Buffer

- **G1 (groß, einfach):** Bitonic Sort im OpenCL-Kernel läuft aktuell immer über
  `MAX_FRAMES=1024` (~Zeilen 37-142), unabhängig vom tatsächlichen `n`. Auf die
  nächstgrößere Zweierpotenz $\ge n$ begrenzen (Padding-Elemente mit neutralem
  Sentinel-Wert, wie es der CUDA-Pfad mit `MaxFrames`-Tiering bereits tut).
  Das ist der größte Einzelhebel für den OpenCL-Pfad.
- Thread-private Arrays reduzieren (`sort_indices`, `deviations`, `sorted_values`,
  `control_values`/`control_weights`, ~250-252/479-480) analog zu WP-C/B-Ideen.
- Synchrones `kernel.run(...,true)` (~Zeile 798) durch asynchrones Enqueue +
  Double-Buffering ersetzen (analog WP-F für CUDA).
- `auto_reject`-Modus, der aktuell komplett auf CPU zurückfällt (~575-584): erst
  messen, ob das in der Praxis überhaupt ins Gewicht fällt, bevor eine
  GPU-Implementierung investiert wird.

## 9. WP-H — AQMH_MAPS: Messen, dann entscheiden

- Vor jeder Erweiterung des GPU-Anteils in `compute_aqmh_quality_map`
  (`aqmh_quality_map.cpp:950-1086`) zuerst profilen, welcher Pyramiden-Schritt
  (Laplacian, SNR, Artefakt-Erkennung, PSI-Kombination) tatsächlich Zeit kostet.
  Nicht blind alles auf GPU verlagern.
- Worker-Planung (`runner_shared.cpp` `compute_aqmh_map_worker_plan`, ~478-520)
  von `WorkerParallelProfile::CpuBound` auf ein GPU-bewusstes Profil umstellen,
  wenn ein GPU-Backend aktiv ist — sonst serialisieren viele CPU-Worker auf
  einer GPU-Stream-Queue (`WorkerCudaStreams`, `acceleration.cpp:2095-2134`).
  Das ist unabhängig von echten Kernel-Änderungen und risikoarm.

## 10. Build/Test-Kommandos

```bash
# CPU-only Referenzbuild (immer zuerst, als Baseline)
cmake -S tile_compile_cpp -B build -DBUILD_TESTS=ON \
  -DTILE_COMPILE_ENABLE_CUDA=OFF > /tmp/out_cpp_configure.txt 2>&1
cmake --build build --target tile_compile_runner tests -j$(nproc) \
  > /tmp/out_cpp_build.txt 2>&1
./build/tests "[aqmh]" > /tmp/out_cpp_tests.txt 2>&1

# CUDA-Build (auf Maschine mit nvcc/GPU)
cmake -S tile_compile_cpp -B build_cuda -DBUILD_TESTS=ON \
  > /tmp/out_cuda_configure.txt 2>&1   # TILE_COMPILE_ENABLE_CUDA auto-ON bei nvcc
cmake --build build_cuda --target tile_compile_runner tests -j$(nproc) \
  > /tmp/out_cuda_build.txt 2>&1
./build_cuda/tests "[aqmh]" > /tmp/out_cuda_tests.txt 2>&1

# Gezielt der Parity-Test:
./build_cuda/tests "aqmh_cuda_reconstruction_matches_cpu_streaming_reference"

# Profiling (jeweils pro WP vor/nach vergleichen)
nsys profile -o /tmp/aqmh_wp_X ./build_cuda/tile_compile_runner <args>
ncu --set full -o /tmp/aqmh_wp_X_ncu ./build_cuda/tile_compile_runner <args>
```

## 11. Rollout-Reihenfolge

| # | WP | Begründung für Reihenfolge |
|---|---|---|
| 1 | F | risikoärmste, reine Host-Orchestrierung, kein Zahlenrisiko |
| 2 | A/R2 | größter Einzelhebel (Q-Map-Reload 3× entfällt) |
| 3 | G1 | größter Einzelhebel im OpenCL-Pfad, unabhängig von A/B/C |
| 4 | A/R1 | falls R2 zu riskant/komplex zuerst — Fallback-Zwischenschritt |
| 5 | B | nur falls C/D den Registerdruck nicht ausreichend lösen |
| 6 | C | Occupancy/Local-Memory, nach Messen mit `ncu` |
| 7 | E | Bandbreite senken — nur nach expliziter Freigabe (Zahlenänderung) |
| 8 | D | Coalescing/Layout — größerer Umbau, spät wegen Testaufwand |
| 9 | H | Maps-GPU-Ausbau nur nach Profiling-Beleg |
| 10 | C+ | Nachjustierung Occupancy nach allen Layoutänderungen |

## 12. Risiken

- **State-Bugs in der Session-Klasse (WP-A):** veraltete/falsch wiederverwendete
  Q-Maps zwischen Kanälen — durch den ergänzenden Test in §2.5 abfangen.
- **fp16-Rundung (WP-E):** kann Sigma-Clipping-Entscheidungen an Schwellwert-Kanten
  kippen — nur mit explizitem Auftrag und verschärftem Toleranztest.
- **OpenCL-Padding-Sentinel (WP-G1):** falscher Sentinel-Wert kann Sortierreihenfolge
  verfälschen — Sentinel muss garantiert "schlechter als jeder reale Wert" sein
  und aus der finalen Aggregation ausgeschlossen bleiben.
- **Domänenfrage Luma-Pass:** nicht eigenmächtig entfernen (siehe §1, §2.4).

## 13. Definition of Done (pro WP)

- [ ] Baseline-Messung (vor der Änderung) vorhanden (nsys/ncu oder zumindest Wall-Time).
- [ ] Änderung flag-gated, CPU-Fallback unverändert.
- [ ] Parity-Test grün, Toleranzen nicht aufgeweicht.
- [ ] Neue/angepasste Tests für die spezifische WP (falls zutreffend, s. §2.5).
- [ ] Nachher-Messung zeigt tatsächlichen Gewinn (nicht nur angenommen).
- [ ] Ein Commit, klare Commit-Message mit WP-Kennung (z. B. `perf(aqmh-cuda): WP-F parallel host commit`).
- [ ] Bei Domänenfragen (z. B. Luma-Pass-Entfernung): explizite Rückfrage vor Umsetzung.
