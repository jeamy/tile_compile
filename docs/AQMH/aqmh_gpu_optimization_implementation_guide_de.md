# AQMH GPU-Optimierung — Detaillierte Implementationsanleitung

> Ziel: „ideale" GPU-Performance für die Phasen **AQMH_RECONSTRUCTION** und
> **AQMH_MAPS**, ohne die numerische CPU-Parität und die Resume-/Cache-Verträge
> zu brechen.
>
> Diese Anleitung ist so geschrieben, dass sie **lokal auf einer Maschine mit
> CUDA-Toolkit und GPU** ausgeführt werden kann. Der zugehörige
> **Handshake-Prompt** (`aqmh_gpu_optimization_handshake_prompt_de.md`) fasst
> Reihenfolge, Guardrails und Abnahmekriterien für die lokale Ausführung
> zusammen.

---

## 0. Ausgangslage und Zielbild

### Betroffene Dateien

| Rolle | Datei |
|---|---|
| CUDA-Reconstruction-Kernel + Host-Loop | `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_cuda.cu` |
| OpenCL-Reconstruction | `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_opencl.cpp` |
| CPU-Referenz (Parität) | `tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp` |
| Dispatch | `tile_compile_cpp/src/core/acceleration.cpp` (`AccelerationOps::reconstruct_aqmh`, ~Z. 2593) |
| Header/Config | `tile_compile_cpp/include/tile_compile/reconstruction/aqmh_reconstruction.hpp` |
| RGB-/Luma-Aufrufer (4 Pässe) | `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp` (~Z. 610–688) |
| Maps-Kernel (BoxFilter-Offload) | `tile_compile_cpp/src/metrics/aqmh_quality_map.cpp` (`accelerated_local_variance`, ~Z. 339) |
| Maps-Orchestrierung (Worker/Streams) | `tile_compile_cpp/apps/runner_phase_local_metrics.cpp` (~Z. 470–690) |
| Worker-Plan | `tile_compile_cpp/apps/runner_shared.cpp` (`compute_aqmh_map_worker_plan`, ~Z. 478) |
| Parität-/Regressionstests | `tile_compile_cpp/tests/test_aqmh_reconstruction.cpp`, `test_aqmh_quality_map.cpp`, `test_reconstruction_regression.cpp` |

### Kernbefunde (verdichtet)

1. **AQMH_RECONSTRUCTION läuft bei Debayer-First-RGB 4× vollständig** (Kern/Luma
   + R + G + B), strikt sequenziell verkettet
   (`runner_phase_aqmh_reconstruction.cpp:610` und `:682–684`). Über alle 4 Pässe
   sind **Q-Maps, Frame-Masken, Canvas-Mask, Global-Weights und damit die
   Cherry-Pick-Scores identisch** — nur die Pixelwerte (`d_frames`) je Kanal
   unterscheiden sich. Aktuell werden Q-Maps/Masken 4× aus dem Cache gelesen, 4×
   host-seitig ins pixel-major Layout gepackt und 4× über PCIe hochgeladen.
2. **Der CUDA-Kernel hat massiven Thread-privaten Speicher** (`values`,
   `weights`, `scores`, `scratch_buf`, je `MaxFrames`, plus `tmp_v/tmp_w` in
   `sigma_clip`). Bei hohen Frame-Zahlen → Local-Memory-Spilling → Occupancy am
   Boden (`aqmh_reconstruction_cuda.cu:452–455`, `:313–314`, `:357–358`).
3. **Nicht-coalescierter Gather**: pixel-major Layout `canvas_idx*frame_count+fi`
   → benachbarte Threads greifen `frame_count` floats auseinander zu
   (`aqmh_reconstruction_cuda.cu:463`).
4. **Serieller Host-Commit** (elementweise `result.output(y,x)=...`,
   `:1129–1144`, `:1411–1426`).
5. **OpenCL** sortiert **immer über `MAX_FRAMES=1024`** unabhängig von `n`
   (`aqmh_reconstruction_opencl.cpp:37–39` etc.), hat noch mehr Private-Arrays,
   ist synchron/ohne Double-Buffering, und `auto_reject` fällt komplett auf CPU
   zurück.
6. **AQMH_MAPS** lagert nur den BoxFilter (lokale Varianz/Schärfe) aus; alle
   anderen Schritte laufen CPU/OpenMP. Viele CPU-Worker teilen sich eine GPU mit
   blockierendem `waitForCompletion()` pro Scale
   (`aqmh_quality_map.cpp:410`), Worker-Zahl kommt aus dem CPU-Bound-Profil.

### Erwarteter Gesamteffekt

- **WP-A/WP-B (RGB-Dedup + Multi-Channel-Batch)**: 4-Pass-Block von „~4×" auf
  „~1,3–2× eines Einzelpasses". Größter Hebel, weil genau die teuren geteilten
  Teile (Q-Map-Upload/-Pack, Score-Sort) amortisiert werden.
- **WP-C/WP-D (Occupancy + Coalescing)**: Kernel-interne Beschleunigung, stark
  hardware-abhängig; realistisch 1,5–4× auf den Kernel-Sekunden bei hohem N.
- **WP-E (Transferreduktion)**: −25…−45 % H2D-Volumen.
- **WP-G (OpenCL)**: Größenordnungen auf Nicht-CUDA-GPUs (durch `n`-Bound-Sort).
- **WP-H (Maps)**: entweder messbarer Gewinn durch tieferen Offload, oder
  bewusste Entscheidung, den BoxFilter-Roundtrip zu entfernen.

---

## 1. Guardrails (für alle Work Packages verbindlich)

- **CPU-Parität**: GPU-Ergebnisse müssen die CPU-Referenz innerhalb der in
  `test_aqmh_reconstruction.cpp` verwendeten Toleranzen reproduzieren. Der
  bestehende Test `aqmh_cuda_reconstruction_matches_cpu_streaming_reference`
  (`:293`) ist der Leitplanken-Test; er darf nie schwächer werden.
- **CPU-Fallback bleibt erhalten und getestet** (AGENTS.md: „GPU paths must
  preserve CPU semantics … and retain a tested CPU fallback").
- **Resume-/Cache-Verträge**: Cache-Schlüssel (`execution_backend` in
  `aqmh_quality_map_cache.cpp`) und die geschriebenen Artefakte
  (`out_aqmh_cache->write`, `frame_mask_store.write`) dürfen sich nicht
  unbeabsichtigt ändern. Ein Layout-/Präzisionswechsel der **persistierten**
  Q-Maps ist tabu; fp16 (WP-E) betrifft nur die **GPU-Transfer-Staging-Kopie**,
  nicht den Cache.
- **Keine stillen Verhaltensänderungen**: jede neue Datenpfad-Variante hinter
  einem Config-Flag mit unverändertem Default, bis die Parität lokal auf GPU
  bestätigt ist.
- **Diagnose-Timings erhalten**: die `cuda_*_seconds`-Felder werden von Tests
  und Report ausgewertet (`test_aqmh_reconstruction.cpp:353–357`). Neue Pfade
  müssen sie weiter befüllen (ggf. neue Felder ergänzen statt ersetzen).

---

## 2. Work Package A — 4×-RGB-Deduplizierung (Host-Orchestrierung)

**Priorität: 1 (höchster Wert, geringstes Risiko, ohne Kernel-Änderung).**

### Ziel
Die über alle 4 Pässe **identischen** Eingaben (Q-Maps, Frame-Masken,
Canvas-Mask, Global-Weights) nur **einmal** vorbereiten/hochladen und die
GPU-Buffer über die Pässe **persistent** halten, statt 4× zu allozieren/freizugeben.

### 2.1 Persistente Session statt Einzelaufrufe

Aktuell ruft `runner_phase_aqmh_reconstruction.cpp` viermal
`aqmh_reconstruction_ops.reconstruct_aqmh(...)` auf (`:610`, `:682–684`), und
`reconstruct_aqmh_weighted_cuda` alloziert/freigibt intern jedes Mal alles
(`aqmh_reconstruction_cuda.cu:927–1000`, `:1574–1585`).

**Umsetzung:** Einführung eines Session-Objekts in `aqmh_reconstruction_cuda.cu`:

```cpp
// Neuer öffentlicher Typ (Header: aqmh_reconstruction_cuda.hpp)
class AqmhCudaReconstructionSession {
 public:
  // Alloziert Buffer, Streams, Events; lädt Global-Weights;
  // bestimmt chunk_rows EINMAL. q_map_cache + Masken werden als geteilt
  // markiert.
  bool init(size_t frame_count, metrics::QualityMapCache* q_map_cache,
            const VectorXf& global_weights,
            const std::vector<uint8_t>& canvas_mask,
            int width, int height,
            const AqmhReconstructionConfig& shared_cfg,
            const AqmhMaskLoader&, const AqmhFrameRegionLoader&,
            const AqmhMaskRegionLoader&);

  // Ein Reconstruction-Durchlauf über eine Wert-Ebene (Luma oder R/G/B).
  // Verwendet die in init() hochgeladenen/gepackten Q-Maps + Masken erneut.
  AqmhReconstructionResult run_plane(
      const AqmhFrameLoader& load_frame,
      const AqmhFrameRegionLoader& load_frame_region,
      bool compute_uniform_control,
      const AqmhProgressCallback& progress);

  ~AqmhCudaReconstructionSession();  // gibt alles frei
};
```

- `init()` übernimmt die heutige Vorbereitung aus
  `reconstruct_aqmh_weighted_cuda` bis einschließlich Buffer-/Stream-/Event-Setup
  (`:866–1063`).
- `run_plane()` übernimmt die Chunk-Schleife (`:1092–1508`), **aber**:
  - Q-Maps und Masken werden nur beim **ersten** `run_plane` (bzw. in `init`)
    gepackt/hochgeladen und danach als „resident" behandelt (siehe 2.2).
  - Nur `d_frames` wird je Plane neu gepackt/hochgeladen.
- `reconstruct_aqmh_weighted_cuda(...)` bleibt als dünner Wrapper bestehen
  (Session mit genau einem `run_plane`), damit Einzelaufrufer und Tests
  unverändert funktionieren.

### 2.2 Q-Maps/Masken GPU-resident halten

Zentrale Erkenntnis: Q-Maps + Masken hängen **nicht** vom Kanal ab. Zwei
Varianten, je nach VRAM:

- **Variante R1 (bevorzugt, wenn VRAM reicht):** Q-Maps + Masken **einmal pro
  Chunk** hochladen und über die 4 Plane-Läufe im Device-Buffer belassen. Das
  bedeutet: die Chunk-Schleife wird so umgestellt, dass sie **pro Chunk alle
  Kanäle** abarbeitet (Chunk-außen, Kanal-innen). Dann wird pro Chunk
  Q-Map/Maske genau 1× hochgeladen und 4 Kernel-Launches (Luma/R/G/B) darauf
  ausgeführt, jeder mit eigener `d_frames`-Ebene und eigenem Output.
  → maximale Ersparnis, aber invasivster Umbau der Schleifenstruktur.
- **Variante R2 (einfacher, konservativ):** Chunk-außen, Kanal-innen bleibt
  ungenutzt; stattdessen wird nur der **Host-Pack** von Q-Maps/Masken einmal
  erzeugt und für alle 4 Pässe wiederverwendet (spart Cache-Reads + Host-Pack,
  aber nicht den H2D-Upload). Deutlich weniger Umbau, ~50–70 % des Gewinns.

**Empfehlung:** R2 zuerst umsetzen und messen; R1 nur, wenn die Messung den
H2D-Upload der Q-Maps als weiterhin dominant ausweist (`cuda_h2d_seconds`).

### 2.3 Score-Wiederverwendung

Der Score `gw * fmax(0,q)` (`aqmh_reconstruction_cuda.cu:479–480`) und damit die
Cherry-Pick-Sortierreihenfolge sind kanalunabhängig. In R1 (4 Kernel-Launches
pro Chunk) kann die **Score-Sortierung einmal** berechnet und die resultierende
Index-Permutation für alle 4 Kanäle wiederverwendet werden — das ist der
teuerste Kernel-Teil. Das erfordert einen Kernel, der die Sortier-Permutation in
einen Device-Buffer schreibt (Phase 1) und einen zweiten Kernel, der pro Kanal
nur Sigma-Clip + gewichteten Mittelwert auf der permutierten Reihenfolge rechnet
(Phase 2). → Deckt sich mit WP-B; dort im Detail.

### 2.4 Aufrufer anpassen

`runner_phase_aqmh_reconstruction.cpp`:
- Wenn `debayer_first_rgb` **und** Backend == CUDA: eine Session erzeugen,
  darüber Luma + R + G + B laufen lassen.
- Sonst: unveränderter Pfad (Einzelaufrufe, CPU/OpenCL).
- **Fachliche Prüfung (nicht mechanisch):** ob der separate Kern-/Luma-Pass
  (`:610`) im RGB-Fall überhaupt nötig ist oder aus R/G/B ableitbar wäre
  (`RgbLumaDetailTransfer`, `:136–194`, berechnet Luma = 0.25R+0.5G+0.25B). Das
  ist eine Domänenentscheidung — **nicht** ohne ausdrückliche Zustimmung
  umsetzen; nur als Kommentar/Frage dokumentieren.

### 2.5 Parität
- `AqmhReconstructionResult` je Plane muss bitidentisch zum heutigen
  Einzelaufruf sein (die Wiederverwendung ändert nur *wann/wie oft* Daten
  bewegt werden, nicht *was* gerechnet wird). Neuer Test:
  `aqmh_cuda_session_multichannel_matches_individual_calls` (siehe §9).

---

## 3. Work Package B — Multi-Channel-Batched Kernel (CUDA, zweistufig)

**Priorität: 2 (der eigentliche Reconstruction-Gewinn; baut auf WP-A/R1 auf).**

### Ziel
Gather + Score + Cherry-Pick-Sortierung **einmal** für alle Kanäle; nur
Sigma-Clip + gewichteter Mittelwert je Kanal.

### 3.1 Zweistufiger Kernel

- **Kernel 1 (`aqmh_select_kernel`)** pro Pixel:
  - Gather der gültigen Sample-Indizes (Maske + finite q), Score-Berechnung.
  - Cherry-Pick-Sortierung → schreibt die **selektierte Index-Liste** und
    `k_effective` in Device-Buffer (`int32` Indizes, `uint16` k je Pixel).
  - Diese Ausgabe ist **kanalunabhängig** und wird von allen Kanälen genutzt.
- **Kernel 2 (`aqmh_reduce_kernel`)** pro (Pixel, Kanal):
  - liest die selektierten Indizes, holt die Kanalwerte, führt Sigma-Clip +
    gewichteten Mittelwert aus, schreibt `output`/`weight_sum`.
  - Sigma-Clip ist kanalspezifisch (Clipping auf Werten), daher hier.

**Achtung Semantik:** Der heutige Sigma-Clip verändert die Sample-Menge
(`n`) iterativ **abhängig von den Werten**. Da die Werte je Kanal verschieden
sind, **muss** Sigma-Clip je Kanal separat laufen — das ist korrekt so. Nur
Gather + Score + Cherry-Pick werden geteilt. Die Parität bleibt exakt, weil
Cherry-Pick nur von Scores abhängt (bestätigt in
`aqmh_reconstruction_cuda.cu:485–487, 519–534`).

### 3.2 Speicher
- Selektions-Buffer: `frame_count`-breite Index-Liste je Pixel ist zu groß für
  alle Pixel gleichzeitig → nur **chunkweise** materialisieren (passt zur
  Chunk-Struktur). Alternativ nur `k_effective` + kompakte Indexliste bis
  `k_max` speichern.
- Damit sinkt der Thread-private Speicher in Kernel 2 auf die **selektierten**
  `k` Samples statt `MaxFrames` (hilft zusätzlich WP-C).

### 3.3 Fallbacks
- Wenn `cherry_pick == false`: Kernel 1 entfällt weitgehend (nur Gather/Score),
  Selektion = alle gültigen Samples. Trotzdem Gather einmal teilen.
- Bei nur 1 Kanal (kein Debayer-First-RGB): Zweistufigkeit bringt nichts →
  einstufigen Kernel behalten (Flag-gesteuert).

---

## 4. Work Package C — Occupancy / Local-Memory-Druck (CUDA)

**Priorität: 3 (hardwareabhängig, nur mit GPU verifizierbar).**

### Maßnahmen
1. **`sigma_clip` `tmp_v/tmp_w` eliminieren** (`:313–314`, `:357–358`): Die
   „keep_floor"-Permutation kann in-place über die Index-Permutation erfolgen
   (wie im Cherry-Pick-Kompaktierungszweig `:195–203`), statt zwei zusätzliche
   `MaxFrames`-Arrays anzulegen. Spart 8·MaxFrames Byte/Thread.
2. **`scratch_buf` als `short` ist gut** — beibehalten. Prüfen, ob `values`
   nach WP-B nur noch `k_max` statt `MaxFrames` braucht.
3. **`__launch_bounds__`** am Kernel setzen, um dem Compiler das Register-Budget
   vorzugeben; Blockgröße experimentell tunen (heute fix `block(32,8)`,
   `:1049`). Bei hohem Local-Mem-Druck sind kleinere Blöcke oft schneller.
4. **Optional Shared-Memory-kooperativer Sort** (ein Warp pro Pixel statt ein
   Thread): größter Occupancy-Gewinn, aber höchstes Risiko und größter Umbau —
   **nur** angehen, wenn Profiling (Nsight Compute) Local-Memory-Traffic /
   Sort als Top-Stall ausweist. Als **separates, letztes** WP behandeln.

### Verifikation
- Nsight Compute: `achieved_occupancy`, `local_memory` load/store, `stall_lg`
  vor/nach jeder Maßnahme dokumentieren.

---

## 5. Work Package D — Coalescing / Layout (CUDA)

**Priorität: 3 (mit WP-B kombinierbar).**

- Heute pixel-major (`canvas_idx*frame_count+fi`) → Gather über den Warp
  nicht-coalesciert (`:463`).
- **Option D1:** Beim H2D bereits **frame-major** (`fi*chunk_pixels+canvas_idx`)
  packen und den Gather-Load coalesciert über Shared-Memory-Staging in die
  thread-privaten Arrays ziehen. Der Rest des Kernels bleibt pixel-lokal.
- **Trade-off:** pixel-major hilft dem seriellen Per-Thread-Lesen; die
  Coalescing-Variante hilft der Bandbreite. Nur mit Profiling entscheiden.
- **Wichtig:** Host-Pack-Layout, Kernel-Indexierung und (falls resident, WP-A)
  die Q-Map-/Masken-Buffer müssen konsistent umgestellt werden.

---

## 6. Work Package E — Transferreduktion (CUDA)

**Priorität: 3.**

1. **Q-Maps als `__half` (fp16) stagen**: Q geht nur als `fmax(0,q)`-Gewicht
   ein; fp16 im **Transfer** ist ausreichend, sofern der Score in fp32
   dequantisiert wird. **Nicht** den persistierten Cache ändern — nur die
   Host-Staging-Kopie + Device-Buffer. Kernel liest `__half`, konvertiert zu
   `float`. Halbiert das Q-Map-H2D-Volumen.
   - Parität prüfen: fp16-Quantisierung des Scores kann Cherry-Pick-Cutoffs
     minimal verschieben. Toleranztest nötig; bei Verletzung fp16 nur optional
     (Flag).
2. **Masken bit-packen** (1 Bit statt 1 Byte): 8× kleiner. Kernel entpackt per
   Bit-Test. H2D-Maskenvolumen −87,5 %.
3. Beide hinter Flags (`cfg.gpu_half_qmaps`, `cfg.gpu_packed_masks`), Default
   aus, bis Parität lokal bestätigt.

---

## 7. Work Package F — Paralleler Host-Commit (CUDA)

**Priorität: 2 (einfach, sicher, CPU-baubar).**

- Heute elementweise Commit-Schleifen (`:1129–1144`, `:1411–1426`,
  `:1451–1465`, `:1486–1500`).
- Da `result.output` (Eigen `Matrix2Df`, row-major?) und `h_output` zeilenweise
  gleiches Layout haben, pro Zeile `std::memcpy`/`Eigen::Map`-Block-Assign statt
  Skalarkopie; die Schleife mit `#pragma omp parallel for` über `yy`
  parallelisieren.
- **Achtung Eigen-Speicherordnung prüfen:** `Matrix2Df` Spalten-/Zeilenordnung
  in `core/types.hpp` verifizieren, bevor `memcpy` verwendet wird; sonst nur
  OpenMP-parallelisierte Skalarkopie.
- `cuda_result_commit_seconds` weiter befüllen.

---

## 8. Work Package G — OpenCL-Reconstruction

**Priorität: 2 für den Sort-Fix (großer Gewinn, überschaubares Risiko).**

1. **Bitonic auf `n` begrenzen**: statt Schleifen bis `MAX_FRAMES=1024`
   (`:37–39`, `:63–65`, `:90–92`, `:118–120`) die nächste Zweierpotenz `≥ n`
   verwenden und Sentinel-Padding nur bis dahin. Größter Einzelgewinn im
   OpenCL-Pfad.
2. **Private-Arrays reduzieren** (`sort_indices`, `deviations`, `sorted_values`,
   `control_*`, `:250–252`, `:479–480`) — zusammenlegen/wiederverwenden.
3. **Async + Double-Buffering** analog CUDA (`kernel.run(...,true)` blockiert,
   `:798`); mindestens Uploads/Downloads über nicht-blockierende `UMat`-Pfade.
4. **`auto_reject` auf GPU** statt CPU-Fallback (`:575–584`) — optional,
   niedrigere Priorität.
5. OpenCL hat **keine** Parität-Absicherung via Test — vor Merge einen
   CPU-Vergleichstest analog `aqmh_cuda_reconstruction_matches_cpu_streaming_reference`
   ergänzen (nur wenn OpenCL-Runtime im CI/lokal verfügbar).

---

## 9. Work Package H — AQMH_MAPS

**Priorität: 3–4 (erst nach Reconstruction, da Maps nicht ver-4-facht wird).**

1. **Entscheidungsmessung zuerst:** GPU-BoxFilter (`accelerated_local_variance`,
   `:339`) vs. CPU-Sliding-Window (`local_variance_linear`, `:252`) bei realer
   Auflösung/Frame-Zahl benchmarken. Der CPU-Pfad ist transferfrei und O(Pixel);
   der GPU-Pfad hat pro Scale Up/Download + `waitForCompletion()` (`:410`).
   - Wenn CPU ≥ GPU: BoxFilter-Offload **entfernen** oder nur bei sehr großen
     Kernels/Auflösungen aktivieren.
2. **Falls GPU bleibt:**
   - **Worker-Zahl für GPU-Backend entkoppeln** (heute `CpuBound`,
     `runner_shared.cpp:491–492`): bei CUDA wenige Worker (1–2) mit tiefen
     Streams, statt vieler CPU-Threads, die eine GPU blockieren.
   - **Tiefer offloaden**: SNR/Artefakt/Laplacian ebenfalls auf GPU, über
     **alle Scales/Frames gebatcht**, statt pro Scale ein blockierender Filter.
   - `waitForCompletion()` pro Scale vermeiden → Stream über die Scales laufen
     lassen, erst am Frame-Ende synchronisieren.
3. Parität via `test_aqmh_quality_map.cpp` (bestehende Toleranzen).

---

## 10. Test- und Paritätsstrategie

### Build
```bash
cd tile_compile_cpp
# Ohne GPU (CPU-Pfade, Host-Logik, Kompilierbarkeit der .cpp):
cmake -S . -B build -DBUILD_TESTS=ON > /tmp/out_cpp_configure.txt 2>&1
cmake --build build --target tile_compile_runner tests -j2 > /tmp/out_cpp_build.txt 2>&1
./build/tests > /tmp/out_cpp_tests.txt 2>&1

# Mit GPU (CUDA-Backend aktiv):
cmake -S . -B build-cuda -DBUILD_TESTS=ON -DTILE_COMPILE_ENABLE_CUDA=ON \
      > /tmp/out_cuda_configure.txt 2>&1
cmake --build build-cuda --target tile_compile_runner tests -j2 \
      > /tmp/out_cuda_build.txt 2>&1
./build-cuda/tests "[aqmh]" > /tmp/out_cuda_tests.txt 2>&1
```
> Hinweis: `TILE_COMPILE_ENABLE_CUDA` defaultet auf ON, wenn `nvcc` gefunden wird
> (`CMakeLists.txt:74–91`). Für reine CPU-Builds explizit `=OFF` setzen.

### Neue Tests (mindestens)
- `aqmh_cuda_session_multichannel_matches_individual_calls`: Session (Luma+R+G+B)
  vs. 4 Einzelaufrufe → bitidentische `output`/`weight_sum` je Plane.
- `aqmh_cuda_two_stage_matches_single_stage` (WP-B): zweistufiger Kernel vs.
  heutiger einstufiger Kernel.
- Toleranztests für WP-E (fp16 q-maps, packed masks) gegen CPU-Referenz.
- OpenCL-CPU-Vergleich (WP-G), sofern OpenCL-Runtime vorhanden.

### Verifikationsreihenfolge pro WP
1. CPU-Build grün + `./build/tests` grün (Kompilierbarkeit, keine Regression).
2. CUDA-Build grün.
3. `[aqmh]`-Tests grün, inkl. neuer Paritätstests.
4. Profiling (Nsight Systems/Compute) vor/nach → Kennzahlen dokumentieren.
5. End-to-End-Lauf `tile_compile_runner` auf einem Referenz-Datensatz;
   `cuda_*_seconds` und Gesamt-Phasenzeit vergleichen.

---

## 11. Empfohlene Umsetzungsreihenfolge (Rollout)

| Reihenfolge | WP | Baubar ohne GPU? | Risiko | Gewinn |
|---|---|---|---|---|
| 1 | **F** Host-Commit parallel | ja | niedrig | klein–mittel |
| 2 | **A/R2** RGB-Pack-Dedup | ja (Logik), GPU-Verify | niedrig | groß |
| 3 | **G1** OpenCL Bitonic-`n`-Bound | ja (Kompil.), OpenCL-Verify | niedrig | groß (OpenCL) |
| 4 | **A/R1** Q-Maps resident (Chunk-außen/Kanal-innen) | GPU nötig | mittel | groß |
| 5 | **B** Zweistufiger Multi-Channel-Kernel | GPU nötig | mittel–hoch | groß |
| 6 | **C** Local-Mem/Occupancy (ohne Shared-Sort) | GPU nötig | mittel | mittel |
| 7 | **E** fp16 q-maps / packed masks | GPU nötig | mittel | mittel |
| 8 | **D** Coalescing-Layout | GPU nötig | mittel | mittel |
| 9 | **H** Maps (Messung → Entscheidung) | teils | mittel | mittel |
| 10 | **C+** Shared-Memory-kooperativer Sort | GPU nötig | hoch | groß (bei hohem N) |

Jedes WP als **eigener Commit** auf `claude/aqmh-gpu-optimization-urpo5e`, mit
grünem Build + Tests vor dem nächsten. Neue Datenpfade hinter Flags mit Default
= altes Verhalten, bis GPU-Parität lokal bestätigt ist.

---

## 12. Risiken und Stolperfallen

- **Eigen-Speicherordnung** (`Matrix2Df`) vor jedem `memcpy` prüfen (WP-F/A).
- **fp16-Score-Drift** kann Cherry-Pick-Cutoffs kippen → Paritätstoleranz eng
  halten, sonst Flag-optional (WP-E).
- **VRAM-Budget** in R1: Q-Maps resident + 4 Output-Sets erhöhen den
  Peak-Verbrauch; die adaptive Chunk-/Double-Buffer-Logik
  (`:924–949`) muss den Multi-Channel-Fall einkalkulieren.
- **Resume-Cache-Invalidierung** vermeiden: keine Änderung an persistierten
  Q-Map-/Masken-Formaten oder `execution_backend`-Schlüsseln.
- **Sandbox ohne GPU** ≠ „CUDA absent" (AGENTS.md:84): CUDA-Pfade nicht
  wegoptimieren, nur weil die Build-Umgebung keine GPU sieht.
- **Nsight-Profiling** braucht echte GPU + ggf. erhöhte Kernel-Trace-Rechte.

---

## 13. Definition of Done

- Alle umgesetzten WP haben grünen CPU- **und** CUDA-Build.
- `./build-cuda/tests "[aqmh]"` grün inkl. neuer Paritätstests.
- Für jedes Kernel-relevante WP: Nsight-Kennzahlen vor/nach dokumentiert.
- End-to-End: Reconstruction-Phasenzeit auf Referenzdatensatz gemessen und der
  4-Pass-Block nachweislich amortisiert (`cuda_h2d_seconds`,
  `cuda_kernel_seconds`, Gesamt).
- Config-Doku (`configuration_reference*.md`, Schema-JSON/YAML) aktualisiert,
  falls neue Flags eingeführt wurden (AGENTS.md §Configuration).
- CPU-Fallback unverändert funktionsfähig und getestet.
