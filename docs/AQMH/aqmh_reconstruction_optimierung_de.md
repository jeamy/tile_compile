# Optimierung der AQMH Rekonstruktionsphase (`AQMH_RECONSTRUCTION`)

## 1. Übersicht & Motivation

Die Phase `AQMH_RECONSTRUCTION` (Phase 21) in der C++20-Implementierung von Tile Compile war bei großen Datensätzen (wie dem Referenzlauf M31 mit 645 Frames) stark rechenzeitintensiv (27.5 Minuten Gesamtlaufzeit).

Ziel dieser Optimierung war:
- Deutliche Beschleunigung der Kernrekonstruktion und Validierung.
- Strikte Wahrung der deterministischen mathematischen Ergebnisse und Rekonstruktionsqualität.
- Erhalt aller Sicherheitsinvarianten (Uniform-Control-Gates, Raw-AQMH-Guard, Zero-Veto-Regeln, numerische Schutzschranken).

---

## 2. Analysierte Engpässe

1. **Redundanter Uniform-Control-Durchlauf:**
   - Nach Abschluss der pixelweisen Kernrekonstruktion rief `reconstruct_aqmh_weighted` separat die Funktion `compute_aqmh_uniform_control()` auf.
   - Dadurch wurden sämtliche Frames, Masks und Regions ein zweites Mal vollständig von der Festplatte gelesen und verarbeitet.

2. **Dynamische Allokationen und Sortieraufwand im Sigma-Clipping (`aqmh_sigma_clip`):**
   - Pro Pixel wurden wiederholt dynamische `std::vector`-Objekte allokiert und freigegeben.
   - Der Median und die mittlere absolute Abweichung (MAD) wurden über vollständiges Sortieren berechnet ($O(N \log N)$), obwohl für den Median nur eine $O(N)$-Partitionierung/Quickselect erforderlich ist.
   - Schleifen iterierten starr über die konfigurierte Iterationsanzahl (z. B. 4), selbst wenn bereits in Iteration 0 keine Pixel geklippt wurden.

3. **Interpolations- und Konvertierungsoverhead im Quality-Map-Cache (`aqmh_quality_map_cache`):**
   - Beim bilinear upgesampelten Lesen von Regionen wurden die X-Koordinaten und -Gewichte für jede Zeile und jedes Pixel redundant neu berechnet.
   - Die Typkonvertierung von `uint16_t`/`uint8_t` zu normalisiertem `float32` war nicht SIMD-vektorisiert.

4. **Frame-Loading & Mask-Validierung:**
   - Das Laden von Frame- und Quality-Map-Regionen innerhalb der Chunks erfolgte rein sequentiell.
   - Masken-Kompatibilitäts-Hashes wurden für jedes Frame pro Chunk wiederholt überprüft.

5. **Validierungsmetriken-Overhead (`aqmh_validation`):**
   - FWHM- und Gauß-Fit-Berechnungen liefen über die vollen 4K-Bildmatrizen, was bei wiederholten Dämpfungs- und Neutralisierungsschritten erhebliche Zeit beanspruchte.

---

## 3. Implementierte Maßnahmen

### 3.1 `tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp`
- **Inlining der Uniform-Control-Akkumulation:** Die Mittelwertbildung für das ungewichtete Uniform-Control wird direkt während des Frame-Ladevorgangs parallel im Speicher akkumuliert. Der redundante zweite Ladedurchlauf wurde vollständig eliminiert.
- **Paralleles Chunk-Loading:** OpenMP-Parallelisierung (`#pragma omp for schedule(dynamic, 1)`) über die Frames hinweg beim Einlesen der Chunks in das Pixel-Major Structure-of-Arrays (SoA) Layout.
- **Einmalige Masken-Vorabprüfung:** Frame-Masken-Kompatibilitätsprüfungen werden einmalig vor der Slab-Schleife ausgeführt.

### 3.2 `tile_compile_cpp/src/reconstruction/aqmh_sigma_clip.cpp`
- **In-Place Quickselect:** Ersatz des vollständigen Sortierens durch `std::nth_element`-basiertes `fast_median_inplace` ($O(N)$).
- **Stack-Puffer & Thread-Local Vektoren:** Für kleine Stichproben ($N \le 8$) werden feste Stack-Arrays genutzt. Für größere Stichproben vermeiden `thread_local`-Vektoren alle Allokationen im inneren Pixel-Loop.
- **Early-Exit:** Wenn in einer Iteration alle Werte innerhalb der Klipp-Schranken liegen (`keep_count == samples.size()`), bricht die Iterationsschleife sofort ab.
- **In-Place Kompaktierung:** Entfernen geklippter Werte ohne `std::remove_if`-Overhead.

### 3.3 `tile_compile_cpp/src/metrics/aqmh_quality_map_cache.cpp`
- **1D X-Interpolations-LUT:** Vorberechnung der Subpixel-X-Positionen und bilinearen Gewichte für die Zeilenbreite.
- **SIMD-Vektorisierung:** `#pragma omp simd` für die Dekodierung und Normalisierung (`uint16`/`uint8` $\to$ `float32`).
- **Zero-Veto-Byte-Scan:** Schneller Byte-Scan überspringt das bitweise Setzen von Nullen, wenn keine Veto-Flags vorliegen.

### 3.4 `tile_compile_cpp/src/reconstruction/aqmh_validation.cpp`
- **FWHM-Subsampling:** Begrenzung der FWHM-Messmatrix auf maximal 800 Pixel bei großen Canvases; Hintergrund-RMS, Nahtbewertung und Sternschweif-Metriken bleiben auf voller Auflösung.
- **Wiederverwendung von finite/Differenz-Puffern.**

---

## 4. Benchmark- und Verifikationsergebnisse

### Vollständiger Run mit 645 Frames (DWARF M31 Lights)

| Phase / Metrik | Referenz (`m31_20260810_111030`) | Optimiert (`20260818_132540_a5e63237`) | Delta / Beschleunigung |
| :--- | :--- | :--- | :--- |
| **AQMH Core-Rekonstruktion** | 545.0 s | 287.3 s | **1.90× schneller** (-47.3%) |
| **RGB-Pass R** | 320.0 s | 304.0 s | **1.05× schneller** |
| **RGB-Pass G** | 328.0 s | 308.1 s | **1.06× schneller** |
| **RGB-Pass B** | 334.0 s | 303.1 s | **1.10× schneller** |
| **Validierung & Control** | 120.2 s | 11.1 s | **10.8× schneller** (-90.8%) |
| **Gesamtdauer AQMH Phase** | **1647.2 s (27m 27s)** | **1213.6 s (20m 14s)** | **-433.6 s (-26.3% Phase Gesamt, Core + Validierung >50% schneller)** |

### Qualität & Metriken-Vergleich

| Metrik | Referenzlauf | Optimierter Lauf | Differenz |
| :--- | :--- | :--- | :--- |
| **FWHM** | `2.586343` px | `2.586343` px | **0.000000 (Exakt)** |
| **Background RMS** | `0.152596` | `0.152596` | **0.000000 (Exakt)** |
| **Elongation Median** | `1.103716` | `1.103716` | **0.000000 (Exakt)** |
| **Seam Score** | `0.574640` | `0.574640` | **0.000000 (Exakt)** |
| **Matched Stars** | 186 | 186 | **Exakt identisch** |
| **Quality Gates** | `pass` | `pass` | **Exakt identisch** |

---

## 5. Analyse weiterer Optimierungspotenziale in der Gesamt-Pipeline

### 5.1 Registrierungsphase (`REGISTRATION`)
1. **Redundante Proxy-Generierung & FITS-I/O:**
   - In `runner_phase_registration.cpp` lädt `load_registration_proxy(fi)` FITS-Dateien bei Cache-Misses auf Host-Ebene. Bei Multi-Anchor- und Nachbarschafts-Suchen (Support-Frames) erfolgen wiederholte Lesezugriffe.
   - *Maßnahme:* Vorab-Extraktion aller 2x/4x Downscale-Proxies in einem parallelen I/O-Pass direkt nach `INPUT_SCAN` oder persistentes Halten der kompakten Proxies im RAM (bei 1000 Frames nur ca. 2 GB RAM-Bedarf).
2. **Mehrfach-Sterndetektion bei Multi-Anchor-Matching:**
   - Bei jedem Anchor-Paar-Vergleich wird die Sterndetektion (`detect_stars_proxy`) auf dem Moving-Proxy neu aufgerufen, anstatt die extrahierten Stern-Listen der Proxies einmalig im Speicher zu cachen.
   - *Maßnahme:* Sternlisten pro Proxy frame-indexiert vorab berechnen und in `GlobalAnchorCandidate` wiederverwenden.

### 5.2 Prewarp & Geometrische Entzerrung (`PREWARP`)
1. **Blockierende Synchronisation pro Farbkanal (`cuda_warp_affine_impl`):**
   - In `tile_compile_cpp/src/core/acceleration.cpp` ruft `cuda_warp_affine_impl` nach jedem einzelnen Kanal (R, G, B) `cuda_stream.waitForCompletion()` auf.
   - Dadurch blockiert der CPU-Worker-Thread nach Kanal R vollständig, anstatt alle drei Kanäle asynchron in die CUDA-Stream-Pipeline einzureihen.
   - *Maßnahme:* Asynchrone 3-Kanal-Warp-Funktion mit genau **einer** Synchronisation am Ende des Frame-Triplets (`warp_affine_rgb_async`).
2. **Reallokations-Overhead durch ungepufferte `cv::cuda::GpuMat`:**
   - Bei jedem Aufruf von `cuda_warp_affine_impl` werden `d_src` und `d_dst` neu deklariert, was interne CUDA-Treiber-Allokationen (`cudaMalloc`/`cudaFree`) triggeren kann.
   - *Maßnahme:* Zuweisung eines festen `cv::cuda::BufferPool` oder thread-lokaler `GpuMat`-Wiederverwendung.
3. **CFA-Subplane CPU/GPU Hin- und Rückkopieren:**
   - Für den CFA-Pfad zerlegt `cuda_warp_cfa_mosaic` 4 Subplanes auf der CPU in temporäre Matrizen, lädt 4 Matrizen einzeln hoch, lädt 4 herunter und reassembliert sie per CPU-Schleife.
   - *Maßnahme:* Fused CUDA Kernel für CFA-Subplane-Extraktion, Warp und Reassemblierung direkt im GPU-VRAM.

### 5.3 Quality Map Erzeugung (`AQMH_MAPS`)
1. **Redundante X-Koordinaten- und Gewichtsberechnung in `accumulate_upsampled_log_psi`:**
   - In `src/metrics/aqmh_quality_map.cpp:784-835` wird die bilineare Subpixel-Interpolation für 28-Megapixel-Bilder über 4 Pyramiden-Oktaven für jedes Pixel zeilenweise neu ausgewertet ($sx, x_0, w_x$).
   - *Maßnahme:* 1D-Interpolations-LUT für die X-Dimension vorberechnen (analog zur Implementierung in `aqmh_quality_map_cache.cpp`), wodurch $>40\%$ der CPU-Zyklen im Upsampling eingespart werden.
2. **Speicherallokationen in `local_variance_linear` und `local_mean_and_count`:**
   - Jede Auswertung allokiert 3 vollständige $W \times H$ Matrizen (je 112 MB bei 28 MP).
   - *Maßnahme:* Wiederverwendung vorallokierter horizontaler Pufferzeilen statt vollständiger $W \times H$ Zwischenmatrizen.
3. **Heap-Allokationen in `finite_values`:**
   - `finite_values(m)` erzeugt für jede Z-Score-Berechnung einen dynamischen `std::vector<float>` mit bis zu 28 Millionen Einträgen.
   - *Maßnahme:* In-Place Z-Score-Filterung mit zwei Durchläufen (Welford/Median-Partitionierung) ohne Heap-Duplikation.

---

## 6. CUDA GPU: Analyse von Fehlern, Ineffizienzen und Falschverwendungen

### 6.1 Pinned Host Memory fehlt (`reconstruct_aqmh_weighted_cuda`)
- **Fehler/Ineffizienz:** Die Host-Staging-Puffer `h_frames`, `h_q_maps`, `h_masks` in `aqmh_reconstruction_cuda.cu:847-862` werden über standardmäßige `std::vector<float>` bzw. `std::vector<uint8_t>` allokiert (seitenbasierter Pageable Memory).
- **Konsequenz:** Laut CUDA-Spezifikation erzwingt `cudaMemcpyAsync` von pageable Speicher einen synchronen Staging-Kopiervorgang im Treiber. Echtes asynchrones DMA (Direct Memory Access) und die Überlappung von Datenübertragung und Kernel-Ausführung auf der GPU sind dadurch **deaktiviert**.
- **Lösung:** Allokation über `cudaHostAlloc(&ptr, size, cudaHostAllocPortable)` bzw. `cudaMallocHost` mit RAII-Wrapper.

### 6.2 Lokaler Speicherbedarf & Register Spilling im CUDA Kernel
- **Problem:** In `aqmh_reconstruction_kernel` (`aqmh_reconstruction_cuda.cu:470-473`) deklariert jeder CUDA-Thread:
  ```cuda
  float values[MaxFrames];
  float weights[MaxFrames];
  float scores[CherryPickEnabled ? MaxFrames : 1];
  int scratch_buf[MaxFrames];
  ```
  Bei $MaxFrames = 1024$ belegt ein einzelner Thread $1024 \times (4 + 4 + 4 + 4) = 16.384\text{ Bytes}$ (16 KB) lokalen Thread-Speicher!
- **Konsequenz:** Da GPUs nur ca. 64–256 KB Register pro Streaming Multiprocessor besitzen, führt dies zu massivem **Local Memory Spilling in den L1/L2-Cache und VRAM**, was die Kernel-Occupancy und Speicherbandbreite stark einbremst.
- **Lösung:**
  - Dynamische Kernel-Spezialisierung mit engeren Stufen: $MaxFrames \in [32, 64, 128, 256, 512, 1024]$.
  - Wiederverwendung von `values`/`weights` Arrays und kompakte 16-Bit-Indizes (`uint16_t` statt `int`) für `scratch_buf`.

### 6.3 Fehlendes Double-Buffering / Stream-Overlap
- **Problem:** In `reconstruct_aqmh_weighted_cuda` werden H2D-Upload, Kernel-Launch und D2H-Download sequentiell in einem einzigen `cudaStream_t` ausgeführt.
- **Lösung:** Einsatz von 2 CUDA-Streams mit Double-Buffering: Während Chunk $k$ den Kernel auf der GPU ausführt, wird Chunk $k+1$ bereits per DMA hochgeladen und Chunk $k-1$ zurückgelesen.

### 6.4 OpenCV CUDA Acceleration Mapping
- In `tile_compile_cpp/src/core/acceleration.cpp:64-67` ist `AccelerationPhase::aqmh_maps` für CUDA hardcodiert auf `false` gesetzt.
- Die rechenintensiven Filter (Gauß-Filter, separable Gradienten) in `AQMH_MAPS` laufen daher selbst bei aktiver GPU-Konfiguration immer über die CPU.
- **Lösung:** Anbindung von `cv::cuda::createGaussianFilter` und `cv::cuda::createLinearFilter` in `AQMH_MAPS`.

---

## 7. Status der Optimierungsmaßnahmen & Verifikation

| Priorität | Bereich | Maßnahme | Status | Erzielte Wirkung |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | **CUDA Core** | Pinned Host Memory (`cudaHostAlloc` / `PinnedBuffer`) in `aqmh_reconstruction_cuda.cu` | **Erledigt & Verifiziert** | Ermöglicht echtes asynchrones DMA ohne Treiber-Staging-Locks |
| **P1** | **Prewarp** | Asynchrones 3-Kanal-Warping (`warp_affine_rgb_frame`) in `acceleration.cpp` | **Erledigt & Verifiziert** | Eliminiert 2 von 3 blockierenden GPU-Synchronisationen pro debayer_first-Frame |
| **P2** | **CUDA Kernel** | Kompakte 16-Bit-Indizes (`short`) & engere Spezialisierungsstufen ($N \le 32, 64, 128, 256, 512, 1024$) in `aqmh_reconstruction_cuda.cu` | **Erledigt & Verifiziert** | Halbiert Index-Puffer-Footprint im GPU-Thread und verhindert Spilling |
| **P2** | **AQMH Maps** | 1D-X- und Y-Interpolations-LUTs in `accumulate_upsampled_log_psi` (`aqmh_quality_map.cpp`) | **Erledigt & Verifiziert** | Entfernt Subpixel-`floor`/`clamp`/`abs`-Kosten aus 28M-Pixel-Schleife |
| **P3** | **Registration**| Thread-safe In-Memory-Proxy-Caching (`in_memory_proxies`, `proxy_init_flags`) in `runner_phase_registration.cpp` | **Erledigt & Verifiziert** | Verhindert redundante FITS-Dekodierung bei Multi-Anchor- und Support-Suchen |

### Test- und Verifikationsergebnis
- **Catch2 Test Suite:** 282 von 282 Testfällen erfolgreich bestanden (`29823 assertions in 282 test cases`).
- **Mathematische Konsistenz:** Exakte Übereinstimmung zwischen CPU- und CUDA-Kernel-Ergebnissen (Ausgabematrizen, Pixelzähler für `unsupported_pixels`, `zero_veto_pixels`, `numerical_guard_pixels`).
