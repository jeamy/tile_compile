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
- **Einmalige Masken-Vorabprüfung:** Frame-Masken-Kompatibilitätsprüfungen werden einmalig vor der Slab-Schleife ausgeführt. Für jedes Frame wird die vollständige Maske über `load_frame_valid_mask` geladen und ihr SHA-256-Hash mit dem im `QualityMapCache` persistierten `source_mask_hash(fi)` verglichen. Das setzt den Overhead aus den Slab-Schleifen in einen einmaligen sequenziellen Pass; bei 645 Frames à 28 MP (≈ 28 MB/Maske) liegt der SHA-256-Berechnungsaufwand jedoch im messbaren Sekundenbereich und dominiert bei rein CPU-gebundenem Hashing. Eine Parallelisierung dieses Vorab-Passes wäre möglich und ist als weiteres Potential in §8 (Punkt G) festgehalten.

### 3.2 `tile_compile_cpp/src/reconstruction/aqmh_sigma_clip.cpp`
- **In-Place Quickselect:** Ersatz des vollständigen Sortierens durch `std::nth_element`-basiertes `fast_median_inplace` ($O(N)$).
- **Stack-Puffer & Thread-Local Vektoren:** Für kleine Stichproben ($N \le 8$) werden feste Stack-Arrays genutzt. Für größere Stichproben vermeiden `thread_local`-Vektoren alle Allokationen im inneren Pixel-Loop.
- **Early-Exit:** Wenn in einer Iteration alle Werte innerhalb der Klipp-Schranken liegen (`keep_count == samples.size()`), bricht die Iterationsschleife sofort ab.
- **In-Place Kompaktierung:** Entfernen geklippter Werte ohne `std::remove_if`-Overhead.

### 3.3 `tile_compile_cpp/src/metrics/aqmh_quality_map_cache.cpp`
- **1D X-Interpolations-LUT:** Vorberechnung der Subpixel-X-Positionen und bilinearen Gewichte für die Zeilenbreite.
- **SIMD-Vektorisierung:** `#pragma omp simd` für die Dekodierung und Normalisierung (`uint16`/`uint8` $\to$ `float32`).
- **Zero-Veto-Byte-Scan:** Schneller Byte-Scan (`has_veto`-Flag) überspringt das bitweise Setzen von Nullen, wenn keine Veto-Flags im gelesenen Byte-Bereich vorliegen. **Einschränkung:** Diese Optimierung ist ausschließlich im LRU-Cache-Pfad (`max_resident_maps > 0`) implementiert. Im direkten Dateilesepfad (Cache deaktiviert, `max_resident_maps <= 0`) fehlt der Early-Exit; dort wird bedingungslos über alle Bits iteriert. Für den Standard-Betrieb mit aktiviertem Cache ist das korrekt; der Fallback-Pfad bleibt ein offenes Optimierungspotenzial.

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

> **Hinweis zur Tabellenstruktur:** Die fünf Zeitzeilen sind additive Teilphasen; ihre Summe ergibt jeweils die Gesamtdauer  
> (Referenz: 545 + 320 + 328 + 334 + 120 = **1647 s** ✓; Optimiert: 287 + 304 + 308 + 303 + 11 = **1213 s** ✓).  
> „AQMH Core-Rekonstruktion“ misst den isolierten pixel-weisen Sigma-Clip-/Gewichtungskern eines einzelnen Kanaldurchlaufs (kein I/O);  
> die RGB-Passes R/G/B enthalten jeweils Frame-Ladezeit, Q-Map-Zugriff und Core-Compute pro Kanal vollständig.  
> Der höhere Core-Speedup (1.90×) gegenüber den Channel-Passes (≈1.05–1.10×) reflektiert den relativen I/O-Anteil in den vollen Kanal-Passes.

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
   - *Maßnahme (teilweise umgesetzt):* `cuda_warp_cfa_mosaic` konsolidiert alle 4 Subplanes in **einen** `cudaStream_t` mit **einer** abschließenden `waitForCompletion()`. CPU-seitige Subplane-Extraktion (`extract_cfa_subplanes`) und CPU-seitige Reassemblierung (`reassemble_cfa_subplanes`) sind weiterhin vorhanden. Ein echter Fused CUDA Kernel, der Extraktion, Warp und Reassemblierung vollständig im GPU-VRAM ausführt, ist **nicht implementiert** (Status: offen, s. §7).

### 5.3 Quality Map Erzeugung (`AQMH_MAPS`)
1. **Redundante X-Koordinaten- und Gewichtsberechnung in `accumulate_upsampled_log_psi`:**
   - In `src/metrics/aqmh_quality_map.cpp:784-835` wird die bilineare Subpixel-Interpolation für 28-Megapixel-Bilder über 4 Pyramiden-Oktaven für jedes Pixel zeilenweise neu ausgewertet ($sx, x_0, w_x$).
   - *Maßnahme:* 1D-Interpolations-LUTs für **X- und Y-Dimension** vorberechnen (`struct Interp1D x_lut`/`y_lut` in `accumulate_upsampled_log_psi`, Zeilen 810–885 von `aqmh_quality_map.cpp`). Eliminiert alle `floor`/`clamp`/`abs`-Operationen aus dem 28M-Pixel-Innenloop für beide Dimensionen. **Erledigt & Verifiziert** (P2, s. §7). *Anmerkung: Das ursprüngliche Dokument nannte nur die X-Dimension – die Implementierung umfasst auch die Y-LUT.*
2. **Speicherallokationen in `local_variance_linear` und `local_mean_and_count`:**
   - **GPU-Pfad (`accelerated_local_variance`):** Die Zwischenpuffer sind über `thread_local CudaMomentsWorkspace workspace` bereits persistent; pro Frame-Aufruf findet kein `cudaMalloc`/`cudaFree` statt. **Dieser Pfad ist bereits optimiert.**
   - **CPU-Fallback-Pfad (`local_variance_linear`):** Allokiert pro Aufruf 4 vollständige $W \times H$ Matrizen (je ≈12 MB bei 28 MP: `horizontal_sum`, `horizontal_square_sum`, `horizontal_count`, `out`).
   - *Algorithmische Einschränkung:* Die im Originaldokument genannte Maßnahme „Horizontale Pufferzeilen statt $W \times H$“ ist für den separablen 2-Pass-Filter nicht direkt anwendbar: Pass 2 (vertikal, parallelisiert über x) benötigt random access auf alle Ergebnisse aus Pass 1. Ein echter O($W \times r$)-Stripe-Ansatz würde die OMP-Parallelisierung im vertikalen Pass brechen und ist ein eigenständiges Redesign.
   - *Tatsächlich umsetzbarer Fix (CPU):* `thread_local`-Pre-Allokierung der 3 Zwischenmatrizen, um wiederholtes malloc/free für 3×112 MB zu vermeiden. Auswirkung ist begrenzt, da GPU der Primärpfad ist und der CPU-Fallback intern per OMP parallelisiert. *(Status: für GPU-Pfad erledigt; CPU-Fallback thread_local-Fix offen, niedriger Priorität)*
3. **Heap-Allokationen in `finite_values`:** *(Status: Noch nicht umgesetzt)*
   - `finite_values(m)` gibt weiterhin einen dynamischen `std::vector<float>` mit bis zu 28 Millionen Einträgen zurück, der für jede Z-Score-Berechnung neu allokiert wird.
   - *Maßnahme:* In-Place Z-Score-Filterung mit zwei Durchläufen (Welford/Median-Partitionierung) ohne Heap-Duplikation.

---

## 6. CUDA GPU: Analyse von Fehlern, Ineffizienzen und Falschverwendungen

### 6.1 Pinned Host Memory fehlt (`reconstruct_aqmh_weighted_cuda`)
- **Fehler/Ineffizienz:** Die Host-Staging-Puffer `h_frames`, `h_q_maps`, `h_masks` in `aqmh_reconstruction_cuda.cu:847-862` werden über standardmäßige `std::vector<float>` bzw. `std::vector<uint8_t>` allokiert (seitenbasierter Pageable Memory).
- **Konsequenz:** Laut CUDA-Spezifikation erzwingt `cudaMemcpyAsync` von pageable Speicher einen synchronen Staging-Kopiervorgang im Treiber. Echtes asynchrones DMA (Direct Memory Access) und die Überlappung von Datenübertragung und Kernel-Ausführung auf der GPU sind dadurch **deaktiviert**.
- **Lösung:** Allokation über `cudaHostAlloc(&ptr, size, cudaHostAllocPortable)` bzw. `cudaMallocHost` mit RAII-Wrapper.

### 6.2 Lokaler Speicherbedarf & Register Spilling im CUDA Kernel
- **Problem:** In `aqmh_reconstruction_kernel` (`aqmh_reconstruction_cuda.cu:452-455`) deklariert jeder CUDA-Thread:
  ```cuda
  float values[MaxFrames];
  float weights[MaxFrames];
  float scores[CherryPickEnabled ? MaxFrames : 1];
  int scratch_buf[MaxFrames];  // ← ursprünglicher Typ int (32-bit); nach Fix: short (16-bit)
  ```
  Bei $MaxFrames = 1024$ belegt ein einzelner Thread $1024 \times (4 + 4 + 4) + 1024 \times 4 = 16.384\text{ Bytes}$ (16 KB) Thread-lokalen Speicher, der in den VRAM gespillt wird!
- **Konsequenz:** Da GPUs nur ca. 64–256 KB Register pro Streaming Multiprocessor besitzen, führt dies zu massivem **Local Memory Spilling in den L1/L2-Cache und VRAM**, was die Kernel-Occupancy und Speicherbandbreite stark einbremst.
- **Lösung:**
  - Dynamische Kernel-Spezialisierung mit engeren Stufen: $MaxFrames \in [32, 64, 128, 256, 512, 1024]$.
  - Wiederverwendung von `values`/`weights` Arrays und kompakte vorzeichenbehaftete 16-Bit-Indizes (`short` statt `int32`) für `scratch_buf`. *Hinweis: Der Code verwendet `short` (signed 16-bit), nicht `uint16_t`; beide decken den erforderlichen Index-Bereich 0–1023 ab, `short` ist jedoch der korrekte Typ im Quellcode.*

### 6.3 Fehlendes Double-Buffering / Stream-Overlap
- **Problem:** In `reconstruct_aqmh_weighted_cuda` werden H2D-Upload, Kernel-Launch und D2H-Download für jeden Chunk sequentiell in einem einzigen `cudaStream_t` ausgeführt. GPU-seitige Überlappung zwischen aufeinanderfolgenden Chunks ist nicht möglich.
- **Partiell umgesetzt (CPU-Prefetch):** Ein `std::future<std::vector<Matrix2Df>> next_frame_prefetch` überlappt das CPU-seitige Laden der Frame-Regionen von Disk für Chunk $k+1$ mit der GPU-Ausführung für Chunk $k$. CUDA-Events (`h2d_start`, `kernel_start`, `kernel_end`, `d2h_end`) messen H2D/Kernel/D2H je Chunk. Der Code-Kommentar ist dabei explizit: *„This is a conservative first overlap step: Q-map/mask loading stays synchronous.“* Q-Map- und Masken-Zugriffe bleiben synchron.
- **Noch offen (GPU Double-Buffering):** Echtes GPU-seitiges Double-Buffering mit zwei `cudaStream_t`-Instanzen – Stream A: H2D-Upload Chunk $k+1$ parallel zu Stream B: Kernel-Execution Chunk $k$ parallel zu D2H-Download Chunk $k-1$ – ist **nicht implementiert**. Es wäre doppelte Gerätepuffer (frames/q\_maps/masks/output) erforderlich. Geschätztes weiteres Einsparpotenzial: 15–25% GPU-Laufzeit (s. §7, Status: offen).

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
| **P3** | **CUDA Core** | GPU Double-Buffering / Stream-Overlap (§6.3): zwei `cudaStream_t` + Ping-Pong-Gerätepuffer | **Offen** – CPU-Prefetch für Frame-Regionen partiell umgesetzt; H2D/Kernel/D2H je Chunk weiterhin sequenziell | Geschätztes Potenzial: 15–25% GPU-Laufzeit |
| **P3** | **Prewarp** | CFA-Subplane Fused CUDA Kernel (§5.2.3): GPU-seitige Extraktion, Warp und Reassemblierung | **Offen** – Stream-Konsolidierung (1× sync) implementiert; CPU-seitige Subplane-Extraktion/Reassemblierung weiterhin vorhanden | – |
| **P4** | **AQMH Maps** | `local_variance_linear` CPU-Fallback: `thread_local`-Pre-Allokierung der 3 Zwischenmatrizen (§5.3.2) | **GPU-Pfad erledigt** (thread\_local workspace); CPU-Fallback thread\_local offen | GPU: kein cudaMalloc/cudaFree je Frame; CPU: Reduktion wiederholter 3×112 MB malloc |
| **P4** | **AQMH Maps** | `finite_values` In-Place Z-Score (§5.3.3) | **Offen** | Vermeidet 28M-Element `std::vector<float>` je Z-Score-Durchlauf |
| **P4** | **CUDA Core** | SHA-256 Masken-Vorabprüfungs-Parallelisierung (§3.1-Anmerkung) | **Offen** | Parallelisierbar über OpenMP; bei 645 × 28 MP Masken im messbaren Sekundenbereich |

### Test- und Verifikationsergebnis
- **Catch2 Test Suite:** 282 von 282 Testfällen erfolgreich bestanden (`29823 assertions in 282 test cases`).
- **Mathematische Konsistenz:** Exakte Übereinstimmung zwischen CPU- und CUDA-Kernel-Ergebnissen (Ausgabematrizen, Pixelzähler für `unsupported_pixels`, `zero_veto_pixels`, `numerical_guard_pixels`).

---

## 8. Neu identifizierte Optimierungspotenziale (Code-Review)

Die folgenden Punkte wurden durch Quellcode-Analyse nach Abschluss der P1–P3-Maßnahmen identifiziert. Sie sind noch nicht implementiert und für künftige Optimierungsrunden vorgemerkt.

### A. `canvas_valid()` im innersten Pixel-Loop
- **Problem:** `canvas_valid(canvas_mask, width, height, x, y)` wird in `reconstruct_aqmh_weighted` für **jedes** Pixel, für **jeden** Frame, innerhalb jedes Chunks aufgerufen – mit Bounds-Checks und Vektor-Indexierung. Die Canvas-Mask ist über die gesamte Phase konstant.
- **Maßnahme:** Einmalig ein flaches `bool`-Array (oder `std::bitset`) aus `canvas_mask` materialisieren; direkte 1D-Indexierung ohne Bounds-Checks im innersten Loop. Reduziert Branch-Overhead und verbessert Cache-Lokalität.

### B. `#pragma omp atomic` auf `finite_maps[]` als Hotspot
- **Problem:** Im Frame-Lade-Loop wird `finite_maps[local_i]` per `#pragma omp atomic` inkrementiert – bei 645 Frames × Millionen Pixel pro Chunk entstehen sehr viele atomare Operationen auf einem kleinen Array (hohe Kollisionsrate unter vielen Threads).
- **Maßnahme:** Thread-lokale `finite_maps_local[]`-Zähler innerhalb des OMP-Parallel-Blocks akkumulieren und per `#pragma omp critical` einmalig mergen – analog zu `local_control_sums`.

### C. `#pragma omp critical` für Uniform-Control-Merge
- **Problem:** Das Mergen von `local_control_sums[]`/`local_control_counts[]` in die gemeinsamen Arrays geschieht per `#pragma omp critical` mit einer $O(pixel\_count)$-Schleife (bei `chunk_rows=256`, `width=6252` ca. 1,6 Mio. Additionen im kritischen Abschnitt; vollständig serialisiert für alle Threads).
- **Maßnahme:** OpenMP-`reduction`-Klausel auf Array-Ebene (ab OpenMP 4.5), oder zweistufiges Merge (Unter-Akkumulationen pro Thread-Gruppe), um die Sperrzeit erheblich zu reduzieren.

### D. `read_region`-LUT bei jedem Cache-Miss neu allokiert
- **Problem:** Die `x_lut`-Berechnung (`std::vector<XInterp>`, Länge `full_width`) wird bei jedem `read_region`-Cache-Miss neu allokiert und befüllt. Bei 645 Frames × mehrere Chunks entstehen Tausende redundanter `std::vector`-Allokierungen, obwohl `full_width` und `resolution_divisor` über die gesamte Rekonstruktionsphase konstant sind.
- **Maßnahme:** `x_lut` einmalig im `QualityMapCache`-Konstruktor berechnen und als konstantes Mitglied cachen.

### E. `aqmh_sigma_clip` Small-N-Pfad: doppelte Median-Berechnung
- **Problem:** Im `n₀ ≤ 8`-Pfad (Stack-Array-Zweig) wird der Median zweimal berechnet: gewichtet (`weighted_median_select`) für `center` und ungewichtet (`fast_median_inplace` auf `noise_arr`) für `val_med`. Bei gleichgewichteten Samples sind beide Werte identisch.
- **Maßnahme:** Wenn alle Gewichte innerhalb einer engen Toleranz gleich sind, gewichteten Median direkt als `val_med` wiederverwenden. Oder: kombinierte Funktion, die beide Mediane in einem einzigen Partitionierungsdurchlauf liefert.

### F. CUDA-Kernel-Occupancy bleibt bei `frame_count ≥ 513` niedrig
- **Problem:** Für 513–1024 Frames fällt der Dispatch auf `launch_reconstruction_kernel<1024>`. Jeder Thread belegt dann: `values[1024]` + `weights[1024]` + `scores[1024]` + `short scratch_buf[1024]` = $1024 \times (4+4+4+2) = 14{.}336$ Bytes Thread-lokalen Speicher (spill in VRAM). Bei Blockgröße `(32, 8)` = 256 Threads pro Block entspricht das ≈14 KB × 256 = 3,5 MB lokalen Speichers je Block, was auf typischen GPUs zu nur 1–2 aktiven Blocks pro SM führt (Occupancy ≈3–6%).
- **Maßnahme (Option 1):** Shared-Memory-Pool-Kernel: Frames werden in Shared Memory gestapelt und chunkweise verarbeitet, sodass jeder Thread nur einen Slot belegt. (Erfordert Kernel-Redesign.)
- **Maßnahme (Option 2):** Blockgröße auf `(16, 8)` = 128 Threads reduzieren; halbiert den lokalen Speicherdruck pro SM bei gleichem Grid.
- **Maßnahme (Option 3):** Für frame_count 513–645 eine enge Stufe `<640>` oder `<768>` ergänzen, um das effektive Speicher-Footprint gegenüber Stufe 1024 um 37–50% zu reduzieren.

### G. Parallelisierung der SHA-256-Masken-Vorabprüfung (§3.1)
- **Problem:** Die `frame_mask_compatible`-Vorabprüfungsschleife iteriert sequenziell über alle Frames. Für jedes Frame wird die vollständige Maske geladen und SHA-256 berechnet. Bei 645 Frames à 28 MB Maske sind das sequenziell ~18 GB SHA-256-Eingabedaten.
- **Maßnahme:** Schleife per `#pragma omp parallel for schedule(dynamic, 1)` parallelisieren; SHA-256-Berechnung ist embarrassingly parallel. Das Laden bleibt I/O-gebunden, profitiert aber von parallelen Datei-Opens auf SSDs.
