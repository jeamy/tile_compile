# 01 - Architektur und Quelltrennung

## 1. Grundsatz

Apple-spezifischer Code lebt ausschließlich unter `tile_compile_cpp/metal/`.
Kein `.mm`, kein `.metal`, kein `#import <Metal/...>` außerhalb dieses
Teilbaums. Der gemeinsame Code (`src/`, `include/`, `apps/`) bleibt frei von
Objective-C++ und Metal-Typen und kennt Metal nur über die Registry aus 3.

## 2. Verzeichnislayout

```
tile_compile_cpp/
  metal/
    CMakeLists.txt              # target tile_compile_metal (STATIC), nur APPLE
    include/tile_compile/metal/
      metal_backend.hpp         # reines C++: Probe, Factory, Bildoperationen
      metal_types.hpp           # POD-Strukturen der Host<->Shader-Grenze
    src/
      metal_device.mm           # MTLDevice, Queue, Library, Budget, Familien-Gates
      metal_buffers.mm          # Buffer-Wrapper (Shared, NoCopy), Workspace
      fd_v2_kernel.mm           # ForwardDrizzleV2MetalKernel (Host-Orchestrierung)
      fd_v2_local_warp.mm       # Local-Warp-Pfad (optional getrennt)
      fd_v2_estimator.mm        # Pilot + Full-Frame-Estimator
      warp_affine.mm            # Prewarp-Kernel-Host
      box_filter.mm             # Source-Quality-Box-Filter-Host
      metal_register.cpp        # tile_compile_metal_register()
    shaders/
      common.h                  # MSL: Double-Float, Makros, Strukturen (Host-geteilt)
      df_math.h                 # MSL: TwoSum/TwoProd/DF-Ops
      polygon.metal             # Sutherland-Hodgman, Fläche
      scatter_v2.metal          # affine/cached/samples/local Scatter
      fold_v2.metal             # Fold + Akkumulation + Reservoir
      finalize_v2.metal         # Finalize, SFR build/vote/reduce
      estimator_v2.metal        # k_full_seed, k_full_apply
      warp_affine.metal
      box_filter.metal
    emulation/                  # PORTABEL (auch Linux): DF-Emulation + Harness
      df_math.hpp               # identische Algebra wie shaders/df_math.h
      precision_harness.cpp     # siehe 02
    tests/
      CMakeLists.txt
      test_metal_*.cpp          # Catch2, SKIP ohne Device
```

`shaders/df_math.h` und `emulation/df_math.hpp` müssen dieselbe Algebra
beschreiben (gleiche Operationsfolge). Ein Test vergleicht beide auf
Zufalls- und Grenzwerteingaben; so kann die Präzision auf Linux mit dem
Shader-Algorithmus untersucht werden.

## 3. Schnittstellen zum gemeinsamen Code

Nur diese Berührungspunkte im gemeinsamen Baum sind erlaubt:

### 3.1 Backend-neutrale Typen (mechanische Verschiebung)

`forward_drizzle_cuda.hpp` enthält heute backend-neutrale POD-Typen, die auch
CPU-Code und Treiber einbinden: `ForwardDrizzleV2LocalWarp`,
`ForwardDrizzleV2FrameQuality`, `ForwardDrizzleV2PackedQualityPlane`,
`ForwardDrizzleV2CachedLeaf`, `ForwardDrizzleV2SourceSample`,
`ForwardDrizzleV2AlignedQuality`, `ForwardDrizzleV2PixelResult`,
`ForwardDrizzleV2PrototypeStats`, `ForwardDrizzleV2KernelConfig` u. a.

- Neuer Header `forward_drizzle_v2_types.hpp` nimmt sie auf;
  `forward_drizzle_cuda.hpp` inkludiert ihn (bestehende Includes bleiben
  gültig, keine Namensänderung).
- Verifikation: reiner Verschiebe-Diff, gesamte Suite unverändert grün, kein
  Symbol umbenannt.

### 3.2 Backend-Registry (neu, plattformneutral)

`include/tile_compile/core/gpu_backend.hpp`:

```cpp
enum class GpuBackend { none, cuda, metal };

struct GpuKernelFactory {
  bool (*runtime_available)();                        // Gerät + Fähigkeiten ok
  std::unique_ptr<reconstruction::ForwardDrizzleV2Kernel> (*create)();
  std::string (*device_name)();
  DeviceMemory (*device_memory)();                    // frei/gesamt/Working-Set
};
void register_gpu_kernel_factory(GpuBackend, GpuKernelFactory);
const GpuKernelFactory *gpu_kernel_factory(GpuBackend);
```

- CUDA registriert sich über seine bestehende Funktionen (dünner Adapter,
  Verhalten unverändert).
- Metal: `tile_compile_metal_register()` wird **explizit** aus dem
  `AccelerationContext`-Konstruktor unter `#if TILE_COMPILE_WITH_METAL`
  aufgerufen (keine statische Initialisierungsreihenfolge).
- Bildoperationen (Prewarp, Box-Filter) analog als Funktionszeiger-Tabelle
  `GpuImageOps` (warp_affine_mono/rgb/cfa, box_filter_sums).

### 3.3 Acceleration-Schicht

- `AccelerationBackend` bekommt `metal`; `AccelerationCapabilities` bekommt
  `tile_compile_with_metal`, `metal_runtime`.
- `select_acceleration_backend`: `auto` wählt je Phase `cuda > metal >
  opencl > cpu`, nur wenn buildbar **und** Gerät vorhanden. Explizit
  angefordertes, nicht verfügbares Backend: `request_honored=false` mit
  `fallback_reason`, Lauf auf CPU (wie heute bei CUDA).
- `AccelerationOps::warp_affine_*`: neue Verzweigung für `metal` über
  `GpuImageOps`; die `cv::cuda::Stream*`-Parameter bleiben für CUDA, für
  Metal wird ein neutraler Handle `GpuStreamHandle` (opaker Zeiger)
  ergänzt. `WorkerCudaStreams` bleibt CUDA-spezifisch; Metal führt pro
  Worker eine `MTLCommandQueue`-Instanz im eigenen Teilbaum.

### 3.4 Treiber und Runner

- `make_kernel(bool cuda)` in `forward_drizzle_v2_driver.cpp` wird zu
  `make_kernel(GpuBackend)`; `ForwardDrizzleV2DriverOptions::prefer_cuda`
  bekommt ein Geschwisterfeld `GpuBackend preferred_gpu` (Default aus
  `prefer_cuda`, damit alte Aufrufer/Tests unverändert bleiben).
- `backend_used`: `"metal_v2"`; `cuda_fallback_reason` wird zusätzlich als
  neutrales `gpu_fallback_reason` geführt (altes Feld bleibt gefüllt, damit
  Reports/Checkpoints/GUI kompatibel bleiben).
- `runner_forward_drizzle.cpp`: `fd_backend` aus der Auswahl ableiten
  (`"cuda"|"metal"|"cpu"`), Warntext zu `full_frame_estimator` auf "kein
  GPU-Backend" verallgemeinern, Telemetrie `gpu` auch für `metal_v2`.
- Fehlerklasse: `ForwardDrizzleGpuError` als Alias von
  `ForwardDrizzleCudaError` (gleicher Typ, damit Fänger beider Namen
  funktionieren).
- Fault-Injection: zusätzlich `TILE_COMPILE_FD_V2_GPU_FAULT_AFTER_BANDS`
  (das CUDA-Env bleibt gültig und wirkt auf das jeweils gewählte Backend).

## 4. Abhängigkeitsregeln

- `metal/` darf gemeinsame Header inkludieren; der gemeinsame Baum darf
  `metal/` nur über `gpu_backend.hpp` und das Kompilat-Define
  `TILE_COMPILE_WITH_METAL` berühren.
- `metal/include` ist reines C++ (kein Objective-C): gemeinsamer Code, der
  es inkludiert, bleibt mit jedem Compiler übersetzbar.
- Shader-Host-Strukturen (`metal_types.hpp` / `shaders/common.h`) haben
  feste Layouts (`static_assert` auf Größe/Offset, `alignas`), da MSL und
  C++ Padding unterschiedlich behandeln können.

## 5. Ressourcenmodell (Unified Memory)

- Kein Upload-/Download-Pipelining: Host-Quellpuffer (Source-Fenster,
  Sigma2, Quality) werden als `MTLStorageModeShared`-Puffer bereitgestellt
  (bei seitenausgerichteten Puffern `newBufferWithBytesNoCopy`, sonst
  Kopie in einen persistenten Ring). Die CUDA-Konstrukte `ev_up0/ev_up1`,
  Pin-Slots und `cudaMemcpyAsync` entfallen; `upload_seconds` wird als
  Bereitstellungszeit (Kopie/Wrap) geführt, damit die Telemetrie
  vergleichbar bleibt.
- Speicherbudget: `MTLDevice.recommendedMaxWorkingSetSize` und
  `currentAllocatedSize` ersetzen `cudaMemGetInfo`. CPU und GPU teilen den
  RAM; der Working-Set-Vertrag (§11.13) muss Metal-Puffer mitzählen.
  `ForwardDrizzleV2MemoryPlan` (Gate 5) erhält die Metal-Budgetzahl als
  Eingabe; `maxBufferLength` ist Obergrenze pro Puffer (zu verifizieren) und
  begrenzt Ebenengrößen je Band.
- Einmalige Reservierung pro Lauf (`reserve`), `begin_band` ohne
  Allokation (`driver_hotpath_allocations == 0` bleibt Gate).

## 6. Ausführungsmodell

- Pro Kernel-Instanz eine `MTLCommandQueue`; pro Frame ein Command-Buffer
  mit den Encodern Scatter -> Fold. Kein globaler Device-Sync im Hotpath
  (`device_global_synchronizations == 0`); Synchronisation nur an
  Bandgrenzen (`waitUntilCompleted` bzw. `MTLSharedEvent`).
- Lange Dispatches (Fold, Finalize, SFR) werden in Teil-Dispatches über
  Zeilen-/Spaltenbereiche zerlegt, damit kein einzelner Command-Buffer
  ein GPU-Zeitlimit überschreitet (Grenze zu verifizieren).
- `stats().kernel_seconds` aus `GPUStartTime/GPUEndTime` der
  Command-Buffer.
