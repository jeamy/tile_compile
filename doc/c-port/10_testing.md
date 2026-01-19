# Phase 8: Testing und Validierung

## Ziel

Umfassende Tests und Validierung der C++ Portierung gegen die Python-Referenzimplementierung.

**Geschätzte Dauer**: 2-3 Wochen

---

## 8.1 Testing-Strategie

### Ebenen

1. **Unit-Tests**: Einzelne Funktionen und Klassen
2. **Integrationstests**: Modul-Interaktionen
3. **Regressionstests**: Vergleich mit Python-Ausgaben
4. **Performance-Tests**: Geschwindigkeitsvergleich

---

## 8.2 Unit-Test-Framework

### Catch2 Setup

```cmake
# CMakeLists.txt
find_package(Catch2 3 REQUIRED)

add_executable(tests
    tests/main.cpp
    tests/core/test_utils.cpp
    tests/core/test_events.cpp
    tests/io/test_fits.cpp
    tests/image/test_processing.cpp
    tests/registration/test_registration.cpp
    tests/metrics/test_metrics.cpp
    tests/clustering/test_clustering.cpp
    tests/reconstruction/test_reconstruction.cpp
    tests/synthetic/test_synthetic.cpp
    tests/pipeline/test_phases.cpp
)

target_link_libraries(tests PRIVATE 
    tile_compile_lib 
    Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(tests)
```

### Test-Struktur

```
tests/
├── main.cpp                    # Catch2 main
├── fixtures/                   # Test-Daten
│   ├── test_frame.fits
│   ├── test_config.yaml
│   └── reference_outputs/
├── core/
│   ├── test_utils.cpp
│   └── test_events.cpp
├── io/
│   └── test_fits.cpp
├── image/
│   └── test_processing.cpp
├── registration/
│   └── test_registration.cpp
├── metrics/
│   └── test_metrics.cpp
├── clustering/
│   └── test_clustering.cpp
├── reconstruction/
│   └── test_reconstruction.cpp
├── synthetic/
│   └── test_synthetic.cpp
└── pipeline/
    └── test_phases.cpp
```

---

## 8.3 Regressionstests gegen Python

### Konzept

Für jede Funktion werden Referenz-Ausgaben mit der Python-Implementierung generiert und die C++ Ausgaben dagegen verglichen.

### Python-Referenz-Generator

```python
#!/usr/bin/env python3
"""Generate reference outputs for C++ regression tests."""

import numpy as np
import json
from pathlib import Path

from runner.image_processing import (
    split_cfa_channels,
    reassemble_cfa_mosaic,
    normalize_frame,
    cosmetic_correction,
)
from runner.opencv_registration import (
    opencv_prepare_ecc_image,
    opencv_alignment_score,
)
from tile_compile_backend.metrics import (
    MetricsCalculator,
    TileMetricsCalculator,
)
from tile_compile_backend.clustering import StateClustering

OUTPUT_DIR = Path("tests/fixtures/reference_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_array(name: str, arr: np.ndarray):
    """Save array as .npy file."""
    np.save(OUTPUT_DIR / f"{name}.npy", arr.astype(np.float32))

def save_json(name: str, data: dict):
    """Save dict as JSON."""
    with open(OUTPUT_DIR / f"{name}.json", "w") as f:
        json.dump(data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

def generate_image_processing_refs():
    """Generate reference outputs for image processing functions."""
    
    # Test-Mosaik (4x4 RGGB)
    mosaic = np.array([
        [100, 120, 100, 120],
        [110, 130, 110, 130],
        [100, 120, 100, 120],
        [110, 130, 110, 130],
    ], dtype=np.float32)
    
    save_array("mosaic_4x4", mosaic)
    
    # split_cfa_channels
    channels = split_cfa_channels(mosaic, "RGGB")
    save_array("split_cfa_R", channels["R"])
    save_array("split_cfa_G", channels["G"])
    save_array("split_cfa_B", channels["B"])
    
    # reassemble_cfa_mosaic
    reassembled = reassemble_cfa_mosaic(
        channels["R"], channels["G"], channels["B"], "RGGB"
    )
    save_array("reassemble_cfa", reassembled)
    
    # normalize_frame
    frame = np.random.rand(64, 64).astype(np.float32) * 1000 + 500
    save_array("normalize_input", frame)
    
    normalized_bg = normalize_frame(frame, 750.0, 1.0, "background")
    save_array("normalize_background", normalized_bg)
    
    normalized_add = normalize_frame(frame, 750.0, 500.0, "additive")
    save_array("normalize_additive", normalized_add)
    
    # cosmetic_correction
    hotpixel_frame = np.ones((32, 32), dtype=np.float32) * 100
    hotpixel_frame[15, 15] = 10000  # Hotpixel
    save_array("cosmetic_input", hotpixel_frame)
    
    corrected = cosmetic_correction(hotpixel_frame, sigma_threshold=5.0)
    save_array("cosmetic_output", corrected)

def generate_registration_refs():
    """Generate reference outputs for registration functions."""
    
    # Test-Bild
    img = np.random.rand(64, 64).astype(np.float32)
    save_array("registration_input", img)
    
    # prepare_ecc_image
    prepared = opencv_prepare_ecc_image(img)
    save_array("ecc_prepared", prepared)
    
    # alignment_score (identische Bilder)
    score = opencv_alignment_score(img, img)
    save_json("alignment_score_identical", {"score": float(score)})

def generate_metrics_refs():
    """Generate reference outputs for metrics functions."""
    
    # Test-Frames
    frames = [np.random.rand(64, 64).astype(np.float32) * 1000 for _ in range(5)]
    for i, f in enumerate(frames):
        save_array(f"metrics_frame_{i}", f)
    
    # Global metrics
    global_metrics = MetricsCalculator.calculate_global_metrics(frames)
    save_json("global_metrics", global_metrics)
    
    # Tile metrics
    calc = TileMetricsCalculator(tile_size=16, overlap=0.25)
    tile_metrics = calc.calculate_tile_metrics(frames[0])
    save_json("tile_metrics", tile_metrics)

def generate_clustering_refs():
    """Generate reference outputs for clustering."""
    
    # Einfache 2D-Daten mit 3 Clustern
    np.random.seed(42)
    data = np.vstack([
        np.random.randn(10, 2) + [0, 0],
        np.random.randn(10, 2) + [5, 5],
        np.random.randn(10, 2) + [10, 0],
    ]).astype(np.float32)
    
    save_array("clustering_input", data)
    
    # K-Means mit sklearn
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    
    save_json("clustering_output", {
        "labels": labels.tolist(),
        "centers": kmeans.cluster_centers_.tolist(),
        "n_clusters": 3
    })

if __name__ == "__main__":
    generate_image_processing_refs()
    generate_registration_refs()
    generate_metrics_refs()
    generate_clustering_refs()
    print(f"Reference outputs saved to {OUTPUT_DIR}")
```

### C++ Regressionstests

```cpp
// tests/regression/test_image_processing_regression.cpp

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tile_compile/image/processing.hpp"
#include "test_utils.hpp"

using namespace tile_compile;
using Catch::Matchers::WithinAbs;

namespace {

// Lade Referenz-Array aus .npy Datei
Matrix2Df load_reference(const std::string& name) {
    return test::load_npy("fixtures/reference_outputs/" + name + ".npy");
}

// Vergleiche zwei Matrizen mit Toleranz
void assert_matrices_equal(const Matrix2Df& actual, const Matrix2Df& expected,
                           float tolerance = 1e-5f) {
    REQUIRE(actual.rows() == expected.rows());
    REQUIRE(actual.cols() == expected.cols());
    
    for (int i = 0; i < actual.rows(); ++i) {
        for (int j = 0; j < actual.cols(); ++j) {
            REQUIRE_THAT(actual(i, j), WithinAbs(expected(i, j), tolerance));
        }
    }
}

} // anonymous namespace

TEST_CASE("split_cfa_channels regression", "[regression][image]") {
    Matrix2Df mosaic = load_reference("mosaic_4x4");
    Matrix2Df expected_R = load_reference("split_cfa_R");
    Matrix2Df expected_G = load_reference("split_cfa_G");
    Matrix2Df expected_B = load_reference("split_cfa_B");
    
    auto channels = image::split_cfa_channels(mosaic, BayerPattern::RGGB);
    
    assert_matrices_equal(channels["R"], expected_R);
    assert_matrices_equal(channels["G"], expected_G);
    assert_matrices_equal(channels["B"], expected_B);
}

TEST_CASE("reassemble_cfa_mosaic regression", "[regression][image]") {
    Matrix2Df R = load_reference("split_cfa_R");
    Matrix2Df G = load_reference("split_cfa_G");
    Matrix2Df B = load_reference("split_cfa_B");
    Matrix2Df expected = load_reference("reassemble_cfa");
    
    auto result = image::reassemble_cfa_mosaic(R, G, B, BayerPattern::RGGB);
    
    assert_matrices_equal(result, expected);
}

TEST_CASE("normalize_frame regression", "[regression][image]") {
    Matrix2Df input = load_reference("normalize_input");
    
    SECTION("background mode") {
        Matrix2Df expected = load_reference("normalize_background");
        auto result = image::normalize_frame(input, 750.0f, 1.0f, 
                                              image::NormalizationMode::BACKGROUND);
        assert_matrices_equal(result, expected, 1e-4f);
    }
    
    SECTION("additive mode") {
        Matrix2Df expected = load_reference("normalize_additive");
        auto result = image::normalize_frame(input, 750.0f, 500.0f,
                                              image::NormalizationMode::ADDITIVE);
        assert_matrices_equal(result, expected, 1e-4f);
    }
}

TEST_CASE("cosmetic_correction regression", "[regression][image]") {
    Matrix2Df input = load_reference("cosmetic_input");
    Matrix2Df expected = load_reference("cosmetic_output");
    
    auto result = image::cosmetic_correction(input, 5.0f, true);
    
    // Hotpixel sollte korrigiert sein
    REQUIRE(result(15, 15) < 1000.0f);
    
    // Andere Pixel sollten unverändert sein
    REQUIRE_THAT(result(0, 0), WithinAbs(expected(0, 0), 1e-4f));
}
```

---

## 8.4 Performance-Tests

### Benchmark-Framework

```cpp
// tests/benchmarks/benchmark_image.cpp

#include <benchmark/benchmark.h>
#include "tile_compile/image/processing.hpp"

using namespace tile_compile;

static void BM_split_cfa_channels(benchmark::State& state) {
    int size = state.range(0);
    Matrix2Df mosaic = Matrix2Df::Random(size, size);
    
    for (auto _ : state) {
        auto channels = image::split_cfa_channels(mosaic, BayerPattern::RGGB);
        benchmark::DoNotOptimize(channels);
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_split_cfa_channels)->Range(256, 4096);

static void BM_cosmetic_correction(benchmark::State& state) {
    int size = state.range(0);
    Matrix2Df data = Matrix2Df::Random(size, size) * 1000.0f;
    
    for (auto _ : state) {
        auto result = image::cosmetic_correction(data, 8.0f, true);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
}
BENCHMARK(BM_cosmetic_correction)->Range(256, 4096);

static void BM_ecc_registration(benchmark::State& state) {
    int size = state.range(0);
    Matrix2Df ref = Matrix2Df::Random(size, size);
    Matrix2Df moving = Matrix2Df::Random(size, size);
    
    auto ref01 = registration::prepare_ecc_image(ref);
    auto moving01 = registration::prepare_ecc_image(moving);
    auto init_warp = registration::identity_warp();
    
    for (auto _ : state) {
        auto result = registration::ecc_warp(moving01, ref01, false, init_warp);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ecc_registration)->Range(256, 2048);

BENCHMARK_MAIN();
```

### Performance-Vergleich Python vs C++

```python
#!/usr/bin/env python3
"""Compare Python vs C++ performance."""

import time
import subprocess
import numpy as np
from pathlib import Path

from runner.image_processing import split_cfa_channels, cosmetic_correction
from runner.opencv_registration import opencv_prepare_ecc_image, opencv_ecc_warp

def benchmark_python():
    """Benchmark Python implementation."""
    results = {}
    
    # split_cfa_channels
    for size in [256, 512, 1024, 2048, 4096]:
        mosaic = np.random.rand(size, size).astype(np.float32)
        
        start = time.perf_counter()
        for _ in range(10):
            channels = split_cfa_channels(mosaic, "RGGB")
        elapsed = (time.perf_counter() - start) / 10
        
        results[f"split_cfa_{size}"] = elapsed
        print(f"Python split_cfa {size}x{size}: {elapsed*1000:.2f} ms")
    
    # cosmetic_correction
    for size in [256, 512, 1024, 2048]:
        data = np.random.rand(size, size).astype(np.float32) * 1000
        
        start = time.perf_counter()
        for _ in range(10):
            result = cosmetic_correction(data, 8.0)
        elapsed = (time.perf_counter() - start) / 10
        
        results[f"cosmetic_{size}"] = elapsed
        print(f"Python cosmetic {size}x{size}: {elapsed*1000:.2f} ms")
    
    return results

def benchmark_cpp():
    """Run C++ benchmarks and parse results."""
    result = subprocess.run(
        ["./build/benchmarks", "--benchmark_format=json"],
        capture_output=True,
        text=True
    )
    # Parse JSON output...
    return {}

if __name__ == "__main__":
    print("=== Python Benchmarks ===")
    py_results = benchmark_python()
    
    print("\n=== C++ Benchmarks ===")
    # cpp_results = benchmark_cpp()
    
    print("\n=== Comparison ===")
    # Speedup berechnen...
```

---

## 8.5 End-to-End-Tests

### Vollständiger Pipeline-Test

```cpp
// tests/e2e/test_full_pipeline.cpp

#include <catch2/catch_test_macros.hpp>
#include "tile_compile/pipeline/phases_impl.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/utils.hpp"

using namespace tile_compile;

TEST_CASE("Full pipeline E2E", "[e2e][pipeline]") {
    // Test-Daten vorbereiten
    fs::path test_dir = fs::temp_directory_path() / "tile_compile_test";
    fs::create_directories(test_dir);
    
    // Test-Config erstellen
    fs::path config_path = test_dir / "config.yaml";
    std::ofstream config_file(config_path);
    config_file << R"(
global_metrics:
  weights:
    background: 0.4
    noise: 0.3
    gradient: 0.3
tile:
  size_factor: 8.0
  min_size: 32
  overlap_fraction: 0.25
synthetic:
  frames_min: 3
  frames_max: 10
registration:
  allow_rotation: false
calibration:
  enabled: false
)";
    config_file.close();
    
    // Test-Frames erstellen (synthetische Daten)
    fs::path input_dir = test_dir / "input";
    fs::create_directories(input_dir);
    
    for (int i = 0; i < 5; ++i) {
        Matrix2Df frame = Matrix2Df::Random(256, 256) * 1000.0f + 500.0f;
        io::FitsHeader header;
        io::write_fits_float(input_dir / ("frame_" + std::to_string(i) + ".fits"),
                             frame, header);
    }
    
    // Pipeline ausführen
    config::Config cfg = config::Config::load(config_path);
    auto frames = core::discover_frames(input_dir, "*.fits");
    
    std::string run_id = core::get_run_id();
    fs::path run_dir = test_dir / "runs" / run_id;
    fs::create_directories(run_dir);
    
    std::ofstream log_file(run_dir / "log.jsonl");
    
    pipeline::PipelineContext ctx;
    ctx.run_id = run_id;
    ctx.run_dir = run_dir;
    ctx.project_root = test_dir;
    ctx.cfg = cfg;
    ctx.input_frames = frames;
    ctx.log_stream = &log_file;
    
    pipeline::PipelineRunner runner;
    bool success = runner.run(ctx);
    
    REQUIRE(success);
    
    // Outputs prüfen
    REQUIRE(fs::exists(run_dir / "outputs"));
    
    // Aufräumen
    fs::remove_all(test_dir);
}
```

---

## 8.6 CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/cpp-tests.yml

name: C++ Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y \
          cmake \
          libeigen3-dev \
          libopencv-dev \
          libcfitsio-dev \
          libyaml-cpp-dev \
          nlohmann-json3-dev
    
    - name: Configure
      run: |
        mkdir build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
    
    - name: Build
      run: cmake --build build -j$(nproc)
    
    - name: Run tests
      run: |
        cd build
        ctest --output-on-failure
    
    - name: Generate reference outputs
      run: |
        cd tests
        python3 generate_references.py
    
    - name: Run regression tests
      run: |
        cd build
        ./tests --reporter=junit --out=test_results.xml "[regression]"
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: build/test_results.xml
```

---

## 8.7 Validierungs-Checkliste

### Funktionale Korrektheit

- [ ] Alle Unit-Tests bestehen
- [ ] Alle Regressionstests gegen Python-Referenz bestehen
- [ ] End-to-End-Pipeline produziert identische Ergebnisse wie Python
- [ ] FITS-Dateien sind korrekt lesbar/schreibbar
- [ ] Bayer-Pattern-Erkennung funktioniert
- [ ] Registrierung konvergiert korrekt
- [ ] Metriken sind numerisch stabil
- [ ] Clustering produziert sinnvolle Ergebnisse
- [ ] Rekonstruktion hat keine Artefakte
- [ ] Event-Logging ist vollständig

### Performance

- [ ] C++ ist mindestens 2x schneller als Python
- [ ] Speicherverbrauch ist akzeptabel
- [ ] Keine Memory-Leaks (Valgrind-Check)
- [ ] Parallelisierung funktioniert (OpenMP)

### Robustheit

- [ ] Fehlerbehandlung für ungültige Eingaben
- [ ] Graceful Degradation bei fehlenden Daten
- [ ] Stop-Mechanismus funktioniert
- [ ] Resume funktioniert korrekt

### Kompatibilität

- [ ] GUI kann C++ Backend aufrufen
- [ ] Event-Format ist kompatibel
- [ ] Config-Format ist kompatibel
- [ ] Output-Struktur ist identisch

---

## Checkliste Phase 8

- [ ] Catch2 Test-Framework eingerichtet
- [ ] Unit-Tests für alle Module geschrieben
- [ ] Python-Referenz-Generator erstellt
- [ ] Regressionstests implementiert
- [ ] Performance-Benchmarks erstellt
- [ ] End-to-End-Tests implementiert
- [ ] CI/CD Pipeline konfiguriert
- [ ] Alle Tests bestehen
- [ ] Performance-Ziele erreicht
- [ ] Dokumentation vollständig
