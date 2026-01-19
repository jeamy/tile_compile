# C++ Abhängigkeiten und Bibliotheken

## Übersicht

Dieses Dokument beschreibt alle erforderlichen C++ Bibliotheken und deren Mapping zu den Python-Äquivalenten.

**Haupt-Entwicklungsplattform**: Fedora Linux

---

## Schnellinstallation (Fedora)

Alle erforderlichen Abhängigkeiten mit einem Befehl:

```bash
# Kern-Bibliotheken
sudo dnf install -y \
    eigen3-devel \
    opencv-devel \
    cfitsio-devel \
    yaml-cpp-devel \
    json-devel \
    openssl-devel

# Optionale Bibliotheken
sudo dnf install -y \
    mlpack-devel \
    spdlog-devel \
    cli11-devel \
    catch2-devel

# Build-Tools
sudo dnf install -y \
    cmake \
    gcc-c++ \
    make \
    git
```

---

## Schnellinstallation (Ubuntu/Debian)

```bash
# Kern-Bibliotheken
sudo apt install -y \
    libeigen3-dev \
    libopencv-dev \
    libcfitsio-dev \
    libyaml-cpp-dev \
    nlohmann-json3-dev \
    libssl-dev

# Optionale Bibliotheken
sudo apt install -y \
    libmlpack-dev \
    libspdlog-dev \
    catch2

# Build-Tools
sudo apt install -y \
    cmake \
    g++ \
    make \
    git
```

---

## Kern-Bibliotheken

### 1. Eigen3 (NumPy-Ersatz)

**Python**: `numpy`  
**C++**: `Eigen3`  
**Version**: ≥3.4

**Installation**:
```bash
# Fedora (Haupt-Entwicklungsplattform)
sudo dnf install eigen3-devel

# Ubuntu/Debian
sudo apt install libeigen3-dev

# vcpkg
vcpkg install eigen3

# Conan
conan install eigen/3.4.0
```

**CMake**:
```cmake
find_package(Eigen3 3.4 REQUIRED)
target_link_libraries(tile_compile PRIVATE Eigen3::Eigen)
```

**Verwendung**:
```cpp
#include <Eigen/Dense>

// NumPy: arr = np.zeros((h, w), dtype=np.float32)
Eigen::MatrixXf arr = Eigen::MatrixXf::Zero(h, w);

// NumPy: arr = np.array([1, 2, 3])
Eigen::VectorXf vec(3);
vec << 1, 2, 3;

// NumPy: result = np.median(arr)
// Eigen hat kein eingebautes median, muss implementiert werden
float median = compute_median(arr);

// NumPy: result = arr.mean()
float mean = arr.mean();

// NumPy: result = arr.std()
float std = std::sqrt((arr.array() - mean).square().mean());
```

---

### 2. OpenCV (cv2-Ersatz)

**Python**: `cv2` (opencv-python)  
**C++**: `OpenCV`  
**Version**: ≥4.5

**Installation**:
```bash
# Fedora (Haupt-Entwicklungsplattform)
sudo dnf install opencv-devel

# Ubuntu/Debian
sudo apt install libopencv-dev

# vcpkg
vcpkg install opencv4

# Conan
conan install opencv/4.5.5
```

**CMake**:
```cmake
find_package(OpenCV 4.5 REQUIRED)
target_link_libraries(tile_compile PRIVATE ${OpenCV_LIBS})
```

**Verwendung**:
```cpp
#include <opencv2/opencv.hpp>

// Python: cv2.findTransformECC(...)
cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
double cc = cv::findTransformECC(ref, moving, warp, cv::MOTION_TRANSLATION, criteria);

// Python: cv2.warpAffine(...)
cv::Mat warped;
cv::warpAffine(src, warped, warp, size, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);

// Python: cv2.GaussianBlur(...)
cv::Mat blurred;
cv::GaussianBlur(src, blurred, cv::Size(0, 0), sigma);

// Python: cv2.cvtColor(mosaic, cv2.COLOR_BAYER_GB2RGB)
cv::Mat rgb;
cv::cvtColor(mosaic, rgb, cv::COLOR_BayerGB2RGB);
```

---

### 3. CFITSIO (Astropy FITS-Ersatz)

**Python**: `astropy.io.fits`  
**C++**: `CFITSIO`  
**Version**: ≥4.0

**Installation**:
```bash
# Fedora (Haupt-Entwicklungsplattform)
sudo dnf install cfitsio-devel

# Ubuntu/Debian
sudo apt install libcfitsio-dev

# vcpkg
vcpkg install cfitsio

# Conan
conan install cfitsio/4.2.0
```

**CMake**:
```cmake
find_package(CFITSIO REQUIRED)
target_link_libraries(tile_compile PRIVATE CFITSIO::CFITSIO)
```

**Wrapper-Klasse**:
```cpp
#include <fitsio.h>

namespace tile_compile::io {

class FitsFile {
public:
    static std::pair<Eigen::MatrixXf, FitsHeader> read(const fs::path& path) {
        fitsfile* fptr;
        int status = 0;
        
        fits_open_file(&fptr, path.c_str(), READONLY, &status);
        if (status) throw FitsError(status);
        
        // Dimensionen lesen
        int naxis;
        long naxes[3];
        fits_get_img_dim(fptr, &naxis, &status);
        fits_get_img_size(fptr, 3, naxes, &status);
        
        // Daten lesen
        long fpixel[3] = {1, 1, 1};
        long nelements = naxes[0] * naxes[1];
        std::vector<float> buffer(nelements);
        
        fits_read_pix(fptr, TFLOAT, fpixel, nelements, nullptr, 
                      buffer.data(), nullptr, &status);
        
        // In Eigen-Matrix konvertieren
        Eigen::MatrixXf data(naxes[1], naxes[0]);
        // ... Daten kopieren ...
        
        fits_close_file(fptr, &status);
        return {data, header};
    }
    
    static void write(const fs::path& path, const Eigen::MatrixXf& data,
                      const FitsHeader& header) {
        // ... Implementierung ...
    }
};

} // namespace tile_compile::io
```

---

### 4. yaml-cpp (PyYAML-Ersatz)

**Python**: `yaml`  
**C++**: `yaml-cpp`  
**Version**: ≥0.7

**Installation**:
```bash
# Fedora (Haupt-Entwicklungsplattform)
sudo dnf install yaml-cpp-devel

# Ubuntu/Debian
sudo apt install libyaml-cpp-dev

# vcpkg
vcpkg install yaml-cpp

# Conan
conan install yaml-cpp/0.7.0
```

**CMake**:
```cmake
find_package(yaml-cpp REQUIRED)
target_link_libraries(tile_compile PRIVATE yaml-cpp)
```

**Verwendung**:
```cpp
#include <yaml-cpp/yaml.h>

// Python: cfg = yaml.safe_load(text)
YAML::Node cfg = YAML::Load(yaml_text);

// Python: value = cfg.get("key", default)
auto value = cfg["key"].as<std::string>("default");

// Python: nested = cfg["section"]["subsection"]
auto nested = cfg["section"]["subsection"];
```

---

### 5. nlohmann/json (JSON-Ersatz)

**Python**: `json`  
**C++**: `nlohmann/json`  
**Version**: ≥3.10

**Installation**:
```bash
# Fedora (Haupt-Entwicklungsplattform)
sudo dnf install json-devel

# Ubuntu/Debian
sudo apt install nlohmann-json3-dev

# vcpkg
vcpkg install nlohmann-json

# Conan
conan install nlohmann_json/3.11.2

# Header-only (einfachste Option)
# Einfach json.hpp in include/ kopieren
```

**CMake**:
```cmake
find_package(nlohmann_json REQUIRED)
target_link_libraries(tile_compile PRIVATE nlohmann_json::nlohmann_json)
```

**Verwendung**:
```cpp
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Python: event = {"type": "phase_start", "run_id": run_id}
json event = {
    {"type", "phase_start"},
    {"run_id", run_id},
    {"ts", get_iso_timestamp()}
};

// Python: json.dumps(event)
std::string json_str = event.dump();

// Python: log_fp.write(json_str + "\n")
log_fp << json_str << "\n";
```

---

### 6. mlpack oder dlib (scikit-learn-Ersatz)

**Python**: `sklearn.cluster.KMeans`, `sklearn.preprocessing.StandardScaler`  
**C++**: `mlpack` oder `dlib`

#### Option A: mlpack

**Installation**:
```bash
# Fedora (Haupt-Entwicklungsplattform)
sudo dnf install mlpack-devel

# Ubuntu/Debian
sudo apt install libmlpack-dev

# vcpkg
vcpkg install mlpack
```

**CMake**:
```cmake
find_package(mlpack REQUIRED)
target_link_libraries(tile_compile PRIVATE mlpack::mlpack)
```

**Verwendung**:
```cpp
#include <mlpack/methods/kmeans/kmeans.hpp>

// K-Means Clustering
arma::mat data = eigen_to_arma(state_vectors);
arma::Row<size_t> assignments;
arma::mat centroids;

mlpack::kmeans::KMeans<> kmeans;
kmeans.Cluster(data, n_clusters, assignments, centroids);
```

#### Option B: Eigen-basierte Implementierung

Für einfache K-Means kann eine eigene Implementierung sinnvoller sein:

```cpp
namespace tile_compile::clustering {

class KMeans {
public:
    KMeans(int n_clusters, int max_iters = 100, int n_init = 10);
    
    std::pair<Eigen::VectorXi, Eigen::MatrixXf> fit_predict(
        const Eigen::MatrixXf& data
    );

private:
    int n_clusters_;
    int max_iters_;
    int n_init_;
};

} // namespace tile_compile::clustering
```

---

### 7. Wavelet-Bibliothek (PyWavelets-Ersatz)

**Python**: `pywt`  
**C++**: Optionen:

#### Option A: wavelib

```bash
git clone https://github.com/rafat/wavelib.git
```

#### Option B: Eigen-basierte Haar-Wavelet-Implementierung

Da nur Haar-Wavelets verwendet werden, ist eine eigene Implementierung einfach:

```cpp
namespace tile_compile::wavelets {

struct WaveletCoeffs2D {
    Eigen::MatrixXf approx;
    std::vector<std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>> details;
};

WaveletCoeffs2D wavedec2_haar(const Eigen::MatrixXf& data, int level);

Eigen::MatrixXf waverec2_haar(const WaveletCoeffs2D& coeffs);

} // namespace tile_compile::wavelets
```

---

### 8. OpenSSL oder picosha2 (hashlib-Ersatz)

**Python**: `hashlib.sha256`  
**C++**: `OpenSSL` oder `picosha2` (header-only)

#### Option: picosha2 (empfohlen, header-only)

```cpp
#include "picosha2.h"

std::string sha256_bytes(const std::vector<uint8_t>& data) {
    return picosha2::hash256_hex_string(data);
}
```

---

## Optionale Bibliotheken

### 9. spdlog (Logging)

**Empfohlen** für strukturiertes Logging:

```bash
# Fedora
sudo dnf install spdlog-devel

# Ubuntu/Debian
sudo apt install libspdlog-dev

# vcpkg
vcpkg install spdlog
```

```cpp
#include <spdlog/spdlog.h>

spdlog::info("Phase {} started: {}", phase_num, phase_name);
spdlog::error("Registration failed: {}", error_msg);
```

---

### 10. CLI11 (Argument Parsing)

**Empfohlen** für CLI:

```bash
# Fedora
sudo dnf install cli11-devel

# Ubuntu/Debian (header-only, manuell oder vcpkg)
# vcpkg
vcpkg install cli11
```

```cpp
#include <CLI/CLI.hpp>

int main(int argc, char** argv) {
    CLI::App app{"Tile-Compile Runner"};
    
    std::string config_path;
    app.add_option("--config", config_path, "Path to config.yaml")->required();
    
    std::string input_dir;
    app.add_option("--input-dir", input_dir, "Input directory")->required();
    
    CLI11_PARSE(app, argc, argv);
    // ...
}
```

---

### 11. Catch2 oder GoogleTest (Testing)

**Empfohlen** für Unit-Tests:

```bash
# Fedora
sudo dnf install catch2-devel
# oder
sudo dnf install gtest-devel

# Ubuntu/Debian
sudo apt install catch2
# oder
sudo apt install libgtest-dev

# vcpkg
vcpkg install catch2
# oder
vcpkg install gtest
```

---

## CMakeLists.txt Vorlage

```cmake
cmake_minimum_required(VERSION 3.16)
project(tile_compile VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Abhängigkeiten
find_package(Eigen3 3.4 REQUIRED)
find_package(OpenCV 4.5 REQUIRED)
find_package(CFITSIO REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenSSL REQUIRED)  # oder picosha2 header-only

# Optional
find_package(spdlog)
find_package(CLI11)
find_package(Catch2)

# Bibliothek
add_library(tile_compile_lib
    src/core/events.cpp
    src/core/utils.cpp
    src/io/fits_utils.cpp
    src/image/processing.cpp
    src/registration/opencv_registration.cpp
    src/calibration/calibration.cpp
    src/metrics/metrics.cpp
    src/clustering/clustering.cpp
    src/reconstruction/reconstruction.cpp
    src/synthetic/synthetic.cpp
    src/stacking/sigma_clipping.cpp
    src/grid/tile_grid.cpp
    src/validation/linearity.cpp
    src/config/validate.cpp
    src/pipeline/phases_impl.cpp
)

target_include_directories(tile_compile_lib PUBLIC include)

target_link_libraries(tile_compile_lib PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    CFITSIO::CFITSIO
    yaml-cpp
    nlohmann_json::nlohmann_json
    OpenSSL::Crypto
)

# CLI-Executable
add_executable(tile_compile_runner apps/tile_compile_runner.cpp)
target_link_libraries(tile_compile_runner PRIVATE tile_compile_lib)

if(CLI11_FOUND)
    target_link_libraries(tile_compile_runner PRIVATE CLI11::CLI11)
endif()

# Tests
if(Catch2_FOUND)
    enable_testing()
    add_executable(tests
        tests/test_image_processing.cpp
        tests/test_registration.cpp
        tests/test_metrics.cpp
        # ...
    )
    target_link_libraries(tests PRIVATE tile_compile_lib Catch2::Catch2WithMain)
    add_test(NAME tile_compile_tests COMMAND tests)
endif()
```

---

## vcpkg.json Manifest

```json
{
  "name": "tile-compile",
  "version": "1.0.0",
  "dependencies": [
    "eigen3",
    "opencv4",
    "cfitsio",
    "yaml-cpp",
    "nlohmann-json",
    "openssl",
    "spdlog",
    "cli11",
    "catch2"
  ]
}
```

---

## Conan conanfile.txt

```ini
[requires]
eigen/3.4.0
opencv/4.5.5
cfitsio/4.2.0
yaml-cpp/0.7.0
nlohmann_json/3.11.2
openssl/3.1.0
spdlog/1.11.0
cli11/2.3.2
catch2/3.3.2

[generators]
CMakeDeps
CMakeToolchain
```

---

## Zusammenfassung der Bibliotheken

| Funktion | Python | C++ | Priorität |
|----------|--------|-----|-----------|
| Arrays/Matrizen | numpy | Eigen3 | Kritisch |
| Bildverarbeitung | cv2 | OpenCV | Kritisch |
| FITS I/O | astropy.io.fits | CFITSIO | Kritisch |
| YAML Parsing | yaml | yaml-cpp | Kritisch |
| JSON | json | nlohmann/json | Kritisch |
| K-Means | sklearn | mlpack/eigen | Hoch |
| Wavelets | pywt | wavelib/eigen | Mittel |
| SHA256 | hashlib | picosha2 | Niedrig |
| Logging | logging | spdlog | Optional |
| CLI | argparse | CLI11 | Optional |
| Testing | pytest | Catch2 | Optional |
