# Phase 1: Core-Infrastruktur

## Ziel

Aufbau der grundlegenden C++ Projektstruktur und Core-Utilities.

**Geschätzte Dauer**: 2-3 Wochen

---

## 1.1 Projektstruktur erstellen

```
tile_compile_cpp/
├── CMakeLists.txt
├── vcpkg.json
├── include/
│   └── tile_compile/
│       ├── core/
│       │   ├── types.hpp
│       │   ├── utils.hpp
│       │   ├── events.hpp
│       │   └── error.hpp
│       └── version.hpp
├── src/
│   └── core/
│       ├── utils.cpp
│       └── events.cpp
├── apps/
│   └── tile_compile_runner.cpp
└── tests/
    └── core/
        └── test_utils.cpp
```

---

## 1.2 Grundlegende Typen (types.hpp)

```cpp
#pragma once

#include <Eigen/Dense>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace tile_compile {

namespace fs = std::filesystem;

// Grundlegende Matrix-Typen
using Matrix2Df = Eigen::MatrixXf;
using Matrix2Di = Eigen::MatrixXi;
using Matrix2Db = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXf = Eigen::VectorXf;
using VectorXi = Eigen::VectorXi;

// 3D-Tensor für RGB-Bilder (Channel-First: 3 x H x W)
using Tensor3Df = Eigen::Tensor<float, 3>;

// Affine Transformation (2x3 Matrix)
using AffineWarp = Eigen::Matrix<float, 2, 3>;

// Konfigurationswerte
using ConfigValue = std::variant<
    bool,
    int,
    float,
    double,
    std::string,
    std::vector<int>,
    std::vector<float>,
    std::vector<std::string>
>;
using ConfigMap = std::map<std::string, ConfigValue>;

// Bayer-Pattern
enum class BayerPattern {
    RGGB,
    BGGR,
    GBRG,
    GRBG,
    UNKNOWN
};

BayerPattern parse_bayer_pattern(const std::string& pattern);
std::string bayer_pattern_to_string(BayerPattern pattern);

// Farbmodus
enum class ColorMode {
    OSC,    // One-Shot-Color (Bayer)
    RGB,    // Bereits demosaiced RGB
    MONO    // Monochrom
};

ColorMode parse_color_mode(const std::string& mode);

} // namespace tile_compile
```

---

## 1.3 Fehlerbehandlung (error.hpp)

```cpp
#pragma once

#include <stdexcept>
#include <string>

namespace tile_compile {

class TileCompileError : public std::runtime_error {
public:
    explicit TileCompileError(const std::string& message)
        : std::runtime_error(message) {}
};

class FitsError : public TileCompileError {
public:
    FitsError(int status, const std::string& context = "");
    int status() const { return status_; }
private:
    int status_;
};

class ConfigError : public TileCompileError {
public:
    explicit ConfigError(const std::string& message)
        : TileCompileError(message) {}
};

class RegistrationError : public TileCompileError {
public:
    explicit RegistrationError(const std::string& message)
        : TileCompileError(message) {}
};

class PipelineError : public TileCompileError {
public:
    PipelineError(int phase, const std::string& message);
    int phase() const { return phase_; }
private:
    int phase_;
};

} // namespace tile_compile
```

---

## 1.4 Utilities (utils.hpp / utils.cpp)

### Header

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include <chrono>
#include <vector>

namespace tile_compile::core {

// Dateisystem-Utilities
void safe_symlink_or_copy(const fs::path& src, const fs::path& dst);
void safe_hardlink_or_copy(const fs::path& src, const fs::path& dst);
fs::path pick_output_file(const fs::path& dir, const std::string& prefix,
                          const std::string& ext);

// Frame-Discovery
std::vector<fs::path> discover_frames(const fs::path& input_dir,
                                       const std::string& pattern);

// Datei-Utilities
std::vector<uint8_t> read_bytes(const fs::path& path);
void write_bytes(const fs::path& path, const std::vector<uint8_t>& data);
std::string read_text(const fs::path& path);
void write_text(const fs::path& path, const std::string& text);

// Hash-Funktionen
std::string sha256_bytes(const std::vector<uint8_t>& data);
std::string sha256_file(const fs::path& path);

// Konfiguration
void copy_config(const fs::path& src, const fs::path& dst);
fs::path resolve_project_root(const fs::path& config_path);
// Zeit-Utilities
std::string get_iso_timestamp();
std::string get_run_id();

// Mathematische Utilities
float compute_median(const Matrix2Df& data);
float compute_median(const VectorXf& data);
float compute_mad(const Matrix2Df& data);  // Median Absolute Deviation
float compute_robust_sigma(const Matrix2Df& data);  // 1.4826 * MAD

// Glob-Pattern-Matching
bool matches_glob(const std::string& filename, const std::string& pattern);

} // namespace tile_compile::core
```

### Implementierung (Auszug)

```cpp
#include "tile_compile/core/utils.hpp"
#include <algorithm>
#include <fstream>
#include <random>
#include <regex>

// Für SHA256
#include "picosha2.h"  // oder OpenSSL

namespace tile_compile::core {

std::vector<fs::path> discover_frames(const fs::path& input_dir,
                                       const std::string& pattern) {
    std::vector<fs::path> frames;
    
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        return frames;
    }
    
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;
        
        std::string filename = entry.path().filename().string();
        if (matches_glob(filename, pattern)) {
            frames.push_back(entry.path());
        }
    }
    
    std::sort(frames.begin(), frames.end());
    return frames;
}

std::string sha256_bytes(const std::vector<uint8_t>& data) {
    return picosha2::hash256_hex_string(data);
}

std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::gmtime(&time_t);
    
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buffer);
}

std::string get_run_id() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::gmtime(&time_t);
    
    char time_part[20];
    std::strftime(time_part, sizeof(time_part), "%Y%m%d_%H%M%S", &tm);
    
    // 8 Zeichen Hex-UUID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);
    
    char uuid_part[9];
    std::snprintf(uuid_part, sizeof(uuid_part), "%08x", dis(gen));
    
    return std::string(time_part) + "_" + uuid_part;
}

float compute_median(const Matrix2Df& data) {
    std::vector<float> values(data.data(), data.data() + data.size());
    
    if (values.empty()) return 0.0f;
    
    size_t n = values.size();
    std::nth_element(values.begin(), values.begin() + n / 2, values.end());
    
    if (n % 2 == 0) {
        float mid1 = values[n / 2];
        std::nth_element(values.begin(), values.begin() + n / 2 - 1, values.end());
        float mid2 = values[n / 2 - 1];
        return (mid1 + mid2) / 2.0f;
    }
    
    return values[n / 2];
}

float compute_robust_sigma(const Matrix2Df& data) {
    float med = compute_median(data);
    
    Matrix2Df abs_dev = (data.array() - med).abs();
    float mad = compute_median(abs_dev);
    
    return 1.4826f * mad;
}

bool matches_glob(const std::string& filename, const std::string& pattern) {
    // Einfache Glob-Implementierung für *.fit* etc.
    std::string regex_pattern = pattern;
    
    // Escape special regex characters
    regex_pattern = std::regex_replace(regex_pattern, std::regex("\\."), "\\.");
    // Convert glob wildcards to regex
    regex_pattern = std::regex_replace(regex_pattern, std::regex("\\*"), ".*");
    regex_pattern = std::regex_replace(regex_pattern, std::regex("\\?"), ".");
    
    std::regex re(regex_pattern, std::regex::icase);
    return std::regex_match(filename, re);
}

} // namespace tile_compile::core
```

---

## 1.5 Event-System (events.hpp / events.cpp)

### Header

```cpp
#pragma once

#include <nlohmann/json.hpp>
#include <atomic>
#include <mutex>
#include <ostream>
#include <string>

namespace tile_compile::core {

using json = nlohmann::json;

class EventEmitter {
public:
    EventEmitter() = default;
    
    // Basis-Event-Emission
    void emit(const json& event, std::ostream& log_fp);
    
    // Phasen-Events
    void phase_start(const std::string& run_id, int phase,
                     const std::string& name, std::ostream& log_fp);
    
    void phase_end(const std::string& run_id, int phase,
                   const std::string& status, std::ostream& log_fp,
                   const json& extra = {});
    
    void phase_progress(const std::string& run_id, int phase,
                        float progress, const std::string& message,
                        std::ostream& log_fp);
    
    // Run-Events
    void run_start(const std::string& run_id, const json& metadata,
                   std::ostream& log_fp);
    
    void run_end(const std::string& run_id, bool success,
                 const std::string& status, std::ostream& log_fp,
                 const json& extra = {});
    
    void run_error(const std::string& run_id, const std::string& error,
                   const std::string& traceback, std::ostream& log_fp);
    
    // Stop-Mechanismus
    bool stop_requested() const { return stop_flag_.load(); }
    void request_stop() { stop_flag_.store(true); }
    void reset_stop() { stop_flag_.store(false); }

private:
    std::atomic<bool> stop_flag_{false};
    std::mutex emit_mutex_;
    
    std::string get_timestamp() const;
};

// Globale Instanz (optional)
EventEmitter& get_event_emitter();

} // namespace tile_compile::core
```

### Implementierung

```cpp
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/utils.hpp"

namespace tile_compile::core {

void EventEmitter::emit(const json& event, std::ostream& log_fp) {
    std::lock_guard<std::mutex> lock(emit_mutex_);
    log_fp << event.dump() << "\n";
    log_fp.flush();
}

void EventEmitter::phase_start(const std::string& run_id, int phase,
                                const std::string& name, std::ostream& log_fp) {
    json event = {
        {"type", "phase_start"},
        {"run_id", run_id},
        {"phase", phase},
        {"name", name},
        {"ts", get_timestamp()}
    };
    emit(event, log_fp);
}

void EventEmitter::phase_end(const std::string& run_id, int phase,
                              const std::string& status, std::ostream& log_fp,
                              const json& extra) {
    json event = {
        {"type", "phase_end"},
        {"run_id", run_id},
        {"phase", phase},
        {"status", status},
        {"ts", get_timestamp()}
    };
    
    // Merge extra fields
    for (auto& [key, value] : extra.items()) {
        event[key] = value;
    }
    
    emit(event, log_fp);
}

void EventEmitter::phase_progress(const std::string& run_id, int phase,
                                   float progress, const std::string& message,
                                   std::ostream& log_fp) {
    json event = {
        {"type", "phase_progress"},
        {"run_id", run_id},
        {"phase", phase},
        {"progress", progress},
        {"message", message},
        {"ts", get_timestamp()}
    };
    emit(event, log_fp);
}

void EventEmitter::run_start(const std::string& run_id, const json& metadata,
                              std::ostream& log_fp) {
    json event = {
        {"type", "run_start"},
        {"run_id", run_id},
        {"ts", get_timestamp()}
    };
    
    for (auto& [key, value] : metadata.items()) {
        event[key] = value;
    }
    
    emit(event, log_fp);
}

void EventEmitter::run_end(const std::string& run_id, bool success,
                            const std::string& status, std::ostream& log_fp,
                            const json& extra) {
    json event = {
        {"type", "run_end"},
        {"run_id", run_id},
        {"success", success},
        {"status", status},
        {"ts", get_timestamp()}
    };
    
    for (auto& [key, value] : extra.items()) {
        event[key] = value;
    }
    
    emit(event, log_fp);
}

void EventEmitter::run_error(const std::string& run_id, const std::string& error,
                              const std::string& traceback, std::ostream& log_fp) {
    json event = {
        {"type", "run_error"},
        {"run_id", run_id},
        {"error", error},
        {"traceback", traceback},
        {"ts", get_timestamp()}
    };
    emit(event, log_fp);
}

std::string EventEmitter::get_timestamp() const {
    return get_iso_timestamp();
}

EventEmitter& get_event_emitter() {
    static EventEmitter instance;
    return instance;
}

} // namespace tile_compile::core
```

---

## 1.6 Tests

```cpp
// tests/core/test_utils.cpp
#include <catch2/catch_test_macros.hpp>
#include "tile_compile/core/utils.hpp"

using namespace tile_compile::core;

TEST_CASE("compute_median", "[core][utils]") {
    SECTION("odd number of elements") {
        Eigen::VectorXf v(5);
        v << 1, 3, 5, 7, 9;
        REQUIRE(compute_median(v) == 5.0f);
    }
    
    SECTION("even number of elements") {
        Eigen::VectorXf v(4);
        v << 1, 3, 5, 7;
        REQUIRE(compute_median(v) == 4.0f);
    }
}

TEST_CASE("matches_glob", "[core][utils]") {
    REQUIRE(matches_glob("image.fits", "*.fits"));
    REQUIRE(matches_glob("image.fit", "*.fit*"));
    REQUIRE(matches_glob("IMAGE.FITS", "*.fits"));  // case insensitive
    REQUIRE_FALSE(matches_glob("image.png", "*.fits"));
}

TEST_CASE("get_run_id", "[core][utils]") {
    std::string id1 = get_run_id();
    std::string id2 = get_run_id();
    
    // Format: YYYYMMDD_HHMMSS_xxxxxxxx
    REQUIRE(id1.length() == 24);
    REQUIRE(id1[8] == '_');
    REQUIRE(id1[15] == '_');
    
    // Sollten unterschiedlich sein (UUID-Teil)
    // Hinweis: Kann theoretisch gleich sein, aber sehr unwahrscheinlich
}
```

---

## Checkliste Phase 1

- [ ] CMakeLists.txt erstellt
- [ ] vcpkg.json / conanfile.txt erstellt
- [ ] Projektstruktur angelegt
- [ ] types.hpp implementiert
- [ ] error.hpp implementiert
- [ ] utils.hpp/cpp implementiert
- [ ] events.hpp/cpp implementiert
- [ ] picosha2.h integriert (oder OpenSSL)
- [ ] Unit-Tests für core/ geschrieben
- [ ] Build-System funktioniert
- [ ] Alle Tests bestehen
