# Phase 2: I/O und Konfiguration

## Ziel

Implementierung der FITS-I/O und Konfigurationsverarbeitung.

**Geschätzte Dauer**: 1-2 Wochen

---

## 2.1 FITS I/O (io/fits_utils.hpp)

### Header

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include <map>
#include <optional>
#include <string>
#include <variant>

namespace tile_compile::io {

// FITS Header-Repräsentation
using HeaderValue = std::variant<bool, int, long, float, double, std::string>;

struct FitsHeader {
    std::map<std::string, HeaderValue> keywords;
    std::vector<std::string> comments;
    std::vector<std::string> history;
    
    template<typename T>
    std::optional<T> get(const std::string& key) const;
    
    template<typename T>
    T get_or(const std::string& key, T default_value) const;
    
    void set(const std::string& key, const HeaderValue& value);
    bool has(const std::string& key) const;
};

// FITS-Datei-Prüfung
bool is_fits_image_path(const fs::path& path);
bool is_fits_file(const fs::path& path);

// FITS lesen
std::pair<Matrix2Df, FitsHeader> read_fits_float(const fs::path& path);
std::pair<Tensor3Df, FitsHeader> read_fits_rgb(const fs::path& path);

// FITS schreiben
void write_fits_float(const fs::path& path, const Matrix2Df& data,
                      const FitsHeader& header = {});
void write_fits_rgb(const fs::path& path, const Tensor3Df& data,
                    const FitsHeader& header = {});

// Bayer/CFA-Erkennung
bool fits_is_cfa(const FitsHeader& header);
std::string fits_get_bayerpat(const FitsHeader& header);
BayerPattern fits_get_bayer_pattern(const FitsHeader& header);

// Hilfsfunktionen
int fits_get_bitpix(const fs::path& path);
std::pair<int, int> fits_get_dimensions(const fs::path& path);

} // namespace tile_compile::io
```

### Implementierung

```cpp
#include "tile_compile/io/fits_utils.hpp"
#include "tile_compile/core/error.hpp"
#include <fitsio.h>
#include <algorithm>
#include <cctype>

namespace tile_compile::io {

namespace {

void check_fits_status(int status, const std::string& context = "") {
    if (status != 0) {
        throw FitsError(status, context);
    }
}

std::string fits_error_message(int status) {
    char errmsg[FLEN_ERRMSG];
    fits_get_errstatus(status, errmsg);
    return std::string(errmsg);
}

} // anonymous namespace

FitsError::FitsError(int status, const std::string& context)
    : TileCompileError(context.empty() 
        ? fits_error_message(status)
        : context + ": " + fits_error_message(status))
    , status_(status) {}

bool is_fits_image_path(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    return ext == ".fits" || ext == ".fit" || ext == ".fts" || ext == ".fz";
}

std::pair<Matrix2Df, FitsHeader> read_fits_float(const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    fits_open_file(&fptr, path.string().c_str(), READONLY, &status);
    check_fits_status(status, "Opening " + path.string());
    
    // Cleanup-Guard
    struct FitsGuard {
        fitsfile* ptr;
        ~FitsGuard() { 
            int s = 0; 
            if (ptr) fits_close_file(ptr, &s); 
        }
    } guard{fptr};
    
    // Dimensionen lesen
    int naxis = 0;
    fits_get_img_dim(fptr, &naxis, &status);
    check_fits_status(status, "Getting image dimensions");
    
    if (naxis < 2) {
        throw FitsError(0, "Expected 2D or 3D image, got " + std::to_string(naxis) + "D");
    }
    
    std::vector<long> naxes(naxis);
    fits_get_img_size(fptr, naxis, naxes.data(), &status);
    check_fits_status(status, "Getting image size");
    
    long width = naxes[0];
    long height = naxes[1];
    long nelements = width * height;
    
    // Für 3D: nur erste Ebene lesen (oder Summe)
    if (naxis == 3 && naxes[2] > 1) {
        // Warnung: 3D-Bild, lese nur erste Ebene
    }
    
    // Daten lesen
    std::vector<float> buffer(nelements);
    std::vector<long> fpixel(naxis, 1);
    
    fits_read_pix(fptr, TFLOAT, fpixel.data(), nelements, nullptr,
                  buffer.data(), nullptr, &status);
    check_fits_status(status, "Reading pixel data");
    
    // In Eigen-Matrix konvertieren (FITS ist column-major, Eigen ist column-major)
    // Aber FITS speichert [x, y], wir wollen [row, col] = [y, x]
    Matrix2Df data(height, width);
    for (long y = 0; y < height; ++y) {
        for (long x = 0; x < width; ++x) {
            data(y, x) = buffer[y * width + x];
        }
    }
    
    // Header lesen
    FitsHeader header = read_fits_header(fptr);
    
    return {data, header};
}

void write_fits_float(const fs::path& path, const Matrix2Df& data,
                      const FitsHeader& header) {
    // Existierende Datei löschen
    if (fs::exists(path)) {
        fs::remove(path);
    }
    
    fitsfile* fptr = nullptr;
    int status = 0;
    
    fits_create_file(&fptr, path.string().c_str(), &status);
    check_fits_status(status, "Creating " + path.string());
    
    struct FitsGuard {
        fitsfile* ptr;
        ~FitsGuard() { 
            int s = 0; 
            if (ptr) fits_close_file(ptr, &s); 
        }
    } guard{fptr};
    
    // Bild erstellen
    long naxes[2] = {data.cols(), data.rows()};
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    check_fits_status(status, "Creating image");
    
    // Daten schreiben
    std::vector<float> buffer(data.size());
    for (long y = 0; y < data.rows(); ++y) {
        for (long x = 0; x < data.cols(); ++x) {
            buffer[y * data.cols() + x] = data(y, x);
        }
    }
    
    long fpixel[2] = {1, 1};
    fits_write_pix(fptr, TFLOAT, fpixel, data.size(), buffer.data(), &status);
    check_fits_status(status, "Writing pixel data");
    
    // Header schreiben
    write_fits_header(fptr, header);
}

bool fits_is_cfa(const FitsHeader& header) {
    // Prüfe auf Bayer-Pattern-Keywords
    if (header.has("BAYERPAT")) return true;
    if (header.has("COLORTYP")) {
        auto val = header.get<std::string>("COLORTYP");
        if (val && (*val == "BAYER" || val->find("CFA") != std::string::npos)) {
            return true;
        }
    }
    if (header.has("XBAYROFF") || header.has("YBAYROFF")) return true;
    
    return false;
}

std::string fits_get_bayerpat(const FitsHeader& header) {
    if (header.has("BAYERPAT")) {
        auto val = header.get<std::string>("BAYERPAT");
        if (val) return *val;
    }
    
    // Fallback: aus XBAYROFF/YBAYROFF ableiten
    auto xoff = header.get<int>("XBAYROFF");
    auto yoff = header.get<int>("YBAYROFF");
    
    if (xoff && yoff) {
        int x = *xoff % 2;
        int y = *yoff % 2;
        
        if (x == 0 && y == 0) return "RGGB";
        if (x == 1 && y == 0) return "GRBG";
        if (x == 0 && y == 1) return "GBRG";
        if (x == 1 && y == 1) return "BGGR";
    }
    
    return "GBRG";  // Default
}

BayerPattern fits_get_bayer_pattern(const FitsHeader& header) {
    return parse_bayer_pattern(fits_get_bayerpat(header));
}

} // namespace tile_compile::io
```

---

## 2.2 Konfiguration (config/configuration.hpp)

### Header

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include <yaml-cpp/yaml.h>
#include <optional>

namespace tile_compile::config {

// Gewichtungs-Konfiguration
struct WeightsConfig {
    float background = 0.4f;
    float noise = 0.3f;
    float gradient = 0.3f;
    
    void validate() const;
};

// Globale Metriken-Konfiguration
struct GlobalMetricsConfig {
    WeightsConfig weights;
    std::pair<float, float> clamp = {-3.0f, 3.0f};
};

// Lokale Metriken-Konfiguration
struct LocalMetricsConfig {
    struct StarMode {
        float fwhm_weight = 0.4f;
        float roundness_weight = 0.3f;
        float contrast_weight = 0.3f;
    } star_mode;
    
    struct StructureMode {
        float background_weight = 0.5f;
        float metric_weight = 0.5f;
    } structure_mode;
    
    std::pair<float, float> clamp = {-3.0f, 3.0f};
};

// Tile-Konfiguration
struct TileConfig {
    float size_factor = 8.0f;
    int min_size = 32;
    int max_divisor = 4;
    float overlap_fraction = 0.25f;
};

// Synthetic-Konfiguration
struct SyntheticConfig {
    int frames_min = 15;
    int frames_max = 30;
    std::string weighting = "global";  // "global" | "tile_weighted"
    
    struct ClusteringConfig {
        std::pair<int, int> cluster_count_range = {5, 30};
        bool use_silhouette = false;
    } clustering;
};

// Registrierungs-Konfiguration
struct RegistrationConfig {
    std::string method = "ecc";  // "ecc" | "phase_correlation"
    bool allow_rotation = true;
    float rotation_range_deg = 5.0f;
    int rotation_steps = 11;
    std::string output_dir = "registered";
};

// Kalibrierungs-Konfiguration
struct CalibrationConfig {
    std::optional<fs::path> bias_dir;
    std::optional<fs::path> dark_dir;
    std::optional<fs::path> flat_dir;
    bool enabled = true;
};

// Sigma-Clipping-Konfiguration
struct SigmaClipConfig {
    float sigma_low = 3.0f;
    float sigma_high = 3.0f;
    int max_iters = 3;
    float min_fraction = 0.5f;
};

// Annahmen-Konfiguration (Methodik v3 §1.2)
struct AssumptionsConfig {
    int frames_min = 30;
    int frames_reduced_threshold = 100;
    int frames_optimal = 300;
    float registration_residual_warn_px = 0.5f;
    float registration_residual_max_px = 2.0f;
    float elongation_warn = 1.2f;
    float elongation_max = 1.5f;
    std::pair<int, int> reduced_mode_cluster_range = {5, 15};
};

// Haupt-Konfiguration
struct Config {
    GlobalMetricsConfig global_metrics;
    LocalMetricsConfig local_metrics;
    TileConfig tile;
    SyntheticConfig synthetic;
    RegistrationConfig registration;
    CalibrationConfig calibration;
    SigmaClipConfig sigma_clip;
    AssumptionsConfig assumptions;
    
    // Laden/Speichern
    static Config load(const fs::path& path);
    static Config from_yaml(const YAML::Node& node);
    void save(const fs::path& path) const;
    YAML::Node to_yaml() const;
    
    // Validierung
    void validate() const;
};

} // namespace tile_compile::config
```

### Implementierung (Auszug)

```cpp
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/error.hpp"
#include <fstream>

namespace tile_compile::config {

void WeightsConfig::validate() const {
    float sum = background + noise + gradient;
    if (std::abs(sum - 1.0f) > 1e-6f) {
        throw ConfigError("Weights must sum to 1.0, got " + std::to_string(sum));
    }
}

Config Config::load(const fs::path& path) {
    if (!fs::exists(path)) {
        throw ConfigError("Config file not found: " + path.string());
    }
    
    YAML::Node node = YAML::LoadFile(path.string());
    return from_yaml(node);
}

Config Config::from_yaml(const YAML::Node& node) {
    Config cfg;
    
    // Global Metrics
    if (node["global_metrics"]) {
        auto gm = node["global_metrics"];
        if (gm["weights"]) {
            auto w = gm["weights"];
            cfg.global_metrics.weights.background = w["background"].as<float>(0.4f);
            cfg.global_metrics.weights.noise = w["noise"].as<float>(0.3f);
            cfg.global_metrics.weights.gradient = w["gradient"].as<float>(0.3f);
        }
        if (gm["clamp"]) {
            auto c = gm["clamp"];
            cfg.global_metrics.clamp = {c[0].as<float>(-3.0f), c[1].as<float>(3.0f)};
        }
    }
    
    // Tile
    if (node["tile"]) {
        auto t = node["tile"];
        cfg.tile.size_factor = t["size_factor"].as<float>(8.0f);
        cfg.tile.min_size = t["min_size"].as<int>(32);
        cfg.tile.max_divisor = t["max_divisor"].as<int>(4);
        cfg.tile.overlap_fraction = t["overlap_fraction"].as<float>(0.25f);
    }
    
    // Synthetic
    if (node["synthetic"]) {
        auto s = node["synthetic"];
        cfg.synthetic.frames_min = s["frames_min"].as<int>(15);
        cfg.synthetic.frames_max = s["frames_max"].as<int>(30);
        cfg.synthetic.weighting = s["weighting"].as<std::string>("global");
        
        if (s["clustering"]) {
            auto c = s["clustering"];
            if (c["cluster_count_range"]) {
                auto r = c["cluster_count_range"];
                cfg.synthetic.clustering.cluster_count_range = {
                    r[0].as<int>(5), r[1].as<int>(30)
                };
            }
            cfg.synthetic.clustering.use_silhouette = c["use_silhouette"].as<bool>(false);
        }
    }
    
    // Registration
    if (node["registration"]) {
        auto r = node["registration"];
        cfg.registration.method = r["method"].as<std::string>("ecc");
        cfg.registration.allow_rotation = r["allow_rotation"].as<bool>(true);
        cfg.registration.rotation_range_deg = r["rotation_range_deg"].as<float>(5.0f);
        cfg.registration.rotation_steps = r["rotation_steps"].as<int>(11);
        cfg.registration.output_dir = r["output_dir"].as<std::string>("registered");
    }
    
    // Calibration
    if (node["calibration"]) {
        auto c = node["calibration"];
        cfg.calibration.enabled = c["enabled"].as<bool>(true);
        if (c["bias_dir"]) cfg.calibration.bias_dir = c["bias_dir"].as<std::string>();
        if (c["dark_dir"]) cfg.calibration.dark_dir = c["dark_dir"].as<std::string>();
        if (c["flat_dir"]) cfg.calibration.flat_dir = c["flat_dir"].as<std::string>();
    }
    
    // Assumptions
    if (node["assumptions"]) {
        auto a = node["assumptions"];
        cfg.assumptions.frames_min = a["frames_min"].as<int>(30);
        cfg.assumptions.frames_reduced_threshold = a["frames_reduced_threshold"].as<int>(100);
        cfg.assumptions.frames_optimal = a["frames_optimal"].as<int>(300);
        cfg.assumptions.registration_residual_warn_px = 
            a["registration_residual_warn_px"].as<float>(0.5f);
        cfg.assumptions.registration_residual_max_px = 
            a["registration_residual_max_px"].as<float>(2.0f);
    }
    
    return cfg;
}

void Config::validate() const {
    global_metrics.weights.validate();
    
    if (tile.overlap_fraction < 0.0f || tile.overlap_fraction > 0.5f) {
        throw ConfigError("tile.overlap_fraction must be in [0, 0.5]");
    }
    
    if (synthetic.frames_max < synthetic.frames_min) {
        throw ConfigError("synthetic.frames_max must be >= synthetic.frames_min");
    }
    
    if (assumptions.frames_min >= assumptions.frames_reduced_threshold) {
        throw ConfigError("assumptions.frames_min must be < frames_reduced_threshold");
    }
}

} // namespace tile_compile::config
```

---

## 2.3 Konfigurationsvalidierung (config/validate.hpp)

```cpp
#pragma once

#include <string>
#include <vector>
#include <optional>

namespace tile_compile::config {

struct ValidationIssue {
    std::string severity;  // "error" | "warning"
    std::string code;
    std::string path;
    std::string message;
};

struct ValidationResult {
    bool valid;
    std::vector<ValidationIssue> errors;
    std::vector<ValidationIssue> warnings;
};

ValidationResult validate_config_yaml_text(
    const std::string& yaml_text,
    const std::optional<fs::path>& schema_path = std::nullopt
);

ValidationResult validate_config(const Config& cfg);

} // namespace tile_compile::config
```

---

## 2.4 Schema-Validierung

Für JSON-Schema-Validierung kann `nlohmann/json-schema-validator` verwendet werden:

```bash
vcpkg install nlohmann-json-schema-validator
```

```cpp
#include <nlohmann/json-schema.hpp>

ValidationResult validate_config_yaml_text(const std::string& yaml_text,
                                            const std::optional<fs::path>& schema_path) {
    ValidationResult result;
    result.valid = true;
    
    // YAML parsen
    YAML::Node node;
    try {
        node = YAML::Load(yaml_text);
    } catch (const YAML::Exception& e) {
        result.valid = false;
        result.errors.push_back({
            "error", "yaml_parse_error", "$", e.what()
        });
        return result;
    }
    
    // In JSON konvertieren für Schema-Validierung
    nlohmann::json json_cfg = yaml_to_json(node);
    
    // Schema laden und validieren
    if (schema_path) {
        nlohmann::json schema = nlohmann::json::parse(
            std::ifstream(*schema_path)
        );
        
        nlohmann::json_schema::json_validator validator;
        validator.set_root_schema(schema);
        
        try {
            validator.validate(json_cfg);
        } catch (const std::exception& e) {
            result.valid = false;
            result.errors.push_back({
                "error", "schema_validation_error", "$", e.what()
            });
        }
    }
    
    // Zusätzliche semantische Prüfungen
    // ... (wie in Python validate.py)
    
    return result;
}
```

---

## Checkliste Phase 2

- [ ] fits_utils.hpp/cpp implementiert
- [ ] FitsHeader-Klasse implementiert
- [ ] read_fits_float() funktioniert
- [ ] write_fits_float() funktioniert
- [ ] Bayer-Pattern-Erkennung funktioniert
- [ ] configuration.hpp/cpp implementiert
- [ ] Config::load() funktioniert
- [ ] Config::validate() funktioniert
- [ ] validate.hpp/cpp implementiert
- [ ] Unit-Tests für I/O geschrieben
- [ ] Unit-Tests für Config geschrieben
- [ ] Integration mit Phase 1 getestet
