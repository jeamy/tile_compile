# Phase 3: Bildverarbeitung

## Ziel

Portierung der Bildverarbeitungsfunktionen aus `runner/image_processing.py`.

**Geschätzte Dauer**: 2-3 Wochen

---

## 3.1 Übersicht der zu portierenden Funktionen

| Python-Funktion | C++ Funktion | Priorität |
|-----------------|--------------|-----------|
| `split_cfa_channels()` | `split_cfa_channels()` | Hoch |
| `reassemble_cfa_mosaic()` | `reassemble_cfa_mosaic()` | Hoch |
| `demosaic_cfa()` | `demosaic_cfa()` | Hoch |
| `normalize_frame()` | `normalize_frame()` | Hoch |
| `cosmetic_correction()` | `cosmetic_correction()` | Hoch |
| `warp_cfa_mosaic_via_subplanes()` | `warp_cfa_mosaic_via_subplanes()` | Hoch |
| `cfa_downsample_sum2x2()` | `cfa_downsample_sum2x2()` | Mittel |
| `split_rgb_frame()` | `split_rgb_frame()` | Mittel |
| `compute_frame_medians()` | `compute_frame_medians()` | Mittel |

---

## 3.2 Header (image/processing.hpp)

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include <map>
#include <optional>
#include <string>

namespace tile_compile::image {

// Bayer-Pattern-Positionen
struct BayerPositions {
    std::pair<int, int> R;
    std::pair<int, int> G1;
    std::pair<int, int> G2;
    std::pair<int, int> B;
};

BayerPositions get_bayer_positions(BayerPattern pattern);

// CFA/Bayer-Verarbeitung
std::map<std::string, Matrix2Df> split_cfa_channels(
    const Matrix2Df& mosaic,
    BayerPattern bayer_pattern
);

Matrix2Df reassemble_cfa_mosaic(
    const Matrix2Df& r_plane,
    const Matrix2Df& g_plane,
    const Matrix2Df& b_plane,
    BayerPattern bayer_pattern
);

// Demosaicing via OpenCV
Tensor3Df demosaic_cfa(
    const Matrix2Df& mosaic,
    BayerPattern bayer_pattern
);

// RGB-Frame-Splitting
std::map<std::string, Matrix2Df> split_rgb_frame(const Tensor3Df& data);
std::map<std::string, Matrix2Df> split_rgb_frame(const Matrix2Df& data);  // Mono

// Normalisierung (Methodik v3 §3.1)
enum class NormalizationMode {
    BACKGROUND,  // Divisive: I' = I / B_f (Methodik v3)
    ADDITIVE,    // Legacy: I' = I - (B_f - B_target)
    NONE
};

Matrix2Df normalize_frame(
    const Matrix2Df& frame,
    float frame_median,
    float target_median,
    NormalizationMode mode
);

// Hotpixel-Korrektur
Matrix2Df cosmetic_correction(
    const Matrix2Df& data,
    float sigma_threshold = 8.0f,
    bool hot_only = true
);

// CFA Warping (für Registrierung)
Matrix2Df warp_cfa_mosaic_via_subplanes(
    const Matrix2Df& mosaic,
    const AffineWarp& warp,
    std::optional<std::pair<int, int>> out_shape = std::nullopt,
    const std::string& border_mode = "replicate",
    float border_value = 0.0f,
    const std::string& interpolation = "linear"
);

// Hilfsfunktionen
Matrix2Df cfa_downsample_sum2x2(const Matrix2Df& mosaic);

std::pair<std::vector<float>, float> compute_frame_medians(
    const std::vector<Matrix2Df>& frames
);

// Konvertierung
Matrix2Df to_uint8(const Matrix2Df& img);

} // namespace tile_compile::image
```

---

## 3.3 Implementierung

### split_cfa_channels

```cpp
#include "tile_compile/image/processing.hpp"
#include <opencv2/opencv.hpp>

namespace tile_compile::image {

BayerPositions get_bayer_positions(BayerPattern pattern) {
    switch (pattern) {
        case BayerPattern::RGGB:
            return {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        case BayerPattern::BGGR:
            return {{1, 1}, {0, 1}, {1, 0}, {0, 0}};
        case BayerPattern::GBRG:
            return {{1, 0}, {0, 0}, {1, 1}, {0, 1}};
        case BayerPattern::GRBG:
            return {{0, 1}, {0, 0}, {1, 1}, {1, 0}};
        default:
            return get_bayer_positions(BayerPattern::GBRG);
    }
}

std::map<std::string, Matrix2Df> split_cfa_channels(
    const Matrix2Df& mosaic,
    BayerPattern bayer_pattern
) {
    auto pos = get_bayer_positions(bayer_pattern);
    
    // Auf gerade Dimensionen zuschneiden
    int h = mosaic.rows() - (mosaic.rows() % 2);
    int w = mosaic.cols() - (mosaic.cols() % 2);
    
    int h2 = h / 2;
    int w2 = w / 2;
    
    Matrix2Df r_plane(h2, w2);
    Matrix2Df g1_plane(h2, w2);
    Matrix2Df g2_plane(h2, w2);
    Matrix2Df b_plane(h2, w2);
    
    // Subsampling
    for (int y = 0; y < h2; ++y) {
        for (int x = 0; x < w2; ++x) {
            r_plane(y, x) = mosaic(2*y + pos.R.first, 2*x + pos.R.second);
            g1_plane(y, x) = mosaic(2*y + pos.G1.first, 2*x + pos.G1.second);
            g2_plane(y, x) = mosaic(2*y + pos.G2.first, 2*x + pos.G2.second);
            b_plane(y, x) = mosaic(2*y + pos.B.first, 2*x + pos.B.second);
        }
    }
    
    // G = Mittelwert von G1 und G2
    Matrix2Df g_plane = (g1_plane + g2_plane) * 0.5f;
    
    return {
        {"R", r_plane},
        {"G", g_plane},
        {"B", b_plane}
    };
}
```

### reassemble_cfa_mosaic

```cpp
Matrix2Df reassemble_cfa_mosaic(
    const Matrix2Df& r_plane,
    const Matrix2Df& g_plane,
    const Matrix2Df& b_plane,
    BayerPattern bayer_pattern
) {
    auto pos = get_bayer_positions(bayer_pattern);
    
    int h2 = r_plane.rows();
    int w2 = r_plane.cols();
    int h = h2 * 2;
    int w = w2 * 2;
    
    Matrix2Df mosaic = Matrix2Df::Zero(h, w);
    
    for (int y = 0; y < h2; ++y) {
        for (int x = 0; x < w2; ++x) {
            mosaic(2*y + pos.R.first, 2*x + pos.R.second) = r_plane(y, x);
            mosaic(2*y + pos.B.first, 2*x + pos.B.second) = b_plane(y, x);
            mosaic(2*y + pos.G1.first, 2*x + pos.G1.second) = g_plane(y, x);
            mosaic(2*y + pos.G2.first, 2*x + pos.G2.second) = g_plane(y, x);
        }
    }
    
    return mosaic;
}
```

### demosaic_cfa

```cpp
Tensor3Df demosaic_cfa(
    const Matrix2Df& mosaic,
    BayerPattern bayer_pattern
) {
    // Auf gerade Dimensionen zuschneiden
    int h = mosaic.rows() - (mosaic.rows() % 2);
    int w = mosaic.cols() - (mosaic.cols() % 2);
    
    // OpenCV Bayer-Code bestimmen
    int cv_code;
    switch (bayer_pattern) {
        case BayerPattern::RGGB: cv_code = cv::COLOR_BayerRG2RGB; break;
        case BayerPattern::BGGR: cv_code = cv::COLOR_BayerBG2RGB; break;
        case BayerPattern::GBRG: cv_code = cv::COLOR_BayerGB2RGB; break;
        case BayerPattern::GRBG: cv_code = cv::COLOR_BayerGR2RGB; break;
        default: cv_code = cv::COLOR_BayerGB2RGB; break;
    }
    
    // Normalisieren auf 0-65535 für uint16
    float mosaic_min = mosaic.minCoeff();
    float mosaic_max = mosaic.maxCoeff();
    float mosaic_range = mosaic_max - mosaic_min;
    
    // Eigen zu OpenCV konvertieren
    cv::Mat mosaic_cv(h, w, CV_32F);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            mosaic_cv.at<float>(y, x) = mosaic(y, x);
        }
    }
    
    // Auf uint16 normalisieren
    cv::Mat mosaic_u16;
    if (mosaic_range > 0) {
        cv::Mat normalized = (mosaic_cv - mosaic_min) / mosaic_range * 65535.0f;
        normalized.convertTo(mosaic_u16, CV_16U);
    } else {
        mosaic_u16 = cv::Mat::zeros(h, w, CV_16U);
    }
    
    // Demosaicing
    cv::Mat rgb_cv;
    cv::cvtColor(mosaic_u16, rgb_cv, cv_code);
    
    // Zurück zu float und Original-Range
    cv::Mat rgb_float;
    rgb_cv.convertTo(rgb_float, CV_32F);
    if (mosaic_range > 0) {
        rgb_float = rgb_float / 65535.0f * mosaic_range + mosaic_min;
    }
    
    // OpenCV zu Eigen Tensor konvertieren (3 x H x W)
    Tensor3Df result(3, h, w);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cv::Vec3f pixel = rgb_float.at<cv::Vec3f>(y, x);
            result(0, y, x) = pixel[2];  // R (OpenCV ist BGR)
            result(1, y, x) = pixel[1];  // G
            result(2, y, x) = pixel[0];  // B
        }
    }
    
    return result;
}
```

### cosmetic_correction

```cpp
namespace {

Matrix2Df local_median_3x3(const Matrix2Df& data) {
    int h = data.rows();
    int w = data.cols();
    
    Matrix2Df result(h, w);
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::vector<float> neighbors;
            neighbors.reserve(8);
            
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    
                    int ny = std::clamp(y + dy, 0, h - 1);
                    int nx = std::clamp(x + dx, 0, w - 1);
                    neighbors.push_back(data(ny, nx));
                }
            }
            
            std::nth_element(neighbors.begin(), 
                            neighbors.begin() + neighbors.size() / 2,
                            neighbors.end());
            result(y, x) = neighbors[neighbors.size() / 2];
        }
    }
    
    return result;
}

} // anonymous namespace

Matrix2Df cosmetic_correction(
    const Matrix2Df& data,
    float sigma_threshold,
    bool hot_only
) {
    Matrix2Df result = data;
    
    // Robuste Statistiken berechnen
    float med = compute_median(data);
    float mad = compute_mad(data);
    float robust_sigma = 1.4826f * mad;
    
    if (robust_sigma <= 0) {
        return result;
    }
    
    // Schwellwerte
    float threshold_hi = med + sigma_threshold * robust_sigma;
    float threshold_lo = med - sigma_threshold * robust_sigma;
    
    // Hotpixel-Maske erstellen
    Matrix2Db hot_mask = (data.array() > threshold_hi).matrix();
    
    if (!hot_only) {
        Matrix2Db cold_mask = (data.array() < threshold_lo).matrix();
        hot_mask = hot_mask.array() || cold_mask.array();
    }
    
    // Prüfen ob Hotpixel vorhanden
    if (!hot_mask.any()) {
        return result;
    }
    
    // Lokalen Median berechnen
    Matrix2Df local_med = local_median_3x3(data);
    
    // Hotpixel ersetzen
    for (int y = 0; y < data.rows(); ++y) {
        for (int x = 0; x < data.cols(); ++x) {
            if (hot_mask(y, x)) {
                result(y, x) = local_med(y, x);
            }
        }
    }
    
    return result;
}
```

### warp_cfa_mosaic_via_subplanes

```cpp
Matrix2Df warp_cfa_mosaic_via_subplanes(
    const Matrix2Df& mosaic,
    const AffineWarp& warp,
    std::optional<std::pair<int, int>> out_shape,
    const std::string& border_mode,
    float border_value,
    const std::string& interpolation
) {
    int h = mosaic.rows() - (mosaic.rows() % 2);
    int w = mosaic.cols() - (mosaic.cols() % 2);
    
    int out_h, out_w;
    if (out_shape) {
        out_h = out_shape->first - (out_shape->first % 2);
        out_w = out_shape->second - (out_shape->second % 2);
    } else {
        out_h = h;
        out_w = w;
    }
    
    // Subplanes extrahieren
    int h2 = h / 2;
    int w2 = w / 2;
    
    Matrix2Df a(h2, w2), b(h2, w2), c(h2, w2), d(h2, w2);
    for (int y = 0; y < h2; ++y) {
        for (int x = 0; x < w2; ++x) {
            a(y, x) = mosaic(2*y, 2*x);
            b(y, x) = mosaic(2*y, 2*x + 1);
            c(y, x) = mosaic(2*y + 1, 2*x);
            d(y, x) = mosaic(2*y + 1, 2*x + 1);
        }
    }
    
    // Warp-Matrizen für jede Subplane anpassen
    Eigen::Vector2f delta_a(-0.25f, -0.25f);
    Eigen::Vector2f delta_b(0.25f, -0.25f);
    Eigen::Vector2f delta_c(-0.25f, 0.25f);
    Eigen::Vector2f delta_d(0.25f, 0.25f);
    
    Eigen::Matrix2f A = warp.block<2, 2>(0, 0);
    Eigen::Vector2f t = warp.col(2);
    
    auto make_warp = [&](const Eigen::Vector2f& delta) -> AffineWarp {
        AffineWarp w;
        w.block<2, 2>(0, 0) = A;
        w.col(2) = t + A * delta - delta;
        return w;
    };
    
    AffineWarp warp_a = make_warp(delta_a);
    AffineWarp warp_b = make_warp(delta_b);
    AffineWarp warp_c = make_warp(delta_c);
    AffineWarp warp_d = make_warp(delta_d);
    
    // OpenCV Interpolation und Border-Mode
    int interp_cv = (interpolation == "nearest") ? cv::INTER_NEAREST : cv::INTER_LINEAR;
    int flags = interp_cv | cv::WARP_INVERSE_MAP;
    
    int border_cv;
    if (border_mode == "constant" || border_mode == "black") {
        border_cv = cv::BORDER_CONSTANT;
    } else if (border_mode == "reflect") {
        border_cv = cv::BORDER_REFLECT_101;
    } else {
        border_cv = cv::BORDER_REPLICATE;
    }
    
    // Warping durchführen
    auto warp_subplane = [&](const Matrix2Df& plane, const AffineWarp& w) -> Matrix2Df {
        cv::Mat src = eigen_to_cv(plane);
        cv::Mat warp_mat = eigen_to_cv_warp(w);
        cv::Mat dst;
        
        cv::warpAffine(src, dst, warp_mat, 
                       cv::Size(out_w / 2, out_h / 2),
                       flags, border_cv, border_value);
        
        return cv_to_eigen(dst);
    };
    
    Matrix2Df a_w = warp_subplane(a, warp_a);
    Matrix2Df b_w = warp_subplane(b, warp_b);
    Matrix2Df c_w = warp_subplane(c, warp_c);
    Matrix2Df d_w = warp_subplane(d, warp_d);
    
    // Reassemblieren
    Matrix2Df out = Matrix2Df::Zero(out_h, out_w);
    int out_h2 = out_h / 2;
    int out_w2 = out_w / 2;
    
    for (int y = 0; y < out_h2; ++y) {
        for (int x = 0; x < out_w2; ++x) {
            out(2*y, 2*x) = a_w(y, x);
            out(2*y, 2*x + 1) = b_w(y, x);
            out(2*y + 1, 2*x) = c_w(y, x);
            out(2*y + 1, 2*x + 1) = d_w(y, x);
        }
    }
    
    return out;
}
```

### normalize_frame

```cpp
Matrix2Df normalize_frame(
    const Matrix2Df& frame,
    float frame_median,
    float target_median,
    NormalizationMode mode
) {
    switch (mode) {
        case NormalizationMode::BACKGROUND:
            // Methodik v3 §3.1: I' = I / B_f (divisive)
            if (frame_median > 1e-10f) {
                if (target_median > 1e-10f) {
                    return frame * (target_median / frame_median);
                }
                return frame / frame_median;
            }
            return frame;
            
        case NormalizationMode::ADDITIVE:
            // Legacy: I' = I - (B_f - B_target)
            return frame.array() - (frame_median - target_median);
            
        case NormalizationMode::NONE:
        default:
            return frame;
    }
}
```

---

## 3.4 Hilfsfunktionen für Eigen ↔ OpenCV

```cpp
// In einem separaten Header: image/cv_utils.hpp

#pragma once

#include "tile_compile/core/types.hpp"
#include <opencv2/opencv.hpp>

namespace tile_compile::image {

inline cv::Mat eigen_to_cv(const Matrix2Df& m) {
    cv::Mat result(m.rows(), m.cols(), CV_32F);
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            result.at<float>(y, x) = m(y, x);
        }
    }
    return result;
}

inline Matrix2Df cv_to_eigen(const cv::Mat& m) {
    Matrix2Df result(m.rows, m.cols);
    for (int y = 0; y < m.rows; ++y) {
        for (int x = 0; x < m.cols; ++x) {
            result(y, x) = m.at<float>(y, x);
        }
    }
    return result;
}

inline cv::Mat eigen_to_cv_warp(const AffineWarp& w) {
    cv::Mat result(2, 3, CV_32F);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            result.at<float>(i, j) = w(i, j);
        }
    }
    return result;
}

inline AffineWarp cv_to_eigen_warp(const cv::Mat& m) {
    AffineWarp result;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            result(i, j) = m.at<float>(i, j);
        }
    }
    return result;
}

} // namespace tile_compile::image
```

---

## 3.5 Tests

```cpp
// tests/image/test_processing.cpp

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tile_compile/image/processing.hpp"

using namespace tile_compile::image;
using Catch::Matchers::WithinAbs;

TEST_CASE("split_cfa_channels", "[image]") {
    // 4x4 Test-Mosaik (RGGB)
    Matrix2Df mosaic(4, 4);
    mosaic << 1, 2, 1, 2,
              3, 4, 3, 4,
              1, 2, 1, 2,
              3, 4, 3, 4;
    
    auto channels = split_cfa_channels(mosaic, BayerPattern::RGGB);
    
    REQUIRE(channels.count("R") == 1);
    REQUIRE(channels.count("G") == 1);
    REQUIRE(channels.count("B") == 1);
    
    REQUIRE(channels["R"].rows() == 2);
    REQUIRE(channels["R"].cols() == 2);
    
    // R ist an Position (0,0) in RGGB
    REQUIRE_THAT(channels["R"](0, 0), WithinAbs(1.0f, 1e-6f));
}

TEST_CASE("reassemble_cfa_mosaic", "[image]") {
    Matrix2Df r(2, 2), g(2, 2), b(2, 2);
    r << 1, 1, 1, 1;
    g << 2, 2, 2, 2;
    b << 3, 3, 3, 3;
    
    auto mosaic = reassemble_cfa_mosaic(r, g, b, BayerPattern::RGGB);
    
    REQUIRE(mosaic.rows() == 4);
    REQUIRE(mosaic.cols() == 4);
    
    // RGGB: R at (0,0), G at (0,1) and (1,0), B at (1,1)
    REQUIRE_THAT(mosaic(0, 0), WithinAbs(1.0f, 1e-6f));  // R
    REQUIRE_THAT(mosaic(0, 1), WithinAbs(2.0f, 1e-6f));  // G
    REQUIRE_THAT(mosaic(1, 0), WithinAbs(2.0f, 1e-6f));  // G
    REQUIRE_THAT(mosaic(1, 1), WithinAbs(3.0f, 1e-6f));  // B
}

TEST_CASE("cosmetic_correction", "[image]") {
    Matrix2Df data = Matrix2Df::Constant(10, 10, 100.0f);
    data(5, 5) = 10000.0f;  // Hotpixel
    
    auto corrected = cosmetic_correction(data, 5.0f, true);
    
    // Hotpixel sollte ersetzt sein
    REQUIRE(corrected(5, 5) < 1000.0f);
    
    // Andere Pixel unverändert
    REQUIRE_THAT(corrected(0, 0), WithinAbs(100.0f, 1e-6f));
}

TEST_CASE("normalize_frame", "[image]") {
    Matrix2Df frame(2, 2);
    frame << 100, 200, 300, 400;
    
    SECTION("background mode") {
        auto normalized = normalize_frame(frame, 250.0f, 1.0f, 
                                          NormalizationMode::BACKGROUND);
        // I' = I * (target / frame_median) = I * (1 / 250)
        REQUIRE_THAT(normalized(0, 0), WithinAbs(0.4f, 1e-6f));
    }
    
    SECTION("additive mode") {
        auto normalized = normalize_frame(frame, 250.0f, 200.0f,
                                          NormalizationMode::ADDITIVE);
        // I' = I - (250 - 200) = I - 50
        REQUIRE_THAT(normalized(0, 0), WithinAbs(50.0f, 1e-6f));
    }
}
```

---

## Checkliste Phase 3

- [ ] processing.hpp Header erstellt
- [ ] cv_utils.hpp für Eigen↔OpenCV Konvertierung
- [ ] `split_cfa_channels()` implementiert
- [ ] `reassemble_cfa_mosaic()` implementiert
- [ ] `demosaic_cfa()` implementiert
- [ ] `normalize_frame()` implementiert
- [ ] `cosmetic_correction()` implementiert
- [ ] `warp_cfa_mosaic_via_subplanes()` implementiert
- [ ] `cfa_downsample_sum2x2()` implementiert
- [ ] `split_rgb_frame()` implementiert
- [ ] `compute_frame_medians()` implementiert
- [ ] Unit-Tests geschrieben
- [ ] Integration mit Phase 1+2 getestet
- [ ] Performance-Vergleich mit Python-Version
