# Phase 4: Registrierung

## Ziel

Portierung der Registrierungsfunktionen aus `runner/opencv_registration.py`.

**Geschätzte Dauer**: 2-3 Wochen

---

## 4.1 Übersicht

| Python-Funktion | C++ Funktion | Beschreibung |
|-----------------|--------------|--------------|
| `opencv_prepare_ecc_image()` | `prepare_ecc_image()` | Bildvorbereitung für ECC |
| `opencv_count_stars()` | `count_stars()` | Stern-Zählung |
| `opencv_ecc_warp()` | `ecc_warp()` | ECC-Warp-Berechnung |
| `opencv_phasecorr_translation()` | `phase_correlation()` | Phase-Korrelation |
| `opencv_alignment_score()` | `alignment_score()` | Alignment-Bewertung |
| `opencv_best_translation_init()` | `best_translation_init()` | Beste initiale Warp |

---

## 4.2 Header (registration/opencv_registration.hpp)

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include <opencv2/opencv.hpp>
#include <optional>
#include <tuple>

namespace tile_compile::registration {

// ECC-Bildvorbereitung
// Normalisiert und filtert Bild für optimale ECC-Registrierung
cv::Mat prepare_ecc_image(const cv::Mat& img);
Matrix2Df prepare_ecc_image(const Matrix2Df& img);

// Stern-Zählung mit goodFeaturesToTrack
int count_stars(const cv::Mat& img01);
int count_stars(const Matrix2Df& img01);

// ECC-Warp-Berechnung
struct ECCResult {
    AffineWarp warp;
    float correlation_coefficient;
    bool success;
    std::string error_message;
};

ECCResult ecc_warp(
    const cv::Mat& moving01,
    const cv::Mat& ref01,
    bool allow_rotation,
    const cv::Mat& init_warp
);

ECCResult ecc_warp(
    const Matrix2Df& moving01,
    const Matrix2Df& ref01,
    bool allow_rotation,
    const AffineWarp& init_warp
);

// Phase-Korrelation für initiale Translation
std::pair<float, float> phase_correlation(
    const cv::Mat& moving01,
    const cv::Mat& ref01
);

std::pair<float, float> phase_correlation(
    const Matrix2Df& moving01,
    const Matrix2Df& ref01
);

// Alignment-Score (Korrelationskoeffizient)
float alignment_score(
    const cv::Mat& moving01,
    const cv::Mat& ref01
);

float alignment_score(
    const Matrix2Df& moving01,
    const Matrix2Df& ref01
);

// Beste initiale Warp finden (Translation + optionaler Rotations-Sweep)
struct TranslationInitConfig {
    bool rotation_sweep = true;
    float rotation_range_deg = 5.0f;
    int rotation_steps = 11;
};

AffineWarp best_translation_init(
    const cv::Mat& moving01,
    const cv::Mat& ref01,
    const TranslationInitConfig& config = {}
);

AffineWarp best_translation_init(
    const Matrix2Df& moving01,
    const Matrix2Df& ref01,
    const TranslationInitConfig& config = {}
);

// Hilfsfunktionen
AffineWarp identity_warp();
AffineWarp translation_warp(float dx, float dy);
AffineWarp rotation_warp(float angle_rad, float cx, float cy);

// Warp anwenden
Matrix2Df apply_warp(
    const Matrix2Df& img,
    const AffineWarp& warp,
    std::optional<std::pair<int, int>> out_size = std::nullopt,
    int interpolation = cv::INTER_LINEAR,
    int border_mode = cv::BORDER_REPLICATE,
    float border_value = 0.0f
);

} // namespace tile_compile::registration
```

---

## 4.3 Implementierung

### prepare_ecc_image

```cpp
#include "tile_compile/registration/opencv_registration.hpp"
#include "tile_compile/image/cv_utils.hpp"

namespace tile_compile::registration {

cv::Mat prepare_ecc_image(const cv::Mat& img) {
    cv::Mat f;
    img.convertTo(f, CV_32F);
    
    // Median subtrahieren
    cv::Scalar med = cv::mean(f);  // Approximation, für echten Median siehe unten
    f = f - med[0];
    
    // Standardabweichung normalisieren
    cv::Scalar mean, stddev;
    cv::meanStdDev(f, mean, stddev);
    if (stddev[0] > 0) {
        f = f / stddev[0];
    }
    
    // Hintergrund-Subtraktion (Gaussian Blur)
    cv::Mat bg;
    cv::GaussianBlur(f, bg, cv::Size(0, 0), 12.0);
    f = f - bg;
    
    // Leichtes Glätten
    cv::GaussianBlur(f, f, cv::Size(0, 0), 1.0);
    
    // Auf [0, 1] normalisieren
    cv::normalize(f, f, 0.0, 1.0, cv::NORM_MINMAX);
    
    return f;
}

Matrix2Df prepare_ecc_image(const Matrix2Df& img) {
    cv::Mat cv_img = image::eigen_to_cv(img);
    cv::Mat result = prepare_ecc_image(cv_img);
    return image::cv_to_eigen(result);
}
```

### count_stars

```cpp
int count_stars(const cv::Mat& img01) {
    std::vector<cv::Point2f> corners;
    
    cv::goodFeaturesToTrack(
        img01,
        corners,
        1200,       // maxCorners
        0.01,       // qualityLevel
        5,          // minDistance
        cv::noArray(),
        7           // blockSize
    );
    
    return static_cast<int>(corners.size());
}

int count_stars(const Matrix2Df& img01) {
    cv::Mat cv_img = image::eigen_to_cv(img01);
    return count_stars(cv_img);
}
```

### ecc_warp

```cpp
ECCResult ecc_warp(
    const cv::Mat& moving01,
    const cv::Mat& ref01,
    bool allow_rotation,
    const cv::Mat& init_warp
) {
    ECCResult result;
    result.success = false;
    
    int motion_type = allow_rotation ? cv::MOTION_EUCLIDEAN : cv::MOTION_TRANSLATION;
    
    cv::TermCriteria criteria(
        cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
        200,    // maxCount
        1e-6    // epsilon
    );
    
    cv::Mat warp = init_warp.clone();
    
    try {
        double cc = cv::findTransformECC(
            ref01,
            moving01,
            warp,
            motion_type,
            criteria
        );
        
        result.warp = image::cv_to_eigen_warp(warp);
        result.correlation_coefficient = static_cast<float>(cc);
        result.success = true;
    } catch (const cv::Exception& e) {
        result.error_message = e.what();
        result.warp = image::cv_to_eigen_warp(init_warp);
        result.correlation_coefficient = 0.0f;
    }
    
    return result;
}

ECCResult ecc_warp(
    const Matrix2Df& moving01,
    const Matrix2Df& ref01,
    bool allow_rotation,
    const AffineWarp& init_warp
) {
    cv::Mat cv_moving = image::eigen_to_cv(moving01);
    cv::Mat cv_ref = image::eigen_to_cv(ref01);
    cv::Mat cv_warp = image::eigen_to_cv_warp(init_warp);
    
    return ecc_warp(cv_moving, cv_ref, allow_rotation, cv_warp);
}
```

### phase_correlation

```cpp
std::pair<float, float> phase_correlation(
    const cv::Mat& moving01,
    const cv::Mat& ref01
) {
    cv::Mat hann;
    cv::createHanningWindow(hann, ref01.size(), CV_32F);
    
    cv::Point2d shift = cv::phaseCorrelate(ref01, moving01, hann);
    
    return {static_cast<float>(shift.x), static_cast<float>(shift.y)};
}

std::pair<float, float> phase_correlation(
    const Matrix2Df& moving01,
    const Matrix2Df& ref01
) {
    cv::Mat cv_moving = image::eigen_to_cv(moving01);
    cv::Mat cv_ref = image::eigen_to_cv(ref01);
    
    return phase_correlation(cv_moving, cv_ref);
}
```

### alignment_score

```cpp
float alignment_score(
    const cv::Mat& moving01,
    const cv::Mat& ref01
) {
    cv::Scalar mean_a = cv::mean(moving01);
    cv::Scalar mean_b = cv::mean(ref01);
    
    cv::Mat da = moving01 - mean_a[0];
    cv::Mat db = ref01 - mean_b[0];
    
    double sum_da_sq = cv::sum(da.mul(da))[0];
    double sum_db_sq = cv::sum(db.mul(db))[0];
    double sum_da_db = cv::sum(da.mul(db))[0];
    
    double denom = std::sqrt(sum_da_sq * sum_db_sq);
    
    if (denom < 1e-12) {
        return 0.0f;
    }
    
    return static_cast<float>(sum_da_db / denom);
}

float alignment_score(
    const Matrix2Df& moving01,
    const Matrix2Df& ref01
) {
    float mean_a = moving01.mean();
    float mean_b = ref01.mean();
    
    Matrix2Df da = moving01.array() - mean_a;
    Matrix2Df db = ref01.array() - mean_b;
    
    float sum_da_sq = (da.array() * da.array()).sum();
    float sum_db_sq = (db.array() * db.array()).sum();
    float sum_da_db = (da.array() * db.array()).sum();
    
    float denom = std::sqrt(sum_da_sq * sum_db_sq);
    
    if (denom < 1e-12f) {
        return 0.0f;
    }
    
    return sum_da_db / denom;
}
```

### best_translation_init

```cpp
AffineWarp best_translation_init(
    const cv::Mat& moving01,
    const cv::Mat& ref01,
    const TranslationInitConfig& config
) {
    // Phase-Korrelation für initiale Translation
    auto [dx, dy] = phase_correlation(moving01, ref01);
    
    int h = ref01.rows;
    int w = ref01.cols;
    float cx = w / 2.0f;
    float cy = h / 2.0f;
    
    // Kandidaten-Translationen
    std::vector<std::pair<float, float>> translations = {
        {dx, dy},
        {0.0f, 0.0f},
        {dx * 0.5f, dy * 0.5f}
    };
    
    // Rotationswinkel
    std::vector<float> angles_deg;
    if (config.rotation_sweep) {
        float step = 2.0f * config.rotation_range_deg / (config.rotation_steps - 1);
        for (int i = 0; i < config.rotation_steps; ++i) {
            angles_deg.push_back(-config.rotation_range_deg + i * step);
        }
    } else {
        angles_deg.push_back(0.0f);
    }
    
    // Alle Kandidaten generieren
    std::vector<AffineWarp> candidates;
    for (const auto& [tx, ty] : translations) {
        for (float angle_deg : angles_deg) {
            float theta = angle_deg * M_PI / 180.0f;
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);
            
            // Rotation um Bildmitte + Translation
            AffineWarp warp;
            warp << cos_t, -sin_t, -cos_t * cx + sin_t * cy + cx + tx,
                    sin_t,  cos_t, -sin_t * cx - cos_t * cy + cy + ty;
            
            candidates.push_back(warp);
        }
    }
    
    // Besten Kandidaten finden
    AffineWarp best = candidates[0];
    float best_score = -1.0f;
    
    for (const auto& cand : candidates) {
        try {
            cv::Mat warp_cv = image::eigen_to_cv_warp(cand);
            cv::Mat warped;
            cv::warpAffine(moving01, warped, warp_cv, ref01.size(), cv::INTER_LINEAR);
            
            float score = alignment_score(warped, ref01);
            if (score > best_score) {
                best_score = score;
                best = cand;
            }
        } catch (...) {
            // Kandidat überspringen bei Fehler
        }
    }
    
    return best;
}

AffineWarp best_translation_init(
    const Matrix2Df& moving01,
    const Matrix2Df& ref01,
    const TranslationInitConfig& config
) {
    cv::Mat cv_moving = image::eigen_to_cv(moving01);
    cv::Mat cv_ref = image::eigen_to_cv(ref01);
    
    return best_translation_init(cv_moving, cv_ref, config);
}
```

### Hilfsfunktionen

```cpp
AffineWarp identity_warp() {
    AffineWarp w;
    w << 1, 0, 0,
         0, 1, 0;
    return w;
}

AffineWarp translation_warp(float dx, float dy) {
    AffineWarp w;
    w << 1, 0, dx,
         0, 1, dy;
    return w;
}

AffineWarp rotation_warp(float angle_rad, float cx, float cy) {
    float cos_t = std::cos(angle_rad);
    float sin_t = std::sin(angle_rad);
    
    AffineWarp w;
    w << cos_t, -sin_t, -cos_t * cx + sin_t * cy + cx,
         sin_t,  cos_t, -sin_t * cx - cos_t * cy + cy;
    return w;
}

Matrix2Df apply_warp(
    const Matrix2Df& img,
    const AffineWarp& warp,
    std::optional<std::pair<int, int>> out_size,
    int interpolation,
    int border_mode,
    float border_value
) {
    cv::Mat cv_img = image::eigen_to_cv(img);
    cv::Mat cv_warp = image::eigen_to_cv_warp(warp);
    
    cv::Size size;
    if (out_size) {
        size = cv::Size(out_size->second, out_size->first);
    } else {
        size = cv_img.size();
    }
    
    cv::Mat result;
    cv::warpAffine(
        cv_img, result, cv_warp, size,
        interpolation | cv::WARP_INVERSE_MAP,
        border_mode, border_value
    );
    
    return image::cv_to_eigen(result);
}

} // namespace tile_compile::registration
```

---

## 4.4 Tests

```cpp
// tests/registration/test_registration.cpp

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tile_compile/registration/opencv_registration.hpp"

using namespace tile_compile::registration;
using Catch::Matchers::WithinAbs;

TEST_CASE("identity_warp", "[registration]") {
    auto w = identity_warp();
    
    REQUIRE_THAT(w(0, 0), WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(w(0, 1), WithinAbs(0.0f, 1e-6f));
    REQUIRE_THAT(w(0, 2), WithinAbs(0.0f, 1e-6f));
    REQUIRE_THAT(w(1, 0), WithinAbs(0.0f, 1e-6f));
    REQUIRE_THAT(w(1, 1), WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(w(1, 2), WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("translation_warp", "[registration]") {
    auto w = translation_warp(10.0f, 20.0f);
    
    REQUIRE_THAT(w(0, 2), WithinAbs(10.0f, 1e-6f));
    REQUIRE_THAT(w(1, 2), WithinAbs(20.0f, 1e-6f));
}

TEST_CASE("alignment_score", "[registration]") {
    Matrix2Df a = Matrix2Df::Random(64, 64);
    
    SECTION("identical images") {
        float score = alignment_score(a, a);
        REQUIRE_THAT(score, WithinAbs(1.0f, 1e-5f));
    }
    
    SECTION("negated image") {
        Matrix2Df b = -a;
        float score = alignment_score(a, b);
        REQUIRE_THAT(score, WithinAbs(-1.0f, 1e-5f));
    }
}

TEST_CASE("phase_correlation", "[registration]") {
    // Einfaches Testbild
    Matrix2Df ref = Matrix2Df::Zero(64, 64);
    ref.block(20, 20, 10, 10).setConstant(1.0f);
    
    // Verschobenes Bild
    Matrix2Df moved = Matrix2Df::Zero(64, 64);
    moved.block(25, 30, 10, 10).setConstant(1.0f);  // +5, +10 verschoben
    
    auto [dx, dy] = phase_correlation(moved, ref);
    
    // Sollte ungefähr (10, 5) sein (x, y)
    REQUIRE(std::abs(dx - 10.0f) < 2.0f);
    REQUIRE(std::abs(dy - 5.0f) < 2.0f);
}

TEST_CASE("prepare_ecc_image", "[registration]") {
    Matrix2Df img = Matrix2Df::Random(64, 64) * 1000.0f + 500.0f;
    
    auto prepared = prepare_ecc_image(img);
    
    // Sollte auf [0, 1] normalisiert sein
    REQUIRE(prepared.minCoeff() >= -0.01f);
    REQUIRE(prepared.maxCoeff() <= 1.01f);
}
```

---

## Checkliste Phase 4

- [ ] opencv_registration.hpp Header erstellt
- [ ] `prepare_ecc_image()` implementiert
- [ ] `count_stars()` implementiert
- [ ] `ecc_warp()` implementiert
- [ ] `phase_correlation()` implementiert
- [ ] `alignment_score()` implementiert
- [ ] `best_translation_init()` implementiert
- [ ] Hilfsfunktionen implementiert
- [ ] `apply_warp()` implementiert
- [ ] Unit-Tests geschrieben
- [ ] Integration mit Phase 3 getestet
- [ ] Performance-Vergleich mit Python
