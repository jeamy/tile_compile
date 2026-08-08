#include "tile_compile/config/configuration.hpp"
#include "tile_compile/registration/global_registration.hpp"
#include "tile_compile/registration/astrometric_rescue.hpp"

#if __has_include(<catch2/catch_test_macros.hpp>)
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

namespace tile_compile::config {

// §4.1, §8.B — Tests für get_effective_chain_depth
TEST_CASE("get_effective_chain_depth auto calculation") {
    RegistrationConfig cfg;
    
    // Test: auto (0) mit verschiedenen Frame-Anzahlen
    cfg.max_blind_chain_depth = 0;  // auto
    
    // N=100 -> N/10=10, aber min=12 -> 12
    REQUIRE(get_effective_chain_depth(100, cfg) == 12);
    
    // N=200 -> N/10=20 -> 20
    REQUIRE(get_effective_chain_depth(200, cfg) == 20);
    
    // N=500 -> N/10=50 -> 50
    REQUIRE(get_effective_chain_depth(500, cfg) == 50);
    
    // N=1000 -> N/10=100, aber max=50 -> 50
    REQUIRE(get_effective_chain_depth(1000, cfg) == 50);
    
    // N=50 -> N/10=5, aber min=12 -> 12
    REQUIRE(get_effective_chain_depth(50, cfg) == 12);
}

TEST_CASE("get_effective_chain_depth manual override") {
    RegistrationConfig cfg;
    
    // Manuelle Überschreibung
    cfg.max_blind_chain_depth = 25;
    REQUIRE(get_effective_chain_depth(100, cfg) == 25);
    REQUIRE(get_effective_chain_depth(1000, cfg) == 25);
    
    cfg.max_blind_chain_depth = 15;
    REQUIRE(get_effective_chain_depth(200, cfg) == 15);
}

TEST_CASE("RegistrationConfig new parameter defaults") {
    RegistrationConfig cfg;
    
    // §4.1, §8.B — Blind-Chain Parameter
    REQUIRE(cfg.max_blind_chain_depth == 0);  // auto
    REQUIRE(cfg.blind_chain_strong_anchor_cc == Catch::Approx(0.08f));
    REQUIRE(cfg.blind_chain_drift_threshold_px == Catch::Approx(2.0f));
    
    // §4.13 — Astrometrische Rescue
    REQUIRE(cfg.use_astrometry == true);
    
    // §4.4, §8.D — Lokale Hintergrundsubtraktion
    REQUIRE(cfg.enable_local_background_subtraction == false);

    // Conservative post-registration affine refinement is opt-in.
    REQUIRE(cfg.affine_refinement_enabled == false);
}

TEST_CASE("RegistrationConfig parses affine refinement opt-in") {
    const YAML::Node node = YAML::Load(R"(
registration:
  affine_refinement_enabled: true
)");
    const Config cfg = Config::from_yaml(node);
    REQUIRE(cfg.registration.affine_refinement_enabled);
    REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("RegistrationConfig parameter validation ranges") {
    // max_blind_chain_depth: [0, 100]
    REQUIRE(RegistrationConfig{}.max_blind_chain_depth >= 0);
    
    // blind_chain_strong_anchor_cc: [0.01, 0.5]
    REQUIRE(RegistrationConfig{}.blind_chain_strong_anchor_cc >= 0.01f);
    REQUIRE(RegistrationConfig{}.blind_chain_strong_anchor_cc <= 0.5f);
    
    // blind_chain_drift_threshold_px: [0.5, 10.0]
    REQUIRE(RegistrationConfig{}.blind_chain_drift_threshold_px >= 0.5f);
    REQUIRE(RegistrationConfig{}.blind_chain_drift_threshold_px <= 10.0f);
}

} // namespace tile_compile::config

namespace tile_compile::registration {

// §4.4, §8.D — Test für lokale Hintergrundsubtraktion
TEST_CASE("detect_stars_simple with local background subtraction") {
    // Erstelle ein Bild mit Gradient (simuliert Mondlicht)
    const int h = 64, w = 64;
    Matrix2Df img(h, w);
    
    // Horizontaler Gradient + einige Sterne
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float gradient = static_cast<float>(x) / w * 50.0f;  // Gradient 0-50
            img(y, x) = gradient + 10.0f;  // Basis + Gradient
        }
    }
    
    // Füge einige realistischere Sterne als Gauß-Peaks hinzu.
    // Einzelpixel würden von der Hotpixel-Filterung korrekt verworfen.
    auto add_star = [&](int cx, int cy, float amplitude) {
        for (int y = std::max(0, cy - 3); y < std::min(h, cy + 4); ++y) {
            for (int x = std::max(0, cx - 3); x < std::min(w, cx + 4); ++x) {
                float dx = static_cast<float>(x - cx);
                float dy = static_cast<float>(y - cy);
                img(y, x) += amplitude * std::exp(-(dx * dx + dy * dy) / 2.0f);
            }
        }
    };

    add_star(w / 4, h / 4, 100.0f);
    add_star(w / 2, h / 2, 150.0f);
    add_star(3 * w / 4, 3 * h / 4, 120.0f);
    
    // Ohne Hintergrundsubtraktion sollten weniger Sterne gefunden werden
    // (wegen des hohen Gradienten)
    auto stars_no_bg = detect_stars_simple(img, 10, false);
    
    // Mit Hintergrundsubtraktion sollten mehr Sterne gefunden werden
    auto stars_with_bg = detect_stars_simple(img, 10, true);
    
    // Mit Hintergrundsubtraktion sollten mindestens so viele Sterne gefunden werden
    REQUIRE(stars_with_bg.size() >= stars_no_bg.size());
    REQUIRE(stars_with_bg.size() >= 3);
}

TEST_CASE("detect_stars_simple basic functionality") {
    // Erstelle ein einfaches Sternfeld
    const int h = 64, w = 64;
    Matrix2Df img = Matrix2Df::Zero(h, w);
    
    // Füge einige Sterne als Gauß-Peaks hinzu
    auto add_star = [&](int cx, int cy, float amplitude) {
        for (int y = std::max(0, cy-3); y < std::min(h, cy+4); ++y) {
            for (int x = std::max(0, cx-3); x < std::min(w, cx+4); ++x) {
                float dx = static_cast<float>(x - cx);
                float dy = static_cast<float>(y - cy);
                img(y, x) += amplitude * std::exp(-(dx*dx + dy*dy) / 2.0f);
            }
        }
    };
    
    add_star(16, 16, 100.0f);
    add_star(48, 16, 80.0f);
    add_star(32, 48, 90.0f);
    add_star(16, 48, 70.0f);
    add_star(48, 48, 85.0f);
    
    auto stars = detect_stars_simple(img, 10, false);
    
    // Sollte mindestens 3 Sterne finden
    REQUIRE(stars.size() >= 3);
    
    // Sollte höchstens topk Sterne zurückgeben
    REQUIRE(stars.size() <= 10);
}

// §4.13 — Tests für astrometrische Rescue (ohne ASTAP-Abhängigkeit)
TEST_CASE("is_astap_available returns false when binary missing") {
    // Mit leeren Pfaden sollte ASTAP nicht verfügbar sein
    REQUIRE_FALSE(is_astap_available("", ""));
    
    // Mit nicht-existentem Pfad sollte es nicht verfügbar sein
    REQUIRE_FALSE(is_astap_available("/nonexistent/astap_cli", "/nonexistent"));
}

TEST_CASE("wcs_to_similarity_warp basic transformation") {
    using namespace tile_compile::astrometry;
    
    // Erstelle zwei identische WCS (keine Transformation)
    WCS wcs_mov;
    wcs_mov.crpix1 = 256.0;
    wcs_mov.crpix2 = 256.0;
    wcs_mov.crval1 = 100.0;
    wcs_mov.crval2 = 20.0;
    wcs_mov.cd1_1 = 0.001;  // ~3.6 arcsec/pixel
    wcs_mov.cd1_2 = 0.0;
    wcs_mov.cd2_1 = 0.0;
    wcs_mov.cd2_2 = 0.001;
    wcs_mov.naxis1 = 512;
    wcs_mov.naxis2 = 512;
    
    WCS wcs_ref = wcs_mov;
    
    auto warp = wcs_to_similarity_warp(wcs_mov, wcs_ref, 1.0f);
    
    // Bei identischen WCS sollte der Warp nahezu Identity sein
    REQUIRE(warp(0, 0) == Catch::Approx(1.0f).margin(0.01f));
    REQUIRE(warp(1, 1) == Catch::Approx(1.0f).margin(0.01f));
    REQUIRE(warp(0, 1) == Catch::Approx(0.0f).margin(0.01f));
    REQUIRE(warp(1, 0) == Catch::Approx(0.0f).margin(0.01f));
}

} // namespace tile_compile::registration

#else
// Stub when Catch2 is not available
int tile_compile_registration_new_features_tests_bootstrap() { return 0; }
#endif
