#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/config/configuration.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("aqmh_diagnostics_level_parses_and_validates") {
  const auto valid_levels = {"none", "summary", "full"};
  for (const auto* level : valid_levels) {
    YAML::Node node = YAML::Load(std::string("\n") +
                                 "method: aqmh\n"
                                 "aqmh:\n"
                                 "  enabled: true\n"
                                 "  diagnostics:\n"
                                 "    enabled: true\n"
                                 "    level: " + level + "\n");
    auto cfg = tile_compile::config::Config::from_yaml(node);
    REQUIRE(cfg.aqmh.diagnostics.level == level);
    REQUIRE_NOTHROW(cfg.validate());
  }
}

TEST_CASE("aqmh_diagnostics_level_rejects_invalid_value") {
  YAML::Node node = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  diagnostics:
    enabled: true
    level: verbose
)");
  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}

TEST_CASE("aqmh_diagnostics_regions_parses_default_and_custom") {
  YAML::Node defaults = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
)");
  auto cfg_default = tile_compile::config::Config::from_yaml(defaults);
  REQUIRE(cfg_default.aqmh.diagnostics.regions);
  REQUIRE_NOTHROW(cfg_default.validate());

  YAML::Node disabled = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  diagnostics:
    regions: false
)");
  auto cfg_disabled = tile_compile::config::Config::from_yaml(disabled);
  REQUIRE_FALSE(cfg_disabled.aqmh.diagnostics.regions);
  REQUIRE_NOTHROW(cfg_disabled.validate());
}

TEST_CASE("aqmh_diagnostics_format_parses_and_validates") {
  YAML::Node json = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  diagnostics:
    format: json
)");
  auto cfg_json = tile_compile::config::Config::from_yaml(json);
  REQUIRE(cfg_json.aqmh.diagnostics.format == "json");
  REQUIRE_NOTHROW(cfg_json.validate());

  YAML::Node binary = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  diagnostics:
    format: binary
)");
  auto cfg_binary = tile_compile::config::Config::from_yaml(binary);
  REQUIRE(cfg_binary.aqmh.diagnostics.format == "binary");
  REQUIRE_NOTHROW(cfg_binary.validate());
}

TEST_CASE("aqmh_diagnostics_format_rejects_invalid_value") {
  YAML::Node node = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  diagnostics:
    format: xml
)");
  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}

TEST_CASE("aqmh_reconstruction_chunk_rows_parses_and_validates") {
  YAML::Node node = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  reconstruction:
    chunk_rows: 64
)");
  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.aqmh.reconstruction.chunk_rows == 64);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("aqmh_reconstruction_chunk_rows_rejects_negative_value") {
  YAML::Node node = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  reconstruction:
    chunk_rows: -1
)");
  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_THROWS(cfg.validate());
}

TEST_CASE("aqmh_reconstruction_prewarp_cache_cleanup_policy_parses") {
  YAML::Node defaults = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
)");
  auto cfg_default = tile_compile::config::Config::from_yaml(defaults);
  REQUIRE(cfg_default.aqmh.reconstruction.delete_prewarped_cache_after_run);

  YAML::Node retained = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  reconstruction:
    delete_prewarped_cache_after_run: false
)");
  auto cfg_retained = tile_compile::config::Config::from_yaml(retained);
  REQUIRE_FALSE(cfg_retained.aqmh.reconstruction.delete_prewarped_cache_after_run);
  REQUIRE_NOTHROW(cfg_retained.validate());
}

TEST_CASE("aqmh_reconstruction_clip_sigma_drives_symmetric_thresholds") {
  YAML::Node node = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  reconstruction:
    clip_sigma: 2.75
)");
  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.aqmh.reconstruction.clip_sigma == 2.75f);
  REQUIRE(cfg.aqmh.reconstruction.clip_sigma_low == 2.75f);
  REQUIRE(cfg.aqmh.reconstruction.clip_sigma_high == 2.75f);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("aqmh_reconstruction_explicit_asymmetric_clip_thresholds_win") {
  YAML::Node node = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  reconstruction:
    clip_sigma: 3.0
    clip_sigma_low: 2.0
    clip_sigma_high: 1.5
)");
  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE(cfg.aqmh.reconstruction.clip_sigma == 3.0f);
  REQUIRE(cfg.aqmh.reconstruction.clip_sigma_low == 2.0f);
  REQUIRE(cfg.aqmh.reconstruction.clip_sigma_high == 1.5f);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("aqmh_reconstruction_ignores_legacy_gpu_backend_value") {
  YAML::Node node = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  reconstruction:
    gpu_reconstruction: opencv_cuda
)");
  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("aqmh_diagnostics_disabled_level_none_still_validates") {
  YAML::Node node = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  diagnostics:
    enabled: false
    level: full
)");
  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_FALSE(cfg.aqmh.diagnostics.enabled);
  REQUIRE(cfg.aqmh.diagnostics.level == "full");
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("debayer_first_defaults_to_true") {
  YAML::Node defaults = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
)");
  auto cfg = tile_compile::config::Config::from_yaml(defaults);
  REQUIRE(cfg.aqmh.reconstruction.debayer_first);
  REQUIRE(cfg.aqmh.reconstruction.pre_debayer_method == "edge_aware");
  REQUIRE(cfg.aqmh.reconstruction.rgb_q_map_mode == "shared_luma");
  REQUIRE(cfg.aqmh.reconstruction.rgb_memory_strategy == "sequential");
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("debayer_first_explicit_false_parses") {
  YAML::Node node = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  reconstruction:
    debayer_first: false
)");
  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_FALSE(cfg.aqmh.reconstruction.debayer_first);
  REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("debayer_first_pre_debayer_method_validates_enum") {
  for (const auto* m : {"edge_aware", "bilinear", "nearest"}) {
    YAML::Node node = YAML::Load(std::string("\n") +
                                 "method: aqmh\n"
                                 "aqmh:\n"
                                 "  enabled: true\n"
                                 "  reconstruction:\n"
                                 "    pre_debayer_method: " + m + "\n");
    auto cfg = tile_compile::config::Config::from_yaml(node);
    REQUIRE(cfg.aqmh.reconstruction.pre_debayer_method == m);
    REQUIRE_NOTHROW(cfg.validate());
  }
  YAML::Node bad = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  reconstruction:
    pre_debayer_method: vng
)");
  auto cfg_bad = tile_compile::config::Config::from_yaml(bad);
  REQUIRE_THROWS(cfg_bad.validate());
}

TEST_CASE("debayer_first_rgb_q_map_mode_validates_enum") {
  for (const auto* m : {"shared_luma", "per_channel"}) {
    YAML::Node node = YAML::Load(std::string("\n") +
                                 "method: aqmh\n"
                                 "aqmh:\n"
                                 "  enabled: true\n"
                                 "  reconstruction:\n"
                                 "    rgb_q_map_mode: " + m + "\n");
    auto cfg = tile_compile::config::Config::from_yaml(node);
    REQUIRE(cfg.aqmh.reconstruction.rgb_q_map_mode == m);
    REQUIRE_NOTHROW(cfg.validate());
  }
  YAML::Node bad = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  reconstruction:
    rgb_q_map_mode: hybrid
)");
  auto cfg_bad = tile_compile::config::Config::from_yaml(bad);
  REQUIRE_THROWS(cfg_bad.validate());
}

TEST_CASE("debayer_first_rgb_memory_strategy_validates_enum") {
  for (const auto* m : {"sequential", "parallel"}) {
    YAML::Node node = YAML::Load(std::string("\n") +
                                 "method: aqmh\n"
                                 "aqmh:\n"
                                 "  enabled: true\n"
                                 "  reconstruction:\n"
                                 "    rgb_memory_strategy: " + m + "\n");
    auto cfg = tile_compile::config::Config::from_yaml(node);
    REQUIRE(cfg.aqmh.reconstruction.rgb_memory_strategy == m);
    REQUIRE_NOTHROW(cfg.validate());
  }
  YAML::Node bad = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  reconstruction:
    rgb_memory_strategy: streaming
)");
  auto cfg_bad = tile_compile::config::Config::from_yaml(bad);
  REQUIRE_THROWS(cfg_bad.validate());
}

TEST_CASE("debayer_first_roundtrips_through_serialization") {
  YAML::Node node = YAML::Load(R"(
method: aqmh
aqmh:
  enabled: true
  reconstruction:
    debayer_first: false
    pre_debayer_method: bilinear
    rgb_q_map_mode: per_channel
    rgb_memory_strategy: parallel
)");
  auto cfg = tile_compile::config::Config::from_yaml(node);
  REQUIRE_FALSE(cfg.aqmh.reconstruction.debayer_first);
  REQUIRE(cfg.aqmh.reconstruction.pre_debayer_method == "bilinear");
  REQUIRE(cfg.aqmh.reconstruction.rgb_q_map_mode == "per_channel");
  REQUIRE(cfg.aqmh.reconstruction.rgb_memory_strategy == "parallel");
  REQUIRE_NOTHROW(cfg.validate());
  YAML::Node serialized = cfg.to_yaml();
  auto reparsed = tile_compile::config::Config::from_yaml(serialized);
  REQUIRE_FALSE(reparsed.aqmh.reconstruction.debayer_first);
  REQUIRE(reparsed.aqmh.reconstruction.pre_debayer_method == "bilinear");
  REQUIRE(reparsed.aqmh.reconstruction.rgb_q_map_mode == "per_channel");
  REQUIRE(reparsed.aqmh.reconstruction.rgb_memory_strategy == "parallel");
}
#endif
