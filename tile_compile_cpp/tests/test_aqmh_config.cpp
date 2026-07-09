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
#endif
