#include <catch2/catch_test_macros.hpp>

#include "tile_compile/astrometry/astap_paths.hpp"
#include "tile_compile/config/path_check.hpp"
#include "tile_compile/preprocessing/contract.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace fs = std::filesystem;
using tile_compile::config::Config;
using tile_compile::config::PathCheckScope;
using tile_compile::config::PathIssue;
using tile_compile::config::check_config_paths;
using tile_compile::config::has_blocking_path_issue;

namespace {

struct TempDir {
  fs::path path;
  TempDir() {
    path = fs::temp_directory_path() /
           ("tc_path_check_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

void touch(const fs::path &p) {
  fs::create_directories(p.parent_path());
  std::ofstream(p) << "x";
}

const PathIssue *find_issue(const std::vector<PathIssue> &issues, const std::string &key) {
  for (const auto &i : issues)
    if (i.key == key) return &i;
  return nullptr;
}

#ifndef _WIN32
// Temporarily override an environment variable (restored on destruction).
struct EnvOverride {
  std::string name;
  std::optional<std::string> old;
  EnvOverride(std::string n, const std::string &value) : name(std::move(n)) {
    if (const char *v = std::getenv(name.c_str())) old = v;
    setenv(name.c_str(), value.c_str(), 1);
  }
  ~EnvOverride() {
    if (old) setenv(name.c_str(), old->c_str(), 1);
    else unsetenv(name.c_str());
  }
};
#endif

Config base_config() {
  Config cfg;
  cfg.calibration.use_bias = false;
  cfg.calibration.use_dark = false;
  cfg.calibration.use_flat = false;
  cfg.astrometry.enabled = false;
  cfg.registration.use_astrometry = false;
  cfg.pcc.enabled = false;
  return cfg;
}

} // namespace

TEST_CASE("path check: empty paths are never reported", "[config][path_check]") {
  Config cfg = base_config();
  cfg.calibration.use_dark = true;  // validity is cfg.validate()'s job, not ours
  cfg.astrometry.enabled = true;
  cfg.pcc.enabled = true;
  CHECK(check_config_paths(cfg, fs::current_path()).empty());
}

TEST_CASE("path check: calibration master", "[config][path_check]") {
  TempDir tmp;
  Config cfg = base_config();
  cfg.calibration.dark_master = (tmp.path / "missing_dark.fits").string();

  SECTION("needed and no other source -> blocking") {
    cfg.calibration.use_dark = true;
    const auto issues = check_config_paths(cfg, tmp.path);
    const auto *i = find_issue(issues, "calibration.dark_master");
    REQUIRE(i != nullptr);
    CHECK(i->status == "missing");
    CHECK(i->blocking);
    CHECK(has_blocking_path_issue(issues));
  }
  SECTION("needed but darks_dir exists -> fallback, not blocking") {
    cfg.calibration.use_dark = true;
    fs::create_directories(tmp.path / "darks");
    cfg.calibration.darks_dir = (tmp.path / "darks").string();
    const auto issues = check_config_paths(cfg, tmp.path);
    const auto *i = find_issue(issues, "calibration.dark_master");
    REQUIRE(i != nullptr);
    CHECK(i->status == "fallback_used");
    CHECK_FALSE(i->blocking);
    CHECK(find_issue(issues, "calibration.darks_dir") == nullptr);
  }
  SECTION("use_dark off -> unused warning only") {
    const auto issues = check_config_paths(cfg, tmp.path);
    const auto *i = find_issue(issues, "calibration.dark_master");
    REQUIRE(i != nullptr);
    CHECK(i->status == "unused");
    CHECK_FALSE(has_blocking_path_issue(issues));
  }
  SECTION("calibration out of scope (resume) -> unused") {
    cfg.calibration.use_dark = true;
    PathCheckScope scope;
    scope.calibration = false;
    const auto issues = check_config_paths(cfg, tmp.path, scope);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].status == "unused");
    CHECK_FALSE(issues[0].blocking);
  }
}

TEST_CASE("path check: relative calibration paths resolve against project root",
          "[config][path_check]") {
  TempDir tmp;
  touch(tmp.path / "masters" / "dark.fits");
  Config cfg = base_config();
  cfg.calibration.use_dark = true;
  cfg.calibration.dark_master = "  masters/dark.fits ";
  CHECK(check_config_paths(cfg, tmp.path).empty());

  const auto issues = check_config_paths(cfg, tmp.path / "elsewhere");
  REQUIRE(issues.size() == 1);
  CHECK(issues[0].blocking);
  CHECK(issues[0].resolved == (tmp.path / "elsewhere" / "masters" / "dark.fits").string());
}

TEST_CASE("path check: missing calibration directory", "[config][path_check]") {
  TempDir tmp;
  touch(tmp.path / "flat.fits");
  Config cfg = base_config();
  cfg.calibration.use_flat = true;
  cfg.calibration.flats_dir = (tmp.path / "no_flats").string();

  SECTION("master exists -> fallback") {
    cfg.calibration.flat_master = (tmp.path / "flat.fits").string();
    const auto issues = check_config_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].key == "calibration.flats_dir");
    CHECK(issues[0].status == "fallback_used");
    CHECK_FALSE(issues[0].blocking);
  }
  SECTION("no master -> blocking") {
    const auto issues = check_config_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].blocking);
  }
}

TEST_CASE("path check: preprocess calibration semantics", "[config][path_check]") {
  TempDir tmp;
  touch(tmp.path / "dark.fits");
  tile_compile::preprocessing::Config cfg;

  SECTION("a non-empty master enables its type -> missing master blocks") {
    cfg.calibration.dark_master = (tmp.path / "gone.fits").string();
    const auto issues = tile_compile::config::check_preprocessing_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].key == "calibration.dark_master");
    CHECK(issues[0].blocking);
  }
  SECTION("missing directory is unused when the master is used") {
    cfg.calibration.dark_master = (tmp.path / "dark.fits").string();
    cfg.darks_dir = (tmp.path / "no_darks").string();
    const auto issues = tile_compile::config::check_preprocessing_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].key == "darks_dir");
    CHECK(issues[0].status == "unused");
    CHECK_FALSE(issues[0].blocking);
  }
  SECTION("missing directory without master blocks") {
    cfg.darks_dir = (tmp.path / "no_darks").string();
    const auto issues = tile_compile::config::check_preprocessing_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].blocking);
  }
  SECTION("dark-flats only matter when a flat is applied") {
    cfg.darkflats_dir = (tmp.path / "no_darkflats").string();
    auto issues = tile_compile::config::check_preprocessing_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].status == "unused");
    fs::create_directories(tmp.path / "flats");
    cfg.flats_dir = (tmp.path / "flats").string();
    issues = tile_compile::config::check_preprocessing_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].key == "darkflats_dir");
    CHECK(issues[0].blocking);
  }
  SECTION("ASTAP/PCC paths are ignored without their config block") {
    cfg.astrometry.astap_data_dir = (tmp.path / "no_astap").string();
    cfg.pcc.siril_catalog_dir = (tmp.path / "no_cat").string();
    CHECK(tile_compile::config::check_preprocessing_paths(cfg, tmp.path).empty());
  }
}

#ifndef _WIN32
TEST_CASE("path check: ASTAP data dir falls back to the per-user default",
          "[config][path_check]") {
  TempDir tmp;
  fs::create_directories(tmp.path / "emptybin");
  // Isolate from any ASTAP on this machine: no PATH hits, controlled HOME.
  EnvOverride path_env("PATH", (tmp.path / "emptybin").string());
  EnvOverride home_env("HOME", (tmp.path / "home").string());

  Config cfg = base_config();
  cfg.astrometry.enabled = true;
  cfg.astrometry.astap_data_dir = (tmp.path / "other_machine" / "astap").string();

  SECTION("no default installation -> blocking") {
    const auto issues = check_config_paths(cfg, tmp.path);
    const auto *i = find_issue(issues, "astrometry.astap_data_dir");
    REQUIRE(i != nullptr);
    CHECK(i->status == "missing");
    CHECK(i->blocking);
  }
  SECTION("astrometry disabled, registration rescue on -> warning only") {
    cfg.astrometry.enabled = false;
    cfg.registration.use_astrometry = true;
    const auto issues = check_config_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].status == "missing");
    CHECK_FALSE(issues[0].blocking);
  }
  SECTION("default installation with catalog -> default_used, and the solver uses it") {
    const fs::path def = tmp.path / "home" / ".local" / "share" / "tile_compile" / "astap";
    touch(def / "astap_cli");
    touch(def / "d50_0101.1476");
    const auto issues = check_config_paths(cfg, tmp.path);
    const auto *i = find_issue(issues, "astrometry.astap_data_dir");
    REQUIRE(i != nullptr);
    CHECK(i->status == "default_used");
    CHECK_FALSE(i->blocking);
    CHECK(fs::path(i->fallback) == def);

    const auto resolved = tile_compile::astrometry::resolve_astap_paths(
        "", cfg.astrometry.astap_data_dir);
    CHECK(fs::path(resolved.data_dir) == def);
    CHECK(resolved.binary == def / "astap_cli");
  }
}

TEST_CASE("path check: Siril catalog dir", "[config][path_check]") {
  TempDir tmp;
  Config cfg = base_config();
  cfg.pcc.enabled = true;
  cfg.pcc.siril_catalog_dir = (tmp.path / "missing_catalog").string();
  const fs::path def = tmp.path / "default_catalog";
  fs::create_directories(def);
  EnvOverride cat_env("TILE_COMPILE_SIRIL_CATALOG_DIR", def.string());

  SECTION("default catalog present -> default_used") {
    touch(def / "siril_cat1_healpix8_xpsamp_0.dat");
    cfg.pcc.source = "siril";
    const auto issues = check_config_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].status == "default_used");
    CHECK_FALSE(issues[0].blocking);
  }
  SECTION("no catalog, source siril -> blocking") {
    cfg.pcc.source = "siril";
    const auto issues = check_config_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].status == "missing");
    CHECK(issues[0].blocking);
  }
  SECTION("no catalog, source auto -> online fallback warning") {
    cfg.pcc.source = "auto";
    const auto issues = check_config_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].status == "online_fallback");
    CHECK_FALSE(issues[0].blocking);
  }
  SECTION("PCC queries VizieR only -> unused") {
    cfg.pcc.source = "vizier_gaia";
    const auto issues = check_config_paths(cfg, tmp.path);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].status == "unused");
  }
}
#endif
