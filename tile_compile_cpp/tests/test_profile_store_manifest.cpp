// M3 tests for the profile store manifest (plan section 23 M3 acceptance:
// "Raw wird atomar mit Checksumme persistiert; Uniform-Fallback
// funktioniert"). Deterministic data/hashing module --- synthetic tests
// with known ground truth are the complete verification here.

#include "tile_compile/reconstruction/profile_store_manifest.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

using namespace tile_compile;
using namespace tile_compile::reconstruction;
namespace fs = std::filesystem;

namespace {
struct TempDir {
  fs::path path;
  TempDir() {
    path = fs::temp_directory_path() /
           ("profile-store-test-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(reinterpret_cast<uintptr_t>(this)));
    fs::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

void write_file(const fs::path& p, const std::string& content) {
  std::ofstream(p, std::ios::binary) << content;
}
}  // namespace

TEST_CASE("profile store manifest: build + verify a complete store succeeds") {
  TempDir dir;
  write_file(dir.path / "R_value.fits", "fake fits bytes R value");
  write_file(dir.path / "R_weight_sum.fits", "fake fits bytes R weight");
  write_file(dir.path / "G_value.fits", "fake fits bytes G value");

  auto manifest = build_profile_store_manifest(
      "raw", 128, 64, dir.path, {"R_value", "R_weight_sum", "G_value"});
  REQUIRE(manifest.profile == "raw");
  REQUIRE(manifest.planes.size() == 3);
  // canonical (sorted) order
  REQUIRE(manifest.planes[0].name == "G_value");
  REQUIRE(manifest.planes[1].name == "R_value");
  REQUIRE(manifest.planes[2].name == "R_weight_sum");
  REQUIRE_FALSE(manifest.manifest_hash.empty());

  auto v = verify_profile_store(dir.path, manifest);
  REQUIRE(v.usable);
  REQUIRE(v.manifest_hash_ok);
  REQUIRE(v.missing.empty());
  REQUIRE(v.corrupt.empty());
}

TEST_CASE("profile store manifest: a corrupted plane file is caught, store "
          "not usable (audit A6 --- Uniform fallback path)") {
  TempDir dir;
  write_file(dir.path / "R_value.fits", "original bytes");
  auto manifest = build_profile_store_manifest("raw", 10, 10, dir.path, {"R_value"});
  REQUIRE(verify_profile_store(dir.path, manifest).usable);

  write_file(dir.path / "R_value.fits", "TAMPERED bytes");  // same name, different content
  auto v = verify_profile_store(dir.path, manifest);
  REQUIRE_FALSE(v.usable);
  REQUIRE(v.corrupt.size() == 1);
  REQUIRE(v.corrupt[0] == "R_value");
  REQUIRE(v.missing.empty());
}

TEST_CASE("profile store manifest: a missing plane file is reported, store "
          "not usable") {
  TempDir dir;
  write_file(dir.path / "R_value.fits", "bytes");
  write_file(dir.path / "R_weight_sum.fits", "bytes2");
  auto manifest =
      build_profile_store_manifest("raw", 10, 10, dir.path, {"R_value", "R_weight_sum"});

  fs::remove(dir.path / "R_weight_sum.fits");
  auto v = verify_profile_store(dir.path, manifest);
  REQUIRE_FALSE(v.usable);
  REQUIRE(v.missing.size() == 1);
  REQUIRE(v.missing[0] == "R_weight_sum");
}

TEST_CASE("profile store manifest: round-trips through JSON and re-validates "
          "its hash; a tampered manifest is rejected") {
  TempDir dir;
  write_file(dir.path / "L_value.fits", "mono bytes");
  auto manifest = build_profile_store_manifest("uniform", 20, 20, dir.path, {"L_value"});

  const std::string js = serialize_profile_store_manifest(manifest);
  ProfileStoreManifest parsed;
  std::string error;
  REQUIRE(parse_profile_store_manifest(js, parsed, error));
  REQUIRE(error.empty());
  REQUIRE(parsed.manifest_hash == manifest.manifest_hash);
  REQUIRE(parsed.planes.size() == 1);

  // Flip a checksum digit without touching manifest_hash.
  std::string tampered = js;
  const auto pos = tampered.find(manifest.planes[0].sha256);
  REQUIRE(pos != std::string::npos);
  tampered[pos] = (tampered[pos] == 'a') ? 'b' : 'a';
  ProfileStoreManifest bad;
  REQUIRE_FALSE(parse_profile_store_manifest(tampered, bad, error));
  REQUIRE_FALSE(error.empty());
}

TEST_CASE("profile store manifest hash: stable for equal manifests, changes "
          "when any tracked field changes") {
  TempDir dir;
  write_file(dir.path / "A.fits", "aaa");
  write_file(dir.path / "B.fits", "bbb");
  auto m1 = build_profile_store_manifest("raw", 4, 4, dir.path, {"A", "B"});
  auto m1_again = build_profile_store_manifest("raw", 4, 4, dir.path, {"A", "B"});
  REQUIRE(m1.manifest_hash == m1_again.manifest_hash);

  auto m2 = build_profile_store_manifest("raw", 4, 8, dir.path, {"A", "B"});  // height
  REQUIRE(m1.manifest_hash != m2.manifest_hash);

  auto m3 = build_profile_store_manifest("uniform", 4, 4, dir.path, {"A", "B"});  // profile
  REQUIRE(m1.manifest_hash != m3.manifest_hash);

  write_file(dir.path / "B.fits", "BBB-changed");  // file content -> checksum
  auto m4 = build_profile_store_manifest("raw", 4, 4, dir.path, {"A", "B"});
  REQUIRE(m1.manifest_hash != m4.manifest_hash);
}

TEST_CASE("profile store manifest: rehashed malformed stores are never usable", "[drizzle-audit]") {
  TempDir dir;
  write_file(dir.path / "L_value.fits", "bytes");
  const auto good = build_profile_store_manifest("uniform", 2, 2, dir.path, {"L_value"});
  for (int kind = 0; kind < 5; ++kind) {
    auto bad = good;
    if (kind == 0) bad.planes.clear();
    if (kind == 1) bad.planes[0].name = "../L_value";
    if (kind == 2) bad.planes.push_back(bad.planes[0]);
    if (kind == 3) bad.planes[0].width = 0;
    if (kind == 4) bad.planes[0].sha256.clear();
    bad.manifest_hash = compute_profile_store_manifest_hash(bad);
    ProfileStoreManifest parsed;
    std::string error;
    REQUIRE_FALSE(parse_profile_store_manifest(serialize_profile_store_manifest(bad), parsed, error));
    REQUIRE_FALSE(verify_profile_store(dir.path, bad).usable);
  }
  REQUIRE_THROWS(build_profile_store_manifest("raw", 2, 2, dir.path, {}));
  REQUIRE_THROWS(build_profile_store_manifest("raw", 2, 2, dir.path, {"../L_value"}));
}
