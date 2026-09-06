#pragma once

// Profile store manifest --- milestone M3 (plan section 23 M3 acceptance:
// "Raw wird atomar mit Checksumme persistiert; Uniform-Fallback
// funktioniert"). A small, plane-file-agnostic manifest that records every
// persisted profile plane (name, dimensions, sha256 of the file bytes) plus
// a canonical hash over the whole manifest, so a resume can fail-closed on
// a truncated / corrupted / missing plane file rather than silently reading
// a bad reconstruction baseline (audit 2026-09-05 finding A6).
//
// The plane files themselves are written by the caller (currently
// io::write_fits_float, which is atomic --- stage + fsync + rename, see
// fits_io.cpp / plan 30.14). This module only tracks and verifies them.

#include "tile_compile/core/types.hpp"

#include <string>
#include <vector>

namespace tile_compile::reconstruction {

struct ProfileStoreEntry {
  std::string name;    // logical name, e.g. "R_value" --- also the file stem
  std::string sha256;  // sha256 of the file's raw bytes
  int width = 0;
  int height = 0;
};

struct ProfileStoreManifest {
  static constexpr int kSchemaVersion = 1;
  std::string profile;                    // e.g. "raw" or "uniform"
  int internal_width = 0;
  int internal_height = 0;
  std::vector<ProfileStoreEntry> planes;  // canonical order = sorted by name
  std::string manifest_hash;              // canonical, over every field above
};

struct ProfileStoreVerifyResult {
  bool usable = false;                    // every listed plane present + checksum matches
  std::vector<std::string> missing;       // manifest entries with no file on disk
  std::vector<std::string> corrupt;       // files present but sha256 mismatch
  bool manifest_hash_ok = false;          // manifest_hash matches a fresh recompute
};

// Builds a manifest for the plane files `<dir>/<name>.fits` (name from
// `plane_names`). Reads each file to hash it, so the files must already be
// committed. `plane_names` is sorted internally for canonical order.
ProfileStoreManifest build_profile_store_manifest(
    const std::string& profile, int internal_width, int internal_height,
    const fs::path& dir, const std::vector<std::string>& plane_names);

std::string compute_profile_store_manifest_hash(const ProfileStoreManifest& m);

std::string serialize_profile_store_manifest(const ProfileStoreManifest& m);
bool parse_profile_store_manifest(const std::string& text, ProfileStoreManifest& out,
                                  std::string& error);

// Re-hashes every plane file in `dir` and compares against `manifest`.
// `usable` is true only when the manifest hash re-validates AND every listed
// plane is present with a matching checksum. This is file integrity only:
// callers must separately validate required channel/plane completeness, FITS
// dimensions, provenance and all predecessor artifacts before resume/fallback.
ProfileStoreVerifyResult verify_profile_store(const fs::path& dir,
                                              const ProfileStoreManifest& manifest);

}  // namespace tile_compile::reconstruction
