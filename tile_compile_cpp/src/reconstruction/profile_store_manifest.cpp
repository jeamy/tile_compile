#include "tile_compile/reconstruction/profile_store_manifest.hpp"

#include "tile_compile/core/utils.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <system_error>
#include <set>
#include <stdexcept>

namespace tile_compile::reconstruction {

using json = nlohmann::json;

namespace {

struct ByteSink {
  std::vector<uint8_t> bytes;
  void u32(uint32_t v) {
    bytes.push_back(static_cast<uint8_t>(v & 0xff));
    bytes.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
    bytes.push_back(static_cast<uint8_t>((v >> 16) & 0xff));
    bytes.push_back(static_cast<uint8_t>((v >> 24) & 0xff));
  }
  void i32(int32_t v) { u32(static_cast<uint32_t>(v)); }
  void u64(uint64_t v) {
    u32(static_cast<uint32_t>(v & 0xffffffffu));
    u32(static_cast<uint32_t>((v >> 32) & 0xffffffffu));
  }
  void str(const std::string& s) {
    u64(s.size());
    bytes.insert(bytes.end(), s.begin(), s.end());
  }
};

bool valid_manifest_shape(const ProfileStoreManifest &m) {
  if (m.profile.empty() || m.internal_width <= 0 || m.internal_height <= 0 || m.planes.empty())
    return false;
  std::set<std::string> names;
  for (const auto &p : m.planes) {
    // Logical stems only: no absolute paths, traversal or alternate streams.
    if (p.name.empty() || p.name.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos ||
        !names.insert(p.name).second || p.width != m.internal_width ||
        p.height != m.internal_height)
      return false;
    // T1: v1 requires a 64-hex sha256; v2 requires bytes > 0.
    if (m.schema_version >= 2) {
      if (p.bytes == 0) return false;
    } else {
      if (p.sha256.size() != 64 ||
          p.sha256.find_first_not_of("0123456789abcdef") != std::string::npos)
        return false;
    }
  }
  return true;
}

}  // namespace

std::string compute_profile_store_manifest_hash(const ProfileStoreManifest& m) {
  ByteSink s;
  s.str("profile_store_manifest:v1");
  s.i32(m.schema_version);
  s.str(m.profile);
  s.i32(m.internal_width);
  s.i32(m.internal_height);
  s.u64(m.planes.size());
  for (const auto& p : m.planes) {
    s.str(p.name);
    // T1: v1 manifests hash the sha256 field; v2 manifests hash the bytes field.
    if (m.schema_version >= 2)
      s.u64(p.bytes);
    else
      s.str(p.sha256);
    s.i32(p.width);
    s.i32(p.height);
  }
  return core::sha256_bytes(s.bytes);
}

ProfileStoreManifest build_profile_store_manifest(const std::string& profile,
                                                 int internal_width, int internal_height,
                                                 const fs::path& dir,
                                                 const std::vector<std::string>& plane_names) {
  ProfileStoreManifest m;
  m.schema_version = ProfileStoreManifest::kSchemaVersion;
  m.profile = profile;
  m.internal_width = internal_width;
  m.internal_height = internal_height;

  std::vector<std::string> names = plane_names;
  std::sort(names.begin(), names.end());  // canonical order
  for (const auto& name : names) {
    const fs::path file = dir / (name + ".fits");
    if (!fs::is_regular_file(fs::symlink_status(file)))
      throw std::invalid_argument("PROFILE_STORE_INVALID_FILE");
    ProfileStoreEntry e;
    e.name = name;
    e.width = internal_width;
    e.height = internal_height;
    e.bytes = fs::file_size(file);  // T1: set before probe
    ProfileStoreManifest probe = m;
    probe.planes.push_back(e);
    if (!valid_manifest_shape(probe))
      throw std::invalid_argument("PROFILE_STORE_INVALID_MANIFEST");
    m.planes.push_back(std::move(e));
  }
  if (!valid_manifest_shape(m))
    throw std::invalid_argument("PROFILE_STORE_INVALID_MANIFEST");
  m.manifest_hash = compute_profile_store_manifest_hash(m);
  return m;
}

std::string serialize_profile_store_manifest(const ProfileStoreManifest& m) {
  json j;
  j["schema_version"] = m.schema_version;
  j["profile"] = m.profile;
  j["internal_width"] = m.internal_width;
  j["internal_height"] = m.internal_height;
  j["manifest_hash"] = m.manifest_hash;
  j["planes"] = json::array();
  for (const auto& p : m.planes) {
    json pe = {
        {"name", p.name},
        {"width", p.width},
        {"height", p.height},
    };
    // T1: v2 writes bytes; v1 writes sha256.
    if (m.schema_version >= 2)
      pe["bytes"] = p.bytes;
    else
      pe["sha256"] = p.sha256;
    j["planes"].push_back(pe);
  }
  return j.dump(2);
}

bool parse_profile_store_manifest(const std::string& text, ProfileStoreManifest& out,
                                  std::string& error) {
  try {
    const json j = json::parse(text);
    const int sv = j.value("schema_version", -1);
    if (sv != 1 && sv != 2) {
      error = "unsupported profile_store_manifest schema_version";
      return false;
    }
    ProfileStoreManifest m;
    m.schema_version = sv;
    m.profile = j.at("profile").get<std::string>();
    m.internal_width = j.at("internal_width").get<int>();
    m.internal_height = j.at("internal_height").get<int>();
    m.manifest_hash = j.at("manifest_hash").get<std::string>();
    if (!j.at("planes").is_array())
      throw std::invalid_argument("PROFILE_STORE_PLANES_NOT_ARRAY");
    for (const auto& jp : j.at("planes")) {
      ProfileStoreEntry e;
      e.name = jp.at("name").get<std::string>();
      e.width = jp.at("width").get<int>();
      e.height = jp.at("height").get<int>();
      // T1: v1 has sha256, v2 has bytes. Parse whichever is present.
      if (jp.contains("sha256"))
        e.sha256 = jp.at("sha256").get<std::string>();
      if (jp.contains("bytes"))
        e.bytes = jp.at("bytes").get<std::uintmax_t>();
      m.planes.push_back(std::move(e));
    }
    if (m.manifest_hash != compute_profile_store_manifest_hash(m)) {
      error = "profile_store_manifest hash mismatch";
      return false;
    }
    if (!valid_manifest_shape(m))
      throw std::invalid_argument("PROFILE_STORE_INVALID_MANIFEST");
    error.clear();
    out = std::move(m);
    return true;
  } catch (const std::exception& e) {
    error = std::string("profile_store_manifest parse error: ") + e.what();
    return false;
  }
}

ProfileStoreVerifyResult verify_profile_store(const fs::path& dir,
                                             const ProfileStoreManifest& manifest) {
  ProfileStoreVerifyResult r;
  r.manifest_hash_ok =
      manifest.manifest_hash == compute_profile_store_manifest_hash(manifest);
  if (!r.manifest_hash_ok || !valid_manifest_shape(manifest)) return r;
  for (const auto& p : manifest.planes) {
    const fs::path f = dir / (p.name + ".fits");
    std::error_code ec;
    if (!fs::exists(f, ec)) {
      r.missing.push_back(p.name);
      continue;
    }
    if (!fs::is_regular_file(fs::symlink_status(f, ec)) || ec) {
      r.corrupt.push_back(p.name);
      continue;
    }
    // T1: size check only (trusted run). v1 manifests without a bytes field
    // fall back to existence check only.
    if (manifest.schema_version >= 2 && p.bytes > 0) {
      if (fs::file_size(f) != p.bytes) r.corrupt.push_back(p.name);
    }
  }
  r.usable = r.manifest_hash_ok && r.missing.empty() && r.corrupt.empty();
  return r;
}

}  // namespace tile_compile::reconstruction
