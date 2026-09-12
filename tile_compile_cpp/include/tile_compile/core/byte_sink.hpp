#pragma once

// Byte-exact canonical encoder shared by every content-identity hash in the
// codebase (little-endian integers, IEEE-754 float bit patterns with NaN
// payload normalized so a NaN never makes a hash unstable, length-prefixed
// strings). Previously reimplemented independently in
// registration_sampling_plan.cpp, profile_store_manifest.cpp,
// source_quality_map_cache.cpp, and quality_frame_weight_plan.cpp -- each a
// different subset of these methods, none byte-incompatible with another.
// Consolidated here so a future encoding change (or bug fix) only has one
// place to make it.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace tile_compile::core {

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
  void f32(float v) {
    if (std::isnan(v)) v = std::numeric_limits<float>::quiet_NaN();
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    u32(bits);
  }
  void b(bool v) { bytes.push_back(v ? 1 : 0); }
  void str(const std::string& s) {
    u64(s.size());
    bytes.insert(bytes.end(), s.begin(), s.end());
  }
};

} // namespace tile_compile::core
