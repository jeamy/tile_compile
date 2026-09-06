#pragma once
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include <map>

namespace tile_compile::reconstruction {

// Existing runner cache format: <source_index>.raw, row-major native float32.
// Publication records existing files; it does not normalize, repair, or copy
// them. A later changed/truncated file fails closed when its bytes are loaded.
std::string publish_normalized_source_manifest(
    const fs::path &root, const registration::RegistrationSamplingPlan &plan);

class VerifiedNormalizedSourceCache {
  fs::path root_;
  int width_ = 0, height_ = 0;
  std::map<size_t, std::string> hashes_;
  std::string manifest_hash_, context_hash_;
  Matrix2Df image_;
public:
  VerifiedNormalizedSourceCache(const fs::path &root,
      const registration::RegistrationSamplingPlan &expected,
      size_t memory_budget_mb = 512);
  const Matrix2Df &load(size_t source_index);
  const std::string &manifest_hash() const { return manifest_hash_; }
  bool matches(const registration::RegistrationSamplingPlan &plan) const;
};

} // namespace tile_compile::reconstruction
