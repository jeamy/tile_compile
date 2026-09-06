#include "tile_compile/reconstruction/normalized_source_cache.hpp"
#include "tile_compile/core/utils.hpp"
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <bit>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace tile_compile::reconstruction {
namespace {
using json = nlohmann::json;
std::string digest(const json &j) {
  const auto text = j.dump();
  return core::sha256_bytes(std::vector<uint8_t>(text.begin(),text.end()));
}
json context(const registration::RegistrationSamplingPlan &plan) {
  static_assert(Matrix2Df::IsRowMajor);
  if (std::endian::native != std::endian::little || sizeof(float) != 4 ||
      !std::numeric_limits<float>::is_iec559 || plan.source_width <= 0 ||
      plan.source_height <= 0 || plan.source_identity_hash.empty() || plan.frames.empty() ||
      (plan.color_mode != ColorMode::MONO && plan.color_mode != ColorMode::OSC) ||
      (plan.color_mode == ColorMode::OSC && plan.bayer_pattern == BayerPattern::UNKNOWN))
    throw std::invalid_argument("NORMALIZED_CACHE_INVALID_CONTEXT");
  std::map<size_t,std::string> frames;
  std::set<std::string> ids;
  for (const auto &f : plan.frames)
    if (f.frame_id.empty() || !frames.emplace(f.source_index,f.frame_id).second ||
        !ids.insert(f.frame_id).second)
      throw std::invalid_argument("NORMALIZED_CACHE_DUPLICATE_FRAME");
  json entries = json::array();
  for (const auto &[index,id] : frames)
    entries.push_back({{"source_index",index},{"frame_id",id}});
  return {{"source_identity_hash",plan.source_identity_hash},
      {"width",plan.source_width},{"height",plan.source_height},
      {"color_mode",static_cast<int>(plan.color_mode)},
      {"bayer_pattern",static_cast<int>(plan.bayer_pattern)},
      {"cfa_origin_x",plan.cfa_origin_x},{"cfa_origin_y",plan.cfa_origin_y},
      {"encoding","ieee754-float32-le-row-major"},{"frames",entries}};
}
size_t frame_bytes(int width,int height) {
  const uint64_t count = static_cast<uint64_t>(width)*height;
  if (count > std::numeric_limits<size_t>::max()/sizeof(float) ||
      count > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())/sizeof(float))
    throw std::runtime_error("NORMALIZED_CACHE_SIZE_OVERFLOW");
  return static_cast<size_t>(count)*sizeof(float);
}
void require_file(const fs::path &p,size_t bytes) {
  if (!fs::is_regular_file(fs::symlink_status(p)) || fs::file_size(p)!=bytes)
    throw std::runtime_error("NORMALIZED_CACHE_MISSING_OR_INVALID_FILE");
}
}
std::string publish_normalized_source_manifest(
    const fs::path &root,const registration::RegistrationSamplingPlan &plan) {
  const auto expected = context(plan);
  json manifest = {{"schema_version",1},{"context",expected},{"files",json::array()}};
  const size_t bytes = frame_bytes(plan.source_width,plan.source_height);
  for (const auto &entry : expected.at("frames")) {
    const size_t index = entry.at("source_index").get<size_t>();
    const auto file = root/(std::to_string(index)+".raw");
    require_file(file,bytes);
    manifest["files"].push_back({{"source_index",index},{"bytes",bytes},
                                {"sha256",core::sha256_file(file)}});
  }
  const auto hash = digest(manifest);
  manifest["manifest_hash"] = hash;
  core::write_text_atomic(root/"normalized_source_manifest.json",manifest.dump(2));
  return hash;
}
VerifiedNormalizedSourceCache::VerifiedNormalizedSourceCache(
    const fs::path &root,const registration::RegistrationSamplingPlan &expected,
    size_t memory_budget_mb) : root_(root),width_(expected.source_width),
    height_(expected.source_height) {
  const auto ctx = context(expected);
  const size_t bytes = frame_bytes(width_,height_);
  if (memory_budget_mb > std::numeric_limits<size_t>::max()/(1024*1024) ||
      memory_budget_mb < 2 || bytes > memory_budget_mb*1024*1024-1024*1024)
    throw std::runtime_error("NORMALIZED_CACHE_MEMORY_BUDGET");
  const auto path = root/"normalized_source_manifest.json";
  if (!fs::is_regular_file(fs::symlink_status(path)) ||
      fs::file_size(path)>std::min<size_t>(16*1024*1024,memory_budget_mb*1024*1024/16))
    throw std::runtime_error("NORMALIZED_CACHE_INVALID_MANIFEST_FILE");
  std::ifstream file(path);
  auto manifest = json::parse(file);
  manifest_hash_=manifest.at("manifest_hash").get<std::string>();
  manifest.erase("manifest_hash");
  if (manifest.at("schema_version")!=1 || manifest.at("context")!=ctx ||
      digest(manifest)!=manifest_hash_ || !manifest.at("files").is_array() ||
      manifest.at("files").size()!=expected.frames.size())
    throw std::runtime_error("NORMALIZED_CACHE_CONTEXT_OR_HASH_MISMATCH");
  size_t i=0;
  for (const auto &entry : manifest.at("files")) {
    if (!entry.at("source_index").is_number_unsigned() || !entry.at("bytes").is_number_unsigned())
      throw std::runtime_error("NORMALIZED_CACHE_INVALID_ENTRY_TYPE");
    const size_t index=entry.at("source_index").get<size_t>();
    const auto hash=entry.at("sha256").get<std::string>();
    if (index!=ctx.at("frames").at(i++).at("source_index").get<size_t>() ||
        entry.at("bytes").get<size_t>()!=bytes || hash.size()!=64 ||
        hash.find_first_not_of("0123456789abcdef")!=std::string::npos)
      throw std::runtime_error("NORMALIZED_CACHE_INVALID_ENTRY");
    require_file(root/(std::to_string(index)+".raw"),bytes);
    hashes_.emplace(index,hash);
  }
  context_hash_=digest(ctx);
}
bool VerifiedNormalizedSourceCache::matches(const registration::RegistrationSamplingPlan &plan) const {
  return digest(context(plan))==context_hash_;
}
const Matrix2Df &VerifiedNormalizedSourceCache::load(size_t source_index) {
  const auto found=hashes_.find(source_index);
  if (found==hashes_.end()) throw std::invalid_argument("NORMALIZED_CACHE_UNKNOWN_FRAME");
  const size_t bytes=frame_bytes(width_,height_);
  const auto path=root_/(std::to_string(source_index)+".raw");
  require_file(path,bytes);
  image_.resize(0,0);
  image_.resize(height_,width_);
  std::ifstream file(path,std::ios::binary);
  file.read(reinterpret_cast<char *>(image_.data()),static_cast<std::streamsize>(bytes));
  if (!file || file.peek()!=std::char_traits<char>::eof()) {
    image_.resize(0,0);
    throw std::runtime_error("NORMALIZED_CACHE_READ_FAILED");
  }
  // Hash the actual image bytes, not a second read of a possibly replaced file.
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char *>(image_.data()),bytes,hash);
  std::ostringstream encoded;
  for (unsigned char b : hash) encoded<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<int>(b);
  if (encoded.str()!=found->second) {
    image_.resize(0,0);
    throw std::runtime_error("NORMALIZED_CACHE_CONTENT_MISMATCH");
  }
  return image_;
}
} // namespace tile_compile::reconstruction
