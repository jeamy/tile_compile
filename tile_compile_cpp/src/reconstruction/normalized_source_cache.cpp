#include "tile_compile/reconstruction/normalized_source_cache.hpp"
#include "tile_compile/core/utils.hpp"
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <chrono>
#include <memory>
#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

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
// Rows per block for the run-internal block-check index: a ~256 KiB target,
// rounded DOWN to a whole number of rows (>= 1) so a block never straddles a
// row and a Y window maps to a contiguous block range.
constexpr size_t kBlockTargetBytes = 256*1024;
size_t block_rows_for(int width) {
  const size_t row_bytes = std::max<size_t>(static_cast<size_t>(width)*sizeof(float),1);
  return std::max<size_t>(1, kBlockTargetBytes/row_bytes);
}
std::string hex_digest(const unsigned char *d,size_t n) {
  std::ostringstream enc;
  for (size_t i=0;i<n;++i)
    enc<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<int>(d[i]);
  return enc.str();
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
  // LRU capacity (plan §30.72 O2): how many whole frames fit in the budget,
  // leaving 1 MiB slack. At least one; never more than the manifest holds.
  const size_t usable = memory_budget_mb*1024*1024 - 1024*1024;
  capacity_ = std::max<size_t>(1, std::min<size_t>(hashes_.size(),
                                                   bytes ? usable/bytes : 1));
  block_rows_ = block_rows_for(width_);
}

VerifiedNormalizedSourceCache::VerifiedNormalizedSourceCache(
    const VerifiedNormalizedSourceCache &proto, size_t memory_budget_mb)
    : root_(proto.root_), width_(proto.width_), height_(proto.height_),
      hashes_(proto.hashes_), manifest_hash_(proto.manifest_hash_),
      context_hash_(proto.context_hash_) {
  if (memory_budget_mb < 2)
    throw std::runtime_error("NORMALIZED_CACHE_MEMORY_BUDGET");
  const size_t bytes = frame_bytes(width_, height_);
  const size_t usable = memory_budget_mb*1024*1024 - 1024*1024;
  capacity_ = std::max<size_t>(1, std::min<size_t>(hashes_.size(),
                                                   bytes ? usable/bytes : 1));
  block_rows_ = block_rows_for(width_);
}
bool VerifiedNormalizedSourceCache::matches(const registration::RegistrationSamplingPlan &plan) const {
  return digest(context(plan))==context_hash_;
}
const Matrix2Df &VerifiedNormalizedSourceCache::load(size_t source_index) {
  ++load_calls_;
  const auto hit=resident_.find(source_index);
  if (hit!=resident_.end()) {
    const auto path=root_/(std::to_string(source_index)+".raw");
    std::error_code ec;
    const auto sz=fs::file_size(path,ec);
    const auto mt=fs::last_write_time(path,ec);
    if (!ec && sz==hit->second->file_size && mt==hit->second->mtime &&
        mt<hit->second->verified_at) {
      // Unchanged AND last written strictly before we verified it: promote to
      // front, no read, no SHA-256. The `mt < verified_at` guard closes the
      // same-mtime-tick rewrite window that a size+mtime match alone leaves.
      ++lru_hits_;
      lru_.splice(lru_.begin(),lru_,hit->second);
      return hit->second->image;
    }
    // Size or mtime moved, a same-tick rewrite is possible, or stat failed:
    // drop the stale entry and fall through to a full verifying reload.
    lru_.erase(hit->second);
    resident_.erase(hit);
  }
  return verify_and_insert(source_index);
}

const Matrix2Df &VerifiedNormalizedSourceCache::verify_and_insert(
    size_t source_index) {
  const auto found=hashes_.find(source_index);
  if (found==hashes_.end()) throw std::invalid_argument("NORMALIZED_CACHE_UNKNOWN_FRAME");
  const size_t bytes=frame_bytes(width_,height_);
  const auto path=root_/(std::to_string(source_index)+".raw");
  require_file(path,bytes);
  Matrix2Df image;
  image.resize(height_,width_);
  std::ifstream file(path,std::ios::binary);
  file.read(reinterpret_cast<char *>(image.data()),static_cast<std::streamsize>(bytes));
  if (!file || file.peek()!=std::char_traits<char>::eof())
    throw std::runtime_error("NORMALIZED_CACHE_READ_FAILED");
  bytes_read_+=bytes;  // counted on both entry points (load + region-first touch)
  // Hash the actual image bytes, not a second read of a possibly replaced file.
  unsigned char hash[SHA256_DIGEST_LENGTH];
  const auto sha_t0=std::chrono::steady_clock::now();
  SHA256(reinterpret_cast<const unsigned char *>(image.data()),bytes,hash);
  whole_file_sha_seconds_+=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-sha_t0).count();
  ++hash_computations_;
  std::ostringstream encoded;
  for (unsigned char b : hash) encoded<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<int>(b);
  if (encoded.str()!=found->second)
    throw std::runtime_error("NORMALIZED_CACHE_CONTENT_MISMATCH");
  // §30.81 step 3a-3: build the block-check index from the SAME verified
  // buffer, before it is moved into the LRU entry --- no second read, no
  // second whole-file hash. A later read_rect then just re-checks the (size,
  // mtime, verified_at) triple and reuses these digests.
  publish_block_index(source_index,path,
      digest_blocks(reinterpret_cast<const unsigned char *>(image.data()),bytes));
  std::error_code ec;
  Entry e;
  e.index=source_index;
  e.image=std::move(image);
  e.file_size=fs::file_size(path,ec);
  e.mtime=fs::last_write_time(path,ec);
  // Captured AFTER the read+hash: a hit is trusted only if the file's mtime is
  // strictly older than this instant, so a write concurrent with (or in the
  // same fs tick as) our read is never mistaken for "unchanged".
  e.verified_at=fs::file_time_type::clock::now();
  lru_.push_front(std::move(e));
  resident_[source_index]=lru_.begin();
  // Evict least-recently-used; never the entry just inserted (front). A held
  // reference to some OTHER frame is invalidated here, matching the previous
  // single-buffer contract where any load() invalidated the prior reference.
  while (lru_.size()>capacity_ && lru_.size()>1) {
    resident_.erase(lru_.back().index);
    lru_.pop_back();
    ++evictions_;
  }
  return lru_.front().image;
}

std::vector<std::array<unsigned char,32>>
VerifiedNormalizedSourceCache::digest_blocks(
    const unsigned char *data,size_t bytes) const {
  const size_t blk_bytes=block_rows_*static_cast<size_t>(width_)*sizeof(float);
  std::vector<std::array<unsigned char,32>> digs;
  digs.reserve(bytes/std::max<size_t>(blk_bytes,1)+1);
  for (size_t off=0; off<bytes; off+=blk_bytes) {
    const size_t n=std::min(blk_bytes,bytes-off);
    std::array<unsigned char,32> d{};
    SHA256(data+off,n,d.data());
    digs.push_back(d);
  }
  return digs;
}

void VerifiedNormalizedSourceCache::publish_block_index(
    size_t source_index,const fs::path &path,
    std::vector<std::array<unsigned char,32>> digs) {
  BlockIndex bi;
  bi.block_sha=std::move(digs);
  std::error_code ec;
  bi.file_size=fs::file_size(path,ec);
  bi.mtime=fs::last_write_time(path,ec);
  // Captured AFTER read+hash, matching load()'s discipline: the index is
  // trusted later only while the file's mtime stays strictly older than this.
  bi.verified_at=fs::file_time_type::clock::now();
  block_index_[source_index]=std::move(bi);
  ++block_index_builds_;
}

const VerifiedNormalizedSourceCache::BlockIndex &
VerifiedNormalizedSourceCache::ensure_block_index(size_t source_index) {
  const auto found=hashes_.find(source_index);
  if (found==hashes_.end()) throw std::invalid_argument("NORMALIZED_CACHE_UNKNOWN_FRAME");
  const size_t bytes=frame_bytes(width_,height_);
  const auto path=root_/(std::to_string(source_index)+".raw");
  if (auto it=block_index_.find(source_index); it!=block_index_.end()) {
    std::error_code ec;
    const auto sz=fs::file_size(path,ec);
    const auto mt=fs::last_write_time(path,ec);
    if (!ec && sz==it->second.file_size && mt==it->second.mtime &&
        mt<it->second.verified_at)
      return it->second;                     // trusted across image-LRU eviction
    block_index_.erase(it);                  // drift: rebuild = full re-verify
  }
  // Region-first touch (load() never ran for this index): stream the file in
  // block-sized chunks so peak scratch is ONE block, not a whole frame. The
  // running whole-file SHA-256 must still match the manifest before any block
  // digest is published.
  require_file(path,bytes);
  const size_t blk_bytes=block_rows_*static_cast<size_t>(width_)*sizeof(float);
  std::vector<unsigned char> blk(blk_bytes);
  std::vector<std::array<unsigned char,32>> digs;
  digs.reserve(bytes/std::max<size_t>(blk_bytes,1)+1);
  std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(
      EVP_MD_CTX_new(),&EVP_MD_CTX_free);
  if (!ctx || EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)!=1)
    throw std::runtime_error("NORMALIZED_CACHE_READ_FAILED");
  std::ifstream file(path,std::ios::binary);
  for (size_t off=0; off<bytes; off+=blk_bytes) {
    const size_t n=std::min(blk_bytes,bytes-off);
    file.read(reinterpret_cast<char *>(blk.data()),static_cast<std::streamsize>(n));
    if (!file) throw std::runtime_error("NORMALIZED_CACHE_READ_FAILED");
    EVP_DigestUpdate(ctx.get(),blk.data(),n);
    std::array<unsigned char,32> d{};
    SHA256(blk.data(),n,d.data());
    digs.push_back(d);
  }
  if (file.peek()!=std::char_traits<char>::eof())
    throw std::runtime_error("NORMALIZED_CACHE_READ_FAILED");
  bytes_read_+=bytes;
  unsigned char whole[EVP_MAX_MD_SIZE];
  unsigned int whole_len=0;
  EVP_DigestFinal_ex(ctx.get(),whole,&whole_len);
  ++hash_computations_;
  if (hex_digest(whole,whole_len)!=found->second)
    throw std::runtime_error("NORMALIZED_CACHE_CONTENT_MISMATCH");
  publish_block_index(source_index,path,std::move(digs));
  return block_index_.at(source_index);
}

Matrix2Df VerifiedNormalizedSourceCache::read_rect(
    size_t source_index,int y0,int y1,int x0,int x1) {
  y0=std::clamp(y0,0,height_); y1=std::clamp(y1,0,height_);
  x0=std::clamp(x0,0,width_);  x1=std::clamp(x1,0,width_);
  if (y1<=y0 || x1<=x0) return Matrix2Df(0,0);
  const BlockIndex &bi=ensure_block_index(source_index);
  const size_t frame_total=frame_bytes(width_,height_);
  const size_t row_bytes=static_cast<size_t>(width_)*sizeof(float);
  const size_t blk_bytes=block_rows_*row_bytes;
  const size_t b0=static_cast<size_t>(y0)/block_rows_;
  const size_t b1=static_cast<size_t>(y1-1)/block_rows_;
  const size_t span_y0=b0*block_rows_;
  const size_t span_y1=std::min<size_t>(static_cast<size_t>(height_),(b1+1)*block_rows_);
  const size_t span_off=span_y0*row_bytes;
  const size_t span_bytes=(span_y1-span_y0)*row_bytes;
  std::vector<unsigned char> buf(span_bytes);
  {
    const auto path=root_/(std::to_string(source_index)+".raw");
    std::ifstream file(path,std::ios::binary);
    file.seekg(static_cast<std::streamoff>(span_off));
    file.read(reinterpret_cast<char *>(buf.data()),static_cast<std::streamsize>(span_bytes));
    if (!file) throw std::runtime_error("NORMALIZED_CACHE_READ_FAILED");
  }
  bytes_read_+=span_bytes;
  // Verify every block the Y window covers against its published digest.
  for (size_t b=b0;b<=b1;++b) {
    if (b>=bi.block_sha.size())
      throw std::runtime_error("NORMALIZED_CACHE_BLOCK_INDEX_MISMATCH");
    const size_t off=b*blk_bytes;
    const size_t n=std::min(blk_bytes,frame_total-off);
    std::array<unsigned char,32> d{};
    SHA256(buf.data()+(off-span_off),n,d.data());
    ++blocks_verified_;
    if (d!=bi.block_sha[b])
      throw std::runtime_error("NORMALIZED_CACHE_BLOCK_MISMATCH");
  }
  Matrix2Df out(y1-y0,x1-x0);
  const auto *base=reinterpret_cast<const float *>(buf.data());
  for (int y=y0;y<y1;++y) {
    const float *row=base+(static_cast<size_t>(y)-span_y0)*static_cast<size_t>(width_);
    for (int x=x0;x<x1;++x) out(y-y0,x-x0)=row[x];
  }
  expanded_floats_+=static_cast<std::uint64_t>(y1-y0)*static_cast<std::uint64_t>(x1-x0);
  return out;
}
} // namespace tile_compile::reconstruction
