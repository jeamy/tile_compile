#include "tile_compile/reconstruction/source_quality_map_cache.hpp"

#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/reconstruction/source_quality_maps.hpp"
#include "tile_compile/reconstruction/source_quality_proxy.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tile_compile::reconstruction {

using json = nlohmann::json;

namespace {

using tile_compile::core::nan_value;

constexpr char kBinMagic[4] = {'S', 'Q', 'M', '1'};
// Schema 2: body is n little-endian uint16 value cells followed by n uint8
// hard-veto cells (1 = an exact Q=0 hard veto was covered; forces NaN on
// read regardless of the value cell). Schema 1 (values only) is no longer
// written or accepted.
constexpr uint32_t kBinSchema = 2;

// Byte-exact canonical encoder --- same convention as
// registration_sampling_plan.cpp / quality_frame_weight_plan.cpp (little
// endian, length-prefixed strings).
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
  void str(const std::string &s) {
    u64(s.size());
    bytes.insert(bytes.end(), s.begin(), s.end());
  }
};

bool finite_f(float v) {
  return (std::bit_cast<uint32_t>(v) & 0x7f800000u) != 0x7f800000u;
}

int storage_dim(int source_dim, int divisor) {
  return (source_dim + divisor - 1) / divisor;
}

void write_bin_atomic(const fs::path &target, int storage_w, int storage_h,
                      int divisor, std::size_t source_index,
                      const std::vector<uint16_t> &cells,
                      const std::vector<uint8_t> &veto_cells) {
  core::AtomicOutput out(target);
  {
    std::ofstream f(out.path(), std::ios::binary);
    if (!f) throw std::runtime_error("SQM_CACHE_BIN_OPEN_FAILED: " +
                                     target.string());
    ByteSink h;
    h.bytes.insert(h.bytes.end(), kBinMagic, kBinMagic + 4);
    h.u32(kBinSchema);
    h.u32(static_cast<uint32_t>(storage_w));
    h.u32(static_cast<uint32_t>(storage_h));
    h.u32(static_cast<uint32_t>(divisor));
    h.u64(static_cast<uint64_t>(source_index));
    f.write(reinterpret_cast<const char *>(h.bytes.data()),
            static_cast<std::streamsize>(h.bytes.size()));
    std::vector<uint8_t> body(cells.size() * 2);
    for (std::size_t i = 0; i < cells.size(); ++i) {
      body[2 * i] = static_cast<uint8_t>(cells[i] & 0xff);
      body[2 * i + 1] = static_cast<uint8_t>((cells[i] >> 8) & 0xff);
    }
    f.write(reinterpret_cast<const char *>(body.data()),
            static_cast<std::streamsize>(body.size()));
    f.write(reinterpret_cast<const char *>(veto_cells.data()),
            static_cast<std::streamsize>(veto_cells.size()));
    if (!f) throw std::runtime_error("SQM_CACHE_BIN_WRITE_FAILED: " +
                                     target.string());
  }
  out.commit();
}

// Fixed .bin header: magic(4) + u32 schema + u32 storage_w + u32 storage_h
// + u32 divisor + u64 source_index. Cells (storage_w*storage_h uint16 LE,
// row-major) follow at kBinHeaderBytes; the veto block (one uint8 per cell)
// follows the cells.
constexpr std::streamoff kBinHeaderBytes = 4 + 4 * 4 + 8;

struct BinHeader {
  int storage_w = 0, storage_h = 0, divisor = 0;
  std::size_t source_index = 0;
};

BinHeader read_bin_header(std::ifstream &f, const fs::path &path) {
  char magic[4];
  f.read(magic, 4);
  if (!f || std::memcmp(magic, kBinMagic, 4) != 0)
    throw std::runtime_error("SQM_CACHE_BIN_BAD_MAGIC: " + path.string());
  auto rd_u32 = [&]() -> uint32_t {
    uint8_t b[4];
    f.read(reinterpret_cast<char *>(b), 4);
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
           (static_cast<uint32_t>(b[2]) << 16) |
           (static_cast<uint32_t>(b[3]) << 24);
  };
  if (rd_u32() != kBinSchema)
    throw std::runtime_error("SQM_CACHE_BIN_BAD_SCHEMA: " + path.string());
  BinHeader h;
  h.storage_w = static_cast<int>(rd_u32());
  h.storage_h = static_cast<int>(rd_u32());
  h.divisor = static_cast<int>(rd_u32());
  const uint32_t lo = rd_u32();
  const uint32_t hi = rd_u32();
  h.source_index =
      static_cast<std::size_t>(lo) | (static_cast<std::size_t>(hi) << 32);
  if (!f || h.storage_w <= 0 || h.storage_h <= 0 || h.divisor <= 0)
    throw std::runtime_error("SQM_CACHE_BIN_BAD_DIMS: " + path.string());
  return h;
}

// §30.81 step 3a-2b: read ONLY the storage cells covering
// [cy0, cy1] x [cx0, cx1] (inclusive) --- one contiguous seek+read per cell
// row, for the value block and the veto block. `win` is
// (cy1-cy0+1) x (cx1-cx0+1), row-major; caller rebases (cy - cy0, cx - cx0).
struct BinWindow {
  int storage_w = 0, storage_h = 0, divisor = 0;
  int cy0 = 0, cx0 = 0, win_h = 0, win_w = 0;
  std::vector<uint16_t> cells;
  std::vector<uint8_t> veto;
};

BinWindow read_bin_window(const fs::path &path, int cy0, int cy1, int cx0,
                          int cx1) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("SQM_CACHE_BIN_MISSING: " + path.string());
  const BinHeader h = read_bin_header(f, path);
  cy0 = std::clamp(cy0, 0, h.storage_h - 1);
  cy1 = std::clamp(cy1, cy0, h.storage_h - 1);
  cx0 = std::clamp(cx0, 0, h.storage_w - 1);
  cx1 = std::clamp(cx1, cx0, h.storage_w - 1);
  BinWindow w;
  w.storage_w = h.storage_w;
  w.storage_h = h.storage_h;
  w.divisor = h.divisor;
  w.cy0 = cy0;
  w.cx0 = cx0;
  w.win_h = cy1 - cy0 + 1;
  w.win_w = cx1 - cx0 + 1;
  const std::size_t run = static_cast<std::size_t>(w.win_w);
  const std::size_t total = run * static_cast<std::size_t>(w.win_h);
  const std::streamoff n_cells = static_cast<std::streamoff>(h.storage_w) *
                                 static_cast<std::streamoff>(h.storage_h);
  const std::streamoff veto_base = kBinHeaderBytes + n_cells * 2;
  w.cells.resize(total);
  w.veto.resize(total);
  std::vector<uint8_t> row(run * 2);
  for (int cy = cy0; cy <= cy1; ++cy) {
    const std::streamoff first =
        static_cast<std::streamoff>(cy) * h.storage_w + cx0;
    const std::size_t out = static_cast<std::size_t>(cy - cy0) * run;
    f.seekg(kBinHeaderBytes + first * 2, std::ios::beg);
    f.read(reinterpret_cast<char *>(row.data()),
           static_cast<std::streamsize>(row.size()));
    if (static_cast<std::size_t>(f.gcount()) != row.size())
      throw std::runtime_error("SQM_CACHE_BIN_TRUNCATED: " + path.string());
    for (std::size_t i = 0; i < run; ++i)
      w.cells[out + i] = static_cast<uint16_t>(row[2 * i]) |
                         (static_cast<uint16_t>(row[2 * i + 1]) << 8);
    f.seekg(veto_base + first, std::ios::beg);
    f.read(reinterpret_cast<char *>(w.veto.data() + out),
           static_cast<std::streamsize>(run));
    if (static_cast<std::size_t>(f.gcount()) != run)
      throw std::runtime_error("SQM_CACHE_BIN_TRUNCATED_VETO: " + path.string());
  }
  return w;
}

std::string stream_file_name(const std::string &stream,
                             std::size_t source_index) {
  char idx[16];
  std::snprintf(idx, sizeof(idx), "%06zu", source_index);
  std::string tag = stream;
  if (stream.rfind("scale_", 0) == 0)
    tag = "s" + stream.substr(6);
  return stream + "/source_quality_" + tag + "_" + idx + ".bin";
}

std::string cache_manifest_hash(const SourceQualityCacheMetadata &m) {
  // Canonical manifest hash; the field source_quality_cache_hash is excluded
  // from its own computation (plan 13.4).
  ByteSink s;
  s.str("sqm-cache-manifest-v1");
  s.i32(m.schema_version);
  s.str(m.coordinate_space);
  s.i32(m.source_width);
  s.i32(m.source_height);
  s.i32(m.storage_divisor);
  s.str(m.dtype);
  s.str(m.source_identity_hash);
  s.str(m.normalized_cache_hash);
  s.str(m.source_quality_config_hash);
  s.i32(m.proxy_version);
  s.i32(m.cfa_origin_x);
  s.i32(m.cfa_origin_y);
  std::vector<std::string> streams = m.streams;
  std::sort(streams.begin(), streams.end());
  s.u64(streams.size());
  for (const auto &st : streams) s.str(st);
  std::vector<SourceQualityCacheFileEntry> files = m.files;
  std::sort(files.begin(), files.end(), [](const auto &a, const auto &b) {
    if (a.stream != b.stream) return a.stream < b.stream;
    return a.source_index < b.source_index;
  });
  s.u64(files.size());
  for (const auto &fe : files) {
    s.str(fe.stream);
    s.u64(fe.source_index);
    s.str(fe.name);
    // T1: v1 manifests hash the sha256 field; v2 manifests hash the bytes field.
    if (m.schema_version >= 2)
      s.u64(fe.bytes);
    else
      s.str(fe.sha256);
  }
  return core::sha256_bytes(s.bytes);
}

}  // namespace

uint16_t quantize_quality(float v) {
  if (!finite_f(v) || v <= 0.0f) return 0u;
  const float clamped = v > 1.0f ? 1.0f : v;
  const long q = std::lround(static_cast<double>(clamped) * 65535.0);
  if (q <= 0) return 1u;
  if (q >= 65535) return 65535u;
  return static_cast<uint16_t>(q);
}

float dequantize_quality(uint16_t q) {
  if (q == 0u) return nan_value();
  return static_cast<float>(q) / 65535.0f;
}

std::string compute_source_quality_identity_hash(
    const registration::RegistrationSamplingPlan &plan,
    const std::string &normalized_cache_hash) {
  ByteSink s;
  s.str("sqm-identity-v1");
  s.i32(plan.source_width);
  s.i32(plan.source_height);
  s.i32(static_cast<int32_t>(plan.color_mode));
  s.i32(static_cast<int32_t>(plan.bayer_pattern));
  s.i32(plan.cfa_origin_x);
  s.i32(plan.cfa_origin_y);
  s.str(normalized_cache_hash);
  s.u64(plan.frames.size());
  for (const auto &f : plan.frames) {
    s.str(f.frame_id);
    s.u64(f.source_index);
  }
  return core::sha256_bytes(s.bytes);
}

std::string compute_scale_quality_config_hash(
    const config::AqmhPyramidConfig &p,
    const SourceQualityMapCacheConfig &c) {
  ByteSink s;
  s.str("sqm-config-v1");
  s.i32(c.proxy_version);
  s.i32(c.storage_divisor);
  s.str(c.dtype);
  s.i32(p.scales);
  s.i32(p.base_window_px);
  s.f32(p.w_sharp);
  s.f32(p.w_snr);
  s.f32(p.score_scale);
  s.f32(p.k_artifact);
  s.f32(p.frac_artifact_max);
  return core::sha256_bytes(s.bytes);
}

// --- Writer ---------------------------------------------------------------

SourceQualityMapCacheWriter::SourceQualityMapCacheWriter(
    fs::path root, const registration::RegistrationSamplingPlan &plan,
    std::string normalized_cache_hash, const config::AqmhPyramidConfig &pyramid,
    SourceQualityMapCacheConfig cache_cfg)
    : root_(std::move(root)),
      source_width_(plan.source_width),
      source_height_(plan.source_height),
      cfa_origin_x_(plan.cfa_origin_x),
      cfa_origin_y_(plan.cfa_origin_y),
      normalized_cache_hash_(std::move(normalized_cache_hash)),
      cfg_(std::move(cache_cfg)) {
  if (source_width_ <= 0 || source_height_ <= 0)
    throw std::invalid_argument("SQM_CACHE_BAD_SOURCE_DIMS");
  if (cfg_.storage_divisor <= 0)
    throw std::invalid_argument("SQM_CACHE_BAD_STORAGE_DIVISOR");
  if (cfg_.dtype != "uint16")
    throw std::invalid_argument("SQM_CACHE_UNSUPPORTED_DTYPE");
  identity_hash_ =
      compute_source_quality_identity_hash(plan, normalized_cache_hash_);
  config_hash_ = compute_scale_quality_config_hash(pyramid, cfg_);
  fs::create_directories(root_);
}

void SourceQualityMapCacheWriter::put(const std::string &stream,
                                      std::size_t source_index,
                                      const Matrix2Df &m) {
  if (m.rows() != source_height_ || m.cols() != source_width_)
    throw std::invalid_argument("SQM_CACHE_MAP_GEOMETRY_MISMATCH");
  const int d = cfg_.storage_divisor;
  const int sw = storage_dim(source_width_, d);
  const int sh = storage_dim(source_height_, d);
  const std::size_t ncells = static_cast<std::size_t>(sw) * sh;
  std::vector<uint16_t> cells(ncells, 0u);
  std::vector<uint8_t> veto_cells(ncells, 0u);

  // Downsample (plan 13.5): the value cell is the valid-mean over strictly
  // positive covered source pixels, so good data next to an unsupported
  // (NaN) border survives. A SEPARATE hard-veto cell is set when ANY covered
  // source pixel is an exact Q=0 hard veto (finite and <= 0); the read path
  // then forces NaN there so an exact zero-veto can never resample positive.
  // Cells with neither positive nor hard-veto data (all NaN = no support)
  // store value 0 -> decode NaN, veto 0.
  for (int cy = 0; cy < sh; ++cy) {
    for (int cx = 0; cx < sw; ++cx) {
      double sum = 0.0;
      int count = 0;
      bool hard_veto = false;
      for (int y = cy * d; y < std::min(source_height_, (cy + 1) * d); ++y) {
        for (int x = cx * d; x < std::min(source_width_, (cx + 1) * d); ++x) {
          const float v = m(y, x);
          if (!finite_f(v)) continue;      // no support -- neither veto nor data
          if (v <= 0.0f) {
            hard_veto = true;              // explicit Q=0 hard veto
          } else {
            sum += v;
            ++count;
          }
        }
      }
      const std::size_t idx =
          static_cast<std::size_t>(cy) * static_cast<std::size_t>(sw) + cx;
      cells[idx] = count == 0
                       ? 0u
                       : quantize_quality(static_cast<float>(sum / count));
      veto_cells[idx] = hard_veto ? 1u : 0u;
    }
  }

  const std::string name = stream_file_name(stream, source_index);
  const fs::path target = root_ / name;
  {
    // AtomicOutput's staging dir lives under target.parent_path(), so the
    // stream directory must exist before write_bin_atomic. Concurrent
    // create_directories on the same path can throw on some libstdc++, so
    // serialise just this step.
    std::lock_guard<std::mutex> lk(files_mu_);
    fs::create_directories(target.parent_path());
  }
  // Lock-free: distinct (stream, source_index) -> distinct file, written via
  // AtomicOutput (its own unique staging dir).
  write_bin_atomic(target, sw, sh, d, source_index, cells, veto_cells);

  SourceQualityCacheFileEntry e;
  e.stream = stream;
  e.source_index = source_index;
  e.name = name;
  e.bytes = fs::file_size(target);  // T1: size only, no SHA-256
  {
    std::lock_guard<std::mutex> lk(files_mu_);
    files_.erase(std::remove_if(files_.begin(), files_.end(),
                                [&](const SourceQualityCacheFileEntry &fe) {
                                  return fe.stream == stream &&
                                         fe.source_index == source_index;
                                }),
                 files_.end());
    files_.push_back(std::move(e));
  }
}

SourceQualityCacheMetadata SourceQualityMapCacheWriter::commit() {
  SourceQualityCacheMetadata m;
  m.schema_version = 2;  // T1: trusted-run schema (no per-file SHA-256)
  m.coordinate_space = "source_cfa";
  m.source_width = source_width_;
  m.source_height = source_height_;
  m.storage_divisor = cfg_.storage_divisor;
  m.dtype = cfg_.dtype;
  m.source_identity_hash = identity_hash_;
  m.normalized_cache_hash = normalized_cache_hash_;
  m.source_quality_config_hash = config_hash_;
  m.proxy_version = cfg_.proxy_version;
  m.cfa_origin_x = cfa_origin_x_;
  m.cfa_origin_y = cfa_origin_y_;
  m.files = files_;

  std::sort(m.files.begin(), m.files.end(), [](const auto &a, const auto &b) {
    if (a.stream != b.stream) return a.stream < b.stream;
    return a.source_index < b.source_index;
  });
  for (const auto &fe : m.files)
    if (std::find(m.streams.begin(), m.streams.end(), fe.stream) ==
        m.streams.end())
      m.streams.push_back(fe.stream);
  std::sort(m.streams.begin(), m.streams.end());

  m.source_quality_cache_hash = cache_manifest_hash(m);

  json j;
  j["schema_version"] = m.schema_version;
  j["coordinate_space"] = m.coordinate_space;
  j["source_width"] = m.source_width;
  j["source_height"] = m.source_height;
  j["storage_divisor"] = m.storage_divisor;
  j["dtype"] = m.dtype;
  j["source_identity_hash"] = m.source_identity_hash;
  j["normalized_cache_hash"] = m.normalized_cache_hash;
  j["source_quality_config_hash"] = m.source_quality_config_hash;
  j["source_quality_cache_hash"] = m.source_quality_cache_hash;
  j["proxy_version"] = m.proxy_version;
  j["cfa_origin_x"] = m.cfa_origin_x;
  j["cfa_origin_y"] = m.cfa_origin_y;
  j["streams"] = m.streams;
  j["files"] = json::array();
  for (const auto &fe : m.files)
    j["files"].push_back({{"stream", fe.stream},
                          {"source_index", fe.source_index},
                          {"name", fe.name},
                          {"bytes", fe.bytes}});

  core::write_text_atomic(root_ / "metadata.json", j.dump(2));
  return m;
}

// --- Reader ---------------------------------------------------------------

SourceQualityMapCacheReader::SourceQualityMapCacheReader(
    fs::path root, std::string expected_identity_hash,
    std::string expected_config_hash)
    : root_(std::move(root)) {
  try {
    const fs::path meta_path = root_ / "metadata.json";
    if (!fs::is_regular_file(meta_path) ||
        fs::file_size(meta_path) > 8u * 1024u * 1024u) {
      error_ = "SQM_CACHE_NO_METADATA";
      return;
    }
    std::ifstream f(meta_path);
    json j = json::parse(f);
    meta_.schema_version = j.at("schema_version").get<int>();
    meta_.coordinate_space = j.at("coordinate_space").get<std::string>();
    meta_.source_width = j.at("source_width").get<int>();
    meta_.source_height = j.at("source_height").get<int>();
    meta_.storage_divisor = j.at("storage_divisor").get<int>();
    meta_.dtype = j.at("dtype").get<std::string>();
    meta_.source_identity_hash =
        j.at("source_identity_hash").get<std::string>();
    meta_.normalized_cache_hash =
        j.at("normalized_cache_hash").get<std::string>();
    meta_.source_quality_config_hash =
        j.at("source_quality_config_hash").get<std::string>();
    meta_.source_quality_cache_hash =
        j.at("source_quality_cache_hash").get<std::string>();
    meta_.proxy_version = j.at("proxy_version").get<int>();
    meta_.cfa_origin_x = j.at("cfa_origin_x").get<int>();
    meta_.cfa_origin_y = j.at("cfa_origin_y").get<int>();
    meta_.streams = j.at("streams").get<std::vector<std::string>>();
    for (const auto &fe : j.at("files")) {
      SourceQualityCacheFileEntry e;
      e.stream = fe.at("stream").get<std::string>();
      e.source_index = fe.at("source_index").get<std::size_t>();
      e.name = fe.at("name").get<std::string>();
      // T1: v1 has sha256, v2 has bytes. Parse whichever is present.
      if (fe.contains("sha256"))
        e.sha256 = fe.at("sha256").get<std::string>();
      if (fe.contains("bytes"))
        e.bytes = fe.at("bytes").get<std::uintmax_t>();
      else if (fs::is_regular_file(root_ / e.name))
        e.bytes = fs::file_size(root_ / e.name);
      meta_.files.push_back(std::move(e));
    }

    if (meta_.schema_version != 1 && meta_.schema_version != 2) {
      error_ = "SQM_CACHE_BAD_SCHEMA"; return;
    }
    if (meta_.dtype != "uint16") { error_ = "SQM_CACHE_BAD_DTYPE"; return; }
    if (!expected_identity_hash.empty() &&
        meta_.source_identity_hash != expected_identity_hash) {
      error_ = "SQM_CACHE_IDENTITY_MISMATCH";
      return;
    }
    if (!expected_config_hash.empty() &&
        meta_.source_quality_config_hash != expected_config_hash) {
      error_ = "SQM_CACHE_CONFIG_MISMATCH";
      return;
    }
    if (cache_manifest_hash(meta_) != meta_.source_quality_cache_hash) {
      error_ = "SQM_CACHE_MANIFEST_HASH_MISMATCH";
      return;
    }
    // T1: size check only (trusted run). v1 caches with sha256 are accepted
    // without re-hashing; a truncated or missing file still fails closed.
    for (const auto &fe : meta_.files) {
      const fs::path p = root_ / fe.name;
      if (!fs::is_regular_file(p) || fs::file_size(p) != fe.bytes) {
        error_ = "SQM_CACHE_FILE_CORRUPT: " + fe.name;
        return;
      }
    }
    // T2: build the O(1) lookup index. meta_.files order is preserved for
    // deterministic metadata identity; the index maps (stream, source_index)
    // to the position in meta_.files.
    file_index_.reserve(meta_.files.size());
    for (std::size_t i = 0; i < meta_.files.size(); ++i) {
      const auto &fe = meta_.files[i];
      file_index_[{fe.stream, fe.source_index}] = i;
    }
    usable_ = true;
  } catch (const std::exception &e) {
    error_ = std::string("SQM_CACHE_METADATA_PARSE: ") + e.what();
    usable_ = false;
  }
}

bool SourceQualityMapCacheReader::has(const std::string &stream,
                                      std::size_t source_index) const {
  // T2: O(1) lookup via the constructor-built index.
  return file_index_.find({stream, source_index}) != file_index_.end();
}

fs::path SourceQualityMapCacheReader::file_path(
    const std::string &stream, std::size_t source_index) const {
  // T2: O(1) lookup via the constructor-built index.
  const auto it = file_index_.find({stream, source_index});
  if (it == file_index_.end())
    throw std::runtime_error("SQM_CACHE_STREAM_FRAME_ABSENT: " + stream);
  return root_ / meta_.files[it->second].name;
}

Matrix2Df SourceQualityMapCacheReader::read_region(const std::string &stream,
                                                   std::size_t source_index,
                                                   int y0, int y1) const {
  return read_rect(stream, source_index, y0, y1, 0, meta_.source_width);
}

Matrix2Df SourceQualityMapCacheReader::read_rect(const std::string &stream,
                                                 std::size_t source_index,
                                                 int y0, int y1, int x0,
                                                 int x1) const {
  if (!usable_) throw std::runtime_error("SQM_CACHE_NOT_USABLE: " + error_);
  y0 = std::max(0, y0);
  y1 = std::min(meta_.source_height, y1);
  x0 = std::max(0, x0);
  x1 = std::min(meta_.source_width, x1);
  if (y1 <= y0 || x1 <= x0) return Matrix2Df(0, 0);
  const int d = meta_.storage_divisor;
  // §30.81 step 3a-2b: seek-read only the storage cells covering the rect.
  const BinWindow w = read_bin_window(
      file_path(stream, source_index), y0 / d, (y1 - 1) / d, x0 / d,
      (x1 - 1) / d);
  if (w.divisor != d || w.storage_w != storage_dim(meta_.source_width, d) ||
      w.storage_h != storage_dim(meta_.source_height, d))
    throw std::runtime_error("SQM_CACHE_BIN_GEOMETRY_MISMATCH: " + stream);
  bin_loads_.fetch_add(1, std::memory_order_relaxed);
  bin_cells_decoded_.fetch_add(
      static_cast<std::uint64_t>(w.win_h) * static_cast<std::uint64_t>(w.win_w),
      std::memory_order_relaxed);
  expanded_floats_.fetch_add(
      static_cast<std::uint64_t>(y1 - y0) * static_cast<std::uint64_t>(x1 - x0),
      std::memory_order_relaxed);

  Matrix2Df out(y1 - y0, x1 - x0);
  for (int y = y0; y < y1; ++y) {
    // Storage cell picked from the ABSOLUTE y (edge-clamp unchanged), then
    // rebased into the window that was actually read.
    const int cy = std::min(w.storage_h - 1, y / d);
    const std::size_t wy = static_cast<std::size_t>(cy - w.cy0) * w.win_w;
    for (int x = x0; x < x1; ++x) {
      const int cx = std::min(w.storage_w - 1, x / d);  // ABSOLUTE x
      const std::size_t wi = wy + static_cast<std::size_t>(cx - w.cx0);
      // Hard-veto cell forces NaN regardless of the value cell (plan 13.5).
      out(y - y0, x - x0) = w.veto[wi]
                                ? std::numeric_limits<float>::quiet_NaN()
                                : dequantize_quality(w.cells[wi]);
    }
  }
  return out;
}

Matrix2Df SourceQualityMapCacheReader::read_full(
    const std::string &stream, std::size_t source_index) const {
  return read_rect(stream, source_index, 0, meta_.source_height, 0,
                   meta_.source_width);
}

// --- Orchestrator -------------------------------------------------------------

SourceQualityMapsBuildResult build_source_quality_map_cache(
    const fs::path &cache_root,
    const registration::RegistrationSamplingPlan &plan,
    VerifiedNormalizedSourceCache &cache,
    const config::AqmhPyramidConfig &pyramid,
    SourceQualityMapCacheConfig cache_cfg, int workers) {
  const std::string normalized_cache_hash = cache.manifest_hash();
  SourceQualityMapCacheWriter writer(cache_root, plan, normalized_cache_hash,
                                     pyramid, cache_cfg);

  SourceQualityMapsBuildResult r;
  r.source_identity_hash = writer.identity_hash();
  r.source_quality_config_hash = writer.config_hash();

  std::vector<const registration::FrameSamplingTransform *> valid;
  for (const auto &f : plan.frames)
    if (f.valid)
      valid.push_back(&f);
  const int nf = static_cast<int>(valid.size());
  int nw = std::max(1, std::min(workers, nf));

  // T3: per-frame metrics collection (optional, when star_max_corners > 0).
  const bool compute_metrics = cache_cfg.star_max_corners > 0;
  std::vector<SourceQualityFrameMetrics> all_metrics;
  if (compute_metrics) all_metrics.resize(nf);

  int max_scales = 0;
  // `failed` is the lock-free fast-path check; `worker_error` is only ever
  // touched inside the sqm_error critical section. A plain read of
  // std::exception_ptr racing with its assignment would be UB.
  std::atomic<bool> failed{false};
  std::exception_ptr worker_error;
  auto record_error = [&] {
#pragma omp critical(sqm_error)
    if (!worker_error)
      worker_error = std::current_exception();
    failed.store(true, std::memory_order_relaxed);
  };

  // One process for each frame: load (own cache clone) -> proxy -> quality
  // maps, buffered locally; then a short critical section flushes the writer.
  // The writer sorts its file list at commit(), so completion order does not
  // affect the committed bytes or source_quality_cache_hash.
  auto process_frame = [&](int idx, VerifiedNormalizedSourceCache &wc) {
    const auto &f = *valid[static_cast<size_t>(idx)];
    const Matrix2Df &source = wc.load(f.source_index);
    const auto proxy = compute_source_quality_proxy_v1(
        source, plan.color_mode, plan.bayer_pattern, plan.cfa_origin_x,
        plan.cfa_origin_y);
    const Matrix2Df &analysis = proxy.proxy_full;

    std::vector<std::pair<std::string, Matrix2Df>> pending;
    QualityScaleMapSink sink = [&](int scale_index, int /*downsample_factor*/,
                                   const Matrix2Df &psi_source_geom) {
      pending.emplace_back("scale_" + std::to_string(scale_index),
                           psi_source_geom);
    };
    const auto maps = compute_source_quality_maps(
        analysis, /*source_valid_mask=*/{}, analysis.cols(), analysis.rows(),
        pyramid, sink);
    pending.emplace_back("composite", maps.q_map);
    pending.emplace_back("artifact", maps.artifact_confidence);

    // writer.put is internally thread-safe (plan §30.72 R4): the heavy
    // downsample/quantize/write runs concurrently; only the file-list
    // update is briefly locked. Only the tiny scalar reduction needs a
    // critical here.
    for (const auto &[stream, mtx] : pending)
      writer.put(stream, f.source_index, mtx);

    // T3: compute per-frame metrics from the already-computed proxy.
    // FrameStarMetrics.wfwhm is computed with ref_star_count=0 (=> wfwhm=fwhm);
    // the weight calculation does not use wfwhm, so this is safe.
    if (compute_metrics) {
      const auto fm = metrics::calculate_frame_metrics(analysis);
      const auto sm = metrics::measure_frame_stars(
          analysis, /*ref_star_count=*/0, cache_cfg.star_max_corners,
          cache_cfg.star_patch_radius);
      SourceQualityFrameMetrics &out = all_metrics[static_cast<size_t>(idx)];
      out.source_index = f.source_index;
      out.frame_id = f.frame_id;
      out.background = fm.background;
      out.noise = fm.noise;
      out.gradient_energy = fm.gradient_energy;
      out.sky_gradient = fm.sky_gradient;
      out.quality_score = fm.quality_score;
      out.fwhm = sm.fwhm;
      out.fwhm_x = sm.fwhm_x;
      out.fwhm_y = sm.fwhm_y;
      out.roundness = sm.roundness;
      out.wfwhm = sm.wfwhm;
      out.star_count = sm.star_count;
    }

#pragma omp critical(sqm_reduce)
    {
      max_scales = std::max(max_scales, maps.diagnostics.computed_scales);
      ++r.frames;
    }
  };

  if (nw <= 1) {
    for (int i = 0; i < nf; ++i)
      process_frame(i, cache);
  } else {
#ifdef _OPENMP
    const size_t worker_mb =
        cache.frame_byte_size() / (1024 * 1024) + 4; // ~2 frames resident
#pragma omp parallel num_threads(nw)
    {
      // Per-worker cache clone: independent LRU, no shared load() state. The
      // clone constructor can throw (budget check); a throw escaping the
      // parallel region would call std::terminate, so catch it here. Every
      // thread must still reach the `omp for` worksharing region below (a
      // skipped one deadlocks the team), so a clone failure just sets `failed`
      // and every iteration becomes a no-op.
      std::optional<VerifiedNormalizedSourceCache> wc;
      try {
        wc.emplace(cache, worker_mb);
      } catch (...) {
        record_error();
      }
#pragma omp for schedule(dynamic, 1)
      for (int i = 0; i < nf; ++i) {
        if (failed.load(std::memory_order_relaxed) || !wc)
          continue;
        try {
          process_frame(i, *wc);
        } catch (...) {
          record_error();
        }
      }
    }
#else
    for (int i = 0; i < nf; ++i)
      process_frame(i, cache);
#endif
  }
  if (worker_error)
    std::rethrow_exception(worker_error);

  const auto meta = writer.commit();
  r.source_quality_cache_hash = meta.source_quality_cache_hash;
  r.streams = meta.streams;
  r.computed_scales = max_scales;

  // T3: write the metrics artifact sorted by source_index.
  if (compute_metrics) {
    std::sort(all_metrics.begin(), all_metrics.end(),
              [](const SourceQualityFrameMetrics &a,
                 const SourceQualityFrameMetrics &b) {
                return a.source_index < b.source_index;
              });
    SourceQualityMetricsArtifact ma;
    ma.schema_version = 1;
    ma.source_identity_hash = r.source_identity_hash;
    ma.source_quality_config_hash = r.source_quality_config_hash;
    ma.normalized_cache_hash = normalized_cache_hash;
    ma.frames = std::move(all_metrics);
    write_source_quality_metrics(cache_root / "source_quality_metrics-v1.json", ma);
  }

  return r;
}

// --- T3: metrics artifact ---------------------------------------------------

void write_source_quality_metrics(
    const fs::path &path,
    const SourceQualityMetricsArtifact &artifact) {
  json j;
  j["schema_version"] = artifact.schema_version;
  j["source_identity_hash"] = artifact.source_identity_hash;
  j["source_quality_config_hash"] = artifact.source_quality_config_hash;
  j["normalized_cache_hash"] = artifact.normalized_cache_hash;
  json frames = json::array();
  for (const auto &f : artifact.frames) {
    frames.push_back({
        {"source_index", f.source_index},
        {"frame_id", f.frame_id},
        {"background", f.background},
        {"noise", f.noise},
        {"gradient_energy", f.gradient_energy},
        {"sky_gradient", f.sky_gradient},
        {"quality_score", f.quality_score},
        {"fwhm", f.fwhm},
        {"fwhm_x", f.fwhm_x},
        {"fwhm_y", f.fwhm_y},
        {"roundness", f.roundness},
        {"wfwhm", f.wfwhm},
        {"star_count", f.star_count},
    });
  }
  j["frames"] = frames;
  core::write_text_atomic(path, j.dump(2));
}

bool load_source_quality_metrics(
    const fs::path &path,
    const std::string &expected_identity_hash,
    const std::string &expected_config_hash,
    const std::string &expected_normalized_cache_hash,
    SourceQualityMetricsArtifact &artifact,
    std::string &error) {
  try {
    if (!fs::is_regular_file(fs::symlink_status(path))) {
      error = "SQM_METRICS_MISSING";
      return false;
    }
    if (fs::file_size(path) > 64u * 1024u * 1024u) {
      error = "SQM_METRICS_TOO_LARGE";
      return false;
    }
    std::ifstream f(path);
    json j = json::parse(f);
    artifact.schema_version = j.at("schema_version").get<int>();
    if (artifact.schema_version != 1) {
      error = "SQM_METRICS_BAD_SCHEMA";
      return false;
    }
    artifact.source_identity_hash =
        j.at("source_identity_hash").get<std::string>();
    artifact.source_quality_config_hash =
        j.at("source_quality_config_hash").get<std::string>();
    artifact.normalized_cache_hash =
        j.at("normalized_cache_hash").get<std::string>();
    if (!expected_identity_hash.empty() &&
        artifact.source_identity_hash != expected_identity_hash) {
      error = "SQM_METRICS_IDENTITY_MISMATCH";
      return false;
    }
    if (!expected_config_hash.empty() &&
        artifact.source_quality_config_hash != expected_config_hash) {
      error = "SQM_METRICS_CONFIG_MISMATCH";
      return false;
    }
    if (!expected_normalized_cache_hash.empty() &&
        artifact.normalized_cache_hash != expected_normalized_cache_hash) {
      error = "SQM_METRICS_CACHE_MISMATCH";
      return false;
    }
    artifact.frames.clear();
    for (const auto &fj : j.at("frames")) {
      SourceQualityFrameMetrics fm;
      fm.source_index = fj.at("source_index").get<std::size_t>();
      fm.frame_id = fj.at("frame_id").get<std::string>();
      fm.background = fj.at("background").get<float>();
      fm.noise = fj.at("noise").get<float>();
      fm.gradient_energy = fj.at("gradient_energy").get<float>();
      fm.sky_gradient = fj.at("sky_gradient").get<float>();
      fm.quality_score = fj.at("quality_score").get<float>();
      fm.fwhm = fj.at("fwhm").get<float>();
      fm.fwhm_x = fj.at("fwhm_x").get<float>();
      fm.fwhm_y = fj.at("fwhm_y").get<float>();
      fm.roundness = fj.at("roundness").get<float>();
      fm.wfwhm = fj.at("wfwhm").get<float>();
      fm.star_count = fj.at("star_count").get<int>();
      artifact.frames.push_back(std::move(fm));
    }
    // Verify sorted by source_index (deterministic identity).
    for (std::size_t i = 1; i < artifact.frames.size(); ++i) {
      if (artifact.frames[i].source_index <=
          artifact.frames[i - 1].source_index) {
        error = "SQM_METRICS_NOT_SORTED";
        return false;
      }
    }
    return true;
  } catch (const std::exception &e) {
    error = std::string("SQM_METRICS_PARSE: ") + e.what();
    return false;
  }
}

}  // namespace tile_compile::reconstruction
