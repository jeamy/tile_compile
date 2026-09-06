#include "tile_compile/reconstruction/source_quality_map_cache.hpp"

#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/reconstruction/source_quality_maps.hpp"
#include "tile_compile/reconstruction/source_quality_proxy.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace tile_compile::reconstruction {

using json = nlohmann::json;

namespace {

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
float nan_value() { return std::numeric_limits<float>::quiet_NaN(); }

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

struct BinContents {
  int storage_w = 0, storage_h = 0, divisor = 0;
  std::size_t source_index = 0;
  std::vector<uint16_t> cells;
  std::vector<uint8_t> veto_cells;
};

BinContents read_bin(const fs::path &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("SQM_CACHE_BIN_MISSING: " + path.string());
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
  const uint32_t schema = rd_u32();
  if (schema != kBinSchema)
    throw std::runtime_error("SQM_CACHE_BIN_BAD_SCHEMA: " + path.string());
  BinContents c;
  c.storage_w = static_cast<int>(rd_u32());
  c.storage_h = static_cast<int>(rd_u32());
  c.divisor = static_cast<int>(rd_u32());
  const uint32_t lo = rd_u32();
  const uint32_t hi = rd_u32();
  c.source_index = static_cast<std::size_t>(lo) |
                   (static_cast<std::size_t>(hi) << 32);
  if (c.storage_w <= 0 || c.storage_h <= 0 || c.divisor <= 0)
    throw std::runtime_error("SQM_CACHE_BIN_BAD_DIMS: " + path.string());
  const std::size_t n = static_cast<std::size_t>(c.storage_w) *
                        static_cast<std::size_t>(c.storage_h);
  std::vector<uint8_t> body(n * 2);
  f.read(reinterpret_cast<char *>(body.data()),
         static_cast<std::streamsize>(body.size()));
  if (static_cast<std::size_t>(f.gcount()) != body.size())
    throw std::runtime_error("SQM_CACHE_BIN_TRUNCATED: " + path.string());
  c.cells.resize(n);
  for (std::size_t i = 0; i < n; ++i)
    c.cells[i] = static_cast<uint16_t>(body[2 * i]) |
                 (static_cast<uint16_t>(body[2 * i + 1]) << 8);
  c.veto_cells.resize(n);
  f.read(reinterpret_cast<char *>(c.veto_cells.data()),
         static_cast<std::streamsize>(n));
  if (static_cast<std::size_t>(f.gcount()) != n)
    throw std::runtime_error("SQM_CACHE_BIN_TRUNCATED_VETO: " + path.string());
  return c;
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
  fs::create_directories(target.parent_path());
  write_bin_atomic(target, sw, sh, d, source_index, cells, veto_cells);

  files_.erase(std::remove_if(files_.begin(), files_.end(),
                              [&](const SourceQualityCacheFileEntry &e) {
                                return e.stream == stream &&
                                       e.source_index == source_index;
                              }),
               files_.end());
  SourceQualityCacheFileEntry e;
  e.stream = stream;
  e.source_index = source_index;
  e.name = name;
  e.sha256 = core::sha256_file(target);
  files_.push_back(std::move(e));
}

SourceQualityCacheMetadata SourceQualityMapCacheWriter::commit() {
  SourceQualityCacheMetadata m;
  m.schema_version = 1;
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
                          {"sha256", fe.sha256}});

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
      e.sha256 = fe.at("sha256").get<std::string>();
      meta_.files.push_back(std::move(e));
    }

    if (meta_.schema_version != 1) { error_ = "SQM_CACHE_BAD_SCHEMA"; return; }
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
    for (const auto &fe : meta_.files) {
      const fs::path p = root_ / fe.name;
      if (!fs::is_regular_file(p) || core::sha256_file(p) != fe.sha256) {
        error_ = "SQM_CACHE_FILE_CORRUPT: " + fe.name;
        return;
      }
    }
    usable_ = true;
  } catch (const std::exception &e) {
    error_ = std::string("SQM_CACHE_METADATA_PARSE: ") + e.what();
    usable_ = false;
  }
}

bool SourceQualityMapCacheReader::has(const std::string &stream,
                                      std::size_t source_index) const {
  for (const auto &fe : meta_.files)
    if (fe.stream == stream && fe.source_index == source_index) return true;
  return false;
}

fs::path SourceQualityMapCacheReader::file_path(
    const std::string &stream, std::size_t source_index) const {
  for (const auto &fe : meta_.files)
    if (fe.stream == stream && fe.source_index == source_index)
      return root_ / fe.name;
  throw std::runtime_error("SQM_CACHE_STREAM_FRAME_ABSENT: " + stream);
}

Matrix2Df SourceQualityMapCacheReader::read_region(const std::string &stream,
                                                   std::size_t source_index,
                                                   int y0, int y1) const {
  if (!usable_) throw std::runtime_error("SQM_CACHE_NOT_USABLE: " + error_);
  y0 = std::max(0, y0);
  y1 = std::min(meta_.source_height, y1);
  if (y1 <= y0) return Matrix2Df(0, 0);
  const BinContents c = read_bin(file_path(stream, source_index));
  const int d = meta_.storage_divisor;
  if (c.divisor != d ||
      c.storage_w != storage_dim(meta_.source_width, d) ||
      c.storage_h != storage_dim(meta_.source_height, d))
    throw std::runtime_error("SQM_CACHE_BIN_GEOMETRY_MISMATCH: " + stream);

  Matrix2Df out(y1 - y0, meta_.source_width);
  for (int y = y0; y < y1; ++y) {
    const int cy = std::min(c.storage_h - 1, y / d);
    for (int x = 0; x < meta_.source_width; ++x) {
      const int cx = std::min(c.storage_w - 1, x / d);
      const std::size_t ci = static_cast<std::size_t>(cy) * c.storage_w + cx;
      // Hard-veto cell forces NaN regardless of the value cell (plan 13.5).
      out(y - y0, x) = c.veto_cells[ci]
                           ? std::numeric_limits<float>::quiet_NaN()
                           : dequantize_quality(c.cells[ci]);
    }
  }
  return out;
}

Matrix2Df SourceQualityMapCacheReader::read_full(
    const std::string &stream, std::size_t source_index) const {
  return read_region(stream, source_index, 0, meta_.source_height);
}

// --- Orchestrator -------------------------------------------------------------

SourceQualityMapsBuildResult build_source_quality_map_cache(
    const fs::path &cache_root,
    const registration::RegistrationSamplingPlan &plan,
    VerifiedNormalizedSourceCache &cache,
    const config::AqmhPyramidConfig &pyramid,
    SourceQualityMapCacheConfig cache_cfg) {
  const std::string normalized_cache_hash = cache.manifest_hash();
  SourceQualityMapCacheWriter writer(cache_root, plan, normalized_cache_hash,
                                     pyramid, cache_cfg);

  SourceQualityMapsBuildResult r;
  r.source_identity_hash = writer.identity_hash();
  r.source_quality_config_hash = writer.config_hash();

  int max_scales = 0;
  for (const auto &f : plan.frames) {
    if (!f.valid) continue;
    const Matrix2Df &source = cache.load(f.source_index);

    const auto proxy = compute_source_quality_proxy_v1(
        source, plan.color_mode, plan.bayer_pattern, plan.cfa_origin_x,
        plan.cfa_origin_y);
    const Matrix2Df &analysis = proxy.proxy_full;

    QualityScaleMapSink sink = [&](int scale_index, int /*downsample_factor*/,
                                   const Matrix2Df &psi_source_geom) {
      writer.put("scale_" + std::to_string(scale_index), f.source_index,
                 psi_source_geom);
    };
    const auto maps = compute_source_quality_maps(
        analysis, /*source_valid_mask=*/{}, analysis.cols(), analysis.rows(),
        pyramid, sink);

    writer.put("composite", f.source_index, maps.q_map);
    writer.put("artifact", f.source_index, maps.artifact_confidence);
    max_scales = std::max(max_scales, maps.diagnostics.computed_scales);
    ++r.frames;
  }

  const auto meta = writer.commit();
  r.source_quality_cache_hash = meta.source_quality_cache_hash;
  r.streams = meta.streams;
  r.computed_scales = max_scales;
  return r;
}

}  // namespace tile_compile::reconstruction
