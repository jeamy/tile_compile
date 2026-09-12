#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/reconstruction/source_quality_map_cache.hpp"
#include "tile_compile/reconstruction/multiband_fusion.hpp"
#include "tile_compile/reconstruction/global_quality.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>

namespace tile_compile::reconstruction {
namespace {
using json=nlohmann::json;
size_t budget_bytes(size_t mb) {
  if (!mb || mb>std::numeric_limits<size_t>::max()/(1024*1024))
    throw std::invalid_argument("SOURCE_QUALITY_INVALID_BUDGET");
  return mb*1024*1024;
}
void validate_sampling(const registration::RegistrationSamplingPlan &sampling) {
  if (sampling.source_identity_hash.empty() || sampling.frames.empty() ||
      sampling.plan_hash!=registration::compute_plan_hash(sampling))
    throw std::invalid_argument("SOURCE_QUALITY_INVALID_SAMPLING_IDENTITY");
  registration::RegistrationSamplingPlan parsed;
  std::string error;
  if (!registration::parse_from_json_string(
          registration::serialize_to_json_string(sampling), parsed, error))
    throw std::invalid_argument("SOURCE_QUALITY_INVALID_SAMPLING_PLAN: " + error);

}
void preflight(const registration::RegistrationSamplingPlan &sampling,
               const GlobalQualityConfig &cfg,size_t mb) {
  const size_t budget_limit=budget_bytes(mb);
  if (sampling.frames.size()>budget_limit/4096)
    throw std::runtime_error("SOURCE_QUALITY_METADATA_MEMORY_BUDGET");
  if (sampling.frames.size()>static_cast<size_t>(std::numeric_limits<int>::max()) ||
      cfg.star_max_corners<1 || cfg.star_patch_radius<1)
    throw std::invalid_argument("SOURCE_QUALITY_INVALID_CONFIG");
  for (float v : {cfg.w_bg,cfg.w_noise,cfg.w_grad,cfg.w_fwhm,cfg.w_roundness,
                  cfg.w_star_count,cfg.clamp_lo,cfg.clamp_hi,cfg.weight_exponent_scale})
    if (!std::isfinite(v)) throw std::invalid_argument("SOURCE_QUALITY_INVALID_CONFIG");
  if (cfg.clamp_lo>cfg.clamp_hi) throw std::invalid_argument("SOURCE_QUALITY_INVALID_CONFIG");
  // Conservative single-frame CPU working estimate, including proxies and
  // image-metric scratch. No frame-count-multiplied full-image allocation.
  auto geometry=sampling;
  geometry.canvas_width_native=sampling.source_width;
  geometry.canvas_height_native=sampling.source_height;
  config::ReconstructionDrizzleConfig resources;
  resources.internal_scale=1;
  resources.pixfrac=1;
  resources.chunk_rows=sampling.source_height;
  resources.memory_budget_mb=mb;
  const uint64_t side=static_cast<uint64_t>(cfg.star_patch_radius)*2+1;
  const uint64_t budget=budget_bytes(mb);
  if (side>std::sqrt(static_cast<long double>(budget/8)))
    throw std::runtime_error("SOURCE_QUALITY_MEMORY_BUDGET");
  const uint64_t scratch=8*1024*1024+side*side*8+
      static_cast<uint64_t>(cfg.star_max_corners)*128+
      static_cast<uint64_t>(sampling.frames.size())*4096;
  if (scratch>budget) throw std::runtime_error("SOURCE_QUALITY_MEMORY_BUDGET");
  plan_drizzle_memory(geometry,resources,128,static_cast<size_t>(scratch));
}
}
std::vector<float> resolve_quality_frame_weights(
    const QualityFrameWeightPlan &quality,
    const registration::RegistrationSamplingPlan &sampling,
    const GlobalQualityConfig &cfg,size_t memory_budget_mb) {
  const size_t budget=budget_bytes(memory_budget_mb);
  if (sampling.frames.size()>budget/4096)
    throw std::runtime_error("SOURCE_QUALITY_METADATA_MEMORY_BUDGET");
  if (quality.frames.size()!=sampling.frames.size())
    throw std::invalid_argument("SOURCE_QUALITY_FRAME_COUNT_MISMATCH");
  validate_sampling(sampling);
  QualityFrameWeightPlan checked;
  std::string error;
  if (!parse_quality_frame_weight_plan(serialize_quality_frame_weight_plan(quality),checked,error) ||
      checked.source_identity_hash!=sampling.source_identity_hash ||
      checked.sampling_plan_hash!=sampling.plan_hash ||
      checked.source_quality_config_hash!=compute_source_quality_config_hash(cfg) ||
      checked.frames.size()!=sampling.frames.size())
    throw std::invalid_argument("SOURCE_QUALITY_PLAN_CONTEXT_MISMATCH: "+error);
  std::map<std::string,const QualityFrameWeight *> by_id;
  for (const auto &f : checked.frames) by_id.emplace(f.frame_id,&f);
  size_t slots=0;
  std::set<size_t> indices;
  std::set<std::string> ids;
  for (const auto &f : sampling.frames) {
    const auto match=by_id.find(f.frame_id);
    if (!indices.insert(f.source_index).second || !ids.insert(f.frame_id).second ||
        f.source_index==std::numeric_limits<size_t>::max() || match==by_id.end() ||
        match->second->model_prediction_factor!=f.model_prediction_factor ||
        match->second->registration_residual_factor!=f.registration_residual_factor)
      throw std::invalid_argument("SOURCE_QUALITY_FRAME_MISMATCH");
    slots=std::max(slots,f.source_index+1);
  }
  if (slots>(budget-sampling.frames.size()*4096)/sizeof(float))
    throw std::runtime_error("SOURCE_QUALITY_WEIGHT_VECTOR_MEMORY_BUDGET");
  std::vector<float> weights(slots,0.0f);
  for (const auto &f : sampling.frames) weights[f.source_index]=by_id.at(f.frame_id)->g_eff;
  return weights;
}
QualityFrameWeightPlan persist_source_quality_artifact(
    const fs::path &path,const registration::RegistrationSamplingPlan &sampling,
    VerifiedNormalizedSourceCache &cache,const GlobalQualityConfig &cfg,size_t mb,
    int workers) {
  preflight(sampling,cfg,mb);
  validate_sampling(sampling);
  if (!cache.matches(sampling)) throw std::invalid_argument("SOURCE_QUALITY_CACHE_CONTEXT_MISMATCH");
  // Plan §30.72 R4: frames 1..n-1 run concurrently, each worker on its own
  // verified cache clone (load() is not thread-safe). Frame 0 stays serial
  // (fixes ref_star_count). Result is bit-identical --- the per-frame metrics
  // are pure functions of the frame.
  const size_t worker_mb = cache.frame_byte_size()/(1024*1024) + 4;
  const auto weights=compute_global_quality_weights(sampling.frames.size(),
      [&](size_t i)->const Matrix2Df & { return cache.load(sampling.frames.at(i).source_index); },
      sampling.color_mode,sampling.bayer_pattern,sampling.cfa_origin_x,sampling.cfa_origin_y,cfg,
      workers,
      [&]()->SourceImageProvider {
        auto wc=std::make_shared<VerifiedNormalizedSourceCache>(cache,worker_mb);
        return [wc,&sampling](size_t i)->const Matrix2Df & {
          return wc->load(sampling.frames.at(i).source_index);
        };
      });
  auto plan=build_quality_frame_weight_plan(sampling,weights,compute_source_quality_config_hash(cfg));
  resolve_quality_frame_weights(plan,sampling,cfg,mb);
  json artifact={{"schema_version",1},{"normalized_cache_hash",cache.manifest_hash()},
                 {"quality_plan",json::parse(serialize_quality_frame_weight_plan(plan))}};
  core::write_text_atomic(path,artifact.dump(2));
  return plan;
}

// T3: load pre-computed metrics from SOURCE_QUALITY_MAPS and compute weights
// without reloading/re-running compute_source_quality_proxy_v1 per frame.
QualityFrameWeightPlan persist_source_quality_artifact(
    const fs::path &path,const registration::RegistrationSamplingPlan &sampling,
    VerifiedNormalizedSourceCache &cache,const GlobalQualityConfig &cfg,
    const fs::path &metrics_path,
    size_t mb,int workers) {
  preflight(sampling,cfg,mb);
  validate_sampling(sampling);
  if (!cache.matches(sampling)) throw std::invalid_argument("SOURCE_QUALITY_CACHE_CONTEXT_MISMATCH");

  // Load the metrics artifact. The config hash is NOT validated here because
  // the SQM config hash (compute_scale_quality_config_hash) differs from the
  // GQ config hash (compute_source_quality_config_hash). The GQ config is
  // validated separately through the QualityFrameWeightPlan. The identity
  // and cache hash are sufficient to ensure the metrics are from the same
  // source data.
  SourceQualityMetricsArtifact ma;
  std::string merr;
  if (!load_source_quality_metrics(metrics_path,
      compute_source_quality_identity_hash(sampling,cache.manifest_hash()),
      /*expected_config_hash=*/"",
      cache.manifest_hash(),ma,merr))
    throw std::runtime_error("SOURCE_QUALITY_METRICS_UNUSABLE: "+merr);

  // Map metrics to plan frame order (by source_index slot).
  std::size_t slots=0;
  for (const auto &f:sampling.frames)
    slots=std::max(slots,f.source_index+1);
  std::vector<FrameMetrics> frame_metrics(slots);
  std::vector<metrics::FrameStarMetrics> star_metrics(slots);
  // Build a lookup by source_index.
  std::map<std::size_t,std::size_t> by_idx;
  for (std::size_t i=0;i<ma.frames.size();++i)
    by_idx[ma.frames[i].source_index]=i;
  for (const auto &f:sampling.frames) {
    auto it=by_idx.find(f.source_index);
    if (it==by_idx.end())
      throw std::invalid_argument("SOURCE_QUALITY_METRICS_FRAME_MISSING");
    const auto &fm=ma.frames[it->second];
    frame_metrics[f.source_index]={fm.background,fm.noise,
        fm.gradient_energy,fm.sky_gradient,fm.quality_score};
    star_metrics[f.source_index]={fm.fwhm,fm.fwhm_x,fm.fwhm_y,
        fm.roundness,fm.wfwhm,fm.star_count};
  }

  const auto weights=compute_global_quality_weights_from_metrics(
      frame_metrics,star_metrics,cfg);
  auto plan=build_quality_frame_weight_plan(sampling,weights,compute_source_quality_config_hash(cfg));
  resolve_quality_frame_weights(plan,sampling,cfg,mb);
  json artifact={{"schema_version",1},{"normalized_cache_hash",cache.manifest_hash()},
                 {"quality_plan",json::parse(serialize_quality_frame_weight_plan(plan))},
                 {"metrics_source",metrics_path.filename().string()}};
  core::write_text_atomic(path,artifact.dump(2));
  return plan;
}
QualityFrameWeightPlan load_source_quality_artifact(
    const fs::path &path,const registration::RegistrationSamplingPlan &sampling,
    const VerifiedNormalizedSourceCache &cache,const GlobalQualityConfig &cfg,size_t mb) {
  const size_t budget=budget_bytes(mb);
  if (sampling.frames.size()>budget/4096)
    throw std::runtime_error("SOURCE_QUALITY_METADATA_MEMORY_BUDGET");
  validate_sampling(sampling);
  if (!cache.matches(sampling) || !fs::is_regular_file(fs::symlink_status(path)) ||
      fs::file_size(path)>std::min<size_t>(16*1024*1024,budget/16))
    throw std::invalid_argument("SOURCE_QUALITY_INVALID_ARTIFACT");
  std::ifstream file(path);
  const auto artifact=json::parse(file);
  if (artifact.at("schema_version")!=1 || artifact.at("normalized_cache_hash")!=cache.manifest_hash())
    throw std::invalid_argument("SOURCE_QUALITY_PREDECESSOR_MISMATCH");
  QualityFrameWeightPlan plan;
  std::string error;
  if (!parse_quality_frame_weight_plan(artifact.at("quality_plan").dump(),plan,error))
    throw std::invalid_argument("SOURCE_QUALITY_INVALID_PLAN: "+error);
  resolve_quality_frame_weights(plan,sampling,cfg,mb);
  return plan;
}
DrizzleStoreResult persist_forward_drizzle_from_predecessors(
    const fs::path &store_root,const fs::path &quality_artifact,
    const registration::RegistrationSamplingPlan &sampling,
    VerifiedNormalizedSourceCache &cache,const GlobalQualityConfig &quality_cfg,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const fs::path &source_quality_cache_root, int workers) {
  const size_t mb=drizzle_cfg.memory_budget_mb ? drizzle_cfg.memory_budget_mb : 512;
  const auto quality=load_source_quality_artifact(quality_artifact,sampling,cache,quality_cfg,mb);
  const auto weights=resolve_quality_frame_weights(quality,sampling,quality_cfg,mb);

  // M5: optionally consume the source composite Q-maps as Q_composite_f,c(q).
  // (Fine/Medium's scale_0/scale_1 streams are wired in a later M6 batch.)
  DrizzleStorePredecessors predecessors{cache.manifest_hash(),quality.plan_hash,{}};
  FrameQualityProvider quality_of;  // null => Q_composite = 1.0 (unchanged Raw)
  FrameQualityRectProvider quality_rect_of;  // banded variant (A2)
  std::unique_ptr<SourceQualityMapCacheReader> qreader;
  if (!source_quality_cache_root.empty()) {
    qreader=std::make_unique<SourceQualityMapCacheReader>(
        source_quality_cache_root,
        compute_source_quality_identity_hash(sampling,cache.manifest_hash()),
        /*expected_config_hash=*/"");
    if (!qreader->usable())
      throw std::runtime_error("FORWARD_DRIZZLE_SOURCE_QUALITY_CACHE_UNUSABLE: "+
                               qreader->error());
    predecessors.source_quality_cache_hash=
        qreader->metadata().source_quality_cache_hash;
    quality_of=[reader=qreader.get(),buf=Matrix2Df(),
                idx=std::size_t(-1)](std::size_t source_index) mutable
        -> FrameQualityMaps {
      if (source_index!=idx) {
        buf=reader->read_full("composite",source_index);
        idx=source_index;
      }
      return FrameQualityMaps{&buf,nullptr,nullptr};
    };
    // A2: same composite stream, rectangle reads (the rect is the frame's
    // scan box; an empty rect is a pure existence probe).
    quality_rect_of=[reader=qreader.get(),buf=Matrix2Df()](
        std::size_t si,int y0,int y1,int x0,int x1) mutable
        -> FrameQualityMaps {
      const int sh=reader->metadata().source_height;
      const int sw=reader->metadata().source_width;
      if (y1<0) y1=sh;
      if (x1<0) x1=sw;
      y0=std::clamp(y0,0,sh); y1=std::clamp(y1,y0,sh);
      x0=std::clamp(x0,0,sw); x1=std::clamp(x1,x0,sw);
      buf=reader->read_rect("composite",si,y0,y1,x0,x1);
      FrameQualityMaps m{&buf,nullptr,nullptr,nullptr};
      m.y_origin=y0;
      m.x_origin=x0;
      return m;
    };
  }

  // A1: banded source reads through the verified cache's read_rect.
  const SourceImageRectProvider source_rect_of=
      [&cache](std::size_t index,int y0,int y1,int x0,int x1) {
        return cache.read_rect(index,y0,y1,x0,x1);
      };
  return persist_forward_drizzle_uniform_and_raw(store_root,sampling,
      [&](size_t index)->const Matrix2Df & { return cache.load(index); },
      drizzle_cfg,clipping_cfg,subdivision,weights,predecessors,quality_of,
      workers,source_rect_of,quality_rect_of);
}

MultibandStoreContract multiband_store_contract_from_config(
    const config::ReconstructionMultibandConfig &cfg) {
  MultibandStoreContract c;
  c.enabled=cfg.enabled;
  c.levels=cfg.levels;
  c.fine_quality_exponent=cfg.fine_quality_exponent;
  c.medium_quality_exponent=cfg.medium_quality_exponent;
  c.alpha.alpha_cap=cfg.alpha_cap;
  c.alpha.min_effective_samples=cfg.min_effective_samples;
  c.alpha.full_effective_samples=cfg.full_effective_samples;
  c.confidence.min_quality_separation=cfg.min_quality_separation;
  c.confidence.full_quality_separation=cfg.full_quality_separation;
  // Energy guard and the remaining confidence edges are not config-exposed
  // yet; their defaults still enter multiband_config_hash.
  return c;
}

namespace {
// One profile, rows [y0,y1), from an already-verified generation directory.
ForwardDrizzleUniformResult read_store_profile_region(
    const fs::path &gen,const DrizzleStoreIdentity &id,const std::string &profile,
    int y0,int y1,size_t budget_mb) {
  ForwardDrizzleUniformResult r;
  r.color_mode=id.color_mode;
  r.internal_width=id.width;
  r.internal_height=y1-y0;
  auto rd=[&](const char *ch){
    return read_drizzle_profile_region_preverified(gen,id,profile,ch,0,y0,id.width,y1-y0,budget_mb);
  };
  if (id.color_mode==ColorMode::MONO) r.L=rd("L");
  else { r.R=rd("R"); r.G=rd("G"); r.B=rd("B"); }
  return r;
}
std::vector<float> read_store_alpha_map_region(
    const fs::path &gen,const DrizzleStoreIdentity &id,const std::string &name,
    int y0,int y1,size_t budget_mb) {
  return read_drizzle_profile_region_preverified(gen,id,name,"X",0,y0,id.width,y1-y0,budget_mb).value;
}

// Combine up to 3 channel planes into the fixed working luminance
// (kWorkingLumaDefinition) over an n-pixel block: MONO -> channel 0 as-is;
// OSC -> 0.25 R + 0.5 G + 0.25 B. Luma-supported only where EVERY active
// channel is finite and supported.
void combine_luma(const std::vector<const std::vector<float> *> &vals,
                  const std::vector<const std::vector<uint8_t> *> &sups,int nch,
                  std::size_t n,std::vector<float> &luma,
                  std::vector<uint8_t> &sup) {
  luma.assign(n,std::numeric_limits<float>::quiet_NaN());
  sup.assign(n,0u);
  const double wgt[3]={nch==1?1.0:kWorkingLumaWeightsOsc[0],
                       kWorkingLumaWeightsOsc[1],kWorkingLumaWeightsOsc[2]};
  for (std::size_t i=0;i<n;++i) {
    double acc=0.0; bool ok=true;
    for (int c=0;c<nch;++c) {
      if (!(*sups[c])[i] || !std::isfinite((*vals[c])[i])) { ok=false; break; }
      acc+=wgt[c]*(*vals[c])[i];
    }
    if (ok) { luma[i]=static_cast<float>(acc); sup[i]=1u; }
  }
}
// N6 (redundant-reload analysis): helpers for the rolling fusion window.
// fuse_multiband_store_to_image reads [y0-halo, y1+halo) per chunk, so the
// 2*halo border rows of every stream are decoded twice per boundary. The
// helpers below slide a buffered window: drop the rows before the new start,
// read only the missing tail rows, append them --- each stored row is read
// exactly once.
void profile_plane_drop_rows(ProfilePlane &p,int drop) {
  if (drop<=0||p.empty()) return;
  const std::size_t n=static_cast<std::size_t>(drop)*static_cast<std::size_t>(p.width);
  auto shift=[&](auto &v){
    if (v.size()<n){ v.clear(); return; }
    std::move(v.begin()+static_cast<std::ptrdiff_t>(n),v.end(),v.begin());
    v.resize(v.size()-n);
  };
  shift(p.value); shift(p.weight_sum); shift(p.n_eff); shift(p.support);
  p.height=std::max(0,p.height-drop);
}
void profile_plane_append_rows(ProfilePlane &p,const ProfilePlane &tail) {
  if (tail.empty()) return;
  auto app=[](auto &dst,const auto &src){ dst.insert(dst.end(),src.begin(),src.end()); };
  app(p.value,tail.value); app(p.weight_sum,tail.weight_sum);
  app(p.n_eff,tail.n_eff); app(p.support,tail.support);
  p.width=tail.width;
  p.height+=tail.height;
}
void uniform_result_drop_rows(ForwardDrizzleUniformResult &r,int drop) {
  if (drop<=0) return;
  if (r.color_mode==ColorMode::MONO) profile_plane_drop_rows(r.L,drop);
  else { profile_plane_drop_rows(r.R,drop); profile_plane_drop_rows(r.G,drop);
         profile_plane_drop_rows(r.B,drop); }
  r.internal_height=std::max(0,r.internal_height-drop);
}
void uniform_result_append_rows(ForwardDrizzleUniformResult &r,
                                const ForwardDrizzleUniformResult &tail) {
  if (tail.internal_height<=0) return;
  if (r.color_mode==ColorMode::MONO) profile_plane_append_rows(r.L,tail.L);
  else { profile_plane_append_rows(r.R,tail.R); profile_plane_append_rows(r.G,tail.G);
         profile_plane_append_rows(r.B,tail.B); }
  r.color_mode=tail.color_mode;
  r.internal_width=tail.internal_width;
  r.internal_height+=tail.internal_height;
}
void flat_rows_drop(std::vector<float> &v,int drop,int w) {
  if (drop<=0) return;
  const std::size_t n=static_cast<std::size_t>(drop)*static_cast<std::size_t>(w);
  if (v.size()<n){ v.clear(); return; }
  std::move(v.begin()+static_cast<std::ptrdiff_t>(n),v.end(),v.begin());
  v.resize(v.size()-n);
}
}  // namespace

MultibandStoreBuildResult persist_multiband_store_from_predecessors(
    const fs::path &store_root,const fs::path &quality_artifact,
    const registration::RegistrationSamplingPlan &sampling,
    VerifiedNormalizedSourceCache &cache,const GlobalQualityConfig &quality_cfg,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clipping_cfg,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    const fs::path &source_quality_cache_root,
    const ForwardDrizzleSubdivisionParams &subdivision,
    const std::string &acceleration_backend, int workers) {
  if (source_quality_cache_root.empty())
    throw std::invalid_argument("MULTIBAND_REQUIRES_SOURCE_QUALITY_CACHE");
  const auto contract=multiband_store_contract_from_config(multiband_cfg);
  if (!contract.enabled)
    throw std::invalid_argument("MULTIBAND_DISABLED_IN_CONFIG");

  const size_t mb=drizzle_cfg.memory_budget_mb ? drizzle_cfg.memory_budget_mb : 512;
  const auto quality=load_source_quality_artifact(quality_artifact,sampling,cache,quality_cfg,mb);
  const auto weights=resolve_quality_frame_weights(quality,sampling,quality_cfg,mb);

  DrizzleStorePredecessors predecessors{cache.manifest_hash(),quality.plan_hash,{}};
  auto qreader=std::make_unique<SourceQualityMapCacheReader>(
      source_quality_cache_root,
      compute_source_quality_identity_hash(sampling,cache.manifest_hash()),
      /*expected_config_hash=*/"");
  if (!qreader->usable())
    throw std::runtime_error("MULTIBAND_SOURCE_QUALITY_CACHE_UNUSABLE: "+qreader->error());
  predecessors.source_quality_cache_hash=qreader->metadata().source_quality_cache_hash;

  const bool need_medium=contract.levels>=2;
  // §30.81 step 3a-2: decode Q maps for only the source rectangle
  // [y0,y1) x [x0,x1) the caller asks for (its records' exact source bbox).
  // A negative y1/x1 means the full extent; y0==y1 (or x0==x1) is a pure
  // existence probe -- no decode, but an existing stream still returns a
  // non-null (empty) pointer so `need_qc` etc. resolve. The `comp`/`s0`/...
  // members are reused allocations, NOT a cache: the rect differs on every
  // call under the tile-outer/frame-inner loop, so there is nothing to hit.
  FrameQualityRectProvider quality_of=
      [reader=qreader.get(),need_medium,
       comp=Matrix2Df(),s0=Matrix2Df(),s1=Matrix2Df(),art=Matrix2Df()](
          std::size_t si,int y0,int y1,int x0,int x1) mutable -> FrameQualityMaps {
    const int sh=reader->metadata().source_height;
    const int sw=reader->metadata().source_width;
    if (y1<0) y1=sh;
    if (x1<0) x1=sw;
    y0=std::clamp(y0,0,sh); y1=std::clamp(y1,y0,sh);
    x0=std::clamp(x0,0,sw); x1=std::clamp(x1,x0,sw);
    const bool have_s0=reader->has("scale_0",si);
    const bool have_s1=need_medium && reader->has("scale_1",si);
    const bool have_art=reader->has("artifact",si);
    // composite is mandatory; finer scale + artifact may be absent for a small
    // image -- a missing map is a null pointer (weight degrades), not a fault.
    comp=reader->read_rect("composite",si,y0,y1,x0,x1);
    if (have_s0) s0=reader->read_rect("scale_0",si,y0,y1,x0,x1);
    if (have_s1) s1=reader->read_rect("scale_1",si,y0,y1,x0,x1);
    if (have_art) art=reader->read_rect("artifact",si,y0,y1,x0,x1);
    FrameQualityMaps m{&comp, have_s0?&s0:nullptr, have_s1?&s1:nullptr,
                       have_art?&art:nullptr};
    m.y_origin=y0;
    m.x_origin=x0;
    return m;
  };

  const auto source_of=[&](size_t index)->const Matrix2Df & { return cache.load(index); };
  // A1: banded source reads. VerifiedNormalizedSourceCache::read_rect loads
  // only the covering rows, so each (stripe/band, frame) touches just its
  // inverse-mapped scan box instead of decoding the full frame.
  const SourceImageRectProvider source_rect_of=
      [&cache](std::size_t index,int y0,int y1,int x0,int x1) {
        return cache.read_rect(index,y0,y1,x0,x1);
      };
  const auto build=[&](const ForwardDrizzleCudaOptions &cuda){
    return persist_forward_drizzle_multiband(
        store_root,sampling,source_of,drizzle_cfg,clipping_cfg,contract,
        quality_of,subdivision,weights,predecessors,cuda,workers,
        source_rect_of);
  };

  MultibandStoreBuildResult out;
  // Plan 19: a CUDA attempt is made only when the resolved backend is "cuda"
  // AND there is something to attempt --- a real device path (slice 2) or an
  // armed fault injection (the restart-contract test). Anything else runs the
  // CPU reference path directly.
  const bool want_cuda=acceleration_backend=="cuda";
  const bool can_attempt_cuda=want_cuda &&
      (forward_drizzle_cuda_runtime_available() ||
       forward_drizzle_cuda_fault_after_chunks()>=0);
  if (want_cuda && !can_attempt_cuda)
    out.cuda_fallback_reason="forward_drizzle_cuda_unavailable";

  if (can_attempt_cuda) {
    try {
      ForwardDrizzleCudaOptions cuda; cuda.attempt=true;
      out.store=build(cuda);
      // plan 19.6.2: local-warp frames rode the hybrid CPU-geometry ->
      // GPU-rasterization path; label the committed store accordingly.
      out.backend_used=
          out.store.cuda_timing.hybrid_local_frames>0 ? "cuda_hybrid" : "cuda";
    } catch (const ForwardDrizzleCudaError &e) {
      // Plan 19.4: the uncommitted generation is already discarded by
      // StoreWriter's destructor; restart the ENTIRE build on the CPU
      // reference path. Not recursive --- a failing CPU restart propagates.
      out.cuda_fallback_reason=e.what();
      out.store=build({});
      out.backend_used="cpu";
    }
  } else {
    out.store=build({});
    out.backend_used="cpu";
  }
  out.identity=out.store.identity;  // the identity actually written
  out.q_bin_loads=qreader->bin_loads();
  out.q_bin_cells_decoded=qreader->bin_cells_decoded();
  out.q_expanded_floats=qreader->expanded_floats();
  return out;
}

fs::path MultibandCandidateSpool::plane_path(const std::string &candidate,
                                            int c) const {
  return dir / (candidate + "_" + std::to_string(c) + ".f32");
}

MultibandFusionMemoryPlan plan_multiband_fusion_memory(
    int width,int height,int nch,int levels,int chunk_rows,int halo_rows,
    bool with_candidate_luma,bool with_candidate_spool,std::size_t budget_bytes) {
  MultibandFusionMemoryPlan p;
  p.budget_bytes=budget_bytes;
  if (width<=0||height<=0||nch<=0||levels<=0||chunk_rows<=0) return p;  // !fits
  const auto sat_mul=[](std::size_t a,std::size_t b)->std::size_t{
    if (a!=0 && b>std::numeric_limits<std::size_t>::max()/a)
      return std::numeric_limits<std::size_t>::max();
    return a*b;
  };
  const auto sat_add=[](std::size_t a,std::size_t b)->std::size_t{
    return (b>std::numeric_limits<std::size_t>::max()-a)
        ? std::numeric_limits<std::size_t>::max() : a+b;
  };
  const std::size_t n=sat_mul(static_cast<std::size_t>(width),
                              static_cast<std::size_t>(height));
  const std::size_t nc=static_cast<std::size_t>(nch);
  const std::size_t lv=static_cast<std::size_t>(levels);
  const std::size_t stripe_rows=static_cast<std::size_t>(chunk_rows)+
      2u*static_cast<std::size_t>(std::max(0,halo_rows));
  const std::size_t stripe_px=sat_mul(stripe_rows,
                                      static_cast<std::size_t>(width));

  // Final X_out held whole: nch planes * 4 bytes.
  p.final_image_bytes=sat_mul(sat_mul(nc,n),4u);
  // One resident fusion stripe: U/R/F/M (value+support, all nch) + the fused
  // `sub` (value+support+levels alpha) + 3 alpha maps + combine_luma scratch.
  // Generous per-pixel upper bound: nch*40 + levels*8 + 64 bytes.
  p.stripe_working_bytes=sat_mul(stripe_px,
      sat_add(sat_add(sat_mul(nc,40u),sat_mul(lv,8u)),64u));
  // Candidate working-luminance buffers held whole (plan 15 selection needs the
  // full field): 3 luma (4) + support (1) + levels alpha (4).
  p.candidate_luma_bytes=with_candidate_luma
      ? sat_mul(n,sat_add(13u,sat_mul(lv,4u))) : 0u;
  // Spool write path is zero-copy from the stripe buffers; only OS write
  // buffering is transient. Budget up to 64 rows per (candidate, channel).
  p.spool_stripe_bytes=with_candidate_spool
      ? sat_mul(sat_mul(sat_mul(3u,nc),
                        sat_mul(static_cast<std::size_t>(width),
                                std::min<std::size_t>(64u,stripe_rows))),4u)
      : 0u;
  // Delivery: the runner reads back one spooled plane at a time for FITS export.
  p.delivery_readback_bytes=with_candidate_spool ? sat_mul(n,4u) : 0u;

  std::size_t sum=0;
  for (std::size_t v : {p.final_image_bytes,p.stripe_working_bytes,
                        p.candidate_luma_bytes,p.spool_stripe_bytes,
                        p.delivery_readback_bytes})
    sum=sat_add(sum,v);
  // 5% of the working set, with a modest absolute floor that is itself capped
  // at the working set so a tiny image under a tiny explicit budget is not
  // rejected by the margin alone.
  p.margin_bytes=std::max<std::size_t>(
      sum/20u,std::min<std::size_t>(std::size_t(64)<<20,sum));
  p.estimated_peak_bytes=sat_add(sum,p.margin_bytes);
  p.fits=p.estimated_peak_bytes<=budget_bytes;

  // Plan 11.13(2) + 11.11 temp space: the spool writes 3 * nch * N * 4 bytes to
  // the temp filesystem. `required_free_temp` here uses the 2 GiB floor; the
  // caller refines it with the real filesystem capacity and fills
  // available_temp_bytes / temp_space_ok.
  if (with_candidate_spool) {
    p.spool_temp_bytes=sat_mul(sat_mul(sat_mul(3u,nc),n),4u);
    p.required_free_temp_bytes=sat_add(
        static_cast<std::size_t>(static_cast<long double>(p.spool_temp_bytes)*1.20L),
        std::size_t(2)<<30);
  }
  return p;
}

std::vector<float> read_candidate_spool_plane(
    const MultibandCandidateSpool &spool,const std::string &candidate,int c) {
  if (!spool.populated || c<0 || c>=spool.nch || spool.width<=0 ||
      spool.height<=0)
    throw std::runtime_error("SPOOL_PLANE_NOT_AVAILABLE");
  const std::size_t n=static_cast<std::size_t>(spool.width)*spool.height;
  const auto path=spool.plane_path(candidate,c);
  std::error_code ec;
  if (fs::file_size(path,ec)!=static_cast<std::uintmax_t>(n*sizeof(float)))
    throw std::runtime_error("SPOOL_PLANE_SIZE_MISMATCH");
  std::ifstream in(path.string(),std::ios::binary);
  if (!in) throw std::runtime_error("SPOOL_PLANE_OPEN_FAILED");
  std::vector<float> plane(n);
  in.read(reinterpret_cast<char *>(plane.data()),
          static_cast<std::streamsize>(n*sizeof(float)));
  if (!in) throw std::runtime_error("SPOOL_PLANE_READ_FAILED");
  return plane;
}

long long fuse_multiband_store_to_image(
    const fs::path &store_root,const DrizzleStoreIdentity &identity,
    const fs::path &final_image_path,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    int chunk_rows,size_t memory_budget_mb,
    MultibandCandidateLuma *candidates_out,
    MultibandCandidateSpool *spool_out,
    MultibandFusionMemoryPlan *mem_plan_out) {
  if (identity.multiband_levels<1)
    throw std::invalid_argument("FUSE_STORE_NOT_A_MULTIBAND_IDENTITY");
  // The caller must pass the config the store was built with. Guard the one
  // field that changes the plane set / band assignment; the alpha/guard edges
  // are trusted from `multiband_cfg` (in the runner flow write and fuse share
  // one config, so they cannot drift).
  if (multiband_cfg.enabled && multiband_cfg.levels!=identity.multiband_levels)
    throw std::invalid_argument("FUSE_STORE_MULTIBAND_LEVELS_MISMATCH");
  const bool need_medium=identity.multiband_levels>=2;
  // Plan 11.13: `memory_budget_mb == 0` is "unset" -> internal 256 MiB floor.
  // A non-zero value is an EXPLICIT budget and is honoured verbatim (no silent
  // raising); the phase fails closed below if the pre-plan does not fit it.
  const size_t budget=memory_budget_mb ? memory_budget_mb : size_t(256);
  // Verify the generation ONCE (a full rehash), then read every stripe from
  // the verified directory --- otherwise the striped reads below would rehash
  // the whole store O(H/chunk) times.
  const auto verified=verify_drizzle_profile_store(store_root,identity);
  if (!verified.usable) throw std::runtime_error(verified.error);
  const fs::path &gen=verified.generation_dir;
  const auto contract=multiband_store_contract_from_config(multiband_cfg);
  config::ReconstructionMultibandConfig fcfg=multiband_cfg;
  fcfg.levels=identity.multiband_levels;
  const int W=identity.width, H=identity.height;
  const int chunk=std::max(1,std::min(H,chunk_rows>0?chunk_rows:64));
  const int halo=multiband_fusion_halo_rows(fcfg.levels);
  const bool mono=identity.color_mode==ColorMode::MONO;
  const std::size_t N=static_cast<std::size_t>(W)*H;
  const int nch_plan=mono?1:3;

  // Plan 11.13(1)+(4): pre-plan the whole MULTIBAND working set and fail closed
  // BEFORE the first large allocation. Nothing durable has been created yet ---
  // only the (already committed) store was read --- so a throw here leaves the
  // prior valid generation and any prior outputs untouched.
  MultibandFusionMemoryPlan mem_plan=plan_multiband_fusion_memory(
      W,H,nch_plan,fcfg.levels,chunk,halo,
      candidates_out!=nullptr,spool_out!=nullptr,
      static_cast<std::size_t>(budget)*1024*1024);
  // Plan 11.11 temp space: refine required_free_temp with the real capacity of
  // the spool filesystem and record what is actually available.
  if (spool_out) {
    std::error_code ec;
    if (!fs::is_directory(spool_out->dir,ec)) {
      if (mem_plan_out) *mem_plan_out=mem_plan;
      throw std::runtime_error("FUSE_STORE_SPOOL_DIR_MISSING");
    }
    const auto sp=fs::space(spool_out->dir,ec);
    mem_plan.available_temp_bytes=ec ? 0u : static_cast<std::size_t>(sp.available);
    const std::size_t cap=ec ? 0u : static_cast<std::size_t>(sp.capacity);
    const std::size_t reserve=std::max<std::size_t>(
        std::size_t(2)<<30,static_cast<std::size_t>(cap/20));
    mem_plan.required_free_temp_bytes=
        static_cast<std::size_t>(
            static_cast<long double>(mem_plan.spool_temp_bytes)*1.20L)+reserve;
    mem_plan.temp_space_ok=
        mem_plan.available_temp_bytes>=mem_plan.required_free_temp_bytes;
  }
  if (mem_plan_out) *mem_plan_out=mem_plan;
  if (!mem_plan.fits)
    throw std::runtime_error("MULTIBAND_MEMORY_BUDGET");
  if (!mem_plan.temp_space_ok)
    throw std::runtime_error("MULTIBAND_TEMP_SPACE");

  // Only the final image is held whole (1 plane MONO / 3 planes OSC); the
  // store reads are striped so peak resident input is O(chunk + 2*halo) rows,
  // independent of the full frame size --- the path scales to large mosaics
  // (M31/M42 full-res) the same as to the test fixtures.
  std::array<std::vector<float>,3> out;
  const int nch=mono?1:3;
  for (int c=0;c<nch;++c) out[c].assign(N,std::numeric_limits<float>::quiet_NaN());
  long long pixels_supported=0;

  if (candidates_out) {
    *candidates_out={};
    candidates_out->width=W;
    candidates_out->height=H;
    candidates_out->uniform_luma.assign(N,std::numeric_limits<float>::quiet_NaN());
    candidates_out->raw_luma.assign(N,std::numeric_limits<float>::quiet_NaN());
    candidates_out->multiband_luma.assign(N,std::numeric_limits<float>::quiet_NaN());
    candidates_out->uniform_support.assign(N,0u);
    candidates_out->alpha_final_by_band.assign(
        static_cast<std::size_t>(fcfg.levels),{});
  }

  // Plan 11.13(2): candidate channels are streamed stripe-wise to a scratch
  // directory, not held whole. One append stream per (candidate, channel); the
  // stripe loop below writes exactly the `core` rows in strictly increasing y
  // order, so each file ends up a row-major width*height plane.
  static constexpr const char *kCandNames[3]={"uniform","raw","multiband"};
  std::array<std::array<std::ofstream,3>,3> spool_os;
  if (spool_out) {
    spool_out->width=W; spool_out->height=H; spool_out->nch=nch; spool_out->mono=mono;
    spool_out->populated=false;
    // (spool_out->dir existence + temp-space were checked pre-allocation above.)
    for (int k=0;k<3;++k)
      for (int c=0;c<nch;++c) {
        const auto p=spool_out->plane_path(kCandNames[k],c);
        spool_os[static_cast<std::size_t>(k)][static_cast<std::size_t>(c)].open(
            p.string(),std::ios::binary|std::ios::trunc);
        if (!spool_os[static_cast<std::size_t>(k)][static_cast<std::size_t>(c)])
          throw std::runtime_error("FUSE_STORE_SPOOL_OPEN_FAILED");
      }
  }

  // N6: rolling window over the profile/alpha streams. Consecutive chunks
  // overlap by 2*halo rows; instead of re-reading the halo band per chunk,
  // slide the buffered window --- drop the rows before ys and read only the
  // missing tail [buf_y1, ye). Every stored row is decoded exactly once.
  ForwardDrizzleUniformResult U,R,F,M;
  std::vector<float> a_sep,a_art,a_reg;
  int buf_y0=0,buf_y1=0;   // rows [buf_y0, buf_y1) currently buffered
  for (int y0=0;y0<H;y0+=chunk) {
    const int y1=std::min(H,y0+chunk);
    const int ys=std::max(0,y0-halo), ye=std::min(H,y1+halo);
    const int sub_h=ye-ys;
    const bool window_empty=(buf_y1<=buf_y0);
    if (window_empty || ys<buf_y0 || ys>buf_y1) {
      U=read_store_profile_region(gen,identity,"uniform",ys,ye,budget);
      R=read_store_profile_region(gen,identity,"raw",ys,ye,budget);
      F=read_store_profile_region(gen,identity,"fine",ys,ye,budget);
      M=need_medium
          ? read_store_profile_region(gen,identity,"medium",ys,ye,budget)
          : ForwardDrizzleUniformResult{};
      a_sep=read_store_alpha_map_region(gen,identity,"alpha_separation",ys,ye,budget);
      a_art=read_store_alpha_map_region(gen,identity,"alpha_artifact",ys,ye,budget);
      a_reg=read_store_alpha_map_region(gen,identity,"alpha_registration",ys,ye,budget);
    } else {
      const int drop=ys-buf_y0;
      uniform_result_drop_rows(U,drop); uniform_result_drop_rows(R,drop);
      uniform_result_drop_rows(F,drop); uniform_result_drop_rows(M,drop);
      flat_rows_drop(a_sep,drop,W); flat_rows_drop(a_art,drop,W);
      flat_rows_drop(a_reg,drop,W);
      if (ye>buf_y1) {
        uniform_result_append_rows(U,read_store_profile_region(gen,identity,"uniform",buf_y1,ye,budget));
        uniform_result_append_rows(R,read_store_profile_region(gen,identity,"raw",buf_y1,ye,budget));
        uniform_result_append_rows(F,read_store_profile_region(gen,identity,"fine",buf_y1,ye,budget));
        if (need_medium)
          uniform_result_append_rows(M,read_store_profile_region(gen,identity,"medium",buf_y1,ye,budget));
        const auto t_sep=read_store_alpha_map_region(gen,identity,"alpha_separation",buf_y1,ye,budget);
        const auto t_art=read_store_alpha_map_region(gen,identity,"alpha_artifact",buf_y1,ye,budget);
        const auto t_reg=read_store_alpha_map_region(gen,identity,"alpha_registration",buf_y1,ye,budget);
        a_sep.insert(a_sep.end(),t_sep.begin(),t_sep.end());
        a_art.insert(a_art.end(),t_art.begin(),t_art.end());
        a_reg.insert(a_reg.end(),t_reg.begin(),t_reg.end());
      }
    }
    buf_y0=ys; buf_y1=ye;

    const auto sub=fuse_multiband(U,R,F,M,identity.color_mode,W,sub_h,fcfg,
                                  contract.alpha,contract.guard,a_sep,a_art,a_reg,{});
    const std::vector<float> *sv[3]; const std::vector<uint8_t> *ss[3];
    if (mono){ sv[0]=&sub.L; ss[0]=&sub.support_L; }
    else { sv[0]=&sub.R; sv[1]=&sub.G; sv[2]=&sub.B;
           ss[0]=&sub.support_R; ss[1]=&sub.support_G; ss[2]=&sub.support_B; }
    const std::size_t src_off=static_cast<std::size_t>(y0-ys)*W;
    const std::size_t dst_off=static_cast<std::size_t>(y0)*W;
    const std::size_t core=static_cast<std::size_t>(y1-y0)*W;
    for (int c=0;c<nch;++c) {
      std::copy(sv[c]->begin()+src_off,sv[c]->begin()+src_off+core,
                out[c].begin()+dst_off);
      for (std::size_t k=0;k<core;++k) if ((*ss[c])[src_off+k]) ++pixels_supported;
    }

    if (spool_out) {
      const std::vector<float> *uv3[3], *rv3[3];
      if (mono) { uv3[0]=&U.L.value; rv3[0]=&R.L.value; }
      else { uv3[0]=&U.R.value; uv3[1]=&U.G.value; uv3[2]=&U.B.value;
             rv3[0]=&R.R.value; rv3[1]=&R.G.value; rv3[2]=&R.B.value; }
      const std::vector<float> *src[3]={nullptr,nullptr,nullptr};
      const std::streamsize nbytes=
          static_cast<std::streamsize>(core*sizeof(float));
      for (int c=0;c<nch;++c) {
        src[0]=uv3[c]; src[1]=rv3[c]; src[2]=sv[c];  // uniform / raw / multiband
        for (int k=0;k<3;++k) {
          auto &os=spool_os[static_cast<std::size_t>(k)][static_cast<std::size_t>(c)];
          os.write(reinterpret_cast<const char *>(src[k]->data()+src_off),nbytes);
          if (!os) throw std::runtime_error("FUSE_STORE_SPOOL_WRITE_FAILED");
        }
      }
    }

    if (candidates_out) {
      const std::size_t bn=static_cast<std::size_t>(sub_h)*W;
      std::vector<const std::vector<float> *> uv,rv,mv;
      std::vector<const std::vector<uint8_t> *> usp,rsp,msp;
      if (mono) {
        uv={&U.L.value}; usp={&U.L.support};
        rv={&R.L.value}; rsp={&R.L.support};
        mv={&sub.L};      msp={&sub.support_L};
      } else {
        uv={&U.R.value,&U.G.value,&U.B.value};
        usp={&U.R.support,&U.G.support,&U.B.support};
        rv={&R.R.value,&R.G.value,&R.B.value};
        rsp={&R.R.support,&R.G.support,&R.B.support};
        mv={&sub.R,&sub.G,&sub.B};
        msp={&sub.support_R,&sub.support_G,&sub.support_B};
      }
      std::vector<float> lu,lr,lm; std::vector<uint8_t> su,sr_,sm;
      combine_luma(uv,usp,nch,bn,lu,su);
      combine_luma(rv,rsp,nch,bn,lr,sr_);
      combine_luma(mv,msp,nch,bn,lm,sm);
      std::copy(lu.begin()+src_off,lu.begin()+src_off+core,
                candidates_out->uniform_luma.begin()+dst_off);
      std::copy(lr.begin()+src_off,lr.begin()+src_off+core,
                candidates_out->raw_luma.begin()+dst_off);
      std::copy(lm.begin()+src_off,lm.begin()+src_off+core,
                candidates_out->multiband_luma.begin()+dst_off);
      std::copy(su.begin()+src_off,su.begin()+src_off+core,
                candidates_out->uniform_support.begin()+dst_off);
      for (int b=0;b<fcfg.levels;++b) {
        const auto &af=sub.alpha_final[static_cast<std::size_t>(b)];
        if (af.empty()) continue;
        auto &dstv=candidates_out->alpha_final_by_band[static_cast<std::size_t>(b)];
        if (dstv.empty()) dstv.assign(N,0.0f);
        std::copy(af.begin()+src_off,af.begin()+src_off+core,dstv.begin()+dst_off);
      }
    }
  }

  if (spool_out) {
    for (int k=0;k<3;++k)
      for (int c=0;c<nch;++c) {
        auto &os=spool_os[static_cast<std::size_t>(k)][static_cast<std::size_t>(c)];
        os.flush(); os.close();
        std::error_code ec;
        const auto want=static_cast<std::uintmax_t>(N*sizeof(float));
        if (fs::file_size(spool_out->plane_path(kCandNames[k],c),ec)!=want)
          throw std::runtime_error("FUSE_STORE_SPOOL_SIZE_MISMATCH");
      }
    spool_out->populated=true;
  }

  io::FitsHeader header;
  if (mono) {
    io::write_fits_float_rows(final_image_path,out[0],H,W,header);
  } else {
    auto to_mat=[&](const std::vector<float> &v){
      Matrix2Df m(H,W);
      for (int y=0;y<H;++y) for (int x=0;x<W;++x)
        m(y,x)=v[static_cast<std::size_t>(y)*W+x];
      return m;
    };
    io::write_fits_rgb(final_image_path,to_mat(out[0]),to_mat(out[1]),to_mat(out[2]),header);
  }
  return pixels_supported;
}

} // namespace tile_compile::reconstruction
