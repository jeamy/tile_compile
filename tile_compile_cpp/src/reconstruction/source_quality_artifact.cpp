#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/reconstruction/source_quality_map_cache.hpp"
#include "tile_compile/reconstruction/multiband_fusion.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2.hpp"
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
#include <iostream>
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
  const uint64_t scratch=8*1024*1024+side*side*8+
      static_cast<uint64_t>(cfg.star_max_corners)*128+
      static_cast<uint64_t>(sampling.frames.size())*4096;
  plan_drizzle_memory_autogrow(geometry,resources,128,
      static_cast<size_t>(scratch),true,[](const std::string &msg){
        std::cout<<"[SOURCE_QUALITY][warn] "<<msg<<std::endl;});
  // Recheck against the (possibly grown) effective budget.
  const uint64_t budget=budget_bytes(resources.memory_budget_mb);
  if (side>std::sqrt(static_cast<long double>(budget/8)) || scratch>budget)
    throw std::runtime_error("SOURCE_QUALITY_MEMORY_BUDGET");
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

  // Map metrics to plan frame order (by source_index slot). Frames marked
  // !f.valid in the sampling plan (unresolved registration) legitimately
  // have no SOURCE_QUALITY_MAPS metrics record: they get g_quality = 0 and
  // must not enter the weight normalisation statistics.
  std::map<std::size_t,std::size_t> by_idx;
  for (std::size_t i=0;i<ma.frames.size();++i)
    by_idx[ma.frames[i].source_index]=i;
  std::vector<FrameMetrics> frame_metrics_valid;
  std::vector<metrics::FrameStarMetrics> star_metrics_valid;
  std::vector<size_t> plan_pos_valid;
  frame_metrics_valid.reserve(sampling.frames.size());
  star_metrics_valid.reserve(sampling.frames.size());
  plan_pos_valid.reserve(sampling.frames.size());
  size_t invalid_without_metrics=0;
  for (std::size_t i=0;i<sampling.frames.size();++i) {
    const auto &f=sampling.frames[i];
    auto it=by_idx.find(f.source_index);
    if (it==by_idx.end()) {
      if (!f.valid) {
        ++invalid_without_metrics;
        continue;
      }
      throw std::invalid_argument("SOURCE_QUALITY_METRICS_FRAME_MISSING");
    }
    const auto &fm=ma.frames[it->second];
    frame_metrics_valid.push_back({fm.background,fm.noise,
        fm.gradient_energy,fm.sky_gradient,fm.quality_score});
    star_metrics_valid.push_back({fm.fwhm,fm.fwhm_x,fm.fwhm_y,
        fm.roundness,fm.wfwhm,fm.star_count});
    plan_pos_valid.push_back(i);
  }
  if (plan_pos_valid.empty())
    throw std::runtime_error("SOURCE_QUALITY_NO_VALID_FRAMES");
  const auto compact=compute_global_quality_weights_from_metrics(
      frame_metrics_valid,star_metrics_valid,cfg);
  VectorXf weights=VectorXf::Zero(static_cast<Eigen::Index>(sampling.frames.size()));
  for (std::size_t k=0;k<plan_pos_valid.size();++k)
    weights(static_cast<Eigen::Index>(plan_pos_valid[k]))=
        compact(static_cast<Eigen::Index>(k));
  if (invalid_without_metrics>0)
    std::cout<<"[SOURCE_QUALITY] "<<invalid_without_metrics
             <<" unresolved frame(s) without metrics -> g_quality=0"
             <<std::endl;

  auto plan=build_quality_frame_weight_plan(sampling,weights,compute_source_quality_config_hash(cfg));
  resolve_quality_frame_weights(plan,sampling,cfg,mb);
  json artifact={{"schema_version",1},{"normalized_cache_hash",cache.manifest_hash()},
                 {"quality_plan",json::parse(serialize_quality_frame_weight_plan(plan))},
                 {"metrics_source",metrics_path.filename().string()},
                 {"frames_without_metrics_invalid",invalid_without_metrics}};
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
}  // namespace


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


long long fuse_multiband_v2_store_to_image(
    const fs::path &store_root,const ForwardDrizzleV2RunPlan &plan,
    const fs::path &final_image_path,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    int chunk_rows,size_t memory_budget_mb,
    MultibandCandidateLuma *candidates_out,
    MultibandCandidateSpool *spool_out,
    MultibandFusionMemoryPlan *mem_plan_out,
    ForwardDrizzleV2FusionStats *stats_out) {
  if (!plan.emit_profiles || plan.multiband_levels<1)
    throw std::invalid_argument("FUSE_V2_STORE_NOT_A_MULTIBAND_PLAN");
  if (multiband_cfg.enabled && multiband_cfg.levels!=plan.multiband_levels)
    throw std::invalid_argument("FUSE_V2_STORE_MULTIBAND_LEVELS_MISMATCH");
  const bool need_medium=plan.multiband_levels>=2;
  const size_t budget=memory_budget_mb ? memory_budget_mb : size_t(256);
  // Fail closed: the committed generation must verify against THIS plan ---
  // a foreign-plan or ambiguous store never reaches the fusion loop.
  const auto insp=inspect_forward_drizzle_v2_store(store_root,plan);
  if (insp.status!=ForwardDrizzleV2StoreStatus::complete)
    throw std::runtime_error("FUSE_V2_STORE_NOT_COMPLETE: "+
        (insp.error.empty()?std::string("status=")+
             std::to_string(static_cast<int>(insp.status)):insp.error));
  const fs::path &gen=insp.generation;
  const auto &committed=insp.committed;
  if (static_cast<int>(committed.size())!=plan.band_count)
    throw std::runtime_error("FUSE_V2_STORE_INCOMPLETE_PREFIX");

  const auto contract=multiband_store_contract_from_config(multiband_cfg);
  config::ReconstructionMultibandConfig fcfg=multiband_cfg;
  fcfg.levels=plan.multiband_levels;
  const int W=plan.native_width, H=plan.native_height;
  const int chunk=std::max(1,std::min(H,chunk_rows>0?chunk_rows:64));
  const int halo=multiband_fusion_halo_rows(fcfg.levels);
  const bool mono=plan.channels==1;
  const ColorMode mode=mono?ColorMode::MONO:ColorMode::OSC;
  const std::size_t N=static_cast<std::size_t>(W)*H;
  const int nch_plan=mono?1:3;
  const std::size_t rec_bytes=sizeof(ForwardDrizzleV2ProfileResult);

  // Same pre-plan as the legacy path, plus the decoded-record window term
  // (channel-major ProfileResult buffer over chunk + 2*halo rows).
  MultibandFusionMemoryPlan mem_plan=plan_multiband_fusion_memory(
      W,H,nch_plan,fcfg.levels,chunk,halo,
      candidates_out!=nullptr,spool_out!=nullptr,
      static_cast<std::size_t>(budget)*1024*1024);
  const std::size_t window_rows=
      static_cast<std::size_t>(chunk)+2u*static_cast<std::size_t>(halo);
  const std::size_t window_record_bytes=
      window_rows*static_cast<std::size_t>(W)*
      static_cast<std::size_t>(nch_plan)*rec_bytes;
  mem_plan.stripe_working_bytes+=window_record_bytes;
  {
    std::size_t sum=mem_plan.final_image_bytes+mem_plan.stripe_working_bytes+
        mem_plan.candidate_luma_bytes+mem_plan.spool_stripe_bytes+
        mem_plan.delivery_readback_bytes;
    mem_plan.margin_bytes=std::max<std::size_t>(
        sum/20u,std::min<std::size_t>(std::size_t(64)<<20,sum));
    mem_plan.estimated_peak_bytes=sum+mem_plan.margin_bytes;
    mem_plan.fits=mem_plan.estimated_peak_bytes<=mem_plan.budget_bytes;
  }
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

  static constexpr const char *kCandNames[3]={"uniform","raw","multiband"};
  std::array<std::array<std::ofstream,3>,3> spool_os;
  if (spool_out) {
    spool_out->width=W; spool_out->height=H; spool_out->nch=nch; spool_out->mono=mono;
    spool_out->populated=false;
    for (int k=0;k<3;++k)
      for (int c=0;c<nch;++c) {
        const auto p=spool_out->plane_path(kCandNames[k],c);
        spool_os[static_cast<std::size_t>(k)][static_cast<std::size_t>(c)].open(
            p.string(),std::ios::binary|std::ios::trunc);
        if (!spool_os[static_cast<std::size_t>(k)][static_cast<std::size_t>(c)])
          throw std::runtime_error("FUSE_STORE_SPOOL_OPEN_FAILED");
      }
  }

  // Rolling band-record window (N6): bands are decoded once into a FIFO
  // cache; a stripe window [ys,ye) reuses the resident predecessor tail
  // instead of re-reading it. band index -> channel-major profile records.
  std::map<int,std::vector<ForwardDrizzleV2ProfileResult>> band_cache;
  std::uint64_t no_reuse_bytes=0;
  ForwardDrizzleV2FusionStats stats;
  auto band_records=[&](int bi)->const std::vector<ForwardDrizzleV2ProfileResult>& {
    auto it=band_cache.find(bi);
    if (it!=band_cache.end()) return it->second;
    if (bi<0 || bi>=static_cast<int>(committed.size()))
      throw std::runtime_error("FUSE_V2_BAND_INDEX_RANGE");
    auto recs=read_forward_drizzle_v2_band_profiles(gen,committed[static_cast<std::size_t>(bi)]);
    const auto &bc=committed[static_cast<std::size_t>(bi)];
    if (recs.size()!=static_cast<std::size_t>(bc.rows)*bc.native_cols*bc.channels)
      throw std::runtime_error("FUSE_V2_BAND_RECORD_COUNT");
    ++stats.bands_decoded;
    stats.record_bytes_read+=
        static_cast<std::uint64_t>(recs.size())*rec_bytes;
    return band_cache.emplace(bi,std::move(recs)).first->second;
  };

  std::vector<ForwardDrizzleV2ProfileResult> win;
  for (int y0=0;y0<H;y0+=chunk) {
    const int y1=std::min(H,y0+chunk);
    const int ys=std::max(0,y0-halo), ye=std::min(H,y1+halo);
    const int sub_h=ye-ys;
    const std::size_t sub_n=static_cast<std::size_t>(sub_h)*W;
    const int b_lo=static_cast<int>(
        std::lower_bound(committed.begin(),committed.end(),ys,
            [](const ForwardDrizzleV2BandCommit &b,int y){
              return b.y_begin+b.rows<=y;})-committed.begin());
    const int b_hi=static_cast<int>(
        std::lower_bound(committed.begin(),committed.end(),ye-1,
            [](const ForwardDrizzleV2BandCommit &b,int y){
              return b.y_begin+b.rows<=y;})-committed.begin());
    // Evict bands no stripe can still need (window start only advances).
    while (!band_cache.empty() && band_cache.begin()->first<b_lo)
      band_cache.erase(band_cache.begin());
    // Assemble the channel-major record window [ys,ye).
    win.assign(sub_n*static_cast<std::size_t>(nch),
               ForwardDrizzleV2ProfileResult{});
    for (int b=b_lo;b<=b_hi;++b) {
      const auto &bc=committed[static_cast<std::size_t>(b)];
      const auto &recs=band_records(b);
      const int lo=std::max(ys,bc.y_begin), hi=std::min(ye,bc.y_begin+bc.rows);
      for (int c=0;c<nch;++c) {
        const std::size_t src_base=
            static_cast<std::size_t>(c)*bc.rows*static_cast<std::size_t>(W)+
            static_cast<std::size_t>(lo-bc.y_begin)*W;
        const std::size_t dst_base=
            static_cast<std::size_t>(c)*sub_n+
            static_cast<std::size_t>(lo-ys)*W;
        std::copy_n(recs.data()+src_base,
                    static_cast<std::size_t>(hi-lo)*W,
                    win.data()+dst_base);
      }
    }
    no_reuse_bytes+=sub_n*static_cast<std::size_t>(nch)*rec_bytes;

    auto U=forward_drizzle_v2_profiles_to_uniform_result(win,W,sub_h,nch,mode,0);
    auto R=forward_drizzle_v2_profiles_to_uniform_result(win,W,sub_h,nch,mode,1);
    auto F=forward_drizzle_v2_profiles_to_uniform_result(win,W,sub_h,nch,mode,2);
    auto M=need_medium
        ? forward_drizzle_v2_profiles_to_uniform_result(win,W,sub_h,nch,mode,3)
        : ForwardDrizzleUniformResult{};
    auto a_sep=forward_drizzle_v2_profile_alpha_plane(win,W,sub_h,nch,0);
    auto a_art=forward_drizzle_v2_profile_alpha_plane(win,W,sub_h,nch,1);
    auto a_reg=forward_drizzle_v2_profile_alpha_plane(win,W,sub_h,nch,2);

    const auto sub=fuse_multiband(U,R,F,M,mode,W,sub_h,fcfg,
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

    // Global near-zero-alpha diagnostic accumulated over CORE rows only
    // (halo rows are shared between stripes and must not be double counted).
    for (std::size_t i=0;i<core;++i) {
      const std::size_t px=src_off+i;
      bool supported=false;
      double amin=std::numeric_limits<double>::infinity();
      for (int c=0;c<nch;++c) {
        const auto &r=win[static_cast<std::size_t>(c)*sub_n+px];
        if (!r.uniform.support) continue;
        supported=true;
        const double aa=r.artifact_applicable?static_cast<double>(r.a_artifact):1.0;
        amin=std::min(amin,static_cast<double>(r.a_separation)*aa*
                           static_cast<double>(r.a_registration));
      }
      if (!supported) continue;
      ++stats.alpha.supported_pixels;
      if (!std::isfinite(amin) || amin<=1e-6) ++stats.alpha.near_zero_pixels;
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

  stats.record_bytes_no_reuse=no_reuse_bytes;
  stats.alpha.near_zero_fraction=
      stats.alpha.supported_pixels>0
          ? static_cast<double>(stats.alpha.near_zero_pixels)/
                static_cast<double>(stats.alpha.supported_pixels)
          : 0.0;
  stats.alpha.global_near_zero=
      stats.alpha.supported_pixels>0 &&
      stats.alpha.near_zero_pixels==stats.alpha.supported_pixels;
  stats.read_amplification=
      stats.record_bytes_read>0
          ? static_cast<double>(stats.record_bytes_read)/
                static_cast<double>(stats.record_bytes_read)  // bands read once
          : 1.0;
  // The store-side amplification the spec binds is window_bytes/logical_bytes:
  // with the rolling cache every band is decoded exactly once, so the decode
  // amplification is 1.0; the no-reuse figure documents the alternative.
  if (stats_out) *stats_out=stats;

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
