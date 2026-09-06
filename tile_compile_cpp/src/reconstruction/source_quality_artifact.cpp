#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/reconstruction/source_quality_map_cache.hpp"
#include "tile_compile/reconstruction/multiband_fusion.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
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
    VerifiedNormalizedSourceCache &cache,const GlobalQualityConfig &cfg,size_t mb) {
  preflight(sampling,cfg,mb);
  validate_sampling(sampling);
  if (!cache.matches(sampling)) throw std::invalid_argument("SOURCE_QUALITY_CACHE_CONTEXT_MISMATCH");
  const auto weights=compute_global_quality_weights(sampling.frames.size(),
      [&](size_t i)->const Matrix2Df & { return cache.load(sampling.frames.at(i).source_index); },
      sampling.color_mode,sampling.bayer_pattern,sampling.cfa_origin_x,sampling.cfa_origin_y,cfg);
  auto plan=build_quality_frame_weight_plan(sampling,weights,compute_source_quality_config_hash(cfg));
  resolve_quality_frame_weights(plan,sampling,cfg,mb);
  json artifact={{"schema_version",1},{"normalized_cache_hash",cache.manifest_hash()},
                 {"quality_plan",json::parse(serialize_quality_frame_weight_plan(plan))}};
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
    const fs::path &source_quality_cache_root) {
  const size_t mb=drizzle_cfg.memory_budget_mb ? drizzle_cfg.memory_budget_mb : 512;
  const auto quality=load_source_quality_artifact(quality_artifact,sampling,cache,quality_cfg,mb);
  const auto weights=resolve_quality_frame_weights(quality,sampling,quality_cfg,mb);

  // M5: optionally consume the source composite Q-maps as Q_composite_f,c(q).
  // (Fine/Medium's scale_0/scale_1 streams are wired in a later M6 batch.)
  DrizzleStorePredecessors predecessors{cache.manifest_hash(),quality.plan_hash,{}};
  FrameQualityProvider quality_of;  // null => Q_composite = 1.0 (unchanged Raw)
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
  }

  return persist_forward_drizzle_uniform_and_raw(store_root,sampling,
      [&](size_t index)->const Matrix2Df & { return cache.load(index); },
      drizzle_cfg,clipping_cfg,subdivision,weights,predecessors,quality_of);
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
    const std::string &acceleration_backend) {
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
  FrameQualityProvider quality_of=
      [reader=qreader.get(),need_medium,
       comp=Matrix2Df(),s0=Matrix2Df(),s1=Matrix2Df(),art=Matrix2Df(),
       have_s0=false,have_s1=false,have_art=false,
       idx=std::size_t(-1)](std::size_t source_index) mutable -> FrameQualityMaps {
    if (source_index!=idx) {
      // composite is mandatory; the finer scale + artifact streams may be
      // absent for a small image (fewer pyramid scales) --- a missing map is
      // a null pointer (weight degrades), never a hard failure here.
      comp=reader->read_full("composite",source_index);
      have_s0=reader->has("scale_0",source_index);
      if (have_s0) s0=reader->read_full("scale_0",source_index);
      have_s1=need_medium && reader->has("scale_1",source_index);
      if (have_s1) s1=reader->read_full("scale_1",source_index);
      have_art=reader->has("artifact",source_index);
      if (have_art) art=reader->read_full("artifact",source_index);
      idx=source_index;
    }
    return FrameQualityMaps{&comp, have_s0?&s0:nullptr, have_s1?&s1:nullptr,
                            have_art?&art:nullptr};
  };

  const auto source_of=[&](size_t index)->const Matrix2Df & { return cache.load(index); };
  const auto build=[&](const ForwardDrizzleCudaOptions &cuda){
    return persist_forward_drizzle_multiband(
        store_root,sampling,source_of,drizzle_cfg,clipping_cfg,contract,
        quality_of,subdivision,weights,predecessors,cuda);
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
      out.backend_used="cuda";
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
  return out;
}

long long fuse_multiband_store_to_image(
    const fs::path &store_root,const DrizzleStoreIdentity &identity,
    const fs::path &final_image_path,
    const config::ReconstructionMultibandConfig &multiband_cfg,
    int chunk_rows,size_t memory_budget_mb,
    MultibandCandidateLuma *candidates_out) {
  if (identity.multiband_levels<1)
    throw std::invalid_argument("FUSE_STORE_NOT_A_MULTIBAND_IDENTITY");
  // The caller must pass the config the store was built with. Guard the one
  // field that changes the plane set / band assignment; the alpha/guard edges
  // are trusted from `multiband_cfg` (in the runner flow write and fuse share
  // one config, so they cannot drift).
  if (multiband_cfg.enabled && multiband_cfg.levels!=identity.multiband_levels)
    throw std::invalid_argument("FUSE_STORE_MULTIBAND_LEVELS_MISMATCH");
  const bool need_medium=identity.multiband_levels>=2;
  const size_t budget=std::max<size_t>(memory_budget_mb,256);
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

  for (int y0=0;y0<H;y0+=chunk) {
    const int y1=std::min(H,y0+chunk);
    const int ys=std::max(0,y0-halo), ye=std::min(H,y1+halo);
    const int sub_h=ye-ys;
    const auto U=read_store_profile_region(gen,identity,"uniform",ys,ye,budget);
    const auto R=read_store_profile_region(gen,identity,"raw",ys,ye,budget);
    const auto F=read_store_profile_region(gen,identity,"fine",ys,ye,budget);
    const auto M=need_medium
        ? read_store_profile_region(gen,identity,"medium",ys,ye,budget)
        : ForwardDrizzleUniformResult{};
    const auto a_sep=read_store_alpha_map_region(gen,identity,"alpha_separation",ys,ye,budget);
    const auto a_art=read_store_alpha_map_region(gen,identity,"alpha_artifact",ys,ye,budget);
    const auto a_reg=read_store_alpha_map_region(gen,identity,"alpha_registration",ys,ye,budget);

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
