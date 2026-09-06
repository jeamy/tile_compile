#include "runner_forward_drizzle.hpp"
#include "tile_compile/config/legacy_config_migration.hpp"
#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/reconstruction/source_quality_map_cache.hpp"
#include "tile_compile/reconstruction/multiband_validation.hpp"
#include "tile_compile/reconstruction/multiband_fusion.hpp"
#include "tile_compile/reconstruction/output_scale.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/registration/sampling_geometry.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>

namespace tile_compile::runner {
namespace {
using core::json;
constexpr const char *scope = "forward_drizzle_m1_m3";
const std::vector<std::string> geometry_files = {
    "registration_sampling.json", "sampling_geometry.json",
    "sampling_geometry_analysis_common_mask.fits",
    "sampling_geometry_reconstruction_support_mask.fits", "forward_common_overlap.json"};
json checked_json(const fs::path &path) {
  if (!fs::is_regular_file(fs::symlink_status(path)) || fs::file_size(path)>16*1024*1024)
    throw std::runtime_error("FORWARD_STAGE_INVALID_JSON_FILE");
  std::ifstream file(path);
  return json::parse(file);
}
reconstruction::GlobalQualityConfig quality_config(const config::Config &cfg) {
  reconstruction::GlobalQualityConfig q;
  q.w_bg=cfg.global_metrics.weights.background;
  q.w_noise=cfg.global_metrics.weights.noise;
  q.w_grad=cfg.global_metrics.weights.gradient;
  q.w_fwhm=cfg.global_metrics.weights.fwhm;
  q.w_roundness=cfg.global_metrics.weights.roundness;
  q.w_star_count=cfg.global_metrics.weights.star_count;
  q.clamp_lo=cfg.global_metrics.clamp[0]; q.clamp_hi=cfg.global_metrics.clamp[1];
  q.adaptive_weights=cfg.global_metrics.adaptive_weights;
  q.weight_exponent_scale=cfg.global_metrics.weight_exponent_scale;
  return q;
}
json validate_provenance(const fs::path &dir,
                         const registration::RegistrationSamplingPlan &plan) {
  const auto provenance=checked_json(dir/"artifacts/run_provenance.json");
  if (provenance.at("execution_scope")!=scope ||
      provenance.at("config").at("sha256")!=core::sha256_file(dir/"config.yaml"))
    throw std::runtime_error("FORWARD_STAGE_CONFIG_OR_SCOPE_MISMATCH");
  const auto identity=provenance.at("input_manifest").at("sha256").get<std::string>()+":"+
                      provenance.at("config").at("sha256").get<std::string>();
  if (plan.source_identity_hash!=core::sha256_bytes(std::vector<uint8_t>(identity.begin(),identity.end())))
    throw std::runtime_error("FORWARD_STAGE_SOURCE_IDENTITY_MISMATCH");
  registration::RegistrationSamplingPlan parsed;
  std::string error;
  if (!registration::parse_from_json_string(registration::serialize_to_json_string(plan),parsed,error))
    throw std::runtime_error("FORWARD_STAGE_INVALID_SAMPLING_PLAN: "+error);
  return provenance;
}
}

bool run_forward_drizzle_stages(const std::string &run_id,const config::Config &cfg,
    const fs::path &dir,const registration::RegistrationSamplingPlan &sampling,
    RunnerFrameCache *fresh_cache,core::EventEmitter &emitter,std::ostream &log,
    const std::string &resume_from) {
  std::optional<Phase> active;
  auto begin=[&](Phase phase) { active=phase; emitter.phase_start(run_id,phase,phase_to_string(phase),log); };
  auto end=[&](const json &extra=json::object()) {
    emitter.phase_end(run_id,*active,"ok",extra,log); active.reset();
  };
  try {
    if (!resume_from.empty() && resume_from!="GLOBAL_QUALITY" && resume_from!="FORWARD_DRIZZLE")
      throw std::invalid_argument("FORWARD_STAGE_UNSUPPORTED_RESUME_PHASE");
    const auto provenance=validate_provenance(dir,sampling);
    auto reconstruction_cfg=cfg.reconstruction;
    if (!reconstruction_cfg.drizzle.memory_budget_mb)
      reconstruction_cfg.drizzle.memory_budget_mb=static_cast<size_t>(std::max(1,cfg.runtime_limits.memory_budget));
    const auto &drizzle=reconstruction_cfg.drizzle;
    const auto artifacts=dir/"artifacts";
    const auto cache_dir=dir/"cache/normalized_frames";
    const auto checkpoint_path=artifacts/"forward_drizzle_checkpoint.json";
    const auto geometry_hash=registration::compute_coverage_geometry_hash(
        sampling,drizzle,reconstruction_cfg.common_overlap_required_fraction);
    json checkpoint;
    if (resume_from.empty()) {
      if (!fresh_cache) throw std::runtime_error("FORWARD_STAGE_NORMALIZED_CACHE_REQUIRED");
      begin(Phase::NORMALIZED_CACHE);
      fresh_cache->seal_normalized_cache(sampling);
      end();
      begin(Phase::SAMPLING_GEOMETRY);
      auto coverage=registration::compute_geometric_coverage(sampling,drizzle.internal_scale,
          drizzle.pixfrac,reconstruction_cfg.coverage_gate,
          reconstruction_cfg.common_overlap_required_fraction,1,drizzle,false);
      io::FitsHeader header;
      header.set("MASKTYPE",std::string("SAMPLING_GEOMETRY"));
      io::write_fits_mask_rows(artifacts/geometry_files[2],coverage.analysis_common_mask,
                              coverage.internal_height,coverage.internal_width,header);
      io::write_fits_mask_rows(artifacts/geometry_files[3],coverage.reconstruction_support_mask,
                              coverage.internal_height,coverage.internal_width,header);
      core::write_text_atomic(artifacts/"sampling_geometry.json",
          registration::serialize_sampling_geometry_json(sampling,geometry_hash,drizzle.kernel,
              drizzle.pixfrac,drizzle.internal_scale,coverage));
      if (!coverage.gate.passed) throw std::runtime_error("FORWARD_STAGE_COVERAGE_GATE_FAILED");
      end({{"analysis_pixels",coverage.gate.analysis_pixels}});
      begin(Phase::COMMON_OVERLAP);
      core::write_text_atomic(artifacts/"forward_common_overlap.json",json({
        {"schema_version",1},{"source","sampling_geometry"},{"geometry_hash",geometry_hash},
        {"analysis_mask",geometry_files[2]},{"support_mask",geometry_files[3]},
        {"width",coverage.internal_width},{"height",coverage.internal_height},
        {"analysis_pixels",coverage.gate.analysis_pixels}}).dump(2));
      coverage={};
      reconstruction::VerifiedNormalizedSourceCache cache(cache_dir,sampling,drizzle.memory_budget_mb);
      checkpoint={{"schema_version",1},{"execution_scope",scope},
        {"config_sha256",provenance.at("config").at("sha256")},
        {"sampling_plan_hash",sampling.plan_hash},{"geometry_hash",geometry_hash},
        {"cache_manifest_hash",cache.manifest_hash()},{"artifacts",json::object()}};
      for (const auto &name:geometry_files) checkpoint["artifacts"][name]=core::sha256_file(artifacts/name);
      core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
      end();
    } else {
      checkpoint=checked_json(checkpoint_path);
      if (checkpoint.at("schema_version")!=1 || checkpoint.at("execution_scope")!=scope ||
          checkpoint.at("config_sha256")!=provenance.at("config").at("sha256") ||
          checkpoint.at("sampling_plan_hash")!=sampling.plan_hash || checkpoint.at("geometry_hash")!=geometry_hash ||
          checkpoint.at("artifacts").size()!=geometry_files.size())
        throw std::runtime_error("FORWARD_STAGE_CHECKPOINT_MISMATCH");
      for (const auto &name:geometry_files)
        if (checkpoint.at("artifacts").at(name)!=core::sha256_file(artifacts/name))
          throw std::runtime_error("FORWARD_STAGE_PREDECESSOR_CORRUPT: "+name);
    }
    reconstruction::VerifiedNormalizedSourceCache cache(cache_dir,sampling,drizzle.memory_budget_mb);
    if (checkpoint.at("cache_manifest_hash")!=cache.manifest_hash())
      throw std::runtime_error("FORWARD_STAGE_CACHE_MANIFEST_CHANGED");
    // Check every retained source before announcing a resumable phase start.
    for (const auto &f:sampling.frames) cache.load(f.source_index);
    const auto qcfg=quality_config(cfg);
    const auto sqm_cache_root=dir/"cache/source_quality_maps";
    if (resume_from.empty()) {
      begin(Phase::SOURCE_QUALITY_MAPS);
      const auto sqm=reconstruction::build_source_quality_map_cache(
          sqm_cache_root,sampling,cache,cfg.aqmh.pyramid);
      checkpoint["source_quality_identity_hash"]=sqm.source_identity_hash;
      checkpoint["source_quality_config_hash"]=sqm.source_quality_config_hash;
      checkpoint["source_quality_cache_hash"]=sqm.source_quality_cache_hash;
      core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
      end({{"frames",sqm.frames},{"computed_scales",sqm.computed_scales},
           {"streams",sqm.streams},
           {"source_quality_cache_hash",sqm.source_quality_cache_hash}});
    } else {
      reconstruction::SourceQualityMapCacheReader sqm_reader(
          sqm_cache_root,checkpoint.value("source_quality_identity_hash",""),
          checkpoint.value("source_quality_config_hash",""));
      if (!sqm_reader.usable())
        throw std::runtime_error("FORWARD_STAGE_SOURCE_QUALITY_CACHE_UNUSABLE: "+
                                 sqm_reader.error());
      if (sqm_reader.metadata().source_quality_cache_hash!=
          checkpoint.value("source_quality_cache_hash",""))
        throw std::runtime_error("FORWARD_STAGE_SOURCE_QUALITY_CACHE_CHANGED");
    }
    const auto quality_path=artifacts/"source_quality_plan.json";
    if (resume_from!="FORWARD_DRIZZLE") {
      begin(Phase::GLOBAL_QUALITY);
      const auto quality=reconstruction::persist_source_quality_artifact(
          quality_path,sampling,cache,qcfg,drizzle.memory_budget_mb);
      checkpoint["quality_plan_hash"]=quality.plan_hash;
      core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
      end({{"quality_plan_hash",quality.plan_hash}});
    } else {
      const auto quality=reconstruction::load_source_quality_artifact(
          quality_path,sampling,cache,qcfg,drizzle.memory_budget_mb);
      if (checkpoint.at("quality_plan_hash")!=quality.plan_hash)
        throw std::runtime_error("FORWARD_STAGE_QUALITY_PLAN_CHANGED");
    }
    const bool applied_2x2=drizzle.internal_scale==2 && drizzle.output_scale==1;
    // M6: the single-method path is multiband. FORWARD_DRIZZLE persists
    // uniform+raw+fine+(medium)+the four alpha-confidence maps in one store
    // (2/1 area-averages fine/medium by the 2x2 mean and the channel-min
    // confidence maps by 2x2 min + AND support); MULTIBAND then fuses to
    // the final X_out image.
    const bool want_multiband=reconstruction_cfg.multiband.enabled;

    // Plan 19: resolve the FORWARD_DRIZZLE acceleration intent. `select_*` says
    // whether `cuda` is buildable for this phase; passing "cuda" downstream is
    // an intent, not a guarantee --- persist_multiband_store_from_predecessors
    // owns the "is there a usable device / fault armed" decision and the
    // plan-19.4 full-phase CPU restart on ForwardDrizzleCudaError.
    const auto fd_accel=core::select_acceleration_backend(
        cfg.runtime_limits.acceleration_backend,core::AccelerationPhase::forward_drizzle);
    const std::string fd_backend=
        fd_accel.selected==core::AccelerationBackend::cuda ? "cuda" : "cpu";
    std::string fd_backend_used="cpu", fd_cuda_fallback_reason;

    begin(Phase::FORWARD_DRIZZLE);
    const auto profiles_root=artifacts/"forward_drizzle_profiles";
    reconstruction::DrizzleStoreIdentity mb_identity;
    reconstruction::DrizzleStoreResult result;
    if (want_multiband) {
      auto built=reconstruction::persist_multiband_store_from_predecessors(
          profiles_root,quality_path,sampling,cache,qcfg,drizzle,
          reconstruction_cfg.clipping,reconstruction_cfg.multiband,sqm_cache_root,
          {},fd_backend);
      result=built.store;
      mb_identity=built.identity;
      fd_backend_used=built.backend_used;
      fd_cuda_fallback_reason=built.cuda_fallback_reason;
      checkpoint["multiband_reconstruction_hash"]=mb_identity.reconstruction_hash;
      checkpoint["multiband_levels"]=mb_identity.multiband_levels;
    } else {
      result=reconstruction::persist_forward_drizzle_from_predecessors(
          profiles_root,quality_path,sampling,cache,qcfg,
          drizzle,reconstruction_cfg.clipping,{},sqm_cache_root);
    }
    checkpoint["profiles_current_sha256"]=core::sha256_file(profiles_root/"current.json");
    checkpoint["forward_drizzle_backend"]=fd_backend_used;
    if (!fd_cuda_fallback_reason.empty())
      checkpoint["forward_drizzle_cuda_fallback_reason"]=fd_cuda_fallback_reason;
    core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
    {
      json extra={{"generation",result.generation_dir.filename().string()},
         {"estimated_peak_bytes",result.diagnostics.estimated_peak_bytes},
         {"internal_scale",drizzle.internal_scale},
         {"output_scale",drizzle.output_scale},
         {"output_scale_applied",applied_2x2},
         {"multiband",want_multiband},
         {"acceleration_backend",fd_backend_used},
         {"kernel_noise_sigma_factor",
          reconstruction::kernel_noise_correlation_sigma_factor(
              drizzle.pixfrac,drizzle.internal_scale)}};
      if (!fd_cuda_fallback_reason.empty())
        extra["cuda_fallback_reason"]=fd_cuda_fallback_reason;
      end(extra);
    }

    bool final_image_available=false;
    if (want_multiband) {
      begin(Phase::MULTIBAND);
      const auto final_image=artifacts/"reconstruction_multiband.fits";
      reconstruction::MultibandCandidateLuma cand;
      const auto pixels=reconstruction::fuse_multiband_store_to_image(
          profiles_root,mb_identity,final_image,reconstruction_cfg.multiband,
          drizzle.chunk_rows,drizzle.memory_budget_mb,&cand);
      checkpoint["final_image_sha256"]=core::sha256_file(final_image);

      // Plan 15: three-way candidate selection on the fixed working luminance
      // (drizzle_uniform / drizzle_raw / drizzle_multiband), stars detected
      // ONCE on the uniform control. This RECORDS the decision (16.3); it does
      // not (yet) change which file is delivered downstream.
      auto to_mat=[](const std::vector<float> &v,int w,int h){
        Matrix2Df m(h,w);
        for (int y=0;y<h;++y) for (int x=0;x<w;++x)
          m(y,x)=v[static_cast<std::size_t>(y)*w+x];
        return m;
      };
      const auto uni_m=to_mat(cand.uniform_luma,cand.width,cand.height);
      const auto raw_m=to_mat(cand.raw_luma,cand.width,cand.height);
      const auto mb_m=to_mat(cand.multiband_luma,cand.width,cand.height);
      const auto stars=reconstruction::prepare_validation_samples(
          uni_m,cand.width,cand.height,cand.uniform_support,cand.alpha_final_by_band);
      // One config object feeds both the selection and its provenance hash, so
      // validation_config_hash is structurally the config that was used.
      const reconstruction::MultibandValidationConfig val_cfg{};
      const auto sel=reconstruction::select_reconstruction_candidate(
          uni_m,raw_m,mb_m,cand.width,cand.height,stars,val_cfg,cand.uniform_support);
      const std::string sel_name=
          sel.selected==reconstruction::SelectedCandidate::kDrizzleMultiband ? "drizzle_multiband"
          : sel.selected==reconstruction::SelectedCandidate::kDrizzleRaw ? "drizzle_raw"
          : "drizzle_uniform";

      auto metric_json=[](const reconstruction::ValidationMetric &m){
        // A non-applicable metric serialises value:null uniformly, so a
        // consumer can never read a default 0.0 or a NaN as a measurement.
        json j={{"value",(m.applicable && std::isfinite(m.value))
                             ? json(m.value) : json(nullptr)},
                {"applicable",m.applicable},
                {"sample_count",m.sample_count}};
        if (m.ci_low!=0.0 || m.ci_high!=0.0) { j["ci_low"]=m.ci_low; j["ci_high"]=m.ci_high; }
        if (!m.reason_if_not_applicable.empty())
          j["reason_if_not_applicable"]=m.reason_if_not_applicable;
        return j;
      };
      auto cand_json=[&](const reconstruction::CandidateMetrics &c){
        return json{{"median_fwhm",metric_json(c.median_fwhm)},
                    {"p90_fwhm",metric_json(c.p90_fwhm)},
                    {"tail",metric_json(c.tail)},
                    {"elongation",metric_json(c.elongation)},
                    {"background_rms",metric_json(c.background_rms)},
                    {"seam_score",metric_json(c.seam_score)},
                    {"support_ok",c.support_ok},{"numerics_ok",c.numerics_ok}};
      };
      json fwd={
        {"schema_version",1},
        {"pipeline_method","cfa_forward_drizzle_multiband"},
        {"pipeline_contract_version",1},
        {"sampling_plan_hash",sampling.plan_hash},
        {"coverage_geometry_hash",geometry_hash},
        {"multiband_reconstruction_hash",mb_identity.reconstruction_hash},
        {"multiband_levels",mb_identity.multiband_levels},
        {"luma_definition",reconstruction::kWorkingLumaDefinition},
        {"validation",{
          {"version",reconstruction::kMultibandValidationVersion},
          {"validation_config_hash",
           reconstruction::multiband_validation_config_hash(val_cfg)},
          {"stars_total",sel.stars_total},
          {"stars_multiband_effective",sel.stars_multiband_effective},
          {"multiband_star_sample_count",sel.multiband_star_sample_count},
          {"drizzle_uniform",cand_json(sel.uniform)},
          {"drizzle_raw",cand_json(sel.raw)},
          {"drizzle_multiband",cand_json(sel.multiband)}}},
        {"selected_candidate",sel_name},
        {"selection_reason",sel.reason},
        {"fallback_reason",
         sel.selected==reconstruction::SelectedCandidate::kDrizzleMultiband
             ? json(nullptr) : json(sel.reason)},
        {"outputs",json::array({json{{"path",final_image.filename().string()},
                                     {"sha256",checkpoint["final_image_sha256"]}}})},
        {"commit_complete",true}};
      core::write_text_atomic(artifacts/"forward_drizzle.json",fwd.dump(2));
      // No checkpoint hash guard for forward_drizzle.json: MULTIBAND fully
      // regenerates it on every (re)run, so there is nothing to verify on
      // resume (unlike the immutable geometry predecessors).
      checkpoint["selected_candidate"]=sel_name;
      checkpoint["status"]="final_image_ready";
      core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
      final_image_available=true;
      end({{"final_image",final_image.filename().string()},
           {"pixels_supported",pixels},
           {"selected_candidate",sel_name},
           {"selection_reason",sel.reason},
           {"stars_total",sel.stars_total},
           {"multiband_levels",mb_identity.multiband_levels}});
    } else {
      checkpoint["status"]="reconstruction_ready";
      core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
    }

    // Cache-lifetime contract (plan 16.2). Only after a fully committed final
    // image: the internal transactional U/R/F/M profile store is a
    // reconstruction cache, never a downstream-resume predecessor, so it is
    // deleted by default and kept (as a hashed cache) only on request. The
    // source caches are kept by default; deleting them disables
    // resume-reconstruction and is announced as such. Reported in the run_end
    // event, not as a phase, so the phase sequence is unchanged.
    json cache_actions=json::object();
    if (final_image_available) {
      if (want_multiband && !reconstruction_cfg.keep_profile_cache_after_run) {
        std::error_code ec; fs::remove_all(profiles_root,ec);
        cache_actions["profile_cache"]=ec ? "delete_failed" : "deleted";
        checkpoint["profile_cache_retained"]=false;
      } else if (want_multiband) {
        cache_actions["profile_cache"]="retained";
        checkpoint["profile_cache_retained"]=true;
      }
      if (reconstruction_cfg.delete_source_cache_after_run) {
        std::error_code e1,e2;
        fs::remove_all(sqm_cache_root,e1);
        fs::remove_all(cache_dir,e2);  // cache/normalized_frames
        cache_actions["source_cache"]=(e1||e2) ? "delete_failed" : "deleted";
        cache_actions["resume_reconstruction_disabled"]=true;
        checkpoint["source_cache_retained"]=false;
      } else {
        cache_actions["source_cache"]="retained";
        checkpoint["source_cache_retained"]=true;
      }
      core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
    }

    emitter.run_end(run_id,true,
                    final_image_available?"final_image_ready":"reconstruction_ready",
                    log,{{"execution_scope",scope},
                         {"final_image_available",final_image_available},
                         {"cache_retention",cache_actions}});
    return true;
  } catch (const std::exception &e) {
    if (active) emitter.phase_end(run_id,*active,"error",{{"error",e.what()}},log);
    emitter.run_end(run_id,false,"error",log,{{"message",e.what()}});
    return false;
  }
}
} // namespace tile_compile::runner

int resume_forward_drizzle_command(const std::string &path,const std::string &phase) {
#ifdef TILE_COMPILE_LEGACY_REFERENCE
  (void)path; (void)phase;
  std::cerr<<"LEGACY_REFERENCE_RESUME_DISABLED\n";
  return 1;
#else
  using namespace tile_compile;
  try {
    const fs::path dir=fs::absolute(path);
    // No logs or artifacts are opened for writing before basic identity checks.
    const auto provenance=runner::checked_json(dir/"artifacts/run_provenance.json");
    if (provenance.at("execution_scope")!="forward_drizzle_m1_m3" ||
        provenance.at("config").at("sha256")!=core::sha256_file(dir/"config.yaml"))
      throw std::runtime_error("FORWARD_STAGE_CONFIG_OR_SCOPE_MISMATCH");
    config::ConfigMigrationReport migration;
    const auto cfg=config::Config::from_yaml_text_migrated(core::read_text(dir/"config.yaml"),migration);
    cfg.validate();
    registration::RegistrationSamplingPlan sampling;
    std::string error;
    if (!registration::parse_from_json_string(core::read_text(dir/"artifacts/registration_sampling.json"),sampling,error))
      throw std::runtime_error(error);
    std::ofstream log(dir/"logs/run_events.jsonl",std::ios::app);
    if (!log) throw std::runtime_error("FORWARD_STAGE_LOG_UNAVAILABLE");
    core::EventEmitter emitter;
    return runner::run_forward_drizzle_stages(dir.filename().string(),cfg,dir,sampling,
                                             nullptr,emitter,log,phase) ? 0:1;
  } catch (const std::exception &e) { std::cerr<<e.what()<<std::endl; return 1; }
#endif
}
