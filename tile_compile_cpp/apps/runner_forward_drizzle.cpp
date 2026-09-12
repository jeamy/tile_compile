#include "runner_forward_drizzle.hpp"
#include "runner_downstream.hpp"
#include "tile_compile/config/legacy_config_migration.hpp"
#include "tile_compile/reconstruction/source_quality_artifact.hpp"
#include "tile_compile/reconstruction/source_quality_map_cache.hpp"
#include "tile_compile/reconstruction/multiband_validation.hpp"
#include "tile_compile/reconstruction/multiband_fusion.hpp"
#include "tile_compile/reconstruction/output_scale.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/core/build_info.hpp"
#include "tile_compile/reconstruction/drizzle_geometry_cache.hpp"
#include "tile_compile/reconstruction/drizzle_geometry_stats.hpp"
#include "tile_compile/registration/sampling_geometry.hpp"
#include <algorithm>
#include <cstdlib>
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <thread>

#include <sys/resource.h>

namespace tile_compile::runner {
namespace {
using core::json;
constexpr const char *scope = "forward_drizzle_m1_m3";

// Process-lifetime peak RSS, in KiB (Linux ru_maxrss is already KiB). This is a
// LIFETIME maximum -- the difference of two such readings is NOT phase-local
// growth (plan 11.13(4)); it is reported only as `rss_process_peak_kib`.
long long read_maxrss_kb() {
  struct rusage ru {};
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
  return static_cast<long long>(ru.ru_maxrss);
}
// Current / peak resident set size from /proc/self/status, in KiB. `VmRSS` is
// the live figure used for phase-scoped growth (sampled at phase begin/end);
// `VmHWM` is the process-lifetime high-water mark. 0 if unreadable.
long long read_proc_status_kb(const char *key) {
  std::ifstream st("/proc/self/status");
  std::string line;
  const std::string want = std::string(key) + ":";
  while (std::getline(st, line)) {
    if (line.rfind(want, 0) != 0) continue;
    long long kb = 0;
    bool seen = false;
    for (std::size_t i = want.size(); i < line.size(); ++i) {
      const char ch = line[i];
      if (ch >= '0' && ch <= '9') { kb = kb * 10 + (ch - '0'); seen = true; }
      else if (seen) break;  // stop at the first token boundary after the number
    }
    return kb;
  }
  return 0;
}
long long read_vmrss_kb() { return read_proc_status_kb("VmRSS"); }
long long read_vmhwm_kb() { return read_proc_status_kb("VmHWM"); }
// Plan 11.11 / 11.11.1: the throughput baseline is only comparable on a named
// reference machine, so `forward_drizzle.json` records the CPU model string.
// Linux `/proc/cpuinfo`; empty when unreadable (never fatal).
std::string read_cpu_model_name() {
  std::ifstream ci("/proc/cpuinfo");
  std::string line;
  while (std::getline(ci, line)) {
    if (line.rfind("model name", 0) != 0) continue;
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string v = line.substr(colon + 1);
    const auto b = v.find_first_not_of(" \t");
    const auto e = v.find_last_not_of(" \t\r\n");
    return (b == std::string::npos) ? std::string() : v.substr(b, e - b + 1);
  }
  return {};
}
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
  // Plan 11.14 P0: geometry instrumentation on for the whole stage. Integer
  // counters + coarse timers only --- no compute-hash / pixel effect (the
  // [cuda-parity] and [forward-runner] gates confirm). Per-variant counters
  // separate SAMPLING_GEOMETRY (coverage_*) from FORWARD_DRIZZLE
  // (production_uniform_raw / contrib_* / hybrid_cpu_geometry). Written to
  // artifacts/forward_drizzle_geometry_profile.json after FORWARD_DRIZZLE.
  reconstruction::geomstats::ScopedEnable geom_p0(true);
  const long long rss_baseline_kb=read_maxrss_kb();
  json phase_seconds=json::object();
  // Plan 11.13(4): phase-scoped RSS growth measured against the live VmRSS at
  // phase start (not a delta of two lifetime maxima).
  json phase_rss=json::object();
  std::chrono::steady_clock::time_point phase_t0;
  long long phase_rss_start_kb=0;
  auto begin=[&](Phase phase) {
    active=phase; phase_t0=std::chrono::steady_clock::now();
    phase_rss_start_kb=read_vmrss_kb();
    emitter.phase_start(run_id,phase,phase_to_string(phase),log);
  };
  auto end=[&](const json &extra=json::object()) {
    const std::string name=phase_to_string(*active);
    phase_seconds[name]=
        std::chrono::duration<double>(std::chrono::steady_clock::now()-phase_t0).count();
    const long long rss_end_kb=read_vmrss_kb();
    phase_rss[name]={{"start_kib",phase_rss_start_kb},
                     {"end_kib",rss_end_kb},
                     {"growth_kib",rss_end_kb-phase_rss_start_kb}};
    emitter.phase_end(run_id,*active,"ok",extra,log); active.reset();
  };
  try {
    if (!resume_from.empty() && resume_from!="GLOBAL_QUALITY" && resume_from!="FORWARD_DRIZZLE")
      throw std::invalid_argument("FORWARD_STAGE_UNSUPPORTED_RESUME_PHASE");
    const auto provenance=validate_provenance(dir,sampling);
    auto reconstruction_cfg=cfg.reconstruction;
    if (!reconstruction_cfg.drizzle.memory_budget_mb)
      reconstruction_cfg.drizzle.memory_budget_mb=static_cast<size_t>(std::max(1,cfg.runtime_limits.memory_budget));
    // §30.81 step 3a-3 baseline sweep: override the FORWARD_DRIZZLE memory
    // budget without editing config.yaml, so `resume-reconstruction
    // --from-phase FORWARD_DRIZZLE` keeps its checkpoint valid across a
    // budget sweep. Affects only the source-LRU capacity and the band
    // planner; the committed store is budget-invariant (CPU path bit-exact).
    if (const char *e=std::getenv("TC_FORWARD_DRIZZLE_MEMORY_BUDGET_MB")) {
      if (const long v=std::atol(e); v>=2)
        reconstruction_cfg.drizzle.memory_budget_mb=static_cast<size_t>(v);
    }
    const auto &drizzle=reconstruction_cfg.drizzle;
    const auto artifacts=dir/"artifacts";
    const bool downstream_requested = cfg.astrometry.enabled || cfg.bge.method != "none" ||
        cfg.pcc.enabled || cfg.hypermetric_stretch.enabled;
    const std::string normalization_id = downstream_requested
        ? std::to_string(fs::file_size(artifacts/"normalization.json")) : std::string();
    const auto cache_dir=dir/"cache/normalized_frames";
    const auto checkpoint_path=artifacts/"forward_drizzle_checkpoint.json";
    const auto geometry_hash=registration::compute_coverage_geometry_hash(
        sampling,drizzle,reconstruction_cfg.common_overlap_required_fraction);

    // Plan 11.14 P1/P2: authoritative local-warp geometry cache. Built once at
    // SAMPLING_GEOMETRY, then published (thread-local guard) for coverage,
    // GLOBAL_QUALITY and FORWARD_DRIZZLE so no consumer re-runs sample_leaves
    // per stripe. Affine-only runs skip it entirely (behaviour unchanged).
    const reconstruction::ForwardDrizzleSubdivisionParams geom_sub{};
    std::vector<reconstruction::GeometryVariant> geom_variants{
        {drizzle.pixfrac}};
    if (drizzle.pixfrac != 1.0f) geom_variants.push_back({1.0f});
    std::vector<reconstruction::GeometryCacheIdentity> geom_ids;
    for (const auto &gv : geom_variants)
      geom_ids.push_back(reconstruction::make_geometry_cache_identity(
          sampling, drizzle, gv, geom_sub));
    std::vector<std::size_t> geom_local_indices;
    for (const auto &f : sampling.frames)
      if (f.valid && f.has_smooth_local_model)
        geom_local_indices.push_back(f.source_index);
    std::sort(geom_local_indices.begin(), geom_local_indices.end());
    const bool geom_any_local = !geom_local_indices.empty();
    const fs::path geom_cache_root = dir / "artifacts/forward_drizzle_geometry";
    std::optional<reconstruction::DrizzleGeometryCacheReader> geom_reader;
    std::optional<reconstruction::ScopedActiveGeometryCache> geom_guard;
    auto open_geometry_cache = [&](bool from_resume) {
      // On a fresh run the build just fsync'd every file --- the .rows hash +
      // structural offset/length checks are sufficient. A resume re-reads a
      // possibly stale-on-disk cache, so it pays the full .leaves byte
      // verification (one-time, not the per-phase hot path).
      geom_reader.emplace(geom_cache_root, geom_ids, geom_local_indices,
                          /*verify_record_bytes=*/from_resume);
      geom_guard.emplace(&*geom_reader);
    };

    json checkpoint;
    json checkpoint_geometry_cache;  // filled if a local-warp geometry cache is built
    if (resume_from.empty()) {
      if (!fresh_cache) throw std::runtime_error("FORWARD_STAGE_NORMALIZED_CACHE_REQUIRED");
      begin(Phase::NORMALIZED_CACHE);
      fresh_cache->seal_normalized_cache(sampling);
      end();
      begin(Phase::SAMPLING_GEOMETRY);
      double geometry_cache_seconds=0.0;
      if (geom_any_local) {
        // Disk pre-flight (plan 11.14.3): a conservative 4 leaves/sample bound
        // per variant, x1.5 margin. Records dominate; the row index is tiny.
        const std::uint64_t sp=
            static_cast<std::uint64_t>(sampling.source_width)*sampling.source_height;
        const std::uint64_t est_bytes=
            static_cast<std::uint64_t>(geom_local_indices.size())*sp*4ull*72ull*
            geom_variants.size();
        std::error_code sec;
        const auto space=std::filesystem::space(dir,sec);
        if (!sec && space.available < est_bytes + est_bytes/2)
          throw std::runtime_error("FORWARD_STAGE_GEOMETRY_CACHE_DISK: need ~"+
              std::to_string(est_bytes/(1024*1024))+" MiB");
        // Plan 11.14.5 P3: each (variant, frame) build task is independent and
        // its per-worker footprint is one source row of records --- safe to
        // parallelise even while the reduction stays single-threaded. The
        // committed store is byte-identical to the 1-worker reference.
        // TC_GEOMETRY_CACHE_WORKERS overrides (=1 restores the reference mode).
        int geom_workers=static_cast<int>(std::thread::hardware_concurrency());
        if (geom_workers<1) geom_workers=1;
        if (const char *e=std::getenv("TC_GEOMETRY_CACHE_WORKERS")) {
          const int v=std::atoi(e);
          if (v>=1) geom_workers=v;
        }
        geom_workers=std::min<int>(geom_workers,
            std::max<std::size_t>(1,geom_local_indices.size()*geom_variants.size()));
        const auto gc0=std::chrono::steady_clock::now();
        const auto geom_progress=[&](std::size_t done,std::size_t total,const std::string &detail){
          const float progress = total ? static_cast<float>(done)/static_cast<float>(total) : 1.0f;
          emitter.phase_progress(run_id,Phase::SAMPLING_GEOMETRY,progress,detail,log);
        };
        const auto built=reconstruction::build_drizzle_geometry_cache(
            geom_cache_root,sampling,drizzle,geom_variants,geom_sub,
            static_cast<std::uint64_t>(drizzle.memory_budget_mb)<<20,geom_workers,
            geom_progress);
        geometry_cache_seconds=
            std::chrono::duration<double>(std::chrono::steady_clock::now()-gc0).count();
        open_geometry_cache(false);
        json gc;
        gc["generation"]=built.generation_dir.filename().string();
        gc["manifest_bytes"]=fs::file_size(built.generation_dir/"manifest.json");
        gc["local_source_indices"]=geom_local_indices;
        gc["variants"]=json::array();
        for (size_t i=0;i<geom_variants.size();++i)
          gc["variants"].push_back({{"pixfrac",geom_variants[i].pixfrac},
                                    {"geometry_hash",geom_ids[i].geometry_hash}});
        gc["total_leaves"]=built.total_leaves;
        gc["total_record_bytes"]=built.total_record_bytes;
        gc["workers_used"]=built.workers_used;
        gc["build_sample_leaves_seconds"]=built.sample_leaves_seconds;
        gc["build_write_seconds"]=built.write_seconds;
        checkpoint_geometry_cache=gc;
      }
      // Plan §30.72 O1: SAMPLING_GEOMETRY coverage is stripe-parallel and
      // bit-identical to the serial run. Same resolution as the geometry-cache
      // and forward-drizzle worker counts; TC_SAMPLING_GEOMETRY_WORKERS=1
      // restores the exact serial reference.
      int cov_workers=std::max(1,cfg.runtime_limits.parallel_workers);
      if (const int hw=static_cast<int>(std::thread::hardware_concurrency()); hw>=1)
        cov_workers=std::min(cov_workers,hw);
      if (const char *e=std::getenv("TC_SAMPLING_GEOMETRY_WORKERS")) {
        const int v=std::atoi(e);
        if (v>=1) cov_workers=v;
      }
      const auto coverage_progress=[&](std::size_t done,std::size_t total,const std::string &detail){
        const float progress = total ? static_cast<float>(done)/static_cast<float>(total) : 1.0f;
        emitter.phase_progress(run_id,Phase::SAMPLING_GEOMETRY,progress,detail,log);
      };
      auto coverage=registration::compute_geometric_coverage(sampling,drizzle.internal_scale,
          drizzle.pixfrac,reconstruction_cfg.coverage_gate,
          reconstruction_cfg.common_overlap_required_fraction,cov_workers,drizzle,false,
          coverage_progress);
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
      {
        json sg_extra={{"analysis_pixels",coverage.gate.analysis_pixels},
                       {"coverage_workers",coverage.gate.workers_used}};
        if (geom_any_local) {
          sg_extra["geometry_cache_seconds"]=geometry_cache_seconds;
          sg_extra["geometry_cache_leaves"]=checkpoint_geometry_cache.value("total_leaves",0);
        }
        end(sg_extra);
      }
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
      for (const auto &name:geometry_files) checkpoint["artifacts"][name]=fs::file_size(artifacts/name);
      if (!checkpoint_geometry_cache.is_null())
        checkpoint["geometry_cache"]=checkpoint_geometry_cache;
      if (downstream_requested) checkpoint["normalization_bytes"]=normalization_id;
      core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
      end();
    } else {
      checkpoint=checked_json(checkpoint_path);
      if (downstream_requested && checkpoint.value("normalization_bytes",std::string())!=normalization_id)
        throw std::runtime_error("FORWARD_STAGE_NORMALIZATION_PREDECESSOR_MISMATCH");
      if (checkpoint.at("schema_version")!=1 || checkpoint.at("execution_scope")!=scope ||
          checkpoint.at("config_sha256")!=provenance.at("config").at("sha256") ||
          checkpoint.at("sampling_plan_hash")!=sampling.plan_hash || checkpoint.at("geometry_hash")!=geometry_hash ||
          checkpoint.at("artifacts").size()!=geometry_files.size())
        throw std::runtime_error("FORWARD_STAGE_CHECKPOINT_MISMATCH");
      for (const auto &name:geometry_files)
        if (checkpoint.at("artifacts").at(name)!=fs::file_size(artifacts/name))
          throw std::runtime_error("FORWARD_STAGE_PREDECESSOR_CORRUPT: "+name);
      // Plan 11.14: re-open + verify the local-warp geometry cache before any
      // resumable phase that consumes it. The reader re-checks identity,
      // population and every record offset; the checkpoint pins the generation
      // and its manifest digest so a rebuilt / swapped cache is rejected.
      const bool ckpt_has_geom=checkpoint.contains("geometry_cache") &&
                               !checkpoint.at("geometry_cache").is_null();
      if (geom_any_local != ckpt_has_geom)
        throw std::runtime_error("FORWARD_STAGE_GEOMETRY_CACHE_PRESENCE_MISMATCH");
      if (ckpt_has_geom) {
        const auto &gc=checkpoint.at("geometry_cache");
        if (gc.at("variants").size()!=geom_ids.size())
          throw std::runtime_error("FORWARD_STAGE_GEOMETRY_CACHE_VARIANT_MISMATCH");
        for (size_t i=0;i<geom_ids.size();++i)
          if (gc.at("variants").at(i).at("geometry_hash")!=geom_ids[i].geometry_hash)
            throw std::runtime_error("FORWARD_STAGE_GEOMETRY_CACHE_IDENTITY_MISMATCH");
        std::vector<std::size_t> ckpt_idx=
            gc.at("local_source_indices").get<std::vector<std::size_t>>();
        std::sort(ckpt_idx.begin(),ckpt_idx.end());
        if (ckpt_idx!=geom_local_indices)
          throw std::runtime_error("FORWARD_STAGE_GEOMETRY_CACHE_POPULATION_MISMATCH");
        open_geometry_cache(true);  // throws on any structural / checksum fault
        const fs::path gen_dir=
            geom_cache_root/gc.at("generation").get<std::string>();
        if (!fs::is_regular_file(gen_dir/"manifest.json") ||
            fs::file_size(gen_dir/"manifest.json")!=
                gc.at("manifest_bytes").get<std::uintmax_t>())
          throw std::runtime_error("FORWARD_STAGE_GEOMETRY_CACHE_MANIFEST_CHANGED");
      }
    }
    reconstruction::VerifiedNormalizedSourceCache cache(cache_dir,sampling,drizzle.memory_budget_mb);
    if (checkpoint.at("cache_manifest_hash")!=cache.manifest_hash())
      throw std::runtime_error("FORWARD_STAGE_CACHE_MANIFEST_CHANGED");
    // T1: the pre-phase full-source retention scan is removed. In the trusted-
    // run model the cache is sealed and unchanged; SQM/GQ load on demand. The
    // parallel SQM workers use independent cache clones that would not see
    // these pre-loaded entries anyway.
    const auto qcfg=quality_config(cfg);
    const auto sqm_cache_root=dir/"cache/source_quality_maps";
    if (resume_from.empty()) {
      begin(Phase::SOURCE_QUALITY_MAPS);
      // Plan §30.72 O3: per-frame proxy + quality maps run concurrently; the
      // committed store is byte-identical. TC_SOURCE_QUALITY_WORKERS=1 restores
      // the serial reference.
      int sqm_workers=std::max(1,cfg.runtime_limits.parallel_workers);
      if (const int hw=static_cast<int>(std::thread::hardware_concurrency()); hw>=1)
        sqm_workers=std::min(sqm_workers,hw);
      if (const char *e=std::getenv("TC_SOURCE_QUALITY_WORKERS")) {
        const int v=std::atoi(e);
        if (v>=1) sqm_workers=v;
      }
      // T3: pass star detection parameters so SQM also computes per-frame
      // metrics and writes source_quality_metrics-v1.json.
      reconstruction::SourceQualityMapCacheConfig sqm_cfg;
      sqm_cfg.star_max_corners=qcfg.star_max_corners;
      sqm_cfg.star_patch_radius=qcfg.star_patch_radius;
      const auto sqm=reconstruction::build_source_quality_map_cache(
          sqm_cache_root,sampling,cache,cfg.aqmh.pyramid,sqm_cfg,sqm_workers);
      checkpoint["source_quality_identity_hash"]=sqm.source_identity_hash;
      checkpoint["source_quality_config_hash"]=sqm.source_quality_config_hash;
      checkpoint["source_quality_cache_hash"]=sqm.source_quality_cache_hash;
      core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
      end({{"frames",sqm.frames},{"computed_scales",sqm.computed_scales},
           {"streams",sqm.streams},{"workers",sqm_workers},
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
      // Plan §30.72 R4: parallel per-frame proxy/metrics; bit-identical.
      // TC_GLOBAL_QUALITY_WORKERS=1 restores the serial reference.
      int gq_workers=std::max(1,cfg.runtime_limits.parallel_workers);
      if (const int hw=static_cast<int>(std::thread::hardware_concurrency()); hw>=1)
        gq_workers=std::min(gq_workers,hw);
      if (const char *e=std::getenv("TC_GLOBAL_QUALITY_WORKERS")) {
        const int v=std::atoi(e);
        if (v>=1) gq_workers=v;
      }
      // T3: use the pre-computed metrics from SOURCE_QUALITY_MAPS instead of
      // reloading and re-running compute_source_quality_proxy_v1 per frame.
      const auto metrics_path=sqm_cache_root/"source_quality_metrics-v1.json";
      const auto quality=reconstruction::persist_source_quality_artifact(
          quality_path,sampling,cache,qcfg,metrics_path,
          drizzle.memory_budget_mb,gq_workers);
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

    // Plan 11.11.1: probe the GPU name for the throughput-baseline machine
    // BEFORE any phase begins --- the AccelerationContext ctor calls
    // cv::cuda::setDevice(), which can initialise a CUDA context, and doing that
    // mid-MULTIBAND would land after that phase's RSS start is captured and skew
    // §11.13 phase-growth accounting. The forward-drizzle CUDA path re-selects
    // the device itself, so this adds no perturbation to the actual compute.
    // Only probed when the resolved intent is GPU; never fatal.
    std::string gpu_device_name;
    if (fd_accel.using_gpu) {
      try {
        core::AccelerationContext acc(cfg.runtime_limits.acceleration_backend);
        gpu_device_name = acc.capabilities().device_name;
      } catch (...) { gpu_device_name.clear(); }
    }

    // Requested CPU parallelism. Streaming resolves a budgeted worker count
    // including concurrent clipping/alpha and geometry-reader scratch, then
    // records the actual OpenMP team size separately. CUDA ignores this count.
    int fd_workers=std::max(1,cfg.runtime_limits.parallel_workers);
    if (const int hw=static_cast<int>(std::thread::hardware_concurrency()); hw>=1)
      fd_workers=std::min(fd_workers,hw);
    if (const char *e=std::getenv("TC_FORWARD_DRIZZLE_WORKERS")) {
      const int v=std::atoi(e);
      if (v>=1) fd_workers=v;
    }
    if (const auto hw=std::thread::hardware_concurrency(); hw>0)
      fd_workers=std::min(fd_workers,static_cast<int>(hw));

    begin(Phase::FORWARD_DRIZZLE);
    const auto profiles_root=artifacts/"forward_drizzle_profiles";
    reconstruction::DrizzleStoreIdentity mb_identity;
    reconstruction::DrizzleStoreResult result;
    if (want_multiband) {
      auto built=reconstruction::persist_multiband_store_from_predecessors(
          profiles_root,quality_path,sampling,cache,qcfg,drizzle,
          reconstruction_cfg.clipping,reconstruction_cfg.multiband,sqm_cache_root,
          {},fd_backend,fd_workers);
      result=built.store;
      mb_identity=built.identity;
      fd_backend_used=built.backend_used;
      fd_cuda_fallback_reason=built.cuda_fallback_reason;
      checkpoint["multiband_reconstruction_hash"]=mb_identity.reconstruction_hash;
      checkpoint["multiband_levels"]=mb_identity.multiband_levels;
    } else {
      result=reconstruction::persist_forward_drizzle_from_predecessors(
          profiles_root,quality_path,sampling,cache,qcfg,
          drizzle,reconstruction_cfg.clipping,{},sqm_cache_root,fd_workers);
    }
    checkpoint["profiles_current_bytes"]=fs::file_size(profiles_root/"current.json");
    checkpoint["forward_drizzle_backend"]=fd_backend_used;
    // P3 Teil 2: resolved CPU-reduction worker count (informational; the store
    // commit hash is invariant to it, so it is NOT resume-validated). On the
    // CUDA stripe path the device band chunking runs instead --- the value then
    // only applies to a CUDA->CPU restart.
    checkpoint["forward_drizzle_reduction_workers_requested"]=fd_workers;
    const bool cpu_reduction=fd_backend_used=="cpu";
    checkpoint["forward_drizzle_reduction_workers"]=
        cpu_reduction ? result.diagnostics.workers_used : 0;
    checkpoint["forward_drizzle_reduction_workers_budgeted"]=
        cpu_reduction ? result.diagnostics.workers_budgeted : 0;
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
         {"reduction_workers_requested",fd_workers},
         {"reduction_workers",cpu_reduction ? result.diagnostics.workers_used : 0},
         {"reduction_worker_scratch_bytes",result.diagnostics.worker_scratch_bytes},
         {"kernel_noise_sigma_factor",
          reconstruction::kernel_noise_correlation_sigma_factor(
              drizzle.pixfrac,drizzle.internal_scale)}};
      if (!fd_cuda_fallback_reason.empty())
        extra["cuda_fallback_reason"]=fd_cuda_fallback_reason;
      // T1: source-cache diagnostics (trusted run, no SHA counters).
      extra["source_cache"]={
        {"capacity_frames",cache.capacity_frames()},
        {"frames",sampling.frames.size()},
        {"memory_budget_mb",drizzle.memory_budget_mb},
        {"load_calls",cache.load_call_count()},
        {"lru_hits",cache.lru_hit_count()},
        {"evictions",cache.eviction_count()},
        {"bytes_read",cache.bytes_read()}};
      end(extra);
    }

    // Plan 11.14 P0: persist the geometry counters + timers gathered across
    // SAMPLING_GEOMETRY and FORWARD_DRIZZLE. Diagnostic-only; no checkpoint
    // hash guard (like forward_drizzle.json).
    {
      auto geom_profile=json::parse(reconstruction::geomstats::to_json());
      // P3 Teil 2 honesty: with >1 reduction worker the process-global geomstats
      // registry is DISABLED for the FORWARD_DRIZZLE stripe loop (not
      // concurrency-safe), so the production_uniform_raw / contrib_* / hybrid
      // counters below reflect only what ran serially --- a zero there is
      // "not recorded", NOT "the cache served everything". SAMPLING_GEOMETRY
      // (coverage_*) is unaffected (still single-threaded).
      if (result.diagnostics.reduction_stats_suppressed)
        geom_profile["forward_drizzle_stage_stats_suppressed_reduction_workers"]=
            result.diagnostics.workers_budgeted;
      // P3 Teil 2 shared-budget term: the largest per-source-row leaf-record
      // block a band worker seek-reads (one such buffer per concurrent worker).
      if (geom_reader)
        geom_profile["geometry_cache_max_row_record_count"]=
            geom_reader->max_row_record_count();
      core::write_text_atomic(artifacts/"forward_drizzle_geometry_profile.json",
          geom_profile.dump(2));
    }

    bool final_image_available=false;
    if (want_multiband) {
      begin(Phase::MULTIBAND);
      // Internal artifact: the multiband X_out in one file (kept for debugging
      // and as the fuse commit target). The canonical deliverables are the
      // plan-16.1 per-channel files under outputs/, written after selection.
      const auto mb_internal=artifacts/"reconstruction_multiband.fits";
      reconstruction::MultibandCandidateLuma cand;
      // Plan 11.13(2): candidate channels stream to a phase-local scratch dir;
      // only the delivered planes are read back (one at a time) for export.
      reconstruction::MultibandCandidateSpool spool;
      spool.dir=artifacts/"multiband_candidate_spool";
      { std::error_code ec; fs::remove_all(spool.dir,ec);
        fs::create_directories(spool.dir,ec); }
      // Plan 11.13(3): the candidate spool is scratch --- remove it on ANY exit
      // from this phase (success or a mid-fuse/-delivery throw), so a failed run
      // never leaves up to 3*nch full-size .f32 planes on the temp filesystem.
      struct SpoolGuard {
        fs::path dir;
        ~SpoolGuard() { std::error_code ec; fs::remove_all(dir,ec); }
      } spool_guard{spool.dir};
      reconstruction::MultibandFusionMemoryPlan mem_plan;
      const auto pixels=reconstruction::fuse_multiband_store_to_image(
          profiles_root,mb_identity,mb_internal,reconstruction_cfg.multiband,
          drizzle.chunk_rows,drizzle.memory_budget_mb,&cand,&spool,&mem_plan);
      checkpoint["final_image_bytes"]=fs::file_size(mb_internal);

      // Plan 16.4: per-band alpha-confidence summary. `cand.alpha_final_by_band`
      // is freed with the luma buffers below (line ~`cand={}`), so the summary
      // is computed here. An empty inner vector means alpha == 1 across the band
      // (no fallback anywhere). Denominator is the luma support (every active
      // channel co-present) so the fraction is "of the pixels the reconstruction
      // actually delivered", not of the whole canvas.
      json alpha_confidence_summary=json::array();
      {
        // alpha_final_by_band[b] is populated in source_quality_artifact.cpp as
        // either empty (alpha == 1 across the band) or assign(W*H, ...) --- the
        // SAME full-resolution grid as uniform_support (also W*H). Fail loudly if
        // a future change ever breaks that so the fraction never silently means
        // nothing (a coarse alpha buffer indexed with fine-grid indices).
        const std::size_t sup_n=cand.uniform_support.size();
        long long support_px=0;
        for (auto s:cand.uniform_support) if (s) ++support_px;
        for (std::size_t b=0;b<cand.alpha_final_by_band.size();++b) {
          const auto &af=cand.alpha_final_by_band[b];
          json band={{"band",static_cast<int>(b)},
                     {"support_px",support_px}};
          if (af.empty()) {
            band["alpha_below_one_px"]=0;
            band["alpha_below_one_fraction"]=0.0;
            band["mean_alpha_on_support"]=1.0;
            band["min_alpha_on_support"]=1.0;
          } else {
            if (af.size()!=sup_n)
              throw std::runtime_error(
                  "FORWARD_STAGE_ALPHA_GRID_MISMATCH: alpha_final_by_band vs uniform_support");
            long long below=0, counted=0; double sum=0.0, mn=1.0;
            for (std::size_t i=0;i<sup_n;++i) {
              if (!cand.uniform_support[i]) continue;
              const float a=af[i];
              if (!std::isfinite(a)) continue;
              ++counted; sum+=a; if (a<mn) mn=a;
              if (a<1.0f) ++below;
            }
            band["alpha_below_one_px"]=below;
            band["alpha_below_one_fraction"]=
                counted>0 ? static_cast<double>(below)/static_cast<double>(counted) : 0.0;
            band["mean_alpha_on_support"]=
                counted>0 ? sum/static_cast<double>(counted) : 1.0;
            band["min_alpha_on_support"]=mn;
          }
          alpha_confidence_summary.push_back(std::move(band));
        }
      }

      // Plan 15: three-way candidate selection on the fixed working luminance
      // (drizzle_uniform / drizzle_raw / drizzle_multiband), stars detected
      // ONCE on the uniform control (16.3).
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

      // ---- Plan 16.1 delivery: per-channel outputs/ in output_scale geometry,
      // in the normalised linear working space (same as
      // reconstruction_multiband.fits). forward_drizzle_raw_* is the IMMUTABLE
      // Raw baseline (always written, regardless of which candidate won);
      // reconstructed_* carries the selected candidate; `full` adds the
      // uniform/multiband control FITS. These are NEW names with no downstream
      // consumer. The STACKING pass-through outputs/stacked[_rgb].fits are the
      // legacy canonical downstream entry points and MUST carry the 17.4
      // normalisation undo (scale_r/g/b, background). The downstream bridge
      // writes them after releasing the reconstruction candidate buffers.
      cand={};  // free the luma buffers; selection is done
      const auto outdir=dir/"outputs";
      { std::error_code ec; fs::create_directories(outdir,ec); }
      const std::vector<std::string> ch=
          spool.mono ? std::vector<std::string>{"L"}
                     : std::vector<std::string>{"R","G","B"};
      json out_list=json::array();
      auto record=[&](const fs::path &p){
        out_list.push_back({{"path","outputs/"+p.filename().string()},
                            {"size",static_cast<long long>(fs::file_size(p))}});
      };
      // Plan 11.13(2): each delivered plane is read back from the spool one at a
      // time (peak = one plane), written straight to FITS, then freed.
      auto emit_set=[&](const std::string &prefix,const std::string &candidate){
        for (std::size_t c=0;c<ch.size();++c) {
          const auto plane=reconstruction::read_candidate_spool_plane(
              spool,candidate,static_cast<int>(c));
          const auto p=outdir/(prefix+"_"+ch[c]+".fit");
          io::FitsHeader h;
          io::write_fits_float_rows(p,plane,spool.height,spool.width,h);
          record(p);
        }
      };
      const std::string sel_candidate=
          sel.selected==reconstruction::SelectedCandidate::kDrizzleUniform ? "uniform"
          : sel.selected==reconstruction::SelectedCandidate::kDrizzleRaw ? "raw"
          : "multiband";
      emit_set("forward_drizzle_raw","raw");
      emit_set("reconstructed",sel_candidate);
      if (reconstruction_cfg.diagnostics.level=="full") {
        emit_set("forward_drizzle_uniform","uniform");
        emit_set("forward_drizzle_multiband","multiband");
      }
      checkpoint["outputs"]=json::object();
      for (const auto &o:out_list)
        checkpoint["outputs"][o.at("path").get<std::string>()]=o.at("size");

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
      // Plan 16.4 mandatory diagnostics --- only fields actually measured on
      // this run are emitted; per-kernel timing / retries belong to M7 slice 2
      // and stay absent rather than as a misleading {}.
      // Plan 11.13(4): the estimated working set and the phase-scoped RSS peak
      // are reported SEPARATELY. Phase growth is VmRSS-now minus VmRSS at the
      // MULTIBAND phase start (`phase_rss_start_kb`), never a delta of two
      // process-lifetime maxima.
      const long long rss_maxrss_kb=read_maxrss_kb();          // ru_maxrss, lifetime
      const long long rss_process_peak_kb=read_vmhwm_kb();     // VmHWM, lifetime
      const long long mb_phase_rss_now_kb=read_vmrss_kb();
      const long long mb_phase_rss_growth_kb=
          mb_phase_rss_now_kb-phase_rss_start_kb;
      const long long mb_budget_mb=
          static_cast<long long>(drizzle.memory_budget_mb
              ? drizzle.memory_budget_mb : 256);
      // Plan 11.11 envelope: growth <= budget*1.05 + 256 MiB.
      const long long mb_envelope_kb=
          static_cast<long long>(mb_budget_mb*1024*1.05)+256*1024;
      const bool mb_growth_within_envelope=
          mb_phase_rss_growth_kb<=mb_envelope_kb;
      json local_warp={
        {"local_model_samples_total",result.diagnostics.local_model_samples_total},
        {"local_model_samples_discarded",
         result.diagnostics.local_model_samples_discarded}};
      { json fx=json::array();
        for (const auto &fr:result.diagnostics.frames_excluded_subdivision_error_rate)
          fx.push_back({{"frame_id",fr.first},{"inversion_error_rate",fr.second}});
        local_warp["frames_excluded_subdivision_error_rate"]=fx; }
      json fwd={
        // v2 (2026-09-08, plan M8): additive -- throughput, runtime_environment,
        // flux_space, alpha_confidence_summary. No existing key changed shape.
        {"schema_version",2},
        {"pipeline_method","cfa_forward_drizzle_multiband"},
        {"pipeline_contract_version",1},
        {"sampling_plan_hash",sampling.plan_hash},
        {"coverage_geometry_hash",geometry_hash},
        {"multiband_reconstruction_hash",mb_identity.reconstruction_hash},
        {"multiband_levels",mb_identity.multiband_levels},
        {"luma_definition",reconstruction::kWorkingLumaDefinition},
        {"geometry",{
          {"source_width",sampling.source_width},
          {"source_height",sampling.source_height},
          {"canvas_width_native",sampling.canvas_width_native},
          {"canvas_height_native",sampling.canvas_height_native},
          {"reconstruction_width",mb_identity.width},
          {"reconstruction_height",mb_identity.height},
          {"internal_scale",drizzle.internal_scale},
          {"output_scale",drizzle.output_scale},
          {"output_scale_applied",applied_2x2},
          {"kernel",drizzle.kernel},
          {"pixfrac",drizzle.pixfrac}}},
        {"clipping",{
          {"pixel_channel_evaluations",result.clipping.pixel_channel_evaluations},
          {"pixel_channel_rejected",result.clipping.pixel_channel_rejected},
          {"candidate_contributions_clipped",
           result.clipping.candidate_contributions_clipped},
          {"pixel_channel_guard_fallback",
           result.clipping.pixel_channel_guard_fallback}}},
        {"local_warp",local_warp},
        {"pixels_supported",pixels},
        {"acceleration",{
          {"forward_drizzle_backend",fd_backend_used},
          {"cuda_fallback_reason",
           fd_cuda_fallback_reason.empty() ? json(nullptr)
                                           : json(fd_cuda_fallback_reason)},
          {"workers_used",cpu_reduction ? result.diagnostics.workers_used : 0},
          {"resolved_chunk_rows",result.diagnostics.resolved_chunk_rows},
          // Plan 19.4/19.6: populated only when the CUDA per-stripe path
          // (accumulate_pair_by_frame_cuda via run_cuda_chunked) actually ran.
          {"cuda_stripe_path",
           result.cuda_timing.used
               ? json{{"bands",result.cuda_timing.bands},
                      {"resolved_chunk_rows",
                       result.cuda_timing.resolved_chunk_rows},
                      {"min_chunk_rows",result.cuda_timing.min_chunk_rows},
                      {"bytes_per_row",
                       static_cast<long long>(result.cuda_timing.bytes_per_row)},
                      // §30.80: three-way split of bytes_per_row + host ceiling
                      // + observed band collapse.
                      {"cand_row_bytes",
                       static_cast<long long>(
                           result.cuda_timing.cand_row_bytes)},
                      {"rec_row_bytes",
                       static_cast<long long>(result.cuda_timing.rec_row_bytes)},
                      {"acc_row_bytes",
                       static_cast<long long>(result.cuda_timing.acc_row_bytes)},
                      {"host_budget_bytes",
                       static_cast<long long>(
                           result.cuda_timing.host_budget_bytes)},
                      {"band_halvings",result.cuda_timing.band_halvings},
                      {"min_band_rows",result.cuda_timing.min_band_rows},
                      {"max_band_rows",result.cuda_timing.max_band_rows},
                      // §30.81: per-band column tiling of the host buffer
                      {"resolved_tile_w",result.cuda_timing.resolved_tile_w},
                      {"min_tile_w",result.cuda_timing.min_tile_w},
                      {"max_tiles_per_band",
                       result.cuda_timing.max_tiles_per_band},
                      {"device_free_bytes",
                       static_cast<long long>(
                           result.cuda_timing.device_free_bytes)},
                      {"stripe_seconds",result.cuda_timing.stripe_seconds},
                      {"total_seconds",result.cuda_timing.total_seconds},
                      {"hybrid_local_frames",
                       result.cuda_timing.hybrid_local_frames},
                      {"hybrid_cpu_seconds",
                       result.cuda_timing.hybrid_cpu_seconds},
                      {"hybrid_gpu_raster_seconds",
                       result.cuda_timing.hybrid_gpu_raster_seconds},
                      {"hybrid_leaf_cells",
                       static_cast<long long>(
                           result.cuda_timing.hybrid_leaf_cells)}}
               : json(nullptr)}}},
        {"resources",{
          // FORWARD_DRIZZLE store-build estimate (unchanged).
          {"estimated_peak_bytes",result.diagnostics.estimated_peak_bytes},
          // Plan 11.13(1)+(4): the MULTIBAND fuse/validate/export working-set
          // pre-plan that gated the phase, reported next to (not merged with)
          // the measured RSS.
          {"multiband_estimated_working_set_bytes",
           static_cast<long long>(mem_plan.estimated_peak_bytes)},
          {"multiband_working_set_breakdown",{
            {"final_image_bytes",
             static_cast<long long>(mem_plan.final_image_bytes)},
            {"stripe_working_bytes",
             static_cast<long long>(mem_plan.stripe_working_bytes)},
            {"candidate_luma_bytes",
             static_cast<long long>(mem_plan.candidate_luma_bytes)},
            {"spool_stripe_bytes",
             static_cast<long long>(mem_plan.spool_stripe_bytes)},
            {"delivery_readback_bytes",
             static_cast<long long>(mem_plan.delivery_readback_bytes)},
            {"margin_bytes",static_cast<long long>(mem_plan.margin_bytes)}}},
          {"multiband_working_set_budget_bytes",
           static_cast<long long>(mem_plan.budget_bytes)},
          {"multiband_working_set_fits_budget",mem_plan.fits},
          // Plan 11.11 temp space for the candidate spool.
          {"multiband_spool_temp_bytes",
           static_cast<long long>(mem_plan.spool_temp_bytes)},
          {"multiband_required_free_temp_bytes",
           static_cast<long long>(mem_plan.required_free_temp_bytes)},
          {"multiband_available_temp_bytes",
           static_cast<long long>(mem_plan.available_temp_bytes)},
          {"multiband_temp_space_ok",mem_plan.temp_space_ok},
          // Process-lifetime maxima --- LABELLED as such, never differenced for
          // phase growth.
          {"rss_process_peak_kib",rss_process_peak_kb},
          {"rss_maxrss_kib",rss_maxrss_kb},
          {"rss_baseline_kib",rss_baseline_kb},
          // Phase-scoped: VmRSS at MULTIBAND start vs. now. Computed inline (not
          // read from `phase_rss` below) because this JSON is assembled before
          // the MULTIBAND phase `end()` runs, so `phase_rss` has no MULTIBAND
          // entry yet.
          {"multiband_phase_rss_start_kib",phase_rss_start_kb},
          {"multiband_phase_rss_now_kib",mb_phase_rss_now_kb},
          {"multiband_phase_rss_growth_kib",mb_phase_rss_growth_kb},
          {"multiband_phase_rss_envelope_kib",mb_envelope_kb},
          {"multiband_phase_rss_within_envelope",mb_growth_within_envelope},
          {"phase_rss",phase_rss},
          {"memory_budget_mb",
           static_cast<long long>(drizzle.memory_budget_mb)}}},
        {"timing_seconds",phase_seconds},
        // Plan 11.11 / 11.11.1: the throughput gate is
        //   throughput = processed_source_samples / forward_drizzle_wall_seconds
        // on a NAMED reference machine. `processed_source_samples` is the nominal
        // input CFA sample count (frames actually in the plan x sensor pixels) --
        // a stable, reproducible denominator that needs no hot-loop counter; a
        // post-mask refinement, if added later, is a NEW field and does not
        // redefine this metric.
        {"throughput",{
          {"frames_used",static_cast<long long>(sampling.frames.size())},
          {"source_width",sampling.source_width},
          {"source_height",sampling.source_height},
          {"processed_source_samples",
           static_cast<long long>(sampling.frames.size())*
               static_cast<long long>(sampling.source_width)*
               static_cast<long long>(sampling.source_height)},
          {"forward_drizzle_wall_seconds",
           phase_seconds.value("FORWARD_DRIZZLE",0.0)},
          {"source_samples_per_second",
           phase_seconds.value("FORWARD_DRIZZLE",0.0)>0.0
               ? (static_cast<double>(sampling.frames.size())*
                  static_cast<double>(sampling.source_width)*
                  static_cast<double>(sampling.source_height))/
                     phase_seconds.value("FORWARD_DRIZZLE",0.0)
               : 0.0}}},
        // Plan 11.11 / 23.1: the machine the throughput baseline is pinned to.
        // Build provenance comes from the generated build-info header; hardware
        // and thread counts are read here so a report can state "same machine?"
        // without a second artifact.
        {"runtime_environment",{
          {"build",core::build_info_json(/*include_runtime_binary=*/false)},
          {"hardware",{
            {"cpu_model",read_cpu_model_name()},
            {"logical_cores",
             static_cast<long long>(std::thread::hardware_concurrency())},
            // Emitted only when a CUDA forward-drizzle path actually committed;
            // a run that resolved GPU then fell back to CPU reports gpu:null.
            {"gpu",(!gpu_device_name.empty() && fd_backend_used.rfind("cuda",0)==0)
                       ? json(gpu_device_name) : json(nullptr)}}},
          {"threads",{
            {"parallel_workers_config",cfg.runtime_limits.parallel_workers},
            {"workers_used",cpu_reduction ? result.diagnostics.workers_used : 0}}}}},
        // Plan 23.1 (M4 carry-over -> M8 report): state the flux space of the
        // delivered planes unambiguously. The STACKING 17.4 normalisation undo
        // is applied only to the separate canonical downstream products.
        {"flux_space",{
          {"space","normalised_linear_working"},
          {"luma_definition",reconstruction::kWorkingLumaDefinition},
          {"same_space_as","reconstruction_multiband.fits"},
          {"stacking_normalisation_undo_applied",false},
          {"note","Normalized reconstruction products; restored downstream inputs are recorded separately in forward_downstream_inputs.json"}}},
        {"alpha_confidence_summary",alpha_confidence_summary},
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
        {"outputs",out_list},
        {"commit_complete",true}};
      core::write_text_atomic(artifacts/"forward_drizzle.json",fwd.dump(2));
      // No checkpoint hash guard for forward_drizzle.json: MULTIBAND fully
      // regenerates it on every (re)run, so there is nothing to verify on
      // resume (unlike the immutable geometry predecessors).
      checkpoint["selected_candidate"]=sel_name;
      checkpoint["status"]="final_image_ready";
      core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
      // (spool_guard removes the scratch spool on scope exit.)
      final_image_available=true;
      end({{"final_image",std::string("outputs/reconstructed_")+ch.front()+".fit"},
           {"outputs_count",static_cast<int>(out_list.size())},
           {"pixels_supported",pixels},
           {"selected_candidate",sel_name},
           {"selection_reason",sel.reason},
           {"stars_total",sel.stars_total},
           {"multiband_levels",mb_identity.multiband_levels}});
    } else {
      checkpoint["status"]="reconstruction_ready";
      core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
    }

    if (final_image_available && downstream_requested) {
      if (sampling.color_mode == ColorMode::OSC && cfg.hypermetric_stretch.enabled &&
          (!cfg.pcc.enabled || !cfg.astrometry.enabled))
        throw std::runtime_error("FORWARD_HMS_REQUIRES_ASTROMETRY_AND_PCC");
      begin(Phase::STACKING);
      write_forward_downstream_inputs(dir,sampling,drizzle,&emitter,run_id,&log);
      end({{"mode","forward_drizzle_pass_through"},
           {"photometry_applied_once",true},{"output_scale",drizzle.output_scale}});
      checkpoint["downstream_status"]="running";
      core::write_text_atomic(checkpoint_path,checkpoint.dump(2));
      if (sampling.color_mode == ColorMode::OSC) {
        const auto started=std::chrono::steady_clock::now();
        const auto abort_downstream=[&](const std::string &) {
          return std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()
              > cfg.runtime_limits.hard_abort_hours*3600.0;
        };
        if (run_rgb_downstream(dir,run_id,cfg,"ASTROMETRY",log,abort_downstream,true)!=0)
          throw std::runtime_error("FORWARD_DOWNSTREAM_FAILED");
      } else {
        for (const auto phase : {Phase::ASTROMETRY,Phase::BGE,Phase::PCC,Phase::HYPERMETRIC_STRETCH}) {
          begin(phase);
          emitter.phase_end(run_id,phase,"skipped",{{"reason","mono_rgb_downstream_not_applicable"}},log);
          active.reset();
        }
      }
      checkpoint["downstream_status"]="complete";
      if (sampling.color_mode == ColorMode::OSC && cfg.hypermetric_stretch.enabled) {
        fs::path hms_path(cfg.hypermetric_stretch.output_rgb);
        if (hms_path.is_relative()) hms_path=dir/"outputs"/hms_path;
        checkpoint["final_output"]={{"path",hms_path.string()},
            {"bytes",fs::file_size(hms_path)}};
      }
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
