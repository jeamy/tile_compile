#include "../apps/runner_forward_drizzle.hpp"
#include "../apps/runner_downstream.hpp"
#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/reconstruction/normalized_source_cache.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_driver.hpp"
#include "tile_compile/reconstruction/forward_drizzle_v2_store.hpp"
#include "tile_compile/reconstruction/multiband_validation.hpp"
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
using namespace tile_compile;

TEST_CASE("forward downstream restores RGB photometry once and preserves raw inputs",
          "[forward-downstream]") {
  core::AtomicOutput temporary(fs::temp_directory_path()/"forward-output-test");
  const auto dir=temporary.path();
  fs::create_directories(dir/"outputs");
  fs::create_directories(dir/"artifacts");
  registration::RegistrationSamplingPlan sampling;
  sampling.color_mode=ColorMode::OSC;
  sampling.bayer_pattern=BayerPattern::RGGB;
  sampling.source_width=sampling.source_height=8;
  sampling.canvas_width_native=sampling.canvas_height_native=8;
  config::ReconstructionDrizzleConfig cfg;
  cfg.internal_scale=2; cfg.output_scale=1; cfg.memory_budget_mb=32;
  core::write_text_atomic(dir/"artifacts/run_provenance.json",
      core::json({{"input_manifest",{{"entries",core::json::array()}}}}).dump());
  core::write_text_atomic(dir/"artifacts/normalization.json",
      core::json({{"P_r",{2.0}},{"P_g",{3.0}},{"P_b",{4.0}},
                  {"B_r",{10.0}},{"B_g",{20.0}},{"B_b",{30.0}}}).dump());
  for (const auto *c : {"R","G","B"}) {
    io::write_fits_float(dir/"outputs"/(std::string("reconstructed_")+c+".fit"),
        Matrix2Df::Constant(8,8,5.0f),{});
    io::write_fits_float(dir/"outputs"/(std::string("forward_drizzle_raw_")+c+".fit"),
        Matrix2Df::Constant(8,8,7.0f),{});
  }
  auto mask=Matrix2Df::Ones(16,16).eval(); mask(0,0)=0;
  io::write_fits_float(dir/"artifacts/sampling_geometry_analysis_common_mask.fits",mask,{});
  const auto input_hash=core::sha256_file(dir/"outputs/reconstructed_R.fit");
  const auto raw_hash=core::sha256_file(dir/"outputs/forward_drizzle_raw_R.fit");
  runner::write_forward_downstream_inputs(dir,sampling,cfg);
  const auto once=core::sha256_file(dir/"outputs/stacked_rgb.fits");
  runner::write_forward_downstream_inputs(dir,sampling,cfg);
  REQUIRE(core::sha256_file(dir/"outputs/stacked_rgb.fits")==once);
  REQUIRE(core::sha256_file(dir/"outputs/reconstructed_R.fit")==input_hash);
  REQUIRE(core::sha256_file(dir/"outputs/forward_drizzle_raw_R.fit")==raw_hash);
  const auto rgb=io::read_fits_rgb(dir/"outputs/stacked_rgb.fits");
  REQUIRE(rgb.R(3,3)==20.0f); REQUIRE(rgb.G(3,3)==35.0f); REQUIRE(rgb.B(3,3)==50.0f);
  const auto overlap=io::read_fits_pixels_float(dir/"outputs/common_overlap_mask.fits");
  REQUIRE(overlap(0,0)==0); REQUIRE(overlap(1,1)==1);
  config::Config downstream;
  downstream.astrometry.enabled=false; downstream.bge.method="none";
  downstream.pcc.enabled=false; downstream.hypermetric_stretch.enabled=false;
  std::ostringstream events;
  REQUIRE(runner::run_rgb_downstream(dir,"synthetic",downstream,"ASTROMETRY",events,
      [](const std::string &){return false;})==0);
  REQUIRE(events.str().find("DEBAYER")==std::string::npos);
  core::write_text_atomic(dir/"artifacts/normalization.json","{}");
  REQUIRE_THROWS(runner::write_forward_downstream_inputs(dir,sampling,cfg));
  REQUIRE(core::sha256_file(dir/"outputs/stacked_rgb.fits")==once);
}

namespace {
// Process-global fault injection; disarm even if a REQUIRE throws.
struct CudaFaultGuard {
  explicit CudaFaultGuard(int n) {
    reconstruction::set_forward_drizzle_v2_cuda_fault_after_bands(n);
  }
  ~CudaFaultGuard() {
    reconstruction::set_forward_drizzle_v2_cuda_fault_after_bands(-1);
  }
  CudaFaultGuard(const CudaFaultGuard &) = delete;
  CudaFaultGuard &operator=(const CudaFaultGuard &) = delete;
};
// Plan 11.14.5 P3 Teil 2: pin the FORWARD_DRIZZLE CPU-reduction worker count
// for a test (=1 restores the serial reference path; process-global env).
struct ForwardDrizzleWorkersEnvGuard {
  explicit ForwardDrizzleWorkersEnvGuard(const char *v) {
    if (v) ::setenv("TC_FORWARD_DRIZZLE_WORKERS", v, 1);
    else ::unsetenv("TC_FORWARD_DRIZZLE_WORKERS");
  }
  ~ForwardDrizzleWorkersEnvGuard() { ::unsetenv("TC_FORWARD_DRIZZLE_WORKERS"); }
  ForwardDrizzleWorkersEnvGuard(const ForwardDrizzleWorkersEnvGuard &) = delete;
  ForwardDrizzleWorkersEnvGuard &
  operator=(const ForwardDrizzleWorkersEnvGuard &) = delete;
};
struct Fixture {
  core::AtomicOutput staging{fs::temp_directory_path()/"runner-forward-test"};
  fs::path dir=staging.path();
  config::Config cfg;
  registration::RegistrationSamplingPlan plan;
  std::unique_ptr<runner::RunnerFrameCache> cache;
  Fixture() {
    fs::create_directories(dir/"artifacts"); fs::create_directories(dir/"logs");
    const std::string yaml=R"(astrometry:
  enabled: false
bge:
  method: none
pcc:
  enabled: false
hypermetric_stretch:
  enabled: false
data:
  color_mode: MONO
runtime_limits:
  memory_budget: 32
reconstruction:
  drizzle:
    internal_scale: 1
    output_scale: 1
    pixfrac: 1.0
    memory_budget_mb: 32
    chunk_rows: 3
  coverage_gate:
    min_channel_n_eff_floor: 1.0
    min_analysis_pixels: 16
  clipping:
    min_n_eff: 1.0
)";
    core::write_text_atomic(dir/"config.yaml",yaml);
    cfg=config::Config::from_yaml_text(yaml);
    const auto config_hash=core::sha256_file(dir/"config.yaml");
    const std::string identity="synthetic-input:"+config_hash;
    plan.source_identity_hash=core::sha256_bytes(std::vector<uint8_t>(identity.begin(),identity.end()));
    plan.source_width=plan.source_height=32;
    plan.canvas_width_native=plan.canvas_height_native=32;
    plan.color_mode=ColorMode::MONO;
    cache=std::make_unique<runner::RunnerFrameCache>(dir/"cache/normalized_frames",2,32,32);
    for (size_t i=0;i<2;++i) {
      registration::FrameSamplingTransform frame;
      frame.frame_id=plan.source_identity_hash+":"+std::to_string(i);
      frame.source_index=i; frame.valid=frame.source_to_canvas_affine_valid=true;
      plan.frames.push_back(frame);
      cache->store_normalized(i,Matrix2Df::Constant(32,32,10.0f+i));
    }
    plan.plan_hash=registration::compute_plan_hash(plan);
    core::write_text_atomic(dir/"artifacts/registration_sampling.json",registration::serialize_to_json_string(plan));
    core::write_text_atomic(dir/"artifacts/run_provenance.json",core::json({
        {"execution_scope","forward_drizzle_m1_m3"},{"config",{{"sha256",config_hash}}},
        {"input_manifest",{{"sha256","synthetic-input"}}}}).dump());
  }
  ~Fixture() { cache.reset(); std::error_code ec; fs::remove_all(dir,ec); }
  bool execute(std::ostream &out,const std::string &resume="") {
    core::EventEmitter emitter;
    return runner::run_forward_drizzle_stages("test",cfg,dir,plan,
                                             resume.empty()?cache.get():nullptr,emitter,out,resume);
  }
};
std::vector<core::json> events(const std::string &text) {
  std::istringstream lines(text); std::string line; std::vector<core::json> result;
  while (std::getline(lines,line)) if (!line.empty()) result.push_back(core::json::parse(line));
  return result;
}
}
TEST_CASE("forward downstream resume from HYPERMETRIC_STRETCH reuses the PCC output",
          "[forward-runner][forward-downstream]") {
  core::AtomicOutput temporary(fs::temp_directory_path()/"forward-hms-resume-test");
  const auto dir=temporary.path();
  fs::create_directories(dir/"outputs");
  fs::create_directories(dir/"artifacts");
  fs::create_directories(dir/"logs");
  // The resume contract: persisted reconstruction RGB + linear PCC output +
  // the canvas/overlap masks HMS consumes.
  const auto plane=Matrix2Df::Constant(8,8,10.0f);
  io::write_fits_rgb(dir/"outputs/stacked_rgb.fits",plane,plane,plane,{});
  io::write_fits_rgb(dir/"outputs/stacked_rgb_pcc.fits",plane,plane,plane,{});
  const auto mask=Matrix2Df::Ones(8,8).eval();
  io::write_fits_float(dir/"outputs/canvas_mask.fits",mask,{});
  io::write_fits_float(dir/"outputs/common_overlap_mask.fits",mask,{});
  config::Config cfg;
  cfg.astrometry.enabled=false; cfg.bge.method="none"; cfg.pcc.enabled=false;
  cfg.hypermetric_stretch.enabled=true;
  std::ostringstream evlog;
  REQUIRE(runner::run_rgb_downstream(dir,"synthetic",cfg,"HYPERMETRIC_STRETCH",
      evlog,[](const std::string &){return false;})==0);
  REQUIRE(fs::exists(dir/"outputs/stacked_rgb_hms.fits"));
  int hms_starts=0; bool hms_ok=false; bool saw_downstream_end=false;
  for (const auto &e:events(evlog.str())) {
    if (e["type"]=="phase_start"&&e["phase_name"]=="HYPERMETRIC_STRETCH") ++hms_starts;
    if (e["type"]=="phase_end"&&e["phase_name"]=="HYPERMETRIC_STRETCH"&&e["status"]=="ok") hms_ok=true;
    if (e["type"]=="downstream_end") saw_downstream_end=e.value("success",false);
  }
  REQUIRE(hms_starts==1);  // exactly one phase_start for the resume entry
  REQUIRE(hms_ok);
  REQUIRE(saw_downstream_end);
}

TEST_CASE("forward downstream resume from HYPERMETRIC_STRETCH fails closed "
          "without the PCC output",
          "[forward-runner][forward-downstream]") {
  core::AtomicOutput temporary(fs::temp_directory_path()/"forward-hms-resume-missing");
  const auto dir=temporary.path();
  fs::create_directories(dir/"outputs");
  fs::create_directories(dir/"artifacts");
  fs::create_directories(dir/"logs");
  const auto plane=Matrix2Df::Constant(8,8,10.0f);
  io::write_fits_rgb(dir/"outputs/stacked_rgb.fits",plane,plane,plane,{});
  config::Config cfg;
  cfg.astrometry.enabled=false; cfg.bge.method="none"; cfg.pcc.enabled=false;
  cfg.hypermetric_stretch.enabled=true;
  std::ostringstream evlog;
  REQUIRE(runner::run_rgb_downstream(dir,"synthetic",cfg,"HYPERMETRIC_STRETCH",
      evlog,[](const std::string &){return false;})!=0);
  REQUIRE_FALSE(fs::exists(dir/"outputs/stacked_rgb_hms.fits"));
  bool saw_failure=false;
  for (const auto &e:events(evlog.str()))
    if (e["type"]=="downstream_end"&&e.value("status",std::string())=="missing_pcc_rgb")
      saw_failure=true;
  REQUIRE(saw_failure);
}

TEST_CASE("forward runner: ordered phases retain cache and never create prewarp frames", "[forward-runner]") {
  Fixture f; std::ostringstream log;
  REQUIRE(f.execute(log));
  std::vector<std::string> started,ended;
  for (const auto &event:events(log.str())) {
    if (event["type"]=="phase_start") started.push_back(event["phase_name"]);
    if (event["type"]=="phase_end") {
      REQUIRE(event["status"]=="ok"); ended.push_back(event["phase_name"]);
    }
  }
  // multiband is the default single method: FORWARD_DRIZZLE persists the
  // uniform+raw+fine+(medium)+alpha store, MULTIBAND fuses it to X_out.
  const std::vector<std::string> expected={"NORMALIZED_CACHE","SAMPLING_GEOMETRY","COMMON_OVERLAP","SOURCE_QUALITY_MAPS","GLOBAL_QUALITY","FORWARD_DRIZZLE","MULTIBAND"};
  REQUIRE(started==expected); REQUIRE(ended==expected);
  REQUIRE(events(log.str()).back()["final_image_available"]==true);
  REQUIRE(events(log.str()).back()["status"]=="final_image_ready");
  {
    // Plan 19 / §30.54: the FORWARD_DRIZZLE phase records the acceleration
    // backend it actually ran on. This affine 1/1 fixture runs on the CUDA
    // banded v2 path when the host has a usable device, otherwise cpu_v2.
    core::json fd_end;
    for (const auto &event:events(log.str()))
      if (event["type"]=="phase_end" && event["phase_name"]=="FORWARD_DRIZZLE") fd_end=event;
    const std::string be=fd_end.at("acceleration_backend").get<std::string>();
    REQUIRE((be=="cpu_v2" || be=="cuda_v2"));
    // The phase_end event carries cuda_fallback_reason only when a CUDA attempt
    // fell back; a committed CUDA build omits it.
    if (be=="cuda_v2") REQUIRE_FALSE(fd_end.contains("cuda_fallback_reason"));
  }
  REQUIRE(fs::exists(f.dir/"artifacts/reconstruction_multiband.fits"));
  {
    const auto img=io::read_fits_pixels_float(f.dir/"artifacts/reconstruction_multiband.fits");
    REQUIRE(img.rows()==32);
    REQUIRE(img.cols()==32);
    int finite=0;
    for (int y=0;y<img.rows();++y) for (int x=0;x<img.cols();++x)
      if (std::isfinite(img(y,x))) ++finite;
    REQUIRE(finite>0);  // the interior is reconstructed
  }
  {
    // Cache-lifetime contract (plan 16.2, v2 cutover): the v2 band store is the
    // only reconstruction store and is retained after a committed final image;
    // the source caches are retained (resume-reconstruction stays possible).
    // run_end reports it.
    const auto re=events(log.str()).back();
    REQUIRE(re["cache_retention"]["profile_cache"]=="retained");
    REQUIRE(re["cache_retention"]["source_cache"]=="retained");
    REQUIRE(fs::exists(f.dir/"artifacts/forward_drizzle_v2/current.json"));
    REQUIRE_FALSE(fs::exists(f.dir/"artifacts/forward_drizzle_profiles"));
    REQUIRE(fs::exists(f.dir/"cache/normalized_frames"));
  }
  {
    // Plan 15 three-way selection is RECORDED in forward_drizzle.json and on
    // the MULTIBAND phase-end event. The fixture is 2 constant frames + an
    // all-NaN artefact stream => alpha == 0 => multiband == raw, and the
    // near-constant control has ~0 background RMS => that mandatory safety
    // metric is non-applicable => Raw drops to the Uniform control.
    REQUIRE(fs::exists(f.dir/"artifacts/forward_drizzle.json"));
    // Plan 11.14 P0: the geometry profile is written on every successful stage
    // and carries at least the production + coverage geometry variants.
    REQUIRE(fs::exists(f.dir/"artifacts/forward_drizzle_geometry_profile.json"));
    {
      std::ifstream gp(f.dir/"artifacts/forward_drizzle_geometry_profile.json");
      const auto gj=core::json::parse(gp);
      REQUIRE(gj.at("context").at("prepared_frames").get<int>()>0);
      REQUIRE(gj.at("variants").is_object());
      REQUIRE_FALSE(gj.at("variants").empty());
      bool any_leaves=false;
      for (auto it=gj.at("variants").begin();it!=gj.at("variants").end();++it)
        any_leaves=any_leaves||it.value().at("leaf_cells_emitted").get<long long>()>0;
      REQUIRE(any_leaves);
    }
    std::ifstream fj(f.dir/"artifacts/forward_drizzle.json");
    const auto j=core::json::parse(fj);
    const std::string sel=j.at("selected_candidate");
    INFO("selected_candidate="<<sel<<" reason="<<j.at("selection_reason"));
    REQUIRE((sel=="drizzle_uniform"||sel=="drizzle_raw"||sel=="drizzle_multiband"));
    REQUIRE(sel=="drizzle_uniform");
    REQUIRE(j.at("fallback_reason").is_string());  // non-multiband => a reason
    REQUIRE(j.at("luma_definition")=="0.25R+0.50G+0.25B");
    REQUIRE(j.at("validation").at("drizzle_uniform").contains("median_fwhm"));
    // The selection is reproducible via a hash of its versioned constants +
    // effective config, WITHOUT touching the drizzle-store identity hash.
    REQUIRE(j.at("validation").at("validation_config_hash").get<std::string>().size()==64);
    REQUIRE(j.at("validation").at("validation_config_hash")==
            reconstruction::multiband_validation_config_hash());
    core::json phase_end;
    for (const auto &event:events(log.str()))
      if (event["type"]=="phase_end" && event["phase_name"]=="MULTIBAND") phase_end=event;
    REQUIRE(phase_end.at("selected_candidate")==sel);

    // Plan 16.1 delivery (MONO fixture): the immutable Raw baseline + the
    // selected candidate land under outputs/. The canonical stacked[_rgb].fits
    // pass-through is deferred to M10 (needs the 17.4 normalisation undo), so
    // it must NOT be written here.
    REQUIRE(fs::exists(f.dir/"outputs/forward_drizzle_raw_L.fit"));
    REQUIRE(fs::exists(f.dir/"outputs/reconstructed_L.fit"));
    REQUIRE_FALSE(fs::exists(f.dir/"outputs/stacked.fits"));
    REQUIRE_FALSE(fs::exists(f.dir/"outputs/stacked_rgb.fits"));
    // summary diagnostics => no uniform/multiband control FITS.
    REQUIRE_FALSE(fs::exists(f.dir/"outputs/forward_drizzle_uniform_L.fit"));
    REQUIRE_FALSE(fs::exists(f.dir/"outputs/forward_drizzle_multiband_L.fit"));
    // forward_drizzle.json lists every delivered file with size + sha256.
    const auto outs=j.at("outputs");
    REQUIRE(outs.is_array());
    REQUIRE(outs.size()==2);
    for (const auto &o:outs) {
      REQUIRE(o.at("path").get<std::string>().rfind("outputs/",0)==0);
      REQUIRE(o.at("size").get<long long>()>0);
    }
    // The checkpoint keys resume on the same delivered set.
    std::ifstream cf(f.dir/"artifacts/forward_drizzle_checkpoint.json");
    const auto ck=core::json::parse(cf);
    REQUIRE(ck.at("outputs").size()==2);
    REQUIRE(ck.at("outputs").contains("outputs/reconstructed_L.fit"));

    // Plan 16.4 mandatory diagnostics that this run actually measures.
    REQUIRE(j.at("geometry").at("internal_scale").get<int>()>=1);
    REQUIRE(j.at("geometry").at("kernel").is_string());
    REQUIRE(j.at("geometry").at("reconstruction_width")==32);
    REQUIRE(j.at("clipping").contains("pixel_channel_evaluations"));
    {
      const std::string be=
          j.at("acceleration").at("forward_drizzle_backend").get<std::string>();
      REQUIRE((be=="cpu_v2" || be=="cuda_v2"));
      if (be=="cuda_v2")
        REQUIRE(j.at("acceleration").at("cuda_fallback_reason").is_null());
    }
    // null when no CUDA attempt was made or the device path committed (affine,
    // hybrid local-warp, or mode-2/1 with the host 2x2 fold); a string only when
    // a CUDA build resolved and then fell back (e.g. no usable device).
    REQUIRE((j.at("acceleration").at("cuda_fallback_reason").is_null()||
             j.at("acceleration").at("cuda_fallback_reason").is_string()));
    // Plan 11.13(4): lifetime peak and phase-scoped growth are distinct fields.
    REQUIRE(j.at("resources").at("rss_process_peak_kib").get<long long>()>0);
    REQUIRE(j.at("resources").at("memory_budget_mb").get<long long>()>0);
    REQUIRE(j.at("resources").at("multiband_estimated_working_set_bytes")
                .get<long long>()>0);
    REQUIRE(j.at("resources").at("multiband_working_set_fits_budget")
                .get<bool>());
    REQUIRE(j.at("resources").contains("multiband_phase_rss_growth_kib"));
    REQUIRE(j.at("resources").contains("multiband_working_set_breakdown"));
    REQUIRE(j.at("resources").at("multiband_phase_rss_within_envelope")
                .get<bool>());
    REQUIRE(j.at("resources").at("multiband_temp_space_ok").get<bool>());
    // Plan 11.13(3): the candidate spool is scratch and is gone after the run.
    REQUIRE_FALSE(fs::exists(f.dir/"artifacts/multiband_candidate_spool"));
    REQUIRE(j.at("timing_seconds").contains("FORWARD_DRIZZLE"));
    REQUIRE(j.at("timing_seconds").at("FORWARD_DRIZZLE").get<double>()>=0.0);
    REQUIRE(j.at("pixels_supported").get<long long>()>0);

    // Plan M8 / schema v2: throughput denominator + reference-machine block +
    // flux space + per-band alpha summary.
    REQUIRE(j.at("schema_version").get<int>()==2);
    {
      const auto &tp=j.at("throughput");
      REQUIRE(tp.at("frames_used").get<long long>()>0);
      const long long pss=tp.at("processed_source_samples").get<long long>();
      REQUIRE(pss==tp.at("frames_used").get<long long>()*
                   tp.at("source_width").get<long long>()*
                   tp.at("source_height").get<long long>());
      REQUIRE(tp.at("forward_drizzle_wall_seconds").get<double>()>=0.0);
      REQUIRE(tp.at("source_samples_per_second").get<double>()>=0.0);
    }
    {
      const auto &re=j.at("runtime_environment");
      REQUIRE(re.at("build").at("toolchain").at("build_type").is_string());
      REQUIRE(re.at("hardware").contains("cpu_model"));
      REQUIRE(re.at("hardware").at("logical_cores").get<long long>()>=0);
      REQUIRE((re.at("hardware").at("gpu").is_null()||
               re.at("hardware").at("gpu").is_string()));
      const auto backend=j.at("acceleration").at("forward_drizzle_backend").get<std::string>();
      const int cpu_workers=re.at("threads").at("workers_used").get<int>();
      if (backend.rfind("cuda",0)==0) REQUIRE(cpu_workers==0);
      else REQUIRE(cpu_workers>=1);
    }
    {
      const auto &fx=j.at("flux_space");
      REQUIRE(fx.at("space")=="normalised_linear_working");
      REQUIRE(fx.at("stacking_normalisation_undo_applied").get<bool>()==false);
    }
    {
      const auto &as=j.at("alpha_confidence_summary");
      REQUIRE(as.is_array());
      for (const auto &b:as) {
        REQUIRE(b.at("support_px").get<long long>()>=0);
        const double f=b.at("alpha_below_one_fraction").get<double>();
        REQUIRE((f>=0.0 && f<=1.0));
        const double m=b.at("mean_alpha_on_support").get<double>();
        REQUIRE((m>=0.0 && m<=1.0+1e-9));
      }
    }
  }
  REQUIRE_FALSE(fs::exists(f.dir/"cache/prewarped_frames"));
  REQUIRE_THROWS(f.cache->store_normalized(0,Matrix2Df::Ones(32,32)));
  f.cache.reset();
  reconstruction::VerifiedNormalizedSourceCache kept(f.dir/"cache/normalized_frames",f.plan,32);
  REQUIRE(kept.load(0).minCoeff()==10.0f);
  REQUIRE(phase_to_int(Phase::PREWARP)==2);
  REQUIRE(phase_to_int(Phase::FORWARD_DRIZZLE)==27);
  REQUIRE(phase_to_int(Phase::SOURCE_QUALITY_MAPS)==28);
  REQUIRE(phase_to_int(Phase::MULTIBAND)==29);
}
TEST_CASE("forward runner: an injected FORWARD_DRIZZLE CUDA fault restarts the "
          "phase on CPU and still delivers the final image (plan 19.4)",
          "[forward-runner]") {
  if (core::select_acceleration_backend(
          "auto", core::AccelerationPhase::forward_drizzle)
          .selected != core::AccelerationBackend::cuda) {
    SUCCEED("cuda backend not buildable here; the restart path is covered "
            "bit-exact by test_drizzle_profile_store");
    return;
  }
  Fixture f; std::ostringstream log;
  {
    CudaFaultGuard guard(0);
    REQUIRE(f.execute(log));
  }
  core::json fd_end;
  for (const auto &event : events(log.str()))
    if (event["type"] == "phase_end" && event["phase_name"] == "FORWARD_DRIZZLE")
      fd_end = event;
  REQUIRE(fd_end.at("acceleration_backend") == "cpu_v2");
  REQUIRE(fd_end.at("cuda_fallback_reason").get<std::string>().find(
              "injected fault") != std::string::npos);
  REQUIRE(events(log.str()).back()["status"] == "final_image_ready");
  REQUIRE(fs::exists(f.dir / "artifacts/reconstruction_multiband.fits"));

  // The restarted (CPU) run must be identical to a clean CPU run.
  Fixture clean; std::ostringstream clean_log;
  REQUIRE(clean.execute(clean_log));
  REQUIRE(core::sha256_file(f.dir / "artifacts/reconstruction_multiband.fits") ==
          core::sha256_file(clean.dir / "artifacts/reconstruction_multiband.fits"));
}

TEST_CASE("forward runner: diagnostics.level and profile-cache retention do not "
          "change the computed result (plan M8 acceptance)",
          "[forward-runner]") {
  // §23.1 / M8: "`summary`/`full` und Profilcache-Retention verändern keine
  // Rechenergebnisse." diagnostics.level=full only writes EXTRA control FITS;
  // keep_profile_cache_after_run / delete_source_cache_after_run only act on
  // caches AFTER the committed image. None feeds the store math.
  // Scope: this proves retention does NOT change the run that produces the
  // store. It does not exercise a *subsequent* re-fuse consuming a retained
  // profile store -- that is a separate case if it ever matters.
  struct Run {
    std::string final_image_sha, multiband_fits_sha, reconstructed_sha,
        raw_sha, selected;
    core::json cache_retention;
  };
  auto run_with = [](const std::string &level, bool keep_profile,
                     bool delete_source) {
    Fixture f;
    f.cfg.reconstruction.diagnostics.level = level;
    f.cfg.reconstruction.keep_profile_cache_after_run = keep_profile;
    f.cfg.reconstruction.delete_source_cache_after_run = delete_source;
    std::ostringstream log;
    REQUIRE(f.execute(log));
    const auto tail = events(log.str()).back();
    REQUIRE(tail["status"] == "final_image_ready");
    std::ifstream cf(f.dir / "artifacts/forward_drizzle_checkpoint.json");
    const auto ck = core::json::parse(cf);
    Run r;
    r.final_image_sha = std::to_string(ck.at("final_image_bytes").get<std::uintmax_t>());
    r.multiband_fits_sha =
        core::sha256_file(f.dir / "artifacts/reconstruction_multiband.fits");
    r.reconstructed_sha =
        core::sha256_file(f.dir / "outputs/reconstructed_L.fit");
    r.raw_sha = core::sha256_file(f.dir /
                                  "outputs/forward_drizzle_raw_L.fit");
    std::ifstream fj(f.dir / "artifacts/forward_drizzle.json");
    r.selected = core::json::parse(fj).at("selected_candidate").get<std::string>();
    r.cache_retention = tail["cache_retention"];
    // full adds the two control sets; summary must not have written them.
    const bool full_extras =
        fs::exists(f.dir / "outputs/forward_drizzle_uniform_L.fit");
    REQUIRE(full_extras == (level == "full"));
    return r;
  };

  const Run base = run_with("summary", /*keep*/ false, /*del_src*/ false);
  const Run full = run_with("full", /*keep*/ false, /*del_src*/ false);
  const Run kept = run_with("summary", /*keep*/ true, /*del_src*/ false);
  const Run full_kept = run_with("full", /*keep*/ true, /*del_src*/ false);

  for (const Run *r : {&full, &kept, &full_kept}) {
    REQUIRE(r->final_image_sha == base.final_image_sha);
    REQUIRE(r->multiband_fits_sha == base.multiband_fits_sha);
    REQUIRE(r->reconstructed_sha == base.reconstructed_sha);
    REQUIRE(r->raw_sha == base.raw_sha);
    REQUIRE(r->selected == base.selected);
  }
  // Retention flags act only on the caches, and are announced. The v2 band
  // store is the only reconstruction store and is always retained.
  REQUIRE(base.cache_retention["profile_cache"] == "retained");
  REQUIRE(kept.cache_retention["profile_cache"] == "retained");
  REQUIRE(base.cache_retention["source_cache"] == "retained");

  // delete_source_cache_after_run: same computed result, source cache gone,
  // resume-reconstruction announced as disabled.
  const Run del_src = run_with("summary", /*keep*/ false, /*del_src*/ true);
  REQUIRE(del_src.final_image_sha == base.final_image_sha);
  REQUIRE(del_src.multiband_fits_sha == base.multiband_fits_sha);
  REQUIRE(del_src.cache_retention["source_cache"] == "deleted");
  REQUIRE(del_src.cache_retention["resume_reconstruction_disabled"] == true);
}

TEST_CASE("forward runner: geometry veto never completes overlap or reconstruction", "[forward-runner]") {
  Fixture f;
  f.cfg.reconstruction.coverage_gate.min_channel_n_eff_floor=3;
  std::ostringstream log;
  REQUIRE_FALSE(f.execute(log));
  for (const auto &event:events(log.str()))
    if (event["type"]=="phase_start") {
      REQUIRE(event["phase_name"]!="COMMON_OVERLAP");
      REQUIRE(event["phase_name"]!="GLOBAL_QUALITY");
      REQUIRE(event["phase_name"]!="FORWARD_DRIZZLE");
    }
  REQUIRE_FALSE(fs::exists(f.dir/"artifacts/forward_drizzle_checkpoint.json"));
}
TEST_CASE("forward runner: resume validates predecessors before starting a phase", "[forward-runner]") {
  Fixture f; std::ostringstream first;
  f.cfg.reconstruction.keep_profile_cache_after_run=true;  // inspect the store across runs
  REQUIRE(f.execute(first));
  const auto current=f.dir/"artifacts/forward_drizzle_v2/current.json";
  std::ostringstream resumed;
  REQUIRE(f.execute(resumed,"FORWARD_DRIZZLE"));
  std::vector<std::string> starts;
  for (const auto &event:events(resumed.str())) if (event["type"]=="phase_start") starts.push_back(event["phase_name"]);
  REQUIRE(starts==(std::vector<std::string>{"FORWARD_DRIZZLE","MULTIBAND"}));
  // The v2 driver may reuse the committed generation or republish a new one;
  // either way the published store stays valid.
  REQUIRE(fs::exists(current));
  const auto valid=core::sha256_file(current);
  // T1 trusted run: same-size content change is NOT detected. Truncate instead.
  { std::ofstream file(f.dir/"cache/normalized_frames/0.raw",std::ios::binary|std::ios::trunc); file<<"short"; }
  std::ostringstream rejected;
  REQUIRE_FALSE(f.execute(rejected,"FORWARD_DRIZZLE"));
  for (const auto &event:events(rejected.str())) REQUIRE(event["type"]!="phase_start");
  REQUIRE(core::sha256_file(current)==valid);
}
TEST_CASE("forward runner: changed config or geometric artifact rejects resume", "[forward-runner]") {
  Fixture f; std::ostringstream first; REQUIRE(f.execute(first));
  core::write_text_atomic(f.dir/"artifacts/forward_common_overlap.json","{}");
  std::ostringstream rejected;
  REQUIRE_FALSE(f.execute(rejected,"GLOBAL_QUALITY"));
  for (const auto &event:events(rejected.str())) REQUIRE(event["type"]!="phase_start");
}

// Plan 11.14 P1/P2 --- a run with LOCAL-WARP frames builds the geometry cache
// at SAMPLING_GEOMETRY, publishes it for the downstream phases (so no consumer
// re-runs sample_leaves per stripe), records it in the checkpoint, re-opens +
// verifies it on resume, and rejects a tampered cache.
namespace {
struct LocalWarpFixture {
  core::AtomicOutput staging{fs::temp_directory_path()/"runner-fd-localwarp"};
  fs::path dir=staging.path();
  config::Config cfg;
  registration::RegistrationSamplingPlan plan;
  std::unique_ptr<runner::RunnerFrameCache> cache;
  LocalWarpFixture() {
    fs::create_directories(dir/"artifacts"); fs::create_directories(dir/"logs");
    const std::string yaml=R"(astrometry:
  enabled: false
bge:
  method: none
pcc:
  enabled: false
hypermetric_stretch:
  enabled: false
data:
  color_mode: MONO
runtime_limits:
  memory_budget: 32
reconstruction:
  drizzle:
    internal_scale: 1
    output_scale: 1
    pixfrac: 0.8
    memory_budget_mb: 32
    chunk_rows: 3
  coverage_gate:
    min_channel_n_eff_floor: 1.0
    min_analysis_pixels: 16
  clipping:
    min_n_eff: 1.0
)";
    core::write_text_atomic(dir/"config.yaml",yaml);
    cfg=config::Config::from_yaml_text(yaml);
    const auto config_hash=core::sha256_file(dir/"config.yaml");
    const std::string identity="synthetic-input:"+config_hash;
    plan.source_identity_hash=core::sha256_bytes(std::vector<uint8_t>(identity.begin(),identity.end()));
    plan.source_width=plan.source_height=32;
    plan.canvas_width_native=plan.canvas_height_native=48;
    plan.color_mode=ColorMode::MONO;
    cache=std::make_unique<runner::RunnerFrameCache>(dir/"cache/normalized_frames",3,32,32);
    for (size_t i=0;i<3;++i) {
      registration::FrameSamplingTransform frame;
      frame.frame_id=plan.source_identity_hash+":"+std::to_string(i);
      frame.source_index=i; frame.valid=frame.source_to_canvas_affine_valid=true;
      const float tx=8.0f+0.2f*static_cast<float>(i), ty=8.0f-0.1f*static_cast<float>(i);
      frame.source_to_canvas.setZero();
      frame.source_to_canvas(0,0)=1.0f; frame.source_to_canvas(1,1)=1.0f;
      frame.source_to_canvas(0,2)=tx; frame.source_to_canvas(1,2)=ty;
      frame.canvas_to_source.setZero();
      frame.canvas_to_source(0,0)=1.0f; frame.canvas_to_source(1,1)=1.0f;
      frame.canvas_to_source(0,2)=-tx; frame.canvas_to_source(1,2)=-ty;
      frame.has_smooth_local_model=true;
      frame.smooth_local_model.valid=true;
      frame.smooth_local_model.image_rows=48;
      frame.smooth_local_model.image_cols=48;
      frame.smooth_local_model.coeff_x.setZero();
      frame.smooth_local_model.coeff_y.setZero();
      frame.smooth_local_model.coeff_x[0]=0.10f+0.02f*i;
      frame.smooth_local_model.coeff_y[0]=-0.07f;
      frame.model_coordinate_scale=1.0f;
      plan.frames.push_back(frame);
      cache->store_normalized(i,Matrix2Df::Constant(32,32,10.0f+i));
    }
    plan.plan_hash=registration::compute_plan_hash(plan);
    core::write_text_atomic(dir/"artifacts/registration_sampling.json",registration::serialize_to_json_string(plan));
    core::write_text_atomic(dir/"artifacts/run_provenance.json",core::json({
        {"execution_scope","forward_drizzle_m1_m3"},{"config",{{"sha256",config_hash}}},
        {"input_manifest",{{"sha256","synthetic-input"}}}}).dump());
  }
  ~LocalWarpFixture() { cache.reset(); std::error_code ec; fs::remove_all(dir,ec); }
  bool execute(std::ostream &out,const std::string &resume="") {
    core::EventEmitter emitter;
    return runner::run_forward_drizzle_stages("test",cfg,dir,plan,
        resume.empty()?cache.get():nullptr,emitter,out,resume);
  }
};
} // namespace

TEST_CASE("forward runner: local-warp geometry cache is built, published, "
          "checkpointed and re-verified on resume (plan 11.14 P1/P2)",
          "[forward-runner][geometry-cache]") {
  // Pin the serial reduction path: the FORWARD_DRIZZLE stripe-consumer geomstats
  // counters this test inspects (production_uniform_raw etc.) are only recorded
  // at 1 reduction worker (the registry is disabled for band parallelism).
  ForwardDrizzleWorkersEnvGuard reduction_workers("1");
  LocalWarpFixture f;
  f.cfg.reconstruction.keep_profile_cache_after_run=true;
  std::ostringstream first;
  const bool ok_first=f.execute(first);
  INFO("first run log:\n"<<first.str());
  REQUIRE(ok_first);

  // Cache materialised + recorded.
  REQUIRE(fs::exists(f.dir/"artifacts/forward_drizzle_geometry/current.json"));
  const auto ckpt=core::json::parse(std::ifstream(
      f.dir/"artifacts/forward_drizzle_checkpoint.json"));
  REQUIRE(ckpt.contains("geometry_cache"));
  const auto &gc=ckpt.at("geometry_cache");
  REQUIRE(gc.at("variants").size()==2u);            // pixfrac 0.8 + footprint 1.0
  REQUIRE(gc.at("local_source_indices").size()==3u);
  REQUIRE(gc.at("total_leaves").get<long long>()>0);

  // The published cache means no consumer re-ran the local geometry: the P0
  // profile shows zero sample_leaves / basis evals for the stripe consumers.
  const auto prof=core::json::parse(std::ifstream(
      f.dir/"artifacts/forward_drizzle_geometry_profile.json"));
  const auto &vars=prof.at("variants");
  for (const char *name : {"production_uniform_raw","coverage_cfa",
                           "coverage_footprint","prepare_exclusion_scan"}) {
    if (!vars.contains(name)) continue;
    INFO("variant "<<name);
    REQUIRE(vars.at(name).at("invert_iterations").get<long long>()==0);
    REQUIRE(vars.at(name).at("top_level_sample_leaves_calls").get<long long>()==0);
  }

  // Resume from FORWARD_DRIZZLE: the cache is re-opened + fully re-verified.
  const auto current=f.dir/"artifacts/forward_drizzle_v2/current.json";
  const auto before=core::sha256_file(current);
  std::ostringstream resumed;
  REQUIRE(f.execute(resumed,"FORWARD_DRIZZLE"));
  std::vector<std::string> starts;
  for (const auto &e:events(resumed.str()))
    if (e["type"]=="phase_start") starts.push_back(e["phase_name"]);
  REQUIRE(starts==(std::vector<std::string>{"FORWARD_DRIZZLE","MULTIBAND"}));
  // Re-drizzle from the same verified cache -> the store is rewritten but the
  // run succeeds; the geometry did not change.
  REQUIRE(fs::exists(current));

  // Tamper with the committed cache -> resume must refuse before any phase.
  {
    const fs::path gen=f.dir/"artifacts/forward_drizzle_geometry"/
        gc.at("generation").get<std::string>();
    bool hit=false;
    for (const auto &e:fs::directory_iterator(gen)) {
      if (e.path().extension()!=".leaves") continue;
      // T1 trusted run: truncate to change size (same-size corruption not detected).
      fs::resize_file(e.path(),fs::file_size(e.path())-8);
      hit=true; break;
    }
    REQUIRE(hit);
  }
  std::ostringstream rejected;
  REQUIRE_FALSE(f.execute(rejected,"FORWARD_DRIZZLE"));
  for (const auto &e:events(rejected.str())) REQUIRE(e["type"]!="phase_start");
}

TEST_CASE("forward runner: FORWARD_DRIZZLE reports zero reduction workers "
          "under v2 and leaves the committed store bit-identical",
          "[forward-runner][geometry-cache][geometry-parallel]") {
  // Exercise the CPU scheduler even on GPU hosts; CUDA has no CPU row workers.
  CudaFaultGuard force_cpu_fallback(0);
  // current.json embeds the clock-derived generation name, so it differs
  // run-to-run even for identical content. Hash the committed band-record
  // files instead --- that is the actual pixel payload.
  auto content_digest = [](const fs::path &store_root) {
    const auto cur = core::json::parse(std::ifstream(store_root / "current.json"));
    const fs::path gen = store_root / cur.at("generation").get<std::string>();
    std::vector<std::string> shas;
    for (const auto &e : fs::directory_iterator(gen))
      if (e.path().extension() == ".bin")
        shas.push_back(e.path().filename().string() + ":" +
                       core::sha256_file(e.path()));
    std::sort(shas.begin(), shas.end());
    std::string joined;
    for (const auto &s : shas) joined += s + "\n";
    REQUIRE(shas.size() >= 1u);
    return core::sha256_bytes(
        std::vector<uint8_t>(joined.begin(), joined.end()));
  };

  auto run_at = [&](const char *workers) {
    ForwardDrizzleWorkersEnvGuard env(workers);
    LocalWarpFixture f;
    f.cfg.reconstruction.keep_profile_cache_after_run = true;
    std::ostringstream out;
    const bool ok = f.execute(out);
    INFO("workers=" << workers << " log:\n" << out.str());
    REQUIRE(ok);
    const auto digest =
        content_digest(f.dir / "artifacts/forward_drizzle_v2");
    const auto ckpt = core::json::parse(std::ifstream(
        f.dir / "artifacts/forward_drizzle_checkpoint.json"));
    REQUIRE(ckpt.contains("forward_drizzle_reduction_workers"));
    const auto prof = core::json::parse(std::ifstream(
        f.dir / "artifacts/forward_drizzle_geometry_profile.json"));
    struct R {
      std::string digest;
      int workers;
      bool suppressed_note;
    };
    return R{digest,
             ckpt.at("forward_drizzle_reduction_workers").get<int>(),
             prof.contains(
                 "forward_drizzle_stage_stats_suppressed_reduction_workers")};
  };

  const auto ref = run_at("1");
  // The v2 path has no reduction workers: the checkpoint reports 0 regardless
  // of the requested count.
  REQUIRE(ref.workers == 0);
  REQUIRE_FALSE(ref.suppressed_note);
  for (const char *w : {"2", "4"}) {
    const auto got = run_at(w);
    INFO("workers=" << w);
    REQUIRE(got.workers == 0);
    REQUIRE(got.digest == ref.digest);  // store commit invariant to worker count
    REQUIRE_FALSE(got.suppressed_note);
  }
}

TEST_CASE("forward downstream normalization provenance is checked before resume events",
          "[forward-runner][forward-downstream]") {
  Fixture f;
  f.cfg.astrometry.enabled=true;
  f.cfg.hypermetric_stretch.enabled=false;
  const auto normalization=f.dir/"artifacts/normalization.json";
  core::write_text_atomic(normalization,
      core::json({{"P_mono",{2.0}}, {"B_mono",{7.0}}}).dump());
  std::ostringstream log;
  REQUIRE(f.execute(log));
  REQUIRE(fs::exists(f.dir/"outputs/stacked.fits"));
  REQUIRE_FALSE(fs::exists(f.dir/"outputs/stacked_rgb.fits"));
  const auto raw=core::sha256_file(f.dir/"outputs/forward_drizzle_raw_L.fit");
  const auto checkpoint=core::json::parse(core::read_text(f.dir/"artifacts/forward_drizzle_checkpoint.json"));
  REQUIRE(checkpoint.at("normalization_bytes")==std::to_string(fs::file_size(normalization)));
  // T1 trusted run: size-based check. Write a different-sized normalization.
  core::write_text_atomic(normalization,
      core::json({{"P_mono",{200.0}}, {"B_mono",{7.0}}}).dump());
  std::ostringstream resumed;
  REQUIRE_FALSE(f.execute(resumed,"FORWARD_DRIZZLE"));
  for (const auto &e:events(resumed.str())) REQUIRE(e["type"]!="phase_start");
  REQUIRE(core::sha256_file(f.dir/"outputs/forward_drizzle_raw_L.fit")==raw);
}

TEST_CASE("forward runner uses the banded v2 path and MULTIBAND v2 store "
          "unconditionally",
          "[forward-runner][gate10]") {
  // Gate-10 wiring contract: the v2 producer commits a Gate-7 band store
  // (artifacts/forward_drizzle_v2) unconditionally --- with
  // TC_FORWARD_DRIZZLE_V2 unset --- the phase checkpoint binds its plan_hash,
  // and MULTIBAND fuses it through the v2 adapter without any legacy profile
  // store.
  Fixture f;
  std::ostringstream log;
  REQUIRE(f.execute(log));
  REQUIRE(events(log.str()).back()["status"] == "final_image_ready");

  // The v2 producer committed a published Gate-7 generation even with
  // TC_FORWARD_DRIZZLE_V2 unset.
  const auto v2_root = f.dir / "artifacts/forward_drizzle_v2";
  REQUIRE(fs::exists(v2_root / "current.json"));
  reconstruction::ForwardDrizzleV2RunPlan detected;
  std::string load_error;
  REQUIRE(reconstruction::load_forward_drizzle_v2_published_plan(
      v2_root, detected, load_error));
  const auto insp =
      reconstruction::inspect_forward_drizzle_v2_store(v2_root, detected);
  REQUIRE(insp.status ==
          reconstruction::ForwardDrizzleV2StoreStatus::complete);
  REQUIRE(insp.committed.size() ==
          static_cast<std::size_t>(detected.band_count));

  // The FORWARD_DRIZZLE phase-end event records the selection + backend.
  core::json fd_end;
  for (const auto &event : events(log.str()))
    if (event["type"] == "phase_end" &&
        event["phase_name"] == "FORWARD_DRIZZLE")
      fd_end = event;
  REQUIRE(fd_end.at("forward_drizzle_v2").get<bool>());
  const std::string be =
      fd_end.at("acceleration_backend").get<std::string>();
  REQUIRE((be == "cpu_v2" || be == "cuda_v2"));

  // Honest aggregate telemetry contract (performance tranche 1): device
  // frame timing, provider/enqueue wall timing and the Q/source counters.
  for (const char *key : {"v2_max_frame_seconds",
                          "v2_max_provider_enqueue_seconds",
                          "v2_quality_bytes_uploaded",
                          "v2_source_samples_launched",
                          "v2_quality_frames_processed",
                          "v2_upload_seconds", "v2_kernel_seconds",
                          "v2_download_seconds", "v2_reserved_device_bytes",
                          "v2_frames_skipped_empty_window",
                          "v2_quality_expanded_floats",
                          "v2_phase_wall_seconds", "v2_commit_seconds",
                          "v2_provider_source_seconds",
                          "v2_provider_quality_seconds",
                          "v2_provider_source_bytes_read",
                          "v2_provider_source_read_calls",
                          "v2_provider_hotpath_allocations",
                          "v2_workspace_reservations",
                          "v2_band_resets",
                          "v2_driver_hotpath_allocations",
                          "v2_provider_quality_cells_read",
                          "v2_provider_quality_expanded_floats",
                          "v2_provider_quality_denominator_bytes",
                          "v2_source_read_amplification",
                          "v2_source_read_amplification_applicable",
                          "v2_launched_sample_amplification",
                          "v2_launched_sample_amplification_applicable",
                          "v2_quality_read_amplification",
                          "v2_quality_read_amplification_applicable",
                          // Tranche-6 committed-geometry telemetry.
                          "v2_geometry_leaf_records_read",
                          "v2_geometry_leaf_record_bytes_read",
                          "v2_geometry_unique_source_samples",
                          "v2_geometry_cache_enumerations",
                          "v2_cached_leaf_records_launched",
                          "v2_cached_leaf_bytes_uploaded",
                          // Tranche-7/8 affine telemetry.
                          "v2_affine_pieces_processed",
                          "v2_empty_affine_tiles",
                          "v2_target_tile_cols_native",
                          "v2_target_tile_cols_native_applicable",
                          "v2_affine_samples_processed",
                          "v2_affine_span_rows",
                          "v2_affine_sample_path"})
    REQUIRE(fd_end.contains(key));

  // Tranche-2 band-aware source windows: launched source samples and the
  // affine source/Q reads are bounded by the exact per-band scan boxes, never
  // the full frame per band.
  const std::uint64_t src_area = 32ull * 32ull;
  const std::uint64_t full_baseline =
      detected.frame_count *
      static_cast<std::uint64_t>(detected.band_count) * src_area;
  const std::uint64_t launched =
      fd_end.at("v2_source_samples_launched").get<std::uint64_t>();
  REQUIRE(launched > 0);
  REQUIRE(launched <= full_baseline);
  REQUIRE(fd_end.at("v2_frames_skipped_empty_window")
              .get<std::uint64_t>() <=
          detected.frame_count *
              static_cast<std::uint64_t>(detected.band_count));
  REQUIRE(fd_end.at("source_cache").at("bytes_read")
              .get<std::uint64_t>() <= full_baseline * sizeof(float));
  // Tranche-4 hot path: provider buffers are pre-reserved outside the calls,
  // so no provider call may grow a reusable buffer.
  REQUIRE(fd_end.at("v2_provider_hotpath_allocations")
              .get<std::uint64_t>() == 0);
  // Tranche-5 persistent workspace: exactly one reservation per backend
  // attempt, one begin_band per computed band, no driver record-buffer
  // growth.
  REQUIRE(fd_end.at("v2_workspace_reservations").get<std::uint64_t>() == 1);
  REQUIRE(fd_end.at("v2_reserve_allocations").get<std::uint64_t>() == 1);
  REQUIRE(fd_end.at("v2_band_resets").get<std::uint64_t>() ==
          static_cast<std::uint64_t>(detected.band_count));
  REQUIRE(fd_end.at("v2_driver_hotpath_allocations")
              .get<std::uint64_t>() == 0);
  REQUIRE(fd_end.at("v2_provider_quality_expanded_floats")
              .get<std::uint64_t>() == 0);
  REQUIRE(fd_end.at("v2_provider_source_read_calls")
              .get<std::uint64_t>() > 0);
  // Tranche 8: production affine frames run through the canonical ragged
  // sample list (one full-target piece per frame); the compatibility
  // target-tile key is explicitly not applicable.
  REQUIRE(fd_end.at("v2_affine_sample_path").get<bool>());
  REQUIRE(fd_end.at("v2_target_tile_cols_native").is_null());
  REQUIRE_FALSE(
      fd_end.at("v2_target_tile_cols_native_applicable").get<bool>());
  REQUIRE(fd_end.at("v2_affine_samples_processed").get<std::uint64_t>() >
          0);
  REQUIRE(fd_end.at("v2_affine_samples_processed")
              .get<std::uint64_t>() <=
          fd_end.at("v2_source_samples_launched").get<std::uint64_t>());
  REQUIRE(fd_end.at("v2_affine_span_rows").get<std::uint64_t>() > 0);
  REQUIRE(fd_end.at("v2_affine_pieces_processed").get<std::uint64_t>() ==
          0);

  // The checkpoint carries the plan hash --- a plan/config change fails
  // closed on resume.
  const auto ck = core::json::parse(core::read_text(
      f.dir / "artifacts/forward_drizzle_checkpoint.json"));
  REQUIRE(ck.at("forward_drizzle_v2").get<bool>());
  REQUIRE(ck.at("forward_drizzle_v2_plan_hash").get<std::string>() ==
          detected.plan_hash);
  REQUIRE_FALSE(
      ck.at("forward_drizzle_v2_commit_hash").get<std::string>().empty());
  REQUIRE(ck.at("forward_drizzle_v2_commit_hash").get<std::string>() ==
          insp.commit_hash);

  // MULTIBAND auto-detected the published store and fused the final image;
  // no legacy profile store was produced.
  REQUIRE(fs::exists(f.dir / "artifacts/reconstruction_multiband.fits"));
  REQUIRE_FALSE(fs::exists(f.dir / "artifacts/forward_drizzle_profiles"));
}

TEST_CASE("forward runner: v2 consumes the committed geometry cache for "
          "local frames and fails closed without it (tranche 6)",
          "[forward-runner][geometry-cache][gate10]") {
  // Keep the serial reduction path so the geomstats consumer counters are
  // recorded honestly.
  ForwardDrizzleWorkersEnvGuard workers("1");
  LocalWarpFixture f;
  std::ostringstream log;
  const bool ok = f.execute(log);
  INFO("log:\n" << log.str());
  REQUIRE(ok);

  // The v2 producer consumed committed cache leaves for all three local
  // frames; no inversion/subdivision ran inside FORWARD_DRIZZLE.
  core::json fd_end;
  for (const auto &e : events(log.str()))
    if (e["type"] == "phase_end" && e["phase_name"] == "FORWARD_DRIZZLE")
      fd_end = e;
  REQUIRE(fd_end.at("forward_drizzle_v2").get<bool>());
  const std::uint64_t read =
      fd_end.at("v2_geometry_leaf_records_read").get<std::uint64_t>();
  const std::uint64_t launched =
      fd_end.at("v2_cached_leaf_records_launched").get<std::uint64_t>();
  REQUIRE(read > 0);
  REQUIRE(read == launched);
  REQUIRE(fd_end.at("v2_geometry_leaf_record_bytes_read")
              .get<std::uint64_t>() >=
          read * 72u);
  REQUIRE(fd_end.at("v2_geometry_cache_enumerations").get<std::uint64_t>() >
          0);
  REQUIRE(fd_end.at("v2_geometry_unique_source_samples")
              .get<std::uint64_t>() > 0);
  REQUIRE(fd_end.at("v2_cached_leaf_bytes_uploaded").get<std::uint64_t>() >=
          launched * 80u);
  // The FORWARD_DRIZZLE phase never re-ran the local geometry: zero
  // sample_leaves calls recorded for the stripe consumers.
  const auto prof = core::json::parse(std::ifstream(
      f.dir / "artifacts/forward_drizzle_geometry_profile.json"));
  for (const char *name : {"production_uniform_raw", "coverage_cfa",
                           "coverage_footprint"}) {
    if (!prof.at("variants").contains(name)) continue;
    INFO("variant " << name);
    REQUIRE(prof.at("variants").at(name).at("top_level_sample_leaves_calls")
                .get<long long>() == 0);
    REQUIRE(prof.at("variants").at(name).at("invert_iterations")
                .get<long long>() == 0);
  }
  // The store completed.
  const auto v2_root = f.dir / "artifacts/forward_drizzle_v2";
  reconstruction::ForwardDrizzleV2RunPlan detected;
  std::string load_error;
  REQUIRE(reconstruction::load_forward_drizzle_v2_published_plan(
      v2_root, detected, load_error));
  REQUIRE(reconstruction::inspect_forward_drizzle_v2_store(v2_root, detected)
              .status ==
          reconstruction::ForwardDrizzleV2StoreStatus::complete);

  // Fail-closed: remove the committed cache; a resume must refuse before
  // any phase starts (a local frame cannot be served without it).
  fs::remove_all(f.dir / "artifacts/forward_drizzle_geometry");
  std::ostringstream rejected;
  REQUIRE_FALSE(f.execute(rejected, "FORWARD_DRIZZLE"));
  for (const auto &e : events(rejected.str()))
    REQUIRE(e["type"] != "phase_start");
}
