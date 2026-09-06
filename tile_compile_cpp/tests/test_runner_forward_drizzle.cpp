#include "../apps/runner_forward_drizzle.hpp"
#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/core/acceleration.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/reconstruction/normalized_source_cache.hpp"
#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"
#include "tile_compile/reconstruction/multiband_validation.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <sstream>
#include <fstream>
using namespace tile_compile;
namespace {
// Process-global fault injection; disarm even if a REQUIRE throws.
struct CudaFaultGuard {
  explicit CudaFaultGuard(int n) {
    reconstruction::set_forward_drizzle_cuda_fault_after_chunks(n);
  }
  ~CudaFaultGuard() {
    reconstruction::set_forward_drizzle_cuda_fault_after_chunks(-1);
  }
  CudaFaultGuard(const CudaFaultGuard &) = delete;
  CudaFaultGuard &operator=(const CudaFaultGuard &) = delete;
};
struct Fixture {
  core::AtomicOutput staging{fs::temp_directory_path()/"runner-forward-test"};
  fs::path dir=staging.path();
  config::Config cfg;
  registration::RegistrationSamplingPlan plan;
  std::unique_ptr<runner::RunnerFrameCache> cache;
  Fixture() {
    fs::create_directories(dir/"artifacts"); fs::create_directories(dir/"logs");
    const std::string yaml=R"(data:
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
    // Plan 19: the FORWARD_DRIZZLE phase records the acceleration backend it
    // actually ran on. Slice 1 has no CUDA kernels, so it is always "cpu"
    // (whether or not "auto"/"cuda" was the resolved intent).
    core::json fd_end;
    for (const auto &event:events(log.str()))
      if (event["type"]=="phase_end" && event["phase_name"]=="FORWARD_DRIZZLE") fd_end=event;
    REQUIRE(fd_end.at("acceleration_backend")=="cpu");
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
    // Cache-lifetime contract (plan 16.2): with the default config the internal
    // profile store is deleted after a committed final image; the source caches
    // are retained (resume-reconstruction stays possible). run_end reports it.
    const auto re=events(log.str()).back();
    REQUIRE(re["cache_retention"]["profile_cache"]=="deleted");
    REQUIRE(re["cache_retention"]["source_cache"]=="retained");
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
  }
  REQUIRE_FALSE(fs::exists(f.dir/"cache/prewarped_frames"));
  REQUIRE_THROWS(f.cache->store_normalized(0,Matrix2Df::Ones(32,32)));
  f.cache.reset();
  reconstruction::VerifiedNormalizedSourceCache kept(f.dir/"cache/normalized_frames",f.plan,32);
  REQUIRE(kept.load(0).minCoeff()==10.0f);
  REQUIRE(phase_to_int(Phase::PREWARP)==2);
  REQUIRE(phase_to_int(Phase::AQMH_BGE_INPUTS)==23);
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
    CudaFaultGuard guard(1);
    REQUIRE(f.execute(log));
  }
  core::json fd_end;
  for (const auto &event : events(log.str()))
    if (event["type"] == "phase_end" && event["phase_name"] == "FORWARD_DRIZZLE")
      fd_end = event;
  REQUIRE(fd_end.at("acceleration_backend") == "cpu");
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
  const auto current=f.dir/"artifacts/forward_drizzle_profiles/current.json";
  const auto prior=core::sha256_file(current);
  std::ostringstream resumed;
  REQUIRE(f.execute(resumed,"FORWARD_DRIZZLE"));
  std::vector<std::string> starts;
  for (const auto &event:events(resumed.str())) if (event["type"]=="phase_start") starts.push_back(event["phase_name"]);
  REQUIRE(starts==(std::vector<std::string>{"FORWARD_DRIZZLE","MULTIBAND"}));
  REQUIRE(core::sha256_file(current)!=prior);
  const auto valid=core::sha256_file(current);
  { std::fstream file(f.dir/"cache/normalized_frames/0.raw",std::ios::in|std::ios::out|std::ios::binary); file.put('X'); }
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
