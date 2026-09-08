#include "backend_test_harness.hpp"

#include <cstdlib>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

// Plan M8 (§23.1): the run report must surface the single-method CFA-forward-
// drizzle / multiband contract from forward_drizzle.json (+ the coverage gate
// from sampling_geometry.json) so the displayed numbers can be checked against
// the real run artifacts. This fixture mirrors the shape of a real M42 run.
//
// The test renders the SAME fixture twice:
//   * locale "en" - the English base strings must survive report_en.json's
//     greedy longest-key-first replacement pass verbatim (a partial match
//     against one of its keys would corrupt the new section mid-string).
//   * locale "de" - the German card titles from report_de.json must actually
//     appear, and the English forms must be gone. This is the check that
//     discriminates a broken i18n key match: an escaped character in a base
//     title (e.g. "&" -> "&amp;") silently reverts the card to English while
//     the "en" pass above stays green.
namespace {

nlohmann::json make_fd_fixture() {
    return nlohmann::json{
        {"schema_version", 2},
        {"pipeline_method", "cfa_forward_drizzle_multiband"},
        {"pipeline_contract_version", 1},
        {"coverage_geometry_hash", "3b2e5442dd25b220e4db6458aac3b76bd60c478bd55f314fa31a7ba1a6b6cfe8"},
        {"luma_definition", "0.25R+0.50G+0.25B"},
        {"pixels_supported", 24879983},
        {"geometry", {
            {"source_width", 3840}, {"source_height", 2160},
            {"reconstruction_width", 3858}, {"reconstruction_height", 2190},
            {"internal_scale", 1}, {"output_scale", 1}, {"output_scale_applied", false},
            {"kernel", "square"}, {"pixfrac", 0.8}
        }},
        {"clipping", {{"pixel_channel_evaluations", 25072320}, {"pixel_channel_rejected", 173667},
                      {"candidate_contributions_clipped", 93750053}}},
        {"throughput", {
            {"frames_used", 40}, {"source_width", 3840}, {"source_height", 2160},
            {"processed_source_samples", 331776000},
            {"forward_drizzle_wall_seconds", 1831.95},
            {"source_samples_per_second", 181104.0}
        }},
        {"runtime_environment", {
            {"build", {{"toolchain", {{"build_type", "Release"}}},
                       {"source", {{"git_describe", "1ab46847"}, {"git_dirty", true}}}}},
            {"hardware", {{"cpu_model", "AMD Ryzen 7 3700X 8-Core Processor"},
                          {"logical_cores", 16}, {"gpu", "NVIDIA GeForce GTX 1660 Ti"}}},
            {"threads", {{"parallel_workers_config", 4}, {"workers_used", 1}}}
        }},
        {"flux_space", {
            {"space", "normalised_linear_working"},
            {"luma_definition", "0.25R+0.50G+0.25B"},
            {"same_space_as", "reconstruction_multiband.fits"},
            {"stacking_normalisation_undo_applied", false},
            {"note", "17.4 scale_r/g/b + pedestal undo pending M10 cutover"}
        }},
        {"alpha_confidence_summary", {
            {{"band", 0}, {"support_px", 24879983}, {"alpha_below_one_px", 120000},
             {"alpha_below_one_fraction", 0.0048}, {"mean_alpha_on_support", 0.997}, {"min_alpha_on_support", 0.41}},
            {{"band", 1}, {"support_px", 24879983}, {"alpha_below_one_px", 0},
             {"alpha_below_one_fraction", 0.0}, {"mean_alpha_on_support", 1.0}, {"min_alpha_on_support", 1.0}}
        }},
        {"resources", {
            {"rss_process_peak_kib", 3963900},
            {"multiband_phase_rss_within_envelope", true},
            {"multiband_temp_space_ok", true},
            {"multiband_working_set_fits_budget", true}
        }},
        {"acceleration", {
            {"forward_drizzle_backend", "cuda_hybrid"},
            {"cuda_fallback_reason", nullptr},
            {"cuda_stripe_path", {{"bands", 24}, {"resolved_chunk_rows", 95},
                                  {"hybrid_local_frames", 1}, {"hybrid_cpu_seconds", 444.17},
                                  {"hybrid_gpu_raster_seconds", 0.6}}}
        }},
        {"timing_seconds", {{"FORWARD_DRIZZLE", 1831.95}}},
        {"validation", {
            {"stars_total", 102}, {"stars_multiband_effective", 1},
            {"drizzle_uniform", {
                {"background_rms", {{"value", 1.0296}, {"applicable", true}, {"sample_count", 0}}},
                {"seam_score", {{"value", 1.2365}, {"applicable", true}, {"sample_count", 4294}}},
                {"median_fwhm", {{"value", nullptr}, {"applicable", false}, {"sample_count", 0},
                                 {"reason_if_not_applicable", "fewer than 20 effective stars"}}},
                {"p90_fwhm", {{"value", nullptr}, {"applicable", false}, {"sample_count", 0}}},
                {"tail", {{"value", nullptr}, {"applicable", false}, {"sample_count", 0}}},
                {"elongation", {{"value", nullptr}, {"applicable", false}, {"sample_count", 0}}},
                {"support_ok", true}, {"numerics_ok", true}
            }},
            {"drizzle_raw", {
                {"background_rms", {{"value", 1.1221}, {"applicable", true}, {"sample_count", 0}}},
                {"seam_score", {{"value", 1.2330}, {"applicable", true}, {"sample_count", 4294}}},
                {"median_fwhm", {{"value", nullptr}, {"applicable", false}, {"sample_count", 0}}},
                {"p90_fwhm", {{"value", nullptr}, {"applicable", false}, {"sample_count", 0}}},
                {"tail", {{"value", nullptr}, {"applicable", false}, {"sample_count", 0}}},
                {"elongation", {{"value", nullptr}, {"applicable", false}, {"sample_count", 0}}},
                {"support_ok", true}, {"numerics_ok", true}
            }},
            {"drizzle_multiband", {
                {"background_rms", {{"value", 1.1218}, {"applicable", true}, {"sample_count", 0}}},
                {"seam_score", {{"value", 1.1891}, {"applicable", true}, {"sample_count", 4294}}},
                {"median_fwhm", {{"value", nullptr}, {"applicable", false}, {"sample_count", 1}}},
                {"p90_fwhm", {{"value", nullptr}, {"applicable", false}, {"sample_count", 1}}},
                {"tail", {{"value", nullptr}, {"applicable", false}, {"sample_count", 0}}},
                {"elongation", {{"value", nullptr}, {"applicable", false}, {"sample_count", 0}}},
                {"support_ok", true}, {"numerics_ok", true}
            }}
        }},
        {"selected_candidate", "drizzle_uniform"},
        {"selection_reason", "raw rejected -> uniform: raw background_rms regression vs uniform"},
        {"fallback_reason", "raw rejected -> uniform: raw background_rms regression vs uniform"}
    };
}

nlohmann::json make_sg_fixture() {
    return nlohmann::json{
        {"coverage_geometry_hash", "3b2e5442dd25b220e4db6458aac3b76bd60c478bd55f314fa31a7ba1a6b6cfe8"},
        {"coverage_gate", {
            {"passed", true},
            {"min_supported_fraction", 1.0},
            {"min_channel_n_eff_p10", 15.0322},
            {"largest_internal_hole_area_px", 0},
            {"analysis_pixels", 8240098},
            {"valid_frames", 40},
            {"supported_fraction", {{"R", 1.0}, {"G", 1.0}, {"B", 1.0}}},
            {"geometric_uniform_neff_p10", {{"R", 15.0322}, {"G", 30.3721}, {"B", 15.0327}}},
            {"violations", nlohmann::json::array()}
        }}
    };
}

// Boots the backend fresh (start() picks up the current TILE_COMPILE_REPORT_LOCALE
// env), renders the fixture, and returns report.html.
std::string render_report(BackendHarness& harness) {
    harness.start();

    const std::vector<nlohmann::json> events = {
        {{"ts", "2026-09-08T08:20:00Z"}, {"type", "run_start"}, {"run_id", "fd_run"}, {"frames_discovered", 40}},
        {{"ts", "2026-09-08T08:20:01Z"}, {"type", "phase_start"}, {"phase_name", "FORWARD_DRIZZLE"}},
        {{"ts", "2026-09-08T08:35:01Z"}, {"type", "phase_end"}, {"phase_name", "FORWARD_DRIZZLE"}, {"status", "ok"}},
        {{"ts", "2026-09-08T08:40:00Z"}, {"type", "run_end"}, {"status", "ok"}}
    };
    const fs::path run_dir = harness.create_run("fd_run", events);
    harness.make_file("runs/fd_run/artifacts/forward_drizzle.json", make_fd_fixture().dump(2));
    harness.make_file("runs/fd_run/artifacts/sampling_geometry.json", make_sg_fixture().dump(2));

    const auto stats_job = harness.post_json("/api/runs/fd_run/stats", {{"run_dir", run_dir.string()}});
    expect_equal(stats_job["_http_status"].get<long>(), 202L, "stats job accepted");
    const auto done = harness.wait_for_job(stats_job["job_id"].get<std::string>(), 20.0);
    expect_equal(done["state"].get<std::string>(), "ok", "stats job completed");

    return slurp_file(run_dir / "artifacts" / "report.html");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    setenv("TILE_COMPILE_BACKEND_REPORT_EVENTS_MAX", "512", 1);
    setenv("TILE_COMPILE_BACKEND_REPORT_JSON_FILE_BYTES", "4194304", 1);
    setenv("TILE_COMPILE_BACKEND_RETAINED_JOBS", "16", 1);

    BackendHarness harness(argv[1], argv[2], argv[3], argv[4]);
    try {
        // ---- Pass 1: English -------------------------------------------------
        setenv("TILE_COMPILE_REPORT_LOCALE", "en", 1);
        const std::string en = render_report(harness);

        // English base strings survive the en translation pass verbatim.
        expect_true(en.find("CFA Forward Drizzle / Multiband") != std::string::npos,
                    "forward-drizzle section title survives en pass");
        expect_true(en.find("Coverage and geometry") != std::string::npos,
                    "coverage and geometry card title survives en pass");
        expect_true(en.find("Candidate selection and gates") != std::string::npos,
                    "candidate selection card title survives en pass");
        expect_true(en.find("Flux space and pixel scale") != std::string::npos,
                    "flux space card title survives en pass");
        expect_true(en.find("Resources and throughput") != std::string::npos,
                    "resources and throughput card title survives en pass");
        expect_true(en.find("Alpha confidence") != std::string::npos,
                    "alpha confidence card title survives en pass");
        expect_true(en.find("cfa_forward_drizzle_multiband") != std::string::npos,
                    "pipeline method value rendered");
        expect_true(en.find("drizzle_uniform") != std::string::npos,
                    "selected candidate rendered");
        expect_true(en.find("raw rejected -&gt; uniform") != std::string::npos,
                    "selection reason rendered");
        expect_true(en.find("331776000") != std::string::npos,
                    "processed source samples rendered");
        expect_true(en.find("AMD Ryzen 7 3700X 8-Core Processor") != std::string::npos,
                    "cpu model rendered");
        expect_true(en.find("15.0322") != std::string::npos || en.find("15.032") != std::string::npos,
                    "coverage-gate n_eff p10 rendered");
        expect_true(en.find("normalised_linear_working") != std::string::npos,
                    "flux space value rendered");

        // ---- Pass 2: German ------------------------------------------------
        // The i18n key match must actually fire: German card titles present,
        // English forms gone. This is what an "&" -> "&amp;" style base-title
        // escape would break while pass 1 stayed green.
        setenv("TILE_COMPILE_REPORT_LOCALE", "de", 1);
        const std::string de = render_report(harness);

        expect_true(de.find("CFA-Forward-Drizzle / Multiband") != std::string::npos,
                    "section title translated to German");
        expect_true(de.find("Abdeckung und Geometrie") != std::string::npos,
                    "coverage card title translated to German");
        expect_true(de.find("Kandidatenauswahl und Gates") != std::string::npos,
                    "candidate selection card title translated to German");
        expect_true(de.find("Fluxraum und Pixelmaßstab") != std::string::npos,
                    "flux space card title translated to German");
        expect_true(de.find("Rauschdiagnostik") != std::string::npos,
                    "noise diagnostics card title translated to German");
        expect_true(de.find("Ressourcen und Durchsatz") != std::string::npos,
                    "resources card title translated to German");
        expect_true(de.find("Alpha-Konfidenz") != std::string::npos,
                    "alpha confidence card title translated to German");
        // Also cover the render_kv_table row-label path (the bulk of the keys),
        // not just card/section titles.
        expect_true(de.find("Verwendete Frames") != std::string::npos,
                    "kv-table row label 'Frames used' translated to German");
        expect_true(de.find("Auswahlgrund") != std::string::npos,
                    "kv-table row label 'Selection reason' translated to German");

        expect_true(de.find(">Coverage and geometry<") == std::string::npos,
                    "English coverage card title gone from German report");
        expect_true(de.find(">Candidate selection and gates<") == std::string::npos,
                    "English candidate card title gone from German report");
        expect_true(de.find(">Resources and throughput<") == std::string::npos,
                    "English resources card title gone from German report");

        harness.stop();
        return 0;
    } catch (const std::exception& e) {
        harness.stop();
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
