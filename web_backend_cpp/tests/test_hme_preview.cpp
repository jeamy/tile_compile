// HMS preview: a run that kept its pre-chroma-denoise state
// (outputs/pcc_predenoise_*.fit, hypermetric_stretch.params_from_pre_denoise)
// is previewed with the stretch parameters of that state, like the runner.
#include "services/hme_preview_service.hpp"

#include <fitsio.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int failures = 0;

#define EXPECT(cond, msg) \
    do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
         else { std::cout << "ok: " << msg << "\n"; } } while(0)

namespace {

constexpr int kSize = 96;

// Sky with noise shared by all channels plus optional chroma noise (R up,
// B down) and a few stars; same seed for every call.
std::vector<std::vector<float>> make_rgb(float chroma_sigma) {
    std::mt19937 rng(5);
    std::normal_distribution<float> common(0.0f, 4.0f), chroma(0.0f, 1.0f);
    std::vector<std::vector<float>> rgb(3, std::vector<float>(kSize * kSize));
    for (int i = 0; i < kSize * kSize; ++i) {
        const float sky = 300.0f + common(rng);
        const float c = chroma_sigma * chroma(rng);
        rgb[0][i] = sky + c;
        rgb[1][i] = sky;
        rgb[2][i] = sky - c;
    }
    for (int s = 0; s < 12; ++s) {
        const int i = (8 + (s * 23) % 80) * kSize + 8 + (s * 37) % 80;
        rgb[0][i] = rgb[1][i] = rgb[2][i] = 40000.0f;
    }
    return rgb;
}

void write_fits(const fs::path& path, const std::vector<float>& data, int planes) {
    fs::create_directories(path.parent_path());
    fs::remove(path);
    fitsfile* f = nullptr;
    int status = 0;
    long naxes[3] = {kSize, kSize, planes};
    if (fits_create_file(&f, path.string().c_str(), &status) ||
        fits_create_img(f, FLOAT_IMG, planes > 1 ? 3 : 2, naxes, &status) ||
        fits_write_img(f, TFLOAT, 1, static_cast<LONGLONG>(data.size()),
                       const_cast<float*>(data.data()), &status) ||
        fits_close_file(f, &status)) {
        throw std::runtime_error("FITS write failed: " + path.string());
    }
}

void write_run(const fs::path& run, const std::vector<std::vector<float>>& pcc,
               const std::vector<std::vector<float>>* predenoise) {
    std::vector<float> cube;
    for (const auto& plane : pcc) cube.insert(cube.end(), plane.begin(), plane.end());
    write_fits(run / "outputs" / "stacked_rgb_pcc.fits", cube, 3);
    if (predenoise != nullptr) {
        const char* names[3] = {"R", "G", "B"};
        for (int c = 0; c < 3; ++c)
            write_fits(run / "outputs" / (std::string("pcc_predenoise_") + names[c] + ".fit"),
                       (*predenoise)[c], 1);
    }
}

} // namespace

int main() {
    try {
        const fs::path root = fs::temp_directory_path() /
                              ("tc_hme_preview_" + std::to_string(::getpid()));
        fs::remove_all(root);
        const auto noisy = make_rgb(12.0f);  // before the chroma pass
        const auto clean = make_rgb(0.0f);   // after a chroma-only denoise

        write_run(root / "plain", noisy, nullptr);
        write_run(root / "denoised", clean, &noisy);
        write_run(root / "denoised_no_reference", clean, nullptr);

        const nlohmann::json params = {{"mode", "ready_to_use"}};
        const auto plain = tile_compile::web::create_hme_preview(root / "plain", params);
        const auto with_ref = tile_compile::web::create_hme_preview(root / "denoised", params);
        const auto opted_out = tile_compile::web::create_hme_preview(
            root / "denoised", {{"mode", "ready_to_use"}, {"params_from_pre_denoise", false}});
        const auto no_ref = tile_compile::web::create_hme_preview(
            root / "denoised_no_reference", params);

        EXPECT(plain.ok && with_ref.ok && opted_out.ok && no_ref.ok, "all previews render");
        EXPECT(plain.diagnostics.value("params_source", "") == "image",
               "a run without pre-denoise state uses the image's parameters");
        EXPECT(with_ref.diagnostics.value("params_source", "") == "pre_denoise_reference",
               "a run with pre-denoise state uses its parameters");
        EXPECT(opted_out.diagnostics.value("params_source", "") == "image",
               "params_from_pre_denoise=false opts out");
        EXPECT(no_ref.diagnostics.value("params_source", "") == "image",
               "missing files fall back to the image");
        EXPECT(with_ref.diagnostics.value("anchor", -1.0) == plain.diagnostics.value("anchor", -2.0) &&
                   with_ref.diagnostics.value("log_d", -1.0) == plain.diagnostics.value("log_d", -2.0),
               "the denoised preview gets the anchor and log D of the pre-denoise state");
        auto scaling = [](const tile_compile::web::HmePreviewResult& r) {
            return r.diagnostics.value("output_scaling", nlohmann::json::object());
        };
        EXPECT(scaling(with_ref) == scaling(plain),
               "the denoised preview gets the output floors, scale and midtone of the pre-denoise state");
        EXPECT(scaling(no_ref) != scaling(plain),
               "without the reference the denoised preview derives other output scaling");
        fs::remove_all(root);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception: " << e.what() << "\n";
        return 1;
    }
    if (failures > 0) {
        std::cerr << "\n" << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "\nAll tests passed\n";
    return 0;
}
