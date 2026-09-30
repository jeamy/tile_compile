#include "backend_test_harness.hpp"
#include "services/pi/pi_scan_manifest.hpp"

#include <cstdio>
#include <fstream>
#include <unistd.h>

using nlohmann::json;
namespace fs = std::filesystem;

int main() {
    const fs::path root = fs::temp_directory_path() / ("pi_scan_manifest_test_" + std::to_string(::getpid()));
    try {
        fs::remove_all(root);
        fs::create_directories(root);
        std::ofstream(root / "a.fits") << "frame-a";
        std::ofstream(root / "b.fit.fz") << "frame-b";
        const json metrics = {{"frames", json::array({{{"file_name", "b.fit.fz"}}, {{"file_name", "a.fits"}}})}};
        const json first = tile_compile::pi::build_scan_dataset_manifest(root, metrics);
        expect_equal(static_cast<long>(first.size()), 2L, "two measured files");
        expect_true(first[0]["id"] == "a.fits" && first[0]["size"] == 7, "sorted metadata manifest");
        expect_true(tile_compile::pi::build_scan_dataset_manifest(root, metrics) == first, "unchanged input is stable");
        std::ofstream(root / "a.fits", std::ios::app) << "changed";
        expect_true(tile_compile::pi::build_scan_dataset_manifest(root, metrics) != first, "changed file metadata changes identity");
        std::ofstream(root / "new.fts") << "new";
        bool changed_set = false;
        try { (void)tile_compile::pi::build_scan_dataset_manifest(root, metrics); }
        catch (const std::exception&) { changed_set = true; }
        expect_true(changed_set, "added FITS file invalidates scan metrics");
        fs::remove(root / "new.fts");
        const json traversal = {{"frames", json::array({{{"file_name", "../a.fits"}}})}};
        bool invalid_name = false;
        try { (void)tile_compile::pi::build_scan_dataset_manifest(root, traversal); }
        catch (const std::exception&) { invalid_name = true; }
        expect_true(invalid_name, "path-like frame names refused");
        fs::remove_all(root);
        std::puts("pi_scan_manifest: all checks passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        fs::remove_all(root);
        return 1;
    }
    return 0;
}
