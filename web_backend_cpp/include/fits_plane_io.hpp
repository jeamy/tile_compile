#pragma once
// Float planes of a FITS image for the preview services (BGE, HMS, PI live
// edit). Each call opens the file, reads and closes it; errors throw
// std::runtime_error naming the file.
#include <fitsio.h>

#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

inline std::string fits_status_message(int status) {
    char text[FLEN_STATUS]{};
    fits_get_errstatus(status, text);
    return text;
}

namespace fits_plane_detail {

struct OpenImage {
    fitsfile* file = nullptr;
    long width = 0, height = 0, planes = 0;
    explicit OpenImage(const std::filesystem::path& path) {
        int status = 0;
        if (fits_open_file(&file, path.string().c_str(), READONLY, &status))
            throw std::runtime_error("Cannot open FITS " + path.string() + ": " + fits_status_message(status));
        int naxis = 0;
        long axes[3]{1, 1, 1};
        if (fits_get_img_dim(file, &naxis, &status) || fits_get_img_size(file, 3, axes, &status) || naxis < 2) {
            close();
            throw std::runtime_error("Invalid FITS image " + path.string());
        }
        width = axes[0];
        height = axes[1];
        planes = naxis >= 3 ? axes[2] : 1;
    }
    ~OpenImage() { close(); }
    OpenImage(const OpenImage&) = delete;
    OpenImage& operator=(const OpenImage&) = delete;
    void close() {
        int status = 0;
        if (file) fits_close_file(file, &status);
        file = nullptr;
    }
    std::vector<float> read(const std::filesystem::path& path, long plane) const {
        std::vector<float> pixels(static_cast<size_t>(width * height));
        long first[3]{1, 1, plane};
        int any_null = 0, status = 0;
        if (fits_read_pix(file, TFLOAT, first, width * height, nullptr, pixels.data(), &any_null, &status))
            throw std::runtime_error("Cannot read FITS " + path.string() + ": " + fits_status_message(status));
        return pixels;
    }
};

}  // namespace fits_plane_detail

/// Plane `plane` (1-based) of the image, row-major; width, height and the
/// number of planes are returned through the out parameters.
inline std::vector<float> read_fits_plane(const std::filesystem::path& path, long plane,
                                          long& width, long& height, long& planes) {
    fits_plane_detail::OpenImage image(path);
    width = image.width;
    height = image.height;
    planes = image.planes;
    if (plane < 1 || plane > planes)
        throw std::runtime_error("Missing FITS plane " + std::to_string(plane) + " in " + path.string());
    return image.read(path, plane);
}

/// The first three planes of an RGB cube.
inline std::array<std::vector<float>, 3> read_fits_rgb_cube(const std::filesystem::path& path,
                                                           long& width, long& height) {
    fits_plane_detail::OpenImage image(path);
    width = image.width;
    height = image.height;
    if (image.planes < 3) throw std::runtime_error(path.string() + " has fewer than three planes");
    return {image.read(path, 1), image.read(path, 2), image.read(path, 3)};
}
