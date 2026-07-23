#include "services/pi/pi_image_ops.hpp"

#include <iostream>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <vector>

static int failures = 0;

#define EXPECT(cond, msg) \
    do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
         else { std::cout << "ok: " << msg << "\n"; } } while(0)

#define EXPECT_NEAR(a, b, eps, msg) \
    do { if (std::abs((a) - (b)) > (eps)) { std::cerr << "FAIL: " << msg \
         << " (got " << (a) << ", expected " << (b) << ")\n"; ++failures; } \
         else { std::cout << "ok: " << msg << "\n"; } } while(0)

static cv::Mat make_test_image(int w = 64, int h = 64) {
    cv::Mat img(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(x * 4 % 256),
                static_cast<uchar>(y * 4 % 256),
                static_cast<uchar>((x + y) * 2 % 256));
        }
    }
    return img;
}

static double channel_mean(const cv::Mat& img, int channel) {
    cv::Scalar s = cv::mean(img);
    return s[channel];
}

static double channel_stddev(const cv::Mat& img) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    return stddev[0];
}

int main() {
    try {
        auto img = make_test_image();

        // --- brightness ---
        {
            auto bright = tile_compile::pi::apply_brightness(img, 0.5, 0.0, 0.0);
            EXPECT(!bright.empty(), "brightness produces non-empty image");
            EXPECT(bright.size() == img.size(), "brightness preserves size");
            EXPECT(channel_mean(bright, 1) > channel_mean(img, 1),
                   "brightness midtones=0.5 increases mean");

            auto dark = tile_compile::pi::apply_brightness(img, -0.5, 0.0, 0.0);
            EXPECT(channel_mean(dark, 1) < channel_mean(img, 1),
                   "brightness midtones=-0.5 decreases mean");
        }

        // --- contrast ---
        {
            auto high = tile_compile::pi::apply_contrast(img, 0.5);
            EXPECT(!high.empty(), "contrast produces non-empty image");
            // Sigmoid contrast pushes values away from 0.5 center,
            // increasing spread for images with mid-range content.
            // Use a gradient image centered around 128 for a reliable test.
            cv::Mat gradient(64, 64, CV_8UC3);
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x)
                    gradient.at<cv::Vec3b>(y, x) = cv::Vec3b(128, 128, 128);
            // Add a slight ramp so there's some variance
            for (int x = 0; x < 64; ++x)
                for (int y = 0; y < 64; ++y)
                    gradient.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(100 + x);
            auto grad_high = tile_compile::pi::apply_contrast(gradient, 0.5);
            EXPECT(channel_stddev(grad_high) > channel_stddev(gradient),
                   "contrast amount=0.5 increases stddev on gradient");

            auto low = tile_compile::pi::apply_contrast(img, -0.5);
            EXPECT(channel_stddev(low) < channel_stddev(img),
                   "contrast amount=-0.5 decreases stddev");
        }

        // --- saturation ---
        {
            auto sat = tile_compile::pi::apply_saturation(img, 1.0);
            EXPECT(!sat.empty(), "saturation produces non-empty image");
            // Higher saturation -> more color deviation from gray
            cv::Mat gray_orig, gray_sat;
            cv::cvtColor(img, gray_orig, cv::COLOR_BGR2GRAY);
            cv::cvtColor(sat, gray_sat, cv::COLOR_BGR2GRAY);
            // Saturation should not change luminance significantly
            // but should change color channels
            EXPECT(channel_stddev(sat) >= channel_stddev(img) - 1.0,
                   "saturation amount=1.0 does not reduce stddev");
        }

        // --- sharpen ---
        {
            auto sharp = tile_compile::pi::apply_sharpen(img, 0.5, 2.0);
            EXPECT(!sharp.empty(), "sharpen produces non-empty image");
            EXPECT(sharp.size() == img.size(), "sharpen preserves size");
            // Sharpening should change the image
            cv::Mat diff;
            cv::absdiff(img, sharp, diff);
            EXPECT(cv::sum(diff)[0] > 0, "sharpen changes image");
        }

        // --- denoise ---
        {
            // Add noise to test denoise
            cv::Mat noise(img.size(), img.type());
            cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(30));
            cv::Mat noisy;
            cv::add(img, noise, noisy);
            auto denoised = tile_compile::pi::apply_denoise(noisy, 0.5, false);
            EXPECT(!denoised.empty(), "denoise produces non-empty image");
            // Denoised should have lower variance than noisy
            EXPECT(channel_stddev(denoised) <= channel_stddev(noisy) + 1.0,
                   "denoise reduces or maintains variance");

            auto denoised_lum = tile_compile::pi::apply_denoise(noisy, 0.5, true);
            EXPECT(!denoised_lum.empty(), "denoise luminance=true produces non-empty image");
        }

        // --- rmgreen ---
        {
            // Create an image with green dominance
            cv::Mat green_img(8, 8, CV_8UC3, cv::Vec3b(50, 200, 50));
            auto result = tile_compile::pi::apply_rmgreen(green_img, 1.0);
            EXPECT(!result.empty(), "rmgreen produces non-empty image");
            // G channel should be reduced
            EXPECT(channel_mean(result, 1) < channel_mean(green_img, 1),
                   "rmgreen strength=1.0 reduces green channel");
            // G should be reduced by min(B,R)*strength = 50*1.0 = 50
            cv::Scalar s = cv::mean(result);
            EXPECT_NEAR(s[1], 200.0 - 50.0, 2.0,
                        "rmgreen: G reduced by min(B,R)*strength");
        }

        // --- clahe ---
        {
            auto clahe_out = tile_compile::pi::apply_clahe(img, 3.0, 8);
            EXPECT(!clahe_out.empty(), "clahe produces non-empty image");
            EXPECT(clahe_out.size() == img.size(), "clahe preserves size");
            // CLAHE should change the image
            cv::Mat diff;
            cv::absdiff(img, clahe_out, diff);
            EXPECT(cv::sum(diff)[0] > 0, "clahe changes image");
        }

        // --- bilateral ---
        {
            auto bilat = tile_compile::pi::apply_bilateral(img, 9, 75.0, 75.0);
            EXPECT(!bilat.empty(), "bilateral produces non-empty image");
            EXPECT(bilat.size() == img.size(), "bilateral preserves size");
        }

        // --- threshold ---
        {
            auto thresh = tile_compile::pi::apply_threshold(img, 0.1, 0.9);
            EXPECT(!thresh.empty(), "threshold produces non-empty image");
            // Pixels below black_point should be 0
            // Check that some pixels are now 0
            cv::Mat gray;
            cv::cvtColor(thresh, gray, cv::COLOR_BGR2GRAY);
            int zeros = cv::countNonZero(gray == 0);
            EXPECT(zeros > 0, "threshold black_point=0.1 creates black pixels");
            // Pixels above white_point should be 255
            int whites = cv::countNonZero(gray == 255);
            EXPECT(whites > 0, "threshold white_point=0.9 creates white pixels");
        }

        // --- invert ---
        {
            auto inv = tile_compile::pi::apply_invert(img);
            EXPECT(!inv.empty(), "invert produces non-empty image");
            // Check that inversion is 255 - original
            EXPECT(inv.at<cv::Vec3b>(0, 0)[0] == static_cast<uchar>(255 - img.at<cv::Vec3b>(0, 0)[0]),
                   "invert: pixel = 255 - original");
        }

        // --- crop ---
        {
            auto cropped = tile_compile::pi::apply_crop(img, 10, 10, 20, 20);
            EXPECT(!cropped.empty(), "crop produces non-empty image");
            EXPECT(cropped.cols == 20 && cropped.rows == 20,
                   "crop produces correct size");
        }

        // --- invert_op ---
        {
            nlohmann::json brightness_op = {
                {"type", "brightness"},
                {"params", {{"midtones", 0.15}, {"shadows", 0.0}, {"highlights", 0.0}}}
            };
            auto inv = tile_compile::pi::invert_op(brightness_op);
            EXPECT(inv["type"].get<std::string>() == "brightness",
                   "invert_op(brightness) returns brightness");
            EXPECT_NEAR(inv["params"]["midtones"].get<double>(), -0.15, 1e-9,
                        "invert_op(brightness) negates midtones");

            nlohmann::json contrast_op = {
                {"type", "contrast"},
                {"params", {{"amount", 0.1}}}
            };
            auto inv_c = tile_compile::pi::invert_op(contrast_op);
            EXPECT_NEAR(inv_c["params"]["amount"].get<double>(), -0.1, 1e-9,
                        "invert_op(contrast) negates amount");

            nlohmann::json clahe_op = {
                {"type", "clahe"},
                {"params", {{"cliplimit", 3.0}, {"tilesize", 8}}}
            };
            auto inv_cl = tile_compile::pi::invert_op(clahe_op);
            EXPECT(inv_cl["type"].get<std::string>() == "noop",
                   "invert_op(clahe) returns noop");

            nlohmann::json denoise_op = {
                {"type", "denoise"},
                {"params", {{"strength", 0.5}, {"luminance", true}}}
            };
            auto inv_d = tile_compile::pi::invert_op(denoise_op);
            EXPECT(inv_d["type"].get<std::string>() == "noop",
                   "invert_op(denoise) returns noop");
        }

        // --- validate_op ---
        {
            nlohmann::json bad = {{"type", "brightness"}, {"params", {{"midtones", 5.0}, {"shadows", 0.0}, {"highlights", 0.0}}}};
            auto v = tile_compile::pi::validate_op(bad);
            EXPECT(!v.empty() && v.contains("error"),
                   "validate_op rejects out-of-range midtones");

            nlohmann::json unknown = {{"type", "foobar"}, {"params", {}}};
            auto vu = tile_compile::pi::validate_op(unknown);
            EXPECT(!vu.empty() && vu.contains("error"),
                   "validate_op rejects unknown type");

            nlohmann::json good = {{"type", "contrast"}, {"params", {{"amount", 0.1}}}};
            auto vg = tile_compile::pi::validate_op(good);
            EXPECT(vg.empty(), "validate_op accepts valid contrast");
        }

        // --- apply_image_op dispatch ---
        {
            nlohmann::json op = {{"type", "contrast"}, {"params", {{"amount", 0.2}}}};
            auto res = tile_compile::pi::apply_image_op(img, op);
            EXPECT(res.success, "apply_image_op(contrast) succeeds");
            EXPECT(!res.image.empty(), "apply_image_op(contrast) produces image");

            nlohmann::json bad_op = {{"type", "unknown"}, {"params", {}}};
            auto bad_res = tile_compile::pi::apply_image_op(img, bad_op);
            EXPECT(!bad_res.success, "apply_image_op(unknown) fails");
            EXPECT(!bad_res.error.empty(), "apply_image_op(unknown) has error message");

            nlohmann::json reset_op = {{"type", "reset"}, {"params", {}}};
            auto reset_res = tile_compile::pi::apply_image_op(img, reset_op);
            EXPECT(reset_res.success, "apply_image_op(reset) succeeds");
            // Reset should return a clone of the input
            cv::Mat diff;
            cv::absdiff(img, reset_res.image, diff);
            EXPECT(cv::sum(diff)[0] == 0, "apply_image_op(reset) returns original");
        }

        // --- Phase 2 operations ---
        {
            const std::vector<nlohmann::json> phase2 = {
                {{"type", "vibrance"}, {"params", {{"amount", 0.2}}}},
                {{"type", "color_temperature"}, {"params", {{"amount", 0.2}}}},
                {{"type", "unpurple"}, {"params", {{"amount", 0.5}}}},
                {{"type", "fixbanding"}, {"params", {{"amount", 0.5}, {"sigma", 2.0}}}},
                {{"type", "star_desaturation"}, {"params", {{"amount", 0.5}}}},
                {{"type", "dehaze"}, {"params", {{"amount", 0.4}}}}
            };
            for (const auto& op : phase2) {
                auto validation = tile_compile::pi::validate_op(op);
                EXPECT(validation.empty(), "phase 2 operation validates");
                auto result = tile_compile::pi::apply_image_op(img, op);
                EXPECT(result.success && result.image.size() == img.size(),
                       "phase 2 operation applies to display image");
            }
            cv::Mat linear;
            img.convertTo(linear, CV_32FC3, 1.0 / 255.0);
            auto linear_result = tile_compile::pi::apply_image_op_fits(linear, phase2.front());
            EXPECT(linear_result.success && linear_result.image.type() == CV_32FC3,
                   "phase 2 operation applies to linear FITS image");
        }

    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        ++failures;
    }

    if (failures > 0) {
        std::cerr << "\n" << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "\nAll tests passed\n";
    return 0;
}
