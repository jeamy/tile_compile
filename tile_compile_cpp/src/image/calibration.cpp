#include "tile_compile/core/types.hpp"
#include "tile_compile/core/errors.hpp"
#include "tile_compile/io/fits_io.hpp"

namespace tile_compile::image {

Matrix2Df create_master_bias(const std::vector<Matrix2Df>& bias_frames) {
    if (bias_frames.empty()) {
        throw TileCompileError("No bias frames provided");
    }
    
    int h = bias_frames[0].rows();
    int w = bias_frames[0].cols();
    Matrix2Df master = Matrix2Df::Zero(h, w);
    
    for (const auto& frame : bias_frames) {
        master += frame;
    }
    master /= static_cast<float>(bias_frames.size());
    
    return master;
}

Matrix2Df create_master_dark(const std::vector<Matrix2Df>& dark_frames, 
                              const Matrix2Df* bias) {
    if (dark_frames.empty()) {
        throw TileCompileError("No dark frames provided");
    }
    
    int h = dark_frames[0].rows();
    int w = dark_frames[0].cols();
    Matrix2Df master = Matrix2Df::Zero(h, w);
    
    for (const auto& frame : dark_frames) {
        if (bias) {
            master += (frame - *bias);
        } else {
            master += frame;
        }
    }
    master /= static_cast<float>(dark_frames.size());
    
    return master;
}

Matrix2Df create_master_flat(const std::vector<Matrix2Df>& flat_frames,
                              const Matrix2Df* bias,
                              const Matrix2Df* dark) {
    if (flat_frames.empty()) {
        throw TileCompileError("No flat frames provided");
    }
    
    int h = flat_frames[0].rows();
    int w = flat_frames[0].cols();
    Matrix2Df master = Matrix2Df::Zero(h, w);
    
    for (const auto& frame : flat_frames) {
        Matrix2Df corrected = frame;
        if (bias) corrected -= *bias;
        if (dark) corrected -= *dark;
        master += corrected;
    }
    master /= static_cast<float>(flat_frames.size());
    
    float median = master.mean();
    if (median > 0) {
        master /= median;
    }
    
    return master;
}

Matrix2Df calibrate_frame(const Matrix2Df& frame,
                          const Matrix2Df* bias,
                          const Matrix2Df* dark,
                          const Matrix2Df* flat,
                          float denom_eps) {
    Matrix2Df result = frame;
    
    if (bias) {
        result -= *bias;
    }
    
    if (dark) {
        result -= *dark;
    }
    
    if (flat) {
        result = result.array() / (flat->array().max(denom_eps));
    }
    
    return result;
}

} // namespace tile_compile::image
