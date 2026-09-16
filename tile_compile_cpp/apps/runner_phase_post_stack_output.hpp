#pragma once

#include "runner_shared.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/events.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/io/fits_io.hpp"

#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace tile_compile::runner {

/// Write a stretched RGB snapshot (used for BGE/PCC intermediate outputs).
/// Applies canvas mask, then robust p99.9 stretch if requested, writes as
/// uint32 or float32 depending on stretch flag.
void write_stretched_rgb_snapshot(
    const std::filesystem::path &path,
    const Matrix2Df &R_src,
    const Matrix2Df &G_src,
    const Matrix2Df &B_src,
    const std::vector<uint8_t> &canvas_mask,
    const std::vector<uint8_t> &statistics_mask,
    int canvas_rows,
    int canvas_cols,
    const io::FitsHeader &hdr,
    bool apply_stretch,
    const char *stage_tag);

} // namespace tile_compile::runner
