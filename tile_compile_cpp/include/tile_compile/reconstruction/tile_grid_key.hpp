#pragma once

#include <cstdint>

namespace tile_compile::reconstruction {

inline uint64_t tile_grid_key(int row, int col) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) ^
         static_cast<uint32_t>(col);
}

}  // namespace tile_compile::reconstruction

