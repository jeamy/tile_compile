#include "tile_compile/core/types.hpp"

namespace tile_compile::pipeline {

std::string phase_name(Phase phase) {
    return phase_to_string(phase);
}

} // namespace tile_compile::pipeline
