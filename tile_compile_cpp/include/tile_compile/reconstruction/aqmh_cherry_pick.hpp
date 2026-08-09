#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/reconstruction/aqmh_sigma_clip.hpp"

#include <vector>

namespace tile_compile::reconstruction {

float aqmh_effective_k_frac(
    int n_rankable, float base,
    const std::vector<config::AqmhCherryPickConfig::Tier> &tiers);
int aqmh_k_nominal(int n_rankable, float fraction);
std::vector<AqmhWeightedSample> aqmh_select_top_k(
    std::vector<AqmhWeightedSample> samples, int k_min_required,
    float fraction,
    const std::vector<config::AqmhCherryPickConfig::Tier> &tiers,
    int *nominal_k, float *rank_margin);
std::vector<AqmhWeightedSample> aqmh_select_auto_reject(
    std::vector<AqmhWeightedSample> samples, int k_min_required,
    float reject_below_best_fraction, float min_keep_fraction,
    float margin_min, int *nominal_k, float *rank_margin);

} // namespace tile_compile::reconstruction
