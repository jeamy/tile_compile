#pragma once

// Controlled multi-band reconstruction --- milestone M6 of the CFA-forward-
// drizzle plan (docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungs
// plan_de.md, sections 14.2-14.3).
//
// This is the whole-image IN-MEMORY REFERENCE path (plan 14.7 allows it for
// tests). It combines frequency bands of four profiles that share the same
// accepted CFA samples but different quality weights:
//   U = Uniform control        -> the coarse residual C_U,L
//   R = Raw forward drizzle     -> the blend base of EVERY band difference
//   F = Fine profile (scale-0 Q) -> the finest detail band D1
//   M = Medium profile (scale-1 Q) -> D2
//
// Band assignment for `levels = L` (plan 14.3):
//   L=1: D1<-F,                              coarse C1<-U
//   L=2: D1<-F, D2<-M,                       coarse C2<-U
//   L>=3: D1<-F, D2<-M, D3..DL<-R,           coarse CL<-U
// Blend (plan 14.3):
//   D_out,j = D_R,j + alpha_j * (D_profile,j - D_R,j)
//   X_out   = C_U,L + sum_j D_out,j
// For a band whose profile source is R, alpha_j is ineffective. An invalid
// detail profile forces alpha_j = 0 at that pixel; a missing Raw band makes
// the whole multi-band pixel invalid (plan 14.2). The adaptive alpha of
// plan 14.4 is a LATER batch --- here alpha is caller-provided.

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"
#include "tile_compile/reconstruction/adaptive_alpha.hpp"
#include "tile_compile/reconstruction/alpha_guard.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"  // ProfilePlane

#include <cstdint>
#include <vector>

namespace tile_compile::reconstruction {

// Fixed working luminance for the energy guard (plan 14.5) and the plan-16.3
// artifact's `luma_definition`. MONO uses L directly; OSC combines the three
// channels with these weights. Single source of truth so the recorded
// `luma_definition` string and the actual math cannot drift.
inline constexpr double kWorkingLumaWeightsOsc[3] = {0.25, 0.5, 0.25};
inline constexpr const char *kWorkingLumaDefinition = "0.25R+0.50G+0.25B";

struct MultibandChannelInput {
  const ProfilePlane *uniform = nullptr;  // required
  const ProfilePlane *raw = nullptr;      // required
  const ProfilePlane *fine = nullptr;     // required if levels >= 1 (always)
  const ProfilePlane *medium = nullptr;   // required if levels >= 2
};

struct MultibandFusionParams {
  int levels = 3;  // [1, 4]
};

struct MultibandChannelResult {
  int width = 0;
  int height = 0;
  std::vector<float> value;       // X_out; NaN off support
  std::vector<uint8_t> support;   // M_U,L && M_R,L && all D_R,j valid
  long long pixels_supported = 0;
};

// Reference multi-band fusion for ONE channel. `alpha_by_band[j]` is the
// per-pixel alpha for band j+1 (row-major, size width*height, values in
// [0,1]); an empty inner vector means alpha == 1 for that band. `alpha_by_band`
// itself may be empty (alpha == 1 for every band). Bands whose source is R
// ignore alpha entirely.
MultibandChannelResult fuse_multiband_channel(
    const MultibandChannelInput &in, int width, int height,
    const MultibandFusionParams &params,
    const std::vector<std::vector<float>> &alpha_by_band = {});

// ---- Whole-frame orchestrator (in-memory reference path, plan 14) --------

struct MultibandResult {
  int width = 0;
  int height = 0;
  ColorMode mode = ColorMode::MONO;
  // X_out per active channel (row-major, NaN off support).
  std::vector<float> R, G, B, L;
  std::vector<uint8_t> support_R, support_G, support_B, support_L;
  // The shared alpha actually applied to band j (0-based): alpha_pre ->
  // energy guard -> B3 smoothing. Empty for Raw-sourced bands.
  std::vector<std::vector<float>> alpha_final;
  long long pixels_supported = 0;  // sum over channels
};

// Runs the full plan-14 pipeline on whole-frame profile planes:
//   adaptive alpha (14.4) -> local energy guard (14.5) -> B3 alpha smoothing
//   (14.7) -> per-channel band blend (14.3).
// One alpha_j is computed and shared across R/G/B (plan 14.6); the energy
// guard works on the fixed working luminance 0.25R+0.5G+0.25B (D_L for MONO).
// `medium` may be empty when levels < 2. A_separation / A_artifact /
// A_registration are supplied as pre-reduced per-pixel maps (empty => 1);
// `background_band_floor[j]` is background_band_floor_j (empty => 0).
MultibandResult fuse_multiband(
    const ForwardDrizzleUniformResult &uniform,
    const ForwardDrizzleUniformResult &raw,
    const ForwardDrizzleUniformResult &fine,
    const ForwardDrizzleUniformResult &medium, ColorMode mode, int width,
    int height, const config::ReconstructionMultibandConfig &mb_cfg,
    const AdaptiveAlphaParams &alpha_params = {},
    const EnergyGuardParams &guard_params = {},
    const std::vector<float> &a_separation = {},
    const std::vector<float> &a_artifact = {},
    const std::vector<float> &a_registration = {},
    const std::vector<double> &background_band_floor = {});

// ---- Streamed (row-striped) fusion, verified bit-exact vs. fuse_multiband ---

// Vertical halo (internal px) a fusion stripe must fetch on each side of its
// committed core rows so that every committed pixel is byte-identical to the
// whole-frame fuse_multiband result. Covers: cumulative a-trous vertical
// reach over all `levels` (2*(2^levels - 1)), the widest energy-guard MAD
// window (energy_guard_window_radius(levels)) and the B3 alpha-smoothing
// reach (2). Deliberately conservative --- the guard only runs on bands 1-2,
// so the strict lower bound is smaller; over-padding only costs work.
int multiband_fusion_halo_rows(int levels);

// Same contract and result as fuse_multiband(), computed one row-stripe of
// `chunk_rows` committed rows at a time (each stripe internally runs
// fuse_multiband on its core +/- multiband_fusion_halo_rows() and commits
// only the core). Byte-identical to fuse_multiband() on every supported
// pixel EXCEPT, by construction, where a 4-connected alpha-support component
// is joined only by a path that leaves a stripe's fetched region (the B3
// flood fill is the one non-local step); such divergence is bounded and
// confined to pixels next to a support pinch. `chunk_rows <= 0` => one stripe
// (identical to fuse_multiband). Raw-sourced bands keep an EMPTY alpha_final.
MultibandResult fuse_multiband_streamed(
    const ForwardDrizzleUniformResult &uniform,
    const ForwardDrizzleUniformResult &raw,
    const ForwardDrizzleUniformResult &fine,
    const ForwardDrizzleUniformResult &medium, ColorMode mode, int width,
    int height, const config::ReconstructionMultibandConfig &mb_cfg,
    int chunk_rows, const AdaptiveAlphaParams &alpha_params = {},
    const EnergyGuardParams &guard_params = {},
    const std::vector<float> &a_separation = {},
    const std::vector<float> &a_artifact = {},
    const std::vector<float> &a_registration = {},
    const std::vector<double> &background_band_floor = {});

// ---- End-to-end reference path (drizzle U/R/F/M + alpha maps -> fuse) -----

struct MultibandReconstructionParams {
  // `multiband.fine_quality_exponent` / `.medium_quality_exponent` are the sole
  // source of truth for the Fine/Medium Q-weight exponents (plan 14.3).
  config::ReconstructionMultibandConfig multiband{};
  AdaptiveAlphaParams alpha{};
  EnergyGuardParams guard{};
  AlphaConfidenceParams alpha_confidence{};
  std::vector<double> background_band_floor{};  // per band; empty => 0
};

// Runs one whole-frame forward-drizzle emitting U/R/F/M and the
// A_separation/A_artifact/A_registration maps, then fuses per plan 14. This
// is the in-memory reference (plan 14.7) --- production streams from the
// transactional stores instead. `quality_of` must supply composite + artifact
// (and scale0/scale1 for levels >= 1/2).
MultibandResult reconstruct_multiband_reference(
    const registration::RegistrationSamplingPlan &plan,
    const SourceImageProvider &source_of,
    const config::ReconstructionDrizzleConfig &drizzle_cfg,
    const config::ReconstructionClippingConfig &clip_cfg,
    const FrameQualityProvider &quality_of,
    const MultibandReconstructionParams &params,
    const ForwardDrizzleSubdivisionParams &subdivision = {},
    const std::vector<float> &g_eff_by_source_index = {});

}  // namespace tile_compile::reconstruction
