# tile_compile_cpp example profiles (Methodik v3.2)

All example files in this folder are **complete standalone configurations** and include
all currently available config options with inline explanations.

They are kept in sync with v3.2 runner/config parser defaults, including:

- `dithering.*`
  - documents acquisition dithering expectation and shift threshold used for diagnostics.
- `chroma_denoise.*`
  - includes full OSC-oriented chroma-noise reduction block with inline comments.
  - contains preset hints directly in YAML comments:
    - conservative
    - balanced
    - aggressive
  - in MONO profile this block is intentionally present but disabled for completeness.
- `stacking.cluster_quality_weighting.*`
  - includes full v3.2.2 cluster-quality weighting block:
    - `enabled`
    - `kappa_cluster`
    - `cap_enabled`
    - `cap_ratio`
  - documents the weighting model used in final cluster aggregation:
    - `w_k = exp(kappa_cluster * Q_k)`
  - includes optional dominance cap semantics:
    - `w_k <= cap_ratio * median_j(w_j)` (only when `cap_enabled: true`)

## Profiles

- `tile_compile.full_mode.example.yaml`
  - For datasets with enough usable frames for full mode.
  - Chroma denoise profile: balanced (reference values).
- `tile_compile.reduced_mode.example.yaml`
  - For 50..(frames_reduced_threshold-1) usable frames.
  - Chroma denoise profile: moderately conservative (detail-preserving).
- `tile_compile.emergency_mode.example.yaml`
  - Enables emergency reduced mode for datasets with <50 usable frames.
  - Chroma denoise profile: aggressive (strong chroma noise suppression).
- `tile_compile.smart_telescope_dwarf_seestar.example.yaml`
  - Suggested full config for DWARF / ZWO Seestar OSC stacks.
  - Chroma denoise profile: conservative (protect small-scale detail).
- `tile_compile.canon_low_n_high_quality.example.yaml`
  - Suggested OSC config for Canon-style datasets with low frame count but high/consistent quality.
  - Anti-grid focus for reduced/emergency operation: larger tiles, higher overlap, conservative weighting.
- `tile_compile.mono_full_mode.example.yaml`
  - Suggested full config for MONO datasets in full mode.
  - Chroma denoise block included for completeness, but disabled by default.
- `tile_compile.mono_small_n_anti_grid.example.yaml`
  - Suggested MONO config for small frame counts (e.g. 10..40) where tile seams/patterns can appear.
  - Anti-grid focus: larger tiles, higher overlap, conservative local weighting, reduced denoise aggressiveness.
- `tile_compile.mono_small_n_ultra_conservative.example.yaml`
  - Suggested MONO config for very small frame counts (e.g. 8..25) where maximum seam stability is preferred over aggressive enhancement.
  - Ultra-conservative focus: tile denoise disabled, very soft local weighting, larger tiles and higher overlap.

## Usage

1. Copy one example file and adapt paths/device-specific values:
   - `run_dir`
   - `input.pattern`
   - `data.image_width` / `data.image_height`
   - `data.bayer_pattern`
   - (optional) tune `dithering.min_shift_px` to your mount behavior
   - (optional, OSC) tune `chroma_denoise.blend.amount` and `chroma_denoise.apply_stage`
   - (optional) tune `stacking.cluster_quality_weighting.*` if cluster weighting is too strong/weak
2. Run directly with:

```bash
./build/tile_compile_runner --config examples/tile_compile.full_mode.example.yaml
```

### Optional: merge workflow (overlay style)

If you still prefer overlay-style usage, merge your base config with a profile
using your YAML merge tool of choice, then run with the merged file.
