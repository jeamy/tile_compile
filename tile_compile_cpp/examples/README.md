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
- `tile_compile.mono_full_mode.example.yaml`
  - Suggested full config for MONO datasets in full mode.
  - Chroma denoise block included for completeness, but disabled by default.

## Usage

1. Copy one example file and adapt paths/device-specific values:
   - `run_dir`
   - `input.pattern`
   - `data.image_width` / `data.image_height`
   - `data.bayer_pattern`
   - (optional) tune `dithering.min_shift_px` to your mount behavior
   - (optional, OSC) tune `chroma_denoise.blend.amount` and `chroma_denoise.apply_stage`
2. Run directly with:

```bash
./build/tile_compile_runner --config examples/tile_compile.full_mode.example.yaml
```

### Optional: merge workflow (overlay style)

If you still prefer overlay-style usage, merge your base config with a profile
using your YAML merge tool of choice, then run with the merged file.
