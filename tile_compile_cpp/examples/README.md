# tile_compile_cpp example profiles (Methodik v3.2)

All example files in this folder are **complete standalone configurations** and include
all currently available config options with inline explanations.

## Profiles

- `tile_compile.full_mode.example.yaml`
  - For datasets with enough usable frames for full mode.
- `tile_compile.reduced_mode.example.yaml`
  - For 50..(frames_reduced_threshold-1) usable frames.
- `tile_compile.emergency_mode.example.yaml`
  - Enables emergency reduced mode for datasets with <50 usable frames.
- `tile_compile.smart_telescope_dwarf_seestar.example.yaml`
  - Suggested full config for DWARF / ZWO Seestar OSC stacks.
- `tile_compile.mono_full_mode.example.yaml`
  - Suggested full config for MONO datasets in full mode.

## Usage

1. Copy one example file and adapt paths/device-specific values:
   - `run_dir`
   - `input.pattern`
   - `data.image_width` / `data.image_height`
   - `data.bayer_pattern`
2. Run directly with:

```bash
./build/tile_compile_runner --config examples/tile_compile.full_mode.example.yaml
```

### Optional: merge workflow (overlay style)

If you still prefer overlay-style usage, merge your base config with a profile
using your YAML merge tool of choice, then run with the merged file.
