# Raw Stack GUI

> **Note:** This feature is not optimized and is retained for legacy reasons only.

The Raw Stack page provides standalone preprocessing from FITS light frames to a final stacked and post-processed image, running fully separately from the normal Tile-Compile run studio.

## Pipeline phases

Calibration → CFA/Mono Prep → Registration → Quality Analysis → Frame Filtering → Stacking (Sigma/Median/Winsor) → Astrometry (ASTAP) → BGE → PCC → HyperMetric Stretch

All configurable parameters (sigma-clip, rejection method, stacking weighting, BGE, PCC, Astrometry, and HyperMetric Stretch) are taken directly from the Parameter Studio configuration — no hardcoded values.

## Documentation

- English: [Raw Stack GUI (EN)](../raw_stack_gui_en.md)
- German: [Raw Stack GUI (DE)](../raw_stack_gui_de.md)
