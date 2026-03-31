# Test Plan: PCC White Reference Fix

## Problem Summary
Current PCC produces green cast on IC434 because:
- HEAD uses fixed neutral reference (wrg=1.0, wbg=1.0)
- On red fields, this over-corrects red channel → green cast
- Old adaptive reference used all-star median → biased on red fields

## Solution
Use catalog-white stars (Teff 5500-7000K) for reference:
- Field-independent (white stars are white everywhere)
- Robust to field color composition
- Physically meaningful

## Test Scenarios

### 1. IC434 (Red Field)
**Expected before fix:**
- Matrix: R~0.97, G=1.0, B=1.0 → green cast
- wrg=1.0, wbg=1.0 (neutral reference)

**Expected after fix:**
- White stars identified from catalog
- wrg > 1.0 (compensates for red field)
- Matrix: More balanced R/G/B ratios
- No green cast

### 2. Neutral Field (M31, NGC7000)
**Expected:**
- Should work identically (wrg≈1.0, wbg≈1.0)
- No regression

### 3. Blue Field (if available)
**Expected:**
- wrg < 1.0, wbg may vary
- No blue/magenta cast

## Validation

Run on IC434:
```bash
cd /home/mux/programme/tile_compile
./tile_compile_cpp/build/tile_compile_cli resume \
  --run-dir runs/ic434_20260330_134819 \
  --from-phase PCC
```

Check logs for:
1. `[PCC] White reference: wrg=X wbg=Y method=... white_stars=N`
2. Compare wrg/wbg values
3. Check final matrix
4. Visual inspection of stacked_rgb_pcc.fits

Expected improvement:
- wrg should be > 1.0 (not 1.0)
- method should be "white_stars_5500_7000K" or "white_stars_broad_4500_8000K"
- Matrix R gain should be closer to 1.0
- No green cast in output

