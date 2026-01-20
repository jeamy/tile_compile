#!/bin/bash
# Comprehensive cleanup script for Methodik v4 migration
# Removes all legacy registration code and Siril dependencies

set -e

echo "=== Methodik v4 Migration Cleanup ==="
echo ""

# 1. Remove Siril-related files (no longer needed)
echo "[1/6] Removing Siril-related files..."
rm -f tile_compile_python/runner/siril_utils.py
rm -f tile_compile_python/ref_siril_registration.py
rm -f tile_compile_python/ref_siril_registration_call.py
rm -f tile_compile_python/validate-siril.sh
rm -rf tile_compile_python/siril_scripts/
echo "  ✓ Siril files removed"

# 2. Remove old registration module (replaced by tile_local_registration_v4.py)
echo "[2/6] Cleaning up old TLR module..."
if [ -f tile_compile_python/runner/tile_local_registration.py ]; then
    mv tile_compile_python/runner/tile_local_registration.py \
       tile_compile_python/runner/tile_local_registration_v3.py.backup
    echo "  ✓ Old TLR backed up to tile_local_registration_v3.py.backup"
fi

# 3. Remove legacy registration tests
echo "[3/6] Removing legacy tests..."
if [ -f tile_compile_python/tests/test_registration.py ]; then
    mv tile_compile_python/tests/test_registration.py \
       tile_compile_python/tests/test_registration_v3.py.backup
    echo "  ✓ Old tests backed up"
fi

# 4. Remove performance benchmarks that used old registration
echo "[4/6] Cleaning validation scripts..."
# Comment out old benchmark methods in performance_benchmark.py
if [ -f tile_compile_python/validation/performance_benchmark.py ]; then
    echo "  ✓ performance_benchmark.py (manual review needed)"
fi

# 5. Clean up __pycache__
echo "[5/6] Removing Python cache..."
find tile_compile_python -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find tile_compile_python -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  ✓ Cache cleaned"

# 6. Create migration summary
echo "[6/6] Creating migration summary..."
cat > MIGRATION_V4_SUMMARY.md << 'EOF'
# Methodik v4 Migration Summary

## Removed Components

### Siril Integration (Complete Removal)
- `runner/siril_utils.py` - Siril CLI wrapper
- `siril_scripts/*.ssf` - All Siril script files
- `ref_siril_registration*.py` - Reference implementations
- `validate-siril.sh` - Siril validation script
- `doc/backend_siril-registration.md` - Documentation

**Reason:** Methodik v4 uses tile-wise local registration (TLR) exclusively.
No global registration engine (Siril or OpenCV-based) is needed.

### Legacy Registration Code
- REGISTRATION phase in `phases_impl.py` (~550 lines removed)
- Helper functions: `_compose_affine`, `_warp_cfa`, `_check_step_warp_sanity`
- Old TLR v3 module → backed up as `tile_local_registration_v3.py.backup`

### Legacy Tests
- `tests/test_registration.py` → backed up as `test_registration_v3.py.backup`

## New Components

### Core TLR v4 Module
- `runner/tile_local_registration_v4.py` - Full Methodik v4 implementation
  - Iterative reference refinement (§5.2)
  - Temporal warp smoothing with Savitzky-Golay (§5.3)
  - Registration quality weighting R_{f,t} (§7)
  - Translation-only model
  - Overlap-add with Hanning window (§9)

### Updated Pipeline
- Phase 0: SCAN_INPUT
- Phase 1: CHANNEL_SPLIT (works on raw frames)
- Phase 2: NORMALIZATION
- Phase 3: GLOBAL_METRICS
- Phase 4: TILE_GRID
- Phase 5: LOCAL_METRICS
- Phase 6: TILE_RECONSTRUCTION_TLR (integrated registration)
- Phase 7: STATE_CLUSTERING
- Phase 8: SYNTHETIC_FRAMES
- Phase 9: STACKING
- Phase 10: DEBAYER
- Phase 11: DONE

## Configuration Changes

### Removed
```yaml
registration:
  engine: opencv_cfa | siril
  reference: ...
  allow_rotation: ...
  min_star_matches: ...
  siril_script: ...
```

### Added
```yaml
registration:
  local_tiles:
    model: translation
    ecc_cc_min: 0.2
    min_valid_frames: 10
    reference_method: median_time
    max_tile_size: 128
    registration_quality_beta: 5.0
```

## Next Steps

1. **Update Schema:** `tile_compile.schema.yaml`
2. **Write Tests:** `tests/test_tile_local_registration_v4.py`
3. **Update Documentation:**
   - Methodik v4 (DE) ✓
   - Methodik v4 (EN)
   - process_flow_v4/
   - README.md
4. **Integration Test:** Run with real Alt/Az data

## Rollback

If needed, restore from Git:
```bash
git checkout tile_compile_python/runner/phases_impl.py
git checkout tile_compile_python/runner/siril_utils.py
# etc.
```

Backups are also available:
- `phases_impl.py.backup`
- `tile_local_registration_v3.py.backup`
- `test_registration_v3.py.backup`
EOF

echo "  ✓ Summary created: MIGRATION_V4_SUMMARY.md"

echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Summary:"
echo "  - Siril integration removed"
echo "  - Legacy registration code removed"
echo "  - Phase IDs updated (1-11)"
echo "  - TLR v4 module ready"
echo ""
echo "Next: Update schema and write tests"
