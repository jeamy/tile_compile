#!/bin/bash
# Iterative BGE parameter optimization script
# Resumes from BGE phase, analyzes results, adjusts parameters until improvement

set -e

# Configuration
RUN_DIR="/home/mux/programme/tile_compile/tile_compile_cpp/build/runs/20260303_092027_250beffd"
TILE_NAME="IC434_ligths_all"
WORK_DIR="${RUN_DIR}/${TILE_NAME}"
CONFIG_FILE="${WORK_DIR}/config.yaml"
OUTPUT_DIR="${WORK_DIR}/outputs"
RUNNER_BIN="/home/mux/programme/tile_compile/tile_compile_cpp/build/tile_compile_runner"

# Reference files
BASELINE_FILE="${OUTPUT_DIR}/stacked_rgb_pcc_iter1.fits"
SOLVE_FILE="${OUTPUT_DIR}/stacked_rgb_solve.fits"

# Optimization parameters
MAX_ITERATIONS=5
BASELINE_GRADIENT=0.0
BEST_GRADIENT=999999.0
ITERATION=0
START_ITERATION=2

echo "============================================================"
echo "BGE ITERATIVE OPTIMIZATION"
echo "============================================================"
echo "Run directory: ${RUN_DIR}"
echo "Tile: ${TILE_NAME}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo ""

# Check if baseline exists
if [ ! -f "${BASELINE_FILE}" ]; then
    echo "ERROR: Baseline file not found: ${BASELINE_FILE}"
    exit 1
fi

# Analyze baseline
echo "Analyzing baseline gradient..."
python3 /tmp/analyze_gradient_simple.py "${BASELINE_FILE}" > "${OUTPUT_DIR}/baseline_gradient.txt"
BASELINE_GRADIENT=$(grep "Left gradient:" "${OUTPUT_DIR}/baseline_gradient.txt" | awk '{print $3}')
BEST_GRADIENT=${BASELINE_GRADIENT}

echo "Baseline left gradient: ${BASELINE_GRADIENT}"
echo ""

# Backup original config
cp "${CONFIG_FILE}" "${CONFIG_FILE}.original"

# Parameter sets to try (progressive smoothing increase)
declare -a PARAM_SETS=(
    "1.8:3e-2:0.7:36:56"    # Iteration 1: moderate increase
    "2.0:5e-2:0.6:38:54"    # Iteration 2: stronger smoothing
    "2.2:7e-2:0.5:40:52"    # Iteration 3: aggressive smoothing
    "2.5:1e-1:0.4:42:50"    # Iteration 4: very aggressive
    "2.8:1.5e-1:0.3:44:48"  # Iteration 5: maximum smoothing
)

# Iteration loop
for ITERATION in {2..5}; do
    echo "============================================================"
    echo "ITERATION ${ITERATION}/${MAX_ITERATIONS}"
    echo "============================================================"
    
    # Get parameter set
    PARAMS=${PARAM_SETS[$((ITERATION-1))]}
    IFS=':' read -r MU LAMBDA EPSILON N_G G_MIN <<< "$PARAMS"
    
    echo "Testing parameters:"
    echo "  rbf_mu_factor: ${MU}"
    echo "  rbf_lambda: ${LAMBDA}"
    echo "  rbf_epsilon: ${EPSILON}"
    echo "  N_g: ${N_G}"
    echo "  G_min_px: ${G_MIN}"
    echo ""
    
    # Update config with new parameters
    python3 /tmp/update_bge_params.py "${CONFIG_FILE}" "${MU}" "${LAMBDA}" "${EPSILON}" "${N_G}" "${G_MIN}"
    
    # Resume from BGE phase
    echo "Running tile_compile_runner resume --run-dir ${WORK_DIR} --from-phase BGE..."
    ${RUNNER_BIN} resume --run-dir "${WORK_DIR}" --from-phase BGE 2>&1 | tee "${OUTPUT_DIR}/iteration_${ITERATION}.log"
    
    # Check if PCC output was created
    ITER_OUTPUT="${OUTPUT_DIR}/stacked_rgb_pcc.fits"
    if [ ! -f "${ITER_OUTPUT}" ]; then
        echo "ERROR: Iteration ${ITERATION} failed - no output file"
        continue
    fi
    
    # Rename output for this iteration
    mv "${ITER_OUTPUT}" "${OUTPUT_DIR}/stacked_rgb_pcc_iter${ITERATION}.fits"
    ITER_OUTPUT="${OUTPUT_DIR}/stacked_rgb_pcc_iter${ITERATION}.fits"
    
    # Analyze gradient
    echo ""
    echo "Analyzing iteration ${ITERATION} gradient..."
    python3 /tmp/analyze_gradient_simple.py "${ITER_OUTPUT}" > "${OUTPUT_DIR}/iter${ITERATION}_gradient.txt"
    ITER_GRADIENT=$(grep "Left gradient:" "${OUTPUT_DIR}/iter${ITERATION}_gradient.txt" | awk '{print $3}')
    
    echo "Iteration ${ITERATION} left gradient: ${ITER_GRADIENT}"
    echo "Best gradient so far: ${BEST_GRADIENT}"
    echo ""
    
    # Compare with best
    IMPROVEMENT=$(python3 -c "print(${BEST_GRADIENT} - ${ITER_GRADIENT})")
    IMPROVEMENT_PCT=$(python3 -c "print((${BEST_GRADIENT} - ${ITER_GRADIENT}) / ${BEST_GRADIENT} * 100)")
    
    echo "Improvement: ${IMPROVEMENT} (${IMPROVEMENT_PCT}%)"
    
    # Check if this is better
    IS_BETTER=$(python3 -c "print(1 if ${ITER_GRADIENT} < ${BEST_GRADIENT} else 0)")
    
    if [ "${IS_BETTER}" == "1" ]; then
        echo "✓ IMPROVEMENT FOUND!"
        BEST_GRADIENT=${ITER_GRADIENT}
        BEST_ITERATION=${ITERATION}
        
        # Copy as best result
        cp "${ITER_OUTPUT}" "${OUTPUT_DIR}/stacked_rgb_pcc_best.fits"
        cp "${CONFIG_FILE}" "${OUTPUT_DIR}/config_best_iter${ITERATION}.yaml"
        
        # Check if improvement is significant enough to stop
        SIGNIFICANT=$(python3 -c "print(1 if ${IMPROVEMENT_PCT} > 10.0 else 0)")
        if [ "${SIGNIFICANT}" == "1" ]; then
            echo "Significant improvement (>10%) achieved. Stopping optimization."
            break
        fi
    else
        echo "✗ No improvement over best result"
    fi
    
    echo ""
done

# Final summary
echo "============================================================"
echo "OPTIMIZATION COMPLETE"
echo "============================================================"
echo "Baseline gradient: ${BASELINE_GRADIENT}"
echo "Best gradient: ${BEST_GRADIENT}"
echo "Best iteration: ${BEST_ITERATION}"
echo ""
echo "Files created in ${OUTPUT_DIR}:"
echo "  - stacked_rgb_pcc_best.fits (best result)"
echo "  - config_best_iter${BEST_ITERATION}.yaml (best config)"
echo "  - stacked_rgb_pcc_iter*.fits (all iterations)"
echo ""

# Restore best config as final
if [ -n "${BEST_ITERATION}" ]; then
    cp "${OUTPUT_DIR}/config_best_iter${BEST_ITERATION}.yaml" "${CONFIG_FILE}"
    echo "Best configuration restored to ${CONFIG_FILE}"
fi

echo "Optimization complete!"
