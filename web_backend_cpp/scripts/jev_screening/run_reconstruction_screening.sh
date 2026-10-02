#!/bin/bash
# Run a screening chain: one `tile_compile_runner reconstruct` per config of a config directory, strictly one after the other.
#
#   run_reconstruction_screening.sh --config-dir DIR --input-dir INPUT --runs-dir RUNS --prefix scr_ic4605_120 \
#                                   [--max-frames 120] [--status FILE] [--min-free-gb 150] [--runner PATH] [--project-root DIR]
#
# DIR comes from make_screening_configs.py (names.json gives the order, control first). Run ids are <prefix>_<name>.
# - A finished arm is pruned (cache/, outputs/calibrated/) with tile_compile_cpp/scripts/prune_evaluation_run.py.
# - A failed arm does NOT stop the chain: a run that fails a protected gate (e.g. FORWARD_STAGE_COVERAGE_GATE_FAILED) is a result,
#   so the failure and its reason are recorded and the chain continues. Only too little free disk space stops it.
# - Put RUNS on the NVMe: on the RAID (/media/data) a run takes about five times as long.
# Status lines (one per event): "<id> started|ok|FAILED ...|NOT STARTED ..." and a final "DONE".
set -u
CONFIG_DIR=; INPUT_DIR=; RUNS_DIR=; PREFIX=; MAX_FRAMES=0; STATUS=/tmp/screening_status.txt; MIN_FREE_GB=150
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../../.." && pwd)"
RUNNER="$PROJECT_ROOT/tile_compile_cpp/build/tile_compile_runner"
while [ $# -gt 0 ]; do
  case "$1" in
    --config-dir) CONFIG_DIR=$2; shift 2;; --input-dir) INPUT_DIR=$2; shift 2;; --runs-dir) RUNS_DIR=$2; shift 2;;
    --prefix) PREFIX=$2; shift 2;; --max-frames) MAX_FRAMES=$2; shift 2;; --status) STATUS=$2; shift 2;;
    --min-free-gb) MIN_FREE_GB=$2; shift 2;; --runner) RUNNER=$2; shift 2;; --project-root) PROJECT_ROOT=$2; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
for v in CONFIG_DIR INPUT_DIR RUNS_DIR PREFIX; do [ -n "${!v}" ] || { echo "missing --${v,,}" | tr _ - >&2; exit 2; }; done
[ -x "$RUNNER" ] || { echo "runner not found: $RUNNER" >&2; exit 2; }

run_succeeded() {  # run id -> 0 when the last run_end event says success
  python3 - "$RUNS_DIR/$1/logs/run_events.jsonl" <<'PY'
import json, sys
last = None
try:
    for line in open(sys.argv[1]):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "run_end":
            last = e
except OSError:
    pass
sys.exit(0 if last and last.get("success") is True else 1)
PY
}

for name in $(python3 -c "import json,sys;print(' '.join(json.load(open(sys.argv[1]+'/names.json'))))" "$CONFIG_DIR"); do
  id="${PREFIX}_${name}"
  free_gb=$(df --output=avail -BG "$RUNS_DIR" | tail -1 | tr -dc 0-9)
  if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then echo "$id NOT STARTED: only ${free_gb} GB free - chain stopped" >> "$STATUS"; break; fi
  echo "$id started $(date -u +%FT%TZ)" >> "$STATUS"
  ( cd "$PROJECT_ROOT" && "$RUNNER" reconstruct --config "$CONFIG_DIR/$name.yaml" --input-dir "$INPUT_DIR" --runs-dir "$RUNS_DIR" \
      --project-root "$PROJECT_ROOT" --run-id "$id" --max-frames "$MAX_FRAMES" > "/tmp/${id}.log" 2>&1 )
  rc=$?
  if [ $rc -eq 0 ] && run_succeeded "$id"; then
    python3 "$PROJECT_ROOT/tile_compile_cpp/scripts/prune_evaluation_run.py" "$RUNS_DIR/$id" --apply >> "/tmp/${PREFIX}_prune.log" 2>&1
    echo "$id ok $(date -u +%FT%TZ)" >> "$STATUS"
  else
    reason=$(grep -ho "FORWARD_[A-Z_0-9]*" "$RUNS_DIR/$id/logs/run_events.jsonl" 2>/dev/null | tail -1)
    [ -n "$reason" ] || reason=$(tail -1 "/tmp/${id}.log" | cut -c1-120)
    echo "$id FAILED rc=$rc ${reason} - continuing" >> "$STATUS"
    rm -rf "$RUNS_DIR/$id/cache" "$RUNS_DIR/$id/outputs/calibrated"
  fi
done
echo DONE >> "$STATUS"
