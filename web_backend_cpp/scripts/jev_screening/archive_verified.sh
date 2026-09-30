#!/bin/bash
# Move finished runs from the fast NVMe to the archive, verified: copy, compare by checksum, delete the source only if identical.
#
#   archive_verified.sh --src /media/tc_500 --dst /media/data/tile_compile_cache/jev-test [--status FILE] RUN_ID [RUN_ID ...]
#
# - Refuses to overwrite a run that already exists at the destination (it is skipped and reported).
# - `rsync -acn --delete` after the copy must list no difference; otherwise the source is kept ("VERIFY FAILED").
# - Never run it on a run that is still being written. Status lines: "<id> moved ok", "... COPY FAILED", "... VERIFY FAILED", "DONE".
set -u
SRC=; DST=; STATUS=/tmp/archive_runs_status.txt; IDS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --src) SRC=$2; shift 2;; --dst) DST=$2; shift 2;; --status) STATUS=$2; shift 2;;
    -*) echo "unknown argument: $1" >&2; exit 2;; *) IDS+=("$1"); shift;;
  esac
done
[ -d "$SRC" ] && [ -d "$DST" ] && [ ${#IDS[@]} -gt 0 ] || { echo "usage: archive_verified.sh --src DIR --dst DIR RUN_ID..." >&2; exit 2; }
command -v rsync > /dev/null || { echo "rsync is required" >&2; exit 2; }
for id in "${IDS[@]}"; do
  case "$id" in */*|.|..) echo "$id is not a plain run id" >> "$STATUS"; continue;; esac
  [ -d "$SRC/$id" ] || { echo "$id missing at source" >> "$STATUS"; continue; }
  [ -e "$DST/$id" ] && { echo "$id already exists at the destination - skipped" >> "$STATUS"; continue; }
  if ! rsync -a "$SRC/$id/" "$DST/$id/" >> /tmp/archive_runs_rsync.log 2>&1; then echo "$id COPY FAILED" >> "$STATUS"; continue; fi
  differences=$(rsync -acn --delete "$SRC/$id/" "$DST/$id/" | grep -v '^$' | wc -l)
  if [ "$differences" -eq 0 ]; then rm -rf "$SRC/$id" && echo "$id moved ok $(date -u +%T)" >> "$STATUS"
  else echo "$id VERIFY FAILED ($differences differences) - source kept" >> "$STATUS"; fi
done
echo DONE >> "$STATUS"
