#!/bin/bash
set -u
BASE=/media/data/siril_compare
LOG=$BASE/driver.log
declare -A SRC=(
 [ic434]="/media/tc_ssd/IC434_ligths_all"
 [ic5070]="/media/tc_ssd/IC5070_2"
 [m42]="/media/tc_ssd/M42_02.2026_lights_all"
 [m66]="/media/tc_ssd/M66_lights"
)
ORDER="ic434 ic5070 m42 m66"
echo "=== driver start $(date) ===" | tee -a "$LOG"
for OBJ in $ORDER; do
  S="${SRC[$OBJ]}"
  W="$BASE/$OBJ"
  echo "--- $OBJ  src=$S  $(date)" | tee -a "$LOG"
  rm -rf "$W"; mkdir -p "$W/process"
  SCRIPT="$W/$OBJ.ssf"
  cat > "$SCRIPT" << SSF
requires 1.2.0
link light -out=$W/process
cd $W/process
calibrate light -debayer
register pp_light
stack r_pp_light rej 3 3 -norm=addscale -output_norm -out=$BASE/${OBJ}_siril
close
SSF
  df -h /media/data | tail -1 | tee -a "$LOG"
  /usr/bin/time -v siril-cli -d "$S" -s "$SCRIPT" > "$W/siril.log" 2>&1
  RC=$?
  echo "    rc=$RC  $(date)" | tee -a "$LOG"
  ls -la "$BASE/${OBJ}_siril.fit" 2>&1 | tee -a "$LOG"
  tail -5 "$W/siril.log" | tee -a "$LOG"
  rm -rf "$W/process"
  echo "    cleaned process/  $(date)" | tee -a "$LOG"
done
echo "=== driver done $(date) ===" | tee -a "$LOG"
