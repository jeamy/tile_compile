#!/bin/bash
BASE=/media/data/siril_compare
if ! pgrep -f run_all.sh >/dev/null; then echo "driver NOT running"; else
  echo "driver running, elapsed $(ps -o etime= -p $(pgrep -f run_all.sh|head -1)|tr -d ' ')"
fi
for OBJ in ic434 ic5070 m42 m66; do
  OUT="$BASE/${OBJ}_siril.fit"
  if [ -f "$OUT" ]; then
    printf "  %-7s DONE  %s\n" "$OBJ" "$(ls -la "$OUT"|awk '{print $5" bytes  "$6" "$7" "$8}')"
  elif [ -d "$BASE/$OBJ/process" ]; then
    L="$BASE/$OBJ/siril.log"
    STAGE=$(grep -aoE "running command (calibrate|register|stack)" "$L" 2>/dev/null|tail -1|awk '{print $3}')
    PCT=$(tail -c 2000 "$L" 2>/dev/null|tr '\r' '\n'|grep -aoE "[0-9]+\.[0-9]+%"|tail -1)
    PP=$(ls "$BASE/$OBJ/process"/pp_light_*.fit 2>/dev/null|wc -l)
    RP=$(ls "$BASE/$OBJ/process"/r_pp_light_*.fit 2>/dev/null|wc -l)
    printf "  %-7s ACTIVE stage=%-9s %-7s  pp=%s r=%s\n" "$OBJ" "${STAGE:-?}" "${PCT:-}" "$PP" "$RP"
  else
    printf "  %-7s pending\n" "$OBJ"
  fi
done
df -h /media/data | tail -1
