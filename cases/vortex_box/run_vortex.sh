#!/bin/bash
# vortex_box inside the mhh-vortex image. Computes in /tmp/w (bind mounts on
# Windows are slow and fail on large writes) and copies the output to /out.
# Arguments: KEY=VALUE overrides for vortex_box.ini (e.g. cs=0.23 vf_amp=0.3),
# plus QT0=<kg/kg> for the near-surface humidity of the input profile.
set -uo pipefail
CASE=vortex_box
mkdir -p /tmp/w && cd /tmp/w || exit 1
cp /app/cases/vortex_box/vortex_box.ini /app/cases/vortex_box/vortex_box_input.py .
for kv in "$@"; do
  case "${kv%%=*}" in
    QT0) sed -i "s/^theta0, qt0 = 300., .*/theta0, qt0 = 300., ${kv#*=}/" vortex_box_input.py ;;
    *)   sed -i "s/^${kv%%=*}=.*/${kv}/" ${CASE}.ini ;;
  esac
  echo "override: ${kv}"
done
echo "=== input $(date -Is)"; python3 vortex_box_input.py || exit 3
echo "=== init $(date -Is)";  /app/build/microhh init ${CASE} > init.log 2>&1 || { tail -20 init.log; exit 4; }
echo "=== run $(date -Is)"
s=$(date +%s)
/app/build/microhh run ${CASE} > run.log 2>&1; rc=$?
echo "=== run finished rc=${rc} wall=$(( $(date +%s) - s )) s $(date -Is)"
tail -2 ${CASE}.out 2>/dev/null
echo "copied rc=${rc}"
exit ${rc}
