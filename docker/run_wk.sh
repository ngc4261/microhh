#!/bin/bash
# Run the weisman_klemp supercell case end to end inside the sim-hub container (code MHH).
#
# Expects the current directory (/work, mounted by sim-hub) to hold weisman_klemp.ini
# (written by sim_ui/adapter.py). Any arguments are passed to bubble_theta.py
# (e.g. --bubamp 2 --zbub 1400 --lxybub 10000 --lzbub 1400 --seed 2).
#
# Stages (each prints a marker that adapter.progress() reads from run.log):
#   input   python3 weisman_klemp_input.py -> weisman_klemp_input.nc, and the surface
#           pressure it prints is written into [thermo] pbot (it depends on ktot/zsize).
#   init    microhh init weisman_klemp
#   bubble  python3 bubble_theta.py <args>   (edits thl.0000000 in place)
#   run     microhh run weisman_klemp        (iteration table goes to weisman_klemp.out)
set -uo pipefail

CASE=weisman_klemp
BIN=/app/build/microhh
SRC=/app/cases/weisman_klemp

stage() { echo "=== STAGE: $1  $(date -Is) ==="; }
fail()  { echo "=== STAGE: failed rc=$1 ==="; exit "$1"; }

if [ ! -f "${CASE}.ini" ]; then
  echo "ERROR: ${CASE}.ini not found in $(pwd)"; fail 2
fi
cp -n "${SRC}/weisman_klemp_input.py" .
cp -n "${SRC}/bubble_theta.py" .

stage input
pbot=$(python3 weisman_klemp_input.py | tee /dev/stderr | sed -n 's/.*surface pressure in ini file to: *//p' | tr -d '[:space:]')
[ -n "${pbot}" ] || { echo "ERROR: could not read surface pressure from weisman_klemp_input.py"; fail 3; }
sed -i "s/^pbot=.*/pbot=${pbot}/" "${CASE}.ini"
echo "pbot set to ${pbot}"

stage init
stdbuf -oL -eL "${BIN}" init "${CASE}" || fail $?

stage bubble
python3 bubble_theta.py "$@" || fail $?

stage "run start"
stdbuf -oL -eL "${BIN}" run "${CASE}"
rc=$?
if [ "${rc}" -ne 0 ]; then fail "${rc}"; fi
echo "=== STAGE: run finished rc=0  $(date -Is) ==="
