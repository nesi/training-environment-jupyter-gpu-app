#!/bin/bash
#
# Show seff for every job scan-cpus.sh submitted, one after another.
#
# Read down the "Peak GPU Utilisation" column first. It should climb as the
# core count rises and then stop climbing - that flattening-off point is the
# number of CPUs your job actually needs. Asking for more than that does not
# make the GPU any busier; it only lowers your CPU efficiency and makes your
# job harder for the scheduler to place.
#
# Then read "Avg CPU Utilisation". Once that starts falling away while GPU
# utilisation stays flat, you have gone past the useful point.

set -e

if [[ ! -f .jobids ]]; then
    echo "No .jobids file. Run ./scan-cpus.sh first." >&2
    exit 1
fi

while read -r JOBID; do
    [[ -z "${JOBID}" ]] && continue
    echo "============================================================"
    seff "${JOBID}"
    echo
done < .jobids
