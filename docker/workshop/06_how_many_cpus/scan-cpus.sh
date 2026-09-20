#!/bin/bash
#
# Submit the same GPU job at several CPU counts, so you can compare them.
#
# There is no formula for "how many CPUs does a GPU job need". It depends
# entirely on how much work your code does per batch before the GPU sees it,
# and the only reliable way to find out is to measure your own job. This runs
# the measurement.
#
#   ./scan-cpus.sh
#   squeue --me                # wait for all of them to finish
#   ./compare.sh               # or run seff on each by hand
#
# A reasonable starting point before you have measured anything is
# --cpus-per-task 2, which is also what the cluster documentation suggests.
# Two is rarely badly wrong. One is often badly wrong.
#
# This scans 1, 2 and 4 because that is what your training session has. On the
# cluster, where a GPU node has 64 or 168 cores, keep going - 8, 16 - until
# the GPU utilisation stops climbing.

set -e

JOBIDS=()
for CPUS in 1 2 4; do
    JOBID=$(sbatch --parsable \
        --job-name "cpus-${CPUS}" \
        --cpus-per-task "${CPUS}" \
        --mem 4GB \
        --gpus-per-node l4:1 \
        --time 00:20:00 \
        --output "cpus-${CPUS}-%j.out" \
        --wrap "python3 ../04_reading_seff/pipeline.py --workers ${CPUS}")
    echo "submitted ${JOBID} with ${CPUS} CPU(s)"
    JOBIDS+=("${JOBID}")
done

printf '%s\n' "${JOBIDS[@]}" > .jobids
echo
echo "Job IDs saved to .jobids. When they have finished:"
echo "  ./compare.sh"
