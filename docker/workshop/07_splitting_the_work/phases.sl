#!/bin/bash -e
#SBATCH --job-name      phases
#SBATCH --account       nesi99991
#SBATCH --time          00:15:00
#SBATCH --cpus-per-task 2
#SBATCH --mem           4GB
#SBATCH --gpus-per-node l4:1
#SBATCH --output        phases-%j.out

# Run the same workload twice: once balanced, once dominated by I/O.
#
# Afterwards, run 'seff' on this job. The GPU utilisation figure it reports is
# the same number the script prints as "the GPU was busy for N% of this run" -
# arrived at from the other direction. seff measures it from outside; the
# script measures it from inside. Agreeing with each other is what makes both
# of them worth trusting.

echo "=== A reasonably balanced workflow ==="
python3 profile_phases.py

echo
echo "=== The same computation, but reading and writing much more ==="
python3 profile_phases.py --heavy-io

echo
echo "Run 'seff ${SLURM_JOB_ID}' to see the whole job's GPU utilisation."
echo "It will sit between the two figures above, because it is the average"
echo "over both halves."
