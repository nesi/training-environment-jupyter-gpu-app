#!/bin/bash -e
#SBATCH --job-name      vram
#SBATCH --account       nesi99991
#SBATCH --time          00:10:00
#SBATCH --cpus-per-task 2
#SBATCH --mem           4GB
#SBATCH --gpus-per-node l4:1
#SBATCH --output        vram-%j.out

# Both memory demonstrations, in a job.
#
# Note the deliberate mismatch in this script's own request: 4 GB of --mem,
# and a GPU whose VRAM is whatever an L4 has. Those two numbers have nothing
# to do with each other, which is the point of the first script.

echo "=== Two separate pools ==="
python3 two_pools.py

echo
echo "=== Sizing and exhausting VRAM ==="
python3 how_much_vram.py

echo
echo "Afterwards, run 'seff ${SLURM_JOB_ID}'."
echo "Peak Mem Utilisation is the RAM line. Peak GPU Memory Util is the VRAM"
echo "line. This job should be low on the first and high on the second."
