#!/bin/bash -e
#SBATCH --job-name      starved
#SBATCH --account       nesi99991
#SBATCH --time          00:20:00
#SBATCH --cpus-per-task 1
#SBATCH --mem           4GB
#SBATCH --gpus-per-node l4:1
#SBATCH --output        starved-%j.out

# One CPU core feeding a GPU.
#
# The job will finish. Nothing will look broken. Run seff afterwards and read
# the two GPU lines:
#
#   seff <JOBID>
#
# Then run fed.sl, which is this same script with --cpus-per-task 4, and
# compare. Same work, same GPU, same result - different efficiency.

python3 pipeline.py
