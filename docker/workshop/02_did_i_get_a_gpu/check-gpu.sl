#!/bin/bash -e
#SBATCH --job-name      check-gpu
#SBATCH --account       nesi99991
#SBATCH --time          00:05:00
#SBATCH --cpus-per-task 2
#SBATCH --mem           2GB
#SBATCH --gpus-per-node l4:1
#SBATCH --output        check-gpu-%j.out

# Run the three-part check inside a real job.
#
# Try it twice: once as it is, and once with the --gpus-per-node line commented
# out. The two output files side by side are the clearest picture you will get
# of what "my job did not use the GPU" actually looks like.

python3 check_gpu.py
