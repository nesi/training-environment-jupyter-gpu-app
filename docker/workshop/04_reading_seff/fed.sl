#!/bin/bash -e
#SBATCH --job-name      fed
#SBATCH --account       nesi99991
#SBATCH --time          00:20:00
#SBATCH --cpus-per-task 4
#SBATCH --mem           4GB
#SBATCH --gpus-per-node l4:1
#SBATCH --output        fed-%j.out

# The same job as starved.sl with four cores instead of one.
#
# After both have finished:
#
#   seff <starved JOBID>
#   seff <fed JOBID>
#
# What to compare, and what each line is telling you:
#
#   Peak GPU Utilisation   how much of the GPU you actually used. This is the
#                          number that says whether the GPU was worth asking
#                          for.
#   Avg CPU Utilisation    how much of the CPU you asked for you used. Low
#                          here means you asked for too many cores.
#   Peak Mem Utilisation   RAM, not VRAM. Low means you asked for too much.
#   Peak GPU Memory Util   VRAM. Low means a smaller GPU would have done.
#
# The two efficiencies pull against each other, and that is the real skill:
# adding cores raises GPU utilisation until the GPU is saturated, and every
# core you add after that just lowers your CPU efficiency instead.

python3 pipeline.py
