#!/bin/bash
#SBATCH --job-name      forgot-the-gpu
#SBATCH --account       nesi99991
#SBATCH --time          00:05:00
#SBATCH --cpus-per-task 2
#SBATCH --mem           2GB
#SBATCH --output        forgot-the-gpu-%j.out

# This is hello-gpu.sl with one line deleted: --gpus-per-node.
#
# Note there is no -e on the shebang line, unlike every other script here.
# That is on purpose: without it the script stops at the first failure, and
# the failure is the thing worth looking at.
#
# Submit it and watch what happens. Slurm accepts the job, does not warn you,
# and starts it - sooner than the GPU job, in fact, because any node will take
# a job that needs no GPU. Then it runs without one.
#
# On a real cluster this is how people lose a week. Most software does not
# stop when it cannot find a GPU; it quietly carries on using the CPU, thirty
# times slower, and nothing in the output says why. The two clues are below:
# an empty CUDA_VISIBLE_DEVICES, and an nvidia-smi that finds nothing.
#
# Afterwards, run 'seff' on this job and on the hello-gpu one. This job has no
# GPU lines at all. That absence is how you tell, weeks later, which of your
# jobs were really using the GPU you thought you asked for.

echo "Job ${SLURM_JOB_ID} is running on ${SLURMD_NODENAME}"
echo "Slurm gave this job GPU number: '${CUDA_VISIBLE_DEVICES}'"
echo

# An empty CUDA_VISIBLE_DEVICES means "you may use no GPUs at all", so every
# GPU library on the node will quietly decide there is no hardware and fall
# back to the CPU.

nvidia-smi
