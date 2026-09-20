#!/bin/bash -e
#SBATCH --job-name      hello-gpu
#SBATCH --account       nesi99991
#SBATCH --time          00:05:00
#SBATCH --cpus-per-task 2
#SBATCH --mem           2GB
#SBATCH --gpus-per-node l4:1
#SBATCH --output        hello-gpu-%j.out

# The smallest complete GPU job. Four things are being asked for, and all four
# matter:
#
#   --cpus-per-task 2   CPU cores. A GPU job still needs CPUs to feed it.
#   --mem 2GB           ordinary RAM. This is NOT the GPU's memory.
#   --gpus-per-node     the GPU itself: <type>:<how many>.
#   --time              how long before Slurm stops the job.
#
# Leave out --gpus-per-node and the job still runs - it just runs without a
# GPU. That is the single most common GPU mistake, and 02-forgot-the-gpu.sl
# is what it looks like.

echo "Job ${SLURM_JOB_ID} is running on ${SLURMD_NODENAME}"
echo "Slurm gave this job GPU number: '${CUDA_VISIBLE_DEVICES}'"
echo

# Every GPU library on the machine reads CUDA_VISIBLE_DEVICES to decide which
# devices it is allowed to touch. Slurm sets it for you when you ask for a GPU.
# If it is empty, you did not get one.

nvidia-smi
