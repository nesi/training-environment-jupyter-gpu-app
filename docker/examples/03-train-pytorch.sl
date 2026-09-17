#!/bin/bash -e
#SBATCH --job-name=train
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=train-%j.out

# A PyTorch training job written exactly as it would be for a real GPU.
#
# While it runs, open another terminal and watch it with `nvtop`. You should
# see memory claimed as the model and batches are placed on the device, and
# utilisation rise and fall with each epoch.

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}, GPUs=${SLURM_JOB_GPUS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

python3 "${SLURM_SUBMIT_DIR}/train_mnist.py" --epochs 3 --batch-size 128
