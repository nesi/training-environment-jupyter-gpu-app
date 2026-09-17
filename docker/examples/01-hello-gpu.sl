#!/bin/bash -e
#SBATCH --job-name=hello-gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=hello-gpu-%j.out

# The smallest useful GPU job: ask for a device, then prove you got one.
#
# Submit it with:
#     sbatch 01-hello-gpu.sl
# then watch it with `squeue` and read the output file it names.

echo "Job ${SLURM_JOB_ID} running on ${SLURMD_NODENAME}"
echo "CPUs allocated  : ${SLURM_CPUS_PER_TASK}"
# Note these are two different things, and the names invite confusion:
#   SLURM_GPUS_ON_NODE is how many GPUs you got
#   SLURM_JOB_GPUS     is which ones, by index - so "0" means device 0, not none
echo "GPUs allocated  : ${SLURM_GPUS_ON_NODE:-0}"
echo "GPU device IDs  : ${SLURM_JOB_GPUS:-none}"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
echo

# Every GPU job should start by confirming the device is actually there.
nvidia-smi

echo
echo "Now the same question from Python:"
python3 - <<'PY'
import gpuemu

print(f"  device visible : {gpuemu.is_available()}")
print(f"  device count   : {gpuemu.device_count()}")
for i, gpu in enumerate(gpuemu.snapshot().gpus):
    print(f"  GPU {i}: {gpu.name}, {gpu.mem_total // 1024**2} MiB")
PY
