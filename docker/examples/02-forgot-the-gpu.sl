#!/bin/bash
#SBATCH --job-name=no-gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=no-gpu-%j.out

# Deliberately broken: there is no --gpus-per-node line.
#
# This is the single most common mistake in GPU batch jobs, and the symptom is
# confusing the first time you meet it: the job runs, exits successfully, and
# quietly does all its work on the CPU. Run it, read the output, then compare
# with 01-hello-gpu.sl.

echo "Allocated GPUs: ${SLURM_JOB_GPUS:-none}"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
echo

echo "nvidia-smi says:"
nvidia-smi || echo "  ...no device, because none was requested."

echo
python3 - <<'PY'
import gpuemu

if gpuemu.is_available():
    print("Python can see a GPU.")
else:
    print("Python cannot see a GPU: CUDA_VISIBLE_DEVICES is empty.")
    print("Add '#SBATCH --gpus-per-node=1' and resubmit.")
PY
