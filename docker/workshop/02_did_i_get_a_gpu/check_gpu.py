#!/usr/bin/env python3
"""The first thing to run inside any GPU job.

``nvidia-smi`` answers one question: is there a GPU here that I am allowed to
use? That is worth knowing, and it is all it is good for as a check. It tells
you nothing about whether *your program* found the device, and most GPU
libraries will fall back to the CPU without saying so.

So ask the library, not the driver. This script asks both and prints them side
by side, which is enough to separate the three cases you can be in:

  1. No GPU allocated          -> fix your Slurm request
  2. GPU allocated, not seen   -> fix your software (wrong build, wrong module)
  3. GPU allocated and seen    -> good, now find out if you are using it well

Copy this into your own workflow. Four lines at the top of a job that print
what the job can see cost nothing and save entire runs.
"""

import os
import subprocess
import sys


def heading(text):
    print()
    print(text)
    print("-" * len(text))


heading("1. What Slurm gave this job")

visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if visible is None:
    print("  CUDA_VISIBLE_DEVICES is not set at all.")
    print("  You are probably not inside a Slurm job.")
elif visible == "":
    print("  CUDA_VISIBLE_DEVICES is set but EMPTY.")
    print("  This job asked for no GPUs, so it has none. Add to your script:")
    print("      #SBATCH --gpus-per-node l4:1")
else:
    print(f"  CUDA_VISIBLE_DEVICES = {visible}")
    print(f"  This job may use {len(visible.split(','))} GPU(s).")

print(f"  CPUs allocated: {os.environ.get('SLURM_CPUS_PER_TASK', 'unknown')}")
print(f"  RAM allocated:  {os.environ.get('SLURM_MEM_PER_NODE', 'unknown')} MB")


heading("2. What the driver reports (nvidia-smi)")

try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode == 0 and out.stdout.strip():
        for line in out.stdout.strip().splitlines():
            print(f"  {line}")
    else:
        print("  nvidia-smi found no GPU for this job.")
        print(f"  {(out.stderr or out.stdout).strip().splitlines()[0] if (out.stderr or out.stdout).strip() else ''}")
except FileNotFoundError:
    print("  nvidia-smi is not installed here.")
except subprocess.TimeoutExpired:
    print("  nvidia-smi did not respond.")


heading("3. What your software reports (PyTorch)")

# This is the check that actually matters. A job can have a perfectly good GPU
# sitting idle beside it because the software was never built to use one.
try:
    import torch
except ImportError:
    print("  PyTorch is not installed; skipping.")
    sys.exit(0)

print(f"  torch version:         {torch.__version__}")
print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print()
    print("  PyTorch cannot see a GPU. If section 1 showed one, then the")
    print("  problem is your software, not your Slurm request - a CPU-only")
    print("  build of PyTorch is the usual cause.")
    sys.exit(0)

print(f"  device count:          {torch.cuda.device_count()}")
print(f"  device name:           {torch.cuda.get_device_name(0)}")

major, minor = torch.cuda.get_device_capability(0)
print(f"  compute capability:    {major}.{minor}")

total = torch.cuda.get_device_properties(0).total_memory
print(f"  device memory (VRAM):  {total / 1024**3:.1f} GB")

# Prove it end to end, rather than trusting a flag.
x = torch.ones(1000, 1000, device="cuda")
result = float((x @ x)[0, 0])
print(f"  test calculation:      {result:.0f} (expected 1000)")

print()
print("  This job has a GPU and your software is using it.")
