#!/usr/bin/env python3
"""Work out how much VRAM you need, then watch what happens when you run out.

You cannot ask Slurm for VRAM. You choose a GPU, and the VRAM comes with it. So
"which GPU should I ask for?" is, more often than not, really the question
"how much VRAM does my work need?" - and that one you can answer on paper
before you queue anything.

The arithmetic is simple. Every number your program holds on the GPU costs
bytes:

    float64  (double precision)   8 bytes
    float32  (single precision)   4 bytes
    float16  / bfloat16           2 bytes

Multiply by how many numbers you hold at once. For training a neural network
the rule of thumb is roughly 4x the size of the model itself, because you hold
the weights, their gradients, and two optimiser states for each.

The second half of this script allocates larger and larger blocks until the
GPU refuses, so you can see the error you will actually get.
"""

import torch

if not torch.cuda.is_available():
    raise SystemExit("No GPU visible. Add '#SBATCH --gpus-per-node l4:1'.")

props = torch.cuda.get_device_properties(0)
total_gb = props.total_memory / 1024**3

print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"VRAM: {total_gb:.2f} GB")

# ---------------------------------------------------------------- on paper

print()
print("How much VRAM would a model need?")
print()
print(f"  {'parameters':>12} {'fp32 weights':>14} {'fp32 training':>15} {'fp16 weights':>14}")
print("  " + "-" * 58)
for params in (1e6, 1e7, 1e8, 1e9, 7e9, 7e10):
    weights32 = params * 4 / 1024**3
    training32 = weights32 * 4  # weights + gradients + two optimiser states
    weights16 = params * 2 / 1024**3
    label = f"{params / 1e9:.0f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"
    print(f"  {label:>12} {weights32:>13.2f}G {training32:>14.2f}G {weights16:>13.2f}G")

print()
print("  Read the row for your model, then pick the smallest GPU it fits in.")
print("  A 7B model needs ~28 GB just to train in fp32: too big for an L4")
print("  (24 GB), comfortable on an A100 (80 GB).")

# ------------------------------------------------------------- in practice

print()
print("Now filling this GPU until it refuses.")
print()

blocks = []
mb = 0
step_mb = max(32, int(total_gb * 1024 / 16))

try:
    while True:
        # 1 MB = 262144 float32 values.
        blocks.append(torch.zeros(step_mb * 262144, dtype=torch.float32, device="cuda"))
        mb += step_mb
        print(f"  allocated {mb:>6} MB  ({mb / 1024:.2f} GB)")
except torch.cuda.OutOfMemoryError as exc:
    print()
    print("  Out of memory. This is the error, in full:")
    print()
    for line in str(exc).splitlines():
        print(f"    {line}")
    print()
    print("  Learn to read it. The useful parts are:")
    print("    'Tried to allocate'  - what your program wanted next")
    print("    'capacity'           - what the card has in total")
    print("    'already allocated'  - what you were holding at the time")
    print()
    print("  What to do about it, in the order worth trying:")
    print("    1. Use a smaller batch size. Halving the batch usually halves")
    print("       the activation memory, and costs you very little.")
    print("    2. Use a lower precision, if your work tolerates it (fp32 ->")
    print("       fp16 halves the memory).")
    print("    3. Ask for a GPU with more VRAM.")
    print()
    print("  Adding --mem to your Slurm script will not help. That is RAM.")
finally:
    blocks.clear()
    torch.cuda.empty_cache()
