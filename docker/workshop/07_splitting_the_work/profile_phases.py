#!/usr/bin/env python3
"""Find out which parts of your work belong on the GPU - and which never will.

A GPU is very good at doing the same arithmetic to thousands of numbers at
once. It is no better than a CPU, and often worse, at everything else: reading
files, decompressing, parsing text, branching on conditions, talking to a
database, writing results out.

Almost every real workflow is a mixture. The useful question is not "can this
run on a GPU?" but "what fraction of my runtime could a GPU help with?" -
because that fraction is a hard ceiling on what any GPU can do for you. If
70% of your job is reading and writing files, then even an infinitely fast GPU
makes your job only 30% faster.

That ceiling has a name - Amdahl's law - and you do not need the maths, just
the habit: measure the phases before you optimise, and before you ask for a
bigger GPU.

This script walks through a workload shaped like a real one and prints where
the time actually went.

    python3 profile_phases.py
    python3 profile_phases.py --heavy-io     # what a badly balanced job does
"""

import argparse
import math
import time

import torch

from gpuemu.phases import cpu_phase, gpu_phase, report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument(
    "--heavy-io",
    action="store_true",
    help="simulate a workflow dominated by reading and writing",
)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise SystemExit("No GPU visible. Add '#SBATCH --gpus-per-node l4:1'.")

io_cost = 3.0 if args.heavy_io else 0.6

print("A workflow in four stages. Watch which one dominates.\n")

weights = torch.randn(1024, 1024, device="cuda")

for round_number in range(args.rounds):
    print(f"round {round_number + 1}/{args.rounds}")

    # ---- Stage 1: read the data in. Pure I/O. A GPU cannot help. --------
    with cpu_phase("1. read input"):
        time.sleep(io_cost)

    # ---- Stage 2: clean and reshape it. CPU work, sometimes parallel. ---
    with cpu_phase("2. prepare"):
        total = 0.0
        for i in range(1, 400_000):
            total += math.sqrt(i)

    # ---- Stage 3: the actual computation. This is the GPU's part. -------
    # Big, regular, identical arithmetic over a lot of numbers at once -
    # exactly the shape a GPU is built for.
    with gpu_phase("3. compute"):
        x = torch.randn(1024, 1024, device="cuda")
        for _ in range(20):
            x = torch.tanh(x @ weights)

    # ---- Stage 4: write the results. I/O again. -------------------------
    with cpu_phase("4. write output"):
        time.sleep(io_cost / 2)

report()

print("What to do with this:")
print()
print("  If stage 3 is most of the time")
print("    The GPU is the bottleneck and a faster or larger GPU will help.")
print("    This is the case where upgrading hardware is the right answer.")
print()
print("  If stages 1 and 4 dominate")
print("    You are limited by storage, not compute. Read larger chunks at a")
print("    time, use a faster filesystem, or overlap reading with computing.")
print("    A better GPU changes nothing.")
print()
print("  If stage 2 dominates")
print("    Ask for more CPUs and do the preparation in parallel - see")
print("    chapter 6 - or move that step onto the GPU too if it is arithmetic.")
print()
print("Try running this again with --heavy-io to see the second case.")
