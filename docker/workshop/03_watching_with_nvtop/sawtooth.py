#!/usr/bin/env python3
"""A job with an obvious shape, so you can learn to read nvtop against it.

Real jobs do not announce what they are doing. This one does: it alternates
between a stretch of GPU work and a stretch of doing nothing, and prints each
change as it happens. Watch it in nvtop and the two traces tell you two
different stories.

  GPU utilisation  swings between high and zero, following the phases.
  GPU memory       rises once at the start and then stays flat.

That difference is the thing to take away. Memory shows what you have
*reserved*; utilisation shows what you are *using*. A job holding 20 GB of
VRAM at 0% utilisation is a job doing nothing, and nvidia-smi run at the wrong
moment will happily show you a number that suggests otherwise.

Run for about four minutes by default:

    python3 sawtooth.py
    python3 sawtooth.py --minutes 10
"""

import argparse
import time

import torch

from gpuemu.phases import cpu_phase, gpu_phase, report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--minutes", type=float, default=4.0, help="how long to run")
parser.add_argument("--busy", type=float, default=20.0, help="seconds of GPU work")
parser.add_argument("--idle", type=float, default=10.0, help="seconds of waiting")
args = parser.parse_args()

if not torch.cuda.is_available():
    raise SystemExit(
        "No GPU visible to this process. Did the job ask for one?\n"
        "Add '#SBATCH --gpus-per-node l4:1' and try again."
    )

# Reserve some device memory up front and hold it for the whole run. This is
# what a model's weights do: allocated once, resident until the job ends,
# whether or not anything is being computed.
held = torch.ones(4096, 4096, device="cuda")
print(f"Holding {held.numel() * held.element_size() / 1024**2:.0f} MB of VRAM "
      "for the whole run.\n")

deadline = time.time() + args.minutes * 60
cycle = 0

while time.time() < deadline:
    cycle += 1

    print(f"[{time.strftime('%H:%M:%S')}] cycle {cycle}: GPU BUSY for {args.busy:.0f}s")
    with gpu_phase("compute"):
        end = time.time() + args.busy
        while time.time() < end:
            held = held @ held.T
            held /= held.norm()

    print(f"[{time.strftime('%H:%M:%S')}] cycle {cycle}: GPU IDLE for {args.idle:.0f}s "
          "(memory stays allocated)")
    with cpu_phase("waiting"):
        time.sleep(args.idle)

print("\nDone.")
report()
