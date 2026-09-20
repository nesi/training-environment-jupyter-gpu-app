#!/usr/bin/env python3
"""RAM and VRAM are two separate pools. Asking for one does not get you the other.

This trips people up constantly, because both are called "memory" and both are
measured in gigabytes:

  RAM   ordinary system memory, attached to the CPU.
        You ask for it with  #SBATCH --mem 8GB
        Running out of it gets your job killed by Slurm.

  VRAM  memory on the GPU board itself.
        You do NOT ask for it directly. You get whatever the GPU you asked
        for happens to have - an L4 comes with 24 GB, an A100 with 80 GB.
        Running out of it raises an out-of-memory error inside your program.

So "my job needs 40 GB" is an ambiguous statement, and the two halves of the
answer are found in different places: --mem for the first, and your choice of
GPU for the second.

This script allocates in each pool in turn and shows which counter moves.
"""

import subprocess
import time

import torch


def vram_used_mb():
    """What the driver says is allocated on the GPU right now."""
    # The emulator republishes device state a few times a second, so read it
    # after a short pause rather than the instant an allocation returns.
    # Otherwise the number you print is the one from just before you allocated.
    time.sleep(0.5)
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def host_ram_mb():
    """This process's resident set size - ordinary RAM, in MB."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return 0


if not torch.cuda.is_available():
    raise SystemExit("No GPU visible. Add '#SBATCH --gpus-per-node l4:1'.")

total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM on this card: {total_vram:.1f} GB")
print()
print(f"{'':<34}{'RAM (MB)':>12}{'VRAM (MB)':>12}")
print("-" * 58)
print(f"{'at the start':<34}{host_ram_mb():>12}{vram_used_mb():>12}")

# --- allocate 256 MB of ordinary RAM ------------------------------------
# A numpy-style host array. The GPU knows nothing about this.
host_block = torch.zeros(64 * 1024 * 1024, dtype=torch.float32)  # 256 MB
print(f"{'after 256 MB in RAM':<34}{host_ram_mb():>12}{vram_used_mb():>12}")
print("   ^ RAM went up. VRAM did not move: the GPU cannot see this data.")
print()

# --- move it to the GPU --------------------------------------------------
device_block = host_block.to("cuda")
print(f"{'after moving it to the GPU':<34}{host_ram_mb():>12}{vram_used_mb():>12}")
print("   ^ VRAM went up. The data now exists in the GPU's own memory.")
print()

# --- free the host copy --------------------------------------------------
del host_block
print(f"{'after freeing the RAM copy':<34}{host_ram_mb():>12}{vram_used_mb():>12}")
print("   ^ VRAM is unchanged. Freeing RAM does nothing for VRAM, and the")
print("     reverse is also true. They are not connected.")
print()
print("     Notice the RAM column did not drop either. Python holds on to")
print("     memory it has finished with, to reuse it. That is why a job's")
print("     'Peak Mem Utilisation' in seff reflects the most it ever needed,")
print("     not what it was using at the end.")
print()

del device_block
torch.cuda.empty_cache()

print("Two practical consequences:")
print()
print("  * If your program dies with 'CUDA out of memory', asking Slurm for")
print("    more --mem will not help. You need a GPU with more VRAM.")
print()
print("  * If your job is killed by Slurm for exceeding its memory limit,")
print("    a bigger GPU will not help. You need more --mem.")
