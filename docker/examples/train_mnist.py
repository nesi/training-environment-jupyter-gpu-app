#!/usr/bin/env python3
"""A small CUDA-style training script, unchanged from what you would write for
a real GPU.

Nothing here is emulator-specific. The device is selected the usual way, the
model and batches are moved with ``.to(device)``, and memory is reported with
``torch.cuda.memory_allocated``. On this training environment that all works
against the emulated L4, with the arithmetic running on the CPU.

Because the data is synthetic, this needs no download and no dataset on disk,
which keeps it usable in a session with no outbound network.
"""

from __future__ import annotations

import argparse
import time

# Must come before torch is used, so the CUDA entry points are in place.
import gpuemu.torch_shim  # noqa: F401
import torch
import torch.nn as nn


class SmallNet(nn.Module):
    """A deliberately over-wide MLP, so its memory footprint is visible in nvtop."""

    def __init__(self, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--batches", type=int, default=60, help="batches per epoch")
    ap.add_argument("--hidden", type=int, default=2048)
    args = ap.parse_args()

    # The standard incantation. On a machine with no GPU allocated it falls
    # back to the CPU, which is exactly what it would do on the cluster.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"  compute capability {cap[0]}.{cap[1]}")
        total = torch.cuda.get_device_properties(0).total_memory
        print(f"  {total / 1024**3:.1f} GiB device memory")
    print()

    torch.manual_seed(0)
    model = SmallNet(args.hidden).to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        print(f"Model on device: {torch.cuda.memory_allocated() / 1024**2:.0f} MiB\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        running = 0.0

        for _ in range(args.batches):
            # Synthetic batch, generated straight onto the device.
            images = torch.randn(args.batch_size, 1, 28, 28, device=device)
            labels = torch.randint(0, 10, (args.batch_size,), device=device)

            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimiser.step()
            running += loss.item()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - started
        line = (
            f"epoch {epoch}/{args.epochs}  "
            f"loss {running / args.batches:.4f}  "
            f"{elapsed:.1f}s"
        )
        if torch.cuda.is_available():
            line += f"  peak {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB"
        print(line)

    print("\nDone.")
    # Worth repeating at the end of every run in this environment.
    print(
        "Reminder: this device is emulated. The timings above are CPU timings "
        "and say nothing about GPU performance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
