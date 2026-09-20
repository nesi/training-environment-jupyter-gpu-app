#!/usr/bin/env python3
"""A realistic shape of GPU job: prepare data on the CPU, compute on the GPU.

Almost every GPU workload alternates like this. Images are decoded and resized,
sequences are tokenised, tables are filtered - all on the CPU - and then a
batch is handed to the GPU. While the CPU prepares batch N+1, the GPU has
nothing to do with batch N but wait.

That is why "how many CPUs should I ask for?" is a GPU question. The GPU can
only be as busy as the CPUs can keep it, and a GPU job with too few CPUs is a
job that paid for a GPU and then starved it.

The preparation here is real CPU work spread over however many cores the job
was given, so the number of cores genuinely changes the answer:

    python3 pipeline.py                 # uses $SLURM_CPUS_PER_TASK
    python3 pipeline.py --workers 1
    python3 pipeline.py --batches 40
"""

import argparse
import math
import multiprocessing as mp
import os

import torch

from gpuemu.phases import cpu_phase, gpu_phase, report


def prepare_one(seed):
    """Stand-in for decoding and cleaning one record. Deliberately CPU-bound.

    Sized so that preparing a batch costs noticeably more than computing on
    it, which is the situation this chapter is about. Plenty of real pipelines
    are this lopsided and their owners have no idea.
    """
    total = 0.0
    for i in range(seed % 97 + 1, 600000):
        total += math.sqrt(i) * math.sin(i)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        help="CPU processes preparing data (default: the job's cores)",
    )
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--records", type=int, default=12, help="records per batch")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU visible to this process. Add '#SBATCH --gpus-per-node l4:1'."
        )

    # On real hardware the GPU phase costs the same no matter how many CPU
    # cores the job was given - the GPU does that work, and it does not care.
    # In this environment the "GPU" arithmetic actually runs on the CPU, so
    # without this line it would speed up with more cores too, and the exercise
    # would not measure what it claims to measure.
    torch.set_num_threads(1)

    print(f"Preparing data on {args.workers} CPU worker(s)")
    print(f"Computing on {torch.cuda.get_device_name(0)}")
    print(f"{args.batches} batches of {args.records} records\n")

    weights = torch.randn(1024, 1024, device="cuda")

    with mp.Pool(args.workers) as pool:
        for batch in range(args.batches):
            # ---- CPU: get the next batch ready -------------------------
            # This is the part that scales with --cpus-per-task.
            with cpu_phase("prepare data"):
                pool.map(prepare_one, range(batch * args.records,
                                            (batch + 1) * args.records))

            # ---- GPU: do the actual computation ------------------------
            # This does not get faster with more CPUs. It cannot start until
            # the phase above has finished.
            with gpu_phase("compute"):
                x = torch.randn(1024, 1024, device="cuda")
                for _ in range(4):
                    x = torch.tanh(x @ weights)

            print(f"  batch {batch + 1}/{args.batches} done")

    report()
    print("Now run 'seff <JOBID>' and compare the GPU utilisation line")
    print("against the percentage above.")


if __name__ == "__main__":
    main()
