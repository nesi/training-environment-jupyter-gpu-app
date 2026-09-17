# REANNZ training environment GPU JupyterLab app

JupyterLab app for running GPU workshops on the NeSI training environment,
**on infrastructure that has no GPUs**.

The session presents an emulated NVIDIA L4. `nvidia-smi` and `nvtop` show a
device with live telemetry, `sbatch`/`squeue`/`scancel` queue and run jobs that
request GPUs, PyTorch's CUDA API works, and device memory is genuinely limited
so oversubscribing it raises a real out-of-memory error.

All computation runs on the CPU. There is no GPU anywhere.

---

## What this can and cannot teach

This exists because most of what a GPU workshop covers does not actually need
the silicon. The parts that do cannot be faked, and pretending otherwise would
be worse than useless.

| Taught faithfully | Cannot be taught here |
|---|---|
| Requesting a GPU from the scheduler, and checking you got one | Anything about speed |
| Reading `nvidia-smi`: memory, utilisation, processes, power | Kernel-level performance, occupancy, profiling |
| Watching a job live with `nvtop` | Tensor cores, mixed-precision speedups |
| Diagnosing "my job ran but ignored the GPU" | Multi-GPU scaling behaviour, NCCL |
| Device memory budgeting, and recovering from OOM | Custom CUDA C++ extensions, Triton, `torch.compile` |
| Writing correct CUDA kernels — threads, blocks, shared memory, races | Whether one kernel is faster than another |
| Structuring a batch job script around a GPU allocation | GPU-specific numerical behaviour (TF32, etc.) |

**The single rule to give learners: no timing measured in this environment
means anything about GPU performance.** The session banner, the welcome
notebook and the example training script all say so, in those words. Please do
not remove those notices — an emulator this convincing is only safe to use if
it is loud about what it is.

---

## How the emulation works

Three pieces, in `docker/gpuemu/`.

### 1. A stand-in NVML library

`nvml/libnvidia_ml.c` builds `libnvidia-ml.so.1` and installs it on the loader
path. This works because GPU monitoring tools never link NVML at build time:
nvtop `dlopen`s `libnvidia-ml.so.1` and resolves every entry point with
`dlsym`. A shared object exporting the right symbols with the right signatures
is, from the caller's side, indistinguishable from the real driver — and needs
no kernel module, no `/dev/nvidia*`, and no privileges.

It implements the ~45 NVML functions nvtop, `nvidia-smi` and `pynvml` actually
call. Functions for hardware the L4 does not have (NVLink, MIG) return
`NVML_ERROR_NOT_SUPPORTED`, which is what a real L4 returns, so callers hide
those panels exactly as they would on the real card.

### 2. A device simulator

`gpuemu/daemon.py` (`gpuemud`) maintains the device's state and publishes it
through a small mmap'd file that the C shim reads. It models the behaviour that
makes the readings worth watching:

- utilisation ramps and decays rather than snapping between 0 and 100
- clocks boost under load, drop to idle, and trim near the thermal limit
- power rises faster than linearly with utilisation, capped at 72 W
- temperature lags power with a ~45 s time constant, so a finished job still
  looks warm — which is the relationship a learner should come away with
- the fan reports `N/A`, because the L4 is passively cooled

Utilisation is derived, by default, from **the CPU time the claiming process
and its children actually burn**. That is what connects the display to the
learner's own code: when their training loop runs, the emulated GPU is busy;
when it stops, it idles. Nothing has to be annotated for this to work.

### 3. Memory accounting that bites

A process claims device memory by dropping a small JSON file in a claims
directory; the daemon aggregates the live ones and reaps any whose owner has
died, so a killed notebook kernel frees its memory the way a real driver would
clean up a dead context.

Claims are checked against the card's real capacity, so asking for more than
23 GB fails with a genuine out-of-memory error. Hitting OOM and learning to
read it is a large part of what a GPU workshop is for, and an emulator with
infinite memory would quietly teach the opposite.

### PyTorch

`gpuemu/torch_shim.py` makes `torch.cuda` work: `is_available()` is true,
`.cuda()` and `.to("cuda")` place tensors, `get_device_name()` returns
`NVIDIA L4`, and every tensor placed on "cuda" is counted against the emulated
card and reported to NVML. Exceeding it raises `torch.cuda.OutOfMemoryError`
with the message PyTorch would really produce.

Tensors live in host memory, so `tensor.device` reports `cpu`. Set
`GPUEMU_TORCH_SPOOF_DEVICE=1` to have it report `cuda:0` instead, for
demonstrations where the illusion matters more than the accuracy.

### CUDA kernels

The image sets `NUMBA_ENABLE_CUDASIM=1`, so Numba's `@cuda.jit` kernels
execute in the interpreter with faithful CUDA semantics: `threadIdx`,
`blockIdx`, grid-stride loops, shared memory, `syncthreads` and race conditions
all behave as they would on hardware. This is the one part of the environment
where learners write GPU code and get genuinely correct GPU behaviour back,
including the bugs — remove a `syncthreads` and the answer goes wrong for the
right reason. It is slow, so keep problem sizes small.

See `docker/examples/04-cuda-kernel.py`.

### Batch jobs

`gpuemu/slurm.py` provides `sbatch`, `squeue`, `scancel`, `sinfo`, `sacct`,
`scontrol show job` and `srun` over a single-node queue inside the container,
with Slurm's flags, output columns and job states. `#SBATCH` directives are
parsed from the script's leading comment block and stop at the first real
command, as Slurm does.

The detail that earns its place: **a job that did not request a GPU is started
with `CUDA_VISIBLE_DEVICES` empty**, so it genuinely cannot see the device and
`nvidia-smi` inside it fails. Forgetting `--gpus-per-node` therefore produces
the same baffling symptom here as it does on Mahuika, which is exactly the
lesson. `docker/examples/02-forgot-the-gpu.sl` is that mistake, on purpose.

It is a teaching scaffold, not Slurm: one node, first-come-first-served, no
fair-share, no backfill, no accounting database.

---

## Repository layout

```
form.yml               session options: CPUs, memory, GPU model and count, wall time
submit.yml.erb         k8s pod spec; passes GPUEMU_DEVICE / GPUEMU_GPUS through
template/script.sh.erb starts the emulator, copies notebooks, launches JupyterLab
docker/Dockerfile      the session image
docker/gpuemu/         the emulator (see docker/gpuemu/README.md)
docker/notebooks/      workshop notebooks, copied to ~/gpu-training/
docker/examples/       job scripts and training scripts, copied to ~/gpu-training/examples/
```

## Session options

`gpu_model` chooses which card is presented — `l4` (24 GB), `a100` (40 GB) or
`h100` (80 GB). It changes the reported name, memory, clocks and power
envelope, and nothing else; no configuration is any more or less real than the
others.

`gpu_vram` overrides how much device memory the card reports. **Setting this to
1 GB is the cheapest way to teach memory pressure**: a learner hits a genuine
out-of-memory error with a tensor that costs the session almost nothing, so
batch-sizing and OOM-recovery exercises work without needing 24 GB of real RAM
to fill. Leave it at the card default for workshops that do not cover memory.

`gpu_count` presents 1, 2 or 4 devices, which is how to teach device selection
and `CUDA_VISIBLE_DEVICES`. The scheduler allocates them to jobs
independently.

CPU and memory default to 4 cores and 8 GB. Do not reduce the CPU allocation
much below that: emulated GPU utilisation is derived from real CPU use, so on
one or two cores every trivial job pins the meter at 100% and the reading stops
teaching anything.

## Building and testing

The image build runs the emulator's test suite, loads the NVML shim and
enumerates the device through it, so a broken emulator fails the build rather
than surfacing in front of a workshop.

Locally:

```bash
cd docker/gpuemu
python3 -m pytest tests -q          # 44 tests: ABI, accounting, telemetry, scheduler
```

The ABI test compiles `nvml/abi_dump.c` and compares the C compiler's struct
layout against the Python `struct` format. If those two drift apart nothing
raises on its own — the shim just reads the wrong offsets and reports
convincing nonsense — so that test is the thing keeping the whole mechanism
honest.

Full image:

```bash
docker build -t gpu-app docker/
docker run --rm -it gpu-app bash -lc 'gpuemu-ctl start && nvidia-smi && nvtop'
```

Pushing to this repo triggers the container build in
`.github/workflows/build_container.yml`.

## Deploying it

The app is deployed through the [training-environment][te] repo. The `gpu`
branch there is configured for this app with one trainer and one training user;
see [the deployment tutorial][deploy].

[te]: https://github.com/nesi/training-environment
[deploy]: https://nesi.github.io/training-environment/tutorials/deployment-on-nesi/

## Configuration reference

Read by the emulator at startup; set in `submit.yml.erb` or the Dockerfile.

| Variable | Default | Effect |
|---|---|---|
| `GPUEMU_DEVICE` | `l4` | Which card to emulate (`l4`, `a100`, `h100`) |
| `GPUEMU_MEM_TOTAL` | card default | Override device memory, e.g. `1GiB` |
| `GPUEMU_GPUS` | `1` | How many devices to present (max 8) |
| `GPUEMU_UTIL_GAIN` | `1.0` | Scales derived utilisation; raise if jobs look idle |
| `GPUEMU_AUTOCLAIM` | `1` | Claim the device when a process imports torch or numba.cuda |
| `GPUEMU_TORCH_SPOOF_DEVICE` | unset | Make `tensor.device` report `cuda:0` |
| `GPUEMU_STATE_FILE` | `/run/gpuemu/state.bin` | Where device state lives |
| `NUMBA_ENABLE_CUDASIM` | `1` | Run `@cuda.jit` kernels in the simulator |

## Troubleshooting

**`nvidia-smi` says it cannot talk to the driver.** The daemon is not running.
`gpuemu-ctl status`, then `gpuemu-ctl start`.

**A job stays `PENDING` forever.** The scheduler is not running — the same
fix. `sbatch` warns about this at submission time.

**nvtop shows no devices.** The shim is not on the loader path. Check
`ldconfig -p | grep nvidia-ml` and that `/etc/ld.so.conf.d/gpuemu.conf` exists.

**Utilisation sits at 100% for trivial work, or never rises.** It is derived
from CPU use against the container's CPU allocation. Give the session more
cores, or tune `GPUEMU_UTIL_GAIN`.

**Something needs real CUDA.** Custom CUDA extensions, Triton kernels,
`torch.compile` with a CUDA backend and bitsandbytes cannot work here and will
fail where they try to load device code. There is no workaround short of real
hardware; design the workshop around it.
