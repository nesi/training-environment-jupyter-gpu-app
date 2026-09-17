# Shown when a learner opens a terminal in the session.
#
# The emulator is convincing on purpose, which makes saying so out loud a
# requirement rather than a nicety. Someone who benchmarks here and reports the
# numbers as GPU results has been misled by us, not by their own carelessness.

if [ -n "${PS1:-}" ] && [ -z "${GPUEMU_BANNER_SHOWN:-}" ]; then
    export GPUEMU_BANNER_SHOWN=1
    cat <<'BANNER'

  ┌────────────────────────────────────────────────────────────────────┐
  │  This session has an EMULATED NVIDIA L4. There is no real GPU.     │
  │                                                                    │
  │  nvidia-smi, nvtop, sbatch and torch.cuda all work, and memory     │
  │  limits are enforced - but the arithmetic runs on the CPU, so      │
  │  timings here mean nothing about GPU performance.                  │
  │                                                                    │
  │  nvidia-smi          state of the emulated card                    │
  │  nvtop               live view (q to quit)                         │
  │  sbatch job.sl       submit a job   ·   squeue    check the queue  │
  │  gpuemu-ctl status   check the emulator itself                     │
  │                                                                    │
  │  Notebooks: ~/gpu-training/                                        │
  └────────────────────────────────────────────────────────────────────┘

BANNER
fi
