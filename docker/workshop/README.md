# Introduction to GPUs — exercises

Everything here runs from the **terminal**. In JupyterLab, open one with
**File → New → Terminal**, then:

```bash
cd ~/gpu-training
ls
```

The chapters are numbered in the order the workshop covers them. Each folder
holds the scripts for that chapter, and the lesson text for all of them is at
<https://nesi.github.io/reannz-intro-gpu-workshop/>.

| Folder | Chapter |
|---|---|
| `01_requesting_a_gpu` | Asking Slurm for a GPU |
| `02_did_i_get_a_gpu` | Checking you actually got one |
| `03_watching_with_nvtop` | Watching a job while it runs |
| `04_reading_seff` | Reading the report after it finishes |
| `05_ram_and_vram` | Two kinds of memory, and how much to ask for |
| `06_how_many_cpus` | How many CPU cores a GPU job needs |
| `07_splitting_the_work` | Which parts of your work belong on a GPU |
| `08_precision` | Single and double precision, and what they cost |
| `09_choosing_a_gpu` | Putting it together: which card to ask for |

## The commands you will use

| Command | What it is for |
|---|---|
| `sbatch script.sl` | Submit a job |
| `squeue --me` | See your jobs. `ST` is the state: `PD` pending, `R` running |
| `svisit <jobid>` | Open a terminal inside a running job |
| `nvtop` | Watch GPU utilisation and memory live. `q` to quit |
| `nvidia-smi` | Check a GPU is there. A snapshot, not a monitor |
| `seff <jobid>` | See what a finished job actually used |
| `scancel <jobid>` | Stop a job |

## About this environment

**There is no GPU in this training environment.** It emulates one, so that a
workshop about using GPUs can run on hardware that has none.

What that means in practice:

* `nvidia-smi`, `nvtop`, `seff`, `sbatch` and PyTorch's CUDA API all behave
  the way they do on the cluster. Utilisation, memory, processes, out-of-memory
  errors and job accounting are all real.
* The arithmetic runs on the CPU. **No timing you measure here says anything
  about GPU performance.** Nothing in this workshop asks you to time anything,
  and if you find yourself comparing two runs by their wall-clock time, stop —
  that is the one question this environment cannot answer.
* The emulated card reports **1 GB of VRAM**, not the 24 GB a real L4 has.
  That is deliberate: it makes running out of memory something you can do in
  a few seconds with a tensor that costs the session almost nothing.

Everything you learn about *reading* these tools transfers unchanged. Nothing
you learn about speed does.
