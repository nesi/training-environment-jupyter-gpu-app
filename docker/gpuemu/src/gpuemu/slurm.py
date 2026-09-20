"""A small Slurm-shaped batch scheduler for the training container.

NeSI runs Slurm, so a GPU workshop has to teach ``sbatch``, ``#SBATCH
--gpus-per-node``, ``squeue`` and reading a job's output file. This provides
those commands against a single-node queue inside the session container, with
the same flags, the same output columns and the same job states, so what a
learner practises here transfers to Mahuika unchanged.

It is a teaching scaffold, not Slurm. There is one node, no fair-share, no
backfill, no accounting database, no multi-node anything. Jobs run as
subprocesses of the scheduler, scheduled first-come-first-served whenever the
resources they asked for are free.

The part that matters most for teaching GPUs is the allocation: a job that did
not ask for a GPU is started with ``CUDA_VISIBLE_DEVICES`` set to empty, so it
genuinely cannot see the device, and ``nvidia-smi`` inside it fails exactly as
it would on a real cluster. Forgetting ``--gpus-per-node`` therefore produces
the same confusing symptom here as it does in production, which is precisely
the lesson.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pwd
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .shm import default_state_file

NODE_NAME = "gpunode001"
PARTITION = "gpu"

PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
TIMEOUT = "TIMEOUT"

ACTIVE_STATES = {PENDING, RUNNING}

_STATE_ABBREV = {
    PENDING: "PD",
    RUNNING: "R",
    COMPLETED: "CD",
    FAILED: "F",
    CANCELLED: "CA",
    TIMEOUT: "TO",
}


def slurm_dir() -> Path:
    env = os.environ.get("GPUEMU_SLURM_DIR")
    d = Path(env) if env else default_state_file().parent / "slurm"
    (d / "jobs").mkdir(parents=True, exist_ok=True)
    for p in (d, d / "jobs"):
        try:
            os.chmod(p, 0o1777)
        except OSError:
            pass
    return d


# ---------------------------------------------------------------- resources


def node_cpus() -> int:
    from .daemon import container_cpus

    return max(1, int(container_cpus()))


def node_memory_mb() -> int:
    """Memory the container may use, in MB."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text == "max":
            break
        try:
            val = int(text)
            # cgroup v1 reports a sentinel when unlimited.
            if 0 < val < (1 << 62):
                return max(256, val // (1024 * 1024))
        except ValueError:
            continue
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return max(256, pages * size // (1024 * 1024))
    except (ValueError, OSError):
        return 4096


def node_gpus() -> int:
    from . import spec

    return spec.device_count()


# ---------------------------------------------------------------- job model


@dataclass
class Job:
    job_id: int
    name: str
    user: str
    script: str
    workdir: str
    stdout: str
    stderr: str
    partition: str = PARTITION
    account: str = ""
    state: str = PENDING
    cpus: int = 1
    mem_mb: int = 1024
    gpus: int = 0
    ntasks: int = 1
    time_limit_s: int = 3600
    submit_time: float = field(default_factory=time.time)
    start_time: float = 0.0
    end_time: float = 0.0
    exit_code: int = 0
    pid: int = 0
    gpu_ids: list[int] = field(default_factory=list)
    reason: str = "None"
    qos: str = ""

    # What the job actually used, for seff. CPU time and peak RSS come from
    # wait4() when the job ends (see jobacct.py); the GPU figures have to be
    # sampled while it runs, because utilisation is a rate and there is nothing
    # left to read once the process is gone.
    cpu_seconds: float = 0.0
    max_rss_mb: float = 0.0
    gpu_util_sum: float = 0.0
    gpu_util_samples: int = 0
    gpu_mem_peak_mb: float = 0.0

    @property
    def mean_gpu_util(self) -> float:
        """Mean utilisation over the job's life.

        seff labels this "Peak GPU Utilisation", and for a job with one step -
        which is every job here - that is what the cluster reports too: Slurm's
        gpuutil TRES is the *average* over a step, and "peak" refers to the
        largest across overlapping steps. Reporting a true instantaneous peak
        would be useless anyway, since any job that touches the GPU at all
        touches 100% of it for an instant.
        """
        if not self.gpu_util_samples:
            return 0.0
        return self.gpu_util_sum / self.gpu_util_samples

    @property
    def elapsed(self) -> float:
        if not self.start_time:
            return 0.0
        end = self.end_time or time.time()
        return max(0.0, end - self.start_time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


class JobStore:
    """Job records on disk, one JSON file each, with a lock around mutations."""

    def __init__(self, root: Path | None = None):
        self.root = root or slurm_dir()
        self.jobs_dir = self.root / "jobs"

    def _lock(self):
        fh = open(self.root / ".lock", "a+b")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return fh

    @staticmethod
    def _unlock(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()

    def next_id(self) -> int:
        lock = self._lock()
        try:
            counter = self.root / "next_job_id"
            try:
                current = int(counter.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                current = 1000
            counter.write_text(str(current + 1), encoding="utf-8")
            return current
        finally:
            self._unlock(lock)

    def path(self, job_id: int) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def save(self, job: Job) -> None:
        tmp = self.path(job.job_id).with_suffix(".tmp")
        tmp.write_text(job.to_json(), encoding="utf-8")
        os.replace(tmp, self.path(job.job_id))
        try:
            os.chmod(self.path(job.job_id), 0o666)
        except OSError:
            pass

    def load(self, job_id: int) -> Job | None:
        try:
            data = json.loads(self.path(job_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return Job(**data)

    def all(self) -> list[Job]:
        out = []
        for f in sorted(self.jobs_dir.glob("*.json")):
            try:
                out.append(Job(**json.loads(f.read_text(encoding="utf-8"))))
            except (OSError, ValueError, TypeError):
                continue
        return sorted(out, key=lambda j: j.job_id)


# ---------------------------------------------------------------- parsing


def parse_time_limit(value: str) -> int:
    """Parse Slurm's time formats into seconds.

    Accepts minutes, MM:SS, HH:MM:SS, D-HH, D-HH:MM and D-HH:MM:SS, which is
    what ``--time`` takes.
    """
    value = value.strip()
    if not value:
        raise ValueError("empty time limit")
    days = 0
    if "-" in value:
        day_part, _, value = value.partition("-")
        days = int(day_part)
    parts = value.split(":") if value else ["0"]
    if len(parts) == 1:
        if days:
            # "D-HH"
            return days * 86400 + int(parts[0]) * 3600
        return int(parts[0]) * 60
    if len(parts) == 2:
        if days:
            return days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"invalid time limit {value!r}")


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours}:{minutes:02d}:{secs:02d}"


def parse_memory(value: str) -> int:
    """Parse ``--mem`` into MB. A bare number means MB, as in Slurm."""
    value = value.strip().upper()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)B?", value)
    if not m:
        raise ValueError(f"invalid memory {value!r}")
    num = float(m.group(1))
    return int(num * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[m.group(2)])


def parse_gres(value: str) -> int:
    """``--gres=gpu:2`` or ``--gres=gpu:l4:2`` -> 2."""
    for item in value.split(","):
        bits = item.strip().split(":")
        if bits and bits[0] == "gpu":
            if len(bits) == 1:
                return 1
            return int(bits[-1])
    return 0


def parse_gpus(value: str) -> int:
    """``--gpus-per-node=1`` or ``=l4:1`` -> 1."""
    value = value.strip()
    if ":" in value:
        return int(value.split(":")[-1])
    return int(value)


def _sbatch_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sbatch", add_help=False)
    ap.add_argument("-J", "--job-name", default=None)
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("-e", "--error", default=None)
    ap.add_argument("-t", "--time", default=None)
    ap.add_argument("-c", "--cpus-per-task", type=int, default=None)
    ap.add_argument("-n", "--ntasks", type=int, default=None)
    ap.add_argument("-N", "--nodes", type=int, default=None)
    ap.add_argument("-p", "--partition", default=None)
    ap.add_argument("-A", "--account", default=None)
    ap.add_argument("-D", "--chdir", default=None)
    ap.add_argument("--mem", default=None)
    ap.add_argument("--mem-per-cpu", default=None)
    ap.add_argument("--gres", default=None)
    ap.add_argument("-G", "--gpus", default=None)
    ap.add_argument("--gpus-per-node", default=None)
    ap.add_argument("--gpus-per-task", default=None)
    ap.add_argument("--hint", default=None)
    ap.add_argument("--wrap", default=None)
    # Print just the job ID, for scripts that capture it. Without this a
    # workshop cannot teach JOBID=$(sbatch --parsable ...), which is how you
    # submit several jobs and then compare them.
    ap.add_argument("--parsable", action="store_true")
    # Accepted and recorded rather than acted on. A workshop teaches people to
    # write `--qos debug` for a quick test, and a script that errors out on the
    # flag it was just told to use teaches the opposite.
    ap.add_argument("-q", "--qos", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("-h", "--help", action="store_true")
    return ap


def read_sbatch_directives(script_path: Path) -> list[str]:
    """Pull ``#SBATCH`` options out of a script's leading comment block.

    Slurm stops at the first line that is neither blank, a comment, nor the
    shebang, so directives placed after the first real command are ignored.
    Mirroring that is worth it: a learner who puts #SBATCH lines too low sees
    them ignored here too, and learns why.
    """
    args: list[str] = []
    try:
        lines = script_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return args
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i == 0 and stripped.startswith("#!"):
            continue
        if not stripped:
            continue
        if stripped.startswith("#SBATCH"):
            body = stripped[len("#SBATCH") :].strip()
            if body:
                args.extend(shlex.split(body))
            continue
        if stripped.startswith("#"):
            continue
        break
    return args


def _expand_pattern(pattern: str, job_id: int, name: str, user: str) -> str:
    return (
        pattern.replace("%j", str(job_id))
        .replace("%J", str(job_id))
        .replace("%x", name)
        .replace("%u", user)
        .replace("%N", NODE_NAME)
        .replace("%A", str(job_id))
    )


# ---------------------------------------------------------------- sbatch


def sbatch(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = _sbatch_parser()

    if not argv or "-h" in argv or "--help" in argv:
        ap.print_help()
        return 0 if argv else 1

    cli_args, rest = ap.parse_known_args(argv)

    store = JobStore()
    user = _current_user()
    cwd = Path(cli_args.chdir or os.getcwd())

    if cli_args.wrap is not None:
        script_body = f"#!/bin/bash\n{cli_args.wrap}\n"
        script_path = None
        directives: list[str] = []
    else:
        if not rest:
            print("sbatch: error: no script specified", file=sys.stderr)
            return 1
        script_path = Path(rest[0]).expanduser()
        if not script_path.is_absolute():
            script_path = (cwd / script_path).resolve()
        if not script_path.is_file():
            print(f"sbatch: error: Unable to open file {rest[0]}", file=sys.stderr)
            return 1
        script_body = script_path.read_text(encoding="utf-8", errors="replace")
        directives = read_sbatch_directives(script_path)

    # Command-line options win over #SBATCH directives, as in Slurm.
    file_args, _ = ap.parse_known_args(directives)
    merged = argparse.Namespace()
    for key in vars(cli_args):
        cli_val = getattr(cli_args, key)
        setattr(merged, key, cli_val if cli_val is not None else getattr(file_args, key))

    job_id = store.next_id()
    name = merged.job_name or (script_path.name if script_path else "wrap")

    gpus = 0
    if merged.gres:
        gpus = parse_gres(merged.gres)
    if merged.gpus_per_node:
        gpus = max(gpus, parse_gpus(merged.gpus_per_node))
    if merged.gpus:
        gpus = max(gpus, parse_gpus(str(merged.gpus)))
    if merged.gpus_per_task:
        gpus = max(gpus, parse_gpus(merged.gpus_per_task) * (merged.ntasks or 1))

    cpus = merged.cpus_per_task or 1
    try:
        if merged.mem:
            mem_mb = parse_memory(merged.mem)
        elif merged.mem_per_cpu:
            mem_mb = parse_memory(merged.mem_per_cpu) * cpus
        else:
            mem_mb = 1024 * cpus
    except ValueError as exc:
        print(f"sbatch: error: {exc}", file=sys.stderr)
        return 1

    try:
        time_limit = parse_time_limit(merged.time) if merged.time else 3600
    except ValueError as exc:
        print(f"sbatch: error: invalid --time: {exc}", file=sys.stderr)
        return 1

    # Reject what cannot be run, rather than queueing it forever.
    if gpus > node_gpus():
        print(
            f"sbatch: error: Batch job submission failed: Requested node "
            f"configuration is not available ({gpus} GPUs requested, "
            f"{node_gpus()} on {NODE_NAME})",
            file=sys.stderr,
        )
        return 1
    if cpus > node_cpus():
        print(
            f"sbatch: error: Batch job submission failed: Requested node "
            f"configuration is not available ({cpus} CPUs requested, "
            f"{node_cpus()} on {NODE_NAME})",
            file=sys.stderr,
        )
        return 1

    out_pattern = merged.output or "slurm-%j.out"
    err_pattern = merged.error or out_pattern
    stdout = cwd / _expand_pattern(out_pattern, job_id, name, user)
    stderr = cwd / _expand_pattern(err_pattern, job_id, name, user)

    # Snapshot the script so editing it after submission does not change the
    # queued job, which is how Slurm behaves.
    spool = store.root / "scripts"
    spool.mkdir(parents=True, exist_ok=True)
    spooled = spool / f"{job_id}.sh"
    spooled.write_text(script_body, encoding="utf-8")
    spooled.chmod(0o755)

    job = Job(
        job_id=job_id,
        name=name,
        user=user,
        script=str(spooled),
        workdir=str(cwd),
        stdout=str(stdout),
        stderr=str(stderr),
        partition=merged.partition or PARTITION,
        account=merged.account or "",
        cpus=cpus,
        mem_mb=mem_mb,
        gpus=gpus,
        ntasks=merged.ntasks or 1,
        time_limit_s=time_limit,
        reason="None",
        qos=merged.qos or "",
    )
    store.save(job)
    if merged.parsable:
        print(job_id)
    else:
        print(f"Submitted batch job {job_id}")
    if not _scheduler_running():
        print(
            "sbatch: warning: the scheduler does not appear to be running, so this "
            "job will stay PENDING. Start it with 'gpuemu-ctl start'.",
            file=sys.stderr,
        )
    return 0


def _current_user() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return os.environ.get("USER", "user")


def _scheduler_running() -> bool:
    pid_file = slurm_dir() / "slurmd.pid"
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------- squeue


def squeue(argv: list[str] | None = None) -> int:
    # add_help=False because in Slurm -h means --noheader, not --help. Letting
    # argparse claim -h would break every script that pipes squeue output.
    ap = argparse.ArgumentParser(prog="squeue", add_help=False)
    ap.add_argument("-u", "--user", default=None)
    ap.add_argument("-j", "--jobs", default=None)
    ap.add_argument("-p", "--partition", default=None)
    ap.add_argument("-t", "--states", default=None)
    ap.add_argument("-l", "--long", action="store_true")
    ap.add_argument("-h", "--noheader", action="store_true")
    ap.add_argument("--me", action="store_true")
    ap.add_argument("--help", action="help", help="show this help message and exit")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    jobs = JobStore().all()
    if args.states:
        wanted = {s.strip().upper() for s in args.states.split(",")}
        expanded = set()
        for w in wanted:
            expanded.add(w)
            for full, abbrev in _STATE_ABBREV.items():
                if w == abbrev:
                    expanded.add(full)
        jobs = [j for j in jobs if j.state in expanded]
    else:
        jobs = [j for j in jobs if j.state in ACTIVE_STATES]
    if args.me:
        args.user = _current_user()
    if args.user:
        jobs = [j for j in jobs if j.user == args.user]
    if args.partition:
        jobs = [j for j in jobs if j.partition == args.partition]
    if args.jobs:
        ids = {int(x) for x in args.jobs.split(",") if x.strip().isdigit()}
        jobs = [j for j in jobs if j.job_id in ids]

    if not args.noheader:
        print(
            f"{'JOBID':>18} {'PARTITION':>9} {'NAME':>8} {'USER':>8} "
            f"{'ST':>2} {'TIME':>10} {'NODES':>5} NODELIST(REASON)"
        )
    for j in jobs:
        nodelist = NODE_NAME if j.state == RUNNING else f"({j.reason})"
        print(
            f"{j.job_id:>18} {j.partition:>9} {j.name[:8]:>8} {j.user[:8]:>8} "
            f"{_STATE_ABBREV.get(j.state, j.state[:2]):>2} "
            f"{format_duration(j.elapsed):>10} {1:>5} {nodelist}"
        )
    return 0


# ---------------------------------------------------------------- scancel


def scancel(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scancel", add_help=True)
    ap.add_argument("job_ids", nargs="*")
    ap.add_argument("-u", "--user", default=None)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    store = JobStore()
    targets: list[Job] = []
    if args.user:
        targets = [j for j in store.all() if j.user == args.user and j.state in ACTIVE_STATES]
    for raw in args.job_ids:
        if not raw.isdigit():
            print(f"scancel: error: Invalid job id {raw}", file=sys.stderr)
            return 1
        job = store.load(int(raw))
        if job is None:
            print(f"scancel: error: Kill job error on job id {raw}: Invalid job id specified",
                  file=sys.stderr)
            continue
        targets.append(job)

    for job in targets:
        if job.state not in ACTIVE_STATES:
            continue
        # Mark it cancelled first; the scheduler notices and kills the process.
        job.state = CANCELLED
        job.reason = "JobCancelled"
        if not job.end_time:
            job.end_time = time.time()
        store.save(job)
        if job.pid:
            try:
                os.killpg(os.getpgid(job.pid), signal.SIGTERM)
            except OSError:
                pass
    return 0


# ---------------------------------------------------------------- sinfo/sacct


def sinfo(argv: list[str] | None = None) -> int:
    # As with squeue, -h is --noheader in Slurm, not --help.
    ap = argparse.ArgumentParser(prog="sinfo", add_help=False)
    ap.add_argument("-N", "--Node", action="store_true")
    ap.add_argument("-l", "--long", action="store_true")
    ap.add_argument("-h", "--noheader", action="store_true")
    ap.add_argument("--help", action="help", help="show this help message and exit")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    jobs = JobStore().all()
    running = [j for j in jobs if j.state == RUNNING]
    used_cpus = sum(j.cpus for j in running)
    state = "mix" if running else "idle"
    if used_cpus >= node_cpus():
        state = "alloc"

    if not args.noheader:
        print(f"{'PARTITION':<12}{'AVAIL':<7}{'TIMELIMIT':<11}{'NODES':<7}{'STATE':<7}NODELIST")
    print(f"{PARTITION + '*':<12}{'up':<7}{'7-00:00:00':<11}{1:<7}{state:<7}{NODE_NAME}")
    return 0


def sacct(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sacct", add_help=True)
    ap.add_argument("-j", "--jobs", default=None)
    ap.add_argument("-u", "--user", default=None)
    ap.add_argument("-X", "--allocations", action="store_true")
    # sacct spells it -n, unlike squeue and sinfo, so there is no -h clash here.
    ap.add_argument("-n", "--noheader", action="store_true")
    ap.add_argument("--format", default=None)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    jobs = JobStore().all()
    if args.jobs:
        ids = {int(x) for x in args.jobs.split(",") if x.strip().isdigit()}
        jobs = [j for j in jobs if j.job_id in ids]
    if args.user:
        jobs = [j for j in jobs if j.user == args.user]

    if not args.noheader:
        print(
            f"{'JobID':<12}{'JobName':<12}{'Partition':<11}{'Account':<10}"
            f"{'AllocCPUS':>9} {'State':<12}{'ExitCode':<9}{'Elapsed':<11}"
        )
        print(
            f"{'-' * 11:<12}{'-' * 11:<12}{'-' * 10:<11}{'-' * 9:<10}"
            f"{'-' * 9:>9} {'-' * 11:<12}{'-' * 8:<9}{'-' * 10:<11}"
        )
    for j in jobs:
        print(
            f"{j.job_id:<12}{j.name[:11]:<12}{j.partition:<11}{(j.account or 'default')[:9]:<10}"
            f"{j.cpus:>9} {j.state:<12}{f'{j.exit_code}:0':<9}{format_duration(j.elapsed):<11}"
        )
    return 0


# ------------------------------------------------------------------- seff
#
# Deliberately a line-for-line match of the cluster's own seff
# (github.com/nesi/opt-nesi-bin, nn_seff), down to the column the '%' lands in
# and the "kB/MB/GB" rounding, because the whole value of teaching it here is
# that the output a learner reads in the workshop is the output they will read
# on the cluster. The differences are in where the numbers come from: the real
# one asks sacct for accounting records, this one reads the job file the
# emulator's scheduler wrote.


def _seff_time(seconds: float) -> str:
    """``[D-]HH:MM:SS``, as seff's time2str formats it."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    prefix = "" if days < 1 else f"{days}-"
    return prefix + "{:02}:{:02}:{:02}".format(hours, minutes, secs)


def _seff_bytes(kbytes: float) -> str:
    """``284.46 MB``, as seff's kbytes2str formats it. Input is kibibytes."""
    from math import log

    if kbytes <= 0:
        return "%.2f %sB" % (0.0, "M")
    mul = 1024
    exp = min(int(log(kbytes) / log(mul)), 5)
    prefix = "kMGTPE"[exp]
    return "%.2f %sB" % (kbytes / mul**exp, prefix)


def _seff_pct(label: str, pct: float, detail: str = "") -> str:
    """``Label:  99%  detail``.

    Every label is padded to 22 characters and every percentage is a whole
    number right-aligned in 3, which is what lines the '%' up in column 26
    across rows whose labels differ in length.
    """
    row = f"{label:<22}{pct: >3.0f}%"
    return f"{row}  {detail}".rstrip()


def _device_total_gb() -> float:
    """Capacity of the emulated card, in GB, for the GPU memory line.

    The real seff looks the board size up in a static per-partition table. Here
    we ask the device, because the emulated card's memory is configurable and a
    learner who reads "of 23 GB" while nvidia-smi says 1024MiB has been taught
    to distrust the tool.
    """
    try:
        from .shm import StateReader

        reader = StateReader.try_open()
        if reader is not None:
            try:
                snap = reader.read()
                if snap.gpus:
                    return snap.gpus[0].mem_total / (1024**3)
            finally:
                reader.close()
    except (OSError, ValueError):
        pass
    from .spec import selected_device

    return selected_device().mem_total_mib / 1024


_SEFF_USAGE = """Usage: seff [Options] <JobID>
       Options:
       -M    Cluster
       -h    Help
       -j    JobID
       -v    Version"""


def _seff_one(job: Job, show_cluster: bool) -> None:
    if show_cluster:
        print("Cluster:", os.environ.get("GPUEMU_CLUSTER", "training"))
    print("Job ID:", job.job_id)
    print("State:", job.state)

    # Efficiency figures for a job that is still going would be measured
    # against a wall-time that has not finished happening, so seff declines to
    # give them rather than give misleading ones.
    if job.state in ACTIVE_STATES:
        print(f"Efficiency not available for {job.state} jobs.")
        return

    if job.ntasks > 0:
        print("Tasks:", job.ntasks)
    print("Cores:", job.cpus)
    if job.ntasks > 1:
        print("Nodes: 1")

    wall = job.elapsed
    limit = job.time_limit_s
    print(
        _seff_pct(
            "Job Wall-time:",
            (100 * wall / limit) if limit else 0.0,
            f"{_seff_time(wall)} of {_seff_time(limit)} time limit",
        )
    )

    core_wall = wall * job.cpus
    print(
        _seff_pct(
            "Avg CPU Utilisation:",
            (job.cpu_seconds / core_wall * 100) if core_wall else 0.0,
            f"{_seff_time(job.cpu_seconds)} of {_seff_time(core_wall)} core-walltime",
        )
    )

    req_kb = job.mem_mb * 1024
    used_kb = job.max_rss_mb * 1024
    print(
        _seff_pct(
            "Peak Mem Utilisation:",
            (100 * used_kb / req_kb) if req_kb else 0.0,
            f"{_seff_bytes(used_kb)} of {_seff_bytes(req_kb)}",
        )
    )

    # No GPU requested means no GPU lines, exactly as on the cluster - and that
    # absence is itself the diagnosis. If you expected these two lines and they
    # are not here, the job never had a GPU in the first place.
    if not job.gpus:
        return

    alloc_gb = _device_total_gb() * max(1, len(job.gpu_ids))
    print(_seff_pct("Peak GPU Utilisation:", job.mean_gpu_util))
    print(
        _seff_pct(
            "Peak GPU Memory Util:",
            (100 * job.gpu_mem_peak_mb / (alloc_gb * 1024)) if alloc_gb else 0.0,
            f"{_seff_bytes(job.gpu_mem_peak_mb * 1024)} of {alloc_gb:.0f} GB",
        )
    )


def seff(argv: list[str] | None = None) -> int:
    """Report how much of its allocation a finished job actually used.

    The two GPU lines are the reason this matters for a GPU workshop. A job can
    report 99% CPU efficiency and still have left the GPU completely idle, and
    nothing else a researcher runs after the fact will tell them so.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    # getopt rather than argparse, to accept the same clustered short options
    # the real seff does and to keep '-h' meaning help rather than argparse's
    # auto-generated usage.
    import getopt

    try:
        opts, rest = getopt.getopt(argv, "hvdfj:M:")
    except getopt.GetoptError as exc:
        print(f"seff: {exc}", file=sys.stderr)
        print(_SEFF_USAGE, file=sys.stderr)
        return 1
    flags = dict(opts)

    if "-v" in flags:
        print("Training environment version of seff (emulated GPU)")
        return 1
    if "-h" in flags or not (rest or "-j" in flags):
        print(_SEFF_USAGE)
        return 1

    job_ids = [flags["-j"]] if "-j" in flags else []
    job_ids.extend(rest)

    store = JobStore()
    jobs = []
    for raw in job_ids:
        for part in str(raw).split(","):
            # Accept 1234_5 and 1234.batch, which is what people paste in.
            base = part.strip().split(".")[0].split("_")[0]
            if not base.isdigit():
                continue
            job = store.load(int(base))
            if job is not None:
                jobs.append(job)

    if not jobs:
        print("Job not found.", file=sys.stderr)
        return 2

    for index, job in enumerate(jobs):
        if index:
            print()
        _seff_one(job, show_cluster="-M" in flags)
    return 0


# ----------------------------------------------------------------- svisit


def svisit(argv: list[str] | None = None) -> int:
    """Open a terminal inside a running job, so nvtop can watch its GPU.

    On the cluster this is a wrapper around ``srun --pty --overlap --jobid=``,
    which drops you onto the compute node your job landed on - the only place
    from which you can see the GPU that job is using. Here there is one node
    and you are already on it, so the only real work is finding the job and
    handing you its ``CUDA_VISIBLE_DEVICES``. The habit is what transfers:
    find the running job, visit it, run nvtop.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    test_only = False
    job_id: str | None = None
    command: list[str] = []

    while argv:
        arg = argv[0]
        if arg == "-h":
            print(
                "Usage:\n"
                "  svisit [-t] [[-j] job] [command]\n"
                "    -t        Test instead - just show the command svisit would use.\n"
                "    -j job    A Slurm Job ID. If not given, your most recent running\n"
                "              job is used, as found by 'squeue --me'.\n"
                "    command   Defaults to your shell.\n"
                "\n"
                "Description:\n"
                "  svisit starts a terminal session inside your currently running\n"
                "  Slurm job, which is where you can watch its GPU with nvtop."
            )
            return 0
        if arg == "-t":
            test_only = True
        elif arg == "-j":
            argv.pop(0)
            job_id = argv[0] if argv else None
        elif job_id is None and re.fullmatch(r"[0-9_.]+", arg):
            job_id = arg
        else:
            command = list(argv)
            break
        argv.pop(0)

    store = JobStore()
    if job_id is None:
        running = [j for j in store.all() if j.state == RUNNING and j.user == _current_user()]
        if not running:
            print(
                "No JobID provided and no running job found either. "
                "'svisit -h' for help.",
                file=sys.stderr,
            )
            return 1
        job = running[-1]
    else:
        base = job_id.split(".")[0].split("_")[0]
        job = store.load(int(base)) if base.isdigit() else None
        if job is None:
            print(f"svisit: job {job_id} not found.", file=sys.stderr)
            return 1

    if job.state != RUNNING:
        print(
            f"svisit: job {job.job_id} is {job.state}, not RUNNING, so there is no "
            "job to visit. 'squeue --me' shows what is still going.",
            file=sys.stderr,
        )
        return 1

    if not command:
        command = [os.environ.get("SHELL", "/bin/bash"), "-l"]

    if test_only:
        print(f"srun --pty --overlap --jobid={job.job_id} {' '.join(command)}")
        return 0

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in job.gpu_ids)
    env["SLURM_JOB_ID"] = str(job.job_id)
    env["SLURM_JOBID"] = str(job.job_id)
    print(f"Visiting {NODE_NAME}, where job {job.job_id} ({job.name}) is running.")
    return subprocess.call(command, cwd=job.workdir, env=env)


def scontrol(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "show" and argv[1].startswith("job"):
        store = JobStore()
        jobs = store.all()
        if len(argv) >= 3 and argv[2].isdigit():
            job = store.load(int(argv[2]))
            jobs = [job] if job else []
            if not jobs:
                print(f"slurm_load_jobs error: Invalid job id specified", file=sys.stderr)
                return 1
        for j in jobs:
            print(f"JobId={j.job_id} JobName={j.name}")
            print(f"   UserId={j.user} Account={j.account or 'default'}")
            print(f"   JobState={j.state} Reason={j.reason} ExitCode={j.exit_code}:0")
            print(f"   RunTime={format_duration(j.elapsed)} TimeLimit={format_duration(j.time_limit_s)}")
            print(f"   Partition={j.partition} NodeList={NODE_NAME if j.state == RUNNING else '(null)'}")
            print(f"   NumNodes=1 NumCPUs={j.cpus} NumTasks={j.ntasks} CPUs/Task={j.cpus}")
            print(f"   TRES=cpu={j.cpus},mem={j.mem_mb}M,gres/gpu={j.gpus}")
            print(f"   WorkDir={j.workdir}")
            print(f"   StdOut={j.stdout}")
            print(f"   StdErr={j.stderr}")
            print()
        return 0
    print("scontrol: only 'scontrol show job [id]' is supported by the emulator",
          file=sys.stderr)
    return 1


def srun(argv: list[str] | None = None) -> int:
    """Run a command in the foreground with a GPU allocation.

    Real srun talks to the controller; here it simply applies the same
    environment a batch job would get and waits, which is enough for the
    interactive "try it now" step of a workshop.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = _sbatch_parser()
    args, rest = ap.parse_known_args(argv)
    if not rest:
        print("srun: error: no command given", file=sys.stderr)
        return 1

    gpus = 0
    if args.gres:
        gpus = parse_gres(args.gres)
    if args.gpus_per_node:
        gpus = max(gpus, parse_gpus(args.gpus_per_node))
    if args.gpus:
        gpus = max(gpus, parse_gpus(str(args.gpus)))

    env = dict(os.environ)
    env.update(
        _job_environment(
            job_id=0,
            name=rest[0],
            user=_current_user(),
            cpus=args.cpus_per_task or 1,
            mem_mb=1024,
            gpu_ids=list(range(gpus)),
            ntasks=args.ntasks or 1,
            workdir=os.getcwd(),
        )
    )
    return subprocess.call(rest, env=env)


# ---------------------------------------------------------------- scheduler


def _job_environment(
    job_id: int,
    name: str,
    user: str,
    cpus: int,
    mem_mb: int,
    gpu_ids: list[int],
    ntasks: int,
    workdir: str,
) -> dict[str, str]:
    env = {
        "SLURM_JOB_ID": str(job_id),
        "SLURM_JOBID": str(job_id),
        "SLURM_JOB_NAME": name,
        "SLURM_JOB_USER": user,
        "SLURM_JOB_PARTITION": PARTITION,
        "SLURM_CPUS_PER_TASK": str(cpus),
        "SLURM_CPUS_ON_NODE": str(cpus),
        "SLURM_JOB_CPUS_PER_NODE": str(cpus),
        "SLURM_MEM_PER_NODE": str(mem_mb),
        "SLURM_NTASKS": str(ntasks),
        "SLURM_NNODES": "1",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NODELIST": NODE_NAME,
        "SLURM_JOB_NODELIST": NODE_NAME,
        "SLURM_SUBMIT_DIR": workdir,
        "SLURMD_NODENAME": NODE_NAME,
        # Keep threading libraries inside the job's CPU allocation, which is
        # also what makes the emulated GPU utilisation meaningful.
        "OMP_NUM_THREADS": str(cpus),
    }
    # This is the line that teaches the lesson: no GPU requested, no GPU
    # visible. An empty CUDA_VISIBLE_DEVICES hides the device completely.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
    if gpu_ids:
        env["SLURM_JOB_GPUS"] = ",".join(str(i) for i in gpu_ids)
        env["SLURM_GPUS_ON_NODE"] = str(len(gpu_ids))
        env["GPU_DEVICE_ORDINAL"] = ",".join(str(i) for i in gpu_ids)
    return env


class Scheduler:
    """First-come-first-served over a single node's resources."""

    def __init__(self, verbose: bool = False):
        self.store = JobStore()
        self.verbose = verbose
        self.running: dict[int, subprocess.Popen] = {}
        self.total_cpus = node_cpus()
        self.total_mem = node_memory_mb()
        self.total_gpus = node_gpus()
        self._stop = False
        # GPU samples, kept in memory between flushes so a running job is not
        # rewritten to disk twice a second.
        self._acc: dict[int, dict[str, float]] = {}
        self._reader = None
        self._last_flush = 0.0

    def stop(self, *_):
        self._stop = True

    def run(self) -> None:
        pid_file = self.store.root / "slurmd.pid"
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
        try:
            while not self._stop:
                try:
                    self.tick()
                except Exception as exc:
                    if self.verbose:
                        print(f"slurmd: {exc}", file=sys.stderr)
                time.sleep(0.5)
        finally:
            self._terminate_all()
            try:
                pid_file.unlink()
            except OSError:
                pass

    def tick(self) -> None:
        self._sample_gpu()
        self._reap()
        self._enforce_limits()
        self._launch_eligible()
        self._flush_samples()

    # -- accounting --------------------------------------------------

    def usage_dir(self) -> Path:
        d = self.store.root / "usage"
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(d, 0o1777)
        except OSError:
            pass
        return d

    def _sample_gpu(self) -> None:
        """Record GPU utilisation and memory for each running job.

        Utilisation is a rate, not a total: unlike CPU time there is nothing
        left to read once the job has exited, so it has to be sampled while the
        job is alive. We read the devices the job was allocated, which is what
        Slurm's own GPU accounting does. Reading the whole device rather than
        just this job's claims is correct here because the scheduler hands each
        GPU to one job at a time, so the device *is* the job.
        """
        running = [j for j in self.store.all() if j.state == RUNNING and j.gpu_ids]
        if not running:
            return
        try:
            if self._reader is None:
                from .shm import StateReader

                self._reader = StateReader.try_open()
            if self._reader is None:
                return
            snap = self._reader.read()
        except (OSError, ValueError):
            self._reader = None
            return

        for job in running:
            acc = self._acc.setdefault(
                job.job_id, {"util_sum": 0.0, "n": 0.0, "mem_peak": 0.0}
            )
            utils: list[int] = []
            mem_mb = 0.0
            for idx in job.gpu_ids:
                if idx >= len(snap.gpus):
                    continue
                gpu = snap.gpus[idx]
                utils.append(gpu.util_gpu)
                mem_mb += gpu.mem_used / (1024 * 1024)
            if not utils:
                continue
            acc["util_sum"] += sum(utils) / len(utils)
            acc["n"] += 1
            acc["mem_peak"] = max(acc["mem_peak"], mem_mb)

    def _apply_samples(self, job: Job) -> None:
        acc = self._acc.get(job.job_id)
        if not acc or not acc["n"]:
            return
        job.gpu_util_sum = acc["util_sum"]
        job.gpu_util_samples = int(acc["n"])
        job.gpu_mem_peak_mb = acc["mem_peak"]

    def _flush_samples(self) -> None:
        """Persist samples periodically so seff works on a job that is still running."""
        now = time.time()
        if now - self._last_flush < 2.0:
            return
        self._last_flush = now
        for job_id in list(self._acc):
            job = self.store.load(job_id)
            if job is None or job.state != RUNNING:
                continue
            self._apply_samples(job)
            self.store.save(job)

    def _read_usage(self, job: Job) -> None:
        """Take the kernel's CPU and memory accounting from the jobacct sidecar."""
        path = self.usage_dir() / f"{job.job_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        try:
            job.cpu_seconds = float(data.get("cpu_seconds", 0.0))
            job.max_rss_mb = float(data.get("max_rss_mb", 0.0))
        except (TypeError, ValueError):
            pass
        try:
            path.unlink()
        except OSError:
            pass

    def _finalise(self, job: Job) -> None:
        self._apply_samples(job)
        self._acc.pop(job.job_id, None)
        self._read_usage(job)

    # -- phases ------------------------------------------------------

    def _reap(self) -> None:
        for job_id, proc in list(self.running.items()):
            code = proc.poll()
            if code is None:
                continue
            del self.running[job_id]
            job = self.store.load(job_id)
            if job is None:
                self._acc.pop(job_id, None)
                continue
            self._finalise(job)
            job.end_time = time.time()
            job.exit_code = code if code >= 0 else 128 - code
            if job.state == CANCELLED:
                pass  # scancel already recorded the outcome
            elif code == 0:
                job.state = COMPLETED
                job.reason = "None"
            else:
                job.state = FAILED
                job.reason = "NonZeroExitCode"
            job.pid = 0
            self.store.save(job)
            if self.verbose:
                print(f"slurmd: job {job_id} -> {job.state} ({code})", file=sys.stderr)

    def _enforce_limits(self) -> None:
        now = time.time()
        for job in self.store.all():
            if job.state == CANCELLED and job.job_id in self.running:
                self._kill(job, CANCELLED, "JobCancelled")
            elif job.state == RUNNING and job.time_limit_s:
                if now - job.start_time > job.time_limit_s:
                    self._kill(job, TIMEOUT, "TimeLimit")

    def _kill(self, job: Job, state: str, reason: str) -> None:
        proc = self.running.pop(job.job_id, None)
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
        self._finalise(job)
        job.state = state
        job.reason = reason
        job.end_time = time.time()
        job.pid = 0
        self.store.save(job)

    def _used(self) -> tuple[int, int, set[int]]:
        cpus = mem = 0
        gpus: set[int] = set()
        for job in self.store.all():
            if job.state != RUNNING:
                continue
            cpus += job.cpus
            mem += job.mem_mb
            gpus.update(job.gpu_ids)
        return cpus, mem, gpus

    def _launch_eligible(self) -> None:
        pending = [j for j in self.store.all() if j.state == PENDING]
        if not pending:
            return
        used_cpus, used_mem, used_gpus = self._used()

        for job in pending:
            free_gpus = [i for i in range(self.total_gpus) if i not in used_gpus]
            if job.cpus + used_cpus > self.total_cpus:
                self._mark_reason(job, "Resources")
                continue
            if job.mem_mb + used_mem > self.total_mem:
                self._mark_reason(job, "Resources")
                continue
            if job.gpus > len(free_gpus):
                self._mark_reason(job, "Resources")
                continue

            allocated = free_gpus[: job.gpus]
            self._start(job, allocated)
            used_cpus += job.cpus
            used_mem += job.mem_mb
            used_gpus.update(allocated)

    def _mark_reason(self, job: Job, reason: str) -> None:
        if job.reason != reason:
            job.reason = reason
            self.store.save(job)

    def _start(self, job: Job, gpu_ids: list[int]) -> None:
        env = dict(os.environ)
        env.update(
            _job_environment(
                job_id=job.job_id,
                name=job.name,
                user=job.user,
                cpus=job.cpus,
                mem_mb=job.mem_mb,
                gpu_ids=gpu_ids,
                ntasks=job.ntasks,
                workdir=job.workdir,
            )
        )

        Path(job.stdout).parent.mkdir(parents=True, exist_ok=True)
        out = open(job.stdout, "ab", buffering=0)
        err = out if job.stderr == job.stdout else open(job.stderr, "ab", buffering=0)

        # Run the script under the accounting wrapper rather than bash directly,
        # so seff can report the CPU time and peak memory the kernel measured
        # instead of whatever happened to be alive at the last poll.
        usage_file = self.usage_dir() / f"{job.job_id}.json"
        try:
            usage_file.unlink()
        except OSError:
            pass

        # Invoked by path rather than `-m gpuemu.jobacct`: the scheduler may be
        # running from a source checkout where the package is on sys.path but
        # not on PYTHONPATH, and a child process would not inherit that.
        # jobacct imports nothing but the standard library, so this works
        # whether gpuemu is installed or not.
        wrapper = str(Path(__file__).resolve().parent / "jobacct.py")

        try:
            proc = subprocess.Popen(
                [sys.executable, wrapper, str(usage_file), job.script],
                cwd=job.workdir,
                env=env,
                stdout=out,
                stderr=err,
                stdin=subprocess.DEVNULL,
                # Own process group, so a timeout or scancel takes the whole
                # tree down rather than leaving orphaned children holding GPU
                # memory.
                start_new_session=True,
            )
        except OSError as exc:
            job.state = FAILED
            job.reason = f"LaunchFailed:{exc.errno}"
            job.end_time = time.time()
            job.exit_code = 1
            self.store.save(job)
            return

        self.running[job.job_id] = proc
        job.state = RUNNING
        job.reason = "None"
        job.pid = proc.pid
        job.gpu_ids = gpu_ids
        job.start_time = time.time()
        self.store.save(job)
        if self.verbose:
            print(
                f"slurmd: job {job.job_id} started pid={proc.pid} gpus={gpu_ids}",
                file=sys.stderr,
            )

    def _terminate_all(self) -> None:
        for job_id, proc in list(self.running.items()):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                pass


def slurmd(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpuemu-slurmd", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    sched = Scheduler(verbose=args.verbose)
    signal.signal(signal.SIGTERM, sched.stop)
    signal.signal(signal.SIGINT, sched.stop)
    if args.verbose:
        print(
            f"gpuemu-slurmd: node {NODE_NAME} cpus={sched.total_cpus} "
            f"mem={sched.total_mem}MB gpus={sched.total_gpus}",
            file=sys.stderr,
        )
    sched.run()
    return 0
