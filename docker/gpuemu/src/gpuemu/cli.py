"""gpuemu-ctl and gpuemu-burn.

``gpuemu-ctl`` starts, stops and inspects the two background processes the
emulator needs: the device daemon that maintains telemetry, and the scheduler
that runs batch jobs. The session startup script calls ``gpuemu-ctl start``, so
learners mostly need ``status`` when something looks wrong.

``gpuemu-burn`` puts a known load on the device. It exists so a trainer can
demonstrate nvtop reacting to something without first writing a training script,
and so learners have a reference point for what "busy" looks like.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import client, spec
from .shm import StateReader, default_state_file
from .slurm import JobStore, slurm_dir


def _pid_file(name: str) -> Path:
    d = default_state_file().parent
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.pid"


def _read_pid(name: str) -> int | None:
    try:
        pid = int(_pid_file(name).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _spawn(name: str, argv: list[str]) -> int:
    """Start a background daemon and record its pid."""
    existing = _read_pid(name)
    if existing:
        return existing
    log = default_state_file().parent / f"{name}.log"
    with open(log, "ab", buffering=0) as fh:
        proc = subprocess.Popen(
            argv,
            stdout=fh,
            stderr=fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    _pid_file(name).write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def _stop(name: str) -> bool:
    pid = _read_pid(name)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    for _ in range(40):
        time.sleep(0.05)
        try:
            os.kill(pid, 0)
        except OSError:
            break
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        _pid_file(name).unlink()
    except OSError:
        pass
    return True


def cmd_start(args) -> int:
    gpud = _spawn("gpuemud", [sys.executable, "-m", "gpuemu.daemon"])
    # Wait for the first frame so that an nvidia-smi immediately after
    # 'gpuemu-ctl start' succeeds rather than racing the daemon.
    deadline = time.time() + 5
    while time.time() < deadline:
        if StateReader.try_open() is not None:
            break
        time.sleep(0.05)
    sched = _spawn("gpuemu-slurmd", [sys.executable, "-c",
                                     "from gpuemu.slurm import slurmd; raise SystemExit(slurmd([]))"])
    print(f"gpuemud running (pid {gpud})")
    print(f"gpuemu-slurmd running (pid {sched})")
    return 0


def cmd_stop(args) -> int:
    for name in ("gpuemu-slurmd", "gpuemud"):
        print(f"{name}: {'stopped' if _stop(name) else 'not running'}")
    return 0


def cmd_restart(args) -> int:
    cmd_stop(args)
    return cmd_start(args)


def cmd_status(args) -> int:
    dev = spec.selected_device()
    print(f"Emulated device : {dev.name} x{spec.device_count()}")
    print(f"State file      : {default_state_file()}")
    print(f"Claims dir      : {client.claims_dir()}")
    print(f"Slurm dir       : {slurm_dir()}")
    print()

    for name in ("gpuemud", "gpuemu-slurmd"):
        pid = _read_pid(name)
        print(f"{name:<16}: {'running (pid ' + str(pid) + ')' if pid else 'NOT RUNNING'}")

    reader = StateReader.try_open()
    if reader is None:
        print("\nNo state file yet. Run 'gpuemu-ctl start'.")
        return 1
    try:
        snap = reader.read()
    except ValueError as exc:
        print(f"\nState file unreadable: {exc}")
        return 1
    finally:
        reader.close()

    age = time.time() - snap.timestamp
    print(f"\nLast update     : {age:.1f}s ago" + ("  (stale!)" if age > 5 else ""))
    for i, g in enumerate(snap.gpus):
        print(
            f"  GPU {i}: {g.name}  util={g.util_gpu}%  "
            f"mem={g.mem_used // (1024 * 1024)}/{g.mem_total // (1024 * 1024)}MiB  "
            f"temp={g.temp}C  power={g.power_mw / 1000:.0f}W  procs={len(g.processes)}"
        )

    jobs = JobStore().all()
    active = [j for j in jobs if j.state in {"PENDING", "RUNNING"}]
    print(f"\nJobs            : {len(active)} active, {len(jobs)} total")
    return 0


def cmd_reset(args) -> int:
    """Clear job history and stale claims, leaving the daemons running."""
    removed = 0
    for f in client.claims_dir().glob("*.json"):
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    jobs_removed = 0
    if args.jobs:
        for f in (slurm_dir() / "jobs").glob("*.json"):
            try:
                f.unlink()
                jobs_removed += 1
            except OSError:
                pass
    print(f"Cleared {removed} claim(s), {jobs_removed} job record(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpuemu-ctl", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start", help="start the device daemon and scheduler")
    sub.add_parser("stop", help="stop both daemons")
    sub.add_parser("restart", help="stop then start")
    sub.add_parser("status", help="show device and job state")
    reset = sub.add_parser("reset", help="clear stale claims")
    reset.add_argument("--jobs", action="store_true", help="also clear job history")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    return {
        "start": cmd_start,
        "stop": cmd_stop,
        "restart": cmd_restart,
        "status": cmd_status,
        "reset": cmd_reset,
    }[args.cmd](args)


# ---------------------------------------------------------------- burn


def burn(argv: list[str] | None = None) -> int:
    """Put a known, steady load on the emulated device."""
    ap = argparse.ArgumentParser(
        prog="gpuemu-burn",
        description="Generate GPU load for demonstrations.",
    )
    ap.add_argument("-t", "--time", type=float, default=60.0, help="seconds to run")
    ap.add_argument("-m", "--memory", default="2GiB", help="device memory to hold")
    ap.add_argument(
        "-u", "--util", default="auto",
        help="target utilisation percent, or 'auto' to follow real CPU use",
    )
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None, help="CPU threads to spin")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    visible = client.visible_devices()
    if visible is not None and args.device not in visible:
        print(
            f"gpuemu-burn: GPU {args.device} is not visible to this process "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}).\n"
            "If this is a batch job, did you ask for a GPU with --gpus-per-node=1?",
            file=sys.stderr,
        )
        return 1

    util: str | float = args.util
    if args.util != "auto":
        try:
            util = float(args.util)
        except ValueError:
            print(f"gpuemu-burn: invalid --util {args.util!r}", file=sys.stderr)
            return 1

    try:
        claim = client.Claim(
            device=args.device, memory=args.memory, util=util, name="gpuemu-burn"
        ).open()
    except client.OutOfMemoryError as exc:
        print(f"gpuemu-burn: {exc}", file=sys.stderr)
        return 1

    n_workers = args.workers
    if n_workers is None:
        # Inside a batch job, stay within the CPUs the job actually asked for.
        # Spinning up a worker per container core would let a 2-core job use
        # the whole node, which is both wrong and a bad thing to demonstrate.
        allocated = os.environ.get("SLURM_CPUS_PER_TASK")
        if allocated and allocated.isdigit():
            n_workers = max(1, int(allocated))
        else:
            from .daemon import container_cpus

            n_workers = max(1, int(container_cpus()))

    print(
        f"Burning GPU {args.device} for {args.time:.0f}s "
        f"holding {args.memory} across {n_workers} worker(s). Watch with nvtop."
    )

    stop_at = time.time() + args.time
    try:
        if util == "auto":
            _spin(n_workers, stop_at)
        else:
            # An explicit target still needs the process to look alive, but
            # there is no reason to melt the CPU doing it.
            while time.time() < stop_at:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        claim.close()
    return 0


def _burn_worker(stop_at: float) -> None:
    x = 0.0
    while time.time() < stop_at:
        for i in range(20000):
            x += math.sin(i) * math.cos(i)


def _spin(n_workers: int, stop_at: float) -> None:
    """Burn CPU across real cores so the 'auto' utilisation model sees load.

    Processes rather than threads: the GIL would pin threads to a single core,
    and the daemon would then report a fraction of the utilisation the demo is
    meant to show. Children count towards the claim because the daemon sums the
    whole process tree.
    """
    import multiprocessing

    procs = [
        multiprocessing.Process(target=_burn_worker, args=(stop_at,), daemon=True)
        for _ in range(n_workers)
    ]
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
