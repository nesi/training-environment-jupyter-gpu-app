"""Run a batch script and record what it actually consumed.

``seff`` is only worth teaching if its numbers are true, and the obvious way to
get them - poll ``/proc`` from the scheduler - cannot produce true numbers. A
job that runs for four seconds is gone before the next sample, and ``/proc``
only ever shows processes that are still alive, so every short job would report
roughly zero CPU time and every workshop exercise would teach the wrong lesson
about efficiency.

``wait4()`` asks the kernel instead. It returns the accumulated CPU time and
peak resident memory of the child *and of every descendant the child reaped*,
which is the same accounting real Slurm reads out of the job's cgroup. A job
that burns five CPU-seconds reports five CPU-seconds whether it ran for four
seconds or four hours.

The wrapper sits between the scheduler and ``/bin/bash``, so it also has to be
transparent to everything else the scheduler does: it keeps the child in its
own process group's foreground, forwards the signals ``scancel`` and the
time-limit enforcer send, and exits with the child's status (128+N when the
child died on a signal) so the recorded exit code is unchanged.
"""

from __future__ import annotations

import json
import os
import resource
import signal
import sys
from pathlib import Path

# Signals the scheduler may send us that are really meant for the job.
_FORWARD = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)


def _write_usage(path: Path, payload: dict) -> None:
    """Write the sidecar atomically, and never fail the job over it.

    If this cannot be written the job still ran; seff will simply report the
    CPU line as unavailable rather than the whole thing falling over.
    """
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
        os.chmod(path, 0o666)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: gpuemu-jobacct <usage-file> <script>", file=sys.stderr)
        return 2
    usage_file, script = Path(argv[0]), argv[1]

    pid = os.fork()
    if pid == 0:
        try:
            os.execv("/bin/bash", ["/bin/bash", script])
        except OSError as exc:
            print(f"gpuemu-jobacct: cannot exec {script}: {exc}", file=sys.stderr)
        os._exit(127)

    def forward(signum, _frame):
        try:
            os.kill(pid, signum)
        except OSError:
            pass

    for sig in _FORWARD:
        try:
            signal.signal(sig, forward)
        except (OSError, ValueError):
            pass

    # EINTR is expected here: a forwarded signal interrupts the wait, and the
    # child has usually not finished dying yet, so we go back to waiting.
    while True:
        try:
            _, status, usage = os.wait4(pid, 0)
            break
        except InterruptedError:
            continue
        except ChildProcessError:
            status, usage = 0, None
            break

    if usage is not None:
        # ru_maxrss is kilobytes on Linux and bytes on macOS. The container is
        # Linux, but the test suite also runs on a developer's Mac, and a
        # thousand-fold error in the memory line is exactly the kind of thing
        # that looks plausible enough to go unnoticed.
        rss_mb = usage.ru_maxrss / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
        _write_usage(
            usage_file,
            {
                "cpu_seconds": usage.ru_utime + usage.ru_stime,
                "max_rss_mb": rss_mb,
            },
        )

    if os.WIFSIGNALED(status):
        # Re-raise on ourselves so the scheduler sees the same death the job
        # died of, rather than a plain exit code that hides a SIGKILL.
        sig = os.WTERMSIG(status)
        signal.signal(sig, signal.SIG_DFL)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.kill(os.getpid(), sig)
    return os.WEXITSTATUS(status)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
