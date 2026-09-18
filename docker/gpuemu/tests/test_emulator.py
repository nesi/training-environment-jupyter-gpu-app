"""Behaviour tests for the emulator: accounting, telemetry and the scheduler."""

from __future__ import annotations

import os
import time

import pytest

from gpuemu import client, slurm, spec
from gpuemu.daemon import Daemon


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point every component at a throwaway state directory."""
    monkeypatch.setenv("GPUEMU_STATE_FILE", str(tmp_path / "state.bin"))
    monkeypatch.setenv("GPUEMU_CLAIMS_DIR", str(tmp_path / "claims"))
    monkeypatch.setenv("GPUEMU_SLURM_DIR", str(tmp_path / "slurm"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    yield


# ---------------------------------------------------------------- sizes


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1024", 1024),
        ("1KiB", 1024),
        ("1MiB", 1024**2),
        ("2GiB", 2 * 1024**3),
        ("1GB", 1000**3),
        ("1.5GiB", int(1.5 * 1024**3)),
    ],
)
def test_parse_size(text, expected):
    assert client.parse_size(text) == expected


def test_parse_size_rejects_nonsense():
    with pytest.raises(ValueError):
        client.parse_size("12 bananas")


# ---------------------------------------------------------------- memory


def test_claim_appears_in_device_state():
    daemon = Daemon()
    with client.gpu(memory="2GiB", util=50):
        daemon.tick(0.2)
        snap = _read(daemon)
        gpu = snap.gpus[0]
        assert gpu.mem_used >= 2 * 1024**3
        assert len(gpu.processes) == 1
        assert gpu.processes[0].pid == os.getpid()


def test_memory_is_released_when_claim_closes():
    daemon = Daemon()
    claim = client.gpu(memory="2GiB", util=10).open()
    daemon.tick(0.2)
    assert _read(daemon).gpus[0].mem_used >= 2 * 1024**3

    claim.close()
    daemon.tick(0.2)
    # Only the baseline megabyte should remain.
    assert _read(daemon).gpus[0].mem_used < 16 * 1024**2


def test_oversubscribing_memory_raises_oom():
    dev = spec.selected_device()
    too_much = dev.mem_total_bytes * 2
    with pytest.raises(client.OutOfMemoryError) as excinfo:
        client.gpu(memory=too_much).open()
    msg = str(excinfo.value)
    assert "out of memory" in msg.lower()
    assert "free" in msg.lower()


def test_second_claim_cannot_exceed_remaining_capacity():
    dev = spec.selected_device()
    capacity = dev.mem_total_bytes - dev.mem_reserved_bytes
    first = client.gpu(memory=int(capacity * 0.8)).open()
    try:
        with pytest.raises(client.OutOfMemoryError):
            client.gpu(memory=int(capacity * 0.5)).open()
    finally:
        first.close()


def test_dead_process_claim_is_reaped():
    """A claim whose owner has gone must not keep holding memory."""
    daemon = Daemon()
    stale = client.Claim(memory="4GiB", pid=2**30)  # a pid that cannot exist
    stale._open = True
    stale._flush()

    daemon.tick(0.2)
    assert _read(daemon).gpus[0].mem_used < 16 * 1024**2
    assert not stale._path.exists()


# ---------------------------------------------------------------- telemetry


def test_idle_device_looks_idle():
    daemon = Daemon()
    daemon.tick(0.2)
    gpu = _read(daemon).gpus[0]
    dev = spec.selected_device()
    assert gpu.util_gpu == 0
    assert gpu.pstate == 8
    assert gpu.power_mw <= dev.power_idle_w * 1000 * 1.2
    assert gpu.fan_speed == -1  # the L4 is passively cooled


def test_load_raises_utilisation_power_and_clocks():
    daemon = Daemon()
    with client.gpu(memory="1GiB", util=100):
        # Several ticks, because the model deliberately ramps rather than jumps.
        for _ in range(40):
            daemon.tick(0.2)
        gpu = _read(daemon).gpus[0]

    dev = spec.selected_device()
    assert gpu.util_gpu > 90
    assert gpu.pstate == 0
    assert gpu.power_mw > dev.power_idle_w * 1000 * 2
    assert gpu.power_mw <= dev.power_limit_w * 1000
    assert gpu.clock_sm > dev.idle_clock_gr_mhz * 2


def test_temperature_lags_behind_load():
    """Heat must not appear instantly, or the reading teaches nothing."""
    daemon = Daemon()
    dev = spec.selected_device()
    with client.gpu(memory="1GiB", util=100):
        daemon.tick(0.2)
        after_one_tick = _read(daemon).gpus[0].temp
        assert after_one_tick < dev.temp_idle_c + 3

        for _ in range(300):  # a minute of simulated time
            daemon.tick(0.2)
        warmed = _read(daemon).gpus[0].temp

    assert warmed > after_one_tick + 5
    assert warmed < dev.temp_slowdown_c


def test_utilisation_never_exceeds_one_hundred():
    """Two greedy claims share the card rather than summing past full."""
    daemon = Daemon()
    with client.gpu(memory="1GiB", util=80), client.gpu(memory="1GiB", util=80):
        for _ in range(40):
            daemon.tick(0.2)
        gpu = _read(daemon).gpus[0]
    assert gpu.util_gpu <= 100
    assert sum(p.sm_util for p in gpu.processes) <= 101  # rounding slack


def _read(daemon):
    from gpuemu.shm import StateReader

    with StateReader(daemon.writer.path) as r:
        return r.read()


# ---------------------------------------------------------------- visibility


def test_no_gpu_visible_when_cuda_visible_devices_is_empty(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert client.visible_devices() == []


def test_all_visible_when_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert client.visible_devices() is None


# ---------------------------------------------------------------- slurm


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("30", 1800),
        ("10:00", 600),
        ("1:00:00", 3600),
        ("2:30:00", 9000),
        ("1-00:00:00", 86400),
        ("1-12", 129600),
    ],
)
def test_parse_time_limit(text, seconds):
    assert slurm.parse_time_limit(text) == seconds


@pytest.mark.parametrize(
    "text,mb", [("1024", 1024), ("2G", 2048), ("512M", 512), ("1T", 1024 * 1024)]
)
def test_parse_memory(text, mb):
    assert slurm.parse_memory(text) == mb


@pytest.mark.parametrize(
    "text,n", [("gpu:1", 1), ("gpu:2", 2), ("gpu:l4:1", 1), ("gpu", 1), ("nvme:1", 0)]
)
def test_parse_gres(text, n):
    assert slurm.parse_gres(text) == n


def test_sbatch_directives_stop_at_first_command(tmp_path):
    script = tmp_path / "job.sl"
    script.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=real\n"
        "#SBATCH --gpus-per-node=1\n"
        "\n"
        "echo hello\n"
        "#SBATCH --job-name=ignored\n"
    )
    args = slurm.read_sbatch_directives(script)
    assert "--job-name=real" in args
    assert "--gpus-per-node=1" in args
    assert "--job-name=ignored" not in args


def test_sbatch_queues_a_job(tmp_path, capsys):
    script = tmp_path / "job.sl"
    script.write_text("#!/bin/bash\n#SBATCH --job-name=test\n#SBATCH --gpus-per-node=1\necho hi\n")
    monkey_cwd(tmp_path)

    rc = slurm.sbatch([str(script)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Submitted batch job" in out

    jobs = slurm.JobStore().all()
    assert len(jobs) == 1
    assert jobs[0].name == "test"
    assert jobs[0].gpus == 1
    assert jobs[0].state == slurm.PENDING


def test_sbatch_rejects_more_gpus_than_exist(tmp_path, capsys):
    script = tmp_path / "job.sl"
    script.write_text("#!/bin/bash\n#SBATCH --gpus-per-node=99\necho hi\n")
    monkey_cwd(tmp_path)

    rc = slurm.sbatch([str(script)])
    assert rc == 1
    assert "not available" in capsys.readouterr().err


def test_job_without_gpu_request_gets_no_visible_device():
    env = slurm._job_environment(
        job_id=1, name="j", user="u", cpus=1, mem_mb=512,
        gpu_ids=[], ntasks=1, workdir="/tmp",
    )
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert "SLURM_JOB_GPUS" not in env


def test_job_with_gpu_request_sees_it():
    env = slurm._job_environment(
        job_id=1, name="j", user="u", cpus=1, mem_mb=512,
        gpu_ids=[0], ntasks=1, workdir="/tmp",
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["SLURM_JOB_GPUS"] == "0"
    assert env["SLURM_GPUS_ON_NODE"] == "1"


def test_scheduler_runs_a_job_to_completion(tmp_path):
    script = tmp_path / "job.sl"
    script.write_text(
        "#!/bin/bash\n#SBATCH --gpus-per-node=1\n"
        'echo "devices=$CUDA_VISIBLE_DEVICES"\n'
    )
    monkey_cwd(tmp_path)
    slurm.sbatch([str(script)])

    sched = slurm.Scheduler()
    deadline = time.time() + 30
    while time.time() < deadline:
        sched.tick()
        job = slurm.JobStore().all()[0]
        if job.state not in slurm.ACTIVE_STATES:
            break
        time.sleep(0.1)

    job = slurm.JobStore().all()[0]
    assert job.state == slurm.COMPLETED, f"job ended {job.state}"
    assert job.exit_code == 0
    output = (tmp_path / f"slurm-{job.job_id}.out").read_text()
    assert "devices=0" in output


def test_scheduler_records_failure_exit_code(tmp_path):
    script = tmp_path / "bad.sl"
    script.write_text("#!/bin/bash\nexit 7\n")
    monkey_cwd(tmp_path)
    slurm.sbatch([str(script)])

    sched = slurm.Scheduler()
    deadline = time.time() + 30
    while time.time() < deadline:
        sched.tick()
        job = slurm.JobStore().all()[0]
        if job.state not in slurm.ACTIVE_STATES:
            break
        time.sleep(0.1)

    job = slurm.JobStore().all()[0]
    assert job.state == slurm.FAILED
    assert job.exit_code == 7


def monkey_cwd(path):
    os.chdir(path)


# ---------------------------------------------------------------- vram override


@pytest.mark.parametrize(
    "text,expected_mib",
    [("1GiB", 1024), ("512MiB", 512), ("2G", 2048), ("2048", 2048), ("8GiB", 8192)],
)
def test_mem_total_override(monkeypatch, text, expected_mib):
    monkeypatch.setenv("GPUEMU_MEM_TOTAL", text)
    dev = spec.selected_device()
    assert dev.mem_total_mib == expected_mib
    # The driver reservation stays proportional, so "free" remains plausible.
    assert 0 < dev.mem_reserved_mib < dev.mem_total_mib * 0.1


def test_mem_total_override_is_reported_everywhere(monkeypatch, capsys):
    """A 1 GB card must look like a 1 GB card to nvidia-smi and to OOM checks."""
    monkeypatch.setenv("GPUEMU_MEM_TOTAL", "1GiB")
    from gpuemu import smi

    daemon = Daemon()
    daemon.tick(0.2)
    assert _read(daemon).gpus[0].mem_total == 1024 * 1024**2

    smi.main(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    assert capsys.readouterr().out.strip() == "1024"

    # And the limit actually binds: 2 GiB will not fit on a 1 GiB card.
    with pytest.raises(client.OutOfMemoryError):
        client.gpu(memory="2GiB").open()


def test_rejects_unparseable_mem_total(monkeypatch):
    monkeypatch.setenv("GPUEMU_MEM_TOTAL", "loads")
    with pytest.raises(SystemExit):
        spec.selected_device()


# ---------------------------------------------------------------- devices


@pytest.mark.parametrize("key", ["l4", "a100", "h100", "rtxpro6000"])
def test_every_device_is_internally_consistent(monkeypatch, key):
    """Each card must be self-consistent enough to model and to display."""
    monkeypatch.setenv("GPUEMU_DEVICE", key)
    monkeypatch.delenv("GPUEMU_MEM_TOTAL", raising=False)
    d = spec.selected_device()

    assert d.mem_reserved_mib < d.mem_total_mib
    assert d.power_idle_w < d.power_limit_w
    assert d.power_min_limit_w <= d.power_limit_w
    assert d.temp_idle_c < d.temp_max_load_c < d.temp_slowdown_c < d.temp_shutdown_c
    assert d.idle_clock_gr_mhz < d.max_clock_gr_mhz
    assert d.cuda_cores > 0 and d.sm_count > 0
    assert d.architecture and d.architecture != "Unknown"

    # And it must actually drive the simulation without blowing a limit.
    daemon = Daemon()
    with client.gpu(memory="256MiB", util=100):
        for _ in range(60):
            daemon.tick(0.2)
        gpu = _read(daemon).gpus[0]
    assert gpu.name == d.name
    assert gpu.util_gpu > 80
    assert gpu.power_mw <= d.power_limit_w * 1000
    assert gpu.temp < d.temp_slowdown_c


def test_rtx_pro_6000_headline_figures(monkeypatch):
    """The numbers a learner would check against NVIDIA's published specs."""
    monkeypatch.setenv("GPUEMU_DEVICE", "rtxpro6000")
    monkeypatch.delenv("GPUEMU_MEM_TOTAL", raising=False)
    d = spec.selected_device()

    assert d.name == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert d.architecture == "Blackwell"
    assert d.cuda_cores == 24064
    assert d.sm_count == 188
    assert (d.cc_major, d.cc_minor) == (12, 0)
    assert d.power_limit_w == 600.0
    assert d.pcie_max_gen == 5
    assert not d.has_fan  # Server Edition is passive
    assert 95 <= d.mem_total_mib / 1024 <= 96  # ~96 GB

    # FP32 = cores x 2 x clock should land on NVIDIA's quoted 120 TFLOPS.
    tflops = d.cuda_cores * 2 * (d.max_clock_gr_mhz * 1e6) / 1e12
    assert 118 <= tflops <= 122, tflops


def test_unknown_device_names_are_listed_in_the_error(monkeypatch):
    monkeypatch.setenv("GPUEMU_DEVICE", "rtx4090")
    with pytest.raises(SystemExit) as exc:
        spec.selected_device()
    assert "rtxpro6000" in str(exc.value)
