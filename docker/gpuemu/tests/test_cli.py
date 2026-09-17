"""Every command-line entry point must at least parse its arguments and run.

These exist because a `squeue -h` that crashed on an argparse option conflict
shipped past the behaviour tests: nothing had ever *invoked* squeue, only the
functions underneath it. Slurm spells several flags differently from argparse's
defaults (-h is --noheader in squeue and sinfo, not --help), so the parsers need
exercising directly.
"""

from __future__ import annotations

import pytest

from gpuemu import cli, slurm, smi


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUEMU_STATE_FILE", str(tmp_path / "state.bin"))
    monkeypatch.setenv("GPUEMU_CLAIMS_DIR", str(tmp_path / "claims"))
    monkeypatch.setenv("GPUEMU_SLURM_DIR", str(tmp_path / "slurm"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def device(tmp_path):
    """A single published device frame, as if the daemon had ticked once."""
    from gpuemu.daemon import Daemon

    d = Daemon()
    d.tick(0.2)
    return d


# ---------------------------------------------------------------- slurm CLIs


def test_squeue_runs(capsys):
    assert slurm.squeue([]) == 0
    assert "JOBID" in capsys.readouterr().out


def test_squeue_dash_h_means_noheader(capsys):
    """Slurm's -h suppresses the header; it must not be argparse's --help."""
    assert slurm.squeue(["-h"]) == 0
    assert "JOBID" not in capsys.readouterr().out


def test_squeue_long_help_still_works():
    with pytest.raises(SystemExit) as exc:
        slurm.squeue(["--help"])
    assert exc.value.code == 0


def test_sinfo_runs(capsys):
    assert slurm.sinfo([]) == 0
    out = capsys.readouterr().out
    assert "PARTITION" in out
    assert slurm.NODE_NAME in out


def test_sinfo_dash_h_means_noheader(capsys):
    assert slurm.sinfo(["-h"]) == 0
    out = capsys.readouterr().out
    assert "PARTITION" not in out
    assert slurm.NODE_NAME in out


def test_sacct_runs(capsys):
    assert slurm.sacct([]) == 0
    assert "JobID" in capsys.readouterr().out


def test_sacct_noheader(capsys):
    assert slurm.sacct(["-n"]) == 0
    assert "JobID" not in capsys.readouterr().out


def test_scancel_on_unknown_job_is_not_fatal(capsys):
    assert slurm.scancel(["999999"]) == 0
    assert "Invalid job id" in capsys.readouterr().err


def test_full_submit_query_cancel_cycle(tmp_path, capsys):
    script = tmp_path / "long.sl"
    script.write_text(
        "#!/bin/bash\n#SBATCH --gpus-per-node=1\n#SBATCH --cpus-per-task=2\nsleep 300\n"
    )
    assert slurm.sbatch([str(script)]) == 0
    capsys.readouterr()

    jobs = slurm.JobStore().all()
    assert len(jobs) == 1
    job_id = jobs[0].job_id
    assert jobs[0].cpus == 2

    assert slurm.squeue([]) == 0
    assert str(job_id) in capsys.readouterr().out

    assert slurm.scontrol(["show", "job", str(job_id)]) == 0
    out = capsys.readouterr().out
    assert f"JobId={job_id}" in out
    assert "gres/gpu=1" in out

    assert slurm.scancel([str(job_id)]) == 0
    assert slurm.JobStore().load(job_id).state == slurm.CANCELLED


def test_sbatch_wrap_needs_no_script_file(capsys):
    assert slurm.sbatch(["--wrap", "echo hi", "--job-name", "wrapped"]) == 0
    assert "Submitted batch job" in capsys.readouterr().out
    assert slurm.JobStore().all()[0].name == "wrapped"


def test_sbatch_cli_flag_beats_script_directive(tmp_path, capsys):
    script = tmp_path / "j.sl"
    script.write_text("#!/bin/bash\n#SBATCH --job-name=from-file\necho hi\n")
    assert slurm.sbatch(["--job-name", "from-cli", str(script)]) == 0
    capsys.readouterr()
    assert slurm.JobStore().all()[0].name == "from-cli"


# ---------------------------------------------------------------- nvidia-smi


def test_nvidia_smi_table(device, capsys):
    assert smi.main([]) == 0
    out = capsys.readouterr().out
    assert "NVIDIA-SMI" in out
    assert "NVIDIA L4" in out
    assert "No running processes found" in out
    # Every boxed line is exactly the real tool's width.
    box = [ln for ln in out.splitlines() if ln.startswith(("+", "|"))]
    assert box and all(len(ln) == 91 for ln in box)


def test_nvidia_smi_list(device, capsys):
    assert smi.main(["-L"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("GPU 0: NVIDIA L4 (UUID: GPU-")


def test_nvidia_smi_query_csv(device, capsys):
    assert smi.main(["--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"]) == 0
    assert capsys.readouterr().out.strip() == "0, NVIDIA L4, 23034"


def test_nvidia_smi_query_rejects_unknown_field(device):
    with pytest.raises(SystemExit):
        smi.main(["--query-gpu=not_a_field", "--format=csv"])


def test_nvidia_smi_full_query(device, capsys):
    assert smi.main(["-q"]) == 0
    out = capsys.readouterr().out
    assert "Product Name" in out
    assert "FB Memory Usage" in out


def test_nvidia_smi_dmon_and_pmon(device, capsys):
    assert smi.main(["dmon"]) == 0
    assert "gtemp" in capsys.readouterr().out
    assert smi.main(["pmon"]) == 0
    assert "command" in capsys.readouterr().out


def test_nvidia_smi_refuses_to_pretend_it_changed_settings(device, capsys):
    """Accepting -pm silently would teach a command that does nothing."""
    assert smi.main(["-pm", "1"]) == 3
    assert "not supported" in capsys.readouterr().err


def test_nvidia_smi_without_daemon_explains_itself(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUEMU_STATE_FILE", str(tmp_path / "absent.bin"))
    with pytest.raises(SystemExit) as exc:
        smi.main([])
    assert "couldn't communicate with the NVIDIA driver" in str(exc.value)


# ---------------------------------------------------------------- gpuemu-ctl


def test_ctl_status_without_daemon_reports_failure(capsys):
    assert cli.main(["status"]) == 1
    assert "NOT RUNNING" in capsys.readouterr().out


def test_ctl_status_with_device(device, capsys):
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "NVIDIA L4" in out
    assert "GPU 0" in out


# ------------------------------------------------- device visibility in tools


def test_nvidia_smi_finds_nothing_when_no_gpu_allocated(device, monkeypatch):
    """A job that did not request a GPU must not be shown one.

    Real nvidia-smi ignores CUDA_VISIBLE_DEVICES, but on a cluster it runs in a
    cgroup without the device nodes and reports the same thing, so this is the
    behaviour learners will actually meet.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(SystemExit) as exc:
        smi.main([])
    assert "No devices were found" in str(exc.value)


def test_nvidia_smi_shows_the_device_when_allocated(device, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert smi.main(["-L"]) == 0
    assert "NVIDIA L4" in capsys.readouterr().out
