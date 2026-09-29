"""start.sh: how the container's entrypoint hands over to the service.

Run for real with `python3` and `nvidia-smi` replaced by small scripts, so what is
checked is the shell's behaviour, not its text:

* SIGTERM must reach the service. Without `exec` bash stays PID 1 and does not
  forward it, so `docker stop` (and every image update) ended in SIGKILL and the
  service never got the chance to stop its running job.
* an existing CUDA_VISIBLE_DEVICES / PYTORCH_CUDA_ALLOC_CONF wins over the
  script's guesses. `export CUDA_VISIBLE_DEVICES=0` made GPU selection from
  compose impossible.
* the RAM size comes from /proc/meminfo. It used to come from `free`, which is
  procps and is not installed in the image, so it was always 0 and every start
  took the "Low RAM" branch.
"""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from test_piper_training_service_support import SERVICE_DIR

START = SERVICE_DIR / "start.sh"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bash") is None,
    reason="start.sh needs Linux and bash",
)

FAKE_PYTHON = """#!/bin/bash
# Stands in for `python3 app.py`: reports what it inherited, then waits for SIGTERM.
echo "PID=$$"
echo "ARGS=$*"
echo "CUDA=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "ALLOC=${PYTORCH_CUDA_ALLOC_CONF-<unset>}"
trap 'echo GOT-SIGTERM; exit 0' TERM
echo READY
while true; do sleep 0.05; done
"""

FAKE_NVIDIA_SMI = """#!/bin/bash
echo "Fake GPU, 16384"
"""

# The image has no procps. Installed here as a `free` that finds nothing, so a script that still
# asks it for the RAM size gets what the container gave it: no answer.
FAKE_FREE = """#!/bin/sh
exit 127
"""


def _executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def start_service(tmp_path: Path, extra_env: dict, unset: tuple = ()):
    """Run start.sh; returns (process, reported settings). Caller must stop the process."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "python3", FAKE_PYTHON)
    _executable(bin_dir / "nvidia-smi", FAKE_NVIDIA_SMI)
    _executable(bin_dir / "free", FAKE_FREE)

    env = {k: v for k, v in os.environ.items()
           if k not in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "TRAINING_MEMORY_OPTIMIZATION", *unset)}
    env["PATH"] = f"{bin_dir}:/usr/local/bin:/usr/bin:/bin"
    env.update(extra_env)

    proc = subprocess.Popen(
        ["bash", str(START)], cwd=tmp_path, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    reported: dict = {"lines": []}
    for line in proc.stdout:
        reported["lines"].append(line.rstrip())
        key, sep, value = line.rstrip().partition("=")
        if sep and key in ("PID", "CUDA", "ALLOC", "ARGS"):
            reported[key] = value
        if line.strip() == "READY":
            break
    return proc, reported


def stop(proc, reported):
    """Best-effort cleanup that also kills an orphaned service left by a broken start.sh."""
    if proc.poll() is None:
        proc.kill()
    if reported.get("PID", "").isdigit():
        try:
            os.kill(int(reported["PID"]), signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=5)


def test_the_service_replaces_the_shell_so_sigterm_reaches_it(tmp_path):
    proc, reported = start_service(tmp_path, {})
    try:
        assert reported.get("ARGS") == "app.py"
        assert int(reported["PID"]) == proc.pid, (
            "the service is a child of the shell: SIGTERM from `docker stop` goes to bash, "
            "which does not forward it")
        proc.send_signal(signal.SIGTERM)
        remaining = proc.stdout.read()
        assert proc.wait(timeout=10) == 0
        assert "GOT-SIGTERM" in remaining, "the service never saw the signal"
    finally:
        stop(proc, reported)


@pytest.mark.parametrize("given,expected", [
    (None, "0"),            # nothing configured: the script's default
    ("1", "1"),             # compose picked another card
    ("0,1", "0,1"),
    ("", ""),               # explicitly empty hides every GPU; that is a choice, not a gap
])
def test_an_existing_cuda_visible_devices_is_respected(tmp_path, given, expected):
    extra = {} if given is None else {"CUDA_VISIBLE_DEVICES": given}
    proc, reported = start_service(tmp_path, extra)
    try:
        assert reported.get("CUDA") == expected
    finally:
        stop(proc, reported)


def test_an_existing_allocator_setting_is_not_overridden(tmp_path):
    proc, reported = start_service(tmp_path, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    try:
        assert reported.get("ALLOC") == "expandable_segments:True"
    finally:
        stop(proc, reported)


def _meminfo(tmp_path: Path, kb: int) -> str:
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:       {kb} kB\nMemFree:        {kb // 2} kB\nMemAvailable:   {kb // 2} kB\nBuffers:        1 kB\n")
    return str(path)


def _reported_ram(reported) -> str:
    return next((line for line in reported["lines"] if "Total system RAM" in line), "")


@pytest.mark.parametrize("kb, gigabytes, alloc", [
    (8 * 1024 * 1024, 8, "max_split_size_mb:512"),             # a genuinely small machine
    (int(7.6 * 1024 * 1024), 8, "max_split_size_mb:512"),      # 8 GB installed reports ~7.6 GiB
    (15 * 1024 * 1024, 15, "max_split_size_mb:512"),
    (16481980, 16, "max_split_size_mb:1024"),                  # 16 GB installed: ~15.7 GiB, is not "low"
    (24 * 1024 * 1024, 24, "max_split_size_mb:1024"),
    (64 * 1024 * 1024, 64, "max_split_size_mb:1024"),
])
def test_the_ram_size_comes_from_proc_meminfo_and_picks_the_allocator_default(tmp_path, kb, gigabytes, alloc):
    """Without procps `free` answers nothing: the size was 0 and every start was a "Low RAM" start."""
    proc, reported = start_service(tmp_path, {"MEMINFO_FILE": _meminfo(tmp_path, kb)})
    try:
        assert _reported_ram(reported).endswith(f"{gigabytes}GB"), reported["lines"]
        assert reported.get("ALLOC") == alloc
    finally:
        stop(proc, reported)


def test_the_real_proc_meminfo_is_used_when_nothing_overrides_it(tmp_path):
    proc, reported = start_service(tmp_path, {}, unset=("MEMINFO_FILE",))
    try:
        with open("/proc/meminfo") as f:
            total_kb = next(int(line.split()[1]) for line in f if line.startswith("MemTotal:"))
        expected = (total_kb + 524288) // 1048576
        assert _reported_ram(reported).endswith(f"{expected}GB"), (reported["lines"], expected)
        assert expected == 0 or "unknown" not in _reported_ram(reported)
    finally:
        stop(proc, reported)


@pytest.mark.parametrize("content", ["", "no memory line here\n", "MemTotal: lots kB\n"])
def test_an_unreadable_meminfo_is_reported_as_unknown_not_as_no_ram(tmp_path, content):
    """"Could not tell" used to be reported as 0 GB, which took the low-RAM branch."""
    path = tmp_path / "meminfo"
    path.write_text(content)
    proc, reported = start_service(tmp_path, {"MEMINFO_FILE": str(path)})
    try:
        assert "unknown" in _reported_ram(reported)
        assert "Low RAM" not in "\n".join(reported["lines"])
        assert reported.get("ALLOC") == "max_split_size_mb:1024"
    finally:
        stop(proc, reported)


def test_a_missing_meminfo_file_is_survived(tmp_path):
    proc, reported = start_service(tmp_path, {"MEMINFO_FILE": str(tmp_path / "nope")})
    try:
        assert "unknown" in _reported_ram(reported)
        assert reported.get("ALLOC") == "max_split_size_mb:1024"
    finally:
        stop(proc, reported)


def test_an_explicit_allocator_setting_wins_at_every_ram_size(tmp_path):
    proc, reported = start_service(
        tmp_path, {"MEMINFO_FILE": _meminfo(tmp_path, 4 * 1024 * 1024), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    try:
        assert "Low RAM" in "\n".join(reported["lines"]), "the branch still runs, it just does not override"
        assert reported.get("ALLOC") == "expandable_segments:True"
    finally:
        stop(proc, reported)


def test_the_script_no_longer_depends_on_procps_or_exports_an_unread_variable():
    text = START.read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "free -" not in code and "free " not in code.replace("freed", ""), "`free` is procps, not in the image"
    assert "TRAINING_MEMORY_OPTIMIZATION" not in code, "exported, and read by nothing in the repo"
