"""start.sh: how the container's entrypoint hands over to the service.

Run for real with `python3` and `nvidia-smi` replaced by small scripts, so what is
checked is the shell's behaviour, not its text:

* SIGTERM must reach the service. Without `exec` bash stays PID 1 and does not
  forward it, so `docker stop` (and every image update) ended in SIGKILL and the
  service never got the chance to stop its running job.
* an existing CUDA_VISIBLE_DEVICES / PYTORCH_CUDA_ALLOC_CONF wins over the
  script's guesses. `export CUDA_VISIBLE_DEVICES=0` made GPU selection from
  compose impossible.
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
    not sys.platform.startswith("linux") or shutil.which("bash") is None
    or shutil.which("free") is None or shutil.which("awk") is None,
    reason="start.sh needs a Linux userland (bash, free, awk)",
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


def _executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def start_service(tmp_path: Path, extra_env: dict, unset: tuple = ()):
    """Run start.sh; returns (process, reported settings). Caller must stop the process."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "python3", FAKE_PYTHON)
    _executable(bin_dir / "nvidia-smi", FAKE_NVIDIA_SMI)

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
