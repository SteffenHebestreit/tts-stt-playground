"""Image recipes: the pieces that decide what a rebuild installs.

Nothing here builds an image (there is no daemon in CI). The RUN instructions that
decide the outcome are read out of the Dockerfiles and executed with `pip3`, `wget` and
friends replaced by recorders, or, for the torch check, against a stand-in `torch`.

* D1  the Piper voice seed builds the same image tag as the Piper service, with the same
      build arguments (two builds wrote one tag; a custom PIPER_VOICES could be dropped);
* D2  the STT images run Python 3.11, not the base image's end-of-life 3.10 (on which the
      resolver held numpy, scipy, scikit-learn and onnxruntime back);
* D3  the STT images pin torch/torchaudio through ARGs like every other image, and the CUDA
      one checks for Blackwell kernels at build time;
* D4  NeMo is installed with torch held by a constraint file (parakeet, canary);
* D5  the Piper release asset name survives a builder that does not set TARGETARCH.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from compose_helpers import REPO_ROOT, load_compose
from packaging.version import Version

STT = REPO_ROOT / "stt-service"
STT_CUDA, STT_ROCM = STT / "Dockerfile", STT / "Dockerfile.rocm"
NEMO_IMAGES = [REPO_ROOT / "parakeet-asr-service" / "Dockerfile", REPO_ROOT / "canary-asr-service" / "Dockerfile"]
PIPER_TTS = REPO_ROOT / "piper-tts-service" / "Dockerfile"


def _instructions(dockerfile: Path) -> list[tuple[str, str]]:
    text = dockerfile.read_text(encoding="utf-8").replace("\\\n", " ")
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, rest = line.partition(" ")
        out.append((keyword.upper(), rest.strip()))
    return out


def _runs(dockerfile: Path) -> list[str]:
    return [args for keyword, args in _instructions(dockerfile) if keyword == "RUN"]


def _arg_default(dockerfile: Path, name: str) -> str:
    for keyword, args in _instructions(dockerfile):
        if keyword == "ARG" and args.startswith(f"{name}="):
            return shlex.split(args.split("=", 1)[1])[0]
    raise AssertionError(f"{dockerfile.parent.name}/{dockerfile.name} declares no ARG {name} with a default")


def _all_dockerfiles() -> list[Path]:
    return sorted(p for p in REPO_ROOT.glob("*/Dockerfile*") if p.is_file() and p.parent.name != "tests")


def _shell(command: str, tmp_path: Path, env: dict, stubs: dict[str, str]):
    """Run *command* with each name in *stubs* replaced by a script (its body) on the PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in stubs.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    return subprocess.run(
        ["sh", "-c", command], cwd=tmp_path, capture_output=True, text=True, timeout=60,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), **env},
        stdin=subprocess.PIPE,   # an explicit pipe: /dev/null is not reliable in every sandbox
    )


# --- D1: one build recipe for the Piper image ------------------------------------------------------------------


def _compose_services() -> dict:
    return load_compose(REPO_ROOT / "docker-compose.yml")["services"]


def test_the_voice_seed_and_the_piper_service_are_the_same_build():
    services = _compose_services()
    seed, piper = services["piper-voices-seed"], services["piper-tts-service"]

    assert seed["image"] == piper["image"], "they share a tag, so their builds must be one build"
    assert seed["build"] == piper["build"]
    assert set(piper["build"]["args"]) == {"PIPER_VOICES", "PIPER_VOICES_REVISION"}


def test_a_custom_piper_voices_reaches_both_builds_in_the_real_compose_parser():
    """`docker compose config` needs no daemon. The seed used to be rendered without the arguments."""
    if not shutil.which("docker"):
        pytest.skip("docker CLI not installed")
    probe = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True, stdin=subprocess.PIPE)
    if probe.returncode != 0:
        pytest.skip("docker compose plugin not installed")

    rendered = subprocess.run(
        ["docker", "compose", "--env-file", ".env.example", "--profile", "piper-tts", "-f", "docker-compose.yml",
         "config", "--format", "json"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, stdin=subprocess.PIPE,
        env={**os.environ, "PIPER_VOICES": "de_DE-thorsten-medium", "PIPER_VOICES_REVISION": "v9.9.9"},
    )
    assert rendered.returncode == 0, rendered.stderr
    services = json.loads(rendered.stdout)["services"]

    for name in ("piper-voices-seed", "piper-tts-service"):
        assert services[name]["build"]["args"] == {
            "PIPER_VOICES": "de_DE-thorsten-medium", "PIPER_VOICES_REVISION": "v9.9.9"}, name
        assert services[name]["build"]["dockerfile"] == "Dockerfile", name


# --- D2: Python 3.11 in the STT images ------------------------------------------------------------------------------


def _apt_packages(dockerfile: Path) -> list[str]:
    for run in _runs(dockerfile):
        if "apt-get install" in run:
            after = run.split("apt-get install", 1)[1]
            return [t for t in shlex.split(after.split("&&")[0]) if not t.startswith("-")]
    return []


@pytest.mark.parametrize("dockerfile", [STT_CUDA, STT_ROCM], ids=lambda p: p.name)
def test_the_stt_images_install_python_3_11_and_point_python3_at_it(dockerfile):
    packages = _apt_packages(dockerfile)
    assert "python3.11" in packages and "python3.11-dev" in packages
    assert "python3" not in packages and "python3-dev" not in packages, "the base's 3.10 is end of life"

    links = " ".join(_runs(dockerfile))
    assert "ln -sf /usr/bin/python3.11 /usr/bin/python3" in links, "pip3 and `python3 -m ...` would still run 3.10"
    assert "ln -sf /usr/bin/python3.11 /usr/bin/python " in links + " ", "start.sh execs `python -m uvicorn`"


def _python_version(dockerfile: Path):
    for keyword, args in _instructions(dockerfile):
        if keyword == "FROM" and args.startswith("python:"):
            return Version(args.split(":", 1)[1].split("-")[0])
    versions = [t.removeprefix("python") for t in _apt_packages(dockerfile)
                if t.startswith("python3.") and not t.endswith("-dev")]
    return max(map(Version, versions)) if versions else None


@pytest.mark.parametrize(
    "dockerfile",
    [p for p in _all_dockerfiles() if any("pip" in run for run in _runs(p))],
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_no_image_runs_the_base_images_python_3_10(dockerfile):
    """3.10 reaches end of life in 2026-10 and holds the resolver back; every image names its Python."""
    version = _python_version(dockerfile)
    assert version is not None, f"{dockerfile.parent.name}/{dockerfile.name} runs the base image's own python3"
    assert version >= Version("3.11")


# --- D3: pinned torch in the STT images ---------------------------------------------------------------------------------


def _torch_install(dockerfile: Path) -> list[str]:
    return shlex.split(next(run for run in _runs(dockerfile) if "download.pytorch.org/whl/" in run).split("&&")[0])


def test_the_stt_cuda_image_pins_a_cu128_pair_ctranslate2_is_happy_with():
    torch, audio = _arg_default(STT_CUDA, "TORCH_VERSION"), _arg_default(STT_CUDA, "TORCHAUDIO_VERSION")
    assert torch == audio, "torch and torchaudio are released in lockstep"
    assert Version("2.7") <= Version(torch) < Version("2.12"), "cu128 wheels exist for 2.7 through 2.11 only"

    argv = _torch_install(STT_CUDA)
    assert "torch==${TORCH_VERSION}" in argv and "torchaudio==${TORCHAUDIO_VERSION}" in argv
    assert argv[argv.index("--index-url") + 1] == "https://download.pytorch.org/whl/cu128"
    assert "--no-cache-dir" in argv
    # ctranslate2 4.x needs CUDA 12 + cuDNN 9: the base image, not this pin, provides them.
    assert _instructions(STT_CUDA)[0] == ("FROM", "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04")


def test_the_stt_rocm_image_pins_the_newest_release_its_index_serves():
    torch, audio = _arg_default(STT_ROCM, "TORCH_VERSION"), _arg_default(STT_ROCM, "TORCHAUDIO_VERSION")
    assert torch == audio == "2.5.1"

    argv = _torch_install(STT_ROCM)
    assert "torch==${TORCH_VERSION}" in argv and "torchaudio==${TORCHAUDIO_VERSION}" in argv
    assert argv[argv.index("--index-url") + 1] == "https://download.pytorch.org/whl/rocm6.2"


def _check_programs(dockerfile: Path) -> list[str]:
    programs = []
    for run in _runs(dockerfile):
        argv = shlex.split(run)
        programs += [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == "-c" and "_cuda_getArchFlags" in argv[i + 1]]
    return programs


def _run_torch_check(program: str, tmp_path: Path, *, cuda="12.8", flags="sm_75 sm_90 sm_120"):
    package = tmp_path / "torch"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text(textwrap.dedent(f"""
        from types import SimpleNamespace
        __version__ = "2.11.0+cu128"
        version = SimpleNamespace(cuda={cuda!r})
        _C = SimpleNamespace(_cuda_getArchFlags=lambda: {flags!r})
        cuda = SimpleNamespace(get_arch_list=lambda: [])   # what a build host without a GPU answers
    """))
    done = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, env={"PYTHONPATH": str(tmp_path), "PATH": ""},
        capture_output=True, text=True, timeout=30, stdin=subprocess.PIPE,
    )
    return done.returncode, done.stdout, done.stderr


def test_the_stt_cuda_image_checks_for_blackwell_kernels_in_the_torch_install_step(tmp_path):
    runs = _runs(STT_CUDA)
    torch_step = next(i for i, run in enumerate(runs) if "download.pytorch.org/whl/" in run)
    programs = _check_programs(STT_CUDA)

    assert len(programs) == 1 and "_cuda_getArchFlags" in runs[torch_step], "the check must be part of the torch step"
    code, out, err = _run_torch_check(programs[0], tmp_path)
    assert code == 0, err
    assert "sm_120" in out


@pytest.mark.parametrize("kwargs", [
    {"cuda": "12.6"}, {"cuda": "13.0"}, {"cuda": None}, {"flags": "sm_75 sm_80 sm_90"}, {"flags": ""},
])
def test_an_stt_torch_that_cannot_run_on_an_rtx_50_card_fails_the_build(tmp_path, kwargs):
    code, _, err = _run_torch_check(_check_programs(STT_CUDA)[0], tmp_path, **kwargs)
    assert code != 0 and "FATAL" in err and "sm_120" in err


def test_the_rocm_stt_image_has_no_cuda_check():
    runs = " ".join(_runs(STT_ROCM))
    assert "_cuda_getArchFlags" not in runs and "cu128" not in runs and "get_arch_list" not in runs


@pytest.mark.parametrize(
    "dockerfile",
    [p for p in _all_dockerfiles() if any("whl/cu128" in run for run in _runs(p))],
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_every_cu128_image_pins_torch_and_checks_for_blackwell_at_build_time(dockerfile):
    """A rebuild must not move to a wheel whose cu128 build dropped sm_120 without failing the build."""
    argv = _torch_install(dockerfile)
    assert "torch==${TORCH_VERSION}" in argv, f"{dockerfile.parent.name} installs an unpinned torch"
    assert Version(_arg_default(dockerfile, "TORCH_VERSION")) >= Version("2.7")
    assert _check_programs(dockerfile), f"{dockerfile.parent.name} has no build-time Blackwell check"


# --- D4: NeMo installs with torch held ---------------------------------------------------------------------------------


@pytest.mark.parametrize("dockerfile", NEMO_IMAGES, ids=lambda p: p.parent.name)
def test_the_nemo_install_holds_torch_and_torchaudio_with_a_constraint_file(dockerfile, tmp_path):
    """A later NeMo whose requirements ask for another torch fails the resolver at the install,
    not at the post-check after gigabytes of a different wheel."""
    command = next(run for run in _runs(dockerfile) if "requirements.app.txt" in run)
    constraints = tmp_path / "torch-constraints.txt"
    (tmp_path / "requirements.txt").write_text((dockerfile.parent / "requirements.txt").read_text(encoding="utf-8"))
    torch, audio = _arg_default(dockerfile, "TORCH_VERSION"), _arg_default(dockerfile, "TORCHAUDIO_VERSION")

    done = _shell(
        command.replace("/tmp/torch-constraints.txt", str(constraints)), tmp_path,
        env={"TORCH_VERSION": torch, "TORCHAUDIO_VERSION": audio, "NEMO_TOOLKIT_SPEC": ">=3.0.0,<3.1",
             "RECORD": str(tmp_path / "pip-argv")},
        stubs={"pip3": 'for a in "$@"; do echo "$a"; done > "$RECORD"', "python3": "exit 0"},
    )

    assert done.returncode == 0, done.stderr
    argv = (tmp_path / "pip-argv").read_text().split("\n")
    assert argv[argv.index("-c") + 1] == str(constraints)
    assert constraints.read_text().split() == [f"torch=={torch}", f"torchaudio=={audio}"]
    assert "nemo_toolkit[asr]>=3.0.0,<3.1" in argv, "the NeMo spec is unchanged"
    assert argv.index("-c") < argv.index("nemo_toolkit[asr]>=3.0.0,<3.1"), "-c must apply to the whole install"
    assert argv[argv.index("-r") + 1] == "requirements.app.txt"


@pytest.mark.parametrize("dockerfile", NEMO_IMAGES, ids=lambda p: p.parent.name)
def test_the_nemo_defaults_are_untouched(dockerfile):
    assert _arg_default(dockerfile, "NEMO_TOOLKIT_SPEC") == ">=3.0.0,<3.1"
    assert _arg_default(dockerfile, "TORCH_VERSION") == _arg_default(dockerfile, "TORCHAUDIO_VERSION") == "2.11.0"


# --- D5: the Piper release asset without BuildKit ---------------------------------------------------------------------


def _piper_download(tmp_path: Path, *, targetarch: str | None, machine: str) -> str:
    command = next(run for run in _runs(PIPER_TTS) if "piper_" in run and "wget" in run)
    env = {} if targetarch is None else {"TARGETARCH": targetarch}
    done = _shell(
        command, tmp_path, env={**env, "RECORD": str(tmp_path / "wget-argv")},
        stubs={
            "wget": 'for a in "$@"; do echo "$a"; done > "$RECORD"',
            "tar": "exit 0", "rm": "exit 0", "ln": "exit 0",
            "uname": f'echo {machine}',
        },
    )
    assert done.returncode == 0, done.stderr
    return next(a for a in (tmp_path / "wget-argv").read_text().split("\n") if a.startswith("https://"))


BASE = "https://github.com/rhasspy/piper/releases/download/v1.2.0/"


@pytest.mark.parametrize("machine, asset", [
    ("x86_64", "piper_amd64.tar.gz"),
    ("aarch64", "piper_arm64.tar.gz"),      # a Rockchip board building with the classic builder
    ("arm64", "piper_arm64.tar.gz"),
    ("riscv64", "piper_amd64.tar.gz"),      # unknown host: the documented default
])
def test_a_builder_without_targetarch_gets_a_real_asset_name(tmp_path, machine, asset):
    """The classic builder and podman leave TARGETARCH unset: the URL used to be piper_.tar.gz (404)."""
    assert _piper_download(tmp_path, targetarch=None, machine=machine) == BASE + asset


def test_an_empty_targetarch_is_treated_as_unset(tmp_path):
    assert _piper_download(tmp_path, targetarch="", machine="x86_64") == BASE + "piper_amd64.tar.gz"


@pytest.mark.parametrize("targetarch, machine", [("arm64", "x86_64"), ("amd64", "aarch64")])
def test_buildkits_value_wins_over_the_hosts_architecture(tmp_path, targetarch, machine):
    """A cross-build under BuildKit: the target, not the host, names the asset."""
    assert _piper_download(tmp_path, targetarch=targetarch, machine=machine) == BASE + f"piper_{targetarch}.tar.gz"
