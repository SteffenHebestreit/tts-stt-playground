"""What the training service's images install, and in which order.

`COPY requirements.txt` sat before the torch layer, so editing any line of
requirements.txt invalidated the layer cache from there on and downloaded the
multi-GB torch wheel again. The images now install torch first, like the others.

The CUDA image installs torch from the cu128 index, so it gets the same build-time
Blackwell check the other cu128 images have (compiled-in arch list through
`torch._C._cuda_getArchFlags()`: `torch.cuda.get_arch_list()` is empty on a build
host without a GPU and would fail every build). The ROCm image has no CUDA
requirement to check and none is invented for it.

Nothing here builds an image (no daemon); the RUN commands are read from the files and
the check is executed against a stand-in `torch`.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from packaging.version import Version

SERVICE = Path(__file__).resolve().parents[1] / "piper-training-service"
CUDA = SERVICE / "Dockerfile"
ROCM = SERVICE / "Dockerfile.rocm"


def _instructions(dockerfile: Path) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments), continuations folded in, comments dropped."""
    text = dockerfile.read_text(encoding="utf-8").replace("\\\n", " ")
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, rest = line.partition(" ")
        out.append((keyword.upper(), rest.strip()))
    return out


def _index(instructions, predicate) -> int:
    return next(i for i, (keyword, args) in enumerate(instructions) if predicate(keyword, args))


def _arg_default(dockerfile: Path, name: str) -> str:
    for keyword, args in _instructions(dockerfile):
        if keyword == "ARG" and args.startswith(f"{name}="):
            return args.split("=", 1)[1]
    raise AssertionError(f"{dockerfile.name} declares no ARG {name} with a default")


def _torch_install(instructions):
    return _index(instructions, lambda k, a: k == "RUN" and "download.pytorch.org/whl/" in a)


def _requirements_copy(instructions):
    return _index(instructions, lambda k, a: k == "COPY" and a.startswith("requirements.txt"))


def _requirements_install(instructions):
    return _index(instructions, lambda k, a: k == "RUN" and "-r requirements.txt" in a)


@pytest.mark.parametrize("dockerfile", [CUDA, ROCM], ids=lambda p: p.name)
def test_torch_is_installed_before_requirements_txt_is_copied(dockerfile):
    """A requirements edit must leave the torch layer cached."""
    instructions = _instructions(dockerfile)
    torch_at, copy_at = _torch_install(instructions), _requirements_copy(instructions)

    assert torch_at < copy_at, (
        f"{dockerfile.name}: torch is installed after `COPY requirements.txt`, so every requirements "
        "edit re-downloads it")
    assert copy_at < _requirements_install(instructions)


@pytest.mark.parametrize("dockerfile", [CUDA, ROCM], ids=lambda p: p.name)
def test_nothing_before_the_torch_layer_depends_on_files_that_change_often(dockerfile):
    """`COPY . .` (the application) must come after torch too, or every code edit re-downloads it."""
    instructions = _instructions(dockerfile)
    torch_at = _torch_install(instructions)
    copies = [args for keyword, args in instructions[:torch_at] if keyword in ("COPY", "ADD")]
    assert copies == [], copies


@pytest.mark.parametrize("dockerfile, index", [(CUDA, "cu128"), (ROCM, "rocm6.2")], ids=["cuda", "rocm"])
def test_torch_is_pinned_through_an_arg_and_held_during_the_requirements_install(dockerfile, index):
    instructions = _instructions(dockerfile)
    version = _arg_default(dockerfile, "TORCH_VERSION")
    assert Version(version)

    argv = shlex.split(instructions[_torch_install(instructions)][1])
    assert "torch==${TORCH_VERSION}" in argv
    assert argv[argv.index("--index-url") + 1] == f"https://download.pytorch.org/whl/{index}"
    assert "--no-cache-dir" in argv

    install = instructions[_requirements_install(instructions)][1]
    assert "-c /tmp/torch-constraints.txt" in install, "a requirement could replace the pinned wheel"
    assert "torch==%s" in install and '"${TORCH_VERSION}"' in install


def test_the_cuda_image_stays_on_a_release_that_has_cu128_wheels():
    """PyTorch publishes cu128 wheels for 2.7 through 2.11 only."""
    version = Version(_arg_default(CUDA, "TORCH_VERSION"))
    assert Version("2.7") <= version < Version("2.12")


# --- the build-time Blackwell check (CUDA image only) ----------------------------------------------------


def _torch_checks(dockerfile: Path) -> list[str]:
    programs = []
    for keyword, args in _instructions(dockerfile):
        if keyword != "RUN":
            continue
        argv = shlex.split(args)
        for i, token in enumerate(argv[:-1]):
            if token == "-c" and "_cuda_getArchFlags" in argv[i + 1]:
                programs.append(argv[i + 1])
    return programs


def _run_check(program: str, tmp_path: Path, *, cuda="12.8", flags="sm_75 sm_90 sm_120", arch_list=None):
    package = tmp_path / "torch"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text(textwrap.dedent(f"""
        from types import SimpleNamespace
        __version__ = "2.11.0+cu128"
        version = SimpleNamespace(cuda={cuda!r})
        _C = SimpleNamespace(_cuda_getArchFlags=lambda: {flags!r})
        # what torch.cuda.get_arch_list() answers on a host without a GPU, i.e. during `docker build`
        cuda = SimpleNamespace(get_arch_list=lambda: {list(arch_list or [])!r})
    """))
    done = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, env={"PYTHONPATH": str(tmp_path), "PATH": ""},
        capture_output=True, text=True, timeout=30, stdin=subprocess.PIPE,
    )
    return done.returncode, done.stdout, done.stderr


def test_the_cuda_image_checks_for_blackwell_kernels_right_after_installing_torch():
    instructions = _instructions(CUDA)
    check_at = _index(instructions, lambda k, a: k == "RUN" and "_cuda_getArchFlags" in a)

    assert len(_torch_checks(CUDA)) == 1
    assert _torch_install(instructions) < check_at < _requirements_copy(instructions), (
        "the check must run before anything else is downloaded, so a bad wheel fails the build fast")


def test_a_cuda_12_8_build_with_blackwell_kernels_passes_although_the_build_host_has_no_gpu(tmp_path):
    """Regression guard for the classic mistake: get_arch_list() is [] here, so a check built on it never passes."""
    code, _, err = _run_check(_torch_checks(CUDA)[0], tmp_path, arch_list=[])
    assert code == 0, err


@pytest.mark.parametrize("case, kwargs", [
    ("cuda 12.6", {"cuda": "12.6"}),
    ("cuda 13.0 (the PyPI default of torch 2.11)", {"cuda": "13.0"}),
    ("a CPU wheel", {"cuda": None}),
    ("cu128 without sm_120", {"flags": "sm_75 sm_80 sm_90"}),
    ("no arch flags", {"flags": ""}),
])
def test_a_torch_that_cannot_run_on_an_rtx_50_card_fails_the_build(tmp_path, case, kwargs):
    code, _, err = _run_check(_torch_checks(CUDA)[0], tmp_path, **kwargs)
    assert code != 0, f"{case}: the build would have shipped it"
    assert "AssertionError" in err and ("12.8" in err or "sm_120" in err)


def test_the_check_never_asks_the_gpu_api_for_the_arch_list():
    for dockerfile in (CUDA, ROCM):
        assert not any("get_arch_list" in args for keyword, args in _instructions(dockerfile) if keyword == "RUN")


def test_the_rocm_image_is_not_given_a_cuda_requirement():
    """It installs from the rocm6.2 index; a cu128 / sm_120 assertion there could never pass."""
    runs = " ".join(args for keyword, args in _instructions(ROCM) if keyword == "RUN")
    assert "cu128" not in runs and "sm_120" not in runs and "_cuda_getArchFlags" not in runs
    assert "rocm6.2" in runs


# --- the constraint file is what the RUN says it is ---------------------------------------------------------


@pytest.mark.parametrize("dockerfile", [CUDA, ROCM], ids=lambda p: p.name)
def test_the_requirements_install_really_passes_the_pinned_torch_as_a_constraint(dockerfile, tmp_path):
    """Run the RUN's shell with pip3 replaced by a recorder."""
    instructions = _instructions(dockerfile)
    command = instructions[_requirements_install(instructions)][1]
    version = _arg_default(dockerfile, "TORCH_VERSION")
    constraints = tmp_path / "torch-constraints.txt"
    (tmp_path / "requirements.txt").write_text((SERVICE / "requirements.txt").read_text(encoding="utf-8"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pip = bin_dir / "pip3"
    pip.write_text('#!/bin/sh\nfor a in "$@"; do echo "$a"; done > "$RECORD"\n')
    pip.chmod(0o755)

    done = subprocess.run(
        ["sh", "-c", command.replace("/tmp/torch-constraints.txt", str(constraints))],
        cwd=tmp_path, env={"PATH": f"{bin_dir}:/usr/bin:/bin", "TORCH_VERSION": version, "RECORD": str(tmp_path / "argv")},
        capture_output=True, text=True, timeout=30, stdin=subprocess.PIPE,
    )

    assert done.returncode == 0, done.stderr
    argv = (tmp_path / "argv").read_text().split("\n")
    assert argv[argv.index("-c") + 1] == str(constraints)
    assert "-r" in argv and argv[argv.index("-r") + 1] == "requirements.txt"
    assert constraints.read_text().split() == [f"torch=={version}"]
